"""Portable material comparisons and geometry-scoped alias evidence, no JIT."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
import marshal
from pathlib import Path
from types import SimpleNamespace

import pytest

import mechanics_runtime as data
from W2W import component_compiled_material as compiled
from tools.operators import catalog as aliases
from tools.operators import runtime


@pytest.fixture(scope="module")
def pinned_sources():
    sources = {}
    for name,pin in compiled.SOURCE_PINS.items():
        raw = (compiled.SOURCE_ROOT/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == pin
        sources[name] = raw
    return sources


def material_at(root, parameters, sources):
    root.mkdir()
    for name,raw in sources.items():
        (root/name).write_bytes(raw)
    path = root/"exp12_cu_material_point.py"
    # Use the real deployment transform (including eager annotations), but no
    # constitutive execution, native class modification or material JIT.
    cu,_ = compiled._execute(path,sources[path.name])
    card = cu.CuCard(**parameters)
    kernels = SimpleNamespace(update_batch=cu.trial_update_batch,
        tangent_batch=cu.trial_tangent_action_batch,elastic_stiffness=cu.isotropic_stiffness)
    binding = data.CorrectedMaterialBinding(material_kernels=kernels,cu_card=card,
        source_path=str(path),source_sha256=compiled.SOURCE_PINS[path.name],
        card_json=data._material_card_json(card),scope="synthetic_fixture",
        kernel_functions=tuple((f,f.__code__) for f in data._material_kernel_functions(kernels)),
        card_methods=tuple((f,f.__code__) for f in data._material_card_methods(card)))
    material = SimpleNamespace(binding=binding,cu_card=card,material_kernels=kernels)
    provenance = data.corrected_material_provenance(binding)
    provenance["sources"] = [dict(path=str(root/name),sha256=compiled.SOURCE_PINS[name]) for name in sources]
    return material,provenance


@pytest.fixture(params=("historical","public"))
def material_pair(tmp_path,pinned_sources,request):
    parameters = runtime.HISTORICAL_CARD if request.param == "historical" else runtime.PUBLIC_CARD
    old, historical = material_at(tmp_path/"original",parameters,pinned_sources)
    current, provenance = material_at(tmp_path/"relocated",parameters,pinned_sources)
    return old,historical,current,provenance


def test_same_root_keeps_legacy_fingerprints(material_pair,monkeypatch):
    material,provenance,_,_ = material_pair
    before = deepcopy(provenance)
    def forbidden(*args):
        raise AssertionError("same-root comparison must not rebase method code")
    monkeypatch.setattr(runtime,"_card_method_fingerprints",forbidden)
    runtime.validate_material_provenance(material,provenance,deepcopy(provenance))
    assert provenance == before


def test_relocated_real_cu_methods_compare_without_mutation(material_pair):
    _,historical,material,current = material_pair
    assert current["card_methods"] != historical["card_methods"]
    before = deepcopy((current,historical))
    codes = tuple(f.__code__ for f,_ in material.binding.card_methods)
    runtime.validate_material_provenance(material,current,historical)
    assert (current,historical) == before
    assert all(f.__code__ is code and code.co_filename == material.binding.source_path
        for (f,_),code in zip(material.binding.card_methods,codes))


def test_trusted_compilation_fingerprints_include_nested_method_code():
    raw = "class CuCard:\n    def elastic(self,x):\n        def inner(y):\n            return [x+v for v in y]\n        return inner\nraise AssertionError('comparison must never execute')\n"
    filename = str(Path("/original/cu.py"))
    original = runtime._card_method_fingerprints(raw,filename,("elastic",))
    assert original != runtime._card_method_fingerprints(raw,"/relocated/cu.py",("elastic",))
    assert original != runtime._card_method_fingerprints(raw.replace("x+v","x-v"),filename,("elastic",))


@pytest.mark.parametrize("side",("current","historical"))
@pytest.mark.parametrize("field",tuple(runtime.HISTORICAL_CARD))
def test_all_nineteen_card_fields_remain_required(material_pair,side,field):
    _,historical,material,current = material_pair
    target = current if side == "current" else historical
    value = target["cu_card"][field]
    target["cu_card"][field] = value+"_altered" if isinstance(value,str) else value+1.
    with pytest.raises(ValueError):
        runtime.validate_material_provenance(material,current,historical)


@pytest.mark.parametrize("side",("current","historical"))
@pytest.mark.parametrize("mutation",("clock","cu_source","acceleration_source","missing_source",
    "duplicate_source","source_filename","source_sha","method_hash","missing_method"))
def test_altered_source_clock_and_method_evidence_is_rejected(material_pair,side,mutation):
    _,historical,material,current = material_pair
    target = current if side == "current" else historical
    if mutation == "clock":
        target["clock_semantics"] = "interval_local_clock"
    elif mutation in ("cu_source","acceleration_source"):
        name = "exp12_cu_material_point.py" if mutation == "cu_source" else "exp12_all_gauss_exact_acceleration.py"
        next(item for item in target["sources"] if Path(item["path"]).name == name)["sha256"] = "0"*64
    elif mutation == "missing_source":
        target["sources"].pop()
    elif mutation == "duplicate_source":
        target["sources"][1] = dict(target["sources"][0])
    elif mutation == "source_filename":
        target["source_path"] = "/unbound/exp12_cu_material_point.py"
    elif mutation == "source_sha":
        target["source_sha256"] = "0"*64
    elif mutation == "method_hash":
        target["card_methods"]["elastic"] = "0"*64
    else:
        del target["card_methods"]["elastic"]
    with pytest.raises(ValueError):
        runtime.validate_material_provenance(material,current,historical)


def test_altered_live_method_is_rejected_even_with_rewritten_claim(material_pair,monkeypatch):
    _,historical,material,current = material_pair
    method = type(material.cu_card).elastic
    replacement = (lambda self,temperature: (1.,.1)).__code__.replace(co_filename=material.binding.source_path)
    monkeypatch.setattr(method,"__code__",replacement)
    current["card_methods"]["elastic"] = hashlib.sha256(marshal.dumps(replacement)).hexdigest()
    with pytest.raises(ValueError,match="implementation changed"):
        runtime.validate_material_provenance(material,current,historical)


def test_altered_relocated_source_bytes_are_rejected(material_pair):
    _,historical,material,current = material_pair
    Path(material.binding.source_path).write_bytes(b"# altered relocated fixture source\n")
    with pytest.raises(ValueError,match="source bytes differ"):
        runtime.validate_material_provenance(material,current,historical)


@pytest.mark.parametrize("mutation",("card","clock"))
def test_altered_live_card_or_clock_binding_is_rejected(material_pair,monkeypatch,mutation):
    _,historical,material,current = material_pair
    if mutation == "card":
        monkeypatch.setattr(material,"cu_card",replace(material.cu_card,e_room_mpa=1.))
    else:
        monkeypatch.setattr(material,"binding",replace(material.binding,clock_semantics="local_time"))
    with pytest.raises(ValueError,match="card, kernels or clock binding changed"):
        runtime.validate_material_provenance(material,current,historical)


@pytest.mark.parametrize("size,qualified",((10.,True),(11.,False)))
def test_alias_equivalence_is_scoped_to_original_geometry(monkeypatch,size,qualified):
    geometry = dict(width_mm=10.,length_mm=10.,total_thickness_um=50.,
        front_bi_thickness_um=1.5,back_bi_thickness_um=1.5,beol_thickness_um=10.)
    record = dict(geometry=geometry,pitch_um=4.,area_scalable=True,off_footprint_execution="native",
        directory="release/BI4_Prepared50",manifest_sha256="a"*64)
    common = dict(manifest_sha256=record["manifest_sha256"],baseline_sha256="b"*64)
    complete = json.dumps(common).encode()
    equivalence = json.dumps(dict(common,passed=True,all_operands_identical=True,
        all_retained_GP_state_signatures_identical=True)).encode()
    record.update(export_complete_sha256=hashlib.sha256(complete).hexdigest(),
        equivalence_sha256=hashlib.sha256(equivalence).hexdigest())
    monkeypatch.setattr(aliases,"catalog",lambda:dict(operators={"BI4_Prepared50":record}))
    monkeypatch.setenv("YAP_NORMALIZED_OPERATOR_ROOT","/explicit/normalized")
    def component_open(path,*args):
        assert str(path).startswith("/explicit/normalized/release/BI4_Prepared50/")
        return io.BytesIO(complete if path.name == "EXPORT_COMPLETE.json" else equivalence)
    monkeypatch.setattr(aliases,"application_module",lambda name:SimpleNamespace(
        _component_open=component_open,COMPONENT_PROJECT_ROOT=Path("/default/data")))
    requested = dict(geometry,width_mm=size,length_mm=size)
    library = object()
    def load(path,pin,actual,**options):
        assert actual == requested and pin == record["manifest_sha256"]
        assert options["area_scalable"] is True and options["off_footprint_execution"] == "native"
        return library,dict(geometry=actual)
    monkeypatch.setattr(aliases,"load_library",load)
    result,proof = aliases.load_prepared_coupon_library("coupon:BI4_Prepared50",requested)
    assert result is library and proof["operator_inputs_verified"] is True
    assert proof["migration_equivalence_qualified"] is qualified
    assert proof["migration_equivalence_sha256"] == record["equivalence_sha256"]


@pytest.mark.parametrize("legacy",("YAP_COUPON_OPERATOR_ROOT","YAP_SHARED_COUPON_OPERATOR_ROOT"))
def test_obsolete_root_override_warns_and_uses_explicit_new_catalog(monkeypatch,legacy):
    monkeypatch.setenv(legacy,"/obsolete/data")
    monkeypatch.setenv("YAP_COUPON_OPERATOR_CATALOG","/explicit/catalog.json")
    sentinel = object()
    def read(path):
        assert path == "/explicit/catalog.json"
        return sentinel
    monkeypatch.setattr(aliases,"_read",read)
    with pytest.warns(UserWarning,match=legacy+".*ignored"):
        assert aliases.catalog() is sentinel


@pytest.mark.parametrize("tag",("absent","matching","different",None,""))
def test_optional_catalog_python_abi_admission(tmp_path,monkeypatch,tag):
    document = dict(schema="yap.normalized_operator_catalog.v1",operators={"fixture":{}})
    current = aliases.sys.implementation.cache_tag
    if tag != "absent":
        document["python_cache_tag"] = (current if tag == "matching" else
            "cpython-311" if tag == "different" and current != "cpython-311" else
            "cpython-312" if tag == "different" else tag)
    path = tmp_path/"catalog.json"
    path.write_text(json.dumps(document))
    monkeypatch.setenv("YAP_COUPON_OPERATOR_CATALOG",str(path))
    monkeypatch.delenv("YAP_COUPON_OPERATOR_ROOT",raising=False)
    monkeypatch.delenv("YAP_SHARED_COUPON_OPERATOR_ROOT",raising=False)
    def forbidden(*args,**kwargs):
        raise AssertionError("catalog ABI admission must precede packet loading")
    monkeypatch.setattr(aliases,"load_library",forbidden)
    if tag in ("absent","matching"):
        assert aliases.catalog() == document
    else:
        with pytest.raises(ValueError,match="Python ABI mismatch: producer .*running .*matching Python"):
            aliases.load_prepared_coupon_library("coupon:fixture",{})


def test_catalog_abi_tag_does_not_globally_require_python312(tmp_path,monkeypatch):
    document = dict(schema="yap.normalized_operator_catalog.v1",operators={"fixture":{}},
        python_cache_tag="cpython-311")
    path = tmp_path/"custom.json"
    path.write_text(json.dumps(document))
    monkeypatch.setattr(aliases.sys.implementation,"cache_tag","cpython-311")
    assert aliases._read(str(path)) == document
