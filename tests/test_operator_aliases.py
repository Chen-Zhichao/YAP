"""Named data aliases require immutable migration evidence before construction."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools import coupon_operator_library as facade
from tools.operators import catalog, runtime


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def alias(tmp_path, monkeypatch):
    geometry = dict(width_mm=10., length_mm=10., total_thickness_um=50.,
                    front_bi_thickness_um=1.5, back_bi_thickness_um=1.5, beol_thickness_um=10.)
    record = dict(directory="example", geometry=geometry, pitch_um=7., area_scalable=False,
                  manifest_sha256="a"*64, load_forecast_bytes=1024)
    directory = tmp_path/"example"
    directory.mkdir()
    complete = dict(manifest_sha256=record["manifest_sha256"], baseline_sha256="b"*64)
    equivalence = dict(complete, passed=True, all_operands_identical=True,
                       all_retained_GP_state_signatures_identical=True)
    def publish(name, document, field):
        raw = json.dumps(document, sort_keys=True).encode("ascii")
        (directory/name).write_bytes(raw)
        record[field] = hashlib.sha256(raw).hexdigest()
    publish("EXPORT_COMPLETE.json", complete, "export_complete_sha256")
    publish("MIGRATION_EQUIVALENCE.json", equivalence, "equivalence_sha256")
    monkeypatch.setattr(catalog, "catalog", lambda: dict(operators=dict(example=record)))
    monkeypatch.setenv("YAP_NORMALIZED_OPERATOR_ROOT", str(tmp_path))
    calls = []
    result = object()
    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return result, dict(marker="common numerical loader")
    monkeypatch.setattr(catalog, "load_library", load)
    return dict(record=record, directory=directory, complete=complete, equivalence=equivalence,
                publish=publish, calls=calls, result=result)


@pytest.mark.parametrize("api", [catalog, facade], ids=["catalog", "facade"])
def test_verified_alias_uses_common_loader_and_resource_hooks(alias, api):
    calls = []
    result, proof = api.load_prepared_coupon_library("coupon:example", alias["record"]["geometry"],
        shared_check=lambda: calls.append("check"),
        shared_admit=lambda stage, amount: calls.append((stage, amount)))
    assert result is alias["result"]
    assert proof["operator_inputs_verified"] is True
    assert proof["migration_equivalence_qualified"] is True
    assert proof["migration_equivalence_sha256"] == alias["record"]["equivalence_sha256"]
    args, kwargs = alias["calls"][0]
    assert args[:2] == (alias["directory"], alias["record"]["manifest_sha256"])
    assert args[2] == alias["record"]["geometry"]
    assert kwargs["pitch_um"] == 7.
    assert calls[0] == calls[-1] == "check"
    assert calls[1] == ("NORMALIZED_ALIAS_LOAD", dict(additional_allocation_forecast_bytes=1024))


def test_facade_is_common_api_with_private_material_compatibility_only():
    assert facade.prepared_operator_request is catalog.prepared_operator_request
    assert facade.load_prepared_coupon_library is catalog.load_prepared_coupon_library
    assert facade.CouponOperatorLibrary is runtime.CouponOperatorLibrary
    assert facade._compiled_material is runtime._compiled_material
    assert "_compiled_material" not in facade.__all__


@pytest.mark.parametrize("name", ["EXPORT_COMPLETE.json", "MIGRATION_EQUIVALENCE.json"])
@pytest.mark.parametrize("fault", ["missing", "tampered"])
def test_incomplete_or_changed_alias_never_constructs(alias, name, fault):
    path = alias["directory"]/name
    if fault == "missing":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises((ValueError, FileNotFoundError)):
        catalog.load_prepared_coupon_library("coupon:example", alias["record"]["geometry"])
    assert not alias["calls"]


@pytest.mark.parametrize("field,value", [("passed", False), ("all_operands_identical", False),
    ("all_retained_GP_state_signatures_identical", False), ("manifest_sha256", "c"*64),
    ("baseline_sha256", "c"*64)])
def test_even_pinned_evidence_must_join_and_pass(alias, field, value):
    document = dict(alias["equivalence"], **{field: value})
    alias["publish"]("MIGRATION_EQUIVALENCE.json", document, "equivalence_sha256")
    with pytest.raises(ValueError, match="equivalence"):
        catalog.load_prepared_coupon_library("coupon:example", alias["record"]["geometry"])
    assert not alias["calls"]


@pytest.mark.parametrize("directory", ["../other", "/absolute", ".snapshot/data", "#snapshot/data"])
def test_alias_directory_cannot_escape_or_enter_snapshots(alias, directory):
    alias["record"]["directory"] = directory
    with pytest.raises(ValueError):
        catalog.load_prepared_coupon_library("coupon:example", alias["record"]["geometry"])
    assert not alias["calls"]


def test_alias_admission_refusal_prevents_numeric_load(alias):
    with pytest.raises(ValueError, match="admission denied"):
        catalog.load_prepared_coupon_library("coupon:example", alias["record"]["geometry"],
                                             shared_admit=lambda *args: False)
    assert not alias["calls"]


def test_published_catalog_keeps_all_95_named_aliases():
    index = catalog._read(str(ROOT/"prepared_coupon_catalog.json"))
    grid = {f"BI{pitch}_Prepared{total}" for pitch in range(2, 11) for total in range(20, 101, 10)}
    records = index["operators"]
    assert len(grid) == 81
    assert grid <= records.keys()
    assert len(records) == 95
    for record in records.values():
        for field in ("manifest_sha256", "export_complete_sha256", "equivalence_sha256"):
            assert len(record[field]) == 64 and set(record[field]) <= set("0123456789abcdef")


def test_all_95_published_aliases_reach_common_decoder_through_facade(monkeypatch):
    index = catalog._read(str(ROOT/"prepared_coupon_catalog.json"))
    monkeypatch.setattr(catalog, "catalog", lambda: index)
    calls = []
    sentinel = object()
    def decoder(directory, pin, geometry, **options):
        calls.append((directory, pin, geometry, options))
        return sentinel, dict(decoder="common numerical loader")
    monkeypatch.setattr(catalog, "load_library", decoder)
    for key, record in index["operators"].items():
        result, proof = facade.load_prepared_coupon_library("coupon:"+key, record["geometry"])
        assert result is sentinel
        assert proof["coupon_key"] == key
        assert proof["migration_equivalence_qualified"] is True
        assert proof["migration_equivalence_sha256"] == record["equivalence_sha256"]
        directory, pin, geometry, options = calls[-1]
        assert directory.parts[-len(Path(record["directory"]).parts):] == Path(record["directory"]).parts
        assert pin == record["manifest_sha256"]
        assert geometry == record["geometry"]
        assert options["pitch_um"] == record["pitch_um"]
    assert len(calls) == 95


@pytest.mark.parametrize("pitch", range(2, 11))
@pytest.mark.parametrize("total", range(20, 101, 10))
def test_all_81_grid_alias_requests_preserve_geometry(pitch, total, monkeypatch):
    index = catalog._read(str(ROOT/"prepared_coupon_catalog.json"))
    monkeypatch.setattr(catalog, "catalog", lambda: index)
    key = f"BI{pitch}_Prepared{total}"
    geometry = index["operators"][key]["geometry"]
    request = facade.prepared_operator_request("coupon:"+key, geometry, layout=(pitch, pitch, pitch/2.))
    assert request == catalog.prepared_operator_request("coupon:"+key, geometry, layout=(pitch, pitch, pitch/2.))
    assert request["coupon_key"] == key
    assert request["total_thickness_um"] == total
    assert request["bonding_pitch_um"] == pitch
    assert request["retained_si_thickness_um"] == total-13
    assert not request["operator_inputs_verified"]
    assert not request["assembly_accuracy_validated"]
    assert not request["W2W_thinning_available"]


@pytest.mark.parametrize("namespace", ["tools", "YAP.tools"])
def test_common_runtime_imports_without_retired_geometry_loaders(tmp_path, namespace):
    code = '''
import importlib, importlib.abc, sys
prefix = sys.argv[1]
retired = ("bi", "native_coupon", "sampled_coupon", "shared_coupon", "source50_coupon_acceleration",
           "coupon_reference_tangent", "coupon_streamed_acceleration")
class BlockRetired(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.endswith(".operators.migrate"):
            raise AssertionError("maintainer migration import: "+fullname)
        for package in ("tools.", "YAP.tools."):
            if fullname.startswith(package) and fullname[len(package):].split(".")[0].startswith(retired):
                raise AssertionError("retired loader import: "+fullname)
sys.meta_path.insert(0, BlockRetired())
for name in ("runtime", "kinematics", "reference", "acceleration", "streamed", "catalog"):
    importlib.import_module(prefix+".operators."+name)
facade = importlib.import_module(prefix+".coupon_operator_library")
catalog = importlib.import_module(prefix+".operators.catalog")
assert facade.load_prepared_coupon_library is catalog.load_prepared_coupon_library
assert facade.prepared_operator_request is catalog.prepared_operator_request
print("COMMON_OPERATORS_ONLY")
'''
    environment = dict(os.environ, PYTHONPATH=str(ROOT if namespace == "tools" else ROOT.parent),
                       PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-B", "-c", code, namespace], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr
    assert "COMMON_OPERATORS_ONLY" in result.stdout
