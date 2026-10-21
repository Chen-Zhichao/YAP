"""Component evidence admission stays separate from legacy assembly packages."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

import mechanics_runtime as runtime


@pytest.fixture
def component_arrays():
    import numpy as np
    from scipy import sparse
    kff = sparse.csr_matrix([[3.,.5],[.5,4.]])
    maps = np.zeros((12,136))
    maps[:6,:6] = np.eye(6)
    maps[6:,:6] = np.eye(6)*.8
    maps[:,134:] = np.random.default_rng(26).normal(size=(12,2))*.1
    return dict(kaa=np.eye(134)*10,c=np.ones((2,134))*.001,
        kff_data=kff.data,kff_indices=kff.indices,kff_indptr=kff.indptr,kff_shape=np.array(kff.shape),
        fm=np.ones(134)*1e-7,ff=np.ones(2)*1e-8,
        B_macro=maps[:,:134],B_free=maps[:,134:],weights=np.array([1e-9,2e-9]),
        reference_temperature_c=np.array(25.),reference_cu_stiffness=None,
        reference_cu_alpha=np.r_[np.ones(3)*17.6e-6,np.zeros(3)])


@pytest.fixture
def corrected_material(monkeypatch, component_arrays):
    import importlib.util
    from types import SimpleNamespace
    source = runtime.ROOT.parent / 'operator_repair_20261015/source/exp12_cu_material_point.py'
    if not source.is_file():
        pytest.skip('optional corrected research material is not installed')
    spec = importlib.util.spec_from_file_location('_component_bridge_test_corrected_cu',source)
    material = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,spec.name,material)
    spec.loader.exec_module(material)
    card = material.CuCard(e_room_mpa=131000.,e_anneal_mpa=131000.,anneal_c=350.,
        alpha_room_per_c=17.6e-6,alpha_anneal_per_c=17.6e-6,
        creep_law='modified_time_hardening',creep_prefactor_mpa_minus_n_per_s=1e-8,
        creep_time_exponent=-.5,creep_exponent=3.)
    kernels = SimpleNamespace(update_batch=material.trial_update_batch,
        tangent_batch=material.trial_tangent_action_batch,elastic_stiffness=material.isotropic_stiffness)
    component_arrays['reference_cu_stiffness'] = material.isotropic_stiffness(*card.elastic(25.))
    return material,card,kernels


def test_component_bridge_preserves_algebra_and_temperature_scope(component_arrays,corrected_material):
    import numpy as np
    _,card,kernels = corrected_material
    arguments = runtime.component_native_layer_arguments(component_arrays,cu_card=card,
        material_kernels=kernels,macro_coordinate_scale=np.ones(134))
    k = arguments['reference_stiffness'](350.)
    np.testing.assert_array_equal(k[:134,134:],component_arrays['c'].T)
    np.testing.assert_array_equal(arguments['strain_maps'].reshape(12,136)[:,:134],component_arrays['B_macro'])
    np.testing.assert_array_equal(arguments['reference_thermal_load'](25.),np.zeros(136))
    np.testing.assert_allclose(arguments['reference_thermal_load'](350.),
        325*np.r_[component_arrays['fm'],component_arrays['ff']],rtol=0,atol=0)
    assert arguments['weights_mm3'] is component_arrays['weights']
    assert not k.flags.writeable
    for bad in (24.,351.,float('nan')):
        with pytest.raises(ValueError): arguments['reference_stiffness'](bad)


@pytest.mark.parametrize('fault',('missing_kernels','scale','card','nonfinite','dimensions'))
def test_component_bridge_refuses_incompatible_contract(component_arrays,corrected_material,fault):
    import numpy as np
    _,card,kernels = corrected_material
    scale = np.ones(134)
    if fault == 'missing_kernels': kernels = None
    elif fault == 'scale': scale[0] = 0
    elif fault == 'card': component_arrays['reference_cu_alpha'][0] *= 2
    elif fault == 'nonfinite': component_arrays['c'][0,0] = float('nan')
    else: component_arrays['kaa'] = np.eye(133)
    with pytest.raises(ValueError):
        runtime.component_native_layer_arguments(component_arrays,cu_card=card,
            material_kernels=kernels,macro_coordinate_scale=scale)


def test_native_layer_with_corrected_kernels_matches_component_force_and_tangent(
        monkeypatch,component_arrays,corrected_material):
    import numpy as np
    _,card,kernels = corrected_material
    native = runtime.ROOT.parent/'Ansys_Verify/12_stack_model_validation'
    monkeypatch.syspath_prepend(str(native))
    import exp12_all_gauss_reduced_fe2 as fe
    arguments = runtime.component_native_layer_arguments(component_arrays,cu_card=card,
        material_kernels=kernels,macro_coordinate_scale=np.ones(134))
    layer = fe.AllGaussReducedFE2Layer(**arguments)
    state = layer.initial_state()
    q, z = np.zeros(134),np.array([.001,-.001])
    q[:3] = [.006,-.002,0]
    direction = np.random.default_rng(31).normal(size=136)*1e-5
    before = state.cu_state.plastic_strain.copy()
    clock = dict(creep_active=True,plastic_active=True,elapsed_time_s=1000.)
    result = layer.evaluate(q,z,350.,600.,state,**clock)
    maps,weights = layer.strain_maps.reshape(12,136),layer.weights_mm3
    mechanical = np.r_[q,z]
    strain = (maps@mechanical).reshape(2,6)
    trial = kernels.update_batch(strain,350.,600.,state.cu_state,card=card,**clock)
    alpha = component_arrays['reference_cu_alpha']
    cref = component_arrays['reference_cu_stiffness']
    reference = (strain-325*alpha)@cref.T
    expected = (layer.reference_stiffness(350.)@mechanical-layer.reference_thermal_load(350.)
        + maps.T@(weights[:,None]*(trial.stress_mpa-reference)).ravel())
    np.testing.assert_allclose(result.response,expected,rtol=1e-12,atol=1e-12)
    delta = (maps@direction).reshape(2,6)
    stress_delta = kernels.tangent_batch(strain,delta,350.,600.,state.cu_state,card=card,**clock)
    expected_tangent = (layer.reference_stiffness(350.)@direction
        + maps.T@(weights[:,None]*(stress_delta-delta@cref.T)).ravel())
    actual = layer.tangent_action(q,z,direction,350.,600.,state,**clock)
    np.testing.assert_allclose(actual,expected_tangent,rtol=1e-12,atol=1e-12)
    many = layer.evaluate_many([q,q],[z,z],350.,600.,[state,state],**clock)
    for value in many:
        np.testing.assert_array_equal(value.cu_trial.stress_mpa,result.cu_trial.stress_mpa)
    np.testing.assert_array_equal(state.cu_state.plastic_strain,before)
    # The old endpoint convention must not silently replace the corrected update.
    legacy_args = {k:v for k,v in arguments.items() if k != 'material_kernels'}
    legacy = fe.AllGaussReducedFE2Layer(**legacy_args).evaluate(q,z,350.,600.,state,**clock)
    assert np.max(abs(legacy.cu_trial.stress_mpa-result.cu_trial.stress_mpa)) > 1e-3


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def catalog(monkeypatch):
    # Always keep generated test inputs in the canonical project, never /tmp.
    with tempfile.TemporaryDirectory(prefix=".component_catalog_test_", dir=runtime.ROOT) as directory:
        root = Path(directory)
        monkeypatch.setattr(runtime, "COMPONENT_PROJECT_ROOT", root)
        monkeypatch.delenv("YAP_COMPONENT_OPERATOR_CATALOG", raising=False)
        target = root / runtime.COMPONENT_CATALOG_RELATIVE
        target.parent.mkdir(parents=True)

        def evidence(name, document):
            path = target.parent / name
            payload = json.dumps(document).encode()
            path.write_bytes(payload)
            return {"path": name, "sha256": sha(payload)}

        components = {}
        for role in ("bi", "beol", "si"):
            payload = ("fixture_" + role).encode()
            (target.parent / (role + ".npz")).write_bytes(payload)
            components[role] = dict(path=role + ".npz", sha256=sha(payload), bytes=len(payload), micro_rank=2)
        entry = dict(
            id="repaired_h48p5", aliases=["h48p5"],
            geometry=dict(bi_pitch_um=2., diameter_pitch_ratio=.5, bi_thickness_um=1.5,
                          beol_thickness_um=10., si_tsv_thickness_um=48.5,
                          total_thickness_um=60., bi_layer_count=1, tsv_pitch_um=6., cell_side_um=6.),
            components=components,
            validation_scope=runtime.COMPONENT_VALIDATION_SCOPE,
            globalbowvalidated_original1081=True, assembly_execution_supported=False,
        )
        second = copy.deepcopy(entry)
        second.update(id="repaired_h87", aliases=["h87"])
        second["geometry"].update(si_tsv_thickness_um=87., total_thickness_um=98.5)
        document = {"schema": "yap.component_operator_catalog.v1", "entries": [entry, second]}
        source, source_pin = str(root / "build.py"), "1" * 64
        acceptance = dict(schema="yap.global_bow_strict_or_admission.v1",
                          state="BOTH_ORIGINAL1081_GLOBAL_BOW_ACCURACY_PASS", global_accuracy_passed=True,
                          scope=dict(states=1081, cycles=3, BI_pitch_um=2., diameter_pitch_ratio=.5,
                                     BI_um=1.5, BEOL_um=10., cell_xz_um=[6., 6.], projection_xz_mm=[10, 10],
                                     warpage_scope="macro_homogenized_projection_not_finite_die_validation",
                                     no_arbitrary_geometry_or_assembly_order_accuracy_claim=True),
                          common_construction=dict(catalog=str(target.parent), source=source, source_sha256=source_pin),
                          cases={})
        for index, item in enumerate(document["entries"]):
            case = item["aliases"][0]
            references = {role: dict(path=str(target.parent / (case + role + "_reference.npz")),
                                    sha256="2" * 64, bytes=5) for role in components}
            reference = evidence("reference.json" if index == 0 else "h87_reference.json",
                                 dict(schema="yap.native_component_reference.v1", case=case,
                                      state="CLOSED_NATIVE_COMPONENT_REFERENCE_NOT_REDUCED_OR_ADMITTED",
                                      artifacts=references))
            recipe = evidence("recipe.json" if index == 0 else "h87_recipe.json",
                              dict(schema="yap.component_cu_riesz_operator.v1", case=case,
                                   state="CLOSED_COMMON_COMPONENT_CU_RIESZ_OPERATORS_UNSCORED",
                                   operator_admission=False, global_accuracy_passed=None,
                                   reference=reference, source_sha256={source: source_pin},
                                   components={role: dict(artifact=copy.deepcopy(binding), micro_rank=2,
                                                         reference=references[role])
                                               for role, binding in item["components"].items()}))
            item["coordinate_identity"] = dict(recipe=recipe, reference=reference)
            accuracy = dict(global_accuracy_passed=True, matched_rows=1081, failed_sample_probe_count=0)
            acceptance["cases"][case] = dict(Si_TSV_um=item["geometry"]["si_tsv_thickness_um"],
                                            total_um=item["geometry"]["total_thickness_um"],
                                            operator_admission="ACCURACY_ONLY_FOR_THE_EXPLICIT_VALIDATED_SCOPE",
                                            native_fullspace_accuracy=accuracy, reduced_operator_accuracy=accuracy)
        acceptance_binding = evidence("acceptance.json", acceptance)
        monkeypatch.setattr(runtime, "COMPONENT_FORMAL_ACCEPTANCE_SHA256", acceptance_binding["sha256"])
        monkeypatch.setattr(runtime, "COMPONENT_RECIPE_SHA256", {})
        for item in document["entries"]:
            item["acceptance"] = copy.deepcopy(acceptance_binding)
            runtime.COMPONENT_RECIPE_SHA256[item["aliases"][0]] = item["coordinate_identity"]["recipe"]["sha256"]

        def save(*, rebind=False):
            if rebind:
                for item in document["entries"]:
                    binding = item["coordinate_identity"]["recipe"]
                    recipe_path = target.parent / binding["path"]
                    recipe_doc = json.loads(recipe_path.read_text())
                    for role in components:
                        recipe_doc["components"][role]["artifact"] = copy.deepcopy(item["components"][role])
                    new = evidence(recipe_path.name, recipe_doc)
                    binding.update(new)
                    runtime.COMPONENT_RECIPE_SHA256[item["aliases"][0]] = new["sha256"]
            target.write_text(json.dumps(document), encoding="ascii")

        save()
        yield root, target, document, save


def test_default_environment_and_explicit_paths(catalog, monkeypatch):
    root, target, _, _ = catalog
    assert runtime.component_operator_catalog_path() == target
    monkeypatch.setenv("YAP_COMPONENT_OPERATOR_CATALOG", str(root / "another.json"))
    assert runtime.component_operator_catalog_path() == root / "another.json"
    assert runtime.component_operator_catalog_path(target) == target
    for invalid in ("", "relative.json", "/tmp/catalog.json"):
        monkeypatch.setenv("YAP_COMPONENT_OPERATOR_CATALOG", invalid)
        with pytest.raises(ValueError):
            runtime.component_operator_catalog_path()


def test_list_binds_evidence_but_does_not_read_npz_payloads(catalog, monkeypatch):
    _, target, document, _ = catalog
    original = runtime._component_hash_stream
    monkeypatch.setattr(runtime, "_component_hash_stream", lambda *_args: pytest.fail("payload read"))
    loaded = runtime.load_component_operator_catalog(target)
    assert loaded["verification"] == "JSON_BINDINGS_AND_FILE_METADATA_VERIFIED"
    assert loaded["entries"][0]["globalbowvalidated_original1081"] is True
    assert loaded["entries"][0]["assembly_execution_supported"] is False
    assert document["entries"][0]["components"]["bi"]["path"] == "bi.npz"
    monkeypatch.setattr(runtime, "_component_hash_stream", original)
    checked = runtime.check_component_operator_catalog(target)
    assert checked["verification"] == "ALL_COMPONENT_SHA256_VERIFIED"
    assert checked["unique_artifacts_checked"] == 3


def test_select_exact_geometry_and_alias_without_native_lookup(catalog, monkeypatch):
    _, target, document, _ = catalog
    monkeypatch.setattr(runtime, "select_native_root", lambda *_args: pytest.fail("legacy substitution"))
    entry = document["entries"][0]
    resolved = runtime.resolve_component_operator(entry["geometry"], catalog_path=target, entry_id="h48p5")
    assert resolved["id"] == entry["id"]
    for geometry, alias in (({**entry["geometry"], "bi_pitch_um": 2.00000001}, None),
                            (entry["geometry"], "h87"),
                            ({**entry["geometry"], "bi_layer_count": 2, "total_thickness_um": 61.5}, None)):
        with pytest.raises(ValueError, match="no unique exact"):
            runtime.resolve_component_operator(geometry, catalog_path=target, entry_id=alias)
    with pytest.raises(RuntimeError, match="does not support assembly"):
        runtime.resolve_component_operator(entry["geometry"], catalog_path=target, for_assembly=True)


@pytest.mark.parametrize("field,value", [("bi_layer_count", True), ("bi_layer_count", 1.),
                                         ("bi_pitch_um", float("inf")), ("bi_pitch_um", 0),
                                         ("bi_pitch_um", 10**1000),
                                         ("diameter_pitch_ratio", 1.1), ("total_thickness_um", 59.)])
def test_invalid_geometry(catalog, field, value):
    _, target, document, _ = catalog
    geometry = {**document["entries"][0]["geometry"], field: value}
    with pytest.raises(ValueError):
        runtime.resolve_component_operator(geometry, catalog_path=target)


def test_missing_bi_count_is_not_inferred(catalog):
    _, target, document, _ = catalog
    geometry = dict(document["entries"][0]["geometry"])
    geometry.pop("bi_layer_count")
    with pytest.raises(ValueError, match="including BI count"):
        runtime.resolve_component_operator(geometry, catalog_path=target)


@pytest.mark.parametrize("field,value", [("assembly_execution_supported", True),
                                         ("globalbowvalidated_original1081", False),
                                         ("validation_scope", "CANDIDATE_NOT_VALIDATED")])
def test_scope_cannot_be_overstated_or_demoted(catalog, field, value):
    _, target, document, save = catalog
    document["entries"][0][field] = value
    save()
    with pytest.raises(ValueError, match="original1081 global bow only"):
        runtime.load_component_operator_catalog(target)


@pytest.mark.parametrize("binding", ["acceptance", "recipe", "reference"])
def test_evidence_tampering(catalog, binding):
    _, target, _, _ = catalog
    (target.parent / (binding + ".json")).write_text('{"changed":true}', encoding="ascii")
    with pytest.raises(ValueError, match="JSON SHA256 mismatch"):
        runtime.load_component_operator_catalog(target)


@pytest.mark.parametrize("change", ["schema", "global_pass", "clock", "cell", "case_geometry", "rom_rows"])
def test_formal_acceptance_semantics_not_just_json_hash(catalog, monkeypatch, change):
    _, target, document, save = catalog
    path = target.parent / "acceptance.json"
    acceptance = json.loads(path.read_text())
    if change == "schema":
        acceptance["schema"] = "arbitrary.claim.v1"
    elif change == "global_pass":
        acceptance["global_accuracy_passed"] = False
    elif change == "clock":
        acceptance["scope"]["states"] = 61
    elif change == "cell":
        acceptance["scope"]["cell_xz_um"] = [8, 8]
    elif change == "case_geometry":
        acceptance["cases"]["h48p5"]["Si_TSV_um"] = 87
    else:
        acceptance["cases"]["h48p5"]["reduced_operator_accuracy"]["matched_rows"] = 61
    payload = json.dumps(acceptance).encode()
    path.write_bytes(payload)
    pin = sha(payload)
    for entry in document["entries"]:
        entry["acceptance"]["sha256"] = pin
    # Isolate semantic checks from the independently pinned trust root.
    monkeypatch.setattr(runtime, "COMPONENT_FORMAL_ACCEPTANCE_SHA256", pin)
    save()
    with pytest.raises(ValueError, match="formal|scope|accuracy"):
        runtime.load_component_operator_catalog(target)


def test_rehashed_arbitrary_claim_cannot_replace_trusted_acceptance(catalog):
    _, target, document, save = catalog
    path = target.parent / "acceptance.json"
    replacement = path.read_bytes() + b" "
    path.write_bytes(replacement)
    for entry in document["entries"]:
        entry["acceptance"]["sha256"] = sha(replacement)
    save()
    with pytest.raises(ValueError, match="pinned formal acceptance"):
        runtime.load_component_operator_catalog(target)


def test_catalog_artifact_claim_must_match_recipe(catalog):
    _, target, document, save = catalog
    document["entries"][0]["components"]["bi"]["sha256"] = "3" * 64
    save()
    with pytest.raises(ValueError, match="differs from the accepted case recipe"):
        runtime.load_component_operator_catalog(target)


@pytest.mark.parametrize("kind", ["recipe", "reference"])
def test_case_report_cannot_be_relabelled(catalog, monkeypatch, kind):
    _, target, document, save = catalog
    binding = document["entries"][0]["coordinate_identity"][kind]
    path = target.parent / binding["path"]
    report = json.loads(path.read_text())
    report["case"] = "h87"
    payload = json.dumps(report).encode()
    path.write_bytes(payload)
    binding["sha256"] = sha(payload)
    if kind == "recipe":
        monkeypatch.setitem(runtime.COMPONENT_RECIPE_SHA256, "h48p5", sha(payload))
    save()
    with pytest.raises(ValueError, match="case or schema mismatch"):
        runtime.load_component_operator_catalog(target)


def test_missing_coordinate_identity_and_duplicate_geometry(catalog):
    _, target, document, save = catalog
    identity = document["entries"][0].pop("coordinate_identity")
    save()
    with pytest.raises(ValueError, match="entry fields"):
        runtime.load_component_operator_catalog(target)
    document["entries"][0]["coordinate_identity"] = identity
    document["entries"][1]["geometry"] = document["entries"][0]["geometry"]
    save()
    with pytest.raises(ValueError, match="duplicate component geometry"):
        runtime.load_component_operator_catalog(target)


@pytest.mark.parametrize("value", ["../outside.npz", "#snapshot/bi.npz", ".snapshot/bi.npz",
                                   "Ansys_Verify/bi.npz", "storage_migration_20261002/bi.npz",
                                   ".migration_hold_test/bi.npz", "/tmp/bi.npz"])
def test_protected_or_escaping_paths_rejected_before_open(catalog, value):
    _, target, document, save = catalog
    document["entries"][0]["components"]["bi"]["path"] = value
    save()
    with pytest.raises(ValueError):
        runtime.load_component_operator_catalog(target)


def test_symlink_leaf_and_parent_rejected(catalog):
    _, target, document, save = catalog
    alias = target.parent / "alias.npz"
    alias.symlink_to("bi.npz")
    document["entries"][0]["components"]["bi"]["path"] = alias.name
    save()
    with pytest.raises(OSError):
        runtime.load_component_operator_catalog(target)
    directory = target.parent / "linked"
    directory.symlink_to(target.parent, target_is_directory=True)
    with pytest.raises(OSError):
        runtime.load_component_operator_catalog(directory / target.name)


def test_payload_size_and_hash_tampering(catalog):
    _, target, _, _ = catalog
    artifact = target.parent / "bi.npz"
    payload = artifact.read_bytes()
    artifact.write_bytes(b"x" * len(payload))
    runtime.load_component_operator_catalog(target)
    with pytest.raises(ValueError, match="artifact SHA256 mismatch"):
        runtime.check_component_operator_catalog(target)
    artifact.write_bytes(payload + b"x")
    with pytest.raises(ValueError, match="byte count mismatch"):
        runtime.load_component_operator_catalog(target)


def test_oversized_and_duplicate_key_json(catalog, monkeypatch):
    _, target, _, _ = catalog
    target.write_text('{"schema":"one","schema":"two"}', encoding="ascii")
    with pytest.raises(ValueError, match="duplicate component JSON key"):
        runtime.load_component_operator_catalog(target)
    monkeypatch.setattr(runtime, "_COMPONENT_JSON_LIMIT", 8)
    with pytest.raises(ValueError, match="bounded nonempty regular file"):
        runtime.load_component_operator_catalog(target)


def test_payload_aggregate_budget(catalog, monkeypatch):
    _, target, _, _ = catalog
    monkeypatch.setattr(runtime, "_COMPONENT_CHECK_LIMIT", 1)
    with pytest.raises(ValueError, match="8 GiB"):
        runtime.check_component_operator_catalog(target)


def test_hash_reads_are_paced_before_io_and_deadline_is_fail_closed(monkeypatch):
    import io
    import time

    payload = b"x" * (2 * 1024**2 + 3)
    clock, reads = [0.], []

    class ObservedStream(io.BytesIO):
        def read(self, size=-1):
            reads.append((clock[0], size))
            return super().read(size)

    monkeypatch.setattr(runtime, "_component_identity", lambda _stream: (0, 0, len(payload), 0, 0))
    with monkeypatch.context() as timer:
        timer.setattr(time, "monotonic", lambda: clock[0])
        timer.setattr(time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
        runtime._component_hash_stream(ObservedStream(payload), sha(payload), 120.)
        assert clock[0] >= len(payload) / (32 * 1024**2)
        assert reads[0][0] >= 1. / 32
        assert max(size for _, size in reads) <= 1024**2
        with pytest.raises(ValueError, match="time budget exceeded"):
            runtime._component_hash_stream(ObservedStream(payload), sha(payload), 0.)


def test_conflicting_shared_artifact_and_duplicate_alias_refused(catalog):
    _, target, document, save = catalog
    document["entries"][1]["components"]["bi"]["micro_rank"] = 3
    save()
    with pytest.raises(ValueError, match="conflicting bindings"):
        runtime.load_component_operator_catalog(target)
    document["entries"][1]["components"]["bi"]["micro_rank"] = 2
    document["entries"][1]["aliases"] = ["h48p5"]
    save()
    with pytest.raises(ValueError, match="duplicate component ID or alias"):
        runtime.load_component_operator_catalog(target)


def numeric_artifact(catalog, object_array=False):
    np = pytest.importorskip("numpy")
    _, target, document, save = catalog
    artifact = target.parent / "bi.npz"
    np.savez(artifact, test_array=np.array([object()] if object_array else [1., 2.]))
    binding = document["entries"][0]["components"]["bi"]
    binding.update(bytes=artifact.stat().st_size, sha256=sha(artifact.read_bytes()))
    document["entries"][1]["components"]["bi"] = copy.deepcopy(binding)
    save(rebind=True)
    return document["entries"][0]["geometry"]


def test_lazy_numeric_arrays_are_read_only(catalog):
    _, target, _, _ = catalog
    geometry = numeric_artifact(catalog)
    arrays = runtime.load_component_operator_arrays(geometry, "bi", catalog_path=target)
    assert arrays["test_array"].tolist() == [1., 2.]
    assert not arrays["test_array"].flags.writeable
    with pytest.raises(ValueError, match="oversized component NPZ"):
        runtime.load_component_operator_arrays(geometry, "bi", catalog_path=target, max_uncompressed_bytes=1)


def test_object_arrays_refused(catalog):
    _, target, _, _ = catalog
    geometry = numeric_artifact(catalog, object_array=True)
    with pytest.raises(ValueError, match="object-valued"):
        runtime.load_component_operator_arrays(geometry, "bi", catalog_path=target)


def test_forged_huge_array_shape_refused_before_allocation(catalog):
    import io
    import zipfile

    np = pytest.importorskip("numpy")
    _, target, document, save = catalog
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, dict(descr="<f8", fortran_order=False, shape=(2**50,)))
    artifact = target.parent / "bi.npz"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("huge.npy", header.getvalue())
    binding = document["entries"][0]["components"]["bi"]
    binding.update(bytes=artifact.stat().st_size, sha256=sha(artifact.read_bytes()))
    document["entries"][1]["components"]["bi"] = copy.deepcopy(binding)
    save(rebind=True)
    with pytest.raises(ValueError, match="invalid byte dimensions"):
        runtime.load_component_operator_arrays(document["entries"][0]["geometry"], "bi", catalog_path=target)


def test_existing_resource_map_unchanged():
    mapping = runtime.load_resource_map(runtime.ROOT / ".runtime_resources.json")
    assert "prepared_p2_native" not in mapping["resources"]
    assert "component_operators" not in mapping["resources"]
    assert mapping["resources"]["compact_p2"] == "results/date_ofat_layer_pod_20260913_v1/bi2_p24_b16"


def test_legacy_package_counts_unchanged_without_importing_native_runtime():
    import ast

    source = runtime.ROOT / "history_mechanics_backend.py"
    assert source.stat().st_size < 128 * 1024
    tree = ast.parse(source.read_text(encoding="utf-8"))
    registries = {node.targets[0].id: node.value for node in tree.body
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    assert len(registries["PREVIEW_PACKAGES"].keys) == 5
    assert len(registries["COMPACT_PACKAGES"].keys) == 4
    prepared = runtime.ROOT / "prepared_operator_catalog.json"
    assert prepared.stat().st_size < 65536
    document = json.loads(prepared.read_text(encoding="utf-8"))
    assert document["schema"] == "yap.prepared_p2_operator_catalog.v1"
    assert set(document["packages"]) == {"30", "80"}


def test_cli_from_other_cwd_is_stdlib_only_and_missing_catalog_fails_closed(catalog):
    root, target, _, _ = catalog
    # The subprocess has the real canonical root; the fixture lives beneath it.
    code = (
        "import importlib.util, sys; "
        f"spec=importlib.util.spec_from_file_location('component_cli', {str(runtime.ROOT / 'mechanics_runtime.py')!r}); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        f"module.COMPONENT_FORMAL_ACCEPTANCE_SHA256={runtime.COMPONENT_FORMAL_ACCEPTANCE_SHA256!r}; "
        f"module.COMPONENT_RECIPE_SHA256={runtime.COMPONENT_RECIPE_SHA256!r}; "
        "result=module._component_catalog_cli(); "
        "assert 'numpy' not in sys.modules; "
        "assert 'NATIVE_ROOT' not in vars(module); sys.exit(result)"
    )
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "YAP_COMPONENT_OPERATOR_CATALOG": str(target)}
    for command in ("list", "check"):
        result = subprocess.run([sys.executable, "-B", "-c", code, command], cwd=root,
                                env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["assembly_execution_supported"] is False
    result = subprocess.run([sys.executable, "-B", "-c", code, "list", "--catalog", str(root / "missing.json")],
                            cwd=root, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    assert "unavailable or invalid" in result.stderr
