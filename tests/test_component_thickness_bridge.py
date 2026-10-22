"""Thickness fallback/physical registration, not new assembly validation."""

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import component_operators as components
from W2W import component_assembly_bridge as bridge
from test_component_operators import entry as exact_entry, geometry, gp_arrays


ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT.parent / "distribution/operators/thickness_family"
HISTORY_FIELDS = {"plastic_strain", "creep_strain", "equivalent_plastic_strain",
                  "equivalent_creep_strain", "cu_state"}


@pytest.fixture(scope="module")
def catalog_entries():
    path = components.DEFAULT_CATALOG
    before = path.read_bytes()
    entries = json.loads(before)["entries"]
    assert len(entries) == 13
    yield entries
    assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(before).digest()


@pytest.mark.parametrize("index", range(13))
def test_all_13_exact_catalog_entries_win_before_family(index, catalog_entries, monkeypatch):
    monkeypatch.setattr(components, "load_catalog", lambda *_a, **_k: {"entries": catalog_entries})
    def forbidden(*_args, **_kwargs):
        pytest.fail("an exact catalog match must not open or resolve a thickness family")
    monkeypatch.setattr(components, "_parameterized_family", forbidden)
    original = copy.deepcopy(catalog_entries[index])
    assert components.resolve_entry("component:" + original["id"]) == original
    assert catalog_entries[index] == original


def test_default_family_is_canonical_v2():
    assert Path(components.DEFAULT_THICKNESS_FAMILY) == FAMILY


def test_absent_family_index_has_clear_unknown_component_error(tmp_path, catalog_entries, monkeypatch):
    monkeypatch.setenv("YAP_THICKNESS_FAMILY_DIR", str(tmp_path / "absent-family"))
    monkeypatch.setattr(components, "load_catalog", lambda *_a, **_k: {"entries": catalog_entries})
    with pytest.raises(ValueError, match="unknown component|family|index"):
        components.resolve_entry("component:BI2_SI32")
    assert components.resolve_entry("component:BI2_SI37")["id"] == "BI2_SI37"


def test_empty_family_directory_reports_missing_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("YAP_THICKNESS_FAMILY_DIR", str(tmp_path))
    monkeypatch.setattr(components, "load_catalog", lambda *_a, **_k: {"entries": []})
    with pytest.raises((ValueError, FileNotFoundError), match="FAMILY|family|manifest"):
        components.resolve_entry("component:BI2_SI32")


def test_activated_si32_is_unit43p5_and_prepared45(catalog_entries, monkeypatch):
    monkeypatch.setenv("YAP_THICKNESS_FAMILY_DIR", str(FAMILY))
    monkeypatch.setattr(components, "load_catalog", lambda *_a, **_k: {"entries": catalog_entries})
    resolved = components.resolve_entry("component:BI2_SI32")
    assert resolved["kind"] == "endpoint_thickness_family"
    assert resolved["family_directory"] == str(FAMILY)
    assert resolved["geometry"]["si_tsv_thickness_um"] == 32.
    assert resolved["geometry"]["total_thickness_um"] == 43.5
    assert resolved["experimental_prepared_total_um"] == 45.
    assert resolved["capabilities"]["aged_state_transfer"] is False
    assert resolved["capabilities"]["W2W_thinning"] is False
    assert resolved["validation"]["prepared_assembly_accuracy"] is False
    assert resolved["validation"]["every_interpolated_thickness_independently_validated"] is False
    request = components.prepared_operator_request("component:BI2_SI32", geometry(45.), layout=(2., 2., 1.))
    assert request["validated_single_BI_total_um"] == 43.5
    assert request["retained_si_thickness_um"] == 32.
    assert request["total_thickness_um"] == 45.
    assert request["physical_readiness"] is request["assembly_accuracy_validated"] is False
    with pytest.raises(ValueError, match="plus one backside BI"):
        components.prepared_operator_request("component:BI2_SI32", geometry(43.5))


@pytest.mark.parametrize("identifier", ["BI2_SI26p999", "BI2_SI37p001", "BI2_SI100"])
def test_family_never_extrapolates(identifier, catalog_entries, monkeypatch):
    monkeypatch.setenv("YAP_THICKNESS_FAMILY_DIR", str(FAMILY))
    monkeypatch.setattr(components, "load_catalog", lambda *_a, **_k: {"entries": catalog_entries})
    with pytest.raises(ValueError, match="outside endpoint bracket; no extrapolation"):
        components.resolve_entry("component:" + identifier)


@pytest.mark.parametrize("identifier", ["BI3_SI32", "BI2_SInan", "BI2_SI-1", "BI2_SI32junk"])
def test_nonfamily_identifier_does_not_open_family(identifier, monkeypatch):
    import thickness_operators
    def forbidden(*_args, **_kwargs):
        pytest.fail("malformed or different-pitch IDs must not open a BI2 thickness family")
    monkeypatch.setattr(thickness_operators, "open_family", forbidden)
    assert components._parameterized_family(identifier) is None


@pytest.fixture
def physical_datasets():
    """Complete synthetic face traces and unmodified ordered GP/operator arrays."""
    unit = exact_entry(si=32.)["geometry"]
    lateral = np.array([(x, z) for x in np.linspace(0., .006, 8)
                        for z in np.linspace(0., .006, 8)])
    ports = np.zeros((192, 64))
    ports[0::3, 0], ports[1::3, 1], ports[2::3, 2] = 1., 1., 1.
    for column in range(3, 64):
        ports[3*(column-3), column], ports[-3, column] = 1., -1.
    assert np.linalg.matrix_rank(ports) == 64
    datasets = {}
    for seed, (role, height, offset) in enumerate((
            ("bi", .0015, 0.), ("beol", .010, .0015), ("si", .032, .0115)), 21):
        xyz = np.zeros((2, 64, 3))
        xyz[:, :, 0], xyz[:, :, 2] = lateral[:, 0], lateral[:, 1]
        xyz[0, :, 1], xyz[1, :, 1] = offset, offset + height
        lifts = np.zeros((2, 192, 134))
        for face in (0, 1):
            lifts[face, :, :6] = bridge._canonical_macro_trace(xyz[face], (face-.5)*height)
        lifts[0, :, 6:70], lifts[1, :, 70:134] = ports, ports
        gp = gp_arrays()
        gp["Cu_positions_mm"][:, 1] = offset + np.linspace(.2, .8, 16)*height
        rng = np.random.default_rng(seed)
        rank = 42
        values = dict(kaa=np.eye(134), c=rng.normal(size=(rank, 134))*.001,
            kff_data=np.linspace(2., 3., rank), kff_indices=np.arange(rank, dtype=np.int32),
            kff_indptr=np.arange(rank+1, dtype=np.int32), kff_shape=np.array([rank, rank]),
            fm=rng.normal(size=134)*1e-6, ff=rng.normal(size=rank)*1e-6,
            B_macro=rng.normal(size=(96, 134)), B_free=rng.normal(size=(96, rank)),
            reference_cu_stiffness=np.eye(6), reference_cu_alpha=np.array([1e-5]*3+[0.]*3),
            reference_temperature_c=np.array(25.), **{key: value.copy() for key, value in gp.items()})
        data = dict(coordinates_mm=xyz.reshape(-1, 3), lifts134=lifts.reshape(-1, 134),
            native_node_indices=np.arange(128), **{key: value.copy() for key, value in gp.items()})
        for value in (*data.values(), *values.values()):
            value.flags.writeable = False
        datasets[role] = dict(data=data, values=values, proof=dict(Cu_GP=16))
    return unit, geometry(45.), np.ones(134), datasets


@pytest.mark.parametrize("canonicalize", [False, True])
def test_physical_bridge_keeps_cross_c_all_gp_and_original_arrays(physical_datasets, canonicalize):
    unit, prepared, scale, datasets = physical_datasets
    before = {role: {part: {key: value.tobytes() for key, value in record[part].items()}
                     for part in ("data", "values")} for role, record in datasets.items()}
    operands, certificate = components._physical_operands_from_arrays(
        unit, prepared, scale, datasets, canonicalize_port_roundoff=canonicalize)
    assert certificate["geometry"]["total_thickness_um"] == 45.
    assert certificate["components"]["retained_si"]["thickness_mm"] == .032
    for source_role, target in (("bi", "bi"), ("beol", "beol"), ("si", "retained_si")):
        source, operand = datasets[source_role], operands[target]
        assert np.count_nonzero(source["values"]["c"]) > 0
        np.testing.assert_array_equal(operand["arrays"]["c"], source["values"]["c"])
        assert operand["arrays"]["c"].shape == (42, 134)
        assert set(operand["arrays"]) == set(components.ARRAY_KEYS)
        assert not HISTORY_FIELDS.intersection(operand["arrays"])
        ids = np.column_stack((source["values"]["Cu_native_element_ids"],
                               source["values"]["Cu_native_gauss_indices"]))
        np.testing.assert_array_equal(operand["ordered_gp"]["element_gauss_ids"], ids)
        np.testing.assert_array_equal(operand["arrays"]["weights"], source["values"]["weights"])
        assert len(operand["arrays"]["weights"]) == 16
        for part in ("data", "values"):
            assert {key: value.tobytes() for key, value in source[part].items()} == before[source_role][part]


@pytest.mark.parametrize("fault", ["gp_order", "gp_position", "gp_weights", "float_ids", "gp_count"])
def test_bridge_rejects_changed_or_interpolated_gp_identity(physical_datasets, fault):
    unit, prepared, scale, datasets = physical_datasets
    record = datasets["si"]
    if fault == "gp_count":
        record["proof"]["Cu_GP"] = 8
    else:
        key = {"gp_order": "Cu_native_gauss_indices", "gp_position": "Cu_positions_mm",
               "gp_weights": "weights", "float_ids": "Cu_native_element_ids"}[fault]
        changed = record["values"][key].copy()
        if fault == "gp_order": changed = np.roll(changed, 1)
        elif fault == "gp_position": changed[0, 1] += 1e-6
        elif fault == "gp_weights": changed[0] *= 1.01
        else: changed = changed.astype(float)
        record["values"][key] = changed
    with pytest.raises(ValueError, match="GP"):
        components._physical_operands_from_arrays(unit, prepared, scale, datasets)


@pytest.mark.parametrize("error", [1e-14, 5e-13])
def test_only_physical_port_roundoff_is_canonicalized(physical_datasets, error):
    unit, prepared, scale, datasets = physical_datasets
    lifts = datasets["si"]["data"]["lifts134"].copy()
    lifts[0, 9] += error
    lifts[192, 73] += error
    datasets["si"]["data"]["lifts134"] = lifts
    before = lifts.copy()
    raw, _ = components._physical_operands_from_arrays(unit, prepared, scale, datasets)
    assert not np.array_equal(raw["bi"]["physical_trace"]["face_lifts"][0, :, 6:70],
                              raw["retained_si"]["physical_trace"]["face_lifts"][0, :, 6:70])
    snapped, _ = components._physical_operands_from_arrays(
        unit, prepared, scale, datasets, canonicalize_port_roundoff=True)
    expected = snapped["bi"]["physical_trace"]["face_lifts"][0, :, 6:70]
    np.testing.assert_array_equal(snapped["retained_si"]["physical_trace"]["face_lifts"][0, :, 6:70], expected)
    np.testing.assert_array_equal(snapped["retained_si"]["physical_trace"]["face_lifts"][1, :, 70:134], expected)
    np.testing.assert_array_equal(lifts, before)
    for role in snapped:
        for key in components.ARRAY_KEYS:
            np.testing.assert_array_equal(snapped[role]["arrays"][key], raw[role]["arrays"][key])


def test_port_discrepancy_above_certified_roundoff_is_rejected(physical_datasets):
    unit, prepared, scale, datasets = physical_datasets
    lifts = datasets["si"]["data"]["lifts134"].copy()
    lifts[0, 9] += 2e-12
    datasets["si"]["data"]["lifts134"] = lifts
    with pytest.raises(ValueError, match="more than registration roundoff"):
        components._physical_operands_from_arrays(unit, prepared, scale, datasets, canonicalize_port_roundoff=True)


@pytest.mark.parametrize("missing", ["bi", "beol", "si"])
def test_complete_physical_roles_are_required(physical_datasets, missing):
    unit, prepared, scale, datasets = physical_datasets
    datasets.pop(missing)
    with pytest.raises(ValueError, match="complete physical component datasets"):
        components._physical_operands_from_arrays(unit, prepared, scale, datasets)


def fake_family(physical_datasets):
    unit, prepared, scale, datasets = physical_datasets
    calls = []
    evaluated = SimpleNamespace(cache_key="b"*64, resolution=SimpleNamespace(si_um=32.),
        component_arrays=lambda: datasets["si"]["values"])
    family = SimpleNamespace(root=FAMILY, manifest_sha256="a"*64,
        manifest=dict(coefficients={"sha256": "c"*64}, fixed_components={"bi": {}, "beol": {}}),
        evaluate=lambda **request: calls.append(("evaluate", request)) or evaluated,
        geometry=lambda *, role, **request: calls.append(("geometry", role, request)) or datasets[role]["data"],
        fixed_component=lambda role: calls.append(("fixed_component", role)) or datasets[role]["values"],
        topology=lambda: {"coordinate_scale": np.r_[scale, np.ones(128)]})
    entry = dict(id="BI2_SI32", kind="endpoint_thickness_family", geometry=unit,
                 family_sha256=family.manifest_sha256, cache_key=evaluated.cache_key)
    return family, evaluated, entry, calls


def test_parameterized_adapter_uses_target_operator_without_aged_gp_state(physical_datasets, monkeypatch):
    family, _, entry, calls = fake_family(physical_datasets)
    monkeypatch.setattr(components, "_parameterized_family", lambda _package: family)
    operands, certificate, provenance = components._parameterized_operands(entry, geometry(45.))
    assert calls.count(("evaluate", {"si_um": 32.})) == 1
    assert {call[1] for call in calls if call[0] == "fixed_component"} == {"bi", "beol"}
    assert certificate["geometry"] == geometry(45.)
    assert provenance["aged_state_transfer"] is provenance["FE_generation"] is provenance["target_POD"] is False
    assert provenance["registration_roundoff_limit_mm"] == 1e-12
    np.testing.assert_array_equal(operands["retained_si"]["arrays"]["c"], physical_datasets[3]["si"]["values"]["c"])
    assert all(not HISTORY_FIELDS.intersection(operand["arrays"]) for operand in operands.values())


@pytest.mark.parametrize("binding", ["family", "evaluation"])
def test_parameterized_adapter_refuses_changed_bindings(physical_datasets, monkeypatch, binding):
    family, evaluated, entry, _ = fake_family(physical_datasets)
    if binding == "family": family.manifest_sha256 = "d"*64
    else: evaluated.cache_key = "d"*64
    monkeypatch.setattr(components, "_parameterized_family", lambda _package: family)
    with pytest.raises(ValueError, match="family changed|evaluation binding changed"):
        components._parameterized_operands(entry, geometry(45.))


@pytest.mark.skipif(os.environ.get("YAP_RUN_REAL_THICKNESS_GEOMETRY") != "1",
                    reason="explicit opt-in bounded real geometry-only diagnostic")
def test_real_family_trace_roundoff_and_gp_identity(tmp_path):
    from thickness_operators import open_family

    family = open_family(FAMILY, memory_cache_bytes=0, disk_cache_bytes=0)
    diagnostic = dict(scope="geometry_only_no_operator_evaluation_or_history", family=str(FAMILY), roles={})
    common = None
    si_identity = None
    for role in ("bi", "beol", "si"):
        data = family.geometry(role=role, si_um=32.)
        xyz = data["coordinates_mm"]
        faces = [np.flatnonzero(xyz[:, 1] == value) for value in (xyz[:, 1].min(), xyz[:, 1].max())]
        faces = [face[np.lexsort((xyz[face, 2], xyz[face, 0]))] for face in faces]
        lifts = data["lifts134"].reshape(-1, 3, 134)
        ports = [lifts[face].reshape(-1, 134)[:, start:start+64]
                 for face, start in zip(faces, (6, 70))]
        if common is None:
            common = ports[0].copy()
        errors = [float(np.max(abs(port-common))) for port in ports]
        assert max(errors) <= 1e-12
        ids = data["Cu_native_element_ids"]
        gauss = data["Cu_native_gauss_indices"]
        assert ids.dtype.kind in "iu" and gauss.dtype.kind in "iu"
        assert ids.shape == gauss.shape and len(ids) > 0
        assert np.all((gauss >= 0) & (gauss < 8))
        diagnostic["roles"][role] = dict(GP=len(ids), face_nodes=len(faces[0]),
            maximum_port_error=errors, bitwise_equal=[bool(np.array_equal(port, common)) for port in ports])
        if role == "si": si_identity = (ids.copy(), gauss.copy())
    for thickness in (27., 37.):
        data = family.geometry(role="si", si_um=thickness)
        np.testing.assert_array_equal(data["Cu_native_element_ids"], si_identity[0])
        np.testing.assert_array_equal(data["Cu_native_gauss_indices"], si_identity[1])
    assert family.cache_info()["evaluations"] == 0
    diagnostic["Si_GP_identity_equal_at_27_32_37"] = True
    diagnostic["canonicalization_needed_for_bitwise_common_trace"] = any(
        not all(role["bitwise_equal"]) for role in diagnostic["roles"].values())
    target = tmp_path / "REAL_GEOMETRY_TRACE.json"
    target.write_text(json.dumps(diagnostic, indent=2)+"\n", encoding="ascii")
    print(json.dumps(dict(diagnostic, receipt=str(target)), sort_keys=True))
