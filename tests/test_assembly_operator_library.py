"""Small native algebra fixtures, not physical validation of the new route."""

from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from W2W import assembly_operator_library as adapter

def geometry():
    return {**adapter._GEOMETRY_UM, "diameter_mm": 300.}


def native(name):
    if not (Path(__file__).resolve().parents[2] / "Ansys_Verify/12_stack_model_validation").is_dir():
        pytest.skip("sibling native mechanics repository is not installed")
    return adapter._import_mechanics_module(name)


def dataset(thickness, rank=1, bubble=True):
    """Analytical plane-stress Si plus a shared-face spring and one bubble."""
    materials = native("exp12_material_separable_reference")
    si = materials.ConstantIsotropicMaterial(**adapter._SI)
    lame, shear = materials.lame_parameters(si.elastic_modulus_mpa, si.poisson_ratio)
    size = 6 + 2 * rank
    e, nu, area = si.elastic_modulus_mpa, si.poisson_ratio, 36.e-6
    q = e / (1. - nu ** 2) * np.array([[1., nu, 0.], [nu, 1., 0.],
                                      [0., 0., (1. - nu) / 2.]])
    k = np.zeros((size, size))
    k[:3, :3] = q * thickness * area
    k[3:6, 3:6] = q * thickness ** 3 * area / 12.
    spring = np.eye(rank) / thickness
    k[6:6 + rank, 6:6 + rank] = spring
    k[6 + rank:, 6 + rank:] = spring
    k[6:6 + rank, 6 + rank:] = -spring
    k[6 + rank:, 6:6 + rank] = -spring
    thermal = np.zeros(size)
    thermal[:2] = si.secant_cte_per_c
    thermal[6:6 + rank] = -.5 * si.secant_cte_per_c * thickness
    thermal[6 + rank:] = .5 * si.secant_cte_per_c * thickness
    f = k @ thermal
    if bubble:
        coupling = np.zeros(size)
        coupling[0], coupling[6], coupling[-1] = .03, -.1, .1
        k = np.block([[k + np.outer(coupling, coupling) / 2., coupling[:, None]],
                      [coupling[None, :], np.array([[2.]])]])
        f = np.r_[f + coupling * .0002 / 2., .0002]
    return SimpleNamespace(
        validate=lambda: None, linear_reference_mode="material_separable_bbar",
        reference_constant_materials={"si": si},
        reference_lambda_stiffness_integrals=np.zeros((1, len(k), len(k))),
        reference_shear_stiffness_integrals=(k / shear)[None],
        reference_volumetric_virtual_strain_integrals=(
            f / ((3. * lame + 2. * shear) * si.secant_cte_per_c))[None],
        reference_temperature_c=25., temperature_grid_c=np.array([25., 350.]),
        cu_card=native("exp12_cu_material_point").CuCard(),
        macro_size=size, bubble_size=int(bubble),
        macro_coordinate_scale=np.r_[np.full(3, .001), np.full(3, .01),
                                     np.full(2 * rank, .001)],
        bubble_coordinate_scale=np.full(int(bubble), .002),
        geometry_signature={"thickness_mm": thickness, "cell_area_mm2": area},
    )


def library(rank=1, bubble=True):
    source = {key: dataset(thickness, rank, bubble)
              for key, thickness in adapter._THICKNESS_MM.items()}
    pure = native("exp12_all_gauss_linear_layer").pure_constant_material_operator_from_dataset
    operators = {key: pure(value) for key, value in source.items()}
    base = native("exp12_mindlin_sequential_runtime").Q6SequentialLibrary(
        operators=operators, thickness_mm=dict(adapter._THICKNESS_MM),
        reflection_matrix=np.eye(rank), rigid_port_mode_indices=tuple(range(rank)),
        membrane_bending_coordinate_scale=operators["bi"].macro_coordinate_scale[:6],
        port_coordinate_scale=np.full(rank, .001), cell_area_mm2=36.e-6,
        port_rank_per_face=rank,
    )
    return base, source


def test_segment_generator_removes_history_and_condenses_bubble():
    base, _ = library()
    original = base.operators["retained_si"]
    operator, audit = adapter._condense_segment(original)
    assert operator.macro_size == 8
    assert operator.bubble_size == 0
    assert operator.cu_point_count == 0
    assert not operator.plastic_active
    assert max(audit.values()) < 1e-7
    k = original.reference_stiffness(25.)
    f = original.reference_thermal_load(26.)
    expected_k = k[:8, :8] - np.outer(k[:8, 8], k[8, :8]) / k[8, 8]
    expected_f = f[:8] - k[:8, 8] * f[8] / k[8, 8]
    np.testing.assert_allclose(operator.reference_stiffness(350.), expected_k, atol=1e-12)
    np.testing.assert_allclose(operator.reference_thermal_load(350.), 325. * expected_f,
                               atol=1e-12)


def test_pairwise_cap_matches_monolithic_native_column_and_analytical_gates():
    base, _ = library()
    segments = {key: adapter._condense_segment(op)[0] for key, op in base.operators.items()}
    cap, audit = adapter._build_cap(base, segments)
    assert audit["segment_count"] == 153
    assert audit["segment_recipe"].count("retained_si") == 14
    assert audit["segment_recipe"].count("bi") == 139
    assert audit["segment_recipe"] == list(reversed(audit["segment_recipe"]))
    assert audit["segment_orientations"] == [-1] * 76 + [1] * 77
    assert audit["coarse_cap_audits"]["segment_count"] == 29
    assert audit["maximum_pair_generalized_size"] == 9
    assert audit["thickness_um"] == pytest.approx(726.5)
    assert audit["maximum_reaction_relative_defect"] < 1e-7
    assert cap.cu_point_count == 0
    topology = adapter._column(base, [(segments[key], base.thickness_mm[key])
                                     for key in audit["segment_recipe"]],
                               orientations=audit["segment_orientations"])
    k = np.zeros((topology.generalized_size, topology.generalized_size))
    f = np.zeros(topology.generalized_size)
    for instance in topology.instances:
        transform, op = instance.macro_transform, instance.operator
        k += transform.T @ op.reference_stiffness(25.) @ transform
        f += transform.T @ op.reference_thermal_load(26.)
    keep = np.r_[np.arange(7), topology.interface_slices[-1].start]
    kc, fc, _ = adapter._condense_checked(k, f, keep, topology.coordinate_scale)
    np.testing.assert_allclose(cap.reference_stiffness(25.), kc, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(cap.reference_thermal_load(350.), 325. * fc,
                               rtol=1e-10, atol=1e-11)
    physical = adapter._audit_cap(base, cap)
    assert physical["maximum_scaled_abd_defect"] < 1e-8
    assert physical["thermal_q6_max_absolute_defect"] < 1e-8
    assert set(physical["maximum_scaled_abd_defect_by_block"]) == {"A", "B", "D"}
    reversed_cap = adapter._oriented_segment(base, cap, .7265, -1)
    reversed_audit = adapter._abd_audit(base, reversed_cap, .7265)
    np.testing.assert_allclose(reversed_audit["abd_per_width"], physical["abd_per_width"],
                               rtol=1e-10, atol=1e-8)
    assert .0015 + .010 + .037 + .7265 == pytest.approx(.775)
    assert .0015 + .010 + .037 + .0015 == pytest.approx(.050)


def test_native_pair_condensation_is_parenthesization_independent():
    base, _ = library()
    parts = [adapter._condense_segment(base.operators[key])[0]
             for key in ("bi", "retained_si", "beol")]
    parts[1] = adapter._oriented_segment(base, parts[1], .037, -1)
    nodes = list(zip(parts, (.0015, .037, .010)))
    left = adapter._join_segments(base, adapter._join_segments(base, *nodes[:2])[0], nodes[2])[0][0]
    right = adapter._join_segments(base, nodes[0], adapter._join_segments(base, *nodes[1:])[0])[0][0]
    scale = left.macro_coordinate_scale
    for temperature in (25., 350.):
        kl, kr = left.reference_stiffness(temperature), right.reference_stiffness(temperature)
        fl, fr = left.reference_thermal_load(temperature), right.reference_thermal_load(temperature)
        np.testing.assert_allclose(scale[:, None] * kl * scale, scale[:, None] * kr * scale,
                                   rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(scale * fl, scale * fr, rtol=1e-10, atol=1e-14)
        for q in np.diag(scale):
            np.testing.assert_allclose(scale * (kl @ q - fl), scale * (kr @ q - fr),
                                       rtol=1e-10, atol=1e-14)


def test_second_symmetric_partition_preserves_tiny_native_reference():
    base, _ = library(bubble=False)
    cap, _ = adapter._build_cap(base, base.operators)
    half = ["bi"] * 143 + ["retained_si"] * 4
    refined, report = adapter._reduce_recipe(base, base.operators,
        half + ["bi"] + list(reversed(half)), [-1] * len(half) + [1] * (len(half) + 1))
    assert report["segment_count"] == 295
    old, new = adapter._abd_audit(base, cap, .7265), adapter._abd_audit(base, refined, .7265)
    np.testing.assert_allclose(old["abd_per_width"], new["abd_per_width"], rtol=1e-10, atol=1e-8)


def test_reaction_gate_detects_corrupt_condensed_thermal_load(monkeypatch):
    preview = native("exp12_w2w775_preview")
    original = preview.schur_condense

    def broken(*args):
        k, f, extension, thermal = original(*args)
        return k, f + .1, extension, thermal

    monkeypatch.setattr(preview, "schur_condense", broken)
    with pytest.raises(ValueError, match="reaction/internal-equilibrium"):
        adapter._condense_checked(np.eye(3), np.ones(3), [0, 2], np.ones(3))


@pytest.mark.parametrize("fault", ["stiffness", "thermal"])
def test_analytical_gate_detects_incorrect_cap(fault):
    base, _ = library(bubble=False)
    cap = adapter._condense_segment(
        native("exp12_all_gauss_linear_layer").pure_constant_material_operator_from_dataset(
            dataset(.7265, bubble=False)))[0]
    k, f = cap.reference_stiffness(25.), cap.reference_thermal_load(26.)
    broken = adapter._linear_operator(k * (2. if fault == "stiffness" else 1.),
        f * (2. if fault == "thermal" else 1.), cap.macro_coordinate_scale, cap.cu_card)
    with pytest.raises(ValueError, match="analytical ABD/thermal"):
        adapter._audit_cap(base, broken)


def test_library_rejects_wrong_thickness_and_coordinate_scales():
    base, _ = library()
    with pytest.raises(ValueError, match="thickness mismatch"):
        adapter._check_library(replace(base, thickness_mm={**base.thickness_mm, "retained_si": .0385}))
    with pytest.raises(ValueError, match="port coordinates mismatch"):
        adapter._check_library(replace(base, port_coordinate_scale=np.array([2.])))


def test_prepared_aliases_preserve_checked_observed_thickness():
    base, _ = library()
    source = replace(base,
        operators={adapter._SOURCE_KEYS[key]: value for key, value in base.operators.items()},
        thickness_mm={adapter._SOURCE_KEYS[key]: value for key, value in base.thickness_mm.items()})
    prepared = adapter._prepared_library(source)
    assert prepared.thickness_mm == base.thickness_mm
    assert prepared.operators == base.operators
    bad = replace(source, thickness_mm={**source.thickness_mm, "tsv_si_h37": .067})
    with pytest.raises(ValueError, match="native dataset thickness mismatch"):
        adapter._prepared_library(bad)


def test_pure_si_rejects_inelastic_segment():
    base, _ = library()
    op = base.operators["bi"]
    op.plastic_active = True
    with pytest.raises(ValueError, match="no Cu history"):
        adapter._condense_segment(op)


def package_file(tmp_path, name="BI6_50", rank=120):
    datasets, hashes, prepared = {}, {}, {}
    for source in adapter._SOURCE_KEYS.values():
        filename = f"{source}.npz"
        content = f"synthetic {source}".encode()
        (tmp_path / filename).write_bytes(content)
        datasets[source], hashes[source] = filename, hashlib.sha256(content).hexdigest()
        prepared[source] = f"prepared_{source}.npz"
        (tmp_path / prepared[source]).write_bytes(b"prepared fixture")
    basis = tmp_path / "port.npz"
    basis.write_bytes(b"fixture basis")
    package = dict(schema="exp12.ofat.geometry_package.v1", name=name, rank=rank,
                   ready=True, state="READY_ENGINEERING_PREVIEW", datasets=datasets,
                   dataset_sha256=hashes, prepared=prepared, port_basis=basis.name,
                   port_basis_sha256=hashlib.sha256(basis.read_bytes()).hexdigest())
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(package))
    return path, package


@pytest.mark.parametrize("name,rank", [("BI6_50", 120), ("BI2_50", 64)])
def test_loader_normalizes_keys_hashes_inputs_and_dispatches_checked_native_load(tmp_path, monkeypatch, name, rank):
    path, package = package_file(tmp_path, name, rank)
    base, sources = library(rank=rank, bubble=False)
    runtime = native("exp12_mindlin_sequential_runtime")
    data_module = native("exp12_layer_rom_dataset")
    loader = Mock(return_value=base)
    monkeypatch.setattr(runtime, "load_q6_sequential_library", loader)
    reverse = {value: key for key, value in adapter._SOURCE_KEYS.items()}
    read = Mock(side_effect=lambda path, **kwargs: sources[reverse[path.stem]])
    monkeypatch.setattr(data_module.LayerROMDataset, "load", read)
    monkeypatch.setattr(adapter, "_condense_segment", lambda op: (op, {"fixture": 0.}))
    cap = base.operators["retained_si"]
    monkeypatch.setattr(adapter, "_build_cap", lambda lib, segments: (cap, {"fixture": True}))
    monkeypatch.setattr(adapter, "_audit_cap", lambda lib, op: {"fixture_only": True})
    loaded, provenance = adapter.load_assembly_library(path, geometry())
    assert set(loaded.operators) == {"bi", "beol", "retained_si", "bulk_si"}
    assert loaded.thickness_mm["bulk_si"] == .7265
    assert loaded.operators["bi"] is base.operators["bi"]
    assert loaded.operators["bulk_si"] is cap
    args, kwargs = loader.call_args
    assert set(args[0]) == set(adapter._SOURCE_KEYS)
    assert args[0]["retained_si"] == tmp_path / "tsv_si_h37.npz"
    assert args[1] == tmp_path / "port.npz"
    assert kwargs["validate"] is True and kwargs["verify_content"] is True
    assert kwargs["port_rank_per_face"] == rank
    assert set(kwargs["prepared_geometry_paths"]) == set(adapter._SOURCE_KEYS)
    assert set(kwargs["frozen_expected_port_basis_sha256_by_dataset"].values()) == {
        package["port_basis_sha256"]}
    assert len(provenance["input_sha256"]) == 8
    for name, digest in provenance["input_sha256"].items():
        assert adapter._sha256(name) == digest
    assert read.call_count == 3
    assert all(call.kwargs == {"validate": True, "verify_content": True}
               for call in read.call_args_list)
    json.dumps(provenance)


@pytest.mark.parametrize("filename", ["bi.npz", "tsv_si_h37.npz", "port.npz"])
def test_loader_rejects_tampering_before_importing_native_modules(tmp_path, monkeypatch, filename):
    path, _ = package_file(tmp_path)
    (tmp_path / filename).write_bytes(b"tampered")
    imported = Mock(side_effect=AssertionError("native loader must not run"))
    monkeypatch.setattr(adapter, "_import_mechanics_module", imported)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        adapter.load_assembly_library(path, geometry())
    imported.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("initial_total_thickness_um", 774.), ("target_total_thickness_um", 51.),
    ("front_bi_thickness_um", 2.), ("back_bi_thickness_um", 0.),
    ("beol_thickness_um", 5.), ("diameter_mm", float("nan")),
])
def test_loader_rejects_unsupported_geometry_before_file_access(field, value):
    with pytest.raises(ValueError, match=field):
        adapter.load_assembly_library("missing-package.json", {**geometry(), field: value})


@pytest.mark.parametrize("field,value", [("rank", 24), ("name", "BI6_80"), ("ready", False),
    ("rank", 64), ("name", "BI2_50"), ("rank", 120.0), ("rank", True)])
def test_loader_rejects_other_packages(tmp_path, field, value):
    path, package = package_file(tmp_path)
    package[field] = value
    path.write_text(json.dumps(package))
    with pytest.raises(ValueError, match="native120 BI6_50"):
        adapter.load_assembly_library(path, geometry())


def test_loader_rejects_wrong_returned_rank(tmp_path, monkeypatch):
    path, _ = package_file(tmp_path, "BI2_50", 64)
    base, _ = library(rank=120, bubble=False)
    runtime = native("exp12_mindlin_sequential_runtime")
    monkeypatch.setattr(runtime, "load_q6_sequential_library", Mock(return_value=base))
    with pytest.raises(ValueError, match="retain native64"):
        adapter.load_assembly_library(path, geometry())


def test_loader_rejects_original_dataset_rank_before_condensation(tmp_path, monkeypatch):
    path, _ = package_file(tmp_path, "BI2_50", 64)
    base, _ = library(rank=64, bubble=False)
    _, wrong = library(rank=120, bubble=False)
    runtime = native("exp12_mindlin_sequential_runtime")
    module = native("exp12_layer_rom_dataset")
    monkeypatch.setattr(runtime, "load_q6_sequential_library", Mock(return_value=base))
    monkeypatch.setattr(module.LayerROMDataset, "load", Mock(return_value=wrong["bi"]))
    with pytest.raises(ValueError, match="segment port coordinates mismatch"):
        adapter.load_assembly_library(path, geometry())


def test_fine_cap_failure_reports_error_without_relaxing_gate(monkeypatch):
    base, _ = library(bubble=False)
    original = adapter._abd_audit
    def failed_audit(*args):
        audit = original(*args)
        audit["maximum_scaled_abd_defect"] = .0159
        audit["maximum_scaled_abd_defect_by_block"]["D"] = .0159
        return audit
    monkeypatch.setattr(adapter, "_abd_audit", failed_audit)
    with pytest.raises(ValueError, match="defect=0.0159, limit=0.01"):
        adapter._build_cap(base, base.operators)


@dataclass
class PreparedFixtureCard:
    value: float = 1.


@dataclass
class PreparedFixtureLibrary:
    operators: dict
    thickness_mm: dict
    membrane_bending_coordinate_scale: np.ndarray
    port_coordinate_scale: np.ndarray
    cell_area_mm2: float = 36e-6
    port_rank_per_face: int = 64
    common_port_gauge_vectors: object = None


def prepared_fixture_geometry(total, kind="die"):
    layers = dict(front_bi_thickness_um=1.5, back_bi_thickness_um=1.5, beol_thickness_um=10.)
    if kind == "die":
        return dict(layers, total_thickness_um=total, width_mm=10., length_mm=10.)
    return dict(layers, initial_total_thickness_um=total, target_total_thickness_um=total, diameter_mm=300.)


@pytest.fixture
def prepared_native64_fixture(tmp_path, monkeypatch):
    # These deliberately non-scientific artifacts exercise only the admission contract.
    for directory in ("packages", "audit", "shared", "datasets", "prepared", "build_receipts"):
        (tmp_path / directory).mkdir()
    def blob(relative, payload):
        path = tmp_path / relative
        path.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()
    def dataset_blob(relative, key):
        path = tmp_path / relative
        content = hashlib.sha256(f"synthetic {key} content".encode()).hexdigest()
        np.savez(path, content_sha256=np.array(content))
        return hashlib.sha256(path.read_bytes()).hexdigest()
    basis_hash = blob("shared/port_basis.npz", b"synthetic native64 basis")
    monkeypatch.setattr(adapter, "_PREPARED_NATIVE64_BASIS_SHA256", basis_hash)
    monkeypatch.setattr(adapter, "_input_path", lambda base, value: (Path(base) / value).resolve())
    catalog_path = tmp_path / "catalog.json"
    monkeypatch.setattr(adapter, "_PREPARED_OPERATOR_CATALOG", catalog_path)
    shared = {}
    for key in ("bi", "beol"):
        shared[key] = (dataset_blob(f"shared/{key}.npz", key),
                       blob(f"shared/{key}_prepared.npz", f"synthetic {key} geometry".encode()))
    parent = dict(schema="exp12.ofat.geometry_package.v1", name="BI2_50", rank=64, ready=True,
        state="READY_ENGINEERING_PREVIEW", port_basis="/old/unavailable/port_basis.npz",
        port_basis_sha256=basis_hash, datasets={key: f"/old/unavailable/{key}.npz" for key in shared},
        dataset_sha256={key: hashes[0] for key, hashes in shared.items()})
    packages, receipts, source_datasets = {}, {}, {}
    for total in (30., 80.):
        retained = total - 13.
        target = f"tsv_si_h{retained:g}"
        hashes = dict(shared)
        hashes[target] = (dataset_blob(f"datasets/{target}.npz", target),
                          blob(f"prepared/{target}.npz", f"synthetic {target} geometry".encode()))
        transfer_hash = blob(f"build_receipts/{target}_bubble.npz", f"synthetic {target} bubbles".encode())
        raw_hash = blob(f"build_receipts/{target}_raw_bubble.npz", f"synthetic {target} raw bubbles".encode())
        ds_paths = {key: f"../shared/{key}.npz" if key in shared else f"../datasets/{key}.npz" for key in hashes}
        pg_paths = {key: f"../shared/{key}_prepared.npz" if key in shared else f"../prepared/{key}.npz" for key in hashes}
        packages[total] = dict(schema="yap.prepared_p2_geometry_package.v1", name=f"BI2_{total:g}", rank=64,
            ready=False, state="CANDIDATE_NOT_VALIDATED", datasets=ds_paths, prepared=pg_paths,
            dataset_sha256={key: value[0] for key, value in hashes.items()},
            prepared_sha256={key: value[1] for key, value in hashes.items()},
            port_basis="../shared/port_basis.npz", port_basis_sha256=basis_hash,
            parent_package="../shared/parent_BI2_50.json", audit_receipt=f"../audit/BI2_{total:g}.json")
        bindings, geometry_bindings = {}, {}
        for key, value in hashes.items():
            thickness = dict(bi=.0015, beol=.010).get(key, retained / 1000.)
            content = hashlib.sha256(f"synthetic {key} content".encode()).hexdigest()
            bubble_size = 3 if key in shared else 167
            bindings[key] = dict(sha256=value[0], dataset_content_sha256=content,
                cu_point_count=2, thickness_mm=thickness, macro_size=134, bubble_size=bubble_size)
            geometry_bindings[key] = dict(sha256=value[1], parent_dataset_content_sha256=content)
            source_datasets[key] = SimpleNamespace(macro_size=134, bubble_size=bubble_size,
                macro_coordinate_scale=np.ones(134), geometry_signature=dict(thickness_mm=thickness, cell_area_mm2=36e-6),
                cu_card=PreparedFixtureCard(), reference_constant_materials={"si": PreparedFixtureCard()},
                reference_temperature_c=25., linear_reference_mode="material_separable_bbar")
        transfer = dict(source_thickness_um=38.5, target_thickness_um=retained, source_pitch_um=6.,
            rank=167, observed_rank=167, source_rank=96, raw_transferred_rank=96, closure_added_rank=71,
            singular_value_min=1., singular_value_max=1., gram_defect=1e-12,
            face_trace=1e-12, relative_periodic_jump=1e-6,
            source_bubble_sha256=hashlib.sha256(b"donor bubbles").hexdigest(),
            source_dataset_sha256=hashlib.sha256(b"donor dataset").hexdigest(),
            transferred_bubble_sha256=transfer_hash,
            method="normalized_thickness_displacement_transfer_plus_inplane_swap_closure",
            transferred_bubble=dict(path=f"../build_receipts/{target}_bubble.npz", sha256=transfer_hash),
            raw_transferred_bubble_sha256=raw_hash,
            raw_transferred_bubble=dict(path=f"../build_receipts/{target}_raw_bubble.npz", sha256=raw_hash),
            raw_transfer_audit=dict(rank=96, observed_rank=96, singular_value_min=1., singular_value_max=1.,
                gram_defect=1e-12, face_trace=1e-12, relative_periodic_jump=1e-6))
        receipts[total] = dict(schema="yap.prepared_p2_operator_audit.v1", state="PASS_STRUCTURAL_ENGINEERING_CANDIDATE",
            total_thickness_um=total, retained_si_thickness_um=retained, bonding_pitch_um=2.,
            bonding_pad_diameter_um=1., rank=64, port_basis_sha256=basis_hash, stage_count=6,
            all_cu_points_retained=True, thermal_geometry_validated=False, face_node_shape=[46, 46], cell_area_mm2=36e-6,
            datasets=bindings, prepared=geometry_bindings, transfer_audit=transfer)
    def write_json(path, value):
        payload = json.dumps(value, sort_keys=True).encode()
        path.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()
    def seal():
        parent_hash = write_json(tmp_path / "shared/parent_BI2_50.json", parent)
        catalog = dict(schema="yap.prepared_p2_operator_catalog.v1", packages={})
        for total in (30., 80.):
            packages[total]["parent_package_sha256"] = parent_hash
            receipts[total]["parent_package_sha256"] = parent_hash
            audit_hash = write_json(tmp_path / f"audit/BI2_{total:g}.json", receipts[total])
            packages[total]["audit_receipt_sha256"] = audit_hash
            package_hash = write_json(tmp_path / f"packages/BI2_{total:g}.json", packages[total])
            catalog["packages"][f"{total:g}"] = dict(package_sha256=package_hash, audit_sha256=audit_hash,
                parent_package_sha256=parent_hash, port_basis_sha256=basis_hash)
        write_json(catalog_path, catalog)
        return catalog
    def load_library(datasets, basis, **options):
        operators, thicknesses = {}, {}
        for key in datasets:
            data = source_datasets[key]
            operators[key] = SimpleNamespace(macro_size=134, bubble_size=data.bubble_size, macro_coordinate_scale=np.ones(134),
                cu_point_count=2, weights_mm3=np.ones(2), cu_card=PreparedFixtureCard())
            thicknesses[key] = data.geometry_signature["thickness_mm"]
        return PreparedFixtureLibrary(operators, thicknesses, np.ones(6), np.ones(64))
    runtime = SimpleNamespace(__file__=adapter.__file__, load_q6_sequential_library=Mock(side_effect=load_library))
    data_module = SimpleNamespace(__file__=adapter.__file__, LayerROMDataset=SimpleNamespace(
        load=Mock(side_effect=lambda path, **_: source_datasets[path.stem])))
    imported = Mock(side_effect=lambda name: {"exp12_mindlin_sequential_runtime": runtime,
                                             "exp12_layer_rom_dataset": data_module}[name])
    monkeypatch.setattr(adapter, "_import_mechanics_module", imported)
    monkeypatch.setattr(adapter, "_build_cap", Mock(side_effect=AssertionError("prepared bodies have no cap")))
    seal()
    return SimpleNamespace(root=tmp_path, packages=packages, receipts=receipts, parent=parent,
        datasets=source_datasets, catalog_path=catalog_path, seal=seal, runtime=runtime,
        data_module=data_module, imported=imported)


@pytest.mark.parametrize("total", [30., 80.])
def test_prepared_native64_loads_exact_layers_without_cap(prepared_native64_fixture, total):
    fixture = prepared_native64_fixture
    path = fixture.root / f"packages/BI2_{total:g}.json"
    geometry = prepared_fixture_geometry(total)
    library, provenance = adapter.load_prepared_die_library(path, geometry)
    assert set(library.operators) == {"bi", "beol", "retained_si"}
    assert library.thickness_mm == dict(bi=.0015, beol=.010, retained_si=(total - 13.) / 1000.)
    assert provenance["port_rank_per_face"] == 64
    assert provenance["operator_inputs_verified"] is True
    assert provenance["physical_readiness"] is False
    assert provenance["frozen_preview"] is False
    assert provenance["removable_bulk_operator_built"] is False
    assert provenance["all_cu_points_retained"] is True
    assert library.operators["retained_si"].bubble_size == 167
    assert "H38P5_TRANSFERRED_HOMOGENEOUS" in provenance["physics_qualification"]
    args, kwargs = fixture.runtime.load_q6_sequential_library.call_args
    assert set(args[0]) == {"bi", "beol", f"tsv_si_h{total - 13.:g}"}
    assert kwargs["validate"] is kwargs["verify_content"] is kwargs["factorize_elements"] is True
    assert kwargs["port_rank_per_face"] == 64
    assert set(kwargs["prepared_geometry_paths"]) == set(args[0])
    assert set(kwargs["frozen_expected_port_basis_sha256_by_dataset"].values()) == {adapter._PREPARED_NATIVE64_BASIS_SHA256}
    assert len(provenance["input_sha256"]) == 13
    assert fixture.data_module.LayerROMDataset.load.call_count == 3
    assert all(call.kwargs == dict(validate=True, verify_content=True)
               for call in fixture.data_module.LayerROMDataset.load.call_args_list)
    assert all(not hasattr(data, "content_sha256") for data in fixture.datasets.values())
    json.dumps(provenance)


@pytest.mark.parametrize("relative", ["packages/BI2_30.json", "audit/BI2_30.json", "shared/parent_BI2_50.json",
    "shared/bi.npz", "datasets/tsv_si_h17.npz", "prepared/tsv_si_h17.npz", "shared/port_basis.npz",
    "build_receipts/tsv_si_h17_bubble.npz", "build_receipts/tsv_si_h17_raw_bubble.npz"])
def test_prepared_native64_rejects_tampering_before_native_import(prepared_native64_fixture, relative):
    fixture = prepared_native64_fixture
    (fixture.root / relative).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        adapter.load_prepared_die_library(fixture.root / "packages/BI2_30.json", prepared_fixture_geometry(30.))
    fixture.imported.assert_not_called()


@pytest.mark.parametrize("total", [30., 80.])
@pytest.mark.parametrize("field,value", [("source_thickness_um", 37.), ("target_thickness_um", 0.),
    ("source_pitch_um", False), ("observed_rank", 2), ("rank", 3.), ("singular_value_min", 0.),
    ("singular_value_max", 1e15), ("gram_defect", 1e-8), ("face_trace", 1e-8),
    ("relative_periodic_jump", 1e-3), ("source_bubble_sha256", "bad"), ("method", ""),
    ("face_trace", float("nan")), ("gram_defect", float("inf")), ("source_rank", 95),
    ("raw_transferred_rank", 95), ("closure_added_rank", 70)])
def test_prepared_native64_rejects_failed_transfer_audit(prepared_native64_fixture, total, field, value):
    fixture = prepared_native64_fixture
    fixture.receipts[total]["transfer_audit"][field] = value
    fixture.seal()
    with pytest.raises(ValueError):
        adapter.load_prepared_die_library(fixture.root / f"packages/BI2_{total:g}.json", prepared_fixture_geometry(total))
    fixture.imported.assert_not_called()


@pytest.mark.parametrize("field,value", [("rank", True), ("stage_count", 5), ("bonding_pitch_um", 6.),
    ("bonding_pad_diameter_um", 3.), ("face_node_shape", [36, 36]), ("all_cu_points_retained", False),
    ("thermal_geometry_validated", True), ("state", "PHYSICALLY_VALIDATED"), ("cell_area_mm2", 0.)])
def test_prepared_native64_rejects_inconsistent_receipt(prepared_native64_fixture, field, value):
    fixture = prepared_native64_fixture
    fixture.receipts[30.][field] = value
    fixture.seal()
    with pytest.raises(ValueError):
        adapter.load_prepared_die_library(fixture.root / "packages/BI2_30.json", prepared_fixture_geometry(30.))
    fixture.imported.assert_not_called()


def test_prepared_native64_requires_both_catalog_admissions(prepared_native64_fixture):
    fixture = prepared_native64_fixture
    catalog = fixture.seal()
    del catalog["packages"]["80"]
    fixture.catalog_path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="both TOTAL30/80"):
        adapter.require_prepared_operator(fixture.root / "packages/BI2_30.json", prepared_fixture_geometry(30.))
    fixture.imported.assert_not_called()


def test_prepared_native64_empty_catalog_needs_no_runtime(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog.json"
    catalog.write_text('{"schema":"yap.prepared_p2_operator_catalog.v1","packages":{}}')
    monkeypatch.setattr(adapter, "_PREPARED_OPERATOR_CATALOG", catalog)
    forbidden = Mock(side_effect=AssertionError("unadmitted packages must not touch runtime inputs"))
    monkeypatch.setattr(adapter, "_input_path", forbidden)
    monkeypatch.setattr(adapter, "_import_mechanics_module", forbidden)
    for total in (30., 80.):
        geometry = prepared_fixture_geometry(total)
        request = adapter.prepared_operator_request(f"BI2_{total:g}.json", geometry)
        assert request["admission_status"] == "UNAVAILABLE_EXACT_P2_OPERATORS_NOT_ADMITTED"
        with pytest.raises(RuntimeError, match="planning only"):
            adapter.load_prepared_die_library(f"BI2_{total:g}.json", geometry)
    forbidden.assert_not_called()


def test_prepared_native64_parent_reuse_is_hash_based(prepared_native64_fixture):
    fixture = prepared_native64_fixture
    fixture.parent["dataset_sha256"]["bi"] = "a" * 64
    fixture.seal()
    with pytest.raises(ValueError, match="byte-identical"):
        adapter.load_prepared_die_library(fixture.root / "packages/BI2_30.json", prepared_fixture_geometry(30.))
    fixture.imported.assert_not_called()


def test_prepared_native64_geometry_content_binding_is_required(prepared_native64_fixture):
    fixture = prepared_native64_fixture
    fixture.receipts[30.]["prepared"]["tsv_si_h17"]["parent_dataset_content_sha256"] = "b" * 64
    fixture.seal()
    with pytest.raises(ValueError, match="bind its dataset content"):
        adapter.load_prepared_die_library(fixture.root / "packages/BI2_30.json", prepared_fixture_geometry(30.))
    fixture.imported.assert_not_called()


@pytest.mark.parametrize("fault", ["cu_card", "si_material", "thickness", "macro_size", "bubble_size"])
def test_prepared_native64_checks_loaded_dataset_material_and_geometry(prepared_native64_fixture, fault):
    fixture = prepared_native64_fixture
    target = fixture.datasets["tsv_si_h17"]
    if fault == "cu_card":
        target.cu_card = PreparedFixtureCard(2.)
    elif fault == "si_material":
        target.reference_constant_materials["si"] = PreparedFixtureCard(2.)
    elif fault == "thickness":
        target.geometry_signature["thickness_mm"] = .037
    else:
        setattr(target, fault, 1)
    with pytest.raises(ValueError):
        adapter.load_prepared_die_library(fixture.root / "packages/BI2_30.json", prepared_fixture_geometry(30.))


def test_prepared_native64_rejects_artifact_escape(prepared_native64_fixture):
    fixture = prepared_native64_fixture
    fixture.packages[30.]["parent_package"] = "../../outside.json"
    fixture.seal()
    with pytest.raises(ValueError, match="escapes"):
        adapter.load_prepared_die_library(fixture.root / "packages/BI2_30.json", prepared_fixture_geometry(30.))
    fixture.imported.assert_not_called()


def test_prepared_native64_rejects_artifact_symlinks(prepared_native64_fixture):
    fixture = prepared_native64_fixture
    alias = fixture.root / "shared/alias_bi.npz"
    alias.symlink_to(fixture.root / "shared/bi.npz")
    fixture.packages[30.]["datasets"]["bi"] = "../shared/alias_bi.npz"
    fixture.seal()
    with pytest.raises(ValueError, match="symlink"):
        adapter.load_prepared_die_library(fixture.root / "packages/BI2_30.json", prepared_fixture_geometry(30.))
    fixture.imported.assert_not_called()


@pytest.mark.parametrize("fault", ["rank", "gauge", "cu_count", "cell_area"])
def test_prepared_native64_checks_returned_library(prepared_native64_fixture, fault):
    fixture = prepared_native64_fixture
    original = fixture.runtime.load_q6_sequential_library.side_effect
    def broken(*args, **kwargs):
        library = original(*args, **kwargs)
        if fault == "rank":
            library.port_rank_per_face = 24
        elif fault == "gauge":
            library.common_port_gauge_vectors = np.ones((64, 1))
        elif fault == "cell_area":
            library.cell_area_mm2 = 4e-6
        else:
            library.operators["tsv_si_h17"].cu_point_count = 1
        return library
    fixture.runtime.load_q6_sequential_library.side_effect = broken
    with pytest.raises(ValueError):
        adapter.load_prepared_die_library(fixture.root / "packages/BI2_30.json", prepared_fixture_geometry(30.))


@pytest.mark.parametrize("field,value", [("schema", "exp12.ofat.geometry_package.v1"), ("name", "BI6_30"),
    ("rank", 24), ("ready", True), ("state", "READY_ENGINEERING_PREVIEW")])
def test_prepared_native64_rejects_other_package_contracts(prepared_native64_fixture, field, value):
    fixture = prepared_native64_fixture
    fixture.packages[30.][field] = value
    fixture.seal()
    with pytest.raises(ValueError, match="exact native64 engineering candidate"):
        adapter.load_prepared_die_library(fixture.root / "packages/BI2_30.json", prepared_fixture_geometry(30.))
    fixture.imported.assert_not_called()


def test_prepared_native64_rejects_h37_substitution(prepared_native64_fixture):
    fixture = prepared_native64_fixture
    for field in ("datasets", "prepared", "dataset_sha256", "prepared_sha256"):
        mapping = fixture.packages[30.][field]
        mapping["tsv_si_h37"] = mapping.pop("tsv_si_h17")
    fixture.seal()
    with pytest.raises(ValueError, match="exactly the target p2 source keys"):
        adapter.load_prepared_die_library(fixture.root / "packages/BI2_30.json", prepared_fixture_geometry(30.))
    fixture.imported.assert_not_called()


def test_prepared_native64_catalog_rejects_duplicate_keys(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog.json"
    catalog.write_text('{"schema":"yap.prepared_p2_operator_catalog.v1","packages":{},"packages":{}}')
    monkeypatch.setattr(adapter, "_PREPARED_OPERATOR_CATALOG", catalog)
    with pytest.raises(ValueError, match="duplicate"):
        adapter.prepared_operator_request("BI2_30.json", prepared_fixture_geometry(30.))


@pytest.mark.parametrize("fault", ["missing", "mismatched", "invalid", "non_scalar", "object"])
def test_prepared_native64_checks_stored_npz_content_digest(prepared_native64_fixture, fault):
    fixture = prepared_native64_fixture
    key = "tsv_si_h17"
    digest = fixture.receipts[30.]["datasets"][key]["dataset_content_sha256"]
    fields = {
        "missing": {},
        "mismatched": dict(content_sha256=np.array("a" * 64)),
        "invalid": dict(content_sha256=np.array("invalid")),
        "non_scalar": dict(content_sha256=np.array([digest])),
        "object": dict(content_sha256=np.array(digest, dtype=object)),
    }
    path = fixture.root / f"datasets/{key}.npz"
    np.savez(path, **fields[fault])
    file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    fixture.packages[30.]["dataset_sha256"][key] = file_digest
    fixture.receipts[30.]["datasets"][key]["sha256"] = file_digest
    fixture.seal()
    with pytest.raises(ValueError, match="dataset content|Object arrays cannot be loaded"):
        adapter.load_prepared_die_library(fixture.root / "packages/BI2_30.json", prepared_fixture_geometry(30.))


@pytest.mark.parametrize("total", [30., 80.])
@pytest.mark.parametrize("field,value", [("observed_rank", 95), ("rank", 95), ("gram_defect", 1e-8),
    ("face_trace", 1e-8), ("relative_periodic_jump", 1e-3), ("singular_value_min", 0.)])
def test_prepared_native64_checks_raw_transfer_separately(prepared_native64_fixture, total, field, value):
    fixture = prepared_native64_fixture
    fixture.receipts[total]["transfer_audit"]["raw_transfer_audit"][field] = value
    fixture.seal()
    with pytest.raises(ValueError):
        adapter.load_prepared_die_library(fixture.root / f"packages/BI2_{total:g}.json", prepared_fixture_geometry(total))
    fixture.imported.assert_not_called()


@pytest.mark.parametrize("relative", ["datasets/tsv_si_h17.npz", "prepared/tsv_si_h17.npz",
    "shared/port_basis.npz", "build_receipts/tsv_si_h17_raw_bubble.npz"])
def test_prepared_native64_missing_artifacts_fail_during_preflight(prepared_native64_fixture, relative):
    fixture = prepared_native64_fixture
    (fixture.root / relative).unlink()
    with pytest.raises(FileNotFoundError):
        adapter.require_prepared_operator(fixture.root / "packages/BI2_30.json", prepared_fixture_geometry(30.))
    fixture.imported.assert_not_called()
