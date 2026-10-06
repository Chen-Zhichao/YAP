"""Small native algebra fixtures, not physical validation of the new route."""

from dataclasses import replace
from pathlib import Path
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from W2W import assembly_operator_library as adapter

if not (Path(__file__).resolve().parents[2] / "Ansys_Verify/12_stack_model_validation").is_dir():
    pytest.skip("sibling native mechanics repository is not installed", allow_module_level=True)


def geometry():
    return {**adapter._GEOMETRY_UM, "diameter_mm": 300.}


def native(name):
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
