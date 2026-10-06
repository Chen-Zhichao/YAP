"""W2W input/metric contracts; synthetic scoring is not physical validation."""

import ast
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import shutil

import numpy as np
from omegaconf import OmegaConf
import pytest
from scipy.linalg import cho_factor, cho_solve

from W2W.assembly_design import load_design
from W2W.assembly_order import build_plan, evaluate_archive
from W2W.wafer_membrane import disk_membrane_metric, solve_disk_membrane
from test_assembly_order import config, overlay_input, synthetic_archive

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "W2W/input/design_6_4hi_p2"
CONFIG = ROOT / "W2W/configs/design_6_4hi_p2/design_6_4hi_p2.yaml"


def settings():
    cfg = OmegaConf.load(CONFIG).w2w_modeling
    cfg.OVERLAY_SAMPLES = 8
    return cfg


def test_normal_design_geometry_and_repeatable_physical_draws(tmp_path):
    cfg = settings()
    d, p, o, r, v = load_design(cfg, DESIGN, "design_6_4hi_p2")
    assert len(p["routes"]) == 5
    assert len(r) == 3
    assert v["site_count"] == 624
    assert v["critical_pad_count"] == 7500000
    assert o["threshold_um"] == pytest.approx(.19764395315565994)
    assert d["stack"]["initial_bows_um"] == [10., -10., 20., -20.]
    shutil.copytree(DESIGN, tmp_path / "design")
    path = tmp_path / "design/stack.yaml"
    stack = OmegaConf.load(path)
    stack.stack.assembly_orders = [p["routes"][-1]["order"]]
    OmegaConf.save(stack, path)
    _, subset, other, _, _ = load_design(cfg, path.parent, "design_6_4hi_p2")
    assert len(subset["routes"]) == 1
    assert other == o


def test_prepared_wafer_design_has_five_fixed_thickness_routes_and_common_overlay():
    name = "design_6_4hi_p2_no_thinning"
    cfg = OmegaConf.load(ROOT / "W2W/configs" / name / f"{name}.yaml").w2w_modeling
    cfg.OVERLAY_SAMPLES = 8
    _, request, overlay, resolved, provenance = load_design(cfg, ROOT / "W2W/input" / name, name)
    _, _, baseline_overlay, _, _ = load_design(settings(), DESIGN, "design_6_4hi_p2")
    assert overlay == baseline_overlay
    assert request["context"]["body_kind"] == "prepared_wafer"
    assert len(request["routes"]) == 5
    assert provenance["site_count"] == 624
    assert "no thinning" in provenance["config_thickness_scope"]
    for record in resolved.values():
        assert record["ITF_TOP_THICK_um"] == record["ITF_BOT_THICK_um"] == 50.
    for route in request["routes"]:
        assert route["final"]["total_thicknesses_um"] == [50.] * 4
        assert len(route["events"]) == 3
        assert {event["interface_id"] for event in route["events"]} == {
            "1.BACK--2.FRONT", "2.BACK--3.FRONT", "3.BACK--4.FRONT"}
        for event in route["events"]:
            assert "thinning" not in event
            assert event["lower_before_preparation"] == event["lower_free"]
            assert event["preparation"] == "prefabricated_back_BI_no_thinning_or_deposition"
            assert not any("thinning" in operation for operation in event["operations"])
            assert all(thickness == 50. for thickness in event["result"]["total_thicknesses_um"])


@pytest.mark.parametrize("change", [
    lambda c: c["fabrication_layers"].pop(),
    lambda c: c["fabrication_layers"][2].__setitem__("thickness_um", 38.5),
    lambda c: c["fabrication_layers"][3].__setitem__("eigenstrain_per_amplitude", 1.),
    lambda c: c["geometry"].__setitem__("target_total_thickness_um", 51.),
])
def test_prepared_wafer_rejects_inconsistent_geometry_or_source(change):
    from W2W.assembly_order import load_stack_input
    stack = load_stack_input(ROOT / "W2W/input/design_6_4hi_p2_no_thinning/stack.yaml")["stack"]
    change(stack)
    with pytest.raises(ValueError):
        build_plan(stack)


@pytest.mark.parametrize("design_name", ["design_6_4hi_p2", "design_6_4hi_p6_reference"])
def test_implicit_alignment_matches_legacy_r0_and_keeps_overlay_rotation(tmp_path, design_name):
    design = ROOT / "W2W/input" / design_name
    cfg = OmegaConf.load(ROOT / "W2W/configs" / design_name / f"{design_name}.yaml").w2w_modeling
    cfg.OVERLAY_SAMPLES = 8
    topology = OmegaConf.load(design / "generated_stack_config.3dbx")
    assert all("orient" not in node for node in topology.Stack.values())
    _, request, overlay, _, _ = load_design(cfg, design, design_name)
    assert np.any(np.asarray(overlay["rotation_rad"]) != 0.)
    shutil.copytree(design, tmp_path / "design")
    for node in topology.Stack.values():
        node.orient = "R0"
    OmegaConf.save(topology, tmp_path / "design/generated_stack_config.3dbx")
    _, legacy_request, legacy_overlay, _, _ = load_design(cfg, tmp_path / "design", design_name)
    assert legacy_request == request
    assert legacy_overlay == overlay


@pytest.mark.parametrize("orientation", ["R90", "R180", None])
def test_unsupported_nominal_orientation_is_not_silently_ignored(tmp_path, orientation):
    shutil.copytree(DESIGN, tmp_path / "design")
    path = tmp_path / "design/generated_stack_config.3dbx"
    topology = OmegaConf.load(path)
    topology.Stack.Wafer_1.orient = orientation
    OmegaConf.save(topology, path)
    with pytest.raises(ValueError, match="fixed-orientation"):
        load_design(settings(), path.parent, "design_6_4hi_p2")


@pytest.mark.parametrize("field,value", [("WAF_R_um", 100000), ("CRITICAL_PAD_FRACTION", .5),
    ("OVERLAY_SAMPLES", True), ("RANDOM_SEED", -1), ("RANDOM_MISALIGNMENT_MEAN_um", 1)])
def test_reject_conflicting_config(field, value):
    cfg = settings()
    cfg[field] = value
    with pytest.raises(ValueError):
        load_design(cfg, DESIGN, "design_6_4hi_p2")


@pytest.mark.parametrize("file,change", [
    ("Wafer_1.3dbf", lambda c: setattr(c.Chiplet_Grid, "pitch", 1.)),
    ("generated_chiplet_definitions.3dbv", lambda c: setattr(c.ChipletDef.Wafer_1, "thickness", 50.)),
    ("generated_stack_config.3dbx", lambda c: setattr(c.Stack.Wafer_1, "orient", "MZ_MX")),
    ("compact_layout.yaml", lambda c: setattr(c, "rows", 10000)),
])
def test_reject_conflicting_design(tmp_path, file, change):
    shutil.copytree(DESIGN, tmp_path / "design")
    path = tmp_path / "design" / file
    data = OmegaConf.load(path)
    change(data)
    OmegaConf.save(data, path)
    with pytest.raises(ValueError):
        load_design(settings(), path.parent, "design_6_4hi_p2")


def test_disk_analytic_sphere_zero_and_reflection():
    nu, k, radius = .28, 1.e-5, 150.
    a = np.array([[1., nu, 0.], [nu, 1., 0.], [0., 0., (1-nu)/2]])
    metric, audit = disk_membrane_metric(a, [k, k, 0.], radius)
    np.testing.assert_allclose(metric, np.eye(2) * (nu+3)*k*k*radius**2/48, atol=1e-20)
    np.testing.assert_array_equal(solve_disk_membrane(a, [0., 0., 0.]).affine_metric, np.zeros((2, 2)))
    assert audit["physical_validation"] is False
    curve = [k, 2*k, .7*k]
    original = solve_disk_membrane(a, curve).affine_metric
    reflected = solve_disk_membrane(a, [k, 2*k, -.7*k]).affine_metric
    s = np.diag([1., -1.])
    np.testing.assert_allclose(reflected, s @ original @ s, atol=1e-20)
    assert abs(original[0, 1]) > 1e-10


def test_disk_matches_original_reference_algebra():
    source = ROOT.parent / "Ansys_Verify/12_stack_model_validation/exp12_w2w_window_overlay_20260914.py"
    if not source.exists():
        pytest.skip("external reference reference source not installed")
    names = {"_finite", "_membrane_matrix", "_moment", "_space", "DiskMembraneSolution", "solve_disk_membrane"}
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    namespace = dict(np=np, math=math, dataclass=dataclass, lru_cache=lru_cache,
                     cho_factor=cho_factor, cho_solve=cho_solve, RADIUS_MM=150.)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    a = np.array([[2., .3, .2], [.3, 1.5, -.1], [.2, -.1, .6]])
    for curve in ([1.e-5, 2.e-5, 3.e-6], [0., 0., 1.e-5], [0., 0., 0.]):
        expected = namespace["solve_disk_membrane"](a, curve)
        actual = solve_disk_membrane(a, curve)
        np.testing.assert_array_equal(actual.affine_metric, expected.affine_metric)
        np.testing.assert_array_equal(actual.coefficients, expected.coefficients)


def tensor_fixture():
    plan = build_plan(config())
    archive = synthetic_archive(plan)
    archive["native_configuration"] = {"metric": {"model": "disk"}}
    for route in archive["routes"]:
        for event in route["events"]:
            for role in ("lower", "upper"):
                event[role]["chi"] = None
                event[role]["affine_face_strain"] = [[0., 0.], [0., 0.]]
    return plan, archive


def test_placement_rotation_remains_in_overlay_scoring():
    plan, archive = tensor_fixture()
    overlay = overlay_input()
    overlay.update(coordinates_um=[[[100000., 0.]]], site_mask=[True], residual_sigma_um=0.)
    result = evaluate_archive(plan, archive, overlay)
    assert all(route["stack_overlay_yield"] == 1. for route in result["routes"])
    overlay["rotation_rad"] = [[1.e-5] * 3]
    rotated = evaluate_archive(plan, archive, overlay)
    assert all(route["stack_overlay_yield"] == 0. for route in rotated["routes"])
    assert all(interface["max_bow_overlay_nm"] == 0.
               for route in rotated["routes"] for interface in route["interfaces"])


def test_tensor_scorer_retains_shear_and_correction():
    plan, archive = tensor_fixture()
    overlay = overlay_input()
    overlay.update(coordinates_um=[[[100000., 0.]]], site_mask=[True], residual_sigma_um=0.)
    for route in archive["routes"]:
        for event in route["events"]:
            event["upper"]["affine_face_strain"] = [[0., 1.e-5], [1.e-5, 0.]]
    result = evaluate_archive(plan, archive, overlay)
    assert all(r["stack_overlay_yield"] == 0. for r in result["routes"])
    interface = result["routes"][0]["interfaces"][0]
    np.testing.assert_allclose(interface["magnification_ppm"], [[0., 10.], [10., 0.]])
    assert interface["max_bow_overlay_nm"] == pytest.approx(1000.)
    original_non_bow = interface["non_bow_overlay_sha256"]
    overlay["magnification_correction_fraction"] = .89
    corrected = evaluate_archive(plan, archive, overlay)["routes"][0]["interfaces"][0]
    assert corrected["max_residual_bow_overlay_nm"] == pytest.approx(110.)
    assert corrected["non_bow_overlay_sha256"] == original_non_bow
    np.testing.assert_allclose(corrected["residual_magnification_ppm"], [[0., 1.1], [1.1, 0.]])
    overlay["magnification_correction_fraction"] = 1.
    result = evaluate_archive(plan, archive, overlay)
    assert all(r["stack_overlay_yield"] == 1. for r in result["routes"])


@pytest.mark.parametrize("matrix", [[[0., 0.]], [[float("nan"), 0.], [0., 0.]], None])
def test_tensor_scorer_rejects_invalid_matrix(matrix):
    plan, archive = tensor_fixture()
    archive["routes"][0]["events"][0]["lower"]["affine_face_strain"] = matrix
    with pytest.raises(ValueError):
        evaluate_archive(plan, archive, overlay_input())


def test_normal_runner_complete_disk_path(tmp_path, monkeypatch):
    from W2W.assembly_design import run_design
    import W2W.assembly_operator_library as loaders
    from test_assembly_mechanics import tiny_library
    import json
    monkeypatch.setattr(loaders, "load_assembly_library",
                        lambda *_, **__: (tiny_library(), {"fixture": "nonphysical_unit_test"}))
    run = run_design(settings(), DESIGN, "design_6_4hi_p2", output_dir=tmp_path,
                     config_path=CONFIG)
    result = json.loads((run / "RESULT.json").read_text())
    assert len(result["routes"]) == 5
    assert result["physical_readiness"] is False
    assert len(list(run.glob("route_*_mechanics.json"))) == 5
    assert len((run / "SUMMARY.csv").read_text().splitlines()) == 16
    assert (run / "input_snapshot/stack.yaml").exists()
    import csv
    rows = list(csv.DictReader((run / "SUMMARY.csv").open()))
    assert "m_xy_ppm" in rows[0] and "max_residual_bow_overlay_nm" in rows[0]
    for physical in ("1.BACK--2.FRONT", "2.BACK--3.FRONT", "3.BACK--4.FRONT"):
        assert len({row["non_bow_overlay_sha256"] for row in rows if row["interface"] == physical}) == 1


def test_magnification_maximum_uses_only_selected_sites_and_critical_pads():
    plan, archive = tensor_fixture()
    overlay = overlay_input()
    overlay.update(coordinates_um=[[[1000., 0.], [140000., 0.]], [[140000., 0.], [145000., 0.]]],
                   site_mask=[True, False], critical_pad_mask=[True, False])
    for route in archive["routes"]:
        for event in route["events"]:
            event["upper"]["affine_face_strain"] = [[1e-5, 0.], [0., -2e-5]]
    result = evaluate_archive(plan, archive, overlay)
    assert all(i["max_bow_overlay_nm"] == pytest.approx(10.)
               for route in result["routes"] for i in route["interfaces"])
