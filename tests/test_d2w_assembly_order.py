"""Prepared-die tree contracts; synthetic fixtures are not physics validation."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf
import pytest

from D2W.assembly_order import build_plan, load_stack_input, load_design, run_design, evaluate_archive
from W2W.assembly_order import archive_template, content_id, DIE_FRAME

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "D2W/input/design_6_4hi_p2"
CONFIG = ROOT / "D2W/configs/design_6_4hi_p2/design_6_4hi_p2.yaml"


def settings():
    cfg = OmegaConf.load(CONFIG).d2w_modeling
    cfg.OVERLAY_SAMPLES = 16
    return cfg


def plan(order="all", bows=(10., -10., 20., -20.)):
    config = load_stack_input(DESIGN / "stack.yaml")["stack"]
    config.update(assembly_orders=order, initial_bows_um=list(bows))
    return build_plan(config)


def synthetic_archive(request):
    archive = archive_template(request)
    archive["status"] = "COMPLETE"
    archive["provenance"] = dict(solver_id="nonphysical_unit_test", implementation_sha256="a" * 64,
        operator_set_sha256="b" * 64, source_transfer_audit="fixture", material_clock_audit="fixture",
        physics_qualification="NONPHYSICAL_TEST_ONLY")
    archive["native_configuration"] = {"metric": {"model": "square_surrogate"}}
    for route, target in zip(request["routes"], archive["routes"]):
        for event, record in zip(route["events"], target["events"]):
            for role in ("lower", "upper"):
                state = event[role + "_free"]
                ids = state["die_ids"]
                bow = sum(request["context"]["initial_bows_um"][i-1] for i in ids)
                record[role] = dict(state_id=state["state_id"], die_ids=ids,
                    total_thicknesses_um=state["total_thicknesses_um"], frame_id=DIE_FRAME,
                    bow_convention="center_minus_mid_edge", bow_um=[bow, bow],
                    face_lever_um=(1 if role == "lower" else -1) * sum(state["total_thicknesses_um"]) / 2,
                    chi=[1., 1.], face_lever_source="synthetic_fixture", face="BACK" if role == "lower" else "FRONT")
            record["checkpoints"] = {event[k]["state_id"]: content_id(event[k]["state_id"]) for k in
                ("lower_before_preparation", "lower_free", "upper_free", "result")}
            record["transfer_audit"] = "fixture"
    return archive


def test_five_trees_immutable_50um_no_thinning_and_explicit_ids():
    request = plan()
    assert len(request["routes"]) == 5
    assert request["schema"] == "yap.d2w_assembly_order_plan.v1"
    assert "dies" in request["initialization"] and "wafers" not in request["initialization"]
    for route in request["routes"]:
        assert route["final"]["total_thicknesses_um"] == [50.] * 4
        assert route["final"]["die_ids"] == [1, 2, 3, 4]
        assert len(route["events"]) == 3
        for event in route["events"]:
            assert event["lower_free"] == event["lower_before_preparation"]
            assert "thinning" not in event
            assert not any("thinning" in op for op in event["operations"])


@pytest.mark.parametrize("order", ["(1,2)", "((1,3),(2,4))", "((1,2),(3,3))", ["(((1,2),3),4)"] * 2])
def test_invalid_trees(order):
    with pytest.raises(ValueError):
        plan(order)


def test_offline_substacks_are_not_silently_pure_d2w():
    cfg = load_stack_input(DESIGN / "stack.yaml")["stack"]
    cfg["allow_offline_substacks"] = False
    with pytest.raises(ValueError, match="offline D2D"):
        build_plan(cfg)
    cfg["assembly_orders"] = "(((1,2),3),4)"
    assert len(build_plan(cfg)["routes"]) == 1


def test_die_local_geometry_and_common_physical_draws(tmp_path):
    cfg = settings()
    _, request, overlay, resolved, provenance = load_design(cfg, DESIGN, "design_6_4hi_p2")
    assert provenance["site_count"] == 1
    assert provenance["critical_pad_count"] == 7500000
    assert len(resolved) == 3
    assert overlay["threshold_um"] == pytest.approx(.19764395315565994)
    assert np.max(np.abs(overlay["coordinates_um"])) < 5000
    assert "WAF_R_um" not in provenance["effective_config"]
    shutil.copytree(DESIGN, tmp_path / "design")
    path = tmp_path / "design/stack.yaml"
    data = OmegaConf.load(path)
    data.stack.assembly_orders = [request["routes"][-1]["order"]]
    OmegaConf.save(data, path)
    _, selected, other, _, _ = load_design(cfg, path.parent, "design_6_4hi_p2")
    assert len(selected["routes"]) == 1
    assert other == overlay


def test_implicit_alignment_matches_legacy_r0_and_keeps_overlay_rotation(tmp_path):
    cfg = settings()
    topology = OmegaConf.load(DESIGN / "generated_stack_config.3dbx")
    assert all("orient" not in node for node in topology.Stack.values())
    _, request, overlay, _, _ = load_design(cfg, DESIGN, "design_6_4hi_p2")
    assert np.any(np.asarray(overlay["rotation_rad"]) != 0.)
    shutil.copytree(DESIGN, tmp_path / "design")
    for node in topology.Stack.values():
        node.orient = "R0"
    OmegaConf.save(topology, tmp_path / "design/generated_stack_config.3dbx")
    _, legacy_request, legacy_overlay, _, _ = load_design(cfg, tmp_path / "design", "design_6_4hi_p2")
    assert legacy_request == request
    assert legacy_overlay == overlay


@pytest.mark.parametrize("orientation", ["R90", "R180", None])
def test_unsupported_nominal_orientation_is_not_silently_ignored(tmp_path, orientation):
    shutil.copytree(DESIGN, tmp_path / "design")
    path = tmp_path / "design/generated_stack_config.3dbx"
    topology = OmegaConf.load(path)
    topology.Stack.Die_2.orient = orientation
    OmegaConf.save(topology, path)
    with pytest.raises(ValueError, match="fixed-orientation"):
        load_design(settings(), path.parent, "design_6_4hi_p2")


@pytest.mark.parametrize("field,value", [("ANNEAL_MODE", "batch"), ("CRITICAL_PAD_FRACTION", .5),
    ("OVERLAY_SAMPLES", True), ("RANDOM_SEED", -1), ("RANDOM_MISALIGNMENT_MEAN_um", 1),
    ("MAGNIFICATION_CORRECTION", 1.1)])
def test_conflicting_config_rejected(field, value):
    cfg = settings()
    cfg[field] = value
    with pytest.raises(ValueError):
        load_design(cfg, DESIGN, "design_6_4hi_p2")


@pytest.mark.parametrize("file,change", [
    ("Die_1.3dbf", lambda c: setattr(c.Chiplet_Grid, "pitch", 6.)),
    ("generated_chiplet_definitions.3dbv", lambda c: setattr(c.ChipletDef.Die_2, "thickness", 775.)),
    ("generated_stack_config.3dbx", lambda c: setattr(c.Stack.Die_2, "orient", "MZ_MX")),
    ("generated_stack_config.3dbx", lambda c: setattr(c.Stack.Die_2, "loc", [100., 0.])),
    ("generated_stack_config.3dbx", lambda c: setattr(c.Stack.Die_2, "root", True)),
    ("generated_stack_config.3dbx", lambda c: setattr(c.Header, "unit", "mm")),
    ("generated_chiplet_definitions.3dbv", lambda c: setattr(c.Header, "unit", "mm")),
    ("compact_layout.yaml", lambda c: setattr(c, "rows", 10000)),
    ("stack.yaml", lambda c: setattr(c.stack.geometry, "length_mm", 11.)),
])
def test_conflicting_design_rejected(tmp_path, file, change):
    shutil.copytree(DESIGN, tmp_path / "design")
    path = tmp_path / "design" / file
    data = OmegaConf.load(path)
    change(data)
    OmegaConf.save(data, path)
    with pytest.raises(ValueError):
        load_design(settings(), path.parent, "design_6_4hi_p2")


def test_scoring_parity_local_frame_and_non_bow_fingerprints():
    _, request, overlay, _, _ = load_design(settings(), DESIGN, "design_6_4hi_p2")
    archive = synthetic_archive(request)
    result = evaluate_archive(request, archive, overlay)
    assert result["schema"] == "yap.d2w_assembly_order_overlay.v1"
    assert "die_centered" in result["magnification_report"]["matrix_convention"]
    assert result["other_yield_mechanisms_included"] is False
    first = result["routes"][0]["interfaces"][0]
    expected = (-2 * (-25 * -20 - 25 * 20) + (-20)**2 - 20**2) / 5000**2
    np.testing.assert_allclose(first["magnification_ppm"], np.eye(2) * expected * 1e6)
    for physical in ("1.BACK--2.FRONT", "2.BACK--3.FRONT", "3.BACK--4.FRONT"):
        assert len({i["non_bow_overlay_sha256"] for r in result["routes"] for i in r["interfaces"] if i["interface_id"] == physical}) == 1
    changed = deepcopy(overlay)
    changed["coordinates_um"] = [[[100000., 0.]] * 4]
    with pytest.raises(ValueError, match="die-local"):
        evaluate_archive(request, archive, changed)
    archive["schema"] = "yap.w2w_assembly_order_mechanics.v1"
    with pytest.raises(ValueError, match="complete mechanics"):
        evaluate_archive(request, archive, overlay)


def test_plan_runner_and_complete_summary(tmp_path, monkeypatch):
    import W2W.assembly_mechanics as mechanics
    import W2W.assembly_operator_library as loaders
    monkeypatch.setattr(loaders, "load_prepared_die_library", lambda *_, **__: (object(), {"fixture": "test_only"}))

    def fake_native(request, *_args, route_complete=None, **_kwargs):
        archive = synthetic_archive(request)
        for route in archive["routes"]:
            route_complete(archive, route)
        return archive

    monkeypatch.setattr(mechanics, "run_native", fake_native)
    out = run_design(settings(), DESIGN, "design_6_4hi_p2", output_dir=tmp_path, config_path=CONFIG)
    result = json.loads((out / "RESULT.json").read_text())
    assert len(result["routes"]) == 5 and not result["physical_readiness"]
    assert len(list(out.glob("route_*_mechanics.json"))) == 5
    with (out / "SUMMARY.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 15 and "max_residual_bow_overlay_nm" in rows[0]
    assert (out / "input_snapshot/stack.yaml").exists()
    assert (out / "RUNTIME_SOURCES.json").exists()


def test_plan_only_rejects_supported_layout_with_unsupported_operator_thickness(tmp_path):
    shutil.copytree(DESIGN, tmp_path / "design")
    path = tmp_path / "design/stack.yaml"
    data = OmegaConf.load(path)
    data.stack.geometry.total_thickness_um = 60.
    data.stack.fabrication_layers[2].thickness_um = 47.
    OmegaConf.save(data, path)
    with pytest.raises(ValueError, match="total_thickness_um"):
        load_design(settings(), path.parent, "design_6_4hi_p2")


def test_cli_plan_only_dispatch_without_loading_operators(tmp_path):
    proc = subprocess.run([sys.executable, str(ROOT / "D2W/calculator_main.py"),
        "--config", str(CONFIG), "--mode", "d2w_modeling", "--ds_name", "design_6_4hi_p2",
        "--ds_dir", str(DESIGN), "--plan-only", "--output-dir", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert proc.returncode == 0, proc.stderr
    files = list(tmp_path.glob("design_6_4hi_p2/*/PLAN.json"))
    assert len(files) == 1
    assert len(json.loads(files[0].read_text())["routes"]) == 5
    assert not list(tmp_path.glob("**/MECHANICS.json"))


def test_native_prepared_leaves_and_two_child_cu_history():
    from test_assembly_mechanics import tiny_library
    from W2W.assembly_mechanics import AssemblyExecutor

    request = plan("((1,2),(3,4))", bows=(0., 0., 0., 0.))
    engine = AssemblyExecutor(request, tiny_library(with_cu=True), temporal_profile="distilled14")
    pairs = []
    for first in (1, 3):
        lower, upper = engine.leaf(first), engine.leaf(first + 1)
        for body in (lower, upper):
            assert [p.role for p in body.placements] == ["front_bi", "beol", "retained_si", "back_bi"]
            assert body.placements[-1].z_top_mm - body.placements[0].z_bottom_mm == pytest.approx(.05)
            assert body.placements[-1].thickness_orientation == -1
            before = engine.checkpoint(body)
            same, audit = engine.prepare_back(body)
            assert same is body and engine.checkpoint(body) == before
            assert "removal" not in audit and "deposition" not in audit
            for layer in body.state.layers.values():
                layer.cu_state.equivalent_plastic_strain[:] = first * .001
        joined, _ = engine.join(lower, upper, "pair")
        joined, _ = engine.anneal(joined)
        pairs.append(joined)
    before = [engine.checkpoint(v) for v in pairs]
    final, audit = engine.join(*pairs, "final")
    assert before == [engine.checkpoint(v) for v in pairs]
    for body in pairs:
        for name, old in body.state.layers.items():
            new = final.state.layers[name]
            for field in ("plastic_strain", "creep_strain", "equivalent_plastic_strain", "equivalent_creep_strain"):
                np.testing.assert_array_equal(getattr(new.cu_state, field), getattr(old.cu_state, field))
                assert not np.shares_memory(getattr(new.cu_state, field), getattr(old.cu_state, field))
            assert final.ages_s[name] == 5200.
    assert audit["held_flat_local_coordinate_drift"] < 1e-8


@pytest.mark.parametrize("index", range(5))
def test_all_five_native_prepared_routes_preserve_ages(index):
    from test_assembly_mechanics import tiny_library
    from W2W.assembly_mechanics import AssemblyExecutor
    request = plan(bows=(0., 0., 0., 0.))
    engine = AssemblyExecutor(request, tiny_library(), temporal_profile="distilled14")
    route = request["routes"][index]
    target = archive_template(request)["routes"][index]
    engine.run_route(route, target)
    body = engine.bodies[route["final"]["state_id"]]
    for die, cycles in enumerate(route["final"]["thermal_cycle_counts"], 1):
        assert body.ages_s[f"die{die:02d}_back_bi"] == 300. + 4900. * cycles
        assert body.ages_s[f"die{die:02d}_front_bi"] == 300. + 4900. * cycles
    assert len(target["events"]) == 3
    assert all("wafer_ids" not in e["lower"] for e in target["events"])
    for state in engine.bodies.values():
        for die_index in {p.die_index for p in state.placements}:
            layers = [p for p in state.placements if p.die_index == die_index]
            assert [p.role for p in layers] == ["front_bi", "beol", "retained_si", "back_bi"]
            assert layers[-1].z_top_mm - layers[0].z_bottom_mm == pytest.approx(.05)
    for event in target["events"]:
        preparation = event["audit"]["preparation"]
        assert preparation["policy"] == "prefabricated_back_BI_no_thinning_or_deposition"
        assert "removal" not in preparation and "deposition" not in preparation


def test_full_square_metric_eigenstrain_basis_and_gauge_invariance():
    from D2W.die_membrane import _eigenstrain_basis, square_membrane_metric
    x, y = np.array([-.8, .4, 1.]), np.array([.3, -.7, .5])
    a, b, c = .2, -.4, .13
    weights = [a*a, b*b, c*c, a*b, a*c, b*c]
    gx, gy = a*x+c*y, c*x+b*y
    expected = np.stack((.5*gx*gx, .5*gy*gy, gx*gy), axis=1)
    np.testing.assert_allclose(_eigenstrain_basis(x, y) @ weights, expected, atol=1e-16)
    stiffness = np.array([[2., .3, .2], [.3, 1.5, -.1], [.2, -.1, .6]])
    curve = [1e-4, 2e-4, .7e-4]
    original, audit = square_membrane_metric(stiffness, curve, 5., elements_per_half_side=8)
    alternative, _ = square_membrane_metric(stiffness, curve, 5., elements_per_half_side=8, gauge_axis=1)
    np.testing.assert_allclose(original, alternative, atol=1e-16, rtol=1e-9)
    assert max(audit["equilibrium_relative_residuals"]) < 1e-8
    assert audit["actual_A_and_twist_retained"] and not audit["physical_validation"]
    np.testing.assert_array_equal(square_membrane_metric(stiffness, [0., 0., 0.], 5., elements_per_half_side=8)[0], np.zeros((2, 2)))
    np.testing.assert_allclose(square_membrane_metric(stiffness*123, curve, 5., elements_per_half_side=8)[0], original, atol=1e-16)
    np.testing.assert_allclose(square_membrane_metric(stiffness, np.asarray(curve)*-2, 5., elements_per_half_side=8)[0], original*4, atol=1e-16)


def test_full_square_metric_reflection_retains_actual_normal_shear_coupling():
    from D2W.die_membrane import square_membrane_metric
    a = np.array([[2., .3, .2], [.3, 1.5, -.1], [.2, -.1, .6]])
    curve = np.array([1e-4, 2e-4, .7e-4])
    metric, _ = square_membrane_metric(a, curve, 5., elements_per_half_side=8)
    engineering = np.diag([1., 1., -1.])
    reflection = np.diag([1., -1.])
    mirrored, _ = square_membrane_metric(engineering @ a @ engineering, engineering @ curve, 5., elements_per_half_side=8)
    np.testing.assert_allclose(mirrored, reflection @ metric @ reflection, atol=1e-16, rtol=1e-9)
    projected = a.copy()
    projected[:2, 2] = projected[2, :2] = 0.
    without_coupling, _ = square_membrane_metric(projected, curve, 5., elements_per_half_side=8)
    assert np.max(np.abs(metric-without_coupling)) > 1e-10


def test_full_square_fit_preserves_reference_sphere_coefficient_and_refinement():
    from D2W.die_membrane import square_membrane_metric
    from W2W.assembly_mechanics import _reference_square_metric
    stiffness = np.array([[1., .28, 0.], [.28, 1., 0.], [0., 0., .36]])
    values = []
    for n in (8, 16, 32):
        metric, _ = square_membrane_metric(stiffness, [2., 2., 0.], .5, elements_per_half_side=n)
        old = _reference_square_metric().square_metric_coefficient(stiffness, elements_per_half_side=n)
        np.testing.assert_allclose(metric, np.eye(2) * old / 4, atol=1e-12, rtol=1e-10)
        values.append(metric[0, 0])
    assert abs(values[2]-values[1]) < abs(values[1]-values[0])


def test_reference_four_quadrant_fit_discards_only_rigid_motion():
    from D2W.die_membrane import _reference_affine_fit
    axis = np.linspace(-1., 1., 17)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    field = np.stack((x + .2*x*y*y, .4*y + .1*x*x*y), axis=-1).reshape(-1, 2, 1)
    expected = _reference_affine_fit(field, axis, .98)
    rigid = np.stack((-.3*y + 4., .3*x - 7.), axis=-1).reshape(-1, 2, 1)
    np.testing.assert_allclose(_reference_affine_fit(field+rigid, axis, .98), expected, atol=1e-14)


def test_full_square_metric_swap_covariance():
    from D2W.die_membrane import square_membrane_metric
    a = np.array([[2., .3, .2], [.3, 1.5, -.1], [.2, -.1, .6]])
    curve = np.array([1e-4, 2e-4, .7e-4])
    swap = np.array([[0., 1.], [1., 0.]])
    engineering = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
    original, _ = square_membrane_metric(a, curve, 5., elements_per_half_side=8)
    transformed, _ = square_membrane_metric(engineering @ a @ engineering, engineering @ curve,
                                           5., elements_per_half_side=8)
    np.testing.assert_allclose(transformed, swap @ original @ swap, atol=1e-16, rtol=1e-9)


@pytest.mark.parametrize("matrix", [-np.eye(3), np.zeros((3, 3)), [[1., 0.], [0., 1.]],
                                  [[1., .1, 0.], [0., 1., 0.], [0., 0., 1.]]])
def test_full_square_metric_rejects_bad_stiffness(matrix):
    from D2W.die_membrane import square_membrane_metric
    with pytest.raises(ValueError):
        square_membrane_metric(matrix, [1e-5, 1e-5, 0.], 5., elements_per_half_side=8)


def test_full_affine_scorer_retains_die_shear():
    _, request, overlay, _, _ = load_design(settings(), DESIGN, "design_6_4hi_p2")
    archive = synthetic_archive(request)
    archive["native_configuration"]["metric"] = {"model": "square_affine"}
    for route in archive["routes"]:
        for event in route["events"]:
            for role in ("lower", "upper"):
                event[role]["chi"] = None
                event[role]["affine_face_strain"] = [[0., 1e-4 if role == "upper" else 0.], [1e-4 if role == "upper" else 0., 0.]]
    overlay["magnification_correction_fraction"] = 0.
    result = evaluate_archive(request, archive, overlay)
    np.testing.assert_allclose(result["routes"][0]["interfaces"][0]["magnification_ppm"], [[0., 100.], [100., 0.]])
    assert all(r["stack_overlay_yield"] < 1. for r in result["routes"])


def test_native_full_affine_roundtrip_with_non_symmetric_fixture():
    from test_assembly_mechanics import tiny_library
    from W2W.assembly_mechanics import run_native
    _, request, overlay, _, _ = load_design(settings(), DESIGN, "design_6_4hi_p2")
    config = load_stack_input(DESIGN / "stack.yaml")["stack"]
    config["assembly_orders"] = "((1,2),(3,4))"
    request = build_plan(config)
    archive = run_native(request, tiny_library(), {"fixture": "nonphysical_test_only"},
        temporal_profile="distilled14", metric={"model": "square_affine"})
    result = evaluate_archive(request, archive, overlay)
    assert len(result["routes"]) == 1
    assert archive["native_configuration"]["metric_implementation_sha256"]
    for event in archive["routes"][0]["events"]:
        assert event["lower"]["chi"] is None
        assert "square_membrane_audit" in event["raw_prebond"]["lower"]
