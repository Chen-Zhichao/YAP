"""F2B process and native-kernel tests; tiny operators are not physical validation."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from W2W import assembly_order as order
from W2W.assembly_mechanics import AssemblyExecutor, run_native
from W2W import assembly_operator_library as operators


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "W2W/input/f2b_4hi50_p24_preview/stack.yaml"


def config(count=4, selection="all"):
    result = order.load_stack_input(INPUT)["stack"]
    result.update(initial_bows_um=[10. * (-1)**i for i in range(count)], assembly_orders=selection)
    return result


def library(with_cu=False):
    from test_assembly_mechanics import tiny_library
    base = tiny_library(with_cu)
    return replace(base, operators={**base.operators, "outer_si": base.operators["bulk_si"]},
        thickness_mm={**base.thickness_mm, "bulk_si": .725, "outer_si": .0015})


def test_public_default_all_five_are_f2b_and_finish_all_wafers():
    source = config()
    before = deepcopy(source)
    plan = order.build_plan(source)
    assert source == before
    assert plan["schema"] == order.F2B_PLAN_SCHEMA
    assert plan["context"]["orientation"] == "lower_BACK_to_upper_FRONT"
    assert plan["physical_readiness"] is False
    assert len(plan["routes"]) == 5
    order.require_executable_plan(plan)
    for route in plan["routes"]:
        assert route["assembled"]["total_thicknesses_um"] == [50., 50., 50., 775.]
        assert route["final"]["total_thicknesses_um"] == [50.] * 4
        assert route["finalization"]["retained_original_pure_si_um"] == 1.5
        assert route["finalization"]["new_back_bi_um"] == 0.
        assert {event["physical_interface_index"] for event in route["events"]} == {0, 1, 2}
        for event in route["events"]:
            assert event["bond_mode"] == "F2B"
            assert event["mating_faces"] == dict(lower="BACK", upper="FRONT")
            assert event["lower_before_preparation"]["total_thicknesses_um"][-1] == 775.
            assert event["lower_free"]["total_thicknesses_um"][-1] == 50.
            assert event["thinning"]["retained_si_um"] == 37.


@pytest.mark.parametrize("count", [2, 3, 5, 8, 32])
def test_selected_general_n_tree(count):
    tree = count
    for wafer in range(count - 1, 0, -1):
        tree = (wafer, tree)
    plan = order.build_plan(config(count, order.tree_text(tree)))
    route = plan["routes"][0]
    assert len(route["events"]) == count - 1
    assert route["final"]["total_thicknesses_um"] == [50.] * count
    assert len(order.all_orders(8)) == 429


def test_all_429_eight_wafer_graphs_are_f2b_and_complete():
    plan = order.build_plan(config(8))
    assert len(plan["routes"]) == 429
    for route in plan["routes"]:
        assert len(route["events"]) == 7
        assert {event["physical_interface_index"] for event in route["events"]} == set(range(7))
        assert all(event["bond_mode"] == "F2B" for event in route["events"])
        assert route["final"]["total_thicknesses_um"] == [50.] * 8


def test_eight_wafer_native_balanced_route_executes():
    plan = order.build_plan(config(8, "(((1,2),(3,4)),((5,6),(7,8)))"))
    run = AssemblyExecutor(plan, library(), temporal_profile="distilled14")
    output = order.archive_template(plan)["routes"][0]
    run.run_route(plan["routes"][0], output)
    assert output["final"]["total_thicknesses_um"] == [50.] * 8
    assert len(output["final"]["layers"]) == 32


def test_target_default_and_775_birth_enforced():
    source = config()
    del source["geometry"]["target_total_thickness_um"]
    assert order.build_plan(source)["context"]["geometry"]["target_total_thickness_um"] == 50.
    source["geometry"]["initial_total_thickness_um"] = 774.
    source["fabrication_layers"][-1]["thickness_um"] = 762.5
    with pytest.raises(ValueError, match="born at 775"):
        order.build_plan(source)


def test_resigned_face_tampering_is_rejected_and_historical_contract_stays_blocked():
    plan = order.build_plan(config())
    route = plan["routes"][0]
    route["events"][0]["bond_mode"] = "F2F"
    route["request_id"] = order.content_id({k: v for k, v in route.items() if k != "request_id"})
    plan["plan_id"] = order.content_id({k: v for k, v in plan.items() if k != "plan_id"})
    with pytest.raises(ValueError, match="modified F2B"):
        order.require_executable_plan(plan)
    historical = order._build_mixed_plan(config())
    order.validate_mixed_plan(historical)
    with pytest.raises(ValueError, match="plan-only"):
        order.require_executable_plan(historical)
    assert order._build_legacy_plan(config())["routes"][0]["final"]["total_thicknesses_um"][-1] == 775.


def test_split_cap_loader_is_explicit_and_does_not_admit_component_native64(monkeypatch):
    geometry = config()["geometry"]
    seen = []
    monkeypatch.setattr(operators, "_load_reference_p2_preview_library",
        lambda package, geometry, **options: seen.append(options) or (None, None))
    operators.load_f2b_assembly_library("resource:compact_p2/packages/BI2_50.json", geometry, frozen_preview=True)
    assert seen == [dict(split_cap=True)]
    for package, preview in (("component:BI2_SI37", False), ("BI2_50.json", False), ("BI6_50.json", True)):
        with pytest.raises(ValueError, match="native64 cap is not yet admitted"):
            operators.load_f2b_assembly_library(package, geometry, frozen_preview=preview)


def test_both_pure_si_slices_exist_from_birth_and_final_slice_is_not_redeposited():
    run = AssemblyExecutor(order.build_plan(config()), library(True), temporal_profile="distilled14")
    born = run.leaf(4)
    assert [p.role for p in born.placements] == [
        "front_bi", "beol", "retained_si", "outer_retained_si", "removable_bulk_si"]
    assert born.placements[0].z_bottom_mm == pytest.approx(.15)
    assert born.placements[-1].z_top_mm - born.placements[0].z_bottom_mm == pytest.approx(.775)
    for layer in born.state.layers.values():
        layer.cu_state.equivalent_plastic_strain[:] = .002
    for name in born.ages_s:
        born.ages_s[name] = 5200.
    before = run.checkpoint(born)
    finished, audit = run.finish_outer_back(born)
    assert run.checkpoint(born) == before
    assert finished.placements[-1] is born.placements[-2]
    assert finished.placements[-1].role == "outer_retained_si"
    assert audit["silicon_deposition"] is False
    assert audit["removal"]["removed_names"] == ["wafer04_removable_bulk_si"]
    assert all(age == 5200. for age in finished.ages_s.values())
    for name, layer in finished.state.layers.items():
        old = born.state.layers[name]
        np.testing.assert_array_equal(layer.birth_macro_coordinates, old.birth_macro_coordinates)
        for field in ("plastic_strain", "creep_strain", "equivalent_plastic_strain", "equivalent_creep_strain"):
            np.testing.assert_array_equal(getattr(layer.cu_state, field), getattr(old.cu_state, field))


@pytest.mark.parametrize("tree", order.all_orders(4), ids=order.tree_text)
def test_all_five_native_f2b_trees_finish_and_preserve_clocks(tree):
    plan = order.build_plan(config(selection=order.tree_text(tree)))
    route = plan["routes"][0]
    run = AssemblyExecutor(plan, library(), temporal_profile="distilled14")
    output = order.archive_template(plan)["routes"][0]
    run.run_route(route, output)
    final = run.bodies[route["final"]["state_id"]]
    assert output["final"]["total_thicknesses_um"] == [50.] * 4
    assert len(final.placements) == 16
    assert final.placements[-1].role == "outer_retained_si"
    assert "wafer04_back_bi" not in final.ages_s
    assert final.ages_s["wafer04_outer_retained_si"] == final.ages_s["wafer04_front_bi"]
    for wafer, cycles in enumerate(route["final"]["thermal_cycle_counts"], 1):
        assert final.ages_s[f"wafer{wafer:02d}_front_bi"] == 300. + 4900. * cycles
    for event in output["events"]:
        assert len(event["audit"]["preparation"]["removal"]["removed_names"]) == 2
        assert all(v == 0 for v in event["audit"]["formation"]["held_flat_history_increment"].values())


def test_native_archive_scores_zero_yield_and_contains_finished_geometry():
    from test_assembly_order import overlay_input
    plan = order.build_plan(config(selection="((1,2),(3,4))"))
    archive = run_native(plan, library(), dict(fixture="not_physical_validation"),
        temporal_profile="distilled14", metric=dict(model="disk"))
    assert archive["physical_readiness"] is False
    assert archive["native_configuration"]["assembly_accuracy_validated"] is False
    overlay = overlay_input()
    overlay.update(coordinates_um=[[[100000., 0.]]], site_mask=[True], residual_sigma_um=0.,
        translation_um=[[[100., 0.]] * 3], rotation_rad=[[0.] * 3], threshold_um=0.)
    result = order.evaluate_archive(plan, archive, overlay)
    assert result["routes"][0]["stack_overlay_yield"] == 0.
    assert archive["routes"][0]["finalization"]["result_checkpoint_sha256"] == archive["routes"][0]["final"]["checkpoint_sha256"]
    archive["routes"][0]["final"]["total_thicknesses_um"][-1] = 775.
    with pytest.raises(ValueError, match="final outer-back finishing"):
        order.evaluate_archive(plan, archive, overlay)


@pytest.mark.parametrize("tree", order.all_orders(4), ids=order.tree_text)
def test_incoming_prethin_keeps_fixed_datums_and_original_si_history(tree):
    source = config(selection=order.tree_text(tree))
    source["prethin_incoming"] = True
    plan = order.build_plan(source)
    route = plan["routes"][0]
    run = AssemblyExecutor(plan, library(True), temporal_profile="distilled14")
    output = order.archive_template(plan)["routes"][0]
    run.run_route(route, output)
    final = run.bodies[route["final"]["state_id"]]
    assert route["assembled"]["total_thicknesses_um"] == [50.] * 4
    assert route["finalization"]["removed_pure_si_um"] == 0.
    assert final.placements[-1].role == "outer_retained_si"
    for wafer in range(1, 5):
        assert next(p for p in final.placements if p.die_index == wafer - 1).z_bottom_mm == pytest.approx((wafer - 1) * .05)
    for event in route["events"]:
        if "upper_before_preparation" not in event:
            continue
        before = run.bodies[event["upper_before_preparation"]["state_id"]]
        after = run.bodies[event["upper_free"]["state_id"]]
        old = before.state.layers[after.placements[-1].name]
        new = after.state.layers[after.placements[-1].name]
        assert after.placements[-1] is before.placements[-2]
        np.testing.assert_array_equal(new.birth_macro_coordinates, old.birth_macro_coordinates)
        assert after.ages_s[after.placements[-1].name] == before.ages_s[after.placements[-1].name]
    from test_assembly_order import overlay_input
    archive = run_native(plan, library(), dict(fixture="not_physical_validation"),
        temporal_profile="distilled14", metric=dict(model="disk"))
    assert order.evaluate_archive(plan, archive, overlay_input())["physical_readiness"] is False


def test_new_yaml_cli_dispatches_f2b_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(operators, "load_f2b_assembly_library",
        lambda *args, **kwargs: (library(), dict(fixture="not_physical_validation")))
    source = order.load_stack_input(INPUT)
    source["stack"]["assembly_orders"] = "(1,(2,(3,4)))"
    source["mechanics"]["temporal_profile"] = "distilled14"
    from omegaconf import OmegaConf
    path = tmp_path / "stack.yaml"
    OmegaConf.save(OmegaConf.create(source), path)
    output = tmp_path / "result.json"
    order.main(["--config", str(path), "--native-mechanics", "--output", str(output)])
    archive = json.loads(output.read_text())
    assert archive["routes"][0]["final"]["total_thicknesses_um"] == [50.] * 4


def test_complete_new_design_inputs_and_normal_runner(tmp_path):
    from omegaconf import OmegaConf
    from W2W.assembly_design import load_design, run_design
    cfg = OmegaConf.load(INPUT.parent / "calculator.yaml").w2w_modeling
    cfg.OVERLAY_SAMPLES = 8
    design, plan, overlay, resolved, provenance = load_design(cfg, INPUT.parent, "f2b_4hi50_p24_preview")
    assert plan["schema"] == order.F2B_PLAN_SCHEMA
    assert len(provenance["input_sha256"]) == 8
    assert len(resolved) == 3
    assert provenance["site_count"] == 624
    assert design["mechanics"]["frozen_preview"] is True
    run = run_design(cfg, INPUT.parent, "f2b_4hi50_p24_preview", output_dir=tmp_path,
        operator_loader=lambda *args, **kwargs: (library(), dict(fixture="not_physical_validation")))
    result = json.loads((run / "RESULT.json").read_text())
    assert len(result["routes"]) == 5
    assert result["physical_readiness"] is False
    assert (run / "SUMMARY.csv").is_file()
