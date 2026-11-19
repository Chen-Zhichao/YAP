"""Real native assembly transactions on tiny, nonphysical whole-body fixtures."""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from D2W.assembly_order import build_plan, load_stack_input
from W2W.assembly_mechanics import AssemblyExecutor, _check_library, run_native
from W2W.assembly_order import archive_template
import test_whole_coupon_formation as controls


@dataclass(frozen=True)
class Library:
    operators: dict
    thickness_mm: dict
    membrane_bending_coordinate_scale: object
    cell_area_mm2: float = 100.
    operator_reference_y_mm: float = .045
    coupling: str = "whole_prepared_coupon"
    port_rank_per_face: int = 0
    common_port_gauge_vectors: object = None


@pytest.fixture(scope="module")
def runtime():
    return controls.runtime.__wrapped__()


def library(runtime):
    op = controls.operand(runtime).operator
    return Library({"whole_prepared": op}, {"whole_prepared": .09}, np.ones(6))


def plan(order="all", bows=(1., -1., 2.)):
    path = Path(__file__).resolve().parents[1] / "D2W/input/design_6_4hi_p2/stack.yaml"
    config = load_stack_input(path)["stack"]
    config["initial_bows_um"] = list(bows)
    config["assembly_orders"] = order
    config["geometry"]["total_thickness_um"] = 90.
    config["fabrication_layers"][2]["thickness_um"] = 77.
    return build_plan(config)


def test_leaf_owns_one_complete_body_and_preserves_natural_datum(runtime, monkeypatch):
    run = AssemblyExecutor(plan(), library(runtime), temporal_profile="component_graded25")
    def forbidden(*args, **kwargs):
        raise AssertionError("whole bodies must not enter the component-port leaf builder")
    monkeypatch.setattr(run.formation, "virgin_die_state", forbidden)
    monkeypatch.setattr(run.placement_api, "_topology", forbidden)
    for index, expected in enumerate((1., -1., 2.), 1):
        body = run.leaf(index)
        assert len(body.placements) == len(body.topology.instances) == 1
        p, instance = body.placements[0], body.topology.instances[0]
        assert p.role == "whole_prepared"
        assert p.z_top_mm - p.z_bottom_mm == pytest.approx(.09)
        assert body.topology.port_rank == 0
        assert body.topology.generalized_size == 6
        assert body.topology.interface_slices == (slice(6, 6),) * 2
        assert instance.operator is run.operators[p.name]
        assert body.ages_s == {p.name: 300.}
        np.testing.assert_allclose(instance.virgin_birth_macro_coordinates,
                                   run.births[index-1].membrane_bending_coordinates())
        np.testing.assert_allclose(body.state.generalized_coordinates[3:5] * 12500., expected)
        np.testing.assert_allclose(instance.macro_transform @ body.state.generalized_coordinates,
                                   instance.virgin_birth_macro_coordinates)
        for field in controls.CU_FIELDS:
            assert not np.any(getattr(body.state.layers[p.name].cu_state, field))
        original = run.checkpoint(body)
        prepared, report = run.prepare_back(body)
        assert prepared is body and run.checkpoint(body) == original
        assert report["whole_prepared_coupon"] and report["retained_material_ages_unchanged"]


@pytest.mark.parametrize("order,counts", [("((1,2),3)", (2, 2, 1)), ("(1,(2,3))", (1, 2, 2))])
def test_both_three_body_orders_keep_histories_clocks_and_offsets(runtime, order, counts):
    request = plan(order)
    run = AssemblyExecutor(request, library(runtime), temporal_profile="component_graded25")
    target = archive_template(request)["routes"][0]
    run.run_route(request["routes"][0], target)
    final = run.bodies[request["routes"][0]["final"]["state_id"]]
    assert len(final.topology.instances) == 3
    assert final.topology.interface_slices == (slice(6, 6),) * 4
    assert set(final.state.layers) == set(final.ages_s)
    for index, cycles in enumerate(counts, 1):
        name = f"die{index:02d}_whole_prepared"
        assert final.ages_s[name] == 300. + 4900. * cycles
        instance = next(i for i in final.topology.instances if i.name == name)
        assert instance.operator is run.operators[name]
        assert np.isfinite(instance.macro_offset).all()
    for event in target["events"]:
        assert event["audit"]["preparation"]["whole_prepared_coupon"]
        formation = event["audit"]["formation"]
        for key in ("held_flat_history_increment", "release_history_increment"):
            assert all(value == 0. for value in formation[key].values())
        for child in formation["bodies_old_then_incoming"]:
            assert all(value == 0. for value in child["history_increments"].values())
    assert np.isfinite(final.state.generalized_coordinates).all()


def test_run_native_labels_coupling_as_experimental(runtime):
    output = run_native(plan("(1,2)", (1., -1.)), library(runtime), {},
                        temporal_profile="component_graded25")
    cfg = output["native_configuration"]
    assert cfg["coupling"] == "experimental_whole_prepared_six_macro"
    assert cfg["microface_compatible"] is cfg["assembly_accuracy_validated"] is False
    assert cfg["W2W_thinning_supported"] is False
    assert output["physical_readiness"] is False


@pytest.mark.parametrize("change", [dict(cell_area_mm2=1.), dict(operator_reference_y_mm=.09),
                                   dict(thickness_mm={"whole_prepared": .05}),
                                   dict(port_rank_per_face=1),
                                   dict(membrane_bending_coordinate_scale=np.zeros(6))])
def test_invalid_whole_body_library_is_rejected(runtime, change):
    with pytest.raises(ValueError, match="invalid whole-prepared"):
        _check_library(replace(library(runtime), **change), plan()["context"]["geometry"],
                       prepared_dies=True)


def test_w2w_and_unqualified_profile_are_rejected(runtime):
    with pytest.raises(ValueError, match="D2W only"):
        _check_library(library(runtime), {}, prepared_dies=False)
    with pytest.raises(ValueError, match="component_graded25"):
        AssemblyExecutor(plan(), library(runtime), temporal_profile="distilled26")


def test_wrong_profile_fails_before_loading_maps(monkeypatch):
    from W2W import assembly_mechanics
    def forbidden(**kwargs):
        raise AssertionError("unqualified profile must fail before runtime configuration or map loading")
    monkeypatch.setattr(assembly_mechanics, "_configure_mechanics_runtime_environment", forbidden)
    with pytest.raises(ValueError, match="component_graded25"):
        assembly_mechanics.run_native_from_settings(plan(),
            dict(operator_package="coupon:BI4_Prepared90", temporal_profile="distilled26"))
