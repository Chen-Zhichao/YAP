"""Native-kernel integration checks using tiny, nonphysical operator fixtures."""

from dataclasses import replace
import json
from pathlib import Path
from omegaconf import OmegaConf

import numpy as np
import pytest

from W2W.assembly_order import _build_legacy_plan as build_plan, archive_template, evaluate_archive, main, load_stack_input
from W2W.assembly_mechanics import AssemblyExecutor, native, run_native, _check_metric_policy

if not (Path(__file__).resolve().parents[2] / "Ansys_Verify/12_stack_model_validation").is_dir():
    pytest.skip("sibling native mechanics repository is not installed", allow_module_level=True)


EXAMPLE = Path(__file__).resolve().parents[1] / "W2W/input/design_6_4hi_p2/stack.yaml"


@pytest.fixture(autouse=True)
def legacy_kernel_cli(monkeypatch):
    import W2W.assembly_order as module
    monkeypatch.setattr(module, "build_plan", build_plan)


def plan(orders="all", bows=(10., -10., 20., -20.)):
    config = load_stack_input(EXAMPLE)["stack"]
    config.update(assembly_orders=orders, initial_bows_um=list(bows))
    return build_plan(config)


def tiny_library(with_cu=False):
    fixture = native("test_exp12_mindlin_sequential_runtime")
    library = fixture.library()
    if not with_cu:
        operators = dict(bi=fixture.operator(2), beol=fixture.operator(3),
                         retained_si=fixture.operator(5), bulk_si=fixture.operator(9))
    else:
        cu = native("test_exp12_free_membrane_bonding")
        operators = {key: cu._operator(seed) for seed, key in
                     enumerate(("bi", "beol", "retained_si", "bulk_si"), 50)}
        cap = operators["bulk_si"]
        operator_type = native("exp12_all_gauss_reduced_fe2").AllGaussReducedFE2Layer
        operators["bulk_si"] = operator_type(macro_size=cap.macro_size, bubble_size=cap.bubble_size,
            reference_stiffness=cap.reference_stiffness, reference_thermal_load=cap.reference_thermal_load,
            strain_maps=np.zeros((0, 6, cap.macro_size + cap.bubble_size)), weights_mm3=np.empty(0),
            macro_coordinate_scale=cap.macro_coordinate_scale, bubble_coordinate_scale=cap.bubble_coordinate_scale,
            cu_card=cap.cu_card, plastic_active=False)
        library = replace(library, reflection_matrix=cu.REFLECTION,
                          rigid_port_mode_indices=(0,), port_coordinate_scale=np.ones(2), port_rank_per_face=2)
    return replace(library, operators=operators,
                   thickness_mm=dict(bi=.0015, beol=.010, retained_si=.037, bulk_si=.7265))


def executor(request=None, with_cu=False):
    return AssemblyExecutor(request or plan(), tiny_library(with_cu), temporal_profile="distilled14")


def test_native_birth_sign_datum_and_fixed_bows():
    run = executor(plan(bows=(0., -10., 20., -30.)))
    for wafer, expected in enumerate((0., -10., 20., -30.), 1):
        body = run.leaf(wafer)
        np.testing.assert_allclose(body.state.generalized_coordinates[3:5] * 300**2 * 1000 / 8, expected)
        np.testing.assert_allclose(body.placements[-1].z_top_mm - body.placements[0].z_bottom_mm, .775)
        assert all(p.birth_reference_z_mm == pytest.approx((wafer - 1) * .05 + .3875)
                   for p in body.placements)
        assert set(body.ages_s.values()) == {300.}


def test_prepare_is_cap_removal_not_state_reinitialization():
    run = executor(with_cu=True)
    body = run.leaf(3)
    # Nonzero histories exercise transfer, independently of a thermal solve.
    for name, layer in body.state.layers.items():
        if "bulk" not in name:
            layer.cu_state.equivalent_plastic_strain[:] = .002
            layer.cu_state.creep_strain[:, 0] = 1e-6
            layer.cu_state.creep_strain[:, 1] = -1e-6
            body.ages_s[name] = 5200.
    before = run.checkpoint(body)
    prepared, audit = run.prepare_back(body)
    assert run.checkpoint(body) == before
    assert prepared.placements[-1].role == "back_bi"
    assert prepared.placements[-1].thickness_orientation == -1
    assert prepared.placements[-1].z_top_mm - prepared.placements[0].z_bottom_mm == pytest.approx(.05)
    assert audit["removal"]["removed_names"] == ["wafer03_removable_bulk_si"]
    for name, layer in prepared.state.layers.items():
        if name.endswith("back_bi"):
            assert prepared.ages_s[name] == 300.
            assert not layer.cu_state.equivalent_plastic_strain.any()
        else:
            assert prepared.ages_s[name] == 5200.
            for field in ("plastic_strain", "creep_strain", "equivalent_plastic_strain", "equivalent_creep_strain"):
                np.testing.assert_array_equal(getattr(layer.cu_state, field), getattr(body.state.layers[name].cu_state, field))
            np.testing.assert_array_equal(layer.birth_macro_coordinates, body.state.layers[name].birth_macro_coordinates)


@pytest.mark.parametrize("order", ["(((1,2),3),4)", "((1,(2,3)),4)", "((1,2),(3,4))",
                                   "(1,((2,3),4))", "(1,(2,(3,4)))"])
def test_all_five_native_routes_keep_ages_and_geometry(order):
    request = plan(order)
    run = executor(request)
    target = archive_template(request)["routes"][0]
    run.run_route(request["routes"][0], target)
    final = run.bodies[request["routes"][0]["final"]["state_id"]]
    assert len(target["events"]) == 3
    for wafer, cycles in enumerate(request["routes"][0]["final"]["thermal_cycle_counts"], 1):
        assert final.ages_s[f"wafer{wafer:02d}_front_bi"] == 300. + cycles * 4900.
        own = [p for p in final.placements if p.die_index == wafer - 1]
        assert (own[-1].z_top_mm - own[0].z_bottom_mm) * 1000 == pytest.approx(775 if wafer == 4 else 50)
    for output in target["events"]:
        assert len(output["checkpoints"]) == 4
        assert output["audit"]["formation"]["held_flat_history_increment"]
        assert all(v == 0 for v in output["audit"]["formation"]["held_flat_history_increment"].values())


def test_balanced_real_cu_children_keep_distinct_histories_and_ownership():
    run = executor(plan("((1,2),(3,4))"), with_cu=True)
    pairs = []
    for first in (1, 3):
        lower, _ = run.prepare_back(run.leaf(first))
        upper = run.leaf(first + 1)
        for body in (lower, upper):
            for layer in body.state.layers.values():
                layer.cu_state.equivalent_plastic_strain[:] = first * .001
        pair, _ = run.join(lower, upper, f"bond{first}")
        pair, _ = run.anneal(pair)
        pairs.append(pair)
    lower, _ = run.prepare_back(pairs[0])
    upper = pairs[1]
    before = [run.checkpoint(body) for body in (lower, upper)]
    joined, record = run.join(lower, upper, "final")
    assert before == [run.checkpoint(body) for body in (lower, upper)]
    for body in (lower, upper):
        for name, old in body.state.layers.items():
            new = joined.state.layers[name]
            for field in ("plastic_strain", "creep_strain", "equivalent_plastic_strain", "equivalent_creep_strain"):
                np.testing.assert_array_equal(getattr(new.cu_state, field), getattr(old.cu_state, field))
                assert not np.shares_memory(getattr(new.cu_state, field), getattr(old.cu_state, field))
            assert joined.ages_s[name] == body.ages_s[name]
    assert joined.ages_s["wafer02_back_bi"] == 300.
    assert joined.ages_s["wafer03_front_bi"] == 5200.
    assert record["held_flat_local_coordinate_drift"] < 1e-8
    assert any(np.max(layer.cu_state.equivalent_creep_strain, initial=0.) > 0
               for body in pairs for layer in body.state.layers.values())


def test_bad_geometry_and_recipe_fail_before_mechanics():
    request = plan()
    lib = tiny_library()
    with pytest.raises(ValueError, match="missing exact"):
        AssemblyExecutor(request, replace(lib, thickness_mm={**lib.thickness_mm, "retained_si": .050}))
    config = load_stack_input(EXAMPLE)["stack"]
    config["process"]["room_temperature_c"] = 20
    with pytest.raises(ValueError, match="thermal recipe"):
        executor(build_plan(config))


def test_native_archive_roundtrip_and_existing_reference_overlay():
    request = plan("((1,2),(3,4))", bows=(0., 0., 0., 0.))
    archive = run_native(request, tiny_library(), {"fixture": "nonphysical_unit_test"},
        temporal_profile="distilled14", metric={"model": "explicit", "chi": [1., 1.],
                                               "source": "unit-test-only metric"})
    archive = json.loads(json.dumps(archive, allow_nan=False))
    assert archive["status"] == "COMPLETE"
    assert "LEGACY_NATURAL_Q6" in archive["provenance"]["physics_qualification"]
    overlay = dict(coordinates_um=[[[1000., 0.]]], critical_pad_mask=[True], site_mask=[True],
        threshold_um=.3, residual_sigma_um=0., translation_um=[[[0., 0.]] * 3],
        rotation_rad=[[0.] * 3], magnification_correction_fraction=0.,
        conditional_interface_independence=True)
    result = evaluate_archive(request, archive, overlay)
    assert result["mechanics_configuration"]["metric"]["source"] == "unit-test-only metric"
    assert result["routes"][0]["stack_overlay_yield"] == 1.
    assert result["physical_readiness"] is False
    for event in archive["routes"][0]["events"]:
        assert len(event["checkpoints"]) == 4
        assert event["raw_prebond"]["lower"]["reference_policy"].startswith("elastic_reference")


@pytest.mark.parametrize("format", ["yaml", "json"])
def test_native_cli_executes_and_writes_mechanics(tmp_path, monkeypatch, format):
    # Keep this a CLI/kernel integration test, without loading gigabyte datasets.
    import W2W.assembly_operator_library as loaders
    monkeypatch.setattr(loaders, "load_assembly_library",
                        lambda *_: (tiny_library(), {"fixture": "nonphysical_unit_test"}))
    config = load_stack_input(EXAMPLE)["stack"]
    config["assembly_orders"] = "(1,(2,(3,4)))"
    (tmp_path / "input.json").write_text(json.dumps(config))
    (tmp_path / "native.json").write_text(json.dumps(dict(operator_package="fixture", temporal_profile="distilled14")))
    args = ["--config", str(tmp_path / "input.json"), "--native-mechanics", str(tmp_path / "native.json"),
            "--output", str(tmp_path / "result.json")]
    if format == "yaml":
        OmegaConf.save(OmegaConf.create(dict(schema="yap.w2w_stack_input.v1", stack=config,
            mechanics=dict(operator_package="fixture", temporal_profile="distilled14"))), tmp_path / "stack.yaml")
        args = ["--config", str(tmp_path / "stack.yaml"), "--native-mechanics",
                "--output", str(tmp_path / "result.json")]
    main(args)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "MECHANICS_COMPLETE_OVERLAY_PENDING"
    assert len(result["routes"][0]["events"]) == 3
    assert result["routes"][0]["final"]["ages_s"]["wafer04_front_bi"] == 15000.
    with pytest.raises(SystemExit):
        main(args)


def test_native_metric_policy_is_explicit():
    request = plan()
    _check_metric_policy(request, None)
    for bad in ({}, [], {"model": "circular", "chi": [1., 1.]},
                {"model": "explicit", "chi": [1., 1.]},
                {"model": "explicit", "source": "test", "chi": [float("nan"), 1.]}):
        with pytest.raises(ValueError):
            _check_metric_policy(request, bad)


def test_failed_thermal_cycle_keeps_committed_history_and_ages(monkeypatch):
    run = executor()
    body = run.leaf(2)
    body.ages_s["wafer02_front_bi"] = 5200.
    before = run.checkpoint(body)
    advance = run.thermal.advance_all_gauss_thermal_step
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("test-only rejected cycle")
        return advance(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(run.thermal, "advance_all_gauss_thermal_step", fail_second)
        with pytest.raises(RuntimeError, match="rejected cycle"):
            run.anneal(body)
    assert len(calls) == 2
    assert run.checkpoint(body) == before
    result, _ = run.anneal(body)
    assert result.ages_s["wafer02_front_bi"] == 10100.
    assert result.ages_s["wafer02_beol"] == 5200.
    assert run.checkpoint(body) == before


def test_elastic_reference_cache_does_not_reuse_age_dependent_tangents():
    run = executor(with_cu=True)
    for op in run.library.operators.values():
        op.cu_card = replace(op.cu_card, creep_law="modified_time_hardening",
            creep_prefactor_mpa_minus_n_per_s=1.43e10, creep_exponent=2.5,
            creep_time_exponent=-.9, creep_activation_term_k=23695.)
    body = run.leaf(2)
    stack = native("exp12_all_gauss_periodic_stack")
    cache = stack.AllGaussStackReferenceCache()

    def solve(age, reference):
        for instance in body.topology.instances:
            instance.operator.age_offset_s = age - 550.
        trial = stack.trial_all_gauss_stack_increment(body.topology.instances, body.state,
            350., 90., elapsed_time_s=550., creep_active=True, plastic_active=False,
            coordinate_scale=body.topology.coordinate_scale,
            constraint_matrix=body.topology.gauge_constraint_matrix,
            constraint_values=body.topology.gauge_constraint_values,
            linear_solver="bfgs_inverse", convergence_measure="residual", tolerance=1e-9,
            max_iterations=256, reference_cache=reference)
        return stack.commit_all_gauss_stack_trial(trial)

    young = solve(5300., cache)
    reused = solve(10300., cache)
    fresh = solve(10300., stack.AllGaussStackReferenceCache())
    np.testing.assert_allclose(reused.generalized_coordinates, fresh.generalized_coordinates, atol=1e-11)
    differences = []
    for name, layer in reused.layers.items():
        np.testing.assert_allclose(layer.cu_state.creep_strain, fresh.layers[name].cu_state.creep_strain, atol=1e-11)
        differences.append(np.max(np.abs(layer.cu_state.creep_strain - young.layers[name].cu_state.creep_strain), initial=0.))
    assert max(differences) > 1e-10
