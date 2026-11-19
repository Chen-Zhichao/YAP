"""2.5D package arithmetic and real native-kernel integration on tiny fixtures."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest

import package_assembly as package


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def private_storage(tmp_path, monkeypatch):
    from mechanics_runtime import configure_runtime_storage
    run = tmp_path/"runs"
    monkeypatch.setenv("YAP_RUN_ROOT", str(run))
    for name in ("TMP", "TEMP", "TMPDIR", "YAP_WORK_DIR", "HOME", "NUMBA_CACHE_DIR", "MPLCONFIGDIR",
                 "PYTHONPYCACHEPREFIX", "PYTHONUSERBASE", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
                 "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_RUNTIME_DIR", "PIP_CACHE_DIR", "JOBLIB_TEMP_FOLDER"):
        monkeypatch.delenv(name, raising=False)
    configure_runtime_storage()
    return run


def test_interposer_reuses_source_not_die_bow_distribution():
    settings = dict(width_mm=30., length_mm=30.)
    ip, source, bow = package.interposer_population(settings, 128, 12)
    assert ip["total_thickness_um"] == 100.
    assert [v["name"] for v in ip["layers"]] == ["Si", "front_BI"]
    assert ip["layers"][0]["eigenstrain_per_amplitude"] == 0
    assert ip["source"]["calibration"]["target_upper_magnitude_um"] == 40.
    assert ip["shared_across_all_external_interfaces"]
    assert source.std() > 0 and bow.std() > 0
    np.testing.assert_allclose(ip["sample_mean_bow_um"], np.mean(bow, axis=0))
    assert not np.array_equal(ip["sample_mean_bow_um"], ip["mean_bow_um"])
    _, again, _ = package.interposer_population(settings, 128, 12)
    np.testing.assert_array_equal(source, again)
    _, source2, bow2 = package.interposer_population(dict(width_mm=15., length_mm=15.), 128, 12)
    np.testing.assert_array_equal(source2, source)
    np.testing.assert_allclose(bow2, bow/4)
    assert not np.isclose(bow.mean(), 40.)


def test_fixed_interposer_bow_and_bad_geometry():
    ip, source, bow = package.interposer_population(dict(width_mm=30., length_mm=30., initial_bow_um=-12.), 4, 2)
    np.testing.assert_allclose(bow, -12.)
    assert np.std(source) == 0.
    with pytest.raises(ValueError, match="not both"):
        package.interposer_population(dict(width_mm=30., length_mm=30., initial_bow_um=0., source_cv=.2), 4, 2)
    with pytest.raises(ValueError, match="square"):
        package.interposer_population(dict(width_mm=30., length_mm=20.), 4, 2)
    with pytest.raises(ValueError, match="positive Si"):
        package.interposer_population(dict(width_mm=30., length_mm=30., total_thickness_um=1.), 4, 2)


def test_elastic_flattening_keeps_sign_coupling_and_quadratic_metric():
    k = np.diag([3., 4., 2., .1, .2, .1])
    k[0, 3] = k[3, 0] = .02
    curve = np.array([.001, -.002, .0004])
    strain, audit = package.elastic_face_strain(k, curve, [10., 10.], .05)
    v = (np.linalg.solve(k[:3, :3], k[:3, 3:])-.05*np.eye(3)) @ curve
    expected = np.array([[v[0], v[2]/2], [v[2]/2, v[1]]])
    np.testing.assert_allclose(audit["linear_face_strain"], expected)
    inverse, negative = package.elastic_face_strain(k, -curve, [10., 10.], .05)
    np.testing.assert_allclose(negative["membrane_metric"], audit["membrane_metric"])
    np.testing.assert_allclose((strain-inverse)/2, expected)
    flat, _ = package.elastic_face_strain(k, np.zeros(3), [10., 10.], .05)
    np.testing.assert_array_equal(flat, np.zeros((2, 2)))
    assert audit["constitutive_state_evolved"] is False


def test_products_are_inside_shared_interposer_expectation():
    internal = np.ones((2, 2))
    final = np.array([[0., 0.], [1., 1.]])
    report, joint = package.compound_package(internal, final)
    assert report["package_overlay_yield"] == .5
    assert report["marginal_product_diagnostic"] == .25
    np.testing.assert_array_equal(joint, [0., 1.])
    report, _ = package.compound_package(internal, final, [.5, .5])
    assert report["package_yield"] == .125
    with pytest.raises(ValueError, match="probabilities"):
        package.compound_package(internal*2, final)


def overlay(samples=2):
    return dict(coordinates_um=[[[100., 0.], [200., 0.]]], critical_pad_mask=[True, True],
        threshold_um=.15, residual_sigma_um=0., magnification_correction_fraction=0.,
        translation_um=np.zeros((samples, 1, 2)), rotation_rad=np.zeros((samples, 1)))


def test_affine_overlay_matches_existing_threshold_and_correction():
    settings = overlay()
    matrix = np.array([np.diag([.001, 0.])])
    survival, maximum = package.score_affine(matrix, settings)
    np.testing.assert_array_equal(survival, [[0.], [0.]])
    np.testing.assert_allclose(maximum, .2)
    settings["magnification_correction_fraction"] = 1.
    survival, _ = package.score_affine(matrix, settings)
    np.testing.assert_array_equal(survival, 1.)
    settings["translation_um"][0, 0, 0] = .16
    survival, _ = package.score_affine(matrix, settings)
    np.testing.assert_array_equal(survival, [[0.], [1.]])


@pytest.mark.parametrize("order", ["(((1,2),3),4)", "((1,2),(3,4))", "(1,(2,(3,4)))"])
def test_batch_plan_has_only_one_thermal_cycle_and_distinct_ids(order):
    from D2W.assembly_order import load_stack_input
    from W2W.assembly_order import _check_plan
    source = load_stack_input(ROOT/"D2W/input/design_6_4hi_p2/stack.yaml")["stack"]
    before = deepcopy(source)
    sequential = package.stack_plan(source, order, "sequential")
    batch = package.stack_plan(source, order, "batch")
    _check_plan(batch)
    assert source == before
    assert batch["context_id"] != sequential["context_id"]
    route = batch["routes"][0]
    assert route["final"]["thermal_cycle_counts"] == [1]*4
    assert route["final"]["state_id"] == route["events"][-1]["result"]["state_id"]
    for event in route["events"][:-1]:
        assert "free_anneal" not in event["operations"]
        assert event["result"]["thermal_cycle_counts"] == [0]*len(event["result"]["die_ids"])
    assert "free_anneal" in route["events"][-1]["operations"]


def test_plan_example_without_loading_operators(private_storage):
    run = package.run_package(ROOT/"input/package_2p5d.yaml", plan_only=True)
    plan = json.loads((run/"PLAN.json").read_text())
    assert len(plan["stacks"]) == 2
    assert not (run/"RESULT.json").exists()
    assert "not_simulated" in plan["package_anneal_mechanics"]
    assert plan["post_attachment_yield_factor"] == 1.
    assert plan["post_attachment_yield_policy"] == package.POST_ATTACHMENT_POLICY


def test_external_pitch_and_noncentral_pad_coordinates():
    cfg = OmegaConf.load(ROOT/"D2W/configs/design_6_4hi_p2/design_6_4hi_p2.yaml").d2w_modeling
    options = dict(pitch_um=4., critical_coordinates_um=[[2500., 100.], [4000., 500.]],
                   translation_std_um=[0., 0.], rotation_std_rad=0.)
    result = package.attachment_overlay(options, cfg, overlay(), np.array([10., 10.]), 2, 12, "external")
    assert result["threshold_um"] > .15
    assert result["coordinates_um"] == [[[2500., 100.], [4000., 500.]]]
    assert cfg.PITCH_r_um == 2.
    with pytest.raises(ValueError, match="inside"):
        package.attachment_overlay(dict(critical_coordinates_um=[[6000., 0.]]), cfg, overlay(),
                                   np.array([10., 10.]), 2, 12, "bad")


def test_bad_positions_and_orders_rejected(tmp_path):
    doc = OmegaConf.to_container(OmegaConf.load(ROOT/"input/package_2p5d.yaml"))
    for entry in doc["stacks"]:
        entry["design_dir"] = str(ROOT/"D2W/input/design_6_4hi_p2")
        entry["config"] = str(ROOT/"D2W/configs/design_6_4hi_p2/design_6_4hi_p2.yaml")
    doc["stacks"][1]["center_mm"] = doc["stacks"][0]["center_mm"]
    path = tmp_path/"bad.yaml"
    OmegaConf.save(OmegaConf.create(doc), path)
    with pytest.raises(ValueError, match="overlap"):
        package.prepare_package(path)
    doc["stacks"][1]["center_mm"] = [100., 0.]
    OmegaConf.save(OmegaConf.create(doc), path)
    with pytest.raises(ValueError, match="outside"):
        package.prepare_package(path)
    doc["stacks"] = doc["stacks"][:1]
    doc["stacks"][0]["assembly_order"] = "all"
    OmegaConf.save(OmegaConf.create(doc), path)
    with pytest.raises(ValueError, match="one assembly_order"):
        package.prepare_package(path)


def test_package_runs_native_histories_and_analytic_attachment(tmp_path, private_storage):
    import test_whole_coupon_formation as controls
    from test_coupon_executor import library
    from generate_assembly_design import generate_design
    generated = generate_design(name="package_native", dies=3, pitch=4, total_thickness=90,
        initial_bows=[1., -1., 2.], orders="sequential", operator_package="coupon:BI4_Prepared90",
        output_root=private_storage/"designs")
    entries = [dict(name=label, center_mm=[position, 0.], design_dir=str(generated["input_dir"]),
        config=str(generated["config_path"]), assembly_order="((1,2),3)", annealing=annealing)
        for label, position, annealing in (("left", -11., "sequential"),
            ("duplicate", 0., "sequential"), ("right", 11., "batch"))]
    doc = dict(name="native_package", samples=8, seed=12,
        interposer=dict(width_mm=40., length_mm=40., initial_bow_um=10.), stacks=entries)
    path = tmp_path/"package.yaml"
    OmegaConf.save(OmegaConf.create(doc), path)
    runtime = controls.runtime.__wrapped__()
    tiny = library(runtime)
    counts = []
    def runner(entry, unused_library, unused_provenance, progress):
        output = package.run_stack(entry, tiny, {}, progress)
        ages = output["archive"]["routes"][0]["final"]["ages_s"]
        counts.append(sorted(ages.values()))
        return output
    run = package.run_package(path, stack_runner=runner)
    result = json.loads((run/"RESULT.json").read_text())
    assert result["status"] == "COMPLETE_OVERLAY_ONLY"
    assert result["joint_samples"] == 8 and len(result["stacks"]) == 3
    assert [s["deterministic_mechanics_reused"] for s in result["stacks"]] == [False, True, False]
    assert result["physical_readiness"] is False
    assert result["post_attachment_yield_factor"] == 1.
    assert result["post_attachment_yield_policy"] == package.POST_ATTACHMENT_POLICY
    assert counts[0] == [5200., 10100., 10100.]
    assert counts[1] == [5200., 5200., 5200.]
    first = json.loads((run/"stack_1_YIELD.json").read_text())
    duplicate = json.loads((run/"stack_2_YIELD.json").read_text())
    assert first["overlay_input_sha256"] != duplicate["overlay_input_sha256"]
    assert (run/"stack_1_MECHANICS.json").read_bytes() == (run/"stack_2_MECHANICS.json").read_bytes()
    with np.load(run/"SAMPLES.npz", allow_pickle=False) as data:
        np.testing.assert_allclose(result["package_overlay_yield"],
            np.mean(np.prod(data["internal_stack_survival"]*data["interposer_interface_survival"], axis=1)))
    for index in (1, 2, 3):
        exported = json.loads((run/f"stack_{index}_ELASTIC.json").read_text())
        assert exported["audit"]["constitutive_state_evolved"] is False
        assert "plastic_tangent" in exported["stiffness_policy"]
