"""Software-contract tests. Fabricated records below are not mechanical evidence."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from W2W.assembly_order import (
    all_orders, archive_template, build_plan, content_id, evaluate_archive,
    main, parse_order, tree_text, load_stack_input,
)
from history_bow_yield import PrebondBody, paired_magnification
from mechanics_runtime import configure_runtime_storage


def temporary_directory():
    return tempfile.TemporaryDirectory(dir=configure_runtime_storage()["TMPDIR"])


EXAMPLE = Path(__file__).resolve().parents[1] / "W2W/input/design_6_4hi_p2/stack.yaml"


def config():
    return load_stack_input(EXAMPLE)["stack"]


def overlay_input(count=4):
    return dict(coordinates_um=[[[1000., 0.]], [[100000., 0.]]],
                critical_pad_mask=[True], site_mask=[True, True],
                threshold_um=.3, residual_sigma_um=.01,
                translation_um=[[[0., 0.]] * (count - 1)], rotation_rad=[[0.] * (count - 1)],
                magnification_correction_fraction=0., conditional_interface_independence=True)


def synthetic_archive(plan):
    """Deterministic fixtures deliberately unrelated to solved Cu mechanics."""
    archive = archive_template(plan)
    archive["status"] = "COMPLETE"
    archive["provenance"] = dict(solver_id="SYNTHETIC_UNIT_TEST_ONLY",
        implementation_sha256="a" * 64, operator_set_sha256="b" * 64,
        source_transfer_audit="synthetic/not-a-physical-audit",
        material_clock_audit="synthetic/not-a-physical-audit", physics_qualification="NONE")
    for result, route in zip(archive["routes"], plan["routes"]):
        for record, event in zip(result["events"], route["events"]):
            for role in ("lower", "upper"):
                state = event[f"{role}_free"]
                bow = sum(plan["context"]["initial_bows_um"][i - 1] for i in state["wafer_ids"])
                record[role] = dict(state_id=state["state_id"], wafer_ids=state["wafer_ids"],
                    total_thicknesses_um=state["total_thicknesses_um"], frame_id=plan["context"]["frame_id"],
                    bow_convention=plan["context"]["bow_convention"], bow_um=[bow, bow],
                    face_lever_um=200. if role == "lower" else -200., chi=[1., 1.],
                    face_lever_source="SYNTHETIC_FIXTURE", face="BACK" if role == "lower" else "FRONT")
            record["checkpoints"] = {event[key]["state_id"]: content_id(event[key]["state_id"])
                                     for key in ("lower_before_preparation", "lower_free", "upper_free", "result")}
            record["transfer_audit"] = "synthetic/not-a-physical-audit"
    return archive


class AssemblyPlanTests(unittest.TestCase):
    def test_five_fixed_order_trees(self):
        expected = {"(((1,2),3),4)", "((1,(2,3)),4)", "((1,2),(3,4))",
                    "(1,((2,3),4))", "(1,(2,(3,4)))"}
        self.assertEqual({tree_text(t) for t in all_orders(4)}, expected)
        self.assertEqual(len(all_orders(3)), 2)
        self.assertEqual(len(all_orders(8)), 429)
        with self.assertRaises(ValueError):
            all_orders(9)

    def test_literal_and_nested_inputs(self):
        tree = parse_order(" ( (1,2), (3,4) ) ", 4)
        self.assertEqual(tree, ((1, 2), (3, 4)))
        self.assertEqual(parse_order([[1, 2], [3, 4]], 4), tree)
        for invalid in ("((1,2),3,4)", "((1,2),(3,3))", "((1,3),(2,4))",
                        "(1,2)", "((1,2),(3,4)) trailing", "__import__('os').getcwd()",
                        [True, [2, [3, 4]]], [1.0, [2, [3, 4]]], [1, [2, [3, 5]]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_order(invalid, 4)

    def test_all_routes_thin_once_and_have_identical_final_geometry(self):
        plan = build_plan(config())
        self.assertEqual(plan["status"], "PLANNED_MECHANICS_REQUIRED")
        self.assertNotIn("best_supplied_orders", plan)
        self.assertNotIn("stack_overlay_yield", json.dumps(plan))
        for route in plan["routes"]:
            self.assertEqual(route["final"]["total_thicknesses_um"], [50., 50., 50., 775.])
            self.assertEqual(len(route["events"]), 3)
            self.assertEqual(sorted(e["thinning"]["wafer"] for e in route["events"]), [1, 2, 3])
            self.assertEqual({e["interface_id"] for e in route["events"]},
                             {"1.BACK--2.FRONT", "2.BACK--3.FRONT", "3.BACK--4.FRONT"})
            for event in route["events"]:
                self.assertEqual(event["thinning"]["retained_si_um"], 37.)
                self.assertEqual(event["operations"][-2:], ["free_anneal", "free_cooldown"])
                self.assertTrue(event["thinning"]["source_amplitude_unchanged"])

    def test_balanced_late_thinning_and_two_processed_children(self):
        cfg = config()
        cfg["assembly_orders"] = "((1,2),(3,4))"
        route = build_plan(cfg)["routes"][0]
        first, second, last = route["events"]
        self.assertEqual(first["result"]["total_thicknesses_um"], [50., 775.])
        self.assertEqual(second["result"]["total_thicknesses_um"], [50., 775.])
        self.assertEqual(last["lower_before_preparation"], first["result"])
        self.assertEqual(last["upper_free"], second["result"])
        self.assertEqual(last["thinning"]["wafer"], 2)
        self.assertEqual(last["lower_free"]["wafer_ids"], [1, 2])
        self.assertEqual(last["lower_free"]["total_thicknesses_um"], [50., 50.])
        self.assertEqual(route["final"]["thermal_cycle_counts"], [2, 2, 2, 2])

    def test_top_down_flow_and_unattached_siblings(self):
        cfg = config()
        cfg["assembly_orders"] = "(1,(2,(3,4)))"
        route = build_plan(cfg)["routes"][0]
        self.assertEqual([e["thinning"]["wafer"] for e in route["events"]], [3, 2, 1])
        self.assertEqual([e["upper_free"]["wafer_ids"] for e in route["events"]], [[4], [3, 4], [2, 3, 4]])
        self.assertEqual(route["final"]["thermal_cycle_counts"], [1, 2, 3, 3])

    def test_signed_zero_births_are_fixed_and_reconstructed(self):
        cfg = config()
        cfg["initial_bows_um"] = [0, -10, 20, -30]
        plan = build_plan(cfg)
        birth = plan["initialization"]
        np.testing.assert_allclose([w["reconstruction_bow_um"] for w in birth["wafers"]], cfg["initial_bows_um"])
        self.assertEqual(birth["wafers"][0]["initial_q6"], [0.] * 6)
        self.assertFalse(birth["recalibrate_after_thinning"])
        self.assertFalse(birth["redraw_per_route"])
        self.assertIn("NOT_RESIDUAL_STRESS", birth["status"])

    def test_existing_beol_only_source_is_preserved(self):
        cfg = config()
        self.assertEqual([layer["eigenstrain_per_amplitude"] for layer in cfg["fabrication_layers"]], [0., 1., 0.])
        self.assertEqual(cfg["fabrication_layers"][1]["e_mpa"], 93300.)
        cfg["fabrication_layers"][0]["eigenstrain_per_amplitude"] = 1.
        with self.assertRaisesRegex(ValueError, "confines its source to BEOL"):
            build_plan(cfg)

    def test_route_selection_and_context_identity(self):
        cfg = config()
        plan = build_plan(cfg)
        selected = [plan["routes"][2]["order"], plan["routes"][0]["order"]]
        cfg["assembly_orders"] = selected
        subset = build_plan(cfg)
        self.assertEqual(subset["context_id"], plan["context_id"])
        self.assertEqual([r["request_id"] for r in subset["routes"]],
                         [plan["routes"][2]["request_id"], plan["routes"][0]["request_id"]])
        cfg["assembly_orders"] = [selected[0], selected[0]]
        with self.assertRaises(ValueError):
            build_plan(cfg)

    def test_invalid_geometry_source_and_process(self):
        for change in (lambda c: c["initial_bows_um"].__setitem__(0, float("nan")),
                       lambda c: c["geometry"].__setitem__("target_total_thickness_um", 10),
                       lambda c: c["fabrication_layers"][2].__setitem__("thickness_um", 775),
                       lambda c: c["fabrication_layers"][2].__setitem__("eigenstrain_per_amplitude", 1),
                       lambda c: c["process"].__setitem__("anneal_support", "carrier"),
                       lambda c: c["process"].__setitem__("waiting_policy", "unspecified")):
            cfg = config()
            change(cfg)
            with self.assertRaises(ValueError):
                build_plan(cfg)


class ArchiveScoringTests(unittest.TestCase):
    def setUp(self):
        self.plan = build_plan(config())
        self.archive = synthetic_archive(self.plan)
        self.overlay = overlay_input()

    def test_score_all_without_claiming_physics_validation(self):
        before = deepcopy((self.plan, self.archive, self.overlay))
        result = evaluate_archive(self.plan, self.archive, self.overlay)
        self.assertEqual(result["physics_qualification"], "UNKNOWN")
        self.assertFalse(result["physical_readiness"])
        self.assertEqual(len(result["routes"]), 5)
        self.assertFalse(result["other_yield_mechanisms_included"])
        for route in result["routes"]:
            self.assertTrue(0 <= route["stack_overlay_yield"] <= 1)
            self.assertEqual(len(route["interfaces"]), 3)
        self.assertEqual(before, (self.plan, self.archive, self.overlay))

    def test_reference_pair_not_bow_subtraction(self):
        kwargs = dict(state_id="free", frame_id="aligned", bow_um=(10., 10.),
                      halfspan_um=(150000., 150000.), chi=(1., 1.), face_lever_source="test")
        lower = PrebondBody(body_id="lower", face_lever_um=200., **kwargs)
        upper = PrebondBody(body_id="upper", face_lever_um=-200., **kwargs)
        np.testing.assert_allclose(paired_magnification(upper, lower), [8000. / 150000.**2] * 2)

    def test_reference_linear_and_quadratic_terms_against_independent_values(self):
        common = dict(state_id="free", frame_id="aligned", halfspan_um=(100., 100.),
                      face_lever_source="test")
        lower = PrebondBody(body_id="lower", bow_um=(2., 3.), face_lever_um=(4., 5.),
                           chi=(.2, .4), **common)
        upper = PrebondBody(body_id="upper", bow_um=(5., 7.), face_lever_um=(-2., -3.),
                           chi=(.6, .8), **common)
        np.testing.assert_allclose(paired_magnification(upper, lower), [50.2 / 10000., 107.6 / 10000.])

    def test_distinct_draws_rotation_and_selected_sites_and_pads(self):
        cfg = config()
        cfg["initial_bows_um"] = [0.] * 4
        plan = build_plan(cfg)
        archive = synthetic_archive(plan)
        overlay = overlay_input()
        overlay.update(coordinates_um=[[[1000., 0.], [100000., 0.]],
                                       [[2000., 0.], [100000., 0.]]],
                       critical_pad_mask=[True, False], site_mask=[True, False],
                       residual_sigma_um=0., translation_um=[[[0., 0.]] * 3] * 3,
                       rotation_rad=[[0., 0., 0.], [.0002, 0., 0.], [.0004, 0., 0.]])
        for result in evaluate_archive(plan, archive, overlay)["routes"]:
            self.assertAlmostEqual(result["stack_overlay_yield"], 2. / 3.)
        overlay["site_mask"] = [True, True]
        for result in evaluate_archive(plan, archive, overlay)["routes"]:
            self.assertAlmostEqual(result["stack_overlay_yield"], .5)
        overlay["site_mask"] = [True, False]
        overlay["critical_pad_mask"] = [False, True]
        for result in evaluate_archive(plan, archive, overlay)["routes"]:
            self.assertAlmostEqual(result["stack_overlay_yield"], 1. / 3.)

    def test_exporter_claim_does_not_establish_validation(self):
        self.archive["provenance"]["physics_qualification"] = "VALIDATED"
        result = evaluate_archive(self.plan, self.archive, self.overlay)
        self.assertEqual(result["physics_qualification"], "UNKNOWN")
        self.assertFalse(result["physical_readiness"])

    def test_equivalent_numeric_representations_of_shared_states(self):
        original = evaluate_archive(self.plan, self.archive, self.overlay)
        for index, route in enumerate(self.archive["routes"]):
            if index % 2 == 0:
                continue
            for event in route["events"]:
                for role in ("lower", "upper"):
                    body = event[role]
                    body["bow_um"] = [int(value) for value in body["bow_um"]]
                    body["chi"] = [int(value) for value in body["chi"]]
                    body["face_lever_um"] = [int(body["face_lever_um"])] * 2
                    body["total_thicknesses_um"] = [int(value) for value in body["total_thicknesses_um"]]
        changed = evaluate_archive(self.plan, self.archive, self.overlay)
        self.assertEqual(original["routes"], changed["routes"])

    def test_registered_site_joint_yield_not_product_of_marginals(self):
        cfg = config()
        cfg["initial_bows_um"] = [10.] * 4
        plan = build_plan(cfg)
        archive = synthetic_archive(plan)
        settings = overlay_input()
        settings["threshold_um"] = .03
        settings["residual_sigma_um"] = 0.
        output = evaluate_archive(plan, archive, settings)
        for route in output["routes"]:
            self.assertEqual(route["stack_overlay_yield"], .5)
            self.assertEqual(route["joint_site_summary"]["product_of_marginal_means"], .125)

    def test_physical_interface_placement_indices_survive_event_reordering(self):
        cfg = config()
        cfg["initial_bows_um"] = [0.] * 4
        plan = build_plan(cfg)
        overlay = overlay_input()
        overlay["residual_sigma_um"] = 0.
        overlay["translation_um"] = [[[0., 0.], [1., 0.], [0., 0.]]]
        output = evaluate_archive(plan, synthetic_archive(plan), overlay)
        for route in output["routes"]:
            yields = {e["interface_id"]: e["overlay_yield"] for e in route["interfaces"]}
            self.assertEqual(yields, {"1.BACK--2.FRONT": 1., "2.BACK--3.FRONT": 0., "3.BACK--4.FRONT": 1.})

    def test_perfect_correction_removes_only_bow_contribution(self):
        self.overlay["magnification_correction_fraction"] = 1.
        self.overlay["residual_sigma_um"] = 0.
        result = evaluate_archive(self.plan, self.archive, self.overlay)
        self.assertTrue(all(r["stack_overlay_yield"] == 1 for r in result["routes"]))
        self.overlay["translation_um"][0][0] = [1., 0.]
        result = evaluate_archive(self.plan, self.archive, self.overlay)
        self.assertTrue(all(r["stack_overlay_yield"] == 0 for r in result["routes"]))

    def test_partial_wrong_or_template_archives_fail_closed(self):
        changes = [lambda a: a.__setitem__("context_id", "wrong"),
                   lambda a: a.__setitem__("status", "NEEDS_MECHANICS"),
                   lambda a: a["routes"].pop(),
                   lambda a: a["routes"].append(deepcopy(a["routes"][0])),
                   lambda a: a["routes"][0]["events"].pop(),
                   lambda a: a["routes"][0]["events"][0].__setitem__("interface_id", "2.BACK--3.FRONT"),
                   lambda a: a["routes"][0]["events"][0].__setitem__("checkpoints", {}),
                   lambda a: a["routes"][0]["events"][0]["lower"].__setitem__("bow_um", [float("nan"), 0]),
                   lambda a: a["routes"][0]["events"][0]["upper"].__setitem__("bow_um", [999., 999.]),
                   lambda a: a["provenance"].__setitem__("implementation_sha256", "not-a-hash")]
        for change in changes:
            archive = deepcopy(self.archive)
            change(archive)
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluate_archive(self.plan, archive, self.overlay)
        with self.assertRaises(ValueError):
            evaluate_archive(self.plan, archive_template(self.plan), self.overlay)

    def test_changed_geometry_source_and_bows_cannot_reuse_archive(self):
        for change in (lambda c: c["initial_bows_um"].__setitem__(0, 11.),
                       lambda c: c["geometry"].__setitem__("diameter_mm", 200.),
                       lambda c: c["fabrication_layers"][0].__setitem__("e_mpa", 71000.)):
            cfg = config()
            change(cfg)
            with self.assertRaises(ValueError):
                evaluate_archive(build_plan(cfg), self.archive, self.overlay)

    def test_shared_state_cannot_change_across_routes(self):
        occurrence = {}
        for route in self.archive["routes"]:
            for event in route["events"]:
                body = event["lower"]
                key = body["state_id"]
                if key in occurrence:
                    body["bow_um"][0] += 1
                    with self.assertRaisesRegex(ValueError, "inconsistent observations"):
                        evaluate_archive(self.plan, self.archive, self.overlay)
                    return
                occurrence[key] = True
        self.fail("test fixture must contain a shared lower free state")

    def test_multiple_draws_and_route_selection_are_repeatable(self):
        self.overlay["translation_um"] *= 3
        self.overlay["rotation_rad"] *= 3
        result = evaluate_archive(self.plan, self.archive, self.overlay)
        cfg = config()
        cfg["assembly_orders"] = [r["order"] for r in self.plan["routes"]][::-1]
        reversed_result = evaluate_archive(build_plan(cfg), self.archive, self.overlay)
        self.assertEqual(result["routes"], reversed_result["routes"][::-1])
        self.assertEqual(result["joint_placement_draws"], 3)

    def test_invalid_overlay_inputs_and_null_archive_fail_closed(self):
        for change in (lambda o: o.__setitem__("critical_pad_mask", [1]),
                       lambda o: o.__setitem__("site_mask", [False, False]),
                       lambda o: o.__setitem__("residual_sigma_um", -1.),
                       lambda o: o.__setitem__("magnification_correction_fraction", 1.01),
                       lambda o: o.__setitem__("coordinates_um", [[[200000., 0.]]]),
                       lambda o: o.__setitem__("translation_um", [[[True, 0.], [0., 0.], [0., 0.]]]),
                       lambda o: o.__setitem__("rotation_rad", [[0., 0.]]),
                       lambda o: o.__setitem__("conditional_interface_independence", False)):
            overlay = deepcopy(self.overlay)
            change(overlay)
            with self.assertRaises(ValueError):
                evaluate_archive(self.plan, self.archive, overlay)
        with self.assertRaises(ValueError):
            evaluate_archive(self.plan, None, self.overlay)

    def test_shared_checkpoint_digest_cannot_change(self):
        seen = set()
        for route in self.archive["routes"]:
            for event in route["events"]:
                for state_id in event["checkpoints"]:
                    if state_id in seen:
                        event["checkpoints"][state_id] = "c" * 64
                        with self.assertRaisesRegex(ValueError, "inconsistent checkpoint"):
                            evaluate_archive(self.plan, self.archive, self.overlay)
                        return
                    seen.add(state_id)
        self.fail("fixture must contain a shared checkpoint")

    def test_modified_plan_is_rejected(self):
        self.plan["routes"][0]["events"][0]["thinning"]["to_total_um"] = 51.
        with self.assertRaisesRegex(ValueError, "modified route"):
            evaluate_archive(self.plan, self.archive, self.overlay)


class CommandTests(unittest.TestCase):
    def test_cli_imported_scoring_roundtrip(self):
        with temporary_directory() as directory:
            path = Path(directory)
            (path / "input.json").write_text(json.dumps(config()))
            plan = build_plan(config())
            (path / "archive.json").write_text(json.dumps(synthetic_archive(plan)))
            (path / "overlay.json").write_text(json.dumps(overlay_input()))
            main(["--config", str(path / "input.json"), "--mechanics-archive", str(path / "archive.json"),
                  "--overlay-input", str(path / "overlay.json"), "--output", str(path / "result.json")])
            result = json.loads((path / "result.json").read_text())
            self.assertEqual(result["status"], "IMPORTED_MECHANICS_OVERLAY_ONLY")
            self.assertEqual(len(result["routes"]), 5)
            self.assertFalse(result["physical_readiness"])

    def test_cli_duplicate_json_keys_and_null_archive(self):
        with temporary_directory() as directory:
            path = Path(directory)
            (path / "bad.json").write_text('{"schema":"a", "schema":"b"}')
            with self.assertRaises(SystemExit) as failure:
                main(["--config", str(path / "bad.json"), "--plan-only", "--output", str(path / "out.json")])
            self.assertEqual(failure.exception.code, 2)
            (path / "null.json").write_text('null')
            (path / "input.json").write_text(json.dumps(config()))
            (path / "overlay.json").write_text(json.dumps(overlay_input()))
            with self.assertRaises(SystemExit) as failure:
                main(["--config", str(path / "input.json"), "--mechanics-archive", str(path / "null.json"),
                      "--overlay-input", str(path / "overlay.json"), "--output", str(path / "out.json")])
            self.assertEqual(failure.exception.code, 2)
            self.assertFalse((path / "out.json").exists())

    def test_plan_command_and_refuse_overwrite(self):
        with temporary_directory() as directory:
            path = Path(directory)
            args = ["--config", str(EXAMPLE), "--plan-only", "--output", str(path / "plan.json"),
                    "--archive-template", str(path / "template.json")]
            main(args)
            self.assertEqual(len(json.loads((path / "plan.json").read_text())["routes"]), 5)
            with self.assertRaises(SystemExit) as exc:
                main(args)
            self.assertEqual(exc.exception.code, 2)

    def test_failed_scoring_publishes_no_report(self):
        with temporary_directory() as directory:
            path = Path(directory)
            (path / "input.json").write_text(json.dumps(config()))
            plan = build_plan(config())
            (path / "archive.json").write_text(json.dumps(archive_template(plan)))
            (path / "overlay.json").write_text(json.dumps(overlay_input()))
            with self.assertRaises(SystemExit):
                main(["--config", str(path / "input.json"), "--mechanics-archive", str(path / "archive.json"),
                      "--overlay-input", str(path / "overlay.json"), "--output", str(path / "result.json")])
            self.assertFalse((path / "result.json").exists())


class StackYamlTests(unittest.TestCase):
    def test_compact_yaml_defaults_preserve_explicit_plan(self):
        compact = OmegaConf.to_container(OmegaConf.load(EXAMPLE), resolve=False)
        self.assertNotIn("schema", compact)
        self.assertNotIn("schema", compact["stack"])
        self.assertEqual(set(compact["stack"]["process"]), {
            "room_temperature_c", "anneal_temperature_c", "ramp_up_s", "dwell_s", "cool_down_s"})
        explicit = load_stack_input(EXAMPLE)
        self.assertEqual(Path(explicit["mechanics"]["operator_package"]).name, "BI2_50.json")
        with temporary_directory() as directory:
            path = Path(directory) / "stack.yaml"
            OmegaConf.save(OmegaConf.create(explicit), path)
            reloaded = load_stack_input(path)
        self.assertEqual(reloaded, explicit)
        self.assertEqual(build_plan(reloaded["stack"]), build_plan(config()))

    def test_internal_defaults_do_not_accept_unsupported_explicit_values(self):
        original = OmegaConf.to_container(OmegaConf.load(EXAMPLE), resolve=False)
        with temporary_directory() as directory:
            path = Path(directory)
            for key in ("recipe_id", "prebond_policy", "waiting_policy", "anneal_support",
                        "bonding_support", "material_clock_policy"):
                for invalid in (None, "unsupported" if key != "recipe_id" else ""):
                    design = deepcopy(original)
                    design["stack"]["process"][key] = invalid
                    OmegaConf.save(OmegaConf.create(design), path / "stack.yaml")
                    with self.subTest(key=key, value=invalid), self.assertRaises(SystemExit) as exc:
                        main(["--config", str(path / "stack.yaml"), "--plan-only",
                              "--output", str(path / "out.json")])
                    self.assertEqual(exc.exception.code, 2)
                    self.assertFalse((path / "out.json").exists())

    def test_default_expansion_still_requires_explicit_numeric_process_inputs(self):
        design = OmegaConf.to_container(OmegaConf.load(EXAMPLE), resolve=False)
        del design["stack"]["process"]["dwell_s"]
        with temporary_directory() as directory:
            path = Path(directory) / "stack.yaml"
            OmegaConf.save(OmegaConf.create(design), path)
            with self.assertRaises(ValueError):
                build_plan(load_stack_input(path)["stack"])

    def test_public_configuration_has_stable_new_metadata_identity(self):
        plan = build_plan(config())
        self.assertEqual(plan["context_id"], "25af65524ef5f84743bafc63d7342dc84329a28260b4c15ef89e73337165af89")
        self.assertEqual([route["request_id"] for route in plan["routes"]], [
            "cc94d0a63d56fdd991d1a3401098616d899d469eecc5e863ad7217738c9ea4c0",
            "ca54f7a1daad1d0dc249a6a65ed30da545bc5e4f5ebe084f8139eadf2191b446",
            "705ab3aca0cef9c90ae32c2769603aa45ebd95750cb224b50f31b9ccf5547672",
            "6cb49d155f8a8dd69932c4c712a2d8b659bda762084d37e8c82bd76e455f3f5e",
            "27dd98c4108f97c802fc236d90ff3f9b157e64d3dac5ba875e2533cc2c4bfdea"])

    def test_yaml_and_legacy_json_plans_are_identical_without_assets(self):
        with temporary_directory() as directory:
            path = Path(directory)
            design = load_stack_input(EXAMPLE)
            design["mechanics"]["operator_package"] = "missing/operator.json"
            design["overlay"] = overlay_input()
            OmegaConf.save(OmegaConf.create(design), path / "stack.yml")
            (path / "input.json").write_text(json.dumps(design["stack"]))
            for name in ("stack.yml", "input.json"):
                main(["--config", str(path / name), "--plan-only", "--output", str(path / (name + ".out"))])
            self.assertEqual(json.loads((path / "stack.yml.out").read_text()),
                             json.loads((path / "input.json.out").read_text()))
            self.assertEqual(load_stack_input(path / "stack.yml")["mechanics"]["operator_package"],
                             str(path / "missing/operator.json"))

    def test_inline_overlay_scores_imported_archive(self):
        with temporary_directory() as directory:
            path = Path(directory)
            design = load_stack_input(EXAMPLE)
            design["overlay"] = overlay_input()
            OmegaConf.save(OmegaConf.create(design), path / "stack.yaml")
            archive = synthetic_archive(build_plan(config()))
            (path / "archive.json").write_text(json.dumps(archive))
            main(["--config", str(path / "stack.yaml"), "--mechanics-archive", str(path / "archive.json"),
                  "--output", str(path / "result.json")])
            self.assertEqual(json.loads((path / "result.json").read_text()),
                             evaluate_archive(build_plan(config()), archive, overlay_input()))

    def test_yaml_rejects_invalid_structure_and_ambiguous_settings(self):
        with temporary_directory() as directory:
            path = Path(directory)
            original = load_stack_input(EXAMPLE)
            changes = [lambda d: d.__setitem__("schema", "wrong"),
                       lambda d: d.__setitem__("mechanic", {}),
                       lambda d: d.__setitem__("stack", None),
                       lambda d: d.__setitem__("overlay", None),
                       lambda d: d.__setitem__("overlay", {}),
                       lambda d: d.__setitem__("mechanics", None),
                       lambda d: d["mechanics"].__setitem__("layer_workers", True),
                       lambda d: d["mechanics"].__setitem__("operator_package", "${oc.env:HOME}"),
                       lambda d: d["stack"].__setitem__("assembly_orders", "((1,3),(2,4))"),
                       lambda d: d["stack"].__setitem__("initial_bows_um", [True, 0., 0., 0.])]
            for change in changes:
                design = deepcopy(original)
                change(design)
                OmegaConf.save(OmegaConf.create(design), path / "stack.yaml")
                with self.subTest(change=change), self.assertRaises(SystemExit) as exc:
                    main(["--config", str(path / "stack.yaml"), "--plan-only", "--output", str(path / "out.json")])
                self.assertEqual(exc.exception.code, 2)
                self.assertFalse((path / "out.json").exists())
            for text in ("schema: a\nschema: b\n", "stack: {schema: a, schema: b}\n",
                         "stack: [\n", "!!python/object/apply:os.system ['false']\n"):
                (path / "stack.yaml").write_text(text)
                with self.assertRaises(ValueError):
                    load_stack_input(path / "stack.yaml")
            for flags in (["--native-mechanics", "external.json"],
                          ["--plan-only", "--overlay-input", "external.json"]):
                with self.assertRaises(SystemExit):
                    main(["--config", str(EXAMPLE), *flags, "--output", str(path / "out.json")])

    def test_missing_inline_mechanics_and_legacy_bare_flag_fail(self):
        with temporary_directory() as directory:
            path = Path(directory)
            OmegaConf.save(OmegaConf.create(dict(schema="yap.w2w_stack_input.v1", stack=config())),
                           path / "stack.yaml")
            (path / "input.json").write_text(json.dumps(config()))
            for name in ("stack.yaml", "input.json"):
                with self.assertRaises(SystemExit):
                    main(["--config", str(path / name), "--native-mechanics", "--output", str(path / "out.json")])
            self.assertFalse((path / "out.json").exists())

    def test_inline_native_settings_and_scoring_without_large_operator_load(self):
        with temporary_directory() as directory:
            path = Path(directory)
            design = load_stack_input(EXAMPLE)
            design["mechanics"].update(operator_package="operators/package.json",
                                       metric=dict(model="explicit", chi=[1., 1.], source="test only"))
            design["overlay"] = overlay_input()
            OmegaConf.save(OmegaConf.create(design), path / "stack.yaml")
            archive = synthetic_archive(build_plan(config()))
            with patch("W2W.assembly_mechanics.run_native_from_settings", return_value=archive) as run:
                main(["--config", str(path / "stack.yaml"), "--native-mechanics",
                      "--output", str(path / "result.json"), "--mechanics-output", str(path / "mechanics.json")])
            self.assertEqual(run.call_args.args[1]["operator_package"], str(path / "operators/package.json"))
            self.assertEqual(json.loads((path / "mechanics.json").read_text()), archive)
            result = json.loads((path / "result.json").read_text())
            self.assertEqual(result["status"], "NATIVE_MECHANICS_OVERLAY_UNVALIDATED")
            self.assertFalse(result["physical_readiness"])


if __name__ == "__main__":
    unittest.main()
