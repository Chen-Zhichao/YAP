"""Symbolic mixed-face process contracts; no native or physical validation."""

import builtins
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import pytest

import W2W.assembly_order as order


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "W2W/input/design_6_4hi_p2/stack.yaml"
SEQUENTIAL = "(((1,2),3),4)"
BALANCED = "((1,2),(3,4))"


@pytest.fixture
def tmp_path():
    scratch = ROOT / ".cache/w2w-process-ledger-tests"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ledger-", dir=scratch) as directory:
        yield Path(directory)


@pytest.fixture(autouse=True)
def workspace_storage(monkeypatch, tmp_path):
    # These fixtures preserve the explicitly historical, plan-only v2 contract.
    monkeypatch.setattr(order, "build_plan", order._build_mixed_plan)
    assert tmp_path.resolve().is_relative_to(ROOT / ".cache")
    for key, directory in {
        "YAP_WORK_DIR": "work", "TMPDIR": "temp", "TMP": "temp", "TEMP": "temp",
        "MPLCONFIGDIR": "matplotlib", "NUMBA_CACHE_DIR": "numba",
    }.items():
        monkeypatch.setenv(key, str(tmp_path / directory))
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    original_import = builtins.__import__

    def no_native(name, *args, **kwargs):
        if "assembly_mechanics" in name or name.startswith(("exp12", "run_exp12")):
            pytest.fail(f"symbolic planning must not import {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_native)


def config(selection="all", target=50.):
    result = order.load_stack_input(INPUT)["stack"]
    result["assembly_orders"] = selection
    result["geometry"]["target_total_thickness_um"] = target
    return result


def routes(plan):
    return {route["order"]: route for route in plan["routes"]}


def resign(plan):
    plan["source_input_sha256"] = order.content_id(plan["source_input"])
    plan["context_id"] = order.content_id(dict(context=plan["context"], initialization=plan["initialization"]))
    for route in plan["routes"]:
        route["context_id"] = plan["context_id"]
        route["request_id"] = order.content_id({key: value for key, value in route.items() if key != "request_id"})
    plan["plan_id"] = order.content_id({key: value for key, value in plan.items() if key != "plan_id"})


def test_all_exposes_two_defined_and_three_distinct_empty_requests():
    source = config()
    before = deepcopy(source)
    plan = order.build_plan(source)
    assert source == before
    assert plan["schema"] == order.MIXED_PLAN_SCHEMA
    assert plan["status"] == "PLAN_ONLY_PARTIAL_PROCESS_DEFINITIONS"
    assert plan["summary"] == dict(requested_routes=5, defined_routes=2, missing_process_definitions=3)
    assert plan["physical_readiness"] is False
    assert plan["native_execution_available"] is False
    by_order = routes(plan)
    assert set(by_order) == {order.tree_text(tree) for tree in order.all_orders(4)}
    assert len({route["request_id"] for route in by_order.values()}) == 5
    for expression, route in by_order.items():
        if expression in (SEQUENTIAL, BALANCED):
            assert route["schema"] == order.MIXED_ROUTE_SCHEMA
            assert route["status"] == "PLAN_ONLY"
            assert len(route["events"]) == 3
        else:
            assert route["schema"] == order.PROCESS_PLACEHOLDER_SCHEMA
            assert route["schema"] != order.MIXED_ROUTE_SCHEMA
            assert route["status"] == "NEEDS_PROCESS_DEFINITION"
            assert route["events"] == []
            assert set(route) == {"schema", "order", "context_id", "status", "events", "blockers", "request_id"}
    assert order.validate_mixed_plan(plan) is None
    assert order.build_plan(source) == plan
    assert order.content_id(plan["source_input"]) == plan["source_input_sha256"]
    assert order.content_id({key: value for key, value in plan.items() if key != "plan_id"}) == plan["plan_id"]


@pytest.mark.parametrize("selection,defined,missing", [
    (SEQUENTIAL, 1, 0), ([SEQUENTIAL, BALANCED], 2, 0), ("(1,(2,(3,4)))", 0, 1),
    ([BALANCED, "((1,(2,3)),4)"], 1, 1),
])
def test_selection_keeps_missing_requests_visible_without_subset_execution(selection, defined, missing):
    plan = order.build_plan(config(selection))
    assert plan["summary"] == dict(requested_routes=defined + missing, defined_routes=defined,
                                   missing_process_definitions=missing)
    assert plan["status"] == ("PLAN_ONLY_PARTIAL_PROCESS_DEFINITIONS" if missing else "PLAN_ONLY")
    order.validate_mixed_plan(plan)
    with pytest.raises(ValueError, match="subset execution are unavailable"):
        order.require_executable_plan(plan)


def test_intrinsic_birth_sources_unchanged_and_proposed_orientations_unconfirmed():
    source = config()
    births = order._births(order._context(source))
    plan = order.build_plan(source)
    assert plan["initialization"] == births
    assert plan["context"]["frame_id"] == order.MIXED_FRAME
    assert plan["context"]["frame_id"] != order.FRAME
    assert plan["context"]["orientation_status"] == "UNCONFIRMED"
    assert plan["context"]["source_frame"] == "intrinsic_wafer_local_front_below_back"
    assert "orientation" not in plan["context"]
    for expression, signs in [(SEQUENTIAL, [-1, 1, 1, 1]), (BALANCED, [-1, 1, -1, 1])]:
        route = routes(plan)[expression]
        assert route["orientation_proposal"] == dict(status="UNCONFIRMED", thickness_orientation=signs,
                                                      coordinate_transform_applied=False)
        selected = order.build_plan(config(expression))
        assert selected["initialization"] == births
        assert selected["context_id"] == plan["context_id"]
        assert selected["routes"][0]["request_id"] == route["request_id"]
    changed = config()
    changed["initial_bows_um"][0] += 1.
    other = order.build_plan(changed)
    assert other["context_id"] != plan["context_id"]
    assert other["initialization"]["wafers"][0]["source_amplitude"] != births["wafers"][0]["source_amplitude"]
    assert other["initialization"]["wafers"][1:] == births["wafers"][1:]


def test_sequential_faces_preparation_and_carrier_order():
    route = order.build_plan(config(SEQUENTIAL))["routes"][0]
    first, second, third = route["events"]
    assert [event["bond_mode"] for event in route["events"]] == ["F2F", "F2B", "F2B"]
    assert [event["mating_faces"] for event in route["events"]] == [
        dict(lower="FRONT", upper="FRONT"), dict(lower="BACK", upper="FRONT"),
        dict(lower="BACK", upper="FRONT")]
    assert first["preparations"] == dict(lower=[], upper=[])
    for event, previous, wafer in [(second, first, 2), (third, second, 3)]:
        assert previous["operations"][-3:] == ["bond", "free_anneal", "free_cooldown"]
        assert event["lower_child"]["after_event"] == previous["event"]
        assert event["preparations"]["upper"] == []
        prep, = event["preparations"]["lower"]
        assert prep["wafer"] == wafer
        assert prep["after_free_anneal_cool_event"] == previous["event"]
        assert prep["operations"] == ["thin_mating_back", "grow_stressfree_back_BI"]
        assert prep["source_amplitude_unchanged"] is True
        assert event["operations"] == ["prepare_lower_mating_back", "bond", "free_anneal", "free_cooldown"]
    finalization = route["finalization"]
    assert finalization["ordering_status"] == "CONFIRMED"
    assert [(op["operation"], op["wafer"]) for op in finalization["operations"]] == [
        ("thin_outer_back_without_new_BI", 4), ("attach_temporary_carrier", 4),
        ("thin_outer_back_without_new_BI", 1)]
    assert finalization["carrier_end_state"] == "UNSPECIFIED"
    assert route["final_ledger"]["thermal_cycle_counts"] == [3, 3, 2, 1]
    assert route["final_ledger"]["added_back_bi_thermal_cycle_counts"] == {"2": 2, "3": 1}


def test_balanced_both_children_prepared_only_after_their_own_cooldowns():
    route = order.build_plan(config(BALANCED))["routes"][0]
    first, second, join = route["events"]
    assert [event["bond_mode"] for event in route["events"]] == ["F2F", "F2F", "B2B"]
    for event in (first, second):
        assert event["preparations"] == dict(lower=[], upper=[])
        assert event["operations"] == ["bond", "free_anneal", "free_cooldown"]
    assert join["mating_faces"] == dict(lower="BACK", upper="BACK")
    assert join["lower_child"] == dict(wafer_ids=[1, 2], after_event="A1")
    assert join["upper_child"] == dict(wafer_ids=[3, 4], after_event="A2")
    for role, wafer, after in [("lower", 2, "A1"), ("upper", 3, "A2")]:
        prep, = join["preparations"][role]
        assert (prep["wafer"], prep["after_free_anneal_cool_event"]) == (wafer, after)
    assert join["operations"] == ["prepare_lower_mating_back", "prepare_upper_mating_back",
                                   "bond", "free_anneal", "free_cooldown"]
    finalization = route["finalization"]
    assert finalization["ordering_status"] == "UNSPECIFIED"
    assert finalization["operations"] == []
    assert {op["wafer"] for op in finalization["required_operations"]} == {1, 4}
    assert finalization["temporary_carrier_ordering"] == "UNSPECIFIED"
    assert route["final_ledger"]["thermal_cycle_counts"] == [2, 2, 2, 2]
    assert route["final_ledger"]["added_back_bi_thermal_cycle_counts"] == {"2": 1, "3": 1}
    assert "BALANCED_OUTER_THINNING_CARRIER_ORDERING_UNSPECIFIED" in route["blockers"]


@pytest.mark.parametrize("target,inner,outer", [(30., 17., 18.5), (50., 37., 38.5), (80., 67., 68.5)])
def test_final_all_target_inner_back_bi_and_outer_without_new_bi(target, inner, outer):
    plan = order.build_plan(config(target=target))
    for route in plan["routes"]:
        if route["schema"] != order.MIXED_ROUTE_SCHEMA:
            continue
        final = route["final_ledger"]
        assert final["status"] == "SYMBOLIC_TARGET_NOT_SOLVED"
        assert final["total_thicknesses_um"] == [target] * 4
        assert final["retained_si_thicknesses_um"] == [outer, inner, inner, outer]
        assert final["new_back_bi_thicknesses_um"] == [0., 1.5, 1.5, 0.]
        for event in route["events"]:
            for preps in event["preparations"].values():
                for prep in preps:
                    assert prep["from_total_um"] == 775.
                    assert (prep["to_total_um"], prep["retained_si_um"], prep["new_back_bi_um"]) == (target, inner, 1.5)
        finalization = route["finalization"]
        ops = finalization.get("required_operations", finalization["operations"])
        for op in ops:
            if op["operation"] == "thin_outer_back_without_new_BI":
                assert op["after_free_anneal_cool_event"] == "A3"
                assert (op["to_total_um"], op["retained_si_um"], op["new_back_bi_um"]) == (target, outer, 0.)


def test_interface_identity_is_neutral_and_overlay_index_is_physical_not_event_order():
    plan = order.build_plan(config())
    for expression in (SEQUENTIAL, BALANCED):
        events = routes(plan)[expression]["events"]
        assert {event["interface_id"]: event["physical_interface_index"] for event in events} == {
            "1--2": 0, "2--3": 1, "3--4": 2}
        assert all(event["anneal_support"] == event["cooldown_support"] == "free" for event in events)
    assert [event["physical_interface_index"] for event in routes(plan)[BALANCED]["events"]] == [0, 2, 1]


@pytest.mark.parametrize("field,value", [
    ("final", {}), ("state_id", "fake"), ("final_ledger", {}), ("predictions", {}),
    ("orientation_proposal", {}), ("events", [{"event": "A1"}]), ("status", "COMPLETE"),
    ("schema", order.MIXED_ROUTE_SCHEMA),
])
def test_placeholder_forbidden_fields_rejected_even_when_hashes_recomputed(field, value):
    plan = order.build_plan(config("(1,(2,(3,4)))"))
    plan["routes"][0][field] = value
    resign(plan)
    with pytest.raises(ValueError, match="invalid or modified mixed-face"):
        order.validate_mixed_plan(plan)


@pytest.mark.parametrize("mutation", [
    "confirm_orientation", "alter_orientation", "wrong_face", "skip_preparation", "wrong_prior_cycle",
    "wrong_physical_index", "outer_bi", "wrong_cycles", "carrier_order", "summary", "context", "birth_q6",
    "extra_root_field", "boolean_index", "readiness", "ranked_subset",
])
def test_semantic_tampering_rejected_even_when_all_hashes_recomputed(mutation):
    plan = order.build_plan(config(SEQUENTIAL))
    route = plan["routes"][0]
    if mutation == "confirm_orientation":
        route["orientation_proposal"]["status"] = "CONFIRMED"
    elif mutation == "alter_orientation":
        route["orientation_proposal"]["thickness_orientation"][0] = 1
    elif mutation == "wrong_face":
        route["events"][0]["mating_faces"]["lower"] = "BACK"
    elif mutation == "skip_preparation":
        route["events"][1]["preparations"]["lower"] = []
    elif mutation == "wrong_prior_cycle":
        route["events"][1]["preparations"]["lower"][0]["after_free_anneal_cool_event"] = "A2"
    elif mutation == "wrong_physical_index":
        route["events"][1]["physical_interface_index"] = 2
    elif mutation == "outer_bi":
        route["finalization"]["operations"][0]["new_back_bi_um"] = 1.5
    elif mutation == "wrong_cycles":
        route["final_ledger"]["thermal_cycle_counts"] = [2, 2, 2, 2]
    elif mutation == "carrier_order":
        route["finalization"]["operations"].reverse()
    elif mutation == "summary":
        plan["summary"]["defined_routes"] = 5
    elif mutation == "context":
        plan["context"]["frame_id"] = order.FRAME
    elif mutation == "birth_q6":
        plan["initialization"]["wafers"][0]["initial_q6"][0] += .001
    elif mutation == "boolean_index":
        route["events"][1]["physical_interface_index"] = True
    elif mutation == "readiness":
        plan["native_execution_available"] = True
    else:
        plan["best_supplied_orders" if mutation == "ranked_subset" else "extra"] = []
    resign(plan)
    with pytest.raises(ValueError, match="invalid or modified mixed-face"):
        order.validate_mixed_plan(plan)


def test_gate_blocks_execution_and_legacy_archive_use_before_scoring(monkeypatch):
    plan = order.build_plan(config())
    def no_overlay(*args, **kwargs):
        pytest.fail("v2 must be rejected before overlay scoring")
    monkeypatch.setattr(order, "_overlay_settings", no_overlay)
    for call in [lambda: order._check_plan(plan), lambda: order.require_executable_plan(plan),
                 lambda: order.archive_template(plan), lambda: order.evaluate_archive(plan, {}, {})]:
        with pytest.raises(ValueError, match="plan-only: pending process qualification"):
            call()
    legacy = order._build_legacy_plan(config())
    assert legacy["schema"] == order.PLAN_SCHEMA == "yap.w2w_assembly_order_plan.v1"
    order._check_plan(legacy)
    order.require_executable_plan(legacy)
    assert order.archive_template(legacy)["schema"] == order.ARCHIVE_SCHEMA
    with pytest.raises(ValueError, match="legacy archives are not process ledgers"):
        order.validate_mixed_plan(legacy)
    with pytest.raises(ValueError, match="legacy archives are not process ledgers"):
        order.validate_mixed_plan(order.archive_template(legacy))


@pytest.mark.parametrize("selection", [[], [SEQUENTIAL, SEQUENTIAL], "((1,3),(2,4))", True])
def test_invalid_order_requests_rejected(selection):
    with pytest.raises(ValueError):
        order.build_plan(config(selection))


def test_no_thinning_and_unbounded_public_processes_rejected():
    source = config()
    source["geometry"]["target_total_thickness_um"] = 775.
    with pytest.raises(ValueError, match="W2W requires thinning"):
        order.build_plan(source)
    source = config()
    source["initial_bows_um"] = [1., 2., 3.]
    with pytest.raises(ValueError, match="exactly four wafers"):
        order.build_plan(source)
    assert len(order._build_legacy_plan(source)["routes"]) == 2


@pytest.mark.parametrize("name,target", [("design_6_4hi_p2", 50.), ("design_6_4hi_p6_reference", 51.5)])
def test_public_yaml_remains_v1_compatible_with_v2_plan_only(name, target):
    design = order.load_stack_input(ROOT / "W2W/input" / name / "stack.yaml")
    assert design["schema"] == order.STACK_SCHEMA
    assert design["stack"]["schema"] == order.INPUT_SCHEMA
    assert design["stack"]["geometry"]["target_total_thickness_um"] == target
    assert design["mechanics"]["operator_package"].startswith("resource:")
    plan = order.build_plan(design["stack"])
    assert plan["summary"]["missing_process_definitions"] == 3
    order.validate_mixed_plan(json.loads(json.dumps(plan)))


def test_plan_only_cli_outputs_counts_without_native_encouragement(tmp_path, capsys):
    output = tmp_path / "plan.json"
    order.main(["--config", str(INPUT), "--plan-only", "--output", str(output)])
    printed = capsys.readouterr().out
    assert "PLAN_ONLY_PARTIAL_PROCESS_DEFINITIONS" in printed
    assert "2 defined process ledgers; 3 missing process definitions" in printed
    assert "--native" not in printed
    assert "Native histories complete" not in printed
    order.validate_mixed_plan(json.loads(output.read_text()))


@pytest.mark.parametrize("mode", ["native_yaml", "native_json", "archive", "template"])
def test_cli_rejects_nonplan_paths_before_external_reads_or_publication(mode, monkeypatch, tmp_path, capsys):
    output = tmp_path / "forbidden-output.json"
    archive_output = tmp_path / "forbidden-template.json"
    source = tmp_path / "input.json"
    source.write_text(json.dumps(config()))
    real_load = order._load
    def only_source(path):
        assert Path(path) == source, "external settings/archive must not be read for v2"
        return real_load(path)
    monkeypatch.setattr(order, "_load", only_source)
    args = ["--output", str(output)]
    if mode == "native_yaml":
        args += ["--config", str(INPUT), "--native-mechanics"]
    elif mode == "native_json":
        args += ["--config", str(source), "--native-mechanics", str(tmp_path / "missing-settings.json")]
    elif mode == "archive":
        args += ["--config", str(source), "--mechanics-archive", str(tmp_path / "missing-archive.json"),
                 "--overlay-input", str(tmp_path / "missing-overlay.json")]
    else:
        args += ["--config", str(INPUT), "--plan-only", "--archive-template", str(archive_output)]
    with pytest.raises(SystemExit) as exc:
        order.main(args)
    assert exc.value.code == 2
    assert "plan-only: pending process qualification" in capsys.readouterr().err
    assert not output.exists()
    assert not archive_output.exists()
