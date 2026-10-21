"""Order-worker configuration and dispatch contracts, without solving physics."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import Mock

from omegaconf import OmegaConf
import pytest

import generate_assembly_design as generator
from D2W import assembly_order as die_order
from W2W import assembly_design, assembly_mechanics, assembly_operator_library
from W2W import assembly_order as wafer_order


ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "D2W": ROOT / "D2W/input/design_6_4hi_p2/stack.yaml",
    "W2W": ROOT / "W2W/input/f2b_4hi50_p24_preview/stack.yaml",
}
MODULES = {"D2W": die_order, "W2W": wafer_order}
INVALID_WORKERS = (True, False, 1.0, 2.5, 0, -1, "2", None)


@pytest.fixture(params=("D2W", "W2W"))
def domain(request):
    return request.param


def design_and_plan(domain):
    module = MODULES[domain]
    design = module.load_stack_input(INPUTS[domain])
    return design, module.build_plan(design["stack"])


def parallel_stub(monkeypatch, function):
    module = ModuleType("W2W.assembly_parallel")
    module.run_parallel = function
    monkeypatch.setitem(sys.modules, module.__name__, module)


def forbid_parent_operators(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("parent attempted native/operator loading"))
    for name in ("load_assembly_library", "load_f2b_assembly_library", "load_prepared_die_library"):
        monkeypatch.setattr(assembly_operator_library, name, forbidden)
    monkeypatch.setattr(assembly_mechanics, "native", forbidden)
    return forbidden


def test_config_default_and_explicit_order_workers_preserve_stack(tmp_path, domain):
    original = OmegaConf.to_container(OmegaConf.load(INPUTS[domain]), resolve=True)
    original["mechanics"].pop("order_workers", None)
    path = tmp_path / "stack.yaml"
    plans = []
    for workers in (None, 1, 3):
        document = deepcopy(original)
        if workers is not None:
            document["mechanics"]["order_workers"] = workers
        OmegaConf.save(OmegaConf.create(document), path)
        loaded = MODULES[domain].load_stack_input(path)
        assert loaded["mechanics"].get("order_workers", 1) == (workers or 1)
        plans.append(MODULES[domain].build_plan(loaded["stack"]))
    assert plans[0] == plans[1] == plans[2]


@pytest.mark.parametrize("workers", INVALID_WORKERS)
def test_config_rejects_non_positive_or_non_integer_workers(tmp_path, domain, workers):
    design, _ = design_and_plan(domain)
    design["mechanics"]["order_workers"] = workers
    path = tmp_path / "stack.yaml"
    OmegaConf.save(OmegaConf.create(design), path)
    with pytest.raises(ValueError, match="order_workers"):
        MODULES[domain].load_stack_input(path)


def test_parallel_config_rejects_nested_layer_workers(tmp_path, domain):
    design, _ = design_and_plan(domain)
    design["mechanics"].update(order_workers=2, layer_workers=2)
    path = tmp_path / "stack.yaml"
    OmegaConf.save(OmegaConf.create(design), path)
    with pytest.raises(ValueError, match="layer_workers"):
        MODULES[domain].load_stack_input(path)


@pytest.mark.parametrize("workers", (None, 1, 3))
def test_generator_api_emits_order_workers(tmp_path, workers):
    options = dict(name="worker_contract", dies=2, pitch=2, total_thickness=50.,
                   initial_bows=[10., -10.], output_root=tmp_path / "generated")
    if workers is not None:
        options["order_workers"] = workers
    result = generator.generate_design(**options)
    design = die_order.load_stack_input(result["input_dir"] / "stack.yaml")
    assert design["mechanics"]["order_workers"] == (workers or 1)


@pytest.mark.parametrize("workers", INVALID_WORKERS)
def test_generator_api_refuses_invalid_workers_before_output(tmp_path, workers):
    output = tmp_path / "generated"
    with pytest.raises(ValueError, match="order_workers"):
        generator.generate_design(name="invalid_workers", dies=2, pitch=2,
            total_thickness=50., initial_bows=[0., 0.], output_root=output,
            order_workers=workers)
    assert not output.exists()


@pytest.mark.parametrize("flag,expected", (([], 1), (["--order-workers", "3"], 3)))
def test_generator_cli_forwards_order_workers(tmp_path, monkeypatch, flag, expected):
    generate = Mock(return_value=dict(input_dir=tmp_path, config_path=tmp_path / "config.yaml",
                                     orders=1, operator_package="component:fixture"))
    monkeypatch.setattr(generator, "generate_design", generate)
    assert generator.main(["--name", "cli_workers", "--dies", "2", "--pitch", "2",
        "--total-thickness", "50", "--initial-bows", "10", "-10", *flag]) == 0
    assert generate.call_args.kwargs["order_workers"] == expected


@pytest.mark.parametrize("value", ("true", "1.5", "0"))
def test_generator_cli_rejects_invalid_order_workers(tmp_path, value):
    output = tmp_path / "generated"
    with pytest.raises(SystemExit) as error:
        generator.main(["--name", "invalid_cli", "--dies", "2", "--pitch", "2",
            "--total-thickness", "50", "--initial-bows", "0", "0",
            "--order-workers", value, "--output-root", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_settings_parallel_dispatch_never_loads_parent_operators(tmp_path, monkeypatch, domain):
    design, plan = design_and_plan(domain)
    settings = dict(design["mechanics"], order_workers=3)
    forbidden = forbid_parent_operators(monkeypatch)
    monkeypatch.setattr(assembly_operator_library, "require_prepared_operator", Mock())
    monkeypatch.setattr(assembly_mechanics, "_configure_mechanics_runtime_environment", Mock())
    monkeypatch.setattr(assembly_mechanics, "_recipe", Mock())
    result = object()
    dispatch = Mock(return_value=result)
    parallel_stub(monkeypatch, dispatch)
    progress = Mock()
    before = deepcopy(plan)
    assert assembly_mechanics.run_native_from_settings(plan, settings, progress=progress,
        worker_directory=tmp_path / "workers") is result
    forbidden.assert_not_called()
    assert dispatch.call_args.args == (plan, settings)
    assert dispatch.call_args.kwargs["work_dir"] == tmp_path / "workers"
    assert dispatch.call_args.kwargs["progress"] is progress
    assert plan == before


@pytest.mark.parametrize("explicit", (False, True))
def test_settings_serial_keeps_existing_loader_and_runner(monkeypatch, domain, explicit):
    design, plan = design_and_plan(domain)
    settings = dict(design["mechanics"])
    settings.pop("order_workers", None)
    if explicit:
        settings["order_workers"] = 1
    monkeypatch.setattr(assembly_operator_library, "require_prepared_operator", Mock())
    configure = Mock()
    monkeypatch.setattr(assembly_mechanics, "_configure_mechanics_runtime_environment", configure)
    monkeypatch.setattr(assembly_mechanics, "_recipe", Mock())
    library, provenance, result = object(), {"fixture": "not_physics"}, object()
    loader = Mock(return_value=(library, provenance))
    for name in ("load_assembly_library", "load_f2b_assembly_library", "load_prepared_die_library"):
        monkeypatch.setattr(assembly_operator_library, name,
            loader if name == ("load_prepared_die_library" if domain == "D2W" else
                               "load_f2b_assembly_library") else Mock(side_effect=AssertionError("wrong loader")))
    native_run = Mock(return_value=result)
    monkeypatch.setattr(assembly_mechanics, "run_native", native_run)
    dispatch = Mock(side_effect=AssertionError("serial execution dispatched subprocesses"))
    parallel_stub(monkeypatch, dispatch)
    progress = Mock()
    assert assembly_mechanics.run_native_from_settings(plan, settings, progress=progress) is result
    dispatch.assert_not_called()
    configure.assert_called_once_with(layer_workers=settings.get("layer_workers", 1))
    loader.assert_called_once_with(settings["operator_package"], plan["context"]["geometry"],
        **({"frozen_preview": True} if settings.get("frozen_preview", False) else {}))
    native_run.assert_called_once_with(plan, library, provenance,
        temporal_profile=settings["temporal_profile"], metric=settings.get("metric"), progress=progress)


@pytest.mark.parametrize("workers", INVALID_WORKERS)
def test_settings_reject_invalid_order_workers_before_loading(monkeypatch, domain, workers):
    design, plan = design_and_plan(domain)
    forbidden = forbid_parent_operators(monkeypatch)
    with pytest.raises(ValueError, match="order_workers"):
        assembly_mechanics.run_native_from_settings(plan, dict(design["mechanics"], order_workers=workers))
    forbidden.assert_not_called()


def test_settings_reject_parallel_layer_workers_before_loading(monkeypatch, domain):
    design, plan = design_and_plan(domain)
    forbidden = forbid_parent_operators(monkeypatch)
    with pytest.raises(ValueError, match="layer_workers"):
        assembly_mechanics.run_native_from_settings(plan,
            dict(design["mechanics"], order_workers=2, layer_workers=2))
    forbidden.assert_not_called()


@pytest.mark.parametrize("workers", (None, 1, 3))
def test_design_runner_dispatch_callbacks_and_worker_source_evidence(tmp_path, monkeypatch, domain, workers):
    design, plan = design_and_plan(domain)
    design["mechanics"].pop("order_workers", None)
    if workers is not None:
        design["mechanics"]["order_workers"] = workers
    archive = wafer_order.archive_template(plan)
    archive.update(status="COMPLETE", native_configuration={})
    worker_source = ROOT / "initial_bow_distribution.py"
    source_hash = hashlib.sha256(worker_source.read_bytes()).hexdigest()
    if workers == 3:
        archive["native_configuration"]["worker_runtime_sources"] = {str(worker_source): source_hash}
    provenance = {"input_sha256": {}}
    design_loader = Mock(return_value=(design, plan, {"fixture": "not_physics"}, {}, provenance))
    monkeypatch.setattr(assembly_design, "_memory_gate", Mock(return_value=128.))
    monkeypatch.setattr(assembly_mechanics, "_recipe", Mock())
    forbidden = forbid_parent_operators(monkeypatch)
    operator_loader = Mock(return_value=(object(), {"fixture": "not_physics"}))
    completed_ids = []

    def execute(request, *_args, progress, route_complete, **_kwargs):
        assert request == plan
        progress("Fixture execution; no physics.")
        for route in archive["routes"]:
            route_complete(archive, route)
            completed_ids.append(route["request_id"])
        return archive

    parallel = Mock(side_effect=execute if workers == 3 else AssertionError("unexpected parallel branch"))
    parallel_stub(monkeypatch, parallel)
    serial = Mock(side_effect=execute if workers != 3 else AssertionError("unexpected serial branch"))
    monkeypatch.setattr(assembly_mechanics, "run_native", serial)

    def evaluate(request, result, overlay):
        assert request == plan and result is archive
        assert overlay == {"fixture": "not_physics"}
        assert completed_ids == [r["request_id"] for r in plan["routes"]]
        return dict(routes=[dict(order=r["order"], interfaces=[]) for r in archive["routes"]],
                    best_supplied_orders=[archive["routes"][0]["order"]], physical_readiness=False)

    monkeypatch.setattr(assembly_design, "evaluate_archive", evaluate)
    run = assembly_design.run_design({}, INPUTS[domain].parent, "fixture_workers",
        output_dir=tmp_path / "runs", design_loader=design_loader, operator_loader=operator_loader)
    forbidden.assert_not_called()
    if workers == 3:
        operator_loader.assert_not_called()
        serial.assert_not_called()
        assert parallel.call_args.kwargs["work_dir"] == run / "order_workers"
        assert parallel.call_args.kwargs["minimum_memory_gib"] == 40.
        sources = json.loads((run / "RUNTIME_SOURCES.json").read_text())
        assert sources[str(worker_source)] == source_hash
        assert (run / "source_snapshot/YAP/initial_bow_distribution.py").read_bytes() == worker_source.read_bytes()
    else:
        parallel.assert_not_called()
        operator_loader.assert_called_once()
        serial.assert_called_once()
    for index, expected in enumerate(archive["routes"], 1):
        partial = json.loads((run / f"route_{index}_mechanics.json").read_text())
        assert partial["routes"] == [expected]
    assert (run / "SUMMARY.csv").exists()
    assert json.loads((run / "RESULT.json").read_text())["status"] == "NATIVE_MECHANICS_OVERLAY_UNVALIDATED"
    assert not (run / "FAILED.json").exists()


def test_design_parallel_failure_never_publishes_complete_ranking(tmp_path, monkeypatch, domain):
    design, plan = design_and_plan(domain)
    design["mechanics"]["order_workers"] = 2
    monkeypatch.setattr(assembly_design, "_memory_gate", Mock(return_value=128.))
    monkeypatch.setattr(assembly_mechanics, "_recipe", Mock())
    forbidden = forbid_parent_operators(monkeypatch)
    parallel_stub(monkeypatch, Mock(side_effect=RuntimeError("fixture worker failure")))
    score = Mock(side_effect=AssertionError("failed mechanics were scored"))
    monkeypatch.setattr(assembly_design, "evaluate_archive", score)
    with pytest.raises(RuntimeError, match="fixture worker failure"):
        assembly_design.run_design({}, INPUTS[domain].parent, "failed_workers",
            output_dir=tmp_path / "runs",
            design_loader=lambda *_: (design, plan, {}, {}, {"input_sha256": {}}))
    forbidden.assert_not_called()
    score.assert_not_called()
    runs = list((tmp_path / "runs/failed_workers").iterdir())
    assert len(runs) == 1
    run = runs[0]
    assert json.loads((run / "FAILED.json").read_text())["status"] == "FAILED_NO_COMPLETE_RANKING"
    for name in ("MECHANICS.json", "RESULT.json", "SUMMARY.csv"):
        assert not (run / name).exists()
