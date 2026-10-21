"""Protocol tests use real, lightweight subprocesses, never a mechanics solver."""

import os
import signal
from pathlib import Path
import subprocess
import sys

import pytest

from W2W import assembly_parallel as parallel


OUTPUT = parallel.PROJECT / "user_runs"
CHILD = r'''
import sys, time
from W2W import assembly_parallel as p
path = p.Path(sys.argv[1])
m = p._read(path)
worker = m["worker_id"]
print("fixture worker", worker, flush=True)
if worker == 0:
    time.sleep(.15)
proof = dict(operator_inputs={"fixture": "pinned"}, material_card={"sha256": "fixed"},
             corrected_material={"source_sha256": "same", "warmup_seconds": worker + 1})
header = dict(schema="fixture.archive", context_id=m["plan"]["context_id"],
    provenance=dict(implementation_sha256="same", operator_set_sha256=str(worker)),
    native_configuration=dict(operator_provenance=proof, temporal_profile="fixture", metric={"model": "explicit"}))
for index in m["route_indices"]:
    route = dict(m["plan"]["routes"][index])
    p._atomic(path.parent / f"route_{index:06d}.json", p._envelope(m, "route",
        route_index=index, request_id=route["request_id"], header=header, route=route))
if m["settings"].get("fixture_interrupt"):
    time.sleep(60)
if m["settings"].get("fixture_bad_run"):
    m["run_id"] = "wrong"
if not m["settings"].get("fixture_no_terminal"):
    p._atomic(path.parent / "terminal.json", p._envelope(m, "terminal", status="SUCCESS",
        route_indices=m["route_indices"], runtime_sources=m["source_pins"],
        route_sha256={str(i): p._hash(path.parent / f"route_{i:06d}.json") for i in m["route_indices"]}))
'''


@pytest.fixture
def harness(monkeypatch, tmp_path):
    assert tmp_path.resolve().is_relative_to(OUTPUT), "use canonical --basetemp for these tests"
    import W2W.assembly_order as orders
    monkeypatch.setattr(orders, "_check_plan", lambda _: None)
    monkeypatch.setattr(orders, "archive_template", lambda plan: dict(
        schema="fixture.archive", routes=[dict(route) for route in plan["routes"]]))
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(4)))
    gates = []
    monkeypatch.setattr(parallel, "_memory_gate", lambda minimum: gates.append(minimum))
    monkeypatch.setattr(parallel, "POLL_SECONDS", .01)
    original = subprocess.Popen
    children = []

    def launch(command, **kwargs):
        env = kwargs["env"]
        for key in ("HOME", "TMPDIR", "YAP_WORK_DIR", "NUMBA_CACHE_DIR", "PYTHONPYCACHEPREFIX"):
            assert Path(env[key]).is_relative_to(tmp_path)
        assert env["OPENBLAS_NUM_THREADS"] == env["OMP_NUM_THREADS"] == env["NUMBA_NUM_THREADS"] == "1"
        child = original([sys.executable, "-B", "-c", CHILD, command[-1]], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(parallel.subprocess, "Popen", launch)
    plan = dict(context_id="context", routes=[dict(request_id=f"request-{i}", order=str(i)) for i in range(4)])
    return plan, tmp_path / "workers", children, gates


def test_ordered_callbacks_static_pins_and_worker_sources(harness):
    plan, root, children, gates = harness
    seen = []

    def completed(archive, route):
        assert archive["provenance"]["operator_set_sha256"] == str(int(route["order"]) % 2)
        seen.append(route["order"])

    result = parallel.run_parallel(plan, dict(order_workers=2, metric={"model": "explicit"}),
        work_dir=root, minimum_memory_gib=2, route_complete=completed)
    assert seen == ["0", "1", "2", "3"]
    assert result["routes"] == plan["routes"]
    assert result["status"] == "COMPLETE"
    config = result["native_configuration"]
    assert config["order_execution"]["effective_workers"] == 2
    assert len(config["order_execution"]["per_worker_configuration"]) == 2
    assert config["worker_runtime_sources"]
    assert gates[0] == 4 and all(gate == 2 for gate in gates[1:])
    assert all(child.poll() == 0 for child in children)


@pytest.mark.parametrize("option,error", [("fixture_no_terminal", FileNotFoundError),
                                         ("fixture_bad_run", ValueError)])
def test_terminal_required_and_bound_to_run(harness, option, error):
    plan, root, children, _ = harness
    with pytest.raises(error):
        parallel.run_parallel(plan, dict(order_workers=2, **{option: True}), work_dir=root)
    assert all(child.poll() is not None for child in children)
    assert (root / "worker_0000/worker.log").exists()


def test_callback_interrupt_reaps_live_workers(harness):
    plan, root, children, _ = harness

    def interrupted(*_):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        parallel.run_parallel(plan, dict(order_workers=2, fixture_interrupt=True),
                              work_dir=root, route_complete=interrupted)
    assert len(children) == 2 and all(child.poll() is not None for child in children)


def test_parent_sigterm_reaps_workers_and_restores_handler(harness):
    plan, root, children, _ = harness
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(*_):
        signal.raise_signal(signal.SIGTERM)

    with pytest.raises(SystemExit) as stopped:
        parallel.run_parallel(plan, dict(order_workers=2, fixture_interrupt=True),
                              work_dir=root, route_complete=interrupted)
    assert stopped.value.code == 128 + signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) is previous
    assert len(children) == 2 and all(child.poll() is not None for child in children)


def test_worker_deadline_reaps_shards(harness, monkeypatch):
    plan, root, children, _ = harness
    monkeypatch.setattr(parallel, "MAX_WORKER_SECONDS", .02)
    with pytest.raises(TimeoutError, match="exceeded"):
        parallel.run_parallel(plan, dict(order_workers=2, fixture_interrupt=True), work_dir=root)
    assert all(child.poll() is not None for child in children)


def test_cpu_admission_before_launch(harness, monkeypatch):
    plan, root, children, _ = harness
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0})
    with pytest.raises(RuntimeError, match="CPU affinity"):
        parallel.run_parallel(plan, dict(order_workers=2), work_dir=root)
    assert not children and not root.exists()


def test_memory_admission_before_launch(harness, monkeypatch):
    plan, root, children, _ = harness

    def insufficient(minimum):
        assert minimum == 80
        raise RuntimeError("insufficient RAM")

    monkeypatch.setattr(parallel, "_memory_gate", insufficient)
    with pytest.raises(RuntimeError, match="RAM"):
        parallel.run_parallel(plan, dict(order_workers=2), work_dir=root)
    assert not children and not root.exists()


@pytest.mark.parametrize("indices", [[True], [-1], [0, 0], [4], []])
def test_invalid_indices(indices):
    with pytest.raises(ValueError, match="route_indices"):
        parallel._indices(indices, 4)


def test_worker_environment_overrides_inherited_storage(tmp_path, monkeypatch):
    assert tmp_path.resolve().is_relative_to(OUTPUT)
    for key in ("HOME", "TMPDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR", "HF_HUB_CACHE"):
        monkeypatch.setenv(key, "/tmp/forbidden")
    env = parallel._environment(tmp_path)
    for key in ("HOME", "TMPDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR", "HF_HUB_CACHE"):
        assert Path(env[key]).is_relative_to(tmp_path)


def test_source_change_is_rejected(tmp_path):
    assert tmp_path.resolve().is_relative_to(OUTPUT)
    with pytest.raises(ValueError, match="source changed"):
        parallel._check_sources({str(Path(parallel.__file__).resolve()): "not-the-source-hash"})


def test_worker_keeps_full_plan_and_loads_once(harness, monkeypatch):
    from contextlib import nullcontext
    import mechanics_runtime
    import threadpoolctl
    import W2W.assembly_order as orders
    import W2W.assembly_mechanics as mechanics
    import W2W.assembly_operator_library as operators

    plan, root, _, _ = harness
    plan["context"] = {"geometry": {}}
    root.mkdir()
    for key, value in parallel._environment(root).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(mechanics_runtime, "configure_runtime_storage", lambda: None)
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", lambda **_: nullcontext())
    for name in ("is_die_plan", "is_f2b_plan", "is_prepared_plan"):
        monkeypatch.setattr(orders, name, lambda _: False)
    monkeypatch.setattr(mechanics, "_recipe", lambda *_: None)
    monkeypatch.setattr(mechanics, "_check_metric_policy", lambda *_: None)
    loaded = []

    def load(*args, **kwargs):
        loaded.append(args)
        return object(), {"fixture": True}

    def run(full_plan, _library, _provenance, *, route_indices, route_complete, **kwargs):
        assert full_plan == plan and route_indices == [0, 2]
        for index in route_indices:
            route_complete(dict(schema="fixture.archive", context_id=plan["context_id"]), plan["routes"][index])

    monkeypatch.setattr(operators, "load_assembly_library", load)
    monkeypatch.setattr(mechanics, "run_native", run)
    manifest = dict(run_id="run", worker_id=0, plan=plan, route_indices=[0, 2],
        settings=dict(operator_package="fixture", temporal_profile="fixture"), source_pins=parallel._sources())
    parallel._atomic(root / "manifest.json", manifest)
    parallel._worker(root / "manifest.json")
    terminal = parallel._read(root / "terminal.json")
    assert len(loaded) == 1 and terminal["status"] == "SUCCESS"
    assert terminal["route_indices"] == [0, 2]
