"""User-storage protocol checks with stdlib-only children, never Cu mechanics."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from W2W import assembly_parallel as parallel


CHILD = r'''
import os, sys, time
from W2W import assembly_parallel as p
path = p.Path(sys.argv[1])
manifest = p._read(path)
root = p.Path(os.environ["YAP_RUN_ROOT"])
p._check_sources(manifest["source_pins"])
assert all(not p.Path(name).is_relative_to(root) for name in manifest["source_pins"])
keys = ("HOME", "YAP_WORK_DIR", "TMPDIR", "TMP", "TEMP", "NUMBA_CACHE_DIR",
        "MPLCONFIGDIR", "PYTHONPYCACHEPREFIX", "PIP_CACHE_DIR", "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "HF_HOME")
locations = {key: os.environ[key] for key in keys}
assert all(p.Path(value).is_relative_to(path.parent) for value in locations.values())
assert p.Path(os.environ["YAP_W2W_SI_CAP_CACHE_DIR"]).is_relative_to(root)
assert not any(name in sys.modules for name in ("numpy", "history_mechanics_backend",
                                               "W2W.assembly_mechanics"))
p._atomic(path.parent / "storage_probe.json", locations)
if manifest["worker_id"] == 0:
    time.sleep(.05)
proof = dict(operator_inputs={"fixture": "not_physics"}, material_card={"sha256": "fixed"})
header = dict(schema="fixture.archive", context_id=manifest["plan"]["context_id"],
    provenance=dict(implementation_sha256="same", operator_set_sha256="same"),
    native_configuration=dict(operator_provenance=proof, temporal_profile="fixture",
                              metric={"model": "explicit"}))
for index in manifest["route_indices"]:
    route = dict(manifest["plan"]["routes"][index])
    p._atomic(path.parent / f"route_{index:06d}.json", p._envelope(manifest, "route",
        route_index=index, request_id=route["request_id"], header=header, route=route))
p._atomic(path.parent / "terminal.json", p._envelope(manifest, "terminal", status="SUCCESS",
    route_indices=manifest["route_indices"], runtime_sources=manifest["source_pins"],
    route_sha256={str(index): p._hash(path.parent / f"route_{index:06d}.json")
                  for index in manifest["route_indices"]}))
print("mock protocol complete; no mechanics", flush=True)
'''


@pytest.fixture
def user_protocol(tmp_path, monkeypatch):
    assert tmp_path.is_relative_to("/app"), "run with a private /app --basetemp"
    assert not tmp_path.is_relative_to(parallel.REPO), "test outputs must be outside source"
    assert tmp_path.stat().st_mode & 0o777 == 0o700
    monkeypatch.setenv("YAP_RUN_ROOT", str(tmp_path))
    monkeypatch.delenv("YAP_W2W_SI_CAP_CACHE_DIR", raising=False)
    orders = ModuleType("W2W.assembly_order")
    orders._check_plan = lambda _: None
    orders.archive_template = lambda plan: dict(
        schema="fixture.archive", routes=[dict(route) for route in plan["routes"]])
    monkeypatch.setitem(sys.modules, "W2W.assembly_order", orders)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0, 1})
    monkeypatch.setattr(parallel, "_memory_gate", lambda _: None)
    monkeypatch.setattr(parallel, "POLL_SECONDS", .01)
    monkeypatch.setattr(parallel, "MAX_WORKER_SECONDS", 10.)
    plan = dict(context_id="fixture-context", routes=[
        dict(request_id=f"request-{index}", order=str(index)) for index in range(4)])
    return tmp_path, plan


def test_real_mock_workers_keep_writes_private_and_check_external_source_pins(user_protocol, monkeypatch):
    root, plan = user_protocol
    shared = parallel.PROJECT / "YAP/mechanics_runtime.py"
    sources = parallel._sources()
    sources[str(shared)] = parallel._hash(shared)
    monkeypatch.setattr(parallel, "_sources", lambda: dict(sources))
    original = subprocess.Popen
    children = []

    def launch(command, **kwargs):
        assert Path(kwargs["cwd"]) == parallel.REPO
        assert kwargs["env"]["YAP_RUN_ROOT"] == str(root)
        child = original([sys.executable, "-B", "-c", CHILD, command[-1]], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(parallel.subprocess, "Popen", launch)
    seen = []
    result = parallel.run_parallel(plan, dict(order_workers=2, metric={"model": "explicit"}),
        work_dir=root / "workers", minimum_memory_gib=1.,
        route_complete=lambda _archive, route: seen.append(route["order"]))
    assert seen == ["0", "1", "2", "3"]
    assert result["status"] == "COMPLETE"
    assert result["physical_readiness"] is False
    assert len(children) == 2 and all(child.poll() == 0 for child in children)
    assert result["native_configuration"]["worker_runtime_sources"] == sources
    assert all(not Path(name).is_relative_to(root) for name in sources)
    for worker in range(2):
        shard = root / "workers" / f"worker_{worker:04d}"
        manifest = json.loads((shard / "manifest.json").read_text())
        assert manifest["source_pins"] == sources
        terminal = json.loads((shard / "terminal.json").read_text())
        assert terminal["status"] == "SUCCESS"
        assert terminal["runtime_sources"] == sources
        probe = json.loads((shard / "storage_probe.json").read_text())
        assert all(Path(value).is_relative_to(shard) for value in probe.values())
        assert "no mechanics" in (shard / "worker.log").read_text()
        for index in manifest["route_indices"]:
            route = shard / f"route_{index:06d}.json"
            assert terminal["route_sha256"][str(index)] == parallel._hash(route)


@pytest.mark.parametrize("entry", ["atomic", "environment", "parallel", "worker"])
def test_missing_run_root_refuses_before_any_write_or_child(user_protocol, monkeypatch, entry):
    root, plan = user_protocol
    monkeypatch.delenv("YAP_RUN_ROOT")
    calls = {
        "atomic": lambda: parallel._atomic(root / "new.json", {}),
        "environment": lambda: parallel._environment(root / "worker"),
        "parallel": lambda: parallel.run_parallel(plan, dict(order_workers=2), work_dir=root / "workers"),
        "worker": lambda: parallel._worker(root / "manifest.json"),
    }
    with monkeypatch.context() as guarded:
        spies = [Mock() for _ in range(4)]
        guarded.setattr(Path, "mkdir", spies[0])
        guarded.setattr(Path, "open", spies[1])
        guarded.setattr(os, "replace", spies[2])
        guarded.setattr(parallel.subprocess, "Popen", spies[3])
        with pytest.raises(ValueError, match="YAP_RUN_ROOT"):
            calls[entry]()
        for spy in spies:
            spy.assert_not_called()


def test_user_run_root_is_not_a_source_allowlist(user_protocol, monkeypatch):
    root, _ = user_protocol
    untrusted = root / "untrusted.py"
    module = ModuleType("fixture_user_output_not_code")
    module.__file__ = str(untrusted)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert str(untrusted) not in parallel._sources()
    with pytest.raises(ValueError, match="run outputs|outside"):
        parallel._check_sources({str(untrusted): "0" * 64})
    source = Path(parallel.__file__).resolve()
    assert parallel._source_path(source) == source
    with pytest.raises(ValueError, match="YAP_RUN_ROOT|read-only"):
        parallel._canonical(source)
    with pytest.raises(ValueError, match="source changed"):
        parallel._check_sources({str(source): "0" * 64})


def test_snapshot_namespaces_use_checkout_root_not_user_parent(user_protocol):
    root, _ = user_protocol
    source = parallel.REPO / "W2W/assembly_design.py"
    shared = parallel.PROJECT / "operator_repair_20261015/source/material.py"
    assert not source.is_relative_to(root)
    assert parallel._source_relative(source) == Path("YAP/W2W/assembly_design.py")
    assert parallel._source_relative(shared) == Path("shared_project/operator_repair_20261015/source/material.py")
    assert parallel._source_relative(source).parts[0] != parallel.REPO.parent.name


def test_native_snapshot_does_not_recreate_protected_directory_names(user_protocol):
    root, _ = user_protocol
    source = parallel.PROJECT / "Ansys_Verify/12_stack_model_validation/exp12_native.py"
    relative = parallel._source_relative(source)
    assert relative == Path("shared_validation/12_stack_model_validation/exp12_native.py")
    target = root / "source_snapshot" / relative
    assert parallel._canonical(target) == target
