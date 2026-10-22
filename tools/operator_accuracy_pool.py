"""Parallel p2 operator accuracy checks with a preserved serial-queue handoff."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.build_prepared_operators as builder
import tools.operator_grid as grid
import tools.operator_thermal_validation as thermal


PRECOMPUTE_SECONDS = 43200
HANDOFF_WAIT_SECONDS = 43200
PREPARATION_SECONDS = 3600
SPAWN_SIGNALS = {signal.SIGTERM, signal.SIGINT, signal.SIGALRM, signal.SIGHUP}


def partitions(records, slots):
    if type(slots) is not int or not 2 <= slots <= 4 or not slots <= len(records) <= 18:
        raise ValueError("pool requires 2--4 nonempty bounded queues")
    if len({r["key"] for r in records}) != len(records):
        raise ValueError("duplicate operator assignment")
    return [records[i::slots] for i in range(slots)]


def resources(work, slots, cases):
    result = builder.resource_check(work)
    if (result["available_memory_gib"] < 64 * slots
            or result["local_free_gib"] < 200 + 64 * cases
            or result["cpu_busy_fraction"] + 8 * slots / result["logical_cpus"] > .9):
        raise RuntimeError("insufficient aggregate pool resources: " + str(result))
    return result


def load_handoff(plan):
    path = grid.regular(plan["handoff_module"])
    if builder.digest(path) != plan["controllers_sha256"][str(path)]:
        raise ValueError("handoff source drift")
    spec = importlib.util.spec_from_file_location("operator_pool_handoff", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_pool(work, expected):
    work = builder.local_scratch(work)
    plan, _ = grid.read_record(work / "GROUP_PLAN.json", expected)
    if (plan.get("schema") != "yap.operator_grid.parallel_screens.v1"
            or plan.get("work") != str(work) or plan.get("execution_host") != builder.physical_host()
            or time.time() >= plan["expires_epoch"]):
        raise ValueError("pool plan policy drift or expiry")
    grid.group_slots(plan)
    native_seconds = max(len(job["candidates"]) for job in plan["jobs"]) * thermal.CASE_SECONDS
    if (plan.get("precompute_seconds") != PRECOMPUTE_SECONDS
            or plan.get("handoff_wait_seconds") != HANDOFF_WAIT_SECONDS
            or plan.get("preparation_seconds") != PREPARATION_SECONDS
            or plan.get("native_seconds") != native_seconds
            or plan.get("exit_grace_seconds") != thermal.EXIT_GRACE_SECONDS
            or plan["expires_epoch"] != plan["created_epoch"] + PREPARATION_SECONDS
                + PRECOMPUTE_SECONDS + HANDOFF_WAIT_SECONDS + native_seconds + thermal.EXIT_GRACE_SECONDS):
        raise ValueError("pool wall budget drift")
    for filename, digest in plan["controllers_sha256"].items():
        if builder.digest(grid.regular(filename)) != digest:
            raise ValueError("pool controller drift")
    return plan, work


def launch(args):
    host = builder.physical_host()
    if type(args.slots) is not int or not 2 <= args.slots <= grid.SCREEN_POOL_MAX_SLOTS[host]:
        raise ValueError("pool slots exceed physical host limit")
    work = builder.local_scratch(args.work)
    if work.exists():
        raise ValueError("fresh pool work required")
    legacy = builder.local_scratch(args.legacy_work)
    old, old_sha = grid.read_record(legacy / "PLAN.json")
    grid.verify_pins(legacy / "SOURCE_PINS.json", old["pins_sha256"])
    status, _ = grid.read_record(legacy / "STATUS.json")
    active_index = next((i for i, r in enumerate(old["candidates"])
                         if r["key"] == status.get("active_candidate")), None)
    if active_index is None:
        raise ValueError("legacy queue has no bound active candidate")
    next_index = active_index + 1
    queues = partitions(old["candidates"][next_index:], args.slots)
    resource = resources(work.parent, args.slots, sum(map(len, queues)))
    files = [Path(__file__).resolve(), Path(builder.__file__).resolve(), Path(grid.__file__).resolve(),
             Path(thermal.__file__).resolve(), grid.regular(args.handoff_module)]
    archive = grid.ARCHIVES / work.name
    if archive.exists() or not work.name.startswith("operator_grid_p2_parallel_"):
        raise ValueError("fresh canonical parallel archive required")
    created = time.time()
    native_seconds = max(map(len, queues)) * thermal.CASE_SECONDS
    plan = dict(schema="yap.operator_grid.parallel_screens.v1", **thermal.scope(),
        work=str(work), archive=str(archive), execution_host=host,
        legacy_work=str(legacy), legacy_plan_sha256=old_sha,
        handoff_module=str(files[-1]), resources=resource,
        controllers_sha256={str(p): builder.digest(p) for p in files},
        jobs=[dict(kind="screen", work=str(work / f"slot{i}"),
                   archive=str(grid.ARCHIVES / f"operator_grid_p2_thermal_{work.name}_slot{i}"),
                   controller=str(Path(thermal.__file__).resolve()), candidates=records)
              for i, records in enumerate(queues)],
        created_epoch=created, expires_epoch=created + PREPARATION_SECONDS + PRECOMPUTE_SECONDS
            + HANDOFF_WAIT_SECONDS + native_seconds + thermal.EXIT_GRACE_SECONDS,
        automatic_retry=False, preparation_seconds=PREPARATION_SECONDS,
        precompute_seconds=PRECOMPUTE_SECONDS, handoff_wait_seconds=HANDOFF_WAIT_SECONDS,
        native_seconds=native_seconds, exit_grace_seconds=thermal.EXIT_GRACE_SECONDS)
    handoff = load_handoff(plan)
    plan["handoff_receipt"] = handoff.arm(legacy)
    if plan["handoff_receipt"]["next_index"] != next_index:
        raise ValueError("legacy queue advanced during assignment; no dispatch")
    builder.configure(work)
    archive.mkdir()
    builder.write_json(work / "GROUP_PLAN.json", plan, immutable=True)
    sha = builder.digest(work / "GROUP_PLAN.json")
    builder.copy_checked(work / "GROUP_PLAN.json", archive / "GROUP_PLAN.json", sha)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "supervisor",
               "--work", str(work), "--plan-sha256", sha]
    observed = thermal.ProcessInventory()
    child = None
    try:
        with (work / "worker.log").open("x") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True, env=builder.configure(work))
        identity = grid.wait_exec_identity(child, command, observed)
        receipt = dict(worker_pid=child.pid, worker_identity=identity, command=command,
                       plan_sha256=sha, launched_epoch=time.time())
        builder.write_json(work / "LAUNCH.json", receipt, immutable=True)
        builder.copy_checked(work / "LAUNCH.json", archive / "LAUNCH.json")
        builder.write_json(work / "LAUNCH_PUBLISHED.json", dict(plan_sha256=sha,
            launch_sha256=builder.digest(work / "LAUNCH.json")), immutable=True)
        print(receipt, flush=True)
        return child
    except BaseException:
        if child is not None:
            thermal.cleanup_spawned(child, observed)
        raise


def job_plans(plan, work, binding):
    old, _ = grid.read_record(Path(plan["legacy_work"]) / "PLAN.json", plan["legacy_plan_sha256"])
    old_pins = grid.verify_pins(Path(plan["legacy_work"]) / "SOURCE_PINS.json", old["pins_sha256"])
    pins = thermal.closure(Path(old["source"]))
    for path, digest in old_pins["files_sha256"].items():
        if path in pins["files_sha256"] and pins["files_sha256"][path] != digest:
            raise ValueError("scientific source changed since serial preflight")
    results = []
    for slot, job in enumerate(plan["jobs"]):
        root, archive = Path(job["work"]), Path(job["archive"])
        root.mkdir()
        archive.mkdir()
        builder.write_json(root / "SOURCE_PINS.json", pins, immutable=True)
        child_plan = dict(old, work=str(root), archive=str(archive), candidates=job["candidates"],
            pins_sha256=builder.digest(root / "SOURCE_PINS.json"), group_binding=dict(binding, slot=slot),
            created_epoch=time.time(), expires_epoch=plan["expires_epoch"], precomputed_rom=True,
            precomputed_rom_keys=[job["candidates"][0]["key"]])
        builder.write_json(root / "PLAN.json", child_plan, immutable=True)
        for name in ("PLAN.json", "SOURCE_PINS.json"):
            builder.copy_checked(root / name, archive / name)
        results.append((child_plan, builder.digest(root / "PLAN.json")))
    return results


def copy_prepared(plan, record, directory, legacy):
    source = legacy / "cases" / record["key"]
    thermal.verify_inputs(record, source)
    contract, contract_sha = grid.read_record(source / "INPUT_CONTRACT.json")
    directory.mkdir(parents=True)
    hashes = dict(contract["input_sha256"])
    hashes["INPUT_CONTRACT.json"] = contract_sha
    for name in (record["dataset"], record["basis"], record["lifts"]):
        hashes[Path(name).name] = record["files_sha256"][name]
    for name, expected in hashes.items():
        builder.copy_checked(source / name, directory / name, expected)
    thermal.verify_inputs(record, directory)


def cpu_job(args):
    plan, work = checked_pool(args.work, args.plan_sha256)
    launch_record, _ = grid.read_record(work / "LAUNCH.json")
    parent = launch_record["worker_identity"]
    if os.getppid() != parent["pid"] or grid.process_identity(parent["pid"]) != parent:
        raise ValueError("CPU job is not a live supervisor child")
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError("cannot set CPU parent-death containment")
    if grid.process_identity(parent["pid"]) != parent or os.getppid() != parent["pid"]:
        raise ValueError("CPU supervisor exited during startup")
    signal.pthread_sigmask(signal.SIG_UNBLOCK, SPAWN_SIGNALS)
    job = plan["jobs"][args.slot]
    root = Path(job["work"])
    child_plan, child_sha = grid.read_record(root / "PLAN.json")
    builder.configure(root)
    grid.verify_pins(root / "SOURCE_PINS.json", child_plan["pins_sha256"])
    common = thermal.authenticated_common(child_plan)
    def deny_solver(**kwargs):
        raise RuntimeError("CPU-only phase cannot launch MAPDL")
    common.run_mapdl = deny_solver
    thermal.import_api(child_plan)
    for record in child_plan["candidates"]:
        directory = root / "cases" / record["key"]
        end = min(time.time() + thermal.CASE_SECONDS, plan["expires_epoch"])
        builder.write_json(root / "CPU_STATUS.json", dict(state="PREPARING", key=record["key"], pid=os.getpid()))
        copy_prepared(child_plan, record, directory, Path(plan["legacy_work"]))
        command = thermal.teacher_command(child_plan, record, directory, plan_only=True)
        sys.argv = command[2:]
        def render():
            try:
                runpy.run_path(command[2], run_name="__main__")
            except SystemExit as exc:
                if exc.code is not None and exc.code != 0:
                    raise
        grid.bounded_cpu(end, render)
        builder.write_json(directory / "PREFLIGHT_COMPLETE.json", dict(**thermal.scope(),
            plan_sha256=child_sha, input_contract_sha256=builder.digest(directory / "INPUT_CONTRACT.json"),
            teacher_manifest_sha256=builder.digest(directory / "preflight/M3_LAYER_INTERFACE_REPLAY_MANIFEST.json")), immutable=True)
    for record in child_plan["candidates"][:1]:
        directory = root / "cases" / record["key"]
        end = min(time.time() + thermal.STAGE_SECONDS, plan["expires_epoch"])
        builder.write_json(root / "CPU_STATUS.json", dict(state="PREDICT", key=record["key"], pid=os.getpid()))
        grid.bounded_cpu(end, lambda: thermal.predict(child_plan, record, directory))
        builder.write_json(directory / "ROM_SEAL.json", dict(**thermal.scope(), files_sha256={
            name: builder.digest(directory / name) for name in (
                "ROM_COMPLETE.json", "ROM_RESPONSE.npz", "ROM_CU_STATES.npz")}), immutable=True)
        builder.write_json(directory / "CPU_PRECOMPUTE_COMPLETE.json", dict(plan_sha256=child_sha,
            pins_sha256=child_plan["pins_sha256"],
            input_contract_sha256=builder.digest(directory / "INPUT_CONTRACT.json"),
            rom_seal_sha256=builder.digest(directory / "ROM_SEAL.json"), no_native_solver=True), immutable=True)
    receipt = dict(**thermal.scope(), plan_sha256=child_sha,
        keys=[r["key"] for r in child_plan["candidates"]], solver_launched=False)
    builder.write_json(root / "PREFLIGHT_COMPLETE.json", receipt, immutable=True)
    builder.copy_checked(root / "PREFLIGHT_COMPLETE.json", Path(child_plan["archive"]) / "PREFLIGHT_COMPLETE.json")
    builder.write_json(root / "CPU_STATUS.json", dict(state="ROM_PRECOMPUTE_COMPLETE", pid=os.getpid()))


def verify_cpu_completion(plan, slot, child_plan, child_sha, identity):
    root = Path(plan["jobs"][slot]["work"])
    grid.read_record(root / "PLAN.json", child_sha)
    grid.verify_pins(root / "SOURCE_PINS.json", child_plan["pins_sha256"])
    status, _ = grid.read_record(root / "CPU_STATUS.json")
    preflight, _ = grid.read_record(root / "PREFLIGHT_COMPLETE.json")
    if (status.get("state") != "ROM_PRECOMPUTE_COMPLETE" or status.get("pid") != identity["pid"]
            or preflight.get("plan_sha256") != child_sha or preflight.get("solver_launched") is not False
            or preflight.get("keys") != [r["key"] for r in child_plan["candidates"]]):
        raise ValueError("CPU phase lacks bound completion evidence")
    for record in child_plan["candidates"]:
        directory = root / "cases" / record["key"]
        thermal.verify_inputs(record, directory)
        receipt, _ = grid.read_record(directory / "PREFLIGHT_COMPLETE.json")
        if (receipt.get("plan_sha256") != child_sha
                or receipt.get("input_contract_sha256") != builder.digest(directory / "INPUT_CONTRACT.json")
                or receipt.get("teacher_manifest_sha256") != builder.digest(
                    directory / "preflight/M3_LAYER_INTERFACE_REPLAY_MANIFEST.json")):
            raise ValueError("CPU deck preflight evidence drift")
    directory = root / "cases" / child_plan["candidates"][0]["key"]
    cached, _ = grid.read_record(directory / "CPU_PRECOMPUTE_COMPLETE.json")
    if (cached.get("plan_sha256") != child_sha or cached.get("pins_sha256") != child_plan["pins_sha256"]
            or cached.get("input_contract_sha256") != builder.digest(directory / "INPUT_CONTRACT.json")
            or cached.get("no_native_solver") is not True):
        raise ValueError("CPU precomputed prediction authority drift")
    thermal.verify_rom_seal(directory, cached["rom_seal_sha256"])


def watch(children, observations, deadline, state, work, binding=None, heartbeat=None, on_complete=None):
    done = set()
    while len(done) < len(children):
        if time.time() >= deadline:
            raise TimeoutError("parallel phase deadline expired")
        if heartbeat is not None:
            heartbeat()
        for slot, (child, observed) in enumerate(zip(children, observations)):
            if slot in done:
                continue
            thermal.refresh_owned(child, observed)
            code = child.poll()
            if code is not None:
                if code != 0:
                    raise RuntimeError(f"parallel slot{slot} failed ({code}); preserve outputs, no retry")
                thermal.drain_success(child, observed, deadline, group_binding=binding)
                if on_complete is not None:
                    on_complete(slot)
                if time.time() >= deadline:
                    raise TimeoutError("parallel completion proof exceeded phase deadline")
                done.add(slot)
        state.update(completed_slots=sorted(done), heartbeat_epoch=time.time())
        builder.write_json(work / "STATUS.json", state)
        if len(done) < len(children):
            time.sleep(2.)


def supervisor(args):
    plan, work = checked_pool(args.work, args.plan_sha256)
    builder.configure(work)
    end = time.time() + 60.
    while not (work / "LAUNCH_PUBLISHED.json").exists():
        if time.time() >= end:
            raise TimeoutError("pool launch publication expired")
        time.sleep(.2)
    pub, _ = grid.read_record(work / "LAUNCH_PUBLISHED.json")
    launch_record, _ = grid.read_record(work / "LAUNCH.json", pub["launch_sha256"])
    if launch_record["worker_identity"] != grid.process_identity(os.getpid()):
        raise ValueError("pool launch identity drift")
    binding = dict(work=str(work), plan_sha256=args.plan_sha256,
                   launch_sha256=pub["launch_sha256"], slot=0)
    child_plans = grid.bounded_cpu(plan["created_epoch"] + plan["preparation_seconds"],
        lambda: job_plans(plan, work, binding))
    state = dict(state="PARALLEL_ROM_PRECOMPUTE", worker_identity=launch_record["worker_identity"], completed_slots=[])
    children, observed = [], []
    try:
        cpu_identities = []
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, SPAWN_SIGNALS)
        try:
            for slot, job in enumerate(plan["jobs"]):
                command = [sys.executable, "-B", str(Path(__file__).resolve()), "cpu-job", "--work", str(work),
                           "--plan-sha256", args.plan_sha256, "--slot", str(slot)]
                with (work / f"cpu_slot{slot}.log").open("x") as stream:
                    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, start_new_session=True, env=builder.configure(Path(job["work"])))
                inventory = thermal.ProcessInventory()
                children.append(child)
                observed.append(inventory)
                thermal.pin_spawned(child, inventory)
                identity = grid.wait_exec_identity(child, command, inventory)
                cpu_identities.append(identity)
                builder.write_json(work / f"CPU_CHILD{slot}.json", identity, immutable=True)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        builder.configure(work)
        state["cpu_children"] = [c.pid for c in children]
        builder.write_json(work / "STATUS.json", state)
        handoff = load_handoff(plan)
        released = [False]
        def poll_handoff():
            if not released[0]:
                released[0] = handoff.release_if_ready(plan["handoff_receipt"])
        precompute_end = min(time.time() + plan["precompute_seconds"],
            plan["created_epoch"] + plan["preparation_seconds"] + plan["precompute_seconds"])
        def completed_cpu(slot):
            child_plan, sha = child_plans[slot]
            grid.bounded_cpu(precompute_end,
                lambda: verify_cpu_completion(plan, slot, child_plan, sha, cpu_identities[slot]))
        watch(children, observed, precompute_end, state, work,
              heartbeat=poll_handoff, on_complete=completed_cpu)
        children, observed = [], []
        state.update(state="WAITING_NATIVE_HANDOFF", completed_slots=[])
        builder.write_json(work / "STATUS.json", state)
        wait_end = min(time.time() + plan["handoff_wait_seconds"], plan["expires_epoch"]
            - plan["native_seconds"] - plan["exit_grace_seconds"])
        while not released[0]:
            if time.time() >= wait_end:
                raise TimeoutError("serial handoff deadline expired")
            poll_handoff()
            time.sleep(2.)
        resources(work, len(plan["jobs"]), sum(len(j["candidates"]) for j in plan["jobs"]))
        if time.time() + plan["native_seconds"] + plan["exit_grace_seconds"] > plan["expires_epoch"]:
            raise TimeoutError("pool lacks the reserved native queue budget")
        with grid.native_host_lock(work) as held:
            try:
                if (held["device"], held["inode"]) != tuple(plan["handoff_receipt"]["host_lock_identity"]):
                    raise ValueError("canonical lock inode changed across handoff")
                builder.write_json(work / "HOST_LOCK.json", dict(**held, worker_identity=launch_record["worker_identity"],
                    plan_sha256=args.plan_sha256), immutable=True)
                grid.authenticate_group(binding)
                previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, SPAWN_SIGNALS)
                try:
                    for slot, (job, (_, sha)) in enumerate(zip(plan["jobs"], child_plans)):
                        child = thermal.launch(SimpleNamespace(work=Path(job["work"]), plan_sha256=sha))
                        inventory = thermal.ProcessInventory()
                        children.append(child)
                        observed.append(inventory)
                        thermal.pin_spawned(child, inventory)
                        registered, _ = thermal.register_group_child(work, args.plan_sha256, slot, job, child)
                        inventory.scope_work = Path(job["work"])
                        inventory.scope_identity = registered["identity"]
                        inventory.scope_launch_sha = registered["launch_sha256"]
                    thermal.publish_group_ready(work, args.plan_sha256, slots=grid.group_slots(plan))
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                state.update(state="PARALLEL_MAPDL_VALIDATION", native_children=[c.pid for c in children], completed_slots=[])
                builder.write_json(work / "STATUS.json", state)
                watch(children, observed, plan["expires_epoch"], state, work, binding=binding)
            except BaseException:
                thermal.contain_group_failure(children, observed, state, work, Path(plan["archive"]))
                raise
            state.update(state="OPERATOR_VALIDATION_BATCH_COMPLETE")
            builder.write_json(work / "STATUS.json", state)
            builder.write_json(Path(plan["archive"]) / "COMPLETE.json", state, immutable=True)
    except BaseException as exc:
        if children:
            thermal.cleanup_group_children(children, observed)
        state.update(state="STOPPED_REQUIRES_REVIEW", error=str(exc))
        builder.write_json(work / "STATUS.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("launch")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--legacy-work", type=Path, required=True)
    p.add_argument("--handoff-module", type=Path, required=True)
    p.add_argument("--slots", type=int, required=True)
    for mode in ("supervisor", "cpu-job"):
        p = sub.add_parser(mode)
        p.add_argument("--work", type=Path, required=True)
        p.add_argument("--plan-sha256", required=True)
        if mode == "cpu-job":
            p.add_argument("--slot", type=int, required=True)
    args = parser.parse_args()
    return {"launch": launch, "supervisor": supervisor, "cpu-job": cpu_job}[args.mode](args)


if __name__ == "__main__":
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt(signum)))
    main()
