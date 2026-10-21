"""Spawn isolated assembly-order shards; only the parent publishes run results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
from uuid import uuid4


PROJECT = Path("/app/nanocad/projects/personal/zchen/YAP_STACK")
REPO = Path(__file__).resolve().parents[1]
PROTOCOL = "yap.assembly_order_worker.v1"
POLL_SECONDS = 0.5
# A stuck loader/solver cannot leave its shard running indefinitely.
MAX_WORKER_SECONDS = 24 * 60 * 60


def _canonical(value):
    path = Path(value).expanduser().absolute()
    resolved = path.resolve()
    if any({"#snapshot", ".snapshot"}.intersection(p.parts) for p in (path, resolved)):
        raise ValueError("worker storage must not enter snapshots")
    if not resolved.is_relative_to(PROJECT) or resolved == PROJECT:
        raise ValueError("worker storage must be inside the canonical project")
    return resolved


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _atomic(path, value):
    path = _canonical(path)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(path)
    os.replace(temporary, path)


def _read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _indices(indices, count):
    if (not isinstance(indices, list) or not indices
            or any(type(i) is not int or not 0 <= i < count for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("route_indices must be distinct non-Boolean in-range integers")
    return indices


def _memory_gate(minimum):
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(fields["MemAvailable"].split()[0]) / 1024**2
    if available < minimum:
        raise RuntimeError(f"available RAM {available:.1f} GiB below order-worker gate {minimum:g} GiB")
    return available


def _sources():
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(PROJECT):
                paths.add(path)
    return {str(path): _hash(path) for path in sorted(paths)}


def _check_sources(sources):
    if not isinstance(sources, dict) or not sources:
        raise ValueError("missing worker source pins")
    for filename, digest in sources.items():
        path = _canonical(filename)
        if path.suffix != ".py" or _hash(path) != digest:
            raise ValueError(f"source changed during assembly execution: {filename}")


def _environment(directory):
    env = dict(os.environ)
    # Override inherited homes and caches, not just Python's tempfile directory.
    locations = {
        "HOME": "home", "USERPROFILE": "home", "XDG_CACHE_HOME": "cache",
        "XDG_CONFIG_HOME": "config", "XDG_DATA_HOME": "data", "XDG_STATE_HOME": "state",
        "XDG_RUNTIME_DIR": "runtime", "APPDATA": "config", "LOCALAPPDATA": "cache",
        "YAP_WORK_DIR": "work", "TMPDIR": "temp", "TMP": "temp", "TEMP": "temp",
        "TEMPDIR": "temp", "MPLCONFIGDIR": "matplotlib", "NUMBA_CACHE_DIR": "numba",
        "PYTHONPYCACHEPREFIX": "pycache", "JOBLIB_TEMP_FOLDER": "temp",
        "PYTHONUSERBASE": "python-user", "IPYTHONDIR": "ipython",
        "JUPYTER_CONFIG_DIR": "jupyter-config", "JUPYTER_DATA_DIR": "jupyter-data",
        "JUPYTER_RUNTIME_DIR": "jupyter-runtime", "PIP_CACHE_DIR": "pip",
        "CUDA_CACHE_PATH": "cuda", "CUPY_CACHE_DIR": "cupy", "TORCH_HOME": "torch",
        "HF_HOME": "huggingface", "HF_HUB_CACHE": "huggingface-hub",
        "HUGGINGFACE_HUB_CACHE": "huggingface-hub", "TRANSFORMERS_CACHE": "transformers",
        "TRITON_CACHE_DIR": "triton",
    }
    for key, leaf in locations.items():
        path = _canonical(directory / leaf)
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    for key in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
                "NUMEXPR_MAX_THREADS", "NUMBA_NUM_THREADS"):
        env[key] = "1"
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO),
               NUMBA_THREADING_LAYER="omp", OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")
    # Small immutable Si-cap matrices may be shared; Cu states never are.
    cap_cache = _canonical(env.get("YAP_W2W_SI_CAP_CACHE_DIR", directory.parent / "silicon_cap_cache"))
    cap_cache.mkdir(parents=True, exist_ok=True)
    env["YAP_W2W_SI_CAP_CACHE_DIR"] = str(cap_cache)
    return env


def _static_identity(header):
    config = header["native_configuration"]
    operator = config["operator_provenance"]
    keys = ("schema", "component_id", "package_path", "package_sha256", "input_sha256", "inputs_sha256",
            "operator_inputs", "material_card", "geometry", "unit_geometry", "port_basis_sha256",
            "port_rank_per_face", "source_operator_keys", "frozen_preview", "bonding_pitch_um",
            "pad_diameter_um", "operator_build_receipt", "operator_build_state",
            "physical_registration", "component_validation")
    binding = operator.get("corrected_material", {})
    def scientific(value):
        if isinstance(value, dict):
            return {key: scientific(item) for key, item in value.items()
                    if key not in {"warmup_seconds", "configured_cache_directory", "cache_directory"}}
        if isinstance(value, list):
            return [scientific(item) for item in value]
        return value
    material = scientific(binding)
    provenance = {k: v for k, v in header["provenance"].items() if k != "operator_set_sha256"}
    # Acceleration counters and compiled-material warmup timings are worker-local.
    configuration = {k: v for k, v in config.items()
                     if k not in ("operator_provenance", "runtime_acceleration")}
    acceleration = config.get("runtime_acceleration") or {}
    configuration["acceleration_pins"] = {k: acceleration[k] for k in (
        "exact_constitutive_kernel", "material_equations_unchanged", "implementation_sha256")
        if k in acceleration}
    caps = operator.get("w2w_si_caps", {})
    return dict(provenance=provenance, configuration=configuration,
                operator={k: scientific(operator[k]) for k in keys if k in operator}, material=material,
                w2w_si_caps={key: caps[key] for key in
                    ("trace_sha256", "material", "operator_sha256", "construction_sha256") if key in caps})


def _envelope(manifest, kind, **payload):
    return dict(schema=PROTOCOL, kind=kind, run_id=manifest["run_id"],
                worker_id=manifest["worker_id"], context_id=manifest["plan"]["context_id"],
                manifest_sha256=_digest(manifest), **payload)


def _validate(envelope, manifest, kind):
    for key, value in _envelope(manifest, kind).items():
        if envelope.get(key) != value:
            raise ValueError(f"worker {kind} envelope mismatch: {key}")


def _exit_code(process):
    if process.returncode is not None:
        return process.returncode
    # Keep exited leaders unreaped, reserving their PID/session for safe group cleanup.
    result = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if result is None:
        return None
    return result.si_status if result.si_code == os.CLD_EXITED else -result.si_status


def _signal_group(process, sig):
    if process.returncode is not None:
        return
    try:
        if os.getsid(process.pid) != process.pid:
            raise RuntimeError("worker lost its private process session")
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _stop(processes):
    # WNOWAIT keeps even an exited leader's session identity pinned until wait().
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for process in processes:
            _signal_group(process, sig)
        if sig == signal.SIGTERM:
            deadline = time.monotonic() + 3
            while any(_exit_code(p) is None for p in processes) and time.monotonic() < deadline:
                time.sleep(0.05)
    for process in processes:
        process.wait()


def run_parallel(plan, settings, *, work_dir, minimum_memory_gib=40., progress=None,
                 route_complete=None):
    """Evaluate static shards in fresh interpreters and return an ordered archive.

    ``work_dir`` must be new and canonical. Completed callbacks run in plan order
    on the caller's thread; a callback failure aborts and reaps every owned child.
    """
    from .assembly_order import _check_plan, archive_template

    _check_plan(plan)
    requested = settings.get("order_workers", 1)
    if type(requested) is not int or requested < 1:
        raise ValueError("order_workers must be a positive integer")
    count = len(plan["routes"])
    if not count:
        raise ValueError("parallel assembly requires at least one route")
    effective = min(requested, count)
    layers = settings.get("layer_workers", 1)
    if type(layers) is not int or layers < 1 or (effective > 1 and layers > 1):
        raise ValueError("parallel orders require layer_workers=1")
    if (isinstance(minimum_memory_gib, bool) or not isinstance(minimum_memory_gib, (int, float))
            or not math.isfinite(minimum_memory_gib) or minimum_memory_gib <= 0):
        raise ValueError("minimum_memory_gib must be finite and positive")
    cpus = len(os.sched_getaffinity(0))
    if effective > cpus:
        raise RuntimeError(f"{effective} order workers exceed CPU affinity budget {cpus}")
    _memory_gate(minimum_memory_gib * effective)
    directory = _canonical(work_dir)
    directory.mkdir(parents=True, exist_ok=False)
    run_id, pins = uuid4().hex, _sources()
    template = archive_template(plan)
    manifests, processes, logs, started = [], [], [], []
    routes, route_headers, headers, sources, finished = {}, {}, {}, dict(pins), set()
    consumed_hashes = {}
    baseline, archive, next_callback = None, None, 0
    previous_termination = None
    if threading.current_thread() is threading.main_thread():
        def terminate(signum, _frame):
            raise SystemExit(128 + signum)
        previous_termination = signal.signal(signal.SIGTERM, terminate)
    try:
        for worker in range(effective):
            shard = directory / f"worker_{worker:04d}"
            shard.mkdir()
            indices = _indices(list(range(worker, count, effective)), count)
            manifest = dict(run_id=run_id, worker_id=worker, plan=plan, settings=settings,
                            route_indices=indices, source_pins=pins)
            manifests.append(manifest)
            _atomic(shard / "manifest.json", manifest)
            env = _environment(shard)
            log = (shard / "worker.log").open("xb")
            logs.append(log)
            process = subprocess.Popen(
                [sys.executable, "-B", "-m", "W2W.assembly_parallel", "--worker",
                 str(shard / "manifest.json")], cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            started.append(time.monotonic())
            if progress:
                progress(f"Started order worker {worker + 1}/{effective}: {len(indices)} routes")
        while len(finished) != effective:
            _memory_gate(minimum_memory_gib)
            for worker, (process, manifest) in enumerate(zip(processes, manifests)):
                shard = directory / f"worker_{worker:04d}"
                code = _exit_code(process)
                if code is None and time.monotonic() - started[worker] > MAX_WORKER_SECONDS:
                    raise TimeoutError(f"order worker {worker} exceeded {MAX_WORKER_SECONDS}s; see {shard / 'worker.log'}")
                for index in manifest["route_indices"]:
                    path = shard / f"route_{index:06d}.json"
                    if index in routes or not path.exists():
                        continue
                    encoded = path.read_bytes()
                    payload = json.loads(encoded)
                    consumed_hashes[index] = hashlib.sha256(encoded).hexdigest()
                    _validate(payload, manifest, "route")
                    route = payload["route"]
                    expected = plan["routes"][index]
                    if (type(payload.get("route_index")) is not int or payload["route_index"] != index
                            or payload.get("request_id") != expected["request_id"]
                            or route.get("request_id") != expected["request_id"]
                            or route.get("order") != expected["order"]):
                        raise ValueError("worker route identity mismatch")
                    header = payload["header"]
                    if (header.get("context_id") != plan["context_id"]
                            or header.get("schema") != template["schema"]):
                        raise ValueError("worker archive context mismatch")
                    identity = _static_identity(header)
                    if baseline is not None and identity != baseline:
                        raise ValueError("worker static operator/material/source pins differ")
                    baseline = identity
                    headers[worker] = header
                    route_headers[index] = header
                    routes[index] = route
                if code is not None and worker not in finished:
                    if code != 0:
                        raise RuntimeError(f"order worker {worker} exited {code}; see {shard / 'worker.log'}")
                    terminal = _read(shard / "terminal.json")
                    _validate(terminal, manifest, "terminal")
                    if terminal.get("status") != "SUCCESS" or terminal.get("route_indices") != manifest["route_indices"]:
                        raise ValueError("worker terminal success/coverage missing")
                    for index in manifest["route_indices"]:
                        digest = terminal["route_sha256"].get(str(index))
                        if (index not in routes or digest != consumed_hashes.get(index)
                                or digest != _hash(shard / f"route_{index:06d}.json")):
                            raise ValueError("worker terminal route proof mismatch")
                    _check_sources(terminal["runtime_sources"])
                    for path, digest in terminal["runtime_sources"].items():
                        if path in sources and sources[path] != digest:
                            raise ValueError(f"worker source disagreement: {path}")
                        sources[path] = digest
                    _signal_group(process, signal.SIGKILL)
                    if process.wait() != 0:
                        raise RuntimeError(f"order worker {worker} failed during retirement")
                    finished.add(worker)
            while next_callback in routes:
                if archive is None:
                    archive = dict(headers[0], routes=template["routes"])
                archive["routes"][next_callback] = routes[next_callback]
                callback_archive = dict(route_headers.pop(next_callback), routes=archive["routes"])
                if route_complete:
                    route_complete(callback_archive, routes[next_callback])
                next_callback += 1
            if len(finished) != effective:
                time.sleep(POLL_SECONDS)
        _check_sources(sources)
        configurations = [headers[i]["native_configuration"] for i in range(effective)]
        archive["provenance"] = dict(archive["provenance"])
        provenances = [config["operator_provenance"] for config in configurations]
        if any(proof != provenances[0] for proof in provenances[1:]):
            from .assembly_order import content_id
            archive["provenance"]["operator_set_sha256"] = content_id(provenances)
        archive["native_configuration"] = dict(configurations[0],
            order_execution=dict(requested_workers=requested, effective_workers=effective,
                strategy="subprocess_static_round_robin", run_id=run_id,
                worker_timeout_seconds=MAX_WORKER_SECONDS,
                per_worker_configuration=configurations), worker_runtime_sources=sources)
        archive["status"] = "COMPLETE" if settings.get("metric") is not None else "MECHANICS_COMPLETE_OVERLAY_PENDING"
        archive["physical_readiness"] = False
        return archive
    except BaseException:
        _stop(processes)
        raise
    finally:
        if previous_termination is not None:
            signal.signal(signal.SIGTERM, previous_termination)
        for log in logs:
            log.close()


def _worker(manifest_path):
    manifest_path = _canonical(manifest_path)
    manifest, shard = _read(manifest_path), manifest_path.parent
    # Check storage before importing any native or numerical modules.
    for key in ("HOME", "YAP_WORK_DIR", "TMPDIR", "TMP", "TEMP", "NUMBA_CACHE_DIR", "MPLCONFIGDIR"):
        if not _canonical(os.environ[key]).is_relative_to(shard):
            raise ValueError(f"worker environment escaped its directory: {key}")
    from mechanics_runtime import configure_runtime_storage
    configure_runtime_storage()
    from .assembly_order import _check_plan, is_die_plan, is_f2b_plan, is_prepared_plan
    from .assembly_mechanics import run_native, _recipe, _check_metric_policy
    from .assembly_operator_library import (load_assembly_library, load_f2b_assembly_library,
                                           load_prepared_die_library, require_prepared_operator)

    plan, settings = manifest["plan"], manifest["settings"]
    _check_plan(plan)
    indices = _indices(manifest["route_indices"], len(plan["routes"]))
    _check_sources(manifest["source_pins"])
    pins = _sources()
    _recipe(plan, settings["temporal_profile"])
    _check_metric_policy(plan, settings.get("metric"))
    preview = settings.get("frozen_preview", False)
    if type(preview) is not bool:
        raise ValueError("frozen_preview must be Boolean")
    if is_prepared_plan(plan):
        require_prepared_operator(settings["operator_package"], plan["context"]["geometry"], frozen_preview=preview)
    loader = (load_prepared_die_library if is_die_plan(plan) else
              load_f2b_assembly_library if is_f2b_plan(plan) else load_assembly_library)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        library, provenance = loader(settings["operator_package"], plan["context"]["geometry"],
                                     **({"frozen_preview": True} if preview else {}))
        _check_sources(pins)
        pins.update(_sources())
        by_request = {plan["routes"][i]["request_id"]: i for i in indices}
        completed = []

        def save(archive, route):
            _check_sources(pins)
            pins.update(_sources())
            index = by_request[route["request_id"]]
            header = {k: v for k, v in archive.items() if k != "routes"}
            _atomic(shard / f"route_{index:06d}.json", _envelope(manifest, "route",
                route_index=index, request_id=route["request_id"], header=header, route=route))
            completed.append(index)

        run_native(plan, library, provenance, temporal_profile=settings["temporal_profile"],
                   metric=settings.get("metric"), route_indices=indices,
                   progress=lambda message: print(message, flush=True), route_complete=save)
    if completed != indices:
        raise ValueError("worker did not complete its assigned routes in order")
    _check_sources(manifest["source_pins"])
    _check_sources(pins)
    _atomic(shard / "terminal.json", _envelope(manifest, "terminal", status="SUCCESS",
        route_indices=completed, runtime_sources=_sources(),
        route_sha256={str(i): _hash(shard / f"route_{i:06d}.json") for i in indices}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        _worker(args.worker)
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
