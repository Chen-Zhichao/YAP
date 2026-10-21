"""Build p2 prepared TOTAL30/80 candidates using immutable native producers.

Run each thickness once on a physical mfd or sysml host. Solver scratch remains local;
publication is a separate, bounded copy of explicit checked artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time


STAGES = ("tangent", "q6", "extract_tangent", "extract_q6", "static_lifts", "q6_operator")
TOTALS = {30: 17., 80: 67.}
BASIS_SHA = "8c2941e8358dba9626177cfd54c2187d4d0878d4ca1108d2652261efd318cb06"
OWNED_WORK_ROOT = Path("/data/puneet/zhichao")
MFD_ALTERNATE_WORK_ROOT = Path("/home/scratch/zhichao")
SYSML_WORK_ROOT = Path("/home/puneet/data/zhichao")
PRODUCER_REVISIONS = {
    "run_exp12_m4_common_port_stiffness.py": "5bf66826de9fd962c637a88ea140b713c473aa4a7d381d2854185cc240fa119b",
    "build_exp12_m4_static_lift_fields.py": "2843d00afd06679e8ac0d2bbee70402dbe7411a32ccb1e8a8667be381905c17d",
}
LEGACY_STAGE_PINS = {
    "bundle": "1259440b79ea3e3417fc230af7be7960f6ffe2d3e824a32edb6c40f07ba7dcdd",
    "resources": "401423f317ea00b6cc661fe8eea38b822d3a20f3cbd3b7dce2b5f1fd39f1441e",
    "archive": "87618637f6459b2b0aa915344f1303db04298c003503143018495dd644e0d4b5",
    "compact": "cba9f64bdf41b610ec9b538858f07c119f7febd89515f84d8ac6df2e74bd8612",
    "parent": "ad734f2a4a09e704a9c229aff75398bf5817eeb9352e19201c7f63a0846a2791",
}


def safe_path(value):
    path = Path(value).expanduser().absolute()
    for candidate in (path, path.resolve()):
        if {"#snapshot", ".snapshot"}.intersection(candidate.parts) or candidate.is_relative_to("/tmp"):
            raise ValueError(f"forbidden storage: {candidate}")
    return path.resolve()


def digest(value):
    path = safe_path(value)
    rate = 32 * 1024**2 if shared_storage(path) else None
    started, count, result = time.monotonic(), 0, hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(block)
            count += len(block)
            pace_io(rate, count, started)
    return result.hexdigest()


def shared_storage(path):
    local_filesystems = {"ext2", "ext3", "ext4", "xfs", "btrfs", "zfs", "tmpfs", "ramfs"}
    try:
        with Path("/proc/self/mountinfo").open() as stream:
            raw = stream.read(256 * 1024 + 1)
        rows = raw.splitlines()
        if len(raw) > 256 * 1024 or not 1 <= len(rows) <= 4096:
            return True
        mounts = []
        for row in rows:
            before, separator, after = row.partition(" - ")
            fields, details = before.split(), after.split()
            if not separator or len(fields) < 6 or len(details) < 3:
                return True
            name = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4])
            mount = Path(name)
            if not mount.is_absolute():
                return True
            if path.is_relative_to(mount):
                mounts.append((len(mount.parts), details[0]))
        if not mounts:
            return True
        longest = max(length for length, _ in mounts)
        return any(kind not in local_filesystems for length, kind in mounts if length == longest)
    except (OSError, ValueError):
        return True


def pace_io(rate, count, started):
    if rate is not None:
        delay = count / rate - (time.monotonic() - started)
        if delay > 0.:
            time.sleep(delay)


def read_json(path):
    return json.loads(safe_path(path).read_text())


def write_json(path, data, *, immutable=False):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if immutable:
        with path.open("x") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = path.with_name(path.name + f".{os.getpid()}.new")
        with temporary.open("x") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)


def physical_host():
    host = socket.gethostname().split(".")[0]
    if host not in {"mfd", "sysml"} or Path("/proc/1/comm").read_text().strip() == "codex":
        raise RuntimeError("launch requires a physical mfd or sysml host")
    return host


def scratch_root(host):
    if host == "mfd":
        return OWNED_WORK_ROOT
    if host == "sysml":
        return SYSML_WORK_ROOT
    raise ValueError("unknown physical execution host")


def local_scratch(value, *, host=None, allow_root=False):
    original = Path(value).expanduser().absolute()
    path = safe_path(original)
    roots = (OWNED_WORK_ROOT, MFD_ALTERNATE_WORK_ROOT, SYSML_WORK_ROOT)
    if not any(path.is_relative_to(root) and (allow_root or path != root) for root in roots):
        raise ValueError("builds require an owned local scratch child directory")
    for parent in (original, *original.parents):
        if parent.is_symlink():
            raise ValueError("local scratch cannot use symlinks or aliases")
    host = physical_host() if host is None else host
    allowed = (OWNED_WORK_ROOT, MFD_ALTERNATE_WORK_ROOT) if host == "mfd" else (scratch_root(host),)
    if not any(path.is_relative_to(root) and (allow_root or path != root) for root in allowed):
        raise ValueError("local scratch does not match the physical execution host")
    return path


def configure(work):
    work = local_scratch(work)
    runtime = work / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    values = dict(TMPDIR=str(runtime), TMP=str(runtime), TEMP=str(runtime),
                  MPLCONFIGDIR=str(runtime / "matplotlib"), NUMBA_CACHE_DIR=str(runtime / "numba"),
                  PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1", MAPDL_HEARTBEAT_SEC="60",
                  YAP_LIFECYCLE_WORK=str(work))
    os.environ.pop("PYTHONPYCACHEPREFIX", None)
    os.environ.update(values)
    sys.dont_write_bytecode = True
    import tempfile
    tempfile.tempdir = str(runtime)
    return dict(os.environ)


def load_api(source, work):
    source = safe_path(source)
    host = physical_host()
    work = local_scratch(work, host=host)
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("prepared_candidate_producer", source / "exp12_ofat_geometry_20260913.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.BASE = source
    module.ROOT = work / "results"
    module.WORK_ROOTS = {host: work / "solver"}
    module.CASES = {
        f"bi2_tsv_h{int(si)}": module.GeometryCase(f"bi2_tsv_h{int(si)}", "bi2", si, host, nx=45, ny=20, rank=64, cores=8)
        for si in TOTALS.values()
    }
    module.PACKAGE_CASES = {}
    module.publish_readiness = lambda: None
    module.sha256 = digest
    return module


def resource_check(work):
    host = physical_host()
    mount = subprocess.run(["findmnt", "-n", "-T", str(work), "-o", "TARGET,SOURCE,FSTYPE"],
                           check=True, capture_output=True, text=True).stdout.strip()
    if "nfs" in mount.lower():
        raise RuntimeError("solver scratch must not be NFS")
    work = local_scratch(work, host=host, allow_root=True)
    before = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
    time.sleep(1.)
    after = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
    delta = [b - a for a, b in zip(before, after)]
    busy = (sum(delta) - delta[3] - delta[4]) / sum(delta)
    wait = delta[4] / sum(delta)
    memory = {line.split(":", 1)[0]: int(line.split(":", 1)[1].split()[0])
              for line in Path("/proc/meminfo").read_text().splitlines()}
    ps = subprocess.run(["ps", "-U", str(os.getuid()), "-o", "pid=,ppid=,pcpu=,rss=,etime=,stat=,comm="],
                        check=True, capture_output=True, text=True, timeout=10).stdout
    rows = ps.splitlines()
    if len(ps) > 262144 or len(rows) > 4096:
        raise RuntimeError("own-UID resource census exceeds its bound")
    active_ansys = []
    for row in rows:
        fields = row.split(maxsplit=6)
        if len(fields) != 7:
            raise RuntimeError("invalid resource process census")
        if "ansys" in fields[6].lower() and not fields[5].startswith("Z"):
            active_ansys.append(row)
    result = dict(host=host, mount=mount, logical_cpus=os.cpu_count(), load_average=os.getloadavg(),
                  available_memory_gib=memory["MemAvailable"] / 1024**2,
                  local_free_gib=shutil.disk_usage(work).free / 1024**3,
                  cpu_busy_fraction=busy, cpu_iowait_fraction=wait,
                  own_ansys_processes=active_ansys,
                  requested_cores=8)
    if result["available_memory_gib"] < 64 or result["local_free_gib"] < 200 or busy + 8 / os.cpu_count() > .9 or wait > .25:
        raise RuntimeError(f"insufficient physical resources: {result}")
    return result


def run_stage(command, log, name, *, environment=None):
    print(f"START {name}: {log}", flush=True)
    with log.open("x") as stream:
        child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        while True:
            try:
                status = child.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                print(f"RUNNING {name}, producer pid={child.pid}, log_bytes={log.stat().st_size}", flush=True)
    if status:
        raise subprocess.CalledProcessError(status, command)
    print(f"COMPLETE {name}", flush=True)


def verify_template(api, case):
    revisions = {}
    for value, original in read_json(case.template)["inputs"].items():
        observed = digest(value)
        accepted = PRODUCER_REVISIONS.get(Path(value).name, original)
        if observed != accepted:
            raise ValueError(f"unreviewed template producer/input: {value}")
        revisions[value] = dict(template_sha256=original, build_sha256=observed)
    for step in api.static_steps(case):
        command = step["command"]
        if "--pure-si-substrate" in command or "--macro-reference-y-offset-mm" in command:
            raise ValueError("new optional pure-Si/shifted-reference producer branches are not admitted")
    return revisions


def build(args):
    work = safe_path(args.work)
    environment = configure(work)
    api = load_api(args.native_source, work)
    case = api.CASES[f"bi2_tsv_h{int(TOTALS[args.total])}"]
    return build_case(api, case, work, environment)


def build_case(api, case, work, environment, *, adopt_stage=None):
    """Build one explicit geometry without changing the legacy package catalog."""
    output = api.ROOT / case.key
    output.mkdir(parents=True, exist_ok=True)
    with (output / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "RUN_STATUS.json").exists():
            raise RuntimeError("existing attempt requires manual review; no automatic retry")
        resource = resource_check(work)
        revisions = verify_template(api, case)
        plan = api.prepare(case)
        write_json(output / "TEMPLATE_PRODUCER_VERSIONS.json", revisions, immutable=True)
        controller = dict(wrapper_sha256=digest(__file__), helper_sha256=digest(api.__file__),
                          resources=resource, case=asdict(case), worker_pid=os.getpid(),
                          temporary_directory=environment["TMPDIR"], solver_timeout_seconds=0)
        write_json(output / "LAUNCH.json", controller, immutable=True)
        status = dict(state="RUNNING", case=case.key, started_at=api.now(), completed=[])
        write_json(output / "RUN_STATUS.json", status)
        try:
            for step in plan["steps"]:
                api.verify_inputs(plan)
                resource_check(work)
                start = time.monotonic()
                status.update(stage=step["name"])
                write_json(output / "RUN_STATUS.json", status)
                adopted = adopt_stage is not None and step["name"] in STAGES[:2]
                if adopted:
                    adopt_stage(step)
                else:
                    run_stage(step["command"], output / f'{step["name"]}.log', step["name"])
                api.verify_static_stage(case, step, seal=True)
                status["completed"].append(dict(stage=step["name"], elapsed_s=time.monotonic() - start,
                                               execution="ADOPTED_SEALED_STAGE" if adopted else "EXECUTED"))
                write_json(output / "RUN_STATUS.json", status)
            api.verify_inputs(plan)
            resource_check(work)
            print("BUILD target all-Gauss strain maps and material integrals", flush=True)
            api.build_dataset(case, environment=environment)
            status.update(state="COMPLETE_ENGINEERING_CANDIDATE", completed_at=api.now(), thermal_geometry_validated=False)
            write_json(output / "RUN_STATUS.json", status)
            print(json.dumps(status, sort_keys=True), flush=True)
        except BaseException as error:
            status.update(state="STOP_REQUIRES_REVIEW", error=f"{type(error).__name__}: {error}")
            write_json(output / "RUN_STATUS.json", status)
            raise


def transfer_audit(api, case, *, audit_output=None):
    import numpy as np
    from build_exp12_m4_direct_bubble_basis import maximum_periodic_jump, orthonormalize_homogeneous_periodic
    from build_exp12_m4_all_gauss_layer_dataset import close_bubble_basis_under_inplane_swap
    output = api.ROOT / case.key
    report = read_json(output / "bubble/TRANSFER_MANIFEST.json")
    path = output / "bubble/M4_DIRECT_ZERO_TRACE_BUBBLE_BASIS.npz"
    source_path, source_dataset = api.donor_files(case)
    with np.load(source_path, allow_pickle=False) as source, np.load(path, allow_pickle=False) as target:
        source_height = float(np.ptp(source["coordinates_mm"][:, 1])) * 1000.
        coordinates, modes = target["coordinates_mm"], target["bubble_modes"]
        node_ids = target["node_ids"].copy()
        face_indices = (target["bottom_node_indices"], target["top_node_indices"])
        scales = target["bubble_coordinate_scale"]
        api.verify_volume(case, coordinates, target["node_ids"])
        faces = np.r_[target["bottom_node_indices"], target["top_node_indices"]]
        face = float(np.max(np.abs(modes[faces]), initial=0.))
        jump = float(maximum_periodic_jump(modes, coordinates)) / max(float(np.max(np.abs(modes))), np.finfo(float).tiny)
        flat = modes.reshape(-1, modes.shape[2])
        singular = np.linalg.svd(flat, compute_uv=False)
        gram = float(np.max(np.abs(flat.T @ flat - np.eye(modes.shape[2]))))
        rank = int(np.count_nonzero(singular > singular[0] * 1e-10))
        if (abs(source_height - 38.5) > 1e-9 or face > 1e-9 or jump > 1e-4 or gram > 1e-9
                or rank != modes.shape[2] or report["rank"] != rank
                or source["bubble_modes"].shape[2] != rank or not np.isfinite(modes).all()):
            raise ValueError("independent transferred-bubble audit failed")
        raw = dict(rank=rank, observed_rank=rank, gram_defect=gram, face_trace=face,
                   relative_periodic_jump=jump, singular_value_min=float(singular[-1]),
                   singular_value_max=float(singular[0]))
        modes = modes.copy()
        modes[faces] = 0.
        modes, scales, added, _, _ = close_bubble_basis_under_inplane_swap(modes, coordinates, scales, None)
        modes[faces] = 0.
        modes, _ = orthonormalize_homogeneous_periodic(modes, coordinates, face_indices)
        flat = modes.reshape(-1, modes.shape[2])
        singular = np.linalg.svd(flat, compute_uv=False)
        final_rank = int(np.count_nonzero(singular > singular[0] * 1e-10))
        face = float(np.max(np.abs(modes[faces]), initial=0.))
        jump = float(maximum_periodic_jump(modes, coordinates)) / max(float(np.max(np.abs(modes))), np.finfo(float).tiny)
        gram = float(np.max(np.abs(flat.T @ flat - np.eye(modes.shape[2]))))
        if final_rank != rank + added or face > 1e-9 or jump > 1e-4 or gram > 1e-9:
            raise ValueError("independent final in-plane-closed bubble audit failed")
        final_path = output / "bubble/AUDITED_FINAL_BUBBLE_SPACE.npz" if audit_output is None else safe_path(audit_output)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            raise FileExistsError(final_path)
        np.savez_compressed(final_path, coordinates_mm=coordinates, node_ids=node_ids,
                            bubble_modes=modes, bubble_coordinate_scale=scales,
                            bottom_node_indices=face_indices[0], top_node_indices=face_indices[1])
    return dict(source_thickness_um=source_height, target_thickness_um=case.substrate_um,
                source_pitch_um=6., source_rank=rank, raw_transferred_rank=rank,
                closure_added_rank=added, raw_transfer_audit=raw, rank=final_rank, observed_rank=final_rank,
                singular_value_min=float(singular[-1]), singular_value_max=float(singular[0]),
                rank_relative_tolerance=1e-10, gram_defect=gram, face_trace=face, relative_periodic_jump=jump,
                source_bubble_sha256=digest(source_path), source_dataset_sha256=digest(source_dataset),
                raw_transferred_bubble_sha256=digest(path), transferred_bubble_sha256=digest(final_path),
                raw_transferred_bubble=dict(path=f"../build_receipts/{case.key}/bubble/M4_DIRECT_ZERO_TRACE_BUBBLE_BASIS.npz",
                                            sha256=digest(path)),
                transferred_bubble=dict(path=f"../build_receipts/{case.key}/bubble/AUDITED_FINAL_BUBBLE_SPACE.npz",
                                        sha256=digest(final_path)),
                method="normalized_thickness_displacement_transfer_plus_inplane_swap_closure",
                producer_transfer_method=report["method"],
                transfer_implementation_sha256=digest(api.__file__), thermal_geometry_validated=False)


def copy_checked(source, target, expected=None):
    source, target = safe_path(source), safe_path(target)
    if target.exists():
        raise FileExistsError(target)
    reference = digest(source) if expected is None else expected
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".{os.getpid()}.copying")
    rate = 24 * 1024**2 if shared_storage(source) or shared_storage(target) else None
    start, count = time.monotonic(), 0
    result, owned = hashlib.sha256(), None
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            info = os.fstat(writer.fileno())
            owned = (info.st_dev, info.st_ino)
            for block in iter(lambda: reader.read(4 * 1024**2), b""):
                writer.write(block)
                result.update(block)
                count += len(block)
                pace_io(rate, count, start)
            writer.flush()
            os.fsync(writer.fileno())
        observed = result.hexdigest()
        if observed != reference:
            raise ValueError(f"source SHA256 mismatch: {source}")
        if digest(temporary) != observed:
            raise ValueError(f"independent archive readback failed: {target}")
        info = temporary.lstat()
        if (info.st_dev, info.st_ino) != owned:
            raise ValueError("copy staging identity changed before publication")
        os.link(temporary, target, follow_symlinks=False)
        return observed
    finally:
        if owned is not None:
            try:
                info = temporary.lstat()
                if (info.st_dev, info.st_ino) == owned:
                    temporary.unlink()
            except FileNotFoundError:
                pass


def publish(args):
    configure(safe_path(args.work))
    api = load_api(args.native_source, args.work)
    from exp12_layer_rom_dataset import LayerROMDataset
    from exp12_port_basis_contract import validate_dataset_port_basis_contracts
    archive = safe_path(args.archive)
    canonical = Path("/app/nanocad/projects/personal/zchen/YAP_STACK/Ansys_Verify/12_stack_model_validation/results")
    if archive.parent != canonical or not archive.name.startswith("prepared_p2_totals_"):
        raise ValueError("publication requires a new neutral canonical results directory")
    if archive.exists():
        raise FileExistsError("publication refuses existing destinations; review partial artifacts manually")
    parent_path = safe_path(args.parent_package)
    parent = read_json(parent_path)
    parent_sha = digest(parent_path)
    if (parent.get("schema") != "exp12.ofat.geometry_package.v1" or parent.get("name") != "BI2_50"
            or parent.get("ready") is not True or parent.get("state") != "READY_ENGINEERING_PREVIEW"
            or parent.get("rank") != 64 or parent.get("port_basis_sha256") != BASIS_SHA):
        raise ValueError("wrong p2 parent")
    audits, ready = {}, {}
    for total, si in TOTALS.items():
        case = api.CASES[f"bi2_tsv_h{int(si)}"]
        if read_json(api.ROOT / case.key / "RUN_STATUS.json")["state"] != "COMPLETE_ENGINEERING_CANDIDATE":
            raise RuntimeError("both target builds must complete before admission")
        plan = read_json(api.ROOT / case.key / "STATIC_PLAN.json")
        api.verify_inputs(plan)
        for step in plan["steps"]:
            api.verify_static_stage(case, step)
        audits[total] = transfer_audit(api, case)
        ready[total] = read_json(api.ROOT / case.key / "DATASET_READY.json")
    archive.mkdir()
    inventory = {}
    def copy(source, relative, expected=None):
        if relative in inventory:
            observed = digest(source)
            if inventory[relative] != observed or (expected is not None and expected != observed):
                raise ValueError(f"conflicting archive destination: {relative}")
            return relative
        inventory[relative] = copy_checked(source, archive / relative, expected)
        return relative
    copy(parent_path, "shared/parent_BI2_50.json", parent_sha)
    copy(parent["port_basis"], "shared/port_basis.npz", BASIS_SHA)
    shared_dataset, shared_prepared, records, prepared_records = {}, {}, {}, {}
    for key in ("bi", "beol"):
        shared_dataset[key] = copy(parent["datasets"][key], f"shared/{key}.npz", parent["dataset_sha256"][key])
        shared_prepared[key] = copy(parent["prepared"][key], f"shared/{key}_prepared.npz")
    catalog = dict(schema="yap.prepared_p2_operator_catalog.v1", packages={})
    for total, si in TOTALS.items():
        case = api.CASES[f"bi2_tsv_h{int(si)}"]
        key, output = case.library_key, api.ROOT / case.key
        sources = {**shared_dataset, key: copy(ready[total]["dataset_file"], f"datasets/{key}.npz", ready[total]["dataset_sha256"])}
        prepared = {**shared_prepared, key: copy(ready[total]["prepared_geometry_file"], f"prepared/{key}.npz", ready[total]["prepared_geometry_sha256"])}
        loaded = {name: LayerROMDataset.load(archive / relative, validate=True, verify_content=True)
                  for name, relative in sources.items()}
        gauge = validate_dataset_port_basis_contracts(loaded, archive / "shared/port_basis.npz")
        for name, dataset in loaded.items():
            with __import__("numpy").load(archive / sources[name], allow_pickle=False) as data:
                content = str(data["content_sha256"].item())
            records[name] = dict(sha256=inventory[sources[name]], dataset_content_sha256=content,
                                 cu_point_count=len(dataset.cu_weights_mm3), thickness_mm=dataset.geometry_signature["thickness_mm"],
                                 macro_size=dataset.macro_size, bubble_size=dataset.bubble_size)
            with __import__("numpy").load(archive / prepared[name], allow_pickle=False) as data:
                content = json.loads(str(data["metadata_json"].item()))["source_content_sha256"]
            if content != records[name]["dataset_content_sha256"]:
                raise ValueError("prepared geometry does not bind its target dataset")
            prepared_records[name] = dict(sha256=inventory[prepared[name]], parent_dataset_content_sha256=content)
        if (abs(records[key]["thickness_mm"] * 1000. - si) > 1e-9 or records[key]["macro_size"] != 134
                or records[key]["cu_point_count"] != 64160
                or records[key]["bubble_size"] != audits[total]["rank"]
                or records["bi"]["cu_point_count"] != 19440 or records["beol"]["cu_point_count"] != 102096):
            raise ValueError("new target dataset geometry mismatch")
        del loaded
        receipt_hashes = {}
        files = ["STATIC_PLAN.json", "TEMPLATE_PRODUCER_VERSIONS.json", "LAUNCH.json", "RUN_STATUS.json", "DATASET_READY.json", "DATASET_BUILD_INPUTS.json",
                 "bubble/TRANSFER_MANIFEST.json", "bubble/M4_DIRECT_ZERO_TRACE_BUBBLE_BASIS.npz",
                 "bubble/AUDITED_FINAL_BUBBLE_SPACE.npz"]
        files.extend(f"static/{stage}_FROZEN_OUTPUTS.json" for stage in STAGES)
        files.extend(f"{stage}.log" for stage in STAGES)
        files.extend(["dataset.log", "prepare.log", "all_gauss/M4_ALL_GAUSS_LAYER_ROM_DATASET_SUMMARY.json"])
        for relative in files:
            destination = f"build_receipts/{case.key}/{relative}"
            expected = {"bubble/M4_DIRECT_ZERO_TRACE_BUBBLE_BASIS.npz": audits[total]["raw_transferred_bubble_sha256"],
                        "bubble/AUDITED_FINAL_BUBBLE_SPACE.npz": audits[total]["transferred_bubble_sha256"]}.get(relative)
            copy(output / relative, destination, expected)
            receipt_hashes[destination] = inventory[destination]
        for step in read_json(output / "STATIC_PLAN.json")["steps"]:
            for source, expected in api.verify_static_stage(case, step).items():
                relative = str(Path(source).relative_to(output))
                destination = f"build_receipts/{case.key}/{relative}"
                copy(source, destination, expected)
                receipt_hashes[destination] = inventory[destination]
        audit = dict(schema="yap.prepared_p2_operator_audit.v1", state="PASS_STRUCTURAL_ENGINEERING_CANDIDATE",
                     total_thickness_um=total, retained_si_thickness_um=si, bonding_pitch_um=2.,
                     bonding_pad_diameter_um=1., rank=64, face_node_shape=[46, 46], cell_area_mm2=.006**2,
                     port_basis_sha256=BASIS_SHA,
                     parent_package_sha256=parent_sha, datasets={name: records[name] for name in sources},
                     prepared={name: prepared_records[name] for name in prepared}, stage_count=6,
                     all_cu_points_retained=True, thermal_geometry_validated=False,
                     transfer_audit=audits[total], gauge_contract=gauge, producer_receipts_sha256=receipt_hashes,
                     scope="Exact target geometry and native64 ports; transferred h38.5 homogeneous interior space; no target thermal validation")
        audit_path = archive / f"audit/BI2_{total}.json"
        write_json(audit_path, audit, immutable=True)
        audit_sha = digest(audit_path)
        inventory[str(audit_path.relative_to(archive))] = audit_sha
        package = dict(schema="yap.prepared_p2_geometry_package.v1", name=f"BI2_{total}", ready=False,
                       state="CANDIDATE_NOT_VALIDATED", rank=64, port_basis="../shared/port_basis.npz", port_basis_sha256=BASIS_SHA,
                       datasets={name: "../" + relative for name, relative in sources.items()},
                       prepared={name: "../" + relative for name, relative in prepared.items()},
                       dataset_sha256={name: records[name]["sha256"] for name in sources},
                       prepared_sha256={name: prepared_records[name]["sha256"] for name in prepared},
                       parent_package="../shared/parent_BI2_50.json", parent_package_sha256=parent_sha,
                       audit_receipt=f"../audit/BI2_{total}.json", audit_receipt_sha256=audit_sha)
        package_path = archive / f"packages/BI2_{total}.json"
        write_json(package_path, package, immutable=True)
        package_sha = digest(package_path)
        inventory[str(package_path.relative_to(archive))] = package_sha
        catalog["packages"][str(total)] = dict(package_sha256=package_sha, audit_sha256=audit_sha,
                                              parent_package_sha256=parent_sha, port_basis_sha256=BASIS_SHA)
    copy(__file__, "frozen_sources/build_prepared_operators.py")
    copy(api.__file__, "frozen_sources/native_geometry_producer.py")
    write_json(archive / "CATALOG.json", catalog, immutable=True)
    inventory["CATALOG.json"] = digest(archive / "CATALOG.json")
    write_json(archive / "ARCHIVE.json", dict(schema="yap.prepared_operator_archive.v1", files_sha256=inventory,
               source_work=str(safe_path(args.work)), thermal_geometry_validated=False), immutable=True)
    print(json.dumps(dict(archive=str(archive), catalog=catalog), indent=2), flush=True)


def checked_inventory(root, path):
    from package_assembly_runtime import confined_source
    path = confined_source(root, path)
    if not path.is_file() or not 1 <= path.stat().st_size <= 1024**2:
        raise ValueError("inventory must be a bounded regular file")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate inventory key")
            result[key] = value
        return result
    with path.open("rb") as stream:
        payload = stream.read(1024**2 + 1)
    if len(payload) > 1024**2:
        raise ValueError("oversized inventory")
    return json.loads(payload, object_pairs_hook=unique), hashlib.sha256(payload).hexdigest(), path


def stage_resources(args, baseline, old, old_sha, new, new_sha):
    path = baseline / "runtime/RESOURCES.json"
    legacy = getattr(args, "legacy_resource_map", None)
    if legacy is None:
        document, sha, path = checked_inventory(baseline, path)
        if old["files_sha256"].get("YAP/runtime/RESOURCES.json") != sha:
            raise ValueError("resource map differs from its immutable inventory")
        return document, sha, path
    if (os.path.lexists(path) or "YAP/runtime/RESOURCES.json" in old["files_sha256"]
            or old_sha != LEGACY_STAGE_PINS["bundle"] or old.get("delivery_revision") != 3
            or new_sha != LEGACY_STAGE_PINS["archive"]):
        raise ValueError("legacy reconciliation requires the exact reviewed map-free receipts")
    path = Path(legacy).expanduser().absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("legacy map must not use symlinks")
    path = safe_path(path)
    document, sha, path = checked_inventory(path.parent, path)
    if sha != LEGACY_STAGE_PINS["resources"]:
        raise ValueError("legacy resource map differs from its separately reviewed pin")
    mappings = document.get("resources", {})
    if document.get("schema") != "yap.native_resources.v1" or not isinstance(mappings, dict):
        raise ValueError("invalid legacy resource map")
    for key, pin in (("compact_p2", "compact"), ("geometry", "parent")):
        value = mappings.get(key)
        if not isinstance(value, str):
            raise ValueError("legacy map lacks its reviewed directory bindings")
        endpoint = Path("YAP/runtime/native") / value / "packages/BI2_50.json"
        if old["files_sha256"].get(str(endpoint)) != LEGACY_STAGE_PINS[pin]:
            raise ValueError("legacy map does not bind the original manifest endpoints")
    if new["files_sha256"].get("shared/parent_BI2_50.json") != LEGACY_STAGE_PINS["parent"]:
        raise ValueError("legacy native parent differs from the published parent")
    return document, sha, path


def application_files(root):
    from package_assembly_runtime import confined_source, delivery_profiles
    root = Path(root).expanduser().absolute()
    if any(parent.is_symlink() for parent in (root, *root.parents)):
        raise ValueError("application source must not use symlinks")
    selected = []
    for relative in (".", "D2W", "W2W", "D2W/utils", "W2W/utils"):
        directory = confined_source(root, root / relative)
        files = list(directory.glob("*.py"))
        if len(files) > 128:
            raise ValueError("unexpected application-directory size")
        selected.extend(files)
    selected.extend(root / name for name in (
        "requirements.txt", "LICENSE", "README.md", "PORTABLE_ASSEMBLY.md",
        "W2W/ASSEMBLY_ORDER.md", "prepared_operator_catalog.json"))
    for profile in delivery_profiles(True):
        domain, name = profile.split("/")
        files = list(confined_source(root, root / domain / "input" / name).iterdir())
        if len(files) != 8:
            raise ValueError("unexpected complete application-profile inputs")
        selected.extend(files)
        selected.append(root / domain / "configs" / name / f"{name}.yaml")
    if len(selected) > 256 or len(set(selected)) != len(selected):
        raise ValueError("application source list is not bounded and unique")
    result = []
    for path in selected:
        path = confined_source(root, path)
        if not path.is_file():
            raise ValueError("application inputs must be regular files")
        result.append((path, Path("source") / path.relative_to(root), digest(path)))
    return result


def stage_preflight(args):
    """No writes before local-storage, source-confinement and inventory checks."""
    from package_assembly_runtime import confined_source
    work = safe_path(args.work)
    if not work.is_relative_to(OWNED_WORK_ROOT):
        raise ValueError("staging requires owned local scratch")
    baseline, archive = safe_path(args.baseline_runtime), safe_path(args.archive)
    legacy = getattr(args, "legacy_resource_map", None)
    destination = work / ("delivery_source_legacy_v5" if legacy is not None else "delivery_source")
    for target in (work / "runtime", destination):
        confined_source(work, target)
    for source in (baseline, archive):
        for target in (work / "runtime", destination):
            if source.is_relative_to(target) or target.is_relative_to(source):
                raise ValueError("immutable input and writable staging roots overlap")
    resource_check(work)
    if destination.exists():
        raise FileExistsError("existing delivery staging requires manual review")
    old, old_sha, old_manifest = checked_inventory(baseline, baseline / "runtime/BUNDLE.json")
    new, new_sha, new_manifest = checked_inventory(archive, archive / "ARCHIVE.json")
    if (old.get("schema") != "yap.portable_assembly_p2.v1"
            or new.get("schema") != "yap.prepared_operator_archive.v1"
            or new.get("thermal_geometry_validated") is not False):
        raise ValueError("incorrect immutable source inventories")
    for document in (old, new):
        files = document.get("files_sha256")
        if not isinstance(files, dict) or not 1 <= len(files) <= 512:
            raise ValueError("unexpected bounded source inventory")
        for relative, expected in files.items():
            if (not isinstance(relative, str) or not relative or len(relative) > 512
                    or not isinstance(expected, str) or len(expected) != 64
                    or any(char not in "0123456789abcdef" for char in expected)):
                raise ValueError("invalid inventory binding")
    selected = []
    for relative, expected in old["files_sha256"].items():
        name = Path(relative)
        if name.is_relative_to("YAP/runtime/native") or str(name) == "YAP/runtime/200_bow_diff_vs_magnification/nonlinear_face_lever.py":
            selected.append((baseline.parent / name, name.relative_to("YAP"), expected))
    if not 16 <= len(selected) <= 256 or not 1 <= len(new["files_sha256"]) <= 256:
        raise ValueError("unexpected bounded runtime inventory")
    resources, resources_sha, resources_path = stage_resources(args, baseline, old, old_sha, new, new_sha)
    for target in (work / "runtime", destination):
        if resources_path.is_relative_to(target):
            raise ValueError("resource input overlaps writable staging storage")
    if resources.get("schema") != "yap.native_resources.v1" or not isinstance(resources.get("resources"), dict) \
            or not {"compact_p2", "geometry"} <= set(resources["resources"]):
        raise ValueError("baseline lacks required logical resource bindings")
    native = destination / "runtime/native"
    for relative, expected in new["files_sha256"].items():
        selected.append((archive / relative, Path("runtime/native/results/prepared_p2_totals_20261006_v1") / relative, expected))
    application_source = getattr(args, "application_source", None)
    if application_source is not None:
        source_root = Path(application_source).expanduser().absolute()
        for target in (work / "runtime", destination):
            if source_root.is_relative_to(target) or target.is_relative_to(source_root):
                raise ValueError("application source overlaps writable staging storage")
    application = application_files(application_source) if application_source is not None else []
    selected.extend(application)
    if len(selected) > 511:
        raise ValueError("combined staging inventory exceeds 511 files")
    if len({relative for _, relative, _ in selected}) != len(selected):
        raise ValueError("staged destination collision")
    mappings = {key: resources["resources"][key] for key in ("compact_p2", "geometry")}
    for value in mappings.values():
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            raise ValueError("resources require relative native-directory bindings")
        confined_source(native, native / value)
        prefix = Path("runtime/native") / value
        if not any(relative.is_relative_to(prefix) for _, relative, _ in selected):
            raise ValueError("resource has no inventoried runtime closure")
    verified = []
    for source, relative, expected in selected:
        if relative.is_relative_to("source"):
            source = confined_source(Path(application_source).absolute(), source)
        else:
            source = confined_source(baseline if source.is_relative_to(baseline) else archive, source)
        confined_source(destination, destination / relative)
        if not source.is_file() or digest(source) != expected:
            raise ValueError(f"staging source hash mismatch: {source}")
        verified.append((source, relative, expected))
    total = sum(source.stat().st_size for source, _, _ in verified)
    if shutil.disk_usage(work).free < total + 64 * 1024**3:
        raise OSError("insufficient staging scratch space")
    metadata = {old_manifest: old_sha, new_manifest: new_sha, resources_path: resources_sha}
    metadata.update({source: expected for source, _, expected in application})
    return work, destination, native, verified, mappings, total, metadata


def stage_delivery(args):
    """Copy only inventoried runtime files into a new confined local source."""
    work, destination, native, selected, mappings, total, source_metadata = stage_preflight(args)
    configure(work)
    destination.mkdir()
    inventory = {}
    for source, relative, expected in selected:
        inventory[str(relative)] = copy_checked(source, destination / relative, expected)
        (destination / relative).chmod(0o444)
    for path, expected in source_metadata.items():
        if digest(path) != expected:
            raise ValueError("source inventory changed during staging")
    mappings["prepared_p2_native"] = "results/prepared_p2_totals_20261006_v1"
    resource_map = destination / "runtime/RESOURCES.json"
    write_json(resource_map, dict(schema="yap.native_resources.v1", resources=mappings), immutable=True)
    inventory["runtime/RESOURCES.json"] = digest(resource_map)
    write_json(destination / "STAGE.json", dict(schema="yap.prepared_runtime_stage.v1",
               source_metadata_sha256={str(path): expected for path, expected in source_metadata.items()},
               files_sha256=inventory, bytes_copied=total, source_payloads_rewritten=False,
               legacy_map_supplemented=getattr(args, "legacy_resource_map", None) is not None,
               application_source="source" if getattr(args, "application_source", None) is not None else None), immutable=True)
    print(json.dumps(dict(native_source=str(native), resource_map=str(resource_map), bytes_copied=total,
                          stage_inventory=str(destination / "STAGE.json"),
                          stage_inventory_sha256=digest(destination / "STAGE.json"))), flush=True)


def export_staged(args):
    """Authenticate frozen executable inputs before importing the staged application."""
    from package_assembly_runtime import confined_source
    work = safe_path(args.work)
    if not work.is_relative_to(OWNED_WORK_ROOT):
        raise ValueError("export requires owned local scratch")
    root = work / "delivery_source_legacy_v5"
    confined_source(work, work / "runtime")
    if not args.stage_inventory_sha256:
        raise ValueError("export requires an independently pinned staging receipt")
    document, sha, receipt = checked_inventory(root, root / "STAGE.json")
    if (sha != args.stage_inventory_sha256 or document.get("schema") != "yap.prepared_runtime_stage.v1"
            or document.get("application_source") != "source"
            or not isinstance(document.get("files_sha256"), dict)
            or not 1 <= len(document["files_sha256"]) <= 512):
        raise ValueError("wrong frozen application staging receipt")
    executable = root / "source/package_assembly_runtime.py"
    if "source/package_assembly_runtime.py" not in document["files_sha256"]:
        raise ValueError("staging receipt omits the exporter executable")
    for relative, expected in document["files_sha256"].items():
        path = confined_source(root, root / relative)
        if relative.startswith("source/") or path.suffix == ".py" or relative == "runtime/RESOURCES.json":
            if not path.is_file() or digest(path) != expected:
                raise ValueError("frozen executable or configuration input changed")
    destination = safe_path(args.archive)
    parent = Path("/app/nanocad/projects/personal/zchen/YAP_STACK/distribution")
    if destination.parent != parent or destination.name != "yap_prepared_p2_20261006_v1" or destination.exists():
        raise ValueError("export requires the new reviewed canonical delivery directory")
    environment = configure(work)
    environment["YAP_WORK_DIR"] = str(work / "runtime")
    environment.pop("YAP_NATIVE_RUNTIME_DIR", None)
    environment.pop("YAP_RUNTIME_RESOURCES", None)
    environment.pop("PYTHONPATH", None)
    resource_check(work)
    command = [sys.executable, "-I", "-B", "-c",
               "import runpy,sys; sys.path.insert(0,sys.argv[1]); sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0],run_name='__main__')",
               str(executable.parent), str(executable), "--native-source", str(root / "runtime/native"),
               "--resource-map", str(root / "runtime/RESOURCES.json"), "--output-dir", str(destination),
               "--prepared-operators", "--stage-inventory", str(receipt), "--stage-inventory-sha256", sha]
    write_json(work / "EXPORT_LAUNCH_V5.json", dict(wrapper_sha256=digest(__file__),
               stage_inventory_sha256=sha, command=command, resources=resource_check(work)), immutable=True)
    # All runtime paths are explicit before the isolated child imports application code.
    run_stage(command, work / "export_legacy_v5.log", "export_prepared_delivery", environment=environment)
    write_json(work / "EXPORT_READY_V5.json", dict(stage_inventory_sha256=sha,
               delivery=str(destination), bundle_sha256=digest(destination / "YAP/runtime/BUNDLE.json")), immutable=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "publish", "stage", "export"))
    parser.add_argument("--total", type=int, choices=tuple(TOTALS))
    parser.add_argument("--native-source", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--baseline-runtime", type=Path)
    parser.add_argument("--parent-package", type=Path)
    parser.add_argument("--legacy-resource-map", type=Path,
                        help="Explicit reconciliation of the pinned revision-3 map-free baseline")
    parser.add_argument("--application-source", type=Path,
                        help="Freeze the explicit prepared-delivery application inputs during staging")
    parser.add_argument("--stage-inventory-sha256")
    args = parser.parse_args()
    if args.action == "build":
        if args.total is None:
            parser.error("build requires --total")
        build(args)
    elif args.action == "publish":
        if args.archive is None or args.parent_package is None:
            parser.error("publish requires --archive and --parent-package")
        publish(args)
    elif args.action == "stage":
        if args.archive is None or args.baseline_runtime is None:
            parser.error("stage requires --archive and --baseline-runtime")
        stage_delivery(args)
    else:
        if args.archive is None or not args.stage_inventory_sha256:
            parser.error("export requires --archive and --stage-inventory-sha256")
        export_staged(args)


if __name__ == "__main__":
    main()
