"""Plan remaining operator work; execute only explicitly configured foreground recipes.

Planning reads a finite catalog, never numerical payloads or directory trees.
The queue does not manufacture geometry support, scientific admission, launch
approvals or publication receipts. Those remain the scientific backend's job.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import stat
import string
import subprocess
import sys
import time
import warnings


STAGES = ("reference", "construct", "history", "assess", "publish")
LIMIT = 2 * 1024**2
GIB = 1024**3
SCHEMA = "yap.operator_production.v1"
# Add only after an adapter has been qualified with this queue's foreground
# process/environment contract. Existing geometry-specific owners are not yet
# compatible merely because their individual historical executions passed.
QUALIFIED_BACKENDS = {}
FIELDS = {"python", "project_root", "repo_root", "run_dir", "case_dir", "stage_dir",
          "pitch_um", "total_um", "si_um", "coupon_total_um", "cell_um", "diameter_um", "variant"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def literal(value, *, output=False):
    path = Path(value).expanduser().absolute()
    require(str(path) == os.path.normpath(path)
            and not {"..", "#snapshot", ".snapshot"}.intersection(path.parts), "literal nonsnapshot path required")
    require(not any(part.is_symlink() for part in (path, *path.parents)), "symlink paths are not queue inputs/outputs")
    if output:
        resolved = path.resolve()
        require(not any(resolved == root or root in resolved.parents for root in
                        (Path("/tmp"), Path("/u1/ee/zhichao"), Path("/w/ee.00/puneet/zhichao"))),
                "experiment outputs must not use /tmp or the author's shared home")
    return path


def read_json(path):
    path = literal(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size <= LIMIT, "bounded regular JSON required")
        raw = stream.read(LIMIT + 1)
        after = os.fstat(stream.fileno())
    require(identity(before) == identity(after) and len(raw) == before.st_size, "JSON changed during read")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    require(type(value) is dict, "JSON object required")
    return value, hashlib.sha256(raw).hexdigest()


def emit(path, value):
    path = literal(path, output=True)
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    require(len(raw) <= LIMIT, "bounded queue receipt required")
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    return hashlib.sha256(raw).hexdigest()


def source_hash(path):
    path = literal(path)
    require(path.stat().st_size <= LIMIT, "only small explicit sources/reports may be queue pins")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(LIMIT + 1)
        after = os.fstat(stream.fileno())
    require(len(raw) <= LIMIT and identity(before) == identity(after) and len(raw) == before.st_size,
            "source/report changed or exceeds bound")
    return hashlib.sha256(raw).hexdigest()


def load_config(path):
    from omegaconf import OmegaConf
    path = literal(path)
    require(Path(path).stat().st_size <= LIMIT, "bounded YAML required")
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    require(type(value) is dict and value.get("schema") == SCHEMA, "operator-production YAML schema differs")
    return value


def grid(config, catalog):
    pitches, totals = config["pitches_um"], config["prepared_totals_um"]
    require(type(pitches) is list and type(totals) is list and pitches and totals
            and len(pitches) == len(set(pitches)) and len(totals) == len(set(totals))
            and all(type(p) is int and 2 <= p <= 10 for p in pitches)
            and all(type(t) is int and 20 <= t <= 100 and t % 10 == 0 for t in totals), "invalid requested grid")
    require(config.get("diameter_pitch_ratio") == .5, "this campaign fixes d/p=0.5")
    existing = {}
    for entry in catalog.get("entries", []):
        geometry = entry.get("geometry", {})
        if (geometry.get("diameter_pitch_ratio") == .5 and geometry.get("bi_layer_count") == 1
                and geometry.get("bi_thickness_um") == 1.5 and geometry.get("beol_thickness_um") == 10):
            key = geometry["bi_pitch_um"], geometry["si_tsv_thickness_um"]
            require(key not in existing, "ambiguous catalog geometry")
            existing[key] = entry["id"]
    rows = []
    for pitch in sorted(pitches):
        for total in sorted(totals):
            for variant, si in (("inner", total - 13), ("outer", total - 11.5)):
                rows.append(dict(id=f"BI{pitch}_T{total}_{variant}", pitch_um=pitch, total_um=total,
                    variant=variant, si_um=si, coupon_total_um=si + 11.5,
                    cell_um=math.lcm(pitch, 6), diameter_um=pitch / 2,
                    catalog_entry=existing.get((pitch, si)), payload_verified=False,
                    assembly_accuracy_inferred=False, interpolation_accuracy_inferred=False))
    return rows


def validate_recipe(recipe):
    require(type(recipe) is dict and recipe.get("stage") in STAGES, "explicit recipe stage required")
    select = recipe.get("select", {})
    require(set(select) == {"pitch_um", "total_um", "variant"}, "exact recipe geometry selector required")
    require(select["variant"] in ("inner", "outer"), "recipe variant differs")
    argv = recipe.get("argv")
    require(type(argv) is list and argv and all(type(item) is str and item for item in argv), "foreground argv list required")
    require(recipe.get("foreground") is True, "detached dispatchers/launchers are not queue recipes")
    for token in argv:
        for _, field, formatting, conversion in string.Formatter().parse(token):
            require(field is None or field in FIELDS and not formatting and not conversion, "unsupported command placeholder")
    for name in ("source_pins", "prerequisites"):
        pins = recipe.get(name, {})
        require(type(pins) is dict, "explicit source/prerequisite pins required")
        for path, sha in pins.items():
            literal(path)
            require(type(sha) is str and len(sha) == 64 and set(sha) <= set("0123456789abcdef"), "full SHA pin required")
    require(recipe.get("source_pins"), "recipe must pin its actual foreground entry source")
    require(recipe["stage"] == "reference" or recipe.get("prerequisites"),
            "later stages must bind actual closed predecessor evidence")
    report = recipe.get("report", {})
    require(type(report) is dict and type(report.get("path")) is str
            and type(report.get("equals")) is dict and report["equals"], "explicit JSON scope/closure checks required")
    for name in ("memory_gib", "timeout_seconds", "file_cap_gib"):
        require(type(recipe.get(name)) is int and recipe[name] > 0, "positive recipe resource limit required: " + name)
    return recipe


def make_plan(config, catalog, *, run_dir, repo_root, python):
    rows = grid(config, catalog)
    if literal(config["project_root"]) not in literal(run_dir, output=True).parents:
        warnings.warn("Run directory is outside project_root; ensure it is your own writable storage. "
                      "Shared operators and source remain read-only.", stacklevel=2)
    recipes = {}
    for recipe in config.get("recipes", []):
        validate_recipe(recipe)
        s = recipe["select"]
        key = (s["pitch_um"], s["total_um"], s["variant"], recipe["stage"])
        require(key not in recipes, "duplicate recipe selector/stage")
        recipes[key] = recipe
    tasks = []
    used = set()
    for row in rows:
        for stage in STAGES:
            key = row["pitch_um"], row["total_um"], row["variant"], stage
            recipe = recipes.get(key)
            if recipe:
                used.add(key)
            # Published component metadata skips rebuilding, but is not payload
            # readback, full assembly qualification or a new scientific pass.
            status = "CONFIGURED_REQUIRES_PREFLIGHT" if recipe else (
                "CATALOGUED_SCOPE_ONLY" if row["catalog_entry"] else "BLOCKED_IMPLEMENTATION")
            tasks.append(dict(id=row["id"] + "_" + stage, case=row["id"], stage=stage,
                              status=status, recipe=recipe))
    require(set(recipes) == used, "recipe selector outside requested grid")
    resources = config.get("resources", {})
    require(type(resources.get("cores_per_job")) is int and resources["cores_per_job"] > 0
            and type(resources.get("reserve_ram_gib")) is int and resources["reserve_ram_gib"] >= 0,
            "explicit CPU and RAM reserve configuration required")
    interpolation = []
    for pitch in sorted(config["pitches_um"]):
        for left, right in zip(sorted(config["prepared_totals_um"]), sorted(config["prepared_totals_um"])[1:]):
            for variant in ("inner", "outer"):
                interpolation.append(dict(pitch_um=pitch, variant=variant,
                    left_prepared_total_um=left, right_prepared_total_um=right,
                    heldout_prepared_total_um=(left + right) / 2,
                    purpose="validate parameterized/interpolated bow, not construct an exact-point operator",
                    exact_point_operator_requested=False, status="BLOCKED_IMPLEMENTATION"))
    cells = sorted({row["cell_um"] for row in rows})
    return dict(schema=SCHEMA, project_root=str(literal(config["project_root"])),
        repo_root=str(literal(repo_root)), run_dir=str(literal(run_dir, output=True)),
        python=str(literal(Path(python).resolve(strict=True))),
        resources=resources, rows=rows, tasks=tasks, experiments_launched=False,
        component_targets=dict(bi_pitches_um=sorted(config["pitches_um"]),
            si_thicknesses_um=sorted({row["si_um"] for row in rows}), beol_thickness_um=10,
            cell_families=[dict(cell_um=cell, pitches_um=sorted({row["pitch_um"] for row in rows if row["cell_um"] == cell}),
                                reuse_certified=False) for cell in cells]),
        interpolation_validation_targets=interpolation,
        diameter_pitch_ratio=.5, catalog_metadata_only=True, catalog_payloads_read=False,
        scientific_acceptance="backend-scoped cooldown bow relative error <=10%; never exit-code-only",
        stages=list(STAGES), automatic_retry=False, automatic_cleanup=False,
        component_reuse="Reuse only backend-authenticated compatible cell/mesh/interface/material/frame identities",
        coverage_scope="81 nominal design queries are not 81 independent RVE builds or continuous-thickness proof")


def plan_command(args):
    config = load_config(args.config)
    root = literal(config["project_root"])
    catalog_path = literal(root / config["catalog"])
    catalog, catalog_sha = read_json(catalog_path)
    run = literal(args.run_dir, output=True)
    require(not run.exists(), "fresh planning directory required; reuse an existing PLAN with run/status")
    plan = make_plan(config, catalog, run_dir=run, repo_root=Path(__file__).resolve().parents[2], python=sys.executable)
    plan["catalog"] = dict(path=str(catalog_path), sha256=catalog_sha, entries=len(catalog.get("entries", [])))
    plan["config"] = dict(path=str(literal(args.config)), sha256=source_hash(args.config))
    plan["driver_sha256"] = source_hash(__file__)
    run.mkdir(mode=0o700, parents=True)
    sha = emit(run / "PLAN.json", plan)
    with (run / "COVERAGE.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plan["rows"][0]))
        writer.writeheader()
        writer.writerows(plan["rows"])
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "run", "--plan", str(run / "PLAN.json"),
               "--plan-sha256", sha, "--execute"]
    with (run / "run_remaining.sh").open("x") as stream:
        stream.write("#!/usr/bin/env bash\nset -euo pipefail\nexec " + shlex.join(command) + ' "$@"\n')
    (run / "run_remaining.sh").chmod(0o700)
    print(json.dumps(summary(plan), indent=2))
    print("Plan:", run / "PLAN.json", "SHA256:", sha)
    print("No experiments launched. Later: bash", shlex.quote(str(run / "run_remaining.sh")), "--stage construct --jobs 1")


def summary(plan):
    counts = {}
    for task in plan["tasks"]:
        counts[task["status"]] = counts.get(task["status"], 0) + 1
    return dict(nominal_design_queries=len(plan["rows"]) // 2, endpoint_variants=len(plan["rows"]),
                catalogued_variants=sum(bool(row["catalog_entry"]) for row in plan["rows"]),
                stage_counts=counts, interpolation_validation_targets=len(plan["interpolation_validation_targets"]),
                logical_BI_targets=len(plan["component_targets"]["bi_pitches_um"]),
                logical_Si_targets=len(plan["component_targets"]["si_thicknesses_um"]),
                experiments_launched=False, payloads_verified=False)


def resolve_task(plan, task):
    row = next(row for row in plan["rows"] if row["id"] == task["case"])
    stage_dir = Path(plan["run_dir"]) / "attempts" / task["id"]
    values = dict(row, python=plan["python"], project_root=plan["project_root"], repo_root=plan["repo_root"],
                  run_dir=plan["run_dir"], case_dir=str(stage_dir.parent / task["case"]), stage_dir=str(stage_dir))
    recipe = task["recipe"]
    argv = [token.format_map(values) for token in recipe["argv"]]
    report = literal(recipe["report"]["path"].format_map(values))
    require(stage_dir in report.parents, "new recipe report must be inside its private stage directory")
    return argv, report, stage_dir


def report_checks(path, checks):
    value, sha = read_json(path)
    for pointer, expected in checks.items():
        require(pointer.startswith("/"), "report check uses a JSON pointer")
        observed = value
        for key in pointer[1:].split("/"):
            require(not re.search(r"~(?![01])", key), "invalid JSON pointer escape")
            key = key.replace("~1", "/").replace("~0", "~")
            if type(observed) is list:
                require(re.fullmatch(r"0|[1-9][0-9]*", key), "invalid JSON array index")
                observed = observed[int(key)]
            else:
                require(type(observed) is dict, "JSON pointer traverses a scalar")
                observed = observed[key]
        require(type(observed) is type(expected) and observed == expected, "scope/closure check failed: " + pointer)
    return dict(path=str(path), sha256=sha)


def available_ram():
    with Path("/proc/meminfo").open() as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise ValueError("MemAvailable unavailable")


def environment(stage, cores):
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("XDG_") or key in ("PYTHONPATH", "PYTHONUSERBASE"):
            env.pop(key)
    for key in ("HOME", "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
                "XDG_STATE_HOME", "XDG_RUNTIME_DIR", "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "PYTHONPYCACHEPREFIX"):
        path = stage / "environment" / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS=str(cores), NUMBA_NUM_THREADS=str(cores))
    return env


def check_pins(recipe):
    sources, prerequisites = recipe["source_pins"], recipe.get("prerequisites", {})
    require(all(sources[path] == prerequisites[path] for path in sources.keys() & prerequisites.keys()),
            "conflicting source/prerequisite pin drift")
    for path, sha in {**recipe["source_pins"], **recipe.get("prerequisites", {})}.items():
        require(source_hash(path) == sha, "recipe source/prerequisite drift: " + path)


def check_backend(recipe, argv):
    backend = QUALIFIED_BACKENDS.get(recipe.get("backend"))
    require(backend is not None, "BLOCKED_BACKEND_QUALIFICATION: no reviewed queue-compatible owner")
    require(recipe["source_pins"].get(backend["entry"]) == backend["sha256"]
            and len(argv) >= 3 and argv[1] == "-B" and argv[2] == backend["entry"],
            "invoked foreground entry must match its qualified source pin")
    require(recipe["stage"] != "publish" or backend.get("evidence_gated_publisher") is True,
            "publication requires a qualified scientific publisher")


def terminate_owned(active):
    for item in active:
        if item["process"].poll() is None:
            os.killpg(item["process"].pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while any(item["process"].poll() is None for item in active) and time.monotonic() < deadline:
        time.sleep(.1)
    for item in active:
        if item["process"].poll() is None:
            os.killpg(item["process"].pid, signal.SIGKILL)
        item["process"].wait()
        item["log"].close()


def _run_command(args):
    plan, sha = read_json(args.plan)
    require(sha == args.plan_sha256 and plan.get("schema") == SCHEMA, "exact immutable PLAN SHA required")
    require(source_hash(__file__) == plan["driver_sha256"], "queue driver changed; create a new plan")
    require(read_json(plan["catalog"]["path"])[1] == plan["catalog"]["sha256"], "catalog changed; create a new plan")
    require(args.jobs > 0, "positive parallel job count required")
    tasks = [task for task in plan["tasks"] if task["stage"] == args.stage
             and (not args.case or task["case"] in args.case)]
    require(tasks, "no matching stage/case")
    ready = [task for task in tasks if task["status"] == "CONFIGURED_REQUIRES_PREFLIGHT"]
    print(json.dumps(dict(stage=args.stage, selected=len(tasks), configured=len(ready),
        catalogued=sum(task["status"] == "CATALOGUED_SCOPE_ONLY" for task in tasks),
        blocked=sum(task["status"] == "BLOCKED_IMPLEMENTATION" for task in tasks), execute=args.execute)), flush=True)
    if not args.execute:
        for task in ready:
            print(task["id"], shlex.join(resolve_task(plan, task)[0]))
        return 0
    require(ready, "no configured foreground backend for the requested remaining work; inspect PLAN tasks")
    require(plan["run_dir"] == str(literal(Path(args.plan).parent, output=True)), "PLAN moved; make a new plan")
    cores = plan["resources"]["cores_per_job"]
    cpus = sorted(os.sched_getaffinity(0))
    require(len(cpus) >= args.jobs * cores, "parallelism exceeds available CPU affinity")
    active, failed, skipped = [], [], []
    pending = list(ready)
    slots = list(range(args.jobs))
    # An old claim could still refer to live descendants. Do not launch other
    # tasks around it or treat a missing heartbeat as evidence of termination.
    for task in plan["tasks"]:
        stage = literal(Path(plan["run_dir"]) / "attempts" / task["id"], output=True)
        require(stage.parent == Path(plan["run_dir"]) / "attempts", "invalid task directory")
        if stage.exists() and not (stage / "COMPLETE.json").exists():
            print("MANUAL_REVIEW_REQUIRED:", task["id"], flush=True)
            return 1
    try:
        while pending or active:
            while pending and slots:
                task, slot = pending[0], slots[0]
                argv, report, stage = resolve_task(plan, task)
                receipt = stage / "COMPLETE.json"
                if stage.exists():
                    if receipt.exists():
                        done, _ = read_json(receipt)
                        check_pins(task["recipe"])
                        require(done["task"] == task["id"] and done["plan_sha256"] == sha
                                and done["report"] == report_checks(report, task["recipe"]["report"]["equals"]),
                                "completed-stage receipt changed")
                        skipped.append(task["id"])
                        pending.pop(0)
                        continue
                    failed.append(dict(task=task["id"], error="retained incomplete/failed attempt; manual review required"))
                    pending.pop(0)
                    continue
                # Timer and numerical process each inherit the full hard cap.
                reserved = sum(2 * item["memory_gib"] for item in active)
                needed = (plan["resources"]["reserve_ram_gib"] + reserved + 2 * task["recipe"]["memory_gib"]) * GIB
                if available_ram() < needed:
                    if active:
                        break
                    print("RESOURCE_BLOCKED: no job launched; retry manually when memory is available", flush=True)
                    return 75
                check_pins(task["recipe"])
                check_backend(task["recipe"], argv)
                stage.mkdir(mode=0o700, parents=True)
                emit(stage / "CLAIM.json", dict(task=task["id"], plan_sha256=sha,
                    host=socket.gethostname(), argv=argv, created_unix=time.time()))
                recipe = task["recipe"]
                cap, file_cap = recipe["memory_gib"] * GIB, recipe["file_cap_gib"] * GIB
                command = ["/usr/bin/prlimit", f"--as={cap}:{cap}", f"--fsize={file_cap}:{file_cap}",
                    "/usr/bin/taskset", "-c", ",".join(map(str, cpus[slot * cores:(slot + 1) * cores])),
                    "/usr/bin/timeout", "--signal=TERM", "--kill-after=10s", str(recipe["timeout_seconds"]) + "s", *argv]
                log = (stage / "worker.log").open("xb")
                try:
                    process = subprocess.Popen(command, cwd=plan["project_root"], env=environment(stage, cores),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                active.append(dict(task=task, process=process, log=log, slot=slot,
                    memory_gib=recipe["memory_gib"], report=report, stage=stage))
                pending.pop(0)
                slots.pop(0)
                print(f"START {task['id']} pid={process.pid} log={stage / 'worker.log'}", flush=True)
            for item in list(active):
                code = item["process"].poll()
                if code is None:
                    continue
                item["log"].close()
                task, stage = item["task"], item["stage"]
                try:
                    require(code == 0, "foreground backend exit=" + str(code))
                    proof = report_checks(item["report"], task["recipe"]["report"]["equals"])
                    emit(stage / "COMPLETE.json", dict(task=task["id"], plan_sha256=sha, report=proof,
                        state="BACKEND_SCOPE_CHECKS_COMPLETE_NOT_GENERAL_ADMISSION", exit_code=code))
                    print("COMPLETE", task["id"], flush=True)
                except Exception as error:
                    value = dict(task=task["id"], exit_code=code, error=str(error), automatic_retry=False)
                    emit(stage / "FAILED.json", value)
                    failed.append(value)
                    print("FAILED", task["id"], str(error), flush=True)
                slots.append(item["slot"])
                active.remove(item)
            if active:
                time.sleep(.5)
    except BaseException:
        terminate_owned(active)
        raise
    print(json.dumps(dict(failed=failed, verified_completed_skipped=skipped, automatic_retry=False)), flush=True)
    return 1 if failed else 0


def run_command(args):
    if not args.execute:
        return _run_command(args)
    root = literal(Path(args.plan).parent, output=True)
    fd = os.open(root / "RUN.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another invocation owns this queue; do not duplicate live work") from error
        return _run_command(args)
    finally:
        os.close(fd)


def interrupted(signum, frame):
    raise KeyboardInterrupt("operator queue interrupted by signal " + str(signum))


def main(argv=None):
    for number in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Metadata-only inventory and manual launcher; no experiments")
    plan.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "input/operator_production.yaml")
    plan.add_argument("--run-dir", type=Path, required=True)
    run = commands.add_parser("run", help="Dry run unless --execute; one explicitly selected stage")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--plan-sha256", required=True)
    run.add_argument("--stage", choices=STAGES, required=True)
    run.add_argument("--case", action="append")
    run.add_argument("--jobs", type=int, default=1)
    run.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            plan_command(args)
            return 0
        return run_command(args)
    except (ValueError, OSError, KeyError) as error:
        print("STOP:", str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
