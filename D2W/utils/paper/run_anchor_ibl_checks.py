#!/usr/bin/env python3
"""Bounded direct-physics checks of existing IBL under the frozen D2D source.

These deliberately selected probes are not a population yield estimate or a
new training set. No old data, coefficients, or qualification flags are edited.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

YAP = Path(__file__).resolve().parents[3]
BASE = YAP.parent / "Ansys_Verify/12_stack_model_validation"
OUTPUT = YAP / "D2W/output/case_studies/d2d_anchor_preserving_ibl_check_20260918_v1"
PACKAGES = {
    "baseline": "bi6_p32_b16/packages/BI6_50.json",
    "bi2": "bi2_p24_b16/packages/BI2_50.json",
}
sys.path[:0] = [str(YAP), str(YAP.parent), str(BASE)]
from mechanics_runtime import resource_directory


def resource_dir(name):
    return resource_directory(name, native_root=BASE)


def source_study():
    value = os.environ.get("YAP_SOURCE_STUDY")
    if not value:
        raise ValueError("set YAP_SOURCE_STUDY to the private frozen source-study file")
    path = Path(value)
    if {"#snapshot", ".snapshot"}.intersection(path.parts + path.resolve().parts):
        raise ValueError("snapshot paths are not allowed")
    return path.resolve(strict=True)


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(case_id):
    import numpy as np
    from initial_bow_distribution import D2D_INITIAL_BOW
    from exp12_ofat_frozen_study_20260913 import load_frozen_study
    from run_exp12_ofat_composed_fast_20260913 import assembly_contract
    import exp12_layer_ibl_population_20260914 as population

    frozen = load_frozen_study(source_study())
    study = copy.copy(frozen)
    case = study.cases[case_id]
    if case.die_count != 4:
        raise ValueError("pilot requires the frozen 4-Hi case")
    z = np.vstack([np.zeros(4), -3*np.ones(4), 3*np.ones(4),
                   [-2, 2, -2, 2], [2, -2, 2, -2],
                   np.random.default_rng(20260918).standard_normal((3, 4))])
    # Only the private working copy uses the explicit probe bank. The original
    # frozen source file and its calibration/geometry are left unchanged.
    study.source_amplitudes = D2D_INITIAL_BOW.sample(z)
    study.sample_count = len(z)
    roles = ["mean", "all_minus_3sigma", "all_plus_3sigma",
             "alternating_minus_plus_2sigma", "alternating_plus_minus_2sigma",
             "independent_holdout_0", "independent_holdout_1", "independent_holdout_2"]
    package_path = resource_dir("compact_p2" if case_id == "bi2" else "compact_p6") / "packages" / Path(PACKAGES[case_id]).name
    ibl_path = resource_dir("layer_ibl") / case_id / "PACKAGE.json"
    package = json.loads(package_path.read_text())
    ibl = json.loads(ibl_path.read_text())
    ledger = study.ledgers[case_id]
    if ibl["model"]["context"]["geometry"] != ledger:
        raise ValueError("IBL and teacher geometry differ")
    predicted, outside = population.predict_rt(ibl["model"], study.source_amplitudes)
    paths = [source_study(), package_path, ibl_path, Path(__file__), YAP / "initial_bow_distribution.py"]
    plan = dict(schema="yap.anchor_ibl_checks.v1", case=asdict(case),
        distribution=D2D_INITIAL_BOW.audit(), z=z.tolist(), roles=roles,
        amplitudes=study.source_amplitudes.tolist(),
        inputs_sha256={str(p): sha(p) for p in paths},
        assembly=assembly_contract(ledger, package),
        old_ibl_outside_box=outside.tolist(), calibration_sha256=study.calibration.signature,
        population_yield_estimate=False, refit=False, mapdl_validation=False,
        scope="diagnostic checks against existing reduced layer operators, not final qualification")
    return study, case, ledger, package, ibl, predicted, plan


def run(args):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from exp12_ofat_runtime_20260913 import load_model, validate_history
    from exp12_ofat_reference_dedup_20260913 import deduplicated_reference_cache
    from exp12_ofat_compact_solver_20260913 import compact_solver
    import run_exp12_ofat_rt_distillation_pilot_20260913 as pilot
    import exp12_layer_ibl_population_20260914 as population
    from run_exp12_ofat_fast_population_20260913 import prepare_metric

    directory = OUTPUT / args.case
    started = time.perf_counter()
    receipts = []

    def progress(state, **fields):
        row = dict(state=state, host=socket.gethostname(), pid=os.getpid(),
                   elapsed_job_s=time.perf_counter()-started, **fields)
        write(directory / "RUN_STATUS.json", row)
        print(json.dumps(row), flush=True)

    try:
        study, case, ledger, package, ibl, predicted, plan = inputs(args.case)
        if (directory / "PLAN.json").exists():
            raise ValueError("refusing to overwrite a previous attempt")
        plan.update(threads=args.threads, affinity=sorted(os.sched_getaffinity(0)))
        write(directory / "PLAN.json", plan)
        with threadpool_limits(limits=args.threads):
            progress("LOADING_LAYER_OPERATORS")
            model = load_model(ledger, package, temperature_c=case.anneal_c,
                               footprint_mm=case.footprint_mm, layer_workers=1)
            write(directory / "LOADED_OPERATORS.json", dict(load_seconds=model.load_seconds,
                  inputs=model._inputs, implementation=model._implementation))
            for index, role in enumerate(plan["roles"]):
                for path, expected in plan["inputs_sha256"].items():
                    if sha(Path(path)) != expected:
                        raise ValueError(f"input changed while running: {path}")
                birth = study.birth(args.case, index)
                birth["sample_role"] = role
                begin = time.perf_counter()
                progress("SOLVING_HISTORY", probe=index, role=role)
                with deduplicated_reference_cache(model), compact_solver(
                        maximum_pairs=32, cache_entries=4, cache_bytes=2*1024**3):
                    result = model.predict(birth["solver_input"], progress_callback=lambda value:
                        progress("SOLVING_HISTORY", probe=index, detail=value))
                validate_history(result, case.cycles, case.anneal_c)
                elapsed = time.perf_counter()-begin
                result.update(anchor_probe=birth, ofat_case=asdict(case), measured_history_seconds=elapsed)
                write(directory / f"probe_{index:02d}.json", result)
                target = pilot.mechanical_outputs(result, ledger)
                for field in ("labels", "units", "dependencies"):
                    if target[field] != ibl["model"][field]:
                        raise ValueError(f"teacher/IBL channel mismatch: {field}")
                errors = pilot.error_rows(args.case, index, predicted[index],
                    target["values"], target["labels"], target["units"])
                response, audit = prepare_metric(result)
                teacher = population.mechanics(ibl["model"], np.asarray([target["values"]]), response)
                estimate = population.mechanics(ibl["model"], predicted[index:index+1], response)
                comparison = dict(probe=index, role=role, history_seconds=elapsed,
                    amplitudes=birth["source_amplitudes"], channel_errors=errors,
                    failed_channel_count=sum(not r["pass_10percent_or_nearzero_absolute"] for r in errors),
                    teacher_tensor_ppm=(teacher["tensor"][0]*1e6).tolist(),
                    ibl_tensor_ppm=(estimate["tensor"][0]*1e6).tolist(),
                    max_tensor_absolute_error_ppm=float(np.max(np.abs(teacher["tensor"]-estimate["tensor"]))*1e6),
                    metric_audit=audit, qualification="diagnostic_only")
                write(directory / f"comparison_{index:02d}.json", comparison)
                receipts.append({k: v for k, v in comparison.items() if k != "channel_errors"})
                write(directory / "SUMMARY.json", dict(state="RUNNING", probes=receipts))
                progress("PROBE_COMPLETE", probe=index, history_seconds=elapsed,
                         failed_channel_count=comparison["failed_channel_count"])
            write(directory / "SUMMARY.json", dict(state="COMPLETE", probes=receipts,
                  population_yield_estimate=False, qualification="diagnostic_only"))
            progress("COMPLETE")
    except Exception:
        progress("FAILED", traceback=traceback.format_exc())
        raise


def launch(args):
    if socket.gethostname().split(".")[0] != "mfd":
        raise ValueError("launch is currently limited to mfd; sysml is overloaded")
    cpus = list(range(args.cpu_start, args.cpu_start + args.threads))
    if not set(cpus).issubset(os.sched_getaffinity(0)):
        raise ValueError("requested CPUs unavailable")
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    if int(mem["MemAvailable"].split()[0]) < 32*1024**2:
        raise RuntimeError("insufficient available memory; require 32 GiB")
    scratch = Path("/home/scratch/zhichao/YAP_STACK_MAPDL/anchor_ibl_checks_20260918") / args.case
    scratch.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(scratch).free < 30*1024**3:
        raise RuntimeError("insufficient scratch; require 30 GiB")
    directory = OUTPUT / args.case
    directory.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch),
               MPLCONFIGDIR=str(scratch / "mpl"), NUMBA_CACHE_DIR=str(scratch / "numba"),
               MKL_DYNAMIC="FALSE", OMP_WAIT_POLICY="PASSIVE", KMP_BLOCKTIME="0",
               NUMBA_THREADING_LAYER="omp", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[name] = str(args.threads)
    command = ["nice", "-n", "10", "taskset", "-c", ",".join(map(str, cpus)),
               sys.executable, "-B", str(Path(__file__).resolve()), "run", "--case", args.case,
               "--threads", str(args.threads)]
    with (directory / "RUN.log").open("x") as log:
        process = subprocess.Popen(command, cwd=YAP, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(directory / "LAUNCH.json", dict(pid=process.pid, command=command,
          host=socket.gethostname(), launched_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), scratch=str(scratch)))
    time.sleep(2)
    print(json.dumps(dict(pid=process.pid, exit_code=process.poll(), output=str(directory))))
    if process.poll() is not None:
        raise RuntimeError("worker exited during launch; see RUN.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "launch", "run"))
    parser.add_argument("--case", choices=PACKAGES, required=True)
    parser.add_argument("--threads", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--cpu-start", type=int, default=32)
    args = parser.parse_args()
    if args.action == "preflight":
        *_, plan = inputs(args.case)
        print(json.dumps(plan, indent=2))
    elif args.action == "launch":
        launch(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
