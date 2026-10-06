#!/usr/bin/env python3
"""Check the actual heatmap p6 IBL packages on sysml, without refitting."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

import run_anchor_ibl_checks as common

BASE = common.BASE
YAP = common.YAP
OUTPUT = YAP / "D2W/output/case_studies/d2d_thickness_anchor_checks_20260918_v1"
SCRATCH = Path("/home/puneet/zhichao/YAP_STACK_MAPDL/d2d_thickness_anchor_checks_20260918_v1")


def window_root():
    return common.resource_dir("corrected_window")


def context(thickness):
    import numpy as np
    from initial_bow_distribution import D2D_INITIAL_BOW
    from exp12_design_window_20260914 import WindowStudy
    from run_exp12_ofat_composed_fast_20260913 import assembly_contract
    import exp12_layer_ibl_population_20260914 as population

    study = WindowStudy()
    key = f"h{thickness}"
    case = study.cases[key]
    z = np.vstack([np.zeros(4), -3*np.ones(4), 3*np.ones(4),
                   [-2, 2, -2, 2], [2, -2, 2, -2],
                   np.random.default_rng(20260918).standard_normal((3, 4))])
    study.source_amplitudes = D2D_INITIAL_BOW.sample(z)
    study.sample_count = len(z)
    roles = ["mean", "all_minus_3sigma", "all_plus_3sigma",
             "alternating_minus_plus_2sigma", "alternating_plus_minus_2sigma",
             "independent_holdout_0", "independent_holdout_1", "independent_holdout_2"]
    ibl_path = window_root() / f"packages/{key}/PACKAGE.json"
    operator_path = window_root() / f"histories/{key}/FACE_PACKAGE.json"
    ibl = json.loads(ibl_path.read_text())
    package = json.loads(operator_path.read_text())
    ledger = study.ledgers[key]
    if ledger != ibl["model"]["context"]["geometry"]:
        raise ValueError("heatmap IBL and direct solver must have identical tier geometry")
    if common.sha(operator_path) != ibl["model"]["context"]["package_sha256"]:
        raise ValueError("direct operator package differs from the IBL training teacher")
    if study.calibration.signature != ibl["source_calibration_sha256"]:
        raise ValueError("source calibration changed")
    predicted, outside = population.predict_rt(ibl["model"], study.source_amplitudes)
    paths = [common.source_study(), ibl_path, operator_path, Path(__file__), Path(common.__file__),
             YAP / "initial_bow_distribution.py", BASE / "exp12_design_window_20260914.py",
             BASE / "exp12_correct_tier_order_20260914.py"]
    plan = dict(schema="yap.thickness_anchor_checks.v1", case=case.record(),
        distribution=D2D_INITIAL_BOW.audit(), z=z.tolist(), roles=roles,
        amplitudes=study.source_amplitudes.tolist(), old_ibl_outside_box=outside.tolist(),
        inputs_sha256={str(p): common.sha(p) for p in paths},
        calibration_sha256=study.calibration.signature, assembly=assembly_contract(ledger, package),
        tier_order_contract=ledger["tier_order_contract"],
        refit=False, population_yield_estimate=False, mapdl_validation=False,
        scope="matched corrected-tier heatmap IBL versus its direct operators at new sources")
    for i in range(len(z)):
        birth = study.birth(key, i)
        np.testing.assert_array_equal(birth["source_amplitudes"], study.source_amplitudes[i])
        if not birth["sign_roundtrip_passed"]:
            raise ValueError("source sign roundtrip failed")
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

    directory = OUTPUT / f"p6_h{args.thickness}"
    started = time.perf_counter()

    def progress(state, **fields):
        row = dict(state=state, host=socket.gethostname(), pid=os.getpid(),
                   elapsed_job_s=time.perf_counter()-started, **fields)
        common.write(directory / "RUN_STATUS.json", row)
        print(json.dumps(row), flush=True)

    try:
        study, case, ledger, package, ibl, predicted, plan = context(args.thickness)
        if (directory / "PLAN.json").exists():
            raise ValueError("refusing to overwrite an attempted run")
        plan.update(threads=4, affinity=sorted(os.sched_getaffinity(0)))
        common.write(directory / "PLAN.json", plan)
        receipts = []
        with threadpool_limits(limits=4):
            progress("LOADING_LAYER_OPERATORS")
            model = load_model(ledger, package, temperature_c=case.anneal_c,
                               footprint_mm=case.footprint_mm, layer_workers=1)
            common.write(directory / "LOADED_OPERATORS.json", dict(load_seconds=model.load_seconds,
                inputs=model._inputs, implementation=model._implementation))
            for index, role in enumerate(plan["roles"]):
                for path, expected in plan["inputs_sha256"].items():
                    if common.sha(Path(path)) != expected:
                        raise ValueError(f"input changed: {path}")
                birth = study.birth(case.case_id, index)
                birth["sample_role"] = role
                progress("SOLVING_HISTORY", probe=index, role=role)
                begin = time.perf_counter()
                with deduplicated_reference_cache(model), compact_solver(
                        maximum_pairs=32, cache_entries=4, cache_bytes=2*1024**3):
                    result = model.predict(birth["solver_input"], progress_callback=lambda value:
                        progress("SOLVING_HISTORY", probe=index, detail=value))
                validate_history(result, case.cycles, case.anneal_c)
                elapsed = time.perf_counter()-begin
                result.update(anchor_probe=birth, ofat_case=case.record(), measured_history_seconds=elapsed)
                common.write(directory / f"probe_{index:02d}.json", result)
                target = pilot.mechanical_outputs(result, ledger)
                for field in ("labels", "units", "dependencies"):
                    if target[field] != ibl["model"][field]:
                        raise ValueError(f"teacher/IBL channel mismatch: {field}")
                errors = pilot.error_rows(case.case_id, index, predicted[index], target["values"],
                                          target["labels"], target["units"])
                response, audit = prepare_metric(result)
                teacher = population.mechanics(ibl["model"], np.asarray([target["values"]]), response)
                estimate = population.mechanics(ibl["model"], predicted[index:index+1], response)
                row = dict(probe=index, role=role, history_seconds=elapsed,
                    source_amplitudes=birth["source_amplitudes"],
                    failed_channel_count=sum(not e["pass_10percent_or_nearzero_absolute"] for e in errors),
                    max_tensor_absolute_error_ppm=float(np.max(np.abs(teacher["tensor"]-estimate["tensor"]))*1e6),
                    teacher_tensor_ppm=(teacher["tensor"][0]*1e6).tolist(),
                    ibl_tensor_ppm=(estimate["tensor"][0]*1e6).tolist(),
                    teacher_body_bow_um=teacher["body"][0].tolist(),
                    teacher_rt_bow_um=teacher["rt"][0].tolist(), metric_audit=audit)
                common.write(directory / f"comparison_{index:02d}.json", dict(row, channel_errors=errors))
                receipts.append(row)
                common.write(directory / "SUMMARY.json", dict(state="RUNNING", probes=receipts))
                progress("PROBE_COMPLETE", probe=index, history_seconds=elapsed,
                         failed_channel_count=row["failed_channel_count"])
            common.write(directory / "SUMMARY.json", dict(state="COMPLETE", probes=receipts,
                qualification="diagnostic_only", population_yield_estimate=False))
            progress("COMPLETE")
    except Exception:
        progress("FAILED", traceback=traceback.format_exc())
        raise


def launch(args):
    if socket.gethostname().split(".")[0] != "sysml":
        raise ValueError("user requested sysml")
    cpus = list(range(args.cpu_start, args.cpu_start+4))
    if not set(cpus).issubset(os.sched_getaffinity(0)):
        raise ValueError("CPUs unavailable")
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    if int(mem["MemAvailable"].split()[0]) < 48*1024**2:
        raise RuntimeError("require at least 48 GiB available memory")
    scratch = SCRATCH / f"p6_h{args.thickness}"
    scratch.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(scratch).free < 30*1024**3:
        raise RuntimeError("require at least 30 GiB scratch")
    directory = OUTPUT / f"p6_h{args.thickness}"
    directory.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch),
        MPLCONFIGDIR=str(scratch / "mpl"), NUMBA_CACHE_DIR=str(scratch / "numba"),
        OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4",
        MKL_DYNAMIC="FALSE", OMP_WAIT_POLICY="PASSIVE", KMP_BLOCKTIME="0", NUMBA_THREADING_LAYER="omp",
        PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    command = ["nice", "-n", "10", "taskset", "-c", ",".join(map(str, cpus)), sys.executable,
               "-B", str(Path(__file__).resolve()), "run", "--thickness", str(args.thickness)]
    with (directory / "RUN.log").open("x") as log:
        process = subprocess.Popen(command, cwd=YAP, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    common.write(directory / "LAUNCH.json", dict(pid=process.pid, command=command,
        host=socket.gethostname(), launched_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), scratch=str(scratch)))
    time.sleep(2)
    print(json.dumps(dict(pid=process.pid, exit_code=process.poll(), output=str(directory))))
    if process.poll() is not None:
        raise RuntimeError("worker exited; see RUN.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "launch", "run"))
    parser.add_argument("--thickness", type=int, choices=(20, 30), required=True)
    parser.add_argument("--cpu-start", type=int, default=16)
    args = parser.parse_args()
    if args.action == "preflight":
        *_, plan = context(args.thickness)
        print(json.dumps(plan, indent=2))
    elif args.action == "launch":
        launch(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
