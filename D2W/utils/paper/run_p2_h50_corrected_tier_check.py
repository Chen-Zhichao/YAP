#!/usr/bin/env python3
"""Direct p2/h50 check on the corrected heatmap tier ledger."""
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

import run_p2_thickness_pipeline as pipeline


def plan():
    study, case, ledger, _, _, paths = pipeline.source_context(50)
    paths += [pipeline.COMPACT, Path(__file__), Path(pipeline.__file__),
              Path(pipeline.reference.__file__), Path(pipeline.reference.common.__file__),
              pipeline.YAP / "initial_bow_distribution.py"]
    return dict(schema="yap.p2_h50_corrected_tier_check.v1", case=case.record(),
        source_amplitudes=study.source_amplitudes.tolist(),
        calibration_sha256=study.calibration.signature,
        tier_order_contract=ledger["tier_order_contract"],
        input_sha256={str(path): pipeline.sha(path) for path in paths},
        gamma=0.8, placement_samples=10000, refit=False,
        source_population_yield_estimate=False,
        operator_source=str(pipeline.COMPACT),
        scope="direct corrected-tier p2/h50 response versus heatmap baseline transfer")


def run():
    from threadpoolctl import threadpool_limits
    directory = pipeline.ROOT / "h50"
    started = time.perf_counter()

    def report(state, **fields):
        row = dict(state=state, thickness=50, host=socket.gethostname(), pid=os.getpid(),
                   elapsed_job_s=time.perf_counter()-started, **fields)
        pipeline.write(directory / "RUN_STATUS.json", row)
        print(json.dumps(row), flush=True)

    try:
        record = plan()
        pipeline.write(directory / "PLAN.json", record)
        package = json.loads(pipeline.COMPACT.read_text())
        with threadpool_limits(limits=4):
            pipeline.histories(50, record, package, report)
        report("COMPLETE")
    except Exception:
        report("FAILED", traceback=traceback.format_exc())
        raise


def launch(args):
    if socket.gethostname().split(".")[0] != "sysml":
        raise ValueError("user selected sysml")
    plan()
    directory = pipeline.ROOT / "h50"
    directory.mkdir(parents=True, exist_ok=False)
    cpus = list(range(args.cpu_start, args.cpu_start+4))
    if not set(cpus).issubset(os.sched_getaffinity(0)):
        raise ValueError("requested CPUs unavailable")
    scratch = pipeline.SCRATCH / "runtime/h50"
    scratch.mkdir(parents=True, exist_ok=True)
    check = pipeline.resources()
    if check["available_memory_gib"] < 64 or check["free_scratch_gib"] < 30:
        raise RuntimeError(f"insufficient resources: {check}")
    env = dict(os.environ, TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch),
        MPLCONFIGDIR=str(scratch/"mpl"), NUMBA_CACHE_DIR=str(scratch/"numba"),
        OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4",
        MKL_DYNAMIC="FALSE", OMP_WAIT_POLICY="PASSIVE", KMP_BLOCKTIME="0", NUMBA_THREADING_LAYER="omp",
        PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    command = ["nice", "-n", "10", "taskset", "-c", ",".join(map(str, cpus)),
               sys.executable, "-B", str(Path(__file__).resolve()), "run"]
    with (directory/"RUN.log").open("x") as log:
        process = subprocess.Popen(command, cwd=pipeline.YAP, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pipeline.write(directory/"LAUNCH.json", dict(pid=process.pid, command=command,
        host=socket.gethostname(), cpu_affinity=cpus, preflight=check,
        launched_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    time.sleep(2)
    print(json.dumps(dict(pid=process.pid, exit_code=process.poll(), output=str(directory))))
    if process.poll() is not None:
        raise RuntimeError("h50 worker exited during launch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "launch", "run"))
    parser.add_argument("--cpu-start", type=int, default=49)
    args = parser.parse_args()
    if args.action == "preflight":
        print(json.dumps(plan(), indent=2))
    elif args.action == "launch":
        launch(args)
    else:
        run()


if __name__ == "__main__":
    main()
