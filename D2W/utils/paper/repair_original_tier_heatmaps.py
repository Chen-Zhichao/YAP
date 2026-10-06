#!/usr/bin/env python3
"""Restore the original p2 tier contract using retained layer operators."""
from __future__ import annotations

import argparse
import copy
from dataclasses import fields, replace
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

import run_anchor_ibl_checks as common

BASE, YAP = common.BASE, common.YAP
RESULTS = BASE / "results"
ROOT = YAP / "D2W/output/case_studies/original_tier_heatmap_repair_20260918_v1"
OLD = YAP / "D2W/output/case_studies/p2_thickness_transfer_20260918_v1"
HEIGHTS = (20, 30, 40, 60, 70, 80)
CONTRACT = "f2f_base_f2b_stack"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    common.write(path, value)


def read(path):
    return json.loads(path.read_text())


def old_ibl(pitch, height):
    if pitch == 2:
        return common.resource_dir("layer_ibl") / "bi2/PACKAGE.json" if height == 50 else None
    if height in (20, 50, 80):
        return common.resource_dir("layer_ibl") / {20: "total20", 50: "baseline", 80: "total80"}[height] / "PACKAGE.json"
    return common.resource_dir("design_window") / f"packages/h{height}/PACKAGE.json"


def operator_path(pitch, height):
    if pitch == 2 and height != 50:
        return OLD / f"h{height}/OPERATOR_PACKAGE.json"
    if pitch == 6 and height not in (20, 50, 80):
        return common.resource_dir("design_window") / f"histories/h{height}/FACE_PACKAGE.json"
    return common.resource_dir("compact_p2" if pitch == 2 else "compact_p6") / f"packages/BI{pitch}_{height}.json"


def context(pitch, height, z):
    import numpy as np
    import exp12_ofat_case_study_20260913 as source
    from exp12_ofat_frozen_study_20260913 import load_frozen_study
    from initial_bow_distribution import D2D_INITIAL_BOW

    study = load_frozen_study(common.source_study())
    existing = old_ibl(pitch, height)
    package = read(existing) if existing else None
    if package:
        record = package["model"]["context"]["case"]
        case = source.OFATCase(**{f.name: record[f.name] for f in fields(source.OFATCase)})
    else:
        case = replace(study.cases["bi2"], case_id=f"p2_h{height}", total_die_um=float(height),
                       changed_parameter="pitch_thickness_window", label=f"Original tier p2 h{height}")
    ledger = source.case_ledger(case)
    if package and ledger != package["model"]["context"]["geometry"]:
        raise ValueError("retained IBL geometry differs from the original p2 contract")
    material = source.FrozenV2Materials()(case)
    responses = [source.clpt_source_response(source._manufacturing_layers(ledger, i, material))
                 for i in range(4)]
    study.cases = {case.case_id: case}
    study._ledgers_json = source._json({case.case_id: ledger})
    study._materials_json = source._json({case.case_id: material})
    study._responses_json = source._json({case.case_id: responses})
    study.source_amplitudes = D2D_INITIAL_BOW.sample(np.asarray(z, float))
    study.sample_count = len(z)
    return study, case, ledger, package


def probes():
    import numpy as np
    return np.vstack([np.zeros(4), -3*np.ones(4), 3*np.ones(4),
                      [-2, 2, -2, 2], [2, -2, 2, -2],
                      np.random.default_rng(20260918).standard_normal((3, 4))])


def preflight():
    from initial_bow_distribution import D2D_INITIAL_BOW
    report = []
    for pitch in (6, 2):
        for height in (20, 30, 40, 50, 60, 70, 80):
            study, case, ledger, package = context(pitch, height, probes())
            for i in range(8):
                if not study.birth(case.case_id, i)["sign_roundtrip_passed"]:
                    raise ValueError("source sign roundtrip failed")
            path = operator_path(pitch, height)
            if package and common.sha(path) != package["model"]["context"]["package_sha256"]:
                raise ValueError(f"teacher operator identity mismatch: {path}")
            report.append(dict(pitch=pitch, height=height, ibl=str(old_ibl(pitch, height)),
                               operator=str(path), operator_ready=path.exists(),
                               ledger_sha256=common.sha(old_ibl(pitch, height)) if package else None))
    return dict(contract=CONTRACT, distribution=D2D_INITIAL_BOW.audit(), cases=report)


def validate(model, records, response, template, pitch, height):
    import numpy as np
    import exp12_layer_ibl_population_20260914 as pop
    import run_exp12_ofat_rt_distillation_pilot_20260913 as pilot
    import run_exp12_bi2_bi6_d2d_thickness_heatmap_20260915 as heatmap
    import run_exp12_pitch_gamma_sweep_20260915 as scoring

    draws = scoring.placement_draws(4096)
    cfg, fixed, sites, particle, chip, threshold = heatmap.inputs(template, pitch, height, 10., 24)
    checks = []
    for record in records:
        values, _ = pop.predict_rt(model, np.asarray([record["amplitudes"]]))
        target = record["target"]
        if any(model[k] != target[k] for k in ("labels", "units", "dependencies")):
            raise ValueError("teacher and IBL channel schemas differ")
        errors = pilot.error_rows("original_tier", record["index"], values[0],
                                  target["values"], model["labels"], model["units"])
        actual = pop.mechanics(model, np.asarray([target["values"]]), np.asarray(record["response"]))["tensor"]
        estimate = pop.mechanics(model, values, np.asarray(response))["tensor"]
        tensor_error = float(np.max(np.abs(actual-estimate))*1e6)
        differences = []
        for gamma in (.75, .8, .85):
            yields = [scoring.score_gamma(np.repeat(t, 4096, axis=0), gamma,
                      cfg, fixed, sites, particle, chip, threshold, draws)["overlay_yield"]
                      for t in (actual, estimate)]
            differences.append(abs(yields[0]-yields[1]))
        failed = sum(not e["pass_10percent_or_nearzero_absolute"] for e in errors)
        checks.append(dict(index=record["index"], tensor_error_ppm=tensor_error,
                           failed_channels=failed, max_yield_absolute_error=max(differences),
                           passed=failed == 0 and tensor_error <= .1 and max(differences) <= .005))
    return dict(passed=all(c["passed"] for c in checks), checks=checks,
                scope="same-operator response checks; not independent MAPDL thermal validation")


def run_case(pitch, height):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from initial_bow_distribution import D2D_INITIAL_BOW
    from exp12_ofat_runtime_20260913 import load_model, validate_history
    from exp12_ofat_reference_dedup_20260913 import deduplicated_reference_cache
    from exp12_ofat_compact_solver_20260913 import compact_solver
    from run_exp12_ofat_fast_population_20260913 import prepare_metric
    import run_exp12_ofat_rt_distillation_pilot_20260913 as pilot
    import prepare_exp12_layer_ibl_20260914 as fitting

    directory = ROOT / f"p{pitch}_h{height}"
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def progress(state, **kw):
        row = dict(state=state, pitch=pitch, height=height, pid=os.getpid(),
                   elapsed_s=time.monotonic()-started, **kw)
        write(directory / "STATUS.json", row)
        print(json.dumps(row), flush=True)

    try:
        study, case, ledger, retained = context(pitch, height, probes())
        path = operator_path(pitch, height)
        package = read(path)
        context_record = dict(case=case.record(), geometry=ledger, physics="inelastic",
            package_sha256=common.sha(path), source_sha256=common.sha(common.source_study()))
        assets = {str(p): common.sha(p) for p in [Path(__file__), common.source_study(),
                  YAP / "initial_bow_distribution.py", path,
                  BASE / "exp12_ofat_case_study_20260913.py"]}
        assets[str(Path(package["port_basis"]))] = package["port_basis_sha256"]
        assets.update({str(Path(package["datasets"][k])): package["dataset_sha256"][k]
                       for k in ledger["required_operator_thickness_mm"]})
        for p, checksum in assets.items():
            if common.sha(Path(p)) != checksum:
                raise ValueError(f"input hash mismatch: {p}")
        plan = dict(contract=CONTRACT, geometry=ledger, case=case.record(),
                    distribution=D2D_INITIAL_BOW.audit(), assets=assets,
                    training_count=12, heldout_count=4, probes=probes().tolist(),
                    physical_validation=False, operator_candidate=package.get("state"))
        if (directory / "PLAN.json").exists() and read(directory / "PLAN.json") != plan:
            raise ValueError("cannot resume with changed inputs")
        write(directory / "PLAN.json", plan)
        with threadpool_limits(limits=4):
            progress("LOADING_OPERATORS")
            solver = load_model(ledger, package, temperature_c=case.anneal_c,
                                footprint_mm=case.footprint_mm, layer_workers=1)
            write(directory / "LOADED_OPERATORS.json", dict(load_seconds=solver.load_seconds,
                inputs=solver._inputs, implementation=solver._implementation))

            def solve(bank, index, label):
                nonlocal study
                study.source_amplitudes = D2D_INITIAL_BOW.sample(bank)
                study.sample_count = len(bank)
                birth = study.birth(case.case_id, index)
                result_path = directory / f"{label}_{index:02d}.json"
                if result_path.exists():
                    result = read(result_path)
                    if result.get("repair_birth") != birth or result.get("repair_context") != context_record:
                        raise ValueError("saved history has different source or geometry")
                else:
                    progress("SOLVING_HISTORY", bank=label, index=index)
                    begin = time.monotonic()
                    with deduplicated_reference_cache(solver), compact_solver(
                            maximum_pairs=32, cache_entries=4, cache_bytes=2*1024**3):
                        result = solver.predict(birth["solver_input"])
                    result.update(repair_birth=birth, repair_context=context_record,
                                  measured_history_seconds=time.monotonic()-begin)
                    validate_history(result, case.cycles, case.anneal_c)
                    write(result_path, result)
                validate_history(result, case.cycles, case.anneal_c)
                response, audit = prepare_metric(result)
                return dict(index=index, amplitudes=birth["source_amplitudes"],
                            target=pilot.mechanical_outputs(result, ledger),
                            response=np.asarray(response).tolist(), metric_audit=audit,
                            history_seconds=result["measured_history_seconds"])

            template = retained or read(ORIGINAL / "bi2/PACKAGE.json")
            if retained:
                records = [solve(probes(), i, "probe") for i in range(8)]
                progress("VALIDATING_RETAINED_IBL")
                check = validate(retained["model"], records, retained["metric_response"], template, pitch, height)
                write(directory / "REUSE_CHECK.json", check)
                if check["passed"]:
                    write(directory / "READY.json", dict(package=str(old_ibl(pitch, height)),
                          package_sha256=common.sha(old_ibl(pitch, height)), reused=True, contract=CONTRACT,
                          distribution=D2D_INITIAL_BOW.audit(), check=check))
                    progress("COMPLETE_REUSED")
                    return

            # Fit only on the first 12 draws; open the independent four checks after freezing coefficients.
            bank = np.random.default_rng(2026091801).standard_normal((16, 4))
            write(directory / "TRAINING_BANK.json", dict(z=bank.tolist(), training=list(range(12)),
                  holdout=list(range(12,16)), amplitudes=D2D_INITIAL_BOW.sample(bank).tolist()))
            training = [solve(bank, i, "draw") for i in range(12)]
            fit_input = [((np.asarray(r["amplitudes"])-1.)/.1, r["target"], context_record)
                         for r in training]
            model = fitting.fit(fit_input)
            write(directory / "MODEL.json", model)
            response = training[0]["response"]
            heldout = [solve(bank, i, "draw") for i in range(12,16)]
            progress("VALIDATING_NEW_IBL")
            check = validate(model, heldout, response, template, pitch, height)
            write(directory / "FIT_CHECK.json", check)
            if not check["passed"]:
                raise ValueError("new IBL failed independent checks; no publication")
            new = dict(schema=template["schema"], score=copy.deepcopy(template["score"]),
                       scoring_template=str(old_ibl(pitch, height) or ORIGINAL / "bi2/PACKAGE.json"),
                       model=model, metric_response=response, metric_audit=training[0]["metric_audit"],
                       source_calibration_sha256=study.calibration.signature,
                       source_sha256=common.sha(common.source_study()), distribution=D2D_INITIAL_BOW.audit(),
                       validation=dict(response_sampling_check_passed=True, repair_check=check,
                                       physical_validation=False), physical_readiness=False,
                       repair_contract=CONTRACT, scope="original-tier reduced-operator engineering heatmap")
            # Scoring uses current geometry through heatmap.inputs, not the retained template case.
            new["score"]["case"] = case.record()
            write(directory / "PACKAGE.json", new)
            write(directory / "READY.json", dict(package=str(directory / "PACKAGE.json"),
                  package_sha256=common.sha(directory / "PACKAGE.json"), reused=False,
                  contract=CONTRACT, distribution=D2D_INITIAL_BOW.audit(), check=check))
            progress("COMPLETE_FITTED")
    except Exception:
        progress("FAILED", traceback=traceback.format_exc())
        raise


def stop_obsolete_histories():
    receipts = []
    for h in HEIGHTS:
        directory = OLD / f"h{h}"
        ready = directory / "OPERATOR_PACKAGE.json"
        launch = OLD / "workers" / f"h{h}" / "LAUNCH.json"
        if not ready.exists() or not launch.exists():
            continue
        pid = read(launch)["pid"]
        command_path = Path(f"/proc/{pid}/cmdline")
        if not command_path.exists():
            continue
        command = command_path.read_bytes().split(b"\0")
        expected = str(YAP / "D2W/utils/paper/run_p2_thickness_pipeline.py").encode()
        if expected not in command or b"run" not in command or f"h{h}".encode() not in command:
            raise ValueError(f"refusing to signal unrelated pid {pid}")
        # Package publication occurs after both static datasets are closed and verified.
        package = read(ready)
        if len(package.get("prepared", {})) != 0:
            raise ValueError("unexpected operator manifest")
        for key, filename in package["datasets"].items():
            if not Path(filename).is_file():
                raise ValueError(f"missing completed dataset: {key}")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        receipt = dict(pid=pid, height=h, time=time.time(), package=str(ready),
                       action="stopped_old_history_parent_after_operator_publication")
        write(directory / "SUPERSEDED_BY_ORIGINAL_TIER.json", receipt)
        receipts.append(receipt)
    return receipts


def baseline_ready():
    from initial_bow_distribution import D2D_INITIAL_BOW
    for pitch, name in ((6, "baseline"), (2, "bi2")):
        check_root = YAP / f"D2W/output/case_studies/d2d_anchor_preserving_ibl_check_20260918_v1/{name}"
        plan, summary = read(check_root / "PLAN.json"), read(check_root / "SUMMARY.json")
        if summary["state"] != "COMPLETE" or len(summary["probes"]) != 8:
            raise ValueError("baseline check incomplete")
        if plan["distribution"] != D2D_INITIAL_BOW.audit():
            raise ValueError("baseline source distribution changed")
        p = old_ibl(pitch, 50)
        if plan["inputs_sha256"][str(p)] != common.sha(p):
            raise ValueError("baseline package changed since check")
        if any(r["failed_channel_count"] or r["max_tensor_absolute_error_ppm"] > .1 for r in summary["probes"]):
            raise ValueError("baseline check failed")
        write(ROOT / f"p{pitch}_h50/READY.json", dict(package=str(p), package_sha256=common.sha(p),
              reused=True, contract=CONTRACT, distribution=D2D_INITIAL_BOW.audit(),
              checks=str(check_root / "SUMMARY.json")))


def manager():
    import fcntl
    with (ROOT / ".manager.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        baseline_ready()
        started = time.monotonic()
        queue = [(6,h) for h in HEIGHTS] + [(2,h) for h in HEIGHTS]
        slots = [list(range(80+4*i, 84+4*i)) for i in range(6)]
        running, completed, failed, stopped = {}, [], [], []
        while queue or running:
            stopped.extend(stop_obsolete_histories())
            for key, (process, cpus) in list(running.items()):
                if process.poll() is not None:
                    (completed if process.returncode == 0 else failed).append(dict(case=key, code=process.returncode))
                    slots.append(cpus)
                    del running[key]
            memory = dict(line.split(":",1) for line in Path("/proc/meminfo").read_text().splitlines())
            available = int(memory["MemAvailable"].split()[0])/1024**2
            while queue and slots and available > 56:
                chosen = next((x for x in queue if operator_path(*x).exists()), None)
                if chosen is None:
                    break
                queue.remove(chosen)
                pitch,height = chosen
                key = f"p{pitch}_h{height}"
                directory = ROOT/key
                directory.mkdir(parents=True, exist_ok=True)
                if (directory / "READY.json").exists():
                    completed.append(dict(case=key, resumed=True))
                    continue
                cpus = slots.pop(0)
                command = ["nice","-n","10","taskset","-c",",".join(map(str,cpus)),
                           sys.executable,"-B",str(Path(__file__).resolve()),"worker",
                           "--pitch",str(pitch),"--height",str(height)]
                with (directory/"RUN.log").open("a") as log:
                    process = subprocess.Popen(command, cwd=YAP, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                running[key] = (process,cpus)
                write(directory/"LAUNCH.json",dict(pid=process.pid,command=command,cpus=cpus))
                available -= 6
            write(ROOT/"STATUS.json",dict(state="RUNNING",queued=queue,
                running={k:v[0].pid for k,v in running.items()},completed=completed,
                failed=failed,stopped_old_histories=stopped,elapsed_s=time.monotonic()-started))
            if time.monotonic()-started > 12*3600:
                raise RuntimeError("campaign exceeded 12-hour budget; inspect workers before resuming")
            time.sleep(5)
        if failed:
            write(ROOT/"STATUS.json",dict(state="CHECKS_FAILED",failed=failed,completed=completed))
            return
        command=[sys.executable,"-B",str(YAP/"D2W/utils/paper/plot_original_tier_heatmaps.py")]
        with (ROOT/"PLOT.log").open("a") as log:
            result=subprocess.run(command,cwd=YAP,stdout=log,stderr=subprocess.STDOUT)
        write(ROOT/"STATUS.json",dict(state="COMPLETE" if result.returncode == 0 else "PLOT_FAILED",
              completed=completed,plot_exit_code=result.returncode,elapsed_s=time.monotonic()-started,
              stopped_old_histories=stopped))


def launch():
    if socket.gethostname().split(".")[0] != "sysml":
        raise ValueError("launch on sysml")
    if not set(range(80,104)).issubset(os.sched_getaffinity(0)):
        raise ValueError("planned worker CPUs unavailable")
    plan=preflight()
    ROOT.mkdir(parents=True,exist_ok=False)
    write(ROOT/"PLAN.json",plan)
    scratch=Path("/home/puneet/zhichao/YAP_STACK_MAPDL/original_tier_heatmap_repair_20260918_v1")
    scratch.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,TMPDIR=str(scratch),TMP=str(scratch),TEMP=str(scratch),
             MPLCONFIGDIR=str(scratch/"mpl"),NUMBA_CACHE_DIR=str(scratch/"numba"),
             OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",OMP_NUM_THREADS="4",NUMBA_NUM_THREADS="4",
             MKL_DYNAMIC="FALSE",OMP_WAIT_POLICY="PASSIVE",KMP_BLOCKTIME="0",NUMBA_THREADING_LAYER="omp",
             PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="")
    command=[sys.executable,"-B",str(Path(__file__).resolve()),"manager"]
    with (ROOT/"MANAGER.log").open("x") as log:
        p=subprocess.Popen(command,cwd=YAP,env=env,stdin=subprocess.DEVNULL,
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/"LAUNCH.json",dict(pid=p.pid,command=command,time=time.time()))
    print(json.dumps(dict(pid=p.pid,output=str(ROOT))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("preflight","launch","manager","worker"))
    parser.add_argument("--pitch",type=int,choices=(2,6))
    parser.add_argument("--height",type=int,choices=HEIGHTS)
    args=parser.parse_args()
    if args.action == "preflight":
        print(json.dumps(preflight(),indent=2))
    elif args.action == "launch":
        launch()
    elif args.action == "manager":
        manager()
    else:
        run_case(args.pitch,args.height)


if __name__ == "__main__":
    main()
