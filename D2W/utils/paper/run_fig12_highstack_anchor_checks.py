#!/usr/bin/env python3
"""Independent high-stack source probes for the Fig.12 response models."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

import repair_original_tier_heatmaps as repair

ROOT = repair.YAP / "D2W/output/case_studies/fig12_f2f_f2b_anchor_checks_20260918_v1"
JOBS = {"p6_8hi_mean": (6,8,0.,80), "p6_8hi_low3": (6,8,-3.,84),
        "p6_16hi_mean": (6,16,0.,88), "p6_16hi_low3": (6,16,-3.,92),
        "p2_16hi_mean": (2,16,0.,96)}


def context(key):
    import numpy as np
    import exp12_ofat_case_study_20260913 as source
    from exp12_ofat_frozen_study_20260913 import load_frozen_study
    from initial_bow_distribution import D2D_INITIAL_BOW

    pitch,n,z,_=JOBS[key]
    study=load_frozen_study(repair.common.source_study())
    case=study.cases[f"dies{n}"]
    if pitch == 2:
        case=replace(case,bi_pitch_um=2.,case_id=key,label=key,changed_parameter="pitch_die_count")
    ledger=source.case_ledger(case)
    material=source.FrozenV2Materials()(case)
    response=[source.clpt_source_response(source._manufacturing_layers(ledger,i,material)) for i in range(n)]
    study.cases={case.case_id:case}
    study._ledgers_json=source._json({case.case_id:ledger})
    study._materials_json=source._json({case.case_id:material})
    study._responses_json=source._json({case.case_id:response})
    study.source_amplitudes=D2D_INITIAL_BOW.sample(np.full((1,n),z))
    study.sample_count=1
    birth=study.birth(case.case_id,0)
    if not birth["sign_roundtrip_passed"]:
        raise ValueError("source sign roundtrip failed")
    path=repair.operator_path(pitch,50)
    package=repair.read(path)
    reference=repair.read(repair.ORIGINAL/f"dies{n}/PACKAGE.json")
    if pitch == 6:
        if reference["model"]["context"]["geometry"] != ledger:
            raise ValueError("Fig.12 IBL has different tier geometry")
        if reference["model"]["context"]["package_sha256"] != repair.common.sha(path):
            raise ValueError("Fig.12 IBL teacher differs")
    return study,case,ledger,birth,path,package,reference


def _extend_bi2_tensor(bi6_tensor, bi6_reference, bi2_reference):
    import numpy as np

    result = np.asarray(bi6_tensor, dtype=np.float64).copy()
    reference_delta = np.asarray(bi2_reference) - np.asarray(bi6_reference)
    direct_events = min(result.shape[1], reference_delta.shape[1])
    result[:, :direct_events] += reference_delta[:, :direct_events]
    if result.shape[1] > direct_events:
        numerator = np.linalg.norm(bi6_tensor[:, direct_events:], axis=(-2, -1))
        denominator = np.linalg.norm(
            bi6_reference[:, direct_events - 1], axis=(-2, -1)
        )[:, None]
        scale = np.divide(numerator, denominator,
                          out=np.ones_like(numerator), where=denominator > 1e-15)
        result[:, direct_events:] += (
            scale[..., None, None] * reference_delta[:, direct_events - 1, None]
        )
    return result


def run(key):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from exp12_ofat_runtime_20260913 import load_model,validate_history
    from exp12_ofat_reference_dedup_20260913 import deduplicated_reference_cache
    from exp12_ofat_compact_solver_20260913 import compact_solver
    from run_exp12_ofat_fast_population_20260913 import prepare_metric
    import run_exp12_ofat_rt_distillation_pilot_20260913 as pilot
    import exp12_layer_ibl_population_20260914 as pop

    def _mechanics(package, amplitudes):
        values, _ = pop.predict_rt(package["model"], amplitudes)
        result = pop.mechanics(package["model"], values,
                               np.asarray(package["metric_response"]))
        return np.asarray(result["tensor"])

    from initial_bow_distribution import D2D_INITIAL_BOW

    directory=ROOT/key
    started=time.monotonic()
    def report(state,**kw):
        row=dict(state=state,host=socket.gethostname(),pid=os.getpid(),elapsed_s=time.monotonic()-started,**kw)
        repair.write(directory/"STATUS.json",row)
        print(json.dumps(row),flush=True)
    try:
        study,case,ledger,birth,path,package,reference=context(key)
        assets={str(p):repair.common.sha(p) for p in (Path(__file__),path,repair.common.source_study(),
                repair.YAP/"initial_bow_distribution.py",
                repair.ORIGINAL/f"dies{case.die_count}/PACKAGE.json")}
        for k in ledger["required_operator_thickness_mm"]:
            p=Path(package["datasets"][k])
            assets[str(p)]=package["dataset_sha256"][k]
            if repair.common.sha(p) != assets[str(p)]:
                raise ValueError(f"dataset changed: {p}")
        repair.write(directory/"PLAN.json",dict(contract=repair.CONTRACT,case=case.record(),geometry=ledger,
            distribution=D2D_INITIAL_BOW.audit(),birth=birth,assets=assets,
            role="diagnostic_only_not_a_population_yield_estimate",new_mapdl_calls=0))
        with threadpool_limits(limits=4):
            report("LOADING_OPERATORS")
            model=load_model(ledger,package,temperature_c=case.anneal_c,footprint_mm=case.footprint_mm,layer_workers=1)
            repair.write(directory/"LOADED_OPERATORS.json",dict(load_seconds=model.load_seconds,
                inputs=model._inputs,implementation=model._implementation))
            report("SOLVING_HISTORY",cycles=case.cycles)
            with deduplicated_reference_cache(model),compact_solver(maximum_pairs=32,cache_entries=4,cache_bytes=2*1024**3):
                result=model.predict(birth["solver_input"],progress_callback=lambda detail:report("SOLVING_HISTORY",detail=detail))
            validate_history(result,case.cycles,case.anneal_c)
            result.update(fig12_probe=birth,fig12_case=case.record(),measured_seconds=time.monotonic()-started)
            repair.write(directory/"HISTORY.json",result)
            target=pilot.mechanical_outputs(result,ledger)
            response,audit=prepare_metric(result)
            if any(target[k] != reference["model"][k] for k in ("labels","units","dependencies")):
                raise ValueError("response channel mismatch")
            truth=pop.mechanics(reference["model"],np.asarray([target["values"]]),np.asarray(response))["tensor"]
            a=np.asarray([birth["source_amplitudes"]])
            estimated=_mechanics(reference,a)
            channel_errors=None
            if case.bi_pitch_um == 6:
                values,_=pop.predict_rt(reference["model"],a)
                channel_errors=pilot.error_rows(key,0,values[0],target["values"],target["labels"],target["units"])
            else:
                p6=repair.read(repair.ORIGINAL/"baseline/PACKAGE.json")
                parent=repair.read(repair.ORIGINAL/"bi2/PACKAGE.json")
                e6=_mechanics(p6,a[:,:4])
                e2=_mechanics(parent,a[:,:4])
                estimated=_extend_bi2_tensor(estimated,e6,e2)
            error=float(np.max(np.abs(truth-estimated))*1e6)
            summary=dict(contract=repair.CONTRACT,pitch=case.bi_pitch_um,die_count=case.die_count,
                role="mechanical_tensor_probe",source_amplitudes=birth["source_amplitudes"],
                direct_tensor_ppm=(truth[0]*1e6).tolist(),figure_tensor_ppm=(estimated[0]*1e6).tolist(),
                maximum_tensor_error_ppm=error,metric_audit=audit,channel_errors=channel_errors,
                elapsed_s=time.monotonic()-started,physical_validation=False,
                comparison="IBL versus same reduced operator" if case.bi_pitch_um==6 else "16-Hi pitch transfer versus direct P24 history")
            repair.write(directory/"SUMMARY.json",summary)
            report("COMPLETE",maximum_tensor_error_ppm=error)
    except Exception:
        report("FAILED",traceback=traceback.format_exc())
        raise


def launch():
    if socket.gethostname().split(".")[0] != "mfd":
        raise ValueError("user selected mfd")
    mem=dict(line.split(":",1) for line in Path("/proc/meminfo").read_text().splitlines())
    if int(mem["MemAvailable"].split()[0]) < 160*1024**2:
        raise RuntimeError("require 160 GiB available before launching five high-stack probes")
    for key in JOBS:
        context(key)
    ROOT.mkdir(parents=True,exist_ok=False)
    receipts=[]
    for key,(_,_,_,start) in JOBS.items():
        directory=ROOT/key
        directory.mkdir()
        scratch=Path("/home/scratch/zhichao/YAP_STACK_MAPDL/fig12_f2f_f2b_anchor_checks_20260918_v1")/key
        scratch.mkdir(parents=True,exist_ok=True)
        cpus=list(range(start,start+4))
        if not set(cpus).issubset(os.sched_getaffinity(0)):
            raise ValueError("assigned CPUs unavailable")
        env=dict(os.environ,TMPDIR=str(scratch),TMP=str(scratch),TEMP=str(scratch),
            MPLCONFIGDIR=str(scratch/"mpl"),NUMBA_CACHE_DIR=str(scratch/"numba"),
            OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",OMP_NUM_THREADS="4",NUMBA_NUM_THREADS="4",
            MKL_DYNAMIC="FALSE",OMP_WAIT_POLICY="PASSIVE",KMP_BLOCKTIME="0",NUMBA_THREADING_LAYER="omp",
            PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="")
        command=["nice","-n","10","taskset","-c",",".join(map(str,cpus)),sys.executable,
                 "-B",str(Path(__file__).resolve()),"run","--job",key]
        with (directory/"RUN.log").open("x") as log:
            p=subprocess.Popen(command,cwd=repair.YAP,env=env,stdin=subprocess.DEVNULL,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        row=dict(job=key,host=socket.gethostname(),pid=p.pid,command=command,cpus=cpus,time=time.time())
        repair.write(directory/"LAUNCH.json",row)
        receipts.append(row)
    repair.write(ROOT/"LAUNCH.json",receipts)
    print(json.dumps(receipts,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("preflight","launch","run"))
    parser.add_argument("--job",choices=JOBS)
    args=parser.parse_args()
    if args.action == "preflight":
        for key in JOBS:
            _,case,_,_,_,_,_=context(key)
            print(key,case.record())
    elif args.action == "launch":
        launch()
    else:
        run(args.job)


if __name__ == "__main__":
    main()
