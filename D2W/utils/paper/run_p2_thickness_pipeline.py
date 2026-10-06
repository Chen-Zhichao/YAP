#!/usr/bin/env python3
"""Prepare compatible p2 operators and check the heatmap thickness transfer.

Independent geometry-specific static lifts feed the retained layer-ROM
history engine. The current heatmap transfer is evaluated only as a target
for comparison; it never supplies direct-history outputs or fitted data.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

import run_thickness_anchor_checks as reference

BASE, YAP = reference.BASE, reference.YAP
ROOT = YAP / "D2W/output/case_studies/p2_thickness_transfer_20260918_v1"
SCRATCH = Path("/home/puneet/zhichao/YAP_STACK_MAPDL/p2_thickness_transfer_20260918_v1")
THICKNESSES = (20, 30, 40, 60, 70, 80)
write, sha = reference.common.write, reference.common.sha


def compact_package():
    return reference.common.resource_dir("compact_p2") / "packages/BI2_50.json"


def geometry_api():
    import exp12_ofat_geometry_20260913 as api
    # Process-local destinations isolate this campaign from old publishers.
    api.ROOT = ROOT / "operators"
    api.WORK_ROOTS = {"sysml": SCRATCH / "static"}
    cases = [api.GeometryCase(f"bi2_tsv_h{v:g}".replace(".", "p"), "bi2", float(v),
                 "sysml", nx=45, ny=20, rank=64, cores=4)
             for h in THICKNESSES for v in (h-13, h-11.5)]
    api.CASES = {case.key: case for case in cases}
    api.PACKAGE_CASES = {}
    return api


def source_context(h):
    import numpy as np
    from exp12_design_window_20260914 import WindowStudy
    import exp12_ofat_case_study_20260913 as source
    from exp12_correct_tier_order_20260914 import d2d_case_ledger
    from initial_bow_distribution import D2D_INITIAL_BOW
    from run_exp12_bi2_bi6_d2d_thickness_heatmap_20260915 import bi2_transfer
    from run_exp12_pitch_gamma_sweep_20260915 import load_package, mechanics

    study = WindowStudy()
    key = f"h{h}"
    case = replace(study.cases[key], bi_pitch_um=2.0)
    ledger = d2d_case_ledger(case)
    materials = source.FrozenV2Materials()(case)
    responses = [source.clpt_source_response(source._manufacturing_layers(ledger, i, materials))
                 for i in range(case.die_count)]
    study.cases = {key: case}
    study._ledgers_json = source._json({key: ledger})
    study._materials_json = source._json({key: materials})
    study._responses_json = source._json({key: responses})
    z = np.vstack([np.zeros(4), -3*np.ones(4), 3*np.ones(4),
                   [-2, 2, -2, 2], [2, -2, 2, -2],
                   np.random.default_rng(20260918).standard_normal((3, 4))])
    study.source_amplitudes = D2D_INITIAL_BOW.sample(z)
    study.sample_count = len(z)
    paths = [reference.window_root() / f"packages/h{h}/PACKAGE.json",
             reference.window_root() / "packages/h50/PACKAGE.json",
             reference.common.resource_dir("layer_ibl") / "bi2/PACKAGE.json"]
    packages = [load_package(path) for path in paths]
    tensors = [mechanics(p, study.source_amplitudes)[0] for p in packages]
    transfer = bi2_transfer(*tensors)
    if not np.isfinite(transfer).all() or transfer.shape != (8, 3, 2, 2):
        raise ValueError("invalid heatmap transfer output")
    for i in range(8):
        birth = study.birth(key, i)
        if not birth["sign_roundtrip_passed"]:
            raise ValueError("source sign check failed")
    return study, case, ledger, packages[0], transfer, paths


def preflight(h):
    from initial_bow_distribution import D2D_INITIAL_BOW
    api = geometry_api()
    study, case, ledger, _, _, paths = source_context(h)
    cases = [api.CASES[f"bi2_tsv_h{v:g}".replace(".", "p")] for v in (h-13, h-11.5)]
    for item in cases:
        api.validate_basis(item)
        api.static_steps(item)
    compact = json.loads(compact_package().read_text())
    if compact["rank"] != 24:
        raise ValueError("require the frozen p2 P24 package")
    for key in ("bi", "beol"):
        if not Path(compact["datasets"][key]).is_file():
            raise FileNotFoundError(compact["datasets"][key])
    paths += [compact_package(), Path(__file__), Path(reference.__file__),
              Path(reference.common.__file__), reference.common.source_study(),
              YAP / "initial_bow_distribution.py", Path(api.__file__),
              BASE / "run_exp12_design_window_histories_20260914.py",
              BASE / "run_exp12_bi2_bi6_d2d_thickness_heatmap_20260915.py",
              BASE / "run_exp12_pitch_gamma_sweep_20260915.py"]
    return dict(schema="yap.p2_thickness_pipeline.v1", case=case.record(),
        source_distribution=D2D_INITIAL_BOW.audit(), source_amplitudes=study.source_amplitudes.tolist(),
        calibration_sha256=study.calibration.signature, tier_order_contract=ledger["tier_order_contract"],
        exact_static_substrates_um=[c.substrate_um for c in cases],
        input_sha256={str(path): sha(path) for path in paths},
        stages=["exact_static_lifts", "layer_datasets", "P24_projection",
                "eight_direct_histories", "tensor_and_conditional_yield_comparison"],
        gamma=0.8, placement_samples=10000, refit=False, new_mapdl_thermal_validation=False,
        scope="diagnostic transfer check; transferred internal approximation spaces remain provisional")


def resources():
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return dict(available_memory_gib=int(memory["MemAvailable"].split()[0])/1024**2,
                free_scratch_gib=shutil.disk_usage(SCRATCH).free/1024**3)


def prepare_operator(api, case, report):
    from exp12_layer_rom_dataset import LayerROMDataset
    from exp12_port_basis_contract import validate_dataset_port_basis_contracts
    from run_exp12_design_window_histories_20260914 import validate_face_basis, face_projection
    import numpy as np

    directory = api.ROOT / case.key
    plan = api.prepare(case)
    for step in plan["steps"]:
        api.verify_inputs(plan)
        while resources()["available_memory_gib"] < 48:
            report("WAITING_MEMORY", operator=case.key, resources=resources())
            time.sleep(30)
        report("PREPARING_STATIC", operator=case.key, stage=step["name"])
        if Path(step["summary"]).exists():
            api.verify_static_stage(case, step)
            continue
        started = time.perf_counter()
        with (directory / f'{step["name"]}.log').open("x") as log:
            process = subprocess.Popen(step["command"], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
            report("PREPARING_STATIC", operator=case.key, stage=step["name"], child_pid=process.pid)
            code = process.wait()
        if code:
            raise RuntimeError(f"{case.key}/{step['name']} failed with exit {code}")
        api.verify_static_stage(case, step, seal=True)
        write(directory / f'{step["name"]}_TIMING.json', dict(elapsed_s=time.perf_counter()-started))
    report("BUILDING_LAYER_DATASET", operator=case.key)
    ready = api.build_dataset(case, environment=dict(os.environ))
    compact = json.loads(compact_package().read_text())
    with np.load(api.BASIS["bi2"], allow_pickle=False) as f:
        parent = {k: f[k] for k in f.files}
    with np.load(compact["port_basis"], allow_pickle=False) as f:
        child = {k: f[k] for k in f.files}
    p = validate_face_basis(parent, child)
    if p.shape != (64, 24):
        raise ValueError("require matching native64-to-P24 projection")
    dataset = LayerROMDataset.load(ready["dataset_file"])
    reduced, _ = face_projection(dataset, p, compact["port_basis"], api.BASIS_SHA["bi2"],
        provenance=dict(parent_dataset=ready["dataset_file"], parent_sha256=ready["dataset_sha256"],
                        physical_static_geometry=case.substrate_um, no_endpoint_fit=True))
    # The reusable projection math is rank-generic; its historical P32 labels
    # are corrected to describe the actual P24 trial space used here.
    meta = dict(reduced.metadata)
    meta["basis_method"] = "frozen_P24_face_only_all_parent_bubbles"
    meta["coordinate_reduction"] = dict(meta["coordinate_reduction"],
        method="T=diag(I6,P24,P24,Iallparentbubbles)")
    reduced = replace(reduced, metadata=meta,
        macro_coordinate_labels=tuple(label.replace("p32_", "p24_") for label in reduced.macro_coordinate_labels))
    audit = validate_dataset_port_basis_contracts({case.library_key: reduced}, Path(compact["port_basis"]))
    target = directory / "P24_LAYER.npz"
    if target.exists():
        raise FileExistsError(target)
    reduced.save(target)
    receipt = dict(path=str(target), sha256=sha(target), native_receipt=ready,
                   port_basis_sha256=compact["port_basis_sha256"], gauge_audit=audit,
                   parent_bubbles_retained=dataset.bubble_size)
    write(directory / "P24_READY.json", receipt)
    return receipt


def histories(h, plan, package, report):
    import numpy as np
    import exp12_layer_ibl_population_20260914 as population
    import run_exp12_ofat_rt_distillation_pilot_20260913 as pilot
    from exp12_ofat_runtime_20260913 import load_model, validate_history
    from exp12_ofat_reference_dedup_20260913 import deduplicated_reference_cache
    from exp12_ofat_compact_solver_20260913 import compact_solver
    from run_exp12_ofat_fast_population_20260913 import prepare_metric
    import run_exp12_bi2_bi6_d2d_thickness_heatmap_20260915 as heatmap
    import run_exp12_pitch_gamma_sweep_20260915 as scoring

    directory = ROOT / f"h{h}"
    study, case, ledger, p6_package, transfer, _ = source_context(h)
    report("LOADING_HISTORY_OPERATORS")
    model = load_model(ledger, package, temperature_c=case.anneal_c,
                       footprint_mm=case.footprint_mm, layer_workers=1)
    write(directory / "LOADED_OPERATORS.json", dict(load_seconds=model.load_seconds,
        inputs=model._inputs, implementation=model._implementation))
    # Reuse the exact heatmap scorer and common placement samples for paired
    # conditional yields. Probe yields are never averaged as population yield.
    cfg, fixed, sites, particle, chip, threshold = heatmap.inputs(p6_package, 2., float(h), 10., 12)
    draws = scoring.placement_draws(plan["placement_samples"])
    rows = []
    for index in range(8):
        for path, expected in plan["input_sha256"].items():
            if sha(Path(path)) != expected:
                raise ValueError(f"input changed: {path}")
        birth = study.birth(case.case_id, index)
        birth["sample_role"] = "diagnostic_probe_not_population_sample"
        report("SOLVING_HISTORY", probe=index)
        begin = time.perf_counter()
        with deduplicated_reference_cache(model), compact_solver(
                maximum_pairs=32, cache_entries=4, cache_bytes=2*1024**3):
            result = model.predict(birth["solver_input"], progress_callback=lambda value:
                report("SOLVING_HISTORY", probe=index, detail=value))
        validate_history(result, case.cycles, case.anneal_c)
        elapsed = time.perf_counter()-begin
        result.update(anchor_probe=birth, ofat_case=case.record(), measured_history_seconds=elapsed)
        write(directory / f"probe_{index:02d}.json", result)
        output = pilot.mechanical_outputs(result, ledger)
        for field in ("labels", "units", "dependencies"):
            if output[field] != p6_package["model"][field]:
                raise ValueError(f"mechanical channel descriptor mismatch: {field}")
        response, audit = prepare_metric(result)
        # Only the channel layout and footprint are used from the descriptor;
        # every mechanical value and ABD response comes from this p2 history.
        direct = population.mechanics(p6_package["model"], np.asarray([output["values"]]), response)
        pair = []
        for tensor in (direct["tensor"], transfer[index:index+1]):
            pair.append(scoring.score_gamma(np.repeat(tensor, plan["placement_samples"], axis=0),
                plan["gamma"], cfg, fixed, sites, particle, chip, threshold, draws))
        row = dict(probe=index, history_seconds=elapsed, amplitudes=birth["source_amplitudes"],
            direct_tensor_ppm=(direct["tensor"][0]*1e6).tolist(),
            transfer_tensor_ppm=(transfer[index]*1e6).tolist(),
            max_tensor_absolute_error_ppm=float(np.max(np.abs(direct["tensor"][0]-transfer[index]))*1e6),
            direct_body_bow_um=direct["body"][0].tolist(), direct_rt_bow_um=direct["rt"][0].tolist(),
            conditional_yield_direct=pair[0], conditional_yield_transfer=pair[1],
            conditional_overlay_difference_pp=100*(pair[1]["overlay_yield"]-pair[0]["overlay_yield"]),
            metric_audit=audit, source_population_yield_estimate=False)
        write(directory / f"comparison_{index:02d}.json", row)
        rows.append(row)
        write(directory / "SUMMARY.json", dict(state="RUNNING", probes=rows, qualification="diagnostic_only"))
    write(directory / "SUMMARY.json", dict(state="COMPLETE", probes=rows,
        source_population_yield_estimate=False, qualification="diagnostic_only"))


def run_one(h, report):
    from threadpoolctl import threadpool_limits
    directory = ROOT / f"h{h}"
    directory.mkdir(parents=True, exist_ok=False)
    plan = preflight(h)
    write(directory / "PLAN.json", plan)
    api = geometry_api()
    compact = json.loads(compact_package().read_text())
    package = dict(schema="yap.p2_thickness_operator_package.v1", state="CANDIDATE_NOT_VALIDATED",
        rank=24, port_basis=compact["port_basis"], port_basis_sha256=compact["port_basis_sha256"],
        datasets={k: compact["datasets"][k] for k in ("bi", "beol")},
        dataset_sha256={k: compact["dataset_sha256"][k] for k in ("bi", "beol")}, prepared={})
    with threadpool_limits(limits=4):
        for v in (h-13, h-11.5):
            case = api.CASES[f"bi2_tsv_h{v:g}".replace(".", "p")]
            receipt = prepare_operator(api, case, report)
            package["datasets"][case.library_key] = receipt["path"]
            package["dataset_sha256"][case.library_key] = receipt["sha256"]
        write(directory / "OPERATOR_PACKAGE.json", package)
        histories(h, plan, package, report)


def run(args):
    import fcntl
    job = ROOT / "workers" / args.job
    started = time.perf_counter()
    with (job / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for h in args.thicknesses:
            def report(state, **fields):
                row = dict(state=state, thickness=h, host=socket.gethostname(), pid=os.getpid(),
                           elapsed_job_s=time.perf_counter()-started, **fields)
                write(job / "RUN_STATUS.json", row)
                if (ROOT / f"h{h}").exists():
                    write(ROOT / f"h{h}/RUN_STATUS.json", row)
                print(json.dumps(row), flush=True)
            try:
                run_one(h, report)
                report("COMPLETE")
            except Exception:
                report("FAILED", traceback=traceback.format_exc())
                raise
        write(job / "COMPLETE.json", dict(thicknesses=args.thicknesses, elapsed_s=time.perf_counter()-started))


def launch(args):
    if socket.gethostname().split(".")[0] != "sysml":
        raise ValueError("user selected sysml")
    for h in args.thicknesses:
        preflight(h)
        if (ROOT / f"h{h}").exists():
            raise FileExistsError(ROOT / f"h{h}")
    cpus = list(range(args.cpu_start, args.cpu_start+4))
    if not set(cpus).issubset(os.sched_getaffinity(0)):
        raise ValueError("requested CPUs unavailable")
    scratch = SCRATCH / "runtime" / args.job
    scratch.mkdir(parents=True, exist_ok=True)
    check = resources()
    if check["available_memory_gib"] < 80 or check["free_scratch_gib"] < 200:
        raise RuntimeError(f"insufficient launch headroom: {check}")
    job = ROOT / "workers" / args.job
    job.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch),
        MPLCONFIGDIR=str(scratch / "mpl"), NUMBA_CACHE_DIR=str(scratch / "numba"),
        OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4",
        MKL_DYNAMIC="FALSE", OMP_WAIT_POLICY="PASSIVE", KMP_BLOCKTIME="0", NUMBA_THREADING_LAYER="omp",
        PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", MAPDL_HEARTBEAT_SEC="60")
    command = ["nice", "-n", "10", "taskset", "-c", ",".join(map(str, cpus)), sys.executable,
        "-B", str(Path(__file__).resolve()), "run", "--job", args.job,
        "--thicknesses", *map(str, args.thicknesses)]
    with (job / "RUN.log").open("x") as log:
        process = subprocess.Popen(command, cwd=YAP, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(job / "LAUNCH.json", dict(pid=process.pid, command=command, preflight=check,
        host=socket.gethostname(), cpu_affinity=cpus, launched_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    time.sleep(2)
    print(json.dumps(dict(job=args.job, thicknesses=args.thicknesses, pid=process.pid,
                         exit_code=process.poll(), output=str(job))), flush=True)
    if process.poll() is not None:
        raise RuntimeError("worker exited during launch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "launch", "run"))
    parser.add_argument("--thicknesses", nargs="+", type=int, choices=THICKNESSES, required=True)
    parser.add_argument("--job", default="pilot")
    parser.add_argument("--cpu-start", type=int, default=17)
    args = parser.parse_args()
    if len(set(args.thicknesses)) != len(args.thicknesses) or Path(args.job).name != args.job:
        parser.error("require distinct thicknesses and a single-component job name")
    if args.action == "preflight":
        for h in args.thicknesses:
            print(json.dumps(preflight(h), indent=2))
    elif args.action == "launch":
        launch(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
