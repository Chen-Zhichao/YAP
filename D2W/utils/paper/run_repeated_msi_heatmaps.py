#!/usr/bin/env python3
"""Low-priority, separately qualified Repeated-MSI thickness campaign."""
from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

import repair_original_tier_heatmaps as repair

ROOT = repair.YAP / "D2W/output/case_studies/repeated_msi_heatmap_20260918_v1"
CONTRACT = "intended_f2b"
HEIGHTS = (50, 80, 20, 30, 40, 60, 70)


def context(pitch, height, z):
    import numpy as np
    import exp12_ofat_case_study_20260913 as source
    from exp12_ofat_frozen_study_20260913 import load_frozen_study
    from initial_bow_distribution import D2D_INITIAL_BOW

    study = load_frozen_study(repair.common.source_study())
    case = replace(study.cases["bi2" if pitch == 2 else "baseline"],
                   case_id=f"msi_p{pitch}_h{height}", total_die_um=float(height),
                   changed_parameter="orientation_and_thickness", label="Repeated-MSI")
    ledger = source.case_ledger(case)
    first = [p for p in ledger["layers"] if p["die_index"] == 0]
    first = [next(p for p in first if p["role"] == role)
             for role in ("beol", "tsv", "front_bi")]
    cursor = 0.0
    for layer in first:
        if layer["role"] == "front_bi":
            layer["role"], layer["name"] = "back_bi", "d1_back_bi"
        layer.update(z_bottom_um=cursor, z_center_um=cursor + layer["thickness_um"] / 2,
                     z_top_um=cursor + layer["thickness_um"],
                     thickness_orientation=-1 if layer["kind"] == "bi" else 1)
        cursor = layer["z_top_um"]
    ledger["layers"] = first + [p for p in ledger["layers"] if p["die_index"] != 0]
    ledger["dies"][0].update(layer_names=[p["name"] for p in first], source_front_orientation=-1)
    for stage in ledger["stages"]:
        stage["active_layers"] = [p["name"] for p in ledger["layers"]
                                  if p["die_index"] < stage["die_count"]]
    ledger.update(process_id="repeated_msi_prefabricated", tier_order_contract=CONTRACT,
                  builder_sha256=repair.common.sha(Path(__file__)))
    expected = [["beol", "tsv_si", "bi"], ["bi", "beol", "tsv_si", "bi"],
                ["bi", "beol", "tsv_si", "bi"], ["bi", "beol", "tsv_si"]]
    for index, kinds in enumerate(expected):
        layers = [p for p in ledger["layers"] if p["die_index"] == index]
        if [p["kind"] for p in layers] != kinds:
            raise ValueError("Repeated-MSI layer order mismatch")
        if not np.isclose(sum(p["thickness_um"] for p in layers), height):
            raise ValueError("die thickness does not close")
    material = source.FrozenV2Materials()(case)
    responses = [source.clpt_source_response(source._manufacturing_layers(ledger, i, material))
                 for i in range(4)]
    study.cases = {case.case_id: case}
    study._ledgers_json = source._json({case.case_id: ledger})
    study._materials_json = source._json({case.case_id: material})
    study._responses_json = source._json({case.case_id: responses})
    study.source_amplitudes = D2D_INITIAL_BOW.sample(np.asarray(z, float))
    study.sample_count = len(z)
    return study, case, ledger, None


def configure_worker():
    # Each worker has its own interpreter. Never mutate the live Legacy workers.
    repair.ROOT, repair.CONTRACT = ROOT, CONTRACT
    repair.context = context
    repair.old_ibl = lambda pitch, height: None


def score_case(pitch, height):
    import numpy as np
    import plot_original_tier_heatmaps as projection
    import plot_d2d_bi2_bi6_size_thickness_heatmaps as plotting
    import run_exp12_pitch_gamma_sweep_20260915 as scoring
    from initial_bow_distribution import D2D_INITIAL_BOW

    directory = ROOT / f"p{pitch}_h{height}"
    ready = repair.read(directory / "READY.json")
    path = Path(ready["package"])
    if ready["contract"] != CONTRACT or repair.common.sha(path) != ready["package_sha256"]:
        raise ValueError("MSI package qualification mismatch")
    package = scoring.load_package(path)
    rng = np.random.default_rng(np.random.SeedSequence([scoring.SEED, 0]))
    samples = 20_000
    amplitudes = D2D_INITIAL_BOW.sample(rng.standard_normal((samples, 4)))
    draws = scoring.placement_draws(samples)
    rows = []
    for size in plotting.DIE_SIZES_MM:
        tensor, outside = projection.size_tensor(package, amplitudes, size)
        cfg, fixed, sites, particle, chip, threshold = plotting.heatmap_model.inputs(
            package, pitch, height, size, 24)
        for gamma in ((.75, .8, .85) if pitch == 2 else (.75, .8)):
            result = scoring.score_gamma(tensor, gamma, cfg, fixed, sites, particle, chip, threshold, draws)
            rows.append(dict(bond_pitch_um=pitch, die_size_mm=size, total_die_thickness_um=height,
                             gamma=gamma, samples=samples, tier_order_contract=CONTRACT,
                             critical_fraction=.3, source_mean=D2D_INITIAL_BOW.mean,
                             source_std=D2D_INITIAL_BOW.std, source_cv=D2D_INITIAL_BOW.target_cv,
                             outside_ibl_training_box_fraction=outside, **result))
    plotting.write_csv(directory / "YIELDS.csv", rows)


def preflight():
    import numpy as np
    from run_exp12_mixed_beol_sequential_admission import definitions

    expected, _ = definitions("LLLL", orientation_contract=CONTRACT)
    report = []
    for pitch in (2, 6):
        for height in HEIGHTS:
            study, case, ledger, _ = context(pitch, height, repair.probes())
            if [p["thickness_orientation"] for p in ledger["layers"]] != [p.thickness_orientation for p in expected]:
                raise ValueError("orientation differs from existing admitted MSI campaign")
            if len({p["name"] for p in ledger["layers"]}) != len(ledger["layers"]):
                raise ValueError("duplicate layer names")
            for i in range(8):
                if not study.birth(case.case_id, i)["sign_roundtrip_passed"]:
                    raise ValueError("source sign roundtrip failed")
            path = repair.operator_path(pitch, height)
            package = repair.read(path)
            for key, thickness in ledger["required_operator_thickness_mm"].items():
                if key not in package["datasets"] or not Path(package["datasets"][key]).is_file():
                    raise ValueError(f"missing operator: {key}, {thickness}")
            report.append(dict(pitch=pitch, height=height, operator=str(path),
                               mean_source_bows_um=study.birth(case.case_id, 0)["solver_input"]["die_edge_bows_um"]))
    return dict(contract=CONTRACT, name="Repeated-MSI", cases=report,
                script_sha256=repair.common.sha(Path(__file__)),
                source_policy="frozen calibration; anchor preserving CV 0.2",
                priority="secondary to Legacy-Mixed", independent_mapdl_validation=False)


def manager():
    queue = [(p, h) for h in HEIGHTS for p in (2, 6)]
    slots, running, done, failed = [(104, 107), (108, 111)], {}, [], []
    started = time.monotonic()
    with (ROOT / ".manager.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while queue or running:
            for key, (proc, slot) in list(running.items()):
                code = proc.poll()
                if code is not None:
                    (done if code == 0 else failed).append(dict(case=key, exit_code=code))
                    slots.append(slot)
                    del running[key]
            mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            available = int(mem["MemAvailable"].split()[0]) / 1024**2
            # Reserve memory and cores for the primary campaign and other users.
            if queue and slots and available > 160 and os.getloadavg()[0] < 64 and not failed:
                pitch, height = queue.pop(0)
                slot = slots.pop(0)
                key = f"p{pitch}_h{height}"
                directory = ROOT / key
                directory.mkdir(parents=True, exist_ok=True)
                command = ["nice", "-n", "15", "taskset", "-c", f"{slot[0]}-{slot[1]}",
                           sys.executable, "-B", str(Path(__file__).resolve()), "worker",
                           "--pitch", str(pitch), "--height", str(height)]
                with (directory / "RUN.log").open("x") as log:
                    proc = subprocess.Popen(command, cwd=repair.YAP, stdin=subprocess.DEVNULL,
                                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                repair.write(directory / "LAUNCH.json", dict(host=socket.gethostname(), pid=proc.pid,
                                                             command=command, time=time.time()))
                running[key] = (proc, slot)
            repair.write(ROOT / "STATUS.json", dict(state="RUNNING", completed=done, failed=failed,
                         running={k: p.pid for k, (p, _) in running.items()}, queued=queue,
                         elapsed_s=time.monotonic()-started))
            if failed and not running:
                repair.write(ROOT / "STATUS.json", dict(state="STOPPED_FOR_REVIEW", failed=failed,
                                                         completed=done, queued=queue))
                return
            if time.monotonic()-started > 20*3600:
                raise RuntimeError("campaign exceeded bounded launch window; inspect remaining workers")
            time.sleep(15)
        import plot_d2d_bi2_bi6_size_thickness_heatmaps as plotting
        rows = []
        for pitch in (6, 2):
            for height in HEIGHTS:
                rows.extend(plotting.read_csv(ROOT / f"p{pitch}_h{height}/YIELDS.csv"))
        output = ROOT / "figures"
        output.mkdir(exist_ok=True)
        plotting.write_csv(output / "ALL_DATA.csv", rows)
        for gamma, pitches in ((.75, (6., 2.)), (.8, (6., 2.)), (.85, (2.,))):
            subset = [r for r in rows if r["gamma"] == gamma and r["bond_pitch_um"] in pitches]
            plotting.plot(subset, output, gamma, pitches, .3, 0)
        repair.write(ROOT / "STATUS.json", dict(state="COMPLETE", completed=done, failed=failed))


def launch():
    if socket.gethostname().split(".")[0] != "mfd":
        raise ValueError("secondary campaign runs on mfd")
    if not set(range(104, 112)).issubset(os.sched_getaffinity(0)):
        raise ValueError("secondary CPU slots unavailable")
    plan = preflight()
    ROOT.mkdir(parents=True, exist_ok=False)
    repair.write(ROOT / "PLAN.json", plan)
    scratch = Path("/home/scratch/zhichao/YAP_STACK_MAPDL/repeated_msi_heatmap_20260918_v1")
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch),
               MPLCONFIGDIR=str(scratch / "mpl"), NUMBA_CACHE_DIR=str(scratch / "numba"),
               OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4",
               MKL_DYNAMIC="FALSE", OMP_WAIT_POLICY="PASSIVE", KMP_BLOCKTIME="0",
               NUMBA_THREADING_LAYER="omp", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "manager"]
    with (ROOT / "MANAGER.log").open("x") as log:
        proc = subprocess.Popen(command, cwd=repair.YAP, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    repair.write(ROOT / "LAUNCH.json", dict(host="mfd", pid=proc.pid, command=command, time=time.time()))
    print(proc.pid, ROOT, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "launch", "manager", "worker"))
    parser.add_argument("--pitch", type=int, choices=(2, 6))
    parser.add_argument("--height", type=int, choices=HEIGHTS)
    args = parser.parse_args()
    if args.action == "preflight":
        import json
        print(json.dumps(preflight(), indent=2))
    elif args.action == "launch":
        launch()
    elif args.action == "manager":
        try:
            manager()
        except Exception:
            repair.write(ROOT / "MANAGER_ERROR.json", dict(traceback=traceback.format_exc()))
            raise
    else:
        configure_worker()
        repair.run_case(args.pitch, args.height)
        score_case(args.pitch, args.height)


if __name__ == "__main__":
    main()
