#!/usr/bin/env python3
"""Publish heatmaps only from checked F2F-base/F2B-stack response packages."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import repair_original_tier_heatmaps as repair
import plot_d2d_bi2_bi6_size_thickness_heatmaps as plotting
import run_exp12_pitch_gamma_sweep_20260915 as scoring
import exp12_layer_ibl_population_20260914 as population
from initial_bow_distribution import D2D_INITIAL_BOW
import numpy as np

ROOT = repair.ROOT
SAMPLES = 20_000


def size_tensor(package, amplitudes, size):
    model = copy.deepcopy(package["model"])
    ratio = size / float(model["context"]["case"]["size_mm"])
    coefficients = np.asarray(model["coefficients"]).copy()
    bows = np.asarray([name.endswith("_um") for name in model["labels"]])
    if bows.sum() != 16 * 3:
        raise ValueError("unexpected response channels for size projection")
    coefficients[:, bows] *= ratio**2
    model["coefficients"] = coefficients.tolist()
    model["context"]["case"]["size_mm"] = size
    values, outside = population.predict_rt(model, amplitudes)
    return population.mechanics(model, values, np.asarray(package["metric_response"]))["tensor"], float(outside.mean())


def main():
    output = ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(np.random.SeedSequence([scoring.SEED, 0]))
    amplitudes = D2D_INITIAL_BOW.sample(rng.standard_normal((SAMPLES, 4)))
    draws = scoring.placement_draws(SAMPLES)
    cache = repair.YAP / "D2W/output/case_studies/magnification_correction_sensitivity/d2d_gamma_data.csv"
    with cache.open() as stream:
        baseline = {(float(r["bond_pitch_um"]), round(float(r["gamma"]),2)): r
                    for r in csv.DictReader(stream) if r["physics"] == "cu_inelastic"}
    rows, regression, sources = [], [], {}
    for pitch in (6,2):
        for height in (20,30,40,50,60,70,80):
            ready = repair.read(ROOT / f"p{pitch}_h{height}/READY.json")
            path = Path(ready["package"])
            if ready["contract"] != repair.CONTRACT or ready["distribution"] != D2D_INITIAL_BOW.audit():
                raise ValueError("unqualified geometry/source distribution")
            if repair.common.sha(path) != ready["package_sha256"]:
                raise ValueError("response package changed after checks")
            package = scoring.load_package(path)
            study, case, ledger, _ = repair.context(pitch,height,repair.probes())
            if ledger != package["model"]["context"]["geometry"]:
                raise ValueError("heatmap mixes different physical tier conventions")
            sources[f"p{pitch}_h{height}"] = ready
            for size in plotting.DIE_SIZES_MM:
                tensor, outside = size_tensor(package, amplitudes, size)
                cfg, fixed, sites, particle, chip, threshold = plotting.heatmap_model.inputs(
                    package, pitch, height, size, 24)
                for gamma in ((.75,.8,.85) if pitch == 2 else (.75,.8)):
                    result = scoring.score_gamma(tensor,gamma,cfg,fixed,sites,particle,chip,threshold,draws)
                    row = dict(process="D2D",physics="cu_inelastic",tier_order_contract=repair.CONTRACT,
                        mechanical_provenance="direct_original_tier_thickness_IBL",
                        bond_pitch_um=pitch,die_size_mm=size,total_die_thickness_um=height,
                        gamma=gamma,critical_fraction=.3,source_cv=D2D_INITIAL_BOW.target_cv,
                        source_mean=D2D_INITIAL_BOW.mean,source_std=D2D_INITIAL_BOW.std,
                        samples=SAMPLES,threshold_um=threshold,
                        outside_ibl_training_box_fraction=outside,**result)
                    rows.append(row)
                    if size == 10 and height == 50:
                        ref = baseline[(pitch,gamma)]
                        if (int(ref["samples"]) != SAMPLES or
                            not np.isclose(float(ref["source_mean"]),D2D_INITIAL_BOW.mean) or
                            not np.isclose(float(ref["source_std"]),D2D_INITIAL_BOW.std)):
                            raise ValueError("gamma sensitivity cache uses different source sampling")
                        error = abs(float(ref["overlay_yield"])-result["overlay_yield"])
                        regression.append(dict(pitch=pitch,gamma=gamma,absolute_error=error,passed=error<1e-10))
                        repair.write(ROOT/"BASELINE_REGRESSION.json",regression)
                        if error >= 1e-10:
                            raise ValueError(f"baseline differs from gamma sensitivity: {regression[-1]}")
            plotting.write_csv(output/"ALL_DATA.csv", rows)
            print(json.dumps(dict(pitch=pitch,height=height,completed_rows=len(rows))),flush=True)
    for gamma,pitches in ((.75,(6.,2.)),(.8,(6.,2.)),(.85,(2.,))):
        selected=[r for r in rows if r["gamma"] == gamma and r["bond_pitch_um"] in pitches]
        name="_".join(f"BI{p:g}" for p in pitches)
        plotting.write_csv(output/f"D2D_{name}_SIZE_THICKNESS_CRITICAL_30PCT_GAMMA_{plotting.gamma_tag(gamma)}.csv",selected)
        minimum=min(100*r["overlay_yield"] for r in selected)
        color_min=40 if gamma == .85 and minimum>40 else 0
        plotting.plot(selected,output,gamma,pitches,.3,color_min)
    repair.write(output/"MANIFEST.json",dict(contract=repair.CONTRACT,source_packages=sources,
        distribution=D2D_INITIAL_BOW.audit(),samples=SAMPLES,quadrature_order=24,
        baseline_regression=regression,physical_validation=False,
        size_projection="bow channels scale with L^2; metric recomputed for each footprint",
        validation_scope="IBL checked against its reduced layer operators; p2 thickness operators provisional",
        files_sha256={str(p):repair.common.sha(p) for p in [Path(__file__),Path(plotting.__file__)]}))


if __name__ == "__main__":
    main()
