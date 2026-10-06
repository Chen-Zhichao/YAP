#!/usr/bin/env python3
"""Score W2W overlay with a calibrated initial-bow source population.

The latest full-thickness history supplies the mean mechanical state.  Source
sensitivities are identified from the completed h50 source-anchor histories,
then evaluated online for independent incoming-wafer amplitudes.  This avoids
rerunning the mechanical solver while preserving joint source and placement
draws across all interfaces and registered die sites.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
STACK_ROOT = REPO_ROOT.parent
VERIFY_ROOT = STACK_ROOT / "Ansys_Verify" / "12_stack_model_validation"
sys.path.insert(0, str(REPO_ROOT))
from mechanics_runtime import resource_directory
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "W2W"
    / "output"
    / "paper_figures"
    / "W2W_SITE_OVERLAY_YIELD_GAMMA_0p9_SOURCE_CV_0p2.csv"
)

GAMMA = 0.9
SOURCE_CV = 0.2
SAMPLES = 32768
QUADRATURE_ORDER = 12
SEED = 2026091601
TARGET_TOTAL_UM = 50


def anchor_root():
    return resource_directory("w2w_anchors", native_root=VERIFY_ROOT)


def mean_history():
    return resource_directory("w2w_corrected", native_root=VERIFY_ROOT) / "h50_nominal/HISTORY.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--source-cv", type=float, default=SOURCE_CV)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--quadrature-order", type=int, default=QUADRATURE_ORDER)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 0.0 <= args.gamma <= 1.0:
        parser.error("--gamma must lie in [0, 1]")
    if args.source_cv <= 0.0:
        parser.error("--source-cv must be positive")
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.quadrature_order < 4:
        parser.error("--quadrature-order must be at least 4")
    return args


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def history_tensors(history: dict, scorer) -> np.ndarray:
    records = scorer._interface_records(history["formations"])
    return np.asarray(
        [record["affine_magnification_tensor"] for record in records],
        dtype=np.float64,
    )


def fit_source_sensitivity(scorer) -> tuple[np.ndarray, dict]:
    nominal_path = anchor_root() / "h50_nominal" / "HISTORY.json"
    nominal = history_tensors(read_json(nominal_path), scorer)
    amplitudes = []
    responses = []
    paths = sorted(anchor_root().glob("h50_draw*/HISTORY.json"))
    if len(paths) != 16:
        raise ValueError(f"Expected 16 h50 source anchors, found {len(paths)}")
    for path in paths:
        history = read_json(path)
        amplitudes.append(history["source"]["source_amplitudes"])
        responses.append(history_tensors(history, scorer) - nominal)

    x = np.asarray(amplitudes, dtype=np.float64) - 1.0
    y = np.asarray(responses, dtype=np.float64)
    shape = y.shape[1:]
    coefficients = np.linalg.lstsq(x, y.reshape(len(y), -1), rcond=None)[0]
    prediction = (x @ coefficients).reshape(y.shape)
    residual_ppm = (prediction - y) * 1.0e6

    loo = []
    for index in range(len(x)):
        keep = np.arange(len(x)) != index
        fit = np.linalg.lstsq(x[keep], y[keep].reshape(keep.sum(), -1), rcond=None)[0]
        loo.append((x[index] @ fit).reshape(shape) - y[index])
    loo_ppm = np.asarray(loo) * 1.0e6
    diagnostics = {
        "anchor_count": len(paths),
        "fit": "zero-intercept linear response about source amplitudes [1,1,1,1]",
        "training_rmse_ppm": float(np.sqrt(np.mean(residual_ppm**2))),
        "training_max_abs_error_ppm": float(np.max(np.abs(residual_ppm))),
        "leave_one_out_rmse_ppm": float(np.sqrt(np.mean(loo_ppm**2))),
        "leave_one_out_max_abs_error_ppm": float(np.max(np.abs(loo_ppm))),
        "anchor_history_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
        "nominal_anchor_sha256": hashlib.sha256(nominal_path.read_bytes()).hexdigest(),
    }
    return coefficients.reshape(4, *shape), diagnostics


def calibrated_source_distribution(history: dict, source_cv: float) -> dict:
    source = history["source"]
    old_mean = float(np.asarray(source["incoming_mean_bow_um"])[0, 0])
    old_upper = float(np.asarray(source["incoming_mean_plus_3sigma_um"])[0, 0])
    amplitude_mean = old_upper / (old_mean * (1.0 + 3.0 * source_cv))
    amplitude_std = source_cv * amplitude_mean
    return {
        "amplitude_mean": amplitude_mean,
        "amplitude_std": amplitude_std,
        "bow_mean_um": old_mean * amplitude_mean,
        "bow_std_um": old_mean * amplitude_std,
        "bow_mean_plus_3sigma_um": old_mean * (amplitude_mean + 3.0 * amplitude_std),
        "preserved_upper_anchor_um": old_upper,
        "old_unit_amplitude_bow_um": old_mean,
    }


def score(args: argparse.Namespace) -> tuple[dict, dict]:
    sys.path.insert(0, str(VERIFY_ROOT))
    try:
        import exp12_w2w_window_overlay_20260914 as scorer
        import run_exp12_fixed_bump_map_half_anchor_20260914 as fixed_score
    finally:
        sys.path.pop(0)

    history = read_json(mean_history())
    mean_tensors = history_tensors(history, scorer)
    sensitivity, fit_diagnostics = fit_source_sensitivity(scorer)
    distribution = calibrated_source_distribution(history, args.source_cv)
    inputs = scorer._yield_inputs()
    sites = np.asarray(inputs["sites"], dtype=np.float64)
    fixed = fixed_score.fixed_center_grid(inputs["geometry"], args.quadrature_order)

    source_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 0]))
    placement_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 1]))
    source_z = source_rng.standard_normal((args.samples, 4))
    placement_z = placement_rng.standard_normal((args.samples, 3, 3))
    amplitudes = (
        distribution["amplitude_mean"]
        + distribution["amplitude_std"] * source_z
    )
    tensors = mean_tensors[None] + np.einsum(
        "nw,weij->neij", amplitudes - 1.0, sensitivity, optimize=True
    )
    tensors *= 1.0 - args.gamma
    translations = placement_z[..., :2] * scorer.TRANSLATION_SIGMA_UM
    rotations = placement_z[..., 2] * scorer.ROTATION_SIGMA_RAD

    accumulator = scorer._ProbabilityAccumulator(len(sites))
    for first in range(0, args.samples, scorer.DRAW_CHUNK):
        stop = min(args.samples, first + scorer.DRAW_CHUNK)
        matrix = tensors[first:stop] + rotations[first:stop, :, None, None] * scorer.J
        translation = np.einsum("neij,sj->nesi", matrix, sites, optimize=True)
        translation += translations[first:stop, :, None]
        probabilities = fixed_score.fixed_map_survival(
            matrix[:, :, None],
            translation,
            fixed,
            threshold_um=inputs["threshold"],
            sigma_um=scorer.PAD_NOISE_SIGMA_UM,
            batch_size=2048,
        )
        accumulator.add(probabilities)

    result = accumulator.finish()
    report = {
        "schema": "yap.w2w_source_population_overlay.v1",
        "gamma": args.gamma,
        "samples": args.samples,
        "seed": args.seed,
        "quadrature_order": args.quadrature_order,
        "source_distribution": distribution,
        "source_wafers_independent": True,
        "mean_history": str(mean_history()),
        "mean_history_sha256": hashlib.sha256(mean_history().read_bytes()).hexdigest(),
        "sensitivity_anchor_root": str(anchor_root()),
        "sensitivity_transfer_scope": (
            "linear source perturbations from completed h50 anchors transferred "
            "onto the latest full-thickness mean history"
        ),
        "fit_diagnostics": fit_diagnostics,
        "residual_tensor_mean_ppm": (tensors.mean(axis=0) * 1.0e6).tolist(),
        "residual_tensor_std_ppm": (tensors.std(axis=0) * 1.0e6).tolist(),
        "marginal_overlay_yield": result["marginal"].tolist(),
        "registered_cumulative_overlay_yield": result["laws"][scorer.LAWS[0]].tolist(),
        "independent_cumulative_overlay_yield": result["laws"][scorer.LAWS[2]].tolist(),
        "registered_cumulative_mc_se": result["registered_mc_se"].tolist(),
        "threshold_um": float(inputs["threshold"]),
        "random_pad_noise_sigma_um": float(scorer.PAD_NOISE_SIGMA_UM),
        "translation_sigma_um": float(scorer.TRANSLATION_SIGMA_UM),
        "rotation_sigma_rad": float(scorer.ROTATION_SIGMA_RAD),
        "critical_pad_count": int(fixed["critical_pad_count"]),
        "site_count": len(sites),
    }
    return result, report


def write_outputs(
    output: Path,
    result: dict,
    report: dict,
    elapsed_s: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    report["runtime_s"] = elapsed_s
    sites = np.asarray(sys.modules["exp12_w2w_window_overlay_20260914"]._yield_inputs()["sites"])
    interface = np.asarray(result["interface_site"])
    cumulative = np.asarray(result["prefix_site"])
    fields = [
        "target_total_um",
        "site_id",
        "x_mm",
        "y_mm",
        "radius_mm",
        "A1_interface_overlay_yield",
        "A2_interface_overlay_yield",
        "A3_interface_overlay_yield",
        "A1_overlay_yield",
        "A2_cumulative_overlay_yield",
        "A3_cumulative_overlay_yield",
        "A3_failure_probability",
    ]
    with output.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, (x_um, y_um) in enumerate(sites):
            writer.writerow(
                {
                    "target_total_um": TARGET_TOTAL_UM,
                    "site_id": index,
                    "x_mm": x_um / 1000.0,
                    "y_mm": y_um / 1000.0,
                    "radius_mm": np.hypot(x_um, y_um) / 1000.0,
                    "A1_interface_overlay_yield": interface[0, index],
                    "A2_interface_overlay_yield": interface[1, index],
                    "A3_interface_overlay_yield": interface[2, index],
                    "A1_overlay_yield": cumulative[0, index],
                    "A2_cumulative_overlay_yield": cumulative[1, index],
                    "A3_cumulative_overlay_yield": cumulative[2, index],
                    "A3_failure_probability": 1.0 - cumulative[2, index],
                }
            )
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="ascii"
    )


def main() -> int:
    args = parse_args()
    started = perf_counter()
    result, report = score(args)
    elapsed = perf_counter() - started
    write_outputs(args.output.resolve(), result, report, elapsed)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
