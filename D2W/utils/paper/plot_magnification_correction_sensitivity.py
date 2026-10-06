#!/usr/bin/env python3
"""Plot D2D/W2W overlay-yield sensitivity to affine correction efficiency."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
STACK_ROOT = REPO_ROOT.parent
VALIDATION_ROOT = STACK_ROOT / "Ansys_Verify/12_stack_model_validation"
RESULTS_ROOT = VALIDATION_ROOT / "results"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "D2W/output/case_studies/magnification_correction_sensitivity"
)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-yap-mag-correction-sensitivity")
sys.path.insert(0, str(STACK_ROOT))
sys.path.insert(0, str(VALIDATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))
from mechanics_runtime import resource_directory


def d2d_source():
    return resource_directory("pitch_gamma", native_root=VALIDATION_ROOT) / "PITCH_GAMMA_SWEEP.csv"

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import run_exp12_pitch_gamma_sweep_20260915 as d2d_model
import run_exp12_bi2_bi6_w2w_gamma_window_20260915 as w2w_model
from W2W.utils.paper import score_w2w_source_population as w2w_population
from YAP.initial_bow_distribution import D2D_INITIAL_BOW


PITCHES_UM = (6.0, 2.0)
PHYSICS = ("cu_inelastic", "elastic_only")
D2D_GAMMAS = tuple(round(value, 2) for value in np.arange(0.70, 1.001, 0.05))
W2W_GAMMAS = (
    *tuple(round(value, 2) for value in np.arange(0.85, 1.00, 0.02)),
    1.00,
)
D2D_SAMPLES = 20_000
D2D_SOURCE_CV = D2D_INITIAL_BOW.target_cv
D2D_SOURCE_MEAN = D2D_INITIAL_BOW.mean
D2D_SOURCE_STD = D2D_INITIAL_BOW.std
W2W_SAMPLES = 4096
W2W_QUADRATURE_ORDER = 12
W2W_SOURCE_CV = 0.20
W2W_SOURCE_MEAN = (1.0 + 3.0 * 0.10) / (1.0 + 3.0 * W2W_SOURCE_CV)
W2W_SOURCE_STD = W2W_SOURCE_CV * W2W_SOURCE_MEAN


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="ascii") as stream:
        return list(csv.DictReader(stream))


def read_archived_rows(path: Path) -> list[dict]:
    """Load a saved sensitivity table without invoking either yield model."""
    rows = read_csv(path)
    float_fields = (
        "bond_pitch_um",
        "gamma",
        "overlay_yield",
        "die_size_mm",
        "die_thickness_um",
        "source_cv",
        "source_mean",
        "source_std",
    )
    for row in rows:
        for field in float_fields:
            if field in row and row[field] != "":
                row[field] = float(row[field])
        if "samples" in row and row["samples"] != "":
            row["samples"] = int(row["samples"])
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publication_font() -> str:
    for family in ("Calibri", "Carlito"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            continue
    return "DejaVu Sans"


def apply_style() -> str:
    family = publication_font()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.style": "italic",
            "font.weight": "bold",
            "font.size": 15,
            "axes.labelsize": 18,
            "axes.labelweight": "bold",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "legend.fontsize": 14,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    return family


def select_and_extend_d2d_rows(cache_path: Path) -> list[dict]:
    indexed = {}
    if cache_path.exists():
        for row in read_csv(cache_path):
            physics = row.get("physics", "cu_inelastic")
            source_cv = float(row.get("source_cv", 0.10))
            source_mean = float(row.get("source_mean", np.nan))
            source_std = float(row.get("source_std", np.nan))
            if (
                not np.isclose(source_cv, D2D_SOURCE_CV)
                or not np.isclose(source_mean, D2D_SOURCE_MEAN)
                or not np.isclose(source_std, D2D_SOURCE_STD)
            ):
                continue
            key = (
                float(row["bond_pitch_um"]),
                round(float(row["gamma"]), 2),
                physics,
            )
            indexed[key] = {
                **row,
                "bond_pitch_um": key[0],
                "gamma": key[1],
                "physics": physics,
                "overlay_yield": float(row["overlay_yield"]),
                "samples": int(row["samples"]),
                "die_size_mm": float(row["die_size_mm"]),
                "die_thickness_um": float(row["die_thickness_um"]),
                "source_cv": source_cv,
                "source_mean": source_mean,
                "source_std": source_std,
            }
    for row in read_csv(d2d_source()) if np.isclose(D2D_SOURCE_CV, 0.10) else ():
        pitch = float(row["bond_pitch_um"])
        gamma = round(float(row["gamma"]), 2)
        physics = row["physics"]
        if (
            physics not in PHYSICS
            or pitch not in PITCHES_UM
            or gamma not in D2D_GAMMAS
        ):
            continue
        indexed[(pitch, gamma, physics)] = {
            "process": "D2D",
            "physics": physics,
            "bond_pitch_um": pitch,
            "gamma": gamma,
            "overlay_yield": float(row["overlay_yield"]),
            "samples": int(row["samples"]),
            "die_size_mm": 10.0,
            "die_thickness_um": 50.0,
            "source_cv": 0.10,
            "source": str(d2d_source().resolve()),
        }

    missing = [
        (pitch, gamma, physics)
        for pitch in PITCHES_UM
        for gamma in D2D_GAMMAS
        for physics in PHYSICS
        if (pitch, gamma, physics) not in indexed
    ]
    if missing:
        package_root = resource_directory("layer_ibl", native_root=VALIDATION_ROOT)
        packages = {
            2.0: d2d_model.load_package(package_root / "bi2/PACKAGE.json"),
            6.0: d2d_model.load_package(package_root / "baseline/PACKAGE.json"),
        }
        source_z = np.random.default_rng(
            np.random.SeedSequence([d2d_model.SEED, 0])
        ).standard_normal((D2D_SAMPLES, 4))
        amplitudes = D2D_SOURCE_MEAN + D2D_SOURCE_STD * source_z
        inelastic_tensors = {
            pitch: d2d_model.mechanics(package, amplitudes)[0]
            for pitch, package in packages.items()
        }
        elastic_coefficients, elastic_audit = d2d_model.load_elastic_response(
            d2d_model.ELASTIC_ROOT.resolve()
        )
        if not elastic_audit["exact_quadratic_check_passed"]:
            raise ValueError("elastic response audit failed")
        elastic_tensor = d2d_model.elastic_mechanics(elastic_coefficients, amplitudes)
        draws = d2d_model.placement_draws(D2D_SAMPLES)
        inputs = {}
        for pitch, package in packages.items():
            inputs[pitch] = d2d_model.pitch_score(package["score"], pitch)
        for pitch, gamma, physics in missing:
            _, cfg, _, fixed, sites, particle, chip, threshold, _ = inputs[pitch]
            tensor = (
                inelastic_tensors[pitch]
                if physics == "cu_inelastic"
                else elastic_tensor
            )
            result = d2d_model.score_gamma(
                tensor, gamma, cfg, fixed, sites,
                particle, chip, threshold, draws,
            )
            indexed[(pitch, gamma, physics)] = {
                "process": "D2D",
                "physics": physics,
                "bond_pitch_um": pitch,
                "gamma": gamma,
                "overlay_yield": float(result["overlay_yield"]),
                "samples": D2D_SAMPLES,
                "die_size_mm": 10.0,
                "die_thickness_um": 50.0,
                "source_cv": D2D_SOURCE_CV,
                "source_mean": D2D_SOURCE_MEAN,
                "source_std": D2D_SOURCE_STD,
                "source": str(packages[pitch]["score"].get("case", "frozen_ibl")),
            }

    rows = [
        indexed[(pitch, gamma, physics)]
        for pitch in PITCHES_UM
        for gamma in D2D_GAMMAS
        for physics in PHYSICS
    ]
    write_csv(cache_path, rows)
    return rows


@lru_cache(maxsize=1)
def _w2w_source_sensitivity() -> tuple[np.ndarray, dict]:
    return w2w_population.fit_source_sensitivity(w2w_model.scorer)


def _score_w2w_job(arguments: tuple[float, float, str]) -> dict:
    pitch, gamma, physics = arguments
    control = "nominal" if physics == "cu_inelastic" else "elastic"
    root = w2w_model.SOURCES[pitch]
    source = root / f"h50_{control}" / "HISTORY.json"
    raw = source.read_bytes()
    history = json.loads(raw)
    mean_tensors = np.asarray(
        [
            record["affine_magnification_tensor"]
            for record in w2w_model.scorer._interface_records(history["formations"])
        ]
    )
    sensitivity, fit_diagnostics = _w2w_source_sensitivity()
    source_z = np.random.default_rng(
        np.random.SeedSequence([w2w_population.SEED, 0])
    ).standard_normal((W2W_SAMPLES, 4))
    amplitudes = W2W_SOURCE_MEAN + W2W_SOURCE_STD * source_z
    tensors = mean_tensors[None] + np.einsum(
        "nw,weij->neij", amplitudes - 1.0, sensitivity, optimize=True
    )
    residual = (1.0 - gamma) * tensors

    inputs = w2w_model.inputs_for_pitch(pitch, 10.0, W2W_QUADRATURE_ORDER)
    placement_z = np.random.default_rng(
        np.random.SeedSequence([w2w_population.SEED, 1])
    ).standard_normal((W2W_SAMPLES, 3, 3))
    translations = placement_z[..., :2] * w2w_model.scorer.TRANSLATION_SIGMA_UM
    rotations = placement_z[..., 2] * w2w_model.scorer.ROTATION_SIGMA_RAD
    accumulator = w2w_model.scorer._ProbabilityAccumulator(len(inputs["sites"]))
    for first in range(0, W2W_SAMPLES, w2w_model.scorer.DRAW_CHUNK):
        stop = min(W2W_SAMPLES, first + w2w_model.scorer.DRAW_CHUNK)
        matrix = (
            residual[first:stop]
            + rotations[first:stop, :, None, None] * w2w_model.scorer.J
        )
        translation = np.einsum(
            "neij,sj->nesi", matrix, inputs["sites"], optimize=True
        )
        translation += translations[first:stop, :, None]
        probability = w2w_model.fixed_score.fixed_map_survival(
            matrix[:, :, None],
            translation,
            inputs["fixed"],
            threshold_um=inputs["threshold"],
            sigma_um=w2w_model.scorer.PAD_NOISE_SIGMA_UM,
            batch_size=2048,
        )
        accumulator.add(probability)
    values = accumulator.finish()
    overlay = float(values["laws"][w2w_model.scorer.LAWS[0]][-1])
    return {
        "process": "W2W",
        "physics": physics,
        "bond_pitch_um": pitch,
        "gamma": gamma,
        "overlay_yield": overlay,
        "samples": W2W_SAMPLES,
        "die_size_mm": 10.0,
        "die_thickness_um": 50.0,
        "source_cv": W2W_SOURCE_CV,
        "source_mean": W2W_SOURCE_MEAN,
        "source_std": W2W_SOURCE_STD,
        "source_response_loo_rmse_ppm": fit_diagnostics["leave_one_out_rmse_ppm"],
        "source_response_transfer": "p6 h50 anchors applied as source perturbations",
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
    }


def select_and_extend_w2w_rows(cache_path: Path, workers: int) -> list[dict]:
    indexed = {}
    if cache_path.exists():
        for row in read_csv(cache_path):
            physics = row.get("physics", "cu_inelastic")
            if (
                not np.isclose(float(row.get("source_cv", np.nan)), W2W_SOURCE_CV)
                or not np.isclose(float(row.get("source_mean", np.nan)), W2W_SOURCE_MEAN)
                or not np.isclose(float(row.get("source_std", np.nan)), W2W_SOURCE_STD)
            ):
                continue
            key = (
                float(row["bond_pitch_um"]),
                round(float(row["gamma"]), 2),
                physics,
            )
            indexed[key] = {
                **row,
                "bond_pitch_um": key[0],
                "gamma": key[1],
                "physics": physics,
                "overlay_yield": float(row["overlay_yield"]),
                "samples": int(row["samples"]),
                "die_size_mm": float(row["die_size_mm"]),
                "die_thickness_um": float(row["die_thickness_um"]),
                "source_cv": float(row["source_cv"]),
                "source_mean": float(row["source_mean"]),
                "source_std": float(row["source_std"]),
            }

    jobs = [
        (pitch, gamma, physics)
        for pitch in PITCHES_UM
        for gamma in W2W_GAMMAS
        for physics in PHYSICS
        if (pitch, gamma, physics) not in indexed
    ]
    if jobs:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            for row in pool.map(_score_w2w_job, jobs):
                indexed[(row["bond_pitch_um"], row["gamma"], row["physics"])] = row
    rows = [
        indexed[(pitch, gamma, physics)]
        for pitch in PITCHES_UM
        for gamma in W2W_GAMMAS
        for physics in PHYSICS
    ]
    write_csv(cache_path, rows)
    return rows


def plot(d2d_rows: list[dict], w2w_rows: list[dict], output: Path) -> str:
    family = apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.45), constrained_layout=True)
    styles = {
        6.0: {"color": "#0072B2", "marker": "o"},
        2.0: {"color": "#D55E00", "marker": "s"},
    }

    for axis, process, rows, xlim, xticks in zip(
        axes,
        ("D2D", "W2W"),
        (d2d_rows, w2w_rows),
        ((0.69, 1.01), (0.84, 1.01)),
        (
            [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
            [0.85, 0.89, 0.93, 0.97, 1.00],
        ),
        strict=True,
    ):
        for pitch in PITCHES_UM:
            for physics in PHYSICS:
                group = sorted(
                    (
                        row
                        for row in rows
                        if row["bond_pitch_um"] == pitch
                        and row["physics"] == physics
                    ),
                    key=lambda row: row["gamma"],
                )
                axis.plot(
                    [row["gamma"] for row in group],
                    [100.0 * row["overlay_yield"] for row in group],
                    color=styles[pitch]["color"],
                    linestyle="-" if physics == "cu_inelastic" else "--",
                    marker=styles[pitch]["marker"],
                    markerfacecolor=(
                        styles[pitch]["color"]
                        if physics == "cu_inelastic"
                        else "white"
                    ),
                    markersize=5.5,
                    markeredgecolor=styles[pitch]["color"],
                    markeredgewidth=0.8,
                    linewidth=2.2,
                )
        axis.set_xlabel(
            "Mag. correction efficiency, "
            "\N{GREEK SMALL LETTER GAMMA}"
        )
        axis.set_xlim(*xlim)
        axis.set_xticks(xticks)
        axis.set_ylim(-2.0, 102.0)
        axis.set_yticks([0, 20, 40, 60, 80, 100])
        axis.grid(True, color="#D9D9D9", linewidth=0.65, alpha=0.75)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_color("#303030")
        for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            label.set_fontfamily(family)
            label.set_fontweight("bold")
            label.set_fontstyle("italic")

    axes[0].set_ylabel("Overlay yield (%)")
    legend = axes[0].legend(
        handles=[
            Line2D([0], [0], color=styles[6.0]["color"], marker="o",
                   markerfacecolor=styles[6.0]["color"], linewidth=2.0,
                   linestyle="-", label="6 um, Cu-inelastic"),
            Line2D([0], [0], color=styles[6.0]["color"], marker="o",
                   markerfacecolor="white", linewidth=2.0,
                   linestyle="--", label="6 um, Elastic-only"),
            Line2D([0], [0], color=styles[2.0]["color"], marker="s",
                   markerfacecolor=styles[2.0]["color"], linewidth=2.0,
                   linestyle="-", label="2 um, Cu-inelastic"),
            Line2D([0], [0], color=styles[2.0]["color"], marker="s",
                   markerfacecolor="white", linewidth=2.0,
                   linestyle="--", label="2 um, Elastic-only"),
        ],
        frameon=True,
        facecolor="white",
        edgecolor="#BFBFBF",
        framealpha=0.92,
        loc="lower right",
        ncol=1,
        fontsize=12,
        handlelength=1.8,
        handletextpad=0.45,
        labelspacing=0.22,
        borderpad=0.3,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")
        text.set_fontstyle("italic")

    png = output / "magnification_correction_efficiency_sensitivity.png"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return family


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--input-data-dir",
        type=Path,
        help=(
            "Plot archived d2d_gamma_data.csv and w2w_gamma_data.csv from this "
            "directory without running or extending either model."
        ),
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    archived_inputs = None
    if args.input_data_dir is None:
        d2d_rows = select_and_extend_d2d_rows(output / "d2d_gamma_data.csv")
        w2w_rows = select_and_extend_w2w_rows(
            output / "w2w_gamma_data.csv", args.workers
        )
    else:
        input_dir = args.input_data_dir.resolve()
        archived_inputs = {
            "d2d": input_dir / "d2d_gamma_data.csv",
            "w2w": input_dir / "w2w_gamma_data.csv",
        }
        for path in archived_inputs.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        d2d_rows = read_archived_rows(archived_inputs["d2d"])
        w2w_rows = read_archived_rows(archived_inputs["w2w"])
        write_csv(output / "d2d_gamma_data.csv", d2d_rows)
        write_csv(output / "w2w_gamma_data.csv", w2w_rows)
    all_rows = sorted(
        [*d2d_rows, *w2w_rows],
        key=lambda row: (
            row["process"], row["bond_pitch_um"], row["physics"], row["gamma"]
        ),
    )
    write_csv(output / "magnification_correction_efficiency_sensitivity.csv", all_rows)
    font = plot(d2d_rows, w2w_rows, output)
    manifest = {
        "schema": "yap.magnification_correction_efficiency_sensitivity.v1",
        "processes": ["D2D", "W2W"],
        "bond_pitches_um": list(PITCHES_UM),
        "d2d_gamma_schedule": list(D2D_GAMMAS),
        "w2w_gamma_schedule": list(W2W_GAMMAS),
        "die_size_mm": [10.0, 10.0],
        "die_thickness_um": 50.0,
        "physics": list(PHYSICS),
        "stack_height": "4-Hi",
        "yield_metric": "cumulative_overlay_yield_after_three_interfaces",
        "w2w_spatial_compounding": (
            "multiply interface survival at each registered die site, "
            "then average over wafer sites"
        ),
        "w2w_aggregation_law": "registered_site_yield",
        "affine_rule": "E_res=(1-gamma)*E_mech",
        "d2d_samples": D2D_SAMPLES,
        "d2d_source_cv": D2D_SOURCE_CV,
        "d2d_source_mean": D2D_SOURCE_MEAN,
        "d2d_source_std": D2D_SOURCE_STD,
        "w2w_samples": W2W_SAMPLES,
        "w2w_quadrature_order": W2W_QUADRATURE_ORDER,
        "w2w_source_cv": W2W_SOURCE_CV,
        "w2w_source_mean": W2W_SOURCE_MEAN,
        "w2w_source_std": W2W_SOURCE_STD,
        "source_anchor_policy": "preserve the previous mean-plus-three-sigma bow anchor",
        "font_requested": "Calibri Bold Italic",
        "font_rendered": f"{font} Bold Italic",
        "w2w_source_sensitivity_scope": (
            "linear perturbation identified from 16 completed p6 h50 Cu-inelastic "
            "source anchors and transferred to the corresponding p2 and elastic "
            "mean histories"
        ),
        "w2w_source_sensitivity_physical_validation": False,
        "inputs": {
            str(path.resolve()): sha256(path)
            for path in (
                resource_directory("layer_ibl", native_root=VALIDATION_ROOT) / "baseline/PACKAGE.json",
                resource_directory("layer_ibl", native_root=VALIDATION_ROOT) / "bi2/PACKAGE.json",
                w2w_model.SOURCES[6.0] / "h50_nominal/HISTORY.json",
                w2w_model.SOURCES[6.0] / "h50_elastic/HISTORY.json",
                w2w_model.SOURCES[2.0] / "h50_nominal/HISTORY.json",
                w2w_model.SOURCES[2.0] / "h50_elastic/HISTORY.json",
            )
        },
        "w2w_source_anchor_root": str(w2w_population.anchor_root().resolve()),
    }
    if archived_inputs is not None:
        manifest.pop("inputs", None)
        manifest.pop("w2w_source_anchor_root", None)
        archived_files = list(archived_inputs.values())
        source_manifest = args.input_data_dir.resolve() / "MANIFEST.json"
        if source_manifest.is_file():
            archived_files.append(source_manifest)
        manifest.update(
            {
                "data_mode": "archived_csv_plot_only",
                "archived_data_dir": str(args.input_data_dir.resolve()),
                "archived_inputs": {
                    str(path): sha256(path) for path in archived_files
                },
                "models_executed": False,
            }
        )
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="ascii"
    )


if __name__ == "__main__":
    main()
