#!/usr/bin/env python3
"""Generate D2D BI2/BI6 Cu-inelastic size-thickness heatmaps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter


REPO_ROOT = Path(__file__).resolve().parents[3]
STACK_ROOT = REPO_ROOT.parent
VALIDATION_ROOT = STACK_ROOT / "Ansys_Verify/12_stack_model_validation"
DEFAULT_OUTPUT = REPO_ROOT / "D2W/output/case_studies/d2d_size_thickness_heatmaps"

sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO_ROOT))
from mechanics_runtime import configure_runtime_storage, validate_write_path

configure_runtime_storage()
sys.path.insert(0, str(STACK_ROOT))
sys.path.insert(0, str(VALIDATION_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

import run_exp12_bi2_bi6_d2d_thickness_heatmap_20260915 as heatmap_model
import run_exp12_pitch_gamma_sweep_20260915 as pitch_model
from YAP.initial_bow_distribution import (
    AnchorPreservingDistribution,
    D2D_INITIAL_BOW,
)


PITCHES_UM = (6.0, 2.0)
THICKNESSES_UM = tuple(float(value) for value in range(20, 81, 10))
DIE_SIZES_MM = tuple(float(value) for value in range(5, 16))
GAMMA = 0.75
SAMPLES = 10_000
QUADRATURE_ORDER = 12
SOURCE_CV = 0.20


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
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.labelweight": "bold",
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    return family


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    return [
        {
            **row,
            "bond_pitch_um": float(row["bond_pitch_um"]),
            "die_size_mm": float(row["die_size_mm"]),
            "total_die_thickness_um": float(row["total_die_thickness_um"]),
            "gamma": float(row["gamma"]),
            "overlay_yield": float(row["overlay_yield"]),
            "samples": int(row["samples"]),
        }
        for row in rows
    ]


def gamma_tag(gamma: float) -> str:
    return f"{gamma:.2f}".replace(".", "p")


def cache_is_complete(
    rows: list[dict],
    samples: int,
    gamma: float,
    pitches: tuple[float, ...],
    critical_fraction: float,
    source_cv: float,
) -> bool:
    distribution = AnchorPreservingDistribution(
        reference_mean=D2D_INITIAL_BOW.reference_mean,
        reference_cv=D2D_INITIAL_BOW.reference_cv,
        target_cv=source_cv,
    )
    keys = {
        (
            row["bond_pitch_um"],
            row["die_size_mm"],
            row["total_die_thickness_um"],
            row["gamma"],
            row["samples"],
            float(row.get("critical_fraction", 0.30)),
            float(row.get("source_cv", 0.10)),
            float(row.get("source_mean", np.nan)),
            float(row.get("source_std", np.nan)),
        )
        for row in rows
    }
    expected = {
        (
            pitch,
            size,
            thickness,
            gamma,
            samples,
            critical_fraction,
            source_cv,
            distribution.mean,
            distribution.std,
        )
        for pitch in pitches
        for size in DIE_SIZES_MM
        for thickness in THICKNESSES_UM
    }
    return keys == expected


def calculate(
    samples: int,
    quadrature_order: int,
    gamma: float,
    pitches: tuple[float, ...] = PITCHES_UM,
    critical_fraction: float = 0.30,
    source_cv: float = SOURCE_CV,
) -> list[dict]:
    rng = np.random.default_rng(
        np.random.SeedSequence([heatmap_model.audit.SEED, 915, 1])
    )
    distribution = AnchorPreservingDistribution(
        reference_mean=D2D_INITIAL_BOW.reference_mean,
        reference_cv=D2D_INITIAL_BOW.reference_cv,
        target_cv=source_cv,
    )
    amplitudes = distribution.sample(rng.standard_normal((samples, 4)))
    packages = {
        thickness: pitch_model.load_package(
            heatmap_model.PACKAGES / f"h{int(thickness)}/PACKAGE.json"
        )
        for thickness in THICKNESSES_UM
    }

    bi6_tensors = {}
    bi6_outside = {}
    for thickness, package in packages.items():
        bi6_tensors[thickness], bi6_outside[thickness] = pitch_model.mechanics(
            package, amplitudes
        )

    bi2_package = pitch_model.load_package(heatmap_model.BI2)
    bi2_h50, bi2_outside = pitch_model.mechanics(bi2_package, amplitudes)
    bi2_tensors = {
        thickness: heatmap_model.bi2_transfer(
            bi6_tensors[thickness], bi6_tensors[50.0], bi2_h50
        )
        for thickness in THICKNESSES_UM
    }
    draws = pitch_model.placement_draws(samples)

    rows = []
    for thickness in THICKNESSES_UM:
        for size in DIE_SIZES_MM:
            for pitch in pitches:
                cfg, fixed, sites, particle, chip, threshold = heatmap_model.inputs(
                    packages[thickness], pitch, thickness, size, quadrature_order
                )
                geometry = heatmap_model.audit.compact_regular_collection(cfg)
                geometry["critical_fraction"] = critical_fraction
                fixed = heatmap_model.fixed_score.fixed_center_grid(
                    geometry, quadrature_order
                )
                if pitch == 6.0:
                    tensor = bi6_tensors[thickness]
                    outside = bi6_outside[thickness]
                    provenance = "direct_BI6_thickness_package"
                else:
                    tensor = bi2_tensors[thickness]
                    outside = max(bi6_outside[thickness], bi2_outside)
                    provenance = (
                        "h50_BI2_increment_scaled_by_direct_BI6_thickness_response"
                    )
                result = pitch_model.score_gamma(
                    tensor,
                    gamma,
                    cfg,
                    fixed,
                    sites,
                    particle,
                    chip,
                    threshold,
                    draws,
                )
                rows.append(
                    {
                        "process": "D2D",
                        "physics": "cu_inelastic",
                        "mechanical_provenance": provenance,
                        "bond_pitch_um": pitch,
                        "die_size_mm": size,
                        "total_die_thickness_um": thickness,
                        "gamma": gamma,
                        "critical_fraction": critical_fraction,
                        "source_cv": source_cv,
                        "source_mean": distribution.mean,
                        "source_std": distribution.std,
                        "samples": samples,
                        "threshold_um": threshold,
                        "critical_pad_count": fixed["critical_pad_count"],
                        "outside_ibl_training_box_fraction": outside,
                        **result,
                    }
                )
        print(
            json.dumps({"thickness_um": thickness, "completed_rows": len(rows)}),
            flush=True,
        )
    return rows


def yield_grid(rows: list[dict], pitch: float) -> np.ndarray:
    indexed = {
        (row["total_die_thickness_um"], row["die_size_mm"]): row["overlay_yield"]
        for row in rows
        if row["bond_pitch_um"] == pitch
    }
    return np.array(
        [
            [indexed[(thickness, size)] for size in DIE_SIZES_MM]
            for thickness in THICKNESSES_UM
        ]
    )


def draw_step_contour(
    axis,
    values: np.ndarray,
    threshold_percent: float,
    *,
    color: str,
    linestyle,
    linewidth: float,
) -> None:
    high_yield = 100.0 * values > threshold_percent
    x_edges = np.arange(4.5, 16.0, 1.0)
    y_edges = np.arange(15.0, 90.0, 10.0)
    segments = []

    for row in range(high_yield.shape[0]):
        for column in range(1, high_yield.shape[1]):
            if high_yield[row, column] != high_yield[row, column - 1]:
                segment = [
                    (x_edges[column], y_edges[row]),
                    (x_edges[column], y_edges[row + 1]),
                ]
                segments.append(segment)

    for row in range(1, high_yield.shape[0]):
        for column in range(high_yield.shape[1]):
            if high_yield[row, column] != high_yield[row - 1, column]:
                segments.append(
                    [
                        (x_edges[column], y_edges[row]),
                        (x_edges[column + 1], y_edges[row]),
                    ]
                )

    if not segments:
        return
    axis.add_collection(
        LineCollection(
            segments,
            colors=color,
            linewidths=linewidth,
            linestyles=linestyle,
            capstyle="butt",
            joinstyle="miter",
            zorder=4,
        )
    )


def draw_heatmap(
    axis,
    values: np.ndarray,
    title: str,
    color_min: float = 0.0,
):
    image = axis.imshow(
        100.0 * values,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=color_min,
        vmax=100.0,
        cmap="RdYlGn",
        extent=(4.5, 15.5, 15.0, 85.0),
    )
    axis.set_title(title, fontstyle="italic")
    axis.set_xlabel("Square die side (mm)")
    axis.set_xticks([5, 7, 9, 11, 13, 15])
    axis.set_yticks(THICKNESSES_UM)
    axis.tick_params(direction="out", length=3.5)
    draw_step_contour(
        axis,
        values,
        50.0,
        color="black",
        linestyle=(0, (4, 2)),
        linewidth=2.0,
    )
    draw_step_contour(
        axis,
        values,
        90.0,
        color="white",
        linestyle="solid",
        linewidth=2.2,
    )
    axis.scatter(
        [10.0],
        [50.0],
        marker="*",
        s=155,
        facecolor="black",
        edgecolor="white",
        linewidth=1.0,
        zorder=6,
    )
    return image


def plot(
    rows: list[dict],
    output: Path,
    gamma: float,
    pitches: tuple[float, ...] = PITCHES_UM,
    critical_fraction: float = 0.30,
    color_min: float = 0.0,
) -> str:
    family = apply_style()
    grids = {pitch: yield_grid(rows, pitch) for pitch in pitches}

    fig, axes = plt.subplots(
        1, len(pitches), figsize=(3.9 * len(pitches), 2.9),
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    images = []
    for axis, pitch in zip(axes, pitches, strict=True):
        images.append(
            draw_heatmap(
                axis,
                grids[pitch],
                f"Pitch = {pitch:g} \N{MICRO SIGN}m, "
                f"\N{GREEK SMALL LETTER GAMMA} = {gamma:.2f}",
                color_min,
            )
        )
    axes[0].set_ylabel("Die thickness (um)")
    colorbar = fig.colorbar(images[-1], ax=axes, pad=0.025, fraction=0.045)
    colorbar.set_label("Overlay yield (%)")
    colorbar.set_ticks(np.arange(np.ceil(color_min / 10.0) * 10.0, 100.1, 10.0))
    tag = gamma_tag(gamma)
    pitch_tag = "_".join(f"BI{pitch:g}" for pitch in pitches)
    critical_tag = f"CRITICAL_{100.0 * critical_fraction:g}PCT"
    combined = output / (
        f"D2D_{pitch_tag}_SIZE_THICKNESS_HEATMAP_"
        f"{critical_tag}_GAMMA_{tag}"
    )
    fig.savefig(combined.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    if len(pitches) > 1:
        for pitch in pitches:
            fig, axis = plt.subplots(figsize=(3.9, 2.9), constrained_layout=True)
            image = draw_heatmap(
                axis,
                grids[pitch],
                f"Pitch = {pitch:g} \N{MICRO SIGN}m, "
                f"\N{GREEK SMALL LETTER GAMMA} = {gamma:.2f}",
                color_min,
            )
            axis.set_ylabel("Die thickness (um)")
            colorbar = fig.colorbar(image, ax=axis, pad=0.025)
            colorbar.set_label("Overlay yield (%)")
            colorbar.set_ticks(
                np.arange(np.ceil(color_min / 10.0) * 10.0, 100.1, 10.0)
            )
            stem = output / (
                f"D2D_BI{pitch:g}_SIZE_THICKNESS_HEATMAP_"
                f"{critical_tag}_GAMMA_{tag}"
            )
            fig.savefig(
                stem.with_suffix(".png"), dpi=600,
                bbox_inches="tight", pad_inches=0.03,
            )
            plt.close(fig)
    return family


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--quadrature-order", type=int, default=QUADRATURE_ORDER)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument(
        "--pitch", type=float, choices=PITCHES_UM, action="append",
        help="Bond pitch to plot; repeat for multiple pitches (default: both).",
    )
    parser.add_argument("--critical-fraction", type=float, default=0.30)
    parser.add_argument("--source-cv", type=float, default=SOURCE_CV)
    parser.add_argument("--color-min", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args(argv)
    args.output = validate_write_path(args.output)
    if not 0.0 < args.critical_fraction <= 1.0:
        parser.error("--critical-fraction must lie in (0, 1]")
    if not 0.0 <= args.color_min < 100.0:
        parser.error("--color-min must lie in [0, 100)")
    if args.source_cv <= 0.0:
        parser.error("--source-cv must be positive")
    pitches = tuple(args.pitch) if args.pitch else PITCHES_UM

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tag = gamma_tag(args.gamma)
    pitch_tag = "_".join(f"BI{pitch:g}" for pitch in pitches)
    critical_tag = f"CRITICAL_{100.0 * args.critical_fraction:g}PCT"
    csv_path = output / (
        f"D2D_{pitch_tag}_SIZE_THICKNESS_{critical_tag}_GAMMA_{tag}.csv"
    )
    started = perf_counter()
    rows = read_csv(csv_path) if csv_path.exists() else []
    reused_cache = (
        not args.recompute
        and cache_is_complete(
            rows,
            args.samples,
            args.gamma,
            pitches,
            args.critical_fraction,
            args.source_cv,
        )
    )
    if not reused_cache:
        rows = calculate(
            args.samples,
            args.quadrature_order,
            args.gamma,
            pitches,
            args.critical_fraction,
            args.source_cv,
        )
        write_csv(csv_path, rows)

    font = plot(
        rows,
        output,
        args.gamma,
        pitches,
        args.critical_fraction,
        args.color_min,
    )
    script_path = Path(__file__).resolve()
    manifest = {
        "schema": "yap.d2d_bi2_bi6_size_thickness_heatmap.v1",
        "process": "D2D",
        "physics": "cu_inelastic",
        "bond_pitches_um": list(pitches),
        "gamma": args.gamma,
        "critical_fraction": args.critical_fraction,
        "source_cv": args.source_cv,
        "initial_bow_distribution": AnchorPreservingDistribution(
            reference_mean=D2D_INITIAL_BOW.reference_mean,
            reference_cv=D2D_INITIAL_BOW.reference_cv,
            target_cv=args.source_cv,
        ).audit(),
        "color_min_percent": args.color_min,
        "die_sizes_mm": list(DIE_SIZES_MM),
        "total_die_thicknesses_um": list(THICKNESSES_UM),
        "samples": args.samples,
        "quadrature_order": args.quadrature_order,
        "affine_rule": "E_res=(1-gamma)*E_mech",
        "font_requested": "Calibri Bold Italic",
        "font_rendered": f"{font} Bold Italic",
        "cache_reused": reused_cache,
        "runtime_s": perf_counter() - started,
        "script": str(script_path),
        "script_sha256": file_sha256(script_path),
    }
    (output / f"MANIFEST_{pitch_tag}_{critical_tag}_GAMMA_{tag}.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="ascii"
    )


if __name__ == "__main__":
    main()
