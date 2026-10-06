#!/usr/bin/env python3
"""Create the publication P3 TSV-patch blind warpage-map comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np


YAP_STACK_ROOT = Path(__file__).resolve().parents[4]
VALIDATION_ROOT = YAP_STACK_ROOT / "Ansys_Verify" / "12_stack_model_validation"
PREREG_ROOT = (
    VALIDATION_ROOT
    / "results"
    / "mindlin_p3_floorplan_transfer_blind_preregistration"
)
TARGET_ROOT = (
    VALIDATION_ROOT / "results" / "mindlin_rect_offset_tsv_p3_h20_c1_blind"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "paper_figures"
    / "P3_TSV_PATCH_WARPAGE_MAP_COMPARISON.png"
)

sys.path.insert(0, str(VALIDATION_ROOT))
from exp12_rectangular_bow_map import (  # noqa: E402
    RectangularBowMap,
    detrend_rigid_plane,
    period_low_pass,
)
from score_exp12_mindlin_p3_floorplan_blind import (  # noqa: E402
    prediction_map,
    target_map,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def publication_font() -> str:
    for family in ("Calibri", "Carlito"):
        try:
            font_manager.findfont(
                font_manager.FontProperties(
                    family=family, weight="bold", style="italic"
                ),
                fallback_to_default=False,
            )
            return family
        except ValueError:
            continue
    raise RuntimeError("Calibri or its metric-compatible Carlito font is required")


def configure_fonts() -> str:
    family = publication_font()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 17,
            "font.weight": "bold",
            "font.style": "italic",
            "axes.labelsize": 19,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.titlesize": 19,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 11,
            "mathtext.fontset": "custom",
            "mathtext.rm": f"{family}:style=italic:weight=bold",
            "mathtext.it": f"{family}:style=italic:weight=bold",
            "mathtext.bf": f"{family}:style=italic:weight=bold",
        }
    )
    return family


def load_maps() -> tuple[RectangularBowMap, RectangularBowMap, float, dict[str, float]]:
    preregistration = json.loads(
        (PREREG_ROOT / "BLIND_PREREGISTRATION.json").read_text(encoding="utf-8")
    )
    contract = json.loads(
        (PREREG_ROOT / "P3_SCORE_CONTRACT.json").read_text(encoding="utf-8")
    )
    freeze = json.loads(
        (PREREG_ROOT / "P3_PREDICTION_FREEZE.json").read_text(encoding="utf-8")
    )
    prediction_manifest = json.loads(
        Path(freeze["selected_prediction_manifest"]).read_text(encoding="utf-8")
    )
    model = prediction_map(Path(prediction_manifest["bow_map_file"]))
    mapdl = target_map(TARGET_ROOT / "explicit_3d_top_surface_coolback_maps.csv")
    if model.w_mm.shape != mapdl.w_mm.shape:
        raise RuntimeError("P3 model and MAPDL map shapes differ")
    coordinate_error = max(
        float(np.max(np.abs(model.x_mm - mapdl.x_mm))),
        float(np.max(np.abs(model.y_mm - mapdl.y_mm))),
    )
    if coordinate_error > 1.0e-12:
        raise RuntimeError("P3 coordinates exceed the audited canonicalization tolerance")
    mapdl = RectangularBowMap(model.x_mm, model.y_mm, mapdl.w_mm)
    floorplan = preregistration["case"]["tsv_floorplan"]
    patch = {
        "center_x_um": float(floorplan["center_offset_x_um"]),
        "center_y_um": float(floorplan["center_offset_y_um"]),
        "span_x_um": float(floorplan["span_x_um"]),
        "span_y_um": float(floorplan["span_y_um"]),
    }
    return model, mapdl, float(contract["surface"]["period_low_pass_mm"]), patch


def symmetric_ticks(limit: float, bins: int = 7) -> np.ndarray:
    locator = MaxNLocator(nbins=bins, symmetric=True)
    ticks = locator.tick_values(-limit, limit)
    return ticks[np.abs(ticks) <= limit * (1.0 + 1.0e-12)]


def add_floorplan(axis: plt.Axes, patch: dict[str, float], *, label: bool) -> None:
    x0 = patch["center_x_um"] - 0.5 * patch["span_x_um"]
    y0 = patch["center_y_um"] - 0.5 * patch["span_y_um"]
    axis.add_patch(
        Rectangle(
            (x0, y0),
            patch["span_x_um"],
            patch["span_y_um"],
            fill=False,
            edgecolor="black",
            linewidth=1.7,
            linestyle="--",
            label="TSV floorplan" if label else None,
        )
    )


def main() -> int:
    args = parse_args()
    model, mapdl, period_mm, patch = load_maps()
    model_clean = period_low_pass(detrend_rigid_plane(model), period_mm)
    mapdl_clean = period_low_pass(detrend_rigid_plane(mapdl), period_mm)
    model_um = 1000.0 * model_clean.w_mm
    mapdl_um = 1000.0 * mapdl_clean.w_mm
    difference_um = model_um - mapdl_um
    x_um = 1000.0 * model.x_mm
    y_um = 1000.0 * model.y_mm
    xx, yy = np.meshgrid(x_um, y_um, indexing="xy")
    tick_step_um = 75.0
    x_ticks = np.arange(
        np.ceil(x_um[0] / tick_step_um) * tick_step_um,
        np.floor(x_um[-1] / tick_step_um) * tick_step_um + 0.5 * tick_step_um,
        tick_step_um,
    )
    y_ticks = np.arange(
        np.ceil(y_um[0] / tick_step_um) * tick_step_um,
        np.floor(y_um[-1] / tick_step_um) * tick_step_um + 0.5 * tick_step_um,
        tick_step_um,
    )

    common_limit = float(max(np.max(np.abs(mapdl_um)), np.max(np.abs(model_um))))
    difference_limit = float(np.max(np.abs(difference_um)))
    common_norm = TwoSlopeNorm(vmin=-common_limit, vcenter=0.0, vmax=common_limit)
    difference_norm = TwoSlopeNorm(
        vmin=-difference_limit, vcenter=0.0, vmax=difference_limit
    )

    family = configure_fonts()
    figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.45), constrained_layout=True)
    maps = (mapdl_um, model_um, difference_um)
    titles = ("MAPDL", "Our model", "Model - MAPDL")
    norms = (common_norm, common_norm, difference_norm)
    images = []
    for index, (axis, values, title, norm) in enumerate(
        zip(axes, maps, titles, norms, strict=True)
    ):
        image = axis.pcolormesh(
            xx,
            yy,
            values,
            shading="auto",
            cmap="RdBu_r",
            norm=norm,
            rasterized=True,
        )
        images.append(image)
        add_floorplan(axis, patch, label=index == 0)
        axis.set_title(title, pad=4)
        axis.set_xlabel("x (µm)", labelpad=1)
        axis.set_xticks(x_ticks)
        axis.set_yticks(y_ticks)
        plt.setp(
            axis.get_xticklabels(),
            rotation=0,
            ha="center",
        )
        axis.set_aspect("equal")
        axis.tick_params(width=1.0, length=3, pad=1)
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)
    axes[0].set_ylabel("y (µm)", labelpad=1)
    axes[0].legend(loc="lower right", framealpha=0.9, borderpad=0.25, handlelength=1.8)

    common_bar = figure.colorbar(
        images[1],
        ax=list(axes[:2]),
        location="right",
        shrink=0.88,
        pad=0.025,
        aspect=24,
        ticks=symmetric_ticks(common_limit),
    )
    common_bar.set_label("Warpage (µm)", labelpad=2)
    error_bar = figure.colorbar(
        images[2],
        ax=axes[2],
        location="right",
        shrink=0.88,
        pad=0.025,
        aspect=24,
        ticks=symmetric_ticks(difference_limit),
    )
    error_bar.set_label("Difference (µm)", labelpad=2)
    for bar in (common_bar, error_bar):
        bar.ax.tick_params(labelsize=14, width=0.9, length=2, pad=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    font_path = font_manager.findfont(
        font_manager.FontProperties(family=family, weight="bold", style="italic"),
        fallback_to_default=False,
    )
    print(f"{args.output.resolve()}\nfont={family} Bold Italic\nfont_file={font_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
