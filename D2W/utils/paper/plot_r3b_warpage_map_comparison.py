#!/usr/bin/env python3
"""Create the publication R3B blind warpage-map comparison figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
import numpy as np


YAP_STACK_ROOT = Path(__file__).resolve().parents[4]
VALIDATION_ROOT = YAP_STACK_ROOT / "Ansys_Verify" / "12_stack_model_validation"
PREREG_ROOT = (
    VALIDATION_ROOT
    / "results"
    / "mindlin_r3b_dependency_closed_mesh_matched_rectangular_blind_preregistration"
)
CASE_ID = "R3B_DEPENDENCY_CLOSED_MESH_MATCHED_RECTANGULAR_380X460_H20_C1"
PREDICTION_ROOT = PREREG_ROOT / "predictions" / CASE_ID
TARGET_ROOT = (
    VALIDATION_ROOT
    / "results"
    / "mindlin_r3b_mesh_matched_rect_380x460_h20_n190x230_y2_y4_y10_c1_r160_h40_blind"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "paper_figures"
    / "R3B_WARPAGE_MAP_COMPARISON.png"
)

sys.path.insert(0, str(VALIDATION_ROOT))
from exp12_rectangular_bow_map import (  # noqa: E402
    RectangularBowMap,
    detrend_rigid_plane,
    period_low_pass,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def load_prediction() -> RectangularBowMap:
    path = PREDICTION_ROOT / "R3B_COOLBACK_BOW_MAP.npz"
    with np.load(path, allow_pickle=False) as archive:
        return RectangularBowMap(
            x_mm=archive["x_mm"],
            y_mm=archive["y_mm"],
            w_mm=archive["w_mm"],
        )


def load_target(coupon_x_mm: float, coupon_y_mm: float, nx: int, ny: int) -> RectangularBowMap:
    path = TARGET_ROOT / "explicit_3d_top_surface_coolback_maps.csv"
    values = np.empty((ny + 1, nx + 1), dtype=float)
    occupied = np.zeros_like(values, dtype=bool)
    dx = coupon_x_mm / nx
    dy = coupon_y_mm / ny
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["cycle"]) != 1:
                continue
            ix = int(np.rint(float(row["x_mm"]) / dx))
            iy = int(np.rint(float(row["z_mm"]) / dy))
            values[iy, ix] = float(row["uy_mm"])
            occupied[iy, ix] = True
    if not np.all(occupied):
        raise RuntimeError("R3B MAPDL target map is incomplete")
    return RectangularBowMap(
        x_mm=np.linspace(-0.5 * coupon_x_mm, 0.5 * coupon_x_mm, nx + 1),
        y_mm=np.linspace(-0.5 * coupon_y_mm, 0.5 * coupon_y_mm, ny + 1),
        w_mm=values,
    )


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
            "mathtext.fontset": "custom",
            "mathtext.rm": f"{family}:style=italic:weight=bold",
            "mathtext.it": f"{family}:style=italic:weight=bold",
            "mathtext.bf": f"{family}:style=italic:weight=bold",
        }
    )
    return family


def main() -> int:
    args = parse_args()
    contract = json.loads(
        (VALIDATION_ROOT / "R3B_MESH_MATCHED_SCORE_CONTRACT.json").read_text(
            encoding="utf-8"
        )
    )
    geometry = contract["geometry"]
    coupon_x_mm, coupon_y_mm = map(float, geometry["coupon_mm"])
    nx, ny = map(int, geometry["mesh"])
    period_mm = float(contract["surface"]["period_low_pass_mm"])

    model = load_prediction()
    mapdl = load_target(coupon_x_mm, coupon_y_mm, nx, ny)
    if not np.array_equal(model.x_mm, mapdl.x_mm) or not np.array_equal(
        model.y_mm, mapdl.y_mm
    ):
        raise RuntimeError("R3B model and MAPDL coordinates do not match")

    model_clean = period_low_pass(detrend_rigid_plane(model), period_mm)
    mapdl_clean = period_low_pass(detrend_rigid_plane(mapdl), period_mm)
    model_um = 1000.0 * model_clean.w_mm
    mapdl_um = 1000.0 * mapdl_clean.w_mm
    difference_um = model_um - mapdl_um
    x_um = 1000.0 * model.x_mm
    y_um = 1000.0 * model.y_mm
    xx, yy = np.meshgrid(x_um, y_um, indexing="xy")

    common_limit = float(max(np.max(np.abs(mapdl_um)), np.max(np.abs(model_um))))
    difference_limit = float(np.max(np.abs(difference_um)))
    common_norm = TwoSlopeNorm(vmin=-common_limit, vcenter=0.0, vmax=common_limit)
    difference_norm = TwoSlopeNorm(
        vmin=-difference_limit, vcenter=0.0, vmax=difference_limit
    )

    family = configure_fonts()
    figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.15), constrained_layout=True)
    maps = (mapdl_um, model_um, difference_um)
    titles = ("MAPDL", "YAP-Stack", "Model - MAPDL")
    norms = (common_norm, common_norm, difference_norm)
    images = []
    for axis, values, title, norm in zip(axes, maps, titles, norms, strict=True):
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
        axis.set_title(title, pad=4)
        axis.set_xlabel("x (µm)", labelpad=1)
        axis.set_aspect("equal")
        axis.tick_params(width=1.0, length=3, pad=1)
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)
    axes[0].set_ylabel("y (µm)", labelpad=1)

    common_bar = figure.colorbar(
        images[1],
        ax=list(axes[:2]),
        location="right",
        shrink=0.88,
        pad=0.025,
        aspect=24,
        ticks=np.arange(-0.75, 0.7501, 0.25),
    )
    common_bar.set_label("Warpage (µm)", labelpad=2)
    error_bar = figure.colorbar(
        images[2],
        ax=axes[2],
        location="right",
        shrink=0.88,
        pad=0.025,
        aspect=24,
        ticks=np.arange(-0.01, 0.0101, 0.005),
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
