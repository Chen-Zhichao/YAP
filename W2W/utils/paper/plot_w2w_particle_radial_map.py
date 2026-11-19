#!/usr/bin/env python3
"""Plot W2W 4-Hi particle- and overlay-yield maps on one color scale."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mechanics_runtime import configure_runtime_storage, validate_write_path

configure_runtime_storage()
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle
import numpy as np

from plot_w2w_spatial_compounding_comparison import (
    CURRENT_CONFIG,
    DEFAULT_INPUT,
    DIE_SIZE_MM,
    WAFER_RADIUS_MM,
    configure_style,
    load_current_yield_components,
    load_site_yields,
)


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "paper_figures"
    / "W2W_4HI_PARTICLE_OVERLAY_RADIAL_MAP.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def draw_map(
    axis: plt.Axes,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    values_pct: np.ndarray,
    title: str,
    norm: plt.Normalize,
) -> PatchCollection:
    patches = [
        Rectangle(
            (x - DIE_SIZE_MM / 2.0, y - DIE_SIZE_MM / 2.0),
            DIE_SIZE_MM,
            DIE_SIZE_MM,
        )
        for x, y in zip(x_mm, y_mm, strict=True)
    ]
    collection = PatchCollection(
        patches,
        cmap="viridis",
        norm=norm,
        linewidth=0.13,
        edgecolor=(1.0, 1.0, 1.0, 0.62),
        rasterized=True,
    )
    collection.set_array(values_pct)
    axis.add_collection(collection)
    axis.add_patch(
        Circle(
            (0.0, 0.0),
            WAFER_RADIUS_MM,
            fill=False,
            edgecolor="black",
            linewidth=1.1,
            zorder=4,
        )
    )
    axis.set_aspect("equal")
    axis.set_xlim(-154.0, 154.0)
    axis.set_ylim(-154.0, 154.0)
    axis.set_xticks([-150, -75, 0, 75, 150])
    axis.set_yticks([-150, -75, 0, 75, 150])
    axis.set_xlabel("Wafer x (mm)")
    axis.set_ylabel("Wafer y (mm)")
    axis.set_title(title, pad=7.0)
    axis.tick_params(direction="out", length=3.5)
    return collection


def write_site_csv(
    path: Path,
    data: dict[str, np.ndarray],
    particle_stack: np.ndarray,
    overlay_stack: np.ndarray,
) -> None:
    radius_mm = np.hypot(data["x_mm"], data["y_mm"])
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "site_id",
                "x_mm",
                "y_mm",
                "radius_mm",
                "stack_4hi_particle_yield",
                "stack_4hi_overlay_yield",
            ]
        )
        for row in zip(
            data["site_id"],
            data["x_mm"],
            data["y_mm"],
            radius_mm,
            particle_stack,
            overlay_stack,
            strict=True,
        ):
            writer.writerow(
                [int(row[0]), *(f"{float(value):.9f}" for value in row[1:])]
            )


def main() -> None:
    args = parse_args()
    args.output = validate_write_path(args.output)
    configure_style()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    site_data = load_site_yields(input_path)
    components = load_current_yield_components(site_data)
    particle = np.asarray(components["particle"], dtype=np.float64)
    particle_stack_map = np.prod(particle[:, :3], axis=1)
    overlay_stack_map = np.asarray(site_data["cumulative"][:, 2], dtype=np.float64)

    all_values_pct = 100.0 * np.concatenate((particle_stack_map, overlay_stack_map))
    lower = np.floor(all_values_pct.min())
    upper = np.ceil(all_values_pct.max())
    norm = plt.Normalize(lower, upper)

    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.75), constrained_layout=True)
    collection = draw_map(
        axes[0],
        site_data["x_mm"],
        site_data["y_mm"],
        100.0 * particle_stack_map,
        "4-Hi particle yield",
        norm,
    )
    draw_map(
        axes[1],
        site_data["x_mm"],
        site_data["y_mm"],
        100.0 * overlay_stack_map,
        "4-Hi overlay yield",
        norm,
    )
    colorbar = figure.colorbar(collection, ax=axes, pad=0.025, shrink=0.91)
    colorbar.set_label("Yield (%)")
    colorbar.ax.tick_params(direction="out", length=3.5)

    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    csv_path = output_path.with_suffix(".csv")
    write_site_csv(csv_path, site_data, particle_stack_map, overlay_stack_map)

    radius_mm = np.hypot(site_data["x_mm"], site_data["y_mm"])
    center = radius_mm <= 25.0
    edge = radius_mm >= 125.0
    print(f"Config: {CURRENT_CONFIG}")
    print(f"Figure: {output_path}")
    print(f"Site data: {csv_path}")
    print(
        "4-Hi particle: "
        f"mean={np.mean(particle_stack_map):.9f}, "
        f"center={np.mean(particle_stack_map[center]):.9f}, "
        f"edge={np.mean(particle_stack_map[edge]):.9f}"
    )
    print(
        "4-Hi overlay: "
        f"mean={np.mean(overlay_stack_map):.9f}, "
        f"center={np.mean(overlay_stack_map[center]):.9f}, "
        f"edge={np.mean(overlay_stack_map[edge]):.9f}"
    )


if __name__ == "__main__":
    main()
