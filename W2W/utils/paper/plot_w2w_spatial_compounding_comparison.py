#!/usr/bin/env python3
"""Plot full W2W stack yield under spatial and independent compounding."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
STACK_ROOT = REPO_ROOT.parent
DEFAULT_INPUT = "resource:w2w_affine/SITE_OVERLAY_YIELD.csv"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "W2W"
    / "output"
    / "paper_figures"
    / "W2W_SPATIAL_COMPOUNDING_COMPARISON.png"
)
CONFIGS_BY_MAX_HEIGHT = {
    4: (
        REPO_ROOT / "W2W" / "configs" / "design_6_4hi_p6" / "design_6_4hi_p6.yaml",
        REPO_ROOT / "W2W" / "input" / "design_6_4hi_p6",
        "design_6_4hi_p6",
    ),
    8: (
        REPO_ROOT / "W2W" / "configs" / "design_6_8hi_p6" / "design_6_8hi_p6.yaml",
        REPO_ROOT / "W2W" / "input" / "design_6_8hi_p6",
        "design_6_8hi_p6",
    ),
}

sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO_ROOT))
from mechanics_runtime import configure_runtime_storage, validate_write_path

configure_runtime_storage()

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle
import numpy as np


TARGET_TOTAL_UM = 50.0
WAFER_RADIUS_MM = 150.0
DIE_SIZE_MM = 10.0
GAMMA = 0.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument(
        "--comparison-input",
        type=Path,
        help="Optional site-yield table supplying one comparison bar group.",
    )
    parser.add_argument(
        "--comparison-gamma",
        type=float,
        help="Gamma associated with --comparison-input.",
    )
    parser.add_argument(
        "--comparison-height",
        type=int,
        default=4,
        help="Stack height extracted from --comparison-input (default: 4).",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--heights",
        type=int,
        nargs="+",
        help="Stack heights to plot; defaults to 2/4/8 or 2/3/4 from the input.",
    )
    parser.add_argument(
        "--map-height",
        type=int,
        help="Stack height shown in the two wafer maps; defaults to the tallest bar.",
    )
    args = parser.parse_args()
    args.output = validate_write_path(args.output)
    if str(args.input).startswith("resource:"):
        sys.path.insert(0, str(REPO_ROOT))
        from mechanics_runtime import resolve_resource
        args.input = resolve_resource(str(args.input))
    if not 0.0 <= args.gamma <= 1.0:
        parser.error("--gamma must lie in [0, 1]")
    if (args.comparison_input is None) != (args.comparison_gamma is None):
        parser.error("--comparison-input and --comparison-gamma must be used together")
    if args.comparison_gamma is not None and not 0.0 <= args.comparison_gamma <= 1.0:
        parser.error("--comparison-gamma must lie in [0, 1]")
    return args


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


def configure_style() -> str:
    family = publication_font()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 19,
            "font.weight": "bold",
            "font.style": "italic",
            "axes.labelsize": 21,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.titlesize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 16,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "mathtext.fontset": "custom",
            "mathtext.rm": f"{family}:style=italic:weight=bold",
            "mathtext.it": f"{family}:style=italic:weight=bold",
            "mathtext.bf": f"{family}:style=italic:weight=bold",
            "savefig.facecolor": "white",
        }
    )
    return family


def load_site_yields(path: Path) -> dict[str, np.ndarray]:
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="ascii")
    if "target_total_um" in table.dtype.names:
        selected = table[np.isclose(table["target_total_um"], TARGET_TOTAL_UM)]
    else:
        selected = table
    selected = np.atleast_1d(selected)
    selected.sort(order="site_id")
    if selected.size != 624:
        raise ValueError(f"Expected 624 die sites, found {selected.size}")

    interface_count = 0
    while f"A{interface_count + 1}_interface_overlay_yield" in table.dtype.names:
        interface_count += 1
    if interface_count not in (3, 7):
        raise ValueError(
            f"Expected either three or seven overlay interfaces, found {interface_count}"
        )
    cumulative_columns = []
    for index in range(1, interface_count + 1):
        candidates = (
            f"A{index}_cumulative_overlay_yield",
            f"A{index}_overlay_yield",
        )
        column = next((name for name in candidates if name in table.dtype.names), None)
        if column is None:
            raise ValueError(f"Missing cumulative overlay yield for A{index}")
        cumulative_columns.append(column)
    cumulative = np.column_stack(
        [selected[name] for name in cumulative_columns]
    ).astype(np.float64)
    if np.any((cumulative < 0.0) | (cumulative > 1.0)):
        raise ValueError("Cumulative site yields must lie in [0, 1]")

    interface_columns = tuple(
        f"A{index}_interface_overlay_yield"
        for index in range(1, interface_count + 1)
    )
    if all(name in table.dtype.names for name in interface_columns):
        interface = np.column_stack(
            [selected[name] for name in interface_columns]
        ).astype(np.float64)
    else:
        interface = np.empty_like(cumulative)
        interface[:, 0] = cumulative[:, 0]
        for column in range(1, cumulative.shape[1]):
            denominator = cumulative[:, column - 1]
            if np.any(denominator <= 0.0):
                raise ValueError(
                    "Cannot recover interface yield from a zero prefix yield; "
                    "provide explicit A1-A3 interface overlay columns"
                )
            interface[:, column] = cumulative[:, column] / denominator
    if not np.allclose(interface[:, 0], cumulative[:, 0], atol=2e-12):
        raise ValueError("A1 interface and cumulative overlay yields must agree")

    return {
        "site_id": selected["site_id"].astype(np.int64),
        "x_mm": selected["x_mm"].astype(np.float64),
        "y_mm": selected["y_mm"].astype(np.float64),
        "interface": interface,
        "cumulative": cumulative,
    }


def load_current_yield_components(
    site_data: dict[str, np.ndarray],
    config_path: Path,
    design_root: Path,
    design_name: str,
) -> dict[str, object]:
    """Evaluate non-overlay mechanisms with the current W2W calculators."""
    w2w_root = REPO_ROOT / "W2W"
    sys.path.insert(0, str(w2w_root))
    try:
        from omegaconf import OmegaConf
        from Cu_expansion_yield_calculator import stack_stress_yield_calculator
        from chiplet_yield import chiplet_yield_records
        from defect_yield_calculator import stack_defect_yield_calculator
        from utils.util import convert_3dblox_to_pad_bitmap, get_config_dict
        from wafer_die_stack_initialization import WaferStack

        skeleton = OmegaConf.load(config_path)["w2w_modeling"]
        cfg_dict = get_config_dict(
            cfg_folder=str(config_path.parent),
            cfg_skeleton=skeleton,
            ds_name=design_name,
            input_ds_dir=str(design_root),
            _3dbv_path=str(design_root / "generated_chiplet_definitions.3dbv"),
            _3dbx_path=str(design_root / "generated_stack_config.3dbx"),
            mode="w2w_modeling",
        )
        bitmap_collections = {}
        for interface_name, cfg in cfg_dict.items():
            cfg.plot_flag = False
            cfg.OUTPUT_DIR = str(w2w_root / "output") + os.sep
            bitmap_collections[interface_name] = convert_3dblox_to_pad_bitmap(
                cfg=cfg,
                _bmap_path=str(design_root / f"{cfg.INTERFACE}.bmap"),
                criticality_path=str(
                    design_root / f"{cfg.INTERFACE}_criticality.txt"
                ),
                pad_arrange_pattern=cfg.PAD_ARRANGE_PATTERN,
            )

        wafer_stack = WaferStack(
            cfg_dict=cfg_dict,
            pad_bitmap_collection_dict=bitmap_collections,
            mode="w2w_modeling",
        )
        stack_defect_yield_calculator(cfg_dict, wafer_stack)
        stack_stress_yield_calculator(cfg_dict, wafer_stack)

        interface_names = list(cfg_dict)
        required_interfaces = site_data["interface"].shape[1]
        if len(interface_names) < required_interfaces:
            raise ValueError(
                f"The comparison requires {required_interfaces} interfaces, "
                f"but the selected design provides {len(interface_names)}"
            )
        interface_names = interface_names[:required_interfaces]

        first_interface = wafer_stack.interfaces.interface_dict[interface_names[0]]
        model_xy_mm = np.asarray(
            [die.die_center for die in first_interface.die_list], dtype=np.float64
        ) / 1000.0
        model_index = {
            tuple(np.round(coords, 6)): index
            for index, coords in enumerate(model_xy_mm)
        }
        site_order = np.asarray(
            [
                model_index[tuple(np.round(coords, 6))]
                for coords in np.column_stack((site_data["x_mm"], site_data["y_mm"]))
            ],
            dtype=np.int64,
        )

        particle = np.column_stack(
            [
                wafer_stack.die_yield_list_per_interface_dict[name]["particle"][
                    site_order
                ]
                for name in interface_names
            ]
        )
        mechanical = np.column_stack(
            [
                wafer_stack.die_yield_list_per_interface_dict[name]["mechanical"][
                    site_order
                ]
                for name in interface_names
            ]
        )
        records = chiplet_yield_records(cfg_dict, str(design_root))
        required_dies = required_interfaces + 1
        if len(records) < required_dies:
            raise ValueError(
                f"The comparison requires {required_dies} die records, "
                f"but the selected design provides {len(records)}"
            )
        chip_yields = np.asarray(
            [record["yield"] for record in records[:required_dies]], dtype=np.float64
        )

        for name, values in (("particle", particle), ("mechanical", mechanical)):
            if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
                raise ValueError(f"Invalid {name} yield returned by W2W calculator")
        return {
            "particle": particle,
            "mechanical": mechanical,
            "chip_yields": chip_yields,
            "chip_records": records[:required_dies],
            "interface_names": interface_names,
        }
    finally:
        if sys.path and sys.path[0] == str(w2w_root):
            sys.path.pop(0)


def draw_wafer_map(
    axis: plt.Axes,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    values: np.ndarray,
    title: str,
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
        norm=plt.Normalize(0.0, 100.0),
        linewidth=0.13,
        edgecolor=(1.0, 1.0, 1.0, 0.62),
        rasterized=True,
    )
    collection.set_array(100.0 * values)
    axis.add_collection(collection)
    axis.add_patch(
        Circle(
            (0.0, 0.0),
            WAFER_RADIUS_MM,
            fill=False,
            edgecolor="black",
            linewidth=1.15,
            zorder=4,
        )
    )
    axis.set_aspect("equal")
    axis.set_xlim(-154.0, 154.0)
    axis.set_ylim(-154.0, 154.0)
    axis.set_title(title, pad=7.0)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return collection


def compounding_results(
    data: dict[str, np.ndarray],
    components: dict[str, object],
    heights: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], list[dict[str, float]]]:
    overlay = np.asarray(data["interface"], dtype=np.float64)
    particle = np.asarray(components["particle"], dtype=np.float64)
    mechanical = np.asarray(components["mechanical"], dtype=np.float64)
    chip_yields = np.asarray(components["chip_yields"], dtype=np.float64)
    if overlay.shape != particle.shape or overlay.shape != mechanical.shape:
        raise ValueError("Overlay, particle, and mechanical site maps must align")

    spatial = []
    independent = []
    stack_maps = {}
    rows = []
    for height in heights:
        interface_count = height - 1
        selected = slice(0, interface_count)
        chip_stack_yield = float(np.prod(chip_yields[:height]))

        overlay_map = np.asarray(data["cumulative"][:, interface_count - 1])
        particle_map = np.prod(particle[:, selected], axis=1)
        mechanical_map = np.prod(mechanical[:, selected], axis=1)
        bonding_map = overlay_map * particle_map * mechanical_map
        total_map = bonding_map * chip_stack_yield
        stack_maps[height] = total_map

        spatial_total = float(np.mean(total_map))
        independent_total = float(
            np.prod(
                np.mean(
                    overlay[:, selected]
                    * particle[:, selected]
                    * mechanical[:, selected],
                    axis=0,
                )
            )
            * chip_stack_yield
        )
        spatial.append(spatial_total)
        independent.append(independent_total)
        rows.append(
            {
                "stack_height": height,
                "bonded_interfaces": interface_count,
                "spatial_overlay_yield": float(np.mean(overlay_map)),
                "independent_overlay_yield": float(
                    np.prod(np.mean(overlay[:, selected], axis=0))
                ),
                "spatial_particle_yield": float(np.mean(particle_map)),
                "independent_particle_yield": float(
                    np.prod(np.mean(particle[:, selected], axis=0))
                ),
                "spatial_mechanical_yield": float(np.mean(mechanical_map)),
                "independent_mechanical_yield": float(
                    np.prod(np.mean(mechanical[:, selected], axis=0))
                ),
                "chip_stack_yield": chip_stack_yield,
                "spatial_total_yield": spatial_total,
                "independent_total_yield": independent_total,
            }
        )
    return np.asarray(spatial), np.asarray(independent), stack_maps, rows


def draw_bars(
    axis: plt.Axes,
    entries: list[dict[str, object]],
) -> None:
    x = np.arange(len(entries), dtype=np.float64)
    width = 0.34
    spatial_pct = 100.0 * np.asarray(
        [entry["spatial"] for entry in entries], dtype=np.float64
    )
    independent_pct = 100.0 * np.asarray(
        [entry["independent"] for entry in entries], dtype=np.float64
    )
    bars_spatial = axis.bar(
        x - width / 2.0,
        spatial_pct,
        width,
        color="#2f7d4a",
        edgecolor="white",
        linewidth=0.55,
        label=r"Spatial compounding $Y_{\mathrm{stack}}^{\mathrm{W2W}}$",
        zorder=3,
    )
    bars_independent = axis.bar(
        x + width / 2.0,
        independent_pct,
        width,
        color="#c44e52",
        edgecolor="white",
        linewidth=0.55,
        label=r"Independent product $Y_{\mathrm{ind}}$",
        zorder=3,
    )
    for bars, values, x_offset in (
        (bars_spatial, spatial_pct, -0.010),
        (bars_independent, independent_pct, 0.010),
    ):
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0 + x_offset,
                10.0,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=18,
                fontweight=800,
                fontstyle="italic",
                color="white",
                rotation=90,
                rotation_mode="anchor",
                clip_on=True,
            )
    height_counts = {
        entry["height"]: sum(
            candidate["height"] == entry["height"] for candidate in entries
        )
        for entry in entries
    }
    for index, entry in enumerate(entries):
        if height_counts[entry["height"]] > 1:
            axis.text(
                x[index],
                max(spatial_pct[index], independent_pct[index]) + 2.0,
                rf"$\gamma={entry['gamma']:.2f}$",
                ha="center",
                va="bottom",
                fontsize=17,
                fontweight="bold",
                fontstyle="italic",
                color="black",
                clip_on=True,
            )
    axis.set_xticks(x, [str(entry["label"]) for entry in entries])
    axis.set_ylabel("Overall stack yield (%)")
    axis.set_ylim(0.0, 100.0)
    axis.set_yticks(np.arange(0.0, 100.1, 20.0))
    axis.grid(axis="y", color="0.86", linewidth=0.75, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        loc="upper center",
        frameon=False,
        ncol=1,
        fontsize=18,
        handlelength=1.15,
        handletextpad=0.4,
        labelspacing=0.12,
        bbox_to_anchor=(0.5, 1.105),
        borderaxespad=0.0,
    )


def write_results(
    output: Path,
    input_path: Path,
    font_family: str,
    spatial: np.ndarray,
    independent: np.ndarray,
    rows: list[dict[str, float]],
    components: dict[str, object],
    gamma: float,
    config_path: Path,
    heights: tuple[int, ...],
    map_height: int,
    comparison: dict[str, object] | None,
) -> None:
    output_rows = []
    for row in rows:
        output_rows.append({"gamma": gamma, **row})
        if comparison is not None and row["stack_height"] == comparison["height"]:
            output_rows.append(
                {
                    "gamma": comparison["gamma"],
                    **comparison["row"],
                }
            )
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream)
        fieldnames = list(output_rows[0]) + ["spatial_minus_independent_pp"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            formatted = {
                key: value if isinstance(value, int) else f"{value:.9f}"
                for key, value in row.items()
            }
            formatted["spatial_minus_independent_pp"] = (
                f"{100.0 * (row['spatial_total_yield'] - row['independent_total_yield']):.6f}"
            )
            writer.writerow(formatted)

    manifest = {
        "figure": str(output),
        "input": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "target_total_um": TARGET_TOTAL_UM,
        "gamma": gamma,
        "comparison": (
            {
                "input": str(comparison["input"]),
                "input_sha256": hashlib.sha256(
                    Path(comparison["input"]).read_bytes()
                ).hexdigest(),
                "gamma": comparison["gamma"],
                "stack_height": comparison["height"],
            }
            if comparison is not None
            else None
        ),
        "wafer_radius_mm": WAFER_RADIUS_MM,
        "die_size_mm": DIE_SIZE_MM,
        "site_count": 624,
        "current_config": str(config_path),
        "current_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "active_yield_components": [
            "overlay (including warpage-induced overlay)",
            "particle",
            "Cu recess/mechanical",
            "chip fabrication",
        ],
        "disabled_by_current_policy": ["ESD", "standalone final-warpage yield"],
        "interfaces": list(components["interface_names"]),
        "chiplets": list(components["chip_records"]),
        "left_map": f"registered-site {map_height}-Hi full bonding yield",
        "middle_map": f"registered-site {map_height}-Hi total stack yield",
        "spatial_formula": "mean_j(prod_k(Y_bond,kj) * prod_l(Y_chip,l))",
        "independent_formula": "prod_k(mean_j(Y_bond,kj)) * prod_l(Y_chip,l)",
        "height_policy": (
            f"{', '.join(f'{height}-Hi' for height in heights)} use evaluated "
            f"A1-A{max(heights) - 1} maps without extrapolation"
        ),
        "font_requested": "Calibri Bold Italic",
        "font_rendered": f"{font_family} Bold Italic",
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="ascii"
    )


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    font_family = configure_style()
    data = load_site_yields(input_path)
    maximum_height = data["interface"].shape[1] + 1
    default_heights = (2, 4, 8) if maximum_height == 8 else (2, 3, 4)
    heights = tuple(args.heights) if args.heights else default_heights
    if (
        not heights
        or any(height < 2 or height > maximum_height for height in heights)
        or tuple(sorted(set(heights))) != heights
    ):
        raise ValueError(
            f"--heights must be unique, increasing, and within [2, {maximum_height}]"
        )
    map_height = args.map_height if args.map_height is not None else max(heights)
    if map_height not in heights:
        raise ValueError("--map-height must be one of the plotted --heights")
    config_path, design_root, design_name = CONFIGS_BY_MAX_HEIGHT[maximum_height]
    components = load_current_yield_components(
        data,
        config_path,
        design_root,
        design_name,
    )
    spatial, independent, stack_maps, rows = compounding_results(
        data, components, heights
    )

    comparison = None
    bar_entries = [
        {
            "height": height,
            "gamma": args.gamma,
            "spatial": spatial_value,
            "independent": independent_value,
            "label": f"{height}-Hi",
        }
        for height, spatial_value, independent_value in zip(
            heights, spatial, independent, strict=True
        )
    ]
    if args.comparison_input is not None:
        comparison_path = args.comparison_input.resolve()
        comparison_data = load_site_yields(comparison_path)
        if comparison_data["interface"].shape != data["interface"].shape:
            raise ValueError("Primary and comparison site-yield tables must align")
        comparison_height = args.comparison_height
        if comparison_height not in heights:
            raise ValueError("--comparison-height must be one of --heights")
        comparison_spatial, comparison_independent, _, comparison_rows = (
            compounding_results(
                comparison_data,
                components,
                (comparison_height,),
            )
        )
        comparison = {
            "input": comparison_path,
            "gamma": args.comparison_gamma,
            "height": comparison_height,
            "row": comparison_rows[0],
        }
        insert_at = heights.index(comparison_height) + 1
        bar_entries.insert(
            insert_at,
            {
                "height": comparison_height,
                "gamma": args.comparison_gamma,
                "spatial": comparison_spatial[0],
                "independent": comparison_independent[0],
                "label": f"{comparison_height}-Hi",
            },
        )

    figure = plt.figure(figsize=(11.6, 4.05), constrained_layout=False)
    outer_grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.78, 1.0),
        left=0.018,
        right=0.99,
        bottom=0.17,
        top=0.94,
        wspace=0.25,
    )
    map_grid = outer_grid[0, 0].subgridspec(
        1,
        4,
        width_ratios=(1.0, 1.0, 0.09, 0.05),
        wspace=0.015,
    )
    left_axis = figure.add_subplot(map_grid[0, 0])
    middle_axis = figure.add_subplot(map_grid[0, 1])
    color_axis = figure.add_subplot(map_grid[0, 3])
    bar_axis = figure.add_subplot(outer_grid[0, 1])

    display_height = map_height
    chip_stack_yield = float(
        np.prod(np.asarray(components["chip_yields"])[:display_height])
    )
    bonding_map = stack_maps[display_height] / chip_stack_yield

    draw_wafer_map(
        left_axis,
        data["x_mm"],
        data["y_mm"],
        bonding_map,
        f"{display_height}-Hi bonding yield",
    )
    image = draw_wafer_map(
        middle_axis,
        data["x_mm"],
        data["y_mm"],
        stack_maps[display_height],
        f"{display_height}-Hi stack yield",
    )
    colorbar = figure.colorbar(
        image,
        cax=color_axis,
        orientation="vertical",
        ticks=np.arange(0, 101, 20),
    )
    colorbar.ax.set_title("Yield (%)", pad=8.0, fontsize=20)
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.tick_params(labelsize=17.5, width=0.9, length=3.0)
    draw_bars(bar_axis, bar_entries)

    figure.savefig(output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    write_results(
        output,
        input_path,
        font_family,
        spatial,
        independent,
        rows,
        components,
        args.gamma,
        config_path,
        heights,
        map_height,
        comparison,
    )

    print(f"Saved: {output}")
    print(f"Font: {font_family} Bold Italic")
    for height, spatial_value, independent_value in zip(
        heights, spatial, independent, strict=True
    ):
        print(
            f"{height}-Hi: spatial={spatial_value:.6f}, "
            f"independent={independent_value:.6f}, "
            f"gap={100.0 * (spatial_value - independent_value):.6f} pp"
        )
    if comparison is not None:
        row = comparison["row"]
        print(
            f"{comparison['height']}-Hi at gamma={comparison['gamma']:.2f}: "
            f"spatial={row['spatial_total_yield']:.6f}, "
            f"independent={row['independent_total_yield']:.6f}, "
            "gap="
            f"{100.0 * (row['spatial_total_yield'] - row['independent_total_yield']):.6f} pp"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
