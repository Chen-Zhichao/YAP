#!/usr/bin/env python3
import argparse
import csv
import math
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot particle sensitivity results.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input TSV file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output PNG file.",
    )
    parser.add_argument(
        "--ylim",
        default=None,
        help="Optional y-limits as 'min,max' in percent.",
    )
    return parser.parse_args()


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if not row.get("variant") or not row.get("D1") or not row.get("edge_width_um"):
                continue
            if not row.get("stack_assembly_yield"):
                continue
            rows.append(
                {
                    "design": row["design"].strip(),
                    "ratio": row["ratio"].strip(),
                    "variant": row["variant"].strip(),
                    "D0": float(row["D0"]),
                    "D1": float(row["D1"]),
                    "edge_width_um": int(float(row["edge_width_um"])),
                    "stack_assembly_yield": float(row["stack_assembly_yield"]),
                }
            )
    return rows


def set_plot_style():
    plt.rcParams.update(
        {
            "font.size": 21,
            "axes.titlesize": 21,
            "axes.labelsize": 21,
            "xtick.labelsize": 21,
            "ytick.labelsize": 21,
            "legend.fontsize": 21,
            "figure.dpi": 150,
        }
    )


def format_cm2(value_um2):
    value_cm2 = value_um2 * 1e8
    return f"{value_cm2:.1f}"


def auto_ylim(rows):
    values = [row["stack_assembly_yield"] * 100.0 for row in rows]
    min_y = min(values)
    max_y = max(values)
    lower = max(0.0, math.floor((min_y - 3.0) / 5.0) * 5.0)
    upper = min(100.0, math.ceil((max_y + 1.0) / 5.0) * 5.0)
    if upper - lower < 10:
        lower = max(0.0, upper - 10.0)
    return lower, upper


def plot(rows, out_path, ylim=None):
    set_plot_style()

    variants_present = {row["variant"] for row in rows}
    variant_order = [v for v in ["Center_IO", "Edge_IO", "Random_IO"] if v in variants_present]
    d1_values = sorted({row["D1"] for row in rows})
    widths = sorted({row["edge_width_um"] for row in rows})

    if not variant_order:
        raise SystemExit("No variants found in TSV.")

    if ylim is None:
        y_min, y_max = auto_ylim(rows)
    else:
        y_min, y_max = ylim

    fig, axes = plt.subplots(1, len(variant_order), figsize=(7 * len(variant_order), 6), squeeze=False)
    axes = axes[0]

    d0_values = sorted({row["D0"] for row in rows})
    d0_label = format_cm2(d0_values[0]) if len(d0_values) == 1 else "varied"

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    width_to_color = {
        width: colors[idx % len(colors)]
        for idx, width in enumerate(widths)
    }

    for ax_idx, (ax, variant) in enumerate(zip(axes, variant_order)):
        variant_rows = [row for row in rows if row["variant"] == variant]
        x = np.arange(len(d1_values))
        bar_width = 0.8 / max(1, len(widths))

        for idx, width_um in enumerate(widths):
            ys = []
            for d1 in d1_values:
                matched = [
                    row for row in variant_rows
                    if row["D1"] == d1 and row["edge_width_um"] == width_um
                ]
                ys.append(matched[0]["stack_assembly_yield"] * 100.0 if matched else 0.0)
            ax.bar(
                x + idx * bar_width,
                ys,
                width=bar_width,
                label=f"width {width_um} um",
                color=width_to_color[width_um],
                alpha=0.9,
                edgecolor="black",
                linewidth=0.6,
            )

        ax.set_xticks(x + (len(widths) - 1) * bar_width / 2)
        ax.set_xticklabels([format_cm2(v) for v in d1_values])
        ax.set_title(f"{variant} ($D_0$ fixed at {d0_label} cm$^{{-2}}$)")
        ax.set_xlabel(r"$D_1$ (cm$^{-2}$)")
        ax.set_ylabel("Stack bonding yield (%)")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        if ax_idx == 0:
            ax.legend(loc="center right")
        else:
            ax.legend(loc="best")

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.88, wspace=0.22)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    rows = load_rows(args.input)
    if not rows:
        raise SystemExit("No valid rows found in TSV.")

    ylim = None
    if args.ylim:
        parts = [p.strip() for p in args.ylim.split(",")]
        if len(parts) != 2:
            raise SystemExit("--ylim must be 'min,max' in percent")
        ylim = (float(parts[0]), float(parts[1]))

    plot(rows, args.output, ylim=ylim)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()
