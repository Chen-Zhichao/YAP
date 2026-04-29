#!/usr/bin/env python3
import argparse
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def set_plot_style():
    plt.rcParams.update(
        {
            "font.size": 26,
            "axes.titlesize": 26,
            "axes.labelsize": 26,
            "xtick.labelsize": 26,
            "ytick.labelsize": 26,
            "legend.fontsize": 26,
            "legend.frameon": True,
            "text.usetex": False,
            "hatch.linewidth": 1.6,
        }
    )


def parse_args():
    p = argparse.ArgumentParser(description="Plot ESD sensitivity results.")
    p.add_argument(
        "--input",
        default="output/esd_sensitivity_check/esd_sensitivity_design_1.tsv",
        help="TSV file with ESD sensitivity results.",
    )
    p.add_argument(
        "--output",
        default="output/esd_sensitivity_check/esd_sensitivity_design_1.png",
        help="Output PNG path.",
    )
    p.add_argument(
        "--ylim",
        default=None,
        help="Optional y-limits as 'min,max' in percent",
    )
    return p.parse_args()


def load_rows(path: str):
    rows = []
    with open(path, "r", newline="") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    for line in lines[1:]:
        parts = [part.strip() for part in line.split("\t")]
        parts = [part for part in parts if part != ""]
        if len(parts) < 5:
            continue

        head = parts[:4]
        tail = " ".join(parts[4:])
        tail_values = re.split(r"\s+", tail.strip())
        if len(tail_values) < 4:
            continue

        try:
            row = {
                "design": head[0],
                "ratio": head[1],
                "variant": head[2],
                "vmax_v": float(head[3]),
                "tilt_std_deg": float(tail_values[0]),
                "stack_assembly_yield": float(tail_values[1]),
                "compute_yield": float(tail_values[2]),
                "memory_yield": float(tail_values[3]),
            }
        except ValueError:
            continue
        rows.append(row)
    return rows


def plot(rows, out_path: str, ylim=None):
    if not rows:
        raise SystemExit("No valid rows found in TSV.")

    set_plot_style()

    all_max = 0.0
    for row in rows:
        all_max = max(
            all_max,
            row["compute_yield"] * 100.0,
            row["memory_yield"] * 100.0,
        )

    if ylim is None:
        y_min, y_max = 0.0, min(100.0, all_max * 1.05)
    else:
        y_min, y_max = ylim

    tilt_values = sorted({row["tilt_std_deg"] for row in rows})
    voltages = sorted({row["vmax_v"] for row in rows})
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    metric_specs = [("compute_yield", "Compute"), ("memory_yield", "Memory")]
    metric_colors = {
        "compute_yield": colors[0],
        "memory_yield": colors[1],
    }
    tilt_hatches = {
        tilt_values[i]: ("" if i == 0 else "//")
        for i in range(len(tilt_values))
    }

    fig, ax = plt.subplots(1, 1, figsize=(16, 6), dpi=150)
    variants = ["Center_IO", "Edge_IO"]
    groups = [(variant, voltage) for variant in variants for voltage in voltages]
    combos = [(metric, tilt) for metric, _ in metric_specs for tilt in tilt_values]
    x = np.arange(len(groups))
    bar_width = 0.8 / max(1, len(combos))

    grouped_all = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        grouped_all[row["variant"]][row["tilt_std_deg"]][row["vmax_v"]] = row

    for idx, (metric, tilt_std) in enumerate(combos):
        ys = []
        for variant, voltage in groups:
            row = grouped_all.get(variant, {}).get(tilt_std, {}).get(voltage)
            ys.append((row[metric] * 100.0) if row else 0.0)
        offsets = x + idx * bar_width
        metric_label = "Comp." if metric == "compute_yield" else "Mem."
        tilt_label = "low tilt" if tilt_std == min(tilt_values) else "high tilt"
        ax.bar(
            offsets,
            ys,
            width=bar_width,
            color=metric_colors[metric],
            hatch=tilt_hatches[tilt_std],
            label=f"{metric_label}, {tilt_label}",
            alpha=0.9,
            edgecolor="black",
            linewidth=0.6,
        )

    ax.set_xticks(x + (len(combos) - 1) * bar_width / 2)
    ax.set_xticklabels([f"{variant}, {int(voltage)}V" for variant, voltage in groups])
    ax.set_ylabel("ESD yield (%)")
    # ax.set_title("Design 1 ESD sensitivity")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_ylim(y_min, y_max)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        uniq_handles.append(handle)
        uniq_labels.append(label)
    ax.legend(
        uniq_handles,
        uniq_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
    )

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.96)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    rows = load_rows(args.input)
    ylim = None
    if args.ylim:
        try:
            parts = [p.strip() for p in args.ylim.split(",")]
            ylim = (float(parts[0]), float(parts[1]))
        except (ValueError, IndexError):
            raise SystemExit("--ylim must be 'min,max' in percent")
    plot(rows, args.output, ylim=ylim)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()
