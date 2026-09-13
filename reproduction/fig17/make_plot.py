#!/usr/bin/env python3
"""Plot current-code Fig. 17 and the historical-overlay audit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LAYOUTS = ["full", "sparse", "peripheral", "centralized"]
LABELS = ["Full", "Sparse", "Peripheral", "Centralized"]
COMPONENTS = ["Y_ovl", "Y_cr", "Y_df", "Y_bond"]
DISPLAY = [r"$Y_{ovl}$", r"$Y_{cr}$", r"$Y_{df}$", r"$Y_{bond}$"]


def plot_breakdown(cases: list[dict], stem: str, title_note: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), sharey=True)
    colors = ["#1976b9", "#e85d04", "#e9ad19", "#71308f"]
    x = np.arange(4)
    width = 0.19
    for axis, side in zip(axes, ("w2w", "d2w")):
        side_cases = {row["layout"]: row for row in cases if row["side"] == side}
        for index, (component, label, color) in enumerate(zip(COMPONENTS, DISPLAY, colors)):
            actual = [side_cases[layout]["actual"][component] for layout in LAYOUTS]
            paper = [side_cases[layout]["reference"][component] for layout in LAYOUTS]
            positions = x + (index - 1.5) * width
            axis.bar(positions, actual, width, color=color, label=label)
            axis.scatter(positions, paper, marker="_", s=150, linewidths=2.0,
                         color="black", zorder=4)
        axis.set_xticks(x, LABELS, rotation=12)
        axis.set_ylim(0.75, 1.00)
        axis.grid(axis="y", alpha=0.22)
        axis.set_title(side.upper())
    axes[0].set_ylabel("Yield")
    axes[1].legend(ncol=2, fontsize=9, loc="lower right")
    fig.suptitle(title_note)
    fig.text(0.5, 0.015, "Bars: evaluated yap+ / black ticks: paper Fig. 17c", ha="center")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    for suffix in ("png", "pdf"):
        fig.savefig(RESULTS / f"{stem}.{suffix}", dpi=220)
    plt.close(fig)


def main() -> None:
    current = json.loads((RESULTS / "current_summary.json").read_text())
    legacy = json.loads((RESULTS / "legacy_overlay_summary.json").read_text())
    # This profile demonstrates that the remaining non-overlay residual is
    # controlled by the paper's undisclosed logical main/copy ratio.  It is
    # explicitly labeled as inferred in config, JSON, and README.
    cases = current["profiles"]["inferred_logical_mapping"]["cases"]
    plot_breakdown(
        cases, "fig17_current_all",
        "Fig. 17 — current sample-wise overlay without historical D2W wafer scaling",
    )
    reproduction_cases = current["profiles"]["paper_era_wafer_scaled"]["cases"]
    plot_breakdown(
        reproduction_cases, "fig17_all",
        "Fig. 17 — paper-era reconstruction with explicit D2W wafer scaling",
    )

    august = legacy["profiles"]["august_2025_git_exact"]["cases"]
    august_map = {(row["side"], row["layout"]): row for row in august}
    current_map = {(row["side"], row["layout"]): row for row in cases}
    scaled_boundary = {
        row["layout"]: row
        for row in legacy["wafer_scaled_boundary_sensitivity"]["cases"]
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    x = np.arange(4)
    for axis, side in zip(axes, ("w2w", "d2w")):
        current_values = [
            current_map[(side, layout)]["actual"]["Y_ovl"] for layout in LAYOUTS
        ]
        paper_values = [august_map[(side, layout)]["paper_Y_ovl"] for layout in LAYOUTS]
        old_values = [august_map[(side, layout)]["actual_Y_ovl"] for layout in LAYOUTS]
        axis.plot(x, paper_values, "ko-", label="Paper")
        axis.plot(x, current_values, "s-", color="#1976b9", label="Current sample-wise (Table I)")
        axis.plot(x, old_values, "^-", color="#d94801", label="Git 331ac05 formula/config")
        if side == "d2w":
            axis.plot(
                x, [scaled_boundary[layout]["actual_Y_ovl"] for layout in LAYOUTS],
                "d--", color="#238b45",
                label="Sensitivity: May scale + Aug boundaries",
            )
        axis.set_xticks(x, LABELS, rotation=12)
        axis.set_ylim(0.88, 1.00)
        axis.grid(alpha=0.22)
        axis.set_title(side.upper())
    axes[0].set_ylabel(r"$Y_{ovl}$")
    axes[1].legend(fontsize=9, loc="lower right")
    fig.suptitle("Fig. 17 historical-overlay compatibility audit")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig17_legacy_overlay_audit.{suffix}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
