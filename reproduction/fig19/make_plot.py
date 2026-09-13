#!/usr/bin/env python3
"""Plot all Fig. 19 cases and their recovered-paper-value errors."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex="col")
    colors = ("#7B3294", "#E69F00")
    density_labels = ("1 cm$^{-2}$", "0.1 cm$^{-2}$")
    labels = [case["label"] for case in summary["sides"]["w2w"]["densities"][0]["cases"]]
    x = np.arange(len(labels))
    width = 0.36

    for column, side in enumerate(("w2w", "d2w")):
        side_data = summary["sides"][side]
        for density_index, density in enumerate(side_data["densities"]):
            positions = x + (density_index - 0.5) * width
            actual = np.asarray([case["actual"] for case in density["cases"]])
            reference = np.asarray([case["paper_reference"] for case in density["cases"]])
            errors = np.abs(actual - reference)
            axes[0, column].bar(
                positions, actual, width, color=colors[density_index],
                label=f"current {density_labels[density_index]}",
            )
            axes[0, column].scatter(
                positions, reference, marker="_", s=170, linewidths=2.1,
                color="black",
                label="recovered paper value" if density_index == 0 else None,
                zorder=4,
            )
            axes[1, column].bar(
                positions, errors, width, color=colors[density_index],
                label=density_labels[density_index],
            )
        status = "PASS" if side_data["all_cases_pass"] else "NOT PASS"
        axes[0, column].set_title(
            f"{side.upper()} — {status} ({side_data['passing_cases']}/{side_data['total_cases']})\n"
            f"RMSE={side_data['rmse']:.4f}, max |error|={side_data['max_abs_error']:.4f}"
        )
        axes[0, column].set_ylim(0, 1.0)
        axes[0, column].grid(axis="y", alpha=0.25)
        axes[1, column].axhline(
            summary["pass_abs_tolerance"], color="#C44E52", linestyle="--",
            label=f"tolerance {summary['pass_abs_tolerance']:.2f}",
        )
        axes[1, column].set_xticks(x, labels)
        axes[1, column].set_xlabel("Main-replica spacing (µm)")
        axes[1, column].set_ylim(0, max(0.08, side_data["max_abs_error"] * 1.12))
        axes[1, column].grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Defect yield $Y_{df}$")
    axes[1, 0].set_ylabel("Absolute error")
    axes[0, 0].legend(ncol=3, fontsize=9, loc="upper left")
    axes[1, 0].legend(ncol=3, fontsize=9, loc="upper left")
    fig.suptitle("Fig. 19 — current yap+ results; black ticks are recovered exact paper values")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig19_redundancy_all.{suffix}", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
