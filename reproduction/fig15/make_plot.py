#!/usr/bin/env python3
"""Plot all twelve current-code cases for the best historical profile."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
COLORS = ["#0072B2", "#D55E00", "#E69F00", "#7B3294", "#66A61E"]


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    profile_name = summary["plot_profile"]
    profile = summary["profiles"][profile_name]
    names = summary["component_names"]
    cases = profile["cases"]
    actual = np.asarray([[case["actual"][name] for name in names] for case in cases])
    reference = np.asarray([[case["reference"][name] for name in names] for case in cases])
    errors = np.abs(actual - reference)
    labels = [f"{c['density_cm2']:g}, {c['pitch_um']:g}, {c['area_mm2']:g}" for c in cases]

    fig, (ax, error_ax) = plt.subplots(
        2, 1, figsize=(16, 8.2), gridspec_kw={"height_ratios": [3.2, 1.25]}
    )
    x = np.arange(12)
    width = 0.78 / len(names)
    for column, (name, color) in enumerate(zip(names, COLORS)):
        positions = x + (column - (len(names) - 1) / 2) * width
        ax.bar(positions, actual[:, column], width, label=f"current {name}", color=color)
        ax.scatter(
            positions, reference[:, column], marker="_", s=115,
            linewidths=1.8, color="black", label="paper reference" if column == 0 else None,
            zorder=4,
        )
    status = "PASS" if summary["reproduction_pass"] else "NOT PASS"
    ax.set_title(
        f"Fig. {summary['figure']} {summary['side']} — {profile_name} — {status}\n"
        f"gating RMSE={profile['gating_rmse']:.4g}, "
        f"gating max |error|={profile['gating_max_abs_error']:.4g}"
    )
    ax.set_ylabel("Yield")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.set_xlabel("particle density (cm$^{-2}$), pitch (µm), area (mm$^2$)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=len(names) + 1, loc="lower left")

    image = error_ax.imshow(errors.T, aspect="auto", cmap="magma_r", vmin=0)
    error_ax.set_yticks(np.arange(len(names)), names)
    error_ax.set_xticks(np.arange(12), np.arange(1, 13))
    error_ax.set_xlabel("Configuration index")
    error_ax.set_title(
        f"Absolute error; gating={summary['pass_component_names']}, "
        f"threshold={summary['pass_abs_tolerance']:.3f}"
    )
    fig.colorbar(image, ax=error_ax, label="absolute error", pad=0.01)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig{summary['figure']}_{summary['side'].lower()}_all.{suffix}", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
