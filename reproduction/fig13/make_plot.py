#!/usr/bin/env python3
"""Plot current-code W2W and D2W Figure 13 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_result(results_dir: Path, side: str) -> dict:
    return json.loads((results_dir / f"fig13_{side}.json").read_text())


def draw_axis(ax: plt.Axes, result: dict, side: str) -> None:
    points = result["points"]
    x = np.asarray([item["overall_combined_yield"] for item in points])
    y = np.asarray([item["product_of_individual_yields"] for item in points])
    ax.scatter(x, y, s=12, alpha=0.65, color="#5b65c8", edgecolors="none")
    ax.plot([0.5, 1.0], [0.5, 1.0], "--", color="#d99100", linewidth=1.2)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.5, 1.0)
    ax.set_xticks(np.arange(0.5, 1.01, 0.1))
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_xlabel("Overall combined yield from current-code experiment")
    ax.set_title(
        f"{side.upper()}: MSE={result['mse']:.3e}\n"
        f"paper={result['paper_mse']:.3e}",
        fontweight="bold",
    )
    ax.grid(which="major", alpha=0.25)
    ax.minorticks_on()
    ax.grid(which="minor", linestyle=":", alpha=0.12)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    results = {side: load_result(results_dir, side) for side in ("w2w", "d2w")}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, side in zip(axes, ("w2w", "d2w"), strict=True):
        draw_axis(ax, results[side], side)
    axes[0].set_ylabel("Product of individually simulated yields")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(results_dir / f"fig13_current_code.{suffix}", dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote {results_dir / 'fig13_current_code.png'}")


if __name__ == "__main__":
    main()
