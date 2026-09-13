#!/usr/bin/env python3
"""Run the self-contained reproduction workflow for selected paper figures."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUPPORTED_FIGURES = ("13", "15", "16", "17", "19")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate data, plots, and verification for YAP+ figures."
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=SUPPORTED_FIGURES,
        default=list(SUPPORTED_FIGURES),
        help="Figures to run (default: all).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Parallel case workers for Figures 15/16/17/19 (default: 4).",
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("PYTHON", sys.executable),
        help="Python interpreter used by each figure runner.",
    )
    return parser.parse_args()


def check_environment(python_bin: str) -> None:
    subprocess.run(
        [
            python_bin,
            "-c",
            "import cv2, matplotlib, numpy, omegaconf, scipy, sympy",
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")
    check_environment(args.python)

    for figure in args.figures:
        run_script = HERE / f"fig{figure}" / "run.sh"
        environment = os.environ.copy()
        environment["PYTHON"] = args.python
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment.setdefault("MPLCONFIGDIR", f"/tmp/mpl-yap-fig{figure}")
        command = ["bash", str(run_script)]
        if figure != "13":
            command.extend(["--jobs", str(args.jobs)])
        print(f"\n=== Reproducing Figure {figure} ===", flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)

    print("\nAll requested figure packages completed successfully.")


if __name__ == "__main__":
    main()
