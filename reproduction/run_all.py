#!/usr/bin/env python3
"""Run the self-contained reproduction workflow for selected paper figures."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUPPORTED_FIGURES = ("13", "15", "16", "17", "19")
REQUIRED_SOURCE_FILES = (
    "W2W/overlay_yield_calculator.py",
    "D2W/overlay_yield_calculator.py",
)


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
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate checked-in outputs without regenerating data or plots.",
    )
    return parser.parse_args()


def check_environment(python_bin: str) -> None:
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "mpl-yap-environment-check"),
    )
    try:
        subprocess.run(
            [
                python_bin,
                "-c",
                "import cv2, matplotlib, numpy, omegaconf, scipy, sympy",
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise SystemExit(
            "Python environment check failed. Follow reproduction/README.md and "
            "install reproduction/requirements.txt."
        ) from error


def check_repository_layout() -> None:
    missing = [path for path in REQUIRED_SOURCE_FILES if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(
            "The reproduction directory must remain at the YAP repository root. "
            f"Missing required source files: {', '.join(missing)}"
        )


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")
    check_repository_layout()
    check_environment(args.python)

    for figure in args.figures:
        figure_dir = HERE / f"fig{figure}"
        run_script = figure_dir / "run.sh"
        environment = os.environ.copy()
        environment["PYTHON"] = args.python
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment.setdefault("MPLBACKEND", "Agg")
        environment.setdefault(
            "MPLCONFIGDIR",
            str(Path(tempfile.gettempdir()) / f"mpl-yap-fig{figure}"),
        )
        if args.verify_only:
            command = [args.python, str(figure_dir / "verify.py")]
            action = "Verifying"
        else:
            command = ["bash", str(run_script)]
            if figure != "13":
                command.extend(["--jobs", str(args.jobs)])
            action = "Reproducing"
        print(f"\n=== {action} Figure {figure} ===", flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)

    print("\nAll requested figure packages completed successfully.")


if __name__ == "__main__":
    main()
