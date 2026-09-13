#!/usr/bin/env python3
"""Run every requested Figure 13/15/16/17/19 case and build plots."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--skip-fig13", action="store_true")
    parser.add_argument("--skip-version-audit", action="store_true")
    return parser.parse_args()


def model_command(
    *, case: str, side: str, layout: str = "full", overrides: dict | None = None,
    spacing: int | None = None, defect_only: bool = False,
) -> tuple[str, list[str]]:
    command = [
        sys.executable, str(HERE / "model_worker.py"), "--side", side,
        "--case", case, "--layout", layout, "--output", str(RESULTS / f"{case}.json"),
    ]
    if overrides:
        command.extend(["--overrides", json.dumps(overrides)])
    if spacing is not None:
        command.extend(["--replica-spacing-um", str(spacing)])
    if defect_only:
        command.append("--defect-only")
    return case, command


def all_model_cases() -> list[tuple[str, list[str]]]:
    cases: list[tuple[str, list[str]]] = []
    configurations = [
        (0.1, 1.0, 10), (0.1, 1.0, 50), (0.1, 1.0, 100),
        (0.01, 1.0, 10), (0.01, 1.0, 50), (0.01, 1.0, 100),
        (0.1, 0.3, 10), (0.1, 0.3, 50), (0.1, 0.3, 100),
        (0.01, 0.3, 10), (0.01, 0.3, 50), (0.01, 0.3, 100),
    ]
    for index, (density, pitch, area) in enumerate(configurations, 1):
        side_um = math.sqrt(area) * 1000.0
        overrides = {
            "D0": density / 1.0e8, "PITCH_um": pitch,
            "DIE_W_um": side_um, "DIE_L_um": side_um,
        }
        cases.append(model_command(case=f"fig15_w2w_c{index}", side="w2w", overrides=overrides))
        cases.append(model_command(case=f"fig16_d2w_c{index}", side="d2w", overrides=overrides))

    effective_fit = {
        "PITCH_um": .3,
        "SYSTEM_TRANSLATION_X_STD_um": .02138207133974166,
        "SYSTEM_TRANSLATION_Y_STD_um": .02138207133974166,
        "SYSTEM_ROTATION_MEAN_rad": 3.0331570605484688e-6,
        "SYSTEM_ROTATION_STD_rad": 2.9831571076849285e-8,
        "num_samples": 500000,
    }
    for index, (density, area) in enumerate(
        ((0.1, 10), (0.1, 50), (0.1, 100),
         (0.01, 10), (0.01, 50), (0.01, 100)),
        7,
    ):
        side_um = math.sqrt(area) * 1000.0
        cases.append(model_command(
            case=f"fig16_d2w_c{index}_effective_fit",
            side="d2w",
            overrides={
                **effective_fit,
                "D0": density / 1.0e8,
                "DIE_W_um": side_um,
                "DIE_L_um": side_um,
            },
        ))

    for side in ("w2w", "d2w"):
        for layout in ("full", "sparse", "peripheral", "centralized"):
            cases.append(model_command(
                case=f"fig17_{side}_{layout}", side=side, layout=layout,
                overrides={"PITCH_um": 0.3},
            ))

    for side in ("w2w", "d2w"):
        for density_tag, density_um2 in (("d01", 1.0e-9), ("d1", 1.0e-8)):
            for spacing in (0, 200, 400, 600, 800):
                cases.append(model_command(
                    case=f"fig19_{side}_{density_tag}_spacing{spacing}", side=side,
                    layout="full" if spacing == 0 else "redundant",
                    overrides={"pad_block_dim_um": 200.0, "D0": density_um2},
                    spacing=None if spacing == 0 else spacing, defect_only=True,
                ))
            cases.append(model_command(
                case=f"fig19_{side}_{density_tag}_shared20", side=side,
                layout="shared20",
                overrides={"pad_block_dim_um": 200.0, "D0": density_um2},
                defect_only=True,
            ))
    return cases


def run(command: list[str], env: dict[str, str]) -> str:
    completed = subprocess.run(
        command, cwd=ROOT, env=env, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return completed.stdout


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl-yap-reproduce")
    RESULTS.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        if not args.skip_fig13:
            fig13_commands = [
                (f"fig13_{side}", [
                    sys.executable, str(HERE / "fig13" / "run_sweep.py"), "--side", side,
                    "--output", str(RESULTS / f"fig13_{side}.json"),
                ]) for side in ("w2w", "d2w")
            ]
            print("Running Figure 13 W2W and D2W sweeps...")
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {executor.submit(run, cmd, env): label for label, cmd in fig13_commands}
                for future in as_completed(futures):
                    label = futures[future]
                    future.result()
                    print(f"  finished {label}")

        cases = all_model_cases()
        print(f"Running {len(cases)} analytical cases with {args.jobs} workers...")
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
            futures = {executor.submit(run, command, env): label for label, command in cases}
            for future in as_completed(futures):
                label = futures[future]
                future.result()
                print(f"  finished {label}")

    print("Building plots and report...")
    run([sys.executable, str(HERE / "make_report.py")], env)
    if not args.skip_version_audit:
        print("Auditing Fig. 15/16 against pre-summer code and configuration...")
        run([sys.executable, str(HERE / "audit_fig15_fig16_history.py")], env)
    print(run([sys.executable, str(HERE / "verify_results.py")], env).strip())
    print(f"Done: {RESULTS / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
