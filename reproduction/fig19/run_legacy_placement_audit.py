#!/usr/bin/env python3
"""Evaluate paper-era greedy placements with the current defect calculators."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
REPRODUCTION = HERE.parent
ROOT = REPRODUCTION.parent
RESULTS = HERE / "results" / "legacy_placement"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260120)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--missing-only", action="store_true")
    return parser.parse_args()


def run_one(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def main() -> None:
    args = parse_args()
    exp = OmegaConf.load(HERE / "config.yaml").experiment
    density = next(item for item in exp.densities if str(item.tag) == "d1")
    configurations = [item for item in exp.configurations if item.layout == "redundant"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl-yap-fig19-legacy")

    jobs = []
    profiles = []
    for side in exp.sides:
        profiles.extend(
            [
                (str(side), f"{side}_baseline10mm_r0p3", 10000.0, 0.3),
                (str(side), f"{side}_baseline10mm_r0p5", 10000.0, 0.5),
            ]
        )
    # These are diagnostics only: they are the round die dimensions that make
    # the current no-redundancy calculation approximately equal the paper bar.
    # They must not be presented as paper inputs.
    profiles.extend(
        [
            ("d2w", "d2w_baseline10mm_r0p005", 10000.0, 0.005),
            ("w2w", "fitted_no_redundancy_9p5mm_r0p3", 9500.0, 0.3),
            ("d2w", "fitted_no_redundancy_6mm_r0p3", 6000.0, 0.3),
        ]
    )
    for side, profile_tag, die_size_um, ratio in profiles:
        for configuration in configurations:
            case = f"fig19_legacy_{profile_tag}_{configuration.tag}"
            output = RESULTS / f"{profile_tag}_{configuration.tag}.json"
            overrides = {
                **OmegaConf.to_container(exp.common_overrides, resolve=True),
                "D0": float(density.density_cm2) / 1.0e8,
                "DIE_W_um": die_size_um,
                "DIE_L_um": die_size_um,
            }
            command = [
                sys.executable,
                str(REPRODUCTION / "model_worker.py"),
                "--side", str(side),
                "--case", case,
                "--layout", "redundant",
                "--replica-spacing-um", str(float(configuration.replica_spacing_um)),
                "--redundant-placement", "legacy-greedy",
                "--layout-seed", str(args.seed),
                "--redundant-logical-pad-ratio", str(ratio),
                "--overrides", json.dumps(overrides),
                "--defect-only",
                "--output", str(output),
            ]
            jobs.append(
                (side, profile_tag, ratio, die_size_um, str(configuration.tag), output, command)
            )

    runnable_jobs = [
        job for job in jobs if not (args.missing_only and job[-2].exists())
    ]
    if not args.report_only:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
            futures = {
                executor.submit(run_one, command, env): (side, profile_tag, ratio, tag)
                for side, profile_tag, ratio, _, tag, _, command in runnable_jobs
            }
            for future in as_completed(futures):
                side, profile_tag, ratio, tag = futures[future]
                future.result()
                print(f"finished {side} {profile_tag} ratio={ratio} {tag}")

    summary = {
        "figure": 19,
        "scope": "paper-era placement evaluated by current defect calculators",
        "die_size_um": [10000, 10000],
        "block_dim_um": 200,
        "density_cm2": float(density.density_cm2),
        "layout_seed": args.seed,
        "placement": "December-2025 randomized greedy algorithm including local/global-ID bug",
        "profiles": [],
    }
    for side, profile_tag, die_size_um, ratio in profiles:
        paper = [float(value) for value in density.paper[str(side)]][1:5]
        cases = []
        for configuration, reference in zip(configurations, paper):
            path = RESULTS / f"{profile_tag}_{configuration.tag}.json"
            result = json.loads(path.read_text())
            cases.append(
                {
                    "spacing_um": float(configuration.replica_spacing_um),
                    "yield": float(result["Y_df"]),
                    "paper_reference": reference,
                    "absolute_error": abs(float(result["Y_df"]) - reference),
                    "pair_count_fraction": float(result["paired_block_fraction"]),
                    "unique_paired_block_fraction": float(
                        result["unique_paired_block_fraction"]
                    ),
                    "output": str(path.relative_to(HERE)),
                }
            )
        summary["profiles"].append(
            {
                "tag": profile_tag,
                "side": side,
                "die_size_um": die_size_um,
                "die_size_status": (
                    "paper Table-I baseline"
                    if die_size_um == 10000.0
                    else "diagnostic fit to current no-redundancy yield; not a paper input"
                ),
                "redundant_logical_pad_ratio": ratio,
                "cases": cases,
            }
        )
    output = RESULTS / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(output), "profiles": len(summary["profiles"])}, indent=2))


if __name__ == "__main__":
    main()
