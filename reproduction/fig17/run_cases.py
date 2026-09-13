#!/usr/bin/env python3
"""Run every Fig. 17 layout through the current yap+ calculators."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
REPRODUCTION = HERE.parent
ROOT = REPRODUCTION.parent
RESULTS = HERE / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument(
        "--report-only", action="store_true",
        help="Reuse existing per-case outputs and rebuild summary metadata.",
    )
    return parser.parse_args()


def run_one(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, cwd=ROOT, env=env, check=True, text=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def merged_overrides(experiment, profile_name: str, side: str) -> dict:
    profile = experiment.profiles[profile_name]
    return {
        **OmegaConf.to_container(profile.common, resolve=True),
        **OmegaConf.to_container(profile[side], resolve=True),
        "PITCH_um": float(experiment.pitch_um),
        "D0": float(experiment.density_cm2) / 1.0e8,
        "DIE_W_um": float(experiment.die_width_um),
        "DIE_L_um": float(experiment.die_length_um),
    }


def main() -> None:
    args = parse_args()
    experiment = OmegaConf.load(HERE / "config.yaml").experiment
    selected = args.profile or list(experiment.profiles.keys())
    unknown = sorted(set(selected) - set(experiment.profiles.keys()))
    if unknown:
        raise ValueError(f"Unknown profiles: {unknown}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl-yap-fig17")

    jobs = []
    for profile_name in selected:
        profile_dir = RESULTS / profile_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        for side in ("w2w", "d2w"):
            overrides = merged_overrides(experiment, profile_name, side)
            for layout in experiment.layouts:
                output = profile_dir / f"{side}_{layout}.json"
                case = f"fig17_pkg_{profile_name}_{side}_{layout}"
                command = [
                    sys.executable, str(REPRODUCTION / "model_worker.py"),
                    "--side", side, "--case", case, "--layout", str(layout),
                    "--overrides", json.dumps(overrides), "--output", str(output),
                ]
                jobs.append((profile_name, side, str(layout), command))

    if not args.report_only:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
            futures = {executor.submit(run_one, command, env): key
                       for *key, command in jobs}
            for future in as_completed(futures):
                future.result()
                print("finished", *futures[future])

    components = list(experiment.components)
    tolerance = float(experiment.pass_abs_tolerance)
    profiles = {}
    for profile_name in selected:
        rows = []
        residuals = []
        for side in ("w2w", "d2w"):
            for layout in experiment.layouts:
                path = RESULTS / profile_name / f"{side}_{layout}.json"
                data = json.loads(path.read_text())
                reference = dict(zip(components, map(float, experiment.paper[side][layout])))
                errors = {name: abs(float(data[name]) - reference[name]) for name in components}
                data.update({
                    "figure": 17,
                    "parameter_profile": profile_name,
                    "parameter_source": str(experiment.profiles[profile_name].source),
                    "paper_reference": reference,
                    "absolute_errors": errors,
                    "component_pass": {name: value <= tolerance for name, value in errors.items()},
                    "case_pass": all(value <= tolerance for value in errors.values()),
                })
                path.write_text(json.dumps(data, indent=2) + "\n")
                residuals.extend(float(data[name]) - reference[name] for name in components)
                rows.append({
                    "side": side, "layout": str(layout),
                    "actual": {name: float(data[name]) for name in components},
                    "reference": reference, "absolute_errors": errors,
                    "case_pass": data["case_pass"],
                })
        residuals = np.asarray(residuals)
        profiles[profile_name] = {
            "source": str(experiment.profiles[profile_name].source),
            "overrides": {side: merged_overrides(experiment, profile_name, side)
                          for side in ("w2w", "d2w")},
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "max_abs_error": float(np.max(np.abs(residuals))),
            "passing_cases": sum(row["case_pass"] for row in rows),
            "all_cases_pass": all(row["case_pass"] for row in rows),
            "cases": rows,
        }

    summary = {
        "figure": 17,
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "paper_reference_source": str(experiment.paper_reference_source),
        "components": components,
        "pass_abs_tolerance": tolerance,
        "profiles": profiles,
    }
    (RESULTS / "current_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({name: {key: value for key, value in profile.items()
                             if key not in ("cases", "overrides")}
                      for name, profile in profiles.items()}, indent=2))


if __name__ == "__main__":
    main()
