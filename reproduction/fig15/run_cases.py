#!/usr/bin/env python3
"""Evaluate all configured historical parameter profiles with current yap+."""

from __future__ import annotations

import argparse
import json
import math
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
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument(
        "--report-only", action="store_true",
        help="Reuse existing per-case outputs and rebuild summary metadata.",
    )
    return parser.parse_args()


def run_one(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(
        command, cwd=ROOT, env=env, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )


def main() -> None:
    args = parse_args()
    experiment = OmegaConf.load(args.config).experiment
    figure, side = int(experiment.figure), str(experiment.side)
    selected = args.profile or list(experiment.profiles.keys())
    unknown = sorted(set(selected) - set(experiment.profiles.keys()))
    if unknown:
        raise ValueError(f"Unknown profiles: {unknown}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl-yap-fig15")

    jobs: list[tuple[str, int, Path, list[str]]] = []
    for profile_name in selected:
        profile = experiment.profiles[profile_name]
        profile_dir = RESULTS / profile_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        base = OmegaConf.to_container(profile.overrides, resolve=True)
        for index, case in enumerate(experiment.configurations, 1):
            side_um = math.sqrt(float(case.area_mm2)) * 1000.0
            overrides = {
                **base,
                "D0": float(case.density_cm2) / 1.0e8,
                "PITCH_um": float(case.pitch_um),
                "DIE_W_um": side_um,
                "DIE_L_um": side_um,
            }
            output = profile_dir / f"case_{index:02d}.json"
            case_name = f"fig{figure}_pkg_{profile_name}_c{index}"
            command = [
                sys.executable, str(REPRODUCTION / "model_worker.py"),
                "--side", side, "--case", case_name, "--layout", "full",
                "--overrides", json.dumps(overrides), "--output", str(output),
            ]
            jobs.append((profile_name, index, output, command))

    if not args.report_only:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
            futures = {executor.submit(run_one, command, env): (profile, index)
                       for profile, index, _, command in jobs}
            for future in as_completed(futures):
                profile, index = futures[future]
                future.result()
                print(f"finished {profile} case {index}/12")

    component_names = list(experiment.component_names)
    pass_component_names = list(experiment.pass_component_names)
    overlay_adjusted = bool(experiment.get("overlay_adjusted_diagnostics", False))
    tolerance = float(experiment.pass_abs_tolerance)
    profile_summaries = {}
    for profile_name in selected:
        profile = experiment.profiles[profile_name]
        cases = []
        residuals = []
        gating_residuals = []
        for index, reference_case in enumerate(experiment.configurations, 1):
            path = RESULTS / profile_name / f"case_{index:02d}.json"
            actual_data = json.loads(path.read_text())
            actual = [float(actual_data[name]) for name in component_names]
            reference = [float(value) for value in reference_case.paper]
            errors = [abs(left - right) for left, right in zip(actual, reference)]
            passes = [error <= tolerance for error in errors]
            residuals.extend(left - right for left, right in zip(actual, reference))
            actual_map = dict(zip(component_names, actual))
            reference_map = dict(zip(component_names, reference))
            error_map = dict(zip(component_names, errors))
            gating_residuals.extend(
                actual_map[name] - reference_map[name] for name in pass_component_names
            )
            case_pass = all(error_map[name] <= tolerance for name in pass_component_names)
            adjusted = None
            if overlay_adjusted:
                adjusted_bond = reference_map["Y_ovl"] * actual_map["Y_cr"] * actual_map["Y_df"]
                adjusted_system = adjusted_bond ** (1000.0 / float(reference_case.area_mm2))
                adjusted = {
                    "Y_bond_using_paper_overlay": adjusted_bond,
                    "Y_sys_using_paper_overlay": adjusted_system,
                    "Y_bond_abs_error": abs(adjusted_bond - reference_map["Y_bond"]),
                    "Y_sys_abs_error": abs(adjusted_system - reference_map["Y_sys_1000mm2"]),
                }
            actual_data.update({
                "figure": figure,
                "historical_parameter_profile": profile_name,
                "historical_parameter_source": str(profile.source),
                "paper_reference": reference_map,
                "absolute_errors": error_map,
                "component_pass": dict(zip(component_names, passes)),
                "pass_component_names": pass_component_names,
                "overlay_adjusted_diagnostics": adjusted,
                "raw_all_components_pass": all(passes),
                "case_pass": case_pass,
            })
            path.write_text(json.dumps(actual_data, indent=2) + "\n")
            cases.append({
                "index": index,
                "density_cm2": float(reference_case.density_cm2),
                "pitch_um": float(reference_case.pitch_um),
                "area_mm2": float(reference_case.area_mm2),
                "actual": actual_map,
                "reference": reference_map,
                "absolute_errors": error_map,
                "overlay_adjusted_diagnostics": adjusted,
                "raw_all_components_pass": all(passes),
                "case_pass": case_pass,
            })
        residuals_array = np.asarray(residuals)
        gating_residuals_array = np.asarray(gating_residuals)
        profile_summaries[profile_name] = {
            "source": str(profile.source),
            "eligible_for_pass": bool(profile.eligible_for_pass),
            "overrides": OmegaConf.to_container(profile.overrides, resolve=True),
            "rmse": float(np.sqrt(np.mean(residuals_array**2))),
            "max_abs_error": float(np.max(np.abs(residuals_array))),
            "gating_rmse": float(np.sqrt(np.mean(gating_residuals_array**2))),
            "gating_max_abs_error": float(np.max(np.abs(gating_residuals_array))),
            "passing_cases": sum(case["case_pass"] for case in cases),
            "all_cases_pass": all(case["case_pass"] for case in cases),
            "cases": cases,
        }

    eligible = {name: value for name, value in profile_summaries.items()
                if value["eligible_for_pass"]}
    best = min(eligible, key=lambda name: eligible[name]["gating_rmse"])
    passing_profiles = [name for name, value in eligible.items()
                        if value["all_cases_pass"]]
    configured_plot_profile = str(experiment.plot_profile)
    plot_profile = configured_plot_profile if configured_plot_profile in eligible else best
    summary = {
        "figure": figure,
        "side": side.upper(),
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "reference_source": str(experiment.reference_source),
        "component_names": component_names,
        "pass_component_names": pass_component_names,
        "ignored_components_for_pass": [
            name for name in component_names if name not in pass_component_names
        ],
        "pass_abs_tolerance": tolerance,
        "best_eligible_profile_by_gating_rmse": best,
        "plot_profile": plot_profile,
        "passing_profiles": passing_profiles,
        "reproduction_pass": bool(passing_profiles),
        "profiles": profile_summaries,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "profiles"}, indent=2))


if __name__ == "__main__":
    main()
