#!/usr/bin/env python3
"""Run every Fig. 19 W2W/D2W case with the current yap+ calculators."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
REPRODUCTION = HERE.parent
ROOT = REPRODUCTION.parent
RESULTS = HERE / "results"
sys.path.insert(0, str(REPRODUCTION))

from model_worker import calculator_source_sha256, repository_revision  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--report-only", action="store_true",
        help="rebuild metadata/comparisons from existing per-case current-code JSONs",
    )
    return parser.parse_args()


def run_one(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(
        command, cwd=ROOT, env=env, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )


def main() -> None:
    args = parse_args()
    exp = OmegaConf.load(args.config).experiment
    RESULTS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-yap-fig19"))

    jobs = []
    for side in exp.sides:
        for density in exp.densities:
            for configuration in exp.configurations:
                case_name = f"fig19_pkg_{side}_{density.tag}_{configuration.tag}"
                output = RESULTS / f"{side}_{density.tag}_{configuration.tag}.json"
                overrides = {
                    **OmegaConf.to_container(exp.common_overrides, resolve=True),
                    "D0": float(density.density_cm2) / 1.0e8,
                }
                command = [
                    sys.executable, str(REPRODUCTION / "model_worker.py"),
                    "--side", str(side), "--case", case_name,
                    "--layout", str(configuration.layout),
                    "--overrides", json.dumps(overrides),
                    "--defect-only", "--output", str(output),
                ]
                if configuration.replica_spacing_um is not None:
                    command.extend([
                        "--replica-spacing-um",
                        str(float(configuration.replica_spacing_um)),
                    ])
                jobs.append((str(side), str(density.tag), str(configuration.tag), command))

    if not args.report_only:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
            futures = {executor.submit(run_one, command, env): key
                       for *key, command in jobs}
            for future in as_completed(futures):
                side, density, configuration = futures[future]
                future.result()
                print(f"finished {side} {density} {configuration}")

    tolerance = float(exp.pass_abs_tolerance)
    sides = {}
    all_residuals = []
    all_case_pass = []
    for side in exp.sides:
        density_rows = []
        side_residuals = []
        for density in exp.densities:
            actual = []
            cases = []
            references = [float(value) for value in density.paper[str(side)]]
            for index, (configuration, reference) in enumerate(
                zip(exp.configurations, references), 1
            ):
                path = RESULTS / f"{side}_{density.tag}_{configuration.tag}.json"
                output = json.loads(path.read_text())
                value = float(output[str(exp.result_component)])
                error = abs(value - reference)
                case_pass = error <= tolerance
                output.update({
                    "figure": 19,
                    "paper_reference": reference,
                    "absolute_error": error,
                    "pass_abs_tolerance": tolerance,
                    "case_pass": case_pass,
                    "configuration_label": str(configuration.label),
                    "shared_main_to_replica_ratio": configuration.shared_ratio,
                    "current_code_result": True,
                })
                path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
                actual.append(value)
                side_residuals.append(value - reference)
                all_case_pass.append(case_pass)
                cases.append({
                    "index": index,
                    "tag": str(configuration.tag),
                    "label": str(configuration.label),
                    "actual": value,
                    "paper_reference": reference,
                    "absolute_error": error,
                    "case_pass": case_pass,
                    "output": path.name,
                })
            density_rows.append({
                "tag": str(density.tag),
                "density_cm2": float(density.density_cm2),
                "passing_cases": sum(case["case_pass"] for case in cases),
                "all_cases_pass": all(case["case_pass"] for case in cases),
                "cases": cases,
            })
        side_residuals_array = np.asarray(side_residuals)
        # For a fixed layout, current analytical Y_df must obey
        # Y(D=1) = Y(D=0.1)^10 because Lambda is linear in D.
        high = np.asarray([case["actual"] for case in density_rows[0]["cases"]])
        low = np.asarray([case["actual"] for case in density_rows[1]["cases"]])
        paper_high = np.asarray([case["paper_reference"] for case in density_rows[0]["cases"]])
        paper_low = np.asarray([case["paper_reference"] for case in density_rows[1]["cases"]])
        current_poisson_residual = high - low**10
        paper_poisson_residual = paper_high - paper_low**10
        sides[str(side)] = {
            "rmse": float(np.sqrt(np.mean(side_residuals_array**2))),
            "max_abs_error": float(np.max(np.abs(side_residuals_array))),
            "passing_cases": sum(row["passing_cases"] for row in density_rows),
            "total_cases": sum(len(row["cases"]) for row in density_rows),
            "all_cases_pass": all(row["all_cases_pass"] for row in density_rows),
            "densities": density_rows,
            "poisson_density_consistency": {
                "identity": "Y(D=1 cm^-2) = Y(D=0.1 cm^-2)^10",
                "current_max_abs_residual": float(np.max(np.abs(current_poisson_residual))),
                "paper_reference_max_abs_residual": float(np.max(np.abs(paper_poisson_residual))),
                "current_residuals": current_poisson_residual.tolist(),
                "paper_reference_residuals": paper_poisson_residual.tolist(),
            },
        }
        all_residuals.extend(side_residuals)

    all_residuals_array = np.asarray(all_residuals)
    summary = {
        "figure": 19,
        "repository_commit": repository_revision(),
        "calculator_source_sha256": calculator_source_sha256(),
        "reference_source": str(exp.reference_source),
        "result_component": str(exp.result_component),
        "pass_abs_tolerance": tolerance,
        "total_cases": len(all_case_pass),
        "passing_cases": sum(all_case_pass),
        "reproduction_pass": all(all_case_pass),
        "overall_rmse": float(np.sqrt(np.mean(all_residuals_array**2))),
        "overall_max_abs_error": float(np.max(np.abs(all_residuals_array))),
        "common_overrides": OmegaConf.to_container(exp.common_overrides, resolve=True),
        "current_code_profile_source": str(exp.current_code_profile.source),
        "historical_20to1_audit": OmegaConf.to_container(
            exp.historical_20to1_audit, resolve=True
        ),
        "uploaded_replica_distance_audit": OmegaConf.to_container(
            exp.uploaded_replica_distance_audit, resolve=True
        ),
        "sides": sides,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "sides"}, indent=2))


if __name__ == "__main__":
    main()
