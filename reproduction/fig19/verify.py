#!/usr/bin/env python3
"""Verify Fig. 19 coverage, metadata, equations, and declared status."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE.parent))

from model_worker import assert_result_source_compatible  # noqa: E402


def main() -> None:
    exp = OmegaConf.load(HERE / "config.yaml").experiment
    summary = json.loads((RESULTS / "summary.json").read_text())
    assert summary["figure"] == 19
    assert summary["result_component"] == "Y_df"
    assert summary["total_cases"] == 24
    assert summary["reproduction_pass"] == bool(exp.expected_reproduction_pass)
    assert math.isclose(summary["pass_abs_tolerance"], float(exp.pass_abs_tolerance))
    assert_result_source_compatible(str(summary["repository_commit"]))
    audit_path = RESULTS / "historical_source_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        assert audit["archive_sha256"] == str(
            exp.uploaded_replica_distance_audit.archive_sha256
        )
        assert audit["exact_values_match_config"] is True
        assert audit["npy_distribution_count"] == 8
        assert audit["npy_distribution_scope"].startswith("W2W only")
        assert all(
            row["max_lambda_drift_across_density_columns"] < 1e-11
            for row in audit["npy_distributions"]
        )
        assert audit["exact_bar_poisson_residual"]["w2w"][
            "max_absolute_residual"
        ] < 0.003
        assert audit["exact_bar_poisson_residual"]["d2w"][
            "max_absolute_residual"
        ] < 0.0001
        assert audit["conversion_notebook_role"] == "NPY-to-MAT conversion only"
        assert audit["generation_script_recovered"] is False

    passed = []
    for side in exp.sides:
        side_summary = summary["sides"][str(side)]
        assert side_summary["total_cases"] == 12
        assert len(side_summary["densities"]) == 2
        high, low = side_summary["densities"]
        assert math.isclose(high["density_cm2"], 1.0)
        assert math.isclose(low["density_cm2"], 0.1)
        assert len(high["cases"]) == len(low["cases"]) == 6
        for density in side_summary["densities"]:
            for configured, case in zip(exp.configurations, density["cases"]):
                assert case["tag"] == str(configured.tag)
                output = json.loads((RESULTS / case["output"]).read_text())
                assert output["current_code_result"] is True
                assert output["side"] == str(side)
                assert output["layout"] == str(configured.layout)
                assert math.isclose(output["pitch_um"], 1.0)
                assert math.isclose(output["die_area_mm2"], 100.0)
                assert math.isclose(output["pad_block_dim_um"], 200.0)
                assert output["pad_block_grid_rows"] == 50
                assert output["pad_block_grid_cols"] == 50
                assert output["particle_yield_model"].startswith("Poisson exp(-Lambda)")
                expected_error = abs(output["Y_df"] - output["paper_reference"])
                assert math.isclose(expected_error, output["absolute_error"], abs_tol=1e-12)
                expected_pass = expected_error <= summary["pass_abs_tolerance"]
                assert output["case_pass"] == expected_pass == case["case_pass"]
                if configured.layout == "redundant":
                    assert math.isclose(
                        output["replica_spacing_um"],
                        float(configured.replica_spacing_um),
                    )
                    assert math.isclose(output["physical_redundant_ratio"], 1.0)
                    assert math.isclose(output["configured_redundant_logical_pad_ratio"], 0.5)
                elif configured.layout == "shared20":
                    assert output["shared_main_to_replica_ratio"] == 20
                    assert math.isclose(output["physical_redundant_ratio"], 1.0)
                else:
                    assert configured.tag == "spacing0"
                passed.append(expected_pass)

        current_residual = side_summary["poisson_density_consistency"][
            "current_max_abs_residual"
        ]
        assert current_residual < 1e-11

    assert summary["passing_cases"] == sum(passed)
    assert summary["reproduction_pass"] == all(passed)
    for suffix in ("png", "pdf"):
        assert (RESULTS / f"fig19_redundancy_all.{suffix}").stat().st_size > 0
    print(
        f"PASS: verified 24/24 current-code outputs and package invariants; "
        f"scientific reproduction status = {summary['reproduction_pass']} "
        f"({summary['passing_cases']}/24 bars within tolerance)."
    )


if __name__ == "__main__":
    main()
