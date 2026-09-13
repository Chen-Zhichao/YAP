#!/usr/bin/env python3
"""Verify Fig. 15/16 package structure, equations, coverage, and pass status."""

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
    config = OmegaConf.load(HERE / "config.yaml").experiment
    summary = json.loads((RESULTS / "summary.json").read_text())
    assert summary["figure"] == int(config.figure)
    assert summary["side"] == str(config.side).upper()
    assert len(config.configurations) == 12
    assert summary["reproduction_pass"] == bool(config.expected_reproduction_pass)
    assert summary["pass_component_names"] == list(config.pass_component_names)
    assert summary["plot_profile"] in summary["profiles"]
    if len(summary["profiles"]) == len(config.profiles):
        assert summary["plot_profile"] == str(config.plot_profile)
    assert summary["ignored_components_for_pass"] == []
    assert math.isclose(summary["pass_abs_tolerance"], float(config.pass_abs_tolerance))
    assert_result_source_compatible(str(summary["repository_commit"]))
    scaling = json.loads((RESULTS / "wafer_scaling_audit.json").read_text())
    assert scaling["source_commit"] == "6cc2408"
    table_i_scaling = scaling["profiles"]["paper_table_i"]
    assert table_i_scaling["radial_scale_overlay_rmse"] < table_i_scaling["current_overlay_rmse"]
    assert table_i_scaling["radial_scale_overlay_max_abs_error"] > float(config.pass_abs_tolerance)
    december_scaling = scaling["profiles"]["december_2025_modeling"]
    assert december_scaling["radial_scale_overlay_rmse"] > december_scaling["current_overlay_rmse"]
    eligible_pass = []
    for profile_name, profile in summary["profiles"].items():
        assert len(profile["cases"]) == 12
        case_status = []
        for case in profile["cases"]:
            output = json.loads(
                (RESULTS / profile_name / f"case_{case['index']:02d}.json").read_text()
            )
            assert output["historical_parameter_profile"] == profile_name
            assert math.isclose(output["physical_critical_ratio"], 1.0)
            assert math.isclose(output["physical_redundant_ratio"], 0.0)
            assert math.isclose(
                output["pad_block_dim_um"], float(profile["overrides"]["pad_block_dim_um"])
            )
            assert math.isclose(output["Y_bond"], output["Y_ovl"] * output["Y_cr"] * output["Y_df"], rel_tol=1e-12)
            assert math.isclose(
                output["Y_sys_1000mm2"],
                output["Y_bond"] ** (1000.0 / output["die_area_mm2"]),
                rel_tol=1e-12,
            )
            recomputed = {
                name: abs(output[name] - output["paper_reference"][name])
                for name in summary["component_names"]
            }
            for name, value in recomputed.items():
                assert math.isclose(value, output["absolute_errors"][name], rel_tol=1e-12, abs_tol=1e-12)
            adjusted = output["overlay_adjusted_diagnostics"]
            if bool(config.overlay_adjusted_diagnostics):
                expected_adjusted_bond = (
                    output["paper_reference"]["Y_ovl"] * output["Y_cr"] * output["Y_df"]
                )
                assert math.isclose(
                    adjusted["Y_bond_using_paper_overlay"], expected_adjusted_bond,
                    rel_tol=1e-12,
                )
                assert math.isclose(
                    adjusted["Y_sys_using_paper_overlay"],
                    expected_adjusted_bond ** (1000.0 / output["die_area_mm2"]),
                    rel_tol=1e-12,
                )
            else:
                assert adjusted is None
            case_status.append(all(
                recomputed[name] <= summary["pass_abs_tolerance"]
                for name in summary["pass_component_names"]
            ))
        assert profile["passing_cases"] == sum(case_status)
        assert profile["all_cases_pass"] == all(case_status)
        if profile["eligible_for_pass"] and profile["all_cases_pass"]:
            eligible_pass.append(profile_name)
    assert sorted(summary["passing_profiles"]) == sorted(eligible_pass)
    assert summary["reproduction_pass"] == bool(eligible_pass)
    for suffix in ("png", "pdf"):
        assert (RESULTS / f"fig{config.figure}_{config.side}_all.{suffix}").stat().st_size > 0
    print(
        f"PASS: verified all profiles x 12 cases; scientific reproduction status = "
        f"{summary['reproduction_pass']}."
    )


if __name__ == "__main__":
    main()
