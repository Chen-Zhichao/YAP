#!/usr/bin/env python3
"""Verify Fig. 17 coverage, provenance, equations, and audit status."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE.parent))

from model_worker import assert_result_source_compatible  # noqa: E402


def main() -> None:
    cfg = OmegaConf.load(HERE / "config.yaml").experiment
    current = json.loads((RESULTS / "current_summary.json").read_text())
    legacy = json.loads((RESULTS / "legacy_overlay_summary.json").read_text())
    assert_result_source_compatible(str(current["repository_commit"]))
    assert current["components"] == list(cfg.components)
    assert legacy["audit_only"] and legacy["not_used_for_current_results"]
    assert math.isclose(legacy["pass_abs_tolerance"], float(cfg.pass_abs_tolerance))

    for profile_name, profile in current["profiles"].items():
        assert len(profile["cases"]) == 8
        for row in profile["cases"]:
            path = RESULTS / profile_name / f"{row['side']}_{row['layout']}.json"
            data = json.loads(path.read_text())
            assert math.isclose(data["pitch_um"], 0.3)
            assert math.isclose(data["particle_density_per_cm2"], 0.1)
            assert math.isclose(data["physical_critical_ratio"], 1.0 if row["layout"] == "full" else 0.2)
            assert math.isclose(data["physical_redundant_ratio"], 0.0 if row["layout"] == "full" else 0.5)
            assert math.isclose(data["physical_dummy_ratio"], 0.0 if row["layout"] == "full" else 0.3)
            assert math.isclose(data["Y_bond"], data["Y_ovl"] * data["Y_cr"] * data["Y_df"], rel_tol=1e-12)

    reconstructed = current["profiles"]["paper_era_wafer_scaled"]
    assert reconstructed["all_cases_pass"]
    assert reconstructed["passing_cases"] == 8
    assert reconstructed["max_abs_error"] <= float(cfg.pass_abs_tolerance)
    for row in reconstructed["cases"]:
        if row["side"] == "d2w":
            output = json.loads(
                (RESULTS / "paper_era_wafer_scaled" / f"d2w_{row['layout']}.json").read_text()
            )
            assert output["scale_systematic_distortion_from_wafer"] is True
            assert output["wafer_to_die_distortion_scale"] > 1.0

    for legacy_name, profile in legacy["profiles"].items():
        assert len(profile["cases"]) == 8
        assert profile["all_cases_pass"]
        assert profile["passing_cases"] == 8
        subprocess.run(["git", "cat-file", "-e", f"{profile['source_commit']}^{{commit}}"],
                       cwd=ROOT, check=True)

    august = legacy["profiles"]["august_2025_git_exact"]["cases"]
    values = {(row["side"], row["layout"]): row for row in august}
    current_august = {
        (row["side"], row["layout"]): row
        for row in current["profiles"]["august_2025_modeling"]["cases"]
    }
    max_current_historical_delta = max(
        abs(current_august[key]["actual"]["Y_ovl"] - row["actual_Y_ovl"])
        for key, row in values.items()
    )
    assert max_current_historical_delta <= 1.0e-3
    actual_delta = values[("d2w", "centralized")]["actual_Y_ovl"] - values[("d2w", "full")]["actual_Y_ovl"]
    paper_delta = values[("d2w", "centralized")]["paper_Y_ovl"] - values[("d2w", "full")]["paper_Y_ovl"]
    assert abs(actual_delta - paper_delta) <= 0.005
    composite = legacy["compatibility_composite"]
    assert composite["all_cases_pass"]
    assert composite["passing_cases"] == 8
    assert "not recovered provenance" in composite["layout_mapping_profile"]
    scaled = legacy["wafer_scaled_boundary_sensitivity"]
    assert scaled["not_recovered_in_a_single_commit"]
    assert scaled["passing_cases"] == 4
    assert scaled["max_abs_error"] <= float(cfg.pass_abs_tolerance)
    current_scaled = {
        row["layout"]: row for row in reconstructed["cases"] if row["side"] == "d2w"
    }
    assert max(
        abs(current_scaled[row["layout"]]["actual"]["Y_ovl"] - row["actual_Y_ovl"])
        for row in scaled["cases"]
    ) <= 1.0e-3
    for stem in ("fig17_all", "fig17_current_all", "fig17_legacy_overlay_audit"):
        for suffix in ("png", "pdf"):
            assert (RESULTS / f"{stem}.{suffix}").stat().st_size > 0
    print(
        "PASS: paper-era wafer-scaled reconstruction passes 8/8 layouts; current and "
        f"August-2025 sample-wise overlay differ by at most {max_current_historical_delta:.3g}; "
        "both historical parameter audits and the labeled compatibility composite pass."
    )


if __name__ == "__main__":
    main()
