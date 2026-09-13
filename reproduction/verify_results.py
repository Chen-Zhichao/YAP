#!/usr/bin/env python3
"""Mechanical completeness, unit, identity, and trend checks."""

from __future__ import annotations

import json
import math
from pathlib import Path


RESULTS = Path(__file__).resolve().parent / "results"


def load(stem: str) -> dict:
    return json.loads((RESULTS / f"{stem}.json").read_text())


def close(left: float, right: float, tol: float = 1e-10) -> None:
    if not math.isclose(left, right, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{left} != {right}")


def main() -> None:
    count = 0
    for side in ("w2w", "d2w"):
        f13 = load(f"fig13_{side}"); count += 1
        assert f13["requested_points"] == 300
        assert f13["output_points"] == len(f13["points"]) == 300
        assert f13["points_in_yield_window"] == sum(
            .5 <= p["overall_combined_yield"] <= 1 for p in f13["points"]
        )

    for figure, side in ((15,"w2w"),(16,"d2w")):
        for index in range(1,13):
            item=load(f"fig{figure}_{side}_c{index}"); count += 1
            close(item["Y_bond"],item["Y_ovl"]*item["Y_cr"]*item["Y_df"])
            assert 0 <= item["Y_bond"] <= 1
            close(item["physical_critical_ratio"], 1.0)
            close(item["physical_redundant_ratio"], 0.0)
            close(item["configured_critical_pad_ratio"], 1.0)
            close(item["configured_redundant_pad_ratio"], 0.0)
            expected_dim = 400.0 if side == "w2w" else 100.0
            close(item["pad_block_dim_um"], expected_dim)
            assert item["pad_block_size_pads"] == int(expected_dim / item["pitch_um"])
            close(
                item["effective_pad_block_dim_um"],
                item["pad_block_size_pads"] * item["pitch_um"],
            )
            expected_rows = int(
                math.floor(math.sqrt(item["die_area_mm2"]) * 1000 / item["pitch_um"])
            ) // item["pad_block_size_pads"]
            assert item["pad_block_grid_rows"] == expected_rows
            assert item["pad_block_grid_cols"] == expected_rows
            if figure == 16:
                close(item["Y_sys_1000mm2"], item["Y_bond"] ** (1000/item["die_area_mm2"]))

    for side in ("w2w","d2w"):
        for layout in ("full","sparse","peripheral","centralized"):
            item=load(f"fig17_{side}_{layout}"); count += 1
            close(item["pitch_um"],.3)
            close(item["Y_bond"],item["Y_ovl"]*item["Y_cr"]*item["Y_df"])
            if layout != "full":
                close(item["physical_critical_ratio"],.2)
                close(item["physical_redundant_ratio"],.5)
                close(item["physical_dummy_ratio"],.3)

    for side in ("w2w","d2w"):
        for tag in ("d01","d1"):
            dedicated=[]
            for category in ("spacing0","spacing200","spacing400","spacing600","spacing800","shared20"):
                item=load(f"fig19_{side}_{tag}_{category}"); count += 1
                assert 0 <= item["Y_df"] <= 1
                if category.startswith("spacing") and category != "spacing0":
                    dedicated.append(item["Y_df"])
            if side == "w2w":
                assert dedicated == sorted(dedicated)

    comparison=load("paper_comparison")
    assert comparison["fig13_15_16_17_values_are_approximate_plot_readbacks"] is True
    assert comparison["fig19_values_are_exact_recovered_matlab_arrays"] is True
    for name in ("fig13_correlation.png","fig15_w2w_all.png","fig16_d2w_all.png","fig17_layouts.png","fig19_redundancy.png","RESULTS.md"):
        assert (RESULTS/name).stat().st_size > 0
    audit = load("fig15_16_version_audit")
    assert audit["overlay_solver_check"]["absolute_difference_um"] < 1e-10
    assert audit["verified_500k_sample_effective_fit_all_six_fine_pitch_cases"]["max_abs_error_vs_plot_readback"] < .005
    assert audit["particle_count_model_audit"]["latest_simulator"].startswith("independent Poisson")
    assert audit["critical_pad_and_grid_audit"]["paper_fig15_16"] == {
        "critical_ratio": 1.0,
        "redundant_ratio": 0.0,
        "w2w_grid_um": 400.0,
        "d2w_grid_um": 100.0,
    }
    for name in ("FIG15_16_VERSION_AUDIT.md", "fig16_right3_version_audit.png", "fig16_d2w_all_effective_fit.png"):
        assert (RESULTS/name).stat().st_size > 0
    for figure in (15, 16):
        package = json.loads(
            (RESULTS.parent / f"fig{figure}" / "results" / "summary.json").read_text()
        )
        if figure == 15:
            assert package["reproduction_pass"] is True
            assert all(profile["passing_cases"] == 12 for profile in package["profiles"].values())
        else:
            assert package["reproduction_pass"] is False
            assert package["ignored_components_for_pass"] == []
            assert max(profile["passing_cases"] for profile in package["profiles"].values()) == 8
        assert (RESULTS.parent / f"fig{figure}" / "results" / f"fig{figure}_{package['side'].lower()}_all.png").stat().st_size > 0
    fig17 = json.loads(
        (RESULTS.parent / "fig17" / "results" / "legacy_overlay_summary.json").read_text()
    )
    assert fig17["compatibility_composite"]["all_cases_pass"] is True
    assert all(profile["all_cases_pass"] for profile in fig17["profiles"].values())
    fig17_current = json.loads(
        (RESULTS.parent / "fig17" / "results" / "current_summary.json").read_text()
    )
    assert fig17_current["profiles"]["paper_era_wafer_scaled"]["all_cases_pass"] is True
    assert fig17_current["profiles"]["paper_era_wafer_scaled"]["passing_cases"] == 8
    fig19 = json.loads(
        (RESULTS.parent / "fig19" / "results" / "summary.json").read_text()
    )
    assert fig19["total_cases"] == 24
    assert fig19["passing_cases"] == 12
    assert fig19["reproduction_pass"] is False
    print(f"Validated completeness and invariants for {count} JSON result files.")


if __name__ == "__main__":
    main()
