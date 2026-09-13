#!/usr/bin/env python3
"""Verify the checked-in current-code Figure 13 experiment results."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE.parent))

from model_worker import assert_result_source_compatible  # noqa: E402


def verify_side(side: str) -> dict[str, float | int | str]:
    path = RESULTS / f"fig13_{side}.json"
    result = json.loads(path.read_text())
    points = result["points"]
    assert result["bonding_type"] == side.upper()
    assert result["requested_points"] == 300
    assert result["output_points"] == len(points) == 300
    assert result["yield_window"] == [0.5, 1.0]
    assert result["points_in_yield_window"] == sum(
        0.5 <= point["overall_combined_yield"] <= 1.0 for point in points
    )

    indices = np.asarray([point["index"] for point in points])
    rotation = np.asarray([point["rotation_mean_rad"] for point in points])
    dishing = np.asarray([point["cu_dishing_std_nm"] for point in points])
    density = np.asarray([point["particle_density_per_um2"] for point in points])
    np.testing.assert_array_equal(indices, np.arange(300))
    if side == "w2w":
        expected_rotation = np.linspace(1.45e-6, 1.70e-6, 300)
        expected_dishing = np.linspace(0.97, 1.105, 300)
        expected_density = np.logspace(-10.5, -8.8, 300)
        assert result["w2w_wafer_positions_per_model_point"] == 540
    else:
        expected_rotation = np.linspace(1.30e-5, 1.43e-5, 300)
        expected_dishing = np.linspace(1.0, 1.105, 300)
        expected_density = np.linspace(10.0**-10.5, 10.0**-8.7, 300)
    np.testing.assert_allclose(rotation, expected_rotation, rtol=0, atol=1e-20)
    np.testing.assert_allclose(dishing, expected_dishing, rtol=0, atol=1e-14)
    np.testing.assert_allclose(density, expected_density, rtol=0, atol=1e-22)

    residuals = []
    for point in points:
        components = point["simulated_component_yields"]
        product = components["overlay"] * components["cu_expansion"] * components["defect"]
        if not math.isclose(
            product, point["product_of_individual_yields"], rel_tol=1e-14, abs_tol=1e-14
        ):
            raise AssertionError(f"{side} point {point['index']}: component product mismatch")
        residuals.append(product - point["overall_combined_yield"])

    recomputed_mse = float(np.mean(np.square(residuals)))
    if not math.isclose(recomputed_mse, result["mse"], rel_tol=1e-14, abs_tol=1e-14):
        raise AssertionError(f"{side}: stored MSE does not match point data")
    return {
        "side": side.upper(),
        "points": len(points),
        "mse": recomputed_mse,
        "paper_mse": result["paper_mse"],
        "repository_commit": result["repository_commit"],
    }


def main() -> None:
    config = OmegaConf.load(HERE / "config.yaml").experiment
    assert int(config.points) == 300
    assert list(config.yield_window) == [0.5, 1.0]
    assert float(config.common.pitch_um) == 1.0
    assert float(config.common.die_w_um) == 10000.0
    assert float(config.common.die_l_um) == 10000.0
    assert float(config.common.pad_bottom_ratio) == 0.5
    assert math.isclose(float(config.common.pad_top_to_bottom_ratio), 2.0 / 3.0)
    assert float(config.common.t_0_um) == 0.1
    assert float(config.common.roughness_sigma_m) == 1.0e-9
    assert float(config.common.bow_difference_mean_um) == 0.5
    assert float(config.common.system_translation_x_mean_um) == 0.0
    assert float(config.common.system_translation_y_mean_um) == 0.0
    assert float(config.common.top_dishing_mean_nm) == -10.0
    assert float(config.common.bottom_dishing_mean_nm) == -10.0
    report = {side: verify_side(side) for side in ("w2w", "d2w")}
    for side, metrics in report.items():
        assert_result_source_compatible(str(metrics["repository_commit"]))
    for name in ("fig13_current_code.png", "fig13_current_code.pdf"):
        if not (RESULTS / name).is_file():
            raise FileNotFoundError(f"Missing {RESULTS / name}; run make_plot.py")
    print(json.dumps(report, indent=2))
    print(
        "PASS: legacy 300-point sweep vectors, current-code W2W/D2W yields, "
        "MSEs, commit, and plots verified."
    )


if __name__ == "__main__":
    main()
