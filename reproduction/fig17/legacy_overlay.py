#!/usr/bin/env python3
"""Re-evaluate Fig. 17 with traceable historical overlay reductions.

This is a compatibility audit, not the implementation used for current-code
results.  The equations below are a small, documented re-expression of the
historical Git revisions named in config.yaml.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.optimize import brentq
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
SEED = 20260120


def circle_overlap(distance: float, radius1: float, radius2: float) -> float:
    if distance >= radius1 + radius2:
        return 0.0
    if distance <= abs(radius1 - radius2):
        return math.pi * min(radius1, radius2) ** 2
    c1 = np.clip((distance**2 + radius1**2 - radius2**2) /
                 (2 * distance * radius1), -1.0, 1.0)
    c2 = np.clip((distance**2 + radius2**2 - radius1**2) /
                 (2 * distance * radius2), -1.0, 1.0)
    radical = ((-distance + radius1 + radius2) *
               (distance + radius1 - radius2) *
               (distance - radius1 + radius2) *
               (distance + radius1 + radius2))
    return (radius1**2 * math.acos(c1) + radius2**2 * math.acos(c2)
            - 0.5 * math.sqrt(max(radical, 0.0)))


def allowed_misalignment(pitch: float, bottom_radius: float,
                         top_radius: float) -> float:
    target = 0.5 * math.pi * top_radius**2
    contact = brentq(
        lambda distance: circle_overlap(distance, top_radius, bottom_radius) - target,
        abs(bottom_radius - top_radius), bottom_radius + top_radius,
    )
    critical_distance = 0.5 * pitch - top_radius
    return min(contact, critical_distance)


def layout_ids(height: int, width: int, layout: str, count: int) -> np.ndarray:
    ids = np.arange(height * width)
    rows, cols = ids // width, ids % width
    rings = np.minimum.reduce([rows, cols, height - 1 - rows, width - 1 - cols])
    if layout == "peripheral":
        return np.argsort(rings)[:count]
    if layout == "centralized":
        return np.argsort(-rings)[:count]
    if layout == "sparse":
        stride = 3
        while stride > 1:
            mesh = ids[(rows % stride == 0) & (cols % stride == 0)]
            if len(mesh) >= count:
                break
            stride -= 1
        distance = (np.abs(rows[mesh] - (height - 1) / 2) +
                    np.abs(cols[mesh] - (width - 1) / 2))
        ordered = mesh[np.argsort(distance)]
        return ordered[np.linspace(0, len(ordered) - 1, count, dtype=int)]
    raise ValueError(layout)


def relative_corners(layout: str, block_um: float, grid_size: int,
                     die_width: float, die_length: float,
                     use_critical_boundaries: bool) -> np.ndarray:
    full = np.asarray([
        [-die_width / 2, die_length / 2], [die_width / 2, die_length / 2],
        [-die_width / 2, -die_length / 2], [die_width / 2, -die_length / 2],
    ])
    if layout == "full" or not use_critical_boundaries:
        return full
    ids = layout_ids(grid_size, grid_size, layout,
                     int(math.ceil(0.2 * grid_size * grid_size)))
    rows, cols = ids // grid_size, ids % grid_size
    xmin = -die_width / 2 + cols.min() * block_um
    xmax = min(die_width / 2, -die_width / 2 + (cols.max() + 1) * block_um)
    ymax = die_length / 2 - rows.min() * block_um
    ymin = max(-die_length / 2, die_length / 2 - (rows.max() + 1) * block_um)
    return np.asarray([[xmin, ymax], [xmax, ymax], [xmin, ymin], [xmax, ymin]])


def wafer_die_centers(radius: float, die_width: float, die_length: float,
                      dice_width: float = 1000.0) -> list[np.ndarray]:
    nr = int(2 * radius // (die_length + dice_width) + 1)
    nc = int(2 * radius // (die_width + dice_width) + 1)
    centers = []
    for row in range(nr):
        for col in range(nc):
            cx = -nc * (die_width + dice_width) / 2 + (die_width + dice_width) / 2 + col * (die_width + dice_width)
            cy = nr * (die_length + dice_width) / 2 - (die_length + dice_width) / 2 - row * (die_length + dice_width)
            vertices = ((cx - die_width / 2, cy + die_length / 2),
                        (cx + die_width / 2, cy + die_length / 2),
                        (cx - die_width / 2, cy - die_length / 2),
                        (cx + die_width / 2, cy - die_length / 2))
            if all(math.hypot(x, y) < radius for x, y in vertices):
                centers.append(np.asarray([cx, cy]))
    return centers


def profile_parameters(experiment, profile_name: str, side: str) -> dict:
    profile = experiment.profiles[profile_name]
    return {**OmegaConf.to_container(profile.common, resolve=True),
            **OmegaConf.to_container(profile[side], resolve=True)}


def historical_yield(*, side: str, layout: str, params: dict,
                     radial_scale: bool, use_critical_boundaries: bool,
                     experiment) -> float:
    pitch = float(experiment.pitch_um)
    die_width, die_length = float(experiment.die_width_um), float(experiment.die_length_um)
    bottom = pitch / 2 * float(params["PAD_BOT_R_um_ratio"])
    top = bottom * float(params["PAD_TOP_R_um_ratio"])
    limit = allowed_misalignment(pitch, bottom, top)
    count = int(params["num_samples"])
    rng = np.random.RandomState(SEED)
    tx = rng.normal(float(params["SYSTEM_TRANSLATION_X_MEAN_um"]),
                    float(params["SYSTEM_TRANSLATION_X_STD_um"]), count)
    ty = rng.normal(float(params["SYSTEM_TRANSLATION_Y_MEAN_um"]),
                    float(params["SYSTEM_TRANSLATION_Y_STD_um"]), count)
    rotation = rng.normal(float(params["SYSTEM_ROTATION_MEAN_rad"]),
                          float(params["SYSTEM_ROTATION_STD_rad"]), count)
    mag_mean = (float(params["k_mag"]) * float(params["BOW_DIFFERENCE_MEAN_um"])
                + float(params["M_0"])) / 1e6
    mag_std = (float(params["k_mag"]) * float(params["BOW_DIFFERENCE_STD_um"])) ** 2 / 1e6
    magnification = rng.normal(mag_mean, mag_std, count)
    if side == "d2w" and radial_scale:
        scale = 150000.0 / math.hypot(die_width / 2, die_length / 2)
        rotation *= scale
        magnification *= scale

    block_um = float(params["pad_block_dim_um"])
    grid_size = int((die_width // pitch) // int(block_um / pitch))
    relative = relative_corners(layout, block_um, grid_size, die_width, die_length,
                                use_critical_boundaries)
    centers = [np.zeros(2)] if side == "d2w" else wafer_die_centers(
        150000.0, die_width, die_length)
    yields = []
    for center in centers:
        corners = relative + center
        misalignments = []
        for x, y in corners:
            dx = tx - rotation * y + magnification * x
            dy = ty + rotation * x + magnification * y
            misalignments.append(np.hypot(dx, dy))
        # Correct die-level order: reduce the systematic magnitude across
        # corners within each sample, then evaluate the scalar random-error
        # CDF and average the resulting sample yields.
        worst = np.max(np.asarray(misalignments), axis=0)
        value = norm.cdf(limit - worst,
                         loc=float(params["RANDOM_MISALIGNMENT_MEAN_um"]),
                         scale=float(params["RANDOM_MISALIGNMENT_STD_um"]))
        value -= norm.cdf(-limit - worst,
                          loc=float(params["RANDOM_MISALIGNMENT_MEAN_um"]),
                          scale=float(params["RANDOM_MISALIGNMENT_STD_um"]))
        yields.append(float(np.mean(value)))
    return float(np.mean(yields))


def source_assertions(commit: str, side: str, radial_scale: bool,
                      use_boundaries: bool) -> None:
    source = subprocess.check_output(
        ["git", "show", f"{commit}:{side.upper()}/overlay_yield_calculator.py"],
        cwd=ROOT, text=True,
    )
    assert ".max(axis=0)" in source
    has_scale = "* WAF_R / np.sqrt((DIE_W/2)**2 + (DIE_L/2)**2)" in source
    has_boundaries = "ovl_critical_pad_boundary_coords" in source
    if side == "d2w":
        assert has_scale == radial_scale
    assert has_boundaries == use_boundaries


def main() -> None:
    experiment = OmegaConf.load(HERE / "config.yaml").experiment
    components = list(experiment.components)
    tolerance = float(experiment.pass_abs_tolerance)
    profiles = {}
    for legacy_name, legacy in experiment.legacy_overlay_profiles.items():
        rows = []
        for side in ("w2w", "d2w"):
            source_assertions(str(legacy.source_commit), side,
                              bool(legacy.d2w_radial_scale),
                              bool(legacy.use_critical_boundaries))
            params = profile_parameters(experiment, str(legacy.parameter_profile), side)
            for layout in experiment.layouts:
                actual = historical_yield(
                    side=side, layout=str(layout), params=params,
                    radial_scale=bool(legacy.d2w_radial_scale),
                    use_critical_boundaries=bool(legacy.use_critical_boundaries),
                    experiment=experiment,
                )
                reference = float(experiment.paper[side][layout][0])
                rows.append({
                    "side": side, "layout": str(layout), "actual_Y_ovl": actual,
                    "paper_Y_ovl": reference, "absolute_error": abs(actual - reference),
                    "pass": abs(actual - reference) <= tolerance,
                })
        profiles[legacy_name] = {
            "source_commit": str(legacy.source_commit),
            "description": str(legacy.description),
            "parameter_profile": str(legacy.parameter_profile),
            "d2w_radial_scale": bool(legacy.d2w_radial_scale),
            "use_critical_boundaries": bool(legacy.use_critical_boundaries),
            "passing_cases": sum(row["pass"] for row in rows),
            "all_cases_pass": all(row["pass"] for row in rows),
            "max_abs_error": max(row["absolute_error"] for row in rows),
            "cases": rows,
        }
    # Full paper-era reconstruction: preserve current non-overlay calculators,
    # use the traceable August sample-wise overlay, and
    # use the separately labeled inferred logical-mapping sensitivity profile.
    current = json.loads((RESULTS / "current_summary.json").read_text())
    inferred = {
        (row["side"], row["layout"]): row
        for row in current["profiles"]["inferred_logical_mapping"]["cases"]
    }
    august = {
        (row["side"], row["layout"]): row
        for row in profiles["august_2025_git_exact"]["cases"]
    }
    compatibility_cases = []
    for side in ("w2w", "d2w"):
        for layout in experiment.layouts:
            current_row = inferred[(side, str(layout))]
            overlay_row = august[(side, str(layout))]
            actual = dict(current_row["actual"])
            actual["Y_ovl"] = overlay_row["actual_Y_ovl"]
            actual["Y_bond"] = actual["Y_ovl"] * actual["Y_cr"] * actual["Y_df"]
            reference = current_row["reference"]
            errors = {name: abs(actual[name] - reference[name])
                      for name in experiment.components}
            compatibility_cases.append({
                "side": side, "layout": str(layout), "actual": actual,
                "reference": reference, "absolute_errors": errors,
                "case_pass": all(value <= tolerance for value in errors.values()),
            })
    compatibility = {
        "overlay_source": "august_2025_git_exact (331ac05)",
        "non_overlay_source": "current calculators",
        "layout_mapping_profile": "inferred_logical_mapping (explicit sensitivity fit, not recovered provenance)",
        "passing_cases": sum(row["case_pass"] for row in compatibility_cases),
        "all_cases_pass": all(row["case_pass"] for row in compatibility_cases),
        "max_abs_error": max(max(row["absolute_errors"].values()) for row in compatibility_cases),
        "cases": compatibility_cases,
    }
    # Diagnostic requested by the author: combine the May-2025 wafer/die
    # radial scale with the August-2025 layout-aware critical boundary.  No
    # recovered commit contains both behaviors, so this must remain clearly
    # labeled as a sensitivity result rather than historical provenance.
    scaled_boundary_cases = []
    scaled_params = profile_parameters(experiment, "paper_table_i", "d2w")
    for layout in experiment.layouts:
        actual = historical_yield(
            side="d2w", layout=str(layout), params=scaled_params,
            radial_scale=True, use_critical_boundaries=True,
            experiment=experiment,
        )
        reference = float(experiment.paper.d2w[layout][0])
        scaled_boundary_cases.append({
            "layout": str(layout), "actual_Y_ovl": actual,
            "paper_Y_ovl": reference, "absolute_error": abs(actual - reference),
            "pass": abs(actual - reference) <= tolerance,
        })
    scaled_boundary_sensitivity = {
        "description": (
            "Sensitivity-only hybrid: May-2025 wafer/die radial scale plus "
            "August-2025 critical-layout boundaries, using Table-I parameters."
        ),
        "not_recovered_in_a_single_commit": True,
        "passing_cases": sum(row["pass"] for row in scaled_boundary_cases),
        "all_cases_pass": all(row["pass"] for row in scaled_boundary_cases),
        "max_abs_error": max(row["absolute_error"] for row in scaled_boundary_cases),
        "cases": scaled_boundary_cases,
    }
    output = {
        "audit_only": True,
        "not_used_for_current_results": True,
        "historical_formula": "per-sample max(systematic corner magnitude), then one scalar random-error CDF",
        "seed": SEED,
        "num_samples": int(experiment.profiles.paper_table_i.common.num_samples),
        "pass_abs_tolerance": tolerance,
        "profiles": profiles,
        "compatibility_composite": compatibility,
        "wafer_scaled_boundary_sensitivity": scaled_boundary_sensitivity,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "legacy_overlay_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({name: {key: value for key, value in profile.items() if key != "cases"}
                      for name, profile in profiles.items()}, indent=2))


if __name__ == "__main__":
    main()
