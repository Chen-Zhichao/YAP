#!/usr/bin/env python3
"""Audit the May-2025 D2W wafer-radius scaling against all Fig. 16 cases."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LEGACY_MODULE = HERE.parent / "fig17" / "legacy_overlay.py"


def load_legacy_module():
    spec = importlib.util.spec_from_file_location("fig17_legacy_overlay", LEGACY_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {LEGACY_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parameters(profile) -> dict:
    values = OmegaConf.to_container(profile.overrides, resolve=True)
    defaults = {
        "SYSTEM_TRANSLATION_X_MEAN_um": 0.0,
        "SYSTEM_TRANSLATION_Y_MEAN_um": 0.0,
        "RANDOM_MISALIGNMENT_MEAN_um": 0.0,
        "M_0": 0.0,
    }
    return {**defaults, **values}


def main() -> None:
    legacy = load_legacy_module()
    legacy.source_assertions("6cc2408", "d2w", True, False)
    config = OmegaConf.load(HERE / "config.yaml").experiment
    current = json.loads((RESULTS / "summary.json").read_text())
    output_profiles = {}
    for profile_name in ("paper_table_i", "december_2025_modeling", "december_2025_notebook"):
        params = parameters(config.profiles[profile_name])
        rows = []
        current_rows = {row["index"]: row for row in current["profiles"][profile_name]["cases"]}
        for index, case in enumerate(config.configurations, 1):
            die_side_um = math.sqrt(float(case.area_mm2)) * 1000.0
            experiment = SimpleNamespace(
                pitch_um=float(case.pitch_um),
                die_width_um=die_side_um,
                die_length_um=die_side_um,
            )
            scaled = legacy.historical_yield(
                side="d2w", layout="full", params=params,
                radial_scale=True, use_critical_boundaries=False,
                experiment=experiment,
            )
            reference = float(case.paper[0])
            current_yield = float(current_rows[index]["actual"]["Y_ovl"])
            rows.append({
                "index": index,
                "pitch_um": float(case.pitch_um),
                "area_mm2": float(case.area_mm2),
                "wafer_to_die_radial_scale": 150000.0 / math.hypot(
                    die_side_um / 2.0, die_side_um / 2.0
                ),
                "paper_Y_ovl": reference,
                "current_no_radial_scale_Y_ovl": current_yield,
                "may_2025_radial_scale_Y_ovl": scaled,
                "current_absolute_error": abs(current_yield - reference),
                "radial_scale_absolute_error": abs(scaled - reference),
            })
        current_residuals = np.asarray([
            row["current_no_radial_scale_Y_ovl"] - row["paper_Y_ovl"] for row in rows
        ])
        scaled_residuals = np.asarray([
            row["may_2025_radial_scale_Y_ovl"] - row["paper_Y_ovl"] for row in rows
        ])
        output_profiles[profile_name] = {
            "current_overlay_rmse": float(np.sqrt(np.mean(current_residuals**2))),
            "current_overlay_max_abs_error": float(np.max(np.abs(current_residuals))),
            "radial_scale_overlay_rmse": float(np.sqrt(np.mean(scaled_residuals**2))),
            "radial_scale_overlay_max_abs_error": float(np.max(np.abs(scaled_residuals))),
            "rows": rows,
        }
    output = {
        "figure": 16,
        "audit_only": True,
        "source_commit": "6cc2408",
        "source_expression": (
            "rotation and magnification samples multiplied by "
            "WAF_R / sqrt((DIE_W/2)^2 + (DIE_L/2)^2)"
        ),
        "wafer_radius_um": 150000.0,
        "interpretation": (
            "The scale cancels die-radius dependence of rotation/magnification "
            "displacement and is tested as historical provenance, not fitted input."
        ),
        "profiles": output_profiles,
    }
    (RESULTS / "wafer_scaling_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({name: {key: value for key, value in profile.items() if key != "rows"}
                      for name, profile in output_profiles.items()}, indent=2))


if __name__ == "__main__":
    main()
