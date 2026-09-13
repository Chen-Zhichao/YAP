#!/usr/bin/env python3
"""Run one side of the current-code Figure 13 interaction experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
REPRODUCTION_DIR = HERE.parent
ROOT = REPRODUCTION_DIR.parent
sys.path.insert(0, str(REPRODUCTION_DIR))

import model_worker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("w2w", "d2w"), required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--points", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def particle_density_axis(spec, count: int) -> np.ndarray:
    if spec.spacing == "logspace":
        return np.logspace(float(spec.start_exponent), float(spec.stop_exponent), count)
    if spec.spacing == "linear":
        return np.linspace(float(spec.start), float(spec.stop), count)
    raise ValueError(f"Unknown particle-density spacing: {spec.spacing}")


def main() -> None:
    args = parse_args()
    experiment = OmegaConf.load(args.config).experiment
    side_config = experiment[args.side]
    points = int(args.points if args.points is not None else experiment.points)
    seed = int(args.seed if args.seed is not None else experiment.seed)
    if args.output is None:
        args.output = HERE / "results" / f"fig13_{args.side}.json"
    args.output = args.output.resolve()
    work_dir = HERE / "work" / f"fig13_{args.side}"
    (work_dir / "pad_bitmap").mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)

    api = model_worker.import_side(args.side)
    sample_count = int(side_config.model_samples)
    common = experiment.common
    cfg = model_worker.paper_config(
        args.side,
        {
            "PITCH_um": float(common.pitch_um),
            "DIE_W_um": float(common.die_w_um),
            "DIE_L_um": float(common.die_l_um),
            "PAD_BOT_R_um_ratio": float(common.pad_bottom_ratio),
            "PAD_TOP_R_um_ratio": float(common.pad_top_to_bottom_ratio),
            "t_0": float(common.t_0_um),
            "Roughness_sigma_m": float(common.roughness_sigma_m),
            "BOW_DIFFERENCE_MEAN_um": float(common.bow_difference_mean_um),
            "SYSTEM_TRANSLATION_X_MEAN_um": float(
                common.system_translation_x_mean_um
            ),
            "SYSTEM_TRANSLATION_Y_MEAN_um": float(
                common.system_translation_y_mean_um
            ),
            "TOP_DISH_MEAN_nm": float(common.top_dishing_mean_nm),
            "BOT_DISH_MEAN_nm": float(common.bottom_dishing_mean_nm),
            "SYSTEM_ROTATION_STD_rad": float(side_config.rotation_std_rad),
            "num_samples": sample_count,
        },
        api,
    )
    bitmap = model_worker.compact_bitmap(cfg, "full", None)
    target = (
        model_worker.make_wafer(cfg, bitmap)
        if args.side == "w2w"
        else model_worker.make_die(cfg, bitmap)
    )
    if args.side == "w2w":
        # The archived sweep used the wafer-level model. A stride of one keeps
        # all current-code wafer positions; the option exists only for explicit
        # low-cost diagnostic runs.
        target.die_list = target.die_list[:: int(side_config.wafer_position_stride)]

    base = model_worker.calculate(args.side, cfg, bitmap, api, False)
    base_density = float(cfg.D0)

    rng = np.random.default_rng(seed + (0 if args.side == "w2w" else 1))
    rotation = np.linspace(
        float(side_config.rotation_mean_rad[0]),
        float(side_config.rotation_mean_rad[1]),
        points,
    )
    observations = int(side_config.effective_observations_per_point)
    dish_std = np.linspace(
        float(side_config.cu_dishing_std_nm[0]),
        float(side_config.cu_dishing_std_nm[1]),
        points,
    )
    density_um2 = particle_density_axis(side_config.particle_density_per_um2, points)

    rows = []
    overlay_kwargs = model_worker.common_overlay_kwargs(cfg)
    cu_kwargs = model_worker.common_cu_kwargs(cfg, bitmap)
    for index in range(points):
        api.seed(cfg, announce=False)
        overlay_kwargs["SYSTEM_ROTATION_MEAN_rad"] = float(rotation[index])
        if args.side == "w2w":
            y_ovl = float(api.overlay(cfg=cfg, wafer=target, **overlay_kwargs))
        else:
            y_ovl = float(api.overlay(die=target, **overlay_kwargs)[0])

        sigma = float(dish_std[index])
        cu_kwargs["TOP_DISH_STD_nm"] = sigma
        cu_kwargs["BOT_DISH_STD_nm"] = sigma
        if args.side == "w2w":
            y_cr = float(api.cu(wafer=target, **cu_kwargs))
        else:
            y_cr = float(api.cu(die=target, **cu_kwargs)[0])
        current_density_um2 = float(density_um2[index])
        y_df = float(base["Y_df"] ** (current_density_um2 / base_density))

        component_masks = [
            rng.random(observations) < y_ovl,
            rng.random(observations) < y_cr,
            rng.random(observations) < y_df,
        ]
        component_sim = [float(np.mean(mask)) for mask in component_masks]
        joint_masks = [
            rng.random(observations) < y_ovl,
            rng.random(observations) < y_cr,
            rng.random(observations) < y_df,
        ]
        overall = float(np.mean(joint_masks[0] & joint_masks[1] & joint_masks[2]))
        product = float(np.prod(component_sim))
        rows.append(
            {
                "index": index,
                "rotation_mean_rad": float(rotation[index]),
                "cu_dishing_std_nm": sigma,
                "particle_density_per_um2": current_density_um2,
                "particle_density_per_cm2": current_density_um2 * 1.0e8,
                "model_component_probabilities": {
                    "overlay": y_ovl,
                    "cu_expansion": y_cr,
                    "defect": y_df,
                },
                "simulated_component_yields": {
                    "overlay": component_sim[0],
                    "cu_expansion": component_sim[1],
                    "defect": component_sim[2],
                },
                "product_of_individual_yields": product,
                "overall_combined_yield": overall,
            }
        )

    yield_min, yield_max = (float(value) for value in experiment.yield_window)
    in_range = [
        row for row in rows if yield_min <= row["overall_combined_yield"] <= yield_max
    ]
    x = np.asarray([row["overall_combined_yield"] for row in rows])
    y = np.asarray([row["product_of_individual_yields"] for row in rows])
    result = {
        "figure": 13,
        "bonding_type": args.side.upper(),
        "repository_commit": model_worker.repository_revision(),
        "calculator_source_sha256": model_worker.calculator_source_sha256(),
        "method": "December-2025 legacy sweep vectors evaluated by current yap+ analytical calculators plus deterministic Bernoulli interaction experiment",
        "config": os.path.relpath(args.config.resolve(), ROOT),
        "requested_points": points,
        "output_points": len(rows),
        "points_in_yield_window": len(in_range),
        "yield_window": [yield_min, yield_max],
        "effective_observations_per_point": observations,
        "w2w_wafer_positions_per_model_point": (
            len(target.die_list) if args.side == "w2w" else None
        ),
        "mse": float(np.mean((x - y) ** 2)) if len(x) else None,
        "paper_mse": float(side_config.paper_mse),
        "parameter_ranges": {
            "rotation_mean_rad": [float(rotation.min()), float(rotation.max())],
            "cu_dishing_std_nm": [float(dish_std.min()), float(dish_std.max())],
            "particle_density_per_um2": [float(density_um2.min()), float(density_um2.max())],
            "particle_density_per_cm2": [float(density_um2.min() * 1.0e8), float(density_um2.max() * 1.0e8)],
        },
        "provenance_note": (
            "The 300-point die-size, pitch, particle-density, Cu-dishing, "
            "rotation, roughness, pad-ratio, and bow vectors are copied as "
            "parameters from the uploaded December 2025 simulator_main.py. "
            "All yield calculations use the current yap+ implementations."
        ),
        "points": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "points"}, indent=2))


if __name__ == "__main__":
    main()
