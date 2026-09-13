#!/usr/bin/env python3
"""Audit all six 0.3 um-pitch cases, including the true rightmost three.

This is intentionally separate from the normal reproduction driver.  It
compares the published Table-I inputs with the values left in the historical
configuration and notebooks, and checks both the October-2025 corner-
aggregation change and the August-2026 numerical solver change.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from model_worker import import_side, paper_config


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
AREAS_MM2 = np.array([10.0, 50.0, 100.0])
PAPER_FIG15 = np.array([
    [.960, .994, .980, .934], [.960, .974, .930, .866], [.960, .946, .884, .801],
    [.960, .994, .998, .952], [.960, .974, .994, .925], [.960, .946, .988, .896],
])
PAPER_FIG16 = np.array([
    [.956, .994, .989, .939], [.939, .974, .945, .864], [.920, .946, .900, .784],
    [.956, .994, .999, .950], [.939, .974, .994, .909], [.920, .946, .988, .864],
])
SAMPLES = 120_000
SEED = 20260120
FIT_ROTATION_MEAN = 3.0331570605484688e-6
FIT_ROTATION_STD = 2.9831571076849285e-8
FIT_TRANSLATION_STD = .02138207133974166
LEGACY_SINGLE_PARAMETER_ROTATION_MEAN = 1.4560605092787923e-6


def load_fine_pitch_cases(figure: int, side: str, suffix: str = "") -> np.ndarray:
    return np.asarray(
        [
            [
                json.loads((RESULTS / f"fig{figure}_{side}_c{case}{suffix}.json").read_text())[key]
                for key in ("Y_ovl", "Y_cr", "Y_df", "Y_bond")
            ]
            for case in range(7, 13)
        ]
    )


def legacy_contact_limit(top: float, bottom: float, fraction: float) -> float:
    """Exact pre-August-2026 SymPy/fsolve equation, evaluated numerically."""
    def residual(distance):
        d = np.asarray(distance)
        theta1 = np.arccos((top**2 + d**2 - bottom**2) / (2 * top * d))
        theta2 = np.arccos((bottom**2 + d**2 - top**2) / (2 * bottom * d))
        area = top**2 * theta1 + bottom**2 * theta2 - d * top * np.sin(theta1)
        return area - fraction * np.pi * top**2

    return float(fsolve(residual, bottom)[0])


def fixed_normals() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(SEED)
    return tuple(rng.standard_normal(SAMPLES) for _ in range(4))


Z_TX, Z_TY, Z_ROT, Z_MAG = fixed_normals()


def d2w_overlay_curve(
    *,
    rotation_mean: float,
    rotation_std: float,
    top_ratio: float,
    random_mean: float = 0.0,
    random_std: float = .02,
    translation_mean: float = 0.0,
    translation_std: float = .02,
    magnification_mean: float = .05e-6,
    magnification_std: float = .01e-6,
    corner_aggregation: str = "current",
) -> np.ndarray:
    """Evaluate either historical D2W corner-aggregation implementation.

    ``legacy`` is the implementation used from May through September 2025:
    for each Monte Carlo draw, take the worst of the four corner
    misalignments, evaluate its conditional random-misalignment yield, and
    then average.  ``current`` is the implementation introduced by 742f4cc:
    average the conditional yield at each corner first, then take the worst
    corner mean.
    """
    if corner_aggregation not in {"legacy", "current"}:
        raise ValueError(f"unsupported corner aggregation: {corner_aggregation}")
    pitch = .3
    bottom = pitch / 2 * .5
    top = bottom * top_ratio
    api = import_side("d2w")
    contact_limit = api.overlay.__globals__["_contact_area_misalignment_limit"](
        top, bottom, .5
    )
    critical_distance_limit = .5 * pitch - top
    limit = min(contact_limit, critical_distance_limit)

    tx = translation_mean + translation_std * Z_TX
    ty = translation_mean + translation_std * Z_TY
    rotation = rotation_mean + rotation_std * Z_ROT
    magnification = magnification_mean + magnification_std * Z_MAG
    values = []
    for area in AREAS_MM2:
        side_um = math.sqrt(area) * 1000
        pad_extent = (math.floor(side_um / pitch) - 1) * pitch / 2
        corners = np.asarray(
            [
                [-pad_extent, pad_extent],
                [pad_extent, pad_extent],
                [-pad_extent, -pad_extent],
                [pad_extent, -pad_extent],
            ]
        )
        corner_systematic = []
        for x, y in corners:
            dx = tx - rotation * y + magnification * x
            dy = ty + rotation * x + magnification * y
            corner_systematic.append(np.hypot(dx, dy))
        if corner_aggregation == "legacy":
            systematic = np.max(np.asarray(corner_systematic), axis=0)
            upper = limit - systematic
            lower = -limit - systematic
            values.append(
                np.mean(
                    norm.cdf(upper, loc=random_mean, scale=random_std)
                    - norm.cdf(lower, loc=random_mean, scale=random_std)
                )
            )
            continue
        corner_yields = []
        for systematic in corner_systematic:
            upper = limit - systematic
            lower = -limit - systematic
            corner_yields.append(np.mean(
                norm.cdf(upper, loc=random_mean, scale=random_std)
                - norm.cdf(lower, loc=random_mean, scale=random_std)
            ))
        values.append(min(corner_yields))
    return np.asarray(values)


def w2w_overlay_curves() -> tuple[np.ndarray, np.ndarray]:
    """Return current/legacy W2W results in one pass using Table-I inputs."""
    pitch = .3
    bottom = pitch / 2 * .5
    top = bottom * .6
    api = import_side("d2w")
    contact_limit = api.overlay.__globals__["_contact_area_misalignment_limit"](
        top, bottom, .5
    )
    limit = min(contact_limit, .5 * pitch - top)
    # W2W has thousands of dies on the smallest-area wafer.  Twenty thousand
    # common-random-number draws resolve this historical-method delta well
    # while keeping the audit inexpensive.
    w2w_samples = 20_000
    tx = .02 * Z_TX[:w2w_samples]
    ty = .02 * Z_TY[:w2w_samples]
    rotation = 5e-8 + 1e-8 * Z_ROT[:w2w_samples]
    magnification = .05e-6 + .01e-6 * Z_MAG[:w2w_samples]
    wafer_radius = 150_000.0
    dice_width = 1_000.0
    current_values = []
    legacy_values = []
    for area in AREAS_MM2:
        side_um = math.sqrt(area) * 1000
        pad_extent = (math.floor(side_um / pitch) - 1) * pitch / 2
        row_count = int(2 * wafer_radius // (side_um + dice_width) + 1)
        col_count = row_count
        current_die_yields = []
        legacy_die_yields = []
        for row in range(row_count):
            cy = (row - (row_count - 1) / 2) * (side_um + dice_width)
            for col in range(col_count):
                cx = (col - (col_count - 1) / 2) * (side_um + dice_width)
                vertices = (
                    (cx - side_um / 2, cy - side_um / 2),
                    (cx - side_um / 2, cy + side_um / 2),
                    (cx + side_um / 2, cy - side_um / 2),
                    (cx + side_um / 2, cy + side_um / 2),
                )
                if any(math.hypot(x, y) >= wafer_radius for x, y in vertices):
                    continue
                corners = (
                    (cx - pad_extent, cy + pad_extent),
                    (cx + pad_extent, cy + pad_extent),
                    (cx - pad_extent, cy - pad_extent),
                    (cx + pad_extent, cy - pad_extent),
                )
                corner_systematic = []
                for x, y in corners:
                    dx = tx - rotation * y + magnification * x
                    dy = ty + rotation * x + magnification * y
                    corner_systematic.append(np.hypot(dx, dy))
                corner_yields = []
                for systematic in corner_systematic:
                    corner_yields.append(np.mean(
                        norm.cdf(limit - systematic, loc=0.0, scale=.02)
                        - norm.cdf(-limit - systematic, loc=0.0, scale=.02)
                    ))
                current_die_yields.append(min(corner_yields))
                legacy_systematic = np.max(np.asarray(corner_systematic), axis=0)
                legacy_die_yields.append(np.mean(
                    norm.cdf(limit - legacy_systematic, loc=0.0, scale=.02)
                    - norm.cdf(-limit - legacy_systematic, loc=0.0, scale=.02)
                ))
        current_values.append(float(np.mean(current_die_yields)))
        legacy_values.append(float(np.mean(legacy_die_yields)))
    return np.asarray(current_values), np.asarray(legacy_values)


def main() -> None:
    latest15 = load_fine_pitch_cases(15, "w2w")
    latest16 = load_fine_pitch_cases(16, "d2w")
    verified_fit = load_fine_pitch_cases(16, "d2w", "_effective_fit")

    # Table I and the dedicated reproduction configuration.
    table_curve = d2w_overlay_curve(
        rotation_mean=5e-8, rotation_std=1e-8, top_ratio=.6
    )
    table_curve_legacy_aggregation = d2w_overlay_curve(
        rotation_mean=5e-8,
        rotation_std=1e-8,
        top_ratio=.6,
        corner_aggregation="legacy",
    )
    w2w_table_curve, w2w_table_curve_legacy_aggregation = w2w_overlay_curves()
    # d2w_modeling in the Jan/Apr 2026 config before notebook overrides.
    old_config_curve = d2w_overlay_curve(
        rotation_mean=1e-6,
        rotation_std=5e-7,
        top_ratio=2 / 3,
        magnification_std=.0001e-6,
    )
    old_config_curve_legacy_aggregation = d2w_overlay_curve(
        rotation_mean=1e-6,
        rotation_std=5e-7,
        top_ratio=2 / 3,
        magnification_std=.0001e-6,
        corner_aggregation="legacy",
    )
    # calculator_main.ipynb overwrites the mean with 2e-6 rad.
    old_notebook_curve = d2w_overlay_curve(
        rotation_mean=2e-6,
        rotation_std=5e-7,
        top_ratio=2 / 3,
        magnification_std=.0001e-6,
    )
    old_notebook_curve_legacy_aggregation = d2w_overlay_curve(
        rotation_mean=2e-6,
        rotation_std=5e-7,
        top_ratio=2 / 3,
        magnification_std=.0001e-6,
        corner_aggregation="legacy",
    )
    legacy_single_parameter_fit = d2w_overlay_curve(
        rotation_mean=LEGACY_SINGLE_PARAMETER_ROTATION_MEAN,
        rotation_std=5e-7,
        top_ratio=2 / 3,
        magnification_std=.0001e-6,
        corner_aggregation="legacy",
    )
    fit_mean = FIT_ROTATION_MEAN
    fit_std = FIT_ROTATION_STD
    fit_translation_std = FIT_TRANSLATION_STD
    fitted_curve = d2w_overlay_curve(
        rotation_mean=fit_mean,
        rotation_std=fit_std,
        translation_std=fit_translation_std,
        top_ratio=.6,
    )

    api = import_side("d2w")
    cfg = paper_config("d2w", {"PITCH_um": .3}, api)
    new_limit = api.overlay.__globals__["_contact_area_misalignment_limit"](
        cfg.PAD_TOP_R_um, cfg.PAD_BOT_R_um, cfg.CONTACT_AREA_CONSTRAINT
    )
    old_limit = legacy_contact_limit(
        cfg.PAD_TOP_R_um, cfg.PAD_BOT_R_um, cfg.CONTACT_AREA_CONSTRAINT
    )

    payload = {
        "scope": "all six fine-pitch cases; true rightmost three are density=0.01 cm^-2, pitch=0.3 um, area=10/50/100 mm^2",
        "paper_plot_readback": {
            "fig15_w2w": PAPER_FIG15.tolist(),
            "fig16_d2w": PAPER_FIG16.tolist(),
        },
        "latest_with_table_I_parameters": {
            "fig15_w2w": latest15.tolist(),
            "fig16_d2w": latest16.tolist(),
        },
        "verified_500k_sample_effective_fit_all_six_fine_pitch_cases": {
            "fig16_d2w": verified_fit.tolist(),
            "max_abs_error_vs_plot_readback": float(
                np.max(np.abs(verified_fit - PAPER_FIG16))
            ),
        },
        "d2w_overlay_scenarios": {
            "paper_table_I": table_curve.tolist(),
            "paper_table_I_with_may_to_september_2025_corner_aggregation": table_curve_legacy_aggregation.tolist(),
            "pre_summer_d2w_modeling_config": old_config_curve.tolist(),
            "2025_config_with_may_to_september_2025_corner_aggregation": old_config_curve_legacy_aggregation.tolist(),
            "pre_summer_calculator_notebook": old_notebook_curve.tolist(),
            "2025_notebook_with_may_to_september_2025_corner_aggregation": old_notebook_curve_legacy_aggregation.tolist(),
            "paper_plot_readback": PAPER_FIG16[3:, 0].tolist(),
            "diagnostic_2025_legacy_single_parameter_fit": {
                "rotation_mean_rad": LEGACY_SINGLE_PARAMETER_ROTATION_MEAN,
                "yield": legacy_single_parameter_fit.tolist(),
                "warning": "Inverse fit only; bracketed by the 2025 config (1e-6) and later notebook (2e-6), but not committed in git.",
            },
            "diagnostic_effective_input_fit": {
                "mean_rad": fit_mean,
                "std_rad": fit_std,
                "translation_std_um": fit_translation_std,
                "yield": fitted_curve.tolist(),
                "warning": "Inverse fit only; these values are not present in git history.",
            },
        },
        "w2w_overlay_scenarios": {
            "paper_table_I_current_corner_aggregation": w2w_table_curve.tolist(),
            "paper_table_I_with_may_to_september_2025_corner_aggregation": w2w_table_curve_legacy_aggregation.tolist(),
            "paper_plot_readback": PAPER_FIG15[:3, 0].tolist(),
        },
        "overlay_solver_check": {
            "pre_summer_fsolve_limit_um": old_limit,
            "latest_brentq_limit_um": float(new_limit),
            "absolute_difference_um": abs(old_limit - new_limit),
        },
        "historical_evidence": {
            "paper_pdf_created": "2026-01-20",
            "paper_table_rotation_rad": [5e-8, 1e-8],
            "commit_b6f9651_config_rotation_rad": [1e-6, 5e-7],
            "commit_b6f9651_notebook_rotation_mean_rad": 2e-6,
            "commit_b6f9651_notebook_top_ratio": 2 / 3,
            "commit_b6f9651_simulator_notebook_rotation_mean_rad": 1e-7,
            "simulator_notebook_contains_fig15_16_case_study_save_paths": False,
            "commit_6e80bc8_same_as_b6f9651": True,
            "latest_config_rotation_rad": [1e-6, 5e-7],
            "latest_notebook_rotation_mean_rad": 2e-6,
            "d2w_corner_aggregation_change": {
                "commit": "742f4cc (2025-10-06)",
                "before": "mean(random-yield(max(corner systematic misalignment per sample)))",
                "after": "min(mean(random-yield(corner systematic misalignment)))",
                "reason_in_commit": "Support D2W pad-level yield calculation",
            },
            "2025_w2w_critical_ratio_timeline": {
                "331ac05_2025_08_28": 1.0,
                "a5ae68a_2025_10_09": 1.0,
                "b6f9651_2025_10_21": 0.9,
            },
            "fine_pitch_bitmap_history": {
                "stable_commits": "the per-side file loaded by calculator_main.ipynb is absent",
                "transitional_commit_0a209f2": "root pad_bitmap/bitmap_collection.npy exists: 1000x1000 pads, 25x25 full-critical blocks, block size 40 pads",
                "interpretation": "the surviving file is the 10 mm / 10 um pitch / 400 um-grid baseline, not a complete set of 0.3 um and 10/50/100 mm2 inputs",
            },
        },
        "particle_count_model_audit": {
            "paper_and_analytical_model": "Poisson yield, Y_df = exp(-Lambda), paper equations 23 and 29",
            "pre_summer_simulator": "one fixed total followed by multinomial allocation across wafers/dies",
            "latest_simulator": "independent Poisson count per wafer/die",
            "change_commit": "badf7a9 (2026-08-28)",
            "same_per_unit_mean": True,
            "old_marginal_variance_relative_to_poisson": {
                "w2w_NUM_WAFERS_10": 0.9,
                "d2w_NUM_DIES_10000": 0.9999,
            },
            "old_cross_unit_covariance": "negative; Cov(N_i,N_j)=-lambda/M",
            "fig15_16_analytical_mean_impact": "none: these runs use exp(-Lambda), not simulator particle-count sampling",
            "simulation_impact": "D2W marginal effect is negligible at M=10000; W2W variance and cross-wafer correlation can change materially at M=10",
        },
        "critical_pad_and_grid_audit": {
            "paper_fig15_16": {
                "critical_ratio": 1.0,
                "redundant_ratio": 0.0,
                "w2w_grid_um": 400.0,
                "d2w_grid_um": 100.0,
            },
            "this_reproduction": {
                "critical_ratio": 1.0,
                "redundant_ratio": 0.0,
                "w2w_grid_um": 400.0,
                "d2w_grid_um": 100.0,
                "derived_block_sizes_in_pads": {
                    "w2w_pitch_1": 400,
                    "w2w_pitch_0p3": 1333,
                    "d2w_pitch_1": 100,
                    "d2w_pitch_0p3": 333,
                },
            },
            "pre_summer_modeling_config": {
                "w2w": {"critical_ratio": 0.9, "redundant_ratio": 0.1, "grid_um": 400.0},
                "d2w": {"critical_ratio": 1.0, "redundant_ratio": 0.0, "grid_um": 400.0},
            },
            "latest_default_modeling_config": {
                "w2w": {"critical_ratio": 0.2, "redundant_ratio": 0.5, "grid_um": 400.0},
                "d2w": {"critical_ratio": 1.0, "redundant_ratio": 0.0, "grid_um": 400.0},
            },
            "interpretation": "repository defaults are general demos and do not reproduce Fig.15/16 without paper overrides",
        },
    }
    (RESULTS / "fig15_16_version_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(AREAS_MM2, PAPER_FIG16[3:, 0], "ko-", label="Fig.16 plot readback")
    ax.plot(AREAS_MM2, table_curve, "o-", label="Table I / latest equations")
    ax.plot(AREAS_MM2, table_curve_legacy_aggregation, "s-", label="Table I / pre-Oct. 6, 2025 aggregation")
    ax.plot(AREAS_MM2, old_config_curve, "o-", label="pre-summer config")
    ax.plot(AREAS_MM2, old_config_curve_legacy_aggregation, "s--", label="2025 config / pre-Oct. aggregation")
    ax.plot(AREAS_MM2, old_notebook_curve, "o-", label="pre-summer notebook")
    ax.plot(AREAS_MM2, old_notebook_curve_legacy_aggregation, "s--", label="2025 notebook / pre-Oct. aggregation")
    ax.plot(AREAS_MM2, legacy_single_parameter_fit, ":", label="2025 legacy + one-parameter fit")
    ax.plot(AREAS_MM2, fitted_curve, "--", label="effective-input fit (diagnostic)")
    ax.set_xlabel("Chiplet area (mm$^2$)")
    ax.set_ylabel("D2W overlay yield")
    ax.set_ylim(.90, .98)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig16_right3_version_audit.png", dpi=200)
    plt.close(fig)

    full_latest = np.asarray(
        [
            [
                json.loads((RESULTS / f"fig16_d2w_c{case}.json").read_text())[key]
                for key in ("Y_ovl", "Y_cr", "Y_df", "Y_bond")
            ]
            for case in range(1, 13)
        ]
    )
    full_compat = full_latest.copy()
    full_compat[6:] = verified_fit
    labels = (
        "0.1,1,10", "0.1,1,50", "0.1,1,100",
        "0.01,1,10", "0.01,1,50", "0.01,1,100",
        "0.1,0.3,10", "0.1,0.3,50", "0.1,0.3,100",
        "0.01,0.3,10", "0.01,0.3,50", "0.01,0.3,100",
    )
    colors = ("#0072B2", "#D55E00", "#E69F00", "#7B3294")
    x = np.arange(12)
    width = .19
    fig, ax = plt.subplots(figsize=(16, 5.0))
    for index, (name, color) in enumerate(
        zip(("Yovl", "Ycr", "Ydf", "YD2W"), colors)
    ):
        ax.bar(x + (index - 1.5) * width, full_compat[:, index], width, label=name, color=color)
    ax.set_ylim(.75, 1.005)
    ax.set_ylabel("D2W yield components")
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.set_xlabel("particle density (cm$^{-2}$), pitch (um), chiplet area (mm$^2$)")
    ax.grid(axis="y", alpha=.25)
    areas = np.asarray([10, 50, 100] * 4, dtype=float)
    right = ax.twinx()
    right.bar(
        x + 2.5 * width,
        full_compat[:, 3] ** (1000 / areas),
        width,
        label="Ysys",
        color="#66A61E",
    )
    right.set_ylim(0, 1.005)
    right.set_ylabel("System yield")
    handles, legend_labels = ax.get_legend_handles_labels()
    h2, l2 = right.get_legend_handles_labels()
    ax.legend(handles + h2, legend_labels + l2, ncol=5, loc="lower left")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig16_d2w_all_effective_fit.png", dpi=200)
    plt.close(fig)

    md = f"""# Fig. 15/16 fine-pitch and parameter-history audit

Scope: all six pitch=0.3 um columns.  The actual rightmost three are density
0.01 cm^-2, not 0.1 cm^-2, with areas 10/50/100 mm2.

| D2W overlay source | 10 | 50 | 100 |
|---|---:|---:|---:|
| Paper plot readback | {PAPER_FIG16[3,0]:.4f} | {PAPER_FIG16[4,0]:.4f} | {PAPER_FIG16[5,0]:.4f} |
| Table I + latest equations | {table_curve[0]:.4f} | {table_curve[1]:.4f} | {table_curve[2]:.4f} |
| Table I + May--Sep. 2025 corner aggregation | {table_curve_legacy_aggregation[0]:.4f} | {table_curve_legacy_aggregation[1]:.4f} | {table_curve_legacy_aggregation[2]:.4f} |
| Pre-summer `d2w_modeling` config | {old_config_curve[0]:.4f} | {old_config_curve[1]:.4f} | {old_config_curve[2]:.4f} |
| 2025 config + May--Sep. corner aggregation | {old_config_curve_legacy_aggregation[0]:.4f} | {old_config_curve_legacy_aggregation[1]:.4f} | {old_config_curve_legacy_aggregation[2]:.4f} |
| Pre-summer calculator notebook overrides | {old_notebook_curve[0]:.4f} | {old_notebook_curve[1]:.4f} | {old_notebook_curve[2]:.4f} |
| 2025 notebook + May--Sep. aggregation | {old_notebook_curve_legacy_aggregation[0]:.4f} | {old_notebook_curve_legacy_aggregation[1]:.4f} | {old_notebook_curve_legacy_aggregation[2]:.4f} |
| 2025 legacy + fitted rotation mean only | {legacy_single_parameter_fit[0]:.4f} | {legacy_single_parameter_fit[1]:.4f} | {legacy_single_parameter_fit[2]:.4f} |
| Effective-input inverse fit (diagnostic only) | {fitted_curve[0]:.4f} | {fitted_curve[1]:.4f} | {fitted_curve[2]:.4f} |

The independent 500,000-sample verification gives complete rightmost-three
`(Yovl, Ycr, Ydf, YD2W)` rows:

- 10 mm2: `{tuple(round(float(x), 6) for x in verified_fit[3])}`
- 50 mm2: `{tuple(round(float(x), 6) for x in verified_fit[4])}`
- 100 mm2: `{tuple(round(float(x), 6) for x in verified_fit[5])}`

Its maximum component error against the approximate plot readback is
`{np.max(np.abs(verified_fit-PAPER_FIG16)):.4f}`.

The 2025 history reveals a direct, material contributor to the D2W overlay discrepancy.
Commit `742f4cc` (2025-10-06), whose stated purpose was D2W pad-level yield,
changed the die-level reduction from `mean(yield(max(corner misalignment)))`
to `min(mean(yield(corner misalignment)))`.  These operations are not
equivalent.  With the 2025 D2W config inputs, the pre-October implementation
is much closer to the plotted area dependence, while the current implementation
stays too high.  It does not by itself explain the complete gap at 50/100 mm2,
and using Table I rather than the committed D2W config largely removes this
effect.  Density does not enter overlay, so the same issue applies to both
fine-pitch density groups.

The analogous W2W change was made in commit `29f7b95` on the same date.  With
Table-I inputs, current versus pre-October W2W overlay for 10/50/100 mm2 is
`{tuple(round(float(x), 6) for x in w2w_table_curve)}` versus
`{tuple(round(float(x), 6) for x in w2w_table_curve_legacy_aggregation)}`;
the paper bars are approximately `{tuple(float(x) for x in PAPER_FIG15[:3,0])}`.  This confirms
that the aggregation change is also relevant to Fig. 15, though its numerical
effect is smaller than the D2W discrepancy.

## Particle-count distribution

The paper explicitly uses the Poisson yield model in Eqs. 23 and 29, and both
the pre-summer and latest analytical calculators evaluate `exp(-Lambda)`.
Therefore changing simulator particle generation does not alter the Fig. 15/16
analytical mean values.

The simulator did change in commit `badf7a9`: the pre-summer code fixed one
total particle count and distributed it with a multinomial draw, while the
latest code draws independent Poisson counts per wafer/die.  Both have the same
per-unit mean.  The old marginal variance is `(1-1/M)` times the Poisson
variance and it introduces covariance `-lambda/M` between units.  With 10,000
D2W dies this is negligible (factor 0.9999); with 10 W2W wafers the old variance
is 10% low (factor 0.9), so this can affect Fig. 13 simulation dispersion and
correlation.  The new implementation is the one consistent with an independent
spatial Poisson process.

## Critical-pad fraction and pad-block size

The paper says Fig. 15/16 uses a layout composed exclusively of critical pads,
and Table I plus the Fig. 12 discussion set the analytical grids to 400 um for
W2W and 100 um for D2W.  This reproduction therefore enforces critical/
redundant/dummy = 1/0/0 and those grid dimensions in all 24 configuration rows
(12 per figure).
The derived block sizes are 400 and 1333 pads for W2W at pitch 1 and 0.3 um,
and 100 and 333 pads for D2W; the 0.3 um effective dimensions are 399.9 and
99.9 um because the repository intentionally uses integer truncation.

This override is necessary for current/pre-summer defaults.  However, the
August 28 through October 9, 2025 W2W configs still used 1.0/0.0
critical/redundant, matching the paper; commit `b6f9651` on October 21 changed
the W2W default to 0.9/0.1.  Pre-summer `w2w_modeling` therefore defaults to 0.9/0.1
critical/redundant, while its `d2w_modeling` uses 1/0 but a 400 um grid.  The
latest defaults are 0.2/0.5 for W2W and 1/0 for D2W, both with 400 um grids.
The historical calculator notebooks load those modeling sections and do not
contain an explicit critical-ratio or block-dimension correction.  Thus the
committed default configs are not reliable Fig. 15/16 provenance; the paper's
explicit experiment settings take precedence.

The published Table I says rotation mean/std = 5e-8/1e-8 rad.  In both the
paper-era commit `b6f9651` and pre-summer commit `6e80bc8`, D2W
`d2w_modeling` instead contains 1e-6/5e-7 rad; `calculator_main.ipynb` then
overwrites the mean with 2e-6 rad and uses a top/bottom radius ratio of 2/3
instead of Table I's 0.6.  The latest branch retains those D2W values.  They
explain part, but not all, of the plotted area dependence.

The historical `simulator_main.ipynb` was also inspected.  Its committed D2W
cell overrides rotation mean to 1e-7 rad and is organized around
model-versus-simulation/correlation checks; it does not contain the Fig. 15/16
case-study MAT save paths.  Those save paths occur in `calculator_main.ipynb`,
so the calculator notebook is the stronger provenance source for these bars.

The old notebooks' `<1 um` branch expected
`pad_bitmap/bitmap_collection.npy`. That per-side file is absent from the
stable 2025 commits. A transitional commit (`0a209f2`) does preserve a root
bitmap: a full-critical 1000x1000 pad array with a 25x25 block map and 40-pad
blocks, corresponding to the 10 mm die / 10 um pitch / 400 um-grid baseline.
It is not the missing set of fine-pitch bitmaps for all three die sizes.

The August-2026 contact-overlap rewrite is not the cause: the old fsolve limit
is `{old_limit:.12g}` um and the new brentq limit is `{float(new_limit):.12g}`
um (difference `{abs(old_limit-new_limit):.3g}` um).  The rewrite replaces a
singular, warning-prone root solve with a bracketed analytic circle-overlap
solve while preserving the result.

The latest code also fixes an old D2W radius-loader division typo, adds fixed
random seeds, keys dilation reuse to layout-defining parameters, and rejects a
full-resolution bitmap whose shape does not match the active configuration.
Those changes are consistent with correctness/reproducibility fixes in commit
`badf7a9` ("Fixed bugs on 8/28 by Zhichao"), not a deliberate change to the
overlay physics.  The historical `<1 um` notebook path loaded a saved bitmap;
the new shape check now exposes incompatible cached bitmaps instead of silently
using them.

Matching the three plot bars by changing rotation and translation spread would
require effective rotation mean/std of approximately
`{fit_mean:.4g}`/`{fit_std:.4g}` rad and translation std
`{fit_translation_std:.4g}` um.
That is a diagnostic identifiability result, not recovered provenance: those
numbers do not occur in the inspected git history.  Therefore the remaining
gap should be reported as a paper/config provenance inconsistency unless the
original MAT/NPY inputs or the exact notebook execution state can be recovered.

A more historically constrained diagnostic keeps every August-2025 config
value and the pre-October aggregation, changing only rotation mean. Its best
mean is `{LEGACY_SINGLE_PARAMETER_ROTATION_MEAN:.6g}` rad and produces
`{tuple(round(float(x), 6) for x in legacy_single_parameter_fit)}`. This is
not a recovered parameter, but it lies between the committed August config
(`1e-6`) and October notebook (`2e-6`).
"""
    (RESULTS / "FIG15_16_VERSION_AUDIT.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
