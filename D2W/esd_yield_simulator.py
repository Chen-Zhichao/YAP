# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Tuple
import numpy as np

Z_TOP_UM = 0.1


def _z_linear_coeffs(ax_deg: float, ay_deg: float) -> Tuple[float, float, float]:
    """Return the plane coefficients for R = Ry(ay) @ Rx(ax)."""
    ax = np.deg2rad(float(ax_deg))
    ay = np.deg2rad(float(ay_deg))
    ca, sa = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    a = -sy
    b = cy * sa
    c = cy * ca
    return float(a), float(b), float(c)


def _ipeak_from_die_voltage(area_mm2: float, v_chg: float) -> float:
    """Empirical peak-current model."""
    return 0.0045 * (float(area_mm2) ** 0.35) * math.sqrt(float(v_chg))


def _weibull_cdf(current_a: float, k: float, lam: float) -> float:
    """Weibull cumulative distribution function."""
    current_a = max(current_a, 1e-12)
    return max(0.0, min(1.0, 1.0 - math.exp(-((current_a / lam) ** k))))


def _fail_prob_single(current_a: float, k: float, lam: float, cutoff_a: float) -> float:
    """Return the single-event failure probability."""
    if current_a < cutoff_a:
        return 0.0
    return _weibull_cdf(current_a, k, lam)


def _compute_p_fail_for_die(
    top_die_w_um: float,
    top_die_h_um: float,
    v_chg: float,
    *,
    weibull_k: float,
    weibull_lambda: float,
    cutoff_min_a: float,
) -> float:
    """Return the die-level failure probability for a sampled charging voltage."""
    area_mm2 = (float(top_die_w_um) * 1e-3) * (float(top_die_h_um) * 1e-3)
    i_peak = _ipeak_from_die_voltage(area_mm2, float(v_chg))
    return _fail_prob_single(i_peak, float(weibull_k), float(weibull_lambda), float(cutoff_min_a))


def _arc_distance_um_from_voltage(v_chg: float) -> float:
    """
    Return the maximum air-gap distance [um] that can discharge at voltage v_chg [V].

    Modified Paschen curve:
      V = 97 d                       for d < 3.5 um
      V = 337                        for 3.5 um < d < 7 um
      V = 170 + 2.48 d + 58 sqrt(d) for d > 7 um
    """
    v_chg = max(0.0, float(v_chg))
    if v_chg <= 0.0:
        return 0.0

    plateau_v = 337.0                         # Voltage plateau between 3.5 um and 7 um gap
    small_gap_slope = 97.0                  # Slope of the small-gap linear region (V/um)
    plateau_upper_gap_um = 7.0              # Upper gap limit of the voltage plateau (um)

    if v_chg < plateau_v:
        return v_chg / small_gap_slope

    a = 2.48                            # Coefficient of the linear term in the large-gap region (V/um)
    b = 58.0                           # Coefficient of the sqrt term in the large-gap region (V/sqrt(um))
    c = 170.0 - v_chg                    # Constant term in the large-gap region (V)
    disc = b * b - 4.0 * a * c          # Discriminant of the quadratic equation for the large-gap region
    if disc <= 0.0:
        return plateau_upper_gap_um

    root = (-b + math.sqrt(disc)) / (2.0 * a)
    if root <= 0.0:
        return plateau_upper_gap_um
    return max(plateau_upper_gap_um, root * root)


def _prepare_pad_geometry_cache(
    pad_coords_um: np.ndarray,
    pad_size_um: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return reusable pad geometry arrays for repeated gap evaluations."""
    pad_coords_um = np.asarray(pad_coords_um, dtype=np.float64)
    return pad_coords_um[:, 0], pad_coords_um[:, 1], 0.5 * float(pad_size_um)


def _prepare_die_geometry_cache(
    top_die_w_um: float,
    top_die_h_um: float,
) -> Tuple[float, float]:
    """Return reusable half-size values for die corner comparisons."""
    return 0.5 * float(top_die_w_um), 0.5 * float(top_die_h_um)


def _active_pad_ids_from_bitmap(
    pad_coords_um: np.ndarray,
    dummy_pad_bitmap: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the original pad array plus the ids of pads that are not dummy pads."""
    pad_coords_um = np.asarray(pad_coords_um, dtype=np.float64)
    if pad_coords_um.ndim != 2 or pad_coords_um.shape[1] != 2:
        raise ValueError("pad_coords_um must have shape (n_pads, 2).")

    dummy_pad_bitmap = np.asarray(dummy_pad_bitmap, dtype=bool).reshape(-1)
    if pad_coords_um.shape[0] != dummy_pad_bitmap.shape[0]:
        raise ValueError("pad_coords_um and dummy_pad_bitmap must have the same length.")

    active_ids = np.flatnonzero(~dummy_pad_bitmap)
    if active_ids.size <= 0:
        raise ValueError("dummy_pad_bitmap masks out all pads.")
    return pad_coords_um, active_ids


def _candidate_pad_ids(
    top_dish_um_raw: np.ndarray,
    bot_dish_um: np.ndarray,
    arc_distance_um: float,
) -> np.ndarray:
    """Return pad ids that are close enough to enter first-touch competition."""
    arc_margin_um = max(0.0, float(arc_distance_um))
    return np.where((top_dish_um_raw + bot_dish_um) >= (-arc_margin_um))[0]


def _square_pad_min_gap_vec(
    *,
    cx_um: np.ndarray,
    cy_um: np.ndarray,
    half_pad_um: float,
    top_dish_um_raw: np.ndarray,
    bot_dish_um: np.ndarray,
    z_top_um: float,
    a: float,
    b: float,
    c: float,
    arc_distance_um: float,
) -> np.ndarray:
    """
    Return the exact minimum gap for each axis-aligned square pad.

    With the current model, z is linear in x/y, so the minimum over the four
    pad corners can be written analytically instead of expanding all corners.
    """
    corner_drop_um = float(half_pad_um) * (abs(float(a)) + abs(float(b)))
    return (
        float(z_top_um)
        + float(a) * np.asarray(cx_um, dtype=np.float64)
        + float(b) * np.asarray(cy_um, dtype=np.float64)
        + float(c) * (-np.asarray(top_dish_um_raw, dtype=np.float64))
        - np.asarray(bot_dish_um, dtype=np.float64)
        - max(0.0, float(arc_distance_um))
        - corner_drop_um
    )


def _rotate_and_min_choice(
    *,
    pad_coords_um: np.ndarray,
    pad_ids: np.ndarray,
    pad_size_um: float,
    top_die_w_um: float,
    top_die_h_um: float,
    top_dish_um_raw: np.ndarray,
    bot_dish_um: np.ndarray,
    tilt_x_deg: float,
    tilt_y_deg: float,
    z_top_um: float,
    rng_pick: np.random.Generator,
    pad_x_um: np.ndarray,
    pad_y_um: np.ndarray,
    half_pad_um: float,
    half_die_w_um: float,
    half_die_h_um: float,
    arc_distance_um: float = 0.0,
    atol: float = 1e-12,
) -> Tuple[int | None, bool, float]:
    """
    Apply tilt to the die and candidate pads and return the minimum-gap winner.

    The current geometry model is affine in x/y, so each square pad can use an
    exact analytical minimum over its four corners without explicitly storing
    those corner coordinates.
    """
    a, b, c = _z_linear_coeffs(tilt_x_deg, tilt_y_deg)
    die_gap = (
        float(z_top_um)
        - float(half_die_w_um) * abs(float(a))
        - float(half_die_h_um) * abs(float(b))
    )

    if pad_ids.size <= 0:
        return None, True, float(die_gap)

    pad_gaps = _square_pad_min_gap_vec(
        cx_um=pad_x_um[pad_ids],
        cy_um=pad_y_um[pad_ids],
        half_pad_um=float(half_pad_um),
        top_dish_um_raw=top_dish_um_raw[pad_ids],
        bot_dish_um=bot_dish_um[pad_ids],
        z_top_um=z_top_um,
        a=a,
        b=b,
        c=c,
        arc_distance_um=arc_distance_um,
    )

    min_gap = min(float(die_gap), float(np.min(pad_gaps)))
    is_best = np.isclose(pad_gaps, min_gap, rtol=0.0, atol=atol)
    if np.any(is_best):
        candidate_pad_ids = pad_ids[is_best]
        pick = int(rng_pick.integers(0, candidate_pad_ids.size))
        return int(candidate_pad_ids[pick]), False, float(min_gap)
    return None, True, float(min_gap)


def _best_pad_among_all_pads(
    *,
    pad_coords_um: np.ndarray,
    pad_size_um: float,
    top_dish_um_raw: np.ndarray,
    bot_dish_um: np.ndarray,
    tilt_x_deg: float,
    tilt_y_deg: float,
    z_top_um: float,
    rng_pick: np.random.Generator,
    pad_x_um: np.ndarray,
    pad_y_um: np.ndarray,
    half_pad_um: float,
    arc_distance_um: float = 0.0,
    atol_gap: float = 1e-12,
) -> Tuple[int, float]:
    """Choose the minimum-gap pad across all pads, without any candidate mask."""
    pad_count = pad_coords_um.shape[0]
    if pad_count <= 0:
        raise ValueError("pad_coords_um is empty; cannot choose a fallback pad.")

    a, b, c = _z_linear_coeffs(tilt_x_deg, tilt_y_deg)
    pad_min_gaps = _square_pad_min_gap_vec(
        cx_um=pad_x_um,
        cy_um=pad_y_um,
        half_pad_um=float(half_pad_um),
        top_dish_um_raw=top_dish_um_raw,
        bot_dish_um=bot_dish_um,
        z_top_um=z_top_um,
        a=a,
        b=b,
        c=c,
        arc_distance_um=arc_distance_um,
    )
    best_gap = float(np.min(pad_min_gaps))
    is_best = np.isclose(pad_min_gaps, best_gap, rtol=0.0, atol=atol_gap)
    best_ids = np.where(is_best)[0]

    if best_ids.size == 1:
        return int(best_ids[0]), best_gap
    pick = int(rng_pick.integers(0, best_ids.size))
    return int(best_ids[pick]), best_gap


def _binary_halving_until_pad(
    *,
    pad_coords_um: np.ndarray,
    pad_size_um: float,
    top_die_w_um: float,
    top_die_h_um: float,
    z_top_um: float,
    tilt_x_init_deg: float,
    tilt_y_init_deg: float,
    top_dish_um_raw: np.ndarray,
    bot_dish_um: np.ndarray,
    rng_pick: np.random.Generator,
    arc_distance_um: float = 0.0,
    atol_gap: float = 1e-12,
    atol_tilt_deg: float = 1e-12,
    max_iter_guard: int = 10000,
) -> Tuple[int, float, float, float]:
    """
    Find the first-touch pad.

    The normal path compares die corners and candidate pad corners under the
    current tilt. If the result is still die-only, the tilt is halved until a
    pad appears or the stopping guard is hit. The final fallback always chooses
    the all-pad minimum-gap winner under zero tilt.
    """
    pad_x_um, pad_y_um, half_pad_um = _prepare_pad_geometry_cache(pad_coords_um, pad_size_um)
    half_die_w_um, half_die_h_um = _prepare_die_geometry_cache(top_die_w_um, top_die_h_um)
    candidate_pad_ids = _candidate_pad_ids(top_dish_um_raw, bot_dish_um, arc_distance_um)

    tilt_x = float(tilt_x_init_deg)
    tilt_y = float(tilt_y_init_deg)

    if candidate_pad_ids.size <= 0:
        best_pad, best_gap = _best_pad_among_all_pads(
            pad_coords_um=pad_coords_um,
            pad_size_um=pad_size_um,
            top_dish_um_raw=top_dish_um_raw,
            bot_dish_um=bot_dish_um,
            tilt_x_deg=0.0,
            tilt_y_deg=0.0,
            z_top_um=z_top_um,
            rng_pick=rng_pick,
            pad_x_um=pad_x_um,
            pad_y_um=pad_y_um,
            half_pad_um=half_pad_um,
            arc_distance_um=arc_distance_um,
            atol_gap=atol_gap,
        )
        return best_pad, tilt_x, tilt_y, float(best_gap)

    pad_choice, die_only, min_gap = _rotate_and_min_choice(
        pad_coords_um=pad_coords_um,
        pad_ids=candidate_pad_ids,
        pad_size_um=pad_size_um,
        top_die_w_um=top_die_w_um,
        top_die_h_um=top_die_h_um,
        top_dish_um_raw=top_dish_um_raw,
        bot_dish_um=bot_dish_um,
        tilt_x_deg=tilt_x,
        tilt_y_deg=tilt_y,
        z_top_um=z_top_um,
        rng_pick=rng_pick,
        pad_x_um=pad_x_um,
        pad_y_um=pad_y_um,
        half_pad_um=half_pad_um,
        half_die_w_um=half_die_w_um,
        half_die_h_um=half_die_h_um,
        arc_distance_um=arc_distance_um,
        atol=atol_gap,
    )
    if not die_only:
        return int(pad_choice), tilt_x, tilt_y, float(min_gap)

    iterations = 0
    while die_only:
        tilt_x *= 0.5
        tilt_y *= 0.5
        pad_choice, die_only, min_gap = _rotate_and_min_choice(
            pad_coords_um=pad_coords_um,
            pad_ids=candidate_pad_ids,
            pad_size_um=pad_size_um,
            top_die_w_um=top_die_w_um,
            top_die_h_um=top_die_h_um,
            top_dish_um_raw=top_dish_um_raw,
            bot_dish_um=bot_dish_um,
            tilt_x_deg=tilt_x,
            tilt_y_deg=tilt_y,
            z_top_um=z_top_um,
            rng_pick=rng_pick,
            pad_x_um=pad_x_um,
            pad_y_um=pad_y_um,
            half_pad_um=half_pad_um,
            half_die_w_um=half_die_w_um,
            half_die_h_um=half_die_h_um,
            arc_distance_um=arc_distance_um,
            atol=atol_gap,
        )
        iterations += 1

        if not die_only:
            return int(pad_choice), tilt_x, tilt_y, float(min_gap)

        if (abs(tilt_x) <= atol_tilt_deg and abs(tilt_y) <= atol_tilt_deg) or (iterations >= max_iter_guard):
            best_pad, best_gap = _best_pad_among_all_pads(
                pad_coords_um=pad_coords_um,
                pad_size_um=pad_size_um,
                top_dish_um_raw=top_dish_um_raw,
                bot_dish_um=bot_dish_um,
                tilt_x_deg=0.0,
                tilt_y_deg=0.0,
                z_top_um=z_top_um,
                rng_pick=rng_pick,
                pad_x_um=pad_x_um,
                pad_y_um=pad_y_um,
                half_pad_um=half_pad_um,
                arc_distance_um=arc_distance_um,
                atol_gap=atol_gap,
            )
            return best_pad, tilt_x, tilt_y, float(best_gap)

    best_pad, best_gap = _best_pad_among_all_pads(
        pad_coords_um=pad_coords_um,
        pad_size_um=pad_size_um,
        top_dish_um_raw=top_dish_um_raw,
        bot_dish_um=bot_dish_um,
        tilt_x_deg=0.0,
        tilt_y_deg=0.0,
        z_top_um=z_top_um,
        rng_pick=rng_pick,
        pad_x_um=pad_x_um,
        pad_y_um=pad_y_um,
        half_pad_um=half_pad_um,
        arc_distance_um=arc_distance_um,
        atol_gap=atol_gap,
    )
    return best_pad, tilt_x, tilt_y, float(best_gap)

def esd_failure_simulator(
    *,
    cfg,
    pad_coords_um: np.ndarray,
    top_dish_nm_ext: np.ndarray,
    bot_dish_nm_ext: np.ndarray,
    dummy_pad_bitmap: np.ndarray,
    pad_size_um: float,
    top_die_w_um: float,
    top_die_h_um: float,
    tilt_x_mean_deg: float,
    tilt_x_std_deg: float,
    tilt_y_mean_deg: float,
    tilt_y_std_deg: float,
    base_seed: int,
) -> Tuple[int | None, bool]:
    """Run a single stochastic experiment and return (first_touch_pad, survive_bool)."""
    pad_size_um = float(pad_size_um)
    top_die_w_um = float(top_die_w_um)
    top_die_h_um = float(top_die_h_um)
    tilt_x_mean_deg = float(tilt_x_mean_deg)
    tilt_x_std_deg = float(tilt_x_std_deg)
    tilt_y_mean_deg = float(tilt_y_mean_deg)
    tilt_y_std_deg = float(tilt_y_std_deg)
    base_seed = int(base_seed)

    z_top_um = Z_TOP_UM
    v_min_v = float(cfg.V_MIN_V)
    v_max_v = float(cfg.V_MAX_V)
    weibull_k = float(cfg.WEIBULL_K)
    weibull_lambda = float(cfg.WEIBULL_LAMBDA)
    cutoff_min_a = float(cfg.CUTOFF_MIN_A)

    pad_coords_um, active_ids = _active_pad_ids_from_bitmap(pad_coords_um, dummy_pad_bitmap)
    active_pad_coords_um = pad_coords_um[active_ids]
    top_dish_nm_ext = np.asarray(top_dish_nm_ext, dtype=np.float64).reshape(-1)
    bot_dish_nm_ext = np.asarray(bot_dish_nm_ext, dtype=np.float64).reshape(-1)

    if not (
        pad_coords_um.shape[0] == top_dish_nm_ext.shape[0] == bot_dish_nm_ext.shape[0]
    ):
        raise ValueError("pad_coords_um, top_dish_nm_ext, and bot_dish_nm_ext must have the same length.")

    rng = np.random.default_rng(base_seed ^ 0xA5A5A5A5)
    rng_pick = np.random.default_rng((base_seed ^ 0xA5A5A5A5) ^ 0xDEADBEEF)

    tilt_x = float(rng.normal(tilt_x_mean_deg, tilt_x_std_deg))
    tilt_y = float(rng.normal(tilt_y_mean_deg, tilt_y_std_deg))
    v_chg = float(rng.uniform(v_min_v, v_max_v))
    arc_distance_um = _arc_distance_um_from_voltage(v_chg)

    top_dish_um_raw = top_dish_nm_ext[active_ids] * 1e-3
    bot_dish_um = bot_dish_nm_ext[active_ids] * 1e-3

    pad_choice_active, _, _, _ = _binary_halving_until_pad(
        pad_coords_um=active_pad_coords_um,
        pad_size_um=pad_size_um,
        top_die_w_um=top_die_w_um,
        top_die_h_um=top_die_h_um,
        z_top_um=z_top_um,
        tilt_x_init_deg=tilt_x,
        tilt_y_init_deg=tilt_y,
        top_dish_um_raw=top_dish_um_raw,
        bot_dish_um=bot_dish_um,
        rng_pick=rng_pick,
        arc_distance_um=arc_distance_um,
    )

    pad_choice = int(active_ids[int(pad_choice_active)]) if pad_choice_active is not None else None
    p_fail_single = _compute_p_fail_for_die(
        top_die_w_um,
        top_die_h_um,
        v_chg,
        weibull_k=weibull_k,
        weibull_lambda=weibull_lambda,
        cutoff_min_a=cutoff_min_a,
    )
    survive_bool = not ((pad_choice is not None) and (float(rng.uniform(0.0, 1.0)) < p_fail_single))
    return pad_choice, survive_bool


if __name__ == "__main__":
    raise SystemExit("esd_yield_simulator.py expects external cfg and pad inputs; import this module from the D2W flow.")
