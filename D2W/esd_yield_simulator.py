# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Tuple
import numpy as np

EPS0_F_PER_M = 8.8541878128e-12


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


def _cfg_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null", "nan"}
    return False


def _cfg_first(cfg, keys, default=None):
    for key in keys:
        if hasattr(cfg, "get"):
            value = cfg.get(key, None)
        else:
            value = getattr(cfg, key, None)
        if not _cfg_missing(value):
            return value
    return default


def _cfg_float(cfg, keys, default):
    value = _cfg_first(cfg, keys, default)
    if _cfg_missing(value):
        return float(default)
    return float(value)


def _cfg_str(cfg, keys, default):
    value = _cfg_first(cfg, keys, default)
    if _cfg_missing(value):
        return str(default)
    return str(value)


def _cfg_length_um(cfg, um_keys, m_keys, default_um):
    value = _cfg_first(cfg, um_keys, None)
    if not _cfg_missing(value):
        return float(value)
    value = _cfg_first(cfg, m_keys, None)
    if not _cfg_missing(value):
        return float(value) * 1e6
    return float(default_um)


def _rect_prism_surface_area_um2(length_um: float, width_um: float, thickness_um: float) -> float:
    length_um = float(length_um)
    width_um = float(width_um)
    thickness_um = float(thickness_um)
    if length_um <= 0.0 or width_um <= 0.0 or thickness_um <= 0.0:
        raise ValueError("ESD die length, width, and thickness must be positive.")
    return float(2.0 * (length_um * width_um + length_um * thickness_um + width_um * thickness_um))


def _self_capacitance_from_surface(surface_area_um2: float, cf: float, epsr: float) -> float:
    surface_area_m2 = max(float(surface_area_um2), 0.0) * 1e-12
    if surface_area_m2 <= 0.0:
        return 0.0
    return float(float(cf) * EPS0_F_PER_M * float(epsr) * math.sqrt(4.0 * math.pi * surface_area_m2))


def _floating_stack_ipeak_from_cfg(
    cfg,
    *,
    top_die_w_um: float,
    top_die_l_um: float,
    v_chg: float,
    geff_um: float | None,
) -> float:
    v_delta = abs(float(v_chg))
    if v_delta <= 0.0:
        return 0.0

    top_die_w_um = float(top_die_w_um)
    top_die_l_um = float(top_die_l_um)
    top_t_um = _cfg_length_um(
        cfg,
        ["ESD_TOP_THICK_um", "ITF_TOP_THICK_um"],
        ["T_Sub_T"],
        default_um=1.0,
    )
    bottom_t_um = _cfg_length_um(
        cfg,
        ["ESD_BOTTOM_SUBSTACK_THICK_um", "ITF_BOT_THICK_um"],
        ["B_Sub_T"],
        default_um=top_t_um,
    )

    top_surface_um2 = _cfg_float(
        cfg,
        ["ESD_TOP_SURFACE_AREA_um2"],
        _rect_prism_surface_area_um2(top_die_l_um, top_die_w_um, top_t_um),
    )
    bottom_surface_um2 = _cfg_float(
        cfg,
        ["ESD_BOTTOM_SUBSTACK_SURFACE_AREA_um2"],
        _rect_prism_surface_area_um2(top_die_l_um, top_die_w_um, bottom_t_um),
    )
    overlap_area_um2 = _cfg_float(
        cfg,
        ["ESD_OVERLAP_AREA_um2"],
        top_die_l_um * top_die_w_um,
    )

    cf_die = _cfg_float(cfg, ["Cf_die", "ESD_CF_DIE", "CF_DIE"], 1.0)
    cf_wafer = _cfg_float(cfg, ["Cf_wafer", "ESD_CF_WAFER", "CF_WAFER"], cf_die)
    epsr = _cfg_float(cfg, ["epsr", "ESD_EPSR", "EPSR"], 1.0)
    kc = _cfg_float(cfg, ["Kc", "ESD_KC", "KC"], 1.0e-6)
    bottom_kind = _cfg_str(cfg, ["ESD_BOTTOM_BODY_KIND"], "die").strip().lower()
    cf_bottom = cf_wafer if bottom_kind in {"wafer", "root", "substrate"} else cf_die

    cs_top = _self_capacitance_from_surface(top_surface_um2, cf_die, epsr)
    cs_bottom = _self_capacitance_from_surface(bottom_surface_um2, cf_bottom, epsr)
    ceq_self = 0.0
    if cs_top > 0.0 and cs_bottom > 0.0:
        ceq_self = (cs_top * cs_bottom) / (cs_top + cs_bottom)

    if geff_um is None:
        geff_um = v_delta / 97.0
    min_geff_um = _cfg_float(cfg, ["ESD_MIN_GEFF_um"], 1.0e-3)
    geff_m = max(float(geff_um), float(min_geff_um)) * 1e-6
    cm = float(kc) * EPS0_F_PER_M * float(epsr) * max(float(overlap_area_um2), 0.0) * 1e-12 / geff_m
    ceff_f = ceq_self + cm
    if ceff_f <= 0.0:
        return 0.0

    l_mm = max(top_die_l_um * 1e-3, 1.0e-12)
    w_mm = max(top_die_w_um * 1e-3, 1.0e-12)
    t_l_mm = max(top_t_um * 1e-3, 1.0e-12)
    l0_mm = _cfg_float(cfg, ["l0", "ESD_l0_mm", "ESD_L0_REF_MM"], 1.0)
    m_exp = _cfg_float(cfg, ["m", "ESD_M"], 0.5)
    l0_mm = max(float(l0_mm), 1.0e-12)
    if m_exp < 0.0 or m_exp >= 1.0:
        raise ValueError("ESD inductance exponent m must satisfy 0 <= m < 1.")

    l0_nh = _cfg_float(cfg, ["L0", "ESD_L0_NH"], 0.1)
    kl = _cfg_float(cfg, ["Kl", "ESD_KL"], 1.0)
    log_arg = max(2.0 * l_mm / (w_mm + t_l_mm), 1.0e-12)
    aspect = (w_mm + t_l_mm) / l_mm
    bracket = math.log(log_arg) + 0.2235 * aspect + 0.5
    inductance_nh = float(l0_nh) + 0.2 * float(kl) * ((l_mm / l0_mm) ** float(m_exp)) * bracket
    if inductance_nh <= 0.0:
        raise ValueError("ESD effective inductance must be positive.")

    return float(v_delta * math.sqrt(ceff_f / (inductance_nh * 1e-9)))


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
    cfg=None,
    geff_um: float | None = None,
    weibull_k: float,
    weibull_lambda: float,
    cutoff_min_a: float,
) -> float:
    """Return the die-level failure probability for a sampled charging voltage."""
    if cfg is None:
        raise ValueError("cfg is required for the floating-stack ESD Ipeak model.")
    i_peak = _floating_stack_ipeak_from_cfg(
        cfg,
        top_die_w_um=float(top_die_w_um),
        top_die_l_um=float(top_die_h_um),
        v_chg=float(v_chg),
        geff_um=geff_um,
    )
    return _fail_prob_single(i_peak, float(weibull_k), float(weibull_lambda), float(cutoff_min_a))


def _arc_distance_um_from_voltage(v_chg: float, cfg=None) -> float:
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

    plateau_v = _cfg_float(cfg, ["ESD_ARC_PLATEAU_V"], 337.0)
    small_gap_slope = _cfg_float(cfg, ["ESD_ARC_SMALL_GAP_SLOPE_V_PER_UM"], 97.0)
    plateau_upper_gap_um = _cfg_float(cfg, ["ESD_ARC_PLATEAU_UPPER_GAP_UM"], 7.0)

    if v_chg < plateau_v:
        return v_chg / small_gap_slope

    a = _cfg_float(cfg, ["ESD_ARC_LARGE_GAP_LINEAR_COEFF"], 2.48)
    b = _cfg_float(cfg, ["ESD_ARC_LARGE_GAP_SQRT_COEFF"], 58.0)
    c = _cfg_float(cfg, ["ESD_ARC_LARGE_GAP_OFFSET_V"], 170.0) - v_chg
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
) -> Tuple[int | None, bool]:
    """Run a single stochastic experiment and return (first_touch_pad, survive_bool)."""
    pad_size_um = float(pad_size_um)
    top_die_w_um = float(top_die_w_um)
    top_die_h_um = float(top_die_h_um)
    tilt_x_mean_deg = float(tilt_x_mean_deg)
    tilt_x_std_deg = float(tilt_x_std_deg)
    tilt_y_mean_deg = float(tilt_y_mean_deg)
    tilt_y_std_deg = float(tilt_y_std_deg)

    z_top_um = _cfg_float(cfg, ["ESD_Z_TOP_UM"], 0.1)
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

    rng = np.random.default_rng()
    rng_pick = np.random.default_rng()

    tilt_x = float(rng.normal(tilt_x_mean_deg, tilt_x_std_deg))
    tilt_y = float(rng.normal(tilt_y_mean_deg, tilt_y_std_deg))
    v_chg = float(rng.uniform(v_min_v, v_max_v))
    arc_distance_um = _arc_distance_um_from_voltage(v_chg, cfg=cfg)

    top_dish_um_raw = top_dish_nm_ext[active_ids] * 1e-3
    bot_dish_um = bot_dish_nm_ext[active_ids] * 1e-3

    pad_choice_active, _, _, arc_margin_gap_um = _binary_halving_until_pad(
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
    local_gap_um = float(arc_margin_gap_um) + float(arc_distance_um)
    p_fail_single = _compute_p_fail_for_die(
        top_die_w_um,
        top_die_h_um,
        v_chg,
        cfg=cfg,
        geff_um=local_gap_um,
        weibull_k=weibull_k,
        weibull_lambda=weibull_lambda,
        cutoff_min_a=cutoff_min_a,
    )
    survive_bool = not ((pad_choice is not None) and (float(rng.uniform(0.0, 1.0)) < p_fail_single))
    return pad_choice, survive_bool


if __name__ == "__main__":
    raise SystemExit("esd_yield_simulator.py expects external cfg and pad inputs; import this module from the D2W flow.")
