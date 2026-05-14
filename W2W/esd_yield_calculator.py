#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.special import log_ndtr

EPS0_F_PER_M = 8.8541878128e-12


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


def _cfg_bool(cfg, keys, default=False) -> bool:
    value = _cfg_first(cfg, keys, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _dish_std_nm_from_cfg(cfg, side: str) -> float:
    side = str(side).upper()
    scalar = _cfg_first(cfg, [f"{side}_DISH_STD_nm"], None)
    if not _cfg_missing(scalar):
        return float(scalar)

    components = [
        _cfg_float(cfg, [f"{side}_DISH_STD_L_nm"], 0.0),
        _cfg_float(cfg, [f"{side}_DISH_STD_T_nm"], 0.0),
        _cfg_float(cfg, [f"{side}_DISH_STD_E_nm"], 0.0),
    ]
    return float(math.sqrt(sum(max(value, 0.0) ** 2 for value in components)))


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

    cf_die = _cfg_float(cfg, ["Cf_die", "ESD_CF_DIE", "CF_DIE"], 0.904)
    cf_wafer = _cfg_float(cfg, ["Cf_wafer", "ESD_CF_WAFER", "CF_WAFER"], cf_die)
    epsr = _cfg_float(cfg, ["epsr", "ESD_EPSR", "EPSR"], 1.0)
    kc = _cfg_float(cfg, ["Kc", "ESD_KC", "KC"], 9.67230332132e-06)
    bottom_kind = _cfg_str(cfg, ["ESD_BOTTOM_BODY_KIND"], "wafer").strip().lower()
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
    l0_mm = max(_cfg_float(cfg, ["l0", "ESD_l0_mm", "ESD_L0_REF_MM"], 0.0197457261531), 1.0e-12)
    m_exp = _cfg_float(cfg, ["m", "ESD_M"], 0.5)
    if m_exp < 0.0 or m_exp >= 1.0:
        raise ValueError("ESD inductance exponent m must satisfy 0 <= m < 1.")

    l0_nh = _cfg_float(cfg, ["L0", "ESD_L0_NH"], 1.01907739978e-12)
    kl = _cfg_float(cfg, ["Kl", "ESD_KL"], 1.0)
    log_arg = max(2.0 * l_mm / (w_mm + t_l_mm), 1.0e-12)
    aspect = (w_mm + t_l_mm) / l_mm
    bracket = math.log(log_arg) + 0.2235 * aspect + 0.5
    inductance_nh = float(l0_nh) + 0.2 * float(kl) * ((l_mm / l0_mm) ** float(m_exp)) * bracket
    if inductance_nh <= 0.0:
        raise ValueError("ESD effective inductance must be positive.")

    return float(v_delta * math.sqrt(ceff_f / (inductance_nh * 1e-9)))


def _weibull_cdf(current_a: float, k: float, lam: float) -> float:
    current_a = max(current_a, 1e-12)
    return max(0.0, min(1.0, 1.0 - math.exp(-((current_a / lam) ** k))))


def _fail_prob_single(current_a: float, k: float, lam: float, cutoff_a: float) -> float:
    if current_a < cutoff_a:
        return 0.0
    return _weibull_cdf(current_a, k, lam)


def _compute_p_fail_for_die(
    top_die_w_um: float,
    top_die_h_um: float,
    v_chg: float,
    *,
    cfg,
    geff_um: float | None,
    weibull_k: float,
    weibull_lambda: float,
    cutoff_min_a: float,
) -> float:
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
    disc = b * b - 4.0 * a * c
    if disc <= 0.0:
        return plateau_upper_gap_um

    root = (-b + math.sqrt(disc)) / (2.0 * a)
    if root <= 0.0:
        return plateau_upper_gap_um
    return max(plateau_upper_gap_um, root * root)


def _legendre_quadrature_interval(q: int, low: float, high: float) -> Tuple[np.ndarray, np.ndarray]:
    x, w = leggauss(int(q))
    g = 0.5 * (x + 1.0) * (high - low) + low
    wg = 0.5 * (high - low) * w
    return g.astype(np.float64), wg.astype(np.float64)


def _normal_quadrature(mean: float, std: float, q: int) -> Tuple[np.ndarray, np.ndarray]:
    if float(std) <= 0.0 or int(q) <= 1:
        return np.array([float(mean)], dtype=np.float64), np.array([1.0], dtype=np.float64)
    nodes, weights = hermgauss(int(q))
    values = float(mean) + math.sqrt(2.0) * float(std) * nodes
    weights = weights / math.sqrt(math.pi)
    return values.astype(np.float64), weights.astype(np.float64)


def _w2w_warpage_mean_um_from_cfg(cfg) -> float:
    value_um = _cfg_first(
        cfg,
        [
            "ESD_W2W_WARPAGE_UM",
            "W2W_ESD_WARPAGE_UM",
            "TOP_WAFER_WARPAGE_UM",
            "WAFER_WARPAGE_UM",
        ],
        None,
    )
    if not _cfg_missing(value_um):
        return abs(float(value_um))
    value_m = _cfg_first(
        cfg,
        [
            "ESD_W2W_WARPAGE_M",
            "W2W_ESD_WARPAGE_M",
            "TOP_WAFER_WARPAGE_M",
            "S_INIT_A_M",
        ],
        None,
    )
    if not _cfg_missing(value_m):
        return abs(float(value_m) * 1e6)
    return 0.0


def _w2w_warpage_std_um_from_cfg(cfg) -> float:
    value_um = _cfg_first(
        cfg,
        [
            "ESD_W2W_WARPAGE_STD_UM",
            "W2W_ESD_WARPAGE_STD_UM",
            "TOP_WAFER_WARPAGE_STD_UM",
            "WAFER_WARPAGE_STD_UM",
        ],
        None,
    )
    if not _cfg_missing(value_um):
        return max(float(value_um), 0.0)
    value_m = _cfg_first(
        cfg,
        [
            "ESD_W2W_WARPAGE_STD_M",
            "W2W_ESD_WARPAGE_STD_M",
            "TOP_WAFER_WARPAGE_STD_M",
            "S_INIT_A_STD_M",
        ],
        None,
    )
    if not _cfg_missing(value_m):
        return max(float(value_m) * 1e6, 0.0)
    return 0.0


def _spherical_cap_gap_um(
    x_um: np.ndarray,
    y_um: np.ndarray,
    *,
    wafer_radius_um: float,
    warpage_um: float,
    exact: bool,
) -> np.ndarray:
    x_um = np.asarray(x_um, dtype=np.float64)
    y_um = np.asarray(y_um, dtype=np.float64)
    wafer_radius_um = float(wafer_radius_um)
    warpage_um = abs(float(warpage_um))
    if wafer_radius_um <= 0.0 or warpage_um <= 0.0:
        return np.zeros_like(x_um, dtype=np.float64)

    r2 = x_um * x_um + y_um * y_um
    if (not exact) or (warpage_um / wafer_radius_um < 1.0e-7):
        return warpage_um * r2 / (wafer_radius_um * wafer_radius_um)

    sphere_radius_um = (wafer_radius_um * wafer_radius_um + warpage_um * warpage_um) / (2.0 * warpage_um)
    inside = np.maximum(sphere_radius_um * sphere_radius_um - r2, 0.0)
    sqrt_inside = np.sqrt(inside)
    denom = np.maximum(sphere_radius_um + sqrt_inside, 1.0e-12)
    return r2 / denom


def _w2w_pad_contact_limit_um(
    *,
    pad_coords_um: np.ndarray,
    pad_size_um: float,
    wafer_radius_um: float,
    warpage_um: float,
    z_top_um: float,
    exact_sphere: bool,
) -> np.ndarray:
    """
    Return C_i = g0 + min_corner(d_W2W) for each pad in wafer coordinates.
    """
    pad_coords_um = np.asarray(pad_coords_um, dtype=np.float64)
    if pad_coords_um.ndim != 2 or pad_coords_um.shape[1] != 2:
        raise ValueError("pad_coords_um must have shape (n_pads, 2).")

    half = 0.5 * float(pad_size_um)
    cx = pad_coords_um[:, 0]
    cy = pad_coords_um[:, 1]
    x4 = np.stack([cx - half, cx + half, cx + half, cx - half], axis=1)
    y4 = np.stack([cy - half, cy - half, cy + half, cy + half], axis=1)
    d4 = _spherical_cap_gap_um(
        x4,
        y4,
        wafer_radius_um=wafer_radius_um,
        warpage_um=warpage_um,
        exact=exact_sphere,
    )
    return (float(z_top_um) + np.min(d4, axis=1)).astype(np.float64)


def _fixed_w2w_probability_map_with_arcing(
    *,
    contact_limit_um: np.ndarray,
    mu_h_um: float,
    sigma_h_um: float,
    arc_distance_um: float,
    quadrature_points: int,
    tail_sigma: float,
    chunk_size: int,
    fill_residual_uniformly: bool,
) -> np.ndarray:
    contact_limit_um = np.asarray(contact_limit_um, dtype=np.float64).reshape(-1)
    pad_count = contact_limit_um.size
    if pad_count <= 0:
        return np.zeros((0,), dtype=np.float64)
    if sigma_h_um <= 0.0:
        raise ValueError("Combined dishing sigma must be positive for analytical ESD yield calculation.")

    mean_gap_um = contact_limit_um - float(arc_distance_um) - float(mu_h_um)
    low = float(np.min(mean_gap_um) - float(tail_sigma) * float(sigma_h_um))
    high = float(np.max(contact_limit_um))
    if high <= low:
        high = float(np.max(mean_gap_um) + float(tail_sigma) * float(sigma_h_um))

    g_nodes, g_weights = _legendre_quadrature_interval(int(quadrature_points), low, high)
    prob = np.zeros((pad_count,), dtype=np.float64)
    inactive_log_prob = float(log_ndtr((float(-arc_distance_um) - float(mu_h_um)) / float(sigma_h_um)))
    log_norm = -math.log(float(sigma_h_um)) - 0.5 * math.log(2.0 * math.pi)
    chunk_size = max(1, int(chunk_size))

    for g, w in zip(g_nodes, g_weights):
        valid_mask = g <= contact_limit_um
        if not np.any(valid_mask):
            continue

        log_survival = np.where(
            valid_mask,
            log_ndtr((mean_gap_um - float(g)) / float(sigma_h_um)),
            inactive_log_prob,
        )
        total_log_survival = float(np.sum(log_survival))
        logw = math.log(float(w))

        for start in range(0, pad_count, chunk_size):
            end = min(start + chunk_size, pad_count)
            local_valid = valid_mask[start:end]
            if not np.any(local_valid):
                continue

            local_mean = mean_gap_um[start:end]
            t = (float(g) - local_mean) / float(sigma_h_um)
            logf = -0.5 * t * t + log_norm
            log_integrand = logw + logf + total_log_survival - log_survival[start:end]
            prob[start:end][local_valid] += np.exp(log_integrand[local_valid])

    prob_sum = float(np.sum(prob))
    if prob_sum <= 0.0:
        prob.fill(1.0 / float(pad_count))
        return prob

    if prob_sum < 1.0 and fill_residual_uniformly:
        prob += (1.0 - prob_sum) / float(pad_count)
        return prob

    prob /= prob_sum
    return prob


def _fixed_w2w_critical_probability_with_arcing(
    *,
    contact_limit_um: np.ndarray,
    critical_mask: np.ndarray,
    mu_h_um: float,
    sigma_h_um: float,
    arc_distance_um: float,
    quadrature_points: int,
    tail_sigma: float,
    chunk_size: int,
    fill_residual_uniformly: bool,
) -> float:
    contact_limit_um = np.asarray(contact_limit_um, dtype=np.float64).reshape(-1)
    critical_mask = np.asarray(critical_mask, dtype=bool).reshape(-1)
    pad_count = contact_limit_um.size
    if critical_mask.size != pad_count:
        raise ValueError("critical_mask must have the same length as contact_limit_um.")
    if pad_count <= 0:
        return 0.0
    critical_count = int(np.count_nonzero(critical_mask))
    if critical_count <= 0:
        return 0.0
    if critical_count == pad_count:
        return 1.0
    if sigma_h_um <= 0.0:
        raise ValueError("Combined dishing sigma must be positive for analytical ESD yield calculation.")

    prob = _fixed_w2w_probability_map_with_arcing(
        contact_limit_um=contact_limit_um,
        mu_h_um=mu_h_um,
        sigma_h_um=sigma_h_um,
        arc_distance_um=arc_distance_um,
        quadrature_points=quadrature_points,
        tail_sigma=tail_sigma,
        chunk_size=chunk_size,
        fill_residual_uniformly=fill_residual_uniformly,
    )
    return float(np.clip(np.sum(prob[critical_mask]), 0.0, 1.0))


def _select_candidate_pad_indices(
    *,
    contact_limit_um: np.ndarray,
    sigma_h_um: float,
    candidate_sigma_window: float,
    candidate_min_pads: int,
    candidate_disable_fraction: float,
) -> np.ndarray:
    contact_limit_um = np.asarray(contact_limit_um, dtype=np.float64).reshape(-1)
    pad_count = contact_limit_um.size
    if pad_count <= 0:
        return np.zeros((0,), dtype=np.int64)

    if candidate_sigma_window <= 0.0 or sigma_h_um <= 0.0:
        return np.arange(pad_count, dtype=np.int64)

    min_limit = float(np.min(contact_limit_um))
    threshold = min_limit + float(candidate_sigma_window) * float(sigma_h_um)
    candidate_idx = np.flatnonzero(contact_limit_um <= threshold)

    min_pads = max(1, min(int(candidate_min_pads), pad_count))
    if candidate_idx.size < min_pads:
        candidate_idx = np.argpartition(contact_limit_um, min_pads - 1)[:min_pads]

    if candidate_idx.size / float(pad_count) >= float(candidate_disable_fraction):
        return np.arange(pad_count, dtype=np.int64)

    return np.sort(candidate_idx.astype(np.int64, copy=False))


def center_die_indices(die_list: Sequence, tolerance_um: float | None = None) -> np.ndarray:
    """
    Return the 1/2/4 dies that share the wafer-center first-contact location.

    The generated W2W die grid is symmetric around the wafer center.  If the
    center lies inside one die, this returns one index.  If it lies on a street
    between two or four dies, the symmetric nearest die centers are returned.
    """
    if not die_list:
        return np.zeros((0,), dtype=np.int64)

    centers = np.asarray([die.die_center for die in die_list], dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("Each die must expose a two-element die_center.")

    die_w_um = float(getattr(die_list[0], "DIE_W_um", 1.0))
    die_l_um = float(getattr(die_list[0], "DIE_L_um", 1.0))
    if tolerance_um is None:
        tolerance_um = max(1.0e-6, max(abs(die_w_um), abs(die_l_um), 1.0) * 1.0e-9)
    tolerance_um = max(float(tolerance_um), 0.0)

    min_abs_x = float(np.min(np.abs(centers[:, 0])))
    min_abs_y = float(np.min(np.abs(centers[:, 1])))
    mask = (
        np.isclose(np.abs(centers[:, 0]), min_abs_x, rtol=0.0, atol=tolerance_um)
        & np.isclose(np.abs(centers[:, 1]), min_abs_y, rtol=0.0, atol=tolerance_um)
    )
    idx = np.flatnonzero(mask).astype(np.int64)
    if idx.size > 4:
        order = np.argsort(np.linalg.norm(centers[idx], axis=1))
        idx = idx[order[:4]]
    return np.sort(idx)


def center_contact_case(die_list: Sequence, tolerance_um: float | None = None) -> str:
    count = int(center_die_indices(die_list, tolerance_um=tolerance_um).size)
    if count == 1:
        return "center_on_one_die"
    if count == 2:
        return "center_between_two_dies"
    if count == 4:
        return "center_between_four_dies"
    return f"center_ambiguous_{count}_dies"


def die_esd_yield_calculator(
    *,
    cfg,
    pad_coords_um: np.ndarray,
    esd_critical_pad_mask: np.ndarray,
    pad_size_um: float,
    top_die_w_um: float,
    top_die_h_um: float,
    wafer_radius_um: float,
    top_dish_mean_nm: float,
    top_dish_std_nm: float,
    bot_dish_mean_nm: float,
    bot_dish_std_nm: float,
    warpage_mean_um: float | None = None,
    warpage_std_um: float | None = None,
    z_top_um=None,
) -> float:
    """
    Return the conditional ESD yield for a center-candidate W2W die.

    ``pad_coords_um`` must be wafer-level coordinates. All supplied pads compete
    for first arcing, while ``esd_critical_pad_mask`` decides which first-arcing
    events become die-level ESD failures.
    """
    pad_coords_um = np.asarray(pad_coords_um, dtype=np.float64)
    esd_critical_pad_mask = np.asarray(esd_critical_pad_mask, dtype=bool).reshape(-1)
    if pad_coords_um.ndim != 2 or pad_coords_um.shape[1] != 2:
        raise ValueError("pad_coords_um must have shape (n_pads, 2).")
    pad_count = pad_coords_um.shape[0]
    if esd_critical_pad_mask.size != pad_count:
        raise ValueError("esd_critical_pad_mask must have the same length as pad_coords_um.")
    if pad_count <= 0:
        raise ValueError("pad_coords_um is empty; analytical ESD yield calculation needs at least one pad.")
    if not np.any(esd_critical_pad_mask):
        return 1.0

    pad_size_um = float(pad_size_um)
    top_die_w_um = float(top_die_w_um)
    top_die_h_um = float(top_die_h_um)
    wafer_radius_um = float(wafer_radius_um)
    if z_top_um is None:
        z_top_um = _cfg_float(cfg, ["ESD_Z_TOP_UM"], 0.1)
    else:
        z_top_um = float(z_top_um)

    v_min_v = _cfg_float(cfg, ["V_MIN_V"], 0.0)
    v_max_v = _cfg_float(cfg, ["V_MAX_V"], 5.0)
    weibull_k = _cfg_float(cfg, ["WEIBULL_K"], 4.44985)
    weibull_lambda = _cfg_float(cfg, ["WEIBULL_LAMBDA"], 0.0621816)
    cutoff_min_a = _cfg_float(cfg, ["CUTOFF_MIN_A"], 0.0)

    quadrature_points = int(_cfg_float(cfg, ["ESD_ANALYTICAL_INNER_Q"], 48))
    voltage_q = int(_cfg_float(cfg, ["ESD_ANALYTICAL_VOLTAGE_Q"], 5))
    warpage_q = int(_cfg_float(cfg, ["ESD_W2W_WARPAGE_Q", "W2W_ESD_WARPAGE_Q"], 3))
    tail_sigma = _cfg_float(cfg, ["ESD_ANALYTICAL_TAIL_SIGMA"], 8.0)
    chunk_size = int(_cfg_float(cfg, ["ESD_ANALYTICAL_CHUNK_SIZE"], 100000))
    fill_residual_uniformly = _cfg_bool(cfg, ["ESD_ANALYTICAL_FILL_RESIDUAL_UNIFORMLY"], True)
    candidate_sigma_window = _cfg_float(cfg, ["ESD_ANALYTICAL_CANDIDATE_SIGMA_WINDOW"], 8.0)
    candidate_min_pads = int(_cfg_float(cfg, ["ESD_ANALYTICAL_CANDIDATE_MIN_PADS"], 4096))
    candidate_disable_fraction = _cfg_float(cfg, ["ESD_ANALYTICAL_CANDIDATE_DISABLE_FRACTION"], 0.8)
    exact_sphere = _cfg_bool(cfg, ["ESD_W2W_USE_EXACT_SPHERE", "W2W_ESD_USE_EXACT_SPHERE"], True)
    verbose = _cfg_bool(cfg, ["verbose"], False)

    mu_h_um = (float(top_dish_mean_nm) + float(bot_dish_mean_nm)) * 1e-3
    sigma_h_um = math.sqrt(max(float(top_dish_std_nm), 0.0) ** 2 + max(float(bot_dish_std_nm), 0.0) ** 2) * 1e-3
    if sigma_h_um <= 0.0:
        raise ValueError("Combined dishing sigma is zero. Analytical ESD yield requires positive variation.")

    if warpage_mean_um is None:
        warpage_mean_um = _w2w_warpage_mean_um_from_cfg(cfg)
    if warpage_std_um is None:
        warpage_std_um = _w2w_warpage_std_um_from_cfg(cfg)
    warpage_nodes, warpage_weights = _normal_quadrature(
        abs(float(warpage_mean_um)),
        max(float(warpage_std_um), 0.0),
        warpage_q,
    )

    v_nodes, v_weights = _legendre_quadrature_interval(voltage_q, v_min_v, v_max_v)
    voltage_norm = v_max_v - v_min_v
    if voltage_norm <= 0.0:
        raise ValueError("cfg.V_MAX_V must be greater than cfg.V_MIN_V.")

    die_failure_probability = 0.0
    total_cases = int(len(v_nodes) * len(warpage_nodes))
    case_id = 0

    for v_chg, v_weight in zip(v_nodes, v_weights):
        arc_distance_um = _arc_distance_um_from_voltage(float(v_chg), cfg=cfg)
        p_fail_v = _compute_p_fail_for_die(
            top_die_w_um,
            top_die_h_um,
            float(v_chg),
            cfg=cfg,
            geff_um=arc_distance_um,
            weibull_k=weibull_k,
            weibull_lambda=weibull_lambda,
            cutoff_min_a=cutoff_min_a,
        )

        critical_first_arcing_prob_v = 0.0
        total_w_weight = 0.0
        for warpage_um, warpage_weight in zip(warpage_nodes, warpage_weights):
            contact_limit_um = _w2w_pad_contact_limit_um(
                pad_coords_um=pad_coords_um,
                pad_size_um=pad_size_um,
                wafer_radius_um=wafer_radius_um,
                warpage_um=abs(float(warpage_um)),
                z_top_um=z_top_um,
                exact_sphere=exact_sphere,
            )
            candidate_idx = _select_candidate_pad_indices(
                contact_limit_um=contact_limit_um,
                sigma_h_um=sigma_h_um,
                candidate_sigma_window=candidate_sigma_window,
                candidate_min_pads=candidate_min_pads,
                candidate_disable_fraction=candidate_disable_fraction,
            )
            critical_first_arcing_prob = _fixed_w2w_critical_probability_with_arcing(
                contact_limit_um=contact_limit_um[candidate_idx],
                critical_mask=esd_critical_pad_mask[candidate_idx],
                mu_h_um=mu_h_um,
                sigma_h_um=sigma_h_um,
                arc_distance_um=arc_distance_um,
                quadrature_points=quadrature_points,
                tail_sigma=tail_sigma,
                chunk_size=chunk_size,
                fill_residual_uniformly=fill_residual_uniformly,
            )
            critical_first_arcing_prob_v += float(warpage_weight) * float(critical_first_arcing_prob)
            total_w_weight += float(warpage_weight)
            case_id += 1

            if verbose:
                print(
                    f"[W2W ESD analytical] {case_id}/{total_cases} | "
                    f"V={float(v_chg):.4f} V | w={abs(float(warpage_um)):.4f} um",
                    end="\r",
                    flush=True,
                )

        if total_w_weight > 0.0:
            critical_first_arcing_prob_v /= total_w_weight

        die_failure_probability += (
            float(v_weight) / voltage_norm
            * float(p_fail_v)
            * float(np.clip(critical_first_arcing_prob_v, 0.0, 1.0))
        )

    if verbose:
        print()

    return float(1.0 - np.clip(die_failure_probability, 0.0, 1.0))


def pad_esd_yield_map_generator(
    *,
    cfg,
    pad_coords_um: np.ndarray,
    esd_critical_pad_mask: np.ndarray | None = None,
    dummy_pad_bitmap: np.ndarray | None = None,
    pad_size_um: float,
    pad_pitch_um: float | None = None,
    top_die_w_um: float,
    top_die_h_um: float,
    wafer_radius_um: float,
    die_center_um: Sequence[float] | None = None,
    top_dish_mean_nm: float = 0.0,
    top_dish_std_nm: float = 0.0,
    bot_dish_mean_nm: float = 0.0,
    bot_dish_std_nm: float = 0.0,
) -> Tuple[np.ndarray, None, float]:
    """
    Return a per-pad conditional ESD yield map for one W2W center-candidate die.

    This mirrors the D2W API shape. The returned figure is currently ``None``.
    """
    pad_coords_um = np.asarray(pad_coords_um, dtype=np.float64)
    if die_center_um is not None:
        pad_coords_um = pad_coords_um + np.asarray(die_center_um, dtype=np.float64).reshape(1, 2)
    pad_count = pad_coords_um.shape[0]
    if dummy_pad_bitmap is None:
        dummy_mask = np.zeros((pad_count,), dtype=bool)
    else:
        dummy_mask = np.asarray(dummy_pad_bitmap, dtype=bool).reshape(-1)
    if dummy_mask.size != pad_count:
        raise ValueError("dummy_pad_bitmap must have the same length as pad_coords_um.")
    if esd_critical_pad_mask is None:
        critical_mask = np.ones((pad_count,), dtype=bool)
    else:
        critical_mask = np.asarray(esd_critical_pad_mask, dtype=bool).reshape(-1)
    if critical_mask.size != pad_count:
        raise ValueError("esd_critical_pad_mask must have the same length as pad_coords_um.")

    active_mask = np.isfinite(pad_coords_um[:, 0]) & np.isfinite(pad_coords_um[:, 1]) & ~dummy_mask
    yield_vec = np.ones((pad_count,), dtype=np.float64)
    if not np.any(active_mask):
        return yield_vec, None, 0.0

    active_coords = pad_coords_um[active_mask]
    active_critical = critical_mask[active_mask]

    v_min_v = _cfg_float(cfg, ["V_MIN_V"], 0.0)
    v_max_v = _cfg_float(cfg, ["V_MAX_V"], 5.0)
    voltage_q = int(_cfg_float(cfg, ["ESD_ANALYTICAL_VOLTAGE_Q"], 5))
    weibull_k = _cfg_float(cfg, ["WEIBULL_K"], 4.44985)
    weibull_lambda = _cfg_float(cfg, ["WEIBULL_LAMBDA"], 0.0621816)
    cutoff_min_a = _cfg_float(cfg, ["CUTOFF_MIN_A"], 0.0)
    v_nodes, v_weights = _legendre_quadrature_interval(voltage_q, v_min_v, v_max_v)
    voltage_norm = v_max_v - v_min_v
    p_fail_avg = 0.0
    for v_chg, v_weight in zip(v_nodes, v_weights):
        arc_distance_um = _arc_distance_um_from_voltage(float(v_chg), cfg=cfg)
        p_fail_avg += (float(v_weight) / voltage_norm) * _compute_p_fail_for_die(
            top_die_w_um,
            top_die_h_um,
            float(v_chg),
            cfg=cfg,
            geff_um=arc_distance_um,
            weibull_k=weibull_k,
            weibull_lambda=weibull_lambda,
            cutoff_min_a=cutoff_min_a,
        )

    die_yield = die_esd_yield_calculator(
        cfg=cfg,
        pad_coords_um=active_coords,
        esd_critical_pad_mask=active_critical,
        pad_size_um=pad_size_um,
        top_die_w_um=top_die_w_um,
        top_die_h_um=top_die_h_um,
        wafer_radius_um=wafer_radius_um,
        top_dish_mean_nm=top_dish_mean_nm,
        top_dish_std_nm=top_dish_std_nm,
        bot_dish_mean_nm=bot_dish_mean_nm,
        bot_dish_std_nm=bot_dish_std_nm,
    )
    active_yield = np.ones((int(np.count_nonzero(active_mask)),), dtype=np.float64)
    if np.any(active_critical):
        active_yield[active_critical] = die_yield
    yield_vec[active_mask] = active_yield
    return yield_vec, None, float(p_fail_avg)


def stack_esd_yield_calculator(
    *,
    cfg_dict,
    waf_stack,
    pad_bitmap_collection_dict: dict | None = None,
):
    """
    Calculate W2W ESD yield per die and write it into ``waf_stack``.

    Only the 1/2/4 dies sharing the wafer center are evaluated. Their first
    arcing probability is split evenly by symmetry. All other dies keep ESD
    yield equal to 1.0.
    """
    if pad_bitmap_collection_dict is None:
        pad_bitmap_collection_dict = waf_stack.interfaces.pad_bitmap_collection_dict

    for interface_name, cfg in cfg_dict.items():
        interface = waf_stack.interfaces.interface_dict[interface_name]
        pad_bitmap_collection = pad_bitmap_collection_dict[interface_name]
        die_count = len(interface.die_list)
        die_esd_yield_list = np.ones((die_count,), dtype=np.float64)

        tolerance_um = _cfg_float(cfg, ["ESD_CENTER_TOL_UM", "W2W_ESD_CENTER_TOL_UM"], -1.0)
        tolerance_arg = None if tolerance_um < 0.0 else tolerance_um
        center_indices = center_die_indices(interface.die_list, tolerance_um=tolerance_arg)
        if center_indices.size <= 0:
            waf_stack.die_yield_list_per_interface_dict[interface_name]["ESD"] = die_esd_yield_list
            continue

        base_pad_coords = np.asarray(interface.base_pad_coords, dtype=np.float64)
        if base_pad_coords.ndim != 2 or base_pad_coords.shape[1] != 2:
            raise ValueError(f"{interface_name}: interface.base_pad_coords must have shape (n_pads, 2).")

        esd_critical_mask = np.asarray(
            pad_bitmap_collection["ESD_CRITICAL_PAD_BITMAP"],
            dtype=bool,
        ).reshape(-1)
        dummy_mask = np.asarray(
            pad_bitmap_collection.get("DUMMY_PAD_BITMAP", np.zeros_like(esd_critical_mask)),
            dtype=bool,
        ).reshape(-1)
        if esd_critical_mask.size != base_pad_coords.shape[0] or dummy_mask.size != base_pad_coords.shape[0]:
            raise ValueError(f"{interface_name}: pad bitmap size does not match pad coordinate count.")

        finite_coord_mask = np.isfinite(base_pad_coords[:, 0]) & np.isfinite(base_pad_coords[:, 1])
        active_mask = finite_coord_mask & ~dummy_mask
        if not np.any(active_mask & esd_critical_mask):
            waf_stack.die_yield_list_per_interface_dict[interface_name]["ESD"] = die_esd_yield_list
            continue

        symmetry_weight = 1.0 / float(center_indices.size)
        for die_idx in center_indices:
            die = interface.die_list[int(die_idx)]
            wafer_pad_coords = base_pad_coords[active_mask] + np.asarray(die.die_center, dtype=np.float64).reshape(1, 2)
            conditional_yield = die_esd_yield_calculator(
                cfg=cfg,
                pad_coords_um=wafer_pad_coords,
                esd_critical_pad_mask=esd_critical_mask[active_mask],
                pad_size_um=float(cfg.PAD_TOP_R_um) * 2.0,
                top_die_w_um=float(interface.DIE_W_um),
                top_die_h_um=float(interface.DIE_L_um),
                wafer_radius_um=float(cfg.WAF_R_um),
                top_dish_mean_nm=_cfg_float(cfg, ["TOP_DISH_MEAN_nm"], 0.0),
                top_dish_std_nm=_dish_std_nm_from_cfg(cfg, "TOP"),
                bot_dish_mean_nm=_cfg_float(cfg, ["BOT_DISH_MEAN_nm"], 0.0),
                bot_dish_std_nm=_dish_std_nm_from_cfg(cfg, "BOT"),
            )
            die_failure_probability = 1.0 - float(conditional_yield)
            die_esd_yield_list[int(die_idx)] = 1.0 - symmetry_weight * die_failure_probability

        waf_stack.die_yield_list_per_interface_dict[interface_name]["ESD"] = die_esd_yield_list
