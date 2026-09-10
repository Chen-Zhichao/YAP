# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import hashlib
from typing import Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import log_ndtr, ndtri

EPS0_F_PER_M = 8.8541878128e-12


def _array_digest(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha1()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def _cache_float(value: float) -> float:
    return round(float(value), 15)


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


def _dish_std_nm_from_cfg(cfg, side: str) -> float:
    side = str(side).upper()
    return _cfg_float(cfg, [f"{side}_DISH_STD_nm"], 0.0)


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
    disc = b * b - 4.0 * a * c              # Discriminant of the quadratic equation for the large-gap region
    if disc <= 0.0:
        return plateau_upper_gap_um

    root = (-b + math.sqrt(disc)) / (2.0 * a)
    if root <= 0.0:
        return plateau_upper_gap_um
    return max(plateau_upper_gap_um, root * root)


def _legendre_quadrature_interval(q: int, low: float, high: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre nodes and weights over [low, high]."""
    x, w = leggauss(int(q))
    g = 0.5 * (x + 1.0) * (high - low) + low
    wg = 0.5 * (high - low) * w
    return g.astype(np.float64), wg.astype(np.float64)


def _tilt_polar_quadrature_cases(
    *,
    tilt_x_mean_deg: float,
    tilt_x_std_deg: float,
    tilt_y_mean_deg: float,
    tilt_y_std_deg: float,
    radial_q: int,
    angle_q: int,
) -> list[tuple[float, float, float]]:
    """
    Return quadrature cases for a 2D independent Gaussian tilt distribution.

    The old tensor Gauss-Hermite rule placed a high-weight node exactly at
    zero tilt, which is a poor fit for first-touch ESD because the minimum-gap
    pad selection changes sharply around zero. This rule integrates the
    standard-normal radius through its CDF and averages uniformly over angle,
    so no finite-weight sample is pinned at exactly zero radius.
    """
    tilt_x_mean_deg = float(tilt_x_mean_deg)
    tilt_x_std_deg = max(float(tilt_x_std_deg), 0.0)
    tilt_y_mean_deg = float(tilt_y_mean_deg)
    tilt_y_std_deg = max(float(tilt_y_std_deg), 0.0)
    radial_q = max(1, int(radial_q))
    angle_q = max(1, int(angle_q))

    if tilt_x_std_deg <= 0.0 and tilt_y_std_deg <= 0.0:
        return [(tilt_x_mean_deg, tilt_y_mean_deg, 1.0)]

    u_nodes_raw, u_weights_raw = leggauss(radial_q)
    u_nodes = 0.5 * (u_nodes_raw + 1.0)
    u_weights = 0.5 * u_weights_raw
    u_nodes = np.clip(u_nodes, np.finfo(np.float64).tiny, 1.0 - np.finfo(np.float64).eps)
    radii = np.sqrt(-2.0 * np.log1p(-u_nodes))

    cases = []
    angle_weight = 1.0 / float(angle_q)
    for radius, radial_weight in zip(radii, u_weights):
        for angle_idx in range(angle_q):
            phi = 2.0 * math.pi * (float(angle_idx) + 0.5) * angle_weight
            theta_x_deg = tilt_x_mean_deg + tilt_x_std_deg * float(radius) * math.cos(phi)
            theta_y_deg = tilt_y_mean_deg + tilt_y_std_deg * float(radius) * math.sin(phi)
            cases.append((theta_x_deg, theta_y_deg, float(radial_weight) * angle_weight))

    weight_sum = sum(weight for _, _, weight in cases)
    if weight_sum > 0.0:
        inv_weight_sum = 1.0 / float(weight_sum)
        cases = [
            (theta_x_deg, theta_y_deg, weight * inv_weight_sum)
            for theta_x_deg, theta_y_deg, weight in cases
        ]
    return cases


def _deterministic_contact_limit_um(
    *,
    pad_coords_um: np.ndarray,
    pad_size_um: float,
    tilt_x_deg: float,
    tilt_y_deg: float,
    z_top_um: float,
) -> np.ndarray:
    """
    Return the deterministic contact limit C_i [um] for each pad.

    The exact simulator compares the minimum gap at the lowest pad corner. For the
    analytical calculator we keep that same deterministic corner term, but approximate
    the random top+bottom dishing contribution with a shared Gaussian H_i = T_i + B_i.
    """
    pad_coords_um = np.asarray(pad_coords_um, dtype=np.float64)
    a, b, _ = _z_linear_coeffs(tilt_x_deg, tilt_y_deg)
    half_pad_um = 0.5 * float(pad_size_um)
    corner_drop_um = half_pad_um * (abs(float(a)) + abs(float(b)))
    return (
        float(z_top_um)
        + float(a) * pad_coords_um[:, 0]
        + float(b) * pad_coords_um[:, 1]
        - float(corner_drop_um)
    ).astype(np.float64)


def _fixed_tilt_probability_map(
    *,
    contact_limit_um: np.ndarray,
    mu_h_um: float,
    sigma_h_um: float,
    quadrature_points: int,
    tail_sigma: float,
    chunk_size: int,
    fill_residual_uniformly: bool,
) -> np.ndarray:
    """
    Return the per-pad minimum-gap probability map for fixed tilt.

    This version intentionally does not use voltage or Paschen arcing distance.
    It models the bonding approach as monotonic, so the first-arcing pad is the
    pad with the smallest physical gap; voltage only affects the subsequent
    failure probability.
    """
    contact_limit_um = np.asarray(contact_limit_um, dtype=np.float64).reshape(-1)
    pad_count = contact_limit_um.size
    if pad_count <= 0:
        return np.zeros((0,), dtype=np.float64)
    if sigma_h_um <= 0.0:
        raise ValueError("Combined dishing sigma must be positive for analytical ESD yield calculation.")

    mean_gap_um = contact_limit_um - float(mu_h_um)
    low = float(np.min(mean_gap_um) - float(tail_sigma) * float(sigma_h_um))
    high = float(np.max(mean_gap_um) + float(tail_sigma) * float(sigma_h_um))
    if high <= low:
        high = low + max(float(sigma_h_um), 1.0e-12)

    g_nodes, g_weights = _legendre_quadrature_interval(int(quadrature_points), low, high)
    prob = np.zeros((pad_count,), dtype=np.float64)
    log_norm = -math.log(float(sigma_h_um)) - 0.5 * math.log(2.0 * math.pi)
    chunk_size = max(1, int(chunk_size))

    for g, w in zip(g_nodes, g_weights):
        log_survival = log_ndtr((mean_gap_um - float(g)) / float(sigma_h_um))
        total_log_survival = float(np.sum(log_survival))
        logw = math.log(float(w))

        for start in range(0, pad_count, chunk_size):
            end = min(start + chunk_size, pad_count)
            local_mean = mean_gap_um[start:end]
            t = (float(g) - local_mean) / float(sigma_h_um)
            logf = -0.5 * t * t + log_norm
            log_integrand = logw + logf + total_log_survival - log_survival[start:end]
            prob[start:end] += np.exp(log_integrand)

    prob_sum = float(np.sum(prob))
    if prob_sum <= 0.0:
        prob.fill(1.0 / float(pad_count))
        return prob

    if prob_sum < 1.0 and fill_residual_uniformly:
        prob += (1.0 - prob_sum) / float(pad_count)
        return prob

    prob /= prob_sum
    return prob


def _fixed_tilt_critical_probability(
    *,
    contact_limit_um: np.ndarray,
    critical_mask: np.ndarray,
    mu_h_um: float,
    sigma_h_um: float,
    quadrature_points: int,
    tail_sigma: float,
    chunk_size: int,
    fill_residual_uniformly: bool,
) -> float:
    """Return the probability that the minimum-gap pad is in ``critical_mask``."""
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

    prob = _fixed_tilt_probability_map(
        contact_limit_um=contact_limit_um,
        mu_h_um=mu_h_um,
        sigma_h_um=sigma_h_um,
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
    """
    Return the candidate-pad indices to evaluate for a fixed tilt case.

    The first-touch pad must lie near the minimum deterministic contact limit.
    We keep pads within a small sigma-based window of the minimum limit, with a
    floor on the candidate count. If that window captures most pads, we disable
    pruning and evaluate the full set to avoid approximation artifacts.
    """
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


def pad_esd_yield_map_generator(
    *,
    cfg,
    pad_coords_um: np.ndarray,
    pad_size_um: float,
    pad_pitch_um: float,
    top_die_w_um: float,
    top_die_h_um: float,
    tilt_x_mean_deg: float,
    tilt_x_std_deg: float,
    tilt_y_mean_deg: float,
    tilt_y_std_deg: float,
    top_dish_mean_nm: float,
    top_dish_std_nm: float,
    bot_dish_mean_nm: float,
    bot_dish_std_nm: float,
    z_top_um=None,
) -> Tuple[np.ndarray, None, float]:
    """
    Return the per-pad ESD yield map using the analytical minimum-gap method.

    Output matches the old Monte Carlo generator:
      (valid_pad_yield_map_vec, fig, p_fail_avg)
    """
    pad_size_um = float(pad_size_um)
    top_die_w_um = float(top_die_w_um)
    top_die_h_um = float(top_die_h_um)
    tilt_x_mean_deg = float(tilt_x_mean_deg)
    tilt_x_std_deg = float(tilt_x_std_deg)
    tilt_y_mean_deg = float(tilt_y_mean_deg)
    tilt_y_std_deg = float(tilt_y_std_deg)
    top_dish_mean_nm = float(top_dish_mean_nm)
    top_dish_std_nm = float(top_dish_std_nm)
    bot_dish_mean_nm = float(bot_dish_mean_nm)
    bot_dish_std_nm = float(bot_dish_std_nm)
    if z_top_um is None:
        z_top_um = _cfg_float(cfg, ["ESD_Z_TOP_UM"], 0.1)
    else:
        z_top_um = float(z_top_um)

    weibull_k = float(cfg.WEIBULL_K)
    weibull_lambda = float(cfg.WEIBULL_LAMBDA)
    cutoff_min_a = float(cfg.CUTOFF_MIN_A)

    quadrature_points = int(getattr(cfg, "ESD_ANALYTICAL_INNER_Q", 48))
    tilt_radial_q = int(getattr(cfg, "ESD_ANALYTICAL_TILT_RADIAL_Q", 4))
    tilt_angle_q = int(getattr(cfg, "ESD_ANALYTICAL_TILT_ANGLE_Q", 8))
    tail_sigma = float(getattr(cfg, "ESD_ANALYTICAL_TAIL_SIGMA", 8.0))
    chunk_size = int(getattr(cfg, "ESD_ANALYTICAL_CHUNK_SIZE", 100000))
    fill_residual_uniformly = bool(getattr(cfg, "ESD_ANALYTICAL_FILL_RESIDUAL_UNIFORMLY", True))
    candidate_sigma_window = float(getattr(cfg, "ESD_ANALYTICAL_CANDIDATE_SIGMA_WINDOW", 8.0))
    candidate_min_pads = int(getattr(cfg, "ESD_ANALYTICAL_CANDIDATE_MIN_PADS", 4096))
    candidate_disable_fraction = float(
        getattr(cfg, "ESD_ANALYTICAL_CANDIDATE_DISABLE_FRACTION", 0.8)
    )
    verbose = bool(getattr(cfg, "verbose", False))

    pad_coords_um = np.asarray(pad_coords_um, dtype=np.float64)
    if pad_coords_um.ndim != 2 or pad_coords_um.shape[1] != 2:
        raise ValueError("pad_coords_um must have shape (n_pads, 2).")
    pad_count = pad_coords_um.shape[0]
    active_pad_count = pad_count

    if active_pad_count <= 0:
        raise ValueError("pad_coords_um is empty; analytical ESD yield calculation needs at least one pad.")

    mu_h_um = (float(top_dish_mean_nm) + float(bot_dish_mean_nm)) * 1e-3
    sigma_h_um = math.sqrt(max(float(top_dish_std_nm), 0.0) ** 2 + max(float(bot_dish_std_nm), 0.0) ** 2) * 1e-3
    if sigma_h_um <= 0.0:
        raise ValueError("Combined dishing sigma is zero. Analytical ESD yield requires positive variation.")

    v_chg = _cfg_float(cfg, ["V_CDM"], 5.0)
    arc_distance_um = _arc_distance_um_from_voltage(float(v_chg), cfg=cfg)
    p_fail_avg = _compute_p_fail_for_die(
        top_die_w_um,
        top_die_h_um,
        float(v_chg),
        cfg=cfg,
        geff_um=arc_distance_um,
        weibull_k=weibull_k,
        weibull_lambda=weibull_lambda,
        cutoff_min_a=cutoff_min_a,
    )

    tilt_cases = _tilt_polar_quadrature_cases(
        tilt_x_mean_deg=tilt_x_mean_deg,
        tilt_x_std_deg=tilt_x_std_deg,
        tilt_y_mean_deg=tilt_y_mean_deg,
        tilt_y_std_deg=tilt_y_std_deg,
        radial_q=tilt_radial_q,
        angle_q=tilt_angle_q,
    )
    total_cases = len(tilt_cases)
    case_id = 0
    first_touch_prob = np.zeros((active_pad_count,), dtype=np.float64)
    total_outer_weight = 0.0

    for theta_x_deg, theta_y_deg, outer_coeff in tilt_cases:
        contact_limit_um = _deterministic_contact_limit_um(
            pad_coords_um=pad_coords_um,
            pad_size_um=pad_size_um,
            tilt_x_deg=theta_x_deg,
            tilt_y_deg=theta_y_deg,
            z_top_um=z_top_um,
        )
        candidate_idx = _select_candidate_pad_indices(
            contact_limit_um=contact_limit_um,
            sigma_h_um=sigma_h_um,
            candidate_sigma_window=candidate_sigma_window,
            candidate_min_pads=candidate_min_pads,
            candidate_disable_fraction=candidate_disable_fraction,
        )
        prob_case_local = _fixed_tilt_probability_map(
            contact_limit_um=contact_limit_um[candidate_idx],
            mu_h_um=mu_h_um,
            sigma_h_um=sigma_h_um,
            quadrature_points=quadrature_points,
            tail_sigma=tail_sigma,
            chunk_size=chunk_size,
            fill_residual_uniformly=fill_residual_uniformly,
        )
        prob_case = np.zeros((active_pad_count,), dtype=np.float64)
        prob_case[candidate_idx] = prob_case_local
        first_touch_prob += float(outer_coeff) * prob_case
        total_outer_weight += float(outer_coeff)
        case_id += 1

        if verbose:
            print(
                f"[ESD analytical] {case_id}/{total_cases} | "
                f"theta_x={theta_x_deg:.3e} deg | "
                f"theta_y={theta_y_deg:.3e} deg",
                end="\r",
                flush=True,
            )

    if total_outer_weight > 0.0:
        first_touch_prob /= total_outer_weight
    s = float(np.sum(first_touch_prob))
    if s > 0.0:
        first_touch_prob /= s
    else:
        first_touch_prob.fill(1.0 / float(active_pad_count))

    risk_active = float(p_fail_avg) * first_touch_prob

    if verbose:
        print()

    risk_vec = risk_active.copy()
    valid_pad_yield_map_vec = 1.0 - risk_vec

    fig = None

    return valid_pad_yield_map_vec, fig, float(p_fail_avg)


def _d2w_voltage_failure_average(
    *,
    cfg,
    top_die_w_um: float,
    top_die_h_um: float,
) -> float:
    weibull_k = float(cfg.WEIBULL_K)
    weibull_lambda = float(cfg.WEIBULL_LAMBDA)
    cutoff_min_a = float(cfg.CUTOFF_MIN_A)
    v_chg = _cfg_float(cfg, ["V_CDM"], 5.0)
    arc_distance_um = _arc_distance_um_from_voltage(float(v_chg), cfg=cfg)
    return float(
        _compute_p_fail_for_die(
            top_die_w_um,
            top_die_h_um,
            float(v_chg),
            cfg=cfg,
            geff_um=arc_distance_um,
            weibull_k=weibull_k,
            weibull_lambda=weibull_lambda,
            cutoff_min_a=cutoff_min_a,
        )
    )


def _d2w_first_touch_cache_key(
    *,
    cfg,
    pad_coords_um: np.ndarray,
    esd_critical_pad_mask: np.ndarray,
    pad_size_um: float,
    tilt_x_mean_deg: float,
    tilt_x_std_deg: float,
    tilt_y_mean_deg: float,
    tilt_y_std_deg: float,
    top_dish_mean_nm: float,
    top_dish_std_nm: float,
    bot_dish_mean_nm: float,
    bot_dish_std_nm: float,
    z_top_um: float,
) -> tuple:
    return (
        _array_digest(np.asarray(pad_coords_um, dtype=np.float64)),
        _array_digest(np.asarray(esd_critical_pad_mask, dtype=bool)),
        _cache_float(pad_size_um),
        _cache_float(tilt_x_mean_deg),
        _cache_float(tilt_x_std_deg),
        _cache_float(tilt_y_mean_deg),
        _cache_float(tilt_y_std_deg),
        _cache_float(top_dish_mean_nm),
        _cache_float(top_dish_std_nm),
        _cache_float(bot_dish_mean_nm),
        _cache_float(bot_dish_std_nm),
        _cache_float(z_top_um),
        int(getattr(cfg, "ESD_FIRST_TOUCH_SAMPLES", 1000)),
        int(getattr(cfg, "ESD_FIRST_TOUCH_BATCH_SIZE", 16)),
        int(getattr(cfg, "ESD_FIRST_TOUCH_SEED", 12345)),
    )


def _d2w_first_touch_grid_cache_key(
    *,
    cfg,
    active_bitmap: np.ndarray,
    esd_critical_bitmap: np.ndarray,
    pad_size_um: float,
    tilt_x_mean_deg: float,
    tilt_x_std_deg: float,
    tilt_y_mean_deg: float,
    tilt_y_std_deg: float,
    top_dish_mean_nm: float,
    top_dish_std_nm: float,
    bot_dish_mean_nm: float,
    bot_dish_std_nm: float,
    z_top_um: float,
) -> tuple:
    return (
        _array_digest(np.asarray(active_bitmap, dtype=bool)),
        _array_digest(np.asarray(esd_critical_bitmap, dtype=bool)),
        _cache_float(pad_size_um),
        _cache_float(tilt_x_mean_deg),
        _cache_float(tilt_x_std_deg),
        _cache_float(tilt_y_mean_deg),
        _cache_float(tilt_y_std_deg),
        _cache_float(top_dish_mean_nm),
        _cache_float(top_dish_std_nm),
        _cache_float(bot_dish_mean_nm),
        _cache_float(bot_dish_std_nm),
        _cache_float(z_top_um),
        int(getattr(cfg, "ESD_FIRST_TOUCH_SAMPLES", 1000)),
        int(getattr(cfg, "ESD_FIRST_TOUCH_BATCH_SIZE", 128)),
        int(getattr(cfg, "ESD_FIRST_TOUCH_SEED", 12345)),
        _cache_float(getattr(cfg, "ESD_FIRST_TOUCH_TILE_PITCH_um", 100.0)),
    )


def _d2w_critical_first_touch_probability_grid_sampled(
    *,
    cfg,
    active_bitmap: np.ndarray,
    esd_critical_bitmap: np.ndarray,
    pad_size_um: float,
    tilt_x_mean_deg: float,
    tilt_x_std_deg: float,
    tilt_y_mean_deg: float,
    tilt_y_std_deg: float,
    top_dish_mean_nm: float,
    top_dish_std_nm: float,
    bot_dish_mean_nm: float,
    bot_dish_std_nm: float,
    z_top_um: float,
) -> float:
    """
    Scalable first-touch estimator for very large pad arrays.

    Pads are partitioned into computational tiles. Every pad still follows the
    same independent one-level Gaussian dishing model; a tile does not add a
    shared spatial random effect. Its contribution is represented by the
    maximum of n independent pad heights, preserving first-contact extremes
    while reducing the cost from O(samples * pads) to O(samples * tiles).
    """
    active_bitmap = np.asarray(active_bitmap, dtype=bool)
    esd_critical_bitmap = np.asarray(esd_critical_bitmap, dtype=bool)
    if active_bitmap.shape != esd_critical_bitmap.shape:
        raise ValueError("active_bitmap and esd_critical_bitmap must have the same shape.")

    active_count = int(np.count_nonzero(active_bitmap))
    if active_count <= 0:
        raise ValueError("No non-dummy pads are available for ESD first-touch calculation.")
    critical_count = int(np.count_nonzero(active_bitmap & esd_critical_bitmap))
    if critical_count <= 0:
        return 0.0
    if critical_count == active_count:
        return 1.0

    rows, cols = active_bitmap.shape
    pitch_r_um = float(cfg.PITCH_r_um)
    pitch_c_um = float(cfg.PITCH_c_um)
    tile_pitch_um = max(float(getattr(cfg, "ESD_FIRST_TOUCH_TILE_PITCH_um", 100.0)), min(pitch_r_um, pitch_c_um))
    tile_rows = max(1, int(round(tile_pitch_um / max(pitch_r_um, 1.0e-12))))
    tile_cols = max(1, int(round(tile_pitch_um / max(pitch_c_um, 1.0e-12))))

    tile_x = []
    tile_y = []
    tile_active_count = []
    tile_critical_fraction = []
    x0_um = -0.5 * float(cols - 1) * pitch_c_um
    y0_um = 0.5 * float(rows - 1) * pitch_r_um

    for row_start in range(0, rows, tile_rows):
        row_end = min(row_start + tile_rows, rows)
        row_center = 0.5 * float(row_start + row_end - 1)
        y_um = y0_um - row_center * pitch_r_um
        for col_start in range(0, cols, tile_cols):
            col_end = min(col_start + tile_cols, cols)
            active_tile = active_bitmap[row_start:row_end, col_start:col_end]
            n_active = int(np.count_nonzero(active_tile))
            if n_active <= 0:
                continue
            critical_tile = esd_critical_bitmap[row_start:row_end, col_start:col_end] & active_tile
            n_critical = int(np.count_nonzero(critical_tile))
            col_center = 0.5 * float(col_start + col_end - 1)
            tile_x.append(x0_um + col_center * pitch_c_um)
            tile_y.append(y_um)
            tile_active_count.append(n_active)
            tile_critical_fraction.append(float(n_critical) / float(n_active))

    if not tile_active_count:
        return 0.0

    tile_x = np.asarray(tile_x, dtype=np.float32)
    tile_y = np.asarray(tile_y, dtype=np.float32)
    tile_active_count = np.asarray(tile_active_count, dtype=np.float32)
    tile_critical_fraction = np.asarray(tile_critical_fraction, dtype=np.float32)
    tile_count = tile_active_count.size

    mu_h_um = np.float32((float(top_dish_mean_nm) + float(bot_dish_mean_nm)) * 1e-3)
    sigma_h_um = np.float32(
        math.sqrt(max(float(top_dish_std_nm), 0.0) ** 2 + max(float(bot_dish_std_nm), 0.0) ** 2) * 1e-3
    )
    sample_count = max(1, int(getattr(cfg, "ESD_FIRST_TOUCH_SAMPLES", 1000)))
    batch_size = max(1, int(getattr(cfg, "ESD_FIRST_TOUCH_BATCH_SIZE", 128)))
    seed = int(getattr(cfg, "ESD_FIRST_TOUCH_SEED", 12345))
    rng = np.random.default_rng(seed)

    pad_size_half_um = np.float32(0.5 * float(pad_size_um))
    z_top_um = np.float32(float(z_top_um))
    critical_probability_sum = 0.0
    simulated = 0
    while simulated < sample_count:
        this_batch = min(batch_size, sample_count - simulated)
        theta_x_samples = rng.normal(
            float(tilt_x_mean_deg),
            max(float(tilt_x_std_deg), 0.0),
            size=this_batch,
        )
        theta_y_samples = rng.normal(
            float(tilt_y_mean_deg),
            max(float(tilt_y_std_deg), 0.0),
            size=this_batch,
        )

        a_samples = np.empty((this_batch,), dtype=np.float32)
        b_samples = np.empty((this_batch,), dtype=np.float32)
        for sample_idx, (theta_x_deg, theta_y_deg) in enumerate(zip(theta_x_samples, theta_y_samples)):
            a, b, _ = _z_linear_coeffs(float(theta_x_deg), float(theta_y_deg))
            a_samples[sample_idx] = np.float32(a)
            b_samples[sample_idx] = np.float32(b)

        corner_drop_um = pad_size_half_um * (np.abs(a_samples) + np.abs(b_samples))
        deterministic_gap = (
            z_top_um
            + a_samples[:, None] * tile_x[None, :]
            + b_samples[:, None] * tile_y[None, :]
            - corner_drop_um[:, None]
        )

        if sigma_h_um > 0.0:
            u = rng.random((this_batch, tile_count), dtype=np.float32)
            u = np.clip(u, np.finfo(np.float32).tiny, 1.0 - np.finfo(np.float32).eps)
            max_quantile = np.exp(np.log(u).astype(np.float32) / tile_active_count[None, :])
            max_quantile = np.clip(max_quantile, np.finfo(np.float32).tiny, 1.0 - np.finfo(np.float32).eps)
            tile_max_height = mu_h_um + sigma_h_um * ndtri(max_quantile).astype(np.float32)
        else:
            tile_max_height = np.full((this_batch, tile_count), float(mu_h_um), dtype=np.float32)

        winner_tile = np.argmin(deterministic_gap - tile_max_height, axis=1)
        critical_probability_sum += float(np.sum(tile_critical_fraction[winner_tile]))
        simulated += this_batch

    return float(np.clip(critical_probability_sum / float(sample_count), 0.0, 1.0))


def _d2w_critical_first_touch_probability(
    *,
    cfg,
    pad_coords_um: np.ndarray,
    esd_critical_pad_mask: np.ndarray,
    pad_size_um: float,
    tilt_x_mean_deg: float,
    tilt_x_std_deg: float,
    tilt_y_mean_deg: float,
    tilt_y_std_deg: float,
    top_dish_mean_nm: float,
    top_dish_std_nm: float,
    bot_dish_mean_nm: float,
    bot_dish_std_nm: float,
    z_top_um: float,
) -> float:
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
        return 0.0

    mu_h_um = (float(top_dish_mean_nm) + float(bot_dish_mean_nm)) * 1e-3
    sigma_h_um = math.sqrt(max(float(top_dish_std_nm), 0.0) ** 2 + max(float(bot_dish_std_nm), 0.0) ** 2) * 1e-3

    sample_count = max(1, int(getattr(cfg, "ESD_FIRST_TOUCH_SAMPLES", 1000)))
    batch_size = max(1, int(getattr(cfg, "ESD_FIRST_TOUCH_BATCH_SIZE", 16)))
    seed = int(getattr(cfg, "ESD_FIRST_TOUCH_SEED", 12345))
    verbose = bool(getattr(cfg, "verbose", False))

    rng = np.random.default_rng(seed)
    x_um = np.asarray(pad_coords_um[:, 0], dtype=np.float32)
    y_um = np.asarray(pad_coords_um[:, 1], dtype=np.float32)
    critical_mask = np.asarray(esd_critical_pad_mask, dtype=bool)
    pad_size_half_um = np.float32(0.5 * float(pad_size_um))
    z_top_um = np.float32(float(z_top_um))
    mu_h_um = np.float32(mu_h_um)
    sigma_h_um = np.float32(max(float(sigma_h_um), 0.0))

    critical_hits = 0
    simulated = 0
    while simulated < sample_count:
        this_batch = min(batch_size, sample_count - simulated)
        theta_x_samples = rng.normal(
            float(tilt_x_mean_deg),
            max(float(tilt_x_std_deg), 0.0),
            size=this_batch,
        )
        theta_y_samples = rng.normal(
            float(tilt_y_mean_deg),
            max(float(tilt_y_std_deg), 0.0),
            size=this_batch,
        )

        a_samples = np.empty((this_batch,), dtype=np.float32)
        b_samples = np.empty((this_batch,), dtype=np.float32)
        for sample_idx, (theta_x_deg, theta_y_deg) in enumerate(
            zip(theta_x_samples, theta_y_samples)
        ):
            a, b, _ = _z_linear_coeffs(float(theta_x_deg), float(theta_y_deg))
            a_samples[sample_idx] = np.float32(a)
            b_samples[sample_idx] = np.float32(b)

        corner_drop_um = pad_size_half_um * (
            np.abs(a_samples) + np.abs(b_samples)
        )
        deterministic_gap = (
            z_top_um
            + a_samples[:, None] * x_um[None, :]
            + b_samples[:, None] * y_um[None, :]
            - corner_drop_um[:, None]
        )
        if sigma_h_um > 0.0:
            dish_height = rng.normal(
                float(mu_h_um),
                float(sigma_h_um),
                size=(this_batch, pad_count),
            ).astype(np.float32)
        else:
            dish_height = np.full(
                (this_batch, pad_count),
                float(mu_h_um),
                dtype=np.float32,
            )
        first_touch_idx = np.argmin(deterministic_gap - dish_height, axis=1)
        critical_hits += int(np.count_nonzero(critical_mask[first_touch_idx]))
        simulated += this_batch

        if verbose:
            print(
                f"[ESD sampled first-touch] {simulated}/{sample_count}",
                end="\r",
                flush=True,
            )

    if verbose:
        print()

    return float(critical_hits / float(sample_count))


def die_esd_yield_calculator(
    *,
    cfg,
    pad_coords_um: np.ndarray,
    esd_critical_pad_mask: np.ndarray,
    pad_size_um: float,
    top_die_w_um: float,
    top_die_h_um: float,
    tilt_x_mean_deg: float,
    tilt_x_std_deg: float,
    tilt_y_mean_deg: float,
    tilt_y_std_deg: float,
    top_dish_mean_nm: float,
    top_dish_std_nm: float,
    bot_dish_mean_nm: float,
    bot_dish_std_nm: float,
    z_top_um=None,
) -> float:
    """
    Return die-level ESD yield without building a per-pad risk heatmap.

    The only required first-touch statistic is the probability that the
    minimum-gap pad belongs to the critical set. This function aggregates that
    probability directly inside the fixed-tilt quadrature.
    """
    pad_size_um = float(pad_size_um)
    top_die_w_um = float(top_die_w_um)
    top_die_h_um = float(top_die_h_um)
    tilt_x_mean_deg = float(tilt_x_mean_deg)
    tilt_x_std_deg = float(tilt_x_std_deg)
    tilt_y_mean_deg = float(tilt_y_mean_deg)
    tilt_y_std_deg = float(tilt_y_std_deg)
    top_dish_mean_nm = float(top_dish_mean_nm)
    top_dish_std_nm = float(top_dish_std_nm)
    bot_dish_mean_nm = float(bot_dish_mean_nm)
    bot_dish_std_nm = float(bot_dish_std_nm)
    if z_top_um is None:
        z_top_um = _cfg_float(cfg, ["ESD_Z_TOP_UM"], 0.1)
    else:
        z_top_um = float(z_top_um)

    pad_coords_um = np.asarray(pad_coords_um, dtype=np.float64)
    esd_critical_pad_mask = np.asarray(esd_critical_pad_mask, dtype=bool).reshape(-1)
    if pad_coords_um.ndim != 2 or pad_coords_um.shape[1] != 2:
        raise ValueError("pad_coords_um must have shape (n_pads, 2).")
    pad_count = pad_coords_um.shape[0]
    if esd_critical_pad_mask.size != pad_count:
        raise ValueError("esd_critical_pad_mask must have the same length as pad_coords_um.")
    if pad_count <= 0:
        raise ValueError("pad_coords_um is empty; analytical ESD yield calculation needs at least one pad.")

    critical_count = int(np.count_nonzero(esd_critical_pad_mask))
    if critical_count <= 0:
        return 1.0

    p_fail_avg = _d2w_voltage_failure_average(
        cfg=cfg,
        top_die_w_um=top_die_w_um,
        top_die_h_um=top_die_h_um,
    )
    critical_first_touch_prob = _d2w_critical_first_touch_probability(
        cfg=cfg,
        pad_coords_um=pad_coords_um,
        esd_critical_pad_mask=esd_critical_pad_mask,
        pad_size_um=pad_size_um,
        tilt_x_mean_deg=tilt_x_mean_deg,
        tilt_x_std_deg=tilt_x_std_deg,
        tilt_y_mean_deg=tilt_y_mean_deg,
        tilt_y_std_deg=tilt_y_std_deg,
        top_dish_mean_nm=top_dish_mean_nm,
        top_dish_std_nm=top_dish_std_nm,
        bot_dish_mean_nm=bot_dish_mean_nm,
        bot_dish_std_nm=bot_dish_std_nm,
        z_top_um=z_top_um,
    )
    die_failure_probability = float(p_fail_avg) * float(np.clip(critical_first_touch_prob, 0.0, 1.0))

    return float(1.0 - np.clip(die_failure_probability, 0.0, 1.0))



def stack_esd_yield_calculator(
    *,
    cfg_dict,
    die_stack,
):
    """
    Calculate D2W ESD yield for every interface and write into die_stack.

    This analytical die-level ESD yield currently ignores redundant-pad
    tolerance. All physical, non-dummy pads participate in the first-touch ESD
    competition, and the die-level failure probability is the sum of per-pad
    ESD risks over critical pads only.
    """
    first_touch_cache = {}
    for interface_name, cfg in cfg_dict.items():
        interface = die_stack.interfaces.interface_dict[interface_name]
        pad_bitmap_collection = die_stack.interfaces.pad_bitmap_collection_dict[interface_name]

        pad_coords = getattr(interface, "pad_coords", None)
        if pad_coords is None:
            pad_coords = die_stack.interfaces.base_pad_coords_dict.get(interface_name)
        esd_critical_bitmap = pad_bitmap_collection["ESD_CRITICAL_PAD_BITMAP"]
        esd_critical_bitmap_2d = np.asarray(
            esd_critical_bitmap,
            dtype=bool,
        )
        dummy_bitmap_2d = np.asarray(
            pad_bitmap_collection.get("DUMMY_PAD_BITMAP", np.zeros_like(esd_critical_bitmap_2d)),
            dtype=bool,
        )
        critical_bitmap_2d = np.asarray(
            pad_bitmap_collection.get("CRITICAL_PAD_BITMAP", np.zeros_like(esd_critical_bitmap_2d)),
            dtype=bool,
        )
        redundant_bitmap_2d = np.asarray(
            pad_bitmap_collection.get("REDUNDANT_PAD_BITMAP", np.zeros_like(esd_critical_bitmap_2d)),
            dtype=bool,
        )
        power_ground_bitmap_2d = np.asarray(
            pad_bitmap_collection.get("POWER_GROUND_PAD_BITMAP", np.zeros_like(esd_critical_bitmap_2d)),
            dtype=bool,
        )
        active_bitmap_2d = (
            critical_bitmap_2d
            | redundant_bitmap_2d
            | power_ground_bitmap_2d
            | esd_critical_bitmap_2d
        ) & (~dummy_bitmap_2d)

        if not np.any(esd_critical_bitmap_2d & active_bitmap_2d):
            die_stack.die_yield_per_interface_dict[interface_name]["ESD"] = 1.0
            continue

        pad_size_um = float(cfg.PAD_TOP_R_um) * 2.0
        top_dish_mean_nm = _cfg_float(cfg, ["TOP_DISH_MEAN_nm"], 0.0)
        top_dish_std_nm = _dish_std_nm_from_cfg(cfg, "TOP")
        bot_dish_mean_nm = _cfg_float(cfg, ["BOT_DISH_MEAN_nm"], 0.0)
        bot_dish_std_nm = _dish_std_nm_from_cfg(cfg, "BOT")
        z_top_um = _cfg_float(cfg, ["ESD_Z_TOP_UM"], 0.1)

        active_pad_count = int(np.count_nonzero(active_bitmap_2d))
        large_pad_threshold = int(getattr(cfg, "ESD_FIRST_TOUCH_GRID_THRESHOLD_PADS", 100_000))
        use_grid_first_touch = (
            pad_coords is None
            or active_pad_count >= large_pad_threshold
            or str(getattr(cfg, "ESD_FIRST_TOUCH_METHOD", "auto")).strip().lower() == "grid"
        )

        if use_grid_first_touch:
            first_touch_key = _d2w_first_touch_grid_cache_key(
                cfg=cfg,
                active_bitmap=active_bitmap_2d,
                esd_critical_bitmap=esd_critical_bitmap_2d,
                pad_size_um=pad_size_um,
                tilt_x_mean_deg=float(cfg.TILT_X_MEAN_DEG),
                tilt_x_std_deg=float(cfg.TILT_X_STD_DEG),
                tilt_y_mean_deg=float(cfg.TILT_Y_MEAN_DEG),
                tilt_y_std_deg=float(cfg.TILT_Y_STD_DEG),
                top_dish_mean_nm=top_dish_mean_nm,
                top_dish_std_nm=top_dish_std_nm,
                bot_dish_mean_nm=bot_dish_mean_nm,
                bot_dish_std_nm=bot_dish_std_nm,
                z_top_um=z_top_um,
            )
            if first_touch_key not in first_touch_cache:
                first_touch_cache[first_touch_key] = _d2w_critical_first_touch_probability_grid_sampled(
                    cfg=cfg,
                    active_bitmap=active_bitmap_2d,
                    esd_critical_bitmap=esd_critical_bitmap_2d,
                    pad_size_um=pad_size_um,
                    tilt_x_mean_deg=float(cfg.TILT_X_MEAN_DEG),
                    tilt_x_std_deg=float(cfg.TILT_X_STD_DEG),
                    tilt_y_mean_deg=float(cfg.TILT_Y_MEAN_DEG),
                    tilt_y_std_deg=float(cfg.TILT_Y_STD_DEG),
                    top_dish_mean_nm=top_dish_mean_nm,
                    top_dish_std_nm=top_dish_std_nm,
                    bot_dish_mean_nm=bot_dish_mean_nm,
                    bot_dish_std_nm=bot_dish_std_nm,
                    z_top_um=z_top_um,
                )
        else:
            pad_coords = np.asarray(pad_coords, dtype=np.float64)
            if pad_coords.ndim != 2 or pad_coords.shape[1] != 2:
                raise ValueError(f"{interface_name}: interface.pad_coords must have shape (n_pads, 2).")
            pad_count = pad_coords.shape[0]
            esd_critical_mask = esd_critical_bitmap_2d.reshape(-1)
            dummy_mask = dummy_bitmap_2d.reshape(-1)
            if esd_critical_mask.shape[0] != pad_count or dummy_mask.shape[0] != pad_count:
                raise ValueError(
                    f"{interface_name}: pad bitmap size does not match pad coordinate count."
                )
            finite_coord_mask = np.isfinite(pad_coords[:, 0]) & np.isfinite(pad_coords[:, 1])
            active_mask = finite_coord_mask & active_bitmap_2d.reshape(-1)
            active_pad_coords = pad_coords[active_mask]
            active_esd_critical_mask = esd_critical_mask[active_mask]
            first_touch_key = _d2w_first_touch_cache_key(
                cfg=cfg,
                pad_coords_um=active_pad_coords,
                esd_critical_pad_mask=active_esd_critical_mask,
                pad_size_um=pad_size_um,
                tilt_x_mean_deg=float(cfg.TILT_X_MEAN_DEG),
                tilt_x_std_deg=float(cfg.TILT_X_STD_DEG),
                tilt_y_mean_deg=float(cfg.TILT_Y_MEAN_DEG),
                tilt_y_std_deg=float(cfg.TILT_Y_STD_DEG),
                top_dish_mean_nm=top_dish_mean_nm,
                top_dish_std_nm=top_dish_std_nm,
                bot_dish_mean_nm=bot_dish_mean_nm,
                bot_dish_std_nm=bot_dish_std_nm,
                z_top_um=z_top_um,
            )
            if first_touch_key not in first_touch_cache:
                first_touch_cache[first_touch_key] = _d2w_critical_first_touch_probability(
                    cfg=cfg,
                    pad_coords_um=active_pad_coords,
                    esd_critical_pad_mask=active_esd_critical_mask,
                    pad_size_um=pad_size_um,
                    tilt_x_mean_deg=float(cfg.TILT_X_MEAN_DEG),
                    tilt_x_std_deg=float(cfg.TILT_X_STD_DEG),
                    tilt_y_mean_deg=float(cfg.TILT_Y_MEAN_DEG),
                    tilt_y_std_deg=float(cfg.TILT_Y_STD_DEG),
                    top_dish_mean_nm=top_dish_mean_nm,
                    top_dish_std_nm=top_dish_std_nm,
                    bot_dish_mean_nm=bot_dish_mean_nm,
                    bot_dish_std_nm=bot_dish_std_nm,
                    z_top_um=z_top_um,
                )

        p_fail_avg = _d2w_voltage_failure_average(
            cfg=cfg,
            top_die_w_um=float(interface.DIE_W_um),
            top_die_h_um=float(interface.DIE_L_um),
        )
        die_failure_probability = float(p_fail_avg) * float(first_touch_cache[first_touch_key])
        die_esd_yield = float(1.0 - np.clip(die_failure_probability, 0.0, 1.0))
        die_stack.die_yield_per_interface_dict[interface_name]["ESD"] = die_esd_yield
