# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.special import log_ndtr

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


def _fixed_tilt_probability_map_with_arcing(
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
    """
    Return the per-pad first-touch probability map for fixed tilt and fixed voltage.

    This is the analytical counterpart to the Monte Carlo path. It follows the demo's
    exact fixed-tilt minimum-gap integral, while treating arcing as a pad-eligibility
    threshold on H_i = T_i + B_i.
    """
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

    for g, w in zip(g_nodes, g_weights):
        valid_mask = (g <= contact_limit_um)
        if not np.any(valid_mask):
            continue

        log_survival = np.where(
            valid_mask,
            log_ndtr((mean_gap_um - float(g)) / float(sigma_h_um)),
            inactive_log_prob,
        )
        total_log_survival = float(np.sum(log_survival))
        logw = math.log(float(w))

        for start in range(0, pad_count, int(chunk_size)):
            end = min(start + int(chunk_size), pad_count)
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


def _plot_probability_over_pads_with_pitch(
    pad_coords_um: np.ndarray,
    prob_vec: np.ndarray,
    *,
    pitch_um: float,
    die_w_um: float,
    die_h_um: float,
    title: str,
) -> plt.Figure:
    """Plot one display square per pad, using pitch as the display square size."""
    fig, ax = plt.subplots()
    try:
        fig.canvas.toolbar_visible = True
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
    except Exception:
        pass

    vmax = float(prob_vec.max()) if prob_vec.size > 0 else 0.0
    norm_max = vmax if vmax > 0.0 else 1.0
    half_pix = 0.5 * float(pitch_um)

    for (x, y), prob in zip(pad_coords_um, prob_vec):
        if prob <= 0.0:
            continue
        rect = Rectangle((x - half_pix, y - half_pix), 2.0 * half_pix, 2.0 * half_pix, linewidth=0.0)
        rect.set_facecolor(plt.cm.viridis(prob / norm_max))
        rect.set_edgecolor("none")
        ax.add_patch(rect)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-float(die_w_um) / 2.0, float(die_w_um) / 2.0)
    ax.set_ylim(-float(die_h_um) / 2.0, float(die_h_um) / 2.0)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("x (um), center at 0")
    ax.set_ylabel("y (um), top is smaller")

    sm = mpl.cm.ScalarMappable(cmap="viridis", norm=mpl.colors.Normalize(vmin=0.0, vmax=norm_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Risk = P_first-touch * p_fail_single")
    return fig


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
) -> Tuple[np.ndarray, plt.Figure | None, float]:
    """
    Return the per-pad ESD yield map using the analytical minimum-gap method.

    Output matches the old Monte Carlo generator:
      (valid_pad_yield_map_vec, fig, p_fail_avg)
    """
    pad_size_um = float(pad_size_um)
    pad_pitch_um = float(pad_pitch_um)
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

    v_min_v = float(cfg.V_MIN_V)
    v_max_v = float(cfg.V_MAX_V)
    weibull_k = float(cfg.WEIBULL_K)
    weibull_lambda = float(cfg.WEIBULL_LAMBDA)
    cutoff_min_a = float(cfg.CUTOFF_MIN_A)

    quadrature_points = int(getattr(cfg, "ESD_ANALYTICAL_INNER_Q", 48))
    outer_qx = int(getattr(cfg, "ESD_ANALYTICAL_OUTER_QX", 5))
    outer_qy = int(getattr(cfg, "ESD_ANALYTICAL_OUTER_QY", 5))
    voltage_q = int(getattr(cfg, "ESD_ANALYTICAL_VOLTAGE_Q", 5))
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

    x_nodes, x_weights = hermgauss(outer_qx)
    y_nodes, y_weights = hermgauss(outer_qy)
    v_nodes, v_weights = _legendre_quadrature_interval(voltage_q, float(v_min_v), float(v_max_v))
    voltage_norm = float(v_max_v) - float(v_min_v)
    if voltage_norm <= 0.0:
        raise ValueError("cfg.V_MAX_V must be greater than cfg.V_MIN_V.")

    total_cases = int(outer_qx) * int(outer_qy) * int(voltage_q)
    case_id = 0

    risk_active = np.zeros((active_pad_count,), dtype=np.float64)
    p_fail_avg = 0.0

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
        # print(f"Voltage {float(v_chg):.4f} V has die-level failure probability {p_fail_v:.6e} and arc distance {arc_distance_um:.2f} um")
        p_fail_avg += (float(v_weight) / voltage_norm) * float(p_fail_v)

        prob_v = np.zeros((active_pad_count,), dtype=np.float64)
        total_outer_weight = 0.0

        for xa, wa in zip(x_nodes, x_weights):
            theta_x_deg = float(tilt_x_mean_deg) + math.sqrt(2.0) * float(tilt_x_std_deg) * float(xa)

            for yb, wb in zip(y_nodes, y_weights):
                theta_y_deg = float(tilt_y_mean_deg) + math.sqrt(2.0) * float(tilt_y_std_deg) * float(yb)
                outer_coeff = float(wa * wb / math.pi)

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
                prob_case_local = _fixed_tilt_probability_map_with_arcing(
                    contact_limit_um=contact_limit_um[candidate_idx],
                    mu_h_um=mu_h_um,
                    sigma_h_um=sigma_h_um,
                    arc_distance_um=arc_distance_um,
                    quadrature_points=quadrature_points,
                    tail_sigma=tail_sigma,
                    chunk_size=chunk_size,
                    fill_residual_uniformly=fill_residual_uniformly,
                )
                prob_case = np.zeros((active_pad_count,), dtype=np.float64)
                prob_case[candidate_idx] = prob_case_local
                prob_v += outer_coeff * prob_case
                total_outer_weight += outer_coeff
                case_id += 1

                if verbose:
                    print(
                        f"[ESD analytical] {case_id}/{total_cases} | "
                        f"V={float(v_chg):.4f} V | "
                        f"theta_x={theta_x_deg:.3e} deg | "
                        f"theta_y={theta_y_deg:.3e} deg",
                        end="\r",
                        flush=True,
                    )

        if total_outer_weight > 0.0:
            prob_v /= total_outer_weight
        s = float(np.sum(prob_v))
        if s > 0.0:
            prob_v /= s
        else:
            prob_v.fill(1.0 / float(active_pad_count))

        risk_active += (float(v_weight) / voltage_norm) * float(p_fail_v) * prob_v

    if verbose:
        print()

    risk_vec = risk_active.copy()
    valid_pad_yield_map_vec = 1.0 - risk_vec

    # if bool(getattr(cfg, "plot_flag", False)):
    #     fig = _plot_probability_over_pads_with_pitch(
    #         pad_coords_um=pad_coords_um,
    #         prob_vec=risk_vec,
    #         pitch_um=pad_pitch_um,
    #         die_w_um=top_die_w_um,
    #         die_h_um=top_die_h_um,
    #         title="Risk Pad Map = E[1(first-touch pad) * p_fail(V)], analytical",
    #     )
    # else:
    #     fig = None
    fig = None

    return valid_pad_yield_map_vec, fig, float(p_fail_avg)



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
    for interface_name, cfg in cfg_dict.items():
        interface = die_stack.interfaces.interface_dict[interface_name]
        pad_bitmap_collection = die_stack.interfaces.pad_bitmap_collection_dict[interface_name]

        pad_coords = np.asarray(interface.pad_coords, dtype=np.float64)
        if pad_coords.ndim != 2 or pad_coords.shape[1] != 2:
            raise ValueError(f"{interface_name}: interface.pad_coords must have shape (n_pads, 2).")

        pad_count = pad_coords.shape[0]
        critical_mask = np.asarray(
            pad_bitmap_collection["CRITICAL_PAD_BITMAP"],
            dtype=bool,
        ).reshape(-1)
        dummy_mask = np.asarray(
            pad_bitmap_collection.get("DUMMY_PAD_BITMAP", np.zeros_like(critical_mask)),
            dtype=bool,
        ).reshape(-1)

        if critical_mask.shape[0] != pad_count or dummy_mask.shape[0] != pad_count:
            raise ValueError(
                f"{interface_name}: pad bitmap size does not match pad coordinate count."
            )

        finite_coord_mask = np.isfinite(pad_coords[:, 0]) & np.isfinite(pad_coords[:, 1])
        active_mask = finite_coord_mask & ~dummy_mask
        if not np.any(critical_mask & active_mask):
            die_stack.die_yield_per_interface_dict[interface_name]["ESD"] = 1.0
            full_pad_yield = np.ones((pad_count,), dtype=np.float64)
            interface.pad_yield_map["Y_esd"] = full_pad_yield.reshape(
                int(cfg.PAD_ARR_ROW),
                int(cfg.PAD_ARR_COL),
            )
            continue

        active_pad_yield_vec, _, _ = pad_esd_yield_map_generator(
            cfg=cfg,
            pad_coords_um=pad_coords[active_mask],
            pad_size_um=float(cfg.PAD_TOP_R_um) * 2.0,
            pad_pitch_um=max(float(cfg.PITCH_r_um), float(cfg.PITCH_c_um)),
            top_die_w_um=float(interface.DIE_W_um),
            top_die_h_um=float(interface.DIE_L_um),
            tilt_x_mean_deg=float(cfg.TILT_X_MEAN_DEG),
            tilt_x_std_deg=float(cfg.TILT_X_STD_DEG),
            tilt_y_mean_deg=float(cfg.TILT_Y_MEAN_DEG),
            tilt_y_std_deg=float(cfg.TILT_Y_STD_DEG),
            top_dish_mean_nm=float(cfg.TOP_DISH_MEAN_nm),
            top_dish_std_nm=float(cfg.TOP_DISH_STD_nm),
            bot_dish_mean_nm=float(cfg.BOT_DISH_MEAN_nm),
            bot_dish_std_nm=float(cfg.BOT_DISH_STD_nm),
        )

        active_pad_risk_vec = np.clip(1.0 - active_pad_yield_vec, 0.0, 1.0)
        full_pad_risk = np.zeros((pad_count,), dtype=np.float64)
        full_pad_risk[active_mask] = active_pad_risk_vec

        die_failure_probability = float(
            np.clip(np.sum(full_pad_risk[critical_mask & active_mask]), 0.0, 1.0)
        )
        die_stack.die_yield_per_interface_dict[interface_name]["ESD"] = float(
            1.0 - die_failure_probability
        )

        full_pad_yield = 1.0 - full_pad_risk
        interface.pad_yield_map["Y_esd"] = full_pad_yield.reshape(
            int(cfg.PAD_ARR_ROW),
            int(cfg.PAD_ARR_COL),
        )
