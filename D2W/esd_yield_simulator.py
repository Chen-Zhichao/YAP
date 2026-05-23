# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Tuple
import numpy as np
from scipy.special import ndtri

EPS0_F_PER_M = 8.8541878128e-12
_esd_rng = np.random.default_rng()


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
    Return the effective discharge gap [um] used by the ESD failure model.

    First-arcing pad selection is based on physical pad gaps. The post-arcing
    failure model uses the CDM voltage to determine this effective gap.
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


def _prepare_pad_geometry_cache(
    pad_coords_um: np.ndarray,
    pad_size_um: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return reusable pad geometry arrays for repeated gap evaluations."""
    pad_coords_um = np.asarray(pad_coords_um, dtype=np.float64)
    return pad_coords_um[:, 0], pad_coords_um[:, 1], 0.5 * float(pad_size_um)

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
        - np.asarray(top_dish_um_raw, dtype=np.float64)
        - np.asarray(bot_dish_um, dtype=np.float64)
        - corner_drop_um
    )


def _first_arcing_pad_by_physical_gap(
    *,
    pad_coords_um: np.ndarray,
    pad_size_um: float,
    top_dish_um_raw: np.ndarray,
    bot_dish_um: np.ndarray,
    tilt_x_deg: float,
    tilt_y_deg: float,
    z_top_um: float,
) -> Tuple[int | None, bool, float]:
    """Return the non-dummy pad with the smallest physical top-bottom gap."""
    pad_coords_um = np.asarray(pad_coords_um, dtype=np.float64)
    if pad_coords_um.size == 0:
        return None, True, float("nan")

    pad_x_um, pad_y_um, half_pad_um = _prepare_pad_geometry_cache(
        pad_coords_um,
        pad_size_um,
    )
    a, b, _ = _z_linear_coeffs(tilt_x_deg, tilt_y_deg)
    pad_gaps = _square_pad_min_gap_vec(
        cx_um=pad_x_um,
        cy_um=pad_y_um,
        half_pad_um=float(half_pad_um),
        top_dish_um_raw=top_dish_um_raw,
        bot_dish_um=bot_dish_um,
        z_top_um=z_top_um,
        a=a,
        b=b,
    )
    best_pad = int(np.argmin(pad_gaps))
    return best_pad, False, float(pad_gaps[best_pad])


def build_esd_tile_cache(cfg, active_pad_mask: np.ndarray) -> dict:
    """
    Build a scalable tile cache for accelerated ESD first-arcing simulation.

    Each tile stores the number of active pads and its center coordinate. During
    simulation we sample the maximum top+bottom Cu height in each tile, then
    pick the tile with the smallest effective gap.
    """
    active_pad_mask = np.asarray(active_pad_mask, dtype=bool)
    if active_pad_mask.ndim != 2:
        raise ValueError("active_pad_mask must be a 2D bitmap.")

    rows, cols = active_pad_mask.shape
    pitch_r_um = float(cfg.PITCH_r_um)
    pitch_c_um = float(cfg.PITCH_c_um)
    tile_pitch_um = max(
        float(_cfg_float(cfg, ["ESD_SIM_TILE_PITCH_um", "ESD_FIRST_TOUCH_TILE_PITCH_um"], 100.0)),
        min(pitch_r_um, pitch_c_um),
    )
    tile_rows = max(1, int(round(tile_pitch_um / max(pitch_r_um, 1.0e-12))))
    tile_cols = max(1, int(round(tile_pitch_um / max(pitch_c_um, 1.0e-12))))

    row_starts = []
    row_ends = []
    col_starts = []
    col_ends = []
    tile_x_um = []
    tile_y_um = []
    tile_active_count = []
    tile_active_linear_indices = []
    x0_um = -0.5 * float(cols - 1) * pitch_c_um
    y0_um = 0.5 * float(rows - 1) * pitch_r_um

    for row_start in range(0, rows, tile_rows):
        row_end = min(row_start + tile_rows, rows)
        row_center = 0.5 * float(row_start + row_end - 1)
        y_um = y0_um - row_center * pitch_r_um
        for col_start in range(0, cols, tile_cols):
            col_end = min(col_start + tile_cols, cols)
            n_active = int(np.count_nonzero(active_pad_mask[row_start:row_end, col_start:col_end]))
            if n_active <= 0:
                continue
            active_tile = active_pad_mask[row_start:row_end, col_start:col_end]
            local_active = np.flatnonzero(active_tile)
            local_cols = col_end - col_start
            local_rows = local_active // local_cols
            local_cols_idx = local_active - local_rows * local_cols
            active_linear = (
                (row_start + local_rows) * cols
                + (col_start + local_cols_idx)
            ).astype(np.int64)
            col_center = 0.5 * float(col_start + col_end - 1)
            row_starts.append(row_start)
            row_ends.append(row_end)
            col_starts.append(col_start)
            col_ends.append(col_end)
            tile_x_um.append(x0_um + col_center * pitch_c_um)
            tile_y_um.append(y_um)
            tile_active_count.append(n_active)
            tile_active_linear_indices.append(active_linear)

    if not tile_active_count:
        raise ValueError("No active pads are available for ESD tile simulation.")

    return {
        "active_pad_mask": active_pad_mask,
        "rows": int(rows),
        "cols": int(cols),
        "row_starts": np.asarray(row_starts, dtype=np.int32),
        "row_ends": np.asarray(row_ends, dtype=np.int32),
        "col_starts": np.asarray(col_starts, dtype=np.int32),
        "col_ends": np.asarray(col_ends, dtype=np.int32),
        "tile_x_um": np.asarray(tile_x_um, dtype=np.float32),
        "tile_y_um": np.asarray(tile_y_um, dtype=np.float32),
        "tile_active_count": np.asarray(tile_active_count, dtype=np.float32),
        "tile_active_linear_indices": tile_active_linear_indices,
    }


def _sample_first_arcing_pad_from_tile_cache(
    *,
    cfg,
    tile_cache: dict,
    pad_size_um: float,
    tilt_x_deg: float,
    tilt_y_deg: float,
    z_top_um: float,
) -> int:
    top_mean_nm = _cfg_float(cfg, ["TOP_DISH_MEAN_nm"], 0.0)
    top_std_nm = _cfg_float(cfg, ["TOP_DISH_STD_nm"], 0.0)
    bot_mean_nm = _cfg_float(cfg, ["BOT_DISH_MEAN_nm"], 0.0)
    bot_std_nm = _cfg_float(cfg, ["BOT_DISH_STD_nm"], 0.0)
    mu_h_um = np.float32((top_mean_nm + bot_mean_nm) * 1e-3)
    sigma_h_um = np.float32(math.sqrt(max(top_std_nm, 0.0) ** 2 + max(bot_std_nm, 0.0) ** 2) * 1e-3)

    a, b, _ = _z_linear_coeffs(tilt_x_deg, tilt_y_deg)
    a = np.float32(a)
    b = np.float32(b)
    half_pad_um = np.float32(0.5 * float(pad_size_um))
    corner_drop_um = half_pad_um * (abs(a) + abs(b))
    deterministic_gap = (
        np.float32(float(z_top_um))
        + a * tile_cache["tile_x_um"]
        + b * tile_cache["tile_y_um"]
        - corner_drop_um
    )

    if sigma_h_um > 0.0:
        u = _esd_rng.random(tile_cache["tile_active_count"].shape, dtype=np.float32)
        u = np.clip(u, np.finfo(np.float32).tiny, 1.0 - np.finfo(np.float32).eps)
        max_quantile = np.exp(np.log(u).astype(np.float32) / tile_cache["tile_active_count"])
        max_quantile = np.clip(max_quantile, np.finfo(np.float32).tiny, 1.0 - np.finfo(np.float32).eps)
        tile_max_height = mu_h_um + sigma_h_um * ndtri(max_quantile).astype(np.float32)
    else:
        tile_max_height = np.full_like(deterministic_gap, float(mu_h_um), dtype=np.float32)

    winner_tile = int(np.argmin(deterministic_gap - tile_max_height))
    row_start = int(tile_cache["row_starts"][winner_tile])
    row_end = int(tile_cache["row_ends"][winner_tile])
    col_start = int(tile_cache["col_starts"][winner_tile])
    col_end = int(tile_cache["col_ends"][winner_tile])
    active_tile = tile_cache["active_pad_mask"][row_start:row_end, col_start:col_end]
    local_active = np.flatnonzero(active_tile)
    local_choice = int(local_active[int(_esd_rng.integers(local_active.size))])
    local_cols = col_end - col_start
    row = row_start + local_choice // local_cols
    col = col_start + local_choice % local_cols
    return int(row * int(tile_cache["cols"]) + col)


def _sample_first_arcing_pads_from_tile_cache_batch(
    *,
    cfg,
    tile_cache: dict,
    pad_size_um: float,
    tilt_x_deg: np.ndarray,
    tilt_y_deg: np.ndarray,
    z_top_um: float,
) -> np.ndarray:
    """Vectorized tile first-arcing sampler for one simulation batch."""
    tilt_x_deg = np.asarray(tilt_x_deg, dtype=np.float64).reshape(-1)
    tilt_y_deg = np.asarray(tilt_y_deg, dtype=np.float64).reshape(-1)
    if tilt_x_deg.shape != tilt_y_deg.shape:
        raise ValueError("tilt_x_deg and tilt_y_deg must have the same shape.")
    num_samples = tilt_x_deg.size
    if num_samples <= 0:
        return np.empty((0,), dtype=np.int64)

    top_mean_nm = _cfg_float(cfg, ["TOP_DISH_MEAN_nm"], 0.0)
    top_std_nm = _cfg_float(cfg, ["TOP_DISH_STD_nm"], 0.0)
    bot_mean_nm = _cfg_float(cfg, ["BOT_DISH_MEAN_nm"], 0.0)
    bot_std_nm = _cfg_float(cfg, ["BOT_DISH_STD_nm"], 0.0)
    mu_h_um = np.float32((top_mean_nm + bot_mean_nm) * 1e-3)
    sigma_h_um = np.float32(math.sqrt(max(top_std_nm, 0.0) ** 2 + max(bot_std_nm, 0.0) ** 2) * 1e-3)

    ax = np.deg2rad(tilt_x_deg)
    ay = np.deg2rad(tilt_y_deg)
    a = (-np.sin(ay)).astype(np.float32)
    b = (np.cos(ay) * np.sin(ax)).astype(np.float32)
    half_pad_um = np.float32(0.5 * float(pad_size_um))
    corner_drop_um = half_pad_um * (np.abs(a) + np.abs(b))
    deterministic_gap = (
        np.float32(float(z_top_um))
        + a[:, None] * tile_cache["tile_x_um"][None, :]
        + b[:, None] * tile_cache["tile_y_um"][None, :]
        - corner_drop_um[:, None]
    )

    if sigma_h_um > 0.0:
        shape = deterministic_gap.shape
        u = _esd_rng.random(shape, dtype=np.float32)
        u = np.clip(u, np.finfo(np.float32).tiny, 1.0 - np.finfo(np.float32).eps)
        max_quantile = np.exp(
            np.log(u).astype(np.float32)
            / tile_cache["tile_active_count"][None, :]
        )
        max_quantile = np.clip(max_quantile, np.finfo(np.float32).tiny, 1.0 - np.finfo(np.float32).eps)
        tile_max_height = mu_h_um + sigma_h_um * ndtri(max_quantile).astype(np.float32)
    else:
        tile_max_height = np.full_like(deterministic_gap, float(mu_h_um), dtype=np.float32)

    winner_tiles = np.argmin(deterministic_gap - tile_max_height, axis=1)
    pad_choices = np.empty((num_samples,), dtype=np.int64)
    active_linear_indices = tile_cache["tile_active_linear_indices"]
    for sample_idx, winner_tile in enumerate(winner_tiles):
        candidates = active_linear_indices[int(winner_tile)]
        pad_choices[sample_idx] = candidates[int(_esd_rng.integers(candidates.size))]
    return pad_choices


def esd_failure_simulator_batch(
    *,
    cfg,
    num_samples: int,
    pad_size_um: float,
    top_die_w_um: float,
    top_die_h_um: float,
    tilt_x_mean_deg: float,
    tilt_x_std_deg: float,
    tilt_y_mean_deg: float,
    tilt_y_std_deg: float,
    esd_tile_cache: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a vectorized tile-based ESD batch and return (full_pad_idx, survive_bool)."""
    num_samples = int(num_samples)
    if num_samples <= 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=bool)

    z_top_um = _cfg_float(cfg, ["ESD_Z_TOP_UM"], 0.1)
    v_cdm = _cfg_float(cfg, ["V_CDM"], 5.0)
    weibull_k = float(cfg.WEIBULL_K)
    weibull_lambda = float(cfg.WEIBULL_LAMBDA)
    cutoff_min_a = float(cfg.CUTOFF_MIN_A)

    tilt_x = _esd_rng.normal(float(tilt_x_mean_deg), float(tilt_x_std_deg), size=num_samples)
    tilt_y = _esd_rng.normal(float(tilt_y_mean_deg), float(tilt_y_std_deg), size=num_samples)
    pad_choices = _sample_first_arcing_pads_from_tile_cache_batch(
        cfg=cfg,
        tile_cache=esd_tile_cache,
        pad_size_um=pad_size_um,
        z_top_um=z_top_um,
        tilt_x_deg=tilt_x,
        tilt_y_deg=tilt_y,
    )

    geff_um = _arc_distance_um_from_voltage(float(v_cdm), cfg=cfg)
    p_fail_single = _compute_p_fail_for_die(
        float(top_die_w_um),
        float(top_die_h_um),
        float(v_cdm),
        cfg=cfg,
        geff_um=geff_um,
        weibull_k=weibull_k,
        weibull_lambda=weibull_lambda,
        cutoff_min_a=cutoff_min_a,
    )
    survive_bool = _esd_rng.random(num_samples) >= float(p_fail_single)
    return pad_choices, survive_bool.astype(bool)


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
    esd_tile_cache: dict | None = None,
    return_full_linear_idx: bool = False,
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
    v_cdm = _cfg_float(cfg, ["V_CDM"], 5.0)
    weibull_k = float(cfg.WEIBULL_K)
    weibull_lambda = float(cfg.WEIBULL_LAMBDA)
    cutoff_min_a = float(cfg.CUTOFF_MIN_A)

    tilt_x = float(_esd_rng.normal(tilt_x_mean_deg, tilt_x_std_deg))
    tilt_y = float(_esd_rng.normal(tilt_y_mean_deg, tilt_y_std_deg))
    v_chg = float(v_cdm)

    if esd_tile_cache is not None:
        if not return_full_linear_idx:
            raise ValueError("Tile-based ESD simulation returns full linear pad indices.")
        full_linear_choice = _sample_first_arcing_pad_from_tile_cache(
            cfg=cfg,
            tile_cache=esd_tile_cache,
            pad_size_um=pad_size_um,
            z_top_um=z_top_um,
            tilt_x_deg=tilt_x,
            tilt_y_deg=tilt_y,
        )
        pad_choice = full_linear_choice
    else:
        pad_coords_um, active_ids = _active_pad_ids_from_bitmap(pad_coords_um, dummy_pad_bitmap)
        active_pad_coords_um = pad_coords_um[active_ids]
        top_dish_nm_ext = np.asarray(top_dish_nm_ext, dtype=np.float64).reshape(-1)
        bot_dish_nm_ext = np.asarray(bot_dish_nm_ext, dtype=np.float64).reshape(-1)

        if not (
            pad_coords_um.shape[0] == top_dish_nm_ext.shape[0] == bot_dish_nm_ext.shape[0]
        ):
            raise ValueError("pad_coords_um, top_dish_nm_ext, and bot_dish_nm_ext must have the same length.")

        top_dish_um_raw = top_dish_nm_ext[active_ids] * 1e-3
        bot_dish_um = bot_dish_nm_ext[active_ids] * 1e-3

        pad_choice_active, _, _ = _first_arcing_pad_by_physical_gap(
            pad_coords_um=active_pad_coords_um,
            pad_size_um=pad_size_um,
            z_top_um=z_top_um,
            tilt_x_deg=tilt_x,
            tilt_y_deg=tilt_y,
            top_dish_um_raw=top_dish_um_raw,
            bot_dish_um=bot_dish_um,
        )
        pad_choice = int(active_ids[int(pad_choice_active)]) if pad_choice_active is not None else None

    geff_um = _arc_distance_um_from_voltage(v_chg, cfg=cfg)
    p_fail_single = _compute_p_fail_for_die(
        top_die_w_um,
        top_die_h_um,
        v_chg,
        cfg=cfg,
        geff_um=geff_um,
        weibull_k=weibull_k,
        weibull_lambda=weibull_lambda,
        cutoff_min_a=cutoff_min_a,
    )
    survive_bool = not ((pad_choice is not None) and (float(_esd_rng.uniform(0.0, 1.0)) < p_fail_single))
    return pad_choice, survive_bool


if __name__ == "__main__":
    raise SystemExit("esd_yield_simulator.py expects external cfg and pad inputs; import this module from the D2W flow.")
