#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from esd_yield_calculator import (
    _arc_distance_um_from_voltage,
    _cfg_bool,
    _cfg_float,
    _compute_p_fail_for_die,
    _w2w_pad_contact_limit_um,
    _w2w_warpage_mean_um_from_cfg,
    _w2w_warpage_std_um_from_cfg,
    center_die_indices,
)


def choose_center_die_index(
    die_list: Sequence,
    *,
    rng: np.random.Generator | None = None,
    tolerance_um: float | None = None,
) -> int | None:
    """Randomly choose one of the symmetric wafer-center first-contact dies."""
    indices = center_die_indices(die_list, tolerance_um=tolerance_um)
    if indices.size <= 0:
        return None
    if indices.size == 1:
        return int(indices[0])
    if rng is None:
        rng = np.random.default_rng()
    return int(indices[int(rng.integers(0, indices.size))])


def _active_pad_ids_from_bitmap(
    pad_coords_um: np.ndarray,
    dummy_pad_bitmap: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pad_coords_um = np.asarray(pad_coords_um, dtype=np.float64)
    if pad_coords_um.ndim != 2 or pad_coords_um.shape[1] != 2:
        raise ValueError("pad_coords_um must have shape (n_pads, 2).")

    dummy_pad_bitmap = np.asarray(dummy_pad_bitmap, dtype=bool).reshape(-1)
    if pad_coords_um.shape[0] != dummy_pad_bitmap.shape[0]:
        raise ValueError("pad_coords_um and dummy_pad_bitmap must have the same length.")

    finite_mask = np.isfinite(pad_coords_um[:, 0]) & np.isfinite(pad_coords_um[:, 1])
    active_ids = np.flatnonzero(finite_mask & ~dummy_pad_bitmap)
    if active_ids.size <= 0:
        raise ValueError("dummy_pad_bitmap masks out all finite pads.")
    return pad_coords_um, active_ids


def _candidate_pad_ids(
    top_dish_um_raw: np.ndarray,
    bot_dish_um: np.ndarray,
    arc_distance_um: float,
) -> np.ndarray:
    arc_margin_um = max(0.0, float(arc_distance_um))
    return np.where((np.asarray(top_dish_um_raw) + np.asarray(bot_dish_um)) >= (-arc_margin_um))[0]


def _best_pad_with_spherical_gap(
    *,
    pad_coords_um: np.ndarray,
    pad_ids: np.ndarray,
    pad_size_um: float,
    wafer_radius_um: float,
    warpage_um: float,
    z_top_um: float,
    top_dish_um_raw: np.ndarray,
    bot_dish_um: np.ndarray,
    arc_distance_um: float,
    rng_pick: np.random.Generator,
    exact_sphere: bool,
    atol_gap: float = 1e-12,
) -> Tuple[int, float]:
    if pad_ids.size <= 0:
        pad_ids = np.arange(pad_coords_um.shape[0], dtype=np.int64)
    if pad_ids.size <= 0:
        raise ValueError("pad_coords_um is empty; cannot choose an ESD first-arcing pad.")

    contact_limit_um = _w2w_pad_contact_limit_um(
        pad_coords_um=pad_coords_um[pad_ids],
        pad_size_um=pad_size_um,
        wafer_radius_um=wafer_radius_um,
        warpage_um=warpage_um,
        z_top_um=z_top_um,
        exact_sphere=exact_sphere,
    )
    margin_gap_um = (
        contact_limit_um
        - np.asarray(top_dish_um_raw, dtype=np.float64)[pad_ids]
        - np.asarray(bot_dish_um, dtype=np.float64)[pad_ids]
        - max(0.0, float(arc_distance_um))
    )
    best_gap = float(np.min(margin_gap_um))
    is_best = np.isclose(margin_gap_um, best_gap, rtol=0.0, atol=atol_gap)
    best_ids = pad_ids[is_best]
    pick = int(rng_pick.integers(0, best_ids.size)) if best_ids.size > 1 else 0
    return int(best_ids[pick]), best_gap


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
    wafer_radius_um: float | None = None,
    warpage_um: float | None = None,
    tilt_x_mean_deg: float | None = None,
    tilt_x_std_deg: float | None = None,
    tilt_y_mean_deg: float | None = None,
    tilt_y_std_deg: float | None = None,
) -> Tuple[int | None, bool]:
    """
    Run one W2W ESD experiment for the die already selected by center symmetry.

    ``pad_coords_um`` must be wafer-level pad coordinates for this die. The
    returned pad index is relative to the supplied compressed pad list.
    """
    del tilt_x_mean_deg, tilt_x_std_deg, tilt_y_mean_deg, tilt_y_std_deg

    pad_size_um = float(pad_size_um)
    top_die_w_um = float(top_die_w_um)
    top_die_h_um = float(top_die_h_um)
    if wafer_radius_um is None:
        wafer_radius_um = _cfg_float(cfg, ["WAF_R_um"], 75000.0)
    wafer_radius_um = float(wafer_radius_um)

    z_top_um = _cfg_float(cfg, ["ESD_Z_TOP_UM"], 0.1)
    v_min_v = _cfg_float(cfg, ["V_MIN_V"], 0.0)
    v_max_v = _cfg_float(cfg, ["V_MAX_V"], 5.0)
    weibull_k = _cfg_float(cfg, ["WEIBULL_K"], 4.44985)
    weibull_lambda = _cfg_float(cfg, ["WEIBULL_LAMBDA"], 0.0621816)
    cutoff_min_a = _cfg_float(cfg, ["CUTOFF_MIN_A"], 0.0)
    exact_sphere = _cfg_bool(cfg, ["ESD_W2W_USE_EXACT_SPHERE", "W2W_ESD_USE_EXACT_SPHERE"], True)

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

    v_chg = float(rng.uniform(v_min_v, v_max_v))
    arc_distance_um = _arc_distance_um_from_voltage(v_chg, cfg=cfg)
    if warpage_um is None:
        warpage_mean_um = _w2w_warpage_mean_um_from_cfg(cfg)
        warpage_std_um = _w2w_warpage_std_um_from_cfg(cfg)
        if warpage_std_um > 0.0:
            warpage_um = abs(float(rng.normal(warpage_mean_um, warpage_std_um)))
        else:
            warpage_um = abs(float(warpage_mean_um))
    else:
        warpage_um = abs(float(warpage_um))

    top_dish_um_raw = top_dish_nm_ext[active_ids] * 1e-3
    bot_dish_um = bot_dish_nm_ext[active_ids] * 1e-3
    candidate_pad_ids = _candidate_pad_ids(top_dish_um_raw, bot_dish_um, arc_distance_um)

    pad_choice_active, arc_margin_gap_um = _best_pad_with_spherical_gap(
        pad_coords_um=active_pad_coords_um,
        pad_ids=candidate_pad_ids,
        pad_size_um=pad_size_um,
        wafer_radius_um=wafer_radius_um,
        warpage_um=warpage_um,
        z_top_um=z_top_um,
        top_dish_um_raw=top_dish_um_raw,
        bot_dish_um=bot_dish_um,
        arc_distance_um=arc_distance_um,
        rng_pick=rng_pick,
        exact_sphere=exact_sphere,
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
    raise SystemExit("esd_yield_simulator.py expects external cfg and pad inputs; import it from the W2W flow.")
