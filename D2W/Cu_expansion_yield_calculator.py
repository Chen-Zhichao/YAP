#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#### Author: Zhichao Chen
#### Date: Oct 3, 2025

'''
Cu expansion yield calculator for D2W hybrid bonding:
This module contains functions to calculate die-level and pad-level Cu expansion-induced yield
based on Cu dish distribution and pad layout.
'''

import numpy as np
from scipy.special import ndtr
from debond import debond_dishing_intervals_from_coords


def _cfg_float(cfg, key, default):
    value = cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)
    if value in (None, "None"):
        return float(default)
    return float(value)


def _gh_nodes_weights(n: int):
    x, w = np.polynomial.hermite.hermgauss(int(n))
    z = np.sqrt(2.0) * x
    w_norm = w / np.sqrt(np.pi)
    return z, w_norm


def assign_pads_to_blocks(
    PAD_ARR_ROW: int,
    PAD_ARR_COL: int,
    block_size_r: int,
    block_size_c: int,
) -> np.ndarray:
    block_size_r = max(1, int(block_size_r))
    block_size_c = max(1, int(block_size_c))
    pad_rows, pad_cols = np.meshgrid(
        np.arange(int(PAD_ARR_ROW)),
        np.arange(int(PAD_ARR_COL)),
        indexing="ij",
    )
    pad_rows = pad_rows.ravel()
    pad_cols = pad_cols.ravel()
    block_r = pad_rows // block_size_r
    block_c = pad_cols // block_size_c
    return block_r * int(np.ceil(int(PAD_ARR_COL) / block_size_c)) + block_c


def block_indices_from_flat_indices(
    flat_indices: np.ndarray,
    PAD_ARR_COL: int,
    block_size_r: int,
    block_size_c: int,
) -> np.ndarray:
    flat_indices = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
    block_size_r = max(1, int(block_size_r))
    block_size_c = max(1, int(block_size_c))
    PAD_ARR_COL = int(PAD_ARR_COL)
    pad_rows = flat_indices // PAD_ARR_COL
    pad_cols = flat_indices - pad_rows * PAD_ARR_COL
    block_r = pad_rows // block_size_r
    block_c = pad_cols // block_size_c
    block_indices = block_r * int(np.ceil(PAD_ARR_COL / block_size_c)) + block_c
    if block_indices.size and block_indices.max() <= np.iinfo(np.int32).max:
        return block_indices.astype(np.int32, copy=False)
    return block_indices


def _compress_block_bounds(
    a: np.ndarray,
    b: np.ndarray,
    block_indices: np.ndarray,
    counts: np.ndarray | None,
    bound_quantization_nm: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if bound_quantization_nm is None or float(bound_quantization_nm) <= 0.0:
        return a, b, block_indices, counts

    q = float(bound_quantization_nm)
    a_key = np.rint(a / q).astype(np.int64, copy=False)
    b_key = np.rint(b / q).astype(np.int64, copy=False)
    block_key = np.asarray(block_indices, dtype=np.int64).reshape(-1)
    if counts is None:
        counts = np.ones(a.size, dtype=np.int64)
    else:
        counts = np.asarray(counts, dtype=np.int64).reshape(-1)

    order = np.lexsort((b_key, a_key, block_key))
    a_key = a_key[order]
    b_key = b_key[order]
    block_key = block_key[order]
    counts = counts[order]

    is_new = np.empty(a_key.size, dtype=bool)
    is_new[0] = True
    is_new[1:] = (
        (a_key[1:] != a_key[:-1])
        | (b_key[1:] != b_key[:-1])
        | (block_key[1:] != block_key[:-1])
    )
    starts = np.flatnonzero(is_new)
    compressed_counts = np.add.reduceat(counts, starts)
    compressed_a = a_key[starts].astype(np.float64) * q
    compressed_b = b_key[starts].astype(np.float64) * q
    compressed_blocks = block_key[starts]
    if compressed_blocks.size and compressed_blocks.max() <= np.iinfo(np.int32).max:
        compressed_blocks = compressed_blocks.astype(np.int32, copy=False)
    return compressed_a, compressed_b, compressed_blocks, compressed_counts


def cu_recess_die_yield_spatial(
    mu: float,
    a: np.ndarray,
    b: np.ndarray,
    sigma_L: float,
    sigma_T: float,
    sigma_eps: float,
    block_indices: np.ndarray,
    n_gh_outer: int = 40,
    n_gh_inner: int = 40,
    g: float = None,
    counts: np.ndarray | None = None,
    bound_quantization_nm: float | None = None,
) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    block_indices = np.asarray(block_indices, dtype=np.intp).reshape(-1)
    if a.size == 0:
        return 1.0
    if a.shape != b.shape or a.shape != block_indices.shape:
        raise ValueError("a, b, and block_indices must have the same shape.")
    if counts is not None:
        counts = np.asarray(counts, dtype=np.int64).reshape(-1)
        if counts.shape != a.shape:
            raise ValueError("counts must have the same shape as a.")

    a, b, block_indices, counts = _compress_block_bounds(
        a,
        b,
        block_indices,
        counts,
        bound_quantization_nm,
    )

    order = np.argsort(block_indices, kind="mergesort")
    a_s = a[order]
    b_s = b[order]
    bi_s = block_indices[order]
    counts_s = None if counts is None else counts[order].astype(np.float64, copy=False)
    _, blk_start = np.unique(bi_s, return_index=True)
    block_count = blk_start.size

    zT, wT = _gh_nodes_weights(n_gh_inner)
    inv_sE = 1.0 / float(sigma_eps) if sigma_eps > 0 else 0.0
    p_buf = np.empty(a_s.size, dtype=np.float64)
    q_buf = np.empty(a_s.size, dtype=np.float64)
    log_blk = np.empty((block_count, int(n_gh_inner)), dtype=np.float64)

    if g is not None:
        g_vals = [float(g)]
        g_weights = None
    else:
        zG, wG = _gh_nodes_weights(n_gh_outer)
        g_vals = zG
        g_weights = wG

    log_outer = np.empty(len(g_vals), dtype=np.float64)
    for k, g_val in enumerate(g_vals):
        mu_g = float(mu) + float(sigma_L) * float(g_val)

        for j, z_t in enumerate(zT):
            mean_j = mu_g + float(sigma_T) * float(z_t)
            if sigma_eps > 0:
                np.subtract(b_s, mean_j, out=p_buf)
                p_buf *= inv_sE
                ndtr(p_buf, out=p_buf)
                np.subtract(a_s, mean_j, out=q_buf)
                q_buf *= inv_sE
                ndtr(q_buf, out=q_buf)
                np.subtract(p_buf, q_buf, out=p_buf)
            else:
                p_buf[:] = np.where((mean_j >= a_s) & (mean_j <= b_s), 1.0, 0.0)

            np.clip(p_buf, 1e-300, 1.0, out=p_buf)
            np.log(p_buf, out=p_buf)
            if counts_s is not None:
                p_buf *= counts_s
            log_blk[:, j] = np.add.reduceat(p_buf, blk_start)

        max_log_blk = log_blk.max(axis=1, keepdims=True)
        log_E = max_log_blk[:, 0] + np.log(
            np.sum(wT[None, :] * np.exp(log_blk - max_log_blk), axis=1)
        )
        log_outer[k] = log_E.sum()

    if g is not None:
        return float(np.clip(np.exp(log_outer[0]), 0.0, 1.0))

    max_log_outer = float(log_outer.max())
    die_yield = float(
        np.sum(g_weights * np.exp(log_outer - max_log_outer)) * np.exp(max_log_outer)
    )
    return float(np.clip(die_yield, 0.0, 1.0))


def _dish_sigma_components_nm(cfg):
    top_L = _cfg_float(cfg, "TOP_DISH_STD_L_nm", 0.0)
    top_T = _cfg_float(cfg, "TOP_DISH_STD_T_nm", 0.0)
    top_E = _cfg_float(cfg, "TOP_DISH_STD_E_nm", 0.0)
    bot_L = _cfg_float(cfg, "BOT_DISH_STD_L_nm", 0.0)
    bot_T = _cfg_float(cfg, "BOT_DISH_STD_T_nm", 0.0)
    bot_E = _cfg_float(cfg, "BOT_DISH_STD_E_nm", 0.0)

    sigma_L = np.sqrt(top_L**2 + bot_L**2)
    sigma_T = np.sqrt(top_T**2 + bot_T**2)
    sigma_eps = np.sqrt(top_E**2 + bot_E**2)
    return float(sigma_L), float(sigma_T), float(sigma_eps)


def _independent_all_pass_yield(pass_probs):
    pass_probs = np.asarray(pass_probs, dtype=np.float64).reshape(-1)
    if pass_probs.size == 0:
        return 1.0
    if np.any(pass_probs <= 0.0):
        return 0.0
    return float(np.exp(np.sum(np.log(np.clip(pass_probs, 1e-300, 1.0)))))


def _at_most_k_failures_yield(pass_probs, tolerated_failures):
    pass_probs = np.asarray(pass_probs, dtype=np.float64).reshape(-1)
    tolerated_failures = int(tolerated_failures)

    if pass_probs.size == 0:
        return 1.0
    if tolerated_failures < 0:
        return 0.0
    if tolerated_failures >= pass_probs.size:
        return 1.0
    if tolerated_failures == 0:
        return _independent_all_pass_yield(pass_probs)

    fail_probs = 1.0 - pass_probs
    pmf = np.zeros(tolerated_failures + 1, dtype=np.float64)
    pmf[0] = 1.0

    for fail_prob in np.clip(fail_probs, 0.0, 1.0):
        pass_prob = 1.0 - fail_prob
        next_pmf = pmf * pass_prob
        next_pmf[1:] += pmf[:-1] * fail_prob
        pmf = next_pmf

    return float(np.clip(np.sum(pmf), 0.0, 1.0))


def _redundant_pad_group_yield(pad_pass_prob_full, pad_bitmap_collection):
    group_id_per_pad = pad_bitmap_collection.get("redundant_group_id_per_pad")
    tolerated_mechanical_failures = pad_bitmap_collection.get(
        "redundant_tolerated_mechanical_failures"
    )

    if group_id_per_pad is not None and tolerated_mechanical_failures is not None:
        group_id_per_pad = np.asarray(group_id_per_pad, dtype=np.int64).reshape(-1)
        tolerated_mechanical_failures = np.asarray(
            tolerated_mechanical_failures,
            dtype=np.int64,
        ).reshape(-1)

        redundant_yield = 1.0
        for group_id, tolerated_failures in enumerate(tolerated_mechanical_failures):
            group_mask = group_id_per_pad == group_id
            if not np.any(group_mask):
                continue
            redundant_yield *= _at_most_k_failures_yield(
                pad_pass_prob_full[group_mask],
                tolerated_failures,
            )
            if redundant_yield <= 0.0:
                return 0.0
        return float(np.clip(redundant_yield, 0.0, 1.0))

    redundant_yield = 1.0
    redundant_net_to_1d_physical_mask = pad_bitmap_collection.get(
        "redundant_net_to_1d_physical_mask",
        {},
    )
    criticality_info = pad_bitmap_collection.get("criticality_info", {})

    for net, physical_mask in redundant_net_to_1d_physical_mask.items():
        physical_mask = np.asarray(physical_mask, dtype=np.int64).reshape(-1)
        physical_mask = physical_mask[physical_mask >= 0]
        if physical_mask.size == 0:
            continue
        tolerated_failures = criticality_info[net]["tolerated_mechanical_failures"]
        redundant_yield *= _at_most_k_failures_yield(
            pad_pass_prob_full[physical_mask],
            tolerated_failures,
        )
        if redundant_yield <= 0.0:
            return 0.0
    return float(np.clip(redundant_yield, 0.0, 1.0))


def stack_cu_expansion_yield_calculator(*, cfg_dict, die_stack):
    for interface_name, cfg in cfg_dict.items():
        TOP_DISH_MEAN_nm = cfg.TOP_DISH_MEAN_nm
        BOT_DISH_MEAN_nm = cfg.BOT_DISH_MEAN_nm

        interface = die_stack.interfaces.interface_dict[interface_name]
        pad_bitmap_collection = die_stack.interfaces.pad_bitmap_collection_dict[interface_name]
        critical_pad_mask_flat = (
            pad_bitmap_collection['CRITICAL_PAD_BITMAP'].reshape(-1).astype(bool)
        )
        critical_flat_idx = np.flatnonzero(critical_pad_mask_flat)

        if critical_flat_idx.size == 0:
            die_stack.die_yield_per_interface_dict[interface_name]['mechanical'] = 1.0
            continue

        pad_coords = interface.pad_coords
        critical_pad_coords = pad_coords[critical_pad_mask_flat]
        critical_pad_dishing_bound_array = debond_dishing_intervals_from_coords(cfg, critical_pad_coords) # (num_pads, 2) array: (dishing_low_nm, dishing_high_nm)

        upper_cu_height_limits_critical_pads = - critical_pad_dishing_bound_array[:, 0] * 2 # - upper Cu height limits
        lower_cu_height_limits_critical_pads = - critical_pad_dishing_bound_array[:, 1] * 2 # - lower Cu height limits
        lower_cu_height_limits_critical_pads = np.clip(lower_cu_height_limits_critical_pads, a_max=0, a_min=None)
        upper_cu_height_limits_critical_pads = np.clip(upper_cu_height_limits_critical_pads, a_max=0, a_min=None)  # Clip to ensure upper Cu height limits are <= 0

        block_size_r = max(
            1,
            int(round(_cfg_float(cfg, "TL_um", 200.0) / cfg.PITCH_r_um)),
        )
        block_size_c = max(
            1,
            int(round(_cfg_float(cfg, "TL_um", 200.0) / cfg.PITCH_c_um)),
        )
        block_idx_critical = block_indices_from_flat_indices(
            critical_flat_idx,
            cfg.PAD_ARR_COL,
            block_size_r,
            block_size_c,
        )
        sigma_L, sigma_T, sigma_eps = _dish_sigma_components_nm(cfg)

        critical_pad_group_yield = cu_recess_die_yield_spatial(
            mu=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm,
            a=lower_cu_height_limits_critical_pads,
            b=upper_cu_height_limits_critical_pads,
            sigma_L=sigma_L,
            sigma_T=sigma_T,
            sigma_eps=sigma_eps,
            block_indices=block_idx_critical,
            bound_quantization_nm=_cfg_float(
                cfg,
                "CU_RECESS_BOUND_QUANTIZATION_NM",
                0.001,
            ),
        )

        redundant_pad_group_yield = 1.0     # NOTE: Currently we don't consider redundant pads.
        # redundant_pad_group_yield = _redundant_pad_group_yield(
        #     pad_pass_prob_full,
        #     pad_bitmap_collection,
        # )
        mechanical_cu_expansion_yield = float(critical_pad_group_yield * redundant_pad_group_yield)

        die_stack.die_yield_per_interface_dict[interface_name]['mechanical'] = mechanical_cu_expansion_yield
