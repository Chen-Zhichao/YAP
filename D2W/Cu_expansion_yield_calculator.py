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


def cu_recess_pad_pass_probabilities(mu, lower_limits, upper_limits, sigma):
    """Return independent one-level Cu-recess survival probabilities."""
    lower_limits = np.asarray(lower_limits, dtype=np.float64).reshape(-1)
    upper_limits = np.asarray(upper_limits, dtype=np.float64).reshape(-1)
    if lower_limits.shape != upper_limits.shape:
        raise ValueError("lower_limits and upper_limits must have the same shape.")
    if np.any(lower_limits > upper_limits):
        raise ValueError("Each Cu-recess lower limit must not exceed its upper limit.")

    sigma = float(sigma)
    if sigma < 0.0:
        raise ValueError("Cu-recess standard deviation must be non-negative.")
    if sigma == 0.0:
        return ((mu >= lower_limits) & (mu <= upper_limits)).astype(np.float64)

    pass_probs = (
        ndtr((upper_limits - float(mu)) / sigma)
        - ndtr((lower_limits - float(mu)) / sigma)
    )
    return np.clip(pass_probs, 0.0, 1.0)


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


def _redundant_pad_group_yield(
    redundant_flat_idx,
    redundant_pass_probs,
    pad_bitmap_collection,
):
    redundant_flat_idx = np.asarray(redundant_flat_idx, dtype=np.int64).reshape(-1)
    redundant_pass_probs = np.asarray(
        redundant_pass_probs,
        dtype=np.float64,
    ).reshape(-1)
    if redundant_flat_idx.shape != redundant_pass_probs.shape:
        raise ValueError("Redundant pad indices and pass probabilities must match.")
    if redundant_flat_idx.size == 0:
        return 1.0

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

        group_ids = group_id_per_pad[redundant_flat_idx]
        if np.any(group_ids < 0):
            bad = redundant_flat_idx[group_ids < 0][:5]
            raise ValueError(
                "Redundant pads are missing group IDs, e.g. flat indices "
                f"{bad.tolist()}."
            )

        order = np.argsort(group_ids, kind="mergesort")
        sorted_group_ids = group_ids[order]
        sorted_pass_probs = redundant_pass_probs[order]
        unique_group_ids, group_starts = np.unique(
            sorted_group_ids,
            return_index=True,
        )
        group_stops = np.r_[group_starts[1:], sorted_group_ids.size]

        redundant_yield = 1.0
        for group_id, start, stop in zip(
            unique_group_ids,
            group_starts,
            group_stops,
        ):
            if group_id >= tolerated_mechanical_failures.size:
                raise ValueError(f"Redundant group ID {group_id} has no tolerance entry.")
            redundant_yield *= _at_most_k_failures_yield(
                sorted_pass_probs[start:stop],
                tolerated_mechanical_failures[group_id],
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
        selected_pos = np.searchsorted(redundant_flat_idx, physical_mask)
        in_range = selected_pos < redundant_flat_idx.size
        matches = np.zeros_like(in_range, dtype=bool)
        matches[in_range] = (
            redundant_flat_idx[selected_pos[in_range]] == physical_mask[in_range]
        )
        if not np.all(matches):
            missing = physical_mask[~matches][:5]
            raise ValueError(
                f"Redundant net '{net}' references pads not present in the "
                f"redundant pad bitmap, e.g. {missing.tolist()}."
            )
        redundant_yield *= _at_most_k_failures_yield(
            redundant_pass_probs[selected_pos],
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
        redundant_pad_mask_flat = (
            pad_bitmap_collection['REDUNDANT_PAD_BITMAP'].reshape(-1).astype(bool)
        )
        selected_pad_mask_flat = critical_pad_mask_flat | redundant_pad_mask_flat
        selected_flat_idx = np.flatnonzero(selected_pad_mask_flat)
        critical_flat_idx = np.flatnonzero(critical_pad_mask_flat)
        redundant_flat_idx = np.flatnonzero(redundant_pad_mask_flat)

        if selected_flat_idx.size == 0:
            die_stack.die_yield_per_interface_dict[interface_name]['mechanical'] = 1.0
            continue

        pad_coords = die_stack.interfaces.base_pad_coords_dict.get(interface_name)
        if pad_coords is None:
            raise ValueError(
                f"Interface '{interface_name}' does not have pad coordinates. "
                "Cu mechanical yield modeling needs coordinates for critical "
                "and redundant pads."
            )
        selected_pad_coords = np.asarray(pad_coords, dtype=np.float64)[selected_flat_idx]
        if not np.all(np.isfinite(selected_pad_coords)):
            bad = selected_flat_idx[
                ~np.all(np.isfinite(selected_pad_coords), axis=1)
            ][:5]
            raise ValueError(
                f"Interface '{interface_name}' has selected critical/redundant "
                f"pads with invalid coordinates, e.g. flat indices {bad.tolist()}."
            )
        pad_dishing_bound_array = debond_dishing_intervals_from_coords(
            cfg,
            selected_pad_coords,
        )

        upper_limits = np.clip(
            -pad_dishing_bound_array[:, 0] * 2,
            a_min=None,
            a_max=0,
        )
        lower_limits = np.clip(
            -pad_dishing_bound_array[:, 1] * 2,
            a_min=None,
            a_max=0,
        )

        sigma = float(np.hypot(cfg.TOP_DISH_STD_nm, cfg.BOT_DISH_STD_nm))
        mu = TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm

        critical_pos = np.searchsorted(selected_flat_idx, critical_flat_idx)
        if critical_flat_idx.size > 0:
            critical_pass_probs = cu_recess_pad_pass_probabilities(
                mu,
                lower_limits[critical_pos],
                upper_limits[critical_pos],
                sigma,
            )
            critical_pad_group_yield = _independent_all_pass_yield(
                critical_pass_probs
            )
        else:
            critical_pad_group_yield = 1.0

        redundant_pos = np.searchsorted(selected_flat_idx, redundant_flat_idx)
        if redundant_flat_idx.size > 0:
            redundant_pass_probs = cu_recess_pad_pass_probabilities(
                mu,
                lower_limits[redundant_pos],
                upper_limits[redundant_pos],
                sigma,
            )
            redundant_pad_group_yield = _redundant_pad_group_yield(
                redundant_flat_idx,
                redundant_pass_probs,
                pad_bitmap_collection,
            )
        else:
            redundant_pad_group_yield = 1.0
        mechanical_cu_expansion_yield = float(critical_pad_group_yield * redundant_pad_group_yield)

        die_stack.die_yield_per_interface_dict[interface_name]['mechanical'] = mechanical_cu_expansion_yield
