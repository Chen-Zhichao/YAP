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
from scipy.stats import norm
from debond import debond_dishing_intervals_from_coords


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
        TOP_DISH_STD_nm = cfg.TOP_DISH_STD_nm
        BOT_DISH_MEAN_nm = cfg.BOT_DISH_MEAN_nm
        BOT_DISH_STD_nm = cfg.BOT_DISH_STD_nm

        interface = die_stack.interfaces.interface_dict[interface_name]
        pad_bitmap_collection = die_stack.interfaces.pad_bitmap_collection_dict[interface_name]
        valid_pad_mask = (pad_bitmap_collection['CRITICAL_PAD_BITMAP'] == 1) | \
                         (pad_bitmap_collection['REDUNDANT_PAD_BITMAP'] == 1) | \
                         (pad_bitmap_collection['DUMMY_PAD_BITMAP'] == 1)
        pad_coords = interface.pad_coords
        valid_pad_mask_flat = valid_pad_mask.reshape(-1)
        valid_die_pad_coords = pad_coords[valid_pad_mask_flat]

        if valid_die_pad_coords.size == 0:
            die_stack.die_yield_per_interface_dict[interface_name]['mechanical'] = 1.0
            continue

        valid_pad_dishing_bound_array = debond_dishing_intervals_from_coords(cfg, valid_die_pad_coords) # (num_pads, 2) array: (dishing_low_nm, dishing_high_nm)

        upper_cu_height_limits_valid_pads = - valid_pad_dishing_bound_array[:, 0] * 2 # - upper Cu height limits
        lower_cu_height_limits_valid_pads = - valid_pad_dishing_bound_array[:, 1] * 2 # - lower Cu height limits
        lower_cu_height_limits_valid_pads = np.clip(lower_cu_height_limits_valid_pads, a_max=0, a_min=None)
        upper_cu_height_limits_valid_pads = np.clip(upper_cu_height_limits_valid_pads, a_max=0, a_min=None)  # Clip to ensure upper Cu height limits are <= 0
        
        # Calculate the probability of survival for each valid pad
        pos_valid_pads = norm.cdf(
                            upper_cu_height_limits_valid_pads, 
                            loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, 
                            scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2)
                          ) - \
                          norm.cdf(
                              lower_cu_height_limits_valid_pads, 
                              loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, 
                              scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2)
                          )
        pos_valid_pads = np.clip(pos_valid_pads, 0.0, 1.0)

        pad_pass_prob_full = np.ones(valid_pad_mask_flat.shape[0], dtype=np.float64)
        pad_pass_prob_full[valid_pad_mask_flat] = pos_valid_pads

        critical_pad_mask_flat = pad_bitmap_collection['CRITICAL_PAD_BITMAP'].reshape(-1).astype(bool)
        critical_pad_group_yield = _independent_all_pass_yield(
            pad_pass_prob_full[critical_pad_mask_flat]
        )
        redundant_pad_group_yield = 1.0     # NOTE: Currently we don't consider redundant pads.
        # redundant_pad_group_yield = _redundant_pad_group_yield(
        #     pad_pass_prob_full,
        #     pad_bitmap_collection,
        # )
        mechanical_cu_expansion_yield = float(critical_pad_group_yield * redundant_pad_group_yield)

        die_stack.die_yield_per_interface_dict[interface_name]['mechanical'] = mechanical_cu_expansion_yield
