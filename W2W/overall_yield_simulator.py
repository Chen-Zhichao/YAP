#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Overall yield simulator for hybrid bonding
#### Author: Zhichao Chen
#### Date: Feb 5, 2026

import os
import sys
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
import time
import matplotlib.pyplot as plt
from joblib import dump, load
import yaml

from Cu_gap_simulator import Cu_gap_simulator
from esd_yield_simulator import choose_center_die_index, esd_failure_simulator
from debond import debond_dishing_intervals_from_coords
from Cu_expansion_yield_calculator import stack_stress_yield_calculator
from yield_mechanism_policy import FAILURE_MECHANISMS, active_failure_mechanisms


_CLEAR_LINE = "\033[K"
_MECHANICAL_DIE_YIELD_CACHE = {}


def _print_status(message: str, enabled: bool):
    if enabled:
        print(f"\r{message}{_CLEAR_LINE}", flush=True)


def _print_progress(message: str, enabled: bool):
    if enabled:
        print(f"\r{message}{_CLEAR_LINE}", end="", flush=True)


def _clear_progress_line(enabled: bool):
    if enabled:
        print(f"\r{_CLEAR_LINE}", end="", flush=True)


def total_memory_mb(obj):
    total = sys.getsizeof(obj)
    if isinstance(obj, list):
        for item in obj:
            try:
                # numpy arrays
                total += item.nbytes
            except AttributeError:
                # fallback
                total += sys.getsizeof(item)
    return total / 1024 / 1024  # MB


def _increment_redundant_group_counts(
    new_fail_mask,
    group_id_source,
    redundant_failed_counts,
):
    if (
        group_id_source is None
        or redundant_failed_counts is None
        or not np.any(new_fail_mask)
    ):
        return False

    group_ids = np.asarray(group_id_source[new_fail_mask], dtype=np.int64).reshape(-1)
    group_ids = group_ids[group_ids >= 0]
    if group_ids.size <= 0:
        return False

    redundant_failed_counts += np.bincount(
        group_ids,
        minlength=redundant_failed_counts.shape[0],
    ).astype(redundant_failed_counts.dtype, copy=False)
    return True


def _group_limit_exceeded(redundant_failed_counts, tolerated_failures):
    if redundant_failed_counts is None or tolerated_failures is None:
        return False
    if redundant_failed_counts.shape[0] == 0:
        return False
    return bool(np.any(redundant_failed_counts > tolerated_failures))


def _corner_worst_overlay_misalignment_um(
    corner_coords_um,
    system_translation_x_um,
    system_translation_y_um,
    system_rotation_rad,
    system_magnification_ppm,
):
    coords = np.asarray(corner_coords_um, dtype=float)
    dx = (
        system_translation_x_um
        - system_rotation_rad * coords[:, 1]
        + system_magnification_ppm * coords[:, 0]
    )
    dy = (
        system_translation_y_um
        + system_rotation_rad * coords[:, 0]
        + system_magnification_ppm * coords[:, 1]
    )
    return float(np.max(np.sqrt(dx**2 + dy**2)))


def _overlay_boundary_coords(die):
    coords = getattr(die, "ovl_active_pad_boundary_coords", None)
    if coords is None:
        return die.pad_array_box
    return coords


def _mechanical_die_yield_cache_key(cfg, pad_bitmap_collection):
    dish_keys = (
        "TOP_DISH_MEAN_nm",
        "TOP_DISH_STD_nm",
        "BOT_DISH_MEAN_nm",
        "BOT_DISH_STD_nm",
    )
    cfg_part = tuple(
        (key, round(float(getattr(cfg, key, 0.0)), 12))
        for key in dish_keys
    )
    return (
        str(getattr(cfg, "DESIGN", "")),
        str(getattr(cfg, "INTERFACE", "")),
        int(getattr(cfg, "PAD_ARR_ROW")),
        int(getattr(cfg, "PAD_ARR_COL")),
        round(float(getattr(cfg, "PITCH_r_um")), 12),
        round(float(getattr(cfg, "PITCH_c_um")), 12),
        id(pad_bitmap_collection),
        cfg_part,
    )


def overall_yield_simulator(
    input_args: dict,
    cfg_dict: dict,
    epoch: int,
    waf_stack_list: list,
    num_dies_per_wafer: int,
    pad_bitmap_collection_dict: dict,
):
    die_stack_yield_list = []
    # print("The memory size of the waf_list is {} MB.".format(total_memory_mb(waf_list)))

    # Read the parameters
    NUM_STACKS = len(waf_stack_list)
    verbose = bool(input_args.get('verbose', False))
    global_stack_offset = int(input_args.get('global_stack_offset', epoch * NUM_STACKS))
    active_mechanisms = active_failure_mechanisms(input_args)
    run_overlay = 'overlay' in active_mechanisms
    run_particle = 'particle' in active_mechanisms
    run_mechanical = 'mechanical' in active_mechanisms
    run_esd = 'ESD' in active_mechanisms
    overlay_only_fast_path = active_mechanisms == {'overlay'}
    mechanical_only_fast_path = active_mechanisms == {'mechanical'}
    esd_only_fast_path = active_mechanisms == {'ESD'}

    epoch_fail_map_per_interface_dict = {}    # This dict stores the fail bump maps for all die samples in this epoch for each mechanism
    epoch_fail_vec_per_interface_dict = {}    # This dict stores failure reason (each mechanism) for all die samples in this epoch
    failure_mechanism_list = list(FAILURE_MECHANISMS) + ['overall']

    if verbose:
        for interface_name, cfg in cfg_dict.items():
            epoch_fail_map_per_interface_dict[interface_name], epoch_fail_vec_per_interface_dict[interface_name] = {}, {}
            for failure_mechanism in failure_mechanism_list:
                epoch_fail_map_per_interface_dict[interface_name][failure_mechanism] = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL))
                epoch_fail_vec_per_interface_dict[interface_name][failure_mechanism] = np.zeros((NUM_STACKS, num_dies_per_wafer))
    
    for stack_ind, waf_stack in enumerate(waf_stack_list):
        for interface_ind, (interface_name, waf_interface) in enumerate(waf_stack.interfaces.interface_dict.items()):
            # Read the configuration and pad_bitmap_collection data for this interface
            pad_bitmap_collection = pad_bitmap_collection_dict[interface_name]
            cfg = cfg_dict[interface_name]
            cfg.num_dies_per_wafer = num_dies_per_wafer
            _print_status(
                "Simulating stack {}/{} interface {}/{}: {} ...".format(
                    global_stack_offset + stack_ind + 1,
                    cfg.NUM_WAFER_STACKS,
                    interface_ind + 1,
                    len(waf_stack.interfaces.interface_dict),
                    interface_name,
                ),
                verbose,
            )
            # Read the parameters needed for this interface
            WAF_R_um                        =       cfg.WAF_R_um
            failure_params                   =       waf_stack.interfaces.failure_params_dict[interface_name]
            system_translation_x_um         =       failure_params.get('system_translation_x_um', 0.0)
            system_translation_y_um         =       failure_params.get('system_translation_y_um', 0.0)
            system_rotation_rad             =       failure_params.get('system_rotation_rad', 0.0)
            system_magnification_ppm        =       failure_params.get('system_magnification_ppm', 0.0)
            MAX_ALLOWED_MISALIGNMENT_um     =       failure_params.get('MAX_ALLOWED_MISALIGNMENT_um', np.inf)
            RANDOM_MISALIGNMENT_MEAN_um     =       cfg.RANDOM_MISALIGNMENT_MEAN_um
            RANDOM_MISALIGNMENT_STD_um      =       cfg.RANDOM_MISALIGNMENT_STD_um
            PAD_ARR_W_um, PAD_ARR_L_um      =       cfg.PAD_ARR_W_um, cfg.PAD_ARR_L_um
            PAD_ARR_ROW, PAD_ARR_COL        =       cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL
            TILT_X_MEAN_DEG, TILT_X_STD_DEG    =       cfg.TILT_X_MEAN_DEG, cfg.TILT_X_STD_DEG
            TILT_Y_MEAN_DEG, TILT_Y_STD_DEG    =      cfg.TILT_Y_MEAN_DEG, cfg.TILT_Y_STD_DEG
            PITCH_r_um, PITCH_c_um          =       cfg.PITCH_r_um, cfg.PITCH_c_um
            PAD_TOP_R_um                    =       cfg.PAD_TOP_R_um
            approximate_set                 =       cfg.approximate_set

            # Record the time
            start_time = time.time()
            die_count = 0
            # Read the critical pad bitmap
            die_critical_pad_bitmap = pad_bitmap_collection["CRITICAL_PAD_BITMAP"]
            # Read the redundant critical pad bitmap
            die_redundant_pad_bitmap = pad_bitmap_collection["REDUNDANT_PAD_BITMAP"]
            # Read the ESD-critical pad bitmap
            die_esd_critical_pad_bitmap = pad_bitmap_collection["ESD_CRITICAL_PAD_BITMAP"]
            # Read the redundant net to bump ids mapping
            redundant_net_to_bumpids = pad_bitmap_collection["redundant_net_to_bumpids"]
            valid_pad_mask = (
                (pad_bitmap_collection['CRITICAL_PAD_BITMAP'] == 1)
                | (pad_bitmap_collection['REDUNDANT_PAD_BITMAP'] == 1)
                | (pad_bitmap_collection['DUMMY_PAD_BITMAP'] == 1)
                | (pad_bitmap_collection.get(
                    'POWER_GROUND_PAD_BITMAP',
                    np.zeros_like(pad_bitmap_collection['CRITICAL_PAD_BITMAP'], dtype=bool),
                ) == 1)
            )
            valid_pad_mask_flat = valid_pad_mask.flatten()
            valid_linear_idx = np.flatnonzero(valid_pad_mask_flat)
            valid_dummy_pad_bitmap = pad_bitmap_collection['DUMMY_PAD_BITMAP'].flatten()[valid_pad_mask_flat]
            # Read the mapping from physical pad location to bump id
            mapping_physical_to_bumpid = pad_bitmap_collection["mapping_physical_to_bumpid"]
            # Read the criticality info
            criticality_info = pad_bitmap_collection["criticality_info"]
            # Read the redundant net to 1D physical mask mapping
            redundant_net_to_1d_physical_mask = pad_bitmap_collection["redundant_net_to_1d_physical_mask"]
            redundant_group_id_per_pad = pad_bitmap_collection.get("redundant_group_id_per_pad")
            redundant_tolerated_esd_failures = pad_bitmap_collection.get(
                "redundant_tolerated_esd_failures"
            )
            redundant_tolerated_mechanical_failures = pad_bitmap_collection.get(
                "redundant_tolerated_mechanical_failures"
            )
            if redundant_group_id_per_pad is not None:
                redundant_group_id_grid = np.asarray(
                    redundant_group_id_per_pad,
                    dtype=np.int32,
                ).reshape(PAD_ARR_ROW, PAD_ARR_COL)
            else:
                redundant_group_id_grid = None
            esd_center_tol_um = getattr(cfg, "ESD_CENTER_TOL_UM", None)
            selected_esd_die_ind = (
                choose_center_die_index(
                    waf_interface.die_list,
                    tolerance_um=(
                        None if esd_center_tol_um is None else float(esd_center_tol_um)
                    ),
                )
                if run_esd
                else None
            )

            if esd_only_fast_path:
                if selected_esd_die_ind is None:
                    _print_status(
                        "ESD-only fast path for stack {}/{} interface {}/{}: "
                        "no center die selected.".format(
                            global_stack_offset + stack_ind + 1,
                            cfg.NUM_WAFER_STACKS,
                            interface_ind + 1,
                            len(waf_stack.interfaces.interface_dict),
                        ),
                        verbose,
                    )
                    continue

                die_ind = int(selected_esd_die_ind)
                die = waf_interface.die_list[die_ind]
                die_pad_coords = waf_interface.base_pad_coords + die.die_center
                valid_die_pad_coords = die_pad_coords[valid_pad_mask_flat]
                top_dish, bot_dish = Cu_gap_simulator(
                    cfg=cfg,
                    valid_pad_mask_flat=valid_pad_mask_flat,
                )
                first_contact_pad_idx, survive_bool = esd_failure_simulator(
                    cfg=cfg,
                    pad_coords_um=valid_die_pad_coords,
                    pad_size_um=PAD_TOP_R_um * 2,
                    top_die_w_um=die.DIE_W_um,
                    top_die_h_um=die.DIE_L_um,
                    wafer_radius_um=WAF_R_um,
                    top_dish_nm_ext=top_dish,
                    bot_dish_nm_ext=bot_dish,
                    dummy_pad_bitmap=valid_dummy_pad_bitmap,
                )

                failed = False
                if first_contact_pad_idx is not None and survive_bool == False:
                    full_linear_idx = int(valid_linear_idx[int(first_contact_pad_idx)])
                    r_idx = full_linear_idx // PAD_ARR_COL
                    c_idx = full_linear_idx % PAD_ARR_COL
                    if cfg.verbose:
                        epoch_fail_map_per_interface_dict[interface_name]['ESD'][r_idx, c_idx] += 1
                        epoch_fail_map_per_interface_dict[interface_name]['overall'][r_idx, c_idx] += 1
                    if die_esd_critical_pad_bitmap[r_idx, c_idx] == 1:
                        failed = True
                        waf_stack.die_stack_survival[die_ind] = False
                        waf_interface.die_list[die_ind].survival = False
                        if cfg.verbose:
                            epoch_fail_vec_per_interface_dict[interface_name]['ESD'][stack_ind, die_ind] = 1
                            epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind, die_ind] = 1

                _print_status(
                    "ESD-only fast path for stack {}/{} interface {}/{}: "
                    "{} center die {}.".format(
                        global_stack_offset + stack_ind + 1,
                        cfg.NUM_WAFER_STACKS,
                        interface_ind + 1,
                        len(waf_stack.interfaces.interface_dict),
                        interface_name,
                        "failed" if failed else "survived",
                    ),
                    verbose,
                )
                continue

            if mechanical_only_fast_path:
                cache_key = _mechanical_die_yield_cache_key(cfg, pad_bitmap_collection)
                if cache_key not in _MECHANICAL_DIE_YIELD_CACHE:
                    if not hasattr(waf_stack, "die_yield_list_per_interface_dict"):
                        waf_stack.die_yield_list_per_interface_dict = {
                            name: {"mechanical": np.ones(num_dies_per_wafer, dtype=float)}
                            for name in waf_stack.interfaces.interface_dict
                        }
                    stack_stress_yield_calculator(
                        cfg_dict={interface_name: cfg},
                        waf_stack=waf_stack,
                    )
                    mechanical_yield_array = waf_stack.die_yield_list_per_interface_dict[
                        interface_name
                    ]['mechanical']
                    _MECHANICAL_DIE_YIELD_CACHE[cache_key] = float(
                        np.nanmean(mechanical_yield_array)
                    )
                mechanical_die_yield = _MECHANICAL_DIE_YIELD_CACHE[cache_key]

                candidate_mask = (
                    np.ones(num_dies_per_wafer, dtype=bool)
                    if cfg.verbose
                    else waf_stack.die_stack_survival.astype(bool, copy=True)
                )
                candidate_idx = np.flatnonzero(candidate_mask)
                fail_mask = np.zeros(num_dies_per_wafer, dtype=bool)
                if candidate_idx.size > 0:
                    fail_mask[candidate_idx] = (
                        np.random.random(candidate_idx.size) > mechanical_die_yield
                    )

                if np.any(fail_mask):
                    waf_stack.die_stack_survival[fail_mask] = False
                    for failed_die_ind in np.flatnonzero(fail_mask):
                        waf_interface.die_list[int(failed_die_ind)].survival = False
                    if cfg.verbose:
                        epoch_fail_vec_per_interface_dict[interface_name]['mechanical'][
                            stack_ind, fail_mask
                        ] = 1
                        epoch_fail_vec_per_interface_dict[interface_name]['overall'][
                            stack_ind, fail_mask
                        ] = 1

                _print_status(
                    "Mechanical-only fast path for stack {}/{} interface {}/{}: "
                    "{} yield {:.6f}, failed {}/{} dies.".format(
                        global_stack_offset + stack_ind + 1,
                        cfg.NUM_WAFER_STACKS,
                        interface_ind + 1,
                        len(waf_stack.interfaces.interface_dict),
                        interface_name,
                        mechanical_die_yield,
                        int(np.count_nonzero(fail_mask)),
                        int(candidate_idx.size),
                    ),
                    verbose,
                )
                continue

            if overlay_only_fast_path:
                candidate_mask = (
                    np.ones(num_dies_per_wafer, dtype=bool)
                    if cfg.verbose
                    else waf_stack.die_stack_survival.astype(bool, copy=True)
                )
                candidate_idx = np.flatnonzero(candidate_mask)
                fail_mask = np.zeros(num_dies_per_wafer, dtype=bool)
                if candidate_idx.size > 0:
                    die_list = waf_interface.die_list
                    corner_coords = np.asarray(
                        [_overlay_boundary_coords(die_list[int(idx)]) for idx in candidate_idx],
                        dtype=float,
                    )
                    if corner_coords.shape[1] > 0:
                        x = corner_coords[:, :, 0]
                        y = corner_coords[:, :, 1]
                        dx = (
                            system_translation_x_um
                            - system_rotation_rad * y
                            + system_magnification_ppm * x
                        )
                        dy = (
                            system_translation_y_um
                            + system_rotation_rad * x
                            + system_magnification_ppm * y
                        )
                        worst_boundary_misalignment = np.max(
                            np.sqrt(dx**2 + dy**2),
                            axis=1,
                        )
                        worst_boundary_misalignment += np.random.normal(
                            RANDOM_MISALIGNMENT_MEAN_um,
                            RANDOM_MISALIGNMENT_STD_um,
                            size=candidate_idx.size,
                        )
                        fail_mask[candidate_idx] = (
                            worst_boundary_misalignment >= MAX_ALLOWED_MISALIGNMENT_um
                        )

                if np.any(fail_mask):
                    waf_stack.die_stack_survival[fail_mask] = False
                    for failed_die_ind in np.flatnonzero(fail_mask):
                        waf_interface.die_list[int(failed_die_ind)].survival = False
                    if cfg.verbose:
                        epoch_fail_vec_per_interface_dict[interface_name]['overlay'][
                            stack_ind, fail_mask
                        ] = 1
                        epoch_fail_vec_per_interface_dict[interface_name]['overall'][
                            stack_ind, fail_mask
                        ] = 1

                overlay_die_yield = 1.0
                if candidate_idx.size > 0:
                    overlay_die_yield = 1.0 - float(np.count_nonzero(fail_mask)) / float(candidate_idx.size)
                _print_status(
                    "Overlay-only fast path for stack {}/{} interface {}/{}: "
                    "{} yield {:.6f}, failed {}/{} dies.".format(
                        global_stack_offset + stack_ind + 1,
                        cfg.NUM_WAFER_STACKS,
                        interface_ind + 1,
                        len(waf_stack.interfaces.interface_dict),
                        interface_name,
                        overlay_die_yield,
                        int(np.count_nonzero(fail_mask)),
                        int(candidate_idx.size),
                    ),
                    verbose,
                )
                continue

            for die_ind, die in enumerate(waf_interface.die_list):
                die_pad_coords = waf_interface.base_pad_coords + die.die_center
                valid_die_pad_coords = die_pad_coords[valid_pad_mask_flat]
                die_count += 1
                if die_count % 10 == 0 or die_count == len(waf_interface.die_list):
                    _print_progress(
                        "Processing die {}/{}...Time taken for every 10 dies: {:.2f} seconds".format(
                            die_count,
                            len(waf_interface.die_list),
                            (time.time() - start_time) / die_count * 10,
                        ),
                        verbose,
                    )
                    # start_time = time.time()
                redundant_pad_fail_map = np.zeros((PAD_ARR_ROW, PAD_ARR_COL), dtype=bool)
                if redundant_group_id_grid is not None:
                    redundant_failed_counts = np.zeros(
                        len(redundant_tolerated_mechanical_failures),
                        dtype=np.int32,
                    )
                else:
                    redundant_failed_counts = None
                temp_overall_fail_map = np.zeros((PAD_ARR_ROW, PAD_ARR_COL), dtype=int)  # This map is used to store the fail pads for this die stack for all mechanisms, which will be used for visualization. It is reset for each die.
                
                '''
                Check the overlay errors
                '''
                if run_overlay and approximate_set == 1:
                    overlay_boundary_coords = _overlay_boundary_coords(die)
                    if len(overlay_boundary_coords) == 0:
                        worst_boundary_misalignment = -np.inf
                    else:
                        worst_boundary_misalignment = _corner_worst_overlay_misalignment_um(
                            overlay_boundary_coords,
                            system_translation_x_um,
                            system_translation_y_um,
                            system_rotation_rad,
                            system_magnification_ppm,
                        )
                    worst_boundary_misalignment += np.random.normal(
                        RANDOM_MISALIGNMENT_MEAN_um,
                        RANDOM_MISALIGNMENT_STD_um,
                    )

                    if worst_boundary_misalignment >= MAX_ALLOWED_MISALIGNMENT_um:
                        waf_interface.die_list[die_ind].survival = False
                        waf_stack.die_stack_survival[die_ind] = False
                        if cfg.verbose:
                            epoch_fail_vec_per_interface_dict[interface_name]['overlay'][stack_ind, die_ind] = 1
                            epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind, die_ind] = 1
                        if not cfg.verbose:
                            continue
            
                
                # # Check every net connecting redundant pads, if all the redundant pad replicas fail, then the die fails
                # if any(redundant_bumpid_set.issubset(fail_bump_id_set) for net, redundant_bumpid_set in redundant_net_to_bumpids.items()):
                #     wafer.survival_die -= 1
                #     die.survival = False
                #     break
                if not waf_stack.die_stack_survival[die_ind] and not cfg.verbose:
                    continue
                
                '''
                Check the void defects
                '''
                # # Check the void overlap with the pad
                # Assuming wafer.voids is an array of shape (N, 3), where N is the number of voids. [x, y, r]
                # Critical pad bitmap is a 2D array of shape (PAD_ARR_ROW, PAD_ARR_COL) with 1s for critical pads and 0s for non-critical pads
                if run_particle:
                    voids = np.array(failure_params['voids'])  # shape (N, 3), N is the number of voids
                else:
                    voids = np.empty((0, 3), dtype=float)
                if voids.size > 0:
                    # Coordinates and dimensions of the die pad array box
                    pad_array_box_x = die.pad_array_box[2][0]
                    pad_array_box_y = die.pad_array_box[2][1]

                    # Calculate closest x and y distances for all voids simultaneously
                    closest_x = np.maximum(pad_array_box_x, np.minimum(voids[:, 0], pad_array_box_x + PAD_ARR_W_um))
                    closest_y = np.maximum(pad_array_box_y, np.minimum(voids[:, 1], pad_array_box_y + PAD_ARR_L_um))

                    # Calculate distance from each void to the closest point on the pad array box
                    distances = (closest_x - voids[:, 0]) ** 2 + (closest_y - voids[:, 1]) ** 2

                    # Create a mask for voids overlapping with the pad array box
                    overlap_void_die_mask = distances < voids[:, 2] ** 2  # shape (N,)

                    # Use critical pad bitmap and grid search to find if any void overlaps with the die
                    if np.any(overlap_void_die_mask):
                        # Calculate the pad range we need to consider (critical, near the void)
                        # The i, j here are the indices of the pad array bitmap. The origin is the bottom left corner of the pad array box. 
                        # It is noticed that the origin of the bitmap is the top left corner of the pad array box. Switching is needed.
                        in_die_voids = voids[overlap_void_die_mask]
                        i_coord_min = min(in_die_voids[:, 0] - in_die_voids[:, 2] - PAD_TOP_R_um - pad_array_box_x)
                        i_coord_max = max(in_die_voids[:, 0] + in_die_voids[:, 2] + PAD_TOP_R_um - pad_array_box_x)
                        j_coord_min = min(in_die_voids[:, 1] - in_die_voids[:, 2] - PAD_TOP_R_um - pad_array_box_y)
                        j_coord_max = max(in_die_voids[:, 1] + in_die_voids[:, 2] + PAD_TOP_R_um - pad_array_box_y)
                        i_min = max(0,              int(np.floor(i_coord_min / PITCH_c_um)))    # (col_start)
                        i_max = min(PAD_ARR_COL-1,  int(np.ceil (i_coord_max / PITCH_c_um)))    # H = i_max - i_min + 1 (col_end)
                        j_min = max(0,              int(np.floor(j_coord_min / PITCH_r_um)))    # (row_start)
                        j_max = min(PAD_ARR_ROW-1,  int(np.ceil (j_coord_max / PITCH_r_um)))    # W = j_max - j_min + 1 (row_end)

                        check_pad_x_coords = pad_array_box_x + np.arange(i_min, i_max+1) * PITCH_c_um
                        check_pad_y_coords = pad_array_box_y + np.arange(j_min, j_max+1) * PITCH_r_um
                        check_pad_x_mesh, check_pad_y_mesh = np.meshgrid(check_pad_x_coords, check_pad_y_coords, indexing='xy')

                        # Calculate the distance from each void to the closest point on the critical pads
                        voids_xy = in_die_voids[:, :2]   # shape (N, 2), N is the number of voids
                        voids_x = voids_xy[:, 0][:, np.newaxis, np.newaxis]  # shape (N, 1, 1)
                        voids_y = voids_xy[:, 1][:, np.newaxis, np.newaxis]  # shape (N, 1, 1)
                        voids_r = in_die_voids[:, 2][:, np.newaxis, np.newaxis]  # shape (N, 1, 1)
                        pad_x = check_pad_x_mesh[np.newaxis, :, :]  # shape (1, H, W)
                        # print(pad_x)
                        pad_y = check_pad_y_mesh[np.newaxis, :, :]  # shape (1, H, W)
                        # print(pad_y)
                        dist_sq = (pad_x - voids_x) ** 2 + (pad_y - voids_y) ** 2  # shape (N, H, W)
                        overlap_void_pad_mask = dist_sq < (voids_r + PAD_TOP_R_um) ** 2 # shape (N, H, W)
                        overlap_void_pad_mask = np.any(overlap_void_pad_mask, axis=0)  # shape (H, W)
                        if np.any(overlap_void_pad_mask):
                           waf_interface.die_list[die_ind].voids_occur = True
                            
                        # Get the critical pad bitmap for the pads we need to consider
                        check_critical_pad_bitmap = die_critical_pad_bitmap[PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1]
                        # Get the redundant critical pad bitmap for the pads we need to consider
                        check_redundant_pad_bitmap = die_redundant_pad_bitmap[PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1]
                        # Record the fail pads due to voids
                        if cfg.verbose:
                            sub_fail_map_particle = epoch_fail_map_per_interface_dict[interface_name]['particle'][PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1]
                            sub_fail_map_particle[overlap_void_pad_mask] += 1
                            sub_fail_map_overall = temp_overall_fail_map[PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1]
                            sub_fail_map_overall[overlap_void_pad_mask] = 1
                        # Check if any void overlaps with the critical pads
                        overlap_critical = overlap_void_pad_mask & check_critical_pad_bitmap.astype(bool)
                        if np.any(overlap_critical):
                            # print("Die fails due to critical pad void overlap.")
                            waf_interface.die_list[die_ind].survival = False
                            waf_stack.die_stack_survival[die_ind] = False
                            if cfg.verbose:
                                epoch_fail_vec_per_interface_dict[interface_name]['particle'][stack_ind, die_ind] = 1
                                epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind, die_ind] = 1
                        else:   # Voids overlapping with the redundant pads.
                            # Check if any void overlaps with the redundant critical pads
                            overlap_redundant = overlap_void_pad_mask & check_redundant_pad_bitmap.astype(bool) # shape (H, W)
                            row_slice = slice(PAD_ARR_ROW-j_max-1, PAD_ARR_ROW-j_min)
                            col_slice = slice(i_min, i_max+1)
                            redundant_fail_submap = redundant_pad_fail_map[row_slice, col_slice]
                            new_overlap_redundant = overlap_redundant & (~redundant_fail_submap)
                            redundant_fail_submap[overlap_redundant] = True
                            if redundant_group_id_grid is not None:
                                _increment_redundant_group_counts(
                                    new_overlap_redundant,
                                    redundant_group_id_grid[row_slice, col_slice],
                                    redundant_failed_counts,
                                )
                                if _group_limit_exceeded(
                                    redundant_failed_counts,
                                    redundant_tolerated_mechanical_failures,
                                ):
                                    waf_interface.die_list[die_ind].survival = False
                                    waf_stack.die_stack_survival[die_ind] = False
                                    if cfg.verbose:
                                        epoch_fail_vec_per_interface_dict[interface_name]['particle'][stack_ind, die_ind] = 1
                                        epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind, die_ind] = 1
                            else:
                                for redundant_net, physical_mask in redundant_net_to_1d_physical_mask.items():
                                    tolerated_mechanical_failures = criticality_info[redundant_net]['tolerated_mechanical_failures']
                                    num_fail_pad_in_net = np.sum(redundant_pad_fail_map.flatten()[physical_mask])
                                    if num_fail_pad_in_net > tolerated_mechanical_failures:
                                        waf_interface.die_list[die_ind].survival = False
                                        waf_stack.die_stack_survival[die_ind] = False
                                        if cfg.verbose:
                                            epoch_fail_vec_per_interface_dict[interface_name]['particle'][stack_ind, die_ind] = 1
                                            epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind, die_ind] = 1
                                        if not cfg.verbose:
                                            break
                            # # Get the fail bump indices
                            # fail_bump_id = mapping_physical_to_bumpid[redundant_pad_fail_map == 1]
                            # # Switch to set for easier checking
                            # fail_bump_id_set = set(fail_bump_id.astype(int))
                            # # Check every net connecting redundant pads, if all the redundant pad replicas fail due to voids, then the die fails
                            # if any(redundant_bumpid_set.issubset(fail_bump_id_set) for net, redundant_bumpid_set in redundant_net_to_bumpids.items()):
                            #         print("Die fails due to redundant pad void overlap.")
                            #         waf_stack.die_stack_survival[die_ind] = False
                            #         bot_wafer.die_list[die_ind].survival, top_wafer.die_list[die_ind].survival = False, False
                            #         break

                # Proceed if die still survives
                if not waf_stack.die_stack_survival[die_ind] and not cfg.verbose:
                    continue
                
                '''
                Check the Cu gap, a true Monte Carlo simulator
                '''
                # Check the Cu expansion
                top_dish = bot_dish = None
                needs_esd_for_this_die = (
                    run_esd
                    and selected_esd_die_ind is not None
                    and die_ind == selected_esd_die_ind
                )
                if run_mechanical or needs_esd_for_this_die:
                    top_dish, bot_dish = Cu_gap_simulator(
                        cfg=cfg,
                        valid_pad_mask_flat=valid_pad_mask_flat,
                    )

                if run_mechanical:
                    Cu_gap_in_valid_pads = top_dish + bot_dish
                    Cu_gap_map = np.full((PAD_ARR_ROW, PAD_ARR_COL), np.nan)
                    Cu_gap_map[valid_pad_mask == 1] = Cu_gap_in_valid_pads

                    # Calculate the safe range for single pad Cu recess
                    if not os.path.exists(cfg.OUTPUT_DIR + cfg.DESIGN + '/temp/' + cfg.INTERFACE + "/" + cfg.INTERFACE + "_dishing_bound_array_die_{}.npy".format(die_ind)) or cfg.DEBUG:
                        if not os.path.exists(cfg.OUTPUT_DIR + cfg.DESIGN + '/temp/' + cfg.INTERFACE + '/'):
                            os.makedirs(cfg.OUTPUT_DIR + cfg.DESIGN + '/temp/' + cfg.INTERFACE + '/')
                        # start_time = time.time()
                        valid_pad_dishing_bound_array = debond_dishing_intervals_from_coords(cfg, valid_die_pad_coords) # (num_pads, 2) array: (dishing_low_nm, dishing_high_nm)
                        # print("Dishing bound calculation time: {:.2f} seconds".format(time.time() - start_time))
                        np.save(cfg.OUTPUT_DIR + cfg.DESIGN + '/temp/' + cfg.INTERFACE + "/" + cfg.INTERFACE + "_dishing_bound_array_die_{}.npy".format(die_ind), valid_pad_dishing_bound_array)
                    else:
                        valid_pad_dishing_bound_array = np.load(cfg.OUTPUT_DIR + cfg.DESIGN + '/temp/' + cfg.INTERFACE + "/" + cfg.INTERFACE + "_dishing_bound_array_die_{}.npy".format(die_ind))
                    zeta_0 = np.full((PAD_ARR_ROW, PAD_ARR_COL), np.nan)
                    zeta_1 = np.full((PAD_ARR_ROW, PAD_ARR_COL), np.nan)
                    zeta_0[valid_pad_mask == 1] = - valid_pad_dishing_bound_array[:, 1] * 2 # lower limits of the sum of top and bottom Cu heights
                    zeta_1[valid_pad_mask == 1] = - valid_pad_dishing_bound_array[:, 0] * 2 # upper limits of the sum of top and bottom Cu heights
                    zeta_0 = np.clip(zeta_0, a_max=0, a_min=None)
                    zeta_1 = np.clip(zeta_1, a_max=0, a_min=None)

                    if cfg.verbose:
                        epoch_fail_map_per_interface_dict[interface_name]['mechanical'] += ((Cu_gap_map > zeta_1) | (Cu_gap_map < zeta_0)).astype(int)
                        temp_overall_fail_map |= ((Cu_gap_map > zeta_1) | (Cu_gap_map < zeta_0)).astype(int)
                    # Check critical pad Cu gap
                    critical_pad_Cu_gap = Cu_gap_map * die_critical_pad_bitmap      # Shape: (PAD_ARR_ROW, PAD_ARR_COL)
                    if np.any(critical_pad_Cu_gap > zeta_1 * die_critical_pad_bitmap) or np.any(critical_pad_Cu_gap < zeta_0 * die_critical_pad_bitmap):
                        # print("Die fails due to critical pad Cu gap failure.")
                        waf_interface.die_list[die_ind].survival = False
                        waf_stack.die_stack_survival[die_ind] = False
                        if cfg.verbose:
                            epoch_fail_vec_per_interface_dict[interface_name]['mechanical'][stack_ind, die_ind] = 1
                            epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind, die_ind] = 1
                        if not cfg.verbose:
                            continue

                    # Check redundant pad Cu gap
                    redundant_pad_Cu_gap_fail_mask = (
                        (
                            (Cu_gap_map > zeta_1)
                            | (Cu_gap_map < zeta_0)
                        )
                        & die_redundant_pad_bitmap.astype(bool)
                    )
                    new_redundant_pad_Cu_gap_fail_mask = (
                        redundant_pad_Cu_gap_fail_mask
                        & (~redundant_pad_fail_map)
                    )
                    redundant_pad_fail_map[redundant_pad_Cu_gap_fail_mask] = True
                    if redundant_group_id_grid is not None:
                        _increment_redundant_group_counts(
                            new_redundant_pad_Cu_gap_fail_mask,
                            redundant_group_id_grid,
                            redundant_failed_counts,
                        )
                        if _group_limit_exceeded(
                            redundant_failed_counts,
                            redundant_tolerated_mechanical_failures,
                        ):
                            waf_interface.die_list[die_ind].survival = False
                            waf_stack.die_stack_survival[die_ind] = False
                            if cfg.verbose:
                                epoch_fail_vec_per_interface_dict[interface_name]['mechanical'][stack_ind, die_ind] = 1
                                epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind, die_ind] = 1
                    else:
                        for redundant_net, physical_mask in redundant_net_to_1d_physical_mask.items():
                            tolerated_mechanical_failures = criticality_info[redundant_net]['tolerated_mechanical_failures']
                            num_fail_pad_in_net = np.sum(redundant_pad_fail_map.flatten()[physical_mask])
                            if num_fail_pad_in_net > tolerated_mechanical_failures:
                                waf_interface.die_list[die_ind].survival = False
                                waf_stack.die_stack_survival[die_ind] = False
                                if cfg.verbose:
                                    epoch_fail_vec_per_interface_dict[interface_name]['mechanical'][stack_ind, die_ind] = 1
                                    epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind, die_ind] = 1
                                if not cfg.verbose:
                                    break
                # # Get the fail bump indices
                # fail_bump_id = mapping_physical_to_bumpid[redundant_pad_fail_map == 1]
                # # Switch to set for easier checking
                # fail_bump_id_set = set(fail_bump_id.astype(int))

                # # Check every net connecting redundant pads, if all the redundant pad replicas fail due to voids, then the die fails
                # if any(redundant_bumpid_set.issubset(fail_bump_id_set) for net, redundant_bumpid_set in redundant_net_to_bumpids.items()):
                #     print("Die fails due to redundant pad Cu gap failure.")
                #     waf_stack.die_stack_survival[die_ind] = False
                #     bot_wafer.die_list[die_ind].survival, top_wafer.die_list[die_ind].survival = False, False
                #     break

                '''
                Check the ESD failure
                '''
                if needs_esd_for_this_die:
                    first_contact_pad_idx, survive_bool = esd_failure_simulator(
                                                    cfg=cfg,
                                                    pad_coords_um=valid_die_pad_coords,
                                                    pad_size_um=PAD_TOP_R_um * 2,
                                                    top_die_w_um=die.DIE_W_um,
                                                    top_die_h_um=die.DIE_L_um,
                                                    wafer_radius_um=WAF_R_um,
                                                    top_dish_nm_ext=top_dish,
                                                    bot_dish_nm_ext=bot_dish,
                                                    dummy_pad_bitmap=valid_dummy_pad_bitmap,
                                                    )
                    if first_contact_pad_idx is not None and survive_bool == False:
                        full_linear_idx = int(valid_linear_idx[int(first_contact_pad_idx)])
                        r_idx, c_idx = full_linear_idx // PAD_ARR_COL, full_linear_idx % PAD_ARR_COL
                        if cfg.verbose:
                            epoch_fail_map_per_interface_dict[interface_name]['ESD'][r_idx, c_idx] += 1
                            temp_overall_fail_map[r_idx, c_idx] = 1
                        if die_esd_critical_pad_bitmap[r_idx, c_idx] == 1:
                            # print("Die fails due to ESD")
                            waf_stack.die_stack_survival[die_ind] = False
                            waf_interface.die_list[die_ind].survival = False
                            if cfg.verbose:
                                epoch_fail_vec_per_interface_dict[interface_name]['ESD'][stack_ind, die_ind] = 1
                                epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind, die_ind] = 1
                            continue
                if cfg.verbose:
                    epoch_fail_map_per_interface_dict[interface_name]['overall'] += temp_overall_fail_map.astype(int)               
            _clear_progress_line(verbose)
            # Record the time
            # print("The time for checking wafer {} is {} seconds.".format(waf_ind, time.time() - start_time))
            # # print("The number of survival dies in the wafer is {}.".format(wafer.survival_die))
        
        # Draw the whole wafer
        # Save waf_stack and cfg for visualization
        # dump(waf_stack, "waf_stack.joblib")

        # waf_stack.draw_w2w_stack_2d(cfg_dict, figname=cfg.OUTPUT_DIR + cfg.DESIGN + '/w2w_stack_{}_2d.png'.format(epoch*NUM_STACKS+stack_ind+1))



        # One stack is done, calculate the die stack yield
        die_stack_yield = waf_stack.die_stack_survival.sum() / num_dies_per_wafer
        die_stack_yield_list.append(die_stack_yield)        # die stack yield for 
    

    return die_stack_yield_list, epoch_fail_map_per_interface_dict, epoch_fail_vec_per_interface_dict
