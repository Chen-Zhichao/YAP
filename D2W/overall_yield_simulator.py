#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Overall yield simulator for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from overlay_yield_simulator import die_pad_misalignment
from Cu_gap_simulator import Cu_gap_correlated_simulator
from debond import debond_dishing_intervals_from_coords #, post_bond_warpage_calculator
from esd_yield_simulator import esd_failure_simulator
from utils.util import atomic_save_npy, get_dishing_bound_cache_path

try:
    from warpage_yield_simulator import stack_warpage_fail_vector_for_epoch
except ModuleNotFoundError:
    from D2W.warpage_yield_simulator import stack_warpage_fail_vector_for_epoch


def _increment_redundant_group_counts(
    new_fail_mask: np.ndarray,
    group_id_source: np.ndarray | None,
    redundant_failed_counts: np.ndarray | None,
) -> bool:
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


def _group_limit_exceeded(
    redundant_failed_counts: np.ndarray | None,
    tolerated_failures: np.ndarray | None,
) -> bool:
    if redundant_failed_counts is None or tolerated_failures is None:
        return False
    if redundant_failed_counts.shape[0] == 0:
        return False
    return bool(np.any(redundant_failed_counts > tolerated_failures))


def _build_interface_static_cache(
    *,
    cfg_dict: dict,
    pad_bitmap_collection_dict: dict,
    base_pad_coords_dict: dict,
    input_args: dict,
) -> dict:
    interface_static_cache = {}
    for interface_name, cfg in cfg_dict.items():
        pad_bitmap_collection = pad_bitmap_collection_dict[interface_name]
        critical_pad_bitmap = pad_bitmap_collection["CRITICAL_PAD_BITMAP"].astype(bool)
        redundant_pad_bitmap = pad_bitmap_collection["REDUNDANT_PAD_BITMAP"].astype(bool)
        dummy_pad_bitmap = pad_bitmap_collection["DUMMY_PAD_BITMAP"].astype(bool)
        valid_pad_mask = critical_pad_bitmap | redundant_pad_bitmap | dummy_pad_bitmap
        valid_pad_mask_flat = valid_pad_mask.reshape(-1)
        valid_linear_idx = np.flatnonzero(valid_pad_mask_flat)
        valid_die_pad_coords = np.asarray(
            base_pad_coords_dict[interface_name][valid_pad_mask_flat],
            dtype=np.float32,
        )

        dishing_cache_path = get_dishing_bound_cache_path(cfg, input_args)
        recompute_dishing_bounds = bool(cfg.DEBUG) or not os.path.exists(dishing_cache_path)
        if not recompute_dishing_bounds:
            valid_pad_dishing_bound_array = np.load(dishing_cache_path)
            if valid_pad_dishing_bound_array.shape[0] != valid_die_pad_coords.shape[0]:
                recompute_dishing_bounds = True

        if recompute_dishing_bounds:
            valid_pad_dishing_bound_array = debond_dishing_intervals_from_coords(
                cfg,
                valid_die_pad_coords,
            )
            atomic_save_npy(dishing_cache_path, valid_pad_dishing_bound_array)

        mechanical_active_pad_mask = critical_pad_bitmap | redundant_pad_bitmap
        mechanical_active_pad_mask_flat = mechanical_active_pad_mask.reshape(-1)
        num_mechanical_active_pads = int(np.count_nonzero(mechanical_active_pad_mask_flat))

        use_mechanical_die_level_sampling = False

        die_level_mechanical_yield = None

        interface_static_cache[interface_name] = {
            "valid_pad_mask": valid_pad_mask,
            "valid_pad_mask_flat": valid_pad_mask_flat,
            "valid_linear_idx": valid_linear_idx,
            "valid_die_pad_coords": valid_die_pad_coords,
            "valid_pad_dishing_bound_array": valid_pad_dishing_bound_array,
            "use_mechanical_die_level_sampling": use_mechanical_die_level_sampling,
            "die_level_mechanical_yield": die_level_mechanical_yield,
            "num_mechanical_active_pads": num_mechanical_active_pads,
        }
    return interface_static_cache


def overall_yield_simulator(
    input_args: dict,
    cfg_dict: dict,
    die_stack_list: list,
    pad_bitmap_collection_dict: dict,
    base_pad_coords_dict: dict,
    stack_cfg_dict: dict = None,
):
    die_stack_yield_list = []
    NUM_STACKS = len(die_stack_list)
    pass_die_stack_count = 0
    pass_interface_count_dict = {
        interface_name: 0 for interface_name in cfg_dict
    }
    global_stack_offset = int(input_args.get('global_stack_offset', 0))
    save_failure_maps = bool(input_args.get('save_failure_maps', False))

    epoch_fail_map_per_interface_dict = {}    # This dict stores the fail bump maps for all die samples in this epoch for each mechanism
    epoch_fail_vec_per_interface_dict = {}    # This dict stores failure reason (each mechanism) for all die samples in this epoch
    failure_mechanism_list = ['overlay', 'particle', 'mechanical', 'ESD', 'overall']
    if input_args['verbose']:
        for interface_name, cfg in cfg_dict.items():
            epoch_fail_map_per_interface_dict[interface_name], epoch_fail_vec_per_interface_dict[interface_name] = {}, {}
            for failure_mechanism in failure_mechanism_list:
                if save_failure_maps:
                    epoch_fail_map_per_interface_dict[interface_name][failure_mechanism] = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL))
                epoch_fail_vec_per_interface_dict[interface_name][failure_mechanism] = np.zeros((NUM_STACKS))

    interface_static_cache = _build_interface_static_cache(
        cfg_dict=cfg_dict,
        pad_bitmap_collection_dict=pad_bitmap_collection_dict,
        base_pad_coords_dict=base_pad_coords_dict,
        input_args=input_args,
    )
    stack_warpage_fail_vector = stack_warpage_fail_vector_for_epoch(
        input_args=input_args,
        cfg_dict=cfg_dict,
        stack_cfg_dict=stack_cfg_dict,
        num_samples=NUM_STACKS,
    )


    for stack_ind, die_stack in enumerate(die_stack_list):
        for interface_ind, (interface_name, die_interface) in enumerate(die_stack.interfaces.interface_dict.items()):
            # if stack_ind % 1 == 0:
            #     print("Simulating die stack {}/{} ".format(stack_ind+1, NUM_STACKS), end='\r')
            pad_bitmap_collection = pad_bitmap_collection_dict[interface_name]
            static_cache = interface_static_cache[interface_name]
            cfg = cfg_dict[interface_name]
            temp_overall_fail_map = (
                np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), dtype=bool)
                if save_failure_maps and cfg.verbose
                else None
            )  # This map is used to store the fail pads for this die stack for all mechanisms.

            # Read the configuration parameters for this interface
            PAD_ARR_ROW, PAD_ARR_COL            = cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL
            PAD_ARR_W_um, PAD_ARR_L_um          = cfg.PAD_ARR_W_um, cfg.PAD_ARR_L_um
            PITCH_c_um, PITCH_r_um              = cfg.PITCH_c_um, cfg.PITCH_r_um
            PAD_TOP_R_um                        = cfg.PAD_TOP_R_um
            base_pad_coords                     = base_pad_coords_dict[interface_name]
            system_translation_x_um             = die_stack.interfaces.failure_params_dict[interface_name]['system_translation_x_um']
            system_translation_y_um             = die_stack.interfaces.failure_params_dict[interface_name]['system_translation_y_um']
            system_rotation_rad                 = die_stack.interfaces.failure_params_dict[interface_name]['system_rotation_rad']
            system_magnification_ppm            = die_stack.interfaces.failure_params_dict[interface_name]['system_magnification_ppm']
            MAX_ALLOWED_MISALIGNMENT_um         = die_stack.interfaces.failure_params_dict[interface_name]['MAX_ALLOWED_MISALIGNMENT_um']
            RANDOM_MISALIGNMENT_MEAN_um         = cfg.RANDOM_MISALIGNMENT_MEAN_um
            RANDOM_MISALIGNMENT_STD_um          = cfg.RANDOM_MISALIGNMENT_STD_um
            TILT_X_MEAN_DEG, TILT_X_STD_DEG     = cfg.TILT_X_MEAN_DEG, cfg.TILT_X_STD_DEG
            TILT_Y_MEAN_DEG, TILT_Y_STD_DEG     = cfg.TILT_Y_MEAN_DEG, cfg.TILT_Y_STD_DEG
            approximate_set                     = cfg.approximate_set


            # Get the valid pad mask
            valid_pad_mask = static_cache["valid_pad_mask"]
            valid_pad_mask_flat = static_cache["valid_pad_mask_flat"]
            valid_linear_idx = static_cache["valid_linear_idx"]
            valid_die_pad_coords = static_cache["valid_die_pad_coords"]
            valid_pad_dishing_bound_array = static_cache["valid_pad_dishing_bound_array"]

            # Read the critical pad bitmap
            die_critical_pad_bitmap = pad_bitmap_collection["CRITICAL_PAD_BITMAP"]
            # Read the redundant critical pad bitmap
            die_redundant_pad_bitmap = pad_bitmap_collection["REDUNDANT_PAD_BITMAP"]
            # Read the ESD critical pad bitmap
            die_esd_critical_pad_bitmap = pad_bitmap_collection["ESD_CRITICAL_PAD_BITMAP"]
            # Read the redundant net to bump ids mapping
            redundant_net_to_bumpids = pad_bitmap_collection["redundant_net_to_bumpids"]
            # Get the valid pad mask
            valid_pad_mask = (pad_bitmap_collection['CRITICAL_PAD_BITMAP'] == 1) | (pad_bitmap_collection['REDUNDANT_PAD_BITMAP'] == 1) | (pad_bitmap_collection['DUMMY_PAD_BITMAP'] == 1)
            # Read the mapping from physical to bump id
            mapping_physical_to_bumpid = pad_bitmap_collection["mapping_physical_to_bumpid"]
            # Read the criticality info
            criticality_info = pad_bitmap_collection["criticality_info"]
            # Read the redundant net to 1D physical mask mapping
            redundant_net_to_1d_physical_mask = pad_bitmap_collection["redundant_net_to_1d_physical_mask"]
            redundant_pad_fail_map = np.zeros((PAD_ARR_ROW, PAD_ARR_COL), dtype=bool)
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
                redundant_failed_counts = np.zeros(
                    len(redundant_tolerated_mechanical_failures),
                    dtype=np.int32,
                )
            else:
                redundant_group_id_grid = None
                redundant_failed_counts = None

            """
            Check the overlay errors
            """
            # Check the pad misalignment
            # pad_misalignment_time_start = time.perf_counter()
            die_interface.pad_misalignment = die_pad_misalignment(die_interface=die_interface,
                                                        base_pad_coords=base_pad_coords,
                                                        system_translation_x_um=system_translation_x_um,
                                                        system_translation_y_um=system_translation_y_um,
                                                        system_rotation_rad=system_rotation_rad,
                                                        system_magnification_ppm=system_magnification_ppm,
                                                        RANDOM_MISALIGNMENT_MEAN_um=RANDOM_MISALIGNMENT_MEAN_um,
                                                        RANDOM_MISALIGNMENT_STD_um=RANDOM_MISALIGNMENT_STD_um,
                                                        approximate_set=approximate_set,
                                                        )
            # print(f"Pad misalignment simulation time for stack {stack_ind}, interface {interface_name}: {time.perf_counter() - pad_misalignment_time_start:.2f} seconds.")
            if approximate_set == 1:
                # die fail criteria: any pad_misalignment >= MAX_ALLOWED_MISALIGNMENT_um
                die_interface.pad_misalignment = die_interface.pad_misalignment.reshape(cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL)
                if cfg.verbose and save_failure_maps:
                    epoch_fail_map_per_interface_dict[interface_name]['overlay'] += (die_interface.pad_misalignment >= MAX_ALLOWED_MISALIGNMENT_um).astype(int)
                    temp_overall_fail_map |= (die_interface.pad_misalignment >= MAX_ALLOWED_MISALIGNMENT_um)

                critical_pad_misalignment = die_interface.pad_misalignment * die_critical_pad_bitmap
                # Check if any critical pad misalignment is greater than the maximum allowed misalignment
                if np.any(critical_pad_misalignment >= MAX_ALLOWED_MISALIGNMENT_um):
                    die_interface.survival = False
                    die_stack.survival = False
                    if cfg.verbose:
                        epoch_fail_vec_per_interface_dict[interface_name]['overlay'][stack_ind] = 1
                        epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind] = 1
                    if not cfg.verbose:
                        continue
                # Check if too many redundant pad misalignment is greater than the maximum allowed misalignment
                overlay_redundant_fail_mask = (
                    (die_interface.pad_misalignment > MAX_ALLOWED_MISALIGNMENT_um)
                    & die_redundant_pad_bitmap.astype(bool)
                )
                new_overlay_redundant_fail_mask = (
                    overlay_redundant_fail_mask
                    & (~redundant_pad_fail_map)
                )
                redundant_pad_fail_map[overlay_redundant_fail_mask] = True
                if redundant_group_id_grid is not None:
                    _increment_redundant_group_counts(
                        new_overlay_redundant_fail_mask,
                        redundant_group_id_grid,
                        redundant_failed_counts,
                    )
                    if _group_limit_exceeded(
                        redundant_failed_counts,
                        redundant_tolerated_mechanical_failures,
                    ):
                        die_interface.survival = False
                        die_stack.survival = False
                        if cfg.verbose:
                            epoch_fail_vec_per_interface_dict[interface_name]['overlay'][stack_ind] = 1
                            epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind] = 1
                else:
                    for redundant_net, physical_mask in redundant_net_to_1d_physical_mask.items():
                        tolerated_mechanical_failures = criticality_info[redundant_net]['tolerated_mechanical_failures']
                        num_fail_pad_in_net = np.sum(redundant_pad_fail_map.flatten()[physical_mask])
                        if num_fail_pad_in_net > tolerated_mechanical_failures:
                            die_interface.survival = False
                            die_stack.survival = False
                            if cfg.verbose:
                                epoch_fail_vec_per_interface_dict[interface_name]['overlay'][stack_ind] = 1
                                epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind] = 1
                            break

                # # Get the fail bump indices
                # fail_bump_id = mapping_physical_to_bumpid[redundant_pad_fail_map == 1]
                # # Switch to set for easier checking
                # fail_bump_id_set = set(fail_bump_id.astype(int))

            # Delete the die.pad_misalignment to save memory
            del die_interface.pad_misalignment

            if not die_stack.survival and not cfg.verbose:
                continue

            """
            Check the void defects
            """
            # void_check_time_start = time.perf_counter()
            ## Check the void overlap with the pad
            # Assuming wafer.voids is an array of shape (N, 3), where N is the number of voids. [x, y, r]
            # Critical pad bitmap is a 2D array of shape (PAD_ARR_ROW, PAD_ARR_COL) with 1s for critical pads and 0s for non-critical pads
            voids = np.array(die_stack.interfaces.failure_params_dict[interface_name]['voids']) # shape (N, 3), N is the number of voids
            if voids.size > 0:
                # Coordinates and dimensions of the die pad array box
                pad_array_box_x = die_interface.pad_array_box[2][0]
                pad_array_box_y = die_interface.pad_array_box[2][1]

                # Calculate closest x and y distances for all voids simultaneously
                closest_x = np.maximum(pad_array_box_x, np.minimum(voids[:, 0], pad_array_box_x + PAD_ARR_W_um))
                closest_y = np.maximum(pad_array_box_y, np.minimum(voids[:, 1], pad_array_box_y + PAD_ARR_L_um))

                # Calculate distance from each void to the closest point on the pad array box
                distances = np.sqrt((closest_x - voids[:, 0]) ** 2 + (closest_y - voids[:, 1]) ** 2)

                # Create a mask for voids overlapping with the pad array box
                overlapping_mask = distances < voids[:, 2]

                # Use critical pad bitmap and grid search to find if any void overlaps with the critical pads
                if np.any(overlapping_mask):
                    # Calculate the pad range we need to consider (critical, near the void)
                    for void_index, void in enumerate(voids[overlapping_mask]):
                        # Calculate the pad range we need to consider (critical, near the void)
                        # The i, j here are the indices of the pad array bitmap. The origin is the bottom left corner of the pad array box.
                        # It is noticed that the origin of the bitmap is the top left corner of the pad array box. Switching is needed.
                        i_coords_min = void[0] - void[2] - PAD_TOP_R_um - pad_array_box_x
                        i_coords_max = void[0] + void[2] + PAD_TOP_R_um - pad_array_box_x
                        j_coords_min = void[1] - void[2] - PAD_TOP_R_um - pad_array_box_y
                        j_coords_max = void[1] + void[2] + PAD_TOP_R_um - pad_array_box_y
                        i_min = max(0,              int(np.floor(i_coords_min / PITCH_c_um)))     # (col_start)
                        i_max = min(PAD_ARR_COL-1,  int(np.ceil (i_coords_max / PITCH_c_um))) # H = i_max - i_min + 1 (col_end)
                        j_min = max(0,              int(np.floor(j_coords_min / PITCH_r_um)))     # (row_start)
                        j_max = min(PAD_ARR_ROW-1,  int(np.ceil (j_coords_max / PITCH_r_um))) # W = j_max - j_min + 1 (row_end)

                        check_pad_x_coords = pad_array_box_x + np.arange(i_min, i_max+1) * PITCH_c_um
                        check_pad_y_coords = pad_array_box_y + np.arange(j_min, j_max+1) * PITCH_r_um
                        check_pad_x_mesh, check_pad_y_mesh = np.meshgrid(check_pad_x_coords, check_pad_y_coords, indexing='xy')

                        # Calculate the distance from the void to the closest point on the critical pads
                        dist_sq = (check_pad_x_mesh - void[0]) ** 2 + (check_pad_y_mesh - void[1]) ** 2 # Shape (H, W)
                        overlap_void_pad_mask = (dist_sq < (void[2] + PAD_TOP_R_um) ** 2)      # shape (H, W)
                        if np.any(overlap_void_pad_mask):
                            die_interface.voids_occur = True  # Will draw the die to green if it still survives

                        # check_pad_y_coords grows bottom -> top, but the bitmap slices use
                        # top-left origin. Flip the local overlap mask vertically before
                        # combining it with any bitmap or fail-map slice.
                        overlap_void_pad_mask_bitmap = np.flipud(overlap_void_pad_mask)

                        # Get the critical pad bitmap for the pads we need to consider
                        check_critical_pad_bitmap = die_critical_pad_bitmap[PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1]
                        # Get the redundant critical pad bitmap for the pads we need to consider
                        check_redundant_pad_bitmap = die_redundant_pad_bitmap[PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1]
                        # Record the fail pads due to voids
                        if cfg.verbose and save_failure_maps:
                            sub_fail_map_particle = epoch_fail_map_per_interface_dict[interface_name]['particle'][PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1]
                            sub_fail_map_particle[overlap_void_pad_mask_bitmap] += 1
                            sub_fail_map_overall = temp_overall_fail_map[PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1]
                            sub_fail_map_overall[overlap_void_pad_mask_bitmap] = 1
                        # Check if any void overlaps with the critical pads
                        overlap_critical = overlap_void_pad_mask_bitmap & check_critical_pad_bitmap.astype(bool)
                        if np.any(overlap_critical):
                            die_interface.survival = False
                            die_stack.survival = False
                            if cfg.verbose:
                                epoch_fail_vec_per_interface_dict[interface_name]['particle'][stack_ind] = 1
                                epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind] = 1
                            if not cfg.verbose:
                                break
                        else:
                            # Check if any void overlaps with the redundant critical pads
                            overlap_redundant = overlap_void_pad_mask_bitmap & check_redundant_pad_bitmap.astype(bool)
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
                                    die_interface.survival = False
                                    die_stack.survival = False
                                    if cfg.verbose:
                                        epoch_fail_vec_per_interface_dict[interface_name]['particle'][stack_ind] = 1
                                        epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind] = 1
                                    if not cfg.verbose:
                                        break
                            else:
                                for redundant_net, physical_mask in redundant_net_to_1d_physical_mask.items():
                                    tolerated_mechanical_failures = criticality_info[redundant_net]['tolerated_mechanical_failures']
                                    num_fail_pad_in_net = np.sum(redundant_pad_fail_map.flatten()[physical_mask])
                                    if num_fail_pad_in_net > tolerated_mechanical_failures:
                                        die_interface.survival = False
                                        die_stack.survival = False
                                        if cfg.verbose:
                                            epoch_fail_vec_per_interface_dict[interface_name]['particle'][stack_ind] = 1
                                            epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind] = 1
                                        if not cfg.verbose:
                                            break
                            # # Get the fail bump indices
                            # fail_bump_id = mapping_physical_to_bumpid[redundant_pad_fail_map == 1]
                            # # Switch to set for easier checking
                            # fail_bump_id_set = set(fail_bump_id.astype(int))

                            # # Check every net connecting redundant pads, if all the redundant pad replicas fail due to voids, then the die fails
                            # if any(redundant_bumpid_set.issubset(fail_bump_id_set) for net, redundant_bumpid_set in redundant_net_to_bumpids.items()):
                            #     die.survival = False
                            #     break
                        if not die_stack.survival and not cfg.verbose:
                            break
            # print(f"Void defect simulation time for stack {stack_ind}, interface {interface_name}: {time.perf_counter() - void_check_time_start:.2f} seconds.")
            # Proceed if die still survives
            if not die_stack.survival and not cfg.verbose:
                continue



            '''
            Check the Cu gap, a true Monte Carlo simulator
            '''
            # Check the Cu expansion
            top_dish, bot_dish = Cu_gap_correlated_simulator(
                cfg=cfg,
                valid_pad_mask_flat=valid_pad_mask_flat,
            )

            if static_cache["use_mechanical_die_level_sampling"]:
                die_level_mechanical_yield = static_cache["die_level_mechanical_yield"]
                if float(np.random.random()) > float(die_level_mechanical_yield):
                    die_interface.survival = False
                    die_stack.survival = False
                    if cfg.verbose:
                        epoch_fail_vec_per_interface_dict[interface_name]['mechanical'][stack_ind] = 1
                        epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind] = 1
                    if not cfg.verbose:
                        continue
            else:
                Cu_gap_in_valid_pads = top_dish + bot_dish
                Cu_gap_map = np.full((PAD_ARR_ROW, PAD_ARR_COL), np.nan)
                Cu_gap_map[valid_pad_mask == 1] = Cu_gap_in_valid_pads

                # Calculate the safe range for single pad Cu recess
                zeta_0 = np.full((PAD_ARR_ROW, PAD_ARR_COL), np.nan)    # lower limits to prevent Cu connection open
                zeta_1 = np.full((PAD_ARR_ROW, PAD_ARR_COL), np.nan)    # upper limits to prevent dielectric delamination

                zeta_0[valid_pad_mask == 1] = - valid_pad_dishing_bound_array[:, 1] * 2 # lower limits of the sum of top and bottom Cu heights
                zeta_1[valid_pad_mask == 1] = - valid_pad_dishing_bound_array[:, 0] * 2 # upper limits of the sum of top and bottom Cu heights
                zeta_0 = np.clip(zeta_0, a_max=0, a_min=None)
                zeta_1 = np.clip(zeta_1, a_max=0, a_min=None)

                if cfg.verbose and save_failure_maps:
                    epoch_fail_map_per_interface_dict[interface_name]['mechanical'] += (
                        (Cu_gap_map > zeta_1) | (Cu_gap_map < zeta_0)
                    ).astype(int)
                    temp_overall_fail_map |= ((Cu_gap_map > zeta_1) | (Cu_gap_map < zeta_0))

                # Check critical pad Cu gap
                critical_pad_Cu_gap = Cu_gap_map * die_critical_pad_bitmap  # shape: (PAD_ARR_ROW, PAD_ARR_COL)
                if np.any(critical_pad_Cu_gap > zeta_1 * die_critical_pad_bitmap) or np.any(critical_pad_Cu_gap < zeta_0 * die_critical_pad_bitmap):
                    die_interface.survival = False
                    die_stack.survival = False
                    if cfg.verbose:
                        epoch_fail_vec_per_interface_dict[interface_name]['mechanical'][stack_ind] = 1
                        epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind] = 1
                    if not cfg.verbose:
                        continue

                # Check redundant pad Cu gap
                redundant_pad_Cu_gap_fail_mask = (
                    ((Cu_gap_map < zeta_0) | (Cu_gap_map > zeta_1)) & die_redundant_pad_bitmap.astype(bool)
                )
                new_redundant_pad_Cu_gap_fail_mask = (
                    redundant_pad_Cu_gap_fail_mask & (~redundant_pad_fail_map)
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
                        die_interface.survival = False
                        die_stack.survival = False
                        if cfg.verbose:
                            epoch_fail_vec_per_interface_dict[interface_name]['mechanical'][stack_ind] = 1
                            epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind] = 1
                else:
                    for redundant_net, physical_mask in redundant_net_to_1d_physical_mask.items():
                        tolerated_mechanical_failures = criticality_info[redundant_net]['tolerated_mechanical_failures']
                        num_fail_pad_in_net = np.sum(redundant_pad_fail_map.flatten()[physical_mask])
                        if num_fail_pad_in_net > tolerated_mechanical_failures:
                            die_interface.survival = False
                            die_stack.survival = False
                            if cfg.verbose:
                                epoch_fail_vec_per_interface_dict[interface_name]['mechanical'][stack_ind] = 1
                                epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind] = 1
                            if not cfg.verbose:
                                break

            # # Get the fail bump indices
            # fail_bump_id = mapping_physical_to_bumpid[redundant_pad_fail_map == 1]
            # # Switch to set for easier checking
            # fail_bump_id_set = set(fail_bump_id.astype(int))
            # # Check every net connecting redundant pads, if all the redundant pad replicas fail due to voids, then the die fails
            # if any(redundant_bumpid_set.issubset(fail_bump_id_set) for net, redundant_bumpid_set in redundant_net_to_bumpids.items()):
            #     # print(f"Die {die_ind} fails due to redundant pad Cu gap.")
            #     die_interface.survival = False
            #     die_stack.survival = False
            #     break

            '''
            Check the ESD failure
            '''
            # TODO: ESD failure simulation to be implemented
            esd_pad_idx, survive_bool = esd_failure_simulator(
                                                    cfg=cfg,
                                                    pad_coords_um=valid_die_pad_coords,
                                                    pad_size_um=PAD_TOP_R_um * 2,
                                                    top_die_w_um=die_interface.DIE_W_um,
                                                    top_die_h_um=die_interface.DIE_L_um,
                                                    top_dish_nm_ext=top_dish,
                                                    bot_dish_nm_ext=bot_dish,
                                                    tilt_x_mean_deg=TILT_X_MEAN_DEG,
                                                    tilt_x_std_deg=TILT_X_STD_DEG,
                                                    tilt_y_mean_deg=TILT_Y_MEAN_DEG,
                                                    tilt_y_std_deg=TILT_Y_STD_DEG,
                                                    dummy_pad_bitmap=pad_bitmap_collection['DUMMY_PAD_BITMAP'].flatten()[valid_pad_mask_flat],
                                                    )
            if esd_pad_idx is not None and survive_bool == False:    # One pad will form the first contact and fail
                # esd_pad_idx is indexed within the compressed valid-pad list, so map
                # it back to the full pad-array linear index before decoding row/col.
                full_linear_idx = int(valid_linear_idx[int(esd_pad_idx)])
                r_idx, c_idx = full_linear_idx // PAD_ARR_COL, full_linear_idx % PAD_ARR_COL
                if cfg.verbose and save_failure_maps:
                    epoch_fail_map_per_interface_dict[interface_name]['ESD'][r_idx, c_idx] += 1
                    temp_overall_fail_map[r_idx, c_idx] = 1
                if die_esd_critical_pad_bitmap[r_idx, c_idx] == 1:  # If the failing pad is critical w.r.t. ESD
                    # print(f"Die stack {stack_ind} fails due to ESD on critical pad.")
                    die_interface.survival = False
                    die_stack.survival = False
                    if cfg.verbose:
                        epoch_fail_vec_per_interface_dict[interface_name]['ESD'][stack_ind] = 1
                        epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind] = 1
                    continue
                is_new_esd_redundant_fail = False
                if die_redundant_pad_bitmap[r_idx, c_idx] == 1:
                    is_new_esd_redundant_fail = not redundant_pad_fail_map[r_idx, c_idx]
                    redundant_pad_fail_map[r_idx, c_idx] = True
                    if (
                        is_new_esd_redundant_fail
                        and redundant_group_id_grid is not None
                    ):
                        group_id = int(redundant_group_id_grid[r_idx, c_idx])
                        if group_id >= 0:
                            redundant_failed_counts[group_id] += 1
                    else:
                        is_new_esd_redundant_fail = False
                if (
                    redundant_group_id_grid is not None
                    and is_new_esd_redundant_fail
                    and _group_limit_exceeded(
                        redundant_failed_counts,
                        redundant_tolerated_esd_failures,
                    )
                ):
                    die_interface.survival = False
                    die_stack.survival = False
                    if cfg.verbose:
                        epoch_fail_vec_per_interface_dict[interface_name]['ESD'][stack_ind] = 1
                        epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind] = 1
                elif redundant_group_id_grid is None:
                    for redundant_net, physical_mask in redundant_net_to_1d_physical_mask.items():
                        tolerated_esd_failures = criticality_info[redundant_net]['tolerated_esd_failures']
                        num_fail_pad_in_net = np.sum(redundant_pad_fail_map.flatten()[physical_mask])
                        if num_fail_pad_in_net > tolerated_esd_failures:
                            die_interface.survival = False
                            die_stack.survival = False
                            if cfg.verbose:
                                epoch_fail_vec_per_interface_dict[interface_name]['ESD'][stack_ind] = 1
                                epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind] = 1
                            break

            if cfg.verbose and save_failure_maps:
                epoch_fail_map_per_interface_dict[interface_name]['overall'] += temp_overall_fail_map

        '''
        Check the warpage failure in the stack-level
        '''
        if stack_warpage_fail_vector[stack_ind]:
            die_stack.survival = False

        for interface_name, die_interface in die_stack.interfaces.interface_dict.items():
            if die_interface.survival:
                pass_interface_count_dict[interface_name] += 1
        if die_stack.survival:
            pass_die_stack_count += 1

    die_yield = pass_die_stack_count / NUM_STACKS
    interface_yield_dict = {
        interface_name: pass_count / NUM_STACKS
        for interface_name, pass_count in pass_interface_count_dict.items()
    }
    # print("The yield of dies is {:.2f}%.".format(die_yield * 100))
    die_stack_yield_list.append(die_yield)

    return (
        die_stack_yield_list,
        interface_yield_dict,
        epoch_fail_map_per_interface_dict,
        epoch_fail_vec_per_interface_dict,
    )
