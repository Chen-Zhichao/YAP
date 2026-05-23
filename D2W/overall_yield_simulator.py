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
from Cu_gap_simulator import Cu_gap_simulator
from debond import debond_dishing_intervals_from_coords #, post_bond_warpage_calculator
from esd_yield_simulator import (
    build_esd_tile_cache,
    esd_failure_simulator,
    esd_failure_simulator_batch,
)
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


def _pad_misalignment_for_coords(
    *,
    pad_coords_um: np.ndarray | None,
    die_center_um: np.ndarray,
    system_translation_x_um: float,
    system_translation_y_um: float,
    system_rotation_rad: float,
    system_magnification_ppm: float,
    random_misalignment_mean_um: float,
    random_misalignment_std_um: float,
) -> np.ndarray:
    if pad_coords_um is None or len(pad_coords_um) == 0:
        return np.empty((0,), dtype=np.float64)

    coords = np.asarray(pad_coords_um, dtype=np.float64)
    x = coords[:, 0] + die_center_um[0]
    y = coords[:, 1] + die_center_um[1]
    dx = system_translation_x_um - system_rotation_rad * y + system_magnification_ppm * x
    dy = system_translation_y_um + system_rotation_rad * x + system_magnification_ppm * y
    misalignment = np.sqrt(dx * dx + dy * dy)
    if random_misalignment_std_um > 0.0 or random_misalignment_mean_um != 0.0:
        misalignment += np.random.normal(
            random_misalignment_mean_um,
            random_misalignment_std_um,
            len(coords),
        )
    return misalignment


def _overlay_corner_only_enabled(cfg) -> bool:
    return True


_FAILURE_MECHANISMS = ('overlay', 'particle', 'mechanical', 'ESD', 'warpage')


def _active_failure_mechanisms(input_args):
    raw = input_args.get('mechanism_filter', 'all')
    if raw is None:
        return set(_FAILURE_MECHANISMS)
    if isinstance(raw, (set, list, tuple)):
        requested = [str(item).strip() for item in raw]
    else:
        requested = [item.strip() for item in str(raw).split(',')]
    requested = [item for item in requested if item]
    if not requested or any(item.lower() == 'all' for item in requested):
        return set(_FAILURE_MECHANISMS)

    canonical = {item.lower(): item for item in _FAILURE_MECHANISMS}
    active = set()
    for item in requested:
        key = item.lower()
        if key not in canonical:
            raise ValueError(
                f"Unknown mechanism_filter '{item}'. "
                f"Valid mechanisms: all, {', '.join(_FAILURE_MECHANISMS)}."
            )
        active.add(canonical[key])
    return active


def _dishing_bound_cache_path_for_mask(cfg, input_args: dict, mask_name: str) -> str:
    base_path = get_dishing_bound_cache_path(cfg, input_args)
    stem, ext = os.path.splitext(base_path)
    return f"{stem}__{mask_name}{ext}"


def _load_or_compute_dishing_bounds(
    *,
    cfg,
    input_args: dict,
    coords_um: np.ndarray,
    mask_name: str,
) -> np.ndarray:
    dishing_cache_path = _dishing_bound_cache_path_for_mask(cfg, input_args, mask_name)
    recompute_dishing_bounds = bool(cfg.DEBUG) or not os.path.exists(dishing_cache_path)
    if not recompute_dishing_bounds:
        dishing_bound_array = np.load(dishing_cache_path)
        if dishing_bound_array.shape[0] != coords_um.shape[0]:
            recompute_dishing_bounds = True

    if recompute_dishing_bounds:
        dishing_bound_array = debond_dishing_intervals_from_coords(cfg, coords_um)
        atomic_save_npy(dishing_cache_path, dishing_bound_array)

    return dishing_bound_array


def _build_interface_static_cache(
    *,
    cfg_dict: dict,
    pad_bitmap_collection_dict: dict,
    base_pad_coords_dict: dict,
    input_args: dict,
) -> dict:
    active_mechanisms = _active_failure_mechanisms(input_args)
    run_overlay = 'overlay' in active_mechanisms
    run_mechanical = 'mechanical' in active_mechanisms
    run_esd = 'ESD' in active_mechanisms
    interface_static_cache = {}
    for interface_name, cfg in cfg_dict.items():
        pad_bitmap_collection = pad_bitmap_collection_dict[interface_name]
        critical_pad_bitmap = pad_bitmap_collection["CRITICAL_PAD_BITMAP"].astype(bool)
        redundant_pad_bitmap = pad_bitmap_collection["REDUNDANT_PAD_BITMAP"].astype(bool)
        dummy_pad_bitmap = pad_bitmap_collection["DUMMY_PAD_BITMAP"].astype(bool)
        power_ground_pad_bitmap = pad_bitmap_collection.get(
            "POWER_GROUND_PAD_BITMAP",
            np.zeros_like(critical_pad_bitmap, dtype=bool),
        ).astype(bool)
        valid_pad_mask = (
            critical_pad_bitmap
            | redundant_pad_bitmap
            | dummy_pad_bitmap
            | power_ground_pad_bitmap
        )
        flat_critical = critical_pad_bitmap.reshape(-1)
        flat_redundant = redundant_pad_bitmap.reshape(-1)
        flat_power_ground = power_ground_pad_bitmap.reshape(-1)
        overlay_critical_linear_idx = None
        overlay_redundant_linear_idx = None
        overlay_power_ground_linear_idx = None
        overlay_critical_coords = None
        overlay_redundant_coords = None
        overlay_power_ground_coords = None
        overlay_redundant_group_ids = None
        overlay_corner_only = _overlay_corner_only_enabled(cfg)
        if run_overlay and not overlay_corner_only:
            base_pad_coords = base_pad_coords_dict[interface_name]
            overlay_critical_linear_idx = np.flatnonzero(flat_critical)
            overlay_redundant_linear_idx = np.flatnonzero(flat_redundant)
            overlay_power_ground_linear_idx = np.flatnonzero(flat_power_ground)
            overlay_critical_coords = np.asarray(
                base_pad_coords[overlay_critical_linear_idx],
                dtype=np.float32,
            )
            overlay_redundant_coords = np.asarray(
                base_pad_coords[overlay_redundant_linear_idx],
                dtype=np.float32,
            )
            overlay_power_ground_coords = np.asarray(
                base_pad_coords[overlay_power_ground_linear_idx],
                dtype=np.float32,
            )
            group_id_per_pad = pad_bitmap_collection.get("redundant_group_id_per_pad")
            if group_id_per_pad is not None:
                overlay_redundant_group_ids = np.asarray(
                    group_id_per_pad,
                    dtype=np.int32,
                ).reshape(-1)[overlay_redundant_linear_idx]

        valid_pad_mask_flat = None
        valid_linear_idx = None
        valid_die_pad_coords = None
        valid_pad_dishing_bound_array = None
        esd_tile_cache = None
        if run_esd:
            valid_pad_mask_flat = valid_pad_mask.reshape(-1)
            esd_active_pad_mask = valid_pad_mask & (~dummy_pad_bitmap)
            esd_sim_method = str(getattr(cfg, "ESD_SIMULATION_METHOD", "auto")).strip().lower()
            esd_tile_threshold_pads = int(getattr(cfg, "ESD_SIM_TILE_THRESHOLD_PADS", 100_000))
            if (
                esd_sim_method == "tile"
                or (
                    esd_sim_method == "auto"
                    and int(np.count_nonzero(esd_active_pad_mask)) >= esd_tile_threshold_pads
                )
            ):
                esd_tile_cache = build_esd_tile_cache(cfg, esd_active_pad_mask)
            else:
                valid_linear_idx = np.flatnonzero(valid_pad_mask_flat)
                valid_die_pad_coords = np.asarray(
                    base_pad_coords_dict[interface_name][valid_pad_mask_flat],
                    dtype=np.float32,
                )
                if run_mechanical:
                    valid_pad_dishing_bound_array = _load_or_compute_dishing_bounds(
                        cfg=cfg,
                        input_args=input_args,
                        coords_um=valid_die_pad_coords,
                        mask_name="valid",
                    )

        mechanical_active_pad_mask = critical_pad_bitmap | redundant_pad_bitmap
        mechanical_active_pad_mask_flat = mechanical_active_pad_mask.reshape(-1)
        mechanical_active_linear_idx = np.flatnonzero(mechanical_active_pad_mask_flat)
        num_mechanical_active_pads = int(np.count_nonzero(mechanical_active_pad_mask_flat))
        mechanical_active_die_pad_coords = None
        mechanical_active_dishing_bound_array = None
        mechanical_active_critical_mask = None
        mechanical_active_redundant_mask = None
        mechanical_active_group_ids = None
        if run_mechanical:
            mechanical_active_die_pad_coords = np.asarray(
                base_pad_coords_dict[interface_name][mechanical_active_pad_mask_flat],
                dtype=np.float32,
            )
            mechanical_active_dishing_bound_array = _load_or_compute_dishing_bounds(
                cfg=cfg,
                input_args=input_args,
                coords_um=mechanical_active_die_pad_coords,
                mask_name="mechanical_active",
            )
            mechanical_active_critical_mask = flat_critical[mechanical_active_pad_mask_flat]
            mechanical_active_redundant_mask = flat_redundant[mechanical_active_pad_mask_flat]
            group_id_per_pad = pad_bitmap_collection.get("redundant_group_id_per_pad")
            if group_id_per_pad is not None:
                mechanical_active_group_ids = np.asarray(
                    group_id_per_pad,
                    dtype=np.int32,
                ).reshape(-1)[mechanical_active_pad_mask_flat]

        use_mechanical_die_level_sampling = False

        die_level_mechanical_yield = None

        interface_static_cache[interface_name] = {
            "valid_pad_mask": valid_pad_mask,
            "valid_pad_mask_flat": valid_pad_mask_flat,
            "valid_linear_idx": valid_linear_idx,
            "valid_die_pad_coords": valid_die_pad_coords,
            "valid_pad_dishing_bound_array": valid_pad_dishing_bound_array,
            "esd_tile_cache": esd_tile_cache,
            "use_mechanical_die_level_sampling": use_mechanical_die_level_sampling,
            "die_level_mechanical_yield": die_level_mechanical_yield,
            "overlay_critical_linear_idx": overlay_critical_linear_idx,
            "overlay_redundant_linear_idx": overlay_redundant_linear_idx,
            "overlay_power_ground_linear_idx": overlay_power_ground_linear_idx,
            "overlay_critical_coords": overlay_critical_coords,
            "overlay_redundant_coords": overlay_redundant_coords,
            "overlay_power_ground_coords": overlay_power_ground_coords,
            "overlay_redundant_group_ids": overlay_redundant_group_ids,
            "overlay_corner_only": overlay_corner_only,
            "mechanical_active_pad_mask": mechanical_active_pad_mask,
            "mechanical_active_pad_mask_flat": mechanical_active_pad_mask_flat,
            "mechanical_active_linear_idx": mechanical_active_linear_idx,
            "mechanical_active_dishing_bound_array": mechanical_active_dishing_bound_array,
            "mechanical_active_critical_mask": mechanical_active_critical_mask,
            "mechanical_active_redundant_mask": mechanical_active_redundant_mask,
            "mechanical_active_group_ids": mechanical_active_group_ids,
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
    active_mechanisms = _active_failure_mechanisms(input_args)
    run_overlay = 'overlay' in active_mechanisms
    run_particle = 'particle' in active_mechanisms
    run_mechanical = 'mechanical' in active_mechanisms
    run_esd = 'ESD' in active_mechanisms
    run_warpage = 'warpage' in active_mechanisms

    epoch_fail_map_per_interface_dict = {}    # This dict stores the fail bump maps for all die samples in this epoch for each mechanism
    epoch_fail_vec_per_interface_dict = {}    # This dict stores failure reason (each mechanism) for all die samples in this epoch
    failure_mechanism_list = list(_FAILURE_MECHANISMS) + ['overall']
    if input_args['verbose']:
        for interface_name, cfg in cfg_dict.items():
            epoch_fail_map_per_interface_dict[interface_name], epoch_fail_vec_per_interface_dict[interface_name] = {}, {}
            for failure_mechanism in failure_mechanism_list:
                if save_failure_maps:
                    epoch_fail_map_per_interface_dict[interface_name][failure_mechanism] = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL))
                epoch_fail_vec_per_interface_dict[interface_name][failure_mechanism] = np.zeros((NUM_STACKS))

    if run_overlay or run_mechanical or run_esd:
        interface_static_cache = _build_interface_static_cache(
            cfg_dict=cfg_dict,
            pad_bitmap_collection_dict=pad_bitmap_collection_dict,
            base_pad_coords_dict=base_pad_coords_dict,
            input_args=input_args,
        )
    else:
        interface_static_cache = {interface_name: {} for interface_name in cfg_dict}
    if run_warpage:
        stack_warpage_fail_vector = stack_warpage_fail_vector_for_epoch(
            input_args=input_args,
            cfg_dict=cfg_dict,
            stack_cfg_dict=stack_cfg_dict,
            num_samples=NUM_STACKS,
        )
    else:
        stack_warpage_fail_vector = np.zeros((NUM_STACKS,), dtype=bool)

    esd_batch_result_dict = {}
    if run_esd:
        for interface_name, cfg in cfg_dict.items():
            esd_tile_cache = interface_static_cache.get(interface_name, {}).get("esd_tile_cache")
            if esd_tile_cache is None:
                continue
            esd_batch_result_dict[interface_name] = esd_failure_simulator_batch(
                cfg=cfg,
                num_samples=NUM_STACKS,
                pad_size_um=float(cfg.PAD_TOP_R_um) * 2.0,
                top_die_w_um=float(cfg.DIE_W_um),
                top_die_h_um=float(cfg.DIE_L_um),
                tilt_x_mean_deg=float(cfg.TILT_X_MEAN_DEG),
                tilt_x_std_deg=float(cfg.TILT_X_STD_DEG),
                tilt_y_mean_deg=float(cfg.TILT_Y_MEAN_DEG),
                tilt_y_std_deg=float(cfg.TILT_Y_STD_DEG),
                esd_tile_cache=esd_tile_cache,
            )


    for stack_ind, die_stack in enumerate(die_stack_list):
        for interface_ind, (interface_name, die_interface) in enumerate(die_stack.interfaces.interface_dict.items()):
            # if stack_ind % 1 == 0:
            #     print("Simulating die stack {}/{} ".format(stack_ind+1, NUM_STACKS), end='\r')
            pad_bitmap_collection = pad_bitmap_collection_dict[interface_name]
            static_cache = interface_static_cache.get(interface_name, {})
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
            failure_params                      = die_stack.interfaces.failure_params_dict[interface_name]
            system_translation_x_um             = failure_params.get('system_translation_x_um', 0.0)
            system_translation_y_um             = failure_params.get('system_translation_y_um', 0.0)
            system_rotation_rad                 = failure_params.get('system_rotation_rad', 0.0)
            system_magnification_ppm            = failure_params.get('system_magnification_ppm', 0.0)
            MAX_ALLOWED_MISALIGNMENT_um         = failure_params.get('MAX_ALLOWED_MISALIGNMENT_um', np.inf)
            RANDOM_MISALIGNMENT_MEAN_um         = cfg.RANDOM_MISALIGNMENT_MEAN_um
            RANDOM_MISALIGNMENT_STD_um          = cfg.RANDOM_MISALIGNMENT_STD_um
            TILT_X_MEAN_DEG, TILT_X_STD_DEG     = cfg.TILT_X_MEAN_DEG, cfg.TILT_X_STD_DEG
            TILT_Y_MEAN_DEG, TILT_Y_STD_DEG     = cfg.TILT_Y_MEAN_DEG, cfg.TILT_Y_STD_DEG
            approximate_set                     = cfg.approximate_set


            # Get pad subsets used by Cu mechanical and ESD models.
            if run_esd:
                valid_pad_mask = static_cache["valid_pad_mask"]
                valid_pad_mask_flat = static_cache["valid_pad_mask_flat"]
                valid_linear_idx = static_cache["valid_linear_idx"]
                valid_die_pad_coords = static_cache["valid_die_pad_coords"]
                valid_pad_dishing_bound_array = static_cache["valid_pad_dishing_bound_array"]
                esd_tile_cache = static_cache["esd_tile_cache"]
            else:
                valid_pad_mask = None
                valid_pad_mask_flat = None
                valid_linear_idx = None
                valid_die_pad_coords = None
                valid_pad_dishing_bound_array = None
                esd_tile_cache = None
            if run_mechanical:
                mechanical_active_pad_mask = static_cache["mechanical_active_pad_mask"]
                mechanical_active_pad_mask_flat = static_cache["mechanical_active_pad_mask_flat"]
                mechanical_active_linear_idx = static_cache["mechanical_active_linear_idx"]
                mechanical_active_dishing_bound_array = static_cache["mechanical_active_dishing_bound_array"]
                mechanical_active_critical_mask = static_cache["mechanical_active_critical_mask"]
                mechanical_active_redundant_mask = static_cache["mechanical_active_redundant_mask"]
                mechanical_active_group_ids = static_cache["mechanical_active_group_ids"]
            else:
                mechanical_active_pad_mask = None
                mechanical_active_pad_mask_flat = None
                mechanical_active_linear_idx = None
                mechanical_active_dishing_bound_array = None
                mechanical_active_critical_mask = None
                mechanical_active_redundant_mask = None
                mechanical_active_group_ids = None

            # Read the critical pad bitmap
            die_critical_pad_bitmap = pad_bitmap_collection["CRITICAL_PAD_BITMAP"]
            # Read the redundant critical pad bitmap
            die_redundant_pad_bitmap = pad_bitmap_collection["REDUNDANT_PAD_BITMAP"]
            # Read the power/ground pad bitmap. PG pads are required overlay pads,
            # but they do not use redundant-group tolerance.
            die_power_ground_pad_bitmap = pad_bitmap_collection.get(
                "POWER_GROUND_PAD_BITMAP",
                np.zeros_like(die_critical_pad_bitmap, dtype=bool),
            ).astype(bool)
            if run_overlay:
                overlay_corner_only = static_cache.get("overlay_corner_only", True)
                if overlay_corner_only:
                    overlay_boundary_coords = getattr(
                        die_interface,
                        "ovl_active_pad_boundary_coords",
                        None,
                    )
                    if overlay_boundary_coords is None:
                        overlay_boundary_coords = die_interface.pad_array_box
                    overlay_critical_coords = None
                    overlay_redundant_coords = None
                    overlay_power_ground_coords = None
                    overlay_redundant_linear_idx = None
                    overlay_redundant_group_ids = None
                    overlay_power_ground_linear_idx = None
                else:
                    overlay_boundary_coords = None
                    overlay_critical_coords = static_cache["overlay_critical_coords"]
                    overlay_redundant_coords = static_cache["overlay_redundant_coords"]
                    overlay_power_ground_coords = static_cache["overlay_power_ground_coords"]
                    overlay_redundant_linear_idx = static_cache["overlay_redundant_linear_idx"]
                    overlay_redundant_group_ids = static_cache["overlay_redundant_group_ids"]
                    overlay_power_ground_linear_idx = static_cache["overlay_power_ground_linear_idx"]
                overlay_fail_map_flat = (
                    np.zeros(PAD_ARR_ROW * PAD_ARR_COL, dtype=bool)
                    if cfg.verbose and save_failure_maps and not overlay_corner_only
                    else None
                )
            else:
                overlay_corner_only = True
                overlay_boundary_coords = None
                overlay_critical_coords = None
                overlay_redundant_coords = None
                overlay_power_ground_coords = None
                overlay_redundant_linear_idx = None
                overlay_redundant_group_ids = None
                overlay_power_ground_linear_idx = None
                overlay_fail_map_flat = None
            power_ground_overlay_fail_ratio = float(
                getattr(cfg, "POWER_GROUND_OVERLAY_FAIL_RATIO", 0.10)
            )
            power_ground_overlay_fail_count_th = int(
                np.ceil(
                    power_ground_overlay_fail_ratio
                    * np.count_nonzero(die_power_ground_pad_bitmap)
                )
            )
            # Read the ESD critical pad bitmap
            die_esd_critical_pad_bitmap = pad_bitmap_collection["ESD_CRITICAL_PAD_BITMAP"]
            # Read the redundant net to bump ids mapping
            redundant_net_to_bumpids = pad_bitmap_collection["redundant_net_to_bumpids"]
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
            if run_overlay:
                if overlay_corner_only:
                    overlay_boundary_misalignment = _pad_misalignment_for_coords(
                        pad_coords_um=overlay_boundary_coords,
                        die_center_um=np.zeros(2, dtype=np.float64),
                        system_translation_x_um=system_translation_x_um,
                        system_translation_y_um=system_translation_y_um,
                        system_rotation_rad=system_rotation_rad,
                        system_magnification_ppm=system_magnification_ppm,
                        random_misalignment_mean_um=0.0,
                        random_misalignment_std_um=0.0,
                    )
                    worst_boundary_misalignment = (
                        float(np.max(overlay_boundary_misalignment))
                        if overlay_boundary_misalignment.size > 0
                        else 0.0
                    )
                    if (
                        RANDOM_MISALIGNMENT_STD_um > 0.0
                        or RANDOM_MISALIGNMENT_MEAN_um != 0.0
                    ):
                        worst_boundary_misalignment += float(
                            np.random.normal(
                                RANDOM_MISALIGNMENT_MEAN_um,
                                RANDOM_MISALIGNMENT_STD_um,
                            )
                        )
                    critical_overlay_fail_mask = (
                        np.array(
                            [worst_boundary_misalignment >= MAX_ALLOWED_MISALIGNMENT_um],
                            dtype=bool,
                        )
                    )
                    critical_pad_misalignment = np.array(
                        [worst_boundary_misalignment],
                        dtype=np.float64,
                    )
                else:
                    critical_pad_misalignment = _pad_misalignment_for_coords(
                        pad_coords_um=overlay_critical_coords,
                        die_center_um=die_interface.die_center,
                        system_translation_x_um=system_translation_x_um,
                        system_translation_y_um=system_translation_y_um,
                        system_rotation_rad=system_rotation_rad,
                        system_magnification_ppm=system_magnification_ppm,
                        random_misalignment_mean_um=RANDOM_MISALIGNMENT_MEAN_um,
                        random_misalignment_std_um=RANDOM_MISALIGNMENT_STD_um,
                    )
                    critical_overlay_fail_mask = (
                        critical_pad_misalignment >= MAX_ALLOWED_MISALIGNMENT_um
                    )
            else:
                critical_pad_misalignment = None
                critical_overlay_fail_mask = None
            # print(f"Pad misalignment simulation time for stack {stack_ind}, interface {interface_name}: {time.perf_counter() - pad_misalignment_time_start:.2f} seconds.")
            if run_overlay and approximate_set == 1 and overlay_corner_only:
                if np.any(critical_overlay_fail_mask):
                    die_interface.survival = False
                    die_stack.survival = False
                    if cfg.verbose:
                        epoch_fail_vec_per_interface_dict[interface_name]['overlay'][stack_ind] = 1
                        epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind] = 1
                    if not cfg.verbose:
                        continue

            elif run_overlay and approximate_set == 1:
                # Check if any critical pad misalignment is greater than the maximum allowed misalignment
                if np.any(critical_overlay_fail_mask):
                    die_interface.survival = False
                    die_stack.survival = False
                    if overlay_fail_map_flat is not None:
                        overlay_fail_map_flat[static_cache["overlay_critical_linear_idx"][critical_overlay_fail_mask]] = True
                    if cfg.verbose:
                        epoch_fail_vec_per_interface_dict[interface_name]['overlay'][stack_ind] = 1
                        epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind] = 1
                    if not cfg.verbose:
                        continue
                # Check if too many redundant pad misalignment is greater than the maximum allowed misalignment
                redundant_pad_misalignment = _pad_misalignment_for_coords(
                    pad_coords_um=overlay_redundant_coords,
                    die_center_um=die_interface.die_center,
                    system_translation_x_um=system_translation_x_um,
                    system_translation_y_um=system_translation_y_um,
                    system_rotation_rad=system_rotation_rad,
                    system_magnification_ppm=system_magnification_ppm,
                    random_misalignment_mean_um=RANDOM_MISALIGNMENT_MEAN_um,
                    random_misalignment_std_um=RANDOM_MISALIGNMENT_STD_um,
                )
                overlay_redundant_fail_mask = (
                    redundant_pad_misalignment > MAX_ALLOWED_MISALIGNMENT_um
                )
                redundant_pad_fail_map_flat = redundant_pad_fail_map.reshape(-1)
                new_overlay_redundant_fail_mask = (
                    overlay_redundant_fail_mask
                    & (~redundant_pad_fail_map_flat[overlay_redundant_linear_idx])
                )
                redundant_pad_fail_map_flat[
                    overlay_redundant_linear_idx[overlay_redundant_fail_mask]
                ] = True
                if overlay_fail_map_flat is not None:
                    overlay_fail_map_flat[
                        overlay_redundant_linear_idx[overlay_redundant_fail_mask]
                    ] = True
                if redundant_group_id_grid is not None:
                    _increment_redundant_group_counts(
                        new_overlay_redundant_fail_mask,
                        overlay_redundant_group_ids,
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

                # Check if too many power/ground pads exceed the overlay limit.
                # Power/ground pads are required pads, but tolerate a small
                # failed fraction before the die is considered failed.
                if power_ground_overlay_fail_count_th > 0:
                    power_ground_pad_misalignment = _pad_misalignment_for_coords(
                        pad_coords_um=overlay_power_ground_coords,
                        die_center_um=die_interface.die_center,
                        system_translation_x_um=system_translation_x_um,
                        system_translation_y_um=system_translation_y_um,
                        system_rotation_rad=system_rotation_rad,
                        system_magnification_ppm=system_magnification_ppm,
                        random_misalignment_mean_um=RANDOM_MISALIGNMENT_MEAN_um,
                        random_misalignment_std_um=RANDOM_MISALIGNMENT_STD_um,
                    )
                    power_ground_overlay_fail_mask = (
                        power_ground_pad_misalignment > MAX_ALLOWED_MISALIGNMENT_um
                    )
                    power_ground_overlay_fail_count = int(
                        np.count_nonzero(power_ground_overlay_fail_mask)
                    )
                    if overlay_fail_map_flat is not None:
                        overlay_fail_map_flat[
                            overlay_power_ground_linear_idx[power_ground_overlay_fail_mask]
                        ] = True
                    if power_ground_overlay_fail_count >= power_ground_overlay_fail_count_th:
                        die_interface.survival = False
                        die_stack.survival = False
                        if cfg.verbose:
                            epoch_fail_vec_per_interface_dict[interface_name]['overlay'][stack_ind] = 1
                            epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind] = 1

                if overlay_fail_map_flat is not None:
                    overlay_fail_map = overlay_fail_map_flat.reshape(PAD_ARR_ROW, PAD_ARR_COL)
                    epoch_fail_map_per_interface_dict[interface_name]['overlay'] += overlay_fail_map.astype(int)
                    temp_overall_fail_map |= overlay_fail_map

                # # Get the fail bump indices
                # fail_bump_id = mapping_physical_to_bumpid[redundant_pad_fail_map == 1]
                # # Switch to set for easier checking
                # fail_bump_id_set = set(fail_bump_id.astype(int))

            # Delete the die.pad_misalignment to save memory
            if hasattr(die_interface, "pad_misalignment"):
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
            if run_particle:
                voids = np.array(die_stack.interfaces.failure_params_dict[interface_name]['voids']) # shape (N, 3), N is the number of voids
            else:
                voids = np.empty((0, 3), dtype=float)
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
            top_dish = bot_dish = None
            use_esd_tile_simulator = run_esd and esd_tile_cache is not None
            if run_mechanical or (run_esd and not use_esd_tile_simulator):
                if run_esd and not use_esd_tile_simulator:
                    cu_gap_pad_mask_flat = valid_pad_mask_flat
                else:
                    cu_gap_pad_mask_flat = mechanical_active_pad_mask_flat
                top_dish, bot_dish = Cu_gap_simulator(
                    cfg=cfg,
                    valid_pad_mask_flat=cu_gap_pad_mask_flat,
                )

            if run_mechanical and static_cache["use_mechanical_die_level_sampling"]:
                die_level_mechanical_yield = static_cache["die_level_mechanical_yield"]
                if float(np.random.random()) > float(die_level_mechanical_yield):
                    die_interface.survival = False
                    die_stack.survival = False
                    if cfg.verbose:
                        epoch_fail_vec_per_interface_dict[interface_name]['mechanical'][stack_ind] = 1
                        epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind] = 1
                    if not cfg.verbose:
                        continue
            elif run_mechanical and (not run_esd or use_esd_tile_simulator):
                Cu_gap_in_active_pads = top_dish + bot_dish
                active_zeta_0 = -mechanical_active_dishing_bound_array[:, 1] * 2
                active_zeta_1 = -mechanical_active_dishing_bound_array[:, 0] * 2
                active_zeta_0 = np.clip(active_zeta_0, a_max=0, a_min=None)
                active_zeta_1 = np.clip(active_zeta_1, a_max=0, a_min=None)
                active_Cu_gap_fail_mask = (
                    (Cu_gap_in_active_pads < active_zeta_0)
                    | (Cu_gap_in_active_pads > active_zeta_1)
                )

                if cfg.verbose and save_failure_maps and np.any(active_Cu_gap_fail_mask):
                    fail_linear_idx = mechanical_active_linear_idx[active_Cu_gap_fail_mask]
                    fail_rows = fail_linear_idx // PAD_ARR_COL
                    fail_cols = fail_linear_idx - fail_rows * PAD_ARR_COL
                    epoch_fail_map_per_interface_dict[interface_name]['mechanical'][
                        fail_rows,
                        fail_cols,
                    ] += 1
                    temp_overall_fail_map[fail_rows, fail_cols] = 1

                # Check critical pad Cu gap.
                if np.any(active_Cu_gap_fail_mask & mechanical_active_critical_mask):
                    die_interface.survival = False
                    die_stack.survival = False
                    if cfg.verbose:
                        epoch_fail_vec_per_interface_dict[interface_name]['mechanical'][stack_ind] = 1
                        epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind] = 1
                    if not cfg.verbose:
                        continue

                # Check redundant pad Cu gap.
                active_redundant_fail_mask = (
                    active_Cu_gap_fail_mask & mechanical_active_redundant_mask
                )
                if np.any(active_redundant_fail_mask):
                    fail_linear_idx = mechanical_active_linear_idx[active_redundant_fail_mask]
                    fail_rows = fail_linear_idx // PAD_ARR_COL
                    fail_cols = fail_linear_idx - fail_rows * PAD_ARR_COL
                    new_redundant_fail_mask = ~redundant_pad_fail_map[fail_rows, fail_cols]
                    redundant_pad_fail_map[fail_rows, fail_cols] = True
                    if redundant_group_id_grid is not None:
                        group_ids = mechanical_active_group_ids[active_redundant_fail_mask]
                        group_ids = group_ids[new_redundant_fail_mask]
                        group_ids = group_ids[group_ids >= 0]
                        if group_ids.size > 0:
                            redundant_failed_counts += np.bincount(
                                group_ids.astype(np.int64, copy=False),
                                minlength=redundant_failed_counts.shape[0],
                            ).astype(redundant_failed_counts.dtype, copy=False)
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
                                break
            elif run_mechanical:
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
            # cfg carries per-interface ESD stack geometry from the 3Dblox graph.
            if run_esd:
                esd_batch_result = esd_batch_result_dict.get(interface_name)
                if use_esd_tile_simulator and esd_batch_result is not None:
                    esd_pad_idx = int(esd_batch_result[0][stack_ind])
                    survive_bool = bool(esd_batch_result[1][stack_ind])
                elif use_esd_tile_simulator:
                    esd_pad_coords_um = np.empty((0, 2), dtype=np.float32)
                    esd_top_dish = np.empty((0,), dtype=np.float32)
                    esd_bot_dish = np.empty((0,), dtype=np.float32)
                    esd_dummy_bitmap = np.empty((0,), dtype=bool)
                    esd_pad_idx, survive_bool = esd_failure_simulator(
                                                            cfg=cfg,
                                                            pad_coords_um=esd_pad_coords_um,
                                                            pad_size_um=PAD_TOP_R_um * 2,
                                                            top_die_w_um=die_interface.DIE_W_um,
                                                            top_die_h_um=die_interface.DIE_L_um,
                                                            top_dish_nm_ext=esd_top_dish,
                                                            bot_dish_nm_ext=esd_bot_dish,
                                                            tilt_x_mean_deg=TILT_X_MEAN_DEG,
                                                            tilt_x_std_deg=TILT_X_STD_DEG,
                                                            tilt_y_mean_deg=TILT_Y_MEAN_DEG,
                                                            tilt_y_std_deg=TILT_Y_STD_DEG,
                                                            dummy_pad_bitmap=esd_dummy_bitmap,
                                                            esd_tile_cache=esd_tile_cache,
                                                            return_full_linear_idx=use_esd_tile_simulator,
                                                            )
                else:
                    esd_pad_coords_um = valid_die_pad_coords
                    esd_top_dish = top_dish
                    esd_bot_dish = bot_dish
                    esd_dummy_bitmap = pad_bitmap_collection['DUMMY_PAD_BITMAP'].flatten()[valid_pad_mask_flat]
                    esd_pad_idx, survive_bool = esd_failure_simulator(
                                                            cfg=cfg,
                                                            pad_coords_um=esd_pad_coords_um,
                                                            pad_size_um=PAD_TOP_R_um * 2,
                                                            top_die_w_um=die_interface.DIE_W_um,
                                                            top_die_h_um=die_interface.DIE_L_um,
                                                            top_dish_nm_ext=esd_top_dish,
                                                            bot_dish_nm_ext=esd_bot_dish,
                                                            tilt_x_mean_deg=TILT_X_MEAN_DEG,
                                                            tilt_x_std_deg=TILT_X_STD_DEG,
                                                            tilt_y_mean_deg=TILT_Y_MEAN_DEG,
                                                            tilt_y_std_deg=TILT_Y_STD_DEG,
                                                            dummy_pad_bitmap=esd_dummy_bitmap,
                                                            esd_tile_cache=esd_tile_cache,
                                                            return_full_linear_idx=use_esd_tile_simulator,
                                                            )
            else:
                esd_pad_idx, survive_bool = None, True
            if esd_pad_idx is not None and survive_bool == False:    # One pad will form the first contact and fail
                # esd_pad_idx is indexed within the compressed valid-pad list, so map
                # it back to the full pad-array linear index before decoding row/col.
                if use_esd_tile_simulator:
                    full_linear_idx = int(esd_pad_idx)
                else:
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
            if input_args['verbose']:
                for interface_name in cfg_dict:
                    epoch_fail_vec_per_interface_dict[interface_name]['warpage'][stack_ind] = 1
                    epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_ind] = 1

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
