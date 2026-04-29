#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#### Author: Zhichao Chen
#### Date: Oct 23, 2025

import numpy as np
import time
import math
import matplotlib.pyplot as plt
from overlay_yield_calculator import pad_overlay_yield_map_generator
from defect_yield_calculator import pad_defect_yield_map_generator
from Cu_expansion_yield_calculator import pad_Cu_expansion_yield_map_generator
from utils.util import risk_map_generator, _upsample_pad_yield_map
from esd_yield_calculator import pad_esd_yield_map_generator
from wafer_die_stack_initialization import DieStack


def _sample_grid_indices(length: int, sub_factor: int) -> np.ndarray:
    """Return endpoint-preserving sample indices for a 1D axis."""
    length = int(length)
    sub_factor = max(1, int(sub_factor))
    if length <= 0:
        return np.zeros((0,), dtype=np.int64)
    if sub_factor <= 1 or length <= 2:
        return np.arange(length, dtype=np.int64)

    sample_count = int(math.ceil(length / float(sub_factor)))
    sample_count = max(2, min(length, sample_count))
    return np.unique(
        np.round(np.linspace(0, length - 1, sample_count)).astype(np.int64)
    )


def _auto_esd_pad_map_sub_factor(valid_pad_count: int) -> int:
    """
    Choose a 2D subsampling factor for analytical ESD map generation.

    The ESD calculator scales roughly linearly with pad count, so we try to keep
    the sampled pad count near a manageable target while leaving smaller designs
    untouched.
    """
    valid_pad_count = int(valid_pad_count)
    target_sampled_pads = 100000
    if valid_pad_count <= target_sampled_pads:
        return 1
    return max(1, min(16, int(math.ceil(math.sqrt(valid_pad_count / target_sampled_pads)))))

def Pad_Yield_Map_Generator(
    input_args,
    cfg_dict,
    pad_bitmap_collection_dict,
):
    '''
    This function calculates the die-stack-level yield maps
    '''
    start_time = time.time()

    # Initialize the die stack
    die_stack = DieStack(
        cfg_dict                    =   cfg_dict,
        pad_bitmap_collection_dict  =   pad_bitmap_collection_dict,
        mode                        =   input_args['mode'],
        base_pad_coords_flag        =   True,
    )

    valid_pad_mask_dict = {}
    for interface_name, pad_bitmap_collection in pad_bitmap_collection_dict.items():
        valid_pad_mask_dict[interface_name] = (pad_bitmap_collection['CRITICAL_PAD_BITMAP'] == 1) | (pad_bitmap_collection['REDUNDANT_PAD_BITMAP'] == 1) | (pad_bitmap_collection['DUMMY_PAD_BITMAP'] == 1)

    for interface_name, cfg in cfg_dict.items():
        interface = die_stack.interfaces.interface_dict[interface_name]
        valid_pad_mask = valid_pad_mask_dict[interface_name]
        pad_bitmap_collection = pad_bitmap_collection_dict[interface_name]
        pad_map_shape = pad_bitmap_collection['CRITICAL_PAD_BITMAP'].shape
        valid_die_pad_coords = interface.pad_coords[valid_pad_mask.flatten() == 1]

        print(">>> Calculating pad-level yield maps for interface: {}".format(interface_name))

        # Calculate the overlay yield map
        overlay_start_time = time.perf_counter()
        overlay_pad_yield_map = pad_overlay_yield_map_generator(
            cfg                             =       cfg,
            PAD_TOP_R_um                    =       cfg.PAD_TOP_R_um,
            PAD_BOT_R_um                    =       cfg.PAD_BOT_R_um,
            PAD_ARR_ROW                     =       cfg.PAD_ARR_ROW,
            PAD_ARR_COL                     =       cfg.PAD_ARR_COL,
            PITCH_r_um                      =       cfg.PITCH_r_um,
            PITCH_c_um                      =       cfg.PITCH_c_um,
            num_samples                     =       cfg.num_samples,
            CONTACT_AREA_CONSTRAINT         =       cfg.CONTACT_AREA_CONSTRAINT,
            CRITICAL_DIST_CONSTRAINT        =       cfg.CRITICAL_DIST_CONSTRAINT,
            SYSTEM_MAGNIFICATION_MEAN_ppm   =       cfg.SYSTEM_MAGNIFICATION_MEAN_ppm,
            SYSTEM_MAGNIFICATION_STD_ppm    =       cfg.SYSTEM_MAGNIFICATION_STD_ppm,
            SYSTEM_ROTATION_MEAN_rad        =       cfg.SYSTEM_ROTATION_MEAN_rad,
            SYSTEM_ROTATION_STD_rad         =       cfg.SYSTEM_ROTATION_STD_rad,
            SYSTEM_TRANSLATION_X_MEAN_um    =       cfg.SYSTEM_TRANSLATION_X_MEAN_um,
            SYSTEM_TRANSLATION_X_STD_um     =       cfg.SYSTEM_TRANSLATION_X_STD_um,
            SYSTEM_TRANSLATION_Y_MEAN_um    =       cfg.SYSTEM_TRANSLATION_Y_MEAN_um,
            SYSTEM_TRANSLATION_Y_STD_um     =       cfg.SYSTEM_TRANSLATION_Y_STD_um,
            RANDOM_MISALIGNMENT_MEAN_um     =       cfg.RANDOM_MISALIGNMENT_MEAN_um,
            RANDOM_MISALIGNMENT_STD_um      =       cfg.RANDOM_MISALIGNMENT_STD_um,
            interface                       =       interface,
            pad_yield_flag                  =       cfg.pad_yield_flag,
            pad_yield_map_sub_factor        =       cfg.pad_yield_map_sub_factor,
        )
        overlay_pad_yield_map = _upsample_pad_yield_map(
            overlay_pad_yield_map,
            pad_map_shape,
            cfg.pad_yield_map_sub_factor,
        )
        interface.pad_yield_map['Y_ovl'] = overlay_pad_yield_map
        print(f"Overlay yield calculation took {time.perf_counter() - overlay_start_time:.2f} seconds")
        # raise Exception("Overlay yield calculation done. Stop execution here for debugging.")

        # Calculate the defect yield
        start_time = time.perf_counter()
        defect_pad_yield_map = pad_defect_yield_map_generator(
            cfg               =       cfg,
            D0                =       cfg.D0,
            t_0               =       cfg.t_0,
            z                 =       cfg.z,
            k_r               =       cfg.k_r,
            k_r0              =       cfg.k_r0,
            PAD_TOP_R_um      =       cfg.PAD_TOP_R_um,
            PAD_ARR_ROW       =       cfg.PAD_ARR_ROW,
            PAD_ARR_COL       =       cfg.PAD_ARR_COL,
            interface         =       interface,
            pad_yield_flag    =       cfg.pad_yield_flag,
            pad_yield_map_sub_factor = cfg.pad_yield_map_sub_factor,
        )
        defect_pad_yield_map = _upsample_pad_yield_map(
            defect_pad_yield_map,
            pad_map_shape,
            cfg.pad_yield_map_sub_factor,
        )
        interface.pad_yield_map['Y_df'] = defect_pad_yield_map
        print(f"Defect yield calculation took {time.perf_counter() - start_time:.2f} seconds")

        # Calculate the Cu expansion yield
        Cu_expansion_start_time = time.perf_counter()
        Cu_expansion_pad_yield_map = pad_Cu_expansion_yield_map_generator(
            cfg                 =       cfg,
            interface           =       interface,
            TOP_DISH_MEAN_nm    =       cfg.TOP_DISH_MEAN_nm,
            TOP_DISH_STD_nm     =       cfg.TOP_DISH_STD_nm,
            BOT_DISH_MEAN_nm    =       cfg.BOT_DISH_MEAN_nm,
            BOT_DISH_STD_nm     =       cfg.BOT_DISH_STD_nm,
            pad_bitmap_collection =   pad_bitmap_collection,
        )
        interface.pad_yield_map['Y_ce'] = Cu_expansion_pad_yield_map
        print(f"Cu expansion yield calculation took {time.perf_counter() - Cu_expansion_start_time:.2f} seconds")

        esd_start_time = time.perf_counter()
        esd_pad_map_sub_factor = int(
            getattr(cfg, "ESD_PAD_MAP_SUB_FACTOR", 0)
        )
        if esd_pad_map_sub_factor <= 0:
            esd_pad_map_sub_factor = _auto_esd_pad_map_sub_factor(
                int(np.count_nonzero(valid_pad_mask))
            )

        if esd_pad_map_sub_factor > 1:
            row_idx = _sample_grid_indices(cfg.PAD_ARR_ROW, esd_pad_map_sub_factor)
            col_idx = _sample_grid_indices(cfg.PAD_ARR_COL, esd_pad_map_sub_factor)
            sampled_valid_pad_mask = valid_pad_mask[np.ix_(row_idx, col_idx)]
            sampled_pad_coords = (
                interface.pad_coords.reshape(cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL, 2)[np.ix_(row_idx, col_idx)]
                .reshape(-1, 2)
            )
            sampled_valid_pad_coords = sampled_pad_coords[sampled_valid_pad_mask.reshape(-1) == 1]

            esd_sampled_valid_pad_yield_vec, _, _ = pad_esd_yield_map_generator(
                cfg                   = cfg,
                pad_coords_um         = sampled_valid_pad_coords,
                pad_size_um           = cfg.PAD_TOP_R_um * 2,
                pad_pitch_um          = cfg.PITCH_r_um,
                top_die_w_um          = cfg.DIE_W_um,
                top_die_h_um          = cfg.DIE_L_um,
                tilt_x_mean_deg       = cfg.TILT_X_MEAN_DEG,
                tilt_x_std_deg        = cfg.TILT_X_STD_DEG,
                tilt_y_mean_deg       = cfg.TILT_Y_MEAN_DEG,
                tilt_y_std_deg        = cfg.TILT_Y_STD_DEG,
                top_dish_mean_nm      = cfg.TOP_DISH_MEAN_nm,
                top_dish_std_nm       = cfg.TOP_DISH_STD_nm,
                bot_dish_mean_nm      = cfg.BOT_DISH_MEAN_nm,
                bot_dish_std_nm       = cfg.BOT_DISH_STD_nm,
            )

            sampled_esd_pad_yield_map = np.full(sampled_valid_pad_mask.shape, np.nan)
            sampled_esd_pad_yield_map[sampled_valid_pad_mask == 1] = esd_sampled_valid_pad_yield_vec
            esd_pad_yield_map = _upsample_pad_yield_map(
                sampled_esd_pad_yield_map,
                pad_map_shape,
                esd_pad_map_sub_factor,
            )
            esd_pad_yield_map[valid_pad_mask == 0] = np.nan

            if getattr(cfg, "verbose", False):
                sampled_valid_count = int(np.count_nonzero(sampled_valid_pad_mask))
                full_valid_count = int(np.count_nonzero(valid_pad_mask))
                print(
                    f"ESD analytical map subsampling: factor={esd_pad_map_sub_factor} | "
                    f"sampled_valid_pads={sampled_valid_count}/{full_valid_count}"
                )
        else:
            # Calculate the ESD yield on the full valid-pad set.
            esd_valid_pad_yield_vec, _, _ = pad_esd_yield_map_generator(
                cfg                   = cfg,
                pad_coords_um         = valid_die_pad_coords,
                pad_size_um           = cfg.PAD_TOP_R_um * 2,
                pad_pitch_um          = cfg.PITCH_r_um,
                top_die_w_um          = cfg.DIE_W_um,
                top_die_h_um          = cfg.DIE_L_um,
                tilt_x_mean_deg       = cfg.TILT_X_MEAN_DEG,
                tilt_x_std_deg        = cfg.TILT_X_STD_DEG,
                tilt_y_mean_deg       = cfg.TILT_Y_MEAN_DEG,
                tilt_y_std_deg        = cfg.TILT_Y_STD_DEG,
                top_dish_mean_nm      = cfg.TOP_DISH_MEAN_nm,
                top_dish_std_nm       = cfg.TOP_DISH_STD_nm,
                bot_dish_mean_nm      = cfg.BOT_DISH_MEAN_nm,
                bot_dish_std_nm       = cfg.BOT_DISH_STD_nm,
            )

            esd_pad_yield_map = np.full((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), np.nan)
            esd_pad_yield_map[valid_pad_mask == 1] = esd_valid_pad_yield_vec
        interface.pad_yield_map['Y_esd'] = esd_pad_yield_map
        interface.glb_pad_yield_min_max_dict['Y_esd'] = (np.nanmin(interface.pad_yield_map['Y_esd']), np.nanmax(interface.pad_yield_map['Y_esd']))
        if cfg.plot_flag:
            # Draw the pad yield map
            plt.figure(figsize=(8, 6))
            plt.imshow(
                interface.pad_yield_map['Y_esd'],
                cmap='viridis',
                vmin=interface.glb_pad_yield_min_max_dict['Y_esd'][0],
                vmax=interface.glb_pad_yield_min_max_dict['Y_esd'][1],
                interpolation='nearest',
                )
            plt.colorbar(label='Pad ESD Yield')
            plt.xlabel('Pad Column Index')
            plt.ylabel('Pad Row Index')
            plt.show()

        print(f"ESD yield calculation took {time.perf_counter() - esd_start_time:.2f} seconds")
        interface.pad_yield_map['Y_bond'] = interface.pad_yield_map['Y_ovl'] * interface.pad_yield_map['Y_df'] * interface.pad_yield_map['Y_ce'] * interface.pad_yield_map['Y_esd']
        interface.glb_pad_yield_min_max_dict['Y_bond'] = (np.nanmin(interface.pad_yield_map['Y_bond']), np.nanmax(interface.pad_yield_map['Y_bond']))
        # print(f"Overall pad bonding yield min: {interface.glb_pad_yield_min_max_dict['Y_bond'][0]:.6f}, max: {interface.glb_pad_yield_min_max_dict['Y_bond'][1]:.6f}")

        if cfg.plot_flag:
            # Draw the pad yield map
            plt.figure(figsize=(8, 6))
            plt.imshow(
                interface.pad_yield_map['Y_bond'],
                cmap='viridis',
                vmin=interface.glb_pad_yield_min_max_dict['Y_bond'][0],
                vmax=interface.glb_pad_yield_min_max_dict['Y_bond'][1],
                interpolation='nearest',
                )
            plt.colorbar(label='Pad Bonding Yield')
            plt.xlabel('Pad Column Index')
            plt.ylabel('Pad Row Index')
            plt.show()

        risk_map_generator(cfg=cfg, interface=interface, input_args=input_args)    # Generate and save the risk map in the specified output directory
