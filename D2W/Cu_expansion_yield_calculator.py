#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#### Author: Zhichao Chen
#### Date: Oct 3, 2025

'''
Cu expansion yield calculator for D2W hybrid bonding:
This module contains functions to calculate die-level and pad-level Cu expansion-induced yield
based on Cu dish distribution and pad layout.
'''

import os
import time
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
import matplotlib.pyplot as plt
from debond import debond_dishing_intervals_from_coords




def pad_Cu_expansion_yield_map_generator(*,
                                  cfg,
                                  interface,
                                  TOP_DISH_MEAN_nm: float,
                                  TOP_DISH_STD_nm: float,
                                  BOT_DISH_MEAN_nm: float,
                                  BOT_DISH_STD_nm: float,
                                  pad_bitmap_collection: dict,
                                  ):
    glb_cu_expansion_pad_yield_min = 1.0  # Initialize to a high value
    glb_cu_expansion_pad_yield_max = 0.0  # Initialize to a low value
    valid_pad_mask = (pad_bitmap_collection['CRITICAL_PAD_BITMAP'] == 1) | (pad_bitmap_collection['REDUNDANT_PAD_BITMAP'] == 1) | (pad_bitmap_collection['DUMMY_PAD_BITMAP'] == 1)
    valid_die_pad_coords = interface.pad_coords[valid_pad_mask.flatten() == 1]

    # if not os.path.exists(cfg.OUTPUT_DIR + cfg.INTERFACE + '/' + cfg.INTERFACE + "_dishing_bound_array.npy") or cfg.DEBUG:
    #     start_time = time.time()
    #     valid_pad_dishing_bound_array = debond_dishing_bounds_calculator(cfg, valid_die_pad_coords) # (num_pads, 2) array: (dishing_low_nm, dishing_high_nm)
    #     print("Dishing bound calculation time: {:.2f} seconds".format(time.time() - start_time))
    #     np.save(cfg.OUTPUT_DIR + cfg.INTERFACE + '/' + cfg.INTERFACE + "_dishing_bound_array.npy", valid_pad_dishing_bound_array)
    # else:
    #     print("Loading dishing bound array from file {}".format(cfg.OUTPUT_DIR + cfg.INTERFACE + '/' + cfg.INTERFACE + "_dishing_bound_array.npy"))
    #     valid_pad_dishing_bound_array = np.load(cfg.OUTPUT_DIR + cfg.INTERFACE + '/' + cfg.INTERFACE + "_dishing_bound_array.npy")

    # start_time = time.perf_counter()
    valid_pad_dishing_bound_array = debond_dishing_intervals_from_coords(cfg, valid_die_pad_coords) # (num_pads, 2) array: (dishing_low_nm, dishing_high_nm)
    # print(
    #     "Dishing bound calculation time for {} pads: {:.2f} seconds".format(
    #         valid_die_pad_coords.shape[0],
    #         time.perf_counter() - start_time,
    #     )
    # )

    upper_cu_height_limits_valid_pads = - valid_pad_dishing_bound_array[:, 0] * 2 # - upper Cu height limits
    lower_cu_height_limits_valid_pads = - valid_pad_dishing_bound_array[:, 1] * 2 # - lower Cu height limits
    upper_cu_height_limits_valid_pads = np.clip(upper_cu_height_limits_valid_pads, a_max=0, a_min=None)  # Clip to ensure upper Cu height limits are <= 0
    # print("Max upper Cu height (nm): {:.2f}, Min upper Cu height (nm): {:.2f}".format(np.max(upper_cu_height_limits_valid_pads), np.min(upper_cu_height_limits_valid_pads)))
    # print("Max lower Cu height (nm): {:.2f}, Min lower Cu height (nm): {:.2f}".format(np.max(lower_cu_height_limits_valid_pads), np.min(lower_cu_height_limits_valid_pads)))
    pos_valid_pads = norm.cdf(upper_cu_height_limits_valid_pads, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2)) - \
                     norm.cdf(lower_cu_height_limits_valid_pads, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2))
    pad_yield_map = np.full((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), np.nan)
    pad_yield_map[valid_pad_mask == 1] = pos_valid_pads

    glb_cu_expansion_pad_yield_min = min(glb_cu_expansion_pad_yield_min, np.nanmin(pad_yield_map))
    glb_cu_expansion_pad_yield_max = max(glb_cu_expansion_pad_yield_max, np.nanmax(pad_yield_map))
    interface.glb_pad_yield_min_max_dict['Y_ce'] = (glb_cu_expansion_pad_yield_min, glb_cu_expansion_pad_yield_max)
    # print("Cu Expansion Pad Yield Min: {:.6f}".format(glb_cu_expansion_pad_yield_min))
    # print("Cu Expansion Pad Yield Max: {:.6f}".format(glb_cu_expansion_pad_yield_max))


    if cfg.plot_flag:
        # Draw the pad yield map
        plt.figure(figsize=(8, 6))
        plt.imshow(
            pad_yield_map,
            cmap='viridis',
            vmin=interface.glb_pad_yield_min_max_dict['Y_ce'][0],
            vmax=interface.glb_pad_yield_min_max_dict['Y_ce'][1],
            interpolation='nearest',
            )
        cb = plt.colorbar(label='Pad Cu Expansion Yield')
        cb.ax.yaxis.label.set_size(16)
        plt.title('Pad Mechanical Stress Yield Map', fontsize=16)
        plt.xlabel('Pad Column Index', fontsize=16)
        plt.ylabel('Pad Row Index', fontsize=16)
        plt.show()
        # raise NotImplementedError("Disabled detailed pad yield map plot to reduce runtime.")



    return pad_yield_map
