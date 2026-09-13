#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np
import time
import matplotlib.pyplot as plt
from wafer_die_initialization import die_initialize
from overlay_yield_calculator import overlay_yield_calculator
from defect_yield_calculator import defect_yield_calculator
from Cu_expansion_yield_calculator import Cu_expansion_yield_calculator
from utils.util import configure_random_seed





def Assembly_Yield_Calculator(
    cfg,
    pad_bitmap_collection: dict,
):  
    configure_random_seed(cfg, announce=False)
    expected_bitmap_shape = (cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL)
    actual_bitmap_shape = pad_bitmap_collection["CRITICAL_PAD_BITMAP"].shape
    if actual_bitmap_shape != expected_bitmap_shape:
        raise ValueError(
            "Pad bitmap shape does not match the current modeling configuration: "
            f"bitmap={actual_bitmap_shape}, expected={expected_bitmap_shape}. "
            "Call update_config_items() before generating the bitmap."
        )
    # Initialize the die list
    die_list, _ = die_initialize(
        NUM_DIES            =       1,
        DIE_W_um            =       cfg.DIE_W_um,
        DIE_L_um            =       cfg.DIE_L_um,
        PAD_ARR_W_um        =       cfg.PAD_ARR_W_um,
        PAD_ARR_L_um        =       cfg.PAD_ARR_L_um,
        PAD_ARR_ROW         =       cfg.PAD_ARR_ROW,
        PAD_ARR_COL         =       cfg.PAD_ARR_COL,
        PITCH_um            =       cfg.PITCH_um,
        pad_bitmap_collection = pad_bitmap_collection,  
        pad_yield_flag      =       cfg.pad_yield_flag,
    )
    die = die_list[0]
    # fig, ax = plt.subplots(figsize=(4, 6))
    # die.draw_die(ax)

    # Calculate the overlay yield
    overlay_die_yield, overlay_pad_yield_map = overlay_yield_calculator(
        PAD_TOP_R_um                    =       cfg.PAD_TOP_R_um,
        PAD_BOT_R_um                    =       cfg.PAD_BOT_R_um,
        PAD_ARR_ROW                     =       cfg.PAD_ARR_ROW,
        PAD_ARR_COL                     =       cfg.PAD_ARR_COL,
        PITCH_um                        =       cfg.PITCH_um,
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
        die                             =       die,
        redundant_flag                  =       cfg.redundant_flag,
        pad_yield_flag                  =       cfg.pad_yield_flag,
        pad_yield_map_sub_factor        =       cfg.pad_yield_map_sub_factor,
        scale_systematic_distortion_from_wafer = bool(
            getattr(cfg, "scale_systematic_distortion_from_wafer", False)
        ),
        WAF_R_um                        =       cfg.WAF_R_um,
    )
    die.die_yield['Y_ovl'], die.pad_yield_map['Y_ovl'] = overlay_die_yield, overlay_pad_yield_map

    # Calculate the defect yield
    start_time = time.time()
    defect_die_yield, defect_pad_yield_map = defect_yield_calculator(
        cfg               =       cfg,
        eff_DIE_R         =       cfg.eff_DIE_R,
        D0                =       cfg.D0,
        t_0               =       cfg.t_0,
        z                 =       cfg.z,
        k_r               =       cfg.k_r,
        k_r0              =       cfg.k_r0,
        k_n               =       cfg.k_n,
        k_S               =       cfg.k_S,
        k_L               =       cfg.k_L,
        PAD_TOP_R_um      =       cfg.PAD_TOP_R_um,
        PITCH_um          =       cfg.PITCH_um,
        PAD_ARR_ROW       =       cfg.PAD_ARR_ROW,
        PAD_ARR_COL       =       cfg.PAD_ARR_COL,
        PAD_ARR_W_um      =       cfg.PAD_ARR_W_um,
        PAD_ARR_L_um      =       cfg.PAD_ARR_L_um,
        VOID_SHAPE        =       cfg.VOID_SHAPE,
        die               =       die,
        pad_bitmap_collection  = pad_bitmap_collection,
        pad_yield_flag    =       cfg.pad_yield_flag,
        pad_yield_map_sub_factor = cfg.pad_yield_map_sub_factor,
    )
    die.die_yield['Y_df'], die.pad_yield_map['Y_df'] = defect_die_yield, defect_pad_yield_map
    print(f"Defect yield calculation took {time.time() - start_time:.4f} seconds")

    # Calculate the Cu expansion yield
    Cu_expansion_die_yield, Cu_expansion_pad_yield_map = Cu_expansion_yield_calculator(
        cfg                 =       cfg,
        die                 =       die,
        TOP_DISH_MEAN_nm    =       cfg.TOP_DISH_MEAN_nm,
        TOP_DISH_STD_nm     =       cfg.TOP_DISH_STD_nm,
        BOT_DISH_MEAN_nm    =       cfg.BOT_DISH_MEAN_nm,
        BOT_DISH_STD_nm     =       cfg.BOT_DISH_STD_nm,
        k_et                =       cfg.k_et,
        k_eb                =       cfg.k_eb,
        T_R                 =       cfg.T_R,
        T_anl               =       cfg.T_anl,
        pad_bitmap_collection  = pad_bitmap_collection,
        pad_yield_flag      =       cfg.pad_yield_flag,
    )
    die.die_yield['Y_cr'], die.pad_yield_map['Y_cr'] = Cu_expansion_die_yield, Cu_expansion_pad_yield_map
    assembly_die_yield = overlay_die_yield * defect_die_yield * Cu_expansion_die_yield
    assembly_pad_yield_map = overlay_pad_yield_map * defect_pad_yield_map * Cu_expansion_pad_yield_map if cfg.pad_yield_flag else None
    die.die_yield['Y_asmb'], die.pad_yield_map['Y_asmb'] = assembly_die_yield, assembly_pad_yield_map
    die_yield = die.die_yield
    pad_yield_map = die.pad_yield_map
    del die
    return die_yield, pad_yield_map
