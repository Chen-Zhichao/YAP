#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#### Author: Zhichao Chen
#### Date: Oct 3, 2025

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import os
from scipy.optimize import fsolve
import sympy as sp
from scipy.integrate import quad
import time
from scipy.stats import norm

try:
    from warpage_yield_calculator import get_interface_existing_stack_warpage_map
except ModuleNotFoundError:
    from W2W.warpage_yield_calculator import get_interface_existing_stack_warpage_map

'''
Overlay yield calculator for W2W hybrid bonding:
1. Calculate the maximum allowed misalignment
2. Calculate the systematic misalignment for every pad based on the systematic translation, rotation, and magnification
3. Calculate the overlay yield:
    i. If pad_yield_flag is True, calculate the overlay yield for each pad and return the pad yield map.
    ii. If pad_yield_flag is False, calculate the overlay yield for the die based on the worst-case pad misalignment.
4. Calculate the overall overlay yield for the die.
'''

# Calculate the misalignment of the pad based on the systematic translation, rotation, and magnification
def die_pad_misalignment(
    die,
    system_translation_x_um,
    system_translation_y_um,
    system_rotation_rad,
    system_magnification_ppm,
):
    pad_misalignment = np.zeros(len(die.pad_array_box))
    dx = (system_translation_x_um - system_rotation_rad * die.pad_array_box[:, 1] + system_magnification_ppm * die.pad_array_box[:, 0])
    dy = (system_translation_y_um + system_rotation_rad * die.pad_array_box[:, 0] + system_magnification_ppm * die.pad_array_box[:, 1])
    pad_misalignment = np.sqrt(dx**2 + dy**2)
    return pad_misalignment


def max_allowed_misalignment_calculator(
        cfg, PAD_TOP_R_um, PAD_BOT_R_um, PITCH_r_um, PITCH_c_um, CONTACT_AREA_CONSTRAINT, CRITICAL_DIST_CONSTRAINT
    ):
        # Calculate the overlay misalignment that will fail the contact area constraint
        system_misalignment = sp.symbols("system_misalignment")
        theta1 = sp.acos((PAD_TOP_R_um**2 + system_misalignment**2 - PAD_BOT_R_um**2) / (2 * PAD_TOP_R_um * system_misalignment))
        theta2 = sp.acos((PAD_BOT_R_um**2 + system_misalignment**2 - PAD_TOP_R_um**2) / (2 * PAD_BOT_R_um * system_misalignment))
        contact_area = (PAD_TOP_R_um**2 * theta1 + PAD_BOT_R_um**2 * theta2 - system_misalignment * (PAD_TOP_R_um * sp.sin(theta1)))
        equation = sp.lambdify(system_misalignment, contact_area - CONTACT_AREA_CONSTRAINT * np.pi * PAD_TOP_R_um**2, "numpy")
        max_allowed_misalignment_for_ca = fsolve(equation, PAD_BOT_R_um)
        # print("The overlay misalignment that will fail the contact area constraint is {} um.".format(max_allowed_misalignment_for_ca[0]))
        # Calculate the overlay misalignment that will fail the contact area constraint
        system_misalignment = np.linspace(PAD_BOT_R_um - PAD_TOP_R_um + 1e-9, PAD_BOT_R_um + PAD_TOP_R_um - 1e-9, 1000)
        theta1 = np.arccos((PAD_TOP_R_um**2 + system_misalignment**2 - PAD_BOT_R_um**2) / (2 * PAD_TOP_R_um * system_misalignment))
        theta2 = np.arccos((PAD_BOT_R_um**2 + system_misalignment**2 - PAD_TOP_R_um**2) / (2 * PAD_BOT_R_um * system_misalignment))
        contact_area = (PAD_TOP_R_um**2 * theta1 + PAD_BOT_R_um**2 * theta2 - system_misalignment * (PAD_TOP_R_um * np.sin(theta1)))
        # plt.plot(system_misalignment, contact_area / (np.pi * PAD_TOP_R_um**2))
        # plt.axhline(y=CONTACT_AREA_CONSTRAINT, color="r", linestyle="--")
        # plt.axvline(x=max_allowed_misalignment_for_ca, color="g", linestyle="--")
        # plt.xlabel("System Misalignment (um)")
        # plt.ylabel("Contact Area Ratio")
        # plt.title("Contact Area Ratio vs. System Misalignment")
        # plt.show()

        # Calculate the overlay misalignment that will fail the critical distance constraint
        if cfg.PAD_ARRANGE_PATTERN == 'checkerboard':
            EFF_PITCH_UM = min(np.sqrt(PITCH_r_um ** 2 + PITCH_c_um ** 2), 2 * PITCH_r_um, 2 * PITCH_c_um)
        else:
            EFF_PITCH_UM = min(PITCH_r_um, PITCH_c_um)
        max_allowed_misalignment_for_cd = (1 - CRITICAL_DIST_CONSTRAINT) * EFF_PITCH_UM - 0.5 * (2 * PAD_TOP_R_um) + (CRITICAL_DIST_CONSTRAINT - 0.5) * (2 * PAD_BOT_R_um)

        MAX_ALLOWED_MISALIGNMENT_um = min(max_allowed_misalignment_for_ca[0], max_allowed_misalignment_for_cd)
        # print("The overlay misalignment that will fail the both constraints is {} um.".format(MAX_ALLOWED_MISALIGNMENT))

        return MAX_ALLOWED_MISALIGNMENT_um

def _interface_bow_difference_stats(
    cfg_dict,
    _3dbx_path,
    num_dies_per_wafer=None,
):
    """
    Return Gaussian bow-difference stats for every W2W interface.

    bow_difference = incoming_top_wafer_initial_bow
                     - existing_stack_post_anneal_bow
    """
    if not _3dbx_path or not os.path.exists(_3dbx_path):
        raise FileNotFoundError(
            "W2W overlay bow-difference modeling requires generated_stack_config.3dbx. "
            f"Received: {_3dbx_path}"
        )

    interface_stack_warpage = get_interface_existing_stack_warpage_map(
        cfg_dict,
        _3dbx_path,
        num_dies_per_wafer=num_dies_per_wafer,
    )
    bow_difference_stats = {}
    for interface_name in cfg_dict:
        if interface_name not in interface_stack_warpage:
            raise KeyError(
                f"Interface '{interface_name}' is missing from W2W stack warpage map. "
                "Overlay magnification now derives from incoming wafer initial bow "
                "and existing sequential post-anneal stack warpage."
            )

        stats = interface_stack_warpage[interface_name]
        top_mu_um = float(stats["top_wafer_mu_um"])
        top_sigma_um = max(float(stats["top_wafer_sigma_um"]), 0.0)
        stack_mu_um = float(stats["mu_um"])
        stack_sigma_um = max(float(stats["sigma_um"]), 0.0)
        bow_difference_stats[interface_name] = {
            "bow_difference_mean_um": top_mu_um - stack_mu_um,
            "bow_difference_std_um": float(
                np.sqrt(top_sigma_um**2 + stack_sigma_um**2)
            ),
            "top_wafer_mean_um": top_mu_um,
            "top_wafer_std_um": top_sigma_um,
            "existing_stack_mean_um": stack_mu_um,
            "existing_stack_std_um": stack_sigma_um,
            "source": "sequential_stack_warpage_model",
        }

    return bow_difference_stats


def stack_overlay_yield_calculator(
    cfg_dict: dict,
    waf_stack,
    _3dbx_path: str = None,
):
    bow_difference_stats = _interface_bow_difference_stats(
        cfg_dict,
        _3dbx_path,
        num_dies_per_wafer=waf_stack.num_dies_per_wafer,
    )

    for interface_name, cfg in cfg_dict.items():
        PAD_BOT_R_um, PAD_TOP_R_um = cfg.PAD_BOT_R_um, cfg.PAD_TOP_R_um
        num_samples = cfg.num_samples
        PITCH_r_um, PITCH_c_um = cfg.PITCH_r_um, cfg.PITCH_c_um
        CONTACT_AREA_CONSTRAINT = cfg.CONTACT_AREA_CONSTRAINT
        CRITICAL_DIST_CONSTRAINT = cfg.CRITICAL_DIST_CONSTRAINT
        SYSTEM_ROTATION_MEAN_rad = cfg.SYSTEM_ROTATION_MEAN_rad
        SYSTEM_ROTATION_STD_rad = cfg.SYSTEM_ROTATION_STD_rad
        SYSTEM_TRANSLATION_X_MEAN_um = cfg.SYSTEM_TRANSLATION_X_MEAN_um
        SYSTEM_TRANSLATION_X_STD_um = cfg.SYSTEM_TRANSLATION_X_STD_um
        SYSTEM_TRANSLATION_Y_MEAN_um = cfg.SYSTEM_TRANSLATION_Y_MEAN_um
        SYSTEM_TRANSLATION_Y_STD_um = cfg.SYSTEM_TRANSLATION_Y_STD_um
        RANDOM_MISALIGNMENT_MEAN_um = cfg.RANDOM_MISALIGNMENT_MEAN_um
        RANDOM_MISALIGNMENT_STD_um = cfg.RANDOM_MISALIGNMENT_STD_um

        MAX_ALLOWED_MISALIGNMENT_um = max_allowed_misalignment_calculator(
            cfg,
            PAD_TOP_R_um,
            PAD_BOT_R_um,
            PITCH_r_um,
            PITCH_c_um,
            CONTACT_AREA_CONSTRAINT,
            CRITICAL_DIST_CONSTRAINT,
        )
        system_translation_x_samples_um = np.random.normal(SYSTEM_TRANSLATION_X_MEAN_um, SYSTEM_TRANSLATION_X_STD_um, num_samples)
        system_translation_y_samples_um = np.random.normal(SYSTEM_TRANSLATION_Y_MEAN_um, SYSTEM_TRANSLATION_Y_STD_um, num_samples)
        system_rotation_samples_rad = np.random.normal(SYSTEM_ROTATION_MEAN_rad, SYSTEM_ROTATION_STD_rad, num_samples)
        interface_bow_stats = bow_difference_stats[interface_name]
        magnification_mean = (
            cfg.k_mag * interface_bow_stats["bow_difference_mean_um"] + cfg.M_0
        ) / 1e6
        magnification_sigma = (
            abs(cfg.k_mag) * interface_bow_stats["bow_difference_std_um"]
        ) / 1e6
        system_magnification_samples_ppm = np.random.normal(
            magnification_mean,
            magnification_sigma,
            num_samples,
        )
        overlay_die_yield_list = []

        # print(system_translation_x_samples_um.mean()*1e3, " nm")
        # print(system_translation_y_samples_um.mean()*1e3, " nm")
        # print(system_rotation_samples_rad.mean() * 150e+3 * 1e3, " nm")
        # print(system_magnification_samples_ppm.mean() * 150e+3 * 1e3, " nm")
        
        # # Record the time
        # start_time = time.time()
        for die_id, die in enumerate(waf_stack.interfaces.interface_dict[interface_name].die_list):
            far_dx_samples_0 = (system_translation_x_samples_um - system_rotation_samples_rad * die.pad_array_box[0, 1] + system_magnification_samples_ppm * die.pad_array_box[0, 0])
            far_dy_samples_0 = (system_translation_y_samples_um + system_rotation_samples_rad * die.pad_array_box[0, 0] + system_magnification_samples_ppm * die.pad_array_box[0, 1])
            far_dx_samples_1 = (system_translation_x_samples_um - system_rotation_samples_rad * die.pad_array_box[1, 1] + system_magnification_samples_ppm * die.pad_array_box[1, 0])
            far_dy_samples_1 = (system_translation_y_samples_um + system_rotation_samples_rad * die.pad_array_box[1, 0] + system_magnification_samples_ppm * die.pad_array_box[1, 1])
            far_dx_samples_2 = (system_translation_x_samples_um - system_rotation_samples_rad * die.pad_array_box[2, 1] + system_magnification_samples_ppm * die.pad_array_box[2, 0])
            far_dy_samples_2 = (system_translation_y_samples_um + system_rotation_samples_rad * die.pad_array_box[2, 0] + system_magnification_samples_ppm * die.pad_array_box[2, 1])
            far_dx_samples_3 = (system_translation_x_samples_um - system_rotation_samples_rad * die.pad_array_box[3, 1] + system_magnification_samples_ppm * die.pad_array_box[3, 0])
            far_dy_samples_3 = (system_translation_y_samples_um + system_rotation_samples_rad * die.pad_array_box[3, 0] + system_magnification_samples_ppm * die.pad_array_box[3, 1])
            far_pad_misalignment_samples_0 = np.sqrt(far_dx_samples_0**2 + far_dy_samples_0**2)
            far_pad_misalignment_samples_1 = np.sqrt(far_dx_samples_1**2 + far_dy_samples_1**2)
            far_pad_misalignment_samples_2 = np.sqrt(far_dx_samples_2**2 + far_dy_samples_2**2)
            far_pad_misalignment_samples_3 = np.sqrt(far_dx_samples_3**2 + far_dy_samples_3**2)

            upper_limit_0 = MAX_ALLOWED_MISALIGNMENT_um - far_pad_misalignment_samples_0
            lower_limit_0 = -MAX_ALLOWED_MISALIGNMENT_um - far_pad_misalignment_samples_0
            upper_limit_1 = MAX_ALLOWED_MISALIGNMENT_um - far_pad_misalignment_samples_1
            lower_limit_1 = -MAX_ALLOWED_MISALIGNMENT_um - far_pad_misalignment_samples_1
            upper_limit_2 = MAX_ALLOWED_MISALIGNMENT_um - far_pad_misalignment_samples_2
            lower_limit_2 = -MAX_ALLOWED_MISALIGNMENT_um - far_pad_misalignment_samples_2
            upper_limit_3 = MAX_ALLOWED_MISALIGNMENT_um - far_pad_misalignment_samples_3
            lower_limit_3 = -MAX_ALLOWED_MISALIGNMENT_um - far_pad_misalignment_samples_3
            
            current_die_corner_yield_0 = np.mean(norm.cdf(upper_limit_0, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um) - norm.cdf(lower_limit_0, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um))
            current_die_corner_yield_1 = np.mean(norm.cdf(upper_limit_1, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um) - norm.cdf(lower_limit_1, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um))
            current_die_corner_yield_2 = np.mean(norm.cdf(upper_limit_2, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um) - norm.cdf(lower_limit_2, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um))
            current_die_corner_yield_3 = np.mean(norm.cdf(upper_limit_3, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um) - norm.cdf(lower_limit_3, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um))

            current_die_yield = min(current_die_corner_yield_0, current_die_corner_yield_1, current_die_corner_yield_2, current_die_corner_yield_3)
            overlay_die_yield_list.append(current_die_yield)

        waf_stack.die_yield_list_per_interface_dict[interface_name]['overlay'] = np.array(overlay_die_yield_list)
