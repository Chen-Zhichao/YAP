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
    boundary_coords = getattr(die, "ovl_active_pad_boundary_coords", None)
    if boundary_coords is None:
        boundary_coords = die.pad_array_box
    pad_misalignment = np.zeros(len(boundary_coords))
    dx = (system_translation_x_um - system_rotation_rad * boundary_coords[:, 1] + system_magnification_ppm * boundary_coords[:, 0])
    dy = (system_translation_y_um + system_rotation_rad * boundary_coords[:, 0] + system_magnification_ppm * boundary_coords[:, 1])
    pad_misalignment = np.sqrt(dx**2 + dy**2)
    return pad_misalignment


def _residual_pass_probability(upper_limits_um, mean_um, std_um):
    if std_um <= 0:
        return (mean_um <= upper_limits_um).astype(float)
    return norm.cdf(upper_limits_um, loc=mean_um, scale=std_um)


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

        # print(system_translation_x_samples_um.mean()*1e3, " nm")
        # print(system_translation_y_samples_um.mean()*1e3, " nm")
        # print(system_rotation_samples_rad.mean() * 150e+3 * 1e3, " nm")
        # print(system_magnification_samples_ppm.mean() * 150e+3 * 1e3, " nm")
        
        die_list = waf_stack.interfaces.interface_dict[interface_name].die_list
        boundary_coords_um = np.asarray(
            [
                (
                    die.ovl_active_pad_boundary_coords
                    if getattr(die, "ovl_active_pad_boundary_coords", None) is not None
                    else die.pad_array_box
                )
                for die in die_list
            ],
            dtype=float,
        )
        overlay_die_yield = np.empty(len(die_list), dtype=float)

        if boundary_coords_um.shape[1] == 0:
            waf_stack.die_yield_list_per_interface_dict[interface_name]['overlay'] = np.ones(
                len(die_list),
                dtype=float,
            )
            continue

        tx = system_translation_x_samples_um[:, None, None]
        ty = system_translation_y_samples_um[:, None, None]
        theta = system_rotation_samples_rad[:, None, None]
        mag = system_magnification_samples_ppm[:, None, None]

        # Bound the broadcast working set by the number of sampled global
        # overlay states.  This keeps high-accuracy runs (for example,
        # 50,000 samples) practical without changing the calculation.
        max_broadcast_elements = 4_000_000
        num_corners = boundary_coords_um.shape[1]
        chunk_size = max(
            1,
            min(
                512,
                max_broadcast_elements // max(num_samples * num_corners, 1),
            ),
        )
        for start_idx in range(0, len(die_list), chunk_size):
            stop_idx = min(start_idx + chunk_size, len(die_list))
            coords = boundary_coords_um[start_idx:stop_idx]
            x = coords[None, :, :, 0]
            y = coords[None, :, :, 1]
            dx = tx - theta * y + mag * x
            dy = ty + theta * x + mag * y
            corner_misalignment = np.sqrt(dx**2 + dy**2)
            worst_corner_misalignment = np.max(corner_misalignment, axis=2)
            upper_limits = (
                MAX_ALLOWED_MISALIGNMENT_um - worst_corner_misalignment
            )
            overlay_die_yield[start_idx:stop_idx] = np.mean(
                _residual_pass_probability(
                    upper_limits,
                    RANDOM_MISALIGNMENT_MEAN_um,
                    RANDOM_MISALIGNMENT_STD_um,
                ),
                axis=0,
            )

        waf_stack.die_yield_list_per_interface_dict[interface_name]['overlay'] = overlay_die_yield
