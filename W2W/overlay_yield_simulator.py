#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
from scipy.optimize import fsolve
import sympy as sp
# from scipy.integrate import quad
# from scipy.stats import norm
import time
import os

try:
    from warpage_yield_simulator import sample_interface_bow_difference
except ModuleNotFoundError:
    from W2W.warpage_yield_simulator import sample_interface_bow_difference



# Calculate the misalignment of the pad based on the systematic translation, rotation, and magnification
def die_pad_misalignment(
    die: object,
    base_pad_coords: np.ndarray,
    system_translation_x_um: np.ndarray,
    system_translation_y_um: np.ndarray,
    system_rotation_rad: np.ndarray,
    system_magnification_ppm: np.ndarray,
    RANDOM_MISALIGNMENT_MEAN_um: float,
    RANDOM_MISALIGNMENT_STD_um: float,
    approximate_set: int,
) -> np.ndarray:
    # Consider pad misalignment for all pads
    die_pad_coords = base_pad_coords + die.die_center
    pad_misalignment = np.zeros(len(die_pad_coords))
    dx = (system_translation_x_um - system_rotation_rad * die_pad_coords[:, 1] + system_magnification_ppm * die_pad_coords[:, 0])
    dy = (system_translation_y_um + system_rotation_rad * die_pad_coords[:, 0] + system_magnification_ppm * die_pad_coords[:, 1])
    pad_misalignment = np.sqrt(dx**2 + dy**2) + np.random.normal(RANDOM_MISALIGNMENT_MEAN_um, RANDOM_MISALIGNMENT_STD_um, len(die_pad_coords))

    return pad_misalignment


def MAX_ALLOWED_MISALIGNMENT_calculator(
    cfg, PAD_TOP_R_um, PAD_BOT_R_um, PITCH_r_um, PITCH_c_um, CONTACT_AREA_CONSTRAINT, CRITICAL_DIST_CONSTRAINT
) -> float:
    # Calculate the overlay misalignment that will fail the contact area constraint
    system_misalignment = sp.symbols("system_misalignment")
    theta1 = sp.acos((PAD_TOP_R_um**2 + system_misalignment**2 - PAD_BOT_R_um**2) / (2 * PAD_TOP_R_um * system_misalignment))
    theta2 = sp.acos((PAD_BOT_R_um**2 + system_misalignment**2 - PAD_TOP_R_um**2) / (2 * PAD_BOT_R_um * system_misalignment))
    contact_area = (PAD_TOP_R_um**2 * theta1 + PAD_BOT_R_um**2 * theta2 - system_misalignment * (PAD_TOP_R_um * sp.sin(theta1)))
    equation = sp.lambdify(system_misalignment, contact_area - CONTACT_AREA_CONSTRAINT * np.pi * PAD_TOP_R_um**2, "numpy")
    MAX_ALLOWED_MISALIGNMENT_um_for_ca = fsolve(equation, PAD_BOT_R_um)
    # print("The overlay misalignment that will fail the contact area constraint is {} um.".format(MAX_ALLOWED_MISALIGNMENT_um_for_ca[0]))
    # Calculate the overlay misalignment that will fail the contact area constraint
    system_misalignment = np.linspace(PAD_BOT_R_um - PAD_TOP_R_um + 1e-9, PAD_BOT_R_um + PAD_TOP_R_um - 1e-9, 1000)
    theta1 = np.arccos((PAD_TOP_R_um**2 + system_misalignment**2 - PAD_BOT_R_um**2) / (2 * PAD_TOP_R_um * system_misalignment))
    theta2 = np.arccos((PAD_BOT_R_um**2 + system_misalignment**2 - PAD_TOP_R_um**2) / (2 * PAD_BOT_R_um * system_misalignment))
    contact_area = (PAD_TOP_R_um**2 * theta1 + PAD_BOT_R_um**2 * theta2 - system_misalignment * (PAD_TOP_R_um * np.sin(theta1)))
    # plt.plot(system_misalignment, contact_area / (np.pi * PAD_TOP_R_um**2))
    # plt.axhline(y=CONTACT_AREA_CONSTRAINT, color="r", linestyle="--")
    # plt.axvline(x=MAX_ALLOWED_MISALIGNMENT_um_for_ca, color="g", linestyle="--")
    # plt.xlabel("System Misalignment (um)")
    # plt.ylabel("Contact Area Ratio")
    # plt.title("Contact Area Ratio vs. System Misalignment")
    # plt.show()

    # Calculate the overlay misalignment that will fail the critical distance constraint
    if cfg.PAD_ARRANGE_PATTERN == 'hexagonal':
        EFF_PITCH_UM = min(np.sqrt(PITCH_r_um ** 2 + PITCH_c_um ** 2), 2 * PITCH_r_um, 2 * PITCH_c_um)
    elif cfg.PAD_ARRANGE_PATTERN == 'rectangular':
        EFF_PITCH_UM = min(PITCH_r_um, PITCH_c_um)
    MAX_ALLOWED_MISALIGNMENT_um_for_cd = (1 - CRITICAL_DIST_CONSTRAINT) * EFF_PITCH_UM - 0.5 * (2 * PAD_TOP_R_um) + (CRITICAL_DIST_CONSTRAINT - 0.5) * (2 * PAD_BOT_R_um)
    # print("The overlay misalignment that will fail the critical distance constraint is {} um.".format(MAX_ALLOWED_MISALIGNMENT_um_for_cd))

    MAX_ALLOWED_MISALIGNMENT_um = min(MAX_ALLOWED_MISALIGNMENT_um_for_ca[0], MAX_ALLOWED_MISALIGNMENT_um_for_cd)
    # print("The overlay misalignment that will fail the both constraints is {} um.".format(MAX_ALLOWED_MISALIGNMENT_um))

    return MAX_ALLOWED_MISALIGNMENT_um


def overlay_term_simulator(
    cfg_dict,
    waf_stack_list: list,
    _3dbx_path: str = None,
    bow_difference_samples_by_interface: dict = None,
):
    NUM_STACKS = len(waf_stack_list)
    if bow_difference_samples_by_interface is None:
        bow_difference_samples_by_interface = {}
    if not bow_difference_samples_by_interface:
        if _3dbx_path is None or not os.path.exists(_3dbx_path):
            raise FileNotFoundError(
                "W2W overlay simulation requires generated_stack_config.3dbx "
                "or explicit bow_difference_samples_by_interface because "
                "magnification is derived from warpage."
            )
        bow_difference_samples_by_interface = sample_interface_bow_difference(
            cfg_dict,
            _3dbx_path,
            num_samples=NUM_STACKS,
        )

    for interface_name, cfg in cfg_dict.items():
        # Extract input parameters from the current cfg
        PAD_BOT_R_um, PAD_TOP_R_um = cfg.PAD_BOT_R_um, cfg.PAD_TOP_R_um
        PITCH_r_um, PITCH_c_um = cfg.PITCH_r_um, cfg.PITCH_c_um
        CONTACT_AREA_CONSTRAINT = cfg.CONTACT_AREA_CONSTRAINT
        CRITICAL_DIST_CONSTRAINT = cfg.CRITICAL_DIST_CONSTRAINT
        SYSTEM_ROTATION_MEAN_rad = cfg.SYSTEM_ROTATION_MEAN_rad
        SYSTEM_ROTATION_STD_rad = cfg.SYSTEM_ROTATION_STD_rad
        SYSTEM_TRANSLATION_X_MEAN_um = cfg.SYSTEM_TRANSLATION_X_MEAN_um
        SYSTEM_TRANSLATION_X_STD_um = cfg.SYSTEM_TRANSLATION_X_STD_um
        SYSTEM_TRANSLATION_Y_MEAN_um = cfg.SYSTEM_TRANSLATION_Y_MEAN_um
        SYSTEM_TRANSLATION_Y_STD_um = cfg.SYSTEM_TRANSLATION_Y_STD_um
        k_mag = cfg.k_mag
        M_0 = cfg.M_0

        # Calculate the maximum allowed misalignment
        MAX_ALLOWED_MISALIGNMENT_um = MAX_ALLOWED_MISALIGNMENT_calculator(
            cfg, PAD_TOP_R_um, PAD_BOT_R_um, PITCH_r_um, PITCH_c_um, CONTACT_AREA_CONSTRAINT, CRITICAL_DIST_CONSTRAINT
        )

        # Calculate the systematic translation, rotation, and magnification
        system_translation_x_um_list = (
            np.random.normal(SYSTEM_TRANSLATION_X_MEAN_um, SYSTEM_TRANSLATION_X_STD_um, (NUM_STACKS))
        )
        system_translation_y_um_list = (
            np.random.normal(SYSTEM_TRANSLATION_Y_MEAN_um, SYSTEM_TRANSLATION_Y_STD_um, (NUM_STACKS))
        )
        system_rotation_rad_list = (
            np.random.normal(SYSTEM_ROTATION_MEAN_rad, SYSTEM_ROTATION_STD_rad, (NUM_STACKS))
        )
        if interface_name not in bow_difference_samples_by_interface:
            raise KeyError(
                f"Missing warpage-derived bow-difference samples for W2W interface "
                f"'{interface_name}'."
            )
        bow_difference_list = np.asarray(
            bow_difference_samples_by_interface[interface_name],
            dtype=float,
        )
        if len(bow_difference_list) != NUM_STACKS:
            raise ValueError(
                f"Interface '{interface_name}' has {len(bow_difference_list)} "
                f"bow-difference samples, but {NUM_STACKS} wafer stacks are simulated."
            )
        system_magnification_ppm_list = (
            (k_mag * bow_difference_list + M_0) / 1e6
        )  # systematic magnification unit (ppm)

        # Pass the overlay parameters to the wafer_stacks interface object
        for stack_idx in range(NUM_STACKS):
            waf_stack_list[stack_idx].interfaces.failure_params_dict[interface_name]['MAX_ALLOWED_MISALIGNMENT_um'] = MAX_ALLOWED_MISALIGNMENT_um
            waf_stack_list[stack_idx].interfaces.failure_params_dict[interface_name]['system_translation_x_um'] = system_translation_x_um_list[stack_idx]
            waf_stack_list[stack_idx].interfaces.failure_params_dict[interface_name]['system_translation_y_um'] = system_translation_y_um_list[stack_idx]
            waf_stack_list[stack_idx].interfaces.failure_params_dict[interface_name]['system_rotation_rad'] = system_rotation_rad_list[stack_idx]
            waf_stack_list[stack_idx].interfaces.failure_params_dict[interface_name]['system_magnification_ppm'] = system_magnification_ppm_list[stack_idx]
