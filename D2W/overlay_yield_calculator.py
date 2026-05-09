#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Author: Zhichao Chen
#### Date: Oct 1, 2025

'''
Overlay yield calculator for D2W hybrid bonding:
1. Calculate the maximum allowed misalignment
2. Calculate the systematic misalignment for every pad based on the systematic translation, rotation, and magnification
3. Calculate the interface-level overlay yield from the worst-case pad misalignment.
'''

import numpy as np
from scipy.optimize import fsolve
import sympy as sp
from scipy.stats import norm

# Calculate the misalignment of the pad based on the systematic translation, rotation, and magnification
def interface_pad_misalignment(
    interface,
    system_translation_x_um: float,
    system_translation_y_um: float,
    system_rotation_um: float,
    system_magnification: float,
):
    pad_misalignment = np.zeros(len(interface.pad_array_box))
    dx = (system_translation_x_um - system_rotation_um * interface.pad_array_box[:, 1] + system_magnification * interface.pad_array_box[:, 0])
    dy = (system_translation_y_um + system_rotation_um * interface.pad_array_box[:, 0] + system_magnification * interface.pad_array_box[:, 1])
    pad_misalignment = np.sqrt(dx**2 + dy**2)
    return pad_misalignment

def max_allowed_misalignment_calculator(*,
        cfg,
        PAD_TOP_R_um: float,
        PAD_BOT_R_um: float,
        PITCH_r_um: float,
        PITCH_c_um: float,
        CONTACT_AREA_CONSTRAINT,
        CRITICAL_DIST_CONSTRAINT
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
            PITCH_UM = min(np.sqrt(PITCH_r_um ** 2 + PITCH_c_um ** 2), 2 * PITCH_r_um, 2 * PITCH_c_um)
        else:
            PITCH_UM = min(PITCH_r_um, PITCH_c_um)
        max_allowed_misalignment_for_cd = (1 - CRITICAL_DIST_CONSTRAINT) * PITCH_UM - 0.5 * (2 * PAD_TOP_R_um) + (CRITICAL_DIST_CONSTRAINT - 0.5) * (2 * PAD_BOT_R_um)
        # print("The overlay misalignment that will fail the critical distance constraint is {} um.".format(max_allowed_misalignment_for_cd))

        MAX_ALLOWED_MISALIGNMENT = min(max_allowed_misalignment_for_ca[0], max_allowed_misalignment_for_cd)
        # print("The overlay misalignment that will fail the both constraints is {} um.".format(MAX_ALLOWED_MISALIGNMENT))

        return MAX_ALLOWED_MISALIGNMENT

def overlay_yield_calculator(*,
    cfg,
    PAD_TOP_R_um: float,
    PAD_BOT_R_um: float,
    PAD_ARR_ROW: int,
    PAD_ARR_COL: int,
    PITCH_r_um: float,
    PITCH_c_um: float,
    num_samples: int,
    CONTACT_AREA_CONSTRAINT: float,
    CRITICAL_DIST_CONSTRAINT: float,
    SYSTEM_MAGNIFICATION_MEAN_ppm: float,
    SYSTEM_MAGNIFICATION_STD_ppm: float,
    SYSTEM_ROTATION_MEAN_rad: float,
    SYSTEM_ROTATION_STD_rad: float,
    SYSTEM_TRANSLATION_X_MEAN_um: float,
    SYSTEM_TRANSLATION_X_STD_um: float,
    SYSTEM_TRANSLATION_Y_MEAN_um: float,
    SYSTEM_TRANSLATION_Y_STD_um: float,
    RANDOM_MISALIGNMENT_MEAN_um: float,
    RANDOM_MISALIGNMENT_STD_um: float,
    interface,
    redundant_flag: bool,
):
    MAX_ALLOWED_MISALIGNMENT = max_allowed_misalignment_calculator(
        cfg=cfg,
        PAD_TOP_R_um=PAD_TOP_R_um,
        PAD_BOT_R_um=PAD_BOT_R_um,
        PITCH_r_um=PITCH_r_um,
        PITCH_c_um=PITCH_c_um,
        CONTACT_AREA_CONSTRAINT=CONTACT_AREA_CONSTRAINT,
        CRITICAL_DIST_CONSTRAINT=CRITICAL_DIST_CONSTRAINT,
    )
    num_samples = num_samples
    system_translation_x_samples_um = np.random.normal(SYSTEM_TRANSLATION_X_MEAN_um, SYSTEM_TRANSLATION_X_STD_um, num_samples)
    system_translation_y_samples_um = np.random.normal(SYSTEM_TRANSLATION_Y_MEAN_um, SYSTEM_TRANSLATION_Y_STD_um, num_samples)
    system_rotation_samples_rad = np.random.normal(SYSTEM_ROTATION_MEAN_rad, SYSTEM_ROTATION_STD_rad, num_samples)
    system_magnification_samples = np.random.normal(SYSTEM_MAGNIFICATION_MEAN_ppm, SYSTEM_MAGNIFICATION_STD_ppm, num_samples)
    # print("system_translation_x_samples_um contribution", system_translation_x_samples_um.mean()*1e3, " nm")
    # print("system_translation_y_samples_um contribution", system_translation_y_samples_um.mean()*1e3, " nm")
    # print("system_rotation_samples_rad contribution", system_rotation_samples_rad.mean() * np.sqrt(die.DIE_W_um**2 + die.DIE_L_um**2) * 1e3, " nm")
    # print("system_magnification_samples contribution", system_magnification_samples.mean() * np.sqrt(die.DIE_W_um**2 + die.DIE_L_um**2) * 1e3, " nm")

    # Sample the systematic misalignment for corner pads based on the systematic translation, rotation, and magnification
    # Calculate the die yield based on the worst-case pad misalignment
    if redundant_flag == True:
        far_dx_samples_0 = (system_translation_x_samples_um - system_rotation_samples_rad * interface.ovl_critical_pad_boundary_coords[0, 1] + system_magnification_samples * interface.ovl_critical_pad_boundary_coords[0, 0])
        far_dy_samples_0 = (system_translation_y_samples_um + system_rotation_samples_rad * interface.ovl_critical_pad_boundary_coords[0, 0] + system_magnification_samples * interface.ovl_critical_pad_boundary_coords[0, 1])
        far_dx_samples_1 = (system_translation_x_samples_um - system_rotation_samples_rad * interface.ovl_critical_pad_boundary_coords[1, 1] + system_magnification_samples * interface.ovl_critical_pad_boundary_coords[1, 0])
        far_dy_samples_1 = (system_translation_y_samples_um + system_rotation_samples_rad * interface.ovl_critical_pad_boundary_coords[1, 0] + system_magnification_samples * interface.ovl_critical_pad_boundary_coords[1, 1])
        far_dx_samples_2 = (system_translation_x_samples_um - system_rotation_samples_rad * interface.ovl_critical_pad_boundary_coords[2, 1] + system_magnification_samples * interface.ovl_critical_pad_boundary_coords[2, 0])
        far_dy_samples_2 = (system_translation_y_samples_um + system_rotation_samples_rad * interface.ovl_critical_pad_boundary_coords[2, 0] + system_magnification_samples * interface.ovl_critical_pad_boundary_coords[2, 1])
        far_dx_samples_3 = (system_translation_x_samples_um - system_rotation_samples_rad * interface.ovl_critical_pad_boundary_coords[3, 1] + system_magnification_samples * interface.ovl_critical_pad_boundary_coords[3, 0])
        far_dy_samples_3 = (system_translation_y_samples_um + system_rotation_samples_rad * interface.ovl_critical_pad_boundary_coords[3, 0] + system_magnification_samples * interface.ovl_critical_pad_boundary_coords[3, 1])
    else:
        far_dx_samples_0 = (system_translation_x_samples_um - system_rotation_samples_rad * interface.pad_array_box[0, 1] + system_magnification_samples * interface.pad_array_box[0, 0])
        far_dy_samples_0 = (system_translation_y_samples_um + system_rotation_samples_rad * interface.pad_array_box[0, 0] + system_magnification_samples * interface.pad_array_box[0, 1])
        far_dx_samples_1 = (system_translation_x_samples_um - system_rotation_samples_rad * interface.pad_array_box[1, 1] + system_magnification_samples * interface.pad_array_box[1, 0])
        far_dy_samples_1 = (system_translation_y_samples_um + system_rotation_samples_rad * interface.pad_array_box[1, 0] + system_magnification_samples * interface.pad_array_box[1, 1])
        far_dx_samples_2 = (system_translation_x_samples_um - system_rotation_samples_rad * interface.pad_array_box[2, 1] + system_magnification_samples * interface.pad_array_box[2, 0])
        far_dy_samples_2 = (system_translation_y_samples_um + system_rotation_samples_rad * interface.pad_array_box[2, 0] + system_magnification_samples * interface.pad_array_box[2, 1])
        far_dx_samples_3 = (system_translation_x_samples_um - system_rotation_samples_rad * interface.pad_array_box[3, 1] + system_magnification_samples * interface.pad_array_box[3, 0])
        far_dy_samples_3 = (system_translation_y_samples_um + system_rotation_samples_rad * interface.pad_array_box[3, 0] + system_magnification_samples * interface.pad_array_box[3, 1])
    far_pad_misalignment_samples_0 = np.sqrt(far_dx_samples_0**2 + far_dy_samples_0**2)
    far_pad_misalignment_samples_1 = np.sqrt(far_dx_samples_1**2 + far_dy_samples_1**2)
    far_pad_misalignment_samples_2 = np.sqrt(far_dx_samples_2**2 + far_dy_samples_2**2)
    far_pad_misalignment_samples_3 = np.sqrt(far_dx_samples_3**2 + far_dy_samples_3**2)

    upper_limit_0 = MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_0
    lower_limit_0 = -MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_0
    upper_limit_1 = MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_1
    lower_limit_1 = -MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_1
    upper_limit_2 = MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_2
    lower_limit_2 = -MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_2
    upper_limit_3 = MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_3
    lower_limit_3 = -MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_3

    overlay_die_yield_0 = np.mean(norm.cdf(upper_limit_0, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um) \
                        - norm.cdf(lower_limit_0, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um))
    overlay_die_yield_1 = np.mean(norm.cdf(upper_limit_1, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um) \
                        - norm.cdf(lower_limit_1, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um))
    overlay_die_yield_2 = np.mean(norm.cdf(upper_limit_2, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um) \
                        - norm.cdf(lower_limit_2, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um))
    overlay_die_yield_3 = np.mean(norm.cdf(upper_limit_3, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um) \
                        - norm.cdf(lower_limit_3, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um))
    overlay_die_yield = min(overlay_die_yield_0, overlay_die_yield_1, overlay_die_yield_2, overlay_die_yield_3)


    return overlay_die_yield
