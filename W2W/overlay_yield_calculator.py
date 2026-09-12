#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#### Author: Zhichao Chen
#### Date: Oct 3, 2025

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from scipy.optimize import brentq
from scipy.integrate import quad
import time
from scipy.stats import norm


def _samplewise_worst_corner_yield(
    corner_misalignment_samples_um,
    max_allowed_misalignment_um,
    random_misalignment_mean_um,
    random_misalignment_std_um,
):
    """Average die yield after selecting the worst corner in each sample."""
    corner_samples = np.asarray(corner_misalignment_samples_um)
    if corner_samples.ndim != 2:
        raise ValueError("corner_misalignment_samples_um must be a 2-D array")
    worst_samples = np.max(corner_samples, axis=0)
    upper_limit = max_allowed_misalignment_um - worst_samples
    lower_limit = -max_allowed_misalignment_um - worst_samples
    sample_yield = norm.cdf(
        upper_limit,
        loc=random_misalignment_mean_um,
        scale=random_misalignment_std_um,
    ) - norm.cdf(
        lower_limit,
        loc=random_misalignment_mean_um,
        scale=random_misalignment_std_um,
    )
    return float(np.mean(sample_yield))


def _circle_overlap_area(distance_um, radius_1_um, radius_2_um):
    """Return the overlap area of two circles without singular endpoints."""
    distance_um = float(distance_um)
    radius_1_um = float(radius_1_um)
    radius_2_um = float(radius_2_um)

    if distance_um >= radius_1_um + radius_2_um:
        return 0.0
    if distance_um <= abs(radius_1_um - radius_2_um):
        return np.pi * min(radius_1_um, radius_2_um) ** 2

    cosine_1 = np.clip(
        (distance_um**2 + radius_1_um**2 - radius_2_um**2)
        / (2 * distance_um * radius_1_um),
        -1.0,
        1.0,
    )
    cosine_2 = np.clip(
        (distance_um**2 + radius_2_um**2 - radius_1_um**2)
        / (2 * distance_um * radius_2_um),
        -1.0,
        1.0,
    )
    radical = (
        (-distance_um + radius_1_um + radius_2_um)
        * (distance_um + radius_1_um - radius_2_um)
        * (distance_um - radius_1_um + radius_2_um)
        * (distance_um + radius_1_um + radius_2_um)
    )
    return (
        radius_1_um**2 * np.arccos(cosine_1)
        + radius_2_um**2 * np.arccos(cosine_2)
        - 0.5 * np.sqrt(max(radical, 0.0))
    )


def _contact_area_misalignment_limit(
    pad_top_radius_um, pad_bottom_radius_um, contact_area_constraint
):
    if pad_top_radius_um <= 0 or pad_bottom_radius_um <= 0:
        raise ValueError("Pad radii must be positive.")
    if not 0.0 <= contact_area_constraint <= 1.0:
        raise ValueError("CONTACT_AREA_CONSTRAINT must be between 0 and 1.")

    target_area = contact_area_constraint * np.pi * pad_top_radius_um**2
    maximum_area = np.pi * min(pad_top_radius_um, pad_bottom_radius_um) ** 2
    area_tolerance = np.finfo(float).eps * max(maximum_area, 1.0) * 16
    if target_area > maximum_area + area_tolerance:
        raise ValueError(
            "CONTACT_AREA_CONSTRAINT cannot be met even at perfect alignment "
            "for the configured pad radii."
        )

    lower_distance = abs(pad_top_radius_um - pad_bottom_radius_um)
    upper_distance = pad_top_radius_um + pad_bottom_radius_um
    if np.isclose(target_area, maximum_area, rtol=1e-12, atol=area_tolerance):
        return lower_distance
    if target_area <= 0.0:
        return upper_distance

    return brentq(
        lambda distance: _circle_overlap_area(
            distance, pad_top_radius_um, pad_bottom_radius_um
        )
        - target_area,
        lower_distance,
        upper_distance,
    )

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

def overlay_yield_calculator(
    cfg,
    PAD_ARR_ROW: int,
    PAD_ARR_COL: int,
    PAD_TOP_R_um: float,
    PAD_BOT_R_um: float,
    PITCH_um: float,
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
    wafer,    
    redundant_flag: bool,
    pad_yield_flag: bool = False,
    pad_yield_map_sub_factor: int = 1,
):
    def max_allowed_misalignment_calculator(
        PAD_TOP_R_um, PAD_BOT_R_um, PITCH_um, CONTACT_AREA_CONSTRAINT, CRITICAL_DIST_CONSTRAINT
    ):
        max_allowed_misalignment_for_ca = _contact_area_misalignment_limit(
            PAD_TOP_R_um, PAD_BOT_R_um, CONTACT_AREA_CONSTRAINT
        )

        # Calculate the overlay misalignment that will fail the critical distance constraint
        max_allowed_misalignment_for_cd = (1 - CRITICAL_DIST_CONSTRAINT) * PITCH_um - 0.5 * (2 * PAD_TOP_R_um) + (CRITICAL_DIST_CONSTRAINT - 0.5) * (2 * PAD_BOT_R_um)
        
        # print("The overlay misalignment that will fail the critical distance constraint is {} um.".format(max_allowed_misalignment_for_cd))

        MAX_ALLOWED_MISALIGNMENT = min(max_allowed_misalignment_for_ca, max_allowed_misalignment_for_cd)
        # print("The overlay misalignment that will fail the both constraints is {} um.".format(MAX_ALLOWED_MISALIGNMENT))

        return MAX_ALLOWED_MISALIGNMENT
    
    MAX_ALLOWED_MISALIGNMENT = max_allowed_misalignment_calculator(
        PAD_TOP_R_um,
        PAD_BOT_R_um,
        PITCH_um,
        CONTACT_AREA_CONSTRAINT,
        CRITICAL_DIST_CONSTRAINT,
    )
    num_samples = num_samples
    system_translation_x_samples_um = np.random.normal(SYSTEM_TRANSLATION_X_MEAN_um, SYSTEM_TRANSLATION_X_STD_um, num_samples)
    system_translation_y_samples_um = np.random.normal(SYSTEM_TRANSLATION_Y_MEAN_um, SYSTEM_TRANSLATION_Y_STD_um, num_samples)
    system_rotation_samples_rad = np.random.normal(SYSTEM_ROTATION_MEAN_rad, SYSTEM_ROTATION_STD_rad, num_samples)
    system_magnification_samples_ppm = np.random.normal(SYSTEM_MAGNIFICATION_MEAN_ppm, SYSTEM_MAGNIFICATION_STD_ppm, num_samples)
    overlay_die_yield_list = []
    if cfg.DEBUG == True:
        print("The maximum allowed misalignment is {:.4f} nm.".format(MAX_ALLOWED_MISALIGNMENT * 1e3))
        print("Mean X translation contribution: {:.4f} nm".format(system_translation_x_samples_um.mean() * 1e3))
        print("Mean Y translation contribution: {:.4f} nm".format(system_translation_y_samples_um.mean() * 1e3))
        print("Mean rotation contribution: {:.4f} nm".format(system_rotation_samples_rad.mean() * 150e+3 * 1e3))
        print("Mean magnification contribution: {:.4f} nm".format(system_magnification_samples_ppm.mean() * 150e+3 * 1e3))
    
    # # Record the time
    # start_time = time.time()
    for die_id, die in enumerate(wafer.die_list):
        if redundant_flag == True:
            far_dx_samples_0 = (system_translation_x_samples_um - system_rotation_samples_rad * die.ovl_critical_pad_boundary_coords[0, 1] + system_magnification_samples_ppm * die.ovl_critical_pad_boundary_coords[0, 0])
            far_dy_samples_0 = (system_translation_y_samples_um + system_rotation_samples_rad * die.ovl_critical_pad_boundary_coords[0, 0] + system_magnification_samples_ppm * die.ovl_critical_pad_boundary_coords[0, 1])
            far_dx_samples_1 = (system_translation_x_samples_um - system_rotation_samples_rad * die.ovl_critical_pad_boundary_coords[1, 1] + system_magnification_samples_ppm * die.ovl_critical_pad_boundary_coords[1, 0])
            far_dy_samples_1 = (system_translation_y_samples_um + system_rotation_samples_rad * die.ovl_critical_pad_boundary_coords[1, 0] + system_magnification_samples_ppm * die.ovl_critical_pad_boundary_coords[1, 1])
            far_dx_samples_2 = (system_translation_x_samples_um - system_rotation_samples_rad * die.ovl_critical_pad_boundary_coords[2, 1] + system_magnification_samples_ppm * die.ovl_critical_pad_boundary_coords[2, 0])
            far_dy_samples_2 = (system_translation_y_samples_um + system_rotation_samples_rad * die.ovl_critical_pad_boundary_coords[2, 0] + system_magnification_samples_ppm * die.ovl_critical_pad_boundary_coords[2, 1])
            far_dx_samples_3 = (system_translation_x_samples_um - system_rotation_samples_rad * die.ovl_critical_pad_boundary_coords[3, 1] + system_magnification_samples_ppm * die.ovl_critical_pad_boundary_coords[3, 0])
            far_dy_samples_3 = (system_translation_y_samples_um + system_rotation_samples_rad * die.ovl_critical_pad_boundary_coords[3, 0] + system_magnification_samples_ppm * die.ovl_critical_pad_boundary_coords[3, 1])
        else:
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

        # A Monte Carlo sample is one global systematic-overlay realization.
        # Determine whether that realization survives the worst die corner
        # before averaging across independent realizations.
        current_die_yield = _samplewise_worst_corner_yield(
            [
                far_pad_misalignment_samples_0,
                far_pad_misalignment_samples_1,
                far_pad_misalignment_samples_2,
                far_pad_misalignment_samples_3,
            ],
            MAX_ALLOWED_MISALIGNMENT,
            RANDOM_MISALIGNMENT_MEAN_um,
            RANDOM_MISALIGNMENT_STD_um,
        )
        overlay_die_yield_list.append(current_die_yield)
        
        if pad_yield_flag == True:
            # Pad coordinates are only needed for the optional pad-level map.
            current_die_pad_array = wafer.base_pad_coords + die.die_center
            glb_defect_pad_yield_min = 1.0  # Initialize to a high value
            glb_defect_pad_yield_max = 0.0  # Initialize to a low value
            # Sample the systematic misalignment for every pad based on the systematic translation, rotation, and magnification
            # Calculate the pad yield for each pad and return the pad yield map
            # When calculate the pad yield, we ignore the whether the pad is critical or not.
            nr = math.ceil(PAD_ARR_ROW / pad_yield_map_sub_factor)
            nc = math.ceil(PAD_ARR_COL / pad_yield_map_sub_factor)
            overlay_pad_yield_map = np.zeros((nr, nc))
            for kr in range(nr):
                r = round(kr * (PAD_ARR_ROW - 1) / (nr - 1))
                for kc in range(nc):
                    c = round(kc * (PAD_ARR_COL - 1) / (nc - 1))
                    i = r * PAD_ARR_COL + c
                    dx_array_samples_i = (system_translation_x_samples_um - system_rotation_samples_rad * current_die_pad_array[i, 1] + system_magnification_samples_ppm * current_die_pad_array[i, 0])
                    dy_array_samples_i = (system_translation_y_samples_um + system_rotation_samples_rad * current_die_pad_array[i, 0] + system_magnification_samples_ppm * current_die_pad_array[i, 1])
                    pad_misalignment_samples_i = np.sqrt(dx_array_samples_i**2 + dy_array_samples_i**2)
                    upper_limit_i = MAX_ALLOWED_MISALIGNMENT - pad_misalignment_samples_i
                    lower_limit_i = -MAX_ALLOWED_MISALIGNMENT - pad_misalignment_samples_i
                    overlay_pad_yield_map[kr, kc] = np.mean(
                                                        norm.cdf(upper_limit_i, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um) \
                                                        - norm.cdf(lower_limit_i, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um)
                                                        )
            glb_defect_pad_yield_min = min(glb_defect_pad_yield_min, np.min(overlay_pad_yield_map))
            glb_defect_pad_yield_max = max(glb_defect_pad_yield_max, np.max(overlay_pad_yield_map))
            print("Generated pad-level overlay yield map for die {}.".format(die_id))
            die.pad_yield_map['Y_ovl'] = overlay_pad_yield_map
        else:
            overlay_pad_yield_map = None
    
    if pad_yield_flag:
        wafer.glb_pad_yield_min_max_dict['Y_ovl'] = (glb_defect_pad_yield_min, glb_defect_pad_yield_max)

    overlay_die_yield = np.mean(overlay_die_yield_list)
    # end_time = time.time()
    # print("Time taken to calculate the far pad misalignment: {:.2f} seconds".format(end_time - start_time))
    # print("The overlay die yield is {}.".format(overlay_die_yield))

    return overlay_die_yield
