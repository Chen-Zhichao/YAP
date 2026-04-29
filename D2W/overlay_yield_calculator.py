#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Author: Zhichao Chen
#### Date: Oct 1, 2025

'''
Overlay yield calculator for D2W hybrid bonding:
1. Calculate the maximum allowed misalignment
2. Calculate the systematic misalignment for every pad based on the systematic translation, rotation, and magnification
3. Calculate the overlay yield:
    i. If pad_yield_flag is True, calculate the overlay yield for each pad and return the pad yield map.
    ii. If pad_yield_flag is False, calculate the overlay yield for the interface based on the worst-case pad misalignment.
4. Calculate the overall overlay yield for the interface.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import fsolve
import math
import sympy as sp
from scipy.integrate import quad
from scipy.stats import norm
import time

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @njit(fastmath=True)
    def _normal_cdf_numba(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


    @njit(parallel=True, fastmath=True)
    def _pad_overlay_yield_map_sub_numba(
        pad_x_coords,
        pad_y_coords,
        system_translation_x_samples_um,
        system_translation_y_samples_um,
        system_rotation_samples_rad,
        system_magnification_samples,
        MAX_ALLOWED_MISALIGNMENT,
        RANDOM_MISALIGNMENT_MEAN_um,
        RANDOM_MISALIGNMENT_STD_um,
    ):
        num_pads = pad_x_coords.shape[0]
        num_samples = system_translation_x_samples_um.shape[0]
        overlay_pad_yield_vec = np.empty(num_pads, dtype=np.float64)
        inv_random_misalignment_std = 1.0 / RANDOM_MISALIGNMENT_STD_um

        for pad_id in prange(num_pads):
            pad_x = pad_x_coords[pad_id]
            pad_y = pad_y_coords[pad_id]
            yield_acc = 0.0

            for sample_id in range(num_samples):
                dx = (
                    system_translation_x_samples_um[sample_id]
                    - system_rotation_samples_rad[sample_id] * pad_y
                    + system_magnification_samples[sample_id] * pad_x
                )
                dy = (
                    system_translation_y_samples_um[sample_id]
                    + system_rotation_samples_rad[sample_id] * pad_x
                    + system_magnification_samples[sample_id] * pad_y
                )
                pad_misalignment = math.sqrt(dx * dx + dy * dy)
                upper_limit = (
                    MAX_ALLOWED_MISALIGNMENT
                    - pad_misalignment
                    - RANDOM_MISALIGNMENT_MEAN_um
                ) * inv_random_misalignment_std
                lower_limit = (
                    -MAX_ALLOWED_MISALIGNMENT
                    - pad_misalignment
                    - RANDOM_MISALIGNMENT_MEAN_um
                ) * inv_random_misalignment_std
                yield_acc += _normal_cdf_numba(upper_limit) - _normal_cdf_numba(lower_limit)

            overlay_pad_yield_vec[pad_id] = yield_acc / num_samples

        return overlay_pad_yield_vec

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
    pad_yield_flag: bool = False,
    pad_yield_map_sub_factor: int = 1,
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







def pad_overlay_yield_map_generator(*,
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
    pad_yield_flag: bool = False,
    pad_yield_map_sub_factor: int = 1,
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
    # print("The maximum allowed misalignment is {:.2f} nm.".format(MAX_ALLOWED_MISALIGNMENT * 1e3))
    num_samples = num_samples
    system_translation_x_samples_um = np.random.normal(SYSTEM_TRANSLATION_X_MEAN_um, SYSTEM_TRANSLATION_X_STD_um, num_samples)
    system_translation_y_samples_um = np.random.normal(SYSTEM_TRANSLATION_Y_MEAN_um, SYSTEM_TRANSLATION_Y_STD_um, num_samples)
    system_rotation_samples_rad = np.random.normal(SYSTEM_ROTATION_MEAN_rad, SYSTEM_ROTATION_STD_rad, num_samples)
    system_magnification_samples = np.random.normal(SYSTEM_MAGNIFICATION_MEAN_ppm, SYSTEM_MAGNIFICATION_STD_ppm, num_samples)
    # print("system_translation_x_samples_um contribution", system_translation_x_samples_um.mean()*1e3, " nm")
    # print("system_translation_y_samples_um contribution", system_translation_y_samples_um.mean()*1e3, " nm")
    # print("system_rotation_samples_rad contribution", system_rotation_samples_rad.mean() * np.sqrt(interface.DIE_W_um**2 + interface.DIE_L_um**2) * 1e3, " nm")
    # print("system_magnification_samples contribution", system_magnification_samples.mean() * np.sqrt(interface.DIE_W_um**2 + interface.DIE_L_um**2) * 1e3, " nm")

    if pad_yield_flag == True:
        glb_defect_pad_yield_min = 1.0
        glb_defect_pad_yield_max = 0.0
        # Sample the systematic misalignment for every pad based on the systematic translation, rotation, and magnification
        # Calculate the pad yield for each pad and return the pad yield map
        # When calculate the pad yield, we ignore the whether the pad is critical or not.
        nr = math.ceil(PAD_ARR_ROW / pad_yield_map_sub_factor)
        nc = math.ceil(PAD_ARR_COL / pad_yield_map_sub_factor)
        # print("nr: {}, nc: {}".format(nr, nc))
        overlay_pad_yield_map_sub = np.zeros((nr, nc))
        use_legacy_overlay_pad_yield = bool(getattr(cfg, "overlay_pad_yield_use_legacy", False))

        if NUMBA_AVAILABLE and not use_legacy_overlay_pad_yield:
            if nr == 1:
                r_idx = np.array([0], dtype=np.int64)
            else:
                r_idx = np.round(np.linspace(0, PAD_ARR_ROW - 1, nr)).astype(np.int64)
            if nc == 1:
                c_idx = np.array([0], dtype=np.int64)
            else:
                c_idx = np.round(np.linspace(0, PAD_ARR_COL - 1, nc)).astype(np.int64)

            RR, CC = np.meshgrid(r_idx, c_idx, indexing='ij')
            sampled_pad_linear_idx = (RR * PAD_ARR_COL + CC).reshape(-1)
            sampled_pad_coords = interface.pad_coords[sampled_pad_linear_idx]

            start_time = time.perf_counter()
            overlay_pad_yield_vec = _pad_overlay_yield_map_sub_numba(
                pad_x_coords=np.ascontiguousarray(sampled_pad_coords[:, 0], dtype=np.float64),
                pad_y_coords=np.ascontiguousarray(sampled_pad_coords[:, 1], dtype=np.float64),
                system_translation_x_samples_um=np.ascontiguousarray(system_translation_x_samples_um, dtype=np.float64),
                system_translation_y_samples_um=np.ascontiguousarray(system_translation_y_samples_um, dtype=np.float64),
                system_rotation_samples_rad=np.ascontiguousarray(system_rotation_samples_rad, dtype=np.float64),
                system_magnification_samples=np.ascontiguousarray(system_magnification_samples, dtype=np.float64),
                MAX_ALLOWED_MISALIGNMENT=float(MAX_ALLOWED_MISALIGNMENT),
                RANDOM_MISALIGNMENT_MEAN_um=float(RANDOM_MISALIGNMENT_MEAN_um),
                RANDOM_MISALIGNMENT_STD_um=float(RANDOM_MISALIGNMENT_STD_um),
            )
            overlay_pad_yield_map_sub = overlay_pad_yield_vec.reshape(nr, nc)
            # print(
            #     "Pad yield map generation time for {} pads: {:.2f} seconds (numba)".format(
            #         nr * nc,
            #         time.perf_counter() - start_time,
            #     )
            # )
        else:
            start_time = time.perf_counter()
            for kr in range(nr):
                r = round(kr * (PAD_ARR_ROW - 1) / (nr - 1))
                for kc in range(nc):
                    c = round(kc * (PAD_ARR_COL - 1) / (nc - 1))
                    i = r * PAD_ARR_COL + c
                    dx_array_samples_i = (system_translation_x_samples_um - system_rotation_samples_rad * interface.pad_coords[i, 1] + system_magnification_samples * interface.pad_coords[i, 0])
                    dy_array_samples_i = (system_translation_y_samples_um + system_rotation_samples_rad * interface.pad_coords[i, 0] + system_magnification_samples * interface.pad_coords[i, 1])
                    pad_misalignment_samples_i = np.sqrt(dx_array_samples_i**2 + dy_array_samples_i**2)
                    upper_limit_i = MAX_ALLOWED_MISALIGNMENT - pad_misalignment_samples_i
                    lower_limit_i = -MAX_ALLOWED_MISALIGNMENT - pad_misalignment_samples_i
                    overlay_pad_yield_map_sub[kr, kc] = np.mean(
                                                norm.cdf(upper_limit_i, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um)  \
                                                - norm.cdf(lower_limit_i, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um)
                                            )
            # print(
            #     "Pad yield map generation time for {} pads: {:.2f} seconds (legacy)".format(
            #         PAD_ARR_ROW * PAD_ARR_COL,
            #         time.perf_counter() - start_time,
            #     )
            # )
        glb_defect_pad_yield_min = min(glb_defect_pad_yield_min, np.nanmin(overlay_pad_yield_map_sub))
        glb_defect_pad_yield_max = max(glb_defect_pad_yield_max, np.nanmax(overlay_pad_yield_map_sub))
        interface.glb_pad_yield_min_max_dict['Y_ovl'] = (glb_defect_pad_yield_min, glb_defect_pad_yield_max)

        if cfg.plot_flag:
        # Draw the pad yield map
            plt.figure(figsize=(8, 6))
            plt.imshow(
                overlay_pad_yield_map_sub,
                cmap='viridis',
                vmin=interface.glb_pad_yield_min_max_dict['Y_ovl'][0],
                vmax=interface.glb_pad_yield_min_max_dict['Y_ovl'][1],
                interpolation='nearest',
                )
            plt.colorbar(label='Pad Overlay Yield (Subsampled)')
            plt.xlabel('Pad Column Index')
            plt.ylabel('Pad Row Index')
            plt.show()

    return overlay_pad_yield_map_sub
