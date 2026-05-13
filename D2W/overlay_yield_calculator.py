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
import os
from scipy.optimize import fsolve
import sympy as sp
from scipy.stats import norm

try:
    from warpage_yield_calculator import get_interface_stack_warpage_map
except ModuleNotFoundError:
    from D2W.warpage_yield_calculator import get_interface_stack_warpage_map

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

# def overlay_yield_calculator(*,
#     cfg,
#     PAD_TOP_R_um: float,
#     PAD_BOT_R_um: float,
#     PAD_ARR_ROW: int,
#     PAD_ARR_COL: int,
#     PITCH_r_um: float,
#     PITCH_c_um: float,
#     num_samples: int,
#     CONTACT_AREA_CONSTRAINT: float,
#     CRITICAL_DIST_CONSTRAINT: float,
#     SYSTEM_MAGNIFICATION_MEAN_ppm: float,
#     SYSTEM_MAGNIFICATION_STD_ppm: float,
#     SYSTEM_ROTATION_MEAN_rad: float,
#     SYSTEM_ROTATION_STD_rad: float,
#     SYSTEM_TRANSLATION_X_MEAN_um: float,
#     SYSTEM_TRANSLATION_X_STD_um: float,
#     SYSTEM_TRANSLATION_Y_MEAN_um: float,
#     SYSTEM_TRANSLATION_Y_STD_um: float,
#     RANDOM_MISALIGNMENT_MEAN_um: float,
#     RANDOM_MISALIGNMENT_STD_um: float,
#     interface,
#     redundant_flag: bool,
# ):
#     MAX_ALLOWED_MISALIGNMENT = max_allowed_misalignment_calculator(
#         cfg=cfg,
#         PAD_TOP_R_um=PAD_TOP_R_um,
#         PAD_BOT_R_um=PAD_BOT_R_um,
#         PITCH_r_um=PITCH_r_um,
#         PITCH_c_um=PITCH_c_um,
#         CONTACT_AREA_CONSTRAINT=CONTACT_AREA_CONSTRAINT,
#         CRITICAL_DIST_CONSTRAINT=CRITICAL_DIST_CONSTRAINT,
#     )
#     num_samples = num_samples
#     system_translation_x_samples_um = np.random.normal(SYSTEM_TRANSLATION_X_MEAN_um, SYSTEM_TRANSLATION_X_STD_um, num_samples)
#     system_translation_y_samples_um = np.random.normal(SYSTEM_TRANSLATION_Y_MEAN_um, SYSTEM_TRANSLATION_Y_STD_um, num_samples)
#     system_rotation_samples_rad = np.random.normal(SYSTEM_ROTATION_MEAN_rad, SYSTEM_ROTATION_STD_rad, num_samples)
#     system_magnification_samples = np.random.normal(SYSTEM_MAGNIFICATION_MEAN_ppm, SYSTEM_MAGNIFICATION_STD_ppm, num_samples)
#     # print("system_translation_x_samples_um contribution", system_translation_x_samples_um.mean()*1e3, " nm")
#     # print("system_translation_y_samples_um contribution", system_translation_y_samples_um.mean()*1e3, " nm")
#     # print("system_rotation_samples_rad contribution", system_rotation_samples_rad.mean() * np.sqrt(die.DIE_W_um**2 + die.DIE_L_um**2) * 1e3, " nm")
#     # print("system_magnification_samples contribution", system_magnification_samples.mean() * np.sqrt(die.DIE_W_um**2 + die.DIE_L_um**2) * 1e3, " nm")

#     # Sample the systematic misalignment for corner pads based on the systematic translation, rotation, and magnification
#     # Calculate the die yield based on the worst-case pad misalignment
#     if redundant_flag == True:
#         far_dx_samples_0 = (system_translation_x_samples_um - system_rotation_samples_rad * interface.ovl_critical_pad_boundary_coords[0, 1] + system_magnification_samples * interface.ovl_critical_pad_boundary_coords[0, 0])
#         far_dy_samples_0 = (system_translation_y_samples_um + system_rotation_samples_rad * interface.ovl_critical_pad_boundary_coords[0, 0] + system_magnification_samples * interface.ovl_critical_pad_boundary_coords[0, 1])
#         far_dx_samples_1 = (system_translation_x_samples_um - system_rotation_samples_rad * interface.ovl_critical_pad_boundary_coords[1, 1] + system_magnification_samples * interface.ovl_critical_pad_boundary_coords[1, 0])
#         far_dy_samples_1 = (system_translation_y_samples_um + system_rotation_samples_rad * interface.ovl_critical_pad_boundary_coords[1, 0] + system_magnification_samples * interface.ovl_critical_pad_boundary_coords[1, 1])
#         far_dx_samples_2 = (system_translation_x_samples_um - system_rotation_samples_rad * interface.ovl_critical_pad_boundary_coords[2, 1] + system_magnification_samples * interface.ovl_critical_pad_boundary_coords[2, 0])
#         far_dy_samples_2 = (system_translation_y_samples_um + system_rotation_samples_rad * interface.ovl_critical_pad_boundary_coords[2, 0] + system_magnification_samples * interface.ovl_critical_pad_boundary_coords[2, 1])
#         far_dx_samples_3 = (system_translation_x_samples_um - system_rotation_samples_rad * interface.ovl_critical_pad_boundary_coords[3, 1] + system_magnification_samples * interface.ovl_critical_pad_boundary_coords[3, 0])
#         far_dy_samples_3 = (system_translation_y_samples_um + system_rotation_samples_rad * interface.ovl_critical_pad_boundary_coords[3, 0] + system_magnification_samples * interface.ovl_critical_pad_boundary_coords[3, 1])
#     else:
#         far_dx_samples_0 = (system_translation_x_samples_um - system_rotation_samples_rad * interface.pad_array_box[0, 1] + system_magnification_samples * interface.pad_array_box[0, 0])
#         far_dy_samples_0 = (system_translation_y_samples_um + system_rotation_samples_rad * interface.pad_array_box[0, 0] + system_magnification_samples * interface.pad_array_box[0, 1])
#         far_dx_samples_1 = (system_translation_x_samples_um - system_rotation_samples_rad * interface.pad_array_box[1, 1] + system_magnification_samples * interface.pad_array_box[1, 0])
#         far_dy_samples_1 = (system_translation_y_samples_um + system_rotation_samples_rad * interface.pad_array_box[1, 0] + system_magnification_samples * interface.pad_array_box[1, 1])
#         far_dx_samples_2 = (system_translation_x_samples_um - system_rotation_samples_rad * interface.pad_array_box[2, 1] + system_magnification_samples * interface.pad_array_box[2, 0])
#         far_dy_samples_2 = (system_translation_y_samples_um + system_rotation_samples_rad * interface.pad_array_box[2, 0] + system_magnification_samples * interface.pad_array_box[2, 1])
#         far_dx_samples_3 = (system_translation_x_samples_um - system_rotation_samples_rad * interface.pad_array_box[3, 1] + system_magnification_samples * interface.pad_array_box[3, 0])
#         far_dy_samples_3 = (system_translation_y_samples_um + system_rotation_samples_rad * interface.pad_array_box[3, 0] + system_magnification_samples * interface.pad_array_box[3, 1])
#     far_pad_misalignment_samples_0 = np.sqrt(far_dx_samples_0**2 + far_dy_samples_0**2)
#     far_pad_misalignment_samples_1 = np.sqrt(far_dx_samples_1**2 + far_dy_samples_1**2)
#     far_pad_misalignment_samples_2 = np.sqrt(far_dx_samples_2**2 + far_dy_samples_2**2)
#     far_pad_misalignment_samples_3 = np.sqrt(far_dx_samples_3**2 + far_dy_samples_3**2)

#     upper_limit_0 = MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_0
#     lower_limit_0 = -MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_0
#     upper_limit_1 = MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_1
#     lower_limit_1 = -MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_1
#     upper_limit_2 = MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_2
#     lower_limit_2 = -MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_2
#     upper_limit_3 = MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_3
#     lower_limit_3 = -MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_3

#     overlay_die_yield_0 = np.mean(norm.cdf(upper_limit_0, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um) \
#                         - norm.cdf(lower_limit_0, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um))
#     overlay_die_yield_1 = np.mean(norm.cdf(upper_limit_1, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um) \
#                         - norm.cdf(lower_limit_1, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um))
#     overlay_die_yield_2 = np.mean(norm.cdf(upper_limit_2, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um) \
#                         - norm.cdf(lower_limit_2, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um))
#     overlay_die_yield_3 = np.mean(norm.cdf(upper_limit_3, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um) \
#                         - norm.cdf(lower_limit_3, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um))
#     overlay_die_yield = min(overlay_die_yield_0, overlay_die_yield_1, overlay_die_yield_2, overlay_die_yield_3)


#     return overlay_die_yield


def _cfg_float(cfg, key, default=0.0):
    value = cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)
    if value in (None, "None"):
        return float(default)
    return float(value)


def _interface_bow_difference_stats(cfg_dict, _3dbx_path=None):
    """
    Return Gaussian bow-difference stats for every interface.

    bow_difference = incoming_top_die_initial_bow - existing_substack_warpage
    """
    if not _3dbx_path or not os.path.exists(_3dbx_path):
        raise FileNotFoundError(
            "D2W overlay bow-difference modeling requires generated_stack_config.3dbx. "
            f"Received: {_3dbx_path}"
        )

    interface_stack_warpage = get_interface_stack_warpage_map(cfg_dict, _3dbx_path)
    bow_difference_stats = {}
    for interface_name, cfg in cfg_dict.items():
        if interface_name not in interface_stack_warpage:
            raise KeyError(
                f"Interface '{interface_name}' is missing from stack warpage map. "
                "D2W overlay now derives bow difference from TOP/BOT initial bow "
                "and substack warpage; legacy config fallback has been removed."
            )

        stats = interface_stack_warpage[interface_name]
        top_mu_um = float(stats["top_die_mu_um"])
        top_sigma_um = max(float(stats["top_die_sigma_um"]), 0.0)
        stack_mu_um = float(stats["mu_um"])
        stack_sigma_um = max(float(stats["sigma_um"]), 0.0)
        bow_difference_stats[interface_name] = {
            "bow_difference_mean_um": top_mu_um - stack_mu_um,
            "bow_difference_std_um": float(
                np.sqrt(top_sigma_um**2 + stack_sigma_um**2)
            ),
            "top_die_mean_um": top_mu_um,
            "top_die_std_um": top_sigma_um,
            "existing_stack_mean_um": stack_mu_um,
            "existing_stack_std_um": stack_sigma_um,
            "source": "substack_warpage_model",
        }

    return bow_difference_stats


def stack_overlay_yield_calculator(
    cfg_dict: dict,
    die_stack,
    _3dbx_path: str = None,
):
    """
    Calculate D2W stack-level overlay yield from interface-level overlay yields.

    The current D2W interface model represents one bonding interface in the
    stack. We evaluate the overlay yield for each interface and multiply the
    interface yields to obtain the stack overlay yield.
    """
    interface_overlay_yield_dict = {}
    bow_difference_stats = _interface_bow_difference_stats(cfg_dict, _3dbx_path)

    for interface_name, cfg in cfg_dict.items():
        interface = die_stack.interfaces.interface_dict[interface_name]
        max_allowed_misalignment_um = max_allowed_misalignment_calculator(
            cfg=cfg,
            PAD_TOP_R_um=cfg.PAD_TOP_R_um,
            PAD_BOT_R_um=cfg.PAD_BOT_R_um,
            PITCH_r_um=cfg.PITCH_r_um,
            PITCH_c_um=cfg.PITCH_c_um,
            CONTACT_AREA_CONSTRAINT=cfg.CONTACT_AREA_CONSTRAINT,
            CRITICAL_DIST_CONSTRAINT=cfg.CRITICAL_DIST_CONSTRAINT,
        )

        boundary_coords = getattr(interface, "ovl_critical_pad_boundary_coords", None)
        if boundary_coords is None:
            boundary_coords = interface.pad_array_box
        boundary_coords = np.asarray(boundary_coords, dtype=np.float64)

        num_samples = int(cfg.num_samples)
        system_translation_x_samples_um = np.random.normal(cfg.SYSTEM_TRANSLATION_X_MEAN_um, cfg.SYSTEM_TRANSLATION_X_STD_um, num_samples)
        system_translation_y_samples_um = np.random.normal(cfg.SYSTEM_TRANSLATION_Y_MEAN_um, cfg.SYSTEM_TRANSLATION_Y_STD_um, num_samples)
        system_rotation_samples_rad = np.random.normal(cfg.SYSTEM_ROTATION_MEAN_rad, cfg.SYSTEM_ROTATION_STD_rad, num_samples)
        interface_bow_stats = bow_difference_stats[interface_name]
        magnification_mean = (
            _cfg_float(cfg, "k_mag", 0.0) * interface_bow_stats["bow_difference_mean_um"] + _cfg_float(cfg, "M_0", 0.0)
        ) / 1e6
        magnification_sigma = (abs(_cfg_float(cfg, "k_mag", 0.0)) * interface_bow_stats["bow_difference_std_um"]
        ) / 1e6
        system_magnification_samples_ppm = np.random.normal(magnification_mean, magnification_sigma, num_samples)

        dx_samples = (
            system_translation_x_samples_um[:, None]
            - system_rotation_samples_rad[:, None] * boundary_coords[None, :, 1]
            + system_magnification_samples_ppm[:, None] * boundary_coords[None, :, 0]
        )
        dy_samples = (
            system_translation_y_samples_um[:, None]
            + system_rotation_samples_rad[:, None] * boundary_coords[None, :, 0]
            + system_magnification_samples_ppm[:, None] * boundary_coords[None, :, 1]
        )
        pad_misalignment_samples = np.sqrt(dx_samples**2 + dy_samples**2)

        upper_limits = max_allowed_misalignment_um - pad_misalignment_samples
        lower_limits = -max_allowed_misalignment_um - pad_misalignment_samples
        corner_yields = np.mean(
            norm.cdf(
                upper_limits, loc=cfg.RANDOM_MISALIGNMENT_MEAN_um, scale=cfg.RANDOM_MISALIGNMENT_STD_um,
            )
            - norm.cdf(
                lower_limits, loc=cfg.RANDOM_MISALIGNMENT_MEAN_um, scale=cfg.RANDOM_MISALIGNMENT_STD_um,
            ),
            axis=0,
        )
        interface_overlay_yield = float(np.min(corner_yields))
        die_stack.die_yield_per_interface_dict[interface_name]['overlay'] = interface_overlay_yield
