#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Oct 2, 2025

'''
This module contains functions to calculate die-level and pad-level defect-induced yield based on void size distribution and pad layout.
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.ndimage import distance_transform_edt


def _cfg_get(cfg, key, default=None):
    if hasattr(cfg, "get"):
        value = cfg.get(key, default)
    else:
        value = getattr(cfg, key, default)
    if value in (None, "None"):
        return default
    return value


def _cfg_float(cfg, key, default=None):
    value = _cfg_get(cfg, key, default)
    if value in (None, "None"):
        if default is None:
            raise ValueError(f"Missing required config value: {key}")
        value = default
    return float(value)


def _critical_bitmap_on_die_grid(cfg, critical_pad_bitmap):
    """
    Place the interface critical-pad bitmap on a die-sized grid.

    The grid pitch is the bonding pad pitch. Extra rows/columns are added around
    the pad array so particle centers outside the pad array but inside the die
    are still counted.
    """
    critical_pad_bitmap = np.asarray(critical_pad_bitmap, dtype=bool)
    pad_arr_row, pad_arr_col = critical_pad_bitmap.shape

    pitch_r_um = _cfg_float(cfg, "PITCH_r_um")
    pitch_c_um = _cfg_float(cfg, "PITCH_c_um")
    die_w_um = _cfg_float(cfg, "DIE_W_um")
    die_l_um = _cfg_float(cfg, "DIE_L_um")
    pad_arr_w_um = _cfg_float(cfg, "PAD_ARR_W_um", (pad_arr_col - 1) * pitch_c_um)
    pad_arr_l_um = _cfg_float(cfg, "PAD_ARR_L_um", (pad_arr_row - 1) * pitch_r_um)

    x_pad_min_um = -pad_arr_w_um / 2.0
    x_pad_max_um = x_pad_min_um + (pad_arr_col - 1) * pitch_c_um
    y_pad_max_um = pad_arr_l_um / 2.0
    y_pad_min_um = y_pad_max_um - (pad_arr_row - 1) * pitch_r_um

    n_left = max(0, int(np.ceil((x_pad_min_um + die_w_um / 2.0) / pitch_c_um)))
    n_right = max(0, int(np.ceil((die_w_um / 2.0 - x_pad_max_um) / pitch_c_um)))
    n_top = max(0, int(np.ceil((die_l_um / 2.0 - y_pad_max_um) / pitch_r_um)))
    n_bottom = max(0, int(np.ceil((y_pad_min_um + die_l_um / 2.0) / pitch_r_um)))

    grid = np.zeros(
        (pad_arr_row + n_top + n_bottom, pad_arr_col + n_left + n_right),
        dtype=bool,
    )
    grid[n_top:n_top + pad_arr_row, n_left:n_left + pad_arr_col] = critical_pad_bitmap

    x0_um = x_pad_min_um - n_left * pitch_c_um
    y0_um = y_pad_max_um + n_top * pitch_r_um
    x_coords_um = x0_um + np.arange(grid.shape[1]) * pitch_c_um
    y_coords_um = y0_um - np.arange(grid.shape[0]) * pitch_r_um
    return grid, x_coords_um, y_coords_um


def _distance_from_first_contact_um(cfg, xx_um, yy_um):
    first_contact = _cfg_get(cfg, "first_contact", "center")
    if first_contact == "center":
        return np.sqrt(xx_um**2 + yy_um**2)
    if first_contact == "vertical-edge":
        return np.abs(_cfg_float(cfg, "DIE_W_um") / 2.0 + xx_um)
    if first_contact == "horizontal-edge":
        return np.abs(_cfg_float(cfg, "DIE_L_um") / 2.0 + yy_um)
    if first_contact == "corner":
        return np.sqrt(
            (_cfg_float(cfg, "DIE_W_um") / 2.0 + xx_um) ** 2
            + (_cfg_float(cfg, "DIE_L_um") / 2.0 + yy_um) ** 2
        )
    raise ValueError(f"Unsupported first_contact mode: {first_contact}")


def _particle_density_map_um2(cfg, xx_um, yy_um):
    D0 = _cfg_float(cfg, "D0")
    D1 = _cfg_float(cfg, "D1", D0)
    density = np.full(xx_um.shape, D0, dtype=np.float64)
    edge_region_width_um = _cfg_float(cfg, "EDGE_REGION_WIDTH_um", 300.0)
    if D1 <= D0 or edge_region_width_um <= 0:
        return density

    die_w_um = _cfg_float(cfg, "DIE_W_um")
    die_l_um = _cfg_float(cfg, "DIE_L_um")
    edge_region_width_um = min(edge_region_width_um, die_w_um / 2.0, die_l_um / 2.0)
    if edge_region_width_um <= 0:
        return density

    dist_to_nearest_edge_um = np.minimum(
        die_w_um / 2.0 - np.abs(xx_um),
        die_l_um / 2.0 - np.abs(yy_um),
    )
    edge_weight = np.clip(
        1.0 - dist_to_nearest_edge_um / edge_region_width_um,
        0.0,
        1.0,
    )
    return density + (D1 - D0) * edge_weight


def _main_void_fatal_probability_map(cfg, dist_to_critical_pad_um, xx_um, yy_um):
    """
    Probability that a particle at each grid point creates a main void large
    enough to overlap the nearest critical signal bump.
    """
    pad_top_r_um = _cfg_float(cfg, "PAD_TOP_R_um")
    k_r = _cfg_float(cfg, "k_r")
    k_r0 = _cfg_float(cfg, "k_r0")
    t_0 = _cfg_float(cfg, "t_0")
    z = _cfg_float(cfg, "z")
    if z <= 1.0:
        raise ValueError("Particle thickness exponent z must be greater than 1.")

    required_void_radius_um = np.maximum(dist_to_critical_pad_um - pad_top_r_um, 0.0)
    distance_to_contact_um = _distance_from_first_contact_um(cfg, xx_um, yy_um)
    radius_scale = k_r * distance_to_contact_um + k_r0
    if np.any(radius_scale <= 0):
        raise ValueError("Main void radius scale k_r * L + k_r0 must be positive.")

    required_thickness_um = (required_void_radius_um / radius_scale) ** 2
    fatal_probability = np.ones_like(required_thickness_um, dtype=np.float64)
    mask = required_thickness_um > t_0
    fatal_probability[mask] = (t_0 / required_thickness_um[mask]) ** (z - 1.0)
    return np.clip(fatal_probability, 0.0, 1.0)


def _particle_density_chunk_um2(cfg, x_coords_um, y_coords_um):
    D0 = _cfg_float(cfg, "D0")
    D1 = _cfg_float(cfg, "D1", D0)
    density = np.full(
        (y_coords_um.shape[0], x_coords_um.shape[0]),
        D0,
        dtype=np.float64,
    )
    edge_region_width_um = _cfg_float(cfg, "EDGE_REGION_WIDTH_um", 300.0)
    if D1 <= D0 or edge_region_width_um <= 0:
        return density

    die_w_um = _cfg_float(cfg, "DIE_W_um")
    die_l_um = _cfg_float(cfg, "DIE_L_um")
    edge_region_width_um = min(edge_region_width_um, die_w_um / 2.0, die_l_um / 2.0)
    if edge_region_width_um <= 0:
        return density

    dist_to_nearest_edge_um = np.minimum(
        die_w_um / 2.0 - np.abs(x_coords_um)[None, :],
        die_l_um / 2.0 - np.abs(y_coords_um)[:, None],
    )
    edge_weight = np.clip(
        1.0 - dist_to_nearest_edge_um / edge_region_width_um,
        0.0,
        1.0,
    )
    density += (D1 - D0) * edge_weight
    return density


def _distance_from_first_contact_chunk_um(cfg, x_coords_um, y_coords_um):
    first_contact = _cfg_get(cfg, "first_contact", "center")
    if first_contact == "center":
        return np.sqrt(x_coords_um[None, :] ** 2 + y_coords_um[:, None] ** 2)
    if first_contact == "vertical-edge":
        return np.broadcast_to(
            np.abs(_cfg_float(cfg, "DIE_W_um") / 2.0 + x_coords_um)[None, :],
            (y_coords_um.shape[0], x_coords_um.shape[0]),
        )
    if first_contact == "horizontal-edge":
        return np.broadcast_to(
            np.abs(_cfg_float(cfg, "DIE_L_um") / 2.0 + y_coords_um)[:, None],
            (y_coords_um.shape[0], x_coords_um.shape[0]),
        )
    if first_contact == "corner":
        return np.sqrt(
            (_cfg_float(cfg, "DIE_W_um") / 2.0 + x_coords_um)[None, :] ** 2
            + (_cfg_float(cfg, "DIE_L_um") / 2.0 + y_coords_um)[:, None] ** 2
        )
    raise ValueError(f"Unsupported first_contact mode: {first_contact}")


def _die_mask_chunk(cfg, x_coords_um, y_coords_um):
    return (
        (np.abs(x_coords_um)[None, :] <= _cfg_float(cfg, "DIE_W_um") / 2.0 + 1e-9)
        & (np.abs(y_coords_um)[:, None] <= _cfg_float(cfg, "DIE_L_um") / 2.0 + 1e-9)
    )


def _fatal_particle_integral_from_distance_map(
    cfg,
    dist_to_critical_pad_um,
    x_coords_um,
    y_coords_um,
):
    """
    Chunked evaluation of sum D(x) P_fatal(x) dA.

    The distance map is the only large float64 image retained. All coordinate,
    density, and fatal-probability arrays are created per row chunk so very large
    layouts, such as 10000 x 10000 grids, do not require multiple full-size
    temporary arrays.
    """
    pitch_r_um = _cfg_float(cfg, "PITCH_r_um")
    pitch_c_um = _cfg_float(cfg, "PITCH_c_um")
    cell_area_um2 = pitch_r_um * pitch_c_um
    pad_top_r_um = _cfg_float(cfg, "PAD_TOP_R_um")
    k_r = _cfg_float(cfg, "k_r")
    k_r0 = _cfg_float(cfg, "k_r0")
    t_0 = _cfg_float(cfg, "t_0")
    z = _cfg_float(cfg, "z")
    if z <= 1.0:
        raise ValueError("Particle thickness exponent z must be greater than 1.")

    chunk_rows = int(_cfg_float(cfg, "DEFECT_CALC_CHUNK_ROWS", 512))
    chunk_rows = max(1, min(chunk_rows, dist_to_critical_pad_um.shape[0]))
    avg_fatal_particles = 0.0

    for row_start in range(0, dist_to_critical_pad_um.shape[0], chunk_rows):
        row_end = min(row_start + chunk_rows, dist_to_critical_pad_um.shape[0])
        y_chunk_um = y_coords_um[row_start:row_end]
        dist_chunk_um = dist_to_critical_pad_um[row_start:row_end, :]

        required_void_radius_um = np.maximum(dist_chunk_um - pad_top_r_um, 0.0)
        distance_to_contact_um = _distance_from_first_contact_chunk_um(
            cfg,
            x_coords_um,
            y_chunk_um,
        )
        radius_scale = k_r * distance_to_contact_um + k_r0
        if np.any(radius_scale <= 0):
            raise ValueError("Main void radius scale k_r * L + k_r0 must be positive.")

        required_thickness_um = (required_void_radius_um / radius_scale) ** 2
        fatal_probability = np.ones_like(required_thickness_um, dtype=np.float64)
        needs_large_particle = required_thickness_um > t_0
        fatal_probability[needs_large_particle] = (
            t_0 / required_thickness_um[needs_large_particle]
        ) ** (z - 1.0)

        density = _particle_density_chunk_um2(cfg, x_coords_um, y_chunk_um)
        die_mask = _die_mask_chunk(cfg, x_coords_um, y_chunk_um)
        avg_fatal_particles += float(
            np.sum(density * fatal_probability * die_mask) * cell_area_um2
        )

    return avg_fatal_particles, chunk_rows


def interface_particle_yield_from_critical_bitmap(cfg, pad_bitmap_collection):
    """
    Calculate particle yield from the critical signal bump placement.

    This is a bitmap/dilation-equivalent critical-area calculation for the
    no-redundancy experiments. Non-critical, power/ground, and dummy bumps are
    ignored by this particle-fatal criterion.
    """
    critical_pad_bitmap = np.asarray(
        pad_bitmap_collection["CRITICAL_PAD_BITMAP"],
        dtype=bool,
    )
    if not np.any(critical_pad_bitmap):
        return 1.0, {
            "avg_fatal_particles": 0.0,
            "effective_critical_area_um2": 0.0,
            "num_critical_pads": 0,
        }

    critical_grid, x_coords_um, y_coords_um = _critical_bitmap_on_die_grid(
        cfg,
        critical_pad_bitmap,
    )
    pitch_r_um = _cfg_float(cfg, "PITCH_r_um")
    pitch_c_um = _cfg_float(cfg, "PITCH_c_um")

    dist_to_critical_pad_um = distance_transform_edt(
        ~critical_grid,
        sampling=(pitch_r_um, pitch_c_um),
    )
    grid_shape = critical_grid.shape
    del critical_grid

    avg_fatal_particles, chunk_rows = _fatal_particle_integral_from_distance_map(
        cfg,
        dist_to_critical_pad_um,
        x_coords_um,
        y_coords_um,
    )
    particle_yield = float(np.exp(-avg_fatal_particles))
    D0 = _cfg_float(cfg, "D0")
    effective_critical_area_um2 = (
        avg_fatal_particles / D0 if D0 > 0.0 else 0.0
    )

    info = {
        "avg_fatal_particles": avg_fatal_particles,
        "effective_critical_area_um2": float(effective_critical_area_um2),
        "num_critical_pads": int(np.sum(critical_pad_bitmap)),
        "grid_shape": tuple(int(v) for v in grid_shape),
        "chunk_rows": int(chunk_rows),
        "method": "critical_bitmap_distance_transform_chunked",
    }
    return particle_yield, info


def stack_defect_yield_calculator(
    cfg_dict: dict,
    die_stack,
):
    """
    Calculate D2W particle yield for every interface and write into die_stack.
    """
    for interface_name, cfg in cfg_dict.items():
        pad_bitmap_collection = die_stack.interfaces.pad_bitmap_collection_dict[interface_name]
        particle_yield, info = interface_particle_yield_from_critical_bitmap(
            cfg,
            pad_bitmap_collection,
        )
        die_stack.die_yield_per_interface_dict[interface_name]['particle'] = particle_yield 