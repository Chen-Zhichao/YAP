#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Author: Zhichao Chen
#### Date: Oct 8, 2025

"""
Defect yield calculator for the W2W hybrid bonding process.
- Calculate the critical area of the voids and the defects regarding the die and the pad.
- Calculate the defect yield of on die-level and pad-level.
"""

import numpy as np
import sympy as sp
import math
import os
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


def _cfg_bool(cfg, key, default=False):
    value = _cfg_get(cfg, key, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y")
    return bool(value)


def _critical_bitmap_on_die_grid(cfg, critical_pad_bitmap):
    """
    Place the interface critical-pad bitmap on a die-sized pitch grid.

    The pad bitmap covers only the pad array. The returned grid also includes
    the die margin around the pad array, so particles outside the pad array but
    inside the die are counted in the fatal-particle integral.
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


def _die_mask_chunk(cfg, x_coords_um, y_coords_um):
    return (
        (np.abs(x_coords_um)[None, :] <= _cfg_float(cfg, "DIE_W_um") / 2.0 + 1e-9)
        & (np.abs(y_coords_um)[:, None] <= _cfg_float(cfg, "DIE_L_um") / 2.0 + 1e-9)
    )


def _particle_density_chunk_um2(cfg, x_coords_um, y_coords_um):
    """
    Die-local particle density with optional edge-density enhancement.

    D0 is the baseline density. D1 is the density at the die edge, linearly
    tapering back to D0 over EDGE_REGION_WIDTH_um. If D1 is absent, this
    reduces to a uniform D0 density.
    """
    D0 = _cfg_float(cfg, "D0")
    D1 = _cfg_float(cfg, "D1", D0)
    density = np.full(
        (y_coords_um.shape[0], x_coords_um.shape[0]),
        D0,
        dtype=np.float64,
    )
    edge_region_width_um = _cfg_float(cfg, "EDGE_REGION_WIDTH_um", 300.0)
    if D1 <= D0 or edge_region_width_um <= 0.0:
        return density

    die_w_um = _cfg_float(cfg, "DIE_W_um")
    die_l_um = _cfg_float(cfg, "DIE_L_um")
    edge_region_width_um = min(edge_region_width_um, die_w_um / 2.0, die_l_um / 2.0)
    if edge_region_width_um <= 0.0:
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


def _fatal_probability_from_required_radius(
    required_void_radius_um,
    radius_scale_um,
    t_0,
    z,
):
    """
    P(particle thickness is large enough for the required main-void radius).
    """
    fatal_probability = np.zeros_like(required_void_radius_um, dtype=np.float64)
    overlaps_without_growth = required_void_radius_um <= 0.0
    fatal_probability[overlaps_without_growth] = 1.0

    if np.isscalar(radius_scale_um):
        if radius_scale_um <= 0.0:
            return fatal_probability
        required_thickness_um = (required_void_radius_um / radius_scale_um) ** 2
        positive_scale = np.ones_like(required_void_radius_um, dtype=bool)
    else:
        positive_scale = radius_scale_um > 0.0
        required_thickness_um = np.full_like(required_void_radius_um, np.inf)
        required_thickness_um[positive_scale] = (
            required_void_radius_um[positive_scale] / radius_scale_um[positive_scale]
        ) ** 2

    automatic = positive_scale & (required_thickness_um <= t_0)
    fatal_probability[automatic] = 1.0

    needs_large_particle = positive_scale & (required_thickness_um > t_0)
    fatal_probability[needs_large_particle] = (
        t_0 / required_thickness_um[needs_large_particle]
    ) ** (z - 1.0)
    return np.clip(fatal_probability, 0.0, 1.0)


def _fatal_particle_integral_from_distance_map(
    cfg,
    dist_to_critical_pad_um,
    x_coords_um,
    y_coords_um,
    die_center_um=None,
    die_center_radius_um=None,
):
    """
    Chunked evaluation of the W2W main-void fatal-particle integral.

    If die_center_radius_um is provided, the main-void radius scale uses the
    die-center wafer radius. This matches the older W2W analytical model and
    enables radial-layer caching. If die_center_um is provided instead, the
    radius scale is evaluated at each particle-center location.
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

    if die_center_radius_um is None and die_center_um is None:
        raise ValueError("Provide either die_center_radius_um or die_center_um.")

    chunk_rows = int(_cfg_float(cfg, "DEFECT_CALC_CHUNK_ROWS", 512))
    chunk_rows = max(1, min(chunk_rows, dist_to_critical_pad_um.shape[0]))
    avg_fatal_particles = 0.0

    if die_center_radius_um is not None:
        radius_scale_um = k_r * float(die_center_radius_um) + k_r0
    else:
        die_center_um = np.asarray(die_center_um, dtype=np.float64)
        radius_scale_um = None

    for row_start in range(0, dist_to_critical_pad_um.shape[0], chunk_rows):
        row_end = min(row_start + chunk_rows, dist_to_critical_pad_um.shape[0])
        y_chunk_um = y_coords_um[row_start:row_end]
        dist_chunk_um = dist_to_critical_pad_um[row_start:row_end, :]
        required_void_radius_um = np.maximum(dist_chunk_um - pad_top_r_um, 0.0)

        if radius_scale_um is None:
            wafer_radius_chunk_um = np.sqrt(
                (die_center_um[0] + x_coords_um)[None, :] ** 2
                + (die_center_um[1] + y_chunk_um)[:, None] ** 2
            )
            chunk_radius_scale_um = k_r * wafer_radius_chunk_um + k_r0
        else:
            chunk_radius_scale_um = radius_scale_um

        fatal_probability = _fatal_probability_from_required_radius(
            required_void_radius_um,
            chunk_radius_scale_um,
            t_0,
            z,
        )
        density = _particle_density_chunk_um2(cfg, x_coords_um, y_chunk_um)
        die_mask = _die_mask_chunk(cfg, x_coords_um, y_chunk_um)
        avg_fatal_particles += float(
            np.sum(density * fatal_probability * die_mask) * cell_area_um2
        )

    return avg_fatal_particles, chunk_rows


def _model_grid_coords(cfg, grid_pitch_um, margin_um=0.0):
    die_w_um = _cfg_float(cfg, "DIE_W_um")
    die_l_um = _cfg_float(cfg, "DIE_L_um")
    x_min_um = -die_w_um / 2.0 - margin_um
    x_max_um = die_w_um / 2.0 + margin_um
    y_min_um = -die_l_um / 2.0 - margin_um
    y_max_um = die_l_um / 2.0 + margin_um
    n_cols = max(2, int(np.ceil((x_max_um - x_min_um) / grid_pitch_um)) + 1)
    n_rows = max(2, int(np.ceil((y_max_um - y_min_um) / grid_pitch_um)) + 1)
    x_coords_um = np.linspace(x_min_um, x_max_um, n_cols)
    y_coords_um = np.linspace(y_max_um, y_min_um, n_rows)
    return x_coords_um, y_coords_um


def _critical_pad_centers_from_bitmap(cfg, critical_pad_bitmap):
    critical_rows, critical_cols = np.where(np.asarray(critical_pad_bitmap, dtype=bool))
    if critical_rows.size == 0:
        return np.zeros((0, 2), dtype=np.float64)

    pitch_r_um = _cfg_float(cfg, "PITCH_r_um")
    pitch_c_um = _cfg_float(cfg, "PITCH_c_um")
    pad_arr_row, pad_arr_col = critical_pad_bitmap.shape
    pad_arr_w_um = _cfg_float(cfg, "PAD_ARR_W_um", (pad_arr_col - 1) * pitch_c_um)
    pad_arr_l_um = _cfg_float(cfg, "PAD_ARR_L_um", (pad_arr_row - 1) * pitch_r_um)

    x_pad_min_um = -pad_arr_w_um / 2.0
    y_pad_max_um = pad_arr_l_um / 2.0
    x_um = x_pad_min_um + critical_cols * pitch_c_um
    y_um = y_pad_max_um - critical_rows * pitch_r_um
    return np.column_stack((x_um, y_um)).astype(np.float64)


def _critical_region_on_model_grid(cfg, critical_pad_bitmap, grid_pitch_um, margin_um=0.0):
    """
    Rasterize critical pads on a coarser modeling grid for tail dilation.
    """
    critical_pad_bitmap = np.asarray(critical_pad_bitmap, dtype=bool)
    x_coords_um, y_coords_um = _model_grid_coords(cfg, grid_pitch_um, margin_um=margin_um)
    shape = (y_coords_um.shape[0], x_coords_um.shape[0])
    if not np.any(critical_pad_bitmap):
        return np.zeros(shape, dtype=bool), np.full(shape, np.inf), x_coords_um, y_coords_um

    critical_region_mask = np.zeros(shape, dtype=bool)
    pitch_r_um = _cfg_float(cfg, "PITCH_r_um")
    pitch_c_um = _cfg_float(cfg, "PITCH_c_um")
    pad_arr_row, pad_arr_col = critical_pad_bitmap.shape
    pad_arr_w_um = _cfg_float(cfg, "PAD_ARR_W_um", (pad_arr_col - 1) * pitch_c_um)
    pad_arr_l_um = _cfg_float(cfg, "PAD_ARR_L_um", (pad_arr_row - 1) * pitch_r_um)
    x_pad_min_um = -pad_arr_w_um / 2.0
    y_pad_max_um = pad_arr_l_um / 2.0
    dx_um = abs(x_coords_um[1] - x_coords_um[0])
    dy_um = abs(y_coords_um[0] - y_coords_um[1])

    x_left_edges_um = x_coords_um - dx_um / 2.0
    x_right_edges_um = x_coords_um + dx_um / 2.0
    y_top_edges_um = y_coords_um + dy_um / 2.0
    y_bottom_edges_um = y_coords_um - dy_um / 2.0

    for row_idx, (y_top_um, y_bottom_um) in enumerate(zip(y_top_edges_um, y_bottom_edges_um)):
        pad_row_start = int(np.ceil((y_pad_max_um - y_top_um) / pitch_r_um))
        pad_row_end = int(np.floor((y_pad_max_um - y_bottom_um) / pitch_r_um))
        pad_row_start = max(0, pad_row_start)
        pad_row_end = min(pad_arr_row - 1, pad_row_end)
        if pad_row_start > pad_row_end:
            continue
        row_slice = critical_pad_bitmap[pad_row_start:pad_row_end + 1, :]
        for col_idx, (x_left_um, x_right_um) in enumerate(zip(x_left_edges_um, x_right_edges_um)):
            pad_col_start = int(np.ceil((x_left_um - x_pad_min_um) / pitch_c_um))
            pad_col_end = int(np.floor((x_right_um - x_pad_min_um) / pitch_c_um))
            pad_col_start = max(0, pad_col_start)
            pad_col_end = min(pad_arr_col - 1, pad_col_end)
            if pad_col_start > pad_col_end:
                continue
            critical_region_mask[row_idx, col_idx] = np.any(
                row_slice[:, pad_col_start:pad_col_end + 1]
            )

    pad_top_r_um = _cfg_float(cfg, "PAD_TOP_R_um")
    if pad_top_r_um > min(
        dx_um,
        dy_um,
    ) * 0.5:
        center_cell_dist_um = distance_transform_edt(
            ~critical_region_mask,
            sampling=(
                abs(y_coords_um[0] - y_coords_um[1]),
                abs(x_coords_um[1] - x_coords_um[0]),
            ),
        )
        critical_region_mask = center_cell_dist_um <= pad_top_r_um
    dist_to_center_um = distance_transform_edt(
        ~critical_region_mask,
        sampling=(dy_um, dx_um),
    )
    return critical_region_mask, dist_to_center_um, x_coords_um, y_coords_um


def _line_offsets_for_direction(length_um, theta_rad, pitch_r_um, pitch_c_um):
    if length_um <= 0.0:
        return [(0, 0)]

    step_um = max(min(pitch_r_um, pitch_c_um) * 0.5, 1e-9)
    n_steps = max(1, int(np.ceil(length_um / step_um)))
    s_um = np.linspace(0.0, length_um, n_steps + 1)
    col_offsets = np.rint((s_um * np.cos(theta_rad)) / pitch_c_um).astype(np.int64)
    row_offsets = np.rint(-(s_um * np.sin(theta_rad)) / pitch_r_um).astype(np.int64)
    offsets = np.column_stack((row_offsets, col_offsets))
    offsets = np.unique(offsets, axis=0)
    return [(int(row), int(col)) for row, col in offsets]


def _directed_line_dilation(mask, offsets):
    """
    Directed dilation for particle centers.

    output[p] is true when mask[p + offset] is true for at least one directed
    line offset. This is equivalent to dilating the critical region by the
    reversed directed line segment.
    """
    n_rows, n_cols = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for row_offset, col_offset in offsets:
        if row_offset >= 0:
            src_r = slice(row_offset, n_rows)
            dst_r = slice(0, n_rows - row_offset)
        else:
            src_r = slice(0, n_rows + row_offset)
            dst_r = slice(-row_offset, n_rows)

        if col_offset >= 0:
            src_c = slice(col_offset, n_cols)
            dst_c = slice(0, n_cols - col_offset)
        else:
            src_c = slice(0, n_cols + col_offset)
            dst_c = slice(-col_offset, n_cols)

        if (
            (src_r.stop - src_r.start) <= 0
            or (src_c.stop - src_c.start) <= 0
        ):
            continue
        out[dst_r, dst_c] |= mask[src_r, src_c]
    return out


def _particle_thickness_quadrature(cfg):
    n_nodes = int(_cfg_float(cfg, "DEFECT_TAIL_THICKNESS_NODES", 12))
    n_nodes = max(1, n_nodes)
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    u = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights

    t_0 = _cfg_float(cfg, "t_0")
    z = _cfg_float(cfg, "z")
    if z <= 1.0:
        raise ValueError("Particle thickness exponent z must be greater than 1.")
    thickness_um = t_0 / (1.0 - u) ** (1.0 / (z - 1.0))
    return thickness_um, weights


def _tail_dilation_fatal_integral(
    cfg,
    critical_pad_bitmap,
    die_center_radius_um,
):
    """
    Direction-averaged main-void + directed-tail fatal-particle integral.
    """
    default_grid_pitch_um = max(
        400.0,
        _cfg_float(cfg, "PITCH_r_um"),
        _cfg_float(cfg, "PITCH_c_um"),
    )
    grid_pitch_um = _cfg_float(cfg, "DEFECT_TAIL_GRID_PITCH_um", default_grid_pitch_um)
    grid_pitch_um = max(grid_pitch_um, 1e-9)

    pad_top_r_um = _cfg_float(cfg, "PAD_TOP_R_um")
    radius_scale_um = _cfg_float(cfg, "k_r") * die_center_radius_um + _cfg_float(cfg, "k_r0")
    tail_scale_um = _cfg_float(cfg, "k_L", 0.0) * die_center_radius_um

    n_angles = int(_cfg_float(cfg, "DEFECT_TAIL_NUM_ANGLES", 16))
    n_angles = max(1, n_angles)
    angles_rad = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    thickness_um, thickness_weights = _particle_thickness_quadrature(cfg)
    length_cap_default_um = tail_scale_um * np.sqrt(float(np.max(thickness_um)))
    length_cap_um = _cfg_float(cfg, "DEFECT_TAIL_LENGTH_CAP_um", length_cap_default_um)
    max_main_radius_um = radius_scale_um * np.sqrt(float(np.max(thickness_um)))
    model_margin_um = max(length_cap_um, max_main_radius_um) + pad_top_r_um

    critical_region_mask, dist_to_critical_center_um, x_coords_um, y_coords_um = (
        _critical_region_on_model_grid(
            cfg,
            critical_pad_bitmap,
            grid_pitch_um,
            margin_um=model_margin_um,
        )
    )
    if not np.any(critical_region_mask):
        return 0.0, {
            "tail_grid_shape": tuple(int(v) for v in critical_region_mask.shape),
            "tail_grid_pitch_um": float(grid_pitch_um),
            "tail_model_margin_um": float(model_margin_um),
        }

    pitch_c_eff_um = float(abs(x_coords_um[1] - x_coords_um[0]))
    pitch_r_eff_um = float(abs(y_coords_um[0] - y_coords_um[1]))
    cell_area_um2 = pitch_r_eff_um * pitch_c_eff_um
    required_main_radius_um = np.maximum(
        dist_to_critical_center_um - pad_top_r_um,
        0.0,
    )

    density = _particle_density_chunk_um2(cfg, x_coords_um, y_coords_um)
    fatal_probability = np.zeros(critical_region_mask.shape, dtype=np.float64)

    for thickness, thickness_weight in zip(thickness_um, thickness_weights):
        sqrt_thickness = np.sqrt(float(thickness))
        main_radius_um = radius_scale_um * sqrt_thickness
        main_hit_mask = required_main_radius_um <= main_radius_um

        tail_length_um = min(tail_scale_um * sqrt_thickness, length_cap_um)
        if tail_length_um <= 0.0:
            fatal_probability += thickness_weight * main_hit_mask.astype(np.float64)
            continue

        tail_hit_count = np.zeros(critical_region_mask.shape, dtype=np.float64)
        for theta_rad in angles_rad:
            offsets = _line_offsets_for_direction(
                tail_length_um,
                theta_rad,
                pitch_r_eff_um,
                pitch_c_eff_um,
            )
            tail_hit_count += _directed_line_dilation(
                critical_region_mask,
                offsets,
            )

        tail_hit_probability = tail_hit_count / float(n_angles)
        conditional_fatal_probability = np.where(
            main_hit_mask,
            1.0,
            tail_hit_probability,
        )
        fatal_probability += thickness_weight * conditional_fatal_probability

    avg_fatal_particles = float(
        np.sum(density * fatal_probability) * cell_area_um2
    )
    info = {
        "tail_grid_shape": tuple(int(v) for v in critical_region_mask.shape),
        "tail_grid_pitch_um": float(grid_pitch_um),
        "tail_model_margin_um": float(model_margin_um),
        "tail_num_angles": int(n_angles),
        "tail_num_thickness_nodes": int(len(thickness_um)),
        "tail_length_cap_um": float(length_cap_um),
    }
    return avg_fatal_particles, info


def _particle_yield_array_from_critical_bitmap(cfg, wafer_interface, pad_bitmap_collection):
    """
    Calculate per-die particle yield for an arbitrary critical-pad layout.
    """
    critical_pad_bitmap = np.asarray(
        pad_bitmap_collection["CRITICAL_PAD_BITMAP"],
        dtype=bool,
    )
    if not np.any(critical_pad_bitmap):
        return np.ones(wafer_interface.num_dies, dtype=np.float64), {
            "method": "critical_bitmap_distance_transform",
            "num_critical_pads": 0,
        }

    yield_array = np.full(wafer_interface.num_dies, np.nan, dtype=np.float64)
    use_exact_die_position = _cfg_bool(cfg, "DEFECT_USE_EXACT_DIE_POSITION", False)
    include_void_tail = True

    if include_void_tail:
        dist_to_critical_pad_um = None
        x_coords_um = None
        y_coords_um = None
        grid_shape = critical_pad_bitmap.shape
    else:
        critical_grid, x_coords_um, y_coords_um = _critical_bitmap_on_die_grid(
            cfg,
            critical_pad_bitmap,
        )
        dist_to_critical_pad_um = distance_transform_edt(
            ~critical_grid,
            sampling=(_cfg_float(cfg, "PITCH_r_um"), _cfg_float(cfg, "PITCH_c_um")),
        )
        grid_shape = critical_grid.shape
        del critical_grid

    if use_exact_die_position:
        chunk_rows = 0
        for die_ind, die in enumerate(wafer_interface.die_list):
            if include_void_tail:
                avg_fatal_particles, tail_info = _tail_dilation_fatal_integral(
                    cfg,
                    critical_pad_bitmap,
                    die_center_radius_um=float(np.linalg.norm(die.die_center)),
                )
                chunk_rows = tail_info["tail_grid_shape"][0]
            else:
                avg_fatal_particles, chunk_rows = _fatal_particle_integral_from_distance_map(
                    cfg,
                    dist_to_critical_pad_um,
                    x_coords_um,
                    y_coords_um,
                    die_center_um=die.die_center,
                )
            yield_array[die_ind] = np.exp(-avg_fatal_particles)
        num_groups = wafer_interface.num_dies
        method = (
            "critical_bitmap_direction_averaged_tail_exact_radius"
            if include_void_tail
            else "critical_bitmap_distance_transform_exact_die_position"
        )
    else:
        radius_bin_value = _cfg_get(
            cfg,
            "DEFECT_RADIAL_BIN_UM",
            max(_cfg_float(cfg, "DIE_W_um"), _cfg_float(cfg, "DIE_L_um")),
        )
        radius_bin_um = None if radius_bin_value is None else float(radius_bin_value)
        radius_decimals = int(_cfg_float(cfg, "DEFECT_RADIAL_DECIMALS", 0))
        radial_info = wafer_interface.get_die_radial_layers(
            radius_bin_um=radius_bin_um,
            decimals=radius_decimals,
        )
        chunk_rows = 0
        tail_info = {}
        for group in radial_info["layers"]:
            rep_die = wafer_interface.die_list[int(group["representative_index"])]
            die_center_radius_um = float(np.linalg.norm(rep_die.die_center))
            if include_void_tail:
                avg_fatal_particles, tail_info = _tail_dilation_fatal_integral(
                    cfg,
                    critical_pad_bitmap,
                    die_center_radius_um=die_center_radius_um,
                )
                chunk_rows = tail_info["tail_grid_shape"][0]
            else:
                avg_fatal_particles, chunk_rows = _fatal_particle_integral_from_distance_map(
                    cfg,
                    dist_to_critical_pad_um,
                    x_coords_um,
                    y_coords_um,
                    die_center_radius_um=die_center_radius_um,
                )
            yield_array[group["indices"]] = np.exp(-avg_fatal_particles)
        num_groups = len(radial_info["layers"])
        method = (
            "critical_bitmap_direction_averaged_tail_radial_center_radius"
            if include_void_tail
            else "critical_bitmap_distance_transform_radial_center_radius"
        )

    info = {
        "method": method,
        "num_critical_pads": int(np.sum(critical_pad_bitmap)),
        "grid_shape": tuple(int(v) for v in grid_shape),
        "chunk_rows": int(chunk_rows),
        "num_die_groups": int(num_groups),
        "density_model": "D0_edge_taper_to_D1",
        "include_void_tail": bool(include_void_tail),
    }
    if include_void_tail:
        info.update(tail_info)
    return yield_array, info


def get_bitmap_bounds(*,
        bitmap: np.ndarray,
        pad_block_size: int
    ):
    # Find the bounds of the non-zero pixels in the bitmap
    rows = np.any(bitmap, axis=1)
    cols = np.any(bitmap, axis=0)

    if not np.any(rows) or not np.any(cols):
        return 0, 0

    top, bottom = np.where(rows)[0][[0, -1]] * pad_block_size
    left, right = np.where(cols)[0][[0, -1]] * pad_block_size

    height = bottom - top + 1
    width = right - left + 1

    return width, height









def stack_defect_yield_calculator(
    cfg_dict: dict,
    waf_stack,
):
    """
    Calculate particle yield from the critical-pad layout on each W2W interface.
    """
    for interface_name, cfg in cfg_dict.items():
        wafer_interface = waf_stack.interfaces.interface_dict[interface_name]
        pad_bitmap_collection = waf_stack.interfaces.pad_bitmap_collection_dict[interface_name]

        defect_yield_array, info = _particle_yield_array_from_critical_bitmap(
            cfg,
            wafer_interface,
            pad_bitmap_collection,
        )
        waf_stack.die_yield_list_per_interface_dict[interface_name]['particle'] = defect_yield_array
        waf_stack.interfaces.failure_params_dict[interface_name]["particle_model_info"] = info
