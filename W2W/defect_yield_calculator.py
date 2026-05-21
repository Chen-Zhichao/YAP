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
import hashlib
from scipy.ndimage import distance_transform_edt

try:
    from numba import njit
except Exception:  # pragma: no cover - numba is an optional acceleration path
    njit = None


if njit is not None:
    @njit(cache=False)
    def _redundant_fatal_mask_for_offsets_numba(
        n_rows,
        n_cols,
        group_ids,
        group_start,
        group_end,
        rows,
        cols,
        counts,
        tolerated,
        offsets,
    ):
        scratch = np.zeros((n_rows, n_cols), dtype=np.int32)
        fatal_mask = np.zeros((n_rows, n_cols), dtype=np.bool_)
        touched_rows = np.empty(n_rows * n_cols, dtype=np.int64)
        touched_cols = np.empty(n_rows * n_cols, dtype=np.int64)

        for group_idx in range(group_ids.shape[0]):
            group_id = group_ids[group_idx]
            tolerance = tolerated[group_id]
            touched_count = 0
            for cell_idx in range(group_start[group_idx], group_end[group_idx]):
                pad_row = rows[cell_idx]
                pad_col = cols[cell_idx]
                pad_count = counts[cell_idx]
                for offset_idx in range(offsets.shape[0]):
                    center_row = pad_row - offsets[offset_idx, 0]
                    center_col = pad_col - offsets[offset_idx, 1]
                    if (
                        center_row < 0
                        or center_row >= n_rows
                        or center_col < 0
                        or center_col >= n_cols
                    ):
                        continue
                    if scratch[center_row, center_col] == 0:
                        touched_rows[touched_count] = center_row
                        touched_cols[touched_count] = center_col
                        touched_count += 1
                    scratch[center_row, center_col] += pad_count

            for touched_idx in range(touched_count):
                center_row = touched_rows[touched_idx]
                center_col = touched_cols[touched_idx]
                if scratch[center_row, center_col] > tolerance:
                    fatal_mask[center_row, center_col] = True
                scratch[center_row, center_col] = 0

        return fatal_mask
else:
    _redundant_fatal_mask_for_offsets_numba = None


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


def _pad_coords_for_mask(pad_bitmap_collection, pad_mask):
    pad_coords = np.asarray(pad_bitmap_collection["pad_coords"])
    pad_mask_flat = np.asarray(pad_mask, dtype=bool).reshape(-1)
    finite_coord_mask = np.isfinite(pad_coords[:, 0]) & np.isfinite(pad_coords[:, 1])
    return pad_coords[pad_mask_flat & finite_coord_mask].astype(np.float64, copy=False)


def _redundant_group_arrays_from_collection(pad_bitmap_collection):
    pad_coords = np.asarray(pad_bitmap_collection["pad_coords"])
    total_pad_count = int(pad_coords.shape[0])

    group_id_per_pad = pad_bitmap_collection.get("redundant_group_id_per_pad")
    tolerated_mechanical = pad_bitmap_collection.get(
        "redundant_tolerated_mechanical_failures"
    )
    if group_id_per_pad is not None and tolerated_mechanical is not None:
        return (
            np.asarray(group_id_per_pad, dtype=np.int64).reshape(-1),
            np.asarray(tolerated_mechanical, dtype=np.int64).reshape(-1),
        )

    redundant_net_to_1d_physical_mask = pad_bitmap_collection.get(
        "redundant_net_to_1d_physical_mask",
        {},
    )
    criticality_info = pad_bitmap_collection.get("criticality_info", {})
    group_id_per_pad = np.full(total_pad_count, -1, dtype=np.int64)
    tolerated_mechanical = np.zeros(len(redundant_net_to_1d_physical_mask), dtype=np.int64)

    for group_id, (net, physical_idx) in enumerate(
        redundant_net_to_1d_physical_mask.items()
    ):
        physical_idx = np.asarray(physical_idx, dtype=np.int64).reshape(-1)
        physical_idx = physical_idx[
            (physical_idx >= 0) & (physical_idx < total_pad_count)
        ]
        group_id_per_pad[physical_idx] = group_id
        tolerated_mechanical[group_id] = int(
            criticality_info.get(net, {}).get("tolerated_mechanical_failures", 0)
        )
    return group_id_per_pad, tolerated_mechanical


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


def _disk_offsets(radius_um, pitch_r_um, pitch_c_um):
    if radius_um <= 0.0:
        return [(0, 0)]
    row_lim = int(np.ceil(radius_um / pitch_r_um))
    col_lim = int(np.ceil(radius_um / pitch_c_um))
    offsets = []
    radius_sq = radius_um * radius_um
    for row_offset in range(-row_lim, row_lim + 1):
        dy_um = row_offset * pitch_r_um
        for col_offset in range(-col_lim, col_lim + 1):
            dx_um = col_offset * pitch_c_um
            if dx_um * dx_um + dy_um * dy_um <= radius_sq:
                offsets.append((row_offset, col_offset))
    return offsets or [(0, 0)]


def _structure_offsets_for_tail_and_main(
    main_radius_um,
    tail_length_um,
    theta_rad,
    pitch_r_um,
    pitch_c_um,
):
    disk_offsets = _disk_offsets(main_radius_um, pitch_r_um, pitch_c_um)
    line_offsets = _line_offsets_for_direction(
        tail_length_um,
        theta_rad,
        pitch_r_um,
        pitch_c_um,
    )
    return sorted(set(disk_offsets).union(line_offsets))


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


def _cell_index_from_coords(x_um, y_um, x_coords_um, y_coords_um):
    pitch_c_um = float(abs(x_coords_um[1] - x_coords_um[0]))
    pitch_r_um = float(abs(y_coords_um[0] - y_coords_um[1]))
    col = np.rint((x_um - x_coords_um[0]) / pitch_c_um).astype(np.int64)
    row = np.rint((y_coords_um[0] - y_um) / pitch_r_um).astype(np.int64)
    return row, col


def _redundant_group_cells_on_model_grid(
    pad_bitmap_collection,
    x_coords_um,
    y_coords_um,
    cell_count_mode="actual",
):
    redundant_mask = np.asarray(
        pad_bitmap_collection.get("REDUNDANT_PAD_BITMAP", np.zeros(0, dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    if redundant_mask.size == 0 or not np.any(redundant_mask):
        return None

    pad_coords = np.asarray(pad_bitmap_collection["pad_coords"])
    group_id_per_pad, tolerated_mechanical = _redundant_group_arrays_from_collection(
        pad_bitmap_collection
    )
    finite_coord_mask = np.isfinite(pad_coords[:, 0]) & np.isfinite(pad_coords[:, 1])
    valid_mask = (
        redundant_mask
        & finite_coord_mask
        & (group_id_per_pad[: redundant_mask.size] >= 0)
    )
    if not np.any(valid_mask):
        return None

    red_coords = pad_coords[valid_mask]
    red_group_ids = group_id_per_pad[valid_mask].astype(np.int64, copy=False)
    rows, cols = _cell_index_from_coords(
        red_coords[:, 0].astype(np.float64, copy=False),
        red_coords[:, 1].astype(np.float64, copy=False),
        x_coords_um,
        y_coords_um,
    )
    in_grid = (
        (rows >= 0)
        & (rows < y_coords_um.shape[0])
        & (cols >= 0)
        & (cols < x_coords_um.shape[0])
    )
    if not np.any(in_grid):
        return None

    triples = np.column_stack((red_group_ids[in_grid], rows[in_grid], cols[in_grid]))
    unique_triples, counts = np.unique(triples, axis=0, return_counts=True)
    order = np.argsort(unique_triples[:, 0], kind="stable")
    unique_triples = unique_triples[order]
    counts = counts[order].astype(np.int32, copy=False)
    cell_count_mode = str(cell_count_mode).strip().lower()
    if cell_count_mode in ("binary", "one", "presence"):
        # On a coarse modeling grid, many same-group pads can collapse into a
        # single grid cell. Counting all collapsed pads as simultaneously hit is
        # overly pessimistic for redundancy tolerance. Treat each occupied
        # group-cell as one hit while the critical-pad area still uses the
        # coarse dilation grid.
        counts = np.ones_like(counts, dtype=np.int32)
    elif cell_count_mode not in ("actual", "count"):
        raise ValueError(
            "DEFECT_REDUNDANT_CELL_COUNT_MODE must be 'binary' or 'actual'."
        )

    group_ids = unique_triples[:, 0].astype(np.int64, copy=False)
    group_start = []
    group_end = []
    unique_group_ids = []
    cursor = 0
    while cursor < group_ids.shape[0]:
        group_id = int(group_ids[cursor])
        end = cursor + 1
        while end < group_ids.shape[0] and int(group_ids[end]) == group_id:
            end += 1
        unique_group_ids.append(group_id)
        group_start.append(cursor)
        group_end.append(end)
        cursor = end

    return {
        "group_ids": np.asarray(unique_group_ids, dtype=np.int64),
        "group_start": np.asarray(group_start, dtype=np.int64),
        "group_end": np.asarray(group_end, dtype=np.int64),
        "rows": unique_triples[:, 1].astype(np.int64, copy=False),
        "cols": unique_triples[:, 2].astype(np.int64, copy=False),
        "counts": counts,
        "tolerated_mechanical": np.asarray(tolerated_mechanical, dtype=np.int64),
        "num_redundant_pads": int(np.sum(redundant_mask)),
    }


def _redundant_fatal_mask_for_offsets(
    shape,
    redundant_cell_data,
    offsets,
):
    if redundant_cell_data is None:
        return np.zeros(shape, dtype=bool)

    n_rows, n_cols = shape
    if _redundant_fatal_mask_for_offsets_numba is not None:
        offsets_array = np.asarray(offsets, dtype=np.int64)
        return _redundant_fatal_mask_for_offsets_numba(
            int(n_rows),
            int(n_cols),
            redundant_cell_data["group_ids"],
            redundant_cell_data["group_start"],
            redundant_cell_data["group_end"],
            redundant_cell_data["rows"],
            redundant_cell_data["cols"],
            redundant_cell_data["counts"],
            redundant_cell_data["tolerated_mechanical"],
            offsets_array,
        )

    scratch = np.zeros(shape, dtype=np.int32)
    fatal_mask = np.zeros(shape, dtype=bool)
    touched = []

    group_ids = redundant_cell_data["group_ids"]
    group_start = redundant_cell_data["group_start"]
    group_end = redundant_cell_data["group_end"]
    rows = redundant_cell_data["rows"]
    cols = redundant_cell_data["cols"]
    counts = redundant_cell_data["counts"]
    tolerated = redundant_cell_data["tolerated_mechanical"]

    for group_idx, group_id in enumerate(group_ids):
        tolerance = int(tolerated[int(group_id)])
        touched.clear()
        for cell_idx in range(int(group_start[group_idx]), int(group_end[group_idx])):
            pad_row = int(rows[cell_idx])
            pad_col = int(cols[cell_idx])
            pad_count = int(counts[cell_idx])
            for row_offset, col_offset in offsets:
                center_row = pad_row - int(row_offset)
                center_col = pad_col - int(col_offset)
                if not (0 <= center_row < n_rows and 0 <= center_col < n_cols):
                    continue
                if scratch[center_row, center_col] == 0:
                    touched.append((center_row, center_col))
                scratch[center_row, center_col] += pad_count

        for center_row, center_col in touched:
            if scratch[center_row, center_col] > tolerance:
                fatal_mask[center_row, center_col] = True
            scratch[center_row, center_col] = 0

    return fatal_mask


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


def _build_tail_layout_cache(cfg, pad_bitmap_collection, max_die_center_radius_um):
    critical_pad_bitmap = np.asarray(
        pad_bitmap_collection.get("CRITICAL_PAD_BITMAP", np.zeros(0, dtype=bool)),
        dtype=bool,
    )
    default_grid_pitch_um = max(
        400.0,
        _cfg_float(cfg, "PITCH_r_um"),
        _cfg_float(cfg, "PITCH_c_um"),
    )
    grid_pitch_um = _cfg_float(cfg, "DEFECT_TAIL_GRID_PITCH_um", default_grid_pitch_um)
    grid_pitch_um = max(grid_pitch_um, 1e-9)

    pad_top_r_um = _cfg_float(cfg, "PAD_TOP_R_um")
    n_angles = int(_cfg_float(cfg, "DEFECT_TAIL_NUM_ANGLES", 16))
    n_angles = max(1, n_angles)
    angles_rad = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    thickness_um, thickness_weights = _particle_thickness_quadrature(cfg)
    max_sqrt_thickness = np.sqrt(float(np.max(thickness_um)))

    max_radius_scale_um = (
        _cfg_float(cfg, "k_r") * float(max_die_center_radius_um) + _cfg_float(cfg, "k_r0")
    )
    max_tail_scale_um = _cfg_float(cfg, "k_L", 0.0) * float(max_die_center_radius_um)
    length_cap_default_um = max_tail_scale_um * max_sqrt_thickness
    length_cap_um = _cfg_float(cfg, "DEFECT_TAIL_LENGTH_CAP_um", length_cap_default_um)
    max_main_radius_um = max_radius_scale_um * max_sqrt_thickness
    model_margin_um = max(length_cap_um, max_main_radius_um) + pad_top_r_um

    critical_region_mask, dist_to_critical_center_um, x_coords_um, y_coords_um = (
        _critical_region_on_model_grid(
            cfg,
            critical_pad_bitmap,
            grid_pitch_um,
            margin_um=model_margin_um,
        )
    )
    redundant_cell_data = _redundant_group_cells_on_model_grid(
        pad_bitmap_collection,
        x_coords_um,
        y_coords_um,
        cell_count_mode=_cfg_get(cfg, "DEFECT_REDUNDANT_CELL_COUNT_MODE", "actual"),
    )

    pitch_c_eff_um = float(abs(x_coords_um[1] - x_coords_um[0]))
    pitch_r_eff_um = float(abs(y_coords_um[0] - y_coords_um[1]))
    required_main_radius_um = np.maximum(
        dist_to_critical_center_um - pad_top_r_um,
        0.0,
    )
    density = _particle_density_chunk_um2(cfg, x_coords_um, y_coords_um)

    return {
        "critical_region_mask": critical_region_mask,
        "redundant_cell_data": redundant_cell_data,
        "x_coords_um": x_coords_um,
        "y_coords_um": y_coords_um,
        "pitch_r_eff_um": pitch_r_eff_um,
        "pitch_c_eff_um": pitch_c_eff_um,
        "cell_area_um2": pitch_r_eff_um * pitch_c_eff_um,
        "required_main_radius_um": required_main_radius_um,
        "density": density,
        "angles_rad": angles_rad,
        "thickness_um": thickness_um,
        "thickness_weights": thickness_weights,
        "tail_grid_pitch_um": float(grid_pitch_um),
        "tail_model_margin_um": float(model_margin_um),
        "tail_num_angles": int(n_angles),
        "tail_num_thickness_nodes": int(len(thickness_um)),
        "num_redundant_pads": (
            0 if redundant_cell_data is None else int(redundant_cell_data["num_redundant_pads"])
        ),
        "redundant_group_count": (
            0 if redundant_cell_data is None else int(redundant_cell_data["group_ids"].shape[0])
        ),
    }


def _tail_dilation_fatal_integral(
    cfg,
    pad_bitmap_collection,
    die_center_radius_um,
    layout_cache=None,
):
    """
    Direction-averaged main-void + directed-tail fatal-particle integral.

    Critical pads are fatal when the main void or directed tail overlaps any
    critical pad. Redundant groups are fatal when that same single particle
    structure overlaps more pads in one group than the group's mechanical
    tolerance.
    """
    pad_top_r_um = _cfg_float(cfg, "PAD_TOP_R_um")
    radius_scale_um = _cfg_float(cfg, "k_r") * die_center_radius_um + _cfg_float(cfg, "k_r0")
    tail_scale_um = _cfg_float(cfg, "k_L", 0.0) * die_center_radius_um

    if layout_cache is None:
        layout_cache = _build_tail_layout_cache(
            cfg,
            pad_bitmap_collection,
            max_die_center_radius_um=die_center_radius_um,
        )
    critical_region_mask = layout_cache["critical_region_mask"]
    redundant_cell_data = layout_cache["redundant_cell_data"]
    x_coords_um = layout_cache["x_coords_um"]
    y_coords_um = layout_cache["y_coords_um"]
    pitch_r_eff_um = layout_cache["pitch_r_eff_um"]
    pitch_c_eff_um = layout_cache["pitch_c_eff_um"]
    cell_area_um2 = layout_cache["cell_area_um2"]
    required_main_radius_um = layout_cache["required_main_radius_um"]
    density = layout_cache["density"]
    angles_rad = layout_cache["angles_rad"]
    thickness_um = layout_cache["thickness_um"]
    thickness_weights = layout_cache["thickness_weights"]

    length_cap_default_um = tail_scale_um * np.sqrt(float(np.max(thickness_um)))
    length_cap_um = _cfg_float(cfg, "DEFECT_TAIL_LENGTH_CAP_um", length_cap_default_um)

    if not np.any(critical_region_mask) and redundant_cell_data is None:
        return 0.0, {
            "tail_grid_shape": tuple(int(v) for v in critical_region_mask.shape),
            "tail_grid_pitch_um": float(layout_cache["tail_grid_pitch_um"]),
            "tail_model_margin_um": float(layout_cache["tail_model_margin_um"]),
            "num_redundant_pads": 0,
            "redundant_group_count": 0,
        }

    fatal_probability = np.zeros(critical_region_mask.shape, dtype=np.float64)

    for thickness, thickness_weight in zip(thickness_um, thickness_weights):
        sqrt_thickness = np.sqrt(float(thickness))
        main_radius_um = radius_scale_um * sqrt_thickness
        main_hit_mask = required_main_radius_um <= main_radius_um

        tail_length_um = min(tail_scale_um * sqrt_thickness, length_cap_um)
        if tail_length_um <= 0.0:
            redundant_hit = np.zeros(critical_region_mask.shape, dtype=bool)
            if redundant_cell_data is not None:
                redundant_hit = _redundant_fatal_mask_for_offsets(
                    critical_region_mask.shape,
                    redundant_cell_data,
                    _disk_offsets(main_radius_um, pitch_r_eff_um, pitch_c_eff_um),
                )
            fatal_probability += thickness_weight * (
                main_hit_mask | redundant_hit
            ).astype(np.float64)
            continue

        tail_hit_count = np.zeros(critical_region_mask.shape, dtype=np.float64)
        for theta_rad in angles_rad:
            line_offsets = _line_offsets_for_direction(
                tail_length_um,
                theta_rad,
                pitch_r_eff_um,
                pitch_c_eff_um,
            )
            critical_tail_hit = _directed_line_dilation(
                critical_region_mask,
                line_offsets,
            )
            redundant_hit = np.zeros(critical_region_mask.shape, dtype=bool)
            if redundant_cell_data is not None:
                structure_offsets = _structure_offsets_for_tail_and_main(
                    main_radius_um,
                    tail_length_um,
                    theta_rad,
                    pitch_r_eff_um,
                    pitch_c_eff_um,
                )
                redundant_hit = _redundant_fatal_mask_for_offsets(
                    critical_region_mask.shape,
                    redundant_cell_data,
                    structure_offsets,
                )
            tail_hit_count += (critical_tail_hit | redundant_hit).astype(np.float64)

        tail_hit_probability = tail_hit_count / float(len(angles_rad))
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
        "tail_grid_pitch_um": float(layout_cache["tail_grid_pitch_um"]),
        "tail_model_margin_um": float(layout_cache["tail_model_margin_um"]),
        "tail_num_angles": int(layout_cache["tail_num_angles"]),
        "tail_num_thickness_nodes": int(len(thickness_um)),
        "tail_length_cap_um": float(length_cap_um),
        "num_redundant_pads": int(layout_cache["num_redundant_pads"]),
        "redundant_group_count": int(layout_cache["redundant_group_count"]),
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
    redundant_pad_bitmap = np.asarray(
        pad_bitmap_collection.get("REDUNDANT_PAD_BITMAP", np.zeros(0, dtype=bool)),
        dtype=bool,
    )
    if not np.any(critical_pad_bitmap) and not np.any(redundant_pad_bitmap):
        return np.ones(wafer_interface.num_dies, dtype=np.float64), {
            "method": "critical_bitmap_distance_transform",
            "num_critical_pads": 0,
            "num_redundant_pads": 0,
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
        tail_layout_cache = None
        if include_void_tail:
            max_die_center_radius_um = max(
                float(np.linalg.norm(die.die_center))
                for die in wafer_interface.die_list
            )
            tail_layout_cache = _build_tail_layout_cache(
                cfg,
                pad_bitmap_collection,
                max_die_center_radius_um=max_die_center_radius_um,
            )
        for die_ind, die in enumerate(wafer_interface.die_list):
            if include_void_tail:
                avg_fatal_particles, tail_info = _tail_dilation_fatal_integral(
                    cfg,
                    pad_bitmap_collection,
                    die_center_radius_um=float(np.linalg.norm(die.die_center)),
                    layout_cache=tail_layout_cache,
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
        tail_layout_cache = None
        if include_void_tail:
            max_die_center_radius_um = max(
                float(np.linalg.norm(wafer_interface.die_list[int(group["representative_index"])].die_center))
                for group in radial_info["layers"]
            )
            tail_layout_cache = _build_tail_layout_cache(
                cfg,
                pad_bitmap_collection,
                max_die_center_radius_um=max_die_center_radius_um,
            )
        for group in radial_info["layers"]:
            rep_die = wafer_interface.die_list[int(group["representative_index"])]
            die_center_radius_um = float(np.linalg.norm(rep_die.die_center))
            if include_void_tail:
                avg_fatal_particles, tail_info = _tail_dilation_fatal_integral(
                    cfg,
                    pad_bitmap_collection,
                    die_center_radius_um=die_center_radius_um,
                    layout_cache=tail_layout_cache,
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
        "num_redundant_pads": int(np.sum(redundant_pad_bitmap)),
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


def _array_digest(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _particle_model_cache_key(cfg, wafer_interface, pad_bitmap_collection):
    cfg_keys = (
        "PITCH_r_um",
        "PITCH_c_um",
        "DIE_W_um",
        "DIE_L_um",
        "PAD_TOP_R_um",
        "PAD_ARR_W_um",
        "PAD_ARR_L_um",
        "D0",
        "D1",
        "EDGE_REGION_WIDTH_um",
        "DEFECT_TAIL_GRID_PITCH_um",
        "DEFECT_TAIL_NUM_ANGLES",
        "DEFECT_TAIL_THICKNESS_NODES",
        "DEFECT_TAIL_LENGTH_CAP_um",
        "DEFECT_REDUNDANT_CELL_COUNT_MODE",
        "DEFECT_USE_EXACT_DIE_POSITION",
        "DEFECT_RADIAL_BIN_UM",
        "DEFECT_RADIAL_DECIMALS",
        "t_0",
        "z",
        "k_r",
        "k_r0",
        "k_L",
        "k_n",
        "k_S",
        "VOID_SHAPE",
    )
    key_parts = [("num_dies", int(wafer_interface.num_dies))]
    for cfg_key in cfg_keys:
        key_parts.append((cfg_key, _cfg_get(cfg, cfg_key, None)))

    critical_bitmap = np.asarray(
        pad_bitmap_collection.get("CRITICAL_PAD_BITMAP", np.zeros(0, dtype=bool)),
        dtype=bool,
    )
    redundant_bitmap = np.asarray(
        pad_bitmap_collection.get("REDUNDANT_PAD_BITMAP", np.zeros(0, dtype=bool)),
        dtype=bool,
    )
    group_id_per_pad, tolerated_mechanical = _redundant_group_arrays_from_collection(
        pad_bitmap_collection
    )
    key_parts.extend(
        [
            ("critical_bitmap", _array_digest(critical_bitmap)),
            ("redundant_bitmap", _array_digest(redundant_bitmap)),
            ("redundant_group_id_per_pad", _array_digest(group_id_per_pad)),
            ("redundant_tolerated_mechanical", _array_digest(tolerated_mechanical)),
        ]
    )
    return tuple(key_parts)









def stack_defect_yield_calculator(
    cfg_dict: dict,
    waf_stack,
):
    """
    Calculate particle yield from the critical-pad layout on each W2W interface.
    """
    model_cache = {}
    for interface_name, cfg in cfg_dict.items():
        wafer_interface = waf_stack.interfaces.interface_dict[interface_name]
        pad_bitmap_collection = waf_stack.interfaces.pad_bitmap_collection_dict[interface_name]

        cache_key = _particle_model_cache_key(cfg, wafer_interface, pad_bitmap_collection)
        if cache_key in model_cache:
            defect_yield_array, info = model_cache[cache_key]
            info = dict(info)
            info["cache_hit"] = True
        else:
            defect_yield_array, info = _particle_yield_array_from_critical_bitmap(
                cfg,
                wafer_interface,
                pad_bitmap_collection,
            )
            info = dict(info)
            info["cache_hit"] = False
            model_cache[cache_key] = (defect_yield_array.copy(), dict(info))
        waf_stack.die_yield_list_per_interface_dict[interface_name]['particle'] = defect_yield_array
        waf_stack.interfaces.failure_params_dict[interface_name]["particle_model_info"] = info
