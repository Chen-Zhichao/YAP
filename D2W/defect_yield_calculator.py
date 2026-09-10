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
import hashlib
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

try:
    from numba import njit
except Exception:  # pragma: no cover - numba is an optional speed path
    njit = None


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


def _array_digest(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _particle_model_cache_key(cfg, pad_bitmap_collection):
    """Identify interfaces with identical particle-yield inputs."""
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
        "DEFECT_MODEL_GRID_PITCH_um",
        "DEFECT_MAX_MODEL_GRID_POINTS",
        "DEFECT_DISTANCE_TRANSFORM_MAX_CELLS",
        "DEFECT_REDUNDANT_K_NEIGHBORS",
        "first_contact",
        "t_0",
        "z",
        "k_r",
        "k_r0",
    )
    key_parts = [(key, _cfg_get(cfg, key, None)) for key in cfg_keys]

    critical_bitmap = np.asarray(
        pad_bitmap_collection.get("CRITICAL_PAD_BITMAP", np.zeros(0, dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    redundant_bitmap = np.asarray(
        pad_bitmap_collection.get("REDUNDANT_PAD_BITMAP", np.zeros(0, dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    group_id_per_pad, tolerated_mechanical = _redundant_group_arrays_from_collection(
        pad_bitmap_collection
    )
    sensitive_mask = critical_bitmap | redundant_bitmap
    pad_coords = np.asarray(pad_bitmap_collection["pad_coords"])

    key_parts.extend(
        [
            ("critical_bitmap", _array_digest(critical_bitmap)),
            ("redundant_bitmap", _array_digest(redundant_bitmap)),
            (
                "sensitive_pad_coords",
                _array_digest(pad_coords[sensitive_mask]),
            ),
            (
                "redundant_group_id_per_pad",
                _array_digest(group_id_per_pad[sensitive_mask]),
            ),
            (
                "redundant_tolerated_mechanical",
                _array_digest(tolerated_mechanical),
            ),
        ]
    )
    return tuple(key_parts)


if njit is not None:
    @njit(cache=False)
    def _redundant_required_distance_from_knn_numba(
        neighbor_dist_um,
        neighbor_group_id,
        tolerated_failures,
    ):
        n_points = neighbor_dist_um.shape[0]
        k_neighbors = neighbor_dist_um.shape[1]
        max_seen = k_neighbors
        required_dist_um = np.empty(n_points, dtype=np.float64)

        for point_idx in range(n_points):
            required_dist_um[point_idx] = np.inf
            seen_ids = np.empty(max_seen, dtype=np.int64)
            seen_counts = np.zeros(max_seen, dtype=np.int32)
            n_seen = 0

            for neighbor_idx in range(k_neighbors):
                group_id = neighbor_group_id[point_idx, neighbor_idx]
                if group_id < 0:
                    continue
                dist_um = neighbor_dist_um[point_idx, neighbor_idx]
                if not np.isfinite(dist_um):
                    continue

                found = False
                point_done = False
                for seen_idx in range(n_seen):
                    if seen_ids[seen_idx] == group_id:
                        seen_counts[seen_idx] += 1
                        if seen_counts[seen_idx] > tolerated_failures[group_id]:
                            required_dist_um[point_idx] = dist_um
                            point_done = True
                        found = True
                        break

                if found:
                    if point_done:
                        break
                    continue

                seen_ids[n_seen] = group_id
                seen_counts[n_seen] = 1
                if seen_counts[n_seen] > tolerated_failures[group_id]:
                    required_dist_um[point_idx] = dist_um
                    break
                n_seen += 1

        return required_dist_um
else:
    _redundant_required_distance_from_knn_numba = None


def _redundant_required_distance_from_knn_py(
    neighbor_dist_um,
    neighbor_group_id,
    tolerated_failures,
):
    """Python fallback for the k-nearest redundant-group fatal distance."""
    required_dist_um = np.full(neighbor_dist_um.shape[0], np.inf, dtype=np.float64)
    for point_idx in range(neighbor_dist_um.shape[0]):
        group_counts = {}
        for neighbor_idx in range(neighbor_dist_um.shape[1]):
            group_id = int(neighbor_group_id[point_idx, neighbor_idx])
            if group_id < 0:
                continue
            dist_um = float(neighbor_dist_um[point_idx, neighbor_idx])
            if not np.isfinite(dist_um):
                continue
            count = group_counts.get(group_id, 0) + 1
            if count > int(tolerated_failures[group_id]):
                required_dist_um[point_idx] = dist_um
                break
            group_counts[group_id] = count
    return required_dist_um


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


def _particle_center_grid(cfg):
    """
    Build a particle-center integration grid over the die.

    This grid is deliberately decoupled from the pad pitch. For very fine pad
    pitches, especially 1 um or 0.1 um, tying the integration grid to every pad
    location would create arrays that are too large to be useful. The default
    pitch follows the pad pitch for ordinary cases, then coarsens only when the
    requested grid would exceed DEFECT_MAX_MODEL_GRID_POINTS.
    """
    die_w_um = _cfg_float(cfg, "DIE_W_um")
    die_l_um = _cfg_float(cfg, "DIE_L_um")
    pitch_r_um = _cfg_float(cfg, "PITCH_r_um")
    pitch_c_um = _cfg_float(cfg, "PITCH_c_um")
    requested_pitch_um = _cfg_float(
        cfg,
        "DEFECT_MODEL_GRID_PITCH_um",
        min(pitch_r_um, pitch_c_um),
    )
    if requested_pitch_um <= 0:
        raise ValueError("DEFECT_MODEL_GRID_PITCH_um must be positive.")

    nx_req = max(1, int(np.ceil(die_w_um / requested_pitch_um)))
    ny_req = max(1, int(np.ceil(die_l_um / requested_pitch_um)))
    max_points = int(_cfg_float(cfg, "DEFECT_MAX_MODEL_GRID_POINTS", 25_000_000))
    max_points = max(1, max_points)

    nx = nx_req
    ny = ny_req
    coarsened = False
    requested_points = nx_req * ny_req
    if requested_points > max_points:
        scale = math.sqrt(requested_points / max_points)
        nx = max(1, int(np.ceil(nx_req / scale)))
        ny = max(1, int(np.ceil(ny_req / scale)))
        while nx * ny > max_points:
            if nx >= ny and nx > 1:
                nx -= 1
            elif ny > 1:
                ny -= 1
            else:
                break
        coarsened = True

    cell_w_um = die_w_um / nx
    cell_h_um = die_l_um / ny
    x_coords_um = -die_w_um / 2.0 + (np.arange(nx, dtype=np.float64) + 0.5) * cell_w_um
    y_coords_um = die_l_um / 2.0 - (np.arange(ny, dtype=np.float64) + 0.5) * cell_h_um
    info = {
        "requested_grid_pitch_um": float(requested_pitch_um),
        "model_grid_cell_w_um": float(cell_w_um),
        "model_grid_cell_h_um": float(cell_h_um),
        "requested_grid_shape": (int(ny_req), int(nx_req)),
        "grid_shape": (int(ny), int(nx)),
        "coarsened_grid": bool(coarsened),
        "max_model_grid_points": int(max_points),
    }
    return x_coords_um, y_coords_um, cell_w_um * cell_h_um, info


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


def _redundant_pad_group_data(pad_bitmap_collection):
    pad_coords = np.asarray(pad_bitmap_collection["pad_coords"])
    redundant_mask = np.asarray(
        pad_bitmap_collection.get("REDUNDANT_PAD_BITMAP", np.zeros(0, dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    if redundant_mask.size == 0 or not np.any(redundant_mask):
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    group_id_per_pad, tolerated_mechanical = _redundant_group_arrays_from_collection(
        pad_bitmap_collection
    )
    finite_coord_mask = np.isfinite(pad_coords[:, 0]) & np.isfinite(pad_coords[:, 1])
    valid_redundant_mask = (
        redundant_mask
        & finite_coord_mask
        & (group_id_per_pad[: redundant_mask.size] >= 0)
    )
    redundant_coords_um = pad_coords[valid_redundant_mask].astype(
        np.float64,
        copy=False,
    )
    redundant_group_ids = group_id_per_pad[valid_redundant_mask]
    return redundant_coords_um, redundant_group_ids.astype(np.int64), tolerated_mechanical


def _redundant_required_distance_um(
    redundant_tree,
    redundant_group_ids,
    tolerated_mechanical,
    points_um,
    k_neighbors,
):
    if redundant_tree is None or points_um.shape[0] == 0:
        return np.full(points_um.shape[0], np.inf, dtype=np.float64)

    k_neighbors = max(1, min(int(k_neighbors), redundant_group_ids.shape[0]))
    try:
        neighbor_dist_um, neighbor_idx = redundant_tree.query(
            points_um,
            k=k_neighbors,
            workers=-1,
        )
    except TypeError:
        neighbor_dist_um, neighbor_idx = redundant_tree.query(points_um, k=k_neighbors)
    if k_neighbors == 1:
        neighbor_dist_um = neighbor_dist_um.reshape(-1, 1)
        neighbor_idx = neighbor_idx.reshape(-1, 1)

    neighbor_group_id = np.full(neighbor_idx.shape, -1, dtype=np.int64)
    valid_neighbor = neighbor_idx < redundant_group_ids.shape[0]
    neighbor_group_id[valid_neighbor] = redundant_group_ids[
        neighbor_idx[valid_neighbor]
    ]

    if _redundant_required_distance_from_knn_numba is not None:
        return _redundant_required_distance_from_knn_numba(
            np.asarray(neighbor_dist_um, dtype=np.float64),
            neighbor_group_id,
            np.asarray(tolerated_mechanical, dtype=np.int64),
        )
    return _redundant_required_distance_from_knn_py(
        np.asarray(neighbor_dist_um, dtype=np.float64),
        neighbor_group_id,
        np.asarray(tolerated_mechanical, dtype=np.int64),
    )


def _fatal_probability_from_required_distance(
    cfg,
    required_dist_to_pad_center_um,
    distance_to_contact_um,
):
    pad_top_r_um = _cfg_float(cfg, "PAD_TOP_R_um")
    k_r = _cfg_float(cfg, "k_r")
    k_r0 = _cfg_float(cfg, "k_r0")
    t_0 = _cfg_float(cfg, "t_0")
    z = _cfg_float(cfg, "z")
    if z <= 1.0:
        raise ValueError("Particle thickness exponent z must be greater than 1.")

    fatal_probability = np.zeros(required_dist_to_pad_center_um.shape, dtype=np.float64)
    finite_mask = np.isfinite(required_dist_to_pad_center_um)
    if not np.any(finite_mask):
        return fatal_probability

    required_void_radius_um = np.maximum(
        required_dist_to_pad_center_um[finite_mask] - pad_top_r_um,
        0.0,
    )
    radius_scale = k_r * distance_to_contact_um[finite_mask] + k_r0
    if np.any(radius_scale <= 0):
        raise ValueError("Main void radius scale k_r * L + k_r0 must be positive.")

    required_thickness_um = (required_void_radius_um / radius_scale) ** 2
    finite_fatal = np.ones(required_thickness_um.shape, dtype=np.float64)
    needs_large_particle = required_thickness_um > t_0
    finite_fatal[needs_large_particle] = (
        t_0 / required_thickness_um[needs_large_particle]
    ) ** (z - 1.0)
    fatal_probability[finite_mask] = np.clip(finite_fatal, 0.0, 1.0)
    return fatal_probability


def _fatal_particle_integral_from_pad_layout(
    cfg,
    critical_pad_coords_um,
    redundant_pad_coords_um,
    redundant_group_ids,
    tolerated_mechanical,
):
    """
    Layout-aware particle integral for critical and redundant pads.

    It integrates over particle centers on a configurable die grid. The nearest
    critical pad gives the single-critical fatal radius. Redundant groups use a
    k-nearest-neighbor scan: a particle becomes fatal for group g when it covers
    more than tolerated_mechanical[g] pads from that group. This captures a
    single large void damaging multiple pads in the same redundancy group without
    modeling the much rarer accumulation of several independent particles.
    """
    if critical_pad_coords_um.shape[0] == 0 and redundant_pad_coords_um.shape[0] == 0:
        return 0.0, {
            "method": "layout_kdtree_no_sensitive_pads",
            "grid_shape": (0, 0),
            "chunk_rows": 0,
        }

    critical_tree = (
        cKDTree(critical_pad_coords_um)
        if critical_pad_coords_um.shape[0] > 0
        else None
    )
    redundant_tree = (
        cKDTree(redundant_pad_coords_um)
        if redundant_pad_coords_um.shape[0] > 0
        else None
    )
    redundant_k_neighbors = int(_cfg_float(cfg, "DEFECT_REDUNDANT_K_NEIGHBORS", 128))
    if redundant_tree is not None:
        redundant_k_neighbors = min(redundant_k_neighbors, redundant_pad_coords_um.shape[0])
    else:
        redundant_k_neighbors = 0

    x_coords_um, y_coords_um, cell_area_um2, grid_info = _particle_center_grid(cfg)
    chunk_rows = int(_cfg_float(cfg, "DEFECT_CALC_CHUNK_ROWS", 128))
    chunk_rows = max(1, min(chunk_rows, y_coords_um.shape[0]))
    max_query_points = int(
        _cfg_float(
            cfg,
            "DEFECT_MAX_QUERY_POINTS_PER_CHUNK",
            200_000 if redundant_tree is not None else 1_000_000,
        )
    )
    max_query_points = max(1, max_query_points)
    chunk_rows = min(
        chunk_rows,
        max(1, max_query_points // max(1, x_coords_um.shape[0])),
    )
    avg_fatal_particles = 0.0

    for row_start in range(0, y_coords_um.shape[0], chunk_rows):
        row_end = min(row_start + chunk_rows, y_coords_um.shape[0])
        y_chunk_um = y_coords_um[row_start:row_end]
        xx_um, yy_um = np.meshgrid(x_coords_um, y_chunk_um, indexing="xy")
        points_um = np.column_stack((xx_um.ravel(), yy_um.ravel()))

        if critical_tree is not None:
            try:
                critical_dist_um, _ = critical_tree.query(points_um, k=1, workers=-1)
            except TypeError:
                critical_dist_um, _ = critical_tree.query(points_um, k=1)
        else:
            critical_dist_um = np.full(points_um.shape[0], np.inf, dtype=np.float64)

        if redundant_tree is not None:
            redundant_dist_um = _redundant_required_distance_um(
                redundant_tree,
                redundant_group_ids,
                tolerated_mechanical,
                points_um,
                redundant_k_neighbors,
            )
        else:
            redundant_dist_um = np.full(points_um.shape[0], np.inf, dtype=np.float64)

        required_dist_um = np.minimum(critical_dist_um, redundant_dist_um).reshape(
            y_chunk_um.shape[0],
            x_coords_um.shape[0],
        )
        distance_to_contact_um = _distance_from_first_contact_chunk_um(
            cfg,
            x_coords_um,
            y_chunk_um,
        )
        fatal_probability = _fatal_probability_from_required_distance(
            cfg,
            required_dist_um,
            distance_to_contact_um,
        )
        density = _particle_density_chunk_um2(cfg, x_coords_um, y_chunk_um)
        die_mask = _die_mask_chunk(cfg, x_coords_um, y_chunk_um)
        avg_fatal_particles += float(
            np.sum(density * fatal_probability * die_mask) * cell_area_um2
        )

    info = dict(grid_info)
    info.update(
        {
            "method": "layout_kdtree_critical_redundant_chunked",
            "chunk_rows": int(chunk_rows),
            "max_query_points_per_chunk": int(max_query_points),
            "redundant_k_neighbors": int(redundant_k_neighbors),
        }
    )
    return avg_fatal_particles, info


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


def interface_particle_yield_from_pad_layout(cfg, pad_bitmap_collection):
    """
    Calculate particle yield from critical signal bumps and redundant groups.

    Critical signal pads fail when a single particle void reaches any critical
    pad. A redundant group fails when one particle void reaches more pads in the
    group than the group's mechanical tolerance. Independent multi-particle
    accumulation within one redundant group is intentionally ignored.
    """
    critical_pad_bitmap = np.asarray(
        pad_bitmap_collection.get("CRITICAL_PAD_BITMAP", np.zeros(0, dtype=bool)),
        dtype=bool,
    )
    redundant_pad_bitmap = np.asarray(
        pad_bitmap_collection.get("REDUNDANT_PAD_BITMAP", np.zeros(0, dtype=bool)),
        dtype=bool,
    )
    has_critical = critical_pad_bitmap.size > 0 and np.any(critical_pad_bitmap)
    has_redundant = redundant_pad_bitmap.size > 0 and np.any(redundant_pad_bitmap)

    if not has_critical and not has_redundant:
        return 1.0, {
            "avg_fatal_particles": 0.0,
            "effective_critical_area_um2": 0.0,
            "num_critical_pads": 0,
            "num_redundant_pads": 0,
            "redundant_group_count": 0,
            "method": "no_sensitive_pads",
        }

    distance_transform_max_cells = int(
        _cfg_float(cfg, "DEFECT_DISTANCE_TRANSFORM_MAX_CELLS", 25_000_000)
    )
    if has_critical and not has_redundant:
        grid_cell_count = int(
            math.ceil(_cfg_float(cfg, "DIE_W_um") / _cfg_float(cfg, "PITCH_c_um"))
            * math.ceil(_cfg_float(cfg, "DIE_L_um") / _cfg_float(cfg, "PITCH_r_um"))
        )
        if grid_cell_count <= distance_transform_max_cells:
            return interface_particle_yield_from_critical_bitmap(
                cfg,
                pad_bitmap_collection,
            )

    critical_pad_coords_um = (
        _pad_coords_for_mask(pad_bitmap_collection, critical_pad_bitmap)
        if has_critical
        else np.empty((0, 2), dtype=np.float64)
    )
    (
        redundant_pad_coords_um,
        redundant_group_ids,
        tolerated_mechanical,
    ) = _redundant_pad_group_data(pad_bitmap_collection)

    avg_fatal_particles, info = _fatal_particle_integral_from_pad_layout(
        cfg,
        critical_pad_coords_um,
        redundant_pad_coords_um,
        redundant_group_ids,
        tolerated_mechanical,
    )
    particle_yield = float(np.exp(-avg_fatal_particles))
    D0 = _cfg_float(cfg, "D0")
    effective_critical_area_um2 = (
        avg_fatal_particles / D0 if D0 > 0.0 else 0.0
    )
    info.update(
        {
            "avg_fatal_particles": float(avg_fatal_particles),
            "effective_critical_area_um2": float(effective_critical_area_um2),
            "num_critical_pads": int(critical_pad_coords_um.shape[0]),
            "num_redundant_pads": int(redundant_pad_coords_um.shape[0]),
            "redundant_group_count": int(tolerated_mechanical.shape[0]),
        }
    )
    return particle_yield, info


def stack_defect_yield_calculator(
    cfg_dict: dict,
    die_stack,
):
    """
    Calculate D2W particle yield for every interface and write into die_stack.
    """
    model_cache = {}
    for interface_name, cfg in cfg_dict.items():
        pad_bitmap_collection = die_stack.interfaces.pad_bitmap_collection_dict[interface_name]
        cache_key = _particle_model_cache_key(cfg, pad_bitmap_collection)
        if cache_key in model_cache:
            particle_yield, info, source_interface = model_cache[cache_key]
            info = dict(info)
            info["cache_hit"] = True
            info["cache_source_interface"] = source_interface
        else:
            particle_yield, info = interface_particle_yield_from_pad_layout(
                cfg,
                pad_bitmap_collection,
            )
            info = dict(info)
            info["cache_hit"] = False
            info["cache_source_interface"] = interface_name
            model_cache[cache_key] = (
                float(particle_yield),
                dict(info),
                interface_name,
            )
        die_stack.die_yield_per_interface_dict[interface_name]['particle'] = particle_yield
        die_stack.interfaces.failure_params_dict[interface_name][
            "particle_model_info"
        ] = info
