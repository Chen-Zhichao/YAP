#!/usr/bin/env python3
"""Compact descriptions for very large regular pad arrays."""

import math

import numpy as np


REGULAR_ALL_CRITICAL = "regular_all_critical"
REGULAR_CENTER_IO = "regular_center_io"
COMPACT_REGULAR_LAYOUTS = {REGULAR_ALL_CRITICAL, REGULAR_CENTER_IO}


def _cfg_value(cfg, key, default=None):
    if isinstance(cfg, dict):
        value = cfg.get(key, default)
    elif hasattr(cfg, "get"):
        value = cfg.get(key, default)
    else:
        value = getattr(cfg, key, default)
    return default if value in (None, "None") else value


def _category_counts(total_pads, critical_fraction, power_ground_fraction):
    dummy_fraction = 1.0 - critical_fraction - power_ground_fraction
    desired = (
        np.asarray(
            [critical_fraction, power_ground_fraction, dummy_fraction],
            dtype=np.float64,
        )
        * total_pads
    )
    counts = np.floor(desired).astype(np.int64)
    remaining = total_pads - int(np.sum(counts))
    order = sorted(
        range(3),
        key=lambda index: (desired[index] - counts[index], desired[index]),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    return tuple(int(value) for value in counts)


def _boundary(span_w_um, span_l_um, area_fraction):
    scale = math.sqrt(area_fraction)
    half_w_um = span_w_um * scale / 2.0
    half_l_um = span_l_um * scale / 2.0
    return np.asarray(
        [
            [-half_w_um, half_l_um],
            [half_w_um, half_l_um],
            [-half_w_um, -half_l_um],
            [half_w_um, -half_l_um],
        ],
        dtype=np.float64,
    )


def is_compact_regular_layout(value) -> bool:
    if isinstance(value, dict):
        kind = value.get("layout_kind")
    else:
        kind = getattr(value, "LAYOUT_KIND", None)
    return str(kind).lower() in COMPACT_REGULAR_LAYOUTS


def compact_regular_geometry(cfg) -> dict:
    if not is_compact_regular_layout(cfg):
        raise ValueError("Configuration is not a compact regular layout.")
    pitch_r_um = float(cfg.PITCH_r_um)
    pitch_c_um = float(cfg.PITCH_c_um)
    if pitch_r_um <= 0.0 or pitch_c_um <= 0.0:
        raise ValueError("Regular-layout pitches must be positive.")

    rows = int(math.floor(float(cfg.DIE_L_um) / pitch_r_um))
    cols = int(math.floor(float(cfg.DIE_W_um) / pitch_c_um))
    if rows <= 0 or cols <= 0:
        raise ValueError("Regular layout must contain at least one pad.")
    span_l_um = (rows - 1) * pitch_r_um
    span_w_um = (cols - 1) * pitch_c_um
    layout_kind = str(_cfg_value(cfg, "LAYOUT_KIND")).lower()
    if layout_kind == REGULAR_ALL_CRITICAL:
        critical_fraction = 1.0
        power_ground_fraction = 0.0
    else:
        critical_fraction = float(_cfg_value(cfg, "CRITICAL_PAD_FRACTION", 0.30))
        power_ground_fraction = float(
            _cfg_value(cfg, "POWER_GROUND_PAD_FRACTION", 0.50)
        )
    dummy_fraction = float(
        _cfg_value(
            cfg,
            "DUMMY_PAD_FRACTION",
            1.0 - critical_fraction - power_ground_fraction,
        )
    )
    if (
        critical_fraction < 0.0
        or power_ground_fraction < 0.0
        or dummy_fraction < 0.0
        or not math.isclose(
            critical_fraction + power_ground_fraction + dummy_fraction,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Compact pad fractions must be nonnegative and sum to one.")
    total_pads = rows * cols
    critical_count, power_ground_count, dummy_count = _category_counts(
        total_pads,
        critical_fraction,
        power_ground_fraction,
    )
    active_fraction = critical_fraction + power_ground_fraction
    critical_boundary = _boundary(span_w_um, span_l_um, critical_fraction)
    active_boundary = _boundary(span_w_um, span_l_um, active_fraction)
    return {
        "layout_kind": layout_kind,
        "rows": rows,
        "cols": cols,
        "pitch_r_um": pitch_r_um,
        "pitch_c_um": pitch_c_um,
        "pad_radius_um": float(cfg.PAD_TOP_R_um),
        "pad_array_l_um": span_l_um,
        "pad_array_w_um": span_w_um,
        "critical_fraction": critical_fraction,
        "power_ground_fraction": power_ground_fraction,
        "dummy_fraction": dummy_fraction,
        "critical_region_l_um": span_l_um * math.sqrt(critical_fraction),
        "critical_region_w_um": span_w_um * math.sqrt(critical_fraction),
        "num_critical_pads": critical_count,
        "num_power_ground_pads": power_ground_count,
        "num_dummy_pads": dummy_count,
        "critical_pad_boundary_coords": critical_boundary,
        "active_pad_boundary_coords": active_boundary,
    }


def compact_regular_collection(cfg) -> dict:
    """Return geometry metadata without allocating per-pad arrays."""
    geometry = compact_regular_geometry(cfg)
    return {
        **geometry,
        "num_redundant_pads": 0,
    }


def require_modeling_only(cfg, mode: str) -> None:
    if is_compact_regular_layout(cfg) and "simulation" in str(mode).lower():
        raise ValueError(
            "Compact regular layouts are supported by "
            "calculator_main only; pad-level simulation requires an explicit layout."
        )


def uniform_density_integral_um2(cfg) -> float:
    """Integrate the D0-to-D1 edge-density model over a rectangular die."""
    width = float(cfg.DIE_W_um)
    length = float(cfg.DIE_L_um)
    d0 = float(cfg.D0)
    d1 = float(getattr(cfg, "D1", d0))
    area = width * length
    edge_width = min(
        max(float(getattr(cfg, "EDGE_REGION_WIDTH_um", 0.0)), 0.0),
        width / 2.0,
        length / 2.0,
    )
    edge_weight_area = (
        (width + length) * edge_width - 4.0 * edge_width * edge_width / 3.0
        if edge_width > 0.0 and d1 > d0
        else 0.0
    )
    return d0 * area + max(d1 - d0, 0.0) * edge_weight_area


def minimum_void_covers_entire_die(cfg) -> bool:
    """Whether the smallest main void reaches a pad from every die point."""
    geometry = compact_regular_geometry(cfg)
    if geometry["critical_fraction"] < 1.0:
        return False
    margin_x = (float(cfg.DIE_W_um) - geometry["pad_array_w_um"]) / 2.0
    margin_y = (float(cfg.DIE_L_um) - geometry["pad_array_l_um"]) / 2.0
    max_dx = max(geometry["pitch_c_um"] / 2.0, margin_x)
    max_dy = max(geometry["pitch_r_um"] / 2.0, margin_y)
    farthest_to_center_um = math.hypot(max_dx, max_dy)
    minimum_void_radius_um = float(cfg.k_r0) * math.sqrt(float(cfg.t_0))
    return minimum_void_radius_um + float(cfg.PAD_TOP_R_um) >= farthest_to_center_um
