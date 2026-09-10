#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""One-level Cu-recess yield calculator for W2W hybrid bonding."""

import hashlib
import re

import numpy as np
from scipy.special import ndtr

from debond import debond_dishing_intervals_from_coords


_PG_NET_RE = re.compile(
    r"(^|_)(vdd|vss|vpp|vddq|vddql|gnd|vcc)($|_)",
    re.IGNORECASE,
)
_MECHANICAL_MODEL_CACHE = {}


def _is_non_signal_shared_net(net: str) -> bool:
    lowered = str(net).lower()
    return "dummy" in lowered or _PG_NET_RE.search(str(net)) is not None


def _mechanical_model_cache_key(cfg, pad_bitmap_collection):
    hasher = hashlib.sha1()
    for key in (
        "CRITICAL_PAD_BITMAP",
        "REDUNDANT_PAD_BITMAP",
        "redundant_group_id_per_pad",
        "redundant_tolerated_mechanical_failures",
    ):
        if key not in pad_bitmap_collection:
            continue
        arr = np.asarray(pad_bitmap_collection[key])
        hasher.update(str(arr.shape).encode("utf-8"))
        if arr.dtype == np.bool_:
            hasher.update(np.packbits(arr.reshape(-1)).tobytes())
        else:
            hasher.update(np.ascontiguousarray(arr).tobytes())

    for key in (
        "TOP_DISH_MEAN_nm",
        "TOP_DISH_STD_nm",
        "BOT_DISH_MEAN_nm",
        "BOT_DISH_STD_nm",
        "PITCH_r_um",
        "PITCH_c_um",
        "PAD_ARR_ROW",
        "PAD_ARR_COL",
    ):
        hasher.update(f"{key}={getattr(cfg, key, None)};".encode("utf-8"))
    return hasher.hexdigest()


def cu_recess_pad_pass_probabilities(mu, lower_limits, upper_limits, sigma):
    """Return independent one-level Cu-recess survival probabilities."""
    lower_limits = np.asarray(lower_limits, dtype=np.float64).reshape(-1)
    upper_limits = np.asarray(upper_limits, dtype=np.float64).reshape(-1)
    if lower_limits.shape != upper_limits.shape:
        raise ValueError("lower_limits and upper_limits must have the same shape.")
    if np.any(lower_limits > upper_limits):
        raise ValueError("Each Cu-recess lower limit must not exceed its upper limit.")

    sigma = float(sigma)
    if sigma < 0.0:
        raise ValueError("Cu-recess standard deviation must be non-negative.")
    if sigma == 0.0:
        return ((mu >= lower_limits) & (mu <= upper_limits)).astype(np.float64)

    pass_probs = (
        ndtr((upper_limits - float(mu)) / sigma)
        - ndtr((lower_limits - float(mu)) / sigma)
    )
    return np.clip(pass_probs, 0.0, 1.0)


def _independent_all_pass_yield(pass_probs):
    pass_probs = np.asarray(pass_probs, dtype=np.float64).reshape(-1)
    if pass_probs.size == 0:
        return 1.0
    if np.any(pass_probs <= 0.0):
        return 0.0
    return float(np.exp(np.sum(np.log(np.clip(pass_probs, 1e-300, 1.0)))))


def _at_most_k_failures_yield(pass_probs, tolerated_failures):
    pass_probs = np.asarray(pass_probs, dtype=np.float64).reshape(-1)
    tolerated_failures = int(tolerated_failures)
    if pass_probs.size == 0:
        return 1.0
    if tolerated_failures < 0:
        return 0.0
    if tolerated_failures >= pass_probs.size:
        return 1.0
    if tolerated_failures == 0:
        return _independent_all_pass_yield(pass_probs)

    pmf = np.zeros(tolerated_failures + 1, dtype=np.float64)
    pmf[0] = 1.0
    for fail_prob in np.clip(1.0 - pass_probs, 0.0, 1.0):
        pass_prob = 1.0 - fail_prob
        next_pmf = pmf * pass_prob
        next_pmf[1:] += pmf[:-1] * fail_prob
        pmf = next_pmf
    return float(np.clip(np.sum(pmf), 0.0, 1.0))


def _redundant_pad_group_yield(
    redundant_flat_idx,
    redundant_pass_probs,
    pad_bitmap_collection,
):
    redundant_flat_idx = np.asarray(redundant_flat_idx, dtype=np.int64).reshape(-1)
    redundant_pass_probs = np.asarray(
        redundant_pass_probs,
        dtype=np.float64,
    ).reshape(-1)
    if redundant_flat_idx.shape != redundant_pass_probs.shape:
        raise ValueError("Redundant pad indices and pass probabilities must match.")
    if redundant_flat_idx.size == 0:
        return 1.0

    group_id_per_pad = pad_bitmap_collection.get("redundant_group_id_per_pad")
    tolerated_failures = pad_bitmap_collection.get(
        "redundant_tolerated_mechanical_failures"
    )
    if group_id_per_pad is not None and tolerated_failures is not None:
        group_id_per_pad = np.asarray(group_id_per_pad, dtype=np.int64).reshape(-1)
        tolerated_failures = np.asarray(tolerated_failures, dtype=np.int64).reshape(-1)
        group_ids = group_id_per_pad[redundant_flat_idx]
        if np.any(group_ids < 0):
            bad = redundant_flat_idx[group_ids < 0][:5]
            raise ValueError(
                "Redundant pads are missing group IDs, e.g. flat indices "
                f"{bad.tolist()}."
            )

        order = np.argsort(group_ids, kind="mergesort")
        sorted_group_ids = group_ids[order]
        sorted_pass_probs = redundant_pass_probs[order]
        unique_group_ids, group_starts = np.unique(
            sorted_group_ids,
            return_index=True,
        )
        group_stops = np.r_[group_starts[1:], sorted_group_ids.size]

        redundant_yield = 1.0
        for group_id, start, stop in zip(
            unique_group_ids,
            group_starts,
            group_stops,
        ):
            if group_id >= tolerated_failures.size:
                raise ValueError(f"Redundant group ID {group_id} has no tolerance entry.")
            redundant_yield *= _at_most_k_failures_yield(
                sorted_pass_probs[start:stop],
                tolerated_failures[group_id],
            )
            if redundant_yield <= 0.0:
                return 0.0
        return float(np.clip(redundant_yield, 0.0, 1.0))

    redundant_yield = 1.0
    redundant_nets = pad_bitmap_collection.get(
        "redundant_net_to_1d_physical_mask",
        {},
    )
    criticality_info = pad_bitmap_collection.get("criticality_info", {})
    for net, physical_mask in redundant_nets.items():
        if _is_non_signal_shared_net(net):
            continue
        physical_mask = np.asarray(physical_mask, dtype=np.int64).reshape(-1)
        physical_mask = physical_mask[physical_mask >= 0]
        if physical_mask.size == 0:
            continue

        selected_pos = np.searchsorted(redundant_flat_idx, physical_mask)
        in_range = selected_pos < redundant_flat_idx.size
        matches = np.zeros_like(in_range, dtype=bool)
        matches[in_range] = (
            redundant_flat_idx[selected_pos[in_range]] == physical_mask[in_range]
        )
        if not np.all(matches):
            missing = physical_mask[~matches][:5]
            raise ValueError(
                f"Redundant net '{net}' references pads not present in the "
                f"redundant pad bitmap, e.g. {missing.tolist()}."
            )
        redundant_yield *= _at_most_k_failures_yield(
            redundant_pass_probs[selected_pos],
            criticality_info[net]["tolerated_mechanical_failures"],
        )
        if redundant_yield <= 0.0:
            return 0.0
    return float(np.clip(redundant_yield, 0.0, 1.0))


def stack_stress_yield_calculator(cfg_dict: dict, waf_stack):
    """Calculate one representative die's mechanical yield per interface."""
    for interface_name, cfg in cfg_dict.items():
        cfg.num_dies_per_wafer = waf_stack.num_dies_per_wafer
        interface = waf_stack.interfaces.interface_dict[interface_name]
        pad_bitmap_collection = (
            waf_stack.interfaces.pad_bitmap_collection_dict[interface_name]
        )
        cache_key = _mechanical_model_cache_key(cfg, pad_bitmap_collection)
        stress_yield_array = np.full(interface.num_dies, np.nan, dtype=np.float64)
        if cache_key in _MECHANICAL_MODEL_CACHE:
            stress_yield_array[:] = _MECHANICAL_MODEL_CACHE[cache_key]
            waf_stack.die_yield_list_per_interface_dict[interface_name][
                "mechanical"
            ] = stress_yield_array
            continue

        critical_mask = np.asarray(
            pad_bitmap_collection["CRITICAL_PAD_BITMAP"],
            dtype=bool,
        ).reshape(-1)
        redundant_mask = np.asarray(
            pad_bitmap_collection["REDUNDANT_PAD_BITMAP"],
            dtype=bool,
        ).reshape(-1).copy()
        for net, physical_mask in pad_bitmap_collection.get(
            "redundant_net_to_1d_physical_mask",
            {},
        ).items():
            if not _is_non_signal_shared_net(net):
                continue
            physical_mask = np.asarray(physical_mask, dtype=np.int64).reshape(-1)
            physical_mask = physical_mask[
                (physical_mask >= 0) & (physical_mask < redundant_mask.size)
            ]
            redundant_mask[physical_mask] = False

        selected_mask = critical_mask | redundant_mask
        selected_flat_idx = np.flatnonzero(selected_mask)
        critical_flat_idx = np.flatnonzero(critical_mask)
        redundant_flat_idx = np.flatnonzero(redundant_mask)
        if selected_flat_idx.size == 0:
            stress_yield_array[:] = 1.0
            waf_stack.die_yield_list_per_interface_dict[interface_name][
                "mechanical"
            ] = stress_yield_array
            continue

        if interface.base_pad_coords is None:
            raise ValueError(
                f"Interface '{interface_name}' does not have pad coordinates. "
                "Cu mechanical yield modeling needs coordinates for critical "
                "and redundant pads."
            )
        selected_coords = np.asarray(
            interface.base_pad_coords,
            dtype=np.float64,
        )[selected_flat_idx]
        if not np.all(np.isfinite(selected_coords)):
            bad = selected_flat_idx[
                ~np.all(np.isfinite(selected_coords), axis=1)
            ][:5]
            raise ValueError(
                f"Interface '{interface_name}' has selected critical/redundant "
                f"pads with invalid coordinates, e.g. flat indices {bad.tolist()}."
            )

        intervals = debond_dishing_intervals_from_coords(cfg, selected_coords)
        upper_limits = np.clip(-intervals[:, 0] * 2.0, a_min=None, a_max=0.0)
        lower_limits = np.clip(-intervals[:, 1] * 2.0, a_min=None, a_max=0.0)
        mu = float(cfg.TOP_DISH_MEAN_nm + cfg.BOT_DISH_MEAN_nm)
        sigma = float(np.hypot(cfg.TOP_DISH_STD_nm, cfg.BOT_DISH_STD_nm))

        critical_pos = np.searchsorted(selected_flat_idx, critical_flat_idx)
        critical_probs = cu_recess_pad_pass_probabilities(
            mu,
            lower_limits[critical_pos],
            upper_limits[critical_pos],
            sigma,
        )
        critical_yield = _independent_all_pass_yield(critical_probs)

        redundant_pos = np.searchsorted(selected_flat_idx, redundant_flat_idx)
        redundant_probs = cu_recess_pad_pass_probabilities(
            mu,
            lower_limits[redundant_pos],
            upper_limits[redundant_pos],
            sigma,
        )
        redundant_yield = _redundant_pad_group_yield(
            redundant_flat_idx,
            redundant_probs,
            pad_bitmap_collection,
        )

        stress_die_yield = float(critical_yield * redundant_yield)
        _MECHANICAL_MODEL_CACHE[cache_key] = stress_die_yield
        stress_yield_array[:] = stress_die_yield
        waf_stack.die_yield_list_per_interface_dict[interface_name][
            "mechanical"
        ] = stress_yield_array
