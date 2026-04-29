#!/usr/bin/env python3
"""
Helpers for syncing bump-map names across the two sides of one physical interface.

The two `.bmap` files may use different absolute coordinates and different line order,
but they share the same occupied logical pad-grid positions. We use normalized
grid ranks `(x_rank, y_rank)` as the stable join key.
"""

from __future__ import annotations


def _grid_key_map(entries: list[list[str]]) -> dict[tuple[int, int], tuple[str, str]]:
    points = [(float(parts[2]), float(parts[3])) for parts in entries]
    x_values = sorted({x for x, _ in points})
    y_values = sorted({y for _, y in points})
    x_rank = {value: idx for idx, value in enumerate(x_values)}
    y_rank = {value: idx for idx, value in enumerate(y_values)}

    key_to_names: dict[tuple[int, int], tuple[str, str]] = {}
    for parts in entries:
        key = (x_rank[float(parts[2])], y_rank[float(parts[3])])
        key_to_names[key] = (parts[4], parts[5])
    return key_to_names


def sync_names_by_normalized_grid(
    source_entries: list[list[str]],
    target_entries: list[list[str]],
) -> list[list[str]]:
    source_map = _grid_key_map(source_entries)

    target_points = [(float(parts[2]), float(parts[3])) for parts in target_entries]
    x_values = sorted({x for x, _ in target_points})
    y_values = sorted({y for _, y in target_points})
    x_rank = {value: idx for idx, value in enumerate(x_values)}
    y_rank = {value: idx for idx, value in enumerate(y_values)}

    rewritten: list[list[str]] = []
    for parts in target_entries:
        key = (x_rank[float(parts[2])], y_rank[float(parts[3])])
        if key not in source_map:
            raise KeyError(f"Target key {key} was not found in source bump map.")
        port, net = source_map[key]
        rewritten.append(parts[:4] + [port, net])

    if len(rewritten) != len(source_entries):
        raise ValueError(
            "Source/target bump maps have different entry counts; cannot sync names safely."
        )
    return rewritten
