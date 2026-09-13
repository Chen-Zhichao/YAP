#!/usr/bin/env python3
"""Audit the December-2025 Fig. 19 block-placement semantics.

This is a placement-only audit.  It does not import historical yield results or
use the old generator to produce a new reproduction result.  Instead, it
reimplements the relevant block-ordering and one-to-one pairing statements from
the uploaded source and records the consequences of its local/global-ID mixup.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "legacy_placement_audit.json")
    return parser.parse_args()


def legacy_center_redundant_order(grid_size: int) -> np.ndarray:
    """Return assign_pad_blocks('center') order for a 100%-redundant die."""
    ids = np.arange(grid_size * grid_size)
    rows = ids // grid_size
    cols = ids % grid_size
    rings = np.minimum.reduce(
        [rows, cols, grid_size - 1 - rows, grid_size - 1 - cols]
    )
    mesh = (rows % 3 == 0) | (cols % 3 == 0)
    mesh_ids = ids[mesh]
    gaps = np.setdiff1d(ids, mesh_ids)
    # This is the exact old fallback for mode == 'center'.  np.argsort's
    # default tie ordering is intentionally retained.
    gaps = gaps[np.argsort(-rings[gaps])]
    return np.concatenate((mesh_ids, gaps))


def legacy_neighbor_indices(
    grid_size: int, distance_blocks: int, block_order: np.ndarray
) -> dict[int, list[int]]:
    """KDTree-equivalent candidate indices in the old stable sorting order."""
    local_index = {int(block_id): index for index, block_id in enumerate(block_order)}
    radius = int(math.ceil(distance_blocks + 0.5))
    adjacency: dict[int, list[int]] = {}
    for main_raw in block_order:
        main = int(main_raw)
        row, col = divmod(main, grid_size)
        candidates: list[tuple[int, float]] = []
        for other_row in range(max(0, row - radius), min(grid_size, row + radius + 1)):
            for other_col in range(max(0, col - radius), min(grid_size, col + radius + 1)):
                distance = math.hypot(other_row - row, other_col - col)
                if distance_blocks - 0.1 < distance <= distance_blocks + 0.5:
                    candidates.append(
                        (local_index[other_row * grid_size + other_col], distance)
                    )
        # np.setdiff1d first sorted the KDTree-local indices.  Python's stable
        # sort then preserved that order among equal distances.
        candidates.sort(key=lambda item: (item[1], item[0]))
        adjacency[main] = [index for index, _ in candidates]
    return adjacency


def pair_once(
    *,
    grid_size: int,
    block_order: np.ndarray,
    adjacency: dict[int, list[int]],
    seed: int,
    preserve_old_id_bug: bool,
) -> list[tuple[int, int]]:
    """Run the old randomized greedy loop, optionally correcting its ID test."""
    target_pairs = grid_size * grid_size // 2
    used = np.zeros(grid_size * grid_size, dtype=bool)
    pairs: list[tuple[int, int]] = []
    rng = np.random.RandomState(seed)
    for main_raw in rng.permutation(block_order):
        main = int(main_raw)
        if used[main]:
            continue
        for neighbor_index in adjacency[main]:
            copy = int(block_order[neighbor_index])
            tested_id = neighbor_index if preserve_old_id_bug else copy
            if not used[tested_id]:
                used[main] = True
                used[copy] = True
                pairs.append((main, copy))
                break
        if len(pairs) >= target_pairs:
            break
    return pairs


def summarize_runs(
    *,
    grid_size: int,
    distance_blocks: int,
    block_order: np.ndarray,
    adjacency: dict[int, list[int]],
    samples: int,
    preserve_old_id_bug: bool,
) -> dict:
    target_pairs = grid_size * grid_size // 2
    pair_counts = []
    unique_counts = []
    duplicate_uses = []
    for seed in range(samples):
        pairs = pair_once(
            grid_size=grid_size,
            block_order=block_order,
            adjacency=adjacency,
            seed=seed,
            preserve_old_id_bug=preserve_old_id_bug,
        )
        flattened = [block_id for pair in pairs for block_id in pair]
        pair_counts.append(len(pairs))
        unique_counts.append(len(set(flattened)))
        duplicate_uses.append(len(flattened) - len(set(flattened)))
    return {
        "target_pair_count": target_pairs,
        "completion_count": int(sum(count == target_pairs for count in pair_counts)),
        "completion_fraction": float(np.mean(np.asarray(pair_counts) == target_pairs)),
        "pair_count": {
            "minimum": int(np.min(pair_counts)),
            "mean": float(np.mean(pair_counts)),
            "maximum": int(np.max(pair_counts)),
        },
        "unique_physical_blocks_used": {
            "minimum": int(np.min(unique_counts)),
            "mean": float(np.mean(unique_counts)),
            "maximum": int(np.max(unique_counts)),
        },
        "duplicate_physical_block_uses": {
            "minimum": int(np.min(duplicate_uses)),
            "mean": float(np.mean(duplicate_uses)),
            "maximum": int(np.max(duplicate_uses)),
        },
    }


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")

    cases = []
    for die_size_um in (10000, 3200):
        grid_size = die_size_um // 200
        block_order = legacy_center_redundant_order(grid_size)
        for spacing_um in (200, 400, 600, 800):
            distance_blocks = spacing_um // 200
            adjacency = legacy_neighbor_indices(grid_size, distance_blocks, block_order)
            cases.append(
                {
                    "die_size_um": die_size_um,
                    "block_dim_um": 200,
                    "grid_shape": [grid_size, grid_size],
                    "spacing_um": spacing_um,
                    "distance_blocks": distance_blocks,
                    "block_order_is_natural_id_order": bool(
                        np.array_equal(block_order, np.arange(grid_size * grid_size))
                    ),
                    "historical_local_global_id_mixup": summarize_runs(
                        grid_size=grid_size,
                        distance_blocks=distance_blocks,
                        block_order=block_order,
                        adjacency=adjacency,
                        samples=args.samples,
                        preserve_old_id_bug=True,
                    ),
                    "same_greedy_algorithm_with_id_test_corrected": summarize_runs(
                        grid_size=grid_size,
                        distance_blocks=distance_blocks,
                        block_order=block_order,
                        adjacency=adjacency,
                        samples=args.samples,
                        preserve_old_id_bug=False,
                    ),
                }
            )

    output = {
        "audit_scope": "December-2025 random main-block and dedicated replica-block placement",
        "samples_per_case": args.samples,
        "paper_constraints": {
            "die_size_um": [10000, 10000],
            "die_size_basis": "Table I baseline; Fig. 19 does not state an override",
            "block_dim_um": [200, 200],
            "physical_redundant_ratio": 1.0,
            "main_replica_mapping": "1:1",
            "target_main_fraction": 0.5,
            "target_replica_fraction": 0.5,
        },
        "historical_algorithm": {
            "main_placement": "np.random.permutation of assign_pad_blocks('center') redundant block order",
            "replica_search_band_blocks": "distance - 0.1 < Euclidean distance <= distance + 0.5",
            "replica_choice": "nearest candidate; KDTree-local index breaks equal-distance ties",
            "identified_bug": (
                "availability is tested with the KDTree-local neighbor index, but the selected copy "
                "and used-set updates use the global block ID"
            ),
        },
        "interpretation": (
            "The historical implementation can report the requested N/2 pairs while reusing physical "
            "blocks. Correcting only the local/global-ID test makes the randomized greedy algorithm "
            "generally unable to form a complete non-overlapping N/2 pairing. A maximum-matching "
            "algorithm fixes completeness but is not placement-equivalent to the paper-era code."
        ),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
