#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class AssignmentPartitions:
    critical_indices: list[int]
    redundant_pairs: list[tuple[int, int]]
    pg_indices: list[int]
    dummy_indices: list[int]


def build_grid_decorated(rows: list[int], cols: list[int]) -> list[tuple[int, int, int]]:
    return [(idx, row, col) for idx, (row, col) in enumerate(zip(rows, cols))]


def build_random_rng(seed_key: str) -> random.Random:
    seed = int.from_bytes(hashlib.sha256(seed_key.encode("utf-8")).digest()[:8], "big")
    return random.Random(seed)


def select_adjacent_random_pairs(
    index_to_row_col: dict[int, tuple[int, int]],
    num_pairs: int,
    seed_key: str,
) -> list[tuple[int, int]]:
    if num_pairs <= 0:
        return []

    pos_to_index = {pos: idx for idx, pos in index_to_row_col.items()}
    edges: list[tuple[int, int]] = []
    for idx, (row, col) in index_to_row_col.items():
        for neighbor_pos in ((row, col + 1), (row + 1, col)):
            neighbor_idx = pos_to_index.get(neighbor_pos)
            if neighbor_idx is not None:
                edges.append((idx, neighbor_idx))

    rng = build_random_rng(seed_key)
    rng.shuffle(edges)

    used: set[int] = set()
    selected: list[tuple[int, int]] = []
    for first, second in edges:
        if first in used or second in used:
            continue
        selected.append((first, second))
        used.add(first)
        used.add(second)
        if len(selected) == num_pairs:
            return selected

    raise ValueError(
        f"Unable to find {num_pairs} disjoint adjacent redundant pairs; "
        f"only found {len(selected)}."
    )


def build_random_partitions(
    row_major_indices: list[int],
    index_to_row_col: dict[int, tuple[int, int]],
    critical_count: int,
    redundant_count: int,
    pg_count: int,
    dummy_count: int,
    seed_key: str,
) -> AssignmentPartitions:
    redundant_pairs = select_adjacent_random_pairs(
        index_to_row_col=index_to_row_col,
        num_pairs=redundant_count // 2,
        seed_key=seed_key,
    )
    redundant_flat = [idx for pair in redundant_pairs for idx in pair]
    used = set(redundant_flat)

    remaining = [idx for idx in row_major_indices if idx not in used]
    rng = build_random_rng(seed_key + "::remaining")
    rng.shuffle(remaining)

    critical_indices = remaining[:critical_count]
    pg_start = critical_count
    pg_end = pg_start + pg_count
    pg_indices = remaining[pg_start:pg_end]
    dummy_indices = remaining[pg_end:pg_end + dummy_count]

    expected_total = critical_count + redundant_count + pg_count + dummy_count
    observed_total = (
        len(critical_indices)
        + len(redundant_flat)
        + len(pg_indices)
        + len(dummy_indices)
    )
    if observed_total != expected_total:
        raise AssertionError(
            f"Random partition mismatch: expected {expected_total}, got {observed_total}"
        )

    return AssignmentPartitions(
        critical_indices=critical_indices,
        redundant_pairs=redundant_pairs,
        pg_indices=pg_indices,
        dummy_indices=dummy_indices,
    )
