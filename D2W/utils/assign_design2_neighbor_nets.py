#!/usr/bin/env python3
"""
Retag design_2 Compute_Large bump-map names for neighbor-chiplet traffic.

Rules
-----
- Keep the existing bump-kind placement and ratios unchanged.
- For non-c0 ratios, 50% of critical bumps become internal links.
- For c0_r50_pg50_dm0, 25% of redundant nets become internal links.
- Each compute die splits its internal-link share across its two neighboring compute dies.
- Remaining critical bumps become chiplet-local external critical signals.
- Remaining redundant bumps stay chiplet-local external redundant signals.
- PG and dummy bumps are prefixed with the owning chiplet id.
- Regenerate per-file criticality and one shared-net summary per ratio folder.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
from collections import Counter
from itertools import combinations
from itertools import cycle
from pathlib import Path

import bmap_grid_sync as bgs
import generate_criticality as gc
from bump_assignment_utils import build_random_partitions
import yaml


RATIO_RE = re.compile(
    r"^c(?P<c>\d+(?:[.pd]\d+)?)_r(?P<r>\d+(?:[.pd]\d+)?)_pg_?(?P<pg>\d+(?:[.pd]\d+)?)_dm(?P<dm>\d+(?:[.pd]\d+)?)$"
)
COMPUTE_FILE_RE = re.compile(r"^Compute_Large_(?P<id>\d+)_From_Substrate_Organic\.bmap$")
CHIPLET_PREFIX_RE = re.compile(r"^chiplet_(?P<id>\d+)_(?P<rest>.+)$")
PG_TOKENS = ("VDD", "VSS", "VPP", "VDDQ", "VDDQL")
PG_PATTERN = ("VDD", "VSS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign design_2 compute-die neighbor/shared bump names."
    )
    parser.add_argument(
        "--design-root",
        type=Path,
        default=Path("D2W/input/design_2"),
        help="Root folder for design_2 ratio directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned updates without overwriting files.",
    )
    return parser.parse_args()


def read_entries(path: Path) -> list[list[str]]:
    entries: list[list[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            parts = raw_line.strip().split()
            if not parts:
                continue
            if len(parts) != 6:
                raise ValueError(
                    f"{path}:{line_num} has {len(parts)} columns; expected 6."
                )
            entries.append(parts)
    if not entries:
        raise ValueError(f"No entries found in {path}.")
    return entries


def write_entries(path: Path, entries: list[list[str]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for parts in entries:
            f.write(" ".join(parts) + "\n")
    tmp_path.replace(path)


def parse_ratio_from_path(path: Path) -> tuple[float, float, float, float]:
    for part in path.parts:
        match = RATIO_RE.match(part)
        if match:
            return tuple(
                float(match.group(name).replace("p", ".").replace("d", ".")) / 100.0
                for name in ("c", "r", "pg", "dm")
            )
    raise ValueError(f"Could not find ratio folder in path: {path}")


def realize_counts(total: int, ratios: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    critical_ratio, redundant_ratio, pg_ratio, dummy_ratio = ratios
    desired = [
        critical_ratio * total,
        redundant_ratio * total,
        pg_ratio * total,
        dummy_ratio * total,
    ]
    counts = [int(value) for value in desired]
    remaining = total - sum(counts)
    order = sorted(
        range(4),
        key=lambda idx: (desired[idx] - counts[idx], desired[idx]),
        reverse=True,
    )
    for idx in order[:remaining]:
        counts[idx] += 1

    if counts[1] % 2 == 1:
        best = None
        best_score = None
        for other_idx in (0, 2, 3):
            if counts[other_idx] > 0:
                for delta_rd, delta_other in ((1, -1), (-1, 1)):
                    candidate = counts.copy()
                    candidate[1] += delta_rd
                    candidate[other_idx] += delta_other
                    if candidate[1] < 0 or candidate[other_idx] < 0:
                        continue
                    score = sum((candidate[i] - desired[i]) ** 2 for i in range(4))
                    if best_score is None or score < best_score:
                        best = candidate
                        best_score = score
        if best is None or best[1] % 2 == 1:
            raise ValueError(f"Unable to realize even redundant count for total={total}")
        counts = best

    return tuple(counts)


def infer_mode(path: Path) -> str:
    for part in path.parts:
        if part == "Center_IO":
            return "center"
        if part == "Edge_IO":
            return "edge"
        if part.startswith("Random_"):
            return "random"
    return "input"


def canonical_random_chiplet_type(path: Path) -> str:
    stem_parts = [part for part in path.stem.split("_") if not part.isdigit()]
    canonical_stem = "_".join(stem_parts)
    return re.sub(r"_+", "_", canonical_stem).strip("_")


def build_assignment_order(path: Path, entries: list[list[str]]) -> list[int]:
    mode = infer_mode(path)
    if mode == "input":
        return list(range(len(entries)))

    x_values = sorted({float(parts[2]) for parts in entries})
    y_values = sorted({float(parts[3]) for parts in entries}, reverse=True)
    x_to_col = {value: idx for idx, value in enumerate(x_values)}
    y_to_row = {value: idx for idx, value in enumerate(y_values)}

    decorated: list[tuple[int, int, int]] = []
    for idx, parts in enumerate(entries):
        decorated.append((idx, y_to_row[float(parts[3])], x_to_col[float(parts[2])]))

    if mode == "center":
        center_row = (len(y_values) - 1) / 2.0
        center_col = (len(x_values) - 1) / 2.0
        decorated.sort(
            key=lambda item: (
                max(abs(item[1] - center_row), abs(item[2] - center_col)),
                item[1],
                item[2],
            )
        )
    elif mode == "edge":
        max_row = len(y_values) - 1
        max_col = len(x_values) - 1
        decorated.sort(
            key=lambda item: (
                min(item[1], item[2], max_row - item[1], max_col - item[2]),
                item[1],
                item[2],
            )
        )
    else:
        decorated.sort(key=lambda item: (item[1], item[2]))
        random_group_key = (
            f"{path.parent.as_posix()}::{canonical_random_chiplet_type(path)}"
        )
        seed = int.from_bytes(
            hashlib.sha256(random_group_key.encode("utf-8")).digest()[:8], "big"
        )
        rng = random.Random(seed)
        rng.shuffle(decorated)

    return [idx for idx, _, _ in decorated]


def build_assignment_partitions(path: Path, entries: list[list[str]], counts: tuple[int, int, int, int]):
    mode = infer_mode(path)
    order = build_assignment_order(path, entries)
    critical_count, redundant_count, pg_count, dummy_count = counts
    if mode != "random":
        redundant_indices = order[critical_count : critical_count + redundant_count]
        return {
            "critical_indices": order[:critical_count],
            "redundant_pairs": list(zip(redundant_indices[0::2], redundant_indices[1::2])),
            "pg_indices": order[
                critical_count + redundant_count : critical_count + redundant_count + pg_count
            ],
            "dummy_indices": order[critical_count + redundant_count + pg_count :],
        }

    x_values = sorted({float(parts[2]) for parts in entries})
    y_values = sorted({float(parts[3]) for parts in entries}, reverse=True)
    x_to_col = {value: idx for idx, value in enumerate(x_values)}
    y_to_row = {value: idx for idx, value in enumerate(y_values)}
    index_to_row_col = {
        idx: (y_to_row[float(parts[3])], x_to_col[float(parts[2])])
        for idx, parts in enumerate(entries)
    }
    seed_key = f"{path.parent.as_posix()}::{canonical_random_chiplet_type(path)}"
    partitions = build_random_partitions(
        row_major_indices=order,
        index_to_row_col=index_to_row_col,
        critical_count=critical_count,
        redundant_count=redundant_count,
        pg_count=pg_count,
        dummy_count=dummy_count,
        seed_key=seed_key,
    )
    return {
        "critical_indices": partitions.critical_indices,
        "redundant_pairs": partitions.redundant_pairs,
        "pg_indices": partitions.pg_indices,
        "dummy_indices": partitions.dummy_indices,
    }


def strip_chiplet_prefix(name: str) -> str:
    match = CHIPLET_PREFIX_RE.match(name)
    if match:
        return match.group("rest")
    return name


def write_criticality(path: Path) -> None:
    net_counts: Counter[str] = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            parts = raw_line.strip().split()
            if len(parts) == 6:
                net_counts[parts[5]] += 1

    for profile in gc.ALL_PROFILES:
        out_path = gc.get_output_filename(path, profile=profile)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for net in sorted(net_counts):
                group_size = net_counts[net]
                tolerated_esd, tolerated_mech = gc.tolerated_failures_for_group(
                    group_size=group_size,
                    profile=profile,
                    net=net,
                )
                f.write(f"{net} {group_size} {tolerated_esd} {tolerated_mech}\n")
        tmp_path.replace(out_path)


def discover_compute_neighbor_pairs(design_root: Path) -> tuple[list[int], list[tuple[int, int]]]:
    defs_path = design_root / "generated_chiplet_definitions.3dbv"
    data = yaml.safe_load(defs_path.read_text(encoding="utf-8"))
    regions = data["ChipletDef"]["Substrate_Organic"]["regions"]

    boxes: dict[int, tuple[float, float, float, float]] = {}
    for region_name, region_data in regions.items():
        match = re.search(r"Compute_Large_(\d+)$", region_name)
        if not match:
            continue
        chiplet_id = int(match.group(1))
        coords = region_data["coords"]
        xs = [point[0] for point in coords]
        ys = [point[1] for point in coords]
        boxes[chiplet_id] = (min(xs), min(ys), max(xs), max(ys))

    chiplet_ids = sorted(boxes)
    if len(chiplet_ids) != 4:
        raise ValueError(f"Expected 4 compute dies in design_2, found {len(chiplet_ids)}")

    centers = {
        chiplet_id: ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        for chiplet_id, box in boxes.items()
    }
    x_values = sorted({round(center[0], 6) for center in centers.values()})
    y_values = sorted({round(center[1], 6) for center in centers.values()})
    if len(x_values) != 2 or len(y_values) != 2:
        raise ValueError(
            f"Expected a 2x2 compute grid, found x={x_values}, y={y_values}"
        )

    x_to_col = {value: idx for idx, value in enumerate(x_values)}
    y_to_row = {value: idx for idx, value in enumerate(y_values)}
    by_grid: dict[tuple[int, int], int] = {}
    for chiplet_id, (center_x, center_y) in centers.items():
        by_grid[(y_to_row[round(center_y, 6)], x_to_col[round(center_x, 6)])] = chiplet_id

    required_cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
    if set(by_grid) != required_cells:
        raise ValueError(f"Incomplete compute grid mapping: {by_grid}")

    pairs = sorted(
        {
            tuple(sorted((by_grid[(0, 0)], by_grid[(0, 1)]))),
            tuple(sorted((by_grid[(1, 0)], by_grid[(1, 1)]))),
            tuple(sorted((by_grid[(0, 0)], by_grid[(1, 0)]))),
            tuple(sorted((by_grid[(0, 1)], by_grid[(1, 1)]))),
        }
    )
    return chiplet_ids, pairs


def compute_pair_shared_counts(chiplet_ids: list[int], pairs: list[tuple[int, int]], shared_per_die: int) -> dict[tuple[int, int], int]:
    degree = Counter()
    for a, b in pairs:
        degree[a] += 1
        degree[b] += 1

    if any(degree[chiplet_id] != 2 for chiplet_id in chiplet_ids):
        raise ValueError(f"Expected each chiplet to have degree 2, got {dict(degree)}")

    base = shared_per_die // 2
    remainder = shared_per_die - (2 * base)
    if remainder not in (0, 1):
        raise ValueError(
            f"Unsupported shared-per-die value {shared_per_die} for degree-2 graph"
        )

    pair_counts = {pair: base for pair in pairs}
    if remainder == 0:
        return pair_counts

    sorted_pairs = sorted(pairs)

    def search(remaining: set[int], chosen: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if not remaining:
            return chosen.copy()
        node = min(remaining)
        for pair in sorted_pairs:
            a, b = pair
            if a == node or b == node:
                if a in remaining and b in remaining:
                    remaining.remove(a)
                    remaining.remove(b)
                    chosen.append(pair)
                    result = search(remaining, chosen)
                    if result is not None:
                        return result
                    chosen.pop()
                    remaining.add(a)
                    remaining.add(b)
        return None

    matching = search(set(chiplet_ids), [])
    if matching is None:
        raise ValueError("Could not find perfect matching for odd shared allocation.")

    for pair in matching:
        pair_counts[pair] += 1
    return pair_counts


def discover_compute_groups(design_root: Path) -> dict[str, dict[int, Path]]:
    groups: dict[str, dict[int, Path]] = {}
    for path in sorted(design_root.rglob("Compute_Large_*_From_Substrate_Organic.bmap")):
        match = COMPUTE_FILE_RE.match(path.name)
        if not match:
            continue
        ratio_name = next(part for part in path.parts if RATIO_RE.match(part))
        variant_key = str(path.parent.relative_to(design_root / ratio_name))
        groups.setdefault(f"{ratio_name}/{variant_key}", {})[int(match.group("id"))] = path

    for group_key, mapping in groups.items():
        if set(mapping) != {0, 1, 2, 3}:
            raise ValueError(f"Incomplete compute set under {group_key}: {sorted(mapping)}")
    return groups


def write_shared_summary(
    design_root: Path,
    ratio_name: str,
    pair_to_names: dict[tuple[int, int], list[str]],
    dry_run: bool,
) -> None:
    out_path = design_root / ratio_name / "Compute_Large_interchip_shared_nets.txt"
    if dry_run:
        return
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for pair in sorted(pair_to_names):
            a, b = pair
            f.write(f"[chiplet_{a}<->chiplet_{b}]\n")
            for name in pair_to_names[pair]:
                f.write(name + "\n")
            f.write("\n")
    tmp_path.replace(out_path)


def retag_group(
    group_paths: dict[int, Path],
    pair_counts: dict[tuple[int, int], int],
    use_redundant_shared: bool,
    dry_run: bool,
) -> None:
    entries_by_id = {chiplet_id: read_entries(path) for chiplet_id, path in group_paths.items()}
    counts_by_id = {
        chiplet_id: realize_counts(len(entries), parse_ratio_from_path(group_paths[chiplet_id]))
        for chiplet_id, entries in entries_by_id.items()
    }
    partitions_by_id = {
        chiplet_id: build_assignment_partitions(group_paths[chiplet_id], entries_by_id[chiplet_id], counts_by_id[chiplet_id])
        for chiplet_id in group_paths
    }

    shared_names_by_pair = {
        pair: [f"chiplet_{pair[0]}_chiplet_{pair[1]}_link_{idx:06d}" for idx in range(1, count + 1)]
        for pair, count in sorted(pair_counts.items())
    }

    neighbor_map: dict[int, list[int]] = {chiplet_id: [] for chiplet_id in group_paths}
    for a, b in sorted(pair_counts):
        neighbor_map[a].append(b)
        neighbor_map[b].append(a)
    for chiplet_id in neighbor_map:
        neighbor_map[chiplet_id].sort()

    for chiplet_id, path in group_paths.items():
        entries = entries_by_id[chiplet_id]
        partitions = partitions_by_id[chiplet_id]
        critical_indices = partitions["critical_indices"]
        redundant_pairs = partitions["redundant_pairs"]
        pg_indices = partitions["pg_indices"]
        dummy_indices = partitions["dummy_indices"]

        cursor = 0
        if use_redundant_shared:
            for neighbor_id in neighbor_map[chiplet_id]:
                pair = tuple(sorted((chiplet_id, neighbor_id)))
                names = shared_names_by_pair[pair]
                chunk_size = len(names)
                for (first_idx, second_idx), shared_name in zip(
                    redundant_pairs[cursor : cursor + chunk_size], names
                ):
                    for entry_idx in (first_idx, second_idx):
                        entries[entry_idx][4] = shared_name
                        entries[entry_idx][5] = shared_name
                cursor += chunk_size
            external_critical_indices = critical_indices
            external_redundant_pairs = redundant_pairs[cursor:]
        else:
            for neighbor_id in neighbor_map[chiplet_id]:
                pair = tuple(sorted((chiplet_id, neighbor_id)))
                names = shared_names_by_pair[pair]
                chunk_size = len(names)
                for entry_idx, shared_name in zip(
                    critical_indices[cursor : cursor + chunk_size], names
                ):
                    entries[entry_idx][4] = shared_name
                    entries[entry_idx][5] = shared_name
                cursor += chunk_size
            external_critical_indices = critical_indices[cursor:]
            external_redundant_pairs = redundant_pairs

        for external_idx, entry_idx in enumerate(external_critical_indices, start=1):
            name = f"chiplet_{chiplet_id}_ext_crit_{external_idx:06d}"
            entries[entry_idx][4] = name
            entries[entry_idx][5] = name

        for pair_idx, (first_idx, second_idx) in enumerate(external_redundant_pairs, start=1):
            name = f"chiplet_{chiplet_id}_ext_rd_{pair_idx:06d}"
            for entry_idx in (first_idx, second_idx):
                entries[entry_idx][4] = name
                entries[entry_idx][5] = name

        pg_name_iter = cycle(PG_PATTERN)
        for entry_idx in pg_indices:
            name = f"chiplet_{chiplet_id}_{next(pg_name_iter)}"
            entries[entry_idx][4] = name
            entries[entry_idx][5] = name

        for entry_idx in dummy_indices:
            name = f"chiplet_{chiplet_id}_dummy"
            entries[entry_idx][4] = name
            entries[entry_idx][5] = name

        if not dry_run:
            write_entries(path, entries)
            write_criticality(path)

        substrate_path = path.with_name(
            f"Substrate_Organic_To_Compute_Large_{chiplet_id}.bmap"
        )
        if substrate_path.exists():
            substrate_entries = bgs.sync_names_by_normalized_grid(
                entries,
                read_entries(substrate_path),
            )
            if not dry_run:
                write_entries(substrate_path, substrate_entries)
                write_criticality(substrate_path)


def main() -> None:
    args = parse_args()
    design_root = args.design_root.resolve()
    chiplet_ids, pairs = discover_compute_neighbor_pairs(design_root)
    groups = discover_compute_groups(design_root)

    ratio_to_shared_names: dict[str, dict[tuple[int, int], list[str]]] = {}

    for group_key, group_paths in sorted(groups.items()):
        ratio_name = group_key.split("/", 1)[0]
        sample_id = chiplet_ids[0]
        sample_counts = realize_counts(
            len(read_entries(group_paths[sample_id])),
            parse_ratio_from_path(group_paths[sample_id]),
        )
        critical_count = sample_counts[0]
        redundant_net_count = sample_counts[1] // 2
        use_redundant_shared = critical_count == 0
        shared_per_die = (
            redundant_net_count // 4 if use_redundant_shared else critical_count // 2
        )
        pair_counts = compute_pair_shared_counts(chiplet_ids, pairs, shared_per_die)
        retag_group(group_paths, pair_counts, use_redundant_shared, args.dry_run)

        ratio_to_shared_names.setdefault(
            ratio_name,
            {
                pair: [
                    f"chiplet_{pair[0]}_chiplet_{pair[1]}_link_{idx:06d}"
                    for idx in range(1, count + 1)
                ]
                for pair, count in sorted(pair_counts.items())
            },
        )

        pair_summary = ", ".join(
            f"{a}-{b}:{count}" for (a, b), count in sorted(pair_counts.items())
        )
        print(f"{design_root / group_key}: pair_links={pair_summary}")

    for ratio_name, pair_to_names in sorted(ratio_to_shared_names.items()):
        write_shared_summary(design_root, ratio_name, pair_to_names, args.dry_run)
        total_shared = sum(len(names) for names in pair_to_names.values())
        print(
            f"{design_root / ratio_name / 'Compute_Large_interchip_shared_nets.txt'}: "
            f"{total_shared} shared nets"
        )


if __name__ == "__main__":
    main()
