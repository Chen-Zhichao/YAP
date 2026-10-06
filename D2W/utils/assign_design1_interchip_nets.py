#!/usr/bin/env python3
"""
Retag design_1 Compute_Small / Memory_DRAM bump-map names for inter-chiplet traffic.

Rules
-----
- Keep the existing bump-kind placement and ratios unchanged.
- For non-c0 ratios, use 50% of the smaller die's critical bumps as shared links.
- For c0_r50_pg50_dm0, use 25% of the smaller die's redundant nets as shared links.
- Use the same shared net/port names on both dies for those internal links.
- Rename remaining signal bumps as chiplet-local external signals.
- Prefix PG and dummy names with the owning chiplet id.
- Regenerate per-file criticality and write one shared-net list per ratio folder.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
from collections import Counter
from itertools import cycle
from pathlib import Path

import bmap_grid_sync as bgs
import generate_criticality as gc
from bump_assignment_utils import build_random_partitions

RATIO_RE = re.compile(
    r"^c(?P<c>\d+(?:[.pd]\d+)?)_r(?P<r>\d+(?:[.pd]\d+)?)_pg_?(?P<pg>\d+(?:[.pd]\d+)?)_dm(?P<dm>\d+(?:[.pd]\d+)?)$"
)
CHIPLET_PREFIX_RE = re.compile(r"^chiplet_(?P<id>\d+)_(?P<rest>.+)$")
PG_PATTERN = ("VDD", "VSS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign design_1 chiplet shared/external bump names."
    )
    parser.add_argument(
        "--design-root",
        type=Path,
        default=Path("D2W/input/design_1"),
        help="Root folder for design_1 ratio directories.",
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
            hashlib.sha256(random_group_key.encode("utf-8")).digest()[:8],
            "big",
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


def retag_pair(
    compute_path: Path,
    memory_path: Path,
    dry_run: bool,
) -> tuple[int, int]:
    compute_entries = read_entries(compute_path)
    memory_entries = read_entries(memory_path)

    ratios = parse_ratio_from_path(compute_path)
    compute_counts = realize_counts(len(compute_entries), ratios)
    memory_counts = realize_counts(len(memory_entries), ratios)

    compute_critical, compute_redundant, compute_pg, compute_dummy = compute_counts
    memory_critical, memory_redundant, memory_pg, memory_dummy = memory_counts

    compute_partitions = build_assignment_partitions(compute_path, compute_entries, compute_counts)
    memory_partitions = build_assignment_partitions(memory_path, memory_entries, memory_counts)

    use_redundant_shared = compute_critical == 0
    if use_redundant_shared:
        compute_redundant_nets = len(compute_partitions["redundant_pairs"])
        memory_redundant_nets = len(memory_partitions["redundant_pairs"])
        shared_count = compute_redundant_nets // 4
        if memory_redundant_nets < shared_count:
            raise ValueError(
                f"{memory_path} has only {memory_redundant_nets} redundant nets; need {shared_count}."
            )
    else:
        shared_count = compute_critical // 2
        if memory_critical < shared_count:
            raise ValueError(
                f"{memory_path} has only {memory_critical} critical bumps; need {shared_count}."
            )

    shared_names = [
        f"chiplet_0_chiplet_1_link_{idx:06d}" for idx in range(1, shared_count + 1)
    ]

    def rewrite(
        entries: list[list[str]],
        partitions,
        counts: tuple[int, int, int, int],
        chiplet_id: int,
        shared_names_for_file: list[str],
    ) -> None:
        critical_indices = partitions["critical_indices"]
        redundant_pairs = partitions["redundant_pairs"]
        pg_indices = partitions["pg_indices"]
        dummy_indices = partitions["dummy_indices"]

        if use_redundant_shared:
            for (first_idx, second_idx), shared_name in zip(
                redundant_pairs[: len(shared_names_for_file)],
                shared_names_for_file,
            ):
                for entry_idx in (first_idx, second_idx):
                    entries[entry_idx][4] = shared_name
                    entries[entry_idx][5] = shared_name
            external_critical_indices = critical_indices
            external_redundant_pairs = redundant_pairs[len(shared_names_for_file) :]
        else:
            for entry_idx, shared_name in zip(
                critical_indices[: len(shared_names_for_file)],
                shared_names_for_file,
            ):
                entries[entry_idx][4] = shared_name
                entries[entry_idx][5] = shared_name
            external_critical_indices = critical_indices[len(shared_names_for_file) :]
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

    rewrite(
        compute_entries,
        compute_partitions,
        compute_counts,
        0,
        shared_names,
    )
    rewrite(
        memory_entries,
        memory_partitions,
        memory_counts,
        1,
        shared_names,
    )

    substrate_compute_path = compute_path.with_name("Substrate_Silicon_To_Compute_Small.bmap")
    substrate_memory_path = memory_path.with_name("Substrate_Silicon_To_Memory_DRAM.bmap")
    substrate_compute_entries = (
        bgs.sync_names_by_normalized_grid(compute_entries, read_entries(substrate_compute_path))
        if substrate_compute_path.exists()
        else None
    )
    substrate_memory_entries = (
        bgs.sync_names_by_normalized_grid(memory_entries, read_entries(substrate_memory_path))
        if substrate_memory_path.exists()
        else None
    )

    if not dry_run:
        write_entries(compute_path, compute_entries)
        write_entries(memory_path, memory_entries)
        write_criticality(compute_path)
        write_criticality(memory_path)
        if substrate_compute_entries is not None:
            write_entries(substrate_compute_path, substrate_compute_entries)
            write_criticality(substrate_compute_path)
        if substrate_memory_entries is not None:
            write_entries(substrate_memory_path, substrate_memory_entries)
            write_criticality(substrate_memory_path)

    remaining_compute_redundant_nets = len(compute_partitions["redundant_pairs"]) - (
        shared_count if use_redundant_shared else 0
    )
    remaining_memory_redundant_nets = len(memory_partitions["redundant_pairs"]) - (
        shared_count if use_redundant_shared else 0
    )

    return shared_count, remaining_compute_redundant_nets + remaining_memory_redundant_nets


def discover_pairs(design_root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for compute_path in sorted(design_root.rglob("Compute_Small_From_Substrate_Silicon.bmap")):
        memory_path = compute_path.with_name("Memory_DRAM_From_Substrate_Silicon.bmap")
        if memory_path.exists():
            pairs.append((compute_path, memory_path))
    if not pairs:
        raise FileNotFoundError(f"No Compute/Memory bump-map pairs found under {design_root}")
    return pairs


def write_shared_net_files(
    design_root: Path,
    ratio_to_shared_names: dict[str, list[str]],
    dry_run: bool,
) -> None:
    for ratio_name, shared_names in sorted(ratio_to_shared_names.items()):
        out_path = design_root / ratio_name / "Compute_Small_to_Memory_DRAM_shared_nets.txt"
        if dry_run:
            continue
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for name in shared_names:
                f.write(name + "\n")
        tmp_path.replace(out_path)


def main() -> None:
    args = parse_args()
    design_root = args.design_root.resolve()
    pairs = discover_pairs(design_root)

    ratio_to_shared_names: dict[str, list[str]] = {}
    for compute_path, memory_path in pairs:
        ratio_name = next(part for part in compute_path.parts if RATIO_RE.match(part))
        shared_count, redundant_net_count = retag_pair(
            compute_path=compute_path,
            memory_path=memory_path,
            dry_run=args.dry_run,
        )
        ratio_to_shared_names.setdefault(
            ratio_name,
            [f"chiplet_0_chiplet_1_link_{idx:06d}" for idx in range(1, shared_count + 1)],
        )
        print(
            f"{compute_path.parent}: shared_links={shared_count}, "
            f"compute_path={compute_path.name}, memory_path={memory_path.name}, "
            f"external_redundant_nets={redundant_net_count}"
        )

    write_shared_net_files(design_root, ratio_to_shared_names, args.dry_run)

    for ratio_name, shared_names in sorted(ratio_to_shared_names.items()):
        print(
            f"{design_root / ratio_name / 'Compute_Small_to_Memory_DRAM_shared_nets.txt'}: "
            f"{len(shared_names)} shared nets"
        )


if __name__ == "__main__":
    main()
