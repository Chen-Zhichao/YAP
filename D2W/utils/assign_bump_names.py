#!/usr/bin/env python3
"""
Assign port/net names to existing bump maps in place.

The script preserves the existing line order in each `.bmap` file and only
replaces the last two columns (`port`, `net`). Category assignment order is
driven by the folder type:

1. `Center_IO`: center outward, ring by ring
2. `Edge_IO`: outer edge inward, ring by ring
3. `Random_*`: deterministic shuffled order shared by the same chiplet type

By default it processes:
  D2W/input/design_1
  D2W/input/design_2
  D2W/input/design_3
  D2W/input/design_4
  D2W/input/design_17
  D2W/input/design_18
  D2W/input/design_19

Examples
--------
Dry-run a single copied file:
  python D2W/utils/assign_bump_names.py --file /tmp/test.bmap --dry-run

Overwrite one file:
  python D2W/utils/assign_bump_names.py --file /tmp/test.bmap

Overwrite all default designs with custom ratios:
  python D2W/utils/assign_bump_names.py --critical-ratio 0.15 --redundant-ratio 0.10 --pg-ratio 0.75 --dummy-ratio 0.00
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import re
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Iterable

from bump_assignment_utils import build_random_partitions


DEFAULT_DESIGNS = ("1", "2", "3", "4", "17", "18", "19")


@dataclass
class BmapEntry:
    instance: str
    bump_type: str
    x: str
    y: str
    port: str
    net: str
    raw_line: str


@dataclass(frozen=True)
class CategoryCounts:
    critical: int
    redundant: int
    pg: int
    dummy: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign critical/redundant/PG/dummy names to ordered bump maps."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("D2W/input"),
        help="Root directory that contains design_* folders.",
    )
    parser.add_argument(
        "--designs",
        type=str,
        default=",".join(DEFAULT_DESIGNS),
        help="Comma-separated design ids to process when --files is not used.",
    )
    file_group = parser.add_mutually_exclusive_group()
    file_group.add_argument(
        "--file",
        type=Path,
        help="Optional single .bmap file to process instead of discovering design folders.",
    )
    file_group.add_argument(
        "--files",
        nargs="*",
        type=Path,
        help="Optional explicit list of .bmap files to process instead of discovering design folders.",
    )
    parser.add_argument(
        "--critical-ratio",
        type=float,
        default=0.15,
        help="Physical bump ratio assigned to critical bumps.",
    )
    parser.add_argument(
        "--redundant-ratio",
        type=float,
        default=0.10,
        help="Physical bump ratio assigned to redundant bumps. Must resolve to an even bump count.",
    )
    parser.add_argument(
        "--pg-ratio",
        type=float,
        default=0.75,
        help="Physical bump ratio assigned to power/ground bumps.",
    )
    parser.add_argument(
        "--dummy-ratio",
        type=float,
        default=0.0,
        help="Physical bump ratio assigned to dummy bumps.",
    )
    parser.add_argument(
        "--critical-name-mode",
        choices=("suffix", "instance", "sequential"),
        default="suffix",
        help="How to name critical bumps. 'suffix' maps ..._b_0_1 -> b_0_1.",
    )
    parser.add_argument(
        "--critical-prefix",
        type=str,
        default="crit_",
        help="Prefix used only when --critical-name-mode=sequential.",
    )
    parser.add_argument(
        "--redundant-prefix",
        type=str,
        default="rd_",
        help="Prefix for redundant pair names.",
    )
    parser.add_argument(
        "--pg-pattern",
        type=str,
        default="VDD,VSS",
        help="Comma-separated PG net names assigned cyclically within the PG block.",
    )
    parser.add_argument(
        "--dummy-name",
        type=str,
        default="dummy",
        help="Port/net name used for dummy bumps.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed without overwriting files.",
    )
    return parser.parse_args()


def parse_design_ids(text: str) -> list[str]:
    design_ids = [item.strip() for item in text.split(",") if item.strip()]
    if not design_ids:
        raise ValueError("No design ids were provided.")
    return design_ids


def discover_bmap_files(input_root: Path, design_ids: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for design_id in design_ids:
        design_dir = input_root / f"design_{design_id}"
        if not design_dir.is_dir():
            raise FileNotFoundError(f"Design directory not found: {design_dir}")
        files.extend(sorted(design_dir.rglob("*.bmap")))
    return files


def read_bmap_entries(path: Path) -> list[BmapEntry]:
    entries: list[BmapEntry] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 6:
                raise ValueError(
                    f"{path}:{line_num} has {len(parts)} columns; expected 6 columns."
                )
            entries.append(
                BmapEntry(
                    instance=parts[0],
                    bump_type=parts[1],
                    x=parts[2],
                    y=parts[3],
                    port=parts[4],
                    net=parts[5],
                    raw_line=stripped,
                )
            )
    if not entries:
        raise ValueError(f"No bump entries found in {path}.")
    return entries


def _validate_ratio(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1, got {value}.")


def _largest_remainder_counts(total: int, desired: list[float]) -> list[int]:
    counts = [math.floor(value) for value in desired]
    remaining = total - sum(counts)
    order = sorted(
        range(len(desired)),
        key=lambda idx: (desired[idx] - counts[idx], desired[idx]),
        reverse=True,
    )
    for idx in order[:remaining]:
        counts[idx] += 1
    return counts


def _score(candidate: list[int], desired: list[float]) -> float:
    return sum((observed - target) ** 2 for observed, target in zip(candidate, desired))


def realize_category_counts(
    total_bumps: int,
    critical_ratio: float,
    redundant_ratio: float,
    pg_ratio: float,
    dummy_ratio: float,
) -> CategoryCounts:
    for name, value in (
        ("critical_ratio", critical_ratio),
        ("redundant_ratio", redundant_ratio),
        ("pg_ratio", pg_ratio),
        ("dummy_ratio", dummy_ratio),
    ):
        _validate_ratio(name, value)

    ratio_sum = critical_ratio + redundant_ratio + pg_ratio + dummy_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "critical_ratio + redundant_ratio + pg_ratio + dummy_ratio must sum to 1.0."
        )

    desired = [
        critical_ratio * total_bumps,
        redundant_ratio * total_bumps,
        pg_ratio * total_bumps,
        dummy_ratio * total_bumps,
    ]
    counts = _largest_remainder_counts(total_bumps, desired)

    if counts[1] % 2 == 1:
        best_candidate = None
        best_score = None
        for other_idx in (0, 2, 3):
            if counts[other_idx] > 0:
                candidate = counts.copy()
                candidate[1] += 1
                candidate[other_idx] -= 1
                score = _score(candidate, desired)
                if best_score is None or score < best_score:
                    best_candidate = candidate
                    best_score = score

            if counts[1] > 0:
                candidate = counts.copy()
                candidate[1] -= 1
                candidate[other_idx] += 1
                score = _score(candidate, desired)
                if best_score is None or score < best_score:
                    best_candidate = candidate
                    best_score = score

        if best_candidate is None or best_candidate[1] % 2 == 1:
            raise ValueError("Unable to resolve an even redundant bump count.")
        counts = best_candidate

    return CategoryCounts(
        critical=counts[0],
        redundant=counts[1],
        pg=counts[2],
        dummy=counts[3],
    )


def extract_suffix_name(instance: str) -> str:
    marker = "_b_"
    if marker in instance:
        return instance[instance.index(marker) + 1 :]
    return instance


def build_critical_name(
    entry: BmapEntry,
    index: int,
    mode: str,
    prefix: str,
) -> str:
    if mode == "instance":
        return entry.instance
    if mode == "sequential":
        return f"{prefix}{index}"
    return extract_suffix_name(entry.instance)


def infer_assignment_mode(path: Path) -> str:
    for part in path.parts:
        if part == "Center_IO":
            return "center"
        if part == "Edge_IO":
            return "edge"
        if part.startswith("Random_"):
            return "random"
    return "input"


def canonical_random_chiplet_type(path: Path) -> str:
    """
    Canonicalize a Random_* file stem so repeated chiplets of the same type
    share one deterministic random assignment template.
    """
    stem_parts = [part for part in path.stem.split("_") if not part.isdigit()]
    canonical_stem = "_".join(stem_parts)
    canonical_stem = re.sub(r"_+", "_", canonical_stem).strip("_")
    return canonical_stem


def build_assignment_order(path: Path, entries: list[BmapEntry]) -> list[int]:
    mode = infer_assignment_mode(path)
    if mode == "input":
        return list(range(len(entries)))

    x_values = sorted({float(entry.x) for entry in entries})
    y_values = sorted({float(entry.y) for entry in entries}, reverse=True)
    x_to_col = {value: idx for idx, value in enumerate(x_values)}
    y_to_row = {value: idx for idx, value in enumerate(y_values)}

    decorated: list[tuple[int, int, int]] = []
    for idx, entry in enumerate(entries):
        col = x_to_col[float(entry.x)]
        row = y_to_row[float(entry.y)]
        decorated.append((idx, row, col))

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


def build_assignment_partitions(path: Path, entries: list[BmapEntry], counts: CategoryCounts):
    mode = infer_assignment_mode(path)
    ordered_indices = build_assignment_order(path, entries)
    if mode != "random":
        critical_end = counts.critical
        redundant_end = critical_end + counts.redundant
        pg_end = redundant_end + counts.pg
        redundant_indices = ordered_indices[critical_end:redundant_end]
        redundant_pairs = list(zip(redundant_indices[0::2], redundant_indices[1::2]))
        return {
            "critical_indices": ordered_indices[:critical_end],
            "redundant_pairs": redundant_pairs,
            "pg_indices": ordered_indices[redundant_end:pg_end],
            "dummy_indices": ordered_indices[pg_end:pg_end + counts.dummy],
        }

    x_values = sorted({float(entry.x) for entry in entries})
    y_values = sorted({float(entry.y) for entry in entries}, reverse=True)
    x_to_col = {value: idx for idx, value in enumerate(x_values)}
    y_to_row = {value: idx for idx, value in enumerate(y_values)}
    index_to_row_col = {
        idx: (y_to_row[float(entry.y)], x_to_col[float(entry.x)])
        for idx, entry in enumerate(entries)
    }
    seed_key = f"{path.parent.as_posix()}::{canonical_random_chiplet_type(path)}"
    partitions = build_random_partitions(
        row_major_indices=ordered_indices,
        index_to_row_col=index_to_row_col,
        critical_count=counts.critical,
        redundant_count=counts.redundant,
        pg_count=counts.pg,
        dummy_count=counts.dummy,
        seed_key=seed_key,
    )
    return {
        "critical_indices": partitions.critical_indices,
        "redundant_pairs": partitions.redundant_pairs,
        "pg_indices": partitions.pg_indices,
        "dummy_indices": partitions.dummy_indices,
    }


def assign_names(
    path: Path,
    entries: list[BmapEntry],
    counts: CategoryCounts,
    critical_name_mode: str,
    critical_prefix: str,
    redundant_prefix: str,
    pg_pattern: list[str],
    dummy_name: str,
) -> list[str]:
    if not pg_pattern:
        raise ValueError("pg_pattern must contain at least one name.")

    rewritten_lines: list[str | None] = [None] * len(entries)
    partitions = build_assignment_partitions(path, entries, counts)

    def rewrite_entry(entry_index: int, name: str) -> None:
        entry = entries[entry_index]
        rewritten_lines[entry_index] = (
            f"{entry.instance} {entry.bump_type} {entry.x} {entry.y} {name} {name}"
        )

    for critical_idx, entry_index in enumerate(partitions["critical_indices"], start=1):
        entry = entries[entry_index]
        name = build_critical_name(
            entry, critical_idx, critical_name_mode, critical_prefix
        )
        rewrite_entry(entry_index, name)

    for redundant_idx, pair in enumerate(partitions["redundant_pairs"], start=1):
        pair_name = f"{redundant_prefix}{redundant_idx}"
        for entry_index in pair:
            rewrite_entry(entry_index, pair_name)

    pg_name_iter = cycle(pg_pattern)
    for entry_index in partitions["pg_indices"]:
        pg_name = next(pg_name_iter)
        rewrite_entry(entry_index, pg_name)

    for entry_index in partitions["dummy_indices"]:
        rewrite_entry(entry_index, dummy_name)

    assigned_count = (
        len(partitions["critical_indices"])
        + 2 * len(partitions["redundant_pairs"])
        + len(partitions["pg_indices"])
        + len(partitions["dummy_indices"])
    )
    if assigned_count != len(entries):
        raise AssertionError("Internal error: not all bump entries were assigned.")

    if any(line is None for line in rewritten_lines):
        raise AssertionError("Internal error: some bump entries were not rewritten.")

    return [line for line in rewritten_lines if line is not None]


def write_lines(path: Path, lines: list[str]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    tmp_path.replace(path)


def summarize_counts(path: Path, counts: CategoryCounts, total_bumps: int) -> str:
    logical_signal_ratio = (counts.critical + counts.redundant / 2) / total_bumps
    return (
        f"{path}: total={total_bumps}, critical={counts.critical}, "
        f"redundant={counts.redundant} ({counts.redundant // 2} nets), "
        f"pg={counts.pg}, dummy={counts.dummy}, "
        f"logical_signal_ratio={logical_signal_ratio:.6f}"
    )


def main() -> None:
    args = parse_args()
    pg_pattern = [name.strip() for name in args.pg_pattern.split(",") if name.strip()]

    if args.file is not None:
        files = [args.file.resolve()]
    elif args.files:
        files = [path.resolve() for path in args.files]
    else:
        design_ids = parse_design_ids(args.designs)
        files = discover_bmap_files(args.input_root.resolve(), design_ids)

    if not files:
        raise FileNotFoundError("No .bmap files were found to process.")

    for path in files:
        entries = read_bmap_entries(path)
        counts = realize_category_counts(
            total_bumps=len(entries),
            critical_ratio=args.critical_ratio,
            redundant_ratio=args.redundant_ratio,
            pg_ratio=args.pg_ratio,
            dummy_ratio=args.dummy_ratio,
        )
        rewritten_lines = assign_names(
            path=path,
            entries=entries,
            counts=counts,
            critical_name_mode=args.critical_name_mode,
            critical_prefix=args.critical_prefix,
            redundant_prefix=args.redundant_prefix,
            pg_pattern=pg_pattern,
            dummy_name=args.dummy_name,
        )

        print(summarize_counts(path, counts, len(entries)))
        if not args.dry_run:
            write_lines(path, rewritten_lines)


if __name__ == "__main__":
    main()
