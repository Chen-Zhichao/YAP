#!/usr/bin/env python3
"""
Package HBM_A / HBM_B into lightweight single-interface design variants.

This script does not require 3dblox wrapper files. Instead it creates
variant directories that each contain:
  - <INTERFACE>.bmap
  - <INTERFACE>_criticality.txt
  - <INTERFACE>_criticality_esd_strict.txt

The existing HBM_A.yaml / HBM_B.yaml configs are used directly with the
legacy single-interface path in pad_risk_map_calculator.py and
simulator_main.py.

Variants:
  - Original: preserve the original placement, but convert NC -> dummy
  - Center_IO: assign critical/redundant/PG/dummy from center outward
  - Edge_IO: assign critical/redundant/PG/dummy from edge inward
  - Random_IO: deterministic random assignment shared per HBM design
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from bump_assignment_utils import build_random_partitions
import generate_criticality as gc

ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = ROOT / "D2W" / "input"
CONFIG_ROOT = ROOT / "D2W" / "configs"
OLD_BMAP_ROOT = INPUT_ROOT / "old_bmap"
VARIANTS = ("Original", "Center_IO", "Edge_IO", "Random_IO")
PG_NET_RE = re.compile(r"^(vdd|vss|vpp|vddq|vddql|vddc|gnd|vcc)", re.IGNORECASE)


@dataclass(frozen=True)
class BmapEntry:
    instance: str
    bump_type: str
    x: float
    y: float
    port: str
    net: str

    @classmethod
    def from_line(cls, line: str) -> "BmapEntry":
        parts = line.strip().split()
        if len(parts) != 6:
            raise ValueError(f"Expected 6 columns, got {len(parts)}: {line}")
        return cls(
            instance=parts[0],
            bump_type=parts[1],
            x=float(parts[2]),
            y=float(parts[3]),
            port=parts[4],
            net=parts[5],
        )

    def render(self, name: str) -> str:
        return (
            f"{self.instance} {self.bump_type} "
            f"{self.x:g} {self.y:g} {name} {name}"
        )


def read_bmap(path: Path) -> list[BmapEntry]:
    return [BmapEntry.from_line(line) for line in path.read_text().splitlines() if line.strip()]


def read_criticality_template(path: Path) -> tuple[list[tuple[str, str, str, str]], dict[str, tuple[str, str, str]]]:
    ordered: list[tuple[str, str, str, str]] = []
    lookup: dict[str, tuple[str, str, str]] = {}
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            net, group_size, esd, mech = parts
            ordered.append((net, group_size, esd, mech))
            lookup[net] = (group_size, esd, mech)
    return ordered, lookup


def classify_nets(entries: list[BmapEntry]) -> tuple[list[str], list[tuple[str, int]], list[str], int]:
    net_counts = Counter(entry.net for entry in entries)
    seen_critical: OrderedDict[str, None] = OrderedDict()
    seen_redundant: OrderedDict[str, int] = OrderedDict()
    pg_names: list[str] = []
    dummy_count = 0

    for entry in entries:
        net = entry.net
        if net == "NC":
            dummy_count += 1
        elif PG_NET_RE.match(net):
            pg_names.append(net)
        elif net_counts[net] > 1:
            seen_redundant.setdefault(net, net_counts[net])
        else:
            seen_critical.setdefault(net, None)

    return list(seen_critical.keys()), list(seen_redundant.items()), pg_names, dummy_count


def build_assignment_order(entries: list[BmapEntry], mode: str, seed_key: str) -> list[int]:
    if mode == "Original":
        return list(range(len(entries)))

    x_values = sorted({entry.x for entry in entries})
    y_values = sorted({entry.y for entry in entries}, reverse=True)
    x_to_col = {value: idx for idx, value in enumerate(x_values)}
    y_to_row = {value: idx for idx, value in enumerate(y_values)}

    decorated: list[tuple[int, int, int]] = []
    for idx, entry in enumerate(entries):
        decorated.append((idx, y_to_row[entry.y], x_to_col[entry.x]))

    if mode == "Center_IO":
        center_row = (len(y_values) - 1) / 2.0
        center_col = (len(x_values) - 1) / 2.0
        decorated.sort(
            key=lambda item: (
                max(abs(item[1] - center_row), abs(item[2] - center_col)),
                item[1],
                item[2],
            )
        )
    elif mode == "Edge_IO":
        max_row = len(y_values) - 1
        max_col = len(x_values) - 1
        decorated.sort(
            key=lambda item: (
                min(item[1], item[2], max_row - item[1], max_col - item[2]),
                item[1],
                item[2],
            )
        )
    elif mode == "Random_IO":
        decorated.sort(key=lambda item: (item[1], item[2]))
        seed = int.from_bytes(hashlib.sha256(seed_key.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)
        rng.shuffle(decorated)
    else:
        raise ValueError(f"Unsupported variant mode: {mode}")

    return [idx for idx, _, _ in decorated]


def build_assignment_partitions(entries: list[BmapEntry], mode: str, seed_key: str):
    order = build_assignment_order(entries, mode, seed_key)
    critical_names, redundant_groups, pg_names, dummy_count = classify_nets(entries)
    critical_count = len(critical_names)
    redundant_count = sum(count for _, count in redundant_groups)
    pg_count = len(pg_names)

    if mode != "Random_IO":
        redundant_indices = order[critical_count : critical_count + redundant_count]
        return {
            "critical_names": critical_names,
            "redundant_groups": redundant_groups,
            "pg_names": pg_names,
            "dummy_count": dummy_count,
            "critical_indices": order[:critical_count],
            "redundant_pairs": list(zip(redundant_indices[0::2], redundant_indices[1::2])),
            "pg_indices": order[
                critical_count + redundant_count : critical_count + redundant_count + pg_count
            ],
            "dummy_indices": order[critical_count + redundant_count + pg_count :],
        }

    x_values = sorted({entry.x for entry in entries})
    y_values = sorted({entry.y for entry in entries}, reverse=True)
    x_to_col = {value: idx for idx, value in enumerate(x_values)}
    y_to_row = {value: idx for idx, value in enumerate(y_values)}
    index_to_row_col = {
        idx: (y_to_row[entry.y], x_to_col[entry.x])
        for idx, entry in enumerate(entries)
    }
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
        "critical_names": critical_names,
        "redundant_groups": redundant_groups,
        "pg_names": pg_names,
        "dummy_count": dummy_count,
        "critical_indices": partitions.critical_indices,
        "redundant_pairs": partitions.redundant_pairs,
        "pg_indices": partitions.pg_indices,
        "dummy_indices": partitions.dummy_indices,
    }


def rewrite_original(entries: list[BmapEntry]) -> list[str]:
    lines = []
    for entry in entries:
        name = "dummy" if entry.net == "NC" else entry.net
        lines.append(entry.render(name))
    return lines


def rewrite_variant(entries: list[BmapEntry], mode: str, design_name: str) -> list[str]:
    if mode == "Original":
        return rewrite_original(entries)

    partitions = build_assignment_partitions(entries, mode, seed_key=f"{design_name}::{mode}")
    critical_names = partitions["critical_names"]
    redundant_groups = partitions["redundant_groups"]
    pg_names = partitions["pg_names"]
    dummy_count = partitions["dummy_count"]
    rewritten: list[str | None] = [None] * len(entries)

    def assign_name(entry_index: int, name: str) -> None:
        rewritten[entry_index] = entries[entry_index].render(name)

    for entry_index, name in zip(partitions["critical_indices"], critical_names):
        assign_name(entry_index, name)

    pair_cursor = 0
    for name, count in redundant_groups:
        group_pairs = partitions["redundant_pairs"][pair_cursor : pair_cursor + count // 2]
        for first_idx, second_idx in group_pairs:
            assign_name(first_idx, name)
            assign_name(second_idx, name)
        pair_cursor += count // 2

    for entry_index, name in zip(partitions["pg_indices"], pg_names):
        assign_name(entry_index, name)

    for entry_index in partitions["dummy_indices"]:
        assign_name(entry_index, "dummy")

    assigned_count = (
        len(partitions["critical_indices"])
        + 2 * len(partitions["redundant_pairs"])
        + len(partitions["pg_indices"])
        + len(partitions["dummy_indices"])
    )
    if assigned_count != len(entries) or any(line is None for line in rewritten):
        raise AssertionError(f"Incomplete rewrite for {design_name} {mode}")
    return [line for line in rewritten if line is not None]


def count_nets_in_rendered_lines(lines: list[str]) -> Counter[str]:
    net_counts: Counter[str] = Counter()
    for line in lines:
        parts = line.split()
        if len(parts) == 6:
            net_counts[parts[5]] += 1
    return net_counts


def build_criticality_lines(
    rendered_lines: list[str],
    ordered_template: list[tuple[str, str, str, str]],
    profile: str,
) -> list[str]:
    net_counts = count_nets_in_rendered_lines(rendered_lines)
    lines: list[str] = []
    emitted: set[str] = set()

    for old_net, *_ in ordered_template:
        new_net = "dummy" if old_net == "NC" else old_net
        if new_net not in net_counts:
            continue
        actual_count = net_counts[new_net]
        tolerated_esd, tolerated_mech = gc.tolerated_failures_for_group(
            group_size=actual_count,
            profile=profile,
            net=new_net,
        )
        lines.append(f"{new_net} {actual_count} {tolerated_esd} {tolerated_mech}")
        emitted.add(new_net)

    for net in sorted(net_counts.keys()):
        if net in emitted:
            continue
        count = net_counts[net]
        tolerated_esd, tolerated_mech = gc.tolerated_failures_for_group(
            group_size=count,
            profile=profile,
            net=net,
        )
        lines.append(f"{net} {count} {tolerated_esd} {tolerated_mech}")

    return lines


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines))


def package_one_hbm(design_name: str) -> None:
    source_bmap = INPUT_ROOT / design_name / f"HBM_footprint_{design_name[-1]}.bmap"
    criticality_template = OLD_BMAP_ROOT / f"HBM_footprint_{design_name[-1]}_criticality.txt"
    entries = read_bmap(source_bmap)
    ordered_template, _ = read_criticality_template(criticality_template)

    def write_variant_outputs(out_dir: Path, rendered_bmap_lines: list[str]) -> None:
        write_lines(out_dir / source_bmap.name, rendered_bmap_lines)
        for profile in gc.ALL_PROFILES:
            write_lines(
                out_dir / gc.get_output_filename(source_bmap, profile=profile).name,
                build_criticality_lines(
                    rendered_lines=rendered_bmap_lines,
                    ordered_template=ordered_template,
                    profile=profile,
                ),
            )

    write_variant_outputs(INPUT_ROOT / design_name, rewrite_original(entries))

    for variant in VARIANTS:
        variant_dir = INPUT_ROOT / design_name / variant
        bmap_lines = rewrite_variant(entries, variant, design_name)
        write_variant_outputs(variant_dir, bmap_lines)


def main() -> None:
    for design_name in ("HBM_A", "HBM_B"):
        package_one_hbm(design_name)
        print(f"Packaged {design_name} variants: {', '.join(VARIANTS)}")


if __name__ == "__main__":
    main()
