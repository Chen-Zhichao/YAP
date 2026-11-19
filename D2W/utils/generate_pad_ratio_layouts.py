#!/usr/bin/env python3
"""
Generate derived pad-ratio layouts from an existing ratio directory.

For each requested target ratio, this script:
1. copies the source ratio directory tree into the target ratio directory
2. rewrites every `.bmap` file with the requested critical/redundant/PG/dummy ratios
3. regenerates every sibling `_criticality.txt` file
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from shutil import copytree

import assign_bump_names as abn
import generate_criticality as gc


def parse_targets(values: list[str]) -> list[tuple[str, float, float, float, float]]:
    targets: list[tuple[str, float, float, float, float]] = []
    for value in values:
        try:
            name, ratios_text = value.split(":", 1)
            critical_text, redundant_text, pg_text, dummy_text = ratios_text.split(",")
        except ValueError as exc:
            raise ValueError(
                f"Invalid --target '{value}'. Expected NAME:critical,redundant,pg,dummy"
            ) from exc
        targets.append(
            (
                name.strip(),
                float(critical_text),
                float(redundant_text),
                float(pg_text),
                float(dummy_text),
            )
        )
    if not targets:
        raise ValueError("At least one --target must be provided.")
    return targets


def parse_designs(text: str) -> list[str]:
    designs = [item.strip() for item in text.split(",") if item.strip()]
    if not designs:
        raise ValueError("No design ids were provided.")
    return designs


def regenerate_ratio_tree(
    design_dir: Path,
    source_ratio: str,
    target_name: str,
    critical_ratio: float,
    redundant_ratio: float,
    pg_ratio: float,
    dummy_ratio: float,
    pg_pattern: list[str],
) -> int:
    src_dir = design_dir / source_ratio
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source ratio directory not found: {src_dir}")

    dst_dir = design_dir / target_name
    copytree(src_dir, dst_dir, dirs_exist_ok=True)

    bmaps = sorted(dst_dir.rglob("*.bmap"))
    if not bmaps:
        raise FileNotFoundError(f"No .bmap files found under {dst_dir}")

    for bmap_path in bmaps:
        entries = abn.read_bmap_entries(bmap_path)
        counts = abn.realize_category_counts(
            total_bumps=len(entries),
            critical_ratio=critical_ratio,
            redundant_ratio=redundant_ratio,
            pg_ratio=pg_ratio,
            dummy_ratio=dummy_ratio,
        )
        rewritten_lines = abn.assign_names(
            path=bmap_path,
            entries=entries,
            counts=counts,
            critical_name_mode="suffix",
            critical_prefix="crit_",
            redundant_prefix="rd_",
            pg_pattern=pg_pattern,
            dummy_name="dummy",
        )
        abn.write_lines(bmap_path, rewritten_lines)

        net_counts = Counter(line.split()[5] for line in rewritten_lines)
        criticality_lines = gc.generate_criticality_lines(dict(net_counts))
        gc.write_criticality_file(
            gc.get_output_filename(bmap_path),
            criticality_lines,
            force=True,
        )

    return len(bmaps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate derived pad-ratio directories from an existing ratio tree."
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
        required=True,
        help="Comma-separated design ids, for example: 1,2,3",
    )
    parser.add_argument(
        "--source-ratio",
        type=str,
        default="c20_r10_pg50_dm20",
        help="Existing ratio directory used as the physical-layout template.",
    )
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        help="Target spec in the form NAME:critical,redundant,pg,dummy. Repeat for multiple targets.",
    )
    parser.add_argument(
        "--pg-pattern",
        type=str,
        default="VDD,VSS",
        help="Comma-separated PG net names assigned cyclically.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    targets = parse_targets(args.target)
    designs = parse_designs(args.designs)
    pg_pattern = [item.strip() for item in args.pg_pattern.split(",") if item.strip()]
    if not pg_pattern:
        raise ValueError("--pg-pattern must contain at least one name.")

    for design in designs:
        design_dir = args.input_root / f"design_{design}"
        if not design_dir.is_dir():
            raise FileNotFoundError(f"Design directory not found: {design_dir}")

        for target_name, critical_ratio, redundant_ratio, pg_ratio, dummy_ratio in targets:
            num_bmaps = regenerate_ratio_tree(
                design_dir=design_dir,
                source_ratio=args.source_ratio,
                target_name=target_name,
                critical_ratio=critical_ratio,
                redundant_ratio=redundant_ratio,
                pg_ratio=pg_ratio,
                dummy_ratio=dummy_ratio,
                pg_pattern=pg_pattern,
            )
            print(
                f"Generated design_{design}/{target_name} ({num_bmaps} bump maps)",
                flush=True,
            )


if __name__ == "__main__":
    main()
