#!/usr/bin/env python3
"""
Generate criticality files from one or more bump maps.

For each net in the bump map, this script writes:
    <net> <group_size> <tolerated_esd_failures> <tolerated_mechanical_failures>

The default rule is:
    tolerated_esd_failures = group_size - 1
    tolerated_mechanical_failures = group_size - 1

That means:
- critical bumps with one copy become: 1 0 0
- redundant pairs with two copies become: 2 1 1

Alternative profile:
    tolerated_esd_failures = 0
    tolerated_mechanical_failures = group_size - 1

By default, the script processes all `.bmap` files under:
  D2W/input/design_1
  D2W/input/design_2
  D2W/input/design_3
  D2W/input/design_4
  D2W/input/design_17
  D2W/input/design_18
  D2W/input/design_19

Examples
--------
Single file:
  python D2W/utils/generate_criticality.py --file path/to/file.bmap

Single file, backward-compatible positional form:
  python D2W/utils/generate_criticality.py path/to/file.bmap

Dry-run one file:
  python D2W/utils/generate_criticality.py --file path/to/file.bmap --dry-run

Batch default designs:
  python D2W/utils/generate_criticality.py --force
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Iterable


DEFAULT_DESIGNS = ("1", "2", "3", "4", "17", "18", "19")
DEFAULT_PROFILE = "default"
ESD_STRICT_PROFILE = "esd_strict"
ALL_PROFILES = (DEFAULT_PROFILE, ESD_STRICT_PROFILE)
PROFILE_TO_SUFFIX = {
    DEFAULT_PROFILE: "_criticality.txt",
    ESD_STRICT_PROFILE: "_criticality_esd_strict.txt",
}
PG_NET_RE = re.compile(r"\b(vdd|vss|vpp|vddq|vddql|gnd|vcc)\b", re.IGNORECASE)
REDUNDANT_NET_RE = re.compile(r"(^|_)rd(_|$)", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate criticality files from one or more .bmap files."
    )
    parser.add_argument(
        "input_bmap_file",
        nargs="?",
        type=Path,
        help="Optional single .bmap file to process (backward-compatible form).",
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
        help="Comma-separated design ids to process when no explicit file list is given.",
    )

    file_group = parser.add_mutually_exclusive_group()
    file_group.add_argument(
        "--file",
        type=Path,
        help="Optional single .bmap file to process.",
    )
    file_group.add_argument(
        "--files",
        nargs="*",
        type=Path,
        help="Optional explicit list of .bmap files to process.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files.",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default=DEFAULT_PROFILE,
        help=(
            "Comma-separated criticality profiles to generate. "
            "Supported: default, esd_strict, both."
        ),
    )
    return parser


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


def read_bmap_nets(filename: Path) -> dict[str, int]:
    """
    Read a bump map file and count bumps per net.
    """
    net_counts: Counter[str] = Counter()

    with open(filename, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 6:
                print(
                    f"Warning: {filename}:{line_num} has {len(parts)} fields "
                    f"(expected 6), skipping: {line}"
                )
                continue

            net_counts[parts[5]] += 1

    if not net_counts:
        raise ValueError(f"No valid bump entries found in {filename}")

    return dict(net_counts)


def parse_profiles(text: str) -> list[str]:
    requested = [item.strip() for item in text.split(",") if item.strip()]
    if not requested:
        raise ValueError("No criticality profile was provided.")

    profiles: list[str] = []
    for item in requested:
        if item == "both":
            for profile in ALL_PROFILES:
                if profile not in profiles:
                    profiles.append(profile)
            continue
        if item not in PROFILE_TO_SUFFIX:
            raise ValueError(
                f"Unsupported criticality profile '{item}'. Supported: "
                f"{', '.join(list(ALL_PROFILES) + ['both'])}"
            )
        if item not in profiles:
            profiles.append(item)
    return profiles


def is_redundant_signal_net(net: str, group_size: int) -> bool:
    if group_size <= 1:
        return False
    lowered = net.lower()
    if lowered == "dummy" or PG_NET_RE.search(lowered):
        return False
    return bool(REDUNDANT_NET_RE.search(lowered))


def tolerated_failures_for_group(
    group_size: int,
    profile: str,
    net: str | None = None,
) -> tuple[int, int]:
    if group_size <= 1:
        return 0, 0
    if profile == DEFAULT_PROFILE:
        tolerated = group_size - 1
        return tolerated, tolerated
    if profile == ESD_STRICT_PROFILE:
        if net is not None and is_redundant_signal_net(net, group_size):
            return 0, group_size - 1
        tolerated = group_size - 1
        return tolerated, tolerated
    raise ValueError(f"Unsupported criticality profile: {profile}")


def generate_criticality_lines(
    net_counts: dict[str, int],
    profile: str = DEFAULT_PROFILE,
) -> list[str]:
    lines: list[str] = []
    for net in sorted(net_counts.keys()):
        group_size = net_counts[net]
        tolerated_esd, tolerated_mech = tolerated_failures_for_group(
            group_size=group_size,
            profile=profile,
            net=net,
        )
        lines.append(f"{net} {group_size} {tolerated_esd} {tolerated_mech}")
    return lines


def get_output_filename(
    input_filename: Path,
    profile: str = DEFAULT_PROFILE,
) -> Path:
    if profile not in PROFILE_TO_SUFFIX:
        raise ValueError(f"Unsupported criticality profile: {profile}")
    return input_filename.with_name(f"{input_filename.stem}{PROFILE_TO_SUFFIX[profile]}")


def resolve_criticality_path(
    input_dir: Path | str,
    interface_name: str,
    profile: str = DEFAULT_PROFILE,
) -> Path:
    return get_output_filename(Path(input_dir) / f"{interface_name}.bmap", profile)


def summarize_net_counts(net_counts: dict[str, int]) -> list[str]:
    criticality_values: defaultdict[float, list[str]] = defaultdict(list)
    for net, count in net_counts.items():
        criticality = 1.0 / count
        criticality_values[criticality].append(net)

    summary_lines: list[str] = []
    for criticality in sorted(criticality_values.keys(), reverse=True):
        nets = criticality_values[criticality]
        bump_count = int(round(1.0 / criticality))
        summary_lines.append(
            f"  Criticality {criticality:.4f} ({bump_count} bump"
            f"{'s' if bump_count > 1 else ''}): {len(nets)} net"
            f"{'s' if len(nets) > 1 else ''}"
        )
    return summary_lines


def write_criticality_file(output_filename: Path, lines: list[str], force: bool) -> None:
    if output_filename.exists() and not force:
        raise FileExistsError(
            f"{output_filename} already exists. Use --force to overwrite it."
        )

    output_filename.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filename, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def write_criticality_variants(
    input_filename: Path,
    net_counts: dict[str, int],
    profiles: Iterable[str],
    force: bool,
    dry_run: bool,
) -> list[Path]:
    written_paths: list[Path] = []
    for profile in profiles:
        output_filename = get_output_filename(input_filename, profile)
        criticality_lines = generate_criticality_lines(net_counts, profile=profile)
        print(f"Output file [{profile}]: {output_filename}")
        if dry_run:
            print(f"Dry-run: not writing output file for profile '{profile}'.")
        else:
            write_criticality_file(output_filename, criticality_lines, force)
            print(
                f"Successfully generated criticality file [{profile}]: {output_filename}"
            )
        written_paths.append(output_filename)
    return written_paths


def resolve_input_files(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[Path]:
    if args.input_bmap_file is not None and (args.file is not None or args.files):
        parser.error("Use only one of positional input_bmap_file, --file, or --files.")

    if args.file is not None:
        return [args.file.resolve()]

    if args.files:
        return [path.resolve() for path in args.files]

    if args.input_bmap_file is not None:
        return [args.input_bmap_file.resolve()]

    design_ids = parse_design_ids(args.designs)
    return discover_bmap_files(args.input_root.resolve(), design_ids)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    files = resolve_input_files(args, parser)
    profiles = parse_profiles(args.profiles)

    if not files:
        raise FileNotFoundError("No .bmap files were found to process.")

    processed_count = 0
    for input_filename in files:
        if not input_filename.exists():
            raise FileNotFoundError(f"Input file '{input_filename}' not found")

        net_counts = read_bmap_nets(input_filename)

        print(f"Reading bump map: {input_filename}")
        print(f"Total nets: {len(net_counts)}")
        for line in summarize_net_counts(net_counts):
            print(line)
        write_criticality_variants(
            input_filename=input_filename,
            net_counts=net_counts,
            profiles=profiles,
            force=args.force,
            dry_run=args.dry_run,
        )

        print()
        processed_count += 1

    print(f"Processed {processed_count} bump map file{'s' if processed_count != 1 else ''}.")


if __name__ == "__main__":
    main()
