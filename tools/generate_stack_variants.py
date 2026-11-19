#!/usr/bin/env python3
"""Generate compact all-critical D2W/W2W design_6 stack variants."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODES = ("D2W", "W2W")


def _dump_yaml(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
        encoding="ascii",
    )


def _config_with_compact_layout(base_text: str, dram_tiers: int, family: dict,
                                mode: str) -> str:
    lines = base_text.splitlines()
    output = []
    index = 0
    while index < len(lines):
        line = lines[index]
        output.append(line)
        stripped = line.strip()
        if mode == "D2W" and stripped.startswith("KNOWN_GOOD_DIE_REFERENCES:"):
            index += 1
            while index < len(lines) and lines[index].lstrip().startswith("-"):
                index += 1
            indent = line[: len(line) - len(line.lstrip())] + "  "
            output.extend(f"{indent}- Memory_DRAM_{tier}" for tier in range(dram_tiers))
            continue
        if stripped.startswith("PAD_ARRANGE_PATTERN:"):
            indent = line[: len(line) - len(line.lstrip())]
            output.append(f'{indent}LAYOUT_KIND: "regular_center_io"')
            output.append(
                f'{indent}CRITICAL_PAD_FRACTION: {family["critical_pad_fraction"]}'
            )
            output.append(
                f'{indent}POWER_GROUND_PAD_FRACTION: {family["power_ground_pad_fraction"]}'
            )
            output.append(f'{indent}DUMMY_PAD_FRACTION: {family["dummy_pad_fraction"]}')
        index += 1
    return "\n".join(output) + "\n"


def _chiplet_definition(
    *,
    width_um: float,
    length_um: float,
    thickness_um: float,
    regions: list[tuple[str, str]],
):
    return {
        "type": "die",
        "design_area": [width_um, length_um],
        "thickness": thickness_um,
        "shrink": 1,
        "regions": {
            region_name: {
                "bmap": bmap_name,
                "side": "back",
                "layer": "UBM",
                "gds_layer": None,
                "coords": [
                    [0.0, 0.0],
                    [width_um, 0.0],
                    [width_um, length_um],
                    [0.0, length_um],
                ],
            }
            for region_name, bmap_name in regions
        },
        "seal_ring_width": [0, 0, 0, 0],
        "scribe_line_remaining_width": [0, 0, 0, 0],
        "tsv": True,
    }


def _write_input_variant(mode: str, variant: dict, family: dict) -> None:
    name = str(variant["name"])
    tiers = int(variant["dram_tiers"])
    pitch_um = float(variant["pitch_um"])
    pad_diameter_um = float(variant["pad_diameter_um"])
    width_um = float(family["die_width_um"])
    length_um = float(family["die_length_um"])
    thickness_um = float(family["die_thickness_um"])
    critical_fraction = float(family["critical_pad_fraction"])
    power_ground_fraction = float(family["power_ground_pad_fraction"])
    dummy_fraction = float(family["dummy_pad_fraction"])
    if not math.isclose(
        critical_fraction + power_ground_fraction + dummy_fraction,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Pad fractions in the variant manifest must sum to one.")
    input_dir = ROOT / mode / "input" / name
    input_dir.mkdir(parents=True, exist_ok=True)

    chiplet_defs = {}
    chiplet_defs["TSV_Interconnect_Die"] = _chiplet_definition(
        width_um=width_um,
        length_um=length_um,
        thickness_um=thickness_um,
        regions=[
            (
                "To_Memory_DRAM_0",
                "TSV_Interconnect_Die_To_Memory_DRAM_0.bmap",
            )
        ],
    )
    for tier in range(tiers):
        regions = [
            (
                (
                    "From_TSV_Interconnect_Die"
                    if tier == 0
                    else f"From_Memory_DRAM_{tier - 1}"
                ),
                (
                    "Memory_DRAM_0_From_TSV_Interconnect_Die.bmap"
                    if tier == 0
                    else f"Memory_DRAM_{tier}_From_Memory_DRAM_{tier - 1}.bmap"
                ),
            )
        ]
        if tier + 1 < tiers:
            regions.append(
                (
                    f"To_Memory_DRAM_{tier + 1}",
                    f"Memory_DRAM_{tier}_To_Memory_DRAM_{tier + 1}.bmap",
                )
            )
        chiplet_defs[f"Memory_DRAM_{tier}"] = _chiplet_definition(
            width_um=width_um,
            length_um=length_um,
            thickness_um=thickness_um,
            regions=regions,
        )

    _dump_yaml(
        input_dir / "generated_chiplet_definitions.3dbv",
        {
            "Header": {"version": 2.0, "unit": "micron", "precision": 2000},
            "ChipletDef": chiplet_defs,
        },
    )

    instances = {
        "TSV_Interconnect_Die_0": {
            "reference": "TSV_Interconnect_Die",
            "is_master": True,
        }
    }
    stack = {
        "TSV_Interconnect_Die_0": {
            "loc": [0.0, 0.0],
            "z": 0.0,
            "orient": "MZ_MX",
            "root": True,
        }
    }
    connections = {}
    for tier in range(tiers):
        instance = f"Memory_DRAM_{tier}_{tier + 1}"
        instances[instance] = {
            "reference": f"Memory_DRAM_{tier}",
            "is_master": True,
        }
        stack[instance] = {
            "loc": [0.0, 0.0],
            "z": (tier + 1) * thickness_um,
            "orient": "R0" if tier % 2 == 0 else "MZ_MX",
        }
        bottom_instance = (
            "TSV_Interconnect_Die_0" if tier == 0 else f"Memory_DRAM_{tier - 1}_{tier}"
        )
        from_region = (
            "From_TSV_Interconnect_Die" if tier == 0 else f"From_Memory_DRAM_{tier - 1}"
        )
        to_region = f"To_Memory_DRAM_{tier}"
        connections[f"Conn{tier + 1}"] = {
            "top": f"{instance}.regions.{from_region}",
            "bot": f"{bottom_instance}.regions.{to_region}",
        }

    _dump_yaml(
        input_dir / "generated_stack_config.3dbx",
        {
            "Header": {
                "version": 2.0,
                "unit": "micron",
                "precision": 2000,
                "include": ["generated_chiplet_definitions.3dbv"],
            },
            "Design": {"name": name},
            "ChipletInst": instances,
            "Stack": stack,
            "Connection": connections,
        },
    )

    dbf = {
        "Bump_Types": {
            "silicon_individual_bonding": {
                "bump_size": pad_diameter_um,
                "through_via_size": pad_diameter_um,
                "bump_subckt": ["DO_NOT_USE.subckt", "DO_NOT_USE"],
                "tv_subckt": ["DO_NOT_USE.subckt", "DO_NOT_USE"],
            }
        },
        "Chiplet_Grid": {"pitch": pitch_um},
    }
    _dump_yaml(input_dir / "TSV_Interconnect_Die.3dbf", dbf)
    for tier in range(tiers):
        _dump_yaml(input_dir / f"Memory_DRAM_{tier}.3dbf", dbf)

    rows = math.floor(length_um / pitch_um)
    cols = math.floor(width_um / pitch_um)
    total_pads = rows * cols
    desired_counts = [
        critical_fraction * total_pads,
        power_ground_fraction * total_pads,
        dummy_fraction * total_pads,
    ]
    category_counts = [math.floor(value) for value in desired_counts]
    remaining = total_pads - sum(category_counts)
    remainder_order = sorted(
        range(3),
        key=lambda index: (
            desired_counts[index] - category_counts[index],
            desired_counts[index],
        ),
        reverse=True,
    )
    for index in remainder_order[:remaining]:
        category_counts[index] += 1
    _dump_yaml(
        input_dir / "compact_layout.yaml",
        {
            "layout_kind": "regular_center_io",
            "rows": rows,
            "cols": cols,
            "pitch_r_um": pitch_um,
            "pitch_c_um": pitch_um,
            "pad_diameter_um": pad_diameter_um,
            "pad_array_l_um": (rows - 1) * pitch_um,
            "pad_array_w_um": (cols - 1) * pitch_um,
            "critical_pad_fraction": critical_fraction,
            "power_ground_pad_fraction": power_ground_fraction,
            "dummy_pad_fraction": dummy_fraction,
            "num_critical_pads_per_interface": category_counts[0],
            "num_power_ground_pads_per_interface": category_counts[1],
            "num_dummy_pads_per_interface": category_counts[2],
            "num_total_pads_per_interface": total_pads,
            "dram_tiers": tiers,
            "bonding_interfaces": tiers,
            "tsv_base_included_in_hi_count": False,
            "modeling_only": True,
        },
    )


def _write_config_variant(mode: str, variant: dict, family: dict) -> None:
    name = str(variant["name"])
    base_path = ROOT / mode / "configs" / "design_6" / "design_6.yaml"
    output_path = ROOT / mode / "configs" / name / f"{name}.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _config_with_compact_layout(
            base_path.read_text(encoding="utf-8"),
            int(variant["dram_tiers"]),
            family,
            mode,
        ),
        encoding="utf-8",
    )


def generate(manifest_path: Path) -> None:
    family = yaml.safe_load(manifest_path.read_text(encoding="ascii"))
    for variant in family["variants"]:
        for mode in MODES:
            _write_input_variant(mode, variant, family)
            _write_config_variant(mode, variant, family)
        rows = math.floor(float(family["die_length_um"]) / float(variant["pitch_um"]))
        cols = math.floor(float(family["die_width_um"]) / float(variant["pitch_um"]))
        print(
            f"Generated {variant['name']}: {variant['dram_tiers']} interfaces, "
            f"{rows}x{cols}={rows * cols:,} total pads/interface"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "stack_variants.yaml",
    )
    args = parser.parse_args()
    generate(args.manifest.resolve())


if __name__ == "__main__":
    main()
