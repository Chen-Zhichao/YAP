#!/usr/bin/env python3
"""
Rebuild design_1 and design_2 from *_old sources with denser bump pitches.

Rules implemented for this rebuild:
- Keep die sizes and stack placement unchanged.
- Keep the old pad-array footprint size (PAD_ARR_W/L_um) unchanged.
- Reduce pitch:
    design_1 -> 5um pitch, 2.5um bump size
    design_2 -> 10um pitch, 5.0um bump size
- Increase pad count accordingly.
- Recreate every .bmap from scratch with placeholder net/port names.
- Reapply design-specific naming + criticality using the existing scripts.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DesignSpec:
    name: str
    old_input_root: Path
    new_input_root: Path
    old_config_root: Path
    new_config_root: Path
    new_pitch_um: float
    new_bump_size_um: float


DESIGN_SPECS = (
    DesignSpec(
        name="design_1",
        old_input_root=REPO_ROOT / "D2W/input/design_1_old",
        new_input_root=REPO_ROOT / "D2W/input/design_1",
        old_config_root=REPO_ROOT / "D2W/configs/design_1_old",
        new_config_root=REPO_ROOT / "D2W/configs/design_1",
        new_pitch_um=5.0,
        new_bump_size_um=2.5,
    ),
    DesignSpec(
        name="design_2",
        old_input_root=REPO_ROOT / "D2W/input/design_2_old",
        new_input_root=REPO_ROOT / "D2W/input/design_2",
        old_config_root=REPO_ROOT / "D2W/configs/design_2_old",
        new_config_root=REPO_ROOT / "D2W/configs/design_2",
        new_pitch_um=10.0,
        new_bump_size_um=5.0,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild dense design_1/design_2 inputs and configs.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Parallel workers for .bmap generation.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete existing target design_1/design_2 directories before rebuilding.",
    )
    return parser.parse_args()


def replace_yaml_scalar(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(\s*{re.escape(key)}:\s*)([^#\n]*)(.*)$", re.MULTILINE)
    updated, count = pattern.subn(rf"\g<1>{value}\g<3>", text)
    if count == 0:
        raise ValueError(f"Key '{key}' not found while updating text.")
    return updated


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def discover_interface_yaml_paths(config_root: Path) -> list[Path]:
    return sorted(
        path
        for path in config_root.glob("*.yaml")
        if "_From_" in path.name or "_To_" in path.name
    )


def parse_interface_geometry(interface_yaml_path: Path) -> dict[str, float]:
    text = read_text(interface_yaml_path)

    def extract(key: str) -> float:
        match = re.search(rf"^{re.escape(key)}:\s*([^\s#]+)", text, re.MULTILINE)
        if not match:
            raise ValueError(f"Missing key {key} in {interface_yaml_path}")
        return float(match.group(1))

    return {
        "DIE_W_um": extract("DIE_W_um"),
        "DIE_L_um": extract("DIE_L_um"),
        "PAD_ARR_W_um": extract("PAD_ARR_W_um"),
        "PAD_ARR_L_um": extract("PAD_ARR_L_um"),
    }


def update_interface_yaml_text(
    text: str,
    *,
    design_name: str,
    new_pitch_um: float,
    new_bump_size_um: float,
    die_w_um: float,
    die_l_um: float,
    pad_arr_w_um: float,
    pad_arr_l_um: float,
) -> str:
    pad_arr_col = int(round(pad_arr_w_um / new_pitch_um)) + 1
    pad_arr_row = int(round(pad_arr_l_um / new_pitch_um)) + 1
    bump_radius_um = new_bump_size_um / 2.0

    updates = {
        "PITCH_r_um": f"{new_pitch_um:.1f}",
        "PITCH_c_um": f"{new_pitch_um:.1f}",
        "DIE_W_um": f"{die_w_um:.4f}".rstrip("0").rstrip("."),
        "DIE_L_um": f"{die_l_um:.4f}".rstrip("0").rstrip("."),
        "PAD_ARR_ROW": str(pad_arr_row),
        "PAD_ARR_COL": str(pad_arr_col),
        "PAD_BOT_R_um": f"{bump_radius_um:.2f}",
        "PAD_TOP_R_um": f"{bump_radius_um:.2f}",
        "PAD_ARR_L_um": f"{pad_arr_l_um:.4f}".rstrip("0").rstrip("."),
        "PAD_ARR_W_um": f"{pad_arr_w_um:.4f}".rstrip("0").rstrip("."),
        "DESIGN": design_name,
    }
    for key, value in updates.items():
        text = replace_yaml_scalar(text, key, value)
    return text


def update_3dbf_text(text: str, *, new_pitch_um: float, new_bump_size_um: float) -> str:
    updates = {
        "pitch": f"{new_pitch_um:.1f}",
        "bump_size": f"{new_bump_size_um:.1f}",
        "through_via_size": f"{new_bump_size_um:.1f}",
    }
    for key, value in updates.items():
        text = replace_yaml_scalar(text, key, value)
    return text


def interface_template_key(design_name: str, bmap_name: str) -> str:
    if design_name == "design_1":
        if "Compute_Small" in bmap_name:
            return "Compute_Small_From_Substrate_Silicon"
        if "Memory_DRAM" in bmap_name:
            return "Memory_DRAM_From_Substrate_Silicon"
    if design_name == "design_2":
        match = re.search(r"Compute_Large_(\d+)", bmap_name)
        if match:
            return f"Compute_Large_{match.group(1)}_From_Substrate_Organic"
    raise ValueError(f"Cannot resolve interface geometry template for {design_name}/{bmap_name}")


def load_bump_type(old_bmap_path: Path) -> str:
    with open(old_bmap_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    parts = first_line.split()
    if len(parts) != 6:
        raise ValueError(f"Unexpected first line in {old_bmap_path}")
    return parts[1]


def generate_bmap(
    out_path: Path,
    interface_name: str,
    bump_type: str,
    die_w_um: float,
    die_l_um: float,
    pad_arr_w_um: float,
    pad_arr_l_um: float,
    pitch_um: float,
) -> None:
    cols = int(round(pad_arr_w_um / pitch_um)) + 1
    rows = int(round(pad_arr_l_um / pitch_um)) + 1

    x0 = (die_w_um - pad_arr_w_um) / 2.0
    y_top = die_l_um - (die_l_um - pad_arr_l_um) / 2.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in range(rows):
            y = y_top - row * pitch_um
            for col in range(cols):
                x = x0 + col * pitch_um
                instance = f"{interface_name}_b_{row}_{col}"
                f.write(
                    f"{instance} {bump_type} {x:.4f} {y:.4f} placeholder placeholder\n"
                )
    tmp.replace(out_path)


def rebuild_design(spec: DesignSpec, jobs: int, keep_existing: bool) -> None:
    if not spec.old_input_root.is_dir():
        raise FileNotFoundError(f"Missing source input root: {spec.old_input_root}")
    if not spec.old_config_root.is_dir():
        raise FileNotFoundError(f"Missing source config root: {spec.old_config_root}")

    if not keep_existing:
        shutil.rmtree(spec.new_input_root, ignore_errors=True)
        shutil.rmtree(spec.new_config_root, ignore_errors=True)

    spec.new_input_root.mkdir(parents=True, exist_ok=True)
    spec.new_config_root.mkdir(parents=True, exist_ok=True)

    # Copy root config YAMLs verbatim, then patch interface YAML geometry.
    for src in sorted(spec.old_config_root.glob("*.yaml")):
        copy_file(src, spec.new_config_root / src.name)

    interface_geom_by_key: dict[str, dict[str, float]] = {}
    for old_if_yaml in discover_interface_yaml_paths(spec.old_config_root):
        interface_geom_by_key[old_if_yaml.stem] = parse_interface_geometry(old_if_yaml)

        new_if_yaml = spec.new_config_root / old_if_yaml.name
        updated = update_interface_yaml_text(
            read_text(old_if_yaml),
            design_name=spec.name,
            new_pitch_um=spec.new_pitch_um,
            new_bump_size_um=spec.new_bump_size_um,
            die_w_um=interface_geom_by_key[old_if_yaml.stem]["DIE_W_um"],
            die_l_um=interface_geom_by_key[old_if_yaml.stem]["DIE_L_um"],
            pad_arr_w_um=interface_geom_by_key[old_if_yaml.stem]["PAD_ARR_W_um"],
            pad_arr_l_um=interface_geom_by_key[old_if_yaml.stem]["PAD_ARR_L_um"],
        )
        write_text(new_if_yaml, updated)

    # Copy static input root files.
    for name in ("generated_chiplet_definitions.3dbv", "generated_stack_config.3dbx", "lef_file.lef", "unsupported_features.txt"):
        src = spec.old_input_root / name
        if src.exists():
            copy_file(src, spec.new_input_root / name)

    # Root .3dbf files with updated pitch/bump size.
    root_3dbfs = sorted(spec.old_input_root.glob("*.3dbf"))
    updated_3dbf_text = {}
    for src in root_3dbfs:
        new_text = update_3dbf_text(
            read_text(src),
            new_pitch_um=spec.new_pitch_um,
            new_bump_size_um=spec.new_bump_size_um,
        )
        updated_3dbf_text[src.name] = new_text
        write_text(spec.new_input_root / src.name, new_text)

    # Mirror ratio/variant directories and copy static variant files.
    for ratio_dir in sorted(path for path in spec.old_input_root.iterdir() if path.is_dir()):
        new_ratio_dir = spec.new_input_root / ratio_dir.name
        new_ratio_dir.mkdir(parents=True, exist_ok=True)
        for variant_dir in sorted(path for path in ratio_dir.iterdir() if path.is_dir()):
            new_variant_dir = new_ratio_dir / variant_dir.name
            new_variant_dir.mkdir(parents=True, exist_ok=True)
            for name in ("generated_chiplet_definitions.3dbv", "generated_stack_config.3dbx"):
                src = variant_dir / name
                if src.exists():
                    copy_file(src, new_variant_dir / name)
            for dbf_name, dbf_text in updated_3dbf_text.items():
                write_text(new_variant_dir / dbf_name, dbf_text)

    # Rebuild every .bmap from scratch.
    tasks = []
    for old_bmap in sorted(spec.old_input_root.rglob("*.bmap")):
        rel = old_bmap.relative_to(spec.old_input_root)
        new_bmap = spec.new_input_root / rel
        key = interface_template_key(spec.name, old_bmap.name)
        geom = interface_geom_by_key[key]
        tasks.append(
            (
                new_bmap,
                old_bmap.stem,
                load_bump_type(old_bmap),
                geom["DIE_W_um"],
                geom["DIE_L_um"],
                geom["PAD_ARR_W_um"],
                geom["PAD_ARR_L_um"],
                spec.new_pitch_um,
            )
        )

    with cf.ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(generate_bmap, *task) for task in tasks]
        for future in cf.as_completed(futures):
            future.result()

    # Apply design-specific naming + criticality regeneration.
    if spec.name == "design_1":
        subprocess.run(
            ["python", str(REPO_ROOT / "D2W/utils/assign_design1_interchip_nets.py"), "--design-root", str(spec.new_input_root)],
            check=True,
            cwd=REPO_ROOT,
        )
    elif spec.name == "design_2":
        subprocess.run(
            ["python", str(REPO_ROOT / "D2W/utils/assign_design2_neighbor_nets.py"), "--design-root", str(spec.new_input_root)],
            check=True,
            cwd=REPO_ROOT,
        )
    else:
        raise ValueError(f"Unsupported design {spec.name}")


def main() -> None:
    args = parse_args()
    for spec in DESIGN_SPECS:
        print(f"=== Rebuilding {spec.name}: pitch={spec.new_pitch_um}um bump={spec.new_bump_size_um}um")
        rebuild_design(spec, jobs=args.jobs, keep_existing=args.keep_existing)
        print(f"=== Finished {spec.name}")


if __name__ == "__main__":
    main()
