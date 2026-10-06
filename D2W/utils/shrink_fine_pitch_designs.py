#!/usr/bin/env python3
"""
Shrink fine-pitch design_1_p5/design_2_p10 in-plane geometry while keeping pitch/bump size fixed.

Rules
-----
- design_1_p5: shrink all die and substrate XY dimensions to 1/2.
- design_2_p10: shrink all die and substrate XY dimensions to 1/3.
- Keep pitch unchanged:
    design_1_p5 -> 5um pitch, 2.5um bump size
    design_2_p10 -> 10um pitch, 5.0um bump size
- Recompute pad-array size from the shrunken pad-array footprint.
- Rewrite root + variant .3dbf/.3dbv/.3dbx files.
- Rebuild every .bmap from scratch.
- Reapply design-specific naming/criticality scripts afterwards.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ShrinkSpec:
    name: str
    design_kind: str
    input_root: Path
    config_root: Path
    scale_xy: float
    pitch_um: float
    bump_size_um: float
    source_name: str | None = None
    source_input_root: Path | None = None
    source_config_root: Path | None = None


SPECS = (
    ShrinkSpec(
        name="design_1_p5",
        design_kind="design_1",
        input_root=REPO_ROOT / "D2W/input/design_1_p5",
        config_root=REPO_ROOT / "D2W/configs/design_1_p5",
        scale_xy=0.5,
        pitch_um=5.0,
        bump_size_um=2.5,
    ),
    ShrinkSpec(
        name="design_2_p10",
        design_kind="design_2",
        input_root=REPO_ROOT / "D2W/input/design_2_p10",
        config_root=REPO_ROOT / "D2W/configs/design_2_p10",
        scale_xy=1.0 / 3.0,
        pitch_um=10.0,
        bump_size_um=5.0,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shrink fine-pitch design_1_p5/design_2_p10 geometry.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Parallel workers for .bmap generation.",
    )
    parser.add_argument("--source-design", help="Source design directory name under D2W/configs and D2W/input.")
    parser.add_argument("--target-design", help="Target design directory name under D2W/configs and D2W/input.")
    parser.add_argument("--design-kind", choices=["design_1", "design_2"], help="Design family for net regeneration.")
    parser.add_argument("--scale-xy", type=float, help="In-plane XY shrink factor.")
    parser.add_argument("--pitch-um", type=float, help="Pad pitch in micron for regenerated interface configs.")
    parser.add_argument("--bump-size-um", type=float, help="Bump size in micron for regenerated interface configs.")
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def clone_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        subprocess.run(["rm", "-rf", str(dst)], check=True)
    subprocess.run(["cp", "-a", str(src), str(dst)], check=True)


def replace_yaml_scalar(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(\s*{re.escape(key)}:\s*)([^#\n]*)(.*)$", re.MULTILINE)
    updated, count = pattern.subn(rf"\g<1>{value}\g<3>", text)
    if count == 0:
        raise ValueError(f"Key '{key}' not found while updating text.")
    return updated


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


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def update_interface_yaml_text(
    text: str,
    *,
    design_name: str,
    pitch_um: float,
    bump_size_um: float,
    scale_xy: float,
    geom: dict[str, float],
) -> str:
    die_w_um = geom["DIE_W_um"] * scale_xy
    die_l_um = geom["DIE_L_um"] * scale_xy
    pad_arr_w_um = geom["PAD_ARR_W_um"] * scale_xy
    pad_arr_l_um = geom["PAD_ARR_L_um"] * scale_xy

    pad_arr_col = int(round(pad_arr_w_um / pitch_um)) + 1
    pad_arr_row = int(round(pad_arr_l_um / pitch_um)) + 1
    bump_radius_um = bump_size_um / 2.0
    eff_die_r = math.sqrt((die_w_um / 2.0) ** 2 + (die_l_um / 2.0) ** 2)
    s_init_a_m = 10e-6 * (eff_die_r / 150000.0) ** 2

    updates = {
        "PITCH_r_um": f"{pitch_um:.1f}",
        "PITCH_c_um": f"{pitch_um:.1f}",
        "DIE_W_um": fmt(die_w_um),
        "DIE_L_um": fmt(die_l_um),
        "PAD_ARR_ROW": str(pad_arr_row),
        "PAD_ARR_COL": str(pad_arr_col),
        "PAD_BOT_R_um": f"{bump_radius_um:.2f}",
        "PAD_TOP_R_um": f"{bump_radius_um:.2f}",
        "PAD_ARR_L_um": fmt(pad_arr_l_um),
        "PAD_ARR_W_um": fmt(pad_arr_w_um),
        "eff_DIE_R": repr(eff_die_r),
        "S_INIT_A_M": repr(s_init_a_m),
        "DESIGN": design_name,
    }
    for key, value in updates.items():
        text = replace_yaml_scalar(text, key, value)
    return text


def scale_point_pair(seq: list[float], scale_xy: float) -> list[float]:
    return [round(float(seq[0]) * scale_xy, 6), round(float(seq[1]) * scale_xy, 6)]


def update_3dbf_text(text: str, *, scale_xy: float) -> str:
    data = yaml.safe_load(text)
    for area in data.get("Voltage_Areas", {}).values():
        coords = area.get("coords")
        if coords:
            area["coords"] = [scale_point_pair(point, scale_xy) for point in coords]
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


def update_3dbv_text(text: str, *, scale_xy: float) -> str:
    data = yaml.safe_load(text)
    for chiplet in data.get("ChipletDef", {}).values():
        design_area = chiplet.get("design_area")
        if design_area:
            chiplet["design_area"] = [round(float(v) * scale_xy, 6) for v in design_area]
        for region in chiplet.get("regions", {}).values():
            coords = region.get("coords")
            if coords:
                region["coords"] = [scale_point_pair(point, scale_xy) for point in coords]
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


def update_3dbx_text(text: str, *, scale_xy: float) -> str:
    data = yaml.safe_load(text)
    for stack_item in data.get("Stack", {}).values():
        loc = stack_item.get("loc")
        if loc:
            stack_item["loc"] = [round(float(v) * scale_xy, 6) for v in loc]
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


def interface_template_key(design_kind: str, bmap_name: str) -> str:
    if design_kind == "design_1":
        if "Compute_Small" in bmap_name:
            return "Compute_Small_From_Substrate_Silicon"
        if "Memory_DRAM" in bmap_name:
            return "Memory_DRAM_From_Substrate_Silicon"
    if design_kind == "design_2":
        match = re.search(r"Compute_Large_(\d+)", bmap_name)
        if match:
            return f"Compute_Large_{match.group(1)}_From_Substrate_Organic"
    raise ValueError(f"Cannot resolve interface geometry template for {design_kind}/{bmap_name}")


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


def rebuild_design(spec: ShrinkSpec, jobs: int) -> None:
    source_input_root = spec.source_input_root or spec.input_root
    source_config_root = spec.source_config_root or spec.config_root
    source_name = spec.source_name or spec.name

    if not source_input_root.is_dir():
        raise FileNotFoundError(f"Missing input root: {source_input_root}")
    if not source_config_root.is_dir():
        raise FileNotFoundError(f"Missing config root: {source_config_root}")

    if source_input_root != spec.input_root:
        clone_tree(source_input_root, spec.input_root)
    if source_config_root != spec.config_root:
        clone_tree(source_config_root, spec.config_root)

    for old_name, new_name in (
        (f"{source_name}.yaml", f"{spec.name}.yaml"),
        (f"{source_name}_overlay_pessimistic.yaml", f"{spec.name}_overlay_pessimistic.yaml"),
        (f"{source_name}_particle_pessimistic.yaml", f"{spec.name}_particle_pessimistic.yaml"),
        (f"{source_name}_mechanical_pessimistic.yaml", f"{spec.name}_mechanical_pessimistic.yaml"),
        (f"{source_name}_ESD_pessimistic.yaml", f"{spec.name}_ESD_pessimistic.yaml"),
    ):
        old_path = spec.config_root / old_name
        new_path = spec.config_root / new_name
        if old_path.exists() and old_path != new_path:
            old_path.replace(new_path)

    for yaml_path in spec.config_root.glob("*.yaml"):
        text = read_text(yaml_path).replace(source_name, spec.name)
        write_text(yaml_path, text)

    interface_geom_by_key: dict[str, dict[str, float]] = {}
    for if_yaml in sorted(path for path in spec.config_root.glob("*.yaml") if "_From_" in path.name or "_To_" in path.name):
        interface_geom_by_key[if_yaml.stem] = parse_interface_geometry(if_yaml)
        updated = update_interface_yaml_text(
            read_text(if_yaml),
            design_name=spec.name,
            pitch_um=spec.pitch_um,
            bump_size_um=spec.bump_size_um,
            scale_xy=spec.scale_xy,
            geom=interface_geom_by_key[if_yaml.stem],
        )
        write_text(if_yaml, updated)

    root_3dbf_text: dict[str, str] = {}
    for root_3dbf in sorted(spec.input_root.glob("*.3dbf")):
        new_text = update_3dbf_text(read_text(root_3dbf), scale_xy=spec.scale_xy)
        root_3dbf_text[root_3dbf.name] = new_text
        write_text(root_3dbf, new_text)

    root_3dbv = spec.input_root / "generated_chiplet_definitions.3dbv"
    if root_3dbv.exists():
        root_3dbv_text = update_3dbv_text(read_text(root_3dbv), scale_xy=spec.scale_xy)
        write_text(root_3dbv, root_3dbv_text)
    else:
        root_3dbv_text = None

    root_3dbx = spec.input_root / "generated_stack_config.3dbx"
    if root_3dbx.exists():
        root_3dbx_text = update_3dbx_text(read_text(root_3dbx), scale_xy=spec.scale_xy)
        write_text(root_3dbx, root_3dbx_text)
    else:
        root_3dbx_text = None

    for ratio_dir in sorted(path for path in spec.input_root.iterdir() if path.is_dir()):
        for variant_dir in sorted(path for path in ratio_dir.iterdir() if path.is_dir()):
            for dbf_name, dbf_text in root_3dbf_text.items():
                write_text(variant_dir / dbf_name, dbf_text)
            if root_3dbv_text is not None:
                write_text(variant_dir / "generated_chiplet_definitions.3dbv", root_3dbv_text)
            if root_3dbx_text is not None:
                write_text(variant_dir / "generated_stack_config.3dbx", root_3dbx_text)

    tasks = []
    for existing_bmap in sorted(spec.input_root.rglob("*.bmap")):
        key = interface_template_key(spec.design_kind, existing_bmap.name)
        old_geom = interface_geom_by_key[key]
        scaled_geom = {
            "die_w": old_geom["DIE_W_um"] * spec.scale_xy,
            "die_l": old_geom["DIE_L_um"] * spec.scale_xy,
            "pad_w": old_geom["PAD_ARR_W_um"] * spec.scale_xy,
            "pad_l": old_geom["PAD_ARR_L_um"] * spec.scale_xy,
        }
        tasks.append(
            (
                existing_bmap,
                existing_bmap.stem,
                load_bump_type(existing_bmap),
                scaled_geom["die_w"],
                scaled_geom["die_l"],
                scaled_geom["pad_w"],
                scaled_geom["pad_l"],
                spec.pitch_um,
            )
        )

    with cf.ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(generate_bmap, *task) for task in tasks]
        for future in cf.as_completed(futures):
            future.result()

    if spec.design_kind == "design_1":
        subprocess.run(
            ["python", str(REPO_ROOT / "D2W/utils/assign_design1_interchip_nets.py"), "--design-root", str(spec.input_root)],
            check=True,
            cwd=REPO_ROOT,
        )
    elif spec.design_kind == "design_2":
        subprocess.run(
            ["python", str(REPO_ROOT / "D2W/utils/assign_design2_neighbor_nets.py"), "--design-root", str(spec.input_root)],
            check=True,
            cwd=REPO_ROOT,
        )
    else:
        raise ValueError(f"Unsupported design kind: {spec.design_kind}")


def main() -> None:
    args = parse_args()
    if args.source_design or args.target_design:
        required = [
            ("--source-design", args.source_design),
            ("--target-design", args.target_design),
            ("--design-kind", args.design_kind),
            ("--scale-xy", args.scale_xy),
            ("--pitch-um", args.pitch_um),
            ("--bump-size-um", args.bump_size_um),
        ]
        missing = [flag for flag, value in required if value is None]
        if missing:
            raise SystemExit(f"Missing required arguments for custom shrink: {', '.join(missing)}")
        specs = (
            ShrinkSpec(
                name=args.target_design,
                design_kind=args.design_kind,
                input_root=REPO_ROOT / "D2W/input" / args.target_design,
                config_root=REPO_ROOT / "D2W/configs" / args.target_design,
                scale_xy=args.scale_xy,
                pitch_um=args.pitch_um,
                bump_size_um=args.bump_size_um,
                source_name=args.source_design,
                source_input_root=REPO_ROOT / "D2W/input" / args.source_design,
                source_config_root=REPO_ROOT / "D2W/configs" / args.source_design,
            ),
        )
    else:
        specs = SPECS

    for spec in specs:
        print(f"=== Shrinking {spec.name}: scale={spec.scale_xy} pitch={spec.pitch_um} bump={spec.bump_size_um}")
        rebuild_design(spec, jobs=args.jobs)
        print(f"=== Finished {spec.name}")


if __name__ == "__main__":
    main()
