#!/usr/bin/env python3
"""Generate normal D2W or F2B W2W inputs, without admitting or running physics."""

import argparse
from copy import deepcopy
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

from mechanics_runtime import user_run_root, validate_user_write_path

if __name__ == '__main__':
    user_run_root()

from omegaconf import OmegaConf

from D2W.assembly_order import build_plan, load_stack_input
from D2W.compact_layout import compact_regular_geometry
from W2W.assembly_order import parse_order, tree_text


PROJECT_ROOT = Path("/app/nanocad/projects/personal/zchen/YAP_STACK")
ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "D2W/input/design_6_4hi_p2"
CONFIG_TEMPLATE = ROOT / "D2W/configs/design_6_4hi_p2/design_6_4hi_p2.yaml"
CALIBRATION_NOTE = (
    "# Generated experimental prepared-die column; no explicit receiving wafer.\n"
    "# Fabrication E/nu are borrowed TOTAL50 calibration assumptions, not\n"
    "# verified material properties for this geometry. Both BIs already exist.\n"
    "# Operator availability, thermal accuracy and assembly yields are not validated.\n"
)


def _canonical_path(value):
    return validate_user_write_path(value, label='generated design')


def _order(selection, count):
    if selection == "all":
        if count > 8:
            raise ValueError("all-orders enumeration supports at most eight dies")
        return "all"
    if selection == "sequential":
        tree = 1
        for index in range(2, count + 1):
            tree = (tree, index)
    elif selection == "balanced":
        def split(first, last):
            if first == last:
                return first
            middle = (first + last) // 2
            return (split(first, middle), split(middle + 1, last))
        tree = split(1, count)
    else:
        tree = parse_order(selection, count)
    return tree_text(tree)


def _write_yaml(path, document, note=""):
    with path.open("x", encoding="ascii") as stream:
        stream.write(note)
        stream.write(OmegaConf.to_yaml(OmegaConf.create(document), resolve=True))


def generate_design(*, name, dies, pitch, total_thickness, initial_bows,
                    orders="sequential", operator_package=None, legacy_total50=False,
                    output_root=None, input_root=None, config_root=None, order_workers=1,
                    bonding_mode="d2w"):
    """Create new input/config directories; never replace an existing design.

    Totals include front BI + BEOL + Si + back BI. Generating a component
    reference does not certify its dual-BI assembly or install its operands.
    """
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", name):
        raise ValueError("name must be a simple ASCII design identifier")
    if type(dies) is not int or not 2 <= dies <= 32:
        raise ValueError("dies must be an integer in 2..32")
    if type(pitch) is not int or not 2 <= pitch <= 10:
        raise ValueError("pitch must be an integer in 2..10 um")
    if type(total_thickness) not in (int, float) or not math.isfinite(total_thickness) or total_thickness <= 13:
        raise ValueError("total thickness must include both BIs, BEOL and positive Si (>13 um)")
    if not isinstance(initial_bows, (list, tuple)) or len(initial_bows) != dies or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in initial_bows):
        raise ValueError("initial bows must contain exactly N finite signed numbers")
    if type(legacy_total50) is not bool:
        raise ValueError("legacy_total50 must be Boolean")
    if type(order_workers) is not int or order_workers < 1:
        raise ValueError("order_workers must be a positive integer")
    if bonding_mode not in ("d2w", "w2w"):
        raise ValueError("bonding_mode must be d2w or w2w")
    domain = bonding_mode.upper()
    wafer_mode = bonding_mode == "w2w"
    template = ROOT / "W2W/input/design_6_4hi_p2" if wafer_mode else TEMPLATE
    config_template = ROOT / "W2W/configs/design_6_4hi_p2/design_6_4hi_p2.yaml" if wafer_mode else CONFIG_TEMPLATE
    prefix = "Wafer" if wafer_mode else "Die"
    initial_total = 775. if wafer_mode else float(total_thickness)
    note = ("# Generated experimental F2B wafer column; 775-um total birth.\n"
            "# Pure-Si caps are removed before lower-back BI preparation.\n"
            "# Unit-component bow passes do not qualify assembly accuracy.\n") if wafer_mode else CALIBRATION_NOTE
    selection = _order(orders, dies)
    if legacy_total50:
        if pitch != 2 or total_thickness != 50 or operator_package is not None:
            raise ValueError("legacy TOTAL50 requires pitch 2, total 50 and no operator-package override")
        package = "resource:compact_p2/packages/BI2_50.json"
    else:
        silicon = format(total_thickness - 13., ".12g").replace(".", "p")
        package = operator_package if operator_package is not None else f"component:BI{pitch}_SI{silicon}"
        if not isinstance(package, str) or not re.fullmatch(r"component:[A-Za-z0-9][A-Za-z0-9_.-]*", package):
            raise ValueError("operator-package must be a component:<id>; use --legacy-total50 for legacy data")
    if (input_root is None) != (config_root is None):
        raise ValueError("input-root and config-root must be supplied together")
    if output_root is not None and input_root is not None:
        raise ValueError("choose output-root or the input-root/config-root pair")
    root = _canonical_path(output_root or user_run_root() / "designs")
    inputs = _canonical_path(input_root) if input_root is not None else root / domain / "input"
    configs = _canonical_path(config_root) if config_root is not None else root / domain / "configs"
    directory, config_directory = inputs / name, configs / name
    if directory == config_directory or directory.is_relative_to(config_directory) or config_directory.is_relative_to(directory):
        raise ValueError("input and config design directories must not overlap")
    for path in (directory, config_directory):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing design: {path}")

    design = OmegaConf.to_container(OmegaConf.load(template / "stack.yaml"), resolve=True)
    stack = design["stack"]
    stack.update(initial_bows_um=list(initial_bows), assembly_orders=selection)
    if wafer_mode:
        stack["geometry"]["target_total_thickness_um"] = float(total_thickness)
    else:
        stack["allow_offline_substacks"] = True
        stack["geometry"]["total_thickness_um"] = float(total_thickness)
        stack["fabrication_layers"][2]["thickness_um"] = float(total_thickness - 13.)
    design["mechanics"]["operator_package"] = package
    design["mechanics"]["temporal_profile"] = "distilled26" if legacy_total50 and not wafer_mode else "component_graded25"
    design["mechanics"]["order_workers"] = order_workers
    design["mechanics"].pop("frozen_preview", None)
    if legacy_total50:
        design["mechanics"]["frozen_preview"] = True
    normalized = deepcopy(stack)
    if wafer_mode:
        from W2W.assembly_order import load_stack_input as input_loader, build_plan as planner
    else:
        input_loader, planner = load_stack_input, build_plan
    normalized["schema"] = "yap.assembly_order_input.v1" if wafer_mode else "yap.d2w_assembly_order_input.v1"
    for key, value in input_loader(template / "stack.yaml")["stack"]["process"].items():
        normalized["process"].setdefault(key, value)
    if wafer_mode:
        from W2W.assembly_order import INPUT_SCHEMA
        normalized["schema"] = INPUT_SCHEMA
    plan = planner(normalized)

    config = OmegaConf.load(config_template)
    for settings in config.values():
        settings.update(DESIGN=name, INPUT_DIR=str(directory),
                        OUTPUT_DIR=str(user_run_root() / "results" / domain),
                        PITCH_r_um=float(pitch), PITCH_c_um=float(pitch),
                        PAD_TOP_R_um=pitch / 4., PAD_BOT_R_um=pitch / 4.,
                        PAD_ARR_ROW=math.floor(10000. / pitch),
                        PAD_ARR_COL=math.floor(10000. / pitch))
        for field in ("ITF_TOP_THICK_um", "ITF_BOT_THICK_um", "DIE_THICHKNESS_um"):
            if field in settings:
                settings[field] = initial_total
    if wafer_mode:
        from W2W.compact_layout import compact_regular_geometry as compact_geometry
    else:
        compact_geometry = compact_regular_geometry
    compact = compact_geometry(config[bonding_mode + "_modeling"])
    layout = OmegaConf.to_container(OmegaConf.load(template / "compact_layout.yaml"), resolve=True)
    for key in ("rows", "cols", "pitch_r_um", "pitch_c_um", "pad_array_l_um", "pad_array_w_um"):
        layout[key] = compact[key]
    layout.update(pad_diameter_um=pitch / 2., bonding_interfaces=dies - 1)
    layout["wafer_count" if wafer_mode else "die_count"] = dies
    for category in ("critical", "power_ground", "dummy"):
        layout[f"num_{category}_pads_per_interface"] = compact[f"num_{category}_pads"]
    layout["num_total_pads_per_interface"] = compact["rows"] * compact["cols"]
    definitions = OmegaConf.to_container(OmegaConf.load(template / "generated_chiplet_definitions.3dbv"), resolve=True)
    prototype = deepcopy(definitions["ChipletDef"][f"{prefix}_2"])
    definitions["ChipletDef"] = {}
    topology = OmegaConf.to_container(OmegaConf.load(template / "generated_stack_config.3dbx"), resolve=True)
    topology.update(Design={"name": name}, ChipletInst={}, Stack={}, Connection={})
    for index in range(1, dies + 1):
        key = f"{prefix}_{index}"
        definition = deepcopy(prototype)
        definition["thickness"] = initial_total
        definition["regions"] = {}
        if index > 1:
            definition["regions"][f"From_{prefix}_{index - 1}"] = deepcopy(prototype["regions"][f"From_{prefix}_1"])
        if index < dies:
            definition["regions"][f"To_{prefix}_{index + 1}"] = deepcopy(prototype["regions"][f"To_{prefix}_3"])
        definitions["ChipletDef"][key] = definition
        topology["ChipletInst"][key] = {"reference": key, "is_master": True}
        topology["Stack"][key] = {"loc": [0., 0.], "z": (index - 1) * initial_total, "orient": "R0"}
        if index == 1:
            topology["Stack"][key]["root"] = True
        else:
            topology["Connection"][f"Conn{index - 1}"] = {
                "top": f"{key}.regions.From_{prefix}_{index - 1}",
                "bot": f"{prefix}_{index - 1}.regions.To_{key}"}
    technology = OmegaConf.to_container(OmegaConf.load(template / f"{prefix}_1.3dbf"), resolve=True)
    technology["Chiplet_Grid"]["pitch"] = float(pitch)
    bump = technology["Bump_Types"]["silicon_individual_bonding"]
    bump.update(bump_size=pitch / 2., through_via_size=pitch / 2.)

    import os
    os.umask(0o077)
    user_run_root().mkdir(mode=0o700, parents=True, exist_ok=True)
    user_run_root()
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    config_directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    _write_yaml(directory / "stack.yaml", design, note)
    _write_yaml(directory / "generated_chiplet_definitions.3dbv", definitions)
    _write_yaml(directory / "generated_stack_config.3dbx", topology)
    _write_yaml(directory / "compact_layout.yaml", layout)
    for index in range(1, dies + 1):
        _write_yaml(directory / f"{prefix}_{index}.3dbf", technology)
    config_path = config_directory / f"{name}.yaml"
    _write_yaml(config_path, config, note)
    return dict(input_dir=directory, config_path=config_path, orders=len(plan["routes"]),
                operator_package=package)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dies", "--wafers", dest="dies", required=True, type=int)
    parser.add_argument("--bonding-mode", choices=("d2w", "w2w"), default="d2w")
    parser.add_argument("--pitch", required=True, type=int)
    parser.add_argument("--total-thickness", required=True, type=float, help="um including both BIs and BEOL")
    parser.add_argument("--initial-bows", required=True, type=float, nargs="+", help="N signed bows in um")
    parser.add_argument("--orders", default="sequential", help="sequential, balanced, all, or a quoted fixed-order tree")
    parser.add_argument("--order-workers", default=1, type=int, help="concurrent assembly orders; default 1 (serial)")
    parser.add_argument("--operator-package")
    parser.add_argument("--legacy-total50", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--config-root", type=Path)
    args = parser.parse_args(argv)
    try:
        result = generate_design(**vars(args))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Inputs: {result['input_dir']}")
    print(f"Full config: {result['config_path']}")
    print(f"Selected orders: {result['orders']}; package: {result['operator_package']}")
    print("Inputs generated only; no operators loaded, mechanics run, or assembly qualification granted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
