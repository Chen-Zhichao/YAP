#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from utils.util import *
import time
import argparse
from pathlib import Path
import sys
from assembly_yield_calculator import Assembly_Yield_Calculator
from chiplet_yield import (
    chiplet_stack_yield,
    chiplet_yield_records,
    print_chiplet_yield_table,
)
from yield_mechanism_policy import active_failure_mechanisms


def _resolve_design_root_and_layout_dir(ds_dir: str) -> tuple[str, str]:
    layout_dir = ds_dir.rstrip("/")
    design_root = layout_dir
    required = ["generated_chiplet_definitions.3dbv", "generated_stack_config.3dbx"]
    if all(os.path.exists(os.path.join(design_root, filename)) for filename in required):
        return design_root, layout_dir

    parent_dir = os.path.dirname(design_root)
    if all(os.path.exists(os.path.join(parent_dir, filename)) for filename in required):
        return parent_dir, layout_dir

    raise FileNotFoundError(
        f"Could not find generated 3Dblox files in '{layout_dir}' or its parent."
    )


def _resolve_module_output_dir(output_dir: str) -> str:
    output_dir = os.fspath(output_dir)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
    output_dir = os.path.normpath(output_dir)
    return output_dir + os.sep


def _anchor_output_dirs_to_module(cfg_dict: dict) -> None:
    for cfg in cfg_dict.values():
        cfg.OUTPUT_DIR = _resolve_module_output_dir(cfg.OUTPUT_DIR)


def parse_args():
    p = argparse.ArgumentParser(description="Simulate assembly yield for D2W hybrid bonding")
    p.add_argument("--config", "-c", required=True, help="Path to skeleton config YAML file")
    p.add_argument("--mode", "-m", required=True, help="Mode to load from config (default: d2w_modeling)")
    p.add_argument("--ds_name", "-d", required=True, help="Name of design (used for output directory naming)")
    p.add_argument("--ds_dir", required=True, help="Path to design directory")
    p.add_argument("--plot", "-plot", default=False, action="store_true", help="Enable plotting of the pad risk map")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output during simulation")
    p.add_argument(
        "--mechanism-filter",
        default="all",
        help=(
            "Comma-separated failure mechanisms to model. "
            "Use all, overlay, particle, mechanical, chiplet, ESD, or warpage. "
            "ESD and final-warpage yield are temporarily disabled; warpage "
            "still contributes to overlay. "
            "Example: --mechanism-filter particle,mechanical"
        ),
    )
    p.add_argument("--debug", action="store_true", help="Enable debug output when loading config")
    p.add_argument("--plan-only", action="store_true", help="Assembly-order designs: validate and emit the plan only")
    p.add_argument("--output-dir", help="Assembly-order designs: override the run output root")
    return p.parse_args()


def main():
    args = parse_args()

    design_input_dir, layout_input_dir = _resolve_design_root_and_layout_dir(args.ds_dir)
    # Determine .3dbv path (chiplet definitions)
    _3dbv_path = design_input_dir + "/generated_chiplet_definitions.3dbv"
    # Determine .3dbx path (stack config)
    _3dbx_path = design_input_dir + "/generated_stack_config.3dbx"
    # Read the config skeleton and update with design parameters
    cfg_skeleton = OmegaConf.load(args.config)[args.mode]
    if cfg_skeleton.get("ASSEMBLY_ORDER_ENABLED", False):
        if cfg_skeleton.ASSEMBLY_ORDER_ENABLED is not True:
            raise ValueError("ASSEMBLY_ORDER_ENABLED must be a Boolean")
        if args.mode != "d2w_modeling":
            raise ValueError("assembly-order designs support modeling only; use --mode d2w_modeling")
        if args.mechanism_filter not in ("all", "overlay"):
            raise ValueError("assembly-order mode currently evaluates overlay yield only")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from D2W.assembly_order import run_design
        run_design(cfg_skeleton, design_input_dir, args.ds_name, plan_only=args.plan_only,
                   output_dir=args.output_dir, config_path=args.config)
        return
    if args.plan_only or args.output_dir:
        raise ValueError("--plan-only/--output-dir require an assembly-order-enabled design")

    start_time = time.time()
    # Load config and update with design and ADK parameters (from .3dbv and .bmap)
    cfg_dict = get_config_dict(cfg_folder=args.config.rsplit('/', 1)[0],
                                cfg_skeleton=cfg_skeleton, 
                                ds_name=args.ds_name,
                                input_ds_dir=design_input_dir,
                                _3dbv_path=_3dbv_path,
                                _3dbx_path=_3dbx_path,
                                mode=args.mode, 
                                debug=args.debug,
                                bmap_input_ds_dir=layout_input_dir)
    cfg_loading_time = time.time()
    print(f"Config loading and processing finished in {cfg_loading_time - start_time:.2f} seconds.")

    _anchor_output_dirs_to_module(cfg_dict)

    # Plotting flag
    for cfg in cfg_dict.values():
        cfg.plot_flag = args.plot
    
    # Create output directory if it doesn't exist
    for cfg in cfg_dict.values():
        output_path = os.path.join(cfg.OUTPUT_DIR, args.ds_name, cfg.INTERFACE)
        os.makedirs(output_path, exist_ok=True)

    input_args = vars(args)
    input_args["design_root_dir"] = design_input_dir
    input_args["layout_ds_dir"] = layout_input_dir
    if active_failure_mechanisms(input_args) == {"chiplet"}:
        records = chiplet_yield_records(cfg_dict, design_input_dir)
        stack_assembly_yield = chiplet_stack_yield(records)
        print_chiplet_yield_table(records)
        print("Calculated bonding stack yield: 1.000000")
        print(f"Calculated total stack yield: {stack_assembly_yield:.6f}")
        print(f"Modeled yield: {stack_assembly_yield:.6f}")
        print_run_separator(f"D2W yield modeling finished for {args.ds_name}")
        return


    bmap_path_dict = {}
    criticality_path_dict = {}
    pad_bitmap_collection_dict = {}
    # Step 1: convert .bmap -> pad bitmap collection
    for interface, cfg in cfg_dict.items():
        bmap_path_dict[interface] = os.path.join(layout_input_dir, f"{cfg.INTERFACE}.bmap")
        criticality_path_dict[interface] = os.path.join(layout_input_dir, f"{cfg.INTERFACE}_criticality.txt")
        pad_bitmap_collection_dict[interface] = convert_3dblox_to_pad_bitmap(cfg=cfg,
                                                            _bmap_path=bmap_path_dict[interface],
                                                            criticality_path=criticality_path_dict[interface],
                                                            pad_arrange_pattern=cfg.PAD_ARRANGE_PATTERN)
    convert_time = time.time()
    print("Pad bitmap collection generation finished in {:.2f} seconds.".format(convert_time - cfg_loading_time))

    # Step 2: calculate stack assembly yield
    print("Calculating stack assembly yield...")
    start_time = time.time()
    stack_assembly_yield, stack_assembly_yield_list = Assembly_Yield_Calculator(
        input_args=input_args,
        cfg_dict=cfg_dict,
        pad_bitmap_collection_dict=pad_bitmap_collection_dict,                                             
    )
    print(f"Modeled yield: {stack_assembly_yield:.6f}")
    print(f"Yield modeling finished in {time.time() - start_time:.2f} s")
    print_run_separator(f"D2W yield modeling finished for {args.ds_name}")

if __name__ == "__main__":
    main()
