#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from utils.util import *
import time
import argparse
from assembly_yield_calculator import Assembly_Yield_Calculator


def parse_args():
    p = argparse.ArgumentParser(description="Simulate assembly yield for D2W hybrid bonding")
    p.add_argument("--config", "-c", required=True, help="Path to skeleton config YAML file")
    p.add_argument("--mode", "-m", required=True, help="Mode to load from config (default: d2w_modeling)")
    p.add_argument("--ds_name", "-d", required=True, help="Name of design (used for output directory naming)")
    p.add_argument("--ds_dir", required=True, help="Path to design directory")
    p.add_argument("--plot", "-plot", default=False, action="store_true", help="Enable plotting of the pad risk map")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output during simulation")
    p.add_argument("--debug", action="store_true", help="Enable debug output when loading config")
    return p.parse_args()


def main():
    args = parse_args()

    # Extract the design input files directory if provided
    input_ds_dir = args.ds_dir
    # Determine .3dbv path (chiplet definitions)
    _3dbv_path = input_ds_dir + "/generated_chiplet_definitions.3dbv"
    # Determine .3dbx path (stack config)
    _3dbx_path = input_ds_dir + "/generated_stack_config.3dbx"
    # Read the config skeleton and update with design parameters
    cfg_skeleton = OmegaConf.load(args.config)[args.mode]

    start_time = time.time()
    # Load config and update with design and ADK parameters (from .3dbv and .bmap)
    cfg_dict = get_config_dict(cfg_folder=args.config.rsplit('/', 1)[0],
                                cfg_skeleton=cfg_skeleton, 
                                ds_name=args.ds_name,
                                input_ds_dir=input_ds_dir,
                                _3dbv_path=_3dbv_path,
                                _3dbx_path=_3dbx_path,
                                mode=args.mode, 
                                debug=args.debug)
    cfg_loading_time = time.time()
    print(f"Config loading and processing finished in {cfg_loading_time - start_time:.2f} seconds.")

    # Plotting flag
    for cfg in cfg_dict.values():
        cfg.plot_flag = args.plot
    
    # Create output directory if it doesn't exist
    for cfg in cfg_dict.values():
        output_path = os.path.join(cfg.OUTPUT_DIR, args.ds_name, cfg.INTERFACE)
        os.makedirs(output_path, exist_ok=True)


    bmap_path_dict = {}
    criticality_path_dict = {}
    pad_bitmap_collection_dict = {}
    # Step 1: convert .bmap -> pad bitmap collection
    for interface, cfg in cfg_dict.items():
        bmap_path_dict[interface] = os.path.join(input_ds_dir, f"{cfg.INTERFACE}.bmap")
        criticality_path_dict[interface] = os.path.join(input_ds_dir, f"{cfg.INTERFACE}_criticality.txt")
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
        input_args=vars(args),
        cfg_dict=cfg_dict,
        pad_bitmap_collection_dict=pad_bitmap_collection_dict,                                             
    )
    print(f"Modeled yield: {stack_assembly_yield:.6f}")
    print(f"Yield modeling finished in {time.time() - start_time:.2f} s")
    print_run_separator(f"D2W yield modeling finished for {args.ds_name}")

if __name__ == "__main__":
    main()
