#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#### Author: Zhichao Chen
#### Date: Oct 23, 2025

import os
import time

from defect_yield_calculator import stack_defect_yield_calculator
from overlay_yield_calculator import stack_overlay_yield_calculator
from Cu_expansion_yield_calculator import stack_cu_expansion_yield_calculator
from esd_yield_calculator import stack_esd_yield_calculator
from warpage_yield_calculator import stack_warpage_yield_calculator
from wafer_die_stack_initialization import DieStack


def Assembly_Yield_Calculator(
    input_args: dict,
    cfg_dict: dict,
    pad_bitmap_collection_dict: dict,
):
    start_time = time.time()

    die_stack = DieStack(
        cfg_dict=cfg_dict,
        pad_bitmap_collection_dict=pad_bitmap_collection_dict,
        mode=input_args["mode"],
        base_pad_coords_flag=True,
    )
    init_time = time.time() - start_time
    print("Die stack initialization time: {:.2f} seconds.".format(init_time))

    ds_dir = input_args.get("design_root_dir", input_args.get("ds_dir", ""))
    _3dbx_path = os.path.join(ds_dir, "generated_stack_config.3dbx") if ds_dir else None
    stack_overlay_yield_calculator(
        cfg_dict=cfg_dict,
        die_stack=die_stack,
        _3dbx_path=_3dbx_path,
    )
    overlay_time = time.time() - start_time - init_time
    print("Overlay yield calculation time: {:.2f} seconds.".format(overlay_time))

    stack_defect_yield_calculator(
        cfg_dict=cfg_dict,
        die_stack=die_stack,
    )
    defect_time = time.time() - start_time - init_time - overlay_time
    print("Particle yield calculation time: {:.2f} seconds.".format(defect_time))

    stack_cu_expansion_yield_calculator(
        cfg_dict=cfg_dict,
        die_stack=die_stack,
    )
    mechanical_time = time.time() - start_time - init_time - overlay_time - defect_time
    print("Mechanical yield calculation time: {:.2f} seconds.".format(mechanical_time))

    stack_esd_yield_calculator(
        cfg_dict=cfg_dict,
        die_stack=die_stack,
    )
    esd_time = time.time() - start_time - init_time - overlay_time - defect_time - mechanical_time
    print("ESD yield calculation time: {:.2f} seconds.".format(esd_time))

    stack_warpage_yield_calculator(
        cfg_dict=cfg_dict,
        die_stack=die_stack,
        _3dbx_path=_3dbx_path,
    )
    warpage_time = (
        time.time()
        - start_time
        - init_time
        - overlay_time
        - defect_time
        - mechanical_time
        - esd_time
    )
    print("Warpage yield calculation time: {:.2f} seconds.".format(warpage_time))

    stack_assembly_yield = die_stack.get_die_stack_yield()
    # Print per-interface yields for debugging/analysis
    die_stack.print_die_stack_yield()

    total_time = time.time() - start_time
    print("Assembly yield calculation time: {:.2f} seconds.".format(total_time))

    return stack_assembly_yield, die_stack.die_yield_per_interface_dict
