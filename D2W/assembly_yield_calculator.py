#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#### Author: Zhichao Chen
#### Date: Oct 23, 2025

import os
import time

from overlay_yield_calculator import stack_overlay_yield_calculator
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

    ds_dir = input_args.get("ds_dir", "")
    _3dbx_path = os.path.join(ds_dir, "generated_stack_config.3dbx") if ds_dir else None
    stack_overlay_yield, interface_overlay_yield_dict = stack_overlay_yield_calculator(
        cfg_dict=cfg_dict,
        die_stack=die_stack,
        _3dbx_path=_3dbx_path,
    )
    overlay_time = time.time() - start_time - init_time
    print("Overlay yield calculation time: {:.2f} seconds.".format(overlay_time))

    return stack_overlay_yield, {
        "overlay": interface_overlay_yield_dict,
    }
