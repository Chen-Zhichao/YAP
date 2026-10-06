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
from chiplet_yield import chiplet_yield_records, print_chiplet_yield_table
from wafer_die_stack_initialization import DieStack
from yield_mechanism_policy import (
    active_failure_mechanisms,
    disabled_requested_mechanisms,
)


def _set_skipped_mechanism_yield_to_one(die_stack, mechanism):
    for interface_name in die_stack.die_yield_per_interface_dict:
        die_stack.die_yield_per_interface_dict[interface_name][mechanism] = 1.0


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
    active_mechanisms = active_failure_mechanisms(input_args)
    disabled_requested = disabled_requested_mechanisms(input_args)
    if disabled_requested:
        print(
            "Temporarily disabled yield mechanisms ignored: {}. "
            "Warpage remains enabled only as an overlay-error input.".format(
                ", ".join(sorted(disabled_requested))
            )
        )
    print("Active yield mechanisms: {}.".format(", ".join(sorted(active_mechanisms))))

    ds_dir = input_args.get("design_root_dir", input_args.get("ds_dir", ""))
    _3dbx_path = os.path.join(ds_dir, "generated_stack_config.3dbx") if ds_dir else None
    if "overlay" in active_mechanisms:
        stack_overlay_yield_calculator(
            cfg_dict=cfg_dict,
            die_stack=die_stack,
            _3dbx_path=_3dbx_path,
        )
    else:
        _set_skipped_mechanism_yield_to_one(die_stack, "overlay")
    overlay_time = time.time() - start_time - init_time
    print("Overlay yield calculation time: {:.2f} seconds.".format(overlay_time))

    if "particle" in active_mechanisms:
        stack_defect_yield_calculator(
            cfg_dict=cfg_dict,
            die_stack=die_stack,
        )
    else:
        _set_skipped_mechanism_yield_to_one(die_stack, "particle")
    defect_time = time.time() - start_time - init_time - overlay_time
    print("Particle yield calculation time: {:.2f} seconds.".format(defect_time))

    if "mechanical" in active_mechanisms:
        stack_cu_expansion_yield_calculator(
            cfg_dict=cfg_dict,
            die_stack=die_stack,
        )
    else:
        _set_skipped_mechanism_yield_to_one(die_stack, "mechanical")
    mechanical_time = time.time() - start_time - init_time - overlay_time - defect_time
    print("Mechanical yield calculation time: {:.2f} seconds.".format(mechanical_time))

    if "ESD" in active_mechanisms:
        stack_esd_yield_calculator(
            cfg_dict=cfg_dict,
            die_stack=die_stack,
        )
    else:
        _set_skipped_mechanism_yield_to_one(die_stack, "ESD")
    esd_time = time.time() - start_time - init_time - overlay_time - defect_time - mechanical_time
    print("ESD yield calculation time: {:.2f} seconds.".format(esd_time))

    if "warpage" in active_mechanisms:
        stack_warpage_yield_calculator(
            cfg_dict=cfg_dict,
            die_stack=die_stack,
            _3dbx_path=_3dbx_path,
        )
    else:
        _set_skipped_mechanism_yield_to_one(die_stack, "warpage")
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

    chiplet_records = []
    chiplet_start_time = time.time()
    if "chiplet" in active_mechanisms:
        chiplet_records = chiplet_yield_records(cfg_dict, ds_dir)
        die_stack.set_chiplet_yield_records(chiplet_records)
    chiplet_time = time.time() - chiplet_start_time
    print("Chiplet yield calculation time: {:.2f} seconds.".format(chiplet_time))

    stack_assembly_yield = die_stack.get_die_stack_yield()
    # Print per-interface yields for debugging/analysis
    die_stack.print_die_stack_yield()
    if chiplet_records:
        print_chiplet_yield_table(chiplet_records)
    bonding_stack_yield = float(die_stack.bonding_stack_yield)
    print(f"Calculated bonding stack yield: {bonding_stack_yield:.6f}")
    print(f"Calculated total stack yield: {stack_assembly_yield:.6f}")

    total_time = time.time() - start_time
    print("Assembly yield calculation time: {:.2f} seconds.".format(total_time))

    return stack_assembly_yield, die_stack.die_yield_per_interface_dict
