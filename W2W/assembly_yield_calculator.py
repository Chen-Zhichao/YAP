#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Feb 3, 2026

import os
import time
import numpy as np
from wafer_die_stack_initialization import WaferStack
from overlay_yield_calculator import stack_overlay_yield_calculator
from defect_yield_calculator import stack_defect_yield_calculator
from Cu_expansion_yield_calculator import stack_stress_yield_calculator
from esd_yield_calculator import stack_esd_yield_calculator
from warpage_yield_calculator import stack_warpage_yield_calculator


_FAILURE_MECHANISMS = ("overlay", "particle", "mechanical", "ESD", "warpage")


def _active_failure_mechanisms(input_args):
    raw = input_args.get("mechanism_filter", "all")
    if raw is None:
        return set(_FAILURE_MECHANISMS)
    if isinstance(raw, (set, list, tuple)):
        requested = [str(item).strip() for item in raw]
    else:
        requested = [item.strip() for item in str(raw).split(",")]
    requested = [item for item in requested if item]
    if not requested or any(item.lower() == "all" for item in requested):
        return set(_FAILURE_MECHANISMS)

    canonical = {item.lower(): item for item in _FAILURE_MECHANISMS}
    active = set()
    for item in requested:
        key = item.lower()
        if key not in canonical:
            raise ValueError(
                f"Unknown mechanism_filter '{item}'. "
                f"Valid mechanisms: all, {', '.join(_FAILURE_MECHANISMS)}."
            )
        active.add(canonical[key])
    return active


def _set_skipped_mechanism_yield_to_one(waf_stack, mechanism):
    for interface_name in waf_stack.die_yield_list_per_interface_dict:
        waf_stack.die_yield_list_per_interface_dict[interface_name][mechanism] = np.ones(
            waf_stack.num_dies_per_wafer,
            dtype=float,
        )


def Assembly_Yield_Calculator(
    input_args: dict,
    cfg_dict: dict,
    pad_bitmap_collection_dict: dict,
):
    start_time = time.time()

    # Initialize the wafer stack with dies and pads
    waf_stack = WaferStack(
        cfg_dict=cfg_dict,
        pad_bitmap_collection_dict=pad_bitmap_collection_dict,
        mode=input_args['mode'],
    )
    waf_stack_init_time = time.time() - start_time
    print("Wafer stack initialization time: {:.2f} seconds.".format(waf_stack_init_time))
    active_mechanisms = _active_failure_mechanisms(input_args)
    print("Active yield mechanisms: {}.".format(", ".join(sorted(active_mechanisms))))
    
    valid_pad_mask_dict = {}
    for interface, pad_bitmap_collection in pad_bitmap_collection_dict.items():
        valid_pad_mask_dict[interface] = (
            (pad_bitmap_collection['CRITICAL_PAD_BITMAP'] == 1)
            | (pad_bitmap_collection['REDUNDANT_PAD_BITMAP'] == 1)
            | (pad_bitmap_collection['DUMMY_PAD_BITMAP'] == 1)
            | (pad_bitmap_collection.get(
                'POWER_GROUND_PAD_BITMAP',
                np.zeros_like(pad_bitmap_collection['CRITICAL_PAD_BITMAP'], dtype=bool),
            ) == 1)
        )

    
    # Calculate the overlay yield
    _3dbx_path = os.path.join(input_args['ds_dir'], "generated_stack_config.3dbx")
    if "overlay" in active_mechanisms:
        stack_overlay_yield_calculator(
            cfg_dict                    =   cfg_dict,
            waf_stack                   =   waf_stack,
            _3dbx_path                  =   _3dbx_path,
        )
    else:
        _set_skipped_mechanism_yield_to_one(waf_stack, "overlay")
    overlay_yield_time = time.time() - start_time - waf_stack_init_time
    print("Overlay yield calculation time: {:.2f} seconds.".format(overlay_yield_time))

    # Calculate the defect distribution
    if "particle" in active_mechanisms:
        stack_defect_yield_calculator(
            cfg_dict                    =   cfg_dict,
            waf_stack                   =   waf_stack,
        )
    else:
        _set_skipped_mechanism_yield_to_one(waf_stack, "particle")
    defect_yield_time = time.time() - start_time - waf_stack_init_time - overlay_yield_time
    print("Defect yield calculation time: {:.2f} seconds.".format(defect_yield_time))

    # Calculate the Cu expansion yield
    if "mechanical" in active_mechanisms:
        stack_stress_yield_calculator(
            cfg_dict                    =   cfg_dict,
            waf_stack                   =   waf_stack,
        )
    else:
        _set_skipped_mechanism_yield_to_one(waf_stack, "mechanical")


    Cu_expansion_yield_time = time.time() - start_time - waf_stack_init_time - overlay_yield_time - defect_yield_time
    print("Cu expansion yield calculation time: {:.2f} seconds.".format(Cu_expansion_yield_time))

    # Calculate the ESD yield
    if "ESD" in active_mechanisms:
        stack_esd_yield_calculator(
            cfg_dict                    =   cfg_dict,
            waf_stack                   =   waf_stack,
            pad_bitmap_collection_dict  =   pad_bitmap_collection_dict,
        )
    else:
        _set_skipped_mechanism_yield_to_one(waf_stack, "ESD")
    esd_yield_time = time.time() - start_time - waf_stack_init_time - overlay_yield_time - defect_yield_time - Cu_expansion_yield_time
    print("ESD yield calculation time: {:.2f} seconds.".format(esd_yield_time))

    # Calculate final stack warpage yield
    if "warpage" in active_mechanisms:
        stack_warpage_yield_calculator(
            cfg_dict                    =   cfg_dict,
            waf_stack                   =   waf_stack,
            _3dbx_path                  =   _3dbx_path,
        )
    else:
        _set_skipped_mechanism_yield_to_one(waf_stack, "warpage")
    warpage_yield_time = time.time() - start_time - waf_stack_init_time - overlay_yield_time - defect_yield_time - Cu_expansion_yield_time - esd_yield_time
    print("Warpage yield calculation time: {:.2f} seconds.".format(warpage_yield_time))

    die_stack_yield, die_stack_yield_list = waf_stack.get_die_stack_yield()
    waf_stack.print_interface_yield_table()
    print(f"Calculated die stack yield: {die_stack_yield:.6f}")
       

    del waf_stack

    return die_stack_yield, die_stack_yield_list
