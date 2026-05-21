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
    active_mechanisms = _active_failure_mechanisms(input_args)
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

    stack_assembly_yield = die_stack.get_die_stack_yield()
    # Print per-interface yields for debugging/analysis
    die_stack.print_die_stack_yield()

    total_time = time.time() - start_time
    print("Assembly yield calculation time: {:.2f} seconds.".format(total_time))

    return stack_assembly_yield, die_stack.die_yield_per_interface_dict
