#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#### Author: Zhichao Chen
#### Date: Feb 9, 2026

import numpy as np
import time
from overlay_yield_calculator import stack_overlay_yield_calculator
from defect_yield_calculator import stack_defect_yield_calculator
from Cu_expansion_yield_calculator import stack_stress_yield_calculator
from esd_yield_calculator import stack_esd_yield_calculator
from wafer_die_stack_initialization import DieStack




def Assembly_Yield_Calculator(
    input_args: dict,
    cfg_dict: dict,
    pad_bitmap_collection_dict: dict,
):  
    '''
    This function calculates the die-stack-level yield
    '''
    start_time = time.time()

    # Initialize the die stack
    die_stack = DieStack(
        cfg_dict                    =   cfg_dict,
        pad_bitmap_collection_dict  =   pad_bitmap_collection_dict,
        mode                        =   input_args['mode'],
        base_pad_coords_flag        =   True,
    )
    
    valid_pad_mask_dict = {}
    for interface_name, pad_bitmap_collection in pad_bitmap_collection_dict.items():
        valid_pad_mask_dict[interface_name] = (pad_bitmap_collection['CRITICAL_PAD_BITMAP'] == 1) | (pad_bitmap_collection['REDUNDANT_PAD_BITMAP'] == 1) | (pad_bitmap_collection['DUMMY_PAD_BITMAP'] == 1)

    
    # Calculate the overlay yield
    stack_overlay_yield_calculator(
        cfg_dict            =       cfg_dict,
        die_stack           =       die_stack,
    )


    # Calculate the defect yield
    stack_defect_yield_calculator(
        cfg_dict            =       cfg_dict,
        die_stack           =       die_stack,
    )

    # Calculate the Cu expansion yield
    stack_stress_yield_calculator(
        cfg_dict                    =   cfg_dict,
        die_stack                   =   die_stack,
        pad_bitmap_collection_dict  =   pad_bitmap_collection_dict,
        valid_pad_mask_dict         =   valid_pad_mask_dict,
    )

    stack_esd_yield_calculator(
        cfg_dict                    =   cfg_dict,
        die_stack                   =   die_stack,
        pad_bitmap_collection_dict  =   pad_bitmap_collection_dict,
    )

    die_stack_yield = die_stack.get_die_stack_yield()
    print(f"Calculated die stack yield: {die_stack_yield:.6f}")

    del die_stack