#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Overall yield simulator for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np
import time
import os

from wafer_die_stack_initialization import wafer_stack_list_initialize
from overlay_yield_simulator import overlay_term_simulator
from defect_yield_simulator import defect_yield_simulator
from overall_yield_simulator import overall_yield_simulator
from utils.util import result_wrapper


def Assembly_Yield_Simulator(
    input_args: dict,
    cfg_skeleton: object,
    cfg_dict: dict,
    pad_bitmap_collection_dict: dict,
): 
    NUM_WAFER_STACKS = cfg_skeleton.NUM_WAFER_STACKS
    SIM_BATCH_SIZE = cfg_skeleton.SIM_BATCH_SIZE
    num_sim_epoch = NUM_WAFER_STACKS // SIM_BATCH_SIZE

    failure_mechanism_list = ['overlay', 'particle', 'mechanical', 'ESD', 'overall']
    epoch_yield_list = []

    if input_args['verbose']:
        print("Verbose mode enabled: Tracking failure reasons for each die.")
        # Initialize a temporary wafer stack to get die count and initialize fail maps/vectors
        temp_waf_stack_list = wafer_stack_list_initialize(
            cfg_dict                    =       cfg_dict,
            pad_bitmap_collection_dict  =       pad_bitmap_collection_dict,
            num_stack_samples           =       1,
        )
        fail_map_per_interface_dict = {}
        fail_vec_per_interface_dict = {}
        num_dies_per_wafer = temp_waf_stack_list[0].num_dies_per_wafer
        for interface_name, waf_interface in temp_waf_stack_list[0].interfaces.interface_dict.items():
            cfg = cfg_dict[interface_name]
            fail_map_per_interface_dict[interface_name], fail_vec_per_interface_dict[interface_name] = {}, {}
            for failure_mechanism in failure_mechanism_list:
                fail_map_per_interface_dict[interface_name][failure_mechanism] = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL))
                fail_vec_per_interface_dict[interface_name][failure_mechanism] = np.zeros((NUM_WAFER_STACKS, num_dies_per_wafer))
        del temp_waf_stack_list
    
    # Iterate over simulation epochs
    for epoch in range(num_sim_epoch):
        # Record the time for each epoch
        start_time = time.time()
        # Initialize the wafer stack
        waf_stack_list = wafer_stack_list_initialize(
            cfg_dict                    =       cfg_dict,
            pad_bitmap_collection_dict  =       pad_bitmap_collection_dict,
            num_stack_samples           =       SIM_BATCH_SIZE,
        )
        num_dies_per_wafer = waf_stack_list[0].num_dies_per_wafer

        # Generate overlay misalignment component samples for each bonding interface in each stack
        overlay_term_simulator(
            cfg_dict                        =       cfg_dict,
            waf_stack_list                  =       waf_stack_list,
        )
    
        # Generate void defects for each bonding interface
        defect_yield_simulator(
            cfg_dict            =       cfg_dict,
            waf_stack_list      =       waf_stack_list,
        )
        
        # Calculate the overall yield
        yield_list, \
        epoch_fail_map_per_interface_dict, \
        epoch_fail_vec_per_interface_dict = overall_yield_simulator(
            input_args                      =       input_args,
            cfg_dict                        =       cfg_dict,
            epoch                           =       epoch,
            waf_stack_list                  =       waf_stack_list,
            num_dies_per_wafer              =       num_dies_per_wafer,
            pad_bitmap_collection_dict      =       pad_bitmap_collection_dict,
        )
        epoch_yield_list.append(yield_list)

        # Update the overall fail maps/vectors
        for interface_name, cfg in cfg_dict.items()   :
            if input_args['verbose']:
                for failure_mechanism in failure_mechanism_list:
                    fail_map_per_interface_dict[interface_name][failure_mechanism]   \
                          += epoch_fail_map_per_interface_dict[interface_name][failure_mechanism]

                for failure_mechanism in failure_mechanism_list:
                    fail_vec_per_interface_dict[interface_name][failure_mechanism][epoch*SIM_BATCH_SIZE:(epoch+1)*SIM_BATCH_SIZE, :] \
                        = epoch_fail_vec_per_interface_dict[interface_name][failure_mechanism]
        print(f"Simulation progress: {(epoch+1)*SIM_BATCH_SIZE}/{NUM_WAFER_STACKS} wafer stacks simulated. \
              Epoch yield: {np.mean(yield_list):.4f}. Time taken: {time.time() - start_time:.2f} seconds.", end='\r')

        del waf_stack_list

    print("\nSimulation for all epochs completed.")
    assembly_yield = np.mean(epoch_yield_list)
    # Remove temporary files if any
    for name in os.listdir(cfg.OUTPUT_DIR + cfg.DESIGN + '/temp'):
        file_path = os.path.join(cfg.OUTPUT_DIR + cfg.DESIGN + '/temp', name)
        if os.path.isfile(file_path):
            os.remove(file_path)
    for interface_name, cfg in cfg_dict.items():
        if input_args['verbose']:
            for failure_mechanism in failure_mechanism_list:
                fail_map_per_interface_dict[interface_name][failure_mechanism]   \
                      /= (num_sim_epoch * SIM_BATCH_SIZE * num_dies_per_wafer)
            # Report the failure reasons statistics
            print("{} die stack failures due to overlay misalignment.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['overlay']))))
            print("{} die stack failures due to particle defects.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['particle']))))
            print("{} die stack failures due to mechanical issues.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['mechanical']))))
            print("{} die stack failures due to ESD issues.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['ESD']))))
            print("{} die stack failures in total.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['overall']))))
            # Save fail map dict
            np.savez(cfg.OUTPUT_DIR + cfg.DESIGN + '/assembly_fail_map_per_interface_dict.npz', **fail_map_per_interface_dict)
            print("Failure heat maps saved to {}.".format(cfg.OUTPUT_DIR + cfg.DESIGN + '/assembly_fail_map_per_interface_dict.npz'))
            # Save fail vec dict
            np.savez(cfg.OUTPUT_DIR + cfg.DESIGN + '/assembly_fail_vec_per_interface_dict.npz', **fail_vec_per_interface_dict)
            print("Failure vectors for all die samples saved to {}.".format(cfg.OUTPUT_DIR + cfg.DESIGN + '/assembly_fail_vec_per_interface_dict.npz'))

        # Plot the results for this interface and save the figures
        result_wrapper(
            mode = input_args['mode'],
            cfg = cfg,
            fail_map_per_interface_dict = fail_map_per_interface_dict if input_args['verbose'] else None,
        )
    return assembly_yield, epoch_yield_list
