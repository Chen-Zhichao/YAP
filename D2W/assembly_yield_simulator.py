#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Overall yield simulator for hybrid bonding
#### Author: Zhichao Chen
#### Date: Jan 27, 2026

import os
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
import time

from wafer_die_stack_initialization import die_stack_list_initialize
from overlay_yield_simulator import overlay_term_simulator
from defect_yield_simulator import defect_yield_simulator
from overall_yield_simulator import overall_yield_simulator
from utils.util import result_wrapper
from yield_mechanism_policy import (
    FAILURE_MECHANISMS,
    active_failure_mechanisms,
    disabled_requested_mechanisms,
)


def _append_file_suffix(filename, file_suffix):
    if not file_suffix:
        return filename
    stem, ext = os.path.splitext(filename)
    return f"{stem}{file_suffix}{ext}"


def Assembly_Yield_Simulator(
    input_args: dict,
    cfg_skeleton: object,
    cfg_dict: dict,
    pad_bitmap_collection_dict: dict,
    stack_cfg_dict: dict = None,
):
    NUM_DIE_STACKS = cfg_skeleton.NUM_DIE_STACKS
    SIM_BATCH_SIZE = cfg_skeleton.SIM_BATCH_SIZE
    failure_mechanism_list = list(FAILURE_MECHANISMS) + ['overall']
    active_mechanisms = active_failure_mechanisms(input_args)
    disabled_requested = disabled_requested_mechanisms(input_args)
    if disabled_requested:
        print(
            "Temporarily disabled yield mechanisms ignored: {}. "
            "Warpage remains enabled only as an overlay-error input.".format(
                ", ".join(sorted(disabled_requested))
            )
        )
    if not active_mechanisms:
        print("Requested yield mechanisms are disabled; returning unity yield.")
        return (
            1.0,
            [1.0] * NUM_DIE_STACKS,
            {interface_name: 1.0 for interface_name in cfg_dict},
        )
    if active_mechanisms == {'ESD'}:
        esd_batch_size = int(getattr(cfg_skeleton, 'ESD_SIM_BATCH_SIZE', 1000))
        SIM_BATCH_SIZE = min(NUM_DIE_STACKS, max(int(SIM_BATCH_SIZE), esd_batch_size))
    num_sim_epoch = int(np.ceil(NUM_DIE_STACKS / SIM_BATCH_SIZE))
    epoch_yield_list = []
    interface_survival_sum_dict = {interface_name: 0.0 for interface_name in cfg_dict}
    skip_verbose_root_artifacts = bool(input_args.get('skip_verbose_root_artifacts', False))
    save_failure_maps = bool(input_args.get('save_failure_maps', False))
    file_suffix = input_args.get('output_file_tag', '')
    stack_cfg_dict = cfg_dict if stack_cfg_dict is None else stack_cfg_dict
    overlay_boundary_flag = 'overlay' in active_mechanisms

    # Initialize a temporary die stack once to extract the reference pad coordinates.
    temp_die_stack_list, base_pad_coords_dict = die_stack_list_initialize(
        cfg_dict=cfg_dict,
        pad_bitmap_collection_dict=pad_bitmap_collection_dict,
        num_stack_samples=1,
        base_pad_coords_flag=True,
        mode='simulation',
        overlay_boundary_flag=overlay_boundary_flag,
    )
    del temp_die_stack_list

    if input_args['verbose']:
        print("Verbose mode enabled: Tracking failure reasons for each die interface.")
        print("Active failure mechanisms: {}.".format(", ".join(sorted(active_mechanisms))))
        fail_map_per_interface_dict = {}
        fail_vec_per_interface_dict = {}
        for interface_name, cfg in cfg_dict.items():
            fail_map_per_interface_dict[interface_name], fail_vec_per_interface_dict[interface_name] = {}, {}
            for failure_mechanism in failure_mechanism_list:
                if save_failure_maps:
                    fail_map_per_interface_dict[interface_name][failure_mechanism] = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL))
                fail_vec_per_interface_dict[interface_name][failure_mechanism] = np.zeros(NUM_DIE_STACKS)

    simulated_stack_count = 0
    for epoch in range(num_sim_epoch):
        current_batch_size = min(SIM_BATCH_SIZE, NUM_DIE_STACKS - simulated_stack_count)
        start_time = time.perf_counter()
        # Initialize the die list (Extract the base pad coordinates seperately for later use, so that a lot of memory can be saved)
        die_stack_list = die_stack_list_initialize(
            cfg_dict=cfg_dict,
            pad_bitmap_collection_dict=pad_bitmap_collection_dict,
            num_stack_samples=current_batch_size,
            mode='simulation',
            overlay_boundary_flag=overlay_boundary_flag,
        )

        # Generate overlay misalignment component samples for each bonding interface in each stack
        if 'overlay' in active_mechanisms:
            overlay_term_simulator(
                cfg_dict              =       cfg_dict,
                die_stack_list        =       die_stack_list,
                input_args            =       input_args,
                stack_cfg_dict        =       stack_cfg_dict,
                simulation_epoch      =       epoch,
            )

        # Generate void defects
        if 'particle' in active_mechanisms:
            defect_yield_simulator(
                cfg_dict        =       cfg_dict,
                die_stack_list  =       die_stack_list,
            )



        # Calculate the overall yield
        epoch_input_args = dict(input_args)
        epoch_input_args['global_stack_offset'] = simulated_stack_count
        epoch_input_args['simulation_epoch'] = epoch

        yield_list, epoch_interface_yield_dict, epoch_fail_map_per_interface_dict, epoch_fail_vec_per_interface_dict = overall_yield_simulator(
            input_args=epoch_input_args,
            cfg_dict=cfg_dict,
            die_stack_list=die_stack_list,
            pad_bitmap_collection_dict=pad_bitmap_collection_dict,
            base_pad_coords_dict=base_pad_coords_dict,
            stack_cfg_dict=stack_cfg_dict,
        )
        epoch_yield_list.extend(yield_list)
        for interface_name, interface_yield in epoch_interface_yield_dict.items():
            interface_survival_sum_dict[interface_name] += float(interface_yield) * current_batch_size

        # Aggregate the fail maps/vectors
        if input_args['verbose']:
            for interface_name, cfg in cfg_dict.items():
                if save_failure_maps:
                    for failure_mechanism in failure_mechanism_list:
                        fail_map_per_interface_dict[interface_name][failure_mechanism]   \
                            += epoch_fail_map_per_interface_dict[interface_name][failure_mechanism]
                for failure_mechanism in failure_mechanism_list:
                    fail_vec_per_interface_dict[interface_name][failure_mechanism][simulated_stack_count:simulated_stack_count+current_batch_size]  \
                        = epoch_fail_vec_per_interface_dict[interface_name][failure_mechanism]

        simulated_stack_count += current_batch_size
        print(f"Simulation progress: {simulated_stack_count} / {NUM_DIE_STACKS} die stacks simulated. \
              Epoch yield: {np.mean(yield_list):.4f}. Time taken: {time.perf_counter() - start_time:.2f} seconds.", end='\r')

        del die_stack_list

    print("\n>>> Simulation Completed. Wrapping up results...")
    assembly_yield = np.mean(epoch_yield_list)
    per_interface_assembly_yield_dict = {
        interface_name: float(interface_survival_sum / NUM_DIE_STACKS)
        for interface_name, interface_survival_sum in interface_survival_sum_dict.items()
    }

    output_root = os.path.join(next(iter(cfg_dict.values())).OUTPUT_DIR, input_args['ds_name'])
    per_interface_yield_path = os.path.join(
        output_root,
        _append_file_suffix('assembly_yield_per_interface.txt', file_suffix),
    )
    with open(per_interface_yield_path, 'w') as f:
        for interface_name, interface_yield in per_interface_assembly_yield_dict.items():
            f.write(f"{interface_name} {interface_yield:.8f}\n")
    print("Per-interface simulation yield saved to {}.".format(per_interface_yield_path))

    # Remove temporary files if any
    for interface_name, cfg in cfg_dict.items():
        for name in os.listdir(cfg.OUTPUT_DIR + cfg.DESIGN + '/temp'):
            file_path = os.path.join(cfg.OUTPUT_DIR + cfg.DESIGN + '/temp', name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        if input_args['verbose']:
            if save_failure_maps:
                for failure_mechanism in failure_mechanism_list:
                    fail_map_per_interface_dict[interface_name][failure_mechanism]   \
                            /= NUM_DIE_STACKS
            # Report the failure reasons statistics
            print("{} die stack failures due to overlay misalignment.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['overlay']))))
            print("{} die stack failures due to particle defects.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['particle']))))
            print("{} die stack failures due to mechanical issues.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['mechanical']))))
            print("{} die stack failures due to ESD issues.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['ESD']))))
            print("{} die stack failures due to warpage issues.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['warpage']))))
            print("{} die stack failures in total.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['overall']))))
            output_dir = os.path.join(cfg.OUTPUT_DIR, input_args['ds_name'])
            if save_failure_maps and not skip_verbose_root_artifacts:
                # Save fail map dict
                fail_map_path = os.path.join(
                    output_dir,
                    _append_file_suffix('assembly_fail_map_per_interface_dict.npz', file_suffix),
                )
                np.savez(fail_map_path, **fail_map_per_interface_dict)
                print("Failure heat maps saved to {}.".format(fail_map_path))
            elif save_failure_maps:
                print("Skipped root-level verbose NPZ artifacts because identical-interface reuse is active.")

            # Save fail vec dict
            fail_vec_path = os.path.join(
                output_dir,
                _append_file_suffix('assembly_fail_vec_per_interface_dict.npz', file_suffix),
            )
            np.savez(fail_vec_path, **fail_vec_per_interface_dict)
            print("Failure vectors for all die samples saved to {}.".format(fail_vec_path))

            if save_failure_maps:
                # Plot the results for this interface and save the figures
                result_wrapper(
                    mode=input_args['mode'],
                    output_dir=os.path.join(cfg.OUTPUT_DIR, input_args['ds_name']),
                    interface=cfg.INTERFACE,
                    fail_map_dict=fail_map_per_interface_dict[interface_name],
                    file_suffix=file_suffix,
                )
            else:
                print("Skipped simulation failure-map artifacts (use --save-failure-maps to enable).")

    return assembly_yield, epoch_yield_list, per_interface_assembly_yield_dict
