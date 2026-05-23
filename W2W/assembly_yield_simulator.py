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
from warpage_yield_simulator import sample_w2w_warpage_process
from esd_yield_simulator import (
    choose_center_die_index,
    esd_failure_batch_simulator,
)


_CLEAR_LINE = "\033[K"


def _print_progress(message: str):
    print(f"\r{message}{_CLEAR_LINE}", end="", flush=True)


_FAILURE_MECHANISMS = ('overlay', 'particle', 'mechanical', 'ESD', 'warpage')


def _active_failure_mechanisms(input_args):
    raw = input_args.get('mechanism_filter', 'all')
    if raw is None:
        return set(_FAILURE_MECHANISMS)
    if isinstance(raw, (set, list, tuple)):
        requested = [str(item).strip() for item in raw]
    else:
        requested = [item.strip() for item in str(raw).split(',')]
    requested = [item for item in requested if item]
    if not requested or any(item.lower() == 'all' for item in requested):
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


def Assembly_Yield_Simulator(
    input_args: dict,
    cfg_skeleton: object,
    cfg_dict: dict,
    pad_bitmap_collection_dict: dict,
): 
    NUM_WAFER_STACKS = cfg_skeleton.NUM_WAFER_STACKS
    SIM_BATCH_SIZE = cfg_skeleton.SIM_BATCH_SIZE
    num_sim_epoch = NUM_WAFER_STACKS // SIM_BATCH_SIZE

    failure_mechanism_list = list(_FAILURE_MECHANISMS) + ['overall']
    active_mechanisms = _active_failure_mechanisms(input_args)
    run_overlay = 'overlay' in active_mechanisms
    run_particle = 'particle' in active_mechanisms
    run_warpage = 'warpage' in active_mechanisms
    warpage_only_fast_path = active_mechanisms == {'warpage'}
    esd_only_fast_path = active_mechanisms == {'ESD'}
    epoch_yield_list = []

    if warpage_only_fast_path:
        _3dbx_path = os.path.join(input_args['ds_dir'], "generated_stack_config.3dbx")
        failed_stack_count = 0
        completed_stack_count = 0

        for start_idx in range(0, NUM_WAFER_STACKS, SIM_BATCH_SIZE):
            current_batch_size = min(SIM_BATCH_SIZE, NUM_WAFER_STACKS - start_idx)
            start_time = time.time()
            warpage_process_samples = sample_w2w_warpage_process(
                cfg_dict            =       cfg_dict,
                _3dbx_path          =       _3dbx_path,
                num_samples         =       current_batch_size,
                num_dies_per_wafer  =       None,
            )
            stack_pass_vector = np.asarray(
                warpage_process_samples["stack_pass_vector"],
                dtype=bool,
            )
            failed_stack_count += int(np.count_nonzero(~stack_pass_vector))
            completed_stack_count += current_batch_size
            epoch_yield = float(np.mean(stack_pass_vector))
            epoch_yield_list.append(epoch_yield)

            _print_progress(
                f"Warpage-only progress: {completed_stack_count}/{NUM_WAFER_STACKS} "
                f"wafer stacks simulated. Epoch yield: {epoch_yield:.4f}. "
                f"Time taken: {time.time() - start_time:.2f} seconds."
            )

        print(f"\r{_CLEAR_LINE}\nSimulation for all epochs completed.")
        if input_args['verbose']:
            print(f"{failed_stack_count} wafer stack failures due to warpage issues.")
            print(f"{failed_stack_count} wafer stack failures in total.")
        assembly_yield = float(np.mean(epoch_yield_list)) if epoch_yield_list else 0.0
        return assembly_yield, epoch_yield_list

    if input_args['verbose']:
        print("Verbose mode enabled: Tracking failure reasons for each die.")
        print("Active failure mechanisms: {}.".format(", ".join(sorted(active_mechanisms))))
        # Initialize a temporary wafer stack to get die count and initialize fail maps/vectors
        temp_waf_stack_list = wafer_stack_list_initialize(
            cfg_dict                    =       cfg_dict,
            pad_bitmap_collection_dict  =       pad_bitmap_collection_dict,
            num_stack_samples           =       1,
            mode                        =       input_args['mode'],
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

    if esd_only_fast_path:
        start_time = time.time()
        template_waf_stack_list = wafer_stack_list_initialize(
            cfg_dict                    =       cfg_dict,
            pad_bitmap_collection_dict  =       pad_bitmap_collection_dict,
            num_stack_samples           =       1,
            mode                        =       input_args['mode'],
        )
        template_waf_stack = template_waf_stack_list[0]
        num_dies_per_wafer = template_waf_stack.num_dies_per_wafer

        if not input_args['verbose']:
            fail_map_per_interface_dict = None
            fail_vec_per_interface_dict = None

        interface_static = {}
        for interface_name, waf_interface in template_waf_stack.interfaces.interface_dict.items():
            cfg = cfg_dict[interface_name]
            pad_bitmap_collection = pad_bitmap_collection_dict[interface_name]
            empty_bitmap = np.zeros_like(
                pad_bitmap_collection['CRITICAL_PAD_BITMAP'],
                dtype=bool,
            )
            valid_pad_mask = (
                (pad_bitmap_collection['CRITICAL_PAD_BITMAP'] == 1)
                | (pad_bitmap_collection['REDUNDANT_PAD_BITMAP'] == 1)
                | (pad_bitmap_collection['DUMMY_PAD_BITMAP'] == 1)
                | (pad_bitmap_collection.get('POWER_GROUND_PAD_BITMAP', empty_bitmap) == 1)
            )
            valid_pad_mask_flat = valid_pad_mask.flatten()
            valid_linear_idx = np.flatnonzero(valid_pad_mask_flat)
            interface_static[interface_name] = {
                "cfg": cfg,
                "waf_interface": waf_interface,
                "valid_pad_mask_flat": valid_pad_mask_flat,
                "valid_linear_idx": valid_linear_idx,
                "valid_dummy_pad_bitmap": pad_bitmap_collection['DUMMY_PAD_BITMAP'].flatten()[valid_pad_mask_flat],
                "die_esd_critical_pad_bitmap": pad_bitmap_collection["ESD_CRITICAL_PAD_BITMAP"],
                "selected_esd_die_ind": choose_center_die_index(
                    waf_interface.die_list,
                    tolerance_um=(
                        None
                        if getattr(cfg, "ESD_CENTER_TOL_UM", None) is None
                        else float(getattr(cfg, "ESD_CENTER_TOL_UM"))
                    ),
                ),
            }

        die_stack_survival = np.ones((NUM_WAFER_STACKS, num_dies_per_wafer), dtype=bool)
        for interface_name, static in interface_static.items():
            cfg = static["cfg"]
            waf_interface = static["waf_interface"]
            die_ind = static["selected_esd_die_ind"]
            if die_ind is None:
                continue

            die = waf_interface.die_list[int(die_ind)]
            valid_pad_mask_flat = static["valid_pad_mask_flat"]
            valid_linear_idx = static["valid_linear_idx"]
            valid_die_pad_coords = (
                waf_interface.base_pad_coords + die.die_center
            )[valid_pad_mask_flat]
            first_contact_pad_idx, survive_bool = esd_failure_batch_simulator(
                cfg=cfg,
                pad_coords_um=valid_die_pad_coords,
                pad_size_um=cfg.PAD_TOP_R_um * 2,
                top_die_w_um=die.DIE_W_um,
                top_die_h_um=die.DIE_L_um,
                wafer_radius_um=cfg.WAF_R_um,
                num_samples=NUM_WAFER_STACKS,
                dummy_pad_bitmap=static["valid_dummy_pad_bitmap"],
            )
            failed_by_discharge = ~np.asarray(survive_bool, dtype=bool)
            for stack_idx in np.flatnonzero(failed_by_discharge):
                if (not input_args['verbose']) and (not die_stack_survival[int(stack_idx), int(die_ind)]):
                    continue
                full_linear_idx = int(valid_linear_idx[int(first_contact_pad_idx[int(stack_idx)])])
                r_idx = full_linear_idx // int(cfg.PAD_ARR_COL)
                c_idx = full_linear_idx % int(cfg.PAD_ARR_COL)
                if input_args['verbose']:
                    fail_map_per_interface_dict[interface_name]['ESD'][r_idx, c_idx] += 1
                    fail_map_per_interface_dict[interface_name]['overall'][r_idx, c_idx] += 1
                if static["die_esd_critical_pad_bitmap"][r_idx, c_idx] == 1:
                    die_stack_survival[int(stack_idx), int(die_ind)] = False
                    if input_args['verbose']:
                        fail_vec_per_interface_dict[interface_name]['ESD'][int(stack_idx), int(die_ind)] = 1
                        fail_vec_per_interface_dict[interface_name]['overall'][int(stack_idx), int(die_ind)] = 1

            _print_progress(
                f"ESD-only interface {interface_name}: "
                f"{int(np.count_nonzero(failed_by_discharge))}/{NUM_WAFER_STACKS} "
                f"discharges failed. Time taken: {time.time() - start_time:.2f} seconds."
            )

        epoch_yield_list = np.mean(die_stack_survival, axis=1).astype(float).tolist()

        print(f"\r{_CLEAR_LINE}\nSimulation for all epochs completed.")
        assembly_yield = float(np.mean(epoch_yield_list)) if epoch_yield_list else 0.0
        for interface_name, cfg in cfg_dict.items():
            if input_args['verbose']:
                for failure_mechanism in failure_mechanism_list:
                    fail_map_per_interface_dict[interface_name][failure_mechanism] /= (
                        NUM_WAFER_STACKS * num_dies_per_wafer
                    )
                print("{} die stack failures due to overlay misalignment.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['overlay']))))
                print("{} die stack failures due to particle defects.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['particle']))))
                print("{} die stack failures due to mechanical issues.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['mechanical']))))
                print("{} die stack failures due to ESD issues.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['ESD']))))
                print("{} die stack failures due to warpage issues.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['warpage']))))
                print("{} die stack failures in total.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['overall']))))
                np.savez(cfg.OUTPUT_DIR + cfg.DESIGN + '/assembly_fail_map_per_interface_dict.npz', **fail_map_per_interface_dict)
                print("Failure heat maps saved to {}.".format(cfg.OUTPUT_DIR + cfg.DESIGN + '/assembly_fail_map_per_interface_dict.npz'))
                np.savez(cfg.OUTPUT_DIR + cfg.DESIGN + '/assembly_fail_vec_per_interface_dict.npz', **fail_vec_per_interface_dict)
                print("Failure vectors for all die samples saved to {}.".format(cfg.OUTPUT_DIR + cfg.DESIGN + '/assembly_fail_vec_per_interface_dict.npz'))

            result_wrapper(
                mode = input_args['mode'],
                cfg = cfg,
                fail_map_per_interface_dict = fail_map_per_interface_dict if input_args['verbose'] else None,
            )
        del template_waf_stack_list
        return assembly_yield, epoch_yield_list
    
    # Iterate over simulation epochs
    for epoch in range(num_sim_epoch):
        # Record the time for each epoch
        start_time = time.time()
        # Initialize the wafer stack
        waf_stack_list = wafer_stack_list_initialize(
            cfg_dict                    =       cfg_dict,
            pad_bitmap_collection_dict  =       pad_bitmap_collection_dict,
            num_stack_samples           =       SIM_BATCH_SIZE,
            mode                        =       input_args['mode'],
        )
        num_dies_per_wafer = waf_stack_list[0].num_dies_per_wafer

        _3dbx_path = os.path.join(input_args['ds_dir'], "generated_stack_config.3dbx")
        if run_overlay or run_warpage:
            warpage_process_samples = sample_w2w_warpage_process(
                cfg_dict            =       cfg_dict,
                _3dbx_path          =       _3dbx_path,
                num_samples         =       SIM_BATCH_SIZE,
                num_dies_per_wafer  =       num_dies_per_wafer,
            )
        else:
            warpage_process_samples = {
                "stack_pass_vector": np.ones(SIM_BATCH_SIZE, dtype=bool),
                "interface_bow_difference_samples": {
                    interface_name: np.zeros(SIM_BATCH_SIZE, dtype=float)
                    for interface_name in cfg_dict
                },
            }

        # Generate overlay misalignment component samples for each bonding interface in each stack
        if run_overlay:
            overlay_term_simulator(
                cfg_dict                        =       cfg_dict,
                waf_stack_list                  =       waf_stack_list,
                _3dbx_path                      =       _3dbx_path,
                bow_difference_samples_by_interface =   warpage_process_samples["interface_bow_difference_samples"],
            )
    
        # Generate void defects for each bonding interface
        if run_particle:
            defect_yield_simulator(
                cfg_dict            =       cfg_dict,
                waf_stack_list      =       waf_stack_list,
            )

        warpage_fail_vector = ~warpage_process_samples["stack_pass_vector"]
        if run_warpage:
            for stack_idx, warpage_failed in enumerate(warpage_fail_vector):
                if warpage_failed:
                    waf_stack_list[stack_idx].die_stack_survival[:] = False
        
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
        if run_warpage and input_args['verbose']:
            for interface_name in cfg_dict:
                for stack_idx, warpage_failed in enumerate(warpage_fail_vector):
                    if warpage_failed:
                        epoch_fail_vec_per_interface_dict[interface_name]['warpage'][stack_idx, :] = 1
                        epoch_fail_vec_per_interface_dict[interface_name]['overall'][stack_idx, :] = 1
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
        _print_progress(
            f"Simulation progress: {(epoch+1)*SIM_BATCH_SIZE}/{NUM_WAFER_STACKS} wafer stacks simulated. "
            f"Epoch yield: {np.mean(yield_list):.4f}. Time taken: {time.time() - start_time:.2f} seconds."
        )

        del waf_stack_list

    print(f"\r{_CLEAR_LINE}\nSimulation for all epochs completed.")
    assembly_yield = np.mean(epoch_yield_list)
    # Remove temporary files if any
    temp_dir = cfg.OUTPUT_DIR + cfg.DESIGN + '/temp'
    if os.path.isdir(temp_dir):
        for name in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, name)
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
            print("{} die stack failures due to warpage issues.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['warpage']))))
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
