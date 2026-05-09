#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import numpy as np
from utils.util import *
import time
import argparse
from assembly_yield_simulator import Assembly_Yield_Simulator
from utils.generate_criticality import DEFAULT_PROFILE, resolve_criticality_path
from utils.interface_reuse import (
    copy_representative_bitmap_outputs,
    copy_representative_simulation_outputs,
    format_group_summary,
    group_raw_identical_interfaces,
    has_reused_interfaces,
    write_group_metadata,
    write_per_interface_yield_file,
)


def _sanitize_output_tag(value: str) -> str:
    safe = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def _build_output_file_tag(config_path: str, criticality_profile: str) -> str:
    config_stem = os.path.splitext(os.path.basename(config_path))[0]
    tag = _sanitize_output_tag(f"{config_stem}__{criticality_profile}")
    return f"__{tag}" if tag else ""


def parse_args():
    p = argparse.ArgumentParser(description="Simulate assembly yield for W2W hybrid bonding")
    p.add_argument("--config", "-c", required=True, help="Path to skeleton config YAML file")
    p.add_argument("--mode", "-m", required=True, help="Mode to load from config (default: w2w_modeling)")
    p.add_argument("--ds_name", "-d", required=True, help="Name of design (used for output directory naming)")
    p.add_argument("--ds_dir", required=True, help="Path to design directory")
    p.add_argument("--plot", "-plot", default=False, action="store_true", help="Enable optional debug plots")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output during simulation")
    p.add_argument(
        "--save-failure-maps",
        action="store_true",
        help=(
            "Save verbose simulation failure-map artifacts (PNG heatmaps and NPZ files). "
            "By default these large files are skipped."
        ),
    )
    p.add_argument("--debug", action="store_true", help="Enable debug output when loading config")
    p.add_argument(
        "--criticality-profile",
        default=DEFAULT_PROFILE,
        choices=("default", "esd_strict"),
        help=(
            "Which criticality-file profile to use: "
            "'default' tolerates R-1 ESD + mechanical failures for replicated nets; "
            "'esd_strict' tolerates R-1 mechanical failures but 0 ESD failures."
        ),
    )
    return p.parse_args()


def write_simulation_summary(
    output_root: str,
    input_args: dict,
    cfg_skeleton: object,
    cfg_dict: dict,
    stack_assembly_yield: float,
    per_interface_yield_dict: dict,
    cfg_loading_time_s: float,
    bitmap_generation_time_s: float,
    simulation_time_s: float,
    total_time_s: float,
    grouped_interfaces: dict,
):
    file_suffix = input_args.get("output_file_tag", "")
    summary_filename = f"assembly_yield_summary{file_suffix}.txt"
    summary_path = os.path.join(output_root, summary_filename)
    first_cfg = next(iter(cfg_dict.values()))
    geometry_keys = [
        "DESIGN",
        "INTERFACE",
        "PITCH_r_um",
        "PITCH_c_um",
        "DIE_W_um",
        "DIE_L_um",
        "PAD_ARR_ROW",
        "PAD_ARR_COL",
        "PAD_BOT_R_um",
        "PAD_TOP_R_um",
        "PAD_ARRANGE_PATTERN",
        "VOID_SHAPE",
        "ITF_TOP_THICK_um",
        "ITF_BOT_THICK_um",
    ]
    common_skip_keys = set(geometry_keys + ["OUTPUT_DIR", "INPUT_DIR"])

    with open(summary_path, "w") as f:
        f.write("Assembly Yield Simulation Summary\n")
        f.write("=================================\n\n")

        f.write("Run Settings\n")
        f.write("------------\n")
        f.write(f"config: {input_args['config']}\n")
        f.write(f"mode: {input_args['mode']}\n")
        f.write(f"ds_name: {input_args['ds_name']}\n")
        f.write(f"ds_dir: {input_args['ds_dir']}\n")
        f.write(f"criticality_profile: {input_args['criticality_profile']}\n")
        f.write(f"verbose: {input_args['verbose']}\n")
        f.write(f"plot: {input_args['plot']}\n")
        f.write(f"save_failure_maps: {input_args['save_failure_maps']}\n")
        f.write("random_source: system_entropy\n")
        f.write(f"NUM_DIE_STACKS: {cfg_skeleton.NUM_DIE_STACKS}\n")
        f.write(f"SIM_BATCH_SIZE: {cfg_skeleton.SIM_BATCH_SIZE}\n")
        f.write(f"num_interfaces: {len(cfg_dict)}\n")
        f.write(f"reused_interface_groups_active: {has_reused_interfaces(grouped_interfaces)}\n\n")

        f.write("Runtimes (seconds)\n")
        f.write("------------------\n")
        f.write(f"config_loading: {cfg_loading_time_s:.2f}\n")
        f.write(f"bitmap_generation: {bitmap_generation_time_s:.2f}\n")
        f.write(f"simulation: {simulation_time_s:.2f}\n")
        f.write(f"total: {total_time_s:.2f}\n\n")

        f.write("Yield Results\n")
        f.write("-------------\n")
        f.write(f"stack_assembly_yield: {stack_assembly_yield:.8f}\n")
        for interface_name in sorted(per_interface_yield_dict):
            f.write(f"{interface_name}: {per_interface_yield_dict[interface_name]:.8f}\n")
        f.write("\n")

        f.write("Common Simulation Parameters\n")
        f.write("----------------------------\n")
        for key in first_cfg.keys():
            if key in common_skip_keys:
                continue
            f.write(f"{key}: {first_cfg[key]}\n")
        f.write("\n")

        f.write("Resolved Interface Geometry\n")
        f.write("---------------------------\n")
        for interface_name in sorted(cfg_dict):
            cfg = cfg_dict[interface_name]
            f.write(f"[{interface_name}]\n")
            for key in geometry_keys:
                if key in cfg:
                    f.write(f"{key}: {cfg[key]}\n")
            f.write("\n")

        if has_reused_interfaces(grouped_interfaces):
            f.write("Collapsed Interface Groups\n")
            f.write("--------------------------\n")
            f.write(format_group_summary(grouped_interfaces))
            f.write("\n")

    return summary_path

def main():
    args = parse_args()
    args.output_file_tag = _build_output_file_tag(args.config, args.criticality_profile)
    cfg_dict = None
    config_stem = os.path.splitext(os.path.basename(args.config))[0]
    cfg_output_dir = args.config.rsplit('/', 1)[0]

    # Extract the design input files directory if provided
    input_ds_dir = args.ds_dir
    # Determine .3dbv path (chiplet definitions)
    _3dbv_path = input_ds_dir + "/generated_chiplet_definitions.3dbv"
    # Determine .3dbx path (stack config)
    _3dbx_path = input_ds_dir + "/generated_stack_config.3dbx"
    # Read the config skeleton and update with design parameters
    cfg_skeleton = OmegaConf.load(args.config)[args.mode]

    print(">>>>>> Starting D2W yield simulation for design: {}".format(args.ds_name))
    start_time = time.perf_counter()
    try:
        # Load config and update with design and ADK parameters (from .3dbv and .bmap)
        if os.path.exists(_3dbv_path) and os.path.exists(_3dbx_path):
            cfg_dict = get_config_dict(cfg_folder=cfg_output_dir,
                                        cfg_skeleton=cfg_skeleton,
                                        ds_name=args.ds_name,
                                        input_ds_dir=input_ds_dir,
                                        _3dbv_path=_3dbv_path,
                                        _3dbx_path=_3dbx_path,
                                        mode=args.mode,
                                        debug=args.debug,
                                        file_suffix=args.output_file_tag)
        else:
            print("Using legacy single-interface input mode (no generated_stack_config.3dbx / generated_chiplet_definitions.3dbv).")
            cfg_dict = get_single_interface_config_dict(
                cfg_folder=cfg_output_dir,
                cfg_skeleton=cfg_skeleton,
                ds_name=args.ds_name,
                input_ds_dir=input_ds_dir,
                mode=args.mode,
                debug=args.debug,
                file_suffix=args.output_file_tag,
            )
        cfg_loading_time = time.perf_counter() - start_time
        print(f"Config loading and processing finished in {cfg_loading_time:.2f} seconds.")

        # Plotting flag
        for cfg in cfg_dict.values():
            cfg.plot_flag = args.plot

        # Create output directory if it doesn't exist
        for cfg in cfg_dict.values():
            output_path = os.path.join(cfg.OUTPUT_DIR, args.ds_name, cfg.INTERFACE)
            os.makedirs(output_path, exist_ok=True)

        # Verbose flag
        for cfg in cfg_dict.values():
            cfg.verbose = args.verbose

        # Run assembly yield simulation for each interface
        assembly_yield_dict = {}
        bmap_path_dict = {}
        criticality_path_dict = {}
        pad_bitmap_collection_dict = {}
        for interface, cfg in cfg_dict.items():
            bmap_path_dict[interface] = resolve_design_file(input_ds_dir, f"{cfg.INTERFACE}.bmap")
            criticality_path_dict[interface] = str(
                resolve_criticality_path(
                    input_dir=input_ds_dir,
                    interface_name=cfg.INTERFACE,
                    profile=args.criticality_profile,
                )
            )
            assert os.path.exists(criticality_path_dict[interface]), (
                f"Criticality file not found for profile '{args.criticality_profile}': "
                f"{criticality_path_dict[interface]}"
            )

        grouped_interfaces = group_raw_identical_interfaces(
            cfg_dict=cfg_dict,
            bmap_path_dict=bmap_path_dict,
            criticality_path_dict=criticality_path_dict,
        )
        output_root = os.path.join(next(iter(cfg_dict.values())).OUTPUT_DIR, args.ds_name)
        if has_reused_interfaces(grouped_interfaces):
            print("Reusing identical interfaces for bitmap generation and simulation:")
            print(format_group_summary(grouped_interfaces))
            metadata_path = write_group_metadata(
                output_root,
                grouped_interfaces,
                filename=f"collapsed_interface_groups{args.output_file_tag}.txt",
            )
            print(f"Collapsed interface groups saved to {metadata_path}.")
            if os.path.exists(_3dbx_path):
                print(
                    "Stack graph is available; full-interface simulation will be used "
                    "so substack bow difference and final stack warpage are evaluated once per stack."
                )

        # Step 1: convert .bmap -> pad bitmap collection
        if has_reused_interfaces(grouped_interfaces):
            for representative, members in grouped_interfaces.items():
                rep_cfg = cfg_dict[representative]
                rep_bitmap_collection = convert_3dblox_to_pad_bitmap(
                    cfg=rep_cfg,
                    _bmap_path=bmap_path_dict[representative],
                    criticality_path=criticality_path_dict[representative],
                    pad_arrange_pattern=rep_cfg.PAD_ARRANGE_PATTERN,
                    input_args=vars(args),
                )
                for interface_name in members:
                    pad_bitmap_collection_dict[interface_name] = rep_bitmap_collection
                for duplicate in members[1:]:
                    copy_representative_bitmap_outputs(
                        output_root=output_root,
                        representative=representative,
                        duplicate=duplicate,
                    )
        else:
            for interface, cfg in cfg_dict.items():
                pad_bitmap_collection_dict[interface] = convert_3dblox_to_pad_bitmap(
                    cfg=cfg,
                    _bmap_path=bmap_path_dict[interface],
                    criticality_path=criticality_path_dict[interface],
                    pad_arrange_pattern=cfg.PAD_ARRANGE_PATTERN,
                    input_args=vars(args),
                )
        convert_time = time.perf_counter() - start_time - cfg_loading_time
        print("Pad bitmap collection generation finished in {:.2f} seconds.".format(convert_time))

        # Step 2: run assembly yield simulator
        print("Running assembly yield simulator over {} die stacks...".format(cfg_skeleton.NUM_DIE_STACKS))
        simulation_start_time = time.time()
        use_representative_simulation = (
            has_reused_interfaces(grouped_interfaces)
            and not os.path.exists(_3dbx_path)
        )
        if use_representative_simulation:
            per_interface_yield_dict = {}
            stack_assembly_yield = 1.0
            for representative, members in grouped_interfaces.items():
                print(
                    f">>> Simulating representative interface {representative} (x{len(members)})"
                )
                rep_args = dict(vars(args))
                rep_args["skip_verbose_root_artifacts"] = True
                _, _, rep_yield_dict = Assembly_Yield_Simulator(
                    input_args=rep_args,
                    cfg_skeleton=cfg_skeleton,
                    cfg_dict={representative: cfg_dict[representative]},
                    pad_bitmap_collection_dict={representative: pad_bitmap_collection_dict[representative]},
                    stack_cfg_dict=cfg_dict,
                )
                representative_yield = rep_yield_dict[representative]
                stack_assembly_yield *= representative_yield ** len(members)
                for interface_name in members:
                    per_interface_yield_dict[interface_name] = representative_yield
                if args.save_failure_maps:
                    for duplicate in members[1:]:
                        copy_representative_simulation_outputs(
                            output_root=output_root,
                            representative=representative,
                            duplicate=duplicate,
                            file_suffix=args.output_file_tag,
                        )

            yield_path = write_per_interface_yield_file(
                output_root,
                per_interface_yield_dict,
                file_suffix=args.output_file_tag,
            )
            print(f"Per-interface simulation yield saved to {yield_path}.")
            if args.verbose:
                note_path = os.path.join(
                    output_root,
                    f"collapsed_interface_simulation_note{args.output_file_tag}.txt",
                )
                with open(note_path, "w") as f:
                    f.write(
                        "Identical-interface reuse was active.\n"
                        + "Per-interface yield was expanded from representative interfaces.\n"
                        + (
                            "Average failure-map PNGs were also expanded from representative interfaces.\n"
                            if args.save_failure_maps else
                            "Failure-map PNG expansion was skipped because --save-failure-maps was not enabled.\n"
                        )
                        + "Root-level per-sample failure-vector NPZ artifacts were skipped because they cannot be "
                        + "expanded to duplicates without inventing sample-wise correlations.\n"
                    )
                print(f"Collapsed simulation note saved to {note_path}.")
        else:
            stack_assembly_yield, _, per_interface_yield_dict = Assembly_Yield_Simulator(
                input_args=vars(args),
                cfg_skeleton=cfg_skeleton,
                cfg_dict=cfg_dict,
                pad_bitmap_collection_dict=pad_bitmap_collection_dict,
            )

        simulation_elapsed = time.time() - simulation_start_time
        total_runtime = time.perf_counter() - start_time

        summary_path = write_simulation_summary(
            output_root=output_root,
            input_args=vars(args),
            cfg_skeleton=cfg_skeleton,
            cfg_dict=cfg_dict,
            stack_assembly_yield=stack_assembly_yield,
            per_interface_yield_dict=per_interface_yield_dict,
            cfg_loading_time_s=cfg_loading_time,
            bitmap_generation_time_s=convert_time,
            simulation_time_s=simulation_elapsed,
            total_time_s=total_runtime,
            grouped_interfaces=grouped_interfaces,
        )

        print(f">>> Yield simulation results for {args.ds_name}: {stack_assembly_yield}")
        print("Per-interface simulation yield:")
        for interface_name, interface_yield in per_interface_yield_dict.items():
            print(f">>  {interface_name}: {interface_yield:.6f}")
        print(f"Assembly yield summary saved to {summary_path}.")
        print("Simulation finished in {:.2f} seconds.".format(simulation_elapsed))
        # Total running time
        print(f"Total D2W assembly yield simulation finished in {total_runtime:.2f} seconds.")
    finally:
        if cfg_dict:
            removed_temp_paths = cleanup_runtime_temp_files(cfg_dict, vars(args))
            if removed_temp_paths:
                print(f"Cleaned {len(removed_temp_paths)} runtime temp files.")
        # Generated interface configs are saved under the design's config folder.

    print_run_separator(f"D2W simulation finished for {args.ds_name}")
if __name__ == "__main__":
    main()
