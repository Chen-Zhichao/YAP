#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monte Carlo simulator for validating D2W substack warpage yield.

This module samples each chiplet layer's initial bow, recomputes the final
warpage of every substack with the same physical stack formula used by the
calculator, and marks a full stack sample as failed when any substack exceeds
the configured absolute warpage threshold.
"""

import os

import numpy as np

try:
    from utils.util import stack_graph_from_3dbx
    from warpage_yield_calculator import (
        _build_layer_df,
        _delta_t_from_cfg,
        _prebond_delta_t_from_cfg,
        _anneal_mode_from_cfg,
        _stack_warpage_threshold_um,
        _top_die_half_length_m,
        compute_total_stack_warpage,
        compute_warpage_yield,
    )
except ModuleNotFoundError:
    from D2W.utils.util import stack_graph_from_3dbx
    from D2W.warpage_yield_calculator import (
        _build_layer_df,
        _delta_t_from_cfg,
        _prebond_delta_t_from_cfg,
        _anneal_mode_from_cfg,
        _stack_warpage_threshold_um,
        _top_die_half_length_m,
        compute_total_stack_warpage,
        compute_warpage_yield,
    )


def _substack_specs(cfg_dict, _3dbx_path):
    specs = []
    for substack in stack_graph_from_3dbx(_3dbx_path):
        substack_id = int(substack["substack_id"])
        chiplets = list(substack["chiplets_bottom_to_top"])
        interfaces = list(substack["interfaces_bottom_to_top"])
        if not interfaces:
            raise ValueError(
                f"Substack {substack_id} has no bonding interfaces; "
                "cannot simulate final stack warpage."
            )

        final_interface = interfaces[-1]
        if final_interface not in cfg_dict:
            raise KeyError(f"Interface '{final_interface}' from {_3dbx_path} is missing in cfg_dict.")

        final_cfg = cfg_dict[final_interface]
        layer_df, layer_bow_sigma_um = _build_layer_df(
            cfg_dict,
            chiplets,
            interfaces,
        )

        specs.append({
            "substack_id": substack_id,
            "chiplets_bottom_to_top": chiplets,
            "interfaces_bottom_to_top": interfaces,
            "final_interface": final_interface,
            "layer_df": layer_df,
            "layer_bow_sigma_um": np.asarray(layer_bow_sigma_um, dtype=float),
            "threshold_um": _stack_warpage_threshold_um(final_cfg),
            "L_m": _top_die_half_length_m(final_cfg),
            "DeltaT_K": _delta_t_from_cfg(final_cfg),
        })
    return specs


def _collect_layer_sample_distributions(specs):
    sample_distributions = {}
    for spec in specs:
        layer_df = spec["layer_df"].reset_index(drop=True)
        layer_sigmas = spec["layer_bow_sigma_um"]
        for layer_index, row in layer_df.iterrows():
            key = str(row["chiplet"])
            dist = {
                "mean_um": float(row["W0_um"]),
                "sigma_um": float(layer_sigmas[layer_index]),
            }
            if key in sample_distributions:
                old = sample_distributions[key]
                if (
                    not np.isclose(old["mean_um"], dist["mean_um"])
                    or not np.isclose(old["sigma_um"], dist["sigma_um"])
                ):
                    raise ValueError(
                        f"Chiplet '{key}' appears with inconsistent initial bow distributions: "
                        f"{old} vs {dist}. Instance-level sampling needs instance IDs."
                    )
            else:
                sample_distributions[key] = dist
    return sample_distributions


def _sample_initial_bows(sample_distributions, num_samples, rng):
    samples = {}
    for key, dist in sample_distributions.items():
        samples[key] = rng.normal(
            loc=dist["mean_um"],
            scale=dist["sigma_um"],
            size=int(num_samples),
        )
    return samples


def _simulate_one_substack(spec, initial_bow_samples, num_samples):
    layer_df = spec["layer_df"].reset_index(drop=True)
    threshold_um = float(spec["threshold_um"])
    warpage_samples_um = _sample_stack_warpage(
        layer_df=layer_df,
        initial_bow_samples=initial_bow_samples,
        num_samples=num_samples,
        DeltaT_K=spec["DeltaT_K"],
        L_m=spec["L_m"],
    )
    pass_vector = np.abs(warpage_samples_um) <= threshold_um
    return warpage_samples_um, pass_vector


def _sample_stack_warpage(layer_df, initial_bow_samples, num_samples, DeltaT_K, L_m):
    warpage_samples_um = np.zeros(int(num_samples), dtype=np.float64)

    for sample_index in range(int(num_samples)):
        sampled_layer_df = layer_df.copy()
        sampled_layer_df["W0_um"] = [
            initial_bow_samples[str(chiplet)][sample_index]
            for chiplet in layer_df["chiplet"]
        ]
        summary, _ = compute_total_stack_warpage(
            sampled_layer_df,
            DeltaT_K=DeltaT_K,
            L_m=L_m,
        )
        warpage_samples_um[sample_index] = summary["signed_W_um"]

    return warpage_samples_um


def _interface_prebond_specs(cfg_dict, _3dbx_path):
    specs = []
    for substack in stack_graph_from_3dbx(_3dbx_path):
        substack_id = int(substack["substack_id"])
        chiplets = list(substack["chiplets_bottom_to_top"])
        interfaces = list(substack["interfaces_bottom_to_top"])

        for interface_index, interface in enumerate(interfaces):
            if interface not in cfg_dict:
                raise KeyError(f"Interface '{interface}' from {_3dbx_path} is missing in cfg_dict.")

            current_cfg = cfg_dict[interface]
            prebond_chiplets = chiplets[:interface_index + 1]
            layer_df, _ = _build_layer_df(
                cfg_dict,
                prebond_chiplets,
                interfaces,
            )

            if interface_index == 0:
                last_completed_interface = None
                DeltaT_K = 0.0
            else:
                last_completed_interface = interfaces[interface_index - 1]
                DeltaT_K = _prebond_delta_t_from_cfg(
                    current_cfg,
                    cfg_dict[last_completed_interface],
                )

            specs.append({
                "substack_id": substack_id,
                "interface": interface,
                "interface_index": interface_index,
                "chiplets_before_bond": prebond_chiplets,
                "incoming_top_chiplet": chiplets[interface_index + 1],
                "interfaces_before_bond": interfaces[:interface_index],
                "last_completed_interface": last_completed_interface,
                "layer_df": layer_df,
                "L_m": _top_die_half_length_m(current_cfg),
                "DeltaT_K": DeltaT_K,
                "anneal_mode": _anneal_mode_from_cfg(current_cfg),
            })

    return specs


def sample_interface_bow_difference(
    cfg_dict,
    _3dbx_path,
    *,
    num_samples,
    return_details=False,
):
    """
    Sample interface-level bow difference for overlay simulation.

    For each bonding interface:

        bow_difference = incoming_top_die_initial_bow - existing_substack_warpage

    The existing substack warpage is recomputed sample-by-sample from the
    already-bonded lower part of the substack using the same physical stack
    model as the warpage yield simulator.
    """
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    rng = np.random.default_rng()
    final_specs = _substack_specs(cfg_dict, _3dbx_path)
    interface_specs = _interface_prebond_specs(cfg_dict, _3dbx_path)
    sample_distributions = _collect_layer_sample_distributions(final_specs)
    initial_bow_samples = _sample_initial_bows(sample_distributions, num_samples, rng)

    bow_difference_samples = {}
    existing_stack_warpage_samples = {}
    details = {}

    for spec in interface_specs:
        interface = spec["interface"]
        if interface in bow_difference_samples:
            raise ValueError(
                f"Interface '{interface}' appears in multiple substacks; "
                "instance-level interface sampling is not yet supported."
            )

        existing_samples_um = _sample_stack_warpage(
            layer_df=spec["layer_df"].reset_index(drop=True),
            initial_bow_samples=initial_bow_samples,
            num_samples=num_samples,
            DeltaT_K=spec["DeltaT_K"],
            L_m=spec["L_m"],
        )
        incoming_samples_um = initial_bow_samples[spec["incoming_top_chiplet"]]
        bow_difference_samples[interface] = incoming_samples_um - existing_samples_um
        existing_stack_warpage_samples[interface] = existing_samples_um

        details[interface] = {
            "substack_id": spec["substack_id"],
            "interface_index": spec["interface_index"],
            "chiplets_before_bond": list(spec["chiplets_before_bond"]),
            "incoming_top_chiplet": spec["incoming_top_chiplet"],
            "interfaces_before_bond": list(spec["interfaces_before_bond"]),
            "last_completed_interface": spec["last_completed_interface"],
            "L_m": float(spec["L_m"]),
            "DeltaT_K": float(spec["DeltaT_K"]),
            "anneal_mode": spec["anneal_mode"],
        }

    if return_details:
        return {
            "interface_bow_difference_samples": bow_difference_samples,
            "existing_stack_warpage_samples": existing_stack_warpage_samples,
            "initial_bow_samples": initial_bow_samples,
            "sample_distributions": sample_distributions,
            "details": details,
            "num_samples": num_samples,
            "random_source": "system_entropy",
        }

    return bow_difference_samples


def write_warpage_yield_simulation_summary(output_path, simulation_result):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("Warpage Yield Monte Carlo Summary\n")
        f.write("=================================\n\n")
        f.write(f"num_samples: {simulation_result['num_samples']}\n")
        f.write(f"random_source: {simulation_result.get('random_source', 'system_entropy')}\n")
        f.write(f"overall_warpage_yield: {simulation_result['overall_warpage_yield']:.8f}\n")
        if simulation_result.get("calculator_reference") is not None:
            ref = simulation_result["calculator_reference"]
            f.write(f"calculator_overall_warpage_yield: {ref['overall_warpage_yield']:.8f}\n")
            f.write(f"overall_yield_delta: {simulation_result['overall_yield_delta_vs_calculator']:.8e}\n")
        f.write("\n")

        f.write("Substack Results\n")
        f.write("----------------\n")
        for substack_id, info in simulation_result["substack_results"].items():
            f.write(f"[substack {substack_id}]\n")
            f.write(f"chiplets_bottom_to_top: {info['chiplets_bottom_to_top']}\n")
            f.write(f"interfaces_bottom_to_top: {info['interfaces_bottom_to_top']}\n")
            f.write(f"final_interface: {info['final_interface']}\n")
            f.write(f"threshold_um: {info['threshold_um']:.6g}\n")
            f.write(f"sample_mean_um: {info['sample_mean_um']:.8f}\n")
            f.write(f"sample_std_um: {info['sample_std_um']:.8f}\n")
            f.write(f"warpage_yield: {info['warpage_yield']:.8f}\n")
            f.write(f"fail_count: {info['fail_count']}\n")
            if "calculator_mu_um" in info:
                f.write(f"calculator_mu_um: {info['calculator_mu_um']:.8f}\n")
                f.write(f"calculator_sigma_um: {info['calculator_sigma_um']:.8f}\n")
                f.write(f"calculator_warpage_yield: {info['calculator_warpage_yield']:.8f}\n")
                f.write(f"yield_delta_vs_calculator: {info['yield_delta_vs_calculator']:.8e}\n")
            f.write("\n")

    return output_path


def simulate_warpage_yield(
    cfg_dict,
    _3dbx_path,
    *,
    num_samples,
    include_calculator_reference=True,
    return_samples=False,
):
    """
    Estimate stack warpage yield by Monte Carlo sampling layer initial bows.

    A stack sample passes only when every final substack satisfies:

        abs(W_substack_um) <= STACK_WARPAGE_TH
    """
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    rng = np.random.default_rng()
    specs = _substack_specs(cfg_dict, _3dbx_path)
    sample_distributions = _collect_layer_sample_distributions(specs)
    initial_bow_samples = _sample_initial_bows(sample_distributions, num_samples, rng)

    substack_results = {}
    stack_pass_vector = np.ones(num_samples, dtype=bool)

    for spec in specs:
        substack_id = int(spec["substack_id"])
        warpage_samples_um, substack_pass_vector = _simulate_one_substack(
            spec,
            initial_bow_samples,
            num_samples,
        )
        stack_pass_vector &= substack_pass_vector

        substack_result = {
            "substack_id": substack_id,
            "chiplets_bottom_to_top": list(spec["chiplets_bottom_to_top"]),
            "interfaces_bottom_to_top": list(spec["interfaces_bottom_to_top"]),
            "final_interface": spec["final_interface"],
            "threshold_um": float(spec["threshold_um"]),
            "L_m": float(spec["L_m"]),
            "DeltaT_K": float(spec["DeltaT_K"]),
            "sample_mean_um": float(np.mean(warpage_samples_um)),
            "sample_std_um": float(np.std(warpage_samples_um, ddof=1)) if num_samples > 1 else 0.0,
            "warpage_yield": float(np.mean(substack_pass_vector)),
            "fail_count": int(np.count_nonzero(~substack_pass_vector)),
            "layers": spec["layer_df"].to_dict(orient="records"),
        }
        if return_samples:
            substack_result["warpage_samples_um"] = warpage_samples_um.copy()
            substack_result["pass_vector"] = substack_pass_vector.copy()
        substack_results[substack_id] = substack_result

    result = {
        "overall_warpage_yield": float(np.mean(stack_pass_vector)),
        "num_samples": num_samples,
        "random_source": "system_entropy",
        "substack_results": substack_results,
        "substack_yield_dict": {
            substack_id: info["warpage_yield"]
            for substack_id, info in substack_results.items()
        },
        "sample_distributions": sample_distributions,
    }
    if return_samples:
        result["stack_pass_vector"] = stack_pass_vector.copy()

    if include_calculator_reference:
        calculator_reference = compute_warpage_yield(cfg_dict, _3dbx_path)
        result["calculator_reference"] = calculator_reference
        result["overall_yield_delta_vs_calculator"] = (
            result["overall_warpage_yield"]
            - calculator_reference["overall_warpage_yield"]
        )
        for substack_id, info in result["substack_results"].items():
            ref_info = calculator_reference["substack_results"].get(substack_id)
            if ref_info is None:
                continue
            info["calculator_mu_um"] = float(ref_info["mu_um"])
            info["calculator_sigma_um"] = float(ref_info["sigma_um"])
            info["calculator_warpage_yield"] = float(ref_info["warpage_yield"])
            info["yield_delta_vs_calculator"] = (
                info["warpage_yield"]
                - float(ref_info["warpage_yield"])
            )
    else:
        result["calculator_reference"] = None
        result["overall_yield_delta_vs_calculator"] = None

    return result


def stack_warpage_fail_vector_for_epoch(
    *,
    input_args,
    cfg_dict,
    stack_cfg_dict=None,
    num_samples,
):
    """
    Return a stack-level final-warpage failure vector for one simulation epoch.

    A sample fails when any final substack has:

        abs(W_substack_um) > STACK_WARPAGE_TH
    """
    ds_dir = input_args.get('ds_dir', '')
    _3dbx_path = os.path.join(ds_dir, 'generated_stack_config.3dbx')
    if not os.path.exists(_3dbx_path):
        raise FileNotFoundError(
            "D2W warpage simulation requires generated_stack_config.3dbx. "
            f"Could not find {_3dbx_path}."
        )

    stack_cfg_dict = cfg_dict if stack_cfg_dict is None else stack_cfg_dict

    try:
        warpage_result = simulate_warpage_yield(
            stack_cfg_dict,
            _3dbx_path,
            num_samples=num_samples,
            include_calculator_reference=False,
            return_samples=True,
        )
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise RuntimeError(
            "D2W final stack warpage simulation failed; yield evaluation cannot "
            "continue with an all-pass fallback."
        ) from exc

    return ~np.asarray(warpage_result["stack_pass_vector"], dtype=bool)


def Warpage_Yield_Simulator(
    input_args,
    cfg_skeleton,
    cfg_dict,
    _3dbx_path,
    *,
    return_samples=False,
    write_summary=False,
):
    num_samples = int(getattr(cfg_skeleton, "NUM_DIE_STACKS", 0) or input_args.get("NUM_DIE_STACKS", 10000))
    result = simulate_warpage_yield(
        cfg_dict,
        _3dbx_path,
        num_samples=num_samples,
        include_calculator_reference=True,
        return_samples=return_samples,
    )

    result["summary_path"] = None
    if write_summary:
        first_cfg = next(iter(cfg_dict.values()))
        ds_name = input_args.get("ds_name", getattr(first_cfg, "DESIGN", "design"))
        file_suffix = input_args.get("output_file_tag", "")
        output_root = os.path.join(first_cfg.OUTPUT_DIR, ds_name)
        summary_path = os.path.join(output_root, f"warpage_yield_summary{file_suffix}.txt")
        result["summary_path"] = write_warpage_yield_simulation_summary(summary_path, result)
    return result
