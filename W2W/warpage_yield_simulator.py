#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monte Carlo simulator for validating W2W sequential stack warpage yield.

Each wafer layer's initial bow is sampled, then each bonding step is followed
by an anneal/release calculation. A full wafer stack sample fails when any
final substack exceeds the configured absolute warpage threshold.
"""

import os

import numpy as np

try:
    from warpage_yield_calculator import (
        _build_layer_df,
        _initial_bow_stats_um,
        _sequential_delta_t_by_step,
        _stack_warpage_threshold_um,
        _wafer_half_length_m,
        compute_sequential_stack_warpage,
        compute_warpage_yield,
        stack_graph_from_3dbx,
    )
except ModuleNotFoundError:
    from W2W.warpage_yield_calculator import (
        _build_layer_df,
        _initial_bow_stats_um,
        _sequential_delta_t_by_step,
        _stack_warpage_threshold_um,
        _wafer_half_length_m,
        compute_sequential_stack_warpage,
        compute_warpage_yield,
        stack_graph_from_3dbx,
    )


def _layer_sample_key(row):
    return str(row.get("chiplet_instance", row["chiplet"]))


def _substack_specs(cfg_dict, _3dbx_path, num_dies_per_wafer=None):
    specs = []
    for substack in stack_graph_from_3dbx(_3dbx_path):
        substack_id = int(substack["substack_id"])
        chiplets = list(substack["chiplets_bottom_to_top"])
        chiplet_instances = list(substack["chiplet_instances_bottom_to_top"])
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
            chiplet_instances,
            num_dies_per_wafer=num_dies_per_wafer,
        )

        specs.append({
            "substack_id": substack_id,
            "chiplets_bottom_to_top": chiplets,
            "chiplet_instances_bottom_to_top": chiplet_instances,
            "interfaces_bottom_to_top": interfaces,
            "final_interface": final_interface,
            "layer_df": layer_df,
            "layer_bow_sigma_um": np.asarray(layer_bow_sigma_um, dtype=float),
            "threshold_um": _stack_warpage_threshold_um(final_cfg),
            "L_m": _wafer_half_length_m(final_cfg),
            "DeltaT_K_by_step": _sequential_delta_t_by_step(cfg_dict, interfaces),
        })
    return specs


def _collect_layer_sample_distributions(specs):
    sample_distributions = {}
    for spec in specs:
        layer_df = spec["layer_df"].reset_index(drop=True)
        layer_sigmas = spec["layer_bow_sigma_um"]
        for layer_index, row in layer_df.iterrows():
            key = _layer_sample_key(row)
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
                        f"Layer instance '{key}' appears with inconsistent initial bow "
                        f"distributions: {old} vs {dist}."
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


def _sample_sequential_stack_warpage(
    layer_df,
    initial_bow_samples,
    num_samples,
    DeltaT_K_by_step,
    L_m,
):
    num_samples = int(num_samples)
    num_steps = max(len(layer_df) - 1, 0)
    warpage_samples_um = np.zeros(num_samples, dtype=np.float64)
    step_samples_um = np.zeros((num_samples, num_steps), dtype=np.float64)

    layer_keys = [_layer_sample_key(row) for _, row in layer_df.iterrows()]
    for sample_index in range(num_samples):
        sampled_layer_df = layer_df.copy()
        sampled_layer_df["W0_um"] = [
            initial_bow_samples[key][sample_index]
            for key in layer_keys
        ]
        final_summary, step_summaries = compute_sequential_stack_warpage(
            sampled_layer_df,
            DeltaT_K_by_step=DeltaT_K_by_step,
            L_m=L_m,
        )
        warpage_samples_um[sample_index] = final_summary["signed_W_um"]
        for step_index, step in enumerate(step_summaries):
            step_samples_um[sample_index, step_index] = step["signed_W_um"]

    return warpage_samples_um, step_samples_um


def _simulate_one_substack(spec, initial_bow_samples, num_samples):
    layer_df = spec["layer_df"].reset_index(drop=True)
    threshold_um = float(spec["threshold_um"])
    warpage_samples_um, step_samples_um = _sample_sequential_stack_warpage(
        layer_df=layer_df,
        initial_bow_samples=initial_bow_samples,
        num_samples=num_samples,
        DeltaT_K_by_step=spec["DeltaT_K_by_step"],
        L_m=spec["L_m"],
    )
    pass_vector = np.abs(warpage_samples_um) <= threshold_um
    return warpage_samples_um, step_samples_um, pass_vector


def sample_interface_bow_difference(
    cfg_dict,
    _3dbx_path,
    *,
    num_samples,
    num_dies_per_wafer=None,
    return_details=False,
):
    """
    Sample interface-level bow difference for W2W overlay simulation.

    For interface k:

        bow_difference = incoming_wafer_initial_bow
                         - existing_stack_post_anneal_bow

    The first interface uses the bottom wafer initial bow as the existing
    stack. Later interfaces use the post-anneal bow from the previous step.
    """
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    rng = np.random.default_rng()
    specs = _substack_specs(
        cfg_dict,
        _3dbx_path,
        num_dies_per_wafer=num_dies_per_wafer,
    )
    sample_distributions = _collect_layer_sample_distributions(specs)
    initial_bow_samples = _sample_initial_bows(sample_distributions, num_samples, rng)

    bow_difference_samples = {}
    existing_stack_warpage_samples = {}
    details = {}

    for spec in specs:
        layer_df = spec["layer_df"].reset_index(drop=True)
        _, step_samples_um = _sample_sequential_stack_warpage(
            layer_df=layer_df,
            initial_bow_samples=initial_bow_samples,
            num_samples=num_samples,
            DeltaT_K_by_step=spec["DeltaT_K_by_step"],
            L_m=spec["L_m"],
        )

        for interface_index, interface in enumerate(spec["interfaces_bottom_to_top"]):
            if interface in bow_difference_samples:
                raise ValueError(
                    f"Interface '{interface}' appears in multiple substacks; "
                    "instance-level interface sampling is not yet supported."
                )

            incoming_key = str(layer_df.loc[interface_index + 1, "chiplet_instance"])
            if interface_index == 0:
                existing_key = str(layer_df.loc[0, "chiplet_instance"])
                existing_samples_um = initial_bow_samples[existing_key]
                last_completed_interface = None
            else:
                existing_samples_um = step_samples_um[:, interface_index - 1]
                last_completed_interface = spec["interfaces_bottom_to_top"][interface_index - 1]

            incoming_samples_um = initial_bow_samples[incoming_key]
            bow_difference_samples[interface] = incoming_samples_um - existing_samples_um
            existing_stack_warpage_samples[interface] = existing_samples_um

            current_cfg = cfg_dict[interface]
            incoming_mu_um, incoming_sigma_um = _initial_bow_stats_um(current_cfg, "TOP")
            details[interface] = {
                "substack_id": spec["substack_id"],
                "interface_index": int(interface_index),
                "chiplets_before_bond": spec["chiplets_bottom_to_top"][:interface_index + 1],
                "chiplet_instances_before_bond": spec["chiplet_instances_bottom_to_top"][:interface_index + 1],
                "incoming_top_chiplet": spec["chiplets_bottom_to_top"][interface_index + 1],
                "incoming_top_chiplet_instance": incoming_key,
                "last_completed_interface": last_completed_interface,
                "incoming_top_mu_um": float(incoming_mu_um),
                "incoming_top_sigma_um": float(incoming_sigma_um),
                "L_m": float(spec["L_m"]),
                "anneal_mode": "sequential",
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


def sample_w2w_warpage_process(
    cfg_dict,
    _3dbx_path,
    *,
    num_samples,
    num_dies_per_wafer=None,
    return_samples=False,
):
    """
    Sample the complete W2W warpage process once.

    The returned interface bow-difference samples and final stack pass vector
    are generated from the same initial bow samples, so overlay magnification
    and final warpage failure remain correlated in simulation.
    """
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    rng = np.random.default_rng()
    specs = _substack_specs(
        cfg_dict,
        _3dbx_path,
        num_dies_per_wafer=num_dies_per_wafer,
    )
    sample_distributions = _collect_layer_sample_distributions(specs)
    initial_bow_samples = _sample_initial_bows(sample_distributions, num_samples, rng)

    bow_difference_samples = {}
    existing_stack_warpage_samples = {}
    substack_results = {}
    stack_pass_vector = np.ones(num_samples, dtype=bool)

    for spec in specs:
        substack_id = int(spec["substack_id"])
        layer_df = spec["layer_df"].reset_index(drop=True)
        final_samples_um, step_samples_um = _sample_sequential_stack_warpage(
            layer_df=layer_df,
            initial_bow_samples=initial_bow_samples,
            num_samples=num_samples,
            DeltaT_K_by_step=spec["DeltaT_K_by_step"],
            L_m=spec["L_m"],
        )
        substack_pass_vector = np.abs(final_samples_um) <= float(spec["threshold_um"])
        stack_pass_vector &= substack_pass_vector

        substack_result = {
            "substack_id": substack_id,
            "chiplets_bottom_to_top": list(spec["chiplets_bottom_to_top"]),
            "chiplet_instances_bottom_to_top": list(spec["chiplet_instances_bottom_to_top"]),
            "interfaces_bottom_to_top": list(spec["interfaces_bottom_to_top"]),
            "final_interface": spec["final_interface"],
            "threshold_um": float(spec["threshold_um"]),
            "warpage_yield": float(np.mean(substack_pass_vector)),
            "fail_count": int(np.count_nonzero(~substack_pass_vector)),
            "sample_mean_um": float(np.mean(final_samples_um)),
            "sample_std_um": float(np.std(final_samples_um, ddof=1)) if num_samples > 1 else 0.0,
        }
        if return_samples:
            substack_result["warpage_samples_um"] = final_samples_um.copy()
            substack_result["step_samples_um"] = step_samples_um.copy()
            substack_result["pass_vector"] = substack_pass_vector.copy()
        substack_results[substack_id] = substack_result

        for interface_index, interface in enumerate(spec["interfaces_bottom_to_top"]):
            if interface in bow_difference_samples:
                raise ValueError(
                    f"Interface '{interface}' appears in multiple substacks; "
                    "instance-level interface sampling is not yet supported."
                )

            incoming_key = str(layer_df.loc[interface_index + 1, "chiplet_instance"])
            if interface_index == 0:
                existing_key = str(layer_df.loc[0, "chiplet_instance"])
                existing_samples_um = initial_bow_samples[existing_key]
            else:
                existing_samples_um = step_samples_um[:, interface_index - 1]

            incoming_samples_um = initial_bow_samples[incoming_key]
            bow_difference_samples[interface] = incoming_samples_um - existing_samples_um
            existing_stack_warpage_samples[interface] = existing_samples_um

    result = {
        "overall_warpage_yield": float(np.mean(stack_pass_vector)),
        "stack_pass_vector": stack_pass_vector.copy(),
        "interface_bow_difference_samples": bow_difference_samples,
        "existing_stack_warpage_samples": existing_stack_warpage_samples,
        "substack_results": substack_results,
        "sample_distributions": sample_distributions,
        "num_samples": num_samples,
        "random_source": "system_entropy",
    }
    if return_samples:
        result["initial_bow_samples"] = initial_bow_samples
    return result


def simulate_warpage_yield(
    cfg_dict,
    _3dbx_path,
    *,
    num_samples,
    num_dies_per_wafer=None,
    include_calculator_reference=True,
    return_samples=False,
):
    """
    Estimate W2W final stack warpage yield by Monte Carlo.
    """
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    rng = np.random.default_rng()
    specs = _substack_specs(
        cfg_dict,
        _3dbx_path,
        num_dies_per_wafer=num_dies_per_wafer,
    )
    sample_distributions = _collect_layer_sample_distributions(specs)
    initial_bow_samples = _sample_initial_bows(sample_distributions, num_samples, rng)

    substack_results = {}
    stack_pass_vector = np.ones(num_samples, dtype=bool)

    for spec in specs:
        substack_id = int(spec["substack_id"])
        warpage_samples_um, step_samples_um, substack_pass_vector = _simulate_one_substack(
            spec,
            initial_bow_samples,
            num_samples,
        )
        stack_pass_vector &= substack_pass_vector

        substack_result = {
            "substack_id": substack_id,
            "chiplets_bottom_to_top": list(spec["chiplets_bottom_to_top"]),
            "chiplet_instances_bottom_to_top": list(spec["chiplet_instances_bottom_to_top"]),
            "interfaces_bottom_to_top": list(spec["interfaces_bottom_to_top"]),
            "final_interface": spec["final_interface"],
            "threshold_um": float(spec["threshold_um"]),
            "L_m": float(spec["L_m"]),
            "DeltaT_K_by_step": spec["DeltaT_K_by_step"].tolist(),
            "anneal_mode": "sequential",
            "sample_mean_um": float(np.mean(warpage_samples_um)),
            "sample_std_um": float(np.std(warpage_samples_um, ddof=1)) if num_samples > 1 else 0.0,
            "warpage_yield": float(np.mean(substack_pass_vector)),
            "fail_count": int(np.count_nonzero(~substack_pass_vector)),
            "layers": spec["layer_df"].to_dict(orient="records"),
        }
        if return_samples:
            substack_result["warpage_samples_um"] = warpage_samples_um.copy()
            substack_result["step_samples_um"] = step_samples_um.copy()
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
        calculator_reference = compute_warpage_yield(
            cfg_dict,
            _3dbx_path,
            num_dies_per_wafer=num_dies_per_wafer,
        )
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
    Return a wafer-stack-level final-warpage failure vector for one epoch.
    """
    ds_dir = input_args.get("ds_dir", "")
    _3dbx_path = os.path.join(ds_dir, "generated_stack_config.3dbx")
    if not os.path.exists(_3dbx_path):
        raise FileNotFoundError(
            "W2W warpage simulation requires generated_stack_config.3dbx. "
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
            "W2W final stack warpage simulation failed; yield evaluation cannot "
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
):
    num_samples = int(
        getattr(cfg_skeleton, "NUM_WAFER_STACKS", 0)
        or input_args.get("NUM_WAFER_STACKS", 10000)
    )
    return simulate_warpage_yield(
        cfg_dict,
        _3dbx_path,
        num_samples=num_samples,
        include_calculator_reference=True,
        return_samples=return_samples,
    )


def warpage_yield_simulator(
    cfg_dict,
    _3dbx_path,
    *,
    num_samples,
    return_samples=False,
):
    return simulate_warpage_yield(
        cfg_dict,
        _3dbx_path,
        num_samples=num_samples,
        include_calculator_reference=True,
        return_samples=return_samples,
    )
