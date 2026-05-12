#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sequential W2W stack warpage yield calculator.

In W2W processing, each wafer-to-wafer bonding step is followed by an anneal.
The post-anneal bow of the already-bonded stack is therefore carried into the
next bonding step. The final warpage criterion is stack-level:

    abs(W_final) <= STACK_WARPAGE_TH

Only the final interface in each root-to-leaf substack receives the warpage
yield value; earlier interfaces are set to 1.0 to avoid double counting.
"""

from collections import defaultdict
import math

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


def _instance_from_3dbx_endpoint(endpoint) -> str:
    return str(endpoint).split(".regions.")[0]


def _interface_from_3dbx_connection(connection) -> str:
    interface_top = str(((connection.bot).split(".")[-1]).split("To_")[-1])
    interface_bot = str(((connection.top).split(".")[-1]).split("From_")[-1])
    return f"{interface_top}_From_{interface_bot}"


def stack_graph_from_3dbx(_3dbx_path: str) -> list[dict]:
    """
    Parse a 3Dblox stack config and return root-to-leaf substacks.

    Each item contains both chiplet reference names, which match cfg_dict
    interface naming, and chiplet instance names, which keep Monte Carlo layer
    samples independent when the same reference appears multiple times.
    """
    stack_config_3dbx = OmegaConf.load(_3dbx_path)
    if "Stack" not in stack_config_3dbx or "ChipletInst" not in stack_config_3dbx:
        raise ValueError(f"{_3dbx_path} must contain Stack and ChipletInst sections.")

    stack = stack_config_3dbx.Stack
    chiplet_inst = stack_config_3dbx.ChipletInst
    stack_order = {name: idx for idx, name in enumerate(stack.keys())}
    children_by_parent = defaultdict(list)
    parent_by_child = {}

    if "Connection" in stack_config_3dbx:
        for connection_order, (connection_name, connection) in enumerate(stack_config_3dbx.Connection.items()):
            top_instance = _instance_from_3dbx_endpoint(connection.top)
            bot_instance = _instance_from_3dbx_endpoint(connection.bot)
            if top_instance not in stack:
                raise ValueError(f"Connection {connection_name} top instance '{top_instance}' is not in Stack.")
            if bot_instance not in stack:
                raise ValueError(f"Connection {connection_name} bottom instance '{bot_instance}' is not in Stack.")
            if top_instance in parent_by_child:
                raise ValueError(
                    f"Instance '{top_instance}' has multiple bottom parents; "
                    "substack extraction expects a tree/DAG of vertical paths."
                )

            children_by_parent[bot_instance].append({
                "child": top_instance,
                "connection": str(connection_name),
                "interface": _interface_from_3dbx_connection(connection),
                "order": connection_order,
            })
            parent_by_child[top_instance] = bot_instance

    roots = [name for name, node in stack.items() if bool(node.get("root", False))]
    if not roots:
        roots = [name for name in stack.keys() if name not in parent_by_child]
    if not roots:
        raise ValueError(f"No stack root could be inferred from {_3dbx_path}.")

    for parent in children_by_parent:
        children_by_parent[parent].sort(key=lambda edge: edge["order"])
    roots.sort(key=lambda name: stack_order[name])

    substacks = []

    def dfs(instance_name, instance_path, chiplet_path, interface_path, active_path):
        if instance_name in active_path:
            cycle = " -> ".join(list(active_path) + [instance_name])
            raise ValueError(f"Cycle detected in stack graph: {cycle}")
        if instance_name not in chiplet_inst:
            raise ValueError(f"Stack instance '{instance_name}' is not in ChipletInst.")

        next_instance_path = instance_path + [str(instance_name)]
        next_chiplet_path = chiplet_path + [str(chiplet_inst[instance_name].reference)]
        child_edges = children_by_parent.get(instance_name, [])
        if not child_edges:
            substacks.append({
                "substack_id": len(substacks),
                "chiplet_instances_bottom_to_top": next_instance_path,
                "chiplets_bottom_to_top": next_chiplet_path,
                "interfaces_bottom_to_top": interface_path,
            })
            return

        next_active_path = set(active_path)
        next_active_path.add(instance_name)
        for edge in child_edges:
            dfs(
                edge["child"],
                next_instance_path,
                next_chiplet_path,
                interface_path + [edge["interface"]],
                next_active_path,
            )

    for root in roots:
        dfs(root, [], [], [], set())

    return substacks


def bow_um_to_curvature(W0_um, L_m):
    """Convert signed bow in um over half-length L_m to curvature in 1/m."""
    return 2.0 * (np.asarray(W0_um, dtype=float) * 1e-6) / (L_m**2)


def curvature_to_bow_um(kappa, L_m):
    """Convert signed curvature in 1/m to signed bow in um."""
    return 0.5 * kappa * L_m**2 * 1e6


def compute_total_stack_warpage(layer_df, DeltaT_K, L_m):
    """
    Compute multilayer total stack bow for one anneal/release event.

    Required layer_df columns:
      - h_um
      - E_GPa
      - nu
      - alpha_ppm_K
      - W0_um
    """
    required = ["h_um", "E_GPa", "nu", "alpha_ppm_K", "W0_um"]
    missing = [col for col in required if col not in layer_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = layer_df.copy().reset_index(drop=True)
    h = df["h_um"].to_numpy(dtype=float) * 1e-6
    E = df["E_GPa"].to_numpy(dtype=float) * 1e9
    nu = df["nu"].to_numpy(dtype=float)
    alpha = df["alpha_ppm_K"].to_numpy(dtype=float) * 1e-6
    kappa0 = bow_um_to_curvature(df["W0_um"].to_numpy(dtype=float), L_m)

    if np.any(h <= 0):
        raise ValueError("All layer thicknesses must be positive.")
    if np.any((nu <= -1.0) | (nu >= 0.5)):
        raise ValueError("Poisson ratios should be in the elastic range (-1, 0.5).")

    E_biaxial = E / (1.0 - nu)
    lam = 1.0 / (E_biaxial * h)
    weights = 1.0 / lam

    h_cumsum = np.cumsum(h)
    hk0_cumsum = np.cumsum(h * kappa0)

    a = h_cumsum - 0.5 * (h[0] + h)
    b = hk0_cumsum - 0.5 * (h[0] * kappa0[0] + h * kappa0)

    delta_alpha = alpha - alpha[0]
    s_alpha = np.sum(delta_alpha * weights) / np.sum(weights)
    s_a = np.sum(a * weights) / np.sum(weights)
    s_b = np.sum(b * weights) / np.sum(weights)

    D = E_biaxial * h**3 / 12.0
    M_T = np.sum((a / lam) * (delta_alpha - s_alpha) * DeltaT_K)
    M_b = np.sum((a / lam) * (b - s_b))
    K_a = np.sum((a / lam) * (a - s_a))
    K_D = np.sum(D)
    M_D0 = np.sum(D * kappa0)

    kappa = (M_D0 + M_T + M_b) / (K_D + K_a)
    W_um = curvature_to_bow_um(kappa, L_m)

    strain_term = (delta_alpha - s_alpha) * DeltaT_K - (a - s_a) * kappa + (b - s_b)
    F = weights * strain_term
    sigma = E_biaxial * strain_term

    summary = {
        "kappa_1_per_m": float(kappa),
        "signed_W_um": float(W_um),
        "abs_W_um": float(abs(W_um)),
        "DeltaT_K": float(DeltaT_K),
        "L_m": float(L_m),
    }
    layer_out = df.copy()
    layer_out["E_biaxial_GPa"] = E_biaxial / 1e9
    layer_out["kappa0_1_per_m"] = kappa0
    layer_out["F_N_per_m"] = F
    layer_out["sigma_MPa"] = sigma / 1e6
    return summary, layer_out


def compute_sequential_stack_warpage(layer_df, DeltaT_K_by_step, L_m):
    """
    Compute W2W warpage after each sequential bonding anneal.

    The already-bonded lower stack is represented by its carried post-release
    bow before each new wafer is appended. Layer stiffness/thicknesses remain
    explicit, while prior layers' initial bow is set to the carried stack bow.
    """
    df = layer_df.copy().reset_index(drop=True)
    num_layers = len(df)
    if num_layers < 1:
        raise ValueError("layer_df must contain at least one layer.")

    delta_t_values = np.asarray(DeltaT_K_by_step, dtype=float)
    if delta_t_values.size == 1 and num_layers > 1:
        delta_t_values = np.full(num_layers - 1, float(delta_t_values[0]))
    if delta_t_values.size != max(num_layers - 1, 0):
        raise ValueError(
            f"DeltaT_K_by_step must have {num_layers - 1} entries for "
            f"{num_layers} layers; got {delta_t_values.size}."
        )

    carried_W_um = float(df.loc[0, "W0_um"])
    step_summaries = []

    for layer_index in range(1, num_layers):
        current_df = df.iloc[:layer_index + 1].copy().reset_index(drop=True)
        current_df.loc[:layer_index - 1, "W0_um"] = carried_W_um
        summary, layer_out = compute_total_stack_warpage(
            current_df,
            DeltaT_K=float(delta_t_values[layer_index - 1]),
            L_m=L_m,
        )
        carried_W_um = float(summary["signed_W_um"])
        step_summaries.append({
            "interface_index": layer_index - 1,
            "layer_count": layer_index + 1,
            "signed_W_um": carried_W_um,
            "abs_W_um": float(abs(carried_W_um)),
            "kappa_1_per_m": float(summary["kappa_1_per_m"]),
            "DeltaT_K": float(summary["DeltaT_K"]),
            "L_m": float(summary["L_m"]),
            "layers": layer_out.to_dict(orient="records"),
        })

    if step_summaries:
        final_summary = {
            "signed_W_um": float(step_summaries[-1]["signed_W_um"]),
            "abs_W_um": float(step_summaries[-1]["abs_W_um"]),
            "kappa_1_per_m": float(step_summaries[-1]["kappa_1_per_m"]),
            "DeltaT_K": float(step_summaries[-1]["DeltaT_K"]),
            "L_m": float(step_summaries[-1]["L_m"]),
        }
    else:
        final_summary = {
            "signed_W_um": carried_W_um,
            "abs_W_um": float(abs(carried_W_um)),
            "kappa_1_per_m": float(bow_um_to_curvature(carried_W_um, L_m)),
            "DeltaT_K": 0.0,
            "L_m": float(L_m),
        }
    return final_summary, step_summaries


def _cfg_get(cfg, key, default=None):
    if hasattr(cfg, "get"):
        value = cfg.get(key, default)
    else:
        value = getattr(cfg, key, default)
    if value in (None, "None"):
        return default
    return value


def _cfg_float(cfg, key, default=None):
    value = _cfg_get(cfg, key, default)
    if value in (None, "None"):
        if default is None:
            raise ValueError(f"Missing required config value: {key}")
        value = default
    return float(value)


def _first_existing_float(cfg, keys, default=None):
    for key in keys:
        value = _cfg_get(cfg, key, None)
        if value not in (None, "None"):
            return float(value)
    if default is None:
        raise ValueError(f"None of these config values are available: {keys}")
    return float(default)


def _initial_bow_stats_um(cfg, side):
    side = side.upper()
    if side not in ("TOP", "BOT"):
        raise ValueError(f"Unknown initial bow side: {side}")

    mean_value = _cfg_get(cfg, f"{side}_INI_BOW_MEAN_um", None)
    if mean_value not in (None, "None"):
        mean = float(mean_value)
    else:
        legacy_s_init_key = "S_INIT_B_M" if side == "TOP" else "S_INIT_A_M"
        legacy_s_init_m = _cfg_get(cfg, legacy_s_init_key, None)
        if legacy_s_init_m not in (None, "None"):
            mean = float(legacy_s_init_m) * 1e6
        else:
            mean = _first_existing_float(
                cfg,
                [
                    "BOW_DIFFERENCE_MEAN_um",
                ],
                default=0.0,
            )
    std = _first_existing_float(
        cfg,
        [
            f"{side}_INI_BOW_STD_um",
            "BOW_DIFFERENCE_STD_um",
        ],
        default=0.0,
    )
    return mean, std


def _delta_t_from_cfg(cfg):
    if _cfg_get(cfg, "T_R", None) is None or _cfg_get(cfg, "T_anl", None) is None:
        return 0.0
    return _cfg_float(cfg, "T_R") - _cfg_float(cfg, "T_anl")


def _wafer_half_length_m(cfg):
    return _cfg_float(cfg, "WAF_R_um") * 1e-6


def _layer_cfg_and_side(cfg_dict, interfaces_bottom_to_top, layer_index):
    if layer_index == 0:
        if not interfaces_bottom_to_top:
            raise ValueError("A root-only substack does not have an interface cfg.")
        return cfg_dict[interfaces_bottom_to_top[0]], "BOT"
    return cfg_dict[interfaces_bottom_to_top[layer_index - 1]], "TOP"


def _effective_material_properties_from_volume_fraction(cfg, mix_prefix):
    material_defaults = {
        "Cu": {
            "E_GPa": _first_existing_float(cfg, ["CU_E_GPA"], default=91.8),
            "nu": _first_existing_float(cfg, ["CU_NU"], default=0.34),
            "alpha_ppm_K": _first_existing_float(cfg, ["CU_ALPHA_PPM"], default=17.6),
        },
        "Sio2": {
            "E_GPa": _first_existing_float(cfg, ["OX_E_GPA"], default=73.0),
            "nu": _first_existing_float(cfg, ["OX_NU"], default=0.17),
            "alpha_ppm_K": _first_existing_float(cfg, ["OX_ALPHA_PPM"], default=0.5),
        },
        "Si": {
            "E_GPa": _first_existing_float(cfg, ["SI_E_GPA"], default=131.0),
            "nu": _first_existing_float(cfg, ["SI_NU"], default=0.28),
            "alpha_ppm_K": _first_existing_float(cfg, ["SI_ALPHA_PPM"], default=2.6),
        },
    }
    volume_keys = {
        "Cu": f"{mix_prefix}_Cu_V",
        "Sio2": f"{mix_prefix}_Sio2_V",
        "Si": f"{mix_prefix}_Si_V",
    }
    has_any_volume = any(_cfg_get(cfg, key, None) is not None for key in volume_keys.values())

    if has_any_volume:
        volumes = {
            material: _first_existing_float(cfg, [key], default=0.0)
            for material, key in volume_keys.items()
        }
        source = "config_volume_fraction"
    else:
        volumes = {"Cu": 0.0, "Sio2": 0.0, "Si": 1.0}
        source = "placeholder_pure_si"

    if any(value < 0.0 for value in volumes.values()):
        raise ValueError(f"{mix_prefix} volume fractions must be non-negative: {volumes}")

    total_volume = sum(volumes.values())
    if total_volume <= 0.0:
        volumes = {"Cu": 0.0, "Sio2": 0.0, "Si": 1.0}
        total_volume = 1.0
        source = "placeholder_pure_si"

    weights = {
        material: volume / total_volume
        for material, volume in volumes.items()
    }
    E_GPa = sum(material_defaults[material]["E_GPa"] * weights[material] for material in weights)
    nu = sum(material_defaults[material]["nu"] * weights[material] for material in weights)
    alpha_ppm_K = sum(
        material_defaults[material]["alpha_ppm_K"] * weights[material]
        for material in weights
    )

    return {
        "E_GPa": float(E_GPa),
        "nu": float(nu),
        "alpha_ppm_K": float(alpha_ppm_K),
        "Cu_V": float(volumes["Cu"]),
        "Sio2_V": float(volumes["Sio2"]),
        "Si_V": float(volumes["Si"]),
        "volume_sum": float(total_volume),
        "material_property_source": source,
    }


def _build_layer_df(
    cfg_dict,
    chiplets_bottom_to_top,
    interfaces_bottom_to_top,
    chiplet_instances_bottom_to_top=None,
):
    rows = []
    sigmas = []
    if chiplet_instances_bottom_to_top is None:
        chiplet_instances_bottom_to_top = chiplets_bottom_to_top

    for layer_index, chiplet in enumerate(chiplets_bottom_to_top):
        cfg, side = _layer_cfg_and_side(cfg_dict, interfaces_bottom_to_top, layer_index)
        h_key = "ITF_BOT_THICK_um" if side == "BOT" else "ITF_TOP_THICK_um"
        mix_prefix = "B_Sub" if side == "BOT" else "T_Sub"
        bow_mean_um, bow_std_um = _initial_bow_stats_um(cfg, side)
        effective_material = _effective_material_properties_from_volume_fraction(cfg, mix_prefix)

        rows.append({
            "chiplet": chiplet,
            "chiplet_instance": chiplet_instances_bottom_to_top[layer_index],
            "side": side.lower(),
            "material_mix": mix_prefix,
            "h_um": _cfg_float(cfg, h_key),
            "E_GPa": effective_material["E_GPa"],
            "nu": effective_material["nu"],
            "alpha_ppm_K": effective_material["alpha_ppm_K"],
            "W0_um": bow_mean_um,
            "Cu_V": effective_material["Cu_V"],
            "Sio2_V": effective_material["Sio2_V"],
            "Si_V": effective_material["Si_V"],
            "volume_sum": effective_material["volume_sum"],
            "material_property_source": effective_material["material_property_source"],
        })
        sigmas.append(bow_std_um)

    return pd.DataFrame(rows), np.asarray(sigmas, dtype=float)


def _sequential_delta_t_by_step(cfg_dict, interfaces_bottom_to_top):
    return np.asarray([
        _delta_t_from_cfg(cfg_dict[interface])
        for interface in interfaces_bottom_to_top
    ], dtype=float)


def _sequential_step_warpage_gaussians(layer_df, layer_bow_sigma_um, DeltaT_K_by_step, L_m):
    final_summary, base_steps = compute_sequential_stack_warpage(
        layer_df,
        DeltaT_K_by_step=DeltaT_K_by_step,
        L_m=L_m,
    )
    if not base_steps:
        mu_um = float(final_summary["signed_W_um"])
        sigma_um = float(layer_bow_sigma_um[0]) if len(layer_bow_sigma_um) else 0.0
        return final_summary, [], mu_um, sigma_um, np.asarray([1.0])

    base_values = np.asarray([step["signed_W_um"] for step in base_steps], dtype=float)
    base_w0 = layer_df["W0_um"].to_numpy(dtype=float)
    sensitivity_matrix = np.zeros((len(base_steps), len(layer_df)), dtype=float)

    for layer_index in range(len(layer_df)):
        perturbed = layer_df.copy()
        perturbed_w0 = base_w0.copy()
        perturbed_w0[layer_index] += 1.0
        perturbed["W0_um"] = perturbed_w0
        _, perturbed_steps = compute_sequential_stack_warpage(
            perturbed,
            DeltaT_K_by_step=DeltaT_K_by_step,
            L_m=L_m,
        )
        perturbed_values = np.asarray(
            [step["signed_W_um"] for step in perturbed_steps],
            dtype=float,
        )
        sensitivity_matrix[:, layer_index] = perturbed_values - base_values

    step_sigmas = np.sqrt(np.sum((sensitivity_matrix * layer_bow_sigma_um) ** 2, axis=1))
    for step_index, step in enumerate(base_steps):
        step["mu_um"] = float(base_values[step_index])
        step["sigma_um"] = float(step_sigmas[step_index])
        step["initial_bow_sensitivity"] = sensitivity_matrix[step_index].tolist()

    final_mu_um = float(base_values[-1])
    final_sigma_um = float(step_sigmas[-1])
    final_sensitivity = sensitivity_matrix[-1]
    return final_summary, base_steps, final_mu_um, final_sigma_um, final_sensitivity


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _gaussian_abs_below_probability(mu, sigma, threshold):
    if threshold < 0.0:
        raise ValueError("Warpage threshold must be non-negative.")
    if sigma < 0.0:
        raise ValueError("Gaussian sigma must be non-negative.")
    if sigma == 0.0:
        return 1.0 if abs(mu) <= threshold else 0.0

    lower = (-threshold - mu) / sigma
    upper = (threshold - mu) / sigma
    probability = _normal_cdf(upper) - _normal_cdf(lower)
    return float(np.clip(probability, 0.0, 1.0))


def _stack_warpage_threshold_um(cfg):
    return _first_existing_float(
        cfg,
        [
            "STACK_WARPAGE_TH",
            "WARPAGE_LIMIT_UM",
        ],
        default=20.0,
    )


def get_interface_existing_stack_warpage_map(cfg_dict, _3dbx_path):
    """
    Return pre-bond existing-stack warpage distributions for W2W overlay.

    For the first interface, the existing stack is just the bottom wafer
    initial bow. For later interfaces, it is the post-anneal bow after the
    previous bonding step.
    """
    interface_stack_warpage = {}
    for substack in stack_graph_from_3dbx(_3dbx_path):
        substack_id = int(substack["substack_id"])
        chiplets = list(substack["chiplets_bottom_to_top"])
        chiplet_instances = list(substack["chiplet_instances_bottom_to_top"])
        interfaces = list(substack["interfaces_bottom_to_top"])
        if not interfaces:
            continue

        final_cfg = cfg_dict[interfaces[-1]]
        L_m = _wafer_half_length_m(final_cfg)
        layer_df, layer_bow_sigma_um = _build_layer_df(
            cfg_dict,
            chiplets,
            interfaces,
            chiplet_instances,
        )
        DeltaT_K_by_step = _sequential_delta_t_by_step(cfg_dict, interfaces)
        _, step_results, _, _, _ = _sequential_step_warpage_gaussians(
            layer_df,
            layer_bow_sigma_um,
            DeltaT_K_by_step,
            L_m,
        )

        for interface_index, interface in enumerate(interfaces):
            current_cfg = cfg_dict[interface]
            incoming_mu_um, incoming_sigma_um = _initial_bow_stats_um(current_cfg, "TOP")
            if interface_index == 0:
                existing_mu_um = float(layer_df.loc[0, "W0_um"])
                existing_sigma_um = float(layer_bow_sigma_um[0])
                last_completed_interface = None
                step_info = None
            else:
                step_info = step_results[interface_index - 1]
                existing_mu_um = float(step_info["mu_um"])
                existing_sigma_um = float(step_info["sigma_um"])
                last_completed_interface = interfaces[interface_index - 1]

            interface_stack_warpage[interface] = {
                "substack_id": substack_id,
                "interface_index": int(interface_index),
                "chiplets_before_bond": chiplets[:interface_index + 1],
                "chiplet_instances_before_bond": chiplet_instances[:interface_index + 1],
                "incoming_top_chiplet": chiplets[interface_index + 1],
                "incoming_top_chiplet_instance": chiplet_instances[interface_index + 1],
                "interfaces_before_bond": interfaces[:interface_index],
                "last_completed_interface": last_completed_interface,
                "mu_um": existing_mu_um,
                "sigma_um": existing_sigma_um,
                "top_wafer_mu_um": float(incoming_mu_um),
                "top_wafer_sigma_um": float(incoming_sigma_um),
                "L_m": float(L_m),
                "post_anneal_step": step_info,
                "anneal_mode": "sequential",
            }

    return interface_stack_warpage


def compute_warpage_yield(cfg_dict, _3dbx_path):
    """
    Compute final W2W total-warpage yield for every substack.
    """
    substack_results = {}
    overall_warpage_yield = 1.0

    for substack in stack_graph_from_3dbx(_3dbx_path):
        substack_id = int(substack["substack_id"])
        chiplets = list(substack["chiplets_bottom_to_top"])
        chiplet_instances = list(substack["chiplet_instances_bottom_to_top"])
        interfaces = list(substack["interfaces_bottom_to_top"])
        if not interfaces:
            raise ValueError(
                f"Substack {substack_id} has no bonding interfaces; "
                "cannot infer final stack warpage cfg."
            )

        final_interface = interfaces[-1]
        if final_interface not in cfg_dict:
            raise KeyError(f"Interface '{final_interface}' from {_3dbx_path} is missing in cfg_dict.")

        final_cfg = cfg_dict[final_interface]
        threshold_um = _stack_warpage_threshold_um(final_cfg)
        L_m = _wafer_half_length_m(final_cfg)
        DeltaT_K_by_step = _sequential_delta_t_by_step(cfg_dict, interfaces)
        layer_df, layer_bow_sigma_um = _build_layer_df(
            cfg_dict,
            chiplets,
            interfaces,
            chiplet_instances,
        )

        final_summary, step_results, mu_um, sigma_um, sensitivities = (
            _sequential_step_warpage_gaussians(
                layer_df,
                layer_bow_sigma_um,
                DeltaT_K_by_step,
                L_m,
            )
        )
        yield_value = _gaussian_abs_below_probability(
            mu=mu_um,
            sigma=sigma_um,
            threshold=threshold_um,
        )
        overall_warpage_yield *= yield_value

        substack_results[substack_id] = {
            "substack_id": substack_id,
            "chiplets_bottom_to_top": chiplets,
            "chiplet_instances_bottom_to_top": chiplet_instances,
            "interfaces_bottom_to_top": interfaces,
            "final_interface": final_interface,
            "mu_um": float(mu_um),
            "sigma_um": float(sigma_um),
            "threshold_um": float(threshold_um),
            "warpage_yield": float(yield_value),
            "pass_condition": "abs(W_final_um) <= threshold_um",
            "L_m": float(L_m),
            "DeltaT_K_by_step": DeltaT_K_by_step.tolist(),
            "anneal_mode": "sequential",
            "final_summary": final_summary,
            "step_results": step_results,
            "layers": layer_df.to_dict(orient="records"),
            "initial_bow_sensitivity": sensitivities.tolist(),
        }

    return {
        "overall_warpage_yield": float(overall_warpage_yield),
        "substack_results": substack_results,
    }


def stack_warpage_yield_calculator(
    cfg_dict: dict,
    waf_stack,
    _3dbx_path: str,
):
    """
    Calculate final substack warpage yield and write it into ``waf_stack``.
    """
    ones = np.ones(waf_stack.num_dies_per_wafer, dtype=float)
    for interface_name in cfg_dict:
        waf_stack.die_yield_list_per_interface_dict[interface_name]["warpage"] = ones.copy()

    result = compute_warpage_yield(cfg_dict, _3dbx_path)
    for info in result["substack_results"].values():
        final_interface = info["final_interface"]
        waf_stack.die_yield_list_per_interface_dict[final_interface]["warpage"] = np.full(
            waf_stack.num_dies_per_wafer,
            float(info["warpage_yield"]),
            dtype=float,
        )

    return result
