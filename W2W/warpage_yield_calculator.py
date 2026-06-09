#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

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

try:
    from utils.util import w2w_area_scaled_layer_volumes
except (ModuleNotFoundError, ImportError):
    from W2W.utils.util import w2w_area_scaled_layer_volumes


# The current W2W MAPDL verification compacts the completed lower stack into
# its released spherical bow before the next bonding/anneal step.  Under that
# compacted-state assumption, carrying an additional scalar residual-curvature
# memory double-counts history.  Keep the parameter available for future
# stress-history models, but default the production recurrence to bow transfer
# only.
DEFAULT_RESIDUAL_CURVATURE_CARRY = 0.0
DEFAULT_RESIDUAL_CURVATURE_MEMORY_DECAY = 0.50
DEFAULT_THERMAL_RESIDUAL_MOMENT_CARRY = 0.90
DEFAULT_THERMAL_RESIDUAL_MOMENT_MEMORY_DECAY = 0.90
DEFAULT_THERMAL_RESIDUAL_MOMENT_TERMINAL_POWER = 2.00


def _w2w_incremental_thermal_delta_t(delta_t):
    """
    Convert a nominal cooldown DeltaT into the incremental W2W thermal step.

    `compute_total_stack_warpage` is a single bonding/release formula whose
    DeltaT convention matches a batch stack assembled stress-free at anneal and
    then cooled to room.  In W2W, however, the already completed lower stack is
    carried from its released room-temperature state into the next anneal before
    cooling again.  The incremental thermal mismatch seen by that carried state
    therefore has the opposite sign from the batch cooldown input.
    """
    return -float(delta_t)


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


def compute_total_stack_warpage(layer_df, DeltaT_K, L_m, thermal_layer_df=None):
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

    if thermal_layer_df is None:
        thermal_df = df
    else:
        thermal_df = thermal_layer_df.copy().reset_index(drop=True)
        missing_thermal = [col for col in required if col not in thermal_df.columns]
        if missing_thermal:
            raise ValueError(f"Missing required thermal-layer columns: {missing_thermal}")

    h_t = thermal_df["h_um"].to_numpy(dtype=float) * 1e-6
    E_t = thermal_df["E_GPa"].to_numpy(dtype=float) * 1e9
    nu_t = thermal_df["nu"].to_numpy(dtype=float)
    alpha_t = thermal_df["alpha_ppm_K"].to_numpy(dtype=float) * 1e-6
    if np.any(h_t <= 0):
        raise ValueError("All thermal layer thicknesses must be positive.")
    if np.any((nu_t <= -1.0) | (nu_t >= 0.5)):
        raise ValueError("Thermal-layer Poisson ratios should be in the elastic range (-1, 0.5).")

    # Axial compatibility follows the Suhir/Kim E/(1-nu) convention.  The
    # thermal bending moment is a plate response, so use the plane-stress plate
    # modulus E/(1-nu^2) for the CTE-mismatch weights.
    E_thermal = E_t / (1.0 - nu_t**2)
    thermal_weights = E_thermal * h_t
    h_t_cumsum = np.cumsum(h_t)
    a_t = h_t_cumsum - 0.5 * (h_t[0] + h_t)
    delta_alpha_t = alpha_t - alpha_t[0]
    s_alpha = np.sum(delta_alpha_t * thermal_weights) / np.sum(thermal_weights)
    s_a = np.sum(a * weights) / np.sum(weights)
    s_b = np.sum(b * weights) / np.sum(weights)

    D = E_biaxial * h**3 / 12.0
    M_T = np.sum(a_t * thermal_weights * (delta_alpha_t - s_alpha) * DeltaT_K)
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
    layer_out["E_thermal_plate_GPa"] = E / (1.0 - nu**2) / 1e9
    layer_out["kappa0_1_per_m"] = kappa0
    layer_out["F_N_per_m"] = F
    layer_out["sigma_MPa"] = sigma / 1e6
    return summary, layer_out


def _compute_total_stack_warpage_arrays(
    h_um,
    E_GPa,
    nu,
    alpha_ppm_K,
    W0_um,
    DeltaT_K,
    L_m,
    thermal_arrays=None,
):
    """Array implementation of compute_total_stack_warpage for fast sweeps."""
    h = np.asarray(h_um, dtype=float) * 1e-6
    E = np.asarray(E_GPa, dtype=float) * 1e9
    nu = np.asarray(nu, dtype=float)
    alpha = np.asarray(alpha_ppm_K, dtype=float) * 1e-6
    kappa0 = bow_um_to_curvature(np.asarray(W0_um, dtype=float), L_m)

    E_biaxial = E / (1.0 - nu)
    weights = E_biaxial * h
    h_cumsum = np.cumsum(h)
    hk0_cumsum = np.cumsum(h * kappa0)
    a = h_cumsum - 0.5 * (h[0] + h)
    b = hk0_cumsum - 0.5 * (h[0] * kappa0[0] + h * kappa0)

    if thermal_arrays is None:
        h_t = h
        E_t = E
        nu_t = nu
        alpha_t = alpha
    else:
        h_t = np.asarray(thermal_arrays["h_um"], dtype=float) * 1e-6
        E_t = np.asarray(thermal_arrays["E_GPa"], dtype=float) * 1e9
        nu_t = np.asarray(thermal_arrays["nu"], dtype=float)
        alpha_t = np.asarray(thermal_arrays["alpha_ppm_K"], dtype=float) * 1e-6

    E_thermal = E_t / (1.0 - nu_t**2)
    thermal_weights = E_thermal * h_t
    h_t_cumsum = np.cumsum(h_t)
    a_t = h_t_cumsum - 0.5 * (h_t[0] + h_t)
    delta_alpha_t = alpha_t - alpha_t[0]
    s_alpha = np.sum(delta_alpha_t * thermal_weights) / np.sum(thermal_weights)
    s_a = np.sum(a * weights) / np.sum(weights)
    s_b = np.sum(b * weights) / np.sum(weights)

    D = E_biaxial * h**3 / 12.0
    M_T = np.sum(a_t * thermal_weights * (delta_alpha_t - s_alpha) * DeltaT_K)
    M_b = np.sum(a * weights * (b - s_b))
    K_a = np.sum(a * weights * (a - s_a))
    K_D = np.sum(D)
    M_D0 = np.sum(D * kappa0)
    kappa = (M_D0 + M_T + M_b) / (K_D + K_a)
    W_um = curvature_to_bow_um(kappa, L_m)
    return {
        "kappa_1_per_m": float(kappa),
        "signed_W_um": float(W_um),
        "abs_W_um": float(abs(W_um)),
        "DeltaT_K": float(DeltaT_K),
        "L_m": float(L_m),
    }


def _compute_total_stack_warpage_arrays_with_residual_strain(
    h_um,
    E_GPa,
    nu,
    alpha_ppm_K,
    W0_um,
    residual_strain,
    DeltaT_K,
    L_m,
    thermal_arrays=None,
):
    """
    Array release solve with a transferred per-layer residual strain state.

    This is an analytical diagnostic for W2W stress-history transfer.  The
    ordinary compacted-state model carries only released bow.  Here, completed
    lower layers additionally carry their previous residual stress strain
    epsilon_res = sigma / C into the next release solve as an initial strain
    field.  No MAPDL result is read by this function.
    """
    h = np.asarray(h_um, dtype=float) * 1e-6
    E = np.asarray(E_GPa, dtype=float) * 1e9
    nu = np.asarray(nu, dtype=float)
    alpha = np.asarray(alpha_ppm_K, dtype=float) * 1e-6
    kappa0 = bow_um_to_curvature(np.asarray(W0_um, dtype=float), L_m)
    residual = np.asarray(residual_strain, dtype=float)
    if residual.size != h.size:
        raise ValueError("residual_strain must have one value per mechanical layer.")

    E_biaxial = E / (1.0 - nu)
    weights = E_biaxial * h
    h_cumsum = np.cumsum(h)
    hk0_cumsum = np.cumsum(h * kappa0)
    a = h_cumsum - 0.5 * (h[0] + h)
    b = hk0_cumsum - 0.5 * (h[0] * kappa0[0] + h * kappa0)

    if thermal_arrays is None:
        h_t = h
        E_t = E
        nu_t = nu
        alpha_t = alpha
    else:
        h_t = np.asarray(thermal_arrays["h_um"], dtype=float) * 1e-6
        E_t = np.asarray(thermal_arrays["E_GPa"], dtype=float) * 1e9
        nu_t = np.asarray(thermal_arrays["nu"], dtype=float)
        alpha_t = np.asarray(thermal_arrays["alpha_ppm_K"], dtype=float) * 1e-6

    E_thermal = E_t / (1.0 - nu_t**2)
    thermal_weights = E_thermal * h_t
    h_t_cumsum = np.cumsum(h_t)
    a_t = h_t_cumsum - 0.5 * (h_t[0] + h_t)
    delta_alpha_t = alpha_t - alpha_t[0]
    s_alpha = np.sum(delta_alpha_t * thermal_weights) / np.sum(thermal_weights)
    s_a = np.sum(a * weights) / np.sum(weights)
    s_b = np.sum(b * weights) / np.sum(weights)
    s_residual = np.sum(residual * weights) / np.sum(weights)

    D = E_biaxial * h**3 / 12.0
    M_T = np.sum(a_t * thermal_weights * (delta_alpha_t - s_alpha) * DeltaT_K)
    M_b = np.sum(a * weights * (b - s_b))
    M_residual = np.sum(a * weights * (residual - s_residual))
    K_a = np.sum(a * weights * (a - s_a))
    K_D = np.sum(D)
    M_D0 = np.sum(D * kappa0)
    denominator = K_D + K_a
    kappa = (M_D0 + M_T + M_b + M_residual) / denominator
    W_um = curvature_to_bow_um(kappa, L_m)

    delta_alpha = alpha - alpha[0]
    strain_term = (
        (delta_alpha - s_alpha) * DeltaT_K
        - (a - s_a) * kappa
        + (b - s_b)
        + (residual - s_residual)
    )

    return {
        "kappa_1_per_m": float(kappa),
        "signed_W_um": float(W_um),
        "abs_W_um": float(abs(W_um)),
        "DeltaT_K": float(DeltaT_K),
        "L_m": float(L_m),
        "residual_generalized_moment_N": float(M_residual),
        "residual_mean_strain": float(s_residual),
        "residual_rms_strain": float(np.sqrt(np.mean(residual**2))),
        "denominator_N_m": float(denominator),
    }, strain_term


def _equivalent_thermal_layer(layer_df, carried_W_um):
    """
    Collapse an already bonded lower substack for the next W2W thermal step.

    A completed lower stack should not have all of its internal CTE mismatch
    reapplied as a new thermal load at every subsequent bonding step.  For the
    incremental thermal term, represent that lower stack by one equivalent
    layer; the detailed layers are still kept in the mechanical/initial-bow
    part of the calculation.
    """
    df = layer_df.copy().reset_index(drop=True)
    h = df["h_um"].to_numpy(dtype=float)
    E = df["E_GPa"].to_numpy(dtype=float)
    nu = df["nu"].to_numpy(dtype=float)
    alpha = df["alpha_ppm_K"].to_numpy(dtype=float)
    if np.any(h <= 0.0):
        raise ValueError("All layers must have positive thickness.")

    h_total = float(np.sum(h))
    E_axial = E / (1.0 - nu)
    E_thermal = E / (1.0 - nu**2)
    axial_weights = E_axial * h
    thermal_weights = E_thermal * h
    nu_eff = float(np.sum(nu * h) / h_total)
    E_axial_eff = float(np.sum(axial_weights) / h_total)
    E_eff = E_axial_eff * (1.0 - nu_eff)
    alpha_eff = float(np.sum(alpha * thermal_weights) / np.sum(thermal_weights))

    return pd.DataFrame([
        {
            "h_um": h_total,
            "E_GPa": E_eff,
            "nu": nu_eff,
            "alpha_ppm_K": alpha_eff,
            "W0_um": float(carried_W_um),
        }
    ])


def _sequential_thermal_layers(layer_df, incoming_layer_index, carried_W_um):
    """
    Build the thermal-mismatch state for one W2W bonding/anneal step.

    The already completed lower stack has seen previous anneals, so its
    internal CTE mismatch should not be re-applied as a fresh thermal load at
    every later bonding step.  Represent the completed lower stack by one
    equivalent thermal layer and keep the incoming wafer explicit for the new
    bonding/anneal event.
    """
    df = layer_df.copy().reset_index(drop=True)
    if incoming_layer_index < 1 or incoming_layer_index >= len(df):
        raise ValueError("incoming_layer_index must identify an appended layer.")

    pieces = []
    completed_lower_stack = df.iloc[:incoming_layer_index].copy()
    pieces.append(_equivalent_thermal_layer(completed_lower_stack, carried_W_um))
    incoming = df.iloc[[incoming_layer_index]].copy().reset_index(drop=True)
    pieces.append(incoming)

    return pd.concat(pieces, ignore_index=True)


def _sequential_thermal_group_layers(layer_df, group_ids, incoming_group, carried_W_um):
    """
    Thermal state builder for experiments that append groups of physical layers.

    Exp03 represents one wafer as separate Si and hybrid-interface sublayers.
    For a W2W step, all completed groups are collapsed to one equivalent thermal
    layer, while the incoming group remains explicit.
    """
    df = layer_df.copy().reset_index(drop=True)
    group_values = pd.Series(group_ids).reset_index(drop=True)
    completed_lower_stack = df.loc[group_values != incoming_group].copy().reset_index(drop=True)
    incoming = df.loc[group_values == incoming_group].copy().reset_index(drop=True)
    if completed_lower_stack.empty:
        raise ValueError("incoming_group must have at least one completed lower group.")
    if incoming.empty:
        raise ValueError("incoming_group must identify at least one incoming layer.")

    return pd.concat(
        [_equivalent_thermal_layer(completed_lower_stack, carried_W_um), incoming],
        ignore_index=True,
    )


def _equivalent_thermal_arrays(h_um, E_GPa, nu, alpha_ppm_K):
    h = np.asarray(h_um, dtype=float)
    E = np.asarray(E_GPa, dtype=float)
    nu = np.asarray(nu, dtype=float)
    alpha = np.asarray(alpha_ppm_K, dtype=float)
    h_total = float(np.sum(h))
    E_axial = E / (1.0 - nu)
    E_thermal = E / (1.0 - nu**2)
    axial_weights = E_axial * h
    thermal_weights = E_thermal * h
    nu_eff = float(np.sum(nu * h) / h_total)
    E_axial_eff = float(np.sum(axial_weights) / h_total)
    return {
        "h_um": h_total,
        "E_GPa": E_axial_eff * (1.0 - nu_eff),
        "nu": nu_eff,
        "alpha_ppm_K": float(np.sum(alpha * thermal_weights) / np.sum(thermal_weights)),
    }


def compute_sequential_stack_warpage_grouped_fast(
    layer_df,
    group_ids,
    DeltaT_K_by_step,
    L_m,
    residual_curvature_carry=DEFAULT_RESIDUAL_CURVATURE_CARRY,
    residual_curvature_memory_decay=DEFAULT_RESIDUAL_CURVATURE_MEMORY_DECAY,
):
    """
    Fast compacted-state W2W recurrence for grouped physical layers.

    This is equivalent to `compute_sequential_stack_warpage_abd_residual` with
    thermal residual correction disabled, but it avoids DataFrame construction
    inside every step.  It is intended for analytical sweeps and Monte Carlo
    where a wafer is represented by multiple physical sublayers.
    """
    df = layer_df.copy().reset_index(drop=True)
    group_values = pd.Series(group_ids).reset_index(drop=True)
    if len(group_values) != len(df):
        raise ValueError("group_ids must have the same length as layer_df.")

    h_all = df["h_um"].to_numpy(dtype=float)
    E_all = df["E_GPa"].to_numpy(dtype=float)
    nu_all = df["nu"].to_numpy(dtype=float)
    alpha_all = df["alpha_ppm_K"].to_numpy(dtype=float)
    W0_all = df["W0_um"].to_numpy(dtype=float)
    groups = list(pd.unique(group_values))
    num_groups = len(groups)
    if num_groups < 1:
        raise ValueError("layer_df must contain at least one layer group.")

    delta_t_values = np.asarray(DeltaT_K_by_step, dtype=float)
    if delta_t_values.size == 1 and num_groups > 1:
        delta_t_values = np.full(num_groups - 1, float(delta_t_values[0]))
    if delta_t_values.size != max(num_groups - 1, 0):
        raise ValueError(
            f"DeltaT_K_by_step must have {num_groups - 1} entries for "
            f"{num_groups} layer groups; got {delta_t_values.size}."
        )

    group_masks = [group_values.to_numpy() == group for group in groups]
    carried_W_um = float(W0_all[group_masks[0]][0])
    previous_W_um = carried_W_um
    residual_state_W_um = 0.0
    carry_gain = float(residual_curvature_carry)
    memory_decay = float(residual_curvature_memory_decay)
    step_summaries = []

    for group_position in range(1, num_groups):
        residual_delta_W_um = carried_W_um - previous_W_um
        residual_state_W_um = memory_decay * residual_state_W_um + residual_delta_W_um
        effective_carried_W_um = carried_W_um + carry_gain * residual_state_W_um

        active_mask = np.logical_or.reduce(group_masks[: group_position + 1])
        lower_mask = np.logical_or.reduce(group_masks[:group_position])
        incoming_mask = group_masks[group_position]
        active_indices = np.nonzero(active_mask)[0]
        lower_indices = np.nonzero(lower_mask)[0]
        incoming_indices = np.nonzero(incoming_mask)[0]

        W0_current = W0_all[active_indices].copy()
        active_lower = np.isin(active_indices, lower_indices)
        W0_current[active_lower] = effective_carried_W_um

        lower_eq = _equivalent_thermal_arrays(
            h_all[lower_indices],
            E_all[lower_indices],
            nu_all[lower_indices],
            alpha_all[lower_indices],
        )
        thermal_arrays = {
            "h_um": np.concatenate(([lower_eq["h_um"]], h_all[incoming_indices])),
            "E_GPa": np.concatenate(([lower_eq["E_GPa"]], E_all[incoming_indices])),
            "nu": np.concatenate(([lower_eq["nu"]], nu_all[incoming_indices])),
            "alpha_ppm_K": np.concatenate(([lower_eq["alpha_ppm_K"]], alpha_all[incoming_indices])),
        }

        nominal_delta_t = float(delta_t_values[group_position - 1])
        step_delta_t = _w2w_incremental_thermal_delta_t(nominal_delta_t)
        summary = _compute_total_stack_warpage_arrays(
            h_all[active_indices],
            E_all[active_indices],
            nu_all[active_indices],
            alpha_all[active_indices],
            W0_current,
            DeltaT_K=step_delta_t,
            L_m=L_m,
            thermal_arrays=thermal_arrays,
        )

        previous_W_um = carried_W_um
        carried_W_um = float(summary["signed_W_um"])
        step_summaries.append({
            "interface_index": group_position - 1,
            "layer_count": group_position + 1,
            "group_id": groups[group_position],
            "signed_W_um": carried_W_um,
            "abs_W_um": float(abs(carried_W_um)),
            "kappa_1_per_m": float(summary["kappa_1_per_m"]),
            "DeltaT_K": float(summary["DeltaT_K"]),
            "nominal_cooldown_DeltaT_K": nominal_delta_t,
            "incremental_thermal_DeltaT_K": step_delta_t,
            "L_m": float(summary["L_m"]),
            "effective_carried_W_um": float(effective_carried_W_um),
            "residual_delta_W_um": float(residual_delta_W_um),
            "residual_state_W_um": float(residual_state_W_um),
            "residual_curvature_carry": carry_gain,
            "residual_curvature_memory_decay": memory_decay,
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
        kappa = float(bow_um_to_curvature(carried_W_um, L_m))
        final_summary = {
            "signed_W_um": carried_W_um,
            "abs_W_um": float(abs(carried_W_um)),
            "kappa_1_per_m": kappa,
            "DeltaT_K": 0.0,
            "L_m": float(L_m),
        }
    return final_summary, step_summaries


def compute_sequential_stack_warpage_grouped_abd_state_transfer(
    layer_df,
    group_ids,
    DeltaT_K_by_step,
    L_m,
):
    """
    W2W recurrence with an analytical ABD-style residual strain state.

    This diagnostic carries two states from each completed partial stack into
    the next bonding/anneal step:

    - released spherical bow, as in the production compacted-state model
    - per-layer residual stress strain, epsilon_res = sigma / (E/(1-nu))

    The residual strain state contributes an additional generalized bending
    moment in the next analytical release solve.  It is derived entirely from
    previous analytical layer stresses and does not read MAPDL results.
    """
    df = layer_df.copy().reset_index(drop=True)
    group_values = pd.Series(group_ids).reset_index(drop=True)
    if len(group_values) != len(df):
        raise ValueError("group_ids must have the same length as layer_df.")

    h_all = df["h_um"].to_numpy(dtype=float)
    E_all = df["E_GPa"].to_numpy(dtype=float)
    nu_all = df["nu"].to_numpy(dtype=float)
    alpha_all = df["alpha_ppm_K"].to_numpy(dtype=float)
    W0_all = df["W0_um"].to_numpy(dtype=float)
    groups = list(pd.unique(group_values))
    num_groups = len(groups)
    if num_groups < 1:
        raise ValueError("layer_df must contain at least one layer group.")

    delta_t_values = np.asarray(DeltaT_K_by_step, dtype=float)
    if delta_t_values.size == 1 and num_groups > 1:
        delta_t_values = np.full(num_groups - 1, float(delta_t_values[0]))
    if delta_t_values.size != max(num_groups - 1, 0):
        raise ValueError(
            f"DeltaT_K_by_step must have {num_groups - 1} entries for "
            f"{num_groups} layer groups; got {delta_t_values.size}."
        )

    group_masks = [group_values.to_numpy() == group for group in groups]
    carried_W_um = float(W0_all[group_masks[0]][0])
    residual_strain_all = np.zeros(len(df), dtype=float)
    step_summaries = []

    for group_position in range(1, num_groups):
        active_mask = np.logical_or.reduce(group_masks[: group_position + 1])
        lower_mask = np.logical_or.reduce(group_masks[:group_position])
        incoming_mask = group_masks[group_position]
        active_indices = np.nonzero(active_mask)[0]
        lower_indices = np.nonzero(lower_mask)[0]
        incoming_indices = np.nonzero(incoming_mask)[0]

        W0_current = W0_all[active_indices].copy()
        active_lower = np.isin(active_indices, lower_indices)
        W0_current[active_lower] = carried_W_um

        lower_eq = _equivalent_thermal_arrays(
            h_all[lower_indices],
            E_all[lower_indices],
            nu_all[lower_indices],
            alpha_all[lower_indices],
        )
        thermal_arrays = {
            "h_um": np.concatenate(([lower_eq["h_um"]], h_all[incoming_indices])),
            "E_GPa": np.concatenate(([lower_eq["E_GPa"]], E_all[incoming_indices])),
            "nu": np.concatenate(([lower_eq["nu"]], nu_all[incoming_indices])),
            "alpha_ppm_K": np.concatenate(([lower_eq["alpha_ppm_K"]], alpha_all[incoming_indices])),
        }

        nominal_delta_t = float(delta_t_values[group_position - 1])
        step_delta_t = _w2w_incremental_thermal_delta_t(nominal_delta_t)
        summary, residual_strain_current = _compute_total_stack_warpage_arrays_with_residual_strain(
            h_all[active_indices],
            E_all[active_indices],
            nu_all[active_indices],
            alpha_all[active_indices],
            W0_current,
            residual_strain_all[active_indices],
            DeltaT_K=step_delta_t,
            L_m=L_m,
            thermal_arrays=thermal_arrays,
        )

        carried_W_um = float(summary["signed_W_um"])
        residual_strain_all[active_indices] = residual_strain_current
        step_summaries.append({
            "interface_index": group_position - 1,
            "layer_count": group_position + 1,
            "group_id": groups[group_position],
            "signed_W_um": carried_W_um,
            "abs_W_um": float(abs(carried_W_um)),
            "kappa_1_per_m": float(summary["kappa_1_per_m"]),
            "DeltaT_K": float(summary["DeltaT_K"]),
            "nominal_cooldown_DeltaT_K": nominal_delta_t,
            "incremental_thermal_DeltaT_K": step_delta_t,
            "L_m": float(summary["L_m"]),
            "abd_state_residual_generalized_moment_N": float(
                summary["residual_generalized_moment_N"]
            ),
            "abd_state_residual_mean_strain": float(summary["residual_mean_strain"]),
            "abd_state_residual_rms_strain": float(summary["residual_rms_strain"]),
            "abd_state_denominator_N_m": float(summary["denominator_N_m"]),
        })

    if step_summaries:
        final_summary = {
            "signed_W_um": float(step_summaries[-1]["signed_W_um"]),
            "abs_W_um": float(step_summaries[-1]["abs_W_um"]),
            "kappa_1_per_m": float(step_summaries[-1]["kappa_1_per_m"]),
            "DeltaT_K": float(step_summaries[-1]["DeltaT_K"]),
            "L_m": float(step_summaries[-1]["L_m"]),
            "abd_state_residual_generalized_moment_N": float(
                step_summaries[-1]["abd_state_residual_generalized_moment_N"]
            ),
            "abd_state_residual_mean_strain": float(
                step_summaries[-1]["abd_state_residual_mean_strain"]
            ),
            "abd_state_residual_rms_strain": float(
                step_summaries[-1]["abd_state_residual_rms_strain"]
            ),
        }
    else:
        kappa = float(bow_um_to_curvature(carried_W_um, L_m))
        final_summary = {
            "signed_W_um": carried_W_um,
            "abs_W_um": float(abs(carried_W_um)),
            "kappa_1_per_m": kappa,
            "DeltaT_K": 0.0,
            "L_m": float(L_m),
            "abd_state_residual_generalized_moment_N": 0.0,
            "abd_state_residual_mean_strain": 0.0,
            "abd_state_residual_rms_strain": 0.0,
        }
    return final_summary, step_summaries


def compute_sequential_stack_warpage_grouped_stress_history(
    layer_df,
    group_ids,
    DeltaT_K_by_step,
    L_m,
):
    """
    W2W recurrence for the exp04 stress-history reflattening setup.

    This path is intentionally separate from the production compacted-state
    recurrence.  Exp04 keeps the complete add-wafer process in one MAPDL load
    history and, before each new bond, flattens all currently active wafers.
    Under that setup the analytical step should:

    - keep each active wafer's original pre-bond bow in the flatten/release
      term instead of replacing completed lower wafers by a carried stack bow;
    - keep all active physical layers in the thermal mismatch solve instead of
      collapsing the completed lower stack to one equivalent thermal layer.

    No fitted scale, pattern gate, or empirical memory coefficient is used.
    """
    df = layer_df.copy().reset_index(drop=True)
    group_values = pd.Series(group_ids).reset_index(drop=True)
    if len(group_values) != len(df):
        raise ValueError("group_ids must have the same length as layer_df.")

    h_all = df["h_um"].to_numpy(dtype=float)
    E_all = df["E_GPa"].to_numpy(dtype=float)
    nu_all = df["nu"].to_numpy(dtype=float)
    alpha_all = df["alpha_ppm_K"].to_numpy(dtype=float)
    W0_all = df["W0_um"].to_numpy(dtype=float)
    groups = list(pd.unique(group_values))
    num_groups = len(groups)
    if num_groups < 1:
        raise ValueError("layer_df must contain at least one layer group.")

    delta_t_values = np.asarray(DeltaT_K_by_step, dtype=float)
    if delta_t_values.size == 1 and num_groups > 1:
        delta_t_values = np.full(num_groups - 1, float(delta_t_values[0]))
    if delta_t_values.size != max(num_groups - 1, 0):
        raise ValueError(
            f"DeltaT_K_by_step must have {num_groups - 1} entries for "
            f"{num_groups} layer groups; got {delta_t_values.size}."
        )

    group_masks = [group_values.to_numpy() == group for group in groups]
    first_group_mask = group_masks[0]
    carried_W_um = float(W0_all[first_group_mask][0])
    step_summaries = []

    for group_position in range(1, num_groups):
        active_mask = np.logical_or.reduce(group_masks[: group_position + 1])
        active_indices = np.nonzero(active_mask)[0]
        W0_current = W0_all[active_indices].copy()
        thermal_arrays = {
            "h_um": h_all[active_indices],
            "E_GPa": E_all[active_indices],
            "nu": nu_all[active_indices],
            "alpha_ppm_K": alpha_all[active_indices],
        }

        nominal_delta_t = float(delta_t_values[group_position - 1])
        step_delta_t = _w2w_incremental_thermal_delta_t(nominal_delta_t)
        summary = _compute_total_stack_warpage_arrays(
            h_all[active_indices],
            E_all[active_indices],
            nu_all[active_indices],
            alpha_all[active_indices],
            W0_current,
            DeltaT_K=step_delta_t,
            L_m=L_m,
            thermal_arrays=thermal_arrays,
        )
        carried_W_um = float(summary["signed_W_um"])
        step_summaries.append({
            "interface_index": group_position - 1,
            "layer_count": group_position + 1,
            "group_id": groups[group_position],
            "signed_W_um": carried_W_um,
            "abs_W_um": float(abs(carried_W_um)),
            "kappa_1_per_m": float(summary["kappa_1_per_m"]),
            "DeltaT_K": float(summary["DeltaT_K"]),
            "nominal_cooldown_DeltaT_K": nominal_delta_t,
            "incremental_thermal_DeltaT_K": step_delta_t,
            "L_m": float(summary["L_m"]),
            "lower_state_w0_mode": "original_layer_bow",
            "thermal_state": "full_active_stack",
            "residual_strain_carry": False,
        })

    if step_summaries:
        final_summary = {
            "signed_W_um": float(step_summaries[-1]["signed_W_um"]),
            "abs_W_um": float(step_summaries[-1]["abs_W_um"]),
            "kappa_1_per_m": float(step_summaries[-1]["kappa_1_per_m"]),
            "DeltaT_K": float(step_summaries[-1]["DeltaT_K"]),
            "L_m": float(step_summaries[-1]["L_m"]),
            "lower_state_w0_mode": "original_layer_bow",
            "thermal_state": "full_active_stack",
            "residual_strain_carry": False,
        }
    else:
        kappa = float(bow_um_to_curvature(carried_W_um, L_m))
        final_summary = {
            "signed_W_um": carried_W_um,
            "abs_W_um": float(abs(carried_W_um)),
            "kappa_1_per_m": kappa,
            "DeltaT_K": 0.0,
            "L_m": float(L_m),
            "lower_state_w0_mode": "original_layer_bow",
            "thermal_state": "full_active_stack",
            "residual_strain_carry": False,
        }
    return final_summary, step_summaries


def _abd_stiffness_terms(layer_df):
    """
    Return 1D ABD stiffness terms using the same axial modulus convention as
    the Suhir/Kim recurrence. These are diagnostic terms for reduced residual
    moment bookkeeping, not a replacement for the baseline release formula.
    """
    df = layer_df.copy().reset_index(drop=True)
    h = df["h_um"].to_numpy(dtype=float) * 1e-6
    E = df["E_GPa"].to_numpy(dtype=float) * 1e9
    nu = df["nu"].to_numpy(dtype=float)
    if np.any(h <= 0.0):
        raise ValueError("All layers must have positive thickness.")
    E_axial = E / (1.0 - nu)
    z_top = np.cumsum(h)
    z_bot = z_top - h
    A = float(np.sum(E_axial * h))
    B = float(np.sum(E_axial * 0.5 * (z_top**2 - z_bot**2)))
    D = float(np.sum(E_axial * (z_top**3 - z_bot**3) / 3.0))
    D_eff = D - B * B / A if A > 0.0 else D
    return {
        "A_N_per_m": A,
        "B_N": B,
        "D_N_m": D,
        "D_eff_N_m": float(D_eff),
    }


def compute_sequential_stack_warpage(
    layer_df,
    DeltaT_K_by_step,
    L_m,
    residual_curvature_carry=DEFAULT_RESIDUAL_CURVATURE_CARRY,
    residual_curvature_memory_decay=DEFAULT_RESIDUAL_CURVATURE_MEMORY_DECAY,
):
    """
    Compute W2W warpage after each sequential bonding anneal.

    The already-bonded lower stack is represented by its carried post-release
    bow before each new wafer is appended. Layer stiffness/thicknesses remain
    explicit, while prior layers' initial bow is set to the carried stack bow.

    For the incremental thermal mismatch, the completed lower stack is collapsed
    to one equivalent layer.  This keeps the recurrence stable and avoids
    repeatedly applying old internal CTE mismatch during later anneals.  The
    thermal step sign is reversed relative to batch cooldown because the carried
    lower stack starts as a released room-temperature body before the next W2W
    anneal.

    The production compacted-state recurrence transfers only the released bow
    of the completed stack.  This matches the current MAPDL verification setup:
    after each step, the lower stack is reintroduced as its released spherical
    shape instead of a full residual stress/strain history.

    A scalar residual-curvature memory can still be enabled explicitly for
    future stress-history studies:

        R_k = lambda * R_(k-1) + (W_k - W_(k-1))
        W_eff,k = W_k + eta * R_k

    For this compacted MAPDL verification, the default eta is zero so W_eff,k
    equals W_k and the history is not counted twice.
    """
    df = layer_df.copy().reset_index(drop=True)
    df["W0_um"] = df["W0_um"].astype(float)
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

    carry_gain = float(residual_curvature_carry)
    memory_decay = float(residual_curvature_memory_decay)
    carried_W_um = float(df.loc[0, "W0_um"])
    previous_W_um = carried_W_um
    residual_state_W_um = 0.0
    step_summaries = []

    for layer_index in range(1, num_layers):
        residual_delta_W_um = carried_W_um - previous_W_um
        residual_state_W_um = memory_decay * residual_state_W_um + residual_delta_W_um
        effective_carried_W_um = carried_W_um + carry_gain * residual_state_W_um
        current_df = df.iloc[:layer_index + 1].copy().reset_index(drop=True)
        current_df.loc[:layer_index - 1, "W0_um"] = effective_carried_W_um
        thermal_df = _sequential_thermal_layers(
            df.iloc[:layer_index + 1].copy(),
            incoming_layer_index=layer_index,
            carried_W_um=effective_carried_W_um,
        )
        nominal_delta_t = float(delta_t_values[layer_index - 1])
        step_delta_t = _w2w_incremental_thermal_delta_t(nominal_delta_t)
        summary, layer_out = compute_total_stack_warpage(
            current_df,
            DeltaT_K=step_delta_t,
            L_m=L_m,
            thermal_layer_df=thermal_df,
        )
        previous_W_um = carried_W_um
        carried_W_um = float(summary["signed_W_um"])
        step_summaries.append({
            "interface_index": layer_index - 1,
            "layer_count": layer_index + 1,
            "signed_W_um": carried_W_um,
            "abs_W_um": float(abs(carried_W_um)),
            "kappa_1_per_m": float(summary["kappa_1_per_m"]),
            "DeltaT_K": float(summary["DeltaT_K"]),
            "nominal_cooldown_DeltaT_K": nominal_delta_t,
            "incremental_thermal_DeltaT_K": step_delta_t,
            "L_m": float(summary["L_m"]),
            "effective_carried_W_um": float(effective_carried_W_um),
            "residual_delta_W_um": float(residual_delta_W_um),
            "residual_state_W_um": float(residual_state_W_um),
            "residual_curvature_carry": carry_gain,
            "residual_curvature_memory_decay": memory_decay,
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


def compute_sequential_stack_warpage_abd_residual(
    layer_df,
    DeltaT_K_by_step,
    L_m,
    group_ids=None,
    residual_curvature_carry=DEFAULT_RESIDUAL_CURVATURE_CARRY,
    residual_curvature_memory_decay=DEFAULT_RESIDUAL_CURVATURE_MEMORY_DECAY,
    thermal_residual_moment_carry=DEFAULT_THERMAL_RESIDUAL_MOMENT_CARRY,
    thermal_residual_moment_memory_decay=DEFAULT_THERMAL_RESIDUAL_MOMENT_MEMORY_DECAY,
    thermal_residual_moment_terminal_power=DEFAULT_THERMAL_RESIDUAL_MOMENT_TERMINAL_POWER,
    feed_corrected_bow=False,
):
    """
    Experimental W2W recurrence with a reduced ABD residual-moment state.

    The baseline recurrence is intentionally preserved.  Each step is first
    solved with the production Suhir/Kim single-step formula using the W2W
    incremental thermal-step sign convention.  A second solve with DeltaT=0
    separates the incremental thermal-mismatch bow contribution.  That thermal
    contribution is accumulated as a decaying residual bending state and
    applied as a moment correction to the reported released bow:

        R_T,k = lambda_T * R_T,k-1 + (W_full,k - W_no_thermal,k)
        q_k = 1 - (k / k_final)^p
        W_abd,k = W_full,k - gamma_T * q_k * R_T,k

    `R_T` is a reduced proxy for the residual thermal bending moment of the
    completed stack.  The terminal weight q_k lets the correction improve
    intermediate physical states while returning the final stack to the
    production baseline. ABD stiffness terms are reported so the proxy can be
    converted to a moment-like diagnostic, but the production baseline path is
    not changed.

    If `group_ids` is supplied, rows with the same group id are appended
    together.  This is useful when a wafer is represented by multiple physical
    sublayers.
    """
    df = layer_df.copy().reset_index(drop=True)
    df["W0_um"] = df["W0_um"].astype(float)
    if group_ids is None:
        group_values = pd.Series(range(len(df)))
    else:
        group_values = pd.Series(group_ids).reset_index(drop=True)
        if len(group_values) != len(df):
            raise ValueError("group_ids must have the same length as layer_df.")

    groups = list(pd.unique(group_values))
    num_groups = len(groups)
    if num_groups < 1:
        raise ValueError("layer_df must contain at least one layer group.")

    delta_t_values = np.asarray(DeltaT_K_by_step, dtype=float)
    if delta_t_values.size == 1 and num_groups > 1:
        delta_t_values = np.full(num_groups - 1, float(delta_t_values[0]))
    if delta_t_values.size != max(num_groups - 1, 0):
        raise ValueError(
            f"DeltaT_K_by_step must have {num_groups - 1} entries for "
            f"{num_groups} layer groups; got {delta_t_values.size}."
        )

    carry_gain = float(residual_curvature_carry)
    memory_decay = float(residual_curvature_memory_decay)
    thermal_gain = float(thermal_residual_moment_carry)
    thermal_memory_decay = float(thermal_residual_moment_memory_decay)
    terminal_power = float(thermal_residual_moment_terminal_power)

    first_group_mask = group_values == groups[0]
    carried_W_um = float(df.loc[first_group_mask, "W0_um"].iloc[0])
    previous_W_um = carried_W_um
    residual_state_W_um = 0.0
    thermal_residual_state_W_um = 0.0
    step_summaries = []

    for group_position in range(1, num_groups):
        incoming_group = groups[group_position]
        group_mask = group_values.isin(groups[:group_position + 1])
        lower_mask = group_values.isin(groups[:group_position])

        residual_delta_W_um = carried_W_um - previous_W_um
        residual_state_W_um = memory_decay * residual_state_W_um + residual_delta_W_um
        effective_carried_W_um = carried_W_um + carry_gain * residual_state_W_um

        current_df = df.loc[group_mask].copy().reset_index(drop=True)
        current_group_values = group_values.loc[group_mask].reset_index(drop=True)
        current_df.loc[current_group_values.isin(groups[:group_position]), "W0_um"] = effective_carried_W_um
        layer_for_solve = current_df[["h_um", "E_GPa", "nu", "alpha_ppm_K", "W0_um"]].reset_index(drop=True)
        thermal_df = _sequential_thermal_group_layers(
            current_df[["h_um", "E_GPa", "nu", "alpha_ppm_K", "W0_um"]].copy(),
            current_group_values,
            incoming_group=incoming_group,
            carried_W_um=effective_carried_W_um,
        )

        nominal_delta_t = float(delta_t_values[group_position - 1])
        step_delta_t = _w2w_incremental_thermal_delta_t(nominal_delta_t)
        baseline_summary, layer_out = compute_total_stack_warpage(
            layer_for_solve,
            DeltaT_K=step_delta_t,
            L_m=L_m,
            thermal_layer_df=thermal_df,
        )
        no_thermal_summary, _ = compute_total_stack_warpage(
            layer_for_solve,
            DeltaT_K=0.0,
            L_m=L_m,
            thermal_layer_df=thermal_df,
        )

        baseline_W_um = float(baseline_summary["signed_W_um"])
        thermal_delta_W_um = baseline_W_um - float(no_thermal_summary["signed_W_um"])
        thermal_residual_state_W_um = (
            thermal_memory_decay * thermal_residual_state_W_um + thermal_delta_W_um
        )
        if num_groups > 1:
            terminal_fraction = group_position / (num_groups - 1)
        else:
            terminal_fraction = 1.0
        terminal_weight = max(0.0, 1.0 - terminal_fraction**terminal_power)
        corrected_W_um = baseline_W_um - thermal_gain * terminal_weight * thermal_residual_state_W_um
        corrected_kappa = float(bow_um_to_curvature(corrected_W_um, L_m))

        abd_terms = _abd_stiffness_terms(layer_for_solve)
        residual_moment_kappa = float(bow_um_to_curvature(thermal_residual_state_W_um, L_m))
        residual_moment_N = abd_terms["D_eff_N_m"] * residual_moment_kappa

        previous_W_um = carried_W_um
        carried_W_um = corrected_W_um if feed_corrected_bow else baseline_W_um
        step_summaries.append({
            "interface_index": group_position - 1,
            "layer_count": group_position + 1,
            "group_id": incoming_group,
            "signed_W_um": float(corrected_W_um),
            "abs_W_um": float(abs(corrected_W_um)),
            "kappa_1_per_m": corrected_kappa,
            "baseline_signed_W_um": baseline_W_um,
            "baseline_kappa_1_per_m": float(baseline_summary["kappa_1_per_m"]),
            "no_thermal_signed_W_um": float(no_thermal_summary["signed_W_um"]),
            "thermal_delta_W_um": float(thermal_delta_W_um),
            "thermal_residual_state_W_um": float(thermal_residual_state_W_um),
            "thermal_residual_moment_N": float(residual_moment_N),
            "thermal_residual_moment_carry": thermal_gain,
            "thermal_residual_moment_memory_decay": thermal_memory_decay,
            "thermal_residual_moment_terminal_power": terminal_power,
            "thermal_residual_moment_terminal_weight": float(terminal_weight),
            "feed_corrected_bow": bool(feed_corrected_bow),
            "DeltaT_K": float(baseline_summary["DeltaT_K"]),
            "nominal_cooldown_DeltaT_K": nominal_delta_t,
            "incremental_thermal_DeltaT_K": step_delta_t,
            "L_m": float(baseline_summary["L_m"]),
            "effective_carried_W_um": float(effective_carried_W_um),
            "residual_delta_W_um": float(residual_delta_W_um),
            "residual_state_W_um": float(residual_state_W_um),
            "residual_curvature_carry": carry_gain,
            "residual_curvature_memory_decay": memory_decay,
            "abd_A_N_per_m": float(abd_terms["A_N_per_m"]),
            "abd_B_N": float(abd_terms["B_N"]),
            "abd_D_N_m": float(abd_terms["D_N_m"]),
            "abd_D_eff_N_m": float(abd_terms["D_eff_N_m"]),
            "layers": layer_out.to_dict(orient="records"),
        })

    if step_summaries:
        final_summary = {
            "signed_W_um": float(step_summaries[-1]["signed_W_um"]),
            "abs_W_um": float(step_summaries[-1]["abs_W_um"]),
            "kappa_1_per_m": float(step_summaries[-1]["kappa_1_per_m"]),
            "baseline_signed_W_um": float(step_summaries[-1]["baseline_signed_W_um"]),
            "baseline_kappa_1_per_m": float(step_summaries[-1]["baseline_kappa_1_per_m"]),
            "thermal_residual_state_W_um": float(step_summaries[-1]["thermal_residual_state_W_um"]),
            "thermal_residual_moment_N": float(step_summaries[-1]["thermal_residual_moment_N"]),
            "DeltaT_K": float(step_summaries[-1]["DeltaT_K"]),
            "L_m": float(step_summaries[-1]["L_m"]),
        }
    else:
        kappa = float(bow_um_to_curvature(carried_W_um, L_m))
        final_summary = {
            "signed_W_um": carried_W_um,
            "abs_W_um": float(abs(carried_W_um)),
            "kappa_1_per_m": kappa,
            "baseline_signed_W_um": carried_W_um,
            "baseline_kappa_1_per_m": kappa,
            "thermal_residual_state_W_um": 0.0,
            "thermal_residual_moment_N": 0.0,
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
            mean = 0.0
    std = _first_existing_float(
        cfg,
        [
            f"{side}_INI_BOW_STD_um",
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


def _effective_material_properties_from_volume_fraction(
    cfg,
    mix_prefix,
    num_dies_per_wafer=None,
):
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
        volumes = w2w_area_scaled_layer_volumes(
            cfg,
            mix_prefix,
            num_dies_per_wafer=num_dies_per_wafer,
        )
        source = "config_volume_fraction"
    else:
        volumes = {"Cu": 0.0, "Sio2": 0.0, "Si": 1.0}
        source = "placeholder_pure_si"

    material_volumes = {
        material: float(volumes.get(material, 0.0))
        for material in material_defaults
    }

    if any(value < 0.0 for value in material_volumes.values()):
        raise ValueError(f"{mix_prefix} volume fractions must be non-negative: {volumes}")

    total_volume = sum(material_volumes.values())
    if total_volume <= 0.0:
        material_volumes = {"Cu": 0.0, "Sio2": 0.0, "Si": 1.0}
        total_volume = 1.0
        source = "placeholder_pure_si"

    weights = {
        material: volume / total_volume
        for material, volume in material_volumes.items()
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
        "Cu_V": float(material_volumes["Cu"]),
        "Sio2_V": float(material_volumes["Sio2"]),
        "Si_V": float(material_volumes["Si"]),
        "volume_sum": float(total_volume),
        "w2w_die_area_fill_factor": float(volumes.get("area_fill_factor", 1.0)),
        "material_property_source": source,
    }


def _build_layer_df(
    cfg_dict,
    chiplets_bottom_to_top,
    interfaces_bottom_to_top,
    chiplet_instances_bottom_to_top=None,
    num_dies_per_wafer=None,
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
        effective_material = _effective_material_properties_from_volume_fraction(
            cfg,
            mix_prefix,
            num_dies_per_wafer=num_dies_per_wafer,
        )

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
            "w2w_die_area_fill_factor": effective_material["w2w_die_area_fill_factor"],
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


def get_interface_existing_stack_warpage_map(
    cfg_dict,
    _3dbx_path,
    num_dies_per_wafer=None,
):
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
            num_dies_per_wafer=num_dies_per_wafer,
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


def compute_warpage_yield(cfg_dict, _3dbx_path, num_dies_per_wafer=None):
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
            num_dies_per_wafer=num_dies_per_wafer,
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

    result = compute_warpage_yield(
        cfg_dict,
        _3dbx_path,
        num_dies_per_wafer=waf_stack.num_dies_per_wafer,
    )
    for info in result["substack_results"].values():
        final_interface = info["final_interface"]
        waf_stack.die_yield_list_per_interface_dict[final_interface]["warpage"] = np.full(
            waf_stack.num_dies_per_wafer,
            float(info["warpage_yield"]),
            dtype=float,
        )

    return result
