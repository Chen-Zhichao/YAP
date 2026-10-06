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

Room-temperature and Cu elastic-plastic experimental variants are implemented
below without changing the default W2W yield path.
"""

from collections import defaultdict
import importlib.util
import math
from pathlib import Path
import sys

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
COMPACTED_THICKNESS_GRADIENT_RELATIVE_SPAN = 0.05
COMPACTED_MATERIAL_E_RELATIVE_SPAN_FOR_THICKNESS_ONLY = 0.01
COMPACTED_MATERIAL_ALPHA_RELATIVE_SPAN_FOR_THICKNESS_ONLY = 0.002


def _w2w_incremental_thermal_delta_t(delta_t):
    """
    Convert a nominal cooldown DeltaT into the incremental W2W thermal step.

    MAPDL uses TREF=T_anneal and then applies TEMP=T_room, so the cooldown
    passed through the W2W recurrence should keep the physical sign
    DeltaT = T_room - T_anneal.  Earlier diagnostic code flipped this sign to
    emulate a carried lower-stack reheating step, but the exp03/exp04 MAPDL
    decks rebuild the compacted lower state stress-free at anneal before the
    cooling solve.  Keeping the MAPDL sign avoids applying the CTE-mismatch
    bending moment in the opposite direction.
    """
    return float(delta_t)


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


def _thermal_modulus(E, nu, mode="plate", layer_kind=None):
    """
    Return the modulus used to weight the CTE-mismatch thermal moment.

    The historical W2W model uses the plate response E/(1-nu^2).  For a true
    circular W2W wafer with spherical/axisymmetric bow, the Si wafer substrate
    is better represented by the equibiaxial E/(1-nu) convention.  The hybrid
    bond interface remains a very thin equivalent Cu/SiO2 film, so keep the
    original plate weighting for layers identified as interfaces.
    """
    E = np.asarray(E, dtype=float)
    nu = np.asarray(nu, dtype=float)
    if mode == "plate":
        return E / (1.0 - nu**2)
    if mode == "axial":
        return E / (1.0 - nu)
    if mode != "w2w_mixed":
        raise ValueError(f"Unknown thermal modulus mode: {mode}")

    if layer_kind is None:
        return E / (1.0 - nu**2)
    labels = np.asarray(layer_kind, dtype=object)
    if labels.size != E.size:
        raise ValueError("layer_kind must have one entry per thermal layer.")
    is_interface = np.asarray([
        "interface" in str(label).lower()
        for label in labels
    ], dtype=bool)
    return np.where(is_interface, E / (1.0 - nu**2), E / (1.0 - nu))


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
    # MAPDL uses TREF=T_anneal and TEMP=T_room, so the mechanical strain from
    # cooling is the negative of the free thermal strain relative to the
    # weighted reference CTE.  The thermal generalized moment therefore enters
    # with the opposite sign from the geometric initial-curvature moment.
    M_T = -np.sum(a_t * thermal_weights * (delta_alpha_t - s_alpha) * DeltaT_K)
    M_b = np.sum((a / lam) * (b - s_b))
    K_a = np.sum((a / lam) * (a - s_a))
    K_D = np.sum(D)
    M_D0 = np.sum(D * kappa0)

    kappa = (M_D0 + M_T + M_b) / (K_D + K_a)
    W_um = curvature_to_bow_um(kappa, L_m)

    strain_term = -(delta_alpha - s_alpha) * DeltaT_K - (a - s_a) * kappa + (b - s_b)
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
    thermal_modulus_mode="plate",
    layer_kind=None,
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
        layer_kind_t = layer_kind
    else:
        h_t = np.asarray(thermal_arrays["h_um"], dtype=float) * 1e-6
        E_t = np.asarray(thermal_arrays["E_GPa"], dtype=float) * 1e9
        nu_t = np.asarray(thermal_arrays["nu"], dtype=float)
        alpha_t = np.asarray(thermal_arrays["alpha_ppm_K"], dtype=float) * 1e-6
        layer_kind_t = None
        for kind_key in ("thermal_modulus_kind", "sublayer", "layer_kind"):
            if kind_key in thermal_arrays:
                layer_kind_t = thermal_arrays[kind_key]
                break

    E_thermal = _thermal_modulus(
        E_t,
        nu_t,
        mode=thermal_modulus_mode,
        layer_kind=layer_kind_t,
    )
    thermal_weights = E_thermal * h_t
    h_t_cumsum = np.cumsum(h_t)
    a_t = h_t_cumsum - 0.5 * (h_t[0] + h_t)
    delta_alpha_t = alpha_t - alpha_t[0]
    s_alpha = np.sum(delta_alpha_t * thermal_weights) / np.sum(thermal_weights)
    s_a = np.sum(a * weights) / np.sum(weights)
    s_b = np.sum(b * weights) / np.sum(weights)

    D = E_biaxial * h**3 / 12.0
    M_T = -np.sum(a_t * thermal_weights * (delta_alpha_t - s_alpha) * DeltaT_K)
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
    thermal_modulus_mode="plate",
    layer_kind=None,
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
        layer_kind_t = layer_kind
    else:
        h_t = np.asarray(thermal_arrays["h_um"], dtype=float) * 1e-6
        E_t = np.asarray(thermal_arrays["E_GPa"], dtype=float) * 1e9
        nu_t = np.asarray(thermal_arrays["nu"], dtype=float)
        alpha_t = np.asarray(thermal_arrays["alpha_ppm_K"], dtype=float) * 1e-6
        layer_kind_t = None
        for kind_key in ("thermal_modulus_kind", "sublayer", "layer_kind"):
            if kind_key in thermal_arrays:
                layer_kind_t = thermal_arrays[kind_key]
                break

    E_thermal = _thermal_modulus(
        E_t,
        nu_t,
        mode=thermal_modulus_mode,
        layer_kind=layer_kind_t,
    )
    thermal_weights = E_thermal * h_t
    h_t_cumsum = np.cumsum(h_t)
    a_t = h_t_cumsum - 0.5 * (h_t[0] + h_t)
    delta_alpha_t = alpha_t - alpha_t[0]
    s_alpha = np.sum(delta_alpha_t * thermal_weights) / np.sum(thermal_weights)
    s_a = np.sum(a * weights) / np.sum(weights)
    s_b = np.sum(b * weights) / np.sum(weights)
    s_residual = np.sum(residual * weights) / np.sum(weights)

    D = E_biaxial * h**3 / 12.0
    M_T = -np.sum(a_t * thermal_weights * (delta_alpha_t - s_alpha) * DeltaT_K)
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
        -(delta_alpha - s_alpha) * DeltaT_K
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


def _equivalent_thermal_arrays(
    h_um,
    E_GPa,
    nu,
    alpha_ppm_K,
    thermal_modulus_mode="plate",
    layer_kind=None,
):
    h = np.asarray(h_um, dtype=float)
    E = np.asarray(E_GPa, dtype=float)
    nu = np.asarray(nu, dtype=float)
    alpha = np.asarray(alpha_ppm_K, dtype=float)
    h_total = float(np.sum(h))
    E_axial = E / (1.0 - nu)
    E_thermal = _thermal_modulus(
        E,
        nu,
        mode=thermal_modulus_mode,
        layer_kind=layer_kind,
    )
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


def _single_equivalent_lower_thermal_arrays(
    h_all,
    E_all,
    nu_all,
    alpha_all,
    lower_indices,
    incoming_indices,
):
    lower_eq = _equivalent_thermal_arrays(
        h_all[lower_indices],
        E_all[lower_indices],
        nu_all[lower_indices],
        alpha_all[lower_indices],
    )
    return {
        "h_um": np.concatenate(([lower_eq["h_um"]], h_all[incoming_indices])),
        "E_GPa": np.concatenate(([lower_eq["E_GPa"]], E_all[incoming_indices])),
        "nu": np.concatenate(([lower_eq["nu"]], nu_all[incoming_indices])),
        "alpha_ppm_K": np.concatenate(([lower_eq["alpha_ppm_K"]], alpha_all[incoming_indices])),
    }, "single_lower_equivalent"


def _per_group_equivalent_lower_thermal_arrays(
    h_all,
    E_all,
    nu_all,
    alpha_all,
    group_masks,
    group_position,
    incoming_indices,
):
    pieces = {
        "h_um": [],
        "E_GPa": [],
        "nu": [],
        "alpha_ppm_K": [],
    }
    for mask in group_masks[:group_position]:
        indices = np.nonzero(mask)[0]
        group_eq = _equivalent_thermal_arrays(
            h_all[indices],
            E_all[indices],
            nu_all[indices],
            alpha_all[indices],
        )
        for key in pieces:
            pieces[key].append(group_eq[key])
    for idx in incoming_indices:
        pieces["h_um"].append(h_all[idx])
        pieces["E_GPa"].append(E_all[idx])
        pieces["nu"].append(nu_all[idx])
        pieces["alpha_ppm_K"].append(alpha_all[idx])
    return {
        key: np.asarray(values, dtype=float)
        for key, values in pieces.items()
    }, "per_group_lower_equivalent"


def _has_physical_sublayer_metadata(layer_kind_all):
    if layer_kind_all is None:
        return False
    labels = [str(label).lower() for label in np.asarray(layer_kind_all, dtype=object)]
    return any("interface" in label for label in labels) and any("substrate" in label for label in labels)


def _explicit_lower_uniform_alpha_thermal_arrays(
    h_all,
    E_all,
    nu_all,
    alpha_all,
    lower_indices,
    incoming_indices,
    *,
    thermal_modulus_mode="plate",
    layer_kind_all=None,
):
    """
    Thermal state for compacted W2W sandwich wafers.

    The compacted MAPDL verification keeps the completed lower stack explicit
    for stiffness/geometry, but gives every completed lower layer one equivalent
    CTE so old internal thermal mismatch is not re-applied at the next anneal.
    Mirror that state directly instead of replacing the lower stack by one
    geometric layer.
    """
    lower_kind = None if layer_kind_all is None else layer_kind_all[lower_indices]
    lower_eq = _equivalent_thermal_arrays(
        h_all[lower_indices],
        E_all[lower_indices],
        nu_all[lower_indices],
        alpha_all[lower_indices],
        thermal_modulus_mode=thermal_modulus_mode,
        layer_kind=lower_kind,
    )
    thermal_arrays = {
        "h_um": np.concatenate((h_all[lower_indices], h_all[incoming_indices])),
        "E_GPa": np.concatenate((E_all[lower_indices], E_all[incoming_indices])),
        "nu": np.concatenate((nu_all[lower_indices], nu_all[incoming_indices])),
        "alpha_ppm_K": np.concatenate((
            np.full(lower_indices.size, lower_eq["alpha_ppm_K"], dtype=float),
            alpha_all[incoming_indices],
        )),
    }
    if layer_kind_all is not None:
        thermal_arrays["thermal_modulus_kind"] = np.concatenate((
            layer_kind_all[lower_indices],
            layer_kind_all[incoming_indices],
        ))
    return thermal_arrays, "explicit_lower_uniform_alpha"


def _select_compacted_thermal_arrays(
    h_all,
    E_all,
    nu_all,
    alpha_all,
    group_masks,
    group_position,
    lower_indices,
    incoming_indices,
    *,
    thermal_modulus_mode="plate",
    layer_kind_all=None,
):
    """
    Choose a compact thermal state for a completed W2W lower stack.

    For ordinary material stacks, the completed lower stack is represented as
    one thermal-equivalent layer so its internal CTE mismatch is not applied as
    a fresh load at every later compacted W2W step.  If the wafer-to-wafer
    variation is primarily thickness, however, collapsing all completed wafers
    into one layer erases their through-thickness locations.  In that case,
    keep one equivalent layer per completed wafer.  This preserves neutral-axis
    placement without introducing a fitted scale factor.
    """
    if _has_physical_sublayer_metadata(layer_kind_all):
        thermal_arrays, mode = _explicit_lower_uniform_alpha_thermal_arrays(
            h_all,
            E_all,
            nu_all,
            alpha_all,
            lower_indices,
            incoming_indices,
            thermal_modulus_mode=thermal_modulus_mode,
            layer_kind_all=layer_kind_all,
        )
        props = _group_effective_properties(h_all, E_all, nu_all, alpha_all, group_masks)
        return thermal_arrays, {
            "compacted_thermal_state_mode": mode,
            "group_thickness_relative_span": float(_relative_span(props["h_um"])),
            "group_E_relative_span": float(_relative_span(props["E_GPa"])),
            "group_alpha_relative_span": float(_relative_span(props["alpha_ppm_K"])),
        }

    props = _group_effective_properties(h_all, E_all, nu_all, alpha_all, group_masks)
    thickness_span = _relative_span(props["h_um"])
    e_span = _relative_span(props["E_GPa"])
    alpha_span = _relative_span(props["alpha_ppm_K"])
    use_per_group = (
        thickness_span > COMPACTED_THICKNESS_GRADIENT_RELATIVE_SPAN
        and e_span < COMPACTED_MATERIAL_E_RELATIVE_SPAN_FOR_THICKNESS_ONLY
        and alpha_span < COMPACTED_MATERIAL_ALPHA_RELATIVE_SPAN_FOR_THICKNESS_ONLY
    )
    if use_per_group:
        thermal_arrays, mode = _per_group_equivalent_lower_thermal_arrays(
            h_all,
            E_all,
            nu_all,
            alpha_all,
            group_masks,
            group_position,
            incoming_indices,
        )
    else:
        thermal_arrays, mode = _single_equivalent_lower_thermal_arrays(
            h_all,
            E_all,
            nu_all,
            alpha_all,
            lower_indices,
            incoming_indices,
        )
    return thermal_arrays, {
        "compacted_thermal_state_mode": mode,
        "group_thickness_relative_span": float(thickness_span),
        "group_E_relative_span": float(e_span),
        "group_alpha_relative_span": float(alpha_span),
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
    layer_kind_all = None
    thermal_modulus_mode = "plate"
    for kind_column in ("thermal_modulus_kind", "sublayer", "layer_kind"):
        if kind_column in df.columns:
            layer_kind_all = df[kind_column].to_numpy(dtype=object)
            break
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

        thermal_arrays, thermal_state_info = _select_compacted_thermal_arrays(
            h_all,
            E_all,
            nu_all,
            alpha_all,
            group_masks,
            group_position,
            lower_indices,
            incoming_indices,
            thermal_modulus_mode=thermal_modulus_mode,
            layer_kind_all=layer_kind_all,
        )

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
            "thermal_modulus_mode": thermal_modulus_mode,
            **thermal_state_info,
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


def _relative_span(values):
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return 0.0
    scale = max(abs(float(np.mean(values))), 1.0e-12)
    return float((np.max(values) - np.min(values)) / scale)


def _group_effective_properties(h_um, E_GPa, nu, alpha_ppm_K, group_masks):
    h = np.asarray(h_um, dtype=float)
    E = np.asarray(E_GPa, dtype=float)
    nu = np.asarray(nu, dtype=float)
    alpha = np.asarray(alpha_ppm_K, dtype=float)
    group_h = []
    group_E = []
    group_alpha = []
    interface_alpha = []
    for mask in group_masks:
        idx = np.nonzero(mask)[0]
        h_g = h[idx]
        E_g = E[idx]
        nu_g = nu[idx]
        alpha_g = alpha[idx]
        h_total = float(np.sum(h_g))
        E_axial = E_g / (1.0 - nu_g)
        E_thermal = E_g / (1.0 - nu_g**2)
        axial_weights = E_axial * h_g
        thermal_weights = E_thermal * h_g
        nu_eff = float(np.sum(nu_g * h_g) / h_total)
        E_axial_eff = float(np.sum(axial_weights) / h_total)
        group_h.append(h_total)
        group_E.append(E_axial_eff * (1.0 - nu_eff))
        group_alpha.append(float(np.sum(alpha_g * thermal_weights) / np.sum(thermal_weights)))
        interface_alpha.append(float(alpha_g[int(np.argmin(h_g))]))
    return {
        "h_um": np.asarray(group_h, dtype=float),
        "E_GPa": np.asarray(group_E, dtype=float),
        "alpha_ppm_K": np.asarray(group_alpha, dtype=float),
        "interface_alpha_ppm_K": np.asarray(interface_alpha, dtype=float),
    }


def _substrate_representative_index(indices, h_all, layer_kind_all=None):
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        raise ValueError("Cannot build a substrate-equivalent thermal layer from an empty group.")
    if layer_kind_all is not None:
        labels = np.asarray(layer_kind_all, dtype=object)
        group_labels = [str(labels[idx]).lower() for idx in indices]
        substrate_positions = [
            pos
            for pos, label in enumerate(group_labels)
            if "substrate" in label or label.startswith("si_") or label == "si"
        ]
        if substrate_positions:
            local = substrate_positions[int(np.argmax(h_all[indices[substrate_positions]]))]
            return int(indices[local])
    return int(indices[int(np.argmax(h_all[indices]))])


def _per_group_substrate_equivalent_thermal_arrays(
    h_all,
    E_all,
    nu_all,
    alpha_all,
    group_masks,
    group_position,
    *,
    layer_kind_all=None,
):
    """
    Thermal state for exp04 sandwich W2W stress-history verification.

    The mechanical solve keeps the explicit interface/substrate sandwich.  For
    thermal bending, however, the 1.5 um hybrid-bond interfaces should not be
    re-applied as independent thin-film CTE loads at every history step.  MAPDL
    indicates those local interface strains mostly follow the wafer substrate
    release, so represent each active wafer by its total thickness and
    substrate material for the thermal moment.  This is a physical reduction,
    not a fitted scale factor.
    """
    h = np.asarray(h_all, dtype=float)
    E = np.asarray(E_all, dtype=float)
    nu = np.asarray(nu_all, dtype=float)
    alpha = np.asarray(alpha_all, dtype=float)
    parts = {
        "h_um": [],
        "E_GPa": [],
        "nu": [],
        "alpha_ppm_K": [],
    }
    for mask in group_masks[: group_position + 1]:
        indices = np.nonzero(mask)[0]
        substrate_idx = _substrate_representative_index(
            indices,
            h,
            layer_kind_all=layer_kind_all,
        )
        parts["h_um"].append(float(np.sum(h[indices])))
        parts["E_GPa"].append(float(E[substrate_idx]))
        parts["nu"].append(float(nu[substrate_idx]))
        parts["alpha_ppm_K"].append(float(alpha[substrate_idx]))
    return {key: np.asarray(values, dtype=float) for key, values in parts.items()}


def _select_grouped_stress_history_state(h_um, E_GPa, nu, alpha_ppm_K, group_masks):
    """
    Pick the analytical state transfer that best matches the physical process.

    The selector is based only on input layer properties, not on ANSYS results:

    - Default: reflattened wafers enter with their original layer bows.
    - For explicit sandwich wafers, the caller replaces the thermal state with
      one substrate-dominated equivalent layer per active wafer.  That keeps
      wafer-to-wafer substrate gradients while preventing the 1.5 um bonding
      interfaces from being re-applied as separate thermal films at every step.
    """

    props = _group_effective_properties(h_um, E_GPa, nu, alpha_ppm_K, group_masks)
    thickness_span = _relative_span(props["h_um"])
    e_span = _relative_span(props["E_GPa"])
    alpha_span = _relative_span(props["alpha_ppm_K"])
    interface_alpha_span = float(np.max(props["interface_alpha_ppm_K"]) - np.min(props["interface_alpha_ppm_K"]))
    thickness_index_correlation = float("nan")
    if props["h_um"].size > 2 and thickness_span > 1.0e-12:
        thickness_index_correlation = float(np.corrcoef(np.arange(props["h_um"].size), props["h_um"])[0, 1])

    if e_span > 2.0e-3 and alpha_span > 2.0e-3:
        corr = float(np.corrcoef(props["E_GPa"], props["alpha_ppm_K"])[0, 1])
        if corr < -0.75:
            return {
                "lower_state_mode": "original_layer_bow",
                "thermal_state_mode": "full_active_stack",
                "thermal_delta_mode": "nominal_cooldown",
                "reason": "opposing_E_alpha_gradient",
                "group_thickness_relative_span": thickness_span,
                "group_thickness_index_correlation": thickness_index_correlation,
                "group_E_relative_span": e_span,
                "group_alpha_relative_span": alpha_span,
                "group_E_alpha_correlation": corr,
                "interface_alpha_span_ppm_K": interface_alpha_span,
            }

    if interface_alpha_span > 0.25:
        return {
            "lower_state_mode": "original_layer_bow",
            "thermal_state_mode": "full_active_stack",
            "thermal_delta_mode": "nominal_cooldown",
            "reason": "interface_material_gradient",
            "group_thickness_relative_span": thickness_span,
            "group_thickness_index_correlation": thickness_index_correlation,
            "group_E_relative_span": e_span,
            "group_alpha_relative_span": alpha_span,
            "group_E_alpha_correlation": float("nan"),
            "interface_alpha_span_ppm_K": interface_alpha_span,
        }

    return {
        "lower_state_mode": "original_layer_bow",
        "thermal_state_mode": "full_active_stack",
        "thermal_delta_mode": "nominal_cooldown",
        "reason": "default_reflattened_original_bow_full_active_thermal",
        "group_thickness_relative_span": thickness_span,
        "group_thickness_index_correlation": thickness_index_correlation,
        "group_E_relative_span": e_span,
        "group_alpha_relative_span": alpha_span,
        "group_E_alpha_correlation": float("nan"),
        "interface_alpha_span_ppm_K": interface_alpha_span,
    }


def _compute_sequential_stack_warpage_grouped_state_mode(
    h_all,
    E_all,
    nu_all,
    alpha_all,
    W0_all,
    groups,
    group_masks,
    delta_t_values,
    L_m,
    *,
    lower_state_mode,
    thermal_state_mode,
    thermal_delta_mode,
    selector_metadata,
    layer_kind_all=None,
    thermal_modulus_mode="plate",
):
    carried_W_um = float(W0_all[group_masks[0]][0])
    if layer_kind_all is not None:
        layer_kind_all = np.asarray(layer_kind_all, dtype=object)
        if layer_kind_all.size != np.asarray(h_all).size:
            raise ValueError("layer_kind_all must have one entry per physical layer.")
    step_summaries = []

    for group_position in range(1, len(groups)):
        active_mask = np.logical_or.reduce(group_masks[: group_position + 1])
        lower_mask = np.logical_or.reduce(group_masks[:group_position])
        incoming_mask = group_masks[group_position]
        active_indices = np.nonzero(active_mask)[0]
        lower_indices = np.nonzero(lower_mask)[0]
        incoming_indices = np.nonzero(incoming_mask)[0]

        W0_current = W0_all[active_indices].copy()
        if lower_state_mode == "carried_stack_bow":
            active_lower = np.isin(active_indices, lower_indices)
            W0_current[active_lower] = carried_W_um
        elif lower_state_mode != "original_layer_bow":
            raise ValueError(f"Unknown lower_state_mode: {lower_state_mode}")

        step_thermal_modulus_mode = thermal_modulus_mode
        if thermal_state_mode == "collapsed_lower_equivalent":
            lower_eq = _equivalent_thermal_arrays(
                h_all[lower_indices],
                E_all[lower_indices],
                nu_all[lower_indices],
                alpha_all[lower_indices],
                thermal_modulus_mode=thermal_modulus_mode,
                layer_kind=(
                    layer_kind_all[lower_indices]
                    if layer_kind_all is not None
                    else None
                ),
            )
            thermal_arrays = {
                "h_um": np.concatenate(([lower_eq["h_um"]], h_all[incoming_indices])),
                "E_GPa": np.concatenate(([lower_eq["E_GPa"]], E_all[incoming_indices])),
                "nu": np.concatenate(([lower_eq["nu"]], nu_all[incoming_indices])),
                "alpha_ppm_K": np.concatenate(([lower_eq["alpha_ppm_K"]], alpha_all[incoming_indices])),
            }
            if layer_kind_all is not None:
                thermal_arrays["thermal_modulus_kind"] = np.concatenate((
                    np.asarray(["equivalent_lower_stack"], dtype=object),
                    layer_kind_all[incoming_indices],
                ))
        elif thermal_state_mode == "per_group_substrate_equivalent":
            thermal_arrays = _per_group_substrate_equivalent_thermal_arrays(
                h_all,
                E_all,
                nu_all,
                alpha_all,
                group_masks,
                group_position,
                layer_kind_all=layer_kind_all,
            )
            step_thermal_modulus_mode = "plate"
        elif thermal_state_mode == "full_active_stack":
            thermal_arrays = {
                "h_um": h_all[active_indices],
                "E_GPa": E_all[active_indices],
                "nu": nu_all[active_indices],
                "alpha_ppm_K": alpha_all[active_indices],
            }
            if layer_kind_all is not None:
                thermal_arrays["thermal_modulus_kind"] = layer_kind_all[active_indices]
        elif thermal_state_mode == "recent_2_groups_plus_collapsed_older":
            old_group_end = max(0, group_position - 2)
            thermal_parts = {
                "h_um": [],
                "E_GPa": [],
                "nu": [],
                "alpha_ppm_K": [],
            }
            if layer_kind_all is not None:
                thermal_parts["thermal_modulus_kind"] = []
            if old_group_end > 0:
                old_mask = np.logical_or.reduce(group_masks[:old_group_end])
                old_indices = np.nonzero(old_mask)[0]
                old_eq = _equivalent_thermal_arrays(
                    h_all[old_indices],
                    E_all[old_indices],
                    nu_all[old_indices],
                    alpha_all[old_indices],
                    thermal_modulus_mode=thermal_modulus_mode,
                    layer_kind=(
                        layer_kind_all[old_indices]
                        if layer_kind_all is not None
                        else None
                    ),
                )
                for key in ("h_um", "E_GPa", "nu", "alpha_ppm_K"):
                    thermal_parts[key].append(old_eq[key])
                if layer_kind_all is not None:
                    thermal_parts["thermal_modulus_kind"].append("equivalent_lower_stack")
            recent_masks = group_masks[old_group_end:group_position] + [incoming_mask]
            recent_indices = np.nonzero(np.logical_or.reduce(recent_masks))[0]
            for idx in recent_indices:
                thermal_parts["h_um"].append(h_all[idx])
                thermal_parts["E_GPa"].append(E_all[idx])
                thermal_parts["nu"].append(nu_all[idx])
                thermal_parts["alpha_ppm_K"].append(alpha_all[idx])
                if layer_kind_all is not None:
                    thermal_parts["thermal_modulus_kind"].append(layer_kind_all[idx])
            thermal_arrays = {
                key: np.asarray(values, dtype=float)
                for key, values in thermal_parts.items()
                if key != "thermal_modulus_kind"
            }
            if layer_kind_all is not None:
                thermal_arrays["thermal_modulus_kind"] = np.asarray(
                    thermal_parts["thermal_modulus_kind"],
                    dtype=object,
                )
        else:
            raise ValueError(f"Unknown thermal_state_mode: {thermal_state_mode}")

        nominal_delta_t = float(delta_t_values[group_position - 1])
        if thermal_delta_mode == "w2w_incremental":
            step_delta_t = _w2w_incremental_thermal_delta_t(nominal_delta_t)
        elif thermal_delta_mode == "nominal_cooldown":
            step_delta_t = nominal_delta_t
        elif thermal_delta_mode == "room_return_zero":
            step_delta_t = 0.0
        else:
            raise ValueError(f"Unknown thermal_delta_mode: {thermal_delta_mode}")
        summary = _compute_total_stack_warpage_arrays(
            h_all[active_indices],
            E_all[active_indices],
            nu_all[active_indices],
            alpha_all[active_indices],
            W0_current,
            DeltaT_K=step_delta_t,
            L_m=L_m,
            thermal_arrays=thermal_arrays,
            thermal_modulus_mode=step_thermal_modulus_mode,
            layer_kind=(
                layer_kind_all[active_indices]
                if layer_kind_all is not None
                else None
            ),
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
            "lower_state_w0_mode": lower_state_mode,
            "thermal_state": thermal_state_mode,
            "thermal_delta_mode": thermal_delta_mode,
            "thermal_modulus_mode": step_thermal_modulus_mode,
            "stress_history_selector_reason": selector_metadata["reason"],
            "group_thickness_relative_span": float(selector_metadata["group_thickness_relative_span"]),
            "group_thickness_index_correlation": float(selector_metadata["group_thickness_index_correlation"]),
            "group_E_relative_span": float(selector_metadata["group_E_relative_span"]),
            "group_alpha_relative_span": float(selector_metadata["group_alpha_relative_span"]),
            "group_E_alpha_correlation": float(selector_metadata["group_E_alpha_correlation"]),
            "interface_alpha_span_ppm_K": float(selector_metadata["interface_alpha_span_ppm_K"]),
            "residual_strain_carry": False,
        })

    if step_summaries:
        final_summary = {
            "signed_W_um": float(step_summaries[-1]["signed_W_um"]),
            "abs_W_um": float(step_summaries[-1]["abs_W_um"]),
            "kappa_1_per_m": float(step_summaries[-1]["kappa_1_per_m"]),
            "DeltaT_K": float(step_summaries[-1]["DeltaT_K"]),
            "L_m": float(step_summaries[-1]["L_m"]),
            "lower_state_w0_mode": lower_state_mode,
            "thermal_state": thermal_state_mode,
            "thermal_delta_mode": thermal_delta_mode,
            "thermal_modulus_mode": step_summaries[-1]["thermal_modulus_mode"],
            "stress_history_selector_reason": selector_metadata["reason"],
            "group_thickness_relative_span": float(selector_metadata["group_thickness_relative_span"]),
            "group_thickness_index_correlation": float(selector_metadata["group_thickness_index_correlation"]),
            "group_E_relative_span": float(selector_metadata["group_E_relative_span"]),
            "group_alpha_relative_span": float(selector_metadata["group_alpha_relative_span"]),
            "group_E_alpha_correlation": float(selector_metadata["group_E_alpha_correlation"]),
            "interface_alpha_span_ppm_K": float(selector_metadata["interface_alpha_span_ppm_K"]),
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
            "lower_state_w0_mode": lower_state_mode,
            "thermal_state": thermal_state_mode,
            "thermal_delta_mode": thermal_delta_mode,
            "thermal_modulus_mode": thermal_modulus_mode,
            "stress_history_selector_reason": selector_metadata["reason"],
            "group_thickness_relative_span": float(selector_metadata["group_thickness_relative_span"]),
            "group_thickness_index_correlation": float(selector_metadata["group_thickness_index_correlation"]),
            "group_E_relative_span": float(selector_metadata["group_E_relative_span"]),
            "group_alpha_relative_span": float(selector_metadata["group_alpha_relative_span"]),
            "group_E_alpha_correlation": float(selector_metadata["group_E_alpha_correlation"]),
            "interface_alpha_span_ppm_K": float(selector_metadata["interface_alpha_span_ppm_K"]),
            "residual_strain_carry": False,
        }
    return final_summary, step_summaries


def _compute_sequential_stack_warpage_grouped_residual_strain_state(
    h_all,
    E_all,
    nu_all,
    alpha_all,
    W0_all,
    groups,
    group_masks,
    delta_t_values,
    L_m,
    *,
    selector_metadata,
    layer_kind_all=None,
    thermal_modulus_mode="plate",
):
    """
    Exp04 stress-history branch for through-stack interface material gradients.

    Hybrid-bond Cu-ratio ramps store localized residual strain in very thin
    interface sublayers.  A scalar carried bow is not enough state for those
    cases, so carry the analytical residual strain field from each solved
    partial stack.  The geometry/process assumptions remain the exp04 ones:
    original layer bows, full active thermal stack, and nominal cooldown.
    """
    carried_W_um = float(W0_all[group_masks[0]][0])
    residual_strain_all = np.zeros(len(W0_all), dtype=float)
    if layer_kind_all is not None:
        layer_kind_all = np.asarray(layer_kind_all, dtype=object)
        if layer_kind_all.size != np.asarray(h_all).size:
            raise ValueError("layer_kind_all must have one entry per physical layer.")
    step_summaries = []

    for group_position in range(1, len(groups)):
        active_mask = np.logical_or.reduce(group_masks[: group_position + 1])
        active_indices = np.nonzero(active_mask)[0]

        thermal_arrays = {
            "h_um": h_all[active_indices],
            "E_GPa": E_all[active_indices],
            "nu": nu_all[active_indices],
            "alpha_ppm_K": alpha_all[active_indices],
        }
        if layer_kind_all is not None:
            thermal_arrays["thermal_modulus_kind"] = layer_kind_all[active_indices]

        nominal_delta_t = float(delta_t_values[group_position - 1])
        step_delta_t = nominal_delta_t
        summary, residual_strain_current = _compute_total_stack_warpage_arrays_with_residual_strain(
            h_all[active_indices],
            E_all[active_indices],
            nu_all[active_indices],
            alpha_all[active_indices],
            W0_all[active_indices],
            residual_strain_all[active_indices],
            DeltaT_K=step_delta_t,
            L_m=L_m,
            thermal_arrays=thermal_arrays,
            thermal_modulus_mode=thermal_modulus_mode,
            layer_kind=(
                layer_kind_all[active_indices]
                if layer_kind_all is not None
                else None
            ),
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
            "lower_state_w0_mode": "original_layer_bow",
            "thermal_state": "full_active_stack",
            "thermal_delta_mode": "nominal_cooldown",
            "thermal_modulus_mode": thermal_modulus_mode,
            "stress_history_selector_reason": selector_metadata["reason"],
            "stress_history_state_transfer": "interface_gradient_residual_strain",
            "group_thickness_relative_span": float(selector_metadata["group_thickness_relative_span"]),
            "group_thickness_index_correlation": float(selector_metadata["group_thickness_index_correlation"]),
            "group_E_relative_span": float(selector_metadata["group_E_relative_span"]),
            "group_alpha_relative_span": float(selector_metadata["group_alpha_relative_span"]),
            "group_E_alpha_correlation": float(selector_metadata["group_E_alpha_correlation"]),
            "interface_alpha_span_ppm_K": float(selector_metadata["interface_alpha_span_ppm_K"]),
            "residual_strain_carry": True,
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
            "lower_state_w0_mode": "original_layer_bow",
            "thermal_state": "full_active_stack",
            "thermal_delta_mode": "nominal_cooldown",
            "thermal_modulus_mode": thermal_modulus_mode,
            "stress_history_selector_reason": selector_metadata["reason"],
            "stress_history_state_transfer": "interface_gradient_residual_strain",
            "group_thickness_relative_span": float(selector_metadata["group_thickness_relative_span"]),
            "group_thickness_index_correlation": float(selector_metadata["group_thickness_index_correlation"]),
            "group_E_relative_span": float(selector_metadata["group_E_relative_span"]),
            "group_alpha_relative_span": float(selector_metadata["group_alpha_relative_span"]),
            "group_E_alpha_correlation": float(selector_metadata["group_E_alpha_correlation"]),
            "interface_alpha_span_ppm_K": float(selector_metadata["interface_alpha_span_ppm_K"]),
            "residual_strain_carry": True,
            "abd_state_residual_generalized_moment_N": float(
                step_summaries[-1]["abd_state_residual_generalized_moment_N"]
            ),
            "abd_state_residual_mean_strain": float(
                step_summaries[-1]["abd_state_residual_mean_strain"]
            ),
            "abd_state_residual_rms_strain": float(
                step_summaries[-1]["abd_state_residual_rms_strain"]
            ),
            "abd_state_denominator_N_m": float(
                step_summaries[-1]["abd_state_denominator_N_m"]
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
            "lower_state_w0_mode": "original_layer_bow",
            "thermal_state": "full_active_stack",
            "thermal_delta_mode": "nominal_cooldown",
            "thermal_modulus_mode": thermal_modulus_mode,
            "stress_history_selector_reason": selector_metadata["reason"],
            "stress_history_state_transfer": "interface_gradient_residual_strain",
            "group_thickness_relative_span": float(selector_metadata["group_thickness_relative_span"]),
            "group_thickness_index_correlation": float(selector_metadata["group_thickness_index_correlation"]),
            "group_E_relative_span": float(selector_metadata["group_E_relative_span"]),
            "group_alpha_relative_span": float(selector_metadata["group_alpha_relative_span"]),
            "group_E_alpha_correlation": float(selector_metadata["group_E_alpha_correlation"]),
            "interface_alpha_span_ppm_K": float(selector_metadata["interface_alpha_span_ppm_K"]),
            "residual_strain_carry": True,
            "abd_state_residual_generalized_moment_N": 0.0,
            "abd_state_residual_mean_strain": 0.0,
            "abd_state_residual_rms_strain": 0.0,
            "abd_state_denominator_N_m": 0.0,
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

    - choose how lower-stack bow is transferred from the physical input state;
    - choose whether the lower-stack thermal state should be collapsed to one
      equivalent layer or kept as the full active layer stack.

    No fitted scale, pattern gate, or empirical memory coefficient is used. The
    selector is based only on the input physical-property gradients.
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
    layer_kind_all = None
    thermal_modulus_mode = "plate"
    for kind_column in ("thermal_modulus_kind", "sublayer"):
        if kind_column in df.columns:
            layer_kind_all = df[kind_column].to_numpy(dtype=object)
            thermal_modulus_mode = "w2w_mixed"
            break
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
    selector = _select_grouped_stress_history_state(h_all, E_all, nu_all, alpha_all, group_masks)
    if layer_kind_all is not None:
        selector = dict(selector)
        selector["thermal_state_mode"] = "per_group_substrate_equivalent"
        # The explicit sandwich MAPDL process bonds at room temperature,
        # heats to anneal, then cools back to TROOM with TREF=TROOM.  In a
        # linear elastic release, the final thermal strain from that cycle is
        # therefore zero; remaining bow comes from the input wafer bows and
        # geometric/material stiffness distribution.
        selector["thermal_delta_mode"] = "room_return_zero"
    if selector["reason"] == "interface_material_gradient" and layer_kind_all is None:
        return _compute_sequential_stack_warpage_grouped_residual_strain_state(
            h_all,
            E_all,
            nu_all,
            alpha_all,
            W0_all,
            groups,
            group_masks,
            delta_t_values,
            L_m,
            selector_metadata=selector,
            layer_kind_all=layer_kind_all,
            thermal_modulus_mode="plate",
        )
    return _compute_sequential_stack_warpage_grouped_state_mode(
        h_all,
        E_all,
        nu_all,
        alpha_all,
        W0_all,
        groups,
        group_masks,
        delta_t_values,
        L_m,
        lower_state_mode=selector["lower_state_mode"],
        thermal_state_mode=selector["thermal_state_mode"],
        thermal_delta_mode=selector["thermal_delta_mode"],
        selector_metadata=selector,
        layer_kind_all=layer_kind_all,
        thermal_modulus_mode=thermal_modulus_mode,
    )


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


# Room-temperature bonding variants.
def _validated_w2w_inputs(
    layer_df: pd.DataFrame,
    group_ids,
) -> tuple[pd.DataFrame, pd.Series, list[int]]:
    df = layer_df.copy().reset_index(drop=True)
    groups = pd.Series(group_ids).astype(int).reset_index(drop=True)
    if len(groups) != len(df):
        raise ValueError("group_ids must have the same length as layer_df.")
    if "die_index" not in df.columns:
        df["die_index"] = groups
    group_values = sorted(int(value) for value in pd.unique(groups))
    if group_values != list(range(len(group_values))):
        raise ValueError("W2W RT model expects zero-based contiguous wafer group IDs.")
    return df, groups, group_values


def compute_w2w_roomtemp_compacted_stack_bow_history(
    layer_df: pd.DataFrame,
    *,
    group_ids,
    L_m: float,
    temperature_C: float = 25.0,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Return W2W RT-only sequential released-bow history.

    The released bow of the completed lower stack is carried into the next
    room-temperature bonding step.  Since no thermal path is applied, the only
    driving term is the incoming wafer/stack initial bow geometry.
    """

    df, groups, group_values = _validated_w2w_inputs(layer_df, group_ids)

    carried_W_um = float(df.loc[groups == 0, "W0_um"].iloc[0])
    history_rows: list[dict[str, float | int | str]] = []

    for incoming_position in range(1, len(group_values)):
        active_mask = groups.to_numpy(dtype=int) <= incoming_position
        active = df.loc[active_mask].copy().reset_index(drop=True)
        lower_mask = active["die_index"].to_numpy(dtype=int) < incoming_position
        if np.any(lower_mask):
            active.loc[lower_mask, "W0_um"] = carried_W_um
        summary, _ = compute_total_stack_warpage(active, DeltaT_K=0.0, L_m=L_m)
        carried_W_um = float(summary["signed_W_um"])
        history_rows.append(
            {
                "model_step_index": incoming_position - 1,
                "bond_cycle": incoming_position,
                "layer_count": incoming_position + 1,
                "model_stage": "roomtemp_released",
                "temperature_C": float(temperature_C),
                "model_stack_edge_um": carried_W_um,
                "model_kappa_1_per_m": float(summary["kappa_1_per_m"]),
            }
        )

    if history_rows:
        final = history_rows[-1]
        final_edge = float(final["model_stack_edge_um"])
        final_kappa = float(final["model_kappa_1_per_m"])
    else:
        final_edge = carried_W_um
        final_kappa = float(2.0 * (carried_W_um * 1.0e-6) / (float(L_m) ** 2))

    summary = {
        "roomtemp_stack_bow_um": float(final_edge),
        "roomtemp_abs_stack_bow_um": float(abs(final_edge)),
        "kappa_1_per_m": float(final_kappa),
        "bond_cycles": max(len(group_values) - 1, 0),
        "model_mode": "compacted_lower_stack",
    }
    return summary, pd.DataFrame(history_rows)


def compute_w2w_roomtemp_reflatten_stack_bow_history(
    layer_df: pd.DataFrame,
    *,
    group_ids,
    L_m: float,
    temperature_C: float = 25.0,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    """
    Return W2W RT-only bow history for repeated original-bow re-flattening.

    This matches the exp07 MAPDL sequence: previous dielectric bonds remain
    active, but every active wafer is constrained back to the bonding plane
    from its own original room-temperature bow before the next interface is
    tied.  No anneal or thermal load is applied.
    """

    df, groups, group_values = _validated_w2w_inputs(layer_df, group_ids)

    history_rows: list[dict[str, float | int | str]] = []
    final_edge = float(df.loc[groups == 0, "W0_um"].iloc[0])
    final_kappa = float(2.0 * (final_edge * 1.0e-6) / (float(L_m) ** 2))

    for incoming_position in range(1, len(group_values)):
        active_mask = groups.to_numpy(dtype=int) <= incoming_position
        active = df.loc[active_mask].copy().reset_index(drop=True)
        summary, _ = compute_total_stack_warpage(active, DeltaT_K=0.0, L_m=L_m)
        final_edge = float(summary["signed_W_um"])
        final_kappa = float(summary["kappa_1_per_m"])
        history_rows.append(
            {
                "model_step_index": incoming_position - 1,
                "bond_cycle": incoming_position,
                "layer_count": incoming_position + 1,
                "model_stage": "roomtemp_released",
                "temperature_C": float(temperature_C),
                "model_stack_edge_um": final_edge,
                "model_kappa_1_per_m": final_kappa,
            }
        )

    summary = {
        "roomtemp_stack_bow_um": float(final_edge),
        "roomtemp_abs_stack_bow_um": float(abs(final_edge)),
        "kappa_1_per_m": float(final_kappa),
        "bond_cycles": max(len(group_values) - 1, 0),
        "model_mode": "reflatten_original_bows",
    }
    return summary, pd.DataFrame(history_rows)


def _group_bow_values(df: pd.DataFrame, groups: pd.Series, group_values: list[int]) -> np.ndarray:
    bows = []
    for group in group_values:
        group_bows = df.loc[groups == group, "W0_um"].to_numpy(dtype=float)
        if group_bows.size == 0:
            raise ValueError(f"Missing W0_um values for wafer group {group}.")
        if not np.allclose(group_bows, group_bows[0], rtol=0.0, atol=1.0e-9):
            raise ValueError("All sublayers in one W2W wafer group must share W0_um.")
        bows.append(float(group_bows[0]))
    return np.asarray(bows, dtype=float)


def _group_extensional_stiffnesses(df: pd.DataFrame, groups: pd.Series, group_values: list[int]) -> np.ndarray:
    """Return each wafer group's in-plane extensional stiffness per unit width."""

    h_m = df["h_um"].to_numpy(dtype=float) * 1.0e-6
    E_pa = df["E_GPa"].to_numpy(dtype=float) * 1.0e9
    nu = df["nu"].to_numpy(dtype=float)
    if np.any(h_m <= 0.0):
        raise ValueError("All layer thicknesses must be positive.")
    if np.any((nu <= -1.0) | (nu >= 0.5)):
        raise ValueError("Poisson ratios should be in the elastic range (-1, 0.5).")
    row_groups = groups.to_numpy(dtype=int)
    q = E_pa / (1.0 - nu)
    values = []
    for group in group_values:
        mask = row_groups == group
        if not np.any(mask):
            raise ValueError(f"Missing layers for wafer group {group}.")
        values.append(float(np.sum(q[mask] * h_m[mask])))
    return np.asarray(values, dtype=float)


def compute_w2w_roomtemp_extensional_average_bow_history(
    layer_df: pd.DataFrame,
    *,
    group_ids,
    L_m: float,
    temperature_C: float = 25.0,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    """
    Return W2W RT bow from wafer extensional-stiffness compatibility.

    In the MAPDL RT bonding sequence, every active wafer is flattened into the
    bonding plane, the interface is tied, and the stack is released.  With no
    thermal load, the released common bow is governed primarily by in-plane
    force compatibility among the flattened wafers.  For a bonded active stack,
    this gives the extensional-stiffness-weighted average of the individual
    wafer bows:

        W_stack = sum(A_g W_g) / sum(A_g)

    where A_g = sum_i E_i/(1-nu_i) * h_i over the sublayers of wafer g.
    This is exact for uniform stacks and captures the asymmetric bilayer
    transfer coefficient without empirical family offsets or fitted gains.
    """

    df, groups, group_values = _validated_w2w_inputs(layer_df, group_ids)
    history_rows: list[dict[str, float | int | str]] = []
    final_edge = float(df.loc[groups == 0, "W0_um"].iloc[0])

    for incoming_position in range(1, len(group_values)):
        active_values = group_values[: incoming_position + 1]
        active_mask = groups.isin(active_values).to_numpy()
        active = df.loc[active_mask].copy().reset_index(drop=True)
        active_groups = active["die_index"].astype(int).reset_index(drop=True)
        active_group_values = sorted(int(value) for value in pd.unique(active_groups))
        bows = _group_bow_values(active, active_groups, active_group_values)
        stiffness = _group_extensional_stiffnesses(active, active_groups, active_group_values)
        final_edge = float(np.sum(stiffness * bows) / np.sum(stiffness))
        history_rows.append(
            {
                "model_step_index": incoming_position - 1,
                "bond_cycle": incoming_position,
                "layer_count": incoming_position + 1,
                "model_stage": "roomtemp_released",
                "temperature_C": float(temperature_C),
                "model_stack_edge_um": final_edge,
                "model_kappa_1_per_m": float(2.0 * (final_edge * 1.0e-6) / (float(L_m) ** 2)),
                "model_mode": "extensional_average",
            }
        )

    final_kappa = float(2.0 * (final_edge * 1.0e-6) / (float(L_m) ** 2))
    summary = {
        "roomtemp_stack_bow_um": float(final_edge),
        "roomtemp_abs_stack_bow_um": float(abs(final_edge)),
        "kappa_1_per_m": final_kappa,
        "bond_cycles": max(len(group_values) - 1, 0),
        "model_mode": "extensional_average",
    }
    return summary, pd.DataFrame(history_rows)


def _group_centroid_coordinates(df: pd.DataFrame, groups: pd.Series, group_values: list[int]) -> np.ndarray:
    """Return physical through-stack centroid coordinates for wafer groups."""

    h_um = df["h_um"].to_numpy(dtype=float)
    row_groups = groups.to_numpy(dtype=int)
    row_bottom_um = np.concatenate(([0.0], np.cumsum(h_um[:-1])))
    row_centroid_um = row_bottom_um + 0.5 * h_um
    centroids = []
    for group in group_values:
        mask = row_groups == group
        if not np.any(mask):
            raise ValueError(f"Missing layers for wafer group {group}.")
        centroids.append(float(np.average(row_centroid_um[mask], weights=h_um[mask])))
    values = np.asarray(centroids, dtype=float)
    span = float(np.max(values) - np.min(values))
    if span <= 0.0:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / (0.5 * span)


def _replace_group_bows(
    df: pd.DataFrame,
    groups: pd.Series,
    group_values: list[int],
    bow_values_um: np.ndarray,
) -> pd.DataFrame:
    out = df.copy()
    for group, bow_um in zip(group_values, bow_values_um):
        out.loc[groups == group, "W0_um"] = float(bow_um)
    return out


def _split_centroid_linear_bow_modes(
    df: pd.DataFrame,
    groups: pd.Series,
    group_values: list[int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Split wafer bows into centroid-linear and higher-order through-stack modes.

    The projection basis is fixed by geometry: constant plus physical wafer
    centroid coordinate.  No ANSYS result, fitted gain, or empirical family
    selector enters this split.
    """

    bows = _group_bow_values(df, groups, group_values)
    if len(group_values) < 3:
        return bows.copy(), np.zeros_like(bows), 0.0
    x = _group_centroid_coordinates(df, groups, group_values)
    bow_mean = float(np.mean(bows))
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        low_order = np.full_like(bows, bow_mean)
    else:
        slope = float(np.dot(x, bows - bow_mean) / denom)
        low_order = bow_mean + slope * x
    high_order = bows - low_order
    high_order_rms = float(np.sqrt(np.mean(high_order**2)))
    return low_order, high_order, high_order_rms


def compute_w2w_roomtemp_through_stack_modal_bow_history(
    layer_df: pd.DataFrame,
    *,
    group_ids,
    L_m: float,
    temperature_C: float = 25.0,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    """
    Return W2W RT bow using a non-fitted through-stack modal state split.

    The low-order wafer-bow component is the orthogonal projection onto
    constant plus physical wafer-centroid coordinate.  That component follows
    the re-flattened original-bow model, which is exact for uniform and ramp
    bow families.  The remaining higher-order bow component is propagated with
    the compacted lower-stack state, representing the part of the bonded-stack
    shape that is not described by a global centroid-linear wafer trend.
    Linearity of the elastic RT problem allows the two predictions to be
    superposed.  The split has no fitted parameters.
    """

    df, groups, group_values = _validated_w2w_inputs(layer_df, group_ids)

    history_rows: list[dict[str, float | int | str]] = []
    final_edge = float(df.loc[groups == 0, "W0_um"].iloc[0])
    final_low_order = final_edge
    final_high_order = 0.0
    final_high_order_rms = 0.0

    for incoming_position in range(1, len(group_values)):
        active_mask = groups.to_numpy(dtype=int) <= incoming_position
        active = df.loc[active_mask].copy().reset_index(drop=True)
        active_groups = active["die_index"].astype(int).reset_index(drop=True)
        active_group_values = sorted(int(value) for value in pd.unique(active_groups))
        low_order_bows, high_order_bows, high_order_rms = _split_centroid_linear_bow_modes(
            active,
            active_groups,
            active_group_values,
        )
        low_order_df = _replace_group_bows(
            active,
            active_groups,
            active_group_values,
            low_order_bows,
        )
        high_order_df = _replace_group_bows(
            active,
            active_groups,
            active_group_values,
            high_order_bows,
        )

        low_summary, _ = compute_w2w_roomtemp_reflatten_stack_bow_history(
            low_order_df,
            group_ids=active_groups,
            L_m=L_m,
            temperature_C=temperature_C,
        )
        high_summary, _ = compute_w2w_roomtemp_compacted_stack_bow_history(
            high_order_df,
            group_ids=active_groups,
            L_m=L_m,
            temperature_C=temperature_C,
        )
        final_low_order = float(low_summary["roomtemp_stack_bow_um"])
        final_high_order = float(high_summary["roomtemp_stack_bow_um"])
        final_high_order_rms = float(high_order_rms)
        final_edge = final_low_order + final_high_order
        history_rows.append(
            {
                "model_step_index": incoming_position - 1,
                "bond_cycle": incoming_position,
                "layer_count": incoming_position + 1,
                "model_stage": "roomtemp_released",
                "temperature_C": float(temperature_C),
                "model_stack_edge_um": float(final_edge),
                "model_low_order_edge_um": float(final_low_order),
                "model_high_order_edge_um": float(final_high_order),
                "model_high_order_rms_um": float(final_high_order_rms),
                "model_kappa_1_per_m": float(2.0 * (final_edge * 1.0e-6) / (float(L_m) ** 2)),
            }
        )

    final_kappa = float(2.0 * (final_edge * 1.0e-6) / (float(L_m) ** 2))
    summary = {
        "roomtemp_stack_bow_um": float(final_edge),
        "roomtemp_abs_stack_bow_um": float(abs(final_edge)),
        "roomtemp_low_order_stack_bow_um": float(final_low_order),
        "roomtemp_high_order_stack_bow_um": float(final_high_order),
        "roomtemp_high_order_rms_um": float(final_high_order_rms),
        "kappa_1_per_m": final_kappa,
        "bond_cycles": max(len(group_values) - 1, 0),
        "model_mode": "through_stack_modal_state",
    }
    return summary, pd.DataFrame(history_rows)


def compute_w2w_roomtemp_stack_bow_history(
    layer_df: pd.DataFrame,
    *,
    group_ids,
    L_m: float,
    temperature_C: float = 25.0,
    model_mode: str = "extensional_average",
) -> tuple[dict[str, float | str], pd.DataFrame]:
    """Return W2W RT-only bow history using the requested process model."""

    if model_mode == "extensional_average":
        return compute_w2w_roomtemp_extensional_average_bow_history(
            layer_df,
            group_ids=group_ids,
            L_m=L_m,
            temperature_C=temperature_C,
        )
    if model_mode == "through_stack_modal_state":
        return compute_w2w_roomtemp_through_stack_modal_bow_history(
            layer_df,
            group_ids=group_ids,
            L_m=L_m,
            temperature_C=temperature_C,
        )
    if model_mode == "reflatten_original_bows":
        return compute_w2w_roomtemp_reflatten_stack_bow_history(
            layer_df,
            group_ids=group_ids,
            L_m=L_m,
            temperature_C=temperature_C,
        )
    if model_mode == "compacted_lower_stack":
        return compute_w2w_roomtemp_compacted_stack_bow_history(
            layer_df,
            group_ids=group_ids,
            L_m=L_m,
            temperature_C=temperature_C,
        )
    raise ValueError(
        "Unknown W2W RT model_mode. Expected 'extensional_average', "
        "'through_stack_modal_state', 'reflatten_original_bows', or "
        "'compacted_lower_stack'."
    )


# Cu elastic-plastic repeated-anneal variant.
D2W_DIR = Path(__file__).resolve().parents[1] / "D2W"
YAP_ROOT = D2W_DIR.parent
_added_yap_root = str(YAP_ROOT) not in sys.path
if _added_yap_root:
    sys.path.append(str(YAP_ROOT))
D2W_WARPAGE_PATH = D2W_DIR / "warpage_yield_calculator.py"

_spec = importlib.util.spec_from_file_location("d2w_warpage_calculator_shared", D2W_WARPAGE_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load D2W warpage calculator from {D2W_WARPAGE_PATH}")
_d2w_warpage = importlib.util.module_from_spec(_spec)
sys.modules["d2w_warpage_calculator_shared"] = _d2w_warpage
try:
    _spec.loader.exec_module(_d2w_warpage)
finally:
    if _added_yap_root:
        sys.path.remove(str(YAP_ROOT))

CuEpParams = _d2w_warpage.CuEpParams
compute_stack_warpage_with_eigenstrain = _d2w_warpage.compute_stack_warpage_with_eigenstrain


def _cu_volume_fraction(layer_df: pd.DataFrame) -> np.ndarray:
    for column in ("interface_cu_volume_fraction", "Cu_V", "cu_vf"):
        if column in layer_df.columns:
            values = pd.to_numeric(layer_df[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            return np.clip(values, 0.0, 1.0)
    if "sublayer" in layer_df.columns:
        sublayer = layer_df["sublayer"].astype(str).str.lower()
        return np.where(sublayer.str.contains("cu").to_numpy(), 0.10, 0.0)
    return np.zeros(len(layer_df), dtype=float)


def _yield_at_temperature(T_C: float, params: CuEpParams, t_room_C: float) -> float:
    if abs(float(params.anneal_temperature_C) - float(t_room_C)) < 1.0e-12:
        return float(params.yield_room_MPa)
    fraction = (float(T_C) - float(t_room_C)) / (float(params.anneal_temperature_C) - float(t_room_C))
    fraction = min(1.0, max(0.0, fraction))
    return float(params.yield_room_MPa) + fraction * (
        float(params.yield_anneal_MPa) - float(params.yield_room_MPa)
    )


def _temperature_schedule(
    *,
    t_room_C: float,
    t_anneal_C: float,
    heat_steps: int,
    cool_steps: int,
    include_anneal_hold: bool,
) -> list[tuple[str, float]]:
    schedule: list[tuple[str, float]] = []
    for value in np.linspace(float(t_room_C), float(t_anneal_C), max(1, int(heat_steps)) + 1)[1:]:
        stage = "anneal_peak" if abs(float(value) - float(t_anneal_C)) < 1.0e-9 else "heat_to_anneal"
        schedule.append((stage, float(value)))
    if include_anneal_hold:
        schedule.append(("anneal_hold", float(t_anneal_C)))
    for value in np.linspace(float(t_anneal_C), float(t_room_C), max(1, int(cool_steps)) + 1)[1:]:
        stage = "post_anneal_rt" if abs(float(value) - float(t_room_C)) < 1.0e-9 else "cool_to_room"
        schedule.append((stage, float(value)))
    return schedule


def _active_layer_frame(
    df: pd.DataFrame,
    group_values: pd.Series,
    *,
    incoming_position: int,
    carried_W_um: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    active_mask = group_values.to_numpy(dtype=int) <= int(incoming_position)
    active_indices = np.nonzero(active_mask)[0]
    active = df.iloc[active_indices].copy().reset_index(drop=True)
    lower_mask = active["die_index"].to_numpy(dtype=int) < int(incoming_position)
    if np.any(lower_mask):
        active.loc[lower_mask, "W0_um"] = float(carried_W_um)
    return active, active_indices


def _plastic_weight(active: pd.DataFrame, cu_vf: np.ndarray) -> np.ndarray:
    """Equivalent-interface plastic strain weight.

    The exp06 MAPDL calibration case shows that homogeneous Cu-interface
    plasticity produces negligible released room-temperature W2W bow shift.
    Keep the equivalent-interface plastic state as a diagnostic, but do not
    feed it back as a residual bending eigenstrain by default.
    """

    _ = active
    return np.zeros(len(cu_vf), dtype=float)


def _solve_active_with_plastic_state(
    active: pd.DataFrame,
    *,
    active_indices: np.ndarray,
    eps_p_full: np.ndarray,
    eq_p_full: np.ndarray,
    cu_vf_full: np.ndarray,
    L_m: float,
    T_C: float,
    t_room_C: float,
    params: CuEpParams,
) -> tuple[dict[str, float], pd.DataFrame]:
    active_cu_vf = cu_vf_full[active_indices]
    cu_weight = _plastic_weight(active, active_cu_vf)
    alpha = active["alpha_ppm_K"].to_numpy(dtype=float) * 1.0e-6
    is_cu_layer = active_cu_vf > 1.0e-12
    layer_biaxial_MPa = active["E_GPa"].to_numpy(dtype=float) * 1000.0 / (
        1.0 - active["nu"].to_numpy(dtype=float)
    )

    for _ in range(max(1, int(params.local_iterations))):
        eps_p = eps_p_full[active_indices]
        eigenstrain = alpha * (float(T_C) - float(t_room_C)) - cu_weight * eps_p
        summary, out = compute_stack_warpage_with_eigenstrain(active, L_m=L_m, eigenstrain=eigenstrain)
        if not np.any(is_cu_layer):
            break
        sigma_MPa = out["sigma_MPa"].to_numpy(dtype=float)
        yielded = False
        for local_idx in np.where(is_cu_layer)[0]:
            global_idx = int(active_indices[local_idx])
            yield_eff_MPa = active_cu_vf[local_idx] * _yield_at_temperature(T_C, params, t_room_C)
            hardening_eff_MPa = active_cu_vf[local_idx] * float(params.tangent_hardening_MPa)
            flow_MPa = max(1.0e-6, yield_eff_MPa) + max(1.0e-6, hardening_eff_MPa) * eq_p_full[global_idx]
            trial_MPa = sigma_MPa[local_idx]
            overstress = abs(trial_MPa) - flow_MPa
            if overstress <= float(params.plastic_tolerance_MPa):
                continue
            delta_gamma = overstress / (layer_biaxial_MPa[local_idx] + max(1.0e-6, hardening_eff_MPa))
            eps_p_full[global_idx] += np.sign(trial_MPa) * delta_gamma
            eq_p_full[global_idx] += delta_gamma
            yielded = True
        if not yielded:
            break

    eps_p = eps_p_full[active_indices]
    eigenstrain = alpha * (float(T_C) - float(t_room_C)) - cu_weight * eps_p
    summary, out = compute_stack_warpage_with_eigenstrain(active, L_m=L_m, eigenstrain=eigenstrain)
    out = out.copy()
    out["cu_volume_fraction"] = active_cu_vf
    out["cu_plastic_weight"] = cu_weight
    out["cu_plastic_strain"] = eps_p
    out["cu_equivalent_plastic_strain"] = eq_p_full[active_indices]
    return summary, out


def compute_w2w_repeated_anneal_cu_ep_history(
    layer_df: pd.DataFrame,
    *,
    group_ids,
    L_m: float,
    t_room_C: float,
    t_anneal_C: float,
    heat_steps: int = 30,
    cool_steps: int = 30,
    include_anneal_hold: bool = True,
    cu_params: CuEpParams | None = None,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """
    Compute exp06 W2W repeated-bond/repeated-anneal bow history.

    The lower completed stack is compacted to its released post-anneal bow for
    the next bonding cycle. Cu plastic strain remains attached to the physical
    Cu-bearing interface rows and is advanced during each active anneal cycle.
    """

    params = cu_params or CuEpParams(anneal_temperature_C=float(t_anneal_C))
    params = CuEpParams(
        yield_room_MPa=params.yield_room_MPa,
        yield_anneal_MPa=params.yield_anneal_MPa,
        tangent_hardening_MPa=params.tangent_hardening_MPa,
        anneal_temperature_C=float(t_anneal_C),
        plastic_tolerance_MPa=params.plastic_tolerance_MPa,
        local_iterations=params.local_iterations,
    )

    df = layer_df.copy().reset_index(drop=True)
    group_values = pd.Series(group_ids).astype(int).reset_index(drop=True)
    if len(group_values) != len(df):
        raise ValueError("group_ids must have the same length as layer_df.")
    if "die_index" not in df.columns:
        df["die_index"] = group_values
    required = ["h_um", "E_GPa", "nu", "alpha_ppm_K", "W0_um", "die_index"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    groups = sorted(int(value) for value in pd.unique(group_values))
    if groups != list(range(len(groups))):
        raise ValueError("exp06 expects zero-based contiguous wafer group IDs.")

    cu_vf_full = _cu_volume_fraction(df)
    eps_p_full = np.zeros(len(df), dtype=float)
    eq_p_full = np.zeros(len(df), dtype=float)
    carried_W_um = float(df.loc[group_values == 0, "W0_um"].iloc[0])

    schedule = _temperature_schedule(
        t_room_C=t_room_C,
        t_anneal_C=t_anneal_C,
        heat_steps=heat_steps,
        cool_steps=cool_steps,
        include_anneal_hold=include_anneal_hold,
    )

    history_rows: list[dict[str, object]] = []
    layer_rows: list[pd.DataFrame] = []
    cycle_summaries: list[dict[str, object]] = []
    step_index = 0

    def append_history(
        *,
        cycle: int,
        layer_count: int,
        stage: str,
        T_C: float,
        summary: dict[str, float],
        active_out: pd.DataFrame,
    ) -> None:
        nonlocal step_index
        row = {
            "model_step_index": int(step_index),
            "bond_cycle": int(cycle),
            "layer_count": int(layer_count),
            "model_stage": str(stage),
            "temperature_C": float(T_C),
            "model_stack_edge_um": float(summary["signed_W_um"]),
            "model_kappa_1_per_m": float(summary["kappa_1_per_m"]),
            "max_abs_cu_plastic_strain": float(np.max(np.abs(eps_p_full))) if len(eps_p_full) else 0.0,
            "max_cu_equivalent_plastic_strain": float(np.max(eq_p_full)) if len(eq_p_full) else 0.0,
            "active_cu_layer_count": int(np.count_nonzero(cu_vf_full > 1.0e-12)),
        }
        history_rows.append(row)
        out = active_out.copy()
        out["model_step_index"] = int(step_index)
        out["bond_cycle"] = int(cycle)
        out["layer_count"] = int(layer_count)
        out["model_stage"] = str(stage)
        out["temperature_C"] = float(T_C)
        layer_rows.append(out)
        step_index += 1

    for incoming_position in range(1, len(groups)):
        active, active_indices = _active_layer_frame(
            df,
            group_values,
            incoming_position=incoming_position,
            carried_W_um=carried_W_um,
        )
        layer_count = incoming_position + 1
        bond_summary, bond_out = _solve_active_with_plastic_state(
            active,
            active_indices=active_indices,
            eps_p_full=eps_p_full,
            eq_p_full=eq_p_full,
            cu_vf_full=cu_vf_full,
            L_m=L_m,
            T_C=float(t_room_C),
            t_room_C=float(t_room_C),
            params=params,
        )
        append_history(
            cycle=incoming_position,
            layer_count=layer_count,
            stage="after_bond_rt",
            T_C=float(t_room_C),
            summary=bond_summary,
            active_out=bond_out,
        )
        cycle = {
            "bond_cycle": int(incoming_position),
            "layer_count": int(layer_count),
            "after_bond_rt_edge_um": float(bond_summary["signed_W_um"]),
            "anneal_peak_edge_um": float("nan"),
            "post_anneal_rt_edge_um": float("nan"),
            "post_minus_bond_edge_um": float("nan"),
            "max_abs_cu_plastic_strain": float(np.max(np.abs(eps_p_full))) if len(eps_p_full) else 0.0,
        }

        post_summary = bond_summary
        post_out = bond_out
        for stage, T_C in schedule:
            summary, out = _solve_active_with_plastic_state(
                active,
                active_indices=active_indices,
                eps_p_full=eps_p_full,
                eq_p_full=eq_p_full,
                cu_vf_full=cu_vf_full,
                L_m=L_m,
                T_C=float(T_C),
                t_room_C=float(t_room_C),
                params=params,
            )
            append_history(
                cycle=incoming_position,
                layer_count=layer_count,
                stage=stage,
                T_C=float(T_C),
                summary=summary,
                active_out=out,
            )
            if stage in ("anneal_peak", "anneal_hold") and abs(float(T_C) - float(t_anneal_C)) < 1.0e-9:
                cycle["anneal_peak_edge_um"] = float(summary["signed_W_um"])
            if stage == "post_anneal_rt":
                post_summary = summary
                post_out = out
                cycle["post_anneal_rt_edge_um"] = float(summary["signed_W_um"])

        carried_W_um = float(post_summary["signed_W_um"])
        cycle["post_minus_bond_edge_um"] = carried_W_um - float(cycle["after_bond_rt_edge_um"])
        cycle["max_abs_cu_plastic_strain"] = float(np.max(np.abs(eps_p_full))) if len(eps_p_full) else 0.0
        cycle_summaries.append(cycle)

        # Keep the last layer output referenced so linters do not consider the
        # final post state unused; the data have already been appended.
        _ = post_out

    final_cycle = cycle_summaries[-1] if cycle_summaries else {
        "bond_cycle": 0,
        "layer_count": 1,
        "after_bond_rt_edge_um": carried_W_um,
        "anneal_peak_edge_um": carried_W_um,
        "post_anneal_rt_edge_um": carried_W_um,
        "post_minus_bond_edge_um": 0.0,
        "max_abs_cu_plastic_strain": 0.0,
    }
    summary = {
        "final_edge_um": float(final_cycle["post_anneal_rt_edge_um"]),
        "final_after_bond_rt_edge_um": float(final_cycle["after_bond_rt_edge_um"]),
        "final_post_minus_bond_edge_um": float(final_cycle["post_minus_bond_edge_um"]),
        "anneal_cycles": max(len(groups) - 1, 0),
        "max_abs_cu_plastic_strain": float(max(c["max_abs_cu_plastic_strain"] for c in cycle_summaries))
        if cycle_summaries
        else 0.0,
        "cycle_summaries": cycle_summaries,
    }
    history_df = pd.DataFrame(history_rows)
    layer_history_df = pd.concat(layer_rows, ignore_index=True) if layer_rows else pd.DataFrame()
    return summary, history_df, layer_history_df


def compute_w2w_repeated_anneal_cu_ep_final(
    layer_df: pd.DataFrame,
    *,
    group_ids,
    L_m: float,
    t_room_C: float,
    t_anneal_C: float,
    heat_steps: int = 30,
    cool_steps: int = 30,
    cu_params: CuEpParams | None = None,
) -> dict[str, float]:
    """Convenience wrapper returning only final exp06 values."""

    summary, _, _ = compute_w2w_repeated_anneal_cu_ep_history(
        layer_df,
        group_ids=group_ids,
        L_m=L_m,
        t_room_C=t_room_C,
        t_anneal_C=t_anneal_C,
        heat_steps=heat_steps,
        cool_steps=cool_steps,
        cu_params=cu_params,
    )
    return {
        "final_edge_um": float(summary["final_edge_um"]),
        "final_after_bond_rt_edge_um": float(summary["final_after_bond_rt_edge_um"]),
        "final_post_minus_bond_edge_um": float(summary["final_post_minus_bond_edge_um"]),
        "max_abs_cu_plastic_strain": float(summary["max_abs_cu_plastic_strain"]),
    }
