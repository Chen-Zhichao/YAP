from __future__ import annotations

"""
W2W room-temperature stack-bow model.

This module is separate from the existing W2W anneal model.  It models a
sequential room-temperature bonding flow with no anneal stage: each incoming
wafer is flattened/bonded at room temperature and the active stack is released
at room temperature.
"""

import numpy as np
import pandas as pd

from warpage_yield_calculator import compute_total_stack_warpage


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
