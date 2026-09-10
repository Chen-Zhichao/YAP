from __future__ import annotations

"""
W2W repeated-anneal warpage model with Cu elastic-plastic residual strain.

This module is intentionally separate from ``warpage_yield_calculator.py``.
It keeps the compacted-state W2W recurrence idea used by the current model, but
adds a Cu plastic strain state for exp06:

1. bond the incoming wafer at room temperature;
2. release the bonding tool and record the room-temperature stack bow;
3. anneal the active stack and cool back to room temperature;
4. carry the released post-anneal bow and Cu plastic state into the next bond.
"""

from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd


D2W_DIR = Path(__file__).resolve().parents[1] / "D2W"
D2W_CU_EP_PATH = D2W_DIR / "warpage_yield_calculator_cu_ep.py"
if str(D2W_DIR) not in sys.path:
    sys.path.insert(0, str(D2W_DIR))

_spec = importlib.util.spec_from_file_location("d2w_warpage_yield_calculator_cu_ep", D2W_CU_EP_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load D2W Cu EP helper from {D2W_CU_EP_PATH}")
_d2w_cu_ep = importlib.util.module_from_spec(_spec)
sys.modules["d2w_warpage_yield_calculator_cu_ep"] = _d2w_cu_ep
_spec.loader.exec_module(_d2w_cu_ep)

CuEpParams = _d2w_cu_ep.CuEpParams
compute_stack_warpage_with_eigenstrain = _d2w_cu_ep.compute_stack_warpage_with_eigenstrain


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
