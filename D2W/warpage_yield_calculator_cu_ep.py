from __future__ import annotations

"""
D2D room-temperature-bond warpage model with Cu elastic-plastic residual strain.

This module intentionally does not replace ``warpage_yield_calculator.py``.
It reuses the same Suhir/Kim-style multilayer curvature structure, but exposes
separate functions for exp05, where all dies are dielectric-bonded at room
temperature, annealed, then cooled back to room temperature.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from warpage_yield_calculator import bow_um_to_curvature, curvature_to_bow_um


CU_E_GPA = 91.8
CU_NU = 0.34
CU_ALPHA_PPM_K = 17.6


@dataclass(frozen=True)
class CuEpParams:
    """Scalar equibiaxial Cu plasticity parameters used by the reduced model."""

    yield_room_MPa: float = 220.0
    yield_anneal_MPa: float = 80.0
    tangent_hardening_MPa: float = 1000.0
    anneal_temperature_C: float = 300.0
    plastic_tolerance_MPa: float = 1.0e-6
    local_iterations: int = 4


def _required_arrays(layer_df: pd.DataFrame) -> dict[str, np.ndarray]:
    required = ["h_um", "E_GPa", "nu", "alpha_ppm_K", "W0_um"]
    missing = [column for column in required if column not in layer_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = layer_df.copy().reset_index(drop=True)
    arrays = {
        "h": df["h_um"].to_numpy(dtype=float) * 1.0e-6,
        "E": df["E_GPa"].to_numpy(dtype=float) * 1.0e9,
        "nu": df["nu"].to_numpy(dtype=float),
        "alpha": df["alpha_ppm_K"].to_numpy(dtype=float) * 1.0e-6,
        "kappa0": None,
    }
    arrays["kappa0"] = bow_um_to_curvature(df["W0_um"].to_numpy(dtype=float), 1.0)
    if np.any(arrays["h"] <= 0.0):
        raise ValueError("All layer thicknesses must be positive.")
    if np.any((arrays["nu"] <= -1.0) | (arrays["nu"] >= 0.5)):
        raise ValueError("Poisson ratios should be in the elastic range (-1, 0.5).")
    return arrays


def _cu_volume_fraction(layer_df: pd.DataFrame) -> np.ndarray:
    for column in ("interface_cu_volume_fraction", "Cu_V", "cu_vf"):
        if column in layer_df.columns:
            values = pd.to_numeric(layer_df[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            return np.clip(values, 0.0, 1.0)
    if "sublayer" in layer_df.columns:
        sublayer = layer_df["sublayer"].astype(str).str.lower()
        return np.where(sublayer.str.contains("cu").to_numpy(), 0.10, 0.0)
    return np.zeros(len(layer_df), dtype=float)


def compute_stack_warpage_with_eigenstrain(
    layer_df: pd.DataFrame,
    *,
    L_m: float,
    eigenstrain: np.ndarray | list[float] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Compute stack bow using the existing elastic curvature form plus eigenstrain.

    ``eigenstrain`` is the stress-free scalar strain carried by each analytical
    layer.  It can include thermal strain, plastic strain, or both.  The
    implementation follows the current D2D model's stiffness conventions:
    ``E/(1-nu)`` for mechanical compatibility and ``E/(1-nu^2)`` for the
    thermal/eigenstrain bending-force weight.
    """

    df = layer_df.copy().reset_index(drop=True)
    arrays = _required_arrays(df)
    h = arrays["h"]
    E = arrays["E"]
    nu = arrays["nu"]
    alpha = arrays["alpha"]
    kappa0 = bow_um_to_curvature(df["W0_um"].to_numpy(dtype=float), L_m)
    free = np.zeros(len(df), dtype=float) if eigenstrain is None else np.asarray(eigenstrain, dtype=float)
    if free.shape != h.shape:
        raise ValueError("eigenstrain length must match layer_df length.")

    E_biaxial = E / (1.0 - nu)
    E_thermal = E / (1.0 - nu**2)
    lam = 1.0 / (E_biaxial * h)
    weights = 1.0 / lam
    thermal_weights = E_thermal * h

    h_cumsum = np.cumsum(h)
    hk0_cumsum = np.cumsum(h * kappa0)
    a = h_cumsum - 0.5 * (h[0] + h)
    b = hk0_cumsum - 0.5 * (h[0] * kappa0[0] + h * kappa0)

    free_delta = free - free[0]
    s_free = float(np.sum(free_delta * thermal_weights) / np.sum(thermal_weights))
    free_centered = free_delta - s_free
    s_a = float(np.sum(a * weights) / np.sum(weights))
    s_b = float(np.sum(b * weights) / np.sum(weights))

    D = E_biaxial * h**3 / 12.0
    M_T = float(np.sum(a * thermal_weights * free_centered))
    M_b = float(np.sum((a / lam) * (b - s_b)))
    K_a = float(np.sum((a / lam) * (a - s_a)))
    K_D = float(np.sum(D))
    M_D0 = float(np.sum(D * kappa0))

    denominator = K_D + K_a
    kappa = (M_D0 + M_T + M_b) / denominator
    common_mechanical_strain = -(a - s_a) * kappa + (b - s_b)
    strain_term = free_centered + common_mechanical_strain
    sigma = E_biaxial * strain_term
    W_um = curvature_to_bow_um(kappa, L_m)

    summary = {
        "kappa_1_per_m": float(kappa),
        "signed_W_um": float(W_um),
        "abs_W_um": float(abs(W_um)),
        "L_m": float(L_m),
        "M_T_N": M_T,
        "M_b_N": M_b,
        "M_D0_N": M_D0,
        "K_a_N_m": K_a,
        "K_D_N_m": K_D,
        "s_free_strain": s_free,
        "s_a_m": s_a,
        "s_b": s_b,
    }
    layer_out = df.copy()
    layer_out["E_biaxial_GPa"] = E_biaxial / 1.0e9
    layer_out["E_thermal_plate_GPa"] = E_thermal / 1.0e9
    layer_out["kappa0_1_per_m"] = kappa0
    layer_out["eigenstrain"] = free
    layer_out["centered_eigenstrain"] = free_centered
    layer_out["common_mechanical_strain"] = common_mechanical_strain
    layer_out["sigma_MPa"] = sigma / 1.0e6
    return summary, layer_out


def compute_d2d_roomtemp_bond_pre_anneal(
    layer_df: pd.DataFrame,
    *,
    L_m: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Return the bonded/released room-temperature bow before annealing."""

    return compute_stack_warpage_with_eigenstrain(layer_df, L_m=L_m)


def _yield_at_temperature(T_C: float, params: CuEpParams) -> float:
    if params.anneal_temperature_C == 25.0:
        return float(params.yield_room_MPa)
    ratio = (float(T_C) - 25.0) / (float(params.anneal_temperature_C) - 25.0)
    ratio = min(1.0, max(0.0, ratio))
    return (1.0 - ratio) * float(params.yield_room_MPa) + ratio * float(params.yield_anneal_MPa)


def _temperature_schedule(
    *,
    t_room_C: float,
    t_anneal_C: float,
    heat_steps: int,
    cool_steps: int,
    include_anneal_hold: bool,
) -> list[tuple[str, float]]:
    heat_steps = max(1, int(heat_steps))
    cool_steps = max(1, int(cool_steps))
    schedule: list[tuple[str, float]] = [("pre_anneal_rt", float(t_room_C))]
    for value in np.linspace(float(t_room_C), float(t_anneal_C), heat_steps + 1)[1:]:
        schedule.append(("heat_to_anneal", float(value)))
    if include_anneal_hold:
        schedule.append(("anneal_hold", float(t_anneal_C)))
    for value in np.linspace(float(t_anneal_C), float(t_room_C), cool_steps + 1)[1:]:
        label = "post_anneal_rt" if abs(float(value) - float(t_room_C)) < 1.0e-9 else "cool_to_room"
        schedule.append((label, float(value)))
    return schedule


def compute_d2d_roomtemp_bond_cu_ep_history(
    layer_df: pd.DataFrame,
    *,
    L_m: float,
    t_room_C: float,
    t_anneal_C: float,
    heat_steps: int = 30,
    cool_steps: int = 30,
    include_anneal_hold: bool = True,
    cu_params: CuEpParams | None = None,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """
    Compute exp05 pre/post RT bow while carrying Cu plastic strain.

    The reduced model treats each Cu-bearing interface layer as an elastic
    SiO2/Cu homogenized layer for stiffness, but carries a separate scalar Cu
    plastic strain state.  The plastic strain enters the interface layer as a
    residual eigenstrain weighted by the Cu stiffness share.
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
    arrays = _required_arrays(df)
    h = arrays["h"]
    E = arrays["E"]
    nu = arrays["nu"]
    alpha = arrays["alpha"]
    E_biaxial = E / (1.0 - nu)
    cu_vf = _cu_volume_fraction(df)
    cu_biaxial_MPa = CU_E_GPA * 1000.0 / (1.0 - CU_NU)
    layer_biaxial_MPa = E_biaxial / 1.0e6
    cu_plastic_weight = np.where(
        cu_vf > 0.0,
        np.clip(cu_vf * cu_biaxial_MPa / np.maximum(layer_biaxial_MPa, 1.0e-30), 0.0, 1.0),
        0.0,
    )
    is_cu_layer = cu_vf > 1.0e-12
    eps_p_cu = np.zeros(len(df), dtype=float)
    eq_p_cu = np.zeros(len(df), dtype=float)

    schedule = _temperature_schedule(
        t_room_C=t_room_C,
        t_anneal_C=t_anneal_C,
        heat_steps=heat_steps,
        cool_steps=cool_steps,
        include_anneal_hold=include_anneal_hold,
    )
    history_rows: list[dict[str, float | int | str]] = []
    layer_rows: list[pd.DataFrame] = []
    selected: dict[str, dict[str, float]] = {}

    def eigenstrain_for_temperature(T_C: float) -> np.ndarray:
        delta_T = float(T_C) - float(t_room_C)
        thermal = alpha * delta_T
        return thermal - cu_plastic_weight * eps_p_cu

    for step_index, (stage, T_C) in enumerate(schedule):
        for _ in range(max(1, int(params.local_iterations))):
            eigenstrain = eigenstrain_for_temperature(T_C)
            summary, out = compute_stack_warpage_with_eigenstrain(df, L_m=L_m, eigenstrain=eigenstrain)
            if not np.any(is_cu_layer):
                break
            s_free = float(summary["s_free_strain"])
            common = out["common_mechanical_strain"].to_numpy(dtype=float)
            reference_free = eigenstrain[0]
            yielded = False
            for idx in np.where(is_cu_layer)[0]:
                cu_thermal = CU_ALPHA_PPM_K * 1.0e-6 * (float(T_C) - float(t_room_C))
                cu_free_delta = (cu_thermal - eps_p_cu[idx]) - reference_free
                cu_strain = cu_free_delta - s_free + common[idx]
                trial_MPa = cu_biaxial_MPa * cu_strain
                flow_MPa = _yield_at_temperature(T_C, params) + float(params.tangent_hardening_MPa) * eq_p_cu[idx]
                overstress = abs(trial_MPa) - flow_MPa
                if overstress <= float(params.plastic_tolerance_MPa):
                    continue
                delta_gamma = overstress / (cu_biaxial_MPa + float(params.tangent_hardening_MPa))
                eps_p_cu[idx] += np.sign(trial_MPa) * delta_gamma
                eq_p_cu[idx] += delta_gamma
                yielded = True
            if not yielded:
                break

        eigenstrain = eigenstrain_for_temperature(T_C)
        summary, out = compute_stack_warpage_with_eigenstrain(df, L_m=L_m, eigenstrain=eigenstrain)
        out = out.copy()
        out["history_step_index"] = step_index
        out["history_stage"] = stage
        out["temperature_C"] = float(T_C)
        out["cu_volume_fraction"] = cu_vf
        out["cu_plastic_weight"] = cu_plastic_weight
        out["cu_plastic_strain"] = eps_p_cu
        out["cu_equivalent_plastic_strain"] = eq_p_cu
        layer_rows.append(out)

        row = {
            "model_step_index": int(step_index),
            "model_stage": stage,
            "temperature_C": float(T_C),
            "model_stack_bow_um": float(summary["signed_W_um"]),
            "model_kappa_1_per_m": float(summary["kappa_1_per_m"]),
            "max_abs_cu_plastic_strain": float(np.max(np.abs(eps_p_cu))) if len(eps_p_cu) else 0.0,
            "max_cu_equivalent_plastic_strain": float(np.max(eq_p_cu)) if len(eq_p_cu) else 0.0,
            "active_cu_layer_count": int(np.count_nonzero(is_cu_layer)),
        }
        history_rows.append(row)
        if stage == "pre_anneal_rt":
            selected["pre_anneal_rt"] = row
        if stage in ("heat_to_anneal", "anneal_hold") and abs(float(T_C) - float(t_anneal_C)) < 1.0e-9:
            selected["anneal"] = row
        if stage == "post_anneal_rt":
            selected["post_anneal_rt"] = row

    pre = selected.get("pre_anneal_rt", history_rows[0])
    post = selected.get("post_anneal_rt", history_rows[-1])
    anneal = selected.get("anneal", history_rows[-1])
    summary = {
        "pre_anneal_rt": pre,
        "anneal": anneal,
        "post_anneal_rt": post,
        "pre_anneal_rt_bow_um": float(pre["model_stack_bow_um"]),
        "anneal_bow_um": float(anneal["model_stack_bow_um"]),
        "post_anneal_rt_bow_um": float(post["model_stack_bow_um"]),
        "post_minus_pre_bow_um": float(post["model_stack_bow_um"]) - float(pre["model_stack_bow_um"]),
        "max_abs_cu_plastic_strain": float(max(row["max_abs_cu_plastic_strain"] for row in history_rows)),
    }
    history_df = pd.DataFrame(history_rows)
    layer_history_df = pd.concat(layer_rows, ignore_index=True) if layer_rows else pd.DataFrame()
    return summary, history_df, layer_history_df


def compute_d2d_roomtemp_bond_cu_ep_pre_post(
    layer_df: pd.DataFrame,
    *,
    L_m: float,
    t_room_C: float,
    t_anneal_C: float,
    heat_steps: int = 30,
    cool_steps: int = 30,
    cu_params: CuEpParams | None = None,
) -> dict[str, float]:
    """Convenience wrapper returning only the official exp05 comparison values."""

    summary, _, _ = compute_d2d_roomtemp_bond_cu_ep_history(
        layer_df,
        L_m=L_m,
        t_room_C=t_room_C,
        t_anneal_C=t_anneal_C,
        heat_steps=heat_steps,
        cool_steps=cool_steps,
        cu_params=cu_params,
    )
    return {
        "pre_anneal_rt_bow_um": float(summary["pre_anneal_rt_bow_um"]),
        "post_anneal_rt_bow_um": float(summary["post_anneal_rt_bow_um"]),
        "post_minus_pre_bow_um": float(summary["post_minus_pre_bow_um"]),
        "max_abs_cu_plastic_strain": float(summary["max_abs_cu_plastic_strain"]),
    }
