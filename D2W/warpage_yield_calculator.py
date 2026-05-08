#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Substack-level total warpage estimates for D2W stacks.

The main entry point, ``get_interface_stack_warpage_map``, returns the
pre-bond bottom-stack warpage distribution for each interface. The warpage
span is referenced to the current top die size, so lateral 2.5D substacks and
small-on-large die cases use the bonding die footprint directly.
"""

import math

import numpy as np
import pandas as pd

try:
    from utils.util import stack_graph_from_3dbx
except ModuleNotFoundError:
    from D2W.utils.util import stack_graph_from_3dbx


def bow_um_to_curvature(W0_um, L_m):
    """Convert signed bow in um over half-length L_m to curvature in 1/m."""
    return 2.0 * (np.asarray(W0_um, dtype=float) * 1e-6) / (L_m**2)


def curvature_to_bow_um(kappa, L_m):
    """Convert signed curvature in 1/m to signed bow in um."""
    return 0.5 * kappa * L_m**2 * 1e6


def compute_total_stack_warpage(layer_df, DeltaT_K, L_m):
    """
    Compute multilayer total stack bow using the model from
    total_stack_warpage_16die.ipynb.

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

    mean = _first_existing_float(
        cfg,
        [
            f"{side}_INI_BOW_MEAN_um",
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


def _top_die_half_length_m(cfg):
    die_w_um = _cfg_float(cfg, "DIE_W_um")
    die_l_um = _cfg_float(cfg, "DIE_L_um")
    return 0.5 * max(die_w_um, die_l_um) * 1e-6


def _layer_cfg_and_side(cfg_dict, interfaces_bottom_to_top, layer_index):
    if layer_index == 0:
        if not interfaces_bottom_to_top:
            raise ValueError("A root-only substack does not have an interface cfg.")
        return cfg_dict[interfaces_bottom_to_top[0]], "BOT"
    return cfg_dict[interfaces_bottom_to_top[layer_index - 1]], "TOP"


def _effective_material_properties_from_volume_fraction(cfg, mix_prefix):
    """
    Return effective E/nu/alpha for a three-material Cu/SiO2/Si mixture.

    Missing volume fractions fall back to a pure-Si placeholder. Present
    fractions are normalized before applying a linear rule of mixtures, matching
    the local effective-layer convention used by debond.py.
    """
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


def _build_layer_df(cfg_dict, chiplets_bottom_to_top, interfaces_bottom_to_top):
    rows = []
    sigmas = []
    for layer_index, chiplet in enumerate(chiplets_bottom_to_top):
        cfg, side = _layer_cfg_and_side(cfg_dict, interfaces_bottom_to_top, layer_index)
        h_key = "ITF_BOT_THICK_um" if side == "BOT" else "ITF_TOP_THICK_um"
        mix_prefix = "B_Sub" if side == "BOT" else "T_Sub"
        bow_mean_um, bow_std_um = _initial_bow_stats_um(cfg, side)
        effective_material = _effective_material_properties_from_volume_fraction(cfg, mix_prefix)

        rows.append({
            "chiplet": chiplet,
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


def _stack_warpage_gaussian(layer_df, layer_bow_sigma_um, DeltaT_K, L_m):
    base_summary, _ = compute_total_stack_warpage(
        layer_df,
        DeltaT_K=DeltaT_K,
        L_m=L_m,
    )
    base_mu_um = float(base_summary["signed_W_um"])

    sensitivities = []
    base_w0 = layer_df["W0_um"].to_numpy(dtype=float)
    for layer_index in range(len(layer_df)):
        perturbed = layer_df.copy()
        perturbed_w0 = base_w0.copy()
        perturbed_w0[layer_index] += 1.0
        perturbed["W0_um"] = perturbed_w0
        perturbed_summary, _ = compute_total_stack_warpage(
            perturbed,
            DeltaT_K=DeltaT_K,
            L_m=L_m,
        )
        sensitivities.append(float(perturbed_summary["signed_W_um"]) - base_mu_um)

    sensitivities = np.asarray(sensitivities, dtype=float)
    sigma_um = float(np.sqrt(np.sum((sensitivities * layer_bow_sigma_um) ** 2)))
    return base_mu_um, sigma_um, sensitivities


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


def get_interface_stack_warpage_map(cfg_dict, _3dbx_path):
    """
    Return pre-bond stack warpage distribution for every D2W interface.

    For interface j in a substack, the returned ``mu_um`` and ``sigma_um`` are
    the Gaussian parameters of the already-bonded bottom stack before bonding
    interface j. The length scale is the current top die half-length from that
    interface cfg.
    """
    substacks = stack_graph_from_3dbx(_3dbx_path)
    interface_stack_warpage = {}

    for substack in substacks:
        chiplets = substack["chiplets_bottom_to_top"]
        interfaces = substack["interfaces_bottom_to_top"]

        for interface_index, interface in enumerate(interfaces):
            if interface not in cfg_dict:
                raise KeyError(f"Interface '{interface}' from {_3dbx_path} is missing in cfg_dict.")

            current_cfg = cfg_dict[interface]
            top_die_mu_um, top_die_sigma_um = _initial_bow_stats_um(current_cfg, "TOP")
            L_m = _top_die_half_length_m(current_cfg)

            prebond_chiplets = chiplets[:interface_index + 1]
            prebond_interfaces = interfaces[:interface_index]
            layer_df, layer_bow_sigma_um = _build_layer_df(
                cfg_dict,
                prebond_chiplets,
                interfaces,
            )

            if interface_index == 0:
                DeltaT_K = 0.0
                last_completed_interface = None
            else:
                last_completed_interface = interfaces[interface_index - 1]
                DeltaT_K = _delta_t_from_cfg(cfg_dict[last_completed_interface])

            stack_mu_um, stack_sigma_um, sensitivities = _stack_warpage_gaussian(
                layer_df,
                layer_bow_sigma_um,
                DeltaT_K=DeltaT_K,
                L_m=L_m,
            )

            interface_stack_warpage[interface] = {
                "substack_id": int(substack["substack_id"]),
                "chiplets_before_bond": list(prebond_chiplets),
                "interfaces_before_bond": list(prebond_interfaces),
                "last_completed_interface": last_completed_interface,
                "incoming_top_chiplet": chiplets[interface_index + 1],
                "mu_um": stack_mu_um,
                "sigma_um": stack_sigma_um,
                "top_die_mu_um": float(top_die_mu_um),
                "top_die_sigma_um": float(top_die_sigma_um),
                "L_m": float(L_m),                  # Die half-length
                "DeltaT_K": float(DeltaT_K),
                "layers": layer_df.to_dict(orient="records"),
                "initial_bow_sensitivity": sensitivities.tolist(),
            }

    return interface_stack_warpage

def compute_warpage_yield(cfg_dict, _3dbx_path):
    """
    Compute final total-warpage yield for every substack in a design.

    For each substack, all chiplets in the root-to-leaf path are treated as the
    final bonded stack. The pass condition is:

        abs(W_total) <= STACK_WARPAGE_TH

    where W_total is modeled as a Gaussian distribution in micrometers.
    """
    substacks = stack_graph_from_3dbx(_3dbx_path)
    substack_results = {}
    overall_warpage_yield = 1.0

    for substack in substacks:
        substack_id = int(substack["substack_id"])
        chiplets = substack["chiplets_bottom_to_top"]
        interfaces = substack["interfaces_bottom_to_top"]
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
        L_m = _top_die_half_length_m(final_cfg)
        DeltaT_K = _delta_t_from_cfg(final_cfg)
        layer_df, layer_bow_sigma_um = _build_layer_df(
            cfg_dict,
            chiplets,
            interfaces,
        )

        mu_um, sigma_um, sensitivities = _stack_warpage_gaussian(
            layer_df,
            layer_bow_sigma_um,
            DeltaT_K=DeltaT_K,
            L_m=L_m,
        )
        yield_value = _gaussian_abs_below_probability(
            mu=mu_um,
            sigma=sigma_um,
            threshold=threshold_um,
        )
        overall_warpage_yield *= yield_value

        substack_results[substack_id] = {
            "substack_id": substack_id,
            "chiplets_bottom_to_top": list(chiplets),
            "interfaces_bottom_to_top": list(interfaces),
            "final_interface": final_interface,
            "mu_um": float(mu_um),
            "sigma_um": float(sigma_um),
            "threshold_um": float(threshold_um),
            "warpage_yield": float(yield_value),
            "pass_condition": "abs(W_total_um) <= threshold_um",
            "L_m": float(L_m),
            "DeltaT_K": float(DeltaT_K),
            "layers": layer_df.to_dict(orient="records"),
            "initial_bow_sensitivity": sensitivities.tolist(),
        }

    return {
        "overall_warpage_yield": float(overall_warpage_yield),
        "substack_results": substack_results,
    }
