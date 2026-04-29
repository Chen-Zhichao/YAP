#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debond.py  (produce dishing intervals from manual coord list)

- Remove ALL external wafer_info reading.
- User defines pad global coordinates (in µm) at the caller.
- Output: numpy.ndarray (N,2), each row = sorted (D_Cu_nm, D_SiO2_nm).

UPDATED (paper Eq.(27)–(36) version):
1) PAD-SCALE CORE replaced by paper Eq.(27)–(36).
2) All paper coefficients are loaded from cfg in __init_params(cfg).
3) Inversion (FIXED WINDOW):
   - SiO2: search D in [-10, 10] nm only; if no root -> return -10
           More negative than -10 nm means the oxide protrusion is treated
           as particle-like and can interfere with bond formation.
   - Cu  : search D in [0, D_contact_max] only, D_contact_max = delta_heat/2
           if no root -> return D_contact_max
   Notes:
     - To avoid phi->0 divergence at the boundary D = D_contact_max, the LUT
       evaluation upper bound uses hi_eval = nextafter(D_contact_max, 0).

FAST VERSION (LUT + VECTORIZED forward model):
- Build sigma->D LUTs by vectorized evaluation of Eq.(32)/(36).
- Invert sigma->D by interpolation on monotone-enforced LUT curves.
- NEW: Build radial dishing LUT r->[D_cu, D_sio2] for each wafer/die state
       (fixed R_m, p_max, beta), so per-pad path becomes:
       coords -> r -> interp D
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple
import math
import matplotlib.pyplot as plt
import numpy as np
from roughness_coefficients import get_eff_contact_area_ratio
from matplotlib.ticker import MultipleLocator

# =============================================================================
# =============================== PARAMETERS ==================================
# =============================================================================

# ---------- (F) Wafer-layer materials ----------
@dataclass(frozen=True)
class Material:
    name: str
    E_Pa: float
    alpha_perC: float
    nu: float

# ---------- (G) Wafer configs ----------
@dataclass
class LayerMix3:
    mat1: Material; V1: float
    mat2: Material; V2: float
    mat3: Material; V3: float
    t_m: float

@dataclass(frozen=True)
class EqLayer:
    E_Pa: float
    alpha_perC: float
    nu: float
    t_m: float

@dataclass
class WaferConfig:
    top: LayerMix3
    bottom: LayerMix3
    L_m: float
    T_C: float
    T0_C: float


@dataclass(frozen=True)
class BondedLayerInput:
    name: str
    E_GPa: float
    alpha_ppm: float
    nu: float
    t_um: float


@dataclass(frozen=True)
class BondedStackResult:
    bond_temperature_C: float
    target_temperature_C: float
    delta_T_C: float
    wafer_diameter_mm: float
    layers_bottom_to_top: Tuple[BondedLayerInput, ...]
    uniform_strain: float
    bending_axis_um: float
    bending_axis_from_bottom_um: float
    curvature_1_per_m: float
    radius_m: float
    warpage_um: float
    warpage_small_deflection_um: float


# ---------------- LUT cache (global) ----------------
_LUT_READY: bool = False
_LUT_SIO2: dict | None = None
_LUT_CU: dict | None = None

# ---------------- Radial dishing LUT cache (global) ----------------
_RDISH_LUT: dict | None = None


def __init_params(cfg):
    global PITCH_UM, DIAM_UM, T_ANNEAL_C, T_REF_C, \
           CU_E_GPA, CU_NU, CU_ALPHA_PPM, OX_E_GPA, OX_NU, OX_ALPHA_PPM, \
           SIGMA_Y_MPA, \
           CRIT_aY2_UM, GC_SIO2_JPM2, GC_CU_JPM2, Effective_Contact_Area, \
           MAT_CU, MAT_SiO2, MAT_Si, \
           WAFER_A, WAFER_B, \
           S_INIT_A_M, S_INIT_B_M, \
           USE_PLOT, \
           C_HEAT_E, C_HEAT_P, C_COOL_E, C_COOL_P, \
           EXP_PHI, BAUSCHINGER, EXP_INVPHI, EXP_AREA, \
           KN_DEN_M, \
           _LUT_READY, _LUT_SIO2, _LUT_CU, _RDISH_LUT

    # reset caches on every init
    _LUT_READY = False
    _LUT_SIO2 = None
    _LUT_CU = None
    _RDISH_LUT = None

    # ---------- (A) Pad-scale: Geometry & Temps ----------
    if cfg.PAD_ARRANGE_PATTERN in ('checkerboard', 'rectangular'):
        PITCH_UM = min(np.sqrt(cfg.PITCH_r_um ** 2 + cfg.PITCH_c_um ** 2), 2 * cfg.PITCH_r_um, 2 * cfg.PITCH_c_um)
    else:
        PITCH_UM = cfg.PITCH_r_um

    DIAM_UM       = cfg.PAD_TOP_R_um * 2      # pad diameter d [µm]
    T_ANNEAL_C    = cfg.T_anl                 # anneal temperature [°C]
    T_REF_C       = cfg.T_R                   # reference temperature [°C]

    # ---------- (B) Pad-scale: Material constants ----------
    CU_E_GPA      = cfg.CU_E_GPA
    CU_NU         = cfg.CU_NU
    CU_ALPHA_PPM  = cfg.CU_ALPHA_PPM
    OX_E_GPA      = cfg.OX_E_GPA
    OX_NU         = cfg.OX_NU
    OX_ALPHA_PPM  = cfg.OX_ALPHA_PPM

    # ---------- (C) Pad-scale: Yield stress ----------
    SIGMA_Y_MPA   = cfg.SIGMA_Y_MPA

    # ---------- (D) Paper Eq.(27)–(36) coefficients (from cfg) ----------
    C_HEAT_E     = float(cfg.C_HEAT_E)        # Eq.(29) coefficient for sigma_e
    C_HEAT_P     = float(cfg.C_HEAT_P)        # Eq.(29) coefficient for sigma_p
    C_COOL_E     = float(cfg.C_COOL_E)        # Eq.(35) coefficient for sigma_e,cool
    C_COOL_P     = float(cfg.C_COOL_P)        # Eq.(35) coefficient for sigma_p,cool
    EXP_PHI      = float(cfg.EXP_PHI)         # Eq.(30)
    BAUSCHINGER  = float(cfg.BAUSCHINGER)     # Eq.(33)
    EXP_INVPHI   = float(cfg.EXP_INVPHI)      # Eq.(36)
    EXP_AREA     = float(cfg.EXP_AREA)        # Eq.(36)
    KN_DEN_M     = float(cfg.KN_DEN_M)        # Eq.(31) denominator length in meters

    # ---------- (E) Critical peeling stress ----------
    CRIT_aY2_UM = cfg.CRIT_aY2_UM
    GC_SIO2_JPM2 = cfg.GC_SIO2_JPM2
    GC_CU_JPM2 = cfg.GC_CU_JPM2
    Effective_Contact_Area = get_eff_contact_area_ratio(
        Asperity_R_m = cfg.Asperity_R_m,
        Roughness_sigma_m = cfg.Roughness_sigma_m,
        eta_s = cfg.eta_s,
        Roughness_constant = cfg.Roughness_constant,
        Adhesion_energy = cfg.Adhesion_energy,
        Dielectric_Young_modulus_Pa = cfg.Dielectric_Young_modulus_Pa,
    )
    assert 0.0 < Effective_Contact_Area <= 1.0, f"Effective_Contact_Area must be in (0,1], got {Effective_Contact_Area}"

    # ---------- (F) Wafer-layer materials ----------
    MAT_CU   = Material("Cu",   E_Pa=cfg.CU_E_GPA*1e9,  alpha_perC=cfg.CU_ALPHA_PPM*1e-6, nu=cfg.CU_NU)
    MAT_SiO2 = Material("SiO2", E_Pa=cfg.OX_E_GPA*1e9,  alpha_perC=cfg.OX_ALPHA_PPM*1e-6, nu=cfg.OX_NU)
    MAT_Si   = Material("Si",   E_Pa=cfg.SI_E_GPA*1e9,  alpha_perC=cfg.SI_ALPHA_PPM*1e-6,  nu=cfg.SI_NU)

    # ---------- (G) Wafer configs ----------
    WAFER_A = WaferConfig(
        top=LayerMix3(MAT_CU,cfg.B_Chip_Cu_V,MAT_SiO2,cfg.B_Chip_Sio2_V,MAT_Si,cfg.B_Chip_Si_V,cfg.B_Chip_T),
        bottom=LayerMix3(MAT_Si,cfg.B_Sub_Si_V,MAT_SiO2,cfg.B_Sub_Sio2_V,MAT_CU,cfg.B_Sub_Cu_V,cfg.B_Sub_T),
        L_m= cfg.eff_DIE_R*1e-6, T_C= cfg.T_anl, T0_C= cfg.T_R
    )
    WAFER_B = WaferConfig(
        top=LayerMix3(MAT_CU,cfg.T_Chip_Cu_V,MAT_SiO2,cfg.T_Chip_Sio2_V,MAT_Si,cfg.T_Chip_Si_V,cfg.T_Chip_T),
        bottom=LayerMix3(MAT_Si,cfg.T_Sub_Si_V,MAT_SiO2,cfg.T_Sub_Sio2_V,MAT_CU,cfg.T_Sub_Cu_V,cfg.T_Sub_T),
        L_m= cfg.eff_DIE_R*1e-6, T_C= cfg.T_anl, T0_C= cfg.T_R
    )

    # ---------- (H) Pre-anneal warpages ----------
    S_INIT_A_M = cfg.S_INIT_A_M
    S_INIT_B_M = cfg.S_INIT_B_M

    # ---------- (J) Optional plotting ----------
    USE_PLOT = False


# =============================================================================
# ============================== PAD-SCALE CORE ===============================
# =============================================================================

def _units():
    return dict(um=1e-6, nm=1e-9, GPa=1e9, MPa=1e6)

def _geom_areas(p_um, d_um):
    U=_units(); p=p_um*U['um']; d=d_um*U['um']
    A_cell=p**2
    A_cu=math.pi*(d**2)/4.0
    A_ox=A_cell-A_cu
    if A_ox<=0:
        raise ValueError("A_ox<=0, check PITCH_UM and DIAM_UM values.")
    return A_cell, A_cu, A_ox

# -------- Paper Eq.(27)–(36) primitives --------

def _sigma_t_thermal_Pa() -> float:
    """Eq.(27) in Pa"""
    dT = (T_ANNEAL_C - T_REF_C)
    E = CU_E_GPA * 1e9
    nu = CU_NU
    dalpha = (CU_ALPHA_PPM - OX_ALPHA_PPM) * 1e-6
    return (E / (1.0 - nu)) * dalpha * dT

def _split_sigma_ep_heat_paper(sigma_t_Pa: float) -> Tuple[float, float]:
    """Eq.(28): sigma_e=min(sigma_t,sigma_y), sigma_p=max(sigma_t-sigma_y,0)"""
    sigma_y_Pa = SIGMA_Y_MPA * 1e6
    sigma_e = min(sigma_t_Pa, sigma_y_Pa)
    sigma_p = max(sigma_t_Pa - sigma_y_Pa, 0.0)
    return sigma_e, sigma_p

def _delta_heat_m(sigma_e_Pa: float, sigma_p_Pa: float) -> float:
    """Eq.(29) -> meters"""
    E = CU_E_GPA * 1e9
    nu = CU_NU
    return (4.0 * nu / E) * (C_HEAT_E * sigma_e_Pa + C_HEAT_P * sigma_p_Pa)

def _k_n_Pa_per_m() -> float:
    """Eq.(31) -> Pa/m"""
    E = CU_E_GPA * 1e9
    nu = CU_NU
    return (2.0 * E) / (KN_DEN_M * (1.0 - nu))

def _sigma_y_cool_Pa() -> float:
    """Eq.(33): sigma_y,cool = (1-BAUSCHINGER)*sigma_y"""
    sigma_y_Pa = SIGMA_Y_MPA * 1e6
    return (1.0 - BAUSCHINGER) * sigma_y_Pa

def _split_sigma_ep_cool_paper(sigma_t_Pa: float) -> Tuple[float, float]:
    """Eq.(34)"""
    syc = _sigma_y_cool_Pa()
    sigma_e = min(sigma_t_Pa, syc)
    sigma_p = max(sigma_t_Pa - syc, 0.0)
    return sigma_e, sigma_p

def _delta_cool_m(sigma_e_Pa: float, sigma_p_Pa: float) -> float:
    """Eq.(35) -> meters"""
    E = CU_E_GPA * 1e9
    nu = CU_NU
    return (4.0 * nu / E) * (C_COOL_E * sigma_e_Pa + C_COOL_P * sigma_p_Pa)

# ===== Vectorized forward core for LUT building =====

def _padscale_precompute_constants() -> dict:
    """
    Precompute D-independent constants used in Eq.(32) and Eq.(36),
    so LUT building becomes vectorized and fast.
    """
    A_cell, A_cu, A_ox = _geom_areas(PITCH_UM, DIAM_UM)

    sigma_t = _sigma_t_thermal_Pa()
    sigma_e_h, sigma_p_h = _split_sigma_ep_heat_paper(sigma_t)
    d_heat = _delta_heat_m(sigma_e_h, sigma_p_h)

    sigma_e_c, sigma_p_c = _split_sigma_ep_cool_paper(sigma_t)
    d_cool = _delta_cool_m(sigma_e_c, sigma_p_c)

    kn = _k_n_Pa_per_m()
    area_factor = (A_cell / A_cu) ** EXP_AREA

    return dict(
        A_cell=float(A_cell),
        A_cu=float(A_cu),
        A_ox=float(A_ox),
        sigma_t=float(sigma_t),
        d_heat=float(d_heat),
        d_cool=float(d_cool),
        kn=float(kn),
        area_factor=float(area_factor),
    )

def _phi_contact_vec(delta_heat_m_val: float, D_nm_vec: np.ndarray) -> np.ndarray:
    """
    Vectorized Eq.(30):
      phi = clip(((delta_heat-2D)/(2D))^EXP_PHI, 0, 1)
    with special-case D=0 -> 1.
    """
    D_nm_vec = np.asarray(D_nm_vec, dtype=np.float64)
    D_m = D_nm_vec * 1e-9

    phi = np.ones_like(D_m)  # D=0 -> 1
    mask = D_m > 0.0
    if not np.any(mask):
        return phi

    numer = delta_heat_m_val - 2.0 * D_m[mask]
    pos = numer > 0.0

    phi_masked = np.zeros_like(D_m[mask])
    if np.any(pos):
        x = numer[pos] / (2.0 * D_m[mask][pos])
        val = x ** EXP_PHI
        phi_masked[pos] = np.clip(val, 0.0, 1.0)

    phi[mask] = phi_masked
    return phi

def _sigma_sio2_vec_MPa(D_nm_vec: np.ndarray, const: dict) -> np.ndarray:
    """
    Vectorized Eq.(32) -> MPa
    sigma = kn*(d_heat-2D)*(phi*A_cu)/A_ox, with opening<=0 or phi<=0 -> 0
    """
    D_nm_vec = np.asarray(D_nm_vec, dtype=np.float64)
    D_m = D_nm_vec * 1e-9

    d_heat = float(const["d_heat"])
    kn = float(const["kn"])
    A_cu = float(const["A_cu"])
    A_ox = float(const["A_ox"])

    opening = d_heat - 2.0 * D_m
    phi = _phi_contact_vec(d_heat, D_nm_vec)

    sigma_Pa = kn * opening * (phi * A_cu) / A_ox
    sigma_Pa = np.where((opening > 0.0) & (phi > 0.0), sigma_Pa, 0.0)
    return sigma_Pa / 1e6

def _sigma_cu_vec_MPa(D_nm_vec: np.ndarray, const: dict) -> np.ndarray:
    """
    Vectorized Eq.(36) -> MPa
    sigma = kn*(d_cool - d_heat + 2D) * (1/phi)^EXP_INVPHI * (A_cell/A_cu)^EXP_AREA
    with phi<=0 or opening<=0 -> 0
    """
    D_nm_vec = np.asarray(D_nm_vec, dtype=np.float64)
    D_m = D_nm_vec * 1e-9

    d_heat = float(const["d_heat"])
    d_cool = float(const["d_cool"])
    kn = float(const["kn"])
    area_factor = float(const["area_factor"])

    phi = _phi_contact_vec(d_heat, D_nm_vec)
    opening = d_cool - d_heat + 2.0 * D_m

    phi_safe = np.maximum(phi, 1e-12)
    factor_phi = (1.0 / phi_safe) ** EXP_INVPHI

    sigma_Pa = kn * opening * factor_phi * area_factor
    sigma_Pa = np.where((opening > 0.0) & (phi > 0.0), sigma_Pa, 0.0)
    return sigma_Pa / 1e6


# =============================================================================
# ============================ Public forward API =============================
# =============================================================================
# (kept for debugging / compatibility; not used in LUT building)

def _phi_contact(delta_heat_m_val: float, D_nm: float) -> float:
    """Scalar Eq.(30)"""
    D_m = float(D_nm) * 1e-9
    if D_m <= 0.0:
        return 1.0
    numer = delta_heat_m_val - 2.0 * D_m
    if numer <= 0.0:
        return 0.0
    x = numer / (2.0 * D_m)
    val = x ** EXP_PHI
    return float(max(0.0, min(1.0, val)))

def _sigma_peel_sio2_paper_MPa(D_nm: float) -> dict:
    """Eq.(32) scalar (debug)"""
    _, A_cu, A_ox = _geom_areas(PITCH_UM, DIAM_UM)

    sigma_t = _sigma_t_thermal_Pa()
    sigma_e, sigma_p = _split_sigma_ep_heat_paper(sigma_t)
    d_heat = _delta_heat_m(sigma_e, sigma_p)

    D_m = float(D_nm) * 1e-9
    opening = d_heat - 2.0 * D_m
    if opening <= 0.0:
        return dict(sigma_peel_MPa=0.0, phi=0.0, delta_heat_nm=float(d_heat/1e-9), reason="no_opening_or_contact")

    phi = _phi_contact(d_heat, D_nm)
    if phi <= 0.0:
        return dict(sigma_peel_MPa=0.0, phi=0.0, delta_heat_nm=float(d_heat/1e-9), reason="phi_zero")

    kn = _k_n_Pa_per_m()
    sigma_Pa = kn * opening * (phi * A_cu) / A_ox
    return dict(
        sigma_peel_MPa=float(sigma_Pa/1e6),
        phi=float(phi),
        delta_heat_nm=float(d_heat/1e-9),
        reason="ok"
    )

def _sigma_peel_cu_paper_MPa(D_nm: float) -> dict:
    """Eq.(36) scalar (debug)"""
    A_cell, A_cu, _ = _geom_areas(PITCH_UM, DIAM_UM)

    sigma_t = _sigma_t_thermal_Pa()
    sigma_e_h, sigma_p_h = _split_sigma_ep_heat_paper(sigma_t)
    d_heat = _delta_heat_m(sigma_e_h, sigma_p_h)

    phi = _phi_contact(d_heat, D_nm)
    if phi <= 0.0:
        return dict(sigma_cu_peel_MPa=0.0, phi=0.0, delta_heat_nm=float(d_heat/1e-9), reason="no_contact_in_heat")

    sigma_e_c, sigma_p_c = _split_sigma_ep_cool_paper(sigma_t)
    d_cool = _delta_cool_m(sigma_e_c, sigma_p_c)

    D_m = float(D_nm) * 1e-9
    opening = d_cool - d_heat + 2.0 * D_m
    if opening <= 0.0:
        return dict(sigma_cu_peel_MPa=0.0, phi=float(phi), delta_heat_nm=float(d_heat/1e-9),
                    delta_cool_nm=float(d_cool/1e-9), reason="no_opening_in_cool")

    kn = _k_n_Pa_per_m()
    factor_phi  = (1.0 / max(phi, 1e-12)) ** EXP_INVPHI
    factor_area = (A_cell / A_cu) ** EXP_AREA

    sigma_Pa = kn * opening * factor_phi * factor_area
    return dict(
        sigma_cu_peel_MPa=float(sigma_Pa/1e6),
        phi=float(phi),
        delta_heat_nm=float(d_heat/1e-9),
        delta_cool_nm=float(d_cool/1e-9),
        reason="ok"
    )

def compute_sigma_peel_MPa_at(D_nm: float) -> dict:
    """Heat-dwell SiO2 peeling stress at given D (paper Eq.32)."""
    out = _sigma_peel_sio2_paper_MPa(D_nm)
    return dict(
        sigma_peel_MPa=float(out["sigma_peel_MPa"]),
        phi_cu=float(out["phi"]),
        delta_eq_nm=float(out["delta_heat_nm"]),   # legacy key
        delta_heat_nm=float(out["delta_heat_nm"]),
        reason=str(out.get("reason", "ok")),
    )

def compute_cu_peel_cool_MPa_at(D_nm: float) -> dict:
    """Cool-down Cu peeling stress at given D (paper Eq.36)."""
    out = _sigma_peel_cu_paper_MPa(D_nm)
    return dict(
        sigma_cu_peel_MPa=float(out["sigma_cu_peel_MPa"]),
        phi_cu=float(out.get("phi", 0.0)),
        delta_heat_nm=float(out.get("delta_heat_nm", 0.0)),
        delta_cool_nm=float(out.get("delta_cool_nm", 0.0)),
        sigma_y_cool_MPa=float(_sigma_y_cool_Pa()/1e6),
        reason=str(out.get("reason", "ok")),
    )


# =============================================================================
# ============================ CRITICAL / INVERT ==============================
# =============================================================================

def sigma_critical_MPa(Gc_Jpm2: float, E_GPa: float, nu: float,
                       aY2_um: float,
                       Effective_Contact_Area: float) -> float:
    E_Pa  = E_GPa * 1e9
    aY2_m = aY2_um * 1e-6
    sigma_Pa = Effective_Contact_Area * math.sqrt((Gc_Jpm2 * E_Pa) / (aY2_m * (1.0 - nu**2)))
    return float(sigma_Pa * 1e-6)

def compute_critical_peeling_all():
    return {
        "sigma_crit_MPa": {
            "SiO2": sigma_critical_MPa(GC_SIO2_JPM2, OX_E_GPA, OX_NU, CRIT_aY2_UM, Effective_Contact_Area),
            "Cu":   sigma_critical_MPa(GC_CU_JPM2,   CU_E_GPA, CU_NU, CRIT_aY2_UM, 1.0),
        }
    }


def _critical_peeling_pair_MPa() -> Tuple[float, float]:
    """Return (sigma_crit_SiO2_MPa, sigma_crit_Cu_MPa)."""
    crits = compute_critical_peeling_all()
    return float(crits["sigma_crit_MPa"]["SiO2"]), float(crits["sigma_crit_MPa"]["Cu"])


def _effcrit_and_dishing_from_global_peel_MPa(p_global_MPa: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert global peeling stress to effective thresholds and inverted dishing arrays.

    Returns
    -------
    sigma_eff_Cu_MPa, sigma_eff_SiO2_MPa, D_cu_nm, D_sio2_nm
    """
    p_mpa = np.asarray(p_global_MPa, dtype=np.float64)
    sigma_crit_SiO2, sigma_crit_Cu = _critical_peeling_pair_MPa()
    sigma_eff_SiO2 = sigma_crit_SiO2 - p_mpa
    sigma_eff_Cu   = sigma_crit_Cu   - p_mpa
    D_sio2_nm = _invert_sio2_from_lut(sigma_eff_SiO2)
    D_cu_nm   = _invert_cu_from_lut(sigma_eff_Cu)
    return sigma_eff_Cu, sigma_eff_SiO2, D_cu_nm, D_sio2_nm


def _ensure_luts_ready(sio2_n: int = 2001, cu_n: int = 4001):
    """
    Build sigma->D LUTs once per __init_params(cfg).

    SiO2 window: D in [-10,10] nm (inclusive)
    Cu  window: D in [0, hi_eval] where hi_eval = nextafter(D_contact_max, 0)
    """
    global _LUT_READY, _LUT_SIO2, _LUT_CU
    if _LUT_READY:
        return

    const = _padscale_precompute_constants()

    # ---- SiO2 LUT ----
    # More negative than -10 nm means the SiO2 protrusion is treated as a
    # particle-like defect that can interfere with bond formation.
    D_sio2 = np.linspace(-10.0, 10.0, int(sio2_n), dtype=np.float64)
    sig_sio2 = _sigma_sio2_vec_MPa(D_sio2, const)

    # enforce monotone non-increasing
    sig_sio2_mono = np.maximum.accumulate(sig_sio2[::-1])[::-1]

    _LUT_SIO2 = dict(
        D_nm=D_sio2,
        sigma_MPa=sig_sio2_mono,
        lo=-10.0,
        hi=10.0,
        f_lo=float(sig_sio2_mono[0]),
        f_hi=float(sig_sio2_mono[-1]),
        n=int(sio2_n),
    )

    # ---- Cu LUT window ----
    delta_heat_nm = float(const["d_heat"] / 1e-9)
    D_contact_max = max(0.0, 0.5 * delta_heat_nm)

    if D_contact_max <= 0.0:
        sig0 = float(_sigma_cu_vec_MPa(np.array([0.0], dtype=np.float64), const)[0])
        _LUT_CU = dict(
            D_nm=np.array([0.0], dtype=np.float64),
            sigma_MPa=np.array([sig0], dtype=np.float64),
            D_contact_max=0.0,
            hi_eval=0.0,
            lo=0.0,
            hi=0.0,
            f_lo=sig0,
            f_hi=sig0,
            n=1,
            mode="no_contact_domain",
        )
        _LUT_READY = True
        return

    hi_eval = float(np.nextafter(D_contact_max, 0.0))
    if hi_eval <= 0.0:
        sig0 = float(_sigma_cu_vec_MPa(np.array([0.0], dtype=np.float64), const)[0])
        _LUT_CU = dict(
            D_nm=np.array([0.0], dtype=np.float64),
            sigma_MPa=np.array([sig0], dtype=np.float64),
            D_contact_max=float(D_contact_max),
            hi_eval=float(hi_eval),
            lo=0.0,
            hi=float(D_contact_max),
            f_lo=sig0,
            f_hi=sig0,
            n=1,
            mode="tiny_contact_domain",
        )
        _LUT_READY = True
        return

    D_cu = np.linspace(0.0, hi_eval, int(cu_n), dtype=np.float64)
    sig_cu = _sigma_cu_vec_MPa(D_cu, const)

    # enforce monotone non-decreasing
    sig_cu_mono = np.maximum.accumulate(sig_cu)

    _LUT_CU = dict(
        D_nm=D_cu,
        sigma_MPa=sig_cu_mono,
        D_contact_max=float(D_contact_max),
        hi_eval=float(hi_eval),
        lo=0.0,
        hi=float(D_contact_max),
        f_lo=float(sig_cu_mono[0]),
        f_hi=float(sig_cu_mono[-1]),
        n=int(cu_n),
        mode="ok",
    )

    _LUT_READY = True


def _invert_sio2_from_lut(sigma_eff_MPa: np.ndarray) -> np.ndarray:
    """
    Vectorized inversion for SiO2 using sigma->D LUT.
    Rule: window [-10,10] nm; if no root -> return -10.
    """
    _ensure_luts_ready()
    lut = _LUT_SIO2
    D = lut["D_nm"]
    f = lut["sigma_MPa"]  # decreasing (monotone-enforced)

    lo = float(lut["lo"])
    f_lo = float(lut["f_lo"])
    f_hi = float(lut["f_hi"])

    t = np.asarray(sigma_eff_MPa, dtype=np.float64)
    out = np.full_like(t, fill_value=lo, dtype=np.float64)

    # valid range for decreasing curve: t in [f_hi, f_lo]
    mask = (t >= f_hi) & (t <= f_lo)
    if np.any(mask):
        # np.interp wants increasing x, so reverse f and D
        f_inc = f[::-1]
        D_inc = D[::-1]
        out[mask] = np.interp(t[mask], f_inc, D_inc)
    return out


def _invert_cu_from_lut(sigma_eff_MPa: np.ndarray) -> np.ndarray:
    """
    Vectorized inversion for Cu using sigma->D LUT.
    Rule: window [0, D_contact_max]; if no root -> return D_contact_max.
    """
    _ensure_luts_ready()
    lut = _LUT_CU
    D = lut["D_nm"]
    f = lut["sigma_MPa"]  # increasing (monotone-enforced)

    D_contact_max = float(lut["D_contact_max"])
    t = np.asarray(sigma_eff_MPa, dtype=np.float64)

    out = np.full_like(t, fill_value=D_contact_max, dtype=np.float64)

    f_lo = float(lut["f_lo"])
    f_hi = float(lut["f_hi"])

    mask = (t >= f_lo) & (t <= f_hi) & (D_contact_max > 0.0) & (float(lut.get("hi_eval", 0.0)) > 0.0)
    if np.any(mask):
        out[mask] = np.interp(t[mask], f, D)
    return out


# Single-value wrappers (kept for compatibility/debug)
def invert_dishing_sio2_given_sigma_eff(sigma_eff_MPa: float) -> Tuple[float, dict]:
    """
    Fixed window inversion for SiO2 (sigma->D LUT):
      - search D in [-10, 10] nm only
      - if no root -> return -10
    """
    t = float(sigma_eff_MPa)
    D_val = float(_invert_sio2_from_lut(np.array([t], dtype=np.float64))[0])

    lut = _LUT_SIO2
    if (D_val == float(lut["lo"])) and not (float(lut["f_hi"]) <= t <= float(lut["f_lo"])):
        return float(lut["lo"]), dict(
            mode="no_root_in_window",
            lo=lut["lo"], hi=lut["hi"], target=t,
            f_lo=float(lut["f_lo"]),
            f_hi=float(lut["f_hi"]),
            n=lut["n"],
        )
    return D_val, dict(mode="ok", lo=lut["lo"], hi=lut["hi"], target=t, n=lut["n"])


def invert_dishing_cu_given_sigma_eff(sigma_eff_MPa: float) -> Tuple[float, dict]:
    """
    Fixed window inversion for Cu (sigma->D LUT):
      - search D in [0, D_contact_max] only
      - if no root -> return D_contact_max
    """
    t = float(sigma_eff_MPa)
    D_val = float(_invert_cu_from_lut(np.array([t], dtype=np.float64))[0])

    lut = _LUT_CU
    if lut.get("mode", "ok") != "ok":
        return D_val, dict(mode=lut.get("mode", "unknown"), target=t, D_contact_max=float(lut.get("D_contact_max", 0.0)))

    f_lo = float(lut["f_lo"])
    f_hi = float(lut["f_hi"])
    if not (f_lo <= t <= f_hi):
        return float(lut["D_contact_max"]), dict(
            mode="no_root_in_window",
            lo=lut["lo"], hi=lut["hi"],
            target=t,
            D_contact_max=float(lut["D_contact_max"]),
            hi_eval=float(lut["hi_eval"]),
            f_lo=f_lo, f_hi=f_hi,
            n=lut["n"],
        )

    return D_val, dict(
        mode="ok",
        lo=lut["lo"], hi=lut["hi"],
        target=t,
        D_contact_max=float(lut["D_contact_max"]),
        hi_eval=float(lut["hi_eval"]),
        n=lut["n"],
    )


# =============================================================================
# ============================ WAFER-LEVEL STACK ==============================
# =============================================================================

def equiv_from_three(mix: LayerMix3) -> EqLayer:
    V1,V2,V3 = mix.V1,mix.V2,mix.V3
    totalV = V1+V2+V3
    if totalV==0.0:
        raise ValueError("Sum of volumes must be >0.")
    aeq = (mix.mat1.alpha_perC*V1 + mix.mat2.alpha_perC*V2 + mix.mat3.alpha_perC*V3)/totalV
    Eeq = (mix.mat1.E_Pa      *V1 + mix.mat2.E_Pa      *V2 + mix.mat3.E_Pa      *V3)/totalV
    nueq= (mix.mat1.nu        *V1 + mix.mat2.nu        *V2 + mix.mat3.nu        *V3)/totalV
    return EqLayer(E_Pa=Eeq, alpha_perC=aeq, nu=nueq, t_m=mix.t_m)

def warpage_D_two_layer_exact(L_m,t_c_m,t_s_m,E_c,E_s,alpha_c,alpha_s,T_C,T0_C):
    ratio = t_c_m / t_s_m
    dT = (T_C - T0_C)
    num_pref = (3.0 * (L_m ** 2)) / (4.0 * (t_c_m + t_s_m))
    numerator = num_pref * ((1.0 + ratio) ** 2) * (alpha_s - alpha_c) * dT
    denom_left  = 3.0 * (1.0 + t_c_m / t_s_m) ** 2
    denom_right = (1.0 + (t_c_m * E_c) / (t_s_m * E_s)) * ((t_c_m ** 2) / (t_s_m ** 2) + (t_s_m * E_s) / (t_c_m * E_c))
    denominator = denom_left + denom_right
    if denominator == 0.0:
        raise ZeroDivisionError("Denominator zero.")
    return numerator / denominator

def combine_two_layers_to_one(top_eq: EqLayer, bot_eq: EqLayer) -> EqLayer:
    Vt,Vs = top_eq.t_m, bot_eq.t_m
    total=Vt+Vs
    if total==0.0:
        raise ValueError("Total thickness is zero.")
    aeq = (top_eq.alpha_perC*Vt + bot_eq.alpha_perC*Vs)/total
    Eeq = (top_eq.E_Pa      *Vt + bot_eq.E_Pa      *Vs)/total
    nueq= (top_eq.nu        *Vt + bot_eq.nu        *Vs)/total
    return EqLayer(E_Pa=Eeq, alpha_perC=aeq, nu=nueq, t_m=total)

@dataclass(frozen=True)
class WaferResult:
    D_m: float
    final_eq: EqLayer

def process_wafer(cfg: WaferConfig) -> WaferResult:
    top_eq = equiv_from_three(cfg.top)
    bot_eq = equiv_from_three(cfg.bottom)

    D = warpage_D_two_layer_exact(cfg.L_m, top_eq.t_m, bot_eq.t_m,
                                  top_eq.E_Pa, bot_eq.E_Pa,
                                  top_eq.alpha_perC, bot_eq.alpha_perC,
                                  cfg.T_C, cfg.T0_C)
    final_eq = combine_two_layers_to_one(top_eq, bot_eq)
    return WaferResult(D_m=D, final_eq=final_eq)


def _eq_layer_to_bonded_layer(name: str, layer_eff: EqLayer) -> BondedLayerInput:
    """Convert an EqLayer in SI units to the bonded-stack layer format used by the post-bond model."""
    return BondedLayerInput(
        name=str(name),
        E_GPa=float(layer_eff.E_Pa) / 1e9,
        alpha_ppm=float(layer_eff.alpha_perC) * 1e6,
        nu=float(layer_eff.nu),
        t_um=float(layer_eff.t_m) * 1e6,
    )


def _resolve_post_bond_wafer_diameter_mm(cfg) -> float:
    """
    Resolve the in-plane diameter used by the post-bond warpage model.

    For D2W post-bond warpage, use twice the die-region diagonal:
      diameter_um = 2 * sqrt(DIE_W_um^2 + DIE_L_um^2)
    """
    die_w_um = getattr(cfg, "DIE_W_um", None)
    die_h_um = getattr(cfg, "DIE_L_um", None)
    if die_w_um is not None and die_h_um is not None and float(die_w_um) > 0.0 and float(die_h_um) > 0.0:
        diameter_um = 2.0 * math.sqrt(float(die_w_um) ** 2 + float(die_h_um) ** 2)
        return diameter_um * 1e-3

    raise ValueError(
        "Cannot resolve post-bond die-region diameter from cfg. Provide positive DIE_W_um and DIE_L_um."
    )


def _build_post_bond_stack_from_cfg(cfg) -> Tuple[BondedLayerInput, ...]:
    """
    Build the bonded stack in physical order from bottom free surface to top free surface.

    Order:
      bottom substrate -> bottom chip -> top substrate -> top chip
    """
    mat_cu = Material("Cu", E_Pa=float(cfg.CU_E_GPA) * 1e9, alpha_perC=float(cfg.CU_ALPHA_PPM) * 1e-6, nu=float(cfg.CU_NU))
    mat_sio2 = Material("SiO2", E_Pa=float(cfg.OX_E_GPA) * 1e9, alpha_perC=float(cfg.OX_ALPHA_PPM) * 1e-6, nu=float(cfg.OX_NU))
    mat_si = Material("Si", E_Pa=float(cfg.SI_E_GPA) * 1e9, alpha_perC=float(cfg.SI_ALPHA_PPM) * 1e-6, nu=float(cfg.SI_NU))

    bottom_sub = _eq_layer_to_bonded_layer(
        "BOTTOM_SUBSTRATE",
        equiv_from_three(LayerMix3(mat_si, cfg.B_Sub_Si_V, mat_sio2, cfg.B_Sub_Sio2_V, mat_cu, cfg.B_Sub_Cu_V, cfg.B_Sub_T)),
    )
    bottom_chip = _eq_layer_to_bonded_layer(
        "BOTTOM_CHIP",
        equiv_from_three(LayerMix3(mat_cu, cfg.B_Chip_Cu_V, mat_sio2, cfg.B_Chip_Sio2_V, mat_si, cfg.B_Chip_Si_V, cfg.B_Chip_T)),
    )
    top_chip = _eq_layer_to_bonded_layer(
        "TOP_CHIP",
        equiv_from_three(LayerMix3(mat_cu, cfg.T_Chip_Cu_V, mat_sio2, cfg.T_Chip_Sio2_V, mat_si, cfg.T_Chip_Si_V, cfg.T_Chip_T)),
    )
    top_sub = _eq_layer_to_bonded_layer(
        "TOP_SUBSTRATE",
        equiv_from_three(LayerMix3(mat_si, cfg.T_Sub_Si_V, mat_sio2, cfg.T_Sub_Sio2_V, mat_cu, cfg.T_Sub_Cu_V, cfg.T_Sub_T)),
    )
    return (bottom_sub, bottom_chip, top_sub, top_chip)


def _biaxial_modulus_pa(E_GPa: float, nu: float) -> float:
    """Return the biaxial modulus E / (1 - nu) in Pa."""
    if float(nu) >= 1.0:
        raise ValueError(f"Poisson ratio must be < 1. Got nu={nu}.")
    return float(E_GPa) * 1e9 / (1.0 - float(nu))


def _compute_warpage_from_curvature(
    curvature_1_per_m: float,
    wafer_diameter_mm: float,
) -> Tuple[float, float]:
    """
    Convert curvature to exact circular-arc warpage and small-deflection warpage, both in um.
    """
    if float(wafer_diameter_mm) <= 0.0:
        raise ValueError("wafer_diameter_mm must be > 0.")

    curvature_abs = abs(float(curvature_1_per_m))
    diameter_m = float(wafer_diameter_mm) * 1e-3
    warpage_small_deflection_m = curvature_abs * diameter_m ** 2 / 8.0

    if math.isclose(curvature_abs, 0.0, abs_tol=1e-18):
        return 0.0, 0.0

    radius_abs_m = 1.0 / curvature_abs
    half_diameter_m = 0.5 * diameter_m
    if half_diameter_m > radius_abs_m:
        raise ValueError(
            "wafer_diameter_mm is too large for the current curvature under the circular-arc assumption."
        )

    warpage_exact_m = radius_abs_m - math.sqrt(radius_abs_m ** 2 - half_diameter_m ** 2)
    return float(warpage_exact_m * 1e6), float(warpage_small_deflection_m * 1e6)


def _compute_post_bond_curvature_formula10(
    *,
    layers_bottom_to_top: Sequence[BondedLayerInput],
    bond_temperature_C: float,
    target_temperature_C: float,
    wafer_diameter_mm: float,
) -> BondedStackResult:
    """
    Exact multilayer curvature from Hsueh (Thin Solid Films 418, 2002).

    This mirrors the flow in effective_layer.py:
      Eq. (6): uniform strain
      Eq. (8): bending-axis location
      Eq. (10): curvature
    """
    if len(layers_bottom_to_top) < 2:
        raise ValueError("Need at least 2 bonded layers to compute curvature.")

    delta_T_C = float(target_temperature_C) - float(bond_temperature_C)

    layers = []
    for layer in layers_bottom_to_top:
        if float(layer.t_um) <= 0.0:
            raise ValueError(f"Layer '{layer.name}' thickness must be > 0.")
        layers.append(
            {
                "name": str(layer.name),
                "E_biaxial_Pa": _biaxial_modulus_pa(layer.E_GPa, layer.nu),
                "alpha_perC": float(layer.alpha_ppm) * 1e-6,
                "t_m": float(layer.t_um) * 1e-6,
                "t_um": float(layer.t_um),
            }
        )

    substrate = layers[0]
    films = layers[1:]

    extensional_sum = substrate["E_biaxial_Pa"] * substrate["t_m"]
    weighted_alpha_sum = extensional_sum * substrate["alpha_perC"]
    for film in films:
        extensional_sum += film["E_biaxial_Pa"] * film["t_m"]
        weighted_alpha_sum += film["E_biaxial_Pa"] * film["t_m"] * film["alpha_perC"]

    uniform_strain = delta_T_C * weighted_alpha_sum / extensional_sum

    h_prev = 0.0
    tb_numerator = -substrate["E_biaxial_Pa"] * substrate["t_m"] ** 2
    for film in films:
        tb_numerator += film["E_biaxial_Pa"] * film["t_m"] * (2.0 * h_prev + film["t_m"])
        h_prev += film["t_m"]
    t_b_m = tb_numerator / (2.0 * extensional_sum)

    h_prev = 0.0
    curvature_numerator = (
        3.0
        * substrate["E_biaxial_Pa"]
        * (uniform_strain - substrate["alpha_perC"] * delta_T_C)
        * substrate["t_m"] ** 2
    )
    for film in films:
        curvature_numerator -= (
            3.0
            * film["E_biaxial_Pa"]
            * film["t_m"]
            * (uniform_strain - film["alpha_perC"] * delta_T_C)
            * (2.0 * h_prev + film["t_m"])
        )
        h_prev += film["t_m"]

    h_prev = 0.0
    curvature_denominator = (
        substrate["E_biaxial_Pa"]
        * substrate["t_m"]
        * (2.0 * substrate["t_m"] ** 2 + 3.0 * substrate["t_m"] * t_b_m)
    )
    for film in films:
        curvature_denominator += (
            film["E_biaxial_Pa"]
            * film["t_m"]
            * (
                6.0 * h_prev ** 2
                + 6.0 * h_prev * film["t_m"]
                + 2.0 * film["t_m"] ** 2
                - 3.0 * t_b_m * (2.0 * h_prev + film["t_m"])
            )
        )
        h_prev += film["t_m"]

    curvature_1_per_m = curvature_numerator / curvature_denominator
    radius_m = math.inf if math.isclose(curvature_1_per_m, 0.0, abs_tol=1e-18) else 1.0 / curvature_1_per_m
    bending_axis_um = t_b_m * 1e6
    bending_axis_from_bottom_um = substrate["t_um"] + bending_axis_um
    warpage_um, warpage_small_deflection_um = _compute_warpage_from_curvature(
        curvature_1_per_m=curvature_1_per_m,
        wafer_diameter_mm=wafer_diameter_mm,
    )

    return BondedStackResult(
        bond_temperature_C=float(bond_temperature_C),
        target_temperature_C=float(target_temperature_C),
        delta_T_C=delta_T_C,
        wafer_diameter_mm=float(wafer_diameter_mm),
        layers_bottom_to_top=tuple(layers_bottom_to_top),
        uniform_strain=float(uniform_strain),
        bending_axis_um=float(bending_axis_um),
        bending_axis_from_bottom_um=float(bending_axis_from_bottom_um),
        curvature_1_per_m=float(curvature_1_per_m),
        radius_m=float(radius_m) if not math.isinf(radius_m) else math.inf,
        warpage_um=float(warpage_um),
        warpage_small_deflection_um=float(warpage_small_deflection_um),
    )


def post_bond_warpage_calculator(cfg) -> float:
    """
    Return post-bond warpage [um] using the effective-layer multilayer curvature flow.

    The current mapping uses the D2W cfg fields directly:
      - bonded stack order: B_Sub -> B_Chip -> T_Sub -> T_Chip
      - bond temperature : cfg.T_R
      - target temperature: cfg.T_anl
      - in-plane diameter : sqrt(DIE_W_um^2 + DIE_L_um^2)
    """
    stack = _build_post_bond_stack_from_cfg(cfg)
    result = _compute_post_bond_curvature_formula10(
        layers_bottom_to_top=stack,
        bond_temperature_C=float(cfg.T_R),
        target_temperature_C=float(cfg.T_anl),
        wafer_diameter_mm=_resolve_post_bond_wafer_diameter_mm(cfg),
    )
    return float(result.warpage_um)


def plate_bending_stiffness(E: float, nu: float, h: float) -> float:
    return E * h**3 / (12.0 * (1.0 - nu**2))

def foundation_stiffness_K_effective(E1,nu1,h1,E2,nu2,h2):
    return 1.0 / ((1.0-nu1)*h1/(3.0*E1) + (1.0-nu2)*h2/(3.0*E2))

def suhir_peeling_two_wafers_bottomA_topB(waferA_eq: EqLayer, waferB_eq: EqLayer, R_m: float,
                                         sag_total_A_m: float, sag_total_B_m: float,
                                         sample_points: int = 500):
    D1 = plate_bending_stiffness(waferA_eq.E_Pa, waferA_eq.nu, waferA_eq.t_m)
    D2 = plate_bending_stiffness(waferB_eq.E_Pa, waferB_eq.nu, waferB_eq.t_m)
    K  = foundation_stiffness_K_effective(waferA_eq.E_Pa, waferA_eq.nu, waferA_eq.t_m,
                                          waferB_eq.E_Pa, waferB_eq.nu, waferB_eq.t_m)
    kappa1 = 2.0 * sag_total_A_m / (R_m**2)
    kappa2 = 2.0 * sag_total_B_m / (R_m**2)
    M = (D1 * D2) / (D1 + D2) * (kappa1 - kappa2)
    beta = ((K * (D1 + D2)) / (4.0 * D1 * D2)) ** 0.25
    p_max = K * M / (2.0 * beta * D1)  # [Pa]
    decay_len = 1.0 / beta
    return {"p_max_Pa": p_max, "beta": beta, "decay_length_m": decay_len}

def peeling_stress_at_points_vec_MPa(peel_dict: dict, coords_mm_np: np.ndarray, R_m: float) -> np.ndarray:
    """
    Retained for compatibility / debugging direct p(r) evaluation.
    Main fast path now uses radial dishing LUT in build_effcrit_and_dishing_arrays().
    """
    if coords_mm_np.ndim != 2 or coords_mm_np.shape[1] != 2:
        raise ValueError("coords_mm_np must be shape (N,2).")
    xy_m = coords_mm_np.astype(np.float64, copy=False) * 1e-3
    r_m  = np.sqrt(xy_m[:,0]**2 + xy_m[:,1]**2)
    if np.any(r_m > R_m + 1e-15):
        idx = np.where(r_m > R_m + 1e-15)[0][:5]
        raise ValueError(f"{idx.size} points lie outside wafer radius R={R_m} m, e.g. indices {idx.tolist()}")

    s = R_m - r_m
    p_max = float(peel_dict["p_max_Pa"])
    beta  = float(peel_dict["beta"])
    p_pa  = p_max * np.exp(-beta*s) * (np.cos(beta*s) - np.sin(beta*s))

    if USE_PLOT:
        plt.figure()
        plt.scatter(r_m*1e3, p_pa/1e6, s=8)
        plt.xlabel("Radius r (mm)")
        plt.ylabel("Peeling Stress p (MPa)")
        plt.title("Peeling Stress vs Radius")
        plt.grid(True)
        plt.show()

    return p_pa / 1e6  # MPa


# =============================================================================
# ====================== RADIAL DISHING LUT (NEW FAST PATH) ===================
# =============================================================================

# =============================================================================
# ===================== PUBLIC API: RADIAL LUT EXPORT / PLOT ==================
# =============================================================================

def _solve_peel_and_radius_from_cfg(cfg, *, include_global_peeling_stress: bool = True):
    """
    Internal helper:
      Given cfg, initialize pad-scale globals and return (peel_dict_or_none, R_stack_m).

    This also initializes globals/caches via __init_params(cfg).
    """
    __init_params(cfg)

    R_stack = float(min(WAFER_A.L_m, WAFER_B.L_m))
    if not include_global_peeling_stress:
        return None, R_stack

    # 1) Wafer-level stack to get peeling kernel
    resA = process_wafer(WAFER_A)  # bottom
    resB = process_wafer(WAFER_B)  # top

    # Keep same sign convention as main API
    s_total_A_m = S_INIT_A_M - resA.D_m
    s_total_B_m = S_INIT_B_M - resB.D_m

    peel = suhir_peeling_two_wafers_bottomA_topB(
        waferA_eq=resA.final_eq,
        waferB_eq=resB.final_eq,
        R_m=R_stack,
        sag_total_A_m=s_total_A_m,
        sag_total_B_m=s_total_B_m,
        sample_points=500,
    )
    return peel, float(R_stack)


def get_radial_dishing_lut_array(cfg=None, *, peel_dict=None, R_m=None,
                                 n_r: int = 4096, r_unit: str = "um",
                                 include_global_peeling_stress: bool = True) -> np.ndarray:
    """
    Public API: return radial LUT array with columns:
        [r, D_sio2_nm, D_cu_nm]

    Two usage modes:
      1) cfg mode:
           get_radial_dishing_lut_array(cfg=cfg, ...)
         -> internally solves wafer stack and peel_dict

      2) direct mode:
           get_radial_dishing_lut_array(peel_dict=..., R_m=..., ...)
         -> reuses existing fixed wafer/die state

    Parameters
    ----------
    cfg : object, optional
        External config object used by __init_params(cfg). If given, cfg mode is used.
    peel_dict : dict, optional
        Must contain at least "p_max_Pa" and "beta" in direct mode.
    R_m : float, optional
        Radius [m] in direct mode.
    n_r : int
        Number of radial samples for LUT.
    r_unit : str
        "m", "mm", or "um"
    include_global_peeling_stress : bool
        If True, include wafer-level global peeling p(r). If False, set p(r)=0 and
        invert using local peeling stress only.

    Returns
    -------
    arr : np.ndarray, shape (n_r, 3)
        columns = [r_(unit), D_sio2_nm, D_cu_nm]
    """
    # Resolve state
    if cfg is not None:
        peel_dict, R_m_local = _solve_peel_and_radius_from_cfg(
            cfg, include_global_peeling_stress=include_global_peeling_stress
        )
    else:
        if R_m is None:
            raise ValueError("Provide either cfg=... OR at least R_m=... in direct mode.")
        if include_global_peeling_stress and peel_dict is None:
            raise ValueError("When include_global_peeling_stress=True in direct mode, provide peel_dict=... and R_m=...")
        R_m_local = float(R_m)
        # In direct mode, caller must have initialized globals previously if cfg-dependent constants changed.
        # We still can proceed if current globals are already valid.
        if R_m_local <= 0.0:
            raise ValueError(f"R_m must be > 0, got {R_m_local}")

    # Build / fetch radial LUT
    lut = _ensure_radial_dishing_lut(
        peel_dict=peel_dict,
        R_m=R_m_local,
        n_r=n_r,
        include_global_peeling_stress=include_global_peeling_stress,
    )

    r_m = np.asarray(lut["r_grid_m"], dtype=np.float64)
    D_sio2 = np.asarray(lut["D_sio2_nm"], dtype=np.float64)
    D_cu   = np.asarray(lut["D_cu_nm"], dtype=np.float64)

    r_unit = str(r_unit).lower()
    if r_unit == "m":
        r_out = r_m
    elif r_unit == "mm":
        r_out = r_m * 1e3
    elif r_unit == "um":
        r_out = r_m * 1e6
    else:
        raise ValueError(f"Unsupported r_unit={r_unit}. Use 'm', 'mm', or 'um'.")

    return np.column_stack([r_out, D_sio2, D_cu])


def plot_radial_dishing_lut(cfg=None, *, peel_dict=None, R_m=None,
                            n_r: int = 4096, r_unit: str = "um",
                            show: bool = True, ax=None,
                            include_global_peeling_stress: bool = True):
    """
    Public API: plot two radial LUT curves:
      x-axis: r
      y-axis: dishing value [nm]
      curves : SiO2 and Cu

    Two usage modes:
      1) cfg mode:   plot_radial_dishing_lut(cfg=cfg, ...)
      2) direct mode plot_radial_dishing_lut(peel_dict=..., R_m=..., ...)

    Returns
    -------
    fig, ax
    """
    arr = get_radial_dishing_lut_array(
        cfg=cfg,
        peel_dict=peel_dict,
        R_m=R_m,
        n_r=n_r,
        r_unit=r_unit,
        include_global_peeling_stress=include_global_peeling_stress,
    )

    r = arr[:, 0]
    D_sio2 = arr[:, 1]
    D_cu   = arr[:, 2]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        created_fig = True
    else:
        fig = ax.figure

    ax.plot(r, D_sio2, label="SiO2 dishing LUT")
    ax.plot(r, D_cu,   label="Cu dishing LUT")

    ax.set_xlabel(f"Radius r ({r_unit})")
    ax.set_ylabel("Dishing (nm)")
    ax.set_title("Radial Dishing LUT")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Optional nicer x ticks for common units
    try:
        rmax = float(np.max(r)) if r.size else 0.0
        if r_unit.lower() == "mm" and rmax > 0:
            ax.xaxis.set_major_locator(MultipleLocator(max(1.0, round(rmax / 10.0))))
        elif r_unit.lower() == "um" and rmax > 0:
            step = max(1.0, rmax / 10.0)
            ax.xaxis.set_major_locator(MultipleLocator(step))
    except Exception:
        pass

    if created_fig:
        fig.tight_layout()
    if show:
        plt.show()

    return fig, ax




def _build_radial_dishing_lut(peel_dict: dict | None, R_m: float, n_r: int = 4096,
                              include_global_peeling_stress: bool = True) -> dict:
    """
    Build radial LUT:
      r [m] -> p_global [MPa] -> sigma_eff -> D_cu_nm / D_sio2_nm

    Valid for one fixed wafer/die state (R_m, p_max, beta) and current cfg.
    """
    global _RDISH_LUT

    if R_m <= 0.0:
        raise ValueError(f"R_m must be > 0, got {R_m}")

    # Ensure pad-scale sigma->D LUTs are ready
    _ensure_luts_ready()

    n_r = max(2, int(n_r))
    r_grid = np.linspace(0.0, float(R_m), n_r, dtype=np.float64)

    if include_global_peeling_stress:
        if peel_dict is None:
            raise ValueError("peel_dict must be provided when include_global_peeling_stress=True.")
        p_max = float(peel_dict["p_max_Pa"])
        beta  = float(peel_dict["beta"])
        s = float(R_m) - r_grid
        p_mpa = (p_max * np.exp(-beta * s) * (np.cos(beta * s) - np.sin(beta * s))) / 1e6
    else:
        p_max = 0.0
        beta = 0.0
        p_mpa = np.zeros_like(r_grid)

    # Effective thresholds and local-only inversion
    sigma_eff_Cu, sigma_eff_SiO2, D_cu_nm, D_sio2_nm = _effcrit_and_dishing_from_global_peel_MPa(p_mpa)

    _RDISH_LUT = dict(
        R_m=float(R_m),
        p_max_Pa=float(p_max),
        beta=float(beta),
        n_r=int(n_r),
        include_global_peeling_stress=bool(include_global_peeling_stress),

        r_grid_m=r_grid,
        p_global_MPa=p_mpa,

        sigma_eff_SiO2_MPa=sigma_eff_SiO2,
        sigma_eff_Cu_MPa=sigma_eff_Cu,

        D_sio2_nm=D_sio2_nm,
        D_cu_nm=D_cu_nm,
    )
    return _RDISH_LUT


def _ensure_radial_dishing_lut(peel_dict: dict | None, R_m: float, n_r: int = 4096,
                               include_global_peeling_stress: bool = True) -> dict:
    """
    Rebuild radial dishing LUT only when the controlling state changes.
    """
    global _RDISH_LUT

    R_m   = float(R_m)
    n_r   = max(2, int(n_r))

    if _RDISH_LUT is None:
        return _build_radial_dishing_lut(
            peel_dict,
            R_m,
            n_r=n_r,
            include_global_peeling_stress=include_global_peeling_stress,
        )

    same_flag = bool(_RDISH_LUT.get("include_global_peeling_stress", True)) == bool(include_global_peeling_stress)
    if not same_flag:
        return _build_radial_dishing_lut(
            peel_dict,
            R_m,
            n_r=n_r,
            include_global_peeling_stress=include_global_peeling_stress,
        )

    same = (
        float(_RDISH_LUT.get("R_m", np.nan)) == R_m and
        int(_RDISH_LUT.get("n_r", -1)) == n_r
    )

    if include_global_peeling_stress:
        if peel_dict is None:
            raise ValueError("peel_dict must be provided when include_global_peeling_stress=True.")
        p_max = float(peel_dict["p_max_Pa"])
        beta  = float(peel_dict["beta"])
        same = same and (
            float(_RDISH_LUT.get("p_max_Pa", np.nan)) == p_max and
            float(_RDISH_LUT.get("beta", np.nan)) == beta
        )

    if not same:
        return _build_radial_dishing_lut(
            peel_dict,
            R_m,
            n_r=n_r,
            include_global_peeling_stress=include_global_peeling_stress,
        )

    return _RDISH_LUT


def _query_radial_dishing_lut_from_r(r_m: np.ndarray, peel_dict: dict | None, R_m: float, n_r: int = 4096,
                                     include_global_peeling_stress: bool = True):
    """
    Query D_cu_nm / D_sio2_nm (and p_global) from radial LUT by r [m].
    """
    lut = _ensure_radial_dishing_lut(
        peel_dict,
        R_m,
        n_r=n_r,
        include_global_peeling_stress=include_global_peeling_stress,
    )

    r = np.asarray(r_m, dtype=np.float64)
    r_clip = np.clip(r, 0.0, float(lut["R_m"]))

    D_cu_nm   = np.interp(r_clip, lut["r_grid_m"], lut["D_cu_nm"])
    D_sio2_nm = np.interp(r_clip, lut["r_grid_m"], lut["D_sio2_nm"])
    p_mpa     = np.interp(r_clip, lut["r_grid_m"], lut["p_global_MPa"])

    return D_cu_nm, D_sio2_nm, p_mpa


# =============================================================================
# ===================== EFFICIENT CRITICAL & DISHING ARRAYS ===================
# =============================================================================

def build_effcrit_and_dishing_arrays(peel_dict: dict | None, coords_mm_np: np.ndarray, R_m: float, *,
                                     include_global_peeling_stress: bool = True)\
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast path:
      coords -> r -> query radial dishing LUT (r -> D_cu, D_sio2)

    Keeps the same return signature:
      effcrit: (N,2) = [sigma_eff_Cu, sigma_eff_SiO2]
      dishing:(N,2) = [D_cu_nm, D_sio2_nm]
    """
    if coords_mm_np.ndim != 2 or coords_mm_np.shape[1] != 2:
        raise ValueError("coords_mm_np must be shape (N,2).")

    # coords(mm) -> r(m)
    xy_m = coords_mm_np.astype(np.float64, copy=False) * 1e-3
    r_m  = np.sqrt(xy_m[:,0]**2 + xy_m[:,1]**2)

    if np.any(r_m > R_m + 1e-15):
        idx = np.where(r_m > R_m + 1e-15)[0][:5]
        raise ValueError(f"{idx.size} points lie outside wafer radius R={R_m} m, e.g. indices {idx.tolist()}")

    # Direct query from radial dishing LUT
    D_cu_nm, D_sio2_nm, p_MPa = _query_radial_dishing_lut_from_r(
        r_m=r_m,
        peel_dict=peel_dict,
        R_m=R_m,
        n_r=4096,
        include_global_peeling_stress=include_global_peeling_stress,
    )

    # Keep effcrit output for compatibility
    sigma_crit_SiO2, sigma_crit_Cu = _critical_peeling_pair_MPa()
    sigma_eff_SiO2 = sigma_crit_SiO2 - p_MPa
    sigma_eff_Cu   = sigma_crit_Cu   - p_MPa

    effcrit = np.column_stack([sigma_eff_Cu, sigma_eff_SiO2])
    dishing = np.column_stack([D_cu_nm, D_sio2_nm])
    return effcrit, dishing


# =============================================================================
# ================================= MAIN ======================================
# =============================================================================

# New main API: radial LUT export + plot, and fast per-pad dishing intervals from coords.

def debond_dishing_bounds_calculator(cfg,
                                     n_r: int = 4096,
                                     radial_lut_r_unit: str = "um",
                                     print_radial_lut: bool = True,
                                     radial_lut_preview_rows: int = 10,
                                     plot_radial_lut_flag: bool = True,
                                     include_global_peeling_stress: bool = True):
    """
    LUT-only main API:
      - Build radial dishing LUT from current wafer/die state
      - Return the LUT table array directly
      - Optionally print preview and plot

    Returns
    -------
    arr : np.ndarray
        columns = [r_(unit), D_sio2_nm, D_cu_nm]
    """
    peel, R_stack = _solve_peel_and_radius_from_cfg(
        cfg, include_global_peeling_stress=include_global_peeling_stress
    )

    # 2) Build / ensure radial LUT cache (this is the target LUT)
    _ensure_radial_dishing_lut(
        peel_dict=peel,
        R_m=R_stack,
        n_r=int(n_r),
        include_global_peeling_stress=include_global_peeling_stress,
    )

    # 3) Export LUT array directly from the built cache
    arr = get_radial_dishing_lut_array(
        peel_dict=peel,
        R_m=R_stack,
        n_r=int(n_r),
        r_unit=radial_lut_r_unit,
        include_global_peeling_stress=include_global_peeling_stress,
    )

    # 4) Optional table preview
    if print_radial_lut:
        n_show = int(max(1, min(radial_lut_preview_rows, arr.shape[0])))
        print(f"[Radial dishing LUT] total rows = {arr.shape[0]}")
        print(f"Columns = [r ({radial_lut_r_unit}), D_sio2_nm, D_cu_nm]")
        print(arr[:n_show])

    # 5) Optional plot
    if plot_radial_lut_flag:
        plot_radial_dishing_lut(
            peel_dict=peel,
            R_m=R_stack,
            n_r=int(n_r),
            r_unit=radial_lut_r_unit,
            show=True,
            include_global_peeling_stress=include_global_peeling_stress,
        )

    # 6) Return LUT array
    return arr


# New main API: per-pad dishing intervals from coords using radial LUT for fast query.

def debond_dishing_intervals_from_coords(cfg,
                                         coords_um: np.ndarray,
                                         *,
                                         n_r: int = 4096,
                                         return_effcrit: bool = False,
                                         return_debug: bool = False,
                                         include_global_peeling_stress: bool = True):
    """
    New main API:
      coords (um) -> per-pad dishing intervals (D_Cu_nm, D_SiO2_nm)

    Fast path:
      - Build (or reuse) radial dishing LUT for this cfg
      - coords -> r -> interp D_cu, D_sio2
      - Return per-pad [sorted(D_cu, D_sio2)]  (N,2)

    Parameters
    ----------
    cfg : object
        External config object used by __init_params(cfg).
    coords_um : np.ndarray, shape (N,2)
        Pad global coordinates in micrometers (um). Columns: [x_um, y_um].
    n_r : int
        Radial LUT resolution (default 4096).
    return_effcrit : bool
        If True, also return effcrit array (N,2) = [sigma_eff_Cu, sigma_eff_SiO2] (MPa).
    return_debug : bool
        If True, also return a debug dict with peel_dict, R_stack, and LUT cache metadata.
    include_global_peeling_stress : bool
        If True, include wafer-level global peeling p(r). If False, ignore global
        peeling and invert using local peeling stress only.

    Returns
    -------
    dishing_intervals : np.ndarray, shape (N,2)
        Each row = sorted(D_Cu_nm, D_SiO2_nm).
        (This keeps your legacy signature style.)

    If return_effcrit=True:
        returns (dishing_intervals, effcrit)

    If return_debug=True:
        returns (dishing_intervals, effcrit?, debug_dict)
    """
    # ---------------------------
    # 0) Validate coords
    # ---------------------------
    coords_um = np.asarray(coords_um, dtype=np.float64)
    if coords_um.ndim != 2 or coords_um.shape[1] != 2:
        raise ValueError("coords_um must be shape (N,2) with columns [x_um, y_um].")

    # ---------------------------
    # 1) Initialize globals + optionally solve peel_dict and R_stack
    # ---------------------------
    peel, R_stack = _solve_peel_and_radius_from_cfg(
        cfg, include_global_peeling_stress=include_global_peeling_stress
    )

    # ---------------------------
    # 2) Ensure radial LUT cache ready for this peel_dict / R_stack
    # ---------------------------
    _ensure_radial_dishing_lut(
        peel_dict=peel,
        R_m=R_stack,
        n_r=int(n_r),
        include_global_peeling_stress=include_global_peeling_stress,
    )

    # ---------------------------
    # 3) coords_um -> r_m
    # ---------------------------
    xy_m = coords_um * 1e-6
    r_m = np.sqrt(xy_m[:, 0] ** 2 + xy_m[:, 1] ** 2)

    if np.any(r_m > R_stack + 1e-15):
        idx = np.where(r_m > R_stack + 1e-15)[0][:5]
        raise ValueError(
            f"{idx.size} points lie outside radius R={R_stack} m, e.g. indices {idx.tolist()} "
            f"(r_max={float(np.max(r_m))} m)"
        )

    # ---------------------------
    # 4) Query radial LUT -> D_cu, D_sio2, p(r)
    # ---------------------------
    D_cu_nm, D_sio2_nm, p_MPa = _query_radial_dishing_lut_from_r(
        r_m=r_m,
        peel_dict=peel,
        R_m=R_stack,
        n_r=int(n_r),
        include_global_peeling_stress=include_global_peeling_stress,
    )

    # ---------------------------
    # 5) Build outputs
    # ---------------------------
    # Legacy style: return (N,2) where each row = sorted(D_Cu_nm, D_SiO2_nm)
    dishing_intervals = np.sort(np.column_stack([D_cu_nm, D_sio2_nm]), axis=1)

    out = (dishing_intervals,)

    effcrit = None
    if return_effcrit or return_debug:
        sigma_crit_SiO2, sigma_crit_Cu = _critical_peeling_pair_MPa()
        sigma_eff_SiO2 = sigma_crit_SiO2 - p_MPa
        sigma_eff_Cu   = sigma_crit_Cu   - p_MPa
        effcrit = np.column_stack([sigma_eff_Cu, sigma_eff_SiO2])  # (N,2)

    if return_effcrit:
        out = (dishing_intervals, effcrit)

    if return_debug:
        debug = {
            "R_stack_m": R_stack,
            "include_global_peeling_stress": bool(include_global_peeling_stress),
            "peel_dict": None if peel is None else dict(peel),
            "n_r": int(n_r),
            "RDISH_cache_keys": None if _RDISH_LUT is None else {
                "R_m": float(_RDISH_LUT.get("R_m", np.nan)),
                "p_max_Pa": float(_RDISH_LUT.get("p_max_Pa", np.nan)),
                "beta": float(_RDISH_LUT.get("beta", np.nan)),
                "n_r": int(_RDISH_LUT.get("n_r", -1)),
                "include_global_peeling_stress": bool(_RDISH_LUT.get("include_global_peeling_stress", True)),
            },
        }
        if return_effcrit:
            out = (dishing_intervals, effcrit, debug)
        else:
            out = (dishing_intervals, debug)

    return out[0] if len(out) == 1 else out
