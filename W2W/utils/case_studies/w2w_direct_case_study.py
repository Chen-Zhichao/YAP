#!/usr/bin/env python3
"""Direct-layout W2W yield study without materializing a pad bitmap.

The runner intentionally reuses the production W2W calculator kernels:
  * overlay geometry and pass criterion from ``overlay_yield_calculator``;
  * direction-averaged void-tail integral from ``defect_yield_calculator``;
  * one-level Cu recess probability from ``Cu_expansion_yield_calculator``.

Only the layout adapter is analytical: a dense rectangular critical region is
written directly onto the same coarse tail-modeling grid that the particle
calculator uses internally.  This avoids allocating a 500 mm^2 / 1 um bitmap.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
from scipy.ndimage import distance_transform_edt

from Cu_expansion_yield_calculator import cu_recess_pad_pass_probabilities
from defect_yield_calculator import (
    _model_grid_coords,
    _particle_density_chunk_um2,
    _particle_thickness_quadrature,
    _tail_dilation_fatal_integral,
)
from overlay_yield_calculator import (
    _residual_pass_probability,
    max_allowed_misalignment_calculator,
)
from wafer_die_stack_initialization import Wafer_Interface


WAFER_RADIUS_UM = 150_000.0
DIE_SAW_WIDTH_UM = 100.0


@dataclass(frozen=True)
class Case:
    area_mm2: float
    pitch_um: float
    critical_fraction: float

    @property
    def die_side_um(self) -> float:
        return float(np.sqrt(self.area_mm2) * 1_000.0)

    @property
    def active_side_um(self) -> float:
        return self.die_side_um * np.sqrt(self.critical_fraction)

    @property
    def active_pad_count(self) -> int:
        return int(round(self.critical_fraction * self.area_mm2 * 1e6 / self.pitch_um**2))


def _cfg(base, **overrides):
    data = OmegaConf.to_container(base, resolve=True)
    data.update(overrides)
    return SimpleNamespace(**data)


def _wafer_interface(case: Case):
    interface = Wafer_Interface(
        wafer_radius=WAFER_RADIUS_UM,
        DIE_W_um=case.die_side_um,
        DIE_L_um=case.die_side_um,
        PAD_ARR_ROW=1,
        PAD_ARR_COL=1,
        PAD_TOP_R_um=0.0,
        PAD_BOT_R_um=0.0,
        base_pad_coords=np.array([[0.0, 0.0]]),
        dice_width=DIE_SAW_WIDTH_UM,
        pad_yield_flag=False,
    )
    vertices = np.array([
        [-case.die_side_um / 2.0, case.die_side_um / 2.0],
        [case.die_side_um / 2.0, case.die_side_um / 2.0],
        [-case.die_side_um / 2.0, -case.die_side_um / 2.0],
        [case.die_side_um / 2.0, -case.die_side_um / 2.0],
    ])
    active = case.active_side_um / 2.0
    pad_box = np.array([
        [-active, active], [active, active], [-active, -active], [active, -active],
    ])
    interface.generate_die(1, vertices, pad_box)
    return interface


def _overlay_yield_array(cfg, case: Case, interface: Wafer_Interface, samples=200_000):
    """Production W2W overlay criterion over absolute wafer coordinates.

    The requested study isolates rotation: translation, magnification, and
    random residual overlay are configured as zero.  The only sampling is the
    user-specified systematic rotation distribution.
    """
    threshold = max_allowed_misalignment_calculator(
        cfg,
        cfg.PAD_TOP_R_um,
        cfg.PAD_BOT_R_um,
        cfg.PITCH_r_um,
        cfg.PITCH_c_um,
        cfg.CONTACT_AREA_CONSTRAINT,
        cfg.CRITICAL_DIST_CONSTRAINT,
    )
    rng = np.random.default_rng(20260721)
    theta = rng.normal(cfg.SYSTEM_ROTATION_MEAN_rad, cfg.SYSTEM_ROTATION_STD_rad, samples)
    corners = np.asarray([die.pad_array_box for die in interface.die_list], dtype=float)
    result = np.empty(len(interface.die_list), dtype=float)
    for start in range(0, len(result), 512):
        stop = min(start + 512, len(result))
        coords = corners[start:stop]
        x = coords[None, :, :, 0]
        y = coords[None, :, :, 1]
        error = np.abs(theta[:, None, None]) * np.sqrt(x * x + y * y)
        worst = np.max(error, axis=2)
        result[start:stop] = np.mean(
            _residual_pass_probability(
                threshold - worst,
                cfg.RANDOM_MISALIGNMENT_MEAN_um,
                cfg.RANDOM_MISALIGNMENT_STD_um,
            ),
            axis=0,
        )
    return result, threshold


def _particle_yield_array(cfg, case: Case, interface: Wafer_Interface):
    """Call W2W's production tail integral with an analytical layout cache."""
    max_radius = max(float(np.linalg.norm(die.die_center)) for die in interface.die_list)
    thickness, weights = _particle_thickness_quadrature(cfg)
    radius_scale = cfg.k_r * max_radius + cfg.k_r0
    tail_scale = cfg.k_L * max_radius
    margin = (
        max(radius_scale, tail_scale) * np.sqrt(float(np.max(thickness)))
        + cfg.PAD_TOP_R_um
    )
    grid_pitch = float(cfg.DEFECT_TAIL_GRID_PITCH_um)
    x, y = _model_grid_coords(cfg, grid_pitch, margin_um=margin)
    xx, yy = np.meshgrid(x, y)
    half_active = case.active_side_um / 2.0
    critical = (np.abs(xx) <= half_active) & (np.abs(yy) <= half_active)
    dy = abs(y[0] - y[1])
    dx = abs(x[1] - x[0])
    dist = distance_transform_edt(~critical, sampling=(dy, dx))
    cache = {
        "critical_region_mask": critical,
        "redundant_cell_data": None,
        "x_coords_um": x,
        "y_coords_um": y,
        "pitch_r_eff_um": dy,
        "pitch_c_eff_um": dx,
        "cell_area_um2": dx * dy,
        "required_main_radius_um": np.maximum(dist - cfg.PAD_TOP_R_um, 0.0),
        "density": _particle_density_chunk_um2(cfg, x, y),
        "angles_rad": np.linspace(0.0, 2.0 * np.pi, int(cfg.DEFECT_TAIL_NUM_ANGLES), endpoint=False),
        "thickness_um": thickness,
        "thickness_weights": weights,
        "tail_grid_pitch_um": grid_pitch,
        "tail_model_margin_um": margin,
        "tail_num_angles": int(cfg.DEFECT_TAIL_NUM_ANGLES),
        "tail_num_thickness_nodes": len(thickness),
        "num_redundant_pads": 0,
        "redundant_group_count": 0,
    }
    yields = np.empty(len(interface.die_list), dtype=float)
    for idx, die in enumerate(interface.die_list):
        integral, _ = _tail_dilation_fatal_integral(
            cfg,
            {"CRITICAL_PAD_BITMAP": np.ones((1, 1), dtype=bool)},
            die_center_radius_um=float(np.linalg.norm(die.die_center)),
            layout_cache=cache,
        )
        yields[idx] = np.exp(-integral)
    return yields


def _mechanical_yield(cfg, case: Case):
    # User-requested independent-pad scaling.  The safe range is supplied by
    # the same one-level Cu model used in W2W's stress calculator.
    safe_lower_nm = -30.044
    safe_upper_nm = 0.0
    pad_yield = cu_recess_pad_pass_probabilities(
        cfg.TOP_DISH_MEAN_nm + cfg.BOT_DISH_MEAN_nm,
        np.array([safe_lower_nm]),
        np.array([safe_upper_nm]),
        np.hypot(cfg.TOP_DISH_STD_nm, cfg.BOT_DISH_STD_nm),
    )[0]
    return float(pad_yield), float(pad_yield**case.active_pad_count)


def run_case(base_cfg, case: Case):
    cfg = _cfg(
        base_cfg,
        DIE_W_um=case.die_side_um,
        DIE_L_um=case.die_side_um,
        PAD_ARR_W_um=case.die_side_um,
        PAD_ARR_L_um=case.die_side_um,
        PAD_ARR_ROW=1,
        PAD_ARR_COL=1,
        PITCH_r_um=case.pitch_um,
        PITCH_c_um=case.pitch_um,
        PAD_TOP_R_um=case.pitch_um / 4.0,
        PAD_BOT_R_um=case.pitch_um / 4.0,
        WAF_R_um=WAFER_RADIUS_UM,
        dice_width=DIE_SAW_WIDTH_UM,
        SYSTEM_TRANSLATION_MEAN_um=0.0,
        SYSTEM_TRANSLATION_STD_um=0.0,
        SYSTEM_TRANSLATION_X_MEAN_um=0.0,
        SYSTEM_TRANSLATION_X_STD_um=0.0,
        SYSTEM_TRANSLATION_Y_MEAN_um=0.0,
        SYSTEM_TRANSLATION_Y_STD_um=0.0,
        RANDOM_MISALIGNMENT_MEAN_um=0.0,
        RANDOM_MISALIGNMENT_STD_um=0.0,
        k_mag=0.0,
        M_0=0.0,
    )
    interface = _wafer_interface(case)
    overlay, threshold = _overlay_yield_array(cfg, case, interface)
    particle = _particle_yield_array(cfg, case, interface)
    p_pad_mech, mechanical = _mechanical_yield(cfg, case)
    total = overlay * particle * mechanical
    return {
        "dies_per_wafer": interface.num_dies,
        "threshold_um": threshold,
        "overlay_mean": float(np.mean(overlay)),
        "particle_mean": float(np.mean(particle)),
        "mechanical": mechanical,
        "pad_mechanical": p_pad_mech,
        "total_mean": float(np.mean(total)),
    }


def main():
    base = OmegaConf.load(
        "configs/design_6/Memory_DRAM_0_From_TSV_Interconnect_Die.yaml"
    )
    base.TOP_DISH_MEAN_nm = -5.0
    base.BOT_DISH_MEAN_nm = -5.0
    base.TOP_DISH_STD_nm = 1.0
    base.BOT_DISH_STD_nm = 1.0
    base.D0 = 1e-11
    base.D1 = 1e-11
    base.SYSTEM_ROTATION_MEAN_rad = 4e-6
    base.SYSTEM_ROTATION_STD_rad = 1e-6

    for critical_fraction, label in ((1.0, "0% redundancy"), (0.9, "10% redundancy")):
        for area in (50.0, 500.0):
            for pitch in (5.0, 1.0):
                result = run_case(base, Case(area, pitch, critical_fraction))
                print(
                    f"{label}, {area:g} mm2, {pitch:g} um: "
                    f"dies={result['dies_per_wafer']}, "
                    f"overlay={result['overlay_mean']:.6f}, "
                    f"particle={result['particle_mean']:.6f}, "
                    f"mechanical={result['mechanical']:.6f}, "
                    f"total={result['total_mean']:.6f}"
                )


if __name__ == "__main__":
    main()
