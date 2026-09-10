#!/usr/bin/env python3
"""Calculator-faithful W2W large-chiplet case study without pad bitmaps.

The regular layouts in this study have either an all-critical pad field or a
centered critical square surrounded by dummy pads.  Storing their 1-um
bitmaps would require hundreds of millions of cells.  This adapter therefore
constructs the *same coarse tail-dilation grid* used by
``W2W.defect_yield_calculator`` directly from the regular geometry, then calls
the calculator's own tail kernel.  Overlay is evaluated by the production W2W
overlay calculator on the production wafer die placement.

This file is deliberately a case-study driver: it does not alter the main
YAP modeling flow.
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.ndimage import distance_transform_edt

from Cu_expansion_yield_calculator import (
    cu_recess_pad_pass_probabilities,
    debond_dishing_intervals_from_coords,
)
from defect_yield_calculator import (
    _model_grid_coords,
    _particle_density_chunk_um2,
    _particle_thickness_quadrature,
    _tail_dilation_fatal_integral,
)
from overlay_yield_calculator import stack_overlay_yield_calculator
from wafer_die_stack_initialization import WaferStack


# This study is intentionally a two-wafer bond, not the four-interface
# design_6 stack.  Keep the physical parameters from design_6's first
# interface, but run exactly one W2W bonding interface.
INTERFACE_FILES = ("Memory_DRAM_0_From_TSV_Interconnect_Die.yaml",)


def _load_design6_cfgs(config_dir: Path):
    cfg_dict = {}
    for filename in INTERFACE_FILES:
        cfg = OmegaConf.load(config_dir / filename)
        cfg_dict[str(cfg.INTERFACE)] = cfg
    return cfg_dict


def _configure_regular_chiplet(
    cfg_dict,
    *,
    area_mm2: float,
    pitch_um: float,
    particle_density_um2: float,
    rotation_mean_rad: float,
    rotation_std_rad: float,
    overlay_samples: int,
    dish_mean_nm: float,
    dish_std_nm: float,
):
    side_um = np.sqrt(area_mm2) * 1_000.0
    # Array dimensions follow the same center-to-center convention as YAP's
    # pad geometry.  The active pad field is therefore one pitch smaller than
    # the physical die edge in each dimension.
    n_axis = max(1, int(np.floor(side_um / pitch_um)))
    pad_arr_span_um = (n_axis - 1) * pitch_um
    pad_radius_um = pitch_um / 4.0

    for cfg in cfg_dict.values():
        cfg.DIE_W_um = float(side_um)
        cfg.DIE_L_um = float(side_um)
        cfg.PITCH_r_um = float(pitch_um)
        cfg.PITCH_c_um = float(pitch_um)
        cfg.PAD_ARR_ROW = int(n_axis)
        cfg.PAD_ARR_COL = int(n_axis)
        cfg.PAD_ARR_W_um = float(pad_arr_span_um)
        cfg.PAD_ARR_L_um = float(pad_arr_span_um)
        cfg.PAD_TOP_R_um = float(pad_radius_um)
        cfg.PAD_BOT_R_um = float(pad_radius_um)
        cfg.SYSTEM_ROTATION_MEAN_rad = float(rotation_mean_rad)
        cfg.SYSTEM_ROTATION_STD_rad = float(rotation_std_rad)
        cfg.num_samples = int(overlay_samples)
        # The case study isolates position-dependent W2W rotation.  It does
        # not include bow-induced magnification error.
        cfg.k_mag = 0.0
        cfg.M_0 = 0.0
        cfg.D0 = float(particle_density_um2)
        cfg.D1 = float(particle_density_um2)
        cfg.TOP_DISH_MEAN_nm = float(dish_mean_nm)
        cfg.BOT_DISH_MEAN_nm = float(dish_mean_nm)
        cfg.TOP_DISH_STD_nm = float(dish_std_nm)
        cfg.BOT_DISH_STD_nm = float(dish_std_nm)

    return side_um, n_axis, pad_arr_span_um


def _minimal_collection(num_critical_pads: int):
    """Enough metadata for W2W's wafer/die placement; never used as a bitmap."""
    return {
        "num_critical_pads": int(num_critical_pads),
        "num_redundant_pads": 0,
        "num_dummy_pads": 0,
        "num_power_ground_pads": 0,
        # A non-None placeholder prevents wafer_interface_initialize from
        # materializing the full 1-um pad-coordinate matrix.
        "pad_coords": np.zeros((1, 2), dtype=np.float64),
    }


def _regular_tail_layout_cache(cfg, *, critical_fraction: float, max_die_radius_um: float):
    """Build the production tail-kernel cache for a centered square layout."""
    thickness_um, thickness_weights = _particle_thickness_quadrature(cfg)
    max_sqrt_thickness = np.sqrt(float(np.max(thickness_um)))
    radius_scale_um = float(cfg.k_r) * max_die_radius_um + float(cfg.k_r0)
    tail_scale_um = float(cfg.k_L) * max_die_radius_um
    length_cap_um = float(
        getattr(cfg, "DEFECT_TAIL_LENGTH_CAP_um", tail_scale_um * max_sqrt_thickness)
    )
    model_margin_um = max(
        length_cap_um,
        radius_scale_um * max_sqrt_thickness,
    ) + float(cfg.PAD_TOP_R_um)
    grid_pitch_um = float(cfg.DEFECT_TAIL_GRID_PITCH_um)
    x_coords_um, y_coords_um = _model_grid_coords(
        cfg,
        grid_pitch_um,
        margin_um=model_margin_um,
    )
    dx_um = float(abs(x_coords_um[1] - x_coords_um[0]))
    dy_um = float(abs(y_coords_um[0] - y_coords_um[1]))

    # This is exactly the coarse-grid occupancy that the production rasterizer
    # would create for a completely filled centered square of critical pads.
    active_w_um = float(cfg.PAD_ARR_W_um) * np.sqrt(critical_fraction)
    active_l_um = float(cfg.PAD_ARR_L_um) * np.sqrt(critical_fraction)
    critical_region_mask = (
        (np.abs(x_coords_um)[None, :] <= active_w_um / 2.0 + dx_um / 2.0)
        & (np.abs(y_coords_um)[:, None] <= active_l_um / 2.0 + dy_um / 2.0)
    )
    dist_to_critical_center_um = distance_transform_edt(
        ~critical_region_mask,
        sampling=(dy_um, dx_um),
    )
    required_main_radius_um = np.maximum(
        dist_to_critical_center_um - float(cfg.PAD_TOP_R_um),
        0.0,
    )
    n_angles = max(1, int(cfg.DEFECT_TAIL_NUM_ANGLES))
    return {
        "critical_region_mask": critical_region_mask,
        "redundant_cell_data": None,
        "x_coords_um": x_coords_um,
        "y_coords_um": y_coords_um,
        "pitch_r_eff_um": dy_um,
        "pitch_c_eff_um": dx_um,
        "cell_area_um2": dx_um * dy_um,
        "required_main_radius_um": required_main_radius_um,
        "density": _particle_density_chunk_um2(cfg, x_coords_um, y_coords_um),
        "angles_rad": np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False),
        "thickness_um": thickness_um,
        "thickness_weights": thickness_weights,
        "tail_grid_pitch_um": grid_pitch_um,
        "tail_model_margin_um": model_margin_um,
        "tail_num_angles": n_angles,
        "tail_num_thickness_nodes": len(thickness_um),
        "num_redundant_pads": 0,
        "redundant_group_count": 0,
    }


def _w2w_particle_yield_array(cfg, wafer_interface, *, critical_fraction: float):
    """Use the production W2W tail kernel once per radial die group."""
    radial_bin_um = float(
        getattr(cfg, "DEFECT_RADIAL_BIN_UM", max(cfg.DIE_W_um, cfg.DIE_L_um))
    )
    radial_decimals = int(getattr(cfg, "DEFECT_RADIAL_DECIMALS", 0))
    layers = wafer_interface.get_die_radial_layers(
        radius_bin_um=radial_bin_um,
        decimals=radial_decimals,
    )["layers"]
    max_radius_um = max(
        float(np.linalg.norm(wafer_interface.die_list[int(group["representative_index"])].die_center))
        for group in layers
    )
    cache = _regular_tail_layout_cache(
        cfg,
        critical_fraction=critical_fraction,
        max_die_radius_um=max_radius_um,
    )
    die_yield = np.empty(wafer_interface.num_dies, dtype=np.float64)
    for group in layers:
        die = wafer_interface.die_list[int(group["representative_index"])]
        fatal_particles, _ = _tail_dilation_fatal_integral(
            cfg,
            {},
            die_center_radius_um=float(np.linalg.norm(die.die_center)),
            layout_cache=cache,
        )
        die_yield[group["indices"]] = np.exp(-fatal_particles)
    return die_yield


def _mechanical_chiplet_yield(cfg, num_critical_pads: int):
    """Production safe interval + independent one-level pad yield to N pads."""
    intervals = debond_dishing_intervals_from_coords(
        cfg,
        np.zeros((1, 2), dtype=np.float64),
    )
    lower_limit = float(np.clip(-intervals[0, 1] * 2.0, a_min=None, a_max=0.0))
    upper_limit = float(np.clip(-intervals[0, 0] * 2.0, a_min=None, a_max=0.0))
    p_pad = cu_recess_pad_pass_probabilities(
        float(cfg.TOP_DISH_MEAN_nm + cfg.BOT_DISH_MEAN_nm),
        np.array([lower_limit]),
        np.array([upper_limit]),
        float(np.hypot(cfg.TOP_DISH_STD_nm, cfg.BOT_DISH_STD_nm)),
    )[0]
    return float(np.exp(num_critical_pads * np.log(max(p_pad, 1e-300)))), p_pad, (
        lower_limit,
        upper_limit,
    )


def run_case(
    *,
    area_mm2: float,
    pitch_um: float,
    critical_fraction: float,
    particle_density_um2: float,
    rotation_mean_rad: float,
    rotation_std_rad: float,
    overlay_samples: int,
    overlay_seed: int,
    dish_mean_nm: float,
    dish_std_nm: float,
    config_dir: Path,
    stack_config_path: Path,
    package_area_mm2: float,
):
    cfg_dict = _load_design6_cfgs(config_dir)
    _, n_axis, _ = _configure_regular_chiplet(
        cfg_dict,
        area_mm2=area_mm2,
        pitch_um=pitch_um,
        particle_density_um2=particle_density_um2,
        rotation_mean_rad=rotation_mean_rad,
        rotation_std_rad=rotation_std_rad,
        overlay_samples=overlay_samples,
        dish_mean_nm=dish_mean_nm,
        dish_std_nm=dish_std_nm,
    )
    num_pads = n_axis * n_axis
    num_critical = int(round(critical_fraction * num_pads))
    collection = _minimal_collection(num_critical)
    collections = {name: copy.deepcopy(collection) for name in cfg_dict}
    waf_stack = WaferStack(cfg_dict, collections, mode="w2w_modeling")

    # Exact production W2W overlay calculator, including sequential-stack bow
    # statistics and absolute wafer coordinates for all die corners.
    # Replaying the same global-overlay samples makes geometry comparisons
    # deterministic: any result difference then comes from the chiplet layout,
    # not a separate Monte Carlo draw.
    np.random.seed(overlay_seed)
    stack_overlay_yield_calculator(cfg_dict, waf_stack, str(stack_config_path))

    interface_results = {}
    overlay_arrays = []
    particle_arrays = []
    mechanical_values = []
    for name, cfg in cfg_dict.items():
        interface = waf_stack.interfaces.interface_dict[name]
        overlay = waf_stack.die_yield_list_per_interface_dict[name]["overlay"]
        particle = _w2w_particle_yield_array(
            cfg,
            interface,
            critical_fraction=critical_fraction,
        )
        mechanical, p_pad, safe_interval = _mechanical_chiplet_yield(
            cfg,
            num_critical,
        )
        overlay_arrays.append(overlay)
        particle_arrays.append(particle)
        mechanical_values.append(mechanical)
        interface_results[name] = {
            "overlay": float(np.mean(overlay)),
            "particle": float(np.mean(particle)),
            "mechanical": mechanical,
            "one_pad_mechanical": p_pad,
            "safe_interval_nm": safe_interval,
        }

    overlay_stack = np.prod(np.vstack(overlay_arrays), axis=0)
    particle_stack = np.prod(np.vstack(particle_arrays), axis=0)
    mechanical_stack = float(np.prod(mechanical_values))
    combined_stack = overlay_stack * particle_stack * mechanical_stack
    single_chiplet = {
        "overlay": float(np.mean(overlay_stack)),
        "particle": float(np.mean(particle_stack)),
        "mechanical": mechanical_stack,
        "overall": float(np.mean(combined_stack)),
    }
    package_chiplets = int(round(package_area_mm2 / area_mm2))
    # The production W2W framework reports the wafer-average die-stack yield.
    # This package number is the separate independent-chiplet composition rule.
    package = {name: value ** package_chiplets for name, value in single_chiplet.items()}
    return {
        "area_mm2": area_mm2,
        "pitch_um": pitch_um,
        "critical_fraction": critical_fraction,
        "layout_label": "0% redundancy" if critical_fraction == 1.0 else "10% redundancy",
        "num_pads": num_pads,
        "num_critical_pads": num_critical,
        "dies_per_wafer": waf_stack.num_dies_per_wafer,
        "interfaces": interface_results,
        "single_chiplet": single_chiplet,
        "package_chiplets": package_chiplets,
        "package": package,
    }


def _print_result(result):
    print(
        f"{result['layout_label']} | {result['area_mm2']:g} mm^2 | "
        f"{result['pitch_um']:g} um | {result['num_pads']:,} pads | "
        f"{result['dies_per_wafer']} dies/wafer"
    )
    print("  Per-interface wafer-average yield:")
    for name, values in result["interfaces"].items():
        print(
            f"    {name}: overlay={values['overlay']:.6f}, "
            f"particle={values['particle']:.6f}, "
            f"mechanical={values['mechanical']:.6f} "
            f"(p_pad={values['one_pad_mechanical']:.12f}, "
            f"safe=[{values['safe_interval_nm'][0]:.4f}, "
            f"{values['safe_interval_nm'][1]:.4f}] nm)"
        )
    single = result["single_chiplet"]
    package = result["package"]
    print(
        "  W2W wafer-average die-stack yield: "
        f"overlay={single['overlay']:.6f}, particle={single['particle']:.6f}, "
        f"mechanical={single['mechanical']:.6f}, overall={single['overall']:.6f}"
    )
    print(
        f"  {result['package_chiplets']}-chiplet 50,000-mm^2 package "
        "(independent-chiplet composition): "
        f"overlay={package['overlay']:.6g}, particle={package['particle']:.6g}, "
        f"mechanical={package['mechanical']:.6g}, overall={package['overall']:.6g}"
    )


def _yield_table_rows(results):
    for result in results:
        single = result["single_chiplet"]
        package = result["package"]
        yield {
            "Layout": result["layout_label"],
            "Chiplet area (mm^2)": result["area_mm2"],
            "Pitch (um)": result["pitch_um"],
            "Pads/chiplet": result["num_pads"],
            "Dies/wafer": result["dies_per_wafer"],
            "Chiplets/system": result["package_chiplets"],
            "Single overlay": single["overlay"],
            "Single particle": single["particle"],
            "Single mechanical": single["mechanical"],
            "Single overall": single["overall"],
            "System overlay": package["overlay"],
            "System particle": package["particle"],
            "System mechanical": package["mechanical"],
            "System overall": package["overall"],
        }


def _save_yield_tables(results, *, output_dir: Path, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(_yield_table_rows(results))
    csv_path = output_dir / "w2w_large_chiplet_yield_table.csv"
    markdown_path = output_dir / "w2w_large_chiplet_yield_table.md"
    fieldnames = list(rows[0])
    yield_columns = (
        "Single overlay",
        "Single particle",
        "Single mechanical",
        "Single overall",
        "System overlay",
        "System particle",
        "System mechanical",
        "System overall",
    )
    csv_rows = []
    for row in rows:
        csv_row = row.copy()
        for column in yield_columns:
            csv_row[column] = f"{float(csv_row[column]):.6f}"
        csv_rows.append(csv_row)
    with csv_path.open("w", newline="", encoding="ascii") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "# W2W Large-Chiplet Yield Case Study",
        "",
        "- Two wafers, one bonding interface",
        f"- D0 = D1 = {args.d0[0]:.1e} /um^2",
        f"- Rotation: {args.rotation_mean:.1e} +/- {args.rotation_std:.1e} rad",
        "- Magnification: k_mag = 0, M_0 = 0",
        f"- Overlay samples: {args.overlay_samples:,}; seed: {args.overlay_seed}",
        f"- System area: {args.package_area:,.0f} mm^2",
        "",
        "| Layout | Chiplet area | Pitch | Single overlay | Single particle | Single mechanical | Single overall | System overlay | System particle | System mechanical | System overall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {Layout} | {area:g} mm^2 | {pitch:g} um | {so:.6f} | {sp:.6f} | "
            "{sm:.6f} | {st:.6f} | {yo:.6f} | {yp:.6f} | {ym:.6f} | {yt:.6f} |".format(
                Layout=row["Layout"],
                area=row["Chiplet area (mm^2)"],
                pitch=row["Pitch (um)"],
                so=row["Single overlay"],
                sp=row["Single particle"],
                sm=row["Single mechanical"],
                st=row["Single overall"],
                yo=row["System overlay"],
                yp=row["System particle"],
                ym=row["System mechanical"],
                yt=row["System overall"],
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"Saved yield tables: {csv_path} and {markdown_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotation-mean", type=float, default=5e-8)
    parser.add_argument("--rotation-std", type=float, default=1e-8)
    parser.add_argument("--overlay-samples", type=int, default=10_000)
    parser.add_argument("--overlay-seed", type=int, default=20260721)
    parser.add_argument("--d0", type=float, nargs="+", default=[1e-11, 1e-10])
    parser.add_argument("--dish-mean", type=float, default=-5.0)
    parser.add_argument("--dish-std", type=float, default=1.0)
    parser.add_argument("--package-area", type=float, default=50_000.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config_dir = root / "configs" / "design_6"
    stack_config_path = root / "two_wafer_case_study.3dbx"
    output_dir = args.output_dir or root / "output" / "case_studies"
    results = []
    for density in args.d0:
        print(f"\nD0 = D1 = {density:.1e} /um^2")
        for critical_fraction in (1.0, 0.9):
            for area_mm2 in (50.0, 500.0):
                for pitch_um in (5.0, 1.0):
                    result = run_case(
                        area_mm2=area_mm2,
                        pitch_um=pitch_um,
                        critical_fraction=critical_fraction,
                        particle_density_um2=density,
                        rotation_mean_rad=args.rotation_mean,
                        rotation_std_rad=args.rotation_std,
                        overlay_samples=args.overlay_samples,
                        overlay_seed=args.overlay_seed,
                        dish_mean_nm=args.dish_mean,
                        dish_std_nm=args.dish_std,
                        config_dir=config_dir,
                        stack_config_path=stack_config_path,
                        package_area_mm2=args.package_area,
                    )
                    results.append(result)
                    if not args.quiet:
                        _print_result(result)
    _save_yield_tables(results, output_dir=output_dir, args=args)


if __name__ == "__main__":
    main()
