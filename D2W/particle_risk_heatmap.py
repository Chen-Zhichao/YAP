#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from defect_yield_calculator import pad_defect_yield_map_generator
from defect_yield_simulator import get_particle_density_map
from utils.util import (
    _upsample_pad_yield_map,
    convert_3dblox_to_pad_bitmap,
    get_config_dict,
    get_single_interface_config_dict,
)
from wafer_die_stack_initialization import DieStack


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw particle-only pad risk heatmaps for D2W hybrid bonding."
    )
    parser.add_argument("--config", "-c", required=True, help="Path to modeling config yaml")
    parser.add_argument(
        "--mode",
        "-m",
        default="d2w_modeling",
        help="Mode to load from config (default: d2w_modeling)",
    )
    parser.add_argument("--ds_dir", required=True, help="Path to design directory")
    parser.add_argument("--bmap", "-b", required=True, help="Path to .bmap file")
    parser.add_argument("--criticality", "-cr", required=True, help="Path to criticality file")
    parser.add_argument(
        "--grid-size",
        type=int,
        default=400,
        help="Grid size for the continuous particle-density heatmap",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    return parser.parse_args()


def particle_density_at_pad_coords(*, cfg, die, D0) -> np.ndarray:
    D1 = float(cfg.get("D1", D0))
    edge_region_width_um = float(cfg.get("EDGE_REGION_WIDTH_um", 300.0))

    local_density = np.full(die.pad_coords.shape[0], float(D0), dtype=np.float64)
    if D1 <= D0 or edge_region_width_um <= 0:
        return local_density

    effective_edge_width_um = min(
        float(edge_region_width_um),
        die.DIE_W_um / 2.0,
        die.DIE_L_um / 2.0,
    )
    if effective_edge_width_um <= 0:
        return local_density

    dist_to_nearest_edge = np.minimum(
        die.DIE_W_um / 2.0 - np.abs(die.pad_coords[:, 0]),
        die.DIE_L_um / 2.0 - np.abs(die.pad_coords[:, 1]),
    )
    edge_weight = np.clip(
        1.0 - dist_to_nearest_edge / effective_edge_width_um,
        0.0,
        1.0,
    )
    return local_density + (float(D1) - float(D0)) * edge_weight


def grid_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(np.sort(centers), dtype=np.float64)
    if centers.size == 0:
        raise ValueError("Need at least one center coordinate to build grid edges.")
    if centers.size == 1:
        half_pitch = 0.5
        return np.array([centers[0] - half_pitch, centers[0] + half_pitch], dtype=np.float64)

    midpoints = 0.5 * (centers[:-1] + centers[1:])
    left_edge = centers[0] - 0.5 * (centers[1] - centers[0])
    right_edge = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.concatenate(([left_edge], midpoints, [right_edge]))


def draw_particle_pad_risk_heatmap(*, cfg, die, particle_failure_risk_map, output_path):
    valid_mask = ~np.isnan(die.pad_coords[:, 0]) & ~np.isnan(die.pad_coords[:, 1])
    x_centers = np.unique(die.pad_coords[valid_mask, 0])
    y_centers = np.unique(die.pad_coords[valid_mask, 1])

    x_edges = grid_edges_from_centers(x_centers)
    y_edges = grid_edges_from_centers(y_centers)
    risk_mesh = np.flipud(np.ma.masked_invalid(particle_failure_risk_map))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        risk_mesh,
        cmap="hot",
        shading="flat",
    )
    fig.colorbar(mesh, ax=ax, label="Particle Failure Probability", format="%.2e")
    ax.set_xlim(-die.DIE_W_um / 2.0, die.DIE_W_um / 2.0)
    ax.set_ylim(-die.DIE_L_um / 2.0, die.DIE_L_um / 2.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(
        "Pad Particle-Induced Void Risk\n"
        f"D0={float(cfg.D0):.3e}, D1={float(cfg.get('D1', cfg.D0)):.3e}, "
        f"w={float(cfg.get('EDGE_REGION_WIDTH_um', 300.0)):.1f} um"
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_density_vs_risk_comparison(
    *,
    cfg,
    die,
    density_x,
    density_y,
    density_map,
    particle_failure_risk_map,
    pad_density,
    output_path,
):
    valid_mask = ~np.isnan(die.pad_coords[:, 0]) & ~np.isnan(die.pad_coords[:, 1])
    x_centers = np.unique(die.pad_coords[valid_mask, 0])
    y_centers = np.unique(die.pad_coords[valid_mask, 1])

    x_edges = grid_edges_from_centers(x_centers)
    y_edges = grid_edges_from_centers(y_centers)
    risk_mesh = np.flipud(np.ma.masked_invalid(particle_failure_risk_map))
    particle_failure_probability = particle_failure_risk_map.flatten()[valid_mask]
    corr = np.corrcoef(pad_density[valid_mask], particle_failure_probability)[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=200, constrained_layout=True)

    density_im = axes[0].imshow(
        density_map,
        origin="lower",
        extent=[density_x[0], density_x[-1], density_y[0], density_y[-1]],
        cmap="hot",
        aspect="equal",
    )
    fig.colorbar(density_im, ax=axes[0], label="Particle Density (1/um^2)", format="%.2e")
    axes[0].set_xlim(-die.DIE_W_um / 2.0, die.DIE_W_um / 2.0)
    axes[0].set_ylim(-die.DIE_L_um / 2.0, die.DIE_L_um / 2.0)
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    axes[0].set_title("Continuous Particle Density")

    risk_im = axes[1].pcolormesh(
        x_edges,
        y_edges,
        risk_mesh,
        cmap="hot",
        shading="flat",
    )
    fig.colorbar(risk_im, ax=axes[1], label="Particle Failure Probability", format="%.2e")
    axes[1].set_xlim(-die.DIE_W_um / 2.0, die.DIE_W_um / 2.0)
    axes[1].set_ylim(-die.DIE_L_um / 2.0, die.DIE_L_um / 2.0)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("y (um)")
    axes[1].set_title(f"Pad Particle Risk\nPad-density correlation = {corr:.4f}")

    fig.suptitle(
        f"{cfg.INTERFACE}: particle density vs pad contamination risk",
        fontsize=12,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return corr


def main():
    args = parse_args()

    ds_name = os.path.basename(os.path.normpath(args.ds_dir))
    cfg_skeleton = OmegaConf.load(args.config)[args.mode]
    _3dbv_path = os.path.join(args.ds_dir, "generated_chiplet_definitions.3dbv")
    _3dbx_path = os.path.join(args.ds_dir, "generated_stack_config.3dbx")
    cfg_folder = args.config.rsplit("/", 1)[0]

    if os.path.exists(_3dbv_path) and os.path.exists(_3dbx_path):
        cfg_dict = get_config_dict(
            cfg_folder=cfg_folder,
            cfg_skeleton=cfg_skeleton,
            ds_name=ds_name,
            input_ds_dir=args.ds_dir,
            _3dbv_path=_3dbv_path,
            _3dbx_path=_3dbx_path,
            mode=args.mode,
            debug=args.debug,
        )
    else:
        cfg_dict = get_single_interface_config_dict(
            cfg_folder=cfg_folder,
            cfg_skeleton=cfg_skeleton,
            ds_name=ds_name,
            input_ds_dir=args.ds_dir,
            mode=args.mode,
            debug=args.debug,
        )

    interface_name = os.path.splitext(os.path.basename(args.bmap))[0]
    if interface_name not in cfg_dict:
        raise ValueError(
            f"Bump map interface '{interface_name}' was not found in the resolved config."
        )

    cfg = cfg_dict[interface_name]
    cfg.plot_flag = False
    cfg.pad_yield_flag = True

    output_dir = os.path.join(cfg.OUTPUT_DIR, ds_name, cfg.INTERFACE)
    os.makedirs(output_dir, exist_ok=True)

    pad_bitmap_collection = convert_3dblox_to_pad_bitmap(
        cfg=cfg,
        _bmap_path=args.bmap,
        criticality_path=args.criticality,
        pad_arrange_pattern=cfg.PAD_ARRANGE_PATTERN,
        input_args={
            "ds_name": ds_name,
            "config": args.config,
            "criticality_profile": "custom",
        },
    )

    die_stack = DieStack(
        cfg_dict={cfg.INTERFACE: cfg},
        pad_bitmap_collection_dict={cfg.INTERFACE: pad_bitmap_collection},
        mode=args.mode,
        base_pad_coords_flag=True,
    )
    interface = die_stack.interfaces.interface_dict[cfg.INTERFACE]

    defect_pad_yield_map = pad_defect_yield_map_generator(
        cfg=cfg,
        D0=cfg.D0,
        t_0=cfg.t_0,
        z=cfg.z,
        k_r=cfg.k_r,
        k_r0=cfg.k_r0,
        PAD_TOP_R_um=cfg.PAD_TOP_R_um,
        PAD_ARR_ROW=cfg.PAD_ARR_ROW,
        PAD_ARR_COL=cfg.PAD_ARR_COL,
        interface=interface,
        pad_yield_flag=True,
        pad_yield_map_sub_factor=cfg.pad_yield_map_sub_factor,
    )
    defect_pad_yield_map = _upsample_pad_yield_map(
        defect_pad_yield_map,
        (cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL),
        cfg.pad_yield_map_sub_factor,
    )
    particle_failure_risk_map = 1.0 - defect_pad_yield_map

    density_x, density_y, density_map = get_particle_density_map(
        D0=cfg.D0,
        D1=float(cfg.get("D1", cfg.D0)),
        DIE_W_um=cfg.DIE_W_um,
        DIE_L_um=cfg.DIE_L_um,
        edge_region_width_um=float(cfg.get("EDGE_REGION_WIDTH_um", 300.0)),
        grid_size=args.grid_size,
    )
    pad_density = particle_density_at_pad_coords(cfg=cfg, die=interface, D0=cfg.D0)

    pad_risk_path = os.path.join(output_dir, "particle_pad_risk_heatmap.png")
    comparison_path = os.path.join(output_dir, "particle_density_vs_pad_risk.png")

    draw_particle_pad_risk_heatmap(
        cfg=cfg,
        die=interface,
        particle_failure_risk_map=particle_failure_risk_map,
        output_path=pad_risk_path,
    )
    corr = draw_density_vs_risk_comparison(
        cfg=cfg,
        die=interface,
        density_x=density_x,
        density_y=density_y,
        density_map=density_map,
        particle_failure_risk_map=particle_failure_risk_map,
        pad_density=pad_density,
        output_path=comparison_path,
    )

    print(f"Saved particle pad risk heatmap to {pad_risk_path}")
    print(f"Saved density-vs-risk comparison to {comparison_path}")
    print(f"Pad-density correlation: {corr:.6f}")


if __name__ == "__main__":
    main()
