#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Oct 2, 2025

'''
This module contains functions to calculate die-level and pad-level defect-induced yield based on void size distribution and pad layout.
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import math

def get_bitmap_bounds(*,
                      bitmap: np.ndarray,
                      pad_block_size: int
    ):
    # Find the bounds of the non-zero pixels in the bitmap
    rows = np.any(bitmap, axis=1)
    cols = np.any(bitmap, axis=0)

    if not np.any(rows) or not np.any(cols):
        return 0, 0

    top, bottom = np.where(rows)[0][[0, -1]] * pad_block_size
    left, right = np.where(cols)[0][[0, -1]]

    height = bottom - top + 1
    width = right - left + 1

    return width, height





def pad_defect_yield_map_generator(
    cfg,
    D0: float,
    t_0: float,
    z: float,
    k_r: float,
    k_r0: float,
    PAD_TOP_R_um: float,
    PAD_ARR_ROW: int,
    PAD_ARR_COL: int,
    interface,
    pad_yield_flag: bool = False,
    pad_yield_map_sub_factor: int = 1,
):
    def pad_dist_from_first_contact(*, cfg, interface) -> np.ndarray:
        if cfg.first_contact == 'center':
            return np.sqrt(interface.pad_coords[:, 0]**2 + interface.pad_coords[:, 1]**2)  # pad to die center distance
        elif cfg.first_contact == 'vertical-edge':
            return np.abs(interface.DIE_W_um / 2 + interface.pad_coords[:, 0])  # pad to left die edge distance
        elif cfg.first_contact == 'horizontal-edge':
            return np.abs(interface.DIE_L_um / 2 + interface.pad_coords[:, 1])  # pad to bottom die edge distance
        elif cfg.first_contact == 'corner':
            return np.sqrt((interface.DIE_W_um / 2 + interface.pad_coords[:, 0])**2 + (interface.DIE_L_um / 2 + interface.pad_coords[:, 1])**2)  # pad to left-bottom die corner distance
        raise ValueError(f"Unsupported first_contact mode: {cfg.first_contact}")

    def particle_density_at_pad_coords(*, interface, D0) -> np.ndarray:
        D1 = float(cfg.get("D1", D0))
        edge_region_width_um = float(cfg.get("EDGE_REGION_WIDTH_um", 300.0))

        local_density = np.full(interface.pad_coords.shape[0], float(D0), dtype=np.float64)
        if D1 <= D0 or edge_region_width_um <= 0:
            return local_density

        effective_edge_width_um = min(
            float(edge_region_width_um),
            interface.DIE_W_um / 2.0,
            interface.DIE_L_um / 2.0,
        )
        if effective_edge_width_um <= 0:
            return local_density

        dist_to_nearest_edge = np.minimum(
            interface.DIE_W_um / 2.0 - np.abs(interface.pad_coords[:, 0]),
            interface.DIE_L_um / 2.0 - np.abs(interface.pad_coords[:, 1]),
        )
        edge_weight = np.clip(
            1.0 - dist_to_nearest_edge / effective_edge_width_um,
            0.0,
            1.0,
        )
        return local_density + (float(D1) - float(D0)) * edge_weight

    def avg_defects_fail_pad_critical(*, cfg, interface, D0, PAD_TOP_R_um, k_r, k_r0, t_0, z) -> np.ndarray:
        '''
        This function calculate the average number of fatal main void defects to the pad
        To calculate the pad-level defect yield, we ignore whether the pad is redundant or not.
        '''
        L0 = pad_dist_from_first_contact(cfg=cfg, interface=interface)

        # Use the formula to calculate the average number of fatal defects per pad
        term = k_r * L0 + k_r0
        part1 = PAD_TOP_R_um**2
        part2 = ((z - 1) / (z - 2)) * (term**2) * t_0
        part3 = (4 * (z - 1) / (2 * z - 3)) * term * PAD_TOP_R_um * t_0
        return np.pi * D0 * (part1 + part2 + part3)

    def avg_defects_fail_pad_critical_with_edge_defects(*, cfg, interface, D0, PAD_TOP_R_um, k_r, k_r0, t_0, z) -> np.ndarray:
        '''
        This function calculate the average number of fatal main void defects to the pad
        with edge-enhanced particle density.

        We preserve the existing closed-form critical-area model and apply a local-density
        approximation, i.e. the effective particle density is evaluated at each pad center
        using the nearest-edge D(x, y) profile.
        '''
        L0 = pad_dist_from_first_contact(cfg=cfg, interface=interface)
        local_particle_density = particle_density_at_pad_coords(interface=interface, D0=D0)

        term = k_r * L0 + k_r0
        part1 = PAD_TOP_R_um**2
        part2 = ((z - 1) / (z - 2)) * (term**2) * t_0
        part3 = (4 * (z - 1) / (2 * z - 3)) * term * PAD_TOP_R_um * t_0
        critical_area = np.pi * (part1 + part2 + part3)
        return local_particle_density * critical_area

    if pad_yield_flag:
        D1 = float(cfg.get("D1", D0))
        edge_region_width_um = float(cfg.get("EDGE_REGION_WIDTH_um", 300.0))
        if D1 > D0 and edge_region_width_um > 0:
            avg_main_voids_per_pad = avg_defects_fail_pad_critical_with_edge_defects(
                cfg=cfg,
                interface=interface,
                D0=D0,
                PAD_TOP_R_um=PAD_TOP_R_um,
                k_r=k_r,
                k_r0=k_r0,
                t_0=t_0,
                z=z,
            )
        else:
            avg_main_voids_per_pad = avg_defects_fail_pad_critical(
                cfg=cfg,
                interface=interface,
                D0=D0,
                PAD_TOP_R_um=PAD_TOP_R_um,
                k_r=k_r,
                k_r0=k_r0,
                t_0=t_0,
                z=z,
            )
    else:
        avg_main_voids_per_pad = None

    if pad_yield_flag == True:
        glb_defect_pad_yield_min = 1.0
        glb_defect_pad_yield_max = 0.0
        particle_defect_pad_yield_map = np.exp(-avg_main_voids_per_pad)
        glb_defect_pad_yield_min = min(glb_defect_pad_yield_min, np.nanmin(particle_defect_pad_yield_map))
        glb_defect_pad_yield_max = max(glb_defect_pad_yield_max, np.nanmax(particle_defect_pad_yield_map))
        interface.glb_pad_yield_min_max_dict['Y_df'] = (glb_defect_pad_yield_min, glb_defect_pad_yield_max)
        # Subsampling the pad yield map to save memory and speed up the plotting
        nr = math.ceil(PAD_ARR_ROW / pad_yield_map_sub_factor)
        nc = math.ceil(PAD_ARR_COL / pad_yield_map_sub_factor)
        r_idx = np.round(np.linspace(0, PAD_ARR_ROW - 1, nr)).astype(int)
        c_idx = np.round(np.linspace(0, PAD_ARR_COL - 1, nc)).astype(int)
        RR, CC = np.meshgrid(r_idx, c_idx, indexing='ij')   # shape (nr, nc)
        I = RR * PAD_ARR_COL + CC  # linear indices. shape (nr, nc)
        particle_defect_pad_yield_map_sub = particle_defect_pad_yield_map[I]
    else:
        particle_defect_pad_yield_map = None
        particle_defect_pad_yield_map_sub = None

    if cfg.plot_flag and pad_yield_flag:
        # Draw heatmap of pad-level defect yield map
        plt.figure(figsize=(8, 6))
        plt.imshow(
            particle_defect_pad_yield_map_sub,
            cmap='viridis',
            vmin=interface.glb_pad_yield_min_max_dict['Y_df'][0],
            vmax=interface.glb_pad_yield_min_max_dict['Y_df'][1],
            interpolation='nearest'
            )
        plt.colorbar(label='Pad-level Defect Yield (Subsampled)')
        plt.title('Pad-level Defect Yield Map')
        plt.xlabel('Pad Column Index')
        plt.ylabel('Pad Row Index')
        plt.show()

    return particle_defect_pad_yield_map_sub
