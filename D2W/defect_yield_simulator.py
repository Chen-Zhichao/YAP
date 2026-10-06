#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Defect yield simulator for the yield model for D2W hybrid bonding
#### Author: Zhichao Chen
#### Date: Oct 4, 2024

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import fsolve
import sympy as sp
from scipy.integrate import quad
from scipy.stats import norm

class particle:
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.thickness = t


class single_void:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r


class void_tail:
    def __init__(self, cfg, x, y, thickness, k_n, k_S, k_L, VOID_SHAPE, DIE_W_um, DIE_L_um):
        self.x = x
        self.y = y
        if cfg.first_contact == 'center':
            self.dist_from_contact = np.sqrt(x**2 + y**2)
        elif cfg.first_contact == 'vertical-edge':
            self.dist_from_contact = np.abs(cfg.DIE_W_um / 2 + x)
        elif cfg.first_contact == 'horizontal-edge':
            self.dist_from_contact = np.abs(cfg.DIE_L_um / 2 + y)
        elif cfg.first_contact == 'corner':
            self.dist_from_contact = np.sqrt((cfg.DIE_W_um / 2 + x)**2 + (cfg.DIE_L_um / 2 + y)**2)
        self.dist_from_contact = np.sqrt(x**2 + y**2)
        self.L = k_L * self.dist_from_contact * np.sqrt(thickness)  # void tail length
        self.n = np.round(
            k_n * self.dist_from_contact * np.sqrt(thickness)
        )  # number of voids in the tail
        self.S = k_S * self.dist_from_contact * np.sqrt(thickness)  # void tail area
        # self.S = k_S * self.dist_from_contact * thickness  # void tail area
        self.voids = []
        if self.n > 0:
            x_incrt = self.L * x / self.dist_from_contact / self.n
            y_incrt = self.L * y / self.dist_from_contact / self.n
            if VOID_SHAPE == "circle":  # r_vt is the radius of the circular void
                r_vt1 = np.sqrt(
                    self.S / ((np.pi * (self.n + 2) * (self.n + 1) * self.n) / 6)
                )
            elif (VOID_SHAPE == "square"):  # r_vt is the half side length of the square void
                r_vt1 = np.sqrt(
                    self.S / ((4 * (self.n + 2) * (self.n + 1) * self.n) / 6)
                )
            # r_vt1 = np.sqrt(self.S / self.n)      # assume the voids are in square shape
            for i in range(int(self.n)):
                x_vt = x + x_incrt * (int(self.n) - i)
                y_vt = y + y_incrt * (int(self.n) - i)
                if np.abs(x_vt) < DIE_W_um / 2 and np.abs(y_vt) < DIE_L_um / 2:
                    self.voids.append(single_void(x_vt, y_vt, r_vt1 * (i + 1)))



def get_particle_density_map(
    *,
    D0,
    D1,
    DIE_W_um,
    DIE_L_um,
    edge_region_width_um=300.0,
    grid_size=300,
):
    x_coords = np.linspace(-DIE_W_um / 2.0, DIE_W_um / 2.0, grid_size)
    y_coords = np.linspace(-DIE_L_um / 2.0, DIE_L_um / 2.0, grid_size)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing="xy")

    density_map = np.full_like(xx, float(D0), dtype=np.float64)
    if D1 <= D0 or edge_region_width_um <= 0:
        return x_coords, y_coords, density_map

    edge_region_width_um = min(
        float(edge_region_width_um),
        DIE_W_um / 2.0,
        DIE_L_um / 2.0,
    )
    if edge_region_width_um <= 0:
        return x_coords, y_coords, density_map

    dist_to_nearest_edge = np.minimum(
        DIE_W_um / 2.0 - np.abs(xx),
        DIE_L_um / 2.0 - np.abs(yy),
    )
    edge_weight = np.clip(
        1.0 - dist_to_nearest_edge / edge_region_width_um,
        0.0,
        1.0,
    )
    density_map += (float(D1) - float(D0)) * edge_weight
    return x_coords, y_coords, density_map


def draw_particle_density_heatmap(
    *,
    cfg,
    D0,
    DIE_W_um,
    DIE_L_um,
    output_path=None,
    grid_size=300,
):
    D1 = float(cfg.get("D1", D0))
    edge_region_width_um = float(cfg.get("EDGE_REGION_WIDTH_um", 300.0))
    x_coords, y_coords, density_map = get_particle_density_map(
        D0=D0,
        D1=D1,
        DIE_W_um=DIE_W_um,
        DIE_L_um=DIE_L_um,
        edge_region_width_um=edge_region_width_um,
        grid_size=grid_size,
    )

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    image = ax.imshow(
        density_map,
        origin="lower",
        extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
        cmap="hot",
        aspect="auto",
    )
    fig.colorbar(image, ax=ax, label="Particle Density (1/um^2)")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(
        "Particle Density Distribution\n"
        f"D0={float(D0):.3e}, D1={D1:.3e}, w={edge_region_width_um:.1f} um"
    )
    ax.set_aspect("equal")

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return x_coords, y_coords, density_map


def cdf_particle_thickness(t, t_0, z):
        return 1 - (t_0 / t) ** (z - 1)

def inverse_cdf_particle_thickness(u, t_0, z):
    return t_0 / (1 - u) ** (1 / (z - 1))

def generate_particles_across_die(particle_thickness, DIE_W_um, DIE_L_um, drop_particle_range):
    particles = []
    for i in range(len(particle_thickness)):
        # x, y = np.random.uniform(-DIE_W_um / 2, DIE_W_um / 2), np.random.uniform(-DIE_L_um / 2, DIE_L_um / 2)
        x, y = np.random.uniform(-DIE_W_um / 2 * drop_particle_range, DIE_W_um / 2 * drop_particle_range), np.random.uniform(-DIE_L_um / 2 * drop_particle_range, DIE_L_um / 2 * drop_particle_range)
        particles.append(particle(x, y, particle_thickness[i]))
    return particles

def generate_particles_at_die_edges(DIE_W_um, DIE_L_um, cfg):
    D0 = float(cfg.D0)
    D1 = float(cfg.D1)
    edge_region_width_um = float(cfg.get("EDGE_REGION_WIDTH_um", 300.0))

    assert D1 >= D0, "D1 should be greater than or equal to D0 to have edge effect"

    edge_region_width_um = min(
        edge_region_width_um,
        DIE_W_um / 2.0,
        DIE_L_um / 2.0,
    )
    assert edge_region_width_um > 0, "EDGE_REGION_WIDTH_um should be positive"

    # The uniform background sampler already contributes D0 everywhere.
    # Here we only sample the excess edge density:
    # delta_D(x, y) = (D1 - D0) * (1 - d(x, y) / w), for d(x, y) < w.
    delta_density_peak = D1 - D0
    num_candidate_particles = np.random.poisson(
        delta_density_peak * DIE_W_um * DIE_L_um
    )
    if num_candidate_particles == 0:
        return []

    x_coords = np.random.uniform(
        -DIE_W_um / 2.0, DIE_W_um / 2.0, num_candidate_particles
    )
    y_coords = np.random.uniform(
        -DIE_L_um / 2.0, DIE_L_um / 2.0, num_candidate_particles
    )

    dist_to_nearest_edge = np.minimum(
        DIE_W_um / 2.0 - np.abs(x_coords),
        DIE_L_um / 2.0 - np.abs(y_coords),
    )
    keep_prob = np.clip(
        1.0 - dist_to_nearest_edge / edge_region_width_um,
        0.0,
        1.0,
    )
    keep_mask = np.random.rand(num_candidate_particles) < keep_prob
    if not np.any(keep_mask):
        return []

    edge_particle_thickness = inverse_cdf_particle_thickness(
        np.random.rand(np.count_nonzero(keep_mask)), cfg.t_0, cfg.z
    )
    return [
        particle(x, y, thickness)
        for x, y, thickness in zip(
            x_coords[keep_mask],
            y_coords[keep_mask],
            edge_particle_thickness,
        )
    ]


# Generate the main void and void tail based on the particles
def generate_voids(cfg, particles, k_r, k_r0, k_n, k_S, k_L, VOID_SHAPE, DIE_W_um, DIE_L_um):
    voids = []
    main_voids = []
    tail_voids = []
    num_main_void = 0
    num_void_in_tail = 0
    for p in particles:
        if cfg.first_contact == 'center':
            distance_to_contact = np.sqrt(p.x**2 + p.y**2)
        elif cfg.first_contact == 'vertical-edge':
            distance_to_contact = np.abs(cfg.DIE_W_um / 2 + p.x)
        elif cfg.first_contact == 'horizontal-edge':
            distance_to_contact = np.abs(cfg.DIE_L_um / 2 + p.y)
        elif cfg.first_contact == 'corner':
            distance_to_contact = np.sqrt((cfg.DIE_W_um / 2 + p.x)**2 + (cfg.DIE_L_um / 2 + p.y)**2)
        # generate main void
        r_mv = (k_r * distance_to_contact + k_r0) * np.sqrt(p.thickness)
        voids.append(single_void(p.x, p.y, r_mv))
        main_voids.append(single_void(p.x, p.y, r_mv))
        num_main_void += 1
        # generate void tail
        void_tail_obj = void_tail(cfg, p.x, p.y, p.thickness, k_n, k_S, k_L, VOID_SHAPE, DIE_W_um, DIE_L_um)
        voids += void_tail_obj.voids
        tail_voids += void_tail_obj.voids
        num_void_in_tail += void_tail_obj.n

    return voids, main_voids, tail_voids

def defect_yield_simulator(
    cfg_dict,
    die_stack_list: list,
):
    """
    Function: "Allocate" particles defects to each die interface in each stack.
                Then generate the voids based on the particles.
    """
    NUM_STACKS = len(die_stack_list)

    # "Allocate" particles to each die interface in each satck
    for interface_name, cfg in cfg_dict.items():
        # Extract the input parameters from the current cfg
        DIE_W_um, DIE_L_um = cfg.DIE_W_um, cfg.DIE_L_um
        D0 = float(cfg.D0)
        t_0 = cfg.t_0
        z = cfg.z
        k_r = cfg.k_r
        k_r0 = cfg.k_r0
        k_n = cfg.k_n
        k_L = cfg.k_L
        k_S = cfg.k_S
        VOID_SHAPE = cfg.VOID_SHAPE

        # num_particles calculation
        drop_particle_range = 1 # the range of the particles to drop regarding the die size
        total_particles = np.random.poisson(
            (drop_particle_range * DIE_W_um)
            * (drop_particle_range * DIE_L_um)
            * D0
            * NUM_STACKS
        )
        if total_particles < 0:
            total_particles = 0
        particles_per_interface = np.random.multinomial(
            total_particles, [1 / NUM_STACKS] * NUM_STACKS
        )
        for stack_ind in range(NUM_STACKS):
            num_particles = particles_per_interface[stack_ind]
            particle_thickness = np.zeros(num_particles)
            u = np.random.rand(num_particles)
            particle_thickness = inverse_cdf_particle_thickness(u, t_0, z)
            particles_across_die = generate_particles_across_die(particle_thickness, DIE_W_um, DIE_L_um, drop_particle_range)
            particles_at_die_edges = generate_particles_at_die_edges(DIE_W_um, DIE_L_um, cfg)
            particles = particles_across_die + particles_at_die_edges

            # Generate the main void and void tail based on the particles for each die
            voids, main_voids, tail_voids = generate_voids(cfg, particles, k_r, k_r0, k_n, k_S, k_L, VOID_SHAPE, DIE_W_um, DIE_L_um)
            # transform the voids struct to array
            voids_arr = np.array([[v.x, v.y, v.r] for v in voids], dtype=float)
            # die_stack_list[stack_ind].interfaces.interface_dict[interface_name].voids = voids_arr
            die_stack_list[stack_ind].interfaces.failure_params_dict[interface_name]['voids'] = np.array(voids_arr)
