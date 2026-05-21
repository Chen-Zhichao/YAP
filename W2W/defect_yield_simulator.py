#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Feb 4, 2026

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
        self.dist_from_center = np.sqrt(x**2 + y**2)
        self.thickness = t


class single_void:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r


class void_tail:
    def __init__(self, x, y, thickness, k_n, k_S, k_L, VOID_SHAPE, WAF_R_um):
        self.x = x      # x coordinate of the particle
        self.y = y      # y coordinate of the particle
        self.dist_from_center = np.sqrt(x**2 + y**2)        # distance from the center of the wafer
        self.L = k_L * self.dist_from_center * np.sqrt(thickness)  # void tail length
        # print("The average void tail length is: ", np.mean(self.L))
        # np.save("void_tail_length.npy", self.L)
        # raise ValueError
        self.n = np.round(
            k_n * self.dist_from_center * np.sqrt(thickness)
        )  # number of voids in the tail
        self.S = k_S * self.dist_from_center * np.sqrt(thickness)  # void tail area
        # self.S = k_S * self.dist_from_center * thickness  # void tail area
        self.voids = []
        '''
        # if self.n > 0:
        #     x_incrt = self.L * x / self.dist_from_center / self.n
        #     y_incrt = self.L * y / self.dist_from_center / self.n
        #     if VOID_SHAPE == "circle":  # r_vt is the radius of the circular void
        #         r_vt1 = np.sqrt(
        #             self.S / ((np.pi * (self.n + 2) * (self.n + 1) * self.n) / 6)
        #         )
        #     elif (VOID_SHAPE == "square"):  # r_vt is the half side length of the square void
        #         r_vt1 = np.sqrt(
        #             self.S / ((4 * (self.n + 2) * (self.n + 1) * self.n) / 6)
        #         )
        #     # r_vt1 = np.sqrt(self.S / self.n)      # assume the voids are in square shape
        #     # for i in range(int(self.n)):
        #     #     x_vt = x + x_incrt * (int(self.n) - i)
        #     #     y_vt = y + y_incrt * (int(self.n) - i)
        #     #     if np.sqrt(x_vt**2 + y_vt**2) < WAF_R_um:
        #     #         self.voids.append(single_void(x_vt, y_vt, r_vt1 * (i + 1)))
        '''

        void_tail_indices = np.where(self.n > 0)[0]
        for i in void_tail_indices:
            x_incrt = self.L[i] * x[i] / self.dist_from_center[i] / self.n[i]
            y_incrt = self.L[i] * y[i] / self.dist_from_center[i] / self.n[i]
            if VOID_SHAPE == "circle":
                r_vt1 = np.sqrt(
                    self.S[i] / ((np.pi * (self.n[i] + 2) * (self.n[i] + 1) * self.n[i]) / 6)
                )
            elif (VOID_SHAPE == "square"):
                r_vt1 = np.sqrt(
                    self.S[i] / ((4 * (self.n[i] + 2) * (self.n[i] + 1) * self.n[i]) / 6)
                )
            x_vt = x[i] + x_incrt * (np.arange(int(self.n[i]), 0, -1))
            y_vt = y[i] + y_incrt * (np.arange(int(self.n[i]), 0, -1))
            r_vt = r_vt1 * (np.arange(int(self.n[i])) + 1)
            dist_from_center_vt = np.sqrt(x_vt**2 + y_vt**2)
            mask = dist_from_center_vt < WAF_R_um
            x_vt = x_vt[mask]
            y_vt = y_vt[mask]
            r_vt = r_vt[mask]
            dist_from_center_vt = dist_from_center_vt[mask]
            self.voids.append(np.column_stack((x_vt, y_vt, r_vt)))
        if len(self.voids) == 0:
            self.voids = np.zeros((0, 3))
        else:
            self.voids = np.vstack(self.voids)

        

def cdf_particle_thickness(t, t_0, z):
    return 1 - (t_0 / t) ** (z - 1)


def inverse_cdf_particle_thickness(u, t_0, z):
    return t_0 / (1 - u) ** (1 / (z - 1))

def random_point_in_circle(radius):
    theta = np.random.uniform(0, 2 * np.pi)  # Generate a random angle
    r = radius * np.sqrt(np.random.uniform(0, 1))  # Generate a random radius
    x = r * np.cos(theta)  # Transform the polar coordinates to x
    y = r * np.sin(theta)  # Transform the polar coordinates to y
    return x, y


def generate_particles(particle_thickness, wafer_radius):
    angles = np.random.uniform(0, 2 * np.pi, len(particle_thickness))
    radii = wafer_radius * np.sqrt(np.random.uniform(0, 1, len(particle_thickness)))
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    dist_from_center = np.sqrt(x**2 + y**2)
    particles = np.column_stack((x, y, dist_from_center, particle_thickness))

    return particles

# Generate the main void and void tail based on the particles
def generate_voids(particles, k_r, k_r0, k_n, k_S, k_L, VOID_SHAPE, WAF_R_um):
    voids = []
    main_voids = []
    tail_voids = []
    if len(particles) == 0:
        return voids, main_voids, tail_voids
    r_mv = (k_r * particles[:, 2] + k_r0) * np.sqrt(particles[:, 3])
    
    voids = np.column_stack((particles[:, 0], particles[:, 1], r_mv))
    main_voids = np.column_stack((particles[:, 0], particles[:, 1], r_mv))
    num_main_void = len(particles)
    tail_voids_obj = void_tail(
        particles[:, 0], particles[:, 1], particles[:, 3], k_n, k_S, k_L, VOID_SHAPE, WAF_R_um
    )
    tail_voids = tail_voids_obj.voids
    num_tail_voids = len(tail_voids)
    voids = np.vstack((main_voids, tail_voids))

    return voids, main_voids, tail_voids




def defect_yield_simulator(
    cfg_dict,
    waf_stack_list: list,
):

    
    NUM_STACKS = len(waf_stack_list)

    # "Allocate" particles to each wafer interface in each satck
    for interface_name, cfg in cfg_dict.items():
        # Extract the input parameters from the current cfg
        WAF_R_um = cfg.WAF_R_um
        D0 = cfg.D0
        t_0 = cfg.t_0
        z = cfg.z
        k_r = cfg.k_r
        k_r0 = cfg.k_r0
        k_n = cfg.k_n
        k_L = cfg.k_L
        k_S = cfg.k_S
        VOID_SHAPE = cfg.VOID_SHAPE
        
        # num_particles calculation
        total_particles = np.random.poisson(
            np.pi * WAF_R_um**2 * NUM_STACKS * D0
        )
        particles_per_interface = np.random.multinomial(
            total_particles, [1 / NUM_STACKS] * NUM_STACKS
        )

        # For each stack, generate the particles and then the voids
        for stack_ind in range(NUM_STACKS):
            num_particles = particles_per_interface[stack_ind]
            particle_thickness = np.zeros(num_particles)
            u = np.random.rand(num_particles)
            particle_thickness = inverse_cdf_particle_thickness(u, t_0, z)
            particles = generate_particles(particle_thickness, WAF_R_um)

            # Generate the main void and void tail based on the particles for each wafer
            voids, main_voids, tail_voids = generate_voids(particles, k_r, k_r0, k_n, k_S, k_L, VOID_SHAPE, WAF_R_um)
            # Pass the void failure parameters to the wafer_stacks interface object
            waf_stack_list[stack_ind].interfaces.failure_params_dict[interface_name]['voids'] = np.array(voids)
