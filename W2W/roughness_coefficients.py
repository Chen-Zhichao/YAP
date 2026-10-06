#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Roughness parameters preparation for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Oct 20, 2025
'''
This module incorporates the roughness profile and calculate the effective contact area ratio:
- A_star_b: normalized effective contact area (0 to 1)
'''

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar


def theta_func(R, sigma, adhesion_w, Young_modulus_Pa_E):
    # Only works when you assume two wafers have the same surface roughness profile
    theta = Young_modulus_Pa_E / adhesion_w * np.sqrt(sigma ** 3 / R)  # dimensionless parameter
    return theta


def integrand_A(x, s_star):
    return (x - s_star) * np.exp(-x**2 / 2)

def A_star(s_star, constant):
    A_star_integral, _ = quad(integrand_A, s_star, np.inf, args=(s_star,))
    # A_value = np.pi * R * sigma * eta_s * A_star_integral / np.sqrt(2 * np.pi)
    A_value = np.pi * constant * A_star_integral / np.sqrt(2 * np.pi)
    return A_value


# 定义第二个积分的 integrand 函数
def integrand_P1(x, s_star):
    return (x - s_star)**(3/2) * np.exp(-x**2 / 2)

def integrand_P2(x):
    return np.exp(-x**2 / 2)

def P_star(s_star, constant, theta):
    P_star_integral1, _ = quad(integrand_P1, s_star, np.inf, args=(s_star,))
    P_star_integral2, _ = quad(integrand_P2, s_star, np.inf)
    # Calculate P* value
    # P_value = R * sigma * eta_s * ((4 * theta) / (3 * np.sqrt(2 * np.pi)) * P_star_integral1 - np.sqrt(2 * np.pi) * P_star_integral2)
    P_value = constant * ((4 * theta) / (3 * np.sqrt(2 * np.pi)) * P_star_integral1 - np.sqrt(2 * np.pi) * P_star_integral2)
    return P_value


def get_eff_contact_area_ratio(
    Asperity_R_m: float,
    Roughness_sigma_m: float,
    eta_s: float,
    Roughness_constant: float,
    Adhesion_energy: float,
    Dielectric_Young_modulus_Pa: float,
) -> float:
    Roughness_sigma_m_renorm = Roughness_sigma_m * np.sqrt(2)
    Young_modulus_Pa_renorm = Dielectric_Young_modulus_Pa * 0.5
    # Calculate theta
    theta = theta_func(R=Asperity_R_m,
                          sigma=Roughness_sigma_m_renorm,
                          adhesion_w=Adhesion_energy,
                          Young_modulus_Pa_E=Young_modulus_Pa_renorm)
    constant = Roughness_constant
    # Calculate s_star_b: assume s_star_b is between -10 and 10
    s_star_b = root_scalar(lambda s_star: P_star(s_star, constant=constant, theta=theta), bracket=[-10, 10], method='brentq')
    # A_star_b: normalized effective contact area
    A_star_b = A_star(s_star_b.root, constant=constant)
    # print("The normalized effective contact area A_star_b is: ", A_star_b)
    return min(A_star_b, 1.0)   # This normalized area should not exceed 1.0