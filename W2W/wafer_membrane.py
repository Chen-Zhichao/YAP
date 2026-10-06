"""reference free-rim disk membrane metric, separated from paper-only yield imports.

The Ritz algebra is reused from exp12_w2w_window_overlay_20260914.py.
It retains actual extensional stiffness and engineering twist; it is not a
square-die approximation or a claim of new physical validation.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _finite(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.shape != shape or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be finite numeric data of shape {shape}")
    return np.asarray(raw, dtype=float)


def _membrane_matrix(matrix):
    a = _finite(matrix, (3, 3), "extensional_matrix")
    scale = float(np.max(np.abs(a)))
    if scale == 0:
        raise ValueError("extensional_matrix must be SPD")
    a = a / scale
    if np.linalg.norm(a - a.T) > 1e-10 * np.linalg.norm(a):
        raise ValueError("extensional_matrix must be symmetric")
    a = .5 * (a + a.T)
    if np.linalg.eigvalsh(a).min() <= 0:
        raise ValueError("extensional_matrix must be SPD")
    return a


def _moment(p, q):
    if p < 0 or q < 0 or p % 2 or q % 2:
        return 0.
    return math.gamma((p + 1) / 2) * math.gamma((q + 1) / 2) / math.gamma((p + q) / 2 + 2)


@lru_cache(maxsize=2)
def _space(degree):
    if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer)) or degree not in (3, 5):
        raise ValueError("disk Ritz degree must be 3 or 5")
    powers = tuple((p, total - p) for total in range(1, degree + 1, 2) for p in range(total + 1))
    n = len(powers)
    xx, yy, xy = (np.zeros((n, n)) for _ in range(3))
    dx, dy = np.zeros((n, 3)), np.zeros((n, 3))
    fit = np.empty((n, 2))
    for i, (p, q) in enumerate(powers):
        fit[i] = (4 / np.pi * _moment(p + 1, q), 4 / np.pi * _moment(p, q + 1))
        for j, (r, s) in enumerate(powers):
            xx[i, j] = p * r * _moment(p + r - 2, q + s)
            yy[i, j] = q * s * _moment(p + r, q + s - 2)
            xy[i, j] = p * s * _moment(p + r - 1, q + s - 1)
        for j, (r, s) in enumerate(((2, 0), (1, 1), (0, 2))):
            dx[i, j] = p * _moment(p + r - 1, q + s)
            dy[i, j] = q * _moment(p + r, q + s - 1)
    # Odd polynomials omit translations; symmetric shear removes rigid rotation.
    ux_y, uy_x = powers.index((0, 1)), n + powers.index((1, 0))
    keep = [i for i in range(2 * n) if i not in (ux_y, uy_x)]
    projection = np.zeros((2 * n, 2 * n - 1))
    projection[keep, np.arange(len(keep))] = 1.
    projection[[ux_y, uy_x], -1] = 1 / np.sqrt(2.)
    return powers, xx, yy, xy, dx, dy, fit, projection


@dataclass(frozen=True)
class DiskMembraneSolution:
    radius_mm: float
    degree: int
    powers: tuple
    coefficients: np.ndarray
    affine_metric: np.ndarray
    equilibrium_relative_residual: float


def solve_disk_membrane(extensional_matrix, curvature_per_mm, radius_mm=150., *, degree=5):
    """Free-rim eigenstrain 0.5*grad(w)*grad(w); engineering twist is twice Kxy."""
    a = _membrane_matrix(extensional_matrix)
    curvature = _finite(curvature_per_mm, (3,), "curvature_per_mm")
    radius = float(_finite(radius_mm, (), "radius_mm"))
    if radius <= 0:
        raise ValueError("radius_mm must be positive")
    powers, xx, yy, xy, dx, dy, fit, projection = _space(degree)
    cross = xy + xy.T
    kxx = a[0, 0]*xx + a[0, 2]*cross + a[2, 2]*yy
    kyy = a[1, 1]*yy + a[1, 2]*cross + a[2, 2]*xx
    kxy = a[0, 1]*xy + a[0, 2]*xx + a[1, 2]*yy + a[2, 2]*xy.T
    stiffness = projection.T @ np.block([[kxx, kxy], [kxy.T, kyy]]) @ projection
    try:
        with np.errstate(over="raise", invalid="raise"):
            x, y, c = curvature * radius * np.array([1., 1., .5])
            eigenstrain = np.array([[.5*x*x, x*c, .5*c*c],
                                   [.5*c*c, y*c, .5*y*y], [x*c, x*y+c*c, y*c]])
            stress = a @ eigenstrain
            force = projection.T @ np.r_[dx @ stress[0] + dy @ stress[2],
                                          dy @ stress[1] + dx @ stress[2]]
    except FloatingPointError as exc:
        raise ValueError("disk curvature exceeds numerical range") from exc
    try:
        solution = cho_solve(cho_factor(stiffness, lower=True), force)
    except np.linalg.LinAlgError as exc:
        raise ValueError("disk Ritz stiffness is numerically noncoercive") from exc
    residual = float(np.linalg.norm(stiffness @ solution - force) /
                     max(np.linalg.norm(force), np.finfo(float).tiny))
    if not np.isfinite(residual) or residual > 1e-9:
        raise ValueError("disk Ritz equilibrium residual exceeds tolerance")
    coefficients = (projection @ solution).reshape(2, len(powers))
    metric = coefficients @ fit
    rotation = .5 * (metric[1, 0] - metric[0, 1])
    coefficients[0, powers.index((0, 1))] += rotation
    coefficients[1, powers.index((1, 0))] -= rotation
    metric = .5 * (metric + metric.T)
    if not np.isfinite(coefficients).all() or not np.isfinite(metric).all():
        raise ValueError("nonfinite disk membrane solution")
    coefficients.setflags(write=False)
    metric.setflags(write=False)
    return DiskMembraneSolution(radius, int(degree), powers, coefficients, metric, residual)


def disk_membrane_metric(extensional_matrix, curvature_per_mm, radius_mm=150., *,
                         max_refinement_overlay_um=.5e-6):
    budget = float(_finite(max_refinement_overlay_um, (), "max_refinement_overlay_um"))
    if budget <= 0:
        raise ValueError("refinement budget must be positive")
    low = solve_disk_membrane(extensional_matrix, curvature_per_mm, radius_mm, degree=3)
    high = solve_disk_membrane(extensional_matrix, curvature_per_mm, radius_mm, degree=5)
    difference = high.coefficients.copy()
    difference[:, :len(low.powers)] -= low.coefficients
    field_change = float(np.linalg.norm(np.abs(difference).sum(axis=1)) * radius_mm * 1000.)
    affine_change = float(np.linalg.norm(high.affine_metric-low.affine_metric, 2) * radius_mm * 1000.)
    if not np.isfinite(field_change) or max(field_change, affine_change) > budget:
        raise ValueError(f"disk degree-3/5 refinement budget exceeded: {field_change:g} um")
    remainder = high.coefficients.copy()
    remainder[:, high.powers.index((1, 0))] -= high.affine_metric[:, 0]
    remainder[:, high.powers.index((0, 1))] -= high.affine_metric[:, 1]
    return high.affine_metric.copy(), dict(
        method="actual_A_free_rim_disk_polynomial_Ritz", degrees=[3, 5],
        fit="full_disk_area_L2_affine_projection",
        actual_A_and_twist_retained=True, field_refinement_bound_um=field_change,
        affine_refinement_edge_um=affine_change, refinement_budget_um=budget,
        equilibrium_relative_residuals=[low.equilibrium_relative_residual, high.equilibrium_relative_residual],
        nonaffine_field_coefficient_bound_um=float(np.linalg.norm(np.abs(remainder).sum(axis=1)) * radius_mm * 1000.),
        continuum_error_bound_claimed=False, physical_validation=False)

