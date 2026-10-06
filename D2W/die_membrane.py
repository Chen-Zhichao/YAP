"""Full-square Q4 shallow-shell membrane metric.

The free-boundary functional is the existing elastic membrane solve with
eigenstrain 0.5*grad(w)*grad(w). Full-domain assembly retains normal/shear
coupling and engineering twist instead of imposing quarter-square symmetry.
This numerical extension is not independent MAPDL/process validation.
"""

from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _finite(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.shape != shape or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be finite numeric data of shape {shape}")
    return raw.astype(float)


def _matrix_key(value):
    a = _finite(value, (3, 3), "extensional_matrix")
    scale = float(np.max(np.abs(a)))
    if scale == 0 or not np.allclose(a, a.T, rtol=1e-12, atol=scale * 1e-14):
        raise ValueError("extensional_matrix must be symmetric SPD")
    if np.linalg.eigvalsh(a / scale).min() <= 0:
        raise ValueError("extensional_matrix must be symmetric SPD")
    a = a / float(np.trace(a[:2, :2]))
    if not np.isfinite(a).all() or np.linalg.eigvalsh(a).min() <= 0:
        raise ValueError("extensional_matrix must be symmetric SPD")
    return tuple(np.round(a.ravel(), 14))


def _eigenstrain_basis(x, y):
    """Columns multiply [a²,b²,c²,ab,ac,bc], with c=engineering_twist*L/2."""
    x, y = np.broadcast_arrays(x, y)
    result = np.zeros(x.shape + (3, 6))
    result[..., 0, 0], result[..., 0, 2], result[..., 0, 4] = .5*x*x, .5*y*y, x*y
    result[..., 1, 1], result[..., 1, 2], result[..., 1, 5] = .5*y*y, .5*x*x, x*y
    result[..., 2, 2], result[..., 2, 3] = x*y, x*y
    result[..., 2, 4], result[..., 2, 5] = x*x, y*y
    return result


def _reference_affine_fit(displacement, coordinates, fit_fraction):
    """Average four quadrant fits, without mirroring the solved field."""
    side = len(coordinates)
    x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
    center = (side // 2) * side + side // 2
    centered = displacement - displacement[center]
    gradients = []
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        mask = ((np.abs(x.ravel()) <= fit_fraction) & (np.abs(y.ravel()) <= fit_fraction)
                & (sx*x.ravel() >= 0) & (sy*y.ravel() >= 0))
        fit = np.column_stack((x.ravel()[mask], y.ravel()[mask]))
        if np.linalg.matrix_rank(fit) != 2:
            raise ValueError("fit window is too small to identify an affine strain")
        fitted = np.linalg.lstsq(fit, centered[mask].reshape(mask.sum(), -1), rcond=None)[0]
        gradients.append(fitted.reshape(2, 2, -1).transpose(2, 1, 0))
    mean = np.mean(gradients, axis=0)
    return .5 * (mean + mean.transpose(0, 2, 1))


@lru_cache(maxsize=32)
def _coefficients(matrix_key, half_elements, fit_fraction, gauge_axis):
    a = np.asarray(matrix_key).reshape(3, 3)
    if np.linalg.eigvalsh(a).min() <= 0:
        raise ValueError("rounded extensional matrix is not SPD")
    n = 2 * half_elements
    side = n + 1
    h = 2. / n
    ix, iy = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
    first = (iy * side + ix).ravel()
    nodes = np.stack((first, first+1, first+side+1, first+side), axis=1)
    dofs = (2*nodes[..., None] + np.arange(2)).reshape(-1, 8)
    force = np.zeros((2 * side**2, 6))
    element_k = np.zeros((8, 8))
    element_f = np.zeros((n*n, 8, 6))
    for eta in (-1 / np.sqrt(3.), 1 / np.sqrt(3.)):
        for xi in (-1 / np.sqrt(3.), 1 / np.sqrt(3.)):
            dx = np.array([-(1-eta), 1-eta, 1+eta, -(1+eta)]) / (2*h)
            dy = np.array([-(1-xi), -(1+xi), 1+xi, 1-xi]) / (2*h)
            b = np.zeros((3, 8))
            b[0, 0::2], b[1, 1::2] = dx, dy
            b[2, 0::2], b[2, 1::2] = dy, dx
            x = -1 + h * (ix.ravel() + .5*(1+xi))
            y = -1 + h * (iy.ravel() + .5*(1+eta))
            element_k += b.T @ a @ b * h*h / 4
            element_f += np.einsum("ij,ejk->eik", b.T @ a, _eigenstrain_basis(x, y)) * h*h / 4
    np.add.at(force, dofs.ravel(), element_f.reshape(-1, 6))
    rows = np.broadcast_to(dofs[:, :, None], (n*n, 8, 8)).ravel()
    cols = np.broadcast_to(dofs[:, None, :], (n*n, 8, 8)).ravel()
    values = np.broadcast_to(element_k, (n*n, 8, 8)).ravel()
    stiffness = coo_matrix((values, (rows, cols)), shape=(len(force), len(force))).tocsc()
    center = half_elements * side + half_elements
    # These three values fix only Tx, Ty and rotation; no rim support is added.
    edge = half_elements * side + n if gauge_axis == 0 else n * side + half_elements
    fixed = np.array([2*center, 2*center+1, 2*edge + (1 if gauge_axis == 0 else 0)])
    free = np.setdiff1d(np.arange(len(force)), fixed)
    k = stiffness[free][:, free]
    displacement = np.zeros_like(force)
    displacement[free] = splu(k).solve(force[free])
    residual = np.linalg.norm(k @ displacement[free] - force[free], axis=0) / np.maximum(
        np.linalg.norm(force[free], axis=0), np.finfo(float).tiny)
    if not np.isfinite(displacement).all() or not np.isfinite(residual).all() or residual.max() > 1e-8:
        raise ValueError("full-square membrane equilibrium residual exceeds tolerance")
    coordinates = np.linspace(-1., 1., side)
    strain = _reference_affine_fit(displacement.reshape(side**2, 2, 6), coordinates, fit_fraction)
    strain.setflags(write=False)
    return strain, tuple(residual)


def square_membrane_metric(extensional_matrix, curvature_per_mm, halfspan_mm, *,
                           elements_per_half_side=48, fit_fraction=.98, gauge_axis=0):
    """Return a symmetric 2x2 strain tensor; lengths mm, curvature 1/mm.

    Engineering twist is twice the Hessian mixed derivative, as in the native
    reference disk contract. Translation/rotation gauges do not enter this strain.
    Six RHS share one factorization for each complete normalized A and policy.
    """
    key = _matrix_key(extensional_matrix)
    curvature = _finite(curvature_per_mm, (3,), "curvature_per_mm")
    halfspan = float(_finite(halfspan_mm, (), "halfspan_mm"))
    if halfspan <= 0 or type(elements_per_half_side) is not int or not 4 <= elements_per_half_side <= 128:
        raise ValueError("positive halfspan and 4..128 elements per half-side required")
    fraction = float(_finite(fit_fraction, (), "fit_fraction"))
    if not 0 < fraction <= 1 or type(gauge_axis) is not int or gauge_axis not in (0, 1):
        raise ValueError("fit_fraction must be in (0,1] and gauge_axis 0 or 1")
    with np.errstate(over="raise", invalid="raise"):
        try:
            x, y, c = curvature * halfspan * np.array([1., 1., .5])
            weights = np.array([x*x, y*y, c*c, x*y, x*c, y*c])
        except FloatingPointError as exc:
            raise ValueError("square curvature exceeds numerical range") from exc
    basis, residuals = _coefficients(key, elements_per_half_side, fraction, gauge_axis)
    metric = np.einsum("i,ijk->jk", weights, basis)
    if not np.isfinite(metric).all():
        raise ValueError("nonfinite square membrane metric")
    return metric, dict(method="REFERENCE_Q4_functional_full_square_affine_extension",
        elements_per_half_side=elements_per_half_side, full_square_elements_per_side=2*elements_per_half_side,
        fit_fraction=fraction, fit="mean_four_quadrant_REFERENCE_window_affine_fit",
        quadrant_fit="no_intercept_equal_weight_signed_global_coordinates",
        actual_A_and_twist_retained=True, rigid_gauge_axis=gauge_axis,
        equilibrium_relative_residuals=list(residuals), curvature_units="1/mm", halfspan_mm=halfspan,
        metric_units="strain", constitutive_history="frozen_during_elastic_flattening",
        continuum_error_bound_claimed=False, physical_validation=False)
