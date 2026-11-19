"""Independent relative x/y pad noise and layout-aware survival bounds.

Exact joint pad survival requires explicit pad coordinates. A compact
bounding rectangle supplies probability bounds without inventing a pad layout.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import ncx2



def _finite(value, name):
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _pair(value, name):
    result = _finite(value, name)
    if result.ndim < 1 or result.shape[-1] != 2:
        raise ValueError(f"{name} must end in two coordinates")
    return result


def _affine(matrix, translation):
    matrix = _finite(matrix, "matrix")
    translation = _pair(translation, "translation")
    if matrix.ndim < 2 or matrix.shape[-2:] != (2, 2):
        raise ValueError("matrix must end in (2,2)")
    shape = np.broadcast_shapes(matrix.shape[:-2], translation.shape[:-1])
    return (np.broadcast_to(matrix, shape + (2, 2)),
            np.broadcast_to(translation, shape + (2,)))


def pad_log_survival(distance_um, threshold_um, sigma_um):
    """log P(||mu + eps|| <= a), eps~N_2(0,sigma^2 I).

    Sigma is per relative-overlay axis, not a radial standard deviation or
    a separate per-wafer value. There is no extra sqrt(2) conversion.
    SciPy/Boost evaluates the noncentral chi-square law (two degrees of
    freedom); survival-function arithmetic retains very small failure tails.
    Survival tails below floating-point range can underflow to zero; these
    routines do not provide arbitrary-precision probability certificates.
    """
    d, a, sigma = np.broadcast_arrays(
        _finite(distance_um, "distance"), _finite(threshold_um, "threshold"),
        _finite(sigma_um, "sigma"))
    if np.any(d < 0) or np.any(a < 0) or np.any(sigma < 0):
        raise ValueError("distance, threshold and sigma must be nonnegative")
    result = np.where(d <= a, 0., -np.inf)
    active = sigma > 0
    if active.any():
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            x = np.square(a[active] / sigma[active])
            nc = np.square(d[active] / sigma[active])
        tail = ncx2.sf(x, df=2, nc=nc)
        if not np.isfinite(tail).all() or np.any((tail < 0) | (tail > 1)):
            raise ArithmeticError("invalid noncentral chi-square survival probability")
        values = np.empty_like(tail)
        near_one = tail <= .5
        values[near_one] = np.log1p(-tail[near_one])
        values[~near_one] = ncx2.logcdf(x[~near_one], df=2, nc=nc[~near_one])
        if np.isnan(values).any() or np.any(values > 0):
            raise ArithmeticError("invalid pad log survival")
        result = np.array(result, copy=True)
        result[active] = values
    return result


def explicit_pad_survival(matrix, translation_um, critical_pad_coordinates_um,
                         *, threshold_um, sigma_um, pad_chunk=2048,
                         batch_chunk=64):
    """Exact conditional all-critical-pad probability for a specified layout.

    Integrates independent pad errors analytically, in bounded chunks. The
    caller selects critical coordinates; no power/dummy/redundancy criterion
    is inferred here. Independence is conditional on A,b, which already
    contain all shared fabrication and placement variables.
    """
    matrix, translation = _affine(matrix, translation_um)
    xy = _pair(critical_pad_coordinates_um, "critical_pad_coordinates_um")
    if xy.ndim != 2:
        raise ValueError("critical coordinates must be (pad,2)")
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_))
           or v <= 0 for v in (pad_chunk, batch_chunk)):
        raise ValueError("positive integer chunk sizes required")
    a = _finite(threshold_um, "threshold")
    sigma = _finite(sigma_um, "sigma")
    shape = np.broadcast_shapes(matrix.shape[:-2], a.shape, sigma.shape)
    matrix = np.broadcast_to(matrix, shape + (2, 2))
    translation = np.broadcast_to(translation, shape + (2,))
    a = np.broadcast_to(a, shape).ravel()
    sigma = np.broadcast_to(sigma, shape).ravel()
    if np.any(a < 0) or np.any(sigma < 0):
        raise ValueError("negative threshold or sigma")
    matrices = matrix.reshape(-1, 2, 2)
    translations = translation.reshape(-1, 2)
    logp = np.zeros(len(matrices))
    for start in range(0, len(matrices), batch_chunk):
        stop = min(start + batch_chunk, len(matrices))
        for first in range(0, len(xy), pad_chunk):
            values = np.einsum("bij,pj->bpi", matrices[start:stop], xy[first:first+pad_chunk])
            values += translations[start:stop, None, :]
            distance = np.hypot(values[..., 0], values[..., 1])
            logp[start:stop] += pad_log_survival(
                distance, a[start:stop, None], sigma[start:stop, None]).sum(axis=1)
    return np.exp(logp.reshape(shape))


@dataclass(frozen=True)
class EnvelopeSurvival:
    lower: np.ndarray
    upper: np.ndarray
    minimum_distance_lower_bound_um: np.ndarray
    maximum_envelope_distance_um: np.ndarray


def compact_envelope_survival_bounds(matrix, translation_um, *,
                                    rectangle_min_um, rectangle_max_um,
                                    critical_pad_count, threshold_um, sigma_um):
    """Layout-agnostic bounds when N critical pads lie in one rectangle.

    No tile populations or explicit coordinates are fabricated. For unknown
    pad membership, even sigma=0 may leave [0,1]; the rectangle corners need
    not be actual pads. These mathematical bounds use ordinary floating
    point probability routines, not certified interval arithmetic.
    """
    matrix, translation = _affine(matrix, translation_um)
    lo = _pair(rectangle_min_um, "rectangle_min_um")
    hi = _pair(rectangle_max_um, "rectangle_max_um")
    if lo.shape != (2,) or hi.shape != (2,) or np.any(lo > hi):
        raise ValueError("require ordered rectangular bounds of shape (2,)")
    raw_count = np.asarray(critical_pad_count)
    if raw_count.ndim != 0 or raw_count.dtype.kind not in "iuf":
        raise ValueError("critical count must be a nonnegative integer, not Boolean")
    value = raw_count.item()
    if not np.isfinite(value) or value < 0 or value > 2**53 or int(value) != value:
        raise ValueError("critical count must be an exactly representable nonnegative integer")
    n = int(value)
    a = _finite(threshold_um, "threshold")
    sigma = _finite(sigma_um, "sigma")
    if np.any(a < 0) or np.any(sigma < 0):
        raise ValueError("negative threshold or sigma")
    corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]],
                        [hi[0], lo[1]], [hi[0], hi[1]]])
    center = .5 * (lo + hi)
    offsets = np.einsum("...ij,pj->...pi", matrix, corners - center)
    center_value = np.einsum("...ij,j->...i", matrix, center) + translation
    corner_values = offsets + center_value[..., None, :]
    radius = np.linalg.norm(offsets, axis=-1).max(axis=-1)
    minimum_lower = np.maximum(0., np.linalg.norm(center_value, axis=-1) - radius)
    maximum = np.linalg.norm(corner_values, axis=-1).max(axis=-1)
    if n == 0:
        lower = upper = np.ones(np.broadcast_shapes(maximum.shape, a.shape, sigma.shape))
    else:
        low_log = pad_log_survival(maximum, a, sigma)
        high_log = pad_log_survival(minimum_lower, a, sigma)
        lower, upper = np.exp(n*low_log), np.exp(n*high_log)
    if np.any(lower > upper + 1e-14):
        raise ArithmeticError("inverted envelope probability bounds")
    return EnvelopeSurvival(lower, upper, minimum_lower, maximum)


def joint_prefix_bounds(lower, upper):
    """Multiply conditional interface bounds first, never marginal means."""
    lower, upper = np.broadcast_arrays(_finite(lower, "lower"), _finite(upper, "upper"))
    if lower.ndim < 1 or lower.shape[-1] == 0 or np.any(
            (lower < 0) | (upper > 1) | (lower > upper)):
        raise ValueError("ordered probability bounds and an interface axis required")
    return np.cumprod(lower, axis=-1), np.cumprod(upper, axis=-1)
