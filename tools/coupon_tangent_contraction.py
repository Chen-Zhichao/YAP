"""Bounded all-GP tangent contraction; no point sampling or constitutive change."""
from types import SimpleNamespace

import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def contract_coordinate_tangent(strain_maps, weights, reference_stiffness, prepared,
        *, correction_times_map, block_size=4096, check=lambda: None):
    """Apply the caller's original C_alg-C_el map routine in bounded blocks.

    Zero-correction points remain represented by the complete reference matrix.
    The zero test is exact, not a tolerance or cubature approximation. Positive
    weights, input maps, Cu state and prepared constitutive coefficients are not
    modified. Summing block contractions changes floating-point reduction order.
    """
    require(callable(correction_times_map) and callable(check)
        and type(block_size) is int and 1 <= block_size <= 131072,
        'explicit original algorithm and bounded block size required')
    require(type(strain_maps) is np.ndarray and strain_maps.dtype == np.float64
        and strain_maps.ndim == 3 and strain_maps.shape[1] == 6,
        'float64 GP-by-six-by-rank maps required')
    n, _, rank = strain_maps.shape
    require(6 <= rank <= 128 and type(weights) is np.ndarray and weights.dtype == np.float64
        and weights.shape == (n,) and np.isfinite(weights).all() and np.all(weights > 0),
        'positive original all-GP weights required')
    require(type(reference_stiffness) is np.ndarray and reference_stiffness.dtype == np.float64
        and reference_stiffness.shape == (rank, rank) and np.isfinite(reference_stiffness).all(),
        'complete original reference stiffness required')
    names = ('dev_trial', 'flow', 'q_trial', 'q_new_derivative', 'beta', 'nonzero')
    for name in names:
        value = getattr(prepared, name, None)
        shape = (n, 6) if name in ('dev_trial', 'flow') else (n,)
        dtype = np.bool_ if name == 'nonzero' else np.float64
        require(type(value) is np.ndarray and value.shape == shape and value.dtype == dtype
            and np.isfinite(value).all(), 'finite original prepared field required: '+name)
    stiffness = prepared.stiffness
    require(np.all(prepared.q_trial >= 0.)
        and np.all(prepared.q_trial[prepared.nonzero] > 0.),
        'original nonzero mask must not permit division by zero')
    require(type(stiffness) is np.ndarray and stiffness.shape == (6, 6)
        and stiffness.dtype == np.float64 and np.isfinite(stiffness).all(),
        'original elastic Cu stiffness required')
    result = reference_stiffness.copy()
    stats = dict(Cu_GP=n, rank=rank, block_size=block_size, active_blocks=0,
        exact_zero_blocks=0, exact_zero_GP=0, point_sampling=False,
        full_reference_stiffness_retained=True, reduction_order_changed=True)
    for first in range(0, n, block_size):
        check()
        last = min(first+block_size, n)
        maps = strain_maps[first:last]
        fields = {name: getattr(prepared, name)[first:last] for name in names}
        require(np.isfinite(maps).all() and np.max(np.abs(maps), initial=0.) <= 1e80
            and np.max(np.abs(stiffness), initial=0.) <= 1e80
            and np.max(np.abs(fields['flow']), initial=0.) <= 1e80
            and np.max(np.abs(fields['dev_trial']), initial=0.) <= 1e80,
            'bounded finite physical map/coefficient inputs required')
        active = (fields['beta'] != 1.) | (fields['nonzero']
            & (fields['q_new_derivative'] != fields['beta']))
        if not np.any(active):
            stats['exact_zero_blocks'] += 1
            stats['exact_zero_GP'] += last-first
            continue
        local = SimpleNamespace(stiffness=stiffness, **fields)
        correction = correction_times_map(maps, local)
        require(type(correction) is np.ndarray and correction.shape == maps.shape
            and correction.dtype == np.float64 and np.isfinite(correction).all(),
            'original algorithm must return finite same-shape corrections')
        weighted = (weights[first:last, None, None]*correction).reshape(-1, rank)
        result += maps.reshape(-1, rank).T@weighted
        stats['active_blocks'] += 1
    require(np.isfinite(result).all(), 'finite assembled tangent required')
    return result, stats
