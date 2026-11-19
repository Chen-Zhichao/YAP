import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.coupon_tangent_contraction import contract_coordinate_tangent
from tools.operators import runtime


def original_algorithm():
    root = Path(__file__).resolve().parents[2]
    path = root/'distribution/operators/artifacts/operator_repair_20261015/source/exp12_all_gauss_exact_acceleration.py'
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == runtime.SOURCE_PINS['acceleration_source']
    functions = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)
        and n.name == '_algorithmic_correction_times_map']
    assert len(functions) == 1
    scope = dict(np=np, Array=np.ndarray, PreparedBatchTangent=object)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), scope)
    return scope['_algorithmic_correction_times_map']


def case(elastic=False):
    rng = np.random.default_rng(48312)
    n, rank = 32, 12
    maps = rng.normal(size=(n, 6, rank))
    weights = rng.uniform(.1, 1., n)
    stiffness = np.diag([3., 3., 3., 1., 1., 1.])
    stiffness[:3, :3] += 1.
    beta = np.ones(n) if elastic else rng.uniform(.6, 1., n)
    derivative = np.ones(n) if elastic else rng.uniform(.1, 1., n)
    prepared = SimpleNamespace(stiffness=stiffness, dev_trial=rng.normal(size=(n, 6)),
        flow=rng.normal(size=(n, 6)), q_trial=rng.uniform(1., 2., n),
        q_new_derivative=derivative, beta=beta, nonzero=np.ones(n, dtype=bool))
    return maps, weights, np.diag(np.arange(1., rank+1.)), prepared


@pytest.mark.parametrize('size', [1, 7, 32, 131072])
def test_same_original_algorithm_same_all_gp_contraction(size):
    maps, weights, reference, prepared = case()
    original = original_algorithm()
    expected = reference + maps.reshape(-1, 12).T@(weights[:, None, None]
        * original(maps, prepared)).reshape(-1, 12)
    actual, proof = contract_coordinate_tangent(maps, weights, reference, prepared,
        correction_times_map=original, block_size=size)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert proof['Cu_GP'] == 32 and not proof['point_sampling']
    np.testing.assert_array_equal(reference, np.diag(np.arange(1., 13.)))


def test_exact_elastic_zero_never_materializes_gp_map_corrections():
    maps, weights, reference, prepared = case(elastic=True)
    def unnecessary(*args):
        raise AssertionError('exact zero branch must not allocate the full correction')
    actual, proof = contract_coordinate_tangent(maps, weights, reference, prepared,
        correction_times_map=unnecessary, block_size=7)
    np.testing.assert_array_equal(actual, reference)
    assert proof['exact_zero_GP'] == len(weights) and proof['active_blocks'] == 0


def test_mixed_blocks_skip_only_identically_zero_corrections():
    maps, weights, reference, prepared = case()
    prepared.beta[:16] = prepared.q_new_derivative[:16] = 1.
    original = original_algorithm()
    expected = reference + maps.reshape(-1, 12).T@(weights[:, None, None]
        * original(maps, prepared)).reshape(-1, 12)
    actual, proof = contract_coordinate_tangent(maps, weights, reference, prepared,
        correction_times_map=original, block_size=16)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert proof['exact_zero_GP'] == 16 and proof['active_blocks'] == 1


@pytest.mark.parametrize('fault', ['maps', 'weights', 'coefficient', 'mask'])
def test_nonfinite_or_negative_inputs_cannot_be_hidden_by_zero_shortcut(fault):
    maps, weights, reference, prepared = case(elastic=True)
    if fault == 'maps':
        maps[0, 0, 0] = np.nan
    elif fault == 'weights':
        weights[0] = -1.
    elif fault == 'coefficient':
        prepared.flow[0, 0] = np.inf
    else:
        prepared.q_trial[0] = 0.
    with pytest.raises(ValueError):
        contract_coordinate_tangent(maps, weights, reference, prepared,
            correction_times_map=original_algorithm())
