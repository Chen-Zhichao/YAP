import ast
from pathlib import Path
from types import CodeType, FunctionType, MethodType, SimpleNamespace

import numpy as np
import pytest

from tools.operators import streamed as subject
from test_coupon_tangent_contraction import case, original_algorithm


def fixture():
    root = Path(__file__).resolve().parents[2]
    path = root/'distribution/operators/artifacts/operator_repair_20261015/source/exp12_all_gauss_exact_acceleration.py'
    tree = ast.parse(path.read_bytes())
    tree.body = [n for n in tree.body if not
        (isinstance(n, ast.ImportFrom) and n.module == '__future__')]
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
        and n.name == '_accelerated_assembled_coordinate_tangent')
    maps, weights, reference, prepared = case()
    scope = dict(np=np, _cache_entry_matches=lambda *args: True,
        _prepare_cached_tangent=lambda *args: prepared,
        _algorithmic_correction_times_map=original_algorithm(),
        _COUNTERS={'assembled_coordinate_tangent_builds': 0})
    module_code = compile(tree, str(path), 'exec', dont_inherit=True)
    function_code = next(c for c in module_code.co_consts if isinstance(c, CodeType)
        and c.co_name == node.name)
    scope[node.name] = FunctionType(function_code, scope)
    scope[node.name].__kwdefaults__ = dict(plastic_active=None, elapsed_time_s=None)
    acceleration = SimpleNamespace(__file__=str(path),
        _accelerated_assembled_coordinate_tangent=scope[node.name])
    class Operator:
        pass
    operator = Operator()
    operator.strain_maps, operator.weights_mm3 = maps, weights
    operator.coordinate_size, operator.plastic_active = 12, True
    operator.reference_stiffness = lambda temperature: reference
    state = SimpleNamespace(cu_state=object())
    operator._exact_tangent_cache = {id(state.cu_state): {}}
    return acceleration, operator, state


def test_whole_function_inverse_and_same_tangent_with_original_cache_logic():
    acceleration, operator, state = fixture()
    method, proof = subject._coordinate_function(acceleration, 7)
    original = acceleration._accelerated_assembled_coordinate_tangent
    args = (operator, np.zeros(6), np.zeros(6), 187.5, 81.25, state)
    kwargs = dict(creep_active=False, plastic_active=True, elapsed_time_s=625.)
    expected, actual = original(*args, **kwargs), method(*args, **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert proof['whole_original_coordinate_function_AST_inverse_exact']
    assert proof['replaced_contraction_statements'] == 5
    assert not proof['point_sampling']
    assert original.__globals__['_COUNTERS']['assembled_coordinate_tangent_builds'] == 2


def test_owned_wrapper_preserves_delegate_guard_and_physical_arguments():
    acceleration, operator, state = fixture()
    function, proof = subject._coordinate_function(acceleration, 7)
    calls = []
    delegate = SimpleNamespace(validate=lambda op: calls.append(('validate', op)),
        arguments=lambda op, name, args, kwargs: calls.append(('arguments', name, args, kwargs)))
    owner = subject._CoordinateOwner(operator, delegate, function, proof)
    owner.bound = MethodType(subject._coordinate, operator)
    vars(operator)[subject.MARKER] = owner
    operator.assembled_coordinate_tangent = owner.bound
    result = operator.assembled_coordinate_tangent(np.zeros(6), np.zeros(6), 187.5, 81.25,
        state, creep_active=False, plastic_active=True, elapsed_time_s=625.)
    assert result.shape == (12, 12)
    assert [row[0] for row in calls] == ['validate', 'arguments', 'validate']
    assert calls[1][1] == 'evaluate' and calls[1][3]['elapsed_time_s'] == 625.
    operator.assembled_coordinate_tangent = lambda *args, **kwargs: result
    with pytest.raises(ValueError, match='owned streamed'):
        owner.validate(operator)
