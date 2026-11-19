"""Standalone synthetic controls; real full-GP/factory acceptance is separate."""
import ctypes
from dataclasses import dataclass
from types import SimpleNamespace

import numba
import numpy as np
import pytest

from tools.operators import reference as subject

REAL_BINDING_VALIDATOR = subject._validate_binding


def immutable(values):
    array = np.ascontiguousarray(values, dtype='<f8')
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class Card:
    creep_exponent: float = 2.5

    def elastic(self, temperature):
        return 131000., .34

    def alpha(self, temperature):
        return 17.6e-6

    def hardening_modulus(self, temperature):
        return 2031.

    def yield_stress(self, temperature):
        return 321.

    def creep_step_coefficient(self, temperature, dt, *, elapsed_time_s=None):
        return 1.e-12


def stiffness(e, nu):
    shear = e / (2. * (1. + nu))
    lame = e * nu / ((1. + nu) * (1. - 2. * nu))
    value = np.zeros((6, 6))
    value[:3, :3] = lame
    value[np.arange(3), np.arange(3)] += 2. * shear
    value[np.arange(3, 6), np.arange(3, 6)] = shear
    return value


def thermal(temperature, card):
    return np.r_[np.full(3, card.alpha(temperature) * (temperature - 25.)), np.zeros(3)]


@dataclass
class State:
    plastic_strain: np.ndarray
    creep_strain: np.ndarray
    equivalent_plastic_strain: np.ndarray
    equivalent_creep_strain: np.ndarray

    @classmethod
    def zeros(cls, n):
        return cls(np.zeros((n, 6)), np.zeros((n, 6)), np.zeros(n), np.zeros(n))

    def validate(self, n):
        for name in subject.FIELDS:
            value = getattr(self, name)
            shape = (n, 6) if name in subject.FIELDS[:2] else (n,)
            if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
                raise ValueError('invalid synthetic committed state')


class Operator:
    macro_size, bubble_size, coordinate_size = 6, 3, 9
    factored_maps, plastic_active = None, True

    def evaluate(self, *args, **kwargs):
        return 'unchanged evaluation'

    evaluate_many = tangent_action = assembled_bubble_tangent = evaluate

    def assembled_coordinate_tangent(self, *args, **kwargs):
        raise AssertionError('unattached original method')


@pytest.fixture
def fixture(monkeypatch):
    def validate(binding, kernels, card):
        assert binding.material_kernels is kernels and binding.cu_card is card
        assert binding.valid
    monkeypatch.setattr(subject, '_validate_binding', validate)
    def make(n=23, seed=2715):
        rng = np.random.default_rng(seed)
        cu = SimpleNamespace(CuBatchState=State, CuCard=Card, isotropic_stiffness=stiffness,
            thermal_strain_vector=thermal, trial_tangent_action_batch=lambda *a, **k: None)
        acceleration = SimpleNamespace(PreparedBatchTangent=lambda **k: SimpleNamespace(**k),
            numba_exact_trial_update_batch=lambda *a, **k: None)
        kernels = SimpleNamespace(update_batch=acceleration.numba_exact_trial_update_batch,
            tangent_batch=cu.trial_tangent_action_batch, elastic_stiffness=cu.isotropic_stiffness)
        card = Card()
        implementation = SimpleNamespace(cu=cu, acceleration=acceleration,
            sources=tuple(('/portable/source/' + name, pin) for name, pin in subject.SOURCE_PINS.items()))
        binding = SimpleNamespace(implementation=implementation, material_kernels=kernels,
            cu_card=card, valid=True)
        material = SimpleNamespace(python_reference=cu, binding=binding, cu_card=card,
            material_kernels=kernels)
        op = Operator()
        op.strain_maps = immutable(rng.normal(size=(n, 6, 9)))
        op.weights_mm3 = immutable(rng.uniform(.01, .02, size=n))
        op.cu_point_count = n
        op.cu_card, op.material_kernels, op.corrected_material_binding = card, kernels, binding
        k = np.eye(9) * 100.
        op.reference_stiffness = lambda temperature: k
        cu_state = State.zeros(n)
        cu_state.plastic_strain[:] = rng.normal(scale=1.e-5, size=(n, 6))
        cu_state.creep_strain[:] = rng.normal(scale=1.e-5, size=(n, 6))
        cu_state.equivalent_plastic_strain[:] = rng.uniform(0., .01, n)
        state = SimpleNamespace(cu_state=cu_state,
            birth_macro_coordinates=rng.normal(scale=1.e-4, size=6),
            birth_bubble_coordinates=rng.normal(scale=1.e-4, size=3))
        return op, material, state
    return make


def root_reference(upper, yield_n, factor, exponent, shear, hardening, plastic_active):
    """Independent scalar Python replay of the literal finite64 algorithm."""
    creep, selected, plastic = [], [], []
    for q, y in zip(upper, yield_n):
        q, y = float(q), float(y)
        def residual(value, coupled):
            if coupled:
                return value + factor * value**exponent + 3.0 * shear * max((value-y)/hardening, 0.0) - q
            return value + factor * value**exponent - q
        low, high = 0., q
        for _ in range(64):
            middle = .5 * (low + high)
            if residual(middle, False) < 0.:
                low = middle
            else:
                high = middle
        c = .5 * (low + high)
        active = bool(plastic_active and c > y + 1.e-10 * max(1., y))
        answer = c
        if active:
            low, high = 0., q
            for _ in range(64):
                middle = .5 * (low + high)
                if residual(middle, True) < 0.:
                    low = middle
                else:
                    high = middle
            answer = .5 * (low + high)
        creep.append(c); selected.append(answer); plastic.append(active)
    return np.array(creep), np.array(selected), np.array(plastic)


def tangent_reference(op, material, state, macro, bubble, temperature, dt, creep, plastic=True):
    """Python reference preparation and directional correction, no subject helpers."""
    x = np.concatenate((macro, bubble)) - np.concatenate((state.birth_macro_coordinates,
                                                        state.birth_bubble_coordinates))
    n = op.cu_point_count
    strain = (op.strain_maps.reshape(-1, 9) @ x).reshape(n, 6)
    card, committed = material.cu_card, state.cu_state
    e, nu = card.elastic(temperature)
    c = stiffness(e, nu)
    trial = strain - thermal(temperature, card)[None, :] - committed.plastic_strain - committed.creep_strain
    stress = trial @ c.T
    pressure = stress[:, :3].mean(axis=1)
    dev = stress.copy(); dev[:, :3] -= pressure[:, None]
    q = np.sqrt(np.maximum(1.5 * (np.sum(dev[:, :3]**2, axis=1) + 2.*np.sum(dev[:, 3:]**2, axis=1)), 0.))
    shear = e / (2.*(1.+nu))
    hard = card.hardening_modulus(temperature)
    yn = card.yield_stress(temperature) + hard * committed.equivalent_plastic_strain
    factor = 3.*shear*dt*(card.creep_step_coefficient(temperature, dt) if creep else 0.)
    _, new, active = root_reference(q, yn, factor, card.creep_exponent, shear, hard, plastic)
    derivative = 1. / (1.+factor*card.creep_exponent*np.maximum(new, 0.)**(card.creep_exponent-1.)
                        + active.astype(float)*3.*shear/hard)
    nonzero = q > np.finfo(float).eps
    beta = np.ones(n); beta[nonzero] = new[nonzero] / q[nonzero]
    flow = np.zeros_like(dev)
    flow[nonzero, :3] = 1.5*dev[nonzero, :3]/q[nonzero, None]
    flow[nonzero, 3:] = 3.*dev[nonzero, 3:]/q[nonzero, None]
    correction = np.empty_like(op.strain_maps)
    for g in range(n):
        for j in range(9):
            elastic = c @ op.strain_maps[g, :, j]
            deviatoric = elastic.copy(); deviatoric[:3] -= elastic[:3].mean()
            correction[g, :, j] = (beta[g]-1.)*deviatoric
            if nonzero[g]:
                correction[g, :, j] += (derivative[g]-beta[g])/q[g]*dev[g]*(flow[g] @ elastic)
    weighted = op.weights_mm3[:, None, None]*correction
    return op.reference_stiffness(temperature).copy() + op.strain_maps.reshape(-1, 9).T @ weighted.reshape(-1, 9)


def attach(op, material, **kwargs):
    return subject.attach_reference_tangent(op, material, admit=lambda *a: None, **kwargs)


def assert_bits(actual, expected):
    for a, b in zip(actual, expected):
        if a.dtype == np.bool_:
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))


@pytest.mark.parametrize('factor', [0., 1.e-8, .015])
@pytest.mark.parametrize('plastic', [False, True])
def test_roots_match_finite_python_reference(factor, plastic):
    upper = np.array([0., -0., 1., np.nextafter(1., np.inf), 320., 400., 999., 10000.])
    yn = np.full(len(upper), 321.)
    kw = dict(factor=factor, exponent=2.5, shear=48880., hardening=2031., plastic_active=plastic)
    actual = subject.roots(upper, yn, **kw)
    expected = root_reference(upper, yn, factor, 2.5, 48880., 2031., plastic)
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-15, atol=0.)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-15, atol=0.)
    np.testing.assert_array_equal(actual[2], expected[2])


def test_f0_bitwise_cutoff_parity_and_no_alias():
    y = 321.
    cutoff = y + 1.e-10*max(1., y)
    upper = np.array([0., -0., 1., np.nextafter(1., np.inf), 1.e-100, 1.e6,
                      np.nextafter(cutoff, -np.inf), cutoff, np.nextafter(cutoff, np.inf)])
    yn = np.full(len(upper), y)
    before = upper.view(np.uint64).copy()
    kw = dict(factor=0., exponent=2.5, shear=48880., hardening=2031., plastic_active=True)
    stats = subject._statistics()
    actual = subject.roots(upper, yn, _statistics=stats, **kw)
    expected = subject.roots(upper, yn, zero_creep_shortcut=False, **kw)
    assert_bits(actual, expected)
    np.testing.assert_array_equal(upper.view(np.uint64), before)
    assert not np.shares_memory(actual[0], upper) and not np.shares_memory(actual[0], actual[1])
    assert actual[0][3] == 1. and not np.signbit(actual[0][1])
    if subject._supported_rounding_platform:
        assert stats['fast_points'] == len(upper)


@pytest.mark.parametrize('q,factor,exponent', [
    (np.nextafter(0., 1.), 0., 2.5), (np.finfo(float).tiny, 0., 2.5),
    (np.nextafter(1.e-100, 0.), 0., 2.5), (np.nextafter(1.e6, np.inf), 0., 2.5),
    (100., 1.e-300, 2.5), (100., 0., 3.),
])
def test_outside_domain_calls_literal_parent(q, factor, exponent):
    upper, yn = np.array([q]), np.array([321.])
    kw = dict(factor=factor, exponent=exponent, shear=48880., hardening=2031., plastic_active=True)
    stats = subject._statistics()
    assert_bits(subject.roots(upper, yn, _statistics=stats, **kw),
                subject.roots(upper, yn, zero_creep_shortcut=False, **kw))
    assert stats['parent_kernel_points'] == 1 and stats['shortcut_domain_calls'] == 0


def test_unsupported_platform_falls_back(monkeypatch):
    monkeypatch.setattr(subject, '_supported_rounding_platform', False)
    stats = subject._statistics()
    subject.roots(np.array([100.]), np.array([321.]), factor=0., exponent=2.5,
                  shear=48880., hardening=2031., plastic_active=True, _statistics=stats)
    assert stats['parent_kernel_points'] == 1 and stats['fast_points'] == 0


@pytest.mark.parametrize('value', [np.nan, np.inf, -1.])
def test_invalid_root_inputs_rejected(value):
    with pytest.raises(ValueError, match='finite contiguous'):
        subject.roots(np.array([value]), np.array([321.]), factor=0., exponent=2.5,
                      shear=48880., hardening=2031., plastic_active=True)


def test_invalid_endpoint_is_not_suppressed(monkeypatch):
    def invalid(*args):
        return np.zeros(1), np.zeros(1), np.zeros(1, dtype=bool), np.ones(1, dtype=bool)
    monkeypatch.setattr(subject, '_roots', invalid)
    with pytest.raises(ValueError, match='invalid scalar root bracket'):
        subject.roots(np.array([100.]), np.array([321.]), factor=1., exponent=2.5,
                      shear=48880., hardening=2031., plastic_active=True)


@pytest.mark.skipif(not subject._supported_rounding_platform, reason='Linux x86_64 FENV control')
def test_each_worker_checks_rounding_and_replays_parent():
    setter = getattr(subject._libc, 'fesetround')
    setter.argtypes, setter.restype = [ctypes.c_int], ctypes.c_int
    @numba.njit(cache=False, fastmath=False, parallel=True)
    def set_workers(mode, n):
        results = np.empty(n, dtype=np.int32)
        for i in numba.prange(n):
            results[i] = setter(mode)
        return results
    upper = np.tile(np.array([1., np.nextafter(1., np.inf), 321., 400.]), 128)
    yn = np.full(len(upper), 321.)
    cutoff = yn + 1.e-10*np.maximum(1., yn)
    # Compile under the normal mode before changing every active worker's FENV.
    subject._roots(upper, yn, cutoff, 0., 2.5, 48880., 2031., True)
    subject._zero(upper, yn, cutoff, upper.copy(), 2.5, 48880., 2031., True)
    set_workers(0, len(upper))
    try:
        assert np.all(set_workers(0x400, len(upper)) == 0)
        expected = subject._roots(upper, yn, cutoff, 0., 2.5, 48880., 2031., True)
        bits = upper.view(np.uint64) & np.uint64(0xfffffffffffffffe)
        actual = subject._zero(upper, yn, cutoff, bits.view(np.float64), 2.5, 48880., 2031., True)
    finally:
        set_workers(0, len(upper))
        setter(0)
    assert not actual[4].any()
    assert_bits(actual[:4], expected)


@pytest.mark.parametrize('creep,plastic,shortcut', [
    (False, True, True), (False, False, True), (True, True, True),
    (True, False, True), (False, True, False),
])
def test_owned_tangent_matches_python_without_mutating_history(fixture, creep, plastic, shortcut):
    op, material, state = fixture()
    macro, bubble = np.full(6, .004), np.full(3, -.002)
    old_methods = tuple(getattr(Operator, n) for n in subject._UNCHANGED_METHODS)
    original = Operator.assembled_coordinate_tangent
    before = [getattr(state.cu_state, n).copy() for n in subject.FIELDS]
    births = state.birth_macro_coordinates.copy(), state.birth_bubble_coordinates.copy()
    owner = attach(op, material, zero_creep_shortcut=shortcut)
    actual = op.assembled_coordinate_tangent(macro, bubble, 350., 90., state,
        creep_active=creep, plastic_active=plastic, elapsed_time_s=1040.)
    expected = tangent_reference(op, material, state, macro, bubble, 350., 90., creep, plastic)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-8)
    for name, previous in zip(subject.FIELDS, before):
        np.testing.assert_array_equal(getattr(state.cu_state, name), previous)
    np.testing.assert_array_equal(state.birth_macro_coordinates, births[0])
    np.testing.assert_array_equal(state.birth_bubble_coordinates, births[1])
    assert tuple(getattr(Operator, n) for n in subject._UNCHANGED_METHODS) == old_methods
    assert Operator.assembled_coordinate_tangent is original
    assert owner.calls == 1 and owner.points == op.cu_point_count and owner.seconds >= 0
    assert owner.root_statistics['calls'] == 1 and not owner.proof['raw_state_cache']


def test_birth_is_subtracted_fresh_and_owners_are_isolated(fixture):
    op, material, state = fixture()
    other, other_material, other_state = fixture(seed=99)
    owner, other_owner = attach(op, material), attach(other, other_material)
    macro, bubble = np.full(6, .003), np.full(3, .001)
    first = op.assembled_coordinate_tangent(macro, bubble, 350., 90., state, creep_active=False)
    state.birth_macro_coordinates += .002
    second = op.assembled_coordinate_tangent(macro, bubble, 350., 90., state, creep_active=False)
    expected = tangent_reference(op, material, state, macro, bubble, 350., 90., False)
    np.testing.assert_allclose(second, expected, rtol=1e-11, atol=1e-8)
    assert not np.array_equal(first, second)
    assert owner.calls == 2 and other_owner.calls == 0 and other_owner.root_statistics['calls'] == 0
    assert owner.prepare is not other_owner.prepare


@pytest.mark.parametrize('change', ['callback', 'admit', 'maps', 'weights', 'method', 'binding', 'prepare', 'source'])
def test_changed_owner_contract_rejected(fixture, change):
    op, material, state = fixture()
    owner = attach(op, material)
    if change == 'callback': owner.check = lambda: None
    elif change == 'admit': owner.admit = lambda *a: None
    elif change == 'maps': op.strain_maps = op.strain_maps.copy()
    elif change == 'weights': op.weights_mm3 = op.weights_mm3.copy()
    elif change == 'method': op.evaluate = lambda *a: None
    elif change == 'binding': op.corrected_material_binding = object()
    elif change == 'prepare': owner.prepare = lambda *a, **k: None
    elif change == 'source': material.binding.implementation.sources = ()
    with pytest.raises(ValueError):
        op.assembled_coordinate_tangent(np.zeros(6), np.zeros(3), 25., 0., state, creep_active=False)


def test_helper_rebinding_rejected(fixture, monkeypatch):
    op, material, state = fixture()
    owner = attach(op, material)
    monkeypatch.setattr(subject, 'correction_times_map', lambda *a: None)
    with pytest.raises(ValueError, match='implementation changed'):
        owner.validate(op)


def test_second_attachment_and_changed_binding_rejected(fixture, monkeypatch):
    op, material, state = fixture()
    attach(op, material)
    with pytest.raises(ValueError, match='new unmodified'):
        attach(op, material)
    op2, material2, _ = fixture()
    monkeypatch.setattr(subject, '_validate_binding', REAL_BINDING_VALIDATOR)
    with pytest.raises(ValueError, match='actual sibling'):
        attach(op2, material2)


@pytest.mark.parametrize('bad', ['writable', 'nan', 'negative_weight', 'source', 'rank', 'block'])
def test_admission_rejects_bad_contract(fixture, bad):
    op, material, _ = fixture()
    kwargs = {}
    if bad == 'writable': op.strain_maps = op.strain_maps.copy()
    elif bad == 'nan':
        maps = op.strain_maps.copy(); maps[0, 0, 0] = np.nan; op.strain_maps = immutable(maps)
    elif bad == 'negative_weight': op.weights_mm3 = immutable(-op.weights_mm3)
    elif bad == 'source': material.binding.implementation.sources = (('x.py', '0'*64),)
    elif bad == 'rank': op.bubble_size = 2
    elif bad == 'block': kwargs['block_size'] = 4096
    with pytest.raises(ValueError):
        attach(op, material, **kwargs)
    assert subject.MARKER not in vars(op)


def test_admission_refusal_and_callback_failure_are_not_hidden(fixture):
    op, material, state = fixture()
    with pytest.raises(ValueError, match='admission refused'):
        subject.attach_reference_tangent(op, material, admit=lambda *a: False)
    assert subject.MARKER not in vars(op)
    fail = [False]
    def check():
        if fail[0]: raise RuntimeError('guard stop')
    owner = attach(op, material, check=check)
    fail[0] = True
    with pytest.raises(RuntimeError, match='guard stop'):
        op.assembled_coordinate_tangent(np.zeros(6), np.zeros(3), 25., 0., state, creep_active=False)
    assert not owner._busy and owner.calls == 0


def test_fixed_block_boundary_and_residual_elastic_state(fixture):
    op, material, state = fixture(n=32769)
    for field in subject.FIELDS: getattr(state.cu_state, field).fill(0.)
    state.birth_macro_coordinates.fill(0.); state.birth_bubble_coordinates.fill(0.)
    owner = attach(op, material)
    result = op.assembled_coordinate_tangent(np.zeros(6), np.zeros(3), 25., 0., state, creep_active=False)
    np.testing.assert_array_equal(result, op.reference_stiffness(25.))
    assert owner.root_statistics['calls'] == 2 and owner.points == 32769


def test_proof_and_statistics_are_readonly(fixture):
    op, material, _ = fixture()
    owner = attach(op, material)
    with pytest.raises(TypeError): owner.proof['block_size'] = 1
    with pytest.raises(TypeError): owner.root_statistics['calls'] = -1
    with pytest.raises(TypeError): owner.proof['ported_source_sha256']['zero_creep'] = 'bad'
