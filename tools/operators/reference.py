"""Owned rank9 dense-map Cu35/483 tangent, without constitutive installation.

The compiled roots, correction map and blocked contraction preserve the qualified
BI7 arithmetic. Only one instance's assembled coordinate tangent is replaced.
The caller owns data authentication, runtime admission and factory qualification.
No committed state, raw prepared tangent or material history is cached here.
"""
import ctypes
import platform
import sys
import time
import weakref
from pathlib import Path
from types import FunctionType, MappingProxyType, MethodType

import numpy as np
from numba import config, njit, prange

MARKER = '_coupon_reference_tangent_owner'
BLOCK_SIZE = 32768
SOURCE_PINS = MappingProxyType({
    'exp12_cu_material_point.py': '35e7016dab8b1fb2cb0a928e7642665d3fc4f00b4f255eb797fb6d6e44b56722',
    'exp12_all_gauss_exact_acceleration.py': '483c9d948931d48beb73561842cb8fe780db285d3b61f158e580229fbdcd465f',
})
PORTED_SOURCE_PINS = MappingProxyType({
    'compiled_reference': '0be86745e1c4afaeda9157834270003c6a273ec5ad9e3bc25ef976da9eed9a54',
    'compiled_maps': 'ff1d49a950c7652e8595f22c6b47dedb649133014bdf0a59678a753e2472f132',
    'blocked_contraction': '5956a1f243ea0468fe4f8048e77cb167d91ce04e5369c23d02ddfe9854678c00',
    'zero_creep': 'cad9d889d304dbbd639ea4784033406d4de111a3ef8af41c46f2b53c1fd4461f',
})
F0_HISTORY_AUDIT_SHA256 = 'f3a607a8a25865bcfa0f62fc8a9eeab91f780731cc1f57a70a2ce331f65b9e4d'
FIELDS = ('plastic_strain', 'creep_strain', 'equivalent_plastic_strain', 'equivalent_creep_strain')
_UNCHANGED_METHODS = ('evaluate', 'evaluate_many', 'tangent_action', 'assembled_bubble_tangent')

_libc = ctypes.CDLL(None)
_thread_rounding = getattr(_libc, 'fegetround', None)
if _thread_rounding is not None:
    _thread_rounding.argtypes = []
    _thread_rounding.restype = ctypes.c_int
_supported_rounding_platform = (sys.platform.startswith('linux')
    and platform.machine() == 'x86_64' and _thread_rounding is not None)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@njit(cache=False, fastmath=False, inline='always')
def _residual(value, upper, factor, exponent, shear, hardening, yield_n, coupled):
    if coupled:
        return (value + factor * value**exponent
                + 3.0 * shear * max((value - yield_n) / hardening, 0.0) - upper)
    return value + factor * value**exponent - upper


@njit(cache=False, fastmath=False, parallel=True)
def _roots(upper, yield_n, cutoff, factor, exponent, shear, hardening, plastic_active):
    count = len(upper)
    creep = np.empty(count, dtype=np.float64)
    selected = np.empty(count, dtype=np.float64)
    plastic = np.zeros(count, dtype=np.bool_)
    invalid = np.zeros(count, dtype=np.bool_)
    for i in prange(count):
        q, y = upper[i], yield_n[i]
        invalid[i] = (
            _residual(0.0, q, factor, exponent, shear, hardening, y, False) > 1.e-12
            or _residual(q, q, factor, exponent, shear, hardening, y, False) < -1.e-12
            or _residual(0.0, q, factor, exponent, shear, hardening, y, True) > 1.e-12
            or _residual(q, q, factor, exponent, shear, hardening, y, True) < -1.e-12)
        low, high = 0.0, q
        for _ in range(64):
            middle = 0.5 * (low + high)
            if _residual(middle, q, factor, exponent, shear, hardening, y, False) < 0.0:
                low = middle
            else:
                high = middle
        creep[i] = 0.5 * (low + high)
        plastic[i] = plastic_active and creep[i] > cutoff[i]
        selected[i] = creep[i]
        if plastic[i]:
            low, high = 0.0, q
            for _ in range(64):
                middle = 0.5 * (low + high)
                if _residual(middle, q, factor, exponent, shear, hardening, y, True) < 0.0:
                    low = middle
                else:
                    high = middle
            selected[i] = 0.5 * (low + high)
    return creep, selected, plastic, invalid


@njit(cache=False, fastmath=False, parallel=True)
def _zero(upper, yield_n, cutoff, creep, exponent, shear, hardening, plastic_active):
    count = len(upper)
    selected = np.empty(count, dtype=np.float64)
    plastic = np.zeros(count, dtype=np.bool_)
    invalid = np.zeros(count, dtype=np.bool_)
    used_fast = np.zeros(count, dtype=np.bool_)
    for i in prange(count):
        q, y = upper[i], yield_n[i]
        invalid[i] = (
            _residual(0.0, q, 0.0, exponent, shear, hardening, y, False) > 1.e-12
            or _residual(q, q, 0.0, exponent, shear, hardening, y, False) < -1.e-12
            or _residual(0.0, q, 0.0, exponent, shear, hardening, y, True) > 1.e-12
            or _residual(q, q, 0.0, exponent, shear, hardening, y, True) < -1.e-12)
        used_fast[i] = _thread_rounding() == 0
        if not used_fast[i]:
            low, high = 0.0, q
            for _ in range(64):
                middle = 0.5 * (low + high)
                if _residual(middle, q, 0.0, exponent, shear, hardening, y, False) < 0.0:
                    low = middle
                else:
                    high = middle
            creep[i] = 0.5 * (low + high)
        plastic[i] = plastic_active and creep[i] > cutoff[i]
        selected[i] = creep[i]
        if plastic[i]:
            low, high = 0.0, q
            for _ in range(64):
                middle = 0.5 * (low + high)
                if _residual(middle, q, 0.0, exponent, shear, hardening, y, True) < 0.0:
                    low = middle
                else:
                    high = middle
            selected[i] = 0.5 * (low + high)
    return creep, selected, plastic, invalid, used_fast


def _eligible(upper, factor, exponent):
    return (_supported_rounding_platform and factor == 0.0 and exponent == 2.5
        and upper.dtype == np.float64 and upper.ndim == 1 and upper.flags.c_contiguous
        and bool(np.all((upper == 0.0) | ((upper >= 1.e-100) & (upper <= 1.e6)))))


def _statistics():
    return dict(calls=0, shortcut_domain_calls=0, fast_points=0,
                worker_rounding_fallback_points=0, parent_kernel_points=0)


def roots(upper, yield_n, *, factor, exponent, shear, hardening, plastic_active,
          zero_creep_shortcut=True, _statistics=None):
    upper, yield_n = np.asarray(upper), np.asarray(yield_n)
    if (upper.dtype != np.float64 or yield_n.dtype != np.float64
            or upper.ndim != 1 or yield_n.shape != upper.shape
            or not upper.flags.c_contiguous or not yield_n.flags.c_contiguous
            or not np.isfinite(upper).all() or not np.isfinite(yield_n).all()
            or np.any(upper < 0.0) or np.any(yield_n < 0.0)
            or not np.isfinite([factor, exponent, shear, hardening]).all()
            or factor < 0.0 or exponent <= 1.0 or shear <= 0.0 or hardening <= 0.0):
        raise ValueError('finite contiguous float64 roots and positive material constants required')
    _require(type(zero_creep_shortcut) is bool, 'explicit boolean F0 selection required')
    cutoff = yield_n + 1.e-10 * np.maximum(1.0, yield_n)
    args = (float(exponent), float(shear), float(hardening), bool(plastic_active))
    stats = _statistics
    if stats is not None:
        stats['calls'] += 1
    if zero_creep_shortcut and _eligible(upper, factor, exponent):
        bits = upper.view(np.uint64) & np.uint64(0xfffffffffffffffe)
        creep = bits.view(np.float64)
        creep[upper == 0.0] = 0.0
        result = _zero(upper, yield_n, cutoff, creep, *args)
        creep, selected, plastic, invalid = result[:4]
        if stats is not None:
            used = int(np.count_nonzero(result[4]))
            stats['shortcut_domain_calls'] += 1
            stats['fast_points'] += used
            stats['worker_rounding_fallback_points'] += len(upper) - used
    else:
        creep, selected, plastic, invalid = _roots(
            upper, yield_n, cutoff, float(factor), *args)
        if stats is not None:
            stats['parent_kernel_points'] += len(upper)
    if invalid.any() or not np.isfinite(selected).all():
        raise ValueError('invalid scalar root bracket or nonfinite compiled root')
    return creep, selected, plastic


def _prepare(np, cu, acceleration, total_strain, temperature_c, delta_time_s,
             committed, trial=None, *, card, creep_active, plastic_active=True,
             elapsed_time_s=None):
    strain = np.asarray(total_strain, dtype=float)
    if (strain.ndim != 2 or strain.shape[1] != 6 or not np.isfinite(strain).all()
            or not np.isfinite(delta_time_s) or delta_time_s < 0.0):
        raise ValueError('finite point-by-six strain and nonnegative time step required')
    committed.validate(len(strain))
    e_mpa, nu = card.elastic(temperature_c)
    stiffness = cu.isotropic_stiffness(e_mpa, nu)
    thermal = cu.thermal_strain_vector(temperature_c, card)
    elastic_trial = (strain - thermal[None, :] - np.asarray(committed.plastic_strain)
                     - np.asarray(committed.creep_strain))
    stress_trial = elastic_trial @ stiffness.T
    pressure = np.mean(stress_trial[:, :3], axis=1)
    dev_trial = stress_trial.copy()
    dev_trial[:, :3] -= pressure[:, None]
    q_trial = np.sqrt(np.maximum(1.5 * (np.sum(dev_trial[:, :3]**2, axis=1)
                      + 2.0 * np.sum(dev_trial[:, 3:]**2, axis=1)), 0.0))
    shear = e_mpa / (2.0 * (1.0 + nu))
    hardening = card.hardening_modulus(temperature_c)
    yield_n = card.yield_stress(temperature_c) + hardening * np.asarray(
        committed.equivalent_plastic_strain)
    coefficient = card.creep_step_coefficient(temperature_c, delta_time_s,
        elapsed_time_s=elapsed_time_s) if creep_active else 0.0
    factor = 3.0 * shear * delta_time_s * coefficient
    exponent = card.creep_exponent
    _, q_new, plastic = roots(q_trial, yield_n, factor=factor, exponent=exponent,
        shear=shear, hardening=hardening, plastic_active=plastic_active)
    derivative = 1.0 / (1.0 + factor * exponent * np.maximum(q_new, 0.0)**(exponent - 1.0)
                        + plastic.astype(float) * 3.0 * shear / hardening)
    nonzero = q_trial > np.finfo(float).eps
    flow = np.zeros_like(dev_trial)
    flow[nonzero, :3] = 1.5 * dev_trial[nonzero, :3] / q_trial[nonzero, None]
    flow[nonzero, 3:] = 3.0 * dev_trial[nonzero, 3:] / q_trial[nonzero, None]
    beta = np.ones_like(q_trial)
    beta[nonzero] = q_new[nonzero] / q_trial[nonzero]
    if not np.isfinite(beta).all() or not np.isfinite(derivative).all():
        raise ValueError('nonfinite compiled tangent coefficients')
    for value in (stiffness, dev_trial, flow, beta, derivative, q_trial, nonzero):
        value.setflags(write=False)
    return acceleration.PreparedBatchTangent(stiffness=stiffness, dev_trial=dev_trial,
        flow=flow, beta=beta, q_new_derivative=derivative, q_trial=q_trial, nonzero=nonzero)


@njit(cache=False, fastmath=False, parallel=True)
def _correction(maps, dev, flow, beta, derivative, q, nonzero, shear, lame):
    output = np.empty_like(maps)
    for g in prange(maps.shape[0]):
        radial = 0.0
        if nonzero[g]:
            radial = (derivative[g]-beta[g])/q[g]
        for j in range(maps.shape[2]):
            normal_sum = 0.0
            normal_sum += maps[g,0,j]
            normal_sum += maps[g,1,j]
            normal_sum += maps[g,2,j]
            s0 = lame*normal_sum + (2.0*shear)*maps[g,0,j]
            s1 = lame*normal_sum + (2.0*shear)*maps[g,1,j]
            s2 = lame*normal_sum + (2.0*shear)*maps[g,2,j]
            s3 = shear*maps[g,3,j]
            s4 = shear*maps[g,4,j]
            s5 = shear*maps[g,5,j]
            pressure = 0.0
            pressure += s0
            pressure += s1
            pressure += s2
            pressure = pressure/3.0
            qmap = 0.0
            qmap += flow[g,0]*s0
            qmap += flow[g,1]*s1
            qmap += flow[g,2]*s2
            qmap += flow[g,3]*s3
            qmap += flow[g,4]*s4
            qmap += flow[g,5]*s5
            output[g,0,j] = (beta[g]-1.0)*(s0-pressure) + (radial*dev[g,0])*qmap
            output[g,1,j] = (beta[g]-1.0)*(s1-pressure) + (radial*dev[g,1])*qmap
            output[g,2,j] = (beta[g]-1.0)*(s2-pressure) + (radial*dev[g,2])*qmap
            output[g,3,j] = (beta[g]-1.0)*s3 + (radial*dev[g,3])*qmap
            output[g,4,j] = (beta[g]-1.0)*s4 + (radial*dev[g,4])*qmap
            output[g,5,j] = (beta[g]-1.0)*s5 + (radial*dev[g,5])*qmap
    return output


def correction_times_map(strain_maps, prepared):
    maps = np.asarray(strain_maps, dtype=float)
    if maps.ndim != 3 or maps.shape[:2] != prepared.dev_trial.shape or maps.shape[1] != 6:
        raise ValueError('strain_maps are incompatible with prepared tangent')
    return _correction(maps, prepared.dev_trial, prepared.flow, prepared.beta,
        prepared.q_new_derivative, prepared.q_trial, prepared.nonzero,
        float(prepared.stiffness[3,3]), float(prepared.stiffness[0,1]))


def _validate_binding(binding, kernels, card):
    runtime = sys.modules.get(type(binding).__module__)
    expected = Path(__file__).resolve().parents[2] / 'mechanics_runtime.py'
    _require(runtime is not None and Path(getattr(runtime, '__file__', '')).resolve() == expected,
             'actual sibling corrected material binding required')
    runtime._validate_corrected_material_binding(binding, material_kernels=kernels, cu_card=card)


def _material_protocol(operator, material):
    binding, cu = material.binding, material.python_reference
    implementation = binding.implementation
    acceleration = implementation.acceleration
    pins = {Path(path).name: pin for path, pin in implementation.sources}
    _require(len(implementation.sources) == 2 and pins == dict(SOURCE_PINS)
        and implementation.cu is cu and binding.cu_card is material.cu_card
        and binding.material_kernels is material.material_kernels
        and operator.corrected_material_binding is binding
        and operator.material_kernels is material.material_kernels and operator.cu_card is material.cu_card
        and material.material_kernels.update_batch is acceleration.numba_exact_trial_update_batch
        and material.material_kernels.tangent_batch is cu.trial_tangent_action_batch
        and material.material_kernels.elastic_stiffness is cu.isotropic_stiffness
        and type(material.cu_card) is cu.CuCard,
        'exact Cu35/483 source protocol and material identities required')
    _validate_binding(binding, material.material_kernels, material.cu_card)


def _method_identity(value):
    function = getattr(value, '__func__', value)
    return (getattr(value, '__self__', None), function, getattr(function, '__code__', None))


class _Owner:
    def __init__(self, operator, material, check, admit, zero_creep_shortcut):
        self.operator = weakref.ref(operator)
        self.material, self.check, self.admit = material, check, admit
        self.maps, self.weights = operator.strain_maps, operator.weights_mm3
        self.block_size, self.zero_creep_shortcut = BLOCK_SIZE, zero_creep_shortcut
        self.calls, self.seconds, self.points = 0, 0.0, 0
        self.bound, self._busy = None, False
        self._callbacks = (check, admit)
        self._binding, self._implementation = material.binding, material.binding.implementation
        self._sources = tuple(material.binding.implementation.sources)
        self._count = operator.cu_point_count
        self._plastic = operator.plastic_active
        self._methods = tuple(_method_identity(getattr(operator, n, None)) for n in _UNCHANGED_METHODS)
        self._reference = _method_identity(operator.reference_stiffness)
        self._binding_validator = _validate_binding
        self._helpers = tuple((name, globals()[name], globals()[name].__code__) for name in
            ('_assembled', '_prepare', 'roots', 'correction_times_map', '_material_protocol'))
        self._kernels = tuple((name, globals()[name], globals()[name].py_func.__code__) for name in
            ('_residual', '_roots', '_zero', '_correction'))
        self._root_statistics = _statistics()
        def private_roots(*args, **kwargs):
            return roots(*args, **kwargs, zero_creep_shortcut=zero_creep_shortcut,
                         _statistics=self._root_statistics)
        self._private_roots = private_roots
        self.prepare = FunctionType(_prepare.__code__, dict(_prepare.__globals__, roots=private_roots),
                                   _prepare.__name__, _prepare.__defaults__, _prepare.__closure__)
        self.prepare.__kwdefaults__ = _prepare.__kwdefaults__
        self._preparation = self.prepare
        self.proof = MappingProxyType(dict(schema='yap.owned_coupon_reference_tangent.v1',
            ported_source_sha256=PORTED_SOURCE_PINS, material_source_sha256=SOURCE_PINS,
            F0_history_audit_sha256=F0_HISTORY_AUDIT_SHA256,
            block_size=BLOCK_SIZE, rank=operator.coordinate_size, Cu_GP=self._count,
            zero_creep_shortcut=zero_creep_shortcut, worker_rounding_checked=True,
            fastmath=False, per_instance_only=True, point_sampling=False,
            native_evaluate_unchanged=True, material_kernels_unchanged=True,
            raw_state_cache=False, factory_C1_qualified=False, assembly_qualified=False))

    @property
    def root_statistics(self):
        return MappingProxyType(self._root_statistics)

    def validate(self, operator):
        _require(self.operator() is operator and vars(operator).get(MARKER) is self
            and operator.assembled_coordinate_tangent is self.bound
            and self.bound.__self__ is operator and self.bound.__func__ is _assembled
            and self.maps is operator.strain_maps and self.weights is operator.weights_mm3
            and type(self.maps) is np.ndarray and self.maps.shape == (self._count, 6, operator.coordinate_size)
            and self.maps.dtype == np.dtype('<f8') and self.maps.flags.c_contiguous
            and not self.maps.flags.writeable and self.weights.shape == (self._count,)
            and self.weights.dtype == np.dtype('<f8') and self.weights.flags.c_contiguous
            and not self.weights.flags.writeable and operator.factored_maps is None
            and operator.macro_size == 6 and operator.coordinate_size == operator.bubble_size+6
            and operator.coordinate_size in (9,11)
            and operator.cu_point_count == self._count and self.block_size == BLOCK_SIZE
            and self.zero_creep_shortcut is self.proof['zero_creep_shortcut']
            and self.check is self._callbacks[0] and self.admit is self._callbacks[1]
            and self.material.binding is self._binding
            and self.material.binding.implementation is self._implementation
            and tuple(self._implementation.sources) == self._sources
            and operator.plastic_active is self._plastic
            and tuple(_method_identity(getattr(operator, n, None)) for n in _UNCHANGED_METHODS) == self._methods
            and _method_identity(operator.reference_stiffness) == self._reference
            and _validate_binding is self._binding_validator
            and self.prepare is self._preparation and self.prepare.__code__ is _prepare.__code__
            and self.prepare.__globals__['roots'] is self._private_roots
            and all(globals()[name] is fn and fn.__code__ is code for name, fn, code in self._helpers)
            and all(globals()[name] is fn and fn.py_func.__code__ is code
                    and not fn.targetoptions.get('fastmath', False) for name, fn, code in self._kernels),
            'owned reference tangent inputs, callbacks or implementation changed')
        _material_protocol(operator, self.material)


def _assembled(operator, macro_coordinates, bubble_coordinates, temperature_c,
               delta_time_s, committed, *, creep_active, plastic_active=None,
               elapsed_time_s=None):
    owner = getattr(operator, MARKER, None)
    _require(type(owner) is _Owner, 'explicit per-instance tangent owner required')
    owner.validate(operator)
    _require(not owner._busy, 'reference tangent owner is not reentrant')
    owner._busy = True
    try:
        started = time.perf_counter()
        macro = np.asarray(macro_coordinates, dtype=float)
        bubble = np.asarray(bubble_coordinates, dtype=float)
        birth_macro = np.asarray(committed.birth_macro_coordinates)
        birth_bubble = np.asarray(committed.birth_bubble_coordinates)
        _require(macro.shape == birth_macro.shape == (6,)
            and bubble.shape == birth_bubble.shape == (operator.bubble_size,)
            and all(np.isfinite(a).all() for a in (macro, bubble, birth_macro, birth_bubble))
            and np.isfinite(temperature_c) and np.isfinite(delta_time_s) and delta_time_s >= 0.0
            and (elapsed_time_s is None or np.isfinite(elapsed_time_s)),
            'finite coordinates, current births and physical clock required')
        committed.cu_state.validate(owner._count)
        mechanical = np.concatenate((macro, bubble)) - np.concatenate((birth_macro, birth_bubble))
        material = owner.material
        cu = material.python_reference
        exact = material.binding.implementation.acceleration
        active = operator.plastic_active if plastic_active is None else bool(plastic_active)
        tangent = operator.reference_stiffness(temperature_c).copy()
        _require(tangent.shape == (operator.coordinate_size, operator.coordinate_size) and tangent.dtype == np.float64
                 and np.isfinite(tangent).all(), 'finite full rank9 reference stiffness required')
        rank = operator.coordinate_size
        count = operator.cu_point_count
        for first in range(0, count, owner.block_size):
            owner.check()
            owner.validate(operator)
            last = min(first + owner.block_size, count)
            maps = owner.maps[first:last]
            local_state = cu.CuBatchState(*(getattr(committed.cu_state, field)[first:last]
                for field in FIELDS))
            strain = (maps.reshape(-1, rank) @ mechanical).reshape(-1, 6)
            prepared = owner.prepare(np, cu, exact, strain, temperature_c, delta_time_s,
                local_state, card=operator.cu_card, creep_active=creep_active,
                plastic_active=active, elapsed_time_s=elapsed_time_s)
            correction = correction_times_map(maps, prepared)
            weighted = owner.weights[first:last, None, None] * correction
            tangent += maps.reshape(-1, rank).T @ weighted.reshape(-1, rank)
        owner.check()
        _require(np.isfinite(tangent).all(), 'nonfinite assembled reference tangent')
        owner.validate(operator)
        owner.calls += 1
        owner.points += count
        owner.seconds += time.perf_counter() - started
        return tangent
    finally:
        owner._busy = False


def attach_reference_tangent(operator, material, *, check=lambda: None, admit,
                             block_size=32768, zero_creep_shortcut=True):
    """Attach once to a new immutable rank9 operator; return its owned evidence.

    Maps/weights remain caller-owned read-only inputs, never copies or caches of
    incoming Cu state. Source custody and full factory C1 acceptance remain the
    caller's responsibility. No numerical execution or JIT warmup occurs here.
    """
    _require(not config.DISABLE_JIT, 'compiled execution required')
    _require(callable(check) and callable(admit) and type(block_size) is int
        and block_size == BLOCK_SIZE and type(zero_creep_shortcut) is bool,
        'callable guards, fixed32768 blocking and boolean F0 selection required')
    _require(MARKER not in vars(operator) and 'assembled_coordinate_tangent' not in vars(operator)
        and all(n not in vars(operator) for n in _UNCHANGED_METHODS)
        and operator.macro_size == 6 and operator.coordinate_size == operator.bubble_size+6
        and operator.coordinate_size in (9,11)
        and operator.factored_maps is None, 'new unmodified dense rank9 instance required')
    maps, weights = operator.strain_maps, operator.weights_mm3
    count = operator.cu_point_count
    _require(type(count) is int and count > 0 and type(maps) is type(weights) is np.ndarray
        and maps.shape == (count, 6, operator.coordinate_size) and weights.shape == (count,)
        and maps.dtype == weights.dtype == np.dtype('<f8')
        and maps.flags.c_contiguous and weights.flags.c_contiguous
        and not maps.flags.writeable and not weights.flags.writeable,
        'readonly contiguous full float64 maps and weights required')
    _material_protocol(operator, material)
    check()
    forecast = 256 * 1024**2 + block_size * (6 * operator.coordinate_size * 8 * 8 + 2048)
    _require(admit('BEFORE_BLOCK_REFERENCE_TANGENT', dict(
        additional_allocation_forecast_bytes=forecast, block_size=block_size,
        all_GP=True, sampling=False)) is not False, 'block workspace admission refused')
    for first in range(0, count, block_size):
        check()
        _require(np.isfinite(maps[first:first+block_size]).all()
            and np.isfinite(weights[first:first+block_size]).all()
            and np.all(weights[first:first+block_size] > 0), 'finite maps and positive weights required')
    _require(operator.strain_maps is maps and operator.weights_mm3 is weights,
             'operator inputs changed during admission')
    owner = _Owner(operator, material, check, admit, zero_creep_shortcut)
    owner.bound = MethodType(_assembled, operator)
    setattr(operator, MARKER, owner)
    operator.assembled_coordinate_tangent = owner.bound
    owner.validate(operator)
    return owner
