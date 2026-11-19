"""Noncommitting force/JVP parity against a caller-authenticated native bridge.

No FE solve, checkpoint adoption, source loading, history commit or file I/O.
The reference callback must use the existing qualified runtime, not these
translated arrays. A passing probe certificate is not assembly admission.
"""
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

if __package__.endswith('.development'):
    from ..operators.kinematics import require, runtime
else:
    from .operators.kinematics import require, runtime


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('ascii')).hexdigest()


@dataclass(frozen=True)
class Probe:
    name: str
    coordinates: object
    directions: object
    temperature_c: float
    delta_time_s: float
    elapsed_time_s: float
    creep_active: bool
    plastic_active: bool
    committed: object
    reference_state: object


def state_sha(state, check=lambda: None):
    """Hash exact committed candidate bytes without allocating a Cu-sized copy."""
    arrays = {name:np.asarray(getattr(state,name)) for name in
              ('bubble_coordinates','birth_macro_coordinates','birth_bubble_coordinates')}
    arrays.update({name:np.asarray(getattr(state.cu_state,name)) for name in
        ('plastic_strain','creep_strain','equivalent_plastic_strain','equivalent_creep_strain')})
    result = hashlib.sha256()
    for name,value in sorted(arrays.items()):
        check()
        require(value.dtype == np.float64 and value.flags.c_contiguous and np.isfinite(value).all(),
                'finite C-order committed float64 fields required')
        result.update(name.encode('ascii'))
        result.update(str(value.shape).encode('ascii'))
        data = memoryview(value).cast('B')
        for first in range(0,len(data),1024**2):
            check()
            result.update(data[first:first+1024**2])
    return result.hexdigest()


def _comparison(actual, expected, atol, rtol):
    require(actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(),
            'finite comparable force/JVP arrays required')
    error = np.abs(actual-expected)
    bound = atol+rtol*np.abs(expected)
    fractions = np.divide(error,bound,out=np.zeros_like(error),where=bound > 0)
    fractions[(bound == 0) & (error != 0)] = np.inf
    worst = float(np.max(fractions))
    return dict(passed=bool(np.all(error <= bound)), maximum_absolute_error=float(np.max(error)),
                maximum_bound_fraction=worst if math.isfinite(worst) else None,
                zero_bound_mismatch=bool(np.any((bound == 0) & (error != 0))))


def certify(translation, layer, reference_probe, probes, *, reference_binding,
            reference_binding_sha256, tolerances, check=lambda: None):
    """Compare actual low-level layer against the independently supplied bridge.

    reference_probe(x_source, directions_source, probe) returns a mapping with
    residual (9,), jvps (9,9; one result per direction), and measured committed
    state SHA256 values before/after. probe.reference_state is the caller's
    immutable reference handle. The callback does not commit either history.

    Binding/source authentication is the caller's responsibility; canonical
    binding SHA prevents accidental evidence drift, not forged attestations.
    No tolerance is silently derived, relaxed or added to original bounds.
    """
    require(callable(reference_probe) and callable(check) and 1 <= len(probes) <= 32,
            'bounded explicit reference probes and resource callback required')
    require(digest(reference_binding) == runtime._component_sha(reference_binding_sha256),
            'reference binding SHA differs')
    require(reference_binding['kind'] in ('qualified_runtime_probe','synthetic_fixture'),
            'explicit actual-runtime versus synthetic reference scope required')
    require(reference_binding['case'] == translation.coupon.case
        and reference_binding['map_sha256'] == translation.coupon.map_sha256
        and reference_binding['native_sha256'] == translation.coupon.native_sha256
        and reference_binding['operand_measure'] == 'integrated_source_cell'
        and reference_binding['source_reference_y_mm'] == translation.provenance['frame_metadata']['source_reference_y_mm']
        and reference_binding['Cu_state_adoption'] is False, 'same case/map/datum/reference measure required')
    sources = reference_binding['source_sha256']
    for name in ('core','Cu','acceleration','reference_bridge'):
        runtime._component_sha(sources[name])
    require(sources['core'] == '089411833177e40bc7facae29b13bf1309ea9d25c31bcaa614aa5be56058fa9b'
        and sources['Cu'] == '35e7016dab8b1fb2cb0a928e7642665d3fc4f00b4f255eb797fb6d6e44b56722'
        and sources['acceleration'] == '483c9d948931d48beb73561842cb8fe780db285d3b61f158e580229fbdcd465f',
        'original qualified core/Cu/acceleration source identities required')
    require(set(tolerances) == {'force_atol','force_rtol','jvp_atol','jvp_rtol'}
        and all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in tolerances.values()),
        'explicit finite original force/JVP tolerances required')
    require(layer.macro_size == 6 and layer.bubble_size == 3
        and np.array_equal(layer.reference_stiffness(25.),translation.stiffness)
        and np.array_equal(layer.reference_thermal_load(350.),325.*translation.thermal_per_c)
        and np.array_equal(layer.weights_mm3,translation.weights_mm3)
        and np.array_equal(layer.strain_maps,translation.strain_maps), 'actual translated low-level layer required')
    transform, factor = translation.source_from_target, translation.area_factor
    rows, names = [], set()
    for probe in probes:
        check()
        require(type(probe.name) is str and probe.name and probe.name not in names, 'distinct named probes required')
        names.add(probe.name)
        x, directions = np.array(probe.coordinates,dtype=float,copy=True), np.array(probe.directions,dtype=float,copy=True)
        require(x.shape == (9,) and directions.shape == (9,9) and np.isfinite(x).all()
            and np.isfinite(directions).all() and np.linalg.matrix_rank(directions) == 9,
            'finite physical coordinates and all nine independent JVP directions required')
        require(type(probe.creep_active) is bool and type(probe.plastic_active) is bool
            and all(type(v) in (int,float) and math.isfinite(v) for v in
                    (probe.temperature_c,probe.delta_time_s,probe.elapsed_time_s))
            and 25. <= probe.temperature_c <= 350. and 0 <= probe.delta_time_s <= probe.elapsed_time_s
            and probe.elapsed_time_s > 0, 'explicit physical interval endpoint and flags required')
        kwargs = dict(creep_active=probe.creep_active,plastic_active=probe.plastic_active,
                      elapsed_time_s=probe.elapsed_time_s)
        before = state_sha(probe.committed,check)
        actual = layer.evaluate(x[:6],x[6:],probe.temperature_c,probe.delta_time_s,probe.committed,**kwargs)
        residual = np.asarray(actual.response).copy()
        jvps = []
        for direction in directions:
            check()
            jvps.append(layer.tangent_action(x[:6],x[6:],direction,probe.temperature_c,
                                             probe.delta_time_s,probe.committed,**kwargs))
        after_candidate = state_sha(probe.committed,check)
        require(before == after_candidate, 'candidate probe mutated committed material/birth state')
        # Detached arguments prevent an untrusted callback from modifying x/T.
        expected = reference_probe((transform@x).copy(),(directions@transform.T).copy(),probe)
        check()
        runtime._component_sha(expected['state_sha256_before'])
        runtime._component_sha(expected['state_sha256_after'])
        require(expected['state_sha256_before'] == expected['state_sha256_after'], 'reference probe mutated committed state')
        require(state_sha(probe.committed,check) == before, 'reference callback mutated candidate state')
        native_r, native_j = np.asarray(expected['residual']), np.asarray(expected['jvps'])
        require(native_r.shape == (9,) and native_j.shape == (9,9), 'full native force and nine JVP columns required')
        wanted_r, wanted_j = factor*(transform.T@native_r), factor*(native_j@transform)
        force = _comparison(residual,wanted_r,tolerances['force_atol'],tolerances['force_rtol'])
        tangent = _comparison(np.asarray(jvps),wanted_j,tolerances['jvp_atol'],tolerances['jvp_rtol'])
        rows.append(dict(name=probe.name,temperature_c=float(probe.temperature_c),delta_time_s=float(probe.delta_time_s),
            elapsed_time_s=float(probe.elapsed_time_s),creep_active=probe.creep_active,plastic_active=probe.plastic_active,
            force=force,jvp=tangent,all_nine_independent_directions=True,
            candidate_state_sha256=before,reference_state_sha256=expected['state_sha256_before'],
            passed=force['passed'] and tangent['passed']))
    passed = all(row['passed'] for row in rows)
    state = ('PROBE_PARITY_PASSED_NOT_HISTORY_OR_ASSEMBLY_ADMISSION' if passed else 'PROBE_PARITY_FAILED')
    if reference_binding['kind'] == 'synthetic_fixture':
        state = 'SYNTHETIC_'+state
    return dict(schema='yap.native_coupon.force_jvp_parity.v1',passed=passed,
        state=state,
        case=translation.coupon.case,map_sha256=translation.coupon.map_sha256,
        frame_metadata_sha256=translation.provenance['frame_metadata_sha256'],
        reference_binding=reference_binding,reference_binding_sha256=reference_binding_sha256,
        reference_kind=reference_binding['kind'],
        reference_source_authentication_by_caller=reference_binding['kind'] == 'qualified_runtime_probe',
        tolerances=dict(tolerances),rows=rows,
        source_from_target=transform.tolist(),area_factor=factor,committed_states_unchanged=True,
        history_replayed=False,Cu_state_adoption=False,prepared=False,YAPallorders=False,
        directMAPDL=False,assembly_execution_supported=False,operator_admission=False)
