"""Rescore sealed operator predictions against conjugate native reactions.

Each invocation creates a new local result directory. The old stress-centroid
scores and candidate datasets remain immutable inputs to this reference epoch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.build_prepared_operators as storage


FORCES = ("Nx", "Ny", "Nxy", "Mx", "My", "Mxy")
AREA_MM2 = 0.006**2


def read_record(path):
    path = storage.safe_path(path)
    if not path.is_file() or path.stat().st_size > 2 * 1024**2:
        raise ValueError(f"missing or oversized record: {path}")
    return json.loads(path.read_text())


def score_response(prediction, reference, times, rows):
    import numpy as np
    import tools.operator_thermal_validation as thermal

    prediction, reference = np.asarray(prediction), np.asarray(reference)
    times = np.asarray(times)
    if (prediction.shape != reference.shape or prediction.ndim != 2
            or prediction.shape[1] != 134 or times.shape != (len(prediction),)
            or len(rows) != len(prediction) or not np.all(np.diff(times) > 0)
            or not all(np.isfinite(a).all() for a in (prediction, reference, times))
            or [r['result_set'] for r in rows] != list(range(1, len(rows) + 1))):
        raise ValueError('invalid prediction/reference/state coverage')
    metrics = np.asarray([[r['scaled_residual_norm'], r['plastic'], r['creep']]
                          for r in rows])
    if not np.isfinite(metrics).all() or np.any(metrics < 0):
        raise ValueError('invalid equilibrium or dissipation metrics')
    gates = {}
    for index, name in enumerate(FORCES):
        p = prediction[:, index] / AREA_MM2
        r = reference[:, index] / AREA_MM2
        absolute = 1e-3 if index < 3 else 1e-5
        gate = thermal.component_gate(p, r, absolute)
        error = abs(p - r)
        limit = np.maximum(absolute, .1 * abs(r))
        worst = int(np.argmax(error / limit))
        failed = np.flatnonzero(error > limit)
        gate.update(units='N/mm' if index < 3 else 'N',
            worst_result_set=worst + 1, worst_time_s=float(times[worst]),
            prediction=float(p[worst]), reference=float(r[worst]),
            error_at_worst=float(error[worst]), allowed_at_worst=float(limit[worst]),
            first_failed_result_set=int(failed[0]) + 1 if len(failed) else None,
            failed_states=int(len(failed)), rms_error=float(np.sqrt(np.mean(error**2))))
        gates[name] = gate
    for face, start in (('bottom_ports', 6), ('top_ports', 70)):
        error = np.linalg.norm(prediction[:, start:start + 64]
                               - reference[:, start:start + 64], axis=1)
        norm = np.linalg.norm(reference[:, start:start + 64], axis=1)
        limit = np.maximum(1e-4, .1 * norm)
        worst = int(np.argmax(error / limit))
        failed = np.flatnonzero(error > limit)
        gates[face] = dict(passed=bool(np.all(error <= limit)), units='N',
            absolute_deadband=1e-4, relative_limit_percent=10.,
            maximum_absolute_error=float(error.max()),
            maximum_normalized_error=float(np.max(error / limit)),
            worst_result_set=worst + 1, error_at_worst=float(error[worst]),
            allowed_at_worst=float(limit[worst]), reference_norm=float(norm[worst]),
            first_failed_result_set=int(failed[0]) + 1 if len(failed) else None,
            failed_states=int(len(failed)))
    gates['bubble_equilibrium'] = dict(passed=bool(metrics[:, 0].max() <= 1e-8),
                                      maximum=float(metrics[:, 0].max()), limit=1e-8)
    plastic, creep = metrics[:, 1:].sum(axis=0)
    gates['inelastic_excitation'] = dict(passed=bool(plastic > 1e-12 and creep > 1e-12),
        plastic_dissipation_Nmm=float(plastic), creep_dissipation_Nmm=float(creep))
    return gates


def run(args):
    work = storage.local_scratch(args.work)
    if work.exists():
        raise FileExistsError('fresh reference-repair result directory required')
    storage.configure(work)
    import numpy as np
    import tools.operator_thermal_validation as thermal
    import tools.operator_work_reference as reference_api

    review = read_record(args.review)
    case = storage.safe_path(review['case'])
    constructor = read_record(args.constructor / 'COMPLETE.json')
    previous = read_record(args.replay / 'COMPLETE.json')
    replay_plan = read_record(args.replay / 'PLAN.json')
    count = previous['states']
    if (review['key'] != constructor['original_key'] or count not in (201, 361, 1081)
            or previous['operator_accuracy_passed'] is not False):
        raise ValueError('candidate or stopped prediction binding differs')
    contract = read_record(case / 'INPUT_CONTRACT.json')
    if storage.digest(case / 'INPUT_CONTRACT.json') != review['input_contract_sha256']:
        raise ValueError('original input contract differs from its sealed review')
    record = contract['candidate']
    if record['key'] != review['key']:
        raise ValueError('original input contract belongs to another geometry')
    if (replay_plan['inputs_sha256'].get(str(storage.safe_path(constructor['candidate'])))
            != constructor['candidate_sha256']
            or replay_plan['inputs_sha256'].get(str(storage.safe_path(args.review)))
            != storage.digest(args.review)
            or review['candidate_sha256'] != record['files_sha256']):
        raise ValueError('sealed replay belongs to another candidate or review')
    native = read_record(args.native_work / 'COMPLETE.json')
    if native['key'] != review['key'] or native['result_sets'] != 1081:
        raise ValueError('native teacher geometry or coverage differs')
    if storage.digest(args.native_work / 'COMPLETE.json') != review['native_complete_sha256']:
        raise ValueError('native receipt differs from the sealed original review')
    lifts = case / Path(record['lifts']).name
    modes = case / Path(record['basis']).name
    pilot = args.native_work / 'teacher/M3_LAYER_INTERFACE_PILOT_REACTIONS.csv'
    face = case / 'teacher/M3_LAYER_INTERFACE_FACE_REACTIONS.csv'
    manifest_path = args.native_work / 'teacher/M3_LAYER_INTERFACE_REPLAY_MANIFEST.json'
    manifest = read_record(manifest_path)
    if (manifest['isolated_mesh_y_frame'] != 'layer_local_0_to_h'
            or manifest['macro_pilot_y_frame'] != 'stack_global'
            or manifest['generalized_coordinates'] !=
                ['eps_x0', 'eps_z0', 'gamma_xz0', 'kappa_x', 'kappa_z', 'kappa_xz']
            or manifest['result_sets'] != 1081 or not manifest['exports_face_reactions']):
        raise ValueError('native coordinate or reference extraction contract differs')
    response_path = storage.safe_path(previous['response_path'])
    if response_path != (args.replay / f'RESPONSE_{count}.npz').resolve():
        raise ValueError('prediction receipt path differs')
    checkpoint_path = args.replay / f'CHECKPOINT_{count}.json'
    checkpoint = read_record(checkpoint_path)
    rows_path = args.replay / f'ROWS_{count}.json'
    expected = {response_path: previous['response_sha256'],
        rows_path: checkpoint['rows_sha256'],
        case / 'PATH.npz': contract['input_sha256']['PATH.npz'],
        lifts: record['files_sha256'][record['lifts']],
        modes: record['files_sha256'][record['basis']],
        case / 'SCORED_RESPONSE.npz': review['scored_response_sha256'],
        case / 'INPUT_CONTRACT.json': review['input_contract_sha256'],
        case / 'TEACHER_SEAL.json': review['teacher_seal_sha256'],
        Path(constructor['candidate']): constructor['candidate_sha256']}
    teacher_seal = read_record(case / 'TEACHER_SEAL.json')
    if storage.digest(case / 'TEACHER_SEAL.json') != review['teacher_seal_sha256']:
        raise ValueError('native teacher seal differs from its sealed review')
    expected[face] = teacher_seal['files_sha256']['teacher/M3_LAYER_INTERFACE_FACE_REACTIONS.csv']
    input_paths = [args.review, args.constructor / 'COMPLETE.json',
        args.replay / 'COMPLETE.json', args.replay / 'PLAN.json',
        checkpoint_path, case / 'INPUT_CONTRACT.json',
        args.native_work / 'COMPLETE.json', manifest_path, pilot,
        Path(__file__).resolve(), Path(reference_api.__file__).resolve(),
        Path(thermal.__file__).resolve(), *expected]
    inputs = {str(storage.safe_path(path)): storage.digest(path) for path in input_paths}
    if any(inputs[str(storage.safe_path(path))] != digest for path, digest in expected.items()):
        raise ValueError('sealed input changed before reference repair')
    storage.write_json(work / 'INPUTS.json', inputs, immutable=True)
    with np.load(case / 'PATH.npz', allow_pickle=False) as saved:
        times = saved['times_s']
    if times.shape != (1081,) or not np.isfinite(times).all():
        raise ValueError('original prescribed clock is incomplete')
    with np.load(response_path, allow_pickle=False) as saved:
        prediction = saved['response']
        if not np.array_equal(saved['time_s'], times[:count]):
            raise ValueError('prediction time differs from prescribed history')
    rows = read_record(rows_path)
    storage.write_json(work / 'STATUS.json', dict(state='EXTRACTING_NATIVE_CONJUGATE_REFERENCE',
        key=review['key'], prediction_states=count, reference_states=1081))
    # The extractor validates coverage and returns work in ROM coordinate units.
    extracted = reference_api.extract_work_reference(
        lift_npz=lifts, modes_npz=modes, pilot_csv=pilot,
        face_csv=face, expected_times_s=times,
        layer_thickness_mm=record['si_thickness_um'] * .001, cell_area_mm2=AREA_MM2)
    reference, extraction = extracted.reference, extracted.metadata
    if reference.shape != (1081, 134):
        raise ValueError('conjugate reference coordinate count differs')
    if any(row['plastic'] > 1e-16 or row['creep'] > 1e-16 for row in rows[:20]):
        raise ValueError('elastic reference calibration prefix contains inelasticity')
    calibration = abs(prediction[:20, :6] - reference[:20, :6]) / AREA_MM2
    if np.any(calibration > np.asarray([1e-8] * 3 + [1e-10] * 3)):
        raise ValueError('independent elastic virtual-work calibration failed')
    with np.load(case / 'SCORED_RESPONSE.npz', allow_pickle=False) as saved:
        legacy = saved['reference']
        if not np.array_equal(saved['time_s'], times):
            raise ValueError('legacy reference clock differs')
    port_roundoff = float(np.max(abs(reference[:, 6:] - legacy[:, 6:])))
    if not np.allclose(reference[:, 6:], legacy[:, 6:], rtol=1e-12, atol=1e-14):
        raise ValueError('reference repair unexpectedly changed port projections')
    reference[:, 6:] = legacy[:, 6:]
    gates = score_response(prediction, reference[:count], times[:count], rows)
    np.savez_compressed(work / 'WORK_REFERENCE.npz', reference=reference, time_s=times)
    np.savez_compressed(work / 'SCORED_RESPONSE.npz', prediction=prediction,
                        reference=reference[:count], time_s=times[:count])
    for path, digest in inputs.items():
        if storage.digest(path) != digest:
            raise ValueError('input changed during reference repair: ' + path)
    passed = all(gate['passed'] for gate in gates.values())
    storage.write_json(work / 'COMPLETE.json', dict(
        schema='yap.operator_conjugate_reference_repair.v1', key=review['key'],
        state='PASS_WORK_REFERENCE_PREFIX_REQUIRES_FULL_HISTORY' if passed and count < 1081
            else 'PASS_WORK_REFERENCE_FULL_RESPONSE' if passed else 'FAIL_WORK_REFERENCE_RESPONSE',
        states=count, full_history_complete=count == 1081, gates=gates,
        reference_extraction=extraction, elastic_calibration_maximum=calibration.max(axis=0).tolist(),
        port_projection_roundoff_N=port_roundoff,
        numerical_gates_unchanged=True, legacy_reference_preserved=True,
        no_native_solver=True, no_history_replay=True,
        operator_accuracy_passed=bool(passed and count == 1081), runtime_admitted=False,
        candidate_sha256=constructor['candidate_sha256'],
        input_sha256=storage.digest(work / 'INPUTS.json'),
        reference_sha256=storage.digest(work / 'WORK_REFERENCE.npz'),
        response_sha256=storage.digest(work / 'SCORED_RESPONSE.npz')), immutable=True)
    storage.write_json(work / 'STATUS.json', dict(state='COMPLETE', key=review['key'],
        states=count, passed=passed, operator_accuracy_passed=bool(passed and count == 1081)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('work', 'review', 'constructor', 'replay', 'native-work'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        if args.work.is_dir() and not (args.work / 'STOP.json').exists():
            storage.write_json(args.work / 'STOP.json', dict(state='STOPPED_REQUIRES_REVIEW',
                error=f'{type(error).__name__}: {error}', operator_accuracy_passed=False), immutable=True)
        raise


if __name__ == '__main__':
    main()
