"""Publish exact component metadata without reading numerical operator payloads.

Progress is a locator, not an admission receipt. Each selected assessment is
checked against its closed history, actual construction and physical reference.
Experimental prepared-D2W capability still requires registration at runtime;
this publisher does not certify prepared assembly or wafer thinning.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys


ROOT = Path('/app/nanocad/projects/personal/zchen/YAP_STACK')
PROGRESS_RELATIVE = ('operator_repair_20261015/progress/deployment_20261019_v1/'
                     'PROGRESS_ACCELERATION_20261021_v2.json')
CATALOG_RELATIVE = 'distribution/yap_component_operators_20261021_v1/CATALOG.json'
SCHEMA = 'yap.component_operator_catalog.v2'
JSON_LIMIT = 1024**2
READ_LIMIT = 64 * JSON_LIMIT
ROLES = ('bi', 'beol', 'si')
PROBES = {'corner_pm_um', 'corner_pp_um', 'edge_x_um', 'edge_y_um'}
CU_SHA = '35e7016dab8b1fb2cb0a928e7642665d3fc4f00b4f255eb797fb6d6e44b56722'
ACCEL_SHA = '483c9d948931d48beb73561842cb8fe780db285d3b61f158e580229fbdcd465f'
P2_SI = (7, 17, 27, 37, 47, 48.5, 57, 67, 77, 87, 88.5)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding(value):
    require(isinstance(value, dict) and isinstance(value.get('path'), str)
            and re.fullmatch(r'[0-9a-f]{64}', str(value.get('sha256'))),
            'exact path/SHA256 binding required')
    return {key: value[key] for key in ('path', 'sha256')}


def same_binding(left, right):
    require(binding(left) == binding(right)
            and ('bytes' not in left or 'bytes' not in right or left['bytes'] == right['bytes']),
            'selected history/construction/reference artifact variant differs')


def finite_nonnegative(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'finite nonnegative error metric required')
    return value


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class MetadataReader:
    """Only literal regular JSON files are opened; numerical files are not read."""

    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.cache = {}
        self.used = 0

    def path(self, value):
        path = Path(value)
        require(path.is_absolute() and path.is_relative_to(self.root)
                and path != self.root and len(path.parts) <= 64
                and '..' not in path.parts
                and not {'#snapshot', '.snapshot', 'mapdl_run_roots', 'mapdl_archives'}.intersection(path.parts)
                and not any(p.startswith(('.migration_hold_', 'storage_migration')) for p in path.parts),
                'literal canonical non-snapshot evidence path required')
        return path

    def _open(self, path):
        directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in path.parts[1:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                os.close(directory)
                directory = child
            descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 dir_fd=directory)
        finally:
            os.close(directory)
        return os.fdopen(descriptor, 'rb')

    def read(self, value):
        expected = binding(value) if isinstance(value, dict) else None
        path = self.path(expected['path'] if expected else value)
        require(path.suffix == '.json', 'metadata-only reader refuses non-JSON payloads')
        if str(path) not in self.cache:
            with self._open(path) as stream:
                info = os.fstat(stream.fileno())
                require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= JSON_LIMIT,
                        'bounded regular evidence JSON required: ' + str(path))
                require(self.used + info.st_size <= READ_LIMIT, 'metadata read budget exceeded')
                raw = stream.read(JSON_LIMIT + 1)
                require(len(raw) == info.st_size and _identity(info) == _identity(os.fstat(stream.fileno())),
                        'evidence changed during metadata read')
            self.used += len(raw)

            def unique(pairs):
                result = {}
                for key, item in pairs:
                    require(key not in result, 'duplicate evidence JSON key')
                    result[key] = item
                return result

            def invalid_constant(token):
                raise ValueError('nonfinite evidence JSON constant: ' + token)

            document = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)
            require(isinstance(document, dict), 'evidence JSON object required')
            observed = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())
            self.cache[str(path)] = (document, observed, _identity(info))
        document, observed, identity = self.cache[str(path)]
        if expected is not None:
            same_binding(expected, observed)
            if 'bytes' in value:
                require(type(value['bytes']) is int and value['bytes'] == identity[2],
                        'evidence byte count differs')
        return document, dict(observed)

    def verify_unchanged(self):
        for name, (_, _, identity) in self.cache.items():
            with self._open(Path(name)) as stream:
                require(_identity(os.fstat(stream.fileno())) == identity,
                        'evidence identity changed during catalog construction')


def score(block, rows):
    """Validate every original strict-OR bow probe, not a progress summary."""
    require(isinstance(block, dict) and block.get('passed') is True
            and block.get('matched_rows') == rows and block.get('failed_sample_probe_count') == 0
            and block.get('sample_probe_count') == 4 * rows
            and set(block.get('by_probe', {})) == PROBES,
            'complete passing four-probe bow assessment required')
    relative, absolute = [], []
    for probe in block['by_probe'].values():
        require(probe.get('passed') is True and probe.get('failed_sample_probe_count') == 0
                and probe.get('sample_probe_count') == rows,
                'bow probe is incomplete or failed')
        relative.append(finite_nonnegative(probe['finite_relative_max_pct']))
        absolute.append(finite_nonnegative(probe['max_abs_error_um']))
    return max(relative), max(absolute)


def _terminal(history):
    require(isinstance(history.get('checkpoints'), list) and history['checkpoints'],
            'closed candidate checkpoint chain required')
    terminal = history['checkpoints'][-1]
    require(terminal.get('original_ordinal') == 1080 and terminal.get('all_Cu14') is True,
            'candidate must close all three cycles with actual Cu history')
    context = terminal.get('context')
    require(isinstance(context, dict) and 'operator' in context and 'reference' in context
            and all(history.get('context', {}).get(key) == value for key, value in context.items()),
            'candidate terminal context differs from closed report')
    return terminal


def _validate_bow_evidence(document):
    """Return scoped bow metrics or refuse a failed/incomplete admission report.

    This validates metadata, not numerical payloads or assembly accuracy. The
    caller must separately verify the report SHA and its exact history binding.
    """
    require(isinstance(document, dict), 'bow evidence must be a JSON object')
    schema = document.get('schema')
    if schema == 'yap.h7_graded73.sealed_prefix_post.v1':
        require(document.get('state') == 'CLOSED_GRADED73_SEALED_PREFIX_BOW_PASS'
                and document.get('original_end') == 1080
                and document.get('interpolant_is_not_solved_Cu_history') is True,
                'closed H7 full-three-cycle audit required')
        cycles = document.get('cycles', [])
        require([c.get('cycle') for c in cycles] == [1, 2, 3], 'all three H7 cycles required')
        maxima = []
        for index, cycle in enumerate(cycles):
            require(cycle.get('original_start') == 360 * index
                    and cycle.get('original_end') == 360 * (index + 1)
                    and cycle.get('combined_bow_passed') is True, 'H7 cycle coverage/pass differs')
            score(cycle['matched_solved'], 25)
            maxima.append(score(cycle['bow_only_interpolant'], 361))
        relative, absolute = (max(m[j] for m in maxima) for j in (0, 1))
        states = 73
    elif schema == 'yap.h77_closed_temporal.independent_review.v1':
        require(document.get('state') == 'INDEPENDENT_BOUNDED_TEMPORAL_AUDIT_PASS_NOT_ADMISSION'
                and all(document.get('checks', {}).get(k) is True for k in (
                    'terminal1081_closed_no_failure', 'all_three_histories_same_material_policy_reference',
                    'all_reported_score_dictionaries_recomputed_exact')),
                'closed H77 independent audit required')
        metric = document['metrics']['combined_ROM61_vs_dense_native']
        for key, count in (('bow_interpolant1081', 1081), ('matched61', 61)):
            require(metric[key].get('passed') is True and metric[key].get('rows') == count
                    and metric[key].get('probe_points') == 4 * count, 'H77 audited bow score failed')
        relative = finite_nonnegative(metric['bow_interpolant1081']['maximum_relative_pct'])
        absolute = finite_nonnegative(metric['bow_interpolant1081']['maximum_absolute_um'])
        states = 61
    elif schema == 'yap.graded_history.sealed_dense_bow_assessment.v1':
        require(document.get('state') == 'CLOSED_GRADED_PREFIX_BOW_PASS'
                and document.get('full_three_cycle_bow_passed') is True
                and document.get('original_end') == 1080 and document.get('candidate_states_solved') == 73,
                'closed graded73 full-three-cycle pass required')
        assessment = document['assessment']
        require(assessment.get('dense_solved_prefix_states') == 1081,
                'graded73 bow reference must contain 1081 observations')
        relative, absolute = score(assessment['bow_only_interpolant'], 1081)
        score(assessment['matched_solved'], 73)
        states = 73
    elif schema == 'yap.component_reduced_clock.accuracy_profile.v1':
        require(document.get('state') == 'CLOSED_REDUCED_CLOCK_ACCURACY_PROFILE_QUALIFICATION'
                and document.get('qualification_passed') is True
                and document.get('actual_solved_states') == 61, 'closed reduced61 qualification required')
        relative, absolute = score(document['dense_bow_interpolant'], 1081)
        states = 61
    else:
        accepted = {
            'yap.geometry_dense_prefix_assessment.v1': 'CLOSED_P2_DENSE_PHASE_PREFIX_SCORED',
            'yap.bi_pitch_family_dense_prefix_assessment.v1': 'CLOSED_P3_COARSE45_DENSE_PHASE_PREFIX_SCORED',
            'yap.bi_pitch_dense_prefix_assessment.v1': 'CLOSED_DENSE_PREFIX_BOW_ASSESSMENT',
        }
        require(schema in accepted and document.get('state') == accepted[schema]
                and document.get('original_end') == 1080 and document.get('dense_solved_states') == 1081
                and document.get('full_dense_history_terminal_failure_present') is not True,
                'closed complete dense-prefix assessment required')
        combined = document['combined_ROM61_vs_dense_native']
        relative, absolute = score(combined['bow_only_interpolant'], 1081)
        score(combined['matched_solved_states'], 61)
        states = 61
    require(document.get('MAPDL_accuracy_claim') is not True,
            'native-FE admission cannot claim direct MAPDL validation')
    return dict(cycles=3, solved_states=states, bow_observations=1081,
                max_relative_pct=relative, max_absolute_um=absolute)


def validate_bow_evidence(document):
    """Validate a supported three-cycle bow receipt, raising ValueError if bad."""
    try:
        return _validate_bow_evidence(document)
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError('incomplete or malformed bow evidence: ' + str(exc)) from exc


def _assessment(reader, receipt, pitch, si):
    report, report_pin = reader.read(receipt)
    metrics = validate_bow_evidence(report)
    schema = report.get('schema')
    if schema == 'yap.h7_graded73.sealed_prefix_post.v1':
        require(si == 7 and pitch == 2, 'H7 audit cannot admit another geometry')
        # The audit pins the terminal payload and construction/reference readset,
        # not the later closed history report. Bind that report through both.
        terminal_pin = report['candidate_q6']
        history_path = Path(terminal_pin['path']).parent / 'REPORT.json'
        history, history_pin = reader.read(history_path)
        same_binding(_terminal(history)['payload'], terminal_pin)
        for field in ('operator', 'reference'):
            pin = history['context'][field]
            require(report['readset'].get(pin['path']) == pin['sha256'],
                    'H7 audit does not bind selected candidate variant')
    elif schema == 'yap.h77_closed_temporal.independent_review.v1':
        require(si == 77 and pitch == 2, 'H77 audit cannot admit another geometry')
        candidates = [(name, pin) for name, pin in report['source_report_bindings'].items()
                      if name.startswith('COMPONENT_EXTENSION_PREDICT_h77_') and name.endswith('/REPORT.json')]
        require(len(candidates) == 1, 'one exact audited H77 candidate required')
        name, pin = candidates[0]
        history, history_pin = reader.read(dict(pin, path=str(Path(report['canonical_runtime_root']) / name)))
    elif schema == 'yap.graded_history.sealed_dense_bow_assessment.v1':
        history, history_pin = reader.read(report['provenance']['candidate_report'])
    elif schema == 'yap.component_reduced_clock.accuracy_profile.v1':
        history, history_pin = reader.read(report['lineage']['history'])
    else:
        history, history_pin = reader.read(report['ROM61'])
        terminal_pin = report.get('ROM61_prefix', report.get('ROM61_checkpoint'))
        require(terminal_pin is not None, 'assessment must bind candidate terminal payload')
        same_binding(_terminal(history)['payload'], terminal_pin)
    return (report_pin, history, history_pin, metrics['solved_states'],
            metrics['max_relative_pct'], metrics['max_absolute_um'])


def _geometry_reference(reader, reference, pitch, si, case):
    require(reference.get('case') in (case, 'extended_si' + f'{si:g}'.replace('.', 'p')),
            'reference case/geometry differs from selected history')
    if reference.get('schema') == 'yap.parameterized_si_reference.v1':
        require(reference.get('state') == 'CLOSED_PARAMETERIZED_SI_REFERENCE_SHARED_BI_BEOL_NOT_ADMITTED'
                and pitch == 2 and reference.get('Si_um') == si
                and reference.get('single_BI_total_um') == si + 11.5,
                'parameterized reference geometry differs')
        contract = reference.get('geometry_contract')
        if contract is not None:
            require(contract.get('BI_pitch_um') == pitch and contract.get('BI_diameter_pitch_ratio') == .5
                    and contract.get('common_face_ports') == 64 and contract.get('component_lifts') == 134
                    and contract.get('cell_xz_um') == [6, 6], 'exact 134/64 reference geometry required')
        else:
            require(si == 37 and reference.get('component_macro_map_shape') == [134, 262]
                    and reference.get('eliminated_macro_map_shape') == [134, 259],
                    'only the original Si37 reference uses the inherited geometry contract')
            seed, _ = reader.read(reference['recipe'])
            _geometry_reference(reader, seed, 2, 48.5, 'h48p5')
            target, _ = reader.read(reference['target_report'])
            proof = target.get('geometry_extension_proof', {})
            require(target.get('state') == 'CLOSED_NATIVE_FULL_INTERFACE_ELASTIC_REFERENCE_UNSCORED'
                    and target.get('Si_um') == si and proof.get('fixed_BI_BEOL_and_xz_unchanged') is True
                    and proof.get('cell_xz_um') == [6, 6] and proof.get('old_K_scaling') is False,
                    'inherited Si37 native geometry proof differs')
            same_binding(target['artifact'], reference['reference'])
    else:
        require(reference.get('schema') == 'yap.native_component_reference.v1'
                and reference.get('state') == 'CLOSED_NATIVE_COMPONENT_REFERENCE_NOT_REDUCED_OR_ADMITTED',
                'closed native component reference required')
        heights = {'bi': 1.5, 'beol': 10., 'si': si}
        for role, thickness in heights.items():
            certificate = reference['component_certificates'][role]
            require(certificate.get('macro_size') == 134
                    and math.isclose(certificate['height_mm'] * 1000, thickness, rel_tol=0, abs_tol=1e-9),
                    'reference physical role height or coordinate count differs')
        registration = reference['native_registration']
        require(registration.get('all259_trace_columns_checked') is True
                and registration.get('every_full_interior_column_checked') is True
                and registration.get('raw_B_fitting') is False, 'full physical native registration required')


def _components(reader, history, pitch, si, states):
    context = history.get('context', {})
    case = 'h' + f'{si:g}'.replace('.', 'p') if pitch == 2 else (
        'bi3_d1p5_si37_cell6_native45_voxel' if pitch == 3 else 'bi6_d3_si37_cell6_native45')
    require(history.get('schema') == 'yap.component_reduced_clock.history.v1'
            and history.get('state') == 'CLOSED_REDUCED_CLOCK_VIRGIN_HISTORY_UNSCORED'
            and history.get('case') == case and history.get('states') == states
            and history.get('final_original_ordinal') == 1080
            and context.get('global_macro_port_coordinates') == 259
            and context.get('Cu_source') == CU_SHA and context.get('acceleration_source') == ACCEL_SHA
            and context.get('all_GP_retained') is True and context.get('Cu_state_adoption') is False
            and context.get('virgin_birth_fields_zero') is True, 'closed exact virgin corrected-material history required')
    ordinals = context.get('selected_original_ordinals', [])
    require(len(ordinals) == states and ordinals[0] == 0 and ordinals[-1] == 1080
            and all(type(n) is int for n in ordinals) and ordinals == sorted(set(ordinals)),
            'actual selected thermal clock differs')
    _terminal(history)
    construction, construction_pin = reader.read(context['operator'])
    reference, reference_pin = reader.read(context['reference'])
    require(construction.get('schema') == 'yap.component_cu_riesz_operator.v1'
            and construction.get('state') == 'CLOSED_COMMON_COMPONENT_CU_RIESZ_OPERATORS_UNSCORED'
            and construction.get('case') == case and set(construction.get('components', {})) == set(ROLES),
            'closed selected component construction required')
    same_binding(construction['reference'], context['reference'])
    same_binding(context['topology'], reference['topology_artifact'])
    _geometry_reference(reader, reference, pitch, si, case)
    if construction.get('registration') is not None:
        registered, _ = reader.read(construction['registration'])
        require(registered.get('schema') == 'yap.component_extension.registration.v1'
                and registered.get('state') == 'CLOSED_ALL259_NATIVE_REGISTRATION_CERTIFIED'
                and registered.get('case') == case
                and registered.get('registration', {}).get('all259_trace_columns_checked') is True,
                'actual construction registration must cover all259')
        same_binding(registered['reference'], context['reference'])
    result = {}
    for role in ROLES:
        component = construction['components'][role]
        artifact = component['artifact']
        same_binding(artifact, context['operator_payloads'][role])
        rank = component['micro_rank']
        require(type(rank) is int and 1 <= rank <= 48 and context['micro_ranks'][role] == rank,
                'selected component micro rank differs')
        require(type(artifact.get('bytes')) is int and 0 < artifact['bytes'] <= 2 * 1024**3,
                'bounded component artifact size required')
        require(reader.path(artifact['path']).suffix == '.npz', 'exact NPZ artifact path required')
        regional = component.get('pooled_reference_identities', {}).get(reference['case'], component.get('reference'))
        same_binding(regional, reference['artifacts'][role])
        certificate = component.get('projection_certificate', {})
        require(certificate.get('macro_coordinates_retained') == 134
                and certificate.get('nonlinear_law_compatibility_by_identical_strain_and_virtual_work') is True,
                'selected reduced component virtual-work certificate required')
        for name in ('all_GP_strain', 'all_macro_point_stress_virtual_work',
                     'arbitrary_point_stress_body_virtual_work'):
            require(isinstance(certificate.get(name), dict)
                    and finite_nonnegative(certificate[name]['relative_error']) <= 2e-8,
                    'component projection/virtual-work certificate failed')
        result[role] = dict(binding(artifact), bytes=artifact['bytes'], micro_rank=rank)
    return result, construction_pin, reference_pin


def build_catalog(progress_path=None, *, root=ROOT):
    """Return the 13-entry catalog; create nothing and read no NPZ payloads."""
    reader = MetadataReader(root)
    progress, _ = reader.read(progress_path or Path(root) / PROGRESS_RELATIVE)
    coverage = progress['audited_current_summary_20261021']['coverage']
    locators = coverage['single_BI_p2_full_three_cycle_passed']
    requested = []
    for si in P2_SI:
        matches = [row for row in locators if row.get('Si_TSV_um') == si]
        require(len(matches) == 1, 'one progress locator per exact P2 Si thickness required')
        requested.append((2, si, matches[0]['evidence']))
    for pitch in (3, 6):
        matches = [row for row in coverage['other_pitches'] if row.get('pitch_um') == pitch]
        require(len(matches) == 1, 'one progress locator per exact BI pitch required')
        requested.append((pitch, 37, matches[0]['evidence']))
    entries = []
    for pitch, si, receipt in requested:
        try:
            report_pin, history, history_pin, states, relative, absolute = _assessment(reader, receipt, pitch, si)
            components, construction, reference = _components(reader, history, pitch, si, states)
            name = f'BI{pitch}_SI{si:g}'.replace('.', 'p') + ('_COARSE45' if pitch == 3 else '')
            entries.append(dict(id=name, geometry=dict(bi_pitch_um=pitch, diameter_pitch_ratio=.5,
                bi_thickness_um=1.5, beol_thickness_um=10, si_tsv_thickness_um=si,
                total_thickness_um=si + 11.5, bi_layer_count=1, tsv_pitch_um=6, cell_side_um=6),
                components=components, construction=construction, reference=reference,
                validation=dict(report=report_pin, history=history_pin, cycles=3, solved_states=states,
                    bow_observations=1081, max_relative_pct=relative, max_absolute_um=absolute,
                    mesh_scope=('coarse45_native_FE_full_time; fine90_time_pending' if pitch == 3
                                else 'native45_FE_single_BI_10x10mm_bow_projection'),
                    reference_kind='native_FE', MAPDL_direct_validation=False),
                capabilities=dict(single_bi_history=True, experimental_prepared_d2w=True,
                                  prepared_assembly_accuracy=False, W2W_thinning=False)))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'BI{pitch} Si{si:g}: {exc}') from exc
    reader.verify_unchanged()
    return dict(schema=SCHEMA, entries=entries)


def publish_catalog(output, *, progress_path=None, root=ROOT):
    """Create a fresh, read-only catalog only at an explicit distribution path."""
    catalog = build_catalog(progress_path, root=root)
    reader = MetadataReader(root)
    path = reader.path(output)
    require(path.is_relative_to(Path(root) / 'distribution') and path.name == 'CATALOG.json',
            'publication requires an explicit canonical distribution CATALOG.json path')
    payload = (json.dumps(catalog, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('ascii')
    require(len(payload) <= JSON_LIMIT, 'catalog descriptor exceeds 1MiB')
    # A colleague delivery may already contain native runtime files. Only create
    # the catalog, and never chmod existing directories or overwrite publication.
    require(path.parent.parent.is_dir(), 'existing distribution root required')
    require(path.parent.parent.resolve(strict=True) == path.parent.parent,
            'distribution parent must not contain symlinks')
    if not path.parent.exists():
        path.parent.mkdir(mode=0o755)
        path.parent.chmod(0o755)
    require(path.parent.is_dir() and path.parent.resolve(strict=True) == path.parent,
            'literal nonsymlink delivery directory required')
    with _exclusive_output(path) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o444)
    return dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(), entries=len(catalog['entries']))


def _exclusive_output(path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444), 'wb')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--progress', type=Path, default=ROOT / PROGRESS_RELATIVE)
    parser.add_argument('--output', type=Path, help='explicit fresh distribution CATALOG.json; otherwise print only')
    options = parser.parse_args(argv)
    try:
        result = (publish_catalog(options.output, progress_path=options.progress) if options.output
                  else build_catalog(options.progress))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print('Component metadata publication refused: ' + str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
