"""Publish exact component metadata without reading numerical operator payloads.

Progress is a locator, not an admission receipt. Each selected assessment is
checked against its closed history, actual construction and physical reference.
Experimental prepared-D2W capability still requires registration at runtime;
this publisher does not certify prepared assembly or wafer thinning.
"""

import argparse
import hashlib
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import stat
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mechanics_runtime import shared_distribution_path


ROOT = Path('/app/nanocad/projects/personal/zchen/YAP_STACK')
PROGRESS_RELATIVE = 'operator_night_20261021/OUTER3_FORMAL_PROGRESS_20261022_v1.json'
CATALOG_RELATIVE = 'distribution/operators/components/CATALOG.json'
SCHEMA = 'yap.component_operator_catalog.v2'
JSON_LIMIT = 1024**2
READ_LIMIT = 64 * JSON_LIMIT
ROLES = ('bi', 'beol', 'si')
PROBES = {'corner_pm_um', 'corner_pp_um', 'edge_x_um', 'edge_y_um'}
CU_SHA = '35e7016dab8b1fb2cb0a928e7642665d3fc4f00b4f255eb797fb6d6e44b56722'
ACCEL_SHA = '483c9d948931d48beb73561842cb8fe780db285d3b61f158e580229fbdcd465f'
P2_SI = (7, 17, 27, 37, 47, 48.5, 57, 67, 77, 87, 88.5)
P2_EXTERIOR_SI = (8.5, 18.5, 38.5)
PITCH_THICKNESS_CASES = {
    (pitch, si): f'bi{pitch}_d' + ('1p5' if pitch == 3 else '3')
    + f'_si{si}_cell6_native45' + ('_voxel' if pitch == 3 else '')
    for pitch in (3, 6) for si in (17, 77)
}
ENDPOINT_CASES = {
    (pitch, si): f'bi{pitch}_d' + ('1p5' if pitch == 3 else '3')
    + f'_si{si}_cell6_native45' + ('_voxel' if pitch == 3 else '')
    for pitch in (3, 6) for si in (7, 87)
}
PITCH_THICKNESS_CASES.update(ENDPOINT_CASES)
MIDDLE_CASES = {
    (pitch, si): f'bi{pitch}_d' + ('1p5' if pitch == 3 else '3')
    + f'_si{si}_cell6_native45' + ('_voxel' if pitch == 3 else '')
    for pitch in (3, 6) for si in (27, 47, 57, 67)
}
PITCH_THICKNESS_CASES.update(MIDDLE_CASES)
MIDDLE_ROOT = ROOT / 'operator_night_20261021'
MIDDLE_REFERENCE_SCHEMA = 'yap.pitch_thickness_middle_component_reuse.reference.v1'
MIDDLE_PINS = {
    'pitch_thickness_middle_bow_assessment_20261022.py': '1116071c8abd18da75f02b6f08f49d16c0eaf140cbef712a9e81ac0be22d30a2',
    'test_pitch_thickness_middle_bow_assessment_20261022.py': '27b3831ea25e6d9449d61ef24e78a64b7b19f6207543c6705d58d69c1618c47a',
    'pitch_thickness_middle_rom_history_20261022.py': 'dde93d65a042483c0b8ab3efa35f9d63651732238f224332fbc2933b09df9153',
    'test_pitch_thickness_middle_rom_history_20261022.py': '5e427eb4315512588f3e546c78dcabc0ddb335cbc8e95b60055c087f45944a97',
    'pitch_thickness_middle_native_dense_20261022.py': '997f23e1d063b163b942b215ae6d8278f8849b249e3a7b592e7a8a9ee76dc4b5',
    'test_pitch_thickness_middle_native_dense_20261022.py': 'c1ce27821ba6e42e19eb891c256418fe45f937f19df5279c9e7c1b9e77771c8d',
    'pitch_thickness_middle_operator_composition_20261022.py': '6a8180c6817cc869efd0008e9f6249ca1a06160ed5af59104ee715c2aa8c294d',
    'test_pitch_thickness_middle_operator_composition_20261022.py': 'c6537431369c97de1e3e2bc96a2c086d196426ab5c4b804f6316343f83026bd4',
    'pitch_thickness_middle_component_reuse_20261022.py': '3ad7f3f6de2f8a9377a02b0e52753d81f5ef3dd7fe0ee5937062bf39602263ab',
    'test_pitch_thickness_middle_component_reuse_20261022.py': 'ed492b627aa0ae0b5f5bf8f4e226082fbc467942c5332b025b586074e8ed02b1',
    'pitch_thickness_middle_cases_20261022.py': '5e18c626b4eccfb47edb73f3d1a4b65dd5c5c77b929f3dbf630ac75547810c1d',
    'pitch_thickness_middle_reference_20261022.py': 'd480cc61e241eada260120b5f9b4b6930a91c9633075de2fcf873e5cd7961601',
    'test_pitch_thickness_middle_reference_20261022.py': 'a726c6cf6b830d23196c33e76728a2c9d3babddc34b0bbeae41e499a20bbf961',
}
MIDDLE_ASSESSOR_PAIR = tuple(MIDDLE_PINS[name] for name in (
    'pitch_thickness_middle_bow_assessment_20261022.py',
    'test_pitch_thickness_middle_bow_assessment_20261022.py'))
ENDPOINT_PINS = {
    'pitch_thickness_endpoint_bow_assessment_20261022.py': '81b08ba98a76f47ae29c508d1b2c5bbfff89d7a9f3e36fbd0af7427e54e1b25b',
    'test_pitch_thickness_endpoint_bow_assessment_20261022.py': '8b12ca633e97c26ab273420e846ff350df99f65c306089e83c3c560d4b97c3d3',
    'pitch_thickness_endpoint_rom_history_20261022.py': '731dc4df1e4d8a596c1d460b858a5f1c4b2767428846ff6dcae5e31a3b4e2315',
    'test_pitch_thickness_endpoint_rom_history_20261022.py': '085b77062df8a537a36042ed2373092cf7fecd144ca49c99d270291d43484391',
    'pitch_thickness_endpoint_native_dense_20261022.py': '81bc5b4e2cf5ac2bd2ededd8fded1142d6f64bb677acf4f8650310fea12841fe',
    'test_pitch_thickness_endpoint_native_dense_20261022.py': 'd3c20e95f5c6d4fb77fd7ee00ec118ecf308c08b9ef78715b928b87a83cca425',
    'pitch_thickness_endpoint_operator_composition_20261022.py': '426ec48d573e9a6a05b41968cc438633bf155472479384ff2fbd2c8e125c53bf',
    'test_pitch_thickness_endpoint_operator_composition_20261022.py': 'cdb61e00c02ca7bc8c361cfb2ca2fbaaa99ff691a98829ede4f7bd717ebbb380',
    'pitch_thickness_endpoint_component_reuse_20261022.py': '2ff4823f38ea5034b7686e601ac28973640c42081d41e15144350d52ad38110c',
    'test_pitch_thickness_endpoint_component_reuse_20261022.py': '28ba6f41c0947ce05d8362a51702ed2169609e02a298f7239d2ca5ff841497b5',
    'pitch_thickness_case_manifest_20261022.py': '03352baf2188903ccfcf192a6c2f70d2a24f4aa10dbc209d2e3da26540c55b3d',
}
ENDPOINT_ASSESSOR_PAIR = tuple(ENDPOINT_PINS[name] for name in (
    'pitch_thickness_endpoint_bow_assessment_20261022.py',
    'test_pitch_thickness_endpoint_bow_assessment_20261022.py'))
ENDPOINT_REFERENCE_SCHEMA = 'yap.pitch_thickness_endpoint_component_reuse.reference.v1'
ENDPOINT_IMPLEMENTATION_PINS = {
    'component_fullspace_validation.py': 'b768cafe3297f3bf341539c51861725aaa3616a8e7bf7fc30f7f9ea023b30005',
    'pitch_thickness_native_dense_20261022.py': 'db9b1e57dbd0cb11007203fd497f945699c0277e550c4091da5f98e3e52af55f',
    'test_pitch_thickness_native_dense_20261022.py': '2815a303c18b45c4579aea38444c57e0e2889dc9dc8b491b0a50ceda5b9de747',
    'pitch_thickness_reduced_history_20261022.py': '9116aebb110cc2e0b405fc7fc626224ec68368dc21797d0d3d9b38b19531ffd1',
    'pitch_thickness_reduced_profiles_20261022.py': '70729ae675420031848b6af875d92bad7887d5c4adb0aa028d692b193cc1850e',
}
GRADED73_ORDINALS = [0, *(360*c+i for c in range(3) for i in (
    *range(20, 161, 20), *(160+j for j in (1, 2, 4, 8, 10, 20, 30, 40)), *range(220, 361, 20)))]
PHASE_ORDINALS = [0, 160, 200, 360, 520, 560, 720, 880, 920, 1080]
PITCH_THICKNESS_ASSESSOR_SHA = '55672d39198128ffc38244676d16a16b7afaf34494e07803b333045fad95d053'
PITCH_THICKNESS_ASSESSOR_TEST_SHA = 'b21d845d159595692eaccbfc87b351fd93a2cc9e7a7b86d2c92480d8dacebf21'
PITCH_THICKNESS_PROFILE_ASSESSOR_SHA = '42e12eae16024533029ab64e79c91fe52ecf31cb5e7dd20c1dbb015d75bd264e'
PITCH_THICKNESS_PROFILE_ASSESSOR_TEST_SHA = '695ff5dff9a796f42f06397d6c96d817d3e0382653bb954314320e178d4d95b5'
PITCH_THICKNESS_CSR_SHA = '0997ae8772a848a0894704dd69f5894a54ec5a68a78a2d32d0bb15a7eafa9f04'
LEGACY_EXTERIOR_ROOT = ROOT / (
    'operator_repair_20261015/development/trial_forcing_closure_v1/continuous_cu_screen_v1/runtime')
LEGACY_EXTERIOR_PINS = {
    'reference': ('PARAMETERIZED_SI_REFERENCE_h38p5_20261019_v1',
                  '6a2c538defc91d2faee3984d1ddb95ec87ebf6e5f911ec07778ae3ba38010419'),
    'recipe': ('COMPONENT_GALERKIN_REFERENCE_h48p5_20261019_v1',
               '4bc871d3eac410a6f53f0f32f7c00718e1f329815c82c0061fd23b2e32b9c1e3'),
    'target_report': ('NATIVE_GEOMETRY_EXTENSION_h38p5_20261019_v2',
                      'a4e4e48dcb72416c349bebaa0c8191c8c8b69e9b1a5a2d40d9f9d3117f376ff7'),
    'registration': ('COMPONENT_EXTENSION_REGISTER_h38p5_20261019_v3',
                     '0ce4f606830945cc36e5ab0759961d16d0a57b506a027b95c3efb9d730e2b2c3'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding(value):
    require(isinstance(value, dict) and isinstance(value.get('path'), str)
            and re.fullmatch(r'[0-9a-f]{64}', str(value.get('sha256'))),
            'exact path/SHA256 binding required')
    return dict(path=str(shared_distribution_path(value['path'])), sha256=value['sha256'])


def same_binding(left, right):
    require(binding(left) == binding(right)
            and ('bytes' not in left or 'bytes' not in right or left['bytes'] == right['bytes']),
            'selected history/construction/reference artifact variant differs')


def _endpoint_source(name):
    return dict(path=str(LEGACY_EXTERIOR_ROOT.parent / name), sha256=ENDPOINT_PINS[name])


def _endpoint_pair(document, stem):
    require((document.get('source_sha256'), document.get('tests_sha256')) == (
        ENDPOINT_PINS[stem + '.py'], ENDPOINT_PINS['test_' + stem + '.py']),
        'exact reviewed endpoint source/test pair required: ' + stem)


def _middle_source(name):
    return dict(path=str(MIDDLE_ROOT / name), sha256=MIDDLE_PINS[name])


def _middle_pair(document, stem):
    require((document.get('source_sha256'), document.get('tests_sha256')) == (
        MIDDLE_PINS[stem + '.py'], MIDDLE_PINS['test_' + stem + '.py']),
        'exact reviewed middle source/test pair required: ' + stem)


def _middle_N_sources(readset):
    require(type(readset) is dict, 'middle source readset must be an object')
    for path, pin in readset.items():
        require(type(path) is str, 'literal middle source readset path required')
        source = Path(path)
        if source.is_relative_to(MIDDLE_ROOT) and source.suffix != '.json':
            require(source.name in MIDDLE_PINS
                    and path == str(MIDDLE_ROOT / source.name)
                    and pin == MIDDLE_PINS[source.name],
                    'unreviewed or mispinned N source in middle readset: ' + path)


def _middle_bow_lineage(report):
    lineage = report['lineage']
    _endpoint_bow_lineage(report, middle=True)
    require(lineage.get('middle_ROM_entry') == _middle_source('pitch_thickness_middle_rom_history_20261022.py')
            and lineage.get('middle_ROM_tests') == _middle_source('test_pitch_thickness_middle_rom_history_20261022.py')
            and lineage.get('endpoint_assessment_implementation') == dict(
                path=str(LEGACY_EXTERIOR_ROOT.parent / 'pitch_thickness_endpoint_bow_assessment_20261022.py'),
                sha256=ENDPOINT_ASSESSOR_PAIR[0], tests_sha256=ENDPOINT_ASSESSOR_PAIR[1], execution_entry=False),
            'middle assessment must bind actual entries and genuine endpoint scorer ancestor')
    proof = lineage.get('middle_authentication_AST', {})
    require(proof.get('inverse_full_module_exact') is True
            and proof.get('changed_expressions') == 2 and proof.get('receipt_keyword_edits') == 2,
            'reviewed middle metadata-only authentication AST required')
    additional = lineage.get('additional_source_pins', {})
    require(all(additional.get(name) == pin for name, pin in MIDDLE_PINS.items()
                if name not in ('pitch_thickness_middle_bow_assessment_20261022.py',
                                'test_pitch_thickness_middle_bow_assessment_20261022.py')),
            'complete exact middle source/test/manifest closure required')


def _endpoint_bow_lineage(report, *, middle=False):
    lineage = report['lineage']
    family = 'middle' if middle else 'endpoint'
    source = _middle_source if middle else _endpoint_source
    require(lineage.get('genuine_assessment_source') == dict(
                path=str(LEGACY_EXTERIOR_ROOT.parent / 'pitch_thickness_bow_assessment_20261022.py'),
                sha256=PITCH_THICKNESS_ASSESSOR_SHA)
            and lineage.get('genuine_assessment_tests') == dict(
                path=str(LEGACY_EXTERIOR_ROOT.parent / 'test_pitch_thickness_bow_assessment_20261022.py'),
                sha256=PITCH_THICKNESS_ASSESSOR_TEST_SHA)
            and lineage.get(family + '_ROM_entry') == source('pitch_thickness_' + family + '_rom_history_20261022.py')
            and lineage.get(family + '_ROM_tests') == source('test_pitch_thickness_' + family + '_rom_history_20261022.py'),
            'endpoint assessment must bind genuine scorer and actual endpoint ROM')
    certificate = lineage.get('producer_authentication_AST', {})
    require(certificate.get('inverse_full_module_exact') is True
            and certificate.get('changed_expressions') == 10
            and certificate.get('scoring_q6_decoder_clock_pairing_state_chain_JVP_and_publication_bodies_unchanged') is True
            and lineage.get('operator_contract_AST', {}).get('inverse_full_module_exact') is True,
            'endpoint producer authentication must preserve scoring/clock/state/JVP bodies')
    additional = lineage.get('additional_source_pins', {})
    require(all(additional.get(name) == pin for name, pin in ENDPOINT_PINS.items()
                if name not in ('pitch_thickness_endpoint_bow_assessment_20261022.py',
                                'test_pitch_thickness_endpoint_bow_assessment_20261022.py')),
            'endpoint assessment source ancestry is incomplete')
    require(report.get('full_span_x_mm') == report.get('full_span_y_mm') == 10.
            and report.get('bow_units') == 'um' and set(report.get('probe_names', [])) == PROBES
            and report.get('selected_original_ordinals') == GRADED73_ORDINALS
            and report.get('dense_1081_ROM_solved_history_claim') is False
            and type(report.get('ROM_strict_linear_certificates')) is int
            and report['ROM_strict_linear_certificates'] > 0,
            'exact endpoint 10mm/four-probe graded73 original clock/JVP scope required')


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
        relocated = shared_distribution_path(path)
        require(relocated.is_relative_to(self.root), 'relocated evidence escapes reader root')
        return relocated

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
    elif schema == 'yap.pitch_thickness_bow_assessment.v1':
        require(document.get('state') == 'CLOSED_FULL_GRADED73_BOW_PROFILE_ASSESSMENT'
                and document.get('through_original_ordinal') == 1080
                and document.get('native_prefix_states') == 1081
                and document.get('actual_ROM_solved_states_compared') == 73
                and document.get('total_ROM_solved_states') == 73
                and document.get('prefix_only') is False
                and document.get('full_native1081_closure_verified') is True
                and document.get('qualification_passed') is True
                and document.get('global_accuracy_passed') is True
                and document.get('ROM_accepted_state_chain_verified') is True,
                'complete independent pitch/thickness three-cycle bow pass required')
        lineage = document['lineage']
        pair = (lineage.get('source_sha256'), lineage.get('tests_sha256'))
        require(pair in ((PITCH_THICKNESS_ASSESSOR_SHA, PITCH_THICKNESS_ASSESSOR_TEST_SHA),
                         (PITCH_THICKNESS_PROFILE_ASSESSOR_SHA, PITCH_THICKNESS_PROFILE_ASSESSOR_TEST_SHA),
                         ENDPOINT_ASSESSOR_PAIR, MIDDLE_ASSESSOR_PAIR)
                and lineage.get('through_original_ordinal') == 1080
                and lineage.get('native_report') is not None
                and document.get('Cu_arrays_decoded') is False
                and document.get('Cu_interpolation') is False,
                'authenticated independent bow producer and full native closure required')
        if pair[0] == PITCH_THICKNESS_PROFILE_ASSESSOR_SHA:
            require(lineage.get('genuine_assessment_source') == dict(
                        path=str(LEGACY_EXTERIOR_ROOT.parent / 'pitch_thickness_bow_assessment_20261022.py'),
                        sha256=PITCH_THICKNESS_ASSESSOR_SHA)
                    and lineage.get('genuine_assessment_tests') == dict(
                        path=str(LEGACY_EXTERIOR_ROOT.parent / 'test_pitch_thickness_bow_assessment_20261022.py'),
                        sha256=PITCH_THICKNESS_ASSESSOR_TEST_SHA)
                    and lineage.get('producer_authentication_AST', {}).get('inverse_full_module_exact') is True
                    and lineage.get('producer_authentication_AST', {}).get(
                        'scoring_decoding_pairing_Cu_JVP_clock_and_gate_unchanged') is True,
                    'profile assessment must preserve genuine scoring and authenticate producer ancestry')
        if pair == ENDPOINT_ASSESSOR_PAIR:
            _endpoint_bow_lineage(document)
        if pair == MIDDLE_ASSESSOR_PAIR:
            _middle_bow_lineage(document)
        score(document['solved_nodes'], 73)
        relative, absolute = score(document['dense_bow_interpolant'], 1081)
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


def _semantic_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('ascii')).hexdigest()


def _endpoint_configuration(reader, lineage, descriptor, filename, transient):
    path = str(Path(descriptor['path']).parent / filename)
    pin = lineage.get('metadata_inputs', {}).get(path)
    require(pin is not None and pin.get('path') == path,
            'endpoint assessor must authenticate sibling configuration: ' + path)
    document, _ = reader.read(pin)
    return {k: v for k, v in document.items() if k not in transient}


def _endpoint_histories(reader, report, history, native, reference, pitch, si, *, middle=False):
    lineage, c, n = report['lineage'], history['context'], native['context']
    family = 'middle' if middle else 'endpoint'
    cases = MIDDLE_CASES if middle else ENDPOINT_CASES
    source = _middle_source if middle else _endpoint_source
    pair = _middle_pair if middle else _endpoint_pair
    pins = MIDDLE_PINS if middle else ENDPOINT_PINS
    key, case = f'p{pitch}_si{si:g}', cases[pitch, si]
    first = 18816 if pitch == 3 else 19248
    regions = [['bi', 0, first], ['beol', first, first + 102096], ['si', first + 102096, first + 166256]]
    pair(history, 'pitch_thickness_' + family + '_rom_history_20261022')
    require(history.get('key') == key and history.get('case') == c.get('case') == n.get('case') == case
            and history.get('initial_ordinal') == 0 and history.get('final_ordinal') == 72
            and history.get('MAPDL_result_reads') == history.get('external_checkpoint_reads') == 0
            and history.get('no_MAPDL_state_adopted') is True and len(history.get('steps', [])) == 72
            and c.get('selected_original_ordinals') == GRADED73_ORDINALS
            and [v.get('original_ordinal') for v in history.get('checkpoints', [])] == PHASE_ORDINALS
            and history.get('certified_linear_solves') == report['ROM_strict_linear_certificates'],
            'complete original-clock endpoint virgin73 history and exact JVP census required')
    require(n.get('selected_original_ordinals') == list(range(1081))
            and n.get('clock_states') == 1081 and n.get('temporal_profile') == 'dense_original1081'
            and native.get('Cu_points') == c.get('Cu_GP') == regions[-1][-1]
            and native.get('interior_coordinates') == 273372 and n.get('GP_regions') == regions
            and [v.get('ordinal') for v in native.get('checkpoints', [])] == PHASE_ORDINALS
            and n.get('MAPDL_reads') == 0 and n.get('component_reference_used') is False
            and n.get('component_certificate_dependency') is False and n.get('ROM_dependency') is False,
            'complete independent endpoint native1081 clock/GP/quotient required')
    require(native.get('source_sha256') == ENDPOINT_IMPLEMENTATION_PINS['component_fullspace_validation.py']
            and n.get('history_wrapper_sha256') == ENDPOINT_IMPLEMENTATION_PINS['pitch_thickness_native_dense_20261022.py']
            and n.get('history_tests_sha256') == ENDPOINT_IMPLEMENTATION_PINS['test_pitch_thickness_native_dense_20261022.py'],
            'genuine native numerical engine and original wrapper identities must remain distinct')
    for context in (c, n):
        require(context.get('Cu_source') == CU_SHA and context.get('acceleration_source') == ACCEL_SHA
                and context.get('thermal_policy') == c.get('thermal_policy')
                and context.get('virgin_birth_fields_zero') is True and context.get('Cu_state_adoption') is False,
                'same corrected material/thermal policy and own virgin endpoint state required')
    require(c.get('thermal_policy') is not None and c.get('external_checkpoint_reads') == 0
            and c.get('all_GP_retained') is True and c.get('global_macro_port_coordinates') == 259,
            'full-GP virgin endpoint ROM scope required')
    for row in native['checkpoints']:
        require(row.get('context') == n and row.get('all_Cu14') is True
                and row.get('no_native_state_adopted') is True, 'complete own-state native checkpoint chain required')
    for row in history['checkpoints']:
        require(row.get('context') == c and row.get('all_Cu14') is True
                and row.get('no_MAPDL_state_adopted') is True, 'complete own-state ROM checkpoint chain required')
    for label, document, context, filename, transient, stem, schema, digest_name in (
        ('ROM', history, history.get(family + '_rom_entry', {}), 'CONFIGURATION.json',
         ('main_go', 'actual_resource_selection'), 'pitch_thickness_' + family + '_rom_history_20261022',
         'yap.pitch_thickness_' + family + '_rom_history.configuration.v1', 'configuration_sha256'),
        ('native', native, n.get(family + '_native_entry', {}), 'DENSE1081_CONFIGURATION.json',
         ('main_go', 'actual_core_selection'), 'pitch_thickness_' + family + '_native_dense_20261022',
         'yap.pitch_thickness_' + family + '_native_dense.configuration.v1', 'dense1081_configuration_sha256')):
        descriptor = lineage['rom' if label == 'ROM' else 'native_report']
        config = _endpoint_configuration(reader, lineage, descriptor, filename, transient)
        pair(config, stem)
        digest = _semantic_digest(config)
        require(config.get('schema') == schema and config.get('key') == key and config.get('case') == case
                and config.get('GP_regions') == regions and config.get('Cu_GP') == regions[-1][-1]
                and config.get('virgin_birth_fields_zero') is True and config.get('external_checkpoint_reads') == 0
                and config.get('thermal_policy') == c['thermal_policy']
                and config.get('readset_sha256') == _semantic_digest(config['readset'])
                and document.get(digest_name) == context.get('configuration_sha256') == digest
                and context.get('source') == source(stem + '.py')
                and context.get('tests') == source('test_' + stem + '.py'),
                'authenticated actual endpoint ' + label + ' execution configuration required')
        manifest = 'pitch_thickness_middle_cases_20261022.py' if middle else 'pitch_thickness_case_manifest_20261022.py'
        for name in (stem + '.py', 'test_' + stem + '.py', manifest):
            require(config['readset'].get(source(name)['path']) == pins[name],
                    'endpoint execution source closure differs: ' + name)
        if label == 'ROM':
            cap = config['limits']['address_space_bytes']
            require(type(cap) is int and cap in (16 * 1024**3, 32 * 1024**3)
                    and config['limits']['minimum_available_bytes'] == cap + 32 * 1024**3
                    and history.get('address_space_bytes') == cap
                    and history.get('AS_trial_profile') == config.get('AS_trial_profile') == f'AS{cap // 1024**3}'
                    and history.get('completed_history_under_executable_AS') is True,
                    'truthful completed endpoint AS16/32 trial required')
            for field, name in (('mathematical_wrapper', 'pitch_thickness_reduced_history_20261022.py'),
                                ('admission_implementation', 'pitch_thickness_reduced_profiles_20261022.py')):
                require(context.get(field) == dict(path=str(LEGACY_EXTERIOR_ROOT.parent/name),
                    sha256=ENDPOINT_IMPLEMENTATION_PINS[name]), 'genuine endpoint ROM implementation identity differs')
            require(context.get('public_ready') is False and context.get('native_history_read') is False,
                    'endpoint producer must not claim publication or teacher-state adoption')
            same_binding(config['operator'], c['operator'])
            same_binding(context['operator'], c['operator'])
        else:
            require(config.get('Cu_state_adoption') is False
                    and config['limits']['address_space_bytes'] in (32 * 1024**3, 64 * 1024**3),
                    'reviewed native AS32/64 profile required')
            same_binding(context['raw_native_report'], reference['native_report'])
            same_binding(config['native_report'], reference['native_report'])
            same_binding(config['native'], reference['reference'])
        if middle:
            _middle_N_sources(config['readset'])
            if label == 'ROM' and 'composer_readset' in config:
                _middle_N_sources(config['composer_readset'])
            names = ('pitch_thickness_middle_reference_20261022.py',
                     'test_pitch_thickness_middle_reference_20261022.py') if label == 'native' else (
                     'pitch_thickness_middle_operator_composition_20261022.py',
                     'test_pitch_thickness_middle_operator_composition_20261022.py',
                     'pitch_thickness_middle_component_reuse_20261022.py',
                     'test_pitch_thickness_middle_component_reuse_20261022.py')
            require(all(config['readset'].get(source(name)['path']) == pins[name] for name in names),
                    'middle actual raw/static/composition source closure differs')


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
    elif schema == 'yap.pitch_thickness_bow_assessment.v1':
        require((pitch, si) in PITCH_THICKNESS_CASES
                and report.get('case') == PITCH_THICKNESS_CASES[pitch, si]
                and report.get('key') == f'p{pitch}_si{si:g}',
                'exact reviewed pitch/thickness bow geometry required')
        endpoint = (pitch, si) in ENDPOINT_CASES
        middle = (pitch, si) in MIDDLE_CASES
        pair = (report['lineage']['source_sha256'], report['lineage']['tests_sha256'])
        require(pair == MIDDLE_ASSESSOR_PAIR if middle else (
                pair == ENDPOINT_ASSESSOR_PAIR if endpoint else
                pair not in (ENDPOINT_ASSESSOR_PAIR, MIDDLE_ASSESSOR_PAIR)),
                'selected geometry requires its reviewed endpoint or original bow producer')
        history, history_pin = reader.read(report['lineage']['rom'])
        native, _ = reader.read(report['lineage']['native_report'])
        checkpoint, _ = reader.read(report['lineage']['native_checkpoint'])
        require(native.get('state') == 'CLOSED_PITCH_THICKNESS_NATIVE_DENSE1081_UNSCORED'
                and native.get('states') == 1081 and native.get('case') == history.get('case')
                and native.get('context') == checkpoint.get('context')
                and native.get('checkpoints', [None])[-1] == checkpoint
                and checkpoint.get('ordinal') == 1080 and checkpoint.get('all_Cu14') is True,
                'selected native history must actually close all1081 states')
        reference, _ = reader.read(history['context']['reference'])
        same_binding(native['context']['native_elastic_report'], reference['native_report'])
        same_binding(native['context']['reference'], reference['reference'])
        same_binding(report['lineage']['rom'], history_pin)
        if endpoint or middle:
            _endpoint_histories(reader, report, history, native, reference, pitch, si, middle=middle)
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
    if reference.get('schema') in ('yap.pitch_thickness_component_reuse.reference.v2', ENDPOINT_REFERENCE_SCHEMA,
                                   MIDDLE_REFERENCE_SCHEMA):
        endpoint = (pitch, si) in ENDPOINT_CASES
        middle = (pitch, si) in MIDDLE_CASES
        source = _middle_source if middle else _endpoint_source
        family = 'middle' if middle else 'endpoint'
        require((reference['schema'] == ENDPOINT_REFERENCE_SCHEMA) is endpoint
                and (reference['schema'] == MIDDLE_REFERENCE_SCHEMA) is middle,
                'endpoint geometry requires its exact static producer schema')
        require((pitch, si) in PITCH_THICKNESS_CASES and case == PITCH_THICKNESS_CASES[pitch, si]
                and reference.get('state') == 'CLOSED_COMPOSED_REGIONAL_REFERENCE_NOT_REDUCED_OR_BOW_ADMITTED'
                and (reference.get('BI_pitch_um'), reference.get('Si_um')) == (pitch, si)
                and (endpoint or middle or reference.get('CSR_adapter_source_sha256') == PITCH_THICKNESS_CSR_SHA)
                and reference.get('old_K_B_scaling') is False
                and reference.get('Cu_state_adoption') is False,
                'closed exact native-equivalent composed geometry required')
        if endpoint or middle:
            require(reference.get('actual_entry') == source('pitch_thickness_' + family + '_component_reuse_20261022.py')
                    and reference.get('actual_tests') == source('test_pitch_thickness_' + family + '_component_reuse_20261022.py')
                    and reference.get('key') == f'p{pitch}_si{si:g}',
                    'genuine endpoint static producer/test/geometry required')
        registration, congruence = reference['native_registration'], reference['congruence']
        require(registration.get('all259_trace_columns_checked') is True
                and registration.get('every_full_interior_column_checked') is True
                and registration.get('rigid_gauge_only') is True
                and registration.get('raw_B_fitting') is False
                and congruence.get('all_columns') == 259 and congruence.get('fitting') is False,
                'complete unfitted native registration required')
        first = 18816 if pitch == 3 else 19248
        regions = [['bi', 0, first], ['beol', first, first + 102096],
                   ['si', first + 102096, first + 166256]]
        require(reference.get('GP_regions') == regions
                and reference.get('native_micro_coordinates') == 273372,
                'complete ordered composed native quotient required')
        for role, start, stop in regions:
            equivalence = reference['fresh_equivalence_certificates'][role]
            strain = congruence['all_GP_strain'][role]
            require(equivalence.get('exact_body_P_and_ordering') is True
                    and equivalence.get('GP_identity_and_material_constants_checked') is True
                    and equivalence.get('historical_harmonic_SPD_transfer_condition_satisfied') is True
                    and strain.get('all_trace_columns') == 259 and strain.get('Cu_points') == stop - start
                    and strain.get('fitted_to_GP_data') is False,
                    'fresh regional P/material/GP identity and historical transfer required')
            if endpoint or middle:
                local = equivalence.get('local_FE_input_equivalence', {})
                require(all(local.get(name) is True for name in (
                    'all_Cu_EN_and_eight_GP_order_equal', 'complete_Cartesian_connectivity_equal',
                    'complete_material_mask_equal')), 'complete endpoint local FE input equivalence required')
        target, _ = reader.read(reference['native_report'])
        require(target.get('state') == 'CLOSED_NATIVE_FULL_INTERFACE_ELASTIC_REFERENCE_UNSCORED'
                and (target.get('BI_pitch_um'), target.get('Si_um'), target.get('case')) == (pitch, si, case)
                and target.get('proof', {}).get('micro_coordinates') == 273372,
                'same target native elastic reference required')
        if middle:
            require(target.get('entry_source_sha256') == MIDDLE_PINS['pitch_thickness_middle_reference_20261022.py']
                    and target.get('entry_tests_sha256') == MIDDLE_PINS['test_pitch_thickness_middle_reference_20261022.py'],
                    'middle raw elastic reference must retain its actual producer identity')
        same_binding(target['artifact'], reference['reference'])
    elif reference.get('schema') == 'yap.parameterized_si_reference.v1':
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
            require(si in (37, 38.5) and reference.get('component_macro_map_shape') == [134, 262]
                    and reference.get('eliminated_macro_map_shape') == [134, 259],
                    'only exact reviewed legacy references use the inherited geometry contract')
            if si == 38.5:
                _legacy_exterior_reference(reader, reference)
            seed, _ = reader.read(reference['recipe'])
            _geometry_reference(reader, seed, 2, 48.5, 'h48p5')
            target, _ = reader.read(reference['target_report'])
            proof = target.get('geometry_extension_proof', {})
            require(target.get('state') == 'CLOSED_NATIVE_FULL_INTERFACE_ELASTIC_REFERENCE_UNSCORED'
                    and target.get('Si_um') == si and proof.get('fixed_BI_BEOL_and_xz_unchanged') is True
                    and proof.get('cell_xz_um') == [6, 6] and proof.get('old_K_scaling') is False,
                    'inherited native geometry proof differs')
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


def _legacy_exterior_reference(reader, reference):
    """Bind the one older exterior reference to its later registration proof."""
    documents = {}
    for name, (directory, sha) in LEGACY_EXTERIOR_PINS.items():
        documents[name], observed = reader.read(dict(
            path=str(LEGACY_EXTERIOR_ROOT / directory / 'REPORT.json'), sha256=sha))
        if name == 'reference':
            require(reference == documents[name], 'exact legacy exterior reference required')
        elif name != 'registration':
            same_binding(reference[name], observed)
    registered = documents['registration']
    proof = registered.get('registration', {})
    require(registered.get('schema') == 'yap.component_extension.registration.v1'
            and registered.get('state') == 'CLOSED_ALL259_NATIVE_REGISTRATION_CERTIFIED'
            and registered.get('case') == 'h38p5'
            and proof.get('all259_trace_columns_checked') is True
            and proof.get('every_full_interior_column_checked') is True
            and proof.get('raw_B_fitting') is False,
            'exact legacy exterior all259 native registration required')
    reference_directory, reference_sha = LEGACY_EXTERIOR_PINS['reference']
    same_binding(registered['reference'], dict(
        path=str(LEGACY_EXTERIOR_ROOT / reference_directory / 'REPORT.json'), sha256=reference_sha))
    same_binding(registered['native'], reference['reference'])
    same_binding(documents['target_report']['artifact'], registered['native'])


def _middle_static_configuration(reader, construction, reference):
    binding = construction['static_configuration_binding']
    require(binding['path'] == str(Path(construction['reference']['path']).parent / 'CONFIGURATION.json'),
            'exact middle static sibling configuration required')
    saved, _ = reader.read(binding)
    config = {k: v for k, v in saved.items() if k not in ('main_go', 'host')}
    _middle_pair(config, 'pitch_thickness_middle_component_reuse_20261022')
    _middle_N_sources(config['readset'])
    require(config.get('schema') == 'yap.pitch_thickness_middle_component_reuse.configuration.v1'
            and config.get('case') == reference['case'] and config.get('key') == reference['key']
            and _semantic_digest(config) == binding['semantic_sha256'] == reference['configuration_sha256']
            and config.get('readset') == reference['source_readset']
            and _semantic_digest(config['readset']) == config.get('readset_sha256')
            and type(config.get('declared_donors')) is dict
            and reference.get('declared_donors') == config['declared_donors']
            and config.get('GP_regions') == reference['GP_regions'],
            'middle static execution configuration/report differs')
    r = construction['static_configuration_reconciliation']
    require(r.get('recorded_configuration') == binding
            and r.get('all_other_fields_exact') is True
            and r.get('original_receipt_and_fresh_configuration_unmodified') is True
            and r.get('stripped_runtime_fields') == ['main_go', 'host']
            and r.get('retained_identity_fields') == [
                'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_mode', 'st_nlink'],
            'middle reconciliation must retain every non-device identity/physics field')
    rows = r['device_observations']
    expected = {'/declared_donors/%s/%s/current_identity/0' % (role, kind)
                for role in ROLES for kind in ('reference_artifact', 'operator_artifact')}
    require(type(rows) is list and len(rows) == 6 and {row['path'] for row in rows} == expected,
            'exactly six declared donor st_dev observations required')
    fresh = deepcopy(config)
    for row in rows:
        _, _, role, kind, _, _ = row['path'].split('/')
        recorded = config['declared_donors'][role][kind]['current_identity'][0]
        require(type(row['recorded']) is int and row['recorded'] == recorded
                and type(row['fresh']) is int and row['fresh'] >= 0
                and row['changed'] is (row['recorded'] != row['fresh']),
                'middle donor device observation differs from exact original configuration')
        fresh['declared_donors'][role][kind]['current_identity'][0] = row['fresh']
    require(_semantic_digest(fresh) == r['fresh_configuration_sha256'],
            'middle fresh configuration differs beyond six explicit st_dev fields')
    if any(row['changed'] for row in rows):
        require({r['recorded_host'].split('.')[0], r['fresh_host'].split('.')[0]} == {'mfd', 'sysml'},
                'device differences require reviewed mfd/sysml host pair')
    return fresh['declared_donors']


def _components(reader, history, pitch, si, states):
    context = history.get('context', {})
    case = PITCH_THICKNESS_CASES.get((pitch, si))
    if case is None:
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
        if reference.get('schema') in ('yap.pitch_thickness_component_reuse.reference.v2', ENDPOINT_REFERENCE_SCHEMA,
                                       MIDDLE_REFERENCE_SCHEMA):
            # This producer stores its fresh all259 registration in the exact
            # reference report already checked by _geometry_reference.
            same_binding(construction['registration'], context['reference'])
        else:
            registered, _ = reader.read(construction['registration'])
            require(registered.get('schema') == 'yap.component_extension.registration.v1'
                    and registered.get('state') == 'CLOSED_ALL259_NATIVE_REGISTRATION_CERTIFIED'
                    and registered.get('case') == case
                    and registered.get('registration', {}).get('all259_trace_columns_checked') is True,
                    'actual construction registration must cover all259')
            same_binding(registered['reference'], context['reference'])
    if (pitch, si) in ENDPOINT_CASES or (pitch, si) in MIDDLE_CASES:
        middle = (pitch, si) in MIDDLE_CASES
        family = 'middle' if middle else 'endpoint'
        (_middle_pair if middle else _endpoint_pair)(
            construction, 'pitch_thickness_' + family + '_operator_composition_20261022')
        require(reference.get('schema') == (MIDDLE_REFERENCE_SCHEMA if middle else ENDPOINT_REFERENCE_SCHEMA)
                and construction.get('actual_static_producer') == reference['actual_entry']
                and construction.get('actual_static_tests') == reference['actual_tests']
                and construction.get('fresh_projection') is False
                and construction.get('fresh_native_registration') == reference['native_registration']
                and construction.get('target_congruence') == reference['congruence']
                and construction.get('GP_regions') == reference['GP_regions']
                and construction.get('old_K_B_scaling') is False
                and construction.get('training', {}).get('context') == {
                    k: context[k] for k in ('Cu_source', 'acceleration_source', 'thermal_policy')}
                and construction['training'].get('native_state_adoption') is False
                and construction['training'].get('cycles2_and3_used_for_training') is False,
                'exact endpoint composition and honest inherited projection/training required')
        same_binding(construction['registration'], context['reference'])
        for role in ROLES:
            require(construction['training']['per_role_provenance'][role].get(
                'fresh_harmonic_or_projection_certificate') is False,
                'historical endpoint projection must not be relabeled fresh')
        if middle:
            _middle_N_sources(construction['readset'])
            require(_semantic_digest(construction['readset']) == construction.get('readset_sha256'),
                    'middle composition source readset digest differs')
            fresh_donors = _middle_static_configuration(reader, construction, reference)
            require(construction.get('per_role_training_provenance') == construction['training']['per_role_provenance'],
                    'middle original per-role training provenance must remain unchanged')
            require(construction['per_role_training_provenance'] == fresh_donors,
                    'middle historical training fields, including absent flags, must remain exact')
            for role in ROLES:
                same_binding(construction['components'][role]['artifact'],
                             reference['declared_donors'][role]['operator_artifact'])
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


def cumulative_artifact_location_document(previous, additions, earlier_locations):
    """Plan a replacement eighth supplement in memory; never publish or rewrite."""
    require(type(previous) is dict and set(previous) == {'schema', 'files'}
            and previous['schema'] == 'yap.operator_artifact_locations.v1'
            and type(previous['files']) is list and type(additions) is list,
            'exact previous supplement and explicit new closure required')
    old, new, earlier = previous['files'], additions, set(earlier_locations)
    require(len(old) == len(set(old)) and len(new) == len(set(new))
            and not set(new).intersection(old), 'duplicate cumulative artifact operands')
    files = sorted(old + new)
    require(1 <= len(files) <= 256 and not set(files).intersection(earlier),
            'cumulative last supplement must stay bounded and disjoint from base/first seven')
    for value in files:
        require(type(value) is str, 'literal artifact location required')
        path = Path(value)
        require(not path.is_absolute() and path.parts and path.parts[0] == 'operator_repair_20261015'
                and str(path) == value and len(path.parts) <= 40 and path.suffix in ('.json', '.npz', '.py')
                and not any(part in ('..', '#snapshot', '.snapshot', 'Ansys_Verify', 'mapdl_run_roots',
                    'mapdl_archives', 'archives', 'source_snapshot')
                    or part.startswith(('.migration_hold_', 'storage_migration')) for part in path.parts),
                'unsafe cumulative artifact location')
    result = dict(schema=previous['schema'], files=files)
    require(len(json.dumps(result, sort_keys=True).encode('ascii')) <= 128 * 1024,
            'bounded cumulative supplement serialization required')
    return result


def build_catalog(progress_path=None, *, root=ROOT):
    """Retain the baseline and incrementally admit declared exterior entries."""
    reader = MetadataReader(root)
    progress, _ = reader.read(progress_path or Path(root) / PROGRESS_RELATIVE)
    coverage = progress['audited_current_summary_20261021']['coverage']
    locators = coverage['single_BI_p2_full_three_cycle_passed']
    requested = []
    optional = tuple(si for si in P2_EXTERIOR_SI
                     if any(row.get('Si_TSV_um') == si for row in locators))
    for si in (*P2_SI, *optional):
        matches = [row for row in locators if row.get('Si_TSV_um') == si]
        require(len(matches) == 1, 'one progress locator per exact P2 Si thickness required')
        requested.append((2, si, matches[0]['evidence']))
    for pitch in (3, 6):
        matches = [row for row in coverage['other_pitches'] if row.get('pitch_um') == pitch]
        require(len(matches) == 1, 'one progress locator per exact BI pitch required')
        requested.append((pitch, 37, matches[0]['evidence']))
    additions = coverage.get('pitch_thickness_passed', [])
    require(type(additions) is list and len(additions) <= len(PITCH_THICKNESS_CASES),
            'bounded incremental pitch/thickness locators required')
    selected = set()
    for row in additions:
        require(isinstance(row, dict) and {'pitch_um', 'Si_TSV_um', 'evidence'} <= row.keys()
                and row['evidence'] is not None,
                'selected pitch/thickness case requires an explicit completed bow evidence locator')
        pair = (row['pitch_um'], row['Si_TSV_um'])
        require(pair in PITCH_THICKNESS_CASES and pair not in selected,
                'one locator per reviewed incremental pitch/thickness geometry required')
        selected.add(pair)
        requested.append((*pair, row['evidence']))
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
        except (KeyError, TypeError, ValueError, OSError) as exc:
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
