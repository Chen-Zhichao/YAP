"""Metadata-only publication coverage; no operators or solver APIs are loaded."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat

import pytest

import tools.publish_component_operators as publisher


@pytest.fixture(scope='module')
def actual_catalog():
    if not (publisher.ROOT / publisher.PROGRESS_RELATIVE).is_file():
        pytest.skip('canonical component evidence is not installed')
    return publisher.build_catalog()


def test_actual_entries_preserve_baseline_and_are_exact_and_metadata_only(monkeypatch):
    if not (publisher.ROOT / publisher.PROGRESS_RELATIVE).is_file():
        pytest.skip('canonical component evidence is not installed')
    opened = []
    original = publisher.MetadataReader._open

    def metadata_open(self, path):
        assert path.suffix == '.json', 'numerical payload opened during publication'
        opened.append(path)
        return original(self, path)

    def no_mkdir(*args, **kwargs):
        pytest.fail('in-memory catalog construction created an output directory')

    monkeypatch.setattr(publisher.MetadataReader, '_open', metadata_open)
    monkeypatch.setattr(publisher.os, 'mkdir', no_mkdir)
    catalog = publisher.build_catalog()
    progress, _ = publisher.MetadataReader().read(publisher.ROOT / publisher.PROGRESS_RELATIVE)
    locators = progress['audited_current_summary_20261021']['coverage']['single_BI_p2_full_three_cycle_passed']
    optional = [si for si in publisher.P2_EXTERIOR_SI
                if any(row.get('Si_TSV_um') == si for row in locators)]
    expected = [f'BI2_SI{si:g}'.replace('.', 'p') for si in publisher.P2_SI]
    optional_ids = [f'BI2_SI{si:g}'.replace('.', 'p') for si in optional]
    expected += optional_ids
    expected += ['BI3_SI37_COARSE45', 'BI6_SI37']
    assert [entry['id'] for entry in catalog['entries']] == expected
    assert set(catalog) == {'schema', 'entries'}
    assert catalog['schema'] == publisher.SCHEMA
    for entry in catalog['entries']:
        assert set(entry) == {'id', 'geometry', 'components', 'construction', 'reference',
                              'validation', 'capabilities'}
        assert len(entry['geometry']) == 9
        assert entry['geometry']['bi_layer_count'] == 1
        assert entry['geometry']['total_thickness_um'] == entry['geometry']['si_tsv_thickness_um'] + 11.5
        assert set(entry['components']) == {'bi', 'beol', 'si'}
        assert entry['validation']['cycles'] == 3
        assert entry['validation']['bow_observations'] == 1081
        assert entry['validation']['solved_states'] == (
            73 if entry['id'] in expected[:4] + optional_ids else 61)
        assert entry['validation']['reference_kind'] == 'native_FE'
        assert entry['validation']['MAPDL_direct_validation'] is False
        assert entry['capabilities'] == dict(single_bi_history=True, experimental_prepared_d2w=True,
                                             prepared_assembly_accuracy=False, W2W_thinning=False)
    assert 'fine90_time_pending' in catalog['entries'][-2]['validation']['mesh_scope']
    assert opened and len(set(opened)) < 128


def test_shared_reader_validator_matches_every_actual_report(actual_catalog):
    reader = publisher.MetadataReader()
    for entry in actual_catalog['entries']:
        report, _ = reader.read(entry['validation']['report'])
        normalized = publisher.validate_bow_evidence(report)
        assert normalized == {key: entry['validation'][key] for key in normalized}


@pytest.mark.parametrize('entry_id', ['BI2_SI7', 'BI2_SI17', 'BI2_SI48p5', 'BI2_SI77',
                                      'BI2_SI47', 'BI3_SI37_COARSE45', 'BI6_SI37'])
def test_refuses_failed_or_open_actual_bow_report(actual_catalog, entry_id):
    entry = next(e for e in actual_catalog['entries'] if e['id'] == entry_id)
    document, _ = publisher.MetadataReader().read(entry['validation']['report'])
    document = copy.deepcopy(document)
    document['state'] = 'RUNNING'
    with pytest.raises(ValueError):
        publisher.validate_bow_evidence(document)


def test_refuses_wrong_probe_pass_even_with_passing_progress(actual_catalog):
    entry = next(e for e in actual_catalog['entries'] if e['id'] == 'BI2_SI47')
    document, _ = publisher.MetadataReader().read(entry['validation']['report'])
    document = copy.deepcopy(document)
    block = document['combined_ROM61_vs_dense_native']['bow_only_interpolant']
    block['by_probe']['edge_x_um']['passed'] = False
    with pytest.raises(ValueError, match='probe'):
        publisher.validate_bow_evidence(document)


def test_progress_claims_are_not_admission(monkeypatch):
    expected_count = len(publisher.build_catalog()['entries'])
    original = publisher.MetadataReader.read

    def read(self, value):
        document, pin = original(self, value)
        if pin['path'].endswith(publisher.PROGRESS_RELATIVE):
            document = copy.deepcopy(document)
            for row in document['audited_current_summary_20261021']['coverage']['single_BI_p2_full_three_cycle_passed']:
                row.update(full_three_cycle_bow_passed=False, max_relative_pct=999., actual_ROM_states=1)
        return document, pin

    monkeypatch.setattr(publisher.MetadataReader, 'read', read)
    assert len(publisher.build_catalog()['entries']) == expected_count


def test_refuses_different_operator_variant_for_same_geometry(actual_catalog):
    entry = next(e for e in actual_catalog['entries'] if e['id'] == 'BI2_SI37')
    reader = publisher.MetadataReader()
    history, _ = reader.read(entry['validation']['history'])
    history = copy.deepcopy(history)
    history['context']['operator_payloads']['si']['sha256'] = '0' * 64
    history['checkpoints'][-1]['context']['operator_payloads']['si']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='variant differs'):
        publisher._components(reader, history, 2, 37, 73)


@pytest.mark.parametrize('entry_id', ['BI2_SI7', 'BI2_SI77'])
def test_special_audits_bind_exact_closed_candidate(actual_catalog, entry_id, monkeypatch):
    entry = next(e for e in actual_catalog['entries'] if e['id'] == entry_id)
    reader = publisher.MetadataReader()
    original = reader.read

    def read(value):
        document, pin = original(value)
        if pin == entry['validation']['report']:
            document = copy.deepcopy(document)
            if entry_id == 'BI2_SI7':
                document['candidate_q6']['sha256'] = '0' * 64
            else:
                for name, item in document['source_report_bindings'].items():
                    if name.startswith('COMPONENT_EXTENSION_PREDICT_h77_'):
                        item['sha256'] = '0' * 64
        return document, pin

    monkeypatch.setattr(reader, 'read', read)
    with pytest.raises(ValueError, match='variant differs'):
        publisher._assessment(reader, entry['validation']['report'], 2,
                              entry['geometry']['si_tsv_thickness_um'])


def test_publication_preserves_existing_delivery_and_is_exclusive(tmp_path, monkeypatch):
    distribution = tmp_path / 'distribution'
    distribution.mkdir()
    delivery = distribution / 'delivery'
    delivery.mkdir(mode=0o750)
    runtime = delivery / 'runtime'
    runtime.mkdir()
    before = stat.S_IMODE(delivery.stat().st_mode)
    catalog = {'schema': publisher.SCHEMA, 'entries': []}
    monkeypatch.setattr(publisher, 'build_catalog', lambda *args, **kwargs: catalog)
    target = delivery / 'CATALOG.json'
    result = publisher.publish_catalog(target, root=tmp_path)
    assert result['sha256'] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert json.loads(target.read_text()) == catalog
    assert stat.S_IMODE(target.stat().st_mode) == 0o444
    assert stat.S_IMODE(delivery.stat().st_mode) == before
    assert runtime.is_dir()
    with pytest.raises(FileExistsError):
        publisher.publish_catalog(target, root=tmp_path)


@pytest.mark.parametrize('relative', ['payload.npz', '#snapshot/a.json', '.snapshot/a.json',
                                      'storage_migration_20261002/a.json'])
def test_metadata_reader_refuses_payload_and_protected_paths(tmp_path, relative):
    with pytest.raises(ValueError):
        publisher.MetadataReader(tmp_path).read(tmp_path / relative)


def test_metadata_reader_checks_sha_and_duplicate_keys(tmp_path):
    path = tmp_path / 'metadata.json'
    path.write_text('{"case":1,"case":2}')
    with pytest.raises(ValueError, match='duplicate'):
        publisher.MetadataReader(tmp_path).read(path)
    path.write_text('{"case":1}')
    with pytest.raises(ValueError, match='variant differs'):
        publisher.MetadataReader(tmp_path).read({'path': str(path), 'sha256': '0' * 64})


def test_cli_without_output_does_not_publish(monkeypatch, capsys):
    monkeypatch.setattr(publisher, 'build_catalog', lambda *args, **kwargs: {'schema': publisher.SCHEMA, 'entries': []})
    monkeypatch.setattr(publisher, 'publish_catalog', lambda *args, **kwargs: pytest.fail('implicit publication'))
    assert publisher.main([]) == 0
    assert json.loads(capsys.readouterr().out)['schema'] == publisher.SCHEMA


def _incremental_metadata(monkeypatch, exterior, *, refuse=None):
    locators = [dict(Si_TSV_um=si, evidence=f'evidence:{si}')
                for si in (*publisher.P2_SI, *exterior)]
    progress = dict(audited_current_summary_20261021=dict(coverage=dict(
        single_BI_p2_full_three_cycle_passed=locators,
        other_pitches=[dict(pitch_um=p, evidence=f'evidence:p{p}') for p in (3, 6)])))
    selected = []

    def read(self, value):
        return progress, {}

    def assessment(reader, receipt, pitch, si):
        selected.append((pitch, si, receipt))
        if si == refuse:
            raise ValueError('candidate assessment did not pass')
        return {}, {}, {}, 73, 5., 10.

    monkeypatch.setattr(publisher.MetadataReader, 'read', read)
    monkeypatch.setattr(publisher, '_assessment', assessment)
    monkeypatch.setattr(publisher, '_components', lambda *args: ({}, {}, {}))
    return selected


@pytest.mark.parametrize('count', [0, 1, 2, 3])
def test_exterior_entries_publish_incrementally_not_as_a_batch(monkeypatch, count):
    extras = publisher.P2_EXTERIOR_SI[:count]
    selected = _incremental_metadata(monkeypatch, extras)
    catalog = publisher.build_catalog()
    assert len(catalog['entries']) == 13 + count
    assert [(p, si) for p, si, _ in selected] == (
        [(2, si) for si in (*publisher.P2_SI, *extras)] + [(3, 37), (6, 37)])
    for si in extras:
        entry = next(e for e in catalog['entries'] if e['id'] == f'BI2_SI{si:g}'.replace('.', 'p'))
        assert entry['geometry']['total_thickness_um'] == si + 11.5
        assert entry['capabilities']['prepared_assembly_accuracy'] is False


def test_optional_progress_entry_still_requires_passing_assessment(monkeypatch):
    _incremental_metadata(monkeypatch, [8.5], refuse=8.5)
    with pytest.raises(ValueError, match='candidate assessment did not pass'):
        publisher.build_catalog()


def test_duplicate_optional_geometry_is_refused(monkeypatch):
    _incremental_metadata(monkeypatch, [8.5, 8.5])
    with pytest.raises(ValueError, match='one progress locator'):
        publisher.build_catalog()


def test_unknown_optional_geometry_is_not_published(monkeypatch):
    selected = _incremental_metadata(monkeypatch, [28.5])
    assert len(publisher.build_catalog()['entries']) == 13
    assert all(si != 28.5 for _, si, _ in selected)


def _synthetic_pitch_thickness_bow():
    def block(rows):
        return dict(passed=True, matched_rows=rows, failed_sample_probe_count=0,
                    sample_probe_count=4 * rows, by_probe={
                        name: dict(passed=True, failed_sample_probe_count=0,
                                   sample_probe_count=rows, finite_relative_max_pct=2.,
                                   max_abs_error_um=3.) for name in publisher.PROBES})
    return dict(schema='yap.pitch_thickness_bow_assessment.v1',
                state='CLOSED_FULL_GRADED73_BOW_PROFILE_ASSESSMENT',
                through_original_ordinal=1080, native_prefix_states=1081,
                actual_ROM_solved_states_compared=73, total_ROM_solved_states=73,
                prefix_only=False, full_native1081_closure_verified=True,
                qualification_passed=True, global_accuracy_passed=True,
                ROM_accepted_state_chain_verified=True, Cu_arrays_decoded=False,
                Cu_interpolation=False, MAPDL_accuracy_claim=False,
                lineage=dict(source_sha256=publisher.PITCH_THICKNESS_ASSESSOR_SHA,
                             tests_sha256=publisher.PITCH_THICKNESS_ASSESSOR_TEST_SHA,
                             through_original_ordinal=1080,
                             native_report=dict(path='synthetic', sha256='0' * 64)),
                solved_nodes=block(73), dense_bow_interpolant=block(1081))


def test_full_pitch_thickness_receipt_uses_same_bow_gate():
    assert publisher.validate_bow_evidence(_synthetic_pitch_thickness_bow()) == dict(
        cycles=3, solved_states=73, bow_observations=1081,
        max_relative_pct=2., max_absolute_um=3.)


def _synthetic_profile_pitch_thickness_bow():
    document = _synthetic_pitch_thickness_bow()
    document['lineage'].update(
        source_sha256=publisher.PITCH_THICKNESS_PROFILE_ASSESSOR_SHA,
        tests_sha256=publisher.PITCH_THICKNESS_PROFILE_ASSESSOR_TEST_SHA,
        genuine_assessment_source=dict(
            path=str(publisher.LEGACY_EXTERIOR_ROOT.parent / 'pitch_thickness_bow_assessment_20261022.py'),
            sha256=publisher.PITCH_THICKNESS_ASSESSOR_SHA),
        genuine_assessment_tests=dict(
            path=str(publisher.LEGACY_EXTERIOR_ROOT.parent / 'test_pitch_thickness_bow_assessment_20261022.py'),
            sha256=publisher.PITCH_THICKNESS_ASSESSOR_TEST_SHA),
        producer_authentication_AST=dict(inverse_full_module_exact=True,
            scoring_decoding_pairing_Cu_JVP_clock_and_gate_unchanged=True))
    return document


def test_authenticated_profile_assessment_uses_unchanged_full_bow_gate():
    assert publisher.validate_bow_evidence(_synthetic_profile_pitch_thickness_bow()) == dict(
        cycles=3, solved_states=73, bow_observations=1081,
        max_relative_pct=2., max_absolute_um=3.)


@pytest.mark.parametrize('change', ['mixed_tests', 'genuine_source', 'genuine_tests', 'AST', 'scoring', 'prefix'])
def test_profile_assessment_cannot_bypass_ancestry_or_full_history(change):
    document = _synthetic_profile_pitch_thickness_bow()
    lineage = document['lineage']
    if change == 'mixed_tests': lineage['tests_sha256'] = publisher.PITCH_THICKNESS_ASSESSOR_TEST_SHA
    if change == 'genuine_source': lineage['genuine_assessment_source']['sha256'] = '0' * 64
    if change == 'genuine_tests': lineage['genuine_assessment_tests']['sha256'] = '0' * 64
    if change == 'AST': lineage['producer_authentication_AST']['inverse_full_module_exact'] = False
    if change == 'scoring': lineage['producer_authentication_AST']['scoring_decoding_pairing_Cu_JVP_clock_and_gate_unchanged'] = False
    if change == 'prefix': document['prefix_only'] = True
    with pytest.raises(ValueError):
        publisher.validate_bow_evidence(document)


@pytest.mark.parametrize('field,value', [
    ('state', 'CLOSED_PREFIX_BOW_DIAGNOSTIC_NOT_FULL_ADMISSION'),
    ('through_original_ordinal', 160), ('prefix_only', True),
    ('full_native1081_closure_verified', False), ('qualification_passed', None),
    ('global_accuracy_passed', False), ('ROM_accepted_state_chain_verified', False),
    ('native_prefix_states', 1080), ('actual_ROM_solved_states_compared', 61),
    ('Cu_interpolation', True), ('MAPDL_accuracy_claim', True),
])
def test_incomplete_pitch_thickness_receipt_is_not_admission(field, value):
    document = _synthetic_pitch_thickness_bow()
    document[field] = value
    with pytest.raises(ValueError):
        publisher.validate_bow_evidence(document)


@pytest.mark.parametrize('field,value', [
    ('source_sha256', '0' * 64), ('tests_sha256', '0' * 64),
    ('native_report', None), ('through_original_ordinal', 160),
])
def test_pitch_thickness_assessor_lineage_is_authenticated(field, value):
    document = _synthetic_pitch_thickness_bow()
    document['lineage'][field] = value
    with pytest.raises(ValueError):
        publisher.validate_bow_evidence(document)


def test_actual_first_heat_prefix_cannot_be_published():
    path = publisher.LEGACY_EXTERIOR_ROOT / (
        'PITCH_THICKNESS_BOW_ASSESSMENT_20261022_p3_si17_through160_v1/REPORT.json')
    if not path.is_file():
        pytest.skip('pitch/thickness experiment evidence is not installed')
    document, _ = publisher.MetadataReader().read(path)
    with pytest.raises(ValueError, match='three-cycle'):
        publisher.validate_bow_evidence(document)


def test_closed_p3_si17_components_are_registered_but_not_published():
    path = publisher.LEGACY_EXTERIOR_ROOT / (
        'PITCH_THICKNESS_REDUCED_GRADED73_20261022_p3_si17_mfd_v1/REPORT.json')
    if not path.is_file():
        pytest.skip('pitch/thickness experiment evidence is not installed')
    reader = publisher.MetadataReader()
    history, _ = reader.read(path)
    components, construction, reference = publisher._components(reader, history, 3, 17, 73)
    assert set(components) == set(publisher.ROLES)
    assert construction == publisher.binding(history['context']['operator'])
    assert reference == publisher.binding(history['context']['reference'])


def test_pitch_thickness_locators_are_incremental(monkeypatch):
    selected = _incremental_metadata(monkeypatch, ())
    original = publisher.MetadataReader.read
    def read(self, value):
        progress, pin = original(self, value)
        progress['audited_current_summary_20261021']['coverage']['pitch_thickness_passed'] = [
            dict(pitch_um=3, Si_TSV_um=17, evidence='synthetic-passed')]
        return progress, pin
    monkeypatch.setattr(publisher.MetadataReader, 'read', read)
    entries = publisher.build_catalog()['entries']
    assert len(entries) == 14
    assert selected[-1] == (3, 17, 'synthetic-passed')
    assert entries[-1]['id'] == 'BI3_SI17_COARSE45'
    assert entries[-1]['geometry']['total_thickness_um'] == 28.5
    assert entries[-1]['capabilities']['prepared_assembly_accuracy'] is False


@pytest.mark.parametrize('pairs', [[(3, 17), (3, 17)], [(4, 17)], [(3, 28)]])
def test_unreviewed_or_duplicate_pitch_thickness_locators_refused(monkeypatch, pairs):
    _incremental_metadata(monkeypatch, ())
    original = publisher.MetadataReader.read
    def read(self, value):
        progress, pin = original(self, value)
        progress['audited_current_summary_20261021']['coverage']['pitch_thickness_passed'] = [
            dict(pitch_um=pitch, Si_TSV_um=si, evidence='synthetic') for pitch, si in pairs]
        return progress, pin
    monkeypatch.setattr(publisher.MetadataReader, 'read', read)
    with pytest.raises(ValueError, match='one locator'):
        publisher.build_catalog()


def _actual_legacy_exterior(reader):
    directory, sha = publisher.LEGACY_EXTERIOR_PINS['reference']
    path = publisher.LEGACY_EXTERIOR_ROOT / directory / 'REPORT.json'
    if not path.is_file():
        pytest.skip('canonical exterior evidence is not installed')
    return reader.read(dict(path=str(path), sha256=sha))[0]


def test_exact_legacy_exterior_uses_separate_registration_without_payloads(monkeypatch):
    reader = publisher.MetadataReader()
    reference = _actual_legacy_exterior(reader)
    original = reader._open

    def metadata_only(path):
        assert path.suffix == '.json'
        return original(path)

    monkeypatch.setattr(reader, '_open', metadata_only)
    assert reference.get('geometry_contract') is None
    publisher._geometry_reference(reader, reference, 2, 38.5, 'h38p5')
    reader.verify_unchanged()


def test_legacy_exterior_does_not_accept_rewritten_reference():
    reader = publisher.MetadataReader()
    reference = copy.deepcopy(_actual_legacy_exterior(reader))
    reference['reference']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='exact legacy exterior reference'):
        publisher._geometry_reference(reader, reference, 2, 38.5, 'h38p5')


@pytest.mark.parametrize('field', ['all259_trace_columns_checked',
                                  'every_full_interior_column_checked', 'raw_B_fitting'])
def test_legacy_exterior_requires_actual_registration_flags(monkeypatch, field):
    reader = publisher.MetadataReader()
    reference = _actual_legacy_exterior(reader)
    original = reader.read
    registration_directory, _ = publisher.LEGACY_EXTERIOR_PINS['registration']

    def read(value):
        document, observed = original(value)
        if Path(observed['path']).parent.name == registration_directory:
            document = copy.deepcopy(document)
            document['registration'][field] = not document['registration'][field]
        return document, observed

    monkeypatch.setattr(reader, 'read', read)
    with pytest.raises(ValueError, match='all259 native registration'):
        publisher._geometry_reference(reader, reference, 2, 38.5, 'h38p5')


def _endpoint_bow(pitch=3, si=7):
    document = _synthetic_pitch_thickness_bow()
    lineage = document['lineage']
    lineage.update(source_sha256=publisher.ENDPOINT_ASSESSOR_PAIR[0],
        tests_sha256=publisher.ENDPOINT_ASSESSOR_PAIR[1],
        genuine_assessment_source=dict(
            path=str(publisher.LEGACY_EXTERIOR_ROOT.parent / 'pitch_thickness_bow_assessment_20261022.py'),
            sha256=publisher.PITCH_THICKNESS_ASSESSOR_SHA),
        genuine_assessment_tests=dict(
            path=str(publisher.LEGACY_EXTERIOR_ROOT.parent / 'test_pitch_thickness_bow_assessment_20261022.py'),
            sha256=publisher.PITCH_THICKNESS_ASSESSOR_TEST_SHA),
        endpoint_ROM_entry=publisher._endpoint_source('pitch_thickness_endpoint_rom_history_20261022.py'),
        endpoint_ROM_tests=publisher._endpoint_source('test_pitch_thickness_endpoint_rom_history_20261022.py'),
        additional_source_pins=dict(publisher.ENDPOINT_PINS),
        producer_authentication_AST=dict(inverse_full_module_exact=True, changed_expressions=10,
            scoring_q6_decoder_clock_pairing_state_chain_JVP_and_publication_bodies_unchanged=True),
        operator_contract_AST=dict(inverse_full_module_exact=True))
    document.update(key=f'p{pitch}_si{si}', case=publisher.ENDPOINT_CASES[pitch, si],
        full_span_x_mm=10., full_span_y_mm=10., bow_units='um', probe_names=sorted(publisher.PROBES),
        selected_original_ordinals=list(publisher.GRADED73_ORDINALS),
        dense_1081_ROM_solved_history_claim=False, ROM_strict_linear_certificates=72)
    return document


def _endpoint_documents(monkeypatch, *, as_gib=16):
    """Real closed static metadata plus explicitly synthetic future histories."""
    reader = publisher.MetadataReader()
    root = publisher.LEGACY_EXTERIOR_ROOT
    operator, operator_pin = reader.read(dict(path=str(root /
        'PITCH_THICKNESS_ENDPOINT_OPERATOR_COMPOSITION_20261022_p3_si7_sysml_v1/REPORT.json'),
        sha256='1280e284b568f4ce70c2adb125b9c34b77a3887ab9b9d7477a4f4555db434878'))
    reference, reference_pin = reader.read(operator['reference'])
    docs = {}

    def add(path, value):
        path = str(path)
        docs[path] = value
        raw = json.dumps(value, sort_keys=True, allow_nan=False).encode('ascii')
        return dict(path=path, sha256=hashlib.sha256(raw).hexdigest())

    report = _endpoint_bow()
    case, key = report['case'], report['key']
    rp = root / 'PITCH_THICKNESS_ENDPOINT_ROM_GRADED73_20261022_p3_si7_synthetic' / 'REPORT.json'
    np = root / 'PITCH_THICKNESS_ENDPOINT_NATIVE_DENSE_20261022_p3_si7_synthetic' / 'REPORT.json'
    regions = reference['GP_regions']
    c = dict(case=case, operator=operator_pin, reference=reference_pin, topology=reference['topology_artifact'],
        operator_payloads={r: v['artifact'] for r, v in operator['components'].items()},
        micro_ranks={r: v['micro_rank'] for r, v in operator['components'].items()},
        global_macro_port_coordinates=259, all_GP_retained=True, Cu_state_adoption=False,
        virgin_birth_fields_zero=True, external_checkpoint_reads=0,
        selected_original_ordinals=list(publisher.GRADED73_ORDINALS), Cu_GP=185072,
        **operator['training']['context'])
    n = dict(case=case, reference=reference['reference'], native_elastic_report=reference['native_report'],
        selected_original_ordinals=list(range(1081)), GP_regions=regions, clock_states=1081,
        temporal_profile='dense_original1081', MAPDL_reads=0, component_reference_used=False,
        component_certificate_dependency=False, ROM_dependency=False,
        virgin_birth_fields_zero=True, Cu_state_adoption=False, **operator['training']['context'])
    n.update(history_wrapper_sha256=publisher.ENDPOINT_IMPLEMENTATION_PINS['pitch_thickness_native_dense_20261022.py'],
        history_tests_sha256=publisher.ENDPOINT_IMPLEMENTATION_PINS['test_pitch_thickness_native_dense_20261022.py'])
    history = dict(schema='yap.component_reduced_clock.history.v1',
        state='CLOSED_REDUCED_CLOCK_VIRGIN_HISTORY_UNSCORED', case=case, key=key,
        source_sha256=publisher.ENDPOINT_PINS['pitch_thickness_endpoint_rom_history_20261022.py'],
        tests_sha256=publisher.ENDPOINT_PINS['test_pitch_thickness_endpoint_rom_history_20261022.py'],
        states=73, initial_ordinal=0, final_ordinal=72, final_original_ordinal=1080,
        MAPDL_result_reads=0, external_checkpoint_reads=0, no_MAPDL_state_adopted=True,
        context=c, steps=[dict(path=f'synthetic-step-{i}') for i in range(72)],
        certified_linear_solves=72, address_space_bytes=as_gib*1024**3,
        AS_trial_profile=f'AS{as_gib}', completed_history_under_executable_AS=True,
        checkpoints=[dict(ordinal=publisher.GRADED73_ORDINALS.index(i), original_ordinal=i,
            all_Cu14=True, no_MAPDL_state_adopted=True, context=c) for i in publisher.PHASE_ORDINALS])
    native = dict(state='CLOSED_PITCH_THICKNESS_NATIVE_DENSE1081_UNSCORED', states=1081,
        case=case, context=n, Cu_points=185072, interior_coordinates=273372,
        source_sha256=publisher.ENDPOINT_IMPLEMENTATION_PINS['component_fullspace_validation.py'],
        checkpoints=[dict(ordinal=i, all_Cu14=True, no_native_state_adopted=True, context=n)
                     for i in publisher.PHASE_ORDINALS])
    config_pins = {}
    for path, stem, schema, cap, extra in (
        (rp, 'pitch_thickness_endpoint_rom_history_20261022',
         'yap.pitch_thickness_endpoint_rom_history.configuration.v1', as_gib, dict(operator=operator_pin)),
        (np, 'pitch_thickness_endpoint_native_dense_20261022',
         'yap.pitch_thickness_endpoint_native_dense.configuration.v1', 32,
         dict(native_report=reference['native_report'], native=reference['reference']))):
        source, tests = (publisher._endpoint_source(name) for name in (stem+'.py', 'test_'+stem+'.py'))
        readset = {v['path']: v['sha256'] for v in (source, tests,
            publisher._endpoint_source('pitch_thickness_case_manifest_20261022.py'))}
        config = dict(schema=schema, key=key, case=case, source_sha256=source['sha256'],
            tests_sha256=tests['sha256'], GP_regions=regions, Cu_GP=185072,
            thermal_policy=c['thermal_policy'], virgin_birth_fields_zero=True, external_checkpoint_reads=0, readset=readset,
            readset_sha256=publisher._semantic_digest(readset), AS_trial_profile=f'AS{cap}',
            limits=dict(address_space_bytes=cap*1024**3, minimum_available_bytes=(cap+32)*1024**3), **extra)
        digest = publisher._semantic_digest(config)
        context = dict(source=source, tests=tests, configuration_sha256=digest)
        if path == rp:
            context.update(operator=operator_pin, public_ready=False, native_history_read=False)
            for field, name in (('mathematical_wrapper', 'pitch_thickness_reduced_history_20261022.py'),
                                ('admission_implementation', 'pitch_thickness_reduced_profiles_20261022.py')):
                context[field] = dict(path=str(publisher.LEGACY_EXTERIOR_ROOT.parent/name),
                    sha256=publisher.ENDPOINT_IMPLEMENTATION_PINS[name])
            history.update(configuration_sha256=digest, endpoint_rom_entry=context)
            name = 'CONFIGURATION.json'
        else:
            config['Cu_state_adoption'] = False
            digest = publisher._semantic_digest(config)
            context['configuration_sha256'] = digest
            context['raw_native_report'] = reference['native_report']
            n['endpoint_native_entry'] = context
            native['dense1081_configuration_sha256'] = digest
            name = 'DENSE1081_CONFIGURATION.json'
        config_pins[str(path.parent/name)] = add(path.parent/name, config)
    history['checkpoints'][-1]['payload'] = dict(path=str(rp.parent/'CHECKPOINT_000072.npz'), sha256='1'*64)
    cp = np.parent/'CHECKPOINT_001080.json'
    report['lineage'].update(rom=add(rp, history), native_report=add(np, native),
        native_checkpoint=add(cp, native['checkpoints'][-1]), metadata_inputs=config_pins)
    assessment = add(root/'SYNTHETIC_ENDPOINT_BOW_NOT_ADMISSION/REPORT.json', report)
    original = reader.read

    def read(value):
        path = value['path'] if isinstance(value, dict) else str(value)
        if path not in docs:
            return original(value)
        observed = add(path, docs[path])
        if isinstance(value, dict):
            publisher.same_binding(value, observed)
        return docs[path], observed

    monkeypatch.setattr(reader, 'read', read)
    return dict(reader=reader, report=report, history=history, native=native,
                reference=reference, operator=operator, docs=docs, assessment=assessment)


@pytest.mark.parametrize('pair', publisher.ENDPOINT_CASES)
def test_endpoint_bow_exact_reviewed_pair_and_full_scope(pair):
    assert publisher.validate_bow_evidence(_endpoint_bow(*pair))['bow_observations'] == 1081


@pytest.mark.parametrize('change', ['ROM_source', 'ROM_tests', 'native_pin', 'static_pin', 'composer_pin',
    'genuine', 'inverse', 'scorer', 'operator_AST', 'prefix', 'probe', 'span', 'clock', 'rows', 'JVP'])
def test_endpoint_bow_lineage_and_scope_cannot_be_bypassed(change):
    report = _endpoint_bow()
    lineage = report['lineage']
    if change == 'ROM_source': lineage['endpoint_ROM_entry']['sha256'] = '0'*64
    if change == 'ROM_tests': lineage['endpoint_ROM_tests']['sha256'] = '0'*64
    if change in ('native_pin', 'static_pin', 'composer_pin'):
        name = {'native_pin': 'native_dense', 'static_pin': 'component_reuse', 'composer_pin': 'operator_composition'}[change]
        lineage['additional_source_pins'][f'pitch_thickness_endpoint_{name}_20261022.py'] = '0'*64
    if change == 'genuine': lineage['genuine_assessment_source']['sha256'] = '0'*64
    if change == 'inverse': lineage['producer_authentication_AST']['inverse_full_module_exact'] = False
    if change == 'scorer': lineage['producer_authentication_AST']['scoring_q6_decoder_clock_pairing_state_chain_JVP_and_publication_bodies_unchanged'] = False
    if change == 'operator_AST': lineage['operator_contract_AST']['inverse_full_module_exact'] = False
    if change == 'prefix': report['prefix_only'] = True
    if change == 'probe': report['dense_bow_interpolant']['by_probe'].pop('edge_x_um')
    if change == 'span': report['full_span_x_mm'] = 5.
    if change == 'clock': report['selected_original_ordinals'][1] = 19
    if change == 'rows': report['dense_bow_interpolant']['matched_rows'] = 1080
    if change == 'JVP': report['ROM_strict_linear_certificates'] = 0
    with pytest.raises(ValueError):
        publisher.validate_bow_evidence(report)


@pytest.mark.parametrize('as_gib', [16, 32])
def test_endpoint_real_static_metadata_with_synthetic_full_history(monkeypatch, as_gib):
    f = _endpoint_documents(monkeypatch, as_gib=as_gib)
    result = publisher._assessment(f['reader'], f['assessment'], 3, 7)
    assert result[3:] == (73, 2., 3.)
    components, _, _ = publisher._components(f['reader'], f['history'], 3, 7, 73)
    assert set(components) == set(publisher.ROLES)
    assert all(Path(p).suffix == '.json' for p in f['reader'].cache)


def test_endpoint_reviewed_pins_match_named_sources():
    for name, sha in dict(publisher.ENDPOINT_PINS, **publisher.ENDPOINT_IMPLEMENTATION_PINS).items():
        assert hashlib.sha256((publisher.LEGACY_EXTERIOR_ROOT.parent/name).read_bytes()).hexdigest() == sha


def test_endpoint_genuine_ROM_metadata_prepare_matches_publication_contract(monkeypatch):
    f = _endpoint_documents(monkeypatch)
    source = publisher._endpoint_source('pitch_thickness_endpoint_rom_history_20261022.py')
    spec = importlib.util.spec_from_file_location('_publisher_reviewed_endpoint_ROM', source['path'])
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    delivered_operator = f['history']['context']['operator']
    # The frozen producer accepts its construction root, not a delivery relocation.
    operator = dict(delivered_operator, path=str(producer.RT /
        'PITCH_THICKNESS_ENDPOINT_OPERATOR_COMPOSITION_20261022_p3_si7_sysml_v1/REPORT.json'))
    publisher.same_binding(operator, delivered_operator)
    layer = producer.private_adapter(16)
    monkeypatch.chdir(publisher.ROOT)
    for name in layer.module.ENV_KEYS:
        monkeypatch.setenv(name, str(producer.RT))
    opened = layer.module.opened

    def metadata_only(path, *args, **kwargs):
        assert Path(path).suffix != '.npz', 'genuine metadata prepare decoded a numerical payload'
        return opened(path, *args, **kwargs)

    monkeypatch.setattr(layer.module, 'opened', metadata_only)
    prepared = producer.prepare('p3_si7', Path(operator['path']), operator['sha256'], source['sha256'],
        publisher.ENDPOINT_PINS['test_pitch_thickness_endpoint_rom_history_20261022.py'],
        as_gib=16, layer=layer, frozen=True)
    config = prepared.configuration
    # The real ROM configuration has virgin/read counters, not a fabricated
    # Cu_state_adoption field. That flag is authenticated in the runtime context.
    assert 'Cu_state_adoption' not in config
    context = producer.endpoint_context(prepared.layer.module, config)
    f['history'].update(configuration_sha256=publisher._semantic_digest(config), endpoint_rom_entry=context)
    lineage = f['report']['lineage']
    path = str(Path(lineage['rom']['path']).parent/'CONFIGURATION.json')
    f['docs'][path] = config
    for pin in (lineage['metadata_inputs'][path], lineage['rom'], f['assessment']):
        pin['sha256'] = hashlib.sha256(json.dumps(f['docs'][pin['path']], sort_keys=True,
                                                allow_nan=False).encode('ascii')).hexdigest()
    assert publisher._assessment(f['reader'], f['assessment'], 3, 7)[3] == 73


@pytest.mark.parametrize('change', ['schema', 'producer', 'GP', 'registration', 'fitting', 'strain_GP',
                                   'local_material', 'micro'])
def test_endpoint_static_transfer_contract_mutations_fail(monkeypatch, change):
    f = _endpoint_documents(monkeypatch)
    reference = copy.deepcopy(f['reference'])
    if change == 'schema': reference['schema'] = 'yap.pitch_thickness_component_reuse.reference.v2'
    if change == 'producer': reference['actual_entry']['sha256'] = '0'*64
    if change == 'GP': reference['GP_regions'][0][2] -= 1
    if change == 'registration': reference['native_registration']['all259_trace_columns_checked'] = False
    if change == 'fitting': reference['congruence']['fitting'] = True
    if change == 'strain_GP': reference['congruence']['all_GP_strain']['si']['Cu_points'] -= 1
    if change == 'local_material': reference['fresh_equivalence_certificates']['si']['local_FE_input_equivalence']['complete_material_mask_equal'] = False
    if change == 'micro': reference['native_micro_coordinates'] -= 1
    with pytest.raises(ValueError):
        publisher._geometry_reference(f['reader'], reference, 3, 7, publisher.ENDPOINT_CASES[3, 7])


@pytest.mark.parametrize('change', ['source', 'static', 'fresh_projection', 'GP', 'artifact', 'certificate'])
def test_endpoint_selected_construction_mutations_fail(monkeypatch, change):
    f = _endpoint_documents(monkeypatch)
    operator = copy.deepcopy(f['operator'])
    if change == 'source': operator['source_sha256'] = '0'*64
    if change == 'static': operator['actual_static_tests']['sha256'] = '0'*64
    if change == 'fresh_projection': operator['fresh_projection'] = True
    if change == 'GP': operator['GP_regions'][0][2] -= 1
    if change == 'artifact': operator['components']['si']['artifact']['sha256'] = '0'*64
    if change == 'certificate': operator['components']['si']['projection_certificate']['all_GP_strain']['relative_error'] = 1.
    original = f['reader'].read

    def read(value):
        document, observed = original(value)
        if observed['path'] == f['history']['context']['operator']['path']:
            document = operator
        return document, observed

    monkeypatch.setattr(f['reader'], 'read', read)
    with pytest.raises(ValueError):
        publisher._components(f['reader'], f['history'], 3, 7, 73)


@pytest.mark.parametrize('change', ['native_geometry', 'native_state', 'native_GP', 'native_count',
    'native_checkpoint', 'native_engine', 'native_wrapper', 'ROM_source', 'ROM_count', 'ROM_JVP',
    'ROM_clock', 'ROM_state', 'ROM_implementation', 'missing_config'])
def test_endpoint_full_history_mutations_fail(monkeypatch, change):
    f = _endpoint_documents(monkeypatch)
    if change == 'native_geometry': f['native']['context']['case'] = publisher.ENDPOINT_CASES[3, 87]
    if change == 'native_state': f['native']['context']['Cu_state_adoption'] = True
    if change == 'native_GP': f['native']['Cu_points'] -= 1
    if change == 'native_count': f['native']['states'] = 1080
    if change == 'native_checkpoint': f['native']['checkpoints'][1]['no_native_state_adopted'] = False
    if change == 'native_engine': f['native']['source_sha256'] = publisher.ENDPOINT_PINS['pitch_thickness_endpoint_native_dense_20261022.py']
    if change == 'native_wrapper': f['native']['context']['history_wrapper_sha256'] = '0'*64
    if change == 'ROM_source': f['history']['source_sha256'] = publisher.PITCH_THICKNESS_ASSESSOR_SHA
    if change == 'ROM_count': f['history']['steps'].pop()
    if change == 'ROM_JVP': f['history']['certified_linear_solves'] -= 1
    if change == 'ROM_clock': f['history']['context']['selected_original_ordinals'][1] = 19
    if change == 'ROM_state': f['history']['context']['virgin_birth_fields_zero'] = False
    if change == 'ROM_implementation': f['history']['endpoint_rom_entry']['mathematical_wrapper']['sha256'] = '0'*64
    if change == 'missing_config': f['report']['lineage']['metadata_inputs'].clear()
    # Re-pin the synthetic documents to exercise semantic, not byte, rejection.
    for pin in (f['assessment'], *(f['report']['lineage'][name] for name in ('rom', 'native_report', 'native_checkpoint'))):
        pin['sha256'] = hashlib.sha256(json.dumps(f['docs'][pin['path']], sort_keys=True,
                                                allow_nan=False).encode('ascii')).hexdigest()
    f['assessment']['sha256'] = hashlib.sha256(json.dumps(f['report'], sort_keys=True,
                                                        allow_nan=False).encode('ascii')).hexdigest()
    with pytest.raises(ValueError):
        publisher._assessment(f['reader'], f['assessment'], 3, 7)


@pytest.mark.parametrize('count', range(5))
def test_endpoint_locators_are_optional_and_independently_incremental(monkeypatch, count):
    selected = _incremental_metadata(monkeypatch, ())
    original = publisher.MetadataReader.read
    pairs = list(publisher.ENDPOINT_CASES)[:count]

    def read(self, value):
        progress, pin = original(self, value)
        progress['audited_current_summary_20261021']['coverage']['pitch_thickness_passed'] = [
            dict(pitch_um=p, Si_TSV_um=si, evidence='synthetic') for p, si in pairs]
        return progress, pin

    monkeypatch.setattr(publisher.MetadataReader, 'read', read)
    entries = publisher.build_catalog()['entries']
    assert len(entries) == 13 + count
    assert [(p, si) for p, si, _ in selected[13:]] == pairs
    assert all(e['capabilities']['prepared_assembly_accuracy'] is False
               and e['capabilities']['W2W_thinning'] is False for e in entries)


def test_selected_endpoint_missing_evidence_is_actionable(monkeypatch):
    _incremental_metadata(monkeypatch, ())
    original = publisher.MetadataReader.read

    def read(self, value):
        progress, pin = original(self, value)
        progress['audited_current_summary_20261021']['coverage']['pitch_thickness_passed'] = [
            dict(pitch_um=3, Si_TSV_um=7)]
        return progress, pin

    monkeypatch.setattr(publisher.MetadataReader, 'read', read)
    with pytest.raises(ValueError, match='explicit completed bow evidence locator'):
        publisher.build_catalog()


def test_selected_endpoint_absent_report_fails_closed(monkeypatch):
    _incremental_metadata(monkeypatch, ())
    original = publisher.MetadataReader.read

    def read(self, value):
        progress, pin = original(self, value)
        progress['audited_current_summary_20261021']['coverage']['pitch_thickness_passed'] = [
            dict(pitch_um=3, Si_TSV_um=7, evidence='missing-endpoint-full-report')]
        return progress, pin

    def assessment(reader, receipt, pitch, si):
        if (pitch, si) in publisher.ENDPOINT_CASES:
            raise FileNotFoundError('selected full endpoint bow report does not exist')
        return {}, {}, {}, 73, 2., 3.

    monkeypatch.setattr(publisher.MetadataReader, 'read', read)
    monkeypatch.setattr(publisher, '_assessment', assessment)
    with pytest.raises(ValueError, match='BI3 Si7:.*full endpoint bow report does not exist'):
        publisher.build_catalog()


def test_current_public_catalog_not_extended_or_rewritten():
    path = publisher.ROOT/'distribution/operators/components/coverage_20261022_v2/CATALOG.json'
    before = path.read_bytes()
    catalog = json.loads(before)
    assert len(catalog['entries']) == 17
    assert not any((e['geometry']['bi_pitch_um'], e['geometry']['si_tsv_thickness_um'])
                   in publisher.ENDPOINT_CASES for e in catalog['entries'])
    publisher.build_catalog()
    assert path.read_bytes() == before


def _middle_bow(pitch=3, si=27):
    report = _endpoint_bow(pitch=3, si=7)
    report.update(key=f'p{pitch}_si{si}', case=publisher.MIDDLE_CASES[pitch, si])
    lineage = report['lineage']
    lineage.pop('endpoint_ROM_entry')
    lineage.pop('endpoint_ROM_tests')
    lineage.update(source_sha256=publisher.MIDDLE_ASSESSOR_PAIR[0],
        tests_sha256=publisher.MIDDLE_ASSESSOR_PAIR[1],
        middle_ROM_entry=publisher._middle_source('pitch_thickness_middle_rom_history_20261022.py'),
        middle_ROM_tests=publisher._middle_source('test_pitch_thickness_middle_rom_history_20261022.py'),
        additional_source_pins=dict(publisher.ENDPOINT_PINS, **publisher.MIDDLE_PINS),
        endpoint_assessment_implementation=dict(
            path=str(publisher.LEGACY_EXTERIOR_ROOT.parent/'pitch_thickness_endpoint_bow_assessment_20261022.py'),
            sha256=publisher.ENDPOINT_ASSESSOR_PAIR[0], tests_sha256=publisher.ENDPOINT_ASSESSOR_PAIR[1],
            execution_entry=False),
        middle_authentication_AST=dict(inverse_full_module_exact=True, changed_expressions=2, receipt_keyword_edits=2))
    return report


def _middle_documents(monkeypatch, pitch=3, si=27, as_gib=16):
    """Real static evidence template; ALL modified/future histories are synthetic."""
    reader = publisher.MetadataReader()
    root = publisher.LEGACY_EXTERIOR_ROOT
    operator, _ = reader.read(dict(path=str(root/
        'PITCH_THICKNESS_MIDDLE_OPERATOR_COMPOSITION_20261022_p3_si27_mfd_v2/REPORT.json'),
        sha256='b6e8f7bf914af1a926f6e2cf06d7faabca85dba34cbf5ad52ef20173815d9f25'))
    reference, _ = reader.read(operator['reference'])
    target, _ = reader.read(reference['native_report'])
    static_config, _ = reader.read(operator['static_configuration_binding'])
    operator, reference, target, static_config = copy.deepcopy((operator, reference, target, static_config))
    report = _middle_bow(pitch, si)
    case, key = report['case'], report['key']
    first = 18816 if pitch == 3 else 19248
    regions = [['bi', 0, first], ['beol', first, first+102096], ['si', first+102096, first+166256]]
    for d in (operator, reference, target, static_config):
        d.update(case=case, key=key, BI_pitch_um=pitch, Si_um=si)
        if 'GP_regions' in d:
            d['GP_regions'] = regions
    for role, start, stop in regions:
        reference['congruence']['all_GP_strain'][role]['Cu_points'] = stop-start
    operator['GP_regions'] = regions
    operator['target_congruence'] = reference['congruence']
    docs = {}

    def add(path, value):
        docs[str(path)] = value
        raw = json.dumps(value, sort_keys=True, allow_nan=False).encode('ascii')
        return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))

    base = root/('SYNTHETIC_MIDDLE_PUBLICATION_NOT_EVIDENCE_'+key)
    target_pin = add(base/'raw/REPORT.json', target)
    reference['native_report'] = static_config['native_report'] = target_pin
    config_pin = add(base/'static/CONFIGURATION.json', static_config)
    clean_static = {k:v for k,v in static_config.items() if k not in ('main_go','host')}
    reference['configuration_sha256'] = publisher._semantic_digest(clean_static)
    config_pin['semantic_sha256'] = reference['configuration_sha256']
    reference_pin = add(base/'static/REPORT.json', reference)
    operator['reference'] = operator['registration'] = reference_pin
    operator['static_configuration_binding'] = config_pin
    rec = operator['static_configuration_reconciliation']
    rec['recorded_configuration'] = config_pin
    rec['fresh_configuration_sha256'] = reference['configuration_sha256']
    operator_pin = add(base/'composition/REPORT.json', operator)
    c = dict(case=case, operator=operator_pin, reference=reference_pin,
        topology=reference['topology_artifact'], operator_payloads={r:v['artifact'] for r,v in operator['components'].items()},
        micro_ranks={r:v['micro_rank'] for r,v in operator['components'].items()},
        global_macro_port_coordinates=259, all_GP_retained=True, Cu_state_adoption=False,
        virgin_birth_fields_zero=True, external_checkpoint_reads=0, Cu_GP=regions[-1][-1],
        selected_original_ordinals=list(publisher.GRADED73_ORDINALS), **operator['training']['context'])
    n = dict(case=case, reference=reference['reference'], native_elastic_report=target_pin,
        selected_original_ordinals=list(range(1081)), GP_regions=regions, clock_states=1081,
        temporal_profile='dense_original1081', MAPDL_reads=0, component_reference_used=False,
        component_certificate_dependency=False, ROM_dependency=False, virgin_birth_fields_zero=True,
        Cu_state_adoption=False, **operator['training']['context'])
    n.update(history_wrapper_sha256=publisher.ENDPOINT_IMPLEMENTATION_PINS['pitch_thickness_native_dense_20261022.py'],
        history_tests_sha256=publisher.ENDPOINT_IMPLEMENTATION_PINS['test_pitch_thickness_native_dense_20261022.py'])
    history = dict(schema='yap.component_reduced_clock.history.v1', state='CLOSED_REDUCED_CLOCK_VIRGIN_HISTORY_UNSCORED',
        key=key, case=case, source_sha256=publisher.MIDDLE_PINS['pitch_thickness_middle_rom_history_20261022.py'],
        tests_sha256=publisher.MIDDLE_PINS['test_pitch_thickness_middle_rom_history_20261022.py'],
        states=73, initial_ordinal=0, final_ordinal=72, final_original_ordinal=1080, context=c,
        MAPDL_result_reads=0, external_checkpoint_reads=0, no_MAPDL_state_adopted=True,
        certified_linear_solves=72, steps=[dict(path=f'synthetic-step-{i}') for i in range(72)],
        address_space_bytes=as_gib*1024**3, AS_trial_profile=f'AS{as_gib}',
        completed_history_under_executable_AS=True,
        checkpoints=[dict(ordinal=publisher.GRADED73_ORDINALS.index(i), original_ordinal=i,
            context=c, all_Cu14=True, no_MAPDL_state_adopted=True) for i in publisher.PHASE_ORDINALS])
    history['checkpoints'][-1]['payload'] = dict(path=str(base/'rom/CHECKPOINT_000072.npz'), sha256='1'*64)
    native = dict(state='CLOSED_PITCH_THICKNESS_NATIVE_DENSE1081_UNSCORED', states=1081, case=case,
        context=n, Cu_points=regions[-1][-1], interior_coordinates=273372,
        source_sha256=publisher.ENDPOINT_IMPLEMENTATION_PINS['component_fullspace_validation.py'],
        checkpoints=[dict(ordinal=i, context=n, all_Cu14=True, no_native_state_adopted=True)
                     for i in publisher.PHASE_ORDINALS])
    metadata = {}
    for label, stem, cap, extra in (
        ('rom', 'pitch_thickness_middle_rom_history_20261022', as_gib, dict(operator=operator_pin)),
        ('native', 'pitch_thickness_middle_native_dense_20261022', 32,
         dict(native_report=target_pin, native=reference['reference'], Cu_state_adoption=False))):
        source, tests = (publisher._middle_source(v) for v in (stem+'.py', 'test_'+stem+'.py'))
        readset = {publisher._middle_source(k)['path']:v for k,v in publisher.MIDDLE_PINS.items()}
        config = dict(schema='yap.'+stem.replace('_20261022','')+'.configuration.v1',
            key=key, case=case, source_sha256=source['sha256'], tests_sha256=tests['sha256'],
            GP_regions=regions, Cu_GP=regions[-1][-1], thermal_policy=c['thermal_policy'],
            virgin_birth_fields_zero=True, external_checkpoint_reads=0,
            readset=readset, readset_sha256=publisher._semantic_digest(readset),
            limits=dict(address_space_bytes=cap*1024**3, minimum_available_bytes=(cap+32)*1024**3),
            AS_trial_profile=f'AS{cap}', **extra)
        digest = publisher._semantic_digest(config)
        context = dict(source=source, tests=tests, configuration_sha256=digest)
        if label == 'rom':
            context.update(operator=operator_pin, public_ready=False, native_history_read=False)
            for field,name in (('mathematical_wrapper','pitch_thickness_reduced_history_20261022.py'),
                               ('admission_implementation','pitch_thickness_reduced_profiles_20261022.py')):
                context[field] = dict(path=str(root.parent/name), sha256=publisher.ENDPOINT_IMPLEMENTATION_PINS[name])
            history.update(configuration_sha256=digest, middle_rom_entry=context)
            name = 'CONFIGURATION.json'
        else:
            context['raw_native_report'] = target_pin
            n['middle_native_entry'] = context
            native['dense1081_configuration_sha256'] = digest
            name = 'DENSE1081_CONFIGURATION.json'
        pin = add(base/label/name, config)
        metadata[pin['path']] = pin
    report['lineage'].update(rom=add(base/'rom/REPORT.json',history),
        native_report=add(base/'native/REPORT.json',native),
        native_checkpoint=add(base/'native/CHECKPOINT_001080.json',native['checkpoints'][-1]), metadata_inputs=metadata)
    assessment = add(base/'assessment/REPORT.json',report)
    original = reader.read

    def read(value):
        path = value['path'] if isinstance(value,dict) else str(value)
        if path not in docs:
            return original(value)
        pin = add(path,docs[path])
        if isinstance(value,dict):
            publisher.same_binding(value,pin)
        return docs[path],pin

    monkeypatch.setattr(reader,'read',read)
    return dict(reader=reader, report=report, history=history, native=native, reference=reference,
                operator=operator, static_config=static_config, docs=docs, assessment=assessment, add=add)


@pytest.mark.parametrize('pitch,si', publisher.MIDDLE_CASES)
def test_middle8_full_metadata_contract(monkeypatch,pitch,si):
    f = _middle_documents(monkeypatch,pitch,si)
    assert publisher._assessment(f['reader'],f['assessment'],pitch,si)[3] == 73
    assert set(publisher._components(f['reader'],f['history'],pitch,si,73)[0]) == set(publisher.ROLES)


@pytest.mark.parametrize('as_gib',[16,32])
def test_middle_explicit_ROM_profiles(monkeypatch,as_gib):
    f = _middle_documents(monkeypatch,as_gib=as_gib)
    assert publisher._assessment(f['reader'],f['assessment'],3,27)[3] == 73


def test_middle_actual_closed_ROM_configuration_matches_publication_contract(monkeypatch):
    f = _middle_documents(monkeypatch)
    reader = publisher.MetadataReader()
    root = publisher.LEGACY_EXTERIOR_ROOT
    history, hp = reader.read(dict(path=str(root/
        'PITCH_THICKNESS_MIDDLE_ROM_HISTORY_20261022_p3_si27_mfd_AS16_v1/REPORT.json'),
        sha256='ab851a715843be5900a7e799a6d4fa1e55094e8ed27f97935e6a26290b0b8439'))
    operator, _ = reader.read(history['context']['operator'])
    reference, _ = reader.read(history['context']['reference'])
    saved, pin = reader.read(str(Path(hp['path']).parent/'CONFIGURATION.json'))
    f['report']['lineage']['metadata_inputs'][pin['path']] = pin
    f['report']['lineage']['rom'] = hp
    f['report']['ROM_strict_linear_certificates'] = history['certified_linear_solves']
    n = f['native']['context']
    n['reference'] = reference['reference']
    n['native_elastic_report'] = reference['native_report']
    np = f['report']['lineage']['native_report']
    config_path = str(Path(np['path']).parent/'DENSE1081_CONFIGURATION.json')
    config = f['docs'][config_path]
    config.update(native=reference['reference'], native_report=reference['native_report'])
    digest = publisher._semantic_digest(config)
    n['middle_native_entry'].update(raw_native_report=reference['native_report'], configuration_sha256=digest)
    f['native']['dense1081_configuration_sha256'] = digest
    f['report']['lineage']['metadata_inputs'][config_path] = f['add'](config_path,config)
    # Only the native closure remains synthetic; this does not score or admit it.
    publisher._endpoint_histories(f['reader'],f['report'],history,f['native'],reference,3,27,middle=True)
    assert set(publisher._components(reader,history,3,27,73)[0]) == set(publisher.ROLES)
    assert operator['fresh_projection'] is False


@pytest.mark.parametrize('count',range(9))
def test_middle_locators_preserve_all_existing24_entries(monkeypatch,count):
    selected = _incremental_metadata(monkeypatch,publisher.P2_EXTERIOR_SI)
    original = publisher.MetadataReader.read
    old_pairs = [(p,si) for p in (3,6) for si in (17,77,7,87)]
    pairs = list(publisher.MIDDLE_CASES)[:count]

    def read(self,value):
        progress,pin = original(self,value)
        progress['audited_current_summary_20261021']['coverage']['pitch_thickness_passed'] = [
            dict(pitch_um=p,Si_TSV_um=si,evidence='synthetic') for p,si in old_pairs]
        return progress,pin

    monkeypatch.setattr(publisher.MetadataReader,'read',read)
    baseline = publisher.build_catalog()['entries']
    assert len(baseline) == 24

    def extended(self,value):
        progress,pin = read(self,value)
        progress['audited_current_summary_20261021']['coverage']['pitch_thickness_passed'] += [
            dict(pitch_um=p,Si_TSV_um=si,evidence='synthetic') for p,si in pairs]
        return progress,pin

    monkeypatch.setattr(publisher.MetadataReader,'read',extended)
    entries = publisher.build_catalog()['entries']
    assert entries[:24] == baseline and len(entries) == 24+count
    assert [(p,si) for p,si,_ in selected[-count:]] == pairs if count else entries == baseline


@pytest.mark.parametrize('change',['prefix','closure','solved_count','native_count','probe','clock',
    'producer','native_engine','native_wrapper','ROM_source','ROM_steps','JVP','case','missing_config'])
def test_middle_history_and_bow_mutations_fail(monkeypatch,change):
    f = _middle_documents(monkeypatch)
    if change == 'prefix': f['report'].update(through_original_ordinal=720,prefix_only=True,qualification_passed=None)
    if change == 'closure': f['report']['full_native1081_closure_verified'] = False
    if change == 'solved_count': f['report']['actual_ROM_solved_states_compared'] = 49
    if change == 'native_count': f['native']['states'] = 1080
    if change == 'probe': f['report']['probe_names'].pop()
    if change == 'clock': f['history']['context']['selected_original_ordinals'][1] = 19
    if change == 'producer': f['report']['lineage']['source_sha256'] = publisher.ENDPOINT_ASSESSOR_PAIR[0]
    if change == 'native_engine': f['native']['source_sha256'] = publisher.MIDDLE_PINS['pitch_thickness_middle_native_dense_20261022.py']
    if change == 'native_wrapper': f['native']['context']['history_wrapper_sha256'] = '0'*64
    if change == 'ROM_source': f['history']['source_sha256'] = publisher.ENDPOINT_PINS['pitch_thickness_endpoint_rom_history_20261022.py']
    if change == 'ROM_steps': f['history']['steps'].pop()
    if change == 'JVP': f['history']['certified_linear_solves'] -= 1
    if change == 'case': f['native']['case'] = publisher.MIDDLE_CASES[6,27]
    if change == 'missing_config': f['report']['lineage']['metadata_inputs'].clear()
    for name in ('rom','native_report','native_checkpoint'):
        pin = f['report']['lineage'][name]
        pin.update(f['add'](pin['path'],f['docs'][pin['path']]))
    f['assessment'].update(f['add'](f['assessment']['path'],f['report']))
    with pytest.raises(ValueError):
        publisher._assessment(f['reader'],f['assessment'],3,27)


def test_middle_pins_are_exact_N_sources():
    for name,pin in publisher.MIDDLE_PINS.items():
        path = publisher.MIDDLE_ROOT/name
        assert publisher._middle_source(name)['path'] == str(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pin


def _rebind_middle_documents(f):
    """Re-sign every affected fixture binding, never the original donor data."""
    config, reference, operator = f['static_config'], f['reference'], f['operator']
    config['readset_sha256'] = publisher._semantic_digest(config['readset'])
    reference['source_readset'] = copy.deepcopy(config['readset'])
    reference['source_readset_sha256'] = config['readset_sha256']
    binding = operator['static_configuration_binding']
    binding.update(f['add'](binding['path'], config))
    clean = {k:v for k,v in config.items() if k not in ('main_go','host')}
    binding['semantic_sha256'] = reference['configuration_sha256'] = publisher._semantic_digest(clean)
    reconciliation = operator['static_configuration_reconciliation']
    fresh = copy.deepcopy(clean)
    for row in reconciliation['device_observations']:
        _, _, role, kind, _, _ = row['path'].split('/')
        fresh['declared_donors'][role][kind]['current_identity'][0] = row['fresh']
    reconciliation['fresh_configuration_sha256'] = publisher._semantic_digest(fresh)
    reconciliation['recorded_configuration'] = binding
    operator['reference'].update(f['add'](operator['reference']['path'], reference))
    operator['readset_sha256'] = publisher._semantic_digest(operator['readset'])
    f['history']['context']['operator'].update(f['add'](
        f['history']['context']['operator']['path'], operator))
    for label, document, entry, digest_key, filename in (
        ('rom', f['history'], f['history']['middle_rom_entry'], 'configuration_sha256', 'CONFIGURATION.json'),
        ('native', f['native'], f['native']['context']['middle_native_entry'],
         'dense1081_configuration_sha256', 'DENSE1081_CONFIGURATION.json')):
        descriptor = f['report']['lineage']['rom' if label == 'rom' else 'native_report']
        path = str(Path(descriptor['path']).parent/filename)
        current = f['docs'][path]
        current['readset_sha256'] = publisher._semantic_digest(current['readset'])
        document[digest_key] = entry['configuration_sha256'] = publisher._semantic_digest(current)
        f['report']['lineage']['metadata_inputs'][path].update(f['add'](path,current))
    for name in ('rom','native_report','native_checkpoint'):
        pin = f['report']['lineage'][name]
        pin.update(f['add'](pin['path'],f['docs'][pin['path']]))
    f['assessment'].update(f['add'](f['assessment']['path'],f['report']))


@pytest.mark.parametrize('role',['rom','native','static','composition'])
@pytest.mark.parametrize('mutation',['unknown_path','known_wrong_SHA'])
def test_middle_rebound_N_source_injections_rejected(monkeypatch,role,mutation):
    f = _middle_documents(monkeypatch)
    if role in ('rom','native'):
        descriptor = f['report']['lineage']['rom' if role == 'rom' else 'native_report']
        filename = 'CONFIGURATION.json' if role == 'rom' else 'DENSE1081_CONFIGURATION.json'
        readset = f['docs'][str(Path(descriptor['path']).parent/filename)]['readset']
    else:
        readset = f['static_config' if role == 'static' else 'operator']['readset']
    path = publisher.MIDDLE_ROOT/('UNREVIEWED_SOURCE.py' if mutation == 'unknown_path' else
                                'pitch_thickness_middle_bow_assessment_20261022.py')
    readset[str(path)] = '0'*64
    _rebind_middle_documents(f)
    with pytest.raises(ValueError,match='N source'):
        if role in ('rom','native'):
            publisher._assessment(f['reader'],f['assessment'],3,27)
        else:
            publisher._components(f['reader'],f['history'],3,27,73)


def test_middle_synchronized_donor_flag_deletion_rejected_against_original_config(monkeypatch):
    f = _middle_documents(monkeypatch)
    original = copy.deepcopy(f['static_config']['declared_donors'])
    for donor in (f['operator']['per_role_training_provenance']['si'],
                  f['operator']['training']['per_role_provenance']['si'],
                  f['reference']['declared_donors']['si']):
        donor.pop('borrowed_load_training_flag_present')
    assert f['static_config']['declared_donors'] == original
    _rebind_middle_documents(f)
    with pytest.raises(ValueError,match='middle static execution configuration/report differs'):
        publisher._components(f['reader'],f['history'],3,27,73)


def _middle_cross_host_donors(monkeypatch):
    f = _middle_documents(monkeypatch)
    reconciliation = f['operator']['static_configuration_reconciliation']
    reconciliation.update(recorded_host='mfd.ee.ucla.edu', fresh_host='sysml.ee.ucla.edu')
    for row in reconciliation['device_observations']:
        _, _, role, kind, _, _ = row['path'].split('/')
        row.update(fresh=row['recorded'] + 7, changed=True)
        for provenance in (f['operator']['per_role_training_provenance'],
                           f['operator']['training']['per_role_provenance']):
            provenance[role][kind]['current_identity'][0] = row['fresh']
    _rebind_middle_documents(f)
    return f


def test_middle_cross_host_donor_devices_match_authenticated_fresh_view(monkeypatch):
    f = _middle_cross_host_donors(monkeypatch)
    before = copy.deepcopy(f['static_config']['declared_donors'])
    assert set(publisher._components(f['reader'], f['history'], 3, 27, 73)[0]) == set(publisher.ROLES)
    assert f['static_config']['declared_donors'] == before
    assert f['reference']['declared_donors'] == before


@pytest.mark.parametrize('change', ['device', 'inode', 'missing_flag', 'host'])
def test_middle_cross_host_reconciliation_cannot_hide_other_provenance_changes(monkeypatch, change):
    f = _middle_cross_host_donors(monkeypatch)
    for provenance in (f['operator']['per_role_training_provenance'],
                       f['operator']['training']['per_role_provenance']):
        donor = provenance['si']
        if change == 'device': donor['operator_artifact']['current_identity'][0] += 1
        if change == 'inode': donor['operator_artifact']['current_identity'][1] += 1
        if change == 'missing_flag': donor.pop('borrowed_load_training_flag_present')
    if change == 'host':
        f['operator']['static_configuration_reconciliation']['fresh_host'] = 'mfd.ee.ucla.edu'
    _rebind_middle_documents(f)
    with pytest.raises(ValueError):
        publisher._components(f['reader'], f['history'], 3, 27, 73)


@pytest.mark.parametrize('si',[47,57])
def test_middle_originally_absent_donor_flags_are_not_fabricated(monkeypatch,si):
    f = _middle_documents(monkeypatch,si=si)
    # Explicitly synthetic original source evidence: absence is preserved in
    # the saved config, static report and both construction provenance views.
    for donor in (f['static_config']['declared_donors']['si'], f['reference']['declared_donors']['si'],
                  f['operator']['per_role_training_provenance']['si'],
                  f['operator']['training']['per_role_provenance']['si']):
        donor.pop('borrowed_load_training',None)
        donor['borrowed_load_training_flag_present'] = False
    _rebind_middle_documents(f)
    before = copy.deepcopy(f['static_config']['declared_donors'])
    assert set(publisher._components(f['reader'],f['history'],3,si,73)[0]) == set(publisher.ROLES)
    assert f['static_config']['declared_donors'] == before
    assert 'borrowed_load_training' not in f['operator']['per_role_training_provenance']['si']


def test_middle_actual_P3Si27_full_pass_remains_valid_metadata_only():
    reader = publisher.MetadataReader()
    report = dict(path=str(publisher.LEGACY_EXTERIOR_ROOT/
        'PITCH_THICKNESS_MIDDLE_BOW_ASSESSMENT_20261022_p3_si27_through1080_v1/REPORT.json'),
        sha256='7636a331f4e6c64c561ed4c2cad6e356c1c089ee3ce1a79996f82a59dafae5dc')
    _,history,_,states,_,_ = publisher._assessment(reader,report,3,27)
    assert states == 73
    assert set(publisher._components(reader,history,3,27,states)[0]) == set(publisher.ROLES)
    assert all(Path(path).suffix == '.json' for path in reader.cache)


@pytest.mark.parametrize('change',['extra_device','inode','material','source','missing_training_flag'])
def test_middle_static_configuration_and_training_mutations_fail(monkeypatch,change):
    f = _middle_documents(monkeypatch)
    if change == 'extra_device': f['operator']['static_configuration_reconciliation']['device_observations'].append(
        dict(path='/native/current_identity/0',recorded=62,fresh=55,changed=True))
    if change == 'inode': f['static_config']['declared_donors']['si']['operator_artifact']['current_identity'][1] += 1
    if change == 'material': f['static_config']['Cu_updates'] = 1
    if change == 'source': f['static_config']['source_sha256'] = '0'*64
    if change == 'missing_training_flag': f['operator']['training']['per_role_provenance']['si'].pop('borrowed_load_training_flag_present')
    with pytest.raises(ValueError):
        publisher._components(f['reader'],f['history'],3,27,73)


def test_cumulative_eighth_supplement_preserves_old_keys_without_mutation():
    old = dict(schema='yap.operator_artifact_locations.v1',files=['operator_repair_20261015/old/REPORT.json'])
    saved = copy.deepcopy(old)
    new = ['operator_repair_20261015/new/REPORT.json']
    merged = publisher.cumulative_artifact_location_document(old,new,{'operator_repair_20261015/base/REPORT.json'})
    assert set(merged['files']) == set(old['files']) | set(new)
    assert old == saved


@pytest.mark.parametrize('extra,earlier',[
    (['operator_repair_20261015/old/REPORT.json'],set()),
    (['operator_repair_20261015/new/REPORT.json'],{'operator_repair_20261015/old/REPORT.json'}),
    (['operator_repair_20261015/../bad/REPORT.json'],set()),
    (['operator_repair_20261015/#snapshot/REPORT.json'],set()),
    (['operator_night_20261021/unknown.py'],set())])
def test_cumulative_map_overlap_traversal_and_unreviewed_root_refused(extra,earlier):
    old = dict(schema='yap.operator_artifact_locations.v1',files=['operator_repair_20261015/old/REPORT.json'])
    with pytest.raises(ValueError):
        publisher.cumulative_artifact_location_document(old,extra,earlier)
