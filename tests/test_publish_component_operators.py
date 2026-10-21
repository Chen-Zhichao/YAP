"""Metadata-only publication coverage; no operators or solver APIs are loaded."""

import copy
import hashlib
import json
from pathlib import Path
import stat

import pytest

import publish_component_operators as publisher


@pytest.fixture(scope='module')
def actual_catalog():
    if not (publisher.ROOT / publisher.PROGRESS_RELATIVE).is_file():
        pytest.skip('canonical component evidence is not installed')
    return publisher.build_catalog()


def test_actual_thirteen_entries_are_exact_and_metadata_only(monkeypatch):
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
    expected = [f'BI2_SI{si:g}'.replace('.', 'p') for si in publisher.P2_SI]
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
        assert entry['validation']['solved_states'] == (73 if entry['id'] in expected[:4] else 61)
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
    original = publisher.MetadataReader.read

    def read(self, value):
        document, pin = original(self, value)
        if pin['path'].endswith(publisher.PROGRESS_RELATIVE):
            document = copy.deepcopy(document)
            for row in document['audited_current_summary_20261021']['coverage']['single_BI_p2_full_three_cycle_passed']:
                row.update(full_three_cycle_bow_passed=False, max_relative_pct=999., actual_ROM_states=1)
        return document, pin

    monkeypatch.setattr(publisher.MetadataReader, 'read', read)
    assert len(publisher.build_catalog()['entries']) == 13


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
