"""Shared qualification wiring only; no real operand load or accuracy claim."""
from copy import deepcopy
from pathlib import Path
import os
from types import SimpleNamespace as NS

import pytest

import component_operators as components
from tools import publish_component_operators as publisher
from W2W import assembly_mechanics as mechanics
from W2W import component_assembly_bridge as bridge
from W2W import component_compiled_material as compiled
from W2W import trace80_component_bundle as bundle


@pytest.fixture
def builder(monkeypatch, tmp_path):
    calls = []
    entry = dict(id='BI4_SI37', profile=components.TRACE80_PROFILE,
                 validation={'software_fixture_only': True}, geometry={'unit_fixture_only': True})
    geometry = {'requested_fixture_only': True}
    trace = object()
    operands = {'bi': {'physical_trace': trace}}
    certificate, inputs = {'software_fixture_only': True}, {'inputs_fixture_only': True}
    library, card_api = object(), object()
    exact, reduced = object(), object()
    modules = dict(exp12_material_card_v2_kim2026=card_api,
                   exp12_all_gauss_exact_acceleration=exact, exp12_all_gauss_reduced_fe2=reduced)
    def native(name):
        calls.append(('native', name))
        return modules[name]
    monkeypatch.setattr(mechanics, 'native', native)
    parameters = {'real_parser_output_placeholder': True}
    card = NS(sha256=components.CU_MATERIAL_CARD_SHA256, python_cu_card_kwargs=lambda:parameters)
    def pinned(api):
        assert api is card_api
        calls.append(('pinned_card',))
        return card
    monkeypatch.setattr(components, '_pinned_cu_card', pinned)
    material = NS(cu_card=object(), material_kernels=object(), binding=object())
    def compile_material(**kwargs):
        assert kwargs == dict(cu_source_path=compiled.SOURCE_ROOT/'exp12_cu_material_point.py',
            cu_source_sha256=compiled.SOURCE_PINS['exp12_cu_material_point.py'],
            acceleration_source_path=compiled.SOURCE_ROOT/'exp12_all_gauss_exact_acceleration.py',
            acceleration_source_sha256=compiled.SOURCE_PINS['exp12_all_gauss_exact_acceleration.py'],
            native_exact_module=exact, reduced_module=reduced,
            cache_directory=tmp_path, card_parameters=parameters)
        calls.append(('compiled_material',))
        return material
    monkeypatch.setattr(compiled, 'create_compiled_material', compile_material)
    monkeypatch.setenv('NUMBA_CACHE_DIR', str(tmp_path))
    proof = {'software_fixture_only': True}
    def build(actual, **kwargs):
        assert actual is operands
        assert kwargs == dict(geometry=geometry, certificate=certificate,
            expected_certificate_sha256=bridge._sha256_document(certificate), cu_card=material.cu_card,
            material_kernels=material.material_kernels, material_binding=material.binding)
        calls.append(('bridge',))
        return library, proof
    monkeypatch.setattr(bridge, 'build_component_assembly_library', build)
    monkeypatch.setattr(components.runtime, 'corrected_material_provenance',
                        lambda binding: {'same_binding': binding is material.binding})
    monkeypatch.setattr(components, 'configure_component_runtime', lambda:calls.append(('configure',)))
    monkeypatch.setattr(components.runtime, 'configure_runtime_storage', lambda:calls.append(('storage',)))
    return NS(calls=calls, entry=entry, geometry=geometry, trace=trace, operands=operands,
              certificate=certificate, inputs=inputs, library=library, proof=proof)


def test_shared_builder_uses_pinned_real_material_and_bridge_sequence(builder):
    f = builder
    library, provenance, trace = components._build_component_library_and_trace(
        f.entry, f.geometry, f.operands, f.certificate, f.inputs)
    assert library is f.library and trace is f.trace
    assert provenance['material_card']['sha256'] == components.CU_MATERIAL_CARD_SHA256
    assert provenance['physical_registration'] is f.proof
    assert provenance['corrected_material'] == {'same_binding': True}
    assert provenance['operator_inputs'] is f.inputs
    assert provenance['W2W_thinning_available'] is False
    assert [row[0] for row in f.calls] == ['native', 'pinned_card', 'native', 'native',
                                         'compiled_material', 'bridge']


@pytest.mark.parametrize('kind', ['trace80', 'legacy', 'family'])
def test_public_load_converts_once_then_uses_shared_real_builder(builder, monkeypatch, kind):
    f = builder
    if kind == 'legacy':
        f.entry.pop('profile')
        f.inputs = {role:object() for role in components.ROLES}
        f.entry['components'] = f.inputs
    elif kind == 'family':
        f.entry['kind'] = 'endpoint_thickness_family'
    monkeypatch.setattr(components, 'prepared_operator_request', lambda *a, **k:None)
    monkeypatch.setattr(components, 'resolve_entry', lambda package:f.entry)
    def convert(*args):
        assert f.calls == [('configure',), ('storage',)]
        f.calls.append(('convert',))
        return f.operands, f.certificate, f.inputs
    monkeypatch.setattr(bundle, 'load_operands', convert)
    monkeypatch.setattr(components, '_parameterized_operands', convert)
    monkeypatch.setattr(components, '_physical_operands', lambda *args:convert(*args)[:2])
    library, _, trace = components._component_library_and_trace('fixture', f.geometry)
    assert library is f.library and trace is f.trace
    assert f.calls.count(('convert',)) == f.calls.count(('bridge',)) == 1


def test_qualification_converts_once_without_public_resolution_or_reload(builder, monkeypatch):
    f = builder
    operand_reader, metadata_reader = object(), object()
    guards = []
    def check():
        guards.append('checked')
    def convert(entry, geometry, reader, metadata, *, check):
        assert (entry, geometry) == (f.entry, f.geometry)
        assert reader is operand_reader and metadata is metadata_reader
        assert f.calls == [('configure',), ('storage',)]
        check(); f.calls.append(('convert',))
        return f.operands, f.certificate, f.inputs, 1234
    monkeypatch.setattr(bundle, '_qualification_operands', convert)
    monkeypatch.setattr(bundle, 'load_operands', lambda *a, **k:pytest.fail('public numeric reload'))
    monkeypatch.setattr(components, 'resolve_entry', lambda *a, **k:pytest.fail('draft catalog resolution'))
    library, _, trace, forecast = components._qualification_library_and_trace(
        f.entry, f.geometry, operand_reader, metadata_reader, check=check)
    assert library is f.library and trace is f.trace and forecast == 1234
    assert len(guards) == 4
    assert f.calls.count(('convert',)) == f.calls.count(('bridge',)) == 1


def test_qualification_rejects_missing_guard_before_runtime_setup(builder):
    f = builder
    with pytest.raises(ValueError, match='guard'):
        components._qualification_library_and_trace(f.entry, f.geometry, None, None, check=None)
    assert not f.calls


def test_qualification_passes_allocation_budget_to_the_shared_converter(builder, monkeypatch):
    f = builder
    budget = 48 * 1024**3
    seen = []

    def convert(entry, geometry, reader, metadata, *, check, maximum_forecast):
        assert maximum_forecast == budget
        seen.append(maximum_forecast)
        return f.operands, f.certificate, f.inputs, 1234

    monkeypatch.setattr(bundle, '_qualification_operands', convert)
    library, _, _, forecast = components._qualification_library_and_trace(
        f.entry, f.geometry, object(), object(), check=lambda:None, maximum_forecast=budget)
    assert library is f.library and forecast == 1234
    assert seen == [budget]


@pytest.mark.parametrize('budget', [0, -1, True, 1.0, '1'])
def test_qualification_rejects_bad_allocation_budget_before_setup(builder, budget):
    f = builder
    with pytest.raises(ValueError, match='allocation budget'):
        components._qualification_library_and_trace(
            f.entry, f.geometry, None, None, check=lambda:None, maximum_forecast=budget)
    with pytest.raises(ValueError, match='allocation budget'):
        bundle._qualification_operands(
            f.entry, f.geometry, None, None, check=lambda:None, maximum_forecast=budget)
    assert not f.calls


def test_qualification_operand_path_forwards_budget_to_preallocation_check(monkeypatch):
    entry, geometry, reader, metadata = object(), object(), object(), object()
    manifest, proofs = object(), object()
    budget = 48 * 1024**3
    seen = []
    monkeypatch.setattr(bundle, '_export', lambda e, m:(manifest, proofs))

    def convert(e, g, r, m, actual_manifest, actual_proofs, *, check, maximum_forecast):
        assert (e, g, r, m, actual_manifest, actual_proofs) == (
            entry, geometry, reader, metadata, manifest, proofs)
        seen.append(maximum_forecast)
        raise ValueError('expanded allocation exceeds qualified envelope before GP allocation')

    monkeypatch.setattr(bundle, '_convert', convert)
    with pytest.raises(ValueError, match='before GP allocation'):
        bundle._qualification_operands(
            entry, geometry, reader, metadata, check=lambda:None, maximum_forecast=budget)
    assert seen == [budget]


@pytest.fixture
def draft(monkeypatch):
    bind = lambda name:dict(path=str(publisher.ROOT/'runtime_checks_20261025'/name), sha256='a'*64)
    locator = dict(pitch_um=4, Si_TSV_um=37, evidence=bind('evidence.json'),
                   physical_geometry=bind('compact.json'))
    calls = []
    def assessment(reader, evidence, pitch, si):
        assert evidence == locator['evidence'] and (pitch, si) == (4, 37)
        calls.append('assessment')
        return bind('assessment.json'), {'history_fixture_only': True}, bind('history.json'), 73, .4, .9
    def component_records(reader, history, pitch, si, states):
        assert history == {'history_fixture_only': True} and (pitch, si, states) == (4, 37, 73)
        calls.append('components')
        return {role:bind(role+'.npz') for role in publisher.ROLES}, bind('construction.json'), bind('reference.json')
    monkeypatch.setattr(publisher, '_assessment', assessment)
    monkeypatch.setattr(publisher, '_components', component_records)
    return NS(locator=locator, calls=calls, bind=bind)


def test_internal_draft_has_no_invented_qualification_and_stable_identity(draft):
    f = draft
    entry = publisher._trace80_entry_draft(object(), f.locator)
    assert f.calls == ['assessment', 'components']
    assert 'runtime_qualification' not in entry['validation']
    assert entry['validation']['solved_states'] == 73
    assert entry['capabilities']['prepared_assembly_accuracy'] is False
    assert entry['capabilities']['W2W_thinning'] is False
    qualified = deepcopy(entry)
    qualified['validation']['runtime_qualification'] = f.bind('qualification.json')
    assert bundle.entry_identity(entry) == bundle.entry_identity(qualified)


def test_public_entry_still_requires_and_validates_qualification(draft, monkeypatch):
    f = draft
    with pytest.raises(ValueError, match='runtime evidence'):
        publisher._trace80_entry(object(), f.locator)
    assert not f.calls
    locator = dict(f.locator, runtime_qualification=f.bind('qualification.json'))
    def reject(entry, reader):
        assert entry['validation']['runtime_qualification'] == locator['runtime_qualification']
        raise ValueError('actual runtime qualification absent')
    monkeypatch.setattr(bundle, 'validate_catalog_entry', reject)
    with pytest.raises(ValueError, match='actual runtime qualification absent'):
        publisher._trace80_entry(object(), locator)


def test_draft_rejects_qualification_placeholder_or_wrong_geometry(draft):
    for key, value in (('runtime_qualification', None), ('pitch_um', 5), ('Si_TSV_um', 47)):
        with pytest.raises(ValueError, match='draft locator'):
            publisher._trace80_entry_draft(object(), dict(draft.locator, **{key:value}))
    assert not draft.calls


@pytest.fixture
def actual_export_draft(monkeypatch):
    """Actual closed export and coupon metadata, never scientific NPZ data."""
    root = publisher.ROOT
    manifest = root/('operator_repair_20261015/development/trial_forcing_closure_v1/'
        'continuous_cu_screen_v1/runtime/BI4_TRACE80_COMPACT_GEOMETRY_20261025_mfd_AS4_manual2/MANIFEST.json')
    if not manifest.is_file():
        pytest.skip('optional real compact-export evidence is not installed')
    original_open = os.open
    def metadata_only(path, *args, **kwargs):
        assert Path(path).suffix not in ('.npz', '.npy'), 'scientific payload opened during metadata audit'
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(os, 'open', metadata_only)
    reader = publisher.MetadataReader()
    assessment = root/'operator_night_20261021/P4_CYCLIC_BOW_ASSESSMENT_20261024_FULL73_v1/REPORT.json'
    _, evidence = reader.read(assessment)
    entry = publisher._trace80_entry_draft(reader, dict(pitch_um=4, Si_TSV_um=37,
        evidence=evidence, physical_geometry=dict(path=str(manifest),
            sha256='3b351d5adf9714ff68979ba077d4f3aab0a71d42d933c2ddfbca3de34c59cbb9')))
    assert 'bytes' not in entry['construction']
    return reader, entry


def test_actual_export_metadata_joins_publisher_sha_only_construction(actual_export_draft):
    reader, entry = actual_export_draft
    manifest, proofs = bundle._export(entry, reader)
    assert manifest['sources']['construction']['bytes'] == bundle.CONSTRUCTION['bytes'] == 15759
    assert set(proofs) == set(bundle.ROLES)
    assert 'runtime_qualification' not in entry['validation']
    assert 'bytes' not in entry['construction']
    assert all(Path(path).suffix == '.json' for path in reader.cache)
    reader.verify_unchanged()


@pytest.mark.parametrize('field,value', [('bytes', 15760), ('sha256', '0'*64)])
def test_actual_export_metadata_refuses_wrong_construction_size_or_pin(actual_export_draft, field, value):
    reader, entry = actual_export_draft
    entry = deepcopy(entry)
    entry['construction'][field] = value
    with pytest.raises(ValueError):
        bundle._export(entry, reader)
