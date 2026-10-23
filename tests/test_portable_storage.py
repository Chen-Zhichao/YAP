"""Location changes preserve frozen evidence and numerical implementations."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import shutil
import subprocess
import sys

import numpy as np
import pytest

import component_operators as components
import mechanics_runtime as runtime


READ_OVERRIDES = ('YAP_NATIVE_RUNTIME_DIR', 'YAP_RUNTIME_RESOURCES',
                  'YAP_COMPONENT_OPERATOR_CATALOG', 'YAP_COMPONENT_OPERATOR_CATALOG_V2',
                  'YAP_W2W_SI_CAP_LIBRARY_DIR', 'YAP_THICKNESS_FAMILY_DIR')


def fresh_runtime(tmp_path, monkeypatch):
    for key in READ_OVERRIDES:
        monkeypatch.delenv(key, raising=False)
    root = tmp_path/'relocated data'
    monkeypatch.setenv('YAP_DATA_ROOT', str(root))
    name = '_yap_portable_storage_fixture'
    spec = importlib.util.spec_from_file_location(name, runtime.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module, root


def frozen_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    path.chmod(0o444)
    return raw, hashlib.sha256(raw).hexdigest()


def test_relocated_receipt_and_exact_artifact_mapping_are_idempotent(tmp_path, monkeypatch):
    m, root = fresh_runtime(tmp_path, monkeypatch)
    relative = Path('operator_repair_20261015/source/receipt.json')
    _, pin = frozen_json(root/'distribution/operators/artifacts/LOCATIONS.json',
        dict(schema='yap.operator_artifact_locations.v1', files=[str(relative)]))
    m.ARTIFACT_LOCATIONS_SHA256 = pin
    m.ARTIFACT_LOCATION_SUPPLEMENTS = ()
    original = m.RECORDED_PROJECT_ROOT/relative
    target = root/'distribution/operators/artifacts'/relative
    raw, sha = frozen_json(target, dict(readset={str(original): 'unaltered identity'}))
    assert m.shared_distribution_path(original) == target
    assert m.shared_distribution_path(target) == target
    assert m._component_json(original, sha)[0] == json.loads(raw)
    assert target.read_bytes() == raw
    unknown = original.with_name('not-listed.json')
    assert m.shared_distribution_path(unknown) == root/relative.with_name('not-listed.json')
    with pytest.raises(FileNotFoundError):
        m._component_json(unknown)


def test_missing_selected_copy_never_reads_existing_original(tmp_path, monkeypatch):
    m, root = fresh_runtime(tmp_path, monkeypatch)
    existing = runtime.COMPONENT_PROJECT_ROOT/'distribution/operators/components/CU_MATERIAL_CARD.json'
    if not existing.is_file():
        pytest.skip('shared material card is not installed')
    with pytest.raises(FileNotFoundError):
        m._component_json(existing)
    assert not root.exists()


def test_specific_external_read_overrides_are_narrow_and_read_only(tmp_path, monkeypatch):
    m, root = fresh_runtime(tmp_path, monkeypatch)
    native, caps, family = (tmp_path/name for name in ('native', 'caps', 'family'))
    catalog = tmp_path/'separate catalog.json'
    for key, path in zip(('YAP_NATIVE_RUNTIME_DIR', 'YAP_W2W_SI_CAP_LIBRARY_DIR',
                          'YAP_THICKNESS_FAMILY_DIR', 'YAP_COMPONENT_OPERATOR_CATALOG_V2'),
                         (native, caps, family, catalog)):
        monkeypatch.setenv(key, str(path))
        assert m._component_path(path) == path
    assert m._component_path(native/'module.py') == native/'module.py'
    with pytest.raises(ValueError, match='outside YAP_DATA_ROOT'):
        m._component_path(tmp_path/'undeclared.json')
    for path in (native/'cache', caps/'cache', family/'cache', catalog):
        with pytest.raises(ValueError, match='read-only'):
            m._storage_path(path, 'cache')


def test_data_root_must_be_selected_before_import(tmp_path, monkeypatch):
    m, root = fresh_runtime(tmp_path, monkeypatch)
    monkeypatch.setenv('YAP_DATA_ROOT', str(tmp_path/'other'))
    with pytest.raises(ValueError, match='before importing'):
        m.shared_distribution_path(root/'distribution/operators/components/CATALOG.json')


def test_selected_native_default_and_historical_geometry_rebase(tmp_path, monkeypatch):
    m, root = fresh_runtime(tmp_path, monkeypatch)
    native = root/'distribution/operators/components/runtime/native'
    native.mkdir(parents=True)
    (native/'exp12_layer_rom_dataset.py').write_text('# fixture\n')
    assert m.select_native_root() == (native, True)
    assert m.resource_map_path(native) == native.parent/'RESOURCES.json'
    for historical in m.HISTORICAL_NATIVE_ROOTS:
        assert m.resolve_native_input(native, historical/'geometry/input.json',
            native_root=native, portable=True) == native/'geometry/input.json'


def test_explicit_data_root_beats_checkout_bundle_without_fallback(tmp_path, monkeypatch):
    m, root = fresh_runtime(tmp_path, monkeypatch)
    checkout = tmp_path/'checkout'
    monkeypatch.setattr(m, 'ROOT', checkout)
    selected = root/'distribution/operators/components/runtime/native'
    bundled = checkout/'runtime/native'
    for path in (selected, bundled):
        path.mkdir(parents=True)
        (path/'exp12_layer_rom_dataset.py').write_text('# fixture\n')
    assert m.select_native_root(checkout) == (selected, True)
    (selected/'exp12_layer_rom_dataset.py').unlink()
    with pytest.raises(FileNotFoundError, match='no legacy fallback'):
        m.select_native_root(checkout)


def test_data_root_alias_precedes_overlapping_recorded_namespace(tmp_path, monkeypatch):
    m, root = fresh_runtime(tmp_path, monkeypatch)
    historical = tmp_path/'historical'
    root.mkdir()
    alias = historical/'distribution/mirror'
    alias.parent.mkdir(parents=True)
    alias.symlink_to(root, target_is_directory=True)
    monkeypatch.setattr(m, 'RECORDED_PROJECT_ROOT', historical)
    monkeypatch.setattr(m, '_DATA_ROOT_SETTING', str(alias))
    monkeypatch.setenv('YAP_DATA_ROOT', str(alias))
    relative = Path('distribution/operators/components/CATALOG.json')
    assert m.shared_distribution_path(alias/relative) == root/relative
    assert m.shared_distribution_path(root/relative) == root/relative


def test_original_and_relocated_frozen_trees_remain_read_only(tmp_path, monkeypatch):
    m, root = fresh_runtime(tmp_path, monkeypatch)
    for base in (m.RECORDED_PROJECT_ROOT, root):
        for name in ('distribution', 'operator_repair_20261015', 'operator_night_20261021'):
            with pytest.raises(ValueError, match='read-only'):
                m._storage_path(base/name/'must-not-write', 'output')
    alias = tmp_path/'frozen-alias'
    alias.symlink_to(m.RECORDED_PROJECT_ROOT/'operator_night_20261021', target_is_directory=True)
    with pytest.raises(ValueError, match='read-only'):
        m._storage_path(alias/'must-not-write', 'output')


@pytest.mark.parametrize('path', ['/tmp/yap-location-validation-only', '/home/portable_user/YAP_runs',
                                  '/scratch/portable_user/work', '/data/portable_user/cache'])
def test_non_app_locations_warn_without_creating_files(path):
    runtime._storage_warning.cache_clear()
    with pytest.warns(runtime.StorageWarning):
        assert runtime.validate_user_write_path(path) == Path(path).resolve()


def test_independent_work_and_cache_overrides_survive_without_chmod(tmp_path, monkeypatch):
    root, work, cache = tmp_path/'run', tmp_path/'work', tmp_path/'separate-cache'
    root.mkdir()
    root.chmod(0o750)
    monkeypatch.setenv('YAP_RUN_ROOT', str(root))
    monkeypatch.setenv('YAP_WORK_DIR', str(work))
    monkeypatch.setenv('NUMBA_CACHE_DIR', str(cache))
    for key in ('TMPDIR', 'TMP', 'TEMP', 'MPLCONFIGDIR', 'PIP_CACHE_DIR', 'PYTHONUSERBASE',
                'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_STATE_HOME',
                'XDG_RUNTIME_DIR', 'JOBLIB_TEMP_FOLDER', 'PYTHONPYCACHEPREFIX'):
        monkeypatch.delenv(key, raising=False)
    # Preserve all process-global environment/tempfile changes through monkeypatch.
    monkeypatch.setenv('HOME', str(tmp_path/'home'))
    monkeypatch.setattr(runtime.tempfile, 'tempdir', None)
    with pytest.warns(runtime.StorageWarning):
        result = runtime.configure_runtime_storage()
    assert result['work'] == work and result['NUMBA_CACHE_DIR'] == cache
    assert os.environ['NUMBA_CACHE_DIR'] == str(cache)
    assert stat.S_IMODE(root.stat().st_mode) == 0o750
    assert work.is_dir() and cache.is_dir()


def test_aliases_cannot_enter_snapshot_or_readonly_input(tmp_path, monkeypatch):
    alias = tmp_path/'alias'
    alias.symlink_to(tmp_path/'.snapshot', target_is_directory=True)
    with pytest.raises(ValueError, match='snapshot'):
        runtime.validate_user_write_path(alias/'cache')
    alias.unlink()
    alias.symlink_to(runtime.ROOT, target_is_directory=True)
    with pytest.raises(ValueError, match='read-only'):
        runtime.validate_user_write_path(alias/'cache')


@pytest.fixture
def portable_caps(tmp_path, monkeypatch):
    from W2W import component_silicon_cap as caps
    for name in ('PROJECT', '_read_cache', '_publish_cache'):
        monkeypatch.setattr(caps, name, getattr(caps, name))
    monkeypatch.setattr(caps, '_yap_storage_binding', None, raising=False)
    published = tmp_path/'published'
    published.mkdir()
    monkeypatch.setenv('YAP_W2W_SI_CAP_LIBRARY_DIR', str(published))
    before = Path(caps.__file__).read_bytes()
    proof = components._configure_cap_storage(caps)
    assert Path(caps.__file__).read_bytes() == before
    assert proof['frozen_numerical_source_sha256'] == hashlib.sha256(before).hexdigest()
    return caps, published


def cache_metadata(caps, spec):
    audits = dict.fromkeys(caps.AUDIT_ERRORS, 0.)
    audits.update(minimum_scaled_eigenvalue_over_spectral_radius=0.,
        fourth_scaled_eigenvalue_over_spectral_radius=1e-4, solve_rhs_count=135,
        algebra_gate_limit=caps.REACTION_LIMIT)
    return dict(specification=spec, audits=audits)


def test_cap_location_changes_preserve_keys_and_refuse_partial_or_overwrite(portable_caps, tmp_path):
    caps, published = portable_caps
    assert caps._published_root() == published
    cache = tmp_path/'cache'
    cache.mkdir()
    spec = dict(fixture=True)
    key = hashlib.sha256(caps._json(spec).encode()).hexdigest()
    caps._publish_cache(cache, key, np.eye(134), np.zeros(134), cache_metadata(caps, spec))
    before = [(cache/(key+suffix)).read_bytes() for suffix in ('.npz', '.json')]
    assert caps._read_cache(cache/(key+'.npz'), cache/(key+'.json'), key) is not None
    with pytest.raises(ValueError, match='overwrite'):
        caps._publish_cache(cache, key, np.eye(134), np.zeros(134), cache_metadata(caps, spec))
    assert before == [(cache/(key+suffix)).read_bytes() for suffix in ('.npz', '.json')]
    partial = 'a'*64
    (cache/(partial+'.npz')).write_bytes(b'preserved partial')
    with pytest.raises(ValueError, match='incomplete'):
        caps._read_cache(cache/(partial+'.npz'), cache/(partial+'.json'), partial)
    assert (cache/(partial+'.npz')).read_bytes() == b'preserved partial'
    (cache/(partial+'.json')).symlink_to(cache/'missing-metadata')
    with pytest.raises(ValueError, match='regular files'):
        caps._read_cache(cache/(partial+'.npz'), cache/(partial+'.json'), partial)
    with pytest.raises(ValueError, match='read-only'):
        caps._publish_cache(published, key, np.eye(134), np.zeros(134), cache_metadata(caps, spec))


def test_cap_configuration_cannot_switch_under_live_users(portable_caps, tmp_path, monkeypatch):
    caps, published = portable_caps
    assert components._configure_cap_storage(caps)['published_root'] == str(published)
    monkeypatch.setenv('YAP_W2W_SI_CAP_LIBRARY_DIR', str(tmp_path/'another'))
    with pytest.raises(ValueError, match='restart Python'):
        components._configure_cap_storage(caps)


def test_real_middle_catalog_metadata_with_original_project_hidden(tmp_path, monkeypatch):
    """Sparse size-only operands are fixtures, never usable numerical operators."""
    bwrap = shutil.which('bwrap')
    if bwrap is None or not components.DEFAULT_CATALOG.is_file():
        pytest.skip('installed delivery and bubblewrap are required for isolation test')
    probe = subprocess.run([bwrap, '--ro-bind', '/', '/', '--unshare-net', '/usr/bin/true'],
                           capture_output=True, timeout=10)
    if probe.returncode:
        pytest.skip('mount-namespace isolation is unavailable')
    original_raw = components.DEFAULT_CATALOG.read_bytes()
    published = json.loads(original_raw)
    selected = next(e for e in published['entries'] if e['id'] == 'BI6_SI57')
    derived = dict(schema=published['schema'], entries=[selected])
    selected_index = tmp_path/'selected.json'
    frozen_json(selected_index, derived)
    reader = components._metadata_reader()
    with monkeypatch.context() as patch:
        patch.setattr(components, '_metadata_reader', lambda: reader)
        expected = components.load_catalog(selected_index)['entries'][0]
    fixture, sparse = tmp_path/'isolated', []
    data, code = fixture/'data', fixture/'code'
    code.mkdir(parents=True)
    for name in ('mechanics_runtime.py', 'component_operators.py',
                 'tools/__init__.py', 'tools/publish_component_operators.py'):
        destination = code/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((runtime.ROOT/name).read_bytes())
        destination.chmod(0o444)
    copies = {}
    for name, (_, observed, _) in reader.cache.items():
        source = Path(name)
        raw = source.read_bytes()
        assert len(raw) <= 1024**2 and hashlib.sha256(raw).hexdigest() == observed['sha256']
        copies[source] = raw
    for name, pin in [('LOCATIONS.json', runtime.ARTIFACT_LOCATIONS_SHA256),
                      *runtime.ARTIFACT_LOCATION_SUPPLEMENTS]:
        source = runtime.COMPONENT_PROJECT_ROOT/'distribution/operators/artifacts'/name
        raw = source.read_bytes()
        assert len(raw) <= 128*1024 and hashlib.sha256(raw).hexdigest() == pin
        copies[source] = raw
    assert len(copies) <= 64 and sum(map(len, copies.values())) < 16*1024**2
    for source, raw in copies.items():
        destination = data/source.relative_to(runtime.COMPONENT_PROJECT_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        destination.chmod(0o444)
    frozen_json(data/components.DEFAULT_CATALOG.relative_to(runtime.COMPONENT_PROJECT_ROOT), derived)
    assert hashlib.sha256(json.dumps(derived).encode()).hexdigest() != hashlib.sha256(original_raw).hexdigest()
    for binding in selected['components'].values():
        source = runtime.shared_distribution_path(binding['path'])
        destination = data/source.relative_to(runtime.COMPONENT_PROJECT_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert 0 < binding['bytes'] <= 2*1024**3
        with destination.open('xb') as stream:
            stream.truncate(binding['bytes'])
        destination.chmod(0o444)
        assert destination.stat().st_blocks*512 <= 65536
        sparse.append(destination)
    child = """
import json, os, pathlib, sys
root = pathlib.Path('/mnt/relocated')
for key in list(os.environ):
    if key.startswith('YAP_'):
        del os.environ[key]
os.environ['YAP_DATA_ROOT'] = str(root/'data')
sys.path.insert(0, str(root/'code'))
import component_operators as c
assert not c.runtime.RECORDED_PROJECT_ROOT.joinpath('distribution/operators/components/CU_MATERIAL_CARD.json').exists()
def readonly(event, args):
    if event == 'open' and isinstance(args[2], int):
        assert not args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND), 'unexpected write'
sys.addaudithook(readonly)
opening = c.runtime._component_open
class MetadataOnly:
    def __init__(self, stream): self.stream = stream
    def __enter__(self): return self
    def __exit__(self, *args): self.stream.close()
    def read(self, *args): raise AssertionError('sparse fixture is not a numerical operand')
    def __getattr__(self, key): return getattr(self.stream, key)
def checked(path, *args, **kw):
    actual = c.runtime._component_path(path)
    assert actual.is_relative_to(root/'data'), actual
    stream = opening(path, *args, **kw)
    return MetadataOnly(stream) if actual.suffix == '.npz' else stream
c.runtime._component_open = checked
catalog = c.load_catalog()
assert len(catalog['entries']) == 1
entry = catalog['entries'][0]
assert entry['id'] == 'BI6_SI57'
assert all(pathlib.Path(b['path']).is_relative_to(root/'data') for b in entry['components'].values())
print(json.dumps(dict(id=entry['id'], validation=entry['validation'],
                     original_project_hidden=True, numerical_operands_verified=False)))
"""
    command = [bwrap, '--ro-bind', '/', '/', '--tmpfs', '/mnt',
        '--dir', '/mnt/relocated', '--ro-bind', str(fixture), '/mnt/relocated',
        '--tmpfs', str(runtime.RECORDED_PROJECT_ROOT), '--remount-ro', str(runtime.RECORDED_PROJECT_ROOT),
        '--unshare-net',
        '--chdir', '/mnt/relocated', sys.executable, '-I', '-B', '-c', child]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert completed.returncode == 0, completed.stderr
        observed = json.loads(completed.stdout)
        assert observed['validation']['max_relative_pct'] == expected['validation']['max_relative_pct']
        assert observed['original_project_hidden'] is True
        assert observed['numerical_operands_verified'] is False
        for source, raw in copies.items():
            assert (data/source.relative_to(runtime.COMPONENT_PROJECT_ROOT)).read_bytes() == raw
        required = next(path for path in copies if 'BOW_ASSESSMENT' in str(path))
        target = data/required.relative_to(runtime.COMPONENT_PROJECT_ROOT)
        held = target.with_name('held.json')
        target.rename(held)
        try:
            missing = subprocess.run(command, capture_output=True, text=True, timeout=30)
            assert missing.returncode != 0 and 'FileNotFoundError' in missing.stderr
        finally:
            held.rename(target)
    finally:
        for placeholder in sparse:
            placeholder.unlink(missing_ok=True)
