"""Source-custody and dispatch tests, never production geometry construction.

Only eight small frozen source files may be read. The physical seed deck is
represented by exception sentinels or synthetic bytes and is never opened.
"""

import builtins
from collections import Counter
import hashlib
from pathlib import Path, PurePosixPath
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from tools import operator_seed_loader as seeds


PROJECT = Path('/app/nanocad/projects/personal/zchen/YAP_STACK')
DIRECTORY = PROJECT / seeds.SOURCE_RELATIVE


class SeedPayloadSentinel(RuntimeError):
    pass


class FakeSourceOwner:
    def __init__(self, root, sources):
        self.root = PurePosixPath(root)
        self.sources = sources
        self.calls = []

    def read_source(self, binding, *, cap, readonly):
        self.calls.append((dict(binding), cap, readonly))
        if set(binding) != {'path', 'sha256'} or readonly is not False:
            raise AssertionError('source owner ABI changed')
        path = PurePosixPath(binding['path'])
        if path == self.root / seeds.SEED_RELATIVE:
            if binding['sha256'] != seeds.SEED_SHA or not 0 < cap <= 16 * 1024**2:
                raise AssertionError('unbounded or unpinned seed request')
            raise SeedPayloadSentinel('physical seed payload deliberately not read')
        if path.parent != self.root / seeds.SOURCE_RELATIVE or path.suffix != '.py':
            raise AssertionError('reader escaped the eight-file source closure')
        name = path.stem
        if name not in seeds.SOURCE_PINS or binding['sha256'] != seeds.SOURCE_PINS[name]:
            raise AssertionError('reader requested an unknown or unpinned source')
        if not 0 < cap <= 256 * 1024:
            raise AssertionError('source read bound changed')
        return self.sources[name]


class OperatorSeedLoaderTests(unittest.TestCase):
    @classmethod
    def read_frozen_sources(cls):
        result = {}
        for name, expected in seeds.SOURCE_PINS.items():
            # No directory listing, glob, deck access or imported source reader.
            with (DIRECTORY / (name + '.py')).open('rb') as stream:
                raw = stream.read(256 * 1024 + 1)
            if not 0 < len(raw) <= 256 * 1024 or hashlib.sha256(raw).hexdigest() != expected:
                raise AssertionError('frozen source bytes differ: ' + name)
            result[name] = raw
        return result

    @classmethod
    def setUpClass(cls):
        cls.sources = cls.read_frozen_sources()

    def setUp(self):
        self.root = PROJECT / 'runtime_checks/SEED_LOADER_UNIT_ONLY_RELOCATED_PROJECT'
        self.owner = FakeSourceOwner(str(self.root), self.sources)
        self.check = mock.Mock()
        self.loader = seeds.SeedSources(project_root=str(self.root),
            read_source=self.owner.read_source, check=self.check)

    def namespace_snapshot(self):
        return {name: module for name, module in sys.modules.items()
                if name in seeds.SOURCE_PINS or name.startswith(('_operator_seed_', '_bi_pitch_'))}

    def test_all_pitch_profiles_are_metadata_only_and_seven_is_blocked(self):
        cells = {2: 6, 3: 6, 4: 12, 5: 30, 6: 6, 7: 42, 8: 24, 9: 18, 10: 30}
        for pitch in range(2, 11):
            with self.subTest(pitch=pitch):
                profile = seeds.profile(pitch)
                self.assertEqual(profile['pitch_um'], pitch)
                self.assertEqual(profile['diameter_um'] / pitch, .5)
                self.assertEqual(profile['source_si_um'], 37.)
                self.assertEqual(profile['cell_um'], cells[pitch])
                self.assertTrue(profile['requires_resource_owner'])
                self.assertFalse(profile['operator_admission'])
                self.assertFalse(profile['mesh_accuracy_inferred'])
                if pitch == 7:
                    self.assertEqual(profile['implementation'], 'BLOCKED_MESH_RECIPE')
                    self.assertIsNone(profile['mesh_recipe'])
                else:
                    self.assertEqual(profile['mesh_recipe'], seeds.RECIPE_IDS[pitch])
        self.assertEqual(self.owner.calls, [])

    def test_profile_invalid_pitches(self):
        for pitch in (True, 2., 1, 11, '8', None):
            with self.subTest(pitch=pitch), self.assertRaises(ValueError):
                seeds.profile(pitch)
        self.assertEqual(self.owner.calls, [])

    def test_seven_construct_is_blocked_before_load_or_owner_io(self):
        with mock.patch.object(self.loader, 'load', side_effect=AssertionError('blocked pitch loaded source')):
            with self.assertRaisesRegex(ValueError, '141'):
                self.loader.construct(None, 7)
        self.assertEqual(self.owner.calls, [])
        self.assertEqual(self.loader._modules, {})

    def test_actual_private_modules_load_only_eight_sources_and_cache(self):
        before = self.namespace_snapshot()
        modules = {name: self.loader.load(name) for name in seeds.SOURCE_PINS}
        for name, module in modules.items():
            self.assertIs(self.loader.load(name), module)
            self.assertEqual(module.__file__, str(self.root / seeds.SOURCE_RELATIVE / (name + '.py')))
            self.assertIsNot(sys.modules.get(name), module)
        counts = Counter(PurePosixPath(binding['path']).stem for binding, _, _ in self.owner.calls)
        self.assertEqual(counts, Counter({name: 1 for name in seeds.SOURCE_PINS}))
        self.assertEqual(self.loader._physical_bytes, sum(map(len, self.sources.values())))
        self.assertEqual(len(self.loader._proofs), 8)
        for name, raw in self.sources.items():
            path = str(self.root / seeds.SOURCE_RELATIVE / (name + '.py'))
            self.assertEqual(self.loader._proofs[path],
                             dict(sha256=seeds.SOURCE_PINS[name], bytes=len(raw)))
        self.assertEqual(self.namespace_snapshot(), before)
        self.assertGreater(self.check.call_count, 8)

    def test_private_paths_and_reader_hooks_are_relocated(self):
        geo = self.loader.load('bi_pitch_geometry')
        self.assertEqual(geo.R, self.root)
        self.assertEqual(geo.D, self.root / seeds.SOURCE_RELATIVE)
        self.assertEqual(geo.RT, self.root / seeds.SOURCE_RELATIVE / 'runtime')
        self.assertEqual(geo.SELF, self.root / seeds.SOURCE_RELATIVE / 'bi_pitch_geometry.py')
        self.assertEqual(geo.read_source, self.loader._read)
        extension = self.loader.load('prepared_geometry_extension')
        self.assertEqual(extension.bounded_read, self.loader._read)
        self.assertEqual(extension.SEED, self.root / seeds.SEED_RELATIVE)
        self.assertEqual(extension.geometry.__file__,
                         str(self.root / seeds.SOURCE_RELATIVE / 'native_periodic_geometry.py'))
        common = self.loader.load('common_cell_coverage_extension')
        self.assertIs(common.geo, geo)
        self.assertEqual(common.OUTPUT, self.root / 'operator_night_20261021/coverage')

    def test_original_private_child_prepare_stops_at_seed_before_numpy(self):
        before = self.namespace_snapshot()
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in ('numpy', 'scipy'):
                raise AssertionError('prepare imported a numerical library before the seed sentinel')
            return original_import(name, *args, **kwargs)

        with mock.patch.object(builtins, '__import__', side_effect=guarded_import):
            geo = self.loader.load('bi_pitch_geometry')
            private = geo.PrivateSources()
            child = private.load('prepared_geometry_extension')
            self.assertIs(private.load('prepared_geometry_extension'), child)
            self.assertEqual(child.D, self.root / seeds.SOURCE_RELATIVE)
            self.assertEqual(child.SEED, self.root / seeds.SEED_RELATIVE)
            self.assertEqual(child.bounded_read, self.loader._read)
            self.assertEqual(child.geometry.__file__,
                             str(self.root / seeds.SOURCE_RELATIVE / 'native_periodic_geometry.py'))
            with self.assertRaisesRegex(SeedPayloadSentinel, 'deliberately not read'):
                child.prepare(37.)
        seed_calls = [call for call in self.owner.calls
                      if call[0]['path'] == str(self.root / seeds.SEED_RELATIVE)]
        self.assertEqual(len(seed_calls), 1)
        self.assertEqual(seed_calls[0][0]['sha256'], seeds.SEED_SHA)
        self.assertEqual(seed_calls[0][1], 16 * 1024**2)
        self.assertEqual(self.namespace_snapshot(), before)

    def test_two_argument_seed_default_accepts_actual_size_synthetic_bytes(self):
        # Match the real deck's size, not its content; no deck or geometry is read.
        payload = b'x' * 12506783
        pin = hashlib.sha256(payload).hexdigest()
        owner = mock.Mock(return_value=payload)
        loader = seeds.SeedSources(project_root=str(self.root), read_source=owner, check=self.check)
        path = self.root / seeds.SEED_RELATIVE
        with mock.patch.object(seeds, 'SEED_SHA', pin):
            self.assertIs(loader._read(path, pin), payload)
            owner.assert_called_once_with(dict(path=str(path), sha256=pin), cap=16 * 1024**2, readonly=False)
            self.assertEqual(loader._physical_bytes, 12506783)
            self.assertEqual(loader._proofs[str(path)], dict(sha256=pin, bytes=12506783))
            self.assertIs(loader._read(path, pin), payload)
            with self.assertRaisesRegex(ValueError, 'tighter bound'):
                loader._read(path, pin, 256 * 1024)
            self.assertEqual(owner.call_count, 1)
            smaller_owner = mock.Mock(return_value=payload)
            smaller = seeds.SeedSources(project_root=str(self.root), read_source=smaller_owner, check=self.check)
            with self.assertRaisesRegex(ValueError, 'changed/incomplete'):
                smaller._read(path, pin, 256 * 1024)
            smaller_owner.assert_called_once_with(dict(path=str(path), sha256=pin), cap=256 * 1024, readonly=False)
            self.assertEqual(smaller._cache, {})
            self.assertEqual(smaller._physical_bytes, 0)

    def test_two_argument_source_default_stays_256KiB(self):
        name = 'native_periodic_geometry'
        path = self.loader.directory / (name + '.py')
        self.assertEqual(self.loader._read(path, seeds.SOURCE_PINS[name]), self.sources[name])
        self.assertEqual(self.owner.calls, [(dict(path=str(path), sha256=seeds.SOURCE_PINS[name]), 256 * 1024, False)])

    def test_default_seed_pin_and_source_closure_still_reject_before_io(self):
        name = 'native_periodic_geometry'
        attempts = ((self.root / seeds.SEED_RELATIVE, '0' * 64),
                    (self.loader.directory / (name + '.py'), '0' * 64),
                    (self.loader.directory / 'unknown.py', '0' * 64),
                    (self.root / 'elsewhere' / (name + '.py'), seeds.SOURCE_PINS[name]))
        for path, pin in attempts:
            with self.subTest(path=str(path)), self.assertRaises(ValueError):
                self.loader._read(path, pin)
        self.assertEqual(self.owner.calls, [])
        self.assertEqual(self.loader._cache, {})

    def test_seed_default_keeps_original_32MiB_cap_reservation(self):
        self.loader._physical_bytes = 17 * 1024**2
        with self.assertRaisesRegex(ValueError, 'cache/read budget'):
            self.loader._read(self.root / seeds.SEED_RELATIVE, seeds.SEED_SHA)
        self.assertEqual(self.owner.calls, [])

    def test_fe_loads_and_imports_are_denied_before_owner_reads(self):
        for name in ('native_monolithic_reference', 'native_monolithic_assembly'):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.loader.load(name)
                with self.assertRaisesRegex(ValueError, 'FE workers/assemblers'):
                    self.loader._import(name)
        self.assertEqual(self.owner.calls, [])

    def test_original_private_sources_cannot_load_fe_worker(self):
        geo = self.loader.load('bi_pitch_geometry')
        count = len(self.owner.calls)
        with self.assertRaisesRegex(ValueError, 'FE worker'):
            geo.PrivateSources().load('native_monolithic_reference')
        self.assertEqual(len(self.owner.calls), count)

    def test_source_reader_rejects_wrong_sha_type_size_and_empty(self):
        name = 'native_periodic_geometry'
        for raw in (self.sources[name] + b'\n', self.sources[name].decode('ascii'), b'', b'x' * (256 * 1024 + 1)):
            owner = mock.Mock(return_value=raw)
            loader = seeds.SeedSources(project_root=str(self.root), read_source=owner, check=self.check)
            with self.subTest(raw_type=type(raw), size=len(raw)), self.assertRaises(ValueError):
                loader.load(name)
            self.assertEqual(loader._modules, {})
            self.assertEqual(loader._cache, {})

    def test_source_scope_and_seed_pin_caps_reject_before_owner(self):
        source = self.loader.directory / 'native_periodic_geometry.py'
        attempts = ((source, '0' * 64, 256 * 1024),
                    (self.loader.directory / 'native_monolithic_reference.py', '0' * 64, 256 * 1024),
                    (self.loader.directory / 'elsewhere' / source.name, seeds.SOURCE_PINS[source.stem], 256 * 1024),
                    (source, seeds.SOURCE_PINS[source.stem], 0),
                    (source, seeds.SOURCE_PINS[source.stem], 256 * 1024 + 1),
                    (self.loader.root / seeds.SEED_RELATIVE, '0' * 64, 16 * 1024**2),
                    (self.loader.root / seeds.SEED_RELATIVE, seeds.SEED_SHA, 16 * 1024**2 + 1))
        for path, sha, cap in attempts:
            with self.subTest(path=str(path), cap=cap), self.assertRaises(ValueError):
                self.loader._read(path, sha, cap)
        self.assertEqual(self.owner.calls, [])

    def test_cached_source_respects_tighter_bound_without_rereading(self):
        name = 'native_periodic_geometry'
        self.loader.load(name)
        count = len(self.owner.calls)
        with self.assertRaisesRegex(ValueError, 'tighter bound'):
            self.loader._read(self.loader.directory / (name + '.py'), seeds.SOURCE_PINS[name],
                              len(self.sources[name]) - 1)
        self.assertEqual(len(self.owner.calls), count)

    def test_cache_budget_and_guard_fail_before_io(self):
        self.loader._physical_bytes = 32 * 1024**2
        with self.assertRaisesRegex(ValueError, 'cache/read budget'):
            self.loader.load('native_periodic_geometry')
        self.assertEqual(self.owner.calls, [])
        self.check.side_effect = RuntimeError('unit owner stopped')
        with self.assertRaisesRegex(RuntimeError, 'owner stopped'):
            self.loader.load('native_periodic_geometry')
        self.assertEqual(self.owner.calls, [])

    def test_failed_postload_guard_does_not_leave_cached_module(self):
        name = 'native_periodic_geometry'

        def guard():
            if name in self.loader._modules:
                raise RuntimeError('postload guard stopped')

        self.loader._check = guard
        with self.assertRaisesRegex(RuntimeError, 'postload guard'):
            self.loader.load(name)
        self.assertNotIn(name, self.loader._modules)

    def test_invalid_roots_and_missing_owner_are_rejected(self):
        for root in ('relative/project', str(self.root / '..' / 'elsewhere'),
                     str(self.root / '#snapshot'), str(self.root / '.snapshot')):
            with self.subTest(root=root), self.assertRaises(ValueError):
                seeds.SeedSources(project_root=root, read_source=self.owner.read_source, check=self.check)
        with self.assertRaises(ValueError):
            seeds.SeedSources(project_root=str(self.root), read_source=None, check=self.check)
        self.assertEqual(self.owner.calls, [])

    def test_constructor_dispatch_uses_only_mock_two_node_results(self):
        import numpy as np
        from scipy import sparse
        api = SimpleNamespace(np=np, sparse=sparse)
        for pitch in seeds.RECIPE_IDS:
            with self.subTest(pitch=pitch):
                info = seeds.profile(pitch)
                basis = dict(coordinates_mm=np.array([[0., 0., 0.],
                    [info['cell_um'] / 1000, .0485, info['cell_um'] / 1000]]),
                    P=sparse.csr_matrix(np.ones((6, 1))),
                    proof=dict(Cartesian_intervals=[seeds.EXPECTED_INTERVALS[pitch], 44,
                                                   seeds.EXPECTED_INTERVALS[pitch]]))
                deck, proof = object(), dict(software_dispatch_fixture_only=True)
                constructor = mock.Mock(return_value=(deck, basis, proof))
                module = SimpleNamespace(prepare=constructor, construct=constructor)
                geometry = object()

                def load(name):
                    return geometry if name == 'common_cell_geometry' else module

                with mock.patch.object(self.loader, 'load', side_effect=load) as loading:
                    result = self.loader.construct(api, pitch)
                if pitch == 2:
                    constructor.assert_called_once_with(37.)
                    self.assertEqual(loading.call_args_list, [mock.call('prepared_geometry_extension')])
                elif pitch in (3, 6):
                    constructor.assert_called_once_with()
                    expected = 'bi3_coarse_candidate' if pitch == 3 else 'bi_pitch_geometry'
                    self.assertEqual(loading.call_args_list, [mock.call(expected)])
                elif pitch == 4:
                    constructor.assert_called_once_with(geometry)
                    self.assertEqual(loading.call_args_list,
                                     [mock.call('common_cell_geometry'), mock.call('common_cell_reference')])
                else:
                    constructor.assert_called_once_with(geometry, pitch, 'coarse')
                    self.assertEqual(loading.call_args_list,
                        [mock.call('common_cell_geometry'), mock.call('common_cell_coverage_extension')])
                self.assertIs(result.deck, deck)
                self.assertIs(result.basis, basis)
                self.assertIs(result.proof, proof)
                self.assertFalse(result.FE_assembly_performed)
                self.assertFalse(result.operator_admission)
                self.assertFalse(result.resource_admission_performed_here)
                self.assertEqual(result.Cu_updates, 0)
        self.assertEqual(self.owner.calls, [])

    def test_mock_constructor_wrong_interval_cell_or_height_is_rejected(self):
        import numpy as np
        from scipy import sparse
        api = SimpleNamespace(np=np, sparse=sparse)
        for fault in ('interval', 'cell', 'height', 'matrix'):
            basis = dict(coordinates_mm=np.array([[0., 0., 0.], [.006, .0485, .006]]),
                P=sparse.csr_matrix(np.ones((6, 1))), proof=dict(Cartesian_intervals=[45, 44, 45]))
            if fault == 'interval':
                basis['proof']['Cartesian_intervals'] = [46, 44, 45]
            elif fault == 'cell':
                basis['coordinates_mm'][1, 0] = .012
            elif fault == 'height':
                basis['coordinates_mm'][1, 1] = .060
            else:
                basis['P'] = np.ones((6, 1))
            module = SimpleNamespace(prepare=mock.Mock(return_value=(object(), basis, {})))
            with self.subTest(fault=fault), mock.patch.object(self.loader, 'load', return_value=module):
                with self.assertRaises(ValueError):
                    self.loader.construct(api, 2)
        self.assertEqual(self.owner.calls, [])

    def test_frozen_source_files_and_standard_module_names_are_unchanged(self):
        before = self.namespace_snapshot()
        for name in seeds.SOURCE_PINS:
            self.loader.load(name)
        self.assertEqual(self.read_frozen_sources(), self.sources)
        self.assertEqual(self.namespace_snapshot(), before)
        with self.assertRaises(TypeError):
            seeds.SOURCE_PINS['invented'] = '0' * 64


if __name__ == '__main__':
    unittest.main()
