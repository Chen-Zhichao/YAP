"""Tiny synthetic fixtures only; temporary storage stays in the canonical project."""

import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import tools.operator_coverage as coverage


class CoverageTests(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ.get('YAP_COVERAGE_TEST_ROOT',
                    str(coverage.ROOT / coverage.PROGRESS / 'coverage_20261019_v1')))
        if not base.is_absolute() or not base.is_relative_to(coverage.ROOT):
            raise ValueError('test scratch must be under the canonical project')
        self.temporary = tempfile.TemporaryDirectory(prefix='.tiny_test_', dir=base)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def inventory(self, **kwargs):
        return coverage.Inventory([self.root], **kwargs)

    def write_json(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding='ascii')
        return path

    def test_grid_has_81_exact_coordinates_and_separate_ledgers(self):
        rows = coverage.coverage_rows([], [])
        self.assertEqual(len(rows), 81)
        self.assertEqual(len({(r['bonding_pitch_um'], r['total_thickness_um']) for r in rows}), 81)
        row = next(r for r in rows if r['bonding_pitch_um'] == 2 and r['total_thickness_um'] == 50)
        self.assertEqual(row['single_BI_si_um'], 38.5)
        self.assertEqual(row['two_BI_si_um'], 37)
        self.assertFalse(any(r['validated_global_bow'] or r['D2W_assembly'] or r['W2W_assembly'] for r in rows))

    def test_formal_cases_do_not_validate_50_or_round_98p5_to_100(self):
        cases = [dict(bonding_pitch_um=2, total_thickness_um=t, payload_exists=True,
                      structural_candidate=True, validated_global_bow=True) for t in (60, 98.5)]
        rows = coverage.coverage_rows([], cases)
        self.assertEqual([(r['bonding_pitch_um'], r['total_thickness_um']) for r in rows if r['validated_global_bow']], [(2, 60)])
        self.assertFalse(any(r['D2W_assembly'] or r['W2W_assembly'] for r in rows))

    def test_legacy_candidate_never_infers_validation(self):
        package = dict(bonding_pitch_um=2, total_thickness_um=50, package_exists=True,
                       payload_exists=True, structural_candidate=True)
        row = coverage.coverage_rows([package], [])[3]
        self.assertTrue(row['payload_exists'])
        self.assertFalse(row['validated_global_bow'])

    def test_registry_is_parsed_without_import_or_execution(self):
        source = "raise RuntimeError('must not execute')\nPREVIEW_PACKAGES={'x': {'path': Path('resource:geometry/a.json'), 'geometry': (2,10,6,50)}}"
        self.assertEqual(coverage.registry(source, 'PREVIEW_PACKAGES')['x']['path'], 'resource:geometry/a.json')
        with self.assertRaises(ValueError):
            coverage.registry("PREVIEW_PACKAGES=__import__('os')", 'PREVIEW_PACKAGES')

    def test_resources_are_relative_confined_and_missing_is_explicit(self):
        self.assertIsNone(coverage.resource_path(self.root, {}, 'resource:geometry/x.json'))
        with self.assertRaises(ValueError):
            coverage.resource_path(self.root, {'geometry': '../bad'}, 'resource:geometry/x.json')
        with self.assertRaises(ValueError):
            coverage.resource_path(self.root, {'geometry': 'ok'}, 'resource:geometry/.snapshot/x.json')

    def test_symlink_leaf_and_ancestor_are_never_followed(self):
        target = self.root / 'target'
        target.write_bytes(b'{}')
        link = self.root / 'link'
        link.symlink_to(target)
        inv = self.inventory()
        self.assertEqual(inv.metadata(link)['status'], 'BLOCKED_NONREGULAR')
        with self.assertRaises(OSError):
            inv.read(link)
        directory = self.root / 'alias'
        directory.symlink_to(self.root, target_is_directory=True)
        self.assertEqual(inv.metadata(directory / 'target')['status'], 'BLOCKED_NOFOLLOW')
        with self.assertRaises(OSError):
            inv.read(directory / 'target')

    def test_bounds_snapshots_and_confinement(self):
        path = self.root / 'small'
        path.write_bytes(b'1234')
        with self.assertRaises(ValueError):
            self.inventory().read(path, limit=3)
        with self.assertRaises(ValueError):
            self.inventory(max_bytes=3).read(path)
        with self.assertRaises(ValueError):
            self.inventory().metadata(self.root / '.snapshot/x')
        with self.assertRaises(ValueError):
            self.inventory().metadata(self.root.parent / 'outside')
        with self.assertRaises(TimeoutError):
            self.inventory(seconds=-1).metadata(path)
        inv = self.inventory(max_files=1)
        inv.metadata(path)
        with self.assertRaises(ValueError):
            inv.metadata(self.root / 'other')

    def test_json_rejects_duplicate_nonfinite_and_bad_hash(self):
        path = self.root / 'input.json'
        path.write_bytes(b'{"x":1,"x":2}')
        with self.assertRaises(ValueError):
            self.inventory().json(path)
        path.write_bytes(b'{"x":NaN}')
        with self.assertRaises(ValueError):
            self.inventory().json(path)
        path.write_bytes(b'{}')
        with self.assertRaises(ValueError):
            self.inventory().json(path, expected='0'*64)

    def test_read_refuses_replaced_path_identity(self):
        path = self.write_json('input.json', {})
        inv = self.inventory()
        metadata = inv.metadata
        def changed(value):
            record = metadata(value)
            record['inode'] += 1
            return record
        with patch.object(inv, 'metadata', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'path identity changed'):
                inv.json(path)

    def test_payloads_only_receive_metadata(self):
        payload = self.root / 'data.npz'
        payload.write_bytes(b'not an npz and must not be decoded')
        package = self.write_json('BI2_50.json', dict(name='BI2_50', state='READY_ENGINEERING_PREVIEW',
            datasets={'si': 'data.npz'}, prepared={'si': 'data.npz'}, port_basis='data.npz'))
        inv = self.inventory()
        record = coverage.package_record(inv, package, self.root, 2, 50)
        self.assertTrue(record['payload_exists'])
        self.assertTrue(record['structural_candidate'])
        self.assertFalse(record['validated_global_bow'])
        self.assertNotIn('sha256', inv.files[str(payload)])
        self.assertEqual(inv.bytes_read, package.stat().st_size)

    def test_missing_dependency_does_not_claim_structural_readiness(self):
        package = self.write_json('BI2_50.json', dict(name='BI2_50', state='READY_ENGINEERING_PREVIEW',
            datasets={'si': 'missing.npz'}, prepared={'si': 'missing.npz'}, port_basis='missing.npz'))
        record = coverage.package_record(self.inventory(), package, self.root, 2, 50)
        self.assertTrue(record['package_exists'])
        self.assertTrue(record['structural_receipt_declared'])
        self.assertFalse(record['payload_exists'])
        self.assertFalse(record['structural_candidate'])

    def test_historical_paths_relocate_without_opening_shared_home(self):
        old = coverage.OLD_NATIVE[0] / 'results/a.npz'
        self.assertEqual(coverage.payload_path(str(old), self.root / 'packages/a.json', self.root), self.root / 'results/a.npz')
        with self.assertRaises(ValueError):
            coverage.payload_path('../../escape', self.root / 'packages/a.json', self.root)

    def test_formal_acceptance_cannot_be_inferred_from_partial_receipt(self):
        with self.assertRaises(ValueError):
            coverage.formal_cases(self.inventory(), {'scope': {}, 'global_accuracy_passed': True})

    def test_output_is_exclusive_and_grid_csv_has_81_rows(self):
        rows = coverage.coverage_rows([], [])
        report = dict(grid_rows=rows, repaired_single_BI=[], expansion_plan={})
        destination = self.root / 'output'
        coverage.write_outputs(destination, report)
        original = (destination / 'COVERAGE.json').read_bytes()
        with self.assertRaises(FileExistsError):
            coverage.write_outputs(destination, report)
        self.assertEqual((destination / 'COVERAGE.json').read_bytes(), original)
        self.assertEqual(len(list(csv.DictReader(io.StringIO((destination / 'GRID81.csv').read_text())))), 81)


if __name__ == '__main__':
    unittest.main()
