"""Software-only planner tests using configured canonical test/run storage.

No backend is executed. Even tests of --execute replace Popen with a mock.
"""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from tools import operator_production as production

ORIGINAL_CHECK_BACKEND = production.check_backend


PROJECT = Path('/app/nanocad/projects/personal/zchen/YAP_STACK')


class OperatorProductionTests(unittest.TestCase):
    def setUp(self):
        value = os.environ.get('TEST_ROOT', os.environ.get('YAP_RUN_ROOT'))
        if not value:
            raise RuntimeError('set TEST_ROOT or YAP_RUN_ROOT to canonical test storage')
        root = Path(value)
        if not root.is_absolute() or {'..', '#snapshot', '.snapshot'}.intersection(root.parts):
            raise RuntimeError('TEST_ROOT must be an absolute nonsnapshot path')
        resolved = root.resolve(strict=True)
        if not resolved.is_relative_to(PROJECT):
            raise RuntimeError('TEST_ROOT must remain under the canonical project')
        relative = resolved.relative_to(PROJECT)
        if not relative.parts or not (relative.parts[0] in ('runtime_checks', 'operator_campaigns')
                                     or relative.parts[0].startswith('runtime_checks_')):
            raise RuntimeError('TEST_ROOT must be under canonical runtime_checks or operator_campaigns storage')
        if not resolved.is_dir():
            raise RuntimeError('TEST_ROOT must already be a directory')
        self.temporary = tempfile.TemporaryDirectory(prefix='operator_production_unit_', dir=resolved)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.popen = self.patch(mock.patch.object(
            production.subprocess, 'Popen', side_effect=AssertionError('unit test attempted a real launch')))
        self.patch(mock.patch.object(production, 'check_backend', return_value=None))
        self.patch(mock.patch.object(production.os, 'sched_getaffinity', return_value=set(range(16))))
        self.patch(mock.patch.object(production, 'available_ram', return_value=1024 * production.GIB))
        self.patch(mock.patch.object(production, 'print', create=True))
        self.source = self.root / 'inert_backend.py'
        self.source.write_text('"""Inert unit-test source pin, never executed."""\n', encoding='ascii')
        self.config = dict(schema=production.SCHEMA, project_root=str(PROJECT),
            pitches_um=list(range(2, 11)), prepared_totals_um=list(range(20, 101, 10)),
            diameter_pitch_ratio=.5, resources=dict(cores_per_job=8, reserve_ram_gib=128), recipes=[])

    def test_shipped_profile_has_no_fixed_shared_ram_reserve(self):
        config = production.load_config(PROJECT / 'YAP/input/operator_production.yaml')
        self.assertEqual(config['resources']['reserve_ram_gib'], 0)
        plan = production.make_plan(config, dict(entries=[]),
            run_dir=self.root / 'shipped_profile', repo_root=PROJECT / 'YAP',
            python='/usr/bin/python3')
        self.assertEqual(plan['resources']['reserve_ram_gib'], 0)
        self.assertFalse(plan['experiments_launched'])
        self.popen.assert_not_called()

    def patch(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode('ascii')
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def entry(self, *, pitch=2, si=7, name='BI2_SI7', **changes):
        geometry = dict(bi_pitch_um=pitch, si_tsv_thickness_um=si,
            diameter_pitch_ratio=.5, bi_layer_count=1, bi_thickness_um=1.5,
            beol_thickness_um=10, total_thickness_um=si + 11.5)
        geometry.update(changes)
        return dict(id=name, geometry=geometry)

    def recipe(self, *, stage='reference', variant='inner', total=20):
        return dict(stage=stage, select=dict(pitch_um=2, total_um=total, variant=variant),
            foreground=True, argv=['unit-test-inert-backend', '{stage_dir}/REPORT.json'],
            source_pins={str(self.source): production.source_hash(self.source)},
            prerequisites={} if stage == 'reference' else {str(self.source): production.source_hash(self.source)},
            memory_gib=8, file_cap_gib=1, timeout_seconds=30,
            report=dict(path='{stage_dir}/REPORT.json', equals={'/state': 'UNIT_FIXTURE_CLOSED'}))

    def plan(self, *, recipes=None, entries=None, full=False):
        config = deepcopy(self.config)
        if not full:
            config.update(pitches_um=[2], prepared_totals_um=[20])
        config['recipes'] = recipes or []
        return production.make_plan(config, dict(entries=entries or []),
            run_dir=self.root / 'run', repo_root=PROJECT / 'YAP', python='/usr/bin/python3')

    def bind_plan(self, plan):
        run = Path(plan['run_dir'])
        run.mkdir(parents=True, exist_ok=True)
        catalog = self.root / 'CATALOG.json'
        catalog_sha = self.write_json(catalog, dict(entries=[]))
        plan['catalog'] = dict(path=str(catalog), sha256=catalog_sha, entries=0)
        plan['driver_sha256'] = production.source_hash(production.__file__)
        path = run / 'PLAN.json'
        sha = self.write_json(path, plan)
        return SimpleNamespace(plan=path, plan_sha256=sha, stage='reference',
                               case=None, jobs=1, execute=True)

    def configured_task(self, plan):
        return next(task for task in plan['tasks'] if task['status'] == 'CONFIGURED_REQUIRES_PREFLIGHT')

    def complete(self, plan, args):
        task = self.configured_task(plan)
        _, report, stage = production.resolve_task(plan, task)
        self.write_json(report, dict(state='UNIT_FIXTURE_CLOSED'))
        receipt = dict(task=task['id'], plan_sha256=args.plan_sha256,
                       report=production.report_checks(report, task['recipe']['report']['equals']))
        self.write_json(stage / 'COMPLETE.json', receipt)
        return task, report, stage, receipt

    def test_grid_has_81_nominal_queries_and_162_unique_variants(self):
        rows = production.grid(self.config, dict(entries=[]))
        self.assertEqual(len(rows), 162)
        self.assertEqual(len({row['id'] for row in rows}), 162)
        self.assertEqual(len({(row['pitch_um'], row['total_um']) for row in rows}), 81)
        self.assertEqual({row['variant'] for row in rows}, {'inner', 'outer'})
        for row in rows:
            self.assertEqual(row['diameter_um'] / row['pitch_um'], .5)
            self.assertEqual(row['si_um'], row['total_um'] - (13 if row['variant'] == 'inner' else 11.5))
            self.assertEqual(row['coupon_total_um'], row['si_um'] + 11.5)
            self.assertFalse(row['payload_verified'])
            self.assertFalse(row['assembly_accuracy_inferred'])
            self.assertFalse(row['interpolation_accuracy_inferred'])

    def test_common_cell_uses_pitch_and_tsv_period(self):
        rows = production.grid(self.config, dict(entries=[]))
        expected = {2: 6, 3: 6, 4: 12, 5: 30, 6: 6, 7: 42, 8: 24, 9: 18, 10: 30}
        self.assertTrue(all(row['cell_um'] == expected[row['pitch_um']] for row in rows))

    def test_invalid_grid_configurations_are_rejected(self):
        for changes in (dict(pitches_um=[2, 2]), dict(pitches_um=[True]), dict(pitches_um=[11]),
                        dict(prepared_totals_um=[25]), dict(prepared_totals_um=[20, 20]),
                        dict(prepared_totals_um=[]), dict(diameter_pitch_ratio=.4)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                production.grid(dict(self.config, **changes), dict(entries=[]))

    def test_catalog_match_is_exact_si_not_coupon_total(self):
        rows = production.grid(self.config, dict(entries=[self.entry()]))
        matched = [row for row in rows if row['catalog_entry']]
        self.assertEqual([(row['id'], row['coupon_total_um']) for row in matched], [('BI2_T20_inner', 18.5)])

    def test_outer_catalog_match_requires_different_si(self):
        rows = production.grid(self.config, dict(entries=[self.entry(si=8.5, name='OUTER')]))
        self.assertEqual([row['id'] for row in rows if row['catalog_entry']], ['BI2_T20_outer'])

    def test_incompatible_catalog_geometry_does_not_skip(self):
        for changes in (dict(diameter_pitch_ratio=.4), dict(bi_layer_count=2),
                        dict(bi_thickness_um=3), dict(beol_thickness_um=9)):
            with self.subTest(changes=changes):
                rows = production.grid(self.config, dict(entries=[self.entry(**changes)]))
                self.assertFalse(any(row['catalog_entry'] for row in rows))

    def test_ambiguous_catalog_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            production.grid(self.config, dict(entries=[self.entry(), self.entry(name='DUPLICATE')]))

    def test_no_recipes_is_blocked_not_launch_ready(self):
        plan = self.plan(full=True)
        self.assertEqual(len(plan['tasks']), 810)
        self.assertEqual({task['status'] for task in plan['tasks']}, {'BLOCKED_IMPLEMENTATION'})
        self.assertEqual(production.summary(plan)['nominal_design_queries'], 81)
        self.assertFalse(plan['experiments_launched'])
        self.assertFalse(plan['automatic_retry'])
        self.assertFalse(plan['automatic_cleanup'])

    def test_full_grid_component_and_interpolation_targets_are_metadata_only(self):
        plan = self.plan(full=True)
        components = plan['component_targets']
        self.assertEqual(components['bi_pitches_um'], list(range(2, 11)))
        expected_si = sorted({total - offset for total in range(20, 101, 10)
                              for offset in (13, 11.5)})
        self.assertEqual(components['si_thicknesses_um'], expected_si)
        self.assertEqual(len(expected_si), 18)
        self.assertEqual(components['beol_thickness_um'], 10)
        families = components['cell_families']
        self.assertEqual({family['cell_um']: family['pitches_um'] for family in families},
                         {6: [2, 3, 6], 12: [4], 18: [9], 24: [8], 30: [5, 10], 42: [7]})
        self.assertEqual(len(families), 6)
        self.assertTrue(all(family['reuse_certified'] is False for family in families))
        targets = plan['interpolation_validation_targets']
        expected = {(pitch, variant, left, left + 10, left + 5)
                    for pitch in range(2, 11) for variant in ('inner', 'outer')
                    for left in range(20, 100, 10)}
        observed = {(target['pitch_um'], target['variant'], target['left_prepared_total_um'],
                     target['right_prepared_total_um'], target['heldout_prepared_total_um'])
                    for target in targets}
        self.assertEqual(len(targets), 144)
        self.assertEqual(observed, expected)
        self.assertEqual({target['heldout_prepared_total_um'] for target in targets},
                         set(range(25, 96, 10)))
        for target in targets:
            self.assertIs(target['exact_point_operator_requested'], False)
            self.assertEqual(target['status'], 'BLOCKED_IMPLEMENTATION')
            self.assertIn('not construct an exact-point operator', target['purpose'])
        self.assertEqual(tuple(plan['stages']), ('reference', 'construct', 'history', 'assess', 'publish'))
        self.assertEqual(len(plan['tasks']), 810)
        self.assertEqual({task['status'] for task in plan['tasks']}, {'BLOCKED_IMPLEMENTATION'})
        self.assertFalse(plan['experiments_launched'])
        self.popen.assert_not_called()

    def test_explicit_recipe_can_reassess_a_catalogued_geometry(self):
        plan = self.plan(recipes=[self.recipe()], entries=[self.entry()])
        tasks = [task for task in plan['tasks'] if task['case'] == 'BI2_T20_inner']
        self.assertEqual(tasks[0]['status'], 'CONFIGURED_REQUIRES_PREFLIGHT')
        self.assertEqual({task['status'] for task in tasks[1:]}, {'CATALOGUED_SCOPE_ONLY'})
        self.assertFalse(plan['rows'][0]['payload_verified'])

    def test_recipe_selector_configures_only_one_stage_and_variant(self):
        plan = self.plan(recipes=[self.recipe()])
        configured = [task['id'] for task in plan['tasks'] if task['status'] == 'CONFIGURED_REQUIRES_PREFLIGHT']
        self.assertEqual(configured, ['BI2_T20_inner_reference'])

    def test_unqualified_backend_is_not_executable(self):
        with self.assertRaisesRegex(ValueError, 'BLOCKED_BACKEND_QUALIFICATION'):
            ORIGINAL_CHECK_BACKEND(self.recipe(), ['unit-test-inert-backend'])

    def test_duplicate_and_out_of_grid_selectors_rejected(self):
        for recipes in ([self.recipe(), self.recipe()], [self.recipe(total=30)]):
            with self.subTest(recipes=recipes), self.assertRaises(ValueError):
                self.plan(recipes=recipes)

    def test_recipe_contract_negatives(self):
        changes = [dict(stage='invented'), dict(foreground=False), dict(argv='shell command'),
                   dict(argv=['']), dict(argv=['{unknown}']), dict(argv=['{pitch_um:04d}']),
                   dict(argv=['{variant!r}']), dict(source_pins={}), dict(memory_gib=0),
                   dict(memory_gib=True), dict(file_cap_gib=-1), dict(timeout_seconds=0),
                   dict(select=dict(pitch_um=2, total_um=20, variant='both')),
                   dict(report=dict(path='{stage_dir}/REPORT.json', equals={})),
                   dict(source_pins={str(self.source): 'TODO'})]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                production.validate_recipe(dict(self.recipe(), **change))

    def test_later_recipe_requires_predecessor_evidence(self):
        recipe = self.recipe(stage='history')
        self.assertIs(production.validate_recipe(recipe), recipe)
        recipe['prerequisites'] = {}
        with self.assertRaisesRegex(ValueError, 'predecessor'):
            production.validate_recipe(recipe)

    def test_dry_run_never_calls_popen_or_creates_attempts(self):
        plan = self.plan(recipes=[self.recipe()])
        args = self.bind_plan(plan)
        args.execute = False
        with mock.patch.object(production, 'available_ram', side_effect=AssertionError('dry run probed RAM')):
            self.assertEqual(production.run_command(args), 0)
        self.popen.assert_not_called()
        self.assertFalse((Path(plan['run_dir']) / 'attempts').exists())

    def test_execute_without_backend_is_blocked(self):
        args = self.bind_plan(self.plan())
        with self.assertRaisesRegex(ValueError, 'no configured'):
            production.run_command(args)
        self.popen.assert_not_called()

    def test_exclusive_run_lock_rejects_second_invocation(self):
        args = self.bind_plan(self.plan(recipes=[self.recipe()]))
        lock = Path(args.plan).parent / 'RUN.lock'
        with lock.open('a+b') as owner:
            production.fcntl.flock(owner.fileno(), production.fcntl.LOCK_EX | production.fcntl.LOCK_NB)
            with mock.patch.object(production, '_run_command') as body:
                with self.assertRaisesRegex(ValueError, 'another invocation owns this queue'):
                    production.run_command(args)
                body.assert_not_called()
        # A retained lock filename alone is not evidence of a live lock owner.
        with mock.patch.object(production, '_run_command', return_value=0) as body:
            self.assertEqual(production.run_command(args), 0)
            body.assert_called_once_with(args)
        self.popen.assert_not_called()

    def test_symlink_run_lock_is_rejected_without_touching_target(self):
        args = self.bind_plan(self.plan(recipes=[self.recipe()]))
        target = self.root / 'lock_target.txt'
        target.write_text('retained target\n', encoding='ascii')
        (Path(args.plan).parent / 'RUN.lock').symlink_to(target)
        with mock.patch.object(production, '_run_command') as body:
            with self.assertRaises((ValueError, OSError)):
                production.run_command(args)
            body.assert_not_called()
        self.assertEqual(target.read_text(encoding='ascii'), 'retained target\n')
        self.popen.assert_not_called()

    def test_original_backend_check_rejects_unqualified_recipe(self):
        self.assertEqual(production.QUALIFIED_BACKENDS, {})
        recipe = self.recipe()
        for backend in (None, 'UNQUALIFIED_UNIT_FIXTURE'):
            with self.subTest(backend=backend):
                if backend is not None:
                    recipe['backend'] = backend
                with self.assertRaisesRegex(ValueError, 'BLOCKED_BACKEND_QUALIFICATION'):
                    ORIGINAL_CHECK_BACKEND(recipe, recipe['argv'])
        self.popen.assert_not_called()

    def test_execute_uses_original_unqualified_backend_gate_before_claim(self):
        plan = self.plan(recipes=[self.recipe()])
        args = self.bind_plan(plan)
        with mock.patch.object(production, 'check_backend', ORIGINAL_CHECK_BACKEND):
            with self.assertRaisesRegex(ValueError, 'BLOCKED_BACKEND_QUALIFICATION'):
                production.run_command(args)
        self.assertFalse((Path(plan['run_dir']) / 'attempts').exists())
        self.popen.assert_not_called()

    def test_case_selector_does_not_run_another_case(self):
        args = self.bind_plan(self.plan(recipes=[self.recipe()]))
        args.case = ['BI2_T20_outer']
        with self.assertRaisesRegex(ValueError, 'no configured'):
            production.run_command(args)
        self.popen.assert_not_called()

    def test_completed_receipt_is_verified_and_skipped(self):
        plan = self.plan(recipes=[self.recipe()])
        args = self.bind_plan(plan)
        self.complete(plan, args)
        self.assertEqual(production.run_command(args), 0)
        self.popen.assert_not_called()

    def test_completed_receipt_wrong_plan_or_task_is_rejected(self):
        for field, value in (('plan_sha256', '0' * 64), ('task', 'OTHER_TASK')):
            with self.subTest(field=field):
                plan = self.plan(recipes=[self.recipe()])
                args = self.bind_plan(plan)
                _, _, stage, receipt = self.complete(plan, args)
                receipt[field] = value
                self.write_json(stage / 'COMPLETE.json', receipt)
                with self.assertRaisesRegex(ValueError, 'completed-stage'):
                    production.run_command(args)
        self.popen.assert_not_called()

    def test_completed_report_changed_bytes_is_rejected(self):
        plan = self.plan(recipes=[self.recipe()])
        args = self.bind_plan(plan)
        _, report, _, _ = self.complete(plan, args)
        self.write_json(report, dict(state='UNIT_FIXTURE_CLOSED', changed=True))
        with self.assertRaisesRegex(ValueError, 'completed-stage'):
            production.run_command(args)
        self.popen.assert_not_called()

    def test_failed_attempt_is_retained_for_manual_review(self):
        plan = self.plan(recipes=[self.recipe()])
        args = self.bind_plan(plan)
        _, _, stage = production.resolve_task(plan, self.configured_task(plan))
        failure = stage / 'FAILED.json'
        self.write_json(failure, dict(error='retained unit-test failure', automatic_retry=False))
        before = failure.read_bytes()
        self.assertEqual(production.run_command(args), 1)
        self.assertEqual(failure.read_bytes(), before)
        self.assertFalse((stage / 'COMPLETE.json').exists())
        self.popen.assert_not_called()

    def test_incomplete_attempt_is_not_restarted(self):
        plan = self.plan(recipes=[self.recipe()])
        args = self.bind_plan(plan)
        _, _, stage = production.resolve_task(plan, self.configured_task(plan))
        stage.mkdir(parents=True)
        self.assertEqual(production.run_command(args), 1)
        self.assertEqual(list(stage.iterdir()), [])
        self.popen.assert_not_called()

    def test_incomplete_later_case_blocks_other_configured_launches(self):
        plan = self.plan(recipes=[self.recipe(), self.recipe(variant='outer')])
        args = self.bind_plan(plan)
        ready = [task for task in plan['tasks'] if task['status'] == 'CONFIGURED_REQUIRES_PREFLIGHT']
        self.assertEqual([task['case'] for task in ready], ['BI2_T20_inner', 'BI2_T20_outer'])
        _, _, fresh = production.resolve_task(plan, ready[0])
        _, _, retained = production.resolve_task(plan, ready[1])
        claim = retained / 'CLAIM.json'
        self.write_json(claim, dict(task=ready[1]['id'], note='possibly live; requires manual review'))
        before = claim.read_bytes()
        with mock.patch.object(production, 'available_ram', side_effect=AssertionError('pre-scan must refuse first')):
            self.assertEqual(production.run_command(args), 1)
        self.assertFalse(fresh.exists())
        self.assertEqual(claim.read_bytes(), before)
        self.assertEqual(list(retained.iterdir()), [claim])
        self.popen.assert_not_called()

    def test_ram_refusal_charges_two_full_caps_plus_reserve(self):
        plan = self.plan(recipes=[self.recipe()])
        args = self.bind_plan(plan)
        required = (128 + 2 * 8) * production.GIB
        with mock.patch.object(production, 'available_ram', return_value=required - 1):
            self.assertEqual(production.run_command(args), 75)
        self.assertFalse((Path(plan['run_dir']) / 'attempts').exists())
        self.popen.assert_not_called()

    def test_case_and_stage_filter_cannot_bypass_another_incomplete_attempt(self):
        plan = self.plan(recipes=[self.recipe()])
        args = self.bind_plan(plan)
        args.case = ['BI2_T20_inner']
        other = Path(plan['run_dir']) / 'attempts' / 'BI2_T20_outer_construct'
        other.mkdir(parents=True)
        self.assertEqual(production.run_command(args), 1)
        self.popen.assert_not_called()

    def test_metadata_read_ignores_access_time_but_not_identity(self):
        path = self.root / 'access_time.json'
        self.write_json(path, dict(state='UNIT_FIXTURE'))
        info = path.stat()
        fields = ('st_dev', 'st_ino', 'st_mode', 'st_uid', 'st_gid', 'st_nlink',
                  'st_size', 'st_mtime_ns', 'st_ctime_ns')
        values = {key: getattr(info, key) for key in fields}
        first = SimpleNamespace(**values, st_atime_ns=1)
        second = SimpleNamespace(**values, st_atime_ns=2)
        with mock.patch.object(production.os, 'fstat', side_effect=[first, second]):
            self.assertEqual(production.read_json(path)[0], dict(state='UNIT_FIXTURE'))
        changed = SimpleNamespace(**dict(values, st_ino=values['st_ino'] + 1), st_atime_ns=2)
        with mock.patch.object(production.os, 'fstat', side_effect=[first, changed]):
            with self.assertRaisesRegex(ValueError, 'changed'):
                production.read_json(path)

    def test_cpu_affinity_refusal_precedes_popen(self):
        args = self.bind_plan(self.plan(recipes=[self.recipe()]))
        args.jobs = 3
        with self.assertRaisesRegex(ValueError, 'CPU affinity'):
            production.run_command(args)
        self.popen.assert_not_called()

    def test_source_pin_mismatch_precedes_claim_and_popen(self):
        recipe = self.recipe()
        recipe['source_pins'][str(self.source)] = '0' * 64
        plan = self.plan(recipes=[recipe])
        args = self.bind_plan(plan)
        with self.assertRaisesRegex(ValueError, 'drift'):
            production.run_command(args)
        self.assertFalse((Path(plan['run_dir']) / 'attempts').exists())
        self.popen.assert_not_called()

    def test_prerequisite_pin_mismatch_is_rejected(self):
        recipe = self.recipe(stage='history')
        recipe['prerequisites'][str(self.source)] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'drift'):
            production.check_pins(recipe)

    def test_prerequisite_cannot_shadow_conflicting_source_pin(self):
        recipe = self.recipe(stage='history')
        recipe['source_pins'][str(self.source)] = '0' * 64
        with self.assertRaises(ValueError):
            production.check_pins(recipe)

    def test_plan_hash_and_catalog_drift_are_rejected(self):
        plan = self.plan(recipes=[self.recipe()])
        args = self.bind_plan(plan)
        correct = args.plan_sha256
        args.plan_sha256 = '0' * 64
        with self.assertRaisesRegex(ValueError, 'PLAN SHA'):
            production.run_command(args)
        args.plan_sha256 = correct
        self.write_json(Path(plan['catalog']['path']), dict(entries=[], changed=True))
        with self.assertRaisesRegex(ValueError, 'catalog changed'):
            production.run_command(args)
        self.popen.assert_not_called()

    def test_path_guards_reject_snapshots_parent_traversal_and_output_roots(self):
        for path in (self.root / '.snapshot' / 'x', self.root / '#snapshot' / 'x', self.root / '..' / 'x'):
            with self.subTest(path=str(path)), self.assertRaises(ValueError):
                production.literal(path)
        for path in ('/tmp/unit-only', '/u1/ee/zhichao/unit-only', '/w/ee.00/puneet/zhichao/unit-only'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                production.literal(path, output=True)

    def test_report_must_be_inside_private_stage_directory(self):
        recipe = self.recipe()
        recipe['report']['path'] = str(self.root / 'elsewhere' / 'REPORT.json')
        plan = self.plan(recipes=[recipe])
        with self.assertRaisesRegex(ValueError, 'private stage'):
            production.resolve_task(plan, self.configured_task(plan))

    def test_symlink_report_is_rejected_even_with_correct_json(self):
        target = self.root / 'actual_report.json'
        self.write_json(target, dict(state='UNIT_FIXTURE_CLOSED'))
        alias = self.root / 'REPORT.json'
        alias.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            production.report_checks(alias, {'/state': 'UNIT_FIXTURE_CLOSED'})
        self.popen.assert_not_called()

    def test_symlink_source_is_rejected_even_with_correct_hash(self):
        alias = self.root / 'source_alias.py'
        alias.symlink_to(self.source)
        recipe = self.recipe()
        recipe['source_pins'] = {str(alias): production.source_hash(self.source)}
        with self.assertRaisesRegex(ValueError, 'symlink'):
            production.validate_recipe(recipe)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            production.check_pins(recipe)
        self.popen.assert_not_called()

    def test_symlink_path_ancestor_is_rejected_for_inputs_and_outputs(self):
        directory = self.root / 'real_directory'
        directory.mkdir()
        self.write_json(directory / 'REPORT.json', dict(state='UNIT_FIXTURE_CLOSED'))
        alias = self.root / 'directory_alias'
        alias.symlink_to(directory, target_is_directory=True)
        for path, output in ((alias / 'REPORT.json', False), (alias / 'new_output.json', True)):
            with self.subTest(path=str(path)), self.assertRaisesRegex(ValueError, 'symlink'):
                production.literal(path, output=output)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            production.read_json(alias / 'REPORT.json')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            production.emit(alias / 'new_output.json', dict(state='UNIT_FIXTURE_CLOSED'))
        self.assertFalse((directory / 'new_output.json').exists())
        self.popen.assert_not_called()

    def test_report_pointer_decodes_escapes_and_array_indices(self):
        report = self.root / 'report.json'
        self.write_json(report, {'a/b': {'~key': [dict(passed=True)]}, 'count': 1})
        proof = production.report_checks(report, {'/a~1b/~0key/0/passed': True, '/count': 1})
        self.assertEqual(proof['sha256'], production.source_hash(report))

    def test_report_pointer_requires_prefix_exact_type_and_value(self):
        report = self.root / 'report.json'
        self.write_json(report, dict(count=1, state='OPEN'))
        for checks in ({'count': 1}, {'/count': True}, {'/count': 1.0}, {'/state': 'CLOSED'}):
            with self.subTest(checks=checks), self.assertRaises(ValueError):
                production.report_checks(report, checks)
        with self.assertRaises(KeyError):
            production.report_checks(report, {'/absent': True})

    def test_report_array_pointer_rejects_non_json_indices(self):
        report = self.root / 'report.json'
        self.write_json(report, dict(rows=['UNIT_FIXTURE_CLOSED']))
        for index in ('-1', '+0', '00'):
            with self.subTest(index=index), self.assertRaises((ValueError, KeyError, IndexError)):
                production.report_checks(report, {'/rows/' + index: 'UNIT_FIXTURE_CLOSED'})

    def test_json_rejects_duplicate_nonfinite_and_nonobject_documents(self):
        path = self.root / 'bad.json'
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '[]'):
            with self.subTest(raw=raw):
                path.write_text(raw, encoding='ascii')
                with self.assertRaises(ValueError):
                    production.read_json(path)

    def test_mock_backend_exit_zero_still_requires_report_scope(self):
        for valid in (False, True):
            with self.subTest(valid=valid):
                plan = self.plan(recipes=[self.recipe(variant='outer' if valid else 'inner')])
                plan['run_dir'] = str(self.root / ('valid_scope_run' if valid else 'invalid_scope_run'))
                args = self.bind_plan(plan)
                task = self.configured_task(plan)
                _, report, stage = production.resolve_task(plan, task)

                def fake_popen(command, **kwargs):
                    self.assertEqual(command[0], '/usr/bin/prlimit')
                    self.assertIn('--as=8589934592:8589934592', command)
                    self.assertTrue(kwargs['start_new_session'])
                    self.assertEqual(kwargs['stdin'], production.subprocess.DEVNULL)
                    self.assertTrue(Path(kwargs['env']['TMPDIR']).is_relative_to(stage / 'environment'))
                    self.write_json(report, dict(state='UNIT_FIXTURE_CLOSED' if valid else 'WRONG_SCOPE'))
                    return SimpleNamespace(pid=123456, poll=lambda: 0)

                self.popen.side_effect = fake_popen
                self.assertEqual(production.run_command(args), 0 if valid else 1)
                self.assertEqual((stage / 'COMPLETE.json').exists(), valid)
                self.assertEqual((stage / 'FAILED.json').exists(), not valid)


if __name__ == '__main__':
    unittest.main()
