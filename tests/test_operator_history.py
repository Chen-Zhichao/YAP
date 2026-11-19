"""Tiny original-core/Newton coordinator tests, not Cu-law qualification.

Only six named source files are read when Main executes this suite. No native
payload, production geometry, PARDISO, history file or subprocess is used.
"""

import builtins
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from tools import operator_history as history


D = Path('/app/nanocad/projects/personal/zchen/YAP_STACK/operator_repair_20261015/development/trial_forcing_closure_v1/continuous_cu_screen_v1')
SOURCES = dict(core='causal_sparse_fe_core.py', newton='causal_sparse_newton.py',
    native_state='native_monolithic_history.py', clock='common_cell_native_c1_history.py',
    builder='native_monolithic_assembly.py', policy='global_coupled_c1_rom.py')


class OperatorHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from scipy import sparse, linalg
        from scipy.sparse import linalg as spla
        cls.np, cls.sparse, cls.linalg, cls.spla = np, sparse, linalg, spla
        cls.raw = {}
        for role, filename in SOURCES.items():
            with (D / filename).open('rb') as stream:
                raw = stream.read(256 * 1024 + 1)
            if not 0 < len(raw) <= 256 * 1024 or hashlib.sha256(raw).hexdigest() != history.SOURCE_PINS[role]:
                raise AssertionError('small frozen history source differs: ' + role)
            cls.raw[role] = raw

        class FakeCuState:
            def __init__(self, plastic_strain, creep_strain, equivalent_plastic_strain, equivalent_creep_strain):
                self.plastic_strain, self.creep_strain = plastic_strain, creep_strain
                self.equivalent_plastic_strain = equivalent_plastic_strain
                self.equivalent_creep_strain = equivalent_creep_strain

            @classmethod
            def zeros(state_type, g):
                return state_type(np.zeros((g, 6)), np.zeros((g, 6)), np.zeros(g), np.zeros(g))

            def copy(self):
                return type(self)(**{name: getattr(self, name).copy() for name in history.FIELDS})

        cls.FakeCuState = FakeCuState

    @staticmethod
    def isotropic(elastic, poisson):
        import numpy as np
        mu = elastic / (2 * (1 + poisson))
        lam = elastic * poisson / ((1 + poisson) * (1 - 2 * poisson))
        c = np.zeros((6, 6))
        c[:3, :3] = lam
        c[np.arange(3), np.arange(3)] += 2 * mu
        c[3:, 3:] = mu * np.eye(3)
        return c

    def setUp(self):
        self.bound = history.bind_sources(self.raw)
        self.check = mock.Mock()
        self.entries, self.exits, self.factor_calls = [], [], []
        self.factor_fail_at, self.suppress_failure = None, False
        self.kernel_log = []
        self.configure(n=2, g=1)

    def configure(self, *, n, g):
        np = self.np
        self.n, self.g = n, g
        self.api = SimpleNamespace(np=np, sparse=self.sparse, linalg=self.linalg, spla=self.spla,
            cu=SimpleNamespace(CuBatchState=self.FakeCuState, isotropic_stiffness=self.isotropic))
        self.card = SimpleNamespace(elastic=lambda temperature: (100., .25), alpha=lambda temperature: 1e-6)
        self.reference = SimpleNamespace(properties=lambda name, temperature: (100., .25, 1e-6))
        k = np.diag(np.arange(2., n + 2))
        c = .01 * (1 + np.arange(n * 6).reshape(n, 6) / (n * 6))
        kaa = np.diag(np.arange(2., 8.)) + c.T @ np.linalg.solve(k, c)
        self.assembly = dict(kff=self.sparse.csr_matrix(k), kaa=kaa, c=c,
            fm=1e-4 * np.arange(1., 7.), ff=1e-4 * np.arange(1., n + 1),
            B_macro=np.zeros((6 * g, 6)), B_free=self.sparse.csr_matrix((6 * g, n)), weights=np.ones(g))
        for value in self.assembly.values():
            if self.sparse.issparse(value):
                for array in (value.data, value.indices, value.indptr):
                    array.setflags(write=False)
            else:
                value.setflags(write=False)
        self.policy = SimpleNamespace(**self.bound.policy, validate=mock.Mock())
        self.dense = self.dense_clock()
        cref, alpha = self.isotropic(100., .25), np.array([1e-6] * 3 + [0.] * 3)

        def update(strain, temperature, dt, incoming, *, card, creep_active, plastic_active, elapsed_time_s):
            self.kernel_log.append((temperature, dt, creep_active, elapsed_time_s))
            state = incoming.copy()
            # Synthetic bookkeeping only; recomputed from the SAME incoming
            # state for each trial evaluation, never accumulated by evaluation.
            if creep_active:
                state.creep_strain += dt * 1e-9
                state.equivalent_creep_strain += dt * 1e-9
            return SimpleNamespace(state=state, stress_mpa=(strain - (temperature - 25.) * alpha) @ cref.T,
                plastic_increment=np.zeros_like(strain), creep_increment=np.zeros_like(strain),
                plastic_dissipation_mpa=np.zeros(len(strain)), creep_dissipation_mpa=np.zeros(len(strain)))

        def prepare(*args, **kwargs):
            return cref

        self.kernels = SimpleNamespace(update_batch=update, prepare_tangent=prepare,
                                        apply_tangent=lambda direction, prepared: direction @ prepared.T)

    def dense_clock(self):
        result = []
        for index in range(1081):
            cycle, local = ((index - 1) // 360, (index - 1) % 360 + 1) if index else (0, 0)
            if local <= 160:
                time, temperature, creep = 300. + 650. * local / 160, 25. + 325. * local / 160, False
            elif local <= 200:
                time, temperature, creep = 950. + 90. * (local - 160), 350., True
            else:
                time, temperature, creep = 4550. + 650. * (local - 200) / 160, 350. - 325. * (local - 200) / 160, False
            result.append(SimpleNamespace(result_set=index + 1, time_s=time + 4900. * cycle,
                temperature_c=temperature, creep_active=creep, plastic_active=True))
        return result

    def physical_context(self, matrix):
        owner, np = self, self.np

        class Physical:
            def solve(self, rhs):
                owner.factor_calls.append(rhs.copy())
                if owner.factor_fail_at == len(owner.factor_calls):
                    raise RuntimeError('unit physical solve failure')
                return np.linalg.solve(matrix.toarray(), rhs)

        class Context:
            def __enter__(self):
                owner.entries.append(matrix.copy())
                return Physical()

            def __exit__(self, kind, error, traceback):
                owner.exits.append(kind)
                return owner.suppress_failure

        return Context()

    def coordinator(self, **changes):
        arguments = dict(assembly=self.assembly, reference=self.reference, card=self.card,
            kernels=self.kernels, policy=self.policy, dense1081=self.dense,
            physical_factor_context=self.physical_context,
            limits=dict(retained_array_bytes=1024**2, diagnostic_bytes_per_step=64 * 1024),
            check=self.check, case='SOFTWARE_LINEAR_NATIVE_NOT_PRODUCTION', native_sha256='0' * 64)
        arguments.update(changes)
        return history.NativeC1(self.api, self.bound, **arguments)

    def assert_failed_closed(self, coordinator, accepted=0):
        self.assertEqual(coordinator.status, 'FAILED')
        self.assertEqual(coordinator.last_accepted_ordinal, accepted)
        self.assertEqual(len(self.entries), 1)
        self.assertEqual(len(self.exits), 1)
        with self.assertRaisesRegex(ValueError, 'no automatic retry'):
            coordinator.run()

    def wrap_trial(self, transform):
        original = self.bound.newton.trial_with_original_policy

        def wrapped(*args, **kwargs):
            trial, diagnostics = original(*args, **kwargs)
            return transform(trial, diagnostics, args)

        return mock.patch.object(self.bound.newton, 'trial_with_original_policy', side_effect=wrapped)

    def test_actual_core_newton_c1_counts_oracle_and_readonly_training_frames(self):
        np = self.np
        for n, g in ((2, 1), (4, 3)):
            with self.subTest(n=n, g=g):
                self.configure(n=n, g=g)
                self.entries.clear(); self.exits.clear(); self.factor_calls.clear()
                coordinator = self.coordinator()
                result = coordinator.run()
                self.assertEqual(coordinator.status, 'CLOSED_NATIVE_C1_UNVALIDATED')
                self.assertEqual(result.q6_history.shape, (25, 6))
                self.assertEqual(set(result.training_coordinates), {8, 16})
                self.assertEqual(result.report['training_original_ordinals'], {8: 160, 16: 200})
                coupled = np.block([[self.assembly['kaa'], self.assembly['c'].T],
                                    [self.assembly['c'], self.assembly['kff'].toarray()]])
                unit = np.linalg.solve(coupled, np.r_[self.assembly['fm'], self.assembly['ff']])
                for ordinal, row in enumerate(result.observations):
                    expected = (row['temperature_c'] - 25.) * unit
                    np.testing.assert_allclose(result.q6_history[ordinal], expected[:6], rtol=2e-8, atol=1e-12)
                    if ordinal in (8, 16):
                        snapshot = result.training_coordinates[ordinal]
                        self.assertEqual(snapshot.shape, (6 + n,))
                        self.assertFalse(snapshot.flags.writeable)
                        np.testing.assert_array_equal(snapshot[:6], result.q6_history[ordinal])
                        np.testing.assert_allclose(snapshot, expected, rtol=2e-8, atol=1e-12)
                        self.assertFalse(np.shares_memory(snapshot, result.final_coordinates))
                self.assertFalse(result.q6_history.flags.writeable)
                self.assertFalse(result.final_coordinates.flags.writeable)
                for field in history.FIELDS:
                    self.assertFalse(getattr(result.final_native_state.cu, field).flags.writeable)
                np.testing.assert_array_equal(result.final_native_state.micro, result.final_coordinates[6:])
                self.assertFalse(np.any(result.final_native_state.birth_micro))
                self.assertFalse(np.any(result.final_native_state.birth_macro))
                self.assertEqual(result.report['Cu_GP'], g)
                self.assertEqual(result.report['native_body_coordinates'], n)
                self.assertEqual(result.report['retained_output_array_bytes'], 8 * (14 * g + 5 * n + 174))
                self.assertFalse(result.report['retained_output_budget_is_peak_RAM'])
                self.assertFalse(result.report['cooldown_used_for_training'])
                self.assertEqual(len(self.entries), 1)
                self.assertEqual(len(self.exits), 1)
                np.testing.assert_array_equal(self.entries[0].toarray(), self.assembly['kff'].toarray())
                with self.assertRaises(ValueError):
                    coordinator.run()

    def test_virgin_cu_initialization_and_creep_off_exact_carry(self):
        original = self.bound.state.committed
        commits = []

        def committed(np, core, coordinates, cu):
            commits.append({field: getattr(cu, field).copy() for field in history.FIELDS})
            return original(np, core, coordinates, cu)

        with mock.patch.object(self.bound.state, 'committed', side_effect=committed):
            result = self.coordinator().run()
        self.assertEqual(len(commits), 25)
        self.assertTrue(all(not self.np.any(value) for value in commits[0].values()))
        for ordinal, row in enumerate(result.observations[1:], 1):
            for field in ('creep_strain', 'equivalent_creep_strain'):
                if not row['creep_active']:
                    self.np.testing.assert_array_equal(commits[ordinal][field], commits[ordinal - 1][field])
        self.assertTrue(self.np.all(commits[16]['creep_strain'] > 0.))
        self.np.testing.assert_array_equal(commits[16]['creep_strain'], commits[24]['creep_strain'])

    def test_observer_receives_detached_small_records_not_state(self):
        observed = []

        def observer(row):
            observed.append((row['ordinal'], row['original_ordinal']))
            self.assertNotIn('coordinates', row)
            self.assertNotIn('cu', row)
            self.assertEqual(len(row['q6']), 6)
            row['q6'][0] = 12345.
            if 'diagnostics' in row:
                row['diagnostics']['strict_acceptances'] = 999

        result = self.coordinator(observer=observer).run()
        self.assertEqual(len(observed), 25)
        self.assertEqual(observed[8], (8, 160))
        self.assertEqual(observed[16], (16, 200))
        self.assertEqual(observed[-1], (24, 360))
        self.assertNotEqual(result.observations[0]['q6'][0], 12345.)
        self.assertEqual(result.report['acceptance_counts']['strict_acceptances'], 24)
        self.assertEqual(result.observations[-1]['time_s'], 5200.)
        self.assertEqual(result.observations[-1]['temperature_c'], 25.)

    def test_original_fallback_branch_and_diagnostics_are_retained(self):
        original = self.bound.newton.trial_increment

        def increment(*args, **kwargs):
            if kwargs['convergence_measure'] == self.bound.policy['primary_convergence_measure']:
                raise self.bound.newton.SparseConvergenceError('unit primary-only forced failure')
            return original(*args, **kwargs)

        with mock.patch.object(self.bound.newton, 'trial_increment', side_effect=increment):
            result = self.coordinator().run()
        counts = result.report['acceptance_counts']
        self.assertEqual(counts['strict_failures'], 24)
        self.assertEqual(counts['residual_fallback_acceptances'], 24)
        self.assertEqual(counts['strict_acceptances'], 0)
        self.assertEqual(counts['cutbacks'], 0)
        self.assertFalse(result.report['residual_fallback_acceptances_are_strict_JVP_certificates'])

    def test_primary_trial_failure_preserved_with_secondary_micro_mutation(self):
        def fail(*args, **kwargs):
            state = next(iter(args[4].values()))
            state.micro.setflags(write=True)
            state.micro[0] += 1.
            state.micro.setflags(write=False)
            raise RuntimeError('unit original primary failure')

        coordinator = self.coordinator()
        with mock.patch.object(self.bound.newton, 'trial_with_original_policy', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'original primary failure'):
                coordinator.run()
        self.assert_failed_closed(coordinator)
        self.assertIsNotNone(coordinator.secondary_failure)
        self.assertEqual(coordinator.secondary_failure['error_type'], 'ValueError')
        if 'reason' in coordinator.secondary_failure:
            self.assertLessEqual(len(coordinator.secondary_failure['reason']), 500)

    def test_successful_trial_incoming_cu_mutation_is_rejected(self):
        def mutate(trial, diagnostics, args):
            incoming = next(iter(args[4].values())).cu.plastic_strain
            incoming.setflags(write=True)
            incoming.flat[0] += 1.
            incoming.setflags(write=False)
            return trial, diagnostics

        coordinator = self.coordinator()
        with self.wrap_trial(mutate):
            with self.assertRaisesRegex(ValueError, 'incoming native history changed'):
                coordinator.run()
        self.assert_failed_closed(coordinator)

    def test_trial_evaluation_coordinate_mismatch_is_rejected(self):
        def mismatch(trial, diagnostics, args):
            return replace(trial, coordinates=trial.coordinates + .01), diagnostics

        coordinator = self.coordinator()
        with self.wrap_trial(mismatch):
            with self.assertRaisesRegex(ValueError, 'trial/evaluation coordinates'):
                coordinator.run()
        self.assert_failed_closed(coordinator)

    def test_accepted_state_copy_alias_to_trial_is_rejected(self):
        def alias(trial, diagnostics, args):
            cu = trial.evaluation.layer_evaluations[0].cu_trial.state
            cu.copy = lambda: cu
            return trial, diagnostics

        coordinator = self.coordinator()
        with self.wrap_trial(alias):
            with self.assertRaisesRegex(ValueError, 'alias'):
                coordinator.run()
        self.assert_failed_closed(coordinator)

    def test_crossfield_copy_alias_to_incoming_is_rejected(self):
        def alias(trial, diagnostics, args):
            incoming = next(iter(args[4].values())).cu
            cu = trial.evaluation.layer_evaluations[0].cu_trial.state
            original_copy = cu.copy

            def copied():
                value = original_copy()
                value.plastic_strain = incoming.creep_strain
                return value

            cu.copy = copied
            return trial, diagnostics

        coordinator = self.coordinator()
        with self.wrap_trial(alias):
            with self.assertRaisesRegex(ValueError, 'alias'):
                coordinator.run()
        self.assert_failed_closed(coordinator)

    def test_accepted_cu_wrong_dtype_or_shape_is_rejected(self):
        for fault in ('dtype', 'shape'):
            self.entries.clear(); self.exits.clear(); self.factor_calls.clear()

            def malformed(trial, diagnostics, args):
                cu = trial.evaluation.layer_evaluations[0].cu_trial.state
                copied = cu.copy

                def bad_copy():
                    value = copied()
                    value.plastic_strain = value.plastic_strain.astype('float32') if fault == 'dtype' else value.plastic_strain[:, :5].copy()
                    return value

                cu.copy = bad_copy
                return trial, diagnostics

            coordinator = self.coordinator()
            with self.subTest(fault=fault), self.wrap_trial(malformed):
                with self.assertRaises(ValueError):
                    coordinator.run()
            self.assert_failed_closed(coordinator)

    def test_creep_off_mutation_cutback_and_failed_trial_never_commit(self):
        original_commit = self.bound.state.committed
        for fault in ('creep_off', 'cutback', 'trial_error'):
            self.entries.clear(); self.exits.clear(); self.factor_calls.clear()

            def bad_trial(trial, diagnostics, args):
                if fault == 'creep_off':
                    trial.evaluation.layer_evaluations[0].cu_trial.state.creep_strain += 1.
                elif fault == 'cutback':
                    diagnostics = dict(diagnostics, cutbacks=1)
                return trial, diagnostics

            coordinator = self.coordinator()
            with mock.patch.object(self.bound.state, 'committed', wraps=original_commit) as commits:
                patch = (mock.patch.object(self.bound.newton, 'trial_with_original_policy', side_effect=RuntimeError('unit failed trial'))
                         if fault == 'trial_error' else self.wrap_trial(bad_trial))
                with self.subTest(fault=fault), patch:
                    with self.assertRaises((ValueError, RuntimeError)):
                        coordinator.run()
                self.assertEqual(commits.call_count, 1)
            self.assert_failed_closed(coordinator)

    def test_policy_and_clock_tamper_during_observation_fail(self):
        for fault in ('policy', 'clock'):
            self.configure(n=2, g=1)
            self.entries.clear(); self.exits.clear(); self.factor_calls.clear()

            def observer(row):
                if row['ordinal'] == 0:
                    if fault == 'policy':
                        self.policy.primary_tolerance *= 10
                    else:
                        self.dense[162].time_s += 1.

            coordinator = self.coordinator(observer=observer)
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                coordinator.run()
            self.assert_failed_closed(coordinator)

    def test_initial_policy_and_exact_clock_negatives_precede_factor(self):
        for fault in ('policy', 'short', 'time', 'temperature', 'activation', 'result_set'):
            self.configure(n=2, g=1)
            self.entries.clear(); self.exits.clear(); self.factor_calls.clear()
            if fault == 'policy':
                self.policy.linear_tolerance = 1e-8
            elif fault == 'short':
                self.dense.pop()
            elif fault == 'time':
                self.dense[360].time_s -= 1.
            elif fault == 'temperature':
                self.dense[360].temperature_c = 26.
            elif fault == 'activation':
                self.dense[162].creep_active = False
            else:
                self.dense[160].result_set += 1
            coordinator = self.coordinator()
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                coordinator.run()
            self.assertEqual(coordinator.status, 'FAILED')
            self.assertEqual(self.entries, [])

    def test_oversized_diagnostic_refuses_before_interval_json_and_commit(self):
        def oversized(trial, diagnostics, args):
            return trial, dict(diagnostics, strict_failure='x' * 2001)

        coordinator = self.coordinator()
        dumps = json.dumps
        original_commit = self.bound.state.committed
        with self.wrap_trial(oversized), mock.patch.object(history.json, 'dumps', wraps=dumps) as encoding:
            with mock.patch.object(self.bound.state, 'committed', wraps=original_commit) as commits:
                with self.assertRaisesRegex(ValueError, 'strict-failure diagnostic'):
                    coordinator.run()
                self.assertEqual(encoding.call_count, 1)
                self.assertEqual(commits.call_count, 1)
        self.assert_failed_closed(coordinator)

    def test_retained_output_budget_refuses_before_model_construction(self):
        coordinator = self.coordinator(limits=dict(retained_array_bytes=1, diagnostic_bytes_per_step=1024))
        with mock.patch.object(self.bound.builder, 'make_layer', side_effect=AssertionError('budget reached model')) as build:
            with self.assertRaisesRegex(ValueError, 'output arrays'):
                coordinator.run()
            build.assert_not_called()
        self.assertEqual(self.entries, [])
        self.assertEqual(coordinator.status, 'FAILED')

    def test_callback_guard_and_commit_exceptions_close_factor(self):
        original_commit = self.bound.state.committed
        for fault in ('callback', 'guard', 'commit'):
            self.entries.clear(); self.exits.clear(); self.factor_calls.clear()
            phase = []

            def observer(row):
                phase.append(True)
                if fault == 'callback':
                    raise RuntimeError('unit callback stop')

            def guard():
                if phase and fault == 'guard':
                    raise RuntimeError('unit guard stop')

            calls = []

            def commit(*args):
                calls.append(True)
                if len(calls) > 1:
                    raise RuntimeError('unit commit stop')
                return original_commit(*args)

            coordinator = self.coordinator(observer=observer, check=guard)
            patch = (mock.patch.object(self.bound.state, 'committed', side_effect=commit)
                     if fault == 'commit' else mock.patch.object(self.bound.state, 'committed', wraps=original_commit))
            with self.subTest(fault=fault), patch:
                with self.assertRaisesRegex(RuntimeError, 'unit'):
                    coordinator.run()
            self.assert_failed_closed(coordinator)

    def test_reference_factor_entry_solve_failure_closes_context(self):
        self.factor_fail_at = 2
        coordinator = self.coordinator()
        with self.assertRaisesRegex(RuntimeError, 'physical solve failure'):
            coordinator.run()
        self.assertEqual(coordinator.status, 'FAILED')
        self.assertEqual(coordinator.last_accepted_ordinal, 0)
        self.assertEqual(len(self.entries), 1)
        self.assertEqual(len(self.exits), 1)

    def test_suppressing_factor_context_cannot_publish_final_observer_failure(self):
        self.suppress_failure = True

        def observer(row):
            if row['ordinal'] == 24:
                raise RuntimeError('unit final observer stop')

        coordinator = self.coordinator(observer=observer)
        with self.assertRaises((ValueError, RuntimeError)):
            coordinator.run()
        self.assert_failed_closed(coordinator, accepted=24)

    def test_source_bindings_token_and_no_io_affinity_environment_sideeffects(self):
        with self.assertRaises(ValueError):
            history.bind_sources(dict(self.raw, core=self.raw['core'] + b'\n'))
        forged = SimpleNamespace(**vars(self.bound))
        forged.token = object()
        with self.assertRaises(ValueError):
            history.NativeC1(self.api, forged, assembly=self.assembly, reference=self.reference,
                card=self.card, kernels=self.kernels, policy=self.policy, dense1081=self.dense,
                physical_factor_context=self.physical_context, limits=dict(retained_array_bytes=1024**2,
                diagnostic_bytes_per_step=1024), check=self.check, case='UNIT', native_sha256='0' * 64)
        environment = dict(os.environ)
        coordinator = self.coordinator()
        with mock.patch.object(builtins, 'open', side_effect=AssertionError('unexpected file I/O')):
            with mock.patch.object(Path, 'open', side_effect=AssertionError('unexpected path I/O')):
                with mock.patch.object(os, 'sched_setaffinity', side_effect=AssertionError('affinity changed')):
                    result = coordinator.run()
        self.assertEqual(dict(os.environ), environment)
        for flag in ('material_callbacks_verified_here', 'card_file_provenance_verified_here',
                     'clock_card_provenance_verified_here', 'host_resource_admission',
                     'dense_time_convergence_verified', 'multi_cycle_accuracy_verified',
                     'MAPDL_accuracy_claim', 'bow_accuracy_validated', 'operator_admission'):
            self.assertFalse(result.report[flag])


if __name__ == '__main__':
    unittest.main()
