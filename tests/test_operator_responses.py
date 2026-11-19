"""Tiny response algebra and lifecycle tests; no PARDISO or production jobs.

Main alone executes this suite. Reads are limited to explicit frozen sources.
The fake factor solves only the small in-memory software-fixture matrices.
"""

import ast
import builtins
import ctypes
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from tools import operator_construction as construction
from tools import operator_reference as reference


PROJECT = Path('/app/nanocad/projects/personal/zchen/YAP_STACK')
NIGHT = PROJECT / 'operator_night_20261021'
D = PROJECT / 'operator_repair_20261015/development/trial_forcing_closure_v1/continuous_cu_screen_v1'
PROVIDERS = dict(elastic=NIGHT / 'NATIVE_ELASTIC_LIFT_20261029_v1/native_elastic_lift_v2.py',
    span=NIGHT / 'NATIVE_NONLINEAR_SPAN_20261029_v1/native_nonlinear_span_v3.py',
    projection=NIGHT / 'NATIVE_DIMENSIONAL_CU_20261029_v1/dimensionprojection.py')


class FakeFactor:
    def __init__(self, np, matrix, *, bad_solve=False, close_error=False, on_solve=None):
        self.np, self.matrix = np, matrix.toarray()
        self.bad_solve, self.close_error, self.on_solve = bad_solve, close_error, on_solve
        self.rhs, self.close_calls = [], 0
        self.closed = False

    def solve(self, rhs):
        self.rhs.append(rhs.copy())
        answer = self.np.zeros_like(rhs) if self.bad_solve else self.np.linalg.solve(self.matrix, rhs)
        if self.on_solve:
            self.on_solve()
        return answer

    def close(self):
        self.close_calls += 1
        if self.close_error:
            raise RuntimeError('unit fake close failure')
        self.closed = True


class FakeFactory:
    def __init__(self, np, *, on_create=None, bad_solve=False, close_error=False, on_solve=None):
        self.np, self.on_create = np, on_create
        self.options = dict(bad_solve=bad_solve, close_error=close_error, on_solve=on_solve)
        self.matrices, self.handles = [], []

    def __call__(self, matrix):
        self.matrices.append(matrix.copy())
        handle = FakeFactor(self.np, matrix, **self.options)
        self.handles.append(handle)
        if self.on_create:
            self.on_create()
        return handle


class OperatorResponsesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from scipy import sparse, linalg
        cls.np, cls.sparse = np, sparse
        cls.scipy_solve = staticmethod(linalg.solve)

        def source(path, expected):
            with path.open('rb') as stream:
                raw = stream.read(256 * 1024 + 1)
            if not 0 < len(raw) <= 256 * 1024 or hashlib.sha256(raw).hexdigest() != expected:
                raise AssertionError('frozen small source differs: ' + str(path))
            return raw

        cls.raw = source(D / 'native_monolithic_reference.py', reference.ELASTIC_REFERENCE_SHA)
        cls.provider_raw = {name: source(path, construction.SOURCE_PINS[name])
                            for name, path in PROVIDERS.items()}

    @staticmethod
    def freeze(value):
        value.setflags(write=False)
        return value

    def freeze_csr(self, value):
        value.sum_duplicates()
        value.sort_indices()
        for array in (value.data, value.indices, value.indptr):
            self.freeze(array)
        return value

    def setUp(self):
        np = self.np
        # A private linalg namespace makes exception injection local to this test.
        self.api = SimpleNamespace(np=np, sparse=self.sparse,
                                   linalg=SimpleNamespace(solve=self.scipy_solve))
        self.bound = reference.bind_responses(self.api, self.raw)
        self.check = mock.Mock()
        self.factory = FakeFactory(np)
        micro, physical = 9, 15
        scales = np.logspace(-2, 2, micro)
        h = 2 * np.eye(micro) + .1 * (np.eye(micro, k=1) + np.eye(micro, k=-1))
        k = scales[:, None] * h * scales[None, :]
        mix = .03 * (np.eye(micro, 6) + np.arange(micro * 6).reshape(micro, 6) / 540.)
        c = scales[:, None] * mix
        schur = np.diag(np.arange(2., 8.)) + .01 * np.ones((6, 6))
        kaa = schur + c.T @ np.linalg.solve(k, c)
        self.assembly = dict(kff=self.freeze_csr(self.sparse.csr_matrix(k)),
            kaa=self.freeze(kaa), c=self.freeze(c),
            fm=self.freeze(np.linspace(.2, .7, 6)),
            ff=self.freeze(scales * np.linspace(.2, .8, micro)))
        p = self.sparse.csr_matrix((np.ones(micro), (np.arange(micro), np.arange(micro))),
                                   shape=(physical, micro))
        a = np.vstack((.01 * np.arange(micro * 6).reshape(micro, 6) / 54., np.eye(6)))
        xyz = np.array([[0., 0., 0.], [.001, .01, .001], [.002, .02, .002],
                        [.003, .03, .003], [.004, .0485, .004]])
        self.basis = dict(P=self.freeze_csr(p), macro_fields=self.freeze(a), coordinates_mm=self.freeze(xyz))

    def responses(self, *, basis=None, assembly=None, factory=None, check=None, bound=None):
        return reference.elastic_operands(self.api, self.bound if bound is None else bound,
            basis=self.basis if basis is None else basis,
            assembly=self.assembly if assembly is None else assembly,
            factor_factory=self.factory if factory is None else factory,
            check=self.check if check is None else check)

    def input_bytes(self):
        values = [self.basis['macro_fields'], self.basis['coordinates_mm']]
        values += [self.assembly[name] for name in ('kaa', 'c', 'fm', 'ff')]
        for matrix in (self.basis['P'], self.assembly['kff']):
            values.extend((matrix.data, matrix.indices, matrix.indptr))
        return [value.tobytes() for value in values]

    def test_exact19_schema_readonly_borrowing_and_unchanged_inputs(self):
        before = self.input_bytes()
        result = self.responses()
        n, physical = 9, 15
        shapes = dict(kaa=(6, 6), c=(n, 6), fm=(6,), ff=(n,), macro_fields=(physical, 6),
            mechanical_q=(6, 6), mechanical_u=(physical, 6), mechanical_reactions=(6, 6),
            schur=(6, 6), thermal_q=(6,), thermal_u=(physical,),
            P_data=self.basis['P'].data.shape, P_indices=self.basis['P'].indices.shape,
            P_indptr=(physical + 1,), P_shape=(2,),
            kff_data=self.assembly['kff'].data.shape, kff_indices=self.assembly['kff'].indices.shape,
            kff_indptr=(n + 1,), kff_shape=(2,))
        self.assertEqual(set(result.arrays), set(shapes))
        self.assertEqual(set(result.arrays), reference.ELASTIC_KEYS)
        for name, shape in shapes.items():
            with self.subTest(name=name):
                array = result.arrays[name]
                self.assertEqual(array.shape, shape)
                self.assertFalse(array.flags.writeable)
                self.assertTrue(array.flags.c_contiguous or array.flags.f_contiguous)
                self.assertTrue(self.np.isfinite(array).all())
                self.assertEqual(array.dtype.kind, 'i' if name.endswith(('_indices', '_indptr', '_shape')) else 'f')
        for name in ('kaa', 'c', 'fm', 'ff'):
            self.assertIs(result.arrays[name], self.assembly[name])
        self.assertIs(result.arrays['macro_fields'], self.basis['macro_fields'])
        for prefix, matrix in (('P', self.basis['P']), ('kff', self.assembly['kff'])):
            for name in ('data', 'indices', 'indptr'):
                self.assertIs(result.arrays[prefix + '_' + name], getattr(matrix, name))
        self.assertEqual(self.input_bytes(), before)

    def test_scaled_factory_rhs_and_seven_solve_lifecycle(self):
        result = self.responses()
        np = self.np
        d = 1 / np.sqrt(self.assembly['kff'].diagonal())
        expected = d[:, None] * self.assembly['kff'].toarray() * d[None, :]
        self.assertEqual(len(self.factory.matrices), 1)
        scaled = self.factory.matrices[0]
        self.assertTrue(scaled.has_canonical_format)
        np.testing.assert_allclose(scaled.toarray(), expected, rtol=2e-15, atol=0.)
        self.assertGreater(self.assembly['kff'].diagonal().max() / self.assembly['kff'].diagonal().min(), 1e7)
        handle = self.factory.handles[0]
        self.assertEqual(len(handle.rhs), 7)
        for j in range(6):
            np.testing.assert_array_equal(handle.rhs[j], d * self.assembly['c'][:, j])
        np.testing.assert_array_equal(handle.rhs[-1], d * self.assembly['ff'])
        self.assertEqual(handle.close_calls, 1)
        self.assertTrue(handle.closed)
        for name, value in (('factory_calls', 1), ('body_solve_calls', 7), ('close_calls', 1)):
            self.assertEqual(result.report[name], value)
        self.assertTrue(result.report['close_returned_successfully'])
        self.assertEqual(len(result.report['relative_body_residuals']), 7)

    def test_coupled_thermal_reactions_and_full_field_match_independent_oracles(self):
        np = self.np
        arrays = self.responses().arrays
        k, c, kaa = self.assembly['kff'].toarray(), self.assembly['c'], self.assembly['kaa']
        coupled = np.block([[kaa, c.T], [c, k]])
        load = np.r_[self.assembly['fm'], self.assembly['ff']]
        thermal = np.linalg.solve(coupled, load)
        np.testing.assert_allclose(arrays['thermal_q'], thermal[:6], rtol=2e-12, atol=1e-14)
        expected_u = self.basis['macro_fields'] @ thermal[:6] + self.basis['P'] @ thermal[6:]
        np.testing.assert_allclose(arrays['thermal_u'], expected_u, rtol=2e-12, atol=1e-14)
        z = self.basis['P'].T @ (arrays['thermal_u'] - self.basis['macro_fields'] @ arrays['thermal_q'])
        np.testing.assert_allclose(c @ arrays['thermal_q'] + k @ z, self.assembly['ff'], rtol=2e-12, atol=1e-13)
        np.testing.assert_allclose(kaa @ arrays['thermal_q'] + c.T @ z, self.assembly['fm'], rtol=2e-12, atol=1e-13)
        y = np.linalg.solve(k, c)
        schur = kaa - c.T @ y
        np.testing.assert_allclose(arrays['schur'], schur, rtol=2e-12, atol=1e-14)
        qm = arrays['mechanical_q']
        np.testing.assert_allclose(arrays['mechanical_reactions'], schur @ qm, rtol=2e-12, atol=1e-18)
        displacement = self.basis['macro_fields'] @ qm - self.basis['P'] @ (y @ qm)
        np.testing.assert_allclose(arrays['mechanical_u'], displacement, rtol=2e-12, atol=1e-18)

    def test_mechanical_amplitudes_use_actual_height_not_unit_columns(self):
        for factor in (1., 2.):
            basis = dict(self.basis)
            xyz = basis['coordinates_mm'].copy()
            xyz[:, 1] *= factor
            basis['coordinates_mm'] = self.freeze(xyz)
            result = self.responses(basis=basis, factory=FakeFactory(self.np))
            h = float(self.np.ptp(xyz[:, 1]))
            expected = self.np.diag(self.np.r_[self.np.full(3, 1e-6), self.np.full(3, 1e-6 / h)])
            self.np.testing.assert_array_equal(result.arrays['mechanical_q'], expected)
            self.assertEqual(result.report['physical_height_mm'], h)
            self.assertFalse(result.report['mechanical_columns_are_unit_amplitude'])

    def test_original_residual_gate_rejects_bad_solve_and_closes(self):
        factory = FakeFactory(self.np, bad_solve=True)
        with self.assertRaisesRegex(ValueError, 'physical sparse solve certificate'):
            self.responses(factory=factory)
        self.assertEqual(len(factory.handles[0].rhs), 1)
        self.assertEqual(factory.handles[0].close_calls, 1)
        self.assertTrue(factory.handles[0].closed)

    def test_postfactory_guard_closes_before_any_solve(self):
        phase = []
        factory = FakeFactory(self.np, on_create=lambda: phase.append('acquired'))

        def guard():
            if phase:
                raise RuntimeError('unit guard after acquire')

        with self.assertRaisesRegex(RuntimeError, 'after acquire'):
            self.responses(factory=factory, check=guard)
        handle = factory.handles[0]
        self.assertEqual(handle.rhs, [])
        self.assertEqual(handle.close_calls, 1)
        self.assertTrue(handle.closed)

    def test_acquire_to_inner_try_gap_closes_once_in_outer_finally(self):
        for close_error in (False, True):
            factory = FakeFactory(self.np, close_error=close_error)

            def interrupted_before_inner_try(basis, assembly, factor_factory, check, record_body_residual):
                d = 1 / self.np.sqrt(assembly['kff'].diagonal())
                scaled = (self.sparse.diags(d) @ assembly['kff'] @ self.sparse.diags(d)).tocsr()
                scaled.sum_duplicates()
                scaled.sort_indices()
                factor_factory(scaled)
                raise RuntimeError('unit gap after acquisition before inner try')

            with self.subTest(close_error=close_error):
                with mock.patch.object(self.bound, 'responses', side_effect=interrupted_before_inner_try):
                    with self.assertRaises(RuntimeError):
                        self.responses(factory=factory)
                self.assertEqual(len(factory.handles), 1)
                self.assertEqual(factory.handles[0].rhs, [])
                self.assertEqual(factory.handles[0].close_calls, 1)
                self.assertEqual(factory.handles[0].closed, not close_error)

    def test_guard_after_body_solve_closes(self):
        phase = []
        factory = FakeFactory(self.np, on_solve=lambda: phase.append('solved'))

        def guard():
            if phase:
                raise RuntimeError('unit guard after solve')

        with self.assertRaisesRegex(RuntimeError, 'after solve'):
            self.responses(factory=factory, check=guard)
        self.assertEqual(len(factory.handles[0].rhs), 1)
        self.assertEqual(factory.handles[0].close_calls, 1)
        self.assertTrue(factory.handles[0].closed)

    def test_dense_thermal_exception_closes_after_seven_body_solves(self):
        with mock.patch.object(self.api.linalg, 'solve', side_effect=RuntimeError('unit dense thermal exception')):
            with self.assertRaisesRegex(RuntimeError, 'dense thermal exception'):
                self.responses()
        self.assertEqual(len(self.factory.handles[0].rhs), 7)
        self.assertEqual(self.factory.handles[0].close_calls, 1)
        self.assertTrue(self.factory.handles[0].closed)

    def test_post_dense_thermal_guard_closes(self):
        phase = []

        def dense(*args, **kwargs):
            result = self.scipy_solve(*args, **kwargs)
            phase.append('thermal')
            return result

        def guard():
            if phase:
                raise RuntimeError('unit post-thermal guard')

        with mock.patch.object(self.api.linalg, 'solve', side_effect=dense):
            with self.assertRaisesRegex(RuntimeError, 'post-thermal guard'):
                self.responses(check=guard)
        self.assertEqual(len(self.factory.handles[0].rhs), 7)
        self.assertTrue(self.factory.handles[0].closed)

    def test_factory_failure_partial_cleanup_is_owners_responsibility(self):
        partial = []

        def factory(matrix):
            handle = FakeFactor(self.np, matrix)
            partial.append(handle)
            handle.close()
            raise RuntimeError('unit owner cleaned partial factory')

        with self.assertRaisesRegex(RuntimeError, 'partial factory'):
            self.responses(factory=factory)
        self.assertEqual(partial[0].close_calls, 1)
        self.assertTrue(partial[0].closed)
        self.assertEqual(partial[0].rhs, [])

    def test_close_exception_prevents_success_result(self):
        factory = FakeFactory(self.np, close_error=True)
        with self.assertRaisesRegex(RuntimeError, 'close failure'):
            self.responses(factory=factory)
        self.assertEqual(len(factory.handles[0].rhs), 7)
        self.assertEqual(factory.handles[0].close_calls, 1)
        self.assertFalse(factory.handles[0].closed)

    def test_writable_input_storage_is_rejected_without_factor(self):
        for name in ('c', 'kff', 'P', 'macro_fields'):
            basis, assembly = dict(self.basis), dict(self.assembly)
            if name in ('P', 'kff'):
                container = basis if name == 'P' else assembly
                container[name] = container[name].copy()
                container[name].data.setflags(write=True)
            elif name == 'c':
                assembly[name] = assembly[name].copy()
            else:
                basis[name] = basis[name].copy()
            factory = mock.Mock(side_effect=AssertionError('invalid input reached factor'))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'immutable'):
                self.responses(basis=basis, assembly=assembly, factory=factory)
            factory.assert_not_called()

    def test_nonfinite_zero_and_overflow_height_refuse_before_factor(self):
        for fault in ('zero', 'nan', 'infinity', 'overflow'):
            basis = dict(self.basis)
            xyz = basis['coordinates_mm'].copy()
            if fault == 'zero':
                xyz[:, 1] = 0.
            elif fault == 'nan':
                xyz[0, 1] = self.np.nan
            elif fault == 'infinity':
                xyz[-1, 1] = self.np.inf
            else:
                xyz[0, 1] = -self.np.finfo('float64').max
                xyz[-1, 1] = self.np.finfo('float64').max
            basis['coordinates_mm'] = self.freeze(xyz)
            factory = mock.Mock(side_effect=AssertionError('invalid height reached factor'))
            with self.subTest(fault=fault), self.np.errstate(over='ignore', invalid='ignore'):
                with self.assertRaises(ValueError):
                    self.responses(basis=basis, factory=factory)
            factory.assert_not_called()

    def test_binding_source_pin_token_and_inverse_ast_proof(self):
        with self.assertRaises(ValueError):
            reference.bind_responses(self.api, self.raw + b'\n')
        bad = SimpleNamespace(**vars(self.bound))
        bad.token = object()
        with self.assertRaisesRegex(ValueError, 'authenticated response binding'):
            self.responses(bound=bad)
        tree = ast.parse(self.raw)
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        anchor = ast.dump(ast.parse("kff,c,kaa = (assembly[name] for name in ('kff','c','kaa'))").body[0])
        start = next(index for index, node in enumerate(run.body) if ast.dump(node) == anchor)
        block = ast.dump(ast.Module(body=run.body[start:start + 7], type_ignores=[]))
        self.assertEqual(self.bound.original_block_AST_sha256, hashlib.sha256(block.encode('utf-8')).hexdigest())
        self.assertTrue(self.bound.inverse_AST_exact)
        self.assertEqual(self.bound.guard_insertions, 7)
        self.assertEqual(self.bound.source_sha256, reference.ELASTIC_REFERENCE_SHA)
        self.assertNotIn('backend', self.bound.responses.__globals__)
        self.assertNotIn('library', self.bound.responses.__globals__)

    def test_no_io_library_load_affinity_or_environment_changes(self):
        before = dict(os.environ)
        with mock.patch.object(builtins, 'open', side_effect=AssertionError('unexpected file I/O')):
            with mock.patch.object(Path, 'open', side_effect=AssertionError('unexpected path I/O')):
                with mock.patch.object(ctypes, 'CDLL', side_effect=AssertionError('unexpected vendor library load')):
                    with mock.patch.object(os, 'sched_setaffinity', side_effect=AssertionError('affinity changed')):
                        bound = reference.bind_responses(self.api, self.raw)
                        result = self.responses(bound=bound)
        self.assertEqual(dict(os.environ), before)
        for key in ('library_loaded_here', 'vendor_timing_claim', 'resource_admission', 'history_executed',
                    'bow_accuracy_validated', 'operator_admission', 'OS_resource_release_verified_here'):
            self.assertFalse(result.report[key])
        self.assertEqual(result.report['Cu_updates'], 0)

    def test_returned_operands_feed_actual_frozen_elastic_provider(self):
        result = self.responses()
        algebra = construction.bind_sources(self.provider_raw)
        coordinator = construction.Construction(self.np, self.sparse, algebra,
            case='SOFTWARE_COUPLED_SPD_NOT_PRODUCTION', native_sha256='0' * 64, check=self.check)
        elastic = coordinator.elastic(result.arrays)
        self.assertEqual(coordinator.state, 'ELASTIC_CLOSED')
        self.assertEqual(elastic['Qbody7'].shape[0], 9)
        self.assertIn(elastic['Qbody7'].shape[1], (6, 7))
        self.assertFalse(coordinator.evidence()['operator_admission'])
        self.assertEqual(coordinator.evidence()['Cu_updates'], 0)

    def test_actual_12hex_assembly_responses_and_elastic_provider_integration(self):
        # Module alias stays local: unittest must not rediscover its TestCase here.
        import test_operator_reference as fixture_module
        fixture_class = fixture_module.OperatorReferenceTests
        fixture_class.setUpClass()
        fixture = fixture_class('test_actual_grouped_assembly_matches_independent_element_oracle')
        fixture.setUp()
        assembled = fixture.assemble()
        factory = FakeFactory(self.np)
        result = reference.elastic_operands(self.api, self.bound, basis=fixture.target.basis,
            assembly=assembled.assembly, factor_factory=factory, check=self.check)
        algebra = construction.bind_sources(self.provider_raw)
        coordinator = construction.Construction(self.np, self.sparse, algebra,
            case='SOFTWARE_12HEX_STATIC_CHAIN_NOT_PRODUCTION', native_sha256='0' * 64, check=self.check)
        elastic = coordinator.elastic(result.arrays)
        self.assertEqual(assembled.proof['regional_subdivisions'], [1, 1, 1])
        self.assertEqual(assembled.proof['Cu_GP'], 24)
        self.assertEqual(coordinator.state, 'ELASTIC_CLOSED')
        self.assertEqual(elastic['Qbody7'].shape[0], fixture.target.basis['P'].shape[1])
        self.assertIn(elastic['Qbody7'].shape[1], (6, 7))
        self.assertEqual(factory.handles[0].close_calls, 1)
        self.assertTrue(factory.handles[0].closed)
        self.assertEqual(len(factory.handles[0].rhs), 7)
        self.assertEqual(coordinator.evidence()['Cu_updates'], 0)


if __name__ == '__main__':
    unittest.main()
