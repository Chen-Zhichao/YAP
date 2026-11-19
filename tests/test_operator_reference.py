"""Actual grouped FE algebra versus a 12-element oracle; software fixtures only.

No factor, native payload, production constructor, history, solver or launcher.
The only inputs read by Main's test execution are the named small source files.
"""

import ast
import builtins
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import sys
from types import CodeType, SimpleNamespace
import unittest
from unittest import mock

from tools import operator_geometry as height
from tools import operator_reference as reference


PROJECT = Path('/app/nanocad/projects/personal/zchen/YAP_STACK')
D = PROJECT / 'operator_repair_20261015/development/trial_forcing_closure_v1/continuous_cu_screen_v1'
E = PROJECT / 'operator_repair_20261015/source'
SOURCE_PATHS = dict(quadrature=D / 'common_cell_quadrature.py', native=D / 'native_monolithic_assembly.py',
                    bbar=E / 'exp12_solid185_bbar.py', material=E / 'exp12_cu_material_point.py',
                    fields=D / 'prepared_geometry_extension.py')
OFFSETS = ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1),
           (0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1))


class OperatorReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from scipy import sparse
        cls.np, cls.sparse = np, sparse
        cls.api = SimpleNamespace(np=np, sparse=sparse)

        def read_source(path, expected):
            with path.open('rb') as stream:
                raw = stream.read(256 * 1024 + 1)
            if not 0 < len(raw) <= 256 * 1024 or hashlib.sha256(raw).hexdigest() != expected:
                raise AssertionError('small frozen source differs: ' + str(path))
            return raw

        cls.raw = {role: read_source(path, reference.SOURCE_PINS[role])
                   for role, path in SOURCE_PATHS.items()}
        geometry_raw = read_source(D / 'native_periodic_geometry.py', height.GEOMETRY_SHA)
        cls.geometry = height.bind_sources(geometry_raw, cls.raw['fields'])

    def setUp(self):
        self.bound = reference.bind_sources(self.api, self.raw)
        self.check = mock.Mock()
        self.limits = dict(chunk_elements=2, max_shape_groups=4096,
                           cache_cap_bytes=8 * 1024**2, output_payload_cap_bytes=8 * 1024**2)
        self.seed, self.seed_basis, self.contract = self.fixture()
        self.target = self.make_target(20, 'inner')

    @staticmethod
    def node(i, j, k):
        return 1 + (i * 4 + j) * 3 + k

    def fixture(self):
        np = self.np
        nodes = {self.node(i, j, k): (.003 * i, y, .003 * k)
                 for i in range(3) for j, y in enumerate((0., .0015, .0115, .0485)) for k in range(3)}
        nodes.update({node: (0., 0., 0.) for node in range(37, 47)})
        controls = ((37, 'UX'), (39, 'UZ'), (45, 'UY'), (38, 'UX'), (40, 'UZ'), (46, 'UY'))
        constraints = {(1, dof): 0. for dof in ('UX', 'UY', 'UZ')}
        constraints.update({(node, dof): 0. for node in range(37, 47)
                            for dof in ('UX', 'UY', 'UZ') if (node, dof) not in controls})
        elements = {}
        for j in range(3):
            for i in range(2):
                for k in range(2):
                    elements[len(elements) + 1] = (
                        tuple(self.node(i + di, j + dj, k + dk) for di, dj, dk in OFFSETS),
                        4 if (i, k) == (0, 0) else j + 1)
        fields = self.geometry.extension.analytic_fields(np, np.asarray([nodes[n] for n in range(1, 37)]))
        equations = {}
        for i in range(3):
            for j in range(4):
                for k in range(3):
                    if i != 2 and k != 2:
                        continue
                    slave, master = self.node(i, j, k), self.node(0 if i == 2 else i, j, 0 if k == 2 else k)
                    for component, dof in enumerate(('UX', 'UY', 'UZ')):
                        jump = fields[slave - 1, component] - fields[master - 1, component]
                        equations[len(equations) + 1] = ((slave, dof, 1.), (master, dof, -1.)) + tuple(
                            (node, label, -float(value)) for (node, label), value in zip(controls, jump) if value != 0.)
        # Distinct synthetic laws exercise material dispatch, not production cards.
        laws = {1: (70000., .20, 1e-6), 2: (11000., .30, 3e-6),
                3: (130000., .28, 2.6e-6), 4: (110000., .34, 17e-6)}
        elastic = {(label, material, 1): value for material, values in laws.items()
                   for label, value in zip(('EX', 'PRXY', 'ALPX'), values)}
        deck = self.geometry.geometry.NativeDeck(nodes=nodes, elements=elements, equations=equations,
            constraints=constraints, elastic=elastic, element_types={1: 'SOLID185'},
            element_options={(1, 2): 0}, first_solve_line=1)
        basis = self.geometry.geometry.six_macro_basis(self.api, deck)
        contract = dict(case='SOFTWARE_12HEX_NOT_PRODUCTION', mesh_recipe='UNIT_2x3x2',
            pitch_um=2, diameter_pitch_ratio=.5, source_si_um=37., cell_xz_um=[6, 6],
            physical_nodes=36, hex_elements=12, deck_nodes=46, equations=60,
            constraints=len(constraints), body_coordinates=45, Cu_GP=24,
            region_Cu_GP=dict(bi=8, beol=8, si=8),
            seed_report=dict(path=str(PROJECT / 'runtime_checks/UNIT_METADATA_ONLY/REPORT.json'),
                             bytes=1, sha256='0' * 64))
        return deck, basis, contract

    def make_target(self, total, variant):
        return height.prepare_target(self.api, self.geometry, seed_deck=self.seed, seed_basis=self.seed_basis,
            request=height.request(2, total, variant), seed_contract=self.contract, check=self.check)

    def assemble(self, target=None, limits=None, check=None, bound=None):
        return reference.assemble_target(self.api, self.bound if bound is None else bound,
            target=self.target if target is None else target, limits=self.limits if limits is None else limits,
            check=self.check if check is None else check)

    def independent_element_oracle(self, target):
        """Direct dense scatter: no grouped plan/cache/assembly/projection helpers."""
        np = self.np
        basis = target.basis
        xyz, cells, mids = (basis[name] for name in ('coordinates_mm', 'cells', 'materials'))
        ndof = 3 * len(xyz)
        stiffness, force = np.zeros((ndof, ndof)), np.zeros(ndof)
        raw_rows, weights, positions, element_ids, gauss_ids = [], [], [], [], []
        laws = {material: tuple(target.deck.elastic[(label, material, 1)]
                                for label in ('EX', 'PRXY', 'ALPX')) for material in range(1, 5)}
        for index, (cell, material) in enumerate(zip(cells, mids)):
            dofs = (3 * cell[:, None] + np.arange(3)).ravel()
            elastic, poisson, alpha = laws[int(material)]
            lam = elastic * poisson / ((1 + poisson) * (1 - 2 * poisson))
            mu = elastic / (2 * (1 + poisson))
            c = np.zeros((6, 6))
            c[:3, :3] = lam
            c[np.arange(3), np.arange(3)] += 2 * mu
            c[3:, 3:] = mu * np.eye(3)
            thermal_stress = c @ np.array([alpha, alpha, alpha, 0., 0., 0.])
            ke, fe = np.zeros((24, 24)), np.zeros(24)
            points = self.bound.primitives.bbar.bbar_points(xyz[cell])
            self.assertEqual(len(points), 8)
            for gauss, point in enumerate(points):
                b, w = point.b_matrix, point.integration_volume_mm3
                ke += w * (b.T @ c @ b)
                fe += w * (b.T @ thermal_stress)
                if material == 4:
                    row = np.zeros((6, ndof))
                    row[:, dofs] = b
                    raw_rows.append(row)
                    weights.append(w)
                    positions.append(point.position_mm)
                    element_ids.append(basis['element_ids'][index])
                    gauss_ids.append(gauss)
            stiffness[np.ix_(dofs, dofs)] += ke
            force[dofs] += fe
        b = np.vstack(raw_rows)
        a, p = basis['macro_fields'], basis['P'].toarray()
        return dict(k=stiffness, f26=force, B_global=b, B_macro=b @ a, B_free=b @ p,
            kaa=a.T @ stiffness @ a, c=p.T @ stiffness @ a, kff=p.T @ stiffness @ p,
            fm=a.T @ force, ff=p.T @ force, weights=np.asarray(weights),
            Cu_positions_mm=np.asarray(positions), Cu_native_element_ids=np.asarray(element_ids),
            Cu_native_gauss_indices=np.asarray(gauss_ids))

    def assert_norm_close(self, observed, expected):
        np = self.np
        if self.sparse.issparse(observed):
            observed = observed.toarray()
        self.assertEqual(observed.shape, expected.shape)
        error = float(np.linalg.norm(observed - expected))
        scale = float(np.linalg.norm(expected))
        self.assertLessEqual(error, 5e-11 * scale)

    def test_actual_grouped_assembly_matches_independent_element_oracle(self):
        for total, variant in ((20, 'inner'), (20, 'outer'), (50, 'inner'), (100, 'outer')):
            with self.subTest(total=total, variant=variant):
                target = self.make_target(total, variant)
                result = self.assemble(target)
                expected = self.independent_element_oracle(target)
                for name, value in expected.items():
                    with self.subTest(array=name):
                        if name in ('Cu_native_element_ids', 'Cu_native_gauss_indices'):
                            self.np.testing.assert_array_equal(result.assembly[name], value)
                        else:
                            self.assert_norm_close(result.assembly[name], value)
                self.assertEqual(result.proof['regional_subdivisions'], [1, 1, 1])
                self.assertGreater(result.proof['physical_shape_cache']['physical_shape_groups'], 1)
                self.assertEqual(result.proof['Cu_GP'], 24)
                self.assertEqual(result.proof['GP_order'], 'original native EN order, then pinned 2x2x2 Gauss order')

    def test_target_subdivisions_are_measured_not_legacy_6_18_20(self):
        for total, variant in ((20, 'inner'), (100, 'outer')):
            target = self.make_target(total, variant)
            self.assertEqual(reference._subdivisions(self.np, target.basis), (1, 1, 1))
            result = self.assemble(target)
            for role in ('bi', 'beol', 'si'):
                self.assertEqual(result.proof['regions'][role]['thickness_intervals'], 1)
                self.assertEqual(result.proof['regions'][role]['Cu_GP'], 8)

    def test_actual_cache_certifies_all_engineering_shears_rigid_modes_and_volume(self):
        result = self.assemble()
        proof = result.proof['physical_shape_cache']
        self.assertLessEqual(proof['maximum_affine_strain_error'], 2e-11)
        self.assertLessEqual(proof['maximum_relative_rigid_error'], 2e-12)
        self.assertLessEqual(proof['maximum_relative_shape_volume_error'], 2e-12)
        for shape in ((.003, .0015, .003), (.003, .007, .003)):
            xyz = self.np.asarray(OFFSETS, dtype=float) * self.np.asarray(shape)
            x, y, z = xyz.T
            fields = self.np.zeros((8, 3, 6))
            fields[:, 0, 0], fields[:, 1, 1], fields[:, 2, 2] = x, y, z
            fields[:, 0, 3] = y
            fields[:, 1, 4] = z
            fields[:, 0, 5] = z
            for point in self.bound.primitives.bbar.bbar_points(xyz):
                self.np.testing.assert_allclose(point.b_matrix @ fields.reshape(24, 6),
                                               self.np.eye(6), rtol=0., atol=2e-11)

    def test_proof_custody_and_nonadmission_are_explicit(self):
        proof = self.assemble().proof
        self.assertEqual(proof['source_sha256'], dict(reference.SOURCE_PINS))
        self.assertTrue(proof['source_bytes_authenticated'])
        self.assertTrue(proof['file_modes_not_read'])
        self.assertNotIn('dependency_modes', proof)
        self.assertNotIn('dependencies_compiled_from_identity_checked_SHA_pinned_bytes', proof)
        self.assertFalse(proof['file_identity_verified_here'])
        self.assertFalse(proof['geometry_provenance_verified_here'])
        self.assertFalse(proof['numerical_definitions_rewritten'])
        self.assertFalse(proof['coordinate_rounding_enabled'])
        self.assertFalse(proof['history_executed'])
        self.assertFalse(proof['operator_admission'])
        self.assertFalse(proof['resource_admission_performed_here'])
        self.assertEqual(proof['FE_factorizations'], 0)
        self.assertEqual(proof['Cu_updates'], 0)
        self.assertIsNone(proof['coordinate_quantum_mm'])
        self.assertEqual(proof['stiffness_symmetrization'], 'unchanged original physical_stiffness')

    def test_missing_interface_and_census_fail_before_assembler(self):
        for fault in ('interface', 'census'):
            basis, proof = dict(self.target.basis), deepcopy(self.target.proof)
            if fault == 'interface':
                xyz = basis['coordinates_mm'].copy()
                xyz[xyz[:, 1] == .0015, 1] = .0016
                xyz.setflags(write=False)
                basis['coordinates_mm'] = xyz
            else:
                proof['Cu_GP'] += 8
            target = height.GeometryTarget(self.target.deck, basis, proof)
            with self.subTest(fault=fault), mock.patch.object(self.bound.quadrature, 'assemble_native') as assembly:
                with self.assertRaises(ValueError):
                    self.assemble(target)
                assembly.assert_not_called()

    def test_wrong_material_p_quotient_and_orientation_fail_before_matrix_build(self):
        for fault in ('material', 'P_support', 'P_quotient', 'orientation'):
            basis = dict(self.target.basis)
            if fault == 'material':
                mids = basis['materials'].copy()
                mids[1] = 3
                basis['materials'] = mids
            elif fault.startswith('P_'):
                p = basis['P'].copy()
                if fault == 'P_support':
                    p.data[0] = 2.
                else:
                    p.indices[[0, 1]] = p.indices[[1, 0]]
                basis['P'] = p
            else:
                basis['cells'] = basis['cells'][:, [0, 1, 5, 4, 3, 2, 6, 7]].copy()
            target = height.GeometryTarget(self.target.deck, basis, self.target.proof)
            with self.subTest(fault=fault), mock.patch.object(self.bound.quadrature, 'physical_stiffness') as matrix:
                with self.assertRaises(ValueError):
                    self.assemble(target)
                matrix.assert_not_called()

    def test_wrong_regional_expected_gp_is_not_silently_adopted(self):
        proof = deepcopy(self.target.proof)
        proof['target_regions']['bi']['Cu_GP'] += 8
        target = height.GeometryTarget(self.target.deck, self.target.basis, proof)
        with self.assertRaisesRegex(ValueError, 'regional ledger'):
            self.assemble(target)

    def test_shape_cache_and_output_budgets_refuse_before_stiffness(self):
        for change in (dict(max_shape_groups=1), dict(cache_cap_bytes=1),
                       dict(output_payload_cap_bytes=1), dict(chunk_elements=0)):
            limits = dict(self.limits, **change)
            with self.subTest(change=change), mock.patch.object(self.bound.quadrature, 'physical_stiffness') as matrix:
                with self.assertRaises(ValueError):
                    self.assemble(limits=limits)
                matrix.assert_not_called()
        with mock.patch.object(self.bound.quadrature, 'quadrature_cache') as cache:
            with self.assertRaises(ValueError):
                self.assemble(limits=dict(self.limits, output_payload_cap_bytes=1))
            cache.assert_not_called()

    def test_explicit_limit_keys_required(self):
        for limits in ({}, dict(self.limits, unexpected=1)):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                self.assemble(limits=limits)

    def test_bound_token_and_api_identity_are_required(self):
        bad = SimpleNamespace(**vars(self.bound))
        bad.token = object()
        with self.assertRaisesRegex(ValueError, 'authenticated FE binding'):
            self.assemble(bound=bad)
        api = SimpleNamespace(np=SimpleNamespace(), sparse=self.sparse)
        with self.assertRaisesRegex(ValueError, 'identical numerical API'):
            reference.assemble_target(api, self.bound, target=self.target, limits=self.limits, check=self.check)

    def test_guard_propagates_at_entry_and_during_actual_quadrature(self):
        with mock.patch.object(self.bound.quadrature, 'assemble_native') as assembly:
            with self.assertRaisesRegex(RuntimeError, 'entry guard'):
                self.assemble(check=mock.Mock(side_effect=RuntimeError('entry guard')))
            assembly.assert_not_called()
        entered = []
        original = self.bound.quadrature.quadrature_cache

        def cache(*args, **kwargs):
            entered.append(True)
            return original(*args, **kwargs)

        def guard():
            if entered:
                raise RuntimeError('actual quadrature checkpoint')

        with mock.patch.object(self.bound.quadrature, 'quadrature_cache', side_effect=cache):
            with mock.patch.object(self.bound.quadrature, 'physical_stiffness') as matrix:
                with self.assertRaisesRegex(RuntimeError, 'actual quadrature checkpoint'):
                    self.assemble(check=guard)
                matrix.assert_not_called()
        self.assertTrue(entered)

    def test_source_pin_closure_and_bounds_reject_before_selection(self):
        mutations = [dict(self.raw, native=self.raw['native'] + b'\n'),
                     dict(self.raw, native=self.raw['native'].decode('ascii')),
                     dict(self.raw, native=b'x' * (256 * 1024 + 1)),
                     {key: value for key, value in self.raw.items() if key != 'material'},
                     dict(self.raw, invented=b'')]
        for raw in mutations:
            with self.subTest(keys=tuple(raw)), mock.patch.object(reference, '_definitions') as definitions:
                with self.assertRaises(ValueError):
                    reference.bind_sources(self.api, raw)
                definitions.assert_not_called()

    def test_selected_assembly_function_bodies_equal_original_bytecode(self):
        def fingerprint(code):
            return (code.co_code, code.co_names, code.co_varnames,
                    tuple(fingerprint(value) if isinstance(value, CodeType) else value for value in code.co_consts))

        tree = ast.parse(self.raw['quadrature'])
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in reference.QUADRATURE_FUNCTIONS:
                compiled = compile(ast.Module(body=[node], type_ignores=[]), '<unit-original>', 'exec', dont_inherit=True)
                expected = next(value for value in compiled.co_consts if isinstance(value, CodeType) and value.co_name == node.name)
                observed = getattr(self.bound.quadrature, node.name).__code__
                self.assertEqual(fingerprint(observed), fingerprint(expected), node.name)
        self.assertEqual(self.bound.quadrature.PINS, {})
        for forbidden in ('load_primitives_from_disk', 'elastic', 'main', 'frozen_module'):
            self.assertFalse(hasattr(self.bound.quadrature, forbidden))

    def test_bind_and_actual_assembly_have_no_io_affinity_or_environment_sideeffects(self):
        environment = dict(os.environ)
        names = ('_operator_reference_quadrature', '_operator_reference_native',
                 '_operator_reference_bbar', '_operator_reference_material', '_operator_reference_fields')
        modules = {name: sys.modules.get(name) for name in names}
        with mock.patch.object(builtins, 'open', side_effect=AssertionError('unexpected file I/O')):
            with mock.patch.object(Path, 'open', side_effect=AssertionError('unexpected path I/O')):
                with mock.patch.object(Path, 'stat', side_effect=AssertionError('unexpected file mode read')):
                    with mock.patch.object(os, 'sched_setaffinity', side_effect=AssertionError('affinity changed')):
                        bound = reference.bind_sources(self.api, self.raw)
                        result = self.assemble(bound=bound)
        self.assertEqual(result.proof['Cu_updates'], 0)
        self.assertEqual(dict(os.environ), environment)
        self.assertEqual({name: sys.modules.get(name) for name in names}, modules)


if __name__ == '__main__':
    unittest.main()
