"""Tiny in-memory tests of authenticated geometry definitions, not FE physics.

Source reads and NumPy/SciPy imports happen only when Main runs this suite.
No production deck, payload, Cu update, solver, subprocess or output is used.
"""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from tools import operator_geometry as height


PROJECT = Path('/app/nanocad/projects/personal/zchen/YAP_STACK')
SOURCE = PROJECT / 'operator_repair_20261015/development/trial_forcing_closure_v1/continuous_cu_screen_v1'
STANDARD = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))
REPOSITORY = ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1),
              (0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1))


class OperatorGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from scipy import sparse
        cls.np = np
        cls.api = SimpleNamespace(np=np, sparse=sparse)

        def bounded_source(name):
            with (SOURCE / name).open('rb') as stream:
                raw = stream.read(height.SOURCE_CAP + 1)
            if not 0 < len(raw) <= height.SOURCE_CAP:
                raise AssertionError('tiny source-only fixture exceeded source cap')
            return raw

        cls.geometry_raw = bounded_source('native_periodic_geometry.py')
        cls.extension_raw = bounded_source('prepared_geometry_extension.py')
        cls.bound = height.bind_sources(cls.geometry_raw, cls.extension_raw)

    def setUp(self):
        self.check = mock.Mock()
        self.deck, self.basis, self.contract = self.fixture()

    @staticmethod
    def node(i, j, k):
        return 1 + (i * 4 + j) * 3 + k

    def independent_a6(self, xyz):
        """Scalar polynomial oracle, independent of the extension implementation."""
        np = self.np
        rows = []
        for px, y, pz in xyz:
            x, z = px - .003, pz - .003
            rows.append(((x, 0., z / 2, y * x, 0., y * z / 2),
                         (0., 0., 0., -x * x / 2, -z * z / 2, -x * z / 2),
                         (0., z, x / 2, 0., y * z, y * x / 2)))
        fields = np.asarray(rows, dtype=float)
        return fields - fields[0].copy()

    def fixture(self, template=STANDARD):
        np = self.np
        ys = (0., .0015, .0115, .0485)
        nodes = {self.node(i, j, k): (.003 * i, y, .003 * k)
                 for i in range(3) for j, y in enumerate(ys) for k in range(3)}
        for node in range(37, 47):
            nodes[node] = (0., 0., 0.)
        controls = ((37, 'UX'), (39, 'UZ'), (45, 'UY'),
                    (38, 'UX'), (40, 'UZ'), (46, 'UY'))
        constraints = {(1, dof): 0. for dof in ('UX', 'UY', 'UZ')}
        constraints.update({(node, dof): 0. for node in range(37, 47)
                            for dof in ('UX', 'UY', 'UZ') if (node, dof) not in controls})
        elements = {}
        for j in range(3):
            for i in range(2):
                for k in range(2):
                    connectivity = tuple(self.node(i + di, j + dj, k + dk) for di, dj, dk in template)
                    elements[len(elements) + 1] = (connectivity, 4 if (i, k) == (0, 0) else j + 1)
        a = self.independent_a6([nodes[node] for node in range(1, 37)])
        equations = {}
        for i in range(3):
            for j in range(4):
                for k in range(3):
                    if i != 2 and k != 2:
                        continue
                    slave = self.node(i, j, k)
                    master = self.node(0 if i == 2 else i, j, 0 if k == 2 else k)
                    for component, dof in enumerate(('UX', 'UY', 'UZ')):
                        jump = a[slave - 1, component] - a[master - 1, component]
                        terms = ((slave, dof, 1.), (master, dof, -1.))
                        terms += tuple((node, label, -float(coefficient))
                                       for (node, label), coefficient in zip(controls, jump) if coefficient != 0.)
                        equations[len(equations) + 1] = terms
        elastic = {(label, material, 1): value for material in range(1, 5)
                   for label, value in (('EX', 100000.), ('PRXY', .25), ('ALPX', 2.6e-6))}
        deck = self.bound.geometry.NativeDeck(nodes=nodes, elements=elements,
            equations=equations, constraints=constraints, elastic=elastic,
            element_types={1: 'SOLID185'}, element_options={(1, 2): 0}, first_solve_line=1)
        basis = self.bound.geometry.six_macro_basis(self.api, deck)
        contract = dict(case='SOFTWARE_TINY_CARTESIAN_NOT_PRODUCTION', mesh_recipe='UNIT_2x3x2',
            pitch_um=2, diameter_pitch_ratio=.5, source_si_um=37., cell_xz_um=[6, 6],
            physical_nodes=36, hex_elements=12, deck_nodes=46, equations=60,
            constraints=len(constraints), body_coordinates=45, Cu_GP=24,
            region_Cu_GP=dict(bi=8, beol=8, si=8),
            seed_report=dict(path=str(PROJECT / 'runtime_checks/UNIT_ONLY_NOT_AUTHENTICATED/REPORT.json'),
                             bytes=1, sha256='0' * 64))
        return deck, basis, contract

    def prepare(self, request=None, *, deck=None, basis=None, contract=None, check=None):
        return height.prepare_target(self.api, self.bound,
            seed_deck=self.deck if deck is None else deck,
            seed_basis=self.basis if basis is None else basis,
            request=height.request(2, 20, 'inner') if request is None else request,
            seed_contract=self.contract if contract is None else contract,
            check=self.check if check is None else check)

    def test_exact_sources_and_selected_definitions(self):
        self.assertEqual(self.bound.source_pins, dict(geometry=height.GEOMETRY_SHA, extension=height.EXTENSION_SHA))
        self.assertFalse(hasattr(self.bound.extension, 'execute'))
        self.assertFalse(hasattr(self.bound.extension, 'proof_for'))
        for geometry, extension in ((self.geometry_raw + b'\n', self.extension_raw),
                                    (self.geometry_raw, self.extension_raw + b'\n'),
                                    (self.geometry_raw.decode('ascii'), self.extension_raw)):
            with self.subTest(geometry_type=type(geometry)), self.assertRaises(ValueError):
                height.bind_sources(geometry, extension)

    def test_seed_contract_uses_measured_not_legacy_hardcoded_census(self):
        seed = SimpleNamespace(deck=self.deck, basis=self.basis,
            profile=dict(pitch_um=2, diameter_um=1., source_si_um=37., cell_um=6,
                         mesh_recipe=self.contract['mesh_recipe']),
            proof=dict(case=self.contract['case'], target_Cu_GP_ledger=185696))
        contract = height.measure_seed_contract(self.api, self.bound, seed=seed,
            seed_report=self.contract['seed_report'], check=self.check)
        self.assertEqual(contract, self.contract)
        self.assertEqual(contract['Cu_GP'], 24)
        self.assertNotEqual(contract['Cu_GP'], seed.proof['target_Cu_GP_ledger'])
        self.prepare(contract=contract)

    def test_seed_contract_rejects_inconsistent_profile(self):
        seed = SimpleNamespace(deck=self.deck, basis=self.basis,
            profile=dict(pitch_um=2, diameter_um=1., source_si_um=48.5, cell_um=6,
                         mesh_recipe=self.contract['mesh_recipe']), proof={})
        with self.assertRaises(ValueError):
            height.measure_seed_contract(self.api, self.bound, seed=seed,
                seed_report=self.contract['seed_report'], check=self.check)

    def test_original_fixture_controls_gauge_and_exact_p_structure(self):
        self.assertEqual(self.basis['P'].shape, (108, 45))
        self.assertEqual(self.basis['P'].nnz, 96)
        self.assertEqual(self.basis['controls'], ((37, 'UX'), (39, 'UZ'), (45, 'UY'),
                                               (38, 'UX'), (40, 'UZ'), (46, 'UY')))
        self.assertEqual(len(self.deck.equations), 60)
        self.assertEqual({key: value for key, value in self.deck.constraints.items() if key[0] <= 36},
                         {(1, dof): 0. for dof in ('UX', 'UY', 'UZ')})
        self.np.testing.assert_array_equal(self.basis['macro_fields'],
            self.independent_a6(self.basis['coordinates_mm']).reshape(108, 6))

    def test_request_inner_outer_and_continuous_thickness(self):
        for pitch in range(2, 11):
            for total, variant, si in ((20, 'inner', 7), (20, 'outer', 8.5),
                                       (100, 'inner', 87), (100, 'outer', 88.5),
                                       (55.25, 'inner', 42.25)):
                with self.subTest(pitch=pitch, total=total, variant=variant):
                    req = height.request(pitch, total, variant)
                    self.assertEqual(req.si_um, si)
                    self.assertEqual(req.diameter_um / pitch, .5)

    def test_request_invalid_values(self):
        for args in ((True, 50, 'inner'), (1, 50, 'inner'), (11, 50, 'inner'),
                     (2., 50, 'inner'), (2, True, 'inner'), (2, 19, 'inner'),
                     (2, 101, 'inner'), (2, float('nan'), 'inner'), (2, 50, 'both')):
            with self.subTest(args=args), self.assertRaises(ValueError):
                height.request(*args)

    def test_endpoint_and_identity_retarget_preserve_geometry_contract(self):
        np = self.np
        before_deck = deepcopy(self.deck)
        before_arrays = {name: self.basis[name].tobytes() for name in
                         ('node_ids', 'coordinates_mm', 'macro_fields', 'element_ids', 'cells', 'materials')}
        for total, variant in ((20, 'inner'), (20, 'outer'), (100, 'inner'), (100, 'outer'),
                               (50, 'inner'), (48.5, 'outer'), (55.25, 'inner')):
            with self.subTest(total=total, variant=variant):
                req = height.request(2, total, variant)
                result = self.prepare(req)
                self.assertIsInstance(result, height.GeometryTarget)
                old, new = self.basis['coordinates_mm'], result.basis['coordinates_mm']
                expected = old.copy()
                above = old[:, 1] > .0115
                if req.si_um != 37.:
                    expected[above, 1] = .0115 + (old[above, 1] - .0115) * (req.si_um / 37.)
                np.testing.assert_array_equal(new, expected)
                np.testing.assert_array_equal(result.basis['macro_fields'], self.independent_a6(new).reshape(108, 6))
                self.assertTrue(self.bound.extension.same_graph(np, self.basis['P'], result.basis['P']))
                self.assertEqual(result.deck.elements, self.deck.elements)
                self.assertEqual(result.deck.elastic, self.deck.elastic)
                self.assertEqual(result.deck.element_options, self.deck.element_options)
                self.assertEqual(result.proof['Cu_GP'], 24)
                for role in ('bi', 'beol', 'si'):
                    source = result.proof['source_regions'][role]
                    target = result.proof['target_regions'][role]
                    self.assertEqual(target['Cu_GP'], 8)
                    ratio = req.si_um / 37. if role == 'si' else 1.
                    self.assertAlmostEqual(target['volume_mm3'], source['volume_mm3'] * ratio, delta=1e-18)
                    self.assertAlmostEqual(target['Cu_volume_mm3'], source['Cu_volume_mm3'] * ratio, delta=1e-18)
                for flag in ('FE_assembly_performed', 'Cu_state_transfer', 'history_executed',
                             'stiffness_or_GP_strain_maps_constructed', 'operator_admission',
                             'geometry_family_validated', 'resource_admission'):
                    self.assertIs(result.proof[flag], False)
                self.assertEqual(result.proof['FE_factorizations'], 0)
                self.assertEqual(result.proof['Cu_updates'], 0)
        self.assertEqual(self.deck, before_deck)
        for name, before in before_arrays.items():
            self.assertEqual(self.basis[name].tobytes(), before)
        self.assertGreater(self.check.call_count, 0)

    def test_both_hex_handedness_templates_preserve_original_cells(self):
        for template in (STANDARD, REPOSITORY):
            with self.subTest(template=template):
                deck, basis, contract = self.fixture(template)
                result = self.prepare(deck=deck, basis=basis, contract=contract)
                self.np.testing.assert_array_equal(result.basis['cells'], basis['cells'])
                self.assertEqual(result.deck.elements, deck.elements)
                self.assertEqual(result.proof['Cu_GP'], 24)

    def test_all_six_target_fields_satisfy_regenerated_ce_and_gauge(self):
        result = self.prepare(height.request(2, 100, 'outer'))
        a = self.independent_a6(result.basis['coordinates_mm'])
        controls = {control: self.np.eye(6)[j] for j, control in enumerate(result.basis['controls'])}
        for terms in result.deck.equations.values():
            residual = self.np.zeros(6)
            for node, dof, coefficient in terms:
                field = a[node - 1, {'UX': 0, 'UY': 1, 'UZ': 2}[dof]] if node <= 36 else controls[(node, dof)]
                residual += coefficient * field
            self.np.testing.assert_allclose(residual, 0., rtol=0., atol=1e-18)
        self.np.testing.assert_array_equal(a[0], self.np.zeros((3, 6)))

    def test_wrong_seed_contracts_and_census_are_rejected(self):
        faults = (dict(source_si_um=48.5), dict(pitch_um=8), dict(cell_xz_um=[24, 24]),
                  dict(diameter_pitch_ratio=.4), dict(physical_nodes=35), dict(hex_elements=11),
                  dict(body_coordinates=44), dict(Cu_GP=185696),
                  dict(region_Cu_GP=dict(bi=0, beol=8, si=16)))
        for fault in faults:
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                self.prepare(contract=dict(self.contract, **fault))

    def test_contract_requires_exact_fields_and_actual_binding_structure(self):
        for mutation in ('missing', 'extra', 'binding'):
            contract = deepcopy(self.contract)
            if mutation == 'missing':
                del contract['mesh_recipe']
            elif mutation == 'extra':
                contract['authenticated'] = True
            else:
                contract['seed_report']['sha256'] = 'PENDING'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.prepare(contract=contract)

    def test_tampered_request_unbound_sources_and_guard_abort(self):
        with self.assertRaises(ValueError):
            self.prepare(height.request(2, 20, 'inner')._replace(si_um=8.5))
        with self.assertRaises(ValueError):
            height.prepare_target(self.api, SimpleNamespace(token=object()), seed_deck=self.deck,
                seed_basis=self.basis, request=height.request(2, 20, 'inner'),
                seed_contract=self.contract, check=self.check)
        with self.assertRaisesRegex(RuntimeError, 'fixture guard stop'):
            self.prepare(check=mock.Mock(side_effect=RuntimeError('fixture guard stop')))

    def test_supplied_basis_must_match_deck_and_be_readonly(self):
        for writable in (False, True):
            basis = dict(self.basis)
            fields = basis['macro_fields'].copy()
            fields[5, 2] += 1e-4
            fields.setflags(write=writable)
            basis['macro_fields'] = fields
            with self.subTest(writable=writable), self.assertRaises(ValueError):
                self.prepare(basis=basis)

    def test_original_core_rejects_prescribed_control_extra_gauge_and_aux_loss(self):
        cases = []
        constraints = dict(self.deck.constraints)
        constraints[(37, 'UX')] = 0.
        cases.append(replace(self.deck, constraints=constraints))
        constraints = dict(self.deck.constraints)
        constraints[(2, 'UY')] = 0.
        cases.append(replace(self.deck, constraints=constraints))
        nodes = dict(self.deck.nodes)
        del nodes[46]
        cases.append(replace(self.deck, nodes=nodes))
        for deck in cases:
            with self.subTest(constraints=len(deck.constraints), nodes=len(deck.nodes)), self.assertRaises(ValueError):
                self.bound.geometry.six_macro_basis(self.api, deck)

    def test_original_retarget_rejects_wrong_source_height_and_malformed_ce(self):
        with self.assertRaises(ValueError):
            self.bound.extension.retarget(self.api, self.deck, 7., source_si_um=48.5)
        equations = dict(self.deck.equations)
        terms = list(equations[1])
        node, dof, _ = terms[0]
        terms[0] = (node, dof, 2.)
        equations[1] = tuple(terms)
        with self.assertRaises(ValueError):
            self.bound.extension.retarget(self.api, replace(self.deck, equations=equations), 7., source_si_um=37.)

    def test_original_retarget_rejects_hinge_crossing_and_nonorigin_node1(self):
        nodes = dict(self.deck.nodes)
        for i in range(3):
            for k in range(3):
                node = self.node(i, 1, k)
                x, _, z = nodes[node]
                nodes[node] = (x, .012, z)
        with self.assertRaisesRegex(ValueError, 'crosses'):
            self.bound.extension.retarget(self.api, replace(self.deck, nodes=nodes), 7., source_si_um=37.)
        nodes = dict(self.deck.nodes)
        nodes[1] = (.0001, 0., 0.)
        with self.assertRaisesRegex(ValueError, 'origin'):
            self.bound.extension.retarget(self.api, replace(self.deck, nodes=nodes), 7., source_si_um=37.)

    def test_target_material_change_is_rejected(self):
        original = self.bound.extension.retarget

        def corrupt(*args, **kwargs):
            target = original(*args, **kwargs)
            elastic = dict(target.elastic)
            elastic[('EX', 4, 1)] *= 2
            return replace(target, elastic=elastic)

        with mock.patch.object(self.bound.extension, 'retarget', side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, 'cards/options/connectivity'):
                self.prepare()

    def test_duplicate_brick_and_mixed_corner_templates_are_rejected(self):
        for fault in ('duplicate', 'mixed'):
            basis = dict(self.basis)
            cells = basis['cells'].copy()
            cells[1] = cells[0] if fault == 'duplicate' else cells[1, [0, 3, 2, 1, 4, 7, 6, 5]]
            cells.setflags(write=False)
            basis['cells'] = cells
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                height._ledger(self.np, basis, self.check)


if __name__ == '__main__':
    unittest.main()
