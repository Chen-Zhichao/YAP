"""Small geometry/source controls only; Main runs under finite resource limits."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np
from scipy import sparse

from tools import prepared_coupon_geometry as prepared


D = Path('/app/nanocad/projects/personal/zchen/YAP_STACK/operator_repair_20261015/'
         'development/trial_forcing_closure_v1/continuous_cu_screen_v1')
SOURCE_FILES = dict(append='prepared_dual_bi_total50_static.py', quadrature='common_cell_quadrature.py',
                    geometry='native_periodic_geometry.py', extension='prepared_geometry_extension.py',
                    periodic='bi_remeshed_geometry.py')
OFFSETS = np.array(((0,0,0),(1,0,0),(1,0,1),(0,0,1),
                    (0,1,0),(1,1,0),(1,1,1),(0,1,1)))


def source_bytes():
    result = {}
    for name, filename in SOURCE_FILES.items():
        with (D / filename).open('rb') as stream:
            raw = stream.read(256 * 1024 + 1)
        if not 0 < len(raw) <= 256 * 1024:
            raise ValueError('bounded source fixture required')
        result[name] = raw
    return result


def small_seed(bound):
    nx, ny, nz = 2, 3, 2
    xyz = np.stack(np.meshgrid(np.linspace(0., .012, 3),
        [0., .0015, .0115, .0485], np.linspace(0., .012, 3), indexing='ij'), axis=-1).reshape(-1, 3)
    count = len(xyz)
    nodes = {i + 1: tuple(row) for i, row in enumerate(xyz)}
    nodes.update({i: (0., 0., 0.) for i in range(count + 1, count + 7 + ny + 1)})
    controls = ((count+1,'UX'),(count+3,'UZ'),(count+5+ny+1,'UY'),
                (count+2,'UX'),(count+4,'UZ'),(count+6+ny+1,'UY'))
    constraints = {(1, d): 0. for d in bound.geometry.DOFS}
    constraints.update({(n, d): 0. for n in range(count+1, count+7+ny+1)
                        for d in bound.geometry.DOFS if (n, d) not in controls})
    elements = {}
    for y, background in enumerate((1, 2, 3)):
        for x in range(nx):
            for z in range(nz):
                corners = OFFSETS + [x, y, z]
                ids = (corners[:, 0]*(ny+1)+corners[:, 1])*(nz+1)+corners[:, 2]+1
                elements[len(elements)+1] = (tuple(map(int, ids)), 4 if x == z else background)
    equations = bound.periodic(np, xyz, ny, nx, controls, bound.analytic, bound.geometry.DOFS)
    deck = bound.geometry.NativeDeck(nodes, elements, equations, constraints,
                                    {('EX', 1, 1): 73000.}, {1: 'SOLID185'}, {(1, 2): 0}, 1)
    basis = bound.geometry.six_macro_basis(NS(np=np, sparse=sparse), deck)
    return deck, basis


class PreparedCouponGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = source_bytes()
        cls.bound = prepared.bind_sources(cls.raw)
        cls.api = NS(np=np, sparse=sparse)
        cls.bbar = NS(NATURAL_NODE_COORDINATES=2*OFFSETS-1)

    def fixture(self):
        deck, basis = small_seed(self.bound)
        target = prepared.append_back_bi(self.api, self.bound, seed_deck=deck, seed_basis=basis,
            check=lambda: None, production=False, divisions=(1, 1, 1))
        return deck, basis, target

    def test_authenticated_AST_inverses_and_no_affinity_side_effect(self):
        before = os.sched_getaffinity(0)
        bound = prepared.bind_sources(self.raw)
        self.assertEqual(before, os.sched_getaffinity(0))
        self.assertTrue(bound.proof['append']['inverse_AST_exact'])
        self.assertTrue(bound.proof['tensor_plan']['inverse_AST_exact'])
        self.assertFalse(bound.proof['original_Bbar_assembly_modified'])
        bad = dict(self.raw, quadrature=self.raw['quadrature'] + b'\n')
        with self.assertRaises(ValueError):
            prepared.bind_sources(bad)

    def test_mirrored_material_ledger_order_and_original_seed_retained(self):
        deck, basis, target = self.fixture()
        old = basis['materials'].reshape(3, 2, 2)
        new = target.basis['materials'].reshape(4, 2, 2)
        np.testing.assert_array_equal(new[:3], old)
        np.testing.assert_array_equal(new[3:], old[:1][::-1])
        self.assertEqual([row[0] for row in target.proof['regions']], ['front_bi','beol','si37','back_bi'])
        self.assertEqual(target.proof['Cu_GP'], 64)
        np.testing.assert_array_equal(target.basis['element_ids'], np.arange(1, 17))
        for name in ('elastic', 'element_types', 'element_options', 'first_solve_line'):
            self.assertEqual(getattr(deck, name), getattr(target.deck, name))
        self.assertEqual(len(deck.elements), 12)
        self.assertEqual(float(target.basis['coordinates_mm'][:, 1].max()), .050)
        self.assertFalse(target.proof['actual_census_independently_qualified'])
        self.assertFalse(target.proof['operator_admission'])

    def test_q6_and_homogeneous_CE_nullspace(self):
        _, _, target = self.fixture()
        basis = target.basis
        p = basis['P']
        np.testing.assert_array_equal(basis['macro_fields'],
            self.bound.analytic(np, basis['coordinates_mm']).reshape(-1, 6))
        self.assertEqual(p.shape, (135, 57))
        u = (p @ np.arange(p.shape[1], dtype=float)).reshape(-1, 3)
        for terms in target.deck.equations.values():
            residual = sum(value*u[node-1, self.bound.geometry.DOFS[dof]]
                           for node, dof, value in terms if node <= len(u))
            self.assertEqual(residual, 0.)
        self.assertEqual(np.linalg.matrix_rank(np.column_stack((basis['macro_fields'], p.toarray()))), 63)

    def test_original_three_region_and_adapted_four_region_plans(self):
        _, basis, target = self.fixture()
        primitives = NS(np=np, sparse=sparse, bbar=self.bbar, analytic_fields=self.bound.analytic)
        original = self.bound.original_tensor_plan(primitives, basis, (1,1,1), None, 4096, 32)
        self.assertEqual(list(original.regions), ['bi','beol','si'])
        with self.assertRaises(ValueError):
            self.bound.original_tensor_plan(primitives, target.basis, (1,1,1,1), None, 4096, 32)
        plan = prepared.tensor_plan(self.api, self.bound, target=target, bbar=self.bbar, check=lambda: None)
        self.assertEqual(list(plan.regions), ['front_bi','beol','si','back_bi'])
        self.assertEqual(plan.gp, 64)
        self.assertIsNone(plan.coordinate_quantum_mm)
        bad = NS(basis=dict(target.basis), divisions=target.divisions)
        bad.basis['materials'] = target.basis['materials'].copy()
        bad.basis['materials'][-4:] = np.where(bad.basis['materials'][-4:] == 1, 2, 4)
        with self.assertRaises(ValueError):
            prepared.tensor_plan(self.api, self.bound, target=bad, bbar=self.bbar, check=lambda: None)

    def test_BI4_production_guard_does_not_admit_synthetic_census(self):
        deck, basis = small_seed(self.bound)
        with self.assertRaises(ValueError):
            prepared.append_back_bi(self.api, self.bound, seed_deck=deck, seed_basis=basis, check=lambda: None)
        seed = NS(deck=deck, basis=basis, proof={}, profile=dict(pitch_um=5, diameter_um=2.5,
            source_si_um=37., cell_um=30, mesh_recipe='BI5_TRANSITION127'))
        with self.assertRaises(ValueError):
            prepared.construct(self.api, self.bound, construct_bi4=lambda: seed, check=lambda: None)

    def test_explicit_shape_ceiling_preserves_exact_width_plan(self):
        _, _, target = self.fixture()
        default = prepared.tensor_plan(self.api, self.bound, target=target,
            bbar=self.bbar, check=lambda: None)
        expanded = prepared.tensor_plan(self.api, self.bound, target=target,
            bbar=self.bbar, check=lambda: None, max_shapes=32768)
        np.testing.assert_array_equal(expanded.shape_ids, default.shape_ids)
        np.testing.assert_array_equal(expanded.shape_widths, default.shape_widths)
        self.assertEqual(expanded.regions, default.regions)
        self.assertIsNone(expanded.coordinate_quantum_mm)
        with self.assertRaises(ValueError):
            prepared.tensor_plan(self.api, self.bound, target=target,
                bbar=self.bbar, check=lambda: None, max_shapes=32769)

    def test_exact_BI4_production_predicates_without_large_arrays(self):
        class Materials:
            def __eq__(self, unused):
                return self

        count = [1038976 // 8]
        extent = [np.array([.012, .0485, .012])]
        numerical = NS(count_nonzero=lambda unused: count[0],
                       ptp=lambda unused, axis: extent[0], allclose=np.allclose)
        source = dict(np=numerical, nx=96, ny=44, nz=96, divisions=(6,18,20),
                      mats=Materials(), xyz=None)
        first, last = (compile(ast.fix_missing_locations(ast.Expression(node)),
                               '<authenticated-production-predicate>', 'eval')
                       for node in self.bound.production_guard_AST)
        self.assertTrue(eval(first, {}, source))
        count[0] -= 1
        self.assertFalse(eval(first, {}, source))
        count[0] += 1
        extent[0] = np.array([.006, .0485, .006])
        self.assertFalse(eval(first, {}, source))
        counts = [121728,569088,348160,121728]
        offset = 0
        ranges = []
        for name, size in zip(('front_bi','beol','si','back_bi'), counts):
            ranges.append([name, offset, offset+size])
            offset += size
        target = dict(result={'P':NS(shape=(1439577,1410045))}, offset=1160704, ranges=ranges)
        self.assertTrue(eval(last, {}, target))
        target['offset'] -= 8
        self.assertFalse(eval(last, {}, target))
        target['offset'] += 8
        ranges[1][2] -= 8
        self.assertFalse(eval(last, {}, target))
        ranges[1][2] += 8
        target['result']['P'].shape = (1439577,1410044)
        self.assertFalse(eval(last, {}, target))


if __name__ == '__main__':
    unittest.main()
