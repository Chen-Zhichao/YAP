"""Tiny synthetic matrix tests; no production geometry, Cu history or solver job."""
import hashlib
import os
from pathlib import Path
import unittest

import numpy as np
from scipy import sparse

from tools import operator_construction as construction


ROOT = Path(os.environ.get("YAP_DATA_ROOT", "/app/nanocad/projects/personal/zchen/YAP_STACK"))
RELATIVE = {
    "elastic": "operator_night_20261021/NATIVE_ELASTIC_LIFT_20261029_v1/native_elastic_lift_v2.py",
    "span": "operator_night_20261021/NATIVE_NONLINEAR_SPAN_20261029_v1/native_nonlinear_span_v3.py",
    "projection": "operator_night_20261021/NATIVE_DIMENSIONAL_CU_20261029_v1/dimensionprojection.py",
}


def frozen(value):
    value.setflags(write=False)
    return value


def fixture(n=9, *, rank=9, thermal=True):
    rng = np.random.default_rng(207)
    physical = n + 6
    columns = np.r_[np.arange(n), [0, 3, 7]]
    p = sparse.csr_matrix((np.ones(n + 3), (np.arange(3, physical), columns)), shape=(physical, n))
    raw = rng.normal(size=(n, n))
    k = raw.T @ raw + np.eye(n) * 4
    c = rng.normal(size=(n, 6))
    lift = -np.linalg.solve(k, c)
    schur = np.diag(np.arange(2., 8.))
    kaa = schur - c.T @ lift
    ff = rng.normal(size=n) if thermal else np.zeros(n)
    w = np.linalg.solve(k, ff)
    a = rng.normal(size=(physical, 6))
    a[:3] = 0
    mq = np.diag(np.linspace(.5, 1., 6))
    tq = rng.normal(size=6)
    fm = schur @ tq + c.T @ w
    arrays = dict(kaa=kaa, c=c, fm=fm, ff=ff, macro_fields=a, mechanical_q=mq,
        mechanical_u=a @ mq + p @ (lift @ mq), mechanical_reactions=schur @ mq,
        thermal_q=tq, thermal_u=a @ tq + p @ (lift @ tq + w), schur=schur)
    for name, matrix in (("P", p), ("kff", sparse.csr_matrix(k))):
        arrays.update({name + "_data": matrix.data, name + "_indices": matrix.indices,
                       name + "_indptr": matrix.indptr, name + "_shape": np.array(matrix.shape, dtype=np.int64)})
    e0, e1 = np.eye(n)[:, 0], np.eye(n)[:, 1]
    first = .2 * e0 if rank >= 8 else np.zeros(n)
    second = -.13 * e0 + .3 * e1 if rank == 9 else .5 * first
    q8, q16 = np.linspace(.01, .06, 6), np.linspace(-.03, .02, 6)
    coordinates = {8: np.r_[q8, lift @ q8 + 2 * w + first],
                   16: np.r_[q16, lift @ q16 + 3 * w + second]}
    for value in (*arrays.values(), *coordinates.values()):
        frozen(value)
    return arrays, coordinates


class OperatorConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = {}
        for name, relative in RELATIVE.items():
            path = ROOT / relative
            if not path.is_file():
                raise unittest.SkipTest("Pinned offline source closure unavailable: " + str(path))
            if path.stat().st_size > 256 * 1024:
                raise ValueError("bounded offline source required")
            cls.raw[name] = path.read_bytes()

    def session(self, check=lambda: None):
        return construction.Construction(np, sparse, construction.bind_sources(self.raw),
            case="SYNTHETIC_ONLY", native_sha256="a" * 64, check=check)

    def basis(self, n=9, rank=9):
        arrays, coordinates = fixture(n, rank=rank)
        session = self.session()
        session.elastic(arrays)
        answer = session.enrich(coordinates, frame=session.frame)
        return session, answer

    def maps(self, n, gp):
        rng = np.random.default_rng(73)
        bm = frozen(rng.normal(size=(6 * gp, 6)))
        bf = sparse.csr_matrix(rng.normal(size=(6 * gp, n)))
        for value in (bf.data, bf.indices, bf.indptr):
            frozen(value)
        return dict(B_macro=bm, B_free=bf)

    def test_frozen_sources_and_tolerances(self):
        bound = construction.bind_sources(self.raw)
        self.assertEqual(dict(bound.source_pins), dict(construction.SOURCE_PINS))
        self.assertEqual(bound.elastic.RTOL, 1e-8)
        self.assertEqual(bound.projection.CERT_RTOL, 1e-10)
        with self.assertRaises(TypeError):
            bound.source_pins["elastic"] = "0" * 64

    def test_source_drift_or_extra_source_rejected(self):
        for raw in (dict(self.raw, elastic=self.raw["elastic"] + b"\n"), dict(self.raw, extra=b"x")):
            with self.subTest(keys=list(raw)), self.assertRaises(ValueError):
                construction.bind_sources(raw)

    def test_matching_pin_metadata_without_source_binding_is_rejected(self):
        unbound = construction.Algebra(None, None, None, construction.SOURCE_PINS, None)
        with self.assertRaisesRegex(ValueError, "authenticated construction providers"):
            construction.Construction(np, sparse, unbound, case="SYNTHETIC_ONLY",
                native_sha256="a" * 64, check=lambda: None)

    def test_dimensions_are_not_bi5_or_bi8_census_constants(self):
        for n in (9, 12, 15):
            with self.subTest(n=n):
                arrays, _ = fixture(n)
                result = self.session().elastic(arrays)
                self.assertEqual(result["Qbody7"].shape, (n, 7))
                self.assertEqual(result["report"]["body_coordinates"], n)

    def test_rank_is_resolved_not_forced_to_nine(self):
        for rank in (7, 8, 9):
            with self.subTest(rank=rank):
                session, result = self.basis(rank=rank)
                self.assertEqual(result["Q_body"].shape[1], rank)
                self.assertFalse(result["report"]["forced_rank"])
                self.assertEqual(session.state, "SPAN_CLOSED_TRAINING_ONLY")

    def test_actual_frozen_projection_all_dynamic_gp_rows(self):
        for n, gp in ((9, 3), (12, 7)):
            with self.subTest(n=n, gp=gp):
                session, basis = self.basis(n=n)
                native = self.maps(n, gp)
                bm, bf = native["B_macro"].copy(), native["B_free"].copy()
                result = session.project(native, expected_gp=gp, block_gp=2)
                np.testing.assert_allclose(result["B_macro"], bm + bf @ basis["Q_body"][:, :6], rtol=1e-11, atol=1e-12)
                np.testing.assert_allclose(result["B_body"], bf @ basis["Q_body"][:, 6:], rtol=1e-11, atol=1e-12)
                np.testing.assert_array_equal(native["B_macro"], bm)
                np.testing.assert_array_equal(native["B_free"].toarray(), bf.toarray())
                self.assertEqual(result["report"]["GP_count"], gp)
                self.assertEqual(session.state, "MAPS_CLOSED_UNVALIDATED")
                self.assertFalse(session.evidence()["operator_admission"])
                self.assertFalse(session.evidence()["numeric_input_custody_verified_here"])

    def test_gp_census_mismatch_is_terminal(self):
        session, _ = self.basis()
        with self.assertRaisesRegex(ValueError, "census"):
            session.project(self.maps(9, 3), expected_gp=4)
        self.assertEqual(session.state, "FAILED")
        with self.assertRaises(ValueError):
            session.project(self.maps(9, 3), expected_gp=3)

    def test_cooldown_snapshot_not_accepted_for_training(self):
        arrays, coordinates = fixture()
        session = self.session()
        session.elastic(arrays)
        with self.assertRaisesRegex(ValueError, "CP8/16"):
            session.enrich(dict(coordinates, **{"24": coordinates[16]}), frame=session.frame)
        self.assertEqual(session.state, "FAILED")

    def test_native_frame_case_or_reference_mismatch_rejected(self):
        arrays, coordinates = fixture()
        session = self.session()
        session.elastic(arrays)
        other = construction.training_frame(case="DIFFERENT_NATIVE", native_sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "frame"):
            session.enrich(coordinates, frame=other)

    def test_mutable_input_and_extra_cu_state_members_rejected(self):
        for field in ("mutable", "cu_state"):
            with self.subTest(field=field):
                arrays, _ = fixture()
                if field == "mutable":
                    arrays["kaa"] = arrays["kaa"].copy()
                else:
                    arrays["plastic_strain"] = frozen(np.zeros((3, 6)))
                session = self.session()
                with self.assertRaises(ValueError):
                    session.elastic(arrays)
                self.assertEqual(session.state, "FAILED")

    def test_elastic_rank_six_not_silently_padded(self):
        arrays, coordinates = fixture(thermal=False)
        session = self.session()
        result = session.elastic(arrays)
        self.assertEqual(result["Qbody7"].shape[1], 6)
        with self.assertRaisesRegex(ValueError, "rank6"):
            session.enrich(coordinates, frame=session.frame)

    def test_numerical_failure_does_not_allow_retry(self):
        arrays, _ = fixture()
        arrays["mechanical_reactions"] = frozen(arrays["mechanical_reactions"] + 1)
        session = self.session()
        with self.assertRaises(ValueError):
            session.elastic(arrays)
        self.assertEqual(session.state, "FAILED")
        with self.assertRaises(ValueError):
            session.elastic(fixture()[0])

    def test_out_of_order_call_never_runs_a_later_provider(self):
        session = self.session()
        with self.assertRaises(ValueError):
            session.project(self.maps(9, 3), expected_gp=3)
        self.assertEqual(session.state, "NEW")

    def test_owner_guard_failure_is_terminal(self):
        calls = []
        def check():
            calls.append(1)
            if len(calls) > 1:
                raise TimeoutError("synthetic expiry")
        session = self.session(check=check)
        with self.assertRaises(TimeoutError):
            session.elastic(fixture()[0])
        self.assertEqual(session.state, "FAILED")

    def test_no_input_mutation_or_admission_claim(self):
        arrays, coordinates = fixture()
        hashes = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in arrays.items()}
        session = self.session()
        session.elastic(arrays)
        session.enrich(coordinates, frame=session.frame)
        self.assertEqual(hashes, {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in arrays.items()})
        proof = session.evidence()
        for field in ("resource_admission", "Cu_state_adoption", "geometry_family_validated", "operator_admission",
                      "bow_accuracy_validated", "component_interface_qualified"):
            self.assertFalse(proof[field])
        self.assertEqual(proof["Cu_updates"], 0)


if __name__ == "__main__":
    unittest.main()
