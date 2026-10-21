import numpy as np
import pytest

import operator_accuracy_review as review
from operator_snapshot_diagnostic import deny_native, frozen_equilibrium


def fixture_result(passed=True):
    candidate = dict(key="tsv_si_h18p5", files_sha256={"data.npz": "sha"})
    result = dict(key=candidate["key"], candidate_sha256=candidate["files_sha256"], states=1081,
        state="PASS_PRESCRIBED_SCREEN" if passed else "FAIL_PRESCRIBED_SCREEN",
        gates={name: dict(passed=True) for name in review.GATES})
    result["gates"]["gamma_xy"]["passed"] = passed
    return candidate, result


@pytest.mark.parametrize("passed", [True, False])
def test_result_state_and_all_gates(passed):
    record, result = fixture_result(passed)
    review.validate_screen(record, result)
    result["state"] = "FAIL_PRESCRIBED_SCREEN" if passed else "PASS_PRESCRIBED_SCREEN"
    with pytest.raises(ValueError):
        review.validate_screen(record, result)


@pytest.mark.parametrize("field,value", [("states", 1080), ("key", "other"), ("candidate_sha256", {})])
def test_reject_mixed_result_identity(field, value):
    record, result = fixture_result()
    result[field] = value
    with pytest.raises(ValueError):
        review.validate_screen(record, result)


def test_live_and_partial_case_not_reviewed_before_commit():
    record, _ = fixture_result()
    plan = dict(candidates=[record])
    assert review.completed_cases(plan, dict(state="SCORE", active_candidate=record["key"], completed=[])) == []
    assert review.completed_cases(plan, dict(completed=[dict(key=record["key"], state="FAIL_PRESCRIBED_SCREEN")])) == [(record, "FAIL_PRESCRIBED_SCREEN")]


def test_duplicate_and_unassigned_results_fail_closed():
    record, _ = fixture_result()
    row = dict(key=record["key"], state="FAIL_PRESCRIBED_SCREEN")
    for completed in ([row, row], [dict(row, key="unassigned")]):
        with pytest.raises(ValueError):
            review.completed_cases(dict(candidates=[record]), dict(completed=completed))


def test_unchanged_near_zero_and_relative_gates():
    p, r = np.zeros((1081, 134)), np.zeros((1081, 134))
    p[373, 1] = .0408174508094142 * .006**2
    p[1078, 3] = .0005088266651508606 * .006**2
    r[11, 0] = 2. * .006**2
    p[11, 0] = 2.1 * .006**2
    report = review.response_diagnosis(p, r, np.arange(1081.))
    assert report["eps_x"]["passed"] is True
    assert report["eps_y"]["worst_result_set"] == 374
    assert report["eps_y"]["error_over_allowed"] == pytest.approx(40.8174508094142)
    assert report["kappa_x"]["error_over_allowed"] == pytest.approx(50.88266651508606)
    assert report["kappa_x"]["units"] == "N"


@pytest.mark.parametrize("kind", ["shape", "nan", "clock"])
def test_invalid_response_coverage(kind):
    p, r, times = np.zeros((1081, 134)), np.zeros((1081, 134)), np.arange(1081.)
    if kind == "shape":
        p = p[:-1]
    elif kind == "nan":
        r[0, 0] = np.nan
    else:
        times[2] = times[1]
    with pytest.raises(ValueError):
        review.response_diagnosis(p, r, times)


def test_one_result_processed_without_batch_ready(tmp_path, monkeypatch):
    record, _ = fixture_result(False)
    plan = dict(work=str(tmp_path), candidates=[record])
    (tmp_path / "STATUS.json").touch()
    monkeypatch.setattr(review, "read", lambda p, expected=None: (plan, "plan") if str(p) == "planfile" else (
        dict(state="TEACHER", completed=[dict(key=record["key"], state="FAIL_PRESCRIBED_SCREEN")]), "status"))
    calls = []
    monkeypatch.setattr(review, "verify_and_review", lambda *a: calls.append(a) or dict(key=record["key"],
        state="FAIL_PRESCRIBED_SCREEN", repair_pending=True, plan_sha256="plan"))
    reviews = {}
    manifest = dict(plans=[dict(path="planfile", sha256="plan")])
    review.snapshot(manifest, tmp_path, reviews)
    review.snapshot(manifest, tmp_path, reviews)
    assert len(calls) == 1
    assert reviews["plan:" + record["key"]]["repair_pending"] is True


def test_bad_evidence_does_not_hide_another_result(tmp_path, monkeypatch):
    def read(path, expected=None):
        if path == "bad":
            raise ValueError("changed seal")
        return dict(work=str(tmp_path), candidates=[], completed=[]), "sha"
    monkeypatch.setattr(review, "read", read)
    (tmp_path / "STATUS.json").touch()
    errors = review.snapshot(dict(plans=[dict(path="bad", sha256="bad"), dict(path="good", sha256="sha")]), tmp_path, {})
    assert len(errors) == 1
    assert errors[0]["plan"] == "bad"


def test_frozen_internal_strain_reconstructs_full_elastic_balance():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(12, 12))
    stiffness = a.T @ a + np.eye(12)
    load, eigen, macro = rng.normal(size=12), rng.normal(size=12), rng.normal(size=6)
    full, response = frozen_equilibrium(stiffness, load, eigen, macro)
    assert np.array_equal(full[:6], macro)
    assert np.max(abs(response[6:])) < 1e-12
    assert np.allclose(response, stiffness @ full - load - eigen)


def test_cpu_diagnostic_cannot_launch_solver():
    with pytest.raises(RuntimeError, match="cannot launch"):
        deny_native(["ansys"])


def test_nan_frozen_equilibrium_fails_closed():
    with pytest.raises(ValueError, match="nonfinite"):
        frozen_equilibrium(np.eye(2), np.array([np.nan, 0.]), np.zeros(2), np.zeros(1))


def test_scientific_epoch_ignores_only_scheduler_location():
    plan = dict(source="/science/layer", cards={"cu": dict(sha256="cu")}, scoring_contract={"limit": .1})
    pins = dict(files_sha256={"/science/layer/material.py": "mechanism", "/controller/grid.py": "dispatch1"})
    epoch = review.scientific_epoch(plan, pins)
    pins["files_sha256"]["/controller/grid.py"] = "dispatch2"
    assert review.scientific_epoch(plan, pins) == epoch
    pins["files_sha256"]["/science/layer/material.py"] = "changed_mechanism"
    assert review.scientific_epoch(plan, pins) != epoch


def test_full_nodal_old_span_has_no_missing_direction():
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu
    from operator_frozen_residual import complement
    k = csc_matrix(np.diag([2., 3., 4.]))
    z = np.eye(3)[:, :1]
    missing, report = complement(k, splu(k), z, np.array([1., 0., 0.]))
    assert np.max(abs(missing)) < 1e-12
    assert report["missing_fraction"] == pytest.approx(0.)


def test_full_nodal_known_complement_energy():
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu
    from operator_frozen_residual import complement
    k = csc_matrix(np.diag([2., 3., 4.]))
    z = np.eye(3)[:, :1]
    missing, report = complement(k, splu(k), z, np.array([1., 3., 0.]))
    assert np.allclose(missing, [0., -1., 0.])
    assert report["missing_energy_Nmm"] == pytest.approx(1.5)
    assert report["total_relaxation_energy_Nmm"] == pytest.approx(1.75)
    assert report["energy_orthogonality"] < 1e-12


def test_both_faces_remove_columns_not_just_rows():
    from scipy.sparse import eye
    from operator_frozen_residual import both_fixed_faces
    xyz = np.array([[0., 0., 0.], [0., .5, 0.], [0., 1., 0.]])
    p, counts = both_fixed_faces(eye(9, format="csr"), xyz)
    assert p.shape == (9, 3)
    assert p[:3].nnz == p[6:].nnz == 0
    assert np.array_equal(counts, np.ones(3))


def test_full_nodal_constraint_map_checks_actual_ce_and_fixed_nodes():
    from scipy.sparse import csr_matrix
    from operator_frozen_residual import constraint_map_audit
    p = csr_matrix(np.array([[1.], [0.], [0.], [1.], [0.], [0.]]))
    ids = np.array([1, 2])
    deck = "\n".join(f"CE,{i},0,1,UX,1,2,UX,-1" for i in range(1, 5251)) + "\n/SOLU"
    assert constraint_map_audit(p, ids, deck) == 5250
    with pytest.raises(ValueError, match="fixed physical"):
        constraint_map_audit(p, ids, "D,1,UX,0\n" + deck)
    with pytest.raises(ValueError, match="constraint"):
        constraint_map_audit(p, ids, deck.replace("2,UX,-1", "2,UX,1", 1))
