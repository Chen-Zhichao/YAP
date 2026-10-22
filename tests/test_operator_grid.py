"""Offline grid bookkeeping and launch guards; no physical solver is invoked."""

import hashlib
import argparse
import json
import csv
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import numpy as np

import tools.operator_grid as grid


def test_constraint_substitution_has_exact_independent_row_space_witness():
    rows = [dict(equation=1, slave=(8, "UZ"), constant=".125",
                 terms=[(8, "UZ", 1), (4, "UZ", -1), (10, "UY", "-.003")]),
            dict(equation=2, slave=(4, "UZ"), constant=0,
                 terms=[(4, "UZ", 1), (3, "UZ", -1), (9, "UZ", "-.006")]),
            dict(equation=3, slave=(9, "UZ"), constant=0,
                 terms=[(9, "UZ", 1), (11, "UZ", -1), (12, "UZ", "-.0015")])]
    result = grid.condense_constraint_rows(rows, prescribed=[(11, "UZ"), (12, "UZ")])
    dofs = sorted({term[:2] for row in rows for term in row["terms"]})
    vector = lambda row: [sum((Fraction(str(c)) for n, d, c in row["terms"] if (n, d) == key), Fraction(0))
                          for key in dofs]+[Fraction(str(row["constant"]))]
    original = {row["equation"]: vector(row) for row in rows}
    for row, proof in zip(result["rows"], result["report"]["row_combinations"], strict=True):
        combined = [sum((Fraction(c)*original[int(i)][j] for i, c in proof.items()), Fraction(0))
                    for j in range(len(dofs)+1)]
        assert vector(row) == combined
        assert proof[str(row["equation"])] == "1"
    first = result["rows"][0]
    assert first["terms"] == [(8, "UZ", 1), (3, "UZ", -1), (10, "UY", Fraction("-.003")),
                              (11, "UZ", Fraction("-.006")), (12, "UZ", Fraction("-.000009"))]
    assert not result["report"]["runtime_admitted"]


@pytest.mark.parametrize("fault", ["cycle", "duplicate_slave", "prescribed_slave", "nonunit", "nonfinite", "duplicate_term"])
def test_constraint_substitution_rejects_invalid_declared_topology(fault):
    rows = [dict(equation=1, slave=(2, "UX"), constant=0, terms=[(2, "UX", 1), (1, "UX", -1)])]
    fixed = []
    if fault == "cycle":
        rows.append(dict(equation=2, slave=(1, "UX"), constant=0, terms=[(1, "UX", 1), (2, "UX", -1)]))
    elif fault == "duplicate_slave":
        rows.append(dict(rows[0], equation=2))
    elif fault == "prescribed_slave":
        fixed = [(2, "UX")]
    elif fault == "nonunit":
        rows[0]["terms"][0] = (2, "UX", 2)
    elif fault == "nonfinite":
        rows[0]["terms"][1] = (1, "UX", "NaN")
    else:
        rows[0]["terms"].append((1, "UX", 1))
    with pytest.raises(ValueError):
        grid.condense_constraint_rows(rows, prescribed=fixed)


def test_literal_ce_condensation_preserves_non_ce_commands_and_appends_terms():
    deck = "CEDELE,ALL\nCE,1,0,8,UZ,1,4,UZ,-1,10,UY,-.003\nD,1,UX,0\nCE,2,0,4,UZ,1,3,UZ,-1,9,UZ,-.006\n/SOLU\nKBC,0\nSOLVE\n"
    rewritten, report = grid.condense_literal_ce_deck(deck, declared_slaves=[(8, "UZ"), (4, "UZ")])
    assert len([line for line in rewritten.splitlines() if line.startswith("CE,1,")]) == 2
    assert [line for line in rewritten.splitlines() if not line.startswith("CE,")] == [
        line for line in deck.splitlines() if not line.startswith("CE,")]
    assert report["diagnostic_only"] and not report["production_training_modified"]
    with pytest.raises(ValueError, match="PREP7"):
        grid.condense_literal_ce_deck("/SOLU\n"+deck, declared_slaves=[(8, "UZ"), (4, "UZ")])
    with pytest.raises(ValueError, match="membership"):
        grid.condense_literal_ce_deck(deck, declared_slaves=[(8, "UZ"), (4, "UZ"), (5, "UZ")])


def independent_uniform_constraint_deck(nx=2, ny=2, nz=2):
    physical = (nx+1)*(ny+1)*(nz+1)
    node = lambda i, j, k: 1+(i*(ny+1)+j)*(nz+1)+k
    rows = []
    for j in range(ny+1):
        qnode, y = physical+5+j, .0015*j/ny
        for d, membrane, curvature in (("UX", physical+1, physical+2),
                                        ("UZ", physical+3, physical+4),
                                        ("UY", physical+6+ny, physical+7+ny)):
            rows.append([(qnode, d, 1), (membrane, d, -1), (curvature, d, -y)])
        for k in range(nz+1):
            left, right, z = node(0,j,k), node(nx,j,k), .006*k/nz-.003
            rows.extend([[(right,"UX",1),(left,"UX",-1),(qnode,"UX",-.006)],
                         [(right,"UY",1),(left,"UY",-1),(physical+7+ny,"UY",.003*z)],
                         [(right,"UZ",1),(left,"UZ",-1),(qnode,"UY",-.003)]])
        for i in range(nx):
            front, back, x = node(i,j,0), node(i,j,nz), .006*i/nx-.003
            rows.extend([[(back,"UX",1),(front,"UX",-1),(qnode,"UY",-.003)],
                         [(back,"UY",1),(front,"UY",-1),(physical+7+ny,"UY",.003*x)],
                         [(back,"UZ",1),(front,"UZ",-1),(qnode,"UZ",-.006)]])
    lines = ["/PREP7", "CEDELE,ALL", "MP,EX,1,100000", "D,1,UX,0"]
    for number, terms in enumerate(rows, 1000000):
        lines.append(",".join(["CE", str(number), "0"]+[str(v) for term in terms for v in term]))
    return "\n".join(lines+["FINISH", "/SOLU", "KBC,0", "SOLVE"])+"\n"


@pytest.mark.parametrize("direction", range(6))
@pytest.mark.parametrize("sign", [-1, 1])
def test_full_grid_candidate_preserves_independent_signed_native_fields(direction, sign):
    deck = independent_uniform_constraint_deck()
    result, report = grid.candidate_training_constraint_deck(deck, nx=2, ny=2, nz=2)
    q = np.zeros(6)
    q[direction] = sign*(.001 if direction < 3 else 1.2)
    ex, ez, gamma, kx, kz, twist = q
    values = {}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                node, x, y, z = 1+(i*3+j)*3+k, .003*i-.003, .00075*j, .003*k-.003
                u = ((ex+kx*y)*x+.5*(gamma+twist*y)*z,
                     -.5*(kx*x*x+kz*z*z+twist*x*z),
                     (ez+kz*y)*z+.5*(gamma+twist*y)*x)
                values.update({(node,d):v for d,v in zip(("UX","UY","UZ"),u,strict=True)})
    values.update({(28,"UX"):ex,(29,"UX"):kx,(30,"UZ"):ez,(31,"UZ"):kz,
                   (35,"UY"):gamma,(36,"UY"):twist})
    for j in range(3):
        values.update({(32+j,"UX"):ex+kx*.00075*j,
                       (32+j,"UZ"):ez+kz*.00075*j,
                       (32+j,"UY"):gamma+twist*.00075*j})
    equations = {}
    for line in result.splitlines():
        if not line.startswith("CE,"):
            continue
        fields = line.split(",")
        equations.setdefault(int(fields[1]), []).extend(
            (int(fields[i]),fields[i+1],float(fields[i+2])) for i in range(3,len(fields),3))
    assert len(equations) == 54
    for terms in equations.values():
        assert abs(sum(c*values[(n,d)] for n,d,c in terms)) < 1e-19
    assert [s for s in result.splitlines() if not s.startswith("CE,")] == [
        s for s in deck.splitlines() if not s.startswith("CE,")]
    assert report["grid"]["physical_nodes"] == 27
    assert not report["native_solver_qualified"] and report["diagnostic_only"]


@pytest.mark.parametrize("fault", ["dimensions", "ordering", "missing", "solution", "late"])
def test_full_grid_candidate_fails_closed_on_membership_changes(fault):
    deck = independent_uniform_constraint_deck()
    dimensions = dict(nx=2,ny=2,nz=2)
    if fault == "dimensions":
        dimensions["nx"] = 3
    elif fault == "ordering":
        deck = deck.replace("CE,1000000,", "CE,2000000,", 1)
    elif fault == "missing":
        deck = "\n".join(line for line in deck.splitlines() if not line.startswith("CE,1000000,"))+"\n"
    elif fault == "solution":
        deck += "/SOLU\n"
    else:
        deck += "CE,1,0,1,UX,1\n"
    with pytest.raises(ValueError):
        grid.candidate_training_constraint_deck(deck, **dimensions)


@pytest.mark.parametrize("direction", range(6))
@pytest.mark.parametrize("sign", [-1, 1])
def test_native_analytic_fields_independently_match_audit_all_six(face_coordinates, direction, sign):
    xyz = face_coordinates
    x, y, z = (xyz[:, :, i] for i in range(3))
    X, Z = x-.003, z-.003
    native = np.zeros(6)
    native[direction] = sign*(.002 if direction < 3 else .7)
    ex, ez, gamma, kx, kz, twist = native
    u = np.empty_like(xyz)
    u[:, :, 0] = (ex+kx*y)*X+.5*(gamma+twist*y)*Z
    u[:, :, 1] = -.5*(kx*X**2+kz*Z**2+twist*X*Z)-.23*y
    u[:, :, 2] = (ez+kz*y)*Z+.5*(gamma+twist*y)*X
    centered = native.copy()
    centered[:3] += .00075*native[3:]
    audited = grid.audit_face_history(xyz, np.stack([np.zeros_like(u), u]), np.stack([np.zeros(6), centered]))
    assert max(audited["report"]["raw_group_spread_mm"]) < 1e-17
    broken = u.copy()
    corner = np.flatnonzero((x[1] == x.max()) & (z[1] == z.max()))[0]
    broken[1, corner, 2] += 5e-8
    with pytest.raises(ValueError, match="periodicity"):
        grid.audit_face_history(xyz, np.stack([np.zeros_like(u), broken]), np.stack([np.zeros(6), centered]))


@pytest.fixture
def face_coordinates():
    x = np.linspace(0., .006, 7)
    xz = np.asarray([(a, b) for a in x for b in x])
    return np.asarray([np.column_stack((xz[:, 0], np.full(len(xz), y), xz[:, 1]))
                       for y in (0., .0015)])


def face_translations(coordinates):
    n = coordinates.shape[1]
    return np.broadcast_to(np.eye(3) / np.sqrt(n), (n, 3, 3)).reshape(3*n, 3).copy()


def test_analytic_q6_traces_keep_native_axes_center_and_engineering_twist(face_coordinates):
    c = grid.face_kinematic_contract(face_coordinates)
    xyz = face_coordinates.reshape(-1, 3)
    X, Y, Z = (xyz-(xyz.min(0)+xyz.max(0))/2).T
    q = c["q6_traces"]
    assert c["q6_labels"] == grid.FACE_Q6_LABELS
    np.testing.assert_array_equal(q[:, 0, 0], X)
    np.testing.assert_array_equal(q[:, 2, 1], xyz[:, 2])
    np.testing.assert_array_equal(q[:, 0, 2], .5*Z)
    np.testing.assert_array_equal(q[:, 2, 2], .5*X)
    np.testing.assert_array_equal(q[:, 0, 5], .5*Y*Z)
    np.testing.assert_array_equal(q[:, 2, 5], .5*Y*X)
    np.testing.assert_array_equal(q[:, 1, 5], -.5*X*Z)
    np.testing.assert_array_equal(q[:, 1, 3], -.5*X**2)
    assert len(c["periodic_groups"]) == 2*(2*7-3)
    assert [len(v) for v in c["periodic_groups"]].count(4) == 2


@pytest.mark.parametrize("fault", ["order", "duplicate", "missing", "nonsquare", "nonuniform", "plane", "swapped", "nan"])
def test_face_kinematics_rejects_unregistered_or_incomplete_faces(face_coordinates, fault):
    c = face_coordinates.copy()
    if fault == "order":
        c[1] = c[1, ::-1]
    elif fault == "duplicate":
        c[:, 1] = c[:, 0]
    elif fault == "missing":
        c = c[:, :-1]
    elif fault == "nonsquare":
        c[:, :, 0] *= 2
    elif fault == "nonuniform":
        c[:, np.isclose(c[0, :, 0], .001), 0] += .0001
    elif fault == "plane":
        c[0, 1, 1] = .0001
    elif fault == "swapped":
        c = c[::-1]
    else:
        c[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        grid.face_kinematic_contract(c)


def test_common_face_reordering_preserves_physical_audit(face_coordinates):
    rng = np.random.default_rng(712)
    perm = rng.permutation(face_coordinates.shape[1])
    u = np.zeros((2, *face_coordinates.shape))
    u[1, 0, :, 1] = -.001
    u[1, 1, :, 1] = .001
    a = grid.audit_face_history(face_coordinates, u, np.zeros((2, 6)))
    b = grid.audit_face_history(face_coordinates[:, perm], u[:, :, perm], np.zeros((2, 6)))
    np.testing.assert_allclose(b["increment_periodic_mm"], a["increment_periodic_mm"][:, :, perm], atol=1e-18)
    np.testing.assert_allclose(a["fixed_scales_mm"], b["fixed_scales_mm"], atol=1e-18)


def test_periodic_rotation_correlated_motion_survives_gauge_and_fit(face_coordinates):
    c = grid.face_kinematic_contract(face_coordinates)
    n = c["node_count"]
    u = np.zeros((2, 2, n, 3))
    u[1, :, :, 1] = 1e-4*np.sin(2*np.pi*face_coordinates[:, :, 0]/.006)
    before = u.copy()
    a = grid.audit_face_history(face_coordinates, u, np.zeros((2, 6)))
    np.testing.assert_array_equal(u, before)
    np.testing.assert_allclose(a["increment_periodic_mm"][1], u[1], atol=1e-18)
    assert max(a["report"]["raw_group_spread_mm"]) < 1e-18
    result = grid.fit_face_basis(a, face_translations(face_coordinates))
    assert not result["report"]["face_pass"]
    assert result["report"]["periodic_pass"]
    # A full-six orthogonal projection manufactures an edge jump on this field.
    G = c["rigid_traces"].reshape(-1, 6)
    coeff, _ = grid.face_checked_lstsq(G, u[1].reshape(-1, 1))
    projected = (u[1].reshape(-1)-G @ coeff[:, 0]).reshape(1, 2*n, 3)
    assert grid.face_periodic_spread(projected, c)[0] > 1e-6


def test_edge_gauge_scales_and_fits_are_invariant_to_shared_rigid_motion(face_coordinates):
    c = grid.face_kinematic_contract(face_coordinates)
    rng = np.random.default_rng(831)
    q = rng.normal(size=(3, 6))*np.asarray([.001, .001, .001, 1., 1., 1.])
    u = np.einsum("ndq,sq->snd", c["q6_traces"], q).reshape(3, *face_coordinates.shape)
    u[:, 0, :, 1] -= 2e-5
    u[:, 1, :, 1] += 2e-5
    angles = rng.normal(size=(3, 6))*.001
    rigid = np.einsum("ndq,sq->snd", c["rigid_traces"], angles).reshape(u.shape)
    a = grid.audit_face_history(face_coordinates, u, q)
    b = grid.audit_face_history(face_coordinates, u+rigid, q)
    np.testing.assert_allclose(a["fixed_scales_mm"], b["fixed_scales_mm"], atol=1e-17)
    np.testing.assert_allclose(a["all_periodic_mm"], b["all_periodic_mm"], atol=1e-17)
    for audit in (a, b):
        fit = grid.fit_face_basis(audit, face_translations(face_coordinates))
        assert fit["report"]["face_pass"] and fit["report"]["periodic_pass"]
        assert fit["report"]["quotient_design"]["rank"] == 9
        assert fit["report"]["rotation_locked_fit"]["rank"] == 6
        np.testing.assert_allclose(fit["port_coefficients"][:, 0, :3]+fit["port_coefficients"][:, 1, :3], 0.)
    assert np.max(abs(a["initial_periodic_mm"])) == pytest.approx(2e-5)


def test_initial_nonperiodicity_cannot_hide_in_zero_increments(face_coordinates):
    u = np.zeros((2, *face_coordinates.shape))
    u[:, :, :, 0] = .001*face_coordinates[None, :, :, 0]
    assert np.max(abs(u[1]-u[0])) == 0
    with pytest.raises(ValueError, match="initial/increment/absolute"):
        grid.audit_face_history(face_coordinates, u, np.zeros((2, 6)))


def test_fixed_q6_is_not_refitted_to_hide_wrong_macro_strain(face_coordinates):
    c = grid.face_kinematic_contract(face_coordinates)
    q = np.zeros((2, 6))
    q[1, 0] = .001
    u = np.einsum("ndq,sq->snd", c["q6_traces"], q).reshape(2, *face_coordinates.shape)
    with pytest.raises(ValueError, match="periodicity"):
        grid.audit_face_history(face_coordinates, u, np.zeros((2, 6)))


def test_corner_audit_measures_full_group_spread_not_anchor_error(face_coordinates):
    c = grid.face_kinematic_contract(face_coordinates)
    v = np.zeros((1, 2*c["node_count"], 3))
    corners = next(group for group in c["periodic_groups"] if len(group) == 4)
    v[0, corners[1], 0], v[0, corners[2], 0] = 6e-11, -6e-11
    assert grid.face_periodic_spread(v, c)[0] == pytest.approx(1.2e-10)


@pytest.mark.parametrize("fault", ["direction", "normalization", "gram", "nonperiodic", "nonfinite"])
def test_face_basis_contract_fails_before_deleting_physical_modes(face_coordinates, fault):
    a = grid.audit_face_history(face_coordinates, np.zeros((2, *face_coordinates.shape)), np.zeros((2, 6)))
    B = face_translations(face_coordinates)
    if fault == "direction":
        B = B[:, [1, 0, 2]]
    elif fault == "normalization":
        B[:, 0] *= 2
    elif fault == "gram":
        B = np.column_stack((B, B[:, 0]))
    elif fault == "nonperiodic":
        v = np.zeros(face_coordinates.shape[1:])
        v[:, 1] = face_coordinates[0, :, 0]-.003
        v = v.reshape(-1)
        B = np.column_stack((B, v/np.linalg.norm(v)))
    else:
        B[0, 0] = np.nan
    with pytest.raises(ValueError):
        grid.fit_face_basis(a, B)


def test_face_zero_scale_keeps_absolute_tolerance_and_no_operator_admission(face_coordinates):
    a = grid.audit_face_history(face_coordinates, np.zeros((1, *face_coordinates.shape)), np.zeros((1, 6)))
    fit = grid.fit_face_basis(a, face_translations(face_coordinates))
    assert a["report"]["fixed_initial_scale_mm"] == 0
    assert a["report"]["periodic_limits_mm"] == [grid.FACE_PBC_ATOL_MM]*3
    assert fit["report"]["face_limits_mm"] == [grid.FACE_ATOL_MM]*3
    assert fit["report"]["face_pass"] and fit["report"]["periodic_pass"]
    assert not fit["report"]["thermal_geometry_validated"] and not fit["report"]["runtime_admitted"]
    assert not fit["report"]["actual_volume_lifts_checked"]


def test_face_lstsq_enforces_rank_with_scaled_units():
    matrix = np.diag([1e-15, 1e15, 1.])
    coefficients, audit = grid.face_checked_lstsq(matrix, matrix)
    np.testing.assert_allclose(coefficients, np.eye(3))
    assert audit["rank"] == 3 and audit["condition_number"] == 1
    with pytest.raises(ValueError, match="rank"):
        grid.face_checked_lstsq(np.ones((9, 2)), np.ones((9, 1)))
    with pytest.raises(ValueError, match="empty"):
        grid.face_checked_lstsq(np.zeros((9, 2)), np.ones((9, 1)))


def synthetic_face_seed(coordinates):
    contract = grid.face_kinematic_contract(coordinates)
    n = contract["node_count"]
    rng = np.random.default_rng(451)
    fields = rng.normal(size=(n, 3, 21))
    for group in contract["periodic_groups"]:
        if np.all(group < n):
            fields[group] = fields[group].mean(axis=0)
    T = face_translations(coordinates)
    fields = fields.reshape(3*n, 21)
    fields -= T@(T.T@fields)
    U, _ = np.linalg.qr(fields)
    return np.column_stack((T, U))


def test_face_seed_is_geometry_prior_with_fresh_coefficients_not_operator_interpolation(face_coordinates):
    source = synthetic_face_seed(face_coordinates)
    raw = source.copy()
    target = face_coordinates.copy()
    target[:, :, [0, 2]] *= 5
    seed = grid.transfer_face_seed(face_coordinates[0][:, (0, 2)], source, target)
    np.testing.assert_array_equal(source, raw)
    B = seed["modes"]
    np.testing.assert_allclose(raw-B@(B.T@raw), 0., atol=1e-14)
    np.testing.assert_allclose(B.T@B, np.eye(24), atol=1e-14)
    np.testing.assert_array_equal(B[:, :3], face_translations(target))
    report = seed["report"]
    assert report["coefficient_convention_changed"] and not report["old_lifts_or_coefficients_reusable"]
    assert not report["mechanics_operator_interpolation"] and not report["volume_lifts_qualified"]
    assert not report["runtime_admitted"] and not report["thermal_geometry_validated"]
    audit = grid.audit_face_history(target, np.zeros((1, *target.shape)), np.zeros((1, 6)))
    assert grid.fit_face_basis(audit, B)["report"]["periodic_pass"]


def test_face_seed_resampling_preserves_span_without_reusing_coefficient_semantics(face_coordinates):
    source = synthetic_face_seed(face_coordinates)
    x = np.linspace(0., .012, 9)
    xz = np.asarray([(a, b) for a in x for b in x])
    target = np.asarray([np.column_stack((xz[:, 0], np.full(len(xz), y), xz[:, 1])) for y in (0., .01)])
    seed = grid.transfer_face_seed(face_coordinates[0][:, (0, 2)], source, target)
    assert seed["modes"].shape == (3*81, 24)
    assert seed["report"]["raw_span_relative_error"] < 1e-12
    assert seed["report"]["target_nodes"] == 81


def test_face_seed_records_bounded_roundoff_correction(face_coordinates):
    source = synthetic_face_seed(face_coordinates)
    source[0, 3] += 1e-12
    raw = source.copy()
    result = grid.transfer_face_seed(face_coordinates[0][:, (0, 2)], source, face_coordinates)
    report = result["report"]
    assert 0 < report["periodic_correction_max"] < report["periodic_correction_limit"]
    assert max(report["source_periodic_spread"]) > 0
    np.testing.assert_array_equal(source, raw)


@pytest.mark.parametrize("fault", ["duplicate", "translation", "gram", "nonfinite", "rank", "seam"])
def test_face_seed_never_repairs_physical_defects_as_roundoff(face_coordinates, fault):
    source = synthetic_face_seed(face_coordinates)
    coords = face_coordinates[0][:, (0, 2)].copy()
    if fault == "duplicate":
        coords[1] = coords[0]
    elif fault == "translation":
        source[:, :3] = source[:, [1, 0, 2]]
    elif fault == "gram":
        source[:, 3] *= 2
    elif fault == "nonfinite":
        coords[0, 0] = np.nan
    elif fault == "rank":
        source = source[:, :23]
    else:
        source[0, 3] += 1e-5
        T = source[:, :3]
        U, _ = np.linalg.qr(source[:, 3:]-T@(T.T@source[:, 3:]))
        source[:, 3:] = U
    with pytest.raises(ValueError):
        grid.transfer_face_seed(coords, source, face_coordinates)


def closed_face_seed(coordinates):
    c = grid.face_kinematic_contract(coordinates)
    n = c["node_count"]
    side = int(np.sqrt(n))
    values = np.random.default_rng(917).normal(size=(n, 3, 32))
    for group in c["periodic_groups"]:
        if np.all(group < n):
            values[group] = values[group].mean(axis=0)
    F = values.copy(); F[:, 1] *= -1
    S = values.reshape(side, side, 3, -1).transpose(1, 0, 2, 3)[:, :, [2, 1, 0]].reshape(values.shape)
    FS = S.copy(); FS[:, 1] *= -1
    T = face_translations(coordinates)
    columns = [T]
    for (f, s), count in zip(((1, 1), (1, -1), (-1, 1), (-1, -1)), (6, 5, 5, 5), strict=True):
        sector = (.25*(values+f*F+s*S+f*s*FS)).reshape(3*n, -1)
        sector -= T@(T.T@sector)
        U, _, _ = np.linalg.svd(sector, full_matrices=False)
        columns.append(U[:, :count])
    return np.column_stack(columns)


def test_common_face_symmetry_audits_actual_fields_not_rounded_involutions(face_coordinates):
    c = grid.face_kinematic_contract(face_coordinates)
    B = closed_face_seed(face_coordinates)
    r = grid.face_symmetry_audit(c, B)
    assert r["report"]["reflection_closure_max"] < 1e-14
    assert r["report"]["swap_closure_max"] < 1e-14
    with pytest.raises(ValueError, match="symmetry"):
        grid.face_symmetry_audit(c, synthetic_face_seed(face_coordinates))


def test_periodic_quotient_preserves_corner_edge_and_interior_metric(face_coordinates):
    c = grid.face_kinematic_contract(face_coordinates)
    q = grid.face_periodic_quotient(c)
    C = grid.face_periodic_expand(q, np.eye(q["columns"]))
    np.testing.assert_allclose(C.T@C, np.eye(q["columns"]), atol=1e-14)
    assert q["report"]["multiplicity_counts"] == {"1": 25, "2": 10, "4": 1}
    rng = np.random.default_rng(538)
    Z = rng.normal(size=(q["columns"], 5))
    X = grid.face_periodic_expand(q, Z)
    np.testing.assert_allclose(grid.face_periodic_reduce(q, X), Z, atol=1e-14)
    np.testing.assert_allclose(X.T@X, Z.T@Z, atol=1e-12)
    n = c["node_count"]
    fields = X.T.reshape(5, n, 3)
    np.testing.assert_array_equal(grid.face_periodic_spread(np.concatenate((fields, fields), axis=1), c), 0.)
    raw = rng.normal(size=X.shape)
    projected = C@(C.T@raw)
    np.testing.assert_allclose(np.linalg.norm(raw)**2,
        np.linalg.norm(projected)**2+np.linalg.norm(raw-projected)**2, atol=1e-12)
    for mode in ("reflection", "swap"):
        transformed = grid.face_transform(c, raw, **{mode: True})
        np.testing.assert_allclose(C@(C.T@transformed),
            grid.face_transform(c, projected, **{mode: True}), atol=1e-14)


def test_periodic_svd_keeps_weak_modes_constrained_without_changing_seed_or_sources(face_coordinates):
    c = grid.face_kinematic_contract(face_coordinates)
    q = grid.face_periodic_quotient(c)
    B = closed_face_seed(face_coordinates)
    B[0, 3] += 2e-14
    Z = np.random.default_rng(178).normal(size=(q["columns"], 12))*np.geomspace(1e-4, 1e-10, 12)
    Y = grid.face_periodic_expand(q, Z)
    seed_bytes, source_bytes = B.tobytes(), Y.tobytes()
    r = grid.face_pod_catalog(c, B, Y, ranks=(24, 28, 48, 128))
    assert B.tobytes() == seed_bytes and Y.tobytes() == source_bytes
    np.testing.assert_array_equal(r["modes"][:, :24], B)
    n = c["node_count"]
    extras = r["modes"][:, 24:].T.reshape(-1, n, 3)
    np.testing.assert_array_equal(grid.face_periodic_spread(np.concatenate((extras, extras), axis=1), c), 0.)
    assert all(row["projection_pass"] for row in r["report"]["sector_projection_audits"])
    assert not r["report"]["previous_dense_svd_span_preservation_claimed"]


def test_rejected_sector_preserves_seed_and_nominal_block_positions(face_coordinates, monkeypatch):
    c = grid.face_kinematic_contract(face_coordinates)
    B = closed_face_seed(face_coordinates)
    q = grid.face_periodic_quotient(c)
    Y = grid.face_periodic_expand(q, np.random.default_rng(216).normal(size=(q["columns"], 9))*1e-4)
    original, calls = grid.face_periodic_reduce, []
    def injected(quotient, fields):
        reduced = original(quotient, fields)
        if not calls:
            reduced[0, 0] += 1e-7
        calls.append(True)
        return reduced
    monkeypatch.setattr(grid, "face_periodic_reduce", injected)
    r = grid.face_pod_catalog(c, B, Y, ranks=(24, 28, 32, 40, 48, 64, 128))
    assert r["candidates"][0]["state"] == "audited"
    np.testing.assert_array_equal(r["modes"][:, :24], B)
    blocks = r["report"]["canonical_blocks"]
    offset = 24
    for block in blocks:
        assert block["first_mode"] == offset
        offset += block["dimension"]
    assert offset == r["report"]["available_resolved_rank"]
    rejected = [b["first_mode"] for b in blocks if not b["construction_pass"]]
    assert rejected
    for row in r["candidates"]:
        if row["rank"] <= offset and any(position < row["rank"] for position in rejected):
            assert row["state"] == "construction_invalid"
            assert row["rejected_block_positions"] == [i for i in rejected if i < row["rank"]]


def test_canonical_block_depends_on_subspace_not_svd_rotation():
    rng = np.random.default_rng(91)
    U, _ = np.linalg.qr(rng.normal(size=(53, 7)))
    rotation, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    first, anchors = grid.canonical_face_block(U)
    second, other = grid.canonical_face_block(U@rotation)
    assert anchors == other
    np.testing.assert_allclose(first, second, atol=1e-14)
    np.testing.assert_allclose(first.T@first, np.eye(7), atol=1e-14)
    np.testing.assert_allclose(U-first@(first.T@U), 0., atol=1e-14)


def test_zero_pod_and_unavailable_high_rank_leave_seed_candidate_usable(face_coordinates):
    c = grid.face_kinematic_contract(face_coordinates)
    B = closed_face_seed(face_coordinates)
    result = grid.face_pod_catalog(c, B, np.zeros((len(B), 9)), ranks=(24, 28, 128))
    assert [r["state"] for r in result["candidates"]] == ["audited", "unavailable", "unavailable"]
    np.testing.assert_array_equal(result["modes"], B)
    assert result["report"]["available_resolved_rank"] == 24
    assert not result["report"]["family_rank_selected"]
    assert not result["report"]["runtime_admitted"]


def test_pod_catalog_prefix_is_stable_across_requested_maximum_rank(face_coordinates):
    c = grid.face_kinematic_contract(face_coordinates)
    B = closed_face_seed(face_coordinates)
    Y = np.random.default_rng(78).normal(size=(7, 7, 3, 9))*1e-4
    Y[-1] = Y[0]; Y[:, -1] = Y[:, 0]
    Y = Y.reshape(len(B), -1)
    a = grid.face_pod_catalog(c, B, Y, ranks=(24, 28, 32))
    b = grid.face_pod_catalog(c, B, Y, ranks=(24, 28, 32, 128))
    np.testing.assert_allclose(a["modes"], b["modes"][:, :a["modes"].shape[1]], atol=1e-14)
    assert a["candidates"] == b["candidates"][:3]
    for r in b["candidates"]:
        if r["state"] == "audited":
            assert r["symmetry"]["reflection_closure_max"] < 1e-10


def test_candidate_audit_failure_is_local_to_that_rank(face_coordinates, monkeypatch):
    c = grid.face_kinematic_contract(face_coordinates)
    B = closed_face_seed(face_coordinates)
    Y = np.random.default_rng(45).normal(size=(7, 7, 3, 12))*1e-4
    Y[-1] = Y[0]; Y[:, -1] = Y[:, 0]
    baseline = grid.face_pod_catalog(c, B, Y.reshape(len(B), -1), ranks=(24, 28, 32))
    original = grid.face_symmetry_audit
    def patched(contract, modes):
        if modes.shape[1] == 28:
            raise ValueError("synthetic higher-prefix audit failure")
        return original(contract, modes)
    monkeypatch.setattr(grid, "face_symmetry_audit", patched)
    r = grid.face_pod_catalog(c, B, Y.reshape(len(B), -1), ranks=(24, 28, 32))
    assert [v["state"] for v in r["candidates"]] == ["audited", "audit_invalid", "audited"]
    np.testing.assert_array_equal(r["modes"], baseline["modes"])
    assert r["candidates"][0] == baseline["candidates"][0]
    assert r["candidates"][2] == baseline["candidates"][2]


@pytest.mark.parametrize("ranks", [(28, 24), (24, 24), (True, 28), (24, 513), ()])
def test_pod_rank_policy_is_explicit_and_ascending(face_coordinates, ranks):
    with pytest.raises(ValueError):
        grid.face_pod_catalog(grid.face_kinematic_contract(face_coordinates), closed_face_seed(face_coordinates),
                              np.zeros((3*face_coordinates.shape[1], 1)), ranks=ranks)


@pytest.mark.parametrize("path", grid.TRAINING_PATHS + grid.TRAINING_HOLDOUTS)
def test_training_path_uses_six_coordinates_and_explicit_holdout_role(path):
    spec = grid.training_spec(3, path=path)
    rows = grid.training_rows(spec)
    grid.validate_training_rows(rows)
    assert spec["cell_um"] == 6 and spec["diameter_um"] == 1.5
    assert spec["initial_q6"] == [0.] * 6
    assert spec["role"] == ("held_out_validation" if path in grid.TRAINING_HOLDOUTS else "training")
    assert spec["reservation_gib"] == 52
    assert grid.RESERVATION_GIB == 64
    assert not spec["runtime_admitted"] and not spec["thermal_geometry_validated"]
    for row in rows:
        if path == "thermal_zero_macro" or row["state"] == "coolback_rt":
            assert all(row[f] == 0. for f in grid.TRAINING_FIELDS)


@pytest.mark.parametrize("direction,normal,gradient", [("kappa_x", "eps_x_at_y0", "strain_gradient_x_1_per_mm"),
    ("kappa_z", "eps_z_at_y0", "strain_gradient_z_1_per_mm"),
    ("kappa_xz", "gamma_xz_at_y0", "strain_gradient_xz_1_per_mm")])
def test_training_curvature_is_centered_in_mm_including_shear(direction, normal, gradient):
    spec = grid.training_spec(3, path=direction + "_plus")
    row = grid.training_rows(spec)[0]
    assert row[normal] == pytest.approx(-.001)
    assert row[normal] + .00075 * row[gradient] == pytest.approx(0.)


@pytest.mark.parametrize("kwargs", [dict(pitch=True), dict(pitch=11), dict(pitch=3, nx=47),
    dict(pitch=7, nx=48), dict(pitch=3, nx=999), dict(pitch=3, si_um=float("nan")),
    dict(pitch=3, path="zero_external_load"), dict(pitch=3, layer="stack")])
def test_training_rejects_invalid_geometry_and_ambiguous_zero_semantics(kwargs):
    with pytest.raises(ValueError):
        grid.training_spec(**kwargs)


@pytest.mark.parametrize("fault", ["duplicate", "missing", "nonfinite"])
def test_training_csv_requires_exact_unique_nine_finite_endpoints(fault):
    rows = grid.training_rows(grid.training_spec(3))
    if fault == "duplicate":
        rows[-1] = rows[0]
    elif fault == "missing":
        rows.pop()
    else:
        rows[-1][grid.TRAINING_FIELDS[-1]] = float("nan")
    with pytest.raises(ValueError):
        grid.validate_training_rows(rows)


def test_parser_adapter_preserves_original_factory_globals_and_argv(tmp_path):
    path = tmp_path / "native.py"
    path.write_text("def parse_args():\n    p = argparse.ArgumentParser()\n    p.add_argument('--mesh', type=int, default=48)\n    return p.parse_args()\n")
    original = lambda: "original"
    native = SimpleNamespace(__file__=str(path), argparse=argparse, parse_args=original)
    argv = list(sys.argv)
    parser = grid.isolated_native_parser(native)
    assert parser.parse_args([]).mesh == 48
    assert parser.parse_args(["--mesh", "60"]).mesh == 60
    assert native.parse_args is original and sys.argv == argv


@pytest.mark.parametrize("body", ["return p.parse_args(['untrusted'])", "return other.parse_args()",
    "return p", "if True:\n        return p.parse_args()\n    return p.parse_args()"])
def test_parser_adapter_fails_closed_on_drift(tmp_path, body):
    path = tmp_path / "native.py"
    path.write_text("def parse_args():\n    p = argparse.ArgumentParser()\n    " + body + "\n")
    with pytest.raises(ValueError, match="parser"):
        grid.isolated_native_parser(SimpleNamespace(__file__=str(path), argparse=argparse))


def training_trace():
    times = [300.]
    for cycle in range(3):
        start = 300. + 4900. * cycle
        for duration, count in ((650., 160), (3600., 40), (650., 160)):
            times.extend(start + duration * i / count for i in range(1, count + 1))
            start += duration
    return [dict(result_set=i + 1, time_s=t, eps_x0=0.) for i, t in enumerate(times)]


@pytest.mark.parametrize("fault", [None, "initial_one_second", "missing", "duplicate", "reordered", "nonfinite"])
def test_training_clock_is_actual_1081_consecutive_sets(fault):
    rows = training_trace()
    if fault == "initial_one_second":
        rows[0]["time_s"] = 1.
    elif fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[1] = rows[0]
    elif fault == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif fault == "nonfinite":
        rows[1]["eps_x0"] = float("nan")
    if fault:
        with pytest.raises(ValueError):
            grid.training_clock_rows(rows)
    else:
        times = grid.training_clock_rows(rows)
        assert len(times) == 1081 and times[0] == 300. and times[-1] == 15000.
        assert [times[i] for i in (360, 720, 1080)] == [5200., 10100., 15000.]


@pytest.mark.parametrize("free,expected", [(251.9, False), (252., True), (257.8, True)])
def test_new_bi_storage_reservation_does_not_weaken_si_gate(monkeypatch, free, expected):
    spec = grid.training_spec(3)
    monkeypatch.setattr(grid.builder, "resource_check", lambda _: dict(local_free_gib=free, own_ansys_processes=[]))
    monkeypatch.setattr(grid, "ansys_blockers", lambda: [])
    if expected:
        assert grid.training_resources(Path("owned"), spec)["local_free_gib"] == free
    else:
        with pytest.raises(RuntimeError, match="reservation"):
            grid.training_resources(Path("owned"), spec)
    assert grid.RESERVATION_GIB == 64


@pytest.fixture
def allocation_work(tmp_path, monkeypatch):
    work = tmp_path/"owned"
    work.mkdir()
    monkeypatch.setattr(grid.builder, "local_scratch", lambda value: Path(value))
    monkeypatch.setattr(grid.builder, "scratch_root", lambda _: tmp_path)
    monkeypatch.setattr(grid.builder, "physical_host", lambda: "mfd")
    return work


def test_local_inventory_counts_directories_and_explicit_copy_link_once(allocation_work):
    work = allocation_work
    directory = work/"extract"; directory.mkdir()
    staging = directory/"result.123.copying"; staging.write_bytes(b"a"*8192)
    target = directory/"result"; os.link(staging, target)
    with pytest.raises(ValueError, match="hard link"):
        grid.local_allocation_snapshot(work)
    report = grid.local_allocation_snapshot(work,
        allowed_link_pairs=(("extract/result.123.copying", "extract/result"),))
    files = [v for v in report["files"] if v["kind"] == "file"]
    assert len(files) == 1 and set(files[0]["names"]) == {"extract/result", "extract/result.123.copying"}
    assert report["logical_bytes"] == work.stat().st_size+directory.stat().st_size+8192
    assert report["allocated_bytes"] == sum(path.stat().st_blocks*512 for path in (work, directory, staging))
    assert not report["launch_authorized"] and not report["process_ownership_authority_inferred"]


@pytest.mark.parametrize("fault", ["symlink", "snapshot", "special", "external_link", "depth", "entries", "owner", "root_replaced"])
def test_local_inventory_rejects_incomplete_or_foreign_accounting(allocation_work, monkeypatch, fault):
    work = allocation_work
    args = {}
    if fault == "symlink":
        (work/"link").symlink_to(work.parent)
    elif fault == "snapshot":
        (work/"#snapshot").mkdir()
    elif fault == "special":
        os.mkfifo(work/"fifo")
    elif fault == "external_link":
        (work/"file").write_bytes(b"foo"); os.link(work/"file", work.parent/"outside")
    elif fault == "depth":
        (work/"nested").mkdir(); args["depth_limit"] = 0
    elif fault == "entries":
        for i in range(3): (work/str(i)).write_bytes(b"a")
        args["entry_limit"] = 2
    elif fault == "owner":
        monkeypatch.setattr(grid.os, "getuid", lambda: work.stat().st_uid+1)
    else:
        original = grid.os.open
        def replaced(path, flags, **kwargs):
            if Path(path) == work:
                work.rename(work.parent/"old");work.mkdir()
            return original(path, flags, **kwargs)
        monkeypatch.setattr(grid.os, "open", replaced)
    with pytest.raises(ValueError):
        grid.local_allocation_snapshot(work, **args)


def test_local_inventory_counts_sparse_logical_size_and_open_deleted_inode(allocation_work):
    work = allocation_work
    path = work/"sparse"
    with path.open("wb") as stream: stream.truncate(1024**2)
    report = grid.local_allocation_snapshot(work)
    assert report["logical_bytes"] > report["allocated_bytes"]
    identity = grid.process_identity(os.getpid())
    assert identity
    with path.open("rb") as first, path.open("rb") as second:
        path.unlink()
        report = grid.local_allocation_snapshot(work, identities=(identity,))
        assert report["open_deleted_inodes"] == (1 if os.fstat(first.fileno()).st_nlink == 0 else 0)
        assert report["logical_bytes"] >= 1024**2
        assert len([v for v in report["files"] if v["kind"] == "file"]) == 1
        assert first.read(1) == second.read(1)


def test_local_inventory_deduplicates_deleted_descriptors_and_visible_inode(allocation_work, monkeypatch):
    work = allocation_work
    path = work/"result"; path.write_bytes(b"a"*8192)
    identity = grid.process_identity(os.getpid())
    scan, readlink = grid.os.scandir, grid.os.readlink
    with path.open("rb") as first, path.open("rb") as second:
        descriptor_paths = {f"/proc/{os.getpid()}/fd/{f.fileno()}" for f in (first, second)}
        class DeletedEntry:
            def __init__(self, entry): self.path = entry.path
            def stat(self, **kwargs):
                info = os.stat(self.path)
                return SimpleNamespace(**{name: getattr(info, name) for name in (
                    "st_uid", "st_dev", "st_ino", "st_mode", "st_size", "st_blocks")}, st_nlink=0)
        @contextmanager
        def scanned(value):
            with scan(value) as entries:
                yield (DeletedEntry(entry) if entry.path in descriptor_paths else entry for entry in entries)
        monkeypatch.setattr(grid.os, "scandir", scanned)
        monkeypatch.setattr(grid.os, "readlink", lambda value: str(path)+" (deleted)" if value in descriptor_paths else readlink(value))
        report = grid.local_allocation_snapshot(work, identities=(identity,))
        assert report["open_deleted_inodes"] == 1
        assert report["logical_bytes"] == work.stat().st_size+8192
        assert len([v for v in report["files"] if v["kind"] == "file"]) == 1


def test_local_inventory_rejects_reused_descriptor_identity(allocation_work, monkeypatch):
    identity = dict(pid=os.getpid(), start_ticks="wrong")
    with pytest.raises(ValueError, match="identity changed"):
        grid.local_allocation_snapshot(allocation_work, identities=(identity,))


def test_allocation_timeout_is_not_an_empty_inventory(allocation_work, monkeypatch):
    clock = iter((0., 3.))
    monkeypatch.setattr(grid.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError):
        grid.local_allocation_snapshot(allocation_work, seconds=2.)


@pytest.mark.parametrize("bad", [True, -1, float("nan")])
def test_allocation_sampler_rejects_invalid_counters(bad):
    sampler = grid.TrainingAllocationSampler(lambda: dict(logical_bytes=bad, allocated_bytes=0, free_bytes=999),
        period=.01, stale_seconds=.5, usage_bytes=10, minimum_free_bytes=100)
    try:
        with pytest.raises(RuntimeError, match="invalid byte"):
            sampler.start()
        assert not sampler.report()["launch_authorized"]
    finally:
        sampler.close()


def test_allocation_sampler_latches_breach_and_keeps_collecting_during_cleanup():
    values = dict(logical_bytes=0, allocated_bytes=0, free_bytes=999)
    sampler = grid.TrainingAllocationSampler(lambda: dict(values), period=.01, stale_seconds=.5,
        usage_bytes=10, minimum_free_bytes=100)
    try:
        sampler.start()
        values["allocated_bytes"] = 11
        limit = time.monotonic()+1.
        while sampler.report()["error"] is None and time.monotonic() < limit: time.sleep(.005)
        with pytest.raises(RuntimeError, match="budget"): sampler.check()
        previous = sampler.report()["samples"]
        values["allocated_bytes"] = 0
        while sampler.report()["samples"] <= previous and time.monotonic() < limit: time.sleep(.005)
        assert sampler.report()["samples"] > previous
        with pytest.raises(RuntimeError, match="budget"): sampler.check()
        assert not sampler.report()["continuous_peak_bound_claimed"]
    finally:
        sampler.close()


def test_allocation_sampler_rejects_stalled_completed_evidence_then_cannot_clear_failure():
    entered, release = threading.Event(), threading.Event()
    calls = []
    def sample():
        if calls:
            entered.set(); release.wait(2.)
        calls.append(True)
        return dict(logical_bytes=0, allocated_bytes=0, free_bytes=999)
    sampler = grid.TrainingAllocationSampler(sample, period=.01, stale_seconds=.05,
        usage_bytes=10, minimum_free_bytes=100)
    try:
        sampler.start()
        assert entered.wait(1.)
        time.sleep(.06)
        with pytest.raises(RuntimeError, match="stale"): sampler.check()
        release.set()
        time.sleep(.02)
        with pytest.raises(RuntimeError, match="stale"): sampler.check()
    finally:
        release.set();sampler.close()


def test_allocation_sampler_thread_exit_blocks_progression():
    sampler = grid.TrainingAllocationSampler(lambda: dict(logical_bytes=0, allocated_bytes=0, free_bytes=999),
        period=.01, stale_seconds=.5, usage_bytes=10, minimum_free_bytes=100)
    try:
        sampler.start()
        sampler._stop.set();sampler._thread.join(1.)
        with pytest.raises(RuntimeError, match="not live"): sampler.check()
    finally:
        sampler.close()


@pytest.mark.parametrize("clock", [1., 300., 2.])
@pytest.mark.parametrize("fault", [None, "rst_clock", "q6_nonzero"])
def test_training_initial_clock_correction_requires_actual_rst_and_zero_q6(clock, fault):
    keys = ("eps_x0", "eps_z0", "kappa_x_1_per_mm", "kappa_z_1_per_mm", "gamma_xz0", "kappa_xz_1_per_mm")
    initial = dict.fromkeys(keys, 0.) | dict(time_s=clock)
    first = dict.fromkeys(keys, 0.) | dict(time_s=300.)
    if fault == "q6_nonzero":
        initial["gamma_xz0"] = .001
    t = 1. if fault == "rst_clock" else 300.
    if clock == 2. or fault:
        with pytest.raises(ValueError):
            grid.training_initial_clock([initial], t, first)
    else:
        audit = grid.training_initial_clock([initial], t, first)
        assert audit["corrected"]["time_s"] == 300.
        assert audit["raw_timestamp_rejected"] == (clock == 1.)
        assert initial["time_s"] == clock


def test_training_zero_macro_trace_rejects_missing_shear_and_actual_nonzero_response():
    rows = training_trace()
    for row in rows:
        row.update(eps_z0=0., kappa_x_1_per_mm=0., kappa_z_1_per_mm=0., gamma_xz0=0., kappa_xz_1_per_mm=0.)
    spec = grid.training_spec(3)
    grid.validate_training_macro(rows, spec)
    rows[100]["gamma_xz0"] = .001
    with pytest.raises(ValueError, match="trajectory"):
        grid.validate_training_macro(rows, spec)


def test_training_deck_audits_cycles_creep_ramps_and_retention():
    records = [dict(time_s=300., substeps=1)] + [dict(time_s=300. + 4900. * c + dt, substeps=n)
        for c in range(3) for dt, n in ((650., 160), (4250., 40), (4900., 160))]
    deck = "TANNEAL = 3.500000000000e+02\nOUTRES,ALL,LAST\nOUTRES,NSOL,ALL\n"
    for rate in ["OFF"] + ["OFF", "ON", "OFF"] * 3:
        deck += f"ALLSEL,ALL\nKBC,0\nBF,ALL,TEMP,TROOM\nRATE,{rate}\nSOLVE\n"
    deck += "/EXIT,NOSAVE\n"
    assert grid.audit_training_deck(deck, records)[-1]["stage"] == "cool_C3"
    for bad in (deck.replace("KBC,0", "KBC,1"), deck.replace("RATE,ON", "RATE,OFF"),
                deck.replace("OUTRES,NSOL,ALL", "OUTRES,NSOL,LAST")):
        with pytest.raises(ValueError):
            grid.audit_training_deck(bad, records)


@pytest.fixture
def sealed_training(tmp_path):
    spec = grid.training_spec(3, path="kappa_xz_plus", nx=12)
    write_json = lambda name, value: (tmp_path / name).write_text(json.dumps(value))
    digest = lambda name: grid.builder.digest(tmp_path / name)
    def write_csv(name, rows):
        with (tmp_path / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_csv("PRESCRIBED_PATH.csv", grid.training_rows(spec))
    endpoint_times = [300.] + [300. + 4900. * c + dt for c in range(3) for dt in (650., 4250., 4900.)]
    endpoints = np.array([[0.] * 6] + [[r[k] for k in grid.TRAINING_FIELDS] for r in grid.training_rows(spec)])
    times = np.asarray(grid.training_clock()[::10])
    macro = [dict(result_set=i, time_s=t, **dict(zip(grid.TRAINING_MACRO_FIELDS,
             [float(np.interp(t, endpoint_times, endpoints[:, j])) for j in range(6)], strict=True)))
             for i, t in zip(range(1, 1082, 10), times, strict=True)]
    write_csv("SNAPSHOT_MACRO.csv", macro)
    xyz = np.array(np.meshgrid(np.linspace(0, .006, 13), np.linspace(0, .0015, 7),
                              np.linspace(0, .006, 13), indexing="ij")).reshape(3, -1).T.copy()
    arrays = dict(node_ids=np.arange(1, len(xyz) + 1, dtype=np.int64), coordinates_mm=xyz,
                  times_s=times, result_sets=np.arange(1, 1082, 10, dtype=np.int64),
                  displacements_mm=np.zeros((109, len(xyz), 3)))
    stages = [dict(time_s=300., substeps=1)] + [dict(time_s=300. + 4900. * c + dt, substeps=n)
              for c in range(3) for dt, n in ((650., 160), (4250., 40), (4900., 160))]
    deck = "TANNEAL = 3.500000000000e+02\nOUTRES,ALL,LAST\nOUTRES,NSOL,ALL\n"
    for rate in ["OFF"] + ["OFF", "ON", "OFF"] * 3:
        deck += f"ALLSEL,ALL\nKBC,0\nBF,ALL,TEMP,TROOM\nRATE,{rate}\nSOLVE\n"
    deck += "/EXIT,NOSAVE\n"
    (tmp_path / "SOLVE.dat").write_text(deck)
    write_json("PREFLIGHT.json", dict(geometry=dict(nodes=len(xyz), elements=12**2 * 6,
        composition=[dict(layer="bi", absolute_fraction_error=.005)]), stages=stages,
        deck_sha256=digest("SOLVE.dat"), protected_validation_targets_read=[],
        thermal_geometry_validated=False, runtime_admitted=False))
    write_json("MAPDL_COMPLETION.json", dict(rc=0, license_error=False))
    raw = {"/retired/not_accessible/job.rdb": "a" * 64, "/retired/not_accessible/job.rst": "b" * 64}
    write_json("RAW_SOLVER_SEAL.json", dict(files_sha256=raw))
    initial = dict.fromkeys(grid.TRAINING_MACRO_FIELDS, 0.) | dict(time_s=1.)
    write_json("INITIAL_CLOCK_AUDIT.json", dict(**grid.training_initial_clock([initial], 300., macro[0]),
        raw_solver_hashes=raw, raw_initial_csv_sha256="c" * 64, endpoint_export_sha256="d" * 64))
    files = {"/retired/" + name: h for name, h in grid.TRAINING_CARDS.items()}
    files.update({"/retired/operator_grid.py": "e" * 64, "/retired/common/ansys_utils.py": grid.COMMON_HELPER_SHA})
    write_json("SOURCE_PINS.json", dict(schema="yap.operator_grid.sources.v1", files_sha256=files,
        common_utils="/retired/common/ansys_utils.py"))
    plan = dict(schema="yap.operator_grid.training.v1", geometry=spec, automatic_retry=False,
        scratch_retained=True, protected_validation_targets_read=[], work="/retired/work",
        host="former_host", expires_epoch=0., source_pins_sha256=digest("SOURCE_PINS.json"),
        review_sha256="e" * 64, path_sha256=digest("PRESCRIBED_PATH.csv"))
    write_json("TRAINING_PLAN.json", plan)
    def reseal():
        np.savez_compressed(tmp_path / "VOLUME_DISPLACEMENT_SNAPSHOTS.npz", **arrays, layer=np.asarray(spec["layer"]))
        receipt = dict(state="TRAINING_EXTRACTED_NOT_OPERATOR", geometry=spec,
            files_sha256={name: digest(name) for name in grid.TRAINING_ARTIFACTS},
            raw_extraction_sha256={name: "f" * 64 for name in ("volume_nodes.csv", "volume_displacements.csv")},
            actual_sets=1081, selected_snapshots=109, snapshot_shape=list(arrays["displacements_mm"].shape),
            arrays={k: dict(dtype=str(v.dtype), shape=list(v.shape),
                sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()) for k, v in arrays.items()},
            source_plan_sha256=digest("TRAINING_PLAN.json"), thermal_geometry_validated=False, runtime_admitted=False)
        write_json("TRAINING_COMPLETE.json", receipt)
        return digest("TRAINING_COMPLETE.json")
    return SimpleNamespace(directory=tmp_path, arrays=arrays, spec=spec, macro=macro,
                           write_json=write_json, write_csv=write_csv, reseal=reseal, digest=digest)


def test_completed_training_relocates_and_expires_without_reopening_raw_solver_or_sources(sealed_training, monkeypatch):
    f = sealed_training
    monkeypatch.setattr(grid, "verify_pins", lambda *a: pytest.fail("do not reopen historical source paths"))
    data = grid.load_training_evidence(f.directory, f.reseal())
    assert data["q6_labels"] == ["eps_x", "eps_y", "gamma_xy", "kappa_x", "kappa_y", "kappa_xy"]
    assert np.max(np.abs(data["q6_centered"][:, 2])) < 1e-15
    assert np.max(data["q6_centered"][:, 5]) == pytest.approx(f.spec["curvature_amplitude_1_per_mm"])
    assert all(len(face) == 169 for face in data["face_indices"])
    assert not data["thermal_geometry_validated"] and not data["runtime_admitted"]


def numerical_face_evidence(fixture):
    evidence = grid.load_training_evidence(fixture.directory, fixture.reseal())
    ids = np.stack(evidence["face_indices"])
    xyz = evidence["arrays"]["coordinates_mm"][ids]
    contract = grid.face_kinematic_contract(xyz)
    values = np.einsum("ndq,sq->snd", contract["q6_traces"], evidence["q6_centered"])
    evidence["arrays"]["displacements_mm"][:, ids] = values.reshape(109, *xyz.shape)
    return evidence


def test_training_pool_keeps_source_gates_and_no_family_claim(sealed_training):
    evidence = numerical_face_evidence(sealed_training)
    original = evidence["arrays"]["displacements_mm"].copy()
    pool = grid.pack_face_training([evidence])
    assert pool["columns_mm"].shape == (3*169, 2*(1+109*2))
    np.testing.assert_array_equal(evidence["arrays"]["displacements_mm"], original)
    assert not pool["report"]["training_path_coverage_complete"]
    assert not pool["report"]["family_coverage_complete"]
    assert not pool["report"]["final_holdouts_read"]
    assert pool["report"]["source_ledger"][0]["raw_evidence_used_for_evaluation"]
    c = pool["histories"][0]["contract"]
    B = closed_face_seed(c["coordinates_mm"])
    catalog = grid.face_pod_catalog(c, B, pool["columns_mm"], ranks=(24, 128))
    evaluation = grid.evaluate_face_catalog(catalog, pool)
    assert evaluation["prototype_minimum_rank"] == 24
    assert not evaluation["family_rank_selected"]
    assert not evaluation["runtime_admitted"] and not evaluation["thermal_geometry_validated"]


@pytest.mark.parametrize("fault", ["holdout", "role", "labels", "twist", "claimed_validation", "registration", "duplicate"])
def test_training_pool_rejects_holdouts_bad_contracts_and_duplicate_sources(sealed_training, fault):
    evidence = numerical_face_evidence(sealed_training)
    if fault == "holdout":
        evidence["geometry"] = grid.training_spec(3, path=grid.TRAINING_HOLDOUTS[0], nx=12)
    elif fault == "role":
        evidence["geometry"]["role"] = "development"
    elif fault == "labels":
        evidence["q6_labels"] = list(reversed(evidence["q6_labels"]))
    elif fault == "twist":
        evidence["engineering_shear_and_twist"] = False
    elif fault == "claimed_validation":
        evidence["thermal_geometry_validated"] = True
    elif fault == "registration":
        evidence["face_indices"] = tuple(face[::-1] for face in evidence["face_indices"])
    with pytest.raises(ValueError):
        grid.pack_face_training([evidence]*2 if fault == "duplicate" else [evidence])


def test_training_pool_does_not_silently_rescale_other_common_cells(sealed_training):
    import copy
    evidence = numerical_face_evidence(sealed_training)
    other = copy.deepcopy(evidence)
    other["geometry"] = grid.training_spec(4, path=other["geometry"]["path"], nx=12)
    other["arrays"]["coordinates_mm"][:, [0, 2]] *= 2
    with pytest.raises(ValueError, match="identical physical"):
        grid.pack_face_training([evidence, other])


@pytest.mark.parametrize("fault", ["wrong_receipt_hash", "changed_file", "duplicate_set", "fractional_set",
    "time", "nonfinite_macro", "trajectory", "node_id", "duplicate_coordinate", "coordinate_extent",
    "nonfinite_displacement", "float32", "solver", "deck", "raw_clock", "card", "plan"])
def test_completed_training_fails_closed_even_for_resealed_inconsistent_evidence(sealed_training, fault):
    f = sealed_training
    if fault in ("duplicate_set", "fractional_set", "time", "nonfinite_macro", "trajectory"):
        key, value = {"duplicate_set": ("result_set", 1), "fractional_set": ("result_set", 11.5),
            "time": ("time_s", 1.), "nonfinite_macro": ("eps_x0", float("nan")),
            "trajectory": ("gamma_xz0", 1.)}[fault]
        f.macro[1][key] = value
        f.write_csv("SNAPSHOT_MACRO.csv", f.macro)
    elif fault == "node_id":
        f.arrays["node_ids"][1] = f.arrays["node_ids"][0]
    elif fault == "duplicate_coordinate":
        f.arrays["coordinates_mm"][1] = f.arrays["coordinates_mm"][0]
    elif fault == "coordinate_extent":
        f.arrays["coordinates_mm"][1, 2] += .0001
    elif fault == "nonfinite_displacement":
        f.arrays["displacements_mm"][1, 1, 1] = float("nan")
    elif fault == "float32":
        f.arrays["displacements_mm"] = f.arrays["displacements_mm"].astype(np.float32)
    elif fault == "solver":
        f.write_json("MAPDL_COMPLETION.json", dict(rc=1, license_error=False))
    elif fault == "deck":
        p = f.directory / "SOLVE.dat"
        p.write_text(p.read_text().replace("KBC,0", "KBC,1"))
    elif fault == "raw_clock":
        p = f.directory / "INITIAL_CLOCK_AUDIT.json"
        v = json.loads(p.read_text()); v["corrected"]["time_s"] = 1.; f.write_json(p.name, v)
    elif fault in ("card", "plan"):
        p = f.directory / ("SOURCE_PINS.json" if fault == "card" else "TRAINING_PLAN.json")
        v = json.loads(p.read_text())
        if fault == "card":
            v["files_sha256"]["/retired/" + next(iter(grid.TRAINING_CARDS))] = "0" * 64
        else:
            v["geometry"]["role"] = "training_tuned_holdout"
        f.write_json(p.name, v)
    sha = f.reseal()
    if fault == "changed_file":
        with (f.directory / "SNAPSHOT_MACRO.csv").open("a") as stream:
            stream.write("changed")
    with pytest.raises(ValueError):
        grid.load_training_evidence(f.directory, "0" * 64 if fault == "wrong_receipt_hash" else sha)


def test_snapshot_trace_checks_independent_clock_not_two_mutually_consistent_arrays(sealed_training):
    f = sealed_training
    f.arrays["times_s"][1] = f.macro[1]["time_s"] = 17.
    f.write_csv("SNAPSHOT_MACRO.csv", f.macro)
    with pytest.raises(ValueError, match="continuous clock"):
        grid.load_training_evidence(f.directory, f.reseal())


def test_npz_header_rejected_before_numpy_allocation(sealed_training, monkeypatch):
    f = sealed_training
    f.reseal()
    np.savez_compressed(f.directory / "VOLUME_DISPLACEMENT_SNAPSHOTS.npz", **f.arrays,
                        layer=np.asarray("bi"), unexpected=np.ones(3))
    receipt = json.loads((f.directory / "TRAINING_COMPLETE.json").read_text())
    receipt["files_sha256"]["VOLUME_DISPLACEMENT_SNAPSHOTS.npz"] = f.digest("VOLUME_DISPLACEMENT_SNAPSHOTS.npz")
    f.write_json("TRAINING_COMPLETE.json", receipt)
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("invalid headers cannot allocate arrays"))
    with pytest.raises(ValueError, match="NPZ members"):
        grid.load_training_evidence(f.directory, f.digest("TRAINING_COMPLETE.json"))


def extraction_deck_fixture():
    lines = ["/BATCH", "FINISH", "/CLEAR,NOSTART", "RESUME,yap_training_c3,rdb", "/POST1", "FILE,yap_training_c3,rst",
             "SET,,,,,,,1"]
    lines += [f"SET,,,,,,,{s}" for s in range(1, 1082, 10)]
    lines += ["FINISH", "/EXIT,NOSAVE"]
    return grid.recovery_extraction_deck(SimpleNamespace(apdl_deck=lambda *a: "\n".join(lines)))


@pytest.mark.parametrize("fault", [None, "header", "missing", "extra", "width", "nonfinite", "fractional", "time", "q6_swap"])
def test_native_endpoint_normalization_is_strict_and_preserves_truncated_raw_header(tmp_path, fault):
    spec = grid.training_spec(3, path="kappa_xz_plus")
    stages = [dict(load_step=1, time_s=300., stage="initial_rt")] + [
        dict(load_step=2 + 3 * c + i, time_s=300. + 4900. * c + dt,
             stage=f"{phase}_C{c+1}") for c in range(3) for i, (dt, phase) in enumerate(((650., "heat"), (4250., "hot"), (4900., "cool")))]
    native_order = ("eps_x_at_y0", "strain_gradient_x_1_per_mm", "eps_z_at_y0", "strain_gradient_z_1_per_mm",
                    "gamma_xz_at_y0", "strain_gradient_xz_1_per_mm")
    q = [[0.] * 6] + [[r[key] for key in native_order] for r in grid.training_rows(spec)]
    rows = [[stage["load_step"], stage["time_s"], *values] for stage, values in zip(stages, q, strict=True)]
    if fault == "missing": rows.pop()
    if fault == "extra": rows.append(rows[-1])
    if fault == "width": rows[1].append(0.)
    if fault == "nonfinite": rows[1][2] = float("nan")
    if fault == "fractional": rows[1][0] = 2.5
    if fault == "time": rows[0][1] = 1.
    if fault == "q6_swap": rows[1][6], rows[1][7] = rows[1][7], rows[1][6]
    p = tmp_path / "raw.csv"
    with p.open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["wrong" if fault == "header" else "load_ste"]); writer.writerows(rows)
    raw_hash = grid.builder.digest(p)
    if fault:
        with pytest.raises(ValueError): grid.training_endpoints(p, stages, spec)
    else:
        normalized = grid.training_endpoints(p, stages, spec)
        assert len(normalized) == 10 and normalized[0]["time_s"] == 300.
    assert grid.builder.digest(p) == raw_hash


@pytest.mark.parametrize("line", ["SOLVE", "/INPUT,foreign", "*USE,foreign", "/SYS,rm anything",
    "/CWD,/source", "SAVE", "/DELETE,old", "RESUME,/source/job,rdb",
    "FILE,/source/job,rst", "*CFOPEN,/source/result,csv", "*GET,X,NODE,1,U,X$SOLVE"])
def test_recovery_deck_rejects_solution_restart_side_effects_and_source_paths(line):
    deck = extraction_deck_fixture()
    assert grid.audit_recovery_extraction_deck(deck) == deck
    with pytest.raises(ValueError): grid.audit_recovery_extraction_deck(deck.replace("FINISH", line, 1))


def test_recovery_deck_requires_actual_result_count_proof_and_exact_set_schedule():
    deck = extraction_deck_fixture()
    for bad in (deck.replace("*GET,YAPSETS,ACTIVE,0,SET,NSET", "*GET,YAPSETS,ACTIVE,0,SET,SBST"),
                deck.replace("SET,,,,,,,11", "SET,,,,,,,12"), deck.replace("/EXIT,NOSAVE", "/EXIT,SAVE")):
        with pytest.raises(ValueError): grid.audit_recovery_extraction_deck(bad)


@pytest.mark.parametrize("fault", [None, "live_worker", "orphan", "pid_reuse", "missing_scope", "scope_binding", "census"])
def test_recovery_source_exit_checks_all_retained_sessions_not_only_empty_live_ids(sealed_training, monkeypatch, fault):
    f = sealed_training
    p = f.directory / "TRAINING_PLAN.json"; plan = json.loads(p.read_text()); plan["work"] = str(f.directory)
    f.write_json(p.name, plan)
    plan_sha = f.digest(p.name)
    identity = dict(pid=65, start_ticks="800", pgid=65, sid=65, command=["old-worker"])
    f.write_json("LAUNCH.json", dict(worker_pid=65, worker_identity=identity, plan_sha256=plan_sha,
        source_pins_sha256=plan["source_pins_sha256"]))
    f.write_json("STATUS.json", dict(state="STOP_REQUIRES_REVIEW", worker_pid=65))
    scope = dict(schema="yap.operator_grid.process_scope.v1", worker_identity=identity,
        launch_sha256=f.digest("LAUNCH.json"), groups=[66, 67], sessions=[66, 67], live_identities=[])
    if fault == "scope_binding": scope["launch_sha256"] = "0" * 64
    f.write_json("PROCESS_SCOPE.json", scope)
    if fault == "missing_scope": (f.directory / "PROCESS_SCOPE.json").unlink()
    monkeypatch.setattr(grid.builder, "physical_host", lambda: "former_host")
    monkeypatch.setattr(grid, "process_identity", lambda *a: identity if fault in ("live_worker", "pid_reuse") else None)
    def census():
        if fault == "census": raise ValueError("census failed")
        return [dict(pid=99, ppid=1, pgid=67, sid=67, state="S", comm="not_named_ansys")] if fault == "orphan" else []
    monkeypatch.setattr(grid, "own_process_census", census)
    if fault:
        with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
            grid.training_source_exit(f.directory, plan_sha, f.digest("STATUS.json"))
    else:
        result = grid.training_source_exit(f.directory, plan_sha, f.digest("STATUS.json"))
        assert result["groups"] == [65, 66, 67]


def test_descriptor_seal_refuses_symlinks_and_detects_replacement(tmp_path):
    p = tmp_path / "original"; p.write_bytes(b"retained")
    before = grid.training_input_seal(p, 100)
    assert before["sha256"] == hashlib.sha256(b"retained").hexdigest()
    other = tmp_path / "other"; other.write_bytes(b"retained"); other.replace(p)
    assert grid.training_input_seal(p, 100) != before
    alias = tmp_path / "alias"; alias.symlink_to(p)
    with pytest.raises(ValueError): grid.training_input_seal(alias, 100)
    with pytest.raises(ValueError): grid.training_input_seal(p, 3)


def test_recovery_tag_cannot_silently_fall_back_to_ordinary_training_verification(sealed_training):
    f = sealed_training; f.reseal()
    receipt = json.loads((f.directory / "TRAINING_COMPLETE.json").read_text())
    receipt["recovery_plan_sha256"] = "0" * 64
    f.write_json("TRAINING_COMPLETE.json", receipt)
    with pytest.raises(ValueError, match="mandatory provenance"):
        grid.load_training_evidence(f.directory, f.digest("TRAINING_COMPLETE.json"))


@pytest.mark.parametrize("expiry", [0., float("nan"), float("inf")])
def test_recovery_expired_or_nonfinite_authority_never_extends_itself(expiry):
    with pytest.raises(TimeoutError): grid.recovery_deadline(dict(expires_epoch=expiry))


@pytest.fixture
def recovery_archive(sealed_training):
    f = sealed_training
    plan = json.loads((f.directory / "TRAINING_PLAN.json").read_text())
    plan["source"] = "/native_tree/campaign"
    f.write_json("TRAINING_PLAN.json", plan)
    preflight = json.loads((f.directory / "PREFLIGHT.json").read_text())
    grid.audit_training_deck((f.directory / "SOLVE.dat").read_text(), preflight["stages"])
    for i, stage in enumerate(preflight["stages"], 1): stage["load_step"] = i
    f.write_json("PREFLIGHT.json", preflight)
    f.write_json("MAPDL_COMPLETION.json", dict(rc=0, subprocess_rc=0, ansys_error_count=0, license_error=False,
        timeout=False, lock_conflict=False))
    identity = dict(pid=65, start_ticks="800", pgid=65, sid=65, command=["historical-worker"])
    f.write_json("ORIGINAL_LAUNCH.json", dict(worker_pid=65, worker_identity=identity,
        plan_sha256=f.digest("TRAINING_PLAN.json"), source_pins_sha256=plan["source_pins_sha256"]))
    f.write_json("ORIGINAL_STATUS.json", dict(state="STOP_REQUIRES_REVIEW", worker_pid=65))
    f.write_json("ORIGINAL_PROCESS_SCOPE.json", dict(schema="yap.operator_grid.process_scope.v1",
        worker_identity=identity, launch_sha256=f.digest("ORIGINAL_LAUNCH.json"), groups=[66], sessions=[66], live_identities=[]))
    proof = dict(launch_sha256=f.digest("ORIGINAL_LAUNCH.json"), scope_sha256=f.digest("ORIGINAL_PROCESS_SCOPE.json"),
                 groups=[65, 66], sessions=[65, 66])
    raw_order = ("eps_x_at_y0", "strain_gradient_x_1_per_mm", "eps_z_at_y0", "strain_gradient_z_1_per_mm",
                 "gamma_xz_at_y0", "strain_gradient_xz_1_per_mm")
    endpoints = np.array([[0.] * 6] + [[r[k] for k in raw_order] for r in grid.training_rows(f.spec)])
    endpoint_times = [s["time_s"] for s in preflight["stages"]]
    def raw_csv(name, header, rows):
        with (f.directory / name).open("w", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(header.split(",")); writer.writerows(rows)
    raw_csv("RAW_ENDPOINT_MACRO.csv", "load_ste", [[i + 1, t, *q] for i, (t, q) in enumerate(zip(endpoint_times, endpoints, strict=True))])
    dense_times = grid.training_clock()
    raw_csv("DENSE_MACRO_RAW.csv", "result_s", [[i + 1, t, *[np.interp(t, endpoint_times, endpoints[:, j]) for j in range(6)]]
        for i, t in enumerate(dense_times)])
    raw_csv("RAW_INITIAL_MACRO.csv", "eps_x0,k", [[0.] * 6 + [1.]])
    normalized = grid.training_endpoints(f.directory / "RAW_ENDPOINT_MACRO.csv", preflight["stages"], f.spec)
    f.write_csv("ALL_STAGE_MACRO.csv", normalized)
    (f.directory / "RST_CLOCK.csv").write_text("1081,15000\n")
    (f.directory / "EXTRACT.dat").write_text(extraction_deck_fixture())
    f.write_json("EXTRACTION_COMPLETION.json", dict(rc=0, ansys_error_count=0, license_error=False))
    source_files = {}
    for name in grid.RECOVERY_METADATA:
        target = "ORIGINAL_" + name if name in ("LAUNCH.json", "STATUS.json", "PROCESS_SCOPE.json") else name
        source_files[name] = dict(device=1, inode=2, bytes=0, mtime_ns=3, ctime_ns=4, sha256=f.digest(target))
    hashes = {"yap_training_c3.rdb": "a" * 64, "yap_training_c3.rst": "b" * 64,
        "yap_training_c3.dat": f.digest("SOLVE.dat"), "date_all_stage_macro.csv": f.digest("RAW_ENDPOINT_MACRO.csv"),
        "free_generalized_macro_substeps.csv": f.digest("DENSE_MACRO_RAW.csv"),
        "free_generalized_initial_macro.csv": f.digest("RAW_INITIAL_MACRO.csv")}
    for name, h in hashes.items():
        source_files["solver/" + name] = dict(device=1, inode=2, bytes=0, mtime_ns=3, ctime_ns=4, sha256=h)
    current_files = dict(grid.read_record(f.directory / "SOURCE_PINS.json")[0]["files_sha256"])
    del current_files["/retired/operator_grid.py"]
    current_files["/recovery/controller/operator_grid.py"] = "9" * 64
    f.write_json("RECOVERY_PINS.json", dict(schema="yap.operator_grid.sources.v1", files_sha256=current_files,
        common_utils="/retired/common/ansys_utils.py"))
    recovery = dict(schema="yap.operator_grid.training_recovery.v1", work="/new/recovery/work", source_work=plan["work"],
        source=plan["source"], geometry=f.spec, source_plan_sha256=f.digest("TRAINING_PLAN.json"),
        source_pins_sha256=plan["source_pins_sha256"], source_stop_sha256=f.digest("ORIGINAL_STATUS.json"),
        recovery_pins_sha256=f.digest("RECOVERY_PINS.json"), original_raw_seal_contemporaneous=False,
        automatic_retry=False, review_sha256="9" * 64, original_files=source_files, source_exit_proof=proof,
        expires_epoch=0., host="retired_recovery_host")
    f.write_json("RECOVERY_PLAN.json", recovery)
    f.write_json("RECOVERY_RAW_INPUTS.json", dict(original_files=source_files, original_raw_seal_contemporaneous=False,
        original_source_exit_proof=proof))
    raw = json.loads((f.directory / "RAW_SOLVER_SEAL.json").read_text())
    raw["original_raw_seal_contemporaneous"] = False
    f.write_json("RAW_SOLVER_SEAL.json", raw)
    initial = json.loads((f.directory / "INITIAL_CLOCK_AUDIT.json").read_text())
    initial["raw_initial_csv_sha256"] = f.digest("RAW_INITIAL_MACRO.csv")
    initial["endpoint_export_sha256"] = f.digest("ALL_STAGE_MACRO.csv")
    f.write_json("INITIAL_CLOCK_AUDIT.json", initial)
    def seal_recovery():
        companion = dict(schema="yap.operator_grid.training_recovery_evidence.v1", state="TRAINING_EXTRACTED_NOT_OPERATOR",
            recovery_plan_sha256=f.digest("RECOVERY_PLAN.json"), source_plan_sha256=f.digest("TRAINING_PLAN.json"),
            files_sha256={name: f.digest(name) for name in grid.RECOVERY_ARTIFACTS},
            original_raw_seal_contemporaneous=False, thermal_geometry_validated=False, runtime_admitted=False)
        f.write_json("RECOVERY_COMPLETE.json", companion)
        f.reseal()
        receipt = grid.read_record(f.directory / "TRAINING_COMPLETE.json")[0]
        receipt.update(recovery_plan_sha256=f.digest("RECOVERY_PLAN.json"), recovery_evidence_sha256=f.digest("RECOVERY_COMPLETE.json"))
        f.write_json("TRAINING_COMPLETE.json", receipt)
        return f.digest("TRAINING_COMPLETE.json")
    return SimpleNamespace(f=f, seal=seal_recovery)


def test_recovered_training_audits_complete_archive_without_original_host_paths(recovery_archive):
    r = recovery_archive
    evidence = grid.load_training_evidence(r.f.directory, r.seal())
    assert evidence["geometry"]["path"] == "kappa_xz_plus"
    assert not evidence["runtime_admitted"]


@pytest.mark.parametrize("fault", ["count", "side_effect", "normalization", "source_binding", "helper", "sealing_claim"])
def test_recovery_companion_resealed_but_inconsistent_evidence_fails(recovery_archive, fault):
    r = recovery_archive; f = r.f
    if fault == "count": (f.directory / "RST_CLOCK.csv").write_text("1080,15000\n")
    if fault == "side_effect":
        with (f.directory / "EXTRACT.dat").open("a") as stream: stream.write("SOLVE\n")
    if fault == "normalization":
        rows = grid.training_csv(f.directory / "ALL_STAGE_MACRO.csv"); rows[0]["time_s"] = 1.; f.write_csv("ALL_STAGE_MACRO.csv", rows)
    if fault in ("source_binding", "helper", "sealing_claim"):
        p = "RECOVERY_PLAN.json"; plan = grid.read_record(f.directory / p)[0]
        if fault == "source_binding": plan["source_pins_sha256"] = "0" * 64
        if fault == "sealing_claim": plan["original_raw_seal_contemporaneous"] = True
        if fault == "helper":
            pins = grid.read_record(f.directory / "RECOVERY_PINS.json")[0]
            pins["files_sha256"]["/retired/common/ansys_utils.py"] = "0" * 64
            f.write_json("RECOVERY_PINS.json", pins); plan["recovery_pins_sha256"] = f.digest("RECOVERY_PINS.json")
        f.write_json(p, plan)
    with pytest.raises(ValueError): grid.load_training_evidence(f.directory, r.seal())


def settlement_clock(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(grid.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(grid.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    return clock


def test_launch_identity_waits_for_two_consecutive_expected_commands(monkeypatch):
    clock = settlement_clock(monkeypatch)
    child = SimpleNamespace(pid=123, poll=lambda: None)
    expected = dict(pid=123, start_ticks="456", pgid=123, sid=123, command=["worker"])
    transient = dict(expected, command=["launching"])
    samples = iter([transient, expected, transient, expected, expected])
    monkeypatch.setattr(grid, "process_identity", lambda _: next(samples))
    observed = {}
    assert grid.wait_exec_identity(child, ["worker"], observed) == expected
    assert clock[0] == pytest.approx(.08)
    assert observed == {(123, "456"): expected}


@pytest.mark.parametrize("fault", [None, "birth", "parent", "anchor"])
def test_proven_discovery_revalidates_parent_chain_without_identity_schema_change(monkeypatch, fault):
    anchor = dict(pid=123, start_ticks="root", pgid=123, sid=123, command=["root"])
    child = dict(pid=124, start_ticks="child", pgid=123, sid=123, command=["child"])
    calls = {123: 0, 124: 0}
    def identity(pid, *, include_parent=False):
        calls[pid] += 1
        result = dict(anchor if pid == 123 else child)
        if fault == "birth" and pid == 124 and calls[pid] > 1:
            result["start_ticks"] = "reused"
        if fault == "anchor" and pid == 123 and calls[pid] > 1:
            result["start_ticks"] = "reused"
        if include_parent:
            result["ppid"] = 1 if pid == 123 else 999 if fault == "parent" else 123
        return result
    monkeypatch.setattr(grid, "process_identity", identity)
    monkeypatch.setattr(grid, "own_process_census", lambda: [dict(pid=123, ppid=1, pgid=123, sid=123, state="S"),
        dict(pid=124, ppid=123, pgid=123, sid=123, state="S")])
    result = grid.proven_descendants(anchor)
    if fault is None:
        assert result == [anchor, child]
        assert all("ppid" not in value for value in result)
    else:
        assert result == []


@pytest.mark.parametrize("field,value", [("start_ticks", "457"), ("pgid", 124), ("sid", 124), ("pid", 124)])
def test_launch_identity_rejects_birth_or_session_changes(monkeypatch, field, value):
    settlement_clock(monkeypatch)
    child = SimpleNamespace(pid=123, poll=lambda: None)
    original = dict(pid=123, start_ticks="456", pgid=123, sid=123, command=["worker"])
    samples = iter([original, dict(original, **{field: value})])
    monkeypatch.setattr(grid, "process_identity", lambda _: next(samples))
    with pytest.raises(ValueError, match="identity"):
        grid.wait_exec_identity(child, ["worker"], {})


@pytest.mark.parametrize("outer_seconds", [None, .03, 0.])
def test_launch_identity_mismatch_has_bounded_budget(monkeypatch, outer_seconds):
    clock = settlement_clock(monkeypatch)
    monkeypatch.setattr(grid.time, "time", lambda: 100.)
    child = SimpleNamespace(pid=123, poll=lambda: None)
    identity = dict(pid=123, start_ticks="456", pgid=123, sid=123, command=["wrong"])
    monkeypatch.setattr(grid, "process_identity", lambda _: identity)
    observed = {}
    with pytest.raises(TimeoutError):
        grid.wait_exec_identity(child, ["worker"], observed,
            deadline=None if outer_seconds is None else 100. + outer_seconds)
    assert clock[0] == pytest.approx(5. if outer_seconds is None else outer_seconds)
    if outer_seconds != 0.:
        assert observed == {(123, "456"): identity}


@pytest.mark.parametrize("exit_poll", [1, 3])
def test_launch_identity_cannot_publish_exited_worker(monkeypatch, exit_poll):
    settlement_clock(monkeypatch)
    calls = [0]
    def poll():
        calls[0] += 1
        return 1 if calls[0] == exit_poll else None
    child = SimpleNamespace(pid=123, poll=poll)
    identity = dict(pid=123, start_ticks="456", pgid=123, sid=123, command=["worker"])
    monkeypatch.setattr(grid, "process_identity", lambda _: identity)
    with pytest.raises(RuntimeError, match="exited"):
        grid.wait_exec_identity(child, ["worker"], {})


def test_full_grid_is_requested_not_claimed_ready():
    plan = grid.coverage_plan()
    assert plan["combinations"] == len(plan["rows"]) == 81
    assert plan["pitches_um"] == list(range(2, 11))
    assert plan["totals_um"] == list(range(20, 101, 10))
    assert plan["physical_readiness"] is False
    assert plan["native_catalog_modified"] is False
    for row in plan["rows"]:
        assert row["thermal_geometry_validated"] is False
        assert row["interpolation_enabled"] is False
        assert row["w2w_runtime_status"].startswith("BLOCKED")
        if row["bonding_pitch_um"] != 2:
            assert row["pitch_build_status"] == "BLOCKED_CELL_MESH_AND_BASIS_REVIEW"
            assert row["d2w_runtime_status"] == "NOT_ADMITTED"


def test_pad_diameter_scales_with_pitch_for_both_domains():
    plan = grid.coverage_plan()
    assert plan["fixed_microgeometry"]["pad_diameter_to_pitch_ratio"] == 0.5
    for row in plan["rows"]:
        assert row["pad_diameter_um"] == row["bonding_pitch_um"] * 0.5
        assert "PAD_POLICY" not in row["pitch_build_status"]
    rows = {row["bonding_pitch_um"]: row for row in plan["rows"]}
    assert [rows[p]["pad_diameter_um"] for p in range(2, 11)] == [
        1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.,
    ]


def test_periodic_supercells_are_not_silently_rescaled():
    rows = {row["bonding_pitch_um"]: row for row in grid.coverage_plan()["rows"]}
    assert [rows[p]["common_cell_side_um"] for p in range(2, 11)] == [6, 6, 12, 30, 6, 42, 24, 18, 30]


def test_thickness_accounting_inner_outer_and_initial():
    plan = grid.coverage_plan()
    for row in plan["rows"]:
        total = row["total_thickness_um"]
        assert row["inner_si_um"] + 13 == total
        assert row["outer_si_um"] + 11.5 == total
        assert row["inner_si_um"] + row["inner_removable_si_um"] == 763.5
        assert row["outer_si_um"] + row["outer_removable_si_um"] == 763.5
        assert row["w2w_endpoint_components"]["outer"][-1] == grid.si_key(row["outer_si_um"])
        assert "back_bi" not in row["w2w_endpoint_components"]["outer"]
    assert plan["initial_wafer_total_um"] == 775
    assert plan["outer_back_bi_deposition"] is False


def test_shared_layers_and_extreme_pilot():
    layers = grid.coverage_plan()["p2_si_layers"]
    assert len(layers) == 18
    assert sum(v["status"] == "NOT_BUILT" for v in layers.values()) == 14
    assert {v["thickness_um"] for v in layers.values() if v["pilot_selected"]} == {7, 8.5, 87, 88.5}
    assert grid.si_key(8.5) == "tsv_si_h8p5"
    assert grid.si_key(88.5) == "tsv_si_h88p5"


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), "7"])
def test_invalid_thickness_has_no_key(value):
    with pytest.raises(ValueError):
        grid.si_key(value)


@pytest.mark.parametrize("path", ["/tmp/grid", "/app/nanocad/#snapshot/grid", "/app/nanocad/.snapshot/grid", "/w/grid"])
def test_local_work_rejects_forbidden_paths(path):
    with pytest.raises(ValueError):
        grid.local_work(path)


def pins_fixture(tmp_path):
    script = tmp_path / "run_exp12_m4_common_port_stiffness.py"
    script.write_text("# fixture, never executed\n")
    pins = tmp_path / "pins.json"
    grid.builder.write_json(pins, dict(schema="yap.operator_grid.sources.v1",
        files_sha256={str(script): grid.builder.digest(script)}), immutable=True)
    return script, pins, grid.builder.digest(pins)


def test_source_pins_authenticate_content_before_execution(tmp_path):
    script, pins, expected = pins_fixture(tmp_path)
    grid.verify_pins(pins, expected)
    script.write_text("# changed\n")
    with pytest.raises(ValueError, match="drift"):
        grid.verify_pins(pins, expected)
    with pytest.raises(ValueError, match="receipt SHA256"):
        grid.verify_pins(pins, "0" * 64)


@pytest.mark.parametrize("fault", ["duplicate", "nonfinite", "symlink", "oversize"])
def test_receipts_are_bounded_unique_and_nonsymlink(tmp_path, fault):
    path = tmp_path / "receipt.json"
    if fault == "duplicate":
        path.write_text('{"files":{},"files":{}}')
    elif fault == "nonfinite":
        path.write_text('{"value":NaN}')
    elif fault == "symlink":
        original = tmp_path / "regular.json"
        original.write_text("{}")
        path.symlink_to(original)
    else:
        path.write_bytes(b" " * (2 * 1024**2 + 1))
    with pytest.raises(ValueError):
        grid.read_record(path)


@pytest.mark.parametrize("deadline", [0., float("nan"), float("inf")])
def test_native_deadline_rejected_before_imports(tmp_path, monkeypatch, deadline):
    script, pins, expected = pins_fixture(tmp_path)
    args = SimpleNamespace(pins=pins, pins_sha256=expected, case_deadline=deadline,
                           command=["--", sys.executable, "-B", str(script)])
    monkeypatch.setattr(grid.runpy, "run_path", lambda *a, **k: pytest.fail("must not run an expired producer"))
    with pytest.raises((ValueError, TimeoutError)):
        grid.native_stage(args)


def test_native_timeout_override_and_timer_cleanup(tmp_path, monkeypatch):
    script, pins, expected = pins_fixture(tmp_path)
    calls, timers = [], []
    utils = SimpleNamespace(run_mapdl=lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(grid, "authenticated_common", lambda _: utils)
    monkeypatch.setattr(grid, "own_process_census", lambda: [])
    monkeypatch.setattr(grid.signal, "signal", lambda *a: None)
    monkeypatch.setattr(grid.signal, "setitimer", lambda *a: timers.append(a))
    original_path, original_args = list(sys.path), list(sys.argv)
    def producer(*args, **kwargs):
        utils.run_mapdl(timeout_sec=0)
        utils.run_mapdl(timeout_sec=10)
        raise RuntimeError("fixture failure")
    monkeypatch.setattr(grid.runpy, "run_path", producer)
    args = SimpleNamespace(pins=pins, pins_sha256=expected, case_deadline=grid.time.time() + 500,
                           command=["--", sys.executable, "-B", str(script)])
    try:
        with pytest.raises(RuntimeError, match="fixture failure"):
            grid.native_stage(args)
    finally:
        sys.path[:] = original_path
        sys.argv[:] = original_args
    assert 0 < calls[0]["timeout_sec"] <= 350
    assert calls[1]["timeout_sec"] == 10
    assert timers[-1][1] == 0


def test_worker_rejects_argument_drift_before_mkdir(tmp_path, monkeypatch):
    monkeypatch.setattr(grid.builder, "physical_host", lambda: "mfd")
    script, pins, expected = pins_fixture(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    (work / "SOURCE_PINS.json").write_bytes(pins.read_bytes())
    record = dict(work=str(work), native_source=str(tmp_path / "native"), si_thicknesses_um=list(grid.PILOT_SI),
                  source_pins_sha256=expected)
    grid.builder.write_json(work / "CAMPAIGN.json", record)
    monkeypatch.setattr(grid, "local_work", lambda _: work)
    monkeypatch.setattr(grid.builder, "configure", lambda _: pytest.fail("invalid request must fail before setup"))
    args = SimpleNamespace(work=work, native_source=tmp_path / "wrong-native", pins_sha256=expected,
                           campaign_sha256=grid.builder.digest(work / "CAMPAIGN.json"))
    with pytest.raises(ValueError, match="arguments or policy differ"):
        grid.worker(args)


def test_baseline_is_pinned_not_inferred_from_files(tmp_path):
    (tmp_path / "runtime").mkdir()
    grid.builder.write_json(tmp_path / "runtime/BUNDLE.json", dict(files_sha256={}))
    with pytest.raises(ValueError, match="receipt SHA256"):
        grid.checked_baseline(tmp_path)


def test_unlisted_common_helper_is_not_imported(tmp_path):
    _, pins, expected = pins_fixture(tmp_path)
    with pytest.raises(ValueError, match="authenticated before import"):
        grid.authenticated_common(grid.verify_pins(pins, expected))


def test_unreviewed_solver_cleanup_helper_is_rejected_before_import(tmp_path):
    script, pins, expected = pins_fixture(tmp_path)
    record = grid.verify_pins(pins, expected)
    record["common_utils"] = str(script)
    with pytest.raises(ValueError, match="process-group-cleanup"):
        grid.authenticated_common(record)


@pytest.mark.parametrize("fault", ["schema", "batch", "stage", "reservation", "retry", "expiry", "nonfinite", "wait", "serialization", "host", "root", "resources", "missing_host"])
def test_campaign_policy_must_match_the_reviewed_budget(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(grid.builder, "physical_host", lambda: "mfd")
    monkeypatch.setattr(grid.builder, "OWNED_WORK_ROOT", tmp_path.parent)
    now = grid.time.time()
    args = SimpleNamespace(native_source=tmp_path / "native", pins_sha256="a" * 64)
    plan = dict(schema="yap.operator_grid.campaign.v4", execution_host="mfd",
        local_scratch_root=str(tmp_path.parent), resources={"host": "mfd"},
        work=str(tmp_path), native_source=str(args.native_source),
        si_thicknesses_um=list(grid.PILOT_SI), source_pins_sha256=args.pins_sha256,
        stage_wall_seconds=grid.STAGE_SECONDS, case_wall_seconds=grid.CASE_SECONDS,
        per_case_local_reservation_gib=grid.RESERVATION_GIB, automatic_retry=False,
        predecessor=None, predecessor_wait_seconds=grid.PREDECESSOR_WAIT_SECONDS, host_serialization=True,
        retained_solver_scratch=True, interpolation_enabled=False, created_epoch=now,
        ansys_wait_seconds=grid.ANSYS_WAIT_SECONDS, adoption=None,
        expires_epoch=now + grid.QUEUE_SECONDS)
    grid.validate_campaign(plan, tmp_path, args)
    field, value = {"schema": ("schema", "other"), "batch": ("si_thicknesses_um", [7, 7]),
        "stage": ("stage_wall_seconds", 0), "reservation": ("per_case_local_reservation_gib", 0),
        "retry": ("automatic_retry", True), "expiry": ("expires_epoch", now + 999999),
        "nonfinite": ("expires_epoch", float("nan")), "wait": ("predecessor_wait_seconds", 999999),
        "serialization": ("host_serialization", False), "host": ("execution_host", "sysml"),
        "root": ("local_scratch_root", "/wrong-root"), "resources": ("resources", {"host": "sysml"}),
        "missing_host": ("execution_host", None)}[fault]
    plan[field] = value
    with pytest.raises(ValueError):
        grid.validate_campaign(plan, tmp_path, args)


@pytest.mark.parametrize("values", [None, [], [7, 7], [77, 27], [True], [17], [37],
                                    [10], [float("nan")], [float("inf")], [7, 8.5, 18.5, 27, 28.5]])
def test_batch_rejects_unbounded_duplicate_existing_or_wrong_endpoints(values):
    with pytest.raises(ValueError):
        grid.validate_batch(values)


def test_middle_batch_has_exact_requested_d2w_totals():
    heights = grid.validate_batch([27, 47, 57, 77])
    assert [height + 13 for height in heights] == [40, 60, 70, 90]
    assert grid.validate_batch(grid.PILOT_SI) == list(grid.PILOT_SI)


@pytest.mark.parametrize("values", [(263.9, []), (300., ["ansys.e"])])
def test_next_case_requires_disk_reservation_and_no_other_own_solver(monkeypatch, values):
    free, jobs = values
    monkeypatch.setattr(grid.builder, "resource_check", lambda _: dict(local_free_gib=free, own_ansys_processes=jobs))
    with pytest.raises(RuntimeError):
        grid.check_case_resources("unused")


@pytest.mark.parametrize("host", ["mfd", "sysml"])
def test_host_lock_refuses_competing_worker_without_wait(tmp_path, monkeypatch, host):
    monkeypatch.setattr(grid.builder, "OWNED_WORK_ROOT", tmp_path)
    monkeypatch.setattr(grid.builder, "SYSML_WORK_ROOT", tmp_path)
    monkeypatch.setattr(grid.builder, "physical_host", lambda: host)
    monkeypatch.setattr(grid, "local_work", lambda p: Path(p))
    with grid.native_host_lock(tmp_path / "campaign"):
        with pytest.raises(BlockingIOError):
            with grid.native_host_lock(tmp_path / "other"):
                pytest.fail("competing worker must not acquire lock")
    with grid.native_host_lock(tmp_path / "campaign"):
        pass


def test_host_lock_rejects_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(grid.builder, "OWNED_WORK_ROOT", tmp_path)
    monkeypatch.setattr(grid.builder, "physical_host", lambda: "mfd")
    monkeypatch.setattr(grid, "local_work", lambda p: Path(p))
    target = tmp_path / "target"
    target.touch()
    (tmp_path / ".yap_operator_grid.lock").symlink_to(target)
    with pytest.raises(OSError):
        with grid.native_host_lock(tmp_path / "campaign"):
            pytest.fail("symlink lock must be rejected")


@pytest.mark.parametrize("listing", ["bad\n", "1 2 3 4 S p\n" * 4097])
def test_process_census_is_bounded_and_not_silently_truncated(monkeypatch, listing):
    monkeypatch.setattr(grid.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=listing))
    with pytest.raises(ValueError):
        grid.own_process_census()


def test_descendant_graph_keeps_only_predecessor_tree(monkeypatch):
    rows = [dict(pid=10, ppid=1), dict(pid=11, ppid=10), dict(pid=12, ppid=11), dict(pid=20, ppid=1)]
    monkeypatch.setattr(grid, "own_process_census", lambda: rows)
    monkeypatch.setattr(grid, "process_identity", lambda pid: dict(pid=pid))
    assert grid.descendant_identities(10) == [dict(pid=p) for p in (10, 11, 12)]


@pytest.mark.parametrize("case", ["clear", "orphan", "ansys"])
def test_terminal_session_census_catches_uncaptured_orphans_and_solvers(monkeypatch, case):
    observed = [dict(pid=10, pgid=10, sid=10)]
    rows = [] if case == "clear" else [dict(pid=99, ppid=1, pgid=30, sid=10 if case == "orphan" else 30,
                                            comm="python" if case == "orphan" else "ansys.e")]
    monkeypatch.setattr(grid, "own_process_census", lambda: rows)
    if case == "ansys":
        with pytest.raises(RuntimeError, match="ANSYS"):
            grid.predecessor_sessions_idle(observed, worker_pid=10)
    else:
        assert grid.predecessor_sessions_idle(observed, worker_pid=10) is (case == "clear")


def test_terminal_census_checks_original_session_even_if_worker_already_exited(monkeypatch):
    monkeypatch.setattr(grid, "own_process_census", lambda: [dict(pid=99, ppid=1, pgid=10, sid=10, comm="python")])
    assert grid.predecessor_sessions_idle([], worker_pid=10) is False


def predecessor_fixture(tmp_path, monkeypatch, *, complete=True):
    work, archive = tmp_path / "prior", tmp_path / "archives/operator_grid_prior"
    work.mkdir()
    archive.mkdir(parents=True)
    monkeypatch.setattr(grid, "local_work", lambda p: Path(p))
    monkeypatch.setattr(grid, "ARCHIVES", archive.parent)
    campaign = dict(schema="yap.operator_grid.campaign.v1", work=str(work), archive=str(archive),
                    pilot_si_um=list(grid.PILOT_SI), source_pins_sha256="a" * 64)
    grid.builder.write_json(work / "CAMPAIGN.json", campaign)
    campaign_hash = grid.builder.digest(work / "CAMPAIGN.json")
    grid.builder.write_json(archive / "CAMPAIGN.json", campaign)
    launch = dict(schema="yap.operator_grid.launch.v1", work=str(work), worker_pid=10,
                  campaign_sha256=campaign_hash, source_pins_sha256="a" * 64, command=["fixture"])
    grid.builder.write_json(work / "LAUNCH.json", launch)
    identity = dict(pid=10, start_ticks="100", command=["fixture"], pgid=10, sid=10)
    binding = dict(work=str(work), campaign_sha256=campaign_hash,
                   launch_sha256=grid.builder.digest(work / "LAUNCH.json"), worker_pid=10, processes=[identity])
    records = []
    for height in grid.PILOT_SI:
        key = "p2_" + grid.si_key(height)
        target = archive / key
        audit = dict(state="PASS_STRUCTURAL_CANDIDATE", thickness_um=height, port_rank_per_face=64,
                     all_cu_points_retained=True, thermal_geometry_validated=False, dataset_sha256="b" * 64)
        grid.builder.write_json(target / "audit/LAYER.json", audit)
        audit_hash = grid.builder.digest(target / "audit/LAYER.json")
        grid.builder.write_json(target / "ARCHIVE.json", dict(runtime_admitted=False,
                                      files_sha256={"audit/LAYER.json": audit_hash}))
        record = dict(archive=str(target), archive_sha256=grid.builder.digest(target / "ARCHIVE.json"),
                      audit_sha256=audit_hash, dataset_sha256="b" * 64, thickness_um=height,
                      state="ARCHIVED_STRUCTURAL_CANDIDATE")
        grid.builder.write_json(archive / f"COMPLETED_{key}.json", record)
        records.append(record)
    state = dict(schema="yap.operator_grid.status.v1", worker_pid=10, completed=records,
                 state="PILOT_COMPLETE_NOT_FULL_GRID" if complete else "RUNNING")
    grid.builder.write_json(work / "STATUS.json", state)
    if complete:
        grid.builder.write_json(archive / "PILOT_COMPLETE.json", state)
    monkeypatch.setattr(grid, "process_identity", lambda pid: None)
    return binding, work, archive, identity


@pytest.mark.parametrize("fault", [None, "partial", "stop", "audit", "completion", "launch"])
def test_predecessor_requires_full_immutable_structural_completion(tmp_path, monkeypatch, fault):
    binding, work, archive, _ = predecessor_fixture(tmp_path, monkeypatch)
    if fault in ("partial", "stop"):
        state = json.loads((work / "STATUS.json").read_text())
        if fault == "partial":
            state["completed"] = state["completed"][:3]
        else:
            state["state"] = "STOP_REQUIRES_REVIEW"
        grid.builder.write_json(work / "STATUS.json", state)
    elif fault == "audit":
        path = archive / "p2_tsv_si_h7/audit/LAYER.json"
        value = json.loads(path.read_text())
        value["all_cu_points_retained"] = False
        grid.builder.write_json(path, value)
    elif fault == "completion":
        (archive / "PILOT_COMPLETE.json").unlink()
    elif fault == "launch":
        value = json.loads((work / "LAUNCH.json").read_text())
        value["worker_pid"] = 20
        grid.builder.write_json(work / "LAUNCH.json", value)
    if fault is None:
        assert grid.predecessor_state(binding)["archived_cases"] == 4
    else:
        with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
            grid.predecessor_state(binding)


@pytest.mark.parametrize("identity", [None, dict(pid=10, start_ticks="101", command=["fixture"], pgid=10, sid=10)])
def test_running_predecessor_disappearance_and_pid_reuse_fail_closed(tmp_path, monkeypatch, identity):
    binding, _, _, _ = predecessor_fixture(tmp_path, monkeypatch, complete=False)
    monkeypatch.setattr(grid, "process_identity", lambda pid: identity)
    with pytest.raises(RuntimeError, match="disappeared or changed"):
        grid.predecessor_state(binding)


def test_predecessor_rechecks_status_if_normal_exit_races_first_read(tmp_path, monkeypatch):
    binding, work, archive, _ = predecessor_fixture(tmp_path, monkeypatch)
    complete = json.loads((archive / "PILOT_COMPLETE.json").read_text())
    grid.builder.write_json(work / "STATUS.json", {**complete, "state": "RUNNING"})
    def normal_exit(pid):
        grid.builder.write_json(work / "STATUS.json", complete)
        return None
    monkeypatch.setattr(grid, "process_identity", normal_exit)
    assert grid.predecessor_state(binding)["archived_cases"] == 4


def test_complete_status_waits_for_publication_only_while_pinned_worker_is_alive(tmp_path, monkeypatch):
    binding, _, archive, identity = predecessor_fixture(tmp_path, monkeypatch)
    (archive / "PILOT_COMPLETE.json").unlink()
    monkeypatch.setattr(grid, "process_identity", lambda pid: identity)
    assert grid.predecessor_state(binding) is None
    monkeypatch.setattr(grid, "process_identity", lambda pid: None)
    with pytest.raises(FileNotFoundError):
        grid.predecessor_state(binding)


def test_completion_is_rechecked_if_publication_finishes_before_worker_exit_check(tmp_path, monkeypatch):
    binding, _, archive, _ = predecessor_fixture(tmp_path, monkeypatch)
    original = grid.read_record
    attempts = []
    def racing_read(path, *args, **kwargs):
        if Path(path) == archive / "PILOT_COMPLETE.json":
            attempts.append(path)
            if len(attempts) == 1:
                raise FileNotFoundError(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(grid, "read_record", racing_read)
    assert grid.predecessor_state(binding)["archived_cases"] == 4
    assert len(attempts) == 2


def test_actual_mapdl_launch_refuses_newly_active_own_solver(tmp_path, monkeypatch):
    script, pins, expected = pins_fixture(tmp_path)
    utils = SimpleNamespace(run_mapdl=lambda **kw: pytest.fail("must not launch another solver"))
    monkeypatch.setattr(grid, "authenticated_common", lambda _: utils)
    monkeypatch.setattr(grid, "wait_ansys_idle", lambda _: (_ for _ in ()).throw(
        RuntimeError("another own-UID ANSYS process is active before MAPDL launch")))
    monkeypatch.setattr(grid.signal, "signal", lambda *a: None)
    monkeypatch.setattr(grid.signal, "setitimer", lambda *a: None)
    monkeypatch.setattr(grid.runpy, "run_path", lambda *a, **kw: utils.run_mapdl(timeout_sec=10))
    args = SimpleNamespace(pins=pins, pins_sha256=expected, case_deadline=grid.time.time() + 500,
                           command=["--", sys.executable, "-B", str(script)])
    original_path, original_args = list(sys.path), list(sys.argv)
    try:
        with pytest.raises(RuntimeError, match="before MAPDL launch"):
            grid.native_stage(args)
    finally:
        sys.path[:] = original_path
        sys.argv[:] = original_args


def test_zombie_parent_does_not_hide_live_solver_child(monkeypatch):
    rows = [dict(pid=10, ppid=1, pgid=10, sid=10, state="Z", comm="ansys261"),
            dict(pid=11, ppid=10, pgid=10, sid=10, state="Sl", comm="ansys.e")]
    monkeypatch.setattr(grid, "own_process_census", lambda: rows)
    monkeypatch.setattr(grid, "process_identity", lambda pid:
        dict(pid=pid, start_ticks="100", command=["ansys.e"], pgid=10, sid=10))
    monkeypatch.setattr(grid.os, "readlink", lambda _: "/opt/ansys/ansys.e")
    assert [p["pid"] for p in grid.ansys_blockers()] == [11]


def test_live_license_client_is_not_exempt(monkeypatch):
    monkeypatch.setattr(grid, "own_process_census", lambda:
        [dict(pid=10, ppid=1, pgid=10, sid=10, state="S", comm="ansyscl")])
    monkeypatch.setattr(grid, "process_identity", lambda pid:
        dict(pid=pid, start_ticks="100", command=["ansyscl"], pgid=10, sid=10))
    monkeypatch.setattr(grid.os, "readlink", lambda _: "/opt/ansys/ansyscl")
    assert grid.ansys_blockers()[0]["comm"] == "ansyscl"


def test_disappeared_blocker_requires_a_fresh_census(monkeypatch):
    calls = iter([[dict(pid=10, ppid=1, pgid=10, sid=10, state="S", comm="ansys.e")], []])
    monkeypatch.setattr(grid, "own_process_census", lambda: next(calls))
    monkeypatch.setattr(grid, "process_identity", lambda _: None)
    assert grid.ansys_blockers() == []


@pytest.mark.parametrize("persistent", [False, True])
def test_ansys_wait_is_bounded_and_never_signals_blockers(monkeypatch, persistent):
    clock = [100.]
    monkeypatch.setattr(grid.time, "time", lambda: clock[0])
    monkeypatch.setattr(grid.time, "sleep", lambda value: clock.__setitem__(0, clock[0] + value))
    monkeypatch.setattr(grid.builder, "physical_host", lambda: "mfd")
    monkeypatch.setattr(grid, "ansys_blockers", lambda:
        [dict(pid=10)] if persistent or clock[0] < 110 else [])
    monkeypatch.setattr(grid.os, "kill", lambda *a: pytest.fail("never signal blockers"))
    if persistent:
        with pytest.raises(RuntimeError, match="before MAPDL launch"):
            grid.wait_ansys_idle(500.)
        assert clock[0] == 100 + grid.ANSYS_WAIT_SECONDS
    else:
        grid.wait_ansys_idle(500.)
        assert clock[0] == 110


def test_mapdl_timeout_is_recomputed_after_wait(tmp_path, monkeypatch):
    script, pins, expected = pins_fixture(tmp_path)
    clock, calls = [100.], []
    monkeypatch.setattr(grid.time, "time", lambda: clock[0])
    monkeypatch.setattr(grid, "wait_ansys_idle", lambda _: clock.__setitem__(0, clock[0] + 15))
    utils = SimpleNamespace(run_mapdl=lambda **kw: calls.append(kw))
    monkeypatch.setattr(grid, "authenticated_common", lambda _: utils)
    monkeypatch.setattr(grid.signal, "signal", lambda *a: None)
    monkeypatch.setattr(grid.signal, "setitimer", lambda *a: None)
    monkeypatch.setattr(grid.runpy, "run_path", lambda *a, **kw: utils.run_mapdl(timeout_sec=0))
    args = SimpleNamespace(pins=pins, pins_sha256=expected, case_deadline=600.,
                           command=["--", sys.executable, "-B", str(script)])
    old_path, old_args = list(sys.path), list(sys.argv)
    try:
        grid.native_stage(args)
    finally:
        sys.path[:], sys.argv[:] = old_path, old_args
    assert calls[0]["timeout_sec"] == 335.


def adoption_fixture(tmp_path, monkeypatch):
    old, work = tmp_path / "old", tmp_path / "new"
    monkeypatch.setattr(grid.builder, "OWNED_WORK_ROOT", tmp_path)
    monkeypatch.setattr(grid.builder, "physical_host", lambda: "mfd")
    key = "p2_tsv_si_h27"
    source_case = old / "results" / key
    receipt = old / "STATUS.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"state":"STOP_REQUIRES_REVIEW"}')
    assets, raw, destinations = {}, {}, {}
    for stage in grid.builder.STAGES[:2]:
        assets[stage] = {}
        for name in ("summary.json", "manifest.json", "operator.npz"):
            path = source_case / "static" / stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic sealed payload")
            assets[stage][str(path)] = grid.builder.digest(path)
        log = source_case / f"{stage}.log"
        log.write_bytes(b"original solver log")
        assets[stage][str(log)] = grid.builder.digest(log)
        job = "e12m4k_tsv_si" if stage == "tangent" else "e12m4ma_tsv_si"
        for extension in ("rdb", "rst"):
            path = old / "solver" / key / stage / "tsv_si" / f"{job}.{extension}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic raw input, never MAPDL")
            raw[str(path)] = dict(stage=stage, identity=grid.file_identity(path))
        destinations[stage] = str(work / "adopted_inputs" / key / stage)
    record = dict(schema="yap.operator_grid.adoption.v1", source_work=str(old), destination_work=str(work),
        case_key=key, adopted_stages=list(grid.builder.STAGES[:2]),
        receipts_sha256={str(receipt): grid.builder.digest(receipt)}, stage_assets_sha256=assets,
        raw_files=raw, raw_copy_dirs=destinations)
    return record, work, SimpleNamespace(ROOT=work / "results"), SimpleNamespace(key=key)


@pytest.mark.parametrize("fault", ["prefix", "destination", "raw_destination", "missing_asset", "raw_escape"])
def test_adoption_rejects_invalid_scope_before_copy(tmp_path, monkeypatch, fault):
    record, work, _, _ = adoption_fixture(tmp_path, monkeypatch)
    if fault == "prefix":
        record["adopted_stages"] = ["tangent", "extract_tangent"]
    elif fault == "destination":
        record["destination_work"] = str(tmp_path / "other")
    elif fault == "raw_destination":
        record["raw_copy_dirs"]["tangent"] = str(tmp_path / "outside")
    elif fault == "missing_asset":
        record["stage_assets_sha256"]["tangent"].pop(next(iter(record["stage_assets_sha256"]["tangent"])))
    else:
        value = record["raw_files"].pop(next(iter(record["raw_files"])))
        record["raw_files"][str(tmp_path / "outside.rst")] = value
    with pytest.raises(ValueError):
        grid.validate_adoption(record, work, 27.)
    assert not work.exists()


def test_adoption_preserves_stop_assets_and_raw_inputs(tmp_path, monkeypatch):
    record, work, api, case = adoption_fixture(tmp_path, monkeypatch)
    grid.validate_adoption(record, work, 27.)
    originals = {p: Path(p).read_bytes() for p in record["receipts_sha256"] | record["raw_files"]}
    callback = grid.adoption_callback(api, case, work, record)
    callback(dict(name="tangent"))
    callback(dict(name="q6"))
    grid.verify_private_inputs(api.ROOT / case.key, record)
    assert all(Path(p).read_bytes() == data for p, data in originals.items())
    inventory = grid.builder.read_json(api.ROOT / case.key / "ADOPTED_INPUTS.json")
    for source, value in inventory["raw_inputs"].items():
        target = Path(value["copy"])
        assert not target.is_symlink() and target.stat().st_ino != Path(source).stat().st_ino
        assert target.read_bytes() == Path(source).read_bytes()
    assert grid.builder.read_json(api.ROOT / case.key / "static/tangent_ADOPTION.json")["main_solve_reexecuted"] is False


def test_changed_raw_input_stops_adoption_without_writes(tmp_path, monkeypatch):
    record, work, api, case = adoption_fixture(tmp_path, monkeypatch)
    Path(next(iter(record["raw_files"]))).write_bytes(b"changed raw input")
    with pytest.raises(ValueError, match="changed since adoption review"):
        grid.adoption_callback(api, case, work, record)(dict(name="tangent"))
    assert not work.exists()


@pytest.mark.parametrize("fault", ["path", "payload"])
def test_private_input_changes_fail_before_extraction(tmp_path, monkeypatch, fault):
    record, work, api, case = adoption_fixture(tmp_path, monkeypatch)
    grid.adoption_callback(api, case, work, record)(dict(name="tangent"))
    output = api.ROOT / case.key
    inventory = grid.builder.read_json(output / "ADOPTED_INPUTS.json")
    first = next(iter(inventory["raw_inputs"].values()))
    if fault == "path":
        first["copy"] = str(tmp_path / "unbound.rst")
        (output / "ADOPTED_INPUTS.json").chmod(0o644)
        grid.builder.write_json(output / "ADOPTED_INPUTS.json", inventory)
    else:
        target = Path(first["copy"])
        target.chmod(0o644)
        target.write_bytes(b"modified private copy")
    with pytest.raises(ValueError, match="private raw"):
        grid.verify_private_inputs(output, record, hashes=False)


def test_exited_source_rejects_cpu_only_orphan(monkeypatch):
    monkeypatch.setattr(grid, "process_identity", lambda _: None)
    monkeypatch.setattr(grid, "own_process_census", lambda:
        [dict(pid=11, ppid=1, pgid=10, sid=10, state="S", comm="python")])
    with pytest.raises(RuntimeError, match="live session/group members"):
        grid.require_exited_source(10)


def test_predecessor_wait_tracks_reparented_child_until_exit(tmp_path, monkeypatch):
    binding, _, _, parent = predecessor_fixture(tmp_path, monkeypatch, complete=False)
    child = dict(pid=11, start_ticks="200", command=["fixture-child"], pgid=11, sid=11)
    clock = [0.]
    monkeypatch.setattr(grid.time, "time", lambda: clock[0])
    monkeypatch.setattr(grid.time, "sleep", lambda amount: clock.__setitem__(0, clock[0] + amount))
    monkeypatch.setattr(grid, "predecessor_state", lambda b: None if clock[0] == 0 else dict(work=b["work"]))
    monkeypatch.setattr(grid, "descendant_identities", lambda pid: [parent, child])
    monkeypatch.setattr(grid, "process_identity", lambda pid: parent if clock[0] == 0 and pid == 10 else (
        child if pid == 11 and clock[0] < 60 else None))
    monkeypatch.setattr(grid, "predecessor_sessions_idle", lambda p, **kwargs: True)
    plan = dict(predecessor=binding, created_epoch=0., expires_epoch=grid.QUEUE_SECONDS,
                si_thicknesses_um=[27, 47, 57, 77])
    work = tmp_path / "next"
    grid.wait_predecessor(plan, work, {})
    assert clock[0] == 60
    proof = json.loads((work / "PREDECESSOR_RELEASE.json").read_text())
    assert proof["predecessor_processes_exited"] is True
    assert len(proof["observed_processes"]) == 2


def test_predecessor_wait_does_not_reset_expiry(tmp_path, monkeypatch):
    binding, _, _, _ = predecessor_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(grid.time, "time", lambda: 100.)
    monkeypatch.setattr(grid, "predecessor_state", lambda b: pytest.fail("expired wait cannot inspect/release"))
    plan = dict(predecessor=binding, created_epoch=0., expires_epoch=100., si_thicknesses_um=[27, 47, 57, 77])
    with pytest.raises(TimeoutError):
        grid.wait_predecessor(plan, tmp_path / "next", {})


def test_predecessor_proof_cannot_release_after_wait_deadline(tmp_path, monkeypatch):
    binding, _, _, _ = predecessor_fixture(tmp_path, monkeypatch)
    times = iter((0., float(grid.PREDECESSOR_WAIT_SECONDS)))
    monkeypatch.setattr(grid.time, "time", lambda: next(times))
    monkeypatch.setattr(grid, "predecessor_state", lambda b: dict(work=b["work"]))
    monkeypatch.setattr(grid, "predecessor_sessions_idle", lambda *a, **kw: True)
    plan = dict(predecessor=binding, created_epoch=0., expires_epoch=grid.QUEUE_SECONDS,
                si_thicknesses_um=[27, 47, 57, 77])
    with pytest.raises(TimeoutError, match="proof finished"):
        grid.wait_predecessor(plan, tmp_path / "next", {})
    assert not (tmp_path / "next/PREDECESSOR_RELEASE.json").exists()


def test_expired_cpu_stage_is_not_invoked():
    with pytest.raises(TimeoutError):
        grid.bounded_cpu(0, lambda: pytest.fail("expired stage cannot execute"))


def test_cpu_timer_and_previous_handler_restored_on_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(grid.signal, "getitimer", lambda *_: (0., 0.))
    monkeypatch.setattr(grid.signal, "signal", lambda *args: calls.append(args) or "previous")
    monkeypatch.setattr(grid.signal, "setitimer", lambda *args: calls.append(args))
    def fail():
        raise RuntimeError("fixture")
    with pytest.raises(RuntimeError, match="fixture"):
        grid.bounded_cpu(grid.time.time() + 60, fail)
    assert calls[-1] == (grid.signal.SIGALRM, "previous")
    assert calls[-2] == (grid.signal.ITIMER_REAL, 0)


@pytest.mark.parametrize("inner_expires_first", [False, True])
def test_nested_cpu_timer_preserves_earliest_deadline(inner_expires_first):
    import subprocess
    code = """
import signal
import time
import tools.operator_grid as grid
start = time.monotonic()
inner_expires_first = INNER
def work():
    inner_deadline = time.time() + (.2 if inner_expires_first else 5)
    grid.bounded_cpu(inner_deadline, lambda: time.sleep(.02 if not inner_expires_first else 5))
    assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
    time.sleep(5)
try:
    grid.bounded_cpu(time.time() + (5 if inner_expires_first else .2), work)
except TimeoutError:
    assert time.monotonic() - start < 2
else:
    raise AssertionError('nested stage cancelled the outer deadline')
assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
""".replace("INNER", repr(inner_expires_first))
    subprocess.run([sys.executable, "-B", "-c", code], check=True, timeout=10,
                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@pytest.mark.parametrize("fault", [None, "map", "ids", "positions", "weights", "nonfinite"])
def test_saved_bubble_audit_checks_actual_maps_points_and_weights(tmp_path, monkeypatch, fault):
    import numpy as np
    coordinates = np.array([(i,j,k) for i in range(2) for j in range(2) for k in range(2)], dtype=float)
    modes = np.arange(24., dtype=float).reshape(8,3,1)
    final = tmp_path / "final.npz"
    np.savez(final, node_ids=np.arange(1,9), coordinates_mm=coordinates, bubble_modes=modes)
    material = SimpleNamespace(material_map=lambda *a, **k: np.full((1,1),4))
    node = SimpleNamespace(structured_node_id=lambda i,j,k,ny,nz: 1+(i*2+j)*2+k)
    rule = tuple(SimpleNamespace(b_matrix=np.ones((6,24))*(g+1), position_mm=np.full(3,g/10),
                                 integration_volume_mm3=.125) for g in range(8))
    kernel = SimpleNamespace(bbar_points=lambda _: rule)
    monkeypatch.setitem(sys.modules, "audit_exp12_m4_python_bbar_reference", material)
    monkeypatch.setitem(sys.modules, "exp12_material_separable_reference", node)
    monkeypatch.setitem(sys.modules, "exp12_solid185_bbar", kernel)
    ordered = modes[[0,4,5,1,2,6,7,3]].reshape(24,1)
    maps = np.stack([p.b_matrix @ ordered for p in rule])
    dataset = SimpleNamespace(cu_point_ids=tuple(f"element_00000000_gauss_{g:02d}" for g in range(8)),
        bubble_strain_maps=maps, cu_positions_mm=np.stack([p.position_mm for p in rule]),
        cu_weights_mm3=np.full(8,.125))
    if fault == "map":
        dataset.bubble_strain_maps[0,0,0] += 1
    elif fault == "ids":
        dataset.cu_point_ids = dataset.cu_point_ids[::-1]
    elif fault == "positions":
        dataset.cu_positions_mm[0,0] += .1
    elif fault == "weights":
        dataset.cu_weights_mm3[0] += .1
    elif fault == "nonfinite":
        dataset.bubble_strain_maps[0,0,0] = np.nan
    manifest = dict(mesh=dict(nx=1,ny=1,nz=1), geometry={})
    if fault is not None:
        with pytest.raises(ValueError):
            grid.audit_saved_bubble_maps(dataset, final, manifest)
    else:
        audit = grid.audit_saved_bubble_maps(dataset, final, manifest)
        assert audit["cu_points"] == 8
        assert audit["maximum_relative_strain_map_error"] == 0


@pytest.mark.parametrize("changed", ["cu", "si"])
def test_changed_material_definitions_are_rejected(changed):
    @dataclass
    class Material:
        elastic_modulus_mpa: float
        temperature_knots: tuple = (25.,350.)
    dataset = SimpleNamespace(reference_temperature_c=25., linear_reference_mode="material_separable_bbar",
        cu_card=Material(110000.), reference_constant_materials={"si": Material(170000.)})
    expected = grid.material_contract(dataset)
    expected = json.loads(json.dumps(expected))
    grid.check_material_contract(dataset, expected)
    if changed == "cu":
        dataset.cu_card = Material(120000.)
    else:
        dataset.reference_constant_materials["si"] = Material(180000.)
    with pytest.raises(ValueError, match="differs from the pinned parent"):
        grid.check_material_contract(dataset, expected)


def test_dataset_alarm_kills_and_waits_for_owned_cpu_subprocess(monkeypatch):
    import subprocess
    children = []
    original = subprocess.Popen
    def tracked(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(subprocess, "Popen", tracked)
    with pytest.raises(TimeoutError):
        grid.bounded_cpu(grid.time.time() + .2,
            lambda: subprocess.run([sys.executable, "-B", "-c", "import time; time.sleep(30)"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True))
    assert len(children) == 1
    assert children[0].poll() is not None
