"""Small full-interior Si fixtures; not Cu or wafer-process validation."""

from dataclasses import dataclass
import copy
import json
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from W2W import component_silicon_cap as cap
from W2W.component_assembly_bridge import _canonical_macro_trace


@dataclass
class Library:
    operators: dict
    thickness_mm: dict
    reflection_matrix: np.ndarray
    membrane_bending_coordinate_scale: np.ndarray
    port_coordinate_scale: np.ndarray
    rigid_port_mode_indices: tuple = (0, 1, 2)
    common_port_gauge_vectors: object = None
    port_rank_per_face: int = 64
    cell_area_mm2: float = .006**2


def fixture_inputs(total=45.):
    nodes = 8
    rng = np.random.default_rng(26)
    # Periodic scalar fields, with the first mode constant.
    modes = np.linalg.qr(np.column_stack((np.ones(49), rng.normal(size=(49, 21)))))[0]
    master = (np.arange(nodes)[:, None] % 7)*7 + np.arange(nodes)[None, :] % 7
    scalar = modes[master.ravel()]
    ports = np.zeros((3*nodes**2, 64))
    reflection = np.ones(64)
    for axis in range(3):
        ports[axis::3, axis] = scalar[:, 0]
        reflection[axis] = -1 if axis == 1 else 1
    column = 3
    for axis, count in enumerate((21, 20, 20)):
        ports[axis::3, column:column+count] = scalar[:, 1:count+1]
        reflection[column:column+count] = -1 if axis == 1 else 1
        column += count
    xyz = np.array([[(x, y, z) for x in np.linspace(0., .006, nodes)
                     for z in np.linspace(0., .006, nodes)] for y in (0., .0015)])
    lifts = np.zeros((2, 3*nodes**2, 136))
    for face in (0, 1):
        lifts[face, :, :6] = _canonical_macro_trace(xyz[face], (face-.5)*.0015)
    lifts[0, :, 6:70], lifts[1, :, 70:134] = ports, ports
    trace = dict(face_coordinates_mm=xyz, face_lifts=lifts,
                 face_node_ids=np.arange(1, 129).reshape(2, 64))
    scale = np.r_[np.full(3, .001), np.full(3, .01), np.full(128, .001)]
    operator = SimpleNamespace(macro_coordinate_scale=scale, cu_card=None)
    library = Library(operators=dict(bi=operator, beol=operator, retained_si=operator),
        thickness_mm=dict(bi=.0015, beol=.010, retained_si=(total-13)/1000),
        reflection_matrix=np.diag(reflection), membrane_bending_coordinate_scale=scale[:6],
        port_coordinate_scale=scale[6:70])
    geometry = dict(initial_total_thickness_um=775., target_total_thickness_um=total,
        front_bi_thickness_um=1.5, back_bi_thickness_um=1.5, beol_thickness_um=10., diameter_mm=300.)
    return library, geometry, trace


def cache_metadata(specification):
    return dict(specification=specification, audits=dict(
        **{name: 0. for name in cap.AUDIT_ERRORS}, solve_rhs_count=135, algebra_gate_limit=1e-7,
        minimum_scaled_eigenvalue_over_spectral_radius=0.,
        fourth_scaled_eigenvalue_over_spectral_radius=.001))


@pytest.fixture(scope="module")
def sources():
    return cap._native_sources()


@pytest.mark.parametrize("total,count,residual", [(40., 72, 15.), (45., 73, None),
    (40.125, 72, 14.875), (50., 71, 15.), (40.000000001, 72, 14.999999999)])
def test_exact_bulk_recipe(total, count, residual):
    actual_count, actual_residual = cap._bulk_recipe(775.-total)
    assert actual_count == count
    if residual is None:
        assert actual_residual is None
    else:
        assert actual_residual == pytest.approx(residual, rel=0, abs=1e-12)
        assert 10 <= actual_residual < 20
    assert actual_count*10+(actual_residual or 0) == 775.-total


def test_physical_face_validation_preserves_inputs():
    library, geometry, trace = fixture_inputs()
    original = copy.deepcopy(trace)
    checked, xs, zs, scale, target, defect = cap._validate(library, geometry, trace)
    assert len(xs) == len(zs) == 8 and target == 45 and defect == 0
    assert scale.shape == (134,)
    for name in trace:
        np.testing.assert_array_equal(trace[name], original[name])
        assert not np.shares_memory(checked[name], trace[name])


def test_inherited_source_roundoff_is_preserved_not_snapped():
    library, geometry, trace = fixture_inputs()
    trace["face_lifts"][0, 0, 10] += 1.21e-12
    trace["face_lifts"][1, 0, 74] += 1.21e-12
    before = copy.deepcopy(trace)
    checked, xs, zs, scale, _, defect = cap._validate(library, geometry, trace)
    assert defect > 1e-12
    for name in trace:
        np.testing.assert_array_equal(checked[name], before[name])
    audit = cap._face_jump_audit(checked["face_lifts"][0, :, 6:70], len(xs), len(zs), scale[6:70])
    assert audit["trace_values_modified"] is False
    assert audit["source_policy"]["relative_limit"] == 2e-8
    assert max(audit["tolerance_by_column"]) > defect


@pytest.mark.parametrize("fault", ["periodic", "reflection", "ledger", "macro", "bubbles", "order"])
def test_bad_physical_contract_refused(fault):
    library, geometry, trace = fixture_inputs()
    if fault == "periodic":
        trace["face_lifts"][0, 0, 10] += .01
        trace["face_lifts"][1, 0, 74] += .01
    elif fault == "reflection":
        library.reflection_matrix = np.eye(64)
    elif fault == "ledger":
        geometry["target_total_thickness_um"] = 40.
    elif fault == "macro":
        trace["face_lifts"][0, 0, 0] += 1e-5
    elif fault == "bubbles":
        trace["face_lifts"][0, 0, 134] = 1.
    else:
        trace["face_coordinates_mm"][:, [0, 1]] = trace["face_coordinates_mm"][:, [1, 0]]
    with pytest.raises(ValueError):
        cap._validate(library, geometry, trace)


@pytest.fixture(scope="module")
def full_fe(sources):
    library, geometry, trace = fixture_inputs()
    _, xs, zs, scale, _, _ = cap._validate(library, geometry, trace)
    assembled = cap._assemble(xs, zs, trace["face_lifts"][0, :, 6:70], 1.5, 3, *sources[:2])
    k, f, lift, injection, mesh = assembled
    kc, fc, audit = cap._condense_fe(k, f, lift, injection, scale)
    return library, k, f, lift, injection, mesh, kc, fc, audit


def test_full_periodic_interior_not_restricted_ritz(full_fe):
    _, _, _, _, injection, mesh, kc, fc, audit = full_fe
    assert injection.shape[1] == 3*7**2*2
    assert mesh["periodic_interior_dofs"] == 294
    assert kc.shape == (134, 134) and fc.shape == (134,)
    assert audit["solve_rhs_count"] == 135
    assert max(value for key, value in audit.items() if key.endswith("relative_defect")) < 1e-7


def test_full_crossblocks_match_independent_dense_elimination(full_fe):
    library, k, f, lift, p, _, kc, fc, _ = full_fe
    interior = (p.T@k@p).toarray()
    coupling = p.T@k@lift
    exact = lift.T@(k@lift)-coupling.T@np.linalg.solve(interior, coupling)
    exact_f = lift.T@f-coupling.T@np.linalg.solve(interior, p.T@f)
    scale = library.operators["bi"].macro_coordinate_scale
    np.testing.assert_allclose(scale[:, None]*kc*scale[None, :],
                               scale[:, None]*exact*scale[None, :], rtol=1e-9, atol=1e-18)
    np.testing.assert_allclose(scale*fc, scale*exact_f, rtol=1e-9, atol=1e-18)


def test_cache_hash_and_atomic_commit(tmp_path):
    k, f = np.eye(134), np.arange(134.)
    spec = dict(fixture_only=True)
    key = cap.hashlib.sha256(cap._json(spec).encode()).hexdigest()
    metadata = cap._publish_cache(tmp_path, key, k, f, cache_metadata(spec))
    cached = cap._read_cache(tmp_path/(key+".npz"), tmp_path/(key+".json"), key)
    np.testing.assert_array_equal(cached[0], k)
    assert cached[2] == metadata
    assert not list(tmp_path.glob("*.pending*"))
    (tmp_path/(key+".npz")).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash"):
        cap._read_cache(tmp_path/(key+".npz"), tmp_path/(key+".json"), key)


@pytest.mark.parametrize("fault", ["missing", "failing", "nan", "boolean", "specification", "nullspace"])
def test_cache_audits_and_specification_fail_closed(tmp_path, fault):
    spec = dict(fixture_only=True)
    key = cap.hashlib.sha256(cap._json(spec).encode()).hexdigest()
    metadata = cache_metadata(spec)
    if fault == "missing":
        del metadata["audits"][cap.AUDIT_ERRORS[0]]
    elif fault == "failing":
        metadata["audits"][cap.AUDIT_ERRORS[0]] = 1e-3
    elif fault == "nan":
        metadata["audits"][cap.AUDIT_ERRORS[0]] = "nan"
    elif fault == "boolean":
        metadata["audits"][cap.AUDIT_ERRORS[0]] = False
    elif fault == "specification":
        metadata["specification"] = dict(fixture_only=False)
    else:
        metadata["audits"]["fourth_scaled_eigenvalue_over_spectral_radius"] = 0.
    cap._publish_cache(tmp_path, key, np.eye(134), np.zeros(134), metadata)
    with pytest.raises(ValueError):
        cap._read_cache(tmp_path/(key+".npz"), tmp_path/(key+".json"), key)


def test_partial_cache_is_not_read_as_committed(tmp_path):
    np.savez(tmp_path/"key.npz", stiffness=np.eye(134), load=np.ones(134))
    assert cap._read_cache(tmp_path/"key.npz", tmp_path/"key.json", "key") is None


def _try_lock(path, queue):
    try:
        with cap._lock(path, timeout=.1):
            queue.put("acquired")
    except TimeoutError:
        queue.put("timeout")


def test_cache_lock_has_finite_cross_process_wait(tmp_path):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    path = tmp_path/"entry.lock"
    with cap._lock(path):
        process = context.Process(target=_try_lock, args=(path, queue))
        process.start()
        process.join(20)
        try:
            assert not process.is_alive()
            assert queue.get(timeout=2) == "timeout"
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
    with cap._lock(path, timeout=.1):
        pass


def test_forbidden_output_and_symlink_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(cap, "PROJECT", tmp_path)
    assert cap._canonical_path(tmp_path/"caps") == tmp_path/"caps"
    with pytest.raises(ValueError):
        cap._canonical_path("/tmp/no")
    with pytest.raises(ValueError):
        cap._canonical_path(tmp_path/".snapshot"/"caps")
    (tmp_path/"link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        cap._canonical_path(tmp_path/"link"/"caps")


def test_published_hit_never_creates_writable_cache(tmp_path, monkeypatch, sources):
    from W2W import assembly_operator_library as adapter
    published, writable = tmp_path/"published", tmp_path/"absent_cache"
    published.mkdir()
    library, geometry, trace = fixture_inputs()
    _, xs, zs, scale, _, _ = cap._validate(library, geometry, trace)
    spec = dict(fixture=True, thickness_um=10., thickness_intervals=6)
    key = cap.hashlib.sha256(cap._json(spec).encode()).hexdigest()
    cap._publish_cache(published, key, np.eye(134), np.zeros(134), cache_metadata(spec))
    monkeypatch.setattr(cap, "_published_root", lambda: published)
    monkeypatch.setattr(adapter, "_linear_operator", lambda k, f, scale, card: (k, f))
    _, proof = cap._segment(writable, dict(fixture=True), xs, zs,
                            trace["face_lifts"][0, :, 6:70], scale, 10., library, sources)
    assert proof["published_library_hit"] is True
    assert not writable.exists()


def test_binary_composition_exact_counts(monkeypatch):
    from W2W import assembly_operator_library as adapter
    def join(library, left, right):
        return (left[0]+right[0], left[1]+right[1]), dict(fixture_only=True)
    monkeypatch.setattr(adapter, "_join_segments", join)
    result, audits = cap._compose(None, (1, .01), 73, (0, .015))
    assert result[0] == 73
    assert result[1] == pytest.approx(.745)
    assert len(audits) < 14
