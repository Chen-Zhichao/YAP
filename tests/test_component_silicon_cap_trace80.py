"""Tiny elastic software checks, not qualification of the 97-square cap."""

import ast
import copy
import hashlib
from pathlib import Path
from types import CodeType, ModuleType, SimpleNamespace

import numpy as np
import pytest

from W2W import component_silicon_cap as legacy
from W2W import component_silicon_cap_trace80 as layout


LEGACY_SHA = "1a1c11ffcaeddbe9031727d4e749e3bba97ba64bfc2f3aa63a4eff8e4d91d851"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def polynomial_trace(xyz, height):
    x = xyz[:, 0] - .5*(xyz[:, 0].min()+xyz[:, 0].max())
    z = xyz[:, 2] - .5*(xyz[:, 2].min()+xyz[:, 2].max())
    fields = np.zeros((len(xyz), 3, 6))
    fields[:, 0, 0], fields[:, 2, 1] = x, z
    fields[:, 0, 2], fields[:, 2, 2] = z/2, x/2
    fields[:, 0, 3], fields[:, 1, 3] = height*x, -x*x/2
    fields[:, 2, 4], fields[:, 1, 4] = height*z, -z*z/2
    fields[:, 0, 5], fields[:, 1, 5], fields[:, 2, 5] = height*z/2, -x*z/2, height*x/2
    return fields.reshape(-1, 6)


def tiny_inputs():
    # Distinct spacings exercise per-brick integration without resampling.
    xs = np.array([0., .0008, .0021, .0040, .0065, .0083, .0109, .012])
    zs = np.array([0., .0011, .0025, .0044, .0068, .0086, .0107, .012])
    rng = np.random.default_rng(8025)
    scalar = np.linalg.qr(np.column_stack((np.ones(49), rng.normal(size=(49, 26)))))[0]
    master = (np.arange(8)[:, None] % 7)*7 + np.arange(8)[None, :] % 7
    values = scalar[master.ravel()]
    ports = np.zeros((192, 80))
    reflection = np.ones(80)
    for axis in range(3):
        ports[axis::3, axis] = values[:, 0]
        reflection[axis] = -1 if axis == 1 else 1
    column = 3
    for axis, count in enumerate((26, 26, 25)):
        ports[axis::3, column:column+count] = values[:, 1:count+1]
        reflection[column:column+count] = -1 if axis == 1 else 1
        column += count
    assert column == 80
    xyz = np.array([[(x, y, z) for x in xs for z in zs] for y in (0., .0015)])
    lifts = np.zeros((2, 192, 168))
    for face in (0, 1):
        lifts[face, :, :6] = polynomial_trace(xyz[face], (face-.5)*.0015)
    lifts[0, :, 6:86], lifts[1, :, 86:166] = ports, ports
    scale = np.r_[np.full(3, .001), np.full(3, .01), np.full(160, .001)]
    operator = SimpleNamespace(macro_coordinate_scale=scale, cu_card=None)
    library = SimpleNamespace(operators=dict(bi=operator, beol=operator, retained_si=operator),
        thickness_mm=dict(bi=.0015, beol=.010, retained_si=.037),
        reflection_matrix=np.diag(reflection), membrane_bending_coordinate_scale=scale[:6],
        port_coordinate_scale=scale[6:86], rigid_port_mode_indices=(0, 1, 2),
        common_port_gauge_vectors=None, port_rank_per_face=80, cell_area_mm2=.012**2)
    geometry = dict(initial_total_thickness_um=775., target_total_thickness_um=50.,
        front_bi_thickness_um=1.5, back_bi_thickness_um=1.5,
        beol_thickness_um=10., diameter_mm=300.)
    trace = dict(face_coordinates_mm=xyz, face_lifts=lifts,
                 face_node_ids=np.arange(1, 129).reshape(2, 64))
    return library, geometry, trace, xs, zs, ports, scale


def cache_metadata(engine, specification):
    return dict(specification=specification, audits=dict(
        **{name: 0. for name in engine.AUDIT_ERRORS}, solve_rhs_count=167,
        algebra_gate_limit=1e-7, minimum_scaled_eigenvalue_over_spectral_radius=0.,
        fourth_scaled_eigenvalue_over_spectral_radius=.001))


def cache_binding(engine):
    from W2W import assembly_operator_library as assembly, component_assembly_bridge as bridge

    library, _, trace, *_ = tiny_inputs()
    sources = engine._native_sources()
    construction = dict(version=engine.VERSION, source_sha256=digest(engine.__file__),
        native_sha256=sources[2], numpy_version=np.__version__,
        bridge_sha256=digest(bridge.__file__), assembly_helpers_sha256=digest(assembly.__file__),
        layout_adaptation=copy.deepcopy(engine._yap_layout_adaptation))
    return dict(material=copy.deepcopy(engine.MATERIAL),
        boundary_contract=copy.deepcopy(engine.BOUNDARY_CONTRACT),
        construction_sha256=hashlib.sha256(engine._json(construction).encode()).hexdigest(),
        coordinate_contract_sha256=engine._hash_arrays(dict(
            scale=library.operators["bi"].macro_coordinate_scale,
            reflection=library.reflection_matrix, rigid_indices=np.asarray((0, 1, 2)))),
        trace_sha256=engine._hash_arrays(dict(face_coordinates_mm=trace["face_coordinates_mm"],
            face_lifts=trace["face_lifts"][:, :, :166])))


@pytest.fixture
def engine():
    return layout.create_trace80_cap_engine()


def test_factory_lineage_and_real_module_globals(engine):
    assert isinstance(engine, ModuleType)
    proof = engine._yap_layout_adaptation
    assert proof["schema"] == "yap.silicon_cap.trace80_layout_adapter.v1"
    assert proof["legacy_source_sha256"] == LEGACY_SHA == digest(legacy.__file__)
    assert proof["adapter_source_sha256"] == digest(layout.__file__)
    assert proof["port_rank_per_face"] == 80
    assert proof["macro_coordinates"] == 166
    assert proof["inverse_ast_exact"] is True
    assert len(proof["generated_ast_sha256"]) == 64
    int(proof["generated_ast_sha256"], 16)
    for name in ("_assemble", "_condense_fe", "_read_cache", "_publish_cache", "augment_w2w_library"):
        assert getattr(engine, name).__globals__ is engine.__dict__
    second = layout.create_trace80_cap_engine()
    assert second is not engine and second.__dict__ is not engine.__dict__
    assert second._yap_layout_adaptation == proof


@pytest.mark.parametrize("field,value", [
    ("schema", "unknown"), ("legacy_source_sha256", "0"*64),
    ("adapter_source_sha256", "0"*64), ("generated_ast_sha256", "not-a-hash"),
    ("port_rank_per_face", 64), ("macro_coordinates", 134),
    ("inverse_ast_exact", False),
])
def test_storage_refuses_unbound_layout_provenance(engine, field, value):
    import component_operators as components

    engine._yap_layout_adaptation = dict(engine._yap_layout_adaptation, **{field: value})
    with pytest.raises(ValueError):
        components._cap_source_provenance(engine)


def test_factory_refuses_unknown_source_before_compilation(tmp_path, monkeypatch):
    changed = tmp_path/"component_silicon_cap.py"
    changed.write_bytes(Path(legacy.__file__).read_bytes()+b"\n")
    monkeypatch.setattr(layout, "LEGACY", changed)
    monkeypatch.setattr(layout, "ModuleType", lambda *_: pytest.fail("untrusted source was compiled"))
    with pytest.raises(ValueError, match="unknown|authenticated"):
        layout.create_trace80_cap_engine()


def test_changed_anchor_count_refuses_before_compilation(monkeypatch):
    changes = copy.deepcopy(layout.CHANGES)
    replacement, count = changes["_assemble"][134]
    changes["_assemble"][134] = (replacement, count+1)
    monkeypatch.setattr(layout, "CHANGES", changes)
    monkeypatch.setattr(layout, "ModuleType", lambda *_: pytest.fail("wrong anchors were compiled"))
    with pytest.raises(ValueError, match="anchor"):
        layout.create_trace80_cap_engine()


def test_complete_ast_independently_reverses_every_dimension_change():
    raw = Path(legacy.__file__).read_bytes()
    transformed, counts = layout._layout_tree(raw)
    functions = {node.name: node for node in transformed.body if isinstance(node, ast.FunctionDef)}
    for name, changes in layout.CHANGES.items():
        inverse = {new: old for old, (new, _) in changes.items()}
        observed = dict.fromkeys(changes, 0)
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Constant) and type(node.value) in (str, int) and node.value in inverse:
                node.value = inverse[node.value]
                observed[node.value] += 1
        assert observed == {old: count for old, (_, count) in changes.items()}
        assert counts[name] == {str(key): value for key, value in observed.items()}
    construction = [node.value for node in ast.walk(functions["augment_w2w_library"])
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "construction"
                                                for t in node.targets)]
    inserted = [key for key in construction[0].keywords if key.arg == "layout_adaptation"]
    assert len(inserted) == 1 and isinstance(inserted[0].value, ast.Name)
    assert inserted[0].value.id == "_yap_layout_adaptation"
    construction[0].keywords.remove(inserted[0])
    assert ast.dump(transformed, include_attributes=False) == ast.dump(ast.parse(raw), include_attributes=False)


def code_signature(code, undo_layout=False):
    inverse = {166: 134, 86: 70, 89: 73, 167: 135}

    def constant(value):
        if isinstance(value, CodeType):
            return code_signature(value, undo_layout)
        if undo_layout and type(value) is int:
            return inverse.get(value, value)
        if isinstance(value, tuple):
            return tuple(constant(item) for item in value)
        return value

    # Location/debug tables differ after compilation; the actual operations do not.
    return (code.co_code, tuple(constant(v) for v in code.co_consts), code.co_names,
            code.co_varnames, code.co_freevars, code.co_cellvars, code.co_argcount,
            code.co_kwonlyargcount, code.co_posonlyargcount, code.co_flags)


@pytest.mark.parametrize("name", ["_assemble", "_condense_fe"])
def test_numeric_bodies_reverse_to_legacy_compiled_operations(engine, name):
    assert code_signature(getattr(engine, name).__code__, True) == code_signature(
        getattr(legacy, name).__code__)


def test_tiny_fixture_periodicity_rank_and_physical_reflection():
    library, _, trace, xs, zs, ports, _ = tiny_inputs()
    assert len(np.unique(np.diff(xs))) > 1 and len(np.unique(np.diff(zs))) > 1
    assert np.linalg.matrix_rank(ports) == 80
    lattice = ports.reshape(8, 8, 3, 80)
    np.testing.assert_array_equal(lattice[0], lattice[-1])
    np.testing.assert_array_equal(lattice[:, 0], lattice[:, -1])
    physical = np.tile([1., -1., 1.], 64)[:, None]*ports
    np.testing.assert_array_equal(physical, ports@library.reflection_matrix)
    np.testing.assert_array_equal(trace["face_lifts"][0, :, 86:], 0.)
    np.testing.assert_array_equal(trace["face_lifts"][1, :, 6:86], 0.)


def test_production_validator_never_accepts_tiny_fixture(engine):
    library, geometry, trace, *_ = tiny_inputs()
    before = copy.deepcopy(trace)
    with pytest.raises(ValueError):
        engine._validate(library, geometry, trace)
    for name in trace:
        np.testing.assert_array_equal(trace[name], before[name])


@pytest.fixture(scope="module")
def production_trace():
    library, geometry, *_ = tiny_inputs()
    unit = np.linspace(0., 1., 97)
    xs, zs = .012*unit**1.35, .012*unit**1.15
    rng = np.random.default_rng(979680)
    scalar = np.linalg.qr(np.column_stack((np.ones(96**2), rng.normal(size=(96**2, 26)))))[0]
    masters = (np.arange(97)[:, None] % 96)*96 + np.arange(97)[None, :] % 96
    values = scalar[masters.ravel()]
    ports = np.zeros((3*97**2, 80))
    for axis in range(3):
        ports[axis::3, axis] = values[:, 0]
    column = 3
    for axis, count in enumerate((26, 26, 25)):
        ports[axis::3, column:column+count] = values[:, 1:count+1]
        column += count
    xyz = np.array([[(x, y, z) for x in xs for z in zs] for y in (0., .0015)])
    lifts = np.zeros((2, 3*97**2, 168))
    for face in (0, 1):
        lifts[face, :, :6] = polynomial_trace(xyz[face], (face-.5)*.0015)
    lifts[0, :, 6:86], lifts[1, :, 86:166] = ports, ports
    trace = dict(face_coordinates_mm=xyz, face_lifts=lifts,
                 face_node_ids=np.arange(1, 2*97**2+1).reshape(2, 97**2))
    for value in trace.values():
        value.flags.writeable = False
    return library, geometry, trace, xs, zs


def test_production97_nonuniform_validation_preserves_actual_trace(engine, production_trace, monkeypatch):
    library, geometry, trace, xs, zs = production_trace
    monkeypatch.setattr(engine, "_assemble", lambda *_a, **_k: pytest.fail("97-square FE is not authorized"))
    before = engine._hash_arrays(trace)
    checked, actual_xs, actual_zs, scale, target, defect = engine._validate(library, geometry, trace)
    np.testing.assert_array_equal(actual_xs, xs)
    np.testing.assert_array_equal(actual_zs, zs)
    assert len(np.unique(np.diff(actual_xs))) > 1 and len(np.unique(np.diff(actual_zs))) > 1
    assert scale.shape == (166,) and target == 50. and defect == 0.
    assert checked["face_coordinates_mm"].shape == (2, 9409, 3)
    assert checked["face_lifts"].shape == (2, 28227, 168)
    assert not np.any(checked["face_lifts"][:, :, 166:])
    assert engine._hash_arrays(checked) == before == engine._hash_arrays(trace)
    for name in trace:
        assert not np.shares_memory(checked[name], trace[name])
        assert not trace[name].flags.writeable


@pytest.mark.parametrize("fault", ["jump", "reflection", "order", "shifted_frame", "bubble"])
def test_production97_bad_physical_contract_refused(engine, production_trace, monkeypatch, fault):
    original_library, geometry, original_trace, *_ = production_trace
    library = copy.deepcopy(original_library)
    trace = {name: value.copy() for name, value in original_trace.items()}
    if fault == "jump":
        trace["face_lifts"][0, 0, 10] += .001
        trace["face_lifts"][1, 0, 90] += .001
    elif fault == "reflection":
        library.reflection_matrix = np.eye(80)
    elif fault == "order":
        trace["face_coordinates_mm"][:, [0, 1]] = trace["face_coordinates_mm"][:, [1, 0]]
    elif fault == "shifted_frame":
        trace["face_coordinates_mm"][:, :, 0] += .0001
    else:
        trace["face_lifts"][0, 0, 166] = .001
    monkeypatch.setattr(engine, "_assemble", lambda *_a, **_k: pytest.fail("97-square FE is not authorized"))
    before = engine._hash_arrays(trace)
    with pytest.raises(ValueError):
        engine._validate(library, geometry, trace)
    assert engine._hash_arrays(trace) == before


def test_production97_forecast_is_census_not_factor_memory_qualification():
    forecast = layout._forecast(97, 97, 6)
    assert forecast["nodes"] == 65863
    assert forecast["elements"] == 55296
    assert forecast["nodal_vector_dofs"] == 197589
    assert forecast["periodic_interior_dofs"] == 138240
    assert forecast["macro_coordinates"] == 166
    assert forecast["dense_work_forecast_bytes"] > 0
    assert forecast["factor_memory_qualified"] is False
    assert forecast["total_peak_memory_bound_claimed"] is False
    assert forecast["existing_library_memory_included"] is False


@pytest.fixture(scope="module")
def full_fe():
    engine = layout.create_trace80_cap_engine()
    library, _, _, xs, zs, ports, scale = tiny_inputs()
    bbar, material, _ = legacy._native_sources()
    k, f, lift, p, mesh = engine._assemble(xs, zs, ports, 1.5, 3, bbar, material)
    kc, fc, audit = engine._condense_fe(k, f, lift, p, scale)
    interior = (p.T@k@p).toarray()
    coupling = p.T@k@lift
    displacement = np.linalg.solve(interior, coupling)
    thermal = np.linalg.solve(interior, p.T@f)
    return engine, library, k, f, lift, p, mesh, kc, fc, audit, coupling, displacement, thermal


def test_full_periodic_interior_and_167_rhs(full_fe):
    _, _, k, _, lift, p, mesh, kc, fc, audit, *_ = full_fe
    assert k.shape == (768, 768) and lift.shape == (768, 166)
    assert p.shape == (768, 294)
    assert mesh["periodic_interior_dofs"] == 294
    assert mesh["node_count"] == 256 and mesh["element_count"] == 147
    assert kc.shape == (166, 166) and fc.shape == (166,)
    assert audit["solve_rhs_count"] == 167
    assert max(v for key, v in audit.items() if key.endswith("relative_defect")) < 1e-7


def test_complete_interior_injection_and_unmodified_macro_fields(full_fe):
    _, _, _, _, lift, p, _, _, _, _, *_ = full_fe
    _, _, _, xs, zs, ports, _ = tiny_inputs()
    dense = p.toarray()
    np.testing.assert_array_equal(dense[:192], 0.)
    np.testing.assert_array_equal(dense[-192:], 0.)
    np.testing.assert_array_equal(np.count_nonzero(dense[192:-192], axis=1), 1)
    assert np.linalg.matrix_rank(dense) == 294
    grid = dense.reshape(4, 8, 8, 3, 294)
    np.testing.assert_array_equal(grid[:, 0], grid[:, -1])
    np.testing.assert_array_equal(grid[:, :, 0], grid[:, :, -1])
    for level, y in enumerate(np.linspace(0., .0015, 4)):
        xyz = np.array([(x, y, z) for x in xs for z in zs])
        section = lift[level*192:(level+1)*192]
        np.testing.assert_allclose(section[:, :6], polynomial_trace(xyz, y-.00075), rtol=0, atol=1e-20)
        np.testing.assert_allclose(section[:, 6:86], (1-level/3)*ports, rtol=0, atol=1e-16)
        np.testing.assert_allclose(section[:, 86:166], (level/3)*ports, rtol=0, atol=1e-16)


def test_all_crossblocks_match_independent_dense_elimination(full_fe):
    _, library, k, f, lift, _, _, kc, fc, _, coupling, displacement, thermal = full_fe
    expected = lift.T@(k@lift)-coupling.T@displacement
    expected_f = lift.T@f-coupling.T@thermal
    scale = library.operators["bi"].macro_coordinate_scale
    np.testing.assert_allclose(scale[:, None]*kc*scale[None, :],
        scale[:, None]*expected*scale[None, :], rtol=1e-9, atol=1e-18)
    np.testing.assert_allclose(scale*fc, scale*expected_f, rtol=1e-9, atol=1e-18)


def test_dense_relaxed_energy_and_thermal_virtual_work(full_fe):
    _, library, k, f, lift, p, _, kc, fc, _, _, displacement, thermal = full_fe
    relaxed = lift-p@displacement
    scale = library.operators["bi"].macro_coordinate_scale
    thermal_reaction = lift.T@(k@(p@thermal)-f)
    np.testing.assert_allclose(scale*thermal_reaction, -scale*fc, rtol=1e-9, atol=1e-18)
    rng = np.random.default_rng(16680)
    for _ in range(4):
        q = scale*rng.normal(size=166)
        u = relaxed@q
        energy = float(u@(k@u))
        assert float(q@kc@q) == pytest.approx(energy, rel=1e-10, abs=1e-18)
        assert float(q@fc) == pytest.approx(float(u@f), rel=1e-10, abs=1e-18)
        residual = p.T@(k@u)
        assert np.linalg.norm(residual) < 1e-9*max(np.linalg.norm(k@u), 1e-30)


def test_three_joint_translation_nulls_no_additional_nulls(full_fe):
    _, library, _, _, _, _, _, kc, fc, _, *_ = full_fe
    rigid = np.zeros((166, 3))
    rigid[6:9] = rigid[86:89] = np.eye(3)
    scale = library.operators["bi"].macro_coordinate_scale
    scaled = scale[:, None]*kc*scale[None, :]
    norm = np.linalg.norm(scaled)
    assert np.linalg.norm(scale[:, None]*(kc@rigid)) < 1e-10*norm*np.linalg.norm(rigid/scale[:, None])
    assert np.linalg.norm(rigid.T@fc) < 1e-10*max(np.linalg.norm(fc), 1e-30)
    eigen = np.linalg.eigvalsh((scaled+scaled.T)/2)
    spectral = np.max(np.abs(eigen))
    assert np.max(np.abs(eigen[:3])) < 1e-9*spectral
    assert eigen[3] > 1e-12*spectral


def test_default_miss_refuses_before_mkdir_or_fe(engine, tmp_path, monkeypatch):
    library, _, _, xs, zs, ports, scale = tiny_inputs()
    published, writable = tmp_path/"absent_published", tmp_path/"absent_work"
    monkeypatch.setattr(engine, "_published_root", lambda: published)
    monkeypatch.setattr(engine, "_assemble", lambda *_a, **_k: pytest.fail("unqualified FE build"))
    with pytest.raises((ValueError, RuntimeError, PermissionError), match="build_guard"):
        engine._segment(writable, cache_binding(engine), xs, zs, ports, scale,
                        10., library, (None, None, {}))
    assert not writable.exists() and not published.exists()


def test_layout_cache_rejects_legacy_dimensions_and_135_rhs(engine, tmp_path):
    spec = dict(cache_binding(engine), thickness_um=10., thickness_intervals=6)
    key = hashlib.sha256(engine._json(spec).encode()).hexdigest()
    metadata = cache_metadata(engine, spec)
    metadata["audits"]["solve_rhs_count"] = 135
    engine._publish_cache(tmp_path, key, np.eye(134), np.zeros(134), metadata)
    with pytest.raises(ValueError):
        engine._read_cache(tmp_path/(key+".npz"), tmp_path/(key+".json"), key)


def test_layout_cache_rejects_legacy_arrays_even_with_167_rhs(engine, tmp_path):
    spec = dict(cache_binding(engine), thickness_um=10., thickness_intervals=6)
    key = hashlib.sha256(engine._json(spec).encode()).hexdigest()
    engine._publish_cache(tmp_path, key, np.eye(134), np.zeros(134), cache_metadata(engine, spec))
    with pytest.raises(ValueError):
        engine._read_cache(tmp_path/(key+".npz"), tmp_path/(key+".json"), key)


@pytest.mark.parametrize("fault", ["missing_boundary", "legacy_boundary", "wrong_source", "wrong_material"])
def test_trace80_cache_requires_current_source_and_boundary(engine, tmp_path, fault):
    spec = dict(cache_binding(engine), thickness_um=10., thickness_intervals=6)
    if fault == "missing_boundary":
        del spec["boundary_contract"]
    elif fault == "legacy_boundary":
        spec["boundary_contract"] = copy.deepcopy(legacy.BOUNDARY_CONTRACT)
    elif fault == "wrong_source":
        spec["construction_sha256"] = "0"*64
    else:
        spec["material"]["elastic_modulus_mpa"] += 1.
    key = hashlib.sha256(engine._json(spec).encode()).hexdigest()
    engine._publish_cache(tmp_path, key, np.eye(166), np.zeros(166), cache_metadata(engine, spec))
    with pytest.raises(ValueError):
        engine._read_cache(tmp_path/(key+".npz"), tmp_path/(key+".json"), key)


@pytest.mark.parametrize("fault", ["source", "boundary"])
def test_invalid_requested_binding_refuses_before_build_guard(tmp_path, monkeypatch, fault):
    requests = []
    engine = layout.create_trace80_cap_engine(build_guard=lambda request: requests.append(request) or False)
    binding = cache_binding(engine)
    if fault == "source":
        binding["construction_sha256"] = "0"*64
    else:
        binding["boundary_contract"] = copy.deepcopy(legacy.BOUNDARY_CONTRACT)
    library, _, _, xs, zs, ports, scale = tiny_inputs()
    monkeypatch.setattr(engine, "_published_root", lambda: tmp_path/"absent_published")
    with pytest.raises(ValueError):
        engine._segment(tmp_path/"absent_work", binding, xs, zs, ports,
                        scale, 10., library, (None, None, {}))
    assert not requests and not (tmp_path/"absent_work").exists()


def test_cache_metadata_swap_after_profile_check_refused(engine, tmp_path, monkeypatch):
    accepted = dict(cache_binding(engine), thickness_um=10., thickness_intervals=6)
    swapped = dict(accepted, boundary_contract=copy.deepcopy(legacy.BOUNDARY_CONTRACT))
    key = hashlib.sha256(engine._json(swapped).encode()).hexdigest()
    final = engine._publish_cache(tmp_path, key, np.eye(166), np.zeros(166), cache_metadata(engine, swapped))
    npz_path, json_path = tmp_path/(key+".npz"), tmp_path/(key+".json")
    # The first profile read sees the valid layout; the raw reader then sees a
    # different, internally self-consistent key/specification and payload pair.
    initial = dict(final, specification=accepted)
    json_path.write_text(engine._json(initial))
    raw_reader = engine._layout_original_read_cache

    def exchange_before_raw_read(*args):
        json_path.write_text(engine._json(final))
        return raw_reader(*args)

    monkeypatch.setattr(engine, "_layout_original_read_cache", exchange_before_raw_read)
    with pytest.raises(ValueError):
        engine._read_cache(npz_path, json_path, key)


def test_generated_cache_hit_and_storage_hooks_use_real_globals(engine, tmp_path, monkeypatch):
    import component_operators as components
    from W2W import assembly_operator_library as assembly

    library, _, _, xs, zs, ports, scale = tiny_inputs()
    published, writable = tmp_path/"published", tmp_path/"absent_work"
    published.mkdir()
    binding = cache_binding(engine)
    spec = dict(binding, thickness_um=10., thickness_intervals=6)
    key = hashlib.sha256(engine._json(spec).encode()).hexdigest()
    engine._publish_cache(published, key, np.eye(166), np.zeros(166), cache_metadata(engine, spec))
    monkeypatch.setenv("YAP_W2W_SI_CAP_LIBRARY_DIR", str(published))
    storage = components._configure_cap_storage(engine)
    assert storage["source_bytes_unchanged"] is False
    assert storage["legacy_source_bytes_unchanged"] is True
    assert storage["numerical_layout_adapted"] is True
    assert storage["layout_adaptation"] == engine._yap_layout_adaptation
    monkeypatch.setattr(engine, "_assemble", lambda *_a, **_k: pytest.fail("cache hit ran FE"))
    monkeypatch.setattr(assembly, "_linear_operator", lambda k, f, scale, card: (k, f))
    (_, thickness), proof = engine._segment(writable, binding, xs, zs,
        ports, scale, 10., library, (None, None, {}))
    assert proof["published_library_hit"] is True and thickness == .010
    assert not writable.exists()
    assert engine.augment_w2w_library.__globals__ is engine.__dict__
    assert engine._assemble.__globals__ is not legacy.__dict__
    writable.mkdir()
    engine._publish_cache(writable, key, np.eye(166), np.zeros(166), cache_metadata(engine, spec))
    pair = [writable/(key+suffix) for suffix in (".npz", ".json")]
    original = [(path.stat().st_ino, digest(path)) for path in pair]
    with pytest.raises(ValueError, match="overwrite|exists"):
        engine._publish_cache(writable, key, np.eye(166), np.zeros(166), cache_metadata(engine, spec))
    assert [(path.stat().st_ino, digest(path)) for path in pair] == original


def test_configured_generated_writer_keeps_exclusive_link_under_race(engine, tmp_path, monkeypatch):
    import component_operators as components

    published, writable = tmp_path/"published", tmp_path/"writable"
    published.mkdir()
    writable.mkdir()
    spec = dict(cache_binding(engine), thickness_um=10., thickness_intervals=6)
    key = hashlib.sha256(engine._json(spec).encode()).hexdigest()
    target = writable/(key+".npz")
    original_sha = engine._file_sha

    def hash_with_concurrent_destination(path):
        if Path(path).name.endswith(".pending.npz"):
            with target.open("xb") as stream:
                stream.write(b"other publication must survive")
        return original_sha(path)

    monkeypatch.setattr(engine, "_file_sha", hash_with_concurrent_destination)
    monkeypatch.setenv("YAP_W2W_SI_CAP_LIBRARY_DIR", str(published))
    components._configure_cap_storage(engine)
    with pytest.raises(FileExistsError):
        engine._publish_cache(writable, key, np.eye(166), np.zeros(166), cache_metadata(engine, spec))
    assert target.read_bytes() == b"other publication must survive"
    assert not (writable/(key+".json")).exists()
    assert not list(writable.glob("*.pending*"))


def test_factory_preserves_legacy_source_constants_and_cached_hit(tmp_path, monkeypatch):
    from W2W import assembly_operator_library as assembly

    before = digest(legacy.__file__)
    version, boundary = legacy.VERSION, copy.deepcopy(legacy.BOUNDARY_CONTRACT)
    original_functions = {name: getattr(legacy, name) for name in ("_assemble", "_condense_fe", "_segment")}
    published = tmp_path/"legacy_published"
    published.mkdir()
    spec = dict(fixture_only=True, thickness_um=10., thickness_intervals=6)
    key = hashlib.sha256(legacy._json(spec).encode()).hexdigest()
    metadata = cache_metadata(legacy, spec)
    metadata["audits"]["solve_rhs_count"] = 135
    legacy._publish_cache(published, key, np.eye(134), np.zeros(134), metadata)
    monkeypatch.setattr(legacy, "_published_root", lambda: published)
    monkeypatch.setattr(assembly, "_linear_operator", lambda k, f, scale, card: (k, f))
    library = SimpleNamespace(operators={"bi": SimpleNamespace(cu_card=None)})

    def hit():
        return legacy._segment(tmp_path/"absent_work", dict(fixture_only=True), None, None,
            None, np.ones(134), 10., library, (None, None, {}))[1]

    initial = hit()
    adapted = layout.create_trace80_cap_engine()
    final = hit()
    assert initial["key"] == final["key"] == key
    assert initial["published_library_hit"] is final["published_library_hit"] is True
    assert adapted.VERSION != version
    assert digest(legacy.__file__) == before == LEGACY_SHA
    assert legacy.VERSION == version and legacy.BOUNDARY_CONTRACT == boundary
    for name, function in original_functions.items():
        assert getattr(legacy, name) is function
    assert not (tmp_path/"absent_work").exists()
