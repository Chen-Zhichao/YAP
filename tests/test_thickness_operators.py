"""Mock distributions exercise the public path without scientific admission."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import thickness_operators as api


def document(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    path.chmod(0o444)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(root, name, arrays):
    path = root / name
    np.savez(path, **arrays)
    path.chmod(0o444)
    return dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


@pytest.fixture(params=[2,42])
def mock_family(tmp_path,request):
    return build_mock_family(tmp_path, request.param)


def build_mock_family(tmp_path, rank, bracket=(27.,37.)):
    root = tmp_path / "distribution"
    root.mkdir()
    gp = 3
    lo, hi = bracket
    half = (hi-lo)/2.
    center = lo+half
    shapes = dict(kaa=(134,134), c=(rank,134), kff=(rank,rank), fm=(134,), ff=(rank,),
                  B_macro=(6*gp,134), B_free=(6*gp,rank), weights=(gp,), macro_map=(134,262))
    degrees = dict(kaa=6,c=4,kff=2,fm=4,ff=2,B_macro=3,B_free=1,weights=1,macro_map=1)
    coefficients = {key:np.zeros((degrees[key]+1,*shape)) for key,shape in shapes.items()}
    for key in shapes:
        coefficients[key][0] = 1.
    coefficients['kff'][0] = np.eye(rank)*2
    coefficients['kaa'][0] = np.eye(134)
    coefficients['c'][0] *= .01
    coefficients['weights'][0] = np.arange(1,gp+1)
    coefficients['weights'][1] = coefficients['weights'][0]*half/center
    coefficients['macro_map'][0] = np.eye(134,262)
    static = dict(reference_cu_stiffness=np.eye(6), reference_cu_alpha=np.ones(6),
        reference_temperature_c=np.array(25.),reference_materials=np.ones((4,3)), macro_offset=np.zeros(134),
        coordinates_unit=np.array([[0.,0.,0.],[1.,1.,1.]]),
        Cu_positions_unit=np.array([[0.,.5,0.]]),V=np.eye(2),native_body_columns=np.array([0,1]),
        native_node_indices=np.array([0,1]),Cu_native_element_ids=np.array([1]),Cu_native_gauss_indices=np.array([0]))
    coeff = artifact(root,"coefficients.npz",coefficients)
    coeff['fields'] = degrees
    stat = artifact(root,"static.npz",static)
    geometry = artifact(root,"geometry.npz",dict(lifts134=np.ones((3,6,134))))
    fixed = artifact(root,"fixed.npz",dict(kaa=np.eye(134)))
    topology = artifact(root,"topology.npz",dict(free_map=np.eye(262,259)))
    bindings = document(root/"RUNTIME_BINDINGS.json", dict(
        runtime_source_sha256=hashlib.sha256(Path(api.__file__).read_bytes()).hexdigest(),
        runtime_imports=["numpy","stdlib"],FE_generator_on_runtime_path=False))
    material = dict(Cu_source="frozen",acceleration_source="frozen")
    physical = dict(BI_pitch_um=2.,diameter_pitch_ratio=.5,BI_um=1.5,BEOL_um=10.,
                    TSV_pitch_um=6.,cell_xz_um=[6.,6.])
    manifest = dict(schema=api.SCHEMA,id="MOCK_NOT_A_SCIENTIFIC_PASS",bracket_um=list(bracket),
        center_um=center,half_span_um=half,geometry=physical,geometry_sha256=api._digest(physical),
        material=material,material_sha256=api._digest(material),
        endpoints={api._endpoint_key(h):dict(operator_sha256="a"*64,reference_sha256="b"*64)
                   for h in bracket},
        constructor_sha256="c"*64,candidate_sha256="d"*64,coefficients=coeff,static=stat,
        geometry_coefficients=geometry,topology=topology,fixed_components=dict(bi=fixed,beol=fixed),
        fixed_geometry=dict(bi=fixed,beol=fixed),
        runtime_bindings_sha256=bindings)
    document(root/"FAMILY.json",manifest)
    return root, tmp_path/"cache"


def activate(root, *, passed=True):
    assessment = dict(passed=passed,failed_observations=0 if passed else 1,matched_states=73,
                      observations=292,strict_pointwise_OR=True,spatial_only=True)
    sha = document(root/"assessment.json",assessment)
    context = dict(constructor_payload=dict(sha256="d"*64),Cu_source="frozen",acceleration_source="frozen",
        thermal_policy=dict(primary_tolerance=1e-8,linear_tolerance=1e-9),
        selected_original_ordinals=[0]+[360*c+j for c in range(3) for j in
            (*range(20,161,20), *(160+i for i in (1,2,4,8,10,20,30,40)), *range(220,361,20))],
        Cu_state_adoption=False)
    histories = {}
    for mode in ("native","reduced"):
        fields = dict(context)
        if mode == "native":
            fields.update(no_PORT64_or_POD=True,full_native_interior_coordinates=273372,Cu_GP=185696)
        pin = document(root/(mode+".json"),dict(mode=mode,states=73,context=fields,candidate=dict(sha256="c"*64)))
        histories[mode+"_history"] = dict(path=mode+".json",sha256=pin)
    family_sha = hashlib.sha256((root/"FAMILY.json").read_bytes()).hexdigest()
    verification = document(root/"build.json",dict(candidate32_pack_equivalence_passed=True,
                                                  family=dict(sha256=family_sha)))
    document(root/"ACTIVATE.json",dict(schema=api.ACTIVATION_SCHEMA,
        family_sha256=family_sha,
        candidate_sha256="d"*64,state="INDEPENDENT_HELDOUT32_GRADED73_BOW_PASS",
        assessment=dict(path="assessment.json",sha256=sha),
        build_verification=dict(path="build.json",sha256=verification),**histories))


def test_pending_is_inspectable_but_cannot_resolve_or_evaluate(mock_family):
    root, cache = mock_family
    family = api.open_family(root,cache_dir=cache)
    assert family.metadata()['activation'] == "PENDING_INDEPENDENT_BOW"
    for method in (family.resolve,family.evaluate,family.geometry):
        with pytest.raises(api.PendingValidationError,match="pending independent"):
            method(prepared_total_um=45.)
    assert not cache.exists()


def test_failed_assessment_cannot_activate(mock_family):
    root, _ = mock_family
    activate(root,passed=False)
    with pytest.raises(api.IntegrityError,match="assessment has not passed"):
        api.open_family(root).evaluate(si_um=32.)


def test_prepared45_and_single43p5_resolve_identically(mock_family):
    root, _ = mock_family
    activate(root)
    family = api.open_family(root)
    a = family.resolve(prepared_total_um=45.)
    b = family.resolve(single_bi_total_um=43.5)
    assert (a.si_um,a.single_bi_total_um,a.prepared_total_um) == (32.,43.5,45.)
    assert a.si_um == b.si_um
    assert family.cache_key(prepared_total_um=45.) == family.cache_key(si_um=32.)
    flags = family.metadata()['capabilities']
    assert flags['experimental_prepared_d2w']
    assert not flags['prepared_assembly_accuracy']
    assert not flags['aged_state_transfer']
    assert not flags['FE_generation']
    assert not flags['legacy_local_cached_tangent_compatible']
    assert flags['maximum_supported_micro_rank'] == 48
    assert flags['rank_gt32_runtime_strategy'] == 'original_SparseModel_JVP_reference_preconditioner'


@pytest.mark.parametrize('query',[dict(si_um=26.999),dict(si_um=37.001),dict(si_um=float('nan')),
    dict(si_um=True),dict(si_um=32.,prepared_total_um=45.),{}])
def test_bad_thickness_is_refused(mock_family,query):
    root, _ = mock_family
    activate(root)
    with pytest.raises(api.ThicknessOperatorError):
        api.open_family(root).resolve(**query)


def test_arbitrary_fraction_and_readonly_component_arrays(mock_family):
    root, _ = mock_family
    activate(root)
    family = api.open_family(root)
    result = family.evaluate(si_um=31.234567)
    rank = len(result.arrays['ff'])
    x = 31.234567/32.
    np.testing.assert_allclose(result.arrays['kff'],np.eye(rank)*2/x)
    np.testing.assert_allclose(result.arrays['c'],np.full((rank,134),.01/x))
    with pytest.raises(ValueError):
        result.arrays['weights'][0] = -1
    csr = result.component_arrays()
    assert tuple(csr['kff_shape']) == (rank,rank)
    assert not csr['kff_data'].flags.writeable
    assert family.geometry(si_um=31.234567)['coordinates_mm'][1,1] == pytest.approx(.031234567)


def test_memory_and_disk_reuse_without_private_imports(mock_family,monkeypatch):
    root, cache = mock_family
    activate(root)
    first = api.open_family(root,cache_dir=cache)
    result = first.evaluate(si_um=32.)
    assert first.evaluate(prepared_total_um=45.).arrays is result.arrays
    assert first.cache_info()['memory_hits'] == 1
    second = api.open_family(root,cache_dir=cache)
    monkeypatch.setattr(second,"_compute",lambda resolution: pytest.fail("disk cache recomputed"))
    np.testing.assert_array_equal(second.evaluate(si_um=32.).arrays['kff'],result.arrays['kff'])
    assert second.cache_info()['disk_hits'] == 1


def test_cache_keys_do_not_round_thickness(mock_family):
    root, _ = mock_family
    activate(root)
    family = api.open_family(root)
    assert family.cache_key(si_um=32.) != family.cache_key(si_um=np.nextafter(32.,33.).item())


def test_disk_budget_does_not_delete_existing_files(mock_family):
    root, cache = mock_family
    activate(root)
    cache.mkdir()
    retained = cache/"retained.txt"
    retained.write_text("not ours")
    family = api.open_family(root,cache_dir=cache,disk_cache_bytes=1)
    family.evaluate(si_um=32.)
    assert retained.read_text() == "not ours"
    assert family.cache_info()['disk_budget_skips'] == 1


def test_tampered_coefficient_file_refused(mock_family):
    root, _ = mock_family
    activate(root)
    path = root/"coefficients.npz"
    path.chmod(0o644)
    with path.open('ab') as stream:
        stream.write(b'corrupt')
    path.chmod(0o444)
    with pytest.raises(api.IntegrityError,match="SHA256"):
        api.open_family(root).evaluate(si_um=32.)


def test_manifest_exposure_cannot_mutate_family(mock_family):
    root, _ = mock_family
    family = api.open_family(root)
    metadata = family.manifest
    metadata['endpoints']['27']['operator_sha256'] = '0'*64
    assert family.manifest['endpoints']['27']['operator_sha256'] != '0'*64


def test_distribution_path_escape_rejected(mock_family):
    root, _ = mock_family
    with pytest.raises(api.IntegrityError):
        api._path(root,'../other.npz')


def test_numeric_component_entry_is_distinct_from_fixed_catalog(mock_family):
    root, _ = mock_family
    activate(root)
    family = api.open_family(root)
    entry = family.resolve_entry('component:BI2_SI32')
    assert entry['schema'] == 'yap.parameterized_component_entry.v1'
    assert entry['geometry']['total_thickness_um'] == 43.5
    assert entry['experimental_prepared_total_um'] == 45.
    assert entry['runtime_strategy'] == 'original_SparseModel_JVP_reference_preconditioner'
    assert not entry['validation']['prepared_assembly_accuracy']
    assert family.resolve_entry('BI2_SI31p25')['geometry']['si_tsv_thickness_um'] == 31.25
    for name in ('BI3_SI32','BI2_SI38','other:BI2_SI32'):
        with pytest.raises(api.ThicknessOperatorError):
            family.resolve_entry(name)


def test_numeric_component_entry_is_gated(mock_family):
    root, _ = mock_family
    with pytest.raises(api.PendingValidationError):
        api.open_family(root).resolve_entry('BI2_SI32')


def test_shared_writer_lock_skips_disk_without_unbounded_growth(mock_family):
    root, cache = mock_family
    activate(root)
    cache.mkdir()
    lock = cache/'.thickness-quota.lock'
    lock.mkdir()
    family = api.open_family(root,cache_dir=cache)
    value = family.evaluate(si_um=32.)
    assert value.arrays['kff'].shape[0] in (2,42)
    assert family.cache_info()['disk_writes'] == 0
    assert family.cache_info()['disk_busy_skips'] == 1
    assert list(cache.iterdir()) == [lock]
    lock.rmdir()


def test_public_module_source_pin_is_captured_at_import():
    assert api._IMPORTED_SOURCE_SHA256 == hashlib.sha256(Path(api.__file__).read_bytes()).hexdigest()


@pytest.mark.parametrize("rank", [2,42])
def test_second_bracket_uses_same_resolve_evaluate_geometry_path(tmp_path, rank):
    root, _ = build_mock_family(tmp_path, rank, (17.,27.))
    activate(root)  # Synthetic lineage only, never scientific admission.
    family = api.open_family(root)
    assert family.metadata()['bracket_um'] == [17.,27.]
    for si in (17.,22.,23.125,27.):
        assert family.resolve(prepared_total_um=si+13.).si_um == si
        values = family.evaluate(si_um=si).arrays
        np.testing.assert_allclose(values['kff'],np.eye(rank)*2/(si/22.))
        np.testing.assert_allclose(values['weights'],np.arange(1,4)*(si/22.))
        y = (si-22.)/5.
        np.testing.assert_allclose(family.geometry(si_um=si)['lifts134'],
                                   np.ones((6,134))*(1+y+2*y*y-1))
    for si in (16.999,27.001):
        with pytest.raises(api.ThicknessOperatorError,match='no extrapolation'):
            family.evaluate(si_um=si)


@pytest.mark.parametrize('change', [dict(bracket_um=[0.,37.]),dict(bracket_um=[37.,27.]),
    dict(bracket_um=[27.,float('inf')]),dict(center_um=32.00001),dict(half_span_um=5.00001),
    dict(endpoints={"27.0":{},"37":{}})])
def test_invalid_bracket_and_noncanonical_keys_refused(tmp_path,change):
    root, _ = build_mock_family(tmp_path,2)
    manifest = json.loads((root/'FAMILY.json').read_text())
    manifest.update(change)
    (root/'FAMILY.json').chmod(0o644)
    document(root/'FAMILY.json',manifest)
    with pytest.raises(api.ThicknessOperatorError):
        api.open_family(root)
