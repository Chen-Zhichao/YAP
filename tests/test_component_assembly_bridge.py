"""Synthetic dual-BI integration, not physical trace or assembly admission."""

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

import mechanics_runtime as runtime
from W2W import component_assembly_bridge as bridge
from W2W.assembly_mechanics import AssemblyExecutor, native


ROOT = Path(__file__).resolve().parents[1]
CU_FIELDS = ("plastic_strain", "creep_strain", "equivalent_plastic_strain", "equivalent_creep_strain")
BIRTH_FIELDS = ("birth_macro_coordinates", "birth_bubble_coordinates")


def document_sha(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("ascii")).hexdigest()


def array_sha(values):
    digest = hashlib.sha256()
    for name, array in sorted(values.items()):
        digest.update(json.dumps([name, array.dtype.str, array.shape], separators=(",", ":")).encode("ascii"))
        digest.update(b"\n")
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def material():
    from mechanics_runtime import shared_distribution_path
    path = shared_distribution_path(ROOT.parent / "operator_repair_20261015/source/exp12_cu_material_point.py")
    spec = importlib.util.spec_from_file_location("_assembly_bridge_corrected_cu", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    card = module.CuCard(e_room_mpa=131000., e_anneal_mpa=131000., anneal_c=350.,
        alpha_room_per_c=17.6e-6, alpha_anneal_per_c=17.6e-6,
        creep_law="modified_time_hardening", creep_prefactor_mpa_minus_n_per_s=1e-12,
        creep_time_exponent=-.5, creep_exponent=3.)
    kernels = SimpleNamespace(update_batch=module.trial_update_batch,
        tangent_batch=module.trial_tangent_action_batch, elastic_stiffness=module.isotropic_stiffness)
    binding = runtime.bind_corrected_material(material_kernels=kernels, cu_card=card,
        source_path=path, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), scope="synthetic_fixture")
    yield card, kernels, binding
    sys.modules.pop(spec.name, None)


def synthetic_operand(card, kernels, thickness, seed):
    size = 135
    rigid = np.zeros(size)
    rigid[[6, 70]] = 1 / np.sqrt(2)
    projector = np.eye(size) - np.outer(rigid, rigid)
    base = projector @ np.diag(np.linspace(1., 2., size)) @ projector
    rng = np.random.default_rng(seed)
    maps = rng.normal(size=(8, 6, size)) * .002
    maps[:, :, 6:134] = 0
    weights = np.linspace(1., 2., 8) * 1e-10
    elastic = kernels.elastic_stiffness(*card.elastic(25.))
    alpha = np.r_[np.full(3, card.alpha(25.)), np.zeros(3)]
    stiffness = base + np.einsum("p,psi,st,ptj->ij", weights, maps, elastic, maps)
    thermal = np.einsum("p,psi,s->i", weights, maps, elastic @ alpha)
    kff = sparse.csr_matrix(stiffness[134:, 134:])
    arrays = dict(kaa=stiffness[:134, :134], c=stiffness[134:, :134],
        kff_data=kff.data, kff_indices=kff.indices, kff_indptr=kff.indptr, kff_shape=np.array(kff.shape),
        fm=thermal[:134], ff=thermal[134:], B_macro=maps.reshape(48, size)[:, :134],
        B_free=maps.reshape(48, size)[:, 134:], weights=weights,
        reference_cu_stiffness=elastic, reference_cu_alpha=alpha, reference_temperature_c=np.array(25.))
    lateral = np.array([(x, z) for x in np.linspace(0., .006, 8) for z in np.linspace(0., .006, 8)])
    xyz = np.zeros((2, 64, 3))
    xyz[:, :, 0] = lateral[:, 0]
    xyz[:, :, 2] = lateral[:, 1]
    xyz[1, :, 1] = thickness
    trace = np.zeros((192, 64))
    trace[::3, 0] = 1.
    for index in range(1, 64):
        trace[3*(index-1), index] = 1.
        trace[-3, index] = -1.
    lifts = np.zeros((2, 192, size))
    lifts[0, :, 6:70], lifts[1, :, 70:134] = trace, trace
    producer = native("build_exp12_m4_static_lift_fields")
    labels = ["eps_x", "eps_y", "gamma_xy", "kappa_x", "kappa_y", "kappa_xy"]
    for face in (0, 1):
        fields = producer.macro_face_lift_fields(xyz[face], (face-.5)*thickness, labels)
        # The producer's historical eps_y translation is registered explicitly.
        fields[:, 2, 1] -= .5*(xyz[face, :, 2].min() + xyz[face, :, 2].max())
        lifts[face, :, :6] = fields.reshape(192, 6)
    physical = dict(face_node_ids=np.arange(1, 129).reshape(2, 64),
                    face_coordinates_mm=xyz, face_lifts=lifts)
    gp = dict(element_gauss_ids=np.column_stack((np.ones(8, dtype=int), np.arange(8))),
              coordinates_mm=np.column_stack((np.linspace(.001, .005, 8),
                  np.linspace(thickness/4, 3*thickness/4, 8), np.linspace(.005, .001, 8))))
    return dict(arrays=arrays, physical_trace=physical, ordered_gp=gp)


@pytest.fixture
def inputs(material):
    card, kernels, binding = material
    geometry = dict(width_mm=10., length_mm=10., total_thickness_um=50.,
                    front_bi_thickness_um=1.5, back_bi_thickness_um=1.5, beol_thickness_um=10.)
    thickness = dict(bi=.0015, beol=.010, retained_si=.037)
    operands = {role: synthetic_operand(card, kernels, value, seed)
                for seed, (role, value) in enumerate(thickness.items(), 23)}
    certificate = dict(schema="yap.internal_component_assembly_certificate.v1", scope="synthetic_fixture",
        geometry=geometry.copy(), bi_layer_count=2, cell_area_mm2=.006**2,
        macro_coordinate_scale=np.ones(134).tolist(), reflection_matrix=np.eye(64).tolist(),
        rigid_port_mode_indices=[0], thickness_orientations=[1, -1], components={},
        macro_contract=dict(labels=["eps_x", "eps_y", "gamma_xy", "kappa_x", "kappa_y", "kappa_xy"],
            input_units=["1", "1", "1", "1/mm", "1/mm", "1/mm"],
            physical_coordinate_order=["x", "thickness_y", "public_y_as_z"],
            displacement_order=["UX", "UY", "UZ"], displacement_units="mm",
            reference="layer_midplane", source_reference_offset_mm=0.,
            in_plane_origin="cell_center", shear_and_twist="engineering", normal_curvature_sign=-1))
    for role, operand in operands.items():
        certificate["components"][role] = dict(thickness_mm=thickness[role], virtual_work_certified=True,
            **{key + "_sha256": array_sha(value) for key, value in operand.items()})
    return dict(operands=operands, geometry=geometry, certificate=certificate,
        expected_certificate_sha256=document_sha(certificate), cu_card=card,
        material_kernels=kernels, material_binding=binding)


def repin(inputs):
    for role, operand in inputs["operands"].items():
        inputs["certificate"]["components"][role].update(
            {key + "_sha256": array_sha(value) for key, value in operand.items()})
    inputs["expected_certificate_sha256"] = document_sha(inputs["certificate"])


def physical_v2_fixture(inputs):
    """Three translations with normal-vector reflection; still synthetic data."""
    cert = inputs["certificate"]
    cert["schema"] = "yap.internal_component_assembly_certificate.v2"
    cert["rigid_port_mode_indices"] = [0, 1, 2]
    reflection = np.eye(64)
    reflection[1, 1] = -1
    cert["reflection_matrix"] = reflection.tolist()
    for operand in inputs["operands"].values():
        lifts = operand["physical_trace"]["face_lifts"]
        for face, offset in ((0, 6), (1, 70)):
            lifts[face, :, offset+1:offset+3] = 0
            lifts[face, 1::3, offset+1] = 1
            lifts[face, 2::3, offset+2] = 1
    repin(inputs)
    return inputs


def test_v2_three_physical_translations_and_nonidentity_reflection(inputs):
    library, evidence = bridge.build_component_assembly_library(**physical_v2_fixture(inputs))
    assert library.rigid_port_mode_indices == (0, 1, 2)
    assert library.common_port_gauge_vectors is None
    assert library.reflection_matrix[1, 1] == -1
    assert evidence["native_face_kinematics"]["maximum_absolute_error_mm"] < 1e-12
    assert evidence["physical_readiness"] is False


@pytest.mark.parametrize("total", [20., 30., 40., 60., 70., 90., 100., 101.5])
def test_catalog_certificate_preserves_exact_target_si_and_native_faces(inputs, total):
    physical_v2_fixture(inputs)
    height = (total-13.)/1000
    inputs["geometry"]["total_thickness_um"] = total
    inputs["certificate"]["geometry"] = inputs["geometry"].copy()
    inputs["certificate"]["components"]["retained_si"]["thickness_mm"] = height
    inputs["operands"]["retained_si"] = synthetic_operand(
        inputs["cu_card"], inputs["material_kernels"], height, 25)
    physical_v2_fixture(inputs)
    with pytest.raises(ValueError, match="registered dual-BI geometry"):
        bridge.build_component_assembly_library(**inputs)
    inputs["certificate"]["schema"] = "yap.component_prepared_assembly_certificate.v3"
    repin(inputs)
    library, proof = bridge.build_component_assembly_library(**inputs)
    assert library.thickness_mm["retained_si"] == height
    assert proof["native_face_kinematics"]["maximum_absolute_error_mm"] < 1e-12
    assert proof["physical_readiness"] is False


@pytest.mark.parametrize("fault", ["nonconstant", "dependent", "reflection", "involution", "gauge"])
def test_v2_rejects_repinned_nonphysical_gauges_or_reflection(inputs, fault):
    physical_v2_fixture(inputs)
    cert = inputs["certificate"]
    if fault in ("nonconstant", "dependent"):
        for operand in inputs["operands"].values():
            lifts = operand["physical_trace"]["face_lifts"]
            if fault == "nonconstant":
                lifts[0, 1, 7] += .1
                lifts[1, 1, 71] += .1
            else:
                lifts[0, :, 7] = lifts[0, :, 6]
                lifts[1, :, 71] = lifts[1, :, 70]
    elif fault == "reflection":
        cert["reflection_matrix"][1][1] = 1
    elif fault == "involution":
        cert["reflection_matrix"][1][1] = .5
    else:
        cert["rigid_port_mode_indices"] = [0]
    repin(inputs)
    with pytest.raises(ValueError):
        bridge.build_component_assembly_library(**inputs)


def test_pinned_library_retains_corrected_material_and_operand_algebra(inputs):
    library, provenance = bridge.build_component_assembly_library(**inputs)
    assert set(library.operators) == {"bi", "beol", "retained_si"}
    assert library.common_port_gauge_vectors is None
    assert library.rigid_port_mode_indices == (0,)
    assert provenance["scope"] == "synthetic_fixture"
    assert provenance["assembly_execution_supported"] is provenance["physical_readiness"] is False
    assert provenance["material"]["clock_semantics"] == runtime.CORRECTED_MATERIAL_CLOCK
    assert provenance["native_face_kinematics"]["face_count"] == 16
    assert provenance["native_face_kinematics"]["shared_interface_count"] == 7
    for role, layer in library.operators.items():
        assert layer.material_kernels is inputs["material_kernels"]
        assert layer.corrected_material_binding is inputs["material_binding"]
        arrays = inputs["operands"][role]["arrays"]
        np.testing.assert_array_equal(layer.reference_stiffness(25.)[:134, :134], arrays["kaa"])
        np.testing.assert_array_equal(layer.weights_mm3, arrays["weights"])
        # Integrated Cu virtual work in the unchanged ordered coordinate basis.
        rng = np.random.default_rng(31)
        displacement, direction = rng.normal(size=(2, 135))
        stress = rng.normal(size=(8, 6))
        force = np.einsum("p,psi,ps->i", layer.weights_mm3, layer.strain_maps, stress)
        strain_delta = np.einsum("psi,i->ps", layer.strain_maps, direction)
        assert direction @ force == pytest.approx(np.sum(layer.weights_mm3[:, None] * stress * strain_delta))
        assert np.isfinite(layer.reference_stiffness(25.) @ displacement).all()
    saved = library.operators["bi"].weights_mm3.copy()
    inputs["operands"]["bi"]["arrays"]["weights"][:] *= 2
    np.testing.assert_array_equal(library.operators["bi"].weights_mm3, saved)


def test_total80_requires_actual_si67_operands_not_single_bi_si68p5(inputs):
    inputs["geometry"]["total_thickness_um"] = 80.
    inputs["certificate"]["geometry"] = inputs["geometry"].copy()
    inputs["certificate"]["components"]["retained_si"]["thickness_mm"] = .067
    inputs["operands"]["retained_si"] = synthetic_operand(
        inputs["cu_card"], inputs["material_kernels"], .067, 25)
    repin(inputs)
    library, _ = bridge.build_component_assembly_library(**inputs)
    executor = AssemblyExecutor(prepared_plan(80.), library, temporal_profile="distilled14")
    body = executor.leaf(1)
    assert library.thickness_mm["retained_si"] == .067
    assert body.placements[-1].z_top_mm - body.placements[0].z_bottom_mm == pytest.approx(.080)
    inputs["certificate"]["components"]["retained_si"]["thickness_mm"] = .0685
    inputs["expected_certificate_sha256"] = document_sha(inputs["certificate"])
    with pytest.raises(ValueError, match="exact role thickness"):
        bridge.build_component_assembly_library(**inputs)


@pytest.mark.parametrize("fault", ["certificate", "arrays", "port_permutation", "macro_trace", "gp_order", "gp_coordinate"])
def test_caller_pins_reject_changed_same_shape_operands(inputs, fault):
    operand = inputs["operands"]["bi"]
    if fault == "certificate":
        inputs["certificate"]["cell_area_mm2"] *= 2
    elif fault == "arrays":
        operand["arrays"]["kaa"][1, 1] *= 2
    elif fault == "port_permutation":
        lifts = operand["physical_trace"]["face_lifts"]
        lifts[0, :, [7, 8]] = lifts[0, :, [8, 7]]
    elif fault == "macro_trace":
        operand["physical_trace"]["face_lifts"][0, 0, 0] += .1
    elif fault == "gp_order":
        operand["ordered_gp"]["element_gauss_ids"] = operand["ordered_gp"]["element_gauss_ids"][::-1].copy()
    else:
        operand["ordered_gp"]["coordinates_mm"][0, 0] += .001
    with pytest.raises(ValueError, match="certificate|SHA256"):
        bridge.build_component_assembly_library(**inputs)


@pytest.mark.parametrize("fault", ["reflection", "gauge", "basis_matrix", "orientation", "single_bi", "thickness", "scale", "virtual_work"])
def test_unsupported_contracts_refused_even_with_new_certificate_pin(inputs, fault):
    cert = inputs["certificate"]
    if fault == "reflection":
        cert["reflection_matrix"][1][1] = -1
    elif fault == "gauge":
        cert["rigid_port_mode_indices"] = [0, 1]
    elif fault == "basis_matrix":
        cert["basis_matrix"] = np.eye(134).tolist()
    elif fault == "orientation":
        cert["thickness_orientations"] = [1]
    elif fault == "single_bi":
        cert["bi_layer_count"] = 1
    elif fault == "thickness":
        cert["components"]["retained_si"]["thickness_mm"] = .0385
    elif fault == "scale":
        cert["macro_coordinate_scale"][70] = 2
    else:
        cert["components"]["bi"]["virtual_work_certified"] = False
    inputs["expected_certificate_sha256"] = document_sha(cert)
    with pytest.raises(ValueError):
        bridge.build_component_assembly_library(**inputs)


@pytest.mark.parametrize("fault", ["permutation", "both_faces_permutation", "bubble_trace", "duplicate_gp", "quadrature_order", "nonrigid", "missing_trace"])
def test_physical_contract_checked_beyond_hashes(inputs, fault):
    operand = inputs["operands"]["bi"]
    lifts = operand["physical_trace"]["face_lifts"]
    if fault == "permutation":
        lifts[0, :, [7, 8]] = lifts[0, :, [8, 7]]
    elif fault == "both_faces_permutation":
        lifts[0, :, [7, 8]] = lifts[0, :, [8, 7]]
        lifts[1, :, [71, 72]] = lifts[1, :, [72, 71]]
    elif fault == "bubble_trace":
        lifts[0, 0, 134] = .1
    elif fault == "duplicate_gp":
        operand["ordered_gp"]["element_gauss_ids"][1] = operand["ordered_gp"]["element_gauss_ids"][0]
    elif fault == "quadrature_order":
        operand["ordered_gp"]["element_gauss_ids"] = operand["ordered_gp"]["element_gauss_ids"][::-1].copy()
    elif fault == "nonrigid":
        lifts[0, 0, 6] = lifts[1, 0, 70] = 2
    else:
        del operand["physical_trace"]
    repin(inputs)
    with pytest.raises(ValueError):
        bridge.build_component_assembly_library(**inputs)


@pytest.mark.parametrize("fault", ["random_macro", "macro_order", "curvature_units", "bottom_reference",
                                   "tensor_shear", "normal_sign", "uncentered_eps_y"])
def test_repinned_inconsistent_macro_trace_is_not_a_physical_certificate(inputs, fault):
    operand = inputs["operands"]["bi"]
    lifts = operand["physical_trace"]["face_lifts"]
    if fault == "random_macro":
        lifts[:, :, :6] = np.random.default_rng(41).normal(size=lifts[:, :, :6].shape) * .01
    elif fault == "macro_order":
        lifts[:, :, [0, 1]] = lifts[:, :, [1, 0]]
    elif fault == "curvature_units":
        lifts[:, :, 3:6] *= 1000.
    elif fault == "bottom_reference":
        lifts[:, :, 3:6] += .00075*lifts[:, :, :3]
    elif fault == "tensor_shear":
        lifts[:, :, 2] *= 2.
        lifts[:, :, 5] *= 2.
    elif fault == "normal_sign":
        lifts[:, 1::3, 3:6] *= -1.
    else:
        lifts[:, 2::3, 1] += .003
    repin(inputs)
    with pytest.raises(ValueError, match="physical macro trace"):
        bridge.build_component_assembly_library(**inputs)


@pytest.mark.parametrize("fault", ["labels", "units", "reference", "axes", "displacement_frame", "offset", "shear"])
def test_explicit_macro_semantics_cannot_be_relabelled_by_repinning(inputs, fault):
    contract = inputs["certificate"]["macro_contract"]
    if fault == "labels":
        contract["labels"][0], contract["labels"][1] = contract["labels"][1], contract["labels"][0]
    elif fault == "units":
        contract["input_units"][3] = "1/um"
    elif fault == "reference":
        contract["reference"] = "layer_bottom"
    elif fault == "axes":
        contract["physical_coordinate_order"] = ["x", "public_y", "thickness_z"]
    elif fault == "displacement_frame":
        contract["displacement_order"] = ["UX", "UZ", "UY"]
    elif fault == "offset":
        contract["source_reference_offset_mm"] = .00075
    else:
        contract["shear_and_twist"] = "tensor"
    repin(inputs)
    with pytest.raises(ValueError, match="q6 order, units and reference"):
        bridge.build_component_assembly_library(**inputs)


def test_identity_reflection_refuses_repinned_normal_displacing_ports(inputs):
    for operand in inputs["operands"].values():
        lifts = operand["physical_trace"]["face_lifts"]
        lifts[0, 1, 7] = lifts[1, 1, 71] = .3
    repin(inputs)
    with pytest.raises(ValueError, match="physical trace/native transform"):
        bridge.build_component_assembly_library(**inputs)


@pytest.mark.parametrize("fault", ["midplane_shift", "backside_curvature_sign", "backside_port_swap"])
def test_physical_trace_checks_actual_native_maps_not_only_local_polynomial(inputs, monkeypatch, fault):
    api = native("exp12_v2_production_sequential_process")
    original = api._topology

    def incompatible_native_topology(*args, **kwargs):
        topology = original(*args, **kwargs)
        instances = list(topology.instances)
        instance = instances[3]
        transform = instance.macro_transform.copy()
        if fault == "midplane_shift":
            transform[:3, 3:6] = 0.
        elif fault == "backside_curvature_sign":
            transform[3:6, 3:6] *= -1.
        else:
            transform[6:70], transform[70:134] = transform[70:134].copy(), transform[6:70].copy()
        instances[3] = replace(instance, macro_transform=transform)
        return replace(topology, instances=tuple(instances))

    monkeypatch.setattr(api, "_topology", incompatible_native_topology)
    with pytest.raises(ValueError, match="physical trace/native transform"):
        bridge.build_component_assembly_library(**inputs)


def prepared_plan(total=50.):
    from D2W.assembly_order import build_plan, load_stack_input
    config = load_stack_input(ROOT / "D2W/input/design_6_4hi_p2/stack.yaml")["stack"]
    config.update(initial_bows_um=[.1, -.1], assembly_orders="(1,2)")
    config["geometry"]["total_thickness_um"] = total
    config["fabrication_layers"][2]["thickness_um"] = total - 13.
    return build_plan(config)


def layer_snapshot(layer):
    return {**{name: getattr(layer.cu_state, name).copy() for name in CU_FIELDS},
            **{name: getattr(layer, name).copy() for name in BIRTH_FIELDS}}


def assert_transferred(actual, snapshot, old):
    for name, expected in snapshot.items():
        observed = getattr(actual.cu_state, name) if name in CU_FIELDS else getattr(actual, name)
        previous = getattr(old.cu_state, name) if name in CU_FIELDS else getattr(old, name)
        np.testing.assert_array_equal(observed, expected)
        assert not np.shares_memory(observed, previous)


def test_existing_prepared_executor_leaf_join_anneal_preserves_cu14_birth_and_ages(inputs, monkeypatch):
    library, _ = bridge.build_component_assembly_library(**inputs)
    executor = AssemblyExecutor(prepared_plan(), library, temporal_profile="distilled14",
        solver_options=dict(linear_solver="assembled_dense", tolerance=1e-9, max_iterations=64))
    # No thinning or deposition is allowed for a prepared dual-BI die.
    def forbidden(*args, **kwargs):
        raise AssertionError("prepared D2W must not remove or append layers")
    monkeypatch.setattr(executor.transactions, "remove_top_all_gauss_biaxial_modules", forbidden)
    monkeypatch.setattr(executor.transactions, "append_stress_free_all_gauss_biaxial_modules", forbidden)
    lower, upper = executor.leaf(1), executor.leaf(2)
    assert len(lower.placements) == len(upper.placements) == 4
    for body in (lower, upper):
        assert body.placements[-1].z_top_mm - body.placements[0].z_bottom_mm == pytest.approx(.05)
        assert {p.role for p in body.placements} == {"front_bi", "beol", "retained_si", "back_bi"}
        for layer in body.state.layers.values():
            assert np.count_nonzero(np.concatenate([getattr(layer.cu_state, f).ravel() for f in CU_FIELDS])) == 0
        prepared, audit = executor.prepare_back(body)
        assert prepared is body and audit["policy"] == "prefabricated_back_BI_no_thinning_or_deposition"
    snapshots = {}
    for index, body in enumerate((lower, upper), 1):
        for ordinal, (name, layer) in enumerate(body.state.layers.items(), 1):
            level = index * ordinal * 1e-9
            layer.cu_state.plastic_strain[:] = level * np.array([1., -1., 0., .2, .3, .4])
            layer.cu_state.creep_strain[:] = level * np.array([.5, -.5, 0., .1, .2, .3])
            layer.cu_state.equivalent_plastic_strain[:] = .001 * index
            layer.cu_state.equivalent_creep_strain[:] = .002 * index
            assert sum(getattr(layer.cu_state, f).size for f in CU_FIELDS) == 14 * 8
            body.ages_s[name] = 300. if name.endswith("back_bi") else 1300. * index
            snapshots[name] = layer_snapshot(layer)
    parents = [executor.checkpoint(body) for body in (lower, upper)]
    joined, record = executor.join(lower, upper, "synthetic_dual_BI")
    assert [executor.checkpoint(body) for body in (lower, upper)] == parents
    for body in (lower, upper):
        for name, old in body.state.layers.items():
            assert_transferred(joined.state.layers[name], snapshots[name], old)
            assert joined.ages_s[name] == body.ages_s[name]
    assert all(value == 0 for value in record["held_flat_history_increment"].values())
    before = executor.checkpoint(joined)
    annealed, report = executor.anneal(joined)
    assert executor.checkpoint(joined) == before
    assert [executor.checkpoint(body) for body in (lower, upper)] == parents
    assert report["duration_s"] == 4900.
    for name, layer in annealed.state.layers.items():
        assert annealed.ages_s[name] == joined.ages_s[name] + 4900.
        for field in BIRTH_FIELDS:
            np.testing.assert_array_equal(getattr(layer, field), snapshots[name][field])
        assert layer.cu_state.plastic_strain.shape == layer.cu_state.creep_strain.shape == (8, 6)
        assert np.isfinite(np.concatenate([getattr(layer.cu_state, f).ravel() for f in CU_FIELDS])).all()
    # Newly constructed physical BI instances have independent virgin births
    # and clocks, even while sharing immutable component algebra.
    fresh_executor = AssemblyExecutor(executor.plan, library, temporal_profile="distilled14")
    fresh = fresh_executor.leaf(1)
    assert set(fresh.ages_s.values()) == {300.}
    for name, layer in fresh.state.layers.items():
        assert not any(np.any(getattr(layer.cu_state, field)) for field in CU_FIELDS)
        assert not np.shares_memory(layer.birth_macro_coordinates, annealed.state.layers[name].birth_macro_coordinates)
    assert executor.operators["die01_front_bi"] is not executor.operators["die01_back_bi"]
    assert fresh_executor.operators["die01_back_bi"] is not executor.operators["die01_back_bi"]


def test_layer_clock_endpoint_offset_is_applied_once_for_residual_batch_and_tangent(inputs):
    from W2W.assembly_mechanics import LayerClockOperator
    library, _ = bridge.build_component_assembly_library(**inputs)
    layer = library.operators["bi"]
    proxy = LayerClockOperator(layer, age_offset_s=1000.)
    state = layer.initial_state()
    q, z = np.zeros(134), np.zeros(1)
    q[0] = .1
    clock = dict(creep_active=True, plastic_active=True)
    before = layer_snapshot(state)
    expected = layer.evaluate(q, z, 350., 200., state, elapsed_time_s=1500., **clock)
    for _ in range(2):
        result = proxy.evaluate(q, z, 350., 200., state, elapsed_time_s=500., **clock)
        np.testing.assert_array_equal(result.response, expected.response)
    batched = proxy.evaluate_many([q, q], [z, z], 350., 200., [state, state], elapsed_time_s=500., **clock)
    for result in batched:
        np.testing.assert_allclose(result.response, expected.response, rtol=1e-14, atol=1e-22)
        np.testing.assert_array_equal(result.cu_trial.state.equivalent_creep_strain,
                                      expected.cu_trial.state.equivalent_creep_strain)
    wrong_clock = layer.evaluate(q, z, 350., 200., state, elapsed_time_s=500., **clock)
    assert np.linalg.norm(expected.cu_trial.state.equivalent_creep_strain
                          - wrong_clock.cu_trial.state.equivalent_creep_strain) > 1e-12
    direction = np.linspace(0., 1e-4, 135)
    action = proxy.tangent_action(q, z, direction, 350., 200., state, elapsed_time_s=500., **clock)
    reference = layer.tangent_action(q, z, direction, 350., 200., state, elapsed_time_s=1500., **clock)
    np.testing.assert_array_equal(action, reference)
    for field in CU_FIELDS:
        np.testing.assert_array_equal(getattr(state.cu_state, field), before[field])
    assert proxy.age_offset_s == 1000.


def test_public_catalog_remains_unavailable_for_assembly():
    with pytest.raises(RuntimeError, match="does not support assembly"):
        runtime.resolve_component_operator({}, for_assembly=True)


def test_material_binding_is_mandatory(inputs):
    inputs["material_binding"] = None
    with pytest.raises(ValueError, match="binding|Binding"):
        bridge.build_component_assembly_library(**inputs)
