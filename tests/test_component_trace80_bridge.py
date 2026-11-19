"""Synthetic 80-port ABI checks, not operator or assembly accuracy evidence."""

import copy

import numpy as np
import pytest
from scipy import sparse

import mechanics_runtime as runtime
from W2W import component_assembly_bridge as bridge
from test_component_assembly_bridge import inputs, material, polynomial_fields, repin


def trace80_fixture(data, *, side=8):
    nodes = side**2
    cell_side = .012 if side == 97 else .006
    old_indices = np.r_[np.arange(70), np.arange(86, 150), 166]
    for role, operand in data["operands"].items():
        old = operand["arrays"]
        kff = sparse.csr_matrix((old["kff_data"], old["kff_indices"], old["kff_indptr"]),
                               shape=tuple(old["kff_shape"])).toarray()
        original = np.block([[old["kaa"], old["c"].T], [old["c"], kff]])
        stiffness = np.eye(167) * 1.5
        stiffness[np.ix_(old_indices, old_indices)] = original
        thermal = np.zeros(167)
        thermal[old_indices] = np.r_[old["fm"], old["ff"]]
        maps = np.zeros((48, 167))
        maps[:, old_indices] = np.column_stack((old["B_macro"], old["B_free"]))
        operand["arrays"] = dict(old, kaa=stiffness[:166, :166], c=stiffness[166:, :166],
            fm=thermal[:166], ff=thermal[166:], B_macro=maps[:, :166], B_free=maps[:, 166:])
        thickness = data["certificate"]["components"][role]["thickness_mm"]
        x, z = np.meshgrid(np.linspace(0., cell_side, side),
                           np.linspace(0., cell_side, side), indexing="ij")
        xyz = np.zeros((2, nodes, 3))
        xyz[:, :, 0], xyz[:, :, 2] = x.ravel(), z.ravel()
        xyz[1, :, 1] = thickness
        lifts = np.zeros((2, 3*nodes, 167))
        for face in (0, 1):
            lifts[face, :, :6] = polynomial_fields(xyz[face], (face-.5)*thickness).reshape(-1, 6)
        trace = np.zeros((3*nodes, 80))
        for component in range(3):
            trace[component::3, component] = 1.
        column = 3
        for component in (0, 2, 1):
            for node in range(nodes-1):
                if column == 80:
                    break
                trace[3*node+component, column] = 1.
                trace[3*(nodes-1)+component, column] = -1.
                column += 1
        assert column == 80
        lifts[0, :, 6:86], lifts[1, :, 86:166] = trace, trace
        operand["physical_trace"] = dict(face_node_ids=np.arange(1, 2*nodes+1).reshape(2, nodes),
                                         face_coordinates_mm=xyz, face_lifts=lifts)
    cert = data["certificate"]
    cert.update(schema="yap.component_prepared_assembly_certificate.v4",
                port_rank_per_face=80, macro_coordinate_scale=np.ones(166).tolist(),
                rigid_port_mode_indices=[0, 1, 2], cell_area_mm2=cell_side**2)
    reflection = np.eye(80)
    reflection[1, 1] = -1.
    cert["reflection_matrix"] = reflection.tolist()
    repin(data)
    return data


@pytest.fixture
def trace80_inputs(inputs):
    return trace80_fixture(inputs)


def test_explicit_trace80_has_native_two_die_raw726_and_free723(trace80_inputs):
    library, evidence = bridge.build_component_assembly_library(**trace80_inputs)
    assert library.port_rank_per_face == 80
    assert library.rigid_port_mode_indices == (0, 1, 2)
    assert all(op.macro_size == 166 for op in library.operators.values())
    proof = evidence["native_face_kinematics"]
    assert proof["port_rank_per_face"] == 80
    assert proof["layer_macro_coordinates"] == 166
    assert proof["two_die_raw_macro_coordinates"] == 726
    assert proof["two_die_gauge_free_macro_coordinates"] == 723
    assert proof["regional_bubbles_are_additional"] is True
    assert proof["maximum_absolute_error_mm"] <= 1e-12
    assert evidence["schema"] == "yap.internal_component_assembly_bridge.v4"
    assert evidence["assembly_execution_supported"] is False
    assert evidence["physical_readiness"] is False
    assert evidence["actual_large_operator_peak_memory_measured"] is False
    assert evidence["component_operand_bytes"] == {
        role: sum(array.nbytes for group in operand.values() for array in group.values())
        for role, operand in trace80_inputs["operands"].items()}


def test_trace80_actual_p4_face_census_97_squared(inputs):
    data = trace80_fixture(inputs, side=97)
    assert data["operands"]["bi"]["physical_trace"]["face_node_ids"].shape == (2, 9409)
    library, proof = bridge.build_component_assembly_library(**data)
    assert library.port_rank_per_face == 80
    assert proof["native_face_kinematics"]["maximum_absolute_error_mm"] <= 1e-12
    assert proof["actual_large_operator_peak_memory_measured"] is False


@pytest.mark.parametrize("ports,nodes", [(64, 4097), (80, 16385)])
def test_face_census_remains_bounded_by_explicit_contract(ports, nodes):
    macro = 6 + 2*ports
    physical = dict(face_node_ids=np.zeros((2, nodes), dtype=int),
                    face_coordinates_mm=np.zeros((2, nodes, 3)),
                    face_lifts=np.broadcast_to(np.zeros((1, 1, 1)), (2, 3*nodes, macro+1)))
    with pytest.raises(ValueError, match="physical face dimensions"):
        bridge._trace_contract(physical, dict(c=np.zeros((1, macro))), .0015, .006**2,
                               port_rank_per_face=ports)


@pytest.mark.parametrize("rank", [64, 79, 81, True, 80., "80", None])
def test_trace80_rank_is_explicit_integer_not_inferred(trace80_inputs, rank):
    trace80_inputs["certificate"]["port_rank_per_face"] = rank
    repin(trace80_inputs)
    with pytest.raises(ValueError, match="explicit integer face-port rank 80"):
        bridge.build_component_assembly_library(**trace80_inputs)


@pytest.mark.parametrize("schema", ["yap.internal_component_assembly_certificate.v1",
                                    "yap.internal_component_assembly_certificate.v2",
                                    "yap.component_prepared_assembly_certificate.v3"])
@pytest.mark.parametrize("retain_rank_key", [False, True])
def test_old_schemas_cannot_opt_into_80_by_shape(trace80_inputs, schema, retain_rank_key):
    trace80_inputs["certificate"]["schema"] = schema
    if not retain_rank_key:
        del trace80_inputs["certificate"]["port_rank_per_face"]
    repin(trace80_inputs)
    with pytest.raises(ValueError):
        bridge.build_component_assembly_library(**trace80_inputs)


def test_mixed_64_and_80_roles_refused_even_if_rehashed(inputs):
    old = copy.deepcopy(inputs["operands"]["beol"])
    data = trace80_fixture(inputs)
    data["operands"]["beol"] = old
    repin(data)
    with pytest.raises(ValueError, match="physical face dimensions"):
        bridge.build_component_assembly_library(**data)


@pytest.mark.parametrize("fault", ["scales", "reflection_size", "reflection_closure", "GP_order"])
def test_v4_retains_scale_reflection_and_gp_checks(trace80_inputs, fault):
    cert = trace80_inputs["certificate"]
    if fault == "scales":
        cert["macro_coordinate_scale"][86] = 2.
    elif fault == "reflection_size":
        cert["reflection_matrix"] = np.eye(64).tolist()
    elif fault == "reflection_closure":
        cert["reflection_matrix"][1][1] = 1.
    else:
        trace80_inputs["operands"]["bi"]["ordered_gp"]["element_gauss_ids"][0, 1] = 7
    repin(trace80_inputs)
    with pytest.raises(ValueError):
        bridge.build_component_assembly_library(**trace80_inputs)


def test_whole_component_budget_is_checked_before_any_copy(trace80_inputs, monkeypatch):
    namespaces = trace80_inputs["operands"]["bi"]
    largest = max(sum(value.nbytes for value in group.values()) for group in namespaces.values())
    total = sum(value.nbytes for group in namespaces.values() for value in group.values())
    limit = (largest + total) // 2
    assert largest < limit < total
    monkeypatch.setattr(bridge, "_TRACE80_COMPONENT_BYTES", limit)

    def must_not_copy(*args, **kwargs):
        pytest.fail("aggregate byte admission must precede all component copies")

    monkeypatch.setattr(bridge, "_numeric_copy", must_not_copy)
    with pytest.raises(ValueError, match="whole-component operand byte budget"):
        bridge.build_component_assembly_library(**trace80_inputs)


def test_numeric_copy_budget_carries_across_namespaces():
    remaining = [40]
    first = bridge._numeric_copy({"array": np.ones(4)}, remaining_bytes=remaining)
    assert first["array"].nbytes == 32 and remaining == [8]
    assert first["array"].flags.writeable is False
    with pytest.raises(ValueError, match="operand byte budget"):
        bridge._numeric_copy({"another": np.ones(2)}, remaining_bytes=remaining)
    assert remaining == [8]


def test_native_conversion_requires_explicit80_and_keeps_micro_limit(trace80_inputs):
    data = trace80_inputs
    args = dict(cu_card=data["cu_card"], material_kernels=data["material_kernels"],
                material_binding=data["material_binding"], macro_coordinate_scale=np.ones(166))
    arrays = data["operands"]["bi"]["arrays"]
    with pytest.raises(ValueError, match="reference dimensions"):
        runtime.component_native_layer_arguments(arrays, **args)
    converted = runtime.component_native_layer_arguments(arrays, port_rank_per_face=80, **args)
    assert converted["macro_size"] == 166 and converted["bubble_size"] == 1
    assert converted["strain_maps"].shape == (8, 6, 167)
    for rank in (True, "80", 80., 79):
        with pytest.raises(ValueError, match="explicit component face-port rank"):
            runtime.component_native_layer_arguments(arrays, port_rank_per_face=rank, **args)
