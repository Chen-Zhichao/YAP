"""Tiny synthetic controls only; Main runs with finite resources and storage."""
from dataclasses import replace

import numpy as np
import pytest

from tools import coupon_assembly as assembly


def operand(role, *, private=3):
    from W2W.assembly_mechanics import native
    size = 6 + private
    layer = native("exp12_all_gauss_reduced_fe2").AllGaussReducedFE2Layer(
        macro_size=6, bubble_size=private,
        reference_stiffness=lambda t: np.eye(size),
        reference_thermal_load=lambda t: (t - 25.) * np.arange(1., size + 1.) * 1e-6,
        strain_maps=np.zeros((0, 6, size)), weights_mm3=np.zeros(0),
        macro_coordinate_scale=np.ones(6), bubble_coordinate_scale=np.ones(private),
        plastic_active=False,
        prepared_cu_integrals=(np.zeros((size, size)), np.zeros((size, size)), np.zeros((size, 6))))
    coupon = role == "whole_coupon"
    return assembly.MacroOperand(
        name=role, role=role, operator=layer, thickness_mm=.0485 if coupon else .0015,
        area_mm2=.000144, reference_y_mm=.02425 if coupon else .0003,
        orientation=1 if coupon else -1,
        included_regions=("bi", "beol", "si") if coupon else ("bi",),
        birth_q6=np.zeros(6), age_offset_s=100. if coupon else 0.)


def body(rows=None):
    rows = (operand("whole_coupon"), operand("backside_bi", private=2)) if rows is None else rows
    return assembly.build_prepared_body(rows, si_thickness_um=37., global_reference_y_mm=.025,
                                        coordinate_scale=np.ones(6))


@pytest.mark.parametrize("sign", [-1, 1])
def test_datum_orientation_and_work(sign):
    transform = assembly.macro_transform(local_reference_y_mm=.0497,
                                         global_reference_y_mm=.025, orientation=sign)
    q = np.arange(1., 7.) * 1e-3
    local = transform @ q
    np.testing.assert_allclose(local[:3], q[:3] + .0247 * q[3:])
    np.testing.assert_allclose(local[3:], sign * q[3:])
    k = np.diag(np.arange(1., 7.))
    f = np.arange(6.) * .01
    np.testing.assert_allclose(q @ (transform.T @ k @ transform) @ q, local @ k @ local)
    np.testing.assert_allclose(q @ (transform.T @ f), local @ f)
    assert not transform.flags.writeable


def test_datum_change_composition():
    a = assembly.macro_transform(local_reference_y_mm=.04, global_reference_y_mm=.01, orientation=-1)
    b = assembly.macro_transform(local_reference_y_mm=.01, global_reference_y_mm=-.02, orientation=1)
    c = assembly.macro_transform(local_reference_y_mm=.04, global_reference_y_mm=-.02, orientation=-1)
    np.testing.assert_allclose(a @ b, c)


def test_real_generic_topology_preserves_modes_births_and_independent_states():
    rows = (operand("whole_coupon"), operand("backside_bi", private=2))
    rows = (replace(rows[0], birth_q6=np.arange(6.) * 1e-4), rows[1])
    value = body(rows)
    assert value.topology.generalized_size == 6 and value.topology.port_rank == 0
    assert value.topology.interface_slices == ()
    assert [i.operator.bubble_size for i in value.topology.instances] == [3, 2]
    assert [i.operator.age_offset_s for i in value.topology.instances] == [100., 0.]
    assert value.topology.instances[0].operator is not value.topology.instances[1].operator
    assert value.evidence["prepared_total_um"] == pytest.approx(50.)
    assert not value.evidence["microface_compatible"]
    assert not value.evidence["paired_prepared_validation_done"]
    assert not value.evidence["paired_stack_validation_done"]
    assert not value.evidence["all_orders_admitted"]
    first, second = value.initial_state(np.ones(6)), value.initial_state()
    for name in first.layers:
        np.testing.assert_array_equal(first.layers[name].birth_macro_coordinates,
                                      rows[0].birth_q6 if name == "whole_coupon" else np.zeros(6))
        assert first.layers[name] is not second.layers[name]
        assert not np.shares_memory(first.layers[name].bubble_coordinates, second.layers[name].bubble_coordinates)
    np.testing.assert_allclose(value.topology.instances[1].macro_transform[:3, 3:], .0247 * np.eye(3))


def test_real_native_trial_and_commit_six_macro_no_port_path():
    value = body()
    initial = value.initial_state()
    trial = value.trial(initial, 100., 1., elapsed_time_s=1., creep_active=False,
                        plastic_active=False, linear_solver="assembled_dense", max_iterations=8)
    committed = value.commit(trial)
    np.testing.assert_array_equal(initial.generalized_coordinates, np.zeros(6))
    assert np.linalg.norm(committed.generalized_coordinates) > 0
    assert set(committed.layers) == {"whole_coupon", "backside_bi"}
    k, f = np.zeros((6, 6)), np.zeros(6)
    for instance in value.topology.instances:
        t = instance.macro_transform
        k += t.T @ t
        f += t.T @ instance.operator.reference_thermal_load(100.)[:6]
    np.testing.assert_allclose(committed.generalized_coordinates, np.linalg.solve(k, f), rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "Si", "area", "height", "orientation", "birth", "ports"])
def test_inventory_rejects_invalid_inputs(kind):
    rows = [operand("whole_coupon"), operand("backside_bi", private=2)]
    if kind == "missing": rows.pop()
    elif kind == "duplicate": rows[1] = rows[0]
    elif kind == "Si": rows[0] = replace(rows[0], included_regions=("si",))
    elif kind == "area": rows[1] = replace(rows[1], area_mm2=.001)
    elif kind == "height": rows[0] = replace(rows[0], thickness_mm=.05)
    elif kind == "orientation": rows[1] = replace(rows[1], orientation=1)
    elif kind == "birth": rows[0] = replace(rows[0], birth_q6=[float("nan")] * 6)
    elif kind == "ports": rows[1].operator.macro_size = 134
    with pytest.raises(ValueError): body(rows)


@pytest.mark.parametrize("changes", [{"orientation": True}, {"orientation": 0},
                                     {"local_reference_y_mm": float("nan")}])
def test_invalid_transform(changes):
    fields = dict(local_reference_y_mm=0., global_reference_y_mm=0., orientation=1)
    fields.update(changes)
    with pytest.raises(ValueError): assembly.macro_transform(**fields)
