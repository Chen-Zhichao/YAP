"""Tiny real-native formation controls, not coupon or assembly qualification."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import coupon_assembly as assembly
from W2W.assembly_mechanics import LayerClockOperator, native


SCALE = np.ones(6)
OPTIONS = dict(linear_solver="assembled_dense", convergence_measure="residual",
               tolerance=1e-10, max_iterations=32)
CU_FIELDS = ("plastic_strain", "creep_strain", "equivalent_plastic_strain",
             "equivalent_creep_strain")
BIRTH_FIELDS = ("birth_macro_coordinates", "birth_bubble_coordinates")


@pytest.fixture(scope="module")
def runtime():
    modules = dict(stack=native("exp12_all_gauss_periodic_stack"),
                   reduced=native("exp12_all_gauss_reduced_fe2"),
                   bonding=native("exp12_free_membrane_bonding"),
                   formation=native("run_exp12_free_membrane_sequential"))
    delivered = Path(__file__).resolve().parents[2] / "distribution/operators/components/runtime/native"
    for module in modules.values():
        assert Path(module.__file__).resolve().is_relative_to(delivered)
    return SimpleNamespace(**modules)


def operand(runtime, index=0, *, age=0.):
    k = np.diag(np.arange(2., 11.))
    for i in range(3):
        k[3 + i, 6 + i] = k[6 + i, 3 + i] = .25
    maps = np.zeros((2, 6, 9))
    maps[:, :, :6] = .01 * np.eye(6)
    maps[:, :3, 6:] = .03 * np.eye(3)
    layer = runtime.reduced.AllGaussReducedFE2Layer(
        macro_size=6, bubble_size=3,
        reference_stiffness=lambda t: k,
        reference_thermal_load=lambda t: (t - 25.) * np.arange(1., 10.) * 1e-6,
        strain_maps=maps, weights_mm3=np.array([1e-5, 2e-5]),
        macro_coordinate_scale=np.ones(6), bubble_coordinate_scale=np.ones(3),
        plastic_active=True)
    datum = .045 + .09 * index
    transform = assembly.macro_transform(local_reference_y_mm=datum,
                                         global_reference_y_mm=0., orientation=1)
    natural = (index + 1) * np.array([1e-4, -6e-5, 2e-5, 2e-3, -1e-3, 4e-4])
    return assembly.MacroOperand(
        name=f"body{index}", role="whole_prepared", operator=layer,
        thickness_mm=.09, area_mm2=.000144, reference_y_mm=datum, orientation=1,
        included_regions=("front_bi", "beol", "si", "back_bi"),
        birth_q6=transform @ natural, age_offset_s=age)


def build(runtime, rows):
    return assembly.build_whole_prepared_body(
        rows, global_reference_y_mm=0., coordinate_scale=SCALE, engine=runtime.stack)


def arrays(state):
    result = {"q": state.generalized_coordinates}
    for name, layer in state.layers.items():
        for field in ("bubble_coordinates", *BIRTH_FIELDS):
            result[name + "/" + field] = getattr(layer, field)
        for field in CU_FIELDS:
            result[name + "/" + field] = getattr(layer.cu_state, field)
    return result


def snapshot(state):
    return {key: value.copy() for key, value in arrays(state).items()}


def assert_unchanged(state, before):
    assert arrays(state).keys() == before.keys()
    for key, value in arrays(state).items():
        np.testing.assert_array_equal(value, before[key], err_msg=key)


def assert_frozen(before, after):
    assert before.layers.keys() == after.layers.keys()
    for name, old in before.layers.items():
        new = after.layers[name]
        for field in BIRTH_FIELDS:
            np.testing.assert_array_equal(getattr(old, field), getattr(new, field))
        for field in CU_FIELDS:
            np.testing.assert_array_equal(getattr(old.cu_state, field), getattr(new.cu_state, field))


def leaf(runtime, row):
    body = build(runtime, (row,))
    instance = body.topology.instances[0]
    q = np.linalg.solve(instance.macro_transform, row.birth_q6)
    state = body.initial_state(q)
    layer = state.layers[row.name]
    seed = 1. + row.reference_y_mm
    layer.birth_bubble_coordinates[:] = seed * np.array([1e-5, -2e-5, 3e-5])
    layer.bubble_coordinates[:] = layer.birth_bubble_coordinates + 1e-4
    for field, factor in (("plastic_strain", 1e-6), ("creep_strain", 2e-6)):
        getattr(layer.cu_state, field)[:] = seed * factor * np.array([1., -1., 0., .2, .3, -.1])
    layer.cu_state.equivalent_plastic_strain[:] = seed * np.array([.002, .003])
    layer.cu_state.equivalent_creep_strain[:] = seed * np.array([.004, .005])
    trial = body.trial(state, 25., 0., elapsed_time_s=7., creep_active=False,
                       plastic_active=False, **OPTIONS)
    free = body.commit(trial)
    assert_frozen(state, free)
    return body, free


def joined_template(lower, upper):
    instances = lower.topology.instances + upper.topology.instances
    return replace(lower.topology, instances=instances,
                   interface_slices=(slice(6, 6),) * (len(instances) + 1))


def test_single_body_real_native_flatten_and_release(runtime):
    row = operand(runtime, age=300.)
    body, free = leaf(runtime, row)
    top = body.topology
    assert top.port_rank == 0 and top.generalized_size == 6
    assert top.interface_slices == (slice(6, 6), slice(6, 6))
    assert top.gauge_constraint_matrix.shape == (0, 6)
    assert top.gauge_constraint_values.shape == (0,)
    assert top.coordinate_labels == assembly.LABELS
    assert not top.coordinate_scale.flags.writeable
    assert body.evidence["prepared_total_um"] == pytest.approx(90.)
    assert body.evidence["included_BI_layers_per_body"] == 2
    for flag in ("microface_compatible", "source_custody_verified_here",
                 "material_compatibility_verified_here", "paired_prepared_validation_done",
                 "paired_stack_validation_done", "assembly_accuracy_admitted",
                 "all_orders_admitted", "W2W_birth_removal_supported"):
        assert body.evidence[flag] is False
    before = snapshot(free)
    _, flat = runtime.bonding.flatten_free_membrane(
        top, free, elapsed_time_s=7., solver_options=OPTIONS)
    assert np.max(np.abs(flat.generalized_coordinates[3:])) <= 1e-12
    assert_frozen(free, flat)
    assert not np.allclose(flat.layers[row.name].bubble_coordinates,
                           free.layers[row.name].bubble_coordinates, rtol=0., atol=1e-10)
    _, released = runtime.formation._free_equilibrium_trial(
        top, flat, temperature_c=25., delta_time_s=0., elapsed_time_s=7.,
        plastic_active=False, solver_options=OPTIONS)
    assert_frozen(flat, released)
    np.testing.assert_allclose(released.generalized_coordinates, free.generalized_coordinates,
                               rtol=1e-7, atol=1e-10)
    assert_unchanged(free, before)


@pytest.mark.parametrize("order", ["left_aged_incoming", "right_aged_incoming_pair"])
def test_three_bodies_native_formation_two_orders(runtime, monkeypatch, order):
    rows = [operand(runtime, index, age=age) for index, age in enumerate((300., 0., 900.))]
    leaves = [leaf(runtime, row) for row in rows]
    originals = {i.name: i for body, _ in leaves for i in body.topology.instances}
    initial_states = [snapshot(state) for _, state in leaves]
    endpoints = {row.name: [] for row in rows}
    for body, _ in leaves:
        instance = body.topology.instances[0]
        base = instance.operator.base
        evaluate = base.evaluate

        def recorded(*args, _evaluate=evaluate, _name=instance.name, **kwargs):
            endpoints[_name].append(kwargs["elapsed_time_s"])
            return _evaluate(*args, **kwargs)

        monkeypatch.setattr(base, "evaluate", recorded)

    def join(lower_pair, upper_pair):
        lower, old = lower_pair
        upper, incoming = upper_pair
        old_before, incoming_before = snapshot(old), snapshot(incoming)
        template = joined_template(lower, upper)
        _, old_flat = runtime.bonding.flatten_free_membrane(
            lower.topology, old, elapsed_time_s=7., solver_options=OPTIONS)
        _, incoming_flat = runtime.bonding.flatten_free_membrane(
            upper.topology, incoming, elapsed_time_s=7., solver_options=OPTIONS)
        capture = runtime.bonding.capture_formation(
            lower.topology, old_flat, upper.topology, incoming_flat, template)
        for source_top, source_state in ((lower.topology, old_flat), (upper.topology, incoming_flat)):
            for source in source_top.instances:
                target = next(i for i in capture.topology.instances if i.name == source.name)
                old_layer = source_state.layers[source.name]
                new_layer = capture.committed.layers[source.name]
                assert new_layer is not old_layer and new_layer.cu_state is not old_layer.cu_state
                for field in ("bubble_coordinates", *BIRTH_FIELDS):
                    a, b = getattr(old_layer, field), getattr(new_layer, field)
                    np.testing.assert_array_equal(a, b)
                    assert not np.shares_memory(a, b)
                for field in CU_FIELDS:
                    a, b = getattr(old_layer.cu_state, field), getattr(new_layer.cu_state, field)
                    np.testing.assert_array_equal(a, b)
                    assert not np.shares_memory(a, b)
                local_flat = source.macro_transform @ source_state.generalized_coordinates + source.macro_offset
                if source.name in old_flat.layers:
                    np.testing.assert_array_equal(target.macro_offset, source.macro_offset)
                    assert not np.shares_memory(target.macro_offset, source.macro_offset)
                else:
                    np.testing.assert_allclose(target.macro_offset,
                                               local_flat - target.macro_transform @ capture.committed.generalized_coordinates,
                                               rtol=0., atol=1e-15)
                assert np.max(np.abs(target.macro_offset[3:])) <= 1e-13
        assert any(np.linalg.norm(i.macro_offset[:3]) > 1e-8 for i in capture.topology.instances)
        top, state, _, record = runtime.formation.form_free_pair(
            lower.topology, old, upper.topology, incoming, template,
            event="tiny_whole_prepared_formation",
            face_z_mm=next(row.reference_y_mm + row.thickness_mm / 2 for row in rows
                           if row.name == lower.topology.instances[-1].name),
            clock_times=(7., 7., 7.), footprint_mm=(10., 10.), cell_area_mm2=.000144,
            solver_options=OPTIONS)
        assert record["maximum_local_coordinate_jump"] <= 1e-13
        assert record["held_flat_local_coordinate_drift"] <= 1e-8
        changes = [row["history_increments"] for row in record["bodies_old_then_incoming"]]
        changes += [record["held_flat_history_increment"], record["release_history_increment"]]
        assert all(value == 0. for change in changes for value in change.values())
        for instance in top.instances:
            original = originals[instance.name]
            assert instance.operator is original.operator
            assert instance.operator.age_offset_s == original.operator.age_offset_s
            np.testing.assert_array_equal(instance.macro_transform, original.macro_transform)
            np.testing.assert_array_equal(instance.virgin_birth_macro_coordinates,
                                          original.virgin_birth_macro_coordinates)
            np.testing.assert_array_equal(instance.macro_offset,
                                          next(i.macro_offset for i in capture.topology.instances
                                               if i.name == instance.name))
        for child in (old, incoming):
            for name, layer in child.layers.items():
                child_only = replace(state, layers={name: state.layers[name]})
                assert_frozen(replace(child, layers={name: layer}), child_only)
        assert_unchanged(old, old_before)
        assert_unchanged(incoming, incoming_before)
        return replace(lower, topology=top), state

    if order == "left_aged_incoming":
        result = join(join(leaves[0], leaves[1]), leaves[2])
    else:
        result = join(leaves[0], join(leaves[1], leaves[2]))
    assert [i.name for i in result[0].topology.instances] == [row.name for row in rows]
    assert result[0].topology.interface_slices == (slice(6, 6),) * 4
    np.testing.assert_array_equal(result[0].topology.coordinate_scale, SCALE)
    for row, (_, state), before in zip(rows, leaves, initial_states):
        assert_unchanged(state, before)
        assert endpoints[row.name]
        assert set(endpoints[row.name]) == {7. + row.age_offset_s}


def test_explicit_global_datums_and_stable_supplied_clocks(runtime):
    row = operand(runtime, age=100.)
    first = LayerClockOperator(row.operator, age_offset_s=100.)
    second = LayerClockOperator(row.operator, age_offset_s=900.)
    rows = (replace(row, operator=first),
            replace(row, name="other", operator=second, age_offset_s=900., reference_y_mm=.225))
    body = build(runtime, rows)
    assert body.topology.instances[0].operator is first
    assert body.topology.instances[1].operator is second
    np.testing.assert_array_equal(body.topology.instances[1].macro_transform[:3, 3:], .225 * np.eye(3))
    assert body.topology.interface_slices == (slice(6, 6),) * 3
    with pytest.raises(ValueError, match="age must remain unchanged"):
        build(runtime, (replace(rows[0], age_offset_s=101.),))
    with pytest.raises(ValueError, match="own stable clock proxy"):
        build(runtime, (rows[0], replace(rows[0], name="duplicate_proxy")))


@pytest.mark.parametrize("namespace", ["short", "long", "observed_long"])
@pytest.mark.parametrize("matching_age", [True, False], ids=["matching_age", "mismatched_age"])
def test_actual_clock_namespace_aliases_preserve_identity_and_single_age(
        runtime, monkeypatch, namespace, matching_age):
    from W2W.assembly_mechanics import LayerClockOperator as ShortClock
    from YAP.W2W.assembly_mechanics import LayerClockOperator as LongClock

    class ObservedLong(LongClock):
        pass

    assert ShortClock is not LongClock
    clock_type = {"short": ShortClock, "long": LongClock, "observed_long": ObservedLong}[namespace]
    row = operand(runtime, age=1.23)
    base = row.operator
    proxy = clock_type(base, age_offset_s=1.23)
    row = replace(row, operator=proxy, age_offset_s=1.23 if matching_age else 1.24)
    if not matching_age:
        with pytest.raises(ValueError, match="age must remain unchanged"):
            build(runtime, (row,))
        return
    body = build(runtime, (row,))
    retained = body.topology.instances[0].operator
    assert retained is proxy
    assert retained.base is base
    assert retained.age_offset_s == 1.23
    endpoints = []
    evaluate = base.evaluate

    def recorded(*args, **kwargs):
        endpoints.append(kwargs["elapsed_time_s"])
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(base, "evaluate", recorded)
    body.trial(body.initial_state(), 25., 0., elapsed_time_s=7., creep_active=False,
               plastic_active=False, **OPTIONS)
    assert endpoints
    np.testing.assert_allclose(endpoints, 7. + 1.23, rtol=0., atol=1e-14)


@pytest.mark.parametrize("change", ["scale", "recenter", "operator", "birth"])
def test_native_embedding_refuses_changed_retained_contract(runtime, change):
    lower = build(runtime, (operand(runtime, 0),))
    upper = build(runtime, (operand(runtime, 1),))
    template = joined_template(lower, upper)
    if change == "scale":
        template = replace(template, coordinate_scale=2 * SCALE)
    else:
        instance = template.instances[1]
        if change == "recenter":
            instance = replace(instance, macro_transform=assembly.macro_transform(
                local_reference_y_mm=.045, global_reference_y_mm=0., orientation=1))
        elif change == "operator":
            instance = replace(instance, operator=LayerClockOperator(instance.operator.base))
        else:
            instance = replace(instance, virgin_birth_macro_coordinates=instance.virgin_birth_macro_coordinates + 1e-5)
        template = replace(template, instances=(template.instances[0], instance))
    with pytest.raises(ValueError):
        runtime.bonding._check_embedding(upper.topology, template, 1)


@pytest.mark.parametrize("kind", ["empty", "duplicate", "role", "regions", "extra_back",
                                  "thickness", "area", "mixed_area", "datum", "global_datum",
                                  "orientation", "bool_orientation", "age", "birth", "macro",
                                  "private", "scale"])
def test_whole_prepared_contract_rejections(runtime, kind):
    rows = [operand(runtime)]
    kwargs = dict(global_reference_y_mm=0., coordinate_scale=SCALE, engine=runtime.stack)
    if kind == "empty": rows = []
    elif kind == "duplicate": rows *= 2
    elif kind == "role": rows[0] = replace(rows[0], role="whole_coupon")
    elif kind == "regions": rows[0] = replace(rows[0], included_regions=("si",))
    elif kind == "extra_back":
        rows[0] = replace(rows[0], included_regions=rows[0].included_regions + ("back_bi",))
    elif kind == "thickness": rows[0] = replace(rows[0], thickness_mm=0.)
    elif kind == "area": rows[0] = replace(rows[0], area_mm2=float("nan"))
    elif kind == "mixed_area": rows.append(replace(operand(runtime, 1), area_mm2=.001))
    elif kind == "datum": rows[0] = replace(rows[0], reference_y_mm=float("inf"))
    elif kind == "global_datum": kwargs["global_reference_y_mm"] = float("nan")
    elif kind == "orientation": rows[0] = replace(rows[0], orientation=-1)
    elif kind == "bool_orientation": rows[0] = replace(rows[0], orientation=True)
    elif kind == "age": rows[0] = replace(rows[0], age_offset_s=-1.)
    elif kind == "birth": rows[0] = replace(rows[0], birth_q6=np.zeros(9))
    elif kind == "macro": rows[0].operator.macro_size = 134
    elif kind == "private": rows[0].operator.bubble_size = 2
    elif kind == "scale": kwargs["coordinate_scale"] = [0.] * 6
    with pytest.raises(ValueError):
        assembly.build_whole_prepared_body(rows, **kwargs)
