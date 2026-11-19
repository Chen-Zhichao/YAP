"""Real Cu35/483 parity and binding guards; not geometry/history admission."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import mechanics_runtime as runtime
from W2W import component_compiled_material as compiled
from W2W.assembly_mechanics import LayerClockOperator, native, require_corrected_material_runtime


CU_FIELDS = ("plastic_strain", "creep_strain", "equivalent_plastic_strain", "equivalent_creep_strain")
TRIAL_FIELDS = ("stress_mpa", "plastic_increment", "creep_increment",
                "plastic_dissipation_mpa", "creep_dissipation_mpa")


@pytest.fixture(scope="module")
def material():
    cache = Path(os.environ["NUMBA_CACHE_DIR"])
    result = compiled.create_compiled_material(
        cu_source_path=compiled.SOURCE_ROOT / "exp12_cu_material_point.py",
        cu_source_sha256=compiled.SOURCE_PINS["exp12_cu_material_point.py"],
        acceleration_source_path=compiled.SOURCE_ROOT / "exp12_all_gauss_exact_acceleration.py",
        acceleration_source_sha256=compiled.SOURCE_PINS["exp12_all_gauss_exact_acceleration.py"],
        native_exact_module=native("exp12_all_gauss_exact_acceleration"),
        reduced_module=native("exp12_all_gauss_reduced_fe2"), cache_directory=cache,
        card_parameters=dict(anneal_c=350., e_room_mpa=131000., e_anneal_mpa=131000.,
            alpha_room_per_c=17.6e-6, alpha_anneal_per_c=17.6e-6,
            creep_law="modified_time_hardening", creep_prefactor_mpa_minus_n_per_s=1.43e10,
            creep_activation_term_k=23695., creep_exponent=2.5, creep_time_exponent=-.9),
        scope="synthetic_fixture")
    return result


def validate(material):
    runtime._validate_corrected_material_binding(material.binding,
        material_kernels=material.material_kernels, cu_card=material.cu_card)


def assert_trial(actual, expected):
    for name in TRIAL_FIELDS:
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), rtol=3e-10, atol=2e-11,
                                   err_msg="unchanged 483 finite-root point comparison: " + name)
    for name in CU_FIELDS:
        np.testing.assert_allclose(getattr(actual.state, name), getattr(expected.state, name), rtol=3e-10, atol=2e-13)


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("readonly", [False, True])
def test_full_point_histories_and_tangents(material, threads, readonly):
    import numba

    previous = numba.get_num_threads()
    numba.set_num_threads(threads)
    try:
        cu, card = material.python_reference, material.cu_card
        rng = np.random.default_rng(14)
        strain = rng.normal(size=(64, 6)) * .0006
        direction = rng.normal(size=strain.shape) * 1e-4
        state = cu.CuBatchState.zeros(64)
        paths = [(25., 0., 300., False, False, 0.),
                 (350., 20., 950., False, True, 1.),
                 (350., 90., 1040., True, True, 1.),
                 (350., 90., 5040., True, True, .8),
                 (350., 45., 4995., True, True, .8),
                 (350., 45., 5040., True, True, .8),
                 (25., 0., 5200., False, True, -.3)]
        for temperature, dt, endpoint, creep, plastic, factor in paths:
            before = {name: getattr(state, name).copy() for name in CU_FIELDS}
            for name in CU_FIELDS:
                getattr(state, name).flags.writeable = not readonly
            values = np.ascontiguousarray(strain * factor)
            values.flags.writeable = not readonly
            kwargs = dict(card=card, creep_active=creep, plastic_active=plastic, elapsed_time_s=endpoint)
            expected = cu.trial_update_batch(values, temperature, dt, state, **kwargs)
            actual = material.material_kernels.update_batch(values, temperature, dt, state, **kwargs)
            repeated = material.material_kernels.update_batch(values, temperature, dt, state, **kwargs)
            assert_trial(actual, expected)
            assert_trial(repeated, expected)
            tangent = material.material_kernels.tangent_batch(values, direction, temperature, dt, state, **kwargs)
            np.testing.assert_array_equal(tangent,
                cu.trial_tangent_action_batch(values, direction, temperature, dt, state, **kwargs))
            for name in CU_FIELDS:
                np.testing.assert_array_equal(getattr(state, name), before[name])
            if not (dt == 90. and endpoint == 5040.):
                state = actual.state.copy()
        validate(material)
    finally:
        numba.set_num_threads(previous)


def test_explicit_provenance_and_no_native_global_install(material):
    validate(material)
    record = runtime.corrected_material_provenance(material.binding)
    assert isinstance(material.binding, runtime.CorrectedMaterialBinding)
    assert record["schema"] == "yap.corrected_compiled_material_binding.v1"
    assert len(record["sources"]) == 2
    assert record["global_installation"] is False
    assert record["dynamic_disk_cache"] is False
    assert record["clock_semantics"] == runtime.CORRECTED_MATERIAL_CLOCK
    assert record["dispatchers"]["_numba_constitutive_kernel"]["targetoptions"]["parallel"] is True
    assert record["physical_readiness"] is record["assembly_execution_supported"] is False
    assert record["warmup_seconds"] == material.warmup_seconds


@pytest.mark.xfail(strict=True, reason="unchanged 483 32-iteration root is not strict Python35 parity at high strain")
def test_high_strain_python35_equivalence_limit_is_not_hidden(material):
    cu, card = material.python_reference, material.cu_card
    values = np.random.default_rng(14).normal(size=(64, 6)) * .006
    state = cu.CuBatchState.zeros(64)
    paths = ((25., 0., 300., False, False, 0.), (350., 20., 950., False, True, 1.),
             (350., 90., 1040., True, True, 1.), (350., 90., 5040., True, True, .8),
             (350., 45., 4995., True, True, .8), (350., 45., 5040., True, True, .8),
             (25., 0., 5200., False, True, -.3))
    for temperature, dt, endpoint, creep, plastic, factor in paths:
        kwargs = dict(card=card, creep_active=creep, plastic_active=plastic, elapsed_time_s=endpoint)
        expected = cu.trial_update_batch(values*factor, temperature, dt, state, **kwargs)
        actual = material.material_kernels.update_batch(values*factor, temperature, dt, state, **kwargs)
        assert_trial(actual, expected)
        state = actual.state.copy()


@pytest.mark.parametrize("mutation", ["kernel", "global", "dispatcher_code", "options", "card", "native"])
def test_mutated_bindings_are_refused(material, monkeypatch, mutation):
    implementation = material.binding.implementation
    with monkeypatch.context() as patch:
        if mutation == "kernel":
            patch.setattr(material.material_kernels, "update_batch", lambda *a, **kw: None)
        elif mutation == "global":
            patch.setattr(implementation.acceleration, "_numba_constitutive_kernel", lambda *a: None)
        elif mutation == "dispatcher_code":
            patch.setattr(implementation.acceleration._numba_constitutive_kernel.py_func, "__code__", (lambda: None).__code__)
        elif mutation == "options":
            patch.setitem(implementation.acceleration._numba_constitutive_kernel.targetoptions, "fastmath", True)
        elif mutation == "card":
            patch.setattr(type(material.cu_card), "creep_step_coefficient", lambda *a, **kw: 1.)
        else:
            patch.setattr(implementation.reduced, "trial_update_batch", lambda *a, **kw: None)
        with pytest.raises((ValueError, RuntimeError)):
            validate(material)
    validate(material)


def layer(material, kernels):
    reduced = material.binding.implementation.reduced
    rng = np.random.default_rng(20)
    maps = rng.normal(size=(8, 6, 3)) * .2
    weights = np.full(8, 1e-8)
    elastic = kernels.elastic_stiffness(*material.cu_card.elastic(25.))
    matrix = np.einsum("g,gsi,st,gtj->ij", weights, maps, elastic, maps) + np.eye(3)
    operator = reduced.AllGaussReducedFE2Layer(macro_size=2, bubble_size=1,
        reference_stiffness=lambda _t: matrix, reference_thermal_load=lambda _t: np.zeros(3),
        strain_maps=maps, weights_mm3=weights, macro_coordinate_scale=np.ones(2),
        bubble_coordinate_scale=np.ones(1), cu_card=material.cu_card, plastic_active=True,
        material_kernels=kernels)
    return operator


def test_native_proxy_unequal_ages_birth_fields_and_localization_are_isolated(material):
    cu = material.python_reference
    baseline = layer(material, SimpleNamespace(update_batch=cu.trial_update_batch,
        tangent_batch=cu.trial_tangent_action_batch, elastic_stiffness=cu.isotropic_stiffness))
    actual = runtime.attach_corrected_material_binding(layer(material, material.material_kernels), material.binding)
    require_corrected_material_runtime(SimpleNamespace(operators={"bi": actual}))
    state = actual.initial_state(birth_macro_coordinates=np.array([.001, -.0004]),
                                 birth_bubble_coordinates=np.array([.0002]))
    before = state.copy()
    maps = actual.strain_maps.copy()
    macro, bubble = np.array([.02, -.015]), np.array([.004])
    for offset in (0., 4000., 9000.):
        proxy = LayerClockOperator(actual, age_offset_s=offset)
        args = dict(creep_active=True, plastic_active=True)
        result = proxy.evaluate(macro, bubble, 350., 90., state, elapsed_time_s=1040., **args)
        expected = baseline.evaluate(macro, bubble, 350., 90., state, elapsed_time_s=1040.+offset, **args)
        assert_trial(result.cu_trial, expected.cu_trial)
        np.testing.assert_allclose(result.response, expected.response, rtol=1e-11, atol=1e-12)
        np.testing.assert_array_equal(proxy.tangent_action(macro, bubble, np.ones(3), 350., 90., state,
            elapsed_time_s=1040., **args), baseline.tangent_action(macro, bubble, np.ones(3), 350., 90., state,
            elapsed_time_s=1040.+offset, **args))
    for name in CU_FIELDS:
        np.testing.assert_array_equal(getattr(state.cu_state, name), getattr(before.cu_state, name))
    for name in ("birth_macro_coordinates", "birth_bubble_coordinates", "bubble_coordinates"):
        np.testing.assert_array_equal(getattr(state, name), getattr(before, name))
    np.testing.assert_array_equal(actual.strain_maps, maps)
    with pytest.raises(ValueError, match="endpoint"):
        LayerClockOperator(actual).evaluate(macro, bubble, 350., 90., state, creep_active=True)


def test_no_one_file_or_python_binding_substitution(material):
    with pytest.raises(ValueError, match="differs"):
        runtime.bind_corrected_material(material_kernels=material.material_kernels, cu_card=material.cu_card,
            source_path=compiled.SOURCE_ROOT / "exp12_cu_material_point.py",
            source_sha256=compiled.SOURCE_PINS["exp12_cu_material_point.py"])
    with pytest.raises(ValueError):
        compiled._source(compiled.SOURCE_ROOT / "exp12_cu_material_point.py", "f"*64,
                         "exp12_cu_material_point.py")
    with pytest.raises(ValueError):
        compiled._cache_directory(Path("/tmp/not-approved"))


def test_sealed_dispatcher_rejects_unreviewed_layout(material):
    state = material.python_reference.CuBatchState.zeros(8)
    with pytest.raises(TypeError, match="matching definition"):
        material.material_kernels.update_batch(np.asfortranarray(np.zeros((8, 6))), 25., 0., state,
            card=material.cu_card, creep_active=False, elapsed_time_s=300.)
    validate(material)


def test_native_global_installer_flag_is_refused(material, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(material.binding.implementation.exact, "_INSTALLED", True)
        with pytest.raises(ValueError):
            validate(material)
    validate(material)


def test_integrated_clock_is_not_endpoint_approximation(material):
    card = material.cu_card
    power = card.creep_time_exponent + 1.
    for endpoint in (1040., 5040., 10040.):
        dt = 90.
        scale = card.creep_prefactor_mpa_minus_n_per_s * np.exp(-card.creep_activation_term_k / 623.15)
        expected = scale * (endpoint**power - (endpoint-dt)**power) / (power*dt)
        actual = card.creep_step_coefficient(350., dt, elapsed_time_s=endpoint)
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=0.)
        assert abs(actual / card.creep_coefficient(350., elapsed_time_s=endpoint) - 1.) > 1e-3


def test_high_strain_thread_count_parity_is_exact(material):
    import numba

    previous = numba.get_num_threads()
    state = material.python_reference.CuBatchState.zeros(64)
    values = np.random.default_rng(14).normal(size=(64, 6)) * .006
    kwargs = dict(card=material.cu_card, creep_active=True, plastic_active=True, elapsed_time_s=1040.)
    try:
        numba.set_num_threads(1)
        expected = material.material_kernels.update_batch(values, 350., 90., state, **kwargs)
        numba.set_num_threads(4)
        actual = material.material_kernels.update_batch(values, 350., 90., state, **kwargs)
        for name in TRIAL_FIELDS:
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
        for name in CU_FIELDS:
            np.testing.assert_array_equal(getattr(actual.state, name), getattr(expected.state, name))
    finally:
        numba.set_num_threads(previous)
