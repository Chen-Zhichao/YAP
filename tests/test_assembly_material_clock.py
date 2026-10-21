"""Layer-clock forwarding against the existing tiny real-Cu operator.

These are constitutive/adapter checks, not production-geometry qualification.
"""

from dataclasses import asdict, fields, is_dataclass, replace
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


NATIVE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "Ansys_Verify"
    / "12_stack_model_validation"
)
if not NATIVE_ROOT.is_dir():
    pytest.skip("sibling native mechanics repository is not installed", allow_module_level=True)
if str(NATIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(NATIVE_ROOT))

from test_exp12_free_membrane_bonding import _operator
from W2W.assembly_mechanics import LayerClockOperator
from W2W import assembly_mechanics as assembly
import mechanics_runtime as runtime


TEMPERATURE_C = 350.0
CU_FIELDS = (
    "plastic_strain", "creep_strain",
    "equivalent_plastic_strain", "equivalent_creep_strain",
)


@pytest.fixture
def material():
    base = _operator(21)
    # Change only creep parameters so the fixture's elastic closures stay valid.
    base.cu_card = replace(
        base.cu_card,
        creep_law="modified_time_hardening",
        creep_prefactor_mpa_minus_n_per_s=1.43e10,
        creep_exponent=2.5,
        creep_time_exponent=-0.9,
        creep_activation_term_k=23695.0,
    )
    state = base.initial_state(
        birth_macro_coordinates=np.linspace(-2e-5, 3e-5, base.macro_size)
    )
    macro = np.linspace(-0.006, 0.012, base.macro_size)
    bubble = np.array([0.002, -0.001])
    return base, state, macro, bubble


def _assert_close(actual, expected):
    if is_dataclass(expected):
        for field in fields(expected):
            _assert_close(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(expected, tuple):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            _assert_close(value, reference)
    else:
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-13)


def _assert_unchanged(actual, expected):
    for name in ("bubble_coordinates", "birth_macro_coordinates", "birth_bubble_coordinates"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for name in CU_FIELDS:
        np.testing.assert_array_equal(
            getattr(actual.cu_state, name), getattr(expected.cu_state, name)
        )


def _evaluate(operator, state, macro, bubble, *, endpoint, dt=90.0, plastic=False):
    return operator.evaluate(
        macro, bubble, TEMPERATURE_C, dt, state,
        creep_active=True, plastic_active=plastic, elapsed_time_s=endpoint,
    )


def test_default_offset_and_delegated_operator_contract(material):
    base, state, macro, bubble = material
    proxy = LayerClockOperator(base)
    assert proxy.age_offset_s == 0.0
    for name in ("macro_size", "bubble_size", "coordinate_size", "cu_point_count"):
        assert getattr(proxy, name) == getattr(base, name)
    for name in ("cu_card", "strain_maps", "weights_mm3", "macro_coordinate_scale"):
        assert getattr(proxy, name) is getattr(base, name)
    _assert_close(proxy.reference_stiffness(TEMPERATURE_C),
                  base.reference_stiffness(TEMPERATURE_C))
    _assert_close(proxy.initial_state(), base.initial_state())
    _assert_close(_evaluate(proxy, state, macro, bubble, endpoint=390.0),
                  _evaluate(base, state, macro, bubble, endpoint=390.0))


@pytest.mark.parametrize("offset", [4000.0, 9000.0])
@pytest.mark.parametrize("plastic", [False, True])
def test_evaluate_uses_physical_endpoint_once_without_mutating_history(material, offset, plastic):
    base, state, macro, bubble = material
    before = state.copy()
    proxy = LayerClockOperator(base, age_offset_s=offset)
    actual = _evaluate(proxy, state, macro, bubble, endpoint=550.0, plastic=plastic)
    expected = _evaluate(base, state, macro, bubble, endpoint=offset + 550.0, plastic=plastic)
    unshifted = _evaluate(base, state, macro, bubble, endpoint=550.0, plastic=plastic)
    doubled = _evaluate(base, state, macro, bubble, endpoint=2 * offset + 550.0, plastic=plastic)
    _assert_close(actual, expected)
    assert np.max(actual.cu_trial.creep_increment) > 0.0
    for wrong in (unshifted, doubled):
        assert np.max(np.abs(actual.cu_trial.creep_increment - wrong.cu_trial.creep_increment)) > 1e-8
    _assert_unchanged(state, before)
    assert proxy.age_offset_s == offset


def test_evaluate_many_matches_batched_and_individual_base_oracles(material):
    base, state, macro, bubble = material
    offset, endpoint = 4000.0, 550.0
    proxy = LayerClockOperator(base, age_offset_s=offset)
    states = [state.copy(), state.copy()]
    states[1].cu_state.equivalent_plastic_strain[:] = 0.002
    macros = [macro, 0.8 * macro]
    bubbles = [bubble, -bubble]
    before = [value.copy() for value in states]
    actual = proxy.evaluate_many(
        macros, bubbles, TEMPERATURE_C, 90.0, states,
        creep_active=True, plastic_active=False, elapsed_time_s=endpoint,
    )
    expected = base.evaluate_many(
        macros, bubbles, TEMPERATURE_C, 90.0, states,
        creep_active=True, plastic_active=False, elapsed_time_s=offset + endpoint,
    )
    _assert_close(actual, expected)
    for index, value in enumerate(actual):
        _assert_close(value, _evaluate(base, states[index], macros[index], bubbles[index],
                                      endpoint=offset + endpoint))
        _assert_unchanged(states[index], before[index])
    assert proxy.age_offset_s == offset


@pytest.mark.parametrize("method", ["tangent_action", "assembled_coordinate_tangent"])
def test_tangents_use_the_same_physical_age_as_residual(material, method):
    base, state, macro, bubble = material
    offset, endpoint = 4000.0, 550.0
    proxy = LayerClockOperator(base, age_offset_s=offset)
    before = state.copy()
    args = (macro, bubble)
    if method == "tangent_action":
        args += (np.linspace(-0.5, 0.75, base.coordinate_size),)
    args += (TEMPERATURE_C, 90.0, state)
    options = dict(creep_active=True, plastic_active=False)
    actual = getattr(proxy, method)(*args, elapsed_time_s=endpoint, **options)
    expected = getattr(base, method)(*args, elapsed_time_s=offset + endpoint, **options)
    wrong = getattr(base, method)(*args, elapsed_time_s=endpoint, **options)
    _assert_close(actual, expected)
    assert np.max(np.abs(actual - wrong)) > 1e-8
    _assert_unchanged(state, before)
    assert proxy.age_offset_s == offset


def test_two_physical_layers_can_share_base_without_synchronizing_ages(material):
    base, state, macro, bubble = material
    old = LayerClockOperator(base, age_offset_s=9000.0)
    incoming = LayerClockOperator(base, age_offset_s=4000.0)
    old_result = _evaluate(old, state, macro, bubble, endpoint=550.0)
    incoming_result = _evaluate(incoming, state, macro, bubble, endpoint=550.0)
    _assert_close(old_result, _evaluate(base, state, macro, bubble, endpoint=9550.0))
    _assert_close(incoming_result, _evaluate(base, state, macro, bubble, endpoint=4550.0))
    assert np.max(np.abs(old_result.cu_trial.creep_increment
                         - incoming_result.cu_trial.creep_increment)) > 1e-8
    _assert_close(_evaluate(old, state, macro, bubble, endpoint=550.0), old_result)
    assert old is not incoming
    assert old.age_offset_s == 9000.0
    assert incoming.age_offset_s == 4000.0


def test_new_bi_at_300_seconds_with_schedule_origin_300_has_zero_offset(material):
    base, state, macro, bubble = material
    entry_age = schedule_start = 300.0
    proxy = LayerClockOperator(base, age_offset_s=entry_age - schedule_start)
    actual = _evaluate(proxy, state, macro, bubble, endpoint=390.0)
    _assert_close(actual, _evaluate(base, state, macro, bubble, endpoint=390.0))
    doubled_prebond = _evaluate(base, state, macro, bubble, endpoint=690.0)
    assert np.max(np.abs(actual.cu_trial.creep_increment
                         - doubled_prebond.cu_trial.creep_increment)) > 1e-8
    assert proxy.age_offset_s == 0.0


@pytest.mark.parametrize("entry_age", [300.0, 5300.0])
def test_discarded_trial_and_cutback_endpoints_keep_offset_fixed(material, entry_age):
    base, state, macro, bubble = material
    schedule_start = 300.0
    offset = entry_age - schedule_start
    proxy = LayerClockOperator(base, age_offset_s=offset)
    before = state.copy()
    # A rejected full step is evaluated, then discarded without material commit.
    full = _evaluate(proxy, state, macro, bubble, endpoint=390.0)
    _assert_close(full, _evaluate(base, state, macro, bubble, endpoint=entry_age + 90.0))
    _assert_unchanged(state, before)
    _assert_close(_evaluate(proxy, state, macro, bubble, endpoint=390.0), full)

    actual_state, expected_state = state.copy(), state.copy()
    for endpoint, physical_endpoint in ((345.0, entry_age + 45.0), (390.0, entry_age + 90.0)):
        actual = _evaluate(proxy, actual_state, macro, bubble, endpoint=endpoint, dt=45.0)
        expected = _evaluate(base, expected_state, macro, bubble,
                             endpoint=physical_endpoint, dt=45.0)
        _assert_close(actual, expected)
        # Commit actual kernel histories along the same prescribed-coordinate path.
        actual_state.cu_state = actual.cu_trial.state.copy()
        expected_state.cu_state = expected.cu_trial.state.copy()
        actual_state.bubble_coordinates = bubble.copy()
        expected_state.bubble_coordinates = bubble.copy()
        assert proxy.age_offset_s == offset
    _assert_close(actual_state, expected_state)
    assert np.max(actual_state.cu_state.equivalent_creep_strain) > 0.0
    _assert_unchanged(state, before)


@pytest.fixture
def corrected_material(monkeypatch, material):
    source = runtime.ROOT.parent / "operator_repair_20261015/source/exp12_cu_material_point.py"
    spec = importlib.util.spec_from_file_location("_assembly_clock_corrected_cu", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    base, state, macro, bubble = material
    card = module.CuCard(**asdict(base.cu_card))
    kernels = SimpleNamespace(update_batch=module.trial_update_batch,
        tangent_batch=module.trial_tangent_action_batch, elastic_stiffness=module.isotropic_stiffness)
    binding = runtime.bind_corrected_material(material_kernels=kernels, cu_card=card,
        source_path=source, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope="synthetic_fixture")
    base.material_kernels, base.cu_card = kernels, card
    runtime.attach_corrected_material_binding(base, binding)
    return base, state, macro, bubble, binding


def test_corrected_binding_provenance_has_explicit_nonadmission_scope(corrected_material):
    base, _, _, _, binding = corrected_material
    proof = runtime.corrected_material_provenance(binding)
    assert json.loads(json.dumps(proof)) == proof
    assert proof["clock_semantics"] == runtime.CORRECTED_MATERIAL_CLOCK
    assert proof["cu_card"] == asdict(base.cu_card)
    assert set(proof["kernels"]) == {"update_batch", "tangent_batch", "elastic_stiffness"}
    assert proof["scope"] == "synthetic_fixture"
    assert proof["assembly_execution_supported"] is proof["physical_readiness"] is False


@pytest.mark.parametrize("fault", ["hash", "source", "scope", "legacy_card"])
def test_corrected_binding_rejects_unbound_implementations(corrected_material, fault):
    base, _, _, _, binding = corrected_material
    options = dict(material_kernels=base.material_kernels, cu_card=base.cu_card,
        source_path=binding.source_path, source_sha256=binding.source_sha256)
    if fault == "hash":
        options["source_sha256"] = "0" * 64
    elif fault == "source":
        options["source_path"] = runtime.__file__
    elif fault == "scope":
        options["scope"] = "production"
    else:
        options["cu_card"] = material_card = _operator(22).cu_card
        assert not hasattr(material_card, "creep_step_coefficient")
    with pytest.raises(ValueError):
        runtime.bind_corrected_material(**options)


@pytest.mark.parametrize("fault", [None, "card", "kernels", "missing_kernels"])
def test_corrected_clone_retains_exact_binding_or_fails(corrected_material, fault):
    base, _, _, _, binding = corrected_material
    clone = copy.copy(base)
    del clone.corrected_material_binding
    if fault == "card":
        clone.cu_card = replace(base.cu_card)
    elif fault == "kernels":
        clone.material_kernels = SimpleNamespace(**vars(base.material_kernels))
    elif fault == "missing_kernels":
        clone.material_kernels = None
    if fault is not None:
        with pytest.raises(ValueError):
            runtime.preserve_corrected_material_binding(base, clone)
        assert not hasattr(clone, "corrected_material_binding")
    else:
        assert runtime.preserve_corrected_material_binding(base, clone) is clone
        assert clone.corrected_material_binding is binding
        assert LayerClockOperator(clone).corrected_material_binding is binding


def test_corrected_binding_detects_mutated_kernel_namespace(corrected_material, monkeypatch):
    base, _, _, _, binding = corrected_material
    monkeypatch.setattr(base.material_kernels, "update_batch", base.material_kernels.tangent_batch)
    with pytest.raises(ValueError, match="implementation changed"):
        runtime.corrected_material_provenance(binding)


def test_corrected_binding_detects_replaced_clock_implementation(corrected_material, monkeypatch):
    base, _, _, _, binding = corrected_material
    monkeypatch.setattr(type(base.cu_card), "creep_step_coefficient", type(base.cu_card).creep_coefficient)
    with pytest.raises(ValueError, match="clock implementation changed"):
        runtime.corrected_material_provenance(binding)


def test_corrected_runtime_rejects_lost_binding(corrected_material):
    base, _, _, _, _ = corrected_material
    del base.corrected_material_binding
    with pytest.raises(ValueError, match="lost"):
        assembly.require_corrected_material_runtime(SimpleNamespace(operators={"bi": base}))
    with pytest.raises(ValueError, match="lost"):
        LayerClockOperator(base)


@pytest.mark.parametrize("enabled", [False, True])
def test_corrected_runtime_rejects_installed_global_acceleration_even_without_preview(
        corrected_material, monkeypatch, enabled):
    base, _, _, _, _ = corrected_material
    exact, _ = assembly._corrected_native_modules()
    monkeypatch.setattr(exact, "_INSTALLED", True)
    calls = []
    monkeypatch.setattr(exact, "uninstall_exact_acceleration", lambda: calls.append("uninstall"))
    monkeypatch.setattr(exact, "install_exact_acceleration", lambda: calls.append("install"))
    with pytest.raises(RuntimeError, match="not uninstalled"):
        with assembly._reference_runtime(enabled, library=SimpleNamespace(operators={"bi": base})):
            pytest.fail("accelerated corrected context must not be entered")
    assert calls == []
    assert exact._INSTALLED is True


def test_corrected_runtime_rejects_actual_method_patch_with_clear_flag(corrected_material, monkeypatch):
    base, _, _, _, _ = corrected_material
    exact, reduced = assembly._corrected_native_modules()
    monkeypatch.setattr(exact, "_INSTALLED", False)
    monkeypatch.setattr(reduced.AllGaussReducedFE2Layer, "tangent_action", exact._accelerated_tangent_action)
    with pytest.raises(RuntimeError, match="unpatched native runtime"):
        assembly.require_corrected_material_runtime(SimpleNamespace(operators={"bi": base}))
    assert reduced.AllGaussReducedFE2Layer.tangent_action is exact._accelerated_tangent_action


def test_corrected_context_refuses_to_enable_legacy_acceleration(corrected_material):
    base, _, _, _, _ = corrected_material
    library = SimpleNamespace(operators={"bi": base})
    with assembly._reference_runtime(False, library=library) as acceleration:
        assert acceleration is None
    with pytest.raises(RuntimeError, match="cannot run"):
        with assembly._reference_runtime(True, library=library):
            pytest.fail("legacy acceleration must not be installed")


def test_legacy_disabled_context_still_does_not_import_acceleration(monkeypatch):
    monkeypatch.setattr(assembly, "native", lambda name: pytest.fail("legacy disabled context imported native"))
    with assembly._reference_runtime(False) as acceleration:
        assert acceleration is None


def test_corrected_clock_requires_explicit_endpoint_and_preserves_ages(corrected_material):
    base, state, macro, bubble, binding = corrected_material
    old, incoming = LayerClockOperator(base, 9000.), LayerClockOperator(base, 4000.)
    before = state.copy()
    assert old.corrected_material_binding is incoming.corrected_material_binding is binding
    for proxy in (old, incoming):
        endpoint = 550. + proxy.age_offset_s
        result = _evaluate(proxy, state, macro, bubble, endpoint=550.)
        expected = _evaluate(base, state, macro, bubble, endpoint=endpoint)
        _assert_close(result, expected)
        many = proxy.evaluate_many([macro, macro], [bubble, bubble], 350., 90., [state, state],
            creep_active=True, plastic_active=False, elapsed_time_s=550.)
        for value in many:
            _assert_close(value, expected)
        _assert_close(proxy.tangent_action(macro, bubble, np.ones(base.coordinate_size), 350., 90., state,
            creep_active=True, plastic_active=False, elapsed_time_s=550.),
            base.tangent_action(macro, bubble, np.ones(base.coordinate_size), 350., 90., state,
                creep_active=True, plastic_active=False, elapsed_time_s=endpoint))
    with pytest.raises(ValueError, match="explicit physical endpoint"):
        LayerClockOperator(base).evaluate(macro, bubble, 350., 90., state, creep_active=True)
    _assert_unchanged(state, before)


def test_corrected_proxy_detects_acceleration_installed_after_construction(corrected_material, monkeypatch):
    base, state, macro, bubble, _ = corrected_material
    proxy = LayerClockOperator(base, 4000.)
    exact, _ = assembly._corrected_native_modules()
    monkeypatch.setattr(exact, "_INSTALLED", True)
    with pytest.raises(RuntimeError, match="not uninstalled"):
        _evaluate(proxy, state, macro, bubble, endpoint=550.)


def test_corrected_proxy_trial_forwards_clock_once(corrected_material, monkeypatch):
    base, _, _, _, _ = corrected_material
    monkeypatch.setattr(base, "trial", lambda **kwargs: kwargs)
    result = LayerClockOperator(base, 4000.).trial(elapsed_time_s=550.)
    assert result["elapsed_time_s"] == 4550.


def test_corrected_clock_cutbacks_and_assembled_tangent_match_oracle(corrected_material):
    base, state, macro, bubble, binding = corrected_material
    proxy = LayerClockOperator(base, 5000.)
    before = state.copy()
    full = _evaluate(proxy, state, macro, bubble, endpoint=390.)
    _assert_close(full, _evaluate(base, state, macro, bubble, endpoint=5390.))
    _assert_unchanged(state, before)
    actual_state, expected_state = state.copy(), state.copy()
    for endpoint in (345., 390.):
        actual = _evaluate(proxy, actual_state, macro, bubble, endpoint=endpoint, dt=45.)
        expected = _evaluate(base, expected_state, macro, bubble, endpoint=endpoint+5000., dt=45.)
        _assert_close(actual, expected)
        _assert_close(proxy.assembled_coordinate_tangent(macro, bubble, 350., 45., actual_state,
            creep_active=True, plastic_active=False, elapsed_time_s=endpoint),
            base.assembled_coordinate_tangent(macro, bubble, 350., 45., expected_state,
                creep_active=True, plastic_active=False, elapsed_time_s=endpoint+5000.))
        actual_state.cu_state = actual.cu_trial.state.copy()
        expected_state.cu_state = expected.cu_trial.state.copy()
    _assert_close(actual_state, expected_state)
    _assert_unchanged(state, before)
    assert proxy.corrected_material_binding is binding
    assert proxy.age_offset_s == 5000.
