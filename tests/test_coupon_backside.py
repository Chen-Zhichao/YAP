"""Small in-memory controls; no production files, runtime or solver imports."""
import copy
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tools import coupon_backside as backside


def fixture(n=12, macro_dependent=True):
    rng = np.random.default_rng(73)
    d = n - 3
    h = rng.normal(size=(d, d))
    h = h.T @ h + np.eye(d)
    k = np.zeros((n, n))
    k[:d, :d] = h
    f = np.r_[rng.normal(size=d), np.zeros(3)]
    b = np.zeros((8, 6, n))
    b[:, :, :d] = rng.normal(size=(8, 6, d))
    rigid = np.eye(n)[:, -3:]
    c = rng.normal(size=(3, n))
    c[:, -3:] = np.eye(3)
    if not macro_dependent:
        c[:, :6] = 0.
    u = np.vstack((np.eye(d), -c[:, :d]))
    metadata = dict(source_sha256="a"*64, gauge_sha256="b"*64, basis_sha256="c"*64,
                    original_coordinates=n, GP_count=8, macro_labels=list(backside.LABELS),
                    strain_order="native_engineering_six_unchanged", included_regions=["bi"],
                    source_area_mm2=.000144, reference_y_mm=.00075)
    return dict(stiffness=k, thermal_per_c=f, strain_maps=b, weights_mm3=np.linspace(.001, .002, 8),
                macro_embedding=u[:, :6].copy(), private_basis=u[:, 6:].copy(), gauge_rows=c,
                translation_basis=rigid, metadata=metadata, expected_metadata_sha256=backside.metadata_sha256(metadata),
                material_binding=object(), check=lambda: None, block_gp=3)


@pytest.mark.parametrize("macro_dependent", [False, True])
def test_exact_gauge_kernel_energy_strain_and_virtual_work(macro_dependent):
    inputs = fixture(macro_dependent=macro_dependent)
    snapshots = {key: value.copy() for key, value in inputs.items() if type(value) is np.ndarray}
    result = backside.transform(**inputs)
    u = result.source_from_reduced
    assert result.macro_size == 6 and result.bubble_size == 3
    np.testing.assert_array_equal(u[:6], np.c_[np.eye(6), np.zeros((6, 3))])
    np.testing.assert_allclose(inputs["gauge_rows"] @ u, 0., atol=1e-14)
    assert np.linalg.matrix_rank(u) == 9
    assert np.linalg.matrix_rank(inputs["stiffness"]) == 9
    np.linalg.cholesky(result.stiffness)
    x = np.arange(1., 10.) * .02
    source = u @ x
    np.testing.assert_allclose(x @ result.stiffness @ x, source @ inputs["stiffness"] @ source)
    np.testing.assert_allclose(x @ result.thermal_per_c, source @ inputs["thermal_per_c"])
    np.testing.assert_allclose(result.strain_maps @ x, inputs["strain_maps"] @ source)
    stress = np.arange(48.).reshape(8, 6) * .03
    old_force = np.einsum("gsi,gs,g->i", inputs["strain_maps"], stress, inputs["weights_mm3"])
    new_force = np.einsum("gsi,gs,g->i", result.strain_maps, stress, result.weights_mm3)
    np.testing.assert_allclose(new_force, u.T @ old_force)
    for key, original in snapshots.items():
        np.testing.assert_array_equal(inputs[key], original)
    for value in (result.stiffness, result.thermal_per_c, result.strain_maps,
                  result.weights_mm3, result.source_from_reduced):
        assert not value.flags.writeable
    assert result.material_binding is inputs["material_binding"]
    np.testing.assert_array_equal(result.weights_mm3, inputs["weights_mm3"])
    assert not np.shares_memory(result.weights_mm3, inputs["weights_mm3"])
    assert not result.evidence["production_ready"]
    assert not result.evidence["paired_physical_validation_done"]
    assert not result.evidence["elastic_condensation"]


def test_185_to_six_plus_176_dimensions_only_synthetic():
    result = backside.transform(**fixture(n=185))
    assert result.stiffness.shape == (182, 182)
    assert result.strain_maps.shape == (8, 6, 182)
    assert result.bubble_size == 176


def test_noncoordinate_translation_span_is_not_guessed():
    inputs = fixture()
    expected = backside.transform(**inputs)
    rng = np.random.default_rng(25)
    change = np.eye(12)
    change[6:, 6:] = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    inputs["stiffness"] = change.T @ inputs["stiffness"] @ change
    inputs["thermal_per_c"] = change.T @ inputs["thermal_per_c"]
    inputs["strain_maps"] = inputs["strain_maps"] @ change
    inputs["gauge_rows"] = inputs["gauge_rows"] @ change
    for key in ("macro_embedding", "private_basis", "translation_basis"):
        inputs[key] = change.T @ inputs[key]
    actual = backside.transform(**inputs)
    np.testing.assert_allclose(actual.stiffness, expected.stiffness, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.thermal_per_c, expected.thermal_per_c, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.strain_maps, expected.strain_maps, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ["bad_q6", "private_q6", "missing_private_mode", "gauge_violation",
                                  "gauge_on_compliant_mode", "dependent_gauge", "rigid_q6",
                                  "rigid_stiffness", "rigid_thermal", "rigid_strain", "negative_energy",
                                  "nonsymmetric", "negative_weight", "nonfinite_map"])
def test_rejects_non_gauge_changes(kind):
    inputs = fixture()
    if kind == "bad_q6": inputs["macro_embedding"][0, 0] = 2.
    elif kind == "private_q6": inputs["private_basis"][0, 0] = 1.
    elif kind == "missing_private_mode": inputs["private_basis"][:, 0] = 0.
    elif kind == "gauge_violation": inputs["macro_embedding"][-1, 0] += 1.
    elif kind == "gauge_on_compliant_mode": inputs["gauge_rows"][:, -3:] = 0.
    elif kind == "dependent_gauge": inputs["gauge_rows"][0] = inputs["gauge_rows"][1]
    elif kind == "rigid_q6": inputs["translation_basis"][0, 0] = 1.
    elif kind == "rigid_stiffness": inputs["stiffness"][-1, -1] = 1.
    elif kind == "rigid_thermal": inputs["thermal_per_c"][-1] = 1.
    elif kind == "rigid_strain": inputs["strain_maps"][0, 0, -1] = 1.
    elif kind == "negative_energy": inputs["stiffness"] *= -1.
    elif kind == "nonsymmetric": inputs["stiffness"][0, 1] += 1.
    elif kind == "negative_weight": inputs["weights_mm3"][0] = -1.
    elif kind == "nonfinite_map": inputs["strain_maps"][0, 0, 0] = np.nan
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        backside.transform(**inputs)


def test_metadata_binding_and_BI_only_inventory():
    inputs = fixture()
    inputs["metadata"] = copy.deepcopy(inputs["metadata"])
    inputs["metadata"]["reference_y_mm"] += .001
    with pytest.raises(ValueError, match="SHA256"):
        backside.transform(**inputs)
    inputs["metadata"]["included_regions"] = ["bi", "beol"]
    inputs["expected_metadata_sha256"] = backside.metadata_sha256(inputs["metadata"])
    with pytest.raises(ValueError, match="BI-only"):
        backside.transform(**inputs)


def test_guard_failure_propagates_without_modifying_input():
    inputs = fixture()
    k = inputs["stiffness"].copy()
    def stop():
        raise RuntimeError("owner stopped")
    inputs["check"] = stop
    with pytest.raises(RuntimeError, match="owner stopped"):
        backside.transform(**inputs)
    np.testing.assert_array_equal(inputs["stiffness"], k)


def layer_fixture(monkeypatch, n=12):
    from tools.operators import kinematics

    inputs = fixture(n=n)
    binding = NS(cu_card=object(), material_kernels=object())
    inputs['material_binding'] = binding
    operands = backside.transform(**inputs)
    calls = []

    class Layer:
        native_coupon_constructor_guard = {'constructor_only': True}

        def __init__(self, **kwargs):
            self.arguments = kwargs

    def authenticate(actual):
        assert actual is binding
        calls.append('material')
        return {'synthetic_fixture': True}, NS(attach_corrected_material_binding=attach)

    def attach(layer, actual):
        assert actual is binding
        layer.corrected_material_binding = actual
        calls.append('attach')
        return layer

    monkeypatch.setattr(kinematics, '_material', authenticate)
    return operands, binding, Layer, calls


@pytest.mark.parametrize('n', [12, 185])
def test_construct_native_layer_retains_operands_binding_and_free_coordinates(monkeypatch, n):
    operands, binding, Layer, calls = layer_fixture(monkeypatch, n)
    layer = backside.construct_layer(operands, layer_type=Layer)
    args = layer.arguments
    assert calls == ['material', 'attach']
    assert args['macro_size'] == 6 and args['bubble_size'] == n-9
    assert args['strain_maps'] is operands.strain_maps
    assert args['weights_mm3'] is operands.weights_mm3
    assert args['cu_card'] is binding.cu_card
    assert args['material_kernels'] is binding.material_kernels
    assert layer.corrected_material_binding is binding
    assert args['plastic_active'] is True
    for temperature in (25., 100., 350.):
        assert args['reference_stiffness'](temperature) is operands.stiffness
        np.testing.assert_array_equal(args['reference_thermal_load'](temperature),
                                      (temperature-25.)*operands.thermal_per_c)
    scales = 1./np.sqrt(np.diag(operands.stiffness))
    np.testing.assert_array_equal(args['macro_coordinate_scale'], scales[:6])
    np.testing.assert_array_equal(args['bubble_coordinate_scale'], scales[6:])
    for key, value in operands.evidence.items():
        if key != 'material_binding_authenticated_here':
            assert layer.backside_operand_binding[key] == value
    assert not layer.backside_operand_binding['actual_dataset_parity_validated']
    assert not layer.backside_operand_binding['reference_K_or_B_corrected']
    assert not layer.backside_operand_binding['force_or_tangent_methods_changed']


def test_construct_default_resolves_native_type_after_material_validation(monkeypatch):
    operands, binding, Layer, calls = layer_fixture(monkeypatch)
    def resolve(coupon):
        assert calls == ['material']
        assert callable(coupon.unit_invariant_layer_type)
        calls.append('native_type')
        return Layer
    monkeypatch.setattr(backside, '_native_layer_type', resolve)
    layer = backside.construct_layer(operands)
    assert calls == ['material', 'native_type', 'attach']
    assert layer.corrected_material_binding is binding


def test_construct_rejects_binding_before_native_constructor(monkeypatch):
    from tools.operators import kinematics
    operands, _, _, _ = layer_fixture(monkeypatch)
    def reject(binding):
        raise ValueError('unreviewed material binding')
    monkeypatch.setattr(kinematics, '_material', reject)
    def forbidden(**kwargs):
        raise AssertionError('constructor must not run')
    with pytest.raises(ValueError, match='material binding'):
        backside.construct_layer(operands, layer_type=forbidden)


def test_construct_rejects_invalid_temperature_and_mismatched_GP_evidence(monkeypatch):
    operands, _, Layer, _ = layer_fixture(monkeypatch)
    layer = backside.construct_layer(operands, layer_type=Layer)
    for temperature in (24., 351., float('nan'), float('inf')):
        for name in ('reference_stiffness', 'reference_thermal_load'):
            with pytest.raises(ValueError, match='25..350C'):
                layer.arguments[name](temperature)
    bad = replace(operands, evidence=dict(operands.evidence, GP_count=7))
    with pytest.raises(ValueError, match='GP evidence'):
        backside.construct_layer(bad, layer_type=Layer)
