"""Reaction-reference repair retains pointwise force and collective port gates."""

import numpy as np
import pytest

from tools.operator_reference_repair import AREA_MM2, score_response


def metrics(count):
    return [dict(result_set=i + 1, scaled_residual_norm=1e-10,
                 plastic=1e-8, creep=1e-8) for i in range(count)]


def test_force_gate_uses_same_state_reference_and_absolute_floor():
    reference = np.zeros((2, 134))
    prediction = reference.copy()
    reference[:, 0] = np.array([1., 0.]) * AREA_MM2
    prediction[:, 0] = np.array([1.02, .0012]) * AREA_MM2
    result = score_response(prediction, reference, [300., 304.], metrics(2))['Nx']
    assert not result['passed']
    assert result['units'] == 'N/mm'
    assert result['first_failed_result_set'] == 2
    assert result['worst_result_set'] == 2
    assert result['maximum_absolute_error'] == pytest.approx(.02)
    assert result['maximum_normalized_error'] == pytest.approx(1.2)
    assert result['allowed_at_worst'] == pytest.approx(.001)


def test_bending_gate_distinguishes_peak_error_and_worst_normalized_state():
    reference = np.zeros((2, 134))
    reference[:, 3] = np.array([.0001, .02]) * AREA_MM2
    prediction = reference.copy()
    prediction[:, 3] += np.array([.000014, .001]) * AREA_MM2
    result = score_response(prediction, reference, [300., 304.], metrics(2))['Mx']
    assert result['units'] == 'N'
    assert not result['passed']
    assert result['worst_result_set'] == 1
    assert result['maximum_absolute_error'] == pytest.approx(.001)
    assert result['maximum_normalized_error'] == pytest.approx(1.4)
    assert result['allowed_at_worst'] == pytest.approx(1e-5)


def test_port_gate_compares_vector_norm_instead_of_individual_coefficients():
    reference = np.zeros((1, 134))
    prediction = reference.copy()
    prediction[0, 6:8] = 8e-5
    result = score_response(prediction, reference, [300.], metrics(1))
    assert not result['bottom_ports']['passed']
    assert result['top_ports']['passed']
    assert result['bottom_ports']['maximum_normalized_error'] == pytest.approx(.8 * 2**.5)


def test_missing_metric_state_and_negative_dissipation_are_rejected():
    prediction = np.zeros((2, 134))
    rows = metrics(2)
    rows[1]['result_set'] = 3
    with pytest.raises(ValueError, match='coverage'):
        score_response(prediction, prediction, [300., 304.], rows)
    rows = metrics(2)
    rows[1]['plastic'] = -1e-8
    with pytest.raises(ValueError, match='dissipation'):
        score_response(prediction, prediction, [300., 304.], rows)


def test_equilibrium_and_excitation_remain_separate_requirements():
    prediction = np.zeros((1, 134))
    rows = [dict(result_set=1, scaled_residual_norm=1e-9, plastic=0., creep=0.)]
    result = score_response(prediction, prediction, [300.], rows)
    assert result['bubble_equilibrium']['passed']
    assert not result['inelastic_excitation']['passed']
