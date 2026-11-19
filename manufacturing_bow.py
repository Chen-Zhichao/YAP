"""Fabrication-induced natural bow from a calibrated front-side source.

The source is an isotropic in-plane eigenstrain in selected front-side layers.
It is calibrated once, held fixed across geometry sweeps, and represented as an
incoming natural state separate from Cu plasticity/creep during bonding.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

try:
    from YAP.history_bow_uncertainty import GaussianDieBow, _array, _scalar
except ModuleNotFoundError:  # Direct execution from the YAP source directory.
    from history_bow_uncertainty import GaussianDieBow, _array, _scalar


CALIBRATION_SCHEMA = "yap.fabrication_source_calibration.v3"
POPULATION_SCHEMA = "yap.fabrication_birth_population.v3"
BIRTH_BATCH_SCHEMA = "yap.fabrication_birth_batch.v2"
MODEL_STATUS = "LINEAR_ABD_ONLY_NOT_ANSYS_RUNTIME_QUALIFIED"
BOW_CONVENTION = "center_minus_mid_edge"
CALIBRATION_METRIC = "abs(mean_signed_bow)+sigma_level*std_signed_bow"
INJECTION_MODE = "natural_birth_q6"
LITERATURE_OVERALL_WARPAGE_UPPER_UM = 80.0
QUADRATIC_OVERALL_TO_MID_EDGE_FACTOR = 0.5
REFERENCE_MID_EDGE_UPPER_UM = 40.0
LITERATURE_ANCHOR = {
    "citation": (
        "B. C. Rao et al., Dielectric Stack Optimization for Die-level "
        "Warpage Reduction for Chip-to-Wafer Hybrid Bonding, ECTC 2024"
    ),
    "doi": "10.1109/ECTC51529.2024.00019",
    "reported_quantity": "overall pre-bond warpage",
    "reported_upper_um": LITERATURE_OVERALL_WARPAGE_UPPER_UM,
}


def _identity(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _readonly(value):
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ManufacturingLayer:
    """Bottom-to-top isotropic layer; source is per scalar amplitude."""

    name: str
    thickness_um: float
    e_mpa: float
    nu: float
    eigenstrain_per_amplitude: float = 0.0
    is_bulk_silicon: bool = False

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("layer name must be nonempty")
        for key in ("thickness_um", "e_mpa", "nu", "eigenstrain_per_amplitude"):
            object.__setattr__(self, key, _scalar(getattr(self, key), key))
        if not isinstance(self.is_bulk_silicon, (bool, np.bool_)):
            raise ValueError("is_bulk_silicon must be Boolean")
        object.__setattr__(self, "is_bulk_silicon", bool(self.is_bulk_silicon))
        if self.thickness_um <= 0 or self.e_mpa <= 0:
            raise ValueError("thickness and modulus must be positive")
        if not -1 < self.nu < 0.5:
            raise ValueError("isotropic Poisson ratio must lie in (-1, 0.5)")
        silicon_name = self.name.strip().lower() in {"si", "silicon", "bulk_si", "tsv_si"}
        if (self.is_bulk_silicon or silicon_name) and self.eigenstrain_per_amplitude != 0:
            raise ValueError("fabrication eigenstrain must be zero in bulk silicon")


def strain_gradient_to_signed_edge_bow_um(curvature_per_mm, span_mm):
    """Convert strain-gradient curvature to center-minus-mid-edge bow."""
    curvature = _scalar(curvature_per_mm, "curvature_per_mm")
    span = _scalar(span_mm, "span_mm")
    if span <= 0:
        raise ValueError("span_mm must be positive")
    return -1000.0 * curvature * span**2 / 8.0


def signed_edge_bow_to_strain_gradient_per_mm(bow_um, span_mm):
    """Inverse of :func:`strain_gradient_to_signed_edge_bow_um`."""
    bow = _scalar(bow_um, "bow_um")
    span = _scalar(span_mm, "span_mm")
    if span <= 0:
        raise ValueError("span_mm must be positive")
    return -8.0 * bow / (1000.0 * span**2)


def overall_warpage_to_mid_edge_bow_um(overall_warpage_um):
    """Map equal-biaxial quadratic corner warpage to center/mid-edge bow."""
    overall = _scalar(overall_warpage_um, "overall_warpage_um")
    if overall <= 0:
        raise ValueError("overall_warpage_um must be positive")
    return QUADRATIC_OVERALL_TO_MID_EDGE_FACTOR * overall


def _plane_stress_q(layer):
    scale = layer.e_mpa / (1.0 - layer.nu**2)
    return scale * np.array([
        [1.0, layer.nu, 0.0],
        [layer.nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - layer.nu) / 2.0],
    ])


def _bulk_silicon_thickness_um(layers):
    silicon = [layer.thickness_um for layer in layers
               if layer.is_bulk_silicon or layer.name.strip().lower()
               in {"si", "silicon", "bulk_si", "tsv_si"}]
    if len(silicon) != 1:
        raise ValueError("reference stack must identify exactly one bulk-Si layer")
    return silicon[0]


def clpt_source_response(layers):
    """Return the free 6x6 ABD response to unit source amplitude.

    Coordinates use mm and MPa. The generalized state is
    ``q6=[eps_x, eps_y, gamma_xy, kappa_x, kappa_y, kappa_xy]`` with
    ``strain(z)=eps0+z*kappa``.
    """
    layers = tuple(layers)
    if not layers or any(not isinstance(layer, ManufacturingLayer) for layer in layers):
        raise ValueError("supply an ordered, nonempty ManufacturingLayer sequence")
    thickness = np.array([layer.thickness_um * 1e-3 for layer in layers])
    z = np.r_[0.0, np.cumsum(thickness)] - thickness.sum() / 2.0
    a = np.zeros((3, 3))
    b = np.zeros((3, 3))
    d = np.zeros((3, 3))
    n_star = np.zeros(3)
    m_star = np.zeros(3)
    for index, layer in enumerate(layers):
        q = _plane_stress_q(layer)
        z0, z1 = z[index], z[index + 1]
        source = layer.eigenstrain_per_amplitude * np.array([1.0, 1.0, 0.0])
        a += q * (z1 - z0)
        b += 0.5 * q * (z1**2 - z0**2)
        d += (1.0 / 3.0) * q * (z1**3 - z0**3)
        n_star += q @ source * (z1 - z0)
        m_star += 0.5 * q @ source * (z1**2 - z0**2)
    abd = np.block([[a, b], [b, d]])
    resultants = np.r_[n_star, m_star]
    try:
        q6 = np.linalg.solve(abd, resultants)
    except np.linalg.LinAlgError as exc:
        raise ValueError("degenerate laminate ABD matrix") from exc
    if not np.isfinite(abd).all() or not np.isfinite(q6).all():
        raise ValueError("nonfinite CLPT response")

    membrane = np.array([layer.e_mpa / (1.0 - layer.nu) for layer in layers])
    centers = 0.5 * (z[:-1] + z[1:])
    weights = membrane * thickness
    neutral_axis = float(weights @ centers / weights.sum())
    condensed = float(np.sum(membrane * (
        thickness**3 / 12.0 + thickness * (centers - neutral_axis)**2)))

    return {
        "layers": [asdict(layer) for layer in layers],
        "total_thickness_um": float(thickness.sum() * 1000.0),
        "A_matrix_N_per_mm": a.tolist(), "B_matrix_N": b.tolist(),
        "D_matrix_N_mm": d.tolist(), "ABD_matrix": abd.tolist(),
        "N_per_amplitude_N_per_mm": n_star.tolist(),
        "M_per_amplitude_N": m_star.tolist(),
        "natural_birth_q6_per_amplitude": q6.tolist(),
        "kappa_x_per_amplitude_per_mm": float(q6[3]),
        "kappa_y_per_amplitude_per_mm": float(q6[4]),
        "kappa_xy_per_amplitude_per_mm": float(q6[5]),
        "kappa_per_amplitude_per_mm": float(0.5 * (q6[3] + q6[4])),
        "neutral_axis_mm": neutral_axis,
        "condensed_bending_N_mm": condensed,
        "source_confined_to_front_build_up": True,
        "virtual_epsilon0_per_amplitude": float(0.5 * (q6[0] + q6[1])),
        "A_N_per_mm": float(weights.sum()),
        "B_N": float(weights @ centers),
        "D_N_mm": float(np.sum(membrane * (
            thickness**3 / 12.0 + thickness * centers**2))),
    }


def source_bow_moments(source_amplitude, source_cv, response, span_mm,
                       *, source_sign=1.0, quadrature_order=21,
                       forward_model=None):
    """Evaluate signed-bow moments for a Gaussian source distribution."""
    amplitude = _scalar(source_amplitude, "source_amplitude")
    cv = _scalar(source_cv, "source_cv")
    sign = _scalar(source_sign, "source_sign")
    span = _scalar(span_mm, "span_mm")
    if amplitude <= 0 or cv < 0 or span <= 0 or sign not in (-1.0, 1.0):
        raise ValueError("source amplitude/span must be positive, CV nonnegative, sign +/-1")
    if isinstance(quadrature_order, bool) or int(quadrature_order) != quadrature_order or quadrature_order < 3:
        raise ValueError("quadrature_order must be an integer >= 3")
    if forward_model is None:
        def forward_model(sample, local_response, local_span):
            curvature = sample * local_response["kappa_per_amplitude_per_mm"]
            return strain_gradient_to_signed_edge_bow_um(curvature, local_span)
    nodes, weights = np.polynomial.hermite.hermgauss(int(quadrature_order))
    samples = sign * amplitude * (1.0 + np.sqrt(2.0) * cv * nodes)
    bows = np.array([_scalar(forward_model(value, response, span), "bow_um") for value in samples])
    probability = weights / np.sqrt(np.pi)
    mean = float(probability @ bows)
    variance = float(probability @ (bows - mean)**2)
    return {
        "mean_signed_bow_um": mean,
        "std_signed_bow_um": float(np.sqrt(max(variance, 0.0))),
        "mean_curvature_per_mm": signed_edge_bow_to_strain_gradient_per_mm(mean, span),
        "quadrature_order": int(quadrature_order),
    }


@dataclass(frozen=True)
class FabricationSourceCalibration:
    nominal_amplitude: float
    source_cv: float
    source_sign: float
    sigma_level: float
    target_upper_magnitude_um: float
    achieved_upper_magnitude_um: float
    reference_span_mm: float
    reference_si_thickness_um: float
    reference_response: dict
    reference_moments: dict
    iterations: int
    model_status: str = MODEL_STATUS
    schema: str = CALIBRATION_SCHEMA

    def __post_init__(self):
        for key in ("nominal_amplitude", "source_cv", "source_sign", "sigma_level",
                    "target_upper_magnitude_um", "achieved_upper_magnitude_um",
                    "reference_span_mm", "reference_si_thickness_um"):
            object.__setattr__(self, key, _scalar(getattr(self, key), key))
        if self.nominal_amplitude <= 0 or self.source_cv < 0:
            raise ValueError("invalid calibrated source distribution")

    def record(self):
        result = asdict(self)
        result.update({"calibration_metric": CALIBRATION_METRIC,
                       "signed_nominal_amplitude": (
                           self.source_sign * self.nominal_amplitude),
                       "bow_convention": BOW_CONVENTION,
                       "magnitude_conversion": CALIBRATION_METRIC,
                       "recalibrate_per_geometry": False,
                       "fabrication_source_reapplied_during_assembly": False,
                       "incoming_cu_state": "virgin"})
        return result

    @property
    def signature(self):
        return _identity(self.record())


def calibrate_fabrication_source(reference_layers, *, reference_span_mm=10.0,
                                 reference_si_thickness_um=50.0,
                                 target_upper_magnitude_um=REFERENCE_MID_EDGE_UPPER_UM,
                                 source_cv=0.10, sigma_level=3.0,
                                 source_sign=1.0, forward_model=None,
                                 quadrature_order=21, relative_tolerance=1e-10,
                                 maximum_iterations=100):
    """Inverse-calibrate one nominal local source on the reference die."""
    reference_layers = tuple(reference_layers)
    actual_si_thickness = _bulk_silicon_thickness_um(reference_layers)
    declared_si_thickness = _scalar(reference_si_thickness_um,
                                    "reference_si_thickness_um")
    if not np.isclose(actual_si_thickness, declared_si_thickness,
                      rtol=0.0, atol=1e-12):
        raise ValueError(
            "reference_si_thickness_um does not match the bulk-Si layer ledger")
    response = clpt_source_response(reference_layers)
    if response["kappa_per_amplitude_per_mm"] == 0:
        raise ValueError("reference front-side source has zero bow response")
    target = _scalar(target_upper_magnitude_um, "target_upper_magnitude_um")
    sigma = _scalar(sigma_level, "sigma_level")
    tolerance = _scalar(relative_tolerance, "relative_tolerance")
    if target <= 0 or sigma < 0 or tolerance <= 0:
        raise ValueError("target/tolerance must be positive and sigma_level nonnegative")

    def evaluate(amplitude):
        moments = source_bow_moments(amplitude, source_cv, response, reference_span_mm,
                                     source_sign=source_sign,
                                     quadrature_order=quadrature_order,
                                     forward_model=forward_model)
        metric = abs(moments["mean_signed_bow_um"]) + sigma * moments["std_signed_bow_um"]
        return metric, moments

    low, high = 0.0, 1e-12
    expansion_count = 0
    value, _ = evaluate(high)
    while value < target:
        high *= 10.0
        expansion_count += 1
        if expansion_count > 30 or not np.isfinite(high):
            raise ValueError("could not bracket the fabrication-source calibration")
        value, _ = evaluate(high)
    moments = None
    for iteration in range(1, int(maximum_iterations) + 1):
        trial = 0.5 * (low + high)
        value, moments = evaluate(trial)
        if abs(value - target) <= tolerance * target:
            break
        if value < target:
            low = trial
        else:
            high = trial
    else:
        raise RuntimeError("fabrication-source calibration did not converge")
    return FabricationSourceCalibration(
        nominal_amplitude=trial, source_cv=source_cv, source_sign=source_sign,
        sigma_level=sigma, target_upper_magnitude_um=target,
        achieved_upper_magnitude_um=value, reference_span_mm=reference_span_mm,
        reference_si_thickness_um=declared_si_thickness,
        reference_response=response, reference_moments=moments,
        iterations=iteration + expansion_count)


def source_scaled_population(layer_stacks, fullspan_mm, *, reference_layers,
                             reference_upper_overall_warpage_um=(
                                 LITERATURE_OVERALL_WARPAGE_UPPER_UM),
                             reference_span_mm=10.0, source_cv=0.10,
                             source_sign=-1.0, calibration=None):
    """Build birth bows from the overall-warpage reference bound."""
    overall = _scalar(reference_upper_overall_warpage_um,
                      "reference_upper_overall_warpage_um")
    target = overall_warpage_to_mid_edge_bow_um(overall)
    if calibration is None:
        reference_layers = tuple(reference_layers)
        calibration = calibrate_fabrication_source(
            reference_layers, reference_span_mm=reference_span_mm,
            reference_si_thickness_um=_bulk_silicon_thickness_um(reference_layers),
            target_upper_magnitude_um=target,
            source_cv=source_cv, source_sign=source_sign)
    if not isinstance(calibration, FabricationSourceCalibration):
        raise ValueError("calibration must be FabricationSourceCalibration")
    if not np.isclose(calibration.target_upper_magnitude_um, target, rtol=1e-9):
        raise ValueError(
            "calibration target must match converted overall-warpage bound")
    responses = [clpt_source_response(layers) for layers in layer_stacks]
    spans = _array(fullspan_mm, "fullspan_mm")
    if not responses or spans.shape != (len(responses), 2) or np.any(spans <= 0):
        raise ValueError("supply positive full x/y spans for every die")
    mean = np.zeros_like(spans)
    covariance = np.zeros((2 * len(responses), 2 * len(responses)))
    signed_per_amplitude = np.zeros_like(spans)
    for index, response in enumerate(responses):
        for axis in range(2):
            curvature = response["natural_birth_q6_per_amplitude"][3 + axis]
            signed_per_amplitude[index, axis] = strain_gradient_to_signed_edge_bow_um(
                curvature, spans[index, axis])
        source_mean = calibration.source_sign * calibration.nominal_amplitude
        source_std = abs(calibration.nominal_amplitude * calibration.source_cv)
        mean[index] = signed_per_amplitude[index] * source_mean
        vector = signed_per_amplitude[index] * source_std
        covariance[2*index:2*index+2, 2*index:2*index+2] = np.outer(vector, vector)
    population = GaussianDieBow(mean, covariance)
    record = {"schema": POPULATION_SCHEMA, "model_status": MODEL_STATUS,
              "injection_mode": INJECTION_MODE, "calibration": calibration.record(),
              "calibration_sha256": calibration.signature,
              "die_responses": responses, "fullspan_mm": spans.tolist(),
              "signed_edge_bow_per_source_amplitude_um": signed_per_amplitude.tolist(),
              "population": population.record(), "population_sha256": population.signature,
              "source_amplitude_mean": calibration.source_sign * calibration.nominal_amplitude,
              "source_amplitude_sigma": calibration.nominal_amplitude * calibration.source_cv,
              "manufacturing_stress_transferred_to_thermal_solver": False,
              "fabrication_source_reapplied_during_assembly": False,
              "incoming_cu_state": "virgin",
              "calibration_schema_current": CALIBRATION_SCHEMA,
              "literature_anchor": LITERATURE_ANCHOR,
              "reference_upper_overall_warpage_um": overall,
              "quadratic_overall_to_mid_edge_factor":
                  QUADRATIC_OVERALL_TO_MID_EDGE_FACTOR,
              "target_upper_mid_edge_bow_um": target,
              "statistical_interpretation":
                  "abs(mean_mid_edge)+3*std_mid_edge; YAP modeling assumption",
              "signed_edge_sigma_per_source_draw_um": (
                  signed_per_amplitude * calibration.source_sign *
                  calibration.nominal_amplitude * calibration.source_cv).tolist()}
    return population, record


@dataclass(frozen=True)
class FabricationBirthBatch:
    """Immutable one-draw-per-body birth state for D2W dies or W2W wafers."""

    body_scope: str
    source_amplitude: np.ndarray
    natural_birth_q6: np.ndarray
    calibration_sha256: str
    shared_across_sites: bool
    seed: int
    schema: str = BIRTH_BATCH_SCHEMA
    injection_mode: str = INJECTION_MODE
    incoming_cu_state: str = "virgin"
    redraw_per_cycle: bool = False
    fabrication_resultants_reapplied: bool = False

    def __post_init__(self):
        if self.body_scope not in {"d2w_die", "w2w_wafer"}:
            raise ValueError("body_scope must be d2w_die or w2w_wafer")
        amplitude = _array(self.source_amplitude, "source_amplitude")
        q6 = _array(self.natural_birth_q6, "natural_birth_q6")
        if amplitude.ndim != 1 or q6.shape != (len(amplitude), 6):
            raise ValueError("birth state must contain one q6 vector per body")
        object.__setattr__(self, "source_amplitude", _readonly(amplitude))
        object.__setattr__(self, "natural_birth_q6", _readonly(q6))
        if self.shared_across_sites != (self.body_scope == "w2w_wafer"):
            raise ValueError("only W2W wafer birth states are shared across die sites")

    def record(self):
        return {"schema": self.schema, "body_scope": self.body_scope,
                "source_amplitude": self.source_amplitude.tolist(),
                "natural_birth_q6": self.natural_birth_q6.tolist(),
                "calibration_sha256": self.calibration_sha256,
                "shared_across_sites": self.shared_across_sites, "seed": int(self.seed),
                "injection_mode": self.injection_mode,
                "incoming_cu_state": self.incoming_cu_state,
                "redraw_per_cycle": self.redraw_per_cycle,
                "fabrication_resultants_reapplied": self.fabrication_resultants_reapplied}


def sample_fabrication_birth_batch(body_count, layers, calibration, *, body_scope, seed):
    """Draw the fabrication source once per incoming die/wafer body."""
    if isinstance(body_count, bool) or int(body_count) != body_count or body_count < 1:
        raise ValueError("body_count must be a positive integer")
    if not isinstance(calibration, FabricationSourceCalibration):
        raise ValueError("calibration must be FabricationSourceCalibration")
    response = clpt_source_response(layers)
    rng = np.random.default_rng(seed)
    amplitude = calibration.source_sign * calibration.nominal_amplitude * (
        1.0 + calibration.source_cv * rng.standard_normal(int(body_count)))
    q6 = amplitude[:, None] * np.asarray(response["natural_birth_q6_per_amplitude"])[None, :]
    return FabricationBirthBatch(
        body_scope=body_scope, source_amplitude=amplitude, natural_birth_q6=q6,
        calibration_sha256=calibration.signature,
        shared_across_sites=(body_scope == "w2w_wafer"), seed=int(seed))


def exp12_initial_q6_payload(batch):
    """Return the strict full-natural-state payload accepted by Exp12.

    Unlike the legacy bow-only input, this preserves the fabrication-induced
    membrane coordinates as well as curvature and twist.  Calibration and RNG
    provenance remain in :meth:`FabricationBirthBatch.record`; the solver also
    binds the exact payload bytes in its prediction freeze.
    """
    if not isinstance(batch, FabricationBirthBatch):
        raise TypeError("batch must be a FabricationBirthBatch")
    return {
        "schema": "exp12.natural_initial_q6.v2",
        "frame": "common_global_xyz_z_thickness",
        "bow_convention": BOW_CONVENTION,
        "die_natural_q6": batch.natural_birth_q6.tolist(),
    }


def geometry_sensitivity_rows(calibration, layer_factory, *, die_sizes_mm=(5.0, 10.0, 15.0),
                              si_thicknesses_um=(20.0, 50.0, 80.0),
                              source_cvs=(0.05, 0.10, 0.20), quadrature_order=21):
    """Propagate one calibration through rebuilt geometry-dependent ABD models."""
    if not isinstance(calibration, FabricationSourceCalibration):
        raise ValueError("calibration must be FabricationSourceCalibration")
    rows = []
    for cv in source_cvs:
        for size in die_sizes_mm:
            for thickness in si_thicknesses_um:
                response = clpt_source_response(layer_factory(float(thickness)))
                moments = source_bow_moments(
                    calibration.nominal_amplitude, cv, response, float(size),
                    source_sign=calibration.source_sign,
                    quadrature_order=quadrature_order)
                mean, std = moments["mean_signed_bow_um"], moments["std_signed_bow_um"]
                rows.append({"die_size_mm": float(size),
                             "si_thickness_um": float(thickness),
                             "source_cv": float(cv),
                             "mean_signed_bow_um": mean,
                             "std_signed_bow_um": std,
                             "mean_minus_3sigma_um": mean - 3.0 * std,
                             "mean_plus_3sigma_um": mean + 3.0 * std,
                             "mean_curvature_per_mm": moments["mean_curvature_per_mm"],
                             "nominal_source_amplitude": calibration.nominal_amplitude,
                             "signed_nominal_source_amplitude": (
                                 calibration.source_sign * calibration.nominal_amplitude),
                             "calibration_sha256": calibration.signature,
                             "model_status": calibration.model_status})
    return rows
