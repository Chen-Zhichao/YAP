"""Opt-in uncertainty propagation for complete, history-dependent bow solves.

All responses are signed paper center-minus-edge/corner bows in um. One
joint initial-bow realization belongs to an entire process, not one cycle.
The affine check below is empirical and local, not a proof of Gaussian
physical outputs, thermal accuracy, or manufacturing yield accuracy.
"""

from dataclasses import dataclass, field
import hashlib
import json
from numbers import Integral
from typing import Mapping

import numpy as np
from scipy.stats import chi2


def _array(value, name):
    if not isinstance(value, np.ndarray):
        objects = np.asarray(value, dtype=object)
        if any(isinstance(v, (bool, np.bool_, str)) for v in objects.flat):
            raise ValueError(f"{name} must not contain strings or Booleans")
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers, not strings or Booleans")
    result = np.array(raw, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _scalar(value, name):
    result = _array(value, name)
    if result.ndim:
        raise ValueError(f"{name} must be a scalar")
    return float(result)


def _count(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _frozen(value):
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _identity(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class GaussianDieBow:
    """Full covariance in order d1/x,d1/y,d2/x,d2/y,..., in um squared.

    Singular covariance is supported without adding artificial random noise.
    Eigenvalues at correlation-matrix roundoff are removed and that numerical
    correction is reported. Scaling to correlations preserves small marginal
    variances next to large ones; no absolute variance floor is imposed.
    """

    mean_um: np.ndarray
    covariance_um2: np.ndarray
    factor_um: np.ndarray = field(init=False, repr=False)
    correlation_reconstruction_error: float = field(init=False)

    def __post_init__(self):
        mean = _array(self.mean_um, "mean_um")
        if mean.ndim != 2 or mean.shape[1] != 2 or mean.shape[0] < 1:
            raise ValueError("mean_um must have shape (die,2)")
        cov = _array(self.covariance_um2, "covariance_um2")
        if cov.shape != (mean.size, mean.size):
            raise ValueError("covariance must follow the complete die/x/y order")
        variance = np.diag(cov)
        if np.any(variance < 0):
            raise ValueError("negative bow variance")
        active = variance > 0
        if np.any(cov[~active, :] != 0) or np.any(cov[:, ~active] != 0):
            raise ValueError("zero-variance coordinates cannot have covariance")
        factor = np.empty((mean.size, 0))
        defect = 0.
        if active.any():
            std = np.sqrt(variance[active])
            corr = cov[np.ix_(active, active)] / std[:, None] / std[None, :]
            if not np.isfinite(corr).all():
                raise ValueError("covariance cannot be normalized")
            tol = 64 * np.finfo(float).eps * len(std)
            if np.max(np.abs(corr-corr.T)) > tol:
                raise ValueError("covariance must be symmetric")
            corr = (corr+corr.T)/2
            eigenvalues, vectors = np.linalg.eigh(corr)
            if eigenvalues.min() < -tol:
                raise ValueError("covariance must be positive semidefinite")
            keep = eigenvalues > tol
            reduced = vectors[:, keep] * np.sqrt(eigenvalues[keep])
            defect = float(np.max(np.abs(reduced @ reduced.T-corr)))
            if defect > 4*tol:
                raise ValueError("PSD factorization lost covariance")
            factor = np.zeros((mean.size, int(keep.sum())))
            factor[active] = std[:, None]*reduced
        object.__setattr__(self, "mean_um", _frozen(mean))
        object.__setattr__(self, "covariance_um2", _frozen(cov))
        object.__setattr__(self, "factor_um", _frozen(factor))
        object.__setattr__(self, "correlation_reconstruction_error", defect)

    @property
    def rank(self):
        return self.factor_um.shape[1]

    @property
    def die_count(self):
        return self.mean_um.shape[0]

    def record(self):
        return dict(schema="yap.joint_initial_bow_gaussian.v1", unit="um",
                    frame="common_global_xyz_z_thickness",
                    bow_convention="center_minus_mid_edge", order="die_major_xy",
                    mean_um=self.mean_um.tolist(), covariance_um2=self.covariance_um2.tolist(),
                    factor_um=self.factor_um.tolist(), numerical_rank=self.rank,
                    correlation_reconstruction_error=self.correlation_reconstruction_error,
                    truncated=False, redraw_per_cycle=False)

    @property
    def signature(self):
        return _identity(self.record())

    def latent_draws(self, count, *, seed):
        return np.random.default_rng(seed).standard_normal((_count(count, "count", 1), self.rank))

    def realizations(self, latent):
        z = _array(latent, "latent")
        if z.ndim != 2 or z.shape[1] != self.rank:
            raise ValueError("latent must have shape (realization,numerical_rank)")
        with np.errstate(over="raise", invalid="raise"):
            try:
                result = self.mean_um.ravel() + z @ self.factor_um.T
            except FloatingPointError as exc:
                raise ValueError("initial-bow realization overflow") from exc
        return result.reshape(len(z), self.die_count, 2)


def reference_die_population(substrate_thickness_um, fullspan_mm, *, rho_xy,
                             reference_substrate_um=38.5, thickness_exponent=None):
    """User anchor: mean=0, 3 sigma edge=50 um at TOTAL 50 um, 10x10 mm.

    Thickness scaling uses Si, NOT die total. The default reference Si is
    50 - 10 BEOL - 1.5 BI = 38.5 um for the single-front-BI reference die.
    Supply each die's actual Si thickness, including the effect of a second
    BI on its thickness ledger. Independent dies; rho_xy=1 is one
    spherical-curvature draw per die.
    Thickness scaling is an explicit statistical hypothesis, not inferred
    from the natural-shape input or a universal multilayer constitutive law.
    Without an exponent only reference-thickness geometries are permitted.
    """
    thickness = _array(substrate_thickness_um, "substrate_thickness_um")
    reference = _scalar(reference_substrate_um, "reference_substrate_um")
    spans = _array(fullspan_mm, "fullspan_mm")
    if thickness.ndim != 1 or not len(thickness) or np.any(thickness <= 0):
        raise ValueError("supply one positive Si substrate thickness per die")
    if reference <= 0:
        raise ValueError("reference_substrate_um must be positive")
    if spans.shape != (len(thickness), 2) or np.any(spans <= 0):
        raise ValueError("supply positive full x/y spans for every die")
    rho = _scalar(rho_xy, "rho_xy")
    if not -1 <= rho <= 1:
        raise ValueError("rho_xy must be between -1 and 1")
    if thickness_exponent is None:
        if np.any(thickness != reference):
            raise ValueError("non-reference thickness requires an explicit thickness_exponent")
        power = 0.
    else:
        power = _scalar(thickness_exponent, "thickness_exponent")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            std = (50./3) * (spans/10.)**2 * (reference/thickness[:, None])**power
            covariance = np.zeros((thickness.size*2, thickness.size*2))
            for i, (sx, sy) in enumerate(std):
                covariance[2*i:2*i+2, 2*i:2*i+2] = [
                    [sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]]
        except FloatingPointError as exc:
            raise ValueError("geometry scaling exceeds finite range") from exc
    if np.any(std <= 0) or np.any(np.diag(covariance) == 0):
        raise ValueError("nonzero reference spread underflowed")
    return GaussianDieBow(np.zeros_like(spans), covariance)


def _response(value, labels=None):
    if not isinstance(value, Mapping) or not value:
        raise ValueError("evaluator must return named whole-history bow responses")
    if any(not isinstance(k, str) or not k.endswith("_um") for k in value):
        raise ValueError("response keys must explicitly end in _um")
    keys = tuple(sorted(value))
    if labels is not None and keys != labels:
        raise ValueError("process event/response schema changed between realizations")
    return keys, np.array([_scalar(value[k], k) for k in keys])


@dataclass(frozen=True)
class AffineBowStudy:
    population: GaussianDieBow
    context_id: str
    labels: tuple[str, ...]
    latent_points: np.ndarray
    purposes: tuple[str, ...]
    exact_um: np.ndarray
    baseline_um: np.ndarray
    sensitivity_um: np.ndarray
    check_radius: float
    fit_count: int
    absolute_tolerance_um: float
    relative_tolerance: float

    def __post_init__(self):
        for name in ("latent_points", "exact_um", "baseline_um", "sensitivity_um"):
            object.__setattr__(self, name, _frozen(_array(getattr(self, name), name)))
        rank, outputs, count = self.population.rank, len(self.labels), len(self.purposes)
        if self.latent_points.shape != (count, rank) or self.exact_um.shape != (count, outputs) \
                or self.baseline_um.shape != (outputs,) or self.sensitivity_um.shape != (outputs, rank):
            raise ValueError("inconsistent whole-history study shapes")
        if self.fit_count != 1+2*rank or (rank and count < self.fit_count+4*rank+2):
            raise ValueError("independent half/outer/mixed checks are required")
        if not np.array_equal(self.baseline_um, self.exact_um[0]) \
                or np.any(self.latent_points[0] != 0):
            raise ValueError("baseline must be the mean-input evaluation")

    @property
    def check_errors_um(self):
        predicted = self.baseline_um + self.latent_points[self.fit_count:] @ self.sensitivity_um.T
        return np.abs(predicted-self.exact_um[self.fit_count:])

    @property
    def check_passed(self):
        reference = self.exact_um[self.fit_count:]
        limits = np.maximum(self.absolute_tolerance_um, self.relative_tolerance*np.abs(reference))
        return bool(np.all(self.check_errors_um <= limits))

    def _check_context(self, context_id):
        if context_id != self.context_id:
            raise ValueError("model/material/process/geometry context does not match")

    def report(self):
        errors = self.check_errors_um
        ref = self.exact_um[self.fit_count:]
        limits = np.maximum(self.absolute_tolerance_um, self.relative_tolerance*np.abs(ref))
        return dict(schema="yap.affine_bow_check.v1", context_id=self.context_id,
                    population_sha256=self.population.signature,
                    fitting_evaluations=self.fit_count,
                    checking_evaluations=len(self.purposes)-self.fit_count,
                    status="PASS_FINITE_CHECKS_ONLY" if self.check_passed else "FAIL_USE_FULL_HISTORY",
                    labels=list(self.labels), check_radius_standard_normal=self.check_radius,
                    gaussian_probability_outside_radius=(float(chi2.sf(self.check_radius**2, self.population.rank))
                                                       if self.population.rank else 0.),
                    absolute_tolerance_um=self.absolute_tolerance_um,
                    relative_tolerance=self.relative_tolerance,
                    maximum_check_absolute_error_um=float(np.max(errors, initial=0.)),
                    maximum_check_scaled_error=float(np.max(errors/limits, initial=0.)),
                    maximum_absolute_error_by_output_um=np.max(errors, axis=0, initial=0.).tolist(),
                    physical_gaussianity_proven=False, mapdl_validation=False, yield_validation=False)

    def affine_moments_preview(self):
        """Moments of the affine approximation, not of the nonlinear solver."""
        if not self.check_passed:
            raise ValueError("failed finite checks; do not publish affine moments")
        return self.baseline_um.copy(), self.sensitivity_um @ self.sensitivity_um.T

    def propagate(self, latent, *, context_id, exact_evaluator=None, max_exact_evaluations=0):
        """Retain EVERY draw. Failed gates and out-of-domain draws need replay.

        The caller's exact evaluator must run a fresh complete history. No
        MAPDL truth is used here. If replay cannot finish, raise rather than
        drop samples and renormalize a biased distribution/yield estimate.
        """
        self._check_context(context_id)
        z = _array(latent, "latent")
        bows = self.population.realizations(z)
        if not len(z):
            raise ValueError("propagation needs at least one draw")
        budget = _count(max_exact_evaluations, "max_exact_evaluations")
        outside = np.linalg.norm(z, axis=1) > self.check_radius
        replay = outside | (not self.check_passed)
        required = int(np.count_nonzero(replay))
        if required and (exact_evaluator is None or required > budget):
            raise ValueError(f"need {required} full-history replays; no samples were dropped")
        result = np.empty((len(z), len(self.labels)))
        result[~replay] = self.baseline_um + z[~replay] @ self.sensitivity_um.T
        for index in np.flatnonzero(replay):
            try:
                _, result[index] = _response(exact_evaluator(bows[index].copy()), self.labels)
            except Exception as exc:
                raise RuntimeError(f"full-history replay failed at draw {index}; propagation incomplete") from exc
        if not np.isfinite(result).all():
            raise ValueError("response propagation overflow")
        return dict(values_um=result, labels=self.labels, initial_bows_um=bows,
                    exact_replay_mask=replay, outside_check_domain_mask=outside,
                    exact_evaluations=required, sample_count=len(z), samples_dropped=0)


def affine_probe_design(population, *, step_sigma=1., coverage_probability=.9973,
                        mixed_checks=8, seed=731):
    """Plan probes without evaluating a history or fitting any responses.

    Return (latent_points, purposes, fit_count, radius). Points are read-only;
    purposes is a tuple, with fitting probes first and held-out checks after.
    Freeze the population factor and realized bows alongside this design for
    parallel workers; a seed alone does not bind the covariance eigenbasis.
    These are finite design checks, not IID Gaussian validation samples.
    """
    if not isinstance(population, GaussianDieBow):
        raise TypeError("population must be GaussianDieBow")
    step = _scalar(step_sigma, "step_sigma")
    coverage = _scalar(coverage_probability, "coverage_probability")
    mixed = _count(mixed_checks, "mixed_checks", 2)
    if step <= 0 or not .5 < coverage < 1:
        raise ValueError("invalid finite-check domain or tolerances")
    rank = population.rank
    radius = float(np.sqrt(chi2.ppf(coverage, rank))) if rank else 0.
    if rank and radius <= step:
        raise ValueError("outer check radius must exceed fitting step")
    points, purposes = [np.zeros(rank)], ["fit_mean"]
    axes = np.eye(rank)
    for i, axis in enumerate(axes):
        for sign in (1, -1):
            points.append(sign*step*axis)
            purposes.append(f"fit_axis{i}_{sign:+d}")
    fit_count = len(points)
    for i, axis in enumerate(axes):
        for label, scale in (("half", step/2), ("outer", radius)):
            for sign in (1, -1):
                points.append(sign*scale*axis)
                purposes.append(f"check_{label}_axis{i}_{sign:+d}")
    rng = np.random.default_rng(seed)
    if rank:
        for i in range(mixed):
            direction = rng.standard_normal(rank)
            direction /= np.linalg.norm(direction)
            points.append(direction * radius * (1. if i % 2 == 0 else .5))
            purposes.append(f"check_mixed_{i}")
    return _frozen(np.stack(points)), tuple(purposes), fit_count, radius


def fit_checked_affine_response(population, evaluator, *, context_id, step_sigma=1.,
                                coverage_probability=.9973, mixed_checks=8, seed=731,
                                absolute_tolerance_um=.02, relative_tolerance=.05):
    """Central probes + held-out half/outer axes and mixed directions.

    Each evaluator(bows_um) response must come from a fresh COMPLETE
    deterministic process; verified cached probe responses are also allowed.
    Output keys identify fixed physical stages and bodies, not list indices.
    The mean and +/- probes alone fit the model. Checks never alter that fit.
    Caller must freeze geometry/material/solver identity and tolerances before
    execution and keep reference MAPDL data outside the callback.
    """
    if not isinstance(population, GaussianDieBow):
        raise TypeError("population must be GaussianDieBow")
    if not isinstance(context_id, str) or not context_id.strip():
        raise ValueError("provide an explicit mechanical context_id")
    step = _scalar(step_sigma, "step_sigma")
    atol = _scalar(absolute_tolerance_um, "absolute_tolerance_um")
    rtol = _scalar(relative_tolerance, "relative_tolerance")
    if atol <= 0 or rtol < 0:
        raise ValueError("invalid finite-check domain or tolerances")
    points, purposes, fit_count, radius = affine_probe_design(
        population, step_sigma=step, coverage_probability=coverage_probability,
        mixed_checks=mixed_checks, seed=seed)
    rank = population.rank
    labels, outputs = None, []
    for bows in population.realizations(points):
        labels, values = _response(evaluator(bows.copy()), labels)
        outputs.append(values)
    exact = np.stack(outputs)
    sensitivity = np.empty((len(labels), rank))
    for i in range(rank):
        sensitivity[:, i] = (exact[1+2*i]-exact[2+2*i])/(2*step)
    return AffineBowStudy(population, context_id, labels, _frozen(points), tuple(purposes),
                          _frozen(exact), _frozen(exact[0]), _frozen(sensitivity),
                          radius, fit_count, atol, rtol)
