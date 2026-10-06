"""Unqualified, solver-independent pre-attachment overlay arithmetic.

Callers supply both released bodies at every attachment of every JOINT draw.
No history, placement draws, pad layout, threshold, or extra noise is inferred.
This computes bow-induced flattening overlay, not total accumulated face strain
or physical manufacturing yield. Event labels are caller assertions, not an
independent audit of mechanical provenance. No files or solvers are accessed.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import re

import numpy as np

try:
    from YAP.history_bow_yield import (
        PrebondBody, affine_overlay_um, compound_site_yield, paired_magnification,
        worst_critical_pad_survival,
    )
except ModuleNotFoundError:  # Direct execution from the YAP source directory.
    from history_bow_yield import (
        PrebondBody, affine_overlay_um, compound_site_yield, paired_magnification,
        worst_critical_pad_survival,
    )


def _numeric(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return np.array(raw, dtype=float, copy=True)


def _mask(value, size, name):
    mask = np.asarray(value)
    if mask.dtype != np.bool_ or mask.shape != (size,) or not mask.any():
        raise ValueError(f"{name} must be an explicit nonempty Boolean mask")
    return mask


@dataclass(frozen=True)
class Q6LinearFlattening:
    """Supplied OLD-minus-INCOMING (x,y,engineering shear) increment in ppm.

    ``semantics`` must be ``linear_flatten_increment_only``: neither absolute
    face strain nor an already metric-corrected quantity is accepted. Shear
    must be exactly zero because Eq.7's directional affine map has no shear
    channel. The caller, not this bridge, establishes the linear-only origin.
    """

    old_minus_incoming_ppm: tuple[float, float, float]
    semantics: str

    def __post_init__(self):
        if self.semantics != "linear_flatten_increment_only":
            raise ValueError("q6 requires linear_flatten_increment_only semantics")
        values = _numeric(self.old_minus_incoming_ppm, "old_minus_incoming_ppm")
        if values.shape != (3,) or values[2] != 0:
            raise ValueError("q6 requires explicit (x,y,shear) ppm with zero shear")
        object.__setattr__(self, "old_minus_incoming_ppm", tuple(float(v) for v in values))


def audited_q6_linear_flattening(values_ppm, *, halfspan_um, max_omitted_overlay_um):
    """Drop a shear channel only after bounding its absolute overlay effect.

    The conservative bound uses the full engineering shear times the corner
    radius (twice the symmetric-shear displacement). Callers must supply the
    budget in advance; the raw channel and bound remain in the returned audit.
    This does not alter the strict direct Q6LinearFlattening constructor.
    """
    values = _numeric(values_ppm, "values_ppm")
    spans = _numeric(halfspan_um, "halfspan_um")
    budget = _numeric(max_omitted_overlay_um, "max_omitted_overlay_um")
    if values.shape != (3,) or spans.shape != (2,) or np.any(spans <= 0):
        raise ValueError("require three q6 increments and two positive halfspans")
    if budget.ndim or float(budget) < 0:
        raise ValueError("require an explicit nonnegative absolute overlay budget")
    bound = float(abs(values[2]) * 1e-6 * np.linalg.norm(spans))
    if not np.isfinite(bound) or bound > float(budget):
        raise ValueError("omitted shear exceeds the absolute overlay budget")
    audit = dict(raw_old_minus_incoming_ppm=values.tolist(),
                 shear_overlay_bound_um=bound, budget_um=float(budget),
                 halfspan_um=spans.tolist(), shear_dropped=bool(values[2] != 0))
    return Q6LinearFlattening((values[0], values[1], 0.),
                             "linear_flatten_increment_only"), audit


@dataclass(frozen=True)
class PrebondEvent:
    """One pair immediately BEFORE A1, A2, etc., including the first bond.

    Require ``phase='pre_attachment'`` and canonical body state IDs
    ``A#/incoming_free`` and ``A#/old_free``. Released assembled ``A#/released``
    states cannot substitute for the free pair. Both A1 bodies are supplied;
    neither is presumed flat. Upper/lower refer to the aligned input frame.
    """

    attachment_event: str
    phase: str
    incoming_upper: PrebondBody
    old_lower: PrebondBody
    q6_linear: Q6LinearFlattening | None = None

    def __post_init__(self):
        if not isinstance(self.attachment_event, str) or not re.fullmatch(
                r"A[1-9][0-9]*", self.attachment_event):
            raise ValueError("attachment_event must identify A1, A2, etc.")
        if self.phase != "pre_attachment":
            raise ValueError("only pre_attachment pairs are accepted")
        for body, role in ((self.incoming_upper, "incoming_free"),
                           (self.old_lower, "old_free")):
            if not isinstance(body, PrebondBody):
                raise TypeError("supply incoming_upper and old_lower PrebondBody records")
            if body.state_id != f"{self.attachment_event}/{role}":
                raise ValueError(f"require the pre-attachment state {self.attachment_event}/{role}")
        if self.incoming_upper.body_id == self.old_lower.body_id:
            raise ValueError("incoming and old must identify distinct bodies")
        if self.q6_linear is not None and not isinstance(self.q6_linear, Q6LinearFlattening):
            raise TypeError("q6_linear must explicitly declare Q6LinearFlattening semantics")


def event_magnification(event):
    """Dimensionless upper-minus-lower bow-induced strain, including metric once."""
    if not isinstance(event, PrebondEvent):
        raise TypeError("require a PrebondEvent")
    if event.q6_linear is None:
        return paired_magnification(event.incoming_upper, event.old_lower)
    # The supplied q6 increment REPLACES the linear lever term, not all of Eq.7.
    metric = paired_magnification(replace(event.incoming_upper, face_lever_um=0.),
                                  replace(event.old_lower, face_lever_um=0.))
    linear = -1e-6 * np.asarray(event.q6_linear.old_minus_incoming_ppm[:2])
    with np.errstate(over="raise", invalid="raise"):
        try:
            return _numeric(linear + metric, "magnification")
        except FloatingPointError as exc:
            raise ValueError("q6 plus metric exceeds the numerical range") from exc


@dataclass(frozen=True)
class ProbabilityContract:
    """Explicit common threshold/noise semantics for one comparison.

    The only supported model is the existing worst-critical-pad signed-scalar
    interval surrogate, NOT radial Gaussian noise or an all-pad probability.
    Threshold and residual sigma are scalars in um shared by events/sites.
    Set sigma explicitly to zero when no distinct residual noise is declared.
    Products require independent interface failures conditional on the supplied
    joint realization and site, not unconditional interface independence.
    """

    model: str
    threshold_semantics: str
    threshold_um: float
    residual_sigma_um: float
    conditional_interface_independence: bool

    def __post_init__(self):
        if self.model != "worst_critical_pad_signed_scalar_interval":
            raise ValueError("require worst_critical_pad_signed_scalar_interval model")
        if self.threshold_semantics != "closed_symmetric_interval_um":
            raise ValueError("require closed_symmetric_interval_um threshold semantics")
        if self.conditional_interface_independence is not True:
            raise ValueError("conditional interface independence must be explicitly True")
        for name in ("threshold_um", "residual_sigma_um"):
            value = _numeric(getattr(self, name), name)
            if value.ndim != 0 or value < 0:
                raise ValueError(f"{name} must be a nonnegative scalar in um")
            object.__setattr__(self, name, float(value))


def evaluate_prebond_history(joint_events, *, expected_event_order, coordinates_um,
                            coordinates_frame_id, translation_um, rotation_rad,
                            critical_pad_mask, site_mask, probability, input_kind,
                            latent_weights=None):
    """Evaluate supplied joint draws without sorting, redrawing, or dropping any.

    ``joint_events`` is (draw,event) of PrebondEvent records. Explicit order
    must be A1 through A# with no gaps; this describes attachments, not thermal
    stages, so sequential and batch histories share the same event convention.
    Coordinates are (site,pad,2), already in the bodies' aligned frame relative
    to the declared placement pivot; this function does not choose that pivot.
    Translation is exactly (draw,event,2), rotation (draw,event), with no
    implicit sharing/broadcast across draws or events. Each row is one joint
    mechanical AND placement realization shared across sites.

    ``probability`` is an explicit ProbabilityContract. Outputs compound each
    event prefix using compound_site_yield: products are inside the joint-draw
    expectation. Marginal-mean products are retained only as named diagnostics.
    ``input_kind`` is 'synthetic' or 'unqualified_supplied'; neither can produce
    a physical-readiness claim. No fabrication/pad-redundancy model is supplied.
    """
    if not isinstance(probability, ProbabilityContract):
        raise TypeError("supply an explicit ProbabilityContract")
    if input_kind not in ("synthetic", "unqualified_supplied"):
        raise ValueError("input_kind must be synthetic or unqualified_supplied")
    order = tuple(expected_event_order)
    if not order or order != tuple(f"A{i}" for i in range(1, len(order) + 1)):
        raise ValueError("expected_event_order must be contiguous A1 onward")
    draws = tuple(tuple(events) for events in joint_events)
    if not draws:
        raise ValueError("require at least one joint realization")
    for events in draws:
        if any(not isinstance(event, PrebondEvent) for event in events):
            raise TypeError("every supplied event must be a PrebondEvent")
        if tuple(event.attachment_event for event in events) != order:
            raise ValueError("missing, duplicate, extra, or reordered attachment events")
        if any(body.frame_id != coordinates_frame_id for event in events
               for body in (event.incoming_upper, event.old_lower)):
            raise ValueError("coordinates and all bodies must share the aligned frame")
    xy = _numeric(coordinates_um, "coordinates_um")
    if xy.ndim != 3 or xy.shape[-1] != 2 or min(xy.shape[:2]) == 0:
        raise ValueError("coordinates_um must be nonempty (site,pad,2)")
    critical = _mask(critical_pad_mask, xy.shape[1], "critical_pad_mask")
    sites = _mask(site_mask, xy.shape[0], "site_mask")
    shifts = _numeric(translation_um, "translation_um")
    angles = _numeric(rotation_rad, "rotation_rad")
    shape = (len(draws), len(order))
    if shifts.shape != (*shape, 2) or angles.shape != shape:
        raise ValueError("placement draws must match (draw,event,2) and (draw,event) exactly")
    magnifications = np.array([[event_magnification(event) for event in events]
                               for events in draws])
    overlay = affine_overlay_um(xy, magnifications[:, :, None, None, :],
                                translation_um=shifts[:, :, None, None, :],
                                rotation_rad=angles[:, :, None, None])
    survival = worst_critical_pad_survival(overlay, critical_pad_mask=critical,
        threshold_um=probability.threshold_um, sigma_um=probability.residual_sigma_um)
    summaries = tuple(compound_site_yield(survival.probability[:, :stop, :], sites,
                                          latent_weights=latent_weights)
                      for stop in range(1, len(order) + 1))
    return dict(schema="yap.prebond_history_bridge.v1", input_kind=input_kind,
        status=("SYNTHETIC_INTEGRATION_ONLY" if input_kind == "synthetic"
                else "UNQUALIFIED_PREBOND_ARITHMETIC"), physical_readiness=False,
        overlay_scope="bow_induced_flattening_not_total_face_strain",
        event_order=order, sample_count=len(draws), samples_dropped=0,
        probability_contract=asdict(probability), magnification=magnifications,
        site_mask=sites.copy(),
        latent_weights=(None if latent_weights is None else _numeric(latent_weights, "latent_weights")),
        overlay_um=overlay, conditional_probabilities=survival.probability,
        worst_pad_index=survival.pad_index, cumulative_summaries=summaries)


def compose_d2d_yield(prebond, *, fabrication_records, expected_die_instances,
                     particle_probabilities, fabrication_source_id, particle_source_id,
                     pad_fractions, factorization_semantics):
    """Total-first, unqualified composition of explicitly supplied components.

    Fabrication records follow YAP.D2W.chiplet_yield.chiplet_yield_records:
    one UNSELECTED physical die per record, with negative-binomial probability
    (1 + D*Acritical/alpha)**(-alpha). Its existing chiplet_stack_yield helper
    multiplies those records ONCE, outside the bonding product. This bridge
    neither reads designs nor computes/guesses their D, Acritical or alpha.

    Particle probabilities must already be (draw,event,site), e.g. supplied
    from D2W.defect_yield_calculator.interface_particle_yield_from_pad_layout.
    That model uses exp(-expected_fatal_particles) with actual critical-pad
    geometry and explicit redundancy rules. Fractions alone cannot supply it.
    Critical chip area is NOT the critical-pad fraction. The explicit 30/50/20
    pad ledger is bookkeeping here, not a generated pad mask or failure rule.

    Require independent fabrication and conditionally independent overlay/
    particle failures given each joint draw/site. Correlation through the
    shared draws is retained. The reported total includes ONLY these supplied
    components; it is not an ESD/recess/other-mechanism model or physical yield.
    """
    if not isinstance(prebond, Mapping) or prebond.get("schema") != "yap.prebond_history_bridge.v1":
        raise ValueError("require an uncompounded prebond bridge result")
    if factorization_semantics != "independent_fabrication_conditional_bonding_mechanisms":
        raise ValueError("declare independent_fabrication_conditional_bonding_mechanisms")
    for source in (fabrication_source_id, particle_source_id):
        if not isinstance(source, str) or not source.strip():
            raise ValueError("explicit fabrication and particle source IDs are required")
    fractions = dict(critical=.3, power=.5, dummy=.2)
    if not isinstance(pad_fractions, Mapping) or set(pad_fractions) != set(fractions):
        raise ValueError("require exactly critical/power/dummy pad fractions; no vacant pads")
    for name, expected in fractions.items():
        value = _numeric(pad_fractions[name], f"{name} pad fraction")
        if value.ndim or float(value) != expected:
            raise ValueError("the explicit pad fractions must be 30/50/20 percent")
    expected = tuple(expected_die_instances)
    if (any(not isinstance(name, str) or not name.strip() for name in expected)
            or len(set(expected)) != len(expected)
            or len(expected) != len(prebond["event_order"]) + 1):
        raise ValueError("require one unique expected instance per physical die")
    records = tuple(dict(record) for record in fabrication_records)
    names = tuple(record.get("instance") for record in records)
    if len(names) != len(expected) or any(names.count(name) != 1 for name in expected):
        raise ValueError("missing, duplicate, or unexpected fabrication die records")
    for record in records:
        if record.get("known_good_die") is not False:
            raise ValueError("fabrication-inclusive yield requires unselected, not KGD, records")
        value = _numeric(record.get("yield"), "fabrication yield")
        if value.ndim or not 0 <= float(value) <= 1:
            raise ValueError("fabrication yield must be a scalar probability")
    particle = _numeric(particle_probabilities, "particle_probabilities")
    overlay = _numeric(prebond["conditional_probabilities"], "conditional_probabilities")
    if particle.shape != overlay.shape or np.any((particle < 0) | (particle > 1)):
        raise ValueError("particle probabilities must match the exact joint draw/event/site axes")
    mask, weights = prebond["site_mask"], prebond["latent_weights"]
    overlay_summary = compound_site_yield(overlay, mask, latent_weights=weights)
    particle_summary = compound_site_yield(particle, mask, latent_weights=weights)
    bonding_summary = compound_site_yield(overlay * particle, mask, latent_weights=weights)
    # Import only the existing scalar aggregation; never call its design-file reader or sampler.
    from YAP.D2W.chiplet_yield import chiplet_stack_yield

    fabrication = chiplet_stack_yield(records)
    return dict(fabrication_inclusive_total_yield=fabrication * bonding_summary.registered_site_yield,
        bonding_yield=bonding_summary.registered_site_yield, fabrication_stack_yield=fabrication,
        overlay_yield=overlay_summary.registered_site_yield,
        particle_yield=particle_summary.registered_site_yield,
        schema="yap.d2d_declared_components.v1", status=prebond["status"], physical_readiness=False,
        primary_metric="fabrication_inclusive_total_yield",
        declared_components=("fabrication", "bow_overlay", "particle"),
        other_failure_mechanisms_included=False, pad_fractions=fractions,
        fabrication_records=records, fabrication_source_id=fabrication_source_id,
        particle_source_id=particle_source_id, factorization_semantics=factorization_semantics,
        event_order=prebond["event_order"], sample_count=prebond["sample_count"])
