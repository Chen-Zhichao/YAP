"""Opt-in, fixed-context RT response distillation for initial-bow sampling.

This approximates complete teacher histories, not Cu constitutive states. The
current input space is one spherical natural-bow amplitude per die, with virgin
incoming Cu. Hot transients, other geometries and arbitrary twist are excluded.
Finite checks are empirical evidence, not a certified error bound or MAPDL test.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np

try:
    from YAP.history_bow_yield import affine_overlay_um
except ModuleNotFoundError:  # Direct execution from the YAP source directory.
    from history_bow_yield import affine_overlay_um


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _numbers(value, name):
    if not isinstance(value, np.ndarray):
        items = np.asarray(value, dtype=object)
        if any(isinstance(x, (str, bool, np.bool_)) for x in items.flat):
            raise ValueError(f"{name}: strings and Booleans are not numbers")
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name}: finite real numbers required")
    return np.array(array, dtype=float, copy=True)


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


class InitialBowResponse:
    """Diagonal-quadratic RT response; no solver is called during inference.

    Rows correspond to JOINT draws, retained throughout all attachments/cycles.
    The coefficient coordinates are physical bow divided by frozen scales, not
    a caller's covariance eigenbasis. Changing a sampling distribution does not
    change those coordinates or grant support outside the checked domain.
    """

    def __init__(self, coefficients, scale_um, labels, dependencies, context,
                 support_radius, qualification=None):
        scale = _numbers(scale_um, "scale_um")
        coeff = _numbers(coefficients, "coefficients")
        if scale.ndim != 1 or not len(scale) or np.any(scale <= 0):
            raise ValueError("one positive normalization scale per die is required")
        labels = tuple(labels)
        if not labels or any(not isinstance(k, str) or not k for k in labels) \
                or len(set(labels)) != len(labels):
            raise ValueError("output labels must be unique nonempty strings")
        if coeff.shape != (1+2*len(scale), len(labels)):
            raise ValueError("coefficient dimensions do not match inputs/outputs")
        if len(dependencies) != len(labels):
            raise ValueError("one dependency declaration per output is required")
        deps = []
        for declared in dependencies:
            values = tuple(declared)
            if len(set(values)) != len(values) or any(
                isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
                or x < 0 or x >= len(scale) for x in values
            ):
                raise ValueError("dependencies must be distinct zero-based die indices")
            deps.append(tuple(int(x) for x in values))
        radius = _numbers(support_radius, "support_radius")
        if radius.ndim != 0 or radius <= 0:
            raise ValueError("positive finite support radius required")
        if not isinstance(context, dict) or not context:
            raise ValueError("explicit geometry/material/process context required")
        self._context_json = _json(context)
        self.scale_um = _readonly(scale)
        self.coefficients = _readonly(coeff)
        self.labels = labels
        self.dependencies = tuple(deps)
        self.support_radius = float(radius)
        self._qualification = deepcopy(qualification)
        for j, allowed in enumerate(self.dependencies):
            for i in set(range(self.die_count))-set(allowed):
                if coeff[1+i, j] != 0 or coeff[1+self.die_count+i, j] != 0:
                    raise ValueError("coefficient violates the declared causal dependencies")
        if qualification is not None:
            self._check_qualification(qualification)

    @property
    def context(self):
        return json.loads(self._context_json)

    @property
    def context_id(self):
        return hashlib.sha256(self._context_json.encode("ascii")).hexdigest()

    @property
    def die_count(self):
        return len(self.scale_um)

    @property
    def model_sha256(self):
        value = dict(context=self.context, labels=self.labels, dependencies=self.dependencies,
                     scale_um=self.scale_um.tolist(), coefficients=self.coefficients.tolist(),
                     basis="physical_bow_diagonal_quadratic_v1")
        return hashlib.sha256(_json(value).encode("ascii")).hexdigest()

    @property
    def qualification(self):
        return deepcopy(self._qualification)

    def _check_qualification(self, report):
        if not isinstance(report, dict) or report.get("status") != "PASS_RT_FINITE_CHECKS_ONLY" \
                or report.get("context_id") != self.context_id \
                or report.get("model_sha256") != self.model_sha256:
            raise ValueError("a passing finite-check report bound to this model is required")
        if report.get("support_radius") != self.support_radius:
            raise ValueError("qualification does not bind the requested support radius")

    def _coordinates(self, bows_um):
        bows = _numbers(bows_um, "bows_um")
        if bows.ndim != 3 or bows.shape[1:] != (self.die_count, 2) or not len(bows):
            raise ValueError("bows_um must be nonempty (joint_draw,die,2)")
        if not np.array_equal(bows[:, :, 0], bows[:, :, 1]):
            raise ValueError("this response supports equal x/y natural bow and zero twist only")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                z = bows[:, :, 0]/self.scale_um
                squared = z*z
            except FloatingPointError as exc:
                raise ValueError("bow normalization overflow") from exc
        return z, squared

    @classmethod
    def fit(cls, bows_um, values, labels, dependencies, context, *,
            scale_um=50./3, support_radius):
        bows = _numbers(bows_um, "bows_um")
        y = _numbers(values, "values")
        if bows.ndim != 3 or bows.shape[2] != 2 or y.shape != (len(bows), len(labels)):
            raise ValueError("training dimensions do not match")
        n = bows.shape[1]
        scale = _numbers(scale_um, "scale_um")
        if scale.ndim == 0:
            scale = np.full(n, float(scale))
        template = cls(np.zeros((1+2*n, len(labels))), scale, labels, dependencies,
                       context, support_radius)
        z, squared = template._coordinates(bows)
        center = np.flatnonzero(np.all(z == 0, axis=1))
        if len(center) != 1:
            raise ValueError("exactly one zero-bow anchor is required")
        features = np.column_stack((z, squared))
        _, singular, _ = np.linalg.svd(features, full_matrices=False)
        if len(singular) != 2*n or singular[-1] <= singular[0]*1e-10:
            raise ValueError("rank-deficient or poorly conditioned training design")
        baseline = y[center[0]].copy()
        slope = np.linalg.lstsq(features, y-baseline, rcond=None)[0]
        coefficients = np.vstack((baseline, slope))
        for j, allowed in enumerate(template.dependencies):
            for i in set(range(n))-set(allowed):
                coefficients[[1+i, 1+n+i], j] = 0.
            # Virgin, stress-free incoming bodies retain their specified shape.
            match = re.fullmatch(r"A(\d+)/(incoming_free|old_free)/(edge_[xy]|corner_p[pm])_um", labels[j])
            if match and (match[2] == "incoming_free" or match[1] == "1"):
                die = int(match[1]) if match[2] == "incoming_free" else 0
                if die not in allowed:
                    raise ValueError("natural-body output has the wrong causal dependency")
                coefficients[:, j] = 0.
                coefficients[1+die, j] = scale[die]*(2 if match[3].startswith("corner") else 1)
        return cls(coefficients, scale, labels, dependencies, context, support_radius)

    def preview(self, bows_um, context_id):
        """Ungated evaluation for scoring only; not a population-admission API."""
        if context_id != self.context_id:
            raise ValueError("geometry/material/process context mismatch")
        z, squared = self._coordinates(bows_um)
        with np.errstate(over="raise", invalid="raise"):
            try:
                values = np.column_stack((np.ones(len(z)), z, squared)) @ self.coefficients
            except FloatingPointError as exc:
                raise ValueError("response overflow") from exc
        if not np.isfinite(values).all():
            raise ValueError("nonfinite response")
        return values

    def evaluate(self, bows_um, context_id, *, allow_unresolved=False):
        """Checked-domain batch prediction; never discard or redraw an input.

        By default every input must be supported. Explicit partial mode keeps
        unsupported rows as NaN with masks; downstream probability estimates
        must retain that unresolved mass. It does not bound bow moments.
        """
        self._check_qualification(self._qualification)
        if context_id != self.context_id:
            raise ValueError("geometry/material/process context mismatch")
        if type(allow_unresolved) is not bool:
            raise ValueError("allow_unresolved must be Boolean")
        z, squared = self._coordinates(bows_um)
        supported = np.sqrt(squared.sum(axis=1)) <= self.support_radius
        if not supported.all() and not allow_unresolved:
            raise ValueError(f"{int((~supported).sum())} draws need full-history evaluation; none were dropped")
        values = np.full((len(z), len(self.labels)), np.nan)
        values[supported] = np.column_stack((np.ones(supported.sum()), z[supported],
                                             squared[supported])) @ self.coefficients
        if not np.isfinite(values[supported]).all():
            raise ValueError("nonfinite response")
        return dict(values=values, labels=self.labels, resolved_mask=supported,
                    outside_domain_mask=~supported, sample_count=len(z), samples_dropped=0,
                    unresolved_count=int((~supported).sum()),
                    status="COMPLETE_FINITE_CHECK_SCOPE" if supported.all() else "INCOMPLETE_NEEDS_FALLBACK",
                    physical_gaussianity_proven=False, mapdl_validation=False)

    def with_qualification(self, report, *, support_radius=None):
        return type(self)(self.coefficients, self.scale_um, self.labels, self.dependencies,
                          self.context, self.support_radius if support_radius is None else support_radius,
                          qualification=report)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        path = directory/"model.npz"
        np.savez(path, coefficients=self.coefficients, scale_um=self.scale_um)
        manifest = dict(schema="yap.initial_bow_rt_response.v1", context=self.context,
                        context_id=self.context_id, labels=self.labels, dependencies=self.dependencies,
                        support_radius=self.support_radius, qualification=self.qualification,
                        model_sha256=self.model_sha256,
                        array_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        scope="fixed_context_RT_and_prebond_only_no_internal_state_or_hot_trajectory")
        (directory/"MANIFEST.json").write_text(_json(manifest)+"\n", encoding="ascii")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory/"MANIFEST.json").read_text())
        path = directory/"model.npz"
        if manifest["schema"] != "yap.initial_bow_rt_response.v1" \
                or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["array_sha256"]:
            raise ValueError("response package checksum/schema mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            result = cls(arrays["coefficients"], arrays["scale_um"], manifest["labels"],
                         manifest["dependencies"], manifest["context"], manifest["support_radius"],
                         qualification=manifest["qualification"])
        if result.model_sha256 != manifest["model_sha256"] or result.context_id != manifest["context_id"]:
            raise ValueError("response model/physical-context checksum mismatch")
        return result


# Sampling and overlay scoring for the fixed-context response.
def overlay_context_signature(context, metric, hull_um):
    """Bind the numerical metric and scored critical-pad coordinates, not a label."""
    values = dict(operator_sha256=metric["operator_sha256"], frame=context["frame"],
        footprint_mm=context["footprint_mm"],
        bodies=[[dict(layer_names=body["layer_names"], chi=float(body["chi"])) for body in pair]
                for pair in metric["coefficients_old_incoming"]],
        critical_hull_um=_numbers(hull_um, "hull_um").tolist())
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("ascii")).hexdigest()


def sample_prebond_overlay(model, bows_um, metric, hull_um, translation_um,
                           rotation_rad, *, threshold_um, overlay_budget_um=.001,
                           chunk_size=4096):
    """RT responses plus empirical overlay-survival intervals for joint draws.

    Placement arrays must be (draw,attachment,2) and (draw,attachment), with
    the same draw order as bows. Every hull point is a critical pad. Reuse the
    saved critical hull only for its exact pad layout and operator context.
    No extra placement/residual noise or probability-factor independence is
    silently introduced. The supported threshold is deterministic, as in the
    current BI10 smoke; fabrication and particle yields remain separate.
    """
    if not isinstance(model, InitialBowResponse):
        raise TypeError("a qualified RT response model is required")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("positive integer chunk_size required")
    context = model.context
    if metric["operator_sha256"] != context["input_sha256"]:
        raise ValueError("elastic metric and response operators do not match")
    pairs = metric["coefficients_old_incoming"]
    events = len(pairs)
    if events != model.die_count-1 or events != 3 or any(len(p) != 2 for p in pairs):
        raise ValueError("this bridge requires the matched three-attachment metric")
    for cycle, pair in enumerate(pairs, 1):
        layers = context["geometry"]["layers"]
        expected = ([p["name"] for p in layers if p["die_index"] < cycle],
                    [p["name"] for p in layers if p["die_index"] == cycle])
        if any(body["layer_names"] != expected[side] for side, body in enumerate(pair)):
            raise ValueError("free-body layer identities differ from the metric reference")
    hull = _numbers(hull_um, "hull_um")
    halfspan = 500.*_numbers(context["footprint_mm"], "footprint_mm")
    if hull.ndim != 2 or hull.shape[1] != 2 or len(hull) < 3 \
            or halfspan.shape != (2,) or np.any(halfspan <= 0) \
            or np.any(np.abs(hull) > halfspan):
        raise ValueError("valid critical hull inside the footprint required")
    threshold = _numbers(threshold_um, "threshold_um")
    budget = _numbers(overlay_budget_um, "overlay_budget_um")
    if threshold.ndim or budget.ndim or threshold < 0 or budget <= 0:
        raise ValueError("nonnegative scalar threshold and positive error budget required")
    evidence = model.qualification
    if not evidence or evidence.get("overlay_budget_um") is None:
        raise ValueError("the response has no qualified overlay budget")
    qualified_budget = _numbers(evidence["overlay_budget_um"], "qualified overlay budget")
    if qualified_budget.ndim or qualified_budget <= 0 or budget < qualified_budget:
        raise ValueError("runtime error budget must not be smaller than the qualified overlay budget")
    if evidence.get("overlay_context_sha256") != overlay_context_signature(context, metric, hull):
        raise ValueError("metric or critical-hull context differs from the overlay qualification")
    predicted = model.evaluate(bows_um, model.context_id, allow_unresolved=True)
    count = predicted["sample_count"]
    shifts = _numbers(translation_um, "translation_um")
    angles = _numbers(rotation_rad, "rotation_rad")
    if shifts.shape != (count, events, 2) or angles.shape != (count, events):
        raise ValueError("placement arrays must retain joint-draw and attachment order")
    columns = {name: i for i, name in enumerate(model.labels)}
    chi = _numbers([[p[0]["chi"], p[1]["chi"]] for p in pairs], "chi")
    old_columns = [[columns[f"A{i}/old_free/edge_{a}_um"] for a in "xy"] for i in range(1, 4)]
    new_columns = [[columns[f"A{i}/incoming_free/edge_{a}_um"] for a in "xy"] for i in range(1, 4)]
    q_columns = [[columns[f"A{i}/flatten_old_minus_incoming_{a}_ppm"]
                  for a in ("x", "y", "xy")] for i in range(1, 4)]
    worst = np.full((count, events), np.nan)
    magnifications = np.full((count, events, 2), np.nan)
    resolved = np.flatnonzero(predicted["resolved_mask"])
    for start in range(0, len(resolved), chunk_size):
        indices = resolved[start:start+chunk_size]
        values = predicted["values"][indices]
        old, incoming, q = values[:, old_columns], values[:, new_columns], values[:, q_columns]
        with np.errstate(over="raise", invalid="raise"):
            e = -1e-6*q[:, :, :2] + (chi[None, :, 1, None]*incoming**2
                                     - chi[None, :, 0, None]*old**2)/halfspan**2
        vector = affine_overlay_um(hull[None, None, :, :], e[:, :, None, :],
            translation_um=shifts[indices, :, None, :], rotation_rad=angles[indices, :, None])
        vector -= .5e-6*q[:, :, None, 2, None]*hull[None, None, :, ::-1]
        magnifications[indices] = e
        worst[indices] = np.linalg.norm(vector, axis=-1).max(axis=-1)
    in_domain = predicted["resolved_mask"][:, None]
    definitely_pass = in_domain & (worst+float(budget) < float(threshold))
    definitely_fail = in_domain & (worst-float(budget) > float(threshold))
    lower = np.cumprod(definitely_pass.astype(float), axis=1).mean(axis=0)
    upper = np.cumprod((~definitely_fail).astype(float), axis=1).mean(axis=0)
    return dict(response=predicted, magnification=magnifications, worst_overlay_um=worst,
                overlay_survival_lower=lower, overlay_survival_upper=upper,
                event_unresolved_mask=~(definitely_pass | definitely_fail),
                outside_domain_count=predicted["unresolved_count"], sample_count=count,
                samples_dropped=0, physical_yield_qualified=False,
                status="EMPIRICAL_SURROGATE_BUDGET_INTERVAL_NOT_CERTIFIED_PHYSICAL_YIELD",
                bound_assumption="in-domain overlay vector error <= empirical budget",
                overlay_budget_um=float(budget), threshold_um=float(threshold))
