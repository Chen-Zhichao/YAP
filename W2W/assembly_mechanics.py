"""Assembly-tree adapter for the existing all-Gauss reference mechanics engine.

No constitutive law or equilibrium solver is defined here. Initial bow uses
the existing stress-free natural-q6 approximation, not a fabrication stress
field. New assembly paths and wafer-scale use remain unvalidated predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import importlib.util
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from .assembly_order import (
    FRAME, DIE_FRAME, CONVENTION, _check_plan, archive_template, content_id,
    is_die_plan, is_f2b_plan, is_prepared_plan, component_key, footprint_mm,
    _check_recipe_request,
)

def _backend():
    try:
        from YAP import history_mechanics_backend
    except ModuleNotFoundError:
        import history_mechanics_backend
    return history_mechanics_backend


def native(name):
    return _backend()._import_mechanics_module(name)


def _configure_mechanics_runtime_environment(**kwargs):
    return _backend()._configure_mechanics_runtime_environment(**kwargs)


def _material_runtime(binding=None):
    if binding is not None:
        module = sys.modules.get(type(binding).__module__)
        if module is not None and Path(getattr(module, "__file__", "")) == Path(__file__).parents[1] / "mechanics_runtime.py":
            return module
    try:
        from YAP import mechanics_runtime
    except ModuleNotFoundError:
        import mechanics_runtime
    return mechanics_runtime


@lru_cache(maxsize=1)
def _corrected_native_modules():
    return native("exp12_all_gauss_exact_acceleration"), native("exp12_all_gauss_reduced_fe2")


def _require_unaccelerated_corrected_material():
    exact, reduced = _corrected_native_modules()
    names = ("evaluate", "evaluate_many", "tangent_action", "assembled_bubble_tangent",
             "assembled_coordinate_tangent")
    patched = any(getattr(reduced.AllGaussReducedFE2Layer, name, None) is not
                  getattr(exact, "_ORIGINAL_" + name.upper()) for name in names)
    if (exact._INSTALLED or exact.reduced is not reduced or patched
            or reduced.trial_update_batch is not exact._ORIGINAL_TRIAL_UPDATE_BATCH):
        raise RuntimeError("corrected material requires an unpatched native runtime; "
                           "legacy global acceleration is unsupported and was not uninstalled")


def require_corrected_material_runtime(library):
    """Refuse lost bindings and installed legacy acceleration before execution."""
    corrected = False
    for operator in library.operators.values():
        if (getattr(operator, "corrected_material_binding", None) is not None
                or getattr(operator, "material_kernels", None) is not None):
            binding = getattr(operator, "corrected_material_binding", None)
            _material_runtime(binding).preserve_corrected_material_binding(operator, operator)
            corrected = True
    if corrected:
        _require_unaccelerated_corrected_material()
    return corrected


class LayerClockOperator:
    """Keep the native operator unchanged, with one clock per physical layer.

    Offsets are fixed during an entire thermal cycle, including rejected
    trials and cutbacks. Different layers must have different proxy objects:
    the native stack batches constitutive calls by operator identity.
    """

    def __init__(self, base, age_offset_s=0.0):
        self.base = base
        self.age_offset_s = float(age_offset_s)
        if (getattr(base, "corrected_material_binding", None) is not None
                or getattr(base, "material_kernels", None) is not None):
            runtime = _material_runtime(getattr(base, "corrected_material_binding", None))
            runtime.preserve_corrected_material_binding(base, self)

    def __getattr__(self, name):
        value = getattr(self.base, name)
        if name == "assembled_bubble_tangent" or (
                name == "trial" and getattr(self.base, "corrected_material_binding", None) is not None):
            return lambda *args, **kwargs: self._call(name, args, kwargs)
        return value

    def _call(self, method, args, kwargs):
        binding = getattr(self, "corrected_material_binding", None)
        if binding is not None:
            _material_runtime(binding).preserve_corrected_material_binding(self.base, self)
            _require_unaccelerated_corrected_material()
        elapsed = kwargs.get("elapsed_time_s")
        if binding is not None and elapsed is None:
            raise ValueError("corrected material requires an explicit physical endpoint clock")
        if elapsed is None:
            if self.age_offset_s != 0:
                raise ValueError("an explicit endpoint time is required for an aged layer")
        else:
            elapsed = float(elapsed) + self.age_offset_s
            if not np.isfinite(elapsed) or elapsed < 0:
                raise ValueError("invalid material endpoint age")
            kwargs["elapsed_time_s"] = elapsed
        return getattr(self.base, method)(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self._call("evaluate", args, kwargs)

    def evaluate_many(self, *args, **kwargs):
        return self._call("evaluate_many", args, kwargs)

    def tangent_action(self, *args, **kwargs):
        return self._call("tangent_action", args, kwargs)

    def assembled_coordinate_tangent(self, *args, **kwargs):
        return self._call("assembled_coordinate_tangent", args, kwargs)


@dataclass(frozen=True)
class AssemblyBody:
    placements: tuple
    topology: object
    state: object
    ages_s: dict


def _recipe(plan, temporal_profile):
    _check_plan(plan)
    _check_recipe_request(plan, temporal_profile)
    module = native("exp12_v2_process_schedule")
    if temporal_profile == "component_graded25":
        exact = module.schedule_from_v2_process_card(1, temporal_profile="exact")
        indices = (0, *range(20, 161, 20), 161, 162, 164, 168, 170, 180, 190, 200,
                   *range(220, 361, 20))
        if len(exact) != 361:
            raise ValueError("component graded25 requires the original 361-state cycle")
        schedule = tuple(exact[i] for i in indices)
    else:
        schedule = module.schedule_from_v2_process_card(1, temporal_profile=temporal_profile)
    card = module.load_v2_process_card()
    process = plan["context"]["process"]
    expected = dict(room_temperature_c=card["temperature"]["room_C"],
                    anneal_temperature_c=card["temperature"]["anneal_C"],
                    **{k: card["time_profile"][k] for k in ("ramp_up_s", "dwell_s", "cool_down_s")})
    if expected["room_temperature_c"] != 25 or any(
            not np.isclose(process[k], value, rtol=0, atol=1e-12) for k, value in expected.items()):
        raise ValueError("native formation requires the existing 25C/350C V2 thermal recipe")
    if process["prebond_policy"] != (
        "reference 300 s initial clock advance once per material birth, creep off; "
        "subsequent RT flatten/capture/release elastic; pressure not explicitly solved"
    ):
        raise ValueError("native executor requires the declared legacy frozen-history prebond policy")
    if schedule[0].time_s != 300 or schedule[0].creep_active:
        raise ValueError("unsupported V2 initial material clock")
    return schedule


def _check_library(library, geometry, *, prepared_dies=False, split_cap=False):
    if library.common_port_gauge_vectors is not None:
        raise ValueError("existing formation templates require a rigid-mode-gauged library")
    front = geometry["front_bi_thickness_um"]
    back = geometry["back_bi_thickness_um"]
    if front != back:
        raise ValueError("one BI operator currently requires equal front/back BI thickness")
    target = geometry["total_thickness_um"] if prepared_dies else geometry["target_total_thickness_um"]
    prepared = prepared_dies
    retained = target - front - back - geometry["beol_thickness_um"]
    expected = dict(bi=front, beol=geometry["beol_thickness_um"], retained_si=retained)
    if not prepared:
        cap = geometry["initial_total_thickness_um"] - front - geometry["beol_thickness_um"] - retained
        expected["bulk_si"] = cap - back if split_cap else cap
        if split_cap:
            expected["outer_si"] = back
    for key, thickness in expected.items():
        if key not in library.operators or not np.isclose(
                library.thickness_mm[key] * 1000, thickness, rtol=0, atol=1e-9):
            raise ValueError(f"missing exact {key} operator for {thickness:g} um")
    if not prepared and len(library.operators["bulk_si"].weights_mm3):
        raise ValueError("the removable substrate must be pure Si, with no Cu integration points")
    if split_cap and (len(library.operators["outer_si"].weights_mm3)
                      or library.operators["outer_si"].plastic_active):
        raise ValueError("the retained original outer slice must be pure linear Si, with no Cu")


class AssemblyExecutor:
    """One independent route; neither mutable clocks nor histories cross routes."""

    def __init__(self, plan, library, *, temporal_profile="distilled26", solver_options=None):
        require_corrected_material_runtime(library)
        _check_plan(plan)
        self.plan = plan
        self.geometry = plan["context"]["geometry"]
        self.prepared_dies = is_die_plan(plan)
        self.prepared_bodies = is_prepared_plan(plan)
        self.split_cap = is_f2b_plan(plan)
        self.ids_key = component_key(plan)
        self.footprint = footprint_mm(plan)
        _check_library(library, self.geometry, prepared_dies=self.prepared_dies, split_cap=self.split_cap)
        self.schedule = _recipe(plan, temporal_profile)
        self.profile = temporal_profile
        self.library = library
        self.placement_api = native("exp12_v2_production_sequential_process")
        self.formation = native("run_exp12_free_membrane_sequential")
        self.bonding = native("exp12_free_membrane_bonding")
        self.transactions = native("exp12_all_gauss_process_transactions")
        self.production = native("run_exp12_v2_production_sequential_all_gauss_v1")
        self.thermal = native("exp12_all_gauss_thermal_advance")
        self.export = native("exp12_prebond_export")
        self.state_api = native("run_exp12_rt_flattening_short_comparison")
        cache_type = native("exp12_all_gauss_periodic_stack").AllGaussStackReferenceCache
        self.solver_options = dict(linear_solver="bfgs_inverse", convergence_measure="residual",
                                  tolerance=1e-9, max_iterations=256,
                                  reference_cache=cache_type(maximum_entries=1024))
        self.solver_options.update(solver_options or {})
        self.thermal_policy = replace(self.thermal.DEFAULT_THERMAL_ADVANCE_POLICY,
                                      linear_solver=self.solver_options["linear_solver"])
        self.operators = {}
        self.thicknesses = {}
        self.births = {}
        shape_type = native("exp12_mindlin_generalized_contract").NaturalBirthShape
        for entry in plan["initialization"]["dies" if self.prepared_dies else "wafers"]:
            q6 = np.array(entry["initial_q6"], dtype=float)
            # Manufacturing and native paper-bow helpers have opposite signs.
            q6[3:] *= -1
            self.births[entry["die" if self.prepared_dies else "wafer"] - 1] = shape_type(*q6)
        self.bodies = {}

    def _local_library(self):
        return replace(self.library, operators=self.operators, thickness_mm=self.thicknesses)

    def _placement(self, wafer, role, key, z, birth_reference=None):
        prefix = "die" if self.prepared_dies else "wafer"
        name = f"{prefix}{wafer:02d}_{role}"
        if name not in self.operators:
            self.operators[name] = LayerClockOperator(self.library.operators[key])
            self.thicknesses[name] = self.library.thickness_mm[key]
        return self.placement_api._placement(self._local_library(), name=name,
            operator_key=name, die_index=wafer - 1, role=role, z_bottom_mm=z,
            thickness_orientation=-1 if role == "back_bi" else 1,
            birth_reference_z_mm=birth_reference)

    def _topology(self, placements, previous=None):
        topology = self.placement_api._topology(self._local_library(), tuple(placements), self.births)
        if previous is not None:
            topology = self.bonding.inherit_formation_offsets(previous, topology)
        return topology

    def leaf(self, wafer):
        # Independent computational datums, not an initial global assembly:
        # full-thickness bodies may overlap numerically but are never joined
        # until their mating surfaces coincide. Aged children are not translated.
        target = self.geometry["total_thickness_um"] if self.prepared_dies else self.geometry["target_total_thickness_um"]
        initial = target if self.prepared_dies else self.geometry["initial_total_thickness_um"]
        bottom = (wafer - 1) * target / 1000
        birth_reference = bottom + initial / 2000
        cursor = bottom
        placements = []
        final_layer = ("back_bi", "bi") if self.prepared_bodies else ("removable_bulk_si", "bulk_si")
        cap_layers = (("outer_retained_si", "outer_si"), final_layer) if self.split_cap else (final_layer,)
        for role, key in (("front_bi", "bi"), ("beol", "beol"), ("retained_si", "retained_si"), *cap_layers):
            p = self._placement(wafer, role, key, cursor, birth_reference)
            placements.append(p)
            cursor = p.z_top_mm
        topology, state = self.formation.virgin_die_state(
            self._local_library(), tuple(placements), self.births)
        bow = state.generalized_coordinates[3:5] * np.asarray(self.footprint)**2 * 1000 / 8
        if not np.allclose(bow, self.plan["context"]["initial_bows_um"][wafer - 1], rtol=1e-10, atol=1e-9):
            raise ValueError("native leaf does not recover the requested signed initial bow")
        return AssemblyBody(tuple(placements), topology, state,
                            {p.name: self.schedule[0].time_s for p in placements})

    def checkpoint(self, body):
        return content_id(dict(state=self.state_api.state_digest(body.state),
            topology=self.formation.topology_signature(body.topology),
            formation=self.formation.formation_signature(body.topology), ages_s=body.ages_s))

    def _set_clock(self, body, schedule_start):
        for instance in body.topology.instances:
            instance.operator.age_offset_s = body.ages_s[instance.name] - schedule_start

    def _release(self, topology, state, ages):
        body = AssemblyBody((), topology, state, ages)
        self._set_clock(body, self.schedule[0].time_s)
        _, released = self.production._free_equilibrium_trial(topology, state,
            temperature_c=25., delta_time_s=0., elapsed_time_s=self.schedule[0].time_s,
            plastic_active=False, solver_options=self.solver_options)
        return released

    def prepare_back(self, body):
        before = self.checkpoint(body)
        if self.prepared_bodies:
            if body.placements[-1].role != "back_bi":
                raise ValueError("prepared body must already own its backside BI")
            return body, dict(policy="prefabricated_back_BI_no_thinning_or_deposition",
                              retained_material_ages_unchanged=True,
                              source_checkpoint_unchanged=True)
        if self.split_cap and body.placements[-1].role == "outer_retained_si":
            retained = body.placements[:-1]
        else:
            if body.placements[-1].role != "removable_bulk_si":
                raise ValueError("back preparation requires an exposed removable Si cap")
            if self.split_cap and body.placements[-2].role != "outer_retained_si":
                raise ValueError("F2B preparation requires both original pure-Si cap slices")
            retained = body.placements[:-2] if self.split_cap else body.placements[:-1]
        top = self._topology(retained, body.topology)
        removal = self.transactions.remove_top_all_gauss_biaxial_modules(body.topology, top, body.state)
        ages = {p.name: body.ages_s[p.name] for p in retained}
        released = self._release(top, removal.committed, ages)
        wafer = retained[-1].die_index + 1
        new = self._placement(wafer, "back_bi", "bi", retained[-1].z_top_mm)
        active = retained + (new,)
        new_top = self._topology(active, top)
        addition = self.transactions.append_stress_free_all_gauss_biaxial_modules(top, new_top, released)
        ages = {**ages, new.name: self.schedule[0].time_s}
        state = self._release(new_top, addition.committed, ages)
        if self.checkpoint(body) != before:
            raise ValueError("preparation mutated the committed source body")
        report = dict(removal=self.production._transfer_record("thin", removal),
                      deposition=self.production._transfer_record("back_BI", addition),
                      retained_material_ages_unchanged=True, source_checkpoint_unchanged=True)
        return AssemblyBody(active, new_top, state, ages), report

    def finish_outer_back(self, body):
        """Remove the top bulk cap, retaining the original 1.5-um Si state."""
        if self.split_cap and body.placements[-1].role == "outer_retained_si":
            return body, dict(policy="original_outer_Si_already_retained_no_further_removal",
                original_outer_Si_retained=True, silicon_deposition=False, new_back_BI=False,
                retained_material_ages_unchanged=True, source_checkpoint_unchanged=True,
                carrier_end_state="removed")
        if not self.split_cap or body.placements[-1].role != "removable_bulk_si" \
                or body.placements[-2].role != "outer_retained_si":
            raise ValueError("outer finishing requires the original split pure-Si cap")
        before = self.checkpoint(body)
        retained = body.placements[:-1]
        top = self._topology(retained, body.topology)
        removal = self.transactions.remove_top_all_gauss_biaxial_modules(body.topology, top, body.state)
        ages = {p.name: body.ages_s[p.name] for p in retained}
        state = self._release(top, removal.committed, ages)
        if self.checkpoint(body) != before:
            raise ValueError("outer finishing mutated its committed source body")
        return AssemblyBody(retained, top, state, ages), dict(
            removal=self.production._transfer_record("outer_back_thin", removal),
            original_outer_Si_retained=True, silicon_deposition=False, new_back_BI=False,
            retained_material_ages_unchanged=True, source_checkpoint_unchanged=True,
            carrier_end_state="removed")

    def _assert_geometry(self, body, expected):
        placements = body.placements
        if any(not np.isclose(a.z_top_mm, b.z_bottom_mm, rtol=0, atol=1e-12)
               for a, b in zip(placements, placements[1:])):
            raise ValueError("assembly body contains a gap or overlapping joined layers")
        actual = [1000 * sum(p.z_top_mm - p.z_bottom_mm for p in placements if p.die_index == wafer - 1)
                  for wafer in expected[self.ids_key]]
        if not np.allclose(actual, expected["total_thicknesses_um"], rtol=0, atol=1e-9):
            raise ValueError("actual assembly thicknesses disagree with the process ledger")

    def join(self, lower, upper, event):
        originals = (self.checkpoint(lower), self.checkpoint(upper))
        if not np.isclose(lower.placements[-1].z_top_mm, upper.placements[0].z_bottom_mm,
                          rtol=0, atol=1e-12):
            raise ValueError("F2B mating faces do not coincide")
        if self.split_cap and (lower.placements[-1].role != "back_bi"
                               or upper.placements[0].role != "front_bi"):
            raise ValueError("F2B joining requires a prepared lower BACK and upper FRONT")
        placements = lower.placements + upper.placements
        joined = self._topology(placements)
        start = self.schedule[0].time_s
        self._set_clock(lower, start)
        self._set_clock(upper, start)
        topology, state, _, record = self.formation.form_free_pair(
            lower.topology, lower.state, upper.topology, upper.state, joined,
            event=event, face_z_mm=lower.placements[-1].z_top_mm,
            clock_times=(start, start, start), footprint_mm=self.footprint,
            cell_area_mm2=self.library.cell_area_mm2, solver_options=self.solver_options)
        if originals != (self.checkpoint(lower), self.checkpoint(upper)):
            raise ValueError("bond formation mutated a committed child")
        ages = {**lower.ages_s, **upper.ages_s}
        return AssemblyBody(placements, topology, state, ages), record

    def anneal(self, body):
        before = self.checkpoint(body)
        start = self.schedule[0].time_s
        self._set_clock(body, start)
        state = body.state.copy()
        diagnostics = self.thermal.ThermalAdvanceDiagnostics()
        for previous, target in zip(self.schedule, self.schedule[1:]):
            _, state, step_diagnostics = self.thermal.advance_all_gauss_thermal_step(
                body.topology, state, previous, target, allow_cutback=True,
                maximum_cutback_depth=8, policy=self.thermal_policy)
            diagnostics += step_diagnostics
        if self.checkpoint(body) != before:
            raise ValueError("thermal trial mutated its committed input")
        duration = self.schedule[-1].time_s - start
        ages = {name: age + duration for name, age in body.ages_s.items()}
        result = replace(body, state=state, ages_s=ages)
        return result, dict(temporal_profile=self.profile, duration_s=duration,
                            ages_before_s=body.ages_s, ages_after_s=ages,
                            diagnostics=asdict(diagnostics))

    def observation(self, body, expected, role, metric):
        abd = self.export.effective_reference_abd(body.topology, self.library.cell_area_mm2)
        raw = self.export.body_record(body_id=expected["state_id"],
            placements=[asdict(p) for p in body.placements], abd=abd,
            curvature=body.state.generalized_coordinates[3:6],
            footprint_mm=self.footprint,
            face="top" if role == "lower" else "bottom")
        levers = raw["path_effective_face_levers_mm"]
        chi = None
        if metric is not None and metric["model"] not in ("disk", "square_affine"):
            curvature = body.state.generalized_coordinates[3:6]
            if metric["model"] == "square_surrogate" and (
                    not np.isclose(curvature[0], curvature[1], rtol=1e-5, atol=1e-12)
                    or abs(curvature[2]) > 1e-12):
                raise ValueError("scalar square metric requires equal principal curvatures and zero twist")
            if metric["model"] == "square_surrogate":
                rotate = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
                if not np.allclose(rotate.T @ abd[:3, :3] @ rotate, abd[:3, :3],
                                   rtol=1e-5, atol=1e-10 * np.linalg.norm(abd[:3, :3])):
                    raise ValueError("scalar square metric requires x/y-symmetric extensional stiffness")
            chi = _metric_coefficients(metric, expected["state_id"], abd)
            if any(value is None for value in levers) or abs(raw["linear_flattening_face_strain"][2]) > 1e-10:
                raise ValueError("directional bow scoring cannot discard this body's coupled shear response")
        record = dict(state_id=expected["state_id"],
            total_thicknesses_um=expected["total_thicknesses_um"], frame_id=DIE_FRAME if self.prepared_dies else FRAME,
            bow_convention=CONVENTION, bow_um=raw["paper_center_minus_edge_bow_um"],
            face_lever_um=[None if value is None else value * 1000 for value in levers],
            face_lever_source="native_condensed_ABD_path_effective_mating_face",
            chi=chi, face="BACK" if role == "lower" else "FRONT")
        record[self.ids_key] = expected[self.ids_key]
        if metric is not None and metric["model"] in ("disk", "square_affine"):
            if metric["model"] == "disk":
                from .wafer_membrane import disk_membrane_metric
                membrane, audit = disk_membrane_metric(abd[:3, :3],
                    body.state.generalized_coordinates[3:6], self.geometry["diameter_mm"] / 2)
                metric_label = "disk"
            else:
                from D2W.die_membrane import square_membrane_metric
                membrane, audit = square_membrane_metric(abd[:3, :3],
                    body.state.generalized_coordinates[3:6], self.footprint[0] / 2)
                metric_label = "square"
            ex, ey, shear = raw["linear_flattening_face_strain"]
            linear = np.array([[ex, shear / 2], [shear / 2, ey]])
            record["affine_face_strain"] = (linear + membrane).tolist()
            # Diagonal levers are descriptive only; the matrix retains A^-1 B coupling.
            record["face_lever_um"] = (1000 * np.asarray(raw["diagonal_face_levers_mm"])).tolist()
            record["face_lever_source"] = "descriptive_diagonal_only_full_affine_matrix_used"
            raw[f"{metric_label}_membrane_metric"] = membrane.tolist()
            raw[f"{metric_label}_membrane_audit"] = audit
        return record, raw

    def summary(self, body):
        return dict(q6=body.state.generalized_coordinates[:6].tolist(),
            ages_s=body.ages_s, checkpoint_sha256=self.checkpoint(body),
            cu_point_count=sum(len(i.operator.weights_mm3) for i in body.topology.instances),
            maximum_equivalent_plastic_strain=max(float(np.max(
                layer.cu_state.equivalent_plastic_strain, initial=0.)) for layer in body.state.layers.values()),
            maximum_equivalent_creep_strain=max(float(np.max(
                layer.cu_state.equivalent_creep_strain, initial=0.)) for layer in body.state.layers.values()))

    def run_route(self, route, target, *, metric=None, progress=None):
        def fetch(expected):
            key = expected["state_id"]
            if key not in self.bodies:
                if len(expected[self.ids_key]) != 1:
                    raise ValueError("assembly event references an unavailable child history")
                self.bodies[key] = self.leaf(expected[self.ids_key][0])
            return self.bodies[key]

        for event, output in zip(route["events"], target["events"]):
            started = perf_counter()
            if progress:
                progress(f"{route['order']} {event['event']}: {event['interface_id']}")
            old = fetch(event["lower_before_preparation"])
            upper_before = fetch(event.get("upper_before_preparation", event["upper_free"]))
            upper = upper_before
            upper_preparation = None
            if "upper_before_preparation" in event:
                upper, upper_preparation = self.finish_outer_back(upper_before)
                self._assert_geometry(upper_before, event["upper_before_preparation"])
                self.bodies[event["upper_free"]["state_id"]] = upper
            lower, preparation = self.prepare_back(old)
            self._assert_geometry(old, event["lower_before_preparation"])
            self._assert_geometry(lower, event["lower_free"])
            self._assert_geometry(upper, event["upper_free"])
            self.bodies[event["lower_free"]["state_id"]] = lower
            lower_record, lower_raw = self.observation(lower, event["lower_free"], "lower", metric)
            upper_record, upper_raw = self.observation(upper, event["upper_free"], "upper", metric)
            joined, formation = self.join(lower, upper, event["event"])
            result, thermal = self.anneal(joined)
            self._assert_geometry(result, event["result"])
            self.bodies[event["result"]["state_id"]] = result
            checkpoints = {event[key]["state_id"]: self.checkpoint(body) for key, body in (
                ("lower_before_preparation", old), ("lower_free", lower),
                ("upper_free", upper), ("result", result))}
            audit = dict(preparation=preparation, formation=formation, thermal=thermal)
            if upper_preparation is not None:
                checkpoints[event["upper_before_preparation"]["state_id"]] = self.checkpoint(upper_before)
                audit["incoming_preparation"] = upper_preparation
            output.update(lower=lower_record, upper=upper_record, checkpoints=checkpoints,
                transfer_audit="embedded:audit:" + content_id(audit), audit=audit,
                raw_prebond=dict(lower=lower_raw, upper=upper_raw),
                result=self.summary(result), wall_seconds=perf_counter() - started)
        if self.split_cap:
            assembled = fetch(route["assembled"])
            final, audit = self.finish_outer_back(assembled)
            self._assert_geometry(final, route["final"])
            self.bodies[route["final"]["state_id"]] = final
            target["finalization"] = dict(route["finalization"], audit=audit,
                source_checkpoint_sha256=self.checkpoint(assembled),
                result_checkpoint_sha256=self.checkpoint(final))
            target["final"] = dict(self.summary(final),
                wafer_ids=list(route["final"]["wafer_ids"]),
                total_thicknesses_um=list(route["final"]["total_thicknesses_um"]),
                layers=[asdict(p) for p in final.placements])
        else:
            target["final"] = self.summary(fetch(route["final"]))


def _metric_coefficients(metric, state_id, abd):
    """Never silently use a square-die metric as a circular-wafer solution."""
    model = metric.get("model")
    if model == "explicit":
        if not isinstance(metric.get("source"), str) or not metric["source"].strip():
            raise ValueError("explicit metric requires a source/approximation declaration")
        values = metric.get("by_state_id", {}).get(state_id, metric.get("chi"))
        values = np.asarray(values, dtype=float)
        if values.shape != (2,) or not np.isfinite(values).all():
            raise ValueError(f"missing finite directional metric coefficients for {state_id}")
        return values.tolist()
    if model == "square_surrogate":
        value = _reference_square_metric().square_metric_coefficient(abd[:3, :3])
        return [value, value]
    raise ValueError("metric model must be explicit or square_surrogate")


@lru_cache(maxsize=1)
def _reference_square_metric():
    root = Path(native("exp12_prebond_export").__file__).parent.parent
    path = root / "200_bow_diff_vs_magnification/nonlinear_face_lever.py"
    spec = importlib.util.spec_from_file_location("yap_existing_reference_square_metric", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _implementation_hash(metric=None):
    names = ("exp12_all_gauss_process_transactions", "exp12_free_membrane_bonding",
             "run_exp12_free_membrane_sequential", "exp12_all_gauss_thermal_advance",
             "exp12_all_gauss_periodic_stack", "exp12_all_gauss_reduced_fe2",
             "exp12_cu_material_point", "exp12_v2_process_schedule", "exp12_prebond_export")
    paths = [Path(__file__), Path(__file__).with_name("assembly_operator_library.py"),
             Path(__file__).with_name("wafer_membrane.py")]
    paths.extend(Path(native(name).__file__) for name in names)
    if metric is not None and metric.get("model") in ("square_surrogate", "square_affine"):
        paths.append(_metric_source_path(metric))
    return content_id({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})


@contextmanager
def _reference_runtime(enabled, *, library=None):
    corrected = library is not None and require_corrected_material_runtime(library)
    if corrected and enabled:
        raise RuntimeError("legacy reference acceleration cannot run with corrected material bindings")
    if not enabled:
        yield None
        return
    exact = native("exp12_all_gauss_exact_acceleration")
    compact = native("exp12_ofat_compact_solver_20260913")
    if exact._INSTALLED:
        raise ValueError("reference tree runtime requires a dedicated, unpatched prediction context")
    try:
        exact.install_exact_acceleration()
        exact.enable_numba_constitutive_acceleration()
        with compact.compact_solver() as metrics:
            yield dict(exact_constitutive_kernel="existing_serial_numba",
                       compact_solver=metrics, material_equations_unchanged=True,
                       implementation_sha256={str(Path(m.__file__).resolve()):
                           hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                           for m in (exact, compact)})
    finally:
        exact.uninstall_exact_acceleration()


def run_native(plan, library, operator_provenance, *, temporal_profile="distilled26",
               metric=None, solver_options=None, progress=None, route_complete=None,
               route_indices=None):
    """Run complete physical histories; score only with declared metric inputs.

    Library objects are reused, but physical-layer proxies and committed
    states are route-local. Checkpoint hashes cover Cu/birth arrays, reference
    offsets, gauges and material ages; this function does not write restarts.
    """
    _check_plan(plan)
    if route_indices is None:
        indices = list(range(len(plan["routes"])))
    else:
        if (not isinstance(route_indices, (list, tuple)) or not route_indices
                or any(type(i) is not int or not 0 <= i < len(plan["routes"]) for i in route_indices)
                or len(set(route_indices)) != len(route_indices)):
            raise ValueError("route_indices must contain distinct valid integer route indices")
        indices = list(route_indices)
    _check_metric_policy(plan, metric)
    output = archive_template(plan)
    output["routes"] = [output["routes"][i] for i in indices]
    output["provenance"] = dict(solver_id="existing_REFERENCE_all_Gauss_tree_adapter",
        implementation_sha256=_implementation_hash(metric), operator_set_sha256=content_id(operator_provenance),
        source_transfer_audit="embedded:legacy_natural_q6_and_native_transaction_audits",
        material_clock_audit="embedded:per_layer_endpoint_ages_no_waiting_evolution",
        physics_qualification="UNVALIDATED_NEW_ASSEMBLY_PATHS_LEGACY_NATURAL_Q6_INITIALIZATION")
    output["native_configuration"] = dict(temporal_profile=temporal_profile, metric=metric,
        operator_provenance=operator_provenance, fabrication_initialization="legacy_stress_free_natural_q6",
        birth_sign_conversion="manufacturing_bending_sign_negated_for_native_paper_bow",
        carrier="ideal_free_membrane_flattening_only_no_explicit_carrier",
        anneal="free", checkpoint_kind="in_memory_state_fingerprints_not_restart_files")
    if is_f2b_plan(plan):
        output["native_configuration"].update(
            carrier=plan["context"]["carrier_policy"],
            cap_material_assumption=plan["context"]["cap_material_assumption"],
            outer_finishing=plan["context"]["outer_finishing"],
            computational_datum=plan["context"]["computational_datum"],
            assembly_accuracy_validated=False)
    if metric is not None and metric.get("model") in ("square_surrogate", "square_affine"):
        path = _metric_source_path(metric)
        output["native_configuration"]["metric_implementation_sha256"] = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()}
    process_api = native("exp12_v2_process_schedule")
    output["native_configuration"]["process_card_sha256"] = process_api.sha256_file(process_api.DEFAULT_CARD)
    with _reference_runtime(operator_provenance.get("frozen_preview") is True, library=library) as acceleration:
        output["native_configuration"]["runtime_acceleration"] = acceleration
        for index, target in zip(indices, output["routes"]):
            route = plan["routes"][index]
            executor = AssemblyExecutor(plan, library, temporal_profile=temporal_profile,
                                        solver_options=solver_options)
            output["native_configuration"]["rt_solver"] = {
                key: value for key, value in executor.solver_options.items() if key != "reference_cache"}
            output["native_configuration"]["thermal_policy"] = asdict(executor.thermal_policy)
            executor.run_route(route, target, metric=metric, progress=progress)
            if route_complete is not None:
                route_complete(output, target)
    output["status"] = "COMPLETE" if metric is not None else "MECHANICS_COMPLETE_OVERLAY_PENDING"
    output["physical_readiness"] = False
    return output


def run_native_from_settings(plan, settings, *, progress=None, worker_directory=None):
    _check_plan(plan)
    from .assembly_operator_library import (
        load_assembly_library, load_f2b_assembly_library, load_prepared_die_library, require_prepared_operator,
    )
    required = {"operator_package", "temporal_profile"}
    optional = {"metric", "layer_workers", "order_workers", "frozen_preview"}
    if not isinstance(settings, dict) or required - settings.keys() or settings.keys() - required - optional:
        raise ValueError("native settings require operator_package/temporal_profile; optional metric/layer_workers/order_workers")
    workers = settings.get("layer_workers", 1)
    if type(workers) is not int or workers < 1:
        raise ValueError("layer_workers must be a positive integer")
    order_workers = settings.get("order_workers", 1)
    if type(order_workers) is not int or order_workers < 1:
        raise ValueError("order_workers must be a positive integer")
    if order_workers > 1 and workers != 1:
        raise ValueError("parallel orders require layer_workers=1")
    preview = settings.get("frozen_preview", False)
    if type(preview) is not bool:
        raise ValueError("frozen_preview must be Boolean")
    if is_prepared_plan(plan):
        require_prepared_operator(settings["operator_package"], plan["context"]["geometry"],
                                  frozen_preview=preview)
    _configure_mechanics_runtime_environment(layer_workers=workers)
    _recipe(plan, settings["temporal_profile"])
    _check_metric_policy(plan, settings.get("metric"))
    if order_workers > 1:
        from .assembly_parallel import run_parallel
        if worker_directory is None:
            from mechanics_runtime import configure_runtime_storage
            from uuid import uuid4
            worker_directory = configure_runtime_storage()["work"] / "assembly_orders" / uuid4().hex
        return run_parallel(plan, settings, work_dir=worker_directory, progress=progress)
    options = {"frozen_preview": True} if preview else {}
    loader = (load_prepared_die_library if is_die_plan(plan) else
              load_f2b_assembly_library if is_f2b_plan(plan) else load_assembly_library)
    library, provenance = loader(settings["operator_package"], plan["context"]["geometry"], **options)
    return run_native(plan, library, provenance, temporal_profile=settings["temporal_profile"],
                      metric=settings.get("metric"), progress=progress)


def _metric_source_path(metric):
    if metric["model"] == "square_affine":
        from D2W import die_membrane
        return Path(die_membrane.__file__).resolve()
    return Path(_reference_square_metric().__file__).resolve()


def _check_metric_policy(plan, metric):
    if metric is None:
        return
    if not isinstance(metric, dict) or metric.get("model") not in ("explicit", "square_surrogate", "square_affine", "disk"):
        raise ValueError("metric must declare explicit coefficients, square_surrogate, square_affine or disk")
    if is_die_plan(plan) and metric["model"] == "disk":
        raise ValueError("prepared die columns cannot use the wafer disk metric")
    if metric["model"] == "square_affine" and not is_die_plan(plan):
        raise ValueError("full-square affine metric requires a declared prepared-die column")
    if metric["model"] in ("square_surrogate", "square_affine") and set(metric) != {"model"}:
        raise ValueError("reference square metrics have no fitted coefficients or additional settings")
    if metric["model"] == "disk" and set(metric) != {"model"}:
        raise ValueError("disk has no fitted coefficients or additional settings")
    if metric["model"] == "explicit":
        for route in plan["routes"]:
            for event in route["events"]:
                for role in ("lower", "upper"):
                    _metric_coefficients(metric, event[role + "_free"]["state_id"], np.eye(6))
