"""Owned-input native graded25 C1 coordinator; no I/O, backend load or launcher.

Original sparse mechanics, Newton policy and commit definitions are retained.
The caller authenticates material callbacks/cards, clock and native operands,
and admits resources before run(). Named output bounds are NOT peak RAM bounds.
This is not dense-time, multi-cycle, MAPDL or operator-delivery qualification.
"""
from __future__ import annotations

import ast
from collections import namedtuple
import hashlib
import json
import math
from types import MappingProxyType, ModuleType, SimpleNamespace as NS

from tools.operator_construction import training_frame
from tools.operator_reference import _definitions, require


SOURCE_PINS = MappingProxyType({
    "core": "089411833177e40bc7facae29b13bf1309ea9d25c31bcaa614aa5be56058fa9b",
    "newton": "757af152a3ee22a8b202bb2a9945db165196a449f61bbf1a3a51c536ebe33140",
    "native_state": "858059695dc22c208effc3db32d74724cedd435447c482238c6be49d0e27ffe5",
    "clock": "00caf8305da32026e41bf677f90fd6014df9b508b69746df0ee7f4b254ced6d6",
    "builder": "c05decbf146226f470b25b3bb8b3d336023ab0ee0c5dcf3cd681ba28efa9419c",
    "policy": "db137bbafd69baf329bf1b1f8c762f781d12d7d5fbd21faf24c4d67263afac48",
})
FIELDS = ("plastic_strain", "creep_strain", "equivalent_plastic_strain", "equivalent_creep_strain")
COUNT_FIELDS = ("strict_acceptances", "strict_failures", "residual_fallback_acceptances",
                "residual_fallback_failures", "cutbacks")
Clock = namedtuple("Clock", "result_set time_s temperature_c creep_active plastic_active")
_TOKEN = object()


def bind_sources(raw):
    require(type(raw) is dict and set(raw) == set(SOURCE_PINS), "exact six-source native history closure required")
    for role, pin in SOURCE_PINS.items():
        value = raw[role]
        require(type(value) is bytes and 0 < len(value) <= 256 * 1024
                and hashlib.sha256(value).hexdigest() == pin, "frozen history source differs: " + role)
    modules = {}
    for role in ("core", "newton"):
        module = ModuleType("_operator_native_history_" + role)
        exec(compile(raw[role], "<authenticated-native-" + role + ">", "exec", dont_inherit=True), vars(module))
        modules[role] = module
    state = _definitions(raw["native_state"], "native_state",
        functions={"state_arrays", "state_sha", "frozen", "committed"}, assignments={"FIELDS"},
        namespace=dict(hashlib=hashlib, require=require))
    clock = _definitions(raw["clock"], "native_clock",
        functions={"selected_ordinals", "select_schedule"}, assignments={"INDICES"}, namespace=dict(require=require))
    builder = _definitions(raw["builder"], "native_layer", functions={"make_layer"})
    policies = [node.value for node in ast.parse(raw["policy"]).body if isinstance(node, ast.Assign)
        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "THERMAL_POLICY"]
    require(len(policies) == 1 and state.FIELDS == FIELDS, "original policy and complete Cu14 field definitions required")
    policy = ast.literal_eval(policies[0])
    require(type(policy) is dict and policy["linear_tolerance"] == 1e-9
            and policy["linear_solver"] == "matrix_free", "original complete matrix-free policy required")
    return NS(token=_TOKEN, core=modules["core"], newton=modules["newton"], state=state,
        clock=clock, builder=builder, policy=MappingProxyType(policy), source_pins=SOURCE_PINS)


def _policy(bound, value):
    for name, expected in bound.policy.items():
        actual = getattr(value, name)
        require(type(actual) is type(expected) and actual == expected, "original complete thermal policy differs: " + name)
    value.validate()


def _clock(bound, dense):
    indices, steps = bound.clock.select_schedule(dense)
    clocks = tuple(Clock(int(step.result_set), float(step.time_s), float(step.temperature_c),
                         bool(step.creep_active), bool(step.plastic_active)) for step in steps)
    require(len(clocks) == 25 and tuple(indices[index] for index in (8, 16, 24)) == (160, 200, 360),
            "original graded25 C1 observation/training clock required")
    return tuple(indices), clocks


def _state(api, layer, state, coordinates, check):
    np = api.np
    check()
    layer.validate_committed(state)
    require(type(coordinates) is np.ndarray and coordinates.shape == (6 + layer.n,)
            and coordinates.dtype == np.dtype("float64") and not coordinates.flags.writeable
            and coordinates.flags.c_contiguous and bool(np.isfinite(coordinates).all()), "immutable finite full native coordinates required")
    require(np.array_equal(state.micro, coordinates[6:]) and bool(np.all(state.birth_micro == 0.))
            and bool(np.all(state.birth_macro == 0.)), "committed micro or virgin birth fields differ")
    for name in FIELDS:
        value = getattr(state.cu, name)
        require(type(value) is np.ndarray and value.dtype == np.dtype("float64") and value.flags.c_contiguous,
                "native Cu14 dtype/order required: " + name)
    check()


def _diagnostics(value, trial):
    require(type(value) is dict and len(value) <= 6 and set(value) in (set(COUNT_FIELDS), set(COUNT_FIELDS) | {"strict_failure"}),
            "bounded original Newton diagnostic fields required")
    result = {}
    for name in COUNT_FIELDS:
        number = value[name]
        require(type(number) is int and 0 <= number <= 1, "bounded original acceptance counts required")
        result[name] = number
    require(result["cutbacks"] == 0 and result["residual_fallback_failures"] == 0
            and result["strict_acceptances"] + result["residual_fallback_acceptances"] == 1,
            "accepted original trial without hidden cutback required")
    if "strict_failure" in value:
        text = value["strict_failure"]
        require(type(text) is str and len(text) <= 2000, "bounded strict-failure diagnostic required")
        result["strict_failure"] = text
    require(type(trial.iterations) is int and 0 < trial.iterations <= 96, "bounded original nonlinear iteration count required")
    for name in ("scaled_residual_norm", "tangent_correction_norm"):
        number = getattr(trial, name)
        require(type(number) in (int, float) and math.isfinite(number) and number >= 0., "finite original trial diagnostic required")
    return result


class NativeC1:
    """One fresh native attempt; no imported state, resume or automatic retry.

    physical_factor_context(matrix) must be a context manager yielding a solve
    handle for physical Kff. It is NOT the scaled-CSR response factory ABI.
    The observer receives detached scalar metadata, never live Cu/full vectors.
    """

    def __init__(self, api, bound, *, assembly, reference, card, kernels, policy, dense1081,
                 physical_factor_context, limits, check, case, native_sha256, observer=None):
        require(getattr(bound, "token", None) is _TOKEN, "authenticated native history binding required")
        require(callable(check) and callable(physical_factor_context) and (observer is None or callable(observer)),
                "owner guard, physical factor context and optional observer required")
        require(type(case) is str and 0 < len(case) <= 256 and case.isprintable(), "bounded explicit native case required")
        require(type(limits) is dict and set(limits) == {"retained_array_bytes", "diagnostic_bytes_per_step"}
                and type(limits["retained_array_bytes"]) is int and 0 < limits["retained_array_bytes"] <= 64 * 1024**3
                and type(limits["diagnostic_bytes_per_step"]) is int and 0 < limits["diagnostic_bytes_per_step"] <= 128 * 1024,
                "explicit bounded retained-output limits required, not a peak-RAM estimate")
        self.frame = training_frame(case=case, native_sha256=native_sha256)
        self._api, self._bound, self._check, self._observer = api, bound, check, observer
        self._limits = dict(limits)
        self._inputs = NS(assembly=assembly, reference=reference, card=card, kernels=kernels,
                          policy=policy, dense=dense1081, factory=physical_factor_context)
        self.status, self.last_accepted_ordinal, self.secondary_failure = "NEW", None, None
        check()

    def run(self):
        require(self.status == "NEW", "terminal or running native attempt; no automatic retry")
        self.status = "RUNNING"
        try:
            result = self._run()
            self.status = "CLOSED_NATIVE_C1_UNVALIDATED"
            return result
        except BaseException:
            self.status = "FAILED"
            raise
        finally:
            self._inputs = None

    def _run(self):
        api, bound, data, check = self._api, self._bound, self._inputs, self._check
        np = api.np
        check()
        _policy(bound, data.policy)
        indices, clocks = _clock(bound, data.dense)
        assembly = data.assembly
        n, g = assembly["kff"].shape[0], len(assembly["weights"])
        require(n > 0 and g > 0 and assembly["kff"].shape == (n, n)
                and assembly["B_macro"].shape == (6 * g, 6) and assembly["B_free"].shape == (6 * g, n),
                "full native body/GP localization ledger required")
        # Outputs: Cu14, committed micro/births, final x, two CP copies, q6[25].
        required = 8 * (14 * g + 5 * n + 174)
        require(required <= self._limits["retained_array_bytes"], "named native output arrays exceed explicit byte limit")
        require(all(data.reference.properties("cu", t) == (*data.card.elastic(t), data.card.alpha(t)) for t in (25., 350.)),
                "original native reference and Cu card frame differ")
        kernels = bound.core.MaterialKernels(*(getattr(data.kernels, name) for name in
            ("update_batch", "prepare_tangent", "apply_tangent")))
        require(all(callable(getattr(kernels, name)) for name in ("update_batch", "prepare_tangent", "apply_tangent")),
                "explicit owner material kernels required")
        check()
        layer = bound.builder.make_layer(api, bound.core, assembly, data.reference, data.card, kernels)
        model = bound.core.SparseModel(api, [layer], np.eye(6))
        require(layer.n == n and layer.g == g and model.reference.size == 6 + n, "constructed native dimensions differ")
        x = bound.state.frozen(np, np.zeros(6 + n))
        state = bound.state.committed(np, bound.core, x, api.cu.CuBatchState.zeros(g))
        _state(api, layer, state, x, check)
        require(all(bool(np.all(getattr(state.cu, name) == 0.)) for name in FIELDS), "fresh virgin zero Cu14 required")
        prefix, snapshots, rows = [x[:6].copy()], {}, []
        totals = {name: 0 for name in COUNT_FIELDS}
        self.last_accepted_ordinal = 0

        def integrity():
            _policy(bound, data.policy)
            require(_clock(bound, data.dense) == (indices, clocks), "selected clock changed during native attempt")
            _state(api, layer, state, x, check)

        def observe(ordinal, clock, diagnostics=None, trial=None):
            row = dict(ordinal=ordinal, original_ordinal=indices[ordinal], time_s=clock.time_s,
                temperature_c=clock.temperature_c, creep_active=clock.creep_active,
                q6=[float(value) for value in x[:6]])
            if diagnostics is not None:
                row.update(diagnostics=diagnostics, iterations=trial.iterations,
                           scaled_residual_norm=trial.scaled_residual_norm,
                           tangent_correction_norm=trial.tangent_correction_norm)
            raw = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            require(len(raw.encode("utf-8")) <= self._limits["diagnostic_bytes_per_step"], "bounded native observer record required")
            rows.append(json.loads(raw))
            if self._observer is not None:
                self._observer(json.loads(raw))
            check()

        completed_inside_context = False
        with model.reference.reference_solver(data.factory) as inverse:
            observe(0, clocks[0])
            for ordinal in range(1, len(clocks)):
                check()
                integrity()
                current, previous = clocks[ordinal], clocks[ordinal - 1]
                before = bound.state.state_sha(state, x)
                step = bound.core.Step(current.temperature_c, current.time_s - previous.time_s,
                                       current.time_s, current.creep_active, current.plastic_active)
                try:
                    trial, diagnostics = bound.newton.trial_with_original_policy(model, inverse, x, step,
                        {layer.name: state}, data.policy, checkpoint=check, progress_callback=lambda event: check())
                except BaseException as primary:
                    try:
                        integrity()
                        require(bound.state.state_sha(state, x) == before, "incoming native state changed on failed trial")
                    except BaseException as secondary:
                        reason = secondary.args[0] if secondary.args and type(secondary.args[0]) is str else type(secondary).__name__
                        self.secondary_failure = dict(phase="incoming verification after trial failure",
                            error_type=type(secondary).__name__, reason=reason[:500])
                        if hasattr(primary, "add_note"):
                            primary.add_note("Secondary incoming verification failed: " + reason[:500])
                    raise
                integrity()
                require(bound.state.state_sha(state, x) == before, "incoming native history changed during trial")
                diagnostics = _diagnostics(diagnostics, trial)
                evaluation = trial.evaluation
                require(evaluation.model is model and len(evaluation.layer_evaluations) == 1
                        and np.array_equal(trial.coordinates, evaluation.coordinates), "accepted trial/evaluation coordinates differ")
                local = evaluation.layer_evaluations[0]
                require(local.layer is layer and local.committed is state and local.step == step,
                        "accepted native trial frame differs")
                accepted_cu = local.cu_trial.state.copy()
                prior_fields = [getattr(state.cu, name) for name in FIELDS]
                prior_fields.extend(getattr(local.cu_trial.state, name) for name in FIELDS)
                copied_fields = []
                for name in FIELDS:
                    value = getattr(accepted_cu, name)
                    shape = (g, 6) if name in FIELDS[:2] else (g,)
                    require(type(value) is np.ndarray and value.shape == shape and value.dtype == np.dtype("float64")
                            and value.flags.c_contiguous, "complete detached native Cu14 copy required")
                    require(all(not np.shares_memory(value, other) for other in prior_fields + copied_fields),
                            "accepted Cu copy aliases trial/incoming history")
                    copied_fields.append(value)
                if not step.creep_active:
                    require(all(np.array_equal(getattr(accepted_cu, name), getattr(state.cu, name))
                        for name in ("creep_strain", "equivalent_creep_strain")), "creep-OFF retained history changed")
                candidate_x = bound.state.frozen(np, trial.coordinates)
                candidate_state = bound.state.committed(np, bound.core, candidate_x, accepted_cu)
                _state(api, layer, candidate_state, candidate_x, check)
                require(bound.state.state_sha(state, x) == before, "commit changed incoming native history")
                x, state = candidate_x, candidate_state
                prefix.append(x[:6].copy())
                self.last_accepted_ordinal = ordinal
                if ordinal in (8, 16):
                    snapshots[ordinal] = bound.state.frozen(np, x)
                for name in COUNT_FIELDS:
                    totals[name] += diagnostics[name]
                observe(ordinal, current, diagnostics, trial)
                del trial, evaluation, local, accepted_cu, candidate_x, candidate_state, prior_fields, copied_fields
            completed_inside_context = True
        require(completed_inside_context, "physical factor context suppressed a native failure")
        integrity()
        q6 = bound.state.frozen(np, np.asarray(prefix))
        retained = [x, q6, state.micro, state.birth_micro, state.birth_macro, *snapshots.values()]
        retained.extend(getattr(state.cu, name) for name in FIELDS)
        actual_bytes = sum(value.nbytes for value in retained)
        require(set(snapshots) == {8, 16} and actual_bytes == required
                and self.last_accepted_ordinal == 24, "complete named native output ledger required")
        check()
        return NS(training_coordinates=snapshots, frame=self.frame, q6_history=q6,
            final_coordinates=x, final_native_state=state, observations=tuple(rows),
            report=dict(state="CLOSED_NATIVE_C1_UNVALIDATED", source_pins=dict(SOURCE_PINS),
                source_bytes_authenticated=True, native_case=dict(self.frame), native_body_coordinates=n, Cu_GP=g,
                selected_original_ordinals=list(indices), states=25, accepted_increments=24, cycles_executed=1,
                training_original_ordinals={8: 160, 16: 200}, cooldown_used_for_training=False,
                coordinate_order="native_physical_q6_then_body_z", virgin_birth_and_Cu_zero=True,
                imported_state_reads=0, native_Cu_state_adopted_by_ROM=False,
                retained_output_array_bytes=actual_bytes, retained_output_limit_bytes=self._limits["retained_array_bytes"],
                retained_output_budget_is_peak_RAM=False, original_model_copies_native_arrays=True,
                physical_factor_ABI="context-managed inverse of physical Kff, not scaled CSR",
                policy=dict(bound.policy), acceptance_counts=totals,
                residual_fallback_acceptances_are_strict_JVP_certificates=False,
                observer_receives_live_Cu_or_full_coordinates=False, numerical_input_custody_verified_here=False,
                material_callbacks_verified_here=False, card_file_provenance_verified_here=False,
                clock_card_provenance_verified_here=False, host_resource_admission=False,
                dense_time_convergence_verified=False, multi_cycle_accuracy_verified=False,
                MAPDL_accuracy_claim=False, bow_accuracy_validated=False, operator_admission=False))
