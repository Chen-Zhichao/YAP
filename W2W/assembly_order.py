"""F2B-only W2W planning and explicitly historical archive utilities.

Historical mixed-face v2 ledgers remain plan-only and are never executed.
The private v1 planner remains unchanged for archive fixtures.
A plan alone is not a mechanics or yield prediction.
Run from the YAP root with ``python -m W2W.assembly_order --help``.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from YAP.manufacturing_bow import (
        ManufacturingLayer, clpt_source_response,
        strain_gradient_to_signed_edge_bow_um,
    )
    from YAP.history_bow_yield import (
        PrebondBody, paired_magnification, affine_overlay_um,
        worst_critical_pad_survival, compound_site_yield,
    )
    from YAP.mechanics_runtime import validate_resource_reference, configure_runtime_storage, validate_write_path
except ModuleNotFoundError:
    from manufacturing_bow import (
        ManufacturingLayer, clpt_source_response,
        strain_gradient_to_signed_edge_bow_um,
    )
    from history_bow_yield import (
        PrebondBody, paired_magnification, affine_overlay_um,
        worst_critical_pad_survival, compound_site_yield,
    )
    from mechanics_runtime import validate_resource_reference, configure_runtime_storage, validate_write_path


INPUT_SCHEMA = "yap.w2w_assembly_order_input.v1"
STACK_SCHEMA = "yap.w2w_stack_input.v1"
PLAN_SCHEMA = "yap.w2w_assembly_order_plan.v1"
F2B_PLAN_SCHEMA = "yap.w2w_f2b_assembly_order_plan.v3"
MIXED_PLAN_SCHEMA = "yap.w2w_assembly_order_plan.v2"
MIXED_ROUTE_SCHEMA = "yap.w2w_symbolic_process_ledger.v2"
PROCESS_PLACEHOLDER_SCHEMA = "yap.w2w_process_definition_request.v2"
ARCHIVE_SCHEMA = "yap.w2w_assembly_order_mechanics.v1"
DIE_PLAN_SCHEMA = "yap.d2w_assembly_order_plan.v1"
DIE_ARCHIVE_SCHEMA = "yap.d2w_assembly_order_mechanics.v1"
DIE_FRAME = "fixed_bottom_to_top_die_centered_front_below_back"
FRAME = "fixed_bottom_to_top_z_front_below_back"
MIXED_FRAME = "pending_route_derived_mixed_face_global_frame"
CONVENTION = "center_minus_mid_edge"
MAX_WAFERS = 32
_PROCESS_DEFAULTS = {
    "recipe_id": "REFERENCE_350C_650s_3600s_650s_tree_extension_unvalidated",
    "prebond_policy": (
        "reference 300 s initial clock advance once per material birth, creep off; "
        "subsequent RT flatten/capture/release elastic; pressure not explicitly solved"
    ),
    "waiting_policy": "no_evolution_between_events",
    "anneal_support": "free",
    "bonding_support": "ideal_flattening_free_membrane",
    "material_clock_policy": "retain_each_material_history_no_reset",
}
_TEMPORAL_PROFILES = frozenset({"exact", "distilled14", "distilled26", "distilled51",
                               "distilled61", "distilled76", "distilled101", "component_graded25"})


def _check_recipe_request(plan, temporal_profile):
    """Check the supported request without reading native schedule/card data."""
    if not isinstance(temporal_profile, str) or temporal_profile not in _TEMPORAL_PROFILES:
        raise ValueError(f"unsupported temporal_profile: {temporal_profile!r}")
    process = plan["context"]["process"]
    expected = dict(room_temperature_c=25., anneal_temperature_c=350.,
                    ramp_up_s=650., dwell_s=3600., cool_down_s=650.)
    if any(not np.isclose(process[key], value, rtol=0., atol=1e-12)
           for key, value in expected.items()):
        raise ValueError("assembly execution requires the existing 25C/350C V2 thermal recipe")
    if process["prebond_policy"] != _PROCESS_DEFAULTS["prebond_policy"]:
        raise ValueError("assembly execution requires the declared legacy frozen-history prebond policy")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_id(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def is_die_plan(plan):
    return plan.get("schema") == DIE_PLAN_SCHEMA


def is_f2b_plan(plan):
    return plan.get("schema") == F2B_PLAN_SCHEMA


def is_prepared_plan(plan):
    return is_die_plan(plan)


def component_key(plan):
    return "die_ids" if is_die_plan(plan) else "wafer_ids"


def footprint_mm(plan):
    geometry = plan["context"]["geometry"]
    return ((geometry["width_mm"], geometry["length_mm"]) if is_die_plan(plan)
            else (geometry["diameter_mm"],) * 2)


def _number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"{name} is outside its permitted range")
    return value


def _keys(value, required, name, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) \
            or set(required) - set(value):
        raise ValueError(f"{name} requires {sorted(required)}; optional {sorted(optional)}")


def _label(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _array(value, name, shape=None):
    raw = np.asarray(value, dtype=object)
    if any(isinstance(item, (bool, np.bool_, str)) for item in raw.flat):
        raise ValueError(f"{name} must contain numbers, not strings or Booleans")
    values = np.asarray(value)
    if values.dtype.kind not in "ifu" or not np.isfinite(values).all() \
            or (shape is not None and values.shape != shape):
        raise ValueError(f"invalid {name}; expected finite numbers of shape {shape}")
    return values.astype(float)


def leaves(tree):
    return (tree,) if type(tree) is int else leaves(tree[0]) + leaves(tree[1])


def tree_text(tree):
    return str(tree) if type(tree) is int else f"({tree_text(tree[0])},{tree_text(tree[1])})"


def parse_order(value, wafer_count):
    """Parse a literal binary tree, never executable Python or wafer permutations."""
    if type(wafer_count) is not int or not 2 <= wafer_count <= MAX_WAFERS:
        raise ValueError(f"wafer_count must be in 2..{MAX_WAFERS}")
    if isinstance(value, str):
        if len(value) > 4096:
            raise ValueError("assembly order is too long")
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError, RecursionError) as exc:
            raise ValueError("assembly order must be a binary tuple such as ((1,2),(3,4))") from exc

    def convert(node, depth=0):
        if depth > MAX_WAFERS:
            raise ValueError("assembly order is too deep")
        if type(node) is int:
            return node
        if not isinstance(node, (list, tuple)) or len(node) != 2:
            raise ValueError("each internal assembly node must have exactly two children")
        return (convert(node[0], depth + 1), convert(node[1], depth + 1))

    tree = convert(value)
    if leaves(tree) != tuple(range(1, wafer_count + 1)):
        raise ValueError("leaves must contain every wafer exactly once in fixed bottom-to-top order")
    return tree


def all_orders(wafer_count):
    """Enumerate ordered binary trees; cap exhaustive enumeration, not selected input."""
    if type(wafer_count) is not int or not 2 <= wafer_count <= 8:
        raise ValueError("all-orders enumeration supports 2..8 wafers; supply selected trees above 8")

    @lru_cache(None)
    def split(first, last):
        if first == last:
            return (first,)
        return tuple((left, right) for k in range(first, last)
                     for left in split(first, k) for right in split(k + 1, last))

    return split(1, wafer_count)


def _context(config):
    _keys(config, {"schema", "initial_bows_um", "geometry", "fabrication_layers",
                   "process", "assembly_orders"}, "input", {"description"})
    if config["schema"] != INPUT_SCHEMA:
        raise ValueError("unsupported assembly-order input schema")
    bows = config["initial_bows_um"]
    if not isinstance(bows, list) or not 2 <= len(bows) <= MAX_WAFERS:
        raise ValueError("supply 2..32 fixed initial wafer bows")
    bows = [_number(bow, "initial bow") for bow in bows]
    required = {"diameter_mm", "initial_total_thickness_um", "target_total_thickness_um",
                "front_bi_thickness_um", "back_bi_thickness_um", "beol_thickness_um"}
    _keys(config["geometry"], required, "geometry")
    geometry = {key: _number(value, key, positive=True)
                for key, value in config["geometry"].items()}
    initial = geometry["initial_total_thickness_um"]
    target = geometry["target_total_thickness_um"]
    front = geometry["front_bi_thickness_um"] + geometry["beol_thickness_um"]
    if not front + geometry["back_bi_thickness_um"] < target < initial:
        raise ValueError("W2W requires thinning: target total must include front BI, BEOL, positive Si, and back BI, and be below initial")

    raw_layers = config["fabrication_layers"]
    layer_count = 3
    if not isinstance(raw_layers, list) or len(raw_layers) != layer_count:
        raise ValueError("fabrication_layers must be [front BI, BEOL, Si], in bottom-to-top order")
    layers = []
    for index, raw in enumerate(raw_layers):
        _keys(raw, {"name", "thickness_um", "e_mpa", "nu", "eigenstrain_per_amplitude"},
              "fabrication layer")
        _label(raw["name"], "layer name")
        numbers = {key: _number(value, key) for key, value in raw.items() if key != "name"}
        layers.append(ManufacturingLayer(name=raw["name"], **numbers,
                                        is_bulk_silicon=index == 2))
    if len({layer.name for layer in layers}) != layer_count:
        raise ValueError("fabrication layer names must be distinct")
    source_free = (0,)
    if any(layers[index].eigenstrain_per_amplitude != 0 for index in source_free):
        raise ValueError("the existing fabrication model confines its source to BEOL, not BI")
    expected = [geometry["front_bi_thickness_um"], geometry["beol_thickness_um"], initial - front]
    if not np.allclose([layer.thickness_um for layer in layers], expected, rtol=0, atol=1e-9):
        raise ValueError("fabrication layer thicknesses do not match the total-thickness ledger")

    required_process = {"recipe_id", "room_temperature_c", "anneal_temperature_c",
                        "ramp_up_s", "dwell_s", "cool_down_s", "prebond_policy",
                        "waiting_policy", "anneal_support", "bonding_support",
                        "material_clock_policy"}
    _keys(config["process"], required_process, "process")
    process = dict(config["process"])
    for key in ("recipe_id", "prebond_policy"):
        _label(process[key], key)
    for key in ("room_temperature_c", "anneal_temperature_c"):
        process[key] = _number(process[key], key)
    if process["anneal_temperature_c"] <= process["room_temperature_c"]:
        raise ValueError("anneal temperature must exceed room temperature")
    for key in ("ramp_up_s", "dwell_s", "cool_down_s"):
        process[key] = _number(process[key], key, positive=True)
    policies = dict(waiting_policy="no_evolution_between_events", anneal_support="free",
                    bonding_support="ideal_flattening_free_membrane",
                    material_clock_policy="retain_each_material_history_no_reset")
    for key, value in policies.items():
        if process[key] != value:
            raise ValueError(f"this planner requires {key}={value}")
    context = dict(initial_bows_um=bows, geometry=geometry, process=process,
                fabrication_layers=[asdict(layer) for layer in layers],
                frame_id=FRAME, bow_convention=CONVENTION,
                orientation="lower_BACK_to_upper_FRONT", thickness_scope="total_including_BEOL_and_BI")
    return context


def _births(context):
    response = clpt_source_response([ManufacturingLayer(**item) for item in context["fabrication_layers"]])
    diameter = context["geometry"]["diameter_mm"]
    unit_bow = strain_gradient_to_signed_edge_bow_um(response["kappa_per_amplitude_per_mm"], diameter)
    if not np.isfinite(unit_bow) or unit_bow == 0:
        raise ValueError("fabrication source has no finite nonzero bow response")
    result = []
    for wafer, bow in enumerate(context["initial_bows_um"], 1):
        amplitude = bow / unit_bow
        q6 = amplitude * np.asarray(response["natural_birth_q6_per_amplitude"])
        if not np.isfinite(amplitude) or not np.isfinite(q6).all():
            raise ValueError("fabrication source inversion overflow")
        result.append(dict(wafer=wafer, initial_bow_um=bow, source_amplitude=amplitude,
                           initial_q6=q6.tolist(), reconstruction_bow_um=amplitude * unit_bow))
    return dict(model="BEOL_eigenstrain_linear_ABD_inverse", source_response=response,
                wafers=result, redraw_per_route=False, recalibrate_after_thinning=False,
                status="CALIBRATION_ONLY_NOT_RESIDUAL_STRESS_HISTORY_INITIALIZATION")


def _build_legacy_plan(config):
    """Private v1 request builder for legacy archive fixtures, not public W2W."""
    context = _context(config)
    count = len(context["initial_bows_um"])
    selection = config["assembly_orders"]
    if selection == "all":
        orders = all_orders(count)
    elif isinstance(selection, str):
        orders = (parse_order(selection, count),)
    elif isinstance(selection, list) and selection:
        orders = tuple(parse_order(value, count) for value in selection)
    else:
        raise ValueError("assembly_orders must be 'all', one expression, or a nonempty list of trees")
    if len(set(orders)) != len(orders):
        raise ValueError("duplicate assembly orders")
    births = _births(context)
    context_id = content_id(dict(context=context, initialization=births))
    geometry = context["geometry"]
    initial, target = (geometry[key] for key in ("initial_total_thickness_um", "target_total_thickness_um"))

    def state_id(tree, phase):
        return content_id(dict(context_id=context_id, tree=tree_text(tree), phase=phase))

    routes = []
    for tree in orders:
        events = []

        def visit(node):
            if type(node) is int:
                return dict(wafer_ids=[node], total_thicknesses_um=[initial],
                            thermal_cycle_counts=[0], state_id=state_id(node, "initial_free"))
            lower = visit(node[0])
            upper = visit(node[1])
            k = lower["wafer_ids"][-1]
            if lower["total_thicknesses_um"][-1] != initial:
                raise ValueError("lower child's exposed top wafer was unexpectedly thinned already")
            prepared = dict(
                lower, total_thicknesses_um=lower["total_thicknesses_um"][:-1] + [target],
                state_id=state_id(node[0], "back_prepared_free"))
            result = dict(wafer_ids=lower["wafer_ids"] + upper["wafer_ids"],
                          total_thicknesses_um=prepared["total_thicknesses_um"] + upper["total_thicknesses_um"],
                          thermal_cycle_counts=[v + 1 for v in lower["thermal_cycle_counts"] + upper["thermal_cycle_counts"]],
                          state_id=state_id(node, "post_anneal_cooldown_free"))
            event = dict(event=f"A{len(events) + 1}", subtree=tree_text(node),
                         interface_id=f"{k}.BACK--{k + 1}.FRONT",
                         lower_before_preparation=lower, lower_free=prepared, upper_free=upper,
                         operations=["observe_both_released_prebond_states",
                                     "independent_ideal_flattening",
                                     "capture_both_child_references_and_Cu_histories",
                                     "bond_lower_BACK_to_upper_FRONT", "release_flattening_constraints",
                                     "free_anneal", "free_cooldown"],
                         result=result)
            event["thinning"] = dict(wafer=k, from_total_um=initial, to_total_um=target,
                retained_si_um=target - geometry["front_bi_thickness_um"]
                - geometry["beol_thickness_um"] - geometry["back_bi_thickness_um"],
                new_back_bi_um=geometry["back_bi_thickness_um"], source_amplitude_unchanged=True)
            event["operations"].insert(0, "carrier_assisted_back_thinning_and_BI_preparation")
            events.append(event)
            return result

        final = visit(tree)
        expected = [target] * (count - 1) + [initial]
        if final["total_thicknesses_um"] != expected or len(events) != count - 1:
            raise ValueError("inconsistent final assembly ledger")
        route = dict(order=tree_text(tree), context_id=context_id, events=events, final=final)
        route["request_id"] = content_id(route)
        routes.append(route)
    return dict(schema=PLAN_SCHEMA, status="PLANNED_MECHANICS_REQUIRED", physical_readiness=False,
                context_id=context_id, context=context, initialization=births, routes=routes,
                required_mechanics=["exact compatible initial/retained/substrate and BI operators",
                                    "declared legacy natural-q6 initialization and retained birth coordinates",
                                    "two-child Cu, reference-offset and material-clock transfer audits",
                                    "released prebond states and post-cycle checkpoints for each event"],
                live_mechanics_backend="W2W.assembly_mechanics (opt-in)")


_MIXED_BLOCKERS = (
    "FINAL_ORIENTATION_UNCONFIRMED",
    "INNER_AND_OUTER_OPERATOR_ADMISSION_PENDING",
    "COMPATIBLE_INITIAL_775_SI_CAP_ADMISSION_PENDING",
    "PREFIX_AND_SUFFIX_HISTORY_TRANSFER_AUDITS_PENDING",
    "NATIVE_ROUTE_FRAME_AND_INITIAL_DATUM_GENERATION_PENDING",
    "TEMPORARY_CARRIER_LIFECYCLE_UNSPECIFIED",
)
_CONFIRMED_ORDERS = {
    "(((1,2),3),4)": ("sequential", [-1, 1, 1, 1]),
    "((1,2),(3,4))": ("balanced", [-1, 1, -1, 1]),
}


def _mixed_orders(config, count):
    selection = config["assembly_orders"]
    if selection == "all":
        orders = all_orders(count)
    elif isinstance(selection, str):
        orders = (parse_order(selection, count),)
    elif isinstance(selection, list) and selection:
        orders = tuple(parse_order(value, count) for value in selection)
    else:
        raise ValueError("assembly_orders must be 'all', one expression, or a nonempty list of trees")
    if len(set(orders)) != len(orders):
        raise ValueError("duplicate assembly orders")
    return orders


def _mixed_route(tree, context, context_id):
    order = tree_text(tree)
    if order not in _CONFIRMED_ORDERS:
        route = dict(schema=PROCESS_PLACEHOLDER_SCHEMA, order=order, context_id=context_id,
                     status="NEEDS_PROCESS_DEFINITION", events=[],
                     blockers=["PROCESS_DEFINITION_NOT_CONFIRMED"])
        route["request_id"] = content_id(route)
        return route

    process_name, orientation = _CONFIRMED_ORDERS[order]
    geometry = context["geometry"]
    initial = geometry["initial_total_thickness_um"]
    target = geometry["target_total_thickness_um"]
    back_bi = geometry["back_bi_thickness_um"]
    outer_si = target - geometry["front_bi_thickness_um"] - geometry["beol_thickness_um"]
    inner_si = outer_si - back_bi
    events = []

    def visit(node):
        if type(node) is int:
            return dict(wafer_ids=[node], thermal_cycles=[0], back_bi_cycles={}, after_event=None)
        lower, upper = visit(node[0]), visit(node[1])
        lower_leaf, upper_leaf = type(node[0]) is int, type(node[1]) is int
        if lower_leaf and upper_leaf:
            faces, bond_mode = ("FRONT", "FRONT"), "F2F"
        elif not lower_leaf and upper_leaf:
            faces, bond_mode = ("BACK", "FRONT"), "F2B"
        elif not lower_leaf and not upper_leaf:
            faces, bond_mode = ("BACK", "BACK"), "B2B"
        else:
            raise ValueError("unconfirmed leaf-stack mixed-face process")
        k = lower["wafer_ids"][-1]
        preparations = {"lower": [], "upper": []}
        back_bi_cycles = {**lower["back_bi_cycles"], **upper["back_bi_cycles"]}
        for role, child, face, wafer in (
                ("lower", lower, faces[0], k), ("upper", upper, faces[1], k + 1)):
            if face != "BACK":
                continue
            if child["after_event"] is None or wafer in back_bi_cycles:
                raise ValueError("mating-back preparation requires one prior free anneal/cool")
            preparations[role].append(dict(
                wafer=wafer, face="BACK", after_free_anneal_cool_event=child["after_event"],
                operations=["thin_mating_back", "grow_stressfree_back_BI"],
                from_total_um=initial, to_total_um=target, retained_si_um=inner_si,
                new_back_bi_um=back_bi, source_amplitude_unchanged=True))
            back_bi_cycles[wafer] = 0
        event_id = f"A{len(events) + 1}"
        events.append(dict(
            event=event_id, subtree=tree_text(node), interface_id=f"{k}--{k + 1}",
            physical_interface_index=k - 1, bond_mode=bond_mode,
            lower_child=dict(wafer_ids=lower["wafer_ids"], after_event=lower["after_event"]),
            upper_child=dict(wafer_ids=upper["wafer_ids"], after_event=upper["after_event"]),
            mating_faces=dict(lower=faces[0], upper=faces[1]), preparations=preparations,
            operations=[f"prepare_{role}_mating_back" for role in ("lower", "upper")
                        if preparations[role]] + ["bond", "free_anneal", "free_cooldown"],
            anneal_support="free", cooldown_support="free"))
        return dict(wafer_ids=lower["wafer_ids"] + upper["wafer_ids"],
                    thermal_cycles=[value + 1 for value in lower["thermal_cycles"] + upper["thermal_cycles"]],
                    back_bi_cycles={wafer: value + 1 for wafer, value in back_bi_cycles.items()},
                    after_event=event_id)

    ledger = visit(tree)

    def outer_thin(wafer):
        return dict(operation="thin_outer_back_without_new_BI", wafer=wafer,
                    after_free_anneal_cool_event=ledger["after_event"],
                    from_total_um=initial, to_total_um=target, retained_si_um=outer_si,
                    new_back_bi_um=0., source_amplitude_unchanged=True)

    if process_name == "sequential":
        finalization = dict(ordering_status="CONFIRMED", operations=[
            outer_thin(4), dict(operation="attach_temporary_carrier", wafer=4, face="BACK"),
            outer_thin(1)], carrier_end_state="UNSPECIFIED")
    else:
        finalization = dict(ordering_status="UNSPECIFIED", operations=[],
                            required_operations=[outer_thin(1), outer_thin(4)],
                            temporary_carrier_ordering="UNSPECIFIED", carrier_end_state="UNSPECIFIED")
    route = dict(
        schema=MIXED_ROUTE_SCHEMA, order=order, context_id=context_id, status="PLAN_ONLY",
        process_definition=process_name,
        orientation_proposal=dict(status="UNCONFIRMED", thickness_orientation=list(orientation),
                                  coordinate_transform_applied=False),
        events=events, finalization=finalization,
        final_ledger=dict(status="SYMBOLIC_TARGET_NOT_SOLVED", wafer_ids=ledger["wafer_ids"],
                          total_thicknesses_um=[target] * 4,
                          retained_si_thicknesses_um=[outer_si, inner_si, inner_si, outer_si],
                          new_back_bi_thicknesses_um=[0., back_bi, back_bi, 0.],
                          thermal_cycle_counts=ledger["thermal_cycles"],
                          added_back_bi_thermal_cycle_counts={str(wafer): count
                              for wafer, count in sorted(ledger["back_bi_cycles"].items())}),
        blockers=list(_MIXED_BLOCKERS) + (
            ["BALANCED_OUTER_THINNING_CARRIER_ORDERING_UNSPECIFIED"] if process_name == "balanced" else []))
    route["request_id"] = content_id(route)
    return route


def _build_mixed_plan(config):
    """Build only the two confirmed symbolic processes; expose other trees as requests."""
    context = _context(config)
    count = len(context["initial_bows_um"])
    if count != 4:
        raise ValueError("mixed-face W2W planning currently requires exactly four wafers")
    orders = _mixed_orders(config, count)
    births = _births(context)
    context.pop("orientation")
    context.update(frame_id=MIXED_FRAME, orientation_status="UNCONFIRMED",
                   source_frame="intrinsic_wafer_local_front_below_back",
                   process_kind="mixed_face_free_anneal_cool_before_back_preparation",
                   initial_body_separation="FULL_INITIAL_THICKNESS_GLOBAL_DATUM_NOT_GENERATED")
    context_id = content_id(dict(context=context, initialization=births))
    routes = [_mixed_route(tree, context, context_id) for tree in orders]
    defined = sum(route["schema"] == MIXED_ROUTE_SCHEMA for route in routes)
    missing = len(routes) - defined
    source_input = json.loads(_json(config))
    plan = dict(
        schema=MIXED_PLAN_SCHEMA,
        status="PLAN_ONLY_PARTIAL_PROCESS_DEFINITIONS" if missing else "PLAN_ONLY",
        physical_readiness=False, native_execution_available=False,
        source_input=source_input, source_input_sha256=content_id(source_input),
        context_id=context_id, context=context, initialization=births, routes=routes,
        summary=dict(requested_routes=len(routes), defined_routes=defined,
                     missing_process_definitions=missing),
        blockers=list(_MIXED_BLOCKERS) + (["PROCESS_DEFINITIONS_MISSING"] if missing else []))
    plan["plan_id"] = content_id(plan)
    return plan


def build_plan(config):
    """Plan every selected ordered tree using only lower-BACK/upper-FRONT bonds."""
    source = json.loads(_json(config))
    source.setdefault("geometry", {}).setdefault("target_total_thickness_um", 50.)
    prethin = source.setdefault("prethin_incoming", False)
    if type(prethin) is not bool:
        raise ValueError("prethin_incoming must be Boolean")
    context = _context({key: value for key, value in source.items() if key != "prethin_incoming"})
    geometry = context["geometry"]
    initial = geometry["initial_total_thickness_um"]
    target = geometry["target_total_thickness_um"]
    if initial != 775.:
        raise ValueError("F2B wafers must be born at 775 um total thickness")
    context.update(
        process_kind="F2B_only_lower_exposed_back_preparation",
        computational_datum="independent_body_bottom_(wafer_id-1)*target_total; not_initial_global_assembly",
        carrier_policy="ideal_thinning_support_removed_before_free_observation_no_carrier_thermal_cycle",
        outer_finishing="global_top_back_thin_without_new_BI_retain_original_pure_Si_slice",
        prethin_incoming=prethin,
        cap_material_assumption="legacy_pure_Si_not_new_TSV_Si_accuracy_admission")
    births = _births(context)
    context_id = content_id(dict(context=context, initialization=births))
    count = len(context["initial_bows_um"])
    routes = []

    def state_id(node, phase):
        return content_id(dict(context_id=context_id, tree=tree_text(node), phase=phase))

    for tree in _mixed_orders(source, count):
        events = []

        def visit(node):
            if type(node) is int:
                return dict(wafer_ids=[node], total_thicknesses_um=[initial],
                            thermal_cycle_counts=[0], state_id=state_id(node, "initial_free"))
            lower, upper = visit(node[0]), visit(node[1])
            wafer = lower["wafer_ids"][-1]
            lower_total = lower["total_thicknesses_um"][-1]
            if lower_total != initial and not (prethin and lower_total == target):
                raise ValueError("lower exposed top wafer must retain its original backside cap")
            upper_before = upper
            if prethin and upper["total_thicknesses_um"][-1] == initial:
                upper = dict(upper, total_thicknesses_um=upper["total_thicknesses_um"][:-1] + [target],
                             state_id=state_id(node[1], "incoming_outer_thinned_free"))
            prepared = dict(lower, total_thicknesses_um=lower["total_thicknesses_um"][:-1] + [target],
                            state_id=state_id(node[0], "back_prepared_free"))
            result = dict(wafer_ids=lower["wafer_ids"] + upper["wafer_ids"],
                total_thicknesses_um=prepared["total_thicknesses_um"] + upper["total_thicknesses_um"],
                thermal_cycle_counts=[v + 1 for v in lower["thermal_cycle_counts"] + upper["thermal_cycle_counts"]],
                state_id=state_id(node, "post_anneal_cooldown_free"))
            events.append(dict(event=f"A{len(events) + 1}", subtree=tree_text(node),
                interface_id=f"{wafer}.BACK--{wafer + 1}.FRONT", physical_interface_index=wafer - 1,
                bond_mode="F2B", mating_faces=dict(lower="BACK", upper="FRONT"),
                lower_before_preparation=lower, lower_free=prepared, upper_free=upper,
                thinning=dict(wafer=wafer, from_total_um=lower_total, to_total_um=target,
                    retained_si_um=target - geometry["front_bi_thickness_um"]
                        - geometry["beol_thickness_um"] - geometry["back_bi_thickness_um"],
                    removed_original_pure_si_um=lower_total - target + geometry["back_bi_thickness_um"],
                    new_back_bi_um=geometry["back_bi_thickness_um"], source_amplitude_unchanged=True),
                operations=["carrier_assisted_back_thinning_and_BI_preparation",
                    "observe_both_released_prebond_states", "independent_ideal_flattening",
                    "capture_both_child_references_and_Cu_histories", "bond_lower_BACK_to_upper_FRONT",
                    "release_flattening_constraints", "free_anneal", "free_cooldown"], result=result))
            if upper != upper_before:
                events[-1]["upper_before_preparation"] = upper_before
                events[-1]["operations"].insert(1, "prethin_incoming_back_without_new_BI")
            return result

        assembled = visit(tree)
        final = dict(assembled, total_thicknesses_um=[target] * count,
                     state_id=state_id(tree, "outer_back_finished_free"))
        route = dict(order=tree_text(tree), context_id=context_id, events=events,
            assembled=assembled, final=final, finalization=dict(
                operation="thin_global_top_back_without_new_BI", wafer=count,
                source_state_id=assembled["state_id"], result_state_id=final["state_id"],
                from_total_um=assembled["total_thicknesses_um"][-1], to_total_um=target,
                removed_pure_si_um=assembled["total_thicknesses_um"][-1] - target,
                retained_original_pure_si_um=geometry["back_bi_thickness_um"],
                retained_state_unchanged=True, new_back_bi_um=0., carrier_end_state="removed"))
        route["request_id"] = content_id(route)
        routes.append(route)
    plan = dict(schema=F2B_PLAN_SCHEMA, status="PLANNED_MECHANICS_REQUIRED", physical_readiness=False,
        native_execution_available=True, context_id=context_id, context=context, initialization=births,
        source_input=source, routes=routes,
        physics_qualification="EXPERIMENTAL_F2B_LEGACY_PURE_SI_CAP_ASSEMBLY_ACCURACY_PENDING")
    plan["plan_id"] = content_id(plan)
    return plan


def validate_mixed_plan(plan):
    """Validate the entire symbolic contract, not just caller-recomputed hashes."""
    if not isinstance(plan, dict) or plan.get("schema") != MIXED_PLAN_SCHEMA:
        raise ValueError("expected a public mixed-face W2W v2 plan; legacy archives are not process ledgers")
    source_input = plan.get("source_input")
    if not isinstance(source_input, dict):
        raise ValueError("mixed-face W2W plan requires its source input")
    expected = _build_mixed_plan(source_input)
    if _json(plan) != _json(expected):
        raise ValueError("invalid or modified mixed-face W2W process ledger")


def require_executable_plan(plan):
    """Reject public mixed-face requests before importing or loading any native backend."""
    if isinstance(plan, dict) and plan.get("schema") == MIXED_PLAN_SCHEMA:
        validate_mixed_plan(plan)
        raise ValueError(
            "mixed-face W2W v2 is plan-only: pending process qualification, final orientation, "
            "inner/outer operators, initial Si cap, bottom/suffix history transfer and carrier lifecycle; "
            "native execution, archive scoring, ranking and subset execution are unavailable")
    _check_plan(plan)


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _check_plan(plan):
    if plan.get("schema") == MIXED_PLAN_SCHEMA:
        require_executable_plan(plan)
    if is_f2b_plan(plan):
        if not isinstance(plan.get("source_input"), dict) or _json(plan) != _json(build_plan(plan["source_input"])):
            raise ValueError("invalid or modified F2B process ledger")
    if plan.get("schema") not in (PLAN_SCHEMA, F2B_PLAN_SCHEMA, DIE_PLAN_SCHEMA) or content_id(dict(
            context=plan.get("context"), initialization=plan.get("initialization"))) != plan.get("context_id"):
        raise ValueError("invalid or modified plan context")
    if is_die_plan(plan) and (plan["context"].get("body_kind") != "prepared_die_column"
                             or plan["context"].get("frame_id") != DIE_FRAME):
        raise ValueError("D2W plans require the prepared die-column local frame")
    for route in plan["routes"]:
        if route.get("context_id") != plan["context_id"] or route.get("request_id") != content_id(
                {key: value for key, value in route.items() if key != "request_id"}):
            raise ValueError("invalid or modified route request")


def archive_template(plan):
    """Emit placeholders, not synthetic mechanics. The template cannot be scored."""
    _check_plan(plan)
    return dict(schema=DIE_ARCHIVE_SCHEMA if is_die_plan(plan) else ARCHIVE_SCHEMA,
                context_id=plan["context_id"], status="NEEDS_MECHANICS",
                provenance=dict(solver_id=None, implementation_sha256=None, operator_set_sha256=None,
                                source_transfer_audit=None, material_clock_audit=None,
                                physics_qualification="UNKNOWN"),
                routes=[dict(request_id=route["request_id"], order=route["order"],
                             events=[dict(event=event["event"], interface_id=event["interface_id"],
                                          lower_state_id=event["lower_free"]["state_id"],
                                          upper_state_id=event["upper_free"]["state_id"],
                                          result_state_id=event["result"]["state_id"],
                                          lower=None, upper=None, checkpoints=None,
                                          transfer_audit=None)
                                     for event in route["events"]]) for route in plan["routes"]])


def _body(record, expected, role, diameter_mm, *, full_affine=False,
          ids_key="wafer_ids", frame=FRAME):
    _keys(record, {"state_id", ids_key, "total_thicknesses_um", "frame_id", "bow_convention",
                   "bow_um", "face_lever_um", "chi", "face_lever_source", "face"}, "prebond body",
          {"affine_face_strain"} if full_affine else ())
    if not isinstance(record[ids_key], list) or any(type(i) is not int for i in record[ids_key]):
        raise ValueError("prebond component IDs must be integers")
    for key in ("state_id", ids_key, "total_thicknesses_um"):
        if record[key] != expected[key]:
            raise ValueError(f"prebond {role} {key} does not match requested history")
    if record["frame_id"] != frame or record["bow_convention"] != CONVENTION \
            or record["face"] != ("BACK" if role == "lower" else "FRONT"):
        raise ValueError("prebond face/frame/bow convention mismatch")
    _array(record["bow_um"], "bow_um", (2,))
    if full_affine:
        if record["chi"] is not None:
            raise ValueError("full affine bodies must not also declare scalar chi")
        _array(record.get("affine_face_strain"), "affine_face_strain", (2, 2))
    else:
        _array(record["chi"], "chi", (2,))
    lever = _array(record["face_lever_um"], "face_lever_um")
    if lever.shape not in ((), (2,)):
        raise ValueError("face lever must be scalar or directional pair")
    return PrebondBody(body_id=("die_" if ids_key == "die_ids" else "wafer_") + "_".join(map(str, expected[ids_key])),
                       state_id=record["state_id"], frame_id=frame,
                       bow_um=record["bow_um"], halfspan_um=(500 * diameter_mm,) * 2,
                       face_lever_um=record["face_lever_um"], chi=(0., 0.) if full_affine else record["chi"],
                       face_lever_source=record["face_lever_source"])


def _check_record(record, expected):
    for key, value in dict(event=expected["event"], interface_id=expected["interface_id"],
                           lower_state_id=expected["lower_free"]["state_id"],
                           upper_state_id=expected["upper_free"]["state_id"],
                           result_state_id=expected["result"]["state_id"]).items():
        if record.get(key) != value:
            raise ValueError(f"mechanics event {key} mismatch")
    _label(record.get("transfer_audit"), "transfer audit reference")
    checkpoints = record.get("checkpoints")
    required = {expected[key]["state_id"] for key in
                ("lower_before_preparation", "lower_free", "upper_free", "result")}
    if "upper_before_preparation" in expected:
        required.add(expected["upper_before_preparation"]["state_id"])
    if not isinstance(checkpoints, dict) or set(checkpoints) != required:
        raise ValueError("missing pre-thinning, prebond, or post-cycle checkpoint references")
    for digest in checkpoints.values():
        _digest(digest, "checkpoint digest")


def _overlay_settings(settings, count, radius_um, *, halfspan_um=None):
    _keys(settings, {"coordinates_um", "critical_pad_mask", "site_mask", "threshold_um",
                     "residual_sigma_um", "translation_um", "rotation_rad",
                     "magnification_correction_fraction", "conditional_interface_independence"}, "overlay")
    if settings["conditional_interface_independence"] is not True:
        raise ValueError("declare conditional interface independence for the joint-site product")
    xy = _array(settings["coordinates_um"], "coordinates_um")
    if xy.ndim != 3 or xy.shape[-1] != 2 or min(xy.shape[:2]) == 0:
        raise ValueError("coordinates_um must be nonempty (site,pad,2) in the wafer-centered frame")
    if halfspan_um is not None:
        halfspan = _array(halfspan_um, "die halfspan", (2,))
        if np.any(halfspan <= 0) or np.any(np.abs(xy) > halfspan + 1e-9):
            raise ValueError("pad coordinate lies outside the die-local footprint")
    elif np.any(np.linalg.norm(xy, axis=-1) > radius_um + 1e-9):
        raise ValueError("pad coordinate lies outside the wafer")
    masks = {}
    for key, length in (("critical_pad_mask", xy.shape[1]), ("site_mask", xy.shape[0])):
        mask = np.asarray(settings[key])
        if mask.dtype != np.bool_ or mask.shape != (length,) or not mask.any():
            raise ValueError(f"{key} must be a nonempty Boolean mask of length {length}")
        masks[key] = mask
    threshold = _number(settings["threshold_um"], "threshold_um", nonnegative=True)
    sigma = _number(settings["residual_sigma_um"], "residual_sigma_um", nonnegative=True)
    correction = _number(settings["magnification_correction_fraction"], "correction", nonnegative=True)
    if correction > 1:
        raise ValueError("magnification correction fraction must be in [0,1]")
    shifts = _array(settings["translation_um"], "translation_um")
    if shifts.ndim != 3 or shifts.shape[1:] != (count - 1, 2) or not len(shifts):
        raise ValueError("translation_um must be (joint_draw,physical_interface,2)")
    angles = _array(settings["rotation_rad"], "rotation_rad", shifts.shape[:2])
    return xy, masks, threshold, sigma, correction, shifts, angles


def evaluate_archive(plan, archive, overlay):
    """Score exact-request imported mechanics; validation remains UNKNOWN.

    Placement arrays are indexed by physical interface (1-2, 2-3, ...), not
    traversal event. The same draws and registered sites serve every route.
    Checkpoint digests and audit references are provenance assertions, not
    independent verification of internal Cu tensors or physical accuracy.
    """
    _check_plan(plan)
    die = is_die_plan(plan)
    archive_schema = DIE_ARCHIVE_SCHEMA if die else ARCHIVE_SCHEMA
    if not isinstance(archive, dict) or archive.get("schema") != archive_schema or archive.get("status") != "COMPLETE" \
            or archive.get("context_id") != plan["context_id"]:
        raise ValueError("complete mechanics archive matching the exact plan context is required")
    provenance = archive.get("provenance")
    _keys(provenance, {"solver_id", "implementation_sha256", "operator_set_sha256",
                       "source_transfer_audit", "material_clock_audit", "physics_qualification"}, "provenance")
    for key in ("implementation_sha256", "operator_set_sha256"):
        _digest(provenance[key], key)
    for key in ("solver_id", "source_transfer_audit", "material_clock_audit", "physics_qualification"):
        _label(provenance[key], key)
    supplied = archive.get("routes")
    if not isinstance(supplied, list) or any(not isinstance(item, dict) for item in supplied):
        raise ValueError("archive routes must be a list of records")
    requests = [item.get("request_id") for item in supplied]
    if any(not isinstance(value, str) for value in requests) or len(set(requests)) != len(requests):
        raise ValueError("duplicate or missing route request IDs")
    by_request = dict(zip(requests, supplied))
    if not {route["request_id"] for route in plan["routes"]}.issubset(by_request):
        raise ValueError("mechanics archive is missing a requested route; partial ranking is disabled")
    count = len(plan["context"]["initial_bows_um"])
    diameter = footprint_mm(plan)[0]
    ids_key = component_key(plan)
    frame = DIE_FRAME if die else FRAME
    domain = {"halfspan_um": np.asarray(footprint_mm(plan)) * 500} if die else {}
    xy, masks, threshold, sigma, correction, shifts, angles = _overlay_settings(overlay, count, 500 * diameter, **domain)
    results = []
    native_configuration = archive.get("native_configuration")
    if native_configuration is None:
        native_configuration = {}
    if not isinstance(native_configuration, dict):
        raise ValueError("mechanics archive configuration must be a mapping")
    metric = native_configuration.get("metric")
    if metric is not None and (not isinstance(metric, dict) or metric.get("model") not in (
        "disk", "square_affine", "square_surrogate", "explicit"
    )):
        raise ValueError("unsupported mechanics archive metric; migrate or regenerate the archive")
    metric = {} if metric is None else metric
    if die and metric.get("model") not in ("square_surrogate", "square_affine", "explicit"):
        raise ValueError("D2W scoring requires a declared die metric, not a wafer disk")
    full_affine = metric.get("model") in ("disk", "square_affine")
    state_observations = {}
    checkpoint_observations = {}
    selected_xy = xy[masks["site_mask"]][:, masks["critical_pad_mask"]].reshape(-1, 2)
    non_bow_hashes = [content_id(dict(translation_um=shifts[:, i].tolist(),
        rotation_rad=angles[:, i].tolist(), residual_sigma_um=sigma)) for i in range(count - 1)]
    for route in plan["routes"]:
        supplied_route = by_request[route["request_id"]]
        if supplied_route.get("order") != route["order"] or not isinstance(supplied_route.get("events"), list) \
                or len(supplied_route["events"]) != count - 1:
            raise ValueError("mechanics route order or event count mismatch")
        if is_f2b_plan(plan):
            finishing = supplied_route.get("finalization", {})
            final = supplied_route.get("final", {})
            if any(finishing.get(key) != value for key, value in route["finalization"].items()) \
                    or any(final.get(key) != route["final"][key]
                           for key in ("wafer_ids", "total_thicknesses_um")):
                raise ValueError("F2B archive requires the requested final outer-back finishing")
            for key in ("source_checkpoint_sha256", "result_checkpoint_sha256"):
                _digest(finishing.get(key), "outer-finishing checkpoint")
            source_checkpoint = supplied_route["events"][-1].get("checkpoints", {}).get(route["assembled"]["state_id"])
            if finishing["result_checkpoint_sha256"] != final.get("checkpoint_sha256") \
                    or finishing["source_checkpoint_sha256"] != source_checkpoint:
                raise ValueError("F2B finishing checkpoints do not match the assembled and final states")
        magnifications, physical_indices = [], []
        for record, expected in zip(supplied_route["events"], route["events"]):
            if not isinstance(record, dict):
                raise ValueError("mechanics event must be a record")
            _check_record(record, expected)
            for state_id, digest in record["checkpoints"].items():
                if state_id in checkpoint_observations and checkpoint_observations[state_id] != digest:
                    raise ValueError("shared history has inconsistent checkpoint digests across routes")
                checkpoint_observations[state_id] = digest
            bodies = {}
            for role in ("lower", "upper"):
                body = record.get(role)
                bodies[role] = _body(body, expected[f"{role}_free"], role, diameter,
                                    full_affine=full_affine, ids_key=ids_key, frame=frame)
                key = (body["state_id"], body["face"])
                normalized = asdict(bodies[role])
                normalized["face_lever_um"] = np.broadcast_to(
                    np.asarray(bodies[role].face_lever_um, dtype=float), (2,)).tolist()
                if full_affine:
                    normalized["affine_face_strain"] = body["affine_face_strain"]
                if key in state_observations and state_observations[key] != normalized:
                    raise ValueError("shared free state has inconsistent observations across routes")
                state_observations[key] = normalized
                # An untouched incoming leaf must reconstruct its fixed initial bow.
                if len(body[ids_key]) == 1 and role == "upper" and "upper_before_preparation" not in expected:
                    initial_bow = plan["context"]["initial_bows_um"][body[ids_key][0] - 1]
                    if not np.allclose(body["bow_um"], initial_bow, atol=1e-8, rtol=1e-8):
                        raise ValueError("incoming leaf bow differs from its fixed fabrication initialization")
            if full_affine:
                # Paper convention: upper minus lower in the common global frame.
                magnifications.append(np.asarray(record["upper"]["affine_face_strain"]) -
                                      np.asarray(record["lower"]["affine_face_strain"]))
            else:
                magnifications.append(paired_magnification(bodies["upper"], bodies["lower"]))
            physical_indices.append(expected["lower_free"][ids_key][-1] - 1)
        magnifications = np.asarray(magnifications)
        corrected = (1 - correction) * magnifications
        if full_affine:
            rotation = np.array([[0., -1.], [1., 0.]])
            matrices = corrected[None] + angles[:, physical_indices, None, None] * rotation
            displacements = np.einsum("neij,spj->nespi", matrices, xy)
            displacements += shifts[:, physical_indices, None, None, :]
        else:
            displacements = affine_overlay_um(
                xy[None, None, ...], corrected[None, :, None, None, :],
                translation_um=shifts[:, physical_indices, None, None, :],
                rotation_rad=angles[:, physical_indices, None, None])
        survival = worst_critical_pad_survival(displacements,
            critical_pad_mask=masks["critical_pad_mask"], threshold_um=threshold, sigma_um=sigma)
        probabilities = survival.probability
        summary = compound_site_yield(probabilities, masks["site_mask"])
        interfaces = []
        for index, expected in enumerate(route["events"]):
            marginal = compound_site_yield(probabilities[:, index:index+1, :], masks["site_mask"])
            matrix = magnifications[index] if full_affine else np.diag(magnifications[index])
            residual_matrix = (1 - correction) * matrix
            maximum = float(np.max(np.linalg.norm(selected_xy @ matrix.T, axis=1))) * 1000.
            interfaces.append(dict(event=expected["event"], interface_id=expected["interface_id"],
                                   lower_state_id=expected["lower_free"]["state_id"],
                                   upper_state_id=expected["upper_free"]["state_id"],
                                   magnification=magnifications[index].tolist(),
                                   residual_magnification=corrected[index].tolist(),
                                   magnification_ppm=(1e6 * matrix).tolist(),
                                   residual_magnification_ppm=(1e6 * residual_matrix).tolist(),
                                   max_bow_overlay_nm=maximum,
                                   max_residual_bow_overlay_nm=(1 - correction) * maximum,
                                   non_bow_overlay_sha256=non_bow_hashes[physical_indices[index]],
                                   overlay_yield=marginal.registered_site_yield))
        results.append(dict(order=route["order"], request_id=route["request_id"], interfaces=interfaces,
                            stack_overlay_yield=summary.registered_site_yield,
                            joint_site_summary=asdict(summary)))
    best = max(item["stack_overlay_yield"] for item in results)
    return dict(schema="yap.d2w_assembly_order_overlay.v1" if die else "yap.w2w_assembly_order_overlay.v1",
                status="IMPORTED_MECHANICS_OVERLAY_ONLY",
                physical_readiness=False, physics_qualification="UNKNOWN",
                context_id=plan["context_id"], mechanics_archive_sha256=content_id(archive),
                overlay_input_sha256=content_id(overlay), supplied_provenance=provenance,
                mechanics_configuration=archive.get("native_configuration"),
                scope="REFERENCE_bow_induced_overlay_not_total_manufacturing_yield",
                other_yield_mechanisms_included=False, joint_placement_draws=len(shifts), routes=results,
                magnification_report=dict(matrix_convention=("upper_minus_lower_in_die_centered_frame" if die
                                                             else "upper_minus_lower_in_wafer_centered_frame"),
                    matrix_units="ppm", dimensional_error_units="nm",
                    dimensional_error_domain="selected_site_and_critical_pad_coordinates",
                    dimensional_error_components="bow_induced_affine_only",
                    correction_fraction=correction,
                    correction_policy="full_affine_matrix_including_shear",
                    local_noise_treatment="same_analytic_sigma_per_physical_interface",
                    other_overlay_shared_across_orders=True),
                best_supplied_orders=[item["order"] for item in results
                                      if np.isclose(item["stack_overlay_yield"], best, rtol=0, atol=1e-12)])


def _load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def load_stack_input(path, *, stack_schema=STACK_SCHEMA, input_schema=INPUT_SCHEMA):
    """Read one design-local YAML without loading operators or changing plan inputs."""
    from omegaconf import OmegaConf
    from omegaconf.errors import OmegaConfBaseException
    from yaml import YAMLError

    try:
        value = OmegaConf.to_container(OmegaConf.load(path), resolve=False, throw_on_missing=True)
    except (OmegaConfBaseException, YAMLError, RecursionError) as exc:
        raise ValueError(f"invalid stack YAML: {exc}") from exc

    def check_types(item):
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise ValueError("stack YAML mapping keys must be strings")
            for child in item.values():
                check_types(child)
        elif isinstance(item, list):
            for child in item:
                check_types(child)
        elif isinstance(item, str):
            if "${" in item:
                raise ValueError("stack YAML requires explicit values, not interpolations")
        elif item is not None and type(item) not in (int, float, bool):
            raise ValueError("stack YAML must contain JSON-compatible values")

    check_types(value)
    _json(value)
    _keys(value, {"stack"}, "stack YAML", {"schema", "mechanics", "overlay"})
    value.setdefault("schema", stack_schema)
    if value["schema"] != stack_schema:
        raise ValueError(f"stack YAML schema must be {stack_schema}")
    if not isinstance(value["stack"], dict):
        raise ValueError("stack must be a mapping")
    stack = value["stack"]
    stack.setdefault("schema", input_schema)
    if stack["schema"] != input_schema:
        raise ValueError(f"stack schema must be {input_schema}")
    process = stack.get("process")
    if not isinstance(process, dict):
        raise ValueError("stack.process must be a mapping")
    # Fixed implementation policies remain in the normalized plan and its hash.
    # Only omission selects defaults; an explicit unsupported policy is an error.
    for key, default in _PROCESS_DEFAULTS.items():
        process.setdefault(key, default)
        if key != "recipe_id" and process[key] != default:
            raise ValueError(f"this planner requires {key}={default}")
    # Resolve only runtime asset paths; the stack payload used for IDs is unchanged.
    if "mechanics" in value:
        settings = value["mechanics"]
        _keys(settings, {"operator_package", "temporal_profile"}, "mechanics", {"metric", "layer_workers", "order_workers", "frozen_preview"})
        if type(settings.get("frozen_preview", False)) is not bool:
            raise ValueError("mechanics.frozen_preview must be Boolean")
        package = _label(settings["operator_package"], "mechanics.operator_package")
        _label(settings["temporal_profile"], "mechanics.temporal_profile")
        workers = settings.get("layer_workers", 1)
        if type(workers) is not int or workers < 1:
            raise ValueError("mechanics.layer_workers must be a positive integer")
        order_workers = settings.get("order_workers", 1)
        if type(order_workers) is not int or order_workers < 1:
            raise ValueError("mechanics.order_workers must be a positive integer")
        if order_workers > 1 and workers != 1:
            raise ValueError("parallel orders require mechanics.layer_workers=1")
        if settings.get("metric") is not None and not isinstance(settings["metric"], dict):
            raise ValueError("mechanics.metric must be a mapping")
        if package.startswith("resource:"):
            validate_resource_reference(package)
        elif package.startswith("component:"):
            if not package.removeprefix("component:").strip():
                raise ValueError("mechanics.operator_package requires a component identifier")
        else:
            settings["operator_package"] = str((Path(path).resolve().parent / package).resolve())
    if "overlay" in value and not isinstance(value["overlay"], dict):
        raise ValueError("overlay must be a mapping; omit the section to disable scoring")
    return value


def _write_new(path, value):
    # A failed evaluation never publishes a partial report or replaces a prior run.
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with validate_write_path(path).open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def main(argv=None):
    configure_runtime_storage()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Design-local stack.yaml (or legacy assembly JSON)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true", help="Emit requests; do not predict mechanics or yield")
    mode.add_argument("--mechanics-archive", help="Complete offline mechanics archive matching this request")
    mode.add_argument("--native-mechanics", nargs="?", const=True, metavar="LEGACY_JSON",
                      help="Execute F2B histories with explicitly supported operators; historical mixed-face plans remain blocked")
    parser.add_argument("--overlay-input", help="Legacy JSON only: explicit reference overlay settings")
    parser.add_argument("--archive-template", help="Write an unfilled mechanics template in plan-only mode")
    parser.add_argument("--mechanics-output", help="Also save native mechanics when scoring in the same run")
    parser.add_argument("--output", required=True, help="New output JSON path; existing files are not replaced")
    args = parser.parse_args(argv)
    try:
        is_yaml = Path(args.config).suffix.lower() in (".yaml", ".yml")
        native_requested = args.native_mechanics is not None
        if is_yaml:
            if args.overlay_input or isinstance(args.native_mechanics, str):
                raise ValueError("stack YAML uses inline mechanics/overlay; do not supply external settings")
            design = load_stack_input(args.config)
            config = design["stack"]
            settings = design.get("mechanics")
            overlay = design.get("overlay")
            if native_requested and settings is None:
                raise ValueError("--native-mechanics requires a mechanics section in stack YAML")
        else:
            if args.native_mechanics is True:
                raise ValueError("legacy JSON requires --native-mechanics SETTINGS_JSON; use stack.yaml for inline settings")
            config = _load(args.config)
            settings, overlay = None, None
        plan = build_plan(config)
        if not args.plan_only:
            require_executable_plan(plan)
        if not is_yaml:
            settings = _load(args.native_mechanics) if native_requested else None
            overlay = _load(args.overlay_input) if args.overlay_input else None
        if args.plan_only and args.overlay_input:
            raise ValueError("--overlay-input requires imported or native mechanics")
        if args.mechanics_archive and (overlay is None or args.archive_template):
            raise ValueError("scoring requires overlay settings and does not accept --archive-template")
        if native_requested and args.archive_template:
            raise ValueError("--archive-template is only used in plan-only mode")
        if args.mechanics_output and not (native_requested and overlay is not None):
            raise ValueError("--mechanics-output requires native mechanics and overlay scoring")
        paths = [validate_write_path(p) for p in (args.output, args.archive_template, args.mechanics_output) if p]
        if len(set(paths)) != len(paths) or any(p.exists() for p in paths):
            raise ValueError("all output paths must be new and distinct")
        if overlay is not None:
            _overlay_settings(overlay, len(plan["context"]["initial_bows_um"]),
                              500 * plan["context"]["geometry"]["diameter_mm"])
        if native_requested:
            from .assembly_mechanics import run_native_from_settings
            if not isinstance(settings, dict):
                raise ValueError("native settings must be a mapping")
            if overlay is not None and settings.get("metric") is None:
                raise ValueError("native overlay scoring requires an explicit metric policy in native settings")
            archive = run_native_from_settings(plan, settings, progress=lambda message: print(message, flush=True))
            output = evaluate_archive(plan, archive, overlay) if overlay is not None else archive
            if overlay is not None:
                output["status"] = "NATIVE_MECHANICS_OVERLAY_UNVALIDATED"
            if args.mechanics_output:
                _write_new(args.mechanics_output, archive)
        else:
            output = plan if args.plan_only else evaluate_archive(
                plan, _load(args.mechanics_archive), overlay)
        if args.archive_template:
            if Path(args.archive_template).resolve() == Path(args.output).resolve():
                raise ValueError("template and output paths must differ")
            if Path(args.output).exists() or Path(args.archive_template).exists():
                raise ValueError("output and template paths must be new")
            _write_new(args.archive_template, archive_template(plan))
        _write_new(args.output, output)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"assembly-order: {exc}\n")
    print(f"{output['status']}: {len(output['routes'])} assembly orders -> {args.output}")
    if args.plan_only:
        if output["schema"] == MIXED_PLAN_SCHEMA:
            summary = output["summary"]
            print(f"{summary['defined_routes']} defined process ledgers; "
                  f"{summary['missing_process_definitions']} missing process definitions.")
            print("Orientation and native qualification pending. No mechanics, yield or ranking predicted.")
        else:
            print("No mechanics or yield predicted.")
    elif "best_supplied_orders" in output:
        print("Best within supplied mechanics: " + ", ".join(output["best_supplied_orders"]))
        print("Physics qualification: UNKNOWN; history integrity is not independent physics validation.")
    else:
        print("Native histories complete; no yield/ranking without declared metric and overlay inputs.")


if __name__ == "__main__":
    main()
