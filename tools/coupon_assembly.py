"""Experimental homogenized prepared body, not a compatible microface model.

Caller supplies authenticated, equally area-normalized six-macro operators and
configures the existing native runtime. No file I/O, registration, Cu bridge,
port conversion, constitutive condensation or new equilibrium solver lives here.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import math
import sys
from types import MappingProxyType

import numpy as np


LABELS = ("eps_x0", "eps_y0", "gamma_xy0", "kappa_x", "kappa_y", "kappa_xy")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _vector(value, name, *, positive=False):
    result = np.array(value, dtype=float, copy=True)
    _require(result.shape == (6,) and np.isfinite(result).all()
             and (not positive or np.all(result > 0)), "invalid " + name)
    result.setflags(write=False)
    return result


def whole_prepared_private_size(operator):
    """Check dimensions only; custody and material admission remain caller-owned."""
    private = operator.bubble_size
    _require(type(operator.macro_size) is int and operator.macro_size == 6
             and type(private) is int and operator.coordinate_size == 6 + private,
             "whole prepared body requires six macros and consistent private coordinates")
    if private == 3:
        return private
    binding = getattr(operator, "shared_coupon_binding", None)
    _require(private in (5, 6, 21, 49, 51) and isinstance(binding, Mapping)
             and binding.get("schema") == "yap.shared_coupon_layer.v1"
             and binding.get("private_coordinates") == private
             and binding.get("reduced_coordinates") == private + 6
             and binding.get("all_native_GP_retained") is True
             and binding.get("public_admission") is False
             and binding.get("assembly_qualified") is False,
             "private5/6/natural21/49/51 whole prepared coordinates require an explicit shared-family binding")
    return private


def macro_transform(*, local_reference_y_mm, global_reference_y_mm, orientation):
    """Global plate q6 -> local q6; tangential axes remain unchanged.

    local_reference_y_mm is the physical location of the operator's datum in
    GLOBAL coordinates. Reversing the normal reverses local curvature only;
    forces assemble through the transpose, with no second operand transform.
    """
    _require(_finite(local_reference_y_mm) and _finite(global_reference_y_mm),
             "finite physical datums required")
    _require(type(orientation) is int and orientation in (-1, 1), "explicit normal orientation required")
    result = np.eye(6)
    result[:3, 3:] = (local_reference_y_mm - global_reference_y_mm) * np.eye(3)
    result[3:, 3:] *= orientation
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MacroOperand:
    name: str
    role: str
    operator: object
    thickness_mm: float
    area_mm2: float
    reference_y_mm: float
    orientation: int
    included_regions: tuple
    birth_q6: object
    age_offset_s: float = 0.0


@dataclass(frozen=True)
class PreparedBody:
    topology: object
    engine: object
    evidence: object

    def initial_state(self, q6=None):
        coordinates = _vector(np.zeros(6) if q6 is None else q6, "initial q6")
        return self.engine.initial_all_gauss_stack_state(self.topology.instances, coordinates)

    def trial(self, committed, temperature_c, delta_time_s, *, elapsed_time_s,
              creep_active, plastic_active=None, **solver_options):
        _require(_finite(temperature_c) and _finite(delta_time_s) and delta_time_s >= 0
                 and _finite(elapsed_time_s) and elapsed_time_s >= 0, "finite physical step/clock required")
        _require(type(creep_active) is bool and (plastic_active is None or type(plastic_active) is bool),
                 "explicit material activity flags required")
        protected = {"coordinate_scale", "constraint_matrix", "constraint_values"}
        _require(not protected.intersection(solver_options), "body topology controls are not solver overrides")
        return self.engine.trial_all_gauss_stack_increment(
            self.topology.instances, committed, temperature_c, delta_time_s,
            elapsed_time_s=elapsed_time_s, creep_active=creep_active, plastic_active=plastic_active,
            coordinate_scale=self.topology.coordinate_scale,
            constraint_matrix=self.topology.gauge_constraint_matrix,
            constraint_values=self.topology.gauge_constraint_values, **solver_options)

    def commit(self, trial):
        return self.engine.commit_all_gauss_stack_trial(trial)


def build_prepared_body(operands, *, si_thickness_um, global_reference_y_mm,
                        coordinate_scale, engine=None):
    """Build one prefabricated dual-BI body; staged attachment is not provided.

    Each operator must already have six macros; all existing private modes and
    all Cu GPs remain untouched. birth_q6 is explicit LOCAL natural strain and
    curvature, not an offset to apply again. reference_y_mm is measured from
    that operand's own local bottom before reflection. Caller owns custody,
    material compatibility, resource admission and paired physical validation.
    """
    rows = tuple(operands)
    _require(len(rows) == 2 and all(type(row) is MacroOperand for row in rows),
             "exact whole_coupon plus backside_bi inventory required")
    _require(tuple(row.role for row in rows) == ("whole_coupon", "backside_bi"),
             "ordered whole_coupon plus backside_bi required; no Si substitution")
    _require(_finite(si_thickness_um) and si_thickness_um > 0 and _finite(global_reference_y_mm),
             "positive Si thickness and finite plate datum required")
    _require(len({row.name for row in rows}) == 2
             and all(type(row.name) is str and row.name for row in rows), "unique layer names required")
    scale = _vector(coordinate_scale, "global coordinate scale", positive=True)
    births = []
    for row, regions, height in zip(rows, (("bi", "beol", "si"), ("bi",)),
                                    (.0115 + si_thickness_um / 1000., .0015)):
        _require(row.included_regions == regions, "explicit physical region inventory differs")
        _require(_finite(row.thickness_mm) and abs(row.thickness_mm - height) <= 1e-12,
                 "coupon/backside BI physical thickness differs")
        _require(_finite(row.area_mm2) and row.area_mm2 > 0, "positive operand area required")
        _require(_finite(row.reference_y_mm) and 0 <= row.reference_y_mm <= row.thickness_mm,
                 "local reference datum must lie within operand")
        _require(type(row.orientation) is int and row.orientation in (-1, 1), "explicit normal orientation required")
        _require(_finite(row.age_offset_s), "finite independent material age required")
        op = row.operator
        _require(type(op.macro_size) is int and op.macro_size == 6
                 and type(op.bubble_size) is int and op.bubble_size >= 0
                 and op.coordinate_size == 6 + op.bubble_size,
                 "proper six-macro operator required; ports are not removed or padded")
        births.append(_vector(row.birth_q6, "local birth q6"))
    _require(rows[0].operator.bubble_size == 3, "whole coupon requires its three original private modes")
    _require(rows[0].orientation == 1 and rows[1].orientation == -1,
             "prepared inventory requires front coupon +1 and backside BI -1")
    _require(rows[0].area_mm2 == rows[1].area_mm2, "operands must already share the same integrated area")
    from W2W.assembly_mechanics import LayerClockOperator, native
    if engine is None:
        engine = native("exp12_all_gauss_periodic_stack")
    instances, cursor = [], 0.0
    for row, birth in zip(rows, births):
        datum = cursor + (row.reference_y_mm if row.orientation == 1
                          else row.thickness_mm - row.reference_y_mm)
        transform = macro_transform(local_reference_y_mm=datum,
                                    global_reference_y_mm=global_reference_y_mm,
                                    orientation=row.orientation)
        offset = np.zeros(6)
        offset.setflags(write=False)
        instance = engine.AllGaussStackLayerInstance(
            name=row.name, operator=LayerClockOperator(row.operator, age_offset_s=row.age_offset_s),
            macro_transform=transform, macro_offset=offset, virgin_birth_macro_coordinates=birth)
        instance.validate(6)
        instances.append(instance)
        cursor += row.thickness_mm
    constraints, values = np.zeros((0, 6)), np.zeros(0)
    constraints.setflags(write=False)
    values.setflags(write=False)
    topology = engine.AllGaussPeriodicPortStackTopology(
        instances=tuple(instances), coordinate_scale=scale, coordinate_labels=LABELS,
        port_rank=0, interface_slices=(), gauge_constraint_matrix=constraints,
        gauge_constraint_values=values)
    evidence = MappingProxyType(dict(
        schema="yap.experimental_homogenized_coupon_body.v1", prepared_total_um=cursor * 1000.,
        physical_inventory=("front_BI1.5", "BEOL10", "Si_TSV", "back_BI1.5"),
        area_mm2=rows[0].area_mm2, global_reference_y_mm=global_reference_y_mm,
        approximate_homogenized_coupling=True, microface_compatible=False,
        local_states_and_clocks_independent=True, source_custody_verified_here=False,
        material_compatibility_verified_here=False, paired_prepared_validation_done=False,
        paired_stack_validation_done=False, assembly_accuracy_admitted=False,
        all_orders_admitted=False, staged_attachment_supported=False,
        accuracy_gate="global cooldown bow relative error <=10% OR absolute error <=1um"))
    return PreparedBody(topology, engine, evidence)


def build_whole_prepared_body(operands, *, global_reference_y_mm,
                              coordinate_scale, engine=None):
    """Initialize whole dual-BI D2W operands on one fixed GLOBAL q6 datum.

    Unlike build_prepared_body, each row.reference_y_mm is the physical
    operator datum in GLOBAL coordinates, not a distance from a local bottom.
    Thickness and area are explicit caller assertions, not authenticated here.
    No backside layer is added. Each row owns all four regions exactly once.

    Create each physical body's clock proxy once. A supplied LayerClockOperator
    is preserved, with a matching age_offset_s; otherwise one is created here.
    Subsequent formation must retain the returned instances/operators and their
    fixed transforms, not rebuild aged children from unwrapped operands. Native
    capture_formation owns offsets and deep state transfer; this builder neither
    adopts histories nor provides removal/deposition or W2W birth semantics.
    """
    rows = tuple(operands)
    _require(rows and all(type(row) is MacroOperand for row in rows),
             "one or more whole-prepared operands required")
    _require(all(type(row.name) is str and row.name for row in rows)
             and len({row.name for row in rows}) == len(rows), "unique body names required")
    _require(_finite(global_reference_y_mm), "finite fixed global datum required")
    scale = _vector(coordinate_scale, "global coordinate scale", positive=True)
    births = []
    for row in rows:
        _require(row.role == "whole_prepared"
                 and row.included_regions == ("front_bi", "beol", "si", "back_bi"),
                 "exact whole-prepared dual-BI inventory required; no extra backside BI")
        _require(_finite(row.thickness_mm) and row.thickness_mm > 0,
                 "positive explicit whole-body thickness required")
        _require(_finite(row.area_mm2) and row.area_mm2 > 0,
                 "positive explicit whole-body area required")
        _require(row.area_mm2 == rows[0].area_mm2,
                 "whole bodies must already share the same integrated area")
        _require(_finite(row.reference_y_mm), "finite physical GLOBAL operator datum required")
        _require(type(row.orientation) is int and row.orientation == 1,
                 "whole-prepared prototype supports only +1 D2W orientation")
        _require(_finite(row.age_offset_s) and row.age_offset_s >= 0,
                 "finite nonnegative independent material age required")
        op = row.operator
        whole_prepared_private_size(op)
        births.append(_vector(row.birth_q6, "local birth q6"))
    from W2W.assembly_mechanics import LayerClockOperator, native
    if engine is None:
        engine = native("exp12_all_gauss_periodic_stack")
    instances, proxies = [], set()
    clock_types = (LayerClockOperator,)
    for module_name in ("W2W.assembly_mechanics", "YAP.W2W.assembly_mechanics"):
        clock_type = getattr(sys.modules.get(module_name), "LayerClockOperator", None)
        if isinstance(clock_type, type) and clock_type not in clock_types:
            clock_types += (clock_type,)
    for row, birth in zip(rows, births):
        if isinstance(row.operator, clock_types):
            operator = row.operator
            _require(operator.age_offset_s == row.age_offset_s,
                     "existing body clock age must remain unchanged")
        else:
            operator = LayerClockOperator(row.operator, age_offset_s=row.age_offset_s)
        _require(id(operator) not in proxies, "each physical body requires its own stable clock proxy")
        proxies.add(id(operator))
        transform = macro_transform(local_reference_y_mm=row.reference_y_mm,
                                    global_reference_y_mm=global_reference_y_mm, orientation=1)
        offset = np.zeros(6)
        offset.setflags(write=False)
        instance = engine.AllGaussStackLayerInstance(
            name=row.name, operator=operator, macro_transform=transform, macro_offset=offset,
            virgin_birth_macro_coordinates=birth)
        instance.validate(6)
        instances.append(instance)
    constraints, values = np.zeros((0, 6)), np.zeros(0)
    constraints.setflags(write=False)
    values.setflags(write=False)
    topology = engine.AllGaussPeriodicPortStackTopology(
        instances=tuple(instances), coordinate_scale=scale, coordinate_labels=LABELS,
        port_rank=0, interface_slices=(slice(6, 6),) * (len(instances) + 1),
        gauge_constraint_matrix=constraints, gauge_constraint_values=values)
    evidence = MappingProxyType(dict(
        schema="yap.experimental_whole_prepared_coupon_body.v1",
        body_names=tuple(row.name for row in rows),
        body_thicknesses_mm=tuple(row.thickness_mm for row in rows),
        prepared_total_um=sum(row.thickness_mm for row in rows) * 1000.,
        physical_inventory=("front_bi", "beol", "si", "back_bi"), included_BI_layers_per_body=2,
        area_mm2=rows[0].area_mm2, global_reference_y_mm=global_reference_y_mm,
        operator_global_reference_y_mm=tuple(row.reference_y_mm for row in rows),
        approximate_homogenized_coupling=True, microface_compatible=False,
        local_states_and_clocks_independent=True, source_custody_verified_here=False,
        material_compatibility_verified_here=False, paired_prepared_validation_done=False,
        paired_stack_validation_done=False, assembly_accuracy_admitted=False,
        all_orders_admitted=False, W2W_birth_removal_supported=False,
        accuracy_gate="global cooldown bow relative error <=10% OR absolute error <=1um"))
    return PreparedBody(topology, engine, evidence)
