"""Internal prepared-D2W component bridge, never a production admission path.

The caller supplies the independently reviewed certificate's SHA256. Hashing
operands here checks that binding; it does not create a physical certificate.
Version 1 retains the synthetic one-translation/identity-reflection contract.
Version 2 checks three physical translations and thickness reflection closure.
No coordinate conversion, rank inference or geometry scaling occurs.
"""

from collections.abc import Mapping
import hashlib
import json

import numpy as np

try:
    from .. import mechanics_runtime as runtime
except ImportError:
    import mechanics_runtime as runtime


_ROLES = ("bi", "beol", "retained_si")
_GEOMETRY_KEYS = {"width_mm", "length_mm", "total_thickness_um",
                  "front_bi_thickness_um", "back_bi_thickness_um", "beol_thickness_um"}
_CERTIFICATE_KEYS = {"schema", "scope", "geometry", "bi_layer_count", "cell_area_mm2",
                     "macro_coordinate_scale", "reflection_matrix", "rigid_port_mode_indices",
                     "thickness_orientations", "macro_contract", "components"}
_MACRO_CONTRACT = {
    "labels": ["eps_x", "eps_y", "gamma_xy", "kappa_x", "kappa_y", "kappa_xy"],
    "input_units": ["1", "1", "1", "1/mm", "1/mm", "1/mm"],
    "physical_coordinate_order": ["x", "thickness_y", "public_y_as_z"],
    "displacement_order": ["UX", "UY", "UZ"],
    "displacement_units": "mm", "reference": "layer_midplane",
    "source_reference_offset_mm": 0., "in_plane_origin": "cell_center",
    "shear_and_twist": "engineering", "normal_curvature_sign": -1,
}
_TRACE_ATOL_MM = 1e-12


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256_document(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("ascii")).hexdigest()


def _numeric_copy(values):
    _require(isinstance(values, Mapping) and 0 < len(values) <= 64,
             "bounded numeric operand mapping required")
    result = {}
    total = 0
    for key, value in values.items():
        _require(isinstance(key, str) and isinstance(value, np.ndarray)
                 and value.dtype.kind in "fiu" and np.isfinite(value).all(),
                 "finite numeric arrays required")
        total += value.nbytes
        _require(total <= 2 * 1024**3, "component operand byte budget exceeded")
        result[key] = np.array(value, copy=True, order="C")
        result[key].flags.writeable = False
    return result


def _sha256_arrays(values):
    digest = hashlib.sha256()
    for key in sorted(values):
        value = values[key]
        header = json.dumps([key, value.dtype.str, value.shape], separators=(",", ":"))
        digest.update(header.encode("ascii") + b"\n")
        digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
    return digest.hexdigest()


def _geometry(geometry, *, physical_v2=False, certified_components=False):
    _require(isinstance(geometry, Mapping) and set(geometry) == _GEOMETRY_KEYS,
             "exact prepared-D2W geometry required")
    for value in geometry.values():
        _require(type(value) in (int, float) and np.isfinite(value) and value > 0,
                 "finite positive geometry required")
    _require(geometry["width_mm"] == geometry["length_mm"], "equal-square prepared dies required")
    allowed = (50., 61.5, 80.) if physical_v2 else (50., 80.)
    total_allowed = geometry["total_thickness_um"] > 13. if certified_components else (
        geometry["total_thickness_um"] in allowed)
    _require(total_allowed
             and geometry["front_bi_thickness_um"] == geometry["back_bi_thickness_um"] == 1.5
             and geometry["beol_thickness_um"] == 10.,
             "bridge requires exact registered dual-BI geometry")
    return {"bi": .0015, "beol": .010,
            "retained_si": (geometry["total_thickness_um"] - 13.) / 1000}


def _canonical_macro_trace(xyz, height):
    """Independent centered engineering-q6 displacement polynomial, in mm."""
    x = xyz[:, 0] - .5 * (xyz[:, 0].min() + xyz[:, 0].max())
    z = xyz[:, 2] - .5 * (xyz[:, 2].min() + xyz[:, 2].max())
    fields = np.zeros((len(xyz), 3, 6))
    fields[:, 0, 0], fields[:, 2, 1] = x, z
    fields[:, 0, 2], fields[:, 2, 2] = .5*z, .5*x
    fields[:, 0, 3], fields[:, 1, 3] = height*x, -.5*x*x
    fields[:, 2, 4], fields[:, 1, 4] = height*z, -.5*z*z
    fields[:, 0, 5], fields[:, 2, 5] = .5*height*z, .5*height*x
    fields[:, 1, 5] = -.5*x*z
    return fields.reshape(-1, 6)


def _trace_contract(physical, arrays, thickness, area, *, physical_v2=False, reflection=None):
    _require(set(physical) == {"face_node_ids", "face_coordinates_mm", "face_lifts"},
             "complete physical face trace required")
    ids, xyz, lifts = (physical[key] for key in
                       ("face_node_ids", "face_coordinates_mm", "face_lifts"))
    n = ids.shape[1] if ids.ndim == 2 else 0
    bubbles = arrays["c"].shape[0]
    _require(22 <= n <= 4096 and ids.shape == (2, n) and ids.dtype.kind in "iu"
             and xyz.shape == (2, n, 3) and lifts.shape == (2, 3*n, 134+bubbles),
             "complete ordered physical face dimensions required")
    _require(all(len(np.unique(face)) == n for face in ids), "duplicate physical face node IDs")
    correct_height = (np.allclose(xyz[0, :, 1], 0., rtol=0, atol=_TRACE_ATOL_MM)
                      and np.allclose(xyz[1, :, 1], thickness, rtol=0, atol=_TRACE_ATOL_MM)) if physical_v2 else (
                          np.all(xyz[0, :, 1] == 0.) and np.all(xyz[1, :, 1] == thickness))
    _require(np.array_equal(xyz[0][:, [0, 2]], xyz[1][:, [0, 2]]) and correct_height,
             "ordered matching faces with exact local thickness required")
    lateral = xyz[0][:, [0, 2]]
    _require(len(np.unique(lateral, axis=0)) == n
             and np.isclose(np.prod(np.ptp(lateral, axis=0)), area, rtol=0, atol=1e-15),
             "physical cell area or face-node census differs")
    for face in (0, 1):
        expected = _canonical_macro_trace(xyz[face], (face - .5)*thickness)
        _require(np.allclose(lifts[face, :, :6], expected, rtol=0, atol=_TRACE_ATOL_MM),
                 "physical macro trace differs from centered local-midplane engineering q6")
    bottom, top = lifts[0, :, 6:70], lifts[1, :, 70:134]
    _require(np.array_equal(bottom, top) and not np.any(lifts[0, :, 70:])
             and not np.any(lifts[1, :, 6:70]) and not np.any(lifts[:, :, 134:]),
             "physical port order/trace or zero-face bubble contract differs")
    _require(np.linalg.matrix_rank(bottom) == 64, "physical face trace is rank deficient")
    rigid = bottom[:, 0].reshape(n, 3)
    _require(np.array_equal(rigid, np.broadcast_to(rigid[0], rigid.shape))
             and np.linalg.norm(rigid[0]) > 0,
             "rigid port zero must be an actual common translation trace")
    if physical_v2:
        translations = bottom[:, :3].reshape(n, 3, 3)
        constant = np.broadcast_to(translations[0], translations.shape)
        _require(np.allclose(translations, constant, rtol=0, atol=_TRACE_ATOL_MM)
                 and np.linalg.matrix_rank(translations[0]) == 3,
                 "three rigid ports must span independent constant physical translations")
        reflected = np.tile([1., -1., 1.], n)[:, None] * bottom
        _require(np.allclose(bottom @ reflection, reflected, rtol=0, atol=_TRACE_ATOL_MM),
                 "reflection does not close on the actual physical displacement trace")


def _native_face_kinematics(library, physical_by_role):
    """Compare physical traces through actual executor-native placement maps.

    Use two prepared dies so both the internal Si/back-BI interface and the
    back-BI(-1)/next-front-BI(+1) interface are checked. Expected displacement
    columns come from the global polynomial, not the native transform under
    test. This is a kinematic identity, not process or material qualification.
    """
    from .assembly_mechanics import native

    api = native("exp12_v2_production_sequential_process")
    placements, cursor = [], 0.
    roles = (("front_bi", "bi"), ("beol", "beol"),
             ("retained_si", "retained_si"), ("back_bi", "bi"))
    for die in range(2):
        for name, role in roles:
            placement = api._placement(library, name=f"bridge_die{die}_{name}", operator_key=role,
                die_index=die, role=name, z_bottom_mm=cursor,
                thickness_orientation=-1 if name == "back_bi" else 1)
            placements.append(placement)
            cursor = placement.z_top_mm
    topology = api._topology(library, tuple(placements), {})
    _require(len(topology.instances) == 8 and len(topology.interface_slices) == 9,
             "actual native topology must retain both complete prepared dies")
    common_ports = physical_by_role["bi"]["face_lifts"][0, :, 6:70]
    maximum_error = 0.
    previous_top = None
    for index, (placement, instance) in enumerate(zip(placements, topology.instances)):
        _require(instance.macro_transform.shape == (134, 582),
                 "actual native transform does not match the explicit two-die coordinate contract")
        physical = physical_by_role[placement.operator_key]
        xyz, lifts = physical["face_coordinates_mm"], physical["face_lifts"]
        sign = placement.thickness_orientation
        vector_frame = np.tile([1., float(sign), 1.], len(xyz[0]))[:, None]
        faces = {}
        for local_face in (0, 1):
            global_face = local_face if sign == 1 else 1-local_face
            global_height = placement.z_bottom_mm if global_face == 0 else placement.z_top_mm
            actual = vector_frame * (lifts[local_face, :, :134] @ instance.macro_transform)
            expected = np.zeros_like(actual)
            expected[:, :6] = _canonical_macro_trace(xyz[local_face], global_height)
            expected[:, topology.interface_slices[index+global_face]] = common_ports
            error = float(np.max(np.abs(actual - expected)))
            _require(error <= _TRACE_ATOL_MM,
                     "physical trace/native transform mismatch, including backside orientation")
            maximum_error = max(maximum_error, error)
            faces[global_face] = actual
        if previous_top is not None:
            _require(np.allclose(previous_top, faces[0], rtol=0, atol=_TRACE_ATOL_MM),
                     "adjacent physical interface displacement columns do not coincide")
        previous_top = faces[1]
    return dict(module_count=8, face_count=16, shared_interface_count=7,
                maximum_absolute_error_mm=maximum_error, absolute_tolerance_mm=_TRACE_ATOL_MM)


def _gp_contract(ordered_gp, arrays):
    _require(set(ordered_gp) == {"element_gauss_ids", "coordinates_mm"},
             "ordered physical Gauss IDs and coordinates required")
    ids, xyz = ordered_gp["element_gauss_ids"], ordered_gp["coordinates_mm"]
    count = len(arrays["weights"])
    _require(ids.shape == (count, 2) and ids.dtype.kind in "iu"
             and xyz.shape == (count, 3) and len(np.unique(ids, axis=0)) == count
             and np.all(ids[:, 0] > 0) and np.all((ids[:, 1] >= 0) & (ids[:, 1] < 8)),
             "unique complete ordered Cu Gauss-point identity required")
    _require(count > 0 and count % 8 == 0
             and np.array_equal(ids[:, 1], np.tile(np.arange(8), count // 8))
             and np.all(ids[:, 0].reshape(-1, 8) == ids[::8, 0, None]),
             "complete ordered eight-point Cu element quadrature required")


def build_component_assembly_library(operands, *, geometry, certificate,
                                     expected_certificate_sha256, cu_card,
                                     material_kernels, material_binding):
    """Build an internal library from pinned, already-certified physical operands.

    Each role supplies arrays, physical_trace and ordered_gp. The externally
    pinned certificate binds all three, exact geometry, scales and virtual-work
    certification. A synthetic fixture certificate makes no physical claim.
    No public catalog resolver, product setting or filesystem loader is used.
    """
    _require(isinstance(expected_certificate_sha256, str)
             and len(expected_certificate_sha256) == 64
             and all(c in "0123456789abcdef" for c in expected_certificate_sha256),
             "caller-bound certificate SHA256 required")
    _require(isinstance(certificate, dict) and set(certificate) == _CERTIFICATE_KEYS,
             "exact internal component certificate required")
    # Hash the detached metadata that will actually govern construction.
    certificate = json.loads(json.dumps(certificate, allow_nan=False))
    _require(_sha256_document(certificate) == expected_certificate_sha256,
             "certificate differs from caller-bound SHA256")
    certified_components = certificate["schema"] == "yap.component_prepared_assembly_certificate.v3"
    physical_v2 = certified_components or certificate["schema"] == "yap.internal_component_assembly_certificate.v2"
    _require(certificate["schema"] in ("yap.internal_component_assembly_certificate.v1",
                                      "yap.internal_component_assembly_certificate.v2",
                                      "yap.component_prepared_assembly_certificate.v3")
             and certificate["scope"] in ("synthetic_fixture", "experimental")
             and type(certificate["bi_layer_count"]) is int and certificate["bi_layer_count"] == 2,
             "internal dual-BI certificate required; single-BI is not interchangeable")
    _require(isinstance(material_binding, runtime.CorrectedMaterialBinding)
             and material_binding.scope == certificate["scope"],
             "explicit matching corrected material binding scope required")
    _require(certificate["macro_contract"] == _MACRO_CONTRACT,
             "explicit centered midplane q6 order, units and reference contract required")
    thicknesses = _geometry(geometry, physical_v2=physical_v2,
                           certified_components=certified_components)
    _require(certificate["geometry"] == dict(geometry), "certificate geometry differs")
    rigid_indices = [0, 1, 2] if physical_v2 else [0]
    _require(certificate["rigid_port_mode_indices"] == rigid_indices
             and all(type(v) is int for v in certificate["rigid_port_mode_indices"]),
             "explicit versioned rigid-mode gauge required")
    _require(certificate["thickness_orientations"] == [1, -1]
             and all(type(v) is int for v in certificate["thickness_orientations"]),
             "prepared backside thickness orientation must be explicitly certified")
    reflection = np.asarray(certificate["reflection_matrix"], dtype=float)
    if physical_v2:
        _require(reflection.shape == (64, 64) and np.isfinite(reflection).all()
                 and np.allclose(reflection.T @ reflection, np.eye(64), rtol=0, atol=1e-12)
                 and np.allclose(reflection @ reflection, np.eye(64), rtol=0, atol=1e-12),
                 "physical reflection must be a finite orthogonal involution")
    else:
        _require(reflection.shape == (64, 64) and np.array_equal(reflection, np.eye(64)),
                 "nonidentity reflection is unsupported by the v1 certificate")
    scale = np.asarray(certificate["macro_coordinate_scale"], dtype=float)
    _require(scale.shape == (134,) and np.isfinite(scale).all() and np.all(scale > 0)
             and np.array_equal(scale[6:70], scale[70:134]), "exact common coordinate scales required")
    area = certificate["cell_area_mm2"]
    _require(type(area) in (int, float) and np.isfinite(area) and area > 0,
             "finite positive certified cell area required")
    _require(isinstance(operands, Mapping) and set(operands) == set(_ROLES)
             and isinstance(certificate["components"], dict)
             and set(certificate["components"]) == set(_ROLES), "exact three component roles required")
    prepared, physical_by_role, common_trace = {}, {}, None
    for role in _ROLES:
        operand, pin = operands[role], certificate["components"][role]
        _require(isinstance(operand, Mapping) and set(operand) == {"arrays", "physical_trace", "ordered_gp"},
                 "physical trace and ordered Gauss evidence are mandatory")
        _require(isinstance(pin, dict) and set(pin) == {"thickness_mm", "arrays_sha256",
                 "physical_trace_sha256", "ordered_gp_sha256", "virtual_work_certified"}
                 and pin["virtual_work_certified"] is True,
                 "pinned physical virtual-work certificate required")
        _require(pin["thickness_mm"] == thicknesses[role], "exact role thickness differs")
        copies = {key: _numeric_copy(operand[key]) for key in operand}
        for key in ("arrays", "physical_trace", "ordered_gp"):
            _require(_sha256_arrays(copies[key]) == pin[key + "_sha256"],
                     role + ": " + key + " differs from physical certificate")
        _trace_contract(copies["physical_trace"], copies["arrays"], thicknesses[role], area,
                        physical_v2=physical_v2, reflection=reflection)
        _gp_contract(copies["ordered_gp"], copies["arrays"])
        physical = copies["physical_trace"]
        physical_by_role[role] = physical
        actual_trace = (physical["face_coordinates_mm"][0][:, [0, 2]],
                        physical["face_lifts"][0, :, 6:70])
        if common_trace is None:
            common_trace = actual_trace
        _require(all(np.array_equal(actual, common) for actual, common in zip(actual_trace, common_trace)),
                 "components do not share the same actual ordered physical port trace")
        prepared[role] = runtime.component_native_layer_arguments(copies["arrays"],
            cu_card=cu_card, material_kernels=material_kernels, macro_coordinate_scale=scale,
            material_binding=material_binding)
    from .assembly_mechanics import native, require_corrected_material_runtime

    layer_type = native("exp12_all_gauss_reduced_fe2").AllGaussReducedFE2Layer
    library_type = native("exp12_mindlin_sequential_runtime").Q6SequentialLibrary
    operators = {role: runtime.attach_corrected_material_binding(layer_type(**arguments), material_binding)
                 for role, arguments in prepared.items()}
    reflection.flags.writeable = scale.flags.writeable = False
    library = library_type(operators=operators, thickness_mm=thicknesses, reflection_matrix=reflection,
        rigid_port_mode_indices=tuple(rigid_indices), membrane_bending_coordinate_scale=scale[:6],
        port_coordinate_scale=scale[6:70], cell_area_mm2=area, port_rank_per_face=64,
        common_port_gauge_vectors=None)
    require_corrected_material_runtime(library)
    kinematics = _native_face_kinematics(library, physical_by_role)
    return library, dict(schema="yap.internal_component_assembly_bridge.v2" if physical_v2
                        else "yap.internal_component_assembly_bridge.v1",
        certificate_sha256=expected_certificate_sha256, scope=certificate["scope"],
        material=runtime.corrected_material_provenance(material_binding),
        macro_contract=certificate["macro_contract"], native_face_kinematics=kinematics,
        assembly_execution_supported=False, physical_readiness=False,
        physics_qualification="INTERNAL_EXPERIMENT_ONLY_NOT_ASSEMBLY_ADMISSION")
