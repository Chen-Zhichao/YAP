"""Caller-bound rigid-gauge congruence and native layer; no decoding or solver.

For BI4, 185 original coordinates become six macros and 176 private unknowns.
Only caller-certified translations are quotiented. Private coordinates remain
free in subsequent nonlinear equilibrium, not elastically condensed away.
Metadata authentication, source custody and resource admission belong to caller.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType

import numpy as np


LABELS = ("eps_x0", "eps_y0", "gamma_xy0", "kappa_x", "kappa_y", "kappa_xy")
RTOL, ATOL, RANK_RTOL = 1e-10, 1e-12, 1e-12


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def metadata_sha256(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("ascii")).hexdigest()


def _array(value, shape, name):
    _require(type(value) is np.ndarray and value.dtype == np.dtype("float64")
             and value.shape == shape and np.isfinite(value).all(), "invalid " + name)


def _full_rank(value, rank, name):
    singular = np.linalg.svd(value, compute_uv=False)
    _require(len(singular) == rank and singular[-1] > RANK_RTOL * singular[0],
             "unresolved or deficient rank: " + name)


def _zero(value, scale, name):
    error = float(np.linalg.norm(value))
    _require(error <= ATOL + RTOL * float(scale), name)
    return error


@dataclass(frozen=True)
class BacksideOperands:
    stiffness: object
    thermal_per_c: object
    strain_maps: object
    weights_mm3: object
    source_from_reduced: object
    material_binding: object
    evidence: object

    @property
    def macro_size(self):
        return 6

    @property
    def bubble_size(self):
        return self.stiffness.shape[0] - 6


def transform(stiffness, thermal_per_c, strain_maps, weights_mm3, *,
              macro_embedding, private_basis, gauge_rows, translation_basis,
              metadata, expected_metadata_sha256, material_binding,
              check, block_gp=512):
    """Apply U=[A,Z] to integrated K/f/B with q_original == q_reduced.

    C may depend on q6: A must then include its compensating private entries.
    U must span ALL ker(C). translation_basis supplies the original three
    zero-q6 rigid translations; C must fix them nonsingularly. No gauge is
    inferred. B has shape (GP, native engineering six, original coordinates).
    The binding is carried by identity, not authenticated by this module.
    """
    _require(callable(check) and material_binding is not None, "caller guard and material binding required")
    check()
    _require(type(block_gp) is int and 1 <= block_gp <= 4096, "bounded GP tile required")
    _require(type(stiffness) is np.ndarray and stiffness.ndim == 2,
             "original stiffness array required")
    n = stiffness.shape[0]
    _require(10 <= n <= 512, "bounded original coordinate count required")
    _require(type(weights_mm3) is np.ndarray and weights_mm3.ndim == 1,
             "original ordered weights required")
    gp = len(weights_mm3)
    _require(gp > 0, "nonempty original GP census required")
    _array(stiffness, (n, n), "stiffness")
    _array(thermal_per_c, (n,), "thermal load")
    _array(weights_mm3, (gp,), "GP weights")
    _require(np.all(weights_mm3 > 0), "positive unchanged GP weights required")
    _require(type(strain_maps) is np.ndarray and strain_maps.dtype == np.dtype("float64")
             and strain_maps.shape == (gp, 6, n), "complete ordered engineering strain map required")
    a, z, c, rigid = macro_embedding, private_basis, gauge_rows, translation_basis
    for value, shape, name in ((a, (n, 6), "macro embedding"), (z, (n, n-9), "private basis"),
                               (c, (3, n), "explicit gauge rows"), (rigid, (n, 3), "certified translations")):
        _array(value, shape, name)
    keys = {"source_sha256", "gauge_sha256", "basis_sha256", "original_coordinates", "GP_count",
            "macro_labels", "strain_order", "included_regions", "source_area_mm2", "reference_y_mm"}
    _require(type(metadata) is dict and set(metadata) == keys, "explicit source/gauge/basis/frame metadata required")
    _require(type(expected_metadata_sha256) is str and len(expected_metadata_sha256) == 64
             and metadata_sha256(metadata) == expected_metadata_sha256, "metadata SHA256 differs")
    for key in ("source_sha256", "gauge_sha256", "basis_sha256"):
        pin = metadata[key]
        _require(type(pin) is str and len(pin) == 64 and set(pin) <= set("0123456789abcdef"),
                 "full source/gauge/basis SHA256 required")
    _require(type(metadata["original_coordinates"]) is int and metadata["original_coordinates"] == n
             and type(metadata["GP_count"]) is int and metadata["GP_count"] == gp
             and metadata["macro_labels"] == list(LABELS)
             and metadata["strain_order"] == "native_engineering_six_unchanged"
             and metadata["included_regions"] == ["bi"], "BI-only coordinate/GP/frame metadata differs")
    _require(all(type(metadata[key]) in (int, float) and math.isfinite(metadata[key])
                 for key in ("source_area_mm2", "reference_y_mm"))
             and metadata["source_area_mm2"] > 0, "finite source area/datum required")
    _require(np.array_equal(a[:6], np.eye(6)) and not np.any(z[:6]) and not np.any(rigid[:6]),
             "exact q6-preserving embedding and zero-q6 translations required")
    u = np.column_stack((a, z))
    _full_rank(c, 3, "gauge")
    _full_rank(rigid, 3, "translations")
    _full_rank(c @ rigid, 3, "gauge on translations")
    _full_rank(u, n-3, "complete gauge kernel")
    gauge_error = _zero(c @ u, np.linalg.norm(c) * np.linalg.norm(u), "embedding violates explicit gauge")
    # Orthonormalize only the supplied rigid span for scale-independent checks.
    r = np.linalg.qr(rigid, mode="reduced")[0]
    _zero(stiffness-stiffness.T, np.linalg.norm(stiffness), "original stiffness is not symmetric")
    _zero(stiffness @ r, np.linalg.norm(stiffness), "removed translations carry stiffness")
    _zero(r.T @ thermal_per_c, np.linalg.norm(thermal_per_c), "removed translations carry thermal work")
    check()
    k, f = u.T @ stiffness @ u, u.T @ thermal_per_c
    # This diagnostic does not symmetrize stored operators or discard modes.
    np.linalg.cholesky((k + k.T) * .5)
    maps = np.empty((gp, 6, n-3), dtype=np.float64)
    maximum_rigid_strain_error = 0.0
    for first in range(0, gp, block_gp):
        check()
        block = strain_maps[first:first+block_gp]
        _require(np.isfinite(block).all(), "nonfinite original GP map")
        error = _zero(block @ r, np.linalg.norm(block), "removed translations carry GP strain")
        maximum_rigid_strain_error = max(maximum_rigid_strain_error, error)
        maps[first:first+block_gp] = block @ u
        _require(np.isfinite(maps[first:first+block_gp]).all(), "nonfinite transformed GP map")
    weights = weights_mm3.copy()
    _require(np.isfinite(k).all() and np.isfinite(f).all(), "nonfinite transformed operands")
    for value in (k, f, maps, weights, u):
        value.setflags(write=False)
    check()
    evidence = MappingProxyType(dict(
        schema="yap.experimental_backside_gauge_congruence.v1", metadata_sha256=expected_metadata_sha256,
        source_sha256=metadata["source_sha256"], gauge_sha256=metadata["gauge_sha256"],
        basis_sha256=metadata["basis_sha256"], source_area_mm2=metadata["source_area_mm2"],
        reference_y_mm=metadata["reference_y_mm"], original_coordinates=n, reduced_coordinates=n-3,
        macro_coordinates=6, private_coordinates=n-9, removed_rigid_translations=3,
        complete_gauge_kernel=True, gauge_residual_norm=gauge_error,
        maximum_rigid_strain_error=maximum_rigid_strain_error, quotient_symmetric_part_SPD=True,
        GP_count=gp, GP_order_preserved=True, engineering_shear_rescaled=False,
        elastic_condensation=False, compliant_mode_truncation=False, private_coordinates_retained_nonlinearly=True,
        original_arrays_modified=False, material_binding_authenticated_here=False,
        source_custody_verified_here=False, Cu_updates=0, paired_physical_validation_done=False,
        operator_admission=False, production_ready=False))
    return BacksideOperands(k, f, maps, weights, u, material_binding, evidence)


def _native_layer_type(coupon):
    try:
        from YAP.W2W.assembly_mechanics import native
    except ModuleNotFoundError as error:
        if error.name != 'YAP':
            raise
        from W2W.assembly_mechanics import native
    return coupon.unit_invariant_layer_type(native('exp12_all_gauss_reduced_fe2'))


def construct_layer(operands, *, layer_type=None):
    """Construct the existing native layer without condensing or correcting K/B.

    Material authentication is delegated to the existing coupon validator.
    Actual-data parity, histories and assembly accuracy remain unqualified.
    """
    from .operators import kinematics as coupon

    _require(isinstance(operands, BacksideOperands), 'BacksideOperands required')
    material, bound_runtime = coupon._material(operands.material_binding)
    k, f = operands.stiffness, operands.thermal_per_c
    n, gp = len(k), len(operands.weights_mm3)
    _require(7 <= n <= 509 and gp > 0, 'bounded retained coordinates and GP census required')
    _array(k, (n, n), 'backside stiffness')
    _array(f, (n,), 'backside thermal load')
    _array(operands.weights_mm3, (gp,), 'backside weights')
    _require(operands.strain_maps.shape == (gp, 6, n)
             and operands.strain_maps.dtype == np.dtype('float64')
             and all(not value.flags.writeable for value in
                     (k, f, operands.strain_maps, operands.weights_mm3))
             and np.all(operands.weights_mm3 > 0) and np.all(np.diag(k) > 0),
             'readonly complete GP operands and positive weights/scales required')
    evidence = operands.evidence
    _require(evidence['GP_count'] == gp and evidence['GP_order_preserved'] is True
             and evidence['macro_coordinates'] == 6 and evidence['private_coordinates'] == n-6
             and evidence['reduced_coordinates'] == n and evidence['elastic_condensation'] is False,
             'backside coordinate/GP evidence differs')
    if layer_type is None:
        layer_type = _native_layer_type(coupon)

    def matrix(temperature):
        temperature = float(temperature)
        _require(math.isfinite(temperature) and 25. <= temperature <= 350.,
                 '25..350C backside reference required')
        return k

    def load(temperature):
        matrix(temperature)
        return (float(temperature)-25.)*f

    scale = 1./np.sqrt(np.diag(k))
    binding = operands.material_binding
    layer = layer_type(macro_size=6, bubble_size=n-6, reference_stiffness=matrix,
        reference_thermal_load=load, strain_maps=operands.strain_maps,
        weights_mm3=operands.weights_mm3, macro_coordinate_scale=scale[:6],
        bubble_coordinate_scale=scale[6:], cu_card=binding.cu_card,
        plastic_active=True, material_kernels=binding.material_kernels)
    bound_runtime.attach_corrected_material_binding(layer, binding)
    layer.backside_operand_binding = dict(evidence, material=material,
        material_binding_authenticated_here=True, actual_dataset_parity_validated=False,
        Cu_state_adoption=False, Cu_updates=0, operator_admission=False,
        assembly_execution_supported=False, production_ready=False,
        force_or_tangent_methods_changed=False, reference_K_or_B_corrected=False,
        component_catalog_registered=False, directMAPDL=False)
    if hasattr(layer_type, 'native_coupon_constructor_guard'):
        layer.backside_operand_binding['constructor_guard'] = dict(layer_type.native_coupon_constructor_guard)
    return layer
