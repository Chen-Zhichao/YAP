"""Checked operators for wafer thinning and prepared p2 assembly bodies.

The cap is equivalent to a chain of existing reduced pure-Si segments. This
does not confer continuum or process validation on the new assembly route.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import stat
from types import FunctionType, SimpleNamespace

import numpy as np

try:
    from YAP.mechanics_runtime import resolve_native_input, relocate_geometry_paths, P2_COMPACT_MANIFEST_SHA256
except ModuleNotFoundError:
    from mechanics_runtime import resolve_native_input, relocate_geometry_paths, P2_COMPACT_MANIFEST_SHA256


_SOURCE_KEYS = {"bi": "bi", "beol": "beol", "retained_si": "tsv_si_h37"}
_THICKNESS_MM = {"bi": .0015, "beol": .010, "retained_si": .037}
_GEOMETRY_UM = {
    "initial_total_thickness_um": 775., "target_total_thickness_um": 50.,
    "front_bi_thickness_um": 1.5, "back_bi_thickness_um": 1.5,
    "beol_thickness_um": 10.,
}
_SI = {"elastic_modulus_mpa": 91800., "poisson_ratio": .28,
       "secant_cte_per_c": 2.6e-6}
_CAP_THICKNESS_MM = .7265
_REACTION_TOLERANCE = 1e-7
_REFERENCE_P2_MANIFEST_SHA256 = P2_COMPACT_MANIFEST_SHA256
_PREPARED_TOTALS_UM = (30., 50., 80.)
_PREPARED_OPERATOR_CATALOG = Path(__file__).resolve().parents[1] / "prepared_operator_catalog.json"
_PREPARED_NATIVE64_BASIS_SHA256 = "8c2941e8358dba9626177cfd54c2187d4d0878d4ca1108d2652261efd318cb06"


def _import_mechanics_module(name):
    # Request validation must not select or import an installed native runtime.
    try:
        from YAP.history_mechanics_backend import _import_mechanics_module as load
    except ModuleNotFoundError:
        from history_mechanics_backend import _import_mechanics_module as load
    return load(name)


def _check_geometry(geometry, *, target_total_um=50., initial_total_um=775.):
    if not isinstance(geometry, Mapping):
        raise TypeError("geometry must be the assembly-plan geometry mapping")
    for key, expected in _GEOMETRY_UM.items():
        if key == "target_total_thickness_um":
            expected = target_total_um
        elif key == "initial_total_thickness_um":
            expected = initial_total_um
        value = geometry.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (
            not np.isfinite(value) or not np.isclose(value, expected, rtol=0., atol=1e-9)
        ):
            raise ValueError(f"{key}: this library requires {expected:g} um")
    diameter = geometry.get("diameter_mm")
    if isinstance(diameter, bool) or not isinstance(diameter, (int, float)) or (
        not np.isfinite(diameter) or diameter <= 0.
    ):
        raise ValueError("diameter_mm must be finite and positive")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_path(base_directory, value):
    return resolve_native_input(base_directory, value)


def _record_input(path, inputs, expected=None):
    observed = _sha256(path)
    if expected is not None and observed != expected:
        raise ValueError(f"input SHA256 mismatch: {path}")
    inputs[str(path)] = observed
    return observed


def _check_library(library, thicknesses=None):
    rank = library.port_rank_per_face
    scale = np.r_[library.membrane_bending_coordinate_scale,
                  library.port_coordinate_scale, library.port_coordinate_scale]
    if scale.shape != (6 + 2 * rank,) or not np.isfinite(scale).all() or np.any(scale <= 0.):
        raise ValueError("invalid native coordinate scales")
    if not np.isfinite(library.cell_area_mm2) or library.cell_area_mm2 <= 0.:
        raise ValueError("invalid native cell area")
    for key, thickness in (thicknesses or _THICKNESS_MM).items():
        if not np.isclose(library.thickness_mm[key], thickness, rtol=0., atol=1e-12):
            raise ValueError(f"{key}: native dataset thickness mismatch")
        op = library.operators[key]
        if op.macro_size != len(scale) or not np.allclose(
            op.macro_coordinate_scale, scale, rtol=0., atol=1e-14
        ):
            raise ValueError(f"{key}: native port coordinates mismatch")


def _condense_checked(stiffness, load, keep, scale):
    schur = _import_mechanics_module("exp12_w2w775_preview").schur_condense
    k, f = np.asarray(stiffness, dtype=float), np.asarray(load, dtype=float)
    keep, scale = np.asarray(keep, dtype=int), np.asarray(scale, dtype=float)
    if not np.isfinite(k).all() or not np.isfinite(f).all():
        raise ValueError("nonfinite cap stiffness or thermal load")
    if not np.allclose(k, k.T, rtol=1e-10, atol=1e-12):
        raise ValueError("cap stiffness is not symmetric")
    kc, fc, extension, thermal = schur(k, f, keep, scale)
    internal = np.setdiff1d(np.arange(len(k)), keep)
    rng = np.random.default_rng(775)
    maximum_reaction = maximum_internal = 0.
    for delta_t in (0., 75., 325.):
        for _ in range(3):
            q = rng.normal(size=len(keep)) * scale[keep] * .0001
            expanded = extension @ q + thermal * delta_t
            force = k @ expanded - f * delta_t
            expected = kc @ q - fc * delta_t
            denominator = max(float(np.linalg.norm(expected)), 1e-12)
            maximum_reaction = max(maximum_reaction,
                float(np.linalg.norm(force[keep] - expected)) / denominator)
            maximum_internal = max(maximum_internal,
                float(np.linalg.norm(force[internal])) / denominator)
    if not np.isfinite([maximum_reaction, maximum_internal]).all() or max(
        maximum_reaction, maximum_internal
    ) > _REACTION_TOLERANCE:
        raise ValueError("cap condensation reaction/internal-equilibrium gate failed")
    if not np.isfinite(kc).all() or not np.isfinite(fc).all():
        raise ValueError("nonfinite condensed cap")
    return kc, fc, {
        "maximum_reaction_relative_defect": maximum_reaction,
        "maximum_internal_equilibrium_relative_defect": maximum_internal,
    }


def _linear_operator(stiffness, load_per_c, scale, cu_card):
    layer = _import_mechanics_module("exp12_all_gauss_reduced_fe2").AllGaussReducedFE2Layer
    size = len(scale)
    return layer(
        macro_size=size, bubble_size=0,
        reference_stiffness=lambda _temperature: stiffness,
        reference_thermal_load=lambda temperature: (float(temperature) - 25.) * load_per_c,
        strain_maps=np.zeros((0, 6, size)), weights_mm3=np.empty(0),
        macro_coordinate_scale=np.asarray(scale).copy(), bubble_coordinate_scale=np.empty(0),
        cu_card=cu_card, plastic_active=False,
    )


def _condense_segment(operator):
    if operator.cu_point_count or operator.plastic_active:
        raise ValueError("pure-Si cap segment must contain no Cu history")
    k = operator.reference_stiffness(25.)
    f = operator.reference_thermal_load(26.)
    if not np.allclose(operator.reference_thermal_load(25.), 0., rtol=0., atol=1e-12):
        raise ValueError("pure-Si reference temperature must be 25 C")
    if not np.allclose(operator.reference_stiffness(350.), k, rtol=1e-12, atol=1e-12) or (
        not np.allclose(operator.reference_thermal_load(350.), 325. * f,
                        rtol=1e-12, atol=1e-12)
    ):
        raise ValueError("pure-Si cap requires constant Si stiffness and CTE")
    scale = np.r_[operator.macro_coordinate_scale, operator.bubble_coordinate_scale]
    kc, fc, audit = _condense_checked(k, f, np.arange(operator.macro_size), scale)
    return _linear_operator(kc, fc, operator.macro_coordinate_scale, operator.cu_card), audit


def _column(library, segments, *, orientations=None):
    native = _import_mechanics_module("exp12_all_gauss_biaxial_column")
    orientations = (1,) * len(segments) if orientations is None else tuple(orientations)
    if len(orientations) != len(segments):
        raise ValueError("each cap segment needs a thickness orientation")
    z = -.5 * sum(thickness for _, thickness in segments)
    modules = []
    for index, (operator, thickness) in enumerate(segments):
        modules.append(native.AllGaussBiaxialPortModule(
            name=f"si_segment_{index}", operator=operator,
            center_z_mm=z + .5 * thickness, thickness_orientation=orientations[index],
        ))
        z += thickness
    return native.build_all_gauss_biaxial_port_column(
        modules, library.reflection_matrix,
        membrane_bending_coordinate_scale=library.membrane_bending_coordinate_scale,
        port_coordinate_scale=library.port_coordinate_scale,
        rigid_port_mode_indices=library.rigid_port_mode_indices,
        common_port_gauge_vectors=library.common_port_gauge_vectors,
    )


def _join_segments(library, left, right):
    topology = _column(library, (left, right))
    size = topology.generalized_size
    k, f = np.zeros((size, size)), np.zeros(size)
    for instance in topology.instances:
        transform, op = instance.macro_transform, instance.operator
        k += transform.T @ op.reference_stiffness(25.) @ transform
        f += transform.T @ op.reference_thermal_load(26.)
    first, last = topology.interface_slices[0], topology.interface_slices[-1]
    keep = np.r_[np.arange(6), np.arange(first.start, first.stop),
                 np.arange(last.start, last.stop)]
    # External rigid modes remain free; only the shared internal face is removed.
    kc, fc, audit = _condense_checked(k, f, keep, topology.coordinate_scale)
    operator = _linear_operator(kc, fc, topology.coordinate_scale[keep], left[0].cu_card)
    return (operator, left[1] + right[1]), audit


def _oriented_segment(library, operator, thickness, orientation):
    if orientation == 1:
        return operator
    topology = _column(library, ((operator, thickness),), orientations=(orientation,))
    transform = topology.instances[0].macro_transform
    return _linear_operator(
        transform.T @ operator.reference_stiffness(25.) @ transform,
        transform.T @ operator.reference_thermal_load(26.),
        topology.coordinate_scale, operator.cu_card,
    )


def _reduce_recipe(library, segments, recipe, orientations):
    oriented = {}
    level = []
    for key, orientation in zip(recipe, orientations, strict=True):
        if (key, orientation) not in oriented:
            oriented[key, orientation] = _oriented_segment(
                library, segments[key], library.thickness_mm[key], orientation)
        level.append((oriented[key, orientation], library.thickness_mm[key]))
    audits = []
    pairs = {}
    while len(level) > 1:
        following = []
        for index in range(0, len(level) - 1, 2):
            left, right = level[index], level[index + 1]
            key = (id(left[0]), left[1], id(right[0]), right[1])
            if key not in pairs:
                pairs[key] = _join_segments(library, left, right)
            merged, audit = pairs[key]
            following.append(merged)
            audits.append(audit)
        if len(level) % 2:
            following.append(level[-1])
        level = following
    operator, thickness = level[0]
    if not np.isclose(thickness, _CAP_THICKNESS_MM, rtol=0., atol=1e-13):
        raise ValueError("pure-Si cap thickness must be 726.5 um")
    return operator, {
        "method": "pairwise_ungauged_segment_and_bubble_static_condensation",
        "segment_recipe": recipe, "segment_count": len(recipe),
        "segment_orientations": list(orientations),
        "unique_pair_condensations": len(pairs),
        "thickness_um": thickness * 1000., "cu_point_count": operator.cu_point_count,
        "maximum_pair_generalized_size": 6 + 3 * library.port_rank_per_face,
        "condensed_macro_port_dofs": operator.macro_size,
        **{key: max(audit[key] for audit in audits) for key in audits[0]},
    }


def _build_cap(library, segments):
    coarse_recipe = ["retained_si"] * 19 + ["beol"] + ["bi"] * 9
    coarse, coarse_audits = _reduce_recipe(
        library, segments, coarse_recipe, [1] * len(coarse_recipe))
    coarse_audits.update(_abd_audit(library, coarse, _CAP_THICKNESS_MM))
    coarse_audits["analytical_abd_gate_passed"] = (
        coarse_audits["maximum_scaled_abd_defect"] <= .01)
    # These are homogeneous replacements on heterogeneous source bases.
    # Refine bending-dominant outer faces and mirror the native bases;
    # preserve the rejected coarse audit, with no rescaling or added gauges.
    half = ["bi"] * 69 + ["retained_si"] * 7
    recipe = half + ["bi"] + list(reversed(half))
    orientations = [-1] * len(half) + [1] * (len(half) + 1)
    cap, audits = _reduce_recipe(library, segments, recipe, orientations)
    audits["coarse_cap_audits"] = coarse_audits
    audits["refinement"] = "symmetric_thin_pure_Si_outer_faces_14x37_plus_139x1p5_um"
    fine_half = ["bi"] * 143 + ["retained_si"] * 4
    fine_recipe = fine_half + ["bi"] + list(reversed(fine_half))
    fine, fine_audit = _reduce_recipe(library, segments, fine_recipe,
                                    [-1] * len(fine_half) + [1] * (len(fine_half) + 1))
    fine_audit.update(_abd_audit(library, fine, _CAP_THICKNESS_MM))
    if fine_audit["maximum_scaled_abd_defect"] > .01:
        raise ValueError(
            "further-refined cap fails the unchanged analytical ABD gate: "
            f"maximum normalized defect={fine_audit['maximum_scaled_abd_defect']:.9g}, "
            f"limit=0.01, blocks={fine_audit['maximum_scaled_abd_defect_by_block']}; "
            f"coarse defect={coarse_audits['maximum_scaled_abd_defect']:.9g}. "
            "This rejects the homogeneous replacement cap, not the source Cu-layer operators.")
    scale = cap.macro_coordinate_scale
    kc = scale[:, None] * cap.reference_stiffness(25.) * scale[None, :]
    kf = scale[:, None] * fine.reference_stiffness(25.) * scale[None, :]
    fc = scale * cap.reference_thermal_load(350.)
    ff = scale * fine.reference_thermal_load(350.)
    fine_audit["scaled_stiffness_change_from_selected"] = float(np.linalg.norm(kf - kc) / np.linalg.norm(kf))
    fine_audit["scaled_thermal_load_change_from_selected"] = float(np.linalg.norm(ff - fc) / np.linalg.norm(ff))
    audits["further_refinement_audits"] = fine_audit
    return cap, audits


def _abd_audit(library, cap, thickness_mm):
    preview = _import_mechanics_module("exp12_prebond_export")
    topology = _column(library, ((cap, thickness_mm),))
    abd = preview.effective_reference_abd(topology, library.cell_area_mm2)
    e, nu, h = _SI["elastic_modulus_mpa"], _SI["poisson_ratio"], thickness_mm
    q = e / (1. - nu ** 2) * np.array([[1., nu, 0.], [nu, 1., 0.],
                                      [0., 0., (1. - nu) / 2.]])
    exact = np.zeros((6, 6))
    exact[:3, :3], exact[3:, 3:] = q * h, q * h ** 3 / 12.
    normalized = (abd - exact) / np.sqrt(np.outer(np.diag(exact), np.diag(exact)))
    if not np.isfinite(normalized).all():
        raise ValueError("nonfinite cap analytical ABD audit")
    return {
        "abd_per_width": abd.tolist(), "analytical_abd": exact.tolist(),
        "normalized_abd_defect": normalized.tolist(),
        "maximum_scaled_abd_defect": float(np.max(np.abs(normalized))),
        "maximum_scaled_abd_defect_by_block": {
            "A": float(np.max(np.abs(normalized[:3, :3]))),
            "B": float(np.max(np.abs(normalized[:3, 3:]))),
            "D": float(np.max(np.abs(normalized[3:, 3:]))),
        },
        "maximum_scaled_abd_defect_index": list(map(int,
            np.unravel_index(np.argmax(np.abs(normalized)), normalized.shape))),
    }


def _audit_cap(library, cap):
    native = _import_mechanics_module("exp12_all_gauss_biaxial_column")
    runner = _import_mechanics_module("run_exp12_v2_production_sequential_all_gauss_v1")
    topology = _column(library, ((cap, _CAP_THICKNESS_MM),))
    audit = _abd_audit(library, cap, _CAP_THICKNESS_MM)
    abd_defect = audit["maximum_scaled_abd_defect"]
    _, free = runner._free_equilibrium_trial(
        topology, native.initial_biaxial_column_state(topology), temperature_c=350.,
        delta_time_s=0., elapsed_time_s=0., plastic_active=False,
        solver_options=dict(linear_solver="assembled_dense", tolerance=1e-10, max_iterations=20),
    )
    expected = np.array([_SI["secant_cte_per_c"] * 325.] * 2 + [0.] * 4)
    actual = np.asarray(free.generalized_coordinates[:6])
    thermal_defect = float(np.max(np.abs(actual - expected)))
    if not np.isfinite([abd_defect, thermal_defect]).all() or (
        abd_defect > .01 or thermal_defect > 1e-8
    ):
        raise ValueError(
            f"cap analytical ABD/thermal gate failed: {abd_defect}, {thermal_defect}; "
            f"ABD blocks={audit['maximum_scaled_abd_defect_by_block']}")
    return {
        **audit,
        "free_thermal_q6": actual.tolist(), "thermal_q6_max_absolute_defect": thermal_defect,
    }


def load_assembly_library(package_path, geometry, *, frozen_preview=False):
    """Load checked BI6_50/120 or BI2_50/64 inputs and construct the Si cap.

    Return ``(Q6SequentialLibrary, provenance)`` with operator keys ``bi``,
    ``beol``, ``retained_si``, and ``bulk_si``. Inputs are read-only; no package,
    mesh, cache, or generated operator is written by this loader. Prepared
    D2W candidates use the separate die loader; no-thinning W2W is retired.
    """
    if type(frozen_preview) is not bool:
        raise ValueError("frozen_preview must be Boolean")
    if geometry.get("initial_total_thickness_um") == geometry.get("target_total_thickness_um"):
        check_prepared_wafer_geometry(geometry)
    if frozen_preview:
        return _load_frozen_preview_library(package_path, geometry)
    _check_geometry(geometry)
    path = _input_path(Path.cwd(), package_path)
    raw = path.read_bytes()
    package = json.loads(raw)
    ranks = {"BI6_50": 120, "BI2_50": 64}
    rank = package.get("rank")
    if (package.get("schema") != "exp12.ofat.geometry_package.v1"
            or type(rank) is not int or ranks.get(package.get("name")) != rank
            or package.get("ready") is not True
            or package.get("state") != "READY_ENGINEERING_PREVIEW"):
        raise ValueError("requires READY_ENGINEERING_PREVIEW native120 BI6_50 or native64 BI2_50 package")
    inputs = {str(path): hashlib.sha256(raw).hexdigest()}
    datasets, prepared = {}, {}
    for public, source in _SOURCE_KEYS.items():
        try:
            datasets[public] = _input_path(path.parent, package["datasets"][source])
            expected = package["dataset_sha256"][source]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"missing dataset/hash for {source}") from exc
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"missing dataset SHA256 for {source}")
        _record_input(datasets[public], inputs, expected)
        if source in package.get("prepared", {}):
            prepared[public] = _input_path(path.parent, package["prepared"][source])
            _record_input(prepared[public], inputs)
    basis = _input_path(path.parent, package["port_basis"])
    expected_basis = package.get("port_basis_sha256")
    if not isinstance(expected_basis, str) or len(expected_basis) != 64:
        raise ValueError("missing port-basis SHA256")
    _record_input(basis, inputs, expected_basis)
    runtime = _import_mechanics_module("exp12_mindlin_sequential_runtime")
    library = runtime.load_q6_sequential_library(
        datasets, basis, validate=True, verify_content=True, port_rank_per_face=rank,
        prepared_geometry_paths=prepared,
        frozen_expected_port_basis_sha256_by_dataset={key: expected_basis for key in datasets},
    )
    if library.port_rank_per_face != rank:
        raise ValueError(f"assembly library must retain native{rank} ports")
    _check_library(library)
    dataset_type = _import_mechanics_module("exp12_layer_rom_dataset").LayerROMDataset
    pure = _import_mechanics_module("exp12_all_gauss_linear_layer")
    segments, segment_audits = {}, {}
    for key, dataset_path in datasets.items():
        dataset = dataset_type.load(dataset_path, validate=True, verify_content=True)
        constants = dataset.reference_constant_materials
        if constants is None or "si" not in constants or asdict(constants["si"]) != _SI:
            raise ValueError("pure-Si cap requires explicit V2 Si constants; no legacy fallback")
        if dataset.reference_temperature_c != 25.:
            raise ValueError("pure-Si cap requires a 25 C reference temperature")
        if dataset.macro_size != 6 + 2 * rank or not np.allclose(
            dataset.macro_coordinate_scale, library.operators[key].macro_coordinate_scale,
            rtol=0., atol=1e-14,
        ):
            raise ValueError("pure-Si segment port coordinates mismatch")
        segments[key], segment_audits[key] = _condense_segment(
            pure.pure_constant_material_operator_from_dataset(dataset, material_name="si")
        )
        segment_audits[key].update(_abd_audit(
            library, segments[key], library.thickness_mm[key]))
        del dataset
    cap, cap_audits = _build_cap(library, segments)
    cap_audits.update(_audit_cap(library, cap))
    cap_audits["segment_audits"] = segment_audits
    cap_audits["constants"] = dict(_SI)
    cap_audits["approximation_scope"] = "equivalent_to_reduced_FE_segments_not_new_continuum_validation"
    library = replace(library, operators={**library.operators, "bulk_si": cap},
                      thickness_mm={**library.thickness_mm, "bulk_si": _CAP_THICKNESS_MM})
    return library, {
        "schema": "yap.assembly_operator_library.v1", "package_path": str(path),
        "input_sha256": inputs, "port_basis_sha256": expected_basis,
        "port_rank_per_face": library.port_rank_per_face,
        "source_operator_keys": dict(_SOURCE_KEYS), "cap_audits": cap_audits,
        "physics_qualification": "ENGINEERING_PREVIEW_NOT_ASSEMBLY_ROUTE_VALIDATED",
    }


def load_f2b_assembly_library(package_path, geometry, *, frozen_preview=False):
    """Load the P24 preview with both original pure-Si cap slices at birth."""
    if isinstance(package_path, str) and package_path.startswith("component:"):
        from component_operators import load_w2w_component_library
        return load_w2w_component_library(package_path, geometry, frozen_preview=frozen_preview)
    _check_geometry(geometry)
    if frozen_preview is not True or Path(package_path).name != "BI2_50.json":
        raise ValueError("F2B split-cap execution currently requires the explicit frozen BI2_50/P24 "
                         "preview; a compatible public native64 cap is not yet admitted")
    return _load_reference_p2_preview_library(package_path, geometry, split_cap=True)


def _check_reference_bulk_audits(audits, *, thickness_um=725.):
    def bounded(value, limit, label):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) \
                or not 0 <= value <= limit:
            raise ValueError(f"reference cap {label} gate failed: {value}")
    bounded(audits["direct_small_stack_relative_defect"], 1e-8, "direct condensation")
    if not audits["block_audits"]:
        raise ValueError("reference cap requires condensation audits")
    for block in audits["block_audits"]:
        bounded(block["reaction_relative_defect"], 1e-7, "reaction")
    if len(audits["bulk_audits"]) != 1 or audits["bulk_audits"][0]["thickness_um"] != thickness_um:
        raise ValueError(f"reference tree adapter requires exactly the {thickness_um:g} um cap audit")
    cap = audits["bulk_audits"][0]
    bounded(cap["abd_scaled_defect"], .05, "full ABD preview")
    bounded(cap["relaxed_normal_scaled_defect"], .01, "relaxed normal")
    bounded(cap["free_thermal_q6_max_absolute_defect"], 1e-8, "thermal")
    for name in ("abd_per_width", "analytical_abd"):
        a = np.asarray(cap[name], dtype=float)
        if a.shape != (6, 6) or not np.isfinite(a).all():
            raise ValueError("reference cap needs finite full ABD matrices")
    if cap["original_full_tensor_one_percent_pass"] != (cap["abd_scaled_defect"] <= .01) \
            or cap["full_tensor_physical_admission"] is not False or cap["shear_terms_modified"] is not False:
        raise ValueError("reference cap qualification labels are inconsistent")


def _load_frozen_preview_library(package_path, geometry):
    """Reuse the frozen reference p6/P32 preview, not a general candidate escape hatch."""
    if Path(package_path).name == "BI2_50.json":
        return _load_reference_p2_preview_library(package_path, geometry)
    _check_geometry(geometry, target_total_um=51.5)
    dataset_module = _import_mechanics_module("exp12_layer_rom_dataset")
    root = Path(dataset_module.__file__).resolve().parent
    freeze_path = _input_path(root, "resource:w2w_corrected/h50_nominal/INPUT_FREEZE.json")
    freeze = json.loads(freeze_path.read_text())
    frozen = freeze["package"]
    sources = frozen["inputs_sha256"]
    known = [digest for name, digest in sources.items()
             if name.endswith("/bi6_p32_b16/packages/BI6_50.json")]
    if len(known) != 1:
        raise ValueError("reference freeze does not identify one p6/P32 manifest")
    path = _input_path(Path.cwd(), package_path)
    inputs = {}
    _record_input(path, inputs, known[0])
    _record_input(freeze_path, inputs)
    package = json.loads(path.read_text())
    if (package.get("schema") != "exp12.ofat.geometry_package.v1" or package.get("name") != "BI6_50"
            or type(package.get("rank")) is not int or package["rank"] != 32
            or package.get("ready") is not False or package.get("state") != "CANDIDATE_NOT_VALIDATED"):
        raise ValueError("reference preview requires the frozen candidate BI6_50/P32 manifest")
    datasets = {key: _input_path(path.parent, package["datasets"][key])
                for key in ("bi", "beol", "tsv_si_h38p5")}
    for key, filename in datasets.items():
        expected = frozen["dataset_sha256"][key]
        if package["dataset_sha256"][key] != expected:
            raise ValueError("compact layer differs from reference freeze")
        _record_input(filename, inputs, expected)
    basis = _input_path(path.parent, package["port_basis"])
    basis_hash = frozen["port_basis_sha256"]
    if package["port_basis_sha256"] != basis_hash:
        raise ValueError("compact basis differs from reference freeze")
    _record_input(basis, inputs, basis_hash)
    api = _import_mechanics_module("exp12_w2w_window_20260914")
    if api.FAMILY != "bi6":
        raise ValueError("reference p6 preview requires an unmodified bi6 factory environment")
    _record_input(api.NATIVE, inputs, package["parent_package_sha256"])
    parent = json.loads(Path(api.NATIVE).read_text())
    _record_input(_input_path(Path(api.NATIVE).parent, parent["port_basis"]), inputs, parent["port_basis_sha256"])
    for key in ("bi", "beol"):
        _record_input(_input_path(Path(api.NATIVE).parent, parent["datasets"][key]), inputs, parent["dataset_sha256"][key])
    runtime = _import_mechanics_module("exp12_mindlin_sequential_runtime")
    library = runtime.load_q6_sequential_library(datasets, basis, port_rank_per_face=32,
        validate=True, verify_content=True, factorize_elements=True,
        frozen_expected_port_basis_sha256_by_dataset={key: basis_hash for key in datasets})
    if library.port_rank_per_face != 32 or library.common_port_gauge_vectors is not None:
        raise ValueError("reference library needs P32 ports and the existing rigid-mode gauge contract")
    # Legacy factory selector 60 requests only {725, 785-60} = {725} um.
    # It is NOT the process target, which is 51.5 um total in this tree experiment.
    factory = _bound_reference_factory(api, library, package, api.NATIVE)
    expanded, audits = factory.augment(60)
    _check_reference_bulk_audits(audits)
    for filename, expected in audits["inputs_sha256"].items():
        _record_input(_input_path(root, filename), inputs, expected)
    aliases = dict(bi="bi", beol="beol", retained_si="tsv_si_h38p5", bulk_si="bulk_725")
    library = replace(expanded,
        operators={key: expanded.operators[source] for key, source in aliases.items()},
        thickness_mm={key: expanded.thickness_mm[source] for key, source in aliases.items()})
    _check_library(library, dict(bi=.0015, beol=.010, retained_si=.0385, bulk_si=.725))
    if library.operators["bulk_si"].cu_point_count:
        raise ValueError("reference cap must not introduce Cu points")
    return library, dict(schema="yap.assembly_operator_library.v1", package_path=str(path),
        input_sha256=inputs, port_basis_sha256=basis_hash, port_rank_per_face=32,
        source_operator_keys=aliases, cap_audits=audits, frozen_preview=True,
        original_package_state=package["state"], original_package_ready=package["ready"],
        reference_freeze=str(freeze_path),
        implementation_sha256={str(Path(m.__file__).resolve()): _sha256(m.__file__)
                               for m in (api, runtime)},
        physics_qualification="REFERENCE_SCOPED_ENGINEERING_PREVIEW_NEW_TREE_PROCESS_NOT_VALIDATED")


def _bound_reference_factory(api, library, package, native_path):
    # Relocate only this constructor's decoded paths; frozen files stay unchanged.
    constructor = api.SiliconBulkFactory.__init__
    native_path = Path(native_path)
    def decode(payload, *args, **kwargs):
        return relocate_geometry_paths(json.loads(payload, *args, **kwargs), native_path.parent)
    namespace = {**constructor.__globals__, "NATIVE": native_path,
                 "json": SimpleNamespace(loads=decode)}
    initialize = FunctionType(constructor.__code__, namespace, constructor.__name__,
                              constructor.__defaults__, constructor.__closure__)
    initialize.__kwdefaults__ = constructor.__kwdefaults__
    factory = api.SiliconBulkFactory.__new__(api.SiliconBulkFactory)
    initialize(factory, library, relocate_geometry_paths(package, native_path.parent))
    return factory


def _reference_cap_audit(api, library, operator, thickness_um):
    topology, _, _, _ = api.assemble_linear(library, [(operator, thickness_um / 1000.)])
    export = _import_mechanics_module("exp12_prebond_export")
    abd = export.effective_reference_abd(topology, library.cell_area_mm2)
    e, nu, h = _SI["elastic_modulus_mpa"], _SI["poisson_ratio"], thickness_um / 1000.
    q = e / (1 - nu * nu) * np.array([[1., nu, 0.], [nu, 1., 0.], [0., 0., (1 - nu) / 2]])
    analytical = np.zeros((6, 6))
    analytical[:3, :3], analytical[3:, 3:] = h * q, h**3 * q / 12.
    scaled = (abd - analytical) / np.sqrt(np.outer(analytical.diagonal(), analytical.diagonal()))
    normal, shear = np.array([0, 1, 3, 4]), np.array([2, 5])
    relaxed = abd[np.ix_(normal, normal)] - abd[np.ix_(normal, shear)] @ np.linalg.solve(
        abd[np.ix_(shear, shear)], abd[np.ix_(shear, normal)])
    ideal_normal = analytical[np.ix_(normal, normal)]
    normal_defect = float(np.max(np.abs((relaxed - ideal_normal) / np.sqrt(
        np.outer(ideal_normal.diagonal(), ideal_normal.diagonal())))))
    state_api = _import_mechanics_module("exp12_all_gauss_biaxial_column")
    production = _import_mechanics_module("run_exp12_v2_production_sequential_all_gauss_v1")
    trial, state = production._free_equilibrium_trial(topology,
        state_api.initial_biaxial_column_state(topology), temperature_c=350., delta_time_s=0.,
        elapsed_time_s=0., plastic_active=False,
        solver_options=dict(linear_solver="assembled_dense", tolerance=1e-10, max_iterations=20))
    ideal_q6 = np.array([_SI["secant_cte_per_c"] * 325.] * 2 + [0.] * 4)
    thermal = state.generalized_coordinates[:6] - ideal_q6
    full_defect = float(np.max(np.abs(scaled)))
    return dict(thickness_um=thickness_um, abd_scaled_defect=full_defect,
        relaxed_normal_scaled_defect=normal_defect, abd_per_width=abd.tolist(),
        analytical_abd=analytical.tolist(),
        free_thermal_q6_max_absolute_defect=float(np.max(np.abs(thermal))),
        free_thermal_q6_defect=thermal.tolist(),
        free_thermal_q6_units=["strain"] * 3 + ["1/mm"] * 3,
        free_thermal_scaled_equilibrium_residual=float(trial.scaled_residual_norm),
        original_full_tensor_one_percent_pass=full_defect <= .01,
        revised_scope="normal_relaxed_1percent_full_tensor_5percent_engineering_preview_only",
        shear_terms_modified=False, full_tensor_physical_admission=False)


def check_prepared_wafer_geometry(geometry):
    """Reject the retired no-thinning W2W path before accessing operators."""
    raise ValueError("prepared no-thinning W2W is retired; equal initial/target wafer totals are unsupported")


def _prepared_total(geometry, key):
    value = geometry.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (
        not np.isfinite(value) or value not in _PREPARED_TOTALS_UM
    ):
        raise ValueError(f"{key}: prepared requests support total thicknesses 30, 50, 80 um")
    return float(value)


def check_prepared_die_geometry(geometry):
    """Validate 30/50/80-um requests without admitting unavailable datasets."""
    expected = dict(front_bi_thickness_um=1.5, back_bi_thickness_um=1.5, beol_thickness_um=10.)
    if not isinstance(geometry, Mapping) or set(geometry) != set(expected) | {
        "total_thickness_um", "width_mm", "length_mm"
    }:
        raise ValueError("invalid prepared die geometry")
    _prepared_total(geometry, "total_thickness_um")
    for key, value in expected.items():
        if isinstance(geometry[key], bool) or not isinstance(geometry[key], (int, float)) or (
            not np.isfinite(geometry[key]) or not np.isclose(geometry[key], value, rtol=0., atol=1e-9)
        ):
            raise ValueError(f"prepared die library requires {key}={value:g}")
    width, length = geometry["width_mm"], geometry["length_mm"]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
           for v in (width, length)) or width <= 0 or width != length:
        raise ValueError("the existing scalar reference die metric requires a square footprint")


def prepared_operator_request(package_path, geometry, *, frozen_preview=False, layout=None):
    """Describe a request without resolving paths or claiming installed data."""
    if isinstance(package_path, str) and package_path.startswith("component:"):
        from component_operators import prepared_operator_request as component_request
        return component_request(package_path, geometry, frozen_preview=frozen_preview, layout=layout)
    if not isinstance(geometry, Mapping):
        raise ValueError("prepared-body geometry must be a mapping")
    die = "total_thickness_um" in geometry
    (check_prepared_die_geometry if die else check_prepared_wafer_geometry)(geometry)
    total = geometry["total_thickness_um" if die else "target_total_thickness_um"]
    if type(frozen_preview) is not bool:
        raise ValueError("frozen_preview must be Boolean")
    expected = f"BI2_{total:g}.json"
    if Path(package_path).name != expected:
        raise ValueError(f"prepared p2/d1 TOTAL{total:g} request requires {expected}; no geometry substitution")
    if layout is not None and tuple(layout) != (2., 2., 1.):
        raise ValueError("prepared-body mechanics and layout require matching p2/d1 geometry")
    if total != 50. and frozen_preview:
        raise ValueError("frozen_preview identifies only the existing TOTAL50 BI2/P24 package")
    if total == 50. and not frozen_preview:
        raise ValueError("prepared-body execution requires the identified reference BI2/P24 preview")
    status = "IDENTIFIED_TOTAL50_PREVIEW_INPUTS_NOT_VERIFIED"
    if total != 50.:
        catalog, _ = _prepared_catalog()
        status = ("IDENTIFIED_NATIVE64_CANDIDATE_INPUTS_NOT_VERIFIED" if catalog["packages"]
                  else "UNAVAILABLE_EXACT_P2_OPERATORS_NOT_ADMITTED")
    return dict(operator_package=str(package_path), total_thickness_um=float(total),
        retained_si_thickness_um=float(total - sum(geometry[key] for key in
            ("front_bi_thickness_um", "back_bi_thickness_um", "beol_thickness_um"))),
        bonding_pitch_um=2., pad_diameter_um=1.,
        admission_status=status,
        operator_inputs_verified=False, physical_readiness=False)


def require_prepared_operator(package_path, geometry, *, frozen_preview=False):
    """Fail before native imports for thicknesses lacking audited p2 inputs."""
    request = prepared_operator_request(package_path, geometry, frozen_preview=frozen_preview)
    if isinstance(package_path, str) and package_path.startswith("component:"):
        from component_operators import configure_component_runtime
        configure_component_runtime()
        return
    if request["total_thickness_um"] != 50.:
        if request["admission_status"] == "UNAVAILABLE_EXACT_P2_OPERATORS_NOT_ADMITTED":
            raise RuntimeError(
                f"Exact p2/d1 prepared-body operators for TOTAL{request['total_thickness_um']:g} "
                f"(retained Si {request['retained_si_thickness_um']:g} um) are unavailable or not admitted; "
                "planning only. Build and audit the exact p2 operators; p6 substitution and h37 scaling are forbidden.")
        _prepared_native64_inputs(package_path, request["total_thickness_um"])


def load_prepared_die_library(package_path, geometry, *, frozen_preview=False):
    """Load admitted prepared p2 layers, without a removable substrate."""
    if isinstance(package_path, str) and package_path.startswith("component:"):
        from component_operators import load_prepared_component_library
        return load_prepared_component_library(package_path, geometry, frozen_preview=frozen_preview)
    require_prepared_operator(package_path, geometry, frozen_preview=frozen_preview)
    if geometry["total_thickness_um"] != 50.:
        return _load_prepared_native64_library(package_path, geometry["total_thickness_um"])
    width = geometry["width_mm"]
    adapter = {**_GEOMETRY_UM, "diameter_mm": width, "initial_total_thickness_um": 50.}
    return _load_reference_p2_preview_library(package_path, adapter, prepared_bodies=True)


def _prepared_sha(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"invalid prepared {label} SHA256")
    return value


def _prepared_number(value, label, *, expected=None, positive=False, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"invalid prepared {label}")
    if value < 0 or (positive and value <= 0) or (maximum is not None and value > maximum) \
            or (expected is not None and value != expected):
        raise ValueError(f"prepared {label} gate failed")
    return value


def _prepared_integer(value, label, *, expected=None, positive=False):
    if type(value) is not int:
        raise ValueError(f"invalid prepared {label}")
    return _prepared_number(value, label, expected=expected, positive=positive)


def _prepared_transfer_space(audit, label):
    if not isinstance(audit, dict):
        raise ValueError(f"prepared receipt requires an independent {label} audit")
    rank = _prepared_integer(audit.get("rank"), f"{label} rank", positive=True)
    _prepared_integer(audit.get("observed_rank"), f"{label} observed rank", expected=rank)
    low = _prepared_number(audit.get("singular_value_min"), f"{label} singular_value_min", positive=True)
    high = _prepared_number(audit.get("singular_value_max"), f"{label} singular_value_max", positive=True)
    if not 1e-12 < low / high <= 1.:
        raise ValueError(f"{label} numerical rank gate failed")
    for field, maximum in dict(gram_defect=1e-9, face_trace=1e-9, relative_periodic_jump=1e-4).items():
        _prepared_number(audit.get(field), f"{label} {field}", maximum=maximum)
    return rank


def _prepared_json(path, inputs, expected=None, *, limit=262144):
    path = Path(path)
    if {"#snapshot", ".snapshot"}.intersection(path.parts + path.resolve().parts) \
            or any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("prepared JSON must not use snapshots or symlinks")
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("prepared JSON must be a regular file")
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("prepared JSON exceeds size limit")
    observed = hashlib.sha256(payload).hexdigest()
    if expected is not None and observed != expected:
        raise ValueError(f"input SHA256 mismatch: {path}")
    inputs[str(path)] = observed
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate prepared JSON key")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError(f"nonfinite prepared JSON constant: {value}")
    value = json.loads(payload, object_pairs_hook=unique, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("prepared JSON must be an object")
    return value


def _prepared_catalog():
    inputs = {}
    value = _prepared_json(_PREPARED_OPERATOR_CATALOG, inputs, limit=65536)
    if set(value) != {"schema", "packages"} or value["schema"] != "yap.prepared_p2_operator_catalog.v1" \
            or not isinstance(value["packages"], dict) or set(value["packages"]) not in (set(), {"30", "80"}):
        raise ValueError("prepared catalog requires both TOTAL30/80 admissions or neither")
    fields = {"package_sha256", "audit_sha256", "parent_package_sha256", "port_basis_sha256"}
    for entry in value["packages"].values():
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError("invalid prepared catalog entry")
        for field in fields:
            _prepared_sha(entry[field], field)
        if entry["port_basis_sha256"] != _PREPARED_NATIVE64_BASIS_SHA256:
            raise ValueError("prepared catalog requires the exact native64 parent basis")
    if value["packages"] and value["packages"]["30"]["parent_package_sha256"] != value["packages"]["80"]["parent_package_sha256"]:
        raise ValueError("prepared targets must share the exact native BI2_50 parent")
    return value, inputs


def _prepared_path(base, value, root):
    if not isinstance(value, str) or not value or len(value) > 512 or Path(value).is_absolute() \
            or value.startswith("resource:") or len(Path(value).parts) > 16:
        raise ValueError("prepared artifact paths must be bounded relative paths")
    lexical = Path(base) / value
    if {"#snapshot", ".snapshot"}.intersection(lexical.parts) \
            or any(parent.is_symlink() for parent in (lexical, *lexical.parents)):
        raise ValueError("prepared artifact path uses a snapshot or symlink")
    path = _input_path(base, value)
    if not path.is_relative_to(root.resolve()):
        raise ValueError("prepared artifact path escapes its package resource")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _prepared_native64_inputs(package_path, total):
    catalog, inputs = _prepared_catalog()
    entry = catalog["packages"].get(f"{total:g}")
    if entry is None:
        raise RuntimeError("exact prepared p2 operators have not been admitted")
    if not str(package_path).startswith("resource:"):
        lexical = Path(package_path)
        if any(parent.is_symlink() for parent in (lexical, *lexical.parents)):
            raise ValueError("prepared package path must not use symlinks")
    path = _input_path(Path.cwd(), package_path)
    package = _prepared_json(path, inputs, entry["package_sha256"])
    if (package.get("schema") != "yap.prepared_p2_geometry_package.v1"
            or package.get("name") != f"BI2_{total:g}" or type(package.get("rank")) is not int
            or package["rank"] != 64 or package.get("ready") is not False
            or package.get("state") != "CANDIDATE_NOT_VALIDATED"):
        raise ValueError("prepared package requires the exact native64 engineering candidate")
    for field in ("parent_package_sha256", "port_basis_sha256"):
        if package.get(field) != entry[field]:
            raise ValueError(f"prepared package {field} differs from catalog")
    if package.get("audit_receipt_sha256") != entry["audit_sha256"]:
        raise ValueError("prepared package audit SHA256 differs from catalog")
    retained = total - 13.
    aliases = dict(bi="bi", beol="beol", retained_si=f"tsv_si_h{retained:g}")
    keys = set(aliases.values())
    for field in ("datasets", "prepared", "dataset_sha256", "prepared_sha256"):
        if not isinstance(package.get(field), dict) or set(package[field]) != keys:
            raise ValueError(f"prepared {field} requires exactly the target p2 source keys")
    root = path.parent.parent if path.parent.name == "packages" else path.parent
    parent_path = _prepared_path(path.parent, package.get("parent_package"), root)
    parent = _prepared_json(parent_path, inputs, entry["parent_package_sha256"])
    if (parent.get("schema") != "exp12.ofat.geometry_package.v1" or parent.get("name") != "BI2_50"
            or type(parent.get("rank")) is not int or parent["rank"] != 64
            or parent.get("ready") is not True or parent.get("state") != "READY_ENGINEERING_PREVIEW"
            or parent.get("port_basis_sha256") != entry["port_basis_sha256"]):
        raise ValueError("prepared package requires the exact native BI2_50 parent")
    if not isinstance(parent.get("dataset_sha256"), dict):
        raise ValueError("native parent requires explicit dataset hashes")
    for key in ("bi", "beol"):
        if package["dataset_sha256"][key] != parent.get("dataset_sha256", {}).get(key):
            raise ValueError(f"prepared {key} must be byte-identical to the native parent")
    receipt_path = _prepared_path(path.parent, package.get("audit_receipt"), root)
    receipt = _prepared_json(receipt_path, inputs, entry["audit_sha256"])
    if (receipt.get("schema") != "yap.prepared_p2_operator_audit.v1"
            or receipt.get("state") != "PASS_STRUCTURAL_ENGINEERING_CANDIDATE"
            or receipt.get("all_cu_points_retained") is not True
            or receipt.get("thermal_geometry_validated") is not False
            or receipt.get("face_node_shape") != [46, 46]
            or any(type(count) is not int for count in receipt["face_node_shape"])):
        raise ValueError("prepared receipt must identify structurally audited, unvalidated p2 inputs")
    for field, expected in dict(total_thickness_um=total, retained_si_thickness_um=retained,
                                bonding_pitch_um=2., bonding_pad_diameter_um=1.).items():
        _prepared_number(receipt.get(field), field, expected=expected)
    for field, expected in dict(rank=64, stage_count=6).items():
        _prepared_integer(receipt.get(field), field, expected=expected)
    _prepared_number(receipt.get("cell_area_mm2"), "cell_area_mm2", positive=True)
    for field in ("parent_package_sha256", "port_basis_sha256"):
        if receipt.get(field) != entry[field]:
            raise ValueError(f"prepared receipt {field} differs from catalog")
    datasets, prepared = {}, {}
    thicknesses = dict(bi=1.5, beol=10., **{aliases["retained_si"]: retained})
    for field, hashes, paths in (("datasets", "dataset_sha256", datasets), ("prepared", "prepared_sha256", prepared)):
        if not isinstance(receipt.get(field), dict) or set(receipt[field]) != keys:
            raise ValueError(f"prepared receipt {field} bindings are incomplete")
        for key in keys:
            expected = _prepared_sha(package[hashes][key], f"{field}/{key}")
            paths[key] = _prepared_path(path.parent, package[field][key], root)
            binding = receipt[field][key]
            if not isinstance(binding, dict) or binding.get("sha256") != expected:
                raise ValueError(f"prepared receipt {field}/{key} artifact binding mismatch")
            if field == "datasets":
                _prepared_number(binding.get("thickness_mm"), f"{key} thickness", expected=thicknesses[key] / 1000.)
                _prepared_integer(binding.get("cu_point_count"), f"{key} Cu count", positive=True)
                _prepared_integer(binding.get("macro_size"), f"{key} macro size", expected=134)
                _prepared_integer(binding.get("bubble_size"), f"{key} bubble size", positive=True)
                _prepared_sha(binding.get("dataset_content_sha256"), f"{key} dataset content")
    for key in keys:
        if receipt["prepared"][key].get("parent_dataset_content_sha256") != receipt["datasets"][key]["dataset_content_sha256"]:
            raise ValueError(f"prepared {key} geometry must bind its dataset content")
    transfer = receipt.get("transfer_audit")
    if not isinstance(transfer, dict):
        raise ValueError("prepared receipt requires an independent bubble-transfer audit")
    for field, expected in dict(source_thickness_um=38.5, target_thickness_um=retained).items():
        _prepared_number(transfer.get(field), field, expected=expected)
    _prepared_number(transfer.get("source_pitch_um"), "source_pitch_um", expected=6.)
    rank = _prepared_transfer_space(transfer, "final bubble space")
    if receipt["datasets"][aliases["retained_si"]]["bubble_size"] != rank:
        raise ValueError("transferred bubble rank differs from target dataset audit")
    source_rank = _prepared_integer(transfer.get("source_rank"), "source_rank", expected=96)
    raw_rank = _prepared_integer(transfer.get("raw_transferred_rank"), "raw_transferred_rank", expected=source_rank)
    if _prepared_transfer_space(transfer.get("raw_transfer_audit"), "raw transfer") != raw_rank:
        raise ValueError("raw transfer audit rank differs from donor rank")
    _prepared_integer(transfer.get("closure_added_rank"), "closure_added_rank", expected=rank - raw_rank)
    for field in ("source_bubble_sha256", "source_dataset_sha256", "transferred_bubble_sha256", "raw_transferred_bubble_sha256"):
        _prepared_sha(transfer.get(field), field)
    if transfer.get("method") != "normalized_thickness_displacement_transfer_plus_inplane_swap_closure":
        raise ValueError("bubble transfer requires the audited in-plane swap closure method")
    bubbles = {}
    for field, hashes in (("transferred_bubble", "transferred_bubble_sha256"),
                          ("raw_transferred_bubble", "raw_transferred_bubble_sha256")):
        bubble = transfer.get(field)
        if not isinstance(bubble, dict) or bubble.get("sha256") != transfer[hashes]:
            raise ValueError(f"{field} artifact binding mismatch")
        bubble_path = _prepared_path(receipt_path.parent, bubble.get("path"), root)
        if bubble_path in bubbles:
            raise ValueError("raw and finalized bubble artifacts must be distinct")
        bubbles[bubble_path] = transfer[hashes]
    basis = _prepared_path(path.parent, package.get("port_basis"), root)
    return path, package, receipt_path, receipt, aliases, datasets, prepared, basis, bubbles, inputs


def _load_prepared_native64_library(package_path, total):
    """Load audited exact-target all-Gauss candidates, without any Si cap."""
    path, package, receipt_path, receipt, aliases, datasets, prepared, basis, bubbles, inputs = \
        _prepared_native64_inputs(package_path, total)
    for field, hashes, paths in (("datasets", "dataset_sha256", datasets), ("prepared", "prepared_sha256", prepared)):
        for key, filename in paths.items():
            _record_input(filename, inputs, package[hashes][key])
    _record_input(basis, inputs, package["port_basis_sha256"])
    for bubble, digest in bubbles.items():
        _record_input(bubble, inputs, digest)
    runtime = _import_mechanics_module("exp12_mindlin_sequential_runtime")
    library = runtime.load_q6_sequential_library(datasets, basis, port_rank_per_face=64,
        validate=True, verify_content=True, factorize_elements=True, prepared_geometry_paths=prepared,
        frozen_expected_port_basis_sha256_by_dataset={key: package["port_basis_sha256"] for key in datasets})
    if type(library.port_rank_per_face) is not int or library.port_rank_per_face != 64 \
            or library.common_port_gauge_vectors is not None:
        raise ValueError("prepared native64 library requires its rigid-mode gauge contract")
    thicknesses = {key: receipt["datasets"][key]["thickness_mm"] for key in datasets}
    _check_library(library, thicknesses)
    data_module = _import_mechanics_module("exp12_layer_rom_dataset")
    cu_cards, si_materials = [], []
    for key in datasets:
        dataset = data_module.LayerROMDataset.load(datasets[key], validate=True, verify_content=True)
        binding, op = receipt["datasets"][key], library.operators[key]
        with np.load(datasets[key], allow_pickle=False) as archive:
            if "content_sha256" not in archive or archive["content_sha256"].shape != ():
                raise ValueError(f"prepared {key} dataset content requires a scalar NPZ digest")
            content_digest = archive["content_sha256"].item()
        if _prepared_sha(content_digest, f"{key} loaded dataset content") != binding["dataset_content_sha256"]:
            raise ValueError(f"prepared {key} dataset content differs from its audit")
        signature = dataset.geometry_signature
        if (dataset.macro_size != 134 or dataset.bubble_size != binding["bubble_size"]
                or op.bubble_size != dataset.bubble_size
                or not np.allclose(dataset.macro_coordinate_scale, op.macro_coordinate_scale, rtol=0., atol=1e-14)
                or not np.isclose(signature["thickness_mm"], thicknesses[key], rtol=0., atol=1e-12)
                or not np.isclose(signature["cell_area_mm2"], receipt["cell_area_mm2"], rtol=0., atol=1e-14)
                or not np.isclose(library.cell_area_mm2, receipt["cell_area_mm2"], rtol=0., atol=1e-14)):
            raise ValueError(f"prepared {key} dataset geometry/coordinate audit mismatch")
        weights = np.asarray(op.weights_mm3)
        if op.cu_point_count != binding["cu_point_count"] or weights.shape != (binding["cu_point_count"],) \
                or not np.isfinite(weights).all() or np.any(weights <= 0.):
            raise ValueError(f"prepared {key} all-Gauss Cu audit mismatch")
        if dataset.reference_temperature_c != 25. or dataset.linear_reference_mode != "material_separable_bbar" \
                or not dataset.reference_constant_materials or "si" not in dataset.reference_constant_materials:
            raise ValueError(f"prepared {key} requires explicit material-separable Si reference data")
        card = asdict(dataset.cu_card)
        if asdict(op.cu_card) != card:
            raise ValueError(f"prepared {key} runtime Cu card differs from dataset")
        cu_cards.append(card)
        si_materials.append(asdict(dataset.reference_constant_materials["si"]))
        del dataset
    if any(card != cu_cards[0] for card in cu_cards[1:]) or any(si != si_materials[0] for si in si_materials[1:]):
        raise ValueError("prepared target must retain the native parent Cu card and Si material")
    result = replace(library,
        operators={key: library.operators[source] for key, source in aliases.items()},
        thickness_mm={key: library.thickness_mm[source] for key, source in aliases.items()})
    _check_library(result, {key: thicknesses[source] for key, source in aliases.items()})
    return result, dict(schema="yap.assembly_operator_library.v1", package_path=str(path),
        input_sha256=inputs, port_basis_sha256=package["port_basis_sha256"], port_rank_per_face=64,
        source_operator_keys=aliases, frozen_preview=False, bonding_pitch_um=2., pad_diameter_um=1.,
        original_package_state=package["state"], original_package_ready=package["ready"],
        operator_build_receipt=str(receipt_path), operator_build_state=receipt["state"],
        transfer_audit=receipt["transfer_audit"], operator_inputs_verified=True, physical_readiness=False,
        all_cu_points_retained=True,
        removable_bulk_operator_built=False,
        implementation_sha256={str(Path(m.__file__).resolve()): _sha256(m.__file__) for m in (runtime, data_module)},
        physics_qualification="EXACT_P2_NATIVE64_H38P5_TRANSFERRED_HOMOGENEOUS_BUBBLES_TARGET_ALL_GAUSS_ENGINEERING_CANDIDATE_NOT_VALIDATED")


def _prepared_library(library):
    aliases = dict(_SOURCE_KEYS)
    _check_library(library, {source: _THICKNESS_MM[key] for key, source in aliases.items()})
    result = replace(library,
        operators={key: library.operators[source] for key, source in aliases.items()},
        thickness_mm={key: library.thickness_mm[source] for key, source in aliases.items()})
    _check_library(result)
    return result


def _load_reference_p2_preview_library(package_path, geometry, *, prepared_bodies=False, split_cap=False):
    """Reuse identified BI2/P24 data for a new TOTAL50 tree preview."""
    _check_geometry(geometry, initial_total_um=50. if prepared_bodies else 775.)
    path = _input_path(Path.cwd(), package_path)
    inputs = {}
    _record_input(path, inputs, _REFERENCE_P2_MANIFEST_SHA256)
    package = json.loads(path.read_text())
    if (package.get("schema") != "exp12.ofat.geometry_package.v1" or package.get("name") != "BI2_50"
            or type(package.get("rank")) is not int or package["rank"] != 24
            or package.get("ready") is not False or package.get("state") != "CANDIDATE_NOT_VALIDATED"):
        raise ValueError("reference p2 preview requires the identified BI2_50/P24 candidate")
    datasets = {key: _input_path(path.parent, package["datasets"][key])
                for key in ("bi", "beol", "tsv_si_h37")}
    for key, filename in datasets.items():
        _record_input(filename, inputs, package["dataset_sha256"][key])
    basis = _input_path(path.parent, package["port_basis"])
    _record_input(basis, inputs, package["port_basis_sha256"])
    receipt_path = path.parent.parent / "MANIFEST.json"
    _record_input(receipt_path, inputs)
    receipt = json.loads(receipt_path.read_text())
    if (receipt["state"] != "BUILT_NOT_VALIDATED" or receipt["training_cases"] != ["bi2"]
            or receipt["port_audit"]["rank"] != 24 or receipt["all_cu_points_retained"] is not True
            or receipt["whole_stack_response_fit"] is not False):
        raise ValueError("BI2/P24 build receipt is inconsistent")
    for filename in datasets.values():
        matches = [item for item in receipt["layers"].values()
                   if _input_path(path.parent.parent, item["path"]) == filename]
        if len(matches) != 1 or matches[0]["sha256"] != inputs[str(filename)]:
            raise ValueError("BI2/P24 dataset differs from its build receipt")
    api = _import_mechanics_module("exp12_w2w_window_20260914")
    root = Path(api.__file__).resolve().parent
    parent_path = _input_path(root, "resource:geometry/packages/BI2_50.json")
    _record_input(parent_path, inputs, package["parent_package_sha256"])
    parent = json.loads(parent_path.read_text())
    _record_input(_input_path(parent_path.parent, parent["port_basis"]), inputs, parent["port_basis_sha256"])
    for key in ("bi", "beol"):
        _record_input(_input_path(parent_path.parent, parent["datasets"][key]), inputs, parent["dataset_sha256"][key])
    runtime = _import_mechanics_module("exp12_mindlin_sequential_runtime")
    library = runtime.load_q6_sequential_library(datasets, basis, port_rank_per_face=24,
        validate=True, verify_content=True, factorize_elements=True,
        frozen_expected_port_basis_sha256_by_dataset={key: package["port_basis_sha256"] for key in datasets})
    if library.port_rank_per_face != 24 or library.common_port_gauge_vectors is not None:
        raise ValueError("BI2/P24 needs its existing rigid-mode gauge contract")
    if prepared_bodies:
        aliases = dict(bi="bi", beol="beol", retained_si="tsv_si_h37")
        library = _prepared_library(library)
        return library, dict(schema="yap.assembly_operator_library.v1", package_path=str(path),
            input_sha256=inputs, port_basis_sha256=package["port_basis_sha256"], port_rank_per_face=24,
            source_operator_keys=aliases, frozen_preview=True, bonding_pitch_um=2.,
            original_package_state=package["state"], original_package_ready=package["ready"],
            operator_build_receipt=str(receipt_path), operator_build_state=receipt["state"],
            removable_bulk_operator_built=False,
            physics_qualification="BI2_P24_PREPARED_TOTAL50_NEW_DIE_ASSEMBLY_PATHS_NOT_VALIDATED")
    factory = _bound_reference_factory(api, library, package, parent_path)
    factory.get(725.)
    cap = factory.blocks[factory.combine(725., 1.5)]
    audits = dict(direct_small_stack_relative_defect=factory.direct_comparison,
        block_audits=factory.audits, bulk_audits=[_reference_cap_audit(api, library, cap, 726.5)],
        inputs_sha256=factory.inputs, pure_Si_parent_bubbles="all native, not compact material-specific POD",
        factory_native_binding=str(parent_path))
    _check_reference_bulk_audits(audits, thickness_um=726.5)
    if split_cap:
        split_audits = []
        for thickness in (1.5, 725.):
            audit = _reference_cap_audit(api, library, factory.blocks[thickness], thickness)
            audit["relaxed_normal_one_percent_pass"] = audit["relaxed_normal_scaled_defect"] <= .01
            audit["free_thermal_1e8_pass"] = audit["free_thermal_q6_max_absolute_defect"] <= 1e-8
            if thickness == 725.:
                _check_reference_bulk_audits(dict(audits, bulk_audits=[audit]), thickness_um=thickness)
            else:
                # The old factory's 1.5-um building block is not a separately
                # admitted bulk operator. Preserve its failed diagnostic; the
                # actual ordered birth cap must pass the unchanged bulk gates.
                audit["qualification"] = "UNQUALIFIED_LEGACY_SLICE_DIAGNOSTIC_NOT_STANDALONE_BULK_ADMISSION"
                if not all(np.isfinite(np.asarray(audit[key], dtype=float)).all()
                           for key in ("abd_per_width", "analytical_abd", "free_thermal_q6_defect")):
                    raise ValueError("legacy retained pure-Si slice has nonfinite ABD or free thermal response")
            split_audits.append(audit)
        ordered, composition = _join_segments(library,
            (factory.blocks[1.5], .0015), (factory.blocks[725.], .725))
        ordered_audit = _reference_cap_audit(api, library, ordered[0], 726.5)
        _check_reference_bulk_audits(dict(audits, bulk_audits=[ordered_audit]), thickness_um=726.5)
        audits["split_cap_audits"] = split_audits
        audits["ordered_split_cap_audit"] = ordered_audit
        audits["ordered_split_condensation_audit"] = composition
        audits["split_cap_birth_policy"] = "original_1p5_and_725_pure_Si_slices_both_present_from_birth"
    for filename, expected in factory.inputs.items():
        _record_input(_input_path(root, filename), inputs, expected)
    aliases = dict(bi="bi", beol="beol", retained_si="tsv_si_h37", bulk_si="pure_si_726p5")
    library = replace(library, operators=dict(bi=library.operators["bi"], beol=library.operators["beol"],
        retained_si=library.operators["tsv_si_h37"], bulk_si=factory.blocks[725.] if split_cap else cap),
        thickness_mm=dict(bi=.0015, beol=.010, retained_si=.037, bulk_si=.725 if split_cap else .7265))
    if split_cap:
        library = replace(library, operators={**library.operators, "outer_si": factory.blocks[1.5]},
                          thickness_mm={**library.thickness_mm, "outer_si": .0015})
        aliases.update(bulk_si="pure_si_725", outer_si="pure_si_1p5")
    _check_library(library, library.thickness_mm)
    if cap.cu_point_count or cap.bubble_size or cap.plastic_active:
        raise ValueError("BI2 preview pure-Si cap must have no Cu history or uncondensed bubbles")
    return library, dict(schema="yap.assembly_operator_library.v1", package_path=str(path),
        input_sha256=inputs, port_basis_sha256=package["port_basis_sha256"], port_rank_per_face=24,
        source_operator_keys=aliases, cap_audits=audits, frozen_preview=True, bonding_pitch_um=2.,
        original_package_state=package["state"], original_package_ready=package["ready"],
        operator_build_receipt=str(receipt_path), operator_build_state=receipt["state"],
        implementation_sha256={str(Path(m.__file__).resolve()): _sha256(m.__file__) for m in (api, runtime)},
        split_cap_from_birth=split_cap, native64_W2W_admitted=False,
        physics_qualification="BI2_P24_TOTAL50_NEW_ASSEMBLY_TREE_ENGINEERING_PREVIEW_NOT_VALIDATED")
