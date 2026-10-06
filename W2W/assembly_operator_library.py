"""Checked native operators for the TOTAL775 -> prepared TOTAL50 adapter.

The cap is equivalent to a chain of existing reduced pure-Si segments. This
does not confer continuum or process validation on the new assembly route.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np

try:
    from YAP.history_mechanics_backend import _import_mechanics_module
    from YAP.mechanics_runtime import resolve_native_input, relocate_geometry_paths, P2_COMPACT_MANIFEST_SHA256
except ModuleNotFoundError:
    from history_mechanics_backend import _import_mechanics_module
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
    mesh, cache, or generated operator is written by this loader.
    """
    if type(frozen_preview) is not bool:
        raise ValueError("frozen_preview must be Boolean")
    if geometry.get("initial_total_thickness_um") == geometry.get("target_total_thickness_um"):
        check_prepared_wafer_geometry(geometry)
        if not frozen_preview or Path(package_path).name != "BI2_50.json":
            raise ValueError("prepared wafer execution requires the identified reference BI2/P24 preview")
        library, provenance = _load_reference_p2_preview_library(package_path, geometry, prepared_bodies=True)
        provenance["physics_qualification"] = "BI2_P24_PREPARED_TOTAL50_NEW_WAFER_ASSEMBLY_PATHS_NOT_VALIDATED"
        return library, provenance
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
    """Require the existing TOTAL50 layers, with both BIs already present."""
    if set(geometry) != set(_GEOMETRY_UM) | {"diameter_mm"}:
        raise ValueError("invalid prepared wafer geometry")
    _check_geometry(geometry, initial_total_um=50.)


def check_prepared_die_geometry(geometry):
    """Check the supported TOTAL50 prepared geometry without opening datasets."""
    expected = dict(total_thickness_um=50., front_bi_thickness_um=1.5,
                    back_bi_thickness_um=1.5, beol_thickness_um=10.)
    if set(geometry) != set(expected) | {"width_mm", "length_mm"}:
        raise ValueError("invalid prepared die geometry")
    for key, value in expected.items():
        if isinstance(geometry[key], bool) or not np.isclose(geometry[key], value, rtol=0., atol=1e-9):
            raise ValueError(f"prepared die library requires {key}={value:g}")
    width, length = geometry["width_mm"], geometry["length_mm"]
    if isinstance(width, bool) or isinstance(length, bool) or not np.isfinite([width, length]).all() or width <= 0 or width != length:
        raise ValueError("the existing scalar reference die metric requires a square footprint")


def load_prepared_die_library(package_path, geometry, *, frozen_preview=False):
    """Use the exact BI2/P24 layers, without a removable wafer substrate."""
    if frozen_preview is not True or Path(package_path).name != "BI2_50.json":
        raise ValueError("prepared die execution currently requires the identified reference BI2/P24 preview")
    check_prepared_die_geometry(geometry)
    width = geometry["width_mm"]
    adapter = {**_GEOMETRY_UM, "diameter_mm": width, "initial_total_thickness_um": 50.}
    return _load_reference_p2_preview_library(package_path, adapter, prepared_bodies=True)


def _load_reference_p2_preview_library(package_path, geometry, *, prepared_bodies=False):
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
        library = replace(library, operators={key: library.operators[value] for key, value in aliases.items()},
                          thickness_mm=dict(bi=.0015, beol=.010, retained_si=.037))
        _check_library(library)
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
    for filename, expected in factory.inputs.items():
        _record_input(_input_path(root, filename), inputs, expected)
    aliases = dict(bi="bi", beol="beol", retained_si="tsv_si_h37", bulk_si="pure_si_726p5")
    library = replace(library, operators=dict(bi=library.operators["bi"], beol=library.operators["beol"],
        retained_si=library.operators["tsv_si_h37"], bulk_si=cap),
        thickness_mm=dict(bi=.0015, beol=.010, retained_si=.037, bulk_si=.7265))
    _check_library(library, library.thickness_mm)
    if cap.cu_point_count or cap.bubble_size or cap.plastic_active:
        raise ValueError("BI2 preview pure-Si cap must have no Cu history or uncondensed bubbles")
    return library, dict(schema="yap.assembly_operator_library.v1", package_path=str(path),
        input_sha256=inputs, port_basis_sha256=package["port_basis_sha256"], port_rank_per_face=24,
        source_operator_keys=aliases, cap_audits=audits, frozen_preview=True, bonding_pitch_um=2.,
        original_package_state=package["state"], original_package_ready=package["ready"],
        operator_build_receipt=str(receipt_path), operator_build_state=receipt["state"],
        implementation_sha256={str(Path(m.__file__).resolve()): _sha256(m.__file__) for m in (api, runtime)},
        physics_qualification="BI2_P24_TOTAL50_NEW_ASSEMBLY_TREE_ENGINEERING_PREVIEW_NOT_VALIDATED")
