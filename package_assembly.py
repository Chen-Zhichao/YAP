"""2.5D overlay yield: completed 3D stacks attached to an ideally flat interposer.

Internal stack histories use the existing Cu-history engine. Final room-
temperature attachment uses frozen-history elastic ABD references only.
Final package annealing and all subsequent processes are assumed to cause no
additional yield loss; their mechanics are intentionally not simulated.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4

import numpy as np
from omegaconf import OmegaConf

sys.dont_write_bytecode = True

from manufacturing_bow import (
    ManufacturingLayer, calibrate_fabrication_source, clpt_source_response,
)
from initial_bow_distribution import AnchorPreservingDistribution
from history_bow_yield import worst_critical_pad_survival


ROOT = Path(__file__).resolve().parent
SCHEMA = "yap.package_assembly.v1"
POST_ATTACHMENT_POLICY = "final_package_anneal_and_subsequent_processes_assumed_no_additional_yield_loss"
ROTATION = np.array([[0., -1.], [1., 0.]])


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name, *, minimum=None, maximum=None):
    _require(type(value) in (int, float) and np.isfinite(value), name + " must be finite numeric data")
    _require(minimum is None or value >= minimum, name + " is below its minimum")
    _require(maximum is None or value <= maximum, name + " exceeds its maximum")
    return float(value)


def _array(value, shape, name):
    raw = np.asarray(value)
    _require(raw.dtype.kind in "iuf" and raw.shape == shape and np.isfinite(raw).all(),
             name + " has invalid numeric values or shape")
    return np.array(raw, dtype=float, copy=True)


def _keys(value, required, optional, name):
    _require(isinstance(value, dict) and not required - value.keys()
             and not value.keys() - required - optional, "invalid fields in " + name)


def _read_yaml(path):
    path = Path(path)
    _require(path.stat().st_size <= 1024**2, "bounded YAML input required")
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    # Reject unresolved interpolations and non-JSON YAML values.
    raw = json.dumps(value, allow_nan=False)
    _require("${" not in raw, "package inputs must use explicit values, not interpolations")
    return value


def _path(base, value):
    _require(isinstance(value, str) and value.strip(), "input path must be nonempty")
    path = (base / value).resolve()
    _require(not {"#snapshot", ".snapshot"}.intersection(path.parts), "snapshot inputs are not allowed")
    return path


def _seed(seed, name, stage):
    identity = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")
    return np.random.SeedSequence([seed, identity, stage])


def _write(path, value):
    from mechanics_runtime import validate_user_write_path
    with validate_user_write_path(path).open("x", encoding="ascii") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def fabrication_reference():
    """The existing 50-um-Si / 10-um-BEOL / 1.5-um-BI calibration ledger."""
    return (
        ManufacturingLayer("tsv_si", 50., 99496.90200129498, .2917809724509618,
                           is_bulk_silicon=True),
        ManufacturingLayer("beol", 10., 93300., .2295, eigenstrain_per_amplitude=1.),
        ManufacturingLayer("bonding_interface", 1.5, 84388.273369263, .20337942194439157),
    )


def interposer_population(settings, samples, seed):
    """One shared source draw per package; no geometry-dependent recalibration."""
    _keys(settings, {"width_mm", "length_mm"}, {"total_thickness_um", "bi_thickness_um",
        "si_e_mpa", "si_nu", "bi_e_mpa", "bi_nu", "source_cv", "initial_bow_um"}, "interposer")
    width = _number(settings["width_mm"], "interposer width", minimum=1e-9)
    length = _number(settings["length_mm"], "interposer length", minimum=1e-9)
    _require(width == length, "the current membrane metric requires a square interposer")
    total = _number(settings.get("total_thickness_um", 100.), "interposer total thickness", minimum=1e-9)
    bi = _number(settings.get("bi_thickness_um", 1.5), "interposer BI thickness", minimum=1e-9)
    _require(total > bi, "interposer total thickness must include BI and positive Si")
    layers = (
        ManufacturingLayer("Si", total-bi, settings.get("si_e_mpa", 131000.),
                           settings.get("si_nu", .28), is_bulk_silicon=True),
        ManufacturingLayer("front_BI", bi, settings.get("bi_e_mpa", 84388.273369263),
                           settings.get("bi_nu", .20337942194439157), eigenstrain_per_amplitude=1.),
    )
    response = clpt_source_response(layers)
    q = np.asarray(response["natural_birth_q6_per_amplitude"])
    # Same sign conversion as AssemblyExecutor: native curvature gives paper bow.
    q = q * np.array([1., 1., 1., -1., -1., -1.])
    unit_bow = 1000*q[3:5]*np.array([width, length])**2/8
    if "initial_bow_um" in settings:
        _require("source_cv" not in settings, "choose initial_bow_um or a source distribution, not both")
        bow = _number(settings["initial_bow_um"], "interposer initial bow")
        amplitudes = np.full(samples, bow/unit_bow[0])
        source_mean, source_std = float(amplitudes[0]), 0.
        source = dict(model="fixed_signed_center_minus_mid_edge_bow", initial_bow_um=bow)
    else:
        calibration = calibrate_fabrication_source(fabrication_reference(),
            reference_si_thickness_um=50., reference_span_mm=10.,
            target_upper_magnitude_um=40., source_cv=.10, sigma_level=3., source_sign=-1.)
        cv = _number(settings.get("source_cv", .20), "interposer source CV", minimum=0.)
        population = AnchorPreservingDistribution(calibration.nominal_amplitude, target_cv=cv)
        z = np.random.default_rng(_seed(seed, "interposer", 0)).standard_normal(samples)
        amplitudes = calibration.source_sign*population.sample(z)
        source_mean, source_std = calibration.source_sign*population.mean, population.std
        source = dict(model="existing_anchor_preserving_fabrication_source",
            calibration=calibration.record(), population=population.audit(),
            signed_mean_amplitude=calibration.source_sign*population.mean,
            std_amplitude=population.std,
            assumption="BEOL-calibrated source statistics applied to interposer top BI; not independently calibrated")
    bows = amplitudes[:, None]*unit_bow
    record = dict(footprint_mm=[width, length], total_thickness_um=total,
        bi_thickness_um=bi, layers=response["layers"],
        elastic_abd=response["ABD_matrix"], native_q6_per_source=q.tolist(),
        source=source, bow_convention="center_minus_mid_edge",
        mean_bow_um=(source_mean*unit_bow).tolist(), std_bow_um=(source_std*np.abs(unit_bow)).tolist(),
        sample_mean_bow_um=np.mean(bows, axis=0).tolist(), sample_std_bow_um=np.std(bows, axis=0).tolist(),
        shared_across_all_external_interfaces=True, independent_physics_validation=False)
    return record, amplitudes, bows


def elastic_face_strain(abd, curvature, footprint_mm, face_z_mm):
    """Frozen elastic flattening increment plus the existing full-square metric."""
    from D2W.die_membrane import square_membrane_metric
    k = _array(abd, (6, 6), "elastic ABD")
    c = _array(curvature, (3,), "native curvature")
    spans = _array(footprint_mm, (2,), "footprint")
    z = _number(face_z_mm, "mating-face datum")
    _require(np.all(spans > 0) and spans[0] == spans[1], "square positive footprint required")
    _require(np.allclose(k, k.T, rtol=1e-10, atol=1e-12), "elastic ABD must be symmetric")
    np.linalg.cholesky(k)
    v = (np.linalg.solve(k[:3, :3], k[:3, 3:])-z*np.eye(3)) @ c
    linear = np.array([[v[0], v[2]/2], [v[2]/2, v[1]]])
    metric, audit = square_membrane_metric(k[:3, :3], c, spans[0]/2)
    return linear+metric, dict(linear_face_strain=linear.tolist(), membrane_metric=metric.tolist(),
                               metric_audit=audit, constitutive_state_evolved=False)


def score_affine(matrices, overlay):
    """Return (joint draw, interface) survival using the existing noise model."""
    m = np.asarray(matrices, dtype=float)
    shifts = np.asarray(overlay["translation_um"], dtype=float)
    angles = np.asarray(overlay["rotation_rad"], dtype=float)
    samples, interfaces = angles.shape
    _require(m.shape in ((interfaces, 2, 2), (samples, interfaces, 2, 2))
             and np.isfinite(m).all(), "affine matrix and placement dimensions differ")
    _array(shifts, (samples, interfaces, 2), "translation draws")
    xy = np.asarray(overlay["coordinates_um"], dtype=float)
    _require(xy.ndim == 3 and xy.shape[0] == 1 and xy.shape[2] == 2
             and len(xy[0]) > 0 and np.isfinite(xy).all(), "one finite die-site pad set required")
    correction = _number(overlay["magnification_correction_fraction"], "magnification correction",
                         minimum=0., maximum=1.)
    transform = (1-correction)*m + angles[:, :, None, None]*ROTATION
    displacement = np.einsum("neij,spj->nespi", transform, xy)
    displacement += shifts[:, :, None, None, :]
    survival = worst_critical_pad_survival(displacement,
        critical_pad_mask=np.asarray(overlay["critical_pad_mask"], dtype=bool),
        threshold_um=overlay["threshold_um"], sigma_um=overlay["residual_sigma_um"])
    return survival.probability[:, :, 0], np.max(np.linalg.norm(displacement, axis=-1), axis=(-1, -2))


def compound_package(internal, attachment, other=None):
    """Products INSIDE the joint expectation preserve shared interposer draws."""
    internal, attachment = np.asarray(internal, dtype=float), np.asarray(attachment, dtype=float)
    _require(internal.ndim == 2 and internal.shape == attachment.shape and min(internal.shape) > 0,
             "matching nonempty (sample, stack) probabilities required")
    for values in (internal, attachment):
        _require(np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)),
                 "probabilities must be in [0,1]")
    factors = np.ones(internal.shape[1]) if other is None else _array(other, (internal.shape[1],), "other yield factors")
    _require(np.all((factors >= 0) & (factors <= 1)), "other yield factors must be in [0,1]")
    joint = np.prod(internal*attachment*factors[None], axis=1)
    return dict(package_overlay_yield=float(np.mean(np.prod(internal*attachment, axis=1))),
        package_yield=float(np.mean(joint)),
        monte_carlo_standard_error=float(np.std(joint, ddof=1)/np.sqrt(len(joint))) if len(joint) > 1 else 0.,
        marginal_product_diagnostic=float(np.prod(np.mean(internal, axis=0)*np.mean(attachment, axis=0)*factors))), joint


def stack_plan(config, order, annealing):
    """Keep the existing sequential plan byte-compatible; rebind batch histories."""
    from D2W.assembly_order import build_plan
    from W2W.assembly_order import content_id
    config = deepcopy(config)
    if order is not None:
        config["assembly_orders"] = order
    plan = build_plan(config)
    _require(len(plan["routes"]) == 1, "each package stack must select one assembly_order, not all orders")
    _require(annealing in ("sequential", "batch"), "stack annealing must be sequential or batch")
    if annealing == "sequential":
        return plan
    plan["context"]["package_stack_annealing"] = "batch_after_all_internal_attachments"
    plan["context_id"] = content_id(dict(context=plan["context"], initialization=plan["initialization"]))
    route = plan["routes"][0]
    route["context_id"] = plan["context_id"]
    visited = set()
    def rebind(value):
        if isinstance(value, (dict, list)):
            if id(value) in visited:
                return
            visited.add(id(value))
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "state_id":
                    value[key] = content_id(dict(original=child, annealing="package_stack_batch"))
                elif key == "thermal_cycle_counts":
                    value[key] = [0]*len(child)
                else:
                    rebind(child)
        elif isinstance(value, list):
            for child in value:
                rebind(child)
    rebind(route)
    for event in route["events"][:-1]:
        event["operations"] = [op for op in event["operations"] if op not in ("free_anneal", "free_cooldown")]
    final = route["events"][-1]["result"]
    final["thermal_cycle_counts"] = [1]*len(final["die_ids"])
    route["final"] = deepcopy(final)
    route["request_id"] = content_id({k: v for k, v in route.items() if k != "request_id"})
    return plan


def prepare_package(path):
    """Validate layout, input snapshots and process plans without loading operators."""
    from D2W.assembly_order import load_design
    document = _read_yaml(path)
    _keys(document, {"name", "interposer", "stacks"}, {"samples", "seed", "attachment"}, "package")
    name = document["name"]
    _require(isinstance(name, str) and name.isascii() and name.replace("_", "").replace("-", "").isalnum()
             and len(name) <= 96, "simple ASCII package name required")
    samples, seed = document.get("samples", 8192), document.get("seed", 20261118)
    _require(type(samples) is int and 2 <= samples <= 100000, "samples must be an integer in 2..100000")
    _require(type(seed) is int and seed >= 0, "seed must be a nonnegative integer")
    _require(isinstance(document["stacks"], list) and 1 <= len(document["stacks"]) <= 64,
             "package requires 1..64 stacks")
    ip, amplitudes, bows = interposer_population(document["interposer"], samples, seed)
    prepared, names = [], set()
    for entry in document["stacks"]:
        _keys(entry, {"name", "design_dir", "config", "center_mm"},
              {"assembly_order", "annealing", "attachment", "other_yield_factor", "initial_bows_um",
               "operator_package", "temporal_profile"}, "stack")
        label = entry["name"]
        _require(isinstance(label, str) and label and label.isascii()
                 and label.replace("_", "").replace("-", "").isalnum()
                 and len(label) <= 96 and label not in names, "unique simple ASCII stack names required")
        names.add(label)
        directory, config_path = (_path(Path(path).resolve().parent, entry[k]) for k in ("design_dir", "config"))
        cfg = OmegaConf.create(_read_yaml(config_path)["d2w_modeling"])
        cfg.OVERLAY_SAMPLES = samples
        cfg.RANDOM_SEED = int(_seed(seed, label, 0).generate_state(1)[0])
        design_name = OmegaConf.load(directory/"generated_stack_config.3dbx").Design.name
        design, _, overlay, _, provenance = load_design(cfg, directory, design_name)
        if "initial_bows_um" in entry:
            design["stack"]["initial_bows_um"] = entry["initial_bows_um"]
        annealing = entry.get("annealing", str(cfg.get("ANNEAL_MODE", "sequential")))
        plan = stack_plan(design["stack"], entry.get("assembly_order"), annealing)
        settings = design["mechanics"]
        if "operator_package" in entry:
            from W2W.assembly_operator_library import prepared_operator_request
            selected = entry["operator_package"]
            _require(isinstance(selected, str) and selected.startswith(("component:", "coupon:")),
                     "operator override must be component:<id> or coupon:<id>")
            settings.update(operator_package=selected, temporal_profile="component_graded25", frozen_preview=False)
            provenance["operator_request"] = prepared_operator_request(selected, plan["context"]["geometry"],
                layout=(cfg.PITCH_r_um, cfg.PITCH_c_um, 2*cfg.PAD_TOP_R_um))
        if "temporal_profile" in entry:
            settings["temporal_profile"] = entry["temporal_profile"]
        from W2W.assembly_order import _check_recipe_request
        _check_recipe_request(plan, settings["temporal_profile"])
        _require(settings.get("layer_workers", 1) == 1, "package stack execution requires layer_workers=1")
        _require(design.get("mechanics", {}).get("metric") == {"model": "square_affine"},
                 "package stack histories require the existing square_affine metric")
        center = _array(entry["center_mm"], (2,), "stack center")
        geometry = plan["context"]["geometry"]
        spans = np.array([geometry["width_mm"], geometry["length_mm"]])
        _require(np.all(np.abs(center)+spans/2 <= np.asarray(ip["footprint_mm"])/2+1e-12),
                 "stack footprint extends outside interposer")
        for previous in prepared:
            _require(not np.all(np.abs(center-previous["center"]) < (spans+previous["spans"])/2-1e-12),
                     "stack footprints overlap")
        factors = _number(entry.get("other_yield_factor", 1.), "other yield factor", minimum=0., maximum=1.)
        attached = dict(document.get("attachment", {}), **entry.get("attachment", {}))
        external = attachment_overlay(attached, cfg, overlay, spans, samples, seed, label)
        provenance["input_sha256"][str(config_path)] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        prepared.append(dict(name=label, center=center, spans=spans, plan=plan, design=design,
            overlay=overlay, external=external, provenance=provenance, annealing=annealing, other=factors))
    return document, prepared, ip, amplitudes, bows


def attachment_overlay(settings, cfg, internal, spans, samples, seed, label):
    _keys(settings, set(), {"pitch_um", "pad_diameter_um", "critical_coordinates_um", "threshold_um",
        "magnification_correction", "translation_mean_um", "translation_std_um", "rotation_mean_rad",
        "rotation_std_rad", "residual_sigma_um"}, "attachment")
    cfg = deepcopy(cfg)
    cfg.DIE_W_um, cfg.DIE_L_um = (1000*spans).tolist()
    xy = np.asarray(internal["coordinates_um"])[0]
    threshold = internal["threshold_um"]
    if "pitch_um" in settings or "pad_diameter_um" in settings:
        from D2W.compact_layout import compact_regular_geometry
        from D2W.overlay_yield_calculator import max_allowed_misalignment_calculator
        pitch = _number(settings.get("pitch_um", cfg.PITCH_r_um), "attachment pitch", minimum=1e-9)
        diameter = _number(settings.get("pad_diameter_um", pitch/2), "pad diameter", minimum=1e-9, maximum=pitch)
        cfg.PITCH_r_um = cfg.PITCH_c_um = pitch
        cfg.PAD_TOP_R_um = cfg.PAD_BOT_R_um = diameter/2
        xy = compact_regular_geometry(cfg)["critical_pad_boundary_coords"]
        threshold = float(max_allowed_misalignment_calculator(cfg=cfg, PAD_TOP_R_um=diameter/2,
            PAD_BOT_R_um=diameter/2, PITCH_r_um=pitch, PITCH_c_um=pitch,
            CONTACT_AREA_CONSTRAINT=cfg.CONTACT_AREA_CONSTRAINT,
            CRITICAL_DIST_CONSTRAINT=cfg.CRITICAL_DIST_CONSTRAINT))
    if "critical_coordinates_um" in settings:
        xy = np.asarray(settings["critical_coordinates_um"])
    _require(xy.ndim == 2 and 1 <= len(xy) <= 4096 and xy.shape[1] == 2
             and xy.dtype.kind in "iuf" and np.isfinite(xy).all()
             and np.all(np.abs(xy) <= 500*spans+1e-9), "critical coordinates must lie inside the stack footprint")
    mean = _array(settings.get("translation_mean_um", [cfg.SYSTEM_TRANSLATION_X_MEAN_um,
        cfg.SYSTEM_TRANSLATION_Y_MEAN_um]), (2,), "translation mean")
    std = _array(settings.get("translation_std_um", [cfg.SYSTEM_TRANSLATION_X_STD_um,
        cfg.SYSTEM_TRANSLATION_Y_STD_um]), (2,), "translation standard deviation")
    _require(np.all(std >= 0), "translation standard deviation must be nonnegative")
    rotation_mean = _number(settings.get("rotation_mean_rad", cfg.SYSTEM_ROTATION_MEAN_rad), "rotation mean")
    rotation_std = _number(settings.get("rotation_std_rad", cfg.SYSTEM_ROTATION_STD_rad), "rotation std", minimum=0.)
    z = np.random.default_rng(_seed(seed, label, 1)).standard_normal((samples, 3))
    return dict(coordinates_um=xy[None].tolist(), critical_pad_mask=[True]*len(xy), site_mask=[True],
        threshold_um=_number(settings.get("threshold_um", threshold), "overlay threshold", minimum=0.),
        residual_sigma_um=_number(settings.get("residual_sigma_um", internal["residual_sigma_um"]), "residual sigma", minimum=0.),
        magnification_correction_fraction=_number(settings.get("magnification_correction",
            internal["magnification_correction_fraction"]), "magnification correction", minimum=0., maximum=1.),
        translation_um=(mean+std*z[:, :2])[:, None].tolist(),
        rotation_rad=(rotation_mean+rotation_std*z[:, 2])[:, None].tolist())


def run_stack(entry, library, provenance, progress):
    from W2W.assembly_mechanics import AssemblyExecutor, _reference_runtime
    from W2W.assembly_order import archive_template, content_id

    class PackageStackExecutor(AssemblyExecutor):
        def anneal(self, body):
            complete = len({p.die_index for p in body.placements}) == len(self.births)
            if entry["annealing"] == "batch" and not complete:
                return body, dict(policy="defer_until_all_internal_bonds", duration_s=0.,
                    ages_before_s=dict(body.ages_s), ages_after_s=dict(body.ages_s))
            return super().anneal(body)

    plan, settings = entry["plan"], entry["design"]["mechanics"]
    archive = archive_template(plan)
    archive["provenance"] = dict(solver_id="existing_Cu_history_engine_package_adapter",
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        operator_set_sha256=content_id(provenance), source_transfer_audit="native_formation_and_checkpoints",
        material_clock_audit="native_layer_ages", physics_qualification="UNVALIDATED_2P5D_EXTENSION")
    archive["native_configuration"] = dict(metric=settings["metric"], temporal_profile=settings["temporal_profile"],
        internal_annealing=entry["annealing"], operator_provenance=provenance)
    with _reference_runtime(provenance.get("frozen_preview") is True, library=library):
        executor = PackageStackExecutor(plan, library, temporal_profile=settings["temporal_profile"])
        route, target = plan["routes"][0], archive["routes"][0]
        executor.run_route(route, target, metric=settings["metric"], progress=progress)
        body = executor.bodies[route["final"]["state_id"]]
        before = target["final"]["checkpoint_sha256"]
        abd = executor.export.effective_reference_abd(body.topology, library.cell_area_mm2)
        curve = np.asarray(body.state.generalized_coordinates[3:6])
        face = min(p.z_bottom_mm for p in body.placements)
        strain, audit = elastic_face_strain(abd, curve, executor.footprint, face)
        _require(executor.checkpoint(body) == before, "elastic stack export mutated Cu history")
        elastic = dict(elastic_abd=abd.tolist(), curvature_per_mm=curve.tolist(),
            q6=body.state.generalized_coordinates[:6].tolist(), footprint_mm=list(executor.footprint),
            mating_face="bottom_FRONT", face_z_mm=face, frozen_history_sha256=before,
            affine_face_strain=strain.tolist(), audit=audit,
            stiffness_policy="elastic_reference_condensed_ports_and_bubbles_at_25C_not_plastic_tangent")
    archive["status"] = "COMPLETE"
    return dict(archive=archive, elastic=elastic, **score_stack_archive(entry, archive))


def score_stack_archive(entry, archive):
    """Reused mechanics never reuses another physical stack's placement draws."""
    from W2W.assembly_order import evaluate_archive
    plan = entry["plan"]
    route = plan["routes"][0]
    report = evaluate_archive(plan, archive, entry["overlay"])
    matrices = np.asarray([r["magnification_ppm"] for r in report["routes"][0]["interfaces"]])*1e-6
    # Archive events follow traversal order; overlay draws follow physical interface.
    indices = [event["lower_free"]["die_ids"][-1]-1 for event in route["events"]]
    overlay = dict(entry["overlay"], translation_um=np.asarray(entry["overlay"]["translation_um"])[:, indices],
                   rotation_rad=np.asarray(entry["overlay"]["rotation_rad"])[:, indices])
    probabilities, _ = score_affine(matrices, overlay)
    conditional = np.prod(probabilities, axis=1)
    _require(np.isclose(np.mean(conditional), report["routes"][0]["stack_overlay_yield"], atol=1e-12, rtol=0),
             "conditional internal yield differs from the existing stack scorer")
    return dict(report=report, conditional=conditional)


def run_package(path, *, output_dir=None, plan_only=False, stack_runner=None):
    from mechanics_runtime import configure_runtime_storage, user_run_root, validate_user_write_path
    from W2W.assembly_operator_library import load_prepared_die_library
    from threadpoolctl import threadpool_limits
    configure_runtime_storage()
    started = perf_counter()
    document, entries, ip, amplitudes, bows = prepare_package(path)
    root = Path(output_dir) if output_dir is not None else user_run_root()/"results/packages"
    run = validate_user_write_path(root/document["name"]/(datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid4().hex[:8]))
    run.mkdir(parents=True, mode=0o700)
    _write(run/"INPUT.json", document)
    _write(run/"PLAN.json", dict(schema=SCHEMA, interposer=ip, stacks=[dict(name=e["name"],
        center_mm=e["center"].tolist(), plan=e["plan"], input_provenance=e["provenance"]) for e in entries],
        flow=["complete_each_3D_stack", "hold_interposer_flat", "independently_flatten_align_attach_all_stacks_at_RT",
              "release_after_all_stacks_attached", "package_batch_anneal"],
        package_anneal_mechanics="not_simulated_by_no_additional_yield_loss_assumption",
        post_attachment_yield_policy=POST_ATTACHMENT_POLICY, post_attachment_yield_factor=1.))
    print(f"2.5D package run: {run}", flush=True)
    if plan_only:
        print(f"Planned {len(entries)} stacks and {len(entries)} interposer interfaces; no operators loaded.", flush=True)
        return run
    def progress(message):
        print(message, flush=True)
        with validate_user_write_path(run/"progress.log").open("a", encoding="ascii") as stream:
            stream.write(message+"\n")
    source_paths = [Path(__file__).resolve(), *(ROOT/name for name in (
        "manufacturing_bow.py", "initial_bow_distribution.py", "history_bow_yield.py",
        "D2W/assembly_order.py", "D2W/die_membrane.py", "W2W/assembly_mechanics.py",
        "W2W/assembly_order.py", "W2W/assembly_operator_library.py"))]
    source_pins = {}
    for source in source_paths:
        raw = source.read_bytes()
        source_pins[str(source)] = hashlib.sha256(raw).hexdigest()
        target = validate_user_write_path(run/"source_snapshot"/source.relative_to(ROOT))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
    _write(run/"SOURCE.json", source_pins)
    records, internal, external, other = [], [], [], []
    try:
        with threadpool_limits(limits=1):
            # Affine IP flattening has a linear source term and a quadratic metric term.
            setup_started = perf_counter()
            abd = np.asarray(ip["elastic_abd"])
            curve = np.asarray(ip["native_q6_per_source"])[3:]
            _, audit = elastic_face_strain(abd, curve, ip["footprint_mm"], ip["total_thickness_um"]/2000)
            ip_strain = (amplitudes[:, None, None]*np.asarray(audit["linear_face_strain"])
                + amplitudes[:, None, None]**2*np.asarray(audit["membrane_metric"]))
            attachment_setup_seconds = perf_counter()-setup_started
            last_key, library, operator_provenance = None, None, None
            completed_models = {}
            for index, entry in enumerate(entries):
                progress(f"Stack {index+1}/{len(entries)} {entry['name']}: {entry['annealing']} internal annealing")
                settings = entry["design"]["mechanics"]
                key = json.dumps([settings, entry["plan"]["context"]["geometry"]], sort_keys=True)
                model_key = (key, entry["plan"]["context_id"], entry["plan"]["routes"][0]["request_id"])
                before = perf_counter()
                reused = model_key in completed_models
                if reused:
                    model = completed_models[model_key]
                    progress("Reusing identical completed mechanics; scoring independent placement draws.")
                    stack = dict(model, **score_stack_archive(entry, model["archive"]))
                else:
                    if stack_runner is None and key != last_key:
                        progress(f"Loading selected operators: {settings['operator_package']}")
                        library = None
                        library, operator_provenance = load_prepared_die_library(settings["operator_package"],
                            entry["plan"]["context"]["geometry"], frozen_preview=settings.get("frozen_preview", False))
                        last_key = key
                    stack = (stack_runner or run_stack)(entry, library, operator_provenance, progress)
                    completed_models[model_key] = {k:stack[k] for k in ("archive", "elastic")}
                internal_seconds = perf_counter()-before
                _write(run/f"stack_{index+1}_MECHANICS.json", stack["archive"])
                _write(run/f"stack_{index+1}_YIELD.json", stack["report"])
                before = perf_counter()
                matrices = np.asarray(stack["elastic"]["affine_face_strain"])[None]-ip_strain
                survival, maximum = score_affine(matrices[:, None], entry["external"])
                final_p = survival[:, 0]
                internal.append(stack["conditional"])
                external.append(final_p)
                other.append(entry["other"])
                _write(run/f"stack_{index+1}_ELASTIC.json", stack["elastic"])
                records.append(dict(name=entry["name"], center_mm=entry["center"].tolist(),
                    deterministic_mechanics_reused=reused,
                    internal_order=entry["plan"]["routes"][0]["order"], internal_annealing=entry["annealing"],
                    internal_interfaces=stack["report"]["routes"][0]["interfaces"],
                    internal_overlay_yield=float(np.mean(stack["conditional"])),
                    interposer_interface_overlay_yield=float(np.mean(final_p)), other_yield_factor=entry["other"],
                    attachment_magnification_mean_ppm=(np.mean(matrices, axis=0)*1e6).tolist(),
                    attachment_magnification_min_ppm=(np.min(matrices, axis=0)*1e6).tolist(),
                    attachment_magnification_max_ppm=(np.max(matrices, axis=0)*1e6).tolist(),
                    attachment_residual_magnification_mean_ppm=(np.mean(matrices, axis=0)*1e6
                        *(1-entry["external"]["magnification_correction_fraction"])).tolist(),
                    attachment_max_overlay_nm=float(np.max(maximum)*1000),
                    attachment_settings={k:v for k,v in entry["external"].items() if k not in ("translation_um", "rotation_rad")},
                    internal_wall_seconds=internal_seconds, attachment_wall_seconds=perf_counter()-before))
                progress(f"{entry['name']}: internal overlay yield {records[-1]['internal_overlay_yield']:.6f}; "
                         f"interposer interface {records[-1]['interposer_interface_overlay_yield']:.6f}")
        internal, external = np.stack(internal, axis=1), np.stack(external, axis=1)
        summary, joint = compound_package(internal, external, other)
        with validate_user_write_path(run/"SAMPLES.npz").open("xb") as stream:
            np.savez_compressed(stream, interposer_source=amplitudes, interposer_bow_um=bows,
                internal_stack_survival=internal, interposer_interface_survival=external, package_survival=joint)
        result = dict(schema=SCHEMA, status="COMPLETE_OVERLAY_ONLY", **summary, stacks=records,
            interposer=ip, joint_samples=len(amplitudes), seed=document.get("seed", 20261118),
            conditional_interface_independence=True, shared_interposer_source=True,
            room_temperature_attachment="independent_ideal_flattening_with_frozen_Cu_history",
            interposer_registration="affine_field_locally_recentered_at_each_stack; no nonaffine_position_effect",
            package_batch_anneal="not_simulated_no_additional_yield_loss_assumed_not_an_extra_overlay_interface",
            post_attachment_yield_policy=POST_ATTACHMENT_POLICY, post_attachment_yield_factor=1.,
            other_yield_mechanisms="only_user_supplied_other_yield_factor; not computed here",
            physical_readiness=False, independent_package_validation=False,
            attachment_setup_wall_seconds=attachment_setup_seconds, wall_seconds=perf_counter()-started)
        _write(run/"RESULT.json", result)
        with validate_user_write_path(run/"SUMMARY.csv").open("x", newline="", encoding="ascii") as stream:
            writer = csv.writer(stream)
            writer.writerow(["stack", "internal_order", "internal_overlay_yield", "interposer_interface_overlay_yield",
                             "internal_seconds", "attachment_seconds", "package_overlay_yield"])
            for r in records:
                writer.writerow([r["name"], r["internal_order"], r["internal_overlay_yield"],
                    r["interposer_interface_overlay_yield"], r["internal_wall_seconds"],
                    r["attachment_wall_seconds"], result["package_overlay_yield"]])
        progress(f"Package overlay yield: {result['package_overlay_yield']:.6f}; elapsed {result['wall_seconds']:.2f} s")
    except KeyboardInterrupt:
        _write(run/"CANCELLED.json", dict(status="CANCELLED", reason="KeyboardInterrupt",
            completed_stacks=len(records), accuracy_verdict="not_assigned"))
        raise
    except Exception as error:
        _write(run/"FAILED.json", dict(error=type(error).__name__, message=str(error)))
        raise
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="2.5D package YAML")
    parser.add_argument("--output-dir", type=Path, help="Results root; defaults below YAP_RUN_ROOT")
    parser.add_argument("--plan-only", action="store_true", help="Validate package without loading mechanics operators")
    args = parser.parse_args(argv)
    run_package(args.config, output_dir=args.output_dir, plan_only=args.plan_only)


if __name__ == "__main__":
    main()
