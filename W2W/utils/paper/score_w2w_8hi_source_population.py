#!/usr/bin/env python3
"""Generate native A1-A7 W2W overlay yields with the current source model."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from time import perf_counter

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mechanics_runtime import configure_runtime_storage, mechanics_work_directory, validate_write_path

configure_runtime_storage()
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFY_ROOT = REPO_ROOT.parent / "Ansys_Verify" / "12_stack_model_validation"
OUTPUT_ROOT = REPO_ROOT / "W2W" / "output" / "paper_figures"
DEFAULT_OUTPUT = OUTPUT_ROOT / "W2W_SITE_OVERLAY_YIELD_8HI_GAMMA_0p9_SOURCE_CV_0p2.csv"
ANCHOR_CACHE = OUTPUT_ROOT / "W2W_8HI_SOURCE_AFFINE_TENSORS.npz"
ANCHOR_DIR = OUTPUT_ROOT / "w2w_8hi_source_anchors"

WAFER_COUNT = 8
INTERFACE_COUNT = WAFER_COUNT - 1
TARGET_TOTAL_UM = 50
GAMMA = 0.9
SOURCE_CV = 0.2
SOURCE_MEAN = 0.8125
SOURCE_STD = SOURCE_MEAN * SOURCE_CV
SOURCE_DELTA = 0.1
SAMPLES = 32768
SEED = 2026091601
QUADRATURE_ORDER = 12

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force-anchors", action="store_true")
    parser.add_argument("--anchor-indices", type=int, nargs="*")
    args = parser.parse_args()
    args.output = validate_write_path(args.output)
    if not 0.0 <= args.gamma <= 1.0:
        parser.error("--gamma must lie in [0, 1]")
    return args


def _install_generalized_process(api, runner, *, wafer_count: int) -> None:
    cycles = wafer_count - 1
    original_specification = api.specification

    def specification(target):
        spec = dict(original_specification(target))
        spec.update(
            wafer_count=wafer_count,
            cycles=cycles,
            final_height_um=775.0 + cycles * (target + 1.5),
            process_id="w2w_full775_8hi_source_population_20260916_v1",
        )
        return spec

    def build_process(library, target, births=None):
        births = births or {}
        removable = 785 - target
        snapshots, order, current = {}, [], []

        def append_bottom_wafer(index, z):
            recipe = [
                ("bulk", "bulk_725"),
                ("tsv", api.tsv_key(38.5)),
                ("beol", "beol"),
                ("front_bi", "bi"),
            ]
            start, values = z, []
            for role, key in recipe:
                placement = api._placement(
                    library,
                    name=f"w{index + 1}_{role}",
                    operator_key=key,
                    die_index=index,
                    role=role,
                    z_bottom_mm=z,
                    thickness_orientation=1,
                    birth_reference_z_mm=start + 0.775 / 2,
                )
                values.append(placement)
                z = placement.z_top_mm
            if not np.isclose(z - start, 0.775, atol=1e-12, rtol=0):
                raise ValueError("bottom wafer thickness does not close")
            return values

        def append_full_incoming(index, z):
            start, values = z, []
            recipe = [
                ("back_bi", "bi"),
                ("tsv", api.tsv_key(target - 11.5)),
                ("removable_bulk", f"bulk_{removable}"),
            ]
            for role, key in recipe:
                placement = api._placement(
                    library,
                    name=f"w{index + 1}_{role}",
                    operator_key=key,
                    die_index=index,
                    role=role,
                    z_bottom_mm=z,
                    thickness_orientation=-1 if role == "back_bi" else 1,
                    birth_reference_z_mm=start + 0.775 / 2,
                )
                values.append(placement)
                z = placement.z_top_mm
            if not np.isclose(z - start, 0.775, atol=1e-12, rtol=0):
                raise ValueError("incoming wafer thickness does not close")
            return values

        def append_post_thin_frontside(index, z):
            values = []
            for role, key in (("beol", "beol"), ("front_bi", "bi")):
                placement = api._placement(
                    library,
                    name=f"w{index + 1}_{role}",
                    operator_key=key,
                    die_index=index,
                    role=role,
                    z_bottom_mm=z,
                    thickness_orientation=1,
                    birth_reference_z_mm=None,
                )
                values.append(placement)
                z = placement.z_top_mm
            return values

        def save(event):
            snapshots[event] = tuple(current)
            order.append(event)

        current.extend(append_bottom_wafer(0, 0.0))
        current.extend(append_full_incoming(1, current[-1].z_top_mm))
        for cycle in range(1, cycles + 1):
            save(f"C{cycle}")
            if current[-1].role != "removable_bulk":
                raise ValueError("only the exposed incoming bulk may be thinned")
            current.pop()
            current.extend(append_post_thin_frontside(cycle, current[-1].z_top_mm))
            save(f"T{cycle}")
            if cycle == cycles:
                break
            save(f"P{cycle}")
            current.extend(append_full_incoming(cycle + 1, current[-1].z_top_mm))
            save(f"A{cycle + 1}")

        for items in snapshots.values():
            for lower, upper in zip(items, items[1:]):
                if not np.isclose(
                    lower.z_top_mm, upper.z_bottom_mm, atol=1e-12, rtol=0
                ):
                    raise ValueError("gap or overlap in wafer ledger")
        expected_height_um = specification(target)["final_height_um"]
        if not np.isclose(
            current[-1].z_top_mm * 1000.0,
            expected_height_um,
            atol=1e-9,
            rtol=0,
        ):
            raise ValueError("final height does not close")
        return api.V2ProductionSequentialProcess(
            tuple(order),
            {
                event: api._topology(library, values, births)
                for event, values in snapshots.items()
            },
            snapshots,
            0.0,
            cycles,
        )

    def source_birth(process, target, amplitudes):
        from manufacturing_bow import ManufacturingLayer, clpt_source_response
        from exp12_natural_bow_input import normalize_initial_bows
        from exp12_ofat_case_study_20260913 import OFATStudy, FrozenV2Materials

        study = OFATStudy()
        values = np.asarray(amplitudes, dtype=float)
        if values.shape != (wafer_count,) or not np.isfinite(values).all():
            raise ValueError("one finite source amplitude per wafer is required")
        materials = FrozenV2Materials()(study.cases["baseline"])
        layers = []
        for placement in process.placements["C1"]:
            if placement.die_index != 0:
                continue
            kind = "tsv_si" if placement.role == "tsv" else placement.operator_key
            is_bulk = placement.role == "bulk"
            material = (
                {
                    "e_mpa": api.SI["elastic_modulus_mpa"],
                    "nu": api.SI["poisson_ratio"],
                }
                if is_bulk
                else materials[kind]
            )
            layers.append(
                ManufacturingLayer(
                    placement.name,
                    (placement.z_top_mm - placement.z_bottom_mm) * 1000.0,
                    material["e_mpa"],
                    material["nu"],
                    eigenstrain_per_amplitude=1.0 if placement.role == "beol" else 0.0,
                    is_bulk_silicon=is_bulk or placement.role == "tsv",
                )
            )
        reference = clpt_source_response(tuple(layers))
        unit = np.repeat(
            np.asarray(reference["natural_birth_q6_per_amplitude"])[None, :],
            wafer_count,
            axis=0,
        )
        physical = (
            values
            * study.calibration.source_sign
            * study.calibration.nominal_amplitude
        )
        bows = -1000.0 * (physical[:, None] * unit)[:, 3:5] * 300.0**2 / 8.0
        payload = {
            "schema": "exp12.natural_initial_bow.v1",
            "frame": "common_global_xyz_z_thickness",
            "bow_convention": "center_minus_mid_edge",
            "die_edge_bows_um": bows.tolist(),
            "engineering_twist_mm_inv": [0.0] * wafer_count,
        }
        normalized = normalize_initial_bows(
            payload, die_count=wafer_count, footprint_mm=(300.0, 300.0)
        )
        mean = (
            -1000.0
            * unit[:, 3:5]
            * (study.calibration.source_sign * study.calibration.nominal_amplitude)
            * 300.0**2
            / 8.0
        )
        std = 0.1 * np.abs(mean)
        return {
            "solver_input": payload,
            "initial_bow": normalized,
            "source_amplitudes": values.tolist(),
            "calibration_sha256": study.calibration.signature,
            "incoming_mean_bow_um": mean.tolist(),
            "incoming_std_bow_um": std.tolist(),
            "incoming_mean_minus_3sigma_um": (mean - 3.0 * std).tolist(),
            "incoming_mean_plus_3sigma_um": (mean + 3.0 * std).tolist(),
            "material_sha256": study.materials["baseline"]["provenance"],
            "source_distribution": "calibrated independent wafer sources",
            "source_transfer": "curvature-only stress-free natural reference",
            "thinning_birth_policy": "never recompute natural shape after removal",
        }

    api.specification = specification
    api.build_process = build_process
    api.source_birth = source_birth

    import exp12_ofat_runtime_20260913 as runtime

    original_schedules = runtime.continuous_schedules

    def schedules(_ignored_cycles, anneal_temperature):
        return original_schedules(cycles, anneal_temperature)

    runtime.continuous_schedules = schedules
    runner.api = api


def _formation_tensors(history: dict, scorer) -> np.ndarray:
    formations = history["formations"]
    if not isinstance(formations, (tuple, list)) or len(formations) != INTERFACE_COUNT:
        raise ValueError(
            f"Expected {INTERFACE_COUNT} ordered formations, received "
            f"{len(formations) if isinstance(formations, (tuple, list)) else 'invalid'}"
        )
    tensors = []
    for index, formation in enumerate(formations, 1):
        if not isinstance(formation, dict) or formation.get("event") != f"A{index}":
            raise ValueError("formations must be ordered A1..A7 without gaps")
        bodies = formation.get("bodies_old_then_incoming")
        if not isinstance(bodies, (tuple, list)) or len(bodies) != 2:
            raise ValueError("each formation requires old and incoming free bodies")
        metrics, increments = [], []
        for body in bodies:
            q6 = np.asarray(body["free_q6"], dtype=np.float64)
            increment = np.asarray(body["flatten_face_increment"], dtype=np.float64)
            abd = np.asarray(body["reference_abd_N_mm_units"], dtype=np.float64)
            if q6.shape != (6,) or increment.shape != (3,) or abd.shape != (6, 6):
                raise ValueError("invalid formation body tensor dimensions")
            metric, _audit = scorer.disk_membrane_metric(abd[:3, :3], q6[3:])
            metrics.append(metric)
            increments.append(increment)
        linear = increments[0] - increments[1]
        linear_tensor = np.asarray(
            [[linear[0], linear[2] / 2.0], [linear[2] / 2.0, linear[1]]],
            dtype=np.float64,
        )
        tensors.append(linear_tensor + metrics[0] - metrics[1])
    values = np.asarray(tensors, dtype=np.float64)
    if values.shape != (INTERFACE_COUNT, 2, 2):
        raise ValueError(f"Expected seven affine tensors, received {values.shape}")
    return values


def build_affine_tensors(
    force: bool,
    anchor_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    if ANCHOR_CACHE.exists() and not force:
        with np.load(ANCHOR_CACHE, allow_pickle=False) as archive:
            mean = archive["mean"]
            sensitivity = archive["sensitivity"]
            runtimes = archive["runtimes"]
            early_mean_defect_ppm = float(archive["early_mean_defect_ppm"])
            early_sensitivity_defect_ppm = float(
                archive["early_sensitivity_defect_ppm"]
            )
        return mean, sensitivity, {
            "runtimes_s": runtimes.tolist(),
            "cached": True,
            "early_mean_defect_ppm": early_mean_defect_ppm,
            "early_sensitivity_defect_ppm": early_sensitivity_defect_ppm,
        }

    requested = set(range(WAFER_COUNT + 1)) if anchor_indices is None else set(anchor_indices)
    if any(index < 0 or index > WAFER_COUNT for index in requested):
        raise ValueError(f"anchor indices must lie in [0, {WAFER_COUNT}]")
    scratch = mechanics_work_directory()
    validate_write_path(ANCHOR_DIR).mkdir(parents=True, exist_ok=True)
    if force:
        for index in requested:
            path = ANCHOR_DIR / f"anchor_{index}.npz"
            if path.exists():
                path.unlink()

    sys.path.insert(0, str(VERIFY_ROOT))
    try:
        import exp12_w2w_window_20260914 as api
        import exp12_w2w_window_overlay_20260914 as scorer
        import run_exp12_w2w_window_20260914 as runner
        from exp12_ofat_compact_solver_20260913 import compact_solver
        from exp12_ofat_reference_dedup_20260913 import deduplicated_reference_cache
        from exp12_ofat_runtime_20260913 import record_reference_matrices
        from threadpoolctl import threadpool_limits

        _install_generalized_process(api, runner, wafer_count=WAFER_COUNT)
        with tempfile.TemporaryDirectory(prefix="yap_w2w8_package_", dir=scratch) as temporary:
            package_root = Path(temporary)
            package = runner.package_for(
                TARGET_TOTAL_UM, package_root, lambda **_values: None
            )
            library, _bulk = runner.load_library(package, TARGET_TOTAL_UM)
            amplitudes = [np.ones(WAFER_COUNT)]
            for wafer in range(WAFER_COUNT):
                perturbed = np.ones(WAFER_COUNT)
                perturbed[wafer] += SOURCE_DELTA
                amplitudes.append(perturbed)

            missing = [
                index
                for index in sorted(requested)
                if not (ANCHOR_DIR / f"anchor_{index}.npz").exists()
            ]
            for anchor_index in missing:
                values = amplitudes[anchor_index]
                print(
                    f"8-Hi mechanics anchor {anchor_index}/{len(amplitudes) - 1}",
                    flush=True,
                )
                with tempfile.TemporaryDirectory(
                    prefix=f"yap_w2w8_anchor{anchor_index}_", dir=scratch
                ) as run_dir:
                    started = perf_counter()
                    with threadpool_limits(limits=1, user_api="blas"):
                        with (
                            record_reference_matrices(),
                            deduplicated_reference_cache(),
                            compact_solver(),
                        ):
                            history = runner.predict(
                                library,
                                TARGET_TOTAL_UM,
                                values.tolist(),
                                Path(run_dir),
                                lambda **_progress: None,
                                elastic=False,
                                cycles=INTERFACE_COUNT,
                            )
                    runtime_s = perf_counter() - started
                    tensor = _formation_tensors(history, scorer)
                    temporary_anchor = ANCHOR_DIR / f"anchor_{anchor_index}.partial.npz"
                    np.savez_compressed(
                        temporary_anchor,
                        tensor=tensor,
                        runtime_s=np.asarray(runtime_s),
                        amplitudes=values,
                    )
                    temporary_anchor.replace(ANCHOR_DIR / f"anchor_{anchor_index}.npz")

        anchor_paths = [ANCHOR_DIR / f"anchor_{index}.npz" for index in range(WAFER_COUNT + 1)]
        if not all(path.exists() for path in anchor_paths):
            completed = [index for index, path in enumerate(anchor_paths) if path.exists()]
            print(f"Completed anchor subset: {completed}")
            return None
        tensors, runtimes = [], []
        for path in anchor_paths:
            with np.load(path, allow_pickle=False) as archive:
                tensors.append(archive["tensor"])
                runtimes.append(float(archive["runtime_s"]))
        tensors = np.asarray(tensors)
        mean = tensors[0]
        sensitivity = (tensors[1:] - mean[None]) / SOURCE_DELTA
        from score_w2w_source_population import mean_history, fit_source_sensitivity
        reference_history = json.loads(
            (
                mean_history()
            ).read_text(encoding="ascii")
        )
        reference_mean = np.asarray(
            [
                record["affine_magnification_tensor"]
                for record in scorer._interface_records(reference_history["formations"])
            ],
            dtype=np.float64,
        )
        early_mean_defect_ppm = float(
            np.max(np.abs(mean[:3] - reference_mean)) * 1.0e6
        )
        reference_sensitivity, _fit_diagnostics = fit_source_sensitivity(scorer)
        early_sensitivity_defect_ppm = float(
            np.max(np.abs(sensitivity[:4, :3] - reference_sensitivity)) * 1.0e6
        )
        if early_mean_defect_ppm > 1.0e-8:
            raise RuntimeError(
                "8-Hi prefix does not reproduce the admitted 4-Hi mean history: "
                f"{early_mean_defect_ppm:g} ppm"
            )
        # The legacy 4-Hi sensitivity is a regression over older-process
        # random anchors transferred onto the latest mean history.  Preserve
        # its difference as a diagnostic; the direct current-process 8-Hi
        # anchors are the internally consistent source response used here.
        ANCHOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ANCHOR_CACHE,
            mean=mean,
            sensitivity=sensitivity,
            runtimes=np.asarray(runtimes),
            early_mean_defect_ppm=early_mean_defect_ppm,
            early_sensitivity_defect_ppm=early_sensitivity_defect_ppm,
        )
        return mean, sensitivity, {
            "runtimes_s": runtimes,
            "cached": False,
            "early_mean_defect_ppm": early_mean_defect_ppm,
            "early_sensitivity_defect_ppm": early_sensitivity_defect_ppm,
            "early_sensitivity_reference_scope": (
                "legacy 4-Hi fitted/transferred response; diagnostic only"
            ),
        }
    finally:
        if sys.path and sys.path[0] == str(VERIFY_ROOT):
            sys.path.pop(0)


def score_population(
    mean: np.ndarray,
    sensitivity: np.ndarray,
    *,
    gamma: float,
    samples: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict]:
    sys.path.insert(0, str(VERIFY_ROOT))
    try:
        import exp12_w2w_window_overlay_20260914 as scorer
        import run_exp12_fixed_bump_map_half_anchor_20260914 as fixed_score
    finally:
        sys.path.pop(0)

    # The frozen overlay helper unnecessarily checks the historical chip
    # density while constructing pad geometry.  Satisfy that legacy preflight
    # in memory only; chip fabrication yield is not part of this overlay scorer.
    import exp12_ofat_yield_20260913 as yield_model
    from omegaconf import OmegaConf

    original_make_yield_config = yield_model.make_yield_config

    def overlay_geometry_config(case, supplied=None):
        if supplied is None:
            supplied = OmegaConf.load(
                yield_model.ROOT / "YAP/D2W/configs/GOLDEN.yaml"
            ).d2w_simulation
            supplied.CHIP_DEFECT_DENSITY_PER_CM2 = 0.1
        return original_make_yield_config(case, supplied)

    scorer._yield_inputs.cache_clear()
    yield_model.make_yield_config = overlay_geometry_config
    try:
        inputs = scorer._yield_inputs()
    finally:
        yield_model.make_yield_config = original_make_yield_config
        scorer._yield_inputs.cache_clear()
    sites = np.asarray(inputs["sites"], dtype=np.float64)
    fixed = fixed_score.fixed_center_grid(inputs["geometry"], QUADRATURE_ORDER)
    source_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    placement_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    source_z = source_rng.standard_normal((samples, WAFER_COUNT))
    placement_z = placement_rng.standard_normal((samples, INTERFACE_COUNT, 3))
    amplitudes = SOURCE_MEAN + SOURCE_STD * source_z
    tensors = mean[None] + np.einsum(
        "nw,weij->neij", amplitudes - 1.0, sensitivity, optimize=True
    )
    tensors *= 1.0 - gamma
    translations = placement_z[..., :2] * scorer.TRANSLATION_SIGMA_UM
    rotations = placement_z[..., 2] * scorer.ROTATION_SIGMA_RAD

    interface_sum = np.zeros((INTERFACE_COUNT, len(sites)), dtype=np.float64)
    prefix_sum = np.zeros_like(interface_sum)
    draw_chunk = min(int(scorer.DRAW_CHUNK), 256)
    for first in range(0, samples, draw_chunk):
        stop = min(samples, first + draw_chunk)
        matrix = tensors[first:stop] + rotations[first:stop, :, None, None] * scorer.J
        translation = np.einsum("neij,sj->nesi", matrix, sites, optimize=True)
        translation += translations[first:stop, :, None]
        probabilities = fixed_score.fixed_map_survival(
            matrix[:, :, None],
            translation,
            fixed,
            threshold_um=inputs["threshold"],
            sigma_um=scorer.PAD_NOISE_SIGMA_UM,
            batch_size=2048,
        )
        interface_sum += probabilities.sum(axis=0)
        prefix_sum += np.cumprod(probabilities, axis=1).sum(axis=0)

    result = {
        "sites": sites,
        "interface_site": interface_sum / samples,
        "prefix_site": prefix_sum / samples,
    }
    report = {
        "schema": "yap.w2w_8hi_source_population_overlay.v1",
        "wafer_count": WAFER_COUNT,
        "interface_count": INTERFACE_COUNT,
        "gamma": gamma,
        "source_amplitude_mean": SOURCE_MEAN,
        "source_amplitude_std": SOURCE_STD,
        "source_cv": SOURCE_CV,
        "samples": samples,
        "seed": seed,
        "quadrature_order": QUADRATURE_ORDER,
        "threshold_um": float(inputs["threshold"]),
        "random_pad_noise_sigma_um": float(scorer.PAD_NOISE_SIGMA_UM),
        "translation_sigma_um": float(scorer.TRANSLATION_SIGMA_UM),
        "rotation_sigma_rad": float(scorer.ROTATION_SIGMA_RAD),
        "critical_pad_count": int(fixed["critical_pad_count"]),
        "site_count": len(sites),
        "affine_method": "native 8-wafer mean plus one-sided source sensitivities",
        "source_delta": SOURCE_DELTA,
    }
    return result, report


def write_outputs(output: Path, result: dict[str, np.ndarray], report: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    sites = result["sites"]
    interface = result["interface_site"]
    cumulative = result["prefix_site"]
    fields = ["site_id", "x_mm", "y_mm", "radius_mm"]
    fields += [f"A{i}_interface_overlay_yield" for i in range(1, 8)]
    fields += [f"A{i}_cumulative_overlay_yield" for i in range(1, 8)]
    with output.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for site_index, (x_um, y_um) in enumerate(sites):
            row = {
                "site_id": site_index,
                "x_mm": x_um / 1000.0,
                "y_mm": y_um / 1000.0,
                "radius_mm": np.hypot(x_um, y_um) / 1000.0,
            }
            for interface_index in range(INTERFACE_COUNT):
                row[f"A{interface_index + 1}_interface_overlay_yield"] = interface[
                    interface_index, site_index
                ]
                row[f"A{interface_index + 1}_cumulative_overlay_yield"] = cumulative[
                    interface_index, site_index
                ]
            writer.writerow(row)
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="ascii"
    )


def main() -> int:
    args = parse_args()
    started = perf_counter()
    affine = build_affine_tensors(args.force_anchors, args.anchor_indices)
    if affine is None:
        return 0
    mean, sensitivity, mechanics = affine
    result, report = score_population(
        mean,
        sensitivity,
        gamma=args.gamma,
        samples=args.samples,
        seed=args.seed,
    )
    report["mechanics"] = mechanics
    report["runtime_s"] = perf_counter() - started
    report["interface_mean_yield"] = result["interface_site"].mean(axis=1).tolist()
    report["cumulative_mean_yield"] = result["prefix_site"].mean(axis=1).tolist()
    write_outputs(args.output.resolve(), result, report)
    print(f"Saved: {args.output.resolve()}")
    for index, value in enumerate(report["cumulative_mean_yield"], 1):
        print(f"A{index} cumulative overlay yield: {value:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
