#!/usr/bin/env python3
"""Build a diagnostic compact layer package from a corrected native capture."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
from mechanics_runtime import configure_runtime_storage, mechanics_work_directory, validate_write_path

configure_runtime_storage()
import numpy as np


ROOT = Path(__file__).resolve().parent
ANSYS_VALIDATION = ROOT.parent / "Ansys_Verify" / "12_stack_model_validation"
CORRECTED_TIER_CONTRACT = "same_orientation_tsv_beol_frontbi_upper_backbi_20260914_v1"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def _safe_output(path):
    path = validate_write_path(path)
    work = mechanics_work_directory().parent
    if not path.is_relative_to(work):
        raise ValueError("candidate output must be under the approved YAP_WORK_DIR")
    return path


def _load_training(capture_dir, package):
    capture_dir = Path(capture_dir).resolve()
    manifest_path = capture_dir / "MANIFEST.json"
    manifest = _read(manifest_path)
    if (
        manifest.get("schema") != "exp12.ofat_coordinate_capture.v1"
        or manifest.get("metrics", {}).get("state") != "COMPLETE"
        or manifest.get("geometry", {}).get("tier_order_contract")
        != CORRECTED_TIER_CONTRACT
    ):
        raise ValueError("training capture is not a complete corrected-tier history")
    package_inputs = {
        str(Path(value).resolve()) for value in package["datasets"].values()
    } | {str(Path(package["port_basis"]).resolve())}
    if not package_inputs.issubset(manifest.get("input_sha256", {})):
        raise ValueError("capture is not bound to every native package input")

    training = defaultdict(lambda: [[], []])
    sources = {}
    for record in manifest["rows"]:
        filename = record["file"]
        if Path(filename).name != filename:
            raise ValueError("capture row must be a direct child")
        with np.load(capture_dir / filename, allow_pickle=False) as arrays:
            for slot, key in enumerate(arrays["operator_keys"].astype(str)):
                if key not in package["datasets"]:
                    raise ValueError(f"capture uses an unknown operator: {key}")
                digest = package["dataset_sha256"][key]
                sources[digest] = (key, Path(package["datasets"][key]))
                for birth in (False, True):
                    training[digest][0].append(
                        arrays[
                            f"{'birth_macro' if birth else 'local_macro'}_{slot:03d}"
                        ].copy()
                    )
                    training[digest][1].append(
                        arrays[f"birth_bubble_{slot:03d}"].copy()
                        if birth
                        else arrays[f"bubble_{slot:03d}"].copy()
                    )
    values = {
        digest: [(np.asarray(macro), np.asarray(bubble))]
        for digest, (macro, bubble) in training.items()
    }
    return manifest_path, manifest, values, sources


def build(capture_dir, native_package, output, *, port_rank, bubble_rank):
    if str(ANSYS_VALIDATION) not in sys.path:
        sys.path.insert(0, str(ANSYS_VALIDATION))
    from build_exp12_ofat_layer_pod_20260913 import (
        common_port_basis,
        load_parent_port_basis,
        transform_layer,
        trial_projection,
        validate_parent_layer_basis,
    )
    from exp12_layer_rom_dataset import LayerROMDataset

    output = _safe_output(output)
    output.mkdir(parents=True, exist_ok=False)
    native_path = Path(native_package).resolve()
    package = _read(native_path)
    if (
        package.get("schema") != "exp12.ofat.geometry_package.v1"
        or package.get("state") != "READY_ENGINEERING_PREVIEW"
        or package.get("ready") is not True
    ):
        raise ValueError("native package is not ready")
    manifest_path, manifest, training, sources = _load_training(capture_dir, package)
    if set(sources) != set(package["dataset_sha256"].values()):
        raise ValueError("capture does not exercise every native package layer")

    started = time.perf_counter()
    parent = load_parent_port_basis(package, manifest)
    native_rank = parent["modes"].shape[1]
    port_rows = [
        macro[:, start:start + native_rank]
        for histories in training.values()
        for macro, _ in histories
        for start in (6, 6 + native_rank)
    ]
    projection, port_audit = common_port_basis(parent, port_rows, port_rank)
    rank = projection.shape[1]
    basis_path = output / "COMMON_PORT_POD.npz"
    np.savez(
        basis_path,
        coords_mm=parent["coords_mm"],
        modes=parent["modes"] @ projection,
        reflection_matrix=projection.T @ parent["reflection_matrix"] @ projection,
        inplane_swap_matrix=projection.T
        @ parent.get("inplane_swap_matrix", np.eye(native_rank))
        @ projection,
        rigid_mode_indices=np.arange(len(parent["rigid_mode_indices"])),
        coefficient_basis=projection,
        rigid_mode_directions=parent["rigid_mode_directions"],
    )
    macro_map = np.zeros((6 + 2 * native_rank, 6 + 2 * rank))
    macro_map[:6, :6] = np.eye(6)
    macro_map[6:6 + native_rank, 6:6 + rank] = projection
    macro_map[6 + native_rank:, 6 + rank:] = projection

    datasets, hashes, layer_audits = {}, {}, {}
    common_scale = None
    for digest, (key, source_path) in sources.items():
        if _sha256(source_path) != digest:
            raise ValueError(f"native layer changed: {key}")
        dataset = LayerROMDataset.load(source_path)
        contract = validate_parent_layer_basis(dataset, package["port_basis_sha256"], native_rank)
        if common_scale is None:
            common_scale = np.r_[
                dataset.macro_coordinate_scale[:6],
                np.full(2 * rank, np.median(dataset.macro_coordinate_scale[6:])),
            ]
        transform, audit = trial_projection(
            dataset.background_stiffness,
            dataset.background_thermal_load,
            macro_map,
            training[digest],
            bubble_rank,
        )
        provenance = {
            "parent_file": str(source_path),
            "parent_file_sha256": digest,
            "parent_port_basis_contract": contract,
            "training_cases": ["yap_corrected_native_c1"],
            "training_draws": ["seed7"],
            "holdout_draws_read": [],
            "tier_order_contract": CORRECTED_TIER_CONTRACT,
            "method": "m=M*a; b=-Kbb^-1*Kbm*M*a+U*z",
            **audit,
        }
        reduced = transform_layer(
            dataset, transform, macro_map.shape[1], common_scale,
            basis_path, provenance,
        )
        target = output / f"{key}_LAYER.npz"
        reduced.save(target)
        np.savez(output / f"{key}_PROJECTION.npz", trial_projection=transform)
        datasets[key] = str(target)
        hashes[key] = _sha256(target)
        layer_audits[key] = {
            **audit,
            "native_coordinates": dataset.macro_size + dataset.bubble_size,
            "compact_coordinates": reduced.macro_size + reduced.bubble_size,
            "cu_points": reduced.selected_cu_point_count,
        }
        del dataset, reduced, transform

    candidate = deepcopy(package)
    candidate.update(
        schema="yap.corrected_compact_geometry_package.v1",
        state="CANDIDATE_NOT_VALIDATED",
        ready=False,
        computationally_ready=True,
        rank=rank,
        port_basis=str(basis_path),
        port_basis_sha256=_sha256(basis_path),
        datasets=datasets,
        dataset_sha256=hashes,
        prepared={},
        parent_package_sha256=_sha256(native_path),
        coordinate_reduction="corrected_tier_common_port_POD_static_lifted_bubble_POD",
        whole_stack_response_fit=False,
        corrected_tier_order=True,
        tier_order_contract=CORRECTED_TIER_CONTRACT,
        training_capture_manifest=str(manifest_path),
        training_capture_manifest_sha256=_sha256(manifest_path),
        port_audit=port_audit,
        layer_audits=layer_audits,
        build_seconds=time.perf_counter() - started,
    )
    package_path = output / "PACKAGE.json"
    _write(package_path, candidate)
    _write(output / "PACKAGE_SHA256.json", {"sha256": _sha256(package_path)})
    return package_path, candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--native-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port-rank", type=int, default=48)
    parser.add_argument("--bubble-rank", type=int, default=32)
    args = parser.parse_args()
    if args.port_rank < 11 or args.bubble_rank < 1:
        raise ValueError("invalid compact ranks")
    path, package = build(
        args.capture, args.native_package, args.output,
        port_rank=args.port_rank, bubble_rank=args.bubble_rank,
    )
    print(json.dumps({
        "package": str(path),
        "rank": package["rank"],
        "build_seconds": package["build_seconds"],
        "layer_audits": package["layer_audits"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
