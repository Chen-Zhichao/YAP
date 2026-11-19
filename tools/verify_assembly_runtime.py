"""Verify an exported p2 runtime without access to its historical project."""

import argparse
import gc
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mechanics_runtime import configure_runtime_storage

ROOT = Path(__file__).resolve().parents[1]


def isolate_original_storage():
    projects = tuple(Path(value) for value in (
        "/w/ee.00/puneet/zhichao/Desktop/YAP_STACK",
        "/u1/ee/zhichao/Desktop/YAP_STACK",
        "/app/nanocad/projects/personal/zchen/YAP_STACK",
    ))
    def audit(event, args):
        if event == "subprocess.Popen":
            command = Path(args[0]).name
            library_query = command == "ldconfig" and list(args[1][1:]) == ["-p"]
            cpu_query = command == "lscpu" and not args[1][1:]
            if command not in ("fc-list", "fc-match") and not library_query and not cpu_query:
                raise RuntimeError("portable verification must not launch solvers or external scripts")
            return
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
        if any(path.is_relative_to(project) for project in projects) and not path.is_relative_to(ROOT):
            raise RuntimeError(f"historical project access denied: {path}")
    sys.addaudithook(audit)


def check_prepared_library(library, inputs, total):
    import numpy as np
    from tools.package_assembly_runtime import PREPARED_BASIS_SHA256, digest
    from W2W.assembly_operator_library import _prepared_json

    if library.port_rank_per_face != 64 or inputs.get("port_rank_per_face") != 64:
        raise ValueError("prepared delivery requires native64 operators")
    expected = {"bi": .0015, "beol": .010, "retained_si": (total - 13.) / 1000.}
    if set(library.operators) != set(expected) or set(library.thickness_mm) != set(expected):
        raise ValueError("prepared delivery requires exactly three operators and no cap")
    for key, thickness in expected.items():
        if not np.isclose(library.thickness_mm[key], thickness, rtol=0., atol=1e-12):
            raise ValueError(f"prepared delivery {key} thickness mismatch")
    if inputs.get("port_basis_sha256") != PREPARED_BASIS_SHA256:
        raise ValueError("prepared delivery port basis mismatch")
    receipt_path = inputs.get("operator_build_receipt")
    expected_hash = inputs.get("input_sha256", {}).get(receipt_path)
    if not receipt_path or expected_hash is None or digest(Path(receipt_path)) != expected_hash:
        raise ValueError("prepared delivery must bind its audited receipt")
    receipt = _prepared_json(receipt_path, {}, expected_hash)
    if receipt.get("all_cu_points_retained") is not True:
        raise ValueError("prepared delivery must verify retention of all Cu integration points")
    aliases = inputs.get("source_operator_keys", {})
    if aliases != {"bi": "bi", "beol": "beol", "retained_si": f"tsv_si_h{total - 13:g}"}:
        raise ValueError("prepared delivery source operator keys mismatch")
    for key, source in aliases.items():
        count = receipt.get("datasets", {}).get(source, {}).get("cu_point_count")
        operator = library.operators[key]
        weights = np.asarray(operator.weights_mm3)
        if (type(count) is not int or count <= 0 or operator.cu_point_count != count
                or weights.shape != (count,) or not np.isfinite(weights).all() or np.any(weights <= 0.)
                or operator.macro_size != 134):
            raise ValueError(f"prepared delivery {key} all-Gauss Cu count mismatch")


def verify(*, load_operators=False, prepared_operators=False):
    configure_runtime_storage()
    if "YAP_NATIVE_RUNTIME_DIR" in os.environ or "YAP_RUNTIME_RESOURCES" in os.environ:
        raise ValueError("delivery verification rejects external runtime/resource overrides")
    if (ROOT / ".runtime_resources.json").exists():
        raise ValueError("delivery verification requires only the inventoried runtime resource map")
    isolate_original_storage()
    from omegaconf import OmegaConf
    from tools.package_assembly_runtime import (
        PREPARED_PROFILES, confined_source, delivery_profiles, digest, prepared_data_files,
    )
    from mechanics_runtime import NATIVE_ROOT, PORTABLE_RUNTIME
    from D2W.assembly_order import load_design as load_d2w
    from W2W.assembly_design import load_design as load_w2w
    from W2W.assembly_order import PLAN_SCHEMA as LEGACY_W2W_PLAN_SCHEMA
    from W2W.assembly_mechanics import _recipe
    from W2W.assembly_operator_library import load_assembly_library, load_prepared_die_library
    from threadpoolctl import threadpool_limits

    if not PORTABLE_RUNTIME or NATIVE_ROOT != ROOT / "runtime/native":
        raise ValueError("verification requires the delivered local native runtime")
    inventory = json.loads((ROOT / "runtime/BUNDLE.json").read_text())
    if "YAP/runtime/RESOURCES.json" not in inventory["files_sha256"]:
        raise ValueError("delivery inventory is missing the runtime resource map")
    for filename, expected in inventory["files_sha256"].items():
        original = ROOT.parent / filename
        path = confined_source(ROOT, original)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"delivery inventory mismatch: {filename}")
    profiles = delivery_profiles(prepared_operators)
    if not set(profiles).issubset(inventory.get("profiles", ())):
        raise ValueError("delivery inventory is missing requested profiles")
    if prepared_operators:
        required = prepared_data_files(NATIVE_ROOT, source_root=ROOT,
                                       map_path=ROOT / "runtime/RESOURCES.json")
        for path, expected in required.items():
            filename = str(path.relative_to(ROOT.parent))
            if inventory["files_sha256"].get(filename) != expected or digest(path) != expected:
                raise ValueError(f"prepared delivery inventory mismatch: {filename}")
        catalog = ROOT / "prepared_operator_catalog.json"
        if str(catalog.relative_to(ROOT.parent)) not in inventory["files_sha256"]:
            raise ValueError("delivery inventory is missing the prepared operator catalog")
        load_operators = True
    reports = {}
    loaders = {"W2W": (load_w2w, load_assembly_library), "D2W": (load_d2w, load_prepared_die_library)}
    for profile in profiles:
        domain, name = profile.split("/")
        load, operator_loader = loaders[domain]
        prepared = profile in PREPARED_PROFILES
        total = float(name.rsplit("_t", 1)[1]) if prepared else None
        cfg = OmegaConf.load(ROOT / domain / "configs" / name / f"{name}.yaml")[domain.lower() + "_modeling"]
        cfg.OVERLAY_SAMPLES = 2
        design, plan, _, _, provenance = load(cfg, ROOT / domain / "input" / name, name)
        settings = design["mechanics"]
        record = dict(orders=len(plan["routes"]), site_count=provenance["site_count"],
                      initial_bows_um=plan["context"]["initial_bows_um"], native_root=str(NATIVE_ROOT))
        if domain == "W2W" and plan["schema"] != LEGACY_W2W_PLAN_SCHEMA:
            record.update(process_plan_status=plan["status"],
                          native_execution_blocked=True, native_operator_loading_verified=False)
            reports[domain] = record
            print(profile + " inventory and process plan verified; native execution blocked", flush=True)
            continue
        _recipe(plan, settings["temporal_profile"])
        if load_operators:
            with threadpool_limits(limits=1):
                library, inputs = operator_loader(settings["operator_package"], plan["context"]["geometry"],
                                                 frozen_preview=settings.get("frozen_preview", False))
            for path in inputs["input_sha256"]:
                source = Path(path)
                if source == ROOT / "prepared_operator_catalog.json" and prepared:
                    confined_source(ROOT, source)
                else:
                    confined_source(NATIVE_ROOT, source)
            if prepared:
                if settings.get("frozen_preview", False):
                    raise ValueError("prepared native64 profiles cannot select frozen_preview")
                check_prepared_library(library, inputs, total)
                totals = plan["context"]["geometry"]
                keys = ("total_thickness_um",) if domain == "D2W" else (
                    "initial_total_thickness_um", "target_total_thickness_um")
                if any(totals[key] != total for key in keys):
                    raise ValueError("prepared profile total thickness mismatch")
                record.update(total_thickness_um=total, retained_si_thickness_um=total - 13.,
                              all_cu_points_retained=True, removable_bulk_operator_built=False)
            record.update(port_rank=library.port_rank_per_face,
                cu_points={key: op.cu_point_count for key, op in library.operators.items()},
                input_files=len(inputs["input_sha256"]), physics_qualification=inputs["physics_qualification"])
            del library
            gc.collect()
        reports[profile if prepared else domain] = record
        print(profile + " local runtime verified", flush=True)
    return dict(status="VERIFIED_INDEPENDENT_RUNTIME" if load_operators else "VERIFIED_INVENTORY_AND_INPUTS",
                native_operator_loading_verified=load_operators, frozen_inputs_rewritten=False,
                full_assembly_histories_rerun=False, profiles=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-operators", action="store_true")
    parser.add_argument("--prepared-operators", action="store_true",
        help="Verify D2W-only delivery and load native64 TOTAL30/80 prepared profiles")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(load_operators=args.load_operators, prepared_operators=args.prepared_operators)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        from mechanics_runtime import validate_write_path
        with validate_write_path(args.report).open("x") as stream:
            stream.write(text)
    print(text)


if __name__ == "__main__":
    main()
