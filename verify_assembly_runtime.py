"""Verify an exported p2 runtime without access to its historical project."""

import argparse
import gc
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent


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


def verify(*, load_operators=False):
    if "YAP_NATIVE_RUNTIME_DIR" in os.environ or "YAP_RUNTIME_RESOURCES" in os.environ:
        raise ValueError("delivery verification rejects external runtime/resource overrides")
    if (ROOT / ".runtime_resources.json").exists():
        raise ValueError("delivery verification requires only the inventoried runtime resource map")
    isolate_original_storage()
    from omegaconf import OmegaConf
    from package_assembly_runtime import digest
    from mechanics_runtime import NATIVE_ROOT, PORTABLE_RUNTIME
    from D2W.assembly_order import load_design as load_d2w
    from W2W.assembly_design import load_design as load_w2w
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
        path = original.resolve()
        if not path.is_relative_to(ROOT) or original.is_symlink() or digest(path) != expected:
            raise ValueError(f"delivery inventory mismatch: {filename}")
    reports = {}
    for domain, load, operator_loader in (
        ("W2W", load_w2w, load_assembly_library), ("D2W", load_d2w, load_prepared_die_library),
    ):
        name = "design_6_4hi_p2"
        cfg = OmegaConf.load(ROOT / domain / "configs" / name / f"{name}.yaml")[domain.lower() + "_modeling"]
        cfg.OVERLAY_SAMPLES = 2
        design, plan, _, _, provenance = load(cfg, ROOT / domain / "input" / name, name)
        settings = design["mechanics"]
        _recipe(plan, settings["temporal_profile"])
        record = dict(orders=len(plan["routes"]), site_count=provenance["site_count"],
                      initial_bows_um=plan["context"]["initial_bows_um"], native_root=str(NATIVE_ROOT))
        if load_operators:
            with threadpool_limits(limits=1):
                library, inputs = operator_loader(settings["operator_package"], plan["context"]["geometry"], frozen_preview=True)
            assert all(Path(path).is_relative_to(NATIVE_ROOT) for path in inputs["input_sha256"])
            record.update(port_rank=library.port_rank_per_face,
                cu_points={key: op.cu_point_count for key, op in library.operators.items()},
                input_files=len(inputs["input_sha256"]), physics_qualification=inputs["physics_qualification"])
            del library
            gc.collect()
        reports[domain] = record
        print(domain + " local runtime verified", flush=True)
    return dict(status="VERIFIED_INDEPENDENT_RUNTIME" if load_operators else "VERIFIED_INVENTORY_AND_INPUTS",
                native_operator_loading_verified=load_operators, frozen_inputs_rewritten=False,
                full_assembly_histories_rerun=False, profiles=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-operators", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(load_operators=args.load_operators)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        with args.report.open("x") as stream:
            stream.write(text)
    print(text)


if __name__ == "__main__":
    main()
