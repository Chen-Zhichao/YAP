"""Export checked p2 runtimes without copying solver campaigns or raw inputs."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf

from mechanics_runtime import (
    HISTORICAL_NATIVE_ROOTS, P2_COMPACT_MANIFEST_SHA256, resolve_resource, resource_directory,
    validate_write_path,
)


ROOT = Path(__file__).resolve().parents[1]
COMPACT = "resource:compact_p2/packages/BI2_50.json"
PARENT = "resource:geometry/packages/BI2_50.json"
BASE_PROFILES = ("D2W/design_6_4hi_p2", "W2W/design_6_4hi_p2")
PREPARED_PROFILES = (
    "D2W/design_6_4hi_p2_t30", "D2W/design_6_4hi_p2_t80",
)
PREPARED_BASIS_SHA256 = "8c2941e8358dba9626177cfd54c2187d4d0878d4ca1108d2652261efd318cb06"
SEEDS = (
    "exp12_w2w_window_20260914", "exp12_mindlin_sequential_runtime", "exp12_v2_process_schedule",
    "exp12_v2_production_sequential_process", "run_exp12_free_membrane_sequential",
    "exp12_free_membrane_bonding", "exp12_all_gauss_process_transactions",
    "run_exp12_v2_production_sequential_all_gauss_v1", "exp12_all_gauss_thermal_advance",
    "exp12_prebond_export", "run_exp12_rt_flattening_short_comparison",
    "exp12_all_gauss_periodic_stack", "exp12_all_gauss_exact_acceleration",
    "exp12_layer_rom_dataset", "run_exp12_design_window_histories_20260914",
)


def digest(path, *, rate_mib=None):
    value = hashlib.sha256()
    started, count = time.monotonic(), 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(block)
            if rate_mib is not None:
                count += len(block)
                time.sleep(max(0., count / (rate_mib * 1024**2) - (time.monotonic() - started)))
    return value.hexdigest()


def delivery_profiles(prepared_operators=False):
    return (BASE_PROFILES[:1] + PREPARED_PROFILES) if prepared_operators else BASE_PROFILES


def staged_bindings(native_root, source_root, map_path, manifest_path, manifest_sha):
    from W2W.assembly_operator_library import _prepared_json, _prepared_sha
    root = native_root.parent.parent
    manifest_path = confined_source(root, Path(manifest_path).absolute())
    if (native_root != root / "runtime/native" or manifest_path != root / "STAGE.json" or source_root != root / "source"
            or map_path is None or Path(map_path).absolute() != root / "runtime/RESOURCES.json"):
        raise ValueError("staging receipt must bind the selected native, application and map roots")
    document = _prepared_json(manifest_path, {}, _prepared_sha(manifest_sha, "stage inventory"), limit=1024**2)
    files = document.get("files_sha256")
    if (document.get("schema") != "yap.prepared_runtime_stage.v1"
            or document.get("source_payloads_rewritten") is not False
            or document.get("application_source") != "source"
            or not isinstance(files, dict) or not 1 <= len(files) <= 512):
        raise ValueError("staging receipt must include bounded frozen application and runtime inputs")
    result = {}
    for relative, expected in files.items():
        if not isinstance(relative, str) or not relative or len(relative) > 512:
            raise ValueError("invalid staged input path")
        path = confined_source(root, root / relative)
        if path in result:
            raise ValueError("duplicate staged input destination")
        result[path] = _prepared_sha(expected, "staged input")
    result[manifest_path] = manifest_sha
    return result, manifest_path


def native_sources(root):
    pending, found = list(SEEDS), {}
    while pending:
        name = pending.pop()
        if name in found:
            continue
        if len(found) >= 256:
            raise ValueError("native import closure exceeds 256 files")
        path = confined_source(root, root / f"{name}.py")
        source = path.read_text()
        found[name] = path
        for node in ast.walk(ast.parse(source)):
            names = ([value.name for value in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for module in names:
                first = module.split(".")[0]
                if confined_source(root, root / f"{first}.py").is_file():
                    pending.append(first)
    return sorted(found.values())


def confined_source(root, path):
    """Reject lexical escapes before resolving or opening a source file."""
    root, path = Path(root).absolute(), Path(path).absolute()
    if {"#snapshot", ".snapshot"}.intersection(root.parts + path.parts):
        raise ValueError("snapshot paths are not allowed")
    if ".." in root.parts + path.parts or not path.is_relative_to(root):
        raise ValueError("source path escapes native root")
    canonical_root = root.resolve()
    if {"#snapshot", ".snapshot"}.intersection(canonical_root.parts):
        raise ValueError("snapshot paths are not allowed")
    candidate = canonical_root
    for part in path.relative_to(root).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("source path must not contain symlinks")
    resolved = candidate.resolve()
    if {"#snapshot", ".snapshot"}.intersection(resolved.parts):
        raise ValueError("snapshot paths are not allowed")
    if not resolved.is_relative_to(canonical_root):
        raise ValueError("source path escapes native root")
    return resolved


def source_reference(root, value):
    root, path = Path(root).absolute(), Path(value)
    if {"#snapshot", ".snapshot"}.intersection(root.parts + path.parts):
        raise ValueError("snapshot paths are not allowed")
    if ".." in root.parts + path.parts:
        raise ValueError("source path escapes native root")
    root = root.resolve()
    if {"#snapshot", ".snapshot"}.intersection(root.parts):
        raise ValueError("snapshot paths are not allowed")
    if path.is_absolute():
        if path.is_relative_to(root):
            return confined_source(root, path)
        for previous in HISTORICAL_NATIVE_ROOTS:
            if path.is_relative_to(previous):
                return confined_source(root, root / path.relative_to(previous))
        raise ValueError(f"unrecognized frozen source path: {path}")
    return confined_source(root, root / path)


def data_files(root, *, map_path=None):
    compact_file = resolve_resource(COMPACT, native_root=root, map_path=map_path)
    parent_file = resolve_resource(PARENT, native_root=root, map_path=map_path)
    if digest(compact_file) != P2_COMPACT_MANIFEST_SHA256:
        raise ValueError("frozen p2 manifest SHA256 mismatch")
    compact = json.loads(compact_file.read_text())
    if digest(parent_file) != compact["parent_package_sha256"]:
        raise ValueError("frozen parent manifest SHA256 mismatch")
    parent = json.loads(parent_file.read_text())
    required = {compact_file: P2_COMPACT_MANIFEST_SHA256, parent_file: compact["parent_package_sha256"],
                compact_file.parent.parent / "MANIFEST.json": None,
                root / "V2_PROCESS_ALIAS_CONTRACT_V1.json": None}
    for package, keys in ((compact, ("bi", "beol", "tsv_si_h37")), (parent, ("bi", "beol"))):
        required[source_reference(root, package["port_basis"])] = package["port_basis_sha256"]
        for key in keys:
            required[source_reference(root, package["datasets"][key])] = package["dataset_sha256"][key]
    alias_path = confined_source(root, root / "V2_PROCESS_ALIAS_CONTRACT_V1.json")
    alias = json.loads(alias_path.read_text())
    required[root / "results/material_card_v2_kim2026/PROCESS_CARD_V2_KIM2026.json"] = alias["source_card_sha256"]
    return required


def prepared_artifact(base, value, resource_root):
    """Resolve package-relative references without leaving their resource."""
    if not isinstance(value, str) or not value or len(value) > 512 or value.startswith("resource:"):
        raise ValueError("invalid prepared artifact path")
    path = Path(value)
    if path.is_absolute() or len(path.parts) > 16:
        raise ValueError("prepared artifact paths must be bounded relative paths")
    if {"#snapshot", ".snapshot"}.intersection(path.parts):
        raise ValueError("snapshot paths are not allowed")
    current = confined_source(resource_root, base)
    for part in path.parts:
        if part == "..":
            if current == resource_root:
                raise ValueError("prepared artifact path escapes its resource")
            current = current.parent
        else:
            current = current / part
        current = confined_source(resource_root, current)
    return current


def prepared_data_files(root, *, source_root=ROOT, map_path=None):
    from W2W.assembly_operator_library import _prepared_json, _prepared_sha

    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    observed = {}
    catalog_path = confined_source(source_root, source_root / "prepared_operator_catalog.json")
    catalog = _prepared_json(catalog_path, observed, limit=65536)
    if set(catalog) != {"schema", "packages"} or catalog["schema"] != "yap.prepared_p2_operator_catalog.v1":
        raise ValueError("invalid prepared operator catalog")
    if not isinstance(catalog["packages"], dict) or set(catalog["packages"]) != {"30", "80"}:
        raise RuntimeError("both TOTAL30/80 audits must be catalog-admitted before prepared export")
    fields = {"package_sha256", "audit_sha256", "parent_package_sha256", "port_basis_sha256"}
    for entry in catalog["packages"].values():
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError("invalid prepared catalog entry")
        for field in fields:
            _prepared_sha(entry[field], field)
        if entry["port_basis_sha256"] != PREPARED_BASIS_SHA256:
            raise ValueError("prepared catalog port basis mismatch")
    if catalog["packages"]["30"]["parent_package_sha256"] != catalog["packages"]["80"]["parent_package_sha256"]:
        raise ValueError("prepared targets must share the native parent")
    resource = resource_directory("prepared_p2_native", native_root=root, map_path=map_path)
    required = {}

    def add(path, expected):
        path = confined_source(root, path)
        expected = _prepared_sha(expected, str(path))
        if path in required and required[path] != expected:
            raise ValueError(f"conflicting prepared input SHA256: {path}")
        required[path] = expected

    for total in (30, 80):
        entry = catalog["packages"][str(total)]
        path = resolve_resource(f"resource:prepared_p2_native/packages/BI2_{total}.json",
                                native_root=root, map_path=map_path)
        package = _prepared_json(path, observed, entry["package_sha256"])
        if (package.get("schema") != "yap.prepared_p2_geometry_package.v1"
                or package.get("name") != f"BI2_{total}" or type(package.get("rank")) is not int
                or package["rank"] != 64 or package.get("ready") is not False
                or package.get("state") != "CANDIDATE_NOT_VALIDATED"):
            raise ValueError("prepared export requires native64 engineering candidates")
        for field in ("parent_package_sha256", "port_basis_sha256"):
            if package.get(field) != entry[field]:
                raise ValueError(f"prepared package {field} differs from catalog")
        if package.get("audit_receipt_sha256") != entry["audit_sha256"]:
            raise ValueError("prepared audit SHA256 differs from catalog")
        add(path, entry["package_sha256"])
        parent_path = prepared_artifact(path.parent, package.get("parent_package"), resource)
        parent = _prepared_json(parent_path, observed, entry["parent_package_sha256"])
        if (parent.get("schema") != "exp12.ofat.geometry_package.v1" or parent.get("name") != "BI2_50"
                or type(parent.get("rank")) is not int or parent["rank"] != 64
                or parent.get("ready") is not True or parent.get("state") != "READY_ENGINEERING_PREVIEW"
                or parent.get("port_basis_sha256") != PREPARED_BASIS_SHA256):
            raise ValueError("prepared export requires its exact native64 BI2_50 parent")
        add(parent_path, entry["parent_package_sha256"])
        receipt_path = prepared_artifact(path.parent, package.get("audit_receipt"), resource)
        receipt = _prepared_json(receipt_path, observed, entry["audit_sha256"])
        if (receipt.get("schema") != "yap.prepared_p2_operator_audit.v1"
                or receipt.get("state") != "PASS_STRUCTURAL_ENGINEERING_CANDIDATE"
                or receipt.get("all_cu_points_retained") is not True
                or receipt.get("thermal_geometry_validated") is not False
                or receipt.get("stage_count") != 6 or receipt.get("rank") != 64
                or receipt.get("face_node_shape") != [46, 46]
                or receipt.get("total_thickness_um") != total
                or receipt.get("retained_si_thickness_um") != total - 13):
            raise ValueError("prepared export requires structurally audited, unvalidated target inputs")
        for field in ("parent_package_sha256", "port_basis_sha256"):
            if receipt.get(field) != entry[field]:
                raise ValueError(f"prepared receipt {field} differs from catalog")
        add(receipt_path, entry["audit_sha256"])
        keys = {"bi", "beol", f"tsv_si_h{total - 13}"}
        for field, hashes in (("datasets", "dataset_sha256"), ("prepared", "prepared_sha256")):
            if any(not isinstance(package.get(name), dict) or set(package[name]) != keys for name in (field, hashes)) \
                    or not isinstance(receipt.get(field), dict) or set(receipt[field]) != keys:
                raise ValueError(f"prepared {field} must bind exactly three target layers")
            for key in keys:
                expected = package[hashes][key]
                if receipt[field][key].get("sha256") != expected:
                    raise ValueError(f"prepared receipt {field}/{key} binding mismatch")
                if field == "datasets" and key in ("bi", "beol") \
                        and expected != parent.get("dataset_sha256", {}).get(key):
                    raise ValueError(f"prepared {key} differs from the native parent")
                add(prepared_artifact(path.parent, package[field][key], resource), expected)
        add(prepared_artifact(path.parent, package.get("port_basis"), resource), PREPARED_BASIS_SHA256)
        transfer = receipt.get("transfer_audit", {})
        if not isinstance(transfer, dict):
            raise ValueError("prepared receipt requires a bubble-transfer audit")
        bubbles = set()
        for field, hashes in (("transferred_bubble", "transferred_bubble_sha256"),
                              ("raw_transferred_bubble", "raw_transferred_bubble_sha256")):
            expected = _prepared_sha(transfer.get(hashes), hashes)
            bubble = transfer.get(field)
            if not isinstance(bubble, dict) or bubble.get("sha256") != expected:
                raise ValueError(f"prepared receipt {field} artifact binding mismatch")
            bubble_path = prepared_artifact(receipt_path.parent, bubble.get("path"), resource)
            if bubble_path in bubbles:
                raise ValueError("raw and finalized bubble artifacts must be distinct")
            bubbles.add(bubble_path)
            add(bubble_path, expected)
        receipts = receipt.get("producer_receipts_sha256")
        if not isinstance(receipts, dict) or not 1 <= len(receipts) <= 128:
            raise ValueError("prepared receipt requires a bounded producer receipt inventory")
        for filename, expected in receipts.items():
            add(prepared_artifact(resource, filename, resource), expected)
    for path, expected in required.items():
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"prepared input SHA256 mismatch: {path}")
    return required


def replace_yaml_scalar(text, key, value):
    lines, count = [], 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(key + ":"):
            indent = line[:len(line) - len(stripped)]
            comment = "  # " + line.split("#", 1)[1].strip() if "#" in line else ""
            line = indent + key + ": " + value + comment + "\n"
            count += 1
        lines.append(line)
    if count == 0:
        raise ValueError(f"missing YAML field: {key}")
    return "".join(lines)


def export(destination, native_root, *, source_root=ROOT, map_path=None, prepared_operators=False,
           stage_inventory=None, stage_inventory_sha256=None):
    if type(prepared_operators) is not bool:
        raise ValueError("prepared_operators must be Boolean")
    destination, native_root = validate_write_path(destination), Path(native_root).absolute()
    if {"#snapshot", ".snapshot"}.intersection(destination.parts + native_root.parts):
        raise ValueError("snapshot paths are not allowed")
    native_root = native_root.resolve()
    source_root = Path(source_root).resolve()
    staged = None
    if (stage_inventory is None) != (stage_inventory_sha256 is None):
        raise ValueError("stage inventory and independent SHA256 must be supplied together")
    if stage_inventory is not None:
        staged, stage_inventory = staged_bindings(native_root, source_root, map_path,
                                                  stage_inventory, stage_inventory_sha256)
        mapping = confined_source(native_root.parent.parent, Path(map_path).absolute())
        if mapping not in staged or digest(mapping) != staged[mapping]:
            raise ValueError("resource map differs from its staged pin")
    if {"#snapshot", ".snapshot"}.intersection(native_root.parts):
        raise ValueError("snapshot paths are not allowed")
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    sources = native_sources(native_root)
    data = data_files(native_root, map_path=map_path)
    catalog_sha256 = None
    if prepared_operators:
        catalog_path = confined_source(source_root, source_root / "prepared_operator_catalog.json")
        catalog_sha256 = digest(catalog_path)
        for path, expected in prepared_data_files(native_root, source_root=source_root, map_path=map_path).items():
            if path in data and data[path] != expected:
                raise ValueError(f"conflicting input SHA256: {path}")
            data[path] = expected
    metric = confined_source(native_root.parent,
                             native_root.parent / "200_bow_diff_vs_magnification/nonlinear_face_lever.py")
    if staged is not None:
        for path in (*sources, *data, metric):
            if path not in staged:
                raise ValueError(f"export input missing from staged inventory: {path}")
            if data.get(path) is not None and data[path] != staged[path]:
                raise ValueError(f"export input conflicts with staged pin: {path}")
            if path in data:
                data[path] = staged[path]
        for path in sources:
            if digest(path) != staged[path]:
                raise ValueError(f"native module differs from its staged pin: {path}")
    resource_names = ("compact_p2", "geometry") + (("prepared_p2_native",) if prepared_operators else ())
    resources = {name: str(resource_directory(name, native_root=native_root, map_path=map_path)
                           .relative_to(native_root.resolve()))
                 for name in resource_names}
    profiles = delivery_profiles(prepared_operators)
    application = []
    directories = (source_root, source_root / "D2W", source_root / "W2W",
                   source_root / "D2W/utils", source_root / "W2W/utils")
    tools = confined_source(source_root, source_root / "tools")
    if tools.is_dir():
        directories += (tools,)
    for directory in directories:
        directory = confined_source(source_root, directory)
        paths = sorted(directory.glob("*.py"))
        if len(paths) > 128:
            raise ValueError(f"unexpected source-directory size: {directory}")
        application.extend(paths)
    application.extend(source_root / name for name in (
        "requirements.txt", "LICENSE", "README.md", "PORTABLE_ASSEMBLY.md", "W2W/ASSEMBLY_ORDER.md"))
    if prepared_operators:
        application.append(source_root / "prepared_operator_catalog.json")
    configs = []
    for profile in profiles:
        domain, name = profile.split("/")
        directory = confined_source(source_root, source_root / domain / "input" / name)
        paths = list(directory.iterdir())
        if len(paths) > 16:
            raise ValueError("unexpected profile-input count")
        application.extend(paths)
        configs.append(source_root / domain / "configs" / name / f"{name}.yaml")
    application.extend(configs)
    if len(application) > 256 or len(set(application)) != len(application):
        raise ValueError("application inputs must be bounded and unique")
    for path in application:
        confined_source(source_root, path)
        if not path.is_file():
            raise ValueError(f"application input must be a regular file: {path}")
        if staged is not None and (path not in staged or digest(path) != staged[path]):
            raise ValueError(f"application input differs from staged inventory: {path}")
    if staged is not None and digest(metric) != staged[metric]:
        raise ValueError("metric differs from its staged pin")
    for path, expected in data.items():
        confined_source(native_root, path)
        if not path.is_file():
            raise ValueError(f"native input must be a regular file: {path}")
        if expected is not None and digest(path) != expected:
            raise ValueError(f"frozen input SHA256 mismatch: {path}")
    total = sum(path.stat().st_size for path in data)
    if shutil.disk_usage(destination.parent).free < total + 1024**3:
        raise OSError("insufficient destination storage")
    destination.mkdir()
    output = destination / "YAP"
    output.mkdir()
    inventory = {}
    marker = output / "ASSEMBLY_RUNTIME.json"
    marker.write_text(json.dumps({"schema": "yap.portable_assembly_p2.v1",
                                 "native_directory": "runtime/native"}, indent=2) + "\n")
    inventory[str(marker.relative_to(destination))] = digest(marker)
    def copy(path, target, expected=None):
        confined_source(output, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if staged is not None:
            if path not in staged or (expected is not None and expected != staged[path]):
                raise ValueError(f"export source lacks a matching staged pin: {path}")
            expected = staged[path]
        observed = digest(path)
        if expected is not None and observed != expected:
            raise ValueError(f"frozen input SHA256 mismatch: {path}")
        started, count = time.monotonic(), 0
        with path.open("rb") as reader, target.open("xb") as writer:
            for block in iter(lambda: reader.read(4 * 1024**2), b""):
                writer.write(block)
                count += len(block)
                time.sleep(max(0., count / (24 * 1024**2) - (time.monotonic() - started)))
            writer.flush()
            os.fsync(writer.fileno())
        if digest(target, rate_mib=32) != observed:
            raise ValueError(f"copied input SHA256 mismatch: {target}")
        inventory[str(target.relative_to(destination))] = observed

    for path in application:
        expected = catalog_sha256 if path == source_root / "prepared_operator_catalog.json" else None
        copy(path, output / path.relative_to(source_root), expected)
    for config in configs:
        target = output / config.relative_to(source_root)
        text = replace_yaml_scalar(target.read_text(), "OUTPUT_DIR", "output/assembly_orders")
        target.write_text(text)
        inventory[str(target.relative_to(destination))] = digest(target)
        original, portable = OmegaConf.load(config), OmegaConf.load(target)
        for mode in original:
            assert list(original[mode]) == list(portable[mode])
            for key in original[mode]:
                if key != "OUTPUT_DIR":
                    assert original[mode][key] == portable[mode][key], (mode, key)
    native_output = output / "runtime/native"
    for path in sources:
        confined_source(native_root, path)
        copy(path, native_output / path.name)
    for path, expected in sorted(data.items()):
        copy(path, native_output / path.relative_to(native_root), expected)
    copy(metric, output / "runtime/200_bow_diff_vs_magnification/nonlinear_face_lever.py")
    resource_map = output / "runtime/RESOURCES.json"
    resource_map.write_text(json.dumps(dict(schema="yap.native_resources.v1", resources=resources),
                                       indent=2, sort_keys=True) + "\n")
    inventory[str(resource_map.relative_to(destination))] = digest(resource_map)
    if staged is not None:
        if inventory[str(resource_map.relative_to(destination))] != staged[Path(map_path).absolute()]:
            raise ValueError("exported resource map differs from its staged pin")
        if digest(stage_inventory) != stage_inventory_sha256:
            raise ValueError("staging receipt changed during export")
        copy(stage_inventory, output / "runtime/SOURCE_STAGE.json", stage_inventory_sha256)
    manifest = dict(schema="yap.portable_assembly_p2.v1", files_sha256=inventory,
        native_modules=[path.stem for path in sources], native_data_bytes=total,
        profiles=list(profiles),
        frozen_inputs_rewritten=False, publication_scope="private_runtime_delivery",
        physics_qualification="ENGINEERING_PREVIEW_NOT_VALIDATED")
    if staged is not None:
        manifest["stage_inventory_sha256"] = stage_inventory_sha256
    (output / "runtime/BUNDLE.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Exported {len(inventory)} files, {total / 1024**3:.3f} GiB native data: {destination}", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--native-source", type=Path,
        default=ROOT.parent / "Ansys_Verify/12_stack_model_validation")
    parser.add_argument("--resource-map", type=Path, help="Private resource directory bindings")
    parser.add_argument("--prepared-operators", action="store_true",
        help="Export D2W-only profiles with catalog-admitted native64 TOTAL30/80 operators")
    parser.add_argument("--stage-inventory", type=Path,
                        help="Authenticate every export input against a frozen staging receipt")
    parser.add_argument("--stage-inventory-sha256",
                        help="Independently pinned staging receipt SHA256")
    args = parser.parse_args()
    export(args.output_dir, args.native_source, map_path=args.resource_map,
           prepared_operators=args.prepared_operators, stage_inventory=args.stage_inventory,
           stage_inventory_sha256=args.stage_inventory_sha256)


if __name__ == "__main__":
    main()
