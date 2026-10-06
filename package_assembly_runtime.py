"""Export the p2 D2W/W2W runtime, without copying solver campaigns or receipts."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

from omegaconf import OmegaConf

from mechanics_runtime import (
    HISTORICAL_NATIVE_ROOTS, P2_COMPACT_MANIFEST_SHA256, resolve_resource, resource_directory,
)


ROOT = Path(__file__).resolve().parent
COMPACT = "resource:compact_p2/packages/BI2_50.json"
PARENT = "resource:geometry/packages/BI2_50.json"
SEEDS = (
    "exp12_w2w_window_20260914", "exp12_mindlin_sequential_runtime", "exp12_v2_process_schedule",
    "exp12_v2_production_sequential_process", "run_exp12_free_membrane_sequential",
    "exp12_free_membrane_bonding", "exp12_all_gauss_process_transactions",
    "run_exp12_v2_production_sequential_all_gauss_v1", "exp12_all_gauss_thermal_advance",
    "exp12_prebond_export", "run_exp12_rt_flattening_short_comparison",
    "exp12_all_gauss_periodic_stack", "exp12_all_gauss_exact_acceleration",
    "exp12_layer_rom_dataset", "run_exp12_design_window_histories_20260914",
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def native_sources(root):
    pending, found = list(SEEDS), {}
    while pending:
        name = pending.pop()
        if name in found:
            continue
        if len(found) >= 256:
            raise ValueError("native import closure exceeds 256 files")
        path = root / f"{name}.py"
        source = path.read_text()
        found[name] = path
        for node in ast.walk(ast.parse(source)):
            names = ([value.name for value in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for module in names:
                first = module.split(".")[0]
                if (root / f"{first}.py").is_file():
                    pending.append(first)
    return sorted(found.values())


def source_reference(root, value):
    path = Path(value)
    if {"#snapshot", ".snapshot"}.intersection(path.parts):
        raise ValueError("snapshot paths are not allowed")
    if path.is_absolute():
        for previous in HISTORICAL_NATIVE_ROOTS:
            if path.is_relative_to(previous):
                return root / path.relative_to(previous)
        raise ValueError(f"unrecognized frozen source path: {path}")
    candidate = root / path
    if not candidate.absolute().is_relative_to(root.absolute()):
        raise ValueError("source path escapes native root")
    return candidate


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
    alias = json.loads((root / "V2_PROCESS_ALIAS_CONTRACT_V1.json").read_text())
    required[root / "results/material_card_v2_kim2026/PROCESS_CARD_V2_KIM2026.json"] = alias["source_card_sha256"]
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


def export(destination, native_root, *, source_root=ROOT, map_path=None):
    destination, native_root = Path(destination).absolute(), Path(native_root).absolute()
    if {"#snapshot", ".snapshot"}.intersection(destination.parts + native_root.parts):
        raise ValueError("snapshot paths are not allowed")
    native_root = native_root.resolve()
    if {"#snapshot", ".snapshot"}.intersection(native_root.parts):
        raise ValueError("snapshot paths are not allowed")
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    sources = native_sources(native_root)
    data = data_files(native_root, map_path=map_path)
    resources = {name: str(resource_directory(name, native_root=native_root, map_path=map_path)
                           .relative_to(native_root.resolve()))
                 for name in ("compact_p2", "geometry")}
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
        target.parent.mkdir(parents=True, exist_ok=True)
        observed = digest(path)
        if expected is not None and observed != expected:
            raise ValueError(f"frozen input SHA256 mismatch: {path}")
        shutil.copyfile(path, target)
        if digest(target) != observed:
            raise ValueError(f"copied input SHA256 mismatch: {target}")
        inventory[str(target.relative_to(destination))] = observed

    # Only these flat source directories are enumerated; no output tree is visited.
    directories = (source_root, source_root / "D2W", source_root / "W2W",
                   source_root / "D2W/utils", source_root / "W2W/utils")
    for directory in directories:
        paths = sorted(directory.glob("*.py"))
        if len(paths) > 128:
            raise ValueError(f"unexpected source-directory size: {directory}")
        for path in paths:
            copy(path, output / path.relative_to(source_root))
    for filename in ("requirements.txt", "LICENSE", "README.md", "PORTABLE_ASSEMBLY.md"):
        copy(source_root / filename, output / filename)
    copy(source_root / "W2W/ASSEMBLY_ORDER.md", output / "W2W/ASSEMBLY_ORDER.md")
    for domain in ("D2W", "W2W"):
        name = "design_6_4hi_p2"
        directory = source_root / domain / "input" / name
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"unexpected design input: {path}")
            copy(path, output / path.relative_to(source_root))
        config = source_root / domain / "configs" / name / f"{name}.yaml"
        target = output / config.relative_to(source_root)
        copy(config, target)
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
        copy(path, native_output / path.name)
    for path, expected in sorted(data.items()):
        copy(path, native_output / path.relative_to(native_root), expected)
    metric = native_root.parent / "200_bow_diff_vs_magnification/nonlinear_face_lever.py"
    copy(metric, output / "runtime/200_bow_diff_vs_magnification/nonlinear_face_lever.py")
    resource_map = output / "runtime/RESOURCES.json"
    resource_map.write_text(json.dumps(dict(schema="yap.native_resources.v1", resources=resources),
                                       indent=2, sort_keys=True) + "\n")
    inventory[str(resource_map.relative_to(destination))] = digest(resource_map)
    manifest = dict(schema="yap.portable_assembly_p2.v1", files_sha256=inventory,
        native_modules=[path.stem for path in sources], native_data_bytes=total,
        profiles=["D2W/design_6_4hi_p2", "W2W/design_6_4hi_p2"],
        frozen_inputs_rewritten=False, publication_scope="private_runtime_delivery",
        physics_qualification="ENGINEERING_PREVIEW_NOT_VALIDATED")
    (output / "runtime/BUNDLE.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Exported {len(inventory)} files, {total / 1024**3:.3f} GiB native data: {destination}", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--native-source", type=Path,
        default=ROOT.parent / "Ansys_Verify/12_stack_model_validation")
    parser.add_argument("--resource-map", type=Path, help="Private resource directory bindings")
    args = parser.parse_args()
    export(args.output_dir, args.native_source, map_path=args.resource_map)


if __name__ == "__main__":
    main()
