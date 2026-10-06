"""Locate native mechanics and relocate frozen references without editing data."""

import json
import os
from pathlib import Path
import re
import stat


ROOT = Path(__file__).resolve().parent
P2_COMPACT_MANIFEST_SHA256 = "cba9f64bdf41b610ec9b538858f07c119f7febd89515f84d8ac6df2e74bd8612"
HISTORICAL_NATIVE_ROOTS = tuple(Path(value) for value in (
    "/w/ee.00/puneet/zhichao/Desktop/YAP_STACK/Ansys_Verify/12_stack_model_validation",
    "/u1/ee/zhichao/Desktop/YAP_STACK/Ansys_Verify/12_stack_model_validation",
    "/app/nanocad/projects/personal/zchen/YAP_STACK/Ansys_Verify/12_stack_model_validation",
))


def select_native_root(root=ROOT):
    override = os.environ.get("YAP_NATIVE_RUNTIME_DIR")
    bundled = Path(root) / "runtime"
    if override is not None:
        candidate, portable = Path(override), True
    elif bundled.exists() or bundled.is_symlink() or (Path(root) / "ASSEMBLY_RUNTIME.json").exists():
        candidate, portable = bundled / "native", True
    else:
        candidate, portable = Path(root).parent / "Ansys_Verify/12_stack_model_validation", False
    if {"#snapshot", ".snapshot"}.intersection(candidate.parts):
        raise ValueError("snapshot paths are not allowed")
    candidate = candidate.resolve()
    if {"#snapshot", ".snapshot"}.intersection(candidate.parts):
        raise ValueError("snapshot paths are not allowed")
    if portable and override is None and not candidate.is_relative_to(Path(root).resolve()):
        raise ValueError("native runtime resolves outside the delivery folder")
    if portable and not (candidate / "exp12_layer_rom_dataset.py").is_file():
        raise FileNotFoundError(f"incomplete native runtime: {candidate}; no legacy fallback")
    return candidate, portable


def __getattr__(name):
    if name not in ("NATIVE_ROOT", "PORTABLE_RUNTIME"):
        raise AttributeError(name)
    globals()["NATIVE_ROOT"], globals()["PORTABLE_RUNTIME"] = select_native_root()
    return globals()[name]


def _runtime_value(name):
    return globals()[name] if name in globals() else __getattr__(name)


def _safe_parts(value):
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError("resource paths must be nonempty relative strings")
    parts = value.split("/")
    if len(parts) > 16 or any(
        part in (".", "..", "#snapshot", ".snapshot")
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts
    ):
        raise ValueError("invalid resource path component")
    return parts


def validate_resource_reference(value):
    """Validate a logical reference without requiring private runtime data."""
    if not isinstance(value, str) or not value.startswith("resource:"):
        raise ValueError("expected resource:<id>/<relative-path>")
    name, separator, suffix = value[9:].partition("/")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) or not separator:
        raise ValueError("invalid resource reference")
    _safe_parts(suffix)
    return name, suffix


def resource_map_path(native_root=None):
    override = os.environ.get("YAP_RUNTIME_RESOURCES")
    local = ROOT / ".runtime_resources.json"
    root = _runtime_value("NATIVE_ROOT") if native_root is None else Path(native_root)
    return Path(override) if override is not None else (
        local if local.exists() else root.parent / "RESOURCES.json")


def load_resource_map(path):
    path = Path(path)
    if {"#snapshot", ".snapshot"}.intersection(path.parts + path.resolve().parts):
        raise ValueError("snapshot paths are not allowed")
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("resource map must not be a symlink")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "private runtime resource map missing; set YAP_RUNTIME_RESOURCES or "
            "provide .runtime_resources.json") from exc
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("resource map must be a regular file")
        payload = stream.read(65537)
    if len(payload) > 65536:
        raise ValueError("resource map exceeds 64 KiB")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate resource map key")
            result[key] = value
        return result
    value = json.loads(payload, object_pairs_hook=unique)
    if not isinstance(value, dict) or set(value) != {"schema", "resources"} \
            or value["schema"] != "yap.native_resources.v1":
        raise ValueError("invalid resource map schema")
    resources = value["resources"]
    if not isinstance(resources, dict) or not 1 <= len(resources) <= 64:
        raise ValueError("resource map must contain 1-64 directory bindings")
    for name, directory in resources.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name):
            raise ValueError("invalid resource ID")
        _safe_parts(directory)
    return value


def resource_directory(name, *, native_root=None, map_path=None):
    root = _runtime_value("NATIVE_ROOT") if native_root is None else Path(native_root)
    if {"#snapshot", ".snapshot"}.intersection(root.parts + root.resolve().parts):
        raise ValueError("snapshot paths are not allowed")
    root = root.resolve()
    mapping = load_resource_map(resource_map_path(root) if map_path is None else map_path)
    if name not in mapping["resources"]:
        raise ValueError(f"resource not registered: {name}")
    path = root
    for part in _safe_parts(mapping["resources"][name]):
        path = path / part
        if path.is_symlink():
            raise ValueError("resource binding must not contain symlinks")
    resolved = path.resolve()
    if {"#snapshot", ".snapshot"}.intersection(resolved.parts) or not resolved.is_relative_to(root):
        raise ValueError("resource binding escapes native runtime")
    return resolved


def resolve_resource(value, *, native_root=None, map_path=None):
    name, suffix = validate_resource_reference(value)
    base = resource_directory(name, native_root=native_root, map_path=map_path)
    path = base
    for part in _safe_parts(suffix):
        path = path / part
        if path.is_symlink():
            raise ValueError("resource path must not contain symlinks")
    resolved = path.resolve()
    if not resolved.is_relative_to(base):
        raise ValueError("resource path escapes its registered directory")
    return resolved


def resolve_native_input(base, value, *, native_root=None, portable=None):
    root = _runtime_value("NATIVE_ROOT") if native_root is None else Path(native_root).resolve()
    confined = _runtime_value("PORTABLE_RUNTIME") if portable is None else portable
    if str(value).startswith("resource:"):
        return resolve_resource(str(value), native_root=root)
    path = Path(value)
    if {"#snapshot", ".snapshot"}.intersection(path.parts):
        raise ValueError("snapshot paths are not allowed")
    if path.is_absolute():
        for old_root in HISTORICAL_NATIVE_ROOTS:
            if path.is_relative_to(old_root):
                path = root / path.relative_to(old_root)
                break
    else:
        path = Path(base) / path
    path = path.resolve()
    if {"#snapshot", ".snapshot"}.intersection(path.parts):
        raise ValueError("snapshot paths are not allowed")
    if confined and not path.is_relative_to(root):
        raise ValueError(f"native input escapes bundled runtime: {value}")
    return path


def relocate_geometry_paths(value, base):
    if not isinstance(value, dict) or "datasets" not in value or "port_basis" not in value:
        return value
    result = dict(value)
    result["port_basis"] = str(resolve_native_input(base, value["port_basis"]))
    result["datasets"] = {key: str(resolve_native_input(base, path))
                          for key, path in value["datasets"].items()}
    return result
