"""Configure working storage and relocate frozen native references."""

import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import warnings
from functools import lru_cache
from dataclasses import asdict, dataclass, field, is_dataclass


ROOT = Path(__file__).resolve().parent
RECORDED_PROJECT_ROOT = Path("/app/nanocad/projects/personal/zchen/YAP_STACK")
_DATA_ROOT_SETTING = os.environ.get("YAP_DATA_ROOT")
P2_COMPACT_MANIFEST_SHA256 = "cba9f64bdf41b610ec9b538858f07c119f7febd89515f84d8ac6df2e74bd8612"
HISTORICAL_NATIVE_ROOTS = tuple(Path(value) for value in (
    "/w/ee.00/puneet/zhichao/Desktop/YAP_STACK/Ansys_Verify/12_stack_model_validation",
    "/u1/ee/zhichao/Desktop/YAP_STACK/Ansys_Verify/12_stack_model_validation",
    "/app/nanocad/projects/personal/zchen/YAP_STACK/Ansys_Verify/12_stack_model_validation",
))


class StorageWarning(UserWarning):
    """An explicitly chosen location is usable but needs deployment care."""


@lru_cache(maxsize=256)
def _storage_warning(message):
    warnings.warn(message, StorageWarning, stacklevel=3)


def _configured_data_root():
    value = _DATA_ROOT_SETTING if _DATA_ROOT_SETTING is not None else str(RECORDED_PROJECT_ROOT)
    path = Path(value).expanduser()
    if (not value or not path.is_absolute() or '..' in path.parts
            or {'#snapshot', '.snapshot'}.intersection(path.parts)):
        raise ValueError('YAP_DATA_ROOT must be an absolute non-snapshot directory')
    resolved = path.resolve()
    if resolved == Path('/') or {'#snapshot', '.snapshot'}.intersection(resolved.parts):
        raise ValueError('YAP_DATA_ROOT must name a non-snapshot data directory, not /')
    return resolved


def _read_locations():
    """Explicit read overrides are narrow boundaries, not common-ancestor grants."""
    directories, files = [COMPONENT_PROJECT_ROOT], []
    for key in ('YAP_NATIVE_RUNTIME_DIR', 'YAP_W2W_SI_CAP_LIBRARY_DIR', 'YAP_THICKNESS_FAMILY_DIR'):
        if key in os.environ:
            path = Path(os.environ[key]).expanduser()
            if not os.environ[key] or not path.is_absolute():
                raise ValueError(f'{key} must be a nonempty absolute path')
            directories.append(shared_distribution_path(path).resolve())
    for key in ('YAP_COMPONENT_OPERATOR_CATALOG', 'YAP_COMPONENT_OPERATOR_CATALOG_V2', 'YAP_RUNTIME_RESOURCES'):
        if key in os.environ:
            path = Path(os.environ[key]).expanduser()
            if not os.environ[key] or not path.is_absolute():
                raise ValueError(f'{key} must be a nonempty absolute path')
            files.append(shared_distribution_path(path))
    return tuple(directories), tuple(files)


def _storage_path(value, label):
    if not isinstance(value, (str, os.PathLike)) or not os.fspath(value):
        raise ValueError(f"{label} must be a nonempty absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute() or path == Path('/') or '..' in path.parts or len(path.parts) > 64:
        raise ValueError(f"{label} must be an absolute path without traversal")
    resolved = path.resolve()
    for candidate in (path, resolved):
        if any(part in {'#snapshot', '.snapshot', '.git', 'Ansys_Verify', 'mapdl_run_roots', 'mapdl_archives'}
               or part.startswith(('.migration_hold_', 'storage_migration')) for part in candidate.parts):
            raise ValueError(f"{label}: protected or snapshot paths are not allowed")
    project = COMPONENT_PROJECT_ROOT
    read_directories, read_files = _read_locations()
    immutable = (ROOT, *[base/name for base in {project, RECORDED_PROJECT_ROOT}
                        for name in ('YAP', 'distribution', 'operator_repair_20261015',
                                     'operator_night_20261021')],
                 *[p for p in read_directories if p != project])
    if any(resolved.is_relative_to(p.resolve()) for p in immutable) or resolved in read_files:
        raise ValueError(f"{label}: source and shared mechanics data are read-only")
    if path != resolved:
        _storage_warning(f'{label}: using resolved storage path {resolved} instead of alias {path}')
    if not resolved.is_relative_to('/app'):
        _storage_warning(f'{label}: {resolved} is outside the recommended /app storage; '
                         'check capacity, persistence and the host storage policy')
    return resolved


def user_run_root():
    """An explicitly selected run directory, never a source-directory fallback."""
    value = os.environ.get('YAP_RUN_ROOT')
    if value is None:
        raise ValueError('set YAP_RUN_ROOT to your own absolute run directory')
    root = _storage_path(value, 'YAP_RUN_ROOT')
    if root.exists():
        info = root.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise NotADirectoryError('YAP_RUN_ROOT must be a directory')
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            _storage_warning(f'YAP_RUN_ROOT: {root} is not a private UID-owned 0700 directory; '
                             'only use trusted writable storage')
        if not os.access(root, os.W_OK | os.X_OK):
            raise PermissionError('YAP_RUN_ROOT is not writable')
    return root


def validate_user_write_path(value, *, label='output'):
    """Honor explicit storage choices without creating or chmodding files."""
    root = user_run_root()
    path = _storage_path(value, label)
    if not path.is_relative_to(root):
        _storage_warning(f'{label}: {path} is outside YAP_RUN_ROOT ({root}); '
                         'include this directory when accounting for or cleaning run data')
    for prefix in (root, *[p for p in reversed(path.parents) if p.is_relative_to(root) and p != root], path):
        try:
            info = prefix.lstat()
        except FileNotFoundError:
            continue
        if info.st_uid != os.getuid() or info.st_mode & 0o022:
            _storage_warning(f'{label}: {prefix} is shared or owned by another UID; '
                             'use only trusted writers for caches and results')
    return path


def validate_write_path(value, *, label="output"):
    """Reject forbidden destinations, including aliases, without creating files."""
    if not os.fspath(value):
        raise ValueError(f"{label} must not be empty")
    path = Path(value).expanduser().absolute()
    if 'YAP_RUN_ROOT' in os.environ:
        return validate_user_write_path(path, label=label)
    return _storage_path(path, label)


def configure_runtime_storage():
    """Initialize caches and Python temporary storage before third-party imports."""
    run_root = user_run_root()
    override = os.environ.get("YAP_WORK_DIR")
    if override is not None and (not override or not Path(override).expanduser().is_absolute()):
        raise ValueError("YAP_WORK_DIR must be a nonempty absolute path")
    work = validate_user_write_path(override if override is not None else run_root / "work",
                               label="YAP_WORK_DIR")
    temporary = next((os.environ[key] for key in ("TMPDIR", "TEMP", "TMP")
                      if key in os.environ), str(work / "temp"))
    defaults = {"TMPDIR": temporary, "TMP": temporary, "TEMP": temporary,
                "MPLCONFIGDIR": str(work / "matplotlib"),
                "NUMBA_CACHE_DIR": str(work / "numba"),
                'XDG_CACHE_HOME': str(work/'xdg-cache'),
                'XDG_CONFIG_HOME': str(work/'xdg-config'),
                'XDG_DATA_HOME': str(work/'xdg-data'),
                'XDG_STATE_HOME': str(work/'xdg-state'),
                'XDG_RUNTIME_DIR': str(work/'xdg-runtime'),
                'PIP_CACHE_DIR': str(work/'pip'),
                'PYTHONUSERBASE': str(work/'python-user'),
                'JOBLIB_TEMP_FOLDER': str(work/'temp')}
    directories = {}
    for key, default in defaults.items():
        value = os.environ.get(key, default)
        if not value or not Path(value).expanduser().is_absolute():
            raise ValueError(f"{key} must be a nonempty absolute path")
        directories[key] = validate_user_write_path(value, label=key)
    directories['HOME'] = validate_user_write_path(work/'home', label='runtime HOME')
    if "PYTHONPYCACHEPREFIX" in os.environ:
        validate_user_write_path(os.environ["PYTHONPYCACHEPREFIX"], label="PYTHONPYCACHEPREFIX")
    published_caps = _component_path(os.environ.get("YAP_W2W_SI_CAP_LIBRARY_DIR",
        COMPONENT_PROJECT_ROOT / "distribution/operators/silicon_caps"))
    # Validate every override before any mkdir; never probe tempfile's fallback paths.
    os.umask(0o077)
    run_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    user_run_root()
    for directory in set(directories.values()) | {work}:
        validate_user_write_path(directory)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not os.access(directory, os.W_OK | os.X_OK):
            raise PermissionError(f"working directory is not writable: {directory}")
    os.environ.update({key: str(directory) for key, directory in directories.items()})
    os.environ['YAP_WORK_DIR'] = str(work)
    os.environ["YAP_W2W_SI_CAP_LIBRARY_DIR"] = str(published_caps)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    tempfile.tempdir = str(directories["TMPDIR"])
    return {"work": work, **directories}


def mechanics_work_directory():
    """Use the explicitly selected mechanics work disk without a hidden fallback."""
    if "YAP_WORK_DIR" not in os.environ:
        raise ValueError("set YAP_WORK_DIR for mechanics builds")
    work = validate_write_path(os.environ["YAP_WORK_DIR"], label="YAP_WORK_DIR")
    configure_runtime_storage()
    path = validate_write_path(work / "mechanics", label="mechanics scratch")
    path.mkdir(parents=True, exist_ok=True)
    return path


def select_native_root(root=ROOT):
    override = os.environ.get("YAP_NATIVE_RUNTIME_DIR")
    bundled = Path(root) / "runtime"
    selected_data = False
    if override is not None:
        candidate, portable = Path(override), True
    elif Path(root) == ROOT and _DATA_ROOT_SETTING is not None:
        candidate, portable = COMPONENT_PROJECT_ROOT / 'distribution/operators/components/runtime/native', True
        selected_data = True
    elif bundled.exists() or bundled.is_symlink() or (Path(root) / "ASSEMBLY_RUNTIME.json").exists():
        candidate, portable = bundled / "native", True
    else:
        candidate, portable = Path(root).parent / "Ansys_Verify/12_stack_model_validation", False
    if {"#snapshot", ".snapshot"}.intersection(candidate.parts):
        raise ValueError("snapshot paths are not allowed")
    candidate = shared_distribution_path(candidate).resolve()
    if {"#snapshot", ".snapshot"}.intersection(candidate.parts):
        raise ValueError("snapshot paths are not allowed")
    if portable and override is None and not selected_data and not candidate.is_relative_to(Path(root).resolve()):
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
    return shared_distribution_path(Path(override) if override is not None else (
        local if _DATA_ROOT_SETTING is None and local.exists() else root.parent / "RESOURCES.json"))


def load_resource_map(path):
    path = shared_distribution_path(path)
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
    root = shared_distribution_path(root)
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
    root = shared_distribution_path(_runtime_value("NATIVE_ROOT") if native_root is None else native_root).resolve()
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
            path = shared_distribution_path(path)
    else:
        path = shared_distribution_path(base) / path
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


# Component admission is separate from all legacy package and resource registries.
COMPONENT_PROJECT_ROOT = _configured_data_root()
COMPONENT_CATALOG_RELATIVE = "distribution/operators/legacy/component_catalog/CATALOG.json"
DISTRIBUTION_RELOCATIONS = {
    "yap_component_operators_20261021_v1": "components",
    "yap_thickness_family_20261021_v2": "thickness_family",
    "yap_w2w_si_caps_20261021_v1": "silicon_caps",
    "yap_prepared_p2_20261006_v1": "legacy/prepared",
    "yap_component_operators_20261019_v1": "legacy/component_catalog",
}
ARTIFACT_LOCATIONS_SHA256 = "d01d5b558c74d88939ea03f8307d10f1fb70c53fa9a5f25f8ee6c0a60cbeb958"
ARTIFACT_LOCATION_SUPPLEMENTS = (
    ("LOCATIONS_OUTER3_20261022_v1.json",
     "0f8c9a60ea398be7eea10de8d6a747f50403ff15041086bedb62d916363709c1"),
    ("LOCATIONS_BI6_SI17_20261022_v1.json",
     "87f0ab8699f0bbe5b392d1429688718d471118f7d139ed46d832bccd6f307ef6"),
    ("LOCATIONS_BI3_SI17_20261022_v1.json",
     "19950a107df9cdd5a0bbdac45ad36df5b6c5b36f44bdc8158a10124d4d1b8d7a"),
    ("LOCATIONS_BI3_SI7_20261022_v1.json",
     "9114a19a555826ce5f4ba41a497aea996dfc7e457407401d0eb41de27918f505"),
    ("LOCATIONS_BI6_SI77_20261022_v1.json",
     "cbf256311a5806703004d25b0c169c934a1de2590c4df3119eb38c783a928743"),
    ("LOCATIONS_BI3_SI77_20261022_v1.json",
     "a3d5d582ad109ff423dbe502fd7837dae251b32d06a3f2e8fb009ac3375136e1"),
    ("LOCATIONS_BI6_SI87_20261022_v1.json",
     "768ba12f2782fdd85d2a7823fcd7e307104dc3b5da3dbef6f2de539df6e80cee"),
    ("LOCATIONS_MIDDLE_20261023_v7.json",
     "0797a2cf70314384c1cc8ba17b109b63a09835de999c51e5b2d9641677c26f3f"),
)
COMPONENT_VALIDATION_SCOPE = "VALIDATED_GLOBAL_BOW_ORIGINAL1081"
COMPONENT_FORMAL_ACCEPTANCE_SHA256 = "3581258499ad4f035c899ca9eb338b5317f3664087210dfe2a12e6b80db21cbc"
COMPONENT_RECIPE_SHA256 = {
    "h48p5": "bd00672746467dba066d5a813074fdeb19c515f6e3dfe55f3aeb12fa8f17e7ed",
    "h87": "c3d2f41c32a29193506aa4174d5efe8dfb1493a1c3e904ccc75bea76c49e5acd",
}
_COMPONENT_JSON_LIMIT = 1024**2
_COMPONENT_FILE_LIMIT = 2 * 1024**3
_COMPONENT_CHECK_LIMIT = 8 * 1024**3
_COMPONENT_GEOMETRY_FIELDS = frozenset((
    "bi_pitch_um", "diameter_pitch_ratio", "bi_thickness_um", "beol_thickness_um",
    "si_tsv_thickness_um", "total_thickness_um", "bi_layer_count", "tsv_pitch_um",
    "cell_side_um",
))


def _artifact_location_files(project, filename, expected_sha256):
    if (not isinstance(filename, str) or Path(filename).name != filename
            or not filename.endswith('.json') or filename in {'#snapshot', '.snapshot'}):
        raise ValueError("literal artifact locations filename required")
    path = project / "distribution/operators/artifacts" / filename
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    import hashlib
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        identity = _component_identity(stream)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 128 * 1024 or before.st_mode & 0o222:
            raise ValueError("bounded read-only artifact locations required")
        raw = stream.read(128 * 1024 + 1)
        if identity != _component_identity(stream) or hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("artifact locations changed or SHA256 mismatch")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate artifact locations key")
            result[key] = value
        return result
    document = json.loads(raw, object_pairs_hook=unique)
    if (set(document) != {"schema", "files"} or document["schema"] != "yap.operator_artifact_locations.v1"
            or not isinstance(document["files"], list) or not 1 <= len(document["files"]) <= 256):
        raise ValueError("invalid artifact locations schema")
    files = set()
    for value in document["files"]:
        if not isinstance(value, str):
            raise ValueError("literal artifact location required")
        relative = Path(value)
        if (relative.is_absolute() or not relative.parts or relative.parts[0] != "operator_repair_20261015"
                or relative.suffix not in (".json", ".npz", ".py") or str(relative) != value
                or len(relative.parts) > 40 or any(part in {"..", "#snapshot", ".snapshot", "Ansys_Verify",
                    "mapdl_run_roots", "mapdl_archives", "archives", "source_snapshot"}
                    or part.startswith((".migration_hold_", "storage_migration")) for part in relative.parts)
                or relative in files):
            raise ValueError("unsafe or duplicate artifact location")
        files.add(relative)
    return frozenset(files)


@lru_cache(maxsize=4)
def _artifact_locations(project):
    """Union independently pinned releases without rewriting the original map."""
    if len(ARTIFACT_LOCATION_SUPPLEMENTS) > 8:
        raise ValueError("bounded artifact supplement list required")
    files = set(_artifact_location_files(project, 'LOCATIONS.json', ARTIFACT_LOCATIONS_SHA256))
    names = {'LOCATIONS.json'}
    for filename, expected_sha256 in ARTIFACT_LOCATION_SUPPLEMENTS:
        if filename in names:
            raise ValueError("duplicate artifact supplement filename")
        additions = _artifact_location_files(project, filename, expected_sha256)
        if files.intersection(additions):
            raise ValueError("artifact location releases must be disjoint")
        files.update(additions)
        names.add(filename)
    # Exact directory aliases support receipt comparisons, never prefix routing.
    locations = set(files)
    for relative in files:
        locations.update(parent for parent in relative.parents if parent != Path("."))
    return frozenset(locations)


def shared_distribution_path(value):
    """Resolve recorded identities into configured storage without rewriting JSON."""
    if os.environ.get('YAP_DATA_ROOT') != _DATA_ROOT_SETTING:
        raise ValueError('set YAP_DATA_ROOT before importing YAP; restart Python to change data roots')
    path = Path(value)
    if {"..", "#snapshot", ".snapshot"}.intersection(path.parts):
        raise ValueError("shared distribution path must not enter snapshots or traverse parents")
    if _DATA_ROOT_SETTING:
        alias = Path(_DATA_ROOT_SETTING).expanduser()
        if alias != COMPONENT_PROJECT_ROOT and path.is_relative_to(alias):
            path = COMPONENT_PROJECT_ROOT / path.relative_to(alias)
    if (COMPONENT_PROJECT_ROOT != RECORDED_PROJECT_ROOT and path.is_relative_to(RECORDED_PROJECT_ROOT)
            and not path.is_relative_to(COMPONENT_PROJECT_ROOT)):
        relative = path.relative_to(RECORDED_PROJECT_ROOT)
        if relative.parts and relative.parts[0] in ('distribution', 'operator_repair_20261015',
                                                    'operator_night_20261021'):
            path = COMPONENT_PROJECT_ROOT / relative
    distribution = COMPONENT_PROJECT_ROOT / "distribution"
    for old_name, new_name in DISTRIBUTION_RELOCATIONS.items():
        previous = distribution / old_name
        if path.is_relative_to(previous):
            return distribution / "operators" / new_name / path.relative_to(previous)
    if path.is_relative_to(COMPONENT_PROJECT_ROOT / "operator_repair_20261015"):
        relative = path.relative_to(COMPONENT_PROJECT_ROOT)
        if relative in _artifact_locations(COMPONENT_PROJECT_ROOT):
            return distribution / "operators/artifacts" / relative
    return path


def _component_path(value, base=None):
    if not isinstance(value, (str, Path)) or not str(value) or len(str(value)) > 4096:
        raise ValueError("component path must be a bounded nonempty path")
    path = Path(value).expanduser()
    if ".." in path.parts or len(path.parts) > 48:
        raise ValueError("component path traversal or excessive depth")
    if not path.is_absolute():
        if base is None:
            raise ValueError("component catalog path must be absolute")
        path = base / path
    path = shared_distribution_path(path)
    for key in ('YAP_NATIVE_RUNTIME_DIR', 'YAP_W2W_SI_CAP_LIBRARY_DIR', 'YAP_THICKNESS_FAMILY_DIR'):
        if os.environ.get(key):
            alias = Path(os.environ[key]).expanduser()
            if path.is_relative_to(alias):
                path = alias.resolve() / path.relative_to(alias)
    directories, files = _read_locations()
    if not any(path.is_relative_to(root) and (path != root or root != COMPONENT_PROJECT_ROOT)
               for root in directories) and path not in files:
        raise ValueError("component path is outside YAP_DATA_ROOT and explicit read overrides")
    if any(part in ("#snapshot", ".snapshot", "Ansys_Verify", "mapdl_run_roots", "mapdl_archives")
           or part.startswith((".migration_hold_", "storage_migration")) for part in path.parts):
        raise ValueError("component paths must not access snapshots or protected migration data")
    return path


def component_operator_catalog_path(catalog_path=None):
    """Return the absolute shared catalog path, independent of the caller's CWD."""
    value = catalog_path if catalog_path is not None else os.environ.get(
        "YAP_COMPONENT_OPERATOR_CATALOG", COMPONENT_PROJECT_ROOT / COMPONENT_CATALOG_RELATIVE)
    return _component_path(value)


def _component_open(path, limit, expected_size=None):
    """Open a regular file with no-follow guards on every path component."""
    path = _component_path(path)
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            next_directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                     dir_fd=directory)
            os.close(directory)
            directory = next_directory
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
                             dir_fd=directory)
    finally:
        os.close(directory)
    stream = os.fdopen(descriptor, "rb")
    try:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= limit:
            raise ValueError(f"component input must be a bounded nonempty regular file: {path}")
        if expected_size is not None and metadata.st_size != expected_size:
            raise ValueError(f"component byte count mismatch: {path}")
    except BaseException:
        stream.close()
        raise
    return stream


def _component_identity(stream):
    metadata = os.fstat(stream.fileno())
    return (metadata.st_dev, metadata.st_ino, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns)


def _component_sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("component SHA256 must contain 64 lowercase hexadecimal characters")
    return value


def _component_json(path, expected=None):
    import hashlib

    with _component_open(path, _COMPONENT_JSON_LIMIT) as stream:
        before = _component_identity(stream)
        payload = stream.read(_COMPONENT_JSON_LIMIT + 1)
        if len(payload) > _COMPONENT_JSON_LIMIT or before != _component_identity(stream):
            raise ValueError("component JSON changed or exceeded its byte limit")
    observed = hashlib.sha256(payload).hexdigest()
    if expected is not None and observed != _component_sha(expected):
        raise ValueError(f"component JSON SHA256 mismatch: {path}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate component JSON key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"nonfinite component JSON constant: {value}")

    document = json.loads(payload, object_pairs_hook=unique, parse_constant=invalid_constant)
    if not isinstance(document, dict):
        raise ValueError("component JSON must be an object")
    return document, observed


def _component_geometry(value):
    import math

    if not isinstance(value, dict) or set(value) != _COMPONENT_GEOMETRY_FIELDS:
        raise ValueError("component geometry requires every exact declared field, including BI count")
    for key, number in value.items():
        try:
            valid = type(number) in (int, float) and math.isfinite(number) and number > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"component geometry {key} must be finite and positive")
    if type(value["bi_layer_count"]) is not int or value["bi_layer_count"] not in (1, 2):
        raise ValueError("component bi_layer_count must be the exact integer 1 or 2")
    if value["diameter_pitch_ratio"] > 1:
        raise ValueError("component diameter_pitch_ratio must not exceed one")
    total = (value["si_tsv_thickness_um"] + value["beol_thickness_um"]
             + value["bi_layer_count"] * value["bi_thickness_um"])
    if not math.isclose(total, value["total_thickness_um"], rel_tol=0., abs_tol=1e-9):
        raise ValueError("component total thickness differs from its declared BI/BEOL/Si ledger")
    return dict(value)


def _component_semantics(entry, acceptance, recipe, reference):
    """Check the admitted case and coordinate/artifact chain, not dimensions alone."""
    geometry = entry["geometry"]
    case = {(48.5, 60.): "h48p5", (87., 98.5): "h87"}.get(
        (geometry["si_tsv_thickness_um"], geometry["total_thickness_um"]))
    if (case is None or case not in [entry["id"], *entry.get("aliases", [])]
            or geometry["bi_layer_count"] != 1 or geometry["tsv_pitch_um"] != 6.):
        raise ValueError("component geometry/alias is outside the admitted single-BI cases")
    identity = entry["coordinate_identity"]
    if (entry["acceptance"]["sha256"] != COMPONENT_FORMAL_ACCEPTANCE_SHA256
            or identity["recipe"]["sha256"] != COMPONENT_RECIPE_SHA256[case]):
        raise ValueError("component evidence is not bound to the pinned formal acceptance and case recipe")
    if (acceptance.get("schema") != "yap.global_bow_strict_or_admission.v1"
            or acceptance.get("state") != "BOTH_ORIGINAL1081_GLOBAL_BOW_ACCURACY_PASS"
            or acceptance.get("global_accuracy_passed") is not True):
        raise ValueError("component formal global-bow acceptance is not a pass")
    scope, cases = acceptance.get("scope"), acceptance.get("cases")
    expected_scope = {"states": 1081, "cycles": 3, "BI_pitch_um": geometry["bi_pitch_um"],
                      "diameter_pitch_ratio": geometry["diameter_pitch_ratio"],
                      "BI_um": geometry["bi_thickness_um"], "BEOL_um": geometry["beol_thickness_um"],
                      "cell_xz_um": [geometry["cell_side_um"], geometry["cell_side_um"]],
                      "projection_xz_mm": [10, 10],
                      "warpage_scope": "macro_homogenized_projection_not_finite_die_validation",
                      "no_arbitrary_geometry_or_assembly_order_accuracy_claim": True}
    if (not isinstance(scope, dict) or any(scope.get(key) != value for key, value in expected_scope.items())
            or not isinstance(cases, dict) or not isinstance(cases.get(case), dict)):
        raise ValueError("component geometry or original1081 ROM scope differs from formal acceptance")
    accepted = cases[case]
    if (accepted.get("Si_TSV_um") != geometry["si_tsv_thickness_um"]
            or accepted.get("total_um") != geometry["total_thickness_um"]
            or accepted.get("operator_admission") != "ACCURACY_ONLY_FOR_THE_EXPLICIT_VALIDATED_SCOPE"):
        raise ValueError("component case geometry differs from formal acceptance")
    for name in ("native_fullspace_accuracy", "reduced_operator_accuracy"):
        accuracy = accepted.get(name)
        if (not isinstance(accuracy, dict) or accuracy.get("global_accuracy_passed") is not True
                or accuracy.get("matched_rows") != 1081 or accuracy.get("failed_sample_probe_count") != 0):
            raise ValueError("component original1081 case accuracy is incomplete")
    if (recipe.get("schema") != "yap.component_cu_riesz_operator.v1" or recipe.get("case") != case
            or recipe.get("state") != "CLOSED_COMMON_COMPONENT_CU_RIESZ_OPERATORS_UNSCORED"
            or reference.get("schema") != "yap.native_component_reference.v1"
            or reference.get("case") != case
            or reference.get("state") != "CLOSED_NATIVE_COMPONENT_REFERENCE_NOT_REDUCED_OR_ADMITTED"):
        raise ValueError("component recipe/reference case or schema mismatch")

    def binding_matches(left, right, base, *, with_bytes=True):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        return (left.get("sha256") == right.get("sha256")
                and (not with_bytes or left.get("bytes") == right.get("bytes"))
                and _component_path(left.get("path"), base) == _component_path(right.get("path"), base))

    recipe_base = Path(identity["recipe"]["path"]).parent
    if not binding_matches(recipe.get("reference"), identity["reference"], recipe_base, with_bytes=False):
        raise ValueError("component coordinate reference is not the case recipe reference")
    construction = acceptance.get("common_construction")
    if (not isinstance(construction, dict)
            or _component_path(construction.get("catalog")) != recipe_base):
        raise ValueError("component recipe is outside the formally accepted shared catalog")
    source = construction.get("source")
    if (not isinstance(recipe.get("source_sha256"), dict)
            or recipe["source_sha256"].get(source) != construction.get("source_sha256")):
        raise ValueError("component construction source differs from formal acceptance")
    sources, references = recipe.get("components"), reference.get("artifacts")
    if not isinstance(sources, dict) or not isinstance(references, dict):
        raise ValueError("component recipe/reference artifacts are missing")
    for role, artifact in entry["components"].items():
        component = sources.get(role)
        if (not isinstance(component, dict)
                or not binding_matches(component.get("artifact"), artifact, recipe_base)
                or component.get("micro_rank") != artifact["micro_rank"]):
            raise ValueError("component artifact or micro rank differs from the accepted case recipe")
        pooled = component.get("pooled_reference_identities")
        regional_reference = pooled.get(case) if isinstance(pooled, dict) else component.get("reference")
        if not binding_matches(regional_reference, references.get(role), recipe_base):
            raise ValueError("component regional coordinate identity differs from its case reference")


def load_component_operator_catalog(catalog_path=None):
    """Verify bounded catalog/evidence JSON and NPZ metadata, not NPZ payloads.

    The publisher binds scientific admission to immutable acceptance evidence.
    This loader checks those bindings; it does not rerun the acceptance analysis.
    No legacy native package is selected or substituted.
    """
    path = component_operator_catalog_path(catalog_path)
    catalog, catalog_sha = _component_json(path)
    if set(catalog) != {"schema", "entries"} or catalog["schema"] != "yap.component_operator_catalog.v1":
        raise ValueError("invalid component operator catalog schema")
    entries = catalog["entries"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 64:
        raise ValueError("component catalog requires 1-64 entries")
    required = {"id", "geometry", "components", "acceptance", "coordinate_identity",
                "validation_scope", "globalbowvalidated_original1081", "assembly_execution_supported"}
    seen, geometries, evidence, artifact_bindings = set(), [], {}, {}

    def bound_json(binding):
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError("component evidence requires path and sha256")
        reference = _component_path(binding["path"], path.parent)
        pin = _component_sha(binding["sha256"])
        key = (reference, pin)
        if key not in evidence:
            evidence[key] = _component_json(reference, pin)[0]
        binding["path"] = str(reference)
        return evidence[key]

    for entry in entries:
        if not isinstance(entry, dict) or not required <= set(entry) <= required | {"aliases"}:
            raise ValueError("invalid component catalog entry fields")
        aliases = entry.get("aliases", [])
        if not isinstance(aliases, list) or len(aliases) > 16:
            raise ValueError("component aliases must be a bounded list")
        for name in [entry["id"], *aliases]:
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", name):
                raise ValueError("invalid component entry ID or alias")
            if name in seen:
                raise ValueError("duplicate component ID or alias")
            seen.add(name)
        geometry = _component_geometry(entry["geometry"])
        if geometry in geometries:
            raise ValueError("ambiguous duplicate component geometry")
        geometries.append(geometry)
        if (entry["validation_scope"] != COMPONENT_VALIDATION_SCOPE
                or entry["globalbowvalidated_original1081"] is not True
                or entry["assembly_execution_supported"] is not False):
            raise ValueError("component admission must be original1081 global bow only; assembly is unsupported")
        acceptance = bound_json(entry["acceptance"])
        identity = entry["coordinate_identity"]
        if not isinstance(identity, dict) or set(identity) != {"recipe", "reference"}:
            raise ValueError("component coordinate identity requires bound recipe and reference manifests")
        recipe = bound_json(identity["recipe"])
        reference = bound_json(identity["reference"])
        components = entry["components"]
        if not isinstance(components, dict) or set(components) != {"bi", "beol", "si"}:
            raise ValueError("component entry requires bi, beol and si")
        for component in components.values():
            if not isinstance(component, dict) or set(component) != {"path", "sha256", "bytes", "micro_rank"}:
                raise ValueError("component artifact requires path, sha256, bytes and micro_rank")
            _component_sha(component["sha256"])
            if type(component["bytes"]) is not int or not 0 < component["bytes"] <= _COMPONENT_FILE_LIMIT:
                raise ValueError("component artifact byte limit exceeded")
            if type(component["micro_rank"]) is not int or not 1 <= component["micro_rank"] <= 4096:
                raise ValueError("invalid component micro_rank")
            artifact = _component_path(component["path"], path.parent)
            if artifact.suffix != ".npz":
                raise ValueError("component artifact must be an NPZ")
            signature = (component["sha256"], component["bytes"], component["micro_rank"])
            if artifact_bindings.setdefault(artifact, signature) != signature:
                raise ValueError("conflicting bindings for shared component artifact")
            with _component_open(artifact, _COMPONENT_FILE_LIMIT, component["bytes"]):
                pass
            component["path"] = str(artifact)
        _component_semantics(entry, acceptance, recipe, reference)
    return {**catalog, "catalog_path": str(path), "catalog_sha256": catalog_sha,
            "verification": "JSON_BINDINGS_AND_FILE_METADATA_VERIFIED"}


def resolve_component_operator(geometry, *, catalog_path=None, entry_id=None, for_assembly=False):
    """Select exactly one declared geometry; never authorize assembly execution."""
    if type(for_assembly) is not bool:
        raise ValueError("for_assembly must be Boolean")
    if for_assembly:
        raise RuntimeError("component global-bow admission does not support assembly execution")
    geometry = _component_geometry(geometry)
    catalog = load_component_operator_catalog(catalog_path)
    matches = [entry for entry in catalog["entries"] if entry["geometry"] == geometry
               and (entry_id is None or entry_id in [entry["id"], *entry.get("aliases", [])])]
    if len(matches) != 1:
        raise ValueError("no unique exact component geometry/BI-count match; no scaling or native64 fallback")
    return matches[0]


def _component_hash_stream(stream, expected, deadline):
    import hashlib
    import time

    before, digest, count = _component_identity(stream), hashlib.sha256(), 0
    started, rate = time.monotonic(), 32 * 1024**2
    while True:
        remaining = max(0, before[2] - count)
        size = min(1024**2, remaining)
        # Charge each read before issuing it; do not bank idle-time burst credit.
        target = max(started + (count + size) / rate, time.monotonic() + size / rate)
        if target > deadline:
            raise ValueError("component payload verification time budget exceeded")
        time.sleep(max(0., target - time.monotonic()))
        if time.monotonic() > deadline:
            raise ValueError("component payload verification time budget exceeded")
        if not remaining:
            break
        block = stream.read(size)
        if not block:
            break
        count += len(block)
        if count > before[2]:
            raise ValueError("component artifact grew during verification")
        digest.update(block)
    if before != _component_identity(stream) or count != before[2]:
        raise ValueError("component artifact changed during verification")
    if digest.hexdigest() != expected:
        raise ValueError("component artifact SHA256 mismatch")


def check_component_operator_catalog(catalog_path=None):
    """Explicitly hash unique component payloads (8 GiB/120 s maximum)."""
    import time

    deadline = time.monotonic() + 120.
    catalog = load_component_operator_catalog(catalog_path)
    artifacts = {}
    for entry in catalog["entries"]:
        for component in entry["components"].values():
            old = artifacts.setdefault(component["path"], component)
            if any(old[key] != component[key] for key in ("sha256", "bytes", "micro_rank")):
                raise ValueError("conflicting bindings for shared component artifact")
    if sum(component["bytes"] for component in artifacts.values()) > _COMPONENT_CHECK_LIMIT:
        raise ValueError("component catalog exceeds the 8 GiB payload verification budget")
    for component in artifacts.values():
        with _component_open(component["path"], _COMPONENT_FILE_LIMIT, component["bytes"]) as stream:
            _component_hash_stream(stream, component["sha256"], deadline)
    return {**catalog, "verification": "ALL_COMPONENT_SHA256_VERIFIED",
            "unique_artifacts_checked": len(artifacts)}


def load_component_operator_arrays(geometry, role, *, catalog_path=None, entry_id=None,
                                   max_uncompressed_bytes=_COMPONENT_FILE_LIMIT):
    """Lazily load immutable numeric arrays, not an executable assembly library."""
    import time
    import zipfile

    if role not in ("bi", "beol", "si"):
        raise ValueError("component role must be bi, beol or si")
    if type(max_uncompressed_bytes) is not int or not 0 < max_uncompressed_bytes <= _COMPONENT_FILE_LIMIT:
        raise ValueError("invalid component decompression budget")
    entry = resolve_component_operator(geometry, catalog_path=catalog_path, entry_id=entry_id)
    component = entry["components"][role]
    with _component_open(component["path"], _COMPONENT_FILE_LIMIT, component["bytes"]) as stream:
        _component_hash_stream(stream, component["sha256"], time.monotonic() + 120.)
        before = _component_identity(stream)
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            members = archive.infolist()
            if (not 1 <= len(members) <= 128 or len({item.filename for item in members}) != len(members)
                    or sum(item.file_size for item in members) > max_uncompressed_bytes
                    or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*\.npy", item.filename)
                           for item in members)):
                raise ValueError("invalid or oversized component NPZ inventory")
            import math
            import numpy as np

            # Inspect array headers before NumPy can allocate from untrusted shapes.
            for item in members:
                with archive.open(item) as member:
                    version = np.lib.format.read_magic(member)
                    if version not in ((1, 0), (2, 0)):
                        raise ValueError("unsupported component NPY version")
                    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                              else np.lib.format.read_array_header_2_0)
                    shape, _, dtype = reader(member, max_header_size=16384)
                    if dtype.hasobject or math.prod(shape) * dtype.itemsize != item.file_size - member.tell():
                        raise ValueError("component array is object-valued or has invalid byte dimensions")
        stream.seek(0)
        with np.load(stream, allow_pickle=False, max_header_size=16384) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if before != _component_identity(stream):
            raise ValueError("component artifact changed during array loading")
    for value in arrays.values():
        value.flags.writeable = False
    return arrays


CORRECTED_MATERIAL_CLOCK = "increment_average_global_time_factor_at_physical_endpoint"
_MATERIAL_KERNEL_NAMES = ("update_batch", "tangent_batch", "elastic_stiffness")
_MATERIAL_CARD_METHODS = ("elastic", "alpha", "yield_stress", "hardening_modulus", "creep_step_coefficient")


@dataclass(frozen=True)
class CorrectedMaterialBinding:
    """Caller-reviewed implementation binding, never a physics admission."""

    material_kernels: object = field(repr=False, compare=False)
    cu_card: object = field(repr=False, compare=False)
    source_path: str
    source_sha256: str
    card_json: str
    kernel_functions: tuple = field(repr=False, compare=False)
    clock_semantics: str = CORRECTED_MATERIAL_CLOCK
    scope: str = "experimental"
    card_methods: tuple = field(default=(), repr=False, compare=False)


@dataclass(frozen=True)
class CorrectedCompiledMaterialBinding(CorrectedMaterialBinding):
    """Explicit multi-source compiled implementation; never assembly admission."""

    implementation: object = field(default=None, repr=False, compare=False)


def _material_card_json(card):
    parameters = getattr(type(card), "__dataclass_params__", None)
    if not is_dataclass(card) or parameters is None or not parameters.frozen:
        raise ValueError("corrected material requires a frozen dataclass Cu card")
    try:
        return json.dumps(asdict(card), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("corrected material Cu card must contain finite JSON values") from exc


def _material_kernel_functions(kernels):
    functions = tuple(getattr(kernels, name, None) for name in _MATERIAL_KERNEL_NAMES)
    if not all(callable(value) and hasattr(value, "__code__") for value in functions):
        raise ValueError("bound material kernels require explicit Python update, tangent and elastic functions")
    return functions


def _material_card_methods(card):
    methods = tuple(getattr(type(card), name, None) for name in _MATERIAL_CARD_METHODS)
    if not all(callable(value) and hasattr(value, "__code__") for value in methods):
        raise ValueError("corrected Cu card requires explicit Python material and increment-clock methods")
    return methods


def bind_corrected_material(*, material_kernels, cu_card, source_path, source_sha256,
                            scope="experimental"):
    """Pin reviewed Python kernels and an immutable card to the physical clock.

    The caller reviews the implementation's semantics. A matching file hash
    and this declaration do not confer constitutive or assembly qualification.
    """
    import time

    if scope not in ("experimental", "synthetic_fixture"):
        raise ValueError("corrected material binding is experimental or synthetic_fixture only")
    path, expected = _component_path(source_path), _component_sha(source_sha256)
    if path.suffix != ".py":
        raise ValueError("corrected material source must be an explicit Python file")
    functions, methods = _material_kernel_functions(material_kernels), _material_card_methods(cu_card)
    if any(Path(value.__code__.co_filename).absolute() != path for value in functions + methods):
        raise ValueError("material kernel implementation differs from the pinned source")
    with _component_open(path, _COMPONENT_JSON_LIMIT) as stream:
        _component_hash_stream(stream, expected, time.monotonic() + 10.)
    # Keep the complete code identities as well as the caller's source-file pin.
    return CorrectedMaterialBinding(material_kernels, cu_card, str(path), expected,
        _material_card_json(cu_card), tuple((value, value.__code__) for value in functions), scope=scope,
        card_methods=tuple((value, value.__code__) for value in methods))


def _validate_corrected_material_binding(binding, *, material_kernels, cu_card):
    if not isinstance(binding, CorrectedMaterialBinding):
        raise ValueError("an explicit CorrectedMaterialBinding is required")
    if (binding.clock_semantics != CORRECTED_MATERIAL_CLOCK
            or binding.scope not in ("experimental", "synthetic_fixture")
            or binding.material_kernels is not material_kernels or binding.cu_card is not cu_card
            or binding.card_json != _material_card_json(cu_card)):
        raise ValueError("corrected material card, kernels or clock binding changed")
    functions = _material_kernel_functions(material_kernels)
    if len(binding.kernel_functions) != len(functions) or any(
            current is not saved or current.__code__ is not code
            for current, (saved, code) in zip(functions, binding.kernel_functions)):
        raise ValueError("corrected material kernel implementation changed")
    methods = _material_card_methods(cu_card)
    if len(binding.card_methods) != len(methods) or any(
            current is not saved or current.__code__ is not code
            for current, (saved, code) in zip(methods, binding.card_methods)):
        raise ValueError("corrected material Cu card or clock implementation changed")
    if isinstance(binding, CorrectedCompiledMaterialBinding):
        try:
            from .W2W.component_compiled_material import validate_compiled_binding
        except ImportError:
            from W2W.component_compiled_material import validate_compiled_binding
        validate_compiled_binding(binding)


def corrected_material_provenance(binding):
    """Return JSON evidence without serializing executable objects."""
    import hashlib
    import marshal

    _validate_corrected_material_binding(binding, material_kernels=binding.material_kernels,
                                         cu_card=binding.cu_card)
    result = dict(schema="yap.corrected_material_binding.v1", scope=binding.scope,
        source_path=binding.source_path, source_sha256=binding.source_sha256,
        clock_semantics=binding.clock_semantics, cu_card=json.loads(binding.card_json),
        kernels={name: dict(qualified_name=function.__qualname__,
            code_sha256=hashlib.sha256(marshal.dumps(code)).hexdigest())
            for name, (function, code) in zip(_MATERIAL_KERNEL_NAMES, binding.kernel_functions)},
        card_methods={name: hashlib.sha256(marshal.dumps(code)).hexdigest()
            for name, (_, code) in zip(_MATERIAL_CARD_METHODS, binding.card_methods)},
        physical_readiness=False, assembly_execution_supported=False)
    if isinstance(binding, CorrectedCompiledMaterialBinding):
        result.update(binding.implementation.provenance())
    return result


def attach_corrected_material_binding(layer, binding):
    """Attach a binding only if the constructed layer retained its exact inputs."""
    _validate_corrected_material_binding(binding, material_kernels=getattr(layer, "material_kernels", None),
                                         cu_card=getattr(layer, "cu_card", None))
    previous = getattr(layer, "corrected_material_binding", None)
    if previous is not None and previous is not binding:
        raise ValueError("cannot replace an existing corrected material binding")
    layer.corrected_material_binding = binding
    return layer


def preserve_corrected_material_binding(source, target):
    """Validate rather than silently repair a cloned operator's material inputs."""
    binding = getattr(source, "corrected_material_binding", None)
    if binding is None:
        if getattr(source, "material_kernels", None) is not None:
            raise ValueError("explicit material kernels lost their corrected binding")
        return target
    _validate_corrected_material_binding(binding, material_kernels=source.material_kernels,
                                         cu_card=source.cu_card)
    return attach_corrected_material_binding(target, binding)


def component_native_layer_arguments(arrays, *, cu_card, material_kernels, macro_coordinate_scale,
                                     material_binding=None):
    """Translate component algebra, without changing basis, Cu law or weights.

    This does not certify a port-library/assembly mapping. Callers must supply
    the corrected material kernels and the intended numerical coordinate scale.
    """
    import numpy as np
    from scipy import sparse

    if material_binding is not None:
        _validate_corrected_material_binding(material_binding, material_kernels=material_kernels,
                                             cu_card=cu_card)
    if not all(callable(getattr(material_kernels, name, None))
               for name in ("update_batch", "tangent_batch", "elastic_stiffness")):
        raise ValueError("explicit corrected update, tangent and elastic kernels are required")
    kaa, c = np.asarray(arrays["kaa"]), np.asarray(arrays["c"])
    shape = tuple(arrays["kff_shape"])
    kff = sparse.csr_matrix((arrays["kff_data"], arrays["kff_indices"], arrays["kff_indptr"]), shape=shape)
    kff.check_format(full_check=True)
    m, n = len(kaa), kff.shape[0]
    if (m != 134 or not 1 <= n <= 48 or kaa.shape != (m,m) or kff.shape != (n,n)
            or c.shape != (n,m)):
        raise ValueError("component reference dimensions differ")
    stiffness = np.block([[kaa,c.T],[c,kff.toarray()]])
    thermal = np.concatenate((arrays["fm"],arrays["ff"]))
    weights = np.asarray(arrays["weights"])
    scale = np.asarray(macro_coordinate_scale,dtype=float)
    if (scale.shape != (m,) or not np.isfinite(scale).all() or np.any(scale <= 0)
            or thermal.shape != (m+n,) or weights.ndim != 1 or np.any(weights <= 0)
            or not np.isfinite(stiffness).all() or not np.isfinite(thermal).all()
            or not np.isfinite(weights).all() or np.any(kff.diagonal() <= 0)):
        raise ValueError("finite component reference, positive weights and scales required")
    if not np.allclose(stiffness,stiffness.T,rtol=1e-10,atol=1e-12):
        raise ValueError("component reference stiffness is not symmetric")
    reference_temperature = float(arrays["reference_temperature_c"])
    if reference_temperature != 25. or cu_card.room_c != reference_temperature:
        raise ValueError("component and Cu reference temperatures must both be 25 C")
    for temperature in (25.,350.):
        elastic = material_kernels.elastic_stiffness(*cu_card.elastic(temperature))
        alpha = np.r_[np.full(3,cu_card.alpha(temperature)),np.zeros(3)]
        if (not np.array_equal(elastic,arrays["reference_cu_stiffness"])
                or not np.array_equal(alpha,arrays["reference_cu_alpha"])):
            raise ValueError("component requires its constant reference Cu stiffness and expansion")
    maps = np.concatenate((np.asarray(arrays["B_macro"]),np.asarray(arrays["B_free"])),axis=1)
    if maps.shape != (6*len(weights),m+n) or not np.isfinite(maps).all():
        raise ValueError("complete ordered component Cu maps are required")
    stiffness.flags.writeable = thermal.flags.writeable = maps.flags.writeable = False

    def load(temperature):
        temperature = float(temperature)
        if not np.isfinite(temperature) or not 25. <= temperature <= 350.:
            raise ValueError("component thermal reference supports 25-350 C only")
        return (temperature-reference_temperature)*thermal

    def matrix(temperature):
        load(temperature)
        return stiffness

    return dict(macro_size=m,bubble_size=n,reference_stiffness=matrix,reference_thermal_load=load,
        strain_maps=maps.reshape(len(weights),6,m+n),weights_mm3=weights,
        macro_coordinate_scale=scale,bubble_coordinate_scale=1./np.sqrt(kff.diagonal()),
        cu_card=cu_card,plastic_active=True,material_kernels=material_kernels)


def load_component_native_layer(geometry, role, *, layer_type, cu_card, material_kernels,
                                macro_coordinate_scale, catalog_path=None, entry_id=None,
                                material_binding=None):
    """Construct an experimental native-API layer; not an assembly library."""
    entry = resolve_component_operator(geometry,catalog_path=catalog_path,entry_id=entry_id)
    arrays = load_component_operator_arrays(geometry,role,catalog_path=catalog_path,entry_id=entry_id)
    arguments = component_native_layer_arguments(arrays,cu_card=cu_card,material_kernels=material_kernels,
        macro_coordinate_scale=macro_coordinate_scale,material_binding=material_binding)
    layer = layer_type(**arguments)
    if material_binding is not None:
        attach_corrected_material_binding(layer, material_binding)
    layer.component_operator_binding = {"catalog_entry":entry["id"],"role":role,
        "artifact":dict(entry["components"][role]),"assembly_execution_supported":False}
    return layer


def _component_catalog_cli(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Inspect shared global-bow-validated components; assembly is unsupported.")
    parser.add_argument("command", choices=("list", "check"))
    parser.add_argument("--catalog", help="absolute catalog path inside the canonical project")
    options = parser.parse_args(argv)
    try:
        catalog = (check_component_operator_catalog if options.command == "check"
                   else load_component_operator_catalog)(options.catalog)
    except (OSError, ValueError, RuntimeError, RecursionError) as exc:
        print(f"Component catalog unavailable or invalid: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"catalog_path": catalog["catalog_path"], "catalog_sha256": catalog["catalog_sha256"],
                      "verification": catalog["verification"], "assembly_execution_supported": False,
                      "entries": [{key: entry.get(key, []) if key == "aliases" else entry[key]
                                   for key in ("id", "aliases", "geometry", "validation_scope",
                                               "globalbowvalidated_original1081", "assembly_execution_supported")}
                                  for entry in catalog["entries"]]}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_component_catalog_cli())
