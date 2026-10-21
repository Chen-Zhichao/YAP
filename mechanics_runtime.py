"""Configure working storage and relocate frozen native references."""

import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from dataclasses import asdict, dataclass, field, is_dataclass


ROOT = Path(__file__).resolve().parent
P2_COMPACT_MANIFEST_SHA256 = "cba9f64bdf41b610ec9b538858f07c119f7febd89515f84d8ac6df2e74bd8612"
HISTORICAL_NATIVE_ROOTS = tuple(Path(value) for value in (
    "/w/ee.00/puneet/zhichao/Desktop/YAP_STACK/Ansys_Verify/12_stack_model_validation",
    "/u1/ee/zhichao/Desktop/YAP_STACK/Ansys_Verify/12_stack_model_validation",
    "/app/nanocad/projects/personal/zchen/YAP_STACK/Ansys_Verify/12_stack_model_validation",
))


def validate_write_path(value, *, label="output"):
    """Reject forbidden destinations, including aliases, without creating files."""
    if not os.fspath(value):
        raise ValueError(f"{label} must not be empty")
    path = Path(value).expanduser().absolute()
    resolved = path.resolve()
    for candidate in (path, resolved):
        if {"#snapshot", ".snapshot"}.intersection(candidate.parts):
            raise ValueError(f"{label}: snapshot paths are not allowed")
        if candidate.is_relative_to("/tmp"):
            raise ValueError(f"{label}: /tmp storage is not allowed; choose a working directory")
    return resolved


def configure_runtime_storage():
    """Initialize caches and Python temporary storage before third-party imports."""
    override = os.environ.get("YAP_WORK_DIR")
    if override is not None and (not override or not Path(override).expanduser().is_absolute()):
        raise ValueError("YAP_WORK_DIR must be a nonempty absolute path")
    work = validate_write_path(override if override is not None else ROOT / ".cache/yap",
                               label="YAP_WORK_DIR")
    temporary = next((os.environ[key] for key in ("TMPDIR", "TEMP", "TMP")
                      if key in os.environ), str(work / "temp"))
    defaults = {"TMPDIR": temporary, "TMP": temporary, "TEMP": temporary,
                "MPLCONFIGDIR": str(work / "matplotlib"),
                "NUMBA_CACHE_DIR": str(work / "numba")}
    directories = {}
    for key, default in defaults.items():
        value = os.environ.get(key, default)
        if not value or not Path(value).expanduser().is_absolute():
            raise ValueError(f"{key} must be a nonempty absolute path")
        directories[key] = validate_write_path(value, label=key)
    if "PYTHONPYCACHEPREFIX" in os.environ:
        validate_write_path(os.environ["PYTHONPYCACHEPREFIX"], label="PYTHONPYCACHEPREFIX")
    # Validate every override before any mkdir; never probe tempfile's fallback paths.
    for directory in set(directories.values()) | {work}:
        directory.mkdir(parents=True, exist_ok=True)
        if not os.access(directory, os.W_OK | os.X_OK):
            raise PermissionError(f"working directory is not writable: {directory}")
    os.environ.update({key: str(directory) for key, directory in directories.items()})
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    tempfile.tempdir = str(directories["TMPDIR"])
    return {"work": work, **directories}


def mechanics_work_directory():
    """Large temporary mechanics jobs require an explicit host-local work disk."""
    if "YAP_WORK_DIR" not in os.environ:
        raise ValueError("set YAP_WORK_DIR to approved host-local scratch for mechanics builds")
    work = validate_write_path(os.environ["YAP_WORK_DIR"], label="YAP_WORK_DIR")
    if any(work.is_relative_to(prefix) for prefix in ("/w", "/u1")):
        raise ValueError("large mechanics work must not use shared-home /w or /u1 storage")
    configure_runtime_storage()
    path = validate_write_path(work / "mechanics", label="mechanics scratch")
    path.mkdir(parents=True, exist_ok=True)
    return path


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


# Component admission is separate from all legacy package and resource registries.
COMPONENT_PROJECT_ROOT = Path("/app/nanocad/projects/personal/zchen/YAP_STACK")
COMPONENT_CATALOG_RELATIVE = "distribution/yap_component_operators_20261019_v1/CATALOG.json"
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


def _component_path(value, base=None):
    if not isinstance(value, (str, Path)) or not str(value) or len(str(value)) > 4096:
        raise ValueError("component path must be a bounded nonempty path")
    path = Path(value)
    if ".." in path.parts or len(path.parts) > 48:
        raise ValueError("component path traversal or excessive depth")
    if not path.is_absolute():
        if base is None:
            raise ValueError("component catalog path must be absolute")
        path = base / path
    if not path.is_relative_to(COMPONENT_PROJECT_ROOT) or path == COMPONENT_PROJECT_ROOT:
        raise ValueError("component paths must remain in the canonical project")
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
