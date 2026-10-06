"""Checked bridge from YAP fabrication birth states to mechanics histories."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import asdict, dataclass
from itertools import product

try:
    from YAP.manufacturing_bow import FabricationBirthBatch, exp12_initial_q6_payload
    from YAP.mechanics_runtime import (NATIVE_ROOT, PORTABLE_RUNTIME, resolve_native_input,
                                      relocate_geometry_paths, configure_runtime_storage, validate_write_path)
except ModuleNotFoundError:  # Direct execution from the YAP source directory.
    from manufacturing_bow import FabricationBirthBatch, exp12_initial_q6_payload
    from mechanics_runtime import (NATIVE_ROOT, PORTABLE_RUNTIME, resolve_native_input,
                                  relocate_geometry_paths, configure_runtime_storage, validate_write_path)


ROOT = Path(__file__).resolve().parent
ANSYS_VALIDATION = NATIVE_ROOT
BACKENDS = {
    "bi10_tsv10_beol20_p20_h50": {
        "profile": ANSYS_VALIDATION / "exp12_bi10_runtime_profile_20260914.py",
        "metric_reference": (
            ANSYS_VALIDATION / "results" / "bi10_yield_smoke_20260911_v1"
            / "METRIC_REFERENCE.json"
        ),
        "bi_pitch_um": 10.0,
        "bi_thickness_um": 1.5,
        "tsv_pitch_um": 10.0,
        "beol_period_um": 20.0,
        "beol_thickness_um": 10.0,
        "retained_substrate_um": 50.0,
        # This legacy profile keeps Si at 50 um, so outer and interior dies do
        # not share one total thickness. It is not an exact member of the new
        # constant-total-die-thickness design domain.
        "total_die_thickness_um": None,
        "maximum_cycles": (2, 3),
        "body_scope": "d2w_die",
    },
}


TARGET_GEOMETRY_DOMAIN = {
    "bi_pitch_um": (10.0, 8.0, 6.0, 4.0, 2.0),
    "beol_thickness_um": (5.0, 10.0),
    "tsv_pitch_um": (20.0, 10.0, 6.0),
    "total_die_thickness_um": (100.0, 80.0, 50.0, 30.0, 20.0),
}


PREVIEW_PACKAGES = {
    "bi6_beol10_tsv6_total20": {
        "path": Path("resource:geometry/packages/BI6_20.json"),
        "geometry": (6.0, 10.0, 6.0, 20.0),
    },
    "bi6_beol10_tsv6_total30": {
        "path": Path("resource:geometry_window/packages/BI6_30.json"),
        "geometry": (6.0, 10.0, 6.0, 30.0),
    },
    "bi6_beol10_tsv6_total50": {
        "path": Path("resource:geometry/packages/BI6_50.json"),
        "geometry": (6.0, 10.0, 6.0, 50.0),
    },
    "bi6_beol10_tsv6_total80": {
        "path": Path("resource:geometry/packages/BI6_80.json"),
        "geometry": (6.0, 10.0, 6.0, 80.0),
    },
    "bi2_beol10_tsv6_total50": {
        "path": Path("resource:geometry/packages/BI2_50.json"),
        "geometry": (2.0, 10.0, 6.0, 50.0),
    },
}


COMPACT_PACKAGES = {
    "bi6_beol10_tsv6_total20": {
        "path": Path("resource:compact_p6/packages/BI6_20.json"),
        "geometry": (6.0, 10.0, 6.0, 20.0),
        "qualification": "UNQUALIFIED_FOR_CORRECTED_TIER_ORDER",
    },
    "bi6_beol10_tsv6_total50": {
        "path": Path("resource:compact_p6/packages/BI6_50.json"),
        "geometry": (6.0, 10.0, 6.0, 50.0),
        "qualification": "FAILED_CORRECTED_TIER_C1_BENCHMARK",
    },
    "bi6_beol10_tsv6_total80": {
        "path": Path("resource:compact_p6/packages/BI6_80.json"),
        "geometry": (6.0, 10.0, 6.0, 80.0),
        "qualification": "UNQUALIFIED_FOR_CORRECTED_TIER_ORDER",
    },
    "bi2_beol10_tsv6_total50": {
        "path": Path("resource:compact_p2/packages/BI2_50.json"),
        "geometry": (2.0, 10.0, 6.0, 50.0),
        "qualification": "UNQUALIFIED_FOR_CORRECTED_TIER_ORDER",
    },
}


@dataclass(frozen=True)
class MechanicsGeometry:
    """Geometry coordinates used to select a history-mechanics operator.

    ``total_die_thickness_um`` includes TSV/Si, BEOL, and every BI layer owned
    by the die. The bottom die owns one front BI layer; every subsequently
    attached die owns back and front BI layers, including the extendable top
    die. Their TSV/Si thicknesses therefore differ at fixed total thickness.
    """

    bi_pitch_um: float
    beol_thickness_um: float
    tsv_pitch_um: float
    total_die_thickness_um: float
    bi_thickness_um: float = 1.5

    def __post_init__(self):
        for key, value in asdict(self).items():
            numeric = float(value)
            if not numeric > 0.0:
                raise ValueError(f"{key} must be positive")
            object.__setattr__(self, key, numeric)

    def silicon_thickness_um(self, *, bi_layer_count=1):
        if isinstance(bi_layer_count, bool) or bi_layer_count not in (1, 2):
            raise ValueError("bi_layer_count must be 1 or 2")
        thickness = (
            self.total_die_thickness_um
            - self.beol_thickness_um
            - bi_layer_count * self.bi_thickness_um
        )
        if thickness <= 0.0:
            raise ValueError("total die thickness leaves no positive TSV/Si thickness")
        return thickness


def _geometry_from_backend(backend):
    if backend.get("total_die_thickness_um") is None:
        return None
    return MechanicsGeometry(
        bi_pitch_um=backend["bi_pitch_um"],
        beol_thickness_um=backend["beol_thickness_um"],
        tsv_pitch_um=backend["tsv_pitch_um"],
        total_die_thickness_um=backend["total_die_thickness_um"],
        bi_thickness_um=backend.get("bi_thickness_um", 1.5),
    )


def _geometry_from_package(record):
    return MechanicsGeometry(*record["geometry"])


def _package_registry(fidelity):
    if fidelity == "native":
        return PREVIEW_PACKAGES
    if fidelity == "compact":
        return COMPACT_PACKAGES
    raise ValueError("fidelity must be 'native' or 'compact'")


def _load_geometry_package(name, *, fidelity, allow_unqualified_compact=False):
    registry = _package_registry(fidelity)
    try:
        record = registry[name]
    except KeyError as exc:
        raise ValueError(f"unknown {fidelity} mechanics package: {name}") from exc
    path = resolve_native_input(ANSYS_VALIDATION, record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    package = relocate_geometry_paths(_load_json(path), path.parent)
    expected_state = {
        "native": ("READY_ENGINEERING_PREVIEW", True),
        "compact": ("CANDIDATE_NOT_VALIDATED", False),
    }[fidelity]
    if package.get("schema") != "exp12.ofat.geometry_package.v1" or (
        package.get("state"), package.get("ready")
    ) != expected_state:
        raise RuntimeError(f"mechanics package is not ready: {path}")
    if fidelity == "compact":
        qualification = record["qualification"]
        if not allow_unqualified_compact:
            raise RuntimeError(
                f"compact mechanics package is diagnostic-only ({qualification}); "
                "pass allow_unqualified_compact=True only for qualification work"
            )
        if name not in PREVIEW_PACKAGES:
            raise RuntimeError(f"compact package has no registered native parent: {path}")
        parent_path = resolve_native_input(ANSYS_VALIDATION, PREVIEW_PACKAGES[name]["path"])
        parent_sha256 = hashlib.sha256(parent_path.read_bytes()).hexdigest()
        if package.get("parent_package_sha256") != parent_sha256:
            raise RuntimeError(f"compact package is not bound to its native parent: {path}")
        if package.get("whole_stack_response_fit") is not False:
            raise RuntimeError(f"compact package unexpectedly fits stack outputs: {path}")
    required = {"bi", "beol"}
    if not required.issubset(package.get("datasets", {})):
        raise RuntimeError(f"mechanics package is incomplete: {path}")
    inputs = [*package["datasets"].values(), package.get("port_basis")]
    missing = [value for value in inputs if not value or not Path(value).is_file()]
    if missing:
        raise FileNotFoundError(f"mechanics package inputs are missing: {missing}")
    return path, package


def resolve_mechanics_package(geometry, *, fidelity="native",
                              allow_unqualified_compact=False):
    """Return an exact native or explicitly requested compact package."""
    if isinstance(geometry, dict):
        geometry = MechanicsGeometry(**geometry)
    if not isinstance(geometry, MechanicsGeometry):
        raise TypeError("geometry must be MechanicsGeometry or a matching dictionary")
    registry = _package_registry(fidelity)
    matches = [
        name for name, record in registry.items()
        if _geometry_from_package(record) == geometry
    ]
    if len(matches) != 1:
        requested = ", ".join(f"{key}={value:g}" for key, value in asdict(geometry).items())
        if matches:
            raise RuntimeError(f"geometry has multiple registered packages: {matches}")
        raise ValueError(
            f"no exact {fidelity} history-mechanics package is registered for " + requested
            + "; interpolation and geometry scaling are disabled"
        )
    path, package = _load_geometry_package(
        matches[0], fidelity=fidelity,
        allow_unqualified_compact=allow_unqualified_compact,
    )
    return {
        "name": matches[0], "path": str(path), "package": package,
        "fidelity": fidelity,
    }


def _import_mechanics_module(name):
    configure_runtime_storage()
    location = str(ANSYS_VALIDATION)
    project = str(ROOT.parent)
    if project not in sys.path:
        sys.path.insert(0, project)
    if location not in sys.path:
        sys.path.insert(0, location)
    def check_origins():
        if not PORTABLE_RUNTIME:
            return
        prefixes = ("exp12_", "run_exp12_", "audit_exp12_", "build_exp12_", "collect_exp12_",
                    "launch_exp12_", "monitor_exp12_", "prepare_exp12_")
        for key, loaded in tuple(sys.modules.items()):
            if key.startswith(prefixes) or key == "stack_bow_conventions":
                filename = getattr(loaded, "__file__", None)
                if filename and not Path(filename).resolve().is_relative_to(ANSYS_VALIDATION):
                    raise ImportError(f"native module outside bundled runtime: {key}: {filename}")
    check_origins()
    module = importlib.import_module(name)
    check_origins()
    if PORTABLE_RUNTIME:
        if not Path(module.__file__).resolve().is_relative_to(ANSYS_VALIDATION):
            raise ImportError(f"native module outside bundled runtime: {name}")
    return module


def _geometry_ledger(geometry, *, die_count):
    if isinstance(die_count, bool) or not isinstance(die_count, int) or not 2 <= die_count <= 16:
        raise ValueError("die_count must be an integer in 2..16")
    case_module = _import_mechanics_module("exp12_ofat_case_study_20260913")
    case = case_module.OFATCase(
        case_id="yap_geometry_runtime",
        label="YAP geometry runtime",
        changed_parameter="explicit_geometry",
        die_count=die_count,
        total_die_um=geometry.total_die_thickness_um,
        bi_pitch_um=geometry.bi_pitch_um,
        tsv_diameter_um=0.5 * geometry.tsv_pitch_um,
        tsv_pitch_um=geometry.tsv_pitch_um,
    )
    tier_module = _import_mechanics_module("exp12_correct_tier_order_20260914")
    ledger = tier_module.d2d_case_ledger(case)
    if not all(
        abs(layer["thickness_um"] - geometry.beol_thickness_um) <= 1e-12
        for layer in ledger["layers"] if layer["kind"] == "beol"
    ):
        raise ValueError("the current generic ledger does not support this BEOL thickness")
    return case, ledger


def _configure_mechanics_runtime_environment(*, layer_workers):
    configure_runtime_storage()
    defaults = {
        "NUMBA_THREADING_LAYER": "omp",
        "NUMBA_NUM_THREADS": str(layer_workers),
        "OMP_NUM_THREADS": str(layer_workers),
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def load_mechanics_geometry_model(geometry, *, die_count, footprint_mm=(10.0, 10.0),
                                  anneal_temperature_c=350, layer_workers=3,
                                  fidelity="native", allow_unqualified_compact=False):
    """Load one exact package with the frozen same-orientation tier ledger."""
    if isinstance(geometry, dict):
        geometry = MechanicsGeometry(**geometry)
    resolved = resolve_mechanics_package(
        geometry, fidelity=fidelity,
        allow_unqualified_compact=allow_unqualified_compact,
    )
    _, ledger = _geometry_ledger(geometry, die_count=die_count)
    if isinstance(layer_workers, bool) or not isinstance(layer_workers, int) or layer_workers < 1:
        raise ValueError("layer_workers must be a positive integer")
    # Numba reads its threading layer at import time. Keep the generic runtime
    # bounded, while preserving an explicitly configured caller environment.
    _configure_mechanics_runtime_environment(layer_workers=layer_workers)
    runtime = _import_mechanics_module("exp12_ofat_runtime_20260913")
    model = runtime.load_model(
        ledger,
        resolved["package"],
        temperature_c=anneal_temperature_c,
        footprint_mm=footprint_mm,
        layer_workers=layer_workers,
    )
    model.yap_geometry = asdict(geometry)
    model.yap_package = {
        key: resolved[key] for key in ("name", "path", "fidelity")
    }
    return model


def predict_mechanics_geometry(model, batch, *, keep_cycle_states=False,
                               progress_callback=None):
    """Evaluate one birth batch using an already-loaded geometry model."""
    if not isinstance(batch, FabricationBirthBatch):
        raise TypeError("batch must be a FabricationBirthBatch")
    expected = int(model._ledger["die_count"])
    if len(batch.source_amplitude) != expected:
        raise ValueError("one persistent birth state is required for every die")
    result = model.predict(
        exp12_initial_q6_payload(batch),
        keep_cycle_states=keep_cycle_states,
        progress_callback=progress_callback,
    )
    result["yap_geometry"] = dict(model.yap_geometry)
    result["yap_package"] = dict(model.yap_package)
    return result


def mechanics_geometry_support():
    """Return the requested domain and currently runnable exact combinations."""
    target_combinations = [
        MechanicsGeometry(*values)
        for values in product(*TARGET_GEOMETRY_DOMAIN.values())
    ]
    registered = {}
    for name, backend in BACKENDS.items():
        geometry = _geometry_from_backend(backend)
        if geometry is not None:
            registered[name] = asdict(geometry)
    tsv_si_supports = {
        (geometry.tsv_pitch_um, geometry.silicon_thickness_um(bi_layer_count=count))
        for geometry in target_combinations
        for count in (1, 2)
    }
    package_ready = {
        name: {
            "geometry": asdict(_geometry_from_package(record)),
            "path": str(record["path"]),
        }
        for name, record in PREVIEW_PACKAGES.items()
    }
    packaged_geometries = {
        _geometry_from_package(record) for record in PREVIEW_PACKAGES.values()
    }
    compact_ready = {
        name: {
            "geometry": asdict(_geometry_from_package(record)),
            "path": str(record["path"]),
            "validation_state": "CANDIDATE_NOT_VALIDATED",
            "qualification": record["qualification"],
        }
        for name, record in COMPACT_PACKAGES.items()
    }
    compact_geometries = {
        _geometry_from_package(record) for record in COMPACT_PACKAGES.values()
    }
    missing = [
        asdict(geometry)
        for geometry in target_combinations
        if geometry not in packaged_geometries
    ]
    return {
        "target_domain": {key: list(values) for key, values in TARGET_GEOMETRY_DOMAIN.items()},
        "target_combination_count": len(target_combinations),
        "minimum_physical_component_count_if_one_compatible_basis": {
            "bi": len(TARGET_GEOMETRY_DOMAIN["bi_pitch_um"]),
            "beol": len(TARGET_GEOMETRY_DOMAIN["beol_thickness_um"]),
            "tsv_si": len(tsv_si_supports),
        },
        "registered_profile_exact_combinations": registered,
        "registered_exact_combinations": package_ready,
        "registered_exact_combination_count": len(packaged_geometries),
        "package_ready_exact_combinations": package_ready,
        "package_ready_exact_combination_count": len(packaged_geometries),
        "compact_candidate_exact_combinations": compact_ready,
        "compact_candidate_exact_combination_count": len(compact_geometries),
        "missing_compact_exact_combination_count": (
            len(target_combinations) - len(compact_geometries)
        ),
        "missing_package_exact_combination_count": (
            len(target_combinations) - len(packaged_geometries)
        ),
        "registered_legacy_backends": {
            name: {
                "retained_substrate_um": backend["retained_substrate_um"],
                "outer_die_total_um": (
                    backend["retained_substrate_um"]
                    + backend["beol_thickness_um"]
                    + backend.get("bi_thickness_um", 1.5)
                ),
                "interior_die_total_um": (
                    backend["retained_substrate_um"]
                    + backend["beol_thickness_um"]
                    + 2 * backend.get("bi_thickness_um", 1.5)
                ),
            }
            for name, backend in BACKENDS.items()
            if _geometry_from_backend(backend) is None
        },
        "missing_exact_combination_count": (
            len(target_combinations) - len(packaged_geometries)
        ),
        "missing_exact_combinations": missing,
        "selection_policy": "exact_registered_package_only",
        "operator_compatibility": (
            "datasets are port-basis-bound; physical component counts are not "
            "runnable counts until one compatible basis is qualified"
        ),
        "unsupported_policy": "fail_closed_no_nearest_neighbor_or_silent_scaling",
    }


def resolve_mechanics_backend(geometry):
    """Resolve one exact geometry to a runnable backend name."""
    if isinstance(geometry, dict):
        geometry = MechanicsGeometry(**geometry)
    if not isinstance(geometry, MechanicsGeometry):
        raise TypeError("geometry must be MechanicsGeometry or a matching dictionary")
    matches = [
        name for name, backend in BACKENDS.items()
        if _geometry_from_backend(backend) is not None
        and _geometry_from_backend(backend) == geometry
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"geometry has multiple registered backends: {matches}")
    requested = ", ".join(f"{key}={value:g}" for key, value in asdict(geometry).items())
    raise ValueError(
        "no exact history-mechanics operator is registered for " + requested
        + "; nearest-neighbor substitution and geometry scaling are disabled"
    )


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def registered_mechanics_backends():
    """Return a detached description of every explicitly registered backend."""
    return json.loads(json.dumps(BACKENDS, default=str))


def _backend(name):
    try:
        backend = BACKENDS[name]
    except KeyError as exc:
        raise ValueError(f"unknown mechanics backend: {name}") from exc
    if not backend["profile"].is_file():
        raise FileNotFoundError(backend["profile"])
    return backend


def write_birth_snapshot(path, batch):
    """Create, but never replace, one provenance-bound full-q6 input file."""
    if not isinstance(batch, FabricationBirthBatch):
        raise TypeError("batch must be a FabricationBirthBatch")
    path = validate_write_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(exp12_initial_q6_payload(batch), indent=2,
                          allow_nan=False) + "\n").encode("ascii")
    with path.open("xb") as stream:
        stream.write(encoded)
    return path, hashlib.sha256(encoded).hexdigest()


def verify_completed_history(output_dir, *, maximum_cycle, birth_snapshot=None):
    """Fail closed unless a complete, internally consistent history is present."""
    output = Path(output_dir).resolve()
    result = _load_json(output / "RESULT.json")
    status = _load_json(output / "RUN_STATUS.json")
    freeze = _load_json(output / "PREDICTION_FREEZE.json")
    formations = _load_json(output / "FORMATIONS.json")
    history = output / "HISTORY.csv"
    history_sha256 = hashlib.sha256(history.read_bytes()).hexdigest()
    if result.get("state") != "COMPLETE_NEW_PROCESS_UNSCORED" \
            or status.get("state") != result["state"]:
        raise RuntimeError("mechanics history did not complete")
    if result.get("history_sha256") != history_sha256:
        raise RuntimeError("mechanics history checksum mismatch")
    expected_checkpoints = [f"C{i}" for i in range(1, maximum_cycle + 1)]
    audits = result.get("checkpoint_audits", [])
    if [entry.get("event") for entry in audits] != expected_checkpoints or any(
            not entry.get("checkpoint_roundtrip_exact")
            or not entry.get("offsets_roundtrip_exact") for entry in audits):
        raise RuntimeError("mechanics checkpoint audit is incomplete")
    expected_formations = [f"A{i}" for i in range(1, maximum_cycle + 1)]
    if [entry.get("event") for entry in formations] != expected_formations:
        raise RuntimeError("nonzero-birth attachment records are incomplete")
    initial = freeze.get("initial_bow", {})
    if initial.get("schema") != "exp12.natural_initial_q6.v2" \
            or initial.get("all_zero") is not False:
        raise RuntimeError("history did not consume a nonzero full-q6 birth state")
    if birth_snapshot is not None:
        source = freeze.get("initial_bow_source", {})
        birth_path = str(Path(birth_snapshot).resolve())
        expected_hash = hashlib.sha256(Path(birth_snapshot).read_bytes()).hexdigest()
        if source != {birth_path: expected_hash}:
            raise RuntimeError("mechanics birth-input provenance mismatch")
    return {
        "state": result["state"],
        "output_dir": str(output),
        "history_path": str(history),
        "history_sha256": history_sha256,
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "formation_events": expected_formations,
        "checkpoint_events": expected_checkpoints,
        "mapdl_validation": result.get("mapdl_validation"),
    }


def load_prebond_events(name, output_dir, *, maximum_cycle, birth_snapshot=None,
                        shear_budget_um=1e-6):
    """Convert a checked solver history into the existing YAP yield bridge."""
    try:
        from YAP.history_bow_prebond import (
            PrebondEvent,
            audited_q6_linear_flattening,
        )
        from YAP.history_bow_yield import PrebondBody
    except ModuleNotFoundError:
        from history_bow_prebond import (
            PrebondEvent,
            audited_q6_linear_flattening,
        )
        from history_bow_yield import PrebondBody

    backend = _backend(name)
    verify_completed_history(
        output_dir, maximum_cycle=maximum_cycle, birth_snapshot=birth_snapshot)
    output = Path(output_dir).resolve()
    freeze = _load_json(output / "PREDICTION_FREEZE.json")
    formations = _load_json(output / "FORMATIONS.json")
    metric = _load_json(backend["metric_reference"])
    pairs = metric.get("coefficients_old_incoming", [])[:maximum_cycle]
    if len(pairs) != maximum_cycle:
        raise RuntimeError("geometric metric does not cover the requested history")
    halfspan_um = tuple(500.0 * float(value) for value in freeze["projection_mm"])
    frame = freeze["initial_bow"]["frame"]
    events, audits = [], []
    for ordinal, (record, pair) in enumerate(zip(formations, pairs, strict=True), 1):
        event_name = f"A{ordinal}"
        if record.get("event") != event_name or len(pair) != 2:
            raise RuntimeError("formation and metric event order differ")
        if record.get("retained_names") != pair[0].get("layer_names") \
                or record.get("incoming_names") != pair[1].get("layer_names"):
            raise RuntimeError("formation body identities differ from the metric")
        q6, shear = audited_q6_linear_flattening(
            record["linear_overlay_flatten_increment_old_minus_incoming_ppm"],
            halfspan_um=halfspan_um,
            max_omitted_overlay_um=shear_budget_um,
        )
        bodies = []
        for side, role in enumerate(("old_free", "incoming_free")):
            detail = record["bodies_old_then_incoming"][side]
            bow = detail["free_paper_bow_um"]
            bodies.append(PrebondBody(
                body_id=f"{event_name}/{role}/body",
                state_id=f"{event_name}/{role}",
                frame_id=frame,
                bow_um=(bow["edge_x_um"], bow["edge_y_um"]),
                halfspan_um=halfspan_um,
                face_lever_um=0.0,
                chi=(pair[side]["chi"], pair[side]["chi"]),
                face_lever_source=(
                    "measured q6 flattening increment replaces the linear lever term"
                ),
            ))
        events.append(PrebondEvent(
            event_name, "pre_attachment", bodies[1], bodies[0], q6))
        audits.append({"event": event_name, "shear": shear})
    return tuple(events), tuple(audits)


def run_mechanics_history(name, output_dir, batch, *, footprint_mm=(10.0, 10.0),
                          maximum_cycle=3):
    """Run one complete registered history from a fabrication birth batch."""
    backend = _backend(name)
    if maximum_cycle not in backend["maximum_cycles"]:
        raise ValueError("cycle count is outside the registered backend scope")
    if batch.body_scope != backend["body_scope"]:
        raise ValueError("fabrication birth scope does not match the backend")
    if len(batch.source_amplitude) != maximum_cycle + 1:
        raise ValueError("one persistent birth state is required for every die")
    configure_runtime_storage()
    output = validate_write_path(output_dir)
    snapshot = validate_write_path(output.parent / f"{output.name}.initial_q6.json")
    write_birth_snapshot(snapshot, batch)
    command = [
        sys.executable, "-B", str(backend["profile"]), "run",
        "--output-dir", str(output),
        "--footprint-mm", *(format(float(value), "g") for value in footprint_mm),
        "--maximum-cycle", str(maximum_cycle),
        "--initial-bow-json", str(snapshot),
    ]
    env = dict(os.environ)
    completed = subprocess.run(command, cwd=ANSYS_VALIDATION, env=env, check=False)
    if completed.returncode:
        raise RuntimeError(f"mechanics backend exited with code {completed.returncode}")
    result = verify_completed_history(
        output, maximum_cycle=maximum_cycle, birth_snapshot=snapshot)
    events, _ = load_prebond_events(
        name, output, maximum_cycle=maximum_cycle, birth_snapshot=snapshot)
    result["prebond_event_count"] = len(events)
    result["birth_snapshot"] = str(snapshot)
    return result


def run_mechanics_geometry(geometry, output_dir, batch, *,
                           footprint_mm=(10.0, 10.0), maximum_cycle=3):
    """Resolve and run an exact registered geometry without name-based routing."""
    name = resolve_mechanics_backend(geometry)
    result = run_mechanics_history(
        name,
        output_dir,
        batch,
        footprint_mm=footprint_mm,
        maximum_cycle=maximum_cycle,
    )
    result["mechanics_backend"] = name
    result["mechanics_geometry"] = asdict(_geometry_from_backend(BACKENDS[name]))
    return result
