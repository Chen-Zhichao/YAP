"""Bounded, prescribed-path thermal screening of immutable p2 candidates.

This development screen does not qualify free bow, Cu fields, mesh/time
convergence, W2W thinning, or a production library. It never edits admission
flags or retrains an operator from the reference response.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.build_prepared_operators as builder
import tools.operator_grid as grid


CU_SHA = "b3a876bcd8847e5ee2366580155b66ef5a3c01a7971a3b44b152b47eb46db8aa"
NONCU_SHA = "45a77a87f93a962defb08ead20ae3676cd41aede84ae4ee43aed90ac7a52acc4"
TEACHER = "run_exp12_m3_layer_interface_replay.py"
Q6 = ("eps_x", "eps_y", "gamma_xy", "kappa_x", "kappa_y", "kappa_xy")
STAGE_SECONDS = 21600
CASE_SECONDS = 45000
WAIT_SECONDS = 86400
SCOPE = "prescribed_synthetic_V2_clock_checked_not_self_consistent_anchor"
FOLLOW_WAIT_SECONDS = 7 * 86400
EXIT_GRACE_SECONDS = 180
BUILD_PREPARATION_SECONDS = 1800
SUCCESS_DRAIN_SECONDS = 60.


class ProcessInventory(dict):
    """Live identities with retained scope evidence after roots are reaped."""

    def __init__(self):
        super().__init__()
        self.root_identity = None
        self.groups, self.sessions = set(), set()
        self.scope_work = self.scope_identity = None
        self.scope_launch_sha = None
        self.process = None

    def __setitem__(self, key, identity):
        if self.root_identity is None:
            self.root_identity = dict(identity)
        self.groups.update([identity["pgid"]] if "pgid" in identity else [])
        self.sessions.update([identity["sid"]] if "sid" in identity else [])
        if max(len(self.groups), len(self.sessions)) > 4096:
            raise RuntimeError("retained process scope exceeds its finite bound")
        super().__setitem__(key, identity)


def scoped_members(observed):
    if not isinstance(observed, ProcessInventory):
        return []
    return [row for row in grid.own_process_census() if not row["state"].startswith("Z")
            and (row["pgid"] in observed.groups or row["sid"] in observed.sessions)]


def import_worker_scope(observed):
    if not isinstance(observed, ProcessInventory) or observed.scope_work is None:
        return
    path = observed.scope_work / "PROCESS_SCOPE.json"
    if not path.exists():
        if observed.process is not None and observed.process.poll() is not None:
            raise RuntimeError("exited worker lacks launch-bound lifecycle evidence")
        return
    record, _ = grid.read_record(path)
    if (record.get("worker_identity") != observed.scope_identity
            or observed.scope_launch_sha is not None and record.get("launch_sha256") != observed.scope_launch_sha):
        raise ValueError("worker lifecycle evidence does not match its launch")
    for key, destination in (("groups", observed.groups), ("sessions", observed.sessions)):
        values = record.get(key)
        if not isinstance(values, list) or len(values) > 4096 or any(type(v) is not int or v <= 0 for v in values):
            raise ValueError("invalid bounded worker process scope")
        destination.update(values)
        if len(destination) > 4096:
            raise RuntimeError("worker process scope exceeds its finite bound")


def publish_worker_scope(observed):
    work = os.environ.get("YAP_LIFECYCLE_WORK")
    if not work or not isinstance(observed, ProcessInventory):
        return
    work = builder.local_scratch(work)
    launch = work / "LAUNCH.json"
    if not launch.exists():
        return
    record, launch_sha = grid.read_record(launch)
    identity = grid.process_identity(os.getpid())
    if identity != record.get("worker_identity"):
        return
    path = work / "PROCESS_SCOPE.json"
    groups, sessions = set(observed.groups), set(observed.sessions)
    if path.exists():
        old, _ = grid.read_record(path)
        if old.get("worker_identity") != identity or old.get("launch_sha256") != launch_sha:
            raise ValueError("lifecycle evidence launch changed")
        groups.update(old["groups"])
        sessions.update(old["sessions"])
    if max(len(groups), len(sessions)) > 4096:
        raise RuntimeError("worker lifecycle scope exceeds its finite bound")
    builder.write_json(path, dict(schema="yap.operator_grid.process_scope.v1", worker_identity=identity,
        launch_sha256=launch_sha, groups=sorted(groups), sessions=sorted(sessions),
        live_identities=live_owned(observed), updated_epoch=time.time()))


def signal_identity(identity, signum):
    import ctypes
    import errno
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "pidfd_open") or not hasattr(libc, "pidfd_send_signal"):
        raise RuntimeError("pidfd unavailable; descendant signalling requires locked quarantine")
    libc.pidfd_open.argtypes, libc.pidfd_open.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
    libc.pidfd_send_signal.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    libc.pidfd_send_signal.restype = ctypes.c_int
    descriptor = libc.pidfd_open(identity["pid"], 0)
    if descriptor < 0:
        error = ctypes.get_errno()
        if error == errno.ESRCH:
            return
        raise OSError(error, "cannot pin owned process for signalling")
    try:
        current = grid.process_identity(identity["pid"])
        if current is None or current["start_ticks"] != identity["start_ticks"]:
            return
        if libc.pidfd_send_signal(descriptor, signum, None, 0) < 0:
            error = ctypes.get_errno()
            if error != errno.ESRCH:
                raise OSError(error, "cannot signal pinned owned process")
    finally:
        os.close(descriptor)


def scope():
    return dict(scope=SCOPE, development_screen_only=True,
                thermal_geometry_validated=False, runtime_admitted=False,
                pending=["independent_free_equilibrium_bow", "Cu_field_closure",
                         "full_displacement_field", "temporal_convergence",
                         "spatial_convergence", "assembly_history_and_thinning"])


def scoring_contract():
    return dict(cycles=3, states=1081, peak_temperature_c=350., prebond_s=300.,
        ramp_steps=160, dwell_steps=40, cool_steps=160, clock="continuous_global_time",
        path="thermal_Si_strain_plus_six_distinct_integer_sine_frequencies",
        membrane_amplitudes=[.0015, .0015, .003],
        curvature_amplitude_times_thickness=[.003, .003, .006],
        prescribed_port_coordinates="zero", resultant_reference="layer_local_center",
        relative_error_limit=.10, N_deadband_N_per_mm=1e-3, M_deadband_N=1e-5,
        port_l2_deadband_N=1e-4, bubble_residual_limit=1e-8, trace_tolerance_mm=1e-10)


def closure(source):
    source = builder.safe_path(source)
    common = source.parent / "common/ansys_utils.py"
    if builder.digest(grid.regular(common)) != grid.COMMON_HELPER_SHA:
        raise ValueError("unreviewed solver cleanup helper")
    seeds = [Path(__file__), Path(grid.__file__), Path(builder.__file__),
             Path(__file__).with_name("__init__.py"), common,
             source / TEACHER, source / "run_exp12_m4_all_gauss_offpath_layer_gate.py",
             source / "exp12_all_gauss_exact_acceleration.py"]
    initializer = common.parent / "__init__.py"
    if initializer.is_file():
        seeds.append(initializer)
    pending, files, started = list(seeds), set(seeds), time.monotonic()
    while pending:
        if len(files) > 96 or time.monotonic() - started > 30:
            raise ValueError("source closure exceeds bounded file/time budget")
        path = grid.regular(pending.pop(), max_bytes=2 * 1024**2)
        modules = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules.update(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                modules.add(node.module)
        for module in modules:
            roots = [source.parent] if module.startswith("common.") else [source,
                source.parent / "09_effective_material_die_validation",
                source.parent / "11_symmetric_mesh_beol_rve_verify"]
            if module.startswith("tools."):
                roots = [Path(__file__).resolve().parents[1], *roots]
            matches = [root / (module.replace(".", "/") + ".py") for root in roots]
            matches = [p for p in matches if p.is_file()]
            if len(matches) > 1:
                raise ValueError(f"ambiguous scientific dependency: {module}")
            if module.startswith("tools.") and not matches:
                raise ValueError(f"missing offline tool dependency: {module}")
            if matches and matches[0] not in files:
                files.add(matches[0])
                pending.append(matches[0])
    return dict(schema="yap.operator_grid.sources.v1", common_utils=str(common),
                files_sha256={str(grid.regular(p)): builder.digest(p) for p in sorted(files)})


def candidate(root):
    root = builder.safe_path(root)
    receipt, receipt_sha = grid.read_record(root / "ARCHIVE.json")
    if receipt.get("schema") != "yap.operator_grid.layer_archive.v1":
        raise ValueError("expected a completed layer archive")
    files = receipt.get("files_sha256", {})
    audits = [name for name in files if name == "audit/LAYER.json"]
    datasets = [name for name in files if name.startswith("datasets/") and name.endswith(".npz")]
    lifts = [name for name in files if name.endswith("/M4_MINDLIN_Q6_STATIC_LIFT_FIELDS.npz")]
    if len(audits) != 1 or len(datasets) != 1 or len(lifts) != 1:
        raise ValueError("missing unique archived audit/dataset/q6 lifts")
    selected = ["ARCHIVE.json", audits[0], datasets[0], lifts[0], "shared/port_basis.npz"]
    hashes = {name: receipt_sha if name == "ARCHIVE.json" else files[name] for name in selected}
    for name, expected in hashes.items():
        if builder.digest(grid.regular(root / name)) != expected:
            raise ValueError(f"candidate archive integrity mismatch: {name}")
    audit, _ = grid.read_record(root / audits[0], files[audits[0]])
    if (audit.get("state") != "PASS_STRUCTURAL_CANDIDATE"
            or audit.get("bonding_pitch_scope_um") != 2.
            or audit.get("all_cu_points_retained") is not True
            or audit.get("port_rank_per_face") != 64
            or audit.get("dataset_sha256") != files[datasets[0]]
            or audit.get("port_basis_sha256") != builder.BASIS_SHA):
        raise ValueError("unsupported structural candidate contract")
    h = audit["thickness_um"]
    grid.si_key(h)
    return dict(root=str(root), key=grid.si_key(h), si_thickness_um=h,
                dataset=datasets[0], lifts=lifts[0], basis="shared/port_basis.npz",
                files_sha256=hashes, cu_card=audit["material_contract"]["cu_card"])


def verify_candidate(record):
    for name, expected in record["files_sha256"].items():
        if builder.digest(grid.regular(Path(record["root"]) / name)) != expected:
            raise ValueError(f"candidate drift: {record['key']}/{name}")


def archived_candidate(root, thickness):
    record = candidate(root)
    archive, _ = grid.read_record(Path(record["root"]) / "ARCHIVE.json")
    inventory = archive.get("files_sha256")
    if not isinstance(inventory, dict) or not 1 <= len(inventory) <= 128:
        raise ValueError("bounded archive inventory required")
    for name, digest in inventory.items():
        if not isinstance(name, str):
            raise ValueError("invalid root-relative archive inventory")
        path = Path(name)
        if (path.is_absolute() or str(path) != name or not path.parts
                or {"..", "#snapshot", ".snapshot"}.intersection(path.parts)
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid root-relative archive inventory")
    audit, _ = grid.read_record(Path(record["root"]) / "audit/LAYER.json",
                              record["files_sha256"]["audit/LAYER.json"])
    if (audit.get("schema") != "yap.operator_grid.layer_candidate.v1"
            or audit.get("layer") != "tsv_si" or audit.get("common_cell_side_um") != 6.
            or audit.get("tsv_pitch_um") != 6. or audit.get("tsv_diameter_um") != 3.
            or record["si_thickness_um"] != thickness or audit.get("library_key") != record["key"]
            or any(archive.get(k) is not False for k in ("runtime_admitted", "thermal_geometry_validated"))
            or any(audit.get(k) is not False for k in (
                "thermal_geometry_validated", "d2w_runtime_enabled", "w2w_runtime_enabled"))):
        raise ValueError("archived candidate geometry/admission contract differs")
    return record


def archive_queue_resources(work, count, *, group_binding=None):
    resource = builder.resource_check(work)
    # Retain all case artifacts; do not rely on automatic deletion between cases.
    required = 200 + grid.RESERVATION_GIB * count
    if not 1 <= count <= 18 or resource["local_free_gib"] < required:
        raise RuntimeError(f"archive queue needs {required} GiB local free space")
    if (grid.foreign_ansys_blockers(group_binding) if group_binding else resource["own_ansys_processes"]):
        raise RuntimeError("another own-UID ANSYS process is active")
    return resource


def archive_plan_command(args):
    host = builder.physical_host()
    work = builder.local_scratch(args.work, host=host)
    archive = builder.safe_path(args.archive)
    if (work.exists() or archive.exists() or archive.parent != grid.ARCHIVES
            or not archive.name.startswith("operator_grid_p2_thermal_")):
        raise ValueError("fresh local work and canonical archive required")
    if not 1 <= len(args.candidate) <= 18:
        raise ValueError("bounded candidate queue required")
    records = [archived_candidate(Path(root), float(h)) for root, h in args.candidate]
    if len({r["key"] for r in records}) != len(records):
        raise ValueError("duplicate archived candidate")
    source = builder.safe_path(args.source)
    pins = closure(source)
    cards = {}
    for name, filename, expected in (("cu", "CU_MATERIAL_CARD_V2_KIM2026.json", CU_SHA),
            ("noncu", "NONCU_MATERIAL_CARD_V2_KIM2026.json", NONCU_SHA),
            ("process", "PROCESS_CARD_V2_KIM2026.json", None)):
        path = grid.regular(source / "results/material_card_v2_kim2026" / filename)
        value, actual = grid.read_record(path, expected)
        if name == "process" and value.get("schema_version") != "yap.thermal_process_card.v2":
            raise ValueError("expected frozen V2 process")
        cards[name] = dict(path=str(path), sha256=actual)
    group_binding = getattr(args, "group_binding", None)
    resources = archive_queue_resources(work.parent, len(records), group_binding=group_binding)
    work.mkdir()
    archive.mkdir()
    builder.write_json(work / "SOURCE_PINS.json", pins, immutable=True)
    plan = dict(schema="yap.operator_grid.thermal_archive_screen.v1", **scope(),
        work=str(work), archive=str(archive), source=str(source), execution_host=host,
        cards=cards, candidates=records, scoring_contract=scoring_contract(),
        pins_sha256=builder.digest(work / "SOURCE_PINS.json"), automatic_retry=False,
        cores=8, stage_seconds=STAGE_SECONDS, case_seconds=CASE_SECONDS,
        wait_seconds=WAIT_SECONDS, created_epoch=time.time(),
        expires_epoch=time.time() + WAIT_SECONDS + len(records) * CASE_SECONDS,
        predecessor=None, group_binding=group_binding, continue_scientific_failures=True,
        resources=resources, provenance="hash_pinned_consumed_structural_archive_artifacts")
    builder.write_json(work / "PLAN.json", plan, immutable=True)
    for name in ("PLAN.json", "SOURCE_PINS.json"):
        builder.copy_checked(work / name, archive / name)
    print(dict(plan_sha256=builder.digest(work / "PLAN.json"), resources=resources), flush=True)


def scientific_screen_result(plan, record, receipt):
    expected_gates = set(Q6) | {"bottom_ports", "top_ports", "bubble_equilibrium", "inelastic_excitation"}
    if (receipt.get("state") not in ("PASS_PRESCRIBED_SCREEN", "FAIL_PRESCRIBED_SCREEN")
            or receipt.get("key") != record["key"] or receipt.get("states") != 1081
            or receipt.get("candidate_sha256") != record["files_sha256"]
            or any(receipt.get(k) != v for k, v in scope().items())
            or any(receipt.get(k) is not False for k in ("runtime_admitted", "thermal_geometry_validated"))
            or receipt.get("development_screen_only") is not True
            or not isinstance(receipt.get("gates"), dict) or set(receipt["gates"]) != expected_gates
            or any(not isinstance(g, dict) or type(g.get("passed")) is not bool for g in receipt["gates"].values())
            or (receipt["state"] == "PASS_PRESCRIBED_SCREEN") != all(
                g["passed"] for g in receipt["gates"].values())):
        raise ValueError("malformed scientific screen receipt")
    return receipt["state"] == "PASS_PRESCRIBED_SCREEN" or (
        plan.get("schema") == "yap.operator_grid.thermal_archive_screen.v1"
        and plan.get("continue_scientific_failures") is True)


def completed_screen_state(completed):
    if any(r["state"] == "FAIL_PRESCRIBED_SCREEN" for r in completed):
        return "BATCH_SCREEN_COMPLETE_WITH_FAILURES_NOT_QUALIFIED"
    return "BATCH_SCREEN_COMPLETE_NOT_QUALIFIED"


def history(process, h_mm, cycles=3):
    """Six nonproportional, reversing coordinates; continuous material age."""
    import numpy as np
    if (not math.isfinite(h_mm) or h_mm <= 0. or process.prebond_s != 300. or process.room_c != 25.
            or process.anneal_c != 350. or cycles != 3):
        raise ValueError("unreviewed thermal screening schedule")
    times, temps, creep = [process.prebond_s], [process.room_c], [process.creep_prebond]
    fractions = [0.]
    for cycle in range(cycles):
        start = process.prebond_s + cycle * (process.ramp_s + process.dwell_s + process.cool_s)
        offset = 0.
        for steps, duration, a, b, active in (
            (160, process.ramp_s, process.room_c, process.anneal_c, process.creep_ramp),
            (40, process.dwell_s, process.anneal_c, process.anneal_c, process.creep_dwell),
            (160, process.cool_s, process.anneal_c, process.room_c, process.creep_cool),
        ):
            for step in range(1, steps + 1):
                elapsed = offset + duration * step / steps
                times.append(start + elapsed)
                temps.append(a + (b - a) * step / steps)
                creep.append(active)
                fractions.append(cycle + elapsed / (process.ramp_s + process.dwell_s + process.cool_s))
            offset += duration
    times, temps = np.asarray(times), np.asarray(temps)
    q = np.zeros((len(times), 134))
    amplitude = np.asarray([.0015, .0015, .003, .003 / h_mm, .003 / h_mm, .006 / h_mm])
    q[:, :6] = np.sin(2 * np.pi * np.asarray(fractions)[:, None] * np.arange(1, 7)) * amplitude
    q[:, :2] += 2.6e-6 * (temps[:, None] - process.room_c)
    q[0] = 0.
    q[360::360] = 0.
    return q, times, temps, np.asarray(creep, dtype=bool), amplitude


def macro_rows(q, times, h_mm):
    center = .0115 + .5 * h_mm
    return [dict(result_set=i + 1, time_s=float(t),
                 eps_x0=float(v[0] - center * v[3]),
                 eps_z0=float(v[1] - center * v[4]),
                 gamma_xz0=float(v[2] - center * v[5]),
                 kappa_x_1_per_mm=float(v[3]), kappa_z_1_per_mm=float(v[4]),
                 kappa_xz_1_per_mm=float(v[5])) for i, (t, v) in enumerate(zip(times, q))]


def component_gate(prediction, reference, deadband, relative=.10):
    import numpy as np
    p, r = np.asarray(prediction), np.asarray(reference)
    if p.shape != r.shape or not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError("nonfinite or mismatched response")
    error = np.abs(p - r)
    limits = np.maximum(deadband, relative * np.abs(r))
    return dict(passed=bool(np.all(error <= limits)), maximum_absolute_error=float(np.max(error)),
                absolute_deadband=deadband, relative_limit_percent=100 * relative,
                maximum_normalized_error=float(np.max(error / limits)))


def import_api(plan):
    authenticated_common(plan)
    sys.path[:0] = [plan["source"], str(Path(plan["source"]).parent)]
    import run_exp12_m4_all_gauss_offpath_layer_gate as api
    from exp12_material_card_v2_kim2026 import load_v2_process_card
    return api, load_v2_process_card(Path(plan["cards"]["process"]["path"]))


def authenticated_common(plan):
    pins = grid.verify_pins(Path(plan["work"]) / "SOURCE_PINS.json", plan["pins_sha256"])
    if "common.ansys_utils" not in sys.modules:
        return grid.authenticated_common(pins)
    module = sys.modules["common.ansys_utils"]
    if (str(grid.regular(module.__file__)) != pins["common_utils"]
            or pins["files_sha256"][pins["common_utils"]] != grid.COMMON_HELPER_SHA):
        raise ValueError("common solver helper namespace drift")
    return module


def validate_materials(plan, record, dataset):
    from exp12_cu_material_point import CuCard
    from exp12_material_card_v2_kim2026 import load_v2_cu_card, load_v2_noncu_card
    cu = load_v2_cu_card(Path(plan["cards"]["cu"]["path"]))
    noncu = load_v2_noncu_card(Path(plan["cards"]["noncu"]["path"]))
    geometry = dataset.geometry_signature
    h = record["si_thickness_um"] * .001
    size = geometry.get("cell_size_mm")
    if (dataset.layer_kind != "tsv_si" or not isinstance(size, (list, tuple)) or len(size) != 3
            or not all(math.isclose(a, b, abs_tol=1e-12, rel_tol=0.) for a, b in zip(size, (.006, h, .006)))
            or not math.isclose(geometry.get("thickness_mm", math.nan), h, abs_tol=1e-12, rel_tol=0.)
            or not math.isclose(geometry.get("cell_area_mm2", math.nan), .006**2, abs_tol=1e-16, rel_tol=0.)
            or geometry.get("inclusion_shape") != "circle"
            or geometry.get("element_formulation") != "SOLID185_KEYOPT2_0_Image_Bbar_2x2x2"
            or geometry.get("periodic_boundary_condition") != "periodic_x_z_with_explicit_bottom_top_ports"):
        raise ValueError("dataset geometry differs from the fixed teacher mesh")
    grid.material_contract(dataset)
    if asdict(dataset.cu_card) != asdict(CuCard(**cu.python_cu_card_kwargs())):
        raise ValueError("candidate Cu constitutive law differs from teacher")
    si = asdict(dataset.reference_constant_materials["si"])
    expected_si = dict(elastic_modulus_mpa=noncu.si_e_mpa, poisson_ratio=noncu.si_nu,
                       secant_cte_per_c=noncu.si_alpha_per_c)
    if si != expected_si or asdict(dataset.cu_card) != record["cu_card"]:
        raise ValueError("candidate archived/non-Cu material contract differs")
    lineage = (dataset.metadata or {}).get("tangent_cards_and_process", {})
    for key in ("cu", "noncu", "process"):
        if lineage.get(key + "_sha256") != plan["cards"][key]["sha256"]:
            raise ValueError("candidate material/process lineage differs from teacher")


def prepare_case(plan, record, directory):
    import numpy as np
    from build_exp12_m4_static_lift_fields import expected_face_lifts, trace_error_summary
    from run_exp12_m3_layer_interface_replay import v2_card_driven_thermal_path
    api, process = import_api(plan)
    verify_candidate(record)
    directory.mkdir(parents=True, exist_ok=False)
    for name in (record["dataset"], record["basis"], record["lifts"]):
        builder.copy_checked(Path(record["root"]) / name, directory / Path(name).name,
                             record["files_sha256"][name])
    dataset = api.LayerROMDataset.load(directory / Path(record["dataset"]).name)
    validate_materials(plan, record, dataset)
    del dataset
    with np.load(directory / Path(record["lifts"]).name, allow_pickle=False) as lifts:
        xyz, labels, fields = lifts["coordinates_mm"], lifts["input_labels"], lifts["lift_fields"]
        saved_faces = (lifts["bottom_node_indices"], lifts["top_node_indices"])
    with np.load(directory / "port_basis.npz", allow_pickle=False) as basis:
        coords, modes = basis["coords_mm"], basis["modes"]
    faces, expected = expected_face_lifts(xyz, coords, modes, labels, 0.)
    audit = trace_error_summary(fields, faces, expected)
    h = float(record["si_thickness_um"]) * .001
    if (not all(np.isfinite(a).all() for a in (xyz, fields, coords, modes))
            or tuple(map(str, labels[:6])) != Q6 or fields.shape != (46 * 21 * 46, 3, 134)
            or coords.shape != (2116, 2) or modes.shape != (6348, 64)
            or not np.isclose(np.ptp(xyz[:, 1]), h, atol=1e-12, rtol=0)
            or not np.allclose(np.ptp(xyz[:, (0, 2)], axis=0), .006, atol=1e-12, rtol=0)
            or audit["maximum_absolute_trace_error"] > 1e-10
            or not all(np.array_equal(a, b) for a, b in zip(faces, saved_faces))):
        raise ValueError("candidate centered q6 face trace contract failed")
    q, times, temps, creep, amplitude = history(process, h)
    rows = macro_rows(q, times, h)
    _, thermal = v2_card_driven_thermal_path(rows, process)
    if not np.allclose([s.temperature_c for s in thermal], temps, rtol=0, atol=1e-9):
        raise ValueError("teacher/ROM temperature path differs")
    if not np.array_equal([s.creep_on for s in thermal], creep):
        raise ValueError("teacher/ROM creep activation differs")
    api.write_csv(directory / "macro_history_stack_rve_macro_substeps.csv", rows)
    face_values = {}
    for face, indices in zip(("bottom", "top"), faces):
        order = np.lexsort((xyz[indices, 2], xyz[indices, 0]))
        selected = indices[order]
        if not np.allclose(xyz[selected][:, (0, 2)], coords, atol=1e-12, rtol=0):
            raise ValueError("face node order differs from replay mesh")
        face_values[face + "_values"] = np.einsum("ndq,tq->tnd", fields[selected], q)
    np.savez_compressed(directory / "FACE_HISTORY.npz", coords_mm=coords, times_s=times, **face_values)
    np.savez_compressed(directory / "PATH.npz", external_coordinates=q, times_s=times,
                        temperatures_c=temps, creep_active=creep)
    singular = np.linalg.svd(q[:, :6] / amplitude, compute_uv=False)
    if singular[-1] <= 1e-6 * singular[0]:
        raise ValueError("synthetic six-coordinate path is rank deficient")
    builder.write_json(directory / "INPUT_CONTRACT.json", dict(
        **scope(), candidate=record, path_rank=6, normalized_path_singular_values=singular.tolist(),
        macro_reference="layer_local_center", teacher_reference="legacy_global_y0",
        initial_state="virgin_zero_coordinates_RT300s_creep_off", cycles=3, states=1081,
        trace_audit=audit, input_sha256={name: builder.digest(directory / name) for name in (
            "FACE_HISTORY.npz", "PATH.npz", "macro_history_stack_rve_macro_substeps.csv")}), immutable=True)


def predict(plan, record, directory):
    import numpy as np
    api, _ = import_api(plan)
    from exp12_all_gauss_exact_acceleration import install_exact_acceleration, enable_numba_constitutive_acceleration
    install_exact_acceleration()
    enable_numba_constitutive_acceleration()
    dataset = api.LayerROMDataset.load(directory / Path(record["dataset"]).name)
    validate_materials(plan, record, dataset)
    operator = api.AllGaussReducedFE2Layer.from_dataset(dataset)
    with np.load(directory / "PATH.npz", allow_pickle=False) as data:
        q, times, temps, creep = (data[name] for name in (
            "external_coordinates", "times_s", "temperatures_c", "creep_active"))
    state = operator.initial_state()
    responses, rows, snapshots = [], [], {}
    for index, (v, elapsed, temperature, active) in enumerate(zip(q, times, temps, creep)):
        trial = api.trial_operator_step(operator, v, float(temperature),
            0. if index == 0 else float(elapsed - times[index - 1]), state,
            creep_active=bool(active), elapsed_time_s=float(elapsed))
        state = operator.commit(trial)
        responses.append(trial.generalized_response.copy())
        rows.append(dict(result_set=index + 1, time_s=float(elapsed), iterations=trial.iterations,
                         scaled_residual_norm=float(trial.scaled_residual_norm),
                         plastic_dissipation_Nmm=float(trial.plastic_dissipation_nmm),
                         creep_dissipation_Nmm=float(trial.creep_dissipation_nmm)))
        if index in (0, 160, 200, 360, 520, 560, 720, 880, 920, 1080):
            for key, value in vars(state.cu_state).items():
                if isinstance(value, np.ndarray):
                    snapshots[f"set_{index + 1}_{key}"] = value.copy()
        if index % 20 == 0:
            api.write_csv(directory / "ROM_PROGRESS.csv", rows)
            print(f"{record['key']} ROM state {index + 1}/1081", flush=True)
    api.write_csv(directory / "ROM_PROGRESS.csv", rows)
    np.savez_compressed(directory / "ROM_RESPONSE.npz", response=np.asarray(responses), time_s=times)
    np.savez_compressed(directory / "ROM_CU_STATES.npz", **snapshots)
    builder.write_json(directory / "ROM_COMPLETE.json", dict(**scope(), states=len(rows),
        maximum_bubble_residual=max(r["scaled_residual_norm"] for r in rows),
        total_plastic_dissipation_Nmm=sum(r["plastic_dissipation_Nmm"] for r in rows),
        total_creep_dissipation_Nmm=sum(r["creep_dissipation_Nmm"] for r in rows),
        response_sha256=builder.digest(directory / "ROM_RESPONSE.npz"),
        cu_states_sha256=builder.digest(directory / "ROM_CU_STATES.npz")), immutable=True)


def teacher_command(plan, record, directory, *, plan_only=False):
    values = [sys.executable, "-B", str(Path(plan["source"]) / TEACHER),
        "--layer", "tsv_si", "--variant", "exact", "--substrate-um", str(record["si_thickness_um"]),
        "--ny-substrate", "20", "--nx", "45", "--nz", "45",
        "--coupon-x-um", "6", "--coupon-z-um", "6", "--bi-pitch-um", "2",
        "--bi-pad-side-ratio", ".5", "--inclusion-shape", "circle",
        "--beol-mesh-pitch-um", "6", "--beol-cu-fraction", ".35",
        "--tsv-pitch-um", "6", "--tsv-side-um", "3",
        "--macro-result-dir", str(directory), "--synthetic-port-history-npz", str(directory / "FACE_HISTORY.npz"),
        "--enable-inplane-shear-twist", "--cycles", "3", "--max-result-set", "1081",
        "--output-ramp-stride", "1", "--output-hold-stride", "1", "--output-cool-stride", "1",
        "--export-face-reactions", "--output-dir", str(directory / ("preflight" if plan_only else "teacher")),
        "--run-root", str(directory / ("preflight_solver" if plan_only else "solver")),
        "--ansys-mode", "smp", "--ansys-np", "8", "--ansys-mem-mb", "16000",
        "--ansys-db-mb", "5000", "--ansys-timeout-sec", str(STAGE_SECONDS - 300)]
    for key, flag in (("cu", "--cu-material-card"), ("noncu", "--noncu-material-card"), ("process", "--process-card")):
        values.extend([flag, plan["cards"][key]["path"]])
    if plan_only:
        values.append("--plan-only")
    return values


def score(plan, record, directory, *, seal_sha256):
    import csv
    import numpy as np
    api, _ = import_api(plan)
    verify_inputs(record, directory)
    sealed = verify_rom_seal(directory, seal_sha256)
    rom, _ = grid.read_record(directory / "ROM_COMPLETE.json", sealed["ROM_COMPLETE.json"])
    if (sealed["ROM_RESPONSE.npz"] != rom["response_sha256"]
            or sealed["ROM_CU_STATES.npz"] != rom["cu_states_sha256"]):
        raise ValueError("sealed ROM prediction changed before teacher scoring")
    h = record["si_thickness_um"] * .001
    with np.load(directory / "port_basis.npz", allow_pickle=False) as b:
        modes = b["modes"]
    with np.load(directory / "ROM_RESPONSE.npz", allow_pickle=False) as data:
        predictions, times = data["response"], data["time_s"]
    targets_path = directory / "teacher/M3_LAYER_INTERFACE_REPLAY_RESULTANTS.csv"
    reactions_path = directory / "teacher/M3_LAYER_INTERFACE_FACE_REACTIONS.csv"
    teacher_seal = None
    if plan.get("schema") == "yap.operator_grid.thermal_archive_screen.v1":
        teacher_seal, _ = grid.read_record(directory / "TEACHER_SEAL.json")
        for name, expected in teacher_seal["files_sha256"].items():
            if builder.digest(grid.regular(directory / name)) != expected:
                raise ValueError("sealed teacher response drift")
    with targets_path.open() as stream:
        rows = list(csv.DictReader(stream))
    if ([int(r["result_set"]) for r in rows] != list(range(1, 1082))
            or not np.allclose([float(r["time_s"]) for r in rows], times, atol=1e-8, rtol=0)):
        raise ValueError("reference states/times incomplete or duplicated")
    with reactions_path.open() as stream:
        seen = set()
        expected_nodes = {face: {1 + (i * 21 + j) * 46 + k for i in range(46) for k in range(46)}
                          for face, j in (("bottom", 0), ("top", 20))}
        for r in csv.DictReader(stream):
            key = (int(r["result_set"]), r["face"], int(r["node_id"]))
            if key in seen or not 1 <= key[0] <= 1081 or key[1] not in expected_nodes or key[2] not in expected_nodes[key[1]]:
                raise ValueError("duplicate or unexpected reference face/state/node")
            seen.add(key)
            if not np.isfinite([float(r[k]) for k in ("rf_x_N", "rf_y_N", "rf_z_N")]).all():
                raise ValueError("nonfinite face reaction")
    if len(seen) != 1081 * 2 * 2116:
        raise ValueError("missing exact reference face nodes/states")
    targets = api.mapdl_endpoint_targets(resultants_csv=targets_path, face_reactions_csv=reactions_path,
        modes=modes, layer_thickness_mm=h, cell_area_mm2=.006**2,
        maximum_result_set=1081, macro_labels=Q6)
    if set(targets) != set(range(1, 1082)) or predictions.shape != (1081, 134):
        raise ValueError("reference/prediction state coverage differs")
    reference = np.stack([targets[i] for i in range(1, 1082)])
    gates = {name: component_gate(predictions[:, i] / .006**2,
        reference[:, i] / .006**2, 1e-3 if i < 3 else 1e-5) for i, name in enumerate(Q6)}
    for face, start in (("bottom", 6), ("top", 70)):
        error = np.linalg.norm(predictions[:, start:start + 64] - reference[:, start:start + 64], axis=1)
        norm = np.linalg.norm(reference[:, start:start + 64], axis=1)
        gates[face + "_ports"] = component_gate(error, np.zeros_like(error), 1e-4)
        limits = np.maximum(1e-4, .10 * norm)
        gates[face + "_ports"].update(passed=bool(np.all(error <= limits)),
            maximum_normalized_error=float(np.max(error / limits)), relative_limit_percent=10.)
    gates["bubble_equilibrium"] = dict(passed=rom["maximum_bubble_residual"] <= 1e-8,
        maximum=rom["maximum_bubble_residual"], limit=1e-8)
    # This is an excitation check, not validation of MAPDL internal variables.
    gates["inelastic_excitation"] = dict(passed=rom["total_plastic_dissipation_Nmm"] > 1e-12
        and rom["total_creep_dissipation_Nmm"] > 1e-12,
        plastic_dissipation_Nmm=rom["total_plastic_dissipation_Nmm"],
        creep_dissipation_Nmm=rom["total_creep_dissipation_Nmm"])
    np.savez_compressed(directory / "SCORED_RESPONSE.npz", prediction=predictions, reference=reference, time_s=times)
    resultants_sha, reactions_sha = builder.digest(targets_path), builder.digest(reactions_path)
    if teacher_seal is not None and teacher_seal["files_sha256"] != {
            "teacher/M3_LAYER_INTERFACE_REPLAY_RESULTANTS.csv": resultants_sha,
            "teacher/M3_LAYER_INTERFACE_FACE_REACTIONS.csv": reactions_sha}:
        raise ValueError("teacher response changed during scoring")
    builder.write_json(directory / "THERMAL_SCREEN.json", dict(**scope(),
        state="PASS_PRESCRIBED_SCREEN" if all(g["passed"] for g in gates.values()) else "FAIL_PRESCRIBED_SCREEN",
        key=record["key"], gates=gates, states=1081, candidate_sha256=record["files_sha256"],
        reference_resultants_sha256=resultants_sha,
        reference_face_reactions_sha256=reactions_sha), immutable=True)


def completed_build_binding(p, *, group_binding=None):
    root = Path(p["work"])
    launch, _ = grid.read_record(root / "LAUNCH.json", p["launch_sha256"])
    campaign, _ = grid.read_record(root / "CAMPAIGN.json", p["campaign_sha256"])
    identity = p["identity"]
    captured = None
    if p.get("identity_source") is not None:
        source = p["identity_source"]
        old_plan, _ = grid.read_record(source["path"], source["sha256"])
        captured = old_plan.get("predecessor", {})
        if (old_plan.get("schema") != "yap.operator_grid.thermal_screen.v1"
                or any(captured.get(k) != p[k] for k in ("work", "archive", "launch_sha256", "campaign_sha256", "identity", "si_thicknesses_um"))):
            raise ValueError("historical build identity guard mismatch")
    if (campaign.get("schema") != "yap.operator_grid.campaign.v4"
            or campaign.get("work") != str(root) or campaign.get("archive") != p["archive"]
            or campaign.get("execution_host") != builder.physical_host()
            or campaign.get("si_thicknesses_um") != p["si_thicknesses_um"]
            or launch.get("worker_pid") != identity["pid"]
            or (launch.get("worker_identity") != identity and not (launch.get("worker_identity") is None and captured is not None))
            or launch.get("command") != identity["command"]
            or launch.get("campaign_sha256") != p["campaign_sha256"]
            or launch.get("source_pins_sha256") != campaign.get("source_pins_sha256")):
        raise ValueError("completed build launch/campaign identity mismatch")
    grid.verify_pins(root / "SOURCE_PINS.json", campaign["source_pins_sha256"])
    status, _ = grid.read_record(root / "STATUS.json")
    completed, _ = grid.read_record(Path(p["archive"]) / "BATCH_COMPLETE.json")
    if (status != completed or completed.get("state") != "BATCH_COMPLETE_NOT_FULL_GRID"
            or completed.get("worker_pid") != identity["pid"]
            or [r.get("thickness_um") for r in completed.get("completed", [])] != p["si_thicknesses_um"]):
        raise ValueError("completed build lacks the exact whole-batch receipt")
    for h, record in zip(p["si_thicknesses_um"], completed["completed"]):
        target = Path(p["archive"]) / ("p2_" + grid.si_key(h))
        receipt, _ = grid.read_record(Path(p["archive"]) / f"COMPLETED_p2_{grid.si_key(h)}.json")
        if receipt != record or record.get("archive") != str(target) or record.get("state") != "ARCHIVED_STRUCTURAL_CANDIDATE":
            raise ValueError("completed build layer receipt mismatch")
        inventory, _ = grid.read_record(target / "ARCHIVE.json", record["archive_sha256"])
        audit, _ = grid.read_record(target / "audit/LAYER.json", record["audit_sha256"])
        if (inventory.get("runtime_admitted") is not False
                or inventory.get("files_sha256", {}).get("audit/LAYER.json") != record["audit_sha256"]
                or audit.get("state") != "PASS_STRUCTURAL_CANDIDATE" or audit.get("thickness_um") != h
                or audit.get("dataset_sha256") != record["dataset_sha256"]
                or audit.get("thermal_geometry_validated") is not False):
            raise ValueError("completed build archive/audit mismatch")
    if grid.process_identity(identity["pid"]) is not None or not grid.predecessor_sessions_idle(
            [identity], worker_pid=identity["pid"], **({"group_binding": group_binding} if group_binding else {})):
        raise RuntimeError("completed build worker/session has not exited")


def plan_command(args):
    host = builder.physical_host()
    work = builder.local_scratch(args.work, host=host)
    if work.exists():
        raise ValueError("new validation work root required")
    archive = builder.safe_path(args.archive)
    if archive.parent != grid.ARCHIVES or archive.exists() or not archive.name.startswith("operator_grid_p2_thermal_"):
        raise ValueError("new canonical thermal archive required")
    source = builder.safe_path(args.source)
    records = [candidate(root) for root in args.candidate]
    if not 1 <= len(records) <= 18 or len({r["key"] for r in records}) != len(records):
        raise ValueError("bounded unique candidate list required")
    cards = {}
    for name, filename, expected in (("cu", "CU_MATERIAL_CARD_V2_KIM2026.json", CU_SHA),
            ("noncu", "NONCU_MATERIAL_CARD_V2_KIM2026.json", NONCU_SHA),
            ("process", "PROCESS_CARD_V2_KIM2026.json", None)):
        path = grid.regular(source / "results/material_card_v2_kim2026" / filename)
        value, actual = grid.read_record(path, expected)
        cards[name] = dict(path=str(path), sha256=actual)
        if name == "process" and value.get("schema_version") != "yap.thermal_process_card.v2":
            raise ValueError("expected frozen V2 process")
    predecessor = builder.local_scratch(args.predecessor, host=host)
    launch, launch_sha = grid.read_record(predecessor / "LAUNCH.json")
    campaign, campaign_sha = grid.read_record(predecessor / "CAMPAIGN.json")
    identity = grid.process_identity(args.predecessor_pid)
    if (campaign.get("schema") != "yap.operator_grid.campaign.v4"
            or launch.get("worker_pid") != args.predecessor_pid
            or campaign.get("execution_host") != host):
        raise ValueError("predecessor must be the exact reviewed build queue")
    exited = identity is None
    identity_source = None
    if exited:
        identity = launch.get("worker_identity")
        if identity is None and getattr(args, "predecessor_identity_plan", None) is not None:
            old_plan, old_sha = grid.read_record(args.predecessor_identity_plan, args.predecessor_identity_sha256)
            identity = old_plan.get("predecessor", {}).get("identity")
            identity_source = dict(path=str(grid.regular(args.predecessor_identity_plan)), sha256=old_sha)
        if not isinstance(identity, dict) or identity.get("pid") != args.predecessor_pid:
            raise ValueError("exited predecessor requires a bound launch identity")
    elif identity["command"] != launch.get("command"):
        raise ValueError("live predecessor launch identity mismatch")
    predecessor_record = dict(work=str(predecessor), launch_sha256=launch_sha,
        campaign_sha256=campaign_sha, identity=identity, archive=campaign["archive"],
        si_thicknesses_um=campaign["si_thicknesses_um"])
    if identity_source is not None:
        predecessor_record["identity_source"] = identity_source
    group_binding = getattr(args, "group_binding", None)
    if group_binding is not None:
        group, _, _ = grid.authenticate_group(group_binding)
        if group["jobs"][group_binding["slot"]]["work"] != str(work) or group["jobs"][group_binding["slot"]]["kind"] != "screen":
            raise ValueError("thermal group membership mismatch")
    if exited:
        completed_build_binding(predecessor_record, **({"group_binding": group_binding} if group_binding else {}))
        predecessor_record["completed_at_plan"] = True
    work.mkdir(parents=True)
    archive.mkdir()
    pins = closure(source)
    builder.write_json(work / "SOURCE_PINS.json", pins, immutable=True)
    plan = dict(schema="yap.operator_grid.thermal_screen.v1", **scope(), work=str(work),
        archive=str(archive), source=str(source), execution_host=host, cards=cards, candidates=records,
        scoring_contract=scoring_contract(),
        pins_sha256=builder.digest(work / "SOURCE_PINS.json"), automatic_retry=False,
        cores=8, stage_seconds=STAGE_SECONDS, case_seconds=CASE_SECONDS,
        wait_seconds=WAIT_SECONDS, created_epoch=time.time(),
        expires_epoch=time.time() + WAIT_SECONDS + len(records) * CASE_SECONDS,
        predecessor=predecessor_record, group_binding=group_binding)
    builder.write_json(work / "PLAN.json", plan, immutable=True)
    for name in ("PLAN.json", "SOURCE_PINS.json"):
        builder.copy_checked(work / name, archive / name)
    print(dict(plan_sha256=builder.digest(work / "PLAN.json"), pins_sha256=plan["pins_sha256"]), flush=True)


def checked_plan(args):
    work = builder.local_scratch(args.work)
    plan, _ = grid.read_record(work / "PLAN.json", args.plan_sha256)
    if (plan.get("schema") not in ("yap.operator_grid.thermal_screen.v1", "yap.operator_grid.thermal_archive_screen.v1")
            or plan.get("work") != str(work) or plan.get("execution_host") != builder.physical_host()
            or plan.get("stage_seconds") != STAGE_SECONDS or plan.get("case_seconds") != CASE_SECONDS
            or plan.get("wait_seconds") != WAIT_SECONDS or plan.get("automatic_retry") is not False
            or plan.get("scoring_contract") != scoring_contract()
            or plan.get("thermal_geometry_validated") is not False or plan.get("runtime_admitted") is not False):
        raise ValueError("thermal validation plan policy drift")
    if plan["schema"] == "yap.operator_grid.thermal_archive_screen.v1":
        if (plan.get("predecessor") is not None
                or plan.get("continue_scientific_failures") is not True
                or not 1 <= len(plan["candidates"]) <= 18
                or len({r["key"] for r in plan["candidates"]}) != len(plan["candidates"])):
            raise ValueError("archive-only authority drift")
        if plan.get("group_binding") is not None:
            group, _, _ = grid.authenticate_group(plan["group_binding"])
            slot = plan["group_binding"]["slot"]
            if (group.get("schema") != "yap.operator_grid.parallel_screens.v1"
                    or group["jobs"][slot]["work"] != str(work)):
                raise ValueError("archive pool membership drift")
    grid.verify_pins(work / "SOURCE_PINS.json", plan["pins_sha256"])
    for value in plan["cards"].values():
        if builder.digest(grid.regular(value["path"])) != value["sha256"]:
            raise ValueError("frozen material/process card drift")
    if time.time() >= plan["expires_epoch"]:
        raise TimeoutError("thermal validation queue expired")
    if plan.get("group_binding") is not None:
        grid.authenticate_group(plan["group_binding"])
    return plan, work


def verify_inputs(record, directory):
    contract, _ = grid.read_record(directory / "INPUT_CONTRACT.json")
    if contract.get("candidate") != record or contract.get("scope") != SCOPE:
        raise ValueError("private input contract differs from planned candidate")
    checks = {}
    for name, expected in contract["input_sha256"].items():
        path = grid.regular(directory / name)
        if path in checks and checks[path] != expected:
            raise ValueError("conflicting input SHA256 expectations")
        checks[path] = expected
    for name in (record["dataset"], record["basis"], record["lifts"]):
        path = grid.regular(directory / Path(name).name)
        expected = record["files_sha256"][name]
        if path in checks and checks[path] != expected:
            raise ValueError("conflicting input SHA256 expectations")
        checks[path] = expected
    for path, expected in checks.items():
        if builder.digest(path) != expected:
            raise ValueError("private input drift")


def verify_rom_seal(directory, expected):
    seal, _ = grid.read_record(directory / "ROM_SEAL.json", expected)
    for name, expected in seal["files_sha256"].items():
        if builder.digest(grid.regular(directory / name)) != expected:
            raise ValueError("sealed ROM prediction drift")
    return seal["files_sha256"]


def preflight_all(args):
    plan, work = checked_plan(args)
    builder.configure(work)
    if (work / "STATUS.json").exists():
        raise ValueError("preflight must precede worker launch")
    for index, record in enumerate(plan["candidates"]):
        deadline = min(time.time() + 1800., plan["expires_epoch"])
        directory = work / "cases" / record["key"]
        command = [sys.executable, "-B", str(Path(__file__)), "prepare", "--work", str(work),
            "--plan-sha256", args.plan_sha256, "--index", str(index), "--deadline", str(deadline)]
        run_child(command, work / f"{record['key']}_prepare_prelaunch.log", deadline + 60.)
        command[3] = "preflight"
        run_child(command, work / f"{record['key']}_preflight_prelaunch.log", deadline + 60.)
        builder.write_json(directory / "PREFLIGHT_COMPLETE.json", dict(**scope(),
            plan_sha256=args.plan_sha256,
            input_contract_sha256=builder.digest(directory / "INPUT_CONTRACT.json"),
            teacher_manifest_sha256=builder.digest(directory / "preflight/M3_LAYER_INTERFACE_REPLAY_MANIFEST.json")), immutable=True)
        print(f"Actual candidate trace and solver-deck preflight passed: {record['key']}", flush=True)
    builder.write_json(work / "PREFLIGHT_COMPLETE.json", dict(**scope(), plan_sha256=args.plan_sha256,
        keys=[r["key"] for r in plan["candidates"]], solver_launched=False), immutable=True)
    builder.copy_checked(work / "PREFLIGHT_COMPLETE.json", Path(plan["archive"]) / "PREFLIGHT_COMPLETE.json")


def wait_build_queue(plan, work, state):
    if plan.get("schema") == "yap.operator_grid.thermal_archive_screen.v1":
        if plan.get("precomputed_rom") is True:
            for record in plan["candidates"]:
                directory = work / "cases" / record["key"]
                verify_inputs(record, directory)
                if record["key"] in plan.get("precomputed_rom_keys", []):
                    cached, _ = grid.read_record(directory / "CPU_PRECOMPUTE_COMPLETE.json")
                    verify_rom_seal(directory, cached["rom_seal_sha256"])
            return
        for record in plan["candidates"]:
            if archived_candidate(Path(record["root"]), record["si_thickness_um"]) != record:
                raise ValueError("archive-only input drift")
        return
    p = plan["predecessor"]
    root = Path(p["work"])
    deadline = min(plan["created_epoch"] + WAIT_SECONDS, plan["expires_epoch"])
    while time.time() < deadline:
        grid.read_record(root / "LAUNCH.json", p["launch_sha256"])
        grid.read_record(root / "CAMPAIGN.json", p["campaign_sha256"])
        status, _ = grid.read_record(root / "STATUS.json")
        identity = grid.process_identity(p["identity"]["pid"])
        if identity is not None and identity != p["identity"]:
            raise ValueError("predecessor process identity drift")
        if (root / "STOP.json").exists() or status.get("state", "").startswith(("FAIL", "STOP")):
            raise RuntimeError("predecessor stopped/failed; no automatic takeover")
        if status.get("state") == "BATCH_COMPLETE_NOT_FULL_GRID" and identity is None:
            completed, _ = grid.read_record(Path(p["archive"]) / "BATCH_COMPLETE.json")
            if (completed.get("state") != status["state"]
                    or completed.get("worker_pid") != p["identity"]["pid"]
                    or completed.get("completed") != status.get("completed")
                    or sorted(r["thickness_um"] for r in completed["completed"]) != sorted(p["si_thicknesses_um"])
                    or any(r["state"] != "ARCHIVED_STRUCTURAL_CANDIDATE" for r in completed["completed"])):
                raise ValueError("predecessor completion is not the bound whole batch")
            if any(r["sid"] == p["identity"]["sid"] or r["pgid"] == p["identity"]["pgid"]
                   for r in grid.own_process_census()):
                raise RuntimeError("predecessor left live descendants")
            if p.get("completed_at_plan"):
                completed_build_binding(p, **({"group_binding": plan["group_binding"]} if plan.get("group_binding") else {}))
            if time.time() >= deadline:
                raise TimeoutError("build predecessor proof exceeded wait deadline")
            return
        if identity is None:
            raise RuntimeError("predecessor exited without a completed batch")
        state.update(state="WAITING_BUILD_QUEUE", predecessor_state=status["state"], heartbeat_epoch=time.time())
        builder.write_json(work / "STATUS.json", state)
        print("Waiting for exact predecessor build queue; no solver launched", flush=True)
        time.sleep(60)
    raise TimeoutError("predecessor wait expired")


def live_owned(observed):
    live = []
    for previous in observed.values():
        current = grid.observed_process_identity(previous)
        if current is not None:
            live.append(current)
    return live


def cleanup_owned(observed):
    # Recorded ancestry and start ticks distinguish these experimental
    # processes from unrelated jobs and from reused PIDs.
    for sig, wait in ((signal.SIGTERM, 90.), (signal.SIGKILL, 30.)):
        if isinstance(observed, ProcessInventory) and observed.process is not None:
            refresh_owned(observed.process, observed)
        for identity in live_owned(observed):
            signal_identity(identity, sig)
        deadline = time.time() + wait
        while time.time() < deadline:
            if isinstance(observed, ProcessInventory) and observed.process is not None:
                refresh_owned(observed.process, observed)
            if not live_owned(observed) and not scoped_members(observed):
                return
            time.sleep(.5)
    raise RuntimeError("owned validation descendants did not exit after bounded cleanup")


def pin_spawned(process, observed):
    if isinstance(observed, ProcessInventory):
        observed.process = process
        observed.groups.add(process.pid)
        observed.sessions.add(process.pid)
    identity = grid.process_identity(process.pid)
    if identity is not None:
        observed[(identity["pid"], identity["start_ticks"])] = identity


def refresh_owned(process, observed):
    strict = isinstance(observed, ProcessInventory)
    if strict:
        import_worker_scope(observed)
    root_exited = process.poll() is not None
    for key, previous in list(observed.items()):
        current = grid.observed_process_identity(previous)
        if (current is None or current["start_ticks"] != previous["start_ticks"]
                or root_exited and previous["pid"] == process.pid):
            del observed[key]
        else:
            observed[key] = current
    if not root_exited:
        current = grid.process_identity(process.pid)
        if current is not None:
            root = observed.get((process.pid, current["start_ticks"]))
            if root is None:
                raise ValueError("owned root identity missing or changed before discovery")
    if strict:
        covered = set()
        for anchor in list(observed.values()):
            if (anchor["pid"], anchor["start_ticks"]) in covered:
                continue
            for identity in grid.proven_descendants(anchor):
                covered.add((identity["pid"], identity["start_ticks"]))
                observed[(identity["pid"], identity["start_ticks"])] = identity
    elif not root_exited:
        for identity in grid.descendant_identities(process.pid):
            observed[(identity["pid"], identity["start_ticks"])] = identity
    if len(observed) > 256:
        raise RuntimeError("live owned process inventory exceeds its finite bound")
    if strict:
        publish_worker_scope(observed)


def cleanup_spawned(process, observed):
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT, signal.SIGALRM})
    try:
        try:
            if isinstance(observed, ProcessInventory):
                refresh_owned(process, observed)
            elif process.poll() is None:
                current = grid.process_identity(process.pid)
                pinned = [p for p in observed.values() if p["pid"] == process.pid]
                if current is not None and (not pinned or current in pinned):
                    for identity in grid.descendant_identities(process.pid):
                        observed[(identity["pid"], identity["start_ticks"])] = identity
        except Exception as exc:
            print(f"Cleanup rediscovery unavailable; using owned identities: {exc}", flush=True)
        cleanup_owned(observed)
    finally:
        # Popen owns this child until reaping. This fallback also covers failed
        # identity discovery; it never signals an unrelated process group.
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=90.)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30.)
            else:
                process.wait(timeout=30.)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def success_ready(process, observed, deadline, drain_end, *, group_binding=None):
    def authority():
        if group_binding is not None:
            grid.authenticate_group(group_binding)
        if time.time() >= deadline or time.monotonic() >= drain_end:
            raise TimeoutError("successful child exit drain exceeded its recorded budget")
    authority()
    refresh_owned(process, observed)
    if live_owned(observed) or scoped_members(observed):
        authority()
        return False
    # Recheck after discovery/census; a slow check cannot extend authority.
    authority()
    refresh_owned(process, observed)
    idle = not live_owned(observed) and not scoped_members(observed)
    authority()
    return idle


def drain_success(process, observed, deadline, *, group_binding=None):
    end = time.monotonic() + min(SUCCESS_DRAIN_SECONDS, max(0., deadline - time.time()))
    while not success_ready(process, observed, deadline, end, group_binding=group_binding):
        time.sleep(min(.5, max(0., end - time.monotonic())))


def lifecycle_diagnostic(process, observed, log, error):
    record = dict(event="CHILD_EXIT_REVIEW", root_pid=process.pid, returncode=process.poll(),
                  error=str(error), recorded_identities=list(observed.values()), epoch=time.time())
    try:
        record.update(live_identities=live_owned(observed), scoped_members=scoped_members(observed))
    except Exception as exc:
        record["census_error"] = str(exc)
    print(json.dumps(record, sort_keys=True), flush=True)
    builder.write_json(log.with_name(log.name + ".lifecycle.json"), record)


def try_lifecycle_diagnostic(process, observed, log, error):
    try:
        lifecycle_diagnostic(process, observed, log, error)
    except BaseException:
        pass


def quarantine_spawned(process, observed, log, error):
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK,
        {signal.SIGTERM, signal.SIGINT, signal.SIGALRM, signal.SIGHUP})
    try:
        while True:
            try:
                refresh_owned(process, observed)
                if process.poll() is not None and not live_owned(observed) and not scoped_members(observed):
                    return
            except BaseException as exc:
                try:
                    print(f"Locked lifecycle quarantine: {exc}", flush=True)
                except BaseException:
                    pass
            try_lifecycle_diagnostic(process, observed, log, error)
            time.sleep(30.)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def run_child(command, log, deadline, *, environment=None, group_binding=None, observer=None):
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("child budget exhausted")
    with log.open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                   env=environment, stdin=subprocess.DEVNULL)
        observed = ProcessInventory()
        try:
            pin_spawned(process, observed)
            while True:
                if observer is not None:
                    observer()
                if group_binding is not None:
                    grid.authenticate_group(group_binding)
                refresh_owned(process, observed)
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("owned stage deadline exhausted")
                try:
                    code = process.wait(timeout=min(2., remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
            if code != 0:
                raise RuntimeError(f"child failed ({code}); see {log}")
            drain_success(process, observed, deadline, group_binding=group_binding)
            if observer is not None:
                observer()
        except BaseException as error:
            try_lifecycle_diagnostic(process, observed, log, error)
            try:
                cleanup_spawned(process, observed)
            except BaseException as cleanup_error:
                try_lifecycle_diagnostic(process, observed, log, cleanup_error)
                quarantine_spawned(process, observed, log, cleanup_error)
            raise


def native(args):
    plan, work = checked_plan(args)
    ansys = authenticated_common(plan)
    original = ansys.run_mapdl
    deadline = min(args.deadline, plan["expires_epoch"])
    def bounded_mapdl(**values):
        grid.wait_ansys_idle(deadline, **({"group_binding": plan["group_binding"]} if plan.get("group_binding") else {}))
        remaining = deadline - time.time() - 180.
        if remaining <= 0:
            raise TimeoutError("MAPDL stage deadline exhausted")
        values["timeout_sec"] = min(values["timeout_sec"], remaining)
        return original(**values)
    ansys.run_mapdl = bounded_mapdl
    if not 0 <= args.index < len(plan["candidates"]) or not math.isfinite(deadline) or deadline <= time.time():
        raise ValueError("invalid native case/deadline")
    record = plan["candidates"][args.index]
    directory = work / "cases" / record["key"]
    verify_inputs(record, directory)
    if not args.preflight_only:
        stage = authorize_stage(plan, work, args, "teacher")
        verify_rom_seal(directory, stage["rom_seal_sha256"])
    command = teacher_command(plan, record, directory, plan_only=args.preflight_only)
    sys.path[:0] = [plan["source"], str(Path(plan["source"]).parent)]
    sys.argv = command[2:]
    grid.bounded_cpu(deadline, lambda: runpy.run_path(command[2], run_name="__main__"))


def authorize_stage(plan, work, args, mode):
    stage, _ = grid.read_record(work / "ACTIVE_STAGE.json", args.stage_sha256)
    identity = grid.process_identity(stage["worker_identity"]["pid"])
    status, _ = grid.read_record(work / "STATUS.json")
    if (identity != stage["worker_identity"] or status.get("worker_identity") != identity
            or stage.get("mode") != mode or stage.get("index") != args.index
            or stage.get("plan_sha256") != args.plan_sha256
            or status.get("state") != mode.upper() or status.get("active_candidate") != plan["candidates"][args.index]["key"]
            or args.deadline > stage["deadline_epoch"]):
        raise ValueError("native execution lacks bound live worker stage ownership")
    pid = os.getpid()
    for _ in range(8):
        if pid == identity["pid"]:
            break
        path = Path(f"/proc/{pid}/stat")
        if path.stat().st_uid != os.getuid():
            raise ValueError("stage process ancestry must belong to this UID")
        raw = path.read_text()
        pid = int(raw[raw.rindex(")") + 2:].split()[1])
    else:
        raise ValueError("stage process is not a descendant of the bound worker")
    if plan.get("group_binding") is not None:
        grid.assert_group_worker(plan["group_binding"], work)
        return stage
    lock = builder.scratch_root(plan["execution_host"]) / ".yap_operator_grid.lock"
    with lock.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return stage
        raise ValueError("worker does not hold shared solver ownership lock")


@contextmanager
def exclusive_worker(work):
    with (work / "validation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def prepare_validation_launch(args):
    plan, work = checked_plan(args)
    preflight, _ = grid.read_record(work / "PREFLIGHT_COMPLETE.json")
    if (preflight.get("plan_sha256") != args.plan_sha256
            or preflight.get("keys") != [r["key"] for r in plan["candidates"]]
            or preflight.get("solver_launched") is not False):
        raise ValueError("actual candidate/deck preflight incomplete")
    if (work / "LAUNCH.json").exists() or (work / "STATUS.json").exists():
        raise ValueError("validation attempt already exists; no automatic retry")
    builder.write_json(work / "LAUNCH_REQUEST.json", dict(plan_sha256=args.plan_sha256,
        launcher_identity=grid.process_identity(os.getpid())), immutable=True)
    environment = builder.configure(work)
    command = [sys.executable, "-B", str(Path(__file__)), "worker", "--work", str(work),
               "--plan-sha256", args.plan_sha256]
    return plan, work, environment, command


def launch(args, *, preparation_deadline=None, prepared=None):
    if prepared is None:
        prepared = prepare_validation_launch(args) if preparation_deadline is None else grid.bounded_cpu(
            preparation_deadline, lambda: prepare_validation_launch(args))
    plan, work, environment, command = prepared
    observed = ProcessInventory()
    child = None
    try:
        if preparation_deadline is not None and time.time() >= preparation_deadline:
            raise TimeoutError("worker launch preparation deadline exhausted")
        with (work / "worker.log").open("x") as stream:
            inherited = (grid.group_host_fd(plan["group_binding"]),) if plan.get("group_binding") else ()
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True,
                                     env=grid.group_launch_environment(environment, plan.get("group_binding")),
                                     pass_fds=inherited)
        identity = grid.wait_exec_identity(child, command, observed, deadline=preparation_deadline)
        receipt = dict(**scope(), worker_pid=child.pid, worker_identity=identity,
                       command=command, plan_sha256=args.plan_sha256, launched_epoch=time.time())
        builder.write_json(work / "LAUNCH.json", receipt, immutable=True)
        builder.copy_checked(work / "LAUNCH.json", Path(plan["archive"]) / "LAUNCH.json")
        if preparation_deadline is not None and time.time() >= preparation_deadline:
            raise TimeoutError("worker launch publication deadline exhausted")
        builder.write_json(work / "LAUNCH_PUBLISHED.json", dict(plan_sha256=args.plan_sha256,
            launch_sha256=builder.digest(work / "LAUNCH.json")), immutable=True)
        if preparation_deadline is not None and time.time() >= preparation_deadline:
            raise TimeoutError("worker launch publication deadline exhausted")
        print(receipt, flush=True)
        return child
    except BaseException:
        if child is not None:
            cleanup_spawned(child, observed)
        raise


def worker(args):
    with exclusive_worker(builder.local_scratch(args.work)):
        return worker_locked(args)


def worker_locked(args):
    plan, work = checked_plan(args)
    builder.configure(work)
    if (work / "STATUS.json").exists():
        raise RuntimeError("existing validation attempt; no automatic retry")
    handshake_end = min(time.time() + 60., plan["expires_epoch"])
    while not (work / "LAUNCH_PUBLISHED.json").exists():
        if time.time() >= handshake_end:
            raise TimeoutError("PID-bound launch publication handshake expired")
        time.sleep(.2)
    publication, _ = grid.read_record(work / "LAUNCH_PUBLISHED.json")
    launch_record, _ = grid.read_record(work / "LAUNCH.json", publication["launch_sha256"])
    identity = grid.process_identity(os.getpid())
    if (publication["plan_sha256"] != args.plan_sha256
            or launch_record["worker_identity"] != identity or launch_record["plan_sha256"] != args.plan_sha256):
        raise ValueError("launch publication worker identity mismatch")
    state = dict(**scope(), state="STARTING", worker_identity=grid.process_identity(os.getpid()),
                 completed=[], started_epoch=time.time())
    builder.write_json(work / "STATUS.json", state)
    publish_worker_scope(ProcessInventory())
    try:
        wait_build_queue(plan, work, state)
        with grid.worker_solver_lock(work, plan.get("group_binding")):
            for index, record in enumerate(plan["candidates"]):
                checked_plan(args)
                if plan["schema"] == "yap.operator_grid.thermal_archive_screen.v1":
                    archive_queue_resources(work, len(plan["candidates"]) - index,
                                            group_binding=plan.get("group_binding"))
                deadline = min(time.time() + CASE_SECONDS, plan["expires_epoch"] - 180.)
                state.update(state="PREPARING", active_candidate=record["key"], heartbeat_epoch=time.time())
                builder.write_json(work / "STATUS.json", state)
                grid.check_case_resources(work, **({"group_binding": plan["group_binding"]} if plan.get("group_binding") else {}))
                directory = work / "cases" / record["key"]
                verify_inputs(record, directory)
                preflight, _ = grid.read_record(directory / "PREFLIGHT_COMPLETE.json")
                if (preflight["plan_sha256"] != args.plan_sha256
                        or preflight["input_contract_sha256"] != builder.digest(directory / "INPUT_CONTRACT.json")
                        or preflight["teacher_manifest_sha256"] != builder.digest(directory / "preflight/M3_LAYER_INTERFACE_REPLAY_MANIFEST.json")):
                    raise ValueError("preflight evidence changed")
                def case_args(mode, end):
                    return [sys.executable, "-B", str(Path(__file__)), mode, "--work", str(work),
                        "--plan-sha256", args.plan_sha256, "--index", str(index), "--deadline", str(end)]
                sealed_sha = None
                for mode in ("predict", "teacher", "score"):
                    if mode == "predict" and record["key"] in plan.get("precomputed_rom_keys", []):
                        cached, _ = grid.read_record(directory / "CPU_PRECOMPUTE_COMPLETE.json")
                        if (cached.get("plan_sha256") != args.plan_sha256
                                or cached.get("input_contract_sha256") != builder.digest(directory / "INPUT_CONTRACT.json")
                                or cached.get("pins_sha256") != plan["pins_sha256"]):
                            raise ValueError("CPU precomputed prediction authority drift")
                        sealed_sha = cached["rom_seal_sha256"]
                        verify_rom_seal(directory, sealed_sha)
                        continue
                    end = min(deadline - 120., time.time() + (STAGE_SECONDS if mode in ("predict", "teacher") else 1800.))
                    state.update(state=mode.upper(), active_candidate=record["key"], heartbeat_epoch=time.time())
                    builder.write_json(work / "STATUS.json", state)
                    stage = dict(worker_identity=state["worker_identity"], mode=mode, index=index,
                                 plan_sha256=args.plan_sha256, deadline_epoch=end, rom_seal_sha256=sealed_sha)
                    builder.write_json(work / "ACTIVE_STAGE.json", stage)
                    command = case_args(mode, end) + ["--stage-sha256", builder.digest(work / "ACTIVE_STAGE.json")]
                    run_child(command, work / f"{record['key']}_{mode}.log", end + 90., group_binding=plan.get("group_binding"))
                    if mode == "predict":
                        builder.write_json(directory / "ROM_SEAL.json", dict(**scope(), files_sha256={
                            name: builder.digest(directory / name) for name in (
                                "ROM_COMPLETE.json", "ROM_RESPONSE.npz", "ROM_CU_STATES.npz")}), immutable=True)
                        sealed_sha = builder.digest(directory / "ROM_SEAL.json")
                    if grid.foreign_ansys_blockers(plan.get("group_binding")):
                        raise RuntimeError("unexpected live ANSYS process after completed stage; no following stage")
                    if mode == "teacher" and plan["schema"] == "yap.operator_grid.thermal_archive_screen.v1":
                        builder.write_json(directory / "TEACHER_SEAL.json", dict(files_sha256={
                            name: builder.digest(directory / name) for name in (
                                "teacher/M3_LAYER_INTERFACE_REPLAY_RESULTANTS.csv",
                                "teacher/M3_LAYER_INTERFACE_FACE_REACTIONS.csv")}), immutable=True)
                receipt, _ = grid.read_record(directory / "THERMAL_SCREEN.json")
                destination = Path(plan["archive"]) / record["key"]
                destination.mkdir()
                for name in ("INPUT_CONTRACT.json", "ROM_COMPLETE.json", "ROM_SEAL.json", "ROM_RESPONSE.npz",
                             "ROM_PROGRESS.csv", "ROM_CU_STATES.npz", "SCORED_RESPONSE.npz", "THERMAL_SCREEN.json",
                             "teacher/M3_LAYER_INTERFACE_REPLAY_RESULTANTS.csv",
                             "teacher/M3_LAYER_INTERFACE_FACE_REACTIONS.csv"):
                    expected = {"teacher/M3_LAYER_INTERFACE_REPLAY_RESULTANTS.csv": receipt["reference_resultants_sha256"],
                                "teacher/M3_LAYER_INTERFACE_FACE_REACTIONS.csv": receipt["reference_face_reactions_sha256"]}.get(name)
                    grid.bounded_cpu(deadline, lambda name=name, expected=expected:
                        builder.copy_checked(directory / name, destination / name, expected))
                if plan["schema"] == "yap.operator_grid.thermal_archive_screen.v1":
                    builder.copy_checked(directory / "TEACHER_SEAL.json", destination / "TEACHER_SEAL.json")
                state["completed"].append(dict(key=record["key"], state=receipt["state"]))
                builder.write_json(work / "STATUS.json", state)
                if not scientific_screen_result(plan, record, receipt):
                    raise RuntimeError(f"thermal screen failed for {record['key']}; preserve evidence, review before next candidate")
            state.update(state=completed_screen_state(state["completed"]), heartbeat_epoch=time.time())
            builder.write_json(work / "STATUS.json", state)
            builder.write_json(Path(plan["archive"]) / "BATCH_SCREEN_COMPLETE.json", state, immutable=True)
    except BaseException as exc:
        state.update(state="STOPPED_REQUIRES_REVIEW", error=str(exc), heartbeat_epoch=time.time())
        builder.write_json(work / "STATUS.json", state)
        builder.write_json(work / "STOP.json", state, immutable=True)
        raise


def bind_thermal_predecessor(root):
    root = builder.local_scratch(root)
    plan, plan_sha = grid.read_record(root / "PLAN.json")
    launch_record, launch_sha = grid.read_record(root / "LAUNCH.json")
    identity = launch_record.get("worker_identity")
    if (plan.get("schema") != "yap.operator_grid.thermal_screen.v1"
            or plan.get("execution_host") != builder.physical_host() or plan.get("work") != str(root)
            or plan.get("runtime_admitted") is not False or plan.get("thermal_geometry_validated") is not False
            or launch_record.get("plan_sha256") != plan_sha or not isinstance(identity, dict)
            or launch_record.get("worker_pid") != identity.get("pid")
            or launch_record.get("command") != identity.get("command")):
        raise ValueError("follow-up requires an identity-bound thermal predecessor")
    binding = dict(work=str(root), plan_sha256=plan_sha, launch_sha256=launch_sha,
        identity=identity, archive=plan["archive"], keys=[r["key"] for r in plan["candidates"]],
        wait_deadline_epoch=min(time.time() + FOLLOW_WAIT_SECONDS, plan["expires_epoch"] + EXIT_GRACE_SECONDS))
    thermal_predecessor_complete(binding)
    return binding


def thermal_predecessor_complete(binding):
    root = Path(binding["work"])
    plan, _ = grid.read_record(root / "PLAN.json", binding["plan_sha256"])
    launch_record, _ = grid.read_record(root / "LAUNCH.json", binding["launch_sha256"])
    if (plan["archive"] != binding["archive"] or [r["key"] for r in plan["candidates"]] != binding["keys"]
            or launch_record["worker_identity"] != binding["identity"]):
        raise ValueError("thermal predecessor binding drift")
    identity = grid.process_identity(binding["identity"]["pid"])
    if identity is not None and identity != binding["identity"]:
        raise ValueError("thermal predecessor process identity drift")
    status, _ = grid.read_record(root / "STATUS.json")
    if status.get("worker_identity") != binding["identity"]:
        raise ValueError("thermal predecessor status identity drift")
    if (root / "STOP.json").exists() or status.get("state", "").startswith(("STOP", "FAIL")):
        raise RuntimeError("thermal predecessor failed; no automatic takeover")
    if status.get("state") != "BATCH_SCREEN_COMPLETE_NOT_QUALIFIED":
        if identity is None:
            status, _ = grid.read_record(root / "STATUS.json")
            if status.get("state") != "BATCH_SCREEN_COMPLETE_NOT_QUALIFIED":
                raise RuntimeError("thermal predecessor exited without successful completion")
        else:
            return False
    if identity is not None:
        return False
    completed, _ = grid.read_record(Path(binding["archive"]) / "BATCH_SCREEN_COMPLETE.json")
    expected = [dict(key=key, state="PASS_PRESCRIBED_SCREEN") for key in binding["keys"]]
    if completed != status or completed.get("completed") != expected:
        raise ValueError("thermal predecessor needs the exact whole-batch screen receipt")
    if not grid.predecessor_sessions_idle([binding["identity"]], worker_pid=binding["identity"]["pid"]):
        raise RuntimeError("thermal predecessor left live descendants")
    for record in plan["candidates"]:
        key = record["key"]
        result, _ = grid.read_record(Path(binding["archive"]) / key / "THERMAL_SCREEN.json")
        if (result.get("state") != "PASS_PRESCRIBED_SCREEN" or result.get("key") != key
                or result.get("candidate_sha256") != record["files_sha256"]):
            raise ValueError("thermal predecessor case screen mismatch")
    return True


def checked_followup(args):
    work = builder.local_scratch(args.work)
    plan, _ = grid.read_record(work / "FOLLOW_UP.json", args.plan_sha256)
    if (plan.get("schema") != "yap.operator_grid.follow_up.v1" or plan.get("work") != str(work)
            or plan.get("execution_host") != builder.physical_host() or plan.get("automatic_retry") is not False
            or plan.get("runtime_admitted") is not False or plan.get("thermal_geometry_validated") is not False):
        raise ValueError("follow-up policy drift")
    grid.validate_batch(plan["si_thicknesses_um"])
    for filename, expected in plan["controllers_sha256"].items():
        if builder.digest(grid.regular(filename)) != expected:
            raise ValueError("frozen follow-up controller drift")
    grid.verify_pins(work / "SOURCE_PINS.json", plan["pins_sha256"])
    if time.time() >= plan["expires_epoch"]:
        raise TimeoutError("follow-up queue expired")
    return plan, work


def followup_launch(args):
    work = builder.local_scratch(args.work)
    archive = builder.safe_path(args.archive)
    if work.exists() or archive.exists() or archive.parent != grid.ARCHIVES or not archive.name.startswith("operator_grid_p2_followup_"):
        raise ValueError("follow-up requires fresh local work and canonical archive")
    if args.review_sha256 != builder.digest(grid.regular(__file__)):
        raise ValueError("follow-up launch requires the reviewed controller digest")
    heights = grid.validate_batch(args.si_thicknesses)
    predecessor = bind_thermal_predecessor(args.after_thermal)
    roots = [builder.safe_path(p[0]) for p in args.candidate]
    expected_heights = [float(p[1]) for p in args.candidate]
    if not 1 <= len(roots) <= 18 - len(heights) or len(set(roots)) != len(roots):
        raise ValueError("bounded unique supplemental candidate roots required")
    allowed = {h for total in grid.TOTALS for h in (total - 13., total - 11.5)}
    if (any(h not in allowed for h in expected_heights)
            or len(set(expected_heights + heights)) != len(expected_heights) + len(heights)):
        raise ValueError("supplemental thicknesses must be explicit, supported and unique")
    candidates = []
    for root, h in zip(roots, expected_heights):
        if not root.is_relative_to(grid.ARCHIVES):
            raise ValueError("supplemental candidates must use canonical layer archives")
        receipt = root / "ARCHIVE.json"
        candidates.append(dict(root=str(root), key=grid.si_key(h), si_thickness_um=h,
            archive_sha256=builder.digest(grid.regular(receipt)) if receipt.exists() else None))
    source = builder.safe_path(args.source)
    source_pins = closure(source)
    controls = [Path(__file__), Path(grid.__file__), Path(builder.__file__)]
    controls += [Path(__file__).with_name("package_assembly_runtime.py"),
                 Path(__file__).resolve().parents[1] / "mechanics_runtime.py"]
    controls_sha = {str(grid.regular(p)): builder.digest(p) for p in controls}
    work.mkdir(parents=True)
    archive.mkdir()
    builder.configure(work)
    builder.write_json(work / "SOURCE_PINS.json", source_pins, immutable=True)
    # Waiting owns no solver lock; each subsequent phase has its own fixed budget.
    count = len(candidates) + len(heights)
    plan = dict(schema="yap.operator_grid.follow_up.v1", **scope(), work=str(work), archive=str(archive),
        execution_host=builder.physical_host(), source=str(source), si_thicknesses_um=heights,
        baseline_runtime=str(builder.safe_path(args.baseline_runtime)),
        parent_package=str(grid.regular(args.parent_package)), predecessor=predecessor,
        candidates=candidates, controllers_sha256=controls_sha, pins_sha256=builder.digest(work / "SOURCE_PINS.json"),
        automatic_retry=False, created_epoch=time.time(), build_preparation_seconds=BUILD_PREPARATION_SECONDS,
        build_seconds=grid.QUEUE_SECONDS + EXIT_GRACE_SECONDS,
        preflight_seconds=count * 3600., screening_seconds=WAIT_SECONDS + count * CASE_SECONDS + EXIT_GRACE_SECONDS,
        expires_epoch=predecessor["wait_deadline_epoch"] + BUILD_PREPARATION_SECONDS + grid.QUEUE_SECONDS + count * 3600.
            + WAIT_SECONDS + count * CASE_SECONDS + 3 * EXIT_GRACE_SECONDS)
    builder.write_json(work / "FOLLOW_UP.json", plan, immutable=True)
    plan_sha = builder.digest(work / "FOLLOW_UP.json")
    for name in ("FOLLOW_UP.json", "SOURCE_PINS.json"):
        builder.copy_checked(work / name, archive / name)
    command = [sys.executable, "-B", str(Path(__file__)), "follow-up-worker", "--work", str(work), "--plan-sha256", plan_sha]
    observed = ProcessInventory()
    child = None
    try:
        with (work / "worker.log").open("x") as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                     env=builder.configure(work), start_new_session=True)
        identity = grid.wait_exec_identity(child, command, observed)
        receipt = dict(**scope(), worker_pid=child.pid, worker_identity=identity, command=command, plan_sha256=plan_sha)
        builder.write_json(work / "LAUNCH.json", receipt, immutable=True)
        builder.copy_checked(work / "LAUNCH.json", archive / "LAUNCH.json")
        builder.write_json(work / "LAUNCH_PUBLISHED.json", dict(plan_sha256=plan_sha,
            launch_sha256=builder.digest(work / "LAUNCH.json")), immutable=True)
        print(receipt, flush=True)
        return child
    except BaseException:
        if child is not None:
            cleanup_spawned(child, observed)
        raise


def monitor_followup_child(child, child_work, deadline, state, work):
    observed = ProcessInventory()
    try:
        pin_spawned(child, observed)
        receipt, launch_sha = grid.read_record(child_work / "LAUNCH.json")
        identity = receipt["worker_identity"]
        if identity["pid"] != child.pid:
            raise ValueError("follow-up child launch identity mismatch")
        observed[(identity["pid"], identity["start_ticks"])] = identity
        observed.scope_work, observed.scope_identity, observed.scope_launch_sha = child_work, identity, launch_sha
        if "campaign_sha256" in receipt:
            phase_plan, _ = grid.read_record(child_work / "CAMPAIGN.json", receipt["campaign_sha256"])
        elif "plan_sha256" in receipt:
            phase_plan, _ = grid.read_record(child_work / "PLAN.json", receipt["plan_sha256"])
        else:
            raise ValueError("follow-up child has no authenticated phase deadline")
        deadline = min(deadline, phase_plan["expires_epoch"] + EXIT_GRACE_SECONDS)
        state.update(child_work=str(child_work), child_identity=identity, phase_deadline_epoch=deadline)
        builder.write_json(work / "STATUS.json", state)
        heartbeat = 0.
        while True:
            current = grid.process_identity(child.pid)
            if current is not None and current != identity:
                raise ValueError("follow-up child process identity drift")
            refresh_owned(child, observed)
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("follow-up phase deadline exhausted")
            if time.time() - heartbeat >= 60.:
                heartbeat = time.time()
                state.update(heartbeat_epoch=heartbeat)
                builder.write_json(work / "STATUS.json", state)
                print(f"Follow-up {state['state']}: child PID {child.pid}", flush=True)
            try:
                code = child.wait(timeout=min(10., remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        if code != 0:
            raise RuntimeError(f"follow-up child failed ({code}); see {child_work}")
        drain_success(child, observed, deadline)
        if time.time() >= deadline:
            raise TimeoutError("follow-up child completion proof exceeded deadline")
    except BaseException as error:
        log = work / "followup_child.log"
        try_lifecycle_diagnostic(child, observed, log, error)
        try:
            cleanup_spawned(child, observed)
        except BaseException as cleanup_error:
            quarantine_spawned(child, observed, log, cleanup_error)
        raise


def followup_worker(args):
    plan, work = checked_followup(args)
    builder.configure(work)
    with exclusive_worker(work):
        if (work / "STATUS.json").exists():
            raise RuntimeError("existing follow-up attempt; no automatic retry")
        handshake_end = time.time() + 60.
        while not (work / "LAUNCH_PUBLISHED.json").exists():
            if time.time() >= handshake_end:
                raise TimeoutError("follow-up launch handshake expired")
            time.sleep(.2)
        published, _ = grid.read_record(work / "LAUNCH_PUBLISHED.json")
        launch_record, _ = grid.read_record(work / "LAUNCH.json", published["launch_sha256"])
        if (published["plan_sha256"] != args.plan_sha256 or launch_record["plan_sha256"] != args.plan_sha256
                or launch_record["worker_identity"] != grid.process_identity(os.getpid())):
            raise ValueError("follow-up launch handshake identity mismatch")
        state = dict(**scope(), state="WAITING_THERMAL_SCREEN", worker_identity=launch_record["worker_identity"],
                     completed=[], heartbeat_epoch=time.time())
        builder.write_json(work / "STATUS.json", state)
        try:
            while True:
                checked_followup(args)
                if time.time() >= plan["predecessor"]["wait_deadline_epoch"]:
                    raise TimeoutError("follow-up predecessor wait expired")
                if thermal_predecessor_complete(plan["predecessor"]):
                    if time.time() >= plan["predecessor"]["wait_deadline_epoch"]:
                        raise TimeoutError("follow-up predecessor proof exceeded deadline")
                    break
                state.update(heartbeat_epoch=time.time())
                builder.write_json(work / "STATUS.json", state)
                print("Waiting for exact thermal batch success and solver ownership release", flush=True)
                time.sleep(60.)
            grid.verify_pins(work / "SOURCE_PINS.json", plan["pins_sha256"])
            builder.write_json(work / "PREDECESSOR_RELEASE.json", plan["predecessor"], immutable=True)
            build_work = work / "build"
            # Ordinary builds require a direct child of the canonical results root.
            build_archive = Path(plan["archive"]).with_name(Path(plan["archive"]).name + "_build")
            state.update(state="BUILDING_REMAINING_P2")
            preparation_end = min(time.time() + plan["build_preparation_seconds"], plan["expires_epoch"])
            end = min(preparation_end + plan["build_seconds"], plan["expires_epoch"])
            child = grid.launch(SimpleNamespace(si_thicknesses=plan["si_thicknesses_um"], work=build_work,
                archive=build_archive, native_source=Path(plan["source"]), baseline_runtime=Path(plan["baseline_runtime"]),
                parent_package=Path(plan["parent_package"]), review_sha256=builder.digest(grid.regular(grid.__file__)),
                after_campaign=None, adopt_from=None), preparation_deadline=preparation_end)
            monitor_followup_child(child, build_work, end, state, work)
            checked_followup(args)
            launch_record, launch_sha = grid.read_record(build_work / "LAUNCH.json")
            _, campaign_sha = grid.read_record(build_work / "CAMPAIGN.json")
            binding = dict(work=str(build_work), archive=str(build_archive), launch_sha256=launch_sha,
                campaign_sha256=campaign_sha, identity=launch_record["worker_identity"], si_thicknesses_um=plan["si_thicknesses_um"])
            grid.bounded_cpu(state["phase_deadline_epoch"], lambda: completed_build_binding(binding))
            state["completed"].append(dict(phase="build", archive=str(build_archive)))
            state.update(state="PREFLIGHT_SUPPLEMENTAL_SCREEN", heartbeat_epoch=time.time())
            builder.write_json(work / "STATUS.json", state)
            roots = []
            for value in plan["candidates"]:
                root = Path(value["root"])
                if value["archive_sha256"] is not None:
                    grid.read_record(root / "ARCHIVE.json", value["archive_sha256"])
                roots.append(root)
            roots.extend(build_archive / ("p2_" + grid.si_key(h)) for h in plan["si_thicknesses_um"])
            expected_keys = [v["key"] for v in plan["candidates"]] + [grid.si_key(h) for h in plan["si_thicknesses_um"]]
            screen_work = work / "screen"
            screen_archive = Path(plan["archive"]).with_name("operator_grid_p2_thermal_"
                + Path(plan["archive"]).name.removeprefix("operator_grid_p2_followup_") + "_supplement")
            end = min(time.time() + plan["preflight_seconds"], plan["expires_epoch"])
            bounded = SimpleNamespace(work=screen_work, archive=screen_archive, source=Path(plan["source"]),
                candidate=roots, predecessor=build_work, predecessor_pid=launch_record["worker_pid"])
            grid.bounded_cpu(end, lambda: plan_command(bounded))
            screen_plan, _ = grid.read_record(screen_work / "PLAN.json")
            if [r["key"] for r in screen_plan["candidates"]] != expected_keys:
                raise ValueError("supplemental screen candidates differ from frozen thicknesses")
            screen_args = SimpleNamespace(work=screen_work, plan_sha256=builder.digest(screen_work / "PLAN.json"))
            grid.bounded_cpu(end, lambda: preflight_all(screen_args))
            checked_followup(args)
            state.update(state="SCREENING_SUPPLEMENTAL_P2")
            end = min(time.time() + plan["screening_seconds"], plan["expires_epoch"])
            child = launch(screen_args, preparation_deadline=min(time.time() + 1800., end))
            monitor_followup_child(child, screen_work, end, state, work)
            proof = grid.bounded_cpu(state["phase_deadline_epoch"],
                lambda: thermal_predecessor_complete(bind_thermal_predecessor(screen_work)))
            if not proof:
                raise RuntimeError("supplemental screen has not released solver ownership")
            state["completed"].append(dict(phase="screen", archive=str(screen_archive)))
            state.update(state="FOLLOW_UP_SCREEN_COMPLETE_NOT_QUALIFIED", heartbeat_epoch=time.time())
            builder.write_json(work / "STATUS.json", state)
            builder.write_json(Path(plan["archive"]) / "FOLLOW_UP_COMPLETE.json", state, immutable=True)
        except BaseException as exc:
            state.update(state="STOPPED_REQUIRES_REVIEW", error=str(exc), heartbeat_epoch=time.time())
            builder.write_json(work / "STATUS.json", state)
            builder.write_json(work / "STOP.json", state, immutable=True)
            builder.copy_checked(work / "STOP.json", Path(plan["archive"]) / "STOP.json")
            raise


def parallel_pair_launch(args):
    host = builder.physical_host()
    work = builder.local_scratch(args.work, host=host)
    archive = builder.safe_path(args.archive)
    if (work.exists() or archive.exists() or archive.parent != grid.ARCHIVES
            or not archive.name.startswith("operator_grid_p2_parallel_")):
        raise ValueError("parallel pair requires fresh owned work and canonical archive")
    if args.review_sha256 != builder.digest(grid.regular(__file__)):
        raise ValueError("parallel launch must bind the reviewed controller")
    old, old_sha = grid.read_record(args.thermal_plan, args.thermal_plan_sha256)
    if (old.get("schema") != "yap.operator_grid.thermal_screen.v1"
            or old.get("execution_host") != host or len(old.get("candidates", [])) != 8):
        raise ValueError("expected the reviewed same-host eight-case screen plan")
    heights = grid.validate_batch(args.si_thicknesses)
    resource = grid.check_case_resources(builder.scratch_root(host))
    if resource["local_free_gib"] < 328 or resource["available_memory_gib"] < 128:
        raise RuntimeError("parallel pair requires 328 GiB disk and 128 GiB available RAM")
    builder.configure(work)
    archive.mkdir()
    controllers = [Path(__file__).resolve(), Path(grid.__file__).resolve(), Path(builder.__file__).resolve()]
    started = time.time()
    jobs = [dict(kind="build", work=str(work / "build"), archive=str(archive) + "_build",
                 controller=str(Path(grid.__file__).resolve())),
            dict(kind="screen", work=str(work / "screen"),
                 archive=str(archive.with_name("operator_grid_p2_thermal_" + archive.name + "_screen")),
                 controller=str(Path(__file__).resolve()))]
    plan = dict(schema="yap.operator_grid.parallel_pair.v1", work=str(work), archive=str(archive),
        execution_host=host, source=str(builder.safe_path(args.source)),
        baseline_runtime=str(builder.safe_path(args.baseline_runtime)),
        parent_package=str(builder.safe_path(args.parent_package)), jobs=jobs,
        controllers_sha256={str(p): builder.digest(grid.regular(p)) for p in controllers},
        thermal_plan=str(grid.regular(args.thermal_plan)), thermal_plan_sha256=old_sha,
        si_thicknesses_um=heights, resources=resource, cores_per_job=8,
        reserved_memory_gib=128, disk_floor_gib=328, preparation_seconds=21600,
        created_epoch=started, expires_epoch=started + 21600 + WAIT_SECONDS + 8 * CASE_SECONDS + EXIT_GRACE_SECONDS,
        automatic_retry=False, runtime_admitted=False)
    builder.write_json(work / "GROUP_PLAN.json", plan, immutable=True)
    plan_sha = builder.digest(work / "GROUP_PLAN.json")
    builder.copy_checked(work / "GROUP_PLAN.json", archive / "GROUP_PLAN.json", plan_sha)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "parallel-worker",
               "--work", str(work), "--plan-sha256", plan_sha]
    child, observed = None, ProcessInventory()
    try:
        with (work / "worker.log").open("x") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True, env=builder.configure(work))
        identity = grid.wait_exec_identity(child, command, observed)
        receipt = dict(command=command, worker_identity=identity, worker_pid=child.pid,
                       plan_sha256=plan_sha, launched_epoch=time.time())
        builder.write_json(work / "LAUNCH.json", receipt, immutable=True)
        builder.copy_checked(work / "LAUNCH.json", archive / "LAUNCH.json")
        builder.write_json(work / "LAUNCH_PUBLISHED.json", dict(plan_sha256=plan_sha,
            launch_sha256=builder.digest(work / "LAUNCH.json")), immutable=True)
        print(receipt, flush=True)
        return child
    except BaseException:
        if child is not None:
            cleanup_spawned(child, observed)
        raise


def register_group_child(work, group_sha, slot, job, child):
    root = Path(job["work"])
    launch_record, launch_sha = grid.read_record(root / "LAUNCH.json")
    filename = "CAMPAIGN.json" if job["kind"] == "build" else "PLAN.json"
    plan, plan_sha = grid.read_record(root / filename)
    identity = grid.process_identity(child.pid)
    if identity is None or launch_record.get("worker_identity") != identity:
        raise ValueError("parallel child exited or changed before registration")
    receipt = dict(work=str(root), group_plan_sha256=group_sha, identity=identity,
        launch_sha256=launch_sha, plan_filename=filename, plan_sha256=plan_sha,
        pins_sha256=plan.get("source_pins_sha256", plan.get("pins_sha256")))
    builder.write_json(work / f"CHILD{slot}.json", receipt, immutable=True)
    return receipt, plan


def cleanup_group_children(children, observations):
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT, signal.SIGALRM})
    failures = []
    try:
        for child, observed in zip(children, observations):
            try:
                cleanup_spawned(child, observed)
            except BaseException as exc:
                failures.append(str(exc))
        if failures:
            raise RuntimeError("parallel pair cleanup incomplete: " + "; ".join(failures))
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def publish_group_ready(work, group_sha, *, slots=(0, 1)):
    temporary = work / "GROUP_READY.pending.json"
    builder.write_json(temporary, dict(group_plan_sha256=group_sha,
        children_sha256={str(i): builder.digest(work / f"CHILD{i}.json") for i in slots}), immutable=True)
    os.link(temporary, work / "GROUP_READY.json", follow_symlinks=False)
    temporary.unlink()


def contain_group_failure(children, observations, state, work, archive):
    def diagnostic(message):
        try:
            print(message, flush=True)
        except BaseException:
            pass
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT, signal.SIGALRM})
    try:
        try:
            cleanup_group_children(children, observations)
        except BaseException as exc:
            state.update(state="CONTAINMENT_FAILED_HOLDING_LOCK", cleanup_error=str(exc), heartbeat_epoch=time.time())
            try:
                builder.write_json(work / "STATUS.json", state)
                builder.write_json(work / "CONTAINMENT_FAILURE.json", state, immutable=True)
                builder.copy_checked(work / "CONTAINMENT_FAILURE.json", archive / "CONTAINMENT_FAILURE.json")
            except BaseException as evidence_error:
                diagnostic(f"Cannot record containment failure: {evidence_error}")
            # No further native work is allowed. Keep exclusive ownership if a
            # recorded process is still alive; never release it on cleanup error.
            while group_has_live_children(children, observations):
                time.sleep(30.)
                state.update(heartbeat_epoch=time.time())
                try:
                    builder.write_json(work / "STATUS.json", state)
                except BaseException as evidence_error:
                    diagnostic(f"Cannot record quarantine heartbeat: {evidence_error}")
        state.update(state="STOPPED_REQUIRES_REVIEW", heartbeat_epoch=time.time())
        builder.write_json(work / "STATUS.json", state)
        builder.write_json(work / "STOP.json", state, immutable=True)
        builder.copy_checked(work / "STOP.json", archive / "STOP.json")
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def group_has_live_children(children, observations):
    try:
        active = False
        for child, observed in zip(children, observations):
            refresh_owned(child, observed)
            active = child.poll() is None or bool(live_owned(observed)) or bool(scoped_members(observed)) or active
        return active
    except BaseException:
        return True


def parallel_pair_worker(args):
    work = builder.local_scratch(args.work)
    plan, _ = grid.read_record(work / "GROUP_PLAN.json", args.plan_sha256)
    archive = builder.safe_path(plan["archive"])
    builder.configure(work)
    end = time.time() + 60.
    while not (work / "LAUNCH_PUBLISHED.json").exists():
        if time.time() >= end:
            raise TimeoutError("parallel launch publication expired")
        time.sleep(.2)
    published, _ = grid.read_record(work / "LAUNCH_PUBLISHED.json")
    receipt, _ = grid.read_record(work / "LAUNCH.json", published["launch_sha256"])
    identity = grid.process_identity(os.getpid())
    if (plan.get("schema") != "yap.operator_grid.parallel_pair.v1" or plan.get("work") != str(work)
            or plan.get("execution_host") != builder.physical_host() or len(plan.get("jobs", [])) != 2
            or published["plan_sha256"] != args.plan_sha256 or receipt.get("worker_identity") != identity):
        raise ValueError("parallel supervisor publication mismatch")
    state = dict(**scope(), state="PREPARING_PAIR", worker_identity=identity, completed=[], heartbeat_epoch=time.time())
    children, observations, registrations, child_plans = [], [], [], []
    builder.write_json(work / "STATUS.json", state)
    with grid.native_host_lock(work) as held:
        builder.write_json(work / "HOST_LOCK.json", dict(**held, worker_identity=identity,
            plan_sha256=args.plan_sha256), immutable=True)
        binding = dict(work=str(work), plan_sha256=args.plan_sha256,
                       launch_sha256=published["launch_sha256"], slot=0)
        try:
            grid.authenticate_group(binding)
            old, _ = grid.read_record(plan["thermal_plan"], plan["thermal_plan_sha256"])
            preparation_end = min(time.time() + plan["preparation_seconds"], plan["expires_epoch"])
            build_job, screen_job = plan["jobs"]
            build_args = SimpleNamespace(work=Path(build_job["work"]), archive=Path(build_job["archive"]),
                native_source=Path(plan["source"]), baseline_runtime=Path(plan["baseline_runtime"]),
                parent_package=Path(plan["parent_package"]), si_thicknesses=plan["si_thicknesses_um"],
                review_sha256=builder.digest(grid.regular(grid.__file__)), after_campaign=None,
                adopt_from=None, group_binding=binding)
            build_prepared = grid.bounded_cpu(preparation_end, lambda: grid.prepare_launch(build_args))
            old_predecessor = old["predecessor"]
            screen_plan_args = SimpleNamespace(work=Path(screen_job["work"]), archive=Path(screen_job["archive"]),
                source=Path(plan["source"]), candidate=[Path(r["root"]) for r in old["candidates"]],
                predecessor=Path(old_predecessor["work"]), predecessor_pid=old_predecessor["identity"]["pid"],
                predecessor_identity_plan=Path(plan["thermal_plan"]),
                predecessor_identity_sha256=plan["thermal_plan_sha256"], group_binding=dict(binding, slot=1))
            grid.bounded_cpu(preparation_end, lambda: plan_command(screen_plan_args))
            screen_args = SimpleNamespace(work=Path(screen_job["work"]),
                plan_sha256=builder.digest(Path(screen_job["work"]) / "PLAN.json"))
            grid.bounded_cpu(preparation_end, lambda: preflight_all(screen_args))
            screen_prepared = grid.bounded_cpu(preparation_end, lambda: prepare_validation_launch(screen_args))
            builder.configure(work)
            grid.check_case_resources(work, group_binding=binding)
            # Keep asynchronous signals out of the spawn/registration handoff.
            # Child roots unblock them after installing parent-death containment.
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT, signal.SIGALRM})
            try:
                for slot, (job, launcher, child_args, prepared) in enumerate((
                    (build_job, grid.launch, build_args, build_prepared),
                    (screen_job, launch, screen_args, screen_prepared))):
                    child = launcher(child_args, prepared=prepared, preparation_deadline=preparation_end)
                    children.append(child)
                    observed = ProcessInventory()
                    observations.append(observed)
                    pin_spawned(child, observed)
                    registered, child_plan = register_group_child(work, args.plan_sha256, slot, job, child)
                    observed.scope_work, observed.scope_identity = Path(registered["work"]), registered["identity"]
                    observed.scope_launch_sha = registered["launch_sha256"]
                    registrations.append(registered)
                    child_plans.append(child_plan)
                if time.time() >= preparation_end:
                    raise TimeoutError("parallel preparation deadline exhausted before group release")
                publish_group_ready(work, args.plan_sha256)
                if time.time() >= preparation_end:
                    raise TimeoutError("parallel preparation deadline exhausted during group release")
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            builder.configure(work)
            state.update(state="RUNNING_PAIR", children=registrations)
            builder.write_json(work / "STATUS.json", state)
            last_heartbeat = 0.
            drain_ends = {}
            while len(state["completed"]) < 2:
                grid.authenticate_group(binding)
                for slot, child in enumerate(children):
                    if slot in state["completed"]:
                        continue
                    refresh_owned(child, observations[slot])
                    deadline = min(plan["expires_epoch"], child_plans[slot]["expires_epoch"] + EXIT_GRACE_SECONDS)
                    if time.time() >= deadline:
                        raise TimeoutError(f"parallel slot {slot} deadline exhausted")
                    code = child.poll()
                    if code is not None:
                        child.wait(timeout=30.)
                        if code != 0:
                            raise RuntimeError(f"parallel slot {slot} failed ({code})")
                        if slot not in drain_ends:
                            drain_ends[slot] = time.monotonic() + min(SUCCESS_DRAIN_SECONDS, deadline - time.time())
                        if success_ready(child, observations[slot], deadline, drain_ends[slot], group_binding=binding):
                            state["completed"].append(slot)
                if time.time() - last_heartbeat >= 60.:
                    last_heartbeat = time.time()
                    state.update(heartbeat_epoch=last_heartbeat)
                    builder.write_json(work / "STATUS.json", state)
                    print(f"Parallel pair: completed={state['completed']}; roots={[c.pid for c in children]}", flush=True)
                if len(state["completed"]) < 2:
                    time.sleep(2.)
            build_root = Path(build_job["work"])
            build_launch, build_sha = grid.read_record(build_root / "LAUNCH.json")
            _, campaign_sha = grid.read_record(build_root / "CAMPAIGN.json")
            grid.bounded_cpu(plan["expires_epoch"], lambda: completed_build_binding(dict(
                work=str(build_root), archive=build_job["archive"], launch_sha256=build_sha,
                campaign_sha256=campaign_sha, identity=build_launch["worker_identity"],
                si_thicknesses_um=plan["si_thicknesses_um"]), group_binding=binding))
            proof = grid.bounded_cpu(plan["expires_epoch"], lambda: thermal_predecessor_complete(
                bind_thermal_predecessor(Path(screen_job["work"]))))
            if not proof:
                raise RuntimeError("parallel thermal child lacks exact successful completion receipts")
            state.update(state="PAIR_COMPLETE_NOT_QUALIFIED", heartbeat_epoch=time.time())
            builder.write_json(work / "STATUS.json", state)
            builder.write_json(archive / "PAIR_COMPLETE.json", state, immutable=True)
        except BaseException as exc:
            state.update(state="STOPPED_REQUIRES_REVIEW", error=str(exc), heartbeat_epoch=time.time())
            contain_group_failure(children, observations, state, work, archive)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("archive-plan")
    for name in ("work", "archive", "source"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--candidate", nargs=2, metavar=("ARCHIVE", "SI_UM"), action="append", required=True)
    p = sub.add_parser("plan")
    for name in ("work", "archive", "source", "predecessor"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--candidate", type=Path, action="append", required=True)
    p.add_argument("--predecessor-pid", type=int, required=True)
    p.add_argument("--predecessor-identity-plan", type=Path)
    p.add_argument("--predecessor-identity-sha256")
    p = sub.add_parser("follow-up-launch")
    for name in ("work", "archive", "source", "after-thermal", "baseline-runtime", "parent-package"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--si-thicknesses", type=float, nargs="+", required=True)
    p.add_argument("--candidate", nargs=2, metavar=("ARCHIVE", "SI_UM"), action="append", required=True)
    p.add_argument("--review-sha256", required=True)
    p = sub.add_parser("follow-up-worker")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    p = sub.add_parser("parallel-launch")
    for name in ("work", "archive", "source", "baseline-runtime", "parent-package", "thermal-plan"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--thermal-plan-sha256", required=True)
    p.add_argument("--si-thicknesses", type=float, nargs="+", required=True)
    p.add_argument("--review-sha256", required=True)
    p = sub.add_parser("parallel-worker")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    for mode in ("worker", "launch", "preflight-all", "prepare", "preflight", "predict", "teacher", "score", "native"):
        p = sub.add_parser(mode)
        p.add_argument("--work", type=Path, required=True)
        p.add_argument("--plan-sha256", required=True)
        if mode not in ("worker", "launch", "preflight-all"):
            p.add_argument("--deadline", type=float, required=True)
            p.add_argument("--index", type=int, required=True)
            p.add_argument("--stage-sha256")
        if mode == "native":
            p.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.mode == "archive-plan":
        return archive_plan_command(args)
    if args.mode == "parallel-launch":
        return parallel_pair_launch(args)
    if args.mode == "parallel-worker":
        return parallel_pair_worker(args)
    if args.mode == "follow-up-launch":
        return followup_launch(args)
    if args.mode == "follow-up-worker":
        return followup_worker(args)
    if args.mode == "plan":
        return plan_command(args)
    if args.mode == "worker":
        return worker(args)
    if args.mode == "launch":
        return launch(args)
    if args.mode == "preflight-all":
        return preflight_all(args)
    if args.mode == "native":
        return native(args)
    plan, work = checked_plan(args)
    builder.configure(work)
    if not 0 <= args.index < len(plan["candidates"]) or not time.time() < args.deadline <= plan["expires_epoch"]:
        raise ValueError("invalid case index/deadline")
    record = plan["candidates"][args.index]
    directory = work / "cases" / record["key"]
    if args.mode in ("preflight", "teacher"):
        command = [sys.executable, "-B", str(Path(__file__)), "native", "--work", str(work),
            "--plan-sha256", args.plan_sha256, "--deadline", str(args.deadline - 60.), "--index", str(args.index)]
        if args.mode == "preflight":
            command.append("--preflight-only")
        else:
            authorize_stage(plan, work, args, "teacher")
            command.extend(["--stage-sha256", args.stage_sha256])
        return run_child(command, work / f"{record['key']}_{args.mode}_native.log", args.deadline,
                         group_binding=plan.get("group_binding"))
    import_api(plan)
    if args.mode not in ("prepare", "score"):
        verify_inputs(record, directory)
    if args.mode in ("predict", "score"):
        stage = authorize_stage(plan, work, args, args.mode)
    function = {"prepare": prepare_case, "predict": predict, "score": score}[args.mode]
    kwargs = dict(seal_sha256=stage["rom_seal_sha256"]) if args.mode == "score" else {}
    grid.bounded_cpu(args.deadline, lambda: function(plan, record, directory, **kwargs))


if __name__ == "__main__":
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"owned validation process interrupted: {signum}")
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    grid.bootstrap_group_environment()
    main()
