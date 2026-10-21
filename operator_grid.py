"""Plan the BI2--10/TOTAL20--100 grid and build bounded p2 Si batches."""

from __future__ import annotations

import argparse
import ast
import csv
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import re
import signal
import shutil
import stat
import subprocess
import sys
import threading
import time
import zipfile

import build_prepared_operators as builder


PITCHES = tuple(range(2, 11))
TOTALS = tuple(range(20, 101, 10))
PAD_DIAMETER_TO_PITCH_RATIO = 0.5
EXISTING_SI = (17., 37., 38.5, 67.)
PILOT_SI = (7., 8.5, 87., 88.5)
NATIVE_HELPER_SHA = "0f2dafa88f700e365086af15697690179f22169f03a1f9e6129e62030d8b27d7"
COMMON_HELPER_SHA = "f47680d89fe820617f2548dcec31f9f9f4675f0074e4d085c7d4d47c1e346f5b"
BASELINE_BUNDLE_SHA = "7fcfcc7586f259924291f8249e18beb5f8c3c1ee8263f9d858e51cbd5682ff45"
STAGE_SECONDS = 7200
CASE_SECONDS = 10800
QUEUE_SECONDS = 57600
PREDECESSOR_WAIT_SECONDS = 10800
MAX_BATCH_CASES = 4
RESERVATION_GIB = 64
SCREEN_POOL_MAX_SLOTS = {"mfd": 4, "sysml": 2}
ANSYS_WAIT_SECONDS = 120
ARCHIVES = Path("/app/nanocad/projects/personal/zchen/YAP_STACK/Ansys_Verify/12_stack_model_validation/results")
EXTRA_SOURCES = (
    "build_exp12_m4_all_gauss_layer_dataset.py", "build_exp12_m4_direct_bubble_basis.py",
    "exp12_layer_rom_dataset.py", "exp12_port_basis_contract.py",
    "exp12_material_separable_reference.py", "exp12_periodic_rigid_gauge.py",
    "prepare_exp12_layer_geometry.py",
    "exp12_solid185_bbar.py", "audit_exp12_m4_python_bbar_reference.py",
)


def si_key(thickness):
    if type(thickness) not in (int, float) or not math.isfinite(thickness) or thickness <= 0:
        raise ValueError("Si thickness must be finite and positive")
    return "tsv_si_h" + format(thickness, "g").replace(".", "p")


def coverage_plan():
    layers = {si_key(h): h for total in TOTALS for h in (total - 13., total - 11.5)}
    rows = []
    for pitch in PITCHES:
        for total in TOTALS:
            rows.append(dict(
                bonding_pitch_um=pitch, total_thickness_um=total,
                common_cell_side_um=math.lcm(pitch, 6),
                pad_diameter_um=pitch * PAD_DIAMETER_TO_PITCH_RATIO,
                inner_si_um=total - 13., outer_si_um=total - 11.5,
                inner_removable_si_um=776.5 - total, outer_removable_si_um=775. - total,
                d2w_components=["front_bi", "beol", si_key(total - 13.), "back_bi"],
                w2w_endpoint_components=dict(inner=["front_bi", "beol", si_key(total - 13.), "back_bi"],
                                             outer=["front_bi", "beol", si_key(total - 11.5)]),
                pitch_build_status="KNOWN_P2_D1_GEOMETRY" if pitch == 2 else "BLOCKED_CELL_MESH_AND_BASIS_REVIEW",
                d2w_runtime_status="EXISTING_CANDIDATE_PROFILE" if pitch == 2 and total in (30, 50, 80)
                                    else "NOT_ADMITTED",
                w2w_runtime_status="BLOCKED_CAPS_ORIENTATION_AND_HISTORY_TRANSFER",
                thermal_geometry_validated=False, interpolation_enabled=False,
            ))
    return dict(schema="yap.operator_grid.plan.v1", pitches_um=list(PITCHES), totals_um=list(TOTALS),
                combinations=len(rows), rows=rows,
                fixed_microgeometry=dict(pad_diameter_to_pitch_ratio=PAD_DIAMETER_TO_PITCH_RATIO,
                                         beol_thickness_um=10., front_bi_thickness_um=1.5,
                                         back_bi_thickness_um=1.5, tsv_pitch_um=6., tsv_diameter_um=3.,
                                         beol_pattern_pitch_um=6., beol_cu_fraction=.35),
                p2_si_layers={key: dict(thickness_um=h,
                    status="EXISTING_SOURCE_REQUIRES_FRESH_CHECK" if h in EXISTING_SI else "NOT_BUILT",
                    pilot_selected=h in PILOT_SI) for key, h in layers.items()},
                initial_wafer_total_um=775., outer_back_bi_deposition=False,
                physical_readiness=False, native_catalog_modified=False,
                scope="Requested coverage, not a claim of installed or validated runtime support")


def regular(path, *, max_bytes=None):
    original = Path(path).expanduser().absolute()
    path = builder.safe_path(original)
    for parent in (original, *original.parents):
        if parent.is_symlink():
            raise ValueError(f"symlink input is not allowed: {original}")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or (max_bytes is not None and info.st_size > max_bytes):
        raise ValueError(f"invalid regular input: {path}")
    return path


def read_record(path, expected=None):
    path = regular(path, max_bytes=2 * 1024**2)
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if expected is not None and observed != expected:
        raise ValueError(f"receipt SHA256 mismatch: {path}")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate receipt key")
            value[key] = item
        return value
    value = json.loads(raw, object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite receipt")))
    if not isinstance(value, dict):
        raise ValueError("receipt must be an object")
    return value, observed


def verify_pins(path, expected):
    pins, _ = read_record(path, expected)
    files = pins.get("files_sha256")
    if pins.get("schema") != "yap.operator_grid.sources.v1" or not isinstance(files, dict) or not 1 <= len(files) <= 96:
        raise ValueError("invalid bounded source closure")
    for filename, digest in files.items():
        if not isinstance(digest, str) or len(digest) != 64 or builder.digest(regular(filename)) != digest:
            raise ValueError(f"producer/dependency drift: {filename}")
    return pins


def authenticated_common(pins):
    from importlib.machinery import PathFinder
    value = pins.get("common_utils")
    if not isinstance(value, str) or value not in pins["files_sha256"]:
        raise ValueError("common solver helper must be authenticated before import")
    if pins["files_sha256"][value] != COMMON_HELPER_SHA:
        raise ValueError("only the reviewed process-group-cleanup solver helper is allowed")
    path = regular(value)
    package = PathFinder.find_spec("common", [str(path.parent.parent)])
    if package is None or not package.submodule_search_locations:
        raise ValueError("missing authenticated common package")
    locations = tuple(Path(p).resolve() for p in package.submodule_search_locations)
    if locations != (path.parent,):
        raise ValueError("common package resolves outside the authenticated closure")
    if package.origin is not None and str(regular(package.origin)) not in pins["files_sha256"]:
        raise ValueError("common package initializer is not authenticated")
    helper = PathFinder.find_spec("common.ansys_utils", list(map(str, locations)))
    if helper is None or helper.origin is None or regular(helper.origin) != path:
        raise ValueError("solver helper resolves outside the authenticated closure")
    if "common" in sys.modules or "common.ansys_utils" in sys.modules:
        raise ValueError("native stage requires a fresh common-module namespace")
    sys.path.insert(0, str(path.parent.parent))
    from common import ansys_utils
    if regular(ansys_utils.__file__) != path:
        raise ValueError("loaded solver helper origin mismatch")
    return ansys_utils


def native_stage(args):
    # Authentication precedes imports from the external scientific producer tree.
    pins = verify_pins(args.pins, args.pins_sha256)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if len(command) < 3 or command[1] != "-B":
        raise ValueError("expected an explicit Python -B producer command")
    script = regular(command[2])
    if str(script) not in pins["files_sha256"] or script.name not in {
        "run_exp12_m4_common_port_stiffness.py", "run_exp12_m4_mindlin_macro_augmentation.py",
        "extract_exp12_m4_volume_displacement_snapshots.py", "build_exp12_m4_static_lift_fields.py",
        "build_exp12_m4_mindlin_q6_layer_operator.py",
    }:
        raise ValueError("native stage is not authenticated")
    if not math.isfinite(args.case_deadline):
        raise ValueError("native stage requires a finite case deadline")
    deadline = min(time.time() + STAGE_SECONDS, args.case_deadline - 60.)
    if deadline <= time.time():
        raise TimeoutError("case deadline exhausted before native stage")
    group_binding = None
    if getattr(args, "campaign_file", None) is not None:
        campaign, _ = read_record(args.campaign_file, args.campaign_sha256)
        if campaign.get("source_pins_sha256") != args.pins_sha256 or not campaign.get("group_binding"):
            raise ValueError("native parallel authority is not bound to its source pins")
        group_binding = campaign["group_binding"]
        assert_group_worker(group_binding, Path(campaign["work"]))
    sys.path[:0] = [str(script.parent), str(script.parent.parent)]
    ansys_utils = authenticated_common(pins)
    original = ansys_utils.run_mapdl
    def bounded_mapdl(**values):
        wait_ansys_idle(deadline, **({"group_binding": group_binding} if group_binding else {}))
        remaining = deadline - time.time() - 90.
        if remaining <= 0:
            raise TimeoutError("native MAPDL deadline exhausted")
        requested = values.get("timeout_sec")
        values["timeout_sec"] = min(requested, remaining) if requested is not None and requested > 0 else remaining
        return original(**values)
    def timeout(_signum, _frame):
        raise TimeoutError("native stage exceeded its recorded wall deadline")
    ansys_utils.run_mapdl = bounded_mapdl
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
    print(json.dumps(dict(native_stage=str(script), deadline_epoch=deadline,
                          effective_mapdl_timeout="remaining recorded stage budget")), flush=True)
    try:
        sys.argv = command[2:]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def bounded_cpu(deadline, function):
    remaining = deadline - time.time()
    if not math.isfinite(remaining) or remaining <= 0:
        raise TimeoutError("CPU audit/dataset/publication deadline exhausted")
    started = time.monotonic()
    previous_remaining, previous_interval = signal.getitimer(signal.ITIMER_REAL)
    if previous_remaining > 0:
        remaining = min(remaining, previous_remaining)
    def timeout(_signum, _frame):
        raise TimeoutError("CPU audit/dataset/publication exceeded its wall deadline")
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        return function()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if previous_remaining > 0:
            restored = previous_remaining - (time.monotonic() - started)
            if restored <= 0:
                raise TimeoutError("enclosing CPU stage exceeded its wall deadline")
            signal.setitimer(signal.ITIMER_REAL, restored, previous_interval)
        if time.time() >= deadline:
            raise TimeoutError("CPU audit/dataset/publication exceeded its wall deadline")


def checked_baseline(root):
    root = builder.safe_path(root)
    inventory, sha = read_record(root / "runtime/BUNDLE.json", BASELINE_BUNDLE_SHA)
    files = inventory.get("files_sha256", {})
    if not isinstance(files, dict) or len(files) > 512:
        raise ValueError("invalid baseline inventory")
    from package_assembly_runtime import confined_source
    for filename, expected in files.items():
        path = confined_source(root, root.parent / filename)
        if builder.digest(regular(path)) != expected:
            raise ValueError(f"baseline changed: {filename}")
    return dict(path=str(root), bundle_sha256=sha, files_verified=len(files),
                delivered_native64_si_um=[17., 67.], native_execution_or_accuracy_inferred=False)


def checked_existing_si(path):
    parent, parent_sha = read_record(path, builder.LEGACY_STAGE_PINS["parent"])
    if parent.get("rank") != 64 or parent.get("port_basis_sha256") != builder.BASIS_SHA:
        raise ValueError("existing Si endpoints require the native64 parent")
    from exp12_layer_rom_dataset import LayerROMDataset
    entries = {}
    for height in (37., 38.5):
        key = si_key(height)
        source = regular(parent["datasets"][key])
        expected = parent["dataset_sha256"][key]
        if builder.digest(source) != expected:
            raise ValueError("existing Si endpoint changed")
        dataset = LayerROMDataset.load(source, validate=True, verify_content=True)
        if (dataset.layer_kind != "tsv_si" or dataset.macro_size != 134
                or abs(dataset.geometry_signature["thickness_mm"] * 1000. - height) > 1e-9
                or dataset.metadata.get("port_basis_sha256") != builder.BASIS_SHA):
            raise ValueError("existing Si endpoint geometry/basis mismatch")
        entries[key] = dict(path=str(source), sha256=expected, thickness_um=height,
                            material_contract=material_contract(dataset),
                            status="VERIFIED_EXTERNAL_COMPONENT_NOT_DELIVERED")
        del dataset
    if entries[si_key(37.)]["material_contract"] != entries[si_key(38.5)]["material_contract"]:
        raise ValueError("existing parent Si endpoints must share Cu and Si definitions")
    return dict(parent_package=str(path), parent_package_sha256=parent_sha, endpoints=entries)


def material_contract(dataset):
    if (dataset.reference_temperature_c != 25. or dataset.linear_reference_mode != "material_separable_bbar"
            or not dataset.reference_constant_materials or "si" not in dataset.reference_constant_materials):
        raise ValueError("target requires explicit material-separable room-temperature Si data")
    result = dict(cu_card=asdict(dataset.cu_card), si=asdict(dataset.reference_constant_materials["si"]))
    return json.loads(json.dumps(result, allow_nan=False, sort_keys=True))


def check_material_contract(dataset, expected):
    observed = material_contract(dataset)
    if observed != expected:
        raise ValueError("target Cu constitutive card or Si reference material differs from the pinned parent")
    return observed


def source_closure(api, cases):
    sources = {Path(__file__), Path(builder.__file__), Path(api.__file__)}
    supervisor = Path(__file__).parent / "operator_thermal_validation.py"
    if supervisor.is_file():
        sources.add(supervisor)
    if builder.digest(regular(api.__file__)) != NATIVE_HELPER_SHA:
        raise ValueError("unreviewed native geometry helper")
    for case in cases:
        sources.add(case.template)
        sources.update(Path(path) for path in builder.read_json(case.template)["inputs"] if str(path).endswith(".py"))
    sources.update(api.BASE / filename for filename in EXTRA_SOURCES)
    common = api.BASE.parent / "common/ansys_utils.py"
    if builder.digest(regular(common)) != COMMON_HELPER_SHA:
        raise ValueError("native solver helper lacks the reviewed interruption cleanup")
    sources.add(common)
    initializer = common.parent / "__init__.py"
    if initializer.is_file():
        sources.add(initializer)
    pending, inspected, started = list(sources), set(), time.monotonic()
    # Follow explicit local imports, not directory trees or shared-mount indexes.
    while pending:
        if len(sources) > 96 or time.monotonic() - started > 30:
            raise ValueError("source closure exceeds finite file/time bound")
        path = regular(pending.pop(), max_bytes=2 * 1024**2)
        if path in inspected or path.suffix != ".py":
            continue
        inspected.add(path)
        syntax = ast.parse(path.read_text())
        modules = set()
        for node in ast.walk(syntax):
            if isinstance(node, ast.Import):
                modules.update(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                modules.add(node.module)
        for module in modules:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9.]*", module):
                raise ValueError("invalid local import declaration")
            candidate = (api.BASE.parent if module.startswith("common.") else api.BASE) / (module.replace(".", "/") + ".py")
            if candidate.is_file() and candidate not in sources:
                sources.add(candidate)
                pending.append(candidate)
    return dict(schema="yap.operator_grid.sources.v1",
                common_utils=str(regular(common)),
                files_sha256={str(regular(path)): builder.digest(path) for path in sorted(sources)})


def audit_saved_bubble_maps(dataset, final_path, manifest):
    import numpy as np
    from audit_exp12_m4_python_bbar_reference import material_map
    from exp12_material_separable_reference import structured_node_id
    from exp12_solid185_bbar import bbar_points
    with np.load(final_path, allow_pickle=False) as fields:
        nodes, coordinates, modes = fields["node_ids"], fields["coordinates_mm"], fields["bubble_modes"]
    nx, ny, nz = (manifest["mesh"][axis] for axis in ("nx", "ny", "nz"))
    materials = np.asarray(material_map("tsv_si", nx=nx, nz=nz,
        coupon_x_um=1000 * np.ptp(coordinates[:, 0]), coupon_z_um=1000 * np.ptp(coordinates[:, 2]),
        geometry=manifest["geometry"]), dtype=int)
    if materials.shape == (nx, nz):
        materials = np.repeat(materials[:, None, :], ny, axis=1)
    elements = np.flatnonzero(materials.transpose(1, 0, 2).ravel() == 4)
    if not len(elements):
        raise ValueError("saved-map audit requires actual Cu elements")
    lookup = {int(node): index for index, node in enumerate(nodes)}
    maximum_error, position_error, weight_error = 0., 0., 0.
    for start in range(0, len(elements), 128):
        selected = elements[start:start + 128]
        connectivity = []
        for element in selected:
            j, remainder = divmod(int(element), nx * nz)
            i, k = divmod(remainder, nz)
            node_positions = ((i,j,k), (i+1,j,k), (i+1,j,k+1), (i,j,k+1),
                              (i,j+1,k), (i+1,j+1,k), (i+1,j+1,k+1), (i,j+1,k+1))
            connectivity.append([lookup[structured_node_id(a,b,c,ny,nz)] for a,b,c in node_positions])
        indices = np.asarray(connectivity)
        bricks = coordinates[indices]
        rule = bbar_points(bricks[0])
        edge_error = np.max(np.abs(bricks - bricks[:, :1] - (bricks[0] - bricks[0, 0])), initial=0.)
        if edge_error > 1e-11:
            raise ValueError("saved-map audit requires congruent structured bricks")
        gauss = len(rule)
        expected_ids = tuple(f"element_{element:08d}_gauss_{point:02d}" for element in selected for point in range(gauss))
        first, stop = start * gauss, (start + len(selected)) * gauss
        if dataset.cu_point_ids[first:stop] != expected_ids:
            raise ValueError("saved Cu point identities do not cover every actual Cu element")
        strain = np.einsum("gsp,epq->egsq", np.stack([p.b_matrix for p in rule]),
            modes[indices].reshape(len(indices), 24, modes.shape[2]), optimize=True).reshape(-1, 6, modes.shape[2])
        saved = dataset.bubble_strain_maps[first:stop]
        if not np.isfinite(saved).all():
            raise ValueError("saved Cu strain maps must be finite")
        error = np.max(np.abs(saved - strain), initial=0.) / max(1., np.max(np.abs(strain), initial=0.))
        maximum_error = max(maximum_error, float(error))
        positions = (bricks[:, :1] + np.stack([p.position_mm - bricks[0,0] for p in rule])[None]).reshape(-1, 3)
        position_error = max(position_error, float(np.max(np.abs(dataset.cu_positions_mm[first:stop] - positions), initial=0.)))
        weights = np.tile([p.integration_volume_mm3 for p in rule], len(indices))
        weight_error = max(weight_error, float(np.max(np.abs(dataset.cu_weights_mm3[first:stop] - weights), initial=0.)))
    if len(elements) * 8 != len(dataset.cu_point_ids) or maximum_error > 1e-10 or position_error > 1e-11 or weight_error > 1e-18:
        raise ValueError("saved Cu maps/positions/weights differ from the independently finalized bubble space")
    return dict(method="direct_all_cu_element_bbar_reconstruction_from_independently_finalized_modes",
                cu_elements=len(elements), cu_points=len(dataset.cu_point_ids),
                maximum_relative_strain_map_error=maximum_error,
                maximum_position_error_mm=position_error, maximum_weight_error_mm3=weight_error)


def publish_layer(api, case, archive, pins, pins_sha, expected_material, *, audit_output=None):
    verify_pins(pins, pins_sha)
    import numpy as np
    from exp12_layer_rom_dataset import LayerROMDataset
    from exp12_port_basis_contract import validate_dataset_port_basis_contracts
    output = api.ROOT / case.key
    plan = builder.read_json(output / "STATIC_PLAN.json")
    api.verify_inputs(plan)
    if builder.read_json(output / "RUN_STATUS.json")["state"] != "COMPLETE_ENGINEERING_CANDIDATE":
        raise ValueError("only completed target operators can be audited")
    assets = {}
    for step in plan["steps"]:
        assets.update(api.verify_static_stage(case, step))
    audit_path = output / "bubble/AUDITED_FINAL_BUBBLE_SPACE.npz" if audit_output is None else builder.safe_path(audit_output)
    if audit_output is not None and audit_path.is_relative_to(output):
        raise ValueError("isolated audit output must not modify its source case")
    audit = builder.transfer_audit(api, case, audit_output=audit_path)
    ready = builder.read_json(output / "DATASET_READY.json")
    dataset_path, prepared_path = regular(ready["dataset_file"]), regular(ready["prepared_geometry_file"])
    if not all(path.is_relative_to(output) for path in (dataset_path, prepared_path)):
        raise ValueError("target operator escapes its case")
    if builder.digest(dataset_path) != ready["dataset_sha256"] or builder.digest(prepared_path) != ready["prepared_geometry_sha256"]:
        raise ValueError("target dataset/prepared geometry changed")
    dataset = LayerROMDataset.load(dataset_path, validate=True, verify_content=True)
    materials = check_material_contract(dataset, expected_material)
    gauge = validate_dataset_port_basis_contracts({case.library_key: dataset}, api.BASIS[case.family])
    weights = np.asarray(dataset.cu_weights_mm3)
    summary = builder.read_json(dataset_path.parent / "M4_ALL_GAUSS_LAYER_ROM_DATASET_SUMMARY.json")
    if (dataset.layer_kind != "tsv_si" or dataset.macro_size != 6 + 2 * case.rank
            or abs(dataset.geometry_signature["thickness_mm"] * 1000. - case.substrate_um) > 1e-9
            or dataset.bubble_size != audit["rank"] or weights.ndim != 1 or len(weights) <= 0
            or not np.isfinite(weights).all() or np.any(weights <= 0.)
            or len(weights) != summary["cu_gauss_point_count"]
            or not np.isclose(weights.sum(), summary["cu_volume_mm3"], rtol=1e-10, atol=0.)):
        raise ValueError("target material volume/points/rank/geometry failed")
    with np.load(dataset_path, allow_pickle=False) as raw:
        content = str(raw["content_sha256"].item())
    with np.load(prepared_path, allow_pickle=False) as raw:
        prepared_content = json.loads(str(raw["metadata_json"].item()))["source_content_sha256"]
    if content != ready["dataset_content_sha256"] or prepared_content != content:
        raise ValueError("target dataset content binding failed")
    bubble_path = output / "bubble/M4_DIRECT_ZERO_TRACE_BUBBLE_BASIS.npz"
    if (dataset.metadata.get("thermal_geometry_validated") is not False
            or dataset.metadata.get("geometry_transfer", {}).get("transferred_bubble_sha256") != audit["raw_transferred_bubble_sha256"]
            or dataset.metadata.get("input_sha256", {}).get(str(bubble_path)) != audit["raw_transferred_bubble_sha256"]
            or summary["dynamic_bubble_rank"] != audit["raw_transferred_rank"]
            or summary["inplane_rotation_closure_rank"] != audit["closure_added_rank"]):
        raise ValueError("saved dataset does not bind the audited approximation space")
    if not np.isclose(weights.sum(), .006**2 * case.substrate_um / 1000. * math.pi / 16., rtol=.02, atol=0.):
        raise ValueError("actual target Cu volume is inconsistent with TSV6/d3 geometry")
    saved_map_audit = audit_saved_bubble_maps(dataset, audit_path,
        builder.read_json(output / "static/tangent/M4_COMMON_PORT_TANGENT_MANIFEST.json"))
    count = len(weights)
    del dataset
    target = archive / case.key
    target.mkdir()
    inventory = {}
    def copy(source, relative, expected=None):
        inventory[relative] = builder.copy_checked(source, target / relative, expected)
    copy(dataset_path, f"datasets/{case.library_key}.npz", ready["dataset_sha256"])
    copy(prepared_path, f"prepared/{case.library_key}.npz", ready["prepared_geometry_sha256"])
    copy(api.BASIS[case.family], "shared/port_basis.npz", builder.BASIS_SHA)
    explicit = ["STATIC_PLAN.json", "TEMPLATE_PRODUCER_VERSIONS.json", "LAUNCH.json", "RUN_STATUS.json",
                "DATASET_READY.json", "DATASET_BUILD_INPUTS.json", "dataset.log", "prepare.log",
                "bubble/TRANSFER_MANIFEST.json", "bubble/M4_DIRECT_ZERO_TRACE_BUBBLE_BASIS.npz",
                "all_gauss/M4_ALL_GAUSS_LAYER_ROM_DATASET_SUMMARY.json"]
    if audit_output is None:
        explicit.append("bubble/AUDITED_FINAL_BUBBLE_SPACE.npz")
    if (output / "ADOPTED_INPUTS.json").is_file():
        explicit += ["ADOPTED_INPUTS.json", "static/tangent_ADOPTION.json", "static/q6_ADOPTION.json"]
    explicit += [f"static/{name}_FROZEN_OUTPUTS.json" for name in builder.STAGES]
    explicit += [f"{name}.log" for name in builder.STAGES]
    for relative in explicit:
        source = regular(output / relative)
        assets[str(source)] = builder.digest(source)
    if len(assets) > 128:
        raise ValueError("layer publication exceeds finite closure")
    for source, expected in sorted(assets.items()):
        source = regular(source)
        if not source.is_relative_to(output):
            raise ValueError("static output escapes its case")
        relative = source.relative_to(output)
        copy(source, f"build_receipts/{case.key}/{relative}", expected)
    if audit_output is not None:
        copy(regular(audit_path), f"build_receipts/{case.key}/bubble/AUDITED_FINAL_BUBBLE_SPACE.npz",
             audit["transferred_bubble_sha256"])
    receipt = dict(schema="yap.operator_grid.layer_candidate.v1", state="PASS_STRUCTURAL_CANDIDATE",
                   library_key=case.library_key, thickness_um=case.substrate_um,
                   layer="tsv_si", bonding_pitch_scope_um=2., tsv_pitch_um=6., tsv_diameter_um=3.,
                   common_cell_side_um=6., port_rank_per_face=case.rank, port_basis_sha256=builder.BASIS_SHA,
                   dataset=f"../datasets/{case.library_key}.npz", dataset_sha256=ready["dataset_sha256"],
                   dataset_content_sha256=content, prepared_geometry=f"../prepared/{case.library_key}.npz",
                   prepared_geometry_sha256=ready["prepared_geometry_sha256"], cu_point_count=count,
                   cu_volume_mm3=float(weights.sum()), all_cu_points_retained=True,
                   transfer_audit=audit, saved_bubble_map_audit=saved_map_audit, gauge_contract=gauge,
                   material_contract=materials,
                   thermal_geometry_validated=False, w2w_runtime_enabled=False, d2w_runtime_enabled=False)
    builder.write_json(target / "audit/LAYER.json", receipt, immutable=True)
    inventory["audit/LAYER.json"] = builder.digest(target / "audit/LAYER.json")
    api.verify_inputs(plan)
    for step in plan["steps"]:
        api.verify_static_stage(case, step)
    verify_pins(pins, pins_sha)
    archive_record = dict(schema="yap.operator_grid.layer_archive.v1", files_sha256=inventory,
                          source_work=str(output), solver_scratch_included=False,
                          thermal_geometry_validated=False, runtime_admitted=False)
    builder.write_json(target / "ARCHIVE.json", archive_record, immutable=True)
    return dict(archive=str(target), archive_sha256=builder.digest(target / "ARCHIVE.json"),
                audit_sha256=inventory["audit/LAYER.json"], thickness_um=case.substrate_um,
                dataset_sha256=ready["dataset_sha256"], state="ARCHIVED_STRUCTURAL_CANDIDATE")


def file_identity(path):
    info = regular(path).stat()
    return dict(device=info.st_dev, inode=info.st_ino, bytes=info.st_size,
                mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns)


def validate_adoption(record, work, height):
    if (not isinstance(record, dict) or record.get("schema") != "yap.operator_grid.adoption.v1"
            or record.get("case_key") != "p2_" + si_key(height)
            or record.get("adopted_stages") != list(builder.STAGES[:2])
            or record.get("destination_work") != str(work)):
        raise ValueError("adoption must bind exactly the first case's tangent/q6 prefix")
    old = local_work(record["source_work"])
    source_case = old / "results" / record["case_key"]
    if old.is_relative_to(work) or work.is_relative_to(old):
        raise ValueError("adoption source and writable work must not overlap")
    receipts = record.get("receipts_sha256", {})
    assets = record.get("stage_assets_sha256", {})
    raw = record.get("raw_files", {})
    if not 1 <= len(receipts) <= 16 or set(assets) != set(builder.STAGES[:2]) or not 4 <= len(raw) <= 6:
        raise ValueError("adoption inventory is missing or exceeds its finite scope")
    for path in receipts:
        if not builder.safe_path(path).is_relative_to(old):
            raise ValueError("adoption receipt escapes its original campaign")
    for stage, files in assets.items():
        if len(files) != 4 or any(not builder.safe_path(p).is_relative_to(source_case) for p in files):
            raise ValueError("adopted stage assets escape their case or are incomplete")
        expected_dir = work / "adopted_inputs" / record["case_key"] / stage
        if record["raw_copy_dirs"].get(stage) != str(expected_dir):
            raise ValueError("private raw copies must use the bound new-work directory")
    for path, value in raw.items():
        stage = value["stage"]
        source_dir = old / "solver" / record["case_key"] / stage / "tsv_si"
        source = builder.safe_path(path)
        if stage not in assets or source.parent != source_dir or source.suffix not in {".rdb", ".rst", ".r001"}:
            raise ValueError("raw adoption input escapes its exact source job")
        if not isinstance(value.get("identity"), dict) or value["identity"].get("bytes", 0) <= 0:
            raise ValueError("raw adoption input requires a nonempty stable file identity")
    if sum(v["identity"]["bytes"] for v in raw.values()) > 24 * 1024**3:
        raise ValueError("adopted raw files exceed the reviewed copy reservation")


def verify_adoption(record):
    for path, expected in record["receipts_sha256"].items():
        if builder.digest(regular(path)) != expected:
            raise ValueError("original adoption receipt changed")
    for path, value in record["raw_files"].items():
        if file_identity(path) != value["identity"]:
            raise ValueError("original raw result changed since adoption review")


def require_exited_source(worker_pid):
    if process_identity(worker_pid) is not None:
        raise RuntimeError("source campaign worker is still alive; do not adopt it")
    if any((row["pgid"] == worker_pid or row["sid"] == worker_pid)
           and not row.get("state", "").startswith("Z") for row in own_process_census()):
        raise RuntimeError("source campaign still has live session/group members")


def bind_adoption(source_work, api, case, work):
    old = local_work(source_work)
    source_case = old / "results" / case.key
    campaign, campaign_sha = read_record(old / "CAMPAIGN.json")
    launch, launch_sha = read_record(old / "LAUNCH.json")
    state, state_sha = read_record(old / "STATUS.json")
    status, status_sha = read_record(source_case / "RUN_STATUS.json")
    static, static_sha = read_record(source_case / "STATIC_PLAN.json")
    if (campaign.get("work") != str(old) or campaign.get("native_source") != str(api.BASE)
            or campaign.get("resources", {}).get("host") != case.host
            or state.get("state") != "STOP_REQUIRES_REVIEW" or state.get("active_case") != case.key
            or status.get("state") != "STOP_REQUIRES_REVIEW" or status.get("stage") != "extract_tangent"
            or [s["stage"] for s in status.get("completed", [])] != list(builder.STAGES[:2])
            or static.get("case") != asdict(case)
            or launch.get("campaign_sha256") != campaign_sha
            or state.get("worker_pid") != launch.get("worker_pid")):
        raise ValueError("only a reviewed stopped two-stage target case can be adopted")
    require_exited_source(launch["worker_pid"])
    verify_pins(old / "SOURCE_PINS.json", campaign["source_pins_sha256"])
    original = builder.load_api(api.BASE, old)
    original.verify_inputs(static)
    receipts = {str(old / "CAMPAIGN.json"): campaign_sha, str(old / "LAUNCH.json"): launch_sha,
                str(old / "STATUS.json"): state_sha, str(source_case / "RUN_STATUS.json"): status_sha,
                str(source_case / "STATIC_PLAN.json"): static_sha,
                str(old / "SOURCE_PINS.json"): campaign["source_pins_sha256"]}
    assets, raw, raw_dirs = {}, {}, {}
    for stage, extraction in zip(builder.STAGES[:2], ("extract_tangent", "extract_q6")):
        step = next(s for s in static["steps"] if s["name"] == stage)
        assets[stage] = original.verify_static_stage(case, step)
        seal = source_case / "static" / f"{stage}_FROZEN_OUTPUTS.json"
        receipts[str(seal)] = builder.digest(regular(seal))
        log = regular(source_case / f"{stage}.log", max_bytes=2 * 1024**2)
        assets[stage][str(log)] = builder.digest(log)
        command = next(s for s in static["steps"] if s["name"] == extraction)["command"]
        directory = builder.safe_path(original.option(command, "--source-dir"))
        job = original.option(command, "--source-job")
        if directory != old / "solver" / case.key / stage / "tsv_si" or job != {
                "tangent": "e12m4k_tsv_si", "q6": "e12m4ma_tsv_si"}[stage]:
            raise ValueError("source extraction commands do not bind the expected raw jobs")
        raw_dirs[stage] = str(work / "adopted_inputs" / case.key / stage)
        for extension in ("rdb", "rst", "r001"):
            path = directory / f"{job}.{extension}"
            if extension == "r001" and not path.exists():
                continue
            raw[str(path)] = dict(stage=stage, identity=file_identity(path))
    record = dict(schema="yap.operator_grid.adoption.v1", source_work=str(old), destination_work=str(work),
                  case_key=case.key, adopted_stages=list(builder.STAGES[:2]), receipts_sha256=receipts,
                  stage_assets_sha256=assets, raw_files=raw, raw_copy_dirs=raw_dirs,
                  old_stop_preserved=True, main_solves_reexecuted=False)
    validate_adoption(record, work, case.substrate_um)
    verify_adoption(record)
    return record


def adoption_callback(api, case, work, record):
    prepared = False
    output = api.ROOT / case.key
    def adopt(step):
        nonlocal prepared
        verify_adoption(record)
        if not prepared:
            inventory = {}
            for source, value in record["raw_files"].items():
                destination = Path(record["raw_copy_dirs"][value["stage"]]) / Path(source).name
                print(f"ADOPT raw input: {source} -> {destination}", flush=True)
                digest = builder.copy_checked(regular(source), destination)
                if builder.digest(regular(source)) != digest or file_identity(source) != value["identity"]:
                    raise ValueError("original raw result changed during checked copying")
                destination.chmod(0o444)
                inventory[source] = dict(copy=str(destination), sha256=digest,
                                         original_identity=value["identity"], copy_identity=file_identity(destination))
            builder.write_json(output / "ADOPTED_INPUTS.json", dict(binding=record, raw_inputs=inventory), immutable=True)
            (output / "ADOPTED_INPUTS.json").chmod(0o444)
            prepared = True
        stage = step["name"]
        if stage not in builder.STAGES[:2]:
            raise ValueError("adoption cannot skip an unreviewed stage")
        source_case = Path(record["source_work"]) / "results" / case.key
        for source, expected in record["stage_assets_sha256"][stage].items():
            relative = regular(source).relative_to(source_case)
            builder.copy_checked(source, output / relative, expected)
        builder.write_json(output / "static" / f"{stage}_ADOPTION.json",
            dict(state="ADOPTED_SEALED_STAGE", stage=stage, original_assets_sha256=record["stage_assets_sha256"][stage],
                 original_receipts_sha256=record["receipts_sha256"], raw_binding="../ADOPTED_INPUTS.json",
                 main_solve_reexecuted=False), immutable=True)
        print(f"ADOPT complete: {stage}; main MAPDL solve skipped", flush=True)
    return adopt


def verify_private_inputs(output, record, *, hashes=True):
    verify_adoption(record)
    inventory, _ = read_record(output / "ADOPTED_INPUTS.json")
    if inventory.get("binding") != record or set(inventory["raw_inputs"]) != set(record["raw_files"]):
        raise ValueError("private raw adoption inventory does not match its campaign")
    for source, value in inventory["raw_inputs"].items():
        expected_path = Path(record["raw_copy_dirs"][record["raw_files"][source]["stage"]]) / Path(source).name
        if value["copy"] != str(expected_path):
            raise ValueError("private raw copy path differs from its authenticated campaign")
        if file_identity(value["copy"]) != value["copy_identity"]:
            raise ValueError("private raw result changed during extraction")
        if hashes and (builder.digest(regular(source)) != value["sha256"]
                       or builder.digest(regular(value["copy"])) != value["sha256"]):
            raise ValueError("original raw result changed during recovery execution")


def validate_campaign(plan, work, args):
    host = builder.physical_host()
    if (plan.get("schema") != "yap.operator_grid.campaign.v4"
            or plan.get("execution_host") != host
            or plan.get("local_scratch_root") != str(builder.scratch_root(host))
            or plan.get("resources", {}).get("host") != host
            or str(work) != plan.get("work") or str(builder.safe_path(args.native_source)) != plan.get("native_source")
            or plan.get("source_pins_sha256") != args.pins_sha256
            or plan.get("stage_wall_seconds") != STAGE_SECONDS or plan.get("case_wall_seconds") != CASE_SECONDS
            or plan.get("per_case_local_reservation_gib") != RESERVATION_GIB
            or plan.get("predecessor_wait_seconds") != PREDECESSOR_WAIT_SECONDS
            or plan.get("host_serialization") is not True
            or plan.get("automatic_retry") is not False or plan.get("retained_solver_scratch") is not True
            or plan.get("interpolation_enabled") is not False):
        raise ValueError("worker arguments or policy differ from the authenticated campaign")
    validate_batch(plan.get("si_thicknesses_um"))
    if plan.get("ansys_wait_seconds") != ANSYS_WAIT_SECONDS or "adoption" not in plan:
        raise ValueError("campaign must bind the reviewed ANSYS wait and adoption policy")
    if plan["adoption"] is not None:
        validate_adoption(plan["adoption"], work, plan["si_thicknesses_um"][0])
    builder.local_scratch(work, host=host)
    if plan.get("group_binding") is not None:
        group, _, _ = authenticate_group(plan["group_binding"])
        if group["jobs"][plan["group_binding"]["slot"]]["work"] != str(work):
            raise ValueError("campaign group membership mismatch")
    if plan.get("predecessor") is not None:
        validate_predecessor_binding(plan["predecessor"])
    created, expires = plan.get("created_epoch"), plan.get("expires_epoch")
    if (type(created) not in (int, float) or type(expires) not in (int, float)
            or not math.isfinite(created) or not math.isfinite(expires)
            or abs(expires - created - QUEUE_SECONDS) > .01 or time.time() >= expires):
        raise ValueError("campaign expiry must retain the reviewed finite queue budget")


def validate_batch(values):
    allowed = {h for total in TOTALS for h in (total - 13., total - 11.5)} - set(EXISTING_SI)
    if (not isinstance(values, (tuple, list)) or not 1 <= len(values) <= MAX_BATCH_CASES
            or any(type(h) not in (int, float) or not math.isfinite(h) or h not in allowed for h in values)
            or list(values) != sorted(set(values))):
        raise ValueError("batch requires 1--4 unique ascending missing p2 Si endpoints")
    return list(values)


def process_identity(pid, *, include_parent=False, expected_start_ticks=None):
    if type(pid) is not int or pid <= 0:
        raise ValueError("invalid process identity PID")
    root = Path(f"/proc/{pid}")
    try:
        if root.stat().st_uid != os.getuid():
            if expected_start_ticks is not None:
                return None
            raise ValueError("predecessor process must belong to this UID")
        with (root / "stat").open() as stream:
            raw = stream.read(16385)
        fields = raw[raw.rindex(")") + 2:].split()
        if expected_start_ticks is not None and len(fields) >= 20 and fields[19] != expected_start_ticks:
            return None
        with (root / "cmdline").open("rb") as stream:
            command = stream.read(65537)
    except (FileNotFoundError, ProcessLookupError):
        return None
    if len(raw) > 16384 or len(command) > 65536:
        raise ValueError("process identity exceeds bounded input")
    if len(fields) < 20 or not all(fields[i].isdigit() for i in (2, 3, 19)):
        raise ValueError("invalid process identity stat")
    if fields[0] == "Z":
        return None
    identity = dict(pid=pid, start_ticks=fields[19], pgid=int(fields[2]), sid=int(fields[3]),
                    command=[part.decode() for part in command.split(bytes([0])) if part])
    if include_parent:
        identity["ppid"] = int(fields[1])
    return identity


def observed_process_identity(previous, *, include_parent=False):
    """Look up a recorded birth; an unrelated reuse is no longer our process."""
    try:
        current = (process_identity(previous["pid"], include_parent=True)
                   if include_parent else process_identity(previous["pid"]))
    except ValueError as error:
        if str(error) == "predecessor process must belong to this UID":
            return None
        raise
    if current is None or current["start_ticks"] != previous["start_ticks"]:
        return None
    return current


def proven_descendants(anchor):
    """Discover only through a live anchor and recheck the entire parent chain."""
    def same_birth(a, b):
        return a is not None and b is not None and (a["pid"], a["start_ticks"]) == (b["pid"], b["start_ticks"])
    current = observed_process_identity(anchor, include_parent=True)
    if not same_birth(current, anchor):
        return []
    rows = {row["pid"]: row for row in own_process_census() if not row["state"].startswith("Z")}
    selected = {anchor["pid"]}
    while True:
        expanded = selected | {pid for pid, row in rows.items() if row["ppid"] in selected}
        if len(expanded) > 256:
            raise RuntimeError("live owned process inventory exceeds its finite bound")
        if expanded == selected:
            break
        selected = expanded
    identities = {}
    for pid in selected:
        try:
            identity = process_identity(pid, include_parent=True)
        except ValueError as error:
            if str(error) != "predecessor process must belong to this UID":
                raise
            return []
        row = rows.get(pid)
        if identity is None or row is None or any(identity[key] != row[key] for key in ("ppid", "pgid", "sid")):
            return []
        identities[pid] = identity
    if not same_birth(identities.get(anchor["pid"]), anchor):
        return []
    for pid, previous in identities.items():
        current = observed_process_identity(previous, include_parent=True)
        if not same_birth(current, previous) or current["ppid"] != previous["ppid"]:
            return []
        if pid != anchor["pid"] and previous["ppid"] not in identities:
            return []
    if not same_birth(observed_process_identity(anchor), anchor):
        return []
    return [{key: value for key, value in identity.items() if key != "ppid"} for identity in identities.values()]


def wait_exec_identity(child, command, observed, *, deadline=None):
    now = time.monotonic()
    end = now + min(5., max(0., deadline - time.time())) if deadline is not None else now + 5.
    lifetime, previous = None, None
    while time.monotonic() < end:
        if child.poll() is not None:
            raise RuntimeError("new worker exited before launch publication")
        identity = process_identity(child.pid)
        if identity is not None:
            if identity["pid"] != child.pid:
                raise ValueError("new worker identity does not match owned child PID")
            birth = (identity["pid"], identity["start_ticks"], identity.get("pgid"), identity.get("sid"))
            if lifetime is not None and birth != lifetime:
                raise ValueError("new worker birth/session identity changed during launch")
            lifetime = birth
            observed[(identity["pid"], identity["start_ticks"])] = identity
            if identity.get("command") == command and identity == previous:
                if child.poll() is not None:
                    raise RuntimeError("new worker exited during launch settlement")
                if time.monotonic() >= end:
                    raise TimeoutError("worker launch identity settlement budget exhausted")
                return identity
            previous = identity if identity.get("command") == command else None
        else:
            previous = None
        time.sleep(min(.02, max(0., end - time.monotonic())))
    raise TimeoutError("worker launch identity did not settle to the expected command")


def own_process_census():
    listing = subprocess.run(["ps", "-U", str(os.getuid()), "-o", "pid=,ppid=,pgid=,sid=,stat=,comm="],
                             check=True, capture_output=True, text=True, timeout=10).stdout
    lines = listing.splitlines()
    if len(listing) > 262144 or len(lines) > 4096:
        raise ValueError("own-UID process graph exceeds bounded inventory")
    result = []
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not all(value.isdigit() for value in fields[:4]):
            raise ValueError("invalid own-UID process graph")
        result.append(dict(zip(("pid", "ppid", "pgid", "sid"), map(int, fields[:4])), state=fields[4], comm=fields[5]))
    return result


def ansys_blockers():
    for attempt in range(2):
        result, raced = [], False
        for row in own_process_census():
            if "ansys" not in row["comm"].lower() or row.get("state", "").startswith("Z"):
                continue
            if "pid" not in row or "state" not in row:
                raise RuntimeError("incomplete ANSYS blocker identity; refusing launch")
            identity = process_identity(row["pid"])
            if identity is None:
                raced = True
                continue
            try:
                executable = os.readlink(f'/proc/{row["pid"]}/exe')
            except (FileNotFoundError, ProcessLookupError):
                raced = True
                continue
            result.append(dict(**identity, ppid=row["ppid"], state=row["state"],
                               executable=executable, comm=row["comm"], uid=os.getuid()))
        if not raced:
            if len(result) > 256:
                raise RuntimeError("ANSYS blocker inventory exceeds its bound")
            return result
    raise RuntimeError("ANSYS process identity changed during census; refusing launch")


def wait_ansys_idle(deadline, *, group_binding=None):
    if not math.isfinite(deadline):
        raise ValueError("ANSYS wait requires a finite deadline")
    end = min(time.time() + ANSYS_WAIT_SECONDS, deadline - 90.)
    while True:
        if time.time() >= deadline - 90.:
            raise TimeoutError("ANSYS wait exhausted the recorded stage cleanup budget")
        blockers = foreign_ansys_blockers(group_binding)
        if not blockers:
            return
        record = dict(event="WAITING_ANSYS_EXIT", host=builder.physical_host(),
                      wait_deadline_epoch=end, blockers=blockers)
        print(json.dumps(record, sort_keys=True), flush=True)
        if time.time() >= end:
            raise RuntimeError("another own-UID ANSYS process is active before MAPDL launch: "
                               + json.dumps(record, sort_keys=True))
        time.sleep(min(5., end - time.time()))


def descendant_identities(pid):
    pairs = own_process_census()
    descendants = {pid}
    while True:
        expanded = descendants | {row["pid"] for row in pairs if row["ppid"] in descendants}
        if len(expanded) > 256:
            raise ValueError("predecessor has too many descendant processes")
        if expanded == descendants:
            break
        descendants = expanded
    return [identity for child in sorted(descendants) if (identity := process_identity(child)) is not None]


def predecessor_sessions_idle(observed, *, worker_pid, group_binding=None):
    groups, sessions = {p["pgid"] for p in observed}, {p["sid"] for p in observed}
    groups.add(worker_pid)
    sessions.add(worker_pid)
    census = own_process_census()
    if (foreign_ansys_blockers(group_binding) if group_binding is not None else
            any("ansys" in row["comm"].lower() and not row.get("state", "").startswith("Z") for row in census)):
        raise RuntimeError("own-UID ANSYS remains active after predecessor completion")
    return not any((row["pgid"] in groups or row["sid"] in sessions)
                   and not row.get("state", "").startswith("Z") for row in census)


def validate_predecessor_binding(binding):
    if not isinstance(binding, dict) or set(binding) != {
        "work", "campaign_sha256", "launch_sha256", "worker_pid", "processes",
    }:
        raise ValueError("invalid predecessor binding")
    local_work(binding["work"])
    if type(binding["worker_pid"]) is not int or binding["worker_pid"] <= 0:
        raise ValueError("invalid predecessor worker PID")
    for field in ("campaign_sha256", "launch_sha256"):
        value = binding[field]
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("invalid predecessor receipt hash")
    processes = binding["processes"]
    if not isinstance(processes, list) or len(processes) > 256:
        raise ValueError("invalid bounded predecessor process inventory")
    for identity in processes:
        if (not isinstance(identity, dict) or set(identity) != {"pid", "start_ticks", "command", "pgid", "sid"}
                or type(identity["pid"]) is not int or identity["pid"] <= 0
                or any(type(identity[k]) is not int or identity[k] <= 0 for k in ("pgid", "sid"))
                or not isinstance(identity["start_ticks"], str)
                or not re.fullmatch(r"[0-9]{1,32}", identity["start_ticks"])
                or not isinstance(identity["command"], list)
                or any(not isinstance(arg, str) for arg in identity["command"])):
            raise ValueError("invalid pinned predecessor process identity")


def predecessor_state(binding):
    validate_predecessor_binding(binding)
    work = local_work(binding["work"])
    old, sha = read_record(work / "CAMPAIGN.json", binding["campaign_sha256"])
    launch, _ = read_record(work / "LAUNCH.json", binding["launch_sha256"])
    archive = builder.safe_path(old["archive"])
    if (old.get("schema") != "yap.operator_grid.campaign.v1" or old.get("pilot_si_um") != list(PILOT_SI)
            or old.get("work") != str(work) or launch.get("work") != str(work)
            or launch.get("schema") != "yap.operator_grid.launch.v1"
            or launch.get("worker_pid") != binding["worker_pid"]
            or launch.get("campaign_sha256") != sha
            or launch.get("source_pins_sha256") != old.get("source_pins_sha256")
            or archive.parent != ARCHIVES or not archive.name.startswith("operator_grid_")):
        raise ValueError("predecessor launch/campaign binding mismatch")
    state, _ = read_record(work / "STATUS.json")
    if state.get("schema") != "yap.operator_grid.status.v1" or state.get("worker_pid") != binding["worker_pid"]:
        raise ValueError("predecessor status identity mismatch")
    if state.get("state") == "RUNNING":
        identity = process_identity(binding["worker_pid"])
        pinned = next((p for p in binding["processes"] if p["pid"] == binding["worker_pid"]), None)
        if identity is not None and (pinned != identity or identity["command"] != launch.get("command")):
            raise RuntimeError("predecessor disappeared or changed identity before immutable completion")
        if identity is not None:
            return None
        # A normal exit can race the first STATUS read; recheck the terminal record.
        state, _ = read_record(work / "STATUS.json")
        if (state.get("schema") != "yap.operator_grid.status.v1"
                or state.get("worker_pid") != binding["worker_pid"] or state.get("state") == "RUNNING"):
            raise RuntimeError("predecessor disappeared or changed identity before immutable completion")
    if state.get("state") != "PILOT_COMPLETE_NOT_FULL_GRID":
        raise RuntimeError("predecessor is not successful; manual review required")
    try:
        complete, complete_sha = read_record(archive / "PILOT_COMPLETE.json")
    except FileNotFoundError:
        identity = process_identity(binding["worker_pid"])
        pinned = next((p for p in binding["processes"] if p["pid"] == binding["worker_pid"]), None)
        if identity is not None and identity == pinned and identity["command"] == launch.get("command"):
            return None
        if identity is not None:
            raise RuntimeError("predecessor changed identity during completion publication")
        # Publication can finish between the failed open and the worker-exit check.
        complete, complete_sha = read_record(archive / "PILOT_COMPLETE.json")
    if complete != state:
        raise ValueError("predecessor status differs from immutable completion")
    read_record(archive / "CAMPAIGN.json", binding["campaign_sha256"])
    records = complete.get("completed")
    if not isinstance(records, list) or len(records) != len(PILOT_SI):
        raise ValueError("predecessor requires all four archived cases")
    for height, record in zip(PILOT_SI, records):
        target = archive / ("p2_" + si_key(height))
        if (record.get("thickness_um") != height or record.get("archive") != str(target)
                or record.get("state") != "ARCHIVED_STRUCTURAL_CANDIDATE"):
            raise ValueError("predecessor case completion mismatch")
        done, _ = read_record(archive / f"COMPLETED_p2_{si_key(height)}.json")
        if done != record:
            raise ValueError("predecessor case receipt mismatch")
        inventory, _ = read_record(target / "ARCHIVE.json", record["archive_sha256"])
        audit, _ = read_record(target / "audit/LAYER.json", record["audit_sha256"])
        if (inventory.get("runtime_admitted") is not False
                or inventory.get("files_sha256", {}).get("audit/LAYER.json") != record["audit_sha256"]
                or audit.get("state") != "PASS_STRUCTURAL_CANDIDATE" or audit.get("thickness_um") != height
                or audit.get("dataset_sha256") != record.get("dataset_sha256")
                or audit.get("port_rank_per_face") != 64 or audit.get("all_cu_points_retained") is not True
                or audit.get("thermal_geometry_validated") is not False):
            raise ValueError("predecessor archive/audit structural contract mismatch")
    return dict(work=str(work), completion_sha256=complete_sha, archived_cases=len(records))


def bind_predecessor(value):
    work = local_work(value)
    _, campaign_sha = read_record(work / "CAMPAIGN.json")
    launch, launch_sha = read_record(work / "LAUNCH.json")
    binding = dict(work=str(work), campaign_sha256=campaign_sha, launch_sha256=launch_sha,
                   worker_pid=launch["worker_pid"], processes=descendant_identities(launch["worker_pid"]))
    predecessor_state(binding)
    return binding


def wait_predecessor(plan, work, state):
    binding = plan["predecessor"]
    if binding is None:
        return
    deadline = min(plan["created_epoch"] + PREDECESSOR_WAIT_SECONDS,
                   plan["expires_epoch"] - len(plan["si_thicknesses_um"]) * CASE_SECONDS)
    observed = {(p["pid"], p["start_ticks"]): p for p in binding["processes"]}
    while True:
        if time.time() >= deadline:
            raise TimeoutError("predecessor wait exhausted the original queue budget")
        complete = predecessor_state(binding)
        if complete is None:
            for identity in descendant_identities(binding["worker_pid"]):
                observed[(identity["pid"], identity["start_ticks"])] = identity
            if len(observed) > 256:
                raise ValueError("observed predecessor process identities exceed the bounded inventory")
        alive = [p for p in observed.values() if (current := process_identity(p["pid"])) is not None
                 and current["start_ticks"] == p["start_ticks"]]
        if complete is not None and not alive and predecessor_sessions_idle(
                list(observed.values()), worker_pid=binding["worker_pid"]):
            if time.time() >= deadline:
                raise TimeoutError("predecessor proof finished after the original wait deadline")
            builder.write_json(work / "PREDECESSOR_RELEASE.json",
                dict(**complete, observed_processes=list(observed.values()), predecessor_processes_exited=True),
                immutable=True)
            return
        state.update(state="WAITING_PREDECESSOR", predecessor_work=binding["work"],
                     predecessor_processes_alive=len(alive), predecessor_wait_deadline_epoch=deadline)
        builder.write_json(work / "STATUS.json", state)
        print(f"WAITING predecessor: immutable_complete={complete is not None}, alive={len(alive)}", flush=True)
        time.sleep(min(30., max(0., deadline - time.time())))


@contextmanager
def native_host_lock(work):
    work = local_work(work)
    path = local_work(builder.scratch_root(builder.physical_host()) / ".yap_operator_grid.lock")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            raise ValueError("host lock must be an owned regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield dict(fd=descriptor, path=str(path), device=info.st_dev, inode=info.st_ino)
    finally:
        os.close(descriptor)


def group_slots(plan):
    jobs = plan.get("jobs", [])
    if plan.get("schema") == "yap.operator_grid.parallel_pair.v1" and len(jobs) == 2:
        return range(2)
    if (plan.get("schema") == "yap.operator_grid.parallel_screens.v1"
            and 2 <= len(jobs) <= SCREEN_POOL_MAX_SLOTS.get(plan.get("execution_host"), 0)
            and all(j.get("kind") == "screen" for j in jobs)
            and len({j.get("work") for j in jobs}) == len(jobs)):
        return range(len(jobs))
    raise ValueError("invalid bounded parallel group")


def authenticate_group(binding):
    if (not isinstance(binding, dict) or set(binding) != {"work", "plan_sha256", "launch_sha256", "slot"}
            or type(binding["slot"]) is not int or not 0 <= binding["slot"] < 4):
        raise ValueError("invalid fixed-pair group binding")
    root = local_work(binding["work"])
    plan, _ = read_record(root / "GROUP_PLAN.json", binding["plan_sha256"])
    launch_record, _ = read_record(root / "LAUNCH.json", binding["launch_sha256"])
    identity = launch_record["worker_identity"]
    if (binding["slot"] not in group_slots(plan) or plan.get("work") != str(root)
            or plan.get("execution_host") != builder.physical_host()
            or launch_record.get("plan_sha256") != binding["plan_sha256"]
            or launch_record.get("command") != identity["command"] or process_identity(identity["pid"]) != identity
            or time.time() >= plan["expires_epoch"]):
        raise ValueError("parallel supervisor identity, policy or expiry mismatch")
    for filename, expected in plan["controllers_sha256"].items():
        if builder.digest(regular(filename)) != expected:
            raise ValueError("parallel supervisor controller drift")
    lock, _ = read_record(root / "HOST_LOCK.json")
    canonical = local_work(builder.scratch_root(builder.physical_host()) / ".yap_operator_grid.lock")
    info = regular(canonical).stat()
    if (lock.get("worker_identity") != identity or lock.get("plan_sha256") != binding["plan_sha256"]
            or lock.get("path") != str(canonical) or (lock.get("device"), lock.get("inode")) != (info.st_dev, info.st_ino)
            or info.st_uid != os.getuid() or info.st_nlink != 1 or type(lock.get("fd")) is not int):
        raise ValueError("parallel host-lock inode/owner mismatch")
    held = Path(f"/proc/{identity['pid']}/fd/{lock['fd']}").stat()
    with Path(f"/proc/{identity['pid']}/fdinfo/{lock['fd']}").open() as stream:
        fdinfo = stream.read(16385)
    if len(fdinfo) > 16384 or (held.st_dev, held.st_ino) != (info.st_dev, info.st_ino):
        raise ValueError("parallel supervisor does not retain its host-lock FD")
    expression = r"lock:\s+\d+:\s+FLOCK\s+ADVISORY\s+WRITE\s+(\d+)\s+([0-9a-fA-F]+):([0-9a-fA-F]+):(\d+)\s+0\s+EOF"
    matches = re.findall(expression, fdinfo)
    if not any((int(pid), int(major, 16), int(minor, 16), int(inode)) ==
               (identity["pid"], os.major(info.st_dev), os.minor(info.st_dev), info.st_ino)
               for pid, major, minor, inode in matches):
        raise ValueError("parallel supervisor lacks the kernel exclusive FLOCK")
    return plan, identity, lock


def group_child(binding, *, slot=None):
    plan, supervisor, _ = authenticate_group(binding)
    selected = binding["slot"] if slot is None else slot
    if type(selected) is not int or selected not in group_slots(plan):
        raise ValueError("parallel group has exactly two slots")
    root = Path(binding["work"])
    path = root / f"CHILD{selected}.json"
    if not path.exists() or not (root / "GROUP_READY.json").exists():
        return None
    ready, _ = read_record(root / "GROUP_READY.json")
    if ready.get("group_plan_sha256") != binding["plan_sha256"]:
        raise ValueError("parallel registration barrier belongs to another group")
    child, _ = read_record(path, ready["children_sha256"][str(selected)])
    job = plan["jobs"][selected]
    work = local_work(job["work"])
    launch_record, _ = read_record(work / "LAUNCH.json", child["launch_sha256"])
    expected_name = "CAMPAIGN.json" if job["kind"] == "build" else "PLAN.json"
    child_plan, _ = read_record(work / expected_name, child["plan_sha256"])
    expected_binding = dict(binding, slot=selected)
    if (child.get("work") != str(work) or child.get("plan_filename") != expected_name
            or child.get("group_plan_sha256") != binding["plan_sha256"]
            or launch_record.get("worker_identity") != child["identity"]
            or launch_record.get("command") != child["identity"]["command"]
            or launch_record.get("campaign_sha256" if job["kind"] == "build" else "plan_sha256") != child["plan_sha256"]
            or child_plan.get("group_binding") != expected_binding
            or child_plan.get("source_pins_sha256", child_plan.get("pins_sha256")) != child["pins_sha256"]
            or child["identity"]["command"][2] != job["controller"]):
        raise ValueError("parallel child registration/plan mismatch")
    if process_identity(child["identity"]["pid"]) != child["identity"]:
        return None
    parentage = [r for r in own_process_census() if r["pid"] == child["identity"]["pid"]]
    if not parentage or process_identity(child["identity"]["pid"]) != child["identity"]:
        return None
    if parentage[0]["ppid"] != supervisor["pid"]:
        raise ValueError("parallel registered root is not a direct supervisor child")
    return child


def foreign_ansys_blockers(group_binding=None):
    blockers = ansys_blockers()
    if group_binding is None:
        return blockers
    plan, _, _ = authenticate_group(group_binding)
    allowed = set()
    for slot in group_slots(plan):
        if slot == group_binding["slot"]:
            continue
        peer = group_child(group_binding, slot=slot)
        if peer is None:
            continue
        verify_pins(Path(peer["work"]) / "SOURCE_PINS.json", peer["pins_sha256"])
        if process_identity(peer["identity"]["pid"]) != peer["identity"]:
            continue
        descendants = descendant_identities(peer["identity"]["pid"])
        if process_identity(peer["identity"]["pid"]) == peer["identity"] and peer["identity"] in descendants:
            allowed.update((p["pid"], p["start_ticks"]) for p in descendants)
    return [p for p in blockers if (p["pid"], p["start_ticks"]) not in allowed]


def assert_group_worker(binding, work):
    plan, _, _ = authenticate_group(binding)
    if plan["jobs"][binding["slot"]]["work"] != str(local_work(work)):
        raise ValueError("parallel authority belongs to another work directory")
    child = group_child(binding)
    if child is None or os.getpid() not in {p["pid"] for p in descendant_identities(child["identity"]["pid"])}:
        raise ValueError("caller is not the registered parallel child")
    ready, _ = read_record(Path(binding["work"]) / "GROUP_READY.json")
    if ready.get("group_plan_sha256") != binding["plan_sha256"]:
        raise ValueError("parallel registration barrier belongs to another group")
    for slot in group_slots(plan):
        _, actual = read_record(Path(binding["work"]) / f"CHILD{slot}.json", ready["children_sha256"][str(slot)])
    return child


def group_host_fd(binding):
    _, _, lock = authenticate_group(binding)
    info = os.fstat(lock["fd"])
    if (info.st_dev, info.st_ino) != (lock["device"], lock["inode"]):
        raise ValueError("parallel host FD is not held by this launcher")
    return lock["fd"]


_GROUP_BOOTSTRAP = None


def group_launch_environment(environment, binding):
    if binding is None:
        return environment
    _, parent, lock = authenticate_group(binding)
    return dict(environment, YAP_PARALLEL_SUPERVISOR=json.dumps(dict(parent=parent,
        fd=lock["fd"], device=lock["device"], inode=lock["inode"])))


def bootstrap_group_environment():
    global _GROUP_BOOTSTRAP
    raw = os.environ.pop("YAP_PARALLEL_SUPERVISOR", None)
    if raw is None:
        return
    import ctypes
    if len(raw) > 65536:
        raise ValueError("parallel bootstrap metadata exceeds its finite bound")
    meta = json.loads(raw)
    parent, descriptor = meta["parent"], meta["fd"]
    held = os.fstat(descriptor)
    if ((held.st_dev, held.st_ino) != (meta["device"], meta["inode"])
            or os.getppid() != parent["pid"] or process_identity(parent["pid"]) != parent):
        raise ValueError("parallel bootstrap parent or inherited FD mismatch")
    os.set_inheritable(descriptor, False)
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"parallel bootstrap interrupted: {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot install early parent-death containment")
    if os.getppid() != parent["pid"] or process_identity(parent["pid"]) != parent:
        raise RuntimeError("parallel supervisor disappeared during bootstrap")
    _GROUP_BOOTSTRAP = meta
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT, signal.SIGALRM, signal.SIGHUP})


@contextmanager
def worker_solver_lock(work, binding=None):
    if binding is None:
        with native_host_lock(work):
            yield
        return
    import ctypes
    _, parent, lock = authenticate_group(binding)
    if os.getppid() != parent["pid"]:
        raise ValueError("parallel worker is not a direct supervisor child")
    inherited = lock["fd"]
    if _GROUP_BOOTSTRAP != dict(parent=parent, fd=inherited, device=lock["device"], inode=lock["inode"]):
        raise ValueError("parallel worker did not complete early containment bootstrap")
    held = os.fstat(inherited)
    if (held.st_dev, held.st_ino) != (lock["device"], lock["inode"]):
        raise ValueError("parallel worker did not inherit the ownership FD")
    os.set_inheritable(inherited, False)
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"parallel worker interrupted: {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot install parallel parent-death containment")
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT, signal.SIGALRM, signal.SIGHUP})
    authenticate_group(binding)
    if os.getppid() != parent["pid"]:
        raise RuntimeError("parallel supervisor disappeared during containment setup")
    deadline = time.time() + 60.
    while not (Path(binding["work"]) / "GROUP_READY.json").exists():
        authenticate_group(binding)
        if time.time() >= deadline:
            raise TimeoutError("parallel child registration barrier expired")
        time.sleep(.2)
    assert_group_worker(binding, work)
    try:
        yield
    finally:
        # Closing a duplicate preserves the supervisor's lock; LOCK_UN would
        # release the shared open-file-description lock for every worker.
        os.close(inherited)


def check_case_resources(work, *, group_binding=None):
    wait_ansys_idle(time.time() + ANSYS_WAIT_SECONDS + 95., **({"group_binding": group_binding} if group_binding else {}))
    resource = builder.resource_check(work)
    count = len(authenticate_group(group_binding)[0]["jobs"]) if group_binding is not None else 1
    if resource["local_free_gib"] < 200 + RESERVATION_GIB * count:
        raise RuntimeError("insufficient local storage for another reserved 64-GiB case")
    if group_binding is not None and (resource["available_memory_gib"] < 64 * count
            or resource["cpu_busy_fraction"] + 8 * count / resource["logical_cpus"] > .9):
        raise RuntimeError("insufficient resources for the reserved two-job group")
    if (foreign_ansys_blockers(group_binding) if group_binding else resource["own_ansys_processes"]):
        raise RuntimeError("another own-UID ANSYS process is active; do not interrupt it")
    return resource


def local_work(value):
    return builder.local_scratch(value)


def worker(args):
    work = local_work(args.work)
    source_pins = work / "SOURCE_PINS.json"
    verify_pins(source_pins, args.pins_sha256)
    plan, _ = read_record(work / "CAMPAIGN.json", args.campaign_sha256)
    validate_campaign(plan, work, args)
    archive = builder.safe_path(plan["archive"])
    if archive.parent != ARCHIVES or not archive.name.startswith("operator_grid_"):
        raise ValueError("grid archive must use a new neutral canonical results directory")
    environment = builder.configure(work)
    group_binding = plan.get("group_binding")
    with worker_solver_lock(work, group_binding), (work / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (work / "STATUS.json").exists():
            raise RuntimeError("existing campaign attempt requires review; no automatic retry")
        state = dict(schema="yap.operator_grid.status.v2", state="RUNNING", worker_pid=os.getpid(),
                     started_epoch=time.time(), completed=[], requested_coverage=81,
                     native_execution_expanded=False)
        builder.write_json(work / "STATUS.json", state)
        from operator_thermal_validation import ProcessInventory, publish_worker_scope
        publish_worker_scope(ProcessInventory())
        archive_identity = None
        try:
            api = builder.load_api(args.native_source, work)
            cases = [api.GeometryCase("p2_" + si_key(h), "bi2", h, plan["execution_host"], nx=45, ny=20, rank=64, cores=8)
                     for h in plan["si_thicknesses_um"]]
            api.CASES = {case.key: case for case in cases}
            original_verify, original_steps, original_dataset = api.verify_inputs, api.static_steps, api.build_dataset
            case_deadline = 0.
            active_adoption = None
            if group_binding is not None:
                from operator_thermal_validation import run_child
                def supervised_stage(command, log, name, *, environment=None):
                    print(f"START {name}: {log}", flush=True)
                    run_child(command, log, case_deadline, environment=environment, group_binding=group_binding)
                    print(f"COMPLETE {name}", flush=True)
                builder.run_stage = supervised_stage
            def verified(record):
                verify_pins(source_pins, args.pins_sha256)
                original_verify(record)
                if active_adoption is not None and (api.ROOT / active_adoption["case_key"] / "ADOPTED_INPUTS.json").exists():
                    bounded_cpu(case_deadline, lambda: verify_private_inputs(
                        api.ROOT / active_adoption["case_key"], active_adoption, hashes=False))
            def supervised_steps(case):
                steps = original_steps(case)
                for step in steps:
                    if active_adoption is not None and step["name"] in {"extract_tangent", "extract_q6"}:
                        stage = "tangent" if step["name"] == "extract_tangent" else "q6"
                        api.set_option(step["command"], "--source-dir", active_adoption["raw_copy_dirs"][stage])
                    authority = ["--campaign-file", str(work / "CAMPAIGN.json"), "--campaign-sha256", args.campaign_sha256] \
                        if group_binding is not None else []
                    step["command"] = [sys.executable, "-B", str(Path(__file__).resolve()), "native-stage",
                        "--pins", str(source_pins), "--pins-sha256", args.pins_sha256,
                        "--case-deadline", str(case_deadline), *authority, "--", *step["command"]]
                return steps
            api.verify_inputs, api.static_steps = verified, supervised_steps
            def bounded_dataset(case, **values):
                return bounded_cpu(case_deadline, lambda: original_dataset(case, **values))
            api.build_dataset = bounded_dataset
            archive.mkdir()
            info = archive.stat()
            archive_identity = (info.st_dev, info.st_ino)
            builder.copy_checked(work / "CAMPAIGN.json", archive / "CAMPAIGN.json", args.campaign_sha256)
            builder.copy_checked(source_pins, archive / "SOURCE_PINS.json", args.pins_sha256)
            wait_predecessor(plan, work, state)
            state.update(state="RUNNING")
            for case in cases:
                now = time.time()
                if now + CASE_SECONDS > plan["expires_epoch"]:
                    raise TimeoutError("queue lacks the reserved case wall budget")
                resource = check_case_resources(work, **({"group_binding": group_binding} if group_binding else {}))
                case_deadline = now + CASE_SECONDS
                active_adoption = plan["adoption"] if plan["adoption"] is not None and \
                    plan["adoption"]["case_key"] == case.key else None
                state.update(active_case=case.key, case_deadline_epoch=case_deadline, resources=resource)
                builder.write_json(work / "STATUS.json", state)
                if active_adoption is None:
                    builder.build_case(api, case, work, environment)
                else:
                    adopter = adoption_callback(api, case, work, active_adoption)
                    bounded_cpu(case_deadline,
                        lambda: builder.build_case(api, case, work, environment, adopt_stage=adopter))
                    bounded_cpu(case_deadline,
                        lambda: verify_private_inputs(api.ROOT / case.key, active_adoption))
                if time.time() > case_deadline:
                    raise TimeoutError("case exceeded its recorded wall budget")
                api.verify_inputs(builder.read_json(api.ROOT / case.key / "STATIC_PLAN.json"))
                builder.resource_check(work)
                record = bounded_cpu(min(case_deadline, plan["expires_epoch"]),
                    lambda: publish_layer(api, case, archive, source_pins, args.pins_sha256,
                        plan["existing_si"]["endpoints"][si_key(38.5)]["material_contract"]))
                verify_pins(source_pins, args.pins_sha256)
                state["completed"].append(record)
                builder.write_json(work / "STATUS.json", state)
                builder.write_json(archive / f"COMPLETED_{case.key}.json", record, immutable=True)
            state.update(state="BATCH_COMPLETE_NOT_FULL_GRID", completed_epoch=time.time())
            builder.write_json(work / "STATUS.json", state)
            builder.write_json(archive / "BATCH_COMPLETE.json", state, immutable=True)
        except BaseException as error:
            state.update(state="STOP_REQUIRES_REVIEW", error=f"{type(error).__name__}: {error}", stopped_epoch=time.time())
            builder.write_json(work / "STATUS.json", state)
            if archive_identity is not None and not archive.is_symlink() and archive.is_dir() \
                    and (archive.stat().st_dev, archive.stat().st_ino) == archive_identity:
                builder.write_json(archive / "STOP.json", state, immutable=True)
            raise


def prepare_launch(args):
    heights = validate_batch(args.si_thicknesses)
    work = local_work(args.work)
    archive = builder.safe_path(args.archive)
    if archive.parent != ARCHIVES or not archive.name.startswith("operator_grid_") or archive.exists():
        raise ValueError("launch requires a new neutral canonical archive")
    if any((work / name).exists() for name in ("CAMPAIGN.json", "SOURCE_PINS.json", "STATUS.json", "LAUNCH.json")):
        raise ValueError("existing campaign requires manual review, never restart it")
    if args.review_sha256 != builder.digest(regular(__file__)):
        raise ValueError("launch must bind the reviewed controller SHA256")
    predecessor = bind_predecessor(args.after_campaign) if args.after_campaign is not None else None
    if predecessor is not None and predecessor["work"] == str(work):
        raise ValueError("a campaign cannot wait for itself")
    host = builder.physical_host()
    group_binding = getattr(args, "group_binding", None)
    if group_binding is not None:
        group, _, _ = authenticate_group(group_binding)
        if group["jobs"][group_binding["slot"]]["work"] != str(work) or group["jobs"][group_binding["slot"]]["kind"] != "build":
            raise ValueError("build preparation group membership mismatch")
    check_case_resources(work if work.exists() else builder.scratch_root(host),
        **({"group_binding": group_binding} if group_binding else {}))
    environment = builder.configure(work)
    baseline = checked_baseline(args.baseline_runtime)
    native = builder.safe_path(args.native_source)
    if builder.digest(regular(native / "exp12_ofat_geometry_20260913.py")) != NATIVE_HELPER_SHA:
        raise ValueError("native geometry helper drift")
    api = builder.load_api(native, work)
    cases = [api.GeometryCase("p2_" + si_key(h), "bi2", h, host, nx=45, ny=20, rank=64, cores=8) for h in heights]
    for case in cases:
        builder.verify_template(api, case)
    adoption = bind_adoption(args.adopt_from, api, cases[0], work) \
        if getattr(args, "adopt_from", None) is not None else None
    pins = source_closure(api, cases)
    builder.write_json(work / "SOURCE_PINS.json", pins, immutable=True)
    pins_sha = builder.digest(work / "SOURCE_PINS.json")
    verify_pins(work / "SOURCE_PINS.json", pins_sha)
    existing_si = checked_existing_si(args.parent_package)
    verify_pins(work / "SOURCE_PINS.json", pins_sha)
    resource = check_case_resources(work, **({"group_binding": group_binding} if group_binding else {}))
    plan = dict(schema="yap.operator_grid.campaign.v4", si_thicknesses_um=heights,
                execution_host=host, local_scratch_root=str(builder.scratch_root(host)),
                work=str(work), native_source=str(native), archive=str(archive), baseline=baseline,
                existing_si=existing_si,
                coverage=coverage_plan(), resources=resource, review_sha256=args.review_sha256,
                source_pins_sha256=pins_sha, created_epoch=time.time(), expires_epoch=time.time() + QUEUE_SECONDS,
                stage_wall_seconds=STAGE_SECONDS, case_wall_seconds=CASE_SECONDS,
                predecessor=predecessor, predecessor_wait_seconds=PREDECESSOR_WAIT_SECONDS,
                host_serialization=True, ansys_wait_seconds=ANSYS_WAIT_SECONDS, adoption=adoption,
                per_case_local_reservation_gib=RESERVATION_GIB,
                retained_solver_scratch=True, automatic_retry=False, interpolation_enabled=False, group_binding=group_binding)
    builder.write_json(work / "CAMPAIGN.json", plan, immutable=True)
    campaign_sha = builder.digest(work / "CAMPAIGN.json")
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--work", str(work),
               "--native-source", str(native), "--pins-sha256", pins_sha, "--campaign-sha256", campaign_sha]
    return work, environment, command, campaign_sha, pins_sha


def launch(args, *, preparation_deadline=None, prepared=None):
    if prepared is None:
        prepared = prepare_launch(args) if preparation_deadline is None else bounded_cpu(
            preparation_deadline, lambda: prepare_launch(args))
    work, environment, command, campaign_sha, pins_sha = prepared
    archive = builder.safe_path(args.archive)
    observed = {}
    child = None
    try:
        if preparation_deadline is not None and time.time() >= preparation_deadline:
            raise TimeoutError("worker launch preparation deadline exhausted")
        with (work / "worker.log").open("x") as log:
            group_binding = getattr(args, "group_binding", None)
            inherited = (group_host_fd(group_binding),) if group_binding else ()
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     env=group_launch_environment(environment, group_binding),
                                     start_new_session=True, pass_fds=inherited)
        identity = wait_exec_identity(child, command, observed, deadline=preparation_deadline)
        receipt = dict(schema="yap.operator_grid.launch.v1", worker_pid=child.pid, worker_identity=identity,
                       command=command, campaign_sha256=campaign_sha, source_pins_sha256=pins_sha,
                       reviewed_controller_sha256=args.review_sha256, work=str(work), archive=str(archive))
        builder.write_json(work / "LAUNCH.json", receipt, immutable=True)
        if preparation_deadline is not None and time.time() >= preparation_deadline:
            raise TimeoutError("worker launch publication deadline exhausted")
        print(json.dumps(receipt, indent=2), flush=True)
        return child
    except BaseException:
        if child is not None:
            from operator_thermal_validation import cleanup_spawned
            cleanup_spawned(child, observed)
        raise


def publish_complete(args):
    deadline = time.time() + CASE_SECONDS
    return bounded_cpu(deadline, lambda: _publish_complete(args, deadline))


def _publish_complete(args, deadline):
    work, old = local_work(args.work), local_work(args.source_work)
    archive = builder.safe_path(args.archive)
    if work.is_relative_to(old) or old.is_relative_to(work):
        raise ValueError("publication audit work must be separate from the stopped source")
    if archive.parent != ARCHIVES or not archive.name.startswith("operator_grid_") or archive.exists():
        raise ValueError("completed publication requires a new neutral archive")
    if any((work / name).exists() for name in ("PUBLICATION_PLAN.json", "SOURCE_PINS.json", "STATUS.json")):
        raise ValueError("existing publication attempt requires manual review")
    if args.review_sha256 != builder.digest(regular(__file__)):
        raise ValueError("publication must bind the reviewed controller SHA256")
    native = builder.safe_path(args.native_source)
    if builder.digest(regular(native / "exp12_ofat_geometry_20260913.py")) != NATIVE_HELPER_SHA:
        raise ValueError("native geometry helper drift")
    source_campaign, source_campaign_sha = read_record(old / "CAMPAIGN.json")
    source_status, source_status_sha = read_record(old / "STATUS.json")
    source_launch, source_launch_sha = read_record(old / "LAUNCH.json")
    if (source_status.get("state") != "STOP_REQUIRES_REVIEW"
            or source_campaign.get("native_source") != str(native)
            or source_campaign.get("resources", {}).get("host") != builder.physical_host()
            or source_campaign.get("work") != str(old) or source_launch.get("work") != str(old)
            or source_launch.get("campaign_sha256") != source_campaign_sha
            or source_launch.get("source_pins_sha256") != source_campaign.get("source_pins_sha256")
            or source_status.get("worker_pid") != source_launch.get("worker_pid")
            or process_identity(source_launch["worker_pid"]) is not None):
        raise ValueError("publication source must be an exited, reviewed stopped campaign")
    require_exited_source(source_launch["worker_pid"])
    verify_pins(old / "SOURCE_PINS.json", source_campaign["source_pins_sha256"])
    api = builder.load_api(native, old)
    key = "p2_" + si_key(args.si_thickness)
    status, status_sha = read_record(api.ROOT / key / "RUN_STATUS.json")
    plan, plan_sha = read_record(api.ROOT / key / "STATIC_PLAN.json")
    ready, ready_sha = read_record(api.ROOT / key / "DATASET_READY.json")
    case = api.GeometryCase(**plan["case"])
    if (case.key != key or case.family != "bi2" or case.host != builder.physical_host()
            or (case.nx, case.ny, case.rank, case.cores) != (45, 20, 64, 8)
            or case.substrate_um != args.si_thickness or case.layer != "tsv_si"
            or status.get("state") != "COMPLETE_ENGINEERING_CANDIDATE"
            or [s["stage"] for s in status.get("completed", [])] != list(builder.STAGES)
            or ready.get("thermal_geometry_validated") is not False):
        raise ValueError("publication requires all six exact native64 target stages and dataset")
    api.verify_inputs(plan)
    for step in plan["steps"]:
        api.verify_static_stage(case, step)
    check_case_resources(work if work.exists() else builder.scratch_root(case.host))
    builder.configure(work)
    pins = source_closure(api, [case])
    builder.write_json(work / "SOURCE_PINS.json", pins, immutable=True)
    pins_sha = builder.digest(work / "SOURCE_PINS.json")
    expected = checked_existing_si(args.parent_package)["endpoints"][si_key(38.5)]["material_contract"]
    publication = dict(schema="yap.operator_grid.completed_publication.v1", source_work=str(old),
                       work=str(work), archive=str(archive), case=asdict(case),
                       original_stop_sha256=source_status_sha, source_case_status_sha256=status_sha,
                       source_campaign_sha256=source_campaign_sha, source_launch_sha256=source_launch_sha,
                       source_static_plan_sha256=plan_sha, source_dataset_ready_sha256=ready_sha,
                       source_pins_sha256=pins_sha, review_sha256=args.review_sha256,
                       old_stop_preserved=True, source_metadata_rewritten=False,
                       source_case_modified=False, audit_output=str(work / "artifacts" / key / "AUDITED_FINAL_BUBBLE_SPACE.npz"))
    builder.write_json(work / "PUBLICATION_PLAN.json", publication, immutable=True)
    state = dict(state="RUNNING_PUBLICATION_ONLY", worker_pid=os.getpid(), active_case=key,
                 started_epoch=time.time(), case_deadline_epoch=deadline)
    builder.write_json(work / "STATUS.json", state)
    archive_identity = None
    try:
        with native_host_lock(work):
            archive.mkdir()
            info = archive.stat()
            archive_identity = (info.st_dev, info.st_ino)
            builder.copy_checked(work / "PUBLICATION_PLAN.json", archive / "PUBLICATION_PLAN.json")
            builder.copy_checked(work / "SOURCE_PINS.json", archive / "SOURCE_PINS.json", pins_sha)
            result = bounded_cpu(state["case_deadline_epoch"], lambda: publish_layer(api, case, archive,
                work / "SOURCE_PINS.json", pins_sha, expected, audit_output=publication["audit_output"]))
            read_record(old / "STATUS.json", source_status_sha)
            read_record(api.ROOT / key / "RUN_STATUS.json", status_sha)
            read_record(api.ROOT / key / "STATIC_PLAN.json", plan_sha)
            read_record(api.ROOT / key / "DATASET_READY.json", ready_sha)
            if time.time() > deadline:
                raise TimeoutError("completed publication exceeded its original operation deadline")
            state.update(state="COMPLETE_PUBLICATION_ONLY", result=result, completed_epoch=time.time())
            builder.write_json(archive / "PUBLICATION_COMPLETE.json", state, immutable=True)
    except BaseException as error:
        state.update(state="STOP_REQUIRES_REVIEW", error=f"{type(error).__name__}: {error}")
        if archive_identity is not None and not archive.is_symlink() and \
                (archive.stat().st_dev, archive.stat().st_ino) == archive_identity:
            builder.write_json(archive / "STOP.json", state, immutable=True)
        raise
    finally:
        builder.write_json(work / "STATUS.json", state)
    print(json.dumps(state, indent=2), flush=True)


TRAINING_NATIVE = "run_exp12_macro_history_stack_rve.py"
TRAINING_OUTPUTS = "run_exp12_date_uniform_total_thickness_v2.py"
TRAINING_EXTRACTOR = "extract_exp12_m4_volume_displacement_snapshots.py"
TRAINING_FIELDS = ("eps_x_at_y0", "eps_z_at_y0", "strain_gradient_x_1_per_mm",
                   "strain_gradient_z_1_per_mm", "gamma_xz_at_y0", "strain_gradient_xz_1_per_mm")
TRAINING_DIRECTIONS = ("eps_x", "eps_z", "kappa_x", "kappa_z", "gamma_xz", "kappa_xz")
TRAINING_PATHS = ("thermal_zero_macro",) + tuple(
    f"{direction}_{sign}" for direction in TRAINING_DIRECTIONS for sign in ("plus", "minus"))
TRAINING_HOLDOUTS = ("mixed_amplitude_holdout", "reversal_holdout")
TRAINING_SECONDS = 28800
TRAINING_ARTIFACTS = ("PRESCRIBED_PATH.csv", "SOLVE.dat", "PREFLIGHT.json", "MAPDL_COMPLETION.json",
                      "RAW_SOLVER_SEAL.json", "INITIAL_CLOCK_AUDIT.json", "SNAPSHOT_MACRO.csv",
                      "VOLUME_DISPLACEMENT_SNAPSHOTS.npz")
TRAINING_MACRO_FIELDS = ("eps_x0", "eps_z0", "kappa_x_1_per_mm", "kappa_z_1_per_mm",
                         "gamma_xz0", "kappa_xz_1_per_mm")
TRAINING_NATIVE_MACRO_FIELDS = ("eps_x0", "kappa_x", "eps_z0", "kappa_z", "gamma_xz0", "kappa_xz")
TRAINING_CARDS = {
    "CU_MATERIAL_CARD_V2_KIM2026.json": "b3a876bcd8847e5ee2366580155b66ef5a3c01a7971a3b44b152b47eb46db8aa",
    "NONCU_MATERIAL_CARD_V2_KIM2026.json": "45a77a87f93a962defb08ead20ae3676cd41aede84ae4ee43aed90ac7a52acc4",
    "PROCESS_CARD_V2_KIM2026.json": "d8d8a14a57bba5370c884300aa3f5d35b9e3ac9adb698cc249ad4692ba253dcc",
}


def training_spec(pitch, layer="bi", si_um=38.5, path="thermal_zero_macro", nx=48):
    if type(pitch) is not int or pitch not in PITCHES or layer not in {"bi", "beol", "tsv_si"}:
        raise ValueError("invalid requested training geometry")
    if type(si_um) not in (int, float) or not math.isfinite(si_um) or si_um <= 0:
        raise ValueError("invalid Si thickness")
    if path not in TRAINING_PATHS + TRAINING_HOLDOUTS:
        raise ValueError("unknown training/holdout path")
    cell = math.lcm(pitch, 6)
    if type(nx) is not int or not 12 <= nx <= 384 or any(nx % (cell // p) for p in (pitch, 6)):
        raise ValueError("mesh must be bounded and commensurate with both physical periods")
    thickness, ny = {"bi": (1.5, 6), "beol": (10., 18), "tsv_si": (float(si_um), 20)}[layer]
    return dict(pitch_um=pitch, cell_um=cell, diameter_um=.5 * pitch, layer=layer,
                thickness_um=thickness, si_um=float(si_um), nx=nx, nz=nx, ny=ny,
                cycles=3, path=path, role="held_out_validation" if path in TRAINING_HOLDOUTS else "training",
                initial_q6=[0.] * 6, strain_amplitude=.001, curvature_amplitude_1_per_mm=2. / thickness,
                reservation_gib=math.ceil((45 if layer == "bi" else 64) * (nx / 45)**2),
                nodal_substeps=True, intermediate_element_results=False,
                thermal_geometry_validated=False, runtime_admitted=False)


def training_rows(spec):
    h_mm, rows = spec["thickness_um"] / 1000., []
    for cycle in range(1, 4):
        for state in ("heat_end", "hot_hold", "coolback_rt"):
            q = [0.] * 6
            factor = 0. if state == "coolback_rt" else 1.
            name = spec["path"]
            if name in TRAINING_HOLDOUTS:
                factor *= .65 if name == "mixed_amplitude_holdout" else (-1.)**cycle
                q = [factor * value for value in (.001, -.0007, .002 / h_mm,
                                                  -.0014 / h_mm, .0008, .0016 / h_mm)]
            elif name != "thermal_zero_macro":
                direction, sign = name.rsplit("_", 1)
                index = TRAINING_DIRECTIONS.index(direction)
                q[index] = factor * (1. if sign == "plus" else -1.) * (
                    spec["strain_amplitude"] if index in (0, 1, 4) else spec["curvature_amplitude_1_per_mm"])
            q[0] -= .5 * h_mm * q[2]
            q[1] -= .5 * h_mm * q[3]
            q[4] -= .5 * h_mm * q[5]
            rows.append(dict(cycle=cycle, state=state, **dict(zip(TRAINING_FIELDS, q, strict=True))))
    return rows


def validate_training_rows(rows):
    expected = {(c, s) for c in range(1, 4) for s in ("heat_end", "hot_hold", "coolback_rt")}
    if len(rows) != 9:
        raise ValueError("training path requires exactly nine endpoints")
    seen = set()
    for row in rows:
        key = (int(row["cycle"]), row["state"])
        if key in seen or key not in expected or not all(math.isfinite(float(row[f])) for f in TRAINING_FIELDS):
            raise ValueError("duplicate, missing or nonfinite training endpoint")
        seen.add(key)
    if seen != expected:
        raise ValueError("training endpoint coverage mismatch")


def isolated_native_parser(native):
    # Adapt only the parser factory; keep the scientific module and sys.argv intact.
    source = regular(native.__file__, max_bytes=2 * 1024**2)
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "parse_args"]
    if len(nodes) != 1:
        raise ValueError("native parser structure changed")
    function = nodes[0]
    returns = [n for n in ast.walk(function) if isinstance(n, ast.Return)]
    terminal = function.body[-1]
    if len(returns) != 1 or not isinstance(terminal, ast.Return) or not isinstance(terminal.value, ast.Call):
        raise ValueError("native parser terminal return changed")
    call = terminal.value
    if (not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name)
            or call.func.value.id != "p" or call.func.attr != "parse_args" or call.args or call.keywords):
        raise ValueError("native parser must end with p.parse_args()")
    terminal.value = ast.Name(id="p", ctx=ast.Load())
    tree = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(vars(native))
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["parse_args"]()


def training_closure(source):
    source = regular(source / TRAINING_NATIVE).parent
    common = regular(source.parent / "common/ansys_utils.py")
    if builder.digest(common) != COMMON_HELPER_SHA:
        raise ValueError("unreviewed solver cleanup helper")
    seeds = [Path(__file__), Path(builder.__file__), Path(__file__).parent / "operator_thermal_validation.py",
             common, source / TRAINING_NATIVE, source / TRAINING_OUTPUTS,
             source / TRAINING_EXTRACTOR, source / "exp12_250c_branch.py"]
    if (common.parent / "__init__.py").is_file():
        seeds.append(common.parent / "__init__.py")
    pending, files, started = list(seeds), set(seeds), time.monotonic()
    while pending:
        if len(files) > 96 or time.monotonic() - started > 30:
            raise ValueError("training source closure exceeds file/time bound")
        path = regular(pending.pop(), max_bytes=2 * 1024**2)
        modules = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules.update(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                modules.add(node.module)
        for module in modules:
            roots = [Path(__file__).parent, source, source.parent / "09_effective_material_die_validation",
                     source.parent / "11_symmetric_mesh_beol_rve_verify"]
            if module.startswith("common."):
                roots = [source.parent]
            matches = list(dict.fromkeys(root / (module.replace(".", "/") + ".py") for root in roots
                                        if (root / (module.replace(".", "/") + ".py")).is_file()))
            if len(matches) > 1:
                raise ValueError(f"ambiguous training dependency: {module}")
            if matches and matches[0] not in files:
                files.add(matches[0])
                pending.append(matches[0])
    digests = {str(regular(p)): builder.digest(p) for p in sorted(files)}
    for name, expected in TRAINING_CARDS.items():
        card = regular(source / "results/material_card_v2_kim2026" / name)
        if builder.digest(card) != expected:
            raise ValueError(f"training material/process card changed: {name}")
        digests[str(card)] = expected
    return dict(schema="yap.operator_grid.sources.v1", common_utils=str(common), files_sha256=digests)


def training_args(native, spec, work, source):
    cards = source / "results/material_card_v2_kim2026"
    argv = ["--output-dir", str(work / "native"), "--run-root", str(work / "solver"),
            "--explicit-result-dir", str(work / "NO_EXTERNAL_TARGET"),
            "--macro-history-csv", str(work / "PRESCRIBED_PATH.csv"),
            "--explicit-layer-sequence", f"{spec['layer']}:{spec['thickness_um']:g}:{spec['ny']}",
            "--macro-bc-mode", "prescribed_generalized_periodic", "--enable-inplane-shear-twist",
            "--per-band-resultants", "--composition-preserving-voxel", "--inclusion-shape", "circle",
            "--fixed-thermal-substeps", "--ramp-substeps", "160", "--ramp-substeps-max", "160",
            "--hold-substeps", "40", "--hold-substeps-max", "40", "--save-nodal-substeps",
            "--cycles", "3", "--initial-prebond-sec", "300", "--ansys-mode", "smp", "--ansys-np", "8",
            "--ansys-mem-mb", "20480", "--ansys-db-mb", "4096", "--tsv-floorplan", "full"]
    values = dict(coupon_x_um=spec["cell_um"], coupon_z_um=spec["cell_um"], projection_x_um=spec["cell_um"],
                  projection_z_um=spec["cell_um"], bi_pitch_um=spec["pitch_um"], bi_pad_side_ratio=.5,
                  bi_um=1.5, beol_um=10., substrate_um=spec["si_um"], beol_mesh_pitch_um=6.,
                  beol_cu_fraction=.35, tsv_pitch_um=6., tsv_side_um=3., nx=spec["nx"], nz=spec["nz"],
                  ny_bi=6, ny_beol=18, ny_substrate=20)
    for key, value in values.items():
        argv.extend(["--" + key.replace("_", "-"), str(value)])
    for key, name in zip(("cu-material-card", "noncu-material-card", "process-card"), TRAINING_CARDS, strict=True):
        argv.extend(["--" + key, str(cards / name)])
    args = isolated_native_parser(native).parse_args(argv)
    native.configure_beol_sheet_geometry(args)
    native.explicit3d.configure_interface_local_refinement(args)
    metadata = native.explicit3d.configure_cards(args)
    args._dense_boundary_history = {}
    if (args.disable_cu_plasticity or args.disable_cu_creep or args.save_all_substeps
            or args.dense_macro_history_csv is not None or args.target_source_csv is not None
            or args.initial_prebond_sec != 300. or args.cycles != 3
            or (args.t_room_c, args.t_anneal_c, args.ramp_up_sec, args.hold_sec, args.cool_down_sec) != (25., 350., 650., 3600., 650.)
            or (args.creep_on_ramp_up, args.creep_on_dwell, args.creep_on_cool_down) != (False, True, False)):
        raise ValueError("configured training physics/process differs from reviewed contract")
    return args, metadata


def training_geometry(native, args, spec):
    import numpy as np
    rows = [r for r in native.beol_sheet_composition_audit(args) if r["layer"] == spec["layer"]]
    if len(rows) != 1 or float(rows[0]["absolute_fraction_error"]) > .01:
        raise ValueError("training Cu composition failed the absolute 0.01 gate")
    material = np.asarray(native.explicit3d.voxel_material_maps(args)[spec["layer"]])
    period = spec["pitch_um"] if spec["layer"] == "bi" else 6
    count, size = spec["cell_um"] // period, spec["nx"] // (spec["cell_um"] // period)
    if material.shape != (spec["nx"], spec["nz"]) or not np.array_equal(material, np.tile(material[:size, :size], (count, count))):
        raise ValueError("training mesh has nonidentical periodic material tiles")
    return dict(composition=rows, cells_per_period=size, repeats_per_axis=count,
                nodes=(spec["nx"] + 1)**2 * (spec["ny"] + 1),
                elements=spec["nx"]**2 * spec["ny"], mesh_convergence_qualified=False)


def audit_training_deck(deck, records):
    times = [300.] + [300. + 4900. * c + dt for c in range(3) for dt in (650., 4250., 4900.)]
    if ([r["time_s"] for r in records] != times or [r["substeps"] for r in records] != [1] + [160, 40, 160] * 3):
        raise ValueError("training clock/substep schedule changed")
    rates, rate = [], None
    for line in deck.splitlines():
        if line.startswith("RATE,"):
            rate = line.split(",")[1]
        if line == "SOLVE":
            rates.append(rate)
    if (rates != ["OFF"] + ["OFF", "ON", "OFF"] * 3 or "BFUNIF,TEMP," in deck
            or deck.count("ALLSEL,ALL\nKBC,0\nBF,ALL,TEMP,") != 10
            or "TANNEAL = 3.500000000000e+02" not in deck
            or "OUTRES,ALL,LAST" not in deck or "OUTRES,NSOL,ALL" not in deck
            or deck.count("/EXIT,NOSAVE") != 1):
        raise ValueError("training temperature/creep/result retention contract changed")
    labels = ["initial_rt"] + [f"{phase}_C{c}" for c in range(1, 4) for phase in ("heat", "hot", "cool")]
    for record, label in zip(records, labels, strict=True):
        record["stage"] = label
    return records


def training_clock():
    expected = [300.]
    for cycle in range(3):
        start = 300. + 4900. * cycle
        for duration, steps in ((650., 160), (3600., 40), (650., 160)):
            expected.extend(start + duration * i / steps for i in range(1, steps + 1))
            start += duration
    return expected


def training_clock_rows(rows):
    expected = training_clock()
    if len(rows) != 1081 or [int(r["result_set"]) for r in rows] != list(range(1, 1082)):
        raise ValueError("training macro history requires 1081 consecutive actual sets")
    for row, t in zip(rows, expected, strict=True):
        if not all(math.isfinite(float(v)) for v in row.values()) or abs(float(row["time_s"]) - t) > 1e-6:
            raise ValueError("training macro history differs from actual continuous process clock")
    return expected


def training_initial_clock(initial, first_snapshot_time, first_macro):
    if (len(initial) != 1 or float(initial[0]["time_s"]) not in (1., 300.)
            or abs(float(first_snapshot_time) - 300.) > 1e-6 or abs(float(first_macro["time_s"]) - 300.) > 1e-6):
        raise ValueError("training initial clock mismatch cannot be reconciled")
    keys = ("eps_x0", "eps_z0", "kappa_x_1_per_mm", "kappa_z_1_per_mm", "gamma_xz0", "kappa_xz_1_per_mm")
    if any(not math.isfinite(float(initial[0][k])) or abs(float(initial[0][k])) > 1e-10
           or abs(float(first_macro[k])) > 1e-10 for k in keys):
        raise ValueError("training initial q6 must be zero, not free thermal equilibrium")
    return dict(raw_initial=initial[0], raw_timestamp_rejected=float(initial[0]["time_s"]) != 300.,
                corrected=dict(initial[0], time_s=300.))


def validate_training_macro(rows, spec):
    times = training_clock_rows(rows)
    validate_selected_training_macro(rows, times, spec)


def validate_selected_training_macro(rows, times, spec):
    endpoint_times = [300.] + [300. + 4900. * c + dt for c in range(3) for dt in (650., 4250., 4900.)]
    endpoints = [[0.] * 6] + [[r[k] for k in TRAINING_FIELDS] for r in training_rows(spec)]
    index = 1
    for row, t in zip(rows, times, strict=True):
        while index < len(endpoint_times) - 1 and t > endpoint_times[index] + 1e-6:
            index += 1
        fraction = (t - endpoint_times[index - 1]) / (endpoint_times[index] - endpoint_times[index - 1])
        for j, key in enumerate(TRAINING_MACRO_FIELDS):
            expected = endpoints[index - 1][j] + fraction * (endpoints[index][j] - endpoints[index - 1][j])
            if not math.isfinite(float(row[key])) or abs(float(row[key]) - expected) > 1e-10 * max(1., abs(expected)):
                raise ValueError("actual macro q6 differs from the prescribed training trajectory")


def training_snapshot_rows(rows, spec):
    selected_sets = list(range(1, 1082, 10))
    times = training_clock()[::10]
    expected_fields = {"result_set", "time_s", *TRAINING_MACRO_FIELDS}
    if len(rows) != 109 or any(set(row) != expected_fields for row in rows):
        raise ValueError("selected macro trace requires exactly 109 rows and the declared q6 fields")
    for row, result_set, t in zip(rows, selected_sets, times, strict=True):
        if (not str(row["result_set"]).isdigit() or int(row["result_set"]) != result_set
                or not all(math.isfinite(float(v)) for v in row.values())
                or abs(float(row["time_s"]) - t) > 1e-6):
            raise ValueError("selected macro sets/order differ from the independent continuous clock")
    validate_selected_training_macro(rows, times, spec)
    return selected_sets, times


def training_array_contract(spec):
    import numpy as np
    count = (spec["nx"] + 1)**2 * (spec["ny"] + 1)
    return dict(node_ids=(np.dtype("int64"), (count,)), coordinates_mm=(np.dtype("float64"), (count, 3)),
                times_s=(np.dtype("float64"), (109,)), result_sets=(np.dtype("int64"), (109,)),
                displacements_mm=(np.dtype("float64"), (109, count, 3)))


def training_face_registration(coordinates, spec):
    import numpy as np
    axes = [np.linspace(0., length, intervals + 1) for length, intervals in
            ((spec["cell_um"] / 1000., spec["nx"]), (spec["thickness_um"] / 1000., spec["ny"]),
             (spec["cell_um"] / 1000., spec["nz"]))]
    indices = []
    for dimension, axis in enumerate(axes):
        index = np.rint(coordinates[:, dimension] / axis[-1] * (len(axis) - 1)).astype(np.int64)
        if (np.any(index < 0) or np.any(index >= len(axis))
                or not np.allclose(coordinates[:, dimension], axis[index], atol=1e-11, rtol=0)):
            raise ValueError("training coordinates differ from the declared physical Cartesian grid")
        indices.append(index)
    flat = np.ravel_multi_index(indices, tuple(len(a) for a in axes))
    if len(np.unique(flat)) != len(coordinates):
        raise ValueError("duplicate/missing physical training coordinates")
    faces = []
    for plane in (0, spec["ny"]):
        face = np.flatnonzero(indices[1] == plane)
        order = np.lexsort((indices[2][face], indices[0][face]))
        face = face[order]
        if len(face) != (spec["nx"] + 1)**2:
            raise ValueError("incomplete registered training face")
        faces.append(face)
    if not np.array_equal(coordinates[faces[0]][:, [0, 2]], coordinates[faces[1]][:, [0, 2]]):
        raise ValueError("opposite training face registration mismatch")
    return tuple(faces)


def training_csv(path):
    with regular(path, max_bytes=256 * 1024).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("duplicate/missing training CSV columns")
        return list(reader)


def training_hash(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("invalid training SHA256")
    return value


def training_numeric_csv(path, columns, truncated_header, *, max_rows=1081):
    # The pinned APDL exporter prints '(A)' headers with a default width of eight.
    with regular(path, max_bytes=2 * 1024**2).open(newline="") as stream:
        reader = csv.reader(stream)
        header = ",".join(value.strip() for value in next(reader, []))
        if header not in (truncated_header, ",".join(columns)):
            raise ValueError("unrecognized native training CSV header")
        rows = []
        for row in reader:
            if len(row) != len(columns) or len(rows) >= max_rows:
                raise ValueError("native training CSV row width/count mismatch")
            values = [float(value) for value in row]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("nonfinite native training CSV")
            rows.append(dict(zip(columns, values, strict=True)))
    return rows


def training_endpoints(path, records, spec):
    columns = ("load_step", "time_s", *TRAINING_NATIVE_MACRO_FIELDS)
    rows = training_numeric_csv(path, columns, "load_ste", max_rows=10)
    if len(rows) != 10 or len(records) != 10:
        raise ValueError("native training requires ten complete endpoints")
    canonical = []
    for row, record in zip(rows, records, strict=True):
        if row["load_step"] != record["load_step"] or abs(row["time_s"] - record["time_s"]) > 1e-6:
            raise ValueError("native endpoint load-step/clock mismatch")
        canonical.append(dict(time_s=row["time_s"], **{
            field: row[native] for field, native in zip(
                TRAINING_MACRO_FIELDS, ("eps_x0", "eps_z0", "kappa_x", "kappa_z", "gamma_xz0", "kappa_xz"), strict=True)}))
    validate_selected_training_macro(canonical, [row["time_s"] for row in canonical], spec)
    return [dict(stage=record["stage"], **row) for row, record in zip(rows, records, strict=True)]


def training_dense_trace(path, spec):
    rows = training_numeric_csv(path, ("result_set", "time_s", *TRAINING_NATIVE_MACRO_FIELDS), "result_s")
    canonical = []
    for row in rows:
        if row["result_set"] != int(row["result_set"]):
            raise ValueError("fractional actual training result-set ID")
        canonical.append(dict(result_set=int(row["result_set"]), time_s=row["time_s"], **{
            field: row[native] for field, native in zip(
                TRAINING_MACRO_FIELDS, ("eps_x0", "eps_z0", "kappa_x", "kappa_z", "gamma_xz0", "kappa_xz"), strict=True)}))
    validate_training_macro(canonical, spec)
    return canonical


def load_training_evidence(directory, receipt_sha256):
    """Read completed sealed evidence; launch expiry/host are not offline authority.

    Original solver/source paths are provenance only. Never reopen live RSTs or
    require the historical producer to equal today's controller.
    """
    import numpy as np
    directory = builder.safe_path(directory)
    receipt, _ = read_record(directory / "TRAINING_COMPLETE.json", training_hash(receipt_sha256))
    spec = receipt.get("geometry", {})
    if spec != training_spec(spec.get("pitch_um"), spec.get("layer"), spec.get("si_um"),
                             spec.get("path"), spec.get("nx")):
        raise ValueError("completed training geometry changed")
    if (receipt.get("state") != "TRAINING_EXTRACTED_NOT_OPERATOR"
            or receipt.get("thermal_geometry_validated") is not False or receipt.get("runtime_admitted") is not False
            or receipt.get("actual_sets") != 1081 or receipt.get("selected_snapshots") != 109
            or set(receipt.get("files_sha256", {})) != set(TRAINING_ARTIFACTS)):
        raise ValueError("invalid completed training receipt")
    plan, plan_sha = read_record(directory / "TRAINING_PLAN.json", training_hash(receipt["source_plan_sha256"]))
    if (plan.get("schema") != "yap.operator_grid.training.v1" or plan.get("geometry") != spec
            or plan.get("automatic_retry") is not False or plan.get("scratch_retained") is not True
            or plan.get("protected_validation_targets_read") != []):
        raise ValueError("completed training plan disagrees with the sealed receipt")
    pins, pins_sha = read_record(directory / "SOURCE_PINS.json", training_hash(plan["source_pins_sha256"]))
    files = pins.get("files_sha256", {})
    if pins.get("schema") != "yap.operator_grid.sources.v1" or not isinstance(files, dict) or not 1 <= len(files) <= 96:
        raise ValueError("invalid completed training source closure")
    for path, digest in files.items():
        builder.safe_path(path)
        training_hash(digest)
    for name, expected in {**TRAINING_CARDS, "ansys_utils.py": COMMON_HELPER_SHA,
                           "operator_grid.py": training_hash(plan["review_sha256"])}.items():
        if [digest for path, digest in files.items() if Path(path).name == name] != [expected]:
            raise ValueError(f"historical training producer/card binding changed: {name}")
    if files.get(pins.get("common_utils")) != COMMON_HELPER_SHA:
        raise ValueError("historical training cleanup helper binding changed")
    contract = training_array_contract(spec)
    expected_bytes = sum(dtype.itemsize * math.prod(shape) for dtype, shape in contract.values())
    if receipt.get("snapshot_shape") != list(contract["displacements_mm"][1]) or set(receipt.get("arrays", {})) != set(contract):
        raise ValueError("completed training array manifest differs from geometry")
    for name, (dtype, shape) in contract.items():
        audit = receipt["arrays"][name]
        if audit.get("dtype") != str(dtype) or audit.get("shape") != list(shape):
            raise ValueError(f"training array dtype/shape manifest mismatch: {name}")
        training_hash(audit["sha256"])
    for name, digest in receipt["files_sha256"].items():
        bound = expected_bytes + 2 * 1024**2 if name.endswith(".npz") else (
            contract["node_ids"][1][0] * 8192 + 2 * 1024**2 if name == "SOLVE.dat" else 2 * 1024**2)
        if builder.digest(regular(directory / name, max_bytes=bound)) != training_hash(digest):
            raise ValueError(f"completed training artifact SHA256 mismatch: {name}")
    if receipt["files_sha256"]["PRESCRIBED_PATH.csv"] != plan.get("path_sha256"):
        raise ValueError("prescribed path plan binding mismatch")
    if training_csv(directory / "PRESCRIBED_PATH.csv") != [
            {k: str(v) for k, v in r.items()} for r in training_rows(spec)]:
        raise ValueError("sealed prescribed path differs from the declared training trajectory")
    preflight, _ = read_record(directory / "PREFLIGHT.json")
    stages = preflight.get("stages", [])
    audit_training_deck((directory / "SOLVE.dat").read_text(), stages)
    composition = preflight.get("geometry", {}).get("composition", [])
    if (preflight.get("deck_sha256") != receipt["files_sha256"]["SOLVE.dat"]
            or preflight.get("geometry", {}).get("nodes") != contract["node_ids"][1][0]
            or preflight.get("geometry", {}).get("elements") != spec["nx"]**2 * spec["ny"]
            or len(composition) != 1 or composition[0].get("layer") != spec["layer"]
            or not math.isfinite(float(composition[0].get("absolute_fraction_error", float("nan"))))
            or not 0 <= composition[0]["absolute_fraction_error"] <= .01
            or preflight.get("protected_validation_targets_read") != []
            or preflight.get("thermal_geometry_validated") is not False or preflight.get("runtime_admitted") is not False):
        raise ValueError("sealed training preflight geometry/physics changed")
    completion, _ = read_record(directory / "MAPDL_COMPLETION.json")
    if completion.get("rc") != 0 or completion.get("license_error"):
        raise ValueError("training solver did not complete cleanly")
    raw, _ = read_record(directory / "RAW_SOLVER_SEAL.json")
    raw_files = raw.get("files_sha256", {})
    if len(raw_files) != 2 or {Path(p).suffix for p in raw_files} != {".rdb", ".rst"}:
        raise ValueError("invalid original training solver seals")
    for path, digest in raw_files.items():
        builder.safe_path(path)
        training_hash(digest)
    if set(receipt.get("raw_extraction_sha256", {})) != {"volume_nodes.csv", "volume_displacements.csv"}:
        raise ValueError("missing raw training extraction provenance")
    for digest in receipt["raw_extraction_sha256"].values():
        training_hash(digest)
    if ("recovery_plan_sha256" in receipt or "recovery_evidence_sha256" in receipt
            or raw.get("original_raw_seal_contemporaneous") is False or (directory / "RECOVERY_COMPLETE.json").exists()):
        validate_recovery_evidence(directory, receipt, plan, preflight)
    macro = training_csv(directory / "SNAPSHOT_MACRO.csv")
    sets, times = training_snapshot_rows(macro, spec)
    initial, _ = read_record(directory / "INITIAL_CLOCK_AUDIT.json")
    baseline = training_initial_clock([initial["raw_initial"]], times[0], macro[0])
    if (any(initial.get(key) != value for key, value in baseline.items())
            or initial.get("raw_solver_hashes") != raw_files):
        raise ValueError("initial clock correction lacks its sealed original solver provenance")
    for name in ("raw_initial_csv_sha256", "endpoint_export_sha256"):
        training_hash(initial[name])
    snapshot_path = directory / "VOLUME_DISPLACEMENT_SNAPSHOTS.npz"
    # Inspect bounded NPY headers before allowing NumPy to allocate/decompress arrays.
    with zipfile.ZipFile(snapshot_path) as archive:
        headers = archive.infolist()
        if sorted(h.filename for h in headers) != sorted(name + ".npy" for name in (*contract, "layer")):
            raise ValueError("unexpected/duplicate training NPZ members")
        for header in headers:
            name = header.filename[:-4]
            limit = (expected_bytes if name != "layer" else 256) + 4096
            if header.file_size > limit or header.flag_bits & 1:
                raise ValueError("training NPZ exceeds declared geometry or is encrypted")
            with archive.open(header) as stream:
                version = np.lib.format.read_magic(stream)
                if version not in ((1, 0), (2, 0)):
                    raise ValueError("unsupported training NPY header")
                shape, fortran, dtype = (np.lib.format.read_array_header_1_0(stream) if version == (1, 0)
                                         else np.lib.format.read_array_header_2_0(stream))
                if name == "layer":
                    valid = shape == () and dtype.kind == "U" and dtype.itemsize == 4 * len(spec["layer"])
                else:
                    valid = (dtype, shape) == contract[name]
                    limit = dtype.itemsize * math.prod(shape) + 4096
                if not valid or fortran or header.file_size > limit:
                    raise ValueError(f"training NPY header differs from sealed contract: {name}")
    with np.load(snapshot_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in contract}
        if archive["layer"].item() != spec["layer"]:
            raise ValueError("training NPZ layer mismatch")
    for name, array in arrays.items():
        if (not np.all(np.isfinite(array)) or hashlib.sha256(memoryview(array).cast("B")).hexdigest()
                != receipt["arrays"][name]["sha256"]):
            raise ValueError(f"training array bytes/nonfinite data mismatch: {name}")
    if (not np.array_equal(arrays["result_sets"], sets) or not np.allclose(arrays["times_s"], times, atol=1e-6, rtol=0)
            or np.any(arrays["node_ids"] <= 0) or len(np.unique(arrays["node_ids"])) != len(arrays["node_ids"])):
        raise ValueError("training snapshot IDs/clock mismatch")
    faces = training_face_registration(arrays["coordinates_mm"], spec)
    h_mid = spec["thickness_um"] / 2000.
    native_q = np.asarray([[float(row[key]) for key in TRAINING_MACRO_FIELDS] for row in macro])
    q6 = native_q[:, [0, 1, 4, 2, 3, 5]].copy()
    q6[:, :3] += h_mid * q6[:, 3:]
    return dict(geometry=spec, receipt_sha256=receipt_sha256, plan_sha256=plan_sha, source_pins_sha256=pins_sha,
                arrays=arrays, face_indices=faces, q6_centered=q6,
                q6_labels=["eps_x", "eps_y", "gamma_xy", "kappa_x", "kappa_y", "kappa_xy"],
                reference_plane_mm=h_mid, engineering_shear_and_twist=True,
                thermal_geometry_validated=False, runtime_admitted=False)


FACE_ATOL_MM, FACE_RTOL = 1e-9, 1e-3
FACE_PBC_ATOL_MM, FACE_PBC_RTOL = 1e-10, 1e-6
FACE_Q6_LABELS = ("eps_x", "eps_y", "gamma_xy", "kappa_x", "kappa_y", "kappa_xy")
FACE_SEED_ROUNDOFF_LIMIT = 1e-10


def condense_constraint_rows(rows, *, prescribed=()):
    """Exact substitution of declared acyclic slaves, not a solver qualification."""
    def dof(value):
        if (not isinstance(value, (tuple, list)) or len(value) != 2
                or type(value[0]) is not int or value[0] <= 0 or value[1] not in ("UX", "UY", "UZ")):
            raise ValueError("invalid constraint DOF")
        return tuple(value)
    def number(value):
        if isinstance(value, bool) or not isinstance(value, (str, int, float, Fraction)):
            raise ValueError("invalid rational constraint coefficient")
        if len(str(value)) > 80:
            raise ValueError("constraint coefficient exceeds its bound")
        result = Fraction(str(value))
        if max(result.numerator.bit_length(), result.denominator.bit_length()) > 512:
            raise ValueError("constraint rational coefficient exceeds its bound")
        return result
    if not isinstance(rows, (tuple, list)) or not 1 <= len(rows) <= 4096:
        raise ValueError("bounded declared constraint rows required")
    if not isinstance(prescribed, (tuple, list)) or len(prescribed) > 4096:
        raise ValueError("bounded prescribed DOFs required")
    fixed = {dof(v) for v in prescribed}
    equations, definitions = set(), {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"equation", "slave", "constant", "terms"}:
            raise ValueError("constraint row requires an explicit slave and literal terms")
        index, slave = row["equation"], dof(row["slave"])
        if type(index) is not int or index <= 0 or index in equations or slave in definitions or slave in fixed:
            raise ValueError("constraint equations/slaves must be unique and unprescribed")
        terms = row["terms"]
        if not isinstance(terms, (tuple, list)) or not 1 <= len(terms) <= 24:
            raise ValueError("bounded constraint terms required")
        parsed = []
        for term in terms:
            if not isinstance(term, (tuple, list)) or len(term) != 3:
                raise ValueError("invalid declared constraint term")
            parsed.append((dof(term[:2]), number(term[2])))
        if parsed[0] != (slave, Fraction(1)) or len({key for key, _ in parsed}) != len(parsed):
            raise ValueError("first constraint term must be its declared normalized unique slave")
        equations.add(index)
        definitions[slave] = (index, number(row["constant"]), parsed[1:])
    cache, active, started = {}, set(), time.monotonic()
    def expand(slave, depth=0):
        if slave in cache:
            return cache[slave]
        if slave in active or depth > 16 or time.monotonic()-started > 10:
            raise ValueError("constraint dependency cycle/depth/time bound exceeded")
        active.add(slave)
        index, constant, terms = definitions[slave]
        masters, combination = {}, {index: Fraction(1)}
        for key, coefficient in terms:
            if not coefficient:
                continue
            if key in definitions:
                child_constant, child_masters, child_combination = expand(key, depth+1)
                constant -= coefficient*child_constant
                for master, value in child_masters.items():
                    masters[master] = masters.get(master, Fraction(0))-coefficient*value
                for equation, value in child_combination.items():
                    combination[equation] = combination.get(equation, Fraction(0))-coefficient*value
            else:
                masters[key] = masters.get(key, Fraction(0))-coefficient
        masters = {key: value for key, value in masters.items() if value}
        combination = {key: value for key, value in combination.items() if value}
        if len(masters) > 24 or len(combination) > 64:
            raise ValueError("condensed constraint expansion exceeds its bound")
        active.remove(slave)
        cache[slave] = constant, masters, combination
        return cache[slave]
    condensed, transformations = [], []
    for row in rows:
        slave = dof(row["slave"])
        constant, masters, combination = expand(slave)
        condensed.append(dict(equation=row["equation"], slave=slave, constant=constant,
            terms=[(*slave, Fraction(1)), *((*key, -value) for key, value in sorted(masters.items()))]))
        transformations.append({str(key): str(value) for key, value in sorted(combination.items())})
    return dict(rows=condensed, report=dict(state="EXACT_ACYCLIC_CONSTRAINT_SUBSTITUTION_NOT_PHYSICS_VALIDATION",
        equation_count=len(rows), original_slaves_retained=True, prescribed_dofs_unchanged=True,
        row_combinations=transformations, coefficient_arithmetic="exact rational",
        solver_serialization_qualified=False, thermal_geometry_validated=False, runtime_admitted=False))


def condense_literal_ce_deck(deck, *, declared_slaves, prescribed=()):
    """Diagnostic-only rewrite of numeric PREP7 CEs; production training is unchanged."""
    if not isinstance(deck, str) or len(deck) > 2*1024**2:
        raise ValueError("bounded literal constraint deck required")
    lines, rows, positions = deck.splitlines(), [], []
    solution = False
    for position, line in enumerate(lines):
        if line.strip().upper() == "/SOLU":
            solution = True
        if not line.startswith("CE,"):
            continue
        if solution:
            raise ValueError("diagnostic CE rewrite is limited to PREP7")
        fields = line.split(",")
        if len(fields) not in (6, 9, 12) or not fields[1].isdigit():
            raise ValueError("numeric single-definition CE rows required")
        terms = [(int(fields[i]), fields[i+1], fields[i+2]) for i in range(3, len(fields), 3)]
        index = len(rows)
        if not isinstance(declared_slaves, (list, tuple)) or index >= len(declared_slaves):
            raise ValueError("every CE slave must be declared independently")
        rows.append(dict(equation=int(fields[1]), slave=declared_slaves[index], constant=fields[2], terms=terms))
        positions.append(position)
    if len(rows) != len(declared_slaves):
        raise ValueError("declared CE membership differs from deck")
    result = condense_constraint_rows(rows, prescribed=prescribed)
    replacement = {}
    for position, row in zip(positions, result["rows"], strict=True):
        commands = []
        for offset in range(0, len(row["terms"]), 3):
            values = ["CE", str(row["equation"]), format(float(row["constant"]), ".17e")]
            for node, component, coefficient in row["terms"][offset:offset+3]:
                values.extend((str(node), component, format(float(coefficient), ".17e")))
            commands.append(",".join(values))
        replacement[position] = commands
    rewritten = "\n".join(value for i, line in enumerate(lines) for value in replacement.get(i, [line]))+"\n"
    report = dict(result["report"], original_deck_sha256=hashlib.sha256(deck.encode("ascii")).hexdigest(),
        rewritten_deck_sha256=hashlib.sha256(rewritten.encode("ascii")).hexdigest(),
        serialized_coefficient_format="17 digits after decimal, scientific notation",
        diagnostic_only=True, production_training_modified=False)
    return rewritten, report


def candidate_training_constraint_deck(deck, *, nx, ny, nz):
    """Inactive uniform-grid diagnostic; preserve non-CE physics and old evidence."""
    if (not isinstance(deck, str) or len(deck) > 8*1024**2
            or any(type(v) is not int or not 2 <= v <= 96 for v in (nx, ny, nz))
            or nx != nz or 3*(ny+1)*(nx+nz+2) > 4096):
        raise ValueError("bounded square uniform-grid constraint diagnostic required")
    deck.encode("ascii")
    lines = deck.splitlines()
    if lines.count("/SOLU") != 1:
        raise ValueError("one solution entry required")
    if "SOLVE" not in lines:
        raise ValueError("first solve boundary required")
    first_solve = lines.index("SOLVE")
    positions = [i for i, line in enumerate(lines) if line.startswith("CE,")]
    if not positions or max(positions) >= first_solve:
        raise ValueError("all candidate constraints must precede the first solve")
    physical = (nx+1)*(ny+1)*(nz+1)
    node = lambda i, j, k: 1+(i*(ny+1)+j)*(nz+1)+k
    slaves = []
    for j in range(ny+1):
        slaves.extend((physical+5+j, d) for d in ("UX", "UZ", "UY"))
        for k in range(nz+1):
            slaves.extend((node(nx, j, k), d) for d in ("UX", "UY", "UZ"))
        for i in range(nx):
            slaves.extend((node(i, j, nz), d) for d in ("UX", "UY", "UZ"))
    if len(positions) != len(slaves):
        raise ValueError("candidate CE count disagrees with independent grid membership")
    for offset, position in enumerate(positions):
        fields = lines[position].split(",")
        if len(fields) < 6 or fields[1] != str(1000000+offset):
            raise ValueError("candidate CE identifiers/order changed")
    prescribed = [(physical+1, "UX"), (physical+2, "UX"),
                  (physical+3, "UZ"), (physical+4, "UZ"),
                  (physical+6+ny, "UY"), (physical+7+ny, "UY")]
    prescribed.extend((1, d) for d in ("UX", "UY", "UZ"))
    # Only the bounded CE block enters the parser; full native decks may be larger.
    block = "\n".join(lines[i] for i in positions)+"\n"
    rewritten, report = condense_literal_ce_deck(block, declared_slaves=slaves, prescribed=prescribed)
    replacements, current = {}, None
    for line in rewritten.splitlines():
        equation = int(line.split(",")[1])
        if equation != current:
            if equation in replacements:
                raise ValueError("noncontiguous serialized constraint")
            replacements[equation] = []
            current = equation
        replacements[equation].append(line)
    if sorted(replacements) != list(range(1000000, 1000000+len(slaves))):
        raise ValueError("candidate serialization lost a constraint")
    replacement = {i: replacements[1000000+j] for j, i in enumerate(positions)}
    result = "\n".join(value for i, line in enumerate(lines) for value in replacement.get(i, [line]))+"\n"
    if ([line for line in result.splitlines() if not line.startswith("CE,")]
            != [line for line in lines if not line.startswith("CE,")]):
        raise ValueError("candidate changed non-constraint physics")
    report = dict(report, candidate_full_deck_sha256=hashlib.sha256(result.encode("ascii")).hexdigest(),
                  original_full_deck_sha256=hashlib.sha256(deck.encode("ascii")).hexdigest(),
                  grid=dict(nx=nx, ny=ny, nz=nz, physical_nodes=physical),
                  production_training_modified=False, native_solver_qualified=False)
    return result, report


def face_checked_lstsq(matrix, values):
    import numpy as np
    matrix, values = np.asarray(matrix, float), np.asarray(values, float)
    if (matrix.ndim != 2 or values.ndim != 2 or matrix.shape[0] != values.shape[0]
            or not 0 < matrix.shape[1] <= matrix.shape[0]
            or not np.isfinite(matrix).all() or not np.isfinite(values).all()):
        raise ValueError("invalid face least-squares arrays")
    scales = np.linalg.norm(matrix, axis=0)
    if np.any(scales <= np.finfo(float).tiny):
        raise ValueError("empty face least-squares column")
    coefficients, _, rank, singular = np.linalg.lstsq(matrix / scales, values, rcond=1e-12)
    if rank != matrix.shape[1]:
        raise ValueError("face least-squares design lost declared column rank")
    return coefficients / scales[:, None], dict(rank=int(rank), columns=matrix.shape[1],
        column_scales=scales.tolist(), singular_values=singular.tolist(),
        condition_number=float(singular[0] / singular[-1]))


def face_kinematic_contract(coordinates):
    """Analytic face traces only; these are not minimum-energy volume lifts."""
    import numpy as np
    xyz = np.asarray(coordinates, float)
    if xyz.ndim != 3 or xyz.shape[0] != 2 or xyz.shape[2] != 3 or not np.isfinite(xyz).all():
        raise ValueError("expected two finite registered faces with shape (2,node,3)")
    xz = xyz[0][:, (0, 2)]
    if not np.array_equal(xz, xyz[1][:, (0, 2)]):
        raise ValueError("opposite face coordinate ordering differs")
    x, z = (np.unique(xz[:, i]) for i in range(2))
    if (len(x) < 3 or len(x) != len(z) or len(x)**2 != len(xz)
            or not np.isclose(np.ptp(x), np.ptp(z), atol=1e-11, rtol=0)
            or not np.allclose(np.diff(x), np.diff(x)[0], atol=1e-11, rtol=0)
            or not np.allclose(np.diff(z), np.diff(z)[0], atol=1e-11, rtol=0)
            or not all(np.all(xyz[j, :, 1] == xyz[j, 0, 1]) for j in (0, 1))
            or not xyz[1, 0, 1] > xyz[0, 0, 1]):
        raise ValueError("faces must be complete uniform square grids, ordered bottom then top")
    ix, iz = np.searchsorted(x, xz[:, 0]), np.searchsorted(z, xz[:, 1])
    if len(np.unique(ix * len(z) + iz)) != len(xz):
        raise ValueError("duplicate or missing face coordinates")
    groups = {}
    for face in (0, 1):
        for node, (a, b) in enumerate(zip(ix, iz, strict=True)):
            groups.setdefault((face, int(a % (len(x)-1)), int(b % (len(z)-1))), []).append(face * len(xz) + node)
    groups = tuple(np.asarray(v, dtype=int) for v in groups.values() if len(v) > 1)
    pairs = np.asarray([(v[0], i) for v in groups for i in v[1:]], dtype=int)
    flat = xyz.reshape(-1, 3)
    centered = flat - (flat.min(0) + flat.max(0)) / 2
    X, Y, Z = centered.T
    q = np.zeros((len(flat), 3, 6))
    q[:, 0, 0], q[:, 2, 1] = X, flat[:, 2]
    q[:, 0, 2], q[:, 2, 2] = .5 * Z, .5 * X
    q[:, 0, 3], q[:, 1, 3] = Y * X, -.5 * X**2
    q[:, 2, 4], q[:, 1, 4] = Y * Z, -.5 * Z**2
    q[:, 0, 5], q[:, 2, 5], q[:, 1, 5] = .5 * Y * Z, .5 * Y * X, -.5 * X * Z
    rigid = np.zeros((len(flat), 3, 6))
    rigid[:, :, :3] = np.eye(3)
    for axis in range(3):
        rigid[:, :, axis+3] = np.cross(np.eye(3)[axis], centered)
    face_checked_lstsq(rigid.reshape(-1, 6), np.zeros((flat.size, 1)))
    return dict(coordinates_mm=xyz.copy(), q6_traces=q, rigid_traces=rigid, pairs=pairs,
                periodic_groups=groups, node_count=len(xz), q6_labels=FACE_Q6_LABELS)


def face_periodic_spread(fields, contract):
    import numpy as np
    fields = np.asarray(fields, float)
    n = 2 * contract["node_count"]
    if fields.ndim != 3 or fields.shape[1:] != (n, 3) or not np.isfinite(fields).all():
        raise ValueError("invalid fields for the complete face periodicity audit")
    # Corners have four equivalents; anchor-only errors can hide their full spread.
    return np.max(np.asarray([np.ptp(fields[:, group], axis=1).max(axis=1)
                              for group in contract["periodic_groups"]]), axis=0)


def audit_face_history(coordinates, displacement, q6_centered, *, reference=None):
    """Fix the periodic gauge without projecting physical fields off rotations."""
    import numpy as np
    contract = face_kinematic_contract(coordinates)
    xyz, n = contract["coordinates_mm"], contract["node_count"]
    u, q6 = np.asarray(displacement, float), np.asarray(q6_centered, float)
    base = np.zeros_like(xyz) if reference is None else np.asarray(reference, float)
    if (u.ndim != 4 or u.shape[1:] != xyz.shape or not len(u) or q6.shape != (len(u), 6)
            or base.shape != xyz.shape or not all(np.isfinite(a).all() for a in (u, q6, base))):
        raise ValueError("invalid face history/centered q6/reference arrays")
    raw_absolute = (u-base).reshape(len(u), 2*n, 3)
    raw_increment = (u-u[:1]).reshape(len(u), 2*n, 3)
    raw = np.concatenate((raw_absolute[:1], raw_increment, raw_absolute))
    prescribed = np.concatenate((q6[:1], q6-q6[:1], q6))
    rigid, macro = contract["rigid_traces"], contract["q6_traces"]
    G = rigid.reshape(-1, 6)
    Q, _ = np.linalg.qr(G / np.linalg.norm(G, axis=0))
    raw_flat = raw.reshape(len(raw), -1)
    scale_fields = raw_flat - (raw_flat @ Q) @ Q.T
    scales = np.max(np.abs(scale_fields), axis=1)
    remainder = raw - np.einsum("ndq,sq->snd", macro, prescribed)
    low, high = contract["pairs"].T
    rotation_design = (rigid[high, :, 3:] - rigid[low, :, 3:]).reshape(-1, 3)
    jumps = (remainder[:, high] - remainder[:, low]).reshape(len(raw), -1)
    angles, rotation_audit = face_checked_lstsq(rotation_design, jumps.T)
    rotated = remainder - np.einsum("ndq,qs->snd", rigid[:, :, 3:], angles)
    spread = face_periodic_spread(rotated, contract)
    limits = FACE_PBC_ATOL_MM + FACE_PBC_RTOL * scales
    if np.any(spread > limits):
        raise ValueError("raw initial/increment/absolute face history violates periodicity after edge-determined rotation")
    shared_translation = rotated.mean(axis=1)
    periodic = rotated - shared_translation[:, None]
    cut = 1 + len(u)
    return dict(contract=contract, initial_periodic_mm=periodic[0].reshape(xyz.shape),
                increment_periodic_mm=periodic[1:cut].reshape(u.shape),
                absolute_periodic_mm=periodic[cut:].reshape(u.shape),
                all_periodic_mm=periodic, fixed_scales_mm=scales,
                report=dict(state="PASS_FACE_KINEMATIC_AUDIT_NOT_OPERATOR", evaluated_states=len(u),
                    registered_nodes_per_face=n, edge_rotation_design=rotation_audit,
                    shared_rotation_radians=angles.T.tolist(), shared_translation_mm=shared_translation.tolist(),
                    raw_group_spread_mm=spread.tolist(), periodic_limits_mm=limits.tolist(),
                    fixed_initial_scale_mm=float(scales[0]), fixed_increment_scales_mm=scales[1:cut].tolist(),
                    fixed_absolute_scales_mm=scales[cut:].tolist(),
                    initial_nonrigid_max_mm=float(np.max(abs(periodic[0]))),
                    gauge="rotation from signed periodic pairs, then one joint-face mean translation",
                    weighting="uniform registered nodes; full-six rigid projection supplies scalar scales only",
                    q6_labels=list(FACE_Q6_LABELS), engineering_shear_and_twist=True,
                    initial_retained_separately=True, raw_projection_or_boundary_zeroing=False,
                    actual_volume_lifts_checked=False, thermal_geometry_validated=False, runtime_admitted=False))


def fit_face_basis(audited, modes):
    """Fit a common face basis with edge-locked rotation and unique translations."""
    import numpy as np
    contract = audited["contract"]
    n, B = contract["node_count"], np.asarray(modes, float)
    if B.ndim != 2 or B.shape[0] != 3*n or not 3 <= B.shape[1] <= 3*n or not np.isfinite(B).all():
        raise ValueError("invalid declared face basis")
    rank = B.shape[1]
    T = np.broadcast_to(np.eye(3) / np.sqrt(n), (n, 3, 3)).reshape(3*n, 3)
    if np.max(abs(B[:, :3]-T)) > 1e-12 or np.max(abs(B.T @ B-np.eye(rank))) > 1e-10:
        raise ValueError("face basis requires matching canonical first-three translations and orthonormal Gram matrix")
    single = B.T.reshape(rank, n, 3)
    both = np.concatenate((single, single), axis=1)
    if np.any(face_periodic_spread(both, contract) > 1e-12):
        raise ValueError("raw face basis columns are not homogeneous periodic")
    # Retain differential translations; remove only duplicate shared translations.
    ports = np.zeros((6*n, 2*rank-3))
    ports[:3*n, :3], ports[3*n:, :3] = T, -T
    ports[:3*n, 3:rank] = B[:, 3:]
    ports[3*n:, rank:] = B[:, 3:]
    rigid = contract["rigid_traces"].reshape(6*n, 6)
    full_design = np.column_stack((ports, rigid))
    _, design_audit = face_checked_lstsq(full_design, np.zeros((6*n, 1)))
    free_design = np.column_stack((ports, rigid[:, :3]))
    source = audited["all_periodic_mm"]
    coefficients, fit_audit = face_checked_lstsq(free_design, source.reshape(len(source), -1).T)
    residual = (source.reshape(len(source), -1)-coefficients.T @ free_design.T).reshape(source.shape)
    maxima = np.max(abs(residual), axis=(1, 2))
    scales = audited["fixed_scales_mm"]
    limits = FACE_ATOL_MM + FACE_RTOL * scales
    pbc = face_periodic_spread(residual, contract)
    pbc_limits = FACE_PBC_ATOL_MM + FACE_PBC_RTOL * scales
    port_coefficients = np.zeros((len(source), 2, rank))
    port_coefficients[:, 0, :3] = coefficients[:3].T
    port_coefficients[:, 1, :3] = -coefficients[:3].T
    port_coefficients[:, 0, 3:] = coefficients[3:rank].T
    port_coefficients[:, 1, 3:] = coefficients[rank:2*rank-3].T
    return dict(residual_mm=residual, port_coefficients=port_coefficients,
                shared_translation_mm=coefficients[-3:].T,
                report=dict(state="EVALUATED_FACE_BASIS_NOT_OPERATOR", port_rank_per_face=rank,
                    quotient_design=design_audit, rotation_locked_fit=fit_audit,
                    maximum_face_error_mm=maxima.tolist(), face_limits_mm=limits.tolist(),
                    periodic_residual_spread_mm=pbc.tolist(), periodic_limits_mm=pbc_limits.tolist(),
                    face_pass=bool(np.all(maxima <= limits)), periodic_pass=bool(np.all(pbc <= pbc_limits)),
                    translation_quotient="a_bottom[j]+a_top[j]=0 for verified canonical translation modes j=0,1,2",
                    initial_increment_absolute_all_evaluated=True, rotation_locked_to_edges=True,
                    actual_volume_lifts_checked=False, thermal_geometry_validated=False, runtime_admitted=False))


def transfer_face_seed(source_coordinates, source_modes, target_coordinates):
    """Transfer the R24 geometric prior, not a mechanics operator or old coefficients."""
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator
    coordinates, source = np.asarray(source_coordinates, float), np.asarray(source_modes, float)
    if (coordinates.ndim != 2 or coordinates.shape[1] != 2 or source.ndim != 2
            or source.shape[0] != 3*len(coordinates) or source.shape[1] < 24
            or not np.isfinite(coordinates).all() or not np.isfinite(source).all()):
        raise ValueError("invalid finite R24 face seed arrays")
    source = source[:, :24].copy()
    source_xyz = np.asarray([np.column_stack((coordinates[:, 0], np.full(len(coordinates), y), coordinates[:, 1]))
                             for y in (0., 1.)])
    old_contract, contract = face_kinematic_contract(source_xyz), face_kinematic_contract(target_coordinates)
    n0, n = len(coordinates), contract["node_count"]
    T0 = np.broadcast_to(np.eye(3)/np.sqrt(n0), (n0, 3, 3)).reshape(-1, 3)
    if np.max(abs(source[:, :3]-T0)) > 1e-12 or np.max(abs(source.T@source-np.eye(24))) > 1e-10:
        raise ValueError("source R24 translation/Gram contract changed")
    old_fields = source.T.reshape(24, n0, 3)
    old_spread = face_periodic_spread(np.concatenate((old_fields, old_fields), axis=1), old_contract)
    if np.max(old_spread) > FACE_SEED_ROUNDOFF_LIMIT:
        raise ValueError("source seed periodicity exceeds the declared roundoff-only bound")
    x, z = (np.unique(coordinates[:, i]) for i in range(2))
    ix, iz = np.searchsorted(x, coordinates[:, 0]), np.searchsorted(z, coordinates[:, 1])
    values = np.empty((len(x), len(z), 3, 24))
    values[ix, iz] = source.reshape(n0, 3, 24)
    physical = contract["coordinates_mm"][0][:, (0, 2)]
    mapped = (physical-physical.min(0))/np.ptp(physical, axis=0)*np.ptp(coordinates, axis=0)+coordinates.min(0)
    fields = RegularGridInterpolator((x, z), values, bounds_error=True)(mapped)
    raw = fields.copy().reshape(3*n, 24)
    spread = face_periodic_spread(np.concatenate((fields.transpose(2, 0, 1),)*2, axis=1), contract)
    if np.max(spread) > FACE_SEED_ROUNDOFF_LIMIT:
        raise ValueError("transferred seed periodicity exceeds the roundoff-only bound")
    for group in contract["periodic_groups"]:
        if np.all(group < n):
            fields[group] = fields[group].mean(axis=0)
    corrected = fields.reshape(3*n, 24)
    correction = float(np.max(abs(corrected-raw)))
    if correction > FACE_SEED_ROUNDOFF_LIMIT:
        raise ValueError("seed periodic correction exceeds its frozen roundoff bound")
    T = np.broadcast_to(np.eye(3)/np.sqrt(n), (n, 3, 3)).reshape(-1, 3)
    remaining = corrected[:, 3:]-T@(T.T@corrected[:, 3:])
    left, singular, _ = np.linalg.svd(remaining, full_matrices=False)
    if len(singular) < 21 or singular[20] <= singular[0]*1e-12:
        raise ValueError("transferred R24 seed lost physical span rank")
    left = left[:, :21]
    signs = np.sign(left[np.argmax(abs(left), axis=0), np.arange(21)])
    modes = np.column_stack((T, left*signs))
    error = float(np.linalg.norm(raw-modes@(modes.T@raw))/np.linalg.norm(raw))
    if error > 1e-10 or np.max(abs(modes.T@modes-np.eye(24))) > 1e-10:
        raise ValueError("roundoff-cleaned seed failed its raw-span/Gram audit")
    return dict(coords_mm=physical.copy(), modes=modes,
                report=dict(state="TRANSFERRED_FACE_SEED_NOT_OPERATOR", rank=24,
                    source_nodes=n0, target_nodes=n, source_periodic_spread=old_spread.tolist(),
                    raw_transferred_periodic_spread=spread.tolist(), periodic_correction_max=correction,
                    periodic_correction_limit=FACE_SEED_ROUNDOFF_LIMIT, raw_span_relative_error=error,
                    retained_singular_values=singular[:21].tolist(), coefficient_convention_changed=True,
                    old_lifts_or_coefficients_reusable=False, mechanics_operator_interpolation=False,
                    volume_lifts_qualified=False, thermal_geometry_validated=False, runtime_admitted=False))


FACE_POD_RANKS = (24, 28, 32, 36, 40, 48, 64, 96, 128)
FACE_POD_RULES = dict(units="mm", weighting="unweighted raw displacement columns",
    absolute_resolution_mm=1e-12, relative_resolution=1e-12,
    periodic_projection_atol_mm=1e-12,
    construction="periodic isometric quotient SVD; not preservation of previous dense-SVD span",
    adjacent_degeneracy_relative_gap=1e-10, anchor_relative_tie=1e-12,
    sector_order=[[1, 1], [1, -1], [-1, 1], [-1, -1]],
    ordering="block maximum singular value, sector index, block index; complete block canonicalized before slicing")


def face_transform(contract, fields, *, reflection=False, swap=False):
    import numpy as np
    values = np.asarray(fields, float)
    n = contract["node_count"]
    if values.ndim != 2 or values.shape[0] != 3*n or not np.isfinite(values).all():
        raise ValueError("invalid common face transformation fields")
    values = values.reshape(n, 3, -1).copy()
    if reflection:
        values[:, 1] *= -1
    if swap:
        xz = contract["coordinates_mm"][0][:, (0, 2)]
        ix, iz = (np.searchsorted(np.unique(xz[:, k]), xz[:, k]) for k in (0, 1))
        side = len(np.unique(ix))
        grid = np.empty((side, side), dtype=int)
        grid[ix, iz] = np.arange(n)
        values = values[grid[iz, ix]][:, [2, 1, 0]]
    return values.reshape(3*n, -1)


def face_periodic_quotient(contract):
    """An orthonormal-column map retains the raw-node metric, including corners."""
    import numpy as np
    checked = face_kinematic_contract(contract["coordinates_mm"])
    xz = checked["coordinates_mm"][0][:, (0, 2)]
    ix, iz = (np.searchsorted(np.unique(xz[:, k]), xz[:, k]) for k in (0, 1))
    period = len(np.unique(ix))-1
    nodes = (ix % period)*period + iz % period
    multiplicity = np.bincount(nodes, minlength=period**2)
    if not np.isin(multiplicity, (1, 2, 4)).all():
        raise ValueError("incomplete periodic face quotient")
    row_ids = (nodes[:, None]*3+np.arange(3)).reshape(-1)
    weights = np.repeat(1/np.sqrt(multiplicity[nodes]), 3)
    grid = np.empty((period+1, period+1), dtype=int)
    grid[ix, iz] = np.arange(len(nodes))
    raw_swap = (3*grid[iz, ix, None]+np.asarray([2, 1, 0])).reshape(-1)
    _, representatives = np.unique(row_ids, return_index=True)
    quotient_swap = row_ids[raw_swap[representatives]]
    if (not np.array_equal(quotient_swap[row_ids], row_ids[raw_swap])
            or not np.array_equal(quotient_swap[quotient_swap], np.arange(3*period**2))
            or not np.array_equal(weights, weights[raw_swap])):
        raise ValueError("periodic quotient does not commute with physical in-plane swap")
    error = float(np.max(abs(multiplicity*(1/np.sqrt(multiplicity))**2-1)))
    if error > 1e-14:
        raise ValueError("periodic quotient lost its raw-node isometry")
    return dict(row_ids=row_ids, weights=weights, columns=3*period**2,
        report=dict(raw_dofs=len(row_ids), quotient_dofs=3*period**2,
            multiplicity_counts={str(k): int(np.count_nonzero(multiplicity == k)) for k in (1, 2, 4)},
            column_isometry_max_error=error, reflection_commutes=True, swap_commutes=True,
            mapping_sha256=hashlib.sha256(row_ids.tobytes()+weights.tobytes()).hexdigest(),
            weighting="each group column has entries1/sqrt(multiplicity); singleton interiors included"))


def face_periodic_reduce(quotient, fields):
    import numpy as np
    values = np.asarray(fields, float)
    if values.ndim != 2 or values.shape[0] != len(quotient["row_ids"]) or not np.isfinite(values).all():
        raise ValueError("invalid periodic quotient input fields")
    reduced = np.zeros((quotient["columns"], values.shape[1]))
    np.add.at(reduced, quotient["row_ids"], quotient["weights"][:, None]*values)
    return reduced


def face_periodic_expand(quotient, fields):
    import numpy as np
    values = np.asarray(fields, float)
    if values.ndim != 2 or values.shape[0] != quotient["columns"] or not np.isfinite(values).all():
        raise ValueError("invalid periodic quotient coordinates")
    return quotient["weights"][:, None]*values[quotient["row_ids"]]


def face_symmetry_audit(contract, modes):
    import numpy as np
    B = np.asarray(modes, float)
    n = contract["node_count"]
    if B.ndim != 2 or B.shape[0] != 3*n or B.shape[1] < 3 or not np.isfinite(B).all():
        raise ValueError("invalid common face symmetry basis")
    T = np.broadcast_to(np.eye(3)/np.sqrt(n), (n, 3, 3)).reshape(3*n, 3)
    F, S = face_transform(contract, B, reflection=True), face_transform(contract, B, swap=True)
    RF, RS = B.T@F, B.T@S
    fields = B.T.reshape(B.shape[1], n, 3)
    spread = face_periodic_spread(np.concatenate((fields, fields), axis=1), contract)
    metrics = dict(gram_max_error=float(np.max(abs(B.T@B-np.eye(B.shape[1])))),
        translation_max_error=float(np.max(abs(B[:, :3]-T))),
        reflection_closure_max=float(np.max(abs(F-B@RF))), swap_closure_max=float(np.max(abs(S-B@RS))),
        reflection_involution_max=float(np.max(abs(RF@RF-np.eye(B.shape[1])))),
        swap_involution_max=float(np.max(abs(RS@RS-np.eye(B.shape[1])))),
        commutator_max=float(np.max(abs(RF@RS-RS@RF))), periodic_spread_max=float(np.max(spread)))
    if any(v > (1e-12 if k in ("translation_max_error", "periodic_spread_max") else 1e-10)
           for k, v in metrics.items()):
        raise ValueError(f"common face basis failed physical symmetry/Gram/periodicity audit: {metrics}")
    return dict(reflection_matrix=RF, inplane_swap_matrix=RS, report=metrics)


def canonical_face_block(vectors):
    """Coordinate anchors fix a complete degenerate subspace before rank slicing."""
    import numpy as np
    U = np.asarray(vectors, float)
    if (U.ndim != 2 or not 0 < U.shape[1] <= U.shape[0] or not np.isfinite(U).all()
            or np.max(abs(U.T@U-np.eye(U.shape[1]))) > 1e-10):
        raise ValueError("invalid orthonormal face singular subspace")
    result, anchors = [], []
    for _ in range(U.shape[1]):
        W = np.column_stack(result) if result else np.empty((len(U), 0))
        leverage = np.sum(U**2, axis=1)-np.sum(W**2, axis=1)
        maximum = float(np.max(leverage))
        if maximum <= np.finfo(float).eps * max(1, U.shape[1]):
            raise ValueError("unresolved face canonicalization anchor")
        index = int(np.flatnonzero(leverage >= maximum*(1-FACE_POD_RULES["anchor_relative_tie"]))[0])
        value = U@U[index]
        for _pass in range(2):
            value -= W@(W.T@value)
        norm = np.linalg.norm(value)
        if norm <= np.finfo(float).eps:
            raise ValueError("empty face canonicalization direction")
        result.append(value/norm)
        anchors.append(index)
    return np.column_stack(result), anchors


def pack_face_training(evidences):
    """Keep raw audited states for evaluation; only POD copies receive edge averaging."""
    import numpy as np
    if not isinstance(evidences, (list, tuple)) or not 1 <= len(evidences) <= 1024:
        raise ValueError("invalid bounded common-face training pool")
    histories, ledger, columns, seen, shared_xz = [], [], [], set(), None
    geometry_paths = {}
    for evidence in evidences:
        spec = evidence["geometry"]
        if (spec != training_spec(spec.get("pitch_um"), spec.get("layer"), spec.get("si_um"), spec.get("path"), spec.get("nx"))
                or spec.get("role") != "training" or spec.get("path") not in TRAINING_PATHS
                or tuple(evidence.get("q6_labels", ())) != FACE_Q6_LABELS
                or evidence.get("engineering_shear_and_twist") is not True
                or evidence.get("thermal_geometry_validated") is not False
                or evidence.get("runtime_admitted") is not False):
            raise ValueError("only declared unqualified training histories may construct a face basis")
        receipt_sha = training_hash(evidence["receipt_sha256"])
        geometry_key = json.dumps({k: v for k, v in spec.items() if k not in ("path", "role")}, sort_keys=True)
        key = (geometry_key, spec["path"])
        if key in seen:
            raise ValueError("duplicate common-face geometry/path evidence")
        seen.add(key)
        geometry_paths.setdefault(geometry_key, set()).add(spec["path"])
        arrays = evidence["arrays"]
        ids = np.stack(evidence["face_indices"])
        registered = np.stack(training_face_registration(arrays["coordinates_mm"], spec))
        if not np.array_equal(ids, registered):
            raise ValueError("common-face evidence registration differs from declared geometry")
        xyz = arrays["coordinates_mm"][ids]
        xz = xyz[0][:, (0, 2)]
        if shared_xz is not None and not np.array_equal(xz, shared_xz):
            raise ValueError("common-face training requires identical physical x/z coordinates and node order")
        shared_xz = xz.copy()
        audited = audit_face_history(xyz, arrays["displacements_mm"][:, ids], evidence["q6_centered"])
        n = audited["contract"]["node_count"]
        fields = audited["all_periodic_mm"].copy()
        raw = fields.copy()
        for group in audited["contract"]["periodic_groups"]:
            fields[:, group] = fields[:, group].mean(axis=1, keepdims=True)
        correction = np.max(abs(fields-raw), axis=(1, 2))
        limits = np.asarray(audited["report"]["periodic_limits_mm"])
        if np.any(correction > limits):
            raise ValueError("POD-only periodic correction exceeds frozen source-local limits")
        count = len(evidence["q6_centered"])
        block = fields.reshape(len(fields), 2, n, 3).transpose(2, 3, 1, 0).reshape(3*n, -1)
        columns.append(block)
        histories.append(audited)
        ledger.append(dict(receipt_sha256=receipt_sha, geometry=spec, states=count, columns=block.shape[1],
            column_order="bottom initial/increments/absolute, then top initial/increments/absolute",
            periodic_correction_mm_by_state=correction.tolist(), periodic_correction_limits_mm=limits.tolist(),
            raw_source_scales_mm=audited["fixed_scales_mm"].tolist(), raw_evidence_used_for_evaluation=True))
    missing = {k: sorted(set(TRAINING_PATHS)-paths) for k, paths in geometry_paths.items()}
    return dict(columns_mm=np.concatenate(columns, axis=1), histories=histories,
                report=dict(source_ledger=ledger, missing_training_paths=missing,
                    training_path_coverage_complete=not any(missing.values()), family_coverage_complete=False,
                    final_holdouts_read=False, geometry_support_inferred=False,
                    volume_lifts_qualified=False, thermal_geometry_validated=False, runtime_admitted=False))


def face_pod_catalog(contract, seed, columns, ranks=FACE_POD_RANKS):
    """One nested canonical block catalog; unsupported high ranks leave lower ranks usable."""
    import numpy as np
    B, Y = np.asarray(seed, float), np.asarray(columns, float)
    if (B.ndim != 2 or B.shape[1] != 24 or Y.ndim != 2 or Y.shape[0] != B.shape[0]
            or not Y.shape[1] or not np.isfinite(Y).all()
            or not isinstance(ranks, (tuple, list)) or not 1 <= len(ranks) <= 32
            or any(type(r) is not int or not 24 <= r <= 512 for r in ranks)
            or sorted(set(ranks)) != list(ranks)):
        raise ValueError("invalid R24 common-face POD inputs or ascending candidate ranks")
    seed_audit = face_symmetry_audit(contract, B)
    n = contract["node_count"]
    shaped = Y.T.reshape(Y.shape[1], n, 3)
    if np.max(face_periodic_spread(np.concatenate((shaped, shaped), axis=1), contract)) > 1e-12:
        raise ValueError("POD columns must already be admitted homogeneous-periodic construction copies")
    remaining = Y-B@(B.T@Y)
    reflected = face_transform(contract, remaining, reflection=True)
    swapped = face_transform(contract, remaining, swap=True)
    both = face_transform(contract, reflected, swap=True)
    sectors, projection_audits = [], []
    quotient = face_periodic_quotient(contract)
    for f, s in FACE_POD_RULES["sector_order"]:
        sector = .25*(remaining+f*reflected+s*swapped+f*s*both)
        sector -= B@(B.T@sector)
        reduced = face_periodic_reduce(quotient, sector)
        residual = sector-face_periodic_expand(quotient, reduced)
        maximum = float(np.max(abs(residual)))
        projection_audits.append(dict(sector=[f, s], maximum_residual_mm=maximum,
            spectral_norm_upper_bound_mm=float(np.linalg.norm(residual)),
            spectral_bound_method="Frobenius norm; not the exact spectral norm",
            projection_limit_mm=FACE_POD_RULES["periodic_projection_atol_mm"],
            projection_pass=maximum <= FACE_POD_RULES["periodic_projection_atol_mm"]))
        left, singular, _ = np.linalg.svd(reduced, full_matrices=False)
        left = face_periodic_expand(quotient, left)
        sectors.append((left, singular, (f, s)))
    leading = max((float(s[0]) for _, s, _ in sectors if len(s)), default=0.)
    cutoff = FACE_POD_RULES["absolute_resolution_mm"]+FACE_POD_RULES["relative_resolution"]*leading
    families, excluded = [], []
    for sector_index, (left, singular, parity) in enumerate(sectors):
        start, block_index = 0, 0
        while start < len(singular):
            stop = start+1
            while (stop < len(singular) and abs(singular[stop-1]-singular[stop]) <=
                   FACE_POD_RULES["adjacent_degeneracy_relative_gap"]*max(singular[stop-1], singular[stop])):
                stop += 1
            if singular[stop-1] > cutoff:
                canonical, anchors = canonical_face_block(left[:, start:stop])
                families.append((float(singular[start]), sector_index, block_index, canonical,
                    dict(sector=list(parity), block_index=block_index, dimension=stop-start, anchors=anchors,
                         singular_max_mm=float(singular[start]), singular_min_mm=float(singular[stop-1]),
                         construction_pass=projection_audits[sector_index]["projection_pass"],
                         projection_spectral_upper_bound_over_sigma_min=(
                             projection_audits[sector_index]["spectral_norm_upper_bound_mm"]/float(singular[stop-1])))))
            elif singular[start] > cutoff:
                excluded.append(dict(sector=list(parity), block_index=block_index, reason="degeneracy_block_crosses_resolution_cutoff"))
            start, block_index = stop, block_index+1
    families.sort(key=lambda item: (-item[0], item[1], item[2]))
    available_rank = 24+sum(item[3].shape[1] for item in families)
    modes = np.column_stack([B, *(item[3] for item in families)])
    metadata, offset = [], 24
    for item in families:
        metadata.append(dict(first_mode=offset, **item[4]))
        offset += item[3].shape[1]
    candidates = []
    for rank in ranks:
        row = dict(rank=rank, state="unavailable", reason="insufficient_resolved_directions")
        if rank <= available_rank:
            rejected = [block for block in metadata if block["first_mode"] < rank and not block["construction_pass"]]
            if rejected:
                candidates.append(dict(rank=rank, state="construction_invalid",
                    reason="prefix includes a rejected periodic-sector projection; no backfill",
                    rejected_block_positions=[block["first_mode"] for block in rejected]))
                continue
            try:
                audit = face_symmetry_audit(contract, modes[:, :rank])
                row = dict(rank=rank, state="audited", symmetry=audit["report"])
            except ValueError as error:
                row = dict(rank=rank, state="audit_invalid", reason=str(error))
        candidates.append(row)
    return dict(modes=modes[:, :min(max(ranks), available_rank)], candidates=candidates,
        report=dict(state="FACE_POD_CATALOG_NOT_OPERATOR", available_resolved_rank=available_rank,
                    numerical_rules=json.loads(json.dumps(FACE_POD_RULES)), global_sector_leading_sigma_mm=leading, resolution_cutoff_mm=cutoff,
                    canonical_blocks=metadata, excluded_blocks=excluded, seed_symmetry=seed_audit["report"],
                    periodic_quotient=quotient["report"], sector_projection_audits=projection_audits,
                    previous_dense_svd_span_preservation_claimed=False,
                    exact_ordered_pod_optimality_claimed=False, family_rank_selected=False,
                    final_holdouts_read=False, thermal_geometry_validated=False, runtime_admitted=False))


def evaluate_face_catalog(catalog, pool):
    candidates, prototype_minimum = [], None
    for original in catalog["candidates"]:
        row = dict(original)
        if row["state"] == "audited":
            try:
                reports = [fit_face_basis(h, catalog["modes"][:, :row["rank"]])["report"] for h in pool["histories"]]
                passed = all(r["face_pass"] and r["periodic_pass"] for r in reports)
                row.update(state="pass" if passed else "gate_failed", source_fits=reports)
                if passed and prototype_minimum is None:
                    prototype_minimum = row["rank"]
            except ValueError as error:
                row.update(state="audit_invalid", reason=str(error))
        candidates.append(row)
    return dict(state="PROTOTYPE_FACE_RANK_DIAGNOSTIC_NOT_FAMILY_SELECTION", candidates=candidates,
                prototype_minimum_rank=prototype_minimum, family_rank_selected=False,
                family_coverage_complete=False, source_pool=pool["report"], pod=catalog["report"],
                final_holdouts_read=False, volume_lifts_qualified=False,
                thermal_geometry_validated=False, runtime_admitted=False)


def training_face_audit(directory, receipt_sha256):
    import numpy as np
    evidence = load_training_evidence(directory, receipt_sha256)
    ids = np.stack(evidence["face_indices"])
    arrays = evidence["arrays"]
    audited = audit_face_history(arrays["coordinates_mm"][ids], arrays["displacements_mm"][:, ids],
                                 evidence["q6_centered"])
    return dict(geometry=evidence["geometry"], receipt_sha256=receipt_sha256, **audited["report"])


def training_resources(work, spec):
    resource = builder.resource_check(work)
    if resource["own_ansys_processes"] or ansys_blockers() or resource["local_free_gib"] < 200 + spec["reservation_gib"]:
        raise RuntimeError("training reservation or own-UID ANSYS idle gate failed")
    return resource


class AllocationNamespaceChanged(RuntimeError):
    pass


def local_allocation_snapshot(work, *, identities=(), allowed_link_pairs=(),
                              entry_limit=512, depth_limit=8, seconds=2.):
    """Bounded local accounting only; it grants no launch or resource authority."""
    work = builder.local_scratch(work)
    if (type(entry_limit) is not int or not 1 <= entry_limit <= 4096
            or type(depth_limit) is not int or not 0 <= depth_limit <= 16
            or type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 15
            or not isinstance(identities, (tuple, list)) or len(identities) > 64
            or not isinstance(allowed_link_pairs, (tuple, list)) or len(allowed_link_pairs) > 64):
        raise ValueError("invalid finite allocation inventory bounds")
    identities = tuple(json.loads(json.dumps(identity)) for identity in identities)
    allowed = set()
    for pair in allowed_link_pairs:
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2 or pair[0] == pair[1]
                or any(not isinstance(name, str) or not name or len(name) > 4096 or Path(name).is_absolute()
                       or any(part in ("..", ".snapshot", "#snapshot") for part in Path(name).parts) for name in pair)):
            raise ValueError("copy link exception requires two explicit safe relative names")
        allowed.add(frozenset(str(Path(name)) for name in pair))
    before = work.stat(follow_symlinks=False)
    if (before.st_uid != os.getuid() or not stat.S_ISDIR(before.st_mode)
            or before.st_dev != builder.scratch_root(builder.physical_host()).stat().st_dev):
        raise ValueError("allocation root is not an owned local directory")
    root_id = (before.st_dev, before.st_ino)
    deadline = time.monotonic()+seconds
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    last_error = None
    for _attempt in range(3):
        records, count = {}, [0]
        def bound():
            if time.monotonic() > deadline:
                raise TimeoutError("allocation inventory deadline exceeded")
        def record(name, info, *, deleted=False):
            if info.st_uid != os.getuid() or info.st_dev != root_id[0]:
                raise ValueError("allocation inventory leaves owned local filesystem")
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise ValueError("allocation inventory contains a symlink or special file")
            key = (info.st_dev, info.st_ino)
            value = records.setdefault(key, dict(device=key[0], inode=key[1],
                kind="directory" if stat.S_ISDIR(info.st_mode) else "file", names=[],
                logical_bytes=0, allocated_bytes=0, links=0, open_deleted=False))
            if name not in value["names"]:
                value["names"].append(name)
            value["logical_bytes"] = max(value["logical_bytes"], info.st_size)
            value["allocated_bytes"] = max(value["allocated_bytes"], info.st_blocks*512)
            value["links"] = max(value["links"], info.st_nlink)
            value["open_deleted"] |= deleted
        def walk(fd, relative, depth):
            bound()
            record(relative or ".", os.fstat(fd))
            with os.scandir(fd) as entries:
                for entry in entries:
                    bound()
                    count[0] += 1
                    if count[0] > entry_limit or entry.name in (".snapshot", "#snapshot"):
                        raise ValueError("allocation entry/snapshot bound exceeded")
                    initial = entry.stat(follow_symlinks=False)
                    if not (stat.S_ISREG(initial.st_mode) or stat.S_ISDIR(initial.st_mode)):
                        raise ValueError("allocation inventory contains a symlink or special file")
                    child = os.open(entry.name, flags | (os.O_DIRECTORY if stat.S_ISDIR(initial.st_mode) else 0), dir_fd=fd)
                    try:
                        info = os.fstat(child)
                        if (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode)) != (
                                initial.st_dev, initial.st_ino, stat.S_IFMT(initial.st_mode)):
                            raise AllocationNamespaceChanged("allocation namespace changed during sample")
                        name = str(Path(relative)/entry.name)
                        record(name, info)
                        if stat.S_ISDIR(info.st_mode):
                            if depth >= depth_limit:
                                raise ValueError("allocation directory depth exceeded")
                            walk(child, name, depth+1)
                    finally:
                        os.close(child)
        try:
            root_fd = os.open(work, flags | os.O_DIRECTORY)
            try:
                info = os.fstat(root_fd)
                if (info.st_dev, info.st_ino) != root_id:
                    raise ValueError("allocation root identity changed")
                walk(root_fd, "", 0)
                record(".", os.fstat(root_fd))
            finally:
                os.close(root_fd)
            for value in records.values():
                if value["kind"] == "file" and value["links"] > 1:
                    names = frozenset(value["names"])
                    if value["links"] != 2 or names not in allowed:
                        if value["links"] == 2 and any(names < pair for pair in allowed):
                            raise AllocationNamespaceChanged("authorized copy link transition is incomplete")
                        raise ValueError("unexplained allocation hard link")
            fd_count, exited = 0, []
            for identity in identities:
                bound()
                pid = identity.get("pid") if isinstance(identity, dict) else None
                if type(pid) is not int or pid <= 0 or not identity.get("start_ticks"):
                    raise ValueError("allocation descriptors lack an immutable process identity")
                current = process_identity(pid)
                if current is None:
                    exited.append(pid)
                    continue
                if current != identity:
                    raise ValueError("allocation descriptor process identity changed")
                with os.scandir(f"/proc/{pid}/fd") as entries:
                    for entry in entries:
                        bound()
                        fd_count += 1
                        if fd_count > 4096:
                            raise ValueError("allocation descriptor enumeration exceeded bound")
                        info = entry.stat(follow_symlinks=True)
                        if stat.S_ISREG(info.st_mode) and info.st_dev == root_id[0] and info.st_nlink == 0:
                            target = os.readlink(entry.path)
                            prefix = str(work)+os.sep
                            if not target.startswith(prefix) or not target.endswith(" (deleted)"):
                                raise ValueError("deleted local descriptor lies outside allocation root")
                            name = target[len(prefix):-len(" (deleted)")]
                            if any(part in ("..", ".snapshot", "#snapshot") for part in Path(name).parts):
                                raise ValueError("unsafe deleted allocation descriptor")
                            record(name, info, deleted=True)
                if process_identity(pid) != identity:
                    raise AllocationNamespaceChanged("process exited during allocation descriptor sample")
            bound()
            after = work.stat(follow_symlinks=False)
            if (after.st_dev, after.st_ino) != root_id:
                raise ValueError("allocation root identity changed")
            values = list(records.values())
            return dict(logical_bytes=sum(v["logical_bytes"] for v in values),
                allocated_bytes=sum(v["allocated_bytes"] for v in values),
                free_bytes=shutil.disk_usage(work).free, root_identity=list(root_id),
                entries=count[0], descriptor_entries=fd_count, exited_identity_pids=exited,
                open_deleted_inodes=sum(v["open_deleted"] for v in values), files=values,
                sample_coverage="bounded visible tree plus supplied identities' deleted-open local files",
                process_ownership_authority_inferred=False, launch_authorized=False)
        except (FileNotFoundError, AllocationNamespaceChanged) as error:
            last_error = error
            bound()
    raise RuntimeError(f"allocation namespace did not settle within bounded retries: {last_error}")


class TrainingAllocationSampler:
    """Inactive collector primitive; the lifecycle owner must enforce check()."""

    def __init__(self, sample, *, period=.5, stale_seconds=5.,
                 usage_bytes=8*1024**3, minimum_free_bytes=208*1024**3):
        if (not callable(sample) or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                for v in (period, stale_seconds)) or period >= stale_seconds
                or any(type(v) is not int or v < 0 for v in (usage_bytes, minimum_free_bytes))):
            raise ValueError("invalid sampled allocation guard limits")
        self.sample, self.period, self.stale_seconds = sample, period, stale_seconds
        self.usage_bytes, self.minimum_free_bytes = usage_bytes, minimum_free_bytes
        self._lock, self._stop, self._ready = threading.Lock(), threading.Event(), threading.Event()
        self._thread, self._latest, self._error, self._closed = None, None, None, False
        self._stats = dict(samples=0, errors=0, maximum_logical_bytes=0, maximum_allocated_bytes=0,
                           minimum_free_bytes=None, maximum_gap_seconds=0., maximum_sample_seconds=0.)

    def _fail(self, message):
        with self._lock:
            self._error = self._error or str(message)[:1000]

    def _run(self):
        while not self._stop.is_set():
            start = time.monotonic()
            try:
                raw = self.sample()
                values = {k: raw[k] for k in ("logical_bytes", "allocated_bytes", "free_bytes")}
                if any(type(v) is not int or v < 0 for v in values.values()):
                    raise ValueError("allocation sample has invalid byte counters")
                end = time.monotonic()
                with self._lock:
                    gap = end-self._latest["completed_monotonic"] if self._latest else end-start
                    self._stats["samples"] += 1
                    for key in ("logical_bytes", "allocated_bytes"):
                        name = "maximum_"+key
                        self._stats[name] = max(self._stats[name], values[key])
                    old = self._stats["minimum_free_bytes"]
                    self._stats["minimum_free_bytes"] = min(values["free_bytes"], old) if old is not None else values["free_bytes"]
                    self._stats["maximum_gap_seconds"] = max(self._stats["maximum_gap_seconds"], gap)
                    self._stats["maximum_sample_seconds"] = max(self._stats["maximum_sample_seconds"], end-start)
                    self._latest = dict(values, started_monotonic=start, completed_monotonic=end,
                                        sequence=self._stats["samples"])
                    if max(values["logical_bytes"], values["allocated_bytes"]) > self.usage_bytes:
                        self._error = self._error or "sampled working allocation exceeded experimental budget"
                    if values["free_bytes"] < self.minimum_free_bytes:
                        self._error = self._error or "sampled filesystem headroom below experimental guard"
                    if max(gap, end-start) > self.stale_seconds:
                        self._error = self._error or "allocation sample or completed-sample gap exceeded freshness bound"
            except BaseException as error:
                self._fail(f"{type(error).__name__}: {error}")
                with self._lock:
                    self._stats["errors"] += 1
            self._ready.set()
            self._stop.wait(self.period)

    def start(self):
        if self._thread is not None or self._closed:
            raise RuntimeError("allocation sampler cannot restart")
        self._thread = threading.Thread(target=self._run, name="training-allocation-sampler", daemon=True)
        self._thread.start()
        if not self._ready.wait(self.stale_seconds):
            self._fail("allocation sampler first valid sample timed out")
        return self.check()

    def check(self):
        if self._closed or self._thread is None or not self._thread.is_alive():
            self._fail("allocation sampler is not live")
        with self._lock:
            if self._latest is None or time.monotonic()-self._latest["completed_monotonic"] > self.stale_seconds:
                self._error = self._error or "allocation sampler completed evidence is missing or stale"
            if self._error:
                raise RuntimeError(self._error)
            return dict(self._latest)

    def report(self):
        with self._lock:
            return dict(self._stats, error=self._error, latest=dict(self._latest) if self._latest else None,
                continuous_peak_bound_claimed=False, launch_authorized=False)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.stale_seconds)
            if self._thread.is_alive():
                self._fail("allocation sampler cannot prove thread exit")
                raise RuntimeError("allocation sampler cannot prove thread exit")
        self._closed = True


RECOVERY_METADATA = ("TRAINING_PLAN.json", "SOURCE_PINS.json", "LAUNCH.json", "STATUS.json",
                     "PROCESS_SCOPE.json", "PREFLIGHT.json", "MAPDL_COMPLETION.json", "SOLVE.dat",
                     "PRESCRIBED_PATH.csv")
RECOVERY_RAW = tuple("solver/" + name for name in (
    "yap_training_c3.rdb", "yap_training_c3.rst", "yap_training_c3.dat", "date_all_stage_macro.csv",
    "free_generalized_macro_substeps.csv", "free_generalized_initial_macro.csv"))
RECOVERY_ARTIFACTS = ("RECOVERY_PLAN.json", "RECOVERY_PINS.json", "RECOVERY_RAW_INPUTS.json",
                      "ALL_STAGE_MACRO.csv", "RAW_ENDPOINT_MACRO.csv", "DENSE_MACRO_RAW.csv", "RAW_INITIAL_MACRO.csv",
                      "RST_CLOCK.csv", "EXTRACT.dat", "EXTRACTION_COMPLETION.json",
                      "ORIGINAL_LAUNCH.json", "ORIGINAL_STATUS.json", "ORIGINAL_PROCESS_SCOPE.json")


def training_source_exit(work, plan_sha256, stop_sha256):
    plan, _ = read_record(work / "TRAINING_PLAN.json", plan_sha256)
    stop, _ = read_record(work / "STATUS.json", stop_sha256)
    launch, launch_sha = read_record(work / "LAUNCH.json")
    scope, scope_sha = read_record(work / "PROCESS_SCOPE.json")
    if (plan.get("schema") != "yap.operator_grid.training.v1" or plan.get("work") != str(work)
            or plan.get("host") != builder.physical_host() or stop.get("state") != "STOP_REQUIRES_REVIEW"
            or launch.get("plan_sha256") != plan_sha256
            or launch.get("source_pins_sha256") != plan.get("source_pins_sha256")
            or stop.get("worker_pid") != launch.get("worker_pid")
            or scope.get("schema") != "yap.operator_grid.process_scope.v1"
            or scope.get("worker_identity") != launch.get("worker_identity")
            or scope.get("launch_sha256") != launch_sha or scope.get("live_identities") != []):
        raise ValueError("training recovery lacks a bound stopped source and exited scope")
    pid = launch["worker_pid"]
    if type(pid) is not int or pid <= 0 or launch["worker_identity"].get("pid") != pid:
        raise ValueError("invalid historical training worker identity")
    require_exited_source(pid)
    groups, sessions = {pid}, {pid}
    for name, selected in (("groups", groups), ("sessions", sessions)):
        values = scope.get(name)
        if not isinstance(values, list) or len(values) > 4096 or any(type(v) is not int or v <= 0 for v in values):
            raise ValueError("invalid retained source process scope")
        selected.update(values)
    if any(not row["state"].startswith("Z") and (row["pgid"] in groups or row["sid"] in sessions)
           for row in own_process_census()):
        raise RuntimeError("historical training scope still contains live or orphaned members")
    return dict(launch_sha256=launch_sha, scope_sha256=scope_sha, groups=sorted(groups), sessions=sorted(sessions))


def training_input_seal(path, max_bytes):
    path = regular(path, max_bytes=max_bytes)
    def identity(info):
        return dict(device=info.st_dev, inode=info.st_ino, bytes=info.st_size,
                    mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise ValueError("training input descriptor exceeds the regular-file bound")
        digest = hashlib.sha256()
        while chunk := stream.read(1024**2):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    current = path.stat(follow_symlinks=False)
    if identity(before) != identity(after) or identity(before) != identity(current):
        raise ValueError("training input changed during descriptor hashing")
    return dict(**identity(before), sha256=digest.hexdigest())


def recovery_deadline(plan, reserve=120.):
    deadline = plan["expires_epoch"] - reserve
    if not math.isfinite(deadline) or deadline <= time.time():
        raise TimeoutError("training recovery deadline has expired")
    return deadline


def validate_recovery_evidence(directory, receipt, historical_plan, preflight):
    if not all(name in receipt for name in ("recovery_plan_sha256", "recovery_evidence_sha256")):
        raise ValueError("recovery-tagged evidence lacks its mandatory provenance chain")
    companion, _ = read_record(directory / "RECOVERY_COMPLETE.json", training_hash(receipt["recovery_evidence_sha256"]))
    if (companion.get("schema") != "yap.operator_grid.training_recovery_evidence.v1"
            or companion.get("state") != "TRAINING_EXTRACTED_NOT_OPERATOR"
            or companion.get("recovery_plan_sha256") != receipt["recovery_plan_sha256"]
            or companion.get("source_plan_sha256") != receipt["source_plan_sha256"]
            or companion.get("original_raw_seal_contemporaneous") is not False
            or companion.get("thermal_geometry_validated") is not False or companion.get("runtime_admitted") is not False
            or set(companion.get("files_sha256", {})) != set(RECOVERY_ARTIFACTS)):
        raise ValueError("invalid recovery companion receipt")
    for name, digest in companion["files_sha256"].items():
        if builder.digest(regular(directory / name, max_bytes=2 * 1024**2)) != training_hash(digest):
            raise ValueError(f"recovery companion artifact hash mismatch: {name}")
    plan, _ = read_record(directory / "RECOVERY_PLAN.json", training_hash(receipt["recovery_plan_sha256"]))
    pins, _ = read_record(directory / "RECOVERY_PINS.json", training_hash(plan["recovery_pins_sha256"]))
    current_files = pins.get("files_sha256", {})
    if (plan.get("schema") != "yap.operator_grid.training_recovery.v1" or plan.get("geometry") != receipt["geometry"]
            or plan.get("source_plan_sha256") != receipt["source_plan_sha256"]
            or plan.get("source_pins_sha256") != historical_plan["source_pins_sha256"]
            or plan.get("source_work") != historical_plan["work"] or plan.get("source") != historical_plan["source"]
            or plan.get("original_raw_seal_contemporaneous") is not False or plan.get("automatic_retry") is not False
            or pins.get("schema") != "yap.operator_grid.sources.v1" or not isinstance(current_files, dict)
            or not 1 <= len(current_files) <= 96
            or [h for p, h in current_files.items() if Path(p).name == "operator_grid.py"] != [plan.get("review_sha256")]):
        raise ValueError("recovery and original producer authorities are not separately bound")
    original_pins, _ = read_record(directory / "SOURCE_PINS.json", historical_plan["source_pins_sha256"])
    for path, digest in current_files.items():
        builder.safe_path(path)
        training_hash(digest)
    for name, expected in {**TRAINING_CARDS, "ansys_utils.py": COMMON_HELPER_SHA}.items():
        if [h for p, h in current_files.items() if Path(p).name == name] != [expected]:
            raise ValueError("recovery material/helper source binding changed")
    for path, digest in original_pins["files_sha256"].items():
        if Path(path).is_relative_to(Path(plan["source"]).parent) and current_files.get(path) != digest:
            raise ValueError("recovery scientific sources differ from historical solver sources")
    seals = plan.get("original_files", {})
    if set(seals) != set(RECOVERY_METADATA + RECOVERY_RAW):
        raise ValueError("recovery source inventory is incomplete")
    for name, seal in seals.items():
        training_hash(seal["sha256"])
        if any(type(seal.get(key)) is not int or seal[key] < 0 for key in ("device", "inode", "bytes", "mtime_ns", "ctime_ns")):
            raise ValueError("invalid recovery-time source identity")
        if name in RECOVERY_METADATA:
            target = "ORIGINAL_" + name if name in ("LAUNCH.json", "STATUS.json", "PROCESS_SCOPE.json") else name
            if builder.digest(directory / target) != seal["sha256"]:
                raise ValueError("historical metadata differs from the recovery-time seal")
    raw_inputs, _ = read_record(directory / "RECOVERY_RAW_INPUTS.json")
    if (raw_inputs.get("original_files") != seals or raw_inputs.get("original_raw_seal_contemporaneous") is not False
            or raw_inputs.get("original_source_exit_proof") != plan["source_exit_proof"]):
        raise ValueError("missing retrospective raw-input provenance")
    if (seals["SOLVE.dat"]["sha256"] != seals["solver/yap_training_c3.dat"]["sha256"]
            or plan["source_stop_sha256"] != seals["STATUS.json"]["sha256"]):
        raise ValueError("original recovery stop/actual deck binding mismatch")
    launch, launch_sha = read_record(directory / "ORIGINAL_LAUNCH.json")
    status, _ = read_record(directory / "ORIGINAL_STATUS.json")
    scope, scope_sha = read_record(directory / "ORIGINAL_PROCESS_SCOPE.json")
    if (launch.get("plan_sha256") != receipt["source_plan_sha256"]
            or launch.get("source_pins_sha256") != historical_plan["source_pins_sha256"]
            or status.get("state") != "STOP_REQUIRES_REVIEW" or status.get("worker_pid") != launch.get("worker_pid")
            or scope.get("worker_identity") != launch.get("worker_identity") or scope.get("launch_sha256") != launch_sha
            or scope.get("live_identities") != []
            or plan["source_exit_proof"]["scope_sha256"] != scope_sha
            or plan["source_exit_proof"]["launch_sha256"] != launch_sha):
        raise ValueError("recovery historical scope/stop chain mismatch")
    for key in ("groups", "sessions"):
        values = scope.get(key)
        if (not isinstance(values, list) or any(type(v) is not int or v <= 0 for v in values)
                or plan["source_exit_proof"].get(key) != sorted(set(values) | {launch["worker_pid"]})):
            raise ValueError("recovery recorded exit proof omits original groups/sessions")
    for raw_name, archived in (("date_all_stage_macro.csv", "RAW_ENDPOINT_MACRO.csv"),
                               ("free_generalized_macro_substeps.csv", "DENSE_MACRO_RAW.csv"),
                               ("free_generalized_initial_macro.csv", "RAW_INITIAL_MACRO.csv")):
        if builder.digest(directory / archived) != seals["solver/" + raw_name]["sha256"]:
            raise ValueError("raw recovery macro evidence changed")
    endpoints = training_endpoints(directory / "RAW_ENDPOINT_MACRO.csv", preflight["stages"], receipt["geometry"])
    normalized = training_csv(directory / "ALL_STAGE_MACRO.csv")
    if len(normalized) != len(endpoints):
        raise ValueError("derived recovery endpoints are incomplete")
    for actual, expected in zip(normalized, endpoints, strict=True):
        if set(actual) != set(expected) or actual["stage"] != expected["stage"] or any(
                float(actual[key]) != value for key, value in expected.items() if key != "stage"):
            raise ValueError("derived recovery normalization differs from preserved raw endpoints")
    dense = training_dense_trace(directory / "DENSE_MACRO_RAW.csv", receipt["geometry"])
    selected = training_csv(directory / "SNAPSHOT_MACRO.csv")
    if any({k: float(v) for k, v in row.items()} != dense[s - 1]
           for row, s in zip(selected, range(1, 1082, 10), strict=True)):
        raise ValueError("recovery selected trace differs from its sealed dense evidence")
    extraction, _ = read_record(directory / "EXTRACTION_COMPLETION.json")
    if extraction.get("rc") != 0 or extraction.get("license_error") or extraction.get("ansys_error_count") != 0:
        raise ValueError("recovery extraction completion failed")
    original_completion, _ = read_record(directory / "MAPDL_COMPLETION.json")
    if (original_completion.get("ansys_error_count") != 0 or original_completion.get("subprocess_rc") != 0
            or original_completion.get("timeout") or original_completion.get("lock_conflict")):
        raise ValueError("recovery historical solve completion is not clean")
    audit_recovery_extraction_deck((directory / "EXTRACT.dat").read_text())
    original_clock, _ = read_record(directory / "INITIAL_CLOCK_AUDIT.json")
    raw_seal, _ = read_record(directory / "RAW_SOLVER_SEAL.json")
    if (original_clock["raw_initial_csv_sha256"] != companion["files_sha256"]["RAW_INITIAL_MACRO.csv"]
            or original_clock["endpoint_export_sha256"] != companion["files_sha256"]["ALL_STAGE_MACRO.csv"]
            or any(h != seals["solver/yap_training_c3" + Path(p).suffix]["sha256"]
                   for p, h in raw_seal["files_sha256"].items())):
        raise ValueError("recovery initial/endpoint/raw-solver evidence binding mismatch")
    with regular(directory / "RST_CLOCK.csv", max_bytes=1024).open(newline="") as stream:
        clocks = list(csv.reader(stream))
    if len(clocks) != 1 or len(clocks[0]) != 2 or [float(v) for v in clocks[0]] != [1081., 15000.]:
        raise ValueError("recovery actual RST count/clock proof failed")


def checked_recovery(args):
    work = local_work(args.work)
    plan, _ = read_record(work / "RECOVERY_PLAN.json", args.plan_sha256)
    if (plan.get("schema") != "yap.operator_grid.training_recovery.v1" or plan.get("work") != str(work)
            or plan.get("host") != builder.physical_host() or plan.get("review_sha256") != builder.digest(regular(__file__))
            or plan.get("original_raw_seal_contemporaneous") is not False or plan.get("automatic_retry") is not False):
        raise ValueError("invalid fresh extraction-only recovery authority")
    recovery_deadline(plan)
    verify_pins(work / "RECOVERY_PINS.json", plan["recovery_pins_sha256"])
    return plan, work


def recovery_extraction_deck(extractor):
    job = "yap_training_c3"
    deck = extractor.apdl_deck(job, tuple(range(1, 1082, 10)))
    proof = ("/POST1\nFILE,yap_training_c3,rst\nSET,LAST\n*GET,YAPSETS,ACTIVE,0,SET,NSET\n"
             "*GET,YAPCLOCK,ACTIVE,0,SET,TIME\n*CFOPEN,rst_clock,csv\n"
             "*VWRITE,YAPSETS,YAPCLOCK\n(F12.0,',',E22.14)\n*CFCLOS\nFINISH\n/EXIT,NOSAVE")
    if deck.count("/EXIT,NOSAVE") != 1:
        raise ValueError("unexpected extraction exit structure")
    deck = deck.replace("/EXIT,NOSAVE", proof)
    return audit_recovery_extraction_deck(deck)


def audit_recovery_extraction_deck(deck):
    job = "yap_training_c3"
    commands = {"/BATCH", "FINISH", "/CLEAR", "RESUME", "/POST1", "FILE", "/NOPR", "SET", "ESEL",
                "NSLE", "*GET", "*DIM", "*VFILL", "*VGET", "*CFOPEN", "*VWRITE", "*VLEN",
                "*VMASK", "*CFCLOS", "ALLSEL", "/GOPR", "/EXIT"}
    formats = {"(A50)", "(A80)", "(F12.0,3(',',E22.14))",
               "(F10.0,',',E22.14,',',F12.0,3(',',E22.14))", "(F12.0,',',E22.14)"}
    for line in deck.splitlines():
        if "$" in line or "!" in line:
            raise ValueError("inline extraction commands/comments are not allowed")
        name = line.split(",", 1)[0]
        if line in formats:
            continue
        if name not in commands:
            raise ValueError(f"unreviewed extraction command: {name}")
        if name in ("RESUME", "FILE") and line not in (f"RESUME,{job},rdb", f"FILE,{job},rst"):
            raise ValueError("extraction database/results must use private fixed-job copies")
        if name == "*CFOPEN" and line not in {f"*CFOPEN,{p},csv" for p in ("volume_nodes", "volume_displacements", "rst_clock")}:
            raise ValueError("extraction output path is outside its private allowlist")
    if ([line for line in deck.splitlines() if line.startswith("SET,")] !=
            ["SET,,,,,,,1"] + [f"SET,,,,,,,{s}" for s in range(1, 1082, 10)] + ["SET,LAST"]
            or deck.splitlines().count("*GET,YAPSETS,ACTIVE,0,SET,NSET") != 1
            or deck.splitlines().count("*GET,YAPCLOCK,ACTIVE,0,SET,TIME") != 1
            or deck.splitlines().count("RESUME,yap_training_c3,rdb") != 1
            or deck.splitlines().count("/EXIT,NOSAVE") != 1):
        raise ValueError("extraction schedule or actual result-file count proof changed")
    return deck


def training_recover_launch(args):
    created_epoch = time.time()
    work, old = local_work(args.work), local_work(args.source_work)
    archive = builder.safe_path(args.archive)
    if (work.exists() or archive.exists() or work.is_relative_to(old) or old.is_relative_to(work)
            or archive.parent != ARCHIVES or not archive.name.startswith("operator_grid_training_")):
        raise ValueError("recovery needs fresh disjoint owned work and archive directories")
    if args.review_sha256 != builder.digest(regular(__file__)):
        raise ValueError("recovery requires the reviewed current controller SHA256")
    source, _ = read_record(old / "TRAINING_PLAN.json", training_hash(args.source_plan_sha256))
    proof = training_source_exit(old, args.source_plan_sha256, training_hash(args.source_stop_sha256))
    spec = source["geometry"]
    if spec != training_spec(spec["pitch_um"], spec["layer"], spec["si_um"], spec["path"], spec["nx"]):
        raise ValueError("original recovery training geometry drift")
    original_pins = verify_pins(old / "SOURCE_PINS.json", source["source_pins_sha256"])
    if [h for p, h in original_pins["files_sha256"].items() if Path(p).name == "operator_grid.py"] != [source["review_sha256"]]:
        raise ValueError("original solver producer is not bound to historical source pins")
    completion, _ = read_record(old / "MAPDL_COMPLETION.json")
    if (completion.get("rc") != 0 or completion.get("subprocess_rc") != 0
            or completion.get("ansys_error_count") != 0 or completion.get("license_error")
            or completion.get("timeout") or completion.get("lock_conflict")):
        raise ValueError("recovery requires a clean completed original solve")
    training_resources(builder.scratch_root(builder.physical_host()), spec)
    environment = builder.configure(work)
    with worker_solver_lock(work):
        proof = training_source_exit(old, args.source_plan_sha256, args.source_stop_sha256)
        limit = spec["reservation_gib"] * 1024**3
        seals = {name: training_input_seal(old / name, limit) for name in RECOVERY_METADATA + RECOVERY_RAW}
        if seals["SOLVE.dat"]["sha256"] != seals["solver/yap_training_c3.dat"]["sha256"]:
            raise ValueError("retained actual solver input differs from the original preflight deck")
        if sum(v["bytes"] for v in seals.values()) > limit:
            raise ValueError("recovery raw inputs exceed the bounded storage reservation")
    pins = training_closure(Path(source["source"]))
    for path, digest in original_pins["files_sha256"].items():
        if Path(path).is_relative_to(Path(source["source"]).parent) and pins["files_sha256"].get(path) != digest:
            raise ValueError("scientific dependency changed since original solve")
    builder.write_json(work / "RECOVERY_PINS.json", pins, immutable=True)
    plan = dict(schema="yap.operator_grid.training_recovery.v1", work=str(work), archive=str(archive),
        source_work=str(old), source=str(Path(source["source"])), source_plan_sha256=args.source_plan_sha256,
        source_stop_sha256=args.source_stop_sha256, source_pins_sha256=source["source_pins_sha256"],
        source_exit_proof=proof, original_files=seals, geometry=spec, host=builder.physical_host(),
        review_sha256=args.review_sha256, recovery_pins_sha256=builder.digest(work / "RECOVERY_PINS.json"),
        original_raw_seal_contemporaneous=False, limitation="Original raw bytes were first sealed at recovery, not at solver completion",
        created_epoch=created_epoch, expires_epoch=created_epoch + 10800., automatic_retry=False,
        thermal_geometry_validated=False, runtime_admitted=False)
    builder.write_json(work / "RECOVERY_PLAN.json", plan, immutable=True)
    plan_sha = builder.digest(work / "RECOVERY_PLAN.json")
    recovery_deadline(plan)
    spawn_training_worker(work, plan_sha, plan["recovery_pins_sha256"], environment, action="training-recover-worker")


def training_recover_native(args):
    plan, work = checked_recovery(args)
    old = local_work(plan["source_work"])
    training_source_exit(old, plan["source_plan_sha256"], plan["source_stop_sha256"])
    builder.configure(work)
    pins = verify_pins(work / "RECOVERY_PINS.json", plan["recovery_pins_sha256"])
    common = authenticated_common(pins)
    source = Path(plan["source"])
    sys.path[:0] = [str(source), str(source.parent / "09_effective_material_die_validation"),
                   str(source.parent / "11_symmetric_mesh_beol_rve_verify")]
    import importlib
    import numpy as np
    extractor = importlib.import_module(Path(TRAINING_EXTRACTOR).stem)
    if str(regular(extractor.__file__)) not in pins["files_sha256"]:
        raise ValueError("recovery extractor is outside its authenticated source closure")
    private = work / "private"
    private.mkdir()
    spec, limit = plan["geometry"], plan["geometry"]["reservation_gib"] * 1024**3
    for name, seal in plan["original_files"].items():
        if training_input_seal(old / name, limit) != seal:
            raise ValueError("original input replaced or changed since recovery admission")
        destination = private / Path(name).name if name in RECOVERY_RAW else work / (
            "ORIGINAL_" + name if name in ("LAUNCH.json", "STATUS.json", "PROCESS_SCOPE.json") else name)
        builder.copy_checked(old / name, destination, seal["sha256"])
        if training_input_seal(old / name, limit) != seal:
            raise ValueError("original input changed while copying")
        recovery_deadline(plan)
    builder.write_json(work / "RECOVERY_RAW_INPUTS.json", dict(original_files=plan["original_files"],
        original_raw_seal_contemporaneous=False, recovery_time_epoch=time.time(),
        original_source_exit_proof=plan["source_exit_proof"]), immutable=True)
    preflight, _ = read_record(work / "PREFLIGHT.json")
    audit_training_deck((work / "SOLVE.dat").read_text(), preflight["stages"])
    if preflight["deck_sha256"] != builder.digest(work / "SOLVE.dat"):
        raise ValueError("historical preflight deck hash mismatch")
    endpoints = training_endpoints(private / "date_all_stage_macro.csv", preflight["stages"], spec)
    common.write_dict_csv(work / "ALL_STAGE_MACRO.csv", endpoints, list(endpoints[0]))
    rows = training_dense_trace(private / "free_generalized_macro_substeps.csv", spec)
    initial_rows = training_numeric_csv(private / "free_generalized_initial_macro.csv",
        (*TRAINING_NATIVE_MACRO_FIELDS, "time_s"), "eps_x0,k", max_rows=1)
    if len(initial_rows) != 1:
        raise ValueError("missing recovery initial macro state")
    initial = dict(time_s=initial_rows[0]["time_s"], **{field: initial_rows[0][native] for field, native in zip(
        TRAINING_MACRO_FIELDS, ("eps_x0", "eps_z0", "kappa_x", "kappa_z", "gamma_xz0", "kappa_xz"), strict=True)})
    training_initial_clock([initial], rows[0]["time_s"], rows[0])
    deck = recovery_extraction_deck(extractor)
    (work / "EXTRACT.dat").write_text(deck, encoding="ascii")
    hashes = {str(private / ("yap_training_c3." + ext)): builder.digest(private / ("yap_training_c3." + ext))
              for ext in ("rdb", "rst")}
    builder.write_json(work / "RAW_SOLVER_SEAL.json", dict(files_sha256=hashes,
        original_raw_seal_contemporaneous=False), immutable=True)
    training_source_exit(old, plan["source_plan_sha256"], plan["source_stop_sha256"])
    training_resources(work, spec)
    deadline = recovery_deadline(plan, 300.)
    info = common.run_mapdl(run_dir=private, jobname="yap_training_extract", dat_text=deck,
                           mem_mb=4096, db_mb=2048, mode="smp", np=2, timeout_sec=min(7200., deadline - time.time()))
    builder.write_json(work / "EXTRACTION_COMPLETION.json", info, immutable=True)
    if info.get("rc") != 0 or info.get("license_error") or info.get("ansys_error_count") != 0:
        raise RuntimeError("recovery displacement extraction failed")
    with regular(private / "rst_clock.csv", max_bytes=1024).open(newline="") as stream:
        clocks = list(csv.reader(stream))
    if len(clocks) != 1 or len(clocks[0]) != 2 or [float(v) for v in clocks[0]] != [1081., 15000.]:
        raise ValueError("actual RST set count/terminal clock differs from the prescribed retained history")
    sets = tuple(range(1, 1082, 10))
    nodes, coordinates = extractor.parse_nodes(private / "volume_nodes.csv")
    times, displacements = extractor.parse_displacements(private / "volume_displacements.csv", nodes, sets)
    if (len(nodes) != preflight["geometry"]["nodes"] or not np.allclose(times, training_clock()[::10], atol=1e-6, rtol=0)):
        raise ValueError("recovery snapshots differ from the actual retained process clock/mesh")
    training_face_registration(coordinates, spec)
    arrays = dict(node_ids=nodes, coordinates_mm=coordinates, result_sets=np.asarray(sets), times_s=times,
                  displacements_mm=displacements)
    np.savez_compressed(work / "VOLUME_DISPLACEMENT_SNAPSHOTS.npz", **arrays, layer=np.asarray(spec["layer"]))
    selected = [rows[s - 1] for s in sets]
    common.write_dict_csv(work / "SNAPSHOT_MACRO.csv", selected, list(selected[0]))
    for source_name, name in (("free_generalized_macro_substeps.csv", "DENSE_MACRO_RAW.csv"),
                             ("date_all_stage_macro.csv", "RAW_ENDPOINT_MACRO.csv"),
                             ("free_generalized_initial_macro.csv", "RAW_INITIAL_MACRO.csv"), ("rst_clock.csv", "RST_CLOCK.csv")):
        builder.copy_checked(private / source_name, work / name, builder.digest(private / source_name))
    builder.write_json(work / "INITIAL_CLOCK_AUDIT.json", dict(**training_initial_clock([initial], times[0], rows[0]),
        raw_initial_csv_sha256=builder.digest(work / "RAW_INITIAL_MACRO.csv"),
        endpoint_export_sha256=builder.digest(work / "ALL_STAGE_MACRO.csv"), raw_solver_hashes=hashes,
        evidence="actual RST initial set and dense macro agree at 300 s; retrospective raw seal and original clock preserved"), immutable=True)
    for path, digest in hashes.items():
        if builder.digest(regular(path)) != digest:
            raise ValueError("private RDB/RST changed during recovery extraction")
    for name, seal in plan["original_files"].items():
        if training_input_seal(old / name, limit) != seal:
            raise ValueError("historical input changed during recovery extraction")
    recovery_deadline(plan)
    companion = dict(schema="yap.operator_grid.training_recovery_evidence.v1", state="TRAINING_EXTRACTED_NOT_OPERATOR",
        recovery_plan_sha256=args.plan_sha256, source_plan_sha256=plan["source_plan_sha256"],
        files_sha256={name: builder.digest(work / name) for name in RECOVERY_ARTIFACTS},
        original_raw_seal_contemporaneous=False, thermal_geometry_validated=False, runtime_admitted=False)
    builder.write_json(work / "RECOVERY_COMPLETE.json", companion, immutable=True)
    receipt = dict(state="TRAINING_EXTRACTED_NOT_OPERATOR", geometry=spec,
        files_sha256={name: builder.digest(work / name) for name in TRAINING_ARTIFACTS},
        raw_extraction_sha256={name: builder.digest(private / name) for name in ("volume_nodes.csv", "volume_displacements.csv")},
        actual_sets=1081, selected_snapshots=109, snapshot_shape=list(displacements.shape),
        arrays={name: dict(dtype=str(a.dtype), shape=list(a.shape), sha256=hashlib.sha256(memoryview(a).cast("B")).hexdigest())
                for name, a in arrays.items()}, source_plan_sha256=plan["source_plan_sha256"],
        recovery_plan_sha256=args.plan_sha256, recovery_evidence_sha256=builder.digest(work / "RECOVERY_COMPLETE.json"),
        thermal_geometry_validated=False, runtime_admitted=False)
    builder.write_json(work / "TRAINING_COMPLETE.json", receipt, immutable=True)


def training_recover_worker(args):
    plan, work = checked_recovery(args)
    environment = builder.configure(work)
    from operator_thermal_validation import run_child, ProcessInventory, publish_worker_scope
    with worker_solver_lock(work):
        state = dict(state="RECOVERY_COPYING_AND_EXTRACTION", worker_pid=os.getpid(),
                     thermal_geometry_validated=False, runtime_admitted=False)
        builder.write_json(work / "RECOVERY_STATUS.json", state, immutable=True)
        publish_worker_scope(ProcessInventory())
        def monitor():
            if shutil.disk_usage(work).free / 1024**3 < 200:
                raise RuntimeError("training recovery reached the local storage safety floor")
        try:
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "training-recover-native",
                       "--work", str(work), "--plan-sha256", args.plan_sha256]
            run_child(command, work / "recovery.log", plan["expires_epoch"], environment=environment, observer=monitor)
            receipt, receipt_sha = read_record(work / "TRAINING_COMPLETE.json")
            load_training_evidence(work, receipt_sha)
            recovery_deadline(plan)
            archive = builder.safe_path(plan["archive"])
            if archive.parent != ARCHIVES or not archive.name.startswith("operator_grid_training_") or archive.exists():
                raise ValueError("recovery publication requires a fresh neutral archive")
            archive.mkdir()
            files = dict(receipt["files_sha256"])
            for name in (*RECOVERY_ARTIFACTS, "RECOVERY_COMPLETE.json", "TRAINING_PLAN.json", "SOURCE_PINS.json"):
                files[name] = builder.digest(work / name)
            for name, digest in files.items():
                recovery_deadline(plan)
                builder.copy_checked(work / name, archive / name, digest)
            recovery_deadline(plan)
            builder.copy_checked(work / "TRAINING_COMPLETE.json", archive / "TRAINING_COMPLETE.json", receipt_sha)
            files["TRAINING_COMPLETE.json"] = receipt_sha
            state.update(state="TRAINING_EXTRACTED_NOT_OPERATOR", archive=str(archive),
                         published_files_sha256=files, completed_epoch=time.time())
            builder.write_json(work / "RECOVERY_STATUS.json", state)
            builder.write_json(archive / "RECOVERY_STATUS.json", state, immutable=True)
        except BaseException as error:
            state.update(state="STOP_REQUIRES_REVIEW", error=f"{type(error).__name__}: {error}")
            builder.write_json(work / "RECOVERY_STATUS.json", state)
            raise


def checked_training(args):
    work = local_work(args.work)
    plan, _ = read_record(work / "TRAINING_PLAN.json", args.plan_sha256)
    verify_pins(work / "SOURCE_PINS.json", plan["source_pins_sha256"])
    spec = plan["geometry"]
    if (plan.get("schema") != "yap.operator_grid.training.v1" or plan["work"] != str(work)
            or plan["host"] != builder.physical_host() or plan["expires_epoch"] <= time.time()
            or spec != training_spec(spec["pitch_um"], spec["layer"], spec["si_um"], spec["path"], spec["nx"])):
        raise ValueError("training authority/geometry expired or changed")
    with regular(work / "PRESCRIBED_PATH.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    validate_training_rows(rows)
    if rows != [{k: str(v) for k, v in r.items()} for r in training_rows(spec)] or builder.digest(work / "PRESCRIBED_PATH.csv") != plan["path_sha256"]:
        raise ValueError("training path changed")
    return plan, work


def training_native(args):
    plan, work = checked_training(args)
    builder.configure(work)
    pins = verify_pins(work / "SOURCE_PINS.json", plan["source_pins_sha256"])
    common = authenticated_common(pins)
    source = Path(plan["source"])
    sys.path[:0] = [str(source), str(source.parent / "09_effective_material_die_validation"),
                   str(source.parent / "11_symmetric_mesh_beol_rve_verify")]
    import importlib
    native = importlib.import_module(Path(TRAINING_NATIVE).stem)
    outputs = importlib.import_module(Path(TRAINING_OUTPUTS).stem)
    extractor = importlib.import_module(Path(TRAINING_EXTRACTOR).stem)
    from exp12_250c_branch import ramped_native_deck
    for module in (native, outputs, extractor):
        if str(regular(module.__file__)) not in pins["files_sha256"]:
            raise ValueError("training import is outside authenticated source closure")
    spec = plan["geometry"]
    namespace, cards = training_args(native, spec, work, source)
    geometry = training_geometry(native, namespace, spec)
    history = native.load_macro_history(namespace)
    job = "yap_training_c3"
    deck = ramped_native_deck(native.build_apdl(namespace, job, history, []), cycles=3)
    records = audit_training_deck(deck, outputs.solve_records(deck))
    deck = outputs.add_all_stage_export(deck, namespace, records)
    audit_training_deck(deck, outputs.solve_records(deck))
    preflight = dict(geometry=geometry, cards=cards, stages=records,
                     deck_sha256=hashlib.sha256(deck.encode("ascii")).hexdigest(),
                     nodal_only_intermediate_results=True, protected_validation_targets_read=[],
                     thermal_geometry_validated=False, runtime_admitted=False)
    if args.preflight_only:
        with (work / "SOLVE.dat").open("x", encoding="ascii") as stream:
            stream.write(deck)
        builder.write_json(work / "PREFLIGHT.json", preflight, immutable=True)
        print(json.dumps(preflight), flush=True)
        return
    frozen, _ = read_record(work / "PREFLIGHT.json")
    if frozen != preflight or (work / "SOLVE.dat").read_text() != deck:
        raise ValueError("training preflight deck or geometry drift")
    deadline = plan["expires_epoch"] - 120.
    training_resources(work, spec)
    run_dir, output = work / "solver", work / "native"
    output.mkdir()
    run_dir.mkdir()
    info = common.run_mapdl(run_dir=run_dir, jobname=job, dat_text=deck, mem_mb=20480,
                           db_mb=4096, mode="smp", np=8, timeout_sec=min(21600., deadline - time.time()))
    builder.write_json(work / "MAPDL_COMPLETION.json", info, immutable=True)
    if info.get("rc", -1) != 0 or info.get("license_error"):
        raise RuntimeError("training MAPDL did not complete cleanly")
    verify_pins(work / "SOURCE_PINS.json", plan["source_pins_sha256"])
    endpoints = training_endpoints(run_dir / "date_all_stage_macro.csv", records, spec)
    common.write_dict_csv(output / "ALL_STAGE_MACRO.csv", endpoints, list(endpoints[0]))
    native.postprocess(namespace, run_dir, info)
    rows = training_dense_trace(run_dir / "free_generalized_macro_substeps.csv", spec)
    hashes = {str(p): builder.digest(regular(p)) for p in (run_dir / (job + ".rdb"), run_dir / (job + ".rst"))}
    builder.write_json(work / "RAW_SOLVER_SEAL.json", dict(files_sha256=hashes), immutable=True)
    sets = extractor.selected_result_sets(1081, 10)
    extraction = work / "extract"
    extraction.mkdir()
    for name, digest in hashes.items():
        builder.copy_checked(Path(name), extraction / Path(name).name, digest)
    if deadline - time.time() < 300.:
        raise TimeoutError("training extraction lacks remaining bounded wall budget")
    extraction_info = common.run_mapdl(run_dir=extraction, jobname="yap_training_extract",
        dat_text=extractor.apdl_deck(job, sets), mem_mb=4096, db_mb=2048, mode="smp", np=2,
        timeout_sec=min(7200., deadline - time.time()))
    if extraction_info.get("rc", -1) != 0 or extraction_info.get("license_error"):
        raise RuntimeError("training displacement extraction failed")
    nodes, coordinates = extractor.parse_nodes(extraction / "volume_nodes.csv")
    times, displacements = extractor.parse_displacements(extraction / "volume_displacements.csv", nodes, sets)
    import numpy as np
    if len(nodes) != geometry["nodes"] or not np.allclose(times, [rows[s - 1]["time_s"] for s in sets], rtol=0, atol=1e-6):
        raise ValueError("training snapshots differ from actual retained macro sets/time")
    for path, expected in hashes.items():
        if builder.digest(regular(path)) != expected:
            raise ValueError("training extraction source changed after sealing")
    selected = [rows[s - 1] for s in sets]
    common.write_dict_csv(work / "SNAPSHOT_MACRO.csv", selected, list(selected[0]))
    np.savez_compressed(work / "VOLUME_DISPLACEMENT_SNAPSHOTS.npz", node_ids=nodes, coordinates_mm=coordinates,
        result_sets=np.asarray(sets), times_s=times, displacements_mm=displacements, layer=np.asarray(spec["layer"]))
    initial = native.parse_free_generalized_initial(run_dir / "free_generalized_initial_macro.csv")
    initial_audit = training_initial_clock(initial, times[0], rows[0])
    builder.write_json(work / "INITIAL_CLOCK_AUDIT.json", dict(**initial_audit,
        raw_initial_csv_sha256=builder.digest(run_dir / "free_generalized_initial_macro.csv"),
        endpoint_export_sha256=builder.digest(output / "ALL_STAGE_MACRO.csv"), raw_solver_hashes=hashes,
        evidence="actual RST set 1, all-stage endpoint and dense macro trace agree at 300 s; raw preserved"), immutable=True)
    receipt = dict(state="TRAINING_EXTRACTED_NOT_OPERATOR", geometry=spec,
        files_sha256={name: builder.digest(work / name) for name in TRAINING_ARTIFACTS},
        raw_extraction_sha256={name: builder.digest(extraction / name) for name in ("volume_nodes.csv", "volume_displacements.csv")},
        actual_sets=1081, selected_snapshots=109, snapshot_shape=list(displacements.shape),
        arrays={name: dict(dtype=str(array.dtype), shape=list(array.shape),
                          sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())
                for name, array in dict(node_ids=nodes, coordinates_mm=coordinates, times_s=times,
                                        result_sets=np.asarray(sets), displacements_mm=displacements).items()},
        source_plan_sha256=args.plan_sha256, thermal_geometry_validated=False, runtime_admitted=False)
    builder.write_json(work / "TRAINING_COMPLETE.json", receipt, immutable=True)
    print(json.dumps(receipt), flush=True)


def training_worker(args):
    plan, work = checked_training(args)
    environment = builder.configure(work)
    from operator_thermal_validation import run_child, ProcessInventory, publish_worker_scope
    with worker_solver_lock(work):
        state = dict(state="PREFLIGHT", worker_pid=os.getpid(), thermal_geometry_validated=False, runtime_admitted=False)
        builder.write_json(work / "STATUS.json", state, immutable=True)
        publish_worker_scope(ProcessInventory())
        baseline = shutil.disk_usage(work).free / 1024**3
        previous = [0.]
        state["initial_local_free_gib"] = state["minimum_sampled_local_free_gib"] = baseline
        def monitor():
            free = shutil.disk_usage(work).free / 1024**3
            state["minimum_sampled_local_free_gib"] = min(free, state["minimum_sampled_local_free_gib"])
            if time.monotonic() - previous[0] >= 30:
                state["heartbeat_epoch"] = time.time()
                builder.write_json(work / "STATUS.json", state)
                previous[0] = time.monotonic()
            if free < 200:
                raise RuntimeError("training local storage safety floor reached; retain data for review")
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "training-native",
                   "--work", str(work), "--plan-sha256", args.plan_sha256]
        try:
            run_child(command + ["--preflight-only"], work / "preflight.log", time.time() + 600.,
                      environment=environment, observer=monitor)
            training_resources(work, plan["geometry"])
            state["state"] = "MAPDL_TRAINING_AND_EXTRACTION"
            builder.write_json(work / "STATUS.json", state)
            run_child(command, work / "training.log", plan["expires_epoch"], environment=environment, observer=monitor)
            receipt, _ = read_record(work / "TRAINING_COMPLETE.json")
            if receipt["source_plan_sha256"] != args.plan_sha256 or receipt["runtime_admitted"] or receipt["thermal_geometry_validated"]:
                raise ValueError("invalid training completion receipt")
            archive = builder.safe_path(plan["archive"])
            if archive.parent != ARCHIVES or not archive.name.startswith("operator_grid_training_"):
                raise ValueError("invalid training publication root")
            archive.mkdir()
            files = dict(receipt["files_sha256"])
            for name in ("TRAINING_PLAN.json", "SOURCE_PINS.json", "TRAINING_COMPLETE.json"):
                files[name] = builder.digest(work / name)
            for name, digest in files.items():
                builder.copy_checked(work / name, archive / name, digest)
            state.update(state="TRAINING_EXTRACTED_NOT_OPERATOR", completed_epoch=time.time(),
                         archive=str(archive), published_files_sha256=files)
            builder.write_json(work / "STATUS.json", state)
            builder.write_json(archive / "STATUS.json", state, immutable=True)
        except BaseException as error:
            state.update(state="STOP_REQUIRES_REVIEW", error=f"{type(error).__name__}: {error}")
            builder.write_json(work / "STATUS.json", state)
            raise


def training_launch(args):
    spec = training_spec(args.pitch, args.layer, args.si_thickness, args.path, args.nx)
    work = local_work(args.work)
    archive = builder.safe_path(args.archive)
    if work.exists() or archive.exists() or archive.parent != ARCHIVES or not archive.name.startswith("operator_grid_training_"):
        raise ValueError("training needs fresh owned work and canonical archive directories")
    if args.review_sha256 != builder.digest(regular(__file__)):
        raise ValueError("training launch requires the reviewed controller SHA256")
    training_resources(builder.scratch_root(builder.physical_host()), spec)
    environment = builder.configure(work)
    pins = training_closure(args.native_source)
    builder.write_json(work / "SOURCE_PINS.json", pins, immutable=True)
    pins_sha = builder.digest(work / "SOURCE_PINS.json")
    verify_pins(work / "SOURCE_PINS.json", pins_sha)
    rows = training_rows(spec)
    validate_training_rows(rows)
    with (work / "PRESCRIBED_PATH.csv").open("x", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plan = dict(schema="yap.operator_grid.training.v1", work=str(work), archive=str(archive),
                source=str(regular(args.native_source / TRAINING_NATIVE).parent), host=builder.physical_host(),
                geometry=spec, source_pins_sha256=pins_sha, review_sha256=args.review_sha256,
                path_sha256=builder.digest(work / "PRESCRIBED_PATH.csv"), created_epoch=time.time(),
                expires_epoch=time.time() + TRAINING_SECONDS, automatic_retry=False,
                scratch_retained=True, protected_validation_targets_read=[])
    builder.write_json(work / "TRAINING_PLAN.json", plan, immutable=True)
    plan_sha = builder.digest(work / "TRAINING_PLAN.json")
    spawn_training_worker(work, plan_sha, pins_sha, environment)


def spawn_training_worker(work, plan_sha, pins_sha, environment, *, action="training-worker"):
    if action not in {"training-worker", "training-recover-worker"}:
        raise ValueError("invalid training worker entry point")
    command = [sys.executable, "-B", str(Path(__file__).resolve()), action,
               "--work", str(work), "--plan-sha256", plan_sha]
    observed, child = {}, None
    try:
        with (work / "worker.log").open("x") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     env=environment, start_new_session=True)
        identity = wait_exec_identity(child, command, observed)
        receipt = dict(worker_pid=child.pid, worker_identity=identity, command=command,
                       plan_sha256=plan_sha, source_pins_sha256=pins_sha, launched_epoch=time.time())
        builder.write_json(work / "LAUNCH.json", receipt, immutable=True)
        print(json.dumps(receipt, indent=2), flush=True)
    except BaseException:
        if child is not None:
            from operator_thermal_validation import cleanup_spawned
            cleanup_spawned(child, observed)
        raise


@contextmanager
def training_signal_guard():
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"training interrupted: {signum}")
    signals = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    previous = {value: signal.getsignal(value) for value in signals}
    try:
        for value in signals:
            signal.signal(value, interrupted)
        yield
    finally:
        for value, handler in previous.items():
            signal.signal(value, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("plan")
    recover = commands.add_parser("training-recover-launch", help="Fresh extraction-only recovery of a stopped completed solve")
    for name in ("work", "archive", "source-work"):
        recover.add_argument("--" + name, type=Path, required=True)
    for name in ("source-plan-sha256", "source-stop-sha256", "review-sha256"):
        recover.add_argument("--" + name, required=True)
    for name in ("training-recover-worker", "training-recover-native"):
        child = commands.add_parser(name)
        child.add_argument("--work", type=Path, required=True)
        child.add_argument("--plan-sha256", required=True)
    audit = commands.add_parser("training-audit", help="Verify completed sealed histories, not operator qualification")
    audit.add_argument("--directory", type=Path, required=True)
    audit.add_argument("--receipt-sha256", required=True)
    face_audit = commands.add_parser("training-face-audit", help="Audit sealed face kinematics, not operator qualification")
    face_audit.add_argument("--directory", type=Path, required=True)
    face_audit.add_argument("--receipt-sha256", required=True)
    train = commands.add_parser("training-launch", help="Fresh prescribed isolated-layer histories, not operator admission")
    for name in ("work", "archive", "native-source"):
        train.add_argument("--" + name, type=Path, required=True)
    train.add_argument("--review-sha256", required=True)
    train.add_argument("--pitch", type=int, choices=PITCHES, required=True)
    train.add_argument("--layer", choices=("bi", "beol", "tsv_si"), default="bi")
    train.add_argument("--si-thickness", type=float, default=38.5)
    train.add_argument("--nx", type=int, default=48)
    train.add_argument("--path", choices=TRAINING_PATHS + TRAINING_HOLDOUTS, default="thermal_zero_macro")
    for name in ("training-worker", "training-native"):
        child = commands.add_parser(name)
        child.add_argument("--work", type=Path, required=True)
        child.add_argument("--plan-sha256", required=True)
        if name == "training-native":
            child.add_argument("--preflight-only", action="store_true")
    start = commands.add_parser("launch")
    for name in ("work", "archive", "native-source", "baseline-runtime", "parent-package"):
        start.add_argument("--" + name, type=Path, required=True)
    start.add_argument("--review-sha256", required=True)
    start.add_argument("--si-thicknesses", type=float, nargs="+", default=list(PILOT_SI),
                       help="Explicit ascending missing p2 Si endpoints, at most four per campaign")
    start.add_argument("--after-campaign", type=Path,
                       help="Wait for immutable completion and process exit of the original p2 pilot")
    start.add_argument("--adopt-from", type=Path,
                       help="Adopt the first case's sealed tangent/q6 from an explicitly reviewed stopped campaign")
    run = commands.add_parser("worker")
    run.add_argument("--work", type=Path, required=True)
    run.add_argument("--native-source", type=Path, required=True)
    run.add_argument("--pins-sha256", required=True)
    run.add_argument("--campaign-sha256", required=True)
    stage = commands.add_parser("native-stage")
    stage.add_argument("--pins", type=Path, required=True)
    stage.add_argument("--pins-sha256", required=True)
    stage.add_argument("--case-deadline", type=float, required=True)
    stage.add_argument("--campaign-file", type=Path)
    stage.add_argument("--campaign-sha256")
    stage.add_argument("command", nargs=argparse.REMAINDER)
    complete = commands.add_parser("publish-complete")
    for name in ("source-work", "work", "archive", "native-source", "parent-package"):
        complete.add_argument("--" + name, type=Path, required=True)
    complete.add_argument("--si-thickness", type=float, required=True)
    complete.add_argument("--review-sha256", required=True)
    args = parser.parse_args()
    if args.action == "plan":
        print(json.dumps(coverage_plan(), indent=2, allow_nan=False))
    elif args.action == "training-face-audit":
        print(json.dumps(training_face_audit(args.directory, args.receipt_sha256), indent=2, allow_nan=False))
    elif args.action == "training-audit":
        evidence = load_training_evidence(args.directory, args.receipt_sha256)
        report = {key: value for key, value in evidence.items() if key not in ("arrays", "face_indices", "q6_centered")}
        report.update(state="SEALED_TRAINING_INTEGRITY_PASSED_NOT_OPERATOR", selected_snapshots=109,
                      registered_face_nodes=len(evidence["face_indices"][0]),
                      snapshot_shape=list(evidence["arrays"]["displacements_mm"].shape))
        print(json.dumps(report, indent=2, allow_nan=False))
    elif args.action.startswith("training-recover-"):
        with training_signal_guard():
            if args.action == "training-recover-native":
                bounded_cpu(checked_recovery(args)[0]["expires_epoch"] - 60., lambda: training_recover_native(args))
            elif args.action == "training-recover-launch":
                bounded_cpu(time.time() + 1800., lambda: training_recover_launch(args))
            else:
                training_recover_worker(args)
    elif args.action.startswith("training-"):
        with training_signal_guard():
            if args.action == "training-native":
                bounded_cpu(checked_training(args)[0]["expires_epoch"] - 60., lambda: training_native(args))
            else:
                {"training-launch": training_launch, "training-worker": training_worker}[args.action](args)
    else:
        {"launch": launch, "worker": worker, "native-stage": native_stage,
         "publish-complete": publish_complete}[args.action](args)


if __name__ == "__main__":
    bootstrap_group_environment()
    main()
