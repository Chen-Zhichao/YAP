"""Separate lift-conjugacy and physical errors using sealed Cu snapshots.

Reconstructing equilibrium with frozen internal strain avoids replaying the
entire nonlinear history. This is diagnostic evidence, never a replacement
for the unchanged MAPDL accuracy screen.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.build_prepared_operators as storage
import tools.operator_grid as grid
import tools.operator_thermal_validation as thermal


def frozen_equilibrium(stiffness, thermal_load, eigenstrain_force, macro):
    import numpy as np
    if not all(np.isfinite(v).all() for v in (stiffness, thermal_load, eigenstrain_force, macro)):
        raise ValueError("nonfinite frozen equilibrium input")
    size = len(macro)
    rhs = thermal_load[size:] + eigenstrain_force[size:] - stiffness[size:, :size] @ macro
    bubble = np.linalg.solve(stiffness[size:, size:], rhs)
    full = np.r_[macro, bubble]
    response = stiffness @ full - thermal_load - eigenstrain_force
    if not np.isfinite(full).all() or not np.isfinite(response).all():
        raise ValueError("nonfinite frozen equilibrium solution")
    return full, response


def deny_native(*args, **kwargs):
    raise RuntimeError("CPU-only diagnostic cannot launch subprocesses or native solvers")


def run(args):
    from tools.operator_accuracy_review import completed_cases, read, validate_screen
    work = storage.local_scratch(args.work)
    if work.exists():
        raise FileExistsError("fresh diagnostic work required")
    storage.configure(work)
    import numpy as np
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("snapshot diagnostic deadline")))
    seconds = 5400 if args.full_nodal else 1800
    signal.alarm(seconds)
    memory_gib = 64 if args.full_nodal else 16
    resource.setrlimit(resource.RLIMIT_AS, (memory_gib * 1024**3, memory_gib * 1024**3))
    plan, plan_sha = read(args.plan, args.plan_sha256)
    if plan["execution_host"] != storage.physical_host():
        raise ValueError("diagnostic must run beside its sealed case")
    status, _ = read(Path(plan["work"]) / "STATUS.json")
    record, old_state = next((r, s) for r, s in completed_cases(plan, status) if r["key"] == args.key)
    case = Path(plan["work"]) / "cases" / args.key
    screen, screen_sha = read(case / "THERMAL_SCREEN.json")
    validate_screen(record, screen)
    if screen["state"] != old_state:
        raise ValueError("completion differs from scientific receipt")
    pins = grid.verify_pins(Path(plan["work"]) / "SOURCE_PINS.json", plan["pins_sha256"])
    seal, _ = read(case / "ROM_SEAL.json")
    for name, digest in seal["files_sha256"].items():
        if storage.digest(case / name) != digest:
            raise ValueError("sealed ROM changed")
    contract, _ = read(case / "INPUT_CONTRACT.json")
    if contract["candidate"] != record:
        raise ValueError("case contract candidate differs")
    thermal.verify_inputs(record, case)
    review, _ = read(args.review, args.review_sha256)
    if (review["case"] != str(case) or review["plan_sha256"] != plan_sha
            or review["screen_sha256"] != screen_sha or review["candidate_sha256"] != record["files_sha256"]):
        raise ValueError("independent review and diagnostic candidate differ")
    if storage.digest(case / "SCORED_RESPONSE.npz") != review["scored_response_sha256"]:
        raise ValueError("reviewed scored response changed")
    for name, digest in contract["input_sha256"].items():
        if storage.digest(case / name) != digest:
            raise ValueError("loading history changed")
    inventory, archive_sha = read(Path(record["root"]) / "ARCHIVE.json", record["files_sha256"]["ARCHIVE.json"])
    artifacts = inventory["files_sha256"]
    suffixes = ("bubble/AUDITED_FINAL_BUBBLE_SPACE.npz", "static/tangent/M4_COMMON_PORT_TANGENT_MANIFEST.json")
    paths = []
    for suffix in suffixes:
        matches = [name for name in artifacts if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError("unique audited bubble/mesh artifact required")
        target = work / Path(matches[0]).name
        storage.copy_checked(Path(record["root"]) / matches[0], target, artifacts[matches[0]])
        paths.append(target)
    storage.write_json(work / "LAUNCH.json", dict(worker_identity=grid.process_identity(os.getpid()),
        plan_sha256=plan_sha, key=args.key, original_screen_sha256=screen_sha,
        source_sha256=storage.digest(__file__), deadline_epoch=time.time() + seconds,
        archive_sha256=archive_sha, no_native_solver=True), immutable=True)
    sys.path[:0] = [plan["source"], str(Path(plan["source"]).parent / "09_effective_material_die_validation"),
                   str(Path(plan["source"]).parent / "11_symmetric_mesh_beol_rve_verify")]
    common = thermal.authenticated_common(plan)
    common.run_ansys = deny_native
    # Matplotlib discovers fonts with fc-list on first import. Complete that
    # dependency initialization before forbidding every subsequent subprocess.
    import matplotlib.font_manager
    subprocess.Popen = subprocess.run = os.system = deny_native
    import run_exp12_m4_all_gauss_offpath_layer_gate as api
    from exp12_cu_material_point import isotropic_stiffness, thermal_strain_vector
    from exp12_material_separable_reference import structured_node_id
    from exp12_solid185_bbar import bbar_points
    from audit_exp12_m4_python_bbar_reference import material_map
    helper = storage.safe_path(args.probe_helper)
    if storage.digest(helper) != args.probe_helper_sha256:
        raise ValueError("independent work probe implementation changed")
    spec = importlib.util.spec_from_file_location("work_probes", helper)
    probes_api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probes_api)
    for module in (api, sys.modules["exp12_cu_material_point"], sys.modules["exp12_material_separable_reference"],
                   sys.modules["exp12_solid185_bbar"], sys.modules["audit_exp12_m4_python_bbar_reference"]):
        path = str(Path(module.__file__).resolve())
        if storage.digest(path) != pins["files_sha256"].get(path):
            raise ValueError("diagnostic science differs from recorded evaluation epoch")
    dataset_path = case / Path(record["dataset"]).name
    if storage.digest(dataset_path) != record["files_sha256"][record["dataset"]]:
        raise ValueError("candidate dataset changed")
    ds = api.LayerROMDataset.load(dataset_path)
    thermal.validate_materials(plan, record, ds)
    op = api.AllGaussReducedFE2Layer.from_dataset(ds)
    with np.load(case / Path(record["lifts"]).name, allow_pickle=False) as a:
        ids, xyz, lifts = a["node_ids"], a["coordinates_mm"], a["lift_fields"]
    with np.load(paths[0], allow_pickle=False) as a:
        if not np.array_equal(a["coordinates_mm"], xyz) or not np.array_equal(a["node_ids"], ids):
            raise ValueError("audited bubble mesh differs")
        bubbles = a["bubble_modes"]
    if (len(ids), ds.macro_size, ds.selected_cu_point_count) != (46 * 21 * 46, 134, 64160):
        raise ValueError("unexpected diagnostic mesh/coordinate coverage")
    fields = np.concatenate((lifts, bubbles), axis=2)
    affine = probes_api.affine_fields(xyz)
    probe_fields = np.concatenate((lifts[:, :, :6], affine, lifts[:, :, :6] - affine), axis=2)
    manifest, _ = read(paths[1])
    nx, ny, nz = (int(manifest["mesh"][k]) for k in ("nx", "ny", "nz"))
    if (nx, ny, nz) != (45, 20, 45):
        raise ValueError("unexpected target mesh")
    materials = np.asarray(material_map("tsv_si", nx=nx, nz=nz, coupon_x_um=6., coupon_z_um=6., geometry=manifest["geometry"]))
    lookup = {int(value): i for i, value in enumerate(ids)}
    offsets = ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1))
    ijk = [(i, j, k) for j in range(ny) for i in range(nx) for k in range(nz)]
    nodes_all = np.asarray([[lookup[structured_node_id(i+a, j+b, k+c, ny, nz)] for a, b, c in offsets] for i, j, k in ijk])
    kinds = np.asarray([materials[i, k] for i, j, k in ijk])
    rule = bbar_points(xyz[nodes_all[0]])
    b = np.stack([point.b_matrix for point in rule])
    weights = np.asarray([point.integration_volume_mm3 for point in rule])
    relative = np.stack([point.position_mm - xyz[nodes_all[0, 0]] for point in rule])
    properties = ds.material_reference().properties("si", 25.)
    if ds.material_reference().properties("si", 350.) != properties:
        raise ValueError("diagnostic requires recorded constant Si properties")
    stiffness_si = isotropic_stiffness(*properties[:2])
    center = .5 * (xyz[:, 1].min() + xyz[:, 1].max())
    nodal, thermal_load = np.zeros((len(ids), 3, 27)), np.zeros(27)
    cu_maps, cu_offset = [], 0
    for start in range(0, len(nodes_all), 256):
        nodes = nodes_all[start:start+256]
        coords = xyz[nodes]
        positions = coords[:, :1, :] + relative[None, :, :]
        strain = np.einsum("gsp,epq->egsq", b, probe_fields[nodes].reshape(-1, 24, 18), optimize=True)
        physical = probes_api.physical_virtual_strains(positions, center)
        centroid = probes_api.physical_virtual_strains(np.broadcast_to(coords.mean(axis=1)[:, None, :], positions.shape), center)[..., 3:]
        probes = np.concatenate((strain, physical, centroid), axis=3)
        si = kinds[start:start+len(nodes)] == 3
        force, load = probes_api.nodal_probe_forces(b, weights, probes[si], stiffness_si)
        np.add.at(nodal, nodes[si], force.reshape(-1, 8, 3, 27))
        thermal_load += load
        cu = np.flatnonzero(kinds[start:start+len(nodes)] == 4)
        if len(cu):
            maps = np.einsum("gsp,epq->egsq", b, fields[nodes[cu]].reshape(-1, 24, op.coordinate_size), optimize=True).reshape(-1, 6, op.coordinate_size)
            expected = op.strain_maps[cu_offset:cu_offset+len(maps)]
            if not np.allclose(maps, expected, atol=1e-8, rtol=0):
                raise ValueError("independent Cu strain mapping differs")
            if not np.allclose(positions[cu].reshape(-1, 3), ds.cu_positions_mm[cu_offset:cu_offset+len(maps)], atol=1e-12, rtol=0):
                raise ValueError("Cu quadrature order differs")
            cu_offset += len(maps)
            cu_maps.append(probes[cu].reshape(-1, 6, 27))
        if start % 4096 == 0:
            print(f"{args.key}: assembled {start+len(nodes)}/{len(nodes_all)} elements", flush=True)
    if cu_offset != ds.selected_cu_point_count:
        raise ValueError("Cu quadrature incomplete")
    cu_maps = np.concatenate(cu_maps)
    si_matrix = nodal.reshape(-1, 27).T @ fields.reshape(-1, op.coordinate_size)
    full_audit = None
    if args.full_nodal:
        from tools.operator_frozen_residual import FrozenResidualAudit
        full_audit = FrozenResidualAudit(work, plan["source"], ids, xyz, fields, materials, ds, op,
            (case / "preflight" / "M3_tsv_si_INTERFACE_REPLAY_exact.dat").read_text())
    del fields, nodal, lifts, bubbles, probe_fields
    with np.load(case / "PATH.npz", allow_pickle=False) as a:
        q, times, temps = a["external_coordinates"], a["times_s"], a["temperatures_c"]
    with np.load(case / "SCORED_RESPONSE.npz", allow_pickle=False) as a:
        previous, teacher = a["prediction"], a["reference"]
        scored_times = a["time_s"]
    with np.load(case / "ROM_RESPONSE.npz", allow_pickle=False) as a:
        if not np.array_equal(previous, a["response"]) or not np.array_equal(scored_times, a["time_s"]):
            raise ValueError("scored response differs from sealed prediction")
    if (previous.shape != (1081, 134) or teacher.shape != previous.shape or q.shape != previous.shape
            or not np.array_equal(scored_times, times)
            or not all(np.isfinite(a).all() for a in (q, times, temps, previous, teacher))):
        raise ValueError("diagnostic coverage or finite values differ")
    samples = []
    with np.load(case / "ROM_CU_STATES.npz", allow_pickle=False) as states:
        sets = sorted({int(k.split("_")[1]) for k in states.files})
        expected_sets = [1, 161, 201, 361, 521, 561, 721, 881, 921, 1081]
        fields_expected = {f"set_{s}_{f}" for s in expected_sets for f in (
            "plastic_strain", "creep_strain", "equivalent_plastic_strain", "equivalent_creep_strain")}
        if sets != expected_sets or set(states.files) != fields_expected:
            raise ValueError("incomplete or unexpected saved Cu snapshot coverage")
        for result_set in sets:
            i, temp = result_set-1, float(temps[result_set-1])
            eigenstrain = states[f"set_{result_set}_plastic_strain"] + states[f"set_{result_set}_creep_strain"]
            if eigenstrain.shape != (64160, 6) or not np.isfinite(eigenstrain).all():
                raise ValueError("invalid saved Cu eigenstrain")
            cu_stiffness = isotropic_stiffness(*ds.cu_card.elastic(temp))
            eigen_force = op._integrate_virtual_work(eigenstrain @ cu_stiffness.T)
            full, response = frozen_equilibrium(op.reference_stiffness(temp), op.reference_thermal_load(temp), eigen_force, q[i])
            reconstruction_error = float(np.max(abs(response[:134]-previous[i])))
            if reconstruction_error > 1e-8 or np.max(abs(response[134:])) > 1e-8:
                raise ValueError("frozen snapshot equilibrium does not reproduce sealed ROM")
            strain = np.einsum("gsq,q->gs", op.strain_maps, full, optimize=True)
            stress = (strain - eigenstrain - thermal_strain_vector(temp, ds.cu_card)) @ cu_stiffness.T
            work_response = si_matrix @ full - thermal_load * properties[2] * (temp-ds.reference_temperature_c)
            work_response += np.einsum("gsq,gs,g->q", cu_maps, stress, ds.cu_weights_mm3, optimize=True)
            if not np.isfinite(stress).all() or not np.isfinite(work_response).all():
                raise ValueError("nonfinite diagnostic stress or virtual work")
            identity_error = float(np.max(abs(work_response[:6]-work_response[6:12]-work_response[12:18])))
            lift_error = float(np.max(abs(work_response[:6]-previous[i, :6])))
            if max(identity_error, lift_error) > 1e-8:
                raise ValueError("independent virtual-work reconstruction differs")
            samples.append(dict(result_set=result_set, time_s=float(times[i]), temperature_c=temp,
                reconstructed_ROM_error=reconstruction_error, lift_reconstruction_error=lift_error,
                same_stress_identity_error=identity_error,
                lift_minus_affine_per_area=((work_response[:6]-work_response[6:12])/.006**2).tolist(),
                Gauss_physical_minus_teacher_per_area=((work_response[18:24]-teacher[i, :6])/.006**2).tolist(),
                centroid_physical_minus_teacher_per_area=((np.r_[work_response[18:21], work_response[24:27]]-teacher[i, :6])/.006**2).tolist()))
            if full_audit is not None:
                full_audit.sample(result_set, temp, full, states[f"set_{result_set}_plastic_strain"],
                    states[f"set_{result_set}_creep_strain"], eigen_force, response)
            storage.write_json(work / "PROGRESS.json", dict(key=args.key, samples=samples))
            print(args.key, "snapshot", result_set, "independent identity", identity_error, flush=True)
    if full_audit is not None:
        full_audit.finish()
    storage.write_json(work / "COMPLETE.json", dict(state="SNAPSHOT_DIAGNOSTIC_COMPLETE_NOT_OPERATOR_PASS",
        key=args.key, original_screen_sha256=screen_sha, original_state=old_state,
        plan_sha256=plan_sha, samples=samples, thresholds_changed=False,
        no_native_solver=True, no_history_replay=True, candidate_modified=False,
        helper_sha256=storage.digest(helper), original_candidate_sha256=record["files_sha256"]), immutable=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--probe-helper", type=Path, required=True)
    parser.add_argument("--probe-helper-sha256", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--full-nodal", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        if args.work.is_dir() and not (args.work / "STOP.json").exists():
            storage.write_json(args.work / "STOP.json", dict(error=f"{type(error).__name__}: {error}", automatic_retry=False), immutable=True)
        raise
