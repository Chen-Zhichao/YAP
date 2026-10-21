"""Observe committed operator results independently of batch completion.

This consumer never changes a candidate, scoring gate, or live controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import build_prepared_operators as storage

Q6 = ("eps_x", "eps_y", "gamma_xy", "kappa_x", "kappa_y", "kappa_xy")
GATES = set(Q6) | {"bottom_ports", "top_ports", "bubble_equilibrium", "inelastic_excitation"}
REPAIR_CONTROLS = (7., 8.5, 17., 18.5, 27., 28.5, 37., 38.5, 47., 48.5,
                   57., 58.5, 67., 68.5, 77., 78.5, 87., 88.5)


def read(path, expected=None):
    path = storage.safe_path(path)
    if not path.is_file() or path.stat().st_size > 2 * 1024**2:
        raise ValueError(f"missing or oversized record: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"record changed: {path}")
    return json.loads(raw), digest


def validate_screen(record, screen):
    gates = screen.get("gates", {})
    if (screen.get("key") != record["key"] or screen.get("states") != 1081
            or screen.get("candidate_sha256") != record["files_sha256"]
            or set(gates) != GATES
            or any(type(g.get("passed")) is not bool for g in gates.values())
            or screen.get("state") not in ("PASS_PRESCRIBED_SCREEN", "FAIL_PRESCRIBED_SCREEN")
            or (screen["state"] == "PASS_PRESCRIBED_SCREEN") != all(g["passed"] for g in gates.values())):
        raise ValueError("malformed operator accuracy result")


def completed_cases(plan, status):
    """STATUS.completed is published only after all case archive copies."""
    candidates = {r["key"]: r for r in plan["candidates"]}
    seen, result = set(), []
    for completed in status.get("completed", []):
        key = completed["key"]
        if key in seen or key not in candidates:
            raise ValueError("duplicate or unassigned completed candidate")
        if completed["state"] not in ("PASS_PRESCRIBED_SCREEN", "FAIL_PRESCRIBED_SCREEN"):
            raise ValueError("invalid completed candidate state")
        seen.add(key)
        result.append((candidates[key], completed["state"]))
    return result


def scientific_epoch(plan, pins):
    root = Path(plan["source"]).parent
    science = {}
    for path, digest in pins["files_sha256"].items():
        file = Path(path)
        if file.is_relative_to(root):
            science[str(file.relative_to(root))] = digest
    payload = dict(science= science, cards={key: value["sha256"] for key, value in plan["cards"].items()},
                   scoring_contract=plan["scoring_contract"])
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def response_diagnosis(prediction, reference, times):
    import numpy as np
    p, r, times = np.asarray(prediction), np.asarray(reference), np.asarray(times)
    if (p.shape != (1081, 134) or r.shape != p.shape or times.shape != (1081,)
            or not all(np.isfinite(a).all() for a in (p, r, times))
            or not np.all(np.diff(times) > 0.)):
        raise ValueError("response coverage or finite values differ")
    p, r = p[:, :6] / .006**2, r[:, :6] / .006**2
    result = {}
    for j, name in enumerate(Q6):
        deadband = 1e-3 if j < 3 else 1e-5
        error = np.abs(p[:, j] - r[:, j])
        limit = np.maximum(deadband, .1 * np.abs(r[:, j]))
        worst = int(np.argmax(error / limit))
        result[name] = dict(units="N/mm" if j < 3 else "N", passed=bool(np.all(error <= limit)),
            failed_states=int(np.count_nonzero(error > limit)), worst_result_set=worst + 1,
            worst_time_s=float(times[worst]), prediction=float(p[worst, j]), reference=float(r[worst, j]),
            absolute_error=float(error[worst]), allowed_error=float(limit[worst]),
            error_over_allowed=float(error[worst] / limit[worst]),
            maximum_absolute_error=float(error.max()), rms_error=float(np.sqrt(np.mean(error**2))),
            first_20_states_maximum_absolute_error=float(error[:20].max()))
    return result


def verify_and_review(plan, plan_sha, record, expected_state, work):
    import numpy as np
    case = Path(plan["work"]) / "cases" / record["key"]
    archive = Path(plan["archive"]) / record["key"]
    screen, screen_sha = read(case / "THERMAL_SCREEN.json")
    validate_screen(record, screen)
    if screen["state"] != expected_state:
        raise ValueError("worker completion and case result differ")
    _, archived_sha = read(archive / "THERMAL_SCREEN.json", screen_sha)
    contract, contract_sha = read(case / "INPUT_CONTRACT.json")
    if contract["candidate"] != record:
        raise ValueError("case candidate/input identity differs")
    rom_seal, rom_sha = read(case / "ROM_SEAL.json")
    teacher_seal, teacher_sha = read(case / "TEACHER_SEAL.json")
    expected_teacher = {
        "teacher/M3_LAYER_INTERFACE_REPLAY_RESULTANTS.csv": screen["reference_resultants_sha256"],
        "teacher/M3_LAYER_INTERFACE_FACE_REACTIONS.csv": screen["reference_face_reactions_sha256"]}
    if teacher_seal["files_sha256"] != expected_teacher:
        raise ValueError("teacher seal differs from scored response")
    required_rom = {"ROM_COMPLETE.json", "ROM_RESPONSE.npz", "ROM_CU_STATES.npz"}
    if set(rom_seal["files_sha256"]) != required_rom:
        raise ValueError("incomplete ROM seal")
    for name, expected in {**rom_seal["files_sha256"], **expected_teacher,
                           **contract["input_sha256"]}.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("nonrelative case evidence")
        if storage.digest(case / name) != expected:
            raise ValueError(f"sealed case evidence changed: {name}")
    scored_sha = storage.digest(case / "SCORED_RESPONSE.npz")
    with np.load(case / "ROM_RESPONSE.npz", allow_pickle=False) as a:
        predictions, times = a["response"], a["time_s"]
    with np.load(case / "SCORED_RESPONSE.npz", allow_pickle=False) as a:
        if not np.array_equal(predictions, a["prediction"]) or not np.array_equal(times, a["time_s"]):
            raise ValueError("scored predictions differ from sealed ROM")
        diagnosis = response_diagnosis(predictions, a["reference"], times)
    if any(diagnosis[k]["passed"] != screen["gates"][k]["passed"] for k in Q6):
        raise ValueError("independent unchanged-gate recomputation differs")
    pins, _ = read(Path(plan["work"]) / "SOURCE_PINS.json", plan["pins_sha256"])
    report = dict(schema="yap.operator_accuracy.review.v1", observer_only=True,
        key=record["key"], si_thickness_um=record["si_thickness_um"], state=screen["state"],
        reviewed_epoch=time.time(), case=str(case), plan_sha256=plan_sha,
        evaluation_epoch=dict(source_pins_sha256=plan["pins_sha256"], cards=plan["cards"],
            scoring_contract=plan["scoring_contract"], source_files_sha256=pins["files_sha256"],
            scientific_epoch_sha256=scientific_epoch(plan, pins)),
        candidate_sha256=record["files_sha256"], input_contract_sha256=contract_sha,
        rom_seal_sha256=rom_sha, teacher_seal_sha256=teacher_sha,
        screen_sha256=screen_sha, scored_response_sha256=scored_sha, archive_screen_sha256=archived_sha,
        component_diagnosis=diagnosis, all_gates=screen["gates"],
        repair_pending=screen["state"] == "FAIL_PRESCRIBED_SCREEN",
        repair_policy="New pinned attempt only. Preserve this result and unchanged gates. No automatic qualification.",
        candidate_admitted=False, runtime_admitted=False)
    name = plan_sha[:16] + "_" + record["key"] + "_" + screen_sha[:16]
    storage.write_json(work / "results" / (name + ".json"), report, immutable=True)
    return report


def snapshot(manifest, work, reviews):
    errors = []
    for entry in manifest["plans"]:
        try:
            plan, plan_sha = read(entry["path"], entry["sha256"])
            status_file = Path(plan["work"]) / "STATUS.json"
            if not status_file.exists():
                continue
            status, _ = read(status_file)
            for record, state in completed_cases(plan, status):
                identity = plan_sha + ":" + record["key"]
                if identity not in reviews:
                    reviews[identity] = verify_and_review(plan, plan_sha, record, state, work)
                    print(record["key"], state, "independently reviewed", flush=True)
        except (OSError, ValueError, KeyError) as error:
            errors.append(dict(plan=entry["path"], error=f"{type(error).__name__}: {error}"))
    storage.write_json(work / "STATUS.json", dict(state="OBSERVING_INDIVIDUAL_RESULTS",
        heartbeat_epoch=time.time(), reviews=[dict(key=r["key"], state=r["state"],
            plan_sha256=r["plan_sha256"], repair_pending=r["repair_pending"]) for r in reviews.values()],
        evidence_errors=errors, observer_only=True, automatic_scientific_edits=False))
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    work = storage.local_scratch(args.work)
    if work.exists():
        raise FileExistsError("fresh review work required; no automatic retry")
    manifest, _ = read(args.plan, args.plan_sha256)
    if manifest["execution_host"] != storage.physical_host() or not 0 < manifest["duration_s"] <= 7 * 86400:
        raise ValueError("observer host or duration differs")
    storage.configure(work)
    end = time.time() + manifest["duration_s"]
    storage.write_json(work / "LAUNCH.json", dict(pid=os.getpid(), start_ticks=Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19],
        plan_sha256=args.plan_sha256, source_sha256=storage.digest(__file__), deadline_epoch=end), immutable=True)
    storage.write_json(work / "REGRESSION_POLICY.json", dict(required_si_thicknesses_um=REPAIR_CONTROLS,
        required_scope="Every affected available operator, prior passing controls, geometry extremes and failed cases.",
        acceptance="Original gates unchanged; a mechanism repair is not propagated until regression results are recorded.",
        missing_control_is_not_a_pass=True, fresh_evaluation_epoch_required=True,
        teacher_reuse="Only complete sealed evidence with unchanged geometry, mesh, materials, loading and extraction.",
        no_automatic_scientific_edits=True), immutable=True)
    reviews = {}
    while time.time() < end:
        snapshot(manifest, work, reviews)
        if args.once:
            break
        time.sleep(30.)


if __name__ == "__main__":
    main()
