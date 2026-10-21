"""Thermal-screen contracts and lifecycle guards; no physical solver calls."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import operator_thermal_validation as thermal


def process():
    return SimpleNamespace(prebond_s=300., room_c=25., anneal_c=350., ramp_s=650.,
        dwell_s=3600., cool_s=650., creep_prebond=False, creep_ramp=False,
        creep_dwell=True, creep_cool=False)


def test_thermal_screen_never_implies_production_admission():
    scope = thermal.scope()
    assert scope["development_screen_only"]
    assert not scope["thermal_geometry_validated"]
    assert not scope["runtime_admitted"]
    assert {"independent_free_equilibrium_bow", "Cu_field_closure", "temporal_convergence",
            "assembly_history_and_thinning"} <= set(scope["pending"])


@pytest.mark.parametrize("h", [.007, .0085, .027, .0885])
def test_virgin_reversing_history_has_continuous_clock(h):
    q, times, temperatures, creep, amplitude = thermal.history(process(), h)
    assert q.shape == (1081, 134)
    assert np.isfinite(q).all()
    assert np.all(np.diff(times) > 0.)
    np.testing.assert_array_equal(times[[0, 160, 200, 360, 720, 1080]],
                                  [300., 950., 4550., 5200., 10100., 15000.])
    assert set(temperatures[[0, 360, 720, 1080]]) == {25.}
    assert set(temperatures[[160, 200, 520, 560, 880, 920]]) == {350.}
    assert creep.sum() == 120
    assert not creep[0] and not creep[160] and creep[161] and creep[200] and not creep[201]
    assert np.all(q[[0, 360, 720, 1080]] == 0.)
    assert np.all(q[:, 6:] == 0.)
    assert np.linalg.matrix_rank(q[:, :6] / amplitude) == 6
    for column in range(6):
        assert q[:, column].min() < 0. < q[:, column].max()


@pytest.mark.parametrize("h", [0., -1., np.inf, np.nan])
def test_bad_thickness_cannot_generate_history(h):
    with pytest.raises(ValueError, match="schedule"):
        thermal.history(process(), h)


def test_process_temperature_and_birth_contract_cannot_drift():
    p = process()
    p.anneal_c = 300.
    with pytest.raises(ValueError):
        thermal.history(p, .027)
    p = process()
    p.prebond_s = 1.
    with pytest.raises(ValueError):
        thermal.history(p, .027)
    with pytest.raises(ValueError):
        thermal.history(process(), .027, cycles=1)


def test_teacher_legacy_origin_is_shifted_from_layer_center():
    q = np.zeros((1, 134))
    q[0, :6] = [.001, -.001, .002, .1, -.2, .3]
    row = thermal.macro_rows(q, [300.], .027)[0]
    center = .0115 + .027 / 2
    assert row["eps_x0"] + center * row["kappa_x_1_per_mm"] == pytest.approx(q[0, 0])
    assert row["eps_z0"] + center * row["kappa_z_1_per_mm"] == pytest.approx(q[0, 1])
    assert row["gamma_xz0"] + center * row["kappa_xz_1_per_mm"] == pytest.approx(q[0, 2])


def test_scoring_near_zero_uses_absolute_deadband():
    assert thermal.component_gate([1e-5, .10], [0., .10], 1e-5)["passed"]
    assert not thermal.component_gate([1.1e-5], [0.], 1e-5)["passed"]
    assert thermal.component_gate([1.09], [1.], 1e-5)["passed"]
    assert not thermal.component_gate([1.11], [1.], 1e-5)["passed"]


@pytest.mark.parametrize("p,r", [([np.nan], [0.]), ([0.], [np.inf]), ([1., 2.], [1.])])
def test_scoring_rejects_nonfinite_or_mismatched_data(p, r):
    with pytest.raises(ValueError):
        thermal.component_gate(p, r, 1e-5)


def test_teacher_uses_exact_geometry_shear_and_no_force_rerun():
    plan = dict(source="/scientific", cards={name: {"path": f"/cards/{name}.json"}
                                            for name in ("cu", "noncu", "process")})
    command = thermal.teacher_command(plan, {"si_thickness_um": 8.5}, Path("/scratch/test"))
    assert "--force-ansys" not in command
    assert command[command.index("--substrate-um") + 1] == "8.5"
    assert command[command.index("--max-result-set") + 1] == "1081"
    assert "--enable-inplane-shear-twist" in command
    assert "--export-face-reactions" in command
    assert command[command.index("--ansys-np") + 1] == "8"
    assert 0 < int(command[command.index("--ansys-timeout-sec") + 1]) < thermal.STAGE_SECONDS
    assert all(not part.startswith("/tmp") for part in command)
    assert "--plan-only" in thermal.teacher_command(plan, {"si_thickness_um": 8.5},
                                                   Path("/scratch/test"), plan_only=True)


def test_wait_never_takes_over_after_predecessor_failure(tmp_path, monkeypatch):
    p = dict(work=str(tmp_path), launch_sha256="a", campaign_sha256="b",
             identity=dict(pid=42), archive=str(tmp_path / "archive"), si_thicknesses_um=[28.5])
    plan = dict(predecessor=p, created_epoch=thermal.time.time(), expires_epoch=thermal.time.time() + 100)
    monkeypatch.setattr(thermal.grid, "read_record", lambda path, *args: (
        {"state": "STOP_REQUIRES_REVIEW"} if path.name == "STATUS.json" else {}, "x"))
    monkeypatch.setattr(thermal.grid, "process_identity", lambda _: p["identity"])
    with pytest.raises(RuntimeError, match="stopped/failed"):
        thermal.wait_build_queue(plan, tmp_path, {})


def test_predecessor_completion_requires_entire_bound_batch(tmp_path, monkeypatch):
    identity = dict(pid=42, sid=42, pgid=42)
    p = dict(work=str(tmp_path), launch_sha256="a", campaign_sha256="b",
        identity=identity, archive=str(tmp_path / "archive"), si_thicknesses_um=[28.5, 48.5])
    completed = dict(state="BATCH_COMPLETE_NOT_FULL_GRID", worker_pid=42,
        completed=[dict(thickness_um=28.5, state="ARCHIVED_STRUCTURAL_CANDIDATE")])
    monkeypatch.setattr(thermal.grid, "read_record", lambda path, *a: (completed, "x"))
    monkeypatch.setattr(thermal.grid, "process_identity", lambda _: None)
    with pytest.raises(ValueError, match="whole batch"):
        thermal.wait_build_queue(dict(predecessor=p, created_epoch=thermal.time.time(),
            expires_epoch=thermal.time.time() + 100), tmp_path, {})


def test_two_workers_cannot_share_one_attempt(tmp_path):
    with thermal.exclusive_worker(tmp_path):
        with pytest.raises(BlockingIOError):
            with thermal.exclusive_worker(tmp_path):
                pytest.fail("second worker acquired same attempt")


def archive_record_fixture(monkeypatch, tmp_path):
    record = dict(root=str(tmp_path), key="tsv_si_h8p5", si_thickness_um=8.5,
                  files_sha256={"data": "a" * 64, "audit/LAYER.json": "b" * 64})
    archive = dict(files_sha256={"datasets/data.npz": "a" * 64},
                   thermal_geometry_validated=False, runtime_admitted=False)
    audit = dict(schema="yap.operator_grid.layer_candidate.v1", layer="tsv_si",
        common_cell_side_um=6., tsv_pitch_um=6., tsv_diameter_um=3., library_key=record["key"],
        thermal_geometry_validated=False, d2w_runtime_enabled=False, w2w_runtime_enabled=False)
    monkeypatch.setattr(thermal, "candidate", lambda _: record)
    monkeypatch.setattr(thermal.grid, "read_record", lambda p, *a: (
        archive if p.name == "ARCHIVE.json" else audit, "b" * 64))
    return record, archive, audit


def test_archive_authority_is_independent_of_build_host(monkeypatch, tmp_path):
    record, _, _ = archive_record_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(thermal.grid, "process_identity", lambda _: pytest.fail("no build process required"))
    assert thermal.archived_candidate(tmp_path, 8.5) == record
    thermal.wait_build_queue(dict(schema="yap.operator_grid.thermal_archive_screen.v1", candidates=[record]), tmp_path, {})
    with pytest.raises(ValueError, match="geometry/admission"):
        thermal.archived_candidate(tmp_path, 7.)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "#snapshot/file", ".snapshot/file", "a//b", "."])
def test_archive_inventory_cannot_escape_root(monkeypatch, tmp_path, name):
    _, archive, _ = archive_record_fixture(monkeypatch, tmp_path)
    archive["files_sha256"] = {name: "a" * 64}
    with pytest.raises(ValueError, match="inventory"):
        thermal.archived_candidate(tmp_path, 8.5)


@pytest.mark.parametrize("field,value", [("tsv_pitch_um", 10.), ("tsv_diameter_um", 4.),
    ("common_cell_side_um", 12.), ("layer", "bi"), ("thermal_geometry_validated", 0),
    ("w2w_runtime_enabled", True)])
def test_archive_rejects_wrong_geometry_and_admission(monkeypatch, tmp_path, field, value):
    _, _, audit = archive_record_fixture(monkeypatch, tmp_path)
    audit[field] = value
    with pytest.raises(ValueError, match="geometry/admission"):
        thermal.archived_candidate(tmp_path, 8.5)


def scientific_receipt():
    record = dict(key="tsv_si_h8p5", files_sha256={"data": "a" * 64})
    receipt = dict(**thermal.scope(), key=record["key"], states=1081,
        candidate_sha256=record["files_sha256"], state="FAIL_PRESCRIBED_SCREEN",
        gates={k: dict(passed=k != "eps_x") for k in (*thermal.Q6,
            "bottom_ports", "top_ports", "bubble_equilibrium", "inelastic_excitation")})
    return record, receipt


def test_only_archive_queue_continues_scientific_failures():
    record, receipt = scientific_receipt()
    plan = dict(schema="yap.operator_grid.thermal_archive_screen.v1", continue_scientific_failures=True)
    assert thermal.scientific_screen_result(plan, record, receipt)
    assert not thermal.scientific_screen_result({}, record, receipt)
    assert thermal.completed_screen_state([dict(state=receipt["state"])]) == "BATCH_SCREEN_COMPLETE_WITH_FAILURES_NOT_QUALIFIED"
    assert thermal.completed_screen_state([dict(state="PASS_PRESCRIBED_SCREEN")]) == "BATCH_SCREEN_COMPLETE_NOT_QUALIFIED"


@pytest.mark.parametrize("field,value", [("state", "OTHER_FAILURE"), ("states", 1000),
    ("key", "wrong"), ("candidate_sha256", {}), ("runtime_admitted", 0), ("gates", {})])
def test_malformed_failure_never_authorizes_continuation(field, value):
    record, receipt = scientific_receipt()
    receipt[field] = value
    with pytest.raises(ValueError, match="malformed"):
        thermal.scientific_screen_result(dict(schema="yap.operator_grid.thermal_archive_screen.v1",
            continue_scientific_failures=True), record, receipt)


def test_archive_queue_reserves_retained_cases(monkeypatch, tmp_path):
    resource = dict(local_free_gib=500., own_ansys_processes=[])
    monkeypatch.setattr(thermal.builder, "resource_check", lambda _: resource)
    assert thermal.archive_queue_resources(tmp_path, 4) == resource
    with pytest.raises(RuntimeError, match="needs"):
        thermal.archive_queue_resources(tmp_path, 6)
    resource["own_ansys_processes"] = ["foreign"]
    with pytest.raises(RuntimeError, match="ANSYS"):
        thermal.archive_queue_resources(tmp_path, 1)


def test_child_expired_budget_cannot_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(thermal.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(TimeoutError):
        thermal.run_child(["no"], tmp_path / "log", 0.)


def test_cleanup_does_not_signal_reused_pids(monkeypatch):
    original = dict(pid=123, start_ticks="10")
    monkeypatch.setattr(thermal.grid, "process_identity", lambda _: dict(pid=123, start_ticks="11"))
    monkeypatch.setattr(thermal.os, "kill", lambda *a: pytest.fail("reused PID must not be signalled"))
    thermal.cleanup_owned({(123, "10"): original})


def test_cleanup_tracks_nested_session_by_identity_not_group(monkeypatch):
    original = dict(pid=123, start_ticks="10", pgid=123, sid=123)
    live = [original]
    monkeypatch.setattr(thermal.grid, "process_identity", lambda _: live[0] if live else None)
    calls = []
    def terminate(identity, sig):
        calls.append((identity["pid"], sig))
        live.clear()
    monkeypatch.setattr(thermal, "signal_identity", terminate)
    thermal.cleanup_owned({(123, "10"): original})
    assert calls == [(123, thermal.signal.SIGTERM)]


def lifecycle_fixture(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(thermal.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(thermal.time, "time", lambda: 100. + clock[0])
    monkeypatch.setattr(thermal.time, "sleep", lambda t: clock.__setitem__(0, clock[0] + t))
    monkeypatch.setattr(thermal, "publish_worker_scope", lambda _: None)
    child = OwnedChild()
    child.returncode = 0
    root = dict(pid=123, start_ticks="root", pgid=123, sid=123, command=["wrapper"])
    helper = dict(pid=124, start_ticks="helper", pgid=123, sid=123, command=["helper"])
    observed = thermal.ProcessInventory()
    observed[(123, "root")] = root
    observed[(124, "helper")] = helper
    return clock, child, observed, helper


def test_success_drain_waits_for_helper_without_signals(monkeypatch):
    clock, child, observed, helper = lifecycle_fixture(monkeypatch)
    monkeypatch.setattr(thermal.grid, "process_identity", lambda pid: helper if pid == 124 and clock[0] < 1. else None)
    monkeypatch.setattr(thermal.grid, "proven_descendants", lambda _: [helper] if clock[0] < 1. else [])
    monkeypatch.setattr(thermal.grid, "own_process_census", lambda: [dict(helper, state="S")] if clock[0] < 1. else [])
    monkeypatch.setattr(thermal, "signal_identity", lambda *a: pytest.fail("success must not signal"))
    thermal.drain_success(child, observed, 200.)
    assert clock[0] == 1.
    assert not observed and observed.sessions == {123}


def test_live_helper_remains_discovery_anchor_after_root_exit(monkeypatch):
    _, child, observed, helper = lifecycle_fixture(monkeypatch)
    late = dict(pid=125, start_ticks="late", pgid=125, sid=125, command=["late"])
    current = {124: helper, 125: late}
    monkeypatch.setattr(thermal.grid, "process_identity", lambda pid: current.get(pid))
    anchors = []
    def discover(anchor):
        anchors.append(anchor["pid"])
        return [helper, late]
    monkeypatch.setattr(thermal.grid, "proven_descendants", discover)
    thermal.refresh_owned(child, observed)
    assert anchors == [124]
    assert (125, "late") in observed and observed.sessions == {123, 125}


def test_unknown_session_member_blocks_seal_without_kill_authority(monkeypatch):
    clock, child, observed, _ = lifecycle_fixture(monkeypatch)
    monkeypatch.setattr(thermal.grid, "process_identity", lambda _: None)
    monkeypatch.setattr(thermal.grid, "proven_descendants", lambda _: pytest.fail("no live anchor"))
    monkeypatch.setattr(thermal.grid, "own_process_census", lambda: [dict(pid=999, pgid=123, sid=123, state="S")])
    monkeypatch.setattr(thermal, "signal_identity", lambda *a: pytest.fail("unknown is not owned"))
    with pytest.raises(TimeoutError, match="exit drain"):
        thermal.drain_success(child, observed, 200.)
    assert clock[0] == 60.
    assert not observed and observed.sessions == {123}


def test_drain_is_capped_by_original_deadline(monkeypatch):
    clock, child, observed, helper = lifecycle_fixture(monkeypatch)
    monkeypatch.setattr(thermal.grid, "process_identity", lambda pid: helper if pid == 124 else None)
    monkeypatch.setattr(thermal.grid, "proven_descendants", lambda _: [helper])
    with pytest.raises(TimeoutError):
        thermal.drain_success(child, observed, 100.3)
    assert clock[0] == pytest.approx(.3)


def test_final_census_cannot_extend_stage_deadline(monkeypatch):
    clock, child, observed, _ = lifecycle_fixture(monkeypatch)
    monkeypatch.setattr(thermal.grid, "process_identity", lambda _: None)
    def census():
        clock[0] = 5.
        return []
    monkeypatch.setattr(thermal.grid, "own_process_census", census)
    with pytest.raises(TimeoutError):
        thermal.drain_success(child, observed, 101.)


def test_unknown_or_unreadable_scope_retains_group_lock(monkeypatch):
    _, child, observed, _ = lifecycle_fixture(monkeypatch)
    monkeypatch.setattr(thermal.grid, "process_identity", lambda _: None)
    monkeypatch.setattr(thermal.grid, "own_process_census", lambda: [dict(pid=999, pgid=123, sid=123, state="S")])
    assert thermal.group_has_live_children([child], [observed])
    def broken():
        raise RuntimeError("census unavailable")
    monkeypatch.setattr(thermal.grid, "own_process_census", broken)
    assert thermal.group_has_live_children([child], [observed])


def test_worker_scope_handoff_is_launch_bound(tmp_path, monkeypatch):
    observed = thermal.ProcessInventory()
    observed.scope_work = tmp_path
    observed.scope_identity = dict(pid=123, start_ticks="worker")
    observed.scope_launch_sha = "launch"
    thermal.builder.write_json(tmp_path / "PROCESS_SCOPE.json", dict(worker_identity=observed.scope_identity,
        launch_sha256="launch", groups=[124], sessions=[124]))
    thermal.import_worker_scope(observed)
    assert observed.sessions == {124}
    thermal.builder.write_json(tmp_path / "PROCESS_SCOPE.json", dict(worker_identity=dict(pid=123, start_ticks="reuse"),
        groups=[999], sessions=[999]))
    with pytest.raises(ValueError, match="does not match its launch"):
        thermal.import_worker_scope(observed)
    assert 999 not in observed.sessions


def test_exited_worker_cannot_drop_scope_receipt(tmp_path):
    observed = thermal.ProcessInventory()
    observed.scope_work, observed.scope_identity = tmp_path, dict(pid=123, start_ticks="worker")
    observed.process = SimpleNamespace(poll=lambda: 0)
    with pytest.raises(RuntimeError, match="lacks launch-bound lifecycle"):
        thermal.import_worker_scope(observed)


def test_wrong_launch_hash_does_not_grant_scope(tmp_path):
    observed = thermal.ProcessInventory()
    observed.scope_work, observed.scope_identity = tmp_path, dict(pid=123, start_ticks="worker")
    observed.scope_launch_sha = "expected"
    thermal.builder.write_json(tmp_path / "PROCESS_SCOPE.json", dict(worker_identity=observed.scope_identity,
        launch_sha256="wrong", groups=[999], sessions=[999]))
    with pytest.raises(ValueError, match="does not match its launch"):
        thermal.import_worker_scope(observed)
    assert not observed.sessions


def test_failed_diagnostic_cannot_bypass_cleanup(tmp_path, monkeypatch):
    child = OwnedChild()
    monkeypatch.setattr(thermal.subprocess, "Popen", lambda *a, **k: child)
    def broken(*a, **k):
        raise RuntimeError("diagnostic or discovery failed")
    monkeypatch.setattr(thermal.grid, "process_identity", broken)
    monkeypatch.setattr(thermal, "lifecycle_diagnostic", broken)
    cleaned = []
    monkeypatch.setattr(thermal, "cleanup_spawned", lambda *a: cleaned.append(a[0].pid))
    with pytest.raises(RuntimeError, match="diagnostic or discovery"):
        thermal.run_child(["fixture"], tmp_path / "log", thermal.time.time() + 60.)
    assert cleaned == [123]


def test_child_observer_failure_retains_owned_cleanup(tmp_path, monkeypatch):
    child = OwnedChild()
    monkeypatch.setattr(thermal.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(thermal, "pin_spawned", lambda *a: None)
    cleaned = []
    monkeypatch.setattr(thermal, "cleanup_spawned", lambda *a: cleaned.append(a[0].pid))
    def observer():
        raise RuntimeError("storage floor")
    with pytest.raises(RuntimeError, match="storage floor"):
        thermal.run_child(["fixture"], tmp_path / "log", thermal.time.time() + 60., observer=observer)
    assert cleaned == [123]


@pytest.mark.parametrize("signum", [thermal.signal.SIGTERM, thermal.signal.SIGHUP, thermal.signal.SIGINT])
def test_training_signal_unwinds_owned_child_before_guard_restores(tmp_path, monkeypatch, signum):
    child = OwnedChild()
    monkeypatch.setattr(thermal.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(thermal, "pin_spawned", lambda *a: None)
    handlers = {value: thermal.signal.getsignal(value) for value in
                (thermal.signal.SIGTERM, thermal.signal.SIGHUP, thermal.signal.SIGINT)}
    cleaned = []
    def cleanup(*a):
        assert thermal.signal.getsignal(signum) is not handlers[signum]
        cleaned.append(a[0].pid)
    monkeypatch.setattr(thermal, "cleanup_spawned", cleanup)
    def observer():
        thermal.signal.raise_signal(signum)
    with pytest.raises(KeyboardInterrupt, match="training interrupted"):
        with thermal.grid.training_signal_guard():
            thermal.run_child(["fixture"], tmp_path / "log", thermal.time.time() + 60., observer=observer)
    assert cleaned == [123]
    assert all(thermal.signal.getsignal(value) is handler for value, handler in handlers.items())


def test_quarantine_retains_ownership_despite_signal_and_broken_diagnostics(tmp_path, monkeypatch):
    child = SimpleNamespace(pid=123, poll=lambda: 0)
    observed = thermal.ProcessInventory()
    refreshes, sleeps, masks = [], [], []
    def refresh(*a):
        refreshes.append(True)
        if len(refreshes) == 1:
            raise KeyboardInterrupt("census interrupted")
    def broken(*a, **k):
        raise OSError("diagnostic output unavailable")
    monkeypatch.setattr(thermal, "refresh_owned", refresh)
    monkeypatch.setattr(thermal, "scoped_members", lambda _: [])
    monkeypatch.setattr(thermal, "lifecycle_diagnostic", broken)
    monkeypatch.setattr("builtins.print", broken)
    monkeypatch.setattr(thermal.time, "sleep", sleeps.append)
    monkeypatch.setattr(thermal.signal, "pthread_sigmask", lambda how, value: masks.append((how, value)) or set())
    thermal.quarantine_spawned(child, observed, tmp_path / "log", RuntimeError("cleanup uncertain"))
    assert len(refreshes) == 2 and sleeps == [30.]
    assert thermal.signal.SIGHUP in masks[0][1]
    assert masks[-1][0] == thermal.signal.SIG_SETMASK


def test_pidfd_unavailable_never_uses_numeric_pid_signal(monkeypatch):
    import ctypes
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(thermal.os, "kill", lambda *a: pytest.fail("no numeric PID signalling"))
    with pytest.raises(RuntimeError, match="pidfd unavailable"):
        thermal.signal_identity(dict(pid=123, start_ticks="root"), thermal.signal.SIGTERM)


def test_real_successful_wrapper_exits_before_its_helper(tmp_path, monkeypatch):
    import subprocess
    import sys
    import time
    monkeypatch.setattr(thermal, "publish_worker_scope", lambda _: None)
    wrapper = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(2)']); time.sleep(.3)"
    process = subprocess.Popen([sys.executable, "-B", "-c", wrapper], start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    observed = thermal.ProcessInventory()
    try:
        thermal.pin_spawned(process, observed)
        end = time.monotonic() + 10.
        while process.poll() is None:
            thermal.refresh_owned(process, observed)
            assert time.monotonic() < end
            time.sleep(.03)
        assert thermal.live_owned(observed)
        thermal.drain_success(process, observed, time.time() + 10.)
        assert not thermal.live_owned(observed) and not thermal.scoped_members(observed)
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10.)


def test_rom_seal_receipt_is_hash_bound(tmp_path):
    thermal.builder.write_json(tmp_path / "ROM_SEAL.json", {"files_sha256": {}}, immutable=True)
    expected = thermal.builder.digest(tmp_path / "ROM_SEAL.json")
    thermal.verify_rom_seal(tmp_path, expected)
    (tmp_path / "ROM_SEAL.json").write_text('{"files_sha256":{},"changed":true}')
    with pytest.raises(ValueError, match="SHA256"):
        thermal.verify_rom_seal(tmp_path, expected)


def input_fixture(tmp_path):
    names = ("dataset.npz", "port_basis.npz", "lifts.npz")
    hashes = {}
    for name in names:
        (tmp_path / name).write_bytes(name.encode())
        hashes[name] = thermal.builder.digest(tmp_path / name)
    record = dict(dataset=names[0], basis=names[1], lifts=names[2], files_sha256=hashes)
    contract = dict(candidate=record, scope=thermal.SCOPE, input_sha256=dict(hashes))
    thermal.builder.write_json(tmp_path / "INPUT_CONTRACT.json", contract)
    return record, contract


def test_input_checks_are_deduplicated_only_within_one_call(tmp_path, monkeypatch):
    record, _ = input_fixture(tmp_path)
    original, reads = thermal.builder.digest, []
    def tracked(path):
        reads.append(path.name)
        return original(path)
    monkeypatch.setattr(thermal.builder, "digest", tracked)
    thermal.verify_inputs(record, tmp_path)
    assert all(reads.count(name) == 1 for name in record["files_sha256"])
    (tmp_path / "dataset.npz").write_bytes(b"modified after preflight")
    with pytest.raises(ValueError, match="private input drift"):
        thermal.verify_inputs(record, tmp_path)
    assert reads.count("dataset.npz") == 2


@pytest.mark.parametrize("alias", [False, True])
def test_input_checks_reject_conflicting_hashes(tmp_path, alias):
    record, contract = input_fixture(tmp_path)
    if alias:
        contract["input_sha256"]["./dataset.npz"] = "0" * 64
    else:
        contract["input_sha256"]["dataset.npz"] = "0" * 64
    thermal.builder.write_json(tmp_path / "INPUT_CONTRACT.json", contract)
    with pytest.raises(ValueError, match="conflicting input"):
        thermal.verify_inputs(record, tmp_path)


def test_score_binds_complete_receipt_to_previously_checked_seal(tmp_path, monkeypatch):
    record = dict(si_thickness_um=7.)
    sealed = {"ROM_COMPLETE.json": "complete", "ROM_RESPONSE.npz": "response", "ROM_CU_STATES.npz": "states"}
    checks = []
    monkeypatch.setattr(thermal, "import_api", lambda _: (None, None))
    monkeypatch.setattr(thermal, "verify_inputs", lambda *a: checks.append("inputs"))
    monkeypatch.setattr(thermal, "verify_rom_seal", lambda *a: sealed)
    def read(path, expected):
        assert path.name == "ROM_COMPLETE.json" and expected == "complete"
        return dict(response_sha256="wrong", cu_states_sha256="states"), "complete"
    monkeypatch.setattr(thermal.grid, "read_record", read)
    monkeypatch.setattr(thermal.builder, "digest", lambda _: pytest.fail("must not rehash sealed large outputs"))
    with pytest.raises(ValueError, match="sealed ROM prediction changed"):
        thermal.score({}, record, tmp_path, seal_sha256="seal")
    assert checks == ["inputs"]


class OwnedChild:
    pid = 123
    def __init__(self):
        self.returncode = None
        self.calls = []
    def poll(self):
        return self.returncode
    def terminate(self):
        self.calls.append("terminate")
    def kill(self):
        self.calls.append("kill")
    def wait(self, timeout):
        self.calls.append("wait")
        self.returncode = 0
        return 0


def test_owned_inventory_limits_live_processes_not_historical_churn(monkeypatch):
    root = dict(pid=123, start_ticks="root")
    child = OwnedChild()
    current = {123: root}
    monkeypatch.setattr(thermal.grid, "process_identity", lambda pid: current.get(pid))
    monkeypatch.setattr(thermal.grid, "descendant_identities", lambda _: list(current.values()))
    observed = {(123, "root"): root}
    for index in range(600):
        current.clear()
        current.update({123: root, 1000 + index: dict(pid=1000 + index, start_ticks=str(index))})
        thermal.refresh_owned(child, observed)
        assert len(observed) == 2


def test_inventory_keeps_live_detached_child_but_drops_reused_pid(monkeypatch):
    root = dict(pid=123, start_ticks="root")
    detached = dict(pid=124, start_ticks="detached")
    current = {123: root, 124: detached, 125: dict(pid=125, start_ticks="new")}
    observed = {(123, "root"): root, (124, "detached"): detached,
                (125, "old"): dict(pid=125, start_ticks="old")}
    monkeypatch.setattr(thermal.grid, "process_identity", lambda pid: current.get(pid))
    monkeypatch.setattr(thermal.grid, "descendant_identities", lambda _: [root])
    thermal.refresh_owned(OwnedChild(), observed)
    assert set(observed) == {(123, "root"), (124, "detached")}


def test_owned_child_exec_transition_keeps_pid_start_identity(monkeypatch):
    before = dict(pid=123, start_ticks="root", command=["launcher"])
    after = dict(before, command=["native_wrapper"])
    monkeypatch.setattr(thermal.grid, "process_identity", lambda _: after)
    monkeypatch.setattr(thermal.grid, "descendant_identities", lambda _: [after])
    observed = {(123, "root"): before}
    thermal.refresh_owned(OwnedChild(), observed)
    assert observed == {(123, "root"): after}


def test_live_inventory_still_has_finite_bound(monkeypatch):
    current = {pid: dict(pid=pid, start_ticks=str(pid)) for pid in range(123, 380)}
    monkeypatch.setattr(thermal.grid, "process_identity", lambda pid: current.get(pid))
    monkeypatch.setattr(thermal.grid, "descendant_identities", lambda _: list(current.values()))
    with pytest.raises(RuntimeError, match="live owned process inventory"):
        thermal.refresh_owned(OwnedChild(), {(123, "123"): current[123]})


def test_cleanup_reaped_root_never_discovers_reused_numeric_pid(monkeypatch):
    child = OwnedChild()
    child.returncode = 0
    monkeypatch.setattr(thermal.grid, "descendant_identities", lambda _: pytest.fail("reaped root"))
    monkeypatch.setattr(thermal, "cleanup_owned", lambda _: None)
    thermal.cleanup_spawned(child, {})
    assert child.calls == ["wait"]


def test_pair_cleanup_contains_both_even_when_one_cleanup_fails(monkeypatch):
    children = [SimpleNamespace(pid=1), SimpleNamespace(pid=2)]
    called = []
    def cleanup(child, _):
        called.append(child.pid)
        if child.pid == 1:
            raise RuntimeError("first cleanup failed")
    monkeypatch.setattr(thermal, "cleanup_spawned", cleanup)
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        thermal.cleanup_group_children(children, [{}, {}])
    assert called == [1, 2]


@pytest.fixture
def live_group(tmp_path, monkeypatch):
    g = thermal.grid
    monkeypatch.setattr(g, "local_work", lambda p: Path(p).resolve())
    monkeypatch.setattr(thermal.builder, "scratch_root", lambda _: tmp_path)
    monkeypatch.setattr(thermal.builder, "physical_host", lambda: "sysml")
    identity = g.process_identity(thermal.os.getpid())
    jobs = [dict(kind="build", work=str(tmp_path / "build"), controller="/build.py"),
            dict(kind="screen", work=str(tmp_path / "screen"), controller="/screen.py")]
    plan = dict(schema="yap.operator_grid.parallel_pair.v1", work=str(tmp_path), jobs=jobs,
        execution_host=thermal.builder.physical_host(), expires_epoch=thermal.time.time() + 60,
        controllers_sha256={str(Path(g.__file__).resolve()): thermal.builder.digest(g.__file__)})
    thermal.builder.write_json(tmp_path / "GROUP_PLAN.json", plan)
    sha = thermal.builder.digest(tmp_path / "GROUP_PLAN.json")
    thermal.builder.write_json(tmp_path / "LAUNCH.json", dict(worker_identity=identity,
        command=identity["command"], plan_sha256=sha))
    binding = dict(work=str(tmp_path), plan_sha256=sha,
        launch_sha256=thermal.builder.digest(tmp_path / "LAUNCH.json"), slot=0)
    with g.native_host_lock(tmp_path) as held:
        thermal.builder.write_json(tmp_path / "HOST_LOCK.json", dict(**held, worker_identity=identity,
            plan_sha256=sha))
        yield binding, held, jobs


def test_parallel_authority_requires_live_kernel_exclusive_lock(live_group):
    binding, held, _ = live_group
    assert thermal.grid.authenticate_group(binding)[2]["fd"] == held["fd"]
    thermal.fcntl.flock(held["fd"], thermal.fcntl.LOCK_UN)
    with pytest.raises(ValueError, match="kernel exclusive"):
        thermal.grid.authenticate_group(binding)


@pytest.mark.parametrize("fault", ["plan_hash", "launch_hash", "inode", "fd", "owner", "expired", "controller"])
def test_parallel_authority_rejects_stale_or_forged_evidence(live_group, fault):
    binding, held, _ = live_group
    root = Path(binding["work"])
    if fault in ("plan_hash", "launch_hash"):
        binding["plan_sha256" if fault == "plan_hash" else "launch_sha256"] = "0" * 64
    elif fault in ("expired", "controller"):
        plan = thermal.builder.read_json(root / "GROUP_PLAN.json")
        if fault == "expired":
            plan["expires_epoch"] = 0
        else:
            plan["controllers_sha256"][str(Path(thermal.grid.__file__).resolve())] = "0" * 64
        thermal.builder.write_json(root / "GROUP_PLAN.json", plan)
    else:
        lock = thermal.builder.read_json(root / "HOST_LOCK.json")
        if fault == "inode":
            lock["inode"] += 1
        elif fault == "fd":
            lock["fd"] = 1000000
        else:
            lock["worker_identity"]["start_ticks"] = "forged"
        thermal.builder.write_json(root / "HOST_LOCK.json", lock)
    with pytest.raises((ValueError, FileNotFoundError)):
        thermal.grid.authenticate_group(binding)


def test_parallel_lock_excludes_legacy_worker(live_group):
    binding, _, _ = live_group
    with pytest.raises(BlockingIOError):
        with thermal.grid.native_host_lock(Path(binding["work"])):
            pytest.fail("legacy worker must not bypass paired supervisor")


def test_peer_allowlist_never_hides_unknown_solver(live_group, monkeypatch):
    binding, _, jobs = live_group
    g = thermal.grid
    root = Path(binding["work"])
    peer = dict(pid=999999, start_ticks="peer", command=["python", "-B", "/screen.py"])
    peer_work = Path(jobs[1]["work"])
    thermal.builder.write_json(peer_work / "PLAN.json", dict(group_binding=dict(binding, slot=1), pins_sha256="pins"))
    thermal.builder.write_json(peer_work / "LAUNCH.json", dict(worker_identity=peer, command=peer["command"],
        plan_sha256=thermal.builder.digest(peer_work / "PLAN.json")))
    thermal.builder.write_json(root / "CHILD1.json", dict(identity=peer, work=str(peer_work),
        group_plan_sha256=binding["plan_sha256"], plan_filename="PLAN.json",
        plan_sha256=thermal.builder.digest(peer_work / "PLAN.json"), pins_sha256="pins",
        launch_sha256=thermal.builder.digest(peer_work / "LAUNCH.json")))
    thermal.builder.write_json(root / "CHILD0.json", {})
    thermal.publish_group_ready(root, binding["plan_sha256"])
    actual_identity = g.process_identity
    monkeypatch.setattr(g, "process_identity", lambda pid: peer if pid == peer["pid"] else actual_identity(pid))
    monkeypatch.setattr(g, "own_process_census", lambda: [dict(pid=peer["pid"], ppid=thermal.os.getpid())])
    own_solver = dict(pid=999998, start_ticks="owned")
    foreign_solver = dict(pid=999997, start_ticks="unknown")
    monkeypatch.setattr(g, "ansys_blockers", lambda: [own_solver, foreign_solver])
    monkeypatch.setattr(g, "descendant_identities", lambda _: [peer, own_solver])
    monkeypatch.setattr(g, "verify_pins", lambda *a: None)
    assert g.foreign_ansys_blockers(binding) == [foreign_solver]
    monkeypatch.setattr(g, "process_identity", lambda pid: None if pid == peer["pid"] else actual_identity(pid))
    assert g.foreign_ansys_blockers(binding) == [own_solver, foreign_solver]


def test_parallel_parent_death_contains_stage_and_retains_lock_until_cleanup(tmp_path):
    import json
    import subprocess
    import sys
    import time
    root_code = '''
import json,os,signal,sys,time
from pathlib import Path
import operator_grid as g
import operator_thermal_validation as t
root=Path(sys.argv[1]); parent=json.loads(sys.argv[2]); lock=json.loads(sys.argv[3])
os.environ['YAP_PARALLEL_SUPERVISOR']=json.dumps(dict(parent=parent,**lock))
g.bootstrap_group_environment()
def authenticate(_):
    if g.process_identity(parent['pid']) != parent:
        raise RuntimeError('parent gone')
    return {},parent,lock
g.authenticate_group=authenticate
g.assert_group_worker=lambda *a: None
leaf="""
import os,signal,sys,time
from pathlib import Path
import build_prepared_operators as b
root=Path(sys.argv[1]); descriptor=int(sys.argv[2])
try:
    os.fstat(descriptor)
except OSError:
    leaked=False
else:
    leaked=True
def stop(*args):
    b.write_json(root/'CLEANING.json',{'pid':os.getpid()})
    time.sleep(1.5)
    sys.exit(0)
signal.signal(signal.SIGTERM,stop)
b.write_json(root/'LEAF_READY.json',{'pid':os.getpid(),'lock_fd_leaked':leaked})
time.sleep(60)
"""
with g.worker_solver_lock(root, {'work':str(root)}):
    t.builder.write_json(root/'ROOT_READY.json',{'identity':g.process_identity(os.getpid())})
    t.run_child([sys.executable,'-B','-c',leaf,str(root),str(lock['fd'])],root/'leaf.log',time.time()+60)
'''
    parent_code = '''
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
import operator_grid as g
import build_prepared_operators as b
root=Path(sys.argv[1]); root_code=sys.argv[2]
descriptor=os.open(root/'lock',os.O_CREAT|os.O_RDWR,0o600)
fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
info=os.fstat(descriptor)
lock={'fd':descriptor,'device':info.st_dev,'inode':info.st_ino}
b.write_json(root/'GROUP_READY.json',{})
with (root/'root.log').open('x') as stream:
    signal_mask=__import__('signal').pthread_sigmask(__import__('signal').SIG_BLOCK,
        {__import__('signal').SIGTERM,__import__('signal').SIGINT,__import__('signal').SIGALRM})
    subprocess.Popen([sys.executable,'-B','-c',root_code,str(root),json.dumps(g.process_identity(os.getpid())),json.dumps(lock)],
        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(descriptor,))
    __import__('signal').pthread_sigmask(__import__('signal').SIG_SETMASK,signal_mask)
time.sleep(60)
'''
    parent = subprocess.Popen([sys.executable, "-B", "-c", parent_code, str(tmp_path), root_code],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    owned = []
    def wait_file(name):
        deadline = time.monotonic() + 15
        while not (tmp_path / name).exists():
            if time.monotonic() >= deadline:
                pytest.fail(f"lifecycle fixture did not reach {name}")
            time.sleep(.05)
        return json.loads((tmp_path / name).read_text())
    try:
        root = wait_file("ROOT_READY.json")["identity"]
        owned.append(root)
        leaf = wait_file("LEAF_READY.json")
        owned.append(thermal.grid.process_identity(leaf["pid"]))
        assert not leaf["lock_fd_leaked"]
        parent.kill()
        parent.wait(timeout=5)
        wait_file("CLEANING.json")
        thermal.os.kill(root["pid"], thermal.signal.SIGTERM)
        with (tmp_path / "lock").open("a") as stream:
            with pytest.raises(BlockingIOError):
                thermal.fcntl.flock(stream, thermal.fcntl.LOCK_EX | thermal.fcntl.LOCK_NB)
        deadline = time.monotonic() + 15
        while any(thermal.grid.process_identity(p["pid"]) is not None for p in owned):
            if time.monotonic() >= deadline:
                pytest.fail("owned root/stage survived parent-death containment")
            time.sleep(.05)
        with (tmp_path / "lock").open("a") as stream:
            thermal.fcntl.flock(stream, thermal.fcntl.LOCK_EX | thermal.fcntl.LOCK_NB)
    finally:
        if parent.poll() is None:
            owned.extend(thermal.grid.descendant_identities(parent.pid))
        for identity in owned:
            if identity and thermal.grid.process_identity(identity["pid"]) == identity:
                try:
                    thermal.os.kill(identity["pid"], thermal.signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)


def test_failed_cleanup_rediscovery_still_reaps_owned_child(monkeypatch):
    child = OwnedChild()
    def failed(*a):
        raise RuntimeError("census unavailable")
    monkeypatch.setattr(thermal.grid, "descendant_identities", failed)
    thermal.cleanup_spawned(child, {})
    assert child.calls == ["terminate", "wait"]


def test_known_cleanup_exception_cannot_skip_popen_fallback(monkeypatch):
    child = OwnedChild()
    monkeypatch.setattr(thermal.grid, "descendant_identities", lambda _: [])
    def failed(*a):
        raise RuntimeError("known cleanup failed")
    monkeypatch.setattr(thermal, "cleanup_owned", failed)
    with pytest.raises(RuntimeError, match="known cleanup"):
        thermal.cleanup_spawned(child, {})
    assert child.calls == ["terminate", "wait"]


def test_initial_identity_discovery_failure_reaps_owned_child(tmp_path, monkeypatch):
    child = OwnedChild()
    monkeypatch.setattr(thermal.subprocess, "Popen", lambda *a, **k: child)
    def failed(*a):
        raise RuntimeError("initial identity unavailable")
    monkeypatch.setattr(thermal.grid, "process_identity", failed)
    monkeypatch.setattr(thermal.grid, "descendant_identities", lambda _: [])
    monkeypatch.setattr(thermal.grid, "own_process_census", lambda: [])
    with pytest.raises(RuntimeError, match="initial identity"):
        thermal.run_child(["fixture"], tmp_path / "log", thermal.time.time() + 60.)
    assert child.calls == ["terminate", "wait"]


def test_launch_publication_failure_cleans_up_before_worker_can_start(tmp_path, monkeypatch):
    child = OwnedChild()
    plan = dict(candidates=[dict(key="h7")], archive=str(tmp_path / "archive"))
    args = SimpleNamespace(plan_sha256="p")
    monkeypatch.setattr(thermal, "checked_plan", lambda _: (plan, tmp_path))
    monkeypatch.setattr(thermal.grid, "read_record", lambda *a: (dict(plan_sha256="p", keys=["h7"], solver_launched=False), "x"))
    monkeypatch.setattr(thermal.builder, "configure", lambda _: {})
    command = []
    def spawn(value, **kwargs):
        command.extend(value)
        return child
    monkeypatch.setattr(thermal.subprocess, "Popen", spawn)
    monkeypatch.setattr(thermal.grid, "process_identity", lambda _: dict(pid=123, start_ticks="10", command=command))
    monkeypatch.setattr(thermal.grid, "descendant_identities", lambda _: [])
    monkeypatch.setattr(thermal, "cleanup_owned", lambda _: None)
    def write(path, *a, **k):
        if path.name == "LAUNCH.json":
            raise RuntimeError("publication failed")
    monkeypatch.setattr(thermal.builder, "write_json", write)
    with pytest.raises(RuntimeError, match="publication failed"):
        thermal.launch(args)
    assert child.calls == ["terminate", "wait"]


@pytest.mark.parametrize("launcher", ["build", "screen"])
def test_slow_launch_publication_cannot_outlive_outer_deadline(tmp_path, monkeypatch, launcher):
    child = OwnedChild()
    command = ["worker"]
    identity = dict(pid=123, start_ticks="10", command=command)
    clock = [99.]
    monkeypatch.setattr(thermal.time, "time", lambda: clock[0])
    monkeypatch.setattr(thermal.grid, "wait_exec_identity", lambda *a, **k: identity)
    monkeypatch.setattr(thermal.subprocess, "Popen", lambda *a, **k: child)
    cleaned = []
    monkeypatch.setattr(thermal, "cleanup_spawned", lambda *a: cleaned.append(a[0].pid))
    original = thermal.builder.write_json
    if launcher == "build":
        def write(path, *a, **k):
            original(path, *a, **k)
            if path.name == "LAUNCH.json":
                clock[0] = 101.
        monkeypatch.setattr(thermal.builder, "write_json", write)
        prepared = (tmp_path, {}, command, "campaign", "pins")
        args = SimpleNamespace(archive=tmp_path / "archive", review_sha256="review")
        launch = thermal.grid.launch
    else:
        monkeypatch.setattr(thermal.builder, "copy_checked", lambda *a: clock.__setitem__(0, 101.))
        prepared = (dict(archive=str(tmp_path / "archive")), tmp_path, {}, command)
        args = SimpleNamespace(plan_sha256="plan")
        launch = thermal.launch
    with pytest.raises(TimeoutError, match="publication deadline"):
        launch(args, prepared=prepared, preparation_deadline=100.)
    assert cleaned == [child.pid]
    assert not (tmp_path / "LAUNCH_PUBLISHED.json").exists()


def closed_build_fixture(tmp_path, monkeypatch):
    g = thermal.grid
    work, archive = tmp_path / "build", tmp_path / "archive"
    identity = dict(pid=123, start_ticks="456", sid=123, pgid=123, command=["python", "-B", "worker"])
    campaign = dict(schema="yap.operator_grid.campaign.v4", work=str(work), archive=str(archive),
        execution_host="sysml", si_thicknesses_um=[78.5], source_pins_sha256="pins")
    thermal.builder.write_json(work / "CAMPAIGN.json", campaign)
    campaign_sha = thermal.builder.digest(work / "CAMPAIGN.json")
    launch = dict(worker_pid=123, worker_identity=identity, command=identity["command"],
        campaign_sha256=campaign_sha, source_pins_sha256="pins")
    thermal.builder.write_json(work / "LAUNCH.json", launch)
    target = archive / "p2_tsv_si_h78p5"
    audit = dict(state="PASS_STRUCTURAL_CANDIDATE", thickness_um=78.5, dataset_sha256="dataset",
                 thermal_geometry_validated=False)
    thermal.builder.write_json(target / "audit/LAYER.json", audit)
    audit_sha = thermal.builder.digest(target / "audit/LAYER.json")
    thermal.builder.write_json(target / "ARCHIVE.json", dict(runtime_admitted=False,
        files_sha256={"audit/LAYER.json": audit_sha}))
    record = dict(thickness_um=78.5, archive=str(target), state="ARCHIVED_STRUCTURAL_CANDIDATE",
        archive_sha256=thermal.builder.digest(target / "ARCHIVE.json"), audit_sha256=audit_sha, dataset_sha256="dataset")
    thermal.builder.write_json(archive / "COMPLETED_p2_tsv_si_h78p5.json", record)
    state = dict(state="BATCH_COMPLETE_NOT_FULL_GRID", worker_pid=123, completed=[record])
    thermal.builder.write_json(work / "STATUS.json", state)
    thermal.builder.write_json(archive / "BATCH_COMPLETE.json", state)
    monkeypatch.setattr(thermal.builder, "physical_host", lambda: "sysml")
    monkeypatch.setattr(g, "verify_pins", lambda *a: None)
    monkeypatch.setattr(g, "process_identity", lambda _: None)
    monkeypatch.setattr(g, "predecessor_sessions_idle", lambda *a, **k: True)
    binding = dict(work=str(work), archive=str(archive), identity=identity, si_thicknesses_um=[78.5],
        launch_sha256=thermal.builder.digest(work / "LAUNCH.json"), campaign_sha256=campaign_sha)
    return binding, state


def test_completed_v4_build_can_be_bound_after_worker_exit(tmp_path, monkeypatch):
    binding, _ = closed_build_fixture(tmp_path, monkeypatch)
    thermal.completed_build_binding(binding)


@pytest.mark.parametrize("fault", ["launch", "whole_batch", "layer", "audit", "live", "orphan", "pins"])
def test_completed_build_binding_cannot_skip_evidence(tmp_path, monkeypatch, fault):
    binding, state = closed_build_fixture(tmp_path, monkeypatch)
    work, archive = Path(binding["work"]), Path(binding["archive"])
    if fault == "launch":
        binding["identity"] = dict(binding["identity"], start_ticks="reused")
    elif fault == "whole_batch":
        state["completed"] = []
        thermal.builder.write_json(work / "STATUS.json", state)
        thermal.builder.write_json(archive / "BATCH_COMPLETE.json", state)
    elif fault == "layer":
        thermal.builder.write_json(archive / "COMPLETED_p2_tsv_si_h78p5.json", {})
    elif fault == "audit":
        thermal.builder.write_json(archive / "p2_tsv_si_h78p5/audit/LAYER.json", {})
    elif fault == "live":
        monkeypatch.setattr(thermal.grid, "process_identity", lambda _: binding["identity"])
    elif fault == "orphan":
        monkeypatch.setattr(thermal.grid, "predecessor_sessions_idle", lambda *a, **k: False)
    else:
        def drift(*a):
            raise ValueError("source drift")
        monkeypatch.setattr(thermal.grid, "verify_pins", drift)
    with pytest.raises((ValueError, RuntimeError)):
        thermal.completed_build_binding(binding)


def thermal_predecessor_fixture(tmp_path, monkeypatch):
    identity = dict(pid=123, start_ticks="456", pgid=123, sid=123, command=["python", "worker"])
    archive = tmp_path / "archive"
    plan = dict(archive=str(archive), candidates=[dict(key="tsv_si_h7", files_sha256={"dataset": "frozen"})])
    thermal.builder.write_json(tmp_path / "PLAN.json", plan)
    thermal.builder.write_json(tmp_path / "LAUNCH.json", dict(worker_identity=identity))
    state = dict(state="BATCH_SCREEN_COMPLETE_NOT_QUALIFIED", worker_identity=identity,
                 completed=[dict(key="tsv_si_h7", state="PASS_PRESCRIBED_SCREEN")])
    thermal.builder.write_json(tmp_path / "STATUS.json", state)
    thermal.builder.write_json(archive / "BATCH_SCREEN_COMPLETE.json", state)
    thermal.builder.write_json(archive / "tsv_si_h7/THERMAL_SCREEN.json",
                              dict(key="tsv_si_h7", state="PASS_PRESCRIBED_SCREEN", candidate_sha256={"dataset": "frozen"}))
    binding = dict(work=str(tmp_path), archive=str(archive), identity=identity, keys=["tsv_si_h7"],
        plan_sha256=thermal.builder.digest(tmp_path / "PLAN.json"),
        launch_sha256=thermal.builder.digest(tmp_path / "LAUNCH.json"))
    monkeypatch.setattr(thermal.grid, "process_identity", lambda _: None)
    monkeypatch.setattr(thermal.grid, "predecessor_sessions_idle", lambda *a, **k: True)
    return binding, state


def test_followup_requires_thermal_success_and_exit(tmp_path, monkeypatch):
    binding, _ = thermal_predecessor_fixture(tmp_path, monkeypatch)
    assert thermal.thermal_predecessor_complete(binding)
    monkeypatch.setattr(thermal.grid, "process_identity", lambda _: binding["identity"])
    assert not thermal.thermal_predecessor_complete(binding)


@pytest.mark.parametrize("fault", ["failed", "partial", "receipt", "case", "candidate_hash", "reused_pid", "orphan", "hash"])
def test_followup_rejects_failed_or_ambiguous_predecessors(tmp_path, monkeypatch, fault):
    binding, state = thermal_predecessor_fixture(tmp_path, monkeypatch)
    archive = Path(binding["archive"])
    if fault == "failed":
        state["state"] = "STOPPED_REQUIRES_REVIEW"
        thermal.builder.write_json(tmp_path / "STATUS.json", state)
    elif fault == "partial":
        state["completed"] = []
        thermal.builder.write_json(tmp_path / "STATUS.json", state)
        thermal.builder.write_json(archive / "BATCH_SCREEN_COMPLETE.json", state)
    elif fault == "receipt":
        thermal.builder.write_json(archive / "BATCH_SCREEN_COMPLETE.json", {})
    elif fault == "case":
        thermal.builder.write_json(archive / "tsv_si_h7/THERMAL_SCREEN.json", dict(state="FAIL_PRESCRIBED_SCREEN"))
    elif fault == "candidate_hash":
        thermal.builder.write_json(archive / "tsv_si_h7/THERMAL_SCREEN.json",
            dict(key="tsv_si_h7", state="PASS_PRESCRIBED_SCREEN", candidate_sha256={"dataset": "substituted"}))
    elif fault == "reused_pid":
        monkeypatch.setattr(thermal.grid, "process_identity", lambda _: dict(binding["identity"], start_ticks="789"))
    elif fault == "orphan":
        monkeypatch.setattr(thermal.grid, "predecessor_sessions_idle", lambda *a, **k: False)
    else:
        thermal.builder.write_json(tmp_path / "PLAN.json", {})
    with pytest.raises((ValueError, RuntimeError)):
        thermal.thermal_predecessor_complete(binding)


def test_followup_wait_does_not_launch_or_take_solver_ownership(tmp_path, monkeypatch):
    identity = dict(pid=123, start_ticks="456")
    plan = dict(predecessor=dict(wait_deadline_epoch=thermal.time.time() + 120), archive=str(tmp_path / "archive"))
    args = SimpleNamespace(work=tmp_path, plan_sha256="plan")
    thermal.builder.write_json(tmp_path / "LAUNCH_PUBLISHED.json", dict(plan_sha256="plan", launch_sha256="launch"))
    monkeypatch.setattr(thermal, "checked_followup", lambda _: (plan, tmp_path))
    monkeypatch.setattr(thermal.builder, "configure", lambda _: {})
    original_read = thermal.grid.read_record
    def read(path, *rest):
        if path.name == "LAUNCH.json":
            return dict(plan_sha256="plan", worker_identity=identity), "launch"
        return original_read(path)
    monkeypatch.setattr(thermal.grid, "read_record", read)
    monkeypatch.setattr(thermal.grid, "process_identity", lambda _: identity)
    monkeypatch.setattr(thermal, "thermal_predecessor_complete", lambda _: False)
    def forbidden(*a, **k):
        pytest.fail("waiting must not launch a build, lock solvers, or require idle resources")
    monkeypatch.setattr(thermal.grid, "launch", forbidden)
    monkeypatch.setattr(thermal.grid, "native_host_lock", forbidden)
    monkeypatch.setattr(thermal.grid, "check_case_resources", forbidden)
    def interrupt(_):
        raise RuntimeError("fixture interrupts passive wait")
    monkeypatch.setattr(thermal.time, "sleep", interrupt)
    with pytest.raises(RuntimeError, match="passive wait"):
        thermal.followup_worker(args)
    assert thermal.builder.read_json(tmp_path / "STOP.json")["state"] == "STOPPED_REQUIRES_REVIEW"


@pytest.mark.parametrize("fault", [None, "deadline", "identity", "exit_code"])
def test_followup_monitors_the_detached_worker_not_its_launcher(tmp_path, monkeypatch, fault):
    identity = dict(pid=123, start_ticks="456")
    child = SimpleNamespace(pid=123, poll=lambda: None, wait=lambda **k: 1 if fault == "exit_code" else 0)
    thermal.builder.write_json(tmp_path / "CAMPAIGN.json", dict(expires_epoch=thermal.time.time() + 60.))
    thermal.builder.write_json(tmp_path / "LAUNCH.json", dict(worker_identity=identity,
        campaign_sha256=thermal.builder.digest(tmp_path / "CAMPAIGN.json")))
    thermal.builder.write_json(tmp_path / "PROCESS_SCOPE.json", dict(worker_identity=identity,
        launch_sha256=thermal.builder.digest(tmp_path / "LAUNCH.json"), groups=[], sessions=[]))
    monkeypatch.setattr(thermal.grid, "process_identity", lambda _: dict(identity, start_ticks="789") if fault == "identity" else identity)
    monkeypatch.setattr(thermal.grid, "descendant_identities", lambda _: [])
    monkeypatch.setattr(thermal.grid, "proven_descendants", lambda _: [])
    monkeypatch.setattr(thermal.grid, "own_process_census", lambda: [])
    monkeypatch.setattr(thermal, "live_owned", lambda _: [])
    cleaned = []
    monkeypatch.setattr(thermal, "cleanup_spawned", lambda *a: cleaned.append(a[0].pid))
    deadline = thermal.time.time() + (-1 if fault == "deadline" else 60)
    if fault is None:
        thermal.monitor_followup_child(child, tmp_path, deadline, dict(state="BUILD"), tmp_path)
        assert not cleaned
    else:
        with pytest.raises((ValueError, RuntimeError, TimeoutError)):
            thermal.monitor_followup_child(child, tmp_path, deadline, dict(state="BUILD"), tmp_path)
        assert cleaned == [123]


def test_monitor_receipt_failure_still_cleans_the_new_child(tmp_path, monkeypatch):
    child = SimpleNamespace(pid=123)
    monkeypatch.setattr(thermal.grid, "process_identity", lambda _: dict(pid=123, start_ticks="456"))
    cleaned = []
    monkeypatch.setattr(thermal, "cleanup_spawned", lambda *a: cleaned.append(a[0].pid))
    with pytest.raises(FileNotFoundError):
        thermal.monitor_followup_child(child, tmp_path, thermal.time.time() + 60., dict(state="BUILD"), tmp_path)
    assert cleaned == [123]


def test_cleanup_never_rediscovers_a_reaped_and_reusable_root(monkeypatch):
    process = SimpleNamespace(pid=123, poll=lambda: 0, wait=lambda **k: 0)
    observed = {(123, "456"): dict(pid=123, start_ticks="456")}
    def unrelated(_):
        pytest.fail("a reaped root PID cannot establish new ancestry")
    monkeypatch.setattr(thermal.grid, "descendant_identities", unrelated)
    monkeypatch.setattr(thermal, "cleanup_owned", lambda values: None)
    thermal.cleanup_spawned(process, observed)
    assert set(observed) == {(123, "456")}


def test_completed_build_is_reauthenticated_at_ownership_release(tmp_path, monkeypatch):
    binding, _ = closed_build_fixture(tmp_path, monkeypatch)
    binding["completed_at_plan"] = True
    monkeypatch.setattr(thermal.grid, "own_process_census", lambda: [])
    thermal.completed_build_binding(binding)
    thermal.builder.write_json(Path(binding["archive"]) / "COMPLETED_p2_tsv_si_h78p5.json", {})
    with pytest.raises(ValueError, match="layer receipt"):
        thermal.wait_build_queue(dict(predecessor=binding, created_epoch=thermal.time.time(),
            expires_epoch=thermal.time.time() + 60), tmp_path, {})


@pytest.mark.parametrize("which", ["build", "screen"])
def test_expired_preparation_cannot_spawn_a_detached_worker(tmp_path, monkeypatch, which):
    def forbidden(*a, **k):
        pytest.fail("expired preparation cannot spawn or prepare a worker")
    monkeypatch.setattr(thermal.subprocess, "Popen", forbidden)
    if which == "build":
        monkeypatch.setattr(thermal.grid, "prepare_launch", forbidden)
        launch = thermal.grid.launch
    else:
        monkeypatch.setattr(thermal, "prepare_validation_launch", forbidden)
        launch = thermal.launch
    with pytest.raises(TimeoutError):
        launch(SimpleNamespace(), preparation_deadline=thermal.time.time() - 1)


@pytest.mark.parametrize("fault", [None, "print", "receipt"])
def test_grid_launch_owns_child_through_publication_and_return(tmp_path, monkeypatch, fault):
    child = OwnedChild()
    identity = dict(pid=child.pid, start_ticks="456", command=["worker"])
    args = SimpleNamespace(archive=tmp_path / "archive", review_sha256="review")
    monkeypatch.setattr(thermal.grid, "prepare_launch", lambda _: (tmp_path, {}, ["worker"], "campaign", "pins"))
    monkeypatch.setattr(thermal.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(thermal.grid, "process_identity", lambda _: identity)
    cleaned = []
    monkeypatch.setattr(thermal, "cleanup_spawned", lambda *a: cleaned.append(a[0].pid))
    def failure(*a, **k):
        raise RuntimeError("publication failure")
    if fault == "print":
        monkeypatch.setattr(thermal.grid, "print", failure, raising=False)
    elif fault == "receipt":
        monkeypatch.setattr(thermal.builder, "write_json", failure)
    if fault is None:
        assert thermal.grid.launch(args) is child
        assert thermal.builder.read_json(tmp_path / "LAUNCH.json")["worker_identity"] == identity
        assert not cleaned
    else:
        with pytest.raises(RuntimeError, match="publication failure"):
            thermal.grid.launch(args)
        assert cleaned == [child.pid]
