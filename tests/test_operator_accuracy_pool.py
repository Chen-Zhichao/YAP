"""Parallel assignments and peer ownership without native solver calls."""

import pytest

import operator_accuracy_pool as pool
import operator_grid as grid


def records(n):
    return [dict(key=f"case{i}") for i in range(n)]


@pytest.mark.parametrize("n,slots", [(5, 4), (6, 2), (4, 4)])
def test_disjoint_nonempty_partitions(n, slots):
    queues = pool.partitions(records(n), slots)
    assert len(queues) == slots
    assert all(queues)
    assert sorted(r["key"] for q in queues for r in q) == sorted(r["key"] for r in records(n))


@pytest.mark.parametrize("slots", [True, 0, 1, 5, 2.5])
def test_invalid_slot_budget(slots):
    with pytest.raises(ValueError):
        pool.partitions(records(6), slots)


def test_duplicate_assignments_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        pool.partitions([dict(key="same")] * 4, 2)


def test_old_pair_slots_remain_fixed():
    assert list(grid.group_slots(dict(schema="yap.operator_grid.parallel_pair.v1", jobs=[{}, {}]))) == [0, 1]
    with pytest.raises(ValueError):
        grid.group_slots(dict(schema="yap.operator_grid.parallel_pair.v1", jobs=[{}, {}, {}]))


@pytest.mark.parametrize("count", [2, 3, 4])
def test_screen_pool_all_slots(count):
    jobs = [dict(kind="screen", work=f"/job{i}") for i in range(count)]
    assert list(grid.group_slots(dict(schema="yap.operator_grid.parallel_screens.v1", execution_host="mfd", jobs=jobs))) == list(range(count))
    jobs[-1]["work"] = jobs[0]["work"]
    with pytest.raises(ValueError):
        grid.group_slots(dict(schema="yap.operator_grid.parallel_screens.v1", execution_host="mfd", jobs=jobs))


def test_all_authenticated_peers_but_no_foreign_solver_allowed(monkeypatch):
    jobs = [dict(kind="screen", work=f"/job{i}") for i in range(4)]
    plan = dict(schema="yap.operator_grid.parallel_screens.v1", execution_host="mfd", jobs=jobs)
    parent = lambda i: dict(pid=i + 100, start_ticks=str(i))
    peers = {i: dict(identity=parent(i), work=f"/job{i}", pins_sha256="pins") for i in range(4)}
    foreign = dict(pid=999, start_ticks="foreign")
    monkeypatch.setattr(grid, "ansys_blockers", lambda: [parent(1), parent(2), parent(3), foreign])
    monkeypatch.setattr(grid, "authenticate_group", lambda _: (plan, {}, {}))
    monkeypatch.setattr(grid, "group_child", lambda _, slot: peers[slot])
    monkeypatch.setattr(grid, "verify_pins", lambda *a: {})
    monkeypatch.setattr(grid, "process_identity", lambda pid: next((parent(i) for i in range(4) if parent(i)["pid"] == pid), None))
    monkeypatch.setattr(grid, "descendant_identities", lambda pid: [grid.process_identity(pid)])
    assert grid.foreign_ansys_blockers(dict(slot=0)) == [foreign]


def test_sysml_pool_ceiling():
    jobs = [dict(kind="screen", work=f"/job{i}") for i in range(3)]
    with pytest.raises(ValueError):
        grid.group_slots(dict(schema="yap.operator_grid.parallel_screens.v1", execution_host="sysml", jobs=jobs))


def test_cpu_zero_exit_without_receipts_is_not_success(tmp_path, monkeypatch):
    plan = dict(jobs=[dict(work=str(tmp_path))])
    child_plan = dict(pins_sha256="pins", candidates=records(1))
    monkeypatch.setattr(pool.grid, "read_record", lambda p, *a: ({}, "sha"))
    monkeypatch.setattr(pool.grid, "verify_pins", lambda *a: {})
    with pytest.raises(ValueError, match="completion"):
        pool.verify_cpu_completion(plan, 0, child_plan, "sha", dict(pid=123))


def test_pool_aggregate_resources(monkeypatch):
    r = dict(available_memory_gib=260., local_free_gib=600., cpu_busy_fraction=.2, logical_cpus=128)
    monkeypatch.setattr(pool.builder, "resource_check", lambda _: r)
    assert pool.resources("work", 4, 5) == r
    r["available_memory_gib"] = 200.
    with pytest.raises(RuntimeError, match="aggregate"):
        pool.resources("work", 4, 5)
    r["available_memory_gib"] = 260.
    r["local_free_gib"] = 500.
    with pytest.raises(RuntimeError, match="aggregate"):
        pool.resources("work", 4, 5)
