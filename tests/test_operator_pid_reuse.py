"""Observed process identities must survive unrelated PID reuse."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.operator_grid as grid
import tools.operator_thermal_validation as thermal


def test_initial_capture_remains_strict_but_observed_foreign_pid_is_absent(monkeypatch):
    pid = os.getpid()
    root = Path(f'/proc/{pid}')
    original = Path.stat

    def foreign(self, *args, **kwargs):
        info = original(self, *args, **kwargs)
        if self == root:
            return SimpleNamespace(st_uid=info.st_uid + 1)
        return info

    monkeypatch.setattr(Path, 'stat', foreign)
    with pytest.raises(ValueError, match='this UID'):
        grid.process_identity(pid)
    assert grid.process_identity(pid, expected_start_ticks='1234') is None


def test_wrong_birth_is_absent_for_observed_identity():
    assert grid.process_identity(os.getpid(), expected_start_ticks='0') is None
    assert grid.process_identity(os.getpid()) is not None


def test_descendant_recheck_drops_pid_reused_between_census_and_recheck(monkeypatch):
    root = dict(pid=17,start_ticks='101',ppid=1,pgid=17,sid=17,command=['owned'])
    child = dict(pid=19,start_ticks='102',ppid=17,pgid=19,sid=19,command=['owned-child'])
    calls = {19:0}

    def identity(pid, *, include_parent=False, expected_start_ticks=None):
        if pid == 17:
            return root if include_parent else {k:v for k,v in root.items() if k!='ppid'}
        calls[19] += 1
        if calls[19] > 1:
            return None
        return child

    monkeypatch.setattr(grid,'process_identity',identity)
    monkeypatch.setattr(grid,'own_process_census',lambda:[
        dict(pid=17,ppid=1,pgid=17,sid=17,state='S',comm='owned'),
        dict(pid=19,ppid=17,pgid=19,sid=19,state='S',comm='owned-child')])
    assert grid.proven_descendants({k:v for k,v in root.items() if k!='ppid'}) == []
    assert calls[19] == 2


def test_live_owned_does_not_treat_reused_pid_as_live(monkeypatch):
    previous = dict(pid=19,start_ticks='102',pgid=19,sid=19,command=['owned-child'])
    def lookup(pid):
        assert pid == 19
        return None
    monkeypatch.setattr(grid,'process_identity',lookup)
    assert thermal.live_owned({(19,'102'):previous}) == []
