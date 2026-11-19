"""Deterministic stale-id regression; no production points or Cu integration."""
from dataclasses import dataclass
import gc
import inspect
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from tools.operators import acceleration


@dataclass
class Cu:
    plastic_strain: object
    creep_strain: object
    equivalent_plastic_strain: object
    equivalent_creep_strain: object


@dataclass
class Entry:
    committed_ref: object


def fresh_cu():
    return Cu(np.zeros((8,6)), np.zeros((8,6)), np.zeros(8), np.zeros(8))


def evaluate(self, macro_coordinates, bubble_coordinates, temperature_c, delta_time_s,
             committed, *, elapsed_time_s, creep_active=False, plastic_active=False):
    pass


def fixture(reference):
    owner = object.__new__(acceleration._Owner)
    state = SimpleNamespace(cu_state=fresh_cu(), birth_macro_coordinates=np.zeros(6),
                            birth_bubble_coordinates=np.zeros(6))
    owner.signatures = {'evaluate': inspect.signature(evaluate)}
    owner.contexts = {}
    owner.cache = {id(state.cu_state): Entry(reference or weakref.ref(state.cu_state))}
    owner.acceleration = SimpleNamespace(_CacheEntry=Entry)
    owner.admit = lambda *args: True
    operator = SimpleNamespace(cu_point_count=8, coordinate_size=12)
    args = (np.zeros(6), np.zeros(6), 25., 0., state)
    return owner, operator, args, state


def test_dead_cache_entry_with_recycled_id_does_not_reject_fresh_cu():
    previous = fresh_cu()
    reference = weakref.ref(previous)
    del previous
    gc.collect()
    assert reference() is None
    owner, operator, args, state = fixture(reference)
    before = state.cu_state.plastic_strain.copy()
    owner.arguments(operator, 'evaluate', args, {'elapsed_time_s': 300.})
    np.testing.assert_array_equal(state.cu_state.plastic_strain, before)
    assert state.cu_state.plastic_strain.flags.writeable


def test_actual_live_cached_cu_still_must_be_frozen():
    owner, operator, args, state = fixture(None)
    with pytest.raises(ValueError, match='cached incoming Cu must remain frozen'):
        owner.arguments(operator, 'evaluate', args, {'elapsed_time_s': 300.})
    for value in vars(state.cu_state).values():
        value.flags.writeable = False
    owner.arguments(operator, 'evaluate', args, {'elapsed_time_s': 300.})
