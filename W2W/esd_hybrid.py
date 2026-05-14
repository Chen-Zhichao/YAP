#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Compatibility shim for the split W2W ESD implementation.

The W2W ESD model now lives in:
  - esd_yield_calculator.py for analytical/modeling flow
  - esd_yield_simulator.py for Monte Carlo flow
"""

from esd_yield_calculator import (  # noqa: F401
    center_contact_case,
    center_die_indices,
    die_esd_yield_calculator,
    pad_esd_yield_map_generator,
    stack_esd_yield_calculator,
)
from esd_yield_simulator import (  # noqa: F401
    choose_center_die_index,
    esd_failure_simulator,
)


if __name__ == "__main__":
    raise SystemExit(
        "W2W esd_hybrid.py has been split. Import esd_yield_calculator.py "
        "for modeling or esd_yield_simulator.py for simulation."
    )
