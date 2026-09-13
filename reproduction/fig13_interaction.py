#!/usr/bin/env python3
"""Compatibility entry point for the archived Figure 13 reproduction."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    command = [sys.executable, str(HERE / "fig13" / "run_sweep.py"), *sys.argv[1:]]
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
