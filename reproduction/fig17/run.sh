#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-/u1/ee/zhichao/anaconda3/envs/yap_env/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-yap-fig17}"
export PYTHONDONTWRITEBYTECODE=1

cd "$REPO_ROOT"
"$PYTHON" reproduction/fig17/run_cases.py "$@"
"$PYTHON" reproduction/fig17/legacy_overlay.py
"$PYTHON" reproduction/fig17/make_plot.py
"$PYTHON" reproduction/fig17/verify.py
