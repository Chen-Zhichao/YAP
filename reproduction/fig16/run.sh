#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl-yap-fig16}"

"${python_bin}" "${script_dir}/run_cases.py" "$@"
"${python_bin}" "${script_dir}/wafer_scaling_audit.py"
"${python_bin}" "${script_dir}/make_plot.py"
"${python_bin}" "${script_dir}/verify.py"
