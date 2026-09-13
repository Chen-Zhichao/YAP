#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
python_bin="${PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl-yap-fig17}"

cd "${repo_root}"
"${python_bin}" "${script_dir}/run_cases.py" "$@"
"${python_bin}" "${script_dir}/legacy_overlay.py"
"${python_bin}" "${script_dir}/make_plot.py"
"${python_bin}" "${script_dir}/verify.py"
