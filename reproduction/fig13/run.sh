#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python3}"

if ! "${python_bin}" -c 'import cv2, matplotlib, numpy, omegaconf, scipy' >/dev/null 2>&1; then
    echo "ERROR: ${python_bin} does not contain all YAP dependencies." >&2
    echo "Set PYTHON to a prepared environment; see README.md." >&2
    exit 2
fi

"${python_bin}" "${script_dir}/run_sweep.py" --side w2w "$@" &
w2w_pid=$!
"${python_bin}" "${script_dir}/run_sweep.py" --side d2w "$@" &
d2w_pid=$!
wait "${w2w_pid}"
wait "${d2w_pid}"
"${python_bin}" "${script_dir}/make_plot.py"
"${python_bin}" "${script_dir}/verify.py"
