#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python3}"

"${python_bin}" "${script_dir}/run_cases.py" "$@"
archive_path="${script_dir}/../../../YAPlus_1204/replica_distance.zip"
if [[ -f "${archive_path}" ]]; then
    "${python_bin}" "${script_dir}/audit_uploaded_archive.py" --archive "${archive_path}"
else
    echo "Historical archive not present; skipping optional source audit: ${archive_path}"
fi
"${python_bin}" "${script_dir}/make_plot.py"
"${python_bin}" "${script_dir}/verify.py"
