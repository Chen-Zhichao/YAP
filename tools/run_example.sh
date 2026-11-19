#!/usr/bin/env bash
# Run the documented, accelerated 4-Hi D2W example in a fresh user-owned directory.
set -euo pipefail

usage() {
    printf '%s\n' 'Usage: bash tools/run_example.sh --run-dir /absolute/output/directory [--data-root ROOT] [--python PYTHON] [--plan-only]'
}

run_base=''
data_root="${YAP_DATA_ROOT:-/app/nanocad/projects/personal/zchen/YAP_STACK}"
python="${YAP_PYTHON:-}"
extra=()
while (($#)); do
    case "$1" in
        --run-dir|--data-root|--python)
            if (($# < 2)); then usage >&2; exit 2; fi
            case "$1" in
                --run-dir) run_base="$2" ;;
                --data-root) data_root="$2" ;;
                --python) python="$2" ;;
            esac
            shift 2 ;;
        --plan-only) extra=(--plan-only); shift ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
if [[ "$run_base" != /* || "$run_base" == / || "$data_root" != /* ]]; then
    usage >&2
    exit 2
fi
if [[ ! -d "$data_root/distribution/operators" ]]; then
    printf 'Operator data not found below %s; set --data-root or YAP_DATA_ROOT.\n' "$data_root" >&2
    exit 2
fi

version_check='import sys,importlib.util,importlib.metadata as m; names=("numpy","scipy","sympy","pandas","omegaconf","numba","matplotlib","threadpoolctl","sklearn"); sys.exit(0 if sys.version_info[:2] in ((3,11),(3,12)) and all(importlib.util.find_spec(n) is not None for n in names) and m.version("numpy")=="1.26.4" and m.version("numba")=="0.59.1" else 1)'
if [[ -n "$python" ]]; then
    if ! "$python" -I -B -c "$version_check" >/dev/null 2>&1; then
        printf 'Select Python 3.11/3.12 with the requirements.txt dependencies using --python.\n' >&2
        exit 2
    fi
else
    for candidate in python python3.12 python3.11 python3 /w/ee.00/puneet/zhichao/anaconda3/bin/python3.12; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -I -B -c "$version_check" >/dev/null 2>&1; then
            python="$(command -v "$candidate")"
            break
        fi
    done
    if [[ -z "$python" ]]; then
        printf 'Python 3.11/3.12 with requirements.txt dependencies is required; select it with --python.\n' >&2
        exit 2
    fi
fi

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
umask 077
export YAP_DATA_ROOT="$data_root"
export YAP_RUN_ROOT="${run_base%/}/trial_$(date +%Y%m%d_%H%M%S)_$$"
export YAP_WORK_DIR="$YAP_RUN_ROOT/work"
export TMPDIR="$YAP_WORK_DIR/temp"
export TMP="$TMPDIR" TEMP="$TMPDIR" PIP_CACHE_DIR="$YAP_WORK_DIR/pip"
export PYTHONDONTWRITEBYTECODE=1
unset NUMBA_CACHE_DIR MPLCONFIGDIR PYTHONPYCACHEPREFIX PYTHONUSERBASE \
    XDG_CACHE_HOME XDG_CONFIG_HOME XDG_DATA_HOME XDG_STATE_HOME XDG_RUNTIME_DIR JOBLIB_TEMP_FOLDER
data="$data_root/distribution/operators/components"
export YAP_COMPONENT_OPERATOR_CATALOG_V2="$data/coverage_20261023_v9/CATALOG.json"
export YAP_NATIVE_RUNTIME_DIR="$data/runtime/native"
export YAP_RUNTIME_RESOURCES="$data/runtime/RESOURCES.json"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=8
name=d2w_4hi_p4_t50_fast
generated="$YAP_RUN_ROOT/designs"
printf 'Example run root: %s\nPython: %s\n' "$YAP_RUN_ROOT" "$python"
"$python" -B "$root/generate_assembly_design.py" \
    --bonding-mode d2w --name "$name" --dies 4 --pitch 4 --total-thickness 50 \
    --initial-bows 10 -10 20 -20 --orders all --order-workers 1 \
    --operator-package coupon:BI4_Prepared50_Quadrature19626 --output-root "$generated"
exec "$python" -B "$root/D2W/calculator_main.py" \
    --config "$generated/D2W/configs/$name/$name.yaml" --mode d2w_modeling --ds_name "$name" \
    --ds_dir "$generated/D2W/input/$name" --mechanism-filter overlay \
    --output-dir "$YAP_RUN_ROOT/results/D2W" "${extra[@]}"
