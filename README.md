# YAP-stack

D2W and W2W assembly-order overlay-yield modeling. Use Linux and Python 3.11 or 3.12.
Run commands from the root of your own YAP checkout.

## 1. Set Up

Choose a private directory in your own `/app` project. All generated inputs,
results and caches go here; shared operators are read-only and are not in Git.

```bash
export YAP_RUN_ROOT=/app/your_project/YAP_runs
umask 077
mkdir -p -m 700 "$YAP_RUN_ROOT"
export YAP_WORK_DIR="$YAP_RUN_ROOT/work"
export TMPDIR="$YAP_WORK_DIR/temp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export PIP_CACHE_DIR="$YAP_WORK_DIR/pip"
export PYTHONDONTWRITEBYTECODE=1
# Clear inherited cache paths; YAP configures its own directories automatically.
unset NUMBA_CACHE_DIR MPLCONFIGDIR PYTHONPYCACHEPREFIX PYTHONUSERBASE \
  XDG_CACHE_HOME XDG_CONFIG_HOME XDG_DATA_HOME XDG_STATE_HOME \
  XDG_RUNTIME_DIR JOBLIB_TEMP_FOLDER
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# First time only. In later shells, just activate this environment.
python3 -m venv "$YAP_RUN_ROOT/venv"
. "$YAP_RUN_ROOT/venv/bin/activate"
python -B -m pip install -r requirements.txt
```

Bind the shared operator data in every new shell. These paths apply on the
shared server; request access if any dependency is unreadable.

```bash
R=/app/nanocad/projects/personal/zchen/YAP_STACK
DATA="$R/distribution/yap_component_operators_20261021_v1"
export YAP_COMPONENT_OPERATOR_CATALOG_V2="$DATA/CATALOG.json"
export YAP_NATIVE_RUNTIME_DIR="$DATA/runtime/native"
export YAP_RUNTIME_RESOURCES="$DATA/runtime/RESOURCES.json"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=4
python -B component_operators.py list
```

## 2. Create Your Design

This example generates four 40-um bodies, BI pitch 2 um, all five assembly
orders, and two concurrent order workers. Choose `MODE=d2w` or `MODE=w2w`.
Use a new name for each design; existing inputs are never overwritten.

```bash
MODE=d2w
DOMAIN=${MODE^^}
NAME=${MODE}_4hi_p2_t40
GENERATED="$YAP_RUN_ROOT/designs"

python -B generate_assembly_design.py \
  --bonding-mode "$MODE" --name "$NAME" \
  --dies 4 --pitch 2 --total-thickness 40 \
  --initial-bows 10 -10 20 -20 \
  --orders all --order-workers 2 --output-root "$GENERATED"
```

Change `--dies` (or `--wafers`) to the body count and provide one signed
center-minus-mid-edge bow per body, in um.
`--orders` accepts `all`, `sequential`, `balanced`, or a quoted tree such as
`'((1,2),(3,4))'`. Eight bodies have 429 fixed-layer-order trees; `all`
supports at most eight. `--order-workers 1` runs serially.

Current component coverage:
- BI2: total thickness 20, 30, 40, 50, 60, 70, 80, 90 or 100 um;
  intermediate thicknesses from 40 through 50 um are also available.
- BI6: total thickness 50 um.
- BI3: total thickness 50 um; add
  `--operator-package component:BI3_SI37_COARSE45` (coarse-time qualification).
- Current layer geometry uses d/p = 0.5, BEOL 10 um and BI 1.5 um per face.

D2W bodies keep their prepared thickness. W2W starts from 775-um wafers and
uses F2B-only bonding with thinning/backside BI preparation. W2W input bows
refer to the original 775-um wafers, not the thinned wafers.

## 3. Edit Settings

Generated files are under `$GENERATED/$DOMAIN`:

| File | What To Set |
| --- | --- |
| `input/$NAME/stack.yaml` | Initial bows, assembly orders, geometry, thermal process, operator selection and `mechanics.order_workers` |
| `configs/$NAME/$NAME.yaml` | Placement, rotation, `MAGNIFICATION_CORRECTION` (gamma), pad and other overlay parameters in the selected modeling section |
| `input/$NAME/compact_layout.yaml` | Compact pad layout and critical-region settings |
| `input/$NAME/*.3dbv`, `*.3dbx`, `*.3dbf` | Body dimensions, stack connectivity, technology and pad geometry |

Prefer regenerating a design when changing thickness, pitch or body count:
these values must agree across the input files. Initial bows and selected
orders can be edited directly in `stack.yaml`. Only changing initial bow
reuses the operators but reruns the assembly histories. The compact example
is a layout surrogate, not an arbitrary real bump map.

## 4. Run

Check the generated configuration first:

```bash
python -B "$DOMAIN/calculator_main.py" \
  --config "$GENERATED/$DOMAIN/configs/$NAME/$NAME.yaml" \
  --mode "${MODE}_modeling" --ds_name "$NAME" \
  --ds_dir "$GENERATED/$DOMAIN/input/$NAME" \
  --mechanism-filter overlay \
  --output-dir "$YAP_RUN_ROOT/results/$DOMAIN" --plan-only
```

Remove `--plan-only` from that command to compute mechanics and yield. Before
launching, check available RAM, disk space and active jobs. The default memory
gate requires 40 GiB available; additional order workers need more resources.

## 5. Read Results

Each run creates `$YAP_RUN_ROOT/results/$DOMAIN/$NAME/<timestamp_id>/`:
- `SUMMARY.csv`: each interface's yield, magnification and residual overlay.
- `RESULT.json`: stack overlay yield and comparison of the requested orders.
- `MECHANICS.json` and `route_*_mechanics.json`: computed mechanics histories.
- `progress.log` and `order_workers/`: progress and parallel-worker records.

Assembly-order runs report overlay yield, not total manufacturing yield.
Component bow checks do not independently validate whole-stack predictions;
these assembly results remain experimental. Operators are not included by
`pip install` or Git, and another server needs the complete dependency data.
