# YAP-stack

YAP-stack models D2W and W2W assembly overlay yield, including the effect of
thermomechanical warpage and Cu inelasticity. It evaluates specified assembly
orders or enumerates all orders while keeping the physical layer order fixed.

## Requirements

- Linux and Python 3.11 or 3.12.
- Python dependencies listed in `requirements.txt`.
- Separately supplied, read-only mechanics operators. Operator data are not in Git.

The supplied migrated `coupon:` release requires Python 3.12 to match its
authenticated material-code fingerprints. Component models retain Python
3.11/3.12 support.

## Installation

From the YAP checkout, choose your own absolute directory for environments,
caches and results:

```bash
export YAP_RUN_ROOT=/absolute/path/to/YAP_runs
export TMPDIR="$YAP_RUN_ROOT/temp"
mkdir -p "$TMPDIR"
python3.12 -m venv "$YAP_RUN_ROOT/venv"
source "$YAP_RUN_ROOT/venv/bin/activate"
python -m pip install --cache-dir "$YAP_RUN_ROOT/pip-cache" -r requirements.txt
```

On the shared server, the example launcher can select the existing compatible
Python environment. Its default operator project is
`/app/nanocad/projects/personal/zchen/YAP_STACK`.
On another server, obtain the operator data separately and set `YAP_DATA_ROOT`
to the directory containing `distribution/operators/`.

## Quick Example

Run this from the YAP checkout; replace the output directory with your own:

```bash
bash tools/run_example.sh --run-dir /absolute/path/to/YAP_runs
```

The script sets up runtime paths, generates a design and evaluates all five
orders of a four-die stack. Each die is 10 x 10 mm and 50 um thick; bonding pitch
is 4 um. The explicit accelerated operator is
`coupon:BI4_Prepared50_Quadrature19626`.

An MFD run took about **112 seconds**, with approximately **3.5 MB of results**,
excluding the environment and operator data. Runtime varies with server load.
This is an experimental D2W quadrature example, not the full-GP default or a W2W
accuracy qualification. Use `--plan-only` to check inputs without mechanics.
Optional `--data-root` and `--python` select other installations.

## Custom Experiments

After activating your environment, generate consistent design and configuration
files. This example uses BI pitch 2 um, total die thickness 40 um and two order
workers:

```bash
python -B generate_assembly_design.py \
  --bonding-mode d2w --name demo_4hi --dies 4 --pitch 2 --total-thickness 40 \
  --initial-bows 10 -10 20 -20 --orders all --order-workers 2 \
  --output-root "$YAP_RUN_ROOT/designs"

python -B D2W/calculator_main.py \
  --config "$YAP_RUN_ROOT/designs/D2W/configs/demo_4hi/demo_4hi.yaml" \
  --mode d2w_modeling --ds_name demo_4hi \
  --ds_dir "$YAP_RUN_ROOT/designs/D2W/input/demo_4hi" \
  --mechanism-filter overlay --output-dir "$YAP_RUN_ROOT/results/D2W"
```

- Geometry and count: set `--dies`, `--pitch` and `--total-thickness` during generation.
  Total thickness includes both BIs, BEOL and Si/TSV.
- Initial bows: supply one signed center-minus-mid-edge bow in um per body.
- Orders: `all`, `sequential`, `balanced`, or a quoted tree such as `'((1,2),(3,4))'`.
  `all` supports up to eight bodies; eight bodies have 429 orders.
- Concurrency: `--order-workers 1` is serial; larger values request parallel orders.
- Generated `stack.yaml` stores bows, orders and mechanics settings; the full
  calculator YAML stores overlay and process parameters. Use a new design name
  when regenerating inputs.

For W2W, generate with `--bonding-mode w2w`, use the generated `W2W` paths and run
`W2W/calculator_main.py --mode w2w_modeling` with the same other arguments.
W2W uses the F2B thinning process; D2W dies retain their specified thickness.

## Operator Coverage

Thicknesses below are total prepared-body thicknesses in um.

| Operator Family | BI Pitch (um) | Total Thickness (um) | Assembly Mode |
| --- | --- | --- | --- |
| `component:` | 2, 3, 6 | 20-100, step 10 | D2W and W2W |
| `component:` | 4 | 50, using component catalog v12 | Experimental D2W and W2W |
| `coupon:BI<PITCH>_Prepared<TOTAL>` | 2-10, integer pitches | 20-100, step 10 | Experimental D2W only; exact 10 x 10 mm |

BI2 component models also support intermediate total thicknesses from 40 through
50 um. BI3 component models use a coarse mesh. Other intermediate geometries
must not be assumed supported. Use `--operator-package` to select an explicit
nondefault package; the generator otherwise selects a component model.

Availability and single-model validation do not independently validate every
assembly history. Whole-prepared `coupon:` models are not W2W models.

The same named `coupon:` selectors are preserved in `prepared_coupon_catalog.json`;
common runtime code loads numerical packets from the migrated release under
`$YAP_DATA_ROOT/distribution/operators/normalized/`, with no per-pitch Python loaders.
Set `YAP_NORMALIZED_OPERATOR_ROOT` to override that data root, or
`YAP_COUPON_OPERATOR_CATALOG` to select another catalog file. The old
`YAP_COUPON_OPERATOR_ROOT` and `YAP_SHARED_COUPON_OPERATOR_ROOT` settings are
replaced by these overrides; old loader bundles must be migrated, not reused.

## Results

The terminal prints progress and the result directory. Main outputs are
`RESULT.json`, `SUMMARY.csv` and `MECHANICS.json`, with input/source snapshots for
reproducibility. Results and caches belong in your selected run directory;
shared operators are read-only.

## 2.5D Packages

```bash
python -B package_assembly.py --config input/package_2p5d.yaml
```

Set interposer dimensions and each stack's existing design/config paths, center,
one assembly order and `annealing: sequential` or `batch` in the package YAML.
Optional `operator_package` overrides the stack's mechanics package, with the
same geometry/pad-layout checks as ordinary D2W runs.
The example attaches two completed 4-Hi stacks to a 100-um Si-plus-BI interposer.
Incoming interposer bow uses the existing manufacturing-source statistics;
transferring that calibration to its top BI is an explicit modeling assumption.
Final attachment is elastic, with ideal independent flattening and a shared
interposer source per package sample. Identical deterministic stack histories
are reused, while each stack retains independent placement-error draws.
Current footprints are square and aligned;
the affine model does not resolve position-dependent interposer deformation.
The final package anneal and all subsequent processes are assumed to cause no
additional yield loss (factor 1), so no further mechanics are simulated.
The reported yield therefore stops at completed room-temperature attachment.
Results are overlay-only unless explicit `other_yield_factor` values
are supplied per stack. `--plan-only` checks inputs without loading operators.

## License

Project source is provided under [Apache-2.0](LICENSE). Third-party dependencies
retain their own licenses. Operator data are distributed separately.
