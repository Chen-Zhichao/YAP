# YAP-stack

- YAP-stack is a Python-based yield modeling and Monte Carlo simulation tool for
  advanced packaging. This branch focuses on wafer-to-wafer (W2W) and
  die-to-wafer (D2W) hybrid bonding, with stack-aware D2W warpage support.
- A [GUI of YAP](http://nanocad.ee.ucla.edu:8081/yap_gui/) and the
  [user guide video](https://youtu.be/8hiKIQ6C7ng) are available.

Current local benchmark examples used in this workspace are:

- `design_2`: 2.5D four-compute-chiplet D2W benchmark.
- `design_6`: 3D D2W/W2W stack benchmark with substack-level warpage support.

Legacy dense benchmark inputs and generated per-interface configs stay local.
Git includes the compact assembly examples `D2W/design_6_4hi_p2`,
`W2W/design_6_4hi_p2` and `W2W/design_6_4hi_p6_reference` historical inputs,
plus the full p6 config references used by tests. The no-thinning W2W branch is
retired. The public W2W path is F2B-only. It reuses the component catalog
with separately checked pure-Si thinning layers; the older BI2/P24 engineering
preview remains available. Historical mixed-face plans remain non-executable.
Prepared D2W p2 TOTAL30/80 operators are built and structurally audited engineering
candidates, not physically validated models. See
[prepared thickness runs](#prepared-total30-and-total80).
The new component-v2 interface exposes 13 exact single-BI three-cycle bow-passed
geometries to experimental prepared-D2W and F2B W2W assembly. Use the
[complete 4-Hi and 8-Hi generator workflow](#generate-complete-d2w-designs);
assembly accuracy and completed new 8-Hi thermal execution are not established.
Source, documentation, requirements and regression tests are publishable;
experiment outputs, caches, native runtime data and operators are not.
For a colleague using a Git checkout on the same server, follow the
[assembly-order experiment instructions](#assembly-order-experiments).
Importing YAP into another project does not install the precomputed mechanics
operators or geometry data. See [missing mechanics data](#missing-mechanics-data).

# File Structure

```
.
+-- D2W/      # Code for D2W hybrid bonding
|   +-- configs/    # Golden config plus local benchmark configs
|   +-- input/      # Local 3dblox inputs, bump maps, and criticality files
|   +-- calculator_main.py
|   +-- simulator_main.py
|   +-- utils/
+-- W2W/      # Code for W2W hybrid bonding
|   +-- configs/
|   +-- input/
|   +-- calculator_main.py
|   +-- simulator_main.py
|   +-- utils/
+-- LICENSE
+-- README.md
+-- requirements.txt
```

# Installation

The per-user storage code update is pending while the current W2W verification
pins the running Python sources. Do not start a new run with this frozen code
until that update is merged. The setup below describes the required per-user
storage policy; shared mechanics data remain read-only.

Use Linux with Python 3.11 or 3.12 and an approved readable checkout. Set
`YAP_RUN_ROOT` to a private directory of your choosing under /app before
installing dependencies or starting Python. It is mandatory: there is no
fallback to the maintainer's project, source tree, home or /tmp. An existing
run root must belong to your actual UID and have mode 0700. Never change
permissions on another user's directory.

```bash
R=/app/nanocad/projects/personal/zchen/YAP_STACK
# Define YAP_RUN_ROOT yourself, for example beneath your own /app project.
: "${YAP_RUN_ROOT:?Set YAP_RUN_ROOT to your own private absolute /app directory}"
export YAP_RUN_ROOT
RUN_ROOT="$YAP_RUN_ROOT"
umask 077
mkdir -p -m 700 "$RUN_ROOT"
export YAP_WORK_DIR="$RUN_ROOT/work"
export TMPDIR="$YAP_WORK_DIR/temp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export NUMBA_CACHE_DIR="$YAP_WORK_DIR/numba"
export MPLCONFIGDIR="$YAP_WORK_DIR/matplotlib"
export PIP_CACHE_DIR="$YAP_WORK_DIR/pip"
export XDG_CACHE_HOME="$YAP_WORK_DIR/xdg-cache"
export XDG_CONFIG_HOME="$YAP_WORK_DIR/xdg-config"
export XDG_DATA_HOME="$YAP_WORK_DIR/xdg-data"
export XDG_STATE_HOME="$YAP_WORK_DIR/xdg-state"
export XDG_RUNTIME_DIR="$YAP_WORK_DIR/xdg-runtime"
export PYTHONUSERBASE="$YAP_WORK_DIR/python-user"
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX
RESULTS="$RUN_ROOT/results"
mkdir -p "$TMPDIR" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$PIP_CACHE_DIR" \
  "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" \
  "$XDG_RUNTIME_DIR" "$RESULTS"
# Run the following commands from your readable YAP checkout.
```

If any step fails for permissions, stop and request narrowly scoped access.
Do not fall back to /tmp, /data, shared home, source directories or archives.
For an independent checkout, clone into `$RUN_ROOT/YAP` and enter that root.
Code and shared mechanics data are read-only during an assembly run; generated
files stay outside the checkout. An inherited cache override outside your run
root is rejected rather than silently reused.

Use an already provisioned Python environment, or create your own environment
inside your selected private run directory:

```bash
python3 -m venv "$RUN_ROOT/venv"
. "$RUN_ROOT/venv/bin/activate"
python -B -m pip install -r requirements.txt
```

# Usage

For the current colleague-facing reduced-operator workflow, use
[assembly-order experiments](#assembly-order-experiments). Generate a complete
design, inspect it with the normal calculator's `--plan-only`, then run
mechanics only after checking resources and data access.

The separate legacy benchmark workflows use `D2W/calculator_main.py`,
`D2W/simulator_main.py` and their W2W equivalents. Dense layouts and generated
per-interface configs are local data, not installed by Git. Use your own
copies below RUN_ROOT, including writable inputs for utilities that generate
criticality/bump files. Set every legacy config's `OUTPUT_DIR` below
`$RUN_ROOT/results`; a working-directory change alone is not storage setup.

In those legacy workflows, `--mechanism-filter chiplet` evaluates physical-die
fabrication yield. The negative-binomial parameters are
`CHIP_DEFECT_DENSITY_PER_CM2`, `CHIP_CRITICAL_AREA_FRACTION` and
`CHIP_DEFECT_CLUSTERING_ALPHA`; they are separate from bonding-particle
parameters `D0` and `D1`. References in `KNOWN_GOOD_DIE_REFERENCES` remain
physical bodies but contribute a fabrication yield of one. These mechanisms
are not scored in the new assembly-order overlay workflow.

## Design 6 Stack Variants

The historical `generate_stack_variants.py` produces compact center-IO
benchmark layouts. It writes its checkout's input/config trees, so do not run
it against shared or frozen source; use a separate user-owned checkout for
approved maintenance. It is not the prepared-die assembly generator.

| Historical Design | DRAM Tiers | Extra TSV Base | Interfaces | Pitch (um) |
| --- | ---: | ---: | ---: | ---: |
| `design_6_4hi_p6` | 4 | 1 | 4 | 6 |
| `design_6_4hi_p1` | 4 | 1 | 4 | 1 |
| `design_6_8hi_p6` | 8 | 1 | 8 | 6 |
| `design_6_16hi_p6` | 16 | 1 | 16 | 6 |

Thus historical 8-Hi has nine physical bodies. Do not use it for the new
eight-leaf/429-tree study. The new `generate_assembly_design.py --dies 8`
creates exactly eight prepared bodies and seven interfaces, without a base.
Compact regular layouts retain region metadata rather than allocating a full
pad bitmap; they support modeling, not pad-level simulation.

## Assembly-Order Experiments

This workflow compares fixed bottom-to-top assembly trees using reduced
operators and independent physical-layer Cu plastic/creep histories. D2W
execution models identical square prepared dies with both BIs already present,
F2B joins, ideal flattening and free anneal/cooldown. It reports bow-induced
overlay yield, not total manufacturing yield or an explicitly solved receiving
wafer. New component-based assembly predictions remain experimental.

### First-Time Setup

Use the [installation storage setup](#installation) before any Python command.
Run assembly commands from the readable YAP checkout root, not inside D2W or
W2W. All generated inputs, configs, results, temporary files and caches must
remain under your explicit `$YAP_RUN_ROOT`. Do not use /tmp, /data, home storage, the
shared source tree or the read-only distribution for new run output.

Choose your own writable /app location, not a directory beneath the
maintainer's project. Missing permissions require narrowly scoped access,
not an automatic alternate directory or recursive chmod/chown. Colleagues
need read/traverse access to
the checkout, delivery, referenced operator/evidence files and the two pinned
material sources under `$R/operator_repair_20261015/source`. The catalog
references existing canonical data; it is not a self-contained operator copy.
Retain those referenced operator, reference and evidence files while the
delivery is in use, even when their parent directory is named as an experiment.

Before a mechanics launch, check the executing host's actual mount, free
storage, available RAM and active jobs. The normal runner defaults to a
40-GiB available-RAM gate; this is not an 8-Hi memory estimate. Keep one layer
worker and the thread limits below. Do not scan shared mounts, snapshots or
protected migration trees to locate data, and do not restart old dispatchers.

### Component Runtime And Catalog

The v2 interface uses this separately published runtime and metadata. Export
all three selectors below in the same shell BEFORE starting the calculator
or importing any YAP mechanics modules. Do not rely on automatic runtime
selection: a stale legacy override or checkout default can select an old
runtime before the component loader configures itself.

```bash
COMPONENT_DATA="$R/distribution/yap_component_operators_20261021_v1"
export YAP_COMPONENT_OPERATOR_CATALOG_V2="$COMPONENT_DATA/CATALOG.json"
export YAP_NATIVE_RUNTIME_DIR="$COMPONENT_DATA/runtime/native"
export YAP_RUNTIME_RESOURCES="$COMPONENT_DATA/runtime/RESOURCES.json"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=4

python -B component_operators.py list
python -B component_operators.py inspect --entry BI2_SI37
```

Set `NUMBA_NUM_THREADS` before starting Python. BLAS/OMP limits alone do not
bound the compiled Cu kernel's thread pool. The measured component smoke used
four Cu threads and one BLAS thread.

`list` and `inspect` validate bounded catalog/evidence JSON and operator file
metadata without reading numerical NPZ payloads or running mechanics. An
explicit integrity check hashes only the selected entry's numerical operands;
it performs potentially substantial, paced I/O but no thermal solve:

```bash
python -B component_operators.py check --entry BI2_SI37
```

The Python APIs are `load_catalog(catalog_path=None)`,
`resolve_entry(identifier, *, catalog_path=None)`,
`prepared_operator_request(package, geometry, *, frozen_preview=False, layout=None)`
and `load_prepared_component_library(package, geometry, *, frozen_preview=False)`.
F2B W2W uses `w2w_operator_request` and `load_w2w_component_library` with
the wafer geometry ledger described below; it does not pass a prepared-die
geometry to the wafer loader.
Normal calculator dispatch selects the component backend with an exact logical
ID such as `component:BI2_SI37`. Do not set `frozen_preview: true` on this path.
Payload loading verifies the selected reduced operands and their existing
physical registration/projection evidence, including ordered Cu points,
interface traces, virtual work and backside reflection. It binds corrected
material kernels and separate layer clocks; no geometry scaling, legacy
material fallback or whole-stack response fit is used.

The component loader's default configuration does not replace existing runtime
environment overrides. Explicit exports above replace old shell values, but
cannot reset modules already imported into Python. Restart Python when switching
between this runtime and a legacy delivery; do not mix their resource maps,
material implementations or mutable states within an existing process.
`YAP_COMPONENT_OPERATOR_CATALOG_V2` is distinct from the preserved v1
`YAP_COMPONENT_OPERATOR_CATALOG` variable.

### Current Component Coverage

The published v2 catalog has exactly the following 13 entries. BI pitch is
in um; all thicknesses are in um. The unit-history ledger contains one BI:
`single-BI total = Si/TSV + 10 BEOL + 1.5 BI`. Prepared D2W adds the
already-prefabricated second BI:
`prepared total = single-BI total + 1.5 = Si/TSV + 13`.

| Component ID | BI Pitch | Si/TSV | Single-BI Total | Prepared Total | Unit-History Scope |
| --- | ---: | ---: | ---: | ---: | --- |
| `BI2_SI7` | 2 | 7 | 18.5 | 20 | Three-cycle bow pass |
| `BI2_SI17` | 2 | 17 | 28.5 | 30 | Three-cycle bow pass |
| `BI2_SI27` | 2 | 27 | 38.5 | 40 | Three-cycle bow pass |
| `BI2_SI37` | 2 | 37 | 48.5 | 50 | Three-cycle bow pass |
| `BI2_SI47` | 2 | 47 | 58.5 | 60 | Three-cycle bow pass |
| `BI2_SI48p5` | 2 | 48.5 | 60 | 61.5 | Three-cycle bow pass |
| `BI2_SI57` | 2 | 57 | 68.5 | 70 | Three-cycle bow pass |
| `BI2_SI67` | 2 | 67 | 78.5 | 80 | Three-cycle bow pass |
| `BI2_SI77` | 2 | 77 | 88.5 | 90 | Three-cycle bow pass |
| `BI2_SI87` | 2 | 87 | 98.5 | 100 | Three-cycle bow pass |
| `BI2_SI88p5` | 2 | 88.5 | 100 | 101.5 | Three-cycle bow pass |
| `BI3_SI37_COARSE45` | 3 | 37 | 48.5 | 50 | Coarse45 three-cycle bow pass; fine90 time pending |
| `BI6_SI37` | 6 | 37 | 48.5 | 50 | Three-cycle bow pass |

These passes compare single-BI, 10 x 10 mm macro-projected global bow against
native-FE reference histories. The strict per-observation criterion is
`relative error < 10% OR absolute error < 1 um`. The actual reduced histories
solve 61 or 73 states across three cycles; bow values between solved states
are interpolated onto 1081 observations, not additional Cu-history solves.
This is not direct MAPDL qualification, Cu-field accuracy admission or
independent prepared dual-BI/assembly accuracy validation. P3 fine-time
qualification remains pending. Other pitches are not enabled by this fixed
catalog. A separately activated thickness family is described below. Older campaign progress
snapshots are historical, not the current support inventory.

### Intermediate Die Thickness

The additional BI2 endpoint family covers Si/TSV 27..37 um, corresponding to
prepared dual-BI totals 40..50 um. Existing catalog entries take priority;
an intermediate request such as `component:BI2_SI32` automatically uses the
family at `$R/distribution/yap_thickness_family_20261021_v2`. An alternate
location may be selected with `YAP_THICKNESS_FAMILY_DIR`.

The runtime evaluates precomputed geometric coefficient arrays, without a
target FE solve, target POD training, or interpolation of aged Cu states.
For example, generate a normal four-die request with the same command format:

```bash
python -B generate_assembly_design.py \
  --name colleague_4hi_p2_t45 --dies 4 --pitch 2 --total-thickness 45 \
  --initial-bows 10 -10 20 -20 --orders all --output-root "$RUN_ROOT/designs"
```

The independent held-out Si32 test passed all 292 bow observations over 73
solved states and three cycles: maximum relative error 0.303%, maximum absolute
error 1.382 um. Its actual unit total is **43.5 um with one BI**, not the
prepared dual-BI total45. This is matched-clock spatial validation, not a
1081-state temporal check, every-thickness validation, or prepared-assembly
accuracy qualification. The original-JVP replay took 1300.9 s; the matched
native reference took 2048.0 s. These are not seconds-level runtimes.

The real YAP physical-library check for prepared45 passed, including ordered
GPs, face traces and corrected material binding. Cold construction took
64.8 s including compilation; this is not a thermal-history runtime. The
normal two-die calculator smoke is separate from physical accuracy admission.
Other thickness brackets are not enabled by this family. F2B W2W reuses its
Cu components with additional homogeneous-Si cap construction, described below.
See [Thickness Operator API](THICKNESS_OPERATORS_API.md) for the standalone
interface, validation gate and bounded optional caches.

### Generate Complete D2W Designs

`generate_assembly_design.py` writes a complete normal design: stack.yaml,
N identical square die definitions/instances, N technology files, N-1 adjacent
connections, compact layout, and the full two-section calculator config
copied structurally from the existing p2 template. Gamma is preserved at 0.86.
There is no extra TSV base. The historical `design_6_8hi_p6` has eight DRAM
tiers plus its base and is not this eight-body experiment.

For four prepared 50-um dies and all five trees:

```bash
GENERATED="$RUN_ROOT/designs"
python -B generate_assembly_design.py \
  --name colleague_4hi_p2_all --dies 4 --pitch 2 --total-thickness 50 \
  --initial-bows 10 -10 20 -20 --orders all --output-root "$GENERATED"
```

For selected eight-body balanced and sequential trees, generate separate names:

```bash
python -B generate_assembly_design.py \
  --name colleague_8hi_p2_balanced --dies 8 --pitch 2 --total-thickness 50 \
  --initial-bows 10 -10 20 -20 5 -5 15 -15 \
  --orders balanced --output-root "$GENERATED"

python -B generate_assembly_design.py \
  --name colleague_8hi_p2_sequential --dies 8 --pitch 2 --total-thickness 50 \
  --initial-bows 10 -10 20 -20 5 -5 15 -15 \
  --orders sequential --output-root "$GENERATED"
```

The component default is `component:BI{pitch}_SI{prepared_total-13}`, with
decimal points written as `p`. Thus prepared TOTAL50 uses `BI2_SI37`, not
the single-BI TOTAL60 operator `BI2_SI48p5`. P3 requires its exact coarse ID:

```bash
python -B generate_assembly_design.py \
  --name colleague_4hi_p3_coarse --dies 4 --pitch 3 --total-thickness 50 \
  --initial-bows 10 -10 20 -20 --orders balanced \
  --operator-package component:BI3_SI37_COARSE45 --output-root "$GENERATED"
```

Generation defaults to one sequential tree. It accepts balanced, all, or a
quoted fixed-order binary tree. Labels are physical bottom-to-top body IDs,
not permutations. Supply exactly N signed bows. Selected trees support
2..32 bodies; exhaustive enumeration is limited to 2..8. Counts above eight
are input capability, not a demonstrated resource or accuracy envelope.
Explicit `--input-root` and `--config-root` may replace `--output-root`;
both must be absolute directories below your YAP_RUN_ROOT.

Existing input or config design directories are refused, never overwritten.
Use a new name for a new experiment. Generated component designs select
`component_graded25`: 25 actual states per thermal cycle, sampled from the
original 361-state schedule at indices 0, 20..160 by 20,
161, 162, 164, 168, 170, 180, 190, 200, and 220..360 by 20.
This is a runtime clock choice, not new assembly accuracy qualification.
The generated fabrication E/nu values are explicitly inherited TOTAL50
calibration assumptions, not independently qualified target-geometry properties.

### Serial Or Parallel Orders

D2W and F2B W2W share the same optional setting in design-local `stack.yaml`:

```yaml
mechanics:
  # Keep the existing operator_package, temporal_profile and metric settings.
  order_workers: 4
  layer_workers: 1
```

`order_workers: 1` (the default when omitted) runs orders serially. A larger
positive integer requests that many isolated worker processes, capped by the
number of selected orders. For generated D2W designs, add `--order-workers 4`
to `generate_assembly_design.py`; the calculator commands are unchanged.
For example, BI2, prepared TOTAL40, eight dies and all 429 orders:

```bash
python -B generate_assembly_design.py \
  --name colleague_8hi_p2_t40_all --dies 8 --pitch 2 --total-thickness 40 \
  --initial-bows 10 -10 20 -20 5 -5 15 -15 \
  --orders all --order-workers 4 --output-root "$GENERATED"
```

Replace the example bows with the actual eight signed inputs. Each worker
loads its own operator library once and evolves independent route histories;
operators are not shared across process memory. To avoid nested parallelism,
parallel-order workers use one BLAS/OpenMP/Numba thread and require
`layer_workers: 1`. The available CPU allocation and aggregate RAM gate must
permit the requested concurrency; it is not silently reduced to fit RAM.
The normal calculator applies `MIN_AVAILABLE_MEMORY_GIB` per effective worker
at admission (default 40 GiB each), not as a measured 8-Hi memory estimate.
No 8-Hi throughput or linear speedup is promised.

Worker work/cache/temp/log files stay below the canonical run directory.
The parent alone scores overlay and writes ordered results, preserving the
same physical-interface placement samples. A failed worker aborts the run,
terminates remaining workers and leaves diagnostics but no complete ranking.
There is no automatic retry. Parallel execution does not broaden the geometry
or accuracy qualification of the selected operator package.

### Plan Then Execute

Use the complete config's `d2w_modeling` section, not the reserved
`d2w_simulation` section. After generation and component runtime setup:

```bash
NAME=colleague_4hi_p2_all
python -B D2W/calculator_main.py \
  --config "$GENERATED/D2W/configs/$NAME/$NAME.yaml" \
  --mode d2w_modeling --ds_name "$NAME" \
  --ds_dir "$GENERATED/D2W/input/$NAME" --mechanism-filter overlay \
  --plan-only --output-dir "$RESULTS/D2W"
```

Planning validates normal inputs, metadata and the declared recipe; it does
not load numerical operators, solve mechanics or produce yields. After the
host/storage/RAM/job checks, run the actual selected experiment by omitting
`--plan-only`:

```bash
python -B D2W/calculator_main.py \
  --config "$GENERATED/D2W/configs/$NAME/$NAME.yaml" \
  --mode d2w_modeling --ds_name "$NAME" \
  --ds_dir "$GENERATED/D2W/input/$NAME" --mechanism-filter overlay \
  --output-dir "$RESULTS/D2W"
```

For a selected 8-Hi experiment, set
`NAME=colleague_8hi_p2_balanced` or `NAME=colleague_8hi_p2_sequential`
and run the same two commands. These are real mechanics entrypoints, not
comparison-only mocks. No completed new 8-Hi thermal run or measured 8-Hi
runtime is established; unit-history passes and regression tests do not
establish that result.

Verification snapshot, October 20 public-runtime checks:

- The v4 software regression completed with **384 passed and one known xfail**.
  The expected failure is the existing strict high-strain constitutive
  comparison; this interface does not claim pointwise Cu-history equivalence.
  The bound-file mode audit passed for 88 files. These are software/access
  checks, not physical accuracy qualification.
- A fresh, frozen public-runtime **two-die** normal-calculator run completed
  C1 with full Cu history on the 25-state cycle. Measured process wall time
  was **379.174 s including loading**, not an 8-Hi timing. The predicted
  stack overlay yield was 99.9948373856%, with
  raw m_xx about -0.972039235 ppm. Evidence is RESULT.json and SUMMARY.csv
  under `$R/component_delivery_20261021/real_pair_v2/runs/component2_pair/20261020_222628_a6e80d2f`.
  The result retains `physical_readiness: false` and unvalidated assembly scope.
- All 13 entries passed real physical-library construction, including exact
  ordered Cu GP/weight comparison against their raw references, interface
  traces, reflection and material bindings. The finite inventory receipt is
  `$R/component_delivery_20261021/real_inventory_v2/REPORT.json`.
  These loading checks are separate from the published single-BI bow evidence
  and do not qualify complete assembly histories.
- All six normal full-config planning checks passed, including 429 eight-body
  trees. Independent external-CWD generation/planning and standalone imports
  also passed with explicit new-runtime exports and canonical writable paths.
- Nine tiny staged eight-body executor cases covered every root split plus
  balanced/sequential trees. Synthetic overlay checks across all 429 trees
  preserved noise identity per physical interface. These fixtures used staged
  annealing and no real Cu history; they do not qualify an eight-die Cu run.

### All 429 Eight-Body Trees

Generate a separate eight-body all-orders request, then plan it through the
normal calculator:

```bash
python -B generate_assembly_design.py \
  --name colleague_8hi_p2_all --dies 8 --pitch 2 --total-thickness 50 \
  --initial-bows 10 -10 20 -20 5 -5 15 -15 \
  --orders all --output-root "$GENERATED"

NAME=colleague_8hi_p2_all
python -B D2W/calculator_main.py \
  --config "$GENERATED/D2W/configs/$NAME/$NAME.yaml" \
  --mode d2w_modeling --ds_name "$NAME" \
  --ds_dir "$GENERATED/D2W/input/$NAME" --mechanism-filter overlay \
  --plan-only --output-dir "$RESULTS/D2W"
```

Eight leaves have 429 fixed-order trees, seven bond/anneal events each:
3,003 anneals in total. Each selected eight-body tree has only seven events.
A balanced tree gives each die three cycles; a sequential tree exposes some
dies to seven, beyond the single-BI three-cycle evidence. The full sweep
contains 12,952 die-cycle exposures. Operators are loaded once per run, but
each route's physical history is recomputed; there is no cross-route history
cache or restart checkpoint. Measure selected-route RAM and runtime before
a separately reviewed all-orders execution. No completion or wall-time claim
is made for the 429-tree sweep.

### Legacy TOTAL50 Runtime

The existing TOTAL50 example remains a separate engineering-preview path
using its old runtime, P24 operators and `distilled26` clock. In a fresh shell
with the same canonical storage setup, select the old matching delivery:

```bash
LEGACY_DATA="$R/distribution/yap_prepared_p2_20261006_v1/YAP"
export YAP_NATIVE_RUNTIME_DIR="$LEGACY_DATA/runtime/native"
export YAP_RUNTIME_RESOURCES="$LEGACY_DATA/runtime/RESOURCES.json"
unset YAP_COMPONENT_OPERATOR_CATALOG_V2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python -B D2W/calculator_main.py \
  --config D2W/configs/design_6_4hi_p2/design_6_4hi_p2.yaml \
  --mode d2w_modeling --ds_name design_6_4hi_p2 \
  --ds_dir D2W/input/design_6_4hi_p2 --mechanism-filter overlay \
  --plan-only --output-dir "$RESULTS/D2W"
```

After resource checks, omit `--plan-only` for actual legacy execution.
Its unchanged stack selects:

```yaml
operator_package: resource:compact_p2/packages/BI2_50.json
temporal_profile: distilled26
frozen_preview: true
```

For a newly generated legacy 8-Hi input, use `--legacy-total50` with pitch 2,
total 50 and a fresh name. Do not combine that flag with a component package.
Legacy reproduction and corrected-component predictions have different
material/runtime bindings and must not be presented as interchangeable.

### Prepared TOTAL30 And TOTAL80

The older `design_6_4hi_p2_t30` and `design_6_4hi_p2_t80` profiles remain
structurally audited native64 engineering candidates under the old delivery.
They use `resource:prepared_p2_native/packages/BI2_30.json` and
`BI2_80.json`, respectively, not component v2. Their physical accuracy is
not established by structural audits. New component TOTAL30/80 designs select
`component:BI2_SI17` and `component:BI2_SI67` instead. Preserve the old
catalog, receipts and deliveries; do not relabel or overwrite them.

### F2B-Only W2W

The component interface accepts the same IDs and thickness resolution as D2W:
BI2 prepared target totals 20, 30, 40, 50, 60, 61.5, 70, 80, 90, 100 and
101.5 um; BI3 target50 (coarse45-state qualification) and BI6 target50; and the BI2 continuous target40--50
family. Unit-component validation scope is unchanged. W2W still starts at
775 um, removes pure-Si backside material and prepares a new BI; it does not
use the prepared/no-thinning D2W process.
The coverage table's prepared-total column is the W2W target total after
backside preparation, including both BIs and BEOL, not the original wafer
thickness or the Si/TSV sublayer thickness.
The immutable unit catalog retains `W2W_thinning: false` for its original
validation scope. Runtime W2W support is supplied by the separate component
adapter and checked Si caps, not by rewriting that historical evidence flag.

Use the normal generator and full W2W calculator config. With the installation
and component-runtime environment above, for BI2 target40 and all five trees:

```bash
GENERATED="$YAP_RUN_ROOT/designs"
NAME=colleague_w2w_4hi_p2_t40_all
python -B generate_assembly_design.py \
  --bonding-mode w2w --name "$NAME" --wafers 4 --pitch 2 \
  --total-thickness 40 --initial-bows 10 -10 20 -20 \
  --orders all --order-workers 2 --output-root "$GENERATED"

python -B W2W/calculator_main.py \
  --config "$GENERATED/W2W/configs/$NAME/$NAME.yaml" \
  --mode w2w_modeling --ds_name "$NAME" \
  --ds_dir "$GENERATED/W2W/input/$NAME" --mechanism-filter overlay \
  --output-dir "$RESULTS/W2W"
```

`--total-thickness 45` selects the same Si32 parameterization as D2W. Use a
new design name for a different experiment. All bows here are examples for
the original 775-um wafers; replace them with the actual signed inputs.
`--dies` and `--wafers` are aliases for the number of physical bodies.

The component loader preserves the native64 physical face trace and corrected
Cu binding, then adds zero-Cu homogeneous H8/Bbar Si operators. The 1.5-um
outer slice exists from birth and is not redeposited. Full periodic interior
relaxation precedes static condensation; binary assembly makes the removable
bulk. Full FE reaction, thermal-work, energy and nullspace checks are mandatory.
Analytic continuum ABD and the split-thin-slice comparison are reported
separately, not mislabeled as independent W2W accuracy validation.

Small checked seed matrices are read from
`$R/distribution/yap_w2w_si_caps_20261021_v1` when available, or the optional
`YAP_W2W_SI_CAP_LIBRARY_DIR`. Missing seeds are constructed at first use and
cached below `YAP_WORK_DIR/w2w_si_caps`, or `YAP_W2W_SI_CAP_CACHE_DIR`.
Parallel workers share the small locked Si cache, never Cu states. A new
fractional pure-Si remainder uses an exact linear FE construction before
condensation; its first-use cost is not an inexpensive interpolated Schur
matrix or new Cu-ROM training. No old private geometry directory is required.

The normal BI2 target40 two-wafer calculator completed its bond, anneal/cool,
final outer thinning and overlay output in 519.84 s on mfd, using one physical
core, one Cu/BLAS thread and previously constructed Si seeds. This is a complete
integration smoke, not a W2W accuracy comparison or a four-wafer timing estimate.
Its evidence is below
`$R/user_runs/w2w_component_20261021/complete_pair40_v1`.

The public W2W planner uses only lower-BACK to upper-FRONT bonds, with fixed
front-below-back orientation. Selected ordered trees support 2 through 32
wafers; exhaustive enumeration supports up to eight. Every wafer is born at
775 um total, and the lower child's exposed top backside is prepared before
every merge, whether that child is virgin or a substack. The default final
per-wafer total is 50 um. Optional `stack.prethin_incoming: true` thins incoming
children before bonding without translating or reinitializing their histories.

The explicit BI2/P24 preview models Si37 plus original pure-Si cap slices of
1.5 and 725 um from birth. Mating-back preparation removes both slices and
adds BI1.5. Final outer-back finishing removes only 725 um, retaining the
original 1.5-um Si state with no new BI. This is the legacy pure-Si-cap
assumption, not new TSV-Si38.5 accuracy evidence. Temporary thinning support
is removed before free observations and contributes no carrier thermal cycle.
Independent computational datums are not an initially assembled 775-um stack;
joining surfaces must coincide, and aged children are never translated.
The original 1.5-um factory slice is not independently admitted: its standalone
relaxed-normal ABD diagnostic is about 2.55%, failing a 1% criterion. That
failure stays in the operator provenance; the ordered full birth cap and
725-um bulk must still pass the unchanged historical bulk-preview gates.

```bash
python -B -m W2W.assembly_order \
  --config W2W/input/f2b_4hi50_p24_preview/stack.yaml \
  --plan-only --output "$RESULTS/f2b_plan.json"
```

The standalone `--native-mechanics` mode produces a mechanics archive only
unless an explicit overlay input is supplied. For the complete pad-layout,
fixed-seed overlay comparison and all-five ranking, use the normal calculator
after host/storage/job preflight, with all runtime/cache paths canonical:

```bash
python -B W2W/calculator_main.py \
  --config W2W/input/f2b_4hi50_p24_preview/calculator.yaml \
  --mode w2w_modeling --ds_name f2b_4hi50_p24_preview \
  --ds_dir W2W/input/f2b_4hi50_p24_preview \
  --mechanism-filter overlay --output-dir "$RESULTS/W2W"
```

This writes `RESULT.json` and `SUMMARY.csv` only after all requested orders
complete, using the existing geometry/overlay scoring implementation. A
per-route mechanics checkpoint alone is not an all-five yield result.
The native64 component path above is experimental: component bow passes
do not establish assembly accuracy. Low or zero overlay yield is valid,
not a reason to reject execution or relax accuracy gates. Historical mixed-face
v2 ledgers remain plan-only under their separate historical contract; old
archives cannot be reused as new F2B results. The retired no-thinning workflow
is not restored.

### Changing Initial Bow

The supplied scalar bows are signed center-minus-mid-edge values, with equal
x/y curvature and zero twist. Planning recomputes complete natural q6 from
the thickness-specific fabrication ledger; this is a stress-free natural-shape
approximation, not reconstructed Cu manufacturing history. Each selected
route starts fresh and evolves its own retained material histories.
Changing bows changes request identities; do not rescale old results or
transplant cached states. The interface does not establish a validated
bow-magnitude range. With the same seed/layout, non-bow placement draws stay
fixed per physical interface across routes.

### Missing Mechanics Data

A Git checkout does not install private operators. Set the exact matching
runtime, resource map and catalog above before mechanics imports. A metadata
listing or plan is not an operator-loading proof. Missing mounts or denied
permissions require maintainer-provided access, not a new environment variable,
recursive permission changes or a search over shared storage. The v2 catalog
still references canonical operands/evidence and pinned material sources.
For older self-contained legacy delivery options, see
[PORTABLE_ASSEMBLY.md](PORTABLE_ASSEMBLY.md); its exporter is not a component-v2
portability claim.

### Results And Limitations

Each invocation creates a fresh timestamp/ID directory under
`$RESULTS/<domain>/<design>/`. `PLAN.json` and input/source snapshots are
written for planning. Actual execution additionally writes per-route
mechanics receipts, `MECHANICS.json`, `SUMMARY.csv` and `RESULT.json`.
The summary contains interface/stack overlay yields and raw/residual
magnification. Non-bow draws are shared by physical interface; stack yield
uses joint same-draw evaluation, not a product of marginal yields.

All new assembly results retain `physical_readiness: false` and the component
unit-history scope. Adding a backside BI and passing kinematic/virtual-work
checks do not establish dual-BI assembly accuracy, W2W thinning accuracy,
MAPDL comparison or total manufacturing yield. State fingerprints and route
JSONs are evidence, not restart files; a failed run does not establish a
complete ranking. Software test passes are not scientific qualification.

### Run Files And Ownership

The shared operator catalog, native runtime, material sources and published
Si matrices are read-only dependencies. They stay with the maintainer; your
run must not write there. Every generated input, result, log and writable
cache stays under your chosen `YAP_RUN_ROOT`:

```text
$YAP_RUN_ROOT/
  designs/{D2W,W2W}/{input,configs}/<name>/
  results/{D2W,W2W}/<name>/<timestamp_id>/
    SUMMARY.csv                 # Per-interface yield and magnification
    RESULT.json                 # Per-order comparison and best supplied order
    MECHANICS.json              # Mechanics history archive
    route_<n>_mechanics.json    # Completed-order records
    INPUTS.json, PLAN.json, RUNTIME_SOURCES.json
    stack.resolved.yaml, <interface>.yaml
    input_snapshot/, source_snapshot/, progress.log
    order_workers/              # Parallel mode: manifests, logs, route/terminal JSON
  work/                         # Temporary files and runtime caches
```

Generated inputs include stack.yaml, compact layout, 3Dblox definitions,
instances, technology files and the full calculator YAML. They are generated
once per new named design, not necessarily on every calculator invocation.
Plan-only runs do not produce yield results; unsuccessful actual runs may
leave `FAILED.json` and partial records without a final ranking. Cache
directories do not imply that every cache type writes files on every run.
No MAPDL `.rst`, `.page` or restart files are generated by this reduced
assembly-order calculator. The standalone thickness API's optional disk cache
and legacy non-assembly utilities are outside this storage-contract change.

# File Formats

**1. Bump Map (.bmap):**

Format: `<instance> <bump_type> <x> <y> <port> <net>`

Example: `Bump_0 uBUMP 115 1610 txdatasb txdatasb`

**2. Criticality (.txt):**

Current format: `<net> <group_size> <tolerated_esd_failures> <tolerated_mechanical_failures>`

Where:

- `group_size`: total number of pads or bumps in the redundancy group.
- `tolerated_esd_failures`: number of ESD failures the group can tolerate before
  failing.
- `tolerated_mechanical_failures`: number of mechanical failures the group can
  tolerate before failing.

Filename variants:

- `*_criticality.txt`: default profile.
- `*_criticality_esd_strict.txt`: strict ESD profile.

Criticality values are calculated when reading the file:

- `esd_criticality = (group_size - tolerated_esd_failures) / group_size`
- `mechanical_criticality = (group_size - tolerated_mechanical_failures) / group_size`

Legacy format is deprecated but still supported:
`<net> <esd_criticality> <mechanical_criticality>`

**3. 3dblox files:**

- `.3dbv`: chiplet definitions, design areas, regions, and thicknesses.
- `.3dbx`: stack configuration containing chiplet instances and connections.
- `.3dbf`: chiplet bump metadata such as pitch and bump size.

# Output

Assembly-order output is described in [results and limitations](#results-and-limitations)
and is explicitly redirected to `$RUN_ROOT/results`. For legacy non-assembly
commands, use user-owned full config copies with every `OUTPUT_DIR` set below
`$RUN_ROOT`; do not rely on old relative `D2W/output` or `W2W/output` defaults.
The following names describe legacy simulation outputs, not assembly-tree receipts.

**1. `assembly_yield_summary__<config_stem>__<criticality_profile>.txt`**

Simulation settings, runtime information, stack assembly yield, and
per-interface yield.

**2. `assembly_yield_per_interface__<config_stem>__<criticality_profile>.txt`**

The simulated assembly yield of each interface.

**3. `assembly_fail_vec_per_interface_dict__<config_stem>__<criticality_profile>.npz`**

Failure vectors for each die sample and failure mechanism. This file is written
in verbose simulation mode.

**4. `assembly_fail_map_per_interface_dict__<config_stem>__<criticality_profile>.npz`**

Average per-pad failure counts. This file requires both `--verbose` and
`--save-failure-maps`.

**5. `simulation_failure_map_<mechanism>__<config_stem>__<criticality_profile>.png`**

Per-interface simulation failure heatmaps for `overlay`, `particle`,
`mechanical`, `ESD`, and `overall`. These PNGs require `--save-failure-maps`.

# Generator Utilities

Useful D2W helper scripts include:

- `generate_assembly_design.py`: generate complete prepared-D2W assembly inputs
  and full calculator configs under a new, canonical user-owned destination.
- `component_operators.py`: inspect the published v2 catalog or explicitly
  verify selected numerical operands, without running mechanics.
- `D2W/utils/generate_criticality.py`: generate default and strict-ESD
  criticality files from bump maps.
- `D2W/utils/assign_bump_names.py`: assign critical, redundant, power/ground,
  and dummy net names to raw bump maps.
- `D2W/utils/assign_design2_neighbor_nets.py`: generate design_2-style
  inter-chip shared link names and shared-net summaries.
- `D2W/utils/generate_pad_ratio_layouts.py`: derive new pad-ratio layout folders
  from an existing ratio tree.
- `D2W/utils/bmap_grid_sync.py`: synchronize bmap names using normalized grid
  coordinates.
- `D2W/utils/plot_bump_kinds.py`: visualize bump categories from a bmap.
- `D2W/utils/plot_design_topology.py`: plot chiplet/interface topology from
  3dblox inputs and shared-net files.

# Notes

- Git includes the declared compact assembly examples and full parameter
  configs. Legacy dense layouts, outputs and caches stay local. Ignored native
  runtime and operator data are still required to run assembly-order mechanics;
  configure the shared runtime as described above.
- Generated user-run results are separate from permanent scientific archives.
  Rerunning an experiment does not authorize cleaning, moving or rewriting
  protected data, migration controls or old scientific receipts.

# Shared Repaired Component Catalog

The colleague interface is now
[component catalog v2](#component-runtime-and-catalog), with the 13 exact
geometries listed in [current coverage](#current-component-coverage).
The original October 19 v1 catalog, its `mechanics_runtime.py list/check`
APIs, trust pins, receipts and data remain unchanged. V1 remains a two-entry,
single-BI global-bow interface that refuses assembly. Use
`component_operators.py`, not the v1 CLI, for new component D2W experiments.

Maintainers can inspect the proposed v2 publication from bounded JSON evidence
without reading numerical payloads or launching a solver:

```bash
python -B publish_component_operators.py
```

Progress is only a locator. The publisher independently binds each closed
assessment to the selected virgin Cu history, exact reduced construction,
physical reference and projection/registration proofs. It records native-FE
three-cycle bow scope, actual solved-state count and explicit pending assembly
accuracy; it does not grant direct MAPDL or W2W qualification.

Only a maintainer should publish metadata, to a fresh distribution path:

```bash
python -B publish_component_operators.py \
  --output "$R/distribution/NEW_REVIEWED_DELIVERY/CATALOG.json"
```

Replace `NEW_REVIEWED_DELIVERY` with a separately reviewed fresh destination.
Do not rerun publication against the existing October 21 catalog. Publication
uses exclusive creation and writes read-only metadata; it neither overwrites
old deliveries nor copies/installs native runtime, Cu card or numerical
operands. A new metadata path alone is not a runnable delivery. Keep published
catalogs and all scientific evidence immutable; never edit historical
construction states to simulate new admission.

# Paper Link

To be continued.
