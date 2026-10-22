# YAP-stack

D2W and W2W assembly-order overlay-yield modeling. Use Linux and Python 3.11 or 3.12.
Run commands from the root of your own YAP checkout.

`D2W/` and `W2W/` contain yield models. `tools/` groups construction, diagnostics,
publication and packaging; its metadata helpers are also used by the runtime.
Root bow/mechanics modules retain paths pinned by existing native backends.
Normal assembly-order runs do not need to invoke maintenance tools.

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
DATA="$R/distribution/operators/components"
export YAP_COMPONENT_OPERATOR_CATALOG_V2="$DATA/CATALOG.json"
export YAP_NATIVE_RUNTIME_DIR="$DATA/runtime/native"
export YAP_RUNTIME_RESOURCES="$DATA/runtime/RESOURCES.json"
export YAP_W2W_SI_CAP_LIBRARY_DIR="$R/distribution/operators/silicon_caps"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=4
python -B component_operators.py list
```

Shared data are grouped under `distribution/operators/`: `components/`,
`artifacts/` (operands, geometry, pinned material sources and validation evidence),
`thickness_family/`, `silicon_caps/`, and `legacy/` for older supported inputs.
The dated operator experiment folders are not required for normal assembly runs.

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

## Appendix: Technical and Maintenance Notes

This appendix is for users inspecting results and coding agents reading or
modifying the project. Sections 1-5 above remain the normal running procedure.
These notes describe the current assembly-order path, not every legacy
calculator, simulator or research script in the repository.

### A. Code Map

| Location | Responsibility |
| --- | --- |
| [generate_assembly_design.py](generate_assembly_design.py) | Generate mutually consistent stack, layout, 3Dblox and full calculator configurations. Does not run mechanics. |
| [component_operators.py](component_operators.py) | Resolve component IDs, check metadata, load operands and construct component-backed assembly libraries. |
| [mechanics_runtime.py](mechanics_runtime.py) | Enforce private output storage, select native runtime/resources and resolve checked shared-data paths. |
| [D2W/assembly_order.py](D2W/assembly_order.py) | Prepared-die input validation, assembly planning and adaptation to the shared engine. |
| [W2W/assembly_order.py](W2W/assembly_order.py) | Tree parsing, wafer process ledgers, archive checks and shared overlay scoring. |
| [W2W/assembly_design.py](W2W/assembly_design.py) | Design-level execution, input/source snapshots, progress and final output. D2W also uses this runner. |
| [W2W/assembly_mechanics.py](W2W/assembly_mechanics.py) | Transfer child states, solve released/joined equilibrium and advance Cu thermal histories. |
| [W2W/assembly_parallel.py](W2W/assembly_parallel.py) | Isolated order workers, resource checks, source pins and result collection. |
| [W2W/component_assembly_bridge.py](W2W/component_assembly_bridge.py) | Register component coordinates, ports and material bindings with the assembly backend. |
| [W2W/component_silicon_cap.py](W2W/component_silicon_cap.py) | Pure-Si additions and checked cap caches for the thick-wafer W2W model. |
| [thickness_operators.py](thickness_operators.py) | Load the admitted thickness family and evaluate parameterized Si/TSV operands. |
| Root `history_*`, `initial_bow_*`, `manufacturing_bow.py` | Shared bow, fabrication-source, uncertainty and legacy-backend interfaces. Similar prefixes do not imply duplicate versions. |
| [tools/](tools/) | Operator construction, diagnostics, publication, packaging and research utilities. `tools.publish_component_operators.MetadataReader` is also a runtime dependency. |
| [tests/](tests/) | Software contracts and synthetic fixtures. Passing these is not independent physical validation. |

The main execution chain is:

```text
full calculator YAML + input/<design>/stack.yaml + layout/3Dblox
  -> domain calculator_main.py
  -> validated assembly plan and fixed overlay inputs
  -> selected component library and independent Cu histories
  -> released mating-face response at each bonding event
  -> per-interface and whole-stack overlay yield
```

### B. Geometry and Operator Selection

The generated `stack.yaml` contains `mechanics.operator_package`; this selects
the mechanics model. The calculator does not search all NPZ files or silently
choose the closest geometry. For example:

```yaml
# Fragment of the existing mechanics mapping, not a replacement stack.yaml.
mechanics:
  operator_package: component:BI2_SI27
  temporal_profile: component_graded25
  layer_workers: 1
  order_workers: 2
  metric: {model: square_affine}  # D2W; W2W uses {model: disk}.
```

For the current prepared geometry, total thickness includes both 1.5-um BIs,
10-um BEOL and retained Si/TSV. Thus total 40 um means Si/TSV thickness
`40 - 1.5 - 10 - 1.5 = 27 um`, hence `BI2_SI27`. The `BI2` part denotes
bonding pitch, not BI thickness. A 45-um total uses `SI32`; noninteger Si
thicknesses use `p` as the decimal marker, as in `SI48p5`. The exact ID and
interpolation eligibility are resolved by the loader.

The component catalog binds BI, BEOL and Si/TSV operands to their geometry,
material, construction and validation evidence. File identity and hash checks
protect those bindings. A catalog is an index, not the numerical operator data.

```bash
# Metadata inspection; this does not run a thermal history.
python -B component_operators.py inspect --entry BI2_SI27

# Stronger operand check; can read/hash large files or evaluate a thickness
# family and therefore can take substantially longer than inspect.
python -B component_operators.py check --entry BI2_SI27
```

Use the coverage list in Section 2 and inspect the selected entry. The generator
can accept geometry values for which execution is not available; generated
files or a successful plan are not proof that all operands exist or are admitted.
Thickness parameterization is available only inside its supported family,
currently the BI2 prepared-total 40-50 um interval. This is not arbitrary
nearest-neighbor substitution, pitch interpolation, or an independently tested
point for every possible thickness. Do not extrapolate it beyond its bracket.

Changing initial bow or assembly order reuses the geometry operators but reruns
mechanical histories. Changing pitch, materials, TSV geometry or layer structure
can require different operators. Regenerate designs after geometry changes;
editing only the operator ID does not update the other input files.

### C. Assembly Trees and Process Assumptions

Body IDs are in fixed bottom-to-top order. A tree changes the grouping and
bonding history, not the final layer permutation. Four bodies have five trees;
eight have 429. Explicit trees can contain 2-32 bodies, but enumerating `all`
is limited to eight. Enumeration and comparison are not an optimization search.

The generator accepts `sequential` and `balanced` as convenience keywords and
converts them to trees. In `stack.yaml`, use `all`, one quoted tree, or a list:

```yaml
stack:
  assembly_orders: ["(((1,2),3),4)", "((1,2),(3,4))"]
```

For `(1,(2,(3,4)))`, the progress line
`(1,(2,(3,4))) A1: 3.BACK--4.FRONT` means the first bonding event joins the
back of body 3 to the front of body 4. Next, body 2 joins substack 34; finally,
body 1 joins substack 234. `A1` is an event label, not a physical interface ID
or an error. `FRONT` is the device/BEOL side and `BACK` the substrate backside.

- D2W keeps each die at its prepared total thickness, with both BIs already
  present. No assembly-time thinning or BI growth is performed. Nonsequential
  trees can require offline D2D substack preparation; generated D2W designs
  enable this explicitly. The receiving object is a free die-site column,
  not an explicitly meshed whole receiving wafer or carrier.
- The current W2W path is F2B-only. Wafers start at 775-um total thickness;
  the lower exposed backside is prepared before its F2B bond. Each join is
  followed by free anneal and cooldown. The final exposed top backside is
  thinned without adding another bonding BI. Read `PLAN.json` for the exact
  event-by-event thickness/material ledger; do not substitute older mixed-face
  or no-thinning process assumptions.
- Both released prebond bodies are observed. Carrier handling uses the existing
  ideal flattening model, not a finite carrier FE model. Free bow must not be
  replaced by zero merely because a carrier is mentioned. Anneal is unrestrained.
- `initial_bows_um` are fixed signed center-minus-mid-edge bows, not peak-to-valley
  warpage or unsigned magnitudes. W2W values refer to the original thick wafers.
  The same realization is used for every tree; it is not redrawn per order.
  Bow calibration through a fabrication source is not a separately validated
  reconstruction of the complete fabrication residual-stress history.

### D. What Is Actually Calculated

Reduced operators are not a table mapping an initial bow directly to a final
yield. They supply reduced mechanical responses for equilibrium and Cu material
updates. Each body retains its own Cu plastic/creep state and material clock
through joining and subsequent thermal cycles. Do not reset these states when
joining substacks or reuse a mutable child state across independent routes.

The normal assembly path derives a full 2-by-2 mating-face affine strain
difference in a common frame, using the released mechanics and the configured
square/disk flattening metric. It is not simply scalar `B_upper - B_lower`.
With correction fraction `gamma`, the bow-induced matrix is scaled by
`1 - gamma`; translation and small-angle rotation are then added at the
registered site/pad coordinates. Gamma acts on the full matrix, including its
off-diagonal terms, not just one diagonal magnification value.

The current scorer uses the worst-critical-pad survival and registered-site
compounding surrogate. It is not a full net-level redundancy analysis or an
independent product over every real bump. Stack yield is compounded at the
joint-draw/site level before averaging; multiplying already averaged interface
yields is not generally equivalent.

For a fair order comparison, placement samples are indexed by physical
interface, not traversal event. The same seed, translation/rotation draws,
threshold, local-noise settings and critical geometry are shared across trees.
`non_bow_overlay_sha256` allows checking this in the output. Total overlay yield
can still change when fixed non-bow errors combine with different mechanical
responses. Fixed initial bows also do not imply binary yield: placement
variation and analytically integrated local noise can remain enabled.

The compact layout describes a regular pad population and a registered critical
region. It avoids expanding a huge bump map, but does not automatically represent
an arbitrary real design. Off-center, irregular or net-specific critical regions
need an explicit compatible representation and validation. Do not claim real-bmap
accuracy merely because the compact example runs.

### E. Parallelism and Runtime

`mechanics.order_workers` is the number of concurrent assembly-order processes,
not the number of CPU threads used by one solve. `1` runs serially; larger
values are capped by the number of requested orders and checked against CPU
affinity. The normal design runner requires `layer_workers: 1`.

Each parallel worker loads its own library and owns independent Cu states.
Worker BLAS/OpenMP/Numba thread settings are restricted to one; setting
`NUMBA_NUM_THREADS` in the parent shell does not override that worker policy.
The default startup memory gate is 40 GiB per effective worker: two workers
require 80 GiB available and five require 200 GiB. This is a conservative
resource gate, not measured resident memory. More workers can increase memory
pressure and shared-filesystem reads instead of giving proportional speedup.
The parallel worker timeout is currently 24 hours for its whole assigned shard,
including loading, not 24 hours per bonding event.

Serial execution loads the library once but creates independent histories for
the requested routes. Do not assume common subtrees are cached across orders.
A four-body all-order run has five routes and three bonding events per route.
Eight-body enumeration therefore costs far more than one eight-body history.

Cold shared-file reads, operand hash checks and compiled-kernel initialization
can take minutes. Nonlinear equilibrium, material updates and thermal cutbacks
add further cost. Pure-Si cap cache behavior also affects W2W initialization.
No universal seconds-level runtime is promised. When reporting timings, record
geometry, domain, order count, worker/thread settings, temporal profile and
whether loading/initialization is included. Do not apply a private accelerated
prototype's timing to a different public execution path.

### F. Output, Progress and Interruption

All writable paths must remain under the user's own absolute `/app` run root.
`YAP_RUN_ROOT` must be owned by the running UID with mode `0700`; pre-existing
directories created with a different mode are not fixed by `mkdir -p`.
Do not use this checkout, shared operator data, another user's run directory,
shared home, `/tmp`, `/data`, or symlink aliases as output storage.

`YAP_WORK_DIR` holds temporary data and caches. `XDG_*`, `NUMBA_CACHE_DIR` and
`MPLCONFIGDIR` are dependency cache/config locations, not scientific model
parameters. YAP assigns private defaults; stale inherited overrides can fail
storage validation. Prefer the setup above over inventing alternate fallback
directories. Re-export the shared-data variables in each new shell.

Useful run artifacts include:

| File | Interpretation |
| --- | --- |
| `INPUTS.json`, `PLAN.json`, `stack.resolved.yaml` | Input provenance, planned trees/process and resolved settings. Plans alone contain no solved yields. |
| `input_snapshot/`, `source_snapshot/`, `RUNTIME_SOURCES.json` | Input/source copies and hashes for reproducibility. |
| `route_N_mechanics.json` | One completed route's mechanics. Its existence does not mean every requested route completed. |
| `MECHANICS.json` | Combined mechanics archive for the run. |
| `SUMMARY.csv` | Interface yields, stack yields, raw/corrected affine magnification in ppm and bow-induced displacement maxima in nm. |
| `RESULT.json` | Final comparison and best among the supplied orders; inspect its scope and qualification fields. |
| `FAILED.json` | Recorded exceptions when handled by the runner. A manual interrupt need not create this file. |

`max_bow_overlay_nm` and `max_residual_bow_overlay_nm` describe bow-induced
in-plane displacement, not out-of-plane warpage height or total overlay including
placement and local noise. Yield values are fractions in `[0,1]`, not percentages.

The terminal prints the run directory, loading/execution milestones, bonding
event labels and completed orders. Serial progress is also in `progress.log`.
Detailed parallel-worker messages are in `order_workers/worker_*/worker.log`.
There is no guaranteed per-temperature-step heartbeat; a quiet terminal alone
does not prove a hung process. Inspect process activity and the appropriate log
before launching a duplicate job. Do not edit code/config while its run is active.

`Ctrl+C` requests termination; an executing compiled kernel may delay handling.
The parallel controller has worker cleanup logic. Wait for termination and
confirm workers have exited before deleting that run's directory. Cu checkpoints
are in-memory state fingerprints, not restart files: interrupted histories cannot
be resumed from them. Partial routes are not a complete ranking. Never delete
the whole shared `user_runs` tree to clean one trial.

Normal component runs consume shared operator data rather than producing MAPDL
`.rst`/`.page` files. Generated inputs, snapshots and numerical caches still use
disk space. Measure the complete private run root when budgeting space; a small
`SUMMARY.csv` is not the entire footprint. Preserve input/provenance and accepted
results before deleting disposable data, and never clean shared artifacts as
though they were a user's temporary outputs.

### G. Dependencies and Validation Boundaries

On the shared server, colleagues use their own checkout/run root and read the
operator data configured in Section 1. Git and Python requirements alone do not
provide those data. Relevant branches of the shared distribution are:

- `components/`: catalog, material card and selected native runtime/resources.
- `artifacts/`: numerical operands, physical geometry, pinned material sources
  and evidence, with an exact location manifest.
- `thickness_family/`: parameterized thickness data and activation evidence.
- `silicon_caps/`: published W2W pure-Si cap data.
- `legacy/`: dependencies for explicitly selected older resource packages.

Copying only a few NPZ files is not a complete deployment. Another server needs
the full selected dependency closure and a checked path/publication setup;
mass-editing absolute paths or scientific receipts is not a supported installer.

Distinguish five separate claims: an operator exists; its structure is valid;
its component bow/history meets its bound validation evidence; an assembly route
executes; and the complete assembly response is independently validated. None
automatically implies the next. Read the entry's evidence and result qualification
fields. `physical_readiness: false` must not be relabeled simply because the
program completed. Component validation and synthetic software tests do not
certify every assembly, footprint, thermal recipe or interpolated thickness.
