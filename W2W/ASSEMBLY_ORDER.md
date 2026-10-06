# W2W Assembly Orders

`assembly_order.py` is an opt-in process planner, **native Cu-history executor**,
and reference overlay evaluator, available through the normal W2W calculator with
`ASSEMBLY_ORDER_ENABLED: true`. Designs without that flag keep the sequential
path. The executor reuses the existing reduced operators, topology
transactions, two-child bond capture, and thermal solver. No new constitutive
law or bow-difference recurrence is substituted. New assembly paths are not
independently MAPDL-validated.

## Prepared Wafers Without Thinning

`design_6_4hi_p2_no_thinning` is a separate fixed-bow W2W experiment. All four
300-mm wafers start and remain 50 um total, including front BI 1.5 um, BEOL
10 um, Si/TSV 37 um, and back BI 1.5 um. Both BIs already exist. No cap is
constructed, no material is removed, and no BI is deposited during assembly.
The initial and target thicknesses in `stack.yaml` are both 50 um; equal totals
select this path without another process-policy setting.

The illustrative signed bows `[10, -10, 20, -20]` um apply directly at 50 um.
They are not converted from the thick-wafer example. Pitch remains 2 um,
gamma remains 0.99, and translation, rotation, local noise, sites and placement
draws are unchanged. Each bond still has the existing free 350 C anneal/cooldown
and retains both children's Cu histories and material ages. This is not an
85 C workload/reliability simulation or a validated physical handling process.

```bash
SRC=/w/ee.00/puneet/zhichao/Desktop/YAP_STACK/YAP
cd /app/nanocad/projects/personal/zchen/YAP_STACK
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /u1/ee/zhichao/anaconda3/bin/python "$SRC/W2W/calculator_main.py" \
  --config "$SRC/W2W/configs/design_6_4hi_p2_no_thinning/design_6_4hi_p2_no_thinning.yaml" \
  --mode w2w_modeling --ds_name design_6_4hi_p2_no_thinning \
  --ds_dir "$SRC/W2W/input/design_6_4hi_p2_no_thinning" --mechanism-filter overlay
```

All five fixed-layer-order trees are selected with `assembly_orders: all`.
Every tree finishes with `[50, 50, 50, 50]` um wafer totals. The output records
per-interface magnification and overlay yield as well as stack overlay yield.
The original `design_6_4hi_p2` remains the separate 775-to-50 thinning experiment.

### Bow-Jump Diagnostic (2026-10-05)

A first-two-event replay of `(((1,2),3),4)` reproduced the 31.125-mm second
cooldown bow with 50 strict thermal-step acceptances. The second formation
release is only -94.278 um. At 350 C, the first 720-s hold changes bow from
-12.959 to +14.541 mm with creep dissipation and zero plastic dissipation.
Frozen-25-C, full-Gauss history attribution assigns +29.330 mm to wafer 3's
new creep tensor, about 94% of the net second-cycle bow change. The six
wafer/plastic/creep contributions close to within 2e-9 um. This is algebraic
attribution of the current model, not independent physical validation.

The macro solve has linear membrane/bending coupling but no finite-wafer
geometric membrane feedback; `disk` only postprocesses its curvature.
The first cycle already reaches about -35.087 mm while hot. Neither a small
final bow nor an overlay ranking validates those thermal histories. The
per-material birth-age clock for an incoming prepared wafer also requires
independent confirmation; no common-global-clock control was run here.
Production physics and the original five-order results are unchanged.
Detailed trace, attribution and report are under the canonical W2W output
`assembly_orders_no_thinning_20261005/bow_jump_diagnostic/run_20261005_1abc630813`.

## Frozen Library Reuse

For the existing reference p6/P32 library, use the separate
`design_6_4hi_p6_reference` input and matching config. This is the approved
**new five-tree capability experiment**, not a reproduction of the original
reference sequential process or population yields. This p6 preview remains separate
from the p2 and legacy p6 designs.

```bash
SRC=/w/ee.00/puneet/zhichao/Desktop/YAP_STACK/YAP
cd /app/nanocad/projects/personal/zchen/YAP_STACK
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /u1/ee/zhichao/anaconda3/bin/python "$SRC/W2W/calculator_main.py" \
  --config "$SRC/W2W/configs/design_6_4hi_p6_reference/design_6_4hi_p6_reference.yaml" \
  --mode w2w_modeling --ds_name design_6_4hi_p6_reference \
  --ds_dir "$SRC/W2W/input/design_6_4hi_p6_reference" --mechanism-filter overlay
```

This input selects 6-um pitch, 3-um pad diameter, rank-32 interface POD,
the existing `distilled101` 350 C schedule, and the same fixed initial bows
`[10, -10, 20, -20]` um. Initial total thickness is 775 um. The retained
Si/TSV layer is 38.5 um, so the prepared total, including both 1.5-um BIs and
10-um BEOL, is **51.5 um**, not 50 um. All five trees finish with wafer totals
`[51.5, 51.5, 51.5, 775]` um under the agreed pre-merge lower-wafer preparation.
Original reference sequential processing instead bonded and annealed the incoming
thick wafer before thinning/build-up, leaving totals `[775, 51.5, 51.5, 51.5]`.

For p6, `mechanics.frozen_preview: true` explicitly permits only the exact compact
manifest, datasets and basis identified by the saved reference input freeze.
It preserves their `CANDIDATE_NOT_VALIDATED` status; it does not admit arbitrary
candidate operators. The existing reference pure-Si factory reuses native geometry
integrals and condenses a 725-um Cu-free cap into the P32 interface basis.
No MAPDL solve or POD retraining is needed. The existing serial Numba Cu
kernel and compact solver are active only within this preview run and restored
afterward. Per-physical-layer material clocks remain independent.

The first tree adapter still uses the generic strict thermal-advance entrypoint
(`residual_and_tangent`). This does **not** enter the compact solver's
residual-only fast path: the first completed three-cycle route recorded 300
`unsupported_original_calls`, zero phase-cache reuses, and 18 compact RT
transactions. These calls use the original general solver; they are not failed
thermal steps. All 300 steps were strictly accepted, without cutbacks. The
first route took 590.85 s on one worker. The saved original reference sequential
record took 83.81 s for its three-cycle history on eight CPU IDs, plus 16.09 s
load/preparation; process and CPU allocation differ, so this is not an A/B
speed benchmark. The sequential runtime's residual-cached stepping, phase-aware predictor,
phase-local tangent reuse and repeated-subtree reuse are not yet integrated
into the tree thermal loop. Operator reuse alone does not reproduce its speed.

The cap is checked against analytical homogeneous-Si **ABD stiffness**, not
against MAPDL warpage. The saved preview gates require relaxed-normal
scaled error <=1%, full-ABD scaled error <=5%, thermal defect <=1e-8,
and unchanged reaction/direct-condensation checks. A real-library preflight
reproduced **0.171309% relaxed-normal** and **3.304677% full-ABD** errors.
Thus it does not pass the original full-ABD 1% criterion, nor establish a
3.3% warpage error. The strict p2/native loader's 1% gate remains unchanged.
All new tree results retain `physical_readiness: false` until independently
qualified.

## Design Input

The supplied `design_6_4hi_p2` has four wafers and three F2B interfaces, without
an extra TSV base. Its normal `.3dbv`, `.3dbx`, `.3dbf` and compact layout files
define 10 x 10 mm die sites, 2 um pitch, 1 um pad diameter, and 30/50/20 percent
critical/power/dummy pads. The matching config is
`W2W/configs/design_6_4hi_p2/design_6_4hi_p2.yaml`.
Like the p6 config, it includes the full simulation/modeling parameter groups.
Assembly trees currently support modeling only; simulation entrypoints reject
these profiles. Particle, chiplet, dishing, peeling, roughness and ESD fields
are retained for format consistency, not enabled by their presence.

The current p2 input reuses the existing reference **BI2/P24** operator package and
the original `distilled26` 350 C thermal profile. Its prepared wafer total is
50 um: front BI 1.5 um + BEOL 10 um + retained Si/TSV 37 um + back BI 1.5 um.
The removable pure-Si cap is 726.5 um. This is a new assembly-tree engineering
preview, not a frozen reproduction of the original reference W2W p2 experiment.
`mechanics.frozen_preview: true` binds the exact p2 compact manifest, basis,
layer hashes, native geometry source, and existing build receipt. It does not
remove their `CANDIDATE_NOT_VALIDATED` / `BUILT_NOT_VALIDATED` qualifications.
The existing reference preview gates remain unchanged: relaxed-normal ABD <=1%,
full ABD <=5%, thermal defect <=1e-8, and reaction/direct-condensation checks.
A real p2 preflight measured 0.332827% relaxed-normal and 3.323415% full-ABD
defects. The strict rank-64 loader's separate full-ABD 1% gate is not relaxed.

During storage migration, use the current source with the canonical working and
output directory (the interpreter path may depend on your environment):

```bash
SRC=/w/ee.00/puneet/zhichao/Desktop/YAP_STACK/YAP
cd /app/nanocad/projects/personal/zchen/YAP_STACK
/u1/ee/zhichao/anaconda3/bin/python "$SRC/W2W/calculator_main.py" \
  --config "$SRC/W2W/configs/design_6_4hi_p2/design_6_4hi_p2.yaml" \
  --mode w2w_modeling --ds_name design_6_4hi_p2 \
  --ds_dir "$SRC/W2W/input/design_6_4hi_p2" --mechanism-filter overlay
```

Add `--plan-only` to validate without loading operators or solving mechanics.
Each run creates a new output directory with resolved interface configs, input
and source snapshots, plan, per-completed-route mechanics, `MECHANICS.json`,
`RESULT.json` and `SUMMARY.csv`. A failed run keeps its completed routes and a
`FAILED.json`, never a complete ranking. Snapshots and fingerprints are audit
data, not solver restart states.

This is a **new fixed-bow experiment**, not a reference population-yield reproduction:
the approved 775-um wafer bows are `[10, -10, 20, -20]` um. W2W translation,
rotation and local-noise statistics follow the reference setup. The original
reference uses gamma=0.89; the current p2 experiment config uses gamma=0.99
as a stronger correction assumption, not a revised reference baseline.
All routes share 8192 placement draws indexed by
physical interface, and 624 registered die sites. Only overlay yield is scored.

Put one stack-level file at `W2W/input/<design>/stack.yaml`. The supplied
[four-wafer p2 design](input/design_6_4hi_p2/stack.yaml) uses one file instead of
separate assembly-order and native-settings JSON configs. Its sections are:

- `stack`: fixed initial wafer bows, geometry, fabrication source, process, and
  assembly orders.
- `mechanics`: operator package, temporal profile, worker count and optional
  metric policy. Required only for native execution.
- `overlay`: optional pad layout, thresholds and fixed placement draws. When
  present, native execution also scores yield; imported-archive scoring requires it.

Users do not need to supply `schema`, `recipe_id`, or the fixed process-policy
strings. The YAML reader supplies them before validation and hashing, and they
remain recorded in the output context. Explicit older schema/policy fields are
still accepted if supported; invalid values are never silently overwritten.
The native executor currently supports only the existing 25/350 C thermal
recipe with 650/3600/650 s ramp/dwell/cooldown. Those numeric inputs are checked
against the native process card, not a way to select an arbitrary recipe.

Omit unused sections rather than setting them to null. Unknown fields and
duplicate keys are rejected; values must be explicit, without interpolations.
`mechanics.operator_package` accepts a logical `resource:<id>/<path>` reference,
a path relative to the YAML file, or an absolute path. Logical references use
the private resource map described in [README setup](../README.md#first-time-setup).
It references existing operator data, not another user settings file.
Output/archive paths on the command line remain relative to the working
directory. Planning does not open operator data or initialize the native solver.

The normal calculator derives overlay inputs from the design layout and config;
do not add an inline `overlay` section in that mode. The lower-level
`python -m W2W.assembly_order` API below still accepts explicit overlay fields
in its stack YAML. JSON mechanics archives remain simulation
data, not extra configuration files. Legacy JSON commands are still accepted,
but YAML cannot be mixed with external `--overlay-input` or native-settings paths.

## Planning

Run from the YAP root:

```bash
python -m W2W.assembly_order \
  --config W2W/input/design_6_4hi_p2/stack.yaml \
  --plan-only \
  --output /app/nanocad/projects/personal/zchen/YAP_STACK/YAP/W2W/output/assembly_order_plan.json \
  --archive-template /app/nanocad/projects/personal/zchen/YAP_STACK/YAP/W2W/output/assembly_order_mechanics_template.json
```

Outputs must be new paths. The example contains **illustrative**, not measured,
initial bows, and reuses the existing fabrication study's effective elastic
properties with its BEOL-only source. This is not a qualified thick-wafer
material model. Its thermal
durations follow the existing reference 350 C recipe; the tree extension is not
independently validated. Planning exports explicit, portable JSON offline requests.

`initial_bows_um` fixes one signed center-minus-mid-edge bow per wafer in its
initial released state. This first interface supports equal x/y initial bows,
no twist, and identical wafer geometries. `assembly_orders` accepts `"all"`,
one expression such as `"((1,2),(3,4))"`, or a list:

```yaml
assembly_orders: ["(((1,2),3),4)", "((1,2),(3,4))", "(1,(2,(3,4)))"]
```

Change `stack.assembly_orders` in the design YAML; use `all` for all five trees
or a single quoted expression for one route. Planning validates any supplied
overlay section but never scores it.

Nominal in-plane alignment is fixed, so omit `orient` from the `.3dbx` Stack
entries. Older explicit `R0` inputs remain compatible; other nominal rotations
or mirrors are rejected. This is separate from placement rotation error, which
the overlay scorer applies using `SYSTEM_ROTATION_MEAN_rad` and
`SYSTEM_ROTATION_STD_rad` in the calculator config.

Leaves are fixed bottom-to-top wafer IDs, not permutations. Four wafers have
five binary trees. Exhaustive enumeration is limited to eight wafers; selected
trees support up to 32. For each merge:

1. Complete both child histories independently.
2. Thin the lower child's top wafer and prepare its backside BI.
3. Observe both released prebond states.
4. Independently flatten, capture both formation references and Cu histories,
   and bond lower BACK to upper FRONT.
5. Release flattening constraints, anneal freely, and cool down freely.

`--plan-only` only requests these operations; `--native-mechanics` executes them.
Annealing affects only the merged body, not unattached siblings. For example,
`((1,2),(3,4))` retains wafer 2 at 775 um through the 1-2 anneal, then thins
wafer 2 within that substack before the final 2-3 bond.

## Thickness And Initialization

Thicknesses are **total wafer thicknesses**. Initially the example owns front
BI + BEOL + Si = 775 um per wafer. A prepared wafer owns front BI + BEOL +
retained Si + new back BI = 50 um. Retained Si is 37 um for the example's
1.5/10/1.5 um build-up. Separate front/back BI slabs are each counted once;
no third, extra interface slab is inserted at merging. Final wafer totals are
`[50, 50, 50, 775]` um for every four-wafer tree.

`fabrication_layers` is the common initial bottom-to-top (front-to-back) layer
list. The source is zero in BI and Si, as in `manufacturing_bow.py`; only BEOL
has an imposed eigenstrain. The planner inverts the linear-ABD response to reproduce
each fixed initial bow, exporting source amplitudes, source-layer properties
and q6. The native adapter intentionally reuses the existing **stress-free
natural-q6 initialization with virgin Cu**, at the original 775-um midplane.
It does not reconstruct a manufacturing residual stress field from bow alone.
The manufacturing/native curvature sign conversion is explicit and each leaf
must recover its supplied signed bow. Existing birth coordinates, formation
offsets and subsequent Cu histories survive thinning; no bow is re-sampled or
recalibrated after thinning. This retains the old approximation rather than
introducing a new fabrication-stress model.

Waiting effects are explicitly excluded. Child material ages and histories
must be retained, not reset or added together at a merge. The internally supplied
`prebond_policy` is a declared, unvalidated extension of the reference clock policy;
the executor implements it using one age ledger per physical layer. Native
thermal stepping receives `age_at_cycle_entry + (step_time - cycle_start)`.
The offset stays fixed during cutbacks; ages commit only after a successful
cycle. The new backside BI starts at the existing 300-s prebond age, while
retained material keeps its age. No creep/plastic evolution is added during
ideal RT flattening, reference capture, or release.

## Native Execution

From the YAP root, after configuring the read-only native runtime and its private
resource map as described in [README setup](../README.md#first-time-setup):

```bash
python -m W2W.assembly_order \
  --config W2W/input/design_6_4hi_p2/stack.yaml \
  --native-mechanics \
  --output /app/nanocad/projects/personal/zchen/YAP_STACK/YAP/W2W/output/assembly_order_native_mechanics.json
```

The strict native BI2/rank-64 path remains available separately from the p2
reference preview. Package loading checks native basis/content identity. It uses
the existing 37-um Si/TSV operator and builds the removable 726.5-um **pure-Si**
cap by static condensation of existing material-separated reduced FE segments;
it does not stretch a nonlinear TSV operator. The cap has no Cu points.
This is equivalence to existing reduced segments, not new continuum validation.
The selected cap uses mirrored pure-Si replacements totaling
`14 x 37 + 139 x 1.5 = 726.5` um. Repeated segment pairs share their condensation
result. Its provenance retains the rejected coarse partition and a further
refinement comparison, as well as unchanged reaction, ABD and thermal gates.
The loader also retains the original BI6_50/rank-120 path. It admits only these
exact name/rank pairs and does not truncate original datasets to fit. Cap
refinement differences are recorded diagnostics, not a proven convergence bound.

The library is loaded once. Each route gets separate committed states and
per-layer clock proxies. Repeated subtree histories are currently recomputed
across routes, so this first executor does **not** promise seconds-level runtime.
RT transactions reuse the production `bfgs_inverse` solver and elastic-reference
cache. Age-dependent quasi-Newton corrections are not cached across calls;
thermal stepping retains the existing strict/fallback/cutback acceptance policy.
No MAPDL solve is launched. `exact` and existing distilled thermal profiles
are selectable; a profile's earlier validation does not validate the new trees.

Mechanics output includes each event's two released q6/ABD/face-lever records,
thinning/deposition and two-child capture audits, per-layer ages, post-cooldown
q6 and Cu summaries, and state fingerprints. Fingerprints include committed Cu
and birth arrays, reference offsets, gauge constraints, and clocks. They are
**in-memory fingerprints, not saved restart files**.

With no metric policy the output status is
`MECHANICS_COMPLETE_OVERLAY_PENDING`. This does not prevent mechanics execution;
it prevents inventing wafer metric coefficients or silently treating a
square-die metric as a circular-wafer result. To score, YAML `mechanics` can
explicitly provide:

- `metric: {"model": "disk"}` uses reference model's actual-A free-rim circular
  membrane Ritz solve and full-area affine fit. The full 2 x 2 flattening-plus-
  membrane matrix retains shear/twist and A-inverse-B coupling. Degree-3/5
  refinement and equilibrium checks are recorded. This is the normal W2W
  design's policy, not a scalar chi or square-die surrogate.
- `metric: {"model": "explicit", "chi": [chi_x, chi_y], "source": "..."}`;
  a `by_state_id` mapping can override the coefficients for individual bodies.
- `metric: {"model": "square_surrogate"}` opts into the existing reference
  square-membrane coefficient as a **declared unvalidated wafer approximation**.
  This scalar mode requires equal principal curvatures and no twist.

The bow/face-lever metric is the existing reference mapping, not `B1-B2`. For native
execution and scoring together, include `overlay` in the same YAML and
optionally pass `--mechanics-output` to retain the mechanics archive. Both output
paths must be new. Pad layout, thresholds and placement draws remain explicit
inputs described below. Unsupported coupled shear is rejected by directional
bow scoring, while mechanics-only output retains the full q6/ABD information.

## Imported Mechanics

The template contains physical interface and child-state IDs for every event.
It deliberately contains null mechanics fields and cannot be scored. The
offline exporter must provide:

- `status: COMPLETE`, unchanged context/request IDs and event ordering;
- solver identity, implementation/operator SHA-256 identities, and source/clock
  audit references under `provenance`;
- `lower` and `upper` released-body records containing `state_id`, `wafer_ids`,
  `total_thicknesses_um`, `frame_id`, `bow_convention`, directional `bow_um`,
  signed `face_lever_um`, directional `chi`, `face_lever_source`, and `face`;
- `checkpoints`, mapping the four requested pre-preparation, lower-free,
  upper-free and post-cooldown state IDs to SHA-256 digests;
- a `transfer_audit` reference for each event.

Use the frame and bow convention from the plan. Lower/upper faces are BACK and
FRONT. Lengths are um, halfspan is wafer radius, and levers are relative to the
applicable stiffness centroid, **not assumed half-thickness**. Both directional
arrays have length two. A lever may be a scalar or a directional pair.

Checkpoint digests and audit references are exporter assertions. The evaluator
does not open checkpoint files or verify Cu tensors. It rejects incomplete
requested routes, wrong context/lineage/geometry, nonfinite values, and
inconsistent records for shared states. Integrity checks are not physical
validation: every scored result retains `physics_qualification: UNKNOWN`
and `physical_readiness: false`, independently of exporter claims.

The YAML's `overlay` section supplies:

- `coordinates_um`: wafer-centered pad coordinates, shape `(site, pad, 2)`;
- `critical_pad_mask`, `site_mask`: nonempty Boolean selections;
- `threshold_um`, `residual_sigma_um`: nonnegative scalars;
- `magnification_correction_fraction`: scalar in `[0,1]`, applied to the
  paired bow-induced magnification before placement errors;
- `translation_um`: shape `(draw, physical_interface, 2)`;
- `rotation_rad`: shape `(draw, physical_interface)`;
- `conditional_interface_independence: true`.

**Physical interface axes are always 1-2, 2-3, 3-4**, not traversal order. All
routes reuse the exact same placement draws. Use explicit zero arrays if no
placement uncertainty is included. No noise, geometry, or pad layout is
invented. Draws and selected sites receive equal weights.

```bash
python -m W2W.assembly_order \
  --config W2W/input/design_6_4hi_p2/stack.yaml \
  --mechanics-archive completed_mechanics.json \
  --output /app/nanocad/projects/personal/zchen/YAP_STACK/YAP/W2W/output/assembly_order_overlay.json
```

This scoring command requires adding the actual overlay parameters to the YAML;
the shipped example omits them because the normal calculator supplies them.

Scoring reuses reference model's `paired_magnification`, affine overlay and the existing
worst-critical-pad signed-scalar interval model. Stack yield averages the
interface-probability product at the **same registered site and joint placement
draw**, not the product of marginal interface yields. Outputs include each
interface's overlay yield, stack overlay yield and all tied best supplied
orders. This is bow-induced overlay, not total manufacturing yield or a
guarantee of manufacturability. A complete archive of all five routes can also
score an input selecting only a subset, without changing its physics context.

Every interface record also includes the full 2 x 2 bow-induced magnification
matrix in ppm and its corrected residual, `(1 - gamma) M`. Matrix sign is
upper minus lower in the common wafer-centered frame; shear entries are
retained. `max_bow_overlay_nm` is `max ||M x||` over the selected registered
critical-pad coordinates, before translation, rotation, or local noise.
`max_residual_bow_overlay_nm` applies the same maximum after correction.
Thus a dimensionless magnification coefficient and its resulting displacement
are reported separately. `SUMMARY.csv` contains both matrices, both displacement
maxima, interface yield, and stack yield.

`non_bow_overlay_sha256` identifies the translation/rotation draws and local
noise sigma for each physical interface. For a given physical interface it
must match across every assembly order. The same magnification correction
fraction applies to the full affine matrix, including shear. Local noise uses
the same analytical Gaussian sigma rather than a new sample for each route.

## Verification And Remaining Work

```bash
PYTHONPATH=. pytest -q tests/test_assembly_order.py tests/test_assembly_mechanics.py \
  tests/test_assembly_material_clock.py tests/test_assembly_operator_library.py \
  tests/test_w2w_assembly_design.py
```

Small test operators exercise the actual native transactions and Cu kernels,
including both-child histories, nonzero-z thinning, and unequal material ages.
They are not physical wafer predictions or MAPDL validation. Remaining work is
full-size assembly-path qualification, thermal-grid/runtime benchmarking, and
qualification of the wafer bow-to-magnification metric. The natural-q6 source
approximation and ideal-carrier assumption remain explicit limitations.

## Earlier Strict P2 Preflight (2026-10-02)

Before selecting the reference BI2/P24 preview, the real BI2/rank-64 775-um
experiment was **blocked before Cu thermal execution**.
The homogeneous 726.5-um removable Si cap failed the unchanged 1% normalized
analytical ABD gate: coarse maximum 1.124964%, further-refined maximum 1.591390%.
The further-refined A/B/D block maxima are 1.377628%, 1.188443e-8%, 1.591390%.
These are normalized stiffness defects, not warpage errors or yield errors.
The gate concerns the new homogeneous replacement cap, not admission of the
original heterogeneous reference Cu-layer operators. More thin-segment composition
did not establish convergence. No cap tolerance was relaxed, no p6 operator
was substituted, and no assembly-order yield/ranking was produced.

The unchanged-gate diagnostics are saved at
`/app/nanocad/projects/personal/zchen/YAP_STACK/YAP/W2W/output/assembly_orders/design_6_4hi_p2/diagnostic_BI2_ovfo3k08/DIAGNOSTICS.json`.
The preceding real load attempt is recorded in
`preflight_BI2_ck0soc6t/FAILED.json` under the same parent. Software fixture
tests do not resolve this physical-model qualification gap. A pure-Si operator
meeting the unchanged gate and compatible with rank-64 ports is needed before
this strict p2 experiment can advance to five-route yield evaluation. The
separately named reference p6/P32 preview above uses the saved reference gates; it is not
a fix or validation of this p2 cap.

## Completed Frozen-Preview Campaign (2026-10-02)

All five p6/P32 trees completed with 1500 strict thermal-step acceptances,
zero cutbacks, matching shared-state fingerprints and identical solver/operator
identities. Routes 1 and 2 ran sequentially; routes 3 to 5 ran as independent
single-thread workers using the same unmodified mechanics entrypoint. The
serial worker was deliberately stopped after route 2 to avoid duplicate work.
This scheduling change is recorded separately and is not a solver speedup.

| Order | Interface 1-2 | Interface 2-3 | Interface 3-4 | Stack Overlay |
| --- | ---: | ---: | ---: | ---: |
| `(1,(2,(3,4)))` | 100% | 100% | 100% | 100% |
| `(1,((2,3),4))` | 100% | 100% | 0% | 0% |
| `((1,2),(3,4))` | 100% | 0% | 100% | 0% |
| `((1,(2,3)),4)` | 100% | 100% | 100% | 100% |
| `(((1,2),3),4)` | 100% | 0% | 0% | 0% |

These are numerically saturated probabilities for the illustrative fixed bows,
not measured yields or a validated process recommendation. Released intermediate
bow can reach about 11 mm in this new process; the ideal-carrier/natural-q6 and
wafer-mechanics approximations remain unqualified for these paths.

Per-route mechanics took 514.35--630.90 s; summed event times were 2945.69 s.
The all-five yield stage, including input preparation and result serialization,
took approximately 77.69 s by output timestamps. It evaluates 8192 shared
placement draws and 624 registered sites per route. These numbers are not the
paper's online-runtime benchmark. The regression suite passed 131 tests.

The private campaign record is `20261002_163928_5677f40e`. Historical archive
paths are unchanged; the public profile is named `design_6_4hi_p6_reference`.

## Completed P2 Campaign (2026-10-02)

All five **W2W p2/BI2/P24** routes completed with the original `distilled26`
profile, 375 strict thermal-step acceptances and zero failures, fallbacks or
cutbacks. The initial 775-um signed wafer bows remain `[10, -10, 20, -20]` um;
the prepared wafer total is 50 um. This is the approved new fixed-bow
capability experiment, not an original reference W2W yield reproduction.

Translation 3-sigma is 10 nm, rotation 3-sigma is 0.1 urad, local-noise 3-sigma
is 20 nm, and gamma is 0.89. Every route uses the same 8192 placement draws and
624 sites. Per-physical-interface non-bow fingerprints match across all routes,
and all non-bow inputs also match the preceding p6 campaign exactly.

| Order | Interface 1-2 | Interface 2-3 | Interface 3-4 | Stack Overlay |
| --- | ---: | ---: | ---: | ---: |
| `(1,(2,(3,4)))` | 55.36% | 20.82% | 100.00% | 20.82% |
| `(1,((2,3),4))` | 18.56% | 100.00% | 0.00% | 0.00% |
| `((1,2),(3,4))` | 100.00% | 0.00% | 100.00% | 0.00% |
| `((1,(2,3)),4)` | 20.42% | 100.00% | 100.00% | 20.42% |
| `(((1,2),3),4)` | 100.00% | 0.00% | 0.00% | 0.00% |

Displayed percentages are rounded; interface 3-4 in the fourth route is
0.9999999746009599, not exactly one. Stack probabilities use the same-site,
same-draw joint evaluation, not the product of marginal interface yields.

The five independent single-thread mechanics workers finished in 431.35 s
of campaign wall time, including operator load/preparation. Subsequent scoring
took 42.50 s. These are capability-run timings, not the paper's online-runtime
benchmark. Some released prebond states reach 20.089 mm bow, so new assembly
paths and their numerical ranking remain unqualified. Thermal convergence
and shared-input checks do not constitute independent physics validation.

The canonical campaign directory is
`/app/nanocad/projects/personal/zchen/YAP_STACK/YAP/W2W/output/assembly_orders/design_6_4hi_p2/p24_all_orders_lh9ub798`.
`SUMMARY.csv` includes all 15 bond events with full raw/residual magnification
matrices in ppm, raw/residual maximum bow-only displacement in nm, and yields.
`RESULT.json` preserves full precision; `OVERLAY_COMPONENTS.json` and
`VERIFICATION.json` record common-input and unit checks. The focused regression
suite passed 138 tests.

## Gamma Sensitivity (2026-10-02)

The existing p2 mechanics archive was rescored at gamma=0.95 and 0.99 without
repeating operator loading, Cu evolution, or thermal solves. All raw
magnification matrices, geometry, placement draws, translation/rotation
statistics, local-noise sigma and overlay thresholds remain identical.
The original gamma=0.89 campaign files are unchanged.

| Order | Gamma 0.89 | Gamma 0.95 | Gamma 0.99 |
| --- | ---: | ---: | ---: |
| `(1,(2,(3,4)))` | 20.82% | 99.20% | 100.00% |
| `(1,((2,3),4))` | 0.00% | approximately 0% | 3.02% |
| `((1,2),(3,4))` | 0.00% | approximately 0% | 2.53% |
| `((1,(2,3)),4)` | 20.42% | 98.84% | 100.00% |
| `(((1,2),3),4)` | 0.00% | 0.00% | 0.00% |

The two approximately-zero gamma=0.95 probabilities are 2.479169595271327e-146
and 5.47780322297803e-196; full-precision results are retained, not thresholded
to manufacture a different outcome. The p2 config now selects gamma=0.99,
leaving 1% of the bow-induced affine mismatch. The last route still has a
maximum residual bow-only displacement of about 6.95 um versus its 0.198-um
overlay tolerance. Stronger correction does not change the actual warpage.
Gamma=0.99 is a parameter-study assumption, not independently verified
equipment capability, and does not establish physical assembly-path validity.

The canonical sensitivity directory is
`/app/nanocad/projects/personal/zchen/YAP_STACK/YAP/W2W/output/assembly_orders/design_6_4hi_p2/gamma_sensitivity_ww_8zbzq`.
`COMPARISON.csv` contains the three correction assumptions; each candidate's
`SUMMARY.csv` retains all 15 per-interface raw/residual magnification reports.
Final `MECHANICS.json`, `RESULT.json` and `SUMMARY.csv` are under
`all_orders_collected/`; per-route evidence, input snapshots, checked source
snapshots and the independent-route scheduling receipt remain in that campaign.
