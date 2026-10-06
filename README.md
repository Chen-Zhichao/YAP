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
`W2W/design_6_4hi_p2`, `W2W/design_6_4hi_p2_no_thinning`, and
`W2W/design_6_4hi_p6_reference`, plus the full p6 config references used by tests.
Source, documentation, requirements and regression tests are publishable;
experiment outputs, caches, native runtime data and operators are not.
For a colleague using a Git checkout on the same server, follow the
[assembly-order experiment instructions](#assembly-order-experiments).

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

1. Clone the repository and enter the repo root.

```
git clone <repo-url>
cd YAP
```

2. Optional: create and activate a virtual environment.

```
conda create -n yap_env python=3.12
conda activate yap_env
```

3. Install dependencies.

```
pip install -r requirements.txt
```

# Usage

- Generate criticality files from bump maps.

  ```
  python D2W/utils/generate_criticality.py --input-root D2W/input --designs 2 --profiles both --force
  ```

  Supported profiles:

  - `default`: replicated nets tolerate `R-1` ESD failures and `R-1`
    mechanical failures.
  - `esd_strict`: replicated signal nets tolerate `0` ESD failures and `R-1`
    mechanical failures.
  - `both`: generate both files.

- Run D2W yield simulation.

  Always enter `D2W` before running D2W simulation commands. This keeps
  relative outputs under `D2W/output/`.

  ```
  cd D2W && python simulator_main.py --config configs/design_2/design_2.yaml --mode d2w_simulation --ds_name design_2 --ds_dir input/design_2 --criticality-profile default --verbose
  ```

- Run D2W yield calculator.

  Always enter `D2W` before running D2W calculator commands. The calculator
  reads the default `*_criticality.txt` files from the design input folder.

  ```
  source /u1/ee/zhichao/anaconda3/etc/profile.d/conda.sh && conda activate general && cd D2W && python calculator_main.py --config configs/design_2/design_2.yaml --mode d2w_modeling --ds_name design_2 --ds_dir input/design_2/c30_r0_pg50_dm20
  ```

- Run W2W yield simulation.

  Always enter `W2W` before running W2W commands. This keeps relative outputs
  under `W2W/output/`.

  ```
  cd W2W && python simulator_main.py --config configs/design_6/design_6.yaml --mode w2w_simulation --ds_name design_6 --ds_dir input/design_6 --verbose
  ```

Use `--mechanism-filter chiplet` to evaluate only physical-die fabrication
yield. The default `all` includes chiplet yield together with the enabled
bonding mechanisms. Configure the negative-binomial model with
`CHIP_DEFECT_DENSITY_PER_CM2`, `CHIP_CRITICAL_AREA_FRACTION`, and
`CHIP_DEFECT_CLUSTERING_ALPHA`; these are separate from bonding-particle
parameters `D0` and `D1`. References listed in
`KNOWN_GOOD_DIE_REFERENCES` remain in the physical stack but contribute a
fabrication yield of one.

## Design 6 Stack Variants

Run `python generate_stack_variants.py` from the repository root to generate
compact center-IO D2W and W2W inputs for:

- `design_6_4hi_p6`: four DRAM tiers, four interfaces, 6 um pitch
- `design_6_4hi_p1`: four DRAM tiers, four interfaces, 1 um pitch
- `design_6_8hi_p6`: eight DRAM tiers, eight interfaces, 6 um pitch
- `design_6_16hi_p6`: sixteen DRAM tiers, sixteen interfaces, 6 um pitch

The TSV interconnect die is the stack base and is not included in the Hi count.
Every die is 10 x 10 mm. Pads are assigned from the die center outward:
30% critical signal, then 50% power delivery, then 20% dummy. These variants
use `LAYOUT_KIND: regular_center_io`, so the calculator keeps only regular-array
and region metadata rather than materializing per-pad bitmaps.

At 6 um pitch each interface has 2,775,556 total pads, including 832,667
critical pads. At 1 um pitch it has 100,000,000 total pads, including
30,000,000 critical pads. Compact variants support modeling only; simulation
requires an explicit pad layout.

For example, run the D2W 8-Hi case with:

```bash
cd D2W
python calculator_main.py \
  --config configs/design_6_8hi_p6/design_6_8hi_p6.yaml \
  --mode d2w_modeling \
  --ds_name design_6_8hi_p6 \
  --ds_dir input/design_6_8hi_p6 \
  --mechanism-filter overlay,particle,mechanical
```

## Assembly-Order Experiments

This optional workflow compares supplied assembly trees at a fixed bottom-to-top
layer order, using reduced operators to retain both children's Cu plastic/creep
histories through F2B bonding and free annealing. It reports overlay yield for
each interface and the whole stack. Existing designs without
`ASSEMBLY_ORDER_ENABLED: true` continue to use the normal YAP workflow above.

The examples have exactly four physical bodies and three interfaces, without
the extra TSV base of legacy design 6:

| Domain | Design | Starting geometry | Assembly preparation | Gamma |
| --- | --- | --- | --- | ---: |
| D2W | `design_6_4hi_p2` | 10 x 10 mm dies, 50 um total | No thinning or BI deposition | 0.86 |
| W2W | `design_6_4hi_p2` | 300-mm wafers, 775 um total | Prebond thinning and back BI deposition; prepared total 50 um | 0.99 |
| W2W | `design_6_4hi_p2_no_thinning` | 300-mm wafers, 50 um total | No thinning or BI deposition | 0.99 |

Each prepared 50-um body contains front BI 1.5 + BEOL 10 + Si/TSV 37 + back BI
1.5 um. D2W and no-thinning W2W already have both BIs at birth. The thermal
recipe is 25 C to 350 C, with 650-s ramp, 3600-s hold and 650-s cooldown.
The examples use fixed illustrative signed bows `[10, -10, 20, -20]` um rather
than sampling an initial-warpage distribution.

### First-Time Setup

Run the commands in this section from the **Git checkout root**, not from inside
`D2W/` or `W2W/`. Use Linux and Python 3.11 or 3.12, and install the dependencies
in your own environment. Configure an approved writable work disk **before**
installing dependencies or starting Python; neither setup nor execution should
write to `/tmp`. Replace the example work path for your account/host:

```bash
export YAP_WORK_DIR=/data/puneet/$USER/yap-work
mkdir -p "$YAP_WORK_DIR/temp"
export TMPDIR="$YAP_WORK_DIR/temp" TMP="$YAP_WORK_DIR/temp" TEMP="$YAP_WORK_DIR/temp"
export NUMBA_CACHE_DIR="$YAP_WORK_DIR/numba" MPLCONFIGDIR="$YAP_WORK_DIR/matplotlib"
export PIP_CACHE_DIR="$YAP_WORK_DIR/pip" PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX
python -m pip install -r requirements.txt
```

Git includes source, compact inputs and complete parameter YAMLs, but not native
mechanics or operator data. On the shared server, reuse the existing read-only
p2 runtime, including its bow-to-magnification implementation:

```bash
export YAP_NATIVE_RUNTIME_DIR=/app/nanocad/projects/personal/zchen/YAP_STACK/distribution/yap_assembly_p2_20261004_v2/YAP/runtime/native
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

YAP validates temporary/cache overrides and output destinations, including
symlink targets, and rejects `/tmp` paths. Python temporary files and mechanics
subprocesses use the configured directory. Without overrides, small caches
use the Git-ignored `.cache/yap/` in the checkout. Large mechanics builds require
an explicit host-local `YAP_WORK_DIR`, not shared-home `/w` or `/u1`. Check the
executing host's mount and available storage before running; accepted results
belong in your shared project, separate from temporary files. Never use the
native operator directory as a work directory.

The checked-in designs use logical operator references. Keep these references
and the other mechanics settings unchanged:

```yaml
operator_package: resource:compact_p2/packages/BI2_50.json
frozen_preview: true
```

Obtain the private resource map matching this runtime from the maintainer.
Keep it outside Git and select it explicitly:

```bash
export YAP_RUNTIME_RESOURCES=/path/to/private/resources.json
```

Alternatively, place the map at `.runtime_resources.json` in your checkout.
`runtime_resources.example.json` shows the format, not real data locations;
its directory values must be replaced with the runtime's actual relative
directories. The p2 runs require `compact_p2` and `geometry` bindings. The p6
profile additionally requires `compact_p6` and `w2w_corrected`.
No directory discovery or fallback is performed. Logical references resolve
only when operators are loaded, so `--plan-only` needs no private resource map.
Older private configurations and archives require migration to the current
field and metric names; unsupported names are rejected, not reinterpreted.
Renamed recipe metadata changes context/cache IDs. Existing raw archives keep
their original identities and must not be relabeled as new runs.
Read permission is required for the runtime and all parent directories. Use
the current Git code and inputs, not the October 4 delivery's older copies.
The current memory gate requires at least 40 GiB available RAM.

Choose your own writable shared-project output directory. Replace
`YOUR_PROJECT` below; do not use the author's output directory or shared home:

```bash
RESULTS=/app/nanocad/projects/personal/YOUR_PROJECT/yap_results
```

For a separate server without access to this shared runtime, see the
[standalone delivery instructions](PORTABLE_ASSEMBLY.md). The existing exporter
does not yet include the newer no-thinning profile. The Git-included
`W2W/design_6_4hi_p6_reference` comparison profile requires its own p6 data; it cannot
use the shared p2 delivery as a substitute.

### Select Assembly Orders

Edit the existing `stack` section of the selected design's `stack.yaml`:

```yaml
stack:
  # Signed center-minus-mid-edge bow in um at the starting thickness.
  initial_bows_um: [10.0, -10.0, 20.0, -20.0]
  assembly_orders: all
  # Alternatively, one quoted tree or a list of quoted trees:
  # assembly_orders: "(((1,2),3),4)"
  # assembly_orders: ["(((1,2),3),4)", "((1,2),(3,4))"]
```

`all` selects all five trees for four bodies. Labels 1 through 4 are fixed
bottom-to-top; every body must appear exactly once, without changing layer
order. For D2W, keep `stack.allow_offline_substacks: true` to permit balanced
and right-nested trees, which include offline D2D substack preparation rather
than only individual D2W placements. Enumeration supports up to eight bodies;
larger designs require selected trees and separately supported geometry data.

The complete calculator config remains in
`<D2W-or-W2W>/configs/<design>/<design>.yaml`. Use its modeling section with
`ASSEMBLY_ORDER_ENABLED: true` and `ANNEAL_MODE: sequential`; its simulation
section is reserved and rejected for assembly trees. Set gamma through
`MAGNIFICATION_CORRECTION` in that modeling section. Nominal in-plane orientation
is fixed; placement rotation remains an overlay error from the config, not an
assembly-tree orientation variable. No extra JSON or user-maintained schema
and process-policy strings are needed.

### Run All Assembly Orders

D2W, fixed 50-um dies:

```bash
python D2W/calculator_main.py \
  --config D2W/configs/design_6_4hi_p2/design_6_4hi_p2.yaml \
  --mode d2w_modeling --ds_name design_6_4hi_p2 \
  --ds_dir D2W/input/design_6_4hi_p2 --mechanism-filter overlay \
  --output-dir "$RESULTS/D2W"
```

W2W, the original 775-um starting wafers with prebond thinning:

```bash
python W2W/calculator_main.py \
  --config W2W/configs/design_6_4hi_p2/design_6_4hi_p2.yaml \
  --mode w2w_modeling --ds_name design_6_4hi_p2 \
  --ds_dir W2W/input/design_6_4hi_p2 --mechanism-filter overlay \
  --output-dir "$RESULTS/W2W"
```

W2W, the newer experiment starting with 50-um wafers and no thinning:

```bash
python W2W/calculator_main.py \
  --config W2W/configs/design_6_4hi_p2_no_thinning/design_6_4hi_p2_no_thinning.yaml \
  --mode w2w_modeling --ds_name design_6_4hi_p2_no_thinning \
  --ds_dir W2W/input/design_6_4hi_p2_no_thinning --mechanism-filter overlay \
  --output-dir "$RESULTS/W2W"
```

Append `--plan-only` to any command to check inputs and enumerate the requested
trees without thermal solves or yield evaluation. This check does not validate
operator loading. Full runs load the selected checked operators once and solve
each tree's Cu history; no new MAPDL operator build is needed for these profiles.

### Results And Limitations

Each run writes a separate directory under
`$RESULTS/<domain>/<design>/<timestamp-and-id>/`:

- `PLAN.json`: supplied trees and process events.
- `MECHANICS.json`: released-body observations, process audits and state
  fingerprints. Cu arrays remain in memory; this is not a restart file.
- `SUMMARY.csv`: per-interface raw/residual magnification in ppm, bow-induced
  overlay error in nm, and interface/stack overlay yield.
- `RESULT.json`: full-precision yields and the best of the supplied orders.

Non-bow translation/rotation draws and local-noise statistics remain the same
for each physical interface across all trees. Stack overlay yield is a joint
same-site, same-draw evaluation, not the product of marginal interface yields.
The compact central critical region is a layout surrogate, not exact real-bmap
all-pad survival. Particle, chiplet, dishing, peeling and ESD parameters remain
in the full config but are not scored in assembly-order mode.

These are **unvalidated engineering studies**, not benchmark reproductions or
qualified process recommendations. The natural-q6 initial-source approximation,
free annealing, ideal flattening and per-material clocks remain model
assumptions. The prepared-wafer 31-mm bow result and creep-clock interpretation
are unresolved. D2W models a free die-site column, not a finite receiving wafer
or carrier; its `square_affine` metric retains normal/shear coupling and
twist but remains unvalidated. W2W uses the `disk` metric. New assembly
paths retain `physical_readiness: false`, and missing data fail rather than
producing a fabricated ranking. Implementation details and experiment evidence
remain in [W2W/ASSEMBLY_ORDER.md](W2W/ASSEMBLY_ORDER.md).

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

When commands are run from inside `D2W`, D2W outputs are written under
`D2W/output/<design_name>/`. W2W commands follow the same convention under
`W2W/output/<design_name>/`.

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
- Runtime files in `D2W/output/`, `W2W/output/`, and generated cache files can
  be regenerated by rerunning the modeling or simulation commands.
- D2W and W2W runs print the terminal duck separator when an experiment
  finishes.

# Paper Link

To be continued.
