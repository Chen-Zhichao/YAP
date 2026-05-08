# YAP-stack Technical Documentation

This document describes the current `yap-stack` repository state. The active
D2W example in this branch is `design_6`. The D2W flow has been updated with the
newer YAP+ style setup for arbitrary 3dblox interfaces, pad-level risk maps,
criticality profiles, and per-interface simulation summaries. HBM and
paper-specific D2W inputs/configs from `YAP_IO_Assign` are intentionally not part
of this repo.

## Table of Contents

1. [Repository Scope](#1-repository-scope)
2. [Entry Points](#2-entry-points)
3. [Configuration Resolution](#3-configuration-resolution)
4. [Input Files and Criticality](#4-input-files-and-criticality)
5. [Pad Bitmap Collections](#5-pad-bitmap-collections)
6. [D2W Pad Risk Map Flow](#6-d2w-pad-risk-map-flow)
7. [D2W Monte Carlo Simulation Flow](#7-d2w-monte-carlo-simulation-flow)
8. [Failure Mechanisms](#8-failure-mechanisms)
9. [Output Files and Runtime Caches](#9-output-files-and-runtime-caches)
10. [W2W Notes](#10-w2w-notes)
11. [Maintainer Notes](#11-maintainer-notes)

## 1. Repository Scope

YAP-stack is a Python platform for hybrid-bonding yield modeling and simulation.
It contains both D2W and W2W code paths:

- `D2W/`: current D2W simulation/modeling flow for 3dblox interfaces.
- `W2W/`: wafer-to-wafer flow. Some W2W modules are older and should be updated
  separately from D2W.
- `D2W/configs/design_6/design_6.yaml`: active D2W skeleton config.
- `D2W/input/design_6/`: active D2W 3dblox, bump-map, and criticality inputs.
- `README.md`: command-oriented quick start.
- `yap-doc.md`: this technical overview.

D2W and W2W share concepts, but their implementation details are not identical.
Avoid copying D2W-only cleanup into W2W unless the W2W call sites have also been
checked.

## 2. Entry Points

Run commands from the repository root.

### D2W

- `D2W/pad_risk_map_calculator.py`
  - Current D2W modeling entry point.
  - Produces per-pad ESD, overlay, particle, mechanical, and overall risk maps.
  - `D2W/calculator_main.py` is a compatibility wrapper for this path.

- `D2W/simulator_main.py`
  - Current D2W Monte Carlo assembly simulation entry point.
  - Produces stack assembly yield, per-interface yield, optional failure vectors,
    and optional failure heatmaps.
    
### W2W

- `W2W/calculator_main.py`
  - W2W modeling/risk-map style entry point.

- `W2W/simulator_main.py`
  - W2W Monte Carlo simulation entry point.

## 3. Configuration Resolution

YAP uses `OmegaConf` YAML sections as skeleton configs. For the active design,
the important sections are:

- `d2w_simulation`
- `d2w_modeling`
- `w2w_simulation`
- `w2w_modeling`

D2W loads `D2W/configs/design_6/design_6.yaml`, selects the requested section,
then resolves interface-specific values through `D2W/utils/util.py`.

### D2W 3dblox Mode

When the design input directory contains:

- `generated_chiplet_definitions.3dbv`
- `generated_stack_config.3dbx`

the D2W entry points call `get_config_dict()`. It walks each connection in the
stack config and creates one resolved config per bonding interface.

For each interface, the loader resolves:

- `INTERFACE_TOP`, `INTERFACE_BOT`, `INTERFACE`
- `DIE_W_um`, `DIE_L_um`
- `PITCH_r_um`, `PITCH_c_um`
- `PAD_TOP_R_um`, `PAD_BOT_R_um`
- `PAD_ARR_ROW`, `PAD_ARR_COL`
- `PAD_ARR_L_um`, `PAD_ARR_W_um`
- `ITF_TOP_THICK_um`, `ITF_BOT_THICK_um`
- derived terms such as `SYSTEM_MAGNIFICATION_*`, `eff_DIE_R`, `S_INIT_A_M`,
  and `S_INIT_B_M`

The resolved configs are written back into the design config folder with the
interface name and output suffix, for example:

```text
D2W/configs/design_6/<interface>__design_6__default.yaml
```

### Legacy Single-Interface Mode

If the 3dblox wrapper files are absent, D2W can still run a single-interface
flow through `get_single_interface_config_dict()`. In that mode the YAML must
define `INTERFACE`, and the corresponding `<INTERFACE>.bmap` file must exist in
the input directory.

### No Hidden D2W Runtime Defaults

D2W no longer uses `apply_d2w_runtime_defaults()` or `_set_default_if_missing()`.
Required model knobs should be explicit in `design_6.yaml`. In particular, the
active D2W config now explicitly contains:

- `WARPAGE_LIMIT_UM`
- `D1`
- `EDGE_REGION_WIDTH_um`
- `CU_RECESS_PAD_FAIL_RATIO`
- `CU_RECESS_DIE_LEVEL_THRESHOLD_PADS`
- `V_MIN_V`, `V_MAX_V`
- `WEIBULL_K`, `WEIBULL_LAMBDA`
- `CUTOFF_MIN_A`
- `ESD_PAD_MAP_SUB_FACTOR`
- `verbose`
- `ITF_TOP_THICK_um`, `ITF_BOT_THICK_um`

This makes the YAML the source of truth. If a paper experiment needs different
values, edit the config rather than adding fallback defaults in code.

## 4. Input Files and Criticality

### Bump Map

The current `.bmap` format is:

```text
<instance> <bump_type> <x> <y> <port> <net>
```

Coordinates are in microns. `convert_3dblox_to_pad_bitmap()` sorts bump entries
from top-left to bottom-right into a run-specific temp copy before generating
pad bitmaps.

### 3dblox Files

- `.3dbv`: chiplet definitions, design areas, and chiplet thickness.
- `.3dbx`: stack connectivity.
- `.3dbf`: per-chiplet bump type and pitch metadata.

The loader supports `Chiplet_Grid.pitch` or explicit
`Chiplet_Grid.pitch_r`/`Chiplet_Grid.pitch_c`.

### Criticality Files

The current generated D2W criticality format is:

```text
<net> <group_size> <tolerated_esd_failures> <tolerated_mechanical_failures>
```

Use `D2W/utils/generate_criticality.py` to generate these files. Supported
profiles are:

- `default`: replicated nets tolerate `group_size - 1` ESD failures and
  `group_size - 1` mechanical failures.
- `esd_strict`: redundant signal nets tolerate `0` ESD failures and
  `group_size - 1` mechanical failures.

The loader resolves profile filenames through `resolve_criticality_path()`:

- `*_criticality.txt`
- `*_criticality_esd_strict.txt`

## 5. Pad Bitmap Collections

`convert_3dblox_to_pad_bitmap()` converts each interface `.bmap` plus its
criticality file into a bitmap collection saved as:

```text
output/<design>/<interface>/<interface>_bitmap_collection.npy
```

Important fields include:

- `CRITICAL_PAD_BITMAP`
- `REDUNDANT_PAD_BITMAP`
- `DUMMY_PAD_BITMAP`
- `ESD_CRITICAL_PAD_BITMAP`
- `pad_coords`
- `mapping_physical_to_bumpid`
- `redundant_net_to_bumpids`
- `redundant_net_to_1d_physical_mask`
- `criticality_info`
- `redundant_group_id_per_pad`
- `redundant_tolerated_esd_failures`
- `redundant_tolerated_mechanical_failures`

`ESD_CRITICAL_PAD_BITMAP` is separate from mechanical/overlay criticality. For
redundant nets, ESD criticality depends on the selected criticality profile.

The D2W flow also collapses identical interfaces when possible, so repeated
interfaces can reuse bitmap, risk-map, and simulation artifacts instead of
recomputing them.

## 6. D2W Pad Risk Map Flow

`D2W/pad_risk_map_calculator.py` performs the analytical/modeling path:

1. Load and resolve per-interface configs.
2. Resolve the selected criticality profile.
3. Generate or reuse bitmap collections.
4. Collapse risk-equivalent interfaces when possible.
5. Call `Pad_Yield_Map_Generator()`.
6. Write text risk maps and PNG mechanism maps.

`Pad_Yield_Map_Generator()` computes pad-level yield maps:

- Overlay: `overlay_yield_calculator.pad_overlay_yield_map_generator()`
- Particle/void: `defect_yield_calculator.pad_defect_yield_map_generator()`
- Mechanical/Cu recess: `Cu_expansion_yield_calculator.pad_Cu_expansion_yield_map_generator()`
- ESD: `esd_yield_calculator.pad_esd_yield_map_generator()`

Large ESD maps may be subsampled. The config field
`ESD_PAD_MAP_SUB_FACTOR` controls this:

- `0` or unset: choose a factor automatically from active pad count.
- `1`: evaluate all active pads.
- `>1`: evaluate an endpoint-preserving sampled grid and upsample the map.

## 7. D2W Monte Carlo Simulation Flow

`D2W/simulator_main.py` performs the stochastic assembly simulation:

1. Load and resolve per-interface configs.
2. Build or reuse bitmap collections.
3. Initialize die-stack samples through `die_stack_list_initialize()`.
4. Draw overlay samples with `overlay_term_simulator()`.
5. Draw particle/void samples with `defect_yield_simulator()`.
6. Evaluate survival with `overall_yield_simulator()`.
7. Aggregate stack yield and per-interface yield.
8. Optionally write verbose failure vectors and failure maps.

The simulation is batched by:

- `NUM_DIE_STACKS`
- `SIM_BATCH_SIZE`

Verbose mode tracks the failure mechanism vectors for:

- `overlay`
- `particle`
- `mechanical`
- `ESD`
- `overall`

Failure heatmaps are only written when `--save-failure-maps` is also enabled.

## 8. Failure Mechanisms

### Overlay

Overlay uses sampled translation, rotation, magnification, and random
misalignment terms. The maximum allowed misalignment is computed from the tighter
of:

- contact area constraint
- critical distance constraint

The D2W simulator can check all pads or an approximation set controlled by
`approximate_set`.

### Particle and Void

D2W particle density uses `D0` and optional edge-enhanced density through:

- `D1`
- `EDGE_REGION_WIDTH_um`

If `D1 <= D0`, the edge enhancement is effectively disabled. Current
`design_6.yaml` keeps `D1` equal to `D0` unless a stronger edge particle model is
desired.

Particle thickness follows the configured power-law model using `t_0` and `z`.
Generated particles are transformed into main voids and tails, then checked
against critical/redundant pad geometry.

### Mechanical / Cu Recess / Debond

Mechanical yield is based on Cu dishing/recess distributions and debond limits:

- `TOP_DISH_MEAN_nm`, `TOP_DISH_STD_nm`
- `BOT_DISH_MEAN_nm`, `BOT_DISH_STD_nm`
- effective layer parameters from `debond.py`
- chiplet/interface thickness from `.3dbv` through `ITF_TOP_THICK_um` and
  `ITF_BOT_THICK_um`

`debond_dishing_intervals_from_coords()` computes valid dishing intervals for
active pads and caches them under the runtime temp folder. For very large pad
arrays, `CU_RECESS_DIE_LEVEL_THRESHOLD_PADS` allows a die-level approximation.

D2W also samples initial chiplet warpage and fails the interface if it exceeds
`WARPAGE_LIMIT_UM`.

### ESD

D2W ESD is no longer driven by `D2W/esd_hybrid.py`.

Current D2W ESD modules are:

- `D2W/esd_yield_simulator.py`
  - Used by `D2W/overall_yield_simulator.py` in the Monte Carlo simulation path.
  - Selects a first-touch pad with sampled tilt, dishing, charging voltage, and
    arcing distance.
  - Applies the Weibull failure model using YAML fields:
    `V_MIN_V`, `V_MAX_V`, `WEIBULL_K`, `WEIBULL_LAMBDA`, and `CUTOFF_MIN_A`.

- `D2W/esd_yield_calculator.py`
  - Used by `D2W/assembly_yield_calculator.py` in the pad risk-map path.
  - Computes analytical ESD yield maps using quadrature, candidate-pad pruning,
    and optional pad-map subsampling.

`D2W/esd_hybrid.py` is a legacy/demo implementation and is not imported by the
current D2W flow. If it exists locally, treat it as removable legacy code. Do not
remove `W2W/esd_hybrid.py` without updating W2W call sites.

## 9. Output Files and Runtime Caches

Main D2W output root:

```text
output/<design>/
```

Per-interface output root:

```text
output/<design>/<interface>/
```

Common output files:

- `<interface>_risk__<config_stem>__<criticality_profile>.map`
- `<interface>_<mechanism>_risk_map__<config_stem>__<criticality_profile>.png`
- `assembly_yield_summary__<config_stem>__<criticality_profile>.txt`
- `assembly_yield_per_interface__<config_stem>__<criticality_profile>.txt`
- `assembly_fail_vec_per_interface_dict__<config_stem>__<criticality_profile>.npz`
- `assembly_fail_map_per_interface_dict__<config_stem>__<criticality_profile>.npz`
- `simulation_failure_map_<mechanism>__<config_stem>__<criticality_profile>.png`

Runtime temp files are isolated under:

```text
output/<design>/temp/
```

The run tag includes design name, config stem, criticality profile, and a hash.
This keeps sorted bump-map copies and dishing-bound caches from colliding across
different runs. Entry points call `cleanup_runtime_temp_files()` for files
associated with the current run.

## 10. W2W Notes

W2W has its own flow and should not be assumed to match D2W one-for-one.

Current W2W notes:

- `W2W/utils/util.py` now extracts `ITF_TOP_THICK_um` and `ITF_BOT_THICK_um`
  from `.3dbv` when resolving `design_6`.
- `W2W/configs/design_6/design_6.yaml` includes the matching thickness fields in
  both simulation and modeling sections.
- `W2W/overall_yield_simulator.py` still imports `W2W/esd_hybrid.py`.
- `W2W/esd_hybrid.py` is still required unless W2W is migrated to the newer D2W
  ESD split.

## 11. Maintainer Notes

When updating D2W:

- Keep new model knobs explicit in `D2W/configs/design_6/design_6.yaml`.
- Do not reintroduce hidden D2W runtime default functions.
- Update both `d2w_simulation` and `d2w_modeling` sections when adding shared
  model parameters.
- Update README commands and this document when adding or replacing entry points.
- Check imports before deleting legacy modules. `D2W/esd_hybrid.py` is not used;
  `W2W/esd_hybrid.py` is still used.
- Treat generated configs, runtime temp files, and output artifacts as
  regenerable unless a specific experiment requires preserving them.
