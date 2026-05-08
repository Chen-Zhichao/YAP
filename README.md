# YAP-stack

YAP-stack is a Python-based yield modeling and Monte Carlo simulation tool for
advanced packaging. This branch focuses on wafer-to-wafer (W2W) and die-to-wafer
(D2W) hybrid bonding, with the newer D2W YAP+ flow integrated into the stack
simulation path.

A [GUI of YAP](http://nanocad.ee.ucla.edu:8081/yap_gui/) and the
[user guide video](https://youtu.be/8hiKIQ6C7ng) are available.

The active D2W examples in this branch are:

- `design_6`: 3D stack example with substack-level warpage support.
- `design_2`: 2.5D four-compute-chiplet example. Its active pad-ratio layout is
  stored under `D2W/input/design_2/c30_r0_pg50_dm20`.

HBM configs and other paper-specific input sets from the IO-assignment workspace
are not included here unless explicitly copied into this repo.

# File Structure

```
.
├── D2W/
│   ├── configs/
│   │   ├── design_2/
│   │   └── design_6/
│   ├── input/
│   │   ├── design_2/
│   │   │   ├── generated_chiplet_definitions.3dbv
│   │   │   ├── generated_stack_config.3dbx
│   │   │   ├── *.3dbf
│   │   │   └── c30_r0_pg50_dm20/
│   │   │       ├── *.bmap
│   │   │       ├── *_criticality.txt
│   │   │       ├── *_criticality_esd_strict.txt
│   │   │       └── Compute_Large_interchip_shared_nets.txt
│   │   └── design_6/
│   ├── pad_risk_map_calculator.py
│   ├── simulator_main.py
│   ├── warpage_yield_calculator.py
│   ├── warpage_yield_simulator.py
│   └── utils/
├── W2W/
├── README.md
└── requirements.txt
```

For D2W 3dblox designs, common files (`.3dbv`, `.3dbx`, `.3dbf`) live in the
design root. Ratio-specific files (`.bmap`, criticality files, shared-net
summaries) may live either directly in the design root or in a single ratio
subdirectory. The current `design_2` uses the ratio-subdirectory style.

# Installation

Run installation commands from the repository root.

For simulation and pad-risk-map generation, first `cd` into the corresponding
flow directory (`D2W` or `W2W`) and run the entrypoint from there. This keeps
relative output paths such as `output/` inside the active flow directory instead
of depending on the shell's previous working directory.

```
conda create -n yap_env python=3.12
conda activate yap_env
pip install -r requirements.txt
```

# D2W Input Notes

## Ratio Folder Resolution

The D2W simulation and risk-map entrypoints resolve bump-map files using this
order:

1. `D2W/input/<design>/<interface>.bmap`
2. `D2W/input/<design>/<single_ratio_folder>/<interface>.bmap`

If more than one ratio subdirectory contains the same interface bmap, the
resolver raises an ambiguity error. Keep only the active ratio folder under the
design root when running with `--ds_dir input/<design>` from inside `D2W`.

## Current `design_2` Layout

The active ratio folder is:

```
D2W/input/design_2/c30_r0_pg50_dm20
```

Each interface bmap has:

- 30% critical bumps.
- 50% power/ground bumps.
- 20% dummy bumps.
- 0% redundant bumps.

For the four compute chiplets, the critical bumps are assigned from the die
center outward. Approximately half of the critical bumps are inter-chip shared
links recorded in:

```
D2W/input/design_2/c30_r0_pg50_dm20/Compute_Large_interchip_shared_nets.txt
```

The remaining critical bumps are named as chiplet-local external critical nets.

# Criticality Generation

Generate both default and strict-ESD criticality files for all bmaps under
`design_2`:

```
python D2W/utils/generate_criticality.py --input-root D2W/input --designs 2 --profiles both --force
```

Generate criticality files from one explicit bump map:

```
python D2W/utils/generate_criticality.py --file D2W/input/design_6/Memory_DRAM_3_From_Memory_DRAM_2.bmap --profiles both --force
```

Supported profiles:

- `default`: replicated nets tolerate `R-1` ESD failures and `R-1` mechanical
  failures.
- `esd_strict`: replicated signal nets tolerate `0` ESD failures and `R-1`
  mechanical failures.
- `both`: generate both files.

# D2W Pad Risk Maps

`pad_risk_map_calculator.py` is the current D2W modeling entrypoint.
`calculator_main.py` remains as a compatibility wrapper.

Benchmark example:

```
cd D2W

python pad_risk_map_calculator.py --config configs/design_2/design_2.yaml --mode d2w_modeling --ds_name design_2 --ds_dir input/design_2 --criticality-profile default --verbose
```

Analytical ESD map notes:

- Large pad arrays are automatically subsampled for ESD map generation and then
  interpolated back to the full pad map.
- Override the ESD grid factor with `ESD_PAD_MAP_SUB_FACTOR` in the YAML.
- If `ESD_PAD_MAP_SUB_FACTOR` is unset or `0`, the code chooses a factor from
  the active pad count.
- `--plot` enables extra interactive mechanism plots; PNG risk maps are written
  by default.

# D2W Yield Simulation

Benchmark example:

```
cd D2W

python simulator_main.py --config configs/design_2/design_2.yaml --mode d2w_simulation --ds_name design_2 --ds_dir input/design_2 --criticality-profile default --verbose
```

The simulator writes an assembly summary and a per-interface yield file.
Runtime temp files are isolated by design name, config stem, and criticality
profile.


# Stack Warpage Flow

The D2W stack-warpage path is split into calculator and simulator modules:

- `D2W/warpage_yield_calculator.py`: Gaussian substack warpage calculator.
- `D2W/warpage_yield_simulator.py`: Monte Carlo validation and stack-level
  failure-vector generation.

The stack graph is built from `.3dbx` using `stack_graph_from_3dbx()` in
`D2W/utils/util.py`. It returns each substack as bottom-to-top chiplet and
interface lists.

Current behavior:

- Final stack warpage is evaluated per substack.
- A die stack fails the warpage criterion when any final substack exceeds the
  absolute threshold `STACK_WARPAGE_TH`. If that key is absent, the code falls
  back to `WARPAGE_LIMIT_UM`, then `20.0`.
- Warpage is a stack-level yield criterion. It is not recorded as a per-interface
  failure mechanism in verbose failure maps.
- Overlay simulation uses substack-aware bow difference samples:

```
bow_difference = incoming_top_die_initial_bow - existing_substack_warpage
```

Initial bow parameters are read from:

- `TOP_INI_BOW_MEAN_um`, `TOP_INI_BOW_STD_um`
- `BOT_INI_BOW_MEAN_um`, `BOT_INI_BOW_STD_um`

If those keys are missing, the code falls back to the legacy
`BOW_DIFFERENCE_MEAN_um` and `BOW_DIFFERENCE_STD_um` values.

Layer elastic properties use volume-fraction effective material properties from
the YAML when available (`*_Cu_V`, `*_Sio2_V`, `*_Si_V` and material constants).
Missing mixture information falls back to a pure-Si placeholder.

Random sampling uses system entropy by default; fixed seeds have been removed
from the D2W simulation chain.

# W2W Flow

Benchmark W2W simulation example:

```
cd W2W && python simulator_main.py --config configs/design_6/design_6.yaml --mode w2w_simulation --ds_name design_6 --ds_dir input/design_6 --verbose
```

W2W runs use the same terminal duck separator at completion.

# File Formats

## Bump Map (`.bmap`)

```
<instance> <bump_type> <x> <y> <port> <net>
```

Example:

```
Bump_0 uBUMP 115 1610 txdatasb txdatasb
```

## Risk Map (`.map`)

```
<x> <y> <esd_failure_probability> <overlay_failure_probability> <particle_failure_probability> <mechanical_failure_probability>
```

Probabilities are float values between `0` and `1`. ESD criticality is applied
to ESD probability; mechanical criticality is applied to overlay, particle, and
mechanical probabilities.

## Criticality (`.txt`)

Current format:

```
<net> <group_size> <tolerated_esd_failures> <tolerated_mechanical_failures>
```

Where:

- `group_size`: total number of pads/bumps in the redundancy group.
- `tolerated_esd_failures`: number of ESD failures the group can tolerate before
  failing.
- `tolerated_mechanical_failures`: number of mechanical failures the group can
  tolerate before failing.

Filename variants:

- `*_criticality.txt`: default profile.
- `*_criticality_esd_strict.txt`: strict ESD profile.

Criticality values are calculated when reading the file:

```
esd_criticality = (group_size - tolerated_esd_failures) / group_size
mechanical_criticality = (group_size - tolerated_mechanical_failures) / group_size
```

Legacy format is deprecated but still supported:

```
<net> <esd_criticality> <mechanical_criticality>
```

## 3dblox Files

- `.3dbv`: chiplet definitions, design areas, regions, and thicknesses.
- `.3dbx`: stack configuration containing chiplet instances and connections.
- `.3dbf`: chiplet bump metadata such as pitch and bump size.

# Output

When commands are run from inside `D2W`, main D2W outputs are written under
`D2W/output/<design_name>/`. W2W commands follow the same convention under
`W2W/output/<design_name>/`.

- `<interface>_risk__<config_stem>__<criticality_profile>.map`: text risk map.
- `<interface>_<mechanism>_risk_map__<config_stem>__<criticality_profile>.png`:
  per-mechanism pad risk maps.
- `assembly_yield_summary__<config_stem>__<criticality_profile>.txt`: simulation
  settings, runtime information, stack assembly yield, and per-interface yield.
- `assembly_yield_per_interface__<config_stem>__<criticality_profile>.txt`:
  per-interface simulated assembly yield.
- `assembly_fail_vec_per_interface_dict__<config_stem>__<criticality_profile>.npz`:
  failure vectors for each die sample and failure mechanism; verbose mode only.
- `assembly_fail_map_per_interface_dict__<config_stem>__<criticality_profile>.npz`:
  average per-pad failure counts; requires `--verbose --save-failure-maps`.
- `simulation_failure_map_<mechanism>__<config_stem>__<criticality_profile>.png`:
  per-interface simulation failure heatmaps for `overlay`, `particle`,
  `mechanical`, `ESD`, and `overall`; requires `--save-failure-maps`.

# Generator Utilities

Useful D2W helpers include:

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

Runtime files in `D2W/output/`, `W2W/output/`, and generated cache files are not
required as source inputs. Re-run the modeling or simulation commands to
regenerate them.
