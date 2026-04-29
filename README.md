# YAP-stack

YAP-stack is a Python-based yield modeling and simulation tool for advanced packaging. It supports arbitrary I/O pad layouts and currently focuses on wafer-to-wafer (W2W) and die-to-wafer (D2W) hybrid bonding.

A [GUI of YAP](http://nanocad.ee.ucla.edu:8081/yap_gui/) and the [user guide video](https://youtu.be/8hiKIQ6C7ng) are available.

The active D2W example in this branch is `design_6`. The D2W code has been updated with the newer YAP+ simulation/modeling flow, but the HBM and paper-specific configs/inputs are not included in this repo.

# File Structure

```
.
├── D2W/                         # Code and examples for D2W hybrid bonding
│   ├── configs/design_6/         # Current D2W design_6 configuration
│   ├── input/design_6/           # Current D2W design_6 3dblox/bmap/criticality inputs
│   ├── pad_risk_map_calculator.py
│   ├── simulator_main.py
│   ├── spatial_correlation_coefficients_main.py
│   └── utils/                    # Bump-map, criticality, plotting, and packaging helpers
├── W2W/                         # Code and examples for W2W hybrid bonding
├── LICENSE
├── README.md
└── requirements.txt
```

# Installation

1. Clone the repository.

```
git clone -b yap-stack https://github.com/Chen-Zhichao/YAP.git
cd ./YAP
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

Run commands from the repository root unless noted otherwise.

## Generate Criticality Files

Generate criticality files from an explicit bump map:

```
python D2W/utils/generate_criticality.py \
  --file D2W/input/design_6/Memory_DRAM_3_From_Memory_DRAM_2.bmap \
  --profiles both \
  --force
```

Supported D2W criticality profiles:

- `default`: replicated redundant nets tolerate `R-1` ESD failures and `R-1` mechanical failures.
- `esd_strict`: replicated redundant nets tolerate `0` ESD failures and `R-1` mechanical failures.
- `both`: generate both profile files.

## D2W Pad Risk Maps

`pad_risk_map_calculator.py` is the current D2W modeling entrypoint. `calculator_main.py` remains as a compatibility wrapper for older commands.

Generate pad-level risk maps for `design_6`:

```
python D2W/pad_risk_map_calculator.py \
  --config D2W/configs/design_6/design_6.yaml \
  --mode d2w_modeling \
  --ds_name design_6 \
  --ds_dir D2W/input/design_6 \
  --criticality-profile default \
  --verbose
```

Equivalent legacy-compatible command:

```
python D2W/calculator_main.py \
  --config D2W/configs/design_6/design_6.yaml \
  --mode d2w_modeling \
  --ds_name design_6 \
  --ds_dir D2W/input/design_6 \
  --criticality-profile default \
  --verbose
```

Notes for analytical ESD maps:

- Large pad arrays are automatically subsampled for ESD map generation and interpolated back to the full pad map.
- Candidate-pad pruning evaluates only pads near the deterministic first-touch edge when safe.
- Override the ESD grid factor with `ESD_PAD_MAP_SUB_FACTOR` in the YAML.
- If `ESD_PAD_MAP_SUB_FACTOR` is unset or `0`, the code chooses a factor from the active pad count.
- `--plot` enables extra interactive mechanism plots; PNG risk maps are written by default.

## D2W Yield Simulation

Run the D2W simulator for `design_6`:

```
python D2W/simulator_main.py \
  --config D2W/configs/design_6/design_6.yaml \
  --mode d2w_simulation \
  --ds_name design_6 \
  --ds_dir D2W/input/design_6 \
  --criticality-profile default \
  --verbose
```

Save verbose simulation failure-map artifacts:

```
python D2W/simulator_main.py \
  --config D2W/configs/design_6/design_6.yaml \
  --mode d2w_simulation \
  --ds_name design_6 \
  --ds_dir D2W/input/design_6 \
  --criticality-profile default \
  --verbose \
  --save-failure-maps
```

The simulator now writes an assembly summary and per-interface yield file. Runtime temp files are isolated by design name, config, and criticality profile.

## D2W Spatial Correlation

Precalculate spatial correlation coefficients for a smaller number of stack samples:

```
python D2W/spatial_correlation_coefficients_main.py \
  --config D2W/configs/design_6/design_6.yaml \
  --mode d2w_simulation \
  --ds_name design_6 \
  --ds_dir D2W/input/design_6 \
  --num-stack-samples 100 \
  --sim-batch-size 10 \
  --criticality-profile default
```

## W2W Flow

Example W2W pad risk map calculation:

```
python W2W/calculator_main.py \
  --config W2W/configs/design_6/design_6.yaml \
  --mode w2w_modeling \
  --ds_name design_6 \
  --ds_dir W2W/input/design_6 \
  --verbose
```

Example W2W simulation:

```
python W2W/simulator_main.py \
  --config W2W/configs/design_6/design_6.yaml \
  --mode w2w_simulation \
  --ds_name design_6 \
  --ds_dir W2W/input/design_6 \
  --verbose
```

# File Formats

**1. Bump Map (`.bmap`)**

Format:

```
<instance> <bump_type> <x> <y> <port> <net>
```

Example:

```
Bump_0 uBUMP 115 1610 txdatasb txdatasb
```

**2. Risk Map (`.map`)**

Format:

```
<x> <y> <esd_failure_probability> <overlay_failure_probability> <particle_failure_probability> <mechanical_failure_probability>
```

Example:

```
115 1610 0.15 0.05 0.03 0.20
```

Probabilities are float values between `0` and `1`. ESD criticality is multiplied by `esd_failure_probability`; mechanical criticality is multiplied by overlay, particle, and mechanical failure probabilities. All four failure modes are considered in the optimization objective.

**3. Criticality (`.txt`)**

Current format:

```
<net1> [net2] [net3] ... <group_size> <tolerated_esd_failures> <tolerated_mechanical_failures>
```

Where:

- `group_size`: total number of pads/bumps in the redundancy group.
- `tolerated_esd_failures`: number of ESD failures the group can tolerate before failing.
- `tolerated_mechanical_failures`: number of mechanical failures the group can tolerate before failing.

Supported filename variants:

- `*_criticality.txt`: default profile.
- `*_criticality_esd_strict.txt`: strict ESD profile.

Criticality values are calculated when reading the file:

- `esd_criticality = (group_size - tolerated_esd_failures) / group_size`
- `mechanical_criticality = (group_size - tolerated_mechanical_failures) / group_size`

Legacy format is deprecated but still supported:

```
<net> <esd_criticality> <mechanical_criticality>
```

**4. 3dblox Files**

- `.3dbv`: stack-level 3dblox file containing chiplet definitions and design areas.
- `.3dbx`: stack configuration file containing chiplet/interface connections.
- `.3dbf`: chiplet file containing bump pitch and bump-size metadata.

# Output

**1. `<interface>_risk__<config_stem>__<criticality_profile>.map`**

Text risk map for an interface. Each line contains pad coordinates and the ESD, overlay, particle, and mechanical failure probabilities.

**2. `<interface>_<mechanism>_risk_map__<config_stem>__<criticality_profile>.png`**

Per-mechanism pad risk maps written by `pad_risk_map_calculator.py`.

**3. `assembly_yield_summary__<config_stem>__<criticality_profile>.txt`**

Simulation summary containing settings, runtime information, stack assembly yield, and per-interface yield.

**4. `assembly_yield_per_interface__<config_stem>__<criticality_profile>.txt`**

Per-interface simulated assembly yield.

**5. `assembly_fail_vec_per_interface_dict__<config_stem>__<criticality_profile>.npz`**

Failure vectors for each die sample and failure mechanism. This is written in verbose simulation mode.

**6. `assembly_fail_map_per_interface_dict__<config_stem>__<criticality_profile>.npz`**

Average per-pad failure counts across simulation samples. This is written only when both `--verbose` and `--save-failure-maps` are enabled.

**7. `simulation_failure_map_<mechanism>__<config_stem>__<criticality_profile>.png`**

Per-interface simulation failure heatmaps for `overlay`, `particle`, `mechanical`, `ESD`, and `overall`. These are written only when `--save-failure-maps` is enabled.

# Generator Utilities

Useful D2W helpers include:

- `D2W/utils/generate_criticality.py`: generate default and strict-ESD criticality files from bump maps.
- `D2W/utils/assign_bump_names.py`: assign net and port names to raw bump maps.
- `D2W/utils/bmap_grid_sync.py`: synchronize bump maps with inferred grid geometry.
- `D2W/utils/plot_bump_kinds.py`: visualize bump categories from a bump map.
- `D2W/utils/plot_design_topology.py`: plot chiplet/interface topology from 3dblox inputs.

# Paper Link

To be continued...
