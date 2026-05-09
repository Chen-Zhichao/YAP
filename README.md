# YAP-stack

- YAP-stack is a Python-based yield modeling and Monte Carlo simulation tool for
  advanced packaging. This branch focuses on wafer-to-wafer (W2W) and
  die-to-wafer (D2W) hybrid bonding, with stack-aware D2W warpage support.
- A [GUI of YAP](http://nanocad.ee.ucla.edu:8081/yap_gui/) and the
  [user guide video](https://youtu.be/8hiKIQ6C7ng) are available.

Current local benchmark examples used in this workspace are:

- `design_2`: 2.5D four-compute-chiplet D2W benchmark.
- `design_6`: 3D D2W/W2W stack benchmark with substack-level warpage support.

Benchmark config and input folders are ignored by git in this branch. Keep them
locally under `D2W/configs/`, `D2W/input/`, `W2W/configs/`, and `W2W/input/`
when running the examples.

# File Structure

```
.
+-- D2W/      # Code for D2W hybrid bonding
|   +-- configs/    # Golden config plus local benchmark configs
|   +-- input/      # Local 3dblox inputs, bump maps, and criticality files
|   +-- calculator_main.py
|   +-- simulator_main.py
|   +-- overlay_yield_calculator.py
|   +-- warpage_yield_calculator.py
|   +-- warpage_yield_simulator.py
|   +-- utils/
+-- W2W/      # Code for W2W hybrid bonding
|   +-- configs/
|   +-- input/
|   +-- calculator_main.py
|   +-- simulator_main.py
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

- Run W2W yield simulation.

  Always enter `W2W` before running W2W commands. This keeps relative outputs
  under `W2W/output/`.

  ```
  cd W2W && python simulator_main.py --config configs/design_6/design_6.yaml --mode w2w_simulation --ds_name design_6 --ds_dir input/design_6 --verbose
  ```

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

- Files matched by `.gitignore` are not required as source inputs on GitHub.
  Keep local benchmark configs and inputs outside git unless they are meant for
  release.
- Runtime files in `D2W/output/`, `W2W/output/`, and generated cache files can
  be regenerated by rerunning the modeling or simulation commands.
- D2W and W2W runs print the terminal duck separator when an experiment
  finishes.

# Paper Link

To be continued.
