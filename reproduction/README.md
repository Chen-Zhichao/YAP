# YAP+ Paper Reproduction Package

This directory is the handoff package for reproducing Figures 13, 15, 16, 17,
and 19 of the YAP+ paper. It must remain inside a checkout of the YAP repository
because the runners import the current W2W and D2W calculators from the parent
repository.

No historical output is used as a current result. Historical source revisions
are consulted only to recover parameters or audit an implementation difference.
Each generated summary records a SHA-256 fingerprint of the calculator sources;
it also records the Git revision when Git metadata is available.

## Package layout

```text
reproduction/
├── README.md              # this guide
├── PROVENANCE.md          # source versions and recovered-input history
├── requirements.txt       # Python dependencies used by the runners
├── model_worker.py        # shared adapter to the current W2W/D2W calculators
├── configs/               # shared paper-baseline configurations
├── run_all.py             # one-command runner for all five figures
├── fig13/
├── fig15/
├── fig16/
├── fig17/
└── fig19/
```

Every figure directory contains:

- an English `README.md` describing the experiment, parameters, result status,
  and known limitations;
- `config.yaml`, including paper reference values or paper-reported metrics;
- a current-code runner (`run_sweep.py` or `run_cases.py`) and `run.sh`;
- `make_plot.py` for regenerating the PNG/PDF;
- `verify.py` for checking coverage, formulas, stored data, and error metrics;
- `results/`, containing current generated data, paper comparisons/errors, and
  the final plot.

Figure-specific audit scripts are retained only when they support a scientific
conclusion in that figure's README. Temporary work directories, Python caches,
the former duplicate top-level result set, and superseded compatibility drivers
are intentionally excluded.

## Fresh-server setup

Python 3.12 is recommended and is the version used for the checked-in results
(Python 3.11--3.13 is supported by the pinned dependencies). No
machine-specific path, pre-existing Conda
environment, display server, historical Git checkout, or external data archive
is required for the formal runs. Starting from a fresh checkout:

```bash
git clone --branch yap+ https://github.com/Chen-Zhichao/YAP.git
cd YAP
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r reproduction/requirements.txt
```

If the repository is supplied as a source archive, extract it and start at the
`cd YAP` step. Git metadata is optional. The runners resolve all source,
configuration, output, and work paths relative to their own locations and use
Matplotlib's noninteractive `Agg` backend.

For review, use the repository commit or release archive supplied with these
materials rather than an independently updated moving branch. The checked-in
calculator fingerprint deliberately rejects results if the W2W/D2W source has
changed.

## Reproduce all figures

Run from the YAP repository root:

```bash
python reproduction/run_all.py --jobs 4
```

To run selected figures:

```bash
python reproduction/run_all.py --figures 13 17 --jobs 4
```

Each figure can also be run independently. For example:

```bash
bash reproduction/fig16/run.sh --jobs 4
```

`--jobs` controls concurrent case workers; reduce it on a small server. To use
an interpreter without activating its environment, pass
`--python /path/to/python` to `run_all.py`, or set `PYTHON=/path/to/python` for
an individual `run.sh`.

To validate the checked-in data and source fingerprints without rerunning the
Monte Carlo calculations:

```bash
python reproduction/run_all.py --verify-only
```

A successful verifier means that the files are internally consistent with the
declared status. It does not turn a figure labeled `NOT PASS` into a successful
paper match.

## Current result summary

| Figure | Coverage | Current status | Primary comparison file |
| --- | --- | --- | --- |
| 13 | W2W + D2W, 300 points each | W2W MSE `1.256e-4` vs paper `1.188e-4`; D2W `2.301e-5` vs `1.449e-5` | `fig13/results/fig13_{w2w,d2w}.json` |
| 15 | W2W, all 12 configurations | **PASS**, 12/12 | `fig15/results/summary.json` |
| 16 | D2W, all 12 configurations and system yield | **NOT PASS**, best recovered profile 8/12 | `fig16/results/summary.json` |
| 17 | W2W + D2W, four layouts each | Numerical reconstruction **PASS**, 8/8; provenance remains qualified | `fig17/results/current_summary.json` |
| 19 | W2W + D2W, both densities, all spacings, and 20:1 | **NOT PASS**, 12/24 bars | `fig19/results/summary.json` |

The individual figure READMEs explain these classifications and identify any
inferred parameter, historical compatibility calculation, or unresolved input.

## Important modeling choices

- The scalar overlay calculation uses the paper-era order: select the worst
  corner within each Monte Carlo sample, calculate that sample's conditional
  yield, and then average across samples.
- Figures 15 and 16 use 100% critical pads and include all 12 published
  density/pitch/area configurations.
- Figure 17 reports raw-current, historical-overlay, and explicitly labeled
  sensitivity profiles separately.
- Figure 19 uses the paper's 200×200 µm block dimension and exact bar values
  recovered from the supplied MATLAB source. Its 20:1 current-code case remains
  clearly labeled as a compact-grid approximation.

See `PROVENANCE.md` and the per-figure READMEs for the complete audit trail.
