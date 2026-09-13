# YAP+ Paper Reproduction Package

This directory is the handoff package for reproducing Figures 13, 15, 16, 17,
and 19 of the YAP+ paper. It must remain inside a checkout of the YAP repository
because the runners import the current W2W and D2W calculators from the parent
repository.

No historical output is used as a current result. Historical source revisions
are consulted only to recover parameters or audit an implementation difference.
Each generated JSON records its calculation commit and provenance.

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

## Environment

The tested environment is:

```text
/u1/ee/zhichao/anaconda3/envs/yap_env/bin/python
```

The required packages are listed in `requirements.txt`. To use another Python
environment, install those dependencies and pass its interpreter through
`PYTHON` or `--python`.

## Reproduce all figures

Run from the YAP repository root:

```bash
MPLCONFIGDIR=/tmp/mpl-yap-reproduction \
PYTHONDONTWRITEBYTECODE=1 \
/u1/ee/zhichao/anaconda3/envs/yap_env/bin/python \
reproduction/run_all.py --jobs 4
```

To run selected figures:

```bash
/u1/ee/zhichao/anaconda3/envs/yap_env/bin/python \
reproduction/run_all.py --figures 13 17 --jobs 4
```

Each figure can also be run independently. For example:

```bash
PYTHON=/u1/ee/zhichao/anaconda3/envs/yap_env/bin/python \
reproduction/fig16/run.sh --jobs 4
```

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
