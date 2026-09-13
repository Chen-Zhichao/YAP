# Figure 15: W2W Reproduction with Historical Parameters and Current `yap+`

This directory covers all 12 configurations in Figure 15 and generates every
result with the current `yap+` W2W calculators. The uploaded December 2025 code
is used only to recover parameters; no legacy engine, MAT/NPY result, or legacy
image is copied into the results.

## Status: PASS

A configuration passes when the absolute errors of `Y_ovl`, `Y_cr`, `Y_df`,
and `Y_W2W` are all at most 0.015 relative to values read from the paper.

| Parameter profile | Passing cases | RMSE | Maximum absolute error | Status |
| --- | ---: | ---: | ---: | --- |
| Paper Table I | 12/12 | 0.004148 | 0.014272 | PASS |
| December 2025 `w2w_modeling` | 12/12 | 0.003120 | 0.010954 | **PASS** |

The December 2025 profile is closer to the paper and is used in the final
plot. It uses rotation mean/standard deviation `5e-8/1e-8 rad`, top-to-bottom
radius ratio `2/3`, translation standard deviation `0.02 µm`, 100% critical
pads, 0% redundant pads, and 400 µm pad blocks.

The 12 columns are the complete Cartesian product of two particle densities
(0.1 and 0.01 cm⁻²), two pitches (1 and 0.3 µm), and three die areas (10, 50,
and 100 mm²). The three rightmost columns are included.

## Rerun

Run from the repository root:

```bash
MPLCONFIGDIR=/tmp/mpl-yap-fig15 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHON=/u1/ee/zhichao/anaconda3/envs/yap_env/bin/python \
reproduction/fig15/run.sh --jobs 4
```

The workflow evaluates both qualifying historical profiles, generates 24
current-code case files, summarizes errors, plots the result, and verifies the
outputs. Use `--profile paper_table_i` or
`--profile december_2025_modeling` for a single-profile diagnostic run. Omit
that option for the complete formal result.

## Files

- `config.yaml`: 12 cases, paper readbacks, two parameter profiles, and the
  pass threshold.
- `run_cases.py`: evaluates every case through the current
  `reproduction/model_worker.py`.
- `make_plot.py`: plots all 12 columns; black ticks show paper reference values.
- `verify.py`: checks coverage, the 100% critical layout, 400 µm blocks, yield
  products, errors, and commit provenance.
- `results/summary.json`: machine-readable per-profile and per-case status.
- `results/fig15_w2w_all.png` and `.pdf`: final current-code plot.
- `results/<profile>/case_01.json` through `case_12.json`: individual runs.

The yield-axis maximum is fixed at 1.0. Because the paper references were read
from a raster figure, 0.015 is a reproduction tolerance consistent with that
readback precision, not a statistical confidence interval for original data.
