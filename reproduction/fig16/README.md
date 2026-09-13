# Figure 16: D2W Reproduction with Historical Parameters and Current `yap+`

This directory covers all 12 configurations in Figure 16 and regenerates them
with the current `yap+` D2W calculators. Paper reference values were transcribed
case by case from the original spreadsheet screenshot supplied by the user. No
legacy engine or legacy computed result is copied into this package.

## Status: NOT FULLY REPRODUCED

The current D2W calculator restores the paper-era scalar-yield order: for each
Monte Carlo sample, take the largest systematic misalignment over the four
corners, calculate that sample's conditional overlay yield, and then average
the sample yields. Consequently, `Y_ovl`, `Y_cr`, `Y_df`, `Y_D2W`, and `Y_sys`
all participate in the pass/fail gate, with an absolute-error limit of 0.015
for every metric.

| Historical parameter profile | Cases passing all five metrics | RMSE | Maximum error | Status |
| --- | ---: | ---: | ---: | --- |
| Paper Table I, 100 µm blocks | 7/12 | 0.029792 | 0.142705 | NOT PASS |
| December 2025 `d2w_modeling`, 50 µm blocks | **8/12** | **0.009217** | **0.043118** | **NOT PASS** |
| December 2025 notebook, 50 µm blocks | 7/12 | 0.011724 | 0.052447 | NOT PASS |

The final plot uses the lowest-error December 2025 `d2w_modeling` profile. Its
settings include top-to-bottom radius ratio `2/3`, translation standard
deviation `0.02 µm`, rotation mean/standard deviation `1e-6/5e-7 rad`, 100%
critical pads, 0% redundant pads, and the historical 50 µm D2W modeling block.
The Table-I profile independently checks the paper's stated 100 µm block, so
the two provenance paths are not mixed.

## Failing configurations

For the best profile, configurations `2`, `9`, `11`, and `12` fail. Their
largest errors all occur in `Y_sys` and are respectively
`0.019231`, `0.018163`, `0.029573`, and `0.043118`. The fine-pitch `Y_ovl`
values are `0.957701/0.947130/0.937493` for 10/50/100 mm², compared with paper
values `0.956360/0.939716/0.921729`. The first two are close, while the 100 mm²
case still differs by `0.015764`.

`Y_cr` and `Y_df` remain close to the paper in all cases, but the area exponent
in system yield amplifies small single-die residuals. Figure 16 therefore
cannot currently be claimed as a full reproduction.

## Wafer-dimension scaling audit

Commit `6cc2408` multiplied D2W rotation and magnification samples by
`WAF_R / hypot(DIE_W/2, DIE_L/2)`. `wafer_scaling_audit.py` uses `git show` to
verify that this expression exists in history and then evaluates all 12 cases
independently. The audit shows:

- With Table-I parameters, overlay RMSE decreases from `0.022254` to
  `0.012601`, but the maximum error remains `0.027935`, above the 0.015 gate.
- The scaling cancels the die-radius dependence, making all three 10/50/100
  mm² fine-pitch overlay yields `0.928426`. This cannot reproduce the paper's
  `0.956360/0.939716/0.921729` area trend.
- Applying the same scaling to the December 2025 modeling parameters increases
  overlay RMSE to `0.627126`, which is clear double counting/over-scaling.

Wafer scaling can therefore help explain a fixed-die-size Figure 17 offset,
but it is not a unified correction for Figure 16.

## The 12 configurations

The column order is:

1. Columns 1–3: 0.1 cm⁻², 1 µm pitch, 10/50/100 mm².
2. Columns 4–6: 0.01 cm⁻², 1 µm pitch, 10/50/100 mm².
3. Columns 7–9: 0.1 cm⁻², 0.3 µm pitch, 10/50/100 mm².
4. Columns 10–12: 0.01 cm⁻², 0.3 µm pitch, 10/50/100 mm².

All five quantities—`Y_ovl`, `Y_cr`, `Y_df`, `Y_D2W`, and `Y_sys`—are stored
in the results and plotted. Validation is not limited to the three rightmost
columns.

## Rerun

Run from the repository root:

```bash
MPLCONFIGDIR=/tmp/mpl-yap-fig16 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHON=/u1/ee/zhichao/anaconda3/envs/yap_env/bin/python \
reproduction/fig16/run.sh --jobs 4
```

## Files

- `config.yaml`: 12 spreadsheet-reference cases, three historical parameter
  profiles, and the five-metric gate policy.
- `run_cases.py`: evaluates 36 cases with the current code and summarizes
  metric-level errors.
- `make_plot.py`: plots all 12 columns with a yield maximum of 1.0.
- `verify.py`: checks the 100% critical layout, profile-specific block size,
  yield identities, system-yield formula, all five gates, errors, and commit.
- `wafer_scaling_audit.py`: reproduces and evaluates the `6cc2408` wafer/die
  radial scaling.
- `results/wafer_scaling_audit.json`: scaling comparison for every profile/case.
- `results/summary.json`: complete per-case status.
- `results/fig16_d2w_all.png` and `.pdf`: final current-code plot.
- `results/<profile>/case_01.json` through `case_12.json`: individual runs.
