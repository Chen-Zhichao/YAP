# Figure 17: W2W/D2W Reproduction for Four I/O Pad Layouts

This directory evaluates all eight Figure 17c cases with the current `yap+`
calculators: **W2W and D2W × Full, Sparse, Peripheral, and Centralized**. It
also performs a traceable regression audit using the paper-era sample-wise
overlay implementation. Legacy source and legacy outputs are not presented as
current-code results.

## Parameters and undisclosed inputs

The paper explicitly specifies a `0.3 µm` pitch. Full uses 100% critical pads;
the other layouts use 20% critical, 50% redundant, and 30% dummy pads. W2W and
D2W use the Table-I block dimensions of `400 µm` and `100 µm`, respectively.
Other baseline values also come from Table I.

The paper does not report `redundant_logical_pad_ratio`, the fraction of the
50% physical redundant pads that actually participate in logical main/copy
groups. This must not be conflated with the stated 50% redundant-pad fraction.
The reproduction therefore retains three distinct profiles:

- `paper_table_i` sets the undisclosed logical mapping to zero, isolating the
  critical-layout contribution that is explicitly supported by the paper.
- `august_2025_modeling` uses `0.3` from Git commit `331ac05` together with that
  commit's pad-size, rotation, and magnification settings.
- `inferred_logical_mapping` uses W2W `0.05`, inferred from the three Figure 17
  `Ydf` bars, and D2W `0.005` from `331ac05`. This profile is explicitly labeled
  as a sensitivity fit, not a recovered historical configuration.

All profiles remain in the output; inferred values do not replace missing
historical parameters.

## Current-code results

Using an absolute tolerance of `0.02` for values read from the paper raster:

- The Table-I profile passes `Ycr/Ydf` for 7 of 8 layouts. All four D2W layouts
  pass, with a maximum error of `0.0021`.
- The only failing non-overlay item is W2W Centralized `Ydf`: current
  `0.96790`, paper approximately `0.942`, error `0.02590`.
- The current calculators restore sample-wise worst-corner overlay. With the
  Table-I profile, 3 of 8 cases pass all four plotted metrics. The remaining
  D2W difference is primarily due to Table-I versus August-2025 D2W modeling
  parameters, not corner aggregation.
- The `august_2025_modeling` profile with logical ratio 0.3 does not reproduce
  the W2W defect breakdown, so citing an old configuration alone is
  insufficient evidence that the full figure is reproduced.

As a missing-parameter sensitivity check, `inferred_logical_mapping` passes all
non-overlay quantities in all 8 cases. Its three non-Full W2W `Ydf` values are
`0.90882/0.91027/0.94010`, versus paper values of approximately
`0.908/0.910/0.942`. Until the original bitmap/configuration is recovered, this
only shows that the missing parameter explains the difference; it does not
establish that `0.05` was the historical setting.

The raw-current result without wafer scaling remains **4/8** and is retained
as `fig17_current_all.png` to document why the earlier D2W overlay plot was too
high.

The main reproduction profile, `paper_era_wafer_scaled`, uses the current
calculator's explicit switch together with paper-era sample-wise aggregation,
May-2025 wafer/die scaling, the current layout-aware boundary, and the clearly
labeled logical-mapping sensitivity. It passes **8/8**, with overall RMSE
`0.004751` and maximum absolute error `0.012880`. Its D2W overlay values are
`0.927461/0.928072/0.927515/0.952880`, replacing the earlier incorrect main-plot
values of approximately `0.9666/0.9666/0.9666/0.9673`.

Figure 17 therefore passes numerically, but its provenance conclusion remains
qualified: scaling and layout-aware boundaries occur separately in historical
code, and no recovered commit contains both. The W2W logical ratio `0.05` is a
readback-based sensitivity value rather than a recovered original setting.

## Can the paper-era sample-wise overlay reproduce the paper?

Yes. This conclusion does not come from arbitrary tuning:
`legacy_overlay.py` directly checks and reproduces algorithmic features from
two historical Git commits.

| Historical implementation | Parameter source | Overlay cases passing | Maximum absolute error | D2W Centralized−Full |
| --- | --- | ---: | ---: | ---: |
| `6cc2408` (May 2025) | Table I | 8/8 | 0.01254 | 0.0000 (critical boundary not yet layout-aware) |
| `331ac05` (August 2025) | Same-commit modeling configuration | 8/8 | 0.01771 | 0.01658 (paper approximately 0.020) |

Commit `331ac05` also reproduces the paper's higher D2W Centralized overlay
trend, with a trend error of only `0.00342`. The correct die-level order is:
for each Monte Carlo sample, take the maximum four-corner systematic
misalignment; evaluate the sample's one-dimensional random-error CDF yield;
then average across samples. The current calculator restores this order. The
earlier D2W implementation in `6cc2408` additionally scaled rotation and
magnification by `wafer radius / die half-diagonal`.

Historical-formula outputs are isolated in `legacy_overlay_summary.json` and a
separate audit plot and serve as regression references for the current
sample-wise implementation. With August-2025 parameters, the current
implementation differs from the eight `331ac05` overlay values by at most
approximately `3.6e-5`, attributable to the Monte Carlo seed. This rules out
aggregation order as the remaining discrepancy; parameter and layout
provenance are responsible.

Wafer-dimension scaling is audited separately. With Table-I parameters, the
May-2025 D2W scaling gives Full `0.92746`, versus paper `0.920`, but that commit
predates layout-aware critical boundaries and therefore produces the same
overlay for all four layouts. Combining that historical scaling with the
August-2025 boundary, only as a sensitivity study, produces
`0.92746/0.92802/0.92746/0.95286`, all within 0.02. This explains Figure 17 but
cannot be described as the original implementation because no recovered commit
contains the combination.

Finally, combining the `331ac05` historical parameter/trace, current
non-overlay calculators, and the clearly marked `inferred_logical_mapping`
passes the complete four-metric breakdown for 8/8 cases, with maximum error
`0.01771`. This establishes numerical closure with the paper-era overlay trace,
while preserving the warning that the original logical-mapping provenance has
not been recovered.

## Rerun

Run from the repository root:

```bash
MPLCONFIGDIR=/tmp/mpl-yap-fig17 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHON=/u1/ee/zhichao/anaconda3/envs/yap_env/bin/python \
reproduction/fig17/run.sh --jobs 4
```

## Files

- `config.yaml`: eight paper readbacks, four current parameter profiles, and
  two historical overlay profiles.
- `run_cases.py`: calls the current `model_worker.py` and current calculators.
- `legacy_overlay.py`: audits historical algorithm compatibility and uses
  `git show` to verify the formulas in history.
- `make_plot.py`: generates the current breakdown and historical
  parameter/scaling audit plots, with a yield-axis maximum of 1.00.
- `verify.py`: checks case coverage, physical ratios, yield products, commit,
  historical formulas, and the D2W layout trend.
- `results/current_summary.json`: current-code per-case results and errors.
- `results/legacy_overlay_summary.json`: historical-formula results and errors.
- `results/fig17_all.png` and `.pdf`: complete current W2W/D2W result.
- `results/fig17_current_all.png` and `.pdf`: raw-current comparison with
  historical wafer scaling disabled.
- `results/fig17_legacy_overlay_audit.png` and `.pdf`: paper/current/historical
  overlay comparison.
