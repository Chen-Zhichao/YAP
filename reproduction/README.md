# Reproducing YAP+ Figures 13, 15, 16, 17, and 19

This directory evaluates every published case in the requested figures. Every
applicable figure is run for both wafer-to-wafer (W2W) and die-to-wafer (D2W).

The source is upstream `yap+` at `ed0449becd4cbc517dc27bda1bd5e8b762dfc6d9`
plus local commits `cc35068`, `1d4772b`, and `7fde3ad`. These restore the
paper-era sample-wise worst-corner calculation, add an explicit opt-in D2W
wafer/die distortion scale, and correct non-overlapping redundant-block
placement. The paper PDF is stored at
`../YAP_plus_paper.pdf`. The YAML files under `configs/` transcribe Table I
instead of using the repository's newer demonstration defaults.

Important conversions are `0.1 cm^-2 = 1e-9 um^-2`, `0.05 urad = 5e-8 rad`,
and the paper's bottom/top pad diameters `0.5/0.3 um` are radii `0.25/0.15 um`.
The latest analytical config loader squares the bow-derived magnification sigma;
the paper configs compensate explicitly so that the effective value is the
Table-I `0.01 ppm`.

## Run

```bash
/u1/ee/zhichao/anaconda3/envs/general/bin/python reproduction/run_all.py --jobs 4
```

The full workflow evaluates 56 paper cases, six explicitly labeled Fig. 16
compatibility cases, and two 300-point interaction sweeps. Use `--jobs 1` on a
memory-constrained machine. Rebuild plots, tables, and the historical audit
with `--report-only`.

Main outputs:

- `PROVENANCE.md`: commits, paper hash, and recovered/lost sweep details
- `results/RESULTS.md`: coverage and numeric comparison summary
- `results/paper_comparison.json`: machine-readable comparison metrics
- `results/fig13_correlation.png`, `fig15_w2w_all.png`,
  `fig16_d2w_all.png`, `fig17_layouts.png`, and `fig19_redundancy.png`
- `results/FIG15_16_VERSION_AUDIT.md`: 2025/paper-era/pre-summer/latest comparison
- `results/fig16_d2w_all_effective_fit.png`: calibrated compatibility view;
  this is clearly separated from the provenance-correct Table-I result
- one JSON file per case, containing effective parameters and yields

Self-contained per-figure packages:

- `fig13/`: recovered 300-point legacy sweeps evaluated with current code;
- `fig15/`: all 12 W2W configurations, historical-parameter comparison and PASS audit;
- `fig16/`: all 12 D2W configurations, all-component gate, wafer-scaling audit, and original-spreadsheet references;
- `fig17/`: all four W2W/D2W layouts, current sample-wise results, and source-asserted historical parameter/scaling audits;
- `fig19/`: all 24 W2W/D2W bars including 20:1, plus an optional hash-checked audit of the uploaded original data archive.

## Coverage and limitations

- Figure 13: 300 points each for W2W and D2W, using the rotation-mean, Cu
  dishing-standard-deviation, and particle-density vectors recovered from the
  uploaded December-2025 W2W/D2W `simulator_main.py` files. All yield values are
  recomputed with the current `yap+` calculators; no old output data is reused.
  The stored results were regenerated after the overlay correction at commit
  `7fde3ad`. The self-contained command, exact parameter table, results, and
  validation are under `fig13/`.
- Figures 15 and 16: all twelve published density/pitch/area configurations,
  including Figure 16's 1000 mm2 system-yield calculation. All six fine-pitch
  columns (both densities) are included. For Fig. 16, Table I, the historical
  D2W config, and the historical calculator notebook contain three different
  rotation parameter sets; see the version audit. The optional compatibility
  config in `configs/fig16_right3_effective_fit.yaml` matches the plotted bars
  but is an inverse fit, not claimed recovered author input. The dedicated
  `fig16/` package treats sample-wise overlay as authoritative and gates all
  five quantities. The best historical profile passes 8/12; cases 2/9/11/12
  fail because small single-die residuals are amplified in `Y_sys`.
- The version audit includes 2025-05/08/10 history. October 6 changed scalar
  overlay aggregation while adding pad-level yield; local commit `cc35068`
  restores the earlier, correct sample-wise reduction. The May D2W code also
  scaled rotation/magnification by wafer radius divided by die half-diagonal.
  This makes equal-pitch Fig.16 overlay almost die-size-independent and cannot
  reproduce the paper's area trend; with December rotation inputs it greatly
  over-scales the error.
- Figures 15 and 16 enforce the full layout (100% critical pads). The Table-I
  profiles use the paper's 400 um W2W / 100 um D2W analytical grids; the
  dedicated Fig. 16 historical profiles separately retain the December-2025
  D2W config's 50 um grid. Each case JSON records the effective choice.
- The latest simulator uses independent Poisson particle counts. The
  pre-summer simulator used a fixed-total multinomial allocation; this changes
  simulator variance/cross-unit correlation, but not the Fig. 15/16 analytical
  means, whose defect equations use the paper's Poisson `exp(-Lambda)` model.
- Figure 17: Full, Sparse, Peripheral, and Centralized for both W2W and D2W at
  0.3 um pitch. The main reconstruction now enables the opt-in historical D2W
  wafer/die scale and passes all 8 layouts (RMSE `0.004751`, maximum error
  `0.012880`). A separate raw-current plot keeps the no-scaling result. The
  numerical result passes, but provenance remains conditional because no
  recovered commit contains both the May scaling and later layout boundaries;
  the logical-pair ratio needed for W2W defect bars is also explicitly inferred.
- Figure 19: both bonding modes, both densities, 0/200/400/600/800 um dedicated
  spacing, and 20:1 shared redundancy. The 20:1 case is a compact-block
  approximation because its microscopic mapping is below the 200 um grid. The
  uploaded `replica_distance.zip` recovers exact numeric values for all 24
  published bars and eight 999x10 distribution dictionaries, but not their
  generating Python simulation driver; `fig19/audit_uploaded_archive.py`
  verifies the archive hash, values, and shapes without copying old outputs
  into the current-code cases.

The compact analytical driver constructs block maps directly, avoiding full
10,000-by-10,000 pad bitmaps. Its grid dimensions match the upstream
`downsample_bitmap()` integer trimming rule; incomplete right/bottom edge
blocks are not promoted to artificial full blocks. The legacy aggregate
`paper_comparison.json` still uses approximate raster readbacks; the dedicated
Fig.19 package supersedes those Fig.19 approximations with recovered exact
numeric values.
