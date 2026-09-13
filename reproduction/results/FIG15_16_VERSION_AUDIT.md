# Fig. 15/16 fine-pitch and parameter-history audit

Scope: all six pitch=0.3 um columns.  The actual rightmost three are density
0.01 cm^-2, not 0.1 cm^-2, with areas 10/50/100 mm2.

| D2W overlay source | 10 | 50 | 100 |
|---|---:|---:|---:|
| Paper plot readback | 0.9560 | 0.9390 | 0.9200 |
| Table I + latest equations | 0.9678 | 0.9678 | 0.9678 |
| Table I + May--Sep. 2025 corner aggregation | 0.9674 | 0.9669 | 0.9665 |
| Pre-summer `d2w_modeling` config | 0.9639 | 0.9618 | 0.9593 |
| 2025 config + May--Sep. corner aggregation | 0.9575 | 0.9469 | 0.9373 |
| Pre-summer calculator notebook overrides | 0.9627 | 0.9557 | 0.9467 |
| 2025 notebook + May--Sep. aggregation | 0.9497 | 0.9239 | 0.8979 |
| 2025 legacy + fitted rotation mean only | 0.9541 | 0.9374 | 0.9213 |
| Effective-input inverse fit (diagnostic only) | 0.9554 | 0.9400 | 0.9196 |

The independent 500,000-sample verification gives complete rightmost-three
`(Yovl, Ycr, Ydf, YD2W)` rows:

- 10 mm2: `(0.955579, 0.994572, 0.99891, 0.949357)`
- 50 mm2: `(0.940313, 0.97315, 0.994779, 0.910288)`
- 100 mm2: `(0.920069, 0.94702, 0.989504, 0.862178)`

Its maximum component error against the approximate plot readback is
`0.0044`.

The 2025 history reveals a direct, material contributor to the D2W overlay discrepancy.
Commit `742f4cc` (2025-10-06), whose stated purpose was D2W pad-level yield,
changed the die-level reduction from `mean(yield(max(corner misalignment)))`
to `min(mean(yield(corner misalignment)))`.  These operations are not
equivalent.  With the 2025 D2W config inputs, the pre-October implementation
is much closer to the plotted area dependence, while the current implementation
stays too high.  It does not by itself explain the complete gap at 50/100 mm2,
and using Table I rather than the committed D2W config largely removes this
effect.  Density does not enter overlay, so the same issue applies to both
fine-pitch density groups.

The analogous W2W change was made in commit `29f7b95` on the same date.  With
Table-I inputs, current versus pre-October W2W overlay for 10/50/100 mm2 is
`(0.963163, 0.963132, 0.963056)` versus
`(0.962869, 0.962469, 0.962115)`;
the paper bars are approximately `(0.96, 0.96, 0.96)`.  This confirms
that the aggregation change is also relevant to Fig. 15, though its numerical
effect is smaller than the D2W discrepancy.

## Particle-count distribution

The paper explicitly uses the Poisson yield model in Eqs. 23 and 29, and both
the pre-summer and latest analytical calculators evaluate `exp(-Lambda)`.
Therefore changing simulator particle generation does not alter the Fig. 15/16
analytical mean values.

The simulator did change in commit `badf7a9`: the pre-summer code fixed one
total particle count and distributed it with a multinomial draw, while the
latest code draws independent Poisson counts per wafer/die.  Both have the same
per-unit mean.  The old marginal variance is `(1-1/M)` times the Poisson
variance and it introduces covariance `-lambda/M` between units.  With 10,000
D2W dies this is negligible (factor 0.9999); with 10 W2W wafers the old variance
is 10% low (factor 0.9), so this can affect Fig. 13 simulation dispersion and
correlation.  The new implementation is the one consistent with an independent
spatial Poisson process.

## Critical-pad fraction and pad-block size

The paper says Fig. 15/16 uses a layout composed exclusively of critical pads,
and Table I plus the Fig. 12 discussion set the analytical grids to 400 um for
W2W and 100 um for D2W.  This reproduction therefore enforces critical/
redundant/dummy = 1/0/0 and those grid dimensions in all 24 configuration rows
(12 per figure).
The derived block sizes are 400 and 1333 pads for W2W at pitch 1 and 0.3 um,
and 100 and 333 pads for D2W; the 0.3 um effective dimensions are 399.9 and
99.9 um because the repository intentionally uses integer truncation.

This override is necessary for current/pre-summer defaults.  However, the
August 28 through October 9, 2025 W2W configs still used 1.0/0.0
critical/redundant, matching the paper; commit `b6f9651` on October 21 changed
the W2W default to 0.9/0.1.  Pre-summer `w2w_modeling` therefore defaults to 0.9/0.1
critical/redundant, while its `d2w_modeling` uses 1/0 but a 400 um grid.  The
latest defaults are 0.2/0.5 for W2W and 1/0 for D2W, both with 400 um grids.
The historical calculator notebooks load those modeling sections and do not
contain an explicit critical-ratio or block-dimension correction.  Thus the
committed default configs are not reliable Fig. 15/16 provenance; the paper's
explicit experiment settings take precedence.

The published Table I says rotation mean/std = 5e-8/1e-8 rad.  In both the
paper-era commit `b6f9651` and pre-summer commit `6e80bc8`, D2W
`d2w_modeling` instead contains 1e-6/5e-7 rad; `calculator_main.ipynb` then
overwrites the mean with 2e-6 rad and uses a top/bottom radius ratio of 2/3
instead of Table I's 0.6.  The latest branch retains those D2W values.  They
explain part, but not all, of the plotted area dependence.

The historical `simulator_main.ipynb` was also inspected.  Its committed D2W
cell overrides rotation mean to 1e-7 rad and is organized around
model-versus-simulation/correlation checks; it does not contain the Fig. 15/16
case-study MAT save paths.  Those save paths occur in `calculator_main.ipynb`,
so the calculator notebook is the stronger provenance source for these bars.

The old notebooks' `<1 um` branch expected
`pad_bitmap/bitmap_collection.npy`. That per-side file is absent from the
stable 2025 commits. A transitional commit (`0a209f2`) does preserve a root
bitmap: a full-critical 1000x1000 pad array with a 25x25 block map and 40-pad
blocks, corresponding to the 10 mm die / 10 um pitch / 400 um-grid baseline.
It is not the missing set of fine-pitch bitmaps for all three die sizes.

The August-2026 contact-overlap rewrite is not the cause: the old fsolve limit
is `0.0702994152293` um and the new brentq limit is `0.0702994152293`
um (difference `3.29e-14` um).  The rewrite replaces a
singular, warning-prone root solve with a bracketed analytic circle-overlap
solve while preserving the result.

The latest code also fixes an old D2W radius-loader division typo, adds fixed
random seeds, keys dilation reuse to layout-defining parameters, and rejects a
full-resolution bitmap whose shape does not match the active configuration.
Those changes are consistent with correctness/reproducibility fixes in commit
`badf7a9` ("Fixed bugs on 8/28 by Zhichao"), not a deliberate change to the
overlay physics.  The historical `<1 um` notebook path loaded a saved bitmap;
the new shape check now exposes incompatible cached bitmaps instead of silently
using them.

Matching the three plot bars by changing rotation and translation spread would
require effective rotation mean/std of approximately
`3.033e-06`/`2.983e-08` rad and translation std
`0.02138` um.
That is a diagnostic identifiability result, not recovered provenance: those
numbers do not occur in the inspected git history.  Therefore the remaining
gap should be reported as a paper/config provenance inconsistency unless the
original MAT/NPY inputs or the exact notebook execution state can be recovered.

A more historically constrained diagnostic keeps every August-2025 config
value and the pre-October aggregation, changing only rotation mean. Its best
mean is `1.45606e-06` rad and produces
`(0.954112, 0.93737, 0.921344)`. This is
not a recovered parameter, but it lies between the committed August config
(`1e-6`) and October notebook (`2e-6`).
