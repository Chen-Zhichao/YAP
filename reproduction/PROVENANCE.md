# Reproduction provenance

- Repository: `https://github.com/Chen-Zhichao/YAP.git`
- Branch: `yap+`
- Upstream base: `ed0449becd4cbc517dc27bda1bd5e8b762dfc6d9`
  (2026-09-02)
- Evaluated local commits: `cc35068c64d354a7413f38705cb8d2bdc625a261`
  (restores sample-wise worst-corner scalar overlay yield), `1d4772b`
  (restores the historical D2W wafer/die distortion scale as an explicit,
  default-off experiment option), and `7fde3ad` (corrects W2W/D2W redundant
  block placement with non-overlapping maximum-cardinality matching)
- Exact pre-summer checkpoint: `6e80bc8998ad2d589d81d9ef45a190d76b9dc733`
  (2026-04-02; the next YAP+ commits are dated 2026-08-28)
- 2025 checkpoints inspected: `6cc2408` (2025-05-30), `0a209f2` and
  `331ac05` (2025-08-28), `742f4cc`/`29f7b95` (2025-10-06), `a5ae68a`
  (2025-10-09), and `b6f9651` (2025-10-21)
- Paper file: `YAP_plus_paper.pdf`
- Paper SHA-256:
  `17c15e8fc37cc8131ed4c98d7dbc1e277c5426901255f0fa7097ceb6d5f9762b`

The git-history-only pre-summer `D2W/simulator_main.py` contains a 300-point
rotation-only experiment whose vector does not reproduce the Fig. 13 range.
The subsequently uploaded December-2025 working copy recovers the paired W2W
and D2W 300-point drivers, including their rotation-mean, Cu-dishing-standard-
deviation, and particle-density vectors. `fig13/config.yaml` transcribes those
vectors exactly, and `fig13/run_sweep.py` evaluates them with the current
`yap+` calculators. No legacy result data or legacy engine is copied into the
reproduction package.

The latest analytical loader derives magnification sigma as
`(k_mag * bow_sigma)^2 / 1e6`, while the simulator uses
`k_mag * bow_sigma / 1e6`. The paper YAML files target the Table-I analytical
value of 0.01 ppm and retain this discrepancy explicitly.

The upstream dilation cache filename does not encode layout, pitch, die size,
or source revision. `model_worker.py` invalidates that case-local cache before
every run so a changed configuration cannot silently reuse stale geometry.

The compact reproduction uses floor-sized block grids, matching upstream
`downsample_bitmap()`, which trims incomplete edge blocks before pooling.  An
earlier reproduction revision used ceil-sized grids and thereby added a full
artificial edge block.  For Fig. 15 column 9 this depressed `Ydf` from
`0.884234` to `0.876499`; the corrected `YW2W` is `0.807260`.

## Fig. 15/16 fine-pitch audit

The most consequential historical difference occurs before the nominal
paper-era/pre-summer checkpoints. Through September 2025, D2W first took the
maximum four-corner systematic misalignment independently for every Monte
Carlo sample, then evaluated and averaged its conditional yield. Commit
`742f4cc` (2025-10-06), while adding pad-level D2W yield, changed this to
computing the mean yield of each corner separately and taking the minimum of
those four means. In notation, it changed `E[min_i g_i(S)]` into
`min_i E[g_i(S)]`; the second quantity is systematically no lower and loses
much of the chiplet-area dependence. Commit `29f7b95` made the analogous W2W
change. The pad/corner maps do not require changing scalar aggregation; this
scalar semantic change was a regression. Local commit `cc35068` restores the
paper-era order while retaining the pad-level marginal maps.

Using the August-2025 D2W modeling inputs with the restored aggregation
gives fine-pitch overlay yields `0.9575/0.9469/0.9373` for 10/50/100 mm2,
versus paper readbacks `0.956/0.939/0.920`. Thus it explains a large part,
though not all,
of the Fig. 16 gap. Keeping those historical inputs and fitting only rotation
mean gives `1.456e-6 rad` and `0.9541/0.9374/0.9213`; this is diagnostic, not
recovered provenance, but the fitted mean lies between the committed 2025
config (`1e-6`) and later notebook (`2e-6`).

The paper's Table I gives rotation mean/std `5e-8/1e-8 rad`. In the paper-era
commit `b6f9651` and the pre-summer commit `6e80bc8`, D2W `d2w_modeling`
instead gives `1e-6/5e-7 rad`, while `D2W/calculator_main.ipynb` overwrites the
mean with `2e-6 rad` and uses top/bottom radius ratio `2/3`. The latest branch
retains the latter config/notebook values. None fully reproduces the rightmost
three Fig. 16 bars; details and numerical comparisons are in
`results/FIG15_16_VERSION_AUDIT.md`.

All five published quantities are now gated. The best recovered profile is the
December-2025 modeling config: 8/12 cases pass at 0.015 absolute tolerance,
with RMSE `0.009217` and maximum error `0.043118`; failures are cases
2/9/11/12 and are dominated by `Y_sys`. Fig. 16 is therefore classified NOT
PASS rather than hiding overlay-dependent quantities from the gate.

The May-2025 D2W code also multiplied rotation and magnification samples by
`WAF_R / hypot(DIE_W/2, DIE_L/2)`. A dedicated source-asserted audit shows that
Table-I inputs then give the same fine-pitch overlay (`0.928426`) for all three
die areas, while the paper gives `0.956360/0.939716/0.921729`. December-2025
rotation inputs become drastically over-scaled. This historical factor can
help explain Fig.17's fixed-die offset but cannot explain Fig.16's area trend.

The historical `D2W/simulator_main.ipynb` was checked as well. Its stored
experiment overrides rotation mean to `1e-7 rad` and contains correlation/
model-versus-simulation work, while the Fig. 15/16 case-study MAT save paths
are in `calculator_main.ipynb`. It is therefore useful supporting context but
not the strongest surviving source for the case-study bars.

The 2025 config history also resolves the critical-pad question. W2W modeling
used 100% critical and 0% redundant pads from `331ac05` through `a5ae68a`,
matching the paper. `b6f9651` changed that default to 90%/10%; the later
pre-summer checkpoint inherited it. D2W modeling remained 100%/0%, but its
committed grid was 400 um rather than the paper's 100 um.

For pitch below 1 um, the notebooks load a saved `bitmap_collection.npy`.
That per-side file is missing from stable 2025 commits. Transitional commit
`0a209f2` contains one root bitmap with a 1000x1000 full-critical pad map,
25x25 blocks, and a 40-pad block size—consistent with the 10 mm / 10 um pitch /
400 um-grid baseline, not all Fig. 15/16 fine-pitch areas. The exact notebook
execution-state bitmaps therefore remain unrecovered.

The August-2026 commit `badf7a9` changed the contact-area threshold solver from
SymPy/fsolve to an analytic circle-overlap function with a bracketed root
solve. The audited limits agree to `3.3e-14 um`, so this is not the source of
the plot discrepancy. The same commit fixes the D2W pad-radius loader's divide
versus multiply typo, introduces deterministic seeds and layout-aware cache
invalidation, and rejects incompatible full-resolution bitmap shapes. These
are correctness/reproducibility changes. The old `<1 um` notebook branch loaded
an existing bitmap without proving it matched the active configuration; the
new validation deliberately exposes that stale-state problem.

`configs/fig16_right3_effective_fit.yaml` is retained only as a diagnostic
compatibility configuration. Its parameters were fitted to approximate raster
readbacks and are not present in git history or asserted to be the original
paper inputs.

## Fig. 17 overlay compatibility audit

`fig17/legacy_overlay.py` reconstructs the scalar overlay implementations from
commits `6cc2408` (May 2025) and `331ac05` (August 2025), and asserts the
defining source expressions with `git show` before evaluating them. Both old
implementations first take the maximum systematic corner displacement for each
Monte Carlo sample and then apply one scalar random-error CDF. Local commit
`cc35068` now uses that same order. With identical August-2025 parameters,
current and historical overlay values differ by at most about `3.6e-5`,
consistent with Monte Carlo seed differences.

All eight W2W/D2W layout overlay bars from each historical revision are within
0.02 of Fig. 17. The August implementation also reproduces the paper's D2W
centralized-minus-full overlay trend to 0.00342. Substituting that traceable old
overlay into current non-overlay calculations gives an 8/8 full-component
compatibility result (maximum error 0.01771), provided the separately labeled
logical-mapping sensitivity profile is used. Its W2W logical ratio of 0.05 was
inferred from the plotted defect yield and is not claimed as recovered author
input. A second sensitivity combines the May wafer/die scale with the August
critical-layout boundaries: its four D2W overlay errors are all below 0.02,
but no recovered commit contains both behaviors, so it is not presented as a
historical fact. These limitations remain explicit in every result and in
`fig17/README.md`.

The same combination is now also executed through the current D2W calculator's
explicit `scale_systematic_distortion_from_wafer` option, rather than only by
the audit re-expression. Together with the labeled logical-mapping sensitivity,
all eight Fig.17 W2W/D2W layouts pass at 0.02 tolerance (RMSE `0.004751`, max
error `0.012880`). Scaling remains default-off, because enabling it globally
would erase the die-area trend required by Fig.16.

## Fig. 19 recovered data audit

The user-uploaded `YAPlus_1204/replica_distance.zip` has SHA-256
`50a128d6b1c273e292a7bc09d37c522919d83f500381c3331446ed4f2efb6c2b`.
Its MATLAB script `w2w_replica_dist_20_40_60_80.m` contains exact numeric arrays
for all 24 published Fig. 19 bars. The archive also contains eight NPY
dictionaries, each with 999 samples by 10 densities, for 3.2 mm and 10 mm dies
at four replica distances. The included notebook only converts those NPY files
to MAT; the Python driver that generated them is absent. The dedicated audit
script checks the archive hash, parses the MATLAB values, and validates all
array shapes without copying the historical arrays into current-code results.
