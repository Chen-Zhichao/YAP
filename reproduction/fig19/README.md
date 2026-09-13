# Figure 19: W2W/D2W Redundant-Replica Reproduction

This directory evaluates all 24 bars in Figure 19 with the current `yap+`
defect calculators: W2W and D2W, particle densities `1` and `0.1 cm^-2`, and
six redundancy configurations consisting of no redundancy, 200/400/600/800 µm
1:1 dedicated redundancy, and 20:1 shared redundancy. Legacy NPY/MAT files and
legacy images are not presented as new results.

## Status: NOT REPRODUCED WITH THE CURRENTLY RECOVERED INPUTS

The per-bar absolute-error threshold is 0.03. The reference values were first
estimated from the paper raster; the MATLAB source in the subsequently uploaded
`replica_distance.zip` recovered exact values for all 24 bars, and the formal
gate now uses those values. Complete results are in `results/summary.json` and
the plot is in `results/fig19_redundancy_all.png` and `.pdf`. Black ticks show
paper values and colored bars show current-code outputs.

| Gate | Pass | RMSE | Maximum absolute error |
|---|---:|---:|---:|
| W2W | 6/12 | 0.0754 | 0.1713 |
| D2W | 6/12 | 0.1516 | 0.3445 |
| Overall | 12/24 | 0.1198 | 0.3445 |

All six W2W bars at `D_t=0.1 cm^-2` pass, while all six at `D_t=1 cm^-2`
fail. For D2W at `D_t=1 cm^-2`, only 600 and 800 µm pass; at
`D_t=0.1 cm^-2`, 200/400/600/800 µm pass. These are individual-bar outcomes
and do not mean that the full figure passes.

The formal configuration uses the paper's explicit 200×200 µm pad blocks and
100% redundant pads, together with Table-I baseline parameters not overridden
by Figure 19. Table I gives a 10×10 mm die and states that its baselines apply
to later experiments unless noted otherwise. Figure 19 does not state a die
size override; Figure 18 immediately before it explicitly compares 10×10 mm
and 3.2×3.2 mm. The formal gate therefore uses 10×10 mm rather than treating a
fitted die size as a paper input.

## Exact data corrected an earlier diagnostic

For a fixed bitmap, the current analytical defect model satisfies
`Y_df = exp(-Lambda)`, with `Lambda` proportional to particle density. The two
density groups must therefore obey:

```text
Y_df(D_t = 1 cm^-2) = Y_df(D_t = 0.1 cm^-2)^10
```

The verifier checks this identity for every current-code column, with residuals
below `1e-11`. An earlier raster readback placed the no-redundancy D2W bar at
`D_t=0.1` near `0.94`, creating an apparent violation. The recovered exact
value is `0.9646`. The paper's maximum D2W identity residual is only `6.3e-5`;
the W2W residual is `0.00222` after four-decimal rounding. There is therefore
no evidence that the density labels or Poisson scaling are incorrect.

The unresolved issue is geometry/critical-area provenance. The exact D2W
no-redundancy value `0.6974` corresponds to `-ln(Y)=0.3604`, whereas current
10×10 mm Table-I geometry gives `-ln(Y)=1.0414`, or `Y=0.35294`. The uploaded
archive contains 999×10 distributions at both 10 mm and 3.2 mm but not the
Python driver that generated the NPY files; the bar-chart MATLAB file hardcodes
the final values. Until the original generator and bitmap are recovered, a
fitted die size or critical-area scale cannot be represented as a paper input.

Additional current-code checks rule out a die-size-only explanation. With a
6×6 mm D2W die, the no-redundancy value becomes `0.69145`, but the
200/400/600/800 µm values are `0.95584/0.99799/0.99887/0.99979`, still unlike
the paper's `0.88782/0.93712/0.99756/0.99879`. With a 9.5×9.5 mm W2W die, the
no-redundancy result is `0.32616`, close to paper `0.332`, while the four
spacing cases remain only `0.35277/0.47469/0.53807/0.64250`. Restoring the
legacy modeling pitch from 1 µm to 10 µm changes results by only about `3e-4`.

The December 2025 and current defect die-yield equations are materially the
same. The best-supported remaining gap is the dedicated-redundancy bitmap/
generator semantics and the exact run seed/driver. The reproduction initially
used a compact perfect-matching map so that current code could run
deterministically. The paper instead describes random main-block placement
followed by replica-candidate search at a specified Euclidean distance. These
cannot be assumed to be the same historical layout.

## Legacy and corrected block-placement audit

`audit_legacy_placement.py` reproduces the legacy random placement line by line
and identifies a local/global ID error. KDTree returns an index into
`redundant_pad_blocks`, but the old implementation tested that local index
against used/available sets containing global block IDs and only afterward
mapped it to a global copy ID. With `assign_pad_blocks('center')`, even a 100%
redundant layout uses mesh-first rather than natural-ID order. The mismatch can
reuse a real physical block while still reporting `N/2` pairs, violating the
paper's non-overlapping half-main/half-replica placement. Correcting only the
ID lookup exposes a second problem: randomized greedy selection commonly
strands the final blocks and cannot form all `N/2` pairs.

Current commit `7fde3ad` fixes both W2W and D2W generators with randomized
maximum-cardinality matching over the requested distance graph. It guarantees
that each physical block appears in at most one pair, avoids greedy stranding,
and supports the 0%-critical/100%-redundant Figure 19 path. Tests verify complete
1250-pair matchings on the 50×50 Figure 19 grid at 1, 2, 3, and 4 block
spacings. This is now a structurally correct current placement method, but it
does not prove that the lost paper run used the same matching realization or
seed, so the historical provenance gap remains.

Feeding the audited legacy placement with fixed seed `20260120` into the
current defect calculators shows that the old W2W modeling default
`redundant_logical_pad_ratio=0.3` is closer to the paper than the textual 0.5
ratio. At the 10 mm baseline, the four spacings have mean/maximum absolute
errors `0.0465/0.0611` for ratio 0.3, worsening to `0.1064/0.1453` at 0.5. The
December 2025 D2W default is `0.005`, not 0.3. A separate D2W ratio-0.3 test
gives `0.0173/0.0420`, ratio 0.5 gives `0.0423/0.1032`, and the default 0.005
gives `0.0433/0.1086`. W2W therefore has direct evidence for a legacy 0.3
default, while D2W would require an unrecovered Figure 19 override. That
inference is not presented as established configuration provenance.

The die-size diagnostic has a similarly clear limit. A fitted 9.5 mm W2W die
combined with legacy placement and ratio 0.3 puts all four dedicated cases
within 0.03, with maximum error `0.0259`, but the paper does not report 9.5 mm.
A fitted 6 mm D2W die does not jointly recover the spacing cases and leaves the
200 µm case off by `0.0884`. A wrong die size is therefore not a common W2W/D2W
explanation, and the formal configuration retains the documented 10 mm value.

## Audit of the uploaded historical archive

`audit_uploaded_archive.py` treats the external zip only as provenance, never
as a current result. Its SHA-256 must equal
`50a128d6b1c273e292a7bc09d37c522919d83f500381c3331446ed4f2efb6c2b`; the 24
MATLAB values must match `config.yaml` exactly; and each of the eight NPY
dictionaries must have shape `999×10`. The audit result is written to
`results/historical_source_audit.json`. The archived notebook performs only an
NPY-to-MAT conversion and cannot recover the missing generation process.

The eight NPY files contain W2W Figure-18-style random-layout distributions.
Their density axes are `logspace(-2,-0.25,10)` for the 10 mm die and
`logspace(-2,0.65,10)` for the 3.2 mm die. Each sample follows a consistent
`exp(-lambda*D)` curve across density, but these grids do not directly contain
Figure 19's `D=1` points. Their mean, median, or extrapolation also does not
produce the exact 24 hardcoded bar values. They validate random-layout trends
but do not replace the missing Figure 19 driver.

## Treatment and version audit of the 20:1 case

The uploaded December 2025 source contains `multi2one_flag` and
`multi2one_ratio`; the W2W modeling configuration retains
`multi2one_ratio: 20`. Its flag is false by default. The D2W default ratio is
80, and no recovered notebook statement overrides it to 20. The archive proves
that the generator once supported multi-to-one mapping, but it does not by
itself prove the D2W run configuration used for the published 20:1 bar.

Git history locates removal of the multi-to-one branch at commit `badf7a9`
(2026-08-28, `Fixed bugs on 8/28 by Zhichao`). That commit simultaneously
rewrote the one-to-one non-overlap/count checks, but neither its message nor
the code documents a scientific reason for removing multi-to-one behavior. No
stronger motivation is inferred here.

In the paper and legacy configuration, main and replica pads in a shared group
are separated by two pitches: only 2 µm at 1 µm pitch, far below the 200 µm
modeling pixel used for Figure 19. In the defect critical-area calculator, 20
main pads and their shared replica therefore occupy one block and can be
disabled by the same large defect. The `shared20` case evaluates this 200 µm
grid limit with the current calculator and should remain close to the
no-redundancy column. Its JSON explicitly records
`shared_main_to_replica_ratio: 20` and surrogate provenance; it is not labeled
as a native full-resolution mapping from the current generator.

## Rerun

Follow the environment setup in `reproduction/README.md`, then run from the
repository root:

```bash
python reproduction/run_all.py --figures 19 --jobs 4
```

The workflow regenerates 24 case JSON files, summarizes errors, creates the
plot, and verifies coverage, calculator-source fingerprint, parameters, the
Poisson density identity, and the declared scientific status. The external
historical archive is not required. If it is available and its provenance is
to be re-audited, run:

```bash
python reproduction/fig19/audit_uploaded_archive.py \
  --archive /path/to/replica_distance.zip
```

## Files

- `config.yaml`: all cases, exact recovered paper values, paper parameters,
  and 20:1 provenance.
- `run_cases.py`: invokes the current `reproduction/model_worker.py` and defect
  calculators.
- `make_plot.py`: plots all W2W/D2W cases and errors, with yield maximum 1.0.
- `verify.py`: checks 24/24 coverage, parameters, errors, the Poisson identity,
  and the declared result status.
- `audit_uploaded_archive.py`: optional read-only audit of the external zip
  hash, exact bar values, and NPY shapes.
- `audit_legacy_placement.py`: audits legacy random main/replica placement, the
  ID bug, and die-size sensitivity.
- `run_legacy_placement_audit.py`: evaluates legacy-placement bitmaps with the
  current defect calculator and compares legacy ratio 0.3 with paper-text 0.5.
- `results/summary.json`: machine-readable summary.
- `results/historical_source_audit.json`: read-only provenance record for the
  uploaded archive.
- `results/legacy_placement_audit.json`: placement statistics for 10 mm and
  3.2 mm dies, four spacings, and 1000 seeds.
- `results/{w2w,d2w}_{d1,d01}_*.json`: current-code results for each case.
