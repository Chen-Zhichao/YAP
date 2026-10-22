# Endpoint Thickness API

```python
from thickness_operators import open_family

family = open_family(nas_family_directory)  # Memory-only derived-result cache.
family.metadata()  # Safe while independent validation is pending.
entry = family.resolve_entry("BI2_SI32")  # Existing numeric ID syntax.
request = family.resolve(prepared_total_um=45.0)
operator = family.evaluate(prepared_total_um=45.0)
si_arrays = operator.component_arrays()
bi_arrays = family.fixed_component("bi")
beol_arrays = family.fixed_component("beol")
topology = family.topology()
physical_geometry = family.geometry(si_um=request.si_um)
```

`resolve` and `evaluate` accept exactly one of `si_um`,
`single_bi_total_um`, or `prepared_total_um`. Any finite Si thickness between
27 and 37 um is eligible; there is no rounding to catalog anchors. A prepared
45 um request means Si32 and single-BI total43.5, not a validated dual-BI45
assembly. Second-BI assembly remains experimental.

Bracket validation and normalized evaluation are manifest-driven: any finite
`0 < lo < hi` has exactly derived center/half-span and canonical endpoint keys.
The first delivered scientific family remains Si27/37 only; synthetic17/27
tests demonstrate runtime reuse, not admission of a second scientific family.

The public runtime requires only this YAP module, NumPy, and the immutable NAS
family directory. It evaluates frozen Chebyshev/Laurent coefficients and
affine coordinate maps. It never imports a private project FE generator,
solves an FE system, builds a target POD, or transfers aged material states.

Absent a bound independent native73 strict pointwise bow PASS, all numerical
entrypoints raise `PendingValidationError`. A held-out Si32 pass activates an
experimental interpolation family, not proof at every thickness, temporal
qualification, finite-die assembly qualification, or W2W thinning admission.

Cache keys bind exact binary thickness, endpoint/reference hashes, frozen
family, geometry, material, runtime source, coefficients and activation.
Arrays are read-only. The default uses only a 2 GiB derived-result memory LRU;
immutable coefficient operands are accounted separately by `cache_info()`.
Disk caching is disabled unless a directory is explicitly supplied, and its
default quota is 8 GiB. It is
explicitly located outside the immutable distribution, bounded without
deleting existing files, and checked on readback. Under this project's storage
policy, use a writable cache beneath the canonical project, never home,
`/tmp`, `/data`, or an archive payload.

Cross-process writes use an exclusive NAS directory lock around quota checking
and publication. A busy lock skips disk caching, not operator evaluation.
A crash-stale lock also skips writes; it is never automatically removed.
Restart Python after updating runtime source: the source pin is captured at
import, so an already-loaded older module cannot impersonate a newer binding.

`component_operators` integration contract: `family.resolve(**request)` supplies the
thickness ledger; `family.evaluate(**request).component_arrays()` supplies the
Si numeric keys already used by the component runtime. The frozen BI/BEOL
pair and topology must come from that same family. This module does not
modify or bypass the existing component catalog loader.

For exact-lookup-first fallback, `family.resolve_entry(identifier)` accepts
`BI2_SI32` or `component:BI2_SI32` and returns a distinct
`yap.parameterized_component_entry.v1` schema. It includes `kind`, `id`,
`family_directory`, `family_sha256`, `cache_key`, single-BI `geometry`,
`experimental_prepared_total_um`, scoped `validation`, `capabilities`, and
`runtime_strategy`. It is not an extra fixed-catalog entry. The equivalent
one-shot call is `resolve_entry(identifier, distribution=family_directory)`.

The Si union has rank42. Its bridge must use the original SparseModel/JVP
with the reference preconditioner, as used by the completed virgin73 replay.
Do not pass this family through the old LocalCachedTangentModel `n<=32`
adapter. The API accepts reduced ranks through48 without importing or
silently selecting that cached adapter.

## Delivered Family And Evidence

The activated delivery is
`/app/nanocad/projects/personal/zchen/YAP_STACK/distribution/operators/thickness_family`.
Its independent Si32 single-BI graded73 comparison passed all 292 observations
across 73 states: maximum relative bow error 0.3029311013%, maximum absolute
error 1.3821059951 um. Each observation passed the strict relative <10% OR
absolute <1 um criterion. Native history took 2047.952 s; original-JVP reduced
history took 1300.895 s. These are spatial results at the shared graded clock,
not independent temporal or prepared-assembly qualification.

Measured on sysml, CPU127, one BLAS thread, 48 GiB address-space limit:

| Public operation | Seconds |
| --- | ---: |
| Si32 initial evaluation including operand loading | 36.502 |
| Si32.4 evaluation with operands already loaded | 2.391 |
| Separate fresh-process Si32.4 initial evaluation | 4.771 |
| Si32.4 reevaluation after clearing only the result LRU | 4.376 |
| Si32.4 result-memory hit | 0.010 |

"Fresh process" means no loaded coefficient operands; the shared OS page
cache was not evicted. Timings are individual observations, not guaranteed
latencies or an assembly-history benchmark. Separate Si32.4 recomputation
was bitwise identical for every returned array, retained rank42, and refused
Si26.999/37.001. It does not independently qualify off-midpoint bow. No FE
build, target POD, disk-cache write or thermal-history replay was performed
by these API timing jobs.

Evidence under the canonical project:

- `operator_repair_20261015/development/trial_forcing_closure_v1/continuous_cu_screen_v1/runtime/THICKNESS_PARAM_PUBLIC_ACTIVE_BENCH_20261021_v1/REPORT.json`
- `operator_repair_20261015/development/trial_forcing_closure_v1/continuous_cu_screen_v1/runtime/THICKNESS_PARAM_PUBLIC_OFFMID_TIMING_20261021_v1/REPORT.json`
- `operator_repair_20261015/development/trial_forcing_closure_v1/continuous_cu_screen_v1/runtime/THICKNESS_PUBLIC_API_TESTS_20261021_v10.xml`: 47 tests passed.

The frozen public module SHA256 remains
`4aa29da06effc4bba67c4367837654289eead063d16bcc0c684114aca5c11a93`.

The separate public component-loader smoke test loaded `component:BI2_SI32`
as an experimental prepared45 two-die library in 64.778 s, including 14.067 s
of material-kernel JIT warmup. Registration covered 8 modules, 16 faces and
7 shared interfaces; maximum face error was 3.1554e-14 mm against a 1e-12 mm
tolerance. Evidence is
`operator_night_20261021/public_parametric_bridge_smoke/REPORT.json`.
That report records zero Cu-history updates: library loading/registration
is not a completed physical assembly-history run or prepared45 accuracy proof.

The normal public prepared45 two-die C1 calculator completed in 294.881 s
including loading. It proves runtime integration, not independent prepared
assembly accuracy. Evidence is `operator_night_20261021/parametric_pair_public_run`.
The independent single-BI bow pass and original-JVP receipts remain unchanged.

## F2B W2W Integration

`component_operators.w2w_operator_request` resolves the same exact-catalog or
parameterized Si geometry as D2W. `load_w2w_component_library` adds separately
checked homogeneous pure-Si layers to represent the original 775-um wafer.
It preserves the Cu operators and does not extend their validation scope.
The Cu-family API above still performs no FE solve; a missing homogeneous-Si
cap seed or exact fractional remainder can require a separate linear FE
construction at first use. Those small condensed Si matrices are cached,
not confused with a new Cu-operator training or interpolation admission.
See the README's F2B-only W2W commands and cache paths.

The later private tiled-cache73 benchmark completed with 72 strict intervals,
zero cutbacks and zero residual-fallback acceptances. Absolute elapsed time
was 787.568 s on sysml (AMD EPYC 9554P; eight physical cores); recorded accepted
intervals totaled 771.561 s. The remaining 16.007 s includes setup, checkpoint
writes and bookkeeping, not a separately measured preload phase. All 268
unchanged true-JVP linear certificates passed with no original-backend fallback.
All 292 comparisons against the existing native73 reference passed, with
maximum relative error 0.3029311013% and absolute error 1.3821059951 um.

Against the retained original-JVP final checkpoint, the largest Cu-state
absolute difference was 1.069e-15 (largest per-field relative L2 difference
4.553e-14); all birth fields were bitwise equal. State channels were compared
directly without introducing a new error acceptance threshold. Evidence:
`operator_repair_20261015/development/trial_forcing_closure_v1/continuous_cu_screen_v1/runtime/THICKNESS_PARAM_LEDGER_CACHE_CONTROL_20261021_v1/ABSOLUTE_TIMING_AND_STATE_PARITY.json`.

The original 1300.895 s run was on mfd, so its ratio to this sysml observation
is not a controlled algorithmic speedup. A fresh same-sysml original-JVP73
timing run is now pending, using the identical candidate and existing native
reference. Its control is `THICKNESS_PARAM_PAIRED_JVP_CONTROL_20261021_v1`
under the same runtime directory. The public frozen API and activation were
not changed by either benchmark.
