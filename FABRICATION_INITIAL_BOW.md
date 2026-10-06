# Fabrication-Induced Initial Bow

`manufacturing_bow.py` models the incoming fabrication state as one isotropic
in-plane eigenstrain amplitude in the effective front-side BEOL layer. The
source is zero in bulk Si and in the bonding-interface layer.

The reference laminate is a 10 x 10 mm die with 50 um of Si, 10 um of BEOL,
and 1.5 um of bonding interface. Thus, 50 um means the Si thickness and the
modeled total thickness is 61.5 um. A full isotropic 6 x 6 ABD system converts
the source resultants into the natural generalized state
`[epsilon_x, epsilon_y, gamma_xy, kappa_x, kappa_y, kappa_xy]`.

The signed-bow convention is center minus mid-edge. For the strain-gradient
curvature convention used by the ABD model,

```text
B_um = -1000 * kappa_per_mm * L_mm^2 / 8.
```

Rao et al.'s less-than-80 um overall-warpage statement is treated as an
upper-end magnitude anchor, not as a measured mean or three-sigma spread. For
the equal-biaxial quadratic reference shape, corner-to-center overall warpage
is twice the center-to-mid-edge bow used by YAP. Therefore the adopted mapping
is

```text
overall upper = 80 um
center-to-mid-edge upper = 0.5 * 80 um = 40 um
abs(mean(B_edge)) + 3 * std(B_edge) = 40 um
```

for the reference geometry at source CV = 10%. The code evaluates source
moments using Gauss-Hermite quadrature, so the same inverse-calibration driver
can call a nonlinear forward model later. The calibrated `A_fab0` is never
recalibrated during the 5/10/15 mm and 20/50/80 um geometry sweep.
The calibration record reports both the positive amplitude magnitude and the
signed nominal `A_fab0`; the baseline layer order uses negative eigenstrain to
produce positive center-minus-mid-edge bow.

All D2D yield case studies use a frozen source CV of 20% while preserving the
original 10%-CV mean-plus-three-sigma bow anchor. In normalized source
coordinates, the reference distribution `N(0.5, 0.05^2)` is therefore mapped
to `N(0.40625, 0.08125^2)`: both distributions have a mean-plus-three-sigma
amplitude of 0.65. Case-study scripts import this policy from
`initial_bow_distribution.py`; changing CV must not silently move the physical
three-sigma warpage anchor.

This follows the inverse-calibration concept in Cai et al., *Journal of
Electronic Packaging* 146(2), 021006 (2024), DOI 10.1115/1.4063919. The
80-um overall-warpage anchor is from Rao et al., IEEE ECTC 2024,
DOI 10.1109/ECTC51529.2024.00019; their paper does not supply the statistical
interpretation. The 10% source CV and the `abs(mean)+3 sigma` population
mapping are stated YAP modeling assumptions. They are not attributed to Rao.

`FabricationBirthBatch` is the integration contract. It draws the source once
per incoming D2W die or W2W wafer and stores an immutable natural `q6` state.
The fabrication resultants are not reapplied during assembly, and the incoming
Cu plastic/creep state remains virgin. W2W birth states are shared by all sites
on a physical wafer; D2W dies receive independent draws.

The source calibration remains marked
`LINEAR_ABD_ONLY_NOT_ANSYS_RUNTIME_QUALIFIED`. The registered
`bi10_tsv10_beol20_p20_h50` full-history backend accepts the complete natural
q6 through `exp12.natural_initial_q6.v2`; it is limited to its checked geometry
and C2/C3 histories. It is not selected automatically for other pitches,
thicknesses, or stack heights.

`history_mechanics_backend.run_mechanics_history` writes an immutable q6 input
snapshot, calls the hash-checked BI10 profile, verifies all history/checkpoint
artifacts, and converts the A1...A3 records into the existing
`history_bow_prebond` yield interface.

The geometry-aware integration uses `MechanicsGeometry` and deliberately calls
the stack input `total_die_thickness_um`. This value includes the retained
TSV/Si, BEOL, and all BI layers owned by one die. For a fixed total die
thickness, only the bottom die has one BI layer; each subsequently attached
die has both back and front BI and therefore uses less TSV/Si. The older BI10 profile instead fixes retained
Si at 50 um, producing 61.5 um outer dies and 63 um interior dies; it remains a
named legacy backend and is not silently treated as a constant-total-thickness
case.

`history_mechanics_backend.resolve_mechanics_package` selects only an exact
registered `(BI pitch, BEOL thickness, TSV pitch, total die thickness)`
package. `load_mechanics_geometry_model` pays package loading and JIT costs
once, after which `predict_mechanics_geometry` can evaluate repeated birth
samples. Missing combinations fail closed; nearest-neighbor substitution and
silent operator scaling are disabled.

Generic geometry runs use the frozen same-orientation tier contract: the
bottom die is `TSV/Si -> BEOL -> front BI`, and every subsequently attached
die is `back BI -> TSV/Si -> BEOL -> front BI`. The earlier OFAT ledger with
reversed upper-die layer order is not used by the YAP bridge.

The loader defaults to `fidelity="native"`. Existing spatially reduced
packages retain all Cu integration points while reducing the layer coordinate
spaces, but they were trained for the superseded tier order. They are labeled
`CANDIDATE_NOT_VALIDATED` and are diagnostic-only; loading one requires both
`fidelity="compact"` and `allow_unqualified_compact=True`.

For corrected-tier `BI6 + BEOL10 + TSV6 + total50`, a same-seed two-die/C1
benchmark measured 39.37 s load plus 178.41 s solve for native coordinates and
13.25 s load plus 26.61 s solve for compact coordinates. The corresponding
center-minus-edge bows were +5.848013 um and +1.903125 um. The 67.46% compact
error fails qualification despite a 5.46x total runtime speedup, so the old
compact basis must not drive yield analysis.

A corrected-tier P48/B32 replacement was then trained from one native C1
accepted-coordinate capture. It retains all Cu integration points and reduces
each tested layer to 135 coordinates. On an unseen source seed, edge-bow error
was 0.00234 um for C1. For the four-die/C3 history, final edge-bow error was
0.00488 um (0.0133%) and total online runtime improved from 1933.64 s to
152.48 s (12.68x). This is a limited `BI6/BEOL10/TSV6/total50`
qualification, not permission to extrapolate the basis to another geometry.
The retained training capture can rebuild the temporary candidate with an
explicit `YAP_WORK_DIR` on approved host-local scratch (see README setup):

```bash
python build_corrected_compact_mechanics.py \
  --capture D2W/output/case_studies/history_mechanics_backend/corrected_native_c1_seed7_coordinates \
  --native-package /path/to/private/operators/BI6_50.json \
  --output "$YAP_WORK_DIR/yap_corrected_p48_b32" --port-rank 48 --bubble-rank 32
```

Layer datasets remain bound to their port basis. The theoretical 67 physical
components for the requested geometry grid are not automatically 67 mutually
composable runtime operators; each basis family must be built and qualified,
or a common cross-geometry basis must first be established.

Run the study from the `YAP` directory:

```bash
python fabrication_initial_bow_study.py
```

Results are written under
`D2W/output/case_studies/fabrication_initial_bow/` as JSON, CSV, and PNG.
