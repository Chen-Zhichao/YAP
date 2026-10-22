# W2W Assembly Orders

Use the [README](../README.md) for setup, design generation and calculator
commands. Set `--bonding-mode w2w` when generating a design.

The current executable workflow uses fixed bottom-to-top layer order and
F2B-only bonds. Wafers start at 775 um total thickness. Before each bond, the
lower child's exposed backside is thinned and prepared with a new BI; the
joined stack then undergoes free anneal and cooldown. Temporary carriers do
not constrain annealing in this model. Final outer-back thinning adds no BI.
Historical F2F/B2B process ledgers are not executable in this workflow.

Edit the generated `input/<name>/stack.yaml`:
- `stack.initial_bows_um`: one fixed signed center-minus-mid-edge bow per
  original 775-um wafer, ordered bottom to top, in um.
- `stack.assembly_orders`: `all`, one quoted tree, or a list of quoted trees.
- `stack.geometry.target_total_thickness_um`: target total thickness,
  including BEOL and the owned BI layers; regenerate geometry files together.
- `stack.process`: thermal ramp, dwell, cooldown and temperatures.
- `mechanics.order_workers`: 1 for serial execution, or a larger concurrency.

Placement, rotation and `MAGNIFICATION_CORRECTION` belong in the full
calculator YAML's `w2w_modeling` section. The same physical interface retains
its non-bow overlay inputs across different assembly orders.

Changing initial bows reuses the operators but requires new mechanics
histories. The prebond comparison uses released mechanical states and the
face-strain/curvature-to-magnification mapping, not a scalar `B1-B2` subtraction.

The current component catalog also serves D2W. Intermediate total thicknesses
40..50 um use the BI2 thickness family. Unit-component bow validation does
not establish independent whole-assembly accuracy; assembly yields remain
experimental. See the README for the currently enabled geometries.
