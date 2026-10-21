# W2W Assembly Orders

## Process Redirect

The no-thinning W2W branch and its prepared-wafer input/config profiles are
retired. The current process starts with 775-um total-thickness wafers. Every
permanent bond is followed by free anneal and cooldown before any thinning.
Carrier-assisted thinning is a separate operation; the carrier does not
constrain annealing in this model.

The two user-specified process ledgers are available for planning only.
Native execution and scoring are blocked pending orientation conventions,
compatible operators and checked material-history transfers. Historical
F2B-only mechanics or yields cannot be reused for this new process.

## Specified Processes

For `(((1,2),3),4)`:

1. Bond W1-W2 F2F; freely anneal and cool.
2. Thin W2 on a temporary carrier and grow its backside Cu/BI.
3. Bond W2-W3 F2B; freely anneal and cool.
4. Thin W3 and grow its backside Cu/BI.
5. Bond W3-W4 F2B; freely anneal and cool.
6. Thin W4 without growing another backside BI.
7. Attach W4 to a temporary carrier, then thin W1 without another BI.

For `((1,2),(3,4))`:

1. Separately bond W1-W2 and W3-W4 F2F; freely anneal and cool each pair.
2. Thin W2 and W3 and grow their mating backside Cu/BI layers.
3. Bond W2-W3 B2B; freely anneal and cool.
4. Thin the outside W1 and W4 without adding backside BI. The exact final
   carrier attachment sequence is not yet specified for this route.

W4 carrier attachment is temporary: it does not add a fourth permanent bond,
overlay yield, or anneal. Supported target totals include BEOL and all owned
BI layers; all four wafers finish at the requested total thickness.

## Thickness And History

With front BI 1.5 um, BEOL 10 um, and deposited backside BI 1.5 um:

| Target Total | Inside Retained Si | Outside Retained Si |
| --- | --- | --- |
| 30 um | 17 um | 18.5 um |
| 50 um | 37 um | 38.5 um |
| 80 um | 67 um | 68.5 um |

Outside wafers do not receive the additional backside BI. A single retained
Si operator therefore cannot represent both inside and outside wafer endpoints.
The prepared D2W Si17/67 candidates do not provide complete W2W30/80 support.

Original layers receive `[3,3,2,1]` anneals in the sequential route, versus
`[2,2,2,2]` in the binary route. Deposited W2/W3 backside BI receives `[2,1]`
versus `[1,1]`. Retained Cu plastic/creep states, birth coordinates, reference
offsets and material clocks must survive thinning and child-stack joining;
they must not be reset to a virgin wafer or copied from an unrelated route.

The symbolic front/back orientations differ between these two routes,
especially for W3. Whether different orders may have different final wafer
orientations is still awaiting confirmation. No evolved stack is implicitly
flipped in the plan or native solver.

## Plan Input

Use `W2W/input/design_6_4hi_p2/stack.yaml` and its complete modeling config.
`stack.initial_bows_um` contains four fixed signed center-minus-mid-edge bows
for the original 775-um wafers. The example values are illustrative, not
measured manufacturing data. Layer order remains bottom to top.

`assembly_orders: all` lists all five binary trees. Two have defined symbolic
process ledgers; the other three have `NEEDS_PROCESS_DEFINITION` placeholders
with no invented states or events. The overall request is incomplete and no
subset is executed or ranked. A quoted tree or a list of quoted trees selects
specific requests; native execution is still blocked.

```bash
python W2W/calculator_main.py \
  --config W2W/configs/design_6_4hi_p2/design_6_4hi_p2.yaml \
  --mode w2w_modeling --ds_name design_6_4hi_p2 \
  --ds_dir W2W/input/design_6_4hi_p2 --mechanism-filter overlay \
  --plan-only --output-dir /your/approved/results/W2W
```

Planning writes `PLAN.json`, input snapshots and source fingerprints. It
does not load private operators or predict magnification, mechanics or yield.
The v2 plan cannot be scored using an old v1 mechanics archive.

## Initial Bow And Overlay

Changing only initial bow at fixed geometry reuses the reduced operators.
The fabrication-source amplitude and complete natural q6 are recalculated;
subsequent Cu and thermal histories must be recomputed. This initialization
does not infer an initial plastic/creep manufacturing history from bow.

Future mechanics will use released prebond states of both children and the
existing face-strain/curvature-to-magnification method, not `B1-B2` or an
assumption that a carrier-held stack has zero free bow. Physical interface
indices retain the same non-bow translation/rotation draws across orders.
The compact central critical-pad region remains a layout surrogate, not a
replacement for an arbitrary real bump map.

See the [README assembly-order guide](../README.md#assembly-order-experiments)
for runtime setup, supported D2W operators and model limitations.
