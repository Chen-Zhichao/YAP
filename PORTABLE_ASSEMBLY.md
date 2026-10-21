# Independent Assembly Runs

This page describes the legacy standalone delivery and exporter. The new
13-entry component-v2 interface uses existing canonical operator and evidence
paths on the shared server; it is not included by `--prepared-operators`.
Follow [component runtime setup](README.md#component-runtime-and-catalog) for
that interface. Copying its catalog or native source directory alone does not
create an independent data bundle on another server.

Keep the exported `YAP/runtime/` directory with the code. Runtime bundles
contain private operators and immutable provenance; they do not belong in the
public Git repository. A Git checkout alone does not contain these data.
Installing Python requirements or importing YAP as another project's dependency
does not supply the operators and geometry datasets. On the same shared server,
use the [missing mechanics data instructions](README.md#missing-mechanics-data)
to bind the existing data package; no MAPDL rebuild is needed for its supported
profiles. On another server, request the full standalone bundle, not only the
source checkout or the example resource map.

Use Linux and Python 3.11 or 3.12. From the exported `YAP` directory:

```bash
export YAP_WORK_DIR=/path/to/approved/work-disk/yap-work
mkdir -p "$YAP_WORK_DIR/temp"
export TMPDIR="$YAP_WORK_DIR/temp" TMP="$YAP_WORK_DIR/temp" TEMP="$YAP_WORK_DIR/temp"
export NUMBA_CACHE_DIR="$YAP_WORK_DIR/numba" MPLCONFIGDIR="$YAP_WORK_DIR/matplotlib"
export PIP_CACHE_DIR="$YAP_WORK_DIR/pip" PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX YAP_NATIVE_RUNTIME_DIR YAP_RUNTIME_RESOURCES
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Replace the work-disk placeholder. No outputs, temporary files or caches may
use `/tmp`. Keep large solver scratch on approved host-local storage and
accepted results on your shared project storage.

## D2W

The existing p2 TOTAL50 profile has prefabricated front and back BI layers;
there is no thinning or deposition during D2W assembly:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python D2W/calculator_main.py \
  --config D2W/configs/design_6_4hi_p2/design_6_4hi_p2.yaml \
  --mode d2w_modeling --ds_name design_6_4hi_p2 \
  --ds_dir D2W/input/design_6_4hi_p2 --mechanism-filter overlay
```

`stack.yaml` selects all five fixed-layer-order trees. Initial bows are
illustrative `[10, -10, 20, -20] um`; replace them with the known signed
center-minus-mid-edge bows. Operators are reusable when only these values
change, but every selected tree's Cu and thermal history must be recomputed.
This does not reconstruct a pre-existing Cu manufacturing history from bow.

The optional prepared-operator delivery also contains
`design_6_4hi_p2_t30` and `design_6_4hi_p2_t80`. Both built operators passed
the pinned structural catalog admission. These candidates use native64 ports rather than
the existing TOTAL50 P24 preview. Their numerical checks do not establish
independent physical validation. See the
[prepared-operator instructions](README.md#prepared-total30-and-total80).

Assembly-order mode scores overlay only. Add `--plan-only` to inspect input
and process requests without native data; this does not verify operator loading.
Full runs require at least 40 GiB available memory. Outputs default to
`D2W/output/assembly_orders`; `--output-dir` selects another approved path.

## W2W

The no-thinning W2W branch is retired. The replacement begins with 775-um
wafers, uses F2F pair bonding followed by free anneal/cool, then appropriate
thinning, backside Cu/BI growth, and F2B or B2B joins. Final outside thinning
does not add a backside BI. W4 carrier attachment is temporary and adds no
permanent bonding interface.

The new W2W request is plan-only: the two specified process ledgers are
recorded symbolically; the other three trees need process definitions.
Orientation and checked history transfers remain unresolved, so native
execution and scoring are blocked. Historical W2W yields are not results
for this new process. See [W2W process planning](W2W/ASSEMBLY_ORDER.md).

## Delivery Checks

For a prepared-operator delivery:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python verify_assembly_runtime.py --prepared-operators
```

For a default delivery, use `--load-operators` instead. Verification checks
inventory and supported native operator loading, not full assembly histories.
It denies access to the original project and rejects external resource-map
overrides. Frozen paths are rebased into the delivered runtime without
rewriting manifests or NPZ files. `runtime/BUNDLE.json` inventories the files;
`runtime/RESOURCES.json` binds logical resources to delivered directories.
Missing files, conflicting imports and changed hashes fail closed.

Maintainers export into a new directory, never overwrite an existing delivery.
Copies are paced at 24 MiB/s; independent SHA256 readback is paced at 32 MiB/s:

```bash
python package_assembly_runtime.py \
  --native-source /path/to/confined/runtime/native \
  --resource-map /path/to/private/resources.json \
  --output-dir /your/shared/new-delivery --prepared-operators
```

Prepared export includes D2W TOTAL50/30/80 only. Default export keeps the
historical two-profile inventory, with the current W2W profile usable only
for process planning. Neither mode enables new W2W mechanics.

The prepared delivery published on October 6, 2026 at
`distribution/yap_prepared_p2_20261006_v1/YAP` passed independent inventory and
native-loading checks for all three D2W profiles. It contains 305 files and
about 8.15 GiB of native data; full assembly histories were not rerun.

For shared-server Git checkouts, use the
[README assembly-order guide](README.md#assembly-order-experiments) to bind a
shared runtime instead of duplicating the private data.
