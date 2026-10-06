# Independent P2 Assembly Runs

The exported folder contains `YAP/`, including native CPU mechanics, frozen
BI2/P24 operators, parent geometry integrals, process cards, and complete D2W
and W2W p2 input/config profiles. No original project mount or ANSYS solve is
needed for these two profiles. This is an unvalidated engineering preview,
not a reproduction of a population-yield benchmark.

Use Linux with Python 3.11 or 3.12. From the exported `YAP` directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
unset YAP_NATIVE_RUNTIME_DIR YAP_RUNTIME_RESOURCES
export NUMBA_CACHE_DIR=/tmp/yap-numba-$USER

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python W2W/calculator_main.py \
  --config W2W/configs/design_6_4hi_p2/design_6_4hi_p2.yaml \
  --mode w2w_modeling --ds_name design_6_4hi_p2 \
  --ds_dir W2W/input/design_6_4hi_p2 --mechanism-filter overlay
```

For D2W, replace every `W2W` with `D2W` and use `--mode d2w_modeling`.
Both `stack.yaml` files select all five fixed-layer-order assembly trees.
Their initial bows are illustrative `[10, -10, 20, -20] um`; replace these
with the measured signed center-to-edge bows before a real study. D2W dies
remain 50 um total throughout; W2W uses the configured F2B preparation events.

The full YAML parameter groups are preserved. Assembly-order execution scores
overlay yield only. Defaults write under `D2W/output/assembly_orders` or
`W2W/output/assembly_orders` inside the exported YAP folder, independently of
the shell working directory. Use `--output-dir /your/output/path` to override.
Add `--plan-only` to check the input; this does not validate native operator
loading. The current memory gate requires at least 40 GiB available RAM.

Verify delivery hashes, both input profiles and native operator loading:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python verify_assembly_runtime.py --load-operators
```

The verifier denies access to the historical project before importing native
code. It checks deployment, not a complete five-order physics rerun.

Keep `runtime/` with the code when moving the folder. Frozen manifests retain
historical absolute paths as provenance; the loader rebases recognized paths
into the local native runtime before reading. Missing local files, wrong
hashes and conflicting native imports fail instead of using original storage.
`runtime/BUNDLE.json` records delivered file hashes; it is a delivery inventory,
not new physical validation. Do not rewrite frozen manifests or NPZ files.
The private `runtime/RESOURCES.json` resolves logical operator references and
is covered by the delivery inventory. The verifier rejects external runtime
or resource-map overrides.

Maintainers can export another copy without copying solver campaigns:

```bash
python package_assembly_runtime.py --output-dir /your/shared/delivery/folder
```

This bundle is scoped to p2/TOTAL50 and the existing thermal recipe. Other
geometries or the p6 reference profile require their own operator/data export.

Runtime bundles and raw execution archives are private scientific dependencies,
not public release artifacts. They retain immutable provenance and original
file names. Publish only the reviewed source/configs and separately reviewed
result summaries; do not publish raw manifests, source snapshots or archives.

## Shared-Server Git Checkouts

The primary assembly-order experiment guide is now the
[Assembly-Order Experiments section of README.md](README.md#assembly-order-experiments).
It includes shared runtime setup, operator-path configuration, all three p2
run commands, order selection and model limitations. Keep this document for
standalone export and delivery verification, not a second copy of that guide.

`verify_assembly_runtime.py` verifies a self-contained exported delivery, not
a Git checkout using an external shared runtime. The October 4 exporter still
includes only the original D2W/W2W p2 profiles, not the newer W2W no-thinning
profile. Use the current Git source and its README for that experiment.
