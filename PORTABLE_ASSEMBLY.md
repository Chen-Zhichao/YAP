# Runtime Delivery

For normal same-server D2W/W2W runs, follow [README.md](README.md).
The operator catalog and numerical dependencies are shared read-only data,
not files installed by Git or `pip`. Another server needs a separately
prepared data delivery with all referenced operators and evidence.

`package_assembly_runtime.py` and `verify_assembly_runtime.py` maintain the
legacy standalone runtime format. They do not export the component-v2 catalog
or the thickness family used by the current generator. Do not use a legacy
bundle as a substitute for the component-v2 data.

Maintainers can inspect the exporter and verifier options with:

```bash
python -B package_assembly_runtime.py --help
python -B verify_assembly_runtime.py --help
```

Use the README storage setup first. Every new delivery, cache and verification
output must stay under the selected private `YAP_RUN_ROOT`; shared source and
published data remain read-only. Export only to a new directory and retain the
bundle's inventory, resource map and immutable provenance.
