"""Synthetic delivery checks; fixtures are not native or physical validation."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import mechanics_runtime as runtime
import tools.package_assembly_runtime as package
import tools.verify_assembly_runtime as verifier


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else value.encode())
    return path


def write_json(path, value):
    return write(path, json.dumps(value, sort_keys=True) + "\n")


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    root = tmp_path / "staged"
    source, native = root / "source", root / "runtime/native"
    root.mkdir()
    source.mkdir()
    native.mkdir(parents=True)
    for relative in ("D2W", "W2W", "D2W/utils", "W2W/utils"):
        (source / relative).mkdir(exist_ok=True)
    for name in ("requirements.txt", "LICENSE", "README.md", "PORTABLE_ASSEMBLY.md", "W2W/ASSEMBLY_ORDER.md"):
        write(source / name, "synthetic delivery fixture\n")
    for profile in package.BASE_PROFILES + package.PREPARED_PROFILES:
        domain, name = profile.split("/")
        write(source / domain / "input" / name / "stack.yaml", "stack: {}\n")
        write(source / domain / "configs" / name / f"{name}.yaml",
              f"{domain.lower()}_modeling:\n  OUTPUT_DIR: original/output\n  RANDOM_SEED: 7\n")
    module = write(native / "exp12_layer_rom_dataset.py", "# synthetic native module\n")
    write(native.parent / "200_bow_diff_vs_magnification/nonlinear_face_lever.py", "# synthetic metric\n")
    legacy = write(native / "results/compact/packages/BI2_50.json", b"synthetic legacy bytes")
    mapping = write(root / "runtime/RESOURCES.json", json.dumps({
        "schema": "yap.native_resources.v1", "resources": {
            "compact_p2": "results/compact", "geometry": "results/parent",
            "prepared_p2_native": "results/prepared",
        },
    }, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(package, "native_sources", lambda _root: [module])
    monkeypatch.setattr(package, "data_files", lambda _root, *, map_path: {legacy: package.digest(legacy)})
    resource = native / "results/prepared"
    basis = write(resource / "shared/port_basis.npz", b"synthetic basis")
    basis_hash = package.digest(basis)
    monkeypatch.setattr(package, "PREPARED_BASIS_SHA256", basis_hash)
    shared, prepared = {}, {}
    for key in ("bi", "beol"):
        shared[key] = write(resource / f"shared/{key}.npz", f"synthetic {key}")
        prepared[key] = write(resource / f"shared/{key}_prepared.npz", f"synthetic prepared {key}")
    parent = write_json(resource / "shared/parent_BI2_50.json", {
        "schema": "exp12.ofat.geometry_package.v1", "name": "BI2_50", "rank": 64,
        "ready": True, "state": "READY_ENGINEERING_PREVIEW", "port_basis_sha256": basis_hash,
        "dataset_sha256": {key: package.digest(path) for key, path in shared.items()},
    })
    catalog = {"schema": "yap.prepared_p2_operator_catalog.v1", "packages": {}}
    packages, receipts = {}, {}
    for total in (30, 80):
        key = f"tsv_si_h{total - 13}"
        datasets = {**shared, key: write(resource / f"datasets/{key}.npz", f"synthetic {key}")}
        geometry = {**prepared, key: write(resource / f"prepared/{key}.npz", f"synthetic geometry {key}")}
        stage = write(resource / f"build_receipts/{key}/STATIC_PLAN.json", "synthetic command freeze\n")
        bubble = write(resource / f"build_receipts/{key}/bubble.npz", "synthetic transferred bubble\n")
        raw_bubble = write(resource / f"build_receipts/{key}/raw_bubble.npz", "synthetic raw transferred bubble\n")
        dataset_hashes = {name: package.digest(path) for name, path in datasets.items()}
        prepared_hashes = {name: package.digest(path) for name, path in geometry.items()}
        audit = {
            "schema": "yap.prepared_p2_operator_audit.v1", "state": "PASS_STRUCTURAL_ENGINEERING_CANDIDATE",
            "all_cu_points_retained": True, "thermal_geometry_validated": False,
            "stage_count": 6, "rank": 64, "face_node_shape": [46, 46],
            "total_thickness_um": total, "retained_si_thickness_um": total - 13,
            "parent_package_sha256": package.digest(parent), "port_basis_sha256": basis_hash,
            "datasets": {name: {"sha256": value, "cu_point_count": 3} for name, value in dataset_hashes.items()},
            "prepared": {name: {"sha256": value} for name, value in prepared_hashes.items()},
            "transfer_audit": {
                "transferred_bubble_sha256": package.digest(bubble),
                "transferred_bubble": {"path": "../" + str(bubble.relative_to(resource)), "sha256": package.digest(bubble)},
                "raw_transferred_bubble_sha256": package.digest(raw_bubble),
                "raw_transferred_bubble": {"path": "../" + str(raw_bubble.relative_to(resource)), "sha256": package.digest(raw_bubble)},
            },
            "producer_receipts_sha256": {str(stage.relative_to(resource)): package.digest(stage),
                                        str(raw_bubble.relative_to(resource)): package.digest(raw_bubble)},
        }
        receipt = write_json(resource / f"audit/BI2_{total}.json", audit)
        manifest = {
            "schema": "yap.prepared_p2_geometry_package.v1", "name": f"BI2_{total}",
            "rank": 64, "ready": False, "state": "CANDIDATE_NOT_VALIDATED",
            "datasets": {name: "../" + str(path.relative_to(resource)) for name, path in datasets.items()},
            "prepared": {name: "../" + str(path.relative_to(resource)) for name, path in geometry.items()},
            "dataset_sha256": dataset_hashes, "prepared_sha256": prepared_hashes,
            "port_basis": "../shared/port_basis.npz", "port_basis_sha256": basis_hash,
            "parent_package": "../shared/parent_BI2_50.json", "parent_package_sha256": package.digest(parent),
            "audit_receipt": f"../audit/BI2_{total}.json", "audit_receipt_sha256": package.digest(receipt),
        }
        manifest_path = write_json(resource / f"packages/BI2_{total}.json", manifest)
        catalog["packages"][str(total)] = {
            "package_sha256": package.digest(manifest_path), "audit_sha256": package.digest(receipt),
            "parent_package_sha256": package.digest(parent), "port_basis_sha256": basis_hash,
        }
        packages[total], receipts[total] = manifest_path, receipt
    catalog_path = write_json(source / "prepared_operator_catalog.json", catalog)
    return SimpleNamespace(source=source, native=native, mapping=mapping, resource=resource,
                           catalog=catalog_path, packages=packages, receipts=receipts)


def reseal_receipt(delivery, total, receipt):
    write_json(delivery.receipts[total], receipt)
    manifest = json.loads(delivery.packages[total].read_text())
    manifest["audit_receipt_sha256"] = package.digest(delivery.receipts[total])
    write_json(delivery.packages[total], manifest)
    catalog = json.loads(delivery.catalog.read_text())
    catalog["packages"][str(total)]["audit_sha256"] = package.digest(delivery.receipts[total])
    catalog["packages"][str(total)]["package_sha256"] = package.digest(delivery.packages[total])
    write_json(delivery.catalog, catalog)


@pytest.mark.parametrize("value", ["../outside.py", "a/../../outside.py", "#snapshot/data.npz", ".snapshot/data.npz"])
def test_source_reference_rejects_lexical_escape_before_read(tmp_path, value):
    with pytest.raises(ValueError):
        package.source_reference(tmp_path, value)


@pytest.mark.parametrize("outside", [False, True])
def test_source_reference_rejects_every_symlink_component(tmp_path, outside):
    root = tmp_path / "native"
    root.mkdir()
    target = tmp_path / "outside" if outside else root / "inside"
    target.mkdir()
    write(target / "data.npz", b"fixture")
    (root / "alias").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        package.source_reference(root, "alias/data.npz")


def test_historical_source_relocated_without_probing_old_alias(tmp_path, monkeypatch):
    target = write(tmp_path / "results/data.npz", b"fixture")
    old = package.HISTORICAL_NATIVE_ROOTS[-1] / "results/data.npz"
    original = Path.is_symlink
    def guarded(path):
        if path.is_relative_to(package.HISTORICAL_NATIVE_ROOTS[-1]):
            pytest.fail("historical lexical paths must be relocated before filesystem access")
        return original(path)
    monkeypatch.setattr(Path, "is_symlink", guarded)
    assert package.source_reference(tmp_path, old) == target
    assert package.source_reference(tmp_path, target) == target


def test_source_reference_rejects_snapshot_root_before_resolution(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "resolve", lambda *_args, **_kwargs: pytest.fail("must reject snapshot root first"))
    with pytest.raises(ValueError, match="snapshot"):
        package.source_reference(tmp_path / "#snapshot", "data.npz")


def test_source_reference_checks_resolved_containment(tmp_path, monkeypatch):
    candidate = tmp_path / "data.npz"
    original = Path.resolve
    monkeypatch.setattr(Path, "resolve", lambda path, *args, **kwargs:
                        tmp_path.parent / "outside.npz" if path == candidate else original(path, *args, **kwargs))
    with pytest.raises(ValueError, match="escapes"):
        package.source_reference(tmp_path, "data.npz")


def test_native_import_closure_rejects_symlinked_module(tmp_path, monkeypatch):
    write(tmp_path / "entry.py", "import linked\n")
    target = write(tmp_path / "real.py", "# fixture\n")
    (tmp_path / "linked.py").symlink_to(target)
    monkeypatch.setattr(package, "SEEDS", ("entry",))
    with pytest.raises(ValueError, match="symlinks"):
        package.native_sources(tmp_path)


def test_prepared_relative_references_cannot_escape_or_hide_symlinks(tmp_path):
    base = tmp_path / "packages"
    base.mkdir()
    assert package.prepared_artifact(base, "../shared/basis.npz", tmp_path) == tmp_path / "shared/basis.npz"
    with pytest.raises(ValueError, match="escapes"):
        package.prepared_artifact(base, "../../outside.npz", tmp_path)
    (tmp_path / "alias").symlink_to(base, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        package.prepared_artifact(base, "../alias/../shared/basis.npz", tmp_path)


@pytest.mark.parametrize("prepared", [False, True])
def test_export_inventory_is_explicit_and_prepared_mode_is_d2w_only(delivery, tmp_path, prepared):
    write(delivery.resource / "unlisted/raw.rst", b"must never be copied")
    destination = tmp_path / "bundle"
    package.export(destination, delivery.native, source_root=delivery.source,
                   map_path=delivery.mapping, prepared_operators=prepared)
    output = destination / "YAP"
    inventory = json.loads((output / "runtime/BUNDLE.json").read_text())
    assert inventory["profiles"] == list(package.delivery_profiles(prepared))
    assert not any("no_thinning" in name for name in inventory["files_sha256"])
    assert (not any(name.startswith("W2W/") for name in inventory["profiles"])) == prepared
    resources = json.loads((output / "runtime/RESOURCES.json").read_text())["resources"]
    assert ("prepared_p2_native" in resources) == prepared
    assert (output / "prepared_operator_catalog.json").exists() == prepared
    assert not (output / "runtime/native/results/prepared/unlisted/raw.rst").exists()
    if prepared:
        for total in (30, 80):
            assert (output / f"runtime/native/results/prepared/audit/BI2_{total}.json").read_bytes() == delivery.receipts[total].read_bytes()
            for name in ("bubble.npz", "raw_bubble.npz"):
                relative = f"runtime/native/results/prepared/build_receipts/tsv_si_h{total - 13}/{name}"
                assert inventory["files_sha256"]["YAP/" + relative] == package.digest(output / relative)
    for filename, expected in inventory["files_sha256"].items():
        assert package.digest(destination / filename) == expected


def test_export_includes_relocated_tools_and_their_pins(delivery, tmp_path):
    for name in ("__init__.py", "operator_grid.py"):
        write(delivery.source / "tools" / name, "# synthetic offline tool\n")
    destination = tmp_path / "tools_bundle"
    package.export(destination, delivery.native, source_root=delivery.source,
                   map_path=delivery.mapping)
    output = destination / "YAP"
    inventory = json.loads((output / "runtime/BUNDLE.json").read_text())
    for name in ("__init__.py", "operator_grid.py"):
        path = output / "tools" / name
        assert path.read_bytes() == (delivery.source / "tools" / name).read_bytes()
        assert inventory["files_sha256"][f"YAP/tools/{name}"] == package.digest(path)


def test_export_paces_copy_and_readback(delivery, tmp_path, monkeypatch):
    sleeps = []
    rates = []
    original = package.digest
    monkeypatch.setattr(package.time, "sleep", sleeps.append)
    def checked_digest(path, *, rate_mib=None):
        rates.append(rate_mib)
        return original(path, rate_mib=rate_mib)
    monkeypatch.setattr(package, "digest", checked_digest)
    package.export(tmp_path / "paced", delivery.native, source_root=delivery.source,
                   map_path=delivery.mapping, prepared_operators=True)
    assert sleeps and all(delay >= 0. for delay in sleeps)
    assert 32 in rates


def frozen_stage(delivery):
    root = delivery.native.parent.parent
    files = {}
    for directory in (delivery.native, delivery.source):
        # The fixture is tiny and lives only under pytest's approved cache.
        for path in directory.rglob("*"):
            if path.is_file():
                files[str(path.relative_to(root))] = package.digest(path)
    metric = delivery.native.parent / "200_bow_diff_vs_magnification/nonlinear_face_lever.py"
    files[str(metric.relative_to(root))] = package.digest(metric)
    files[str(delivery.mapping.relative_to(root))] = package.digest(delivery.mapping)
    path = write_json(root / "STAGE.json", {
        "schema": "yap.prepared_runtime_stage.v1", "files_sha256": files,
        "source_payloads_rewritten": False, "application_source": "source",
    })
    return path, package.digest(path)


def test_export_preserves_every_used_staged_pin(delivery, tmp_path):
    receipt, sha = frozen_stage(delivery)
    destination = tmp_path / "frozen_export"
    package.export(destination, delivery.native, source_root=delivery.source,
                   map_path=delivery.mapping, prepared_operators=True,
                   stage_inventory=receipt, stage_inventory_sha256=sha)
    inventory = json.loads((destination / "YAP/runtime/BUNDLE.json").read_text())
    assert inventory["stage_inventory_sha256"] == sha
    assert inventory["files_sha256"]["YAP/runtime/SOURCE_STAGE.json"] == sha
    assert inventory["files_sha256"]["YAP/runtime/RESOURCES.json"] == package.digest(delivery.mapping)


@pytest.mark.parametrize("fault", ["receipt", "module", "metric", "application", "map", "missing_application", "missing_module"])
def test_staged_export_rejects_missing_or_changed_inputs_before_allocation(delivery, tmp_path, fault):
    receipt, sha = frozen_stage(delivery)
    if fault == "receipt":
        receipt.write_text(receipt.read_text() + " ")
    elif fault in ("missing_application", "missing_module"):
        document = json.loads(receipt.read_text())
        relative = ("source/README.md" if fault == "missing_application" else
                    "runtime/native/exp12_layer_rom_dataset.py")
        del document["files_sha256"][relative]
        write_json(receipt, document)
        sha = package.digest(receipt)
    else:
        path = {
            "module": delivery.native / "exp12_layer_rom_dataset.py",
            "metric": delivery.native.parent / "200_bow_diff_vs_magnification/nonlinear_face_lever.py",
            "application": delivery.source / "README.md", "map": delivery.mapping,
        }[fault]
        path.write_text(path.read_text() + " ")
    destination = tmp_path / "rejected_stage"
    with pytest.raises(ValueError):
        package.export(destination, delivery.native, source_root=delivery.source,
                       map_path=delivery.mapping, prepared_operators=True,
                       stage_inventory=receipt, stage_inventory_sha256=sha)
    assert not destination.exists()


def test_staged_export_rechecks_receipt_before_publication(delivery, tmp_path, monkeypatch):
    receipt, sha = frozen_stage(delivery)
    original = package.os.fsync
    def changed_receipt(descriptor):
        original(descriptor)
        receipt.write_text(receipt.read_text() + " ")
    monkeypatch.setattr(package.os, "fsync", changed_receipt)
    destination = tmp_path / "changed_stage"
    with pytest.raises(ValueError, match="staging receipt changed"):
        package.export(destination, delivery.native, source_root=delivery.source,
                       map_path=delivery.mapping, prepared_operators=True,
                       stage_inventory=receipt, stage_inventory_sha256=sha)
    assert not (destination / "YAP/runtime/BUNDLE.json").exists()


def test_staged_export_requires_exact_native_root(delivery):
    receipt, sha = frozen_stage(delivery)
    wrong = delivery.native.parent.parent / "alternate/native"
    with pytest.raises(ValueError, match="selected native"):
        package.staged_bindings(wrong, delivery.source, delivery.mapping, receipt, sha)


@pytest.mark.parametrize("failure", ["pending", "one_target", "package", "receipt", "basis", "symlink", "escape"])
def test_prepared_export_failures_leave_no_destination(delivery, tmp_path, failure):
    catalog = json.loads(delivery.catalog.read_text())
    if failure in ("pending", "one_target"):
        catalog["packages"] = {} if failure == "pending" else {"30": catalog["packages"]["30"]}
        write_json(delivery.catalog, catalog)
    elif failure in ("package", "receipt"):
        path = delivery.packages[30] if failure == "package" else delivery.receipts[30]
        write(path, path.read_bytes() + b" ")
    elif failure == "basis":
        catalog["packages"]["30"]["port_basis_sha256"] = "f" * 64
        write_json(delivery.catalog, catalog)
    else:
        manifest = json.loads(delivery.packages[30].read_text())
        if failure == "symlink":
            original = delivery.resource / "shared/bi.npz"
            alias = delivery.resource / "shared/alias.npz"
            alias.symlink_to(original)
            manifest["datasets"]["bi"] = "../shared/alias.npz"
        else:
            manifest["datasets"]["bi"] = "../../../outside.npz"
        write_json(delivery.packages[30], manifest)
        catalog["packages"]["30"]["package_sha256"] = package.digest(delivery.packages[30])
        write_json(delivery.catalog, catalog)
    destination = tmp_path / "bundle"
    with pytest.raises((RuntimeError, ValueError)):
        package.export(destination, delivery.native, source_root=delivery.source,
                       map_path=delivery.mapping, prepared_operators=True)
    assert not destination.exists()


@pytest.mark.parametrize("failure", ["missing_binding", "binding_hash", "payload_hash", "same_artifact",
                                     "tampered", "missing_file", "escape", "symlink"])
def test_prepared_export_checks_raw_bubble_binding_before_destination(delivery, tmp_path, failure):
    receipt = json.loads(delivery.receipts[30].read_text())
    transfer = receipt["transfer_audit"]
    raw = delivery.resource / "build_receipts/tsv_si_h17/raw_bubble.npz"
    if failure == "missing_binding":
        del transfer["raw_transferred_bubble"]
    elif failure == "binding_hash":
        transfer["raw_transferred_bubble"]["sha256"] = "f" * 64
    elif failure == "payload_hash":
        transfer["raw_transferred_bubble"]["sha256"] = transfer["raw_transferred_bubble_sha256"] = "f" * 64
    elif failure == "same_artifact":
        transfer["raw_transferred_bubble"] = dict(transfer["transferred_bubble"])
        transfer["raw_transferred_bubble_sha256"] = transfer["transferred_bubble_sha256"]
    elif failure == "tampered":
        write(raw, "tampered raw bubble")
    elif failure == "missing_file":
        raw.unlink()
    elif failure == "escape":
        transfer["raw_transferred_bubble"]["path"] = "../../outside.npz"
    else:
        alias = raw.with_name("alias.npz")
        alias.symlink_to(raw)
        transfer["raw_transferred_bubble"]["path"] = "../" + str(alias.relative_to(delivery.resource))
    reseal_receipt(delivery, 30, receipt)
    destination = tmp_path / "bundle"
    with pytest.raises(ValueError):
        package.export(destination, delivery.native, source_root=delivery.source,
                       map_path=delivery.mapping, prepared_operators=True)
    assert not destination.exists()


def test_prepared_export_raw_bubble_inventory_does_not_depend_on_producer_list(delivery, tmp_path):
    for total in (30, 80):
        receipt = json.loads(delivery.receipts[total].read_text())
        del receipt["producer_receipts_sha256"][f"build_receipts/tsv_si_h{total - 13}/raw_bubble.npz"]
        reseal_receipt(delivery, total, receipt)
    destination = tmp_path / "bundle"
    package.export(destination, delivery.native, source_root=delivery.source,
                   map_path=delivery.mapping, prepared_operators=True)
    inventory = json.loads((destination / "YAP/runtime/BUNDLE.json").read_text())
    for total in (30, 80):
        relative = f"YAP/runtime/native/results/prepared/build_receipts/tsv_si_h{total - 13}/raw_bubble.npz"
        assert inventory["files_sha256"][relative] == package.digest(destination / relative)


def library_and_inputs(root, total):
    receipt_path = root / f"runtime/native/results/prepared/audit/BI2_{total:g}.json"
    receipt = json.loads(receipt_path.read_text())
    aliases = {"bi": "bi", "beol": "beol", "retained_si": f"tsv_si_h{total - 13:g}"}
    operators = {name: SimpleNamespace(cu_point_count=3, weights_mm3=np.ones(3), macro_size=134) for name in aliases}
    library = SimpleNamespace(port_rank_per_face=64, operators=operators,
                              thickness_mm={"bi": .0015, "beol": .010, "retained_si": (total - 13) / 1000})
    inputs = dict(port_rank_per_face=64, port_basis_sha256=receipt["port_basis_sha256"],
                  source_operator_keys=aliases, operator_build_receipt=str(receipt_path),
                  input_sha256={str(receipt_path): package.digest(receipt_path)},
                  physics_qualification="SYNTHETIC_NOT_PHYSICAL_VALIDATION")
    return library, inputs


def test_prepared_verifier_loads_only_d2w_profiles(delivery, tmp_path, monkeypatch):
    package.export(tmp_path / "bundle", delivery.native, source_root=delivery.source,
                   map_path=delivery.mapping, prepared_operators=True)
    root = tmp_path / "bundle/YAP"
    monkeypatch.setattr(verifier, "ROOT", root)
    monkeypatch.setattr(verifier, "configure_runtime_storage", lambda: None)
    monkeypatch.setattr(verifier, "isolate_original_storage", lambda: None)
    monkeypatch.setattr(runtime, "NATIVE_ROOT", root / "runtime/native", raising=False)
    monkeypatch.setattr(runtime, "PORTABLE_RUNTIME", True, raising=False)
    for key in ("YAP_NATIVE_RUNTIME_DIR", "YAP_RUNTIME_RESOURCES"):
        monkeypatch.delenv(key, raising=False)
    from D2W import assembly_order
    from W2W import assembly_design, assembly_mechanics, assembly_operator_library
    calls = []
    def load(_cfg, _directory, name):
        total = float(name.rsplit("_t", 1)[1]) if "_t" in name else 50.
        settings = dict(operator_package=f"resource:prepared_p2_native/packages/BI2_{total:g}.json",
                        temporal_profile="distilled26", frozen_preview=total == 50.)
        request = dict(routes=[{}] * 5, context=dict(initial_bows_um=[10., -10., 20., -20.],
                                                   geometry=dict(total_thickness_um=total)))
        return {"mechanics": settings}, request, None, None, {"site_count": 1}
    def operators(path, geometry, *, frozen_preview):
        total = geometry["total_thickness_um"]
        calls.append((total, frozen_preview))
        if total != 50.:
            return library_and_inputs(root, total)
        return SimpleNamespace(port_rank_per_face=24, operators={}), dict(
            input_sha256={}, physics_qualification="SYNTHETIC_NOT_PHYSICAL_VALIDATION")
    monkeypatch.setattr(assembly_order, "load_design", load)
    monkeypatch.setattr(assembly_operator_library, "load_prepared_die_library", operators)
    monkeypatch.setattr(assembly_mechanics, "_recipe", lambda *_: None)
    monkeypatch.setattr(assembly_design, "load_design", lambda *_: pytest.fail("prepared verification must not load W2W"))
    result = verifier.verify(prepared_operators=True)
    assert calls == [(50., True), (30., False), (80., False)]
    assert result["native_operator_loading_verified"] is True
    assert result["full_assembly_histories_rerun"] is False
    assert result["profiles"]["D2W/design_6_4hi_p2_t30"]["all_cu_points_retained"] is True


@pytest.mark.parametrize("failure", ["rank", "cap", "thickness", "basis", "cu_count", "weights", "macro_size"])
def test_prepared_library_checks_reject_mismatches(delivery, tmp_path, failure):
    package.export(tmp_path / "bundle", delivery.native, source_root=delivery.source,
                   map_path=delivery.mapping, prepared_operators=True)
    library, inputs = library_and_inputs(tmp_path / "bundle/YAP", 30.)
    if failure == "rank":
        library.port_rank_per_face = 24
    elif failure == "cap":
        library.operators["bulk_si"] = library.operators["bi"]
    elif failure == "thickness":
        library.thickness_mm["retained_si"] = .037
    elif failure == "basis":
        inputs["port_basis_sha256"] = "f" * 64
    elif failure == "cu_count":
        library.operators["bi"].cu_point_count = 2
    elif failure == "weights":
        library.operators["bi"].weights_mm3[0] = np.nan
    else:
        library.operators["bi"].macro_size = 54
    with pytest.raises(ValueError):
        verifier.check_prepared_library(library, inputs, 30.)
