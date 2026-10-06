"""Relocation preserves frozen bytes and never falls back to original storage."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

import mechanics_runtime as runtime
from W2W.assembly_operator_library import _bound_reference_factory
from package_assembly_runtime import replace_yaml_scalar


@pytest.mark.parametrize("root", runtime.HISTORICAL_NATIVE_ROOTS)
def test_frozen_absolute_paths_rebased_before_existence_check(tmp_path, root):
    selected = tmp_path / "native"
    selected.mkdir()
    path = runtime.resolve_native_input(selected, root / "results/data.npz", native_root=selected, portable=True)
    assert path == selected / "results/data.npz"
    with pytest.raises(FileNotFoundError):
        path.read_bytes()


@pytest.mark.parametrize("path", ["../../../outside.npz", "/unknown/Ansys_Verify/12_stack_model_validation/data.npz",
                                  "#snapshot/data.npz", ".snapshot/data.npz"])
def test_foreign_paths_traversal_and_snapshots_rejected(tmp_path, path):
    with pytest.raises(ValueError):
        runtime.resolve_native_input(tmp_path, path, native_root=tmp_path, portable=True)


def test_symlink_escape_rejected(tmp_path):
    native = tmp_path / "native"
    native.mkdir()
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"outside")
    (native / "data.npz").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        runtime.resolve_native_input(native, "data.npz", native_root=native, portable=True)


def test_existing_bundle_and_override_fail_closed(tmp_path, monkeypatch):
    monkeypatch.delenv("YAP_NATIVE_RUNTIME_DIR", raising=False)
    (tmp_path / "runtime").mkdir()
    with pytest.raises(FileNotFoundError, match="no legacy fallback"):
        runtime.select_native_root(tmp_path)
    monkeypatch.setenv("YAP_NATIVE_RUNTIME_DIR", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="no legacy fallback"):
        runtime.select_native_root(tmp_path)


def test_deleted_runtime_and_broken_link_do_not_activate_legacy(tmp_path, monkeypatch):
    monkeypatch.delenv("YAP_NATIVE_RUNTIME_DIR", raising=False)
    (tmp_path / "ASSEMBLY_RUNTIME.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="no legacy fallback"):
        runtime.select_native_root(tmp_path)
    (tmp_path / "ASSEMBLY_RUNTIME.json").unlink()
    (tmp_path / "runtime").symlink_to(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="no legacy fallback"):
        runtime.select_native_root(tmp_path)


def test_implicit_bundle_cannot_link_to_external_native_code(tmp_path, monkeypatch):
    monkeypatch.delenv("YAP_NATIVE_RUNTIME_DIR", raising=False)
    bundle, external = tmp_path / "bundle", tmp_path / "original"
    (bundle / "runtime").mkdir(parents=True)
    external.mkdir()
    (external / "exp12_layer_rom_dataset.py").write_text("# external fixture\n")
    (bundle / "runtime/native").symlink_to(external)
    with pytest.raises(ValueError, match="outside the delivery"):
        runtime.select_native_root(bundle)


def test_preloaded_foreign_native_module_rejected_before_import(tmp_path, monkeypatch):
    import sys
    import history_mechanics_backend as backend
    monkeypatch.setattr(backend, "PORTABLE_RUNTIME", True)
    monkeypatch.setattr(backend, "ANSYS_VALIDATION", tmp_path)
    monkeypatch.setitem(sys.modules, "exp12_foreign_fixture", SimpleNamespace(__file__="/foreign/exp12_foreign_fixture.py"))
    def unexpected(_):
        pytest.fail("must reject cached foreign modules before import")
    monkeypatch.setattr(backend.importlib, "import_module", unexpected)
    with pytest.raises(ImportError, match="outside bundled runtime"):
        backend._import_mechanics_module("exp12_new_fixture")


def test_factory_uses_private_decoded_paths_and_keeps_material_json(tmp_path, monkeypatch):
    old = runtime.HISTORICAL_NATIVE_ROOTS[0]
    parent = {"port_basis": str(old / "results/parent.npz"), "datasets": {"bi": str(old / "results/bi.npz")}}
    native_path = tmp_path / "parent.json"
    native_path.write_text(json.dumps(parent))
    monkeypatch.setattr(runtime, "NATIVE_ROOT", tmp_path)
    monkeypatch.setattr(runtime, "PORTABLE_RUNTIME", True)
    monkeypatch.setitem(globals(), "NATIVE", Path("public_parent.json"))
    class Factory:
        def __init__(self, library, package):
            self.parent = json.loads(NATIVE.read_text())
            self.package = package
            self.material = json.loads('{"si": {"nu": 0.28}}')
    api = SimpleNamespace(SiliconBulkFactory=Factory)
    original = Factory.__init__.__globals__["NATIVE"]
    instance = _bound_reference_factory(api, None, parent, native_path)
    assert instance.parent["port_basis"] == str(tmp_path / "results/parent.npz")
    assert instance.package["datasets"]["bi"] == str(tmp_path / "results/bi.npz")
    assert instance.material == {"si": {"nu": .28}}
    assert Factory.__init__.__globals__["NATIVE"] == original
    assert json.loads(native_path.read_text()) == parent


def test_yaml_deployment_changes_keep_fields_order_and_comments():
    text = "model:\n  OUTPUT_DIR: /old/output  # notes\n  OTHER: 42\n"
    assert replace_yaml_scalar(text, "OUTPUT_DIR", "output/assembly_orders") == (
        "model:\n  OUTPUT_DIR: output/assembly_orders  # notes\n  OTHER: 42\n")


def resource_fixture(tmp_path):
    root = tmp_path / "native"
    root.mkdir()
    mapping = tmp_path / "resources.json"
    mapping.write_text(json.dumps(dict(schema="yap.native_resources.v1",
        resources={"compact_p2": "results/compact", "geometry": "results/parent"})))
    return root, mapping


def test_logical_references_resolve_against_explicit_root(tmp_path, monkeypatch):
    root, mapping = resource_fixture(tmp_path)
    monkeypatch.setattr(runtime, "NATIVE_ROOT", tmp_path / "wrong")
    path = runtime.resolve_resource("resource:compact_p2/packages/BI2_50.json",
                                    native_root=root, map_path=mapping)
    assert path == root / "results/compact/packages/BI2_50.json"
    assert not path.exists()


@pytest.mark.parametrize("value", [
    "resource:compact_p2/../outside.json", "resource:compact_p2/./x.json",
    "resource:compact_p2/#snapshot/x.json", "resource:compact_p2/.snapshot/x.json",
    "resource:compact_p2//x.json", "resource:compact_p2/x%2F.json",
    "resource:compact_p2/x?y", "resource:compact_p2/x#y", "resource:compact_p2/x\\y",
    "resource:compact_p2", "resource:/x.json", "resource:unknown/x.json",
])
def test_resource_reference_rejects_unsafe_or_unknown_names(tmp_path, value):
    root, mapping = resource_fixture(tmp_path)
    with pytest.raises(ValueError):
        runtime.resolve_resource(value, native_root=root, map_path=mapping)


@pytest.mark.parametrize("directory", ["../outside", "/outside", "results/.snapshot", "results/#snapshot"])
def test_resource_map_rejects_unsafe_directories(tmp_path, directory):
    root, mapping = resource_fixture(tmp_path)
    mapping.write_text(json.dumps(dict(schema="yap.native_resources.v1", resources={"compact_p2": directory})))
    with pytest.raises(ValueError):
        runtime.resolve_resource("resource:compact_p2/x.json", native_root=root, map_path=mapping)


def test_resource_map_rejects_duplicates_oversize_and_symlinks(tmp_path):
    root, mapping = resource_fixture(tmp_path)
    mapping.write_text('{"schema":"yap.native_resources.v1","resources":{"x":"a","x":"b"}}')
    with pytest.raises(ValueError, match="duplicate"):
        runtime.load_resource_map(mapping)
    mapping.write_bytes(b" " * 65537)
    with pytest.raises(ValueError, match="64 KiB"):
        runtime.load_resource_map(mapping)
    mapping.write_text(json.dumps(dict(schema="yap.native_resources.v1", resources={"compact_p2": "linked"})))
    (root / "linked").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlinks"):
        runtime.resolve_resource("resource:compact_p2/x.json", native_root=root, map_path=mapping)


def test_missing_private_map_has_no_legacy_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match="private runtime resource map missing"):
        runtime.resolve_resource("resource:compact_p2/x.json", native_root=tmp_path,
                                 map_path=tmp_path / "missing.json")


def test_plan_loading_keeps_resource_references_without_private_map(tmp_path, monkeypatch):
    from W2W.assembly_order import load_stack_input, build_plan
    monkeypatch.setenv("YAP_RUNTIME_RESOURCES", str(tmp_path / "missing.json"))
    for domain, name in (("D2W", "design_6_4hi_p2"),
                         ("W2W", "design_6_4hi_p2"),
                         ("W2W", "design_6_4hi_p2_no_thinning"),
                         ("W2W", "design_6_4hi_p6_reference")):
        if domain == "D2W":
            from D2W.assembly_order import build_plan as planner, STACK_SCHEMA, INPUT_SCHEMA
            design = load_stack_input(runtime.ROOT / domain / "input" / name / "stack.yaml",
                                      stack_schema=STACK_SCHEMA, input_schema=INPUT_SCHEMA)
        else:
            planner = build_plan
            design = load_stack_input(runtime.ROOT / domain / "input" / name / "stack.yaml")
        assert design["mechanics"]["operator_package"].startswith("resource:")
        assert len(planner(design["stack"])["routes"]) == 5


@pytest.mark.parametrize("name", ["YAP_NATIVE_RUNTIME_DIR", "YAP_RUNTIME_RESOURCES"])
def test_delivery_verifier_rejects_external_overrides_before_reading_files(monkeypatch, name):
    from verify_assembly_runtime import verify
    monkeypatch.setenv(name, "/unavailable")
    with pytest.raises(ValueError, match="overrides"):
        verify()


@pytest.mark.parametrize("metric", [{"model": "unsupported"}, {}, [], "", False])
def test_unknown_or_malformed_archive_metrics_do_not_fall_back_to_scalar(metric):
    from test_assembly_order import config, overlay_input, synthetic_archive
    from W2W.assembly_order import build_plan, evaluate_archive
    plan = build_plan(config())
    archive = synthetic_archive(plan)
    archive["native_configuration"] = {"metric": metric}
    with pytest.raises(ValueError, match="unsupported mechanics archive metric"):
        evaluate_archive(plan, archive, overlay_input())


@pytest.mark.parametrize("through_symlink", [False, True])
def test_miniature_private_export_inventories_relocatable_resource_map(tmp_path, monkeypatch, through_symlink):
    import package_assembly_runtime as package
    root, mapping = resource_fixture(tmp_path)
    operator = root / "results/compact/packages/BI2_50.json"
    operator.parent.mkdir(parents=True)
    operator.write_bytes(b"synthetic operator bytes, not physical evidence")
    module = root / "exp12_layer_rom_dataset.py"
    module.write_text("# synthetic native code fixture\n")
    metric = root.parent / "200_bow_diff_vs_magnification/nonlinear_face_lever.py"
    metric.parent.mkdir()
    metric.write_text("# synthetic metric fixture\n")
    observed = []
    def data_files(selected, *, map_path):
        observed.append((selected, map_path))
        return {operator: package.digest(operator)}
    monkeypatch.setattr(package, "data_files", data_files)
    monkeypatch.setattr(package, "native_sources", lambda _: [module])
    monkeypatch.setattr(runtime, "NATIVE_ROOT", tmp_path / "foreign")
    selected = root
    if through_symlink:
        selected = tmp_path / "native_link"
        selected.symlink_to(root, target_is_directory=True)
    output = package.export(tmp_path / "delivery", selected, map_path=mapping) / "YAP"
    assert observed == [(root, mapping)]
    delivered_map = output / "runtime/RESOURCES.json"
    inventory = json.loads((output / "runtime/BUNDLE.json").read_text())
    assert inventory["files_sha256"]["YAP/runtime/RESOURCES.json"] == package.digest(delivered_map)
    assert json.loads(delivered_map.read_text())["resources"] == {
        "compact_p2": "results/compact", "geometry": "results/parent"}
    moved = tmp_path / "moved"
    output.rename(moved)
    moved_operator = runtime.resolve_resource("resource:compact_p2/packages/BI2_50.json",
        native_root=moved / "runtime/native", map_path=moved / "runtime/RESOURCES.json")
    assert moved_operator.read_bytes() == operator.read_bytes()
    for domain in ("W2W", "D2W"):
        stack = (moved / domain / "input/design_6_4hi_p2/stack.yaml").read_text()
        assert "operator_package: resource:compact_p2/packages/BI2_50.json" in stack
    moved_map = moved / "runtime/RESOURCES.json"
    assert package.digest(moved_map) == inventory["files_sha256"]["YAP/runtime/RESOURCES.json"]
    moved_map.write_text(moved_map.read_text() + " ")
    assert package.digest(moved_map) != inventory["files_sha256"]["YAP/runtime/RESOURCES.json"]


def test_resource_map_rejects_nonregular_files_and_symlinked_parents(tmp_path):
    fifo = tmp_path / "resources.fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        runtime.load_resource_map(fifo)
    real = tmp_path / "real"
    real.mkdir()
    mapping = real / "resources.json"
    mapping.write_text('{}')
    link = tmp_path / "linked"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        runtime.load_resource_map(link / "resources.json")


def test_exporter_help_does_not_select_environment_native_root():
    environment = dict(os.environ, YAP_NATIVE_RUNTIME_DIR="/unavailable/native/runtime")
    result = subprocess.run([sys.executable, str(runtime.ROOT / "package_assembly_runtime.py"), "--help"],
        capture_output=True, text=True, env=environment, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--resource-map" in result.stdout
