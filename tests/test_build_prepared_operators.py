"""Safety checks for the separately launched offline prepared-body builder."""

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.build_prepared_operators as build


@pytest.mark.parametrize("path", ["/tmp/new-build", "/app/nanocad/#snapshot/build", "/app/nanocad/.snapshot/build"])
def test_forbidden_storage(path):
    with pytest.raises(ValueError, match="forbidden storage"):
        build.safe_path(path)


def test_alias_to_forbidden_storage(tmp_path):
    link = tmp_path / "alias"
    link.symlink_to("/tmp", target_is_directory=True)
    with pytest.raises(ValueError, match="forbidden storage"):
        build.safe_path(link / "build")


def test_work_requires_local_disk_before_creating_files(tmp_path):
    work = tmp_path / "rejected"
    with pytest.raises(ValueError, match="local scratch"):
        build.configure(work)
    assert not work.exists()


@pytest.mark.parametrize("host", ["mfd", "sysml"])
def test_physical_host_allowlist(host, monkeypatch):
    monkeypatch.setattr(build.socket, "gethostname", lambda: host + ".ee.ucla.edu")
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **k:
                        "init" if str(p) == "/proc/1/comm" else original(p, *a, **k))
    assert build.physical_host() == host


@pytest.mark.parametrize("host,pid1", [("unknown", "init"), ("sysml", "codex")])
def test_physical_host_rejects_unknown_or_container(host, pid1, monkeypatch):
    monkeypatch.setattr(build.socket, "gethostname", lambda: host)
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **k:
                        pid1 if str(p) == "/proc/1/comm" else original(p, *a, **k))
    with pytest.raises(RuntimeError, match="physical"):
        build.physical_host()


@pytest.mark.parametrize("host", ["mfd", "sysml"])
def test_scratch_is_bound_to_host_and_strict_child(tmp_path, monkeypatch, host):
    roots = {"mfd": tmp_path / "mfd", "sysml": tmp_path / "sysml"}
    monkeypatch.setattr(build, "OWNED_WORK_ROOT", roots["mfd"])
    monkeypatch.setattr(build, "SYSML_WORK_ROOT", roots["sysml"])
    monkeypatch.setattr(build, "physical_host", lambda: host)
    work = roots[host] / "new-campaign"
    assert build.local_scratch(work) == work
    assert build.local_scratch(roots[host], allow_root=True) == roots[host]
    with pytest.raises(ValueError):
        build.local_scratch(roots[host])
    other = "sysml" if host == "mfd" else "mfd"
    with pytest.raises(ValueError, match="physical execution host"):
        build.local_scratch(roots[other] / "campaign")
    assert not work.exists()


def test_scratch_alias_never_creates_runtime(tmp_path, monkeypatch):
    root = tmp_path / "local"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    monkeypatch.setattr(build, "SYSML_WORK_ROOT", root)
    monkeypatch.setattr(build, "physical_host", lambda: "sysml")
    with pytest.raises(ValueError, match="symlinks or aliases"):
        build.configure(alias / "campaign")
    assert list(root.iterdir()) == []


def test_alternate_mfd_scratch_does_not_move_host_lock(tmp_path, monkeypatch):
    canonical, alternate = tmp_path / "data", tmp_path / "scratch"
    monkeypatch.setattr(build, "OWNED_WORK_ROOT", canonical)
    monkeypatch.setattr(build, "MFD_ALTERNATE_WORK_ROOT", alternate)
    monkeypatch.setattr(build, "physical_host", lambda: "mfd")
    assert build.local_scratch(alternate / "campaign") == alternate / "campaign"
    assert build.scratch_root("mfd") == canonical
    with pytest.raises(ValueError):
        build.local_scratch(alternate)
    with pytest.raises(ValueError, match="physical execution host"):
        build.local_scratch(alternate / "campaign", host="sysml")


@pytest.mark.parametrize("host", ["mfd", "sysml"])
def test_api_uses_physical_host_for_all_cases_and_solver_paths(tmp_path, monkeypatch, host):
    work = tmp_path / "work"
    monkeypatch.setattr(build, "OWNED_WORK_ROOT", tmp_path)
    monkeypatch.setattr(build, "SYSML_WORK_ROOT", tmp_path)
    monkeypatch.setattr(build, "physical_host", lambda: host)
    calls = []
    module = SimpleNamespace(GeometryCase=lambda *a, **k: calls.append((a, k)))
    spec = SimpleNamespace(name="synthetic_api", loader=SimpleNamespace(exec_module=lambda _: None))
    monkeypatch.setattr(build.importlib.util, "spec_from_file_location", lambda *a: spec)
    monkeypatch.setattr(build.importlib.util, "module_from_spec", lambda _: module)
    monkeypatch.setitem(build.sys.modules, spec.name, module)
    api = build.load_api(tmp_path / "source", work)
    assert api.WORK_ROOTS == {host: work / "solver"}
    assert api.ROOT == work / "results"
    assert len(calls) == 2 and all(a[3] == host and k["cores"] == 8 for a, k in calls)


def test_digest_does_not_write(tmp_path):
    path = tmp_path / "input.dat"
    path.write_bytes(b"checked physical input")
    assert build.digest(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("path,kind,expected", [
    ("/home/local/data", "xfs", False),
    ("/home/local/data", "nfs", True),
    ("/home/local/data", "overlay", True),
    ("/home/locality/data", "xfs", True),
    ("/home/with space/data", "ext4", False),
])
def test_storage_pacing_uses_component_aware_mounts(monkeypatch, path, kind, expected):
    mount = "/home/with\\040space" if "space" in path else "/home/local"
    raw = f"1 0 0:1 / / rw - nfs server:/root rw\n2 1 8:1 / {mount} rw - {kind} /dev/a rw\n"
    original = Path.open
    monkeypatch.setattr(Path, "open", lambda p, *a, **k:
        io.StringIO(raw) if str(p) == "/proc/self/mountinfo" else original(p, *a, **k))
    assert build.shared_storage(Path(path)) is expected


@pytest.mark.parametrize("raw", ["", "invalid", "x" * (256 * 1024 + 1)])
def test_unknown_or_oversized_mount_metadata_is_paced(monkeypatch, raw):
    original = Path.open
    monkeypatch.setattr(Path, "open", lambda p, *a, **k:
        io.StringIO(raw) if str(p) == "/proc/self/mountinfo" else original(p, *a, **k))
    assert build.shared_storage(Path("/home/local/input"))


@pytest.mark.parametrize("shared", [False, True])
def test_digest_paces_only_shared_storage(tmp_path, monkeypatch, shared):
    path = tmp_path / "input"
    path.write_bytes(b"data")
    rates = []
    monkeypatch.setattr(build, "shared_storage", lambda _: shared)
    monkeypatch.setattr(build, "pace_io", lambda rate, *a: rates.append(rate))
    assert build.digest(path) == hashlib.sha256(b"data").hexdigest()
    assert rates == [32 * 1024**2 if shared else None]


@pytest.mark.parametrize("source_shared,target_shared", [(False, False), (True, False), (False, True)])
def test_pinned_copy_reads_source_once_and_destination_independently(tmp_path, monkeypatch,
                                                                   source_shared, target_shared):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"physical input")
    reads, rates = [], []
    original = Path.open
    def tracked(path, mode="r", *a, **k):
        if mode == "rb":
            reads.append(path)
        return original(path, mode, *a, **k)
    monkeypatch.setattr(Path, "open", tracked)
    monkeypatch.setattr(build, "shared_storage", lambda p: source_shared if p == source else target_shared)
    monkeypatch.setattr(build, "pace_io", lambda rate, *a: rates.append(rate))
    expected = hashlib.sha256(b"physical input").hexdigest()
    assert build.copy_checked(source, target, expected) == expected
    assert reads.count(source) == 1
    assert len(reads) == 2 and reads[1].name.endswith(".copying")
    assert rates[0] == (24 * 1024**2 if source_shared or target_shared else None)
    assert target.read_bytes() == b"physical input"
    assert set(tmp_path.iterdir()) == {source, target}


def test_copy_rejects_destination_corruption_and_preserves_existing_staging(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"physical input")
    expected = hashlib.sha256(b"physical input").hexdigest()
    original = build.digest
    def corrupt(path):
        if path.name.endswith(".copying"):
            path.write_bytes(b"damaged")
        return original(path)
    monkeypatch.setattr(build, "digest", corrupt)
    with pytest.raises(ValueError, match="readback"):
        build.copy_checked(source, target, expected)
    assert set(tmp_path.iterdir()) == {source}
    staging = target.with_name(target.name + f".{build.os.getpid()}.copying")
    staging.write_bytes(b"previous attempt")
    with pytest.raises(FileExistsError):
        build.copy_checked(source, target, expected)
    assert staging.read_bytes() == b"previous attempt" and not target.exists()


def test_unpinned_copy_detects_source_change_after_prehash(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"original")
    original = build.digest
    def changing(path):
        value = original(path)
        if path == source:
            source.write_bytes(b"changed")
        return value
    monkeypatch.setattr(build, "digest", changing)
    with pytest.raises(ValueError, match="source SHA256"):
        build.copy_checked(source, target)
    assert set(tmp_path.iterdir()) == {source}


def template_api(tmp_path, monkeypatch, commands=()):
    path = tmp_path / "producer.py"
    path.write_text("immutable producer")
    expected = build.digest(path)
    monkeypatch.setattr(build, "read_json", lambda _: {"inputs": {str(path): expected}})
    api = SimpleNamespace(static_steps=lambda _: [{"command": command} for command in commands])
    return api, SimpleNamespace(template=tmp_path / "template.json"), path


def test_template_binds_all_input_hashes(tmp_path, monkeypatch):
    api, case, path = template_api(tmp_path, monkeypatch)
    records = build.verify_template(api, case)
    assert records[str(path)]["template_sha256"] == records[str(path)]["build_sha256"]
    path.write_text("changed producer")
    with pytest.raises(ValueError, match="unreviewed"):
        build.verify_template(api, case)


@pytest.mark.parametrize("option", ["--pure-si-substrate", "--macro-reference-y-offset-mm"])
def test_new_optional_physics_branches_not_admitted(tmp_path, monkeypatch, option):
    api, case, _ = template_api(tmp_path, monkeypatch, [["python", "producer.py", option]])
    with pytest.raises(ValueError, match="not admitted"):
        build.verify_template(api, case)


def test_copy_refuses_wrong_source_and_existing_target(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"source")
    with pytest.raises(ValueError, match="source SHA256 mismatch"):
        build.copy_checked(source, target, "0" * 64)
    assert not target.exists()
    target.write_bytes(b"previous archive")
    with pytest.raises(FileExistsError):
        build.copy_checked(source, target)
    assert target.read_bytes() == b"previous archive"


def test_immutable_records_never_replaced(tmp_path):
    path = tmp_path / "receipt.json"
    build.write_json(path, {"state": "original"}, immutable=True)
    with pytest.raises(FileExistsError):
        build.write_json(path, {"state": "replacement"}, immutable=True)
    assert build.read_json(path) == {"state": "original"}


def test_stage_rejects_overlap_before_environment_or_disk_checks(tmp_path, monkeypatch):
    work = tmp_path / "work"
    monkeypatch.setattr(build, "OWNED_WORK_ROOT", tmp_path)
    monkeypatch.setattr(build, "resource_check", lambda _: pytest.fail("overlap must be checked first"))
    args = SimpleNamespace(work=work, baseline_runtime=work / "runtime", archive=tmp_path / "archive")
    with pytest.raises(ValueError, match="overlap"):
        build.stage_preflight(args)
    assert not work.exists()


def test_local_disk_gate_rejects_nfs_before_any_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(build.socket, "gethostname", lambda: "mfd")
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *a, **k: "init" if str(path) == "/proc/1/comm" else original(path, *a, **k))
    monkeypatch.setattr(build.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="/data server:/share nfs4"))
    with pytest.raises(RuntimeError, match="must not be NFS"):
        build.resource_check(tmp_path)


@pytest.mark.parametrize("fault", ["symlink", "oversize", "duplicate", "directory"])
def test_inventory_metadata_is_bounded_confined_and_regular(tmp_path, fault):
    path = tmp_path / "inventory.json"
    if fault == "symlink":
        real = tmp_path / "real.json"
        real.write_text("{}")
        path.symlink_to(real)
    elif fault == "oversize":
        path.write_bytes(b" " * (1024**2 + 1))
    elif fault == "duplicate":
        path.write_text('{"files":{},"files":{}}')
    else:
        path.mkdir()
    with pytest.raises(ValueError):
        build.checked_inventory(tmp_path, path)


def stage_fixture(tmp_path, monkeypatch):
    work, baseline, archive = (tmp_path / name for name in ("work", "old/YAP", "archive"))
    work.mkdir()
    (baseline / "runtime").mkdir(parents=True)
    archive.mkdir()
    resources = dict(schema="yap.native_resources.v1", resources={"compact_p2": "results/compact", "geometry": "results/geometry"})
    build.write_json(baseline / "runtime/RESOURCES.json", resources)
    files = {"YAP/runtime/RESOURCES.json": build.digest(baseline / "runtime/RESOURCES.json")}
    for index in range(16):
        relative = f"YAP/runtime/native/results/{'compact' if index < 8 else 'geometry'}/file{index}.dat"
        path = baseline.parent / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic payload only")
        files[relative] = build.digest(path)
    old = dict(schema="yap.portable_assembly_p2.v1", files_sha256=files)
    new_path = archive / "candidate.dat"
    new_path.write_bytes(b"synthetic new operator")
    new = dict(schema="yap.prepared_operator_archive.v1", thermal_geometry_validated=False,
               files_sha256={"candidate.dat": build.digest(new_path)})
    build.write_json(baseline / "runtime/BUNDLE.json", old)
    build.write_json(archive / "ARCHIVE.json", new)
    monkeypatch.setattr(build, "resource_check", lambda _: {"mount": "local fixture"})
    monkeypatch.setattr(build, "OWNED_WORK_ROOT", tmp_path)
    args = SimpleNamespace(work=work, baseline_runtime=baseline, archive=archive)
    return args, old, resources


@pytest.mark.parametrize("fault", ["tampered_map", "invalid_resource", "missing_closure", "oversized_files", "nonregular"])
def test_stage_preflight_failure_creates_no_destinations(tmp_path, monkeypatch, fault):
    args, old, resources = stage_fixture(tmp_path, monkeypatch)
    baseline = args.baseline_runtime
    if fault in ("tampered_map", "invalid_resource", "missing_closure"):
        resources["resources"]["compact_p2"] = "../escape" if fault == "invalid_resource" else "results/not_inventoried"
        build.write_json(baseline / "runtime/RESOURCES.json", resources)
        if fault != "tampered_map":
            old["files_sha256"]["YAP/runtime/RESOURCES.json"] = build.digest(baseline / "runtime/RESOURCES.json")
    elif fault == "oversized_files":
        old["files_sha256"] = {f"YAP/runtime/native/file{i}.dat": "a" * 64 for i in range(513)}
    else:
        path = baseline.parent / next(key for key in old["files_sha256"] if key.endswith("file0.dat"))
        path.unlink()
        path.mkdir()
    build.write_json(baseline / "runtime/BUNDLE.json", old)
    with pytest.raises(ValueError):
        build.stage_preflight(args)
    assert not (args.work / "delivery_source").exists()
    assert not (args.work / "runtime").exists()


def test_stage_preflight_returns_only_checked_explicit_files(tmp_path, monkeypatch):
    args, _, _ = stage_fixture(tmp_path, monkeypatch)
    _, destination, _, selected, mappings, total, metadata = build.stage_preflight(args)
    assert len(selected) == 17 and len(metadata) == 3
    assert set(mappings) == {"compact_p2", "geometry"}
    assert total > 0 and not destination.exists()
    for source, _, expected in selected:
        assert build.digest(source) == expected


def legacy_stage_fixture(tmp_path, monkeypatch):
    args, old, resources = stage_fixture(tmp_path, monkeypatch)
    old["delivery_revision"] = 3
    del old["files_sha256"]["YAP/runtime/RESOURCES.json"]
    (args.baseline_runtime / "runtime/RESOURCES.json").unlink()
    pins = {}
    for key, resource in (("compact", "compact_p2"), ("parent", "geometry")):
        relative = f'YAP/runtime/native/{resources["resources"][resource]}/packages/BI2_50.json'
        path = args.baseline_runtime.parent / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic {key} endpoint")
        pins[key] = old["files_sha256"][relative] = build.digest(path)
    build.write_json(args.baseline_runtime / "runtime/BUNDLE.json", old)
    new = build.read_json(args.archive / "ARCHIVE.json")
    parent = args.archive / "shared/parent_BI2_50.json"
    parent.parent.mkdir()
    parent.write_text("synthetic parent endpoint")
    new["files_sha256"]["shared/parent_BI2_50.json"] = pins["parent"]
    # Match the synthetic parent bytes in both inventories.
    native_parent = args.baseline_runtime / f'runtime/native/{resources["resources"]["geometry"]}/packages/BI2_50.json'
    parent.write_bytes(native_parent.read_bytes())
    build.write_json(args.archive / "ARCHIVE.json", new)
    args.legacy_resource_map = tmp_path / "private_resources.json"
    build.write_json(args.legacy_resource_map, resources)
    pins.update(bundle=build.digest(args.baseline_runtime / "runtime/BUNDLE.json"),
                resources=build.digest(args.legacy_resource_map),
                archive=build.digest(args.archive / "ARCHIVE.json"))
    monkeypatch.setattr(build, "LEGACY_STAGE_PINS", pins)
    return args, old, new


def test_legacy_stage_authenticates_external_map_without_rewriting_baseline(tmp_path, monkeypatch):
    args, old, _ = legacy_stage_fixture(tmp_path, monkeypatch)
    _, destination, _, _, _, _, metadata = build.stage_preflight(args)
    assert destination.name == "delivery_source_legacy_v5"
    assert args.legacy_resource_map in metadata
    assert build.read_json(args.baseline_runtime / "runtime/BUNDLE.json") == old
    assert not destination.exists()
    assert not (args.baseline_runtime / "runtime/RESOURCES.json").exists()


@pytest.mark.parametrize("fault", ["bundle", "archive", "map", "entry", "file", "dangling", "endpoint", "parent", "map_alias"])
def test_legacy_stage_never_falls_back_or_weakens_pins(tmp_path, monkeypatch, fault):
    args, old, new = legacy_stage_fixture(tmp_path, monkeypatch)
    pins = dict(build.LEGACY_STAGE_PINS)
    map_path = args.baseline_runtime / "runtime/RESOURCES.json"
    if fault in ("bundle", "archive", "map"):
        pins["resources" if fault == "map" else fault] = "0" * 64
    elif fault == "entry":
        old["files_sha256"]["YAP/runtime/RESOURCES.json"] = "0" * 64
        build.write_json(args.baseline_runtime / "runtime/BUNDLE.json", old)
        pins["bundle"] = build.digest(args.baseline_runtime / "runtime/BUNDLE.json")
    elif fault == "file":
        map_path.write_text("invalid existing modern map")
    elif fault == "dangling":
        map_path.symlink_to(tmp_path / "missing")
    elif fault == "endpoint":
        pins["compact"] = "0" * 64
    elif fault == "parent":
        new["files_sha256"]["shared/parent_BI2_50.json"] = "0" * 64
        build.write_json(args.archive / "ARCHIVE.json", new)
        pins["archive"] = build.digest(args.archive / "ARCHIVE.json")
    else:
        alias = tmp_path / "map_alias.json"
        alias.symlink_to(args.legacy_resource_map)
        args.legacy_resource_map = alias
    monkeypatch.setattr(build, "LEGACY_STAGE_PINS", pins)
    with pytest.raises(ValueError):
        build.stage_preflight(args)
    assert not (args.work / "delivery_source_legacy_v5").exists()


def test_stage_rechecks_metadata_after_copy(tmp_path, monkeypatch):
    args, _, _ = stage_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(build, "configure", lambda _: None)
    original = build.copy_checked
    def mutated_copy(source, target, expected=None):
        result = original(source, target, expected)
        (args.baseline_runtime / "runtime/BUNDLE.json").write_text("changed after preflight")
        return result
    monkeypatch.setattr(build, "copy_checked", mutated_copy)
    with pytest.raises(ValueError, match="inventory changed"):
        build.stage_delivery(args)
    assert not (args.work / "delivery_source/STAGE.json").exists()


def test_application_source_cannot_overlap_writable_staging_roots(tmp_path, monkeypatch):
    args, _, _ = stage_fixture(tmp_path, monkeypatch)
    args.application_source = args.work
    monkeypatch.setattr(build, "application_files", lambda _: pytest.fail("reject before source enumeration"))
    with pytest.raises(ValueError, match="application source overlaps"):
        build.stage_preflight(args)
    assert not (args.work / "delivery_source").exists()


def test_legacy_external_map_rejects_tmp_before_open(tmp_path, monkeypatch):
    args, _, _ = legacy_stage_fixture(tmp_path, monkeypatch)
    args.legacy_resource_map = Path("/tmp/forbidden_input.json")
    original = build.checked_inventory
    def no_tmp_read(root, path):
        assert not Path(path).is_relative_to("/tmp")
        return original(root, path)
    monkeypatch.setattr(build, "checked_inventory", no_tmp_read)
    with pytest.raises(ValueError, match="forbidden storage"):
        build.stage_preflight(args)


@pytest.mark.parametrize("executable", [
    "source/package_assembly_runtime.py", "source/tools/package_assembly_runtime.py",
])
def test_export_launcher_rejects_unpinned_code_before_import(tmp_path, monkeypatch, executable):
    work = tmp_path / "work"
    root = work / "delivery_source_legacy_v5"
    source = root / executable
    source.parent.mkdir(parents=True)
    source.write_text("raise RuntimeError('must never execute')")
    receipt = root / "STAGE.json"
    build.write_json(receipt, {"schema": "yap.prepared_runtime_stage.v1", "application_source": "source",
                              "files_sha256": {executable: "0" * 64}})
    monkeypatch.setattr(build, "OWNED_WORK_ROOT", tmp_path)
    monkeypatch.setattr(build, "configure", lambda _: pytest.fail("no environment or runtime writes before executable pins"))
    monkeypatch.setattr(build, "run_stage", lambda *_: pytest.fail("no executable import"))
    args = SimpleNamespace(work=work, stage_inventory_sha256=build.digest(receipt), archive=tmp_path / "out")
    with pytest.raises(ValueError, match="executable or configuration input changed"):
        build.export_staged(args)


@pytest.mark.parametrize("executable", [
    "source/package_assembly_runtime.py", "source/tools/package_assembly_runtime.py",
])
def test_exporter_selection_preserves_the_receipt_layout(tmp_path, executable):
    assert build.staged_exporter(tmp_path, {executable: "a" * 64}) == tmp_path / executable


@pytest.mark.parametrize("files", [
    {}, {"source/unlisted.py": "a" * 64},
    {"source/package_assembly_runtime.py": "a" * 64,
     "source/tools/package_assembly_runtime.py": "b" * 64},
])
def test_exporter_selection_rejects_missing_or_ambiguous_bindings(tmp_path, files):
    with pytest.raises(ValueError, match="exactly one exporter"):
        build.staged_exporter(tmp_path, files)


@pytest.mark.parametrize("executable", [
    "source/package_assembly_runtime.py", "source/tools/package_assembly_runtime.py",
])
def test_frozen_exporter_launch_uses_application_root_for_both_layouts(tmp_path, monkeypatch, executable):
    work = tmp_path / "work"
    root = work / "delivery_source_legacy_v5"
    source = root / executable
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic exporter; never executed\n")
    receipt = root / "STAGE.json"
    build.write_json(receipt, dict(schema="yap.prepared_runtime_stage.v1", application_source="source",
                                  files_sha256={executable: build.digest(source)}))
    original = receipt.read_bytes()
    destination = Path("/app/nanocad/projects/personal/zchen/YAP_STACK/distribution/yap_prepared_p2_20261006_v1")
    exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda p: False if p == destination else exists(p))
    monkeypatch.setattr(build, "OWNED_WORK_ROOT", tmp_path)
    monkeypatch.setattr(build, "configure", lambda _: {})
    monkeypatch.setattr(build, "resource_check", lambda _: {"fixture": True})

    class LaunchCaptured(Exception):
        pass

    def capture(command, *_args, **_kwargs):
        assert command[1:4] == ["-I", "-B", "-c"]
        assert command[5] == str(root / "source")
        assert command[6] == str(source)
        raise LaunchCaptured

    monkeypatch.setattr(build, "run_stage", capture)
    args = SimpleNamespace(work=work, stage_inventory_sha256=build.digest(receipt), archive=destination)
    with pytest.raises(LaunchCaptured):
        build.export_staged(args)
    assert receipt.read_bytes() == original


def test_stage_rejects_runtime_alias_before_disk_or_inventory_access(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    (work / "runtime").symlink_to(tmp_path / "immutable-source", target_is_directory=True)
    monkeypatch.setattr(build, "OWNED_WORK_ROOT", tmp_path)
    monkeypatch.setattr(build, "resource_check", lambda _: pytest.fail("alias must be checked before disk access"))
    args = SimpleNamespace(work=work, baseline_runtime=tmp_path / "old", archive=tmp_path / "archive")
    with pytest.raises(ValueError, match="symlinks"):
        build.stage_preflight(args)


def test_stage_rejects_outside_owned_root_before_disk_access(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "resource_check", lambda _: pytest.fail("owned root must be checked first"))
    args = SimpleNamespace(work=tmp_path / "work", baseline_runtime=tmp_path / "old", archive=tmp_path / "archive")
    with pytest.raises(ValueError, match="owned local scratch"):
        build.stage_preflight(args)
