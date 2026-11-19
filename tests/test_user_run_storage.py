"""Private output defaults and configurable storage; no mechanics are run."""

import importlib.util
import os
from pathlib import Path
import stat
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path("/app/nanocad/projects/personal/zchen/YAP_STACK")
CACHE_KEYS = (
    "YAP_WORK_DIR", "TMPDIR", "TMP", "TEMP", "MPLCONFIGDIR",
    "NUMBA_CACHE_DIR", "PYTHONPYCACHEPREFIX", "PIP_CACHE_DIR",
    "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
    "XDG_STATE_HOME", "XDG_RUNTIME_DIR", "PYTHONUSERBASE", "JOBLIB_TEMP_FOLDER",
)


@pytest.fixture
def runtime(monkeypatch):
    name = "_yap_user_storage_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "mechanics_runtime.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def storage(monkeypatch, tmp_path):
    assert tmp_path.is_relative_to(Path(os.environ["YAP_RUN_ROOT"]).resolve())
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for key in CACHE_KEYS + ("YAP_RUN_ROOT", "YAP_W2W_SI_CAP_CACHE_DIR"):
        os.environ.pop(key, None)
    root = tmp_path / "run"
    os.environ["YAP_RUN_ROOT"] = str(root)
    os.environ["HOME"] = str(tmp_path / "inherited-home")
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    previous_umask = os.umask(0o077)
    os.umask(previous_umask)
    try:
        yield tmp_path, root
    finally:
        os.umask(previous_umask)


def forbid_mkdir(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("validation unexpectedly attempted mkdir")
    monkeypatch.setattr(Path, "mkdir", unexpected)
    monkeypatch.setattr(os, "mkdir", unexpected)


def fake_owner(monkeypatch, target):
    original = Path.stat

    def changed(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path == target:
            fields = list(result)
            fields[4] = os.getuid() + 1
            return os.stat_result(fields)
        return result
    monkeypatch.setattr(Path, "stat", changed)


def test_missing_root_fails_before_any_mkdir(runtime, storage, monkeypatch):
    case, _ = storage
    os.environ.pop("YAP_RUN_ROOT")
    forbid_mkdir(monkeypatch)
    with pytest.raises(ValueError):
        runtime.configure_runtime_storage()
    assert list(case.iterdir()) == []


@pytest.mark.parametrize("value", ["", "relative/run", "/", "/app/../escaped"])
def test_root_must_be_explicit_absolute(runtime, storage, monkeypatch, value):
    os.environ["YAP_RUN_ROOT"] = value
    forbid_mkdir(monkeypatch)
    with pytest.raises(ValueError):
        runtime.configure_runtime_storage()


def test_explicit_home_expansion_does_not_create_files(runtime, storage, monkeypatch):
    case, _ = storage
    os.environ["YAP_RUN_ROOT"] = "~/chosen-run"
    forbid_mkdir(monkeypatch)
    assert runtime.user_run_root() == case / "inherited-home" / "chosen-run"
    assert list(case.iterdir()) == []


def test_validators_have_no_creation_side_effects(runtime, storage, monkeypatch):
    case, root = storage
    forbid_mkdir(monkeypatch)
    assert runtime.user_run_root() == root
    assert runtime.validate_user_write_path(root / "results" / "RESULT.json") == (
        root / "results" / "RESULT.json")
    assert list(case.iterdir()) == []


@pytest.mark.parametrize("prefix", ["/tmp", "/w", "/u1", "/home", "/data", "/var/tmp"])
def test_non_app_selection_warns_without_creating_files(runtime, storage, monkeypatch, prefix):
    path = Path(prefix) / "yap-user-storage-validation-only"
    os.environ["YAP_RUN_ROOT"] = str(path)
    forbid_mkdir(monkeypatch)
    with pytest.warns(runtime.StorageWarning, match="outside the recommended"):
        assert runtime.user_run_root() == path.resolve()


@pytest.mark.parametrize("part", ["#snapshot", ".snapshot"])
def test_snapshot_root_is_rejected_before_creation(runtime, storage, monkeypatch, part):
    case, _ = storage
    os.environ["YAP_RUN_ROOT"] = str(case / part / "run")
    forbid_mkdir(monkeypatch)
    with pytest.raises(ValueError, match="snapshot"):
        runtime.configure_runtime_storage()


@pytest.mark.parametrize("protected", [
    ROOT, ROOT / "tests" / "must-not-write",
    PROJECT / "distribution" / "must-not-write",
    PROJECT / "operator_repair_20261015" / "must-not-write",
    PROJECT / "operator_night_20261021" / "must-not-write",
])
def test_source_and_operator_roots_are_never_writable(
        runtime, storage, monkeypatch, protected):
    os.environ["YAP_RUN_ROOT"] = str(protected)
    forbid_mkdir(monkeypatch)
    with pytest.raises(ValueError, match="read-only"):
        runtime.configure_runtime_storage()


def test_outside_run_root_warns_but_traversal_is_rejected(runtime, storage, monkeypatch):
    case, root = storage
    forbid_mkdir(monkeypatch)
    for destination in (case / "separate-results" / "RESULT.json",
                        Path(str(root) + "-lookalike") / "RESULT.json"):
        with pytest.warns(runtime.StorageWarning, match="outside YAP_RUN_ROOT"):
            assert runtime.validate_user_write_path(destination) == destination
    with pytest.raises(ValueError, match="traversal"):
        runtime.validate_user_write_path(root / ".." / "escaped.json")


@pytest.mark.parametrize("value", ["", "relative/RESULT.json"])
def test_write_paths_must_be_nonempty_absolute(runtime, storage, value):
    with pytest.raises(ValueError):
        runtime.validate_user_write_path(value)


@pytest.mark.parametrize("part", ["#snapshot", ".snapshot"])
def test_snapshot_descendant_is_rejected(runtime, storage, part):
    _, root = storage
    with pytest.raises(ValueError, match="snapshot"):
        runtime.validate_user_write_path(root / part / "RESULT.json")


def test_configure_creates_private_storage_and_rebases_home(runtime, storage):
    _, root = storage
    result = runtime.configure_runtime_storage()
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert result["work"] == root / "work"
    for key in tuple(k for k in CACHE_KEYS if k != "PYTHONPYCACHEPREFIX") + ("HOME",):
        path = Path(os.environ[key])
        assert path.is_relative_to(root) and path.is_dir(), key
        assert path.stat().st_uid == os.getuid() and not path.stat().st_mode & 0o022
    assert tempfile.tempdir == os.environ["TMPDIR"]
    assert sys.dont_write_bytecode is True
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    assert runtime.configure_runtime_storage() == result


@pytest.mark.parametrize("key", CACHE_KEYS)
def test_independent_cache_override_is_preserved(runtime, storage, key):
    case, _ = storage
    destination = case / "separate-cache" / key.lower()
    os.environ[key] = str(destination)
    with pytest.warns(runtime.StorageWarning, match="outside YAP_RUN_ROOT"):
        runtime.configure_runtime_storage()
    assert Path(os.environ[key]) == destination
    if key != "PYTHONPYCACHEPREFIX":
        assert destination.is_dir()


@pytest.mark.parametrize("mode", [0o770, 0o777, 0o702])
def test_shared_root_warns_without_chmod(runtime, storage, monkeypatch, mode):
    _, root = storage
    root.mkdir(mode=0o700)
    root.chmod(mode)
    forbid_mkdir(monkeypatch)
    with pytest.warns(runtime.StorageWarning, match="not a private"):
        assert runtime.user_run_root() == root
    assert stat.S_IMODE(root.stat().st_mode) == mode


def test_foreign_owned_root_warns_without_creation(runtime, storage, monkeypatch):
    _, root = storage
    root.mkdir(mode=0o700)
    fake_owner(monkeypatch, root)
    forbid_mkdir(monkeypatch)
    with pytest.warns(runtime.StorageWarning, match="not a private"):
        assert runtime.user_run_root() == root


def test_owner_warning_uses_uid_not_environment_name(runtime, storage, monkeypatch):
    _, root = storage
    root.mkdir(mode=0o700)
    os.environ["USER"] = "not-the-owner-name"
    os.environ["LOGNAME"] = "not-the-owner-name"
    assert runtime.user_run_root() == root
    uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: uid + 1)
    with pytest.warns(runtime.StorageWarning, match="not a private"):
        assert runtime.user_run_root() == root


@pytest.mark.parametrize("kind", ["directory", "file"])
@pytest.mark.parametrize("foreign", [False, True])
def test_untrusted_descendant_warns_without_modifying_it(
        runtime, storage, monkeypatch, kind, foreign):
    _, root = storage
    root.mkdir(mode=0o700)
    target = root / "shared"
    if kind == "directory":
        target.mkdir(mode=0o700)
        destination = target / "RESULT.json"
    else:
        target.write_text("unchanged\n", encoding="ascii")
        destination = target
    if foreign:
        fake_owner(monkeypatch, target)
    else:
        target.chmod(0o777 if kind == "directory" else 0o666)
    mode = stat.S_IMODE(target.stat().st_mode)
    forbid_mkdir(monkeypatch)
    with pytest.warns(runtime.StorageWarning, match="shared or owned"):
        assert runtime.validate_user_write_path(destination) == destination
    assert stat.S_IMODE(target.stat().st_mode) == mode
    if kind == "file":
        assert target.read_text() == "unchanged\n"


def test_unwritable_root_is_still_rejected(runtime, storage, monkeypatch):
    _, root = storage
    root.mkdir(mode=0o700)
    monkeypatch.setattr(os, "access", lambda *args, **kwargs: False)
    forbid_mkdir(monkeypatch)
    with pytest.raises(PermissionError, match="not writable"):
        runtime.user_run_root()


def test_symlink_run_root_resolves_without_creation(runtime, storage, monkeypatch):
    case, root = storage
    actual = case / "actual"
    actual.mkdir(mode=0o700)
    root.symlink_to(actual, target_is_directory=True)
    forbid_mkdir(monkeypatch)
    with pytest.warns(runtime.StorageWarning, match="resolved storage path"):
        assert runtime.user_run_root() == actual


@pytest.mark.parametrize("target_kind", ["inside", "outside", "dangling", "file"])
def test_symlink_descendant_resolves_without_writing(runtime, storage, monkeypatch, target_kind):
    case, root = storage
    root.mkdir(mode=0o700)
    target = root / "actual" if target_kind == "inside" else case / "outside"
    if target_kind == "file":
        target.write_text("unchanged\n", encoding="ascii")
    elif target_kind != "dangling":
        target.mkdir(mode=0o700)
    alias = root / "alias"
    alias.symlink_to(target, target_is_directory=target_kind != "file")
    destination = alias if target_kind == "file" else alias / "RESULT.json"
    forbid_mkdir(monkeypatch)
    with pytest.warns(runtime.StorageWarning, match="resolved storage path"):
        assert runtime.validate_user_write_path(destination) == destination.resolve()
    if target_kind == "file":
        assert target.read_text() == "unchanged\n"


@pytest.mark.parametrize("part", ["#snapshot", ".snapshot", "YAP"])
def test_symlink_cannot_bypass_readonly_or_snapshot_guard(
        runtime, storage, monkeypatch, part):
    case, root = storage
    root.mkdir(mode=0o700)
    target = ROOT if part == "YAP" else case / part
    alias = root / "alias"
    alias.symlink_to(target, target_is_directory=True)
    forbid_mkdir(monkeypatch)
    with pytest.raises(ValueError, match="read-only|snapshot"):
        runtime.validate_user_write_path(alias / "must-not-write")
