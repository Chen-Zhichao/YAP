"""User-storage policy tests; no operators or Cu histories are run."""

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
POLICY_ERRORS = (ValueError, PermissionError)


@pytest.fixture
def runtime(monkeypatch):
    # Isolate storage globals from other runtime tests in the same process.
    name = "_yap_user_storage_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "mechanics_runtime.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def storage(monkeypatch, tmp_path):
    assert tmp_path.is_relative_to("/app"), "run with a private /app --basetemp"
    case = tmp_path
    root = case / "run"
    # configure_runtime_storage mutates these globals directly, so isolate them.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for key in CACHE_KEYS + ("YAP_RUN_ROOT", "YAP_W2W_SI_CAP_CACHE_DIR"):
        os.environ.pop(key, None)
    os.environ["YAP_RUN_ROOT"] = str(root)
    os.environ["HOME"] = "/home/inherited-home-must-not-be-written"
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    previous_umask = os.umask(0o077)
    os.umask(previous_umask)
    try:
        yield case, root
    finally:
        os.umask(previous_umask)


def forbid_mkdir(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid configuration attempted mkdir before validation")

    monkeypatch.setattr(Path, "mkdir", unexpected)
    monkeypatch.setattr(os, "mkdir", unexpected)


def fake_owner(monkeypatch, target):
    original_stat = Path.stat

    def changed(path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
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
    with pytest.raises(POLICY_ERRORS):
        runtime.configure_runtime_storage()
    assert list(case.iterdir()) == []


@pytest.mark.parametrize("value", ["", "relative/run", "~/implicit-home"])
def test_root_must_be_explicit_absolute(runtime, storage, monkeypatch, value):
    os.environ["YAP_RUN_ROOT"] = value
    forbid_mkdir(monkeypatch)
    with pytest.raises(POLICY_ERRORS):
        runtime.configure_runtime_storage()


def test_validators_have_no_creation_side_effects(runtime, storage, monkeypatch):
    case, root = storage
    forbid_mkdir(monkeypatch)
    assert runtime.user_run_root() == root
    assert runtime.validate_user_write_path(root / "results" / "RESULT.json") == (
        root / "results" / "RESULT.json"
    )
    assert list(case.iterdir()) == []


@pytest.mark.parametrize("prefix", ["/tmp", "/w", "/u1", "/home", "/data", "/var/tmp"])
def test_root_rejects_non_app_storage(runtime, storage, monkeypatch, prefix):
    os.environ["YAP_RUN_ROOT"] = prefix + "/yap-user-storage-forbidden"
    forbid_mkdir(monkeypatch)
    with pytest.raises(POLICY_ERRORS):
        runtime.configure_runtime_storage()


@pytest.mark.parametrize("part", ["#snapshot", ".snapshot"])
def test_snapshot_root_is_rejected_before_creation(runtime, storage, monkeypatch, part):
    case, _ = storage
    os.environ["YAP_RUN_ROOT"] = str(case / part / "run")
    forbid_mkdir(monkeypatch)
    with pytest.raises(POLICY_ERRORS):
        runtime.configure_runtime_storage()


@pytest.mark.parametrize("protected", [
    PROJECT / "YAP",
    PROJECT / "YAP" / ".cache" / "user-storage-forbidden",
    ROOT,
    ROOT / "tests" / "user-storage-forbidden",
    PROJECT / "distribution" / "user-storage-forbidden",
    PROJECT / "operator_repair_20261014" / "user-storage-forbidden",
    PROJECT / "operator_repair_20261015" / "user-storage-forbidden",
])
def test_source_and_operator_roots_are_never_writable(
        runtime, storage, monkeypatch, protected):
    os.environ["YAP_RUN_ROOT"] = str(protected)
    forbid_mkdir(monkeypatch)
    with pytest.raises(POLICY_ERRORS):
        runtime.configure_runtime_storage()


def test_write_path_must_be_below_selected_root(runtime, storage):
    case, root = storage
    with pytest.raises(POLICY_ERRORS):
        runtime.validate_user_write_path(case / "another-user" / "RESULT.json")
    with pytest.raises(POLICY_ERRORS):
        runtime.validate_user_write_path(Path(str(root) + "-lookalike") / "RESULT.json")
    with pytest.raises(POLICY_ERRORS):
        runtime.validate_user_write_path(root / ".." / "escaped.json")


@pytest.mark.parametrize("value", ["", "relative/RESULT.json"])
def test_write_paths_must_be_nonempty_absolute(runtime, storage, value):
    with pytest.raises(POLICY_ERRORS):
        runtime.validate_user_write_path(value)


@pytest.mark.parametrize("part", ["#snapshot", ".snapshot"])
def test_snapshot_descendant_is_rejected(runtime, storage, part):
    _, root = storage
    with pytest.raises(POLICY_ERRORS):
        runtime.validate_user_write_path(root / part / "RESULT.json")


def test_configure_creates_private_storage_and_rebases_home(runtime, storage):
    _, root = storage
    result = runtime.configure_runtime_storage()
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert result["work"] == root / "work"
    assert Path(os.environ["YAP_WORK_DIR"]) == result["work"]
    # Bytecode is disabled; an unset pycache prefix need not allocate a directory.
    required_keys = tuple(key for key in CACHE_KEYS if key != "PYTHONPYCACHEPREFIX")
    for key in required_keys + ("HOME",):
        path = Path(os.environ[key])
        assert path.is_relative_to(root), key
        assert path.is_dir(), key
        assert path.stat().st_uid == os.getuid(), key
        assert not path.stat().st_mode & 0o022, key
        assert not path.is_symlink(), key
    assert tempfile.tempdir == os.environ["TMPDIR"]
    assert sys.dont_write_bytecode is True
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    assert runtime.configure_runtime_storage() == result


@pytest.mark.parametrize("key", CACHE_KEYS)
def test_conflicting_cache_override_fails_before_any_creation(
        runtime, storage, monkeypatch, key):
    case, _ = storage
    os.environ[key] = str(case / "outside-selected-root" / key.lower())
    forbid_mkdir(monkeypatch)
    with pytest.raises(POLICY_ERRORS):
        runtime.configure_runtime_storage()
    assert list(case.iterdir()) == []


def test_valid_explicit_cache_override_is_preserved(runtime, storage):
    _, root = storage
    os.environ["MPLCONFIGDIR"] = str(root / "custom-matplotlib")
    runtime.configure_runtime_storage()
    assert Path(os.environ["MPLCONFIGDIR"]) == root / "custom-matplotlib"


@pytest.mark.parametrize("mode", [0o770, 0o777, 0o702])
def test_other_writable_root_is_rejected(runtime, storage, monkeypatch, mode):
    _, root = storage
    root.mkdir(mode=0o700)
    root.chmod(mode)
    forbid_mkdir(monkeypatch)
    with pytest.raises(POLICY_ERRORS):
        runtime.configure_runtime_storage()


def test_foreign_owned_root_is_rejected(runtime, storage, monkeypatch):
    _, root = storage
    root.mkdir(mode=0o700)
    fake_owner(monkeypatch, root)
    forbid_mkdir(monkeypatch)
    with pytest.raises(POLICY_ERRORS):
        runtime.configure_runtime_storage()


def test_owner_check_uses_real_uid_not_environment_name(runtime, storage, monkeypatch):
    _, root = storage
    root.mkdir(mode=0o700)
    os.environ["USER"] = "not-the-owner-name"
    os.environ["LOGNAME"] = "not-the-owner-name"
    assert runtime.user_run_root() == root
    uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: uid + 1)
    with pytest.raises(POLICY_ERRORS):
        runtime.user_run_root()


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_foreign_owned_descendant_is_rejected(runtime, storage, monkeypatch, kind):
    _, root = storage
    root.mkdir(mode=0o700)
    target = root / "foreign"
    if kind == "directory":
        target.mkdir(mode=0o700)
        destination = target / "RESULT.json"
    else:
        target.write_text("unchanged\n", encoding="ascii")
        destination = target
    fake_owner(monkeypatch, target)
    with pytest.raises(POLICY_ERRORS):
        runtime.validate_user_write_path(destination)


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_other_writable_descendant_is_rejected(runtime, storage, kind):
    _, root = storage
    root.mkdir(mode=0o700)
    target = root / "unsafe"
    if kind == "directory":
        target.mkdir(mode=0o700)
        destination = target / "RESULT.json"
    else:
        target.write_text("unchanged\n", encoding="ascii")
        destination = target
    target.chmod(0o777 if kind == "directory" else 0o666)
    with pytest.raises(POLICY_ERRORS):
        runtime.validate_user_write_path(destination)


def test_symlink_run_root_is_rejected(runtime, storage, monkeypatch):
    case, root = storage
    actual = case / "actual"
    actual.mkdir(mode=0o700)
    root.symlink_to(actual, target_is_directory=True)
    forbid_mkdir(monkeypatch)
    with pytest.raises(POLICY_ERRORS):
        runtime.configure_runtime_storage()


@pytest.mark.parametrize("target_kind", ["inside", "outside", "dangling", "file"])
def test_symlink_descendants_are_rejected(runtime, storage, target_kind):
    case, root = storage
    root.mkdir(mode=0o700)
    alias = root / "alias"
    target = root / "actual" if target_kind == "inside" else case / "outside"
    if target_kind == "file":
        target.write_text("unchanged\n", encoding="ascii")
    elif target_kind != "dangling":
        target.mkdir(mode=0o700)
    alias.symlink_to(target, target_is_directory=target_kind != "file")
    destination = alias if target_kind == "file" else alias / "RESULT.json"
    with pytest.raises(POLICY_ERRORS):
        runtime.validate_user_write_path(destination)


def test_external_app_root_is_accepted_without_writing_it(runtime, storage, monkeypatch):
    _, local_root = storage
    local_root.mkdir(mode=0o700)
    owned_stat = local_root.stat()
    external = Path("/app/yap-storage-test-owner") / ("uid-" + str(os.getuid()))
    os.environ["YAP_RUN_ROOT"] = str(external)
    original_stat = Path.stat
    original_access = os.access

    def synthetic_stat(path, *args, **kwargs):
        if path == external:
            return owned_stat
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", synthetic_stat)
    monkeypatch.setattr(os, "access", lambda path, mode, **kwargs: (
        True if Path(path) == external else original_access(path, mode, **kwargs)
    ))
    forbid_mkdir(monkeypatch)
    assert runtime.user_run_root() == external
    assert runtime.validate_user_write_path(external / "results" / "RESULT.json") == (
        external / "results" / "RESULT.json"
    )
