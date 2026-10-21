"""Git delivery includes declared profiles without generated or private data."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROFILES = (
    ("D2W", "design_6_4hi_p2"),
    ("D2W", "design_6_4hi_p2_t30"),
    ("D2W", "design_6_4hi_p2_t80"),
    ("W2W", "design_6_4hi_p2"),
    ("W2W", "design_6_4hi_p6_reference"),
)


@pytest.fixture(scope="module")
def policy_repo(tmp_path_factory):
    repo = tmp_path_factory.mktemp("gitignore_policy")
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True,
                   capture_output=True, timeout=10)
    shutil.copyfile(ROOT / ".gitignore", repo / ".gitignore")
    return repo


def ignored_paths(repo, paths):
    result = subprocess.run(
        ["git", "-C", str(repo), "check-ignore", "--no-index", "-z", "--stdin"],
        input=("\0".join(paths) + "\0").encode(), capture_output=True, timeout=10,
    )
    assert result.returncode in (0, 1), result.stderr.decode()
    return set(result.stdout.decode().split("\0")) - {""}


@pytest.mark.parametrize("domain,name", PROFILES)
def test_complete_compact_profiles_are_publishable(policy_repo, domain, name):
    folder = f"{domain}/input/{name}"
    body = "Die" if domain == "D2W" else "Wafer"
    paths = [f"{folder}/{filename}" for filename in (
        "stack.yaml", "compact_layout.yaml", "generated_chiplet_definitions.3dbv",
        "generated_stack_config.3dbx", *(f"{body}_{i}.3dbf" for i in range(1, 5)),
    )]
    paths.append(f"{domain}/configs/{name}/{name}.yaml")
    assert all((ROOT / path).is_file() for path in paths)
    assert not ignored_paths(policy_repo, paths)


@pytest.mark.parametrize("domain,name", PROFILES)
def test_profile_allowlists_exclude_generated_extras(policy_repo, domain, name):
    paths = [
        f"{domain}/input/{name}/{filename}"
        for filename in ("TRACE.json", "large.bmap", "generated.npy", "operator.npz")
    ]
    paths.extend(f"{domain}/configs/{name}/{filename}"
                 for filename in ("resolved.yaml", "generated_interface.yaml"))
    assert ignored_paths(policy_repo, paths) == set(paths)


def test_source_test_references_and_repository_metadata_are_publishable(policy_repo):
    paths = [
        ".gitignore", ".gitattributes", ".github/workflows/tests.yml",
        "README.md", "PORTABLE_ASSEMBLY.md", "requirements.txt", "LICENSE",
        "mechanics_runtime.py", "history_mechanics_backend.py",
        "package_assembly_runtime.py", "verify_assembly_runtime.py",
        "W2W/assembly_mechanics.py", "W2W/assembly_operator_library.py",
        "D2W/assembly_order.py", "D2W/esd_yield_calculator.py",
        "W2W/esd_yield_calculator.py", "tests/test_gitignore.py",
        "D2W/input/.gitkeep", "W2W/input/.gitkeep",
        "D2W/configs/design_6_4hi_p6/design_6_4hi_p6.yaml",
        "W2W/configs/design_6_4hi_p6/design_6_4hi_p6.yaml",
    ]
    assert not ignored_paths(policy_repo, paths)


@pytest.mark.parametrize("path", [
    "output/result.json", "D2W/output/new_run/RESULT.json",
    "D2W/output/case_studies/REPORT.md", "W2W/output/nested/PLAN.json",
    "W2W/output/nested/SUMMARY.csv", "paper_results/figure.json",
    "runtime/native/exp12_layer_rom_dataset.py", "runtime/native/data.npz",
    "ASSEMBLY_RUNTIME.json", "operators/model.npz", "operators/model.npy",
    "operators/model.joblib", "operators/model.hdf5", "run/solution.rst",
    "run/solution.page", "delivery.tar", "delivery.zip", "temp/trace.json",
    "run.log", "run.out", "__pycache__/model.pyc", "venv/bin/python",
    ".pytest_cache/state", ".venv/bin/python", ".vscode/settings.json",
    ".codex/session.json", ".env", ".env.local", ".github/.env",
    ".runtime_resources.json", "runtime/RESOURCES.json",
    "private.key", "private.pem", "#snapshot/file.py", ".snapshot/file.py",
    "D2W/calculator_main.ipynb", "large.bmap", "D2W/input/design_6/large.bmap",
    "D2W/configs/design_6_4hi_p6/generated_interface.yaml",
    "W2W/input/design_6_4hi_p2_no_thinning/large.bmap",
    "W2W/input/design_6_4hi_p2_no_thinning/TRACE.json",
    "W2W/input/design_6_4hi_p2_no_thinning/generated.npy",
    "W2W/configs/design_6_4hi_p2_no_thinning/resolved.yaml",
])
def test_generated_private_and_unapproved_inputs_are_ignored(policy_repo, path):
    assert ignored_paths(policy_repo, [path]) == {path}
