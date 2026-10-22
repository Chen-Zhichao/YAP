"""Relocated offline tools keep CLI, source-pin and delivery contracts."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import tools.build_prepared_operators as builder
import tools.operator_grid as grid
import tools.operator_thermal_validation as thermal


ROOT = Path(__file__).resolve().parents[1]
MOVED = (
    "build_corrected_compact_mechanics", "build_prepared_operators",
    "fabrication_initial_bow_study", "generate_stack_variants",
    "operator_accuracy_pool", "operator_accuracy_review", "operator_coverage",
    "operator_frozen_residual", "operator_grid", "operator_reference_repair",
    "operator_snapshot_diagnostic", "operator_thermal_validation", "operator_work_reference",
    "package_assembly_runtime", "publish_component_operators", "verify_assembly_runtime",
)
CLI = tuple(name for name in MOVED if name not in (
    "operator_frozen_residual", "operator_work_reference"))


def test_offline_tools_have_one_location_without_root_wrappers():
    for name in MOVED:
        assert (ROOT / "tools" / f"{name}.py").is_file()
        assert not (ROOT / f"{name}.py").exists()
        module = importlib.import_module(f"tools.{name}")
        assert Path(module.__file__) == ROOT / "tools" / f"{name}.py"
    for name in ("generate_assembly_design", "mechanics_runtime", "component_operators",
                 "thickness_operators"):
        assert (ROOT / f"{name}.py").is_file()


@pytest.mark.parametrize("name", CLI)
def test_direct_cli_from_unrelated_directory(name, tmp_path):
    env = dict(os.environ, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "tools" / f"{name}.py"), "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()


def test_module_cli_from_checkout():
    result = subprocess.run(
        [sys.executable, "-B", "-m", "tools.operator_grid", "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    directory = checkout / "tools"
    directory.mkdir(parents=True)
    source = tmp_path / "native"
    source.mkdir()
    common = tmp_path / "common"
    common.mkdir()
    (common / "__init__.py").write_text("")
    (common / "ansys_utils.py").write_text("# synthetic helper\n")
    names = {grid.TRAINING_NATIVE, grid.TRAINING_OUTPUTS, grid.TRAINING_EXTRACTOR,
             "exp12_250c_branch.py", thermal.TEACHER,
             "run_exp12_m4_all_gauss_offpath_layer_gate.py",
             "exp12_all_gauss_exact_acceleration.py"}
    for name in names:
        (source / name).write_text("# synthetic native input\n")
    files = {
        "__init__.py": "# synthetic tools package\n",
        "build_prepared_operators.py": "# synthetic builder\n",
        "operator_grid.py": "import tools.operator_accuracy_review as review\n",
        "operator_thermal_validation.py": "import tools.operator_grid as grid\n",
        "operator_accuracy_review.py": "import tools.build_prepared_operators as storage\n",
    }
    for name, text in files.items():
        (directory / name).write_text(text)
    monkeypatch.setattr(grid, "__file__", str(directory / "operator_grid.py"))
    monkeypatch.setattr(thermal, "__file__", str(directory / "operator_thermal_validation.py"))
    monkeypatch.setattr(builder, "__file__", str(directory / "build_prepared_operators.py"))
    monkeypatch.setattr(grid, "COMMON_HELPER_SHA", builder.digest(common / "ansys_utils.py"))
    monkeypatch.setattr(grid, "TRAINING_CARDS", {})
    return source, directory, common


@pytest.mark.parametrize("closure", [grid.training_closure, thermal.closure])
def test_qualified_tool_dependencies_and_initializer_are_pinned(source_tree, closure):
    source, directory, common = source_tree
    pins = closure(source)
    expected_tools = {str(directory / name) for name in (
        "__init__.py", "build_prepared_operators.py", "operator_grid.py",
        "operator_thermal_validation.py", "operator_accuracy_review.py")}
    actual_tools = {name for name in pins["files_sha256"] if Path(name).is_relative_to(directory)}
    assert actual_tools == expected_tools
    assert str(common / "__init__.py") in pins["files_sha256"]
    receipt = directory.parent / "SOURCE_PINS.json"
    receipt.write_text(json.dumps(pins))
    digest = builder.digest(receipt)
    grid.verify_pins(receipt, digest)
    (directory / "__init__.py").write_text("# changed package\n")
    with pytest.raises(ValueError, match="dependency drift"):
        grid.verify_pins(receipt, digest)


@pytest.mark.parametrize("closure", [grid.training_closure, thermal.closure])
def test_missing_qualified_tool_dependency_is_rejected(source_tree, closure):
    source, directory, _ = source_tree
    (directory / "operator_accuracy_review.py").unlink()
    with pytest.raises(ValueError, match="missing offline tool dependency"):
        closure(source)


@pytest.mark.parametrize("closure", [grid.training_closure, thermal.closure])
def test_ambiguous_qualified_tool_dependency_is_rejected(source_tree, closure):
    source, _, _ = source_tree
    duplicate = source / "tools"
    duplicate.mkdir()
    (duplicate / "operator_accuracy_review.py").write_text("# ambiguous dependency\n")
    with pytest.raises(ValueError, match="ambiguous"):
        closure(source)


@pytest.mark.parametrize("include_tools", [False, True])
def test_builder_inventory_supports_legacy_and_reorganized_sources(tmp_path, monkeypatch, include_tools):
    import tools.package_assembly_runtime as package

    for relative in ("D2W/utils", "W2W/utils"):
        (tmp_path / relative).mkdir(parents=True)
    for name in ("requirements.txt", "LICENSE", "README.md", "PORTABLE_ASSEMBLY.md",
                 "W2W/ASSEMBLY_ORDER.md", "prepared_operator_catalog.json"):
        (tmp_path / name).write_text("synthetic fixture\n")
    if include_tools:
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools/__init__.py").write_text("")
        (tmp_path / "tools/operator_grid.py").write_text("# synthetic controller\n")
    monkeypatch.setattr(package, "delivery_profiles", lambda _: ())
    files = builder.application_files(tmp_path)
    selected = {str(target) for _, target, _ in files}
    assert ("source/tools/__init__.py" in selected) == include_tools
    assert ("source/tools/operator_grid.py" in selected) == include_tools


def test_legacy_generator_still_resolves_checkout_geometry():
    from tools import generate_stack_variants

    assert generate_stack_variants.ROOT == ROOT


def test_relocated_packaging_tools_keep_application_and_evidence_roots():
    from tools import package_assembly_runtime, publish_component_operators, verify_assembly_runtime

    assert package_assembly_runtime.ROOT == verify_assembly_runtime.ROOT == ROOT
    assert publish_component_operators.ROOT == Path("/app/nanocad/projects/personal/zchen/YAP_STACK")
