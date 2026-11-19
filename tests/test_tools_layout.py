"""Retained public tools keep their import and CLI entry points."""

import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_TOOLS = (
    "operator_construction", "operator_geometry", "operator_history", "operator_production",
    "operator_reference", "operator_seed_loader", "prepared_coupon_geometry", "native_coupon_parity",
)


@pytest.mark.parametrize("name", DEVELOPMENT_TOOLS)
def test_development_imports_resolve_without_root_forwarders(name):
    expected = ROOT / "tools/development" / (name + ".py")
    module = importlib.import_module("tools." + name)
    assert Path(module.__file__).resolve() == expected
    assert not (ROOT / "tools" / (name + ".py")).exists()


PUBLIC_TOOLS = (
    "build_corrected_compact_mechanics", "fabrication_initial_bow_study",
    "generate_stack_variants", "operator_coverage", "operator_work_reference",
    "package_assembly_runtime", "publish_component_operators", "verify_assembly_runtime",
)
CLI_TOOLS = tuple(name for name in PUBLIC_TOOLS if name != "operator_work_reference")


@pytest.mark.parametrize("name", PUBLIC_TOOLS)
def test_public_tool_has_one_import_location_without_root_wrapper(name):
    expected = ROOT / "tools" / (name + ".py")
    assert expected.is_file()
    assert not (ROOT / (name + ".py")).exists()
    module = importlib.import_module("tools." + name)
    assert Path(module.__file__).resolve() == expected


def test_application_entry_modules_remain_at_checkout_root():
    for name in ("generate_assembly_design", "mechanics_runtime", "component_operators",
                 "thickness_operators"):
        assert (ROOT / (name + ".py")).is_file()


@pytest.mark.parametrize("name", CLI_TOOLS)
def test_public_tool_direct_cli_from_unrelated_directory(name, tmp_path):
    env = dict(os.environ, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "tools" / (name + ".py")), "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()


def test_public_module_cli_from_checkout():
    result = subprocess.run(
        [sys.executable, "-B", "-m", "tools.operator_coverage", "--help"],
        cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()


def test_legacy_input_generator_still_resolves_checkout_geometry():
    from tools import generate_stack_variants

    assert generate_stack_variants.ROOT == ROOT


def test_retained_packaging_tools_keep_application_and_evidence_roots():
    from tools import package_assembly_runtime, publish_component_operators, verify_assembly_runtime

    assert package_assembly_runtime.ROOT == verify_assembly_runtime.ROOT == ROOT
    assert Path(publish_component_operators.ROOT).is_absolute()
    assert publish_component_operators.CATALOG_RELATIVE == "distribution/operators/components/CATALOG.json"
