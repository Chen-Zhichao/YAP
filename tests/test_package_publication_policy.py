"""Synthetic policy cases only; no executable solver decks or solver launches."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.package_assembly_runtime as package


PRIVATE_TOOLS = (
    "build_prepared_operators", "operator_grid", "operator_thermal_validation",
    "operator_accuracy_pool", "operator_accuracy_review", "operator_frozen_residual",
    "operator_reference_repair", "operator_snapshot_diagnostic",
)


@pytest.mark.parametrize("name", PRIVATE_TOOLS)
def test_private_tools_are_not_in_public_inventory(name):
    assert name + ".py" not in package.PUBLIC_TOOL_FILES


@pytest.mark.parametrize("name", ["operator_geometry", "operator_seed_loader", "coupon_operator_library",
                                  "operator_construction", "package_assembly_runtime"])
def test_numerical_runtime_tools_remain_selected(name):
    assert name + ".py" in package.PUBLIC_TOOL_FILES


@pytest.mark.parametrize("source", [
    'def build_apdl():\n    return "deliberately non-executable fixture"\n',
    'def apdl_deck():\n    return "deliberately non-executable fixture"\n',
    'def run_mapdl():\n    raise RuntimeError("fixture only")\n',
    'def example():\n    run_mapdl("invalid fixture argument")\n',
    'from common.ansys_utils import run_mapdl as blocked\ndef example():\n    blocked("invalid fixture argument")\n',
    'def example():\n    launch_mapdl("invalid fixture argument")\n',
    'def example():\n    blocked = backend.run_mapdl\n    blocked("invalid fixture argument")\n',
    'import ansys.mapdl.core as solver\n',
    'from ansys.mapdl.reader import read_binary\n',
    'from common import ansys_utils\n',
    'import ansys_utils as local_solver\n',
    'import subprocess\ndef example():\n    subprocess.run(["ansys99999", "INVALID_FIXTURE_ONLY"])\n',
    'import subprocess\ndef example():\n    command = ["/not/a/real/path/mapdl", "INVALID_FIXTURE_ONLY"]\n    subprocess.Popen(command)\n',
    'def example(path):\n    path.write_text("/PREP7 INVALID_FIXTURE_ONLY")\n',
    'def example():\n    fragment = "/POST1 INVALID_FIXTURE_ONLY"\n    return fragment\n',
])
def test_actual_generator_or_launch_signatures_fail_closed(source):
    with pytest.raises(ValueError, match="publication source policy"):
        package.check_publication_source(source, filename="synthetic.py")


@pytest.mark.parametrize("source", [
    '"""Compare a Python FE model against saved ANSYS/MAPDL evidence."""\n',
    'def response(stiffness, q):\n    return stiffness @ q\n',
    'def geometry(deck):\n    return dict(nodes=deck.nodes, elements=deck.elements, constraints=deck.constraints)\n',
    'class SeedSources:\n    def bind(self, deck):\n        return deck.nodes, deck.elements\n',
    'def parse_commands(text):\n    return [line for line in text.splitlines() if line != "/SOLU"]\n',
    'def evidence():\n    return {"reference": "MAPDL", "solver_run": False}\n',
    'import subprocess\ndef example():\n    subprocess.run(["python", "numerical_model.py"])\n',
])
def test_math_parsing_and_reference_mentions_are_not_solver_scripts(source):
    package.check_publication_source(source, filename="numerical.py")


@pytest.fixture
def tiny_export(tmp_path, monkeypatch):
    source, native = tmp_path / "source", tmp_path / "runtime/native"
    source.mkdir()
    native.mkdir(parents=True)
    for directory in ("D2W", "W2W", "D2W/utils", "W2W/utils", "tools"):
        (source / directory).mkdir(exist_ok=True)
    for name in ("requirements.txt", "LICENSE", "README.md", "PORTABLE_ASSEMBLY.md", "W2W/ASSEMBLY_ORDER.md"):
        (source / name).write_text("synthetic non-physics fixture\n")
    module = native / "numerical.py"
    module.write_text("def response(k, q):\n    return k @ q\n")
    metric = native.parent / "200_bow_diff_vs_magnification/nonlinear_face_lever.py"
    metric.parent.mkdir()
    metric.write_text("def metric(value):\n    return value\n")
    monkeypatch.setattr(package, "SEEDS", ("numerical",))
    monkeypatch.setattr(package, "data_files", lambda *args, **kwargs: {})
    monkeypatch.setattr(package, "delivery_profiles", lambda *args: ())
    monkeypatch.setattr(package, "resource_directory", lambda name, **kwargs: native)
    monkeypatch.setattr(package, "validate_write_path", Path)
    monkeypatch.setattr(package.shutil, "disk_usage", lambda *args: SimpleNamespace(free=10 * 1024**3))
    monkeypatch.setattr(package.time, "sleep", lambda *args: None)
    return SimpleNamespace(source=source, native=native, module=module, metric=metric,
                           destination=tmp_path / "export")


def test_private_and_unlisted_local_tools_are_never_copied(tiny_export):
    f = tiny_export
    for name in (*PRIVATE_TOOLS, "unlisted_local_private"):
        (f.source / "tools" / (name + ".py")).write_text("# physically present, deliberately private\n")
    public = f.source / "tools/operator_geometry.py"
    public.write_text("def nodes(deck):\n    return deck.nodes\n")
    package.export(f.destination, f.native, source_root=f.source)
    copied = f.destination / "YAP/tools"
    assert {path.name for path in copied.iterdir()} == {"operator_geometry.py"}
    assert (copied / public.name).read_bytes() == public.read_bytes()


@pytest.mark.parametrize("location", ["native", "application", "metric", "data_source", "native_dependency"])
def test_rejected_source_fails_before_destination_or_copy(tiny_export, monkeypatch, location):
    f = tiny_export
    invalid = 'def build_apdl():\n    return "non-executable synthetic fixture"\n'
    if location == "native":
        f.module.write_text(invalid)
    elif location == "application":
        (f.source / "ordinary.py").write_text(invalid)
    elif location == "metric":
        f.metric.write_text(invalid)
    elif location == "data_source":
        extra = f.native / "source_copy.py"
        extra.write_text(invalid)
        monkeypatch.setattr(package, "data_files", lambda *args, **kwargs: {extra: None})
    else:
        f.module.write_text("import dependency\n")
        (f.native / "dependency.py").write_text(invalid)
    monkeypatch.setattr(package.shutil, "disk_usage", lambda *args: pytest.fail("policy must reject before output admission"))
    with pytest.raises(ValueError, match="publication source policy"):
        package.export(f.destination, f.native, source_root=f.source)
    assert not f.destination.exists()


def test_native_numerical_closure_remains_complete(tiny_export):
    f = tiny_export
    f.module.write_text("import geometry\ndef response(k, q):\n    return k @ q\n")
    geometry = f.native / "geometry.py"
    geometry.write_text("def nodes(deck):\n    return deck.nodes, deck.elements, deck.constraints\n")
    assert set(package.native_sources(f.native)) == {f.module, geometry}
    package.export(f.destination, f.native, source_root=f.source)
    for original in (f.module, geometry):
        assert (f.destination / "YAP/runtime/native" / original.name).read_bytes() == original.read_bytes()


def test_development_sources_are_packaged_without_private_workflows(tiny_export):
    f = tiny_export
    directory = f.source / "tools/development"
    directory.mkdir()
    (directory / "operator_geometry.py").write_text("# synthetic geometry helper\n")
    (directory / "operator_grid.py").write_text("# private workflow, not published\n")
    package.export(f.destination, f.native, source_root=f.source)
    output = f.destination / "YAP/tools/development"
    assert (output / "operator_geometry.py").is_file()
    assert not (output / "operator_grid.py").exists()


def test_normalized_runtime_package_has_an_explicit_inventory(tiny_export):
    f = tiny_export
    directory = f.source / "tools/operators"
    directory.mkdir()
    expected = {"__init__.py", "__main__.py", "acceleration.py", "catalog.py", "kinematics.py",
                "reference.py", "runtime.py", "streamed.py"}
    assert package.PUBLIC_OPERATOR_FILES == expected
    for name in (*expected, "migrate.py", "unlisted.py"):
        (directory/name).write_text("# synthetic numerical package\n")
    package.export(f.destination, f.native, source_root=f.source)
    output = f.destination / "YAP/tools/operators"
    assert {path.name for path in output.iterdir()} == expected


@pytest.mark.parametrize("name", ["bi7_coupon", "bi5_grid_coupon", "native_coupon",
                                  "sampled_coupon", "shared_coupon", "source50_coupon_acceleration",
                                  "coupon_reference_tangent", "coupon_streamed_acceleration"])
def test_retired_loader_implementations_are_not_public(name):
    assert name + ".py" not in package.PUBLIC_TOOL_FILES
    assert name + ".py" not in package.PUBLIC_OPERATOR_FILES


def test_documented_launcher_is_explicit_optional_source(tiny_export):
    f = tiny_export
    launcher = f.source / "tools/run_example.sh"
    launcher.write_text("#!/bin/sh\n# Non-executable test placeholder; never invoked.\n")
    (f.source / "tools/private_local.sh").write_text("# must not be copied\n")
    package.export(f.destination, f.native, source_root=f.source)
    assert (f.destination / "YAP/tools/run_example.sh").read_bytes() == launcher.read_bytes()
    assert not (f.destination / "YAP/tools/private_local.sh").exists()


def test_solver_launcher_rejected_before_output(tiny_export):
    f = tiny_export
    (f.source / "tools/run_example.sh").write_text("ansys99999 INVALID_FIXTURE_ONLY\n")
    with pytest.raises(ValueError, match="solver shell content"):
        package.export(f.destination, f.native, source_root=f.source)
    assert not f.destination.exists()


def test_packager_itself_is_accepted_by_its_policy():
    source = Path(package.__file__).read_bytes()
    package.check_publication_source(source, filename=package.__file__)


@pytest.mark.parametrize("relative", [
    "tools/development/operator_geometry.py", "tools/development/operator_seed_loader.py",
    "tools/development/operator_construction.py", "tools/development/operator_history.py",
    "W2W/component_compiled_material.py", "mechanics_runtime.py",
])
def test_actual_numerical_adapters_are_not_mistaken_for_solver_generators(relative):
    path = Path(package.__file__).resolve().parents[1] / relative
    package.check_publication_source(path.read_bytes(), filename=relative)
