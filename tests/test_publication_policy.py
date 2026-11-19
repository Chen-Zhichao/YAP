"""Bounded public-source checks; never execute or import private solver code."""

import ast
from functools import lru_cache
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_TOOLS = (
    "build_prepared_operators", "operator_grid", "operator_thermal_validation",
    "operator_accuracy_pool", "operator_accuracy_review", "operator_frozen_residual",
    "operator_reference_repair", "operator_snapshot_diagnostic",
)
PRIVATE_PATHS = tuple("tools/" + n + ".py" for n in PRIVATE_TOOLS) + tuple(
    "tests/test_" + n + ".py" for n in PRIVATE_TOOLS + ("operator_pid_reuse",))
PRIVATE_PREFIX = "D2W/utils/paper/"
DECK_SUFFIXES = (".apdl", ".mac", ".inp", ".cdb", ".wbjn")
PUBLIC_TOOLS = (
    "build_corrected_compact_mechanics", "fabrication_initial_bow_study",
    "generate_stack_variants", "operator_coverage", "operator_work_reference",
    "package_assembly_runtime", "publish_component_operators", "verify_assembly_runtime",
)
SOURCE_DIRS = (".", "tools", "tools/development", "tools/operators", "tests", "D2W", "W2W", "D2W/utils", "W2W/utils",
               "W2W/utils/paper", "W2W/utils/case_studies")
SOLVER_APIS = {"run_mapdl", "launch_mapdl", "build_apdl", "apdl_deck", "ramped_native_deck"}
SOLVER_MODULES = ("ansys", "pyansys", "ansys_utils", "common.ansys_utils")
# A serializer/writer plus command syntax is evidence; prose mentions are not.
COMMAND_SYNTAX = re.compile(r"(?im)^\s*(?:/(?:PREP7|SOLU|POST1|BATCH)\b|"
                            r"(?:CE|CEDELE|ANTYPE|ET|MP|TB),|\*(?:GET|VWRITE|CFOPEN|DIM),)")


def _private(path):
    return path in PRIVATE_PATHS or path.startswith(PRIVATE_PREFIX)


def _git(*args, input=None):
    return subprocess.run(["git", *args], cwd=ROOT, input=input,
                          capture_output=True, timeout=20)


def _require_git():
    if not (ROOT / ".git").exists():
        pytest.skip("Git-specific check in a Git-less source snapshot")


@lru_cache(maxsize=1)
def _candidates():
    if (ROOT / ".git").exists():
        result = _git("ls-files", "-c", "-o", "--exclude-standard", "-z")
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        assert len(result.stdout) <= 2 * 1024**2
        return sorted(set(os.fsdecode(p) for p in result.stdout.split(b"\0") if p))
    # No recursive traversal or private-directory reads in exported snapshots.
    paths = []
    for relative in SOURCE_DIRS:
        directory = ROOT / relative
        if not directory.is_dir():
            continue
        assert not directory.is_symlink()
        with os.scandir(directory) as entries:
            for count, entry in enumerate(entries, 1):
                assert count <= 4096, "source-directory entry bound exceeded"
                if entry.name.startswith(".") or not entry.name.endswith(".py"):
                    continue
                path = (Path(relative) / entry.name).as_posix()
                if not _private(path):
                    paths.append(path)
    return sorted(set(paths))


@lru_cache(maxsize=1)
def _sources():
    paths = [p for p in _candidates() if p.endswith(".py") and not _private(p)]
    assert 0 < len(paths) <= 1024
    result, total = {}, 0
    for relative in paths:
        path = Path(relative)
        assert not path.is_absolute() and not {"..", "#snapshot", ".snapshot"}.intersection(path.parts)
        filename = ROOT / path
        assert not any(p.is_symlink() for p in (filename, *filename.parents))
        size = filename.stat().st_size
        assert size <= 2 * 1024**2
        total += size
        assert total <= 32 * 1024**2
        raw = filename.read_bytes()
        assert len(raw) == size
        result[relative] = ast.parse(raw, filename=relative)
    return result


def _name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _name(node.value) + "." + node.attr
    return ""


def _imports(tree, module):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for row in node.names:
                yield row.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                package = module.split(".")[:-1]
                base = ".".join(package[:len(package) - node.level + 1] + ([base] if base else []))
            yield base, node.lineno
            for row in node.names:
                yield (base + "." if base else "") + row.name, node.lineno
        elif isinstance(node, ast.Call) and _name(node.func).endswith(("import_module", "__import__")):
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                yield node.args[0].value, node.lineno


def _solver_findings(tree):
    findings = []
    for imported, line in _imports(tree, ""):
        if any(imported == name or imported.startswith(name + ".") for name in SOLVER_MODULES):
            findings.append((line, "solver import", imported))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in SOLVER_APIS:
                findings.append((node.lineno, "solver/deck entry point", node.name))
            commands = any(isinstance(n, ast.Constant) and isinstance(n.value, str)
                           and COMMAND_SYNTAX.search(n.value) for n in ast.walk(node))
            serialization = any(isinstance(n, ast.Call) and _name(n.func).split(".")[-1]
                                in {"join", "write", "write_text", "write_bytes"} for n in ast.walk(node))
            if commands and serialization:
                findings.append((node.lineno, "command text serialization", node.name))
        elif isinstance(node, ast.Call):
            name = _name(node.func)
            if name.split(".")[-1] in SOLVER_APIS:
                findings.append((node.lineno, "solver/deck API call", name))
            if name in {"subprocess.run", "subprocess.Popen", "subprocess.call", "os.system"} and node.args:
                words = [n.value for n in ast.walk(node.args[0])
                         if isinstance(n, ast.Constant) and isinstance(n.value, str)]
                if any(re.search(r"(?i)(?:^|[/\\\s])(?:ansys\d*|mapdl)(?:$|[/\\.\s])", w) for w in words):
                    findings.append((node.lineno, "solver process command", name))
    return findings


def _private_module(name):
    return name.split(".")[-1] in PRIVATE_TOOLS or name == "D2W.utils.paper" or name.startswith("D2W.utils.paper.")


def test_git_candidates_exclude_private_workflows_and_solver_decks():
    _require_git()
    bad = [p for p in _candidates() if _private(p) or Path(p).suffix.lower() in DECK_SUFFIXES]
    assert not bad, "private/solver files remain publication candidates: " + repr(bad)


def test_worktree_private_files_and_nested_decks_are_ignored_without_index():
    _require_git()
    probes = list(PRIVATE_PATHS) + [PRIVATE_PREFIX + name for name in
        ("export_ansys_dsf_mesh.py", "run_p2_thickness_pipeline.py", "future_private_driver.py")]
    probes += [prefix + "solver" + suffix for prefix in ("", "examples/nested/", "tools/nested/")
               for suffix in DECK_SUFFIXES]
    result = _git("check-ignore", "--no-index", "-z", "--stdin",
                  input=b"\0".join(os.fsencode(p) for p in probes) + b"\0")
    assert result.returncode in (0, 1), result.stderr.decode(errors="replace")
    ignored = {os.fsdecode(p) for p in result.stdout.split(b"\0") if p}
    assert set(probes) <= ignored, "missing ignore rules: " + repr(sorted(set(probes) - ignored))


def test_retained_public_tools_are_in_source_manifest():
    assert all("tools/" + name + ".py" in _sources() for name in PUBLIC_TOOLS)


def test_public_candidate_sources_have_no_solver_or_deck_behavior():
    bad = {path: findings for path, tree in _sources().items() if (findings := _solver_findings(tree))}
    assert not bad, "solver behavior in public sources: " + repr(bad)


def test_transitive_local_import_closures_do_not_need_private_tools():
    sources = _sources()
    modules = {path[:-3].replace("/", "."): path for path in sources}
    graph, forbidden = {}, {}
    for module, path in modules.items():
        graph[module] = set()
        for imported, line in _imports(sources[path], module):
            if _private_module(imported):
                forbidden.setdefault(module, []).append((line, imported))
            graph[module].update(n for n in (imported, "tools." + imported, "D2W." + imported,
                                             "W2W." + imported) if n in modules)
    bad = []
    for start in graph:
        queue, seen = [(start, [start])], set()
        while queue:
            module, chain = queue.pop(0)
            if module in seen:
                continue
            seen.add(module)
            if module in forbidden:
                bad.append((chain, forbidden[module]))
                break
            queue.extend((child, chain + [child]) for child in graph[module] if child not in seen)
    assert not bad, "private dependencies in public import closure: " + repr(bad[:20])


@pytest.mark.parametrize("name", sorted(SOLVER_APIS))
def test_behavior_guard_rejects_suspicious_entry_points_without_solver_fixtures(name):
    source = "def " + name + "():\n    return 'English placeholder only'\n"
    assert _solver_findings(ast.parse(source))


def test_evidence_only_words_and_numerical_solve_are_not_banned():
    source = '"""Not independent MAPDL validation."""\ndef model(x):\n    return numpy.linalg.solve(x, x[:, 0])\n'
    assert not _solver_findings(ast.parse(source))


def test_relative_and_dynamic_private_imports_are_detected_without_importing():
    for source in ("from . import operator_grid\n", "import tools.operator_grid\n",
                   "importlib.import_module('tools.operator_grid')\n"):
        assert any(_private_module(name) for name, line in _imports(ast.parse(source), "tools.public_reader"))


def test_gitless_snapshot_keeps_static_checks_without_reading_private_folder(tmp_path, monkeypatch):
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools/public_reader.py").write_text("answer = 42\n")
    private = tmp_path / PRIVATE_PREFIX
    private.mkdir(parents=True)
    (private / "not_public.py").write_text("English placeholder, not Python source")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(sys.modules[__name__], "ROOT", tmp_path)
            _candidates.cache_clear()
            _sources.cache_clear()
            with pytest.raises(pytest.skip.Exception):
                _require_git()
            assert set(_sources()) == {"tools/public_reader.py"}
            assert not _solver_findings(_sources()["tools/public_reader.py"])
    finally:
        _candidates.cache_clear()
        _sources.cache_clear()
