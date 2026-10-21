"""Explicit reference preview reuse; fixtures do not establish physical validity."""

from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest

from W2W.assembly_order import _build_legacy_plan as build_plan, load_stack_input
from W2W.assembly_design import load_design, run_design
from W2W.assembly_mechanics import LayerClockOperator, run_native_from_settings
import W2W.assembly_operator_library as loader

ROOT = Path(__file__).resolve().parents[1]
NAME = "design_6_4hi_p6_reference"
DESIGN = ROOT / "W2W/input" / NAME
CONFIG = ROOT / "W2W/configs" / NAME / (NAME + ".yaml")


@pytest.fixture
def material():
    from test_assembly_material_clock import material as native_fixture
    return native_fixture.__wrapped__()


@pytest.mark.parametrize("name,target,retained,critical,threshold", [
    (NAME, 51.5, 38.5, 832667, .5929318594669777),
    ("design_6_4hi_p2", 50., 37., 7500000, .19764395315565994)])
def test_reference_input_and_all_five_thickness_ledgers(name, target, retained, critical, threshold, monkeypatch):
    from W2W import assembly_design, assembly_order
    monkeypatch.setattr(assembly_design, "build_plan", assembly_order._build_mixed_plan)
    directory = ROOT / "W2W/input" / name
    cfg = OmegaConf.load(ROOT / "W2W/configs" / name / (name + ".yaml")).w2w_modeling
    cfg.OVERLAY_SAMPLES = 2
    d, plan, overlay, _, provenance = load_design(cfg, directory, name)
    assert d["mechanics"]["frozen_preview"] is True
    assert d["mechanics"]["temporal_profile"] == ("distilled101" if target == 51.5 else "component_graded25")
    assert len(plan["routes"]) == 5
    for route in plan["routes"]:
        if route["status"] == "NEEDS_PROCESS_DEFINITION":
            assert route["events"] == []
            assert "final_ledger" not in route
            continue
        assert route["final_ledger"]["total_thicknesses_um"] == [target] * 4
        assert route["final_ledger"]["retained_si_thicknesses_um"] == [retained + 1.5, retained, retained, retained + 1.5]
        assert all(preparation["retained_si_um"] == retained
                   for event in route["events"] for role in ("lower", "upper")
                   for preparation in event["preparations"][role])
    assert provenance["site_count"] == 624
    assert provenance["critical_pad_count"] == critical
    assert overlay["threshold_um"] == pytest.approx(threshold)


@pytest.mark.parametrize("fail", [False, True])
def test_factory_native_binding_is_private_and_keeps_constructor_defaults(monkeypatch, fail):
    original = Path("public_bi6.json")
    monkeypatch.setitem(globals(), "NATIVE", original)
    class Factory:
        def __init__(self, library, package, *, mode="unchanged"):
            self.path, self.mode = NATIVE, mode
            if package["fail"]:
                raise ValueError("fixture failure")
    api = SimpleNamespace(SiliconBulkFactory=Factory, NATIVE=original, FAMILY="bi6")
    environment = dict(os.environ)
    if fail:
        with pytest.raises(ValueError, match="fixture failure"):
            loader._bound_reference_factory(api, None, {"fail": True}, Path("private_bi2.json"))
    else:
        factory = loader._bound_reference_factory(api, None, {"fail": False}, Path("private_bi2.json"))
        assert factory.path == Path("private_bi2.json")
        assert factory.mode == "unchanged"
    assert api.NATIVE == original and api.FAMILY == "bi6"
    assert Factory.__init__.__globals__["NATIVE"] == original
    assert dict(os.environ) == environment


def test_p2_preview_rejects_changed_manifest_before_loading_operators(tmp_path):
    path = tmp_path / "BI2_50.json"
    path.write_text('{"name": "BI2_50", "rank": 24}')
    geometry = load_stack_input(ROOT / "W2W/input/design_6_4hi_p2/stack.yaml")["stack"]["geometry"]
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        loader.load_assembly_library(path, geometry, frozen_preview=True)


@pytest.mark.parametrize("total", [30., 50., 80.])
@pytest.mark.parametrize("preview", [False, True])
def test_prepared_wafer_loader_is_retired_before_input_access(monkeypatch, total, preview):
    geometry = dict(loader._GEOMETRY_UM, diameter_mm=300., initial_total_thickness_um=total,
                    target_total_thickness_um=total)
    def forbidden(*_, **__):
        pytest.fail("retired W2W must not consult catalogs, native modules, or operators")
    for name in ("_input_path", "_import_mechanics_module", "_prepared_catalog", "_load_reference_p2_preview_library"):
        monkeypatch.setattr(loader, name, forbidden)
    with pytest.raises(ValueError, match="no-thinning W2W is retired"):
        loader.load_assembly_library(f"BI2_{total:g}.json", geometry, frozen_preview=preview)
    with pytest.raises(ValueError, match="no-thinning W2W is retired"):
        loader.prepared_operator_request(f"BI2_{total:g}.json", geometry, frozen_preview=preview)


def test_p2_cap_audit_requires_final_726p5_not_725():
    audit = good_audits()
    audit["bulk_audits"][0]["thickness_um"] = 726.5
    loader._check_reference_bulk_audits(audit, thickness_um=726.5)
    with pytest.raises(ValueError, match="725"):
        loader._check_reference_bulk_audits(audit)


def good_audits():
    return dict(direct_small_stack_relative_defect=1e-12,
        block_audits=[dict(reaction_relative_defect=1e-10)],
        bulk_audits=[dict(thickness_um=725, abd_scaled_defect=.033,
            relaxed_normal_scaled_defect=.0017, free_thermal_q6_max_absolute_defect=1e-9,
            abd_per_width=np.eye(6).tolist(), analytical_abd=np.eye(6).tolist(),
            original_full_tensor_one_percent_pass=False, full_tensor_physical_admission=False,
            shear_terms_modified=False)])


def test_scoped_reference_gate_does_not_claim_full_one_percent():
    audit = good_audits()
    loader._check_reference_bulk_audits(audit)
    assert audit["bulk_audits"][0]["original_full_tensor_one_percent_pass"] is False


@pytest.mark.parametrize("key,value", [("abd_scaled_defect", .051),
    ("abd_scaled_defect", float("nan")), ("relaxed_normal_scaled_defect", .011),
    ("free_thermal_q6_max_absolute_defect", float("inf")),
    ("free_thermal_q6_max_absolute_defect", 2e-8), ("thickness_um", 735),
    ("original_full_tensor_one_percent_pass", True), ("shear_terms_modified", True)])
def test_reference_gate_rejects_failed_or_inconsistent_audits(key, value):
    audit = good_audits()
    audit["bulk_audits"][0][key] = value
    with pytest.raises(ValueError):
        loader._check_reference_bulk_audits(audit)


def test_optional_bubble_tangent_clock_adjusted_once():
    base = SimpleNamespace()
    proxy = LayerClockOperator(base, 500.)
    assert not hasattr(proxy, "assembled_bubble_tangent")
    base.assembled_bubble_tangent = lambda **kw: kw["elapsed_time_s"]
    base.evaluate = lambda **kw: base.assembled_bubble_tangent(**kw)
    assert proxy.assembled_bubble_tangent(elapsed_time_s=300.) == 800.
    assert proxy.evaluate(elapsed_time_s=300.) == 800.


@pytest.mark.parametrize("plastic", [False, True])
def test_reference_acceleration_preserves_real_cu_response_and_layer_age(material, plastic):
    from test_assembly_material_clock import _assert_close, _assert_unchanged, _evaluate
    from W2W.assembly_mechanics import _reference_runtime, native
    base, state, macro, bubble = material
    before = state.copy()
    proxy = LayerClockOperator(base, 4000.)
    expected = _evaluate(base, state, macro, bubble, endpoint=4550., plastic=plastic)
    had_tangent = hasattr(base, "assembled_bubble_tangent")
    with _reference_runtime(True):
        _assert_close(_evaluate(proxy, state, macro, bubble, endpoint=550., plastic=plastic), expected)
        args = (macro, bubble, 350., 90., state)
        options = dict(creep_active=True, plastic_active=plastic)
        _assert_close(proxy.assembled_bubble_tangent(*args, elapsed_time_s=550., **options),
                      base.assembled_bubble_tangent(*args, elapsed_time_s=4550., **options))
        _assert_unchanged(state, before)
    assert not native("exp12_all_gauss_exact_acceleration")._INSTALLED
    assert hasattr(base, "assembled_bubble_tangent") == had_tangent


@pytest.mark.parametrize("invalid", [1, "true", None])
def test_preview_requires_explicit_boolean(tmp_path, invalid):
    with pytest.raises(ValueError, match="Boolean"):
        loader.load_assembly_library("absent", {}, frozen_preview=invalid)
    design = OmegaConf.load(DESIGN / "stack.yaml")
    design.mechanics.frozen_preview = invalid
    path = tmp_path / "stack.yaml"
    OmegaConf.save(design, path)
    with pytest.raises(ValueError, match="Boolean"):
        load_stack_input(path)


def test_preview_geometry_mismatch_fails_before_file_access():
    geometry = load_stack_input(DESIGN / "stack.yaml")["stack"]["geometry"]
    with pytest.raises(ValueError, match="51.5"):
        loader.load_assembly_library("absent", {**geometry, "target_total_thickness_um": 50.}, frozen_preview=True)


def test_compact_candidate_is_not_admitted_by_strict_loader(tmp_path):
    from test_assembly_operator_library import package_file, geometry
    path, _ = package_file(tmp_path, "BI6_50", 32)
    with pytest.raises(ValueError, match="READY_ENGINEERING_PREVIEW"):
        loader.load_assembly_library(path, geometry())


@pytest.mark.parametrize("name,retained,bulk", [(NAME, .0385, .725), ("design_6_4hi_p2", .037, .7265)])
def test_both_entry_points_dispatch_preview_and_keep_histories(tmp_path, monkeypatch, name, retained, bulk):
    from test_assembly_mechanics import tiny_library
    import W2W.assembly_design as bridge
    monkeypatch.setattr(bridge, "build_plan", build_plan)
    base = tiny_library()
    library = replace(base, thickness_mm={**base.thickness_mm, "retained_si": retained, "bulk_si": bulk})
    calls = []
    def load(*args, **kwargs):
        calls.append(kwargs)
        return library, {"fixture": "nonphysical", "frozen_preview": True}
    monkeypatch.setattr(loader, "load_assembly_library", load)
    directory = ROOT / "W2W/input" / name
    design = load_stack_input(directory / "stack.yaml")
    plan = build_plan(design["stack"])
    archive = run_native_from_settings(plan, design["mechanics"])
    assert len(archive["routes"]) == 5
    for route, request in zip(archive["routes"], plan["routes"]):
        for wafer, cycles in enumerate(request["final"]["thermal_cycle_counts"], 1):
            assert route["final"]["ages_s"][f"wafer{wafer:02d}_front_bi"] == 300.+4900.*cycles
    cfg = OmegaConf.load(ROOT / "W2W/configs" / name / (name + ".yaml")).w2w_modeling
    cfg.OVERLAY_SAMPLES = 2
    run = run_design(cfg, directory, name, output_dir=tmp_path)
    assert len(json.loads((run / "RESULT.json").read_text())["routes"]) == 5
    assert calls == [{"frozen_preview": True}, {"frozen_preview": True}]


@pytest.mark.parametrize("kind", ["die", "wafer"])
def test_shared_settings_dispatches_die_and_primary_wafer_without_native_imports(monkeypatch, kind):
    from W2W import assembly_mechanics
    from W2W.assembly_order import DIE_PLAN_SCHEMA, PLAN_SCHEMA
    from test_assembly_operator_library import prepared_fixture_geometry
    geometry = prepared_fixture_geometry(30., kind) if kind == "die" else dict(loader._GEOMETRY_UM, diameter_mm=300.)
    package = "BI2_30.json" if kind == "die" else "BI2_50.json"
    plan = dict(schema=DIE_PLAN_SCHEMA if kind == "die" else PLAN_SCHEMA,
                context=dict(geometry=geometry, body_kind="prepared_die_column" if kind == "die" else "wafer"))
    calls = []
    def load_die(*args, **kwargs):
        calls.append(("die", args, kwargs))
        return object(), {"fixture": "nonphysical"}
    def load_wafer(*args, **kwargs):
        calls.append(("wafer", args, kwargs))
        return object(), {"fixture": "nonphysical"}
    monkeypatch.setattr(loader, "require_prepared_operator", lambda *_, **__: None)
    monkeypatch.setattr(loader, "load_prepared_die_library", load_die)
    monkeypatch.setattr(loader, "load_assembly_library", load_wafer)
    monkeypatch.setattr(assembly_mechanics, "_configure_mechanics_runtime_environment", lambda **_: None)
    monkeypatch.setattr(assembly_mechanics, "_check_plan", lambda _: None)
    monkeypatch.setattr(assembly_mechanics, "_recipe", lambda *_, **__: None)
    monkeypatch.setattr(assembly_mechanics, "_check_metric_policy", lambda *_, **__: None)
    monkeypatch.setattr(assembly_mechanics, "native", lambda *_: pytest.fail("no native modules may be imported"))
    monkeypatch.setattr(assembly_mechanics, "run_native", lambda *_, **__: {"fixture": "not_a_history_solve"})
    result = run_native_from_settings(plan, dict(operator_package=package, temporal_profile="distilled26"))
    assert result == {"fixture": "not_a_history_solve"}
    assert calls == [(kind, (package, geometry), {})]


def test_total50_prepared_die_keeps_frozen_gate_and_ignores_native64_catalog(monkeypatch):
    from test_assembly_operator_library import prepared_fixture_geometry
    forbidden = lambda *_: pytest.fail("TOTAL50 must not consult the new catalog")
    monkeypatch.setattr(loader, "_prepared_catalog", forbidden)
    monkeypatch.setattr(loader, "_load_prepared_native64_library", forbidden)
    assert loader._REFERENCE_P2_MANIFEST_SHA256 == "cba9f64bdf41b610ec9b538858f07c119f7febd89515f84d8ac6df2e74bd8612"
    marker = object()
    calls = []
    def frozen(path, geometry, *, prepared_bodies=False):
        calls.append((path, geometry, prepared_bodies))
        return marker, {}
    monkeypatch.setattr(loader, "_load_reference_p2_preview_library", frozen)
    assert loader.load_prepared_die_library("BI2_50.json", prepared_fixture_geometry(50., "die"), frozen_preview=True)[0] is marker
    assert len(calls) == 1 and calls[0][2]
    with pytest.raises(ValueError, match="identified reference BI2/P24"):
        loader.load_prepared_die_library("BI2_50.json", prepared_fixture_geometry(50., "die"))
