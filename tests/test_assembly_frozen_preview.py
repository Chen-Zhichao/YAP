"""Explicit reference preview reuse; fixtures do not establish physical validity."""

from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest

from W2W.assembly_order import build_plan, load_stack_input
from W2W.assembly_design import load_design, run_design
from W2W.assembly_mechanics import LayerClockOperator, run_native_from_settings
import W2W.assembly_operator_library as loader
from test_assembly_material_clock import material, _assert_close, _assert_unchanged, _evaluate

ROOT = Path(__file__).resolve().parents[1]
NAME = "design_6_4hi_p6_reference"
DESIGN = ROOT / "W2W/input" / NAME
CONFIG = ROOT / "W2W/configs" / NAME / (NAME + ".yaml")


@pytest.mark.parametrize("name,target,retained,critical,threshold", [
    (NAME, 51.5, 38.5, 832667, .5929318594669777),
    ("design_6_4hi_p2", 50., 37., 7500000, .19764395315565994)])
def test_reference_input_and_all_five_thickness_ledgers(name, target, retained, critical, threshold):
    directory = ROOT / "W2W/input" / name
    cfg = OmegaConf.load(ROOT / "W2W/configs" / name / (name + ".yaml")).w2w_modeling
    cfg.OVERLAY_SAMPLES = 2
    d, plan, overlay, _, provenance = load_design(cfg, directory, name)
    assert d["mechanics"]["frozen_preview"] is True
    assert d["mechanics"]["temporal_profile"] == ("distilled101" if target == 51.5 else "distilled26")
    assert len(plan["routes"]) == 5
    for route in plan["routes"]:
        assert route["final"]["total_thicknesses_um"] == [target, target, target, 775.]
        assert all(event["thinning"]["retained_si_um"] == retained for event in route["events"])
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


def test_prepared_wafer_loader_selects_exact_layers_without_constructing_cap(monkeypatch):
    geometry = load_stack_input(ROOT / "W2W/input/design_6_4hi_p2_no_thinning/stack.yaml")["stack"]["geometry"]
    calls = []
    marker = object()
    def load_layers(path, received, *, prepared_bodies=False):
        calls.append((path, received, prepared_bodies))
        return marker, {"removable_bulk_operator_built": False}
    monkeypatch.setattr(loader, "_load_reference_p2_preview_library", load_layers)
    library, provenance = loader.load_assembly_library("BI2_50.json", geometry, frozen_preview=True)
    assert library is marker
    assert calls == [("BI2_50.json", geometry, True)]
    assert provenance["removable_bulk_operator_built"] is False
    assert "WAFER" in provenance["physics_qualification"]
    with pytest.raises(ValueError, match="identified reference BI2/P24"):
        loader.load_assembly_library("BI2_50.json", geometry)
    with pytest.raises(ValueError, match="identified reference BI2/P24"):
        loader.load_assembly_library("BI6_50.json", geometry, frozen_preview=True)


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
