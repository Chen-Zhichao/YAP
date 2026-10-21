"""Public component requests preserve geometry, clocks and physical evidence."""

from dataclasses import replace
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import component_operators as components
from W2W import assembly_mechanics as mechanics
from W2W.assembly_order import _check_recipe_request


def entry(si=37., pitch=2.):
    return dict(id="BI2_SI37", geometry=dict(bi_pitch_um=pitch, diameter_pitch_ratio=.5,
        bi_thickness_um=1.5, beol_thickness_um=10., si_tsv_thickness_um=si,
        total_thickness_um=si+11.5, bi_layer_count=1, tsv_pitch_um=6., cell_side_um=6.),
        validation=dict(cycles=3, solved_states=73, bow_observations=1081))


def geometry(total=50.):
    return dict(width_mm=10., length_mm=10., total_thickness_um=total,
                front_bi_thickness_um=1.5, back_bi_thickness_um=1.5, beol_thickness_um=10.)


@pytest.mark.parametrize("si", [7., 17., 27., 37., 47., 48.5, 57., 67., 77., 87., 88.5])
def test_prepared_total_uses_the_same_si_with_two_bis(monkeypatch, si):
    monkeypatch.setattr(components, "resolve_entry", lambda *_a, **_k: entry(si))
    request = components.prepared_operator_request("component:BI2_SI37", geometry(si+13.), layout=(2., 2., 1.))
    assert request["validated_single_BI_total_um"] == si+11.5
    assert request["retained_si_thickness_um"] == si
    assert request["assembly_accuracy_validated"] is False
    with pytest.raises(ValueError, match="plus one backside BI"):
        components.prepared_operator_request("component:BI2_SI37", geometry(si+11.5))


@pytest.mark.parametrize("fault", ["pitch", "diameter", "preview", "rectangular", "boolean", "backside"])
def test_wrong_physical_request_is_rejected(monkeypatch, fault):
    monkeypatch.setattr(components, "resolve_entry", lambda *_a, **_k: entry())
    values, layout, preview = geometry(), (2., 2., 1.), False
    if fault == "pitch": layout = (6., 6., 3.)
    elif fault == "diameter": layout = (2., 2., .5)
    elif fault == "preview": preview = True
    elif fault == "rectangular": values["length_mm"] = 11.
    elif fault == "boolean": values["width_mm"] = True
    else: values["back_bi_thickness_um"] = 3.
    with pytest.raises(ValueError):
        components.prepared_operator_request("component:BI2_SI37", values,
                                             layout=layout, frozen_preview=preview)


def test_storage_refuses_external_paths_and_symlink_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(components, "PROJECT", tmp_path)
    assert components.canonical_write_path(tmp_path / "output") == tmp_path / "output"
    with pytest.raises(ValueError): components.canonical_write_path("/tmp/forbidden")
    with pytest.raises(ValueError): components.canonical_write_path(tmp_path.parent / "elsewhere")
    (tmp_path / "alias").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError): components.canonical_write_path(tmp_path / "alias/result")


def test_reference_work_certificate_is_required():
    with pytest.raises(ValueError, match="projection certificate"):
        components._projected_work({})
    fields = ("all_GP_strain", "all_macro_point_stress_virtual_work",
              "arbitrary_point_stress_body_virtual_work", "body_reference_action",
              "macro_reference_action", "free_thermal_field")
    proof = dict(macro_coordinates_retained=134, Cu_updates=0,
        nonlinear_law_compatibility_by_identical_strain_and_virtual_work=True,
        **{key:dict(absolute_error=0., reference_norm=1.) for key in fields})
    assert components._projected_work(dict(projection_certificate=proof)) is proof
    proof["all_GP_strain"]["absolute_error"] = 1.
    with pytest.raises(ValueError, match="certificate failed"):
        components._projected_work(dict(projection_certificate=proof))


def test_runtime_does_not_treat_desktop_session_metadata_as_paths(monkeypatch):
    checked = []
    monkeypatch.setenv("YAP_NATIVE_RUNTIME_DIR", os.environ.get(
        "YAP_NATIVE_RUNTIME_DIR", str(components.DELIVERY / "runtime/native")))
    monkeypatch.setenv("YAP_RUNTIME_RESOURCES", os.environ.get(
        "YAP_RUNTIME_RESOURCES", str(components.DELIVERY / "runtime/RESOURCES.json")))
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    monkeypatch.setattr(components, "canonical_write_path", lambda value, **kw: checked.append(kw["label"]))
    monkeypatch.setattr(components, "_check_loaded_native_root", lambda selected: None)
    components.configure_component_runtime()
    assert "XDG_SESSION_TYPE" not in checked and "XDG_CURRENT_DESKTOP" not in checked


def test_operand_reader_verifies_selected_file_and_rejects_changed_bytes(tmp_path):
    path = tmp_path / "operand.npz"
    np.savez(path, weights=np.array([1., 2.]), coordinates=np.arange(9.).reshape(3, 3))
    raw = path.read_bytes()
    bound = dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    reader = components.OperandReader()
    loaded = reader.arrays(bound, ("weights",))
    assert set(loaded) == {"weights"} and not loaded["weights"].flags.writeable
    assert reader.bytes_hashed == len(raw)
    reader.arrays(bound, ("coordinates",))
    assert reader.bytes_hashed == len(raw)
    path.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        reader.arrays(bound, ("weights",))


def test_component_schedule_keeps_original_physical_clock(monkeypatch):
    from D2W.assembly_order import load_stack_input, build_plan
    root = Path(__file__).resolve().parents[1]
    plan = build_plan(load_stack_input(root/"D2W/input/design_6_4hi_p2/stack.yaml")["stack"])
    _check_recipe_request(plan, "component_graded25")
    schedule = mechanics._recipe(plan, "component_graded25")
    assert len(schedule) == 25
    assert [step.time_s for step in schedule[9:17]] == [1040., 1130., 1310., 1670., 1850., 2750., 3650., 4550.]
    assert schedule[0].time_s == 300. and schedule[-1].time_s == 5200.
    assert all(step.creep_active for step in schedule[9:17])
    assert not any(step.creep_active for step in (*schedule[:9], *schedule[17:]))


@pytest.fixture(scope="module")
def actual_entries():
    if not components.DEFAULT_CATALOG.is_file():
        pytest.skip("published component catalog is not installed")
    from publish_component_operators import MetadataReader
    return MetadataReader().read(components.DEFAULT_CATALOG)[0]["entries"]


def test_all_actual_assessments_bind_exact_candidate_and_geometry(actual_entries):
    from publish_component_operators import MetadataReader
    reader = MetadataReader()
    for item in actual_entries:
        components._verify_validation(item, reader)
    reader.verify_unchanged()


@pytest.mark.parametrize("selected,other", [("BI2_SI37", "BI2_SI48p5"),
                                            ("BI2_SI7", "BI2_SI17"),
                                            ("BI2_SI77", "BI2_SI87")])
def test_assessment_cannot_be_attached_to_another_history(actual_entries, selected, other):
    from publish_component_operators import MetadataReader
    item = copy.deepcopy(next(e for e in actual_entries if e["id"] == selected))
    replacement = next(e for e in actual_entries if e["id"] == other)
    item["validation"]["history"] = replacement["validation"]["history"]
    with pytest.raises(ValueError, match="different candidate history"):
        components._verify_validation(item, MetadataReader())


@pytest.mark.parametrize("fault", ["metric", "geometry", "operator"])
def test_assessment_scope_and_artifact_variant_are_checked(actual_entries, fault):
    from publish_component_operators import MetadataReader
    item = copy.deepcopy(actual_entries[0])
    if fault == "metric":
        item["validation"]["max_relative_pct"] = 0.
    elif fault == "geometry":
        item["geometry"]["tsv_pitch_um"] = 12
    else:
        item["components"]["si"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        components._verify_validation(item, MetadataReader())


def gp_arrays():
    return dict(Cu_native_element_ids=np.repeat(np.array([13, 29]), 8),
        Cu_native_gauss_indices=np.tile(np.arange(8), 2),
        Cu_positions_mm=np.arange(48., dtype=float).reshape(16, 3) * 1e-4,
        weights=np.full(16, 1e-12))


def test_gp_comparison_preserves_global_positions_and_original_operands():
    reference = gp_arrays()
    operator = {key: value.copy() for key, value in reference.items()}
    for array in (*reference.values(), *operator.values()):
        array.flags.writeable = False
    before = {key: value.tobytes() for key, value in operator.items()}
    components._verify_ordered_gauss(reference, operator, {"Cu_GP": 16})
    assert {key: value.tobytes() for key, value in operator.items()} == before


@pytest.mark.parametrize("fault", [*components.GP_KEYS, "order", "count", "float_ids"])
def test_gp_reference_mismatch_is_rejected(fault):
    reference = gp_arrays()
    operator = {key: value.copy() for key, value in reference.items()}
    proof = {"Cu_GP": 16}
    if fault == "order":
        operator = {key: np.roll(value, 8, axis=0) for key, value in operator.items()}
    elif fault == "count":
        proof["Cu_GP"] = 8
    elif fault == "float_ids":
        operator["Cu_native_element_ids"] = operator["Cu_native_element_ids"].astype(float)
    else:
        operator[fault].flat[0] += 1
    with pytest.raises(ValueError, match="GP"):
        components._verify_ordered_gauss(reference, operator, proof)


@pytest.fixture
def card_api(monkeypatch):
    path = components.DELIVERY / "runtime/native/exp12_material_card_v2_kim2026.py"
    if not path.is_file():
        pytest.skip("delivered native card parser is not installed")
    spec = importlib.util.spec_from_file_location("_test_component_pinned_card", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_exact_delivered_cu_card_is_verified(card_api):
    original_read = card_api._read
    card = components._pinned_cu_card(card_api)
    assert card.sha256 == components.CU_MATERIAL_CARD_SHA256
    assert card.python_cu_card_kwargs()["creep_law"] == "modified_time_hardening"
    assert card_api._read is original_read


def test_modified_cu_card_is_rejected_before_parsing(tmp_path, monkeypatch, card_api):
    raw = (components.DELIVERY / "CU_MATERIAL_CARD.json").read_bytes()
    monkeypatch.setattr(components, "DELIVERY", tmp_path)
    (tmp_path / "CU_MATERIAL_CARD.json").write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        components._pinned_cu_card(card_api)


def test_cu_parser_uses_verified_snapshot_without_reopening(tmp_path, monkeypatch, card_api):
    raw = (components.DELIVERY / "CU_MATERIAL_CARD.json").read_bytes()
    target = tmp_path / "CU_MATERIAL_CARD.json"
    target.write_bytes(raw)
    monkeypatch.setattr(components, "DELIVERY", tmp_path)
    original = components.runtime._component_json

    def snapshot(path, expected):
        result = original(path, expected)
        target.write_bytes(b"unverified replacement")
        return result

    monkeypatch.setattr(components.runtime, "_component_json", snapshot)
    assert components._pinned_cu_card(card_api).sha256 == components.CU_MATERIAL_CARD_SHA256


def test_symlink_cu_card_is_refused(tmp_path, monkeypatch, card_api):
    target = components.DELIVERY / "CU_MATERIAL_CARD.json"
    (tmp_path / "CU_MATERIAL_CARD.json").symlink_to(target)
    monkeypatch.setattr(components, "DELIVERY", tmp_path)
    with pytest.raises(OSError):
        components._pinned_cu_card(card_api)


@pytest.mark.parametrize("module_name,key", [("mechanics_runtime", "NATIVE_ROOT"),
    ("YAP.mechanics_runtime", "NATIVE_ROOT"), ("history_mechanics_backend", "ANSYS_VALIDATION"),
    ("YAP.history_mechanics_backend", "ANSYS_VALIDATION")])
def test_already_captured_native_root_mismatch_requires_restart(tmp_path, module_name, key):
    selected = tmp_path / "selected"
    loaded = {module_name: SimpleNamespace(**{key: tmp_path / "old"})}
    with pytest.raises(ValueError, match="before Python and restart"):
        components._check_loaded_native_root(selected, modules=loaded)
    loaded[module_name] = SimpleNamespace(**{key: selected})
    components._check_loaded_native_root(selected, modules=loaded)


def test_lazy_runtime_is_not_captured_by_guard(tmp_path):
    class Lazy:
        def __getattr__(self, name):
            pytest.fail("native guard triggered lazy root capture")
    components._check_loaded_native_root(tmp_path, modules={"mechanics_runtime": Lazy()})


def test_preloaded_native_module_mismatch_is_refused(tmp_path):
    with pytest.raises(ValueError, match="different root"):
        components._check_loaded_native_root(tmp_path / "selected", modules={
            "exp12_all_gauss_reduced_fe2": SimpleNamespace(__file__=str(tmp_path / "old/engine.py"))})
