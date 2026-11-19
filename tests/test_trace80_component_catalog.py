"""TRACE80 evidence and catalog routing, without large numerical payload reads."""

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest

import component_operators as components
from tools import publish_component_operators as publisher


ASSESSMENT = (publisher.ROOT / "operator_night_20261021"
              / "P4_CYCLIC_BOW_ASSESSMENT_20261024_FULL73_v1/REPORT.json")


@pytest.fixture(scope="module")
def measured_evidence():
    if not ASSESSMENT.is_file():
        pytest.skip("optional measured TRACE80 coupon evidence is not installed")
    reader = publisher.MetadataReader()
    assessment, pin = reader.read(ASSESSMENT)
    result = publisher._assessment(reader, pin, 4, 37)
    artifacts, construction, reference = publisher._components(reader, result[1], 4, 37, result[3])
    reader.verify_unchanged()
    assert reader.used < 4 * 1024**2
    assert all(Path(path).suffix == ".json" for path in reader.cache)
    return dict(assessment=assessment, pin=pin, result=result,
                artifacts=artifacts, construction=construction, reference=reference,
                documents={path: deepcopy(row[:2]) for path, row in reader.cache.items()})


class MutatedReader:
    """Fault injection into authenticated metadata, not an evidence producer."""

    def __init__(self, measured, change=lambda document: None):
        self.documents = deepcopy(measured["documents"])
        for document, _ in self.documents.values():
            change(document)

    def read(self, binding):
        key = str(publisher.binding(binding)["path"] if isinstance(binding, dict)
                  else self.path(binding))
        return deepcopy(self.documents[key])

    def path(self, value):
        return publisher.shared_distribution_path(value)


@pytest.mark.parametrize("historical_form", ["string", "binding"])
@pytest.mark.parametrize("public_form", ["string", "binding"])
def test_mutated_reader_aliases_share_faults_without_sharing_returned_objects(
        measured_evidence, historical_form, public_form):
    historical = measured_evidence["assessment"]["rom_report"]
    relocated = publisher.binding(historical)

    def change(document):
        document["fixture_fault"] = {"visible": True}

    reader = MutatedReader(measured_evidence, change)
    historical_request = historical if historical_form == "binding" else historical["path"]
    public_request = relocated if public_form == "binding" else relocated["path"]
    first, first_pin = reader.read(historical_request)
    second, second_pin = reader.read(public_request)
    assert reader.path(historical["path"]) == reader.path(relocated["path"]) == Path(relocated["path"])
    assert first == second
    assert first["fixture_fault"] == {"visible": True}
    assert first_pin == second_pin == relocated
    assert first_pin["sha256"] == historical["sha256"]
    assert first is not second and first["fixture_fault"] is not second["fixture_fault"]
    first["fixture_fault"]["visible"] = False
    first_pin["sha256"] = "0" * 64
    again, again_pin = reader.read(historical_request)
    assert again == second and again["fixture_fault"]["visible"] is True
    assert again_pin == second_pin == relocated


def measured_entry(evidence):
    metrics = publisher._trace80_bow_metrics(evidence["assessment"])
    return dict(id="BI4_SI37", profile=components.TRACE80_PROFILE,
        geometry=deepcopy(publisher.TRACE80_GEOMETRY),
        components=deepcopy(evidence["artifacts"]), construction=deepcopy(evidence["construction"]),
        reference=deepcopy(evidence["reference"]),
        validation=dict(report=evidence["pin"], history=evidence["result"][2],
                        MAPDL_direct_validation=False, **metrics),
        capabilities=dict(single_bi_history=True, experimental_prepared_d2w=True,
                          prepared_assembly_accuracy=False, W2W_thinning=False))


def test_measured_trace80_chain_keeps_actual73_observations(measured_evidence):
    metrics = publisher._validate_bow_evidence(measured_evidence["assessment"])
    assert metrics["solved_states"] == metrics["bow_observations"] == 73
    assert metrics["max_absolute_um"] == pytest.approx(.848125442)
    assert metrics["max_relative_pct"] == pytest.approx(.3858223053)
    components._verify_validation(measured_entry(measured_evidence), MutatedReader(measured_evidence))


@pytest.mark.parametrize("fault", [
    "dense", "MAPDL", "assembly", "interpolation", "ordinal", "clock", "summary",
    "probe", "nonfinite", "bool", "reported_error", "q6", "state_chain", "Cu_reset",
    "material", "thermal_policy", "native_continuation", "projection", "rank", "payload",
])
def test_measured_chain_refuses_scope_lineage_and_response_changes(measured_evidence, fault):
    def change(document):
        schema = document.get("schema")
        if schema == "yap.p4.cyclic_bow_metadata_audit.v1":
            row = document["assessment"]["rows"][1]
            probe = row["probes"][0]
            if fault == "dense": document["dense_time_claim"] = True
            elif fault == "MAPDL": document["MAPDL_claim"] = True
            elif fault == "assembly": document["assembly_admission"] = True
            elif fault == "interpolation": document["assessment"]["interpolation"] = True
            elif fault == "ordinal": row["original_ordinal"] += 1
            elif fault == "clock": row["time_s"] += 1
            elif fault == "summary": document["assessment"]["total"]["max_absolute_error_um"] = 0.
            elif fault == "probe": row["probes"][1]["probe"] = probe["probe"]
            elif fault == "nonfinite": probe["predicted_um"] = float("nan")
            elif fault == "bool": probe["reference_um"] = True
            elif fault == "reported_error": probe["absolute_error_um"] += 1
        elif schema == "yap.component_reduced_clock.history.v1":
            if fault == "Cu_reset": document["context"]["Cu_state_adoption"] = True
            elif fault == "material": document["context"]["Cu_source"] = "0" * 64
            elif fault == "thermal_policy": document["context"]["thermal_policy"]["unmatched"] = True
        elif schema == "yap.p4.native_own_cyclic_tail.result.v1" and fault == "native_continuation":
            document["checkpoints"][-1]["own_state_continuation"] = False
        elif schema == "yap.trace80.component_operators.experimental.v1":
            if fault == "projection":
                document["projection_certificates"]["beol"]["all_GP_strain"]["absolute_error"] = 1e6
            elif fault == "rank": document["components"]["si"]["micro_rank"] += 1
            elif fault == "payload": document["components"]["bi"]["artifact"]["sha256"] = "0" * 64
        elif document.get("ordinal") == 1 and "diagnostics" in document:
            if fault == "q6": document["q6"][3] += .01
            elif fault == "state_chain": document["incoming_state_sha256"] = "0" * 64

    reader = MutatedReader(measured_evidence, change)
    with pytest.raises(ValueError):
        result = publisher._assessment(reader, measured_evidence["pin"], 4, 37)
        publisher._components(reader, result[1], 4, 37, result[3])


@pytest.mark.parametrize("fault", ["geometry", "count", "history", "metrics", "direct_MAPDL", "operand"])
def test_catalog_validation_cannot_relabel_measured_coupon(measured_evidence, fault):
    entry = measured_entry(measured_evidence)
    if fault == "geometry": entry["geometry"]["total_thickness_um"] = 50
    elif fault == "count": entry["validation"]["bow_observations"] = 1081
    elif fault == "history": entry["validation"]["history"]["sha256"] = "0" * 64
    elif fault == "metrics": entry["validation"]["max_absolute_um"] = 0.
    elif fault == "direct_MAPDL": entry["validation"]["MAPDL_direct_validation"] = True
    elif fault == "operand": entry["components"]["bi"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        components._verify_validation(entry, MutatedReader(measured_evidence))


@pytest.mark.parametrize("predicted,reference,passes", [
    (1., 0., False), (.999, 0., True), (11., 10., False), (10.999, 10., True),
    (1.1, 1., True), (float("inf"), 1., False),
])
def test_strict_or_gate_is_pointwise_not_only_a_summary(measured_evidence, predicted, reference, passes):
    report = deepcopy(measured_evidence["assessment"])
    for row in report["assessment"]["rows"]:
        for probe in row["probes"]:
            probe.update(predicted_um=predicted, reference_um=reference,
                absolute_error_um=abs(predicted-reference),
                relative_error_percent=100 * abs(predicted-reference) / abs(reference) if reference else None)
    for name in ("total", "C1_resubstitution", "C2_C3_heldout"):
        summary = report["assessment"][name]
        summary["max_absolute_error_um"] = abs(predicted-reference)
        summary["max_finite_relative_error_percent"] = (
            100 * abs(predicted-reference) / abs(reference) if reference else None)
    if passes:
        publisher._trace80_bow_metrics(report)
    else:
        with pytest.raises(ValueError):
            publisher._trace80_bow_metrics(report)


@pytest.fixture
def routed_catalog(tmp_path, monkeypatch):
    """Synthetic routing only: fake artifacts never enter numerical loading."""
    from W2W import trace80_component_bundle as adapter
    document = dict(schema="yap.component_operator_catalog.v3", entries=[dict(
        id="BI4_SI37", profile=components.TRACE80_PROFILE,
        geometry=deepcopy(publisher.TRACE80_GEOMETRY), components={
            role: dict(path=role+".npz", sha256="1"*64, bytes=32, micro_rank=publisher.TRACE80_RANKS[role])
            for role in components.ROLES},
        construction=dict(path="construction.json", sha256="2"*64),
        reference=dict(path="reference.json", sha256="3"*64),
        physical_geometry=dict(path="compact.json", sha256="4"*64),
        validation={key: dict(path=key+".json", sha256="5"*64)
                    for key in ("report", "history", "runtime_qualification")},
        capabilities=dict(single_bi_history=True, experimental_prepared_d2w=True,
                          prepared_assembly_accuracy=False, W2W_thinning=False))])
    path = tmp_path / "CATALOG.json"
    calls = []

    class Reader:
        def verify_unchanged(self): calls.append("unchanged")

    @contextmanager
    def no_payload(*args):
        calls.append(("stat_only", args))
        yield None

    monkeypatch.setattr(components.runtime, "_component_json", lambda p: (deepcopy(document), "a"*64))
    monkeypatch.setattr(components.runtime, "_component_open", no_payload)
    monkeypatch.setattr(components, "_metadata_reader", Reader)
    monkeypatch.setattr(components, "_verify_validation", lambda e, r: calls.append(("evidence", e)))
    monkeypatch.setattr(adapter, "validate_catalog_entry", lambda e, r: calls.append(("adapter", e)))
    return document, path, calls


def test_v3_uses_same_resolver_and_normalizes_every_bound_path(routed_catalog):
    _, path, calls = routed_catalog
    entry = components.resolve_entry("component:BI4_SI37", catalog_path=path)
    assert entry["id"] == "BI4_SI37"
    for binding in (entry["construction"], entry["reference"], entry["physical_geometry"],
                    *entry["components"].values(), *entry["validation"].values()):
        assert Path(binding["path"]).parent == path.parent
    assert len([row for row in calls if isinstance(row, tuple) and row[0] == "stat_only"]) == 3
    assert len([row for row in calls if isinstance(row, tuple) and row[0] == "adapter"]) == 1


@pytest.mark.parametrize("fault", ["v2", "extra", "profile", "missing_role", "capability"])
def test_v3_profile_is_not_a_legacy_guard_bypass(routed_catalog, fault):
    document, path, _ = routed_catalog
    entry = document["entries"][0]
    if fault == "v2": document["schema"] = "yap.component_operator_catalog.v2"
    elif fault == "extra": entry["unknown"] = True
    elif fault == "profile": entry["profile"] = "unqualified_trace80"
    elif fault == "missing_role": del entry["components"]["beol"]
    elif fault == "capability": entry["capabilities"]["prepared_assembly_accuracy"] = True
    with pytest.raises(ValueError):
        components.load_catalog(path)


def test_trace80_prepared_and_w2w_requests_retain_experimental_scope(routed_catalog, monkeypatch):
    _, path, _ = routed_catalog
    entry = components.resolve_entry("component:BI4_SI37", catalog_path=path)
    monkeypatch.setattr(components, "resolve_entry", lambda p: entry)
    geometry = dict(width_mm=10., length_mm=10., total_thickness_um=50.,
                    front_bi_thickness_um=1.5, back_bi_thickness_um=1.5, beol_thickness_um=10.)
    request = components.prepared_operator_request("component:BI4_SI37", geometry, layout=(4, 4, 2))
    assert request["component_profile"] == components.TRACE80_PROFILE
    assert request["validated_single_BI_total_um"] == 48.5
    assert request["assembly_accuracy_validated"] is False
    assert request["physical_readiness"] is False
    wafer = dict(diameter_mm=300., initial_total_thickness_um=775., target_total_thickness_um=50.,
                 front_bi_thickness_um=1.5, back_bi_thickness_um=1.5, beol_thickness_um=10.)
    wafer_request = components.w2w_operator_request("component:BI4_SI37", wafer, layout=(4, 4, 2))
    assert wafer_request["component_profile"] == components.TRACE80_PROFILE
    assert wafer_request["validated_single_BI_total_um"] == 48.5
    assert wafer_request["initial_total_thickness_um"] == 775.
    assert wafer_request["target_total_thickness_um"] == 50.
    assert wafer_request["admission_status"] == "EXPERIMENTAL_W2W_COMPONENTS_REQUIRE_CHECKED_SILICON_CAPS"
    assert wafer_request["W2W_thinning_available"] is True
    for key in ("assembly_accuracy_validated", "physical_readiness", "operator_inputs_verified"):
        assert wafer_request[key] is False
    assert entry["capabilities"]["W2W_thinning"] is False


@pytest.mark.parametrize("changes,layout", [
    ({"initial_total_thickness_um": 774.}, (4, 4, 2)),
    ({"target_total_thickness_um": 775.}, (4, 4, 2)),
    ({"target_total_thickness_um": 40.}, (4, 4, 2)),
    ({}, (6, 6, 3)),
])
def test_trace80_w2w_request_rejects_invalid_wafer_geometry_and_layout(routed_catalog, monkeypatch,
                                                                    changes, layout):
    _, path, _ = routed_catalog
    entry = components.resolve_entry("component:BI4_SI37", catalog_path=path)
    monkeypatch.setattr(components, "resolve_entry", lambda p: entry)
    wafer = dict(diameter_mm=300., initial_total_thickness_um=775., target_total_thickness_um=50.,
                 front_bi_thickness_um=1.5, back_bi_thickness_um=1.5, beol_thickness_um=10.)
    with pytest.raises(ValueError):
        components.w2w_operator_request("component:BI4_SI37", dict(wafer, **changes), layout=layout)


def test_publisher_emits_actual73_profile_only_after_runtime_validator(measured_evidence, monkeypatch):
    from W2W import trace80_component_bundle as adapter
    locator = dict(pitch_um=4, Si_TSV_um=37, evidence=measured_evidence["pin"],
        physical_geometry=dict(path=str(ASSESSMENT.parent / "synthetic_compact_manifest.json"), sha256="7"*64),
        runtime_qualification=dict(path=str(ASSESSMENT.parent / "synthetic_qualification.json"), sha256="8"*64))
    checked = []
    monkeypatch.setattr(adapter, "validate_catalog_entry", lambda e, r: checked.append(deepcopy(e)))
    entry = publisher._trace80_entry(MutatedReader(measured_evidence), locator)
    assert checked == [entry]
    assert entry["profile"] == components.TRACE80_PROFILE
    assert entry["geometry"]["total_thickness_um"] == 48.5
    assert entry["validation"]["bow_observations"] == 73
    assert entry["validation"]["MAPDL_direct_validation"] is False
    assert entry["capabilities"]["prepared_assembly_accuracy"] is False

    def refuse(*args):
        raise ValueError("runtime qualification absent")

    monkeypatch.setattr(adapter, "validate_catalog_entry", refuse)
    with pytest.raises(ValueError, match="runtime qualification absent"):
        publisher._trace80_entry(MutatedReader(measured_evidence), locator)


@pytest.mark.parametrize("missing", ["physical_geometry", "runtime_qualification"])
def test_publisher_refuses_coupon_only_without_actual_loading_qualification(measured_evidence, missing):
    locator = dict(pitch_um=4, Si_TSV_um=37, evidence=measured_evidence["pin"],
        physical_geometry={}, runtime_qualification={})
    del locator[missing]
    with pytest.raises(ValueError, match="complete runtime evidence"):
        publisher._trace80_entry(MutatedReader(measured_evidence), locator)


@pytest.mark.parametrize("with_trace80", [False, True])
def test_publisher_preserves_v2_until_a_qualified_profile_is_appended(monkeypatch, with_trace80):
    coverage = dict(single_BI_p2_full_three_cycle_passed=[
        dict(Si_TSV_um=si, evidence={}) for si in publisher.P2_SI],
        other_pitches=[dict(pitch_um=pitch, evidence={}) for pitch in (3, 6)])
    if with_trace80:
        coverage["trace80_passed"] = [dict(synthetic_routing_only=True)]

    class Reader:
        def __init__(self, root): pass
        def read(self, value):
            return dict(audited_current_summary_20261021=dict(coverage=coverage)), {}
        def verify_unchanged(self): pass

    monkeypatch.setattr(publisher, "MetadataReader", Reader)
    monkeypatch.setattr(publisher, "_assessment", lambda *a: ({}, {}, {}, 73, 0., 0.))
    monkeypatch.setattr(publisher, "_components", lambda *a: ({}, {}, {}))
    monkeypatch.setattr(publisher, "_trace80_entry", lambda *a: dict(id="BI4_SI37"))
    catalog = publisher.build_catalog()
    assert catalog["schema"] == (publisher.TRACE80_CATALOG_SCHEMA if with_trace80 else publisher.SCHEMA)
    assert len(catalog["entries"]) == len(publisher.P2_SI) + 2 + int(with_trace80)
    assert (catalog["entries"][-1]["id"] == "BI4_SI37") is with_trace80
