"""W2W component wiring and retained-state contracts, not physics admission."""

from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import Mock

import numpy as np
from omegaconf import OmegaConf
import pytest

import component_operators as components
import generate_assembly_design as generator
from D2W import assembly_order as die_order
from W2W import assembly_mechanics as mechanics
from W2W import assembly_operator_library as operators
from W2W import assembly_order as order
from W2W import assembly_parallel as parallel
from W2W import component_assembly_bridge as bridge


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = components.PROJECT / "user_runs"
INPUT = ROOT / "W2W/input/f2b_4hi50_p24_preview/stack.yaml"
CASES = [(f"BI2_SI{str(si).replace('.', 'p')}", float(si), 2.)
         for si in (7, 17, 27, 37, 47, 48.5, 57, 67, 77, 87, 88.5)] + [
    ("BI3_SI37_COARSE45", 37., 3.), ("BI6_SI37", 37., 6.)]
CU_FIELDS = ("plastic_strain", "creep_strain", "equivalent_plastic_strain",
             "equivalent_creep_strain")


def entry(name="BI2_SI37", si=37., pitch=2.):
    return dict(id=name, geometry=dict(bi_pitch_um=pitch, diameter_pitch_ratio=.5,
        bi_thickness_um=1.5, beol_thickness_um=10., si_tsv_thickness_um=si,
        total_thickness_um=si+11.5, bi_layer_count=1, tsv_pitch_um=6., cell_side_um=6.),
        validation=dict(cycles=3, solved_states=73, bow_observations=1081,
                        MAPDL_direct_validation=False))


def geometry(total=50.):
    return dict(diameter_mm=300., initial_total_thickness_um=775.,
        target_total_thickness_um=total, front_bi_thickness_um=1.5,
        back_bi_thickness_um=1.5, beol_thickness_um=10.)


def design_and_plan(total=50., domain="w2w"):
    module = order if domain == "w2w" else die_order
    path = INPUT if domain == "w2w" else ROOT / "D2W/input/design_6_4hi_p2/stack.yaml"
    design = module.load_stack_input(path)
    design["mechanics"].update(operator_package="component:BI2_SI37",
                               temporal_profile="component_graded25")
    design["mechanics"].pop("frozen_preview", None)
    if domain == "w2w":
        design["stack"]["geometry"]["target_total_thickness_um"] = total
    return design, module.build_plan(design["stack"])


@pytest.fixture(autouse=True)
def canonical_test_outputs(tmp_path):
    assert tmp_path.resolve().is_relative_to(OUTPUT), "use the canonical --basetemp"


@pytest.mark.parametrize("name,si,pitch", CASES + [("BI2_SI32", 32., 2.)])
def test_same_thirteen_catalog_and_parametric45_metadata(monkeypatch, name, si, pitch):
    item = entry(name, si, pitch)
    if name == "BI2_SI32":
        item.update(kind="endpoint_thickness_family")
        item["validation"] = dict(heldout_si_um=32., solved_states=73,
                                   bow_observations=292, prepared_assembly_accuracy=False)
    monkeypatch.setattr(components, "resolve_entry", lambda *_a, **_k: item)
    forbidden = Mock(side_effect=AssertionError("metadata request loaded numeric operands"))
    monkeypatch.setattr(components, "OperandReader", forbidden)
    monkeypatch.setattr(components, "configure_component_runtime", forbidden)
    values, original = geometry(si+13.), deepcopy(item)
    before = deepcopy(values)
    request = components.w2w_operator_request("component:"+name, values,
                                             layout=(pitch, pitch, pitch/2.))
    assert request["component_id"] == name
    assert request["retained_si_thickness_um"] == si
    assert request["validated_single_BI_total_um"] == si+11.5
    assert request["target_total_thickness_um"] == si+13.
    assert request["initial_total_thickness_um"] == 775.
    assert request["original_outer_si_um"] == 1.5
    assert request["removable_bulk_si_um"] == 775.-(si+13.)
    assert request["unit_history_validation"] == original["validation"]
    assert request["physical_readiness"] is False
    assert request["assembly_accuracy_validated"] is False
    assert request["operator_inputs_verified"] is False
    assert request["bonding_pitch_um"] == pitch
    assert request["pad_diameter_um"] == pitch/2.
    assert item == original and values == before
    forbidden.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("diameter_mm", 0.), ("diameter_mm", -1.), ("diameter_mm", True),
    ("diameter_mm", float("nan")), ("diameter_mm", float("inf")),
    ("initial_total_thickness_um", 774.), ("initial_total_thickness_um", True),
    ("target_total_thickness_um", 48.5), ("target_total_thickness_um", 775.),
    ("target_total_thickness_um", 776.), ("target_total_thickness_um", 13.),
    ("front_bi_thickness_um", 3.), ("back_bi_thickness_um", 0.),
    ("back_bi_thickness_um", 3.), ("beol_thickness_um", 11.)])
def test_invalid_wafer_geometry_is_rejected(monkeypatch, field, value):
    monkeypatch.setattr(components, "resolve_entry", lambda *_a, **_k: entry())
    values = geometry()
    values[field] = value
    with pytest.raises(ValueError):
        components.w2w_operator_request("component:BI2_SI37", values)


@pytest.mark.parametrize("fault", ("missing", "extra", "prepared", "preview", "pitch", "pad"))
def test_wrong_domain_layout_and_preview_fail_closed(monkeypatch, fault):
    monkeypatch.setattr(components, "resolve_entry", lambda *_a, **_k: entry())
    values, options = geometry(), {}
    if fault == "missing":
        values.pop("diameter_mm")
    elif fault == "extra":
        values["width_mm"] = 300.
    elif fault == "prepared":
        values = dict(width_mm=10., length_mm=10., total_thickness_um=50.,
            front_bi_thickness_um=1.5, back_bi_thickness_um=1.5, beol_thickness_um=10.)
    elif fault == "preview":
        options["frozen_preview"] = True
    else:
        options["layout"] = (6., 6., 3.) if fault == "pitch" else (2., 2., .5)
    with pytest.raises(ValueError):
        components.w2w_operator_request("component:BI2_SI37", values, **options)


def test_f2b_loader_delegates_components_without_opening_legacy_payload(monkeypatch):
    result = (object(), {"fixture": "not physics"})
    load = Mock(return_value=result)
    monkeypatch.setattr(components, "load_w2w_component_library", load)
    forbidden = Mock(side_effect=AssertionError("component request entered old cap factory"))
    monkeypatch.setattr(operators, "_load_reference_p2_preview_library", forbidden)
    values = geometry(45.)
    assert operators.load_f2b_assembly_library("component:BI2_SI32", values) is result
    load.assert_called_once_with("component:BI2_SI32", values, frozen_preview=False)
    forbidden.assert_not_called()


def test_component_loader_preserves_cu_library_and_passes_actual_trace(monkeypatch):
    original, augmented, trace = object(), object(), object()
    proof = dict(component_validation={"scope": "single BI bow"},
                 corrected_material={"source_sha256": "pinned"}, operator_inputs={"bi": "bound"})
    before = deepcopy(proof)
    build = Mock(return_value=(original, proof, trace))
    monkeypatch.setattr(components, "_component_library_and_trace", build)
    monkeypatch.setattr(components, "resolve_entry", lambda *_a, **_k: entry())
    cap = ModuleType("W2W.component_silicon_cap")
    cap.augment_w2w_library = Mock(return_value=(augmented, {"Cu_points": 0}))
    monkeypatch.setitem(sys.modules, cap.__name__, cap)
    values = geometry()
    library, provenance = components.load_w2w_component_library("component:BI2_SI37", values)
    assert library is augmented
    cap.augment_w2w_library.assert_called_once_with(original, values, trace)
    supplied = build.call_args.args[1]
    assert supplied == dict(width_mm=300., length_mm=300., total_thickness_um=50.,
        front_bi_thickness_um=1.5, back_bi_thickness_um=1.5, beol_thickness_um=10.)
    assert proof == before
    for key in ("component_validation", "corrected_material", "operator_inputs"):
        assert provenance[key] == before[key]
    assert provenance["geometry"] == values
    assert provenance["w2w_si_caps"] == {"Cu_points": 0}


def trace_fixture():
    lateral = np.array([(x, z) for x in np.linspace(0., .006, 8)
                        for z in np.linspace(0., .006, 8)])
    xyz = np.zeros((2, 64, 3))
    xyz[:, :, 0], xyz[:, :, 2] = lateral[:, 0], lateral[:, 1]
    xyz[1, :, 1] = .0015
    trace = np.zeros((192, 64))
    for axis in range(3):
        trace[axis::3, axis] = 1.
    for column in range(3, 64):
        trace[3*(column-3), column], trace[-3, column] = 1., -1.
    lifts = np.zeros((2, 192, 134))
    for face, offset in ((0, 6), (1, 70)):
        lifts[face, :, :6] = bridge._canonical_macro_trace(xyz[face], (face-.5)*.0015)
        lifts[face, :, offset:offset+64] = trace
    physical = dict(face_node_ids=np.arange(1, 129).reshape(2, 64),
                    face_coordinates_mm=xyz, face_lifts=lifts)
    reflection = np.eye(64)
    reflection[1, 1] = -1.
    return physical, reflection


@pytest.mark.parametrize("fault", (None, "top_order", "reflection", "rank", "macro"))
def test_actual_trace_contract_rejects_incompatible_faces(fault):
    physical, reflection = trace_fixture()
    if fault == "top_order":
        physical["face_lifts"][1, :, [73, 74]] = physical["face_lifts"][1, :, [74, 73]]
    elif fault == "reflection":
        reflection[1, 1] = 1.
    elif fault == "rank":
        physical["face_lifts"][0, :, 69] = physical["face_lifts"][0, :, 68]
        physical["face_lifts"][1, :, 133] = physical["face_lifts"][1, :, 132]
    elif fault == "macro":
        physical["face_lifts"][1, 0, 3] += 1e-5
    context = nullcontext() if fault is None else pytest.raises(ValueError)
    with context:
        bridge._trace_contract(physical, {"c": np.zeros((0, 134))}, .0015, .006**2,
                               physical_v2=True, reflection=reflection)


@pytest.mark.parametrize("total", (20., 45., 50., 61.5, 100., 101.5))
def test_every_f2b_tree_uses_775_birth_and_two_distinct_thinning_ledgers(total):
    _, plan = design_and_plan(total)
    assert len(plan["routes"]) == 5
    for route in plan["routes"]:
        assert route["final"]["total_thicknesses_um"] == [total]*4
        for event in route["events"]:
            assert event["mating_faces"] == dict(lower="BACK", upper="FRONT")
            assert event["lower_before_preparation"]["total_thicknesses_um"][-1] == 775.
            assert event["thinning"]["retained_si_um"] == total-13.
            assert event["thinning"]["removed_original_pure_si_um"] == 775.-total+1.5
            assert event["thinning"]["new_back_bi_um"] == 1.5
        assert route["finalization"]["removed_pure_si_um"] == 775.-total
        assert route["finalization"]["retained_original_pure_si_um"] == 1.5
        assert route["finalization"]["new_back_bi_um"] == 0.


@pytest.mark.parametrize("finish_outer", (False, True))
def test_thinning_preserves_retained_cu_birth_and_ages(finish_outer):
    from test_w2w_f2b import library as tiny_library

    _, plan = design_and_plan()
    run = mechanics.AssemblyExecutor(plan, tiny_library(True), temporal_profile="component_graded25")
    body = run.leaf(4)
    assert [p.role for p in body.placements[-2:]] == ["outer_retained_si", "removable_bulk_si"]
    assert body.placements[-1].z_top_mm-body.placements[0].z_bottom_mm == pytest.approx(.775)
    for index, (name, layer) in enumerate(body.state.layers.items(), 1):
        body.ages_s[name] = 5200.+index
        for field in CU_FIELDS:
            getattr(layer.cu_state, field)[:] = index*1e-7
    before = run.checkpoint(body)
    changed, audit = (run.finish_outer_back(body) if finish_outer else run.prepare_back(body))
    assert run.checkpoint(body) == before
    assert changed.placements[-1].z_top_mm-changed.placements[0].z_bottom_mm == pytest.approx(.050)
    removed = [body.placements[-1].name] if finish_outer else [p.name for p in body.placements[-2:]]
    assert audit["removal"]["removed_names"] == removed
    for placement in changed.placements:
        new = changed.state.layers[placement.name]
        if placement.role == "back_bi":
            assert not finish_outer and placement.thickness_orientation == -1
            assert changed.ages_s[placement.name] == run.schedule[0].time_s
            assert all(not np.any(getattr(new.cu_state, field)) for field in CU_FIELDS)
            continue
        assert any(placement is old for old in body.placements)
        old = body.state.layers[placement.name]
        assert changed.ages_s[placement.name] == body.ages_s[placement.name]
        for field in CU_FIELDS:
            np.testing.assert_array_equal(getattr(new.cu_state, field), getattr(old.cu_state, field))
            assert not np.shares_memory(getattr(new.cu_state, field), getattr(old.cu_state, field))
        for field in ("birth_macro_coordinates", "birth_bubble_coordinates"):
            np.testing.assert_array_equal(getattr(new, field), getattr(old, field))
    if finish_outer:
        assert changed.placements[-1] is body.placements[-2]
        assert audit["new_back_BI"] is False and audit["silicon_deposition"] is False


@pytest.mark.parametrize("domain", ("d2w", "w2w"))
@pytest.mark.parametrize("workers", (1, 2))
def test_settings_dispatch_preserves_component_domain(monkeypatch, tmp_path, domain, workers):
    design, plan = design_and_plan(domain=domain)
    settings = dict(design["mechanics"], order_workers=workers)
    monkeypatch.setattr(mechanics, "_configure_mechanics_runtime_environment", Mock())
    monkeypatch.setattr(mechanics, "_recipe", Mock())
    monkeypatch.setattr(operators, "require_prepared_operator", Mock())
    library, proof, result = object(), {"fixture": "not physics"}, object()
    expected = "load_f2b_assembly_library" if domain == "w2w" else "load_prepared_die_library"
    loaders = {}
    for name in ("load_assembly_library", "load_f2b_assembly_library", "load_prepared_die_library"):
        loaders[name] = Mock(return_value=(library, proof)) if name == expected else Mock(
            side_effect=AssertionError("wrong domain loader"))
        monkeypatch.setattr(operators, name, loaders[name])
    run = Mock(return_value=result)
    dispatch = Mock(return_value=result)
    monkeypatch.setattr(mechanics, "run_native", run)
    monkeypatch.setattr(parallel, "run_parallel", dispatch)
    directory = tmp_path / "workers"
    assert mechanics.run_native_from_settings(plan, settings, worker_directory=directory) is result
    if workers == 1:
        dispatch.assert_not_called()
        loaders[expected].assert_called_once_with(settings["operator_package"], plan["context"]["geometry"])
        assert run.call_args.args == (plan, library, proof)
    else:
        run.assert_not_called()
        assert dispatch.call_args.args == (plan, settings)
        assert dispatch.call_args.kwargs["work_dir"] == directory
        assert all(not loader.called for loader in loaders.values())


@pytest.mark.parametrize("domain", ("d2w", "w2w"))
def test_parallel_worker_selects_component_domain_once(monkeypatch, tmp_path, domain):
    import mechanics_runtime
    import threadpoolctl

    design, plan = design_and_plan(domain=domain)
    shard = tmp_path / "worker"
    shard.mkdir()
    for key, value in parallel._environment(shard).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(mechanics_runtime, "configure_runtime_storage", Mock())
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", lambda **_kw: nullcontext())
    pins = {str(Path(__file__).resolve()): parallel._hash(Path(__file__))}
    monkeypatch.setattr(parallel, "_sources", lambda: dict(pins))
    monkeypatch.setattr(mechanics, "_recipe", Mock())
    monkeypatch.setattr(operators, "require_prepared_operator", Mock())
    library, proof = object(), {"fixture": "not physics"}
    loader = Mock(return_value=(library, proof))
    expected = "load_f2b_assembly_library" if domain == "w2w" else "load_prepared_die_library"
    for name in ("load_assembly_library", "load_f2b_assembly_library", "load_prepared_die_library"):
        monkeypatch.setattr(operators, name, loader if name == expected else Mock(
            side_effect=AssertionError("worker used wrong domain")))

    def run(request, supplied, provenance, *, route_indices, route_complete, **_kw):
        assert request == plan and supplied is library and provenance is proof
        assert route_indices == [0, 2]
        for index in route_indices:
            route_complete(order.archive_template(plan), plan["routes"][index])

    monkeypatch.setattr(mechanics, "run_native", run)
    settings = dict(design["mechanics"], order_workers=2)
    manifest = dict(run_id="component-domain-fixture", worker_id=0, plan=plan,
                    route_indices=[0, 2], settings=settings, source_pins=pins)
    path = shard / "manifest.json"
    parallel._atomic(path, manifest)
    parallel._worker(path)
    loader.assert_called_once_with(settings["operator_package"], plan["context"]["geometry"])
    terminal = parallel._read(shard / "terminal.json")
    assert terminal["status"] == "SUCCESS" and terminal["route_indices"] == [0, 2]


@pytest.mark.parametrize("field", ("trace_sha256", "material", "operator_sha256", "construction_sha256"))
def test_parallel_merge_binds_the_silicon_cap_scientific_identity(field):
    caps = dict(trace_sha256="trace", material={"elastic_modulus_mpa": 91800.},
        operator_sha256="operator", construction_sha256="construction",
        segments={"outer": {"cache_hit": False, "elapsed_seconds": 10.}})
    header = dict(provenance={"implementation_sha256": "same", "operator_set_sha256": "worker0"},
        native_configuration={"operator_provenance": {"component_id": "BI2_SI32", "w2w_si_caps": caps}})
    other = deepcopy(header)
    other["native_configuration"]["operator_provenance"]["w2w_si_caps"][field] = (
        {"elastic_modulus_mpa": 1.} if field == "material" else "different")
    assert parallel._static_identity(header) != parallel._static_identity(other)
    timing_only = deepcopy(header)
    timing_only["provenance"]["operator_set_sha256"] = "worker1"
    timing_only["native_configuration"]["operator_provenance"]["w2w_si_caps"]["segments"] = {
        "outer": {"cache_hit": True, "elapsed_seconds": .001}}
    assert parallel._static_identity(header) == parallel._static_identity(timing_only)


@pytest.mark.parametrize("domain", ("d2w", "w2w"))
def test_generator_preserves_domain_and_parametric45_target(tmp_path, domain):
    result = generator.generate_design(name="component45_"+domain, dies=2, pitch=2,
        total_thickness=45., initial_bows=[10., -10.], bonding_mode=domain,
        output_root=tmp_path / "generated")
    module = order if domain == "w2w" else die_order
    design = module.load_stack_input(result["input_dir"] / "stack.yaml")
    plan = module.build_plan(design["stack"])
    assert design["mechanics"]["operator_package"] == "component:BI2_SI32"
    assert design["mechanics"]["temporal_profile"] == "component_graded25"
    assert domain.upper() in result["input_dir"].parts
    assert plan["routes"][0]["final"]["total_thicknesses_um"] == [45., 45.]
    definitions = OmegaConf.load(result["input_dir"] / "generated_chiplet_definitions.3dbv").ChipletDef
    prefix, initial = ("Wafer", 775.) if domain == "w2w" else ("Die", 45.)
    assert set(definitions) == {prefix+"_1", prefix+"_2"}
    assert all(value.thickness == initial for value in definitions.values())
    if domain == "w2w":
        assert plan["schema"] == order.F2B_PLAN_SCHEMA
        assert plan["context"]["geometry"] == geometry(45.)
        assert plan["routes"][0]["events"][0]["thinning"]["retained_si_um"] == 32.


def test_generator_cli_wafers_alias_forwards_explicit_domain(monkeypatch, tmp_path):
    generate = Mock(return_value=dict(input_dir=tmp_path, config_path=tmp_path / "calculator.yaml",
        orders=1, operator_package="component:BI2_SI32"))
    monkeypatch.setattr(generator, "generate_design", generate)
    assert generator.main(["--name", "wafers45", "--wafers", "2", "--pitch", "2",
        "--total-thickness", "45", "--initial-bows", "10", "-10",
        "--bonding-mode", "w2w"]) == 0
    assert generate.call_args.kwargs["dies"] == 2
    assert generate.call_args.kwargs["bonding_mode"] == "w2w"


@pytest.fixture(scope="module")
def cap_cache_certificate():
    from W2W import component_silicon_cap as cap
    from test_component_silicon_cap import fixture_inputs

    library, values, trace = fixture_inputs()
    _, xs, zs, scale, _, _ = cap._validate(library, values, trace)
    sources = cap._native_sources()
    ports = trace["face_lifts"][0, :, 6:70]
    stiffness, load, lift, interior, mesh = cap._assemble(xs, zs, ports, 1.5, 3, *sources[:2])
    condensed, thermal, audits = cap._condense_fe(stiffness, load, lift, interior, scale)
    binding = dict(trace_sha256=cap._hash_arrays({"ports": ports}), material=dict(cap.MATERIAL),
        construction_sha256="a"*64, coordinate_contract_sha256="b"*64,
        face_periodic_limit_mm=1e-12)
    specification = dict(binding, thickness_um=1.5, thickness_intervals=3)
    key = cap.hashlib.sha256(cap._json(specification).encode()).hexdigest()
    metadata = dict(specification=specification, mesh=mesh, audits=audits, build_seconds=0.,
        approximation_scope="full_periodic_interior_FE_actual_restricted_face_basis")
    return dict(cap=cap, library=library, xs=xs, zs=zs, ports=ports, scale=scale,
        sources=sources, binding=binding, key=key, stiffness=condensed, load=thermal,
        metadata=metadata)


def test_valid_independent_cap_certificate_can_be_read(tmp_path, cap_cache_certificate):
    fixture = cap_cache_certificate
    cap, key = fixture["cap"], fixture["key"]
    proof = cap._publish_cache(tmp_path, key, fixture["stiffness"], fixture["load"],
                               deepcopy(fixture["metadata"]))
    cached = cap._read_cache(tmp_path/(key+".npz"), tmp_path/(key+".json"), key)
    np.testing.assert_array_equal(cached[0], fixture["stiffness"])
    np.testing.assert_array_equal(cached[1], fixture["load"])
    assert cached[2] == proof


@pytest.mark.parametrize("cache_kind", ("writable", "published"))
@pytest.mark.parametrize("fault", (
    "missing_specification", "changed_thickness", "missing_audits", "empty_audits",
    "missing_reaction", "reaction_failed", "thermal_failed", "energy_failed",
    "reaction_nan", "thermal_infinite", "weakened_gate", "negative_energy",
    "unexpected_nullspace", "missing_thermal_rhs"))
def test_cap_hits_reject_invalid_proofs_without_rebuilding(
        tmp_path, monkeypatch, cap_cache_certificate, cache_kind, fault):
    fixture = cap_cache_certificate
    cap, key = fixture["cap"], fixture["key"]
    writable, published = tmp_path/"writable", tmp_path/"published"
    directory = published if cache_kind == "published" else writable
    directory.mkdir()
    cap._publish_cache(directory, key, fixture["stiffness"], fixture["load"],
                       deepcopy(fixture["metadata"]))
    path = directory/(key+".json")
    metadata = json.loads(path.read_text())
    if fault == "missing_specification":
        metadata.pop("specification")
    elif fault == "changed_thickness":
        metadata["specification"]["thickness_um"] = 10.
    elif fault == "missing_audits":
        metadata.pop("audits")
    elif fault == "empty_audits":
        metadata["audits"] = {}
    elif fault == "missing_reaction":
        metadata["audits"].pop("full_reaction_relative_defect")
    elif fault in ("reaction_failed", "thermal_failed", "energy_failed"):
        field = {"reaction_failed": "full_reaction_relative_defect",
                 "thermal_failed": "thermal_reaction_relative_defect",
                 "energy_failed": "energy_relative_defect"}[fault]
        metadata["audits"][field] = cap.REACTION_LIMIT*2
    elif fault == "reaction_nan":
        metadata["audits"]["full_reaction_relative_defect"] = float("nan")
    elif fault == "thermal_infinite":
        metadata["audits"]["thermal_interior_relative_defect"] = float("inf")
    elif fault == "weakened_gate":
        metadata["audits"]["algebra_gate_limit"] = cap.REACTION_LIMIT*100
    elif fault == "negative_energy":
        metadata["audits"]["minimum_scaled_eigenvalue_over_spectral_radius"] = -.1
    elif fault == "unexpected_nullspace":
        metadata["audits"]["fourth_scaled_eigenvalue_over_spectral_radius"] = 0.
    else:
        metadata["audits"]["solve_rhs_count"] = 134
    path.write_text(json.dumps(metadata)+"\n")
    before_npz, before_json = cap._file_sha(directory/(key+".npz")), path.read_bytes()
    monkeypatch.setattr(cap, "_published_root", lambda: published)
    forbidden = Mock(side_effect=AssertionError("invalid cap proof reached FE or material construction"))
    monkeypatch.setattr(cap, "_assemble", forbidden)
    monkeypatch.setattr(operators, "_linear_operator", forbidden)
    with pytest.raises(ValueError):
        cap._segment(writable, fixture["binding"], fixture["xs"], fixture["zs"],
            fixture["ports"], fixture["scale"], 1.5, fixture["library"], fixture["sources"], intervals=3)
    forbidden.assert_not_called()
    assert cap._file_sha(directory/(key+".npz")) == before_npz
    assert path.read_bytes() == before_json
    if cache_kind == "published":
        assert not writable.exists()
