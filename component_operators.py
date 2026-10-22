"""Shared, bow-validated components and experimental prepared-die assembly.

Catalog metadata is cheap to inspect. Numeric operands are verified only for
the selected geometry. Each library has independent physical-layer states;
the component's unit-history validation remains attached to every result.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from types import FunctionType
import zipfile

import mechanics_runtime as runtime


PROJECT = runtime.COMPONENT_PROJECT_ROOT
DELIVERY = PROJECT / "distribution/yap_component_operators_20261021_v1"
DEFAULT_CATALOG = DELIVERY / "CATALOG.json"
DEFAULT_THICKNESS_FAMILY = PROJECT / "distribution/yap_thickness_family_20261021_v2"
CU_MATERIAL_CARD_SHA256 = "b3a876bcd8847e5ee2366580155b66ef5a3c01a7971a3b44b152b47eb46db8aa"
ARRAY_LIMIT = 6 * 1024**3
READ_LIMIT = 24 * 1024**3
ROLES = ("bi", "beol", "si")
ARRAY_KEYS = ("kaa", "c", "kff_data", "kff_indices", "kff_indptr", "kff_shape",
              "fm", "ff", "B_macro", "B_free", "weights", "reference_cu_stiffness",
              "reference_cu_alpha", "reference_temperature_c")
GP_KEYS = ("Cu_native_element_ids", "Cu_native_gauss_indices", "Cu_positions_mm", "weights")


def _require(value, message):
    if not value:
        raise ValueError(message)


def _binding(value, base):
    _require(isinstance(value, dict) and {"path", "sha256"} <= value.keys(),
             "a bound component artifact is required")
    result = dict(value, path=str(runtime._component_path(value["path"], base)))
    runtime._component_sha(result["sha256"])
    if "bytes" in result:
        _require(type(result["bytes"]) is int and 0 < result["bytes"] <= ARRAY_LIMIT,
                 "invalid component artifact byte count")
    return result


def _same(left, right):
    return all(left.get(key) == right.get(key) for key in ("path", "sha256"))


def _document(binding):
    return runtime._component_json(binding["path"], binding["sha256"])[0]


def _projected_work(component):
    proof = component.get("projection_certificate", {})
    _require(proof.get("macro_coordinates_retained") == 134
             and proof.get("Cu_updates") == 0
             and proof.get("nonlinear_law_compatibility_by_identical_strain_and_virtual_work") is True,
             "selected component needs its complete physical projection certificate")
    for name in ("all_GP_strain", "all_macro_point_stress_virtual_work",
                 "arbitrary_point_stress_body_virtual_work", "body_reference_action",
                 "macro_reference_action", "free_thermal_field"):
        row = proof.get(name, {})
        error, scale = row.get("absolute_error"), row.get("reference_norm")
        _require(type(error) in (int, float) and type(scale) in (int, float)
                 and math.isfinite(error) and math.isfinite(scale) and min(error, scale) >= 0
                 and error <= 1e-12 + 2e-8 * scale,
                 "component strain/virtual-work certificate failed: " + name)
    return proof


def _verify_validation(entry, reader):
    """Use the publisher's exact assessment/candidate/artifact evidence chain."""
    from publish_component_operators import _assessment, _components

    geometry, validation = entry["geometry"], entry["validation"]
    pitch, si = geometry["bi_pitch_um"], geometry["si_tsv_thickness_um"]
    _require(pitch in (2, 3, 6) and (pitch == 2 or si == 37)
             and geometry == dict(bi_pitch_um=pitch, diameter_pitch_ratio=.5,
                 bi_thickness_um=1.5, beol_thickness_um=10, si_tsv_thickness_um=si,
                 total_thickness_um=si+11.5, bi_layer_count=1, tsv_pitch_um=6, cell_side_um=6),
             "catalog geometry differs from the measured component family")
    report_pin, history, history_pin, states, relative, absolute = _assessment(
        reader, validation["report"], pitch, si)
    _require(_same(report_pin, validation["report"]) and _same(history_pin, validation["history"]),
             "bow assessment binds a different candidate history")
    _require(all(validation.get(key) == value for key, value in dict(
        cycles=3, solved_states=states, bow_observations=1081,
        max_relative_pct=relative, max_absolute_um=absolute).items()),
        "catalog bow metrics differ from the bound assessment")
    components, construction, reference = _components(reader, history, pitch, si, states)
    _require(_same(construction, entry["construction"]) and _same(reference, entry["reference"]),
             "validated history used a different operator or reference")
    for role in ROLES:
        _require(components[role] == {key: entry["components"][role][key]
                                     for key in ("path", "sha256", "bytes", "micro_rank")},
                 "validated history used a different component operand")


def load_catalog(catalog_path=None):
    """Read exact construction, reference and bow evidence without NPZ payloads."""
    path = runtime._component_path(catalog_path or os.environ.get(
        "YAP_COMPONENT_OPERATOR_CATALOG_V2", DEFAULT_CATALOG))
    catalog, catalog_sha = runtime._component_json(path)
    _require(set(catalog) == {"schema", "entries"}
             and catalog["schema"] == "yap.component_operator_catalog.v2"
             and isinstance(catalog["entries"], list) and 1 <= len(catalog["entries"]) <= 128,
             "invalid shared component v2 catalog")
    from publish_component_operators import MetadataReader
    evidence_reader = MetadataReader(PROJECT)
    entries, identifiers = [], set()
    required = {"id", "geometry", "components", "construction", "reference", "validation", "capabilities"}
    for original in catalog["entries"]:
        _require(isinstance(original, dict) and set(original) == required,
                 "invalid shared component entry")
        item = json.loads(json.dumps(original, allow_nan=False))
        name = item["id"]
        _require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", name)
                 and name not in identifiers, "invalid or duplicate component ID")
        identifiers.add(name)
        geometry = runtime._component_geometry(item["geometry"])
        _require(geometry["bi_layer_count"] == 1, "v2 unit validation requires its single-BI ledger")
        for key in ("construction", "reference"):
            item[key] = _binding(item[key], path.parent)
        construction, reference = _document(item["construction"]), _document(item["reference"])
        _require(_same(_binding(construction["reference"], Path(item["construction"]["path"]).parent),
                       item["reference"]), "construction and selected reference disagree")
        _require(set(item["components"]) == set(ROLES), "BI, BEOL and Si/TSV components are required")
        for role in ROLES:
            artifact = _binding(item["components"][role], path.parent)
            _require(type(artifact.get("micro_rank")) is int and 1 <= artifact["micro_rank"] <= 48,
                     "unsupported component micro rank")
            built = construction["components"][role]
            expected = _binding(built["artifact"], Path(item["construction"]["path"]).parent)
            _require(_same(expected, artifact) and expected["bytes"] == artifact["bytes"]
                     and built["micro_rank"] == artifact["micro_rank"],
                     "catalog selected a different reduced operand")
            target = _binding(reference["artifacts"][role], Path(item["reference"]["path"]).parent)
            regional = [built.get("reference", {}), *built.get("pooled_reference_identities", {}).values()]
            _require(any(row.get("sha256") == target["sha256"] for row in regional),
                     "component physical coordinates differ from the target reference")
            _projected_work(built)
            with runtime._component_open(artifact["path"], ARRAY_LIMIT, artifact["bytes"]):
                pass
            item["components"][role] = artifact
        validation = item["validation"]
        _require(validation.get("cycles") == 3 and validation.get("bow_observations") == 1081
                 and validation.get("solved_states") in (61, 73)
                 and validation.get("MAPDL_direct_validation") is False,
                 "explicit three-cycle solved-state and bow-observation scope required")
        for key in ("report", "history"):
            validation[key] = _binding(validation[key], path.parent)
        _verify_validation(item, evidence_reader)
        flags = item["capabilities"]
        _require(flags == dict(single_bi_history=True, experimental_prepared_d2w=True,
                               prepared_assembly_accuracy=False, W2W_thinning=False),
                 "component capabilities must retain their measured validation scope")
        entries.append(item)
    evidence_reader.verify_unchanged()
    return dict(schema=catalog["schema"], entries=entries,
                catalog_path=str(path), catalog_sha256=catalog_sha)


def resolve_entry(identifier, *, catalog_path=None):
    name = str(identifier)
    if name.startswith("component:"):
        name = name[len("component:"):]
    matches = [entry for entry in load_catalog(catalog_path)["entries"] if entry["id"] == name]
    if not matches:
        family = _parameterized_family(name)
        if family is not None:
            return family.resolve_entry(name)
    _require(len(matches) == 1, "unknown component ID; inspect component_operators.py list")
    return matches[0]


def _parameterized_family(package):
    name = str(package).removeprefix("component:")
    if re.fullmatch(r"BI2_SI[0-9]+(?:[.p][0-9]+)?", name) is None:
        return None
    path = runtime._component_path(os.environ.get(
        "YAP_THICKNESS_FAMILY_DIR", str(DEFAULT_THICKNESS_FAMILY)))
    if not path.is_dir():
        return None
    from thickness_operators import open_family
    return open_family(path)


def is_component_package(value):
    return isinstance(value, str) and value.startswith("component:")


def prepared_operator_request(package, geometry, *, frozen_preview=False, layout=None):
    entry = resolve_entry(package)
    _require(frozen_preview is False, "component execution uses its explicitly bound corrected material")
    _require(set(geometry) == {"width_mm", "length_mm", "total_thickness_um",
             "front_bi_thickness_um", "back_bi_thickness_um", "beol_thickness_um"},
             "component assembly requires prepared-die geometry")
    for value in geometry.values():
        _require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                 "prepared geometry values must be positive finite numbers")
    g = entry["geometry"]
    _require(geometry["width_mm"] == geometry["length_mm"]
             and geometry["front_bi_thickness_um"] == geometry["back_bi_thickness_um"] == g["bi_thickness_um"]
             and geometry["beol_thickness_um"] == g["beol_thickness_um"]
             and math.isclose(geometry["total_thickness_um"],
                              g["total_thickness_um"] + g["bi_thickness_um"], rel_tol=0, abs_tol=1e-9),
             "prepared total must equal the validated single-BI total plus one backside BI")
    if layout is not None:
        pitch = g["bi_pitch_um"]
        _require(tuple(layout) == (pitch, pitch, pitch*g["diameter_pitch_ratio"]),
                 "component bonding pitch/pad diameter and layout disagree")
    return dict(operator_package=package, component_id=entry["id"],
        total_thickness_um=float(geometry["total_thickness_um"]),
        validated_single_BI_total_um=g["total_thickness_um"],
        retained_si_thickness_um=g["si_tsv_thickness_um"], bonding_pitch_um=g["bi_pitch_um"],
        pad_diameter_um=g["bi_pitch_um"]*g["diameter_pitch_ratio"],
        admission_status="EXPERIMENTAL_COMPONENT_ASSEMBLY_PHYSICAL_REGISTRATION_REQUIRED",
        unit_history_validation=entry["validation"], operator_inputs_verified=False,
        physical_readiness=False, assembly_accuracy_validated=False)


def _w2w_prepared_geometry(geometry):
    _require(isinstance(geometry, dict) and set(geometry) == {
        "diameter_mm", "initial_total_thickness_um", "target_total_thickness_um",
        "front_bi_thickness_um", "back_bi_thickness_um", "beol_thickness_um"},
        "component W2W requires the complete wafer/thinning geometry")
    _require(all(type(v) in (int, float) and math.isfinite(v) and v > 0
                 for v in geometry.values()), "positive finite wafer geometry required")
    _require(geometry["initial_total_thickness_um"] == 775.
             and geometry["target_total_thickness_um"] < 775.,
             "component W2W requires 775-um birth and a smaller thinning target")
    return dict(width_mm=geometry["diameter_mm"], length_mm=geometry["diameter_mm"],
        total_thickness_um=geometry["target_total_thickness_um"],
        **{key: geometry[key] for key in
           ("front_bi_thickness_um", "back_bi_thickness_um", "beol_thickness_um")})


def w2w_operator_request(package, geometry, *, frozen_preview=False, layout=None):
    """Resolve the same Cu components for a thinning wafer, without building caps."""
    request = prepared_operator_request(package, _w2w_prepared_geometry(geometry),
                                       frozen_preview=frozen_preview, layout=layout)
    return dict(request, initial_total_thickness_um=geometry["initial_total_thickness_um"],
        target_total_thickness_um=geometry["target_total_thickness_um"],
        original_outer_si_um=geometry["back_bi_thickness_um"],
        removable_bulk_si_um=775.-geometry["target_total_thickness_um"],
        cap_material_assumption="pure_linear_Si_no_Cu_in_removed_backside",
        admission_status="EXPERIMENTAL_W2W_COMPONENTS_REQUIRE_CHECKED_SILICON_CAPS",
        W2W_thinning_available=True)


def configure_component_runtime():
    """Select the delivered native API before its first import."""
    for key, value in os.environ.items():
        if key in {"YAP_WORK_DIR", "TMPDIR", "TMP", "TEMP", "NUMBA_CACHE_DIR",
                   "MPLCONFIGDIR", "PYTHONPYCACHEPREFIX", "PIP_CACHE_DIR",
                   "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
                   "XDG_STATE_HOME", "XDG_RUNTIME_DIR"}:
            canonical_write_path(value, label=key)
    selected_native = runtime._component_path(os.environ.get(
        "YAP_NATIVE_RUNTIME_DIR", str(DELIVERY / "runtime/native")))
    _check_loaded_native_root(selected_native)
    os.environ.setdefault("YAP_NATIVE_RUNTIME_DIR", str(selected_native))
    os.environ.setdefault("YAP_RUNTIME_RESOURCES", str(DELIVERY / "runtime/RESOURCES.json"))


def _check_loaded_native_root(selected, *, modules=None):
    """Never reuse a backend that captured another native root at import time."""
    selected = Path(selected).resolve()
    loaded = sys.modules if modules is None else modules
    for name in ("mechanics_runtime", "YAP.mechanics_runtime",
                 "history_mechanics_backend", "YAP.history_mechanics_backend"):
        module = loaded.get(name)
        if module is None:
            continue
        attributes = vars(module)
        for key in ("NATIVE_ROOT", "ANSYS_VALIDATION"):
            captured = attributes.get(key)
            _require(captured is None or Path(captured).resolve() == selected,
                     "native runtime was already loaded from a different root; export "
                     "YAP_NATIVE_RUNTIME_DIR=" + str(selected) + " before Python and restart the process")
    prefixes = ("exp12_", "run_exp12_", "audit_exp12_", "build_exp12_", "collect_exp12_",
                "launch_exp12_", "monitor_exp12_", "prepare_exp12_")
    for name, module in tuple(loaded.items()):
        if name.startswith(prefixes) or name == "stack_bow_conventions":
            filename = vars(module).get("__file__") if module is not None else None
            _require(filename is None or Path(filename).resolve().is_relative_to(selected),
                     "native module was already loaded from a different root: " + name)


def _pinned_cu_card(card_api):
    """Parse only the bounded, no-follow, hash-verified delivery-card snapshot."""
    path = DELIVERY / "CU_MATERIAL_CARD.json"
    document, digest = runtime._component_json(path, CU_MATERIAL_CARD_SHA256)

    def verified_read(requested_path, schema):
        _require(Path(requested_path) == path and document.get("schema_version") == schema
                 and isinstance(document.get("identity"), dict)
                 and document["identity"].get("lifecycle") == "frozen_for_v2_regeneration",
                 "pinned Cu card schema or lifecycle differs")
        return path, document, digest

    # Keep the native parser unchanged without reopening a mutable path or
    # patching its module globals. Only its input reader is isolated.
    original = card_api.load_v2_cu_card
    parser = FunctionType(original.__code__, dict(original.__globals__, _read=verified_read),
                          original.__name__, original.__defaults__, original.__closure__)
    parser.__kwdefaults__ = original.__kwdefaults__
    card = parser(path)
    _require(card.sha256 == CU_MATERIAL_CARD_SHA256, "parsed Cu card SHA256 differs")
    return card


def canonical_write_path(value, *, label="component output"):
    return runtime.validate_user_write_path(value, label=label)


class OperandReader:
    """One selected entry, bounded numeric reads and per-load hash reuse."""

    def __init__(self):
        self.proofs = {}
        self.bytes_hashed = 0

    def arrays(self, binding, names=None):
        import numpy as np
        binding = _binding(binding, DEFAULT_CATALOG.parent)
        with runtime._component_open(binding["path"], ARRAY_LIMIT, binding.get("bytes")) as stream:
            before = runtime._component_identity(stream)
            saved = self.proofs.get(binding["path"])
            if saved != (binding["sha256"], before):
                self.bytes_hashed += before[2]
                _require(self.bytes_hashed <= READ_LIMIT, "selected component read budget exceeded")
                runtime._component_hash_stream(stream, binding["sha256"], time.monotonic()+300.)
                self.proofs[binding["path"]] = (binding["sha256"], before)
            stream.seek(0)
            with zipfile.ZipFile(stream) as archive:
                files = archive.infolist()
                _require(0 < len(files) <= 96 and len({f.filename for f in files}) == len(files)
                         and all('/' not in f.filename and f.filename.endswith('.npy') and not f.flag_bits & 1
                                 for f in files) and sum(f.file_size for f in files) <= ARRAY_LIMIT,
                         "bounded, flat numeric component archive required")
                selected = {f.filename for f in files} if names is None else {name+'.npy' for name in names}
                for member in files:
                    if member.filename not in selected:
                        continue
                    with archive.open(member) as handle:
                        version = np.lib.format.read_magic(handle)
                        _require(version in ((1, 0), (2, 0)), "unsupported component NPY version")
                        header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                                  else np.lib.format.read_array_header_2_0)
                        shape, _, dtype = header(handle, max_header_size=4096)
                        _require(dtype.kind in 'fiu' and not dtype.hasobject and len(shape) <= 3
                                 and math.prod(shape)*dtype.itemsize + handle.tell() == member.file_size,
                                 "component array dimensions differ from its bounded numeric payload")
            stream.seek(0)
            with np.load(stream, allow_pickle=False, max_header_size=4096) as archive:
                selected = archive.files if names is None else names
                _require(set(selected) <= set(archive.files), "required component arrays are missing")
                values = {name: archive[name] for name in selected}
            _require(before == runtime._component_identity(stream), "component changed during array loading")
        for value in values.values():
            _require(value.dtype.kind in 'fiu' and np.isfinite(value).all(), "finite numeric component arrays required")
            value.flags.writeable = False
        return values


def _verify_ordered_gauss(reference, operator, proof):
    """Require original GP ordering/weights before any coordinate normalization."""
    import numpy as np

    count = proof.get("Cu_GP")
    _require(type(count) is int and count > 0, "projection certificate needs its actual Cu GP count")
    for key in GP_KEYS:
        expected, observed = reference[key], operator[key]
        shape = (count, 3) if key == "Cu_positions_mm" else (count,)
        _require(expected.shape == observed.shape == shape
                 and np.array_equal(expected, observed),
                 "ordered Cu GP reference mismatch: " + key)
        if key in ("Cu_native_element_ids", "Cu_native_gauss_indices"):
            _require(expected.dtype.kind in "iu" and observed.dtype.kind in "iu",
                     "integer ordered Cu GP identities required")


def _physical_operands(entry, geometry, reader):
    construction, reference = _document(entry["construction"]), _document(entry["reference"])
    base = Path(entry["reference"]["path"]).parent
    topology = reader.arrays(_binding(reference["topology_artifact"], base), ("coordinate_scale",))
    scale = topology["coordinate_scale"][:134]
    _require(scale.shape == (134,), "native macro coordinate scale required")
    g = entry["geometry"]
    datasets = {}
    for role in ROLES:
        proof = _projected_work(construction["components"][role])
        data = reader.arrays(_binding(reference["artifacts"][role], base),
                             ("coordinates_mm", "lifts134", "native_node_indices") + GP_KEYS)
        values = reader.arrays(entry["components"][role],
            ARRAY_KEYS + ("Cu_native_element_ids", "Cu_native_gauss_indices", "Cu_positions_mm"))
        datasets[role] = dict(data=data, values=values, proof=proof)
    return _physical_operands_from_arrays(g, geometry, scale, datasets)


def _physical_operands_from_arrays(g, geometry, scale, datasets, *, canonicalize_port_roundoff=False):
    import numpy as np
    from W2W import component_assembly_bridge as bridge

    _require(set(datasets) == set(ROLES), "complete physical component datasets required")
    thickness = dict(bi=g["bi_thickness_um"] / 1000, beol=g["beol_thickness_um"] / 1000,
                     si=g["si_tsv_thickness_um"] / 1000)
    offsets = dict(bi=0., beol=thickness["bi"], si=thickness["bi"]+thickness["beol"])
    operands, reflection, common_ports = {}, None, None
    for role in ROLES:
        record = datasets[role]
        data, values, proof = record["data"], dict(record["values"]), record["proof"]
        _verify_ordered_gauss(data, values, proof)
        xyz, lifts = data["coordinates_mm"], data["lifts134"]
        _require(xyz.ndim == 2 and xyz.shape[1] == 3 and lifts.shape == (3*len(xyz), 134),
                 "complete physical macro lifts required")
        faces = [np.flatnonzero(xyz[:, 1] == y) for y in (xyz[:, 1].min(), xyz[:, 1].max())]
        faces = [f[np.lexsort((xyz[f, 2], xyz[f, 0]))] for f in faces]
        _require(len(faces[0]) == len(faces[1]), "component has unequal face-node censuses")
        coordinates = np.stack([xyz[f] for f in faces]).copy()
        coordinates[:, :, 1] -= xyz[:, 1].min()
        _require(abs(np.ptp(xyz[:, 1])-thickness[role]) <= 1e-12, "reference physical thickness differs")
        trace = np.stack([lifts.reshape(-1, 3, 134)[f].reshape(-1, 134) for f in faces]).copy()
        expected = np.stack([bridge._canonical_macro_trace(coordinates[f], (f-.5)*thickness[role])
                             for f in (0, 1)])
        delta = (trace[:, :, :6]-expected).reshape(-1, 3, 6)
        translation = delta.mean(axis=0)
        _require(np.max(abs(delta-translation), initial=0) <= 1e-12,
                 "component macro registration requires more than a rigid translation")
        trace[:, :, :6] -= np.tile(translation, (len(faces[0]), 1))[None]
        if canonicalize_port_roundoff:
            if common_ports is None:
                common_ports = trace[0, :, 6:70].copy()
            exact = np.zeros_like(trace[:, :, 6:134])
            exact[0, :, :64], exact[1, :, 64:] = common_ports, common_ports
            _require(np.max(abs(trace[:, :, 6:134]-exact), initial=0) <= 1e-12,
                     "interpolated face ports differ by more than registration roundoff")
            # Polynomial reconstruction can perturb a shared face by a few ulps.
            # Only the already-certified physical trace is made bitwise common.
            trace[:, :, 6:134] = exact
        bottom = trace[0, :, 6:70]
        local_reflection = np.linalg.lstsq(bottom, bottom*np.tile([1., -1., 1.], len(faces[0]))[:, None],
                                          rcond=1e-12)[0]
        if reflection is None:
            reflection = local_reflection
        _require(np.allclose(reflection, local_reflection, rtol=0, atol=1e-12),
                 "component reflection matrices disagree")
        bubbles = values["c"].shape[0]
        physical_lifts = np.zeros((2, 3*len(faces[0]), 134+bubbles))
        physical_lifts[:, :, :134] = trace
        gp = dict(element_gauss_ids=np.column_stack((values.pop("Cu_native_element_ids"),
                    values.pop("Cu_native_gauss_indices"))), coordinates_mm=values.pop("Cu_positions_mm").copy())
        gp["coordinates_mm"][:, 1] -= offsets[role]
        _require(np.min(gp["coordinates_mm"][:, 1]) >= -1e-12
                 and np.max(gp["coordinates_mm"][:, 1]) <= thickness[role]+1e-12,
                 "ordered Gauss positions differ from the component-local thickness ledger")
        public_role = "retained_si" if role == "si" else role
        operands[public_role] = dict(arrays=values, ordered_gp=gp,
            physical_trace=dict(face_node_ids=np.stack([data["native_node_indices"][f]+1 for f in faces]),
                                face_coordinates_mm=coordinates, face_lifts=physical_lifts))
    certificate = dict(schema="yap.component_prepared_assembly_certificate.v3", scope="experimental",
        geometry=dict(geometry), bi_layer_count=2, cell_area_mm2=(g["cell_side_um"]/1000)**2,
        macro_coordinate_scale=scale.tolist(), reflection_matrix=reflection.tolist(),
        rigid_port_mode_indices=[0, 1, 2], thickness_orientations=[1, -1],
        macro_contract=bridge._MACRO_CONTRACT, components={})
    for source_role, role in (("bi", "bi"), ("beol", "beol"), ("si", "retained_si")):
        certificate["components"][role] = dict(thickness_mm=thickness[source_role],
            virtual_work_certified=True,
            **{key+"_sha256": bridge._sha256_arrays(value) for key, value in operands[role].items()})
    return operands, certificate


def _parameterized_operands(entry, geometry):
    family = _parameterized_family(entry["id"])
    _require(family is not None and family.manifest_sha256 == entry["family_sha256"],
             "selected thickness family changed")
    evaluated = family.evaluate(si_um=entry["geometry"]["si_tsv_thickness_um"])
    _require(evaluated.cache_key == entry["cache_key"], "thickness evaluation binding changed")
    datasets = {}
    for role in ROLES:
        data = dict(family.geometry(role=role, si_um=evaluated.resolution.si_um))
        raw = dict(evaluated.component_arrays() if role == "si" else family.fixed_component(role))
        values = {key: raw[key] for key in ARRAY_KEYS}
        for key in GP_KEYS:
            if key not in data:
                _require(key == "weights", "physical thickness-family GP evidence is incomplete")
                data[key] = raw[key]
            values[key] = raw[key] if key in raw else data[key]
        datasets[role] = dict(data=data, values=values,
                              proof=dict(Cu_GP=len(values["weights"])))
    scale = family.topology()["coordinate_scale"][:134]
    operands, certificate = _physical_operands_from_arrays(entry["geometry"], geometry,
        scale, datasets, canonicalize_port_roundoff=True)
    inputs = dict(family_directory=str(family.root), family_sha256=family.manifest_sha256,
        cache_key=evaluated.cache_key, coefficients=family.manifest["coefficients"],
        fixed_components=family.manifest["fixed_components"],
        registration_roundoff_limit_mm=1e-12,
        FE_generation=False, target_POD=False, aged_state_transfer=False)
    return operands, certificate, inputs


def _component_library_and_trace(package, geometry, *, frozen_preview=False):
    """Construct the native assembly library with complete Cu and physical faces."""
    prepared_operator_request(package, geometry, frozen_preview=frozen_preview)
    configure_component_runtime()
    runtime.configure_runtime_storage()
    from W2W import assembly_mechanics as mechanics, component_assembly_bridge as bridge
    card = _pinned_cu_card(mechanics.native("exp12_material_card_v2_kim2026"))
    entry = resolve_entry(package)
    if entry.get("kind") == "endpoint_thickness_family":
        operands, certificate, operator_inputs = _parameterized_operands(entry, geometry)
    else:
        reader = OperandReader()
        operands, certificate = _physical_operands(entry, geometry, reader)
        operator_inputs = {role: entry["components"][role] for role in ROLES}
    from W2W.component_compiled_material import create_compiled_material, SOURCE_ROOT, SOURCE_PINS
    parameters = card.python_cu_card_kwargs()
    material = create_compiled_material(cu_source_path=SOURCE_ROOT/"exp12_cu_material_point.py",
        cu_source_sha256=SOURCE_PINS["exp12_cu_material_point.py"],
        acceleration_source_path=SOURCE_ROOT/"exp12_all_gauss_exact_acceleration.py",
        acceleration_source_sha256=SOURCE_PINS["exp12_all_gauss_exact_acceleration.py"],
        native_exact_module=mechanics.native("exp12_all_gauss_exact_acceleration"),
        reduced_module=mechanics.native("exp12_all_gauss_reduced_fe2"),
        cache_directory=Path(os.environ["NUMBA_CACHE_DIR"]), card_parameters=parameters)
    library, proof = bridge.build_component_assembly_library(operands, geometry=geometry,
        certificate=certificate, expected_certificate_sha256=bridge._sha256_document(certificate),
        cu_card=material.cu_card, material_kernels=material.material_kernels, material_binding=material.binding)
    provenance = dict(schema="yap.component_assembly_runtime.v1", component_id=entry["id"],
        frozen_preview=False, component_validation=entry["validation"], physical_registration=proof,
        geometry=dict(geometry), unit_geometry=entry["geometry"],
        operator_inputs=operator_inputs,
        material_card=dict(path=str(DELIVERY / "CU_MATERIAL_CARD.json"), sha256=card.sha256),
        physics_qualification="EXPERIMENTAL_ASSEMBLY_UNIT_BOW_VALIDATED_ASSEMBLY_ACCURACY_PENDING",
        W2W_thinning_available=False, corrected_material=runtime.corrected_material_provenance(material.binding))
    return library, provenance, operands["bi"]["physical_trace"]


def load_prepared_component_library(package, geometry, *, frozen_preview=False):
    """Construct the prepared-die library; the physical trace stays internal."""
    library, provenance, _ = _component_library_and_trace(package, geometry, frozen_preview=frozen_preview)
    return library, provenance


def load_w2w_component_library(package, geometry, *, frozen_preview=False):
    """Append checked Cu-free Si layers without modifying the Cu operators."""
    request = w2w_operator_request(package, geometry, frozen_preview=frozen_preview)
    library, provenance, trace = _component_library_and_trace(package,
        _w2w_prepared_geometry(geometry), frozen_preview=frozen_preview)
    from W2W.component_silicon_cap import augment_w2w_library
    library, cap_provenance = augment_w2w_library(library, geometry, trace)
    return library, dict(provenance, geometry=dict(geometry), w2w_component_request=request,
        w2w_si_caps=cap_provenance, W2W_thinning_available=True,
        physics_qualification="EXPERIMENTAL_F2B_COMPONENTS_WITH_CHECKED_PURE_SI_CAPS_ASSEMBLY_ACCURACY_PENDING")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "inspect", "check"))
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--entry")
    options = parser.parse_args(argv)
    catalog = load_catalog(options.catalog)
    if options.command != "list":
        parser.error("--entry is required") if not options.entry else None
        entries = [resolve_entry(options.entry, catalog_path=options.catalog)]
    else:
        entries = catalog["entries"]
    if options.command == "check":
        if entries[0].get("kind") == "endpoint_thickness_family":
            family = _parameterized_family(entries[0]["id"])
            family.evaluate(si_um=entries[0]["geometry"]["si_tsv_thickness_um"])
            for role in ("bi", "beol"):
                family.fixed_component(role)
        else:
            reader = OperandReader()
            for role in ROLES:
                reader.arrays(entries[0]["components"][role], ("weights",))
    print(json.dumps(dict(catalog_path=catalog["catalog_path"], catalog_sha256=catalog["catalog_sha256"],
        verification="SELECTED_OPERATOR_HASHES_VERIFIED" if options.command == "check" else "METADATA_VERIFIED",
        entries=[dict(id=e["id"], geometry=e["geometry"],
            prepared_total_thickness_um=e["geometry"]["total_thickness_um"]+e["geometry"]["bi_thickness_um"],
            validation=e["validation"], capabilities=e["capabilities"]) for e in entries]), indent=2))


if __name__ == "__main__":
    main()
