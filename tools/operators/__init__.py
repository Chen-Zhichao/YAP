"""Common numerical runtime for all 95 migrated whole-body coupon selectors.

Pinned packets retain their natural ranks, material cards and execution
policies without pitch-specific Python loaders or adopted Cu history.
Historical qualifications and immutable migration receipts remain separate
from assembly-accuracy admission. The original v1 packet contract is retained.
"""
import hashlib
from importlib import import_module
import json
import io
import math
from pathlib import Path
from types import MappingProxyType
import zipfile

import numpy as np


SCHEMA = "yap.normalized_whole_operator.v1"
SCHEMA_V2 = "yap.normalized_whole_operator.v2"
PROFILE = "cu35_483_constant_rank9_reference_v1"
PREFIX = "coupon:normalized:"
ARRAYS = frozenset(("K", "f", "E", "weights", "macro_scale", "bubble_scale",
    "global_scale", "cu_lambda", "cu_shear", "cu_thermal"))


def application_module(name):
    namespace = __package__.rpartition(".")[0].rpartition(".")[0]
    return import_module((namespace+"." if namespace else "")+name)


def prepared_request(selector, geometry, *, frozen_preview=False, layout=None):
    """Describe an explicitly pinned packet without reading operator storage."""
    require(isinstance(selector,str) and selector.startswith(PREFIX), "normalized selector required")
    fields = selector[len(PREFIX):].split(":",2)
    require(len(fields) == 3, "normalized selector requires manifest SHA256, pitch and directory")
    pin,pitch_text,directory = fields
    require(len(pin) == 64 and all(c in "0123456789abcdef" for c in pin), "invalid manifest SHA256")
    pitch = float(pitch_text)
    path = Path(directory)
    require(path.is_absolute() and path != Path("/")
        and not {"..","#snapshot",".snapshot"}.intersection(path.parts), "literal absolute packet directory required")
    require(frozen_preview is False, "normalized packets are not frozen reference previews")
    require(set(geometry) == {"width_mm","length_mm","total_thickness_um",
        "front_bi_thickness_um","back_bi_thickness_um","beol_thickness_um"}
        and all(type(x) in (int,float) and np.isfinite(x) and x > 0 for x in geometry.values()),
        "normalized operators require explicit prepared-die geometry, not W2W caps")
    require(np.isfinite(pitch) and pitch > 0, "finite positive pitch required")
    retained = geometry["total_thickness_um"]-sum(geometry[k] for k in
        ("front_bi_thickness_um","back_bi_thickness_um","beol_thickness_um"))
    require(retained > 0, "positive Si thickness required")
    require(layout is None or tuple(layout) == (pitch,pitch,pitch/2.), "normalized operator/layout pitch differs")
    return dict(operator_package=selector,normalized_directory=str(path),manifest_sha256=pin,
        total_thickness_um=geometry["total_thickness_um"],retained_si_thickness_um=retained,
        bonding_pitch_um=pitch,pad_diameter_um=pitch/2.,diameter_pitch_ratio=.5,
        admission_status="EXPERIMENTAL_NORMALIZED_D2W_INPUTS_NOT_VERIFIED",
        required_temporal_profile="component_graded25",operator_inputs_verified=False,
        physical_readiness=False,assembly_accuracy_validated=False,W2W_thinning_available=False)


def load_selected_library(selector, geometry, *, frozen_preview=False):
    request = prepared_request(selector,geometry,frozen_preview=frozen_preview)
    library,proof = load_library(request["normalized_directory"],request["manifest_sha256"],geometry,
        pitch_um=request["bonding_pitch_um"])
    return library,{**request,**proof,"operator_inputs_verified":True}


def configure_runtime():
    import_module(".runtime",__package__).configure_coupon_runtime()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def literal(path):
    path = Path(path).absolute()
    require(not {"..", "#snapshot", ".snapshot"}.intersection(path.parts)
            and not any(p.is_symlink() for p in (path, *path.parents)),
            "literal nonsnapshot, nonsymlink operator path required")
    return path


def operand_shapes(document):
    n = document["integration_points"]
    require(type(n) is int and n > 0, "positive integer integration census required")
    rank = document.get("coordinate_size",9)
    require(type(rank) is int and rank in (9,11,12,27,55,57), "supported authenticated reduced dimension required")
    return dict(K=(rank,rank), f=(rank,), E=(n,6,rank), weights=(n,), macro_scale=(6,),
        bubble_scale=(rank-6,), global_scale=(6,), cu_lambda=(rank,rank), cu_shear=(rank,rank), cu_thermal=(rank,6))


def contract(document):
    valid = (document.get("schema") == SCHEMA and document.get("profile") == PROFILE
        and document.get("coordinate_size",9) == 9) or (
        document.get("schema") == SCHEMA_V2 and document.get("profile") == "constant_elastic_dense")
    require(valid,"unsupported normalized operator contract/profile")
    if document.get("schema") == SCHEMA_V2:
        execution = document["execution"]
        require(execution.get("kind") in ("native","owned483","streamed483","coordinate483","reference"),
            "unsupported numerical execution policy")
        if execution["kind"] in ("streamed483","coordinate483","reference"):
            require(type(execution.get("block_size")) is int and 1 <= execution["block_size"] <= 131072,
                "bounded coordinate contraction required")
        if execution["kind"] == "reference":
            require(document["coordinate_size"] in (9,11) and execution["block_size"] == 32768
                and type(execution.get("zero_creep_shortcut")) is bool,"supported reference tangent policy required")


def validate(arrays, document):
    contract(document)
    geometry = document["geometry"]
    require(set(geometry) == {"width_mm", "length_mm", "total_thickness_um",
        "front_bi_thickness_um", "back_bi_thickness_um", "beol_thickness_um"}
        and all(type(x) in (int, float) and np.isfinite(x) and x > 0 for x in geometry.values()),
        "explicit finite prepared-body geometry required")
    require(geometry["total_thickness_um"] > sum(geometry[k] for k in
        ("front_bi_thickness_um", "back_bi_thickness_um", "beol_thickness_um")),
        "positive retained Si thickness required")
    require(set(arrays) == ARRAYS, "exact numeric operands required")
    shapes = operand_shapes(document)
    for key, shape in shapes.items():
        a = arrays[key]
        require(type(a) is np.ndarray and a.dtype == np.dtype("<f8")
                and a.shape == shape and (a.flags.c_contiguous if document.get("operand_layout",{}).get(key,"C") == "C"
                    else a.flags.f_contiguous) and np.isfinite(a).all(),
                "invalid normalized operand: " + key)
    require(np.all(arrays["weights"] > 0) and all(np.all(arrays[k] > 0)
        for k in ("macro_scale", "bubble_scale", "global_scale")), "positive weights/scales required")
    require(np.all(np.diag(arrays["K"]) > 0), "positive reference diagonal required")
    frame = document["frame"]
    require(set(frame) == {"area_mm2", "operator_reference_y_mm", "global_reference_y_mm"}
        and all(type(v) in (float,int) and np.isfinite(v) for v in frame.values())
        and frame["area_mm2"] == geometry["width_mm"] * geometry["length_mm"]
        and frame["operator_reference_y_mm"] == geometry["total_thickness_um"] / 2000.
        and frame["global_reference_y_mm"] == 0.,
        "explicit already-normalized area and midpoint frame required")
    require(type(document["pitch_um"]) in (int,float) and np.isfinite(document["pitch_um"])
        and document["pitch_um"] > 0 and document["diameter_pitch_ratio"] == .5,
        "finite pitch and supported pad ratio required")
    require(document["temperature_range_c"] == [25.,350.]
        and document["reference_temperature_c"] == 25.
        and document["Cu_state_adoption"] is False,
        "constant reference/linear thermal profile and virgin state required")
    for a in arrays.values():
        a.setflags(write=False)


def export_library(library, provenance, destination, *, geometry, pitch_um):
    """Export actual normalized arrays; never export or adopt committed states."""
    registry = import_module(".runtime",__package__)
    validate_write_path = application_module("mechanics_runtime").validate_write_path
    destination = literal(destination)
    validate_write_path(destination, label="normalized operator export")
    require(not destination.exists(), "fresh migration destination required")
    require(library.coupling == "whole_prepared_coupon" and set(library.operators) == {"whole_prepared"}
        and library.port_rank_per_face == 0 and library.port_coordinate_scale.size == 0
        and library.reflection_matrix.shape == (0,0) and library.common_port_gauge_vectors is None
        and library.rigid_port_mode_indices == (), "whole-prepared non-port library required")
    op = library.operators["whole_prepared"]
    require((op.macro_size,op.bubble_size,op.coordinate_size) == (6,3,9)
        and op.plastic_active is True, "unsupported coordinate/material profile")
    material = registry._components().runtime.corrected_material_provenance(op.corrected_material_binding)
    card = {k:getattr(op.cu_card,k,None) for k in {**registry.ACCEPTED_CARD,**registry._ACCEPTED_DEFAULTS}}
    require(card == {**registry.ACCEPTED_CARD,**registry._ACCEPTED_DEFAULTS}, "unsupported Cu card")
    owner = getattr(op, "_coupon_reference_tangent_owner", None)
    require(owner is not None, "reference-tangent profile required, not arbitrary instance methods")
    owner.validate(op)
    require(owner.calls == 0 and owner.zero_creep_shortcut is True,
        "unused supported reference-tangent profile required")
    require(provenance.get("geometry") == dict(geometry)
        and provenance.get("bonding_pitch_um") == pitch_um,
        "export geometry/pitch must match authenticated source provenance")
    require(np.array_equal(op.reference_stiffness(25.),op.reference_stiffness(350.))
        and np.array_equal(op.reference_thermal_load(25.),np.zeros(9)),
        "constant stiffness and zero reference thermal load required")
    arrays = dict(K=op.reference_stiffness(25.),f=op.reference_thermal_load(26.),
        E=op.strain_maps,weights=op.weights_mm3,macro_scale=op.macro_coordinate_scale,
        bubble_scale=op.bubble_coordinate_scale,global_scale=library.membrane_bending_coordinate_scale,
        cu_lambda=op._cu_lambda_stiffness_integral,cu_shear=op._cu_shear_stiffness_integral,
        cu_thermal=op._cu_constant_stress_virtual_work)
    arrays = {k:np.array(v,dtype="<f8",order="C",copy=True) for k,v in arrays.items()}
    require(np.array_equal(op.reference_thermal_load(350.),325.*arrays["f"]),
        "thermal reference must be exactly linear")
    # This migration supports the declared constant-reference profile only;
    # interior probes reject temperature-dependent references, not just endpoints.
    for temperature in np.linspace(25.,350.,651):
        require(np.array_equal(op.reference_stiffness(float(temperature)),arrays["K"])
            and np.array_equal(op.reference_thermal_load(float(temperature)),(temperature-25.)*arrays["f"]),
            "unsupported temperature-dependent reference profile")
    sampling = dict(point_sampling=provenance.get("point_sampling"),
        retained_GP=op.cu_point_count,native_source_GP=provenance.get("native_source_GP"),
        all_native_GP_retained=provenance.get("all_native_GP_retained"))
    require(type(sampling["point_sampling"]) is bool
        and type(sampling["all_native_GP_retained"]) is bool, "explicit source integration scope required")
    document = dict(schema=SCHEMA,profile=PROFILE,geometry=dict(geometry),pitch_um=pitch_um,
        diameter_pitch_ratio=.5,integration_points=op.cu_point_count,
        frame=dict(area_mm2=library.cell_area_mm2,operator_reference_y_mm=library.operator_reference_y_mm,
                   global_reference_y_mm=library.global_reference_y_mm),
        temperature_range_c=[25.,350.],reference_temperature_c=25.,Cu_state_adoption=False,
        material_card=card,material_source_binding=material,
        integration_scope=sampling,historical_provenance=dict(provenance),migration_accuracy_qualified=False)
    validate(arrays,document)
    # Serialize metadata before publishing any packet, so unsupported provenance
    # cannot leave an apparently complete migration behind.
    json.dumps(document,allow_nan=False,sort_keys=True)
    destination.mkdir(mode=0o700)
    try:
        with (destination/"OPERANDS.npz").open("xb") as stream:
            np.savez(stream,**arrays)
        document["packet"] = dict(path="OPERANDS.npz",sha256=sha(destination/"OPERANDS.npz"),
            bytes=(destination/"OPERANDS.npz").stat().st_size)
        with (destination/"MANIFEST.json").open("x",encoding="ascii") as stream:
            json.dump(document,stream,allow_nan=False,sort_keys=True,indent=2)
            stream.write("\n")
    except BaseException:
        # Retain partial migration for diagnosis; never make it discoverable.
        raise
    return sha(destination/"MANIFEST.json")


def load_library(directory, manifest_sha256, geometry, *, pitch_um, maximum_bytes=8*1024**3,
                 area_scalable=False,off_footprint_execution=None):
    """Read one externally pinned packet; no geometry-specific factory is used."""
    registry = import_module(".runtime",__package__)
    unit_invariant_layer_type = import_module(".kinematics",__package__).unit_invariant_layer_type
    attach_reference_tangent = import_module(".reference",__package__).attach_reference_tangent
    native = application_module("W2W.assembly_mechanics").native
    directory = literal(directory)
    path = directory/"MANIFEST.json"
    runtime = registry._components().runtime
    with runtime._component_open(path,1024**2) as stream:
        raw = stream.read(1024**2+1)
    require(len(raw) <= 1024**2 and hashlib.sha256(raw).hexdigest() == manifest_sha256,
            "manifest extent/hash differs")
    def unique(pairs):
        value = {}
        for k,v in pairs:
            require(k not in value,"duplicate manifest key")
            value[k] = v
        return value
    document = json.loads(raw,object_pairs_hook=unique,
        parse_constant=lambda v: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    contract(document)
    shapes = operand_shapes(document)
    source_geometry = document["geometry"]
    changed_footprint = source_geometry != dict(geometry)
    if changed_footprint and area_scalable:
        require(set(geometry) == set(source_geometry) and all(geometry[k] == source_geometry[k]
            for k in geometry if k not in ("width_mm","length_mm"))
            and all(type(v) in (int,float) and np.isfinite(v) and v > 0 for v in geometry.values()),
            "only explicit homogeneous footprint-area scaling is supported")
    require((not changed_footprint or area_scalable) and document["pitch_um"] == pitch_um,
            "requested geometry/pitch differs from normalized packet")
    packet = document["packet"]
    require(packet["path"] == "OPERANDS.npz" and type(packet["bytes"]) is int and type(maximum_bytes) is int
        and 0 < packet["bytes"] <= maximum_bytes, "bounded single packet required")
    path = literal(directory/packet["path"])
    require(registry._shared_allocation_admission("NORMALIZED_OPERATOR_LOAD",
        dict(additional_allocation_forecast_bytes=4*packet["bytes"]+128*1024**2)) is not False,
        "normalized allocation admission denied")
    with runtime._component_open(path,maximum_bytes,packet["bytes"]) as stream:
        packet_bytes = stream.read(packet["bytes"]+1)
    require(len(packet_bytes) == packet["bytes"] and hashlib.sha256(packet_bytes).hexdigest() == packet["sha256"],
        "numeric packet extent/hash differs")
    # Header validation and allocation consume the SAME immutable bytes that
    # were hashed, even if another writer changes the original NFS file.
    with io.BytesIO(packet_bytes) as stream:
        with zipfile.ZipFile(stream) as archive:
            entries = archive.infolist()
            require(len(entries) == len(ARRAYS) and {i.filename for i in entries} == {k+".npy" for k in ARRAYS}
                and sum(i.file_size for i in entries) <= maximum_bytes
                and all(i.compress_type == zipfile.ZIP_STORED and not i.flag_bits & 1 for i in entries),
                "bounded exact stored numeric archive required")
            for entry in entries:
                with archive.open(entry) as member:
                    prefix = member.read(min(entry.file_size,16384))
                header = io.BytesIO(prefix)
                version = np.lib.format.read_magic(header)
                require(version in ((1,0),(2,0)), "unsupported numeric header")
                reader = (np.lib.format.read_array_header_1_0 if version == (1,0)
                          else np.lib.format.read_array_header_2_0)
                shape,fortran,dtype = reader(header)
                require(dtype == np.dtype("<f8") and fortran == (document.get("operand_layout",{}).get(entry.filename[:-4],"C") == "F")
                    and not dtype.hasobject
                    and shape == shapes[entry.filename[:-4]]
                    and all(type(n) is int and n >= 0 for n in shape)
                    and dtype.itemsize * math.prod(shape) + header.tell() == entry.file_size,
                    "invalid numeric shape/header extent")
        stream.seek(0)
        with np.load(stream,allow_pickle=False) as archive:
            arrays = {k:archive[k] for k in ARRAYS}
    validate(arrays,document)
    registry.configure_coupon_runtime()
    registry._components().runtime.configure_runtime_storage()
    material,_ = registry._compiled_material(document["material_card"])
    require(document["material_card"] == {k:getattr(material.cu_card,k,None)
        for k in document["material_card"]}, "material contract differs")
    current_material = runtime.corrected_material_provenance(material.binding)
    historical_material = document["material_source_binding"]
    registry.validate_material_provenance(material,current_material,historical_material)
    computed = native("exp12_prepared_layer_geometry").cu_elastic_geometry_integrals(arrays["E"],arrays["weights"])
    require(all(np.array_equal(a,arrays[k]) for a,k in zip(computed,("cu_lambda","cu_shear","cu_thermal"))),
        "recomputed Cu geometry integrals differ")
    if changed_footprint:
        area = geometry["width_mm"]*geometry["length_mm"]
        factor = area/document["frame"]["area_mm2"]
        require(np.isfinite(factor) and factor > 0,"finite positive area scaling required")
        for key in ("K","f","weights"):
            arrays[key] = factor*arrays[key]
        scales = 1./np.sqrt(np.diag(arrays["K"]))
        arrays["macro_scale"],arrays["bubble_scale"] = scales[:6],scales[6:]
        arrays["global_scale"] = registry._global_scale(arrays["K"],document["frame"]["operator_reference_y_mm"])
        values = native("exp12_prepared_layer_geometry").cu_elastic_geometry_integrals(arrays["E"],arrays["weights"])
        arrays.update(zip(("cu_lambda","cu_shear","cu_thermal"),values))
        document = dict(document,geometry=dict(geometry),frame=dict(document["frame"],area_mm2=area))
        validate(arrays,document)
    def reference(temperature):
        require(type(temperature) in (int,float,np.float64) and np.isfinite(temperature)
            and 25. <= temperature <= 350., "finite 25..350C reference required")
        return arrays["K"]
    def thermal(temperature):
        reference(temperature)
        return (temperature-25.)*arrays["f"]
    rank = document.get("coordinate_size",9)
    op = unit_invariant_layer_type(native("exp12_all_gauss_reduced_fe2"))(
        macro_size=6,bubble_size=rank-6,reference_stiffness=reference,
        reference_thermal_load=thermal,strain_maps=arrays["E"],
        weights_mm3=arrays["weights"],macro_coordinate_scale=arrays["macro_scale"],
        bubble_coordinate_scale=arrays["bubble_scale"],cu_card=material.cu_card,plastic_active=True,
        material_kernels=material.material_kernels,
        prepared_cu_integrals=tuple(arrays[k] for k in ("cu_lambda","cu_shear","cu_thermal")))
    registry._components().runtime.attach_corrected_material_binding(op,material.binding)
    for value in vars(op).values():
        if isinstance(value,np.ndarray):
            value.setflags(write=False)
    scope = document["integration_scope"]
    require(scope.get("retained_GP") == op.cu_point_count
        and type(scope.get("point_sampling")) is bool
        and type(scope.get("all_native_GP_retained")) is bool
        and scope["point_sampling"] is not scope["all_native_GP_retained"],
        "explicit retained integration scope required")
    if document["schema"] == SCHEMA_V2:
        require(type(scope.get("native_source_GP")) is int and scope["native_source_GP"] >= op.cu_point_count
            and (not scope["all_native_GP_retained"] or scope["native_source_GP"] == op.cu_point_count),
            "native versus retained integration census differs")
    execution = document.get("execution",dict(kind="reference",block_size=32768,zero_creep_shortcut=True))
    if changed_footprint and off_footprint_execution is not None:
        require(off_footprint_execution == "native","supported off-footprint execution policy required")
        execution = dict(kind="native")
    op.normalized_operator_binding = MappingProxyType(dict(schema="yap.normalized_operator_binding.v1",
        coordinate_size=rank,integration_points=op.cu_point_count,manifest_sha256=manifest_sha256))
    shared_binding = document.get("shared_coupon_binding")
    if rank != 9:
        require(type(shared_binding) is dict and shared_binding.get("schema") == "yap.shared_coupon_layer.v1"
            and shared_binding.get("private_coordinates") == rank-6
            and shared_binding.get("reduced_coordinates") == rank
            and shared_binding.get("all_native_GP_retained") is True
            and shared_binding.get("public_admission") is False
            and shared_binding.get("assembly_qualified") is False,"authenticated private-coordinate admission required")
        op.shared_coupon_binding = MappingProxyType(dict(shared_binding))
    if execution["kind"] == "reference":
        owner = attach_reference_tangent(op,material,admit=registry._shared_allocation_admission,
            block_size=execution["block_size"],zero_creep_shortcut=execution["zero_creep_shortcut"])
        owner.proof = MappingProxyType(dict(owner.proof,point_sampling=scope["point_sampling"],
            all_native_GP_retained=scope["all_native_GP_retained"],integration_scope_owner="normalized_packet"))
        owner.validate(op)
    elif execution["kind"] == "owned483":
        import_module(".acceleration",__package__).attach_acceleration(op,
            material_binding=material.binding,admit=registry._shared_allocation_admission)
    elif execution["kind"] in ("streamed483","coordinate483"):
        streamed = import_module(".streamed",__package__)
        if execution["kind"] == "coordinate483":
            streamed.attach_coordinate_acceleration(op,material_binding=material.binding,
                admit=registry._shared_allocation_admission,block_size=execution["block_size"])
        else:
            streamed.attach_streamed_acceleration(op,material_binding=material.binding,
                admit=registry._shared_allocation_admission,block_size=execution["block_size"])
    empty = np.empty(0); empty.setflags(write=False)
    library = registry.CouponOperatorLibrary(operators=MappingProxyType(dict(whole_prepared=op)),
        thickness_mm=MappingProxyType(dict(whole_prepared=geometry["total_thickness_um"]/1000.)),
        cell_area_mm2=document["frame"]["area_mm2"],
        operator_reference_y_mm=document["frame"]["operator_reference_y_mm"],
        global_reference_y_mm=document["frame"]["global_reference_y_mm"],
        membrane_bending_coordinate_scale=arrays["global_scale"],port_coordinate_scale=empty,
        reflection_matrix=empty.reshape(0,0))
    return library,dict(schema=document["schema"],geometry=dict(geometry),pitch_um=pitch_um,
        input_sha256={str(directory/"MANIFEST.json"):manifest_sha256,str(path):packet["sha256"]},
        integration_scope=scope,point_sampling=scope["point_sampling"],
        all_native_GP_retained=scope["all_native_GP_retained"],Cu_state_adoption=False,
        material_implementation_equivalence_qualified=False,
        migration_accuracy_qualified=False,assembly_accuracy_validated=False,physical_readiness=False)
