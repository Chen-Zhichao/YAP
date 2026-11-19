"""Common library, material contracts and resource admission for all geometries."""
from collections.abc import Mapping
import ast
from dataclasses import dataclass
from importlib import import_module
import hashlib
import json
import marshal
import os
from pathlib import Path
import resource
from types import CodeType, FunctionType, MappingProxyType

import numpy as np

ACCEPTED_CARD = dict(room_c=25.,anneal_c=350.,e_room_mpa=131000.,e_anneal_mpa=131000.,
    nu_room=.34,nu_anneal=.34,alpha_room_per_c=17.6e-6,alpha_anneal_per_c=17.6e-6,
    yield_room_mpa=321.,yield_anneal_mpa=321.,tangent_mpa=2000.,creep_law="modified_time_hardening",
    creep_prefactor_mpa_minus_n_per_s=1.43e10,creep_activation_term_k=23695.,creep_exponent=2.5,creep_time_exponent=-.9)
_ACCEPTED_DEFAULTS = dict(creep_rate_room=1e-30,creep_rate_anneal=1e-7,creep_reference_stress_mpa=50.)
HISTORICAL_CARD = {**ACCEPTED_CARD,**_ACCEPTED_DEFAULTS}
PUBLIC_CARD = {**ACCEPTED_CARD,"creep_rate_room":1e-300,"creep_rate_anneal":1e-300,"creep_reference_stress_mpa":1.}
_MATERIALS = {}
SOURCE_PINS = MappingProxyType(dict(
    Cu_source="35e7016dab8b1fb2cb0a928e7642665d3fc4f00b4f255eb797fb6d6e44b56722",
    acceleration_source="483c9d948931d48beb73561842cb8fe780db285d3b61f158e580229fbdcd465f"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _freeze(value):
    if isinstance(value,Mapping):
        return MappingProxyType({key:_freeze(v) for key,v in value.items()})
    if isinstance(value,(list,tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def application_module(name):
    namespace = __package__.rpartition(".")[0].rpartition(".")[0]
    return import_module((namespace+"." if namespace else "")+name)


@dataclass(frozen=True)
class CouponOperatorLibrary:
    operators: Mapping
    thickness_mm: Mapping
    cell_area_mm2: float
    operator_reference_y_mm: float
    membrane_bending_coordinate_scale: object
    port_coordinate_scale: object
    reflection_matrix: object
    coupling: str = "whole_prepared_coupon"
    global_reference_y_mm: float = 0.
    port_rank_per_face: int = 0
    rigid_port_mode_indices: tuple = ()
    common_port_gauge_vectors: object = None


def _components():
    return application_module("component_operators")


def configure_coupon_runtime():
    _components().configure_component_runtime()


def validate_card(parameters):
    if parameters != HISTORICAL_CARD and parameters != PUBLIC_CARD:
        raise ValueError("unsupported full nineteen-field Cu material contract")
    return dict(parameters)


def _card_method_fingerprints(raw, filename, names):
    tree = ast.parse(raw,filename=filename)
    tree.body = [node for node in tree.body if not
        (isinstance(node,ast.ImportFrom) and node.module == "__future__")]
    module = compile(tree,filename,"exec",dont_inherit=True)
    classes = [value for value in module.co_consts
        if isinstance(value,CodeType) and value.co_name == "CuCard"]
    require(len(classes) == 1,"one pinned CuCard definition required")
    # Function objects reproduce legacy marshal ownership of names/docstrings;
    # no compiled module, class body or method is executed. Drop AST references
    # to constants before hashing, as the deployment compiler does on return.
    methods = {value.co_name:FunctionType(value,{}) for value in classes[0].co_consts
        if isinstance(value,CodeType)}
    del tree
    require(set(names).issubset(methods),"complete pinned CuCard methods required")
    return {name:hashlib.sha256(marshal.dumps(methods[name].__code__)).hexdigest() for name in names}


def validate_material_provenance(material, current, historical):
    """Compare legacy method hashes without changing any executing code."""
    data = application_module("mechanics_runtime")
    data._validate_corrected_material_binding(material.binding,
        material_kernels=material.material_kernels,cu_card=material.cu_card)
    expected_sources = {"exp12_cu_material_point.py":SOURCE_PINS["Cu_source"],
        "exp12_all_gauss_exact_acceleration.py":SOURCE_PINS["acceleration_source"]}
    paths = []
    for provenance in (current,historical):
        entries = provenance["sources"]
        require(len(entries) == 2 and
            {Path(item["path"]).name:item["sha256"] for item in entries} == expected_sources,
            "material must retain both pinned Cu35/483 sources")
        source = provenance["source_path"]
        require(isinstance(source,str) and Path(source).is_absolute()
            and Path(source).name == "exp12_cu_material_point.py"
            and provenance["source_sha256"] == SOURCE_PINS["Cu_source"]
            and any(item["path"] == source and item["sha256"] == SOURCE_PINS["Cu_source"]
                for item in entries), "material Cu filename/source binding differs")
        validate_card(provenance["cu_card"])
        require(provenance["clock_semantics"] == data.CORRECTED_MATERIAL_CLOCK,
            "material physical-endpoint clock contract differs")
        paths.append(source)
    require(current["cu_card"] == historical["cu_card"] and paths[0] == material.binding.source_path,
        "material source/card contract differs")
    names = data._MATERIAL_CARD_METHODS
    require(set(current["card_methods"]) == set(historical["card_methods"]) == set(names),
        "complete material method fingerprint inventory required")
    codes = {name:code for name,(_,code) in zip(names,material.binding.card_methods)}
    fingerprints = {name:hashlib.sha256(marshal.dumps(code)).hexdigest() for name,code in codes.items()}
    require(fingerprints == current["card_methods"], "current material method code differs")
    if fingerprints == historical["card_methods"]:
        return
    require(paths[0] != paths[1], "material method code differs at unchanged source path")
    with data._component_open(Path(paths[0]),256*1024) as stream:
        raw = stream.read(256*1024+1)
    require(len(raw) <= 256*1024 and hashlib.sha256(raw).hexdigest() == SOURCE_PINS["Cu_source"],
        "relocated Cu source bytes differ from the pin")
    require(_card_method_fingerprints(raw,paths[0],names) == fingerprints,
        "current material methods differ from pinned Cu code")
    require(_card_method_fingerprints(raw,paths[1],names) == historical["card_methods"],
        "material method code differs after source relocation; matching Python-minor ABI required")


def _compiled_material(card_parameters=None):
    parameters = validate_card(HISTORICAL_CARD if card_parameters is None else card_parameters)
    configure_coupon_runtime()
    _components().runtime.configure_runtime_storage()
    mechanics = application_module("W2W.assembly_mechanics")
    compiled = application_module("W2W.component_compiled_material")
    exact,reduced = mechanics.native("exp12_all_gauss_exact_acceleration"),mechanics.native("exp12_all_gauss_reduced_fe2")
    key = (json.dumps(parameters,sort_keys=True),tuple(sorted(compiled.SOURCE_PINS.items())),
        id(exact),id(reduced),str(compiled.SOURCE_ROOT),os.environ["NUMBA_CACHE_DIR"])
    if key not in _MATERIALS:
        _MATERIALS[key] = compiled.create_compiled_material(
            cu_source_path=compiled.SOURCE_ROOT/"exp12_cu_material_point.py",
            cu_source_sha256=compiled.SOURCE_PINS["exp12_cu_material_point.py"],
            acceleration_source_path=compiled.SOURCE_ROOT/"exp12_all_gauss_exact_acceleration.py",
            acceleration_source_sha256=compiled.SOURCE_PINS["exp12_all_gauss_exact_acceleration.py"],
            native_exact_module=exact,reduced_module=reduced,cache_directory=Path(os.environ["NUMBA_CACHE_DIR"]),
            card_parameters=parameters,scope="experimental")
    material = _MATERIALS[key]
    material.binding.implementation.validate()
    if {name:getattr(material.cu_card,name) for name in parameters} != parameters:
        raise ValueError("cached compiled material contract changed")
    return material,dict(corrected_material=_components().runtime.corrected_material_provenance(material.binding))


def _shared_allocation_admission(stage, forecast):
    amount = forecast.get("additional_allocation_forecast_bytes")
    if type(amount) is not int or amount <= 0:
        raise ValueError("positive memory allocation forecast required")
    memory = {}
    for name,wanted in (("/proc/self/status","VmSize"),("/proc/meminfo","MemAvailable")):
        with open(name) as stream:
            for line in stream:
                if line.startswith(wanted+":"):
                    memory[wanted] = int(line.split()[1])*1024
                    break
    margin = 512*1024**2
    if memory["MemAvailable"] <= amount+margin:
        raise MemoryError(stage+": insufficient live available memory")
    soft,hard = resource.getrlimit(resource.RLIMIT_AS)
    finite = [v for v in (soft,hard) if v != resource.RLIM_INFINITY]
    if finite and memory["VmSize"]+amount+margin >= min(finite):
        raise MemoryError(stage+": process address-space limit exceeded")


def _global_scale(stiffness, local_reference_y_mm):
    rank = len(stiffness)
    transform = np.eye(rank)
    transform[:3,3:6] = local_reference_y_mm*np.eye(3)
    diagonal = np.diag(transform.T@stiffness@transform)[:6]
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
        raise ValueError("positive global reference energy required")
    return 1./np.sqrt(diagonal)
