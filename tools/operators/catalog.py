"""Named operator aliases backed by a single numerical data catalog."""
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import warnings

from . import load_library, literal, require
from .runtime import application_module


@lru_cache(maxsize=8)
def _read(path):
    path = literal(path)
    with path.open("rb") as stream:
        raw = stream.read(2*1024**2+1)
    require(len(raw) <= 2*1024**2,"bounded operator alias catalog required")
    document = json.loads(raw)
    require(document.get("schema") == "yap.normalized_operator_catalog.v1"
        and type(document.get("operators")) is dict and 1 <= len(document["operators"]) <= 4096,
        "valid numerical operator alias catalog required")
    if "python_cache_tag" in document:
        producer = document["python_cache_tag"]
        require(producer == sys.implementation.cache_tag,
            f"normalized operator catalog Python ABI mismatch: producer {producer!r}, "
            f"running {sys.implementation.cache_tag!r}; use the matching Python interpreter")
    return document


def catalog():
    obsolete = [name for name in ("YAP_COUPON_OPERATOR_ROOT","YAP_SHARED_COUPON_OPERATOR_ROOT")
        if name in os.environ]
    if obsolete:
        warnings.warn(", ".join(obsolete)+" no longer select coupon data and are ignored; use "
            "YAP_NORMALIZED_OPERATOR_ROOT and YAP_COUPON_OPERATOR_CATALOG instead",
            UserWarning,stacklevel=2)
    default = Path(__file__).resolve().parents[2]/"prepared_coupon_catalog.json"
    path = os.environ.get("YAP_COUPON_OPERATOR_CATALOG",str(default))
    return _read(path)


def prepared_operator_request(selector,geometry,*,frozen_preview=False,layout=None):
    require(isinstance(selector,str) and selector.startswith("coupon:"),"explicit coupon selector required")
    key = selector[len("coupon:"):]
    records = catalog()["operators"]
    require(key in records,"unknown normalized operator alias: "+key)
    record = records[key]
    expected = record["geometry"]
    require(set(geometry) == set(expected) and all(type(v) in (int,float) and math.isfinite(v) and v > 0
        for v in geometry.values()),"explicit finite six-field prepared-die geometry required")
    require(all(geometry[k] == expected[k] for k in expected if k not in ("width_mm","length_mm")),
        "operator thickness/layer geometry differs")
    if not record.get("area_scalable",False):
        require(geometry["width_mm"] == expected["width_mm"]
            and geometry["length_mm"] == expected["length_mm"],"this operator requires its exact footprint")
    require(frozen_preview is False,"normalized operators are not frozen preview packages")
    pitch = record["pitch_um"]
    require(layout is None or tuple(layout) == (pitch,pitch,pitch/2.),"layout and mechanics pitch differ")
    return dict(operator_package=selector,coupon_key=key,total_thickness_um=geometry["total_thickness_um"],
        retained_si_thickness_um=geometry["total_thickness_um"]-sum(geometry[k] for k in
            ("front_bi_thickness_um","back_bi_thickness_um","beol_thickness_um")),
        bonding_pitch_um=pitch,pad_diameter_um=pitch/2.,diameter_pitch_ratio=.5,
        required_temporal_profile="component_graded25",operator_inputs_verified=False,
        physical_readiness=False,assembly_accuracy_validated=False,W2W_thinning_available=False,
        coupling="whole_prepared_coupon",admission_status="EXPERIMENTAL_NORMALIZED_PREPARED_DIE")


def load_prepared_coupon_library(selector,geometry,*,frozen_preview=False,shared_check=None,shared_admit=None):
    request = prepared_operator_request(selector,geometry,frozen_preview=frozen_preview)
    require(shared_check is None or callable(shared_check),"resource check must be callable")
    require(shared_admit is None or callable(shared_admit),"allocation admission must be callable")
    if shared_check is not None:shared_check()
    record = catalog()["operators"][request["coupon_key"]]
    root = Path(os.environ.get("YAP_NORMALIZED_OPERATOR_ROOT",str(
        application_module("mechanics_runtime").COMPONENT_PROJECT_ROOT/"distribution/operators/normalized")))
    path = Path(record["directory"])
    require(not path.is_absolute() and not {"..","#snapshot",".snapshot"}.intersection(path.parts),
        "literal catalog-relative numerical packet directory required")
    data = application_module("mechanics_runtime")
    bound = {}
    for name,key in (("EXPORT_COMPLETE.json","export_complete_sha256"),
                     ("MIGRATION_EQUIVALENCE.json","equivalence_sha256")):
        with data._component_open(root/path/name,2*1024**2) as stream:
            raw = stream.read(2*1024**2+1)
        require(len(raw) <= 2*1024**2 and hashlib.sha256(raw).hexdigest() == record[key],
            "operator migration completion/equivalence hash differs")
        bound[name] = json.loads(raw)
    complete,equivalence = bound["EXPORT_COMPLETE.json"],bound["MIGRATION_EQUIVALENCE.json"]
    require(complete["manifest_sha256"] == equivalence["manifest_sha256"] == record["manifest_sha256"]
        and equivalence.get("passed") is True and equivalence.get("all_operands_identical") is True
        and equivalence.get("all_retained_GP_state_signatures_identical") is True
        and equivalence.get("baseline_sha256") == complete["baseline_sha256"],
        "complete paired migration equivalence required")
    if shared_admit is not None:
        require(shared_admit("NORMALIZED_ALIAS_LOAD",dict(additional_allocation_forecast_bytes=
            record["load_forecast_bytes"])) is not False,"allocation admission denied")
    library,proof = load_library(root/path,record["manifest_sha256"],geometry,
        pitch_um=request["bonding_pitch_um"],area_scalable=record.get("area_scalable",False),
        off_footprint_execution=record.get("off_footprint_execution"))
    if shared_check is not None:shared_check()
    exact_geometry = dict(geometry) == record["geometry"]
    return library,dict(request,**proof,operator_inputs_verified=True,migration_equivalence_qualified=exact_geometry,
        migration_equivalence_sha256=record["equivalence_sha256"])
