"""Geometry is catalog data; runtime profiles never depend on pitch names."""
import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tools.operators import catalog, kinematics, runtime


@pytest.fixture
def records(monkeypatch):
    geometry = dict(width_mm=10.,length_mm=10.,total_thickness_um=60.,
        front_bi_thickness_um=1.5,back_bi_thickness_um=1.5,beol_thickness_um=10.)
    result = {"example":dict(geometry=geometry,pitch_um=7.,area_scalable=False)}
    monkeypatch.setattr(catalog,"catalog",lambda:dict(operators=result))
    return result


def test_catalog_request_contains_no_geometry_specific_python(records):
    request = catalog.prepared_operator_request("coupon:example",records["example"]["geometry"],layout=(7.,7.,3.5))
    assert request["total_thickness_um"] == 60.
    assert request["retained_si_thickness_um"] == 47.
    assert request["operator_inputs_verified"] is False


@pytest.mark.parametrize("field,value",[("total_thickness_um",50.),("width_mm",20.),
    ("front_bi_thickness_um",2.),("length_mm",True)])
def test_request_does_not_relabel_data(records,field,value):
    geometry = dict(records["example"]["geometry"],**{field:value})
    with pytest.raises(ValueError):catalog.prepared_operator_request("coupon:example",geometry)


def test_area_policy_is_explicit_metadata(records):
    records["example"]["area_scalable"] = True
    geometry = dict(records["example"]["geometry"],width_mm=20.,length_mm=15.)
    assert catalog.prepared_operator_request("coupon:example",geometry)["bonding_pitch_um"] == 7.


def test_both_literal_material_contracts_survive():
    assert runtime.validate_card(runtime.HISTORICAL_CARD) == runtime.HISTORICAL_CARD
    assert runtime.validate_card(runtime.PUBLIC_CARD) == runtime.PUBLIC_CARD
    assert runtime.HISTORICAL_CARD != runtime.PUBLIC_CARD
    assert len(runtime.HISTORICAL_CARD) == len(runtime.PUBLIC_CARD) == 19
    with pytest.raises(ValueError):runtime.validate_card(dict(runtime.PUBLIC_CARD,tangent_mpa=3000.))


def test_constructor_guard_preserves_original_operational_matrix():
    matrix = np.diag([1e9,1e-9,1.])
    matrix[0,2] = 1e-7
    before = matrix.copy()
    assert kinematics.stiffness_symmetry(matrix)["passed"]
    np.testing.assert_array_equal(matrix,before)


@pytest.mark.skipif(not os.environ.get("YAP_MIGRATION_BASELINE"),reason="explicit source migration witness")
def test_kinematic_functions_were_extracted_without_math_changes():
    old = Path(os.environ["YAP_MIGRATION_BASELINE"])/"tools/native_coupon.py"
    current = Path(kinematics.__file__)
    def functions(path):
        return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(path.read_bytes()).body
            if isinstance(node,ast.FunctionDef)}
    previous,new = functions(old),functions(current)
    assert set(new) <= set(previous)
    assert all(new[name] == previous[name] for name in new)


def test_all_registered_geometry_aliases_have_complete_equivalence():
    path = Path(__file__).resolve().parents[1]/"prepared_coupon_catalog.json"
    index = catalog._read(str(path))
    records = index["operators"]
    expected = {f"BI{pitch}_Prepared{total}" for pitch in range(2,11) for total in range(20,101,10)}
    assert expected <= records.keys()
    assert len(records) == 95
    project = runtime.application_module("mechanics_runtime").COMPONENT_PROJECT_ROOT
    root = Path(os.environ.get("YAP_NORMALIZED_OPERATOR_ROOT", str(project/"distribution/operators/normalized")))
    for key,record in records.items():
        directory = root/record["directory"]
        documents = {}
        for name, field in (("EXPORT_COMPLETE.json", "export_complete_sha256"),
                            ("MIGRATION_EQUIVALENCE.json", "equivalence_sha256")):
            raw = (directory/name).read_bytes()
            assert len(raw) <= 2*1024**2
            assert hashlib.sha256(raw).hexdigest() == record[field], key
            documents[name] = json.loads(raw)
        complete = documents["EXPORT_COMPLETE.json"]
        evidence = documents["MIGRATION_EQUIVALENCE.json"]
        assert evidence["passed"] is True
        assert evidence["all_operands_identical"] is True
        assert evidence["all_retained_GP_state_signatures_identical"] is True
        assert evidence["manifest_sha256"] == complete["manifest_sha256"] == record["manifest_sha256"]
        assert evidence["baseline_sha256"] == complete["baseline_sha256"]
