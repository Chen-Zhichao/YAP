"""Geometry-independent packet validation; historical parity lives in migration evidence."""
import json
from pathlib import Path

import numpy as np
import pytest

from tools import operators


def fixture():
    arrays = dict(K=np.eye(9),f=np.zeros(9),E=np.zeros((2,6,9)),weights=np.ones(2),
        macro_scale=np.ones(6),bubble_scale=np.ones(3),global_scale=np.ones(6),
        cu_lambda=np.eye(9),cu_shear=np.eye(9),cu_thermal=np.zeros((9,6)))
    document = dict(schema=operators.SCHEMA,profile=operators.PROFILE,integration_points=2,
        geometry=dict(width_mm=10.,length_mm=10.,total_thickness_um=20.,
            front_bi_thickness_um=1.5,back_bi_thickness_um=1.5,beol_thickness_um=10.),
        frame=dict(area_mm2=100.,operator_reference_y_mm=.01,global_reference_y_mm=0.),
        pitch_um=7.,diameter_pitch_ratio=.5,temperature_range_c=[25.,350.],
        reference_temperature_c=25.,Cu_state_adoption=False)
    return arrays,document


def test_numeric_contract():
    arrays,document = fixture()
    operators.validate(arrays,document)
    assert all(not a.flags.writeable for a in arrays.values())


@pytest.mark.parametrize("kind,rank", [("native", 9), ("reference", 9), ("reference", 11),
    ("owned483", 9), ("owned483", 27), ("owned483", 55), ("owned483", 57),
    ("streamed483", 11), ("streamed483", 12), ("coordinate483", 9)])
def test_v2_contract_preserves_natural_dimensions(kind, rank):
    _, document = fixture()
    execution = dict(kind=kind)
    if kind in ("reference", "streamed483", "coordinate483"):
        execution["block_size"] = 32768 if kind == "reference" else 4096
    if kind == "reference":
        execution["zero_creep_shortcut"] = True
    document.update(schema=operators.SCHEMA_V2, profile="constant_elastic_dense",
                    coordinate_size=rank, execution=execution)
    arrays = {name: np.zeros(shape, dtype="<f8") for name, shape in operators.operand_shapes(document).items()}
    arrays["K"] = np.eye(rank)
    for name in ("weights", "macro_scale", "bubble_scale", "global_scale"):
        arrays[name].fill(1.)
    operators.validate(arrays, document)
    assert arrays["E"].shape == (2, 6, rank)
    assert arrays["bubble_scale"].shape == (rank-6,)
    assert all(not value.flags.writeable for value in arrays.values())


@pytest.mark.parametrize("execution,rank", [(dict(kind="unknown"), 9),
    (dict(kind="streamed483", block_size=0), 12),
    (dict(kind="coordinate483", block_size=True), 9),
    (dict(kind="reference", block_size=32768, zero_creep_shortcut=True), 12),
    (dict(kind="reference", block_size=4096, zero_creep_shortcut=True), 11),
    (dict(kind="reference", block_size=32768, zero_creep_shortcut=1), 11)])
def test_v2_rejects_unknown_or_inconsistent_execution_contract(execution, rank):
    _, document = fixture()
    document.update(schema=operators.SCHEMA_V2, profile="constant_elastic_dense",
                    coordinate_size=rank, execution=execution)
    with pytest.raises(ValueError):
        operators.contract(document)


@pytest.mark.parametrize("change",["profile","geometry","weights","maps","state","datum"])
def test_invalid_contract_rejected(change):
    arrays,document = fixture()
    if change == "profile":document["profile"] = "unknown"
    if change == "geometry":document["geometry"]["total_thickness_um"] = 12.
    if change == "weights":arrays["weights"][0] = -1.
    if change == "maps":arrays["E"] = np.zeros((2,6,10))
    if change == "state":document["Cu_state_adoption"] = True
    if change == "datum":document["frame"]["global_reference_y_mm"] = 1.
    with pytest.raises(ValueError):operators.validate(arrays,document)


def test_normalized_request_is_geometry_independent_and_pure(monkeypatch):
    _,document = fixture()
    def no_reads(*args,**kwargs):raise AssertionError("pure request must not read storage")
    monkeypatch.setattr(Path,"open",no_reads)
    for total,pitch in ((20.,7.),(60.,7.),(40.,8.)):
        geometry = dict(document["geometry"],total_thickness_um=total)
        selector = operators.PREFIX+"a"*64+f":{pitch}:/not/installed/operator"
        request = operators.prepared_request(selector,geometry,layout=(pitch,pitch,pitch/2.))
        assert request["bonding_pitch_um"] == pitch
        assert request["operator_inputs_verified"] is False
        assert request["retained_si_thickness_um"] == total-13.


@pytest.mark.parametrize("suffix",["no-pin:7:/data", "a"*64+":nan:/data",
    "a"*64+":7:/data/../other", "a"*64+":7:/data/.snapshot/x", "a"*64+":7:relative"])
def test_bad_selectors_rejected(suffix):
    _,document = fixture()
    with pytest.raises(ValueError):operators.prepared_request(operators.PREFIX+suffix,document["geometry"])


def test_application_dispatches_common_contract_without_legacy_factory(monkeypatch):
    from W2W import assembly_operator_library as api
    _,document = fixture()
    selector = operators.PREFIX+"a"*64+":7:/not/installed/operator"
    request = api.prepared_operator_request(selector,document["geometry"],layout=(7.,7.,3.5))
    assert request["manifest_sha256"] == "a"*64
    sentinel = object()
    monkeypatch.setattr(operators,"load_selected_library",lambda *args,**kwargs:sentinel)
    configured = []
    monkeypatch.setattr(operators,"configure_runtime",lambda:configured.append(True))
    api.require_prepared_operator(selector,document["geometry"])
    assert configured == [True]
    assert api.load_prepared_die_library(selector,document["geometry"]) is sentinel
    with pytest.raises(ValueError):api.prepared_operator_request(selector,dict(diameter_mm=300.))


@pytest.mark.parametrize("change",["hash","rank","compressed","duplicate","object","census"])
def test_bad_packet_rejected_before_material_construction(tmp_path,monkeypatch,change):
    from tools.operators import runtime as registry
    import zipfile
    arrays,document = fixture()
    directory = tmp_path/"packet"
    directory.mkdir()
    packet = directory/"OPERANDS.npz"
    if change == "rank":arrays["K"] = np.eye(10)
    if change == "object":arrays["f"] = np.array([object()],dtype=object)
    writer = np.savez_compressed if change == "compressed" else np.savez
    writer(packet,**arrays)
    if change == "duplicate":
        with zipfile.ZipFile(packet,"a") as archive:
            archive.writestr("f.npy",archive.read("f.npy"))
    if change == "census":document["integration_points"] = 10**30
    document["packet"] = dict(path="OPERANDS.npz",sha256=operators.sha(packet),bytes=packet.stat().st_size)
    if change == "hash":document["packet"]["sha256"] = "a"*64
    manifest = directory/"MANIFEST.json"
    manifest.write_text(json.dumps(document))
    def forbidden():raise AssertionError("reject malformed numeric operands before compiling")
    monkeypatch.setattr(registry,"_compiled_material",forbidden)
    with pytest.raises(ValueError):
        operators.load_library(directory,operators.sha(manifest),document["geometry"],pitch_um=7.)
