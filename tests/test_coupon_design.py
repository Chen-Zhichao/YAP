"""Generate and plan experimental coupon designs without loading operators."""
from omegaconf import OmegaConf
import pytest

from generate_assembly_design import generate_design
from D2W.assembly_order import load_design


def options(tmp_path, monkeypatch):
    monkeypatch.setenv("YAP_RUN_ROOT", str(tmp_path / "runs"))
    return dict(name="coupon90_three", dies=3, pitch=4, total_thickness=90.,
                initial_bows=[1., -1., 2.], orders="all", order_workers=1,
                operator_package="coupon:BI4_Prepared90", output_root=tmp_path / "designs")


def test_generated_complete_design_uses_normal_d2w_input_path(tmp_path, monkeypatch):
    kwargs = options(tmp_path, monkeypatch)
    generated = generate_design(**kwargs)
    cfg = OmegaConf.load(generated["config_path"]).d2w_modeling
    cfg.OVERLAY_SAMPLES = 2
    _, plan, _, _, provenance = load_design(cfg, generated["input_dir"], kwargs["name"])
    assert len(plan["routes"]) == 2
    assert plan["context"]["geometry"]["total_thickness_um"] == 90.
    request = provenance["operator_request"]
    assert request["operator_package"] == "coupon:BI4_Prepared90"
    assert request["bonding_pitch_um"] == 4.
    assert request["operator_inputs_verified"] is False
    assert request["assembly_accuracy_validated"] is False
    assert request["W2W_thinning_available"] is False


@pytest.mark.parametrize("change,match", [(dict(bonding_mode="w2w"), "D2W only"),
                                        (dict(pitch=2), "pitch")])
def test_invalid_coupon_generation_leaves_no_partial_design(tmp_path, monkeypatch, change, match):
    kwargs = {**options(tmp_path, monkeypatch), **change}
    with pytest.raises(ValueError, match=match):
        generate_design(**kwargs)
    assert not (tmp_path / "designs").exists()
