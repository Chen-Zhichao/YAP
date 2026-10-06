"""Full assembly config layout and dispatch, without loading Cu operators."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

from omegaconf import OmegaConf
import pytest

from D2W.assembly_order import load_design as load_d2w
from W2W.assembly_design import load_design as load_w2w


ROOT = Path(__file__).resolve().parents[1]
PROFILES = [
    ("D2W", "design_6_4hi_p2", .86, 2., 5000),
    ("W2W", "design_6_4hi_p2", .99, 2., 5000),
    ("W2W", "design_6_4hi_p6_reference", .89, 6., 1666),
]
ACTIVE_FIELDS = (
    "ASSEMBLY_ORDER_ENABLED", "OUTPUT_DIR", "ANNEAL_MODE", "WAF_R_um", "dice_width",
    "PAD_ARRANGE_PATTERN", "CRITICAL_PAD_FRACTION", "POWER_GROUND_PAD_FRACTION",
    "DUMMY_PAD_FRACTION", "CONTACT_AREA_CONSTRAINT", "CRITICAL_DIST_CONSTRAINT",
    "SYSTEM_TRANSLATION_X_MEAN_um", "SYSTEM_TRANSLATION_X_STD_um",
    "SYSTEM_TRANSLATION_Y_MEAN_um", "SYSTEM_TRANSLATION_Y_STD_um",
    "SYSTEM_ROTATION_MEAN_rad", "SYSTEM_ROTATION_STD_rad",
    "RANDOM_MISALIGNMENT_MEAN_um", "RANDOM_MISALIGNMENT_STD_um",
    "MAGNIFICATION_CORRECTION", "OVERLAY_SAMPLES", "RANDOM_SEED",
    "MIN_AVAILABLE_MEMORY_GIB",
)


def config_path(domain, name):
    return ROOT / domain / "configs" / name / f"{name}.yaml"


@pytest.mark.parametrize("domain,name,gamma,pitch,rows", PROFILES)
def test_full_profiles_keep_p6_sections_and_current_parameters(domain, name, gamma, pitch, rows):
    document = OmegaConf.load(config_path(domain, name))
    reference = OmegaConf.load(ROOT / domain / "configs/design_6_4hi_p6/design_6_4hi_p6.yaml")
    assert set(document) == set(reference)
    for mode, template in reference.items():
        cfg = document[mode]
        assert set(template) <= set(cfg)
        assert cfg.ASSEMBLY_ORDER_ENABLED is True
        assert cfg.ANNEAL_MODE == "sequential"
        assert cfg.MAGNIFICATION_CORRECTION == gamma
        assert cfg.OVERLAY_SAMPLES == 8192
        assert cfg.RANDOM_SEED == 2026091601
        assert cfg.DIE_W_um == cfg.DIE_L_um == 10000.
        assert cfg.PITCH_r_um == cfg.PITCH_c_um == pitch
        assert cfg.PAD_TOP_R_um == cfg.PAD_BOT_R_um == pitch / 4
        assert cfg.PAD_ARR_ROW == cfg.PAD_ARR_COL == rows
        for field in ("ITF_TOP_THICK_um", "ITF_BOT_THICK_um"):
            if field in template:
                assert cfg[field] == (50. if domain == "D2W" else 775.)
        if "DIE_THICHKNESS_um" in template:
            assert cfg.DIE_THICHKNESS_um == 50.
        assert cfg.INPUT_DIR == f"input/{name}/"
        assert cfg.INTERFACE is None
        for field in ("TOP_INI_BOW_MEAN_um", "TOP_INI_BOW_STD_um",
                      "BOT_INI_BOW_MEAN_um", "BOT_INI_BOW_STD_um"):
            assert cfg[field] is None


@pytest.mark.parametrize("domain,name,gamma,pitch,rows", PROFILES)
def test_retained_inactive_groups_do_not_change_plan_or_overlay(domain, name, gamma, pitch, rows):
    full = OmegaConf.load(config_path(domain, name))[f"{domain.lower()}_modeling"]
    full.OVERLAY_SAMPLES = 8
    compact = OmegaConf.create({field: full[field] for field in ACTIVE_FIELDS if field in full})
    loader = load_d2w if domain == "D2W" else load_w2w
    design_dir = ROOT / domain / "input" / name
    _, plan, overlay, _, provenance = loader(full, design_dir, name)
    _, compact_plan, compact_overlay, _, compact_provenance = loader(compact, design_dir, name)
    assert plan == compact_plan
    assert overlay == compact_overlay
    assert provenance["site_count"] == compact_provenance["site_count"]
    assert provenance["critical_pad_count"] == compact_provenance["critical_pad_count"]


@pytest.mark.parametrize("domain,name,gamma,pitch,rows", PROFILES)
@pytest.mark.parametrize("entrypoint", ["calculator_main.py", "simulator_main.py"])
def test_reserved_simulation_profiles_rejected_before_dispatch(tmp_path, domain, name, gamma, pitch, rows, entrypoint):
    design = tmp_path / "design"
    shutil.copytree(ROOT / domain / "input" / name, design)
    original_files = {path.name: path.read_bytes() for path in design.iterdir() if path.is_file()}
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path / "mpl"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    proc = subprocess.run([
        sys.executable, str(ROOT / domain / entrypoint),
        "--config", str(config_path(domain, name)), "--mode", f"{domain.lower()}_simulation",
        "--ds_name", name, "--ds_dir", str(design),
    ], cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
    assert proc.returncode != 0
    assert "assembly-order designs support modeling only" in proc.stderr
    assert {path.name: path.read_bytes() for path in design.iterdir() if path.is_file()} == original_files
