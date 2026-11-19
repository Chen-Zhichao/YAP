"""Generated input consistency only; these tests do not qualify assembly physics."""

import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pytest

import generate_assembly_design as generator
from D2W.assembly_order import build_plan, load_stack_input
from D2W.compact_layout import compact_regular_geometry


@pytest.fixture
def options(tmp_path, monkeypatch):
    monkeypatch.setattr(generator, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv('YAP_RUN_ROOT', str(tmp_path))
    return dict(name="colleague_8hi", dies=8, pitch=2, total_thickness=50.,
                initial_bows=[10., -10., 20., -20., 5., -5., 15., -15.],
                output_root=tmp_path / "generated")


def load_result(result):
    path = result["input_dir"] / "stack.yaml"
    design = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    normalized = load_stack_input(path)
    return design, build_plan(normalized["stack"])


@pytest.mark.parametrize("orders,count", [("sequential", 1), ("balanced", 1), ("all", 429),
    ("((((1,2),(3,4)),(5,6)),(7,8))", 1)])
def test_eight_actual_bodies_and_complete_inputs(options, orders, count):
    result = generator.generate_design(**options, orders=orders)
    directory = result["input_dir"]
    design, plan = load_result(result)
    assert result["orders"] == len(plan["routes"]) == count
    assert design["mechanics"]["operator_package"] == "component:BI2_SI37"
    assert design["mechanics"]["temporal_profile"] == "component_graded25"
    assert "frozen_preview" not in design["mechanics"]
    assert plan["physical_readiness"] is False
    assert plan["context"]["receiving_wafer_model"] == "free_die_site_column_only"
    assert plan["context"]["initial_bows_um"] == options["initial_bows"]
    assert [v["thickness_um"] for v in plan["context"]["fabrication_layers"]] == [1.5, 10., 37., 1.5]
    for route in plan["routes"]:
        assert route["final"]["die_ids"] == list(range(1, 9))
        assert route["final"]["total_thicknesses_um"] == [50.] * 8
        assert len(route["events"]) == 7
        assert all("thinning" not in event for event in route["events"])
    names = {f"Die_{index}" for index in range(1, 9)}
    definitions = OmegaConf.load(directory / "generated_chiplet_definitions.3dbv")
    topology = OmegaConf.load(directory / "generated_stack_config.3dbx")
    assert set(definitions.ChipletDef) == set(topology.ChipletInst) == set(topology.Stack) == names
    assert definitions.Header.unit == topology.Header.unit == "micron"
    assert topology.Design.name == options["name"]
    assert set(topology.Connection) == {f"Conn{index}" for index in range(1, 8)}
    assert {path.name for path in directory.glob("*.3dbf")} == {f"{name}.3dbf" for name in names}
    assert len(list(directory.iterdir())) == 12
    for index in range(1, 9):
        name = f"Die_{index}"
        definition = definitions.ChipletDef[name]
        assert definition.thickness == 50.
        assert list(definition.design_area) == [10000., 10000.]
        assert topology.ChipletInst[name].reference == name
        node = topology.Stack[name]
        assert node.z == (index - 1) * 50.
        assert node.orient == "R0" and list(node.loc) == [0., 0.]
        assert node.get("root", False) is (index == 1)
        regions = ({f"From_Die_{index - 1}"} if index > 1 else set())
        regions |= {f"To_Die_{index + 1}"} if index < 8 else set()
        assert set(definition.regions) == regions
        for label, region in definition.regions.items():
            assert region.side == ("front" if label.startswith("From_") else "back")
            assert list(map(list, region.coords)) == [[0, 0], [10000, 0], [10000, 10000], [0, 10000]]
        if index > 1:
            connection = topology.Connection[f"Conn{index - 1}"]
            assert connection.top == f"{name}.regions.From_Die_{index - 1}"
            assert connection.bot == f"Die_{index - 1}.regions.To_{name}"
    if orders == "balanced":
        assert plan["routes"][0]["final"]["thermal_cycle_counts"] == [3] * 8
    if orders == "sequential":
        assert plan["routes"][0]["final"]["thermal_cycle_counts"] == [7, 7, 6, 5, 4, 3, 2, 1]


@pytest.mark.parametrize("pitch,total,package", [(2, 31.5, "component:BI2_SI18p5"),
    (3, 50., "component:BI3_SI37"), (6, 50., "component:BI6_SI37")])
def test_full_config_gamma_and_pitch_specific_compact_layout(options, pitch, total, package):
    result = generator.generate_design(**{**options, "pitch": pitch, "total_thickness": total})
    document = OmegaConf.load(result["config_path"])
    template = OmegaConf.load(generator.CONFIG_TEMPLATE)
    assert set(document) == set(template) == {"d2w_modeling", "d2w_simulation"}
    for mode, baseline in template.items():
        cfg = document[mode]
        assert set(baseline) <= set(cfg)
        assert cfg.MAGNIFICATION_CORRECTION == baseline.MAGNIFICATION_CORRECTION == .86
        assert cfg.ASSEMBLY_ORDER_ENABLED is True and cfg.ANNEAL_MODE == "sequential"
        assert cfg.OVERLAY_SAMPLES == baseline.OVERLAY_SAMPLES
        assert cfg.RANDOM_SEED == baseline.RANDOM_SEED
        assert cfg.DESIGN == options["name"]
        assert cfg.INPUT_DIR == str(result["input_dir"])
        assert cfg.PITCH_r_um == cfg.PITCH_c_um == pitch
        assert cfg.PAD_TOP_R_um == cfg.PAD_BOT_R_um == pitch / 4
        for field in ("ITF_TOP_THICK_um", "ITF_BOT_THICK_um", "DIE_THICHKNESS_um"):
            if field in baseline:
                assert cfg[field] == total
        assert Path(cfg.OUTPUT_DIR).is_relative_to(generator.PROJECT_ROOT)
    compact = compact_regular_geometry(document.d2w_modeling)
    layout = OmegaConf.load(result["input_dir"] / "compact_layout.yaml")
    for key in ("rows", "cols", "pitch_r_um", "pitch_c_um", "pad_array_l_um", "pad_array_w_um"):
        assert layout[key] == compact[key]
    assert layout.die_count == 8 and layout.bonding_interfaces == 7
    assert layout.pad_diameter_um == pitch / 2
    for category in ("critical", "power_ground", "dummy"):
        assert layout[f"num_{category}_pads_per_interface"] == compact[f"num_{category}_pads"]
    assert layout.num_total_pads_per_interface == compact["rows"] * compact["cols"]
    tech = OmegaConf.load(result["input_dir"] / "Die_8.3dbf")
    assert tech.Chiplet_Grid.pitch == pitch
    assert tech.Bump_Types.silicon_individual_bonding.bump_size == pitch / 2
    design, plan = load_result(result)
    assert design["mechanics"]["operator_package"] == package
    assert plan["context"]["geometry"]["total_thickness_um"] == total
    assert "borrowed TOTAL50 calibration assumptions" in (result["input_dir"] / "stack.yaml").read_text()


def test_legacy_total50_and_explicit_component_reference(options):
    legacy = generator.generate_design(**options, legacy_total50=True)
    design, _ = load_result(legacy)
    assert design["mechanics"]["operator_package"] == "resource:compact_p2/packages/BI2_50.json"
    assert design["mechanics"]["frozen_preview"] is True
    assert design["mechanics"]["temporal_profile"] == "distilled26"
    other = generator.generate_design(**{**options, "name": "explicit"}, operator_package="component:registered_id")
    assert load_result(other)[0]["mechanics"]["operator_package"] == "component:registered_id"


@pytest.mark.parametrize("count,orders", [(2, "all"), (9, "sequential"), (32, "balanced")])
def test_other_counts_without_hidden_base(options, count, orders):
    result = generator.generate_design(**{**options, "dies": count, "initial_bows": [0.] * count}, orders=orders)
    _, plan = load_result(result)
    assert plan["routes"][0]["final"]["die_ids"] == list(range(1, count + 1))
    assert len(plan["routes"][0]["events"]) == count - 1


@pytest.mark.parametrize("change", [
    {"name": "../bad"}, {"name": "bad/name"}, {"dies": 1}, {"dies": 33}, {"dies": True},
    {"initial_bows": [0.]}, {"initial_bows": [float("nan")] * 8},
    {"total_thickness": 13.}, {"total_thickness": float("inf")}, {"pitch": 2.5}, {"pitch": 0},
    {"orders": "((1,2),(3,4))"}, {"orders": "__import__('os')"},
    {"dies": 9, "initial_bows": [0.] * 9, "orders": "all"},
    {"operator_package": "resource:compact_p2/packages/BI2_50.json"},
    {"operator_package": ""},
    {"legacy_total50": True, "pitch": 3}, {"legacy_total50": True, "total_thickness": 80.},
    {"legacy_total50": True, "operator_package": "component:BI2_SI37"},
    {"input_root": Path("relative/input")},
])
def test_invalid_requests_leave_no_design(options, change):
    with pytest.raises(ValueError):
        generator.generate_design(**{**options, **change})
    assert not options["output_root"].exists()


def test_roots_and_no_overwrite(options, tmp_path):
    custom = {**options, "output_root": None, "input_root": tmp_path / "inputs",
              "config_root": tmp_path / "configs"}
    result = generator.generate_design(**custom)
    assert result["input_dir"] == tmp_path / "inputs" / options["name"]
    assert result["config_path"] == tmp_path / "configs" / options["name"] / f"{options['name']}.yaml"
    original = (result["input_dir"] / "stack.yaml").read_bytes()
    with pytest.raises(FileExistsError):
        generator.generate_design(**custom)
    assert (result["input_dir"] / "stack.yaml").read_bytes() == original
    for root in (tmp_path / "#snapshot/design",
                 tmp_path / ".snapshot/design", Path("relative")):
        with pytest.raises(ValueError):
            generator.generate_design(**{**options, "output_root": root})
    with pytest.raises(ValueError, match="overlap"):
        generator.generate_design(**{**custom, "config_root": custom["input_root"]})
    link = tmp_path / "escape"
    target = tmp_path / 'alias_target'
    target.mkdir()
    link.symlink_to(target, target_is_directory=True)
    with pytest.warns(UserWarning, match='resolved storage path'):
        result = generator.generate_design(**{**options, 'output_root': link / 'outside'})
    assert result['input_dir'].is_relative_to(target)


@pytest.mark.parametrize("legacy", [False, True])
def test_cli_creates_full_yaml_without_operator_loading(tmp_path, legacy):
    # The real CLI keeps the canonical root guard; pytest's work root must be canonical.
    if not tmp_path.is_relative_to(generator.PROJECT_ROOT):
        pytest.skip("CLI integration requires --basetemp inside the canonical project")
    root = tmp_path / "cli"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", YAP_NATIVE_RUNTIME_DIR=str(tmp_path / "absent"))
    process = subprocess.run([sys.executable, "-B", str(generator.ROOT / "generate_assembly_design.py"),
        "--name", "cli_8hi", "--dies", "8", "--pitch", "2", "--total-thickness", "50",
        "--initial-bows", "10", "-10", "20", "-20", "5", "-5", "15", "-15",
        "--orders", "all", "--output-root", str(root),
        *(["--legacy-total50"] if legacy else [])],
        env=env, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
    assert "Selected orders: 429" in process.stdout
    assert "no operators loaded" in process.stdout
    config = OmegaConf.load(root / "D2W/configs/cli_8hi/cli_8hi.yaml")
    assert set(config) == {"d2w_modeling", "d2w_simulation"}
    planned = subprocess.run([sys.executable, "-B", str(generator.ROOT / "D2W/calculator_main.py"),
        "--config", str(root / "D2W/configs/cli_8hi/cli_8hi.yaml"),
        "--mode", "d2w_modeling", "--ds_name", "cli_8hi", "--ds_dir", str(root / "D2W/input/cli_8hi"),
        "--mechanism-filter", "overlay", "--plan-only", "--output-dir", str(root / "results")],
        env=env, capture_output=True, text=True, timeout=90)
    assert planned.returncode == 0, planned.stderr
    assert "Planned 429 orders" in planned.stdout
    plans = list((root / "results/cli_8hi").glob("*/PLAN.json"))
    assert len(plans) == 1
    assert not list((root / "results/cli_8hi").glob("*/MECHANICS.json"))
    again = subprocess.run([sys.executable, "-B", str(generator.ROOT / "generate_assembly_design.py"),
        "--name", "cli_8hi", "--dies", "8", "--pitch", "2", "--total-thickness", "50",
        "--initial-bows", *(["0"] * 8), "--output-root", str(root)],
        env=env, capture_output=True, text=True, timeout=60)
    assert again.returncode != 0
    assert "refusing to overwrite" in again.stderr
