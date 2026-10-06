#!/usr/bin/env python3
"""Export the double-sided-fabrication die geometry with MAPDL EPLOT."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import shutil
import sys


YAP_ROOT = Path(__file__).resolve().parents[3]
STACK_ROOT = YAP_ROOT.parent
VERIFY_ROOT = STACK_ROOT / "Ansys_Verify"
EXP09_ROOT = VERIFY_ROOT / "09_effective_material_die_validation"
sys.path.insert(0, str(VERIFY_ROOT))
sys.path.insert(0, str(EXP09_ROOT))

from common.ansys_utils import run_mapdl  # noqa: E402
from exp09_mapdl_helpers import add_base_materials  # noqa: E402
from export_exp09_ansys_mesh_images import add_layer_components  # noqa: E402
from run_exp09_realistic_rve_residual_anneal import (  # noqa: E402
    Case,
    Layer,
    add_realistic_geometry,
    realistic_layers,
)


DEFAULT_OUTPUT = YAP_ROOT / "D2W/output/paper_figures/ansys_dsf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=192)
    parser.add_argument("--nz", type=int, default=192)
    parser.add_argument("--tile-um", type=float, default=100.0)
    parser.add_argument("--reference-span-um", type=float, default=1000.0)
    parser.add_argument("--substrate-um", type=float, default=50.0)
    parser.add_argument("--bi-um", type=float, default=1.5)
    parser.add_argument("--bonding-pad-pitch-um", type=float, default=10.0)
    parser.add_argument("--bonding-pad-ratio", type=float, default=0.5)
    parser.add_argument(
        "--beol-profile",
        choices=("asap7", "intel_m0_m11", "intel_with_plateup"),
        default="intel_with_plateup",
    )
    parser.add_argument(
        "--beol-density", choices=("low", "nominal", "high"), default="nominal"
    )
    parser.add_argument(
        "--tsv-density", choices=("none", "nominal", "dense"), default="nominal"
    )
    parser.add_argument("--ansys-timeout-sec", type=float, default=900.0)
    parser.add_argument("--ansys-mem-mb", type=int, default=2048)
    parser.add_argument("--ansys-db-mb", type=int, default=1024)
    parser.add_argument("--deck-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def double_sided_layers(args: argparse.Namespace) -> list[Layer]:
    """Return back-BI / substrate / BEOL / front-BI layer ordering."""
    bi_um = float(args.bi_um)
    front_stack = realistic_layers(args)
    shifted = [
        replace(layer, y0_um=layer.y0_um + bi_um, y1_um=layer.y1_um + bi_um)
        for layer in front_stack
    ]
    back_bi = Layer(0.0, bi_um, "bonding", "Back_SiO2_Cu_pad_array")
    return [back_bi, *shifted]


def build_apdl(args: argparse.Namespace) -> str:
    case = Case(1, "double_sided_fabrication_mesh", 0.0, "plastic_creep")
    layers = double_sided_layers(args)
    lines: list[str] = [
        "/FILNAME,e09dsf,1",
        "/PREP7",
        "! MAPDL EPLOT source for the YAP-Stack DSF publication schematic.",
        "! Layer order: back BI / Si-TSV / BEOL / front BI.",
        f"! tile={args.tile_um:g} um, substrate={args.substrate_um:g} um, BI={args.bi_um:g} um",
        "/NOPR",
        "SHPP,OFF",
    ]
    add_base_materials(lines)
    add_realistic_geometry(lines, case, layers, args)
    add_layer_components(lines, layers, args)
    lines += [
        "/GOPR",
        "ALLSEL,ALL",
        "/GRAPHICS,FULL",
        "/ESHAPE,1",
        "/EDGE,1,1",
        "/TRIAD,OFF",
        "/NUMBER,1",
        "/PNUM,MAT,1",
        "/PLOPTS,INFO,0",
        "/PLOPTS,LEG1,0",
        "/PLOPTS,LEG2,0",
        "/PLOPTS,LOGO,0",
        "/PLOPTS,FRAME,0",
        "/TITLE, ",
        "/VIEW,1,1,1,1",
        "/VUP,1,Y",
        "/AUTO,1",
        "/SHOW,PNG",
        "EPLOT",
        "/SHOW,CLOSE",
        "FINISH",
        "/EXIT,NOSAVE",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    run_dir = output / "mapdl_run"
    output.mkdir(parents=True, exist_ok=True)
    if args.deck_only:
        deck = run_dir / "e09dsf.dat"
        run_dir.mkdir(parents=True, exist_ok=True)
        deck.write_text(build_apdl(args), encoding="ascii")
        print(deck)
        return 0
    info = run_mapdl(
        run_dir=run_dir,
        jobname="e09dsf",
        dat_text=build_apdl(args),
        mem_mb=int(args.ansys_mem_mb),
        db_mb=int(args.ansys_db_mb),
        mode="smp",
        np=1,
        timeout_sec=float(args.ansys_timeout_sec),
    )
    pngs = sorted(run_dir.glob("*.png"))
    if info.get("rc") != 0 or info.get("ansys_error_count") != 0 or not pngs:
        raise RuntimeError(f"MAPDL DSF export failed: {info}")
    raw = output / "ANSYS_DSF_MATERIAL_ISO_RAW.png"
    shutil.copy2(pngs[0], raw)
    print(
        f"MAPDL rc={info.get('rc')} errors={info.get('ansys_error_count')} "
        f"warnings={info.get('ansys_warning_count')}"
    )
    print(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
