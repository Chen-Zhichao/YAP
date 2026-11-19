"""Offline migration CLI for the common numeric operator contract."""
import argparse
import json

from . import PREFIX, application_module, export_library
from ..coupon_operator_library import load_prepared_coupon_library, prepared_operator_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True,help="Existing authenticated coupon selector")
    parser.add_argument("--output",required=True,help="Fresh directory outside the source and shared distribution")
    parser.add_argument("--pitch-um",type=float,required=True)
    parser.add_argument("--total-um",type=float,required=True)
    parser.add_argument("--width-mm",type=float,default=10.)
    parser.add_argument("--length-mm",type=float,default=10.)
    parser.add_argument("--front-bi-um",type=float,default=1.5)
    parser.add_argument("--back-bi-um",type=float,default=1.5)
    parser.add_argument("--beol-um",type=float,default=10.)
    args = parser.parse_args()
    geometry = dict(width_mm=args.width_mm,length_mm=args.length_mm,total_thickness_um=args.total_um,
        front_bi_thickness_um=args.front_bi_um,back_bi_thickness_um=args.back_bi_um,beol_thickness_um=args.beol_um)
    prepared_operator_request(args.source,geometry,layout=(args.pitch_um,args.pitch_um,args.pitch_um/2.))
    application_module("mechanics_runtime").configure_runtime_storage()
    original,proof = load_prepared_coupon_library(args.source,geometry)
    pin = export_library(original,proof,args.output,geometry=geometry,pitch_um=args.pitch_um)
    from pathlib import Path
    print(json.dumps(dict(operator_package=PREFIX+pin+":"+str(args.pitch_um)+":"+str(Path(args.output).absolute()),
        migration_accuracy_qualified=False,assembly_accuracy_validated=False),indent=2))


if __name__ == "__main__":
    main()
