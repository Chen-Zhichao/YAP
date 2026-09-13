#!/usr/bin/env python3
"""Audit, without copying, the uploaded Fig. 19 historical archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_ARCHIVE = ROOT.parent / "YAPlus_1204" / "replica_distance.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    return parser.parse_args()


def matlab_matrix(source: str, variable: str) -> list[list[float]]:
    match = re.search(rf"{re.escape(variable)}\s*=\s*\[(.*?)\];", source, re.S)
    if match is None:
        raise RuntimeError(f"missing MATLAB matrix {variable}")
    rows = []
    for row in match.group(1).split(";"):
        values = [float(token) for token in re.findall(
            r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", row
        )]
        if values:
            rows.append(values)
    return rows


def main() -> None:
    args = parse_args()
    exp = OmegaConf.load(HERE / "config.yaml").experiment
    audit = exp.uploaded_replica_distance_audit
    digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    assert digest == str(audit.archive_sha256)

    with ZipFile(args.archive) as archive:
        names = set(archive.namelist())
        member = str(audit.exact_bar_source_member)
        source = archive.read(member).decode("utf-8-sig")
        w2w_rows = matlab_matrix(source, "w2w_yield")
        d2w_rows = matlab_matrix(source, "d2w_yield")
        exact = {
            "w2w": {"d1": [row[0] for row in w2w_rows],
                    "d01": [row[1] for row in w2w_rows]},
            "d2w": {"d1": [row[0] for row in d2w_rows],
                    "d01": [row[1] for row in d2w_rows]},
        }
        configured = {
            str(side): {
                str(density.tag): [float(v) for v in density.paper[str(side)]]
                for density in exp.densities
            }
            for side in exp.sides
        }
        assert exact == configured

        npy_members = sorted(name for name in names if name.endswith(".npy"))
        distributions = []
        for name in npy_members:
            mapping = np.load(BytesIO(archive.read(name)), allow_pickle=True).item()
            matrix = np.asarray(list(mapping.values()), dtype=float)
            assert matrix.shape == (999, 10)
            size_match = re.search(r"size_([0-9.]+)_dist_([0-9.]+)", name)
            assert size_match is not None
            die_size_um = float(size_match.group(1))
            distance_pitches = float(size_match.group(2))
            # These axes are stated by the two MATLAB consumers in the zip.
            # The old W2W modeling config uses a 10 um pitch, so dist=20/40/
            # 60/80 in the filenames maps to 200/400/600/800 um in the plot.
            density_axis_cm2 = (
                np.logspace(-2, -0.25, 10)
                if die_size_um == 10000.0
                else np.logspace(-2, 0.65, 10)
            )
            lambdas_per_cm2 = -np.log(matrix) / density_axis_cm2[None, :]
            lambda_per_layout = np.mean(lambdas_per_cm2, axis=1)
            inferred_mean_yield_d01 = float(np.mean(np.exp(-0.1 * lambda_per_layout)))
            inferred_mean_yield_d1 = float(np.mean(np.exp(-lambda_per_layout)))
            paper_index = int(distance_pitches / 20.0)
            distributions.append({
                "member": name,
                "bonding_side": "w2w",
                "die_size_um": die_size_um,
                "distance_pitches_at_old_10um_pitch": distance_pitches,
                "distance_um": distance_pitches * 10.0,
                "density_axis_cm2_from_matlab_consumer": density_axis_cm2.tolist(),
                "dictionary_entries": len(mapping),
                "matrix_shape": list(matrix.shape),
                "minimum": float(matrix.min()),
                "maximum": float(matrix.max()),
                "lambda_per_cm2_mean": float(np.mean(lambda_per_layout)),
                "lambda_per_cm2_std_across_layouts": float(np.std(lambda_per_layout)),
                "max_lambda_drift_across_density_columns": float(
                    np.max(np.ptp(lambdas_per_cm2, axis=1))
                ),
                "inferred_population_mean_yield": {
                    "density_0.1_cm2": inferred_mean_yield_d01,
                    "density_1_cm2": inferred_mean_yield_d1,
                },
                "fig19_exact_w2w_bar": {
                    "density_0.1_cm2": exact["w2w"]["d01"][paper_index],
                    "density_1_cm2": exact["w2w"]["d1"][paper_index],
                } if die_size_um == 10000.0 else None,
            })

        notebook = archive.read(str(audit.conversion_notebook_member)).decode("utf-8")
        notebook_role = (
            "NPY-to-MAT conversion only"
            if "scipy.io.savemat" in notebook and "os.listdir(folder_path)" in notebook
            else "unconfirmed"
        )

    output = {
        "archive": str(args.archive),
        "archive_sha256": digest,
        "exact_bar_source_member": member,
        "exact_bar_values": exact,
        "exact_values_match_config": True,
        "npy_distribution_count": len(distributions),
        "npy_distribution_scope": (
            "W2W only; the two MATLAB plotting consumers are named "
            "w2w_replica_dist_across_die_size_{10,3d2}.m"
        ),
        "npy_distributions": distributions,
        "exact_bar_poisson_residual": {
            side: {
                "identity": "Y(1 cm^-2) - Y(0.1 cm^-2)^10",
                "residuals": (
                    np.asarray(exact[side]["d1"])
                    - np.asarray(exact[side]["d01"]) ** 10
                ).tolist(),
                "max_absolute_residual": float(np.max(np.abs(
                    np.asarray(exact[side]["d1"])
                    - np.asarray(exact[side]["d01"]) ** 10
                ))),
            }
            for side in ("w2w", "d2w")
        },
        "conversion_notebook_role": notebook_role,
        "generation_script_recovered": False,
    }
    destination = HERE / "results" / "historical_source_audit.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "archive_sha256": digest,
        "exact_values_match_config": True,
        "npy_distribution_count": len(distributions),
        "conversion_notebook_role": notebook_role,
    }, indent=2))


if __name__ == "__main__":
    main()
