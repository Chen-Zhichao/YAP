"""Run the calibrated fabrication-induced initial-bow sensitivity study."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from YAP import manufacturing_bow as _bow
except ModuleNotFoundError:  # Direct execution from the YAP source directory.
    import manufacturing_bow as _bow

BOW_CONVENTION = _bow.BOW_CONVENTION
LITERATURE_ANCHOR = _bow.LITERATURE_ANCHOR
MODEL_STATUS = _bow.MODEL_STATUS
REFERENCE_MID_EDGE_UPPER_UM = _bow.REFERENCE_MID_EDGE_UPPER_UM
ManufacturingLayer = _bow.ManufacturingLayer
calibrate_fabrication_source = _bow.calibrate_fabrication_source
geometry_sensitivity_rows = _bow.geometry_sensitivity_rows


SI_E_MPA = 99496.90200129498
SI_NU = 0.2917809724509618
BEOL_E_MPA = 93300.0
BEOL_NU = 0.2295
BI_E_MPA = 84388.273369263
BI_NU = 0.20337942194439157


def baseline_layers(si_thickness_um):
    """Baseline 50-um-Si reference ledger with 10-um BEOL and 1.5-um BI."""
    return (
        ManufacturingLayer("tsv_si", si_thickness_um, SI_E_MPA, SI_NU,
                           is_bulk_silicon=True),
        ManufacturingLayer("beol", 10.0, BEOL_E_MPA, BEOL_NU,
                           eigenstrain_per_amplitude=1.0),
        ManufacturingLayer("bonding_interface", 1.5, BI_E_MPA, BI_NU),
    )


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path, rows):
    cvs = (0.05, 0.10, 0.20)
    sizes = (5.0, 10.0, 15.0)
    colors = {5.0: "#0072B2", 10.0: "#009E73", 15.0: "#D55E00"}
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), sharey=True)
    for axis, cv in zip(axes, cvs):
        for size in sizes:
            subset = sorted(
                (row for row in rows
                 if row["source_cv"] == cv and row["die_size_mm"] == size),
                key=lambda row: row["si_thickness_um"],
            )
            x = [row["si_thickness_um"] for row in subset]
            mean = [row["mean_signed_bow_um"] for row in subset]
            low = [row["mean_minus_3sigma_um"] for row in subset]
            high = [row["mean_plus_3sigma_um"] for row in subset]
            axis.plot(x, mean, marker="o", color=colors[size], label=f"{size:g} mm")
            axis.fill_between(x, low, high, color=colors[size], alpha=0.16)
        axis.set_title(f"Source CV = {100*cv:.0f}%")
        axis.set_xlabel("Si thickness (um)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Initial signed bow (um)")
    axes[-1].legend(title="Square die")
    figure.suptitle("Fabrication-induced initial bow: mean and +/-3 sigma")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    # source_sign=-1 chooses positive center-minus-mid-edge bow for this layer order.
    calibration = calibrate_fabrication_source(
        baseline_layers(50.0), reference_span_mm=10.0,
        reference_si_thickness_um=50.0,
        target_upper_magnitude_um=REFERENCE_MID_EDGE_UPPER_UM, source_cv=0.10,
        sigma_level=3.0, source_sign=-1.0)
    rows = geometry_sensitivity_rows(calibration, baseline_layers)
    payload = {
        "model_status": MODEL_STATUS,
        "bow_convention": BOW_CONVENTION,
        "reference_total_stack_thickness_um": 61.5,
        "literature_anchor": LITERATURE_ANCHOR,
        "reference_upper_overall_warpage_um": 80.0,
        "target_upper_mid_edge_bow_um": REFERENCE_MID_EDGE_UPPER_UM,
        "calibration": calibration.record(),
        "calibration_sha256": calibration.signature,
        "rows": rows,
    }
    (output_dir / "calibration.json").write_text(
        json.dumps(calibration.record(), indent=2) + "\n", encoding="ascii")
    (output_dir / "geometry_sensitivity.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="ascii")
    _write_csv(output_dir / "geometry_sensitivity.csv", rows)
    _plot(output_dir / "initial_bow_sensitivity.png", rows)

    reference = calibration.reference_moments
    print(f"Signed A_fab0 = "
          f"{calibration.source_sign * calibration.nominal_amplitude:.9e}")
    print(f"Reference signed bow = {reference['mean_signed_bow_um']:.6f} "
          f"+/- {reference['std_signed_bow_um']:.6f} um (1 sigma)")
    print(f"Reference |mean|+3sigma = {calibration.achieved_upper_magnitude_um:.6f} um")
    print(f"Saved {len(rows)} geometry/CV cases to {output_dir}")
    return calibration, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("D2W/output/case_studies/fabrication_initial_bow"))
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
