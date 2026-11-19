#!/usr/bin/env python3
"""Negative-binomial fabrication-yield model for physical chiplet instances."""

import math
import os

import numpy as np
from omegaconf import OmegaConf


CHIPLET_DENSITY_KEY = "CHIP_DEFECT_DENSITY_PER_CM2"
CHIPLET_CRITICAL_AREA_KEY = "CHIP_CRITICAL_AREA_FRACTION"
CHIPLET_ALPHA_KEY = "CHIP_DEFECT_CLUSTERING_ALPHA"
KNOWN_GOOD_DIE_REFERENCES_KEY = "KNOWN_GOOD_DIE_REFERENCES"


def _stack_chip_yield_parameters(cfg_dict):
    parameter_values = {
        CHIPLET_DENSITY_KEY: [],
        CHIPLET_CRITICAL_AREA_KEY: [],
        CHIPLET_ALPHA_KEY: [],
    }
    known_good_reference_sets = []
    for interface_name, cfg in cfg_dict.items():
        for key, values in parameter_values.items():
            if not hasattr(cfg, key):
                raise ValueError(
                    f"Missing required config value {key} "
                    f"for interface '{interface_name}'."
                )
            values.append(float(getattr(cfg, key)))
        known_good_references = getattr(cfg, KNOWN_GOOD_DIE_REFERENCES_KEY, [])
        if known_good_references is None:
            known_good_references = []
        known_good_reference_sets.append(
            frozenset(str(reference) for reference in known_good_references)
        )

    if not any(parameter_values.values()):
        raise ValueError("Cannot calculate chiplet yield without interface configs.")

    for key, values in parameter_values.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{key} must be finite.")
        if not np.allclose(values, values[0], rtol=0.0, atol=0.0):
            raise ValueError(
                f"All interfaces must use the same {key}; "
                "per-chiplet process parameters are not enabled yet."
            )

    density = parameter_values[CHIPLET_DENSITY_KEY][0]
    critical_area_fraction = parameter_values[CHIPLET_CRITICAL_AREA_KEY][0]
    alpha = parameter_values[CHIPLET_ALPHA_KEY][0]
    if density < 0.0:
        raise ValueError(f"{CHIPLET_DENSITY_KEY} must be non-negative.")
    if not 0.0 <= critical_area_fraction <= 1.0:
        raise ValueError(f"{CHIPLET_CRITICAL_AREA_KEY} must be in [0, 1].")
    if alpha <= 0.0:
        raise ValueError(f"{CHIPLET_ALPHA_KEY} must be positive.")
    if any(
        references != known_good_reference_sets[0]
        for references in known_good_reference_sets[1:]
    ):
        raise ValueError(
            f"All interfaces must use the same {KNOWN_GOOD_DIE_REFERENCES_KEY}."
        )
    return density, critical_area_fraction, alpha, known_good_reference_sets[0]


def chiplet_yield_records(cfg_dict, design_root_dir):
    """Return one negative-binomial yield record per physical die instance."""
    stack_path = os.path.join(design_root_dir, "generated_stack_config.3dbx")
    definitions_path = os.path.join(
        design_root_dir,
        "generated_chiplet_definitions.3dbv",
    )
    if not os.path.exists(stack_path) or not os.path.exists(definitions_path):
        raise FileNotFoundError(
            "Chiplet yield requires generated_stack_config.3dbx and "
            "generated_chiplet_definitions.3dbv."
        )

    stack_config = OmegaConf.load(stack_path)
    definitions = OmegaConf.load(definitions_path)
    if "Stack" not in stack_config or "ChipletInst" not in stack_config:
        raise ValueError(f"{stack_path} must contain Stack and ChipletInst sections.")
    if "ChipletDef" not in definitions:
        raise ValueError(f"{definitions_path} must contain a ChipletDef section.")
    if str(definitions.Header.unit).lower() != "micron":
        raise ValueError("Chiplet yield currently requires 3DBV dimensions in microns.")

    (
        density_per_cm2,
        critical_area_fraction,
        alpha,
        known_good_references,
    ) = _stack_chip_yield_parameters(cfg_dict)
    records = []
    for instance_name in stack_config.Stack.keys():
        if instance_name not in stack_config.ChipletInst:
            raise ValueError(f"Stack instance '{instance_name}' is missing from ChipletInst.")
        reference = str(stack_config.ChipletInst[instance_name].reference)
        if reference not in definitions.ChipletDef:
            raise ValueError(f"Chiplet definition '{reference}' is missing from ChipletDef.")

        chiplet_definition = definitions.ChipletDef[reference]
        if str(chiplet_definition.get("type", "die")).strip().lower() != "die":
            continue
        design_area = list(chiplet_definition.design_area)
        if len(design_area) != 2:
            raise ValueError(f"Chiplet '{reference}' design_area must contain width and length.")
        width_um, length_um = (float(design_area[0]), float(design_area[1]))
        if width_um <= 0.0 or length_um <= 0.0:
            raise ValueError(f"Chiplet '{reference}' design dimensions must be positive.")

        area_cm2 = width_um * length_um / 1.0e8
        critical_area_cm2 = area_cm2 * critical_area_fraction
        known_good_die = reference in known_good_references
        chip_yield = 1.0
        if not known_good_die:
            chip_yield = math.exp(
                -alpha * math.log1p(density_per_cm2 * critical_area_cm2 / alpha)
            )
        records.append(
            {
                "instance": str(instance_name),
                "reference": reference,
                "area_cm2": area_cm2,
                "critical_area_cm2": critical_area_cm2,
                "defect_density_per_cm2": density_per_cm2,
                "critical_area_fraction": critical_area_fraction,
                "clustering_alpha": alpha,
                "known_good_die": known_good_die,
                "yield": chip_yield,
            }
        )

    if not records:
        raise ValueError("The stack contains no ChipletDef entries with type 'die'.")
    return records


def chiplet_stack_yield(records):
    return float(np.prod([record["yield"] for record in records], dtype=np.float64))


def sample_chiplet_stack_survival(records, sample_shape):
    probabilities = np.asarray([record["yield"] for record in records], dtype=float)
    draws = np.random.random(tuple(np.atleast_1d(sample_shape).astype(int)) + (len(records),))
    return np.all(draws < probabilities, axis=-1)


def print_chiplet_yield_table(records):
    instance_width = max(len("Chiplet Instance"), *(len(r["instance"]) for r in records))
    reference_width = max(len("Reference"), *(len(r["reference"]) for r in records))
    print("Chiplet Yield Table:")
    print(
        f"{'Chiplet Instance':<{instance_width}} "
        f"{'Reference':<{reference_width}} "
        f"{'area_cm2':>10} {'Rcrit':>8} {'alpha':>8} {'KGD':>5} "
        f"{'D_per_cm2':>11} {'yield':>11}"
    )
    for record in records:
        print(
            f"{record['instance']:<{instance_width}} "
            f"{record['reference']:<{reference_width}} "
            f"{record['area_cm2']:>10.6f} "
            f"{record['critical_area_fraction']:>8.6f} "
            f"{record['clustering_alpha']:>8.3f} "
            f"{str(record['known_good_die']):>5} "
            f"{record['defect_density_per_cm2']:>11.6f} "
            f"{record['yield']:>11.6f}"
        )
    print(f"Chiplet stack yield: {chiplet_stack_yield(records):.6f}")
