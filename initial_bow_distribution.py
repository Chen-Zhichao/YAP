"""Frozen initial-bow population policy used by YAP-Stack case studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


ANCHOR_SIGMA_LEVEL = 3.0
TARGET_SOURCE_CV = 0.20
REFERENCE_SOURCE_CV = 0.10
POLICY_NAME = "preserve_reference_mean_plus_3sigma_anchor"


@dataclass(frozen=True)
class AnchorPreservingDistribution:
    reference_mean: float
    reference_cv: float = REFERENCE_SOURCE_CV
    target_cv: float = TARGET_SOURCE_CV
    sigma_level: float = ANCHOR_SIGMA_LEVEL

    @property
    def anchor(self) -> float:
        return self.reference_mean * (
            1.0 + self.sigma_level * self.reference_cv
        )

    @property
    def mean(self) -> float:
        return self.anchor / (1.0 + self.sigma_level * self.target_cv)

    @property
    def std(self) -> float:
        return self.target_cv * self.mean

    def sample(self, standard_normals: np.ndarray) -> np.ndarray:
        return self.mean + self.std * np.asarray(standard_normals, dtype=float)

    def audit(self) -> dict[str, float | str]:
        return {
            "policy": POLICY_NAME,
            **asdict(self),
            "mean": self.mean,
            "std": self.std,
            "preserved_mean_plus_3sigma_anchor": self.anchor,
        }


# D2D source amplitude 0.5 with CV=0.1 was calibrated to an approximately
# 80-um baseline-die three-sigma warpage. W2W uses unit source amplitude for
# the corresponding full-wafer curvature projection.
D2D_INITIAL_BOW = AnchorPreservingDistribution(reference_mean=0.50)
W2W_INITIAL_BOW = AnchorPreservingDistribution(reference_mean=1.00)

