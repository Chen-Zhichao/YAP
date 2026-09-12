"""Regression tests for die-level overlay corner aggregation."""

from __future__ import annotations

import numpy as np
import unittest
from scipy.stats import norm

from D2W.overlay_yield_calculator import (
    _samplewise_worst_corner_yield as d2w_samplewise_yield,
)
from W2W.overlay_yield_calculator import (
    _samplewise_worst_corner_yield as w2w_samplewise_yield,
)


class SamplewiseWorstCornerTest(unittest.TestCase):
    calculators = (w2w_samplewise_yield, d2w_samplewise_yield)

    def test_worst_corner_is_selected_before_sample_average(self):
        corners = np.array(
            [
                [0.00, 0.08, 0.02],
                [0.08, 0.00, 0.04],
                [0.03, 0.05, 0.01],
                [0.02, 0.01, 0.07],
            ]
        )
        limit = 0.10
        random_mean = 0.0
        random_std = 0.02

        worst_per_sample = np.max(corners, axis=0)
        expected = np.mean(
            norm.cdf(limit - worst_per_sample, loc=random_mean, scale=random_std)
            - norm.cdf(-limit - worst_per_sample, loc=random_mean, scale=random_std)
        )
        corner_yields = np.mean(
            norm.cdf(limit - corners, loc=random_mean, scale=random_std)
            - norm.cdf(-limit - corners, loc=random_mean, scale=random_std),
            axis=1,
        )

        for calculator in self.calculators:
            with self.subTest(calculator=calculator.__module__):
                result = calculator(corners, limit, random_mean, random_std)
                self.assertAlmostEqual(result, expected)
                # Guard against the October-2025 regression: averaging each
                # corner before choosing the lowest yield is different.
                self.assertGreater(abs(result - np.min(corner_yields)), 0.05)

    def test_corner_samples_must_be_two_dimensional(self):
        for calculator in self.calculators:
            with self.subTest(calculator=calculator.__module__):
                with self.assertRaisesRegex(ValueError, "2-D"):
                    calculator(np.array([0.01, 0.02]), 0.1, 0.0, 0.02)


if __name__ == "__main__":
    unittest.main()
