"""Regression tests for non-overlapping redundant block placement."""

from __future__ import annotations

import math
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

# Bitmap generation itself does not use OpenCV; only downstream morphology
# helpers in the same modules do.  Keep this unit test independent of that
# optional runtime dependency without leaving a stub in sys.modules.
try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    with patch.dict(sys.modules, {"cv2": types.ModuleType("cv2")}):
        import D2W.pad_bitmap_generation as d2w_generation
        import W2W.pad_bitmap_generation as w2w_generation
else:
    import D2W.pad_bitmap_generation as d2w_generation
    import W2W.pad_bitmap_generation as w2w_generation

place_d2w_pairs = d2w_generation.place_redundant_block_pairs
place_w2w_pairs = w2w_generation.place_redundant_block_pairs


class RedundantBlockMatchingTest(unittest.TestCase):
    placers = (place_w2w_pairs, place_d2w_pairs)

    def test_fig19_full_grid_has_complete_unique_matching(self):
        rows = cols = 50
        blocks = np.arange(rows * cols)
        for placer in self.placers:
            for distance in (1, 2, 3, 4):
                with self.subTest(placer=placer.__module__, distance=distance):
                    np.random.seed(20260120)
                    pairs = placer(blocks, rows, cols, 1250, distance)
                    flattened = [block_id for pair in pairs for block_id in pair]
                    self.assertEqual(len(pairs), 1250)
                    self.assertEqual(len(set(flattened)), 2500)
                    for main, copy in pairs:
                        main_row, main_col = divmod(main, cols)
                        copy_row, copy_col = divmod(copy, cols)
                        actual_distance = math.hypot(
                            main_row - copy_row, main_col - copy_col
                        )
                        self.assertGreater(actual_distance, distance - 0.1)
                        self.assertLessEqual(actual_distance, distance + 0.5)

    def test_seeded_placement_is_reproducible(self):
        blocks = np.arange(16 * 16)
        for placer in self.placers:
            with self.subTest(placer=placer.__module__):
                np.random.seed(17)
                first = placer(blocks, 16, 16, 77, 3)
                np.random.seed(17)
                second = placer(blocks, 16, 16, 77, 3)
                self.assertEqual(first, second)

    def test_impossible_request_reports_maximum_matching(self):
        blocks = np.array([0, 2])
        for placer in self.placers:
            with self.subTest(placer=placer.__module__):
                with self.assertRaisesRegex(ValueError, "maximum matching 0"):
                    placer(blocks, 1, 3, 1, 1)

    def test_full_generators_support_fig19_all_redundant_layout(self):
        cfg = SimpleNamespace(
            PAD_ARR_ROW=40,
            PAD_ARR_COL=40,
            PITCH_um=1.0,
            critical_pad_ratio=0.0,
            redundant_pad_ratio=1.0,
            pad_block_size=4,
            redundant_logical_pad_ratio=0.5,
            redundant_logical_pad_copy=2,
            redundant_logical_pad_dist=4,
            DEBUG=False,
        )
        generators = (
            (w2w_generation, w2w_generation.pad_bitmap_generate),
            (d2w_generation, d2w_generation.pad_bitmap_generate_random),
        )

        for module, generator in generators:
            with self.subTest(generator=generator.__module__):
                with tempfile.TemporaryDirectory() as temp_dir:
                    old_cwd = os.getcwd()
                    try:
                        os.chdir(temp_dir)
                        Path("pad_bitmap").mkdir()
                        np.random.seed(19)
                        with patch.object(module, "draw_pad_bitmap"), redirect_stdout(StringIO()):
                            collection = generator(cfg, "center")
                    finally:
                        os.chdir(old_cwd)

                pairs = collection["redundant_pad_block_pair_dict"]
                flattened = [block_id for pair in pairs.items() for block_id in pair]
                self.assertEqual(len(pairs), 50)
                self.assertEqual(len(flattened), len(set(flattened)))
                self.assertEqual(
                    collection["critical_pad_boundary_bitmap_row_col_block_ind"].shape,
                    (0, 2),
                )
                self.assertEqual(
                    collection["redundant_logical_to_physical_arr"].shape,
                    (800, 2),
                )
                self.assertEqual(
                    np.unique(collection["redundant_logical_to_physical_arr"]).size,
                    1600,
                )


if __name__ == "__main__":
    unittest.main()
