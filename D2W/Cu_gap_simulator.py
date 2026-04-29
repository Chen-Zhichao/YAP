#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Cu gap simulator for the yield model for D2W hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np


_CU_GAP_DTYPE = np.float32
_cu_gap_rng = np.random.default_rng()


def _scaled_normal_samples(mean_nm, std_nm, shape) -> np.ndarray:
    """Draw normal samples using the faster Generator path and keep them in float32."""
    samples = _cu_gap_rng.standard_normal(shape, dtype=_CU_GAP_DTYPE)
    std_nm = np.float32(max(float(std_nm), 0.0))
    mean_nm = np.float32(float(mean_nm))
    if std_nm != 1.0:
        samples *= std_nm
    if mean_nm != 0.0:
        samples += mean_nm
    return samples


def clear_Cu_gap_pool() -> None:
    """Reset the internal Cu-gap RNG state."""
    global _cu_gap_rng
    _cu_gap_rng = np.random.default_rng()


def Cu_gap_batch_simulator(
        TOP_DISH_MEAN_nm,
        TOP_DISH_STD_nm,
        BOT_DISH_MEAN_nm,
        BOT_DISH_STD_nm,
        num_samples,
        num_pads,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate Cu dishing samples for a batch of dies at once.

    The batch dimension is the first axis: (num_samples, num_pads).
    """
    sample_shape = (int(num_samples), int(num_pads))
    top_dish = _scaled_normal_samples(TOP_DISH_MEAN_nm, TOP_DISH_STD_nm, sample_shape)
    bot_dish = _scaled_normal_samples(BOT_DISH_MEAN_nm, BOT_DISH_STD_nm, sample_shape)
    return top_dish, bot_dish



def Cu_gap_simulator(
        TOP_DISH_MEAN_nm,
        TOP_DISH_STD_nm,
        BOT_DISH_MEAN_nm,
        BOT_DISH_STD_nm,
        num_pads,
    ) -> tuple[np.ndarray, np.ndarray]:
    top_dish = _scaled_normal_samples(TOP_DISH_MEAN_nm, TOP_DISH_STD_nm, int(num_pads))
    bot_dish = _scaled_normal_samples(BOT_DISH_MEAN_nm, BOT_DISH_STD_nm, int(num_pads))
    return top_dish, bot_dish
