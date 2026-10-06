#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np


_CU_GAP_DTYPE = np.float32
_cu_gap_rng = np.random.default_rng()


def _cfg_float(cfg, key: str, default: float = 0.0) -> float:
    value = cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)
    if value in (None, "None"):
        return float(default)
    return float(value)


def _dish_params(cfg, side: str) -> tuple[float, float]:
    side = side.upper()
    return (
        _cfg_float(cfg, f"{side}_DISH_MEAN_nm", 0.0),
        _cfg_float(cfg, f"{side}_DISH_STD_nm", 0.0),
    )


def _iid_dish_samples(mean_nm: float, std_nm: float, num_pads: int) -> np.ndarray:
    samples = np.full(num_pads, np.float32(mean_nm), dtype=_CU_GAP_DTYPE)
    std_nm = max(float(std_nm), 0.0)
    if std_nm > 0.0 and num_pads > 0:
        samples += _cu_gap_rng.normal(
            0.0,
            np.float32(std_nm),
            int(num_pads),
        ).astype(_CU_GAP_DTYPE, copy=False)
    return samples


def clear_Cu_gap_pool() -> None:
    """Reset the internal Cu-gap RNG state."""
    global _cu_gap_rng
    _cu_gap_rng = np.random.default_rng()


def Cu_gap_simulator(
    *,
    cfg,
    valid_pad_mask_flat,
) -> tuple[np.ndarray, np.ndarray]:
    valid_pad_mask_flat = np.asarray(valid_pad_mask_flat, dtype=bool).reshape(-1)
    pad_arr_row = int(cfg.PAD_ARR_ROW)
    pad_arr_col = int(cfg.PAD_ARR_COL)
    if valid_pad_mask_flat.size != pad_arr_row * pad_arr_col:
        raise ValueError("valid_pad_mask_flat size does not match PAD_ARR_ROW * PAD_ARR_COL.")

    num_pads = int(np.count_nonzero(valid_pad_mask_flat))
    top_mean, top_std = _dish_params(cfg, "TOP")
    bot_mean, bot_std = _dish_params(cfg, "BOT")
    top_dish = _iid_dish_samples(top_mean, top_std, num_pads)
    bot_dish = _iid_dish_samples(bot_mean, bot_std, num_pads)
    return top_dish, bot_dish
