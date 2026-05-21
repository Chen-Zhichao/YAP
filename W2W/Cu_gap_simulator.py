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


def _std_normal_samples(std_nm, shape) -> np.ndarray:
    samples = _cu_gap_rng.standard_normal(shape, dtype=_CU_GAP_DTYPE)
    std_nm = np.float32(max(float(std_nm), 0.0))
    if std_nm != 1.0:
        samples *= std_nm
    return samples


def _block_indices_from_flat_indices(
    flat_indices: np.ndarray,
    pad_arr_col: int,
    block_size_r: int,
    block_size_c: int,
) -> np.ndarray:
    flat_indices = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
    block_size_r = max(1, int(block_size_r))
    block_size_c = max(1, int(block_size_c))
    pad_arr_col = int(pad_arr_col)
    pad_rows = flat_indices // pad_arr_col
    pad_cols = flat_indices - pad_rows * pad_arr_col
    block_r = pad_rows // block_size_r
    block_c = pad_cols // block_size_c
    block_indices = block_r * int(np.ceil(pad_arr_col / block_size_c)) + block_c
    if block_indices.size and block_indices.max() <= np.iinfo(np.int32).max:
        return block_indices.astype(np.int32, copy=False)
    return block_indices


def _dish_components(cfg, side: str) -> tuple[float, float, float, float]:
    side = side.upper()
    return (
        _cfg_float(cfg, f"{side}_DISH_MEAN_nm", 0.0),
        _cfg_float(cfg, f"{side}_DISH_STD_L_nm", 0.0),
        _cfg_float(cfg, f"{side}_DISH_STD_T_nm", 0.0),
        _cfg_float(cfg, f"{side}_DISH_STD_E_nm", 0.0),
    )


def _correlated_dish_samples(
    *,
    mean_nm: float,
    sigma_L_nm: float,
    sigma_T_nm: float,
    sigma_E_nm: float,
    valid_pad_mask_flat: np.ndarray,
    pad_arr_row: int,
    pad_arr_col: int,
    block_size_r: int,
    block_size_c: int,
    num_blocks: int,
    chunk_rows: int,
) -> np.ndarray:
    num_pads = int(np.count_nonzero(valid_pad_mask_flat))
    samples = np.full(num_pads, np.float32(mean_nm), dtype=_CU_GAP_DTYPE)

    if sigma_L_nm > 0.0:
        samples += _std_normal_samples(sigma_L_nm, ())
    if sigma_T_nm > 0.0 and num_blocks > 0:
        block_offsets = _std_normal_samples(sigma_T_nm, int(num_blocks))
        out_pos = 0
        pad_arr_row = int(pad_arr_row)
        pad_arr_col = int(pad_arr_col)
        chunk_rows = max(1, int(chunk_rows))
        for row0 in range(0, pad_arr_row, chunk_rows):
            row1 = min(pad_arr_row, row0 + chunk_rows)
            start = row0 * pad_arr_col
            stop = row1 * pad_arr_col
            chunk_mask = valid_pad_mask_flat[start:stop]
            n_valid = int(np.count_nonzero(chunk_mask))
            if n_valid == 0:
                continue

            rel_idx = np.flatnonzero(chunk_mask)
            flat_idx = start + rel_idx
            block_idx = _block_indices_from_flat_indices(
                flat_idx,
                pad_arr_col,
                block_size_r,
                block_size_c,
            )
            samples[out_pos:out_pos + n_valid] += block_offsets[block_idx]
            out_pos += n_valid
    if sigma_E_nm > 0.0:
        samples += _std_normal_samples(sigma_E_nm, num_pads)
    return samples


def clear_Cu_gap_pool() -> None:
    """Reset the internal Cu-gap RNG state."""
    global _cu_gap_rng
    _cu_gap_rng = np.random.default_rng()


def Cu_gap_correlated_simulator(
    *,
    cfg,
    valid_pad_mask_flat,
) -> tuple[np.ndarray, np.ndarray]:
    valid_pad_mask_flat = np.asarray(valid_pad_mask_flat, dtype=bool).reshape(-1)
    pad_arr_row = int(cfg.PAD_ARR_ROW)
    pad_arr_col = int(cfg.PAD_ARR_COL)
    if valid_pad_mask_flat.size != pad_arr_row * pad_arr_col:
        raise ValueError("valid_pad_mask_flat size does not match PAD_ARR_ROW * PAD_ARR_COL.")

    block_size_r = max(
        1,
        int(round(_cfg_float(cfg, "TL_um", 200.0) / _cfg_float(cfg, "PITCH_r_um", 1.0))),
    )
    block_size_c = max(
        1,
        int(round(_cfg_float(cfg, "TL_um", 200.0) / _cfg_float(cfg, "PITCH_c_um", 1.0))),
    )
    num_block_rows = int(np.ceil(pad_arr_row / block_size_r))
    num_block_cols = int(np.ceil(pad_arr_col / block_size_c))
    num_blocks = num_block_rows * num_block_cols
    target_chunk_cells = int(_cfg_float(cfg, "CU_GAP_SIM_CHUNK_CELLS", 2_000_000))
    chunk_rows = max(1, target_chunk_cells // max(1, pad_arr_col))

    top_mean, top_L, top_T, top_E = _dish_components(cfg, "TOP")
    bot_mean, bot_L, bot_T, bot_E = _dish_components(cfg, "BOT")
    top_dish = _correlated_dish_samples(
        mean_nm=top_mean,
        sigma_L_nm=top_L,
        sigma_T_nm=top_T,
        sigma_E_nm=top_E,
        valid_pad_mask_flat=valid_pad_mask_flat,
        pad_arr_row=pad_arr_row,
        pad_arr_col=pad_arr_col,
        block_size_r=block_size_r,
        block_size_c=block_size_c,
        num_blocks=num_blocks,
        chunk_rows=chunk_rows,
    )
    bot_dish = _correlated_dish_samples(
        mean_nm=bot_mean,
        sigma_L_nm=bot_L,
        sigma_T_nm=bot_T,
        sigma_E_nm=bot_E,
        valid_pad_mask_flat=valid_pad_mask_flat,
        pad_arr_row=pad_arr_row,
        pad_arr_col=pad_arr_col,
        block_size_r=block_size_r,
        block_size_c=block_size_c,
        num_blocks=num_blocks,
        chunk_rows=chunk_rows,
    )
    return top_dish, bot_dish


def Cu_gap_simulator(
    TOP_DISH_MEAN_nm,
    TOP_DISH_STD_nm,
    BOT_DISH_MEAN_nm,
    BOT_DISH_STD_nm,
    num_pads,
) -> tuple[np.ndarray, np.ndarray]:
    top_dish = np.random.normal(TOP_DISH_MEAN_nm, TOP_DISH_STD_nm, num_pads)
    bot_dish = np.random.normal(BOT_DISH_MEAN_nm, BOT_DISH_STD_nm, num_pads)
    return top_dish, bot_dish
