#!/usr/bin/env python3
"""Run one compact, paper-parameter YAP+ analytical model case.

The published experiments use up to 1.1 billion pads per die at 0.3 um pitch.
The analytical model consumes block-level masks, so this driver constructs those
masks directly instead of allocating the unnecessary full-resolution pad bitmap.
The yield equations themselves are imported from the latest YAP+ source tree.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
REPRODUCTION_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("w2w", "d2w"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overrides", default="{}", help="JSON configuration overrides")
    parser.add_argument(
        "--layout",
        choices=("full", "redundant", "sparse", "peripheral", "centralized", "shared20"),
        default="full",
    )
    parser.add_argument(
        "--replica-spacing-um",
        type=float,
        default=None,
        help="Dedicated redundant-block spacing; valid only for --layout redundant",
    )
    parser.add_argument(
        "--redundant-placement",
        choices=("perfect-matching", "legacy-greedy"),
        default="perfect-matching",
        help="Block-pair placement used by --layout redundant",
    )
    parser.add_argument(
        "--layout-seed",
        type=int,
        default=20260120,
        help="Seed for randomized redundant-block placement",
    )
    parser.add_argument(
        "--redundant-logical-pad-ratio",
        type=float,
        default=0.5,
        help="Logical/physical redundant-pad ratio for --layout redundant",
    )
    parser.add_argument(
        "--defect-only",
        action="store_true",
        help="Skip overlay and Cu-recess calculations",
    )
    return parser.parse_args()


def import_side(side: str):
    side_dir = ROOT / side.upper()
    sys.path.insert(0, str(side_dir))
    if side == "w2w":
        from Cu_expansion_yield_calculator import Cu_expansion_yield_calculator
        from defect_yield_calculator import defect_yield_calculator
        from overlay_yield_calculator import overlay_yield_calculator
        from utils.util import configure_random_seed, update_config_items

        return SimpleNamespace(
            cu=Cu_expansion_yield_calculator,
            defect=defect_yield_calculator,
            overlay=overlay_yield_calculator,
            seed=configure_random_seed,
            update=update_config_items,
        )

    from Cu_expansion_yield_calculator import Cu_expansion_yield_calculator
    from defect_yield_calculator import defect_yield_calculator
    from overlay_yield_calculator import overlay_yield_calculator
    from utils.util import configure_random_seed, update_config_items

    return SimpleNamespace(
        cu=Cu_expansion_yield_calculator,
        defect=defect_yield_calculator,
        overlay=overlay_yield_calculator,
        seed=configure_random_seed,
        update=update_config_items,
    )


def paper_config(side: str, overrides: dict, api) -> object:
    path = REPRODUCTION_DIR / "configs" / f"paper_{side}.yaml"
    section = f"{side}_modeling"
    cfg = OmegaConf.load(path)[section]
    for key, value in overrides.items():
        cfg[key] = value
    api.seed(cfg, announce=False)
    api.update(cfg=cfg, mode=section)
    return cfg


def _perfect_block_pairs(height: int, width: int, distance_blocks: float) -> list[tuple[int, int]]:
    """Find a seeded-random matching at the legacy generator's distance band."""
    rng = np.random.default_rng(20260120 + int(round(distance_blocks * 1000)))
    nodes = [(r, c) for r in range(height) for c in range(width)]
    black = [node for node in nodes if (node[0] + node[1]) % 2 == 0]
    rng.shuffle(black)
    white_index = {
        node: index
        for index, node in enumerate(node for node in nodes if (node[0] + node[1]) % 2)
    }
    offsets: list[tuple[int, int]] = []
    radius = int(math.ceil(distance_blocks + 0.5))
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            norm = math.hypot(dr, dc)
            if distance_blocks - 0.1 < norm <= distance_blocks + 0.5:
                offsets.append((dr, dc))
    rng.shuffle(offsets)
    adjacency: list[list[int]] = []
    for r, c in black:
        candidates = []
        for dr, dc in offsets:
            candidate = (r + dr, c + dc)
            if candidate in white_index:
                candidates.append(white_index[candidate])
        rng.shuffle(candidates)
        adjacency.append(candidates)

    # Hopcroft-Karp maximum matching.
    n_left = len(black)
    n_right = len(white_index)
    left_match = [-1] * n_left
    right_match = [-1] * n_right
    distance = [0] * n_left
    while True:
        queue = []
        for left in range(n_left):
            if left_match[left] < 0:
                distance[left] = 0
                queue.append(left)
            else:
                distance[left] = -1
        found = False
        cursor = 0
        while cursor < len(queue):
            left = queue[cursor]
            cursor += 1
            for right in adjacency[left]:
                mate = right_match[right]
                if mate < 0:
                    found = True
                elif distance[mate] < 0:
                    distance[mate] = distance[left] + 1
                    queue.append(mate)

        if not found:
            break

        def augment(left: int) -> bool:
            for right in adjacency[left]:
                mate = right_match[right]
                if mate < 0 or (
                    distance[mate] == distance[left] + 1 and augment(mate)
                ):
                    left_match[left] = right
                    right_match[right] = left
                    return True
            distance[left] = -1
            return False

        for left in range(n_left):
            if left_match[left] < 0:
                augment(left)

    white = [None] * n_right
    for node, index in white_index.items():
        white[index] = node
    return [
        (black[left][0] * width + black[left][1], white[right][0] * width + white[right][1])
        for left, right in enumerate(left_match)
        if right >= 0
    ]


def _legacy_center_redundant_order(height: int, width: int) -> np.ndarray:
    """Reproduce assign_pad_blocks('center') for a 100%-redundant grid."""
    ids = np.arange(height * width)
    rows = ids // width
    cols = ids % width
    rings = np.minimum.reduce([rows, cols, height - 1 - rows, width - 1 - cols])
    mesh = (rows % 3 == 0) | (cols % 3 == 0)
    mesh_ids = ids[mesh]
    gaps = np.setdiff1d(ids, mesh_ids)
    gaps = gaps[np.argsort(-rings[gaps])]
    return np.concatenate((mesh_ids, gaps))


def _legacy_greedy_block_pairs(
    height: int,
    width: int,
    distance_blocks: float,
    target_pairs: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Reproduce the December-2025 randomized pairing, including its ID bug."""
    block_order = _legacy_center_redundant_order(height, width)
    local_index = {int(block_id): index for index, block_id in enumerate(block_order)}
    radius = int(math.ceil(distance_blocks + 0.5))
    adjacency: dict[int, list[int]] = {}
    for main_raw in block_order:
        main = int(main_raw)
        row, col = divmod(main, width)
        candidates: list[tuple[int, float]] = []
        for other_row in range(max(0, row - radius), min(height, row + radius + 1)):
            for other_col in range(max(0, col - radius), min(width, col + radius + 1)):
                distance = math.hypot(other_row - row, other_col - col)
                if distance_blocks - 0.1 < distance <= distance_blocks + 0.5:
                    candidates.append(
                        (local_index[other_row * width + other_col], distance)
                    )
        candidates.sort(key=lambda item: (item[1], item[0]))
        adjacency[main] = [index for index, _ in candidates]

    # Historical source checked KDTree-local neighbor_index against sets whose
    # members were global block IDs, then mapped it to the actual copy ID.
    used = np.zeros(height * width, dtype=bool)
    rng = np.random.RandomState(seed)
    pairs: list[tuple[int, int]] = []
    for main_raw in rng.permutation(block_order):
        main = int(main_raw)
        if used[main]:
            continue
        for neighbor_index in adjacency[main]:
            if not used[neighbor_index]:
                copy = int(block_order[neighbor_index])
                used[main] = True
                used[copy] = True
                pairs.append((main, copy))
                break
        if len(pairs) >= target_pairs:
            break
    if len(pairs) < target_pairs:
        raise ValueError(
            f"legacy greedy placement found {len(pairs)} of {target_pairs} requested pairs"
        )
    return pairs


def _paper_layout_blocks(
    height: int, width: int, layout: str, num_critical: int
) -> np.ndarray:
    """Block-level equivalent of the repository's Fig. 17 layout assignment."""
    ids = np.arange(height * width)
    rows = ids // width
    cols = ids % width
    rings = np.minimum.reduce([rows, cols, height - 1 - rows, width - 1 - cols])

    if layout == "peripheral":
        return np.argsort(rings)[:num_critical]
    if layout == "centralized":
        return np.argsort(-rings)[:num_critical]
    if layout == "sparse":
        stride = 3
        while stride > 1:
            mesh = ids[(rows % stride == 0) & (cols % stride == 0)]
            if len(mesh) >= num_critical:
                break
            stride -= 1
        center_r, center_c = (height - 1) / 2, (width - 1) / 2
        distance = np.abs(rows[mesh] - center_r) + np.abs(cols[mesh] - center_c)
        ordered = mesh[np.argsort(distance)]
        sample = np.linspace(0, len(ordered) - 1, num_critical, dtype=int)
        return ordered[sample]
    raise ValueError(f"unsupported paper layout: {layout}")


def _paper_redundant_blocks(
    height: int,
    width: int,
    layout: str,
    critical_ids: np.ndarray,
    num_redundant: int,
) -> np.ndarray:
    """Match assign_pad_blocks() for the non-Full Figure 17 layouts."""
    ids = np.arange(height * width)
    rows = ids // width
    cols = ids % width
    rings = np.minimum.reduce([rows, cols, height - 1 - rows, width - 1 - cols])
    boundary = rings == 0 if layout == "peripheral" else np.zeros_like(ids, dtype=bool)
    mesh = ((rows % 3 == 0) | (cols % 3 == 0)) & ~boundary
    candidates = np.setdiff1d(ids[mesh], critical_ids)
    center_r, center_c = (height - 1) / 2, (width - 1) / 2
    manhattan = np.abs(rows - center_r) + np.abs(cols - center_c)
    if len(candidates) >= num_redundant:
        return candidates[np.argsort(manhattan[candidates])][
            :num_redundant
        ]

    remaining = np.setdiff1d(ids, np.concatenate((critical_ids, candidates)))
    if layout == "peripheral":
        remaining = remaining[np.argsort(rings[remaining])]
    elif layout == "centralized":
        remaining = remaining[np.argsort(-rings[remaining])]
    else:
        remaining = remaining[np.argsort(manhattan[remaining])]
    return np.concatenate((candidates, remaining[: num_redundant - len(candidates)]))


def compact_bitmap(
    cfg,
    layout: str,
    replica_spacing_um: float | None,
    redundant_placement: str = "perfect-matching",
    layout_seed: int = 20260120,
    redundant_logical_pad_ratio: float = 0.5,
) -> dict:
    block_size = int(cfg.pad_block_size)
    # Match upstream downsample_bitmap(): incomplete edge blocks are trimmed
    # before block pooling.  Using ceil here adds an artificial full block at
    # the right/bottom edge and overestimates particle-defect critical area.
    height = cfg.PAD_ARR_ROW // block_size
    width = cfg.PAD_ARR_COL // block_size
    if height < 1 or width < 1:
        raise ValueError(
            "pad_block_dim_um is larger than the active pad array; "
            "the upstream block downsampler would produce an empty bitmap"
        )
    empty_pairs = np.zeros((0, height, width), dtype=bool)

    if layout == "full":
        critical = np.ones((height, width), dtype=bool)
        pair_dict: dict[int, int] = {}
        main = empty_pairs
        copy = empty_pairs.copy()
        cfg.critical_pad_ratio = 1.0
        cfg.redundant_pad_ratio = 0.0
        cfg.redundant_logical_pad_ratio = 0.0
        num_critical_pads = int(cfg.PAD_ARR_ROW * cfg.PAD_ARR_COL)
        num_redundant_pads = 0
        num_redundant_logical_pads = 0
        physical_ratios = (1.0, 0.0, 0.0)
        is_same_block = False
    elif layout in ("sparse", "peripheral", "centralized"):
        # Figure 17 specifies 20% critical, 50% redundant, and 30% dummy pads.
        # The paper does not disclose what fraction of the redundant physical
        # pads forms logical main/copy groups.  That ratio is supplied by the
        # selected Fig. 17 profile.  At 0.3 um pitch the configured 80-pitch
        # main/copy separation fits within one model block, so both copies are
        # represented by the same block plane.
        critical = np.zeros((height, width), dtype=bool)
        num_critical_blocks = int(math.ceil(0.20 * height * width))
        critical_ids = _paper_layout_blocks(
            height, width, layout, num_critical_blocks
        )
        critical.flat[critical_ids] = True
        num_redundant_blocks = min(
            int(math.ceil(0.50 * height * width)), height * width - num_critical_blocks
        )
        redundant_ids = _paper_redundant_blocks(
            height, width, layout, critical_ids, num_redundant_blocks
        )
        pads_per_block = block_size * block_size
        # The paper states the physical 20/50/30 critical/redundant/dummy
        # split, but it does not state what fraction of the redundant pads is
        # assigned to logical main/copy groups.  Keep that independent input
        # profile-controlled instead of silently hard-coding the historical
        # 0.30 value here.
        redundant_logical_ratio = float(cfg.redundant_logical_pad_ratio)
        requested_logical_pads = int(
            0.50 * cfg.PAD_ARR_ROW * cfg.PAD_ARR_COL * redundant_logical_ratio
        )
        num_pair_blocks = min(
            len(redundant_ids),
            int(math.ceil(2 * requested_logical_pads / pads_per_block)),
        )
        rng = np.random.default_rng(20260120)
        selected = rng.permutation(redundant_ids)[:num_pair_blocks]
        pair_dict = {int(block_id): int(block_id) for block_id in selected}
        main = np.zeros((num_pair_blocks, height, width), dtype=bool)
        copy = np.zeros_like(main)
        for index, block_id in enumerate(selected):
            main[index, block_id // width, block_id % width] = True
            copy[index, block_id // width, block_id % width] = True
        cfg.critical_pad_ratio = 0.20
        cfg.redundant_pad_ratio = 0.50
        cfg.redundant_logical_pad_ratio = redundant_logical_ratio
        num_critical_pads = int(0.20 * cfg.PAD_ARR_ROW * cfg.PAD_ARR_COL)
        num_redundant_pads = int(0.50 * cfg.PAD_ARR_ROW * cfg.PAD_ARR_COL)
        num_redundant_logical_pads = requested_logical_pads
        physical_ratios = (0.20, 0.50, 0.30)
        # Keeping this false avoids a known 3-D bounds bug in the latest D2W
        # calculator. Pairwise critical-area dilation still uses both planes.
        is_same_block = False
    elif layout == "shared20":
        # Figure 19's 20:1 strategy places each replica beside its 20 mains.
        # Defects are much larger than this microscopic separation, so at the
        # 200 um modeling resolution its defect-sensitive area is the full die,
        # the same limiting value as the no-redundancy case.
        critical = np.ones((height, width), dtype=bool)
        pair_dict = {}
        main = empty_pairs
        copy = empty_pairs.copy()
        cfg.critical_pad_ratio = 0.0
        cfg.redundant_pad_ratio = 1.0
        cfg.redundant_logical_pad_ratio = 20.0 / 21.0
        # The critical surrogate encodes defect-sensitive area only.
        num_critical_pads = int(cfg.PAD_ARR_ROW * cfg.PAD_ARR_COL)
        num_redundant_pads = int(cfg.PAD_ARR_ROW * cfg.PAD_ARR_COL)
        num_redundant_logical_pads = int(num_redundant_pads * 20.0 / 21.0)
        physical_ratios = (0.0, 1.0, 0.0)
        is_same_block = False
    else:
        if not replica_spacing_um:
            raise ValueError("positive --replica-spacing-um is required for redundant layout")
        if not 0 < redundant_logical_pad_ratio <= 0.5:
            raise ValueError("redundant logical pad ratio must be in (0, 0.5]")
        critical = np.zeros((height, width), dtype=bool)
        distance_blocks = replica_spacing_um / float(cfg.pad_block_dim_um)
        target_pairs = int(math.ceil(height * width * redundant_logical_pad_ratio))
        if redundant_placement == "legacy-greedy":
            pairs = _legacy_greedy_block_pairs(
                height, width, distance_blocks, target_pairs, layout_seed
            )
        else:
            pairs = _perfect_block_pairs(height, width, distance_blocks)[:target_pairs]
        pair_dict = dict(pairs)
        main = np.zeros((len(pairs), height, width), dtype=bool)
        copy = np.zeros_like(main)
        for index, (main_id, copy_id) in enumerate(pairs):
            main[index, main_id // width, main_id % width] = True
            copy[index, copy_id // width, copy_id % width] = True
        cfg.critical_pad_ratio = 0.0
        cfg.redundant_pad_ratio = 1.0
        cfg.redundant_logical_pad_ratio = redundant_logical_pad_ratio
        num_critical_pads = 0
        pads_per_block = block_size * block_size
        num_redundant_pads = 2 * len(pairs) * pads_per_block
        num_redundant_logical_pads = len(pairs) * pads_per_block
        physical_ratios = (0.0, 1.0, 0.0)
        is_same_block = False

    return {
        "CRITICAL_PAD_BLOCK_BITMAP": critical,
        "REDUNDANT_MAIN_PAD_BLOCK_BITMAP": main,
        "REDUNDANT_COPY_PAD_BLOCK_BITMAP": copy,
        "is_redundant_copy_same_block": is_same_block,
        "pad_block_size": block_size,
        "critical_pad_ratio": float(cfg.critical_pad_ratio),
        "num_critical_pads": num_critical_pads,
        "num_redundant_pads": num_redundant_pads,
        "num_redundant_logical_pads": num_redundant_logical_pads,
        "redundant_logical_pad_copy": 2,
        "redundant_pad_block_pair_dict": pair_dict,
        "paired_block_fraction": 2 * len(pair_dict) / (height * width),
        "physical_critical_ratio": physical_ratios[0],
        "physical_redundant_ratio": physical_ratios[1],
        "physical_dummy_ratio": physical_ratios[2],
    }


def critical_boundary(cfg, bitmap: dict) -> np.ndarray:
    # The upstream generator determines overlay boundaries from the full pad
    # bitmap before trimming incomplete edge blocks for defect dilation.
    if bitmap["physical_critical_ratio"] == 1.0:
        return die_corners(cfg)
    occupied = np.argwhere(bitmap["CRITICAL_PAD_BLOCK_BITMAP"])
    if len(occupied) == 0:
        return die_corners(cfg)
    block_um = bitmap["pad_block_size"] * cfg.PITCH_um
    xmin = -cfg.PAD_ARR_W_um / 2 + occupied[:, 1].min() * block_um
    xmax = min(
        cfg.PAD_ARR_W_um / 2,
        -cfg.PAD_ARR_W_um / 2 + (occupied[:, 1].max() + 1) * block_um,
    )
    ymax = cfg.PAD_ARR_L_um / 2 - occupied[:, 0].min() * block_um
    ymin = max(
        -cfg.PAD_ARR_L_um / 2,
        cfg.PAD_ARR_L_um / 2 - (occupied[:, 0].max() + 1) * block_um,
    )
    return np.asarray([[xmin, ymax], [xmax, ymax], [xmin, ymin], [xmax, ymin]])


def die_corners(cfg, center=(0.0, 0.0)) -> np.ndarray:
    cx, cy = center
    half_w = cfg.PAD_ARR_W_um / 2
    half_l = cfg.PAD_ARR_L_um / 2
    return np.array(
        [
            [cx - half_w, cy + half_l],
            [cx + half_w, cy + half_l],
            [cx - half_w, cy - half_l],
            [cx + half_w, cy - half_l],
        ],
        dtype=float,
    )


def make_die(cfg, bitmap: dict) -> SimpleNamespace:
    corners = die_corners(cfg)
    boundary = critical_boundary(cfg, bitmap)
    return SimpleNamespace(
        DIE_W_um=cfg.DIE_W_um,
        DIE_L_um=cfg.DIE_L_um,
        die_center=np.zeros(2),
        pad_array_box=corners,
        ovl_critical_pad_boundary_coords=boundary,
        pad_array=None,
        pad_yield_map={},
        die_yield={},
        glb_pad_yield_min_max_dict={},
    )


def make_wafer(cfg, bitmap: dict) -> SimpleNamespace:
    row_count = int(2 * cfg.WAF_R_um // (cfg.DIE_L_um + cfg.dice_width) + 1)
    col_count = int(2 * cfg.WAF_R_um // (cfg.DIE_W_um + cfg.dice_width) + 1)
    dies = []
    for row in range(row_count):
        for col in range(col_count):
            cx = (
                -col_count * (cfg.DIE_W_um + cfg.dice_width) / 2
                + (cfg.DIE_W_um + cfg.dice_width) / 2
                + col * (cfg.DIE_W_um + cfg.dice_width)
            )
            cy = (
                row_count * (cfg.DIE_L_um + cfg.dice_width) / 2
                - (cfg.DIE_L_um + cfg.dice_width) / 2
                - row * (cfg.DIE_L_um + cfg.dice_width)
            )
            vertices = np.array(
                [
                    [cx - cfg.DIE_W_um / 2, cy + cfg.DIE_L_um / 2],
                    [cx + cfg.DIE_W_um / 2, cy + cfg.DIE_L_um / 2],
                    [cx - cfg.DIE_W_um / 2, cy - cfg.DIE_L_um / 2],
                    [cx + cfg.DIE_W_um / 2, cy - cfg.DIE_L_um / 2],
                ]
            )
            if any(np.hypot(vertex[0], vertex[1]) >= cfg.WAF_R_um for vertex in vertices):
                continue
            corners = die_corners(cfg, (cx, cy))
            boundary = critical_boundary(cfg, bitmap) + np.asarray([cx, cy])
            dies.append(
                SimpleNamespace(
                    die_center=np.array([cx, cy]),
                    pad_array_box=corners,
                    ovl_critical_pad_boundary_coords=boundary,
                    pad_yield_map={},
                )
            )
    return SimpleNamespace(
        die_list=dies,
        base_pad_coords=None,
        glb_pad_yield_min_max_dict={},
    )


def common_overlay_kwargs(cfg) -> dict:
    kwargs = {
        "PAD_ARR_ROW": cfg.PAD_ARR_ROW,
        "PAD_ARR_COL": cfg.PAD_ARR_COL,
        "PAD_TOP_R_um": cfg.PAD_TOP_R_um,
        "PAD_BOT_R_um": cfg.PAD_BOT_R_um,
        "PITCH_um": cfg.PITCH_um,
        "num_samples": cfg.num_samples,
        "CONTACT_AREA_CONSTRAINT": cfg.CONTACT_AREA_CONSTRAINT,
        "CRITICAL_DIST_CONSTRAINT": cfg.CRITICAL_DIST_CONSTRAINT,
        "SYSTEM_MAGNIFICATION_MEAN_ppm": cfg.SYSTEM_MAGNIFICATION_MEAN_ppm,
        "SYSTEM_MAGNIFICATION_STD_ppm": cfg.SYSTEM_MAGNIFICATION_STD_ppm,
        "SYSTEM_ROTATION_MEAN_rad": cfg.SYSTEM_ROTATION_MEAN_rad,
        "SYSTEM_ROTATION_STD_rad": cfg.SYSTEM_ROTATION_STD_rad,
        "SYSTEM_TRANSLATION_X_MEAN_um": cfg.SYSTEM_TRANSLATION_X_MEAN_um,
        "SYSTEM_TRANSLATION_X_STD_um": cfg.SYSTEM_TRANSLATION_X_STD_um,
        "SYSTEM_TRANSLATION_Y_MEAN_um": cfg.SYSTEM_TRANSLATION_Y_MEAN_um,
        "SYSTEM_TRANSLATION_Y_STD_um": cfg.SYSTEM_TRANSLATION_Y_STD_um,
        "RANDOM_MISALIGNMENT_MEAN_um": cfg.RANDOM_MISALIGNMENT_MEAN_um,
        "RANDOM_MISALIGNMENT_STD_um": cfg.RANDOM_MISALIGNMENT_STD_um,
        "redundant_flag": cfg.redundant_flag,
        "pad_yield_flag": False,
        "pad_yield_map_sub_factor": cfg.pad_yield_map_sub_factor,
    }
    if hasattr(cfg, "scale_systematic_distortion_from_wafer"):
        kwargs.update({
            "scale_systematic_distortion_from_wafer": bool(
                cfg.scale_systematic_distortion_from_wafer
            ),
            "WAF_R_um": float(cfg.WAF_R_um),
        })
    return kwargs


def common_defect_kwargs(cfg, bitmap: dict) -> dict:
    return {
        "cfg": cfg,
        "D0": cfg.D0,
        "t_0": cfg.t_0,
        "z": cfg.z,
        "k_r": cfg.k_r,
        "k_r0": cfg.k_r0,
        "k_n": cfg.k_n,
        "k_S": cfg.k_S,
        "k_L": cfg.k_L,
        "PAD_TOP_R_um": cfg.PAD_TOP_R_um,
        "PITCH_um": cfg.PITCH_um,
        "PAD_ARR_ROW": cfg.PAD_ARR_ROW,
        "PAD_ARR_COL": cfg.PAD_ARR_COL,
        "PAD_ARR_W_um": cfg.PAD_ARR_W_um,
        "PAD_ARR_L_um": cfg.PAD_ARR_L_um,
        "VOID_SHAPE": cfg.VOID_SHAPE,
        "pad_bitmap_collection": bitmap,
        "pad_yield_flag": False,
        "pad_yield_map_sub_factor": cfg.pad_yield_map_sub_factor,
    }


def common_cu_kwargs(cfg, bitmap: dict) -> dict:
    return {
        "cfg": cfg,
        "TOP_DISH_MEAN_nm": cfg.TOP_DISH_MEAN_nm,
        "TOP_DISH_STD_nm": cfg.TOP_DISH_STD_nm,
        "BOT_DISH_MEAN_nm": cfg.BOT_DISH_MEAN_nm,
        "BOT_DISH_STD_nm": cfg.BOT_DISH_STD_nm,
        "k_et": cfg.k_et,
        "k_eb": cfg.k_eb,
        "T_R": cfg.T_R,
        "T_anl": cfg.T_anl,
        "pad_bitmap_collection": bitmap,
        "pad_yield_flag": False,
    }


def calculate(side: str, cfg, bitmap: dict, api, defect_only: bool) -> dict:
    if side == "w2w":
        wafer = make_wafer(cfg, bitmap)
        defect = api.defect(
            wafer=wafer,
            WAF_R_um=cfg.WAF_R_um,
            num_die=len(wafer.die_list),
            dice_width=cfg.dice_width,
            **common_defect_kwargs(cfg, bitmap),
        )
        if defect_only:
            return {"Y_df": float(defect), "num_dies_on_wafer": len(wafer.die_list)}
        overlay = api.overlay(cfg=cfg, wafer=wafer, **common_overlay_kwargs(cfg))
        cu = api.cu(wafer=wafer, **common_cu_kwargs(cfg, bitmap))
        return {
            "Y_ovl": float(overlay),
            "Y_cr": float(cu),
            "Y_df": float(defect),
            "Y_bond": float(overlay * cu * defect),
            "num_dies_on_wafer": len(wafer.die_list),
        }

    die = make_die(cfg, bitmap)
    defect, _ = api.defect(
        eff_DIE_R=cfg.eff_DIE_R,
        die=die,
        **common_defect_kwargs(cfg, bitmap),
    )
    if defect_only:
        return {"Y_df": float(defect)}
    overlay, _ = api.overlay(die=die, **common_overlay_kwargs(cfg))
    cu, _ = api.cu(die=die, **common_cu_kwargs(cfg, bitmap))
    return {
        "Y_ovl": float(overlay),
        "Y_cr": float(cu),
        "Y_df": float(defect),
        "Y_bond": float(overlay * cu * defect),
    }


def main() -> None:
    args = parse_args()
    # Preserve CLI-relative output semantics before changing into the isolated
    # per-case work directory used by the upstream calculators.
    args.output = args.output.resolve()
    overrides = json.loads(args.overrides)
    work_dir = REPRODUCTION_DIR / "work" / args.case
    (work_dir / "pad_bitmap").mkdir(parents=True, exist_ok=True)
    # Upstream caches the density-normalized dilation integral under a fixed
    # filename without encoding layout, pitch, die size, or code version. A
    # reproduction rerun must invalidate that cache or changed cases can
    # silently reuse stale geometry.
    dilation_cache = work_dir / "pad_bitmap" / "avg_num_defects_per_unit_area.npy"
    if dilation_cache.exists():
        dilation_cache.unlink()
    os.chdir(work_dir)
    api = import_side(args.side)
    cfg = paper_config(args.side, overrides, api)
    bitmap = compact_bitmap(
        cfg,
        args.layout,
        args.replica_spacing_um,
        args.redundant_placement,
        args.layout_seed,
        args.redundant_logical_pad_ratio,
    )
    result = calculate(args.side, cfg, bitmap, api, args.defect_only)
    die_area_mm2 = float(cfg.DIE_W_um * cfg.DIE_L_um / 1.0e6)
    if args.side == "d2w" and not args.defect_only:
        result["Y_sys_1000mm2"] = float(result["Y_bond"] ** (1000.0 / die_area_mm2))
    result.update(
        {
            "case": args.case,
            "side": args.side,
            "layout": args.layout,
            "pitch_um": float(cfg.PITCH_um),
            "die_area_mm2": die_area_mm2,
            "D0_per_um2": float(cfg.D0),
            "particle_density_per_cm2": float(cfg.D0 * 1.0e8),
            "bottom_pad_diameter_um": float(2 * cfg.PAD_BOT_R_um),
            "top_pad_diameter_um": float(2 * cfg.PAD_TOP_R_um),
            "magnification_mean_ppm": float(cfg.SYSTEM_MAGNIFICATION_MEAN_ppm * 1.0e6),
            "magnification_std_ppm": float(cfg.SYSTEM_MAGNIFICATION_STD_ppm * 1.0e6),
            "replica_spacing_um": args.replica_spacing_um,
            "redundant_placement": args.redundant_placement,
            "layout_seed": args.layout_seed,
            "paired_block_fraction": float(bitmap["paired_block_fraction"]),
            "unique_paired_block_fraction": float(
                len(
                    set(bitmap["redundant_pad_block_pair_dict"].keys())
                    | set(bitmap["redundant_pad_block_pair_dict"].values())
                )
                / (bitmap["CRITICAL_PAD_BLOCK_BITMAP"].shape[0]
                   * bitmap["CRITICAL_PAD_BLOCK_BITMAP"].shape[1])
            ),
            "physical_critical_ratio": bitmap["physical_critical_ratio"],
            "physical_redundant_ratio": bitmap["physical_redundant_ratio"],
            "physical_dummy_ratio": bitmap["physical_dummy_ratio"],
            "configured_critical_pad_ratio": float(cfg.critical_pad_ratio),
            "configured_redundant_pad_ratio": float(cfg.redundant_pad_ratio),
            "configured_redundant_logical_pad_ratio": float(cfg.redundant_logical_pad_ratio),
            "pad_block_dim_um": float(cfg.pad_block_dim_um),
            "pad_block_size_pads": int(bitmap["pad_block_size"]),
            "effective_pad_block_dim_um": float(bitmap["pad_block_size"] * cfg.PITCH_um),
            "pad_block_grid_rows": int(bitmap["CRITICAL_PAD_BLOCK_BITMAP"].shape[0]),
            "pad_block_grid_cols": int(bitmap["CRITICAL_PAD_BLOCK_BITMAP"].shape[1]),
            "particle_yield_model": "Poisson exp(-Lambda); analytical run has no sampled particle count",
        }
    )
    if args.side == "d2w":
        scaling_enabled = bool(
            getattr(cfg, "scale_systematic_distortion_from_wafer", False)
        )
        result["scale_systematic_distortion_from_wafer"] = scaling_enabled
        result["wafer_to_die_distortion_scale"] = (
            float(cfg.WAF_R_um / math.hypot(cfg.DIE_W_um / 2, cfg.DIE_L_um / 2))
            if scaling_enabled else 1.0
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
