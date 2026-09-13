#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#### Author: Zhichao Chen
#### Date: Oct 20, 2025
'''
This module generates/read the pad bitmap for the die.
'''

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from matplotlib.colors import ListedColormap
import time
from matplotlib.colors import ListedColormap, BoundaryNorm
import cv2
from scipy.spatial import KDTree
from collections import deque

def downsample_bitmap(bitmap, block_size):
    """
    Downsamples a binary bitmap by taking max value in each block.
    For binary image, this is equivalent to OR pooling.
    """
    if bitmap.ndim == 2:
        h, w = bitmap.shape
        h_new = h // block_size
        w_new = w // block_size

        # Trim to divisible shape
        bitmap = bitmap[:h_new * block_size, :w_new * block_size]

        # Reshape and apply max pooling
        bitmap_down = bitmap.reshape(h_new, block_size, w_new, block_size)
        bitmap_down = bitmap_down.max(axis=(1, 3))
    elif bitmap.ndim == 3:
        n, h, w = bitmap.shape
        h_new = h // block_size
        w_new = w // block_size

        # Trim to divisible shape
        bitmap = bitmap[:, :h_new * block_size, :w_new * block_size]

        # Reshape and apply max pooling
        bitmap_down = bitmap.reshape(n, h_new, block_size, w_new, block_size)
        bitmap_down = bitmap_down.max(axis=(2, 4))
    else:
        raise ValueError("Bitmap must be 2D or 3D.")

    return bitmap_down


def assign_pad_blocks(mode, num_pads_blocks, num_pads_block_row, num_pads_block_col,
                      num_critical_pad_blocks, num_redundant_pad_blocks, 
                      redundant_mesh_spacing=2, sparse_stride=3):
    all_blocks_idx = np.arange(num_pads_blocks)

    block_row_idx = all_blocks_idx // num_pads_block_col
    block_col_idx = all_blocks_idx % num_pads_block_col

    # Calculate the ring index for each block (0 at the outermost ring, increasing towards the center)
    ring_index = np.minimum.reduce([
        block_row_idx,
        block_col_idx,
        num_pads_block_row - 1 - block_row_idx,
        num_pads_block_col - 1 - block_col_idx
    ])

    # === Critical pad assignment ===
    if mode == 'peripheral':
        sorted_ring_blocks = np.argsort(ring_index)
        is_boundary_ring = ring_index == 0
        critical_pad_blocks = sorted_ring_blocks[:num_critical_pad_blocks]

    elif mode == 'center':
        sorted_ring_blocks = np.argsort(-ring_index)
        is_boundary_ring = np.zeros_like(ring_index, dtype=bool)
        critical_pad_blocks = sorted_ring_blocks[:num_critical_pad_blocks]

    elif mode == 'sparse':
        stride = sparse_stride
        found_enough = False

        while stride >= 1 and not found_enough:
            mesh_mask = (block_row_idx % stride == 0) & (block_col_idx % stride == 0)
            mesh_indices = all_blocks_idx[mesh_mask]

            if len(mesh_indices) >= num_critical_pad_blocks:
                found_enough = True
            else:
                stride -= 1  # 自动减小 stride 以获得更密集的 checkerboard

        if not found_enough:
            raise ValueError("Even stride=1 cannot provide enough checkerboard blocks.")

        # Step 2: 均匀采样
        center_row, center_col = (num_pads_block_row - 1) / 2, (num_pads_block_col - 1) / 2
        dist = np.abs(block_row_idx[mesh_indices] - center_row) + np.abs(block_col_idx[mesh_indices] - center_col)
        sorted_indices = mesh_indices[np.argsort(dist)]

        linspace_idx = np.linspace(0, len(sorted_indices) - 1, num_critical_pad_blocks, dtype=int)
        critical_pad_blocks = sorted_indices[linspace_idx]
        is_boundary_ring = np.zeros_like(ring_index, dtype=bool)



    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # === Redundant pad assignment ===
    redundant_mesh_mask = (
        ((block_row_idx % redundant_mesh_spacing == 0) | (block_col_idx % redundant_mesh_spacing == 0)) &
        (~is_boundary_ring)
    )

    potential_redundant_blocks = np.setdiff1d(all_blocks_idx[redundant_mesh_mask], critical_pad_blocks)

    mesh_count = len(potential_redundant_blocks)

    if mesh_count >= num_redundant_pad_blocks:
        # prioritize innermost mesh blocks
        center_row, center_col = (num_pads_block_row - 1) / 2, (num_pads_block_col - 1) / 2
        manhattan_distance = np.abs(block_row_idx - center_row) + np.abs(block_col_idx - center_col)
        sorted_redundant_blocks = potential_redundant_blocks[
            np.argsort(manhattan_distance[potential_redundant_blocks])
        ]
        redundant_pad_blocks = sorted_redundant_blocks[:num_redundant_pad_blocks]
    else:
        redundant_pad_blocks = list(potential_redundant_blocks)
        remaining_count = num_redundant_pad_blocks - mesh_count

        gap_blocks = np.setdiff1d(all_blocks_idx, np.concatenate((critical_pad_blocks, redundant_pad_blocks)))

        if mode == 'peripheral':
            gap_blocks_sorted = gap_blocks[np.argsort(ring_index[gap_blocks])]
        elif mode == 'center':
            gap_blocks_sorted = gap_blocks[np.argsort(-ring_index[gap_blocks])]
        elif mode == 'sparse':
            center_row, center_col = (num_pads_block_row - 1) / 2, (num_pads_block_col - 1) / 2
            manhattan_distance = np.abs(block_row_idx - center_row) + np.abs(block_col_idx - center_col)
            gap_blocks_sorted = gap_blocks[np.argsort(manhattan_distance[gap_blocks])]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        redundant_pad_blocks.extend(gap_blocks_sorted[:remaining_count])
        redundant_pad_blocks = np.array(redundant_pad_blocks)

    # === Dummy pads ===
    dummy_pad_blocks = np.setdiff1d(
        all_blocks_idx,
        np.concatenate((critical_pad_blocks, redundant_pad_blocks))
    )

    return critical_pad_blocks, redundant_pad_blocks, dummy_pad_blocks


def place_redundant_block_pairs(
    redundant_blocks,
    block_rows,
    block_cols,
    target_pairs,
    block_distance,
):
    """Find randomized, non-overlapping pairs in the requested distance band.

    The old greedy allocator could strand unmatched blocks.  It also mixed
    KDTree-local indices with global block IDs, which allowed physical blocks
    to be reused.  This routine builds the undirected candidate graph and uses
    Edmonds' blossom algorithm to obtain a maximum-cardinality matching.
    """
    blocks = np.asarray(redundant_blocks, dtype=int)
    if target_pairs == 0:
        return []
    if target_pairs < 0 or 2 * target_pairs > len(blocks):
        raise ValueError(
            f"Requested {target_pairs} non-overlapping pairs from {len(blocks)} blocks."
        )

    positions = np.column_stack((blocks // block_cols, blocks % block_cols))
    tree = KDTree(positions)
    adjacency = [[] for _ in range(len(blocks))]
    for node, position in enumerate(positions):
        neighbor_nodes = tree.query_ball_point(position, r=block_distance + 0.5)
        for neighbor in neighbor_nodes:
            if neighbor <= node:
                continue
            distance = np.linalg.norm(position - positions[neighbor])
            if block_distance - 0.1 < distance <= block_distance + 0.5:
                adjacency[node].append(neighbor)
                adjacency[neighbor].append(node)

    for neighbors in adjacency:
        np.random.shuffle(neighbors)
    root_order = np.random.permutation(len(blocks))
    match = [-1] * len(blocks)
    parent = [-1] * len(blocks)
    base = list(range(len(blocks)))
    used = [False] * len(blocks)
    blossom = [False] * len(blocks)

    def lowest_common_ancestor(first, second):
        visited = [False] * len(blocks)
        while True:
            first = base[first]
            visited[first] = True
            if match[first] == -1:
                break
            first = parent[match[first]]
        while True:
            second = base[second]
            if visited[second]:
                return second
            second = parent[match[second]]

    def mark_blossom_path(vertex, blossom_base, child):
        while base[vertex] != blossom_base:
            blossom[base[vertex]] = True
            blossom[base[match[vertex]]] = True
            parent[vertex] = child
            child = match[vertex]
            vertex = parent[match[vertex]]

    def find_augmenting_path(root):
        for index in range(len(blocks)):
            used[index] = False
            parent[index] = -1
            base[index] = index
        queue = deque([root])
        used[root] = True
        while queue:
            vertex = queue.popleft()
            for neighbor in adjacency[vertex]:
                if base[vertex] == base[neighbor] or match[vertex] == neighbor:
                    continue
                if neighbor == root or (
                    match[neighbor] != -1 and parent[match[neighbor]] != -1
                ):
                    common_base = lowest_common_ancestor(vertex, neighbor)
                    for index in range(len(blocks)):
                        blossom[index] = False
                    mark_blossom_path(vertex, common_base, neighbor)
                    mark_blossom_path(neighbor, common_base, vertex)
                    for index in range(len(blocks)):
                        if blossom[base[index]]:
                            base[index] = common_base
                            if not used[index]:
                                used[index] = True
                                queue.append(index)
                elif parent[neighbor] == -1:
                    parent[neighbor] = vertex
                    if match[neighbor] == -1:
                        current = neighbor
                        while current != -1:
                            previous = parent[current]
                            next_vertex = match[previous] if previous != -1 else -1
                            match[current] = previous
                            if previous != -1:
                                match[previous] = current
                            current = next_vertex
                        return True
                    neighbor = match[neighbor]
                    used[neighbor] = True
                    queue.append(neighbor)
        return False

    for root_raw in root_order:
        root = int(root_raw)
        if match[root] == -1:
            find_augmenting_path(root)

    matched_pairs = [
        (node, partner)
        for node, partner in enumerate(match)
        if partner != -1 and node < partner
    ]
    if len(matched_pairs) < target_pairs:
        raise ValueError(
            "Not enough non-overlapping pad block pairs at the requested distance "
            f"(required {target_pairs}, maximum matching {len(matched_pairs)})."
        )

    selected = np.random.permutation(len(matched_pairs))[:target_pairs]
    pairs = []
    for pair_index_raw in selected:
        first, second = matched_pairs[int(pair_index_raw)]
        main, copy = int(blocks[first]), int(blocks[second])
        if np.random.random() < 0.5:
            main, copy = copy, main
        pairs.append((main, copy))
    return pairs



def draw_pad_bitmap(bitmap_collection):
    # Draw the critical and redundant pad bitmaps in one figure (critical light red, redundant light blue, dummy light gray)
    CRITICAL_PAD_BITMAP = bitmap_collection["CRITICAL_PAD_BITMAP"]
    REDUNDANT_PAD_BITMAP = bitmap_collection["REDUNDANT_PAD_BITMAP"]
    DUMMY_PAD_BITMAP = bitmap_collection["DUMMY_PAD_BITMAP"]
    ## Use legend to show the color
    PAD_BITMAP = np.zeros_like(CRITICAL_PAD_BITMAP, dtype=int)

    PAD_BITMAP[CRITICAL_PAD_BITMAP == 1] = 1  # red
    PAD_BITMAP[REDUNDANT_PAD_BITMAP == 1] = 2  # blue
    PAD_BITMAP[DUMMY_PAD_BITMAP == 1] = 3  # gray

    plt.figure(figsize=(8, 8))
    cmap = ListedColormap([
        (1.0, 0.5, 0.5),    # 1 - critical (medium red)
        (0.4, 0.4, 0.9),    # 2 - redundant (medium blue)
        (0.8, 0.8, 0.8),    # 3 - dummy (light gray)
    ])
    red_patch = patches.Patch(color=(1.0, 0.7, 0.7), label='Critical Pads')
    blue_patch = patches.Patch(color=(0.7, 0.7, 1.0), label='Redundant Pads')
    gray_patch = patches.Patch(color=(0.8, 0.8, 0.8), label='Dummy Pads')
    plt.legend(
        handles=[red_patch, blue_patch, gray_patch],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.07),
        ncol=3,
        frameon=False
    )
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
    plt.imshow(PAD_BITMAP, cmap=cmap, norm=norm)
    plt.title("Pad Block Bitmap")


    # Save the pad bitmaps
    plt.savefig("pad_bitmap/pad_bitmap.png")
    print("Pad bitmap collections info saved.")

    return

def convert_3dblox_to_pad_bitmap(cfg, blox_bmap_path='pad_bitmap/UCIe_standard.bmap'):
    # Extract configuration parameters
    pad_block_size = cfg.pad_block_size     # In UCIe standard, pad block size is 1 (no downsampling)
    critical_pad_ratio = cfg.critical_pad_ratio
    redundant_pad_ratio = cfg.redundant_pad_ratio
    redundant_logical_pad_copy = cfg.redundant_logical_pad_copy
    redundant_logical_pad_dist = cfg.redundant_logical_pad_dist

    # Read the bump data from the .bmap file
    bump_data = []
    with open(blox_bmap_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 6:
                instance, bump_type, x, y, port, net = parts
                bump_data.append({      # From the top-left corner to the bottom-right corner
                    "bumpid": int(instance.split("_")[1]),
                    "x": int(x),
                    "y": int(y),
                    "port": port,
                    "net": net
                })
    # Convert the bump data to pad bitmap
    port_set = set([bump['port'] for bump in bump_data])
    redundant_port_to_bumpid = dict()
    for bump in bump_data:
        if bump['port'] not in redundant_port_to_bumpid:
            redundant_port_to_bumpid[bump['port']] = []
        redundant_port_to_bumpid[bump['port']].append(bump['bumpid'])
    # Initialize the pad bitmap
    pad_arrange_pattern = 'checkerboard'   # Currently only support checkerboard pattern
    # TODO: You need to modify the simulator to support different pad arrangement patterns
    CRITICAL_PAD_BITMAP = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), dtype=bool)
    CRITICAL_PAD_BLOCK_BITMAP = downsample_bitmap(CRITICAL_PAD_BITMAP, pad_block_size)
    REDUNDANT_PAD_BITMAP = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), dtype=bool)
    REDUNDANT_PAD_BLOCK_BITMAP = downsample_bitmap(REDUNDANT_PAD_BITMAP, pad_block_size)
    if pad_arrange_pattern == 'checkerboard': # This case is for UCIe standard
        for row in range(cfg.PAD_ARR_ROW):
            for col in range(cfg.PAD_ARR_COL):
                if (row + col) % 2 == 1:    # There is a bump (but not necessarily a critical pad)
                    bump_id = int(row // 2 * cfg.PAD_ARR_COL / 2 + col // 2)
                    current_bump_dict = bump_data[bump_id]
                    if len(redundant_port_to_bumpid[current_bump_dict['port']]) == 1:
                        CRITICAL_PAD_BITMAP[row, col] = 1
                    else:
                        REDUNDANT_PAD_BITMAP[row, col] = 1
                else:
                    continue
    else:
        raise NotImplementedError("Currently only support checkerboard pad arrangement pattern.")
    DUMMY_PAD_BITMAP = None # This case is for UCIe standard, so there is no dummy pads
    REDUNDANT_MAIN_PAD_BLOCK_BITMAP = np.zeros_like(REDUNDANT_PAD_BLOCK_BITMAP, dtype=bool)
    REDUNDANT_COPY_PAD_BLOCK_BITMAP = np.zeros_like(REDUNDANT_PAD_BLOCK_BITMAP, dtype=bool)
    # We define that the first redundant pad in the redundant pads is the main pad, others are copy pads
    for row in range(cfg.PAD_ARR_ROW):
        for col in range(cfg.PAD_ARR_COL):
            if REDUNDANT_PAD_BITMAP[row, col]:
                bump_id = int(row // 2 * cfg.PAD_ARR_COL / 2 + col // 2)
                current_bump_dict = bump_data[bump_id]
                port = current_bump_dict['port']
                replica_bumpid_list = redundant_port_to_bumpid[port]
                REDUNDANT_MAIN_PAD_BLOCK_BITMAP[row, col] = True if replica_bumpid_list[0] == current_bump_dict['bumpid'] else False
                REDUNDANT_COPY_PAD_BLOCK_BITMAP[row, col] = True if replica_bumpid_list[0] != current_bump_dict['bumpid'] else False


    # Count the number of pads
    num_critical_pads = np.sum(CRITICAL_PAD_BITMAP)
    num_redundant_pads = np.sum(REDUNDANT_PAD_BITMAP)
    num_dummy_pads = 0 if DUMMY_PAD_BITMAP is None else np.sum(DUMMY_PAD_BITMAP)
    
    # Count the number of logical pads in redundant pads
    num_redundant_logical_pads = 0
    for port in redundant_port_to_bumpid:
        if len(redundant_port_to_bumpid[port]) > 1 and ('vcc' not in port.lower()) and ('vss' not in port.lower()):
            num_redundant_logical_pads += 1
    # Initialize the redundant port alive count dict
    redundant_port_alive_count_dict = dict()
    for port in redundant_port_to_bumpid:
        redundant_port_alive_count_dict[port] = len(redundant_port_to_bumpid[port])
    redundant_logical_pad_ratio = num_redundant_logical_pads / num_redundant_pads if num_redundant_pads > 0 else 0.0

    # Calculate the outmost critical pad coordinates for overlay simulation (4 totally)
    if num_critical_pad_blocks == 0:
        critical_pad_boundary_bitmap_row_col_block_ind = np.empty((0, 2), dtype=int)
    else:
        critical_pad_boundary_bitmap_row_col_block_ind = np.zeros((4, 2), dtype=int)
        row_col_ind = np.argwhere(CRITICAL_PAD_BITMAP == 1)
        top_left_ind = row_col_ind[np.argmin(row_col_ind[:, 0] + row_col_ind[:, 1])]
        top_right_ind = row_col_ind[np.argmin(row_col_ind[:, 0] - row_col_ind[:, 1])]
        bottom_left_ind = row_col_ind[np.argmax(row_col_ind[:, 0] - row_col_ind[:, 1])]
        bottom_right_ind = row_col_ind[np.argmax(row_col_ind[:, 0] + row_col_ind[:, 1])]
        critical_pad_boundary_bitmap_row_col_block_ind[0] = top_left_ind / pad_block_size
        critical_pad_boundary_bitmap_row_col_block_ind[1] = top_right_ind / pad_block_size
        critical_pad_boundary_bitmap_row_col_block_ind[2] = bottom_left_ind / pad_block_size
        critical_pad_boundary_bitmap_row_col_block_ind[3] = bottom_right_ind / pad_block_size
        non_zero_mask = critical_pad_boundary_bitmap_row_col_block_ind != 0
        critical_pad_boundary_bitmap_row_col_block_ind += non_zero_mask

    # Calculate the outmost redundant copy pad coordinates for overlay simulation (4 totally)
    redundant_copy_pad_boundary_bitmap_row_col_block_ind = None
    if len(redundant_port_alive_count_dict) != 0:
        redundant_copy_pad_boundary_bitmap_row_col_block_ind = np.zeros((4, 2), dtype=int)
        row_col_ind = np.argwhere(np.any(REDUNDANT_COPY_PAD_BLOCK_BITMAP, axis=0) == 1)
        top_left_ind = row_col_ind[np.argmin(row_col_ind[:, 0] + row_col_ind[:, 1])]
        top_right_ind = row_col_ind[np.argmin(row_col_ind[:, 0] - row_col_ind[:, 1])]
        bottom_left_ind = row_col_ind[np.argmax(row_col_ind[:, 0] - row_col_ind[:, 1])]
        bottom_right_ind = row_col_ind[np.argmax(row_col_ind[:, 0] + row_col_ind[:, 1])]
        redundant_copy_pad_boundary_bitmap_row_col_block_ind[0] = top_left_ind
        redundant_copy_pad_boundary_bitmap_row_col_block_ind[1] = top_right_ind
        redundant_copy_pad_boundary_bitmap_row_col_block_ind[2] = bottom_left_ind
        redundant_copy_pad_boundary_bitmap_row_col_block_ind[3] = bottom_right_ind


    bitmap_collection = {}
    bitmap_collection["CRITICAL_PAD_BITMAP"] = CRITICAL_PAD_BITMAP
    bitmap_collection["CRITICAL_PAD_BLOCK_BITMAP"] = CRITICAL_PAD_BLOCK_BITMAP
    bitmap_collection["critical_pad_boundary_bitmap_row_col_block_ind"] = critical_pad_boundary_bitmap_row_col_block_ind
    bitmap_collection["redundant_copy_pad_boundary_bitmap_row_col_block_ind"] = redundant_copy_pad_boundary_bitmap_row_col_block_ind
    bitmap_collection["REDUNDANT_PAD_BITMAP"] = REDUNDANT_PAD_BITMAP
    bitmap_collection["DUMMY_PAD_BITMAP"] = DUMMY_PAD_BITMAP
    bitmap_collection["REDUNDANT_MAIN_PAD_BLOCK_BITMAP"] = REDUNDANT_MAIN_PAD_BLOCK_BITMAP
    bitmap_collection["REDUNDANT_COPY_PAD_BLOCK_BITMAP"] = REDUNDANT_COPY_PAD_BLOCK_BITMAP

    bitmap_collection["is_redundant_copy_same_block"] = False
    bitmap_collection["num_critical_pads"] = num_critical_pads
    bitmap_collection["num_redundant_pads"] = num_redundant_pads
    bitmap_collection["num_redundant_logical_pads"] = num_redundant_logical_pads

    bitmap_collection["critical_pad_ratio"] = critical_pad_ratio
    bitmap_collection["redundant_pad_ratio"] = redundant_pad_ratio
    bitmap_collection["redundant_logical_pad_ratio"] = redundant_logical_pad_ratio
    bitmap_collection["redundant_logical_pad_copy"] = redundant_logical_pad_copy
    bitmap_collection["redundant_logical_pad_dist"] = redundant_logical_pad_dist
    bitmap_collection["pad_block_size"] = pad_block_size
    bitmap_collection["redundant_logical_to_physical_arr"] = None
    bitmap_collection["redundant_physical_to_logical_arr"] = None
    bitmap_collection["redundant_pad_block_pair_dict"] = None
    bitmap_collection["redundant_port_alive_count_dict"] = redundant_port_alive_count_dict
    
    # Save the bitmap collection as npy file and mat file
    np.save("pad_bitmap/bitmap_collection.npy", bitmap_collection)
    # sio.savemat("pad_bitmap/bitmap_collection.mat", bitmap_collection)

    # # Draw the critical and redundant pad bitmaps in one figure (critical light red, redundant light blue, dummy light gray)
    draw_pad_bitmap(bitmap_collection)

    # raise ValueError("Pad bitmap generation finished. Please check the pad_bitmap folder.")

    return bitmap_collection


def pad_bitmap_generate(cfg, pad_layout_pattern):
    '''
    This function generates the random pad bitmaps for the die.
    '''
    # Read parameters from the configuration
    PAD_ARR_ROW = cfg.PAD_ARR_ROW
    PAD_ARR_COL = cfg.PAD_ARR_COL

    critical_pad_ratio = cfg.critical_pad_ratio
    redundant_pad_ratio = cfg.redundant_pad_ratio
    pad_block_size = cfg.pad_block_size
    redundant_logical_pad_ratio = cfg.redundant_logical_pad_ratio
    redundant_logical_pad_copy = cfg.redundant_logical_pad_copy
    redundant_logical_pad_dist = cfg.redundant_logical_pad_dist

    # number of types of pads
    num_critical_pads = int(critical_pad_ratio * PAD_ARR_ROW * PAD_ARR_COL)
    num_redundant_pads = int(redundant_pad_ratio * PAD_ARR_ROW * PAD_ARR_COL)
    num_dummy_pads = PAD_ARR_ROW * PAD_ARR_COL - num_critical_pads - num_redundant_pads

    # Initialize the pad bitmaps
    CRITICAL_PAD_BITMAP = np.zeros((PAD_ARR_ROW, PAD_ARR_COL), dtype=bool)
    REDUNDANT_PAD_BITMAP = np.zeros((PAD_ARR_ROW, PAD_ARR_COL), dtype=bool)
    DUMMY_PAD_BITMAP = np.zeros((PAD_ARR_ROW, PAD_ARR_COL), dtype=bool)
    print("PAD_ARR_ROW:", PAD_ARR_ROW)
    print("PAD_ARR_COL:", PAD_ARR_COL)

    # Initialize the pad block
    num_pads_block_row = math.ceil(PAD_ARR_ROW / pad_block_size)
    num_pads_block_col = math.ceil(PAD_ARR_COL / pad_block_size)
    num_pads_blocks = num_pads_block_row * num_pads_block_col
    all_critical = np.isclose(critical_pad_ratio, 1.0) and np.isclose(redundant_pad_ratio, 0.0)
    num_critical_pad_blocks = (
        num_pads_blocks
        if all_critical
        else math.ceil(num_critical_pads / (pad_block_size ** 2))
    )
    num_redundant_pad_blocks = min(math.ceil(num_redundant_pads / (pad_block_size ** 2)), \
                                   num_pads_blocks - num_critical_pad_blocks)
    num_dummy_pad_blocks = num_pads_blocks - num_critical_pad_blocks - num_redundant_pad_blocks

    # Update the number of types of pads
    num_critical_pads = (
        PAD_ARR_ROW * PAD_ARR_COL
        if all_critical
        else num_critical_pad_blocks * pad_block_size ** 2
    )
    num_redundant_pads = num_redundant_pad_blocks * pad_block_size ** 2
    num_redundant_logical_pads = math.ceil(num_redundant_pads * redundant_logical_pad_ratio)

    print("Number of pad blocks:", num_pads_blocks)
    print("Pad block size:", pad_block_size)
    print("Number of critical pad blocks:", num_critical_pad_blocks)
    print("Number of redundant pad blocks:", num_redundant_pad_blocks)
    print("Number of dummy pad blocks:", num_dummy_pad_blocks)

    # Assign pad block locations for critical, redundant, and dummy pad blocks
    critical_pad_blocks, redundant_pad_blocks, dummy_pad_blocks = assign_pad_blocks(pad_layout_pattern,
                                                                                    num_pads_blocks, 
                                                                                    num_pads_block_row, 
                                                                                    num_pads_block_col, 
                                                                                    num_critical_pad_blocks, 
                                                                                    num_redundant_pad_blocks, 
                                                                                    redundant_mesh_spacing=3)

    # Save the pad block indices
    np.save("pad_bitmap/critical_pad_blocks.npy", critical_pad_blocks)
    np.save("pad_bitmap/redundant_pad_blocks.npy", redundant_pad_blocks)

    # Initialize the pad indices for the redundant pads
    redundant_available_physical_ids = []

    # Place critical pad blocks
    for i in range(num_critical_pad_blocks):
        row_start = (critical_pad_blocks[i] // num_pads_block_col) * pad_block_size
        col_start = (critical_pad_blocks[i] % num_pads_block_col) * pad_block_size
        row_end = row_start + pad_block_size
        col_end = col_start + pad_block_size
        # Prevent out of bounds
        row_end = min(row_end, PAD_ARR_ROW)
        col_end = min(col_end, PAD_ARR_COL)
        CRITICAL_PAD_BITMAP[row_start:row_end, col_start:col_end] = 1
    # Calculate the outmost critical pad coordinates for overlay simulation (4 totally)
    if num_critical_pad_blocks == 0:
        critical_pad_boundary_bitmap_row_col_block_ind = np.empty((0, 2), dtype=int)
    else:
        critical_pad_boundary_bitmap_row_col_block_ind = np.zeros((4, 2), dtype=int)
        row_col_ind = np.argwhere(CRITICAL_PAD_BITMAP == 1)
        top_left_ind = row_col_ind[np.argmin(row_col_ind[:, 0] + row_col_ind[:, 1])]
        top_right_ind = row_col_ind[np.argmin(row_col_ind[:, 0] - row_col_ind[:, 1])]
        bottom_left_ind = row_col_ind[np.argmax(row_col_ind[:, 0] - row_col_ind[:, 1])]
        bottom_right_ind = row_col_ind[np.argmax(row_col_ind[:, 0] + row_col_ind[:, 1])]
        critical_pad_boundary_bitmap_row_col_block_ind[0] = top_left_ind / pad_block_size
        critical_pad_boundary_bitmap_row_col_block_ind[1] = top_right_ind / pad_block_size
        critical_pad_boundary_bitmap_row_col_block_ind[2] = bottom_left_ind / pad_block_size
        critical_pad_boundary_bitmap_row_col_block_ind[3] = bottom_right_ind / pad_block_size
        non_zero_mask = critical_pad_boundary_bitmap_row_col_block_ind != 0
        critical_pad_boundary_bitmap_row_col_block_ind += non_zero_mask
    # print("Critical pad boundary bitmap row-col indices:", critical_pad_boundary_bitmap_row_col_block_ind)


    # Place redundant pad blocks
    redundant_pad_block_info_dict = dict()
    for redundant_block_ind in redundant_pad_blocks:
        # print("Processing redundant pad block {}/{}...".format(redundant_block_ind + 1, num_redundant_pad_blocks))
        row_start = (redundant_block_ind // num_pads_block_col) * pad_block_size
        col_start = (redundant_block_ind % num_pads_block_col) * pad_block_size
        row_end = row_start + pad_block_size
        col_end = col_start + pad_block_size
        # Prevent out of bounds
        row_end = min(row_end, PAD_ARR_ROW)
        col_end = min(col_end, PAD_ARR_COL)
        redundant_pad_block_info_dict[redundant_block_ind] = (row_start, col_start, row_end, col_end)
        REDUNDANT_PAD_BITMAP[row_start:row_end, col_start:col_end] = 1
        # Add the redundant pad ids to the available ids
        # You need to calculate the ids for the redundant pads from (row_start, col_start) to (row_end, col_end)
        rows = np.arange(row_start, row_end)
        cols = np.arange(col_start, col_end)
        row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')
        pad_ids = row_grid * PAD_ARR_COL + col_grid
        # Add the pad ids to the available ids
        redundant_available_physical_ids.extend(pad_ids.flatten())
    # print("redundant_available_physical_ids length:", len(redundant_available_physical_ids))
    redundant_available_physical_ids = np.sort(np.array(redundant_available_physical_ids))

    #####
    ## Begin to allocate physical pads for redundant logical pads    
    #####
    redundant_logical_to_physical_arr = np.zeros(
        (num_redundant_logical_pads, redundant_logical_pad_copy), dtype=np.int32
    )
 
    used_pad_ids = set()
    redundant_available_physical_ids_set = set(redundant_available_physical_ids)
    redundant_pad_block_pair_dict = {}
    num_clusters = 0
    if num_redundant_logical_pads == 0:
        pass
    else:    # One main pad has one copy pad
        '''
        One-to-one assignment: | M -> C | M -> C | M -> C | .....
        '''
        # Calculate the number of clusters per pad block to accommodate the main pad and its copies
        # If num_clusters is 0, it means the pad block contains all the main or copy pads
        num_clusters = int((col_end - col_start) / (2 * redundant_logical_pad_dist))
        # Find the pad block pairs that has the required distance
        redundant_pad_block_pair_dict = dict()
        if num_clusters > 0:
            # the main pad and its copies can be placed in the same block
            shuffled_items = list(redundant_pad_block_info_dict.items())
            np.random.shuffle(shuffled_items)  # Shuffle the items to randomize the order
            for idx, (block_ind, scope) in enumerate(shuffled_items):
            # for idx, (block_ind, scope) in enumerate(redundant_pad_block_info_dict.items()):
                redundant_pad_block_pair_dict[block_ind] = block_ind
                row_start, col_start, row_end, col_end = scope
                # Get the pad ids for the redundant pads
                rows = np.arange(row_start, row_end)
                cols = np.arange(col_start, col_end)

                if idx % 20 == 0:
                    print(
                        "Processing redundant block {}/{}..., cluster size: {}".format(
                            idx + 1,
                            num_redundant_pad_blocks,
                            int((col_end - col_start) / (2 * redundant_logical_pad_dist)),
                        ),
                        end="\r",
                        flush=True,
                    )
                cluster_col_starts = col_start + np.arange(num_clusters) * 2 * redundant_logical_pad_dist
                # number of rows and columns within the cluster
                num_rows = row_end - row_start
                num_cols = redundant_logical_pad_dist
                row_offsets = np.arange(num_rows).reshape(-1, 1, 1)
                col_offsets = np.arange(num_cols).reshape(1, -1, 1)   
                cluster_cols = col_offsets + cluster_col_starts
                cluster_rows = row_offsets + row_start
                pad_ids = cluster_rows * PAD_ARR_COL + cluster_cols
                pad_ids = pad_ids.flatten()
                copy_ids = pad_ids + redundant_logical_pad_dist
                # Out of bounds check
                if len(used_pad_ids) + len(pad_ids) + len(copy_ids) > num_redundant_logical_pads * redundant_logical_pad_copy:
                    num_remaining_pads = num_redundant_logical_pads * redundant_logical_pad_copy - len(used_pad_ids)
                    pad_ids = pad_ids[0:int(num_remaining_pads / redundant_logical_pad_copy)]
                    copy_ids = copy_ids[0:int(num_remaining_pads / redundant_logical_pad_copy)]
                current_logical_ind = int(len(used_pad_ids) / redundant_logical_pad_copy)
                # Update the logical to physical mapping (The index is the logical pad id, the value is the physical pad id)
                redundant_logical_to_physical_arr[current_logical_ind:current_logical_ind + len(pad_ids), 0] = pad_ids
                if redundant_logical_pad_copy > 1:
                    redundant_logical_to_physical_arr[current_logical_ind:current_logical_ind + len(copy_ids), 1:] = copy_ids.reshape(-1, redundant_logical_pad_copy - 1)
                # Update the used pad ids
                used_pad_ids.update(pad_ids)
                used_pad_ids.update(copy_ids)
                # Update the available pad ids
                redundant_available_physical_ids_set = redundant_available_physical_ids_set - used_pad_ids
                if len(used_pad_ids) >= num_redundant_logical_pads * redundant_logical_pad_copy:
                    break
        else:
            # The main pad and its copy must occupy different blocks.  Solve
            # the complete non-overlapping pairing problem instead of making
            # irreversible greedy choices that can strand the final blocks.
            pad_block_dist = int(redundant_logical_pad_dist / pad_block_size)
            required_pairs = math.ceil(num_redundant_logical_pads / pad_block_size**2)
            redundant_pad_block_pair_dict = dict(
                place_redundant_block_pairs(
                    redundant_pad_blocks,
                    num_pads_block_row,
                    num_pads_block_col,
                    required_pairs,
                    pad_block_dist,
                )
            )

            # Assign the physical pads for the redundant logical pads
            # Each redundant logical pad has redundant_logical_pad_copy copies
            # Each copy has a distance of at least redundant_logical_pad_dist * PITCH_um
            for idx, (main_block_ind, copy_block_ind) in enumerate(redundant_pad_block_pair_dict.items()):
                if idx % 20 == 0:
                    print(
                        "Processing redundant block {}/{}..., cluster size: {}".format(
                            idx + 1,
                            num_redundant_pad_blocks,
                            int((col_end - col_start) / (2 * redundant_logical_pad_dist)),
                        ),
                        end="\r",
                        flush=True,
                    )
                # Get the pad ids for the main pad and its copies
                row_start, col_start, row_end, col_end = redundant_pad_block_info_dict[main_block_ind]
                rows = np.arange(row_start, row_end)
                cols = np.arange(col_start, col_end)
                row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')
                main_pad_ids = row_grid * PAD_ARR_COL + col_grid
                main_pad_ids = main_pad_ids.flatten()
                
                # Get the pad ids for the copy pad
                row_start, col_start, row_end, col_end = redundant_pad_block_info_dict[copy_block_ind]
                rows = np.arange(row_start, row_end)
                cols = np.arange(col_start, col_end)
                row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')
                copy_pad_ids = row_grid * PAD_ARR_COL + col_grid
                copy_pad_ids = copy_pad_ids.flatten()

                # Check if the pad ids are already used
                if len(used_pad_ids) + len(main_pad_ids) + len(copy_pad_ids) > num_redundant_logical_pads * redundant_logical_pad_copy:
                    num_remaining_pads = num_redundant_logical_pads * redundant_logical_pad_copy - len(used_pad_ids)
                    main_pad_ids = main_pad_ids[0:int(num_remaining_pads / redundant_logical_pad_copy)]
                    copy_pad_ids = copy_pad_ids[0:int(num_remaining_pads / redundant_logical_pad_copy)]
                
                current_logical_ind = int(len(used_pad_ids) / redundant_logical_pad_copy)
                # Update the logical to physical mapping (The index is the logical pad id, the value is the physical pad id)
                redundant_logical_to_physical_arr[current_logical_ind:current_logical_ind + len(main_pad_ids), 0] = main_pad_ids
                if redundant_logical_pad_copy > 1:
                    redundant_logical_to_physical_arr[current_logical_ind:current_logical_ind + len(copy_pad_ids), 1:] = copy_pad_ids.reshape(-1, redundant_logical_pad_copy - 1)
                
                # Update the used pad ids
                used_pad_ids.update(main_pad_ids)
                used_pad_ids.update(copy_pad_ids)
                
            print("\nRedundant blocks and pads assigned.")

    REDUNDANT_MAIN_PAD_BLOCK_BITMAP = np.zeros((len(redundant_pad_block_pair_dict), num_pads_block_row, num_pads_block_col), dtype=bool)
    REDUNDANT_COPY_PAD_BLOCK_BITMAP = np.zeros((len(redundant_pad_block_pair_dict), num_pads_block_row, num_pads_block_col), dtype=bool)
    for idx, (main_block_ind, copy_block_ind) in enumerate(redundant_pad_block_pair_dict.items()):
        main_row = (main_block_ind // num_pads_block_col)
        main_col = (main_block_ind % num_pads_block_col)
        REDUNDANT_MAIN_PAD_BLOCK_BITMAP[idx, main_row, main_col] = True

        copy_row = (copy_block_ind // num_pads_block_col)
        copy_col = (copy_block_ind % num_pads_block_col)
        REDUNDANT_COPY_PAD_BLOCK_BITMAP[idx, copy_row, copy_col] = True

    # Calculate the outmost redundant copy pad coordinates for overlay simulation (4 totally)
    redundant_copy_pad_boundary_bitmap_row_col_block_ind = None
    if len(redundant_pad_block_pair_dict) != 0:
        redundant_copy_pad_boundary_bitmap_row_col_block_ind = np.zeros((4, 2), dtype=int)
        row_col_ind = np.argwhere(np.any(REDUNDANT_COPY_PAD_BLOCK_BITMAP, axis=0) == 1)
        top_left_ind = row_col_ind[np.argmin(row_col_ind[:, 0] + row_col_ind[:, 1])]
        top_right_ind = row_col_ind[np.argmin(row_col_ind[:, 0] - row_col_ind[:, 1])]
        bottom_left_ind = row_col_ind[np.argmax(row_col_ind[:, 0] - row_col_ind[:, 1])]
        bottom_right_ind = row_col_ind[np.argmax(row_col_ind[:, 0] + row_col_ind[:, 1])]
        redundant_copy_pad_boundary_bitmap_row_col_block_ind[0] = top_left_ind
        redundant_copy_pad_boundary_bitmap_row_col_block_ind[1] = top_right_ind
        redundant_copy_pad_boundary_bitmap_row_col_block_ind[2] = bottom_left_ind
        redundant_copy_pad_boundary_bitmap_row_col_block_ind[3] = bottom_right_ind


    # Get the physical pad id to logical pad id mapping
    if num_redundant_logical_pads > 0:
        logical_ids_repeated = np.repeat(
            np.arange(num_redundant_logical_pads, dtype=np.int32),
            redundant_logical_pad_copy,
        )
        physical_ids = redundant_logical_to_physical_arr.flatten()
        redundant_physical_to_logical_arr = np.full(
            PAD_ARR_ROW * PAD_ARR_COL, -1, dtype=np.int32
        )
        redundant_physical_to_logical_arr[physical_ids] = logical_ids_repeated
    else:
        redundant_physical_to_logical_arr = np.empty(0, dtype=np.int32)

    # Save the redundant logical pad -> physical pad mapping and the physical pad -> logical pad mapping
    # np.save("pad_bitmap/redundant_logical_to_physical_arr.npy", redundant_logical_to_physical_arr)
    # np.save("pad_bitmap/redundant_physical_to_logical_arr.npy", redundant_physical_to_logical_arr)

    CRITICAL_PAD_BLOCK_BITMAP = downsample_bitmap(CRITICAL_PAD_BITMAP, pad_block_size)
                
            
    DUMMY_PAD_BITMAP = np.logical_not(CRITICAL_PAD_BITMAP + REDUNDANT_PAD_BITMAP)

    # Save the data in a dictionary: bitmap_collection
    bitmap_collection = {}
    bitmap_collection["CRITICAL_PAD_BITMAP"] = CRITICAL_PAD_BITMAP
    bitmap_collection["CRITICAL_PAD_BLOCK_BITMAP"] = CRITICAL_PAD_BLOCK_BITMAP
    bitmap_collection["critical_pad_boundary_bitmap_row_col_block_ind"] = critical_pad_boundary_bitmap_row_col_block_ind
    bitmap_collection["redundant_copy_pad_boundary_bitmap_row_col_block_ind"] = redundant_copy_pad_boundary_bitmap_row_col_block_ind
    bitmap_collection["REDUNDANT_PAD_BITMAP"] = REDUNDANT_PAD_BITMAP
    bitmap_collection["DUMMY_PAD_BITMAP"] = DUMMY_PAD_BITMAP
    bitmap_collection["REDUNDANT_MAIN_PAD_BLOCK_BITMAP"] = REDUNDANT_MAIN_PAD_BLOCK_BITMAP
    bitmap_collection["REDUNDANT_COPY_PAD_BLOCK_BITMAP"] = REDUNDANT_COPY_PAD_BLOCK_BITMAP

    bitmap_collection["is_redundant_copy_same_block"] = (num_clusters > 0)
    bitmap_collection["num_critical_pads"] = num_critical_pads
    bitmap_collection["num_redundant_pads"] = num_redundant_pads
    bitmap_collection["num_redundant_logical_pads"] = num_redundant_logical_pads

    bitmap_collection["critical_pad_ratio"] = critical_pad_ratio
    bitmap_collection["redundant_pad_ratio"] = redundant_pad_ratio
    bitmap_collection["redundant_logical_pad_ratio"] = redundant_logical_pad_ratio
    bitmap_collection["redundant_logical_pad_copy"] = redundant_logical_pad_copy
    bitmap_collection["redundant_logical_pad_dist"] = redundant_logical_pad_dist
    bitmap_collection["pad_block_size"] = pad_block_size
    bitmap_collection["redundant_logical_to_physical_arr"] = redundant_logical_to_physical_arr
    bitmap_collection["redundant_physical_to_logical_arr"] = redundant_physical_to_logical_arr
    bitmap_collection["redundant_pad_block_pair_dict"] = redundant_pad_block_pair_dict
    
    # Save the bitmap collection as npy file and mat file
    np.save("pad_bitmap/bitmap_collection.npy", bitmap_collection)
    # sio.savemat("pad_bitmap/bitmap_collection.mat", bitmap_collection)

    # # Draw the critical and redundant pad bitmaps in one figure (critical light red, redundant light blue, dummy light gray)
    draw_pad_bitmap(bitmap_collection)

    # raise ValueError("Pad bitmap generation finished. Please check the pad_bitmap folder.")

    return bitmap_collection


def pad_bitmap_generate_read(cfg, bitmap_collection_path):
    '''
    This function load the pad bitmaps for the die.
    '''
    # Read the configuration
    PAD_ARR_ROW = cfg.PAD_ARR_ROW
    PAD_ARR_COL = cfg.PAD_ARR_COL

    bitmap_collection = {}
    # load the pad bitmaps
    bitmap_collection = np.load("bitmap_collection_path", allow_pickle=True).item()
    CRITICAL_PAD_BITMAP = bitmap_collection["CRITICAL_PAD_BITMAP"]
    REDUNDANT_PAD_BITMAP = bitmap_collection["REDUNDANT_PAD_BITMAP"]
    DUMMY_PAD_BITMAP = bitmap_collection["DUMMY_PAD_BITMAP"]
    pad_block_size = bitmap_collection["pad_block_size"]

    print("Number of pad blocks:", round(PAD_ARR_ROW * PAD_ARR_COL / (pad_block_size ** 2)))
    print("Pad block size:", pad_block_size)
    print("Number of critical pad blocks:", round(np.sum(CRITICAL_PAD_BITMAP / (pad_block_size ** 2))))
    print("Number of redundant pad blocks:", round(np.sum(REDUNDANT_PAD_BITMAP / (pad_block_size ** 2))))
    print("Number of dummy pad blocks:", round(np.sum(DUMMY_PAD_BITMAP / (pad_block_size ** 2))))

    # Draw the critical and redundant pad bitmaps in one figure (critical light red, redundant light blue, dummy light gray)
    draw_pad_bitmap(bitmap_collection)

    return bitmap_collection


def build_struture(mode, defect_info, pixel_size):
    if mode == "void_tail":
        '''
        This branch builds a line structure in a void tail shape with length l and orientation theta.
        '''
        l_um = defect_info["l"]
        r_avg_mv_um = defect_info["r_avg_mv"]
        theta_rad = defect_info["theta"]

        # Calculate the coordinates of the line structure
        length_pixels = int(np.round(l_um / pixel_size))
        width_pixels = int(2 * r_avg_mv_um / pixel_size)
        if length_pixels < 1:
            length_pixels = 1
        
        # Set the size of the line structure
        size = length_pixels * 2 + 1
        structure_element = np.zeros((size, size), dtype=bool)

        # Center coordinates
        cx, cy = size // 2, size // 2

        # Calculate the coordinates of the line structure endpoints (centered at (cx, cy))
        x1 = cx - length_pixels / 2 * np.cos(theta_rad)
        y1 = cy - length_pixels / 2 * np.sin(theta_rad)
        x2 = cx + length_pixels / 2 * np.cos(theta_rad)
        y2 = cy + length_pixels / 2 * np.sin(theta_rad)

        # Draw the line structure using Bresenham's line algorithm
        num_points = length_pixels * 2
        xs = np.linspace(x1, x2, num_points)
        ys = np.linspace(y1, y2, num_points)
        for idx, (x, y) in enumerate(zip(xs, ys)):
            x = int(np.round(x))
            y = int(np.round(y))
            if 0 <= x < size and 0 <= y < size:
                structure_element[x, y] = True
                for i in range(1, width_pixels + 1):
                    if y - i >= 0:
                        structure_element[x, y - i] = True
        # Convert the line structure to a binary image
        structure_element = np.uint8(structure_element)
    elif mode == "main_void":
        '''
        This branch builds a line sturcture in a main void shape with radius r_mv. 
        '''
        r_mv_um = defect_info["r_mv"]
        eff_r = np.round(r_mv_um / pixel_size)
        # if eff_r < 1:
        #     eff_r = 1
        side_len_pixels = int(math.ceil(2 * eff_r))
        if side_len_pixels < 3:
            side_len_pixels = 3
        structure_element = np.zeros((side_len_pixels, side_len_pixels), dtype=bool)
        center = side_len_pixels // 2
        for i in range(side_len_pixels):
            for j in range(side_len_pixels):
                if (i - center) ** 2 + (j - center) ** 2 <= int(np.round(eff_r ** 2)):
                    structure_element[i, j] = True
        # Convert the line structure to a binary image
        structure_element = np.uint8(structure_element)
    else:
        raise ValueError("Invalid mode: {}".format(mode))


    return structure_element

def crop_to_nonzero(structure):
    '''
    This function crops the structure to the non-zero area.
    '''
    rows = np.any(structure, axis=1)
    cols = np.any(structure, axis=0)

    if not np.any(rows) or not np.any(cols):
        return structure

    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]

    return structure[top:bottom+1, left:right+1]

def A_critical_l_across_theta(cfg,
                              PITCH_um,
                                l,
                                r_avg_mv,
                                angle_step,
                                bitmap_collection,):
    '''
    This function calculates the critical area of line defects via Monte Carlo simulation.
    Usually for wafer-to-wafer hybrid bonding.
    '''
    # l=3000
    start = time.time()
    # First, read the pad bitmap and make it finer
    CRITICAL_PAD_BLOCK_BITMAP = bitmap_collection["CRITICAL_PAD_BLOCK_BITMAP"]
    REDUNDANT_MAIN_PAD_BLOCK_BITMAP = bitmap_collection["REDUNDANT_MAIN_PAD_BLOCK_BITMAP"]  # (N, H, W), N is the number of block pairs in redundant pads
    REDUNDANT_COPY_PAD_BLOCK_BITMAP = bitmap_collection["REDUNDANT_COPY_PAD_BLOCK_BITMAP"]  # (N, H, W), N is the number of block pairs in redundant pads
    pad_block_size = bitmap_collection["pad_block_size"]
    redundant_pad_block_pair_dict = bitmap_collection["redundant_pad_block_pair_dict"]

    # print("[Time] Downsampling:", time.time() - start)
    # Add padding to the bitmap
    padding_size = int(np.ceil(l / PITCH_um / pad_block_size))

    if cfg.DEBUG == True:
        # Draw REDUNDANT_MAIN_PAD_BLOCK_BITMAP_DILATED
        plt.figure(figsize=(8, 8))
        plt.imshow(CRITICAL_PAD_BLOCK_BITMAP, cmap='gray')
        plt.title("CRITICAL_PAD_BITMAP")
        plt.show()
        sio.savemat("pad_bitmap/CRITICAL_PAD_BLOCK_BITMAP.mat", {"CRITICAL_PAD_BLOCK_BITMAP": CRITICAL_PAD_BLOCK_BITMAP})
        ind=0
        # Draw the redundant pad block bitmaps
        plt.figure(figsize=(8, 8))
        plt.imshow(REDUNDANT_MAIN_PAD_BLOCK_BITMAP[ind], cmap='gray')
        plt.title("REDUNDANT_MAIN_PAD_BLOCK_BITMAP")
        plt.show()
        sio.savemat("pad_bitmap/REDUNDANT_MAIN_PAD_BLOCK_BITMAP.mat", {"REDUNDANT_MAIN_PAD_BLOCK_BITMAP": REDUNDANT_MAIN_PAD_BLOCK_BITMAP[ind]})
        plt.figure(figsize=(8, 8))
        plt.imshow(REDUNDANT_COPY_PAD_BLOCK_BITMAP[ind], cmap='gray')
        plt.title("REDUNDANT_COPY_PAD_BLOCK_BITMAP")
        plt.show()
        sio.savemat("pad_bitmap/REDUNDANT_COPY_PAD_BLOCK_BITMAP.mat", {"REDUNDANT_COPY_PAD_BLOCK_BITMAP": REDUNDANT_COPY_PAD_BLOCK_BITMAP[ind]})
    
    CRITICAL_PAD_BLOCK_BITMAP_EXPAND = np.pad(
        CRITICAL_PAD_BLOCK_BITMAP,
        ((padding_size, padding_size), (padding_size, padding_size)),
        mode='constant',
        constant_values=0
    )
    # if there is no redundant pad, we can skip operations on redundant pad bitmaps
    if REDUNDANT_MAIN_PAD_BLOCK_BITMAP.sum() != 0 and REDUNDANT_COPY_PAD_BLOCK_BITMAP.sum() != 0:
        REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND = np.pad(
            REDUNDANT_MAIN_PAD_BLOCK_BITMAP,
            pad_width=((0, 0), (padding_size, padding_size), (padding_size, padding_size)),
            mode='constant',
            constant_values=0
        )
        REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND = np.pad(
            REDUNDANT_COPY_PAD_BLOCK_BITMAP,
            pad_width=((0, 0), (padding_size, padding_size), (padding_size, padding_size)),
            mode='constant',
            constant_values=0
        )

        N, H, W = REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND.shape
        # Reshape the bitmap to (N * H, W)
        REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND = REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND.reshape(N * H, W)
        REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND = REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND.reshape(N * H, W)
    

    start = time.time()
    t00 = time.time()
    critical_area_list_across_theta = []
    line_defect_info = dict()
    for theta in np.arange(0, 180, angle_step):
        # print("[Time] Processing theta:", theta)
        theta = theta * np.pi / 180
        # Dilate the critical pad bitmap to generate critical area
        line_defect_info["theta"] = theta
        line_defect_info["l"] = l
        line_defect_info["r_avg_mv"] = r_avg_mv
        line_defect = build_struture(mode="void_tail", defect_info=line_defect_info, pixel_size=PITCH_um*pad_block_size)
        line_defect = crop_to_nonzero(line_defect)

        TOTAL_CRITICAL_AREA_BITMAP = np.zeros_like(CRITICAL_PAD_BLOCK_BITMAP_EXPAND[0], dtype=bool)
        CRITICAL_PAD_BITMAP_DILATED = np.zeros_like(CRITICAL_PAD_BLOCK_BITMAP_EXPAND[0], dtype=bool)
        CRITICAL_PAD_BITMAP_DILATED = cv2.dilate(
            CRITICAL_PAD_BLOCK_BITMAP_EXPAND.astype(np.uint8),
            line_defect,
            iterations=1
        )
    
        # print("[Time] Critical pad bitmap dilation:", time.time() - start)
        start = time.time()
        if REDUNDANT_MAIN_PAD_BLOCK_BITMAP.sum() != 0 and REDUNDANT_COPY_PAD_BLOCK_BITMAP.sum() != 0:
            REDUNDANT_PAD_BLOCK_DILATED = np.zeros_like(REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND[0], dtype=bool)
            REDUNDANT_MAIN_PAD_BLOCK_BITMAP_DILATED = cv2.dilate(
                REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND.astype(np.uint8),
                line_defect,
                iterations=1
            )
            REDUNDANT_COPY_PAD_BLOCK_BITMAP_DILATED = cv2.dilate(
                REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND.astype(np.uint8),
                line_defect,
                iterations=1
            )

            redundant_pad_block_pair_dilated = np.logical_and(
                REDUNDANT_MAIN_PAD_BLOCK_BITMAP_DILATED,
                REDUNDANT_COPY_PAD_BLOCK_BITMAP_DILATED
            )
            
            redundant_pad_block_pair_dilated = np.reshape(redundant_pad_block_pair_dilated, (N, H, W))  # (N, H, W)
            redundant_pad_block_pair_dilated = np.any(redundant_pad_block_pair_dilated, axis=0)  # (H, W)
            REDUNDANT_PAD_BLOCK_DILATED = np.logical_or(
                REDUNDANT_PAD_BLOCK_DILATED,
                redundant_pad_block_pair_dilated
            )
            # print("How many redundant pad blocks are dilated:", np.sum(REDUNDANT_PAD_BLOCK_DILATED))
            # print("[Time] Redundant pad block dilation:", time.time() - start)
            TOTAL_CRITICAL_AREA_BITMAP = np.logical_or(
                CRITICAL_PAD_BITMAP_DILATED,
                REDUNDANT_PAD_BLOCK_DILATED
            )
        else:
            TOTAL_CRITICAL_AREA_BITMAP = CRITICAL_PAD_BITMAP_DILATED
        
        if cfg.DEBUG == True:
            # Draw the critical bitmap
            plt.figure(figsize=(8, 8))
            plt.imshow(CRITICAL_PAD_BLOCK_BITMAP, cmap='gray')
            plt.title("CRITICAL_PAD_BLOCK_BITMAP")
            plt.show()
            sio.savemat("pad_bitmap/CRITICAL_PAD_BLOCK_BITMAP.mat", {"CRITICAL_PAD_BLOCK_BITMAP": CRITICAL_PAD_BLOCK_BITMAP})
            # Draw the line structure
            plt.figure(figsize=(8, 8))
            plt.imshow(line_defect, cmap='gray')
            plt.title("Line Structure")
            plt.show()
            sio.savemat("pad_bitmap/line_defect.mat", {"line_defect": line_defect})
            # Draw CRITICAL_PAD_BITMAP_DILATED
            plt.figure(figsize=(8, 8))
            plt.imshow(CRITICAL_PAD_BITMAP_DILATED, cmap='gray')
            plt.title("CRITICAL_PAD_BITMAP_DILATED")
            plt.show()
            sio.savemat("pad_bitmap/CRITICAL_PAD_BITMAP_DILATED.mat", {"CRITICAL_PAD_BITMAP_DILATED": CRITICAL_PAD_BITMAP_DILATED})
            if REDUNDANT_MAIN_PAD_BLOCK_BITMAP.sum() != 0 and REDUNDANT_COPY_PAD_BLOCK_BITMAP.sum() != 0:
                # Draw the redundant pad block pair dilated bitmap
                REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND_original = REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND.reshape(N, H, W)[ind]
                REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND_original = REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND.reshape(N, H, W)[ind]
                REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND_original_ind_dilated = cv2.dilate(
                    REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND_original.astype(np.uint8),
                    line_defect,
                    iterations=1
                )
                REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND_original_ind_dilated = cv2.dilate(
                    REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND_original.astype(np.uint8),
                    line_defect,
                    iterations=1
                )
                plt.figure(figsize=(8, 8))
                plt.imshow(REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND_original_ind_dilated, cmap='gray')
                plt.title("Redundant Pad Block Pair Dilated")
                plt.show()
                sio.savemat("pad_bitmap/REDUNDANT_MAIN_PAD_BLOCK_BITMAP_DILATED.mat", {"REDUNDANT_MAIN_PAD_BLOCK_BITMAP_DILATED": REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND_original_ind_dilated})
                plt.figure(figsize=(8, 8))
                plt.imshow(REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND_original_ind_dilated, cmap='gray')
                plt.title("Redundant Pad Block Pair Dilated")
                plt.show()
                sio.savemat("pad_bitmap/REDUNDANT_COPY_PAD_BLOCK_BITMAP_DILATED.mat", {"REDUNDANT_COPY_PAD_BLOCK_BITMAP_DILATED": REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND_original_ind_dilated})
                cross_bitmap_main_copy = np.logical_and(
                    REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND_original_ind_dilated,
                    REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND_original_ind_dilated
                )
                # Draw the cross bitmap
                plt.figure(figsize=(8, 8))
                plt.imshow(cross_bitmap_main_copy, cmap='gray')
                plt.title("Cross Bitmap Main Copy")
                plt.show()
                sio.savemat("pad_bitmap/cross_bitmap_main_copy.mat", {"cross_bitmap_main_copy": cross_bitmap_main_copy})
            # Save the total critical area bitmap
            sio.savemat("pad_bitmap/TOTAL_CRITICAL_AREA_BITMAP.mat", {"TOTAL_CRITICAL_AREA_BITMAP": TOTAL_CRITICAL_AREA_BITMAP})
            # Draw the cross bitmap
            plt.figure(figsize=(8, 8))
            plt.imshow(TOTAL_CRITICAL_AREA_BITMAP, cmap='gray')
            plt.title("TOATAL CRITICAL AREA BITMAP")
            plt.show()
            np.save("pad_bitmap/TOTAL_CRITICAL_AREA_BITMAP.npy", TOTAL_CRITICAL_AREA_BITMAP)
            # sio.savemat("pad_bitmap/TOTAL_CRITICAL_AREA_BITMAP.mat", {"TOTAL_CRITICAL_AREA_BITMAP": TOTAL_CRITICAL_AREA_BITMAP})
            raise ValueError("Critical area is not correct.")
        # print("How many total critical area blocks are dilated:", np.sum(TOTAL_CRITICAL_AREA_BITMAP))
        # print("Critical area: {}", np.sum(TOTAL_CRITICAL_AREA_BITMAP) * PITCH_um**2 * pad_block_size**2)
        # Calculate the critical area
        critical_area_list_across_theta.append(np.sum(TOTAL_CRITICAL_AREA_BITMAP) * PITCH_um**2 * pad_block_size**2)
        
        # print("[Time] One theta critical area:", time.time() - t0)

    # print("[Time] All theta iteration", time.time() - t00)
    # Average the critical area across all angles
    critical_area_across_theta = np.mean(critical_area_list_across_theta) 
    # print("Critical area across theta:", critical_area_across_theta)
    # raise ValueError("Critical area figure drawing.")
    return critical_area_across_theta




def A_critical_r_mv(cfg,
                    PITCH_um,
                    r_mv,
                    bitmap_collection,):
    '''
    This function calculates the critical area of line defects via Monte Carlo simulation.
    '''
    start = time.time()
    # First, read the pad bitmap and make it finer
    CRITICAL_PAD_BLOCK_BITMAP = bitmap_collection["CRITICAL_PAD_BLOCK_BITMAP"]
    REDUNDANT_MAIN_PAD_BLOCK_BITMAP = bitmap_collection["REDUNDANT_MAIN_PAD_BLOCK_BITMAP"]  # (N, H, W), N is the number of block pairs in redundant pads
    REDUNDANT_COPY_PAD_BLOCK_BITMAP = bitmap_collection["REDUNDANT_COPY_PAD_BLOCK_BITMAP"]  # (N, H, W), N is the number of block pairs in redundant pads
    pad_block_size = bitmap_collection["pad_block_size"]

    # print("[Time] Downsampling:", time.time() - start)
    # Add padding to the bitmap
    padding_size = int(np.ceil(2 * r_mv / PITCH_um / pad_block_size))

    if cfg.DEBUG == True:
        print("r_mv: {:.4f}".format(r_mv))
        # Draw REDUNDANT_MAIN_PAD_BLOCK_BITMAP_DILATED
        plt.figure(figsize=(8, 8))
        plt.imshow(CRITICAL_PAD_BLOCK_BITMAP, cmap='gray')
        plt.title("CRITICAL_PAD_BITMAP")
        plt.show()
        sio.savemat("pad_bitmap/CRITICAL_PAD_BLOCK_BITMAP.mat", {"CRITICAL_PAD_BLOCK_BITMAP": CRITICAL_PAD_BLOCK_BITMAP})
        ind=6
        # Draw the redundant pad block bitmaps
        plt.figure(figsize=(8, 8))
        plt.imshow(REDUNDANT_MAIN_PAD_BLOCK_BITMAP[ind], cmap='gray')
        plt.title("REDUNDANT_MAIN_PAD_BLOCK_BITMAP")
        plt.show()
        sio.savemat("pad_bitmap/REDUNDANT_MAIN_PAD_BLOCK_BITMAP.mat", {"REDUNDANT_MAIN_PAD_BLOCK_BITMAP": REDUNDANT_MAIN_PAD_BLOCK_BITMAP[ind]})
        plt.figure(figsize=(8, 8))
        plt.imshow(REDUNDANT_COPY_PAD_BLOCK_BITMAP[ind], cmap='gray')
        plt.title("REDUNDANT_COPY_PAD_BLOCK_BITMAP")
        plt.show()
        sio.savemat("pad_bitmap/REDUNDANT_COPY_PAD_BLOCK_BITMAP.mat", {"REDUNDANT_COPY_PAD_BLOCK_BITMAP": REDUNDANT_COPY_PAD_BLOCK_BITMAP[ind]})
    
    CRITICAL_PAD_BLOCK_BITMAP_EXPAND = np.pad(
        CRITICAL_PAD_BLOCK_BITMAP,
        ((padding_size, padding_size), (padding_size, padding_size)),
        mode='constant',
        constant_values=0
    )

    REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND = np.pad(
        REDUNDANT_MAIN_PAD_BLOCK_BITMAP,
        pad_width=((0, 0), (padding_size, padding_size), (padding_size, padding_size)),
        mode='constant',
        constant_values=0
    )
    REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND = np.pad(
        REDUNDANT_COPY_PAD_BLOCK_BITMAP,
        pad_width=((0, 0), (padding_size, padding_size), (padding_size, padding_size)),
        mode='constant',
        constant_values=0
    )

    N, H, W = REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND.shape
    # Reshape the bitmap to (N * H, W)
    REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND = REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND.reshape(N * H, W).astype(bool)
    REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND = REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND.reshape(N * H, W).astype(bool)
    # print("data type of REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND:", REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND.dtype)

    start = time.time()
    t00 = time.time()
    void_defect_info = dict()
    void_defect_info["r_mv"] = r_mv
    # Dilate the critical pad bitmap to generate critical area
    void_defect = build_struture(mode="main_void", defect_info=void_defect_info, pixel_size=PITCH_um*pad_block_size)
    void_defect = crop_to_nonzero(void_defect)


    TOTAL_CRITICAL_AREA_BITMAP = np.zeros_like(CRITICAL_PAD_BLOCK_BITMAP_EXPAND[0], dtype=bool)
    CRITICAL_PAD_BITMAP_DILATED = np.zeros_like(CRITICAL_PAD_BLOCK_BITMAP_EXPAND[0], dtype=bool)
    CRITICAL_PAD_BITMAP_DILATED = cv2.dilate(
        CRITICAL_PAD_BLOCK_BITMAP_EXPAND.astype(np.uint8),
        void_defect,
        iterations=1
    )

    # print("[Time] Critical pad bitmap dilation:", time.time() - start)
    start = time.time()
    REDUNDANT_PAD_BLOCK_DILATED = np.zeros_like(REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND[0], dtype=bool)
    REDUNDANT_MAIN_PAD_BLOCK_BITMAP_DILATED = cv2.dilate(
        REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND.astype(np.uint8),
        void_defect,
        iterations=1
    )
    REDUNDANT_COPY_PAD_BLOCK_BITMAP_DILATED = cv2.dilate(
        REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND.astype(np.uint8),
        void_defect,
        iterations=1
    )

    redundant_pad_block_pair_dilated = np.logical_and(
        REDUNDANT_MAIN_PAD_BLOCK_BITMAP_DILATED,
        REDUNDANT_COPY_PAD_BLOCK_BITMAP_DILATED
    )
    
    redundant_pad_block_pair_dilated = np.reshape(redundant_pad_block_pair_dilated, (N, H, W)).astype(bool)  # (N, H, W)
    redundant_pad_block_pair_dilated = np.any(redundant_pad_block_pair_dilated, axis=0)  # (H, W)
    REDUNDANT_PAD_BLOCK_DILATED = np.logical_or(
        REDUNDANT_PAD_BLOCK_DILATED,
        redundant_pad_block_pair_dilated
    )
    # print("How many redundant pad blocks are dilated:", np.sum(REDUNDANT_PAD_BLOCK_DILATED))
    # print("[Time] Redundant pad block dilation:", time.time() - start)
    TOTAL_CRITICAL_AREA_BITMAP = np.logical_or(
        CRITICAL_PAD_BITMAP_DILATED,
        REDUNDANT_PAD_BLOCK_DILATED
    )
    
    if cfg.DEBUG == True:
        # Draw the line structure
        plt.figure(figsize=(8, 8))
        plt.imshow(void_defect, cmap='gray')
        plt.title("Line Structure")
        plt.show()
        sio.savemat("pad_bitmap/void_defect.mat", {"void_defect": void_defect})
        # Draw CRITICAL_PAD_BITMAP_DILATED
        plt.figure(figsize=(8, 8))
        plt.imshow(CRITICAL_PAD_BITMAP_DILATED, cmap='gray')
        plt.title("CRITICAL_PAD_BITMAP_DILATED")
        plt.show()
        sio.savemat("pad_bitmap/CRITICAL_PAD_BITMAP_DILATED.mat", {"CRITICAL_PAD_BITMAP_DILATED": CRITICAL_PAD_BITMAP_DILATED})
        # Draw the redundant pad block pair dilated bitmap
        REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND_original = REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND.reshape(N, H, W)[ind]
        REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND_original = REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND.reshape(N, H, W)[ind]
        REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND_original_ind_dilated = cv2.dilate(
            REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND_original.astype(np.uint8),
            void_defect,
            iterations=1
        )
        REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND_original_ind_dilated = cv2.dilate(
            REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND_original.astype(np.uint8),
            void_defect,
            iterations=1
        )
        plt.figure(figsize=(8, 8))
        plt.imshow(REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND_original_ind_dilated, cmap='gray')
        plt.title("Redundant Pad Block Pair Dilated")
        plt.show()
        sio.savemat("pad_bitmap/REDUNDANT_MAIN_PAD_BLOCK_BITMAP_DILATED.mat", {"REDUNDANT_MAIN_PAD_BLOCK_BITMAP_DILATED": REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND_original_ind_dilated})
        plt.figure(figsize=(8, 8))
        plt.imshow(REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND_original_ind_dilated, cmap='gray')
        plt.title("Redundant Pad Block Pair Dilated")
        plt.show()
        sio.savemat("pad_bitmap/REDUNDANT_COPY_PAD_BLOCK_BITMAP_DILATED.mat", {"REDUNDANT_COPY_PAD_BLOCK_BITMAP_DILATED": REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND_original_ind_dilated})
        cross_bitmap_main_copy = np.logical_and(
            REDUNDANT_MAIN_PAD_BLOCK_BITMAP_EXPAND_original_ind_dilated,
            REDUNDANT_COPY_PAD_BLOCK_BITMAP_EXPAND_original_ind_dilated
        )
        # Draw the cross bitmap
        plt.figure(figsize=(8, 8))
        plt.imshow(cross_bitmap_main_copy, cmap='gray')
        plt.title("Cross Bitmap Main Copy")
        plt.show()
        sio.savemat("pad_bitmap/cross_bitmap_main_copy.mat", {"cross_bitmap_main_copy": cross_bitmap_main_copy})
        # Save the total critical area bitmap
        sio.savemat("pad_bitmap/TOTAL_CRITICAL_AREA_BITMAP.mat", {"TOTAL_CRITICAL_AREA_BITMAP": TOTAL_CRITICAL_AREA_BITMAP})
        # Draw the cross bitmap
        plt.figure(figsize=(8, 8))
        plt.imshow(TOTAL_CRITICAL_AREA_BITMAP, cmap='gray')
        plt.title("TOATAL CRITICAL AREA BITMAP")
        plt.show()
        np.save("pad_bitmap/TOTAL_CRITICAL_AREA_BITMAP.npy", TOTAL_CRITICAL_AREA_BITMAP)
        # sio.savemat("pad_bitmap/TOTAL_CRITICAL_AREA_BITMAP.mat", {"TOTAL_CRITICAL_AREA_BITMAP": TOTAL_CRITICAL_AREA_BITMAP})
        raise ValueError("Critical area is not correct.")

    # print("How many total critical area blocks are dilated:", np.sum(TOTAL_CRITICAL_AREA_BITMAP))
    # print("Critical area: {}", np.sum(TOTAL_CRITICAL_AREA_BITMAP) * PITCH_um**2 * pad_block_size**2)
    # Calculate the critical area
    critical_area_r_mv = np.sum(TOTAL_CRITICAL_AREA_BITMAP) * PITCH_um**2 * pad_block_size**2
    
    # print("Critical area of the main void:", critical_area_r_mv)
    # raise ValueError("Critical area figure drawing.")
    return critical_area_r_mv

