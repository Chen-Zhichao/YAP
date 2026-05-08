#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
import scipy.io as sio
import os
import hashlib
import json
from collections import defaultdict

def add_config_items(cfg, keys, values):
    """
    Add items to the configuration dictionary.

    Args:
        cfg (dict): Configuration dictionary.
        keys (list): List of keys to add.
        values (list): List of values corresponding to the keys.
    """
    if len(keys) != len(values):
        raise ValueError("Keys and values must have the same length.")

    for key, value in zip(keys, values):
        cfg[key] = value


def finalize_cfg_for_mode(cfg, ds_name: str, mode: str):
    """
    Apply mode-derived parameters after geometry/config fields are populated.
    """
    cfg.DESIGN = ds_name
    if mode == "w2w_simulation" or mode == "w2w_modeling":
        cfg.SYSTEM_MAGNIFICATION_MEAN_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_MEAN_um + cfg.M_0) / 1e6
        cfg.SYSTEM_MAGNIFICATION_STD_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_STD_um) ** 2 / 1e6
        cfg.S_INIT_A_M = 10e-6 * (cfg.WAF_R_um / 150000) ** 2
        cfg.S_INIT_B_M = 0.0
    elif mode == "d2w_simulation" or mode == "d2w_modeling":
        cfg.SYSTEM_MAGNIFICATION_MEAN_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_MEAN_um + cfg.M_0) / 1e6
        cfg.SYSTEM_MAGNIFICATION_STD_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_STD_um) ** 2 / 1e6
        cfg.eff_DIE_R = float(np.sqrt((cfg.DIE_W_um / 2) ** 2 + (cfg.DIE_L_um / 2) ** 2))  # Effective die radius (um)
        cfg.S_INIT_A_M = 10e-6 * (cfg.eff_DIE_R / 150000) ** 2
        cfg.S_INIT_B_M = 0.0
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes are 'w2w_simulation', 'w2w_modeling', 'd2w_simulation', and 'd2w_modeling'.")
    return cfg


def _sanitize_runtime_tag(value: str) -> str:
    safe = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def build_runtime_cache_tag(input_args: dict | None) -> str:
    input_args = input_args or {}
    ds_name = str(input_args.get("ds_name", ""))
    config_path = str(input_args.get("config", ""))
    config_stem = os.path.splitext(os.path.basename(config_path))[0] if config_path else "config"
    criticality_profile = str(input_args.get("criticality_profile", "default"))

    raw = f"{ds_name}__{config_stem}__{criticality_profile}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    ds_token = _sanitize_runtime_tag(ds_name)[:80]
    config_token = _sanitize_runtime_tag(config_stem)[:80]
    profile_token = _sanitize_runtime_tag(criticality_profile)[:40]

    parts = [part for part in (ds_token, config_token, profile_token, digest) if part]
    return "__".join(parts) if parts else digest


def get_runtime_temp_dir(cfg, input_args: dict | None) -> str:
    input_args = input_args or {}
    ds_name = str(input_args.get("ds_name", getattr(cfg, "DESIGN", cfg.INTERFACE)))
    temp_dir = os.path.join(cfg.OUTPUT_DIR, ds_name, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def get_sorted_bmap_copy_path(cfg, input_args: dict | None, bmap_path: str) -> str:
    temp_dir = get_runtime_temp_dir(cfg, input_args)
    bmap_stem = os.path.splitext(os.path.basename(bmap_path))[0]
    run_tag = build_runtime_cache_tag(input_args)
    return os.path.join(temp_dir, f"{bmap_stem}__sorted__{run_tag}.bmap")


def get_dishing_bound_cache_path(cfg, input_args: dict | None) -> str:
    temp_dir = get_runtime_temp_dir(cfg, input_args)
    run_tag = build_runtime_cache_tag(input_args)
    return os.path.join(temp_dir, f"{cfg.INTERFACE}_dishing_bound_array__{run_tag}.npy")


def is_cache_fresh(cache_path: str, source_paths: list[str]) -> bool:
    if not os.path.exists(cache_path):
        return False

    cache_mtime = os.path.getmtime(cache_path)
    for source_path in source_paths:
        if not source_path or not os.path.exists(source_path):
            continue
        if os.path.getmtime(source_path) > cache_mtime:
            return False
    return True


def _normalize_cache_value(value):
    if isinstance(value, dict):
        return {
            str(key): _normalize_cache_value(sub_value)
            for key, sub_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_cache_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return round(value, 12)
    return value


def _bitmap_collection_cache_signature(
    cfg,
    bmap_path: str,
    criticality_path: str,
    pad_arrange_pattern: str,
) -> str:
    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    signature_payload = {
        "cfg": _normalize_cache_value(cfg_container),
        "bmap_path": os.path.abspath(bmap_path),
        "bmap_mtime_ns": os.stat(bmap_path).st_mtime_ns,
        "bmap_size": os.stat(bmap_path).st_size,
        "criticality_path": os.path.abspath(criticality_path),
        "criticality_mtime_ns": os.stat(criticality_path).st_mtime_ns,
        "criticality_size": os.stat(criticality_path).st_size,
        "pad_arrange_pattern": str(pad_arrange_pattern),
    }
    payload_json = json.dumps(
        signature_payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload_json.encode("utf-8")).hexdigest()


def atomic_save_npy(path: str, array) -> None:
    temp_path = f"{path}.tmp.{os.getpid()}.npy"
    np.save(temp_path, array)
    os.replace(temp_path, path)


def cleanup_runtime_temp_files(cfg_dict: dict, input_args: dict | None) -> list[str]:
    input_args = input_args or {}
    if not cfg_dict:
        return []

    run_tag = build_runtime_cache_tag(input_args)
    removed_paths = []
    seen_temp_dirs = set()

    for cfg in cfg_dict.values():
        temp_dir = get_runtime_temp_dir(cfg, input_args)
        if temp_dir in seen_temp_dirs:
            continue
        seen_temp_dirs.add(temp_dir)
        if not os.path.isdir(temp_dir):
            continue

        for name in os.listdir(temp_dir):
            if run_tag not in name:
                continue
            if ".tmp." not in name:
                continue
            file_path = os.path.join(temp_dir, name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                removed_paths.append(file_path)

        if not os.listdir(temp_dir):
            os.rmdir(temp_dir)

    return removed_paths


def _ensure_bitmap_collection_group_arrays(bitmap_collection: dict) -> tuple[dict, bool]:
    required_keys = (
        "redundant_group_id_per_pad",
        "redundant_tolerated_esd_failures",
        "redundant_tolerated_mechanical_failures",
    )
    if all(key in bitmap_collection for key in required_keys):
        return bitmap_collection, False

    pad_coords = np.asarray(bitmap_collection["pad_coords"])
    total_pad_count = int(pad_coords.shape[0])
    redundant_group_id_per_pad = np.full(total_pad_count, -1, dtype=np.int32)

    redundant_net_to_1d_physical_mask = bitmap_collection["redundant_net_to_1d_physical_mask"]
    criticality_info = bitmap_collection["criticality_info"]
    group_names = list(redundant_net_to_1d_physical_mask.keys())

    tolerated_esd_failures = np.zeros(len(group_names), dtype=np.int32)
    tolerated_mechanical_failures = np.zeros(len(group_names), dtype=np.int32)

    for group_id, net in enumerate(group_names):
        physical_mask = np.asarray(
            redundant_net_to_1d_physical_mask[net],
            dtype=np.int64,
        ).reshape(-1)
        physical_mask = physical_mask[physical_mask >= 0]
        if physical_mask.size > 0:
            redundant_group_id_per_pad[physical_mask] = group_id

        net_info = criticality_info[net]
        tolerated_esd_failures[group_id] = int(net_info["tolerated_esd_failures"])
        tolerated_mechanical_failures[group_id] = int(
            net_info["tolerated_mechanical_failures"]
        )

    bitmap_collection["redundant_group_id_per_pad"] = redundant_group_id_per_pad
    bitmap_collection["redundant_tolerated_esd_failures"] = tolerated_esd_failures
    bitmap_collection["redundant_tolerated_mechanical_failures"] = (
        tolerated_mechanical_failures
    )
    return bitmap_collection, True

def _upsample_pad_yield_map(pad_yield_map: np.ndarray,
                            pad_map_shape,
                            pad_yield_map_sub_factor: int) -> np.ndarray:
    """
    Upsample a subsampled pad yield map back to the full pad-array shape.

    The sampling grid follows the same endpoint-preserving indexing used in the
    overlay/defect calculators, so we reconstruct the dense map with 1D linear
    interpolation along columns and then rows.
    """
    if pad_yield_map.shape == pad_map_shape or pad_yield_map_sub_factor <= 1:
        return pad_yield_map

    target_rows, target_cols = pad_map_shape
    src_rows, src_cols = pad_yield_map.shape

    row_coords = np.round(np.linspace(0, target_rows - 1, src_rows)).astype(np.float64)
    col_coords = np.round(np.linspace(0, target_cols - 1, src_cols)).astype(np.float64)

    # Guard against duplicate coordinates in very small arrays.
    col_coords, unique_col_idx = np.unique(col_coords, return_index=True)
    pad_yield_map = pad_yield_map[:, unique_col_idx]
    row_coords, unique_row_idx = np.unique(row_coords, return_index=True)
    pad_yield_map = pad_yield_map[unique_row_idx, :]

    full_col_coords = np.arange(target_cols, dtype=np.float64)
    full_row_coords = np.arange(target_rows, dtype=np.float64)

    if pad_yield_map.shape[1] == 1:
        col_upsampled = np.repeat(pad_yield_map, target_cols, axis=1)
    else:
        col_upsampled = np.vstack([
            np.interp(full_col_coords, col_coords, row_vals)
            for row_vals in pad_yield_map
        ])

    if col_upsampled.shape[0] == 1:
        return np.repeat(col_upsampled, target_rows, axis=0)

    full_pad_yield_map = np.vstack([
        np.interp(full_row_coords, row_coords, col_upsampled[:, col_ind])
        for col_ind in range(target_cols)
    ]).T

    return full_pad_yield_map

def get_config_dict(cfg_folder: str,
                    cfg_skeleton: str,
                    ds_name: str,
                    input_ds_dir: str,
                    _3dbv_path: str,
                    _3dbx_path: str,
                    mode: str,
                    debug=False,
                    file_suffix: str = "") -> dict:
    """
    Load base configuration from a YAML file and update with .3dbv and .bmap design parameters.
    args:
        cfg_folder: folder path of the config files
        cfg_skeleton: base config yaml file
        ds_name: design name
        input_ds_dir: input design directory
        _3dbv_path: path to .3dbv file
        mode: mode to load from config (w2w_simulation, w2w_modeling, d2w_simulation, d2w_modeling)
        debug: whether to enable debug output
    returns:
        cfg_dict: dictionary of configuration objects for each stack layer
    """
    cfg_dict = update_config_with_3dblox_params(cfg_skeleton=cfg_skeleton,
                                                input_ds_dir=input_ds_dir,
                                                _3dbv_path=_3dbv_path,
                                                _3dbx_path=_3dbx_path,)
    suffix = file_suffix or ""
    for interface_name, cfg in cfg_dict.items():
        cfg = finalize_cfg_for_mode(cfg, ds_name=ds_name, mode=mode)
        # Save updated config file for reference
        OmegaConf.save(cfg, cfg_folder + f"/{interface_name}{suffix}.yaml")

    if debug:
        cfg.DEBUG = True
        print("Configuration loaded:")
        print(OmegaConf.to_yaml(cfg))



    return cfg_dict


def get_single_interface_config_dict(cfg_folder: str,
                                     cfg_skeleton: object,
                                     ds_name: str,
                                     input_ds_dir: str,
                                     mode: str,
                                     debug=False,
                                     file_suffix: str = "") -> dict:
    """
    Legacy single-interface mode for designs that provide one .bmap directly
    from the config without 3dblox wrapper files.
    """
    cfg = cfg_skeleton.copy()
    if not getattr(cfg, "INTERFACE", None):
        raise ValueError(
            "Single-interface mode requires INTERFACE to be set in the config."
        )

    bmap_path = os.path.join(input_ds_dir, f"{cfg.INTERFACE}.bmap")
    if not os.path.exists(bmap_path):
        raise FileNotFoundError(f"Bump map not found at {bmap_path}")

    if getattr(cfg, "PAD_ARR_ROW", None) in (None, "None") or getattr(cfg, "PAD_ARR_COL", None) in (None, "None"):
        update_config_from_bmap(
            cfg,
            bmap_path,
            y_tol=cfg.PITCH_r_um * 0.1,
            x_tol=cfg.PITCH_c_um * 0.1,
        )
    if getattr(cfg, "PAD_ARR_L_um", None) in (None, "None") or getattr(cfg, "PAD_ARR_W_um", None) in (None, "None"):
        add_config_items(
            cfg,
            keys=["PAD_ARR_L_um", "PAD_ARR_W_um"],
            values=[
                (cfg.PAD_ARR_ROW - 1) * cfg.PITCH_r_um,
                (cfg.PAD_ARR_COL - 1) * cfg.PITCH_c_um,
            ],
        )

    cfg = finalize_cfg_for_mode(cfg, ds_name=ds_name, mode=mode)
    suffix = file_suffix or ""
    OmegaConf.save(cfg, cfg_folder + f"/{cfg.INTERFACE}{suffix}.yaml")

    if debug:
        cfg.DEBUG = True
        print("Configuration loaded:")
        print(OmegaConf.to_yaml(cfg))

    return {cfg.INTERFACE: cfg}



def update_config_from_bmap(cfg, blox_bmap_path, y_tol=0.1, x_tol=0.1):
    """
    Extract pad array layout from .bmap file.

    args:
        cfg: configuration object
        blox_bmap_path: path to .bmap file
        y_tol: tolerance for clustering y coordinates (um), if the difference between two y coordinates is less than y_tol, they are considered in the same row
        x_tol: tolerance for clustering x coordinates (um), if the difference between two x coordinates is less than x_tol, they are considered in the same column
    """
    coords = []

    with open(blox_bmap_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                x, y = float(parts[2]), float(parts[3])
                coords.append((x, y))
            except ValueError:
                continue

    if not coords:
        print("No valid pad coordinates found in the .bmap file.")
        return

    coords = np.array(coords)

    # Rank by y descending for stable clustering statistics.
    coords = coords[np.argsort(-coords[:, 1])]

    # Cluster occupied rows/columns as a sanity signal, but derive the logical
    # array size from the geometric bounding box and pitch. This is more robust
    # for sparse checkerboard footprints such as HBM, where the first row may
    # not contain every logical column.
    y_vals = []
    for y in coords[:, 1]:
        if not y_vals or abs(y - y_vals[-1]) > y_tol:
            y_vals.append(y)
    clustered_rows = len(y_vals)

    x_vals = []
    for x in np.sort(coords[:, 0]):
        if not x_vals or abs(x - x_vals[-1]) > x_tol:
            x_vals.append(x)
    clustered_cols = len(x_vals)

    y_min = float(np.min(coords[:, 1]))
    y_max = float(np.max(coords[:, 1]))
    x_min = float(np.min(coords[:, 0]))
    x_max = float(np.max(coords[:, 0]))

    bbox_rows = int(round((y_max - y_min) / cfg.PITCH_r_um)) + 1
    bbox_cols = int(round((x_max - x_min) / cfg.PITCH_c_um)) + 1

    num_rows = max(clustered_rows, bbox_rows)
    num_cols = max(clustered_cols, bbox_cols)

    add_config_items(cfg, keys=['PAD_ARR_ROW', 'PAD_ARR_COL'], values=[num_rows, num_cols])
    add_config_items(cfg, keys=['PAD_ARR_L_um', 'PAD_ARR_W_um'],
                        values=[(num_rows - 1) * cfg.PITCH_r_um,
                                (num_cols - 1) * cfg.PITCH_c_um])


def _instance_from_3dbx_endpoint(endpoint) -> str:
    return str(endpoint).split(".regions.")[0]


def _interface_from_3dbx_connection(connection) -> str:
    interface_top = str(((connection.bot).split(".")[-1]).split("To_")[-1])
    interface_bot = str(((connection.top).split(".")[-1]).split("From_")[-1])
    return f"{interface_top}_From_{interface_bot}"


def stack_graph_from_3dbx(_3dbx_path: str) -> list[dict]:
    """
    Parse a 3Dblox stack config and return root-to-leaf substacks.

    Returns a list because 2.5D designs can have several lateral substacks on
    the same root/interposer, while a pure 3D stack usually has one path.
    Each item has the form:
        {
            "substack_id": int,
            "chiplets_bottom_to_top": [reference names],
            "interfaces_bottom_to_top": [cfg_dict-compatible interface names],
        }
    """
    stack_config_3dbx = OmegaConf.load(_3dbx_path)
    if "Stack" not in stack_config_3dbx or "ChipletInst" not in stack_config_3dbx:
        raise ValueError(f"{_3dbx_path} must contain Stack and ChipletInst sections.")

    stack = stack_config_3dbx.Stack
    chiplet_inst = stack_config_3dbx.ChipletInst
    stack_order = {name: idx for idx, name in enumerate(stack.keys())}
    children_by_parent = defaultdict(list)
    parent_by_child = {}

    if "Connection" in stack_config_3dbx:
        for connection_order, (connection_name, connection) in enumerate(stack_config_3dbx.Connection.items()):
            top_instance = _instance_from_3dbx_endpoint(connection.top)
            bot_instance = _instance_from_3dbx_endpoint(connection.bot)
            if top_instance not in stack:
                raise ValueError(f"Connection {connection_name} top instance '{top_instance}' is not in Stack.")
            if bot_instance not in stack:
                raise ValueError(f"Connection {connection_name} bottom instance '{bot_instance}' is not in Stack.")
            if top_instance in parent_by_child:
                raise ValueError(
                    f"Instance '{top_instance}' has multiple bottom parents; "
                    "substack extraction expects a tree/DAG of vertical paths."
                )

            edge = {
                "child": top_instance,
                "connection": str(connection_name),
                "interface": _interface_from_3dbx_connection(connection),
                "order": connection_order,
            }
            children_by_parent[bot_instance].append(edge)
            parent_by_child[top_instance] = bot_instance

    roots = [name for name, node in stack.items() if bool(node.get("root", False))]
    if not roots:
        roots = [name for name in stack.keys() if name not in parent_by_child]
    if not roots:
        raise ValueError(f"No stack root could be inferred from {_3dbx_path}.")

    for parent in children_by_parent:
        children_by_parent[parent].sort(key=lambda edge: edge["order"])
    roots.sort(key=lambda name: stack_order[name])

    substacks = []

    def dfs(instance_name: str, chiplet_path: list[str], interface_path: list[str], active_path: set[str]):
        if instance_name in active_path:
            cycle = " -> ".join(list(active_path) + [instance_name])
            raise ValueError(f"Cycle detected in stack graph: {cycle}")
        if instance_name not in chiplet_inst:
            raise ValueError(f"Stack instance '{instance_name}' is not in ChipletInst.")

        next_chiplet_path = chiplet_path + [str(chiplet_inst[instance_name].reference)]
        child_edges = children_by_parent.get(instance_name, [])
        if not child_edges:
            substacks.append({
                "substack_id": len(substacks),
                "chiplets_bottom_to_top": next_chiplet_path,
                "interfaces_bottom_to_top": interface_path,
            })
            return

        next_active_path = set(active_path)
        next_active_path.add(instance_name)
        for edge in child_edges:
            dfs(
                edge["child"],
                next_chiplet_path,
                interface_path + [edge["interface"]],
                next_active_path,
            )

    for root in roots:
        dfs(root, [], [], set())

    return substacks


def update_config_with_3dblox_params(cfg_skeleton: object,
                                    input_ds_dir: str,
                                    _3dbv_path: str,
                                    _3dbx_path: str,):
    """
    Update configuration with design parameters from .3dbv and .bmap files.
    args:
        cfg_skeleton: configuration object skeleton
        input_ds_dir: path to design input files directory
        _3dbv_path: path to .3dbv file (chiplet definitions)
        _3dbx_path: path to .3dbx file (stack configuration)
        _bmap_path: path to .bmap file (bump map)
    file structure:
        input_ds_dir/
          |-  xx_chiplet_definitions.3dbv
          |-  xx_stack_config.3dbx
          |-  XX_From_XX.bmap
          |-  XX.3dbf
          |-  XX_From_XX_criticality.txt
    """
    ### Update cfg_list with design parameters from .3dbv and .bmap files
    cfg_dict = dict()
    stack_config_3dbx = OmegaConf.load(_3dbx_path)

    for _, connection in stack_config_3dbx.Connection.items():
        cfg = cfg_skeleton.copy()

        # Extract interface names
        cfg.INTERFACE_TOP = str(((connection.bot).split('.')[-1]).split('To_')[-1])
        cfg.INTERFACE_BOT = str(((connection.top).split('.')[-1]).split('From_')[-1])
        cfg.INTERFACE = f"{cfg.INTERFACE_TOP}_From_{cfg.INTERFACE_BOT}"

        ### Read .3dbv, .3dbx, and .bmap files
        ## Extract design parameters from .3dbv and .3dbf file
        _3dbv = OmegaConf.load(_3dbv_path)
        _bmap_path = os.path.join(input_ds_dir, f"{cfg.INTERFACE}.bmap")
        top_3dbf_path = os.path.join(input_ds_dir, f"{cfg.INTERFACE_TOP}.3dbf")
        bot_3dbf_path = os.path.join(input_ds_dir, f"{cfg.INTERFACE_BOT}.3dbf")
        top_3dbf = OmegaConf.load(top_3dbf_path)
        bot_3dbf = OmegaConf.load(bot_3dbf_path)

        # Check unit
        assert _3dbv.Header.unit == 'micron', "Only support .3dbv file with unit in microns."

        # Read die width and length
        add_config_items(cfg, keys=['DIE_W_um', 'DIE_L_um'],
                        values=[float(_3dbv.ChipletDef[cfg.INTERFACE_TOP].design_area[0]),
                                float(_3dbv.ChipletDef[cfg.INTERFACE_TOP].design_area[1])])

        # Read bump size, size/2 = radius. Find matching bum type
        bump_type_list = list(top_3dbf.Bump_Types.keys())   # silicon_individual_bonding, organic_individual_bonding, ...
        selected_bump_type = None

        with open(_bmap_path, 'r') as f:
            first_line = f.readline()
            for bump_type in bump_type_list:
                if bump_type in first_line.split()[1]:
                    selected_bump_type = bump_type
                    break

        if selected_bump_type is None:
            raise ValueError(f"No matching bump type found in {_bmap_path} for top chiplet {cfg.INTERFACE_TOP}.")

        add_config_items(cfg, keys=['PAD_TOP_R_um', 'PAD_BOT_R_um'],
                            values=[float(top_3dbf.Bump_Types[selected_bump_type].bump_size) / 2,
                                    float(bot_3dbf.Bump_Types[selected_bump_type].bump_size) / 2])

        # Read pad pitch. Prefer explicit row/column pitches when available;
        # otherwise fall back to the legacy scalar pitch field.
        chiplet_grid = top_3dbf.Chiplet_Grid
        if 'pitch_r' in chiplet_grid and 'pitch_c' in chiplet_grid:
            pitch_r = float(chiplet_grid.pitch_r)
            pitch_c = float(chiplet_grid.pitch_c)
        elif 'pitch' in chiplet_grid:
            pitch_r = float(chiplet_grid.pitch)
            pitch_c = float(chiplet_grid.pitch)
        else:
            raise ValueError(
                f"{top_3dbf_path} must define Chiplet_Grid.pitch or both "
                "Chiplet_Grid.pitch_r and Chiplet_Grid.pitch_c."
            )

        add_config_items(cfg, keys=['PITCH_r_um', 'PITCH_c_um'],
                            values=[pitch_r, pitch_c])

        add_config_items(cfg, keys=['ITF_TOP_THICK_um', 'ITF_BOT_THICK_um'],
                            values=[float(_3dbv.ChipletDef[cfg.INTERFACE_TOP].thickness),
                                    float(_3dbv.ChipletDef[cfg.INTERFACE_BOT].thickness)])

        ## Extract design parameters from .bmap file
        update_config_from_bmap(cfg, _bmap_path,
                                y_tol=cfg.PITCH_r_um * 0.1, x_tol=cfg.PITCH_c_um * 0.1)

        # Store in config dictionary
        cfg_dict[cfg.INTERFACE] = cfg

    return cfg_dict






def draw_pad_bitmap(cfg, bitmap_collection, output_path):
    # Draw the critical and redundant pad bitmaps in one figure (critical light red, redundant light blue, dummy light gray)
    CRITICAL_PAD_BITMAP = bitmap_collection["CRITICAL_PAD_BITMAP"]
    REDUNDANT_PAD_BITMAP = bitmap_collection["REDUNDANT_PAD_BITMAP"]
    DUMMY_PAD_BITMAP = bitmap_collection["DUMMY_PAD_BITMAP"]
    ## Use legend to show the color
    PAD_BITMAP = np.zeros_like(CRITICAL_PAD_BITMAP, dtype=int)

    PAD_BITMAP[CRITICAL_PAD_BITMAP == 1] = 1  # red
    PAD_BITMAP[REDUNDANT_PAD_BITMAP == 1] = 2  # blue
    PAD_BITMAP[DUMMY_PAD_BITMAP == 1] = 3  # green
    # Remaining zeros are non-pad areas
    PAD_BITMAP[PAD_BITMAP == 0] = 4  # non-pad (light gray)

    fig = plt.figure(figsize=(10, 10))
    cmap = ListedColormap([
        (1.0, 0.5, 0.5),    # 1 - critical (medium red)
        (0.4, 0.4, 0.9),    # 2 - redundant (medium blue)
        (0.0, 0.6, 0.0),    # 3 - dummy (medium green)
        (0.9, 0.9, 0.9),    # 4 - non-pad (light gray)
    ])
    red_patch = patches.Patch(color=(1.0, 0.5, 0.5), label='Critical Pads')
    blue_patch = patches.Patch(color=(0.4, 0.4, 0.9), label='Redundant Pads')
    green_patch = patches.Patch(color=(0.0, 0.6, 0.0), label='Dummy Pads')
    light_gray_patch = patches.Patch(color=(0.9, 0.9, 0.9), label='Non-Pad Areas')
    # plt.legend(
    #     handles=[red_patch, blue_patch, green_patch, light_gray_patch],
    #     loc='upper center',
    #     bbox_to_anchor=(0.5, -0.07),
    #     ncol=4,
    #     frameon=False
    # )
    # plt.legend().set_visible(False)
    norm = BoundaryNorm(boundaries=[0.5, 1.5, 2.5, 3.5, 4.5], ncolors=cmap.N)
    plt.axis('off')
    plt.imshow(PAD_BITMAP, cmap=cmap, norm=norm)
    # plt.title("Pad Block Bitmap")


    # Save the pad bitmaps
    plt.savefig(os.path.join(output_path, cfg.INTERFACE + "_pad_bitmap.png"))
    plt.close(fig)
    # print("Pad bitmap collections info saved.")
    return



def sort_pads_bmap(input_path, output_path):
    """
    Read pad data from .bmap file, from top-left to right-bottom order
    sorted by x ascending and y descending.
    - x is the 3rd column (index 2)
    - y is the 4th column (index 3)
    """
    if is_cache_fresh(output_path, [input_path]):
        return

    pads = []
    with open(input_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:  # 至少需要 x, y 两列
                continue
            try:
                x = float(parts[2])  # 第3列是x
                y = float(parts[3])  # 第4列是y
                pads.append((x, y, line.strip()))
            except ValueError:
                continue  # 跳过无法解析的行

    if not pads:
        raise ValueError(f"No valid bump records found in {input_path}")

    # Transform to numpy array for sorting
    data = np.array(pads, dtype=object)

    # x ascending, y descending
    idx = np.lexsort((data[:,0].astype(float), - data[:,1].astype(float)))
    sorted_data = data[idx]

    temp_output_path = f"{output_path}.tmp.{os.getpid()}"
    with open(temp_output_path, 'w') as f:
        for _, _, line in sorted_data:
            f.write(line + '\n')
    os.replace(temp_output_path, output_path)

    # print(f"Sorted the order as from top-left to right-bottom and saved in {output_path}")


def criticality_generator(cfg,
                          bump_data: list,
                          redundant_net_to_bumpids: dict,
                        ):
    '''
    Criticality file output format:
    <port> <esd_criticality> <mechanical_criticality>
    '''
    bump_criticality = list()
    bump_set = set()
    for bump in bump_data:
        port = bump['port']
        net = bump['net']
        if (bump['net'], port) in bump_set:
            continue
        if 'dummy' in net.lower():
            esd_criticality = 0.0
            mechanical_criticality = 0.0
        else:
            num_copies = len(redundant_net_to_bumpids[bump['net']])
            mechanical_criticality = 1.0 / num_copies
            esd_criticality = 1.0 / num_copies

        bump_criticality.append({
            "port": port,
            "esd_criticality": esd_criticality,
            "mechanical_criticality": mechanical_criticality
        })
        bump_set.add((bump['net'], port))
    with open(cfg.OUTPUT_DIR + cfg.INTERFACE + "/" + cfg.INTERFACE + "_criticality.txt", 'w') as f:
        for bump_crit in bump_criticality:
            f.write(f"{bump_crit['port']} {bump_crit['esd_criticality']:.6f} {bump_crit['mechanical_criticality']:.6f}\n")
    print("Criticality file saved in ", cfg.OUTPUT_DIR + cfg.INTERFACE + "/" + cfg.INTERFACE + "_criticality.txt")
    return


def risk_map_generator(cfg,
                    interface: object,
                    input_args
                    ):
    '''
    Risk map output format:
    <pad_coords_x> <pad_coords_y> <esd_failure_probability> <overlay_failure_probability> <particle_failure_probability> <mechanical_failure_probability>
    '''
    file_suffix = str(input_args.get("output_file_tag", ""))
    if not file_suffix:
        config_path = str(input_args.get("config", ""))
        criticality_profile = str(input_args.get("criticality_profile", "default"))
        if config_path:
            config_stem = os.path.splitext(os.path.basename(config_path))[0]
            safe_parts = []
            for ch in f"{config_stem}__{criticality_profile}":
                if ch.isalnum() or ch in ("-", "_"):
                    safe_parts.append(ch)
                else:
                    safe_parts.append("_")
            safe_tag = "".join(safe_parts).strip("_")
            if safe_tag:
                file_suffix = f"__{safe_tag}"

    def append_file_suffix(filename: str) -> str:
        if not file_suffix:
            return filename
        stem, ext = os.path.splitext(filename)
        return f"{stem}{file_suffix}{ext}"

    output_dir = os.path.join(cfg.OUTPUT_DIR, input_args['ds_name'], cfg.INTERFACE)
    risk_map_path = os.path.join(output_dir, append_file_suffix(f"{cfg.INTERFACE}_risk.map"))
    pad_coords = np.asarray(interface.pad_coords, dtype=np.float64)
    valid_mask = np.isfinite(pad_coords[:, 0]) & np.isfinite(pad_coords[:, 1])
    risk_map = np.column_stack(
        (
            pad_coords[valid_mask, 0],
            pad_coords[valid_mask, 1],
            1.0 - np.asarray(interface.pad_yield_map['Y_esd'], dtype=np.float64).reshape(-1)[valid_mask],
            1.0 - np.asarray(interface.pad_yield_map['Y_ovl'], dtype=np.float64).reshape(-1)[valid_mask],
            1.0 - np.asarray(interface.pad_yield_map['Y_df'], dtype=np.float64).reshape(-1)[valid_mask],
            1.0 - np.asarray(interface.pad_yield_map['Y_ce'], dtype=np.float64).reshape(-1)[valid_mask],
        )
    )
    np.savetxt(
        risk_map_path,
        risk_map,
        fmt=["%.6f", "%.6f", "%.12f", "%.12f", "%.12f", "%.12f"],
    )
    print("Risk map file saved in ", risk_map_path)

    mechanism_specs = {
        "esd": ("Y_esd", "ESD Failure Probability"),
        "overlay": ("Y_ovl", "Overlay Failure Probability"),
        "particle": ("Y_df", "Particle Failure Probability"),
        "mechanical": ("Y_ce", "Mechanical Failure Probability"),
        "overall": ("Y_bond", "Overall Failure Probability"),
    }
    for mechanism, (yield_key, colorbar_label) in mechanism_specs.items():
        failure_map = 1.0 - np.asarray(interface.pad_yield_map[yield_key], dtype=np.float64)
        masked_failure_map = np.ma.masked_invalid(failure_map)

        finite_vals = failure_map[np.isfinite(failure_map)]
        vmin = float(np.min(finite_vals)) if finite_vals.size > 0 else 0.0
        vmax = float(np.max(finite_vals)) if finite_vals.size > 0 else 1.0
        if vmax <= 0.0:
            vmax = 1.0

        fig, ax = plt.subplots(figsize=(8, 6))
        extent = None
        x_label = 'Pad Column Index'
        y_label = 'Pad Row Index'
        pitch_c = getattr(cfg, "PITCH_c_um", None)
        pitch_r = getattr(cfg, "PITCH_r_um", None)
        if pitch_c not in (None, "None") and pitch_r not in (None, "None"):
            cols = masked_failure_map.shape[1]
            rows = masked_failure_map.shape[0]
            half_w = (cols - 1) * float(pitch_c) / 2.0
            half_h = (rows - 1) * float(pitch_r) / 2.0
            extent = [-half_w, half_w, half_h, -half_h]
            x_label = 'X (um)'
            y_label = 'Y (um)'

        image = ax.imshow(
            masked_failure_map,
            cmap='viridis',
            interpolation='nearest',
            vmin=vmin,
            vmax=vmax,
            origin='upper',
            extent=extent,
        )
        fig.colorbar(image, ax=ax, label=colorbar_label)
        ax.set_title(f"{mechanism.title()} Risk Map")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        save_path = os.path.join(
            output_dir,
            append_file_suffix(f"{cfg.INTERFACE}_{mechanism}_risk_map.png"),
        )
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print("Failure mechanism risk maps saved in ", output_dir)
    print()

    return




def convert_3dblox_to_pad_bitmap(cfg,
                                 _bmap_path: str,
                                 criticality_path: str,
                                 pad_arrange_pattern: str,
                                 input_args=None):
    '''
    pad_arrange_pattern: 'checkerboard' for UCIe standard and HBM
    '''
    # Create output directory if not exist
    input_args = input_args or {}
    ds_name = str(input_args.get('ds_name', getattr(cfg, 'DESIGN', cfg.INTERFACE)))
    output_path = os.path.join(cfg.OUTPUT_DIR, ds_name, cfg.INTERFACE)
    os.makedirs(output_path, exist_ok=True)
    bitmap_collection_path = os.path.join(
        output_path,
        f"{cfg.INTERFACE}_bitmap_collection.npy",
    )
    cache_signature = _bitmap_collection_cache_signature(
        cfg=cfg,
        bmap_path=_bmap_path,
        criticality_path=criticality_path,
        pad_arrange_pattern=pad_arrange_pattern,
    )

    if is_cache_fresh(bitmap_collection_path, [_bmap_path, criticality_path]):
        bitmap_collection = np.load(bitmap_collection_path, allow_pickle=True).item()
        if bitmap_collection.get("_cache_signature") == cache_signature:
            bitmap_collection, cache_updated = _ensure_bitmap_collection_group_arrays(
                bitmap_collection
            )
            if cache_updated:
                bitmap_collection["_cache_signature"] = cache_signature
                np.save(bitmap_collection_path, bitmap_collection)

            pad_bitmap_path = os.path.join(output_path, f"{cfg.INTERFACE}_pad_bitmap.png")
            if not os.path.exists(pad_bitmap_path):
                draw_pad_bitmap(cfg, bitmap_collection, output_path)
            return bitmap_collection

    sorted_bmap_path = get_sorted_bmap_copy_path(cfg, input_args, _bmap_path)
    sort_pads_bmap(_bmap_path, sorted_bmap_path)

    # Read the bump data from the .bmap file
    bump_data = []
    # Initialize the pad array boundaries
    [pad_array_left, pad_array_right, pad_array_top, pad_array_bottom] = [float('inf'), float('-inf'), float('-inf'), float('inf')]
    with open(sorted_bmap_path, 'r') as f:
        bumpid = 0
        for line in f:
            parts = line.strip().split()
            if len(parts) == 6:
                instance, bump_type, x, y, port, net = parts
                bump_data.append({      # From the top-left corner to the bottom-right corner
                    "bumpid": bumpid,
                    "x": float(x),
                    "y": float(y),
                    "port": port,
                    "net": net
                })
                if float(x) < pad_array_left:
                    pad_array_left = float(x)
                if float(x) > pad_array_right:
                    pad_array_right = float(x)
                if float(y) < pad_array_bottom:
                    pad_array_bottom = float(y)
                if float(y) > pad_array_top:
                    pad_array_top = float(y)
                bumpid += 1

    # Record the 1D physical locations of each pad in redundant nets
    '''Example: {NC: [0, 5, 10], VSS: [1, 6, 11], VDD: [2, 7, 12], ...}'''
    redundant_net_to_1d_physical_mask = dict()
    # Record the bump ids of each pad in redundant nets, bump id is the index in bump_data list
    redundant_net_to_bumpids = dict()

    for bump in bump_data:
        if bump['net'] not in redundant_net_to_bumpids:
            redundant_net_to_bumpids[bump['net']] = set()
            redundant_net_to_1d_physical_mask[bump['net']] = []
        redundant_net_to_bumpids[bump['net']].add(bump['bumpid'])

    # Generate the criticality map
    '''
    Current Format: <net1> [net2] [net3] ... <group_size> <tolerated_esd_failures> <tolerated_mechanical_failures>

    Where:
    - group_size: Total number of pads/bumps in the redundancy group
    - tolerated_esd_failures: Number of ESD failures the group can tolerate before failing
    - tolerated_mechanical_failures: Number of mechanical failures the group can tolerate before failing

    Criticality values are calculated when reading the file:
    - esd_criticality = (group_size - tolerated_esd_failures) / group_size
    - mechanical_criticality = (group_size - tolerated_mechanical_failures) / group_size
    '''
    # criticality_generator(cfg, bump_data, redundant_net_to_bumpids)
    criticality_info = dict()
    with open(criticality_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            net, num_copy, tolerated_esd_failures, tolerated_mechanical_failures = parts
            criticality_info[net] = {
                "tolerated_esd_failures": int(tolerated_esd_failures),
                "tolerated_mechanical_failures": int(tolerated_mechanical_failures)
            }


    # Initialize the pad bitmap
    # TODO: You need to modify the simulator to support different pad arrangement patterns
    CRITICAL_PAD_BITMAP = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), dtype=bool)
    REDUNDANT_PAD_BITMAP = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), dtype=bool)
    DUMMY_PAD_BITMAP = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), dtype=bool)
    ESD_CRITICAL_PAD_BITMAP = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), dtype=bool)
    pad_coords = np.full((cfg.PAD_ARR_ROW * cfg.PAD_ARR_COL, 2), np.nan, dtype=np.float32)  # x, y coordinates of each bump
    # Build a mapping array from physical bump location (r, c) to bump id
    mapping_physical_to_bumpid = np.full((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), np.nan, dtype=np.float32) # Shape: (PAD_ARR_ROW, PAD_ARR_COL)

    if pad_arrange_pattern in ('checkerboard', 'rectangular'):
        for bump in bump_data:
            x = bump['x']
            y = bump['y']
            row = int(round((pad_array_top - y ) / (cfg.PITCH_r_um)))   # Because in checkerboard pattern, the pitch per row is halved
            col = int(round((x - pad_array_left) / (cfg.PITCH_c_um)))   # Because in checkerboard pattern, the pitch per column is halved
            if not (0 <= row < cfg.PAD_ARR_ROW) or not (0 <= col < cfg.PAD_ARR_COL):
                raise IndexError(
                    f"Pad indexing out of bounds for interface {cfg.INTERFACE}: "
                    f"bumpid={bump['bumpid']} net={bump['net']} x={x} y={y} "
                    f"-> row={row}, col={col}, "
                    f"shape=({cfg.PAD_ARR_ROW}, {cfg.PAD_ARR_COL}), "
                    f"pitch=({cfg.PITCH_r_um}, {cfg.PITCH_c_um}), "
                    f"bbox(left={pad_array_left}, right={pad_array_right}, "
                    f"top={pad_array_top}, bottom={pad_array_bottom})"
                )
            mapping_physical_to_bumpid[row, col] = bump['bumpid']
            pad_coords[row * cfg.PAD_ARR_COL + col, 0] = bump['x'] - (pad_array_left + pad_array_right) / 2
            pad_coords[row * cfg.PAD_ARR_COL + col, 1] = bump['y'] - (pad_array_top + pad_array_bottom) / 2
            current_bump_net = bump['net']
            num_copies = len(redundant_net_to_bumpids[current_bump_net])
            if 'dummy' in current_bump_net.lower():
                DUMMY_PAD_BITMAP[row, col] = 1
                continue
            if num_copies == 1:
                CRITICAL_PAD_BITMAP[row, col] = 1
                ESD_CRITICAL_PAD_BITMAP[row, col] = 1
                redundant_net_to_bumpids.pop(current_bump_net, None)
                redundant_net_to_1d_physical_mask.pop(current_bump_net, None)
                continue
            elif num_copies > 1:
                REDUNDANT_PAD_BITMAP[row, col] = 1
                ESD_CRITICAL_PAD_BITMAP[row, col] = 1 if criticality_info[current_bump_net]['tolerated_esd_failures'] == 0 else 0
                redundant_net_to_1d_physical_mask[bump['net']].append(
                    row * cfg.PAD_ARR_COL + col
                )
                continue
    else:
        raise NotImplementedError("Currently only support checkerboard pad arrangement pattern.")

    redundant_net_to_1d_physical_mask = {
        net: np.asarray(physical_mask, dtype=np.int32)
        for net, physical_mask in redundant_net_to_1d_physical_mask.items()
    }

    # Count the number of pads
    num_critical_pads = np.sum(CRITICAL_PAD_BITMAP)
    num_redundant_pads = np.sum(REDUNDANT_PAD_BITMAP)
    num_dummy_pads = 0 if DUMMY_PAD_BITMAP is None else np.sum(DUMMY_PAD_BITMAP)

    bitmap_collection = {}
    bitmap_collection["bump_data"] = bump_data
    bitmap_collection["CRITICAL_PAD_BITMAP"] = CRITICAL_PAD_BITMAP
    bitmap_collection["REDUNDANT_PAD_BITMAP"] = REDUNDANT_PAD_BITMAP
    bitmap_collection["DUMMY_PAD_BITMAP"] = DUMMY_PAD_BITMAP
    bitmap_collection["ESD_CRITICAL_PAD_BITMAP"] = ESD_CRITICAL_PAD_BITMAP
    bitmap_collection["num_critical_pads"] = num_critical_pads
    bitmap_collection["num_redundant_pads"] = num_redundant_pads
    bitmap_collection["num_dummy_pads"] = num_dummy_pads
    bitmap_collection["redundant_net_to_bumpids"] = redundant_net_to_bumpids
    bitmap_collection["redundant_net_to_1d_physical_mask"] = redundant_net_to_1d_physical_mask
    bitmap_collection["pad_coords"] = pad_coords
    bitmap_collection["mapping_physical_to_bumpid"] = mapping_physical_to_bumpid
    bitmap_collection["criticality_info"] = criticality_info
    bitmap_collection, _ = _ensure_bitmap_collection_group_arrays(bitmap_collection)
    bitmap_collection["_cache_signature"] = cache_signature

    # Save the bitmap collection as npy file and mat file
    np.save(bitmap_collection_path, bitmap_collection)
    # sio.savemat(cfg.OUTPUT_DIR + "bitmap_collection.mat", bitmap_collection)

    # # Draw the critical and redundant pad bitmaps in one figure (critical light red, redundant light blue, dummy light gray)
    draw_pad_bitmap(cfg, bitmap_collection, output_path)
    # raise NotImplementedError("Stop here to avoid confusion.")

    return bitmap_collection



def result_wrapper(
        mode: str,
        output_dir: str,
        interface: str,
        fail_map_dict = None,
        file_suffix: str = "",
):
    """
    Wrap up the results, plot them and save the figures.
    """
    save_path = os.path.join(output_dir, interface)
    if mode in ["d2w_simulation", "w2w_simulation"]:
        for mechanism, fail_map in fail_map_dict.items():
            # Draw the failure map and save the figure to the output directory
            figure = plt.figure(figsize=(10, 10))
            plt.imshow(fail_map, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Failure Count')
            plt.title(f'Assembly Failure Map - {mechanism}')
            filename = f"simulation_failure_map_{mechanism}{file_suffix}.png"
            plt.savefig(save_path + f'/{filename}')
            plt.close(figure)
            print(f"Failure map for {mechanism} saved to {save_path + f'/{filename}'}")
