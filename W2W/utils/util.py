#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
import scipy.io as sio
import os


_POWER_GROUND_NET_TOKENS = {"vdd", "vss", "vpp", "vddq", "vddql", "gnd", "vcc"}


def _is_power_ground_net(net: str) -> bool:
    net_tokens = str(net).lower().replace("-", "_").replace(".", "_").split("_")
    return any(token in _POWER_GROUND_NET_TOKENS for token in net_tokens)


def print_run_separator(label: str = "Run finished"):
    duck = [
        "        YAP~",
        "          \\",
        "        __",
        "       /  \\",
        "     ∠)_• / ^_^",
        "      /  /_(•ω•)__",
        "     (      UU    )",
        "   ~~~~~~~~~~~~~~~~~~~~~~~",
    ]
    art_width = max(len(line) for line in duck)
    width = max(80, len(label), art_width)
    art_indent = " " * ((width - art_width) // 2)
    banner = "*" * width
    print("\n" + banner)
    print(label.center(width))
    for line in duck:
        print(art_indent + line)
    print(banner + "\n")


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


def estimate_w2w_num_dies_per_wafer(cfg) -> int:
    """
    Estimate the number of full dies on a W2W wafer using the same placement
    rule as Wafer_Interface.generate_die().
    """
    die_w_um = float(cfg.DIE_W_um)
    die_l_um = float(cfg.DIE_L_um)
    wafer_radius_um = float(cfg.WAF_R_um)
    dice_width_um = float(getattr(cfg, "dice_width", 0.0))
    dice_proportion = float(getattr(cfg, "dice_proportion", 1.0))

    if die_w_um <= 0.0 or die_l_um <= 0.0 or wafer_radius_um <= 0.0:
        raise ValueError(
            "DIE_W_um, DIE_L_um, and WAF_R_um must be positive to estimate "
            "the W2W die area fill factor."
        )

    pitch_x_um = die_w_um + dice_width_um
    pitch_y_um = die_l_um + dice_width_um
    if pitch_x_um <= 0.0 or pitch_y_um <= 0.0:
        raise ValueError("Die pitch including dice_width must be positive.")

    die_col = int(2 * wafer_radius_um // pitch_x_um + 1)
    die_row = int(2 * wafer_radius_um // pitch_y_um + 1)
    wafer_limit_um = wafer_radius_um * dice_proportion
    half_w_um = die_w_um / 2.0
    half_l_um = die_l_um / 2.0

    num_dies = 0
    for i in range(die_row):
        center_y_um = (
            die_row * pitch_y_um / 2.0
            - pitch_y_um / 2.0
            - i * pitch_y_um
        )
        for j in range(die_col):
            center_x_um = (
                -die_col * pitch_x_um / 2.0
                + pitch_x_um / 2.0
                + j * pitch_x_um
            )
            if np.hypot(center_x_um, center_y_um) >= wafer_limit_um:
                continue

            vertices_x = (center_x_um - half_w_um, center_x_um + half_w_um)
            vertices_y = (center_y_um - half_l_um, center_y_um + half_l_um)
            outside = False
            for x_um in vertices_x:
                for y_um in vertices_y:
                    if np.hypot(x_um, y_um) >= wafer_limit_um:
                        outside = True
                        break
                if outside:
                    break
            if not outside:
                num_dies += 1

    return int(num_dies)


def w2w_die_area_fill_factor(cfg, num_dies_per_wafer=None) -> float:
    """
    Return total die area divided by wafer area for W2W effective materials.
    """
    if num_dies_per_wafer is None:
        num_dies_per_wafer = getattr(cfg, "num_dies_per_wafer", None)
    if num_dies_per_wafer is None:
        num_dies_per_wafer = getattr(cfg, "NUM_DIES_PER_WAFER", None)
    if num_dies_per_wafer is None:
        num_dies_per_wafer = estimate_w2w_num_dies_per_wafer(cfg)

    die_area_um2 = float(cfg.DIE_W_um) * float(cfg.DIE_L_um)
    wafer_area_um2 = np.pi * float(cfg.WAF_R_um) ** 2
    if die_area_um2 <= 0.0 or wafer_area_um2 <= 0.0:
        raise ValueError("Die area and wafer area must be positive.")

    fill_factor = float(num_dies_per_wafer) * die_area_um2 / wafer_area_um2
    return float(np.clip(fill_factor, 0.0, 1.0))


def w2w_area_scaled_layer_volumes(cfg, mix_prefix, num_dies_per_wafer=None):
    """
    Scale Cu volume by W2W die-area fill factor and assign the removed Cu
    fraction back to SiO2/Si using their original non-Cu ratio.
    """
    cu = float(getattr(cfg, f"{mix_prefix}_Cu_V"))
    sio2 = float(getattr(cfg, f"{mix_prefix}_Sio2_V"))
    si = float(getattr(cfg, f"{mix_prefix}_Si_V"))
    if cu < 0.0 or sio2 < 0.0 or si < 0.0:
        raise ValueError(
            f"{mix_prefix} volume fractions must be non-negative: "
            f"Cu={cu}, Sio2={sio2}, Si={si}"
        )

    fill_factor = w2w_die_area_fill_factor(
        cfg,
        num_dies_per_wafer=num_dies_per_wafer,
    )
    cu_eff = cu * fill_factor
    released_cu = cu - cu_eff
    non_cu = sio2 + si
    if released_cu > 0.0:
        if non_cu > 0.0:
            sio2 += released_cu * sio2 / non_cu
            si += released_cu * si / non_cu
        else:
            si += released_cu

    return {
        "Cu": float(cu_eff),
        "Sio2": float(sio2),
        "Si": float(si),
        "area_fill_factor": float(fill_factor),
    }

def get_config_dict(
                    cfg_folder: str,
                    cfg_skeleton: str,
                    ds_name: str,
                    input_ds_dir: str,
                    _3dbv_path: str,
                    _3dbx_path: str,
                    mode: str,
                    debug=False) -> dict:
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
    for interface_name, cfg in cfg_dict.items():
        cfg.DESIGN = ds_name
        if mode == "w2w_simulation" or mode == "w2w_modeling":
            cfg.S_INIT_A_M = 100e-6 * (cfg.WAF_R_um / 150000) ** 2
            cfg.S_INIT_B_M = 0.0
        elif mode == "d2w_simulation" or mode == "d2w_modeling":
            cfg.eff_DIE_R = float(np.sqrt((cfg.DIE_W_um / 2) ** 2 + (cfg.DIE_L_um / 2) ** 2))  # Effective die radius (um)
            cfg.S_INIT_A_M = 10e-6 * (cfg.eff_DIE_R / 150000) ** 2
            cfg.S_INIT_B_M = 0.0
        else:
            raise ValueError(f"Unknown mode: {mode}. Supported modes are 'w2w_simulation', 'w2w_modeling', 'd2w_simulation', and 'd2w_modeling'.")


        if debug:
            cfg.DEBUG = True
            print("Configuration loaded:")
            print(OmegaConf.to_yaml(cfg))

        # Save updated config file for reference
        OmegaConf.save(cfg, cfg_folder + f"/{interface_name}.yaml")
    return cfg_dict


def update_config_from_bmap(cfg, _bmap_path, y_tol=0.1, x_tol=0.1):
    """
    Extract pad array layout from .bmap file.

    args:
        cfg: configuration object
        _bmap_path: path to .bmap file
        y_tol: tolerance for clustering y coordinates (um), if the difference between two y coordinates is less than y_tol, they are considered in the same row
        x_tol: tolerance for clustering x coordinates (um), if the difference between two x coordinates is less than x_tol, they are considered in the same column
    """
    coords = []

    with open(_bmap_path, 'r') as f:
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

    # Rank by y descending
    coords = coords[np.argsort(-coords[:, 1])]

    # Cluster to get unique rows
    y_vals = []
    for y in coords[:, 1]:
        if not y_vals or abs(y - y_vals[-1]) > y_tol:
            y_vals.append(y)
    num_rows = len(y_vals)

    # Cluster to get unique columns
    first_row_y = y_vals[0]
    first_row = coords[np.abs(coords[:, 1] - first_row_y) < y_tol]
    x_vals = []
    for x in sorted(first_row[:, 0]):
        if not x_vals or abs(x - x_vals[-1]) > x_tol:
            x_vals.append(x)
    num_cols = len(x_vals)
    
    add_config_items(cfg, keys=['PAD_ARR_ROW', 'PAD_ARR_COL'], values=[num_rows, num_cols])
    add_config_items(cfg, keys=['PAD_ARR_L_um', 'PAD_ARR_W_um'],
                        values=[(num_rows - 1) * cfg.PITCH_r_um,
                                (num_cols - 1) * cfg.PITCH_c_um])

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
        
        # Read pad pitch (top chip pitch) NOTE: currently assume row and col pitch are the same
        add_config_items(cfg, keys=['PITCH_r_um', 'PITCH_c_um'], 
                            values=[float(top_3dbf.Chiplet_Grid.pitch), 
                                    float(top_3dbf.Chiplet_Grid.pitch)])
        
        # Read die thickness
        add_config_items(cfg, keys=['ITF_TOP_THICK_um', 'ITF_BOT_THICK_um'],
                            values=[float(_3dbv.ChipletDef[cfg.INTERFACE_TOP].thickness),
                                    float(_3dbv.ChipletDef[cfg.INTERFACE_BOT].thickness)])

        ## Extract design parameters from .bmap file
        update_config_from_bmap(cfg, _bmap_path, 
                                y_tol=cfg.PITCH_r_um * 0.1, x_tol=cfg.PITCH_c_um * 0.1)

        # Store in config dictionary
        cfg_dict[cfg.INTERFACE] = cfg

    return cfg_dict





def draw_pad_bitmap(cfg, bitmap_collection):
    # Draw the critical and redundant pad bitmaps in one figure (critical light red, redundant light blue, dummy light gray)
    CRITICAL_PAD_BITMAP = bitmap_collection["CRITICAL_PAD_BITMAP"]
    REDUNDANT_PAD_BITMAP = bitmap_collection["REDUNDANT_PAD_BITMAP"]
    DUMMY_PAD_BITMAP = bitmap_collection["DUMMY_PAD_BITMAP"]
    POWER_GROUND_PAD_BITMAP = bitmap_collection.get(
        "POWER_GROUND_PAD_BITMAP",
        np.zeros_like(CRITICAL_PAD_BITMAP, dtype=bool),
    )
    ## Use legend to show the color
    PAD_BITMAP = np.zeros_like(CRITICAL_PAD_BITMAP, dtype=int)

    PAD_BITMAP[CRITICAL_PAD_BITMAP == 1] = 1  # red
    PAD_BITMAP[REDUNDANT_PAD_BITMAP == 1] = 2  # blue
    PAD_BITMAP[DUMMY_PAD_BITMAP == 1] = 3  # green
    PAD_BITMAP[POWER_GROUND_PAD_BITMAP == 1] = 4  # orange
    # Remaining zeros are non-pad areas
    PAD_BITMAP[PAD_BITMAP == 0] = 5  # non-pad (light gray)

    plt.figure(figsize=(15, 15))
    cmap = ListedColormap([
        (1.0, 0.5, 0.5),    # 1 - critical (medium red)
        (0.4, 0.4, 0.9),    # 2 - redundant (medium blue)
        (0.0, 0.6, 0.0),    # 3 - dummy (medium green)
        (1.0, 0.7, 0.2),    # 4 - power/ground (orange)
        (0.9, 0.9, 0.9),    # 5 - non-pad (light gray)
    ])
    red_patch = patches.Patch(color=(1.0, 0.5, 0.5), label='Critical Pads')
    blue_patch = patches.Patch(color=(0.4, 0.4, 0.9), label='Redundant Pads')
    green_patch = patches.Patch(color=(0.0, 0.6, 0.0), label='Dummy Pads')
    orange_patch = patches.Patch(color=(1.0, 0.7, 0.2), label='Power/Ground Pads')
    light_gray_patch = patches.Patch(color=(0.9, 0.9, 0.9), label='Non-Pad Areas')
    plt.legend(
        handles=[red_patch, blue_patch, green_patch, orange_patch, light_gray_patch],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.07),
        ncol=4,
        frameon=False
    )
    norm = BoundaryNorm(boundaries=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], ncolors=cmap.N)
    plt.imshow(PAD_BITMAP, cmap=cmap, norm=norm)
    plt.title("Pad Block Bitmap")


    # Save the pad bitmaps
    plt.savefig(cfg.OUTPUT_DIR + cfg.DESIGN + "/" + cfg.INTERFACE + "/" + cfg.INTERFACE + "_pad_bitmap.png")
    print("Pad bitmap collections info saved.")
    return

def sort_pads_bmap(input_path, output_path):
    """
    Read pad data from .bmap file, from top-left to right-bottom order 
    sorted by x ascending and y descending.
    - x is the 3rd column (index 2)
    - y is the 4th column (index 3)
    """

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

    # Transform to numpy array for sorting
    data = np.array(pads, dtype=object)

    # x ascending, y descending
    idx = np.lexsort((data[:,0].astype(float), - data[:,1].astype(float)))
    sorted_data = data[idx]

    with open(output_path, 'w') as f:
        for _, _, line in sorted_data:
            f.write(line + '\n')

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
    with open(cfg.OUTPUT_DIR + cfg.DESIGN + "/" + cfg.INTERFACE + "/" + cfg.INTERFACE + "_criticality.txt", 'w') as f:
        for bump_crit in bump_criticality:
            f.write(f"{bump_crit['port']} {bump_crit['esd_criticality']:.6f} {bump_crit['mechanical_criticality']:.6f}\n")
    print("Criticality file saved in ", cfg.OUTPUT_DIR + cfg.DESIGN + "/" + cfg.INTERFACE + "/" + cfg.INTERFACE + "_criticality.txt")
    return


# def risk_map_generator(cfg, 
#                         wafer: object,
#                         ):
#     '''
#     Risk map output format:
#     <pad_coords_x> <pad_coords_y> <esd_failure_probability> <overlay_failure_probability> <particle_failure_probability> <mechanical_failure_probability>
#     W2W's die risk map is the average of all dies on the wafer.
#     '''
#     avg_ovl_pad_yield_map = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), dtype=float)
#     avg_df_pad_yield_map = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), dtype=float)
#     avg_ce_pad_yield_map = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), dtype=float)
#     avg_esd_pad_yield_map = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), dtype=float)
#     avg_bond_pad_yield_map = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), dtype=float)
#     num_dies = len(wafer.die_list)
#     for die_id, die in enumerate(wafer.die_list):
#         avg_ovl_pad_yield_map += die.pad_yield_map['Y_ovl']
#         avg_df_pad_yield_map += die.pad_yield_map['Y_df']
#         avg_ce_pad_yield_map += die.pad_yield_map['Y_ce']
#         avg_esd_pad_yield_map += die.pad_yield_map['Y_esd']
#         avg_bond_pad_yield_map += die.pad_yield_map['Y_bond']
#     avg_ovl_pad_yield_map /= num_dies
#     avg_df_pad_yield_map /= num_dies
#     avg_ce_pad_yield_map /= num_dies
#     avg_esd_pad_yield_map /= num_dies
#     avg_bond_pad_yield_map /= num_dies


#     die_coords = wafer.base_pad_coords
#     risk_map = list()
#     for pad_id in range(len(die_coords)):
#         pad_coords_x = die_coords[pad_id, 0]
#         pad_coords_y = die_coords[pad_id, 1]
#         if np.isnan(pad_coords_x) or np.isnan(pad_coords_y):
#             continue
#         pad_ovl_yield = avg_ovl_pad_yield_map.flatten()[pad_id]
#         pad_df_yield = avg_df_pad_yield_map.flatten()[pad_id]
#         pad_ce_yield = avg_ce_pad_yield_map.flatten()[pad_id]
#         pad_esd_yield = avg_esd_pad_yield_map.flatten()[pad_id]
#         risk_map.append({
#             "pad_coords_x": pad_coords_x,
#             "pad_coords_y": pad_coords_y,
#             "esd_failure_probability": 1 - pad_esd_yield,
#             "overlay_failure_probability": 1 - pad_ovl_yield,
#             "particle_failure_probability": 1 - pad_df_yield,
#             "mechanical_failure_probability": 1 - pad_ce_yield,
#         })
#     with open(cfg.OUTPUT_DIR + cfg.DESIGN + "/" + cfg.INTERFACE + "/" + cfg.INTERFACE + "_risk_map.map", 'w') as f:
#         for pad_risk in risk_map:
#             f.write(f"{pad_risk['pad_coords_x']} {pad_risk['pad_coords_y']} {pad_risk['esd_failure_probability']} {pad_risk['overlay_failure_probability']} {pad_risk['particle_failure_probability']} {pad_risk['mechanical_failure_probability']}\n")
#     print("Risk map file saved in ", cfg.OUTPUT_DIR + cfg.DESIGN + "/" + cfg.INTERFACE + "/" + cfg.INTERFACE + "_risk_map.map")
#     return

def convert_3dblox_to_pad_bitmap(cfg, 
                                 _bmap_path: str, 
                                 criticality_path: str,
                                 pad_arrange_pattern: str,
                                 ):
    '''
    This module converts the 3DBlox .bmap file to pad bitmap for YAP to process.
        - pad_arrange_pattern: 'checkerboard' for UCIe standard and HBM
    '''
    # Create output directory if not exist
    if not os.path.exists(cfg.OUTPUT_DIR + cfg.DESIGN + '/' + cfg.INTERFACE):
        os.makedirs(cfg.OUTPUT_DIR + cfg.DESIGN + '/' + cfg.INTERFACE)

    sort_pads_bmap(_bmap_path, _bmap_path)

    # Read the bump data from the .bmap file
    bump_data = []
    # Initialize the pad array boundaries
    [pad_array_left, pad_array_right, pad_array_top, pad_array_bottom] = [float('inf'), float('-inf'), float('-inf'), float('inf')]
    with open(_bmap_path, 'r') as f:
        bumpid = 0  # Initialize bumpid
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
                bumpid += 1  # Increment bumpid after successfully parsing a line
    # Record the 1D physical locations of each pad in redundant nets
    '''Example: {NC: [0, 5, 10], VSS: [1, 6, 11], VDD: [2, 7, 12], ...}'''
    redundant_net_to_1d_physical_mask = dict()   
    # Record the bump ids of each pad in redundant nets, bump id is the index in bump_data list
    redundant_net_to_bumpids = dict()
    for bump in bump_data:
        if bump['net'] not in redundant_net_to_bumpids:
            redundant_net_to_bumpids[bump['net']] = set()
            redundant_net_to_1d_physical_mask[bump['net']] = np.array([], dtype=int)
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

    # Cache frequently-accessed config values as plain Python locals
    # (avoids ~8M expensive OmegaConf __getattr__ resolutions in the bump loop)
    _PAD_ARR_ROW = int(cfg.PAD_ARR_ROW)
    _PAD_ARR_COL = int(cfg.PAD_ARR_COL)
    _PITCH_r_um  = float(cfg.PITCH_r_um)
    _PITCH_c_um  = float(cfg.PITCH_c_um)

    # Initialize the pad bitmap
    # TODO: You need to modify the simulator to support different pad arrangement patterns
    CRITICAL_PAD_BITMAP = np.zeros((_PAD_ARR_ROW, _PAD_ARR_COL), dtype=bool)
    REDUNDANT_PAD_BITMAP = np.zeros((_PAD_ARR_ROW, _PAD_ARR_COL), dtype=bool)
    DUMMY_PAD_BITMAP = np.zeros((_PAD_ARR_ROW, _PAD_ARR_COL), dtype=bool)
    POWER_GROUND_PAD_BITMAP = np.zeros((_PAD_ARR_ROW, _PAD_ARR_COL), dtype=bool)
    ESD_CRITICAL_PAD_BITMAP = np.zeros((_PAD_ARR_ROW, _PAD_ARR_COL), dtype=bool)
    pad_coords = np.full((_PAD_ARR_ROW * _PAD_ARR_COL, 2), np.nan, dtype=np.float32)  # x, y coordinates of each bump
    # Build a mapping array from physical bump location (r, c) to bump id
    mapping_physical_to_bumpid = np.full((_PAD_ARR_ROW, _PAD_ARR_COL), np.nan, dtype=np.float32) # Shape: (PAD_ARR_ROW, PAD_ARR_COL)

    if pad_arrange_pattern in ('checkerboard', 'rectangular'): # This case is for UCIe standard
        for bump in bump_data:
            x = bump['x']
            y = bump['y']
            row = int(round((pad_array_top - y ) / _PITCH_r_um))   # Because in checkerboard pattern, the pitch per row is halved
            col = int(round((x - pad_array_left) / _PITCH_c_um))   # Because in checkerboard pattern, the pitch per column is halved
            mapping_physical_to_bumpid[row, col] = bump['bumpid']
            pad_coords[row * _PAD_ARR_COL + col, 0] = bump['x'] - (pad_array_left + pad_array_right) / 2
            pad_coords[row * _PAD_ARR_COL + col, 1] = bump['y'] - (pad_array_top + pad_array_bottom) / 2
            current_bump_net = bump['net']
            num_copies = len(redundant_net_to_bumpids[current_bump_net])
            if 'dummy' in current_bump_net.lower():
                DUMMY_PAD_BITMAP[row, col] = 1
                continue
            if _is_power_ground_net(current_bump_net):
                POWER_GROUND_PAD_BITMAP[row, col] = 1
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
                redundant_net_to_1d_physical_mask[bump['net']] = np.append(redundant_net_to_1d_physical_mask[bump['net']], row * _PAD_ARR_COL + col)
                continue
    else:   
        raise NotImplementedError("Currently only support checkerboard pad arrangement pattern.")

    redundant_net_to_1d_physical_mask = {
        net: np.asarray(physical_mask, dtype=int)
        for net, physical_mask in redundant_net_to_1d_physical_mask.items()
        if np.asarray(physical_mask).size > 0 and not _is_power_ground_net(net)
    }
    redundant_net_to_bumpids = {
        net: redundant_net_to_bumpids[net]
        for net in redundant_net_to_1d_physical_mask
    }

    # Count the number of pads
    num_critical_pads = np.sum(CRITICAL_PAD_BITMAP)
    num_redundant_pads = np.sum(REDUNDANT_PAD_BITMAP)
    num_dummy_pads = 0 if DUMMY_PAD_BITMAP is None else np.sum(DUMMY_PAD_BITMAP)
    num_power_ground_pads = np.sum(POWER_GROUND_PAD_BITMAP)

    
    # # Count the number of logical pads in redundant pads & Initialize the redundant net alive count dict
    # print("redundant_net_to_bumpids:", redundant_net_to_bumpids)


    bitmap_collection = {}
    bitmap_collection["bump_data"] = bump_data
    bitmap_collection["CRITICAL_PAD_BITMAP"] = CRITICAL_PAD_BITMAP
    bitmap_collection["REDUNDANT_PAD_BITMAP"] = REDUNDANT_PAD_BITMAP
    bitmap_collection["DUMMY_PAD_BITMAP"] = DUMMY_PAD_BITMAP
    bitmap_collection["POWER_GROUND_PAD_BITMAP"] = POWER_GROUND_PAD_BITMAP
    bitmap_collection["ESD_CRITICAL_PAD_BITMAP"] = ESD_CRITICAL_PAD_BITMAP
    bitmap_collection["num_critical_pads"] = num_critical_pads
    bitmap_collection["num_redundant_pads"] = num_redundant_pads
    bitmap_collection["num_dummy_pads"] = num_dummy_pads
    bitmap_collection["num_power_ground_pads"] = num_power_ground_pads
    bitmap_collection["redundant_net_to_bumpids"] = redundant_net_to_bumpids
    bitmap_collection["redundant_net_to_1d_physical_mask"] = redundant_net_to_1d_physical_mask
    bitmap_collection["pad_coords"] = pad_coords
    bitmap_collection["mapping_physical_to_bumpid"] = mapping_physical_to_bumpid
    bitmap_collection["criticality_info"] = criticality_info
    
    
    # Save the bitmap collection as npy file and mat file
    np.save(cfg.OUTPUT_DIR + cfg.DESIGN + "/" + cfg.INTERFACE + "/" + cfg.INTERFACE + "_bitmap_collection.npy", bitmap_collection)
    # sio.savemat(cfg.OUTPUT_DIR + "bitmap_collection.mat", bitmap_collection)

    # # Draw the critical and redundant pad bitmaps in one figure (critical light red, redundant light blue, dummy light gray)
    # draw_pad_bitmap(cfg, bitmap_collection)

    return bitmap_collection



def result_wrapper(
        mode: str,
        cfg: object,
        fail_map_per_interface_dict = None,
):
    """
    Wrap up the results, plot them and save the figures.
    """
    save_path = cfg.OUTPUT_DIR + cfg.DESIGN + "/" + cfg.INTERFACE
    # make directory if not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if mode in ["d2w_simulation", "w2w_simulation"]:
        if fail_map_per_interface_dict is None:
            return
        for mechanism, fail_map in fail_map_per_interface_dict[cfg.INTERFACE].items():
            # Draw the failure map and save the figure to the output directory
            figure = plt.figure(figsize=(10, 10))
            plt.imshow(fail_map, cmap='hot', interpolation='nearest')
            plt.colorbar(label='Failure Count')
            plt.title(f'Assembly Failure Map - {cfg.INTERFACE} - {mechanism}')
            plt.savefig(save_path + f'/failure_map_{mechanism}.png')
            plt.close(figure)
            print(f"Failure map for {mechanism} saved to {save_path + f'/failure_map_{mechanism}.png'}")
