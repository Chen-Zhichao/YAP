#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies stack initialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Jan 16, 2026

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon, Circle

class Die:
    def __init__(
        self, DIE_W_um, DIE_L_um, die_center, NUM_PADS_PER_DIE,
        DIE_VERTEX_COORDS, PAD_ARR_BOX, 
        pad_yield_flag: bool,
    ):
        self.DIE_W_um = DIE_W_um
        self.DIE_L_um = DIE_L_um
        self.die_center = die_center
        self.num_pads = NUM_PADS_PER_DIE
        self.vertices_coords = self.get_vertices_coords(die_center, DIE_VERTEX_COORDS)
        self.pad_array_box = PAD_ARR_BOX + die_center

        self.survival = True
        self.voids_occur = False

        self.die_yield = {}
        self.pad_yield_map = {}
        self.pad_yield_flag = pad_yield_flag

    def get_vertices_coords(self, die_center, DIE_VERTEX_COORDS):
        vertices_coords = DIE_VERTEX_COORDS + die_center
        return vertices_coords
    



class Wafer_Interface:
    def __init__(
        self,
        wafer_radius: float,
        DIE_W_um: float,
        DIE_L_um: float,
        PAD_ARR_ROW: int,
        PAD_ARR_COL: int,
        PAD_TOP_R_um: float,
        PAD_BOT_R_um: float,
        base_pad_coords: np.ndarray,
        dice_width: float,
        pad_yield_flag: bool,
        dice_proportion=1.0,
    ):
        self.wafer_radius = wafer_radius
        self.DIE_W_um = DIE_W_um
        self.DIE_L_um = DIE_L_um
        self.PAD_ARR_ROW = PAD_ARR_ROW
        self.PAD_ARR_COL = PAD_ARR_COL
        self.PAD_TOP_R_um = PAD_TOP_R_um
        self.PAD_BOT_R_um = PAD_BOT_R_um
        self.die_list = []
        self.num_dies = 0
        self.dice_proportion = dice_proportion
        self.voids = []

        self.survival_die = 0
        self.base_pad_coords = base_pad_coords
        self.dice_width = dice_width
        self.pad_yield_flag = pad_yield_flag
        self.glb_pad_yield_min_max_dict = {}

    def generate_die(self, NUM_PADS_PER_DIE, DIE_VERTEX_COORDS, PAD_ARR_BOX):
        die_row = 2 * self.wafer_radius // (self.DIE_L_um + self.dice_width) + 1
        die_col = 2 * self.wafer_radius // (self.DIE_W_um + self.dice_width) + 1
        flag_die_outside = False
        for i in range(int(die_row)):
            for j in range(int(die_col)):
                flag_die_outside = False
                die_center = np.array(
                    [
                        - die_col * (self.DIE_W_um + self.dice_width) / 2
                        + (self.DIE_W_um + self.dice_width) / 2
                        + j * (self.DIE_W_um + self.dice_width),
                        die_row * (self.DIE_L_um + self.dice_width) / 2
                        - (self.DIE_L_um + self.dice_width) / 2
                        - i * (self.DIE_L_um + self.dice_width),
                    ]
                )
                if (
                    np.sqrt(die_center[0] ** 2 + die_center[1] ** 2)
                    >= self.wafer_radius * self.dice_proportion
                ):
                    flag_die_outside = True
                    continue
                die = Die(
                    self.DIE_W_um,
                    self.DIE_L_um,
                    die_center,
                    NUM_PADS_PER_DIE,
                    DIE_VERTEX_COORDS,
                    PAD_ARR_BOX,
                    pad_yield_flag=self.pad_yield_flag
                )
                for vertex in die.vertices_coords:
                    if (
                        np.sqrt(vertex[0] ** 2 + vertex[1] ** 2)
                        >= self.wafer_radius * self.dice_proportion
                    ):
                        flag_die_outside = True
                        break
                if flag_die_outside:
                    continue
                self.die_list.append(die)
        self.num_dies = len(self.die_list)

    def get_die_radial_layers(self, radius_bin_um: float = None, decimals: int = 0):
        """
        Group dies by their distance from wafer center into radial layers.

        Two dies belong to the same layer when their centre-to-origin
        distances round to the same value under the chosen binning.

        Parameters
        ----------
        radius_bin_um : float or None
            Bin width in um.  If given, each die's radius is quantised as
            ``round(r / radius_bin_um) * radius_bin_um`` before grouping.
            If None, radii are simply rounded to *decimals* decimal places.
        decimals : int
            Rounding precision (only used when radius_bin_um is None).

        Returns
        -------
        dict with keys
            'layers' : list[dict]
                Each dict has:
                    'layer_id'             – int, 0-based layer index
                    'indices'              – np.ndarray of die indices
                    'count'                – int, number of dies in layer
                    'representative_index' – int, first die index in layer
                    'radius_um'            – float, binned radius value
            'counts' : np.ndarray of int, count per layer
            'representative_indices' : np.ndarray of int
        """
        cache_key = (radius_bin_um, decimals)
        if not hasattr(self, '_radial_layers_cache'):
            self._radial_layers_cache = {}
        if cache_key in self._radial_layers_cache:
            return self._radial_layers_cache[cache_key]

        radii = np.array(
            [np.linalg.norm(d.die_center) for d in self.die_list],
            dtype=np.float64,
        )

        if radius_bin_um is not None and radius_bin_um > 0:
            binned = np.round(radii / radius_bin_um) * radius_bin_um
        else:
            binned = np.round(radii, decimals=decimals)

        unique_bins, inverse = np.unique(binned, return_inverse=True)

        layers = []
        counts_list = []
        rep_list = []
        for lid, bval in enumerate(unique_bins):
            idx = np.where(inverse == lid)[0]
            rep = int(idx[0])
            layers.append({
                'layer_id': lid,
                'indices': idx,
                'count': len(idx),
                'representative_index': rep,
                'radius_um': float(bval),
            })
            counts_list.append(len(idx))
            rep_list.append(rep)

        result = {
            'layers': layers,
            'counts': np.array(counts_list, dtype=np.int64),
            'representative_indices': np.array(rep_list, dtype=np.int64),
        }
        self._radial_layers_cache[cache_key] = result
        return result

    def draw_wafer_die(self, fig_size=(30, 30), draw_pad_yield_map_option=None):
        fig, ax = plt.subplots(figsize=fig_size, dpi=600)
        wafer_circle = plt.Circle((0, 0), self.wafer_radius, color="black", fill=False)
        ax.add_artist(wafer_circle)
        ax.set_xlim(-self.wafer_radius * 1.1, self.wafer_radius * 1.1)
        ax.set_ylim(-self.wafer_radius * 1.1, self.wafer_radius * 1.1)
        # draw dies
        for die in self.die_list:
            # Draw the pad array box
            polygon_coords = np.array([
                die.vertices_coords[0],  # top-left
                die.vertices_coords[1],  # top-right
                die.vertices_coords[3],  # bottom-right
                die.vertices_coords[2],  # bottom-left
            ])
            die_box = Polygon(polygon_coords, color="blue", fill=False)
            ax.add_patch(die_box)
            if die.survival == False:   # Draw a red edge if the die is not survived
                die_box = Polygon(polygon_coords, color="red", fill=False)
            elif die.voids_occur == True:
                die_box = Polygon(polygon_coords, color="green", fill=False)
            ax.add_patch(die_box)

            # Draw the pad yield map
            '''draw_pad_yield_map_option:
                - 'Y_ovl' : overlay yield map
                - 'Y_df': defect-free yield map
                - 'Y_ce': Cu expansion yield map
                - 'Y_esd': ESD yield map
            '''
            if hasattr(die, 'pad_yield_map') and draw_pad_yield_map_option in die.pad_yield_map:
                draw_pad_yield_map = die.pad_yield_map[draw_pad_yield_map_option]
                ax.imshow(
                    draw_pad_yield_map,
                    extent=[
                        die.vertices_coords[0][0], # x_min
                        die.vertices_coords[1][0], # x_max
                        die.vertices_coords[2][1], # y_min
                        die.vertices_coords[0][1]  # y_max
                    ],
                    origin='upper',
                    cmap='viridis',
                    vmin=self.glb_pad_yield_min_max_dict[draw_pad_yield_map_option][0], # global min
                    vmax=self.glb_pad_yield_min_max_dict[draw_pad_yield_map_option][1], # global max
                    alpha=0.5,
                )
            # draw pads
            # die_pad_coords = die.die_center + self.base_pad_coords
            # for pad in die_pad_coords:
            #     if pad[0] != np.nan and pad[1] != np.nan:   # There is a pad/bump
            #         ax.add_artist(patches.Circle((pad[0], pad[1]), self.PAD_BOT_R_um, color='darkorange', fill=True, alpha=1.0))
            #         ax.add_artist(patches.Circle((pad[0], pad[1]), self.PAD_TOP_R_um, color='lightgreen', fill=True, alpha=1.0))

        # Draw voids
        for v in self.voids:
            ax.add_artist(patches.Circle((v[0], v[1]), v[2], color="black", fill=False))
        ax.set_aspect("equal")
        plt.show()
        # # Save the wafer figure
        fig.savefig("wafer_die.png")    





def wafer_interface_initialize(
    DIE_W_um,
    DIE_L_um,
    PAD_ARR_W_um,
    PAD_ARR_L_um,
    PAD_ARR_ROW,
    PAD_ARR_COL,
    PITCH_r_um,
    PITCH_c_um,
    WAF_R_um,
    PAD_TOP_R_um,
    PAD_BOT_R_um,
    dice_width,
    pad_bitmap_collection,
    pad_yield_flag: bool = False,
):
    waf_interface_list = []
    # Calculate the die center standard coordinates
    DIE_VERTEX_COORDS = np.array(
        [
            [-DIE_W_um / 2, DIE_L_um / 2],
            [DIE_W_um / 2, DIE_L_um / 2],
            [-DIE_W_um / 2, -DIE_L_um / 2],
            [DIE_W_um / 2, -DIE_L_um / 2],
        ]
    )  # die vertex coordinates: [top-left, top-right, bottom-left, bottom-right]
    PAD_ARR_BOX = np.array(
        [
            [-PAD_ARR_W_um / 2, PAD_ARR_L_um / 2], 
            [PAD_ARR_W_um / 2, PAD_ARR_L_um / 2], 
            [-PAD_ARR_W_um / 2, -PAD_ARR_L_um / 2], 
            [PAD_ARR_W_um / 2, -PAD_ARR_L_um / 2]])
    
    # Calculate the total number of pads per die
    NUM_PADS_PER_DIE = pad_bitmap_collection['num_critical_pads'] + pad_bitmap_collection['num_redundant_pads'] + pad_bitmap_collection['num_dummy_pads']
    

    if pad_bitmap_collection['pad_coords'] is not None:
        PAD_COORDS = pad_bitmap_collection['pad_coords']
    else:
        if PITCH_r_um >= 1.0 and PITCH_c_um >= 1.0:
            # Specify the pad coordinates based on pitch and array size
            PAD_COORDS = np.zeros([PAD_ARR_ROW * PAD_ARR_COL, 2], dtype=np.float32)  # pad coordinates: [x, y]
        
            # Create grid of row and column indices
            col_indices = np.arange(PAD_ARR_COL)
            row_indices = np.arange(PAD_ARR_ROW)
            col_grid, row_grid = np.meshgrid(col_indices, row_indices)

            # Calculate x and y coordinates
            x_coords = (-PAD_ARR_W_um / 2 + col_grid * PITCH_c_um).astype(np.float32)
            y_coords = (PAD_ARR_L_um / 2 - row_grid * PITCH_r_um).astype(np.float32)

            # Combine x and y coordinates
            PAD_COORDS = np.stack((x_coords, y_coords), axis=-1).reshape(-1, 2)
        else:
            print("Too many Cu pads... Will not generate the pad coordinates.")
            PAD_COORDS = None
    
    # Initialize the wafer interface and generate the dies and pads
    wafer_interface = Wafer_Interface(
        wafer_radius=WAF_R_um,
        DIE_W_um=DIE_W_um,
        DIE_L_um=DIE_L_um,
        PAD_ARR_ROW=PAD_ARR_ROW,
        PAD_ARR_COL=PAD_ARR_COL,
        PAD_TOP_R_um=PAD_TOP_R_um,
        PAD_BOT_R_um=PAD_BOT_R_um,
        base_pad_coords=PAD_COORDS,
        dice_width=dice_width,
        pad_yield_flag=pad_yield_flag,
    )
    wafer_interface.generate_die(NUM_PADS_PER_DIE, DIE_VERTEX_COORDS, PAD_ARR_BOX)
    wafer_interface.survival_die = len(wafer_interface.die_list)
    # print("{} dies in the wafer.".format(len(wafer.die_list)))
    
    return wafer_interface


class Bonding_Interfaces:
    def __init__(self,
                 cfg_dict: dict,
                 pad_bitmap_collection_dict: dict,
                 ):
        """
        The set of bonding interfaces for a single wafer stack
        """
        self.cfg_dict = cfg_dict
        self.pad_bitmap_collection_dict = pad_bitmap_collection_dict
        self.failure_params_dict = {}
        self.interface_dict = {}
        
        for interface_name in cfg_dict.keys():
            self.failure_params_dict[interface_name] = {}
            # Overlay failure parameters for each bonding interface in each stack
            self.failure_params_dict[interface_name]['MAX_ALLOWED_MISALIGNMENT_um'] = None
            self.failure_params_dict[interface_name]['system_translation_x_um'] = None
            self.failure_params_dict[interface_name]['system_translation_y_um'] = None
            self.failure_params_dict[interface_name]['system_rotation_rad'] = None
            self.failure_params_dict[interface_name]['system_magnification_ppm'] = None
            # Particle-induced void failure parameters for each bonding interface in each stack
            self.failure_params_dict[interface_name]['voids'] = None # each entry is an array of voids (np.ndarray)

    def add_interfaces(self):
        """
        Initialize the wafer stack interfaces for one wafer stack sample.
        """
        for interface_name, cfg in self.cfg_dict.items():
            self.interface_dict[interface_name] = wafer_interface_initialize(
                DIE_W_um                    = cfg.DIE_W_um,
                DIE_L_um                    = cfg.DIE_L_um,
                PAD_ARR_W_um                = cfg.PAD_ARR_W_um,
                PAD_ARR_L_um                = cfg.PAD_ARR_L_um,
                PAD_ARR_ROW                 = cfg.PAD_ARR_ROW,
                PAD_ARR_COL                 = cfg.PAD_ARR_COL,
                PITCH_r_um                  = cfg.PITCH_r_um,
                PITCH_c_um                  = cfg.PITCH_c_um,
                WAF_R_um                    = cfg.WAF_R_um,
                PAD_TOP_R_um                = cfg.PAD_TOP_R_um,
                PAD_BOT_R_um                = cfg.PAD_BOT_R_um,
                dice_width                  = cfg.dice_width,
                pad_bitmap_collection       = self.pad_bitmap_collection_dict[interface_name],
                pad_yield_flag              = cfg.pad_yield_flag,
            )

class WaferStack:
    def __init__(self, 
                 cfg_dict: dict,
                 pad_bitmap_collection_dict: dict,
                 mode = None,
                 ):
        """
        num_bonding_interfaces: Number of bonding interfaces in each stack
        interfaces: Bonding_Interfaces object containing all bonding interfaces
        num_dies_per_wafer: Number of dies per wafer (assumed same for all interfaces)
        die_stack_survival: Boolean array indicating whether each die stack survives
        """
        failure_mechanism_list = ['overlay', 'particle', 'mechanical', 'ESD', 'warpage', 'overall']
        
        self.cfg_dict = cfg_dict
        self.num_bonding_interfaces = len(cfg_dict)
        self.interfaces = Bonding_Interfaces(
            cfg_dict=cfg_dict,
            pad_bitmap_collection_dict=pad_bitmap_collection_dict,
        )
        self.interfaces.add_interfaces()
        self.num_dies_per_wafer = self.interfaces.interface_dict[list(cfg_dict.keys())[0]].num_dies

        if 'simulation' in mode:        # For yield simulation
            self.die_stack_survival = np.ones((self.num_dies_per_wafer), dtype=bool)  # Initialize all die stacks as survived
        elif 'modeling' in mode:        # For yield modeling
            self.die_yield_list_per_interface_dict = {interface_name: {
                failure_mechanism: np.full((self.num_dies_per_wafer), np.nan) for failure_mechanism in failure_mechanism_list
            }   for interface_name in cfg_dict.keys()}

            self.die_stack_yield_list = np.ones((self.num_dies_per_wafer), dtype=float)  # Initialize all die stacks as survived (yield=1)
            self.die_stack_yield = np.nan
        else:
            raise ValueError("Invalid mode. Please specify 'simulation' or 'modeling' in the mode argument.")
        
    def get_die_stack_yield(self):
        """
        Calculate the yield for each die stack based on the failure parameters of each interface.
        """
        for interface_name, interface in self.interfaces.interface_dict.items():
            self.die_yield_list_per_interface_dict[interface_name]['overall'] = self.die_yield_list_per_interface_dict[interface_name]['overlay'] * \
                self.die_yield_list_per_interface_dict[interface_name]['particle'] * \
                self.die_yield_list_per_interface_dict[interface_name]['mechanical'] * \
                self.die_yield_list_per_interface_dict[interface_name]['ESD'] * \
                self.die_yield_list_per_interface_dict[interface_name]['warpage']
            
            # Update the die stack yield list by multiplying the overall yields from each interface
            self.die_stack_yield_list *= self.die_yield_list_per_interface_dict[interface_name]['overall']

        self.die_stack_yield = np.mean(self.die_stack_yield_list)

        return self.die_stack_yield, self.die_stack_yield_list

    def draw_w2w_stack_3d(self, cfg_dict, itf_pitch=1.0, fig_size=(10, 8), dpi=300,
                        draw_pad_yield_map_option=None, draw_voids=True, figname=None):
        """
        waf_itf: the wafer interface object containing the die and pad information for each layer
        itf_pitch: the distance between interfaces in the z direction (can be adjusted for better visualization)
        """
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        cfg = cfg_dict[list(cfg_dict.keys())[0]]  # Get config from the first interface for wafer dimensions

        waf_itf_r = cfg.WAF_R_um
        ax.set_xlim(-waf_itf_r * 1.1, waf_itf_r * 1.1)
        ax.set_ylim(-waf_itf_r * 1.1, waf_itf_r * 1.1)
        ax.set_zlim(-itf_pitch, itf_pitch * (len(self.interfaces.interface_dict) + 1))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Stack (Wafer Index)")

        # Draw each interface
        for itf_idx, waf_itf in enumerate(self.interfaces.interface_dict.values()):
            z = itf_idx * itf_pitch

            # Draw the wafer outline (circle) for each interface
            theta = np.linspace(0, 2*np.pi, 200)
            cx = waf_itf_r * np.cos(theta)
            cy = waf_itf_r * np.sin(theta)
            ax.plot(cx, cy, zs=z, zdir='z', linewidth=0.1, color='gray', alpha=0.5)

            # Draw dies for each interface
            for die in waf_itf.die_list:
                polygon_coords = np.array([
                    die.vertices_coords[0],  # top-left
                    die.vertices_coords[1],  # top-right
                    die.vertices_coords[3],  # bottom-right
                    die.vertices_coords[2],  # bottom-left
                ], dtype=float)

                # Determine edge color based on die status
                if die.survival is False:
                    face_color = (1.0, 0.2, 0.2, 0.3)   # 红：强
                elif getattr(die, "voids_occur", False):
                    face_color = (0.2, 0.8, 0.2, 0.3)   # 绿：中
                else:
                    face_color = (0.6, 0.6, 0.9, 0.05)  # 蓝灰：弱背景

                verts3d = [(x, y, z) for x, y in polygon_coords]
                poly = Poly3DCollection(
                    [verts3d],
                    facecolors=[face_color],   # 透明面，但仍然是“有一个面颜色”
                    edgecolors='none',     # 注意也用 list 包起来，长度=1
                    linewidths=0.6,
                    alpha=0.2
                )
                ax.add_collection3d(poly)

                # pad_yield_map: Directly mapping 2D images in 3D is complicated (requires converting 2D image to mesh/texture)
                # Suggestion: For 3D, focus on die status/voids structure; use single-layer 2D output or plotly texture for pad map.
                # If you strongly need it, I can also provide a version that samples imshow into point cloud/small cubes.

            # # Draw voids (simplified: use 3D scatter to represent centers, size ~ radius)
            # if draw_voids and hasattr(waf_itf, "voids"):
            #     vx = np.array([v[0] for v in waf_itf.voids], dtype=float)
            #     vy = np.array([v[1] for v in waf_itf.voids], dtype=float)
            #     vr = np.array([v[2] for v in waf_itf.voids], dtype=float)

            #     ax.scatter(vx, vy, zs=np.full_like(vx, z), s=np.clip(vr, 1, None)*2, alpha=0.4)

        ax.view_init(elev=25, azim=-55)  # 视角可调
        plt.tight_layout()
        plt.show()

        # Save the figure
        if figname is not None:
            fig.savefig(figname)
        else:
            fig.savefig("w2w_stack_3d.png")



    def draw_w2w_stack_2d(
        cfg_dict,
        waf_stack,
        fig_size=(10, 8),
        dpi=300,
        draw_voids=True,
        overlay_all_layers=True,
        draw_wafer_outline_once=True,
        figname=None,
    ):
        """
        Render a 2D top-view (XY projection) of a wafer-to-wafer (W2W) stack by
        overlaying all wafer interfaces onto a single plane.

        This function projects dies and voids from multiple wafer interfaces
        (layers) onto the same XY plane, which is particularly useful for
        visualizing spatial yield patterns and failure clustering across
        the entire stack.
        """

        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        ax.set_aspect("equal", adjustable="box")
        cfg = cfg_dict[list(cfg_dict.keys())[0]]  # Get config from the first interface for wafer dimensions

        # Set plot limits
        ax.set_xlim(-cfg.WAF_R_um * 1.1, cfg.WAF_R_um * 1.1)
        ax.set_ylim(-cfg.WAF_R_um * 1.1, cfg.WAF_R_um * 1.1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("W2W Stack Top View (2D Overlay)")

        # Draw wafer outline (once by default)
        if draw_wafer_outline_once:
            ax.add_patch(
                Circle(
                    (0, 0),
                    cfg.WAF_R_um,
                    fill=False,
                    linewidth=0.6,
                    edgecolor="gray",
                    alpha=0.7,
                )
            )

        # Iterate over wafer interfaces (layers)
        for itf_idx, interface in enumerate(waf_stack.interfaces.interface_dict.values()):
            if (not overlay_all_layers) and itf_idx != 0:
                continue

            # Optionally draw wafer outline for each layer
            if not draw_wafer_outline_once:
                ax.add_patch(
                    Circle(
                        (0, 0),
                        cfg.WAF_R_um,
                        fill=False,
                        linewidth=0.1,
                        edgecolor="gray",
                        alpha=0.5,
                    )
                )

            # Draw dies using filled polygons (no edges)
            for die in interface.die_list:
                vc = np.asarray(die.vertices_coords, dtype=float)
                if vc.ndim != 2 or vc.shape[0] < 4 or vc.shape[1] < 2:
                    continue

                polygon_coords = np.array([vc[0], vc[1], vc[3], vc[2]], dtype=float)

                # Color encoding based on die status
                if die.survival is False:
                    face_color = (1.0, 0.2, 0.2, 0.3)   # failed die
                elif getattr(die, "voids_occur", False):
                    face_color = (0.2, 0.8, 0.2, 0.3)   # void-related die
                else:
                    face_color = (0.6, 0.6, 0.9, 0.05)  # survived die (background)

                ax.add_patch(
                    Polygon(
                        polygon_coords[:, :2],
                        closed=True,
                        facecolor=face_color,
                        edgecolor="none",
                    )
                )

            # Draw voids as 2D scatter points
            if draw_voids and hasattr(interface, "voids") and interface.voids:
                vx = np.array([v[0] for v in interface.voids], dtype=float)
                vy = np.array([v[1] for v in interface.voids], dtype=float)
                vr = np.array([v[2] for v in interface.voids], dtype=float)

                marker_size = np.clip(vr, 1, None) * 2
                ax.scatter(vx, vy, s=marker_size, alpha=0.4)

        ax.grid(False)
        plt.tight_layout()
        plt.show()

        # Save the figure
        if figname is not None:
            fig.savefig(cfg_dict[list(cfg_dict.keys())[0]].OUTPUT_DIR
                         + cfg_dict[list(cfg_dict.keys())[0]].DESIGN
                         + '/' + figname)


    


        


def wafer_stack_list_initialize(
    cfg_dict: dict,
    pad_bitmap_collection_dict: dict,
    num_stack_samples: int,
    mode: str,
):
    """
    Inputs:
        - cfg_dict: Configuration dictionary containing parameters
        - num_stack_samples: Number of wafer stack samples to generate
        Outputs:
        - wafer_stacks: WaferStack object containing the initialized wafer stack samples
        - mode: 'simulation' or 'modeling' to determine the initialization of yield/survival attributes
    Output Structure:
        
    """
    wafer_stack_list = []
    for _ in range(num_stack_samples):
        wafer_stack = WaferStack(
            cfg_dict=cfg_dict,
            pad_bitmap_collection_dict=pad_bitmap_collection_dict,
            mode=mode,
        )
        wafer_stack_list.append(wafer_stack)
        
    
    return wafer_stack_list





