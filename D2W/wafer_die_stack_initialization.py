#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon

class Die:
    def __init__(
        self, DIE_W_um, DIE_L_um, die_center,
        DIE_VERTEX_COORDS, num_pads,
        PAD_TOP_R_um, PAD_BOT_R_um,
        PAD_ARR_BOX,
        pad_yield_flag: bool,
        BASE_PAD_COORDS: np.ndarray = None,
    ):
        self.DIE_W_um = DIE_W_um
        self.DIE_L_um = DIE_L_um
        self.die_center = die_center
        self.num_pads = num_pads
        self.PAD_TOP_R_um = PAD_TOP_R_um
        self.PAD_BOT_R_um = PAD_BOT_R_um
        self.vertices_coords = self.get_vertices_coords(die_center, DIE_VERTEX_COORDS)
        self.pad_array_box = PAD_ARR_BOX + die_center
        self.pad_coords = BASE_PAD_COORDS + die_center if pad_yield_flag == True else None

        self.survival = True
        self.safe_voids_mask = []
        self.voids = []
        self.voids_occur = False

        self.die_yield = {}
        self.pad_yield_map = {}
        self.glb_pad_yield_min_max_dict = {}

    def get_vertices_coords(self, die_center, DIE_VERTEX_COORDS):
        vertices_coords = DIE_VERTEX_COORDS + die_center
        return vertices_coords

    def draw_die(self, fig_size=(30, 30)):
        fig, ax = plt.subplots(figsize=fig_size)
        # Draw the pad array box
        polygon_coords = np.array([
            self.vertices_coords[0],  # top-left
            self.vertices_coords[1],  # top-right
            self.vertices_coords[3],  # bottom-right
            self.vertices_coords[2],  # bottom-left
        ])
        die_box = Polygon(polygon_coords, color="blue", fill=False)
        ax.add_patch(die_box)
        # draw die outline
        if self.survival == False:
            die_box = Polygon(self.vertices_coords, color="red", fill=False)
        elif self.voids_occur == True:
            die_box = Polygon(self.vertices_coords, color="green", fill=False)
        ax.add_patch(die_box)
        for v in self.voids:
            ax.add_artist(patches.Circle((v[0], v[1]), v[2], color="red", fill=False))
        ax.set_aspect("equal")
        # set x and y axis limits
        ax.set_xlim(-self.DIE_W_um*0.6, self.DIE_W_um*0.6)
        ax.set_ylim(-self.DIE_L_um*0.6, self.DIE_L_um*0.6)
        # draw pads
        for pad in self.pad_coords:
            if pad[0] != np.nan and pad[1] != np.nan:   # There is a pad/bump
                ax.add_artist(patches.Circle((pad[0], pad[1]), self.PAD_BOT_R_um, color='darkorange', fill=True, alpha=1.0))
                ax.add_artist(patches.Circle((pad[0], pad[1]), self.PAD_TOP_R_um, color='lightgreen', fill=True, alpha=1.0))
        plt.show()


class Wafer:
    def __init__(
        self,
        wafer_radius,
        DIE_W_um,
        DIE_L_um,
        PAD_TOP_R_um,
        PAD_BOT_R_um,
        base_pad_coords,
        dice_width,
        dice_proportion=1.0,
    ):
        self.wafer_radius = wafer_radius
        self.DIE_W_um = DIE_W_um
        self.DIE_L_um = DIE_L_um
        self.PAD_TOP_R_um = PAD_TOP_R_um
        self.PAD_BOT_R_um = PAD_BOT_R_um
        self.die_list = []
        self.dice_proportion = dice_proportion
        self.voids = []
        self.safe_voids_mask = []
        self.roughness_voids = []
        self.survival_die = 0
        self.base_pad_coords = base_pad_coords
        self.dice_width = dice_width

    def generate_die(self, DIE_VERTEX_COORDS, PAD_COORDS, PAD_ARR_BOX):
        die_row = 2 * self.wafer_radius // (self.DIE_L_um + self.dice_width) + 1
        die_col = 2 * self.wafer_radius // (self.DIE_W_um + self.dice_width) + 1
        flag_die_outside = False
        for i in range(int(die_row)):
            for j in range(int(die_col)):
                flag_die_outside = False
                die_center = np.array(
                    [
                        -die_col * (self.DIE_W_um + self.dice_width) / 2
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
                    DIE_VERTEX_COORDS,
                    PAD_COORDS,
                    PAD_ARR_BOX
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

    def draw_wafer_die(self, fig_size=(30, 30)):
        fig, ax = plt.subplots(figsize=fig_size)
        wafer_circle = plt.Circle((0, 0), self.wafer_radius, color="black", fill=False)
        ax.add_artist(wafer_circle)
        ax.set_xlim(-self.wafer_radius * 1.1, self.wafer_radius * 1.1)
        ax.set_ylim(-self.wafer_radius * 1.1, self.wafer_radius * 1.1)
        # draw dies
        for die in self.die_list:
            polygon_coords = np.array([
                    die.vertices_coords[0],  # top-left
                    die.vertices_coords[1],  # top-right
                    die.vertices_coords[3],  # bottom-right
                    die.vertices_coords[2],  # bottom-left
                ])
            if die.survival == True:
                die_box = Polygon(polygon_coords, color="green", fill=False)
                ax.add_patch(die_box)
            else:   # Draw a red edge for failed die
                die_box = Polygon(polygon_coords, color="red", fill=False)
                ax.add_patch(die_box)
            # draw pads
            for pad in die.pad_coords:
                if pad[0] != np.nan and pad[1] != np.nan:   # There is a pad/bump
                    ax.add_artist(patches.Circle((pad[0], pad[1]), self.PAD_BOT_R_um, color='darkorange', fill=True, alpha=1.0))
                    ax.add_artist(patches.Circle((pad[0], pad[1]), self.PAD_TOP_R_um, color='lightgreen', fill=True, alpha=1.0))

        # Draw voids
        for v in self.voids:
            ax.add_artist(patches.Circle((v[0], v[1]), v[2], color="red", fill=False))
        ax.set_aspect("equal")
        plt.show()
        # # Save the wafer figure
        # fig.savefig("wafer_die.png")


def die_interface_initialize(
    NUM_DIE_SAMPLES: int,
    DIE_W_um: float,
    DIE_L_um: float,
    PAD_ARR_W_um: float,
    PAD_ARR_L_um: float,
    PAD_ARR_ROW: int,
    PAD_ARR_COL: int,
    PITCH_r_um: float,
    PITCH_c_um: float,
    PAD_TOP_R_um: float,
    PAD_BOT_R_um: float,
    pad_bitmap_collection,
    pad_yield_flag: bool = False,
):
    die_interface_list = []
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

    num_pads = pad_bitmap_collection['num_critical_pads'] + pad_bitmap_collection['num_redundant_pads'] + pad_bitmap_collection['num_dummy_pads']

    if pad_bitmap_collection['pad_coords'] is not None:
        PAD_COORDS = pad_bitmap_collection['pad_coords']
    else:
        if PITCH_r_um >= 1.0 and PITCH_c_um >= 1.0:
            # Specify the pad coordinates
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

    for i in range(NUM_DIE_SAMPLES):
        die = Die(
            DIE_W_um=DIE_W_um,
            DIE_L_um=DIE_L_um,
            die_center=np.array([0, 0]),
            DIE_VERTEX_COORDS=DIE_VERTEX_COORDS,
            num_pads=num_pads,
            PAD_TOP_R_um = PAD_TOP_R_um,
            PAD_BOT_R_um = PAD_BOT_R_um,
            PAD_ARR_BOX=PAD_ARR_BOX,
            pad_yield_flag=pad_yield_flag,
            BASE_PAD_COORDS=PAD_COORDS,
        )
        die_interface_list.append(die)
    return die_interface_list, PAD_COORDS


class Bonding_Interfaces:
    def __init__(self,
                 cfg_dict: dict,
                 pad_bitmap_collection_dict: dict,
                 ):
        """
        The set of bonding interfaces for a single die stack
        """
        self.cfg_dict = cfg_dict
        self.pad_bitmap_collection_dict = pad_bitmap_collection_dict
        self.failure_params_dict = {}
        self.interface_dict = {}
        self.base_pad_coords_dict = {}

        for interface_name in cfg_dict.keys():
            self.failure_params_dict[interface_name] = {}
            # Overlay failure parameters for each bonding interface in each stack
            self.failure_params_dict[interface_name]['MAX_ALLOWED_MISALIGNMENT_um'] = None
            self.failure_params_dict[interface_name]['system_translation_x_um'] = None
            self.failure_params_dict[interface_name]['system_translation_y_um'] = None
            self.failure_params_dict[interface_name]['system_rotation_rad'] = None
            self.failure_params_dict[interface_name]['system_magnification_ppm'] = None
            # Particle-induced void failure parameters for each bonding interface in each stack
            self.failure_params_dict[interface_name]['voids'] = None  # each entry is an array of voids [x, y, r_um]

    def add_interfaces(self, base_pad_coords_flag: bool = False):
        """
        Initialize bonding interfaces for a single die stack.
        """
        for interface_name, cfg in self.cfg_dict.items():
            interface_list, base_pad_coords = die_interface_initialize(
                NUM_DIE_SAMPLES           = 1,
                DIE_W_um                  = cfg.DIE_W_um,
                DIE_L_um                  = cfg.DIE_L_um,
                PAD_ARR_W_um              = cfg.PAD_ARR_W_um,
                PAD_ARR_L_um              = cfg.PAD_ARR_L_um,
                PAD_ARR_ROW               = cfg.PAD_ARR_ROW,
                PAD_ARR_COL               = cfg.PAD_ARR_COL,
                PITCH_r_um                = cfg.PITCH_r_um,
                PITCH_c_um                = cfg.PITCH_c_um,
                PAD_TOP_R_um              = cfg.PAD_TOP_R_um,
                PAD_BOT_R_um              = cfg.PAD_BOT_R_um,
                pad_bitmap_collection     = self.pad_bitmap_collection_dict[interface_name],
                pad_yield_flag            = cfg.pad_yield_flag,
            )
            self.interface_dict[interface_name] = interface_list[0]
            if base_pad_coords_flag:
                self.base_pad_coords_dict[interface_name] = base_pad_coords



class DieStack:
    def __init__(
        self,
        cfg_dict: dict,
        pad_bitmap_collection_dict: dict,
        mode = None,
        base_pad_coords_flag: bool = False,
    ):
        """
        Die Stack object for hybrid bonding yield model.
        """
        failure_mechanism_list = ['overlay', 'particle', 'mechanical', 'ESD', 'warpage', 'overall']

        self.cfg_dict = cfg_dict
        self.num_bonding_interfaces = len(cfg_dict) - 1  # Number of bonding interfaces is number of layers - 1

        self.interfaces = Bonding_Interfaces(
            cfg_dict=cfg_dict,
            pad_bitmap_collection_dict=pad_bitmap_collection_dict,
        )
        self.interfaces.add_interfaces(base_pad_coords_flag=base_pad_coords_flag)

        if 'simulation' in mode:    # For yield simulation
            self.survival = True
        elif 'modeling' in mode:    # For yield modeling
            self.die_yield_per_interface_dict = {interface_name: {
                failure_mechanism: np.nan for failure_mechanism in failure_mechanism_list
            }   for interface_name in cfg_dict.keys()}
            self.die_stack_yield = 1.0
        else:
            raise ValueError("Invalid mode. Please specify 'simulation' or 'modeling' in the mode argument.")

    def get_die_stack_yield(self):
        """
        Calculate the yield for each die stack based on the failure parameters of each interface.
        """
        # For yield modeling, calculate the die stack yield based on the die yield of each interface
        for interface_name in self.cfg_dict.keys():
            self.die_yield_per_interface_dict[interface_name]['overall'] = self.die_yield_per_interface_dict[interface_name]['overlay'] * \
                self.die_yield_per_interface_dict[interface_name]['particle'] * \
                self.die_yield_per_interface_dict[interface_name]['mechanical'] * \
                self.die_yield_per_interface_dict[interface_name]['ESD']
            # Calculate the overall die stack yield by multiplying the die yield of all interfaces
            self.die_stack_yield *= self.die_yield_per_interface_dict[interface_name]['overall']

        return self.die_stack_yield


def die_stack_list_initialize(
    cfg_dict: dict,
    pad_bitmap_collection_dict: dict,
    num_stack_samples: int,
    base_pad_coords_flag: bool = False,
    mode: str = 'simulation',
):
    """
    Inputs:
    - cfg_dict: Configuration object containing parameters
    - num_stack_samples: Number of die stack samples to generate

    Outputs:
    - die_stack: DieStack object containing the initialized die stack samples
    """
    die_stack_list = []
    for _ in range(num_stack_samples):
        die_stack = DieStack(
            cfg_dict=cfg_dict,
            pad_bitmap_collection_dict=pad_bitmap_collection_dict,
            mode=mode,
            base_pad_coords_flag=base_pad_coords_flag,
        )
        die_stack_list.append(die_stack)
    if base_pad_coords_flag:
        base_pad_coords_dict = die_stack_list[0].interfaces.base_pad_coords_dict
        return die_stack_list, base_pad_coords_dict
    else:
        return die_stack_list
