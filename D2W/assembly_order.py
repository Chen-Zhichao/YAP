"""Prepared-die assembly trees using the all-Gauss Cu-history engine.

This is a free, equal-square die-site column, not a finite receiving wafer or
carrier model. Offline substack joins include D2D bonding. Only the
signed-scalar worst-critical-pad overlay surrogate is scored.
"""

from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from W2W.assembly_order import (
    CONVENTION, DIE_FRAME, DIE_PLAN_SCHEMA, MAX_WAFERS, _PROCESS_DEFAULTS,
    _array, _keys, _label, _number, _overlay_settings, all_orders, content_id,
    leaves, parse_order, tree_text, load_stack_input as _load_stack_input,
    evaluate_archive,
    ManufacturingLayer, clpt_source_response, strain_gradient_to_signed_edge_bow_um,
)


INPUT_SCHEMA = "yap.d2w_assembly_order_input.v1"
STACK_SCHEMA = "yap.d2w_stack_input.v1"


def load_stack_input(path):
    return _load_stack_input(path, stack_schema=STACK_SCHEMA, input_schema=INPUT_SCHEMA)


def build_plan(config):
    """Plan fixed layer-order trees; every die retains its prefabricated layers."""
    _keys(config, {"schema", "initial_bows_um", "geometry", "fabrication_layers", "process",
                   "assembly_orders"}, "D2W stack", {"description", "allow_offline_substacks"})
    if config["schema"] != INPUT_SCHEMA:
        raise ValueError("unsupported D2W assembly-order schema")
    bows = config["initial_bows_um"]
    if not isinstance(bows, list) or not 2 <= len(bows) <= MAX_WAFERS:
        raise ValueError("supply 2..32 fixed initial die bows")
    bows = [_number(v, "initial bow") for v in bows]
    _keys(config["geometry"], {"width_mm", "length_mm", "total_thickness_um",
          "front_bi_thickness_um", "back_bi_thickness_um", "beol_thickness_um"}, "die geometry")
    geometry = {k: _number(v, k, positive=True) for k, v in config["geometry"].items()}
    if geometry["width_mm"] != geometry["length_mm"]:
        raise ValueError("the existing reference scalar membrane metric requires equal square die footprints")
    retained = geometry["total_thickness_um"] - sum(geometry[k] for k in
        ("front_bi_thickness_um", "back_bi_thickness_um", "beol_thickness_um"))
    if retained <= 0:
        raise ValueError("total die thickness must include both BIs, BEOL and positive retained Si")
    raw_layers = config["fabrication_layers"]
    if not isinstance(raw_layers, list) or len(raw_layers) != 4:
        raise ValueError("prepared fabrication_layers must be [front BI, BEOL, Si, back BI]")
    layers = []
    for raw in raw_layers:
        _keys(raw, {"name", "thickness_um", "e_mpa", "nu", "eigenstrain_per_amplitude"}, "fabrication layer")
        layers.append(ManufacturingLayer(name=_label(raw["name"], "layer name"),
            **{k: _number(v, k) for k, v in raw.items() if k != "name"}))
    expected = [geometry["front_bi_thickness_um"], geometry["beol_thickness_um"], retained,
                geometry["back_bi_thickness_um"]]
    if len({v.name for v in layers}) != 4 or not np.allclose(
            [v.thickness_um for v in layers], expected, rtol=0., atol=1e-9):
        raise ValueError("fabrication layers disagree with the immutable total-thickness ledger")
    if any(layers[i].eigenstrain_per_amplitude != 0 for i in (0, 2, 3)):
        raise ValueError("the existing fabrication source is confined to BEOL")
    process = dict(config["process"])
    _keys(process, set(_PROCESS_DEFAULTS) | {"room_temperature_c", "anneal_temperature_c",
          "ramp_up_s", "dwell_s", "cool_down_s"}, "process")
    for k, v in _PROCESS_DEFAULTS.items():
        if k != "recipe_id" and process[k] != v:
            raise ValueError(f"unsupported die assembly policy {k}")
    _label(process["recipe_id"], "recipe_id")
    for k in ("room_temperature_c", "anneal_temperature_c", "ramp_up_s", "dwell_s", "cool_down_s"):
        process[k] = _number(process[k], k, positive=k.endswith("_s"))
    if process["anneal_temperature_c"] <= process["room_temperature_c"]:
        raise ValueError("anneal temperature must exceed room temperature")
    offline = config.get("allow_offline_substacks", False)
    if type(offline) is not bool:
        raise ValueError("allow_offline_substacks must be Boolean")
    count = len(bows)
    selection = config["assembly_orders"]
    if selection == "all":
        orders = all_orders(count)
    elif isinstance(selection, str):
        orders = (parse_order(selection, count),)
    elif isinstance(selection, list) and selection:
        orders = tuple(parse_order(v, count) for v in selection)
    else:
        raise ValueError("assembly_orders must be all, one quoted tree, or a nonempty list")
    if len(set(orders)) != len(orders):
        raise ValueError("duplicate assembly orders")
    context = dict(body_kind="prepared_die_column", initial_bows_um=bows, geometry=geometry,
        fabrication_layers=[asdict(v) for v in layers], process=process, frame_id=DIE_FRAME,
        bow_convention=CONVENTION, orientation="lower_BACK_to_upper_FRONT",
        thickness_scope="total_including_BEOL_and_both_prefabricated_BIs",
        allow_offline_substacks=offline, receiving_wafer_model="free_die_site_column_only")
    response = clpt_source_response(layers)
    unit = strain_gradient_to_signed_edge_bow_um(response["kappa_per_amplitude_per_mm"], geometry["width_mm"])
    if not np.isfinite(unit) or unit == 0:
        raise ValueError("fabrication source has no finite nonzero bow response")
    births = dict(model="BEOL_eigenstrain_linear_ABD_inverse", source_response=response,
        dies=[dict(die=i, initial_bow_um=bow, source_amplitude=bow / unit,
                   initial_q6=(bow / unit * np.asarray(response["natural_birth_q6_per_amplitude"])).tolist(),
                   reconstruction_bow_um=bow) for i, bow in enumerate(bows, 1)],
        redraw_per_route=False, recalibrate_after_thinning=False,
        status="CALIBRATION_ONLY_NOT_RESIDUAL_STRESS_HISTORY_INITIALIZATION")
    context_id = content_id(dict(context=context, initialization=births))

    def state_id(tree, phase):
        return content_id(dict(context_id=context_id, tree=tree_text(tree), phase=phase))

    routes = []
    for tree in orders:
        events = []

        def visit(node):
            if type(node) is int:
                return dict(die_ids=[node], total_thicknesses_um=[geometry["total_thickness_um"]],
                            thermal_cycle_counts=[0], state_id=state_id(node, "initial_free"))
            if not offline and leaves(node)[0] != 1:
                raise ValueError("this order requires offline D2D substack assembly; set allow_offline_substacks: true")
            lower, upper = visit(node[0]), visit(node[1])
            k = lower["die_ids"][-1]
            result = dict(die_ids=lower["die_ids"] + upper["die_ids"],
                total_thicknesses_um=lower["total_thicknesses_um"] + upper["total_thicknesses_um"],
                thermal_cycle_counts=[v + 1 for v in lower["thermal_cycle_counts"] + upper["thermal_cycle_counts"]],
                state_id=state_id(node, "post_anneal_cooldown_free"))
            events.append(dict(event=f"A{len(events) + 1}", subtree=tree_text(node),
                interface_id=f"{k}.BACK--{k+1}.FRONT", lower_before_preparation=lower,
                lower_free=lower, upper_free=upper, preparation="prefabricated_back_BI_no_thinning_or_deposition",
                bonding_scope="receiving_die_site" if leaves(node)[0] == 1 else "offline_D2D_substack",
                operations=["observe_both_released_prebond_states", "independent_ideal_flattening",
                    "capture_both_child_references_and_Cu_histories", "bond_lower_BACK_to_upper_FRONT",
                    "release_flattening_constraints", "free_anneal", "free_cooldown"], result=result))
            return result

        final = visit(tree)
        route = dict(order=tree_text(tree), context_id=context_id, events=events, final=final)
        route["request_id"] = content_id(route)
        routes.append(route)
    return dict(schema=DIE_PLAN_SCHEMA, status="PLANNED_MECHANICS_REQUIRED", physical_readiness=False,
        context_id=context_id, context=context, initialization=births, routes=routes,
        live_mechanics_backend="shared_REFERENCE_all_Gauss_prepared_die_tree_adapter")


def load_design(cfg, design_dir, design_name):
    from .compact_layout import compact_regular_geometry
    from .overlay_yield_calculator import max_allowed_misalignment_calculator
    from W2W.assembly_operator_library import check_prepared_die_geometry, prepared_operator_request

    directory = Path(design_dir).resolve()
    paths = [directory / name for name in ("stack.yaml", "generated_chiplet_definitions.3dbv",
             "generated_stack_config.3dbx", "compact_layout.yaml")]
    design = load_stack_input(paths[0])
    if "overlay" in design:
        raise ValueError("normal design mode derives overlay from config/layout; remove inline overlay")
    plan = build_plan(design["stack"])
    if cfg.get("ANNEAL_MODE") != "sequential":
        raise ValueError("D2W assembly trees require explicit ANNEAL_MODE: sequential, not legacy batch anneal")
    geometry = plan["context"]["geometry"]
    mechanics = design.get("mechanics", {})
    if not str(mechanics.get("operator_package", "")).startswith("component:"):
        check_prepared_die_geometry(geometry)
    count = len(plan["context"]["initial_bows_um"])
    names = [f"Die_{i}" for i in range(1, count + 1)]
    definition_file = OmegaConf.load(paths[1])
    definitions = definition_file.ChipletDef
    topology = OmegaConf.load(paths[2])
    if definition_file.Header.unit != "micron" or topology.Header.unit != "micron":
        raise ValueError("D2W 3Dblox definitions and topology must use micron units")
    layout = OmegaConf.load(paths[3])
    if topology.Design.name != design_name or any(set(section) != set(names) for section in
            (definitions, topology.ChipletInst, topology.Stack)):
        raise ValueError("assembly design must contain exactly the declared Die_1..N, with no hidden wafer/base")
    if set(topology.Connection) != {f"Conn{i}" for i in range(1, count)}:
        raise ValueError("D2W assembly design requires exactly N-1 adjacent interfaces")
    dimensions = [geometry["width_mm"] * 1000, geometry["length_mm"] * 1000]
    for i, name in enumerate(names):
        definition, node = definitions[name], topology.Stack[name]
        if definition.type != "die" or list(definition.design_area) != dimensions or definition.thickness != geometry["total_thickness_um"]:
            raise ValueError("all prepared dies must share the stack.yaml footprint and total thickness")
        if topology.ChipletInst[name].reference != name or node.get("orient", "R0") != "R0" or list(node.loc) != [0., 0.] \
                or node.z != i * geometry["total_thickness_um"]:
            raise ValueError("only aligned, fixed-orientation F2B die columns are supported")
        if node.get("root", False) is not (i == 0):
            raise ValueError("only Die_1 may be the receiving die-site root")
        expected_regions = ({f"From_Die_{i}"} if i else set()) | ({f"To_Die_{i+2}"} if i < count - 1 else set())
        if set(definition.regions) != expected_regions:
            raise ValueError("die regions must describe only adjacent F2B interfaces")
        for region_name, region in definition.regions.items():
            face = "front" if region_name.startswith("From_") else "back"
            if region.side != face or list(map(list, region.coords)) != [[0, 0], [dimensions[0], 0], dimensions, [0, dimensions[1]]]:
                raise ValueError("F2B regions must cover the common aligned die footprint")
        dbf = directory / f"{name}.3dbf"
        paths.append(dbf)
        technology = OmegaConf.load(dbf)
        if technology.Chiplet_Grid.pitch != layout.pitch_r_um or layout.pitch_r_um != layout.pitch_c_um \
                or technology.Bump_Types.silicon_individual_bonding.bump_size != layout.pad_diameter_um:
            raise ValueError("3DBF and compact-layout pitch/pad diameter disagree")
        if i:
            connection = topology.Connection[f"Conn{i}"]
            if connection.top != f"{name}.regions.From_Die_{i}" or connection.bot != f"Die_{i}.regions.To_{name}":
                raise ValueError("connections must join adjacent lower BACK to upper FRONT")
    mechanics = design.get("mechanics", {})
    operator_request = prepared_operator_request(mechanics.get("operator_package", ""), geometry,
        frozen_preview=mechanics.get("frozen_preview", False),
        layout=(layout.pitch_r_um, layout.pitch_c_um, layout.pad_diameter_um))
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.DIE_W_um, cfg.DIE_L_um = dimensions
    cfg.PITCH_r_um, cfg.PITCH_c_um = float(layout.pitch_r_um), float(layout.pitch_c_um)
    cfg.PAD_TOP_R_um = cfg.PAD_BOT_R_um = float(layout.pad_diameter_um) / 2
    cfg.LAYOUT_KIND = str(layout.layout_kind)
    for field, key in (("CRITICAL_PAD_FRACTION", "critical_pad_fraction"),
                       ("POWER_GROUND_PAD_FRACTION", "power_ground_pad_fraction"), ("DUMMY_PAD_FRACTION", "dummy_pad_fraction")):
        if field in cfg and cfg[field] != layout[key]:
            raise ValueError(f"{field} differs between config and layout")
        cfg[field] = layout[key]
    compact = compact_regular_geometry(cfg)
    for key in ("rows", "cols", "pad_array_l_um", "pad_array_w_um"):
        if layout[key] != compact[key]:
            raise ValueError(f"compact layout {key} differs from footprint and pitch")
    if compact["num_critical_pads"] <= 0:
        raise ValueError("overlay scoring needs a nonempty critical region")
    boundary = compact["critical_pad_boundary_coords"]
    threshold = max_allowed_misalignment_calculator(cfg=cfg, PAD_TOP_R_um=cfg.PAD_TOP_R_um,
        PAD_BOT_R_um=cfg.PAD_BOT_R_um, PITCH_r_um=cfg.PITCH_r_um, PITCH_c_um=cfg.PITCH_c_um,
        CONTACT_AREA_CONSTRAINT=cfg.CONTACT_AREA_CONSTRAINT, CRITICAL_DIST_CONSTRAINT=cfg.CRITICAL_DIST_CONSTRAINT)
    samples, seed = cfg.OVERLAY_SAMPLES, cfg.RANDOM_SEED
    if type(samples) is not int or samples < 2 or type(seed) is not int or seed < 0:
        raise ValueError("OVERLAY_SAMPLES >= 2 and RANDOM_SEED >= 0 must be integers")
    if _number(cfg.RANDOM_MISALIGNMENT_MEAN_um, "residual mean") != 0:
        raise ValueError("reference signed-scalar residual noise must have zero mean")
    shifts, angles = [], []
    for interface in range(1, count):
        z = np.random.default_rng(np.random.SeedSequence([seed, interface])).standard_normal((samples, 3))
        mean = np.array([_number(cfg[f"SYSTEM_TRANSLATION_{a}_MEAN_um"], "translation mean") for a in ("X", "Y")])
        std = np.array([_number(cfg[f"SYSTEM_TRANSLATION_{a}_STD_um"], "translation sigma", nonnegative=True) for a in ("X", "Y")])
        shifts.append(mean + std * z[:, :2])
        angles.append(_number(cfg.SYSTEM_ROTATION_MEAN_rad, "rotation mean") +
                      _number(cfg.SYSTEM_ROTATION_STD_rad, "rotation sigma", nonnegative=True) * z[:, 2])
    overlay = dict(coordinates_um=boundary[None].tolist(), critical_pad_mask=[True] * len(boundary),
        site_mask=[True], threshold_um=float(threshold), residual_sigma_um=float(cfg.RANDOM_MISALIGNMENT_STD_um),
        translation_um=np.stack(shifts, axis=1).tolist(), rotation_rad=np.stack(angles, axis=1).tolist(),
        magnification_correction_fraction=float(cfg.MAGNIFICATION_CORRECTION), conditional_interface_independence=True)
    _overlay_settings(overlay, count, None, halfspan_um=np.asarray(dimensions) / 2)
    resolved = {}
    for i in range(1, count):
        record = OmegaConf.to_container(cfg, resolve=True)
        record.update(INTERFACE=f"Die_{i+1}_From_Die_{i}", physical_interface=i,
                      PAD_ARR_ROW=compact["rows"], PAD_ARR_COL=compact["cols"],
                      ITF_TOP_THICK_um=geometry["total_thickness_um"], ITF_BOT_THICK_um=geometry["total_thickness_um"])
        resolved[record["INTERFACE"]] = record
    provenance = dict(design=design_name, input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        effective_config=OmegaConf.to_container(cfg, resolve=True), physical_interfaces=[f"{i}.BACK--{i+1}.FRONT" for i in range(1, count)],
        site_count=1, critical_pad_count=compact["num_critical_pads"], critical_geometry="compact_central_rectangle_boundary_surrogate",
        coordinate_frame="die_centered_no_wafer_site_offset", placement_seed=seed,
        placement_seed_rule="SeedSequence([seed, physical_interface])", overlay_input_sha256=content_id(overlay),
        operator_request=operator_request,
        experiment="new_fixed_bow_prepared_die_column_not_REFERENCE_population_reproduction",
        receiving_wafer_model="not_explicitly_solved", probability_model="REFERENCE_signed_scalar_worst_critical_pad_not_joint_all_pad")
    return design, plan, overlay, resolved, provenance


def run_design(cfg, design_dir, design_name, **kwargs):
    from W2W.assembly_design import run_design as run_shared_design
    from W2W.assembly_operator_library import load_prepared_die_library
    return run_shared_design(cfg, design_dir, design_name, design_loader=load_design,
                             operator_loader=load_prepared_die_library, **kwargs)
