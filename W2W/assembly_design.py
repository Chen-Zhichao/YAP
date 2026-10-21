"""Bridge normal W2W design files to history-dependent assembly-order runs."""

from datetime import datetime
import csv
import hashlib
from pathlib import Path
import shutil
import sys
from time import perf_counter
from uuid import uuid4

import numpy as np
from omegaconf import OmegaConf

from .assembly_order import (
    _number, _write_new, _check_recipe_request, build_plan, content_id, evaluate_archive, load_stack_input,
)


def load_design(cfg, design_dir, design_name):
    from .compact_layout import compact_regular_geometry
    from .overlay_yield_calculator import max_allowed_misalignment_calculator
    from .wafer_die_stack_initialization import Wafer_Interface

    directory = Path(design_dir).resolve()
    paths = [directory / name for name in (
        "stack.yaml", "generated_chiplet_definitions.3dbv",
        "generated_stack_config.3dbx", "compact_layout.yaml")]
    design = load_stack_input(paths[0])
    if "overlay" in design:
        raise ValueError("normal design mode derives overlay from its config/layout; remove inline overlay")
    plan = build_plan(design["stack"])
    n = len(plan["context"]["initial_bows_um"])
    definitions = OmegaConf.load(paths[1]).ChipletDef
    topology = OmegaConf.load(paths[2])
    layout = OmegaConf.load(paths[3])
    operator_request = None
    if str(design.get("mechanics", {}).get("operator_package", "")).startswith("component:"):
        from component_operators import w2w_operator_request
        operator_request = w2w_operator_request(design["mechanics"]["operator_package"],
            plan["context"]["geometry"], frozen_preview=design["mechanics"].get("frozen_preview", False),
            layout=(layout.pitch_r_um, layout.pitch_c_um, layout.pad_diameter_um))
    if design.get("mechanics", {}).get("frozen_preview", False):
        name = Path(design["mechanics"]["operator_package"]).name
        expected = {"BI2_50.json": (2., 1.), "BI6_50.json": (6., 3.)}.get(name)
        if expected is None or (layout.pitch_r_um, layout.pitch_c_um, layout.pad_diameter_um) != (
                expected[0], expected[0], expected[1]):
            raise ValueError("reference preview mechanics and pad layout must use the same p2/d1 or p6/d3 geometry")
    names = [f"Wafer_{i}" for i in range(1, n + 1)]
    if str(topology.Design.name) != design_name:
        raise ValueError("3Dblox design name differs from --ds_name")
    if set(definitions) != set(names) or set(topology.ChipletInst) != set(names) or set(topology.Stack) != set(names):
        raise ValueError("assembly design must contain exactly Wafer_1..N, without an extra base")
    if set(topology.Connection) != {f"Conn{i}" for i in range(1, n)}:
        raise ValueError("assembly design requires exactly N-1 adjacent interfaces")
    radius = _number(cfg.WAF_R_um, "WAF_R_um", positive=True)
    if radius != 500 * plan["context"]["geometry"]["diameter_mm"]:
        raise ValueError("wafer radius in config and stack.yaml disagree")
    initial = plan["context"]["geometry"]["initial_total_thickness_um"]
    area = list(definitions[names[0]].design_area)
    if len(area) != 2:
        raise ValueError("die-site design_area must have two dimensions")
    dimensions = [_number(v, "die-site dimension", positive=True) for v in area]
    for i, name in enumerate(names):
        definition, node = definitions[name], topology.Stack[name]
        if list(definition.design_area) != area or definition.thickness != initial:
            raise ValueError("all initial wafer definitions must match stack.yaml and share die-site dimensions")
        if topology.ChipletInst[name].reference != name or node.get("orient", "R0") != "R0" or list(node.loc) != [0., 0.]:
            raise ValueError("only aligned, fixed-orientation wafer instances are supported")
        if node.z != i * initial:
            raise ValueError("3Dblox z coordinates must describe the initial total-thickness ledger")
        dbf = directory / f"{name}.3dbf"
        paths.append(dbf)
        technology = OmegaConf.load(dbf)
        bump = technology.Bump_Types.silicon_individual_bonding
        if technology.Chiplet_Grid.pitch != layout.pitch_r_um or layout.pitch_r_um != layout.pitch_c_um \
                or bump.bump_size != layout.pad_diameter_um:
            raise ValueError("3DBF and compact-layout pitch/pad diameter disagree")
        if i:
            connection = topology.Connection[f"Conn{i}"]
            if connection.top != f"{name}.regions.From_Wafer_{i}" \
                    or connection.bot != f"Wafer_{i}.regions.To_{name}":
                raise ValueError("connections must describe adjacent physical wafer interfaces")
            if definition.regions[f"From_Wafer_{i}"].side != "front" \
                    or definitions[f"Wafer_{i}"].regions[f"To_{name}"].side != "back":
                raise ValueError("input region declarations must match the existing logical pad-map registration")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.DIE_W_um, cfg.DIE_L_um = dimensions
    cfg.PITCH_r_um, cfg.PITCH_c_um = float(layout.pitch_r_um), float(layout.pitch_c_um)
    cfg.PAD_TOP_R_um = cfg.PAD_BOT_R_um = float(layout.pad_diameter_um) / 2
    cfg.LAYOUT_KIND = str(layout.layout_kind)
    for field, key in (("CRITICAL_PAD_FRACTION", "critical_pad_fraction"),
                       ("POWER_GROUND_PAD_FRACTION", "power_ground_pad_fraction"),
                       ("DUMMY_PAD_FRACTION", "dummy_pad_fraction")):
        if field in cfg and cfg[field] != layout[key]:
            raise ValueError(f"{field} differs between config and layout")
        cfg[field] = layout[key]
    geometry = compact_regular_geometry(cfg)
    for key in ("rows", "cols", "pad_array_l_um", "pad_array_w_um"):
        if layout[key] != geometry[key]:
            raise ValueError(f"compact layout {key} does not match dimensions and pitch")
    threshold = max_allowed_misalignment_calculator(cfg, cfg.PAD_TOP_R_um, cfg.PAD_BOT_R_um,
        cfg.PITCH_r_um, cfg.PITCH_c_um, cfg.CONTACT_AREA_CONSTRAINT, cfg.CRITICAL_DIST_CONSTRAINT)
    boundary = geometry["critical_pad_boundary_coords"]
    if cfg.CRITICAL_PAD_FRACTION <= 0:
        raise ValueError("overlay scoring needs a nonempty critical region")
    vertices = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]) * np.asarray(dimensions) / 2
    wafer = Wafer_Interface(radius, *dimensions, geometry["rows"], geometry["cols"],
        cfg.PAD_TOP_R_um, cfg.PAD_BOT_R_um, np.zeros((1, 2)),
        _number(cfg.dice_width, "dice_width", nonnegative=True), False)
    wafer.generate_die(geometry["num_critical_pads"], vertices, boundary)
    sites = np.asarray([die.die_center for die in wafer.die_list])
    if not len(sites):
        raise ValueError("no complete die sites fit on the wafer")
    samples, seed = cfg.OVERLAY_SAMPLES, cfg.RANDOM_SEED
    if type(samples) is not int or samples < 2 or type(seed) is not int or seed < 0:
        raise ValueError("OVERLAY_SAMPLES >= 2 and RANDOM_SEED >= 0 must be integers")
    shifts, angles = [], []
    for interface in range(1, n):
        z = np.random.default_rng(np.random.SeedSequence([seed, interface])).standard_normal((samples, 3))
        mean = np.array([_number(cfg[f"SYSTEM_TRANSLATION_{axis}_MEAN_um"], "translation mean") for axis in ("X", "Y")])
        std = np.array([_number(cfg[f"SYSTEM_TRANSLATION_{axis}_STD_um"], "translation std", nonnegative=True) for axis in ("X", "Y")])
        shifts.append(mean + std * z[:, :2])
        angles.append(_number(cfg.SYSTEM_ROTATION_MEAN_rad, "rotation mean") +
                      _number(cfg.SYSTEM_ROTATION_STD_rad, "rotation std", nonnegative=True) * z[:, 2])
    if _number(cfg.RANDOM_MISALIGNMENT_MEAN_um, "residual mean") != 0:
        raise ValueError("reference signed-scalar residual model requires zero-mean local noise")
    overlay = dict(coordinates_um=(sites[:, None, :] + boundary[None]).tolist(),
        critical_pad_mask=[True] * len(boundary), site_mask=[True] * len(sites),
        threshold_um=float(threshold), residual_sigma_um=float(cfg.RANDOM_MISALIGNMENT_STD_um),
        translation_um=np.stack(shifts, axis=1).tolist(), rotation_rad=np.stack(angles, axis=1).tolist(),
        magnification_correction_fraction=float(cfg.MAGNIFICATION_CORRECTION),
        conditional_interface_independence=True)
    from .assembly_order import _overlay_settings
    _overlay_settings(overlay, n, radius)
    resolved = {}
    for i in range(1, n):
        record = OmegaConf.to_container(cfg, resolve=True)
        record.update(INTERFACE=f"Wafer_{i+1}_From_Wafer_{i}", physical_interface=i,
                      PAD_ARR_ROW=geometry["rows"], PAD_ARR_COL=geometry["cols"],
                      ITF_TOP_THICK_um=initial, ITF_BOT_THICK_um=initial)
        resolved[record["INTERFACE"]] = record
    provenance = dict(design=design_name, input_sha256={
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        effective_config=OmegaConf.to_container(cfg, resolve=True),
        physical_interfaces=[f"{i}--{i+1}" for i in range(1, n)],
        physical_face_selection="assembly process ledger; 3Dblox regions register logical pad layout only",
        site_count=len(sites), critical_pad_count=geometry["num_critical_pads"],
        critical_geometry="production_compact_central_rectangle_convex_boundary",
        placement_seed=seed, placement_seed_rule="SeedSequence([seed, physical_interface])",
        experiment="new_fixed_bow_W2W_assembly_orders_not_REFERENCE_population_reproduction",
        overlay_input_sha256=content_id(overlay), config_thickness_scope="initial; event ledger controls thinning")
    if operator_request is not None:
        provenance["operator_request"] = operator_request
    return design, plan, overlay, resolved, provenance


def _memory_gate(minimum_gib):
    minimum = _number(minimum_gib, "MIN_AVAILABLE_MEMORY_GIB", positive=True)
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        fields[key] = int(value.split()[0])
    available = fields["MemAvailable"] / 1024**2
    if available < minimum:
        raise RuntimeError(f"available RAM {available:.1f} GiB below run gate {minimum:g} GiB")
    return available


def write_summary(path, result):
    fields = ["order", "interface", "interface_overlay_yield", "stack_overlay_yield", "event",
              "m_xx_ppm", "m_xy_ppm", "m_yx_ppm", "m_yy_ppm",
              "residual_m_xx_ppm", "residual_m_xy_ppm", "residual_m_yx_ppm", "residual_m_yy_ppm",
              "max_bow_overlay_nm", "max_residual_bow_overlay_nm", "non_bow_overlay_sha256"]
    with Path(path).open("x", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for route in result["routes"]:
            for interface in route["interfaces"]:
                row = dict(order=route["order"], interface=interface["interface_id"],
                    event=interface["event"], interface_overlay_yield=interface["overlay_yield"],
                    stack_overlay_yield=route["stack_overlay_yield"])
                for prefix, key in (("m_", "magnification_ppm"), ("residual_m_", "residual_magnification_ppm")):
                    for i, j, label in ((0, 0, "xx"), (0, 1, "xy"), (1, 0, "yx"), (1, 1, "yy")):
                        row[prefix + label + "_ppm"] = interface[key][i][j]
                row.update({key: interface[key] for key in fields[-3:]})
                writer.writerow(row)


def run_design(cfg, design_dir, design_name, *, plan_only=False, output_dir=None, config_path=None,
               design_loader=None, operator_loader=None):
    from mechanics_runtime import configure_runtime_storage, validate_write_path

    configure_runtime_storage()
    from threadpoolctl import threadpool_limits

    design, plan, overlay, resolved, provenance = (design_loader or load_design)(cfg, design_dir, design_name)
    from .assembly_order import is_die_plan, is_f2b_plan
    metric_model = "square_affine" if is_die_plan(plan) else "disk"
    settings = design.get("mechanics")
    if not settings or settings.get("metric") != {"model": metric_model}:
        raise ValueError(f"normal assembly-order mode requires mechanics.metric: {{model: {metric_model}}}")
    if settings.get("layer_workers", 1) != 1:
        raise ValueError("this design runner currently requires one layer worker")
    order_workers = settings.get("order_workers", 1)
    if type(order_workers) is not int or order_workers < 1:
        raise ValueError("mechanics.order_workers must be a positive integer")
    _check_recipe_request(plan, settings["temporal_profile"])
    root = Path(output_dir or cfg.OUTPUT_DIR)
    if output_dir is None and not root.is_absolute():
        domain = "D2W" if is_die_plan(plan) else "W2W"
        root = Path(__file__).resolve().parents[1] / domain / root
    root = validate_write_path(root)
    if str(settings.get("operator_package", "")).startswith("component:"):
        from component_operators import canonical_write_path
        root = canonical_write_path(root)
    run = validate_write_path(root / design_name / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]))
    run.mkdir(parents=True, exist_ok=False)
    print(f"Assembly-order run: {run}", flush=True)
    if config_path:
        path = Path(config_path).resolve()
        provenance["config_path"] = str(path)
        provenance["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    frozen = run / "input_snapshot"
    frozen.mkdir()
    for filename in provenance["input_sha256"]:
        shutil.copy2(filename, frozen / Path(filename).name)
    if config_path:
        shutil.copy2(config_path, frozen / "calculator_config.yaml")
    project = Path(__file__).resolve().parents[2]
    sources = {Path(__file__).resolve(), Path(__file__).with_name("calculator_main.py")}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(project) and path.suffix == ".py":
                sources.add(path)
    provenance["source_sha256"] = {}
    for path in sorted(sources):
        target = run / "source_snapshot" / path.relative_to(project)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        provenance["source_sha256"][str(path)] = hashlib.sha256(target.read_bytes()).hexdigest()
    _write_new(run / "INPUTS.json", provenance)
    _write_new(run / "PLAN.json", plan)
    OmegaConf.save(OmegaConf.create(design), run / "stack.resolved.yaml")
    for name, record in resolved.items():
        OmegaConf.save(OmegaConf.create(record), run / f"{name}.yaml")
    if plan_only:
        print(f"Planned {len(plan['routes'])} orders; no mechanics or yield evaluated.")
        print(f"Order concurrency: requested {order_workers}, effective {min(order_workers, len(plan['routes']))}.")
        if "summary" in plan:
            summary = plan["summary"]
            print(f"{summary['defined_routes']} defined process ledgers; "
                  f"{summary['missing_process_definitions']} missing process definitions.")
            print("New mixed-face W2W: orientation and native qualification pending; execution blocked.")
        elif is_f2b_plan(plan):
            print("F2B only; 775-um birth, lower-back preparation before every bond, final outer-back thinning.")
            print("Component execution checks pure-Si caps; the explicit P24 preview remains available. Assembly accuracy is pending.")
        request = provenance.get("operator_request", {})
        if request.get("admission_status") == "UNAVAILABLE_EXACT_P2_OPERATORS_NOT_ADMITTED":
            print("Execution unavailable: exact p2 operators for this thickness have not been built and admitted.")
        return run
    started = perf_counter()
    try:
        from .assembly_order import _check_plan
        _check_plan(plan)
        if "operator_request" in provenance:
            if is_f2b_plan(plan) and str(settings["operator_package"]).startswith("component:"):
                from component_operators import w2w_operator_request
                w2w_operator_request(settings["operator_package"], plan["context"]["geometry"],
                                     frozen_preview=settings.get("frozen_preview", False))
            else:
                from .assembly_operator_library import require_prepared_operator
                require_prepared_operator(settings["operator_package"], plan["context"]["geometry"],
                                          frozen_preview=settings.get("frozen_preview", False))
        from .assembly_mechanics import run_native, _recipe, _check_metric_policy
        from .assembly_operator_library import load_assembly_library, load_f2b_assembly_library
        _recipe(plan, settings["temporal_profile"])
        _check_metric_policy(plan, settings["metric"])
        minimum = cfg.get("MIN_AVAILABLE_MEMORY_GIB", 40.)
        _memory_gate(minimum)
        def progress(message):
            _memory_gate(minimum)
            print(message, flush=True)
            with (run / "progress.log").open("a", encoding="ascii") as stream:
                stream.write(message + "\n")
        def completed(archive, route):
            number = next(i for i, r in enumerate(plan["routes"], 1) if r["request_id"] == route["request_id"])
            partial = dict(archive, routes=[route], status="COMPLETE", physical_readiness=False)
            _write_new(run / f"route_{number}_mechanics.json", partial)
            progress(f"Completed order {number}/{len(plan['routes'])}: {route['order']}")
        with threadpool_limits(limits=1):
            if order_workers > 1:
                from .assembly_parallel import run_parallel
                progress(f"Executing orders with up to {order_workers} isolated worker processes.")
                archive = run_parallel(plan, settings, work_dir=run / "order_workers",
                    minimum_memory_gib=minimum, progress=progress, route_complete=completed)
            else:
                progress("Loading the selected checked layer operators once.")
                options = {"frozen_preview": True} if settings.get("frozen_preview", False) else {}
                loader = operator_loader or (load_f2b_assembly_library if is_f2b_plan(plan) else load_assembly_library)
                library, operators = loader(settings["operator_package"], plan["context"]["geometry"], **options)
                progress("Layer operators loaded; executing complete Cu histories.")
                archive = run_native(plan, library, operators, temporal_profile=settings["temporal_profile"],
                    metric=settings["metric"], progress=progress, route_complete=completed)
        # Native modules and the dynamically loaded metric may arrive after the
        # input snapshot. Preserve their exact sources without rewriting receipts.
        late_sources = set()
        try:
            from YAP.history_mechanics_backend import ANSYS_VALIDATION
        except ModuleNotFoundError:
            from history_mechanics_backend import ANSYS_VALIDATION
        native_root = ANSYS_VALIDATION.resolve()
        for module in tuple(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if path.suffix == ".py" and (path.is_relative_to(project) or path.is_relative_to(native_root)):
                    late_sources.add(path)
        late_sources.update(Path(p) for p in archive.get("native_configuration", {}).get("metric_implementation_sha256", {}))
        worker_sources = archive.get("native_configuration", {}).get("worker_runtime_sources", {})
        for filename, expected in worker_sources.items():
            path = Path(filename).resolve()
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"worker source changed during assembly execution: {path}")
            if not (path.is_relative_to(project) or path.is_relative_to(native_root)):
                raise ValueError(f"worker source is outside the project/native runtime: {path}")
            late_sources.add(path)
        runtime_hashes = {}
        for path in sorted(late_sources):
            relative = path.relative_to(project) if path.is_relative_to(project) else Path("YAP/runtime/native") / path.relative_to(native_root)
            target = run / "source_snapshot" / relative
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"source changed during assembly execution: {path}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            runtime_hashes[str(path)] = digest
        _write_new(run / "RUNTIME_SOURCES.json", runtime_hashes)
        _write_new(run / "MECHANICS.json", archive)
        result = evaluate_archive(plan, archive, overlay)
        result["status"] = "NATIVE_MECHANICS_OVERLAY_UNVALIDATED"
        result["design_inputs"] = provenance
        result["wall_seconds"] = perf_counter() - started
        _write_new(run / "RESULT.json", result)
        write_summary(run / "SUMMARY.csv", result)
        print("Best among supplied orders: " + ", ".join(result["best_supplied_orders"]), flush=True)
        print("Scope: overlay only; new assembly paths are not independently validated.", flush=True)
        return run
    except Exception as exc:
        _write_new(run / "FAILED.json", dict(status="FAILED_NO_COMPLETE_RANKING",
            error_type=type(exc).__name__, message=str(exc), wall_seconds=perf_counter() - started))
        raise
