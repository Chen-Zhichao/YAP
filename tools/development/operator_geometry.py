"""Pure height geometry on a supplied pitch seed, not a pitch/FE/operator builder.

Main authenticates seed provenance and owns resources. This module performs no
file I/O, numerical imports, CLI, process launch, Cu evolution or admission.
bind_sources compiles exact native geometry and only four unchanged extension
definitions plus INTERFACE_MM. prepare_target always retargets FROM Si37.
"""
import ast
from collections import namedtuple
from copy import deepcopy
from dataclasses import replace
import hashlib
import math
from pathlib import PurePosixPath
from types import ModuleType, SimpleNamespace as NS


GEOMETRY_SHA = '4d1165905829da719742666cd78e2ceed2f8592c5ecc786b55b9aa7f6dbaf5d1'
EXTENSION_SHA = '37d7ca5f2317c130865307cc8527476e6f6bdf9127765c2a95d6503d211cd729'
SOURCE_CAP, CONTRACT_CAP, TILE = 256*1024, 16*1024, 512
MAX_NODES, MAX_ELEMENTS, MAX_EQUATIONS = 2000000, 2000000, 1000000
GeometryRequest = namedtuple('GeometryRequest', 'pitch_um total_um variant si_um diameter_um cell_um')
GeometryTarget = namedtuple('GeometryTarget', 'deck basis proof')
CONTRACT_KEYS = frozenset(('case', 'mesh_recipe', 'pitch_um', 'diameter_pitch_ratio',
    'source_si_um', 'cell_xz_um', 'physical_nodes', 'hex_elements', 'deck_nodes',
    'equations', 'constraints', 'body_coordinates', 'Cu_GP', 'region_Cu_GP', 'seed_report'))
_TOKEN = object()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _sha(value):
    return type(value) is str and len(value) == 64 and set(value) <= set('0123456789abcdef')


def bind_sources(geometry_raw, extension_raw):
    for raw, pin in ((geometry_raw, GEOMETRY_SHA), (extension_raw, EXTENSION_SHA)):
        require(type(raw) is bytes and 0 < len(raw) <= SOURCE_CAP
                and hashlib.sha256(raw).hexdigest() == pin, 'exact bounded geometry source required')
    geometry = ModuleType('_operator_height_native_geometry')
    exec(compile(geometry_raw, '<authenticated-native-periodic-geometry>', 'exec',
                 dont_inherit=True), vars(geometry))
    tree = ast.parse(extension_raw)
    names = {'require', 'retarget', 'analytic_fields', 'same_graph'}
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
                or isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'INTERFACE_MM']
    require(len(selected) == 5 and {n.name for n in selected if isinstance(n, ast.FunctionDef)} == names,
            'exact four extension definitions and interface constant required')
    extension = ModuleType('_operator_height_extension_definitions')
    vars(extension).update(geometry=geometry, math=math, replace=replace)
    exec(compile(ast.Module(body=selected, type_ignores=[]),
        '<authenticated-unchanged-height-extension>', 'exec', dont_inherit=True), vars(extension))
    require(extension.INTERFACE_MM == .0115, 'original fixed hinge required')
    return NS(geometry=geometry, extension=extension, token=_TOKEN,
              source_pins=dict(geometry=GEOMETRY_SHA, extension=EXTENSION_SHA))


def request(pitch_um, total_um, variant):
    require(type(pitch_um) is int and 2 <= pitch_um <= 10, 'integer BI pitch2..10 required')
    require(_number(total_um) and 20 <= total_um <= 100, 'continuous finite total20..100 required')
    require(type(variant) is str and variant in ('inner', 'outer'), 'explicit inner/outer variant required')
    return GeometryRequest(pitch_um, float(total_um), variant,
        float(total_um)-(13. if variant == 'inner' else 11.5), pitch_um*.5, math.lcm(pitch_um, 6))


def _contract(value, req):
    require(type(value) is dict and set(value) == CONTRACT_KEYS, 'exact small seed contract required')
    require(all(type(value[k]) is str and value[k].isascii() and value[k].isprintable()
                and 0 < len(value[k]) <= 256 for k in ('case', 'mesh_recipe')),
            'explicit seed case/mesh recipe required')
    require(type(value['pitch_um']) is int and value['pitch_um'] == req.pitch_um
            and _number(value['diameter_pitch_ratio']) and value['diameter_pitch_ratio'] == .5
            and _number(value['source_si_um']) and value['source_si_um'] == 37.
            and type(value['cell_xz_um']) is list and len(value['cell_xz_um']) == 2
            and all(_number(v) and v == req.cell_um for v in value['cell_xz_um']),
            'supplied pitch/cell/d/p/Si37 seed contract differs')
    for key in ('physical_nodes', 'hex_elements', 'deck_nodes', 'equations', 'constraints', 'body_coordinates'):
        require(type(value[key]) is int and value[key] > 0, 'positive seed count required: '+key)
    require(value['physical_nodes'] <= MAX_NODES and value['hex_elements'] <= MAX_ELEMENTS
            and value['equations'] <= MAX_EQUATIONS
            and value['deck_nodes'] <= MAX_NODES+1024
            and value['constraints'] <= 3*value['deck_nodes']
            and value['body_coordinates'] <= 3*value['physical_nodes'], 'finite geometry caps exceeded')
    gp = value['region_Cu_GP']
    require(type(gp) is dict and set(gp) == {'bi', 'beol', 'si'}
            and all(type(v) is int and v >= 0 and v % 8 == 0 for v in gp.values())
            and type(value['Cu_GP']) is int and value['Cu_GP'] == sum(gp.values()), 'exact regional Cu GP census required')
    report = value['seed_report']
    require(type(report) is dict and set(report) == {'path', 'bytes', 'sha256'}
            and type(report['path']) is str and report['path'].startswith('/')
            and report['path'].isascii() and report['path'].isprintable() and len(report['path']) <= 4096
            and type(report['bytes']) is int and 0 < report['bytes'] < 2**63 and _sha(report['sha256']),
            'Main-authenticated seed report binding required, not an authentication boolean')
    path = PurePosixPath(report['path'])
    require(path.anchor == '/' and str(path) == report['path']
            and not {'..', '#snapshot', '.snapshot'}.intersection(path.parts),
            'literal normalized nonsnapshot seed report metadata path required')
    require(2*(len(value['case'])+len(value['mesh_recipe'])+len(report['path']))+2048 <= CONTRACT_CAP,
            'fixed-schema metadata bound exceeded')


def _same_arrays(np, left, right, check):
    require(type(left) is np.ndarray and type(right) is np.ndarray
            and left.shape == right.shape and left.dtype == right.dtype, 'native array shape/dtype differs')
    for first in range(0, len(left), TILE):
        check()
        require(left[first:first+TILE].tobytes(order='C') == right[first:first+TILE].tobytes(order='C'),
                'native array byte identity differs')


def _same_basis(np, ext, supplied, measured, check):
    for name in ('node_ids', 'coordinates_mm', 'macro_fields', 'element_ids', 'cells', 'materials'):
        require(not supplied[name].flags.writeable, 'readonly authenticated seed arrays required')
        _same_arrays(np, supplied[name], measured[name], check)
    require(ext.same_graph(np, supplied['P'], measured['P'])
            and supplied['controls'] == measured['controls'], 'actual seed P graph/control order differs')


def _ledger(np, basis, check):
    """Validate complete axis-aligned HEX8 coverage in bounded element tiles."""
    xyz, cells, mids = (basis[k] for k in ('coordinates_mm', 'cells', 'materials'))
    require(type(xyz) is np.ndarray and xyz.dtype == np.dtype('float64') and xyz.ndim == 2
            and xyz.shape[1] == 3 and 0 < len(xyz) <= MAX_NODES
            and type(cells) is np.ndarray and cells.dtype.kind in 'iu' and cells.ndim == 2
            and cells.shape[1] == 8 and 0 < len(cells) <= MAX_ELEMENTS
            and type(mids) is np.ndarray and mids.shape == (len(cells),) and mids.dtype.kind in 'iu',
            'bounded physical node/HEX8/material arrays required')
    axes = tuple(np.unique(xyz[:, j]) for j in range(3))
    nx, ny, nz = (len(v)-1 for v in axes)
    require(min(nx, ny, nz) > 0 and nx*ny*nz == len(cells), 'complete Cartesian brick census required')
    seen = np.zeros(len(cells), dtype=bool)
    counts, copper, volumes, cu_volumes = ([0]*3 for unused in range(4))
    template = None
    for first in range(0, len(cells), TILE):
        check()
        ids = cells[first:first+TILE]; material = mids[first:first+TILE]
        require(bool(np.all((ids >= 0)&(ids < len(xyz)))), 'HEX node index out of range')
        corners = xyz[ids]; low, high = corners.min(axis=1), corners.max(axis=1)
        widths = high-low
        require(bool(np.isfinite(corners).all()) and bool(np.all(widths > 0.)), 'finite positive brick widths required')
        bits = corners == high[:, None, :]
        require(bool(np.all(bits | (corners == low[:, None, :])))
                and bool(np.all(np.sort(bits.astype('int8')@np.array([1, 2, 4]), axis=1) == np.arange(8))),
                'each HEX must have all eight distinct Cartesian corners')
        if template is None:
            template = bits[0].copy()
            a = template.astype('int8')
            # The supplied FE convention may order axes x,z,y rather than x,y,z.
            require(abs(int(np.cross(a[1]-a[0], a[3]-a[0])@(a[4]-a[0]))) == 1,
                    'independent original HEX corner axes required; either handedness is allowed')
        require(bool(np.all(bits == template)), 'original ordered HEX corner template differs')
        ijk = [np.searchsorted(axis, low[:, j]) for j, axis in enumerate(axes)]
        for j, axis in enumerate(axes):
            require(bool(np.all(ijk[j] < len(axis)-1))
                    and bool(np.all(axis[ijk[j]] == low[:, j]))
                    and bool(np.all(axis[ijk[j]+1] == high[:, j])), 'HEX spans missing/intermediate grid levels')
        anchor = (ijk[1]*nx+ijk[0])*nz+ijk[2]
        require(len(np.unique(anchor)) == len(anchor) and not bool(np.any(seen[anchor])), 'duplicate Cartesian HEX')
        seen[anchor] = True
        for hinge in (.0015, .0115):
            require(not bool(np.any((low[:, 1] < hinge-1e-14)&(high[:, 1] > hinge+1e-14))),
                    'HEX crosses a fixed physical layer interface')
        role = np.where(high[:, 1] <= .0015+1e-14, 0, np.where(low[:, 1] >= .0115-1e-14, 2, 1))
        volume = widths.prod(axis=1)
        for k in range(3):
            pick = role == k; cu = pick & (material == 4)
            require(bool(np.all((material[pick] == k+1)|(material[pick] == 4))), 'wrong regional background material')
            counts[k] += int(np.count_nonzero(pick)); copper[k] += 8*int(np.count_nonzero(cu))
            volumes[k] += float(volume[pick].sum()); cu_volumes[k] += float(volume[cu].sum())
        check()
    require(bool(np.all(seen)) and all(v > 0 for v in counts), 'complete nonempty three-region coverage required')
    return {name:dict(elements=counts[k], Cu_GP=copper[k], volume_mm3=volumes[k],
                     Cu_volume_mm3=cu_volumes[k]) for k, name in enumerate(('bi', 'beol', 'si'))}


def measure_seed_contract(api, bound, *, seed, seed_report, check):
    """Measure the seed-loader result; Main supplies its actual emitted binding.

    Do not copy geometry-specific census fields from legacy proof_for reports.
    This records measured geometry, not seed-file authentication or accuracy.
    """
    require(callable(check) and getattr(bound, 'token', None) is _TOKEN,
            'bound sources and explicit guard required')
    check()
    info, deck, basis = seed.profile, seed.deck, seed.basis
    req = request(info['pitch_um'], 50., 'inner')
    require(info['source_si_um'] == 37. and info['cell_um'] == req.cell_um
            and info['diameter_um'] == req.diameter_um, 'seed profile differs from fixed microgeometry')
    regions = _ledger(api.np, basis, check)
    result = dict(case=seed.proof.get('case', info['mesh_recipe']), mesh_recipe=info['mesh_recipe'],
        pitch_um=req.pitch_um, diameter_pitch_ratio=.5, source_si_um=37.,
        cell_xz_um=[req.cell_um, req.cell_um], physical_nodes=len(basis['coordinates_mm']),
        hex_elements=len(basis['cells']), deck_nodes=len(deck.nodes), equations=len(deck.equations),
        constraints=len(deck.constraints), body_coordinates=basis['P'].shape[1],
        Cu_GP=sum(value['Cu_GP'] for value in regions.values()),
        region_Cu_GP={name:value['Cu_GP'] for name, value in regions.items()},
        seed_report=deepcopy(seed_report))
    _contract(result, req)
    check()
    return result


def prepare_target(api, bound, *, seed_deck, seed_basis, request, seed_contract, check):
    require(callable(check) and getattr(bound, 'token', None) is _TOKEN, 'bound sources and explicit guard required')
    check()
    require(type(request) is GeometryRequest
            and request == globals()['request'](request.pitch_um, request.total_um, request.variant),
            'unaltered explicit geometry request required')
    _contract(seed_contract, request)
    np, ext, geometry = api.np, bound.extension, bound.geometry
    for field, count in (('nodes', 'deck_nodes'), ('elements', 'hex_elements'),
                         ('equations', 'equations'), ('constraints', 'constraints')):
        require(type(getattr(seed_deck, field)) is dict
                and len(getattr(seed_deck, field)) == seed_contract[count], 'seed deck count differs: '+field)
    require(all(0 < len(terms) <= 8 for terms in seed_deck.equations.values()), 'bounded CE term counts required')
    require(seed_basis['coordinates_mm'].shape == (seed_contract['physical_nodes'], 3)
            and seed_basis['cells'].shape == (seed_contract['hex_elements'], 8)
            and seed_basis['P'].shape == (3*seed_contract['physical_nodes'], seed_contract['body_coordinates']),
            'seed basis/deck dimensions differ from contract')
    source_regions = _ledger(np, seed_basis, check)
    require({k:v['Cu_GP'] for k, v in source_regions.items()} == seed_contract['region_Cu_GP'],
            'measured original regional Cu GP census differs')
    xyz = seed_basis['coordinates_mm']
    require(all(abs(float(np.ptp(xyz[:, j]))*1000-request.cell_um) < 1e-10 for j in (0, 2))
            and abs(float(xyz[:, 1].max())-.0485) < 1e-14, 'actual Si37 source extent/cell differs')
    check(); measured = geometry.six_macro_basis(api, seed_deck); check()
    _same_basis(np, ext, seed_basis, measured, check)
    del measured
    target = ext.retarget(api, seed_deck, request.si_um, source_si_um=37.)
    check(); basis = geometry.six_macro_basis(api, target); check()
    require(ext.same_graph(np, seed_basis['P'], basis['P'])
            and seed_basis['controls'] == basis['controls'], 'target changed P graph/reference gauge')
    for name in ('node_ids', 'element_ids', 'cells', 'materials'):
        _same_arrays(np, seed_basis[name], basis[name], check)
    require(all(getattr(target, field) == getattr(seed_deck, field) for field in
        ('elements', 'elastic', 'element_types', 'element_options', 'first_solve_line')), 'native cards/options/connectivity changed')
    for first in range(0, len(xyz), TILE):
        check(); old = xyz[first:first+TILE]; new = basis['coordinates_mm'][first:first+TILE]
        _same_arrays(np, old[:, [0, 2]], new[:, [0, 2]], check)
        fixed = old[:, 1] <= ext.INTERFACE_MM
        _same_arrays(np, old[fixed], new[fixed], check)
        expected = old[:, 1].copy(); above = ~fixed
        if request.si_um != 37.:
            expected[above] = ext.INTERFACE_MM+(expected[above]-ext.INTERFACE_MM)*(request.si_um/37.)
        _same_arrays(np, expected, new[:, 1], check)
    _same_arrays(np, ext.analytic_fields(np, basis['coordinates_mm']).reshape(-1, 6), basis['macro_fields'], check)
    regions = _ledger(np, basis, check)
    require({k:v['Cu_GP'] for k, v in regions.items()} == seed_contract['region_Cu_GP'], 'retarget changed Cu GP identities/counts')
    proof = dict(state='HEIGHT_RETARGET_GEOMETRY_CANDIDATE_ONLY_NOT_ADMITTED', request=request._asdict(),
        seed_contract=deepcopy(seed_contract), source_pins=dict(bound.source_pins),
        source_si_um=37., physical_nodes=len(xyz), body_coordinates=basis['P'].shape[1],
        source_regions=source_regions, target_regions=regions, Cu_GP=sum(v['Cu_GP'] for v in regions.values()),
        physical_volume_units='mm3', Cu_GP_count_recomputed=True, native_geometry_certificate=basis['proof'],
        unchanged_extension_definitions=['retarget', 'analytic_fields', 'same_graph'],
        fixed_BI_BEOL_and_XZ_byte_identity=True, Si_node_Y_exact_affine_mapping=True,
        P_exact_seed_graph=True, ordered_connectivity_materials_unchanged=True,
        cards_options_and_reference_gauge_retained=True, seed_inputs_modified=False,
        pitch_seed_constructed_here=False, seed_file_authentication_performed_here=False,
        seed_provenance_authentication_by_Main_required=True, element_tile_size=TILE,
        FE_assembly_performed=False, FE_factorizations=0, new_FE_quadrature_certified=False,
        Cu_updates=0, Cu_state_transfer=False,
        history_executed=False, stiffness_or_GP_strain_maps_constructed=False,
        interpolation_accuracy_validated=False, geometry_family_validated=False,
        operator_admission=False, assembly_accuracy_admission=False, resource_admission=False)
    check()
    return GeometryTarget(target, basis, proof)
