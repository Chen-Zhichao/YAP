"""TRACE80 layout reuse of the authenticated native64 Si-cap algorithm.

No legacy module is imported or modified. Production calls use the guarded
segment path; raw private FE helpers remain available for tiny software tests.
This factory establishes neither full-size memory nor process qualification.
"""
import ast
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import resource
import stat
import threading
from types import ModuleType


LEGACY_SHA256 = '1a1c11ffcaeddbe9031727d4e749e3bba97ba64bfc2f3aa63a4eff8e4d91d851'
LEGACY = Path(__file__).with_name('component_silicon_cap.py')
VERSION = 'full_periodic_interior_inherited_face_jumps_h8_bbar_si_trace80_layout_v1'
BOUNDARY_CONTRACT = dict(version='inherited_face_jumps_affine_in_normalized_y_trace80_v1',
    policy='source_registration_per_column_absolute_plus_relative_no_trace_modification',
    absolute_limit=1e-12, relative_limit=2e-8, port_rank_per_face=80, macro_coordinates=166,
    source_profile='bi4_d2_si37_cell12_transition96',
    source_reference_sha256='3ca3a3fda2023d0aabab784cedd99b6d774b997fe6daf7c124e3a09cc06f45c5',
    source_periodic_P_equivalence=False,
    source_equivalence_scope='NOT_ESTABLISHED_BY_LAYOUT_ADAPTER',
    interior='complete_periodic_zero_face_fluctuations',
    total_field='canonical_q6_plus_unmodified_face_port_lift_plus_relaxed_periodic_fluctuation')

# Counts are per original function, not unrestricted integer replacement.
CHANGES = {
    '_face_jump_audit': {64:(80, 1)},
    '_validate': {64:(80, 6), 134:(166, 4), 70:(86, 5), 46:(97, 1),
        'Si caps require native64':('Si caps require TRACE80', 1),
        'finite positive native134 scales required':('finite positive native166 scales required', 1),
        'structured square physical face, at most 46 by 46, required':
            ('structured square physical face, at most 97 by 97, required', 1)},
    '_assemble': {134:(166, 2), 70:(86, 2)},
    '_condense_fe': {134:(166, 6), 70:(86, 5), 73:(89, 1), 135:(167, 1)},
    '_read_cache': {134:(166, 6), 135:(167, 1)},
    'augment_w2w_library': {134:(166, 1), 70:(86, 2),
        'EXPERIMENTAL_HOMOGENEOUS_SI_CAP_RESTRICTED_NATIVE64_FACES_NOT_PROCESS_VALIDATION':
            ('EXPERIMENTAL_HOMOGENEOUS_SI_CAP_RESTRICTED_TRACE80_FACES_NOT_PROCESS_VALIDATION', 1)}}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _source(path, expected=None):
    p = Path(path)
    def permitted(candidate):
        return (not {'..', '#snapshot', '.snapshot'}.intersection(candidate.parts)
            and not any(v.startswith(('storage_migration', '.migration_hold_')) for v in candidate.parts))
    _require(p.is_absolute() and permitted(p), 'protected code path refused before resolution')
    p = p.resolve(strict=True)
    _require(permitted(p), 'protected resolved code path refused')
    key = lambda s:(s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    with os.fdopen(os.open(p, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK), 'rb', buffering=0) as f:
        before = os.fstat(f.fileno())
        _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= 1024**2, 'bounded regular cap source required')
        raw = f.read(before.st_size+1)
        _require(len(raw) == before.st_size and key(before) == key(os.fstat(f.fileno())) == key(p.lstat()),
            'cap source identity changed during read')
    pin = hashlib.sha256(raw).hexdigest()
    _require(expected is None or pin == expected, 'unknown legacy cap source refused')
    return raw, pin


def _layout_tree(raw):
    _require(type(raw) is bytes and hashlib.sha256(raw).hexdigest() == LEGACY_SHA256,
        'exact authenticated legacy cap bytes required')
    tree = ast.parse(raw); original = ast.dump(tree, include_attributes=False)
    functions = {n.name:n for n in tree.body if isinstance(n, ast.FunctionDef)}
    changed, counts = [], {}
    for name, replacements in CHANGES.items():
        _require(name in functions, 'missing exact layout function: '+name)
        found = dict.fromkeys(replacements, 0)
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Constant) and type(node.value) in (int, str) and node.value in replacements:
                old = node.value; changed.append((node, old)); found[old] += 1
                node.value = replacements[old][0]
        _require(all(found[k] == replacements[k][1] for k in found), 'layout anchor count differs: '+name)
        counts[name] = {str(k):v for k,v in found.items()}
    construction = [n.value for n in ast.walk(functions['augment_w2w_library'])
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'construction' for t in n.targets)]
    _require(len(construction) == 1 and isinstance(construction[0], ast.Call)
        and isinstance(construction[0].func, ast.Name) and construction[0].func.id == 'dict'
        and not any(k.arg == 'layout_adaptation' for k in construction[0].keywords),
        'one original pre-hash construction dictionary required')
    inserted = ast.keyword(arg='layout_adaptation', value=ast.Name(id='_yap_layout_adaptation', ctx=ast.Load()))
    construction[0].keywords.append(inserted)
    answer = deepcopy(tree)
    construction[0].keywords.remove(inserted)
    for node, old in changed:
        node.value = old
    _require(ast.dump(tree, include_attributes=False) == original, 'inverse complete legacy AST differs')
    return ast.fix_missing_locations(answer), counts


def _forecast(nx, nz, intervals):
    _require(all(type(v) is int for v in (nx, nz, intervals)) and 2 <= nx <= 97
        and 2 <= nz <= 97 and 2 <= intervals <= 6, 'bounded cap grid/interval census required')
    nodes, elements = nx*nz*(intervals+1), (nx-1)*(nz-1)*intervals
    dofs, interior = 3*nodes, 3*(intervals-1)*(nx-1)*(nz-1)
    # Count dense/COO/CSR construction work conservatively; no bound is claimed
    # for sparse factor fill, allocator overhead or the already-loaded library.
    dense = elements*(6*24*24*8+3*24*8+8*8)+dofs*(6*166+8)*8+interior*(4*166+8)*8
    return dict(nodes=nodes, elements=elements, macro_coordinates=166,
        nodal_vector_dofs=dofs, periodic_interior_dofs=interior,
        dense_work_forecast_bytes=dense, factor_memory_qualified=False,
        total_peak_memory_bound_claimed=False, existing_library_memory_included=False)


_OPERATIONS = '''
def _trace80_validate(library, geometry, physical_trace):
    _require(type(library.port_rank_per_face) is int and library.port_rank_per_face == 80,
             'exact production TRACE80 rank required')
    xyz = physical_trace.get('face_coordinates_mm')
    lifts = physical_trace.get('face_lifts')
    _require(isinstance(xyz, np.ndarray) and xyz.shape == (2, 9409, 3)
             and isinstance(lifts, np.ndarray) and lifts.ndim == 3
             and lifts.shape[:2] == (2, 28227) and 166 <= lifts.shape[2] <= 214,
             'exact production97x97 TRACE80 faces required before copying')
    _require(np.allclose(xyz[:, :, [0, 2]].min(axis=(0, 1)), 0., rtol=0, atol=1e-12)
             and np.allclose(xyz[:, :, [0, 2]].max(axis=(0, 1)), .012, rtol=0, atol=1e-12)
             and math.isclose(library.cell_area_mm2, .012**2, rel_tol=0, abs_tol=1e-15),
             'unchanged production0..0.012mm cell frame required')
    return _layout_original_validate(library, geometry, physical_trace)

def _trace80_construction():
    from . import assembly_operator_library as adapter, component_assembly_bridge as bridge
    return dict(version=VERSION, source_sha256=_file_sha(__file__),
                native_sha256=_native_sources()[2], numpy_version=np.__version__,
                bridge_sha256=_file_sha(bridge.__file__),
                assembly_helpers_sha256=_file_sha(adapter.__file__),
                layout_adaptation=_yap_layout_adaptation)

def _trace80_specification(specification):
    _require(isinstance(specification, dict)
             and specification.get('boundary_contract') == BOUNDARY_CONTRACT
             and specification.get('material') == MATERIAL,
             'TRACE80 cache requires its explicit boundary/material profile')
    for field in ('construction_sha256', 'coordinate_contract_sha256', 'trace_sha256'):
        pin = specification.get(field)
        _require(type(pin) is str and len(pin) == 64 and all(c in '0123456789abcdef' for c in pin),
                 'TRACE80 cache lacks source/trace/coordinate binding: '+field)
    current = hashlib.sha256(_json(_trace80_construction()).encode()).hexdigest()
    _require(specification['construction_sha256'] == current,
             'TRACE80 cache construction differs from current engine')

def _trace80_read_cache(npz_path, json_path, key):
    present = (os.path.lexists(npz_path), os.path.lexists(json_path))
    _require(present[0] == present[1], 'incomplete TRACE80 cache pair requires manual review')
    if not present[0]:
        return None
    with os.fdopen(os.open(json_path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= 65536, 'bounded TRACE80 cache metadata required')
        value = json.loads(stream.read(before.st_size+1))
    specification = value.get('specification', {})
    _trace80_specification(specification)
    ready = _layout_original_read_cache(npz_path, json_path, key)
    _require(ready is not None and ready[2].get('specification') == specification,
             'TRACE80 cache specification changed across authenticated reads')
    return ready

def _trace80_permission(stage, forecast):
    _require(callable(_layout_build_guard), 'TRACE80 cache miss/build requires explicit supervised build_guard')
    request = dict(forecast, stage=stage, port_rank_per_face=80,
                   caller_must_verify_fresh_fullcap_admission=True)
    request['actual_AS_pair'] = list(_layout_resource.getrlimit(_layout_resource.RLIMIT_AS))
    _require(_layout_build_guard(request) is True, 'TRACE80 build_guard did not explicitly permit build')
    cap = _layout_resource.getrlimit(_layout_resource.RLIMIT_AS)
    _require(cap[0] == cap[1] and cap[0] > forecast['dense_work_forecast_bytes']
             and cap[0] != _layout_resource.RLIM_INFINITY,
             'finite equal AS hard/soft cap above dense forecast required; factor memory remains unqualified')
    _last_build_forecast.clear(); _last_build_forecast.update(request)

def _trace80_segment(root, binding, xs, zs, ports, scale, thickness_um, library, sources, intervals=6):
    global _assemble, _condense_fe
    specification = dict(binding, thickness_um=float(thickness_um), thickness_intervals=intervals)
    _trace80_specification(specification)
    key = hashlib.sha256(_json(specification).encode()).hexdigest()
    with _layout_execution_lock:
        # Consult the current namespace, including the public storage hooks.
        ready = _read_cache(_published_root()/(key+'.npz'), _published_root()/(key+'.json'), key)
        if ready is None:
            ready = _read_cache(root/(key+'.npz'), root/(key+'.json'), key)
        if ready is not None:
            return _layout_original_segment(root, binding, xs, zs, ports, scale, thickness_um, library, sources, intervals)
        forecast = _layout_forecast(len(xs), len(zs), intervals)
        _trace80_permission('before_cache_miss_mkdir', forecast)
        assemble, condense = _assemble, _condense_fe
        def guarded_assemble(*args, **kwargs):
            _trace80_permission('before_FE_allocation', forecast)
            return assemble(*args, **kwargs)
        def guarded_condense(*args, **kwargs):
            _trace80_permission('before_condensation_and_factorization', forecast)
            return condense(*args, **kwargs)
        _assemble, _condense_fe = guarded_assemble, guarded_condense
        try:
            return _layout_original_segment(root, binding, xs, zs, ports, scale, thickness_um, library, sources, intervals)
        finally:
            _assemble, _condense_fe = assemble, condense
'''


def create_trace80_cap_engine(*, build_guard=None):
    """Create isolated executing globals; no FE, cache access or qualification.

    A build guard receives a stage/forecast dictionary and must return exactly
    True after checking fresh supervised full-hard-cap admission and deadlines.
    Direct private numerical helpers are unguarded tiny-test interfaces, not
    production entry points. Use augment_w2w_library/_segment for real builds.
    """
    _require(build_guard is None or callable(build_guard), 'build_guard must be callable or None')
    raw, _ = _source(LEGACY, LEGACY_SHA256)
    _, adapter_sha = _source(Path(__file__))
    tree, counts = _layout_tree(raw)
    operations = ast.parse(_OPERATIONS)
    generated = ast.fix_missing_locations(ast.Module(body=tree.body+operations.body, type_ignores=[]))
    proof = dict(schema='yap.silicon_cap.trace80_layout_adapter.v1',
        legacy_source_sha256=LEGACY_SHA256, adapter_source_sha256=adapter_sha,
        generated_ast_sha256=hashlib.sha256(ast.dump(generated, include_attributes=False).encode('ascii')).hexdigest(),
        port_rank_per_face=80, macro_coordinates=166, inverse_ast_exact=True,
        legacy_filename=LEGACY.name, adapter_filename=Path(__file__).name,
        anchor_counts=counts, construction_lineage_insertions=1,
        numerical_equations_changed=False, factor_memory_qualified=False,
        boundary_contract_profile='P4_TRACE80_NOT_LEGACY_P_EQUIVALENCE')
    engine = ModuleType(__package__+'._private_trace80_cap')
    engine.__package__, engine.__file__ = __package__, str(Path(__file__).resolve(strict=True))
    engine.__dict__.update(_yap_layout_adaptation=proof, _layout_build_guard=build_guard,
        _layout_resource=resource, _layout_forecast=_forecast,
        _layout_execution_lock=threading.RLock(), _last_build_forecast={})
    exec(compile(generated, LEGACY.name+'::TRACE80-layout', 'exec', dont_inherit=True), engine.__dict__)
    engine.VERSION, engine.BOUNDARY_CONTRACT = VERSION, deepcopy(BOUNDARY_CONTRACT)
    engine.__doc__ = __doc__
    for name in ('_validate', '_read_cache', '_segment'):
        engine.__dict__['_layout_original'+name] = engine.__dict__[name]
        engine.__dict__[name] = engine.__dict__['_trace80'+name]
    # _publish_cache stays the generated original FunctionType: the unified
    # storage hook must be able to replace os in that exact executing function.
    _require(engine._publish_cache.__globals__ is engine.__dict__, 'generated writer globals differ')
    return engine
