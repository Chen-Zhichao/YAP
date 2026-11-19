"""Unit-invariant native constructor and material binding; no asset loaders."""
import ast
import hashlib
import math
from pathlib import Path
from types import CodeType, FunctionType, MappingProxyType

import numpy as np
from .runtime import application_module

runtime = application_module("mechanics_runtime")
NATIVE_LAYER_SHA256 = "00d234646aae22730814ae6c5524a5a74b5ae374550032434e919d3e4b04d4c6"
SYMMETRY_LIMIT = 1e-10


def require(ok, message):
    if not ok:
        raise ValueError(message)

def stiffness_symmetry(stiffness):
    """Diagnostic-only SPD energy metric; never alter the physical stiffness."""
    matrix = np.asarray(stiffness, dtype=float)
    require(matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and len(matrix) > 0
        and np.isfinite(matrix).all(), 'finite square stiffness required')
    diagonal = np.diag(matrix)
    require(np.all(diagonal > 0), 'positive stiffness diagonal required for SPD energy metric')
    scale = 1./np.sqrt(diagonal)
    normalized = scale[:,None]*matrix*scale[None,:]
    skew = matrix-matrix.T
    normalized_skew = scale[:,None]*skew*scale[None,:]
    require(np.isfinite(normalized).all() and np.isfinite(normalized_skew).all(), 'finite energy-scaled stiffness required')
    # H is diagnostic only. Reference actions continue to use the original K.
    symmetric_part = .5*normalized+.5*normalized.T
    try:
        lower = np.linalg.cholesky(symmetric_part)
    except np.linalg.LinAlgError as error:
        raise ValueError('reference symmetric part must be SPD') from error
    left = np.linalg.solve(lower, normalized_skew)
    whitened = np.linalg.solve(lower, left.T).T
    norm = float(np.linalg.norm(whitened, ord=2))
    require(math.isfinite(norm), 'finite unhalved SPD-energy skew required')
    return dict(absolute_skew_maximum=float(np.max(np.abs(skew))),
        diagonally_scaled_skew_maximum=float(np.max(np.abs(normalized_skew))),
        SPD_energy_unhalved_skew_spectral_norm=norm, symmetric_part_SPD=True,
        original_dimensionful_allclose=bool(np.allclose(matrix,matrix.T,rtol=1e-10,atol=1e-12)),
        limit=SYMMETRY_LIMIT, passed=norm <= SYMMETRY_LIMIT, operational_matrix_changed=False)

def _coupon_spd_energy_symmetric(stiffness):
    return stiffness_symmetry(stiffness)['passed']

def _guard_constructor_tree(raw):
    require(hashlib.sha256(raw).hexdigest() == NATIVE_LAYER_SHA256, 'exact native layer source SHA required')
    tree = ast.parse(raw)
    original = ast.dump(tree)
    classes = [node for node in tree.body if isinstance(node,ast.ClassDef) and node.name == 'AllGaussReducedFE2Layer']
    require(len(classes) == 1, 'one original native layer class required')
    initializers = [node for node in classes[0].body if isinstance(node,ast.FunctionDef) and node.name == '__init__']
    require(len(initializers) == 1, 'one original native initializer required')
    predicate = ast.parse('not np.allclose(stiffness, stiffness.T, rtol=1.0e-10, atol=1.0e-12)',mode='eval').body
    matches = [node for node in ast.walk(initializers[0]) if isinstance(node,ast.If)
               and ast.dump(node.test) == ast.dump(predicate)]
    require(len(matches) == 1, 'exact single constructor symmetry predicate required')
    target = matches[0]
    old = target.test
    target.test = ast.copy_location(ast.parse('not _coupon_spd_energy_symmetric(stiffness)',mode='eval').body,old)
    changed = target.test
    target.test = old
    require(ast.dump(tree) == original, 'whole native-source AST inverse differs')
    target.test = changed
    return ast.fix_missing_locations(tree), dict(predicate_replacements=1,whole_source_AST_inverse_verified=True,
        constructor_only=True,guard_only=True,limit=SYMMETRY_LIMIT,unhalved_skew=True,
        metric='norm2(L^-1 D (K-K.T) D L^-T); L L.T = D ((K+K.T)/2) D',
        operational_matrix_symmetrized=False,force_or_tangent_methods_changed=False)

def unit_invariant_layer_type(native_module):
    """Private initializer repair; inherit every operational native method."""
    path = _path(native_module.__file__)
    with runtime._component_open(path,256*1024) as stream:
        before = runtime._component_identity(stream)
        raw = stream.read(256*1024+1)
        require(before == runtime._component_identity(stream), 'native constructor source identity changed')
    tree, proof = _guard_constructor_tree(raw)
    base = native_module.AllGaussReducedFE2Layer
    require(base.__init__.__globals__ is vars(native_module) and native_module.np is np,
            'actual native initializer/module required')
    def initializer(code):
        classes = [value for value in code.co_consts if isinstance(value,CodeType) and value.co_name == base.__name__]
        require(len(classes) == 1, 'one compiled native class required')
        methods = [value for value in classes[0].co_consts if isinstance(value,CodeType) and value.co_name == '__init__']
        require(len(methods) == 1, 'one compiled native initializer required')
        return methods[0]
    original = initializer(compile(raw,str(path),'exec',dont_inherit=True))
    require(base.__init__.__code__ == original and base.__init__.__code__.co_filename == str(path),
            'loaded initializer differs from exact native source')
    modified = initializer(compile(tree,'<native_coupon SPD-energy initializer guard>','exec',dont_inherit=True))
    scope = dict(base.__init__.__globals__,_coupon_spd_energy_symmetric=_coupon_spd_energy_symmetric)
    init = FunctionType(modified,scope,'__init__',base.__init__.__defaults__,base.__init__.__closure__)
    init.__kwdefaults__ = dict(base.__init__.__kwdefaults__ or {})
    init.__annotations__ = dict(base.__init__.__annotations__)
    proof.update(native_source_path=str(path),native_source_sha256=NATIVE_LAYER_SHA256,
        repair_module=__name__,repair_source_path=str(Path(__file__).absolute()),
        initializer_code_authenticated=True,all_other_methods_inherited=True)
    return type('UnitInvariantCouponLayer',(base,),dict(__init__=init,__module__=__name__,
        native_coupon_constructor_guard=MappingProxyType(proof)))

def _path(value):
    path = Path(value).expanduser()
    require(path.is_absolute() and len(path.parts) <= 48 and not any(
        part in ('..', '#snapshot', '.snapshot', 'Ansys_Verify', 'mapdl_run_roots', 'mapdl_archives')
        or part.startswith(('.migration_hold_', 'storage_migration')) for part in path.parts),
        'absolute non-archive/non-snapshot input path required')
    return path

def _material(binding):
    try:
        from YAP.W2W.assembly_mechanics import _material_runtime
    except ModuleNotFoundError as error:
        if error.name != 'YAP':
            raise
        from W2W.assembly_mechanics import _material_runtime
    bound_runtime = _material_runtime(binding)
    bound_runtime._validate_corrected_material_binding(binding, material_kernels=binding.material_kernels, cu_card=binding.cu_card)
    require(isinstance(binding,bound_runtime.CorrectedCompiledMaterialBinding), 'explicit Cu35/483 compiled source binding required')
    evidence = bound_runtime.corrected_material_provenance(binding)
    sources = {Path(row['path']).name:row['sha256'] for row in evidence['sources']}
    require(sources.get('exp12_cu_material_point.py') == '35e7016dab8b1fb2cb0a928e7642665d3fc4f00b4f255eb797fb6d6e44b56722'
        and sources.get('exp12_all_gauss_exact_acceleration.py') == '483c9d948931d48beb73561842cb8fe780db285d3b61f158e580229fbdcd465f',
        'same qualified Cu/acceleration source pair required')
    card = binding.cu_card
    expected = dict(room_c=25.,anneal_c=350.,e_room_mpa=131000.,e_anneal_mpa=131000.,
        nu_room=.34,nu_anneal=.34,alpha_room_per_c=17.6e-6,alpha_anneal_per_c=17.6e-6,
        yield_room_mpa=321.,yield_anneal_mpa=321.,tangent_mpa=2000.,creep_law='modified_time_hardening',
        creep_prefactor_mpa_minus_n_per_s=1.43e10,creep_activation_term_k=23695.,
        creep_exponent=2.5,creep_time_exponent=-.9)
    require(all(getattr(card,key) == value for key,value in expected.items()), 'unchanged frozen b3 material/clock parameters required')
    return evidence, bound_runtime

