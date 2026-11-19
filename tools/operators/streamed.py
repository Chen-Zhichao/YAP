"""Owned streamed coordinate tangent; unchanged original Cu evolution methods."""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
from types import CodeType, FunctionType, MappingProxyType, MethodType
import weakref

from . import acceleration as generic
from . import runtime as shared
from ..coupon_tangent_contraction import contract_coordinate_tangent

MARKER = '_streamed_coupon_coordinate_owner'


def _coordinate_function(acceleration, block_size):
    raw = Path(acceleration.__file__).read_bytes()
    shared.require(hashlib.sha256(raw).hexdigest() == shared.SOURCE_PINS['acceleration_source'],
                   'exact original constitutive acceleration source required')
    tree = ast.parse(raw)
    # Preserve module-import compiler context; match the deployed eager annotations.
    tree.body = [node for node in tree.body if not
        (isinstance(node, ast.ImportFrom) and node.module == '__future__')]
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name == '_accelerated_assembled_coordinate_tangent']
    shared.require(len(functions) == 1, 'one original coordinate-tangent function required')
    function = functions[0]
    original = acceleration._accelerated_assembled_coordinate_tangent
    namespace = dict(original.__globals__)
    def extract(module):
        code = compile(module, str(acceleration.__file__), 'exec', dont_inherit=True)
        methods = [value for value in code.co_consts if isinstance(value, CodeType)
            and value.co_name == function.name]
        shared.require(len(methods) == 1, 'one compiled source method required')
        return methods[0]
    compiled = extract(tree)
    shared.require(compiled.co_code == original.__code__.co_code
        and compiled.co_consts == original.__code__.co_consts
        and compiled.co_names == original.__code__.co_names
        and compiled.co_flags == original.__code__.co_flags,
        'actual private coordinate method must match the pinned source body')
    before = ast.dump(function)
    starts = [i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'correction_times_map' for target in node.targets)]
    ends = [i for i, node in enumerate(function.body) if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name) and node.target.id == 'tangent']
    shared.require(len(starts) == len(ends) == 1 and ends[0]-starts[0]+1 == 5,
                   'exact original five-statement contraction block required')
    first, last = starts[0], ends[0]+1
    saved = function.body[first:last]
    replacement = ast.parse("""
tangent, _streamed_statistics = _streamed_contract(self.strain_maps, self.weights_mm3,
    self.reference_stiffness(temperature_c), tangent_data,
    correction_times_map=_algorithmic_correction_times_map, block_size=_streamed_block_size)
""").body
    function.body[first:last] = replacement
    selected = deepcopy(tree)
    function.body[first:first+len(replacement)] = saved
    shared.require(ast.dump(function) == before, 'whole original coordinate-function AST inverse differs')
    namespace.update(_streamed_contract=contract_coordinate_tangent, _streamed_block_size=block_size)
    code = extract(ast.fix_missing_locations(selected))
    value = FunctionType(code, namespace, original.__name__, original.__defaults__, original.__closure__)
    value.__kwdefaults__ = dict(original.__kwdefaults__ or {})
    value.__doc__ = 'Same cached constitutive tangent, contracted in bounded all-GP blocks.'
    return value, dict(original_source_sha256=hashlib.sha256(raw).hexdigest(),
        whole_original_coordinate_function_AST_inverse_exact=True,
        whole_module_compiler_context_preserved=True, original_method_bytecode_authenticated=True,
        replaced_contraction_statements=5, block_size=block_size,
        constitutive_preparation_and_cache_matching_unchanged=True,
        point_sampling=False, reduction_order_changed=True)


class _CoordinateOwner:
    def __init__(self, operator, delegate, function, proof):
        self.ref = weakref.ref(operator)
        self.delegate, self.function, self.proof = delegate, function, proof
        self.code = function.__code__
        self.helper_code = contract_coordinate_tangent.__code__
        self.namespace = dict(function.__globals__)
        self.namespace_codes = {key:value.__code__ for key,value in self.namespace.items()
            if isinstance(value,FunctionType)}
        self.bound = None

    def validate(self, operator):
        shared.require(self.ref() is operator and vars(operator).get(MARKER) is self
            and vars(operator).get('assembled_coordinate_tangent') is self.bound
            and self.bound.__self__ is operator and self.bound.__func__ is _coordinate
            and self.function.__code__ is self.code
            and self.function.__globals__['_streamed_contract'] is contract_coordinate_tangent
            and self.function.__globals__.keys() == self.namespace.keys()
            and all(self.function.__globals__[key] is value for key,value in self.namespace.items())
            and all(self.namespace[key].__code__ is code for key,code in self.namespace_codes.items())
            and contract_coordinate_tangent.__code__ is self.helper_code,
            'owned streamed coordinate method or helper changed')
        self.delegate.validate(operator)


def _coordinate(operator, *args, **kwargs):
    owner = vars(operator).get(MARKER)
    shared.require(type(owner) is _CoordinateOwner, 'owned streamed coordinate method required')
    owner.validate(operator)
    # The coordinate tangent has the same physical input contract as evaluate.
    owner.delegate.arguments(operator, 'evaluate', args, kwargs)
    try:
        return owner.function(operator, *args, **kwargs)
    finally:
        owner.validate(operator)


def attach_streamed_acceleration(operator, *, material_binding, check=lambda: None, admit,
                                block_size=4096):
    shared.require(type(block_size) is int and 1 <= block_size <= 131072,
                   'bounded exact contraction block size required')
    shared.require(MARKER not in vars(operator), 'streamed owner already attached')
    function, proof = _coordinate_function(material_binding.implementation.acceleration, block_size)
    delegated = generic.attach_acceleration(operator, material_binding=material_binding,
        check=check, admit=admit,coordinate=False)
    delegate = vars(operator)[generic.MARKER]
    owner = _CoordinateOwner(operator, delegate, function, proof)
    owner.bound = MethodType(_coordinate, operator)
    vars(operator)[MARKER] = owner
    vars(operator)['assembled_coordinate_tangent'] = owner.bound
    owner.validate(operator)
    return shared._freeze(dict(schema='yap.streamed_coupon_acceleration.v1',
        per_instance_only=True, Cu_constitutive_functions_unchanged=True,
        unchanged_delegated_methods=list(delegate.methods), original_delegation=delegated,
        coordinate_contraction=proof,
        global_modules_or_classes_mutated=False, full_reference_stiffness_retained=True,
        point_sampling=False, history_accuracy_claim=False, assembly_qualified=False))


class _NativeCoordinateDelegate(generic._Owner):
    """Reuse buffer/material/cache guards without replacing native Cu methods."""

    def __init__(self, operator, binding, runtime, evidence, *, check, admit):
        super().__init__(operator,binding,runtime,evidence,check=check,admit=admit,coordinate=False)
        self.codes = {name:function.__code__ for name,function in self.originals.items()}
        self.wrappers, self.wrapper_codes = {}, {}
        self.evidence = shared._freeze(dict(schema='yap.coordinate_only_instance_guard.v1',
            material=evidence, unchanged_native_methods=list(self.methods),
            coordinate_tangent_only=True, constitutive_methods_replaced=False))

    def validate(self, operator):
        result = super().validate(operator)
        shared.require(not any(name in vars(operator) for name in self.methods),
                       'coordinate-only profile must retain native constitutive methods')
        return result


def attach_coordinate_acceleration(operator, *, material_binding, check=lambda: None, admit,
                                   block_size=4096):
    """Own only the streamed tangent; keep all native evolution/JVP methods."""
    shared.require(callable(check) and callable(admit), 'explicit guard and allocation admission required')
    shared.require(type(block_size) is int and 1 <= block_size <= 131072,
                   'bounded exact contraction block size required')
    shared.require(MARKER not in vars(operator) and generic.MARKER not in vars(operator),
                   'coordinate owner already attached')
    parameters = {name:getattr(material_binding.cu_card,name) for name in shared.HISTORICAL_CARD}
    evidence, bound_runtime = generic.layers._material(material_binding,parameters)
    construction = getattr(operator,'normalized_operator_binding',None)
    shared.require(isinstance(construction,MappingProxyType)
        and construction.get('schema') == 'yap.normalized_operator_binding.v1'
        and construction.get('coordinate_size') == operator.coordinate_size
        and construction.get('integration_points') == operator.cu_point_count
        and getattr(operator,'corrected_material_binding',None) is material_binding,
        'authenticated normalized operator binding required')
    delegate = _NativeCoordinateDelegate(operator,material_binding,bound_runtime,evidence,check=check,admit=admit)
    function, proof = _coordinate_function(material_binding.implementation.acceleration,block_size)
    owner = _CoordinateOwner(operator,delegate,function,proof)
    owner.bound = MethodType(_coordinate,operator)
    vars(operator)[generic.MARKER] = delegate
    vars(operator)[MARKER] = owner
    vars(operator)['assembled_coordinate_tangent'] = owner.bound
    try:
        owner.validate(operator)
    except BaseException:
        for name,value in ((generic.MARKER,delegate),(MARKER,owner),('assembled_coordinate_tangent',owner.bound)):
            if vars(operator).get(name) is value:
                del vars(operator)[name]
        raise
    return shared._freeze(dict(schema='yap.owned_coordinate_tangent.v1',
        coordinate_contraction=proof, coordinate_tangent_only=True,
        constitutive_methods_replaced=False, global_modules_or_classes_mutated=False))
