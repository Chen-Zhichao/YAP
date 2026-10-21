"""Private, explicitly pinned Cu35/483 compilation without native installation.

Only deployment AST changes are made: legacy material imports are removed,
annotations are evaluated eagerly, and dynamic Numba disk caching is disabled.
Numerical function bodies retain their original source locations and code.
"""

import ast
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import time
from types import FunctionType, MappingProxyType, ModuleType, SimpleNamespace

try:
    from .. import mechanics_runtime as runtime
except ImportError:
    import mechanics_runtime as runtime


SOURCE_ROOT = runtime.COMPONENT_PROJECT_ROOT / "operator_repair_20261015/source"
SOURCE_PINS = MappingProxyType({
    "exp12_cu_material_point.py": "35e7016dab8b1fb2cb0a928e7642665d3fc4f00b4f255eb797fb6d6e44b56722",
    "exp12_all_gauss_exact_acceleration.py": "483c9d948931d48beb73561842cb8fe780db285d3b61f158e580229fbdcd465f",
})
_NATIVE_METHODS = ("evaluate", "evaluate_many", "tangent_action",
                   "assembled_bubble_tangent", "assembled_coordinate_tangent")
_DISPATCHERS = ("_numba_positive_root", "_numba_constitutive_kernel")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _source(path, pin, name):
    path = runtime._component_path(path)
    _require(path == SOURCE_ROOT / name and pin == SOURCE_PINS[name],
             "exact reviewed Cu35/483 source path and SHA256 required")
    with runtime._component_open(path, 256 * 1024) as stream:
        before = runtime._component_identity(stream)
        raw = stream.read(256 * 1024 + 1)
        _require(before == runtime._component_identity(stream)
                 and hashlib.sha256(raw).hexdigest() == pin, "compiled source changed or hash mismatch")
    return path, raw


def _cache_directory(path):
    path = runtime._component_path(path)
    _require(os.environ.get("NUMBA_CACHE_DIR") == str(path),
             "explicit matching canonical NUMBA_CACHE_DIR required")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    finally:
        os.close(descriptor)
    return path


def _execute(path, raw, injected=None):
    tree = ast.parse(raw, filename=str(path))
    nodes = []
    removed = []
    caches = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            removed.append("future_annotations")
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "exp12_cu_material_point":
            removed.append(node.module)
            continue
        if isinstance(node, ast.Import) and any(a.name == "exp12_all_gauss_reduced_fe2" for a in node.names):
            _require(len(node.names) == 1, "unexpected mixed native import")
            removed.append("exp12_all_gauss_reduced_fe2")
            continue
        nodes.append(node)
    tree.body = nodes
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == "njit":
                    for keyword in decorator.keywords:
                        if keyword.arg == "cache":
                            _require(isinstance(keyword.value, ast.Constant) and keyword.value.value is True,
                                     "unexpected dispatcher cache declaration")
                            keyword.value.value = False
                            caches += 1
    _require(caches == (2 if injected is not None else 0), "unexpected compiled dispatcher census")
    module = ModuleType("_yap_private_" + path.stem)
    module.__file__ = str(path)
    if injected:
        module.__dict__.update(injected)
    # No private or legacy names are inserted into sys.modules.
    exec(compile(tree, str(path), "exec", dont_inherit=True), module.__dict__)
    return module, dict(ast_sha256=hashlib.sha256(ast.dump(tree).encode("utf-8")).hexdigest(),
                        removed_imports=removed, disabled_disk_cache_decorators=caches)


def _native_functions(exact, reduced):
    _require(isinstance(exact, ModuleType) and isinstance(reduced, ModuleType)
             and exact.reduced is reduced and not exact._INSTALLED,
             "actual uninstalled native exact/reduced modules required")
    values = []
    for name in _NATIVE_METHODS:
        current = getattr(reduced.AllGaussReducedFE2Layer, name, None)
        _require(current is getattr(exact, "_ORIGINAL_" + name.upper()),
                 "legacy native global method patch is forbidden")
        values.append(current)
    _require(reduced.trial_update_batch is exact._ORIGINAL_TRIAL_UPDATE_BATCH,
             "legacy native global constitutive patch is forbidden")
    return tuple(values) + (reduced.trial_update_batch,)


def _function_signature(function):
    return (function, function.__code__, function.__defaults__,
            tuple(sorted((key, id(value)) for key, value in (function.__kwdefaults__ or {}).items())))


def _namespace_signature(module):
    names = tuple(sorted((name, id(value)) for name, value in vars(module).items()
                         if name != "__builtins__"))
    functions = []
    classes = []
    for name, value in sorted(vars(module).items()):
        if isinstance(value, FunctionType):
            functions.append((name, _function_signature(value)))
        elif isinstance(value, type) and value.__module__ == module.__name__:
            members = []
            for key, member in sorted(vars(value).items()):
                if isinstance(member, (classmethod, staticmethod)):
                    member = member.__func__
                if isinstance(member, FunctionType):
                    members.append((key, _function_signature(member)))
                elif isinstance(member, property):
                    members.append((key, _function_signature(member.fget)))
            classes.append((name, value, tuple(members)))
    attribute_names = set()
    for value in vars(module).values():
        function = getattr(value, "py_func", value)
        if isinstance(function, FunctionType):
            attribute_names.update(function.__code__.co_names)
    dependencies = []
    for name, value in sorted(vars(module).items()):
        if isinstance(value, ModuleType):
            attributes = []
            for key in sorted(attribute_names):
                if key in vars(value):
                    member = vars(value)[key]
                    signature = _function_signature(member) if isinstance(member, FunctionType) else id(member)
                    attributes.append((key, signature))
            dependencies.append((name, tuple(attributes)))
    return names, tuple(functions), tuple(classes), tuple(dependencies)


def _dispatcher_signature(dispatcher):
    return (dispatcher, _function_signature(dispatcher.py_func),
            json.dumps(dispatcher.targetoptions, sort_keys=True),
            tuple((str(key), id(value)) for key, value in dispatcher.overloads.items()),
            dispatcher._can_compile)


@dataclass(frozen=True)
class _Implementation:
    cu: object = field(repr=False)
    acceleration: object = field(repr=False)
    exact: object = field(repr=False)
    reduced: object = field(repr=False)
    native_functions: tuple = field(repr=False)
    namespaces: tuple = field(repr=False)
    dispatchers: tuple = field(repr=False)
    versions: tuple
    sources: tuple
    transforms: tuple
    cache_directory: str
    warmup_seconds: float

    def validate(self):
        import numba
        import llvmlite

        _require((numba.__version__, llvmlite.__version__) == self.versions,
                 "compiled dependency version changed")
        current = _native_functions(self.exact, self.reduced)
        _require(tuple(_function_signature(value) if isinstance(value, FunctionType) else value
                       for value in current) == self.native_functions,
                 "native original function binding changed")
        _require((_namespace_signature(self.cu), _namespace_signature(self.acceleration)) == self.namespaces,
                 "compiled material function, class or globals binding changed")
        for name, saved in zip(_DISPATCHERS, self.dispatchers):
            dispatcher = getattr(self.acceleration, name)
            _require(_dispatcher_signature(dispatcher) == saved,
                     "compiled dispatcher code, options or compiled overload changed")
            _require(dispatcher._can_compile is False and dispatcher.nopython_signatures,
                     "sealed nopython dispatcher signatures required")
            options = dispatcher.targetoptions
            _require(not options.get("fastmath", False) and options.get("nopython") is True,
                     "strict nopython without fastmath required")
        _require(self.acceleration._numba_constitutive_kernel.targetoptions.get("parallel") is True,
                 "parallel constitutive dispatcher required")
        _require(self.acceleration.njit is numba.njit and self.acceleration.prange is numba.prange,
                 "Numba dependency binding changed")

    def provenance(self):
        import marshal

        return dict(schema="yap.corrected_compiled_material_binding.v1",
            sources=[dict(path=path, sha256=pin) for path, pin in self.sources],
            ast_transformations=[dict(path=path, **json.loads(record)) for path, record in self.transforms],
            dependencies=dict(numba=self.versions[0], llvmlite=self.versions[1]),
            dispatchers={name: dict(py_func_code_sha256=hashlib.sha256(
                marshal.dumps(saved[1][1])).hexdigest(), targetoptions=json.loads(saved[2]),
                signatures=[value[0] for value in saved[3]]) for name, saved in zip(_DISPATCHERS, self.dispatchers)},
            dynamic_disk_cache=False, configured_cache_directory=self.cache_directory,
            warmup_seconds=self.warmup_seconds, global_installation=False,
            constitutive_equivalence_qualified=False,
            numerical_limit="unchanged 483 finite-iteration root; strict Python35 parity is not universal")


def validate_compiled_binding(binding):
    implementation = binding.implementation
    _require(type(implementation) is _Implementation, "factory-created compiled implementation required")
    _require(binding.material_kernels.update_batch is implementation.acceleration.numba_exact_trial_update_batch
             and binding.material_kernels.tangent_batch is implementation.cu.trial_tangent_action_batch
             and binding.material_kernels.elastic_stiffness is implementation.cu.isotropic_stiffness
             and type(binding.cu_card) is implementation.cu.CuCard,
             "actual compiled update, corrected tangent, elastic and card identities required")
    implementation.validate()


def create_compiled_material(*, cu_source_path, cu_source_sha256,
                             acceleration_source_path, acceleration_source_sha256,
                             native_exact_module, reduced_module, cache_directory,
                             card_parameters=None, scope="experimental"):
    """Return pinned kernels/card/binding and their private Python reference.

    Factory warmup is reported separately. New argument layouts that require
    another JIT overload require a new binding; there is no implicit fallback.
    """
    import numpy as np
    import numba
    import llvmlite

    _require(scope in ("experimental", "synthetic_fixture"), "experimental compiled binding only")
    _require(not numba.config.DISABLE_JIT, "compiled execution required; NUMBA_DISABLE_JIT is forbidden")
    from .assembly_mechanics import native
    _require(native_exact_module is native("exp12_all_gauss_exact_acceleration")
             and reduced_module is native("exp12_all_gauss_reduced_fe2"),
             "selected actual native runtime modules required; no substitutes")
    cache = _cache_directory(cache_directory)
    paths = ( _source(cu_source_path, cu_source_sha256, "exp12_cu_material_point.py"),
              _source(acceleration_source_path, acceleration_source_sha256, "exp12_all_gauss_exact_acceleration.py"))
    originals = _native_functions(native_exact_module, reduced_module)
    cu, cu_transform = _execute(*paths[0])
    names = ("CuBatchState", "CuBatchTrial", "CuCard", "isotropic_stiffness", "thermal_strain_vector")
    injected = {name: getattr(cu, name) for name in names}
    injected.update(reduced=reduced_module, baseline_trial_update_batch=cu.trial_update_batch)
    acceleration, acceleration_transform = _execute(*paths[1], injected)
    card = cu.CuCard(**(card_parameters or {}))
    _require(card.creep_law == "modified_time_hardening", "explicit corrected integral-clock Cu card required")
    _require(np.isfinite(card.creep_step_coefficient(350., 90., elapsed_time_s=1040.))
             and card.creep_step_coefficient(350., 90., elapsed_time_s=1040.) > 0,
             "finite positive corrected integral-clock coefficient required")
    kernels = SimpleNamespace(update_batch=acceleration.numba_exact_trial_update_batch,
        tangent_batch=cu.trial_tangent_action_batch, elastic_stiffness=cu.isotropic_stiffness)
    began = time.monotonic()
    for strain_readonly in (False, True):
        for state_readonly in (False, True):
            state = cu.CuBatchState.zeros(2)
            strain = np.zeros((2, 6))
            strain.flags.writeable = not strain_readonly
            for name in ("plastic_strain", "creep_strain", "equivalent_plastic_strain", "equivalent_creep_strain"):
                getattr(state, name).flags.writeable = not state_readonly
            acceleration.numba_exact_trial_update_batch(strain, 25., 0., state,
                card=card, creep_active=False, plastic_active=False, elapsed_time_s=300.)
    warmup = time.monotonic() - began
    for name in _DISPATCHERS:
        getattr(acceleration, name).disable_compile()
    signatures = tuple(_dispatcher_signature(getattr(acceleration, name)) for name in _DISPATCHERS)
    originals = tuple(_function_signature(value) if isinstance(value, FunctionType) else value for value in originals)
    implementation = _Implementation(cu, acceleration, native_exact_module, reduced_module, originals,
        (_namespace_signature(cu), _namespace_signature(acceleration)), signatures,
        (numba.__version__, llvmlite.__version__),
        tuple((str(path), SOURCE_PINS[path.name]) for path, _ in paths),
        tuple((str(path), json.dumps(transform, sort_keys=True)) for (path, _), transform in
              zip(paths, (cu_transform, acceleration_transform))), str(cache), warmup)
    binding = runtime.CorrectedCompiledMaterialBinding(material_kernels=kernels, cu_card=card,
        source_path=str(paths[0][0]), source_sha256=cu_source_sha256, card_json=runtime._material_card_json(card),
        kernel_functions=tuple((value, value.__code__) for value in runtime._material_kernel_functions(kernels)),
        scope=scope, card_methods=tuple((value, value.__code__) for value in runtime._material_card_methods(card)),
        implementation=implementation)
    runtime._validate_corrected_material_binding(binding, material_kernels=kernels, cu_card=card)
    return SimpleNamespace(material_kernels=kernels, cu_card=card, binding=binding,
                           python_reference=cu, warmup_seconds=warmup)
