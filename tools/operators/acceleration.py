"""Optional owned per-instance delegation to the existing private Cu483 API.

No global install, class patch, material compilation, new tangent formula or
accuracy gate. Caller admits original all-GP work and supplies physical clocks.
Validation is executable ownership evidence, not numerical qualification.
"""
import hashlib
import inspect
import marshal
from pathlib import Path
from types import CodeType, FunctionType, MappingProxyType, MethodType
import weakref
from types import SimpleNamespace
import math

from . import runtime as coupons
from . import kinematics as assets
from . import runtime


def _exact_card(parameters, card=None):
    runtime.validate_card(dict(parameters))
    if card is not None:
        require({name:getattr(card,name,None) for name in parameters} == dict(parameters),
            'exact nineteen-field material contract changed')


def _material(binding, parameters):
    _exact_card(parameters,binding.cu_card)
    return assets._material(binding)


layers = SimpleNamespace(native_assets=assets,math=math,_exact_card=_exact_card,_material=_material)


METHODS = MappingProxyType({name:'_accelerated_'+name for name in (
    'evaluate', 'evaluate_many', 'tangent_action', 'assembled_bubble_tangent',
    'assembled_coordinate_tangent')})
MARKER = '_shared_coupon_acceleration_owner'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def workspace_forecast(operator, *, batches=1):
    """Conservative original-helper reservation, not a measured process peak."""
    gp, rank = operator.cu_point_count, operator.coordinate_size
    require(type(gp) is int and 0 < gp <= 20000000 and 6 < rank <= 128
        and type(batches) is int and 1 <= batches <= 64, 'bounded natural rank/GP/batch count required')
    return dict(additional_allocation_forecast_bytes=8*gp*6*rank*8+batches*gp*1024+64*1024**2,
        Cu_GP=gp, rank=rank, batches=batches, existing_operator_and_library_memory_excluded=True,
        original_483_full_GP_intermediates=True, measured_peak=False, capacity_proven=False)


def _native_methods(native):
    """Authenticate operational Python code against the actual pinned file."""
    assets = layers.native_assets
    path = assets._path(native.__file__)
    with assets.runtime._component_open(path, 256*1024) as stream:
        before = assets.runtime._component_identity(stream)
        raw = stream.read(256*1024+1)
        require(before == assets.runtime._component_identity(stream)
            and hashlib.sha256(raw).hexdigest() == assets.NATIVE_LAYER_SHA256,
            'exact bounded native layer source required')
    code = compile(raw, str(path), 'exec', dont_inherit=True)
    classes = [c for c in code.co_consts if isinstance(c, CodeType) and c.co_name == 'AllGaussReducedFE2Layer']
    require(len(classes) == 1, 'exact native class definition required')
    codes = {c.co_name:c for c in classes[0].co_consts if isinstance(c, CodeType)}
    base = native.AllGaussReducedFE2Layer
    functions = {name:value for name,value in vars(base).items() if isinstance(value, FunctionType)}
    require(all(name in codes and function.__code__ == codes[name]
        and function.__globals__ is vars(native) for name,function in functions.items()),
        'native operational class methods differ from pinned source')
    require(all(name in codes and value.fget.__code__ == codes[name]
        and value.fget.__globals__ is vars(native) for name,value in vars(base).items()
        if isinstance(value,property)), 'native operational property differs from pinned source')
    return base, functions


class _Owner:
    def __init__(self, operator, binding, runtime, evidence, *, check, admit, coordinate):
        self.ref, self.binding, self.runtime = weakref.ref(operator), binding, runtime
        self.impl, self.check, self.admit = binding.implementation, check, admit
        self.acceleration = self.impl.acceleration
        self.methods = {k:v for k,v in METHODS.items() if coordinate or k != 'assembled_coordinate_tangent'}
        self.card_parameters = {name:getattr(binding.cu_card,name) for name in globals()['runtime'].HISTORICAL_CARD}
        self.base, self.originals = _native_methods(self.impl.reduced)
        self.operator_type = type(operator)
        guarded = layers.native_assets.unit_invariant_layer_type(self.impl.reduced)
        require(self.operator_type is self.base or (self.operator_type.__name__ == guarded.__name__
            and self.operator_type.__bases__ == (self.base,)
            and set(vars(self.operator_type)) <= {'__module__', '__doc__', '__init__', 'native_coupon_constructor_guard'}
            and self.operator_type.__init__.__code__ == guarded.__init__.__code__),
            'only original or exact guarded shared native type supported')
        self.initializer = self.operator_type.__init__
        self.initializer_code = self.initializer.__code__
        self.dimensions = (operator.macro_size,operator.bubble_size,operator.coordinate_size,operator.cu_point_count)
        self.buffers = ('strain_maps','weights_mm3','macro_coordinate_scale','bubble_coordinate_scale',
            '_flat_strain_maps','_weighted_flat_strain_maps_transpose','_cu_lambda_stiffness_integral',
            '_cu_shear_stiffness_integral','_cu_constant_stress_virtual_work')
        self.buffer_metadata = self.metadata(operator)
        self.callbacks = operator._reference_stiffness,operator._reference_thermal_load
        self.properties = {name:(value,value.fget.__code__) for name,value in vars(self.base).items()
            if isinstance(value,property)}
        require(not any(name in vars(operator) for name in set(self.originals)|set(METHODS))
            and not hasattr(operator, '_exact_tangent_cache') and not hasattr(operator, MARKER),
            'existing instance overrides, acceleration ownership or cache forbidden')
        self.functions = {name:getattr(self.acceleration, source) for name,source in self.methods.items()}
        self.signatures = {name:inspect.signature(function) for name,function in self.functions.items()}
        for name in self.methods:
            old = self.originals.get(name)
            require(getattr(type(operator), name, None) is old
                and getattr(self.acceleration, '_ORIGINAL_'+name.upper()) is old,
                'exact old class/private source function identity required: '+name)
        self.bound, self.cache, self.contexts = {}, None, {}
        self.evidence = coupons._freeze(dict(schema='yap.shared_coupon_instance_acceleration.v1',
            material=evidence, source_pins=dict(coupons.SOURCE_PINS),
            native_layer_source_sha256=layers.native_assets.NATIVE_LAYER_SHA256,
            methods={name:dict(original_483_function=METHODS[name],
                source_code_sha256=hashlib.sha256(marshal.dumps(function.__code__)).hexdigest())
                for name,function in self.functions.items()},
            per_instance_only=True, global_installation=False, native_class_modified=False,
            original_class_method_present={name:name in self.originals for name in METHODS},
            material_kernels_modified=False, private_namespace_bindings_modified=False,
            private_source_counters_updated_by_original_functions=True,
            guarded_delegation_to_unchanged_483_functions=True,
            cache_miss_tangent='original483 fallback to original native Cu35 tangent; no added fallback',
            cache_and_commit_context='original weak Cu identity/coordinates/clock/flags; birth and buffer guards added',
            weak_cache_entry_identity_checked=True, stale_integer_id_not_a_cache_hit=True,
            constructor_explicit_factory_supported=False, new_constitutive_law=False,
            fixed_operator_buffer_identity_checked=True, caller_array_contents_not_rehashed=True,
            new_accuracy_acceptance_gate=False, speedup_claim=False, accuracy_claim=False,
            assembly_execution_supported=False, public_admission=False))

    def validate(self, operator):
        self.check()
        require(operator is self.ref() and type(operator) is self.operator_type
            and getattr(operator, MARKER, None) is self, 'owned instance acceleration binding lost')
        layers._exact_card(self.card_parameters, self.binding.cu_card)
        self.runtime._validate_corrected_material_binding(self.binding,
            material_kernels=operator.material_kernels, cu_card=operator.cu_card)
        self.impl.validate()
        require(operator.corrected_material_binding is self.binding and operator.factored_maps is None,
            'exact corrected material binding and original plain maps required')
        require((operator.macro_size,operator.bubble_size,operator.coordinate_size,operator.cu_point_count)
            == self.dimensions and operator.plastic_active is True
            and self.metadata(operator) == self.buffer_metadata
            and (operator._reference_stiffness,operator._reference_thermal_load) == self.callbacks,
            'fixed shared operator dimensions/buffers/reference callbacks changed')
        require(all(vars(self.base).get(name) is value and value.fget.__code__ is code
            for name,(value,code) in self.properties.items()), 'native property implementation changed')
        require(all(getattr(self.base,name) is function and function.__code__ is self.codes[name]
            and (name == '__init__' or getattr(type(operator),name) is function)
            for name,function in self.originals.items())
            and type(operator).__init__ is self.initializer
            and self.initializer.__code__ is self.initializer_code,
            'original native class/source changed')
        require(all(getattr(self.base,name,None) is self.originals.get(name) for name in METHODS),
            'native global optional method set changed')
        for name,bound in self.bound.items():
            require(vars(operator).get(name) is bound and bound.__self__ is operator
                and bound.__func__ is self.wrappers[name]
                and bound.__func__.__code__ is self.wrapper_codes[name]
                and bound.__func__.__globals__ is globals()
                and getattr(self.acceleration, METHODS[name]) is self.functions[name],
                'owned instance or original483 function changed: '+name)
        cache = vars(operator).get('_exact_tangent_cache')
        if cache is not None:
            require(type(cache) is dict and (self.cache is None or cache is self.cache),
                'original per-instance cache identity changed')
            self.cache = cache
        else:
            require(self.cache is None, 'original per-instance cache removed')
        self.check()
        return self.evidence

    def metadata(self, operator):
        return tuple((id(a),a.shape,a.strides,a.dtype.str,a.flags.writeable)
            for a in (getattr(operator,name) for name in self.buffers))

    def arguments(self, operator, name, args, kwargs):
        values = self.signatures[name].bind(operator, *args, **kwargs)
        values.apply_defaults()
        values = values.arguments
        end, dt = values['elapsed_time_s'], values['delta_time_s']
        require(type(end) in (int,float) and type(dt) in (int,float)
            and layers.math.isfinite(end) and layers.math.isfinite(dt) and 0 <= dt <= end,
            'explicit physical endpoint and nonnegative increment required')
        committed = values['committed']
        if name == 'evaluate_many':
            require(type(committed) in (tuple,list) and 1 <= len(committed) <= 64,
                'bounded concrete committed batch required; iterators are not consumed')
            states = committed
        else:
            states = (committed,)
        self.contexts = {key:value for key,value in self.contexts.items() if value[0]() is not None}
        for state in states:
            cu = state.cu_state
            fields = tuple(getattr(cu,n) for n in ('plastic_strain','creep_strain',
                'equivalent_plastic_strain','equivalent_creep_strain'))
            birth = (state.birth_macro_coordinates.tobytes(),state.birth_bubble_coordinates.tobytes())
            identity = tuple((id(a),a.shape,a.dtype.str) for a in fields), birth
            old = self.contexts.get(id(cu))
            require(old is None or old[0]() is not cu or old[1] == identity,
                'same Cu identity reused with changed buffers or birth fields')
            require(id(cu) in self.contexts or len(self.contexts) < 4096, 'bounded live cache contexts exceeded')
            self.contexts[id(cu)] = weakref.ref(cu), identity
            entry = None if self.cache is None else self.cache.get(id(cu))
            if entry is not None:
                require(type(entry) is self.acceleration._CacheEntry
                    and type(entry.committed_ref) is weakref.ReferenceType,
                    'original native cache entry and weak reference required')
                cached_cu = entry.committed_ref()
                require(cached_cu is None or cached_cu is cu, 'native cache key/weak identity mismatch')
                # Integer ids can be reused after the weakly held state dies.
                if cached_cu is cu:
                    require(all(not a.flags.writeable for a in fields), 'cached incoming Cu must remain frozen')
        forecast = workspace_forecast(operator,batches=len(states))
        decision = self.admit('BEFORE_ORIGINAL483_INSTANCE_'+name.upper(),forecast)
        require(decision is None or decision is True, 'original483 workspace admission refused')


def _invoke(operator, name, args, kwargs):
    owner = vars(operator).get(MARKER)
    require(type(owner) is _Owner, 'owned per-instance acceleration required')
    owner.validate(operator)
    owner.arguments(operator,name,args,kwargs)
    primary = None
    try:
        return owner.functions[name](operator,*args,**kwargs)
    except BaseException as error:
        primary = error;raise
    finally:
        try:owner.validate(operator)
        except BaseException as cleanup:
            if primary is None:raise
            try:primary.add_note('instance acceleration validation also failed: '+repr(cleanup)[:1024])
            except BaseException:pass


def _evaluate(operator, *args, **kwargs):
    return _invoke(operator,'evaluate',args,kwargs)


def _evaluate_many(operator, *args, **kwargs):
    return _invoke(operator,'evaluate_many',args,kwargs)


def _tangent_action(operator, *args, **kwargs):
    return _invoke(operator,'tangent_action',args,kwargs)


def _assembled_bubble_tangent(operator, *args, **kwargs):
    return _invoke(operator,'assembled_bubble_tangent',args,kwargs)


def _assembled_coordinate_tangent(operator, *args, **kwargs):
    return _invoke(operator,'assembled_coordinate_tangent',args,kwargs)


_WRAPPERS = MappingProxyType({name:globals()['_'+name] for name in METHODS})


def attach_acceleration(operator, *, material_binding, check=lambda:None, admit, coordinate=True):
    """Bind only this operator; explicit guarded validation runs around calls.

    The original483 functions may update their private diagnostic counters and
    this operator's cache. No native namespace binding, class or kernel changes.
    The unqualified custom-constructor path is deliberately refused.
    """
    require(callable(check) and callable(admit), 'explicit guard and allocation admission required')
    check()
    parameters = {name:getattr(material_binding.cu_card,name) for name in runtime.HISTORICAL_CARD}
    evidence,bound_runtime = layers._material(material_binding,parameters)
    require(type(material_binding) is bound_runtime.CorrectedCompiledMaterialBinding,
        'actual CorrectedCompiledMaterialBinding required')
    construction = getattr(operator,'normalized_operator_binding',None)
    require(isinstance(construction,MappingProxyType)
        and construction.get('schema') == 'yap.normalized_operator_binding.v1'
        and construction.get('coordinate_size') == operator.coordinate_size
        and construction.get('integration_points') == operator.cu_point_count
        and getattr(operator,'corrected_material_binding',None) is material_binding
        and operator.coordinate_size == operator.bubble_size+6 and operator.macro_size == 6
        and operator.factored_maps is None,'authenticated normalized operator binding required')
    material_binding.implementation.validate()
    owner = _Owner(operator,material_binding,bound_runtime,evidence,check=check,admit=admit,coordinate=coordinate)
    owner.codes = {name:function.__code__ for name,function in owner.originals.items()}
    owner.wrappers = {name:_WRAPPERS[name] for name in owner.methods}
    owner.wrapper_codes = {name:function.__code__ for name,function in owner.wrappers.items()}
    owner.bound = {name:MethodType(function,operator) for name,function in owner.wrappers.items()}
    vars(operator).update(owner.bound)
    vars(operator)[MARKER] = owner
    try:
        return owner.validate(operator)
    except BaseException:
        for name,bound in owner.bound.items():
            if vars(operator).get(name) is bound:del vars(operator)[name]
        if vars(operator).get(MARKER) is owner:del vars(operator)[MARKER]
        raise


def validate_acceleration(operator):
    """Owned executable-function proof required before accelerated parity use."""
    owner = vars(operator).get(MARKER)
    require(type(owner) is _Owner, 'no owned per-instance acceleration proof')
    return owner.validate(operator)
