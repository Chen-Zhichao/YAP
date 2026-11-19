"""Tiny real native/Cu35/483 functions; compiled-binding plumbing is mocked.

No Numba/JIT calls, disk caches, production packets, launch or qualification.
"""
import ast
from dataclasses import dataclass, replace
import gc
import hashlib
import json
from pathlib import Path
import sys
from types import MappingProxyType, ModuleType, SimpleNamespace as NS
import weakref
from unittest.mock import Mock

import numpy as np
import pytest

from tools.operators import acceleration as subject


@pytest.fixture(scope="module")
def native():
    from W2W.assembly_mechanics import native as resolve
    return resolve("exp12_all_gauss_reduced_fe2")


def source(name):
    root = subject.layers.native_assets.runtime.COMPONENT_PROJECT_ROOT
    path = root/'distribution/operators/artifacts/operator_repair_20261015/source'/name
    with path.open('rb') as stream:raw = stream.read(262145)
    key = 'Cu_source' if name == 'exp12_cu_material_point.py' else 'acceleration_source'
    assert 0 < len(raw) <= 262144 and hashlib.sha256(raw).hexdigest() == subject.coupons.SOURCE_PINS[key]
    return path,raw


def execute(name, tree, namespace):
    module = ModuleType(name);vars(module).update(namespace)
    sys.modules[name] = module
    try:exec(compile(tree,'<pinned-private-python-fixture>','exec',dont_inherit=True),vars(module))
    finally:sys.modules.pop(name,None)
    return module


@pytest.fixture
def setup(native, monkeypatch):
    cu_path,cu_raw = source('exp12_cu_material_point.py')
    cu = execute('_shared_accel_cu35_fixture',cu_raw,{})
    _,raw = source('exp12_all_gauss_exact_acceleration.py')
    names = {'PreparedBatchTangent','_CacheEntry','_cache_entry_matches','_prepare_cached_tangent',
        'freeze_committed_batch_state','_validate_batch_inputs','prepare_batch_tangent',
        'apply_prepared_batch_tangent','_algorithmic_correction_times_map',*subject.METHODS.values()}
    tree = ast.parse(raw)
    selected = [n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert len(selected) == len(names)
    counters = next(n.value for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id == '_COUNTERS' for t in n.targets))
    namespace = dict(np=np,dataclass=dataclass,weakref=weakref,reduced=native,
        CuCard=cu.CuCard,CuBatchState=cu.CuBatchState,CuBatchTrial=cu.CuBatchTrial,
        isotropic_stiffness=cu.isotropic_stiffness,thermal_strain_vector=cu.thermal_strain_vector,
        _COUNTERS=ast.literal_eval(counters))
    namespace.update({'_ORIGINAL_'+name.upper():getattr(native.AllGaussReducedFE2Layer,name,None)
        for name in subject.METHODS})
    private = execute('_shared_accel_483_fixture',ast.fix_missing_locations(ast.Module(body=[
        ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*selected],
        type_ignores=[])),namespace)
    card = cu.CuCard(**dict(subject.runtime.PUBLIC_CARD))
    kernels = NS(update_batch=cu.trial_update_batch,tangent_batch=cu.trial_tangent_action_batch,
        elastic_stiffness=cu.isotropic_stiffness)
    impl = NS(reduced=native,acceleration=private,validate=Mock())
    runtime = subject.layers.native_assets.runtime
    binding = runtime.CorrectedCompiledMaterialBinding(material_kernels=kernels,cu_card=card,
        source_path=str(cu_path),source_sha256=subject.coupons.SOURCE_PINS['Cu_source'],
        card_json=runtime._material_card_json(card),kernel_functions=tuple(
            (f,f.__code__) for f in runtime._material_kernel_functions(kernels)),
        card_methods=tuple((f,f.__code__) for f in runtime._material_card_methods(card)),
        scope='synthetic_fixture',implementation=impl)
    def validate(value,*,material_kernels,cu_card):
        if value is not binding or material_kernels is not kernels or cu_card is not card:
            raise ValueError('synthetic binding identity changed')
    def attach(layer,value):
        validate(value,material_kernels=layer.material_kernels,cu_card=layer.cu_card)
        layer.corrected_material_binding=value;return layer
    checked_runtime = NS(CorrectedCompiledMaterialBinding=runtime.CorrectedCompiledMaterialBinding,
        _validate_corrected_material_binding=validate,attach_corrected_material_binding=attach)
    evidence = dict(schema='yap.corrected_compiled_material_binding.v1',global_installation=False,
        clock_semantics='increment_average_global_time_factor_at_physical_endpoint',
        synthetic_binding_plumbing_only=True,sources=[dict(path=str(cu_path),sha256=subject.coupons.SOURCE_PINS['Cu_source'])])
    def material(value, parameters):
        if type(value) is not runtime.CorrectedCompiledMaterialBinding:
            raise ValueError('actual CorrectedCompiledMaterialBinding required')
        subject.layers._exact_card(parameters,value.cu_card)
        return evidence,checked_runtime
    monkeypatch.setattr(subject.layers,'_material',material)
    def construct(private_rank=51):
        rank = private_rank + 6
        rng = np.random.default_rng(821)
        a = rng.normal(size=(rank, rank))
        stiffness = a.T @ a + 20 * np.eye(rank)
        load = rng.normal(size=rank) * 1e-6
        maps = rng.normal(size=(16, 6, rank)) * .02
        layer = subject.assets.unit_invariant_layer_type(native)(
            macro_size=6, bubble_size=private_rank, reference_stiffness=lambda t: stiffness,
            reference_thermal_load=lambda t: (t-25.)*load, strain_maps=maps,
            weights_mm3=np.linspace(1e-10, 2e-10, 16), macro_coordinate_scale=np.ones(6),
            bubble_coordinate_scale=np.ones(private_rank), cu_card=card, plastic_active=True,
            material_kernels=kernels)
        attach(layer, binding)
        for value in vars(layer).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        layer.normalized_operator_binding = MappingProxyType(dict(
            schema="yap.normalized_operator_binding.v1", coordinate_size=rank, integration_points=16))
        return layer
    return NS(native=native,private=private,cu=cu,binding=binding,impl=impl,construct=construct)


def attach(setup, operator=None, **changes):
    operator = setup.construct() if operator is None else operator
    options = dict(material_binding=setup.binding,check=lambda:None,admit=lambda *a:None)
    options.update(changes)
    return operator,subject.attach_acceleration(operator,**options)


def state_bytes(state):
    return tuple(getattr(state.cu_state,n).tobytes() for n in ('plastic_strain','creep_strain',
        'equivalent_plastic_strain','equivalent_creep_strain'))


@pytest.mark.parametrize('private_rank',[21,49,51])
def test_only_owned_instance_methods_change_and_real_functions_execute(setup,private_rank):
    operator = setup.construct(private_rank);other = setup.construct(private_rank)
    before = {n:getattr(setup.native.AllGaussReducedFE2Layer,n,None) for n in subject.METHODS}
    namespace = {n:id(v) for n,v in vars(setup.private).items()}
    operator,proof = attach(setup,operator)
    assert proof['per_instance_only'] and not proof['global_installation']
    assert proof['material']['synthetic_binding_plumbing_only']
    assert not proof['accuracy_claim'] and not proof['new_accuracy_acceptance_gate']
    assert not proof['assembly_execution_supported']
    with pytest.raises(TypeError):proof['per_instance_only']=False
    assert subject.validate_acceleration(operator) is proof
    for n in subject.METHODS:
        assert getattr(setup.native.AllGaussReducedFE2Layer,n,None) is before[n]
        if before[n] is None:assert not hasattr(other,n)
        else:assert getattr(other,n).__func__ is before[n]
        assert getattr(operator,n).__self__ is operator
    macro = np.linspace(-.02,.02,6);bubble = np.linspace(-.04,.04,private_rank)
    old,other_old = operator.initial_state(),other.initial_state()
    frozen = state_bytes(old)
    options = dict(creep_active=True,plastic_active=True,elapsed_time_s=1040.)
    baseline = other.evaluate(macro,bubble,350.,90.,other_old,**options)
    fast = operator.evaluate(macro,bubble,350.,90.,old,**options)
    np.testing.assert_allclose(fast.response,baseline.response,rtol=1e-13,atol=1e-15)
    assert operator.evaluate(macro,bubble,350.,90.,old,**options) is fast
    direction = np.linspace(-.1,.2,private_rank+6)
    answer = operator.tangent_action(macro,bubble,direction,350.,90.,old,**options)
    dense = operator.assembled_coordinate_tangent(macro,bubble,350.,90.,old,**options)
    small = operator.assembled_bubble_tangent(macro,bubble,350.,90.,old,**options)
    np.testing.assert_allclose(dense@direction,answer,rtol=1e-12,atol=1e-13)
    np.testing.assert_allclose(small,dense[6:,6:],rtol=1e-12,atol=1e-13)
    assert np.linalg.norm(dense-operator.reference_stiffness(350.)) > 1e-12
    assert frozen == state_bytes(old)
    assert all(not getattr(old.cu_state,n).flags.writeable for n in ('plastic_strain','creep_strain',
        'equivalent_plastic_strain','equivalent_creep_strain'))
    assert {n:id(v) for n,v in vars(setup.private).items()} == namespace
    assert setup.impl.validate.call_count >= 10
    assert not hasattr(other,'_exact_tangent_cache')
    assert subject.validate_acceleration(operator) is proof


def test_evaluate_many_and_original_cache_miss_tangent_path(setup):
    operator,_ = attach(setup)
    states = [operator.initial_state(),operator.initial_state()]
    macro,bubble = np.zeros(6),np.zeros(51)
    options = dict(creep_active=False,plastic_active=False,elapsed_time_s=300.)
    # Original483 miss falls through to original native tangent; adapter adds none.
    operator.tangent_action(macro,bubble,np.ones(57),25.,0.,states[0],**options)
    assert not hasattr(operator,'_exact_tangent_cache')
    result = operator.evaluate_many([macro,macro],[bubble,bubble],25.,0.,states,**options)
    assert len(result) == 2
    again = operator.evaluate_many([macro,macro],[bubble,bubble],25.,0.,states,**options)
    assert all(a is b for a,b in zip(result,again))


def test_dead_integer_id_entry_recomputes_real_material_response(setup):
    operator, _ = attach(setup)
    seed = operator.initial_state()
    seed.cu_state.plastic_strain[:, :3] = (1e-4, -5e-5, -5e-5)
    seed.cu_state.equivalent_plastic_strain[:] = 1e-4
    macro, bubble = np.zeros(6), np.zeros(51)
    options = dict(creep_active=False, plastic_active=False, elapsed_time_s=300.)
    previous = operator.evaluate(macro, bubble, 25., 0., seed, **options)
    assert np.linalg.norm(previous.response) > 1e-12
    abandoned = operator.initial_state().cu_state
    dead = weakref.ref(abandoned)
    del abandoned
    gc.collect()
    assert dead() is None
    fresh = operator.initial_state()
    cache = operator._exact_tangent_cache
    cache[id(fresh.cu_state)] = replace(cache[id(seed.cu_state)], committed_ref=dead)
    assert fresh.cu_state.plastic_strain.flags.writeable
    actual = operator.evaluate(macro, bubble, 25., 0., fresh, **options)
    other = setup.construct()
    reference = other.evaluate(macro, bubble, 25., 0., other.initial_state(), **options)
    np.testing.assert_allclose(actual.response, reference.response, rtol=1e-13, atol=1e-15)
    assert actual is not previous and cache[id(fresh.cu_state)].committed_ref() is fresh.cu_state
    assert not fresh.cu_state.plastic_strain.flags.writeable


@pytest.mark.parametrize('fault',['override','bubble_override','cache','binding','wrongtype'])
def test_preexisting_ambiguity_rejected_without_repair(setup,fault):
    operator = setup.construct()
    if fault == 'override':operator.evaluate=lambda *a,**k:None
    if fault == 'bubble_override':operator.assembled_bubble_tangent=lambda *a,**k:None
    if fault == 'cache':operator._exact_tangent_cache={}
    if fault == 'binding':
        value = dict(operator.normalized_operator_binding, schema="unreviewed")
        operator.normalized_operator_binding = subject.runtime._freeze(value)
    if fault == 'wrongtype':operator=object()
    with pytest.raises(ValueError):attach(setup,operator)
    assert not hasattr(operator,subject.MARKER)


def test_reattach_stale_owned_method_global_change_and_kernel_binding_refused(setup,monkeypatch):
    operator,_ = attach(setup)
    with pytest.raises(ValueError):attach(setup,operator)
    monkeypatch.setattr(operator,'evaluate',lambda *a,**k:None)
    with pytest.raises(ValueError):subject.validate_acceleration(operator)
    other,_ = attach(setup,setup.construct())
    monkeypatch.setattr(setup.native.AllGaussReducedFE2Layer,'tangent_action',lambda *a:None)
    with pytest.raises(ValueError):subject.validate_acceleration(other)


def test_clock_birth_cache_identity_and_batch_iterator_refused(setup):
    operator,_ = attach(setup);old = operator.initial_state()
    m,b = np.zeros(6),np.zeros(51)
    with pytest.raises(ValueError):operator.evaluate(m,b,25.,0.,old,creep_active=False)
    opts=dict(creep_active=False,elapsed_time_s=300.)
    operator.evaluate(m,b,25.,0.,old,**opts)
    old.birth_macro_coordinates[0] = .01
    with pytest.raises(ValueError):operator.evaluate(m,b,25.,0.,old,**opts)
    old.birth_macro_coordinates[0] = 0.
    operator._exact_tangent_cache={}
    with pytest.raises(ValueError):subject.validate_acceleration(operator)
    other,_ = attach(setup,setup.construct())
    with pytest.raises(ValueError):other.evaluate_many([m],[b],25.,0.,iter([other.initial_state()]),**opts)


def test_admission_no_fallback_and_failed_attachment_rollback(setup):
    operator,_ = attach(setup,admit=lambda *a:False)
    old=operator.initial_state();before=state_bytes(old)
    with pytest.raises(ValueError):operator.evaluate(np.zeros(6),np.zeros(51),25.,0.,old,
        creep_active=False,elapsed_time_s=300.)
    assert not hasattr(operator,'_exact_tangent_cache') and before==state_bytes(old)
    other=setup.construct()
    setup.impl.validate.side_effect=[None,RuntimeError('post-attachment validation')]
    with pytest.raises(RuntimeError):attach(setup,other)
    assert not hasattr(other,subject.MARKER)
    assert all(n not in vars(other) for n in subject.METHODS)


def test_primary_source_failure_not_masked_by_after_validation(setup,monkeypatch):
    operator,_ = attach(setup)
    old=operator.initial_state()
    owner=vars(operator)[subject.MARKER]
    original_validate=owner.validate;calls=[]
    def validate(value):
        calls.append(1)
        if len(calls)>1:raise RuntimeError('cleanup failed')
        return original_validate(value)
    monkeypatch.setattr(owner,'validate',validate)
    with pytest.raises(ValueError) as caught:
        operator.evaluate(np.zeros(5),np.zeros(51),25.,0.,old,creep_active=False,elapsed_time_s=300.)
    assert 'cleanup failed' in caught.value.__notes__[0]


def test_original483_function_and_all19_card_changes_refused(setup,monkeypatch):
    operator,_=attach(setup)
    monkeypatch.setattr(setup.private,'_accelerated_tangent_action',lambda *a:None)
    with pytest.raises(ValueError):subject.validate_acceleration(operator)
    other=setup.construct()
    object.__setattr__(setup.binding.cu_card,'creep_reference_stress_mpa',50.)
    with pytest.raises(ValueError):attach(setup,other)


def test_forecast_and_unowned_proof_fail_closed(setup):
    operator=setup.construct()
    with pytest.raises(ValueError):subject.validate_acceleration(operator)
    plan=subject.workspace_forecast(operator)
    assert not plan['capacity_proven'] and not plan['measured_peak']
    assert plan['original_483_full_GP_intermediates']
    with pytest.raises(ValueError):subject.workspace_forecast(operator,batches=65)


def test_owned_operator_buffer_and_reference_replacement_refused(setup):
    operator,_=attach(setup)
    operator.weights_mm3=operator.weights_mm3.copy()
    operator.weights_mm3.setflags(write=False)
    with pytest.raises(ValueError):subject.validate_acceleration(operator)
    other,_=attach(setup,setup.construct())
    other._reference_thermal_load=lambda temperature:np.zeros(57)
    with pytest.raises(ValueError):subject.validate_acceleration(other)
