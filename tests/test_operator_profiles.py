"""Shared numerical profiles across ranks and both literal material contracts."""
from types import MappingProxyType

import numpy as np
import pytest

from tools.operators import acceleration, kinematics, reference, runtime, streamed


@pytest.fixture(scope="module",params=[runtime.HISTORICAL_CARD,runtime.PUBLIC_CARD],ids=["historical","public"])
def material(request):
    return runtime._compiled_material(request.param)[0]


def layer(material,rank):
    native = material.binding.implementation.reduced
    arrays = np.random.default_rng(rank).normal(size=(8,6,rank))*.02
    k,f = np.eye(rank)*1000.,np.linspace(-.1,.1,rank)
    op = kinematics.unit_invariant_layer_type(native)(macro_size=6,bubble_size=rank-6,
        reference_stiffness=lambda t:k,reference_thermal_load=lambda t:(t-25.)*f,
        strain_maps=arrays,weights_mm3=np.full(8,1e-5),macro_coordinate_scale=np.ones(6),
        bubble_coordinate_scale=np.ones(rank-6),cu_card=material.cu_card,plastic_active=True,
        material_kernels=material.material_kernels)
    runtime._components().runtime.attach_corrected_material_binding(op,material.binding)
    for value in vars(op).values():
        if isinstance(value,np.ndarray):value.setflags(write=False)
    op.normalized_operator_binding = MappingProxyType(dict(schema="yap.normalized_operator_binding.v1",
        coordinate_size=rank,integration_points=8))
    return op


CASES = [("native",9),("reference",9),("reference",11),("owned483",9),("owned483",27),
    ("owned483",55),("owned483",57),("streamed483",11),("streamed483",12),("coordinate483",9)]


@pytest.mark.parametrize("kind,rank",CASES)
def test_profiles_preserve_native_cu_response_and_actions(material,kind,rank):
    old,new = layer(material,rank),layer(material,rank)
    admit = runtime._shared_allocation_admission
    if kind == "reference":reference.attach_reference_tangent(new,material,admit=admit)
    if kind == "owned483":acceleration.attach_acceleration(new,material_binding=material.binding,admit=admit)
    if kind == "streamed483":streamed.attach_streamed_acceleration(new,material_binding=material.binding,admit=admit)
    if kind == "coordinate483":
        streamed.attach_coordinate_acceleration(new,material_binding=material.binding,admit=admit)
    birth = np.linspace(-1e-4,1e-4,rank)
    states = [op.initial_state(birth_macro_coordinates=birth[:6],birth_bubble_coordinates=birth[6:]) for op in (old,new)]
    q = np.linspace(-.03,.04,rank)
    direction = np.linspace(1e-5,2e-5,rank)
    for temperature,dt,endpoint in ((25.,0.,300.),(350.,20.,320.)):
        kwargs = dict(elapsed_time_s=endpoint,creep_active=dt>0,plastic_active=True)
        for state in states:
            for name in ("plastic_strain","creep_strain","equivalent_plastic_strain","equivalent_creep_strain"):
                getattr(state.cu_state,name).setflags(write=False)
        values = [op.evaluate(q[:6],q[6:],temperature,dt,state,**kwargs) for op,state in zip((old,new),states)]
        np.testing.assert_allclose(values[0].response,values[1].response,rtol=1e-12,atol=1e-11)
        for name in ("plastic_strain","creep_strain","equivalent_plastic_strain","equivalent_creep_strain"):
            np.testing.assert_allclose(getattr(values[0].cu_trial.state,name),getattr(values[1].cu_trial.state,name),rtol=1e-12,atol=1e-13)
        actions = [op.tangent_action(q[:6],q[6:],direction,temperature,dt,state,**kwargs) for op,state in zip((old,new),states)]
        np.testing.assert_allclose(actions[0],actions[1],rtol=1e-11,atol=1e-12)
        tangent = new.assembled_coordinate_tangent(q[:6],q[6:],temperature,dt,states[1],**kwargs)
        np.testing.assert_allclose(tangent@direction,actions[0],rtol=1e-10,atol=1e-12)
        for index,value in enumerate(values):states[index].cu_state = value.cu_trial.state.copy()


def test_streamed_owner_rejects_changed_private_block(material):
    op = layer(material,12)
    streamed.attach_streamed_acceleration(op,material_binding=material.binding,admit=runtime._shared_allocation_admission)
    owner = getattr(op,streamed.MARKER)
    owner.function.__globals__["_streamed_block_size"] = 17
    with pytest.raises(ValueError,match="owned streamed"):
        owner.validate(op)


def test_coordinate_only_keeps_native_methods_and_owns_namespace(material):
    op = layer(material,9)
    names = ("evaluate","evaluate_many","tangent_action","assembled_bubble_tangent")
    before = {name:getattr(getattr(op,name,None),'__func__',None) for name in names}
    streamed.attach_coordinate_acceleration(op,material_binding=material.binding,
        admit=runtime._shared_allocation_admission)
    assert not any(name in vars(op) for name in names)
    assert all(getattr(getattr(op,name,None),'__func__',None) is before[name] for name in names)
    owner = getattr(op,streamed.MARKER)
    owner.validate(op)
    owner.function.__globals__["_streamed_block_size"] = 17
    with pytest.raises(ValueError,match="owned streamed"):
        owner.validate(op)


def test_coordinate_only_rejects_replaced_native_method(material):
    op = layer(material,9)
    streamed.attach_coordinate_acceleration(op,material_binding=material.binding,
        admit=runtime._shared_allocation_admission)
    op.evaluate = lambda *args,**kwargs: None
    with pytest.raises(ValueError,match="retain native"):
        getattr(op,streamed.MARKER).validate(op)
