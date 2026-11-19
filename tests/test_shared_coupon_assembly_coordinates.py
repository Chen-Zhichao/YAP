"""Coordinate compatibility only, not a material or assembly accuracy certificate."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from tools.coupon_assembly import whole_prepared_private_size


def operator(private):
    return SimpleNamespace(macro_size=6,bubble_size=private,coordinate_size=private+6,
        shared_coupon_binding=dict(schema="yap.shared_coupon_layer.v1",
            private_coordinates=private,reduced_coordinates=private+6,
            all_native_GP_retained=True,public_admission=False,assembly_qualified=False))


@pytest.mark.parametrize("private",[3,5,6,21,49,51])
def test_exact_supported_dimensions(private):
    assert whole_prepared_private_size(operator(private)) == private


@pytest.mark.parametrize("private",[0,1,24,48,50,52,True])
def test_unqualified_dimension_refused(private):
    with pytest.raises(ValueError):whole_prepared_private_size(operator(private))


@pytest.mark.parametrize("private",[5,6,21,49,51])
@pytest.mark.parametrize("change",["missing","wrong_rank","wrong_size","ports","GP_drop","public","assembly"])
def test_explicit_natural_rank_contract_required(change,private):
    value=deepcopy(operator(private))
    if change=="missing":del value.shared_coupon_binding
    elif change=="wrong_rank":value.shared_coupon_binding["private_coordinates"]=private+1
    elif change=="wrong_size":value.coordinate_size=private+7
    elif change=="ports":value.macro_size=8
    elif change=="GP_drop":value.shared_coupon_binding["all_native_GP_retained"]=False
    elif change=="public":value.shared_coupon_binding["public_admission"]=True
    elif change=="assembly":value.shared_coupon_binding["assembly_qualified"]=True
    with pytest.raises(ValueError):whole_prepared_private_size(value)


@pytest.mark.parametrize("private",[5,6,21,49,51])
def test_real_native_topology_and_trial_preserve_natural_private_states(private):
    from tools import coupon_assembly as assembly
    from W2W.assembly_mechanics import native
    rank=private+6
    stiffness=np.diag(np.arange(1.,rank+1.))
    thermal=np.arange(1.,rank+1.)*1e-8
    op=native("exp12_all_gauss_reduced_fe2").AllGaussReducedFE2Layer(
        macro_size=6,bubble_size=private,reference_stiffness=lambda t:stiffness,
        reference_thermal_load=lambda t:(t-25.)*thermal,
        strain_maps=np.zeros((0,6,rank)),weights_mm3=np.zeros(0),
        macro_coordinate_scale=np.ones(6),bubble_coordinate_scale=np.ones(private),
        plastic_active=False,prepared_cu_integrals=(np.zeros((rank,rank)),
            np.zeros((rank,rank)),np.zeros((rank,6))))
    op.shared_coupon_binding=operator(private).shared_coupon_binding
    rows=tuple(assembly.MacroOperand(name=f"body{i}",role="whole_prepared",operator=op,
        thickness_mm=.05,area_mm2=.000036,reference_y_mm=.025+.05*i,orientation=1,
        included_regions=("front_bi","beol","si","back_bi"),birth_q6=np.zeros(6),age_offset_s=13.*i)
        for i in range(2))
    body=assembly.build_whole_prepared_body(rows,global_reference_y_mm=0.,coordinate_scale=np.ones(6))
    assert [i.operator.age_offset_s for i in body.topology.instances]==[0.,13.]
    assert body.topology.instances[0].operator is not body.topology.instances[1].operator
    initial=body.initial_state()
    first,second=initial.layers.values()
    assert first.bubble_coordinates.shape==second.bubble_coordinates.shape==(private,)
    assert not np.shares_memory(first.bubble_coordinates,second.bubble_coordinates)
    trial=body.trial(initial,100.,1.,elapsed_time_s=1.,creep_active=False,
        plastic_active=False,linear_solver="assembled_dense",max_iterations=8,
        convergence_measure="residual",tolerance=1e-10)
    committed=body.commit(trial)
    assert np.isfinite(committed.generalized_coordinates).all()
    assert np.linalg.norm(committed.generalized_coordinates)>0.
    for name,state in committed.layers.items():
        assert state.bubble_coordinates.shape==(private,)
        assert np.isfinite(state.bubble_coordinates).all()
        assert not np.any(initial.layers[name].bubble_coordinates)
    assert not body.evidence["assembly_accuracy_admitted"]
    assert body.evidence["local_states_and_clocks_independent"]
