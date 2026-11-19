"""SPD-energy mathematics and authenticated constructor ABI, without old loaders."""
import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tools.operators import kinematics as coupon


def test_SPD_energy_skew_is_unhalved():
    k = np.eye(9)
    k[0,1],k[1,0] = 7.5e-11,-7.5e-11
    report = coupon.stiffness_symmetry(k)
    np.testing.assert_allclose(report['SPD_energy_unhalved_skew_spectral_norm'],1.5e-10,rtol=1e-14)
    assert not report['passed'] and report['symmetric_part_SPD']



@pytest.mark.parametrize('kind',['area','units','datum','combined'])
def test_SPD_energy_skew_congruence_invariant(kind):
    k = np.diag(np.arange(1.,10.))
    k[0,7],k[7,0] = 2e-8,-2e-8
    t = np.eye(9)
    if kind in ('datum','combined'): t[:3,3:6] = -.02425*np.eye(3)
    if kind in ('units','combined'): t = t@np.diag([1e-3,2.,10.,.1,1e3,.5,3.,.2,4.])
    area = 1/.000144 if kind in ('area','combined') else 1.
    transformed = area*(t.T@k@t)
    before = coupon.stiffness_symmetry(k)['SPD_energy_unhalved_skew_spectral_norm']
    after = coupon.stiffness_symmetry(transformed)['SPD_energy_unhalved_skew_spectral_norm']
    np.testing.assert_allclose(after,before,rtol=1e-7,atol=1e-15)



@pytest.mark.parametrize('kind',['nonSPD','nonpositive_diagonal','nonfinite'])
def test_SPD_energy_guard_rejects_invalid_energy(kind):
    k = np.eye(9)
    if kind == 'nonSPD': k[0,1] = k[1,0] = 2.
    elif kind == 'nonpositive_diagonal': k[0,0] = 0.
    else: k[0,1] = np.nan
    with pytest.raises(ValueError): coupon.stiffness_symmetry(k)



def _native_constructor_fixture(k):
    from W2W.assembly_mechanics import native
    original = native('exp12_all_gauss_reduced_fe2')
    private = coupon.unit_invariant_layer_type(original)
    maps = np.arange(16*6*9,dtype=float).reshape(16,6,9)*1e-6
    weights = np.linspace(.001,.002,16)
    kwargs = dict(macro_size=6,bubble_size=3,reference_stiffness=lambda t:k,
        reference_thermal_load=lambda t:np.zeros(9),strain_maps=maps,weights_mm3=weights,
        macro_coordinate_scale=np.ones(6),bubble_coordinate_scale=np.ones(3),
        prepared_cu_integrals=(np.zeros((9,9)),np.zeros((9,9)),np.zeros((9,6))))
    return original,private,kwargs



def test_private_constructor_exact_source_AST_inverse_and_methods():
    original,private,kwargs = _native_constructor_fixture(np.eye(9))
    raw = Path(original.__file__).read_bytes()
    tree,proof = coupon._guard_constructor_tree(raw)
    matches = [node for node in ast.walk(tree) if isinstance(node,ast.If)
        and ast.dump(node.test) == ast.dump(ast.parse('not _coupon_spd_energy_symmetric(stiffness)',mode='eval').body)]
    assert len(matches) == 1
    matches[0].test = ast.parse('not np.allclose(stiffness, stiffness.T, rtol=1.0e-10, atol=1.0e-12)',mode='eval').body
    assert ast.dump(tree) == ast.dump(ast.parse(raw))
    assert proof['whole_source_AST_inverse_verified'] and proof['predicate_replacements'] == 1
    with pytest.raises(ValueError,match='SHA'): coupon._guard_constructor_tree(raw+b'\n')
    for name in ('evaluate','evaluate_many','tangent_action','assembled_coordinate_tangent',
                 'reference_stiffness','reference_thermal_load','initial_state'):
        assert getattr(private,name) is getattr(original.AllGaussReducedFE2Layer,name)
    assert private.__init__ is not original.AllGaussReducedFE2Layer.__init__



def test_real_private_constructor_accepts_scaled_datum_and_retains_raw_operands():
    source = np.eye(9);source[0,7] = 2.5e-14
    t = np.eye(9);t[:3,3:6] = -.02425*np.eye(3)
    shifted = t.T@source@t
    k = shifted/.000144
    assert np.allclose(source,source.T,rtol=1e-10,atol=1e-12)
    assert np.allclose(shifted,shifted.T,rtol=1e-10,atol=1e-12)
    assert not np.allclose(k,k.T,rtol=1e-10,atol=1e-12)
    original,private,kwargs = _native_constructor_fixture(k)
    before = k.copy()
    with pytest.raises(ValueError,match='symmetric'): original.AllGaussReducedFE2Layer(**kwargs)
    layer = private(**kwargs)
    np.testing.assert_array_equal(layer.reference_stiffness(25.),before)
    np.testing.assert_array_equal(k,before)
    np.testing.assert_array_equal(layer.strain_maps,kwargs['strain_maps'])
    np.testing.assert_array_equal(layer.weights_mm3,kwargs['weights_mm3'])
    assert layer.reference_stiffness(25.)[0,7] != layer.reference_stiffness(25.)[7,0]
    assert coupon.stiffness_symmetry(k)['passed']



@pytest.mark.parametrize('kind',['skew','nonSPD'])
def test_real_private_constructor_rejects_skew_and_nonSPD(kind):
    k = np.eye(9)
    if kind == 'skew': k[0,1],k[1,0] = 7.5e-11,-7.5e-11
    else: k[0,1] = k[1,0] = 2.
    original,private,kwargs = _native_constructor_fixture(k)
    with pytest.raises(ValueError): private(**kwargs)



@pytest.mark.skipif(not os.environ.get("YAP_MIGRATION_BASELINE"), reason="explicit immutable source ABI witness")
def test_extracted_kinematics_matches_pinned_source_snapshot():
    baseline = Path(os.environ["YAP_MIGRATION_BASELINE"])
    raw = (baseline.parent / "SOURCE.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "e5da6bd0b77bc38bc317877672f5b90d8af4ee82b69f79ece770bbef8c7f2726"
    pin = json.loads(raw)["files"]["tools/native_coupon.py"]
    original = (baseline / "tools/native_coupon.py").read_bytes()
    assert hashlib.sha256(original).hexdigest() == pin
    def definitions(source):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef)}
    previous = definitions(original)
    current = definitions(Path(coupon.__file__).read_bytes())
    assert current.keys() <= previous.keys()
    assert {name: previous[name] for name in current} == current


