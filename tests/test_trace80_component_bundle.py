"""Software and optional real-header checks, not full GP decoding or accuracy."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy import sparse

import mechanics_runtime as runtime
from W2W import component_assembly_bridge as bridge
from W2W import trace80_component_bundle as m
from test_component_assembly_bridge import inputs, material, polynomial_fields
from test_component_trace80_bridge import trace80_fixture


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class Metadata:
    def __init__(self):
        self.rows = {}
        self.checks = 0

    def save(self,path,value):
        raw = json.dumps(value,sort_keys=True,allow_nan=False).encode('ascii')
        proof = dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        self.rows[str(path)] = value,proof
        return dict(proof)

    def read(self,binding):
        value,proof = self.rows[binding['path']]
        if proof['sha256'] != binding['sha256'] or ('bytes' in binding and binding['bytes'] != proof['bytes']):
            raise ValueError('metadata SHA differs')
        return value,proof

    def verify_unchanged(self):
        self.checks += 1


@pytest.fixture
def bundle(tmp_path,monkeypatch,inputs):
    monkeypatch.setattr(m,'FACE_NODES',64)
    monkeypatch.setattr(m,'COUNTS',dict.fromkeys(m.ROLES,8))
    monkeypatch.setattr(m,'NODES',dict.fromkeys(m.ROLES,128))
    monkeypatch.setattr(m,'BODY_COLUMNS',dict.fromkeys(m.ROLES,42))
    source_pins = {'W2W/trace80_component_bundle.py':'a'*64,'mechanics_runtime.py':'b'*64}
    monkeypatch.setattr(m,'_source_pins',lambda:dict(source_pins))
    metadata = Metadata()
    old = trace80_fixture(inputs)
    arrays,components,proofs,operators = {},{},{},{}
    maps = {}
    for i,role in enumerate(m.ROLES):
        mapping = np.zeros((166,326)); mapping[:6,:6] = np.eye(6)
        mapping[6:86,6+80*i:86+80*i] = np.eye(80)
        mapping[86:,86+80*i:166+80*i] = np.eye(80)
        maps[role+'_macro_map'] = mapping
        source = old['operands']['retained_si' if role == 'si' else role]
        values = dict(source['arrays']); rank = m.RANKS[role]
        old_kff = sparse.csr_matrix((values['kff_data'],values['kff_indices'],values['kff_indptr']),
                                   shape=tuple(values['kff_shape'])).toarray()
        kff = np.eye(rank); kff[0,0] = old_kff[0,0]; kff = sparse.csr_matrix(kff)
        values.update(c=np.vstack((values['c'],np.zeros((rank-1,166)))),ff=np.r_[values['ff'],np.zeros(rank-1)],
            B_free=np.column_stack((values['B_free'],np.zeros((48,rank-1)))),
            kff_data=kff.data,kff_indices=kff.indices,kff_indptr=kff.indptr,kff_shape=np.array(kff.shape),
            macro_map=mapping,macro_offset=np.zeros(166),reference_materials=np.ones((4,3)),
            V=np.zeros((42,rank)),native_body_columns=np.arange(42,dtype='<i4'),
            trace_rank=np.array(80,dtype='<i8'),
            Cu_native_element_ids=np.full(8,i+1,dtype=np.int64),Cu_native_gauss_indices=np.arange(8,dtype=np.int64))
        values['Cu_positions_mm'] = np.column_stack((np.full(8,.006),
            np.linspace(.1,.9,8)*m.THICKNESS[role]+m.OFFSETS[role],np.full(8,.006)))
        physical = source['physical_trace']; xyz = physical['face_coordinates_mm'].copy()
        xyz[:,:,[0,2]] *= 2
        raw = physical['face_lifts'][:,:,:166].copy()
        translation = np.arange(18).reshape(3,6)*1e-5
        for face in (0,1):
            raw[face,:,:6] = polynomial_fields(xyz[face],(face-.5)*m.THICKNESS[role]).reshape(-1,6)
            raw[face,:,:6] += np.tile(translation,(64,1))
        compact = dict(face_node_ids=physical['face_node_ids'].copy(),
            source_face_node_indices=physical['face_node_ids']-1,face_coordinates_mm=xyz,face_macro_lifts=raw,
            trace_reflection=np.asarray(old['certificate']['reflection_matrix']),trace_rank=np.array(80))
        for name in ('Cu_native_element_ids','Cu_native_gauss_indices','Cu_positions_mm','weights',
                     'reference_cu_stiffness','reference_cu_alpha','reference_temperature_c','reference_materials'):
            compact[name] = values[name].copy()
        op_path,geo_path = tmp_path/(role+'_OPERATOR.npz'),tmp_path/(role+'_GEOMETRY.npz')
        arrays[str(op_path)],arrays[str(geo_path)] = values,compact
        operator = dict(path=str(op_path),sha256=(str(i+1)*64),bytes=1000)
        operators[role] = operator
        proof = dict(schema='yap.trace80.compact_geometry_core.v1',role=role,exact_real_P4_census=True,
            synthetic_fixture=False,all_source_GP_retained=True,weights_and_materials_changed=False,
            GP_coordinates_transformed=False,Cu_updates=0,source_node_index_base=0,exported_face_node_id_base=1,
            census=dict(nodes=128,face_nodes=64,gauss_points=8),thickness_mm=m.THICKNESS[role],
            GP_to_local_Y_subtraction_mm=m.OFFSETS[role],
            body_trace=dict(face_P_rows_structurally_zero=True,face_rows_checked=384),
            face_trace=dict(raw_face_lifts_changed=False,translation_applied=False,
                            macro_translation_mm_per_unit_q6=translation.tolist()))
        proofs[role] = proof
        components[role] = dict(source=dict(m.REFERENCES[role]),
            geometry=dict(path=str(geo_path),sha256='c'*64,bytes=1000),
            proof=metadata.save(tmp_path/(role+'_PROOF.json'),proof),body_rank=rank,full_core_proof_preserved=True,
            array_bindings={k:dict(shape=list(v.shape),dtype=v.dtype.str,data_sha256=sha(v)) for k,v in compact.items()})
    monkeypatch.setattr(m,'OPERATORS',operators)
    gauge = np.zeros((3,326))
    for i in range(4):gauge[:,6+80*i:9+80*i] = np.eye(3)/4
    columns = np.r_[np.arange(6),np.arange(9,326)]
    free = np.eye(326)[:,columns]
    for row in range(3):
        for face in (1,2,3):
            free[6+row,np.flatnonzero(columns == 6+80*face+row)[0]] = -1
    topology = dict(coordinate_scale=np.ones(326),free_map=free,gauge_matrix=gauge,gauge_values=np.zeros(3))
    shared = {}
    for name,data in (('topology',topology),('macro_maps',maps)):
        path = tmp_path/(name+'.npz'); arrays[str(path)] = data
        shared[name] = dict(source=dict(m.SHARED[name]),artifact=dict(m.SHARED[name],path=str(path)))
    profile = dict(id='BI4_SI37_TRACE80',case='bi4_d2_si37_cell12_transition96',port_rank_per_face=80,
        local_macro_coordinates=166,coupon_raw_macro_coordinates=326,coupon_gauge_free_macro_coordinates=323,
        coupon_coupled_coordinates=380,one_prepared_dual_BI_raw_macro_coordinates=406,
        one_prepared_dual_BI_gauge_free_macro_coordinates=403,two_prepared_dies_raw_macro_coordinates=726,
        two_prepared_dies_gauge_free_macro_coordinates=723,body_ranks=dict(m.RANKS),physical_face_nodes=64,
        bi_pitch_um=4,diameter_pitch_ratio=.5,si_thickness_um=37,single_bi_coupon_total_um=48.5,
        prospective_dual_bi_total_um=50)
    manifest = dict(schema=m.MANIFEST_SCHEMA,profile=profile,
        sources=dict(construction=dict(m.CONSTRUCTION),unit_manifest=dict(m.UNIT_MANIFEST)),
        operators=operators,shared_inputs=shared,components=components,
        scale_derivation=dict(conditioning_only=True,physical_q6_conversion_performed=False,
            source_indices=list(range(166)),local_data_sha256=sha(topology['coordinate_scale'][:166])),
        qualification=dict.fromkeys(('operator_admission','assembly_accuracy_admission','thinning_admission',
            'public_catalog_activated','raw_face_lifts_changed','macro_translation_applied','actual_runtime_memory_qualified'),False),
        source_closure={str(m.CORE):m.CORE_SHA})
    manifest['qualification']['Cu_updates'] = 0
    entry = dict(id='BI4_SI37',profile=m.PROFILE,geometry=dict(m.UNIT_GEOMETRY),
        construction=dict(m.CONSTRUCTION),reference=dict(path=str(tmp_path/'REFERENCE.json'),sha256='d'*64),
        components={r:dict(v,micro_rank=m.RANKS[r]) for r,v in operators.items()},
        capabilities=dict(single_bi_history=True,experimental_prepared_d2w=True,
                          prepared_assembly_accuracy=False,W2W_thinning=False),
        validation=dict(cycles=3,solved_states=73,bow_observations=73),
        physical_geometry=metadata.save(tmp_path/'MANIFEST.json',manifest))
    geometry = dict(width_mm=10.,length_mm=10.,total_thickness_um=50.,front_bi_thickness_um=1.5,
                    back_bi_thickness_um=1.5,beol_thickness_um=10.)
    qualification = dict(schema=m.QUALIFICATION_SCHEMA,state='CLOSED_TRACE80_RUNTIME_OPERANDS_QUALIFIED',
        profile=m.PROFILE,entry_identity_sha256=m.entry_identity(entry),physical_geometry=entry['physical_geometry'],
        runtime_sources=source_pins,operator_full_SHA_verified=True,all_compact_arrays_verified=True,
        bridge_constructed=True,physical_registration_passed=True,memory_qualified=True,
        prepared_assembly_accuracy=False,W2W_thinning=False,two_prepared_dies_raw_coordinates=726,
        two_prepared_dies_free_coordinates=723,prepared_geometry=geometry,peak_RSS_bytes=128*m.MIB,
        address_space_bytes=4*m.GIB,allocation_forecast_upper_bytes=2*m.GIB)
    entry['validation']['runtime_qualification'] = metadata.save(tmp_path/'QUALIFIED.json',qualification)
    class Operands:
        metadata_reader = metadata
        def arrays(self,binding,names):
            return {k:np.array(arrays[binding['path']][k],copy=True) for k in names}
    def inventory(binding,shapes):
        data = arrays[binding['path']]
        assert set(data) == set(shapes)
        assert all(shape is None or data[k].shape == tuple(shape) for k,shape in shapes.items())
        return {k:(v.shape,v.dtype.str) for k,v in data.items()},sum(v.nbytes for v in data.values())
    monkeypatch.setattr(m,'_inventory',inventory)
    monkeypatch.setattr(m._PublicGuard,'_fields',staticmethod(lambda path,names:
        {name:(64*m.GIB if name == 'MemAvailable' else 128*m.MIB) for name in names}))
    getrlimit = m.resource.getrlimit
    monkeypatch.setattr(m.resource,'getrlimit',lambda kind:
        (4*m.GIB,4*m.GIB) if kind == m.resource.RLIMIT_AS else getrlimit(kind))
    return NS(entry=entry,manifest=manifest,metadata=metadata,proofs=proofs,reader=Operands(),arrays=arrays,
              qualification=qualification,geometry=geometry,topology=topology,maps=maps,original=old)


def test_metadata_validation_never_enters_numeric_loader(bundle,monkeypatch):
    monkeypatch.setattr(m,'_convert',lambda *a,**k:pytest.fail('numeric conversion during metadata validation'))
    assert m.validate_catalog_entry(bundle.entry,bundle.metadata)[2] == bundle.qualification


def test_publisher_reader_may_return_only_path_and_SHA_after_size_validation(bundle):
    original = bundle.metadata.read
    def read(binding):
        value,proof = original(binding)
        return value,{k:proof[k] for k in ('path','sha256')}
    bundle.metadata.read = read
    assert m.validate_catalog_entry(bundle.entry,bundle.metadata)[2] == bundle.qualification


@pytest.mark.parametrize('fault',['profile','unit50','rank','observations1081','missing_qualification',
    'qualification_false','source_drift','geometry_SHA','false_body_trace','proof_synthetic','operator_SHA'])
def test_public_metadata_refuses_wrong_or_unqualified_profile(bundle,fault):
    s = bundle
    if fault == 'profile':s.entry['profile'] = 'trace64'
    elif fault == 'unit50':s.entry['geometry']['total_thickness_um'] = 50.
    elif fault == 'rank':s.entry['components']['si']['micro_rank'] = 17
    elif fault == 'observations1081':s.entry['validation']['bow_observations'] = 1081
    elif fault == 'missing_qualification':s.entry['validation'].pop('runtime_qualification')
    elif fault == 'qualification_false':s.qualification['memory_qualified'] = False
    elif fault == 'source_drift':s.qualification['runtime_sources'] = {}
    elif fault == 'geometry_SHA':s.entry['physical_geometry']['sha256'] = 'f'*64
    elif fault == 'false_body_trace':s.proofs['bi']['body_trace']['face_P_rows_structurally_zero'] = False
    elif fault == 'proof_synthetic':s.proofs['si']['synthetic_fixture'] = True
    else:s.entry['components']['beol']['sha256'] = 'f'*64
    with pytest.raises((ValueError,KeyError)):
        m.validate_catalog_entry(s.entry,s.metadata)


def test_default_public_reader_needs_neither_exact_prlimit_nor_callback(bundle):
    assert not hasattr(bundle.reader,'check')
    operands,cert,inputs_ = m.load_operands(bundle.entry,bundle.geometry,bundle.reader)
    assert cert['schema'].endswith('.v4') and cert['port_rank_per_face'] == 80
    assert inputs_ == bundle.entry['components']
    assert [operands[r]['arrays']['c'].shape[0] for r in ('bi','beol','retained_si')] == [19,20,18]


def test_only_source_certified_rigid_translation_is_removed_without_mutating_raw(bundle):
    raw = bundle.arrays[bundle.manifest['components']['bi']['geometry']['path']]['face_macro_lifts']
    saved = raw.copy()
    operands,cert,_ = m.load_operands(bundle.entry,bundle.geometry,bundle.reader)
    assert np.array_equal(raw,saved)
    for role in ('bi','beol','retained_si'):
        source = 'si' if role == 'retained_si' else role
        bridge._trace_contract(operands[role]['physical_trace'],operands[role]['arrays'],m.THICKNESS[source],
            .012**2,physical_v2=True,reflection=np.asarray(cert['reflection_matrix']),port_rank_per_face=80)


def test_other_equal_square_lateral_sizes_do_not_change_cell_operators(bundle):
    geometry = dict(bundle.geometry,width_mm=20.,length_mm=20.)
    _,cert,_ = m.load_operands(bundle.entry,geometry,bundle.reader)
    assert cert['geometry'] == geometry and cert['cell_area_mm2'] == .012**2


@pytest.mark.parametrize('fault',['scale_block','gauge','map_order','singular_physical_six'])
def test_original_scale_derivation_refuses_wrong_topology_or_maps(bundle,fault):
    top,maps = deepcopy(bundle.topology),deepcopy(bundle.maps)
    if fault == 'scale_block':top['coordinate_scale'][166] = 2.
    elif fault == 'gauge':top['gauge_matrix'][0,0] = 1.
    elif fault == 'map_order':maps['si_macro_map'][6,167] = 1.
    else:maps['beol_macro_map'][:6,:6] = 0
    with pytest.raises(ValueError):m._derive_scale(top,maps)


def test_scale_is_not_an_assumed_ones_vector(bundle):
    top = deepcopy(bundle.topology)
    top['coordinate_scale'] = np.r_[np.arange(1.,7.),np.tile(np.arange(1.,81.),4)]
    assert np.array_equal(m._derive_scale(top,bundle.maps),top['coordinate_scale'][:166])


@pytest.mark.parametrize('fault',['array_SHA','GP_order','translation','reflection','frame'])
def test_conversion_refuses_compact_or_operator_mutation(bundle,fault):
    role = 'bi'; record = bundle.manifest['components'][role]
    data = bundle.arrays[record['geometry']['path']]
    if fault == 'array_SHA':data['weights'][0] *= 2
    elif fault == 'GP_order':
        data['Cu_native_gauss_indices'] = data['Cu_native_gauss_indices'][::-1].copy()
        record['array_bindings']['Cu_native_gauss_indices']['data_sha256'] = sha(data['Cu_native_gauss_indices'])
    elif fault == 'translation':bundle.proofs[role]['face_trace']['macro_translation_mm_per_unit_q6'][0][0] += 1
    else:
        name = 'trace_reflection' if fault == 'reflection' else 'face_coordinates_mm'
        if fault == 'reflection':data[name][0,0] = 2
        else:data[name][:,:,0] += .001
        record['array_bindings'][name]['data_sha256'] = sha(data[name])
    with pytest.raises(ValueError):m.load_operands(bundle.entry,bundle.geometry,bundle.reader)


def test_insufficient_qualified_forecast_refused_before_full_GP_reads(bundle):
    bundle.qualification['allocation_forecast_upper_bytes'] = 1
    bundle.reader.arrays = lambda *a,**k:pytest.fail('GP decode before forecast refusal')
    with pytest.raises(ValueError,match='envelope'):
        m.load_operands(bundle.entry,bundle.geometry,bundle.reader)


def test_old_four_generation_envelope_is_refused_before_GP_decode(bundle):
    total = sum(m.OPERATORS[r]['bytes']+bundle.manifest['components'][r]['geometry']['bytes']
                for r in m.ROLES)
    bundle.qualification['allocation_forecast_upper_bytes'] = 4*total+512*m.MIB
    bundle.reader.arrays = lambda *a,**k:pytest.fail('GP decode before stronger forecast refusal')
    with pytest.raises(ValueError,match='envelope'):
        m.load_operands(bundle.entry,bundle.geometry,bundle.reader)


@pytest.mark.parametrize('soft',[512*m.MIB,1024*m.MIB])
def test_public_guard_refuses_low_finite_address_space(monkeypatch,soft):
    monkeypatch.setattr(m._PublicGuard,'_fields',staticmethod(lambda p,n:
        {k:(64*m.GIB if k == 'MemAvailable' else 128*m.MIB) for k in n}))
    monkeypatch.setattr(m.resource,'getrlimit',lambda _: (soft,soft))
    with pytest.raises(ValueError,match='address-space'):
        m._PublicGuard(2*m.GIB)()


def test_bundle_refuses_low_address_space_before_numeric_decode(bundle,monkeypatch):
    monkeypatch.setattr(m.resource,'getrlimit',lambda _: (m.GIB,m.GIB))
    bundle.reader.arrays = lambda *a,**k:pytest.fail('GP decode before address-space refusal')
    with pytest.raises(ValueError,match='address-space'):
        m.load_operands(bundle.entry,bundle.geometry,bundle.reader)


def test_entry_content_identity_survives_data_relocation(bundle):
    entry = deepcopy(bundle.entry)
    def relocate(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key == 'path':value[key] = '/colleague/external/'+Path(item).name
                else:relocate(item)
    relocate(entry)
    assert m.entry_identity(entry) == m.entry_identity(bundle.entry)


def test_source_qualification_ids_are_relative_and_not_uid_dependent(monkeypatch):
    # Shared code can be owned by a colleague; the source bytes remain bound.
    monkeypatch.setattr(m.os,'getuid',lambda:987654321)
    pins = m._source_pins()
    assert len(pins) == 5 and all(not Path(k).is_absolute() for k in pins)


def test_public_metadata_reads_configured_nonrecorded_root_and_rechecks_cache(tmp_path,monkeypatch):
    monkeypatch.setenv('YAP_DATA_ROOT',str(tmp_path))
    monkeypatch.setattr(runtime,'_DATA_ROOT_SETTING',str(tmp_path))
    monkeypatch.setattr(runtime,'COMPONENT_PROJECT_ROOT',tmp_path)
    monkeypatch.setattr(runtime,'_artifact_locations',lambda root:frozenset())
    monkeypatch.setattr(runtime,'_read_locations',lambda:((tmp_path,),()))
    monkeypatch.setattr(m,'ROOT',Path('/not/the/colleagues/project'))
    path = tmp_path/'proof.json'; raw = b'{"scientific_identity":true}'
    path.write_bytes(raw)
    binding = dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    reader = m.MetadataReader()
    document,observed = reader.read(binding)
    assert runtime.COMPONENT_PROJECT_ROOT != runtime.RECORDED_PROJECT_ROOT
    assert document['scientific_identity'] is True
    assert observed['path'] == str(path)
    assert m._same(binding,dict(binding,path='/old/recorded/path.json'))
    path.write_bytes(b'{}')
    with pytest.raises(ValueError,match='identity'):
        reader.read(binding)


@pytest.mark.parametrize('fault',['extra','compressed','shape','object'])
def test_real_tiny_archive_headers_are_checked_before_numeric_allocation(tmp_path,fault):
    path = tmp_path/'tiny.npz'; arrays = dict(a=np.ones((2,3)))
    if fault == 'extra':arrays['extra'] = np.ones(1)
    elif fault == 'shape':arrays['a'] = np.ones((3,2))
    elif fault == 'object':arrays['a'] = np.array([object()],dtype=object)
    (np.savez_compressed if fault == 'compressed' else np.savez)(path,**arrays)
    binding = dict(path=str(path),bytes=path.stat().st_size,sha256='a'*64)
    with pytest.raises(ValueError):m._inventory(binding,dict(a=(2,3)))


@pytest.mark.parametrize('role',m.ROLES)
def test_actual_trace80_stored_operator_header_contract_without_array_payloads(role):
    binding = m.OPERATORS[role]
    if not Path(binding['path']).is_file():
        pytest.skip('external TRACE80 operator not installed')
    headers,expanded = m._inventory(binding,m._operator_shapes(role))
    assert len(headers) == 23
    assert set(headers)-set(m.OPERATOR_KEYS) == {'V','native_body_columns','trace_rank'}
    assert headers['V'] == ((m.BODY_COLUMNS[role],m.RANKS[role]),'<f8')
    assert headers['native_body_columns'] == ((m.BODY_COLUMNS[role],),'<i4')
    assert headers['trace_rank'] == ((),'<i8')
    assert 0 < expanded <= binding['bytes']


def test_actual_corrected_forecasts_cover_measured_graph_peak_without_GP_decode():
    manifest_path = m.RT/'BI4_TRACE80_COMPACT_GEOMETRY_20261025_mfd_AS4_manual2/MANIFEST.json'
    if not manifest_path.is_file() or not all(Path(v['path']).is_file() for v in m.OPERATORS.values()):
        pytest.skip('external measured TRACE80 inputs not installed')
    manifest = json.loads(manifest_path.read_text())
    expanded = 0
    for role in m.ROLES:
        _,size = m._inventory(m.OPERATORS[role],m._operator_shapes(role)); expanded += size
        compact = manifest['components'][role]
        _,size = m._inventory(compact['geometry'],
            {key:tuple(row['shape']) for key,row in compact['array_bindings'].items()})
        expanded += size
    decoded_forecast = 5*expanded+512*m.MIB
    archive_forecast = sum(5*(m.OPERATORS[r]['bytes']+manifest['components'][r]['geometry']['bytes'])
                           for r in m.ROLES)+512*m.MIB
    assert decoded_forecast == 49_248_031_412
    assert archive_forecast == 49_248_178_592
    assert 44_542_681_088 < decoded_forecast <= archive_forecast < 48*m.GIB


@pytest.mark.parametrize('name,dtype',[('V','<f4'),('native_body_columns','<i8'),('trace_rank','<f8')])
def test_stored_operator_diagnostic_dtype_is_checked_before_runtime_decode(bundle,name,dtype):
    values = bundle.arrays[bundle.entry['components']['bi']['path']]
    values[name] = values[name].astype(dtype)
    with pytest.raises(ValueError,match='metadata dtypes'):
        m.load_operands(bundle.entry,bundle.geometry,bundle.reader)


def test_existing_component_library_caller_reaches_real_trace80_bridge(bundle,material,monkeypatch,tmp_path):
    import component_operators as ops
    from W2W import component_compiled_material as compiled
    card,kernels,old_binding = material
    binding = runtime.bind_corrected_material(material_kernels=kernels,cu_card=card,
        source_path=old_binding.source_path,source_sha256=old_binding.source_sha256,scope='experimental')
    monkeypatch.setattr(ops,'resolve_entry',lambda package:bundle.entry)
    monkeypatch.setattr(ops,'configure_component_runtime',lambda:None)
    monkeypatch.setattr(runtime,'configure_runtime_storage',lambda:None)
    monkeypatch.setattr(ops,'OperandReader',lambda:bundle.reader)
    monkeypatch.setattr(ops,'_pinned_cu_card',lambda api:NS(sha256='a'*64,python_cu_card_kwargs=lambda:{}))
    monkeypatch.setattr(compiled,'create_compiled_material',lambda **kwargs:
        NS(cu_card=card,material_kernels=kernels,binding=binding))
    monkeypatch.setenv('NUMBA_CACHE_DIR',str(tmp_path))
    library,provenance,_ = ops._component_library_and_trace('component:BI4_SI37',bundle.geometry)
    assert library.port_rank_per_face == 80
    physical = provenance['physical_registration']['native_face_kinematics']
    assert physical['two_die_raw_macro_coordinates'] == 726
    assert physical['two_die_gauge_free_macro_coordinates'] == 723
    assert all(layer.macro_size == 166 for layer in library.operators.values())
    assert provenance['component_id'] == 'BI4_SI37' and provenance['W2W_thinning_available'] is False
