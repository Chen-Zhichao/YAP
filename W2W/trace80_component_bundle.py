"""Exact BI4/Si37 compact operands for the existing component catalog route.

The export preserves raw traces. Only a certified common rigid translation is
removed from a separate bridge trace. No reduced operator or Cu law is changed.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import stat
import time
import zipfile


ROOT = Path('/app/nanocad/projects/personal/zchen/YAP_STACK')
RT = ROOT/'operator_repair_20261015/development/trial_forcing_closure_v1/continuous_cu_screen_v1/runtime'
PROFILE = 'trace80_bi4_si37'
ROLES = ('bi', 'beol', 'si')
RANKS = dict(bi=19, beol=20, si=18)
COUNTS = dict(bi=121728, beol=569088, si=348160)
NODES = dict(bi=65863, beol=178771, si=197589)
BODY_COLUMNS = dict(bi=138240, beol=470016, si=525312)
THICKNESS = dict(bi=.0015, beol=.010, si=.037)
OFFSETS = dict(bi=0., beol=.0015, si=.0115)
FACE_NODES = 9409
MIB, GIB = 1024**2, 1024**3
COMPONENT_LIMIT = 6*GIB
MANIFEST_SCHEMA = 'yap.trace80.compact_geometry_manifest.v1'
QUALIFICATION_SCHEMA = 'yap.trace80.runtime_qualification.v1'
UNIT_GEOMETRY = dict(bi_pitch_um=4, diameter_pitch_ratio=.5, bi_thickness_um=1.5,
    beol_thickness_um=10, si_tsv_thickness_um=37, total_thickness_um=48.5,
    bi_layer_count=1, tsv_pitch_um=6, cell_side_um=12)
OPROOT = RT/'ENRICHED_TRACE80_OPERATORS_20261023_sysml_v4'
OPERATORS = {r:dict(path=str(OPROOT/(r+'_OPERATOR_V2.npz')), bytes=n, sha256=h)
    for r,n,h in (
        ('bi',1109045190,'9d90bd80ddf4b8f300ea8e82242cebd42012af802bacb7a01d5da09d67b63f1b'),
        ('beol',5185910750,'fd9c344b0aac1f551b71b086774c7c5d70f551897536cadc76926eb4ba2eaa7a'),
        ('si',3170097622,'c2cef0629aff86530d1090db1f9783477a4cabf63f899ee6dd2da3919c174a7f'))}
REFERENCES = {r:dict(path=str(RT/p),bytes=n,sha256=h) for r,p,n,h in (
    ('bi','ENRICHED_BI_COMPONENT_REFERENCE_p4_trace80_20261023_mfd_v2/bi_COMPONENT_REFERENCE_V2.npz',
     1710927564,'3ca3a3fda2023d0aabab784cedd99b6d774b997fe6daf7c124e3a09cc06f45c5'),
    ('beol','ENRICHED_BEOL_COMPONENT_REFERENCE_p4_trace80_20261023_sysml_v3/beol_COMPONENT_REFERENCE_V2.npz',
     7162549812,'a02bee5e4045f01a25a2666caf1b9d3ce016c09f5a07623992bffd389e287511'),
    ('si','ENRICHED_SI_COMPONENT_REFERENCE_p4_trace80_20261023_mfd_v2/si_COMPONENT_REFERENCE_V2.npz',
     5288769028,'3bb68dd7e15b3df3d434bbb3b95f8c39359dad18e2ceb5456d4fe67cd09fc5d0'))}
TOPROOT = RT/'ENRICHED_TRACE80_TOPOLOGY_20261023_mfd_v3'
SHARED = dict(
    topology=dict(path=str(TOPROOT/'topology_COMPONENT_REFERENCE_V2.npz'),bytes=853886,
        sha256='3e3fa20f4107e64dea1a683ddf810520fc24b1b9fbdf8158a33fa11454126e20'),
    macro_maps=dict(path=str(TOPROOT/'MACRO_MAPS_V2.npz'),bytes=1299578,
        sha256='c4ab31f4c7140b0d945175ff16d0004b50c844a9fc44704fc51ea7ab4aa1f40a'))
CORE = ROOT/'operator_repair_20261015/development/trial_forcing_closure_v1/continuous_cu_screen_v1/trace80_compact_geometry_20261024.py'
CORE_SHA = '5f642f82f54b2fc0759dbdd3f145bbf8c863986ad926269ddecf835a5d6023f7'
CONSTRUCTION = dict(path=str(OPROOT/'REPORT.json'),bytes=15759,
    sha256='c00cad4df720711fa287215d308a3656c2afe78a8e0a2cb6537d0246d6933fd7')
UNIT_MANIFEST = dict(path=str(ROOT/'distribution/operators/validation/BI4_SI37_TRACE80_UNIT_20261024_v1/MANIFEST.json'),
    sha256='77d74f7a490046d97ae38d7dc7e5141dbf4c1b77e281ffd2035d36113daa7e05')
OPERATOR_KEYS = ('kaa','c','fm','ff','B_macro','B_free','weights','macro_map','macro_offset',
    'reference_cu_stiffness','reference_cu_alpha','reference_temperature_c','reference_materials',
    'kff_data','kff_indices','kff_indptr','kff_shape','Cu_positions_mm',
    'Cu_native_element_ids','Cu_native_gauss_indices')
COMPACT_KEYS = ('face_node_ids','source_face_node_indices','face_coordinates_mm','face_macro_lifts',
    'Cu_native_element_ids','Cu_native_gauss_indices','Cu_positions_mm','weights',
    'trace_reflection','trace_rank','reference_cu_stiffness','reference_cu_alpha',
    'reference_temperature_c','reference_materials')


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(value):
    _require(type(value) is str and len(value) == 64
             and all(c in '0123456789abcdef' for c in value), 'complete lowercase SHA required')
    return value


def _path(value):
    runtime = _runtime()
    return runtime._component_path(runtime.shared_distribution_path(value))


def _runtime():
    try:
        from .. import mechanics_runtime as runtime
    except ImportError:
        import mechanics_runtime as runtime
    return runtime


def _binding(value, limit=COMPONENT_LIMIT):
    _require(type(value) is dict and {'path','sha256'} <= set(value), 'bound artifact required')
    _path(value['path']); _sha(value['sha256'])
    if 'bytes' in value:
        _require(type(value['bytes']) is int and 0 < value['bytes'] <= limit, 'artifact byte bound differs')
    return value


def _same(value, expected):
    _binding(value,16*GIB)
    # Paths locate an authenticated copy; SHA/size identify the scientific input.
    # A configured external copy need not retain the producer's historical name.
    return all(value.get(k) == expected[k] for k in ('sha256','bytes') if k in expected)


def _read(reader, binding):
    _binding(binding,MIB)
    value, observed = reader.read(binding)
    # Publisher readers verify the requested size but return path/SHA only.
    expected = dict(path=binding['path'],sha256=binding['sha256'])
    _require(type(value) is dict and _same(observed,expected)
        and ('bytes' not in observed or 'bytes' not in binding or observed['bytes'] == binding['bytes']),
        'authenticated metadata binding differs')
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),
                                    allow_nan=False).encode('ascii')).hexdigest()


def entry_identity(entry):
    value = deepcopy(entry)
    value['validation'].pop('runtime_qualification',None)
    def content(item):
        if isinstance(item,dict):
            return {k:content(v) for k,v in item.items() if not (k == 'path' and 'sha256' in item)}
        if isinstance(item,list):
            return [content(v) for v in item]
        return item
    return _digest(content(value))


def _source_pins():
    yap = Path(__file__).resolve().parents[1]
    names = ('W2W/trace80_component_bundle.py','W2W/component_assembly_bridge.py','mechanics_runtime.py',
             'component_operators.py','tools/publish_component_operators.py')
    result = {}
    for name in names:
        path = yap/name
        _require(not {'#snapshot','.snapshot'}.intersection(path.parts), 'source snapshot access forbidden')
        with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb',buffering=0) as f:
            before = os.fstat(f.fileno())
            _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MIB,
                     'bounded shared runtime source required')
            raw = f.read(before.st_size+1)
            key = lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
            _require(len(raw) == before.st_size and key(before) == key(os.fstat(f.fileno()))
                     == key(path.lstat()), 'runtime source changed during authentication')
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def _prepared_geometry(geometry):
    keys = {'width_mm','length_mm','total_thickness_um','front_bi_thickness_um',
            'back_bi_thickness_um','beol_thickness_um'}
    _require(type(geometry) is dict and set(geometry) == keys and all(
        type(v) in (int,float) and math.isfinite(v) and v > 0 for v in geometry.values())
        and geometry['width_mm'] == geometry['length_mm']
        and geometry['total_thickness_um'] == 50.
        and geometry['front_bi_thickness_um'] == geometry['back_bi_thickness_um'] == 1.5
        and geometry['beol_thickness_um'] == 10., 'exact experimental dual-BI50 geometry required')


def _export(entry, reader):
    _require(entry.get('id') == 'BI4_SI37' and entry.get('profile') == PROFILE
        and entry.get('geometry') == UNIT_GEOMETRY and set(entry.get('components',{})) == set(ROLES),
        'exact single-BI48.5 TRACE80 catalog profile required')
    manifest = _read(reader,entry['physical_geometry'])
    _require(set(manifest) == {'schema','profile','sources','operators','shared_inputs','components',
                              'scale_derivation','qualification','source_closure'}
        and manifest['schema'] == MANIFEST_SCHEMA, 'exact compact export manifest required')
    profile = manifest['profile']
    expected = dict(id='BI4_SI37_TRACE80',case='bi4_d2_si37_cell12_transition96',port_rank_per_face=80,
        local_macro_coordinates=166,coupon_raw_macro_coordinates=326,coupon_gauge_free_macro_coordinates=323,
        coupon_coupled_coordinates=380,one_prepared_dual_BI_raw_macro_coordinates=406,
        one_prepared_dual_BI_gauge_free_macro_coordinates=403,two_prepared_dies_raw_macro_coordinates=726,
        two_prepared_dies_gauge_free_macro_coordinates=723,body_ranks=RANKS,physical_face_nodes=FACE_NODES,
        bi_pitch_um=4,diameter_pitch_ratio=.5,si_thickness_um=37,single_bi_coupon_total_um=48.5,
        prospective_dual_bi_total_um=50)
    _require(profile == expected and set(manifest['operators']) == set(ROLES)
        and set(manifest['components']) == set(ROLES) and set(manifest['shared_inputs']) == set(SHARED),
        'complete actual P4 profile/roles required')
    # Publisher metadata pins carry path/SHA only. Authenticate the known JSON
    # size here without weakening the byte contract of any numeric artifact.
    construction = dict(entry['construction'])
    if 'bytes' not in construction:
        construction['bytes'] = CONSTRUCTION['bytes']
        _read(reader, construction)
    _require(_same(manifest['sources']['construction'],CONSTRUCTION)
        and _same(construction,CONSTRUCTION)
        and _same(manifest['sources']['unit_manifest'],UNIT_MANIFEST)
        and any(Path(path).name == CORE.name and pin == CORE_SHA
                for path,pin in manifest['source_closure'].items()),
        'original construction and frozen compact core required')
    _require(all(manifest['qualification'].get(k) is False for k in (
        'operator_admission','assembly_accuracy_admission','thinning_admission','public_catalog_activated',
        'raw_face_lifts_changed','macro_translation_applied','actual_runtime_memory_qualified'))
        and manifest['qualification'].get('Cu_updates') == 0,
        'export must not invent runtime or scientific qualification')
    proofs = {}
    for role in ROLES:
        _require(_same(manifest['operators'][role],OPERATORS[role])
            and _same(entry['components'][role],OPERATORS[role])
            and entry['components'][role].get('micro_rank') == RANKS[role], 'exact original operator/rank required')
        record = manifest['components'][role]
        _require(set(record) == {'source','geometry','proof','body_rank','full_core_proof_preserved','array_bindings'}
            and _same(record['source'],REFERENCES[role]) and record['body_rank'] == RANKS[role]
            and record['full_core_proof_preserved'] is True, 'exact regional source/export required')
        _binding(record['geometry'],128*MIB)
        proof = _read(reader,record['proof']); proofs[role] = proof
        _require(proof.get('schema') == 'yap.trace80.compact_geometry_core.v1' and proof.get('role') == role
            and proof.get('exact_real_P4_census') is True and proof.get('synthetic_fixture') is False
            and proof.get('all_source_GP_retained') is True and proof.get('weights_and_materials_changed') is False
            and proof.get('GP_coordinates_transformed') is False and proof.get('Cu_updates') == 0
            and proof.get('source_node_index_base') == 0 and proof.get('exported_face_node_id_base') == 1
            and proof.get('census',{}).get('nodes') == NODES[role]
            and proof.get('census',{}).get('face_nodes') == FACE_NODES
            and proof.get('census',{}).get('gauss_points') == COUNTS[role]
            and proof.get('thickness_mm') == THICKNESS[role]
            and proof.get('GP_to_local_Y_subtraction_mm') == OFFSETS[role]
            and proof.get('body_trace',{}).get('face_P_rows_structurally_zero') is True
            and proof.get('body_trace',{}).get('face_rows_checked') == 6*FACE_NODES
            and proof.get('face_trace',{}).get('raw_face_lifts_changed') is False
            and proof.get('face_trace',{}).get('translation_applied') is False,
            'unchanged raw physical faces/complete ordered GP/zero-body-trace proof required')
        _require(set(record['array_bindings']) == set(COMPACT_KEYS), 'exact compact array inventory required')
        for row in record['array_bindings'].values():
            _sha(row['data_sha256'])
    for name, expected_binding in SHARED.items():
        row = manifest['shared_inputs'][name]
        _require(set(row) == {'source','artifact'} and _same(row['source'],expected_binding)
            and all(row['artifact'].get(k) == expected_binding[k] for k in ('sha256','bytes')),
            'unaltered original small topology/maps required')
        _binding(row['artifact'],2*MIB)
    reader.verify_unchanged()
    return manifest,proofs


def validate_catalog_entry(entry, metadata_reader):
    """Metadata only. A successful export is not runtime admission."""
    manifest,proofs = _export(entry,metadata_reader)
    validation = entry['validation']
    _require(validation.get('cycles') == 3 and validation.get('solved_states') == 73
        and validation.get('bow_observations') == 73, '73 actual coupon observations required, not1081')
    qualification = _read(metadata_reader,validation['runtime_qualification'])
    _require(qualification.get('schema') == QUALIFICATION_SCHEMA
        and qualification.get('state') == 'CLOSED_TRACE80_RUNTIME_OPERANDS_QUALIFIED'
        and qualification.get('profile') == PROFILE
        and qualification.get('entry_identity_sha256') == entry_identity(entry)
        and _same(qualification['physical_geometry'],entry['physical_geometry'])
        and qualification.get('runtime_sources') == _source_pins(), 'source-bound actual runtime qualification required')
    for key in ('operator_full_SHA_verified','all_compact_arrays_verified','bridge_constructed',
                'physical_registration_passed','memory_qualified'):
        _require(qualification.get(key) is True, 'positive actual runtime qualification required: '+key)
    _require(qualification.get('prepared_assembly_accuracy') is False
        and qualification.get('W2W_thinning') is False
        and qualification.get('two_prepared_dies_raw_coordinates') == 726
        and qualification.get('two_prepared_dies_free_coordinates') == 723,
        'experimental two-die interface qualification is not accuracy or thinning')
    _prepared_geometry(qualification['prepared_geometry'])
    _require(type(qualification.get('peak_RSS_bytes')) is int and qualification['peak_RSS_bytes'] > 0
        and type(qualification.get('address_space_bytes')) is int
        and qualification['address_space_bytes'] > qualification['peak_RSS_bytes']
        and type(qualification.get('allocation_forecast_upper_bytes')) is int
        and qualification['allocation_forecast_upper_bytes'] < qualification['address_space_bytes'],
        'actual positive measured memory and declared allocation envelope required')
    metadata_reader.verify_unchanged()
    return manifest,proofs,qualification


class MetadataReader:
    """Small authenticated public reads using the existing configured-root API."""
    def __init__(self):
        self.cache,self.identities,self.used = {},{},0

    def read(self,binding):
        runtime = _runtime(); _binding(binding,MIB); path = _path(binding['path'])
        _require(path.suffix == '.json', 'only JSON metadata is authorized')
        key = str(path)
        if key not in self.cache:
            with runtime._component_open(path,MIB,binding.get('bytes')) as stream:
                before = runtime._component_identity(stream)
                self.used += before[2]+1
                _require(self.used <= 32*MIB and len(self.cache) < 128, 'bounded metadata closure exceeded')
                raw = stream.read(before[2]+1)
                _require(len(raw) == before[2] and before == runtime._component_identity(stream),
                         'metadata changed during read')
            pin = hashlib.sha256(raw).hexdigest()
            def unique(pairs):
                result = {}
                for k,v in pairs:
                    _require(k not in result, 'duplicate metadata key')
                    result[k] = v
                return result
            value = json.loads(raw,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(
                ValueError('nonfinite metadata number')))
            self.cache[key] = value,dict(path=key,sha256=pin,bytes=len(raw))
            self.identities[key] = before
        value,proof = self.cache[key]
        _require(_same(proof,binding), 'metadata SHA/size differs')
        self.verify_unchanged()
        return value,proof

    def verify_unchanged(self):
        for name,expected in self.identities.items():
            value = os.stat(name,follow_symlinks=False)
            _require(stat.S_ISREG(value.st_mode) and (value.st_dev,value.st_ino,value.st_size,
                value.st_mtime_ns,value.st_ctime_ns) == expected, 'cached metadata identity changed')


def _metadata_reader(operand_reader):
    return getattr(operand_reader,'metadata_reader',None) or MetadataReader()


def _inventory(binding, expected_shapes):
    """Check every NPY header before any full GP array allocation."""
    import numpy as np
    try:
        from .. import mechanics_runtime as runtime
    except ImportError:
        import mechanics_runtime as runtime
    shapes, expanded = {},0
    with runtime._component_open(binding['path'],COMPONENT_LIMIT,binding['bytes']) as stream:
        before = runtime._component_identity(stream)
        with zipfile.ZipFile(stream) as archive:
            members = archive.infolist()
            _require(len(members) == len(expected_shapes) and len({m.filename for m in members}) == len(members)
                and {m.filename for m in members} == {n+'.npy' for n in expected_shapes}
                and all(m.compress_type == zipfile.ZIP_STORED and not m.flag_bits&1 for m in members),
                'exact flat stored TRACE80 archive inventory required')
            for member in members:
                name = member.filename[:-4]
                with archive.open(member) as handle:
                    version = np.lib.format.read_magic(handle)
                    _require(version in ((1,0),(2,0)), 'bounded original NPY header required')
                    parser = (np.lib.format.read_array_header_1_0 if version == (1,0)
                              else np.lib.format.read_array_header_2_0)
                    shape,fortran,dtype = parser(handle,max_header_size=4096)
                    size = math.prod(shape)*dtype.itemsize
                    _require(not fortran and dtype.kind in 'fiu' and not dtype.hasobject and len(shape) <= 3
                        and size+handle.tell() == member.file_size and size <= COMPONENT_LIMIT
                        and (expected_shapes[name] is None or tuple(shape) == tuple(expected_shapes[name])),
                        'bounded exact TRACE80 member dimensions required: '+name)
                    shapes[name] = (tuple(shape),dtype.str); expanded += size
        _require(expanded <= COMPONENT_LIMIT and before == runtime._component_identity(stream),
            'expanded archive size or descriptor identity changed')
    return shapes,expanded


def _operator_shapes(role):
    g,n = COUNTS[role],RANKS[role]
    return dict(kaa=(166,166),c=(n,166),fm=(166,),ff=(n,),B_macro=(6*g,166),B_free=(6*g,n),
        weights=(g,),macro_map=(166,326),macro_offset=(166,),reference_cu_stiffness=(6,6),
        reference_cu_alpha=(6,),reference_temperature_c=(),reference_materials=(4,3),
        kff_data=None,kff_indices=None,kff_indptr=(n+1,),kff_shape=(2,),
        V=(BODY_COLUMNS[role],n),native_body_columns=(BODY_COLUMNS[role],),trace_rank=(),
        Cu_positions_mm=(g,3),Cu_native_element_ids=(g,),Cu_native_gauss_indices=(g,))


def _derive_scale(topology, maps):
    import numpy as np
    scale,free,gauge,values = (topology[k] for k in ('coordinate_scale','free_map','gauge_matrix','gauge_values'))
    _require(scale.shape == (326,) and scale.dtype == np.dtype('<f8') and np.isfinite(scale).all()
        and np.all(scale > 0) and free.shape == (326,323) and gauge.shape == (3,326)
        and values.shape == (3,) and not np.any(values), 'complete original326/323 topology required')
    blocks = scale[6:].reshape(4,80)
    _require(all(np.array_equal(blocks[0],b) for b in blocks[1:]), 'all four face scales must match exactly')
    canonical = np.zeros((3,326))
    for i in range(4):canonical[:,6+80*i:9+80*i] = np.eye(3)/4
    transform = gauge@canonical.T@np.linalg.inv(canonical@canonical.T)
    _require(np.linalg.matrix_rank(transform) == 3
        and np.max(abs(gauge-transform@canonical)) <= 1e-12
        and np.linalg.norm(gauge@free) <= 1e-11 and np.linalg.matrix_rank(free) == 323,
        'only original three-translation gauge may be eliminated')
    for i,role in enumerate(ROLES):
        actual = maps[role+'_macro_map']; expected = np.zeros((166,326))
        _require(actual.shape == expected.shape and np.isfinite(actual).all(), 'complete166x326 role map required')
        expected[:6,:6] = actual[:6,:6]
        expected[6:86,6+80*i:86+80*i] = np.eye(80)
        expected[86:,86+80*i:166+80*i] = np.eye(80)
        _require(np.array_equal(actual,expected) and np.linalg.matrix_rank(actual[:6,:6]) == 6,
            'original adjacent-face and invertible physical-six map required')
    return scale[:166].copy()


def _convert(entry, geometry, reader, metadata_reader, manifest, proofs, *, check, maximum_forecast=None):
    """Shared conversion used by public loading and supervised qualification."""
    import numpy as np
    from . import component_assembly_bridge as bridge
    _require(callable(check), 'admitted caller guard required before conversion')
    _prepared_geometry(geometry); check()
    inventories,total = {},0
    for role in ROLES:
        headers,operator_bytes = _inventory(entry['components'][role],_operator_shapes(role))
        _require({k:headers[k][1] for k in ('V','native_body_columns','trace_rank')} ==
            dict(V='<f8',native_body_columns='<i4',trace_rank='<i8'),
            'original stored POD/body/trace metadata dtypes differ')
        compact = manifest['components'][role]
        expected = {name:tuple(row['shape']) for name,row in compact['array_bindings'].items()}
        inventories[role],size = _inventory(compact['geometry'],expected); total += size
        total += operator_bytes
        _require(operator_bytes+size+2*3*FACE_NODES*RANKS[role]*8 <= COMPONENT_LIMIT,
            'whole-component array/face/GP forecast exceeds6GiB before allocation')
    # Include one more full generation for simultaneous bridge/native temporaries.
    # The original four-generation estimate was below the measured graph peak.
    # This allocation forecast still needs a separate actual memory qualification.
    forecast = 5*total+512*MIB
    _require(maximum_forecast is None or forecast <= maximum_forecast,
        'expanded allocation exceeds qualified envelope before GP allocation')
    check()
    top = reader.arrays(manifest['shared_inputs']['topology']['artifact'],
                        ('coordinate_scale','free_map','gauge_matrix','gauge_values'))
    maps = reader.arrays(manifest['shared_inputs']['macro_maps']['artifact'],tuple(r+'_macro_map' for r in ROLES))
    scale = _derive_scale(top,maps)
    declared = manifest['scale_derivation']
    _require(declared.get('conditioning_only') is True and declared.get('physical_q6_conversion_performed') is False
        and declared.get('source_indices') == list(range(166))
        and declared.get('local_data_sha256') == hashlib.sha256(scale.tobytes()).hexdigest(),
        'independently derived original scale differs from export proof')
    operands,reflection,common = {},None,None
    for role in ROLES:
        check(); record = manifest['components'][role]
        data = reader.arrays(record['geometry'],COMPACT_KEYS)
        for name,value in data.items():
            binding = record['array_bindings'][name]
            _require(list(value.shape) == binding['shape'] and value.dtype.str == binding['dtype']
                and hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest()
                    == binding['data_sha256'], 'compact array/proof binding differs: '+name)
        values = dict(reader.arrays(entry['components'][role],OPERATOR_KEYS))
        _require(np.array_equal(values['macro_map'],maps[role+'_macro_map'])
            and not np.any(values['macro_offset']), 'operator macro map or affine offset differs')
        for name in ('Cu_native_element_ids','Cu_native_gauss_indices','Cu_positions_mm','weights',
                     'reference_cu_stiffness','reference_cu_alpha','reference_temperature_c','reference_materials'):
            _require(np.array_equal(data[name],values[name]), 'original ordered GP/material differs: '+name)
        xyz,raw = data['face_coordinates_mm'],data['face_macro_lifts']
        _require(xyz.shape == (2,FACE_NODES,3) and raw.shape == (2,3*FACE_NODES,166)
            and data['face_node_ids'].shape == (2,FACE_NODES)
            and np.array_equal(data['face_node_ids'],data['source_face_node_indices']+1),
            'exact ordered physical face/node-base dimensions required')
        lateral = xyz[0][:,[0,2]]
        _require(np.array_equal(lateral,xyz[1][:,[0,2]])
            and np.allclose(lateral.min(axis=0),0.,rtol=0,atol=1e-12)
            and np.allclose(lateral.max(axis=0),.012,rtol=0,atol=1e-12)
            and np.array_equal(np.lexsort((lateral[:,1],lateral[:,0])),np.arange(FACE_NODES)),
            'unchanged0..0.012mm sorted face frame required')
        trace = np.zeros((2,3*FACE_NODES,166+RANKS[role])); trace[:,:,:166] = raw
        translation = np.asarray(proofs[role]['face_trace']['macro_translation_mm_per_unit_q6'])
        _require(translation.shape == (3,6) and np.isfinite(translation).all(), 'certified rigid translation required')
        for face in (0,1):
            expected = bridge._canonical_macro_trace(xyz[face],(face-.5)*THICKNESS[role])
            difference = (raw[face,:,:6]-expected).reshape(FACE_NODES,3,6)
            _require(np.allclose(difference,translation,rtol=0,atol=1e-12),
                'macro trace requires more than the source-certified common rigid translation')
            trace[face,:,:6] -= np.tile(translation,(FACE_NODES,1))
        local = data['trace_reflection']
        _require(local.shape == (80,80) and int(data['trace_rank']) == 80
            and np.allclose(local.T@local,np.eye(80),rtol=0,atol=1e-12)
            and np.allclose(local@local,np.eye(80),rtol=0,atol=1e-12), 'original orthogonal80 reflection required')
        if reflection is None:reflection = local.copy(); common = raw[0,:,6:86].copy()
        _require(np.array_equal(reflection,local) and np.array_equal(raw[0,:,6:86],common),
            'cross-role physical port order/reflection differs')
        gp = dict(element_gauss_ids=np.column_stack((values.pop('Cu_native_element_ids'),
                      values.pop('Cu_native_gauss_indices'))),coordinates_mm=values.pop('Cu_positions_mm').copy())
        gp['coordinates_mm'][:,1] -= OFFSETS[role]
        _require(np.all(gp['coordinates_mm'][:,1] >= -1e-12)
            and np.all(gp['coordinates_mm'][:,1] <= THICKNESS[role]+1e-12), 'original GP role offset differs')
        for name in ('macro_map','macro_offset','reference_materials'):values.pop(name)
        public = 'retained_si' if role == 'si' else role
        operands[public] = dict(arrays=values,ordered_gp=gp,physical_trace=dict(
            face_node_ids=data['face_node_ids'],face_coordinates_mm=xyz,face_lifts=trace))
        _require(bridge._component_bytes(operands[public]) <= COMPONENT_LIMIT,
            'whole TRACE80 component arrays/trace/GP exceed6GiB before bridge copy')
    certificate = dict(schema='yap.component_prepared_assembly_certificate.v4',scope='experimental',
        geometry=dict(geometry),bi_layer_count=2,cell_area_mm2=.012**2,port_rank_per_face=80,
        macro_coordinate_scale=scale.tolist(),reflection_matrix=reflection.tolist(),rigid_port_mode_indices=[0,1,2],
        thickness_orientations=[1,-1],macro_contract=bridge._MACRO_CONTRACT,components={})
    for role,source in (('bi','bi'),('beol','beol'),('retained_si','si')):
        certificate['components'][role] = dict(thickness_mm=THICKNESS[source],virtual_work_certified=True,
            **{k+'_sha256':bridge._sha256_arrays(v) for k,v in operands[role].items()})
    metadata_reader.verify_unchanged(); check()
    return operands,certificate,{r:dict(entry['components'][r]) for r in ROLES},forecast


def _qualification_operands(entry, geometry, operand_reader, metadata_reader, *, check,
                            maximum_forecast=None):
    """Private supervised harness only; this does not qualify or publish a catalog."""
    _require(maximum_forecast is None or (type(maximum_forecast) is int and maximum_forecast > 0),
             'positive integer qualification allocation budget required')
    manifest,proofs = _export(entry,metadata_reader)
    return _convert(entry,geometry,operand_reader,metadata_reader,manifest,proofs,
                    check=check,maximum_forecast=maximum_forecast)


class _PublicGuard:
    """Per-load memory/deadline checks, without research-host membership rules."""
    def __init__(self,forecast,seconds=900):
        self.deadline = time.monotonic()+seconds
        self.initial = True
        self.forecast = forecast
        self.initial_vms = self._memory()['VmSize']

    @staticmethod
    def _fields(path,names):
        with open(path,'rb',buffering=0) as stream:
            raw = stream.read(65537)
        _require(len(raw) <= 65536, 'bounded operating-system memory observation required')
        rows = {}
        for line in raw.splitlines():
            fields = line.split()
            if fields and fields[0].rstrip(b':').decode('ascii') in names:
                _require(len(fields) == 3 and fields[2] == b'kB', 'memory units differ')
                rows[fields[0].rstrip(b':').decode('ascii')] = int(fields[1])*1024
        _require(set(rows) == set(names), 'complete operating-system memory observation required')
        return rows

    def _memory(self):
        return self._fields('/proc/self/status',('VmSize',))

    def __call__(self):
        _require(time.monotonic() < self.deadline, 'component loading deadline exceeded')
        soft,_ = resource.getrlimit(resource.RLIMIT_AS)
        required = self.initial_vms+self.forecast
        _require(soft == resource.RLIM_INFINITY or soft >= required,
                 'finite address-space limit is below the component allocation forecast')
        available = self._fields('/proc/meminfo',('MemAvailable',))['MemAvailable']
        _require(available >= (self.forecast if self.initial else 256*MIB),
                 'insufficient available memory for component loading')
        self.initial = False


def load_operands(entry, geometry, operand_reader):
    metadata = _metadata_reader(operand_reader)
    manifest,proofs,qualification = validate_catalog_entry(entry,metadata)
    _prepared_geometry(geometry)
    # Width/length affect the requested die kinematics, not the fixed cell's
    # maps, GP census, or allocations. The bridge certifies that geometry anew.
    # Reject the forecast before _convert can allocate any full GP maps.
    upper = sum(5*(OPERATORS[r]['bytes']+manifest['components'][r]['geometry']['bytes']) for r in ROLES)+512*MIB
    _require(upper <= qualification['allocation_forecast_upper_bytes'], 'qualified allocation envelope is insufficient')
    local = _PublicGuard(qualification['allocation_forecast_upper_bytes'])
    external = getattr(operand_reader,'check',None)
    _require(external is None or callable(external), 'invalid optional component guard')
    def check():
        local()
        if external is not None:
            external()
    operands,certificate,inputs,forecast = _convert(entry,geometry,operand_reader,metadata,manifest,proofs,
        check=check,maximum_forecast=qualification['allocation_forecast_upper_bytes'])
    return operands,certificate,inputs
