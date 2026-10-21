"""Homogeneous H8/Bbar Si caps on the actual component native64 face trace.

Full periodic interior relaxation is performed before condensation. This is
not interpolation of Cu operators or admission of a new fabrication model.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time
import uuid

import numpy as np


MATERIAL = dict(elastic_modulus_mpa=91800., poisson_ratio=.28,
                secant_cte_per_c=2.6e-6, reference_temperature_c=25.)
VERSION = "full_periodic_interior_inherited_face_jumps_h8_bbar_si_native64_v2"
PROJECT = Path("/app/nanocad/projects/personal/zchen/YAP_STACK")
REACTION_LIMIT = 1e-7
CACHE_LIMIT = 2 * 1024**2
LOCK_TIMEOUT_S = 1800.
FACE_JUMP_ABSOLUTE_LIMIT = 1e-12
SOURCE_REGISTRATION_RELATIVE_LIMIT = 2e-8
BOUNDARY_CONTRACT = dict(version="inherited_face_jumps_affine_in_normalized_y_v1",
    policy="source_registration_per_column_absolute_plus_relative_no_trace_modification",
    absolute_limit=FACE_JUMP_ABSOLUTE_LIMIT, relative_limit=SOURCE_REGISTRATION_RELATIVE_LIMIT,
    source_reference_sha256="33c5371aa7185738c1ca1d822053451596beb9b1963d18936dd5d5cc3eab5c43",
    source_full128_max_affine_jump_departure=1.0079514843670332e-12,
    source_periodic_P_equivalence=True, source_P_shape=[44436, 30375],
    interior="complete_periodic_zero_face_fluctuations",
    total_field="canonical_q6_plus_unmodified_face_port_lift_plus_relaxed_periodic_fluctuation")
AUDIT_ERRORS = ("full_reaction_relative_defect", "crossblock_relative_defect",
    "tangent_relative_defect", "thermal_virtual_work_relative_defect",
    "thermal_reaction_relative_defect", "interior_equilibrium_relative_defect",
    "thermal_interior_relative_defect", "energy_relative_defect", "symmetry_relative_defect",
    "rigid_nullspace_relative_defect", "rigid_thermal_relative_defect")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_arrays(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        array = np.ascontiguousarray(value)
        digest.update(_json([name, array.dtype.str, list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _canonical_path(value):
    path = Path(value)
    _require(path.is_absolute(), "Si cap cache must be absolute")
    _require(not {"#snapshot", ".snapshot"}.intersection(path.parts), "snapshot cache forbidden")
    resolved = path.resolve()
    _require(resolved.is_relative_to(PROJECT), "Si cap cache must be inside the canonical project")
    _require(not {"#snapshot", ".snapshot"}.intersection(resolved.parts), "snapshot cache forbidden")
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink cache forbidden")
    return resolved


def _cache_root():
    override = os.environ.get("YAP_W2W_SI_CAP_CACHE_DIR")
    if override is None:
        work = os.environ.get("YAP_WORK_DIR")
        _require(bool(work), "set canonical YAP_WORK_DIR or YAP_W2W_SI_CAP_CACHE_DIR")
        override = str(Path(work) / "w2w_si_caps")
    root = _canonical_path(override)
    return root


def _published_root():
    return _canonical_path(os.environ.get("YAP_W2W_SI_CAP_LIBRARY_DIR",
        str(PROJECT / "distribution/yap_w2w_si_caps_20261021_v1")))


@contextmanager
def _lock(path, timeout=LOCK_TIMEOUT_S):
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    start = time.monotonic()
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - start >= timeout:
                    raise TimeoutError("Si cap cache lock wait exceeded finite deadline")
                time.sleep(min(.2, timeout))
        yield
    finally:
        os.close(descriptor)


def _native_sources():
    from .assembly_mechanics import native
    bbar = native("exp12_solid185_bbar")
    material = native("exp12_cu_material_point")
    return bbar, material, {Path(m.__file__).name: _file_sha(m.__file__)
                            for m in (bbar, material)}


def _face_jump_audit(ports, nx, nz, port_scale):
    grid = ports.reshape(nx, nz, 3, 64)
    x = np.max(np.abs(grid[0]-grid[-1]), axis=(0, 1))
    z = np.max(np.abs(grid[:, 0]-grid[:, -1]), axis=(0, 1))
    amplitude = np.max(np.abs(ports), axis=0)
    tolerance = FACE_JUMP_ABSOLUTE_LIMIT+SOURCE_REGISTRATION_RELATIVE_LIMIT*amplitude
    maximum = np.maximum(x, z)
    _require(np.all(maximum <= tolerance), "physical port jumps exceed the source registration contract")
    return dict(x_absolute_by_column=x.tolist(), z_absolute_by_column=z.tolist(),
        trace_amplitude_by_column=amplitude.tolist(), tolerance_by_column=tolerance.tolist(),
        normalized_by_trace_amplitude=(maximum/np.maximum(amplitude, np.finfo(float).tiny)).tolist(),
        maximum_absolute_coefficient_jump=float(np.max(maximum)),
        maximum_physical_jump_at_coordinate_scale_mm=float(np.max(maximum*port_scale)),
        source_policy=dict(BOUNDARY_CONTRACT), trace_values_modified=False)


def _validate(library, geometry, physical_trace):
    from .component_assembly_bridge import _canonical_macro_trace
    _require(library.port_rank_per_face == 64, "Si caps require native64")
    _require(tuple(library.rigid_port_mode_indices) == (0, 1, 2)
             and library.common_port_gauge_vectors is None, "three-translation native gauge required")
    scale = np.r_[library.membrane_bending_coordinate_scale,
                  library.port_coordinate_scale, library.port_coordinate_scale]
    _require(scale.shape == (134,) and np.isfinite(scale).all() and np.all(scale > 0),
             "finite positive native134 scales required")
    target = geometry.get("target_total_thickness_um")
    _require(type(target) in (int, float) and math.isfinite(target) and 13 < target < 775,
             "target total must be finite between 13 and 775 um")
    _require(geometry.get("initial_total_thickness_um") == 775
             and geometry.get("front_bi_thickness_um") == 1.5
             and geometry.get("back_bi_thickness_um") == 1.5
             and geometry.get("beol_thickness_um") == 10., "exact 775-um split-cap ledger required")
    for role, expected in dict(bi=.0015, beol=.010, retained_si=(target-13)/1000).items():
        _require(role in library.operators and math.isclose(library.thickness_mm[role], expected,
                 rel_tol=0, abs_tol=1e-12), "component thickness differs: " + role)
        _require(np.array_equal(library.operators[role].macro_coordinate_scale, scale),
                 "component scales differ: " + role)
    _require(set(physical_trace) == {"face_coordinates_mm", "face_lifts", "face_node_ids"},
             "exact physical BI trace required")
    trace = {key: np.array(value, copy=True) for key, value in physical_trace.items()}
    xyz, lifts, ids = (trace[k] for k in ("face_coordinates_mm", "face_lifts", "face_node_ids"))
    _require(xyz.ndim == 3 and xyz.shape[0] == 2 and xyz.shape[2] == 3,
             "two ordered physical faces required")
    n = xyz.shape[1]
    _require(ids.shape == (2, n) and ids.dtype.kind in "iu"
             and all(len(np.unique(face)) == n for face in ids)
             and lifts.ndim == 3 and lifts.shape[:2] == (2, 3*n) and lifts.shape[2] >= 134
             and np.isfinite(xyz).all() and np.isfinite(lifts).all(), "invalid physical face arrays")
    xs, zs = np.unique(xyz[0, :, 0]), np.unique(xyz[0, :, 2])
    _require(8 <= len(xs) == len(zs) <= 46 and len(xs)*len(zs) == n,
             "structured square physical face, at most 46 by 46, required")
    lateral = np.array([(x, z) for x in xs for z in zs])
    _require(np.array_equal(xyz[0, :, [0, 2]].T, lateral)
             and np.array_equal(xyz[0, :, [0, 2]], xyz[1, :, [0, 2]]),
             "physical face order must be x-major, z-minor")
    _require(np.allclose(xyz[0, :, 1], 0, rtol=0, atol=1e-12)
             and np.allclose(xyz[1, :, 1], .0015, rtol=0, atol=1e-12), "1.5-um BI faces required")
    _require(math.isclose(np.ptp(xs), np.ptp(zs), rel_tol=0, abs_tol=1e-12)
             and math.isclose(np.ptp(xs)*np.ptp(zs), library.cell_area_mm2,
                              rel_tol=0, abs_tol=1e-15), "physical face area differs")
    for face in (0, 1):
        _require(np.allclose(lifts[face, :, :6], _canonical_macro_trace(xyz[face], (face-.5)*.0015),
                            rtol=0, atol=1e-12), "BI macro trace is not centered canonical q6")
    ports = lifts[0, :, 6:70]
    _require(np.array_equal(ports, lifts[1, :, 70:134])
             and not np.any(lifts[0, :, 70:]) and not np.any(lifts[1, :, 6:70])
             and not np.any(lifts[:, :, 134:]), "common ports and zero face bubbles required")
    _require(np.linalg.matrix_rank(ports) == 64, "rank deficient physical face")
    jump_audit = _face_jump_audit(ports, len(xs), len(zs), scale[6:70])
    defect = jump_audit["maximum_absolute_coefficient_jump"]
    translations = ports[:, :3].reshape(n, 3, 3)
    _require(np.max(np.abs(translations-translations[:1])) <= 1e-12
             and np.linalg.matrix_rank(translations[0]) == 3, "first ports must span translations")
    reflection = np.asarray(library.reflection_matrix)
    _require(reflection.shape == (64, 64) and np.isfinite(reflection).all()
             and np.allclose(reflection.T@reflection, np.eye(64), rtol=0, atol=1e-12)
             and np.allclose(reflection@reflection, np.eye(64), rtol=0, atol=1e-12)
             and np.allclose(ports@reflection, np.tile([1., -1., 1.], n)[:, None]*ports,
                             rtol=0, atol=1e-12), "physical reflection closure failed")
    return trace, xs, zs, scale, float(target), float(defect)


def _bulk_recipe(thickness_um):
    """No rounding: retain one well-shaped exact remainder off the 10-um grid."""
    _require(math.isfinite(thickness_um) and thickness_um >= 10, "bulk must be at least 10 um")
    count = int(thickness_um // 10.)
    remainder = thickness_um - 10.*count
    if remainder == 0:
        return count, None
    count -= 1
    remainder = thickness_um - 10.*count
    _require(10 <= remainder < 20, "exact residual segment must be in [10,20) um")
    return count, remainder


def _assemble(xs, zs, ports, thickness_um, intervals, bbar, material):
    """Assemble nodal K, a complete periodic interior injection, and a lift."""
    from scipy import sparse
    from .component_assembly_bridge import _canonical_macro_trace
    nx, nz = len(xs)-1, len(zs)-1
    ys = np.linspace(0., thickness_um/1000, intervals+1)
    nface = len(xs)*len(zs)
    coordinates = np.array([(x, y, z) for y in ys for x in xs for z in zs])
    nodes = np.arange(len(coordinates)).reshape(intervals+1, nx+1, nz+1)
    origins = nodes[:-1, :-1, :-1].ravel()
    offsets = np.array([0, nz+1, nz+2, 1, nface, nface+nz+1, nface+nz+2, nface+1])
    connectivity = origins[:, None]+offsets
    dofs = (3*connectivity[:, :, None]+np.arange(3)).reshape(-1, 24)
    constitutive = material.isotropic_stiffness(MATERIAL["elastic_modulus_mpa"], MATERIAL["poisson_ratio"])
    alpha = np.r_[np.full(3, MATERIAL["secant_cte_per_c"]), np.zeros(3)]
    data = np.empty((len(connectivity), 24, 24))
    forces = np.empty((len(connectivity), 24))
    element_cache = {}
    for index, element in enumerate(connectivity):
        local = coordinates[element]-coordinates[element[0]]
        key = local.tobytes()
        if key not in element_cache:
            element_cache[key] = bbar.element_linear_operators(local, constitutive, alpha)
        data[index], forces[index] = element_cache[key]
    rows = np.broadcast_to(dofs[:, :, None], data.shape).ravel()
    columns = np.broadcast_to(dofs[:, None, :], data.shape).ravel()
    size = 3*len(coordinates)
    stiffness = sparse.coo_matrix((data.ravel(), (rows, columns)), shape=(size, size)).tocsr()
    load = np.bincount(dofs.ravel(), weights=forces.ravel(), minlength=size)
    del data, rows, columns, forces
    lift = np.zeros((size, 134))
    for y in range(intervals+1):
        section = slice(3*y*nface, 3*(y+1)*nface)
        lift[section, :6] = _canonical_macro_trace(coordinates[y*nface:(y+1)*nface], ys[y]-.5*ys[-1])
        lift[section, 6:70] = (1-y/intervals)*ports
        lift[section, 70:134] = (y/intervals)*ports
    internal_nodes = nodes[1:-1].ravel()
    ii, xx, zz = np.indices((intervals-1, nx+1, nz+1))
    master = ((ii*nx + xx % nx)*nz + zz % nz).ravel()
    interior_size = 3*(intervals-1)*nx*nz
    injection = sparse.coo_matrix((np.ones(3*len(internal_nodes)),
        ((3*internal_nodes[:, None]+np.arange(3)).ravel(),
         (3*master[:, None]+np.arange(3)).ravel())), shape=(size, interior_size)).tocsr()
    return stiffness, load, lift, injection, dict(node_count=len(coordinates),
        element_count=len(connectivity), periodic_interior_dofs=interior_size,
        unique_element_integrations=len(element_cache), thickness_intervals=intervals)


def _condense_fe(stiffness, load, lift, injection, scale):
    from scipy import sparse
    from scipy.sparse.linalg import splu
    k_lift = stiffness@lift
    coupling = injection.T@k_lift
    interior = (injection.T@stiffness@injection).tocsc()
    diagonal = interior.diagonal()
    _require(np.isfinite(diagonal).all() and np.all(diagonal > 0), "invalid full Si interior diagonal")
    equilibrate = 1/np.sqrt(diagonal)
    factor_start = time.monotonic()
    factor = splu((sparse.diags(equilibrate)@interior@sparse.diags(equilibrate)).tocsc(),
                  permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.)
    factor_seconds = time.monotonic()-factor_start
    interior_load = injection.T@load
    extension = np.empty_like(coupling)
    for first in range(0, 134, 8):
        last = min(first+8, 134)
        extension[:, first:last] = equilibrate[:, None]*factor.solve(
            -equilibrate[:, None]*coupling[:, first:last])
    thermal = equilibrate*factor.solve(equilibrate*interior_load)
    condensed = lift.T@k_lift+coupling.T@extension
    condensed_load = lift.T@load-coupling.T@thermal
    relaxed = lift+injection@extension
    reaction = stiffness@relaxed
    full_condensed = relaxed.T@reaction
    full_load = relaxed.T@load
    scaled = scale[:, None]*condensed*scale[None, :]
    norm = max(float(np.linalg.norm(scaled)), 1e-30)
    stiffness_defect = float(np.linalg.norm(scale[:, None]*(full_condensed-condensed)*scale[None, :])/norm)
    cross_defect = max(float(np.max(np.abs((scale[:, None]*(full_condensed-condensed)*scale[None, :])[a, b]))/norm)
                       for a in (slice(0, 6), slice(6, 70), slice(70, 134))
                       for b in (slice(0, 6), slice(6, 70), slice(70, 134)))
    thermal_defect = float(np.linalg.norm(scale*(full_load-condensed_load)) /
                           max(np.linalg.norm(scale*condensed_load), 1e-30))
    internal_defect = float(np.linalg.norm(equilibrate[:, None]*(interior@extension+coupling)*scale[None, :]) /
                            max(np.linalg.norm(equilibrate[:, None]*coupling*scale[None, :]), 1e-30))
    thermal_internal = float(np.linalg.norm(equilibrate*(interior@thermal-interior_load)) /
                             max(np.linalg.norm(equilibrate*interior_load), 1e-30))
    thermal_reaction = lift.T@(stiffness@(injection@thermal)-load)
    thermal_reaction_defect = float(np.linalg.norm(scale*(thermal_reaction+condensed_load)) /
                                    max(np.linalg.norm(scale*condensed_load), 1e-30))
    symmetry = float(np.linalg.norm(scaled-scaled.T)/norm)
    condensed = .5*(condensed+condensed.T)
    eigenvalues = np.linalg.eigvalsh(.5*(scaled+scaled.T))
    spectral = max(float(np.max(np.abs(eigenvalues))), 1e-30)
    rigid = np.zeros((134, 3))
    rigid[6:9] = rigid[70:73] = np.eye(3)
    null_defect = float(np.linalg.norm(scale[:, None]*(condensed@rigid)) /
                        max(norm*np.linalg.norm(rigid/scale[:, None]), 1e-30))
    null_thermal = float(np.linalg.norm(rigid.T@condensed_load) /
                         max(np.linalg.norm(condensed_load), 1e-30))
    rng = np.random.default_rng(775)
    energy_defect = 0.
    for _ in range(4):
        q = scale*rng.normal(size=134)
        displacement = relaxed@q
        exact = float(displacement@(stiffness@displacement))
        energy_defect = max(energy_defect, abs(float(q@condensed@q)-exact)/max(abs(exact), 1e-30))
    audits = dict(full_reaction_relative_defect=stiffness_defect,
        crossblock_relative_defect=cross_defect, tangent_relative_defect=stiffness_defect,
        thermal_virtual_work_relative_defect=thermal_defect,
        thermal_reaction_relative_defect=thermal_reaction_defect,
        interior_equilibrium_relative_defect=internal_defect,
        thermal_interior_relative_defect=thermal_internal, energy_relative_defect=energy_defect,
        symmetry_relative_defect=symmetry, rigid_nullspace_relative_defect=null_defect,
        rigid_thermal_relative_defect=null_thermal,
        minimum_scaled_eigenvalue_over_spectral_radius=float(eigenvalues[0]/spectral),
        fourth_scaled_eigenvalue_over_spectral_radius=float(eigenvalues[3]/spectral),
        factorization_seconds=factor_seconds, factor_nonzeros=int(factor.L.nnz+factor.U.nnz),
        solve_rhs_count=135, algebra_gate_limit=REACTION_LIMIT)
    errors = [value for name, value in audits.items() if name.endswith("relative_defect")]
    _require(np.isfinite(condensed).all() and np.isfinite(condensed_load).all()
             and np.isfinite(errors).all() and max(errors) <= REACTION_LIMIT,
             "full Si FE reaction/energy/thermal/nullspace gate failed: " + _json(audits))
    _require(eigenvalues[0] >= -1e-9*spectral and eigenvalues[3] > 1e-12*spectral,
             "Si FE stiffness has negative energy or unexpected nullspace")
    return condensed, condensed_load, audits


def _read_cache(npz_path, json_path, key):
    if not npz_path.exists() or not json_path.exists():
        return None
    with os.fdopen(os.open(json_path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        info = os.fstat(stream.fileno())
        _require(stat.S_ISREG(info.st_mode) and info.st_size <= 65536, "Si cap metadata size/type bound")
        metadata = json.loads(stream.read(65537))
    specification = metadata.get("specification")
    _require(isinstance(specification, dict) and metadata.get("key") == key
             and hashlib.sha256(_json(specification).encode()).hexdigest() == key,
             "Si cap cache specification hash differs from requested key")
    audits = metadata.get("audits")
    _require(isinstance(audits, dict) and all(name in audits for name in AUDIT_ERRORS),
             "Si cap cache requires complete successful full-FE audits")
    for name in AUDIT_ERRORS:
        value = audits[name]
        _require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= REACTION_LIMIT,
                 "Si cap cached FE audit failed: " + name)
    for name, low, high in (("minimum_scaled_eigenvalue_over_spectral_radius", -1e-9, 1.),
                           ("fourth_scaled_eigenvalue_over_spectral_radius", 1e-12, 1.)):
        value = audits.get(name)
        _require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high
                 and (name.startswith("minimum") or value > low), "Si cap cached energy/nullspace gate failed")
    _require(audits.get("solve_rhs_count") == 135 and audits.get("algebra_gate_limit") == REACTION_LIMIT,
             "Si cap cached FE solve/gate contract differs")
    with os.fdopen(os.open(npz_path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        info = os.fstat(stream.fileno())
        _require(stat.S_ISREG(info.st_mode) and info.st_size <= CACHE_LIMIT, "Si cap payload size/type bound")
        observed = hashlib.file_digest(stream, "sha256").hexdigest()
        _require(metadata.get("npz_sha256") == observed, "Si cap cache binding/hash mismatch")
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as archive:
            _require(set(archive.files) == {"stiffness", "load"}, "unexpected Si cap cache arrays")
            for name, expected in (("stiffness", (134, 134)), ("load", (134,))):
                _require(archive.zip.getinfo(name+".npy").file_size <= CACHE_LIMIT,
                         "Si cap cache decompression bound exceeded")
                with archive.zip.open(name+".npy") as header_stream:
                    version = np.lib.format.read_magic(header_stream)
                    _require(version in ((1, 0), (2, 0)), "unsupported Si cap array header")
                    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                              else np.lib.format.read_array_header_2_0)
                    shape, _, dtype = reader(header_stream, max_header_size=4096)
                    _require(shape == expected and dtype == np.dtype("float64"), "Si cap array header contract differs")
            stiffness, load = archive["stiffness"], archive["load"]
    _require(stiffness.shape == (134, 134) and load.shape == (134,)
             and np.isfinite(stiffness).all() and np.isfinite(load).all(), "invalid cached Si operator")
    return stiffness, load, metadata


def _publish_cache(root, key, stiffness, load, metadata):
    token = uuid.uuid4().hex
    pending = root/(key+"."+token+".pending.npz")
    pending_json = root/(key+"."+token+".pending.json")
    try:
        with pending.open("xb") as stream:
            np.savez(stream, stiffness=stiffness, load=load)
            stream.flush()
            os.fsync(stream.fileno())
        metadata = dict(metadata, key=key, npz_sha256=_file_sha(pending))
        with pending_json.open("x") as stream:
            stream.write(_json(metadata)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, root/(key+".npz"))
        os.replace(pending_json, root/(key+".json"))
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return metadata
    finally:
        pending.unlink(missing_ok=True)
        pending_json.unlink(missing_ok=True)


def _segment(root, binding, xs, zs, ports, scale, thickness_um, library, sources, intervals=6):
    from .assembly_operator_library import _linear_operator
    spec = dict(binding, thickness_um=float(thickness_um), thickness_intervals=intervals)
    key = hashlib.sha256(_json(spec).encode()).hexdigest()
    start = time.monotonic()
    published = _published_root()
    ready = _read_cache(published/(key+".npz"), published/(key+".json"), key)
    if ready is not None:
        stiffness, load, metadata = ready
        operator = _linear_operator(stiffness, load, scale, library.operators["bi"].cu_card)
        return (operator, thickness_um/1000), dict(metadata, cache_hit=True,
            published_library_hit=True, cache_path=str(published/(key+".npz")),
            elapsed_seconds=time.monotonic()-start)
    root.mkdir(parents=True, exist_ok=True)
    with _lock(root/(key+".lock")):
        cached = _read_cache(root/(key+".npz"), root/(key+".json"), key)
        hit = cached is not None
        if not hit:
            available = next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines()
                             if line.startswith("MemAvailable:"))
            _require(available >= 12*1024**3, "full Si seed requires at least 12 GiB available RAM")
            bbar, material, _ = sources
            k, f, lift, injection, mesh = _assemble(xs, zs, ports, thickness_um, intervals, bbar, material)
            stiffness, load, audits = _condense_fe(k, f, lift, injection, scale)
            metadata = _publish_cache(root, key, stiffness, load, dict(specification=spec,
                mesh=mesh, audits=audits, build_seconds=time.monotonic()-start,
                approximation_scope="full_periodic_interior_FE_actual_restricted_face_basis"))
        else:
            stiffness, load, metadata = cached
    operator = _linear_operator(stiffness, load, scale, library.operators["bi"].cu_card)
    return (operator, thickness_um/1000), dict(metadata, cache_hit=hit,
        published_library_hit=False,
        cache_path=str(root/(key+".npz")), elapsed_seconds=time.monotonic()-start)


def _compose(library, seed, count, residual=None):
    from .assembly_operator_library import _join_segments
    audits = []
    result = residual
    power = seed
    while count:
        if count & 1:
            if result is None:
                result = power
            else:
                result, audit = _join_segments(library, result, power)
                audits.append(audit)
        count >>= 1
        if count:
            power, audit = _join_segments(library, power, power)
            audits.append(audit)
    return result, audits


def augment_w2w_library(library, geometry, physical_trace):
    """Return a new split-cap library and honest experimental FE provenance.

    The original component operators, Cu states, material binding and ledger
    are untouched. Analytic ABD is advisory for the restricted face basis;
    all FE condensation/reaction gates remain mandatory and unmodified.
    """
    from . import assembly_operator_library as adapter, component_assembly_bridge as bridge
    from .assembly_operator_library import _abd_audit, _join_segments
    trace, xs, zs, scale, target, periodic_defect = _validate(library, geometry, physical_trace)
    root = _cache_root()
    sources = _native_sources()
    trace_sha = _hash_arrays(trace)
    construction = dict(version=VERSION, source_sha256=_file_sha(__file__),
        native_sha256=sources[2], numpy_version=np.__version__,
        bridge_sha256=_file_sha(bridge.__file__),
        assembly_helpers_sha256=_file_sha(adapter.__file__))
    construction_sha = hashlib.sha256(_json(construction).encode()).hexdigest()
    # Node IDs and zero bubble padding cannot change a homogeneous Si solve.
    seed_trace_sha = _hash_arrays(dict(face_coordinates_mm=trace["face_coordinates_mm"],
                                      face_lifts=trace["face_lifts"][:, :, :134]))
    binding = dict(trace_sha256=seed_trace_sha, material=MATERIAL, construction_sha256=construction_sha,
        coordinate_contract_sha256=_hash_arrays(dict(scale=scale, reflection=library.reflection_matrix,
                rigid_indices=np.asarray(library.rigid_port_mode_indices))),
        boundary_contract=dict(BOUNDARY_CONTRACT))
    ports = trace["face_lifts"][0, :, 6:70]
    outer, outer_proof = _segment(root, binding, xs, zs, ports, scale, 1.5, library, sources)
    seed, seed_proof = _segment(root, binding, xs, zs, ports, scale, 10., library, sources)
    bulk_um = 775.-target
    count, remainder_um = _bulk_recipe(bulk_um)
    residual, residual_proof = None, None
    if remainder_um is not None:
        residual, residual_proof = _segment(root, binding, xs, zs, ports, scale,
                                          remainder_um, library, sources)
    bulk, composition = _compose(library, seed, count, residual)
    _require(math.isclose(bulk[1]*1000, bulk_um, rel_tol=0, abs_tol=1e-9), "bulk thickness sum differs")
    # A separately integrated thin half exposes the restriction at joined faces.
    half, half_proof = _segment(root, binding, xs, zs, ports, scale, .75, library, sources, intervals=3)
    doubled, thin_join = _join_segments(library, half, half)
    direct_k, joined_k = outer[0].reference_stiffness(25.), doubled[0].reference_stiffness(25.)
    direct_f, joined_f = outer[0].reference_thermal_load(26.), doubled[0].reference_thermal_load(26.)
    thin = dict(method="independent_full_FE_2x0p75um_three_intervals_vs_1p5um_six_intervals",
        scaled_stiffness_relative_change=float(np.linalg.norm(scale[:, None]*(joined_k-direct_k)*scale[None, :]) /
            max(np.linalg.norm(scale[:, None]*direct_k*scale[None, :]), 1e-30)),
        scaled_thermal_relative_change=float(np.linalg.norm(scale*(joined_f-direct_f)) /
            max(np.linalg.norm(scale*direct_f), 1e-30)), join_audit=thin_join,
        status="ADVISORY_FACE_SPACE_RESTRICTION_NOT_CONTINUUM_ADMISSION")
    abd = {name: _abd_audit(library, segment[0], segment[1])
           for name, segment in (("outer_si", outer), ("seed10", seed), ("bulk_si", bulk))}
    for audit in abd.values():
        audit["analytic_one_percent_pass"] = audit["maximum_scaled_abd_defect"] <= .01
        audit["policy"] = "ADVISORY_ACTUAL_FULL_FE_PROJECTED_FACE_SPACE_NO_EXISTING_GATE_CHANGED"
    result = replace(library, operators={**library.operators, "outer_si": outer[0], "bulk_si": bulk[0]},
        thickness_mm={**library.thickness_mm, "outer_si": .0015, "bulk_si": bulk_um/1000})
    operator_sha = _hash_arrays({name+suffix: value for name, segment in (("outer", outer), ("bulk", bulk))
        for suffix, value in (("_k", segment[0].reference_stiffness(25.)),
                              ("_f", segment[0].reference_thermal_load(26.)))})
    provenance = dict(schema="yap.w2w_component_silicon_cap.v1", trace_sha256=trace_sha,
        material=dict(MATERIAL), operator_sha256=operator_sha, construction_sha256=construction_sha,
        construction=construction, bulk_thickness_um=bulk_um, outer_si_thickness_um=1.5,
        seed10_count=count, residual_thickness_um=remainder_um,
        face_max_coefficient_jump=periodic_defect,
        face_jump_contract=_face_jump_audit(ports, len(xs), len(zs), scale[6:70]),
        segments=dict(outer=outer_proof, seed10=seed_proof, residual=residual_proof, independent_half=half_proof),
        composition_audits=composition, independent_thin_check=thin, analytic_abd=abd,
        full_FE_condensation_gate_passed=True, physical_readiness=False,
        process_accuracy_validated=False, per_fractional_remainder_full_FE=True,
        retained_outer_slice_policy="original_birth_reference_and_age_never_redeposited",
        physics_qualification="EXPERIMENTAL_HOMOGENEOUS_SI_CAP_RESTRICTED_NATIVE64_FACES_NOT_PROCESS_VALIDATION")
    return result, provenance
