"""Portable, activation-gated endpoint thickness families. NumPy-only runtime.

All numerical operands come from an immutable distribution. Evaluation is
Chebyshev/Laurent algebra, never FE assembly, factorization, POD or state
transfer. A held-out single-BI bow pass is not assembly qualification.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
from types import MappingProxyType
import uuid
import zipfile

import numpy as np


SCHEMA = "yap.endpoint_thickness_family.v1"
ACTIVATION_SCHEMA = "yap.endpoint_thickness_activation.v1"
MAX_ARTIFACT_BYTES = 4 * 1024**3
MAX_JSON_BYTES = 2 * 1024**2
FIELDS = ("kaa", "c", "kff", "fm", "ff", "B_macro", "B_free", "weights", "macro_map")
_IMPORTED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class ThicknessOperatorError(ValueError):
    pass


class PendingValidationError(ThicknessOperatorError):
    """The family has not been activated by its independent bow assessment."""


class IntegrityError(ThicknessOperatorError):
    pass


def _require(value, message, error=IntegrityError):
    if not value:
        raise error(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def _sha(value):
    _require(isinstance(value, str) and len(value) == 64
             and all(c in "0123456789abcdef" for c in value), "invalid SHA256 binding")
    return value


def _identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _path(root, relative):
    value = Path(relative)
    _require(not value.is_absolute() and value.parts and not set(value.parts).intersection(
        {"..", ".snapshot", "#snapshot"}), "distribution paths must be relative and nonsnapshot")
    path = root / value
    _require(path.resolve() == path and path.is_relative_to(root), "distribution alias refused")
    return path


def _read(path, *, expected=None, cap=MAX_JSON_BYTES, readonly=True):
    _require(path.resolve() == path and not {"#snapshot", ".snapshot"}.intersection(path.parts),
             "literal nonsnapshot file required")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        _require(0 < before.st_size <= cap and (not readonly or not before.st_mode & 0o222),
                 "bounded read-only distribution file required")
        digest = hashlib.sha256()
        while block := stream.read(1024**2):
            digest.update(block)
        _require(_identity(before) == _identity(os.fstat(stream.fileno())) == _identity(path.stat()),
                 "file changed during verification")
    actual = digest.hexdigest()
    _require(expected is None or actual == _sha(expected), "SHA256 mismatch: " + str(path))
    return actual, before.st_size


def _json(path, expected=None):
    sha, size = _read(path, expected=expected)
    raw = path.read_bytes()
    _require(len(raw) == size and hashlib.sha256(raw).hexdigest() == sha, "JSON changed")
    return json.loads(raw), sha


def _arrays(path, *, expected=None, size=None, readonly=True):
    before = path.stat()
    _, observed_size = _read(path, expected=expected, cap=MAX_ARTIFACT_BYTES, readonly=readonly)
    _require(size is None or observed_size == size, "artifact size mismatch")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        _require(0 < len(members) <= 128 and len({v.filename for v in members}) == len(members)
                 and all("/" not in v.filename and v.filename.endswith(".npy") for v in members)
                 and sum(v.file_size for v in members) <= MAX_ARTIFACT_BYTES,
                 "invalid bounded numeric archive")
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    for value in values.values():
        _require(np.issubdtype(value.dtype, np.number) and np.isfinite(value).all(),
                 "finite numeric arrays required")
        value.setflags(write=False)
    _require(_identity(before) == _identity(path.stat()), "numeric artifact changed during load")
    return values


def _number(value, name):
    _require(type(value) in (int, float) and math.isfinite(value),
             name + " must be a finite number", ThicknessOperatorError)
    return float(value)


def _endpoint_key(value):
    return str(int(value)) if value.is_integer() else repr(value)


def _immutable(values):
    for array in values.values():
        array.setflags(write=False)
    return MappingProxyType(values)


@dataclass(frozen=True)
class ThicknessResolution:
    family_id: str
    si_um: float
    single_bi_total_um: float
    prepared_total_um: float
    request_kind: str


@dataclass(frozen=True)
class ThicknessEvaluation:
    resolution: ThicknessResolution
    cache_key: str
    arrays: object
    capabilities: object
    lineage: object

    def component_arrays(self):
        """NPZ-style CSR keys for the existing component runtime's future hook."""
        values = dict(self.arrays)
        matrix = values.pop("kff")
        rank = len(matrix)
        values.update(kff_data=matrix.ravel().copy(),
            kff_indices=np.tile(np.arange(rank, dtype=np.int32), rank),
            kff_indptr=np.arange(rank+1, dtype=np.int32)*rank,
            kff_shape=np.array([rank, rank], dtype=np.int64))
        return _immutable(values)


class ThicknessOperatorFamily:
    """One immutable family, bounded memory LRU and optional explicit disk cache.

    Opening/metadata inspection is permitted while validation is pending.
    resolve/evaluate/geometry/components refuse until ACTIVATE.json binds PASS.
    """
    def __init__(self, distribution, *, cache_dir=None, memory_cache_bytes=2*1024**3,
                 disk_cache_bytes=8*1024**3):
        self.root = Path(distribution).absolute()
        _require(self.root.resolve() == self.root and self.root.is_dir(), "literal family directory required")
        self._manifest, self.manifest_sha256 = _json(self.root / "FAMILY.json")
        m = self._manifest
        bracket = m.get("bracket_um")
        _require(m.get("schema") == SCHEMA and isinstance(bracket, list) and len(bracket) == 2,
                 "two endpoint bracket numbers required")
        lo, hi = (_number(value, "endpoint bracket") for value in bracket)
        self.half_span_um = (hi-lo)/2.
        self.center_um = lo+self.half_span_um
        _require(0 < lo < hi and self.half_span_um > 0
                 and math.isfinite(self.center_um)
                 and type(m.get("center_um")) in (int, float)
                 and type(m.get("half_span_um")) in (int, float)
                 and m["center_um"] == self.center_um and m["half_span_um"] == self.half_span_um,
                 "positive finite bracket with exactly derived center/half-span required")
        self.bracket_um = (lo, hi)
        _require(m.get("geometry_sha256") == _digest(m["geometry"])
                 and m.get("material_sha256") == _digest(m["material"]), "geometry/material binding differs")
        _require(set(m["endpoints"]) == {_endpoint_key(lo), _endpoint_key(hi)},
                 "endpoint keys must be canonical bracket numbers")
        for binding in m["endpoints"].values():
            _sha(binding["operator_sha256"])
            _sha(binding["reference_sha256"])
        _sha(m["constructor_sha256"])
        _sha(m["candidate_sha256"])
        _require(m["coefficients"]["fields"] == {
            "kaa":6, "c":4, "kff":2, "fm":4, "ff":2,
            "B_macro":3, "B_free":1, "weights":1, "macro_map":1}, "unreviewed polynomial recipe")
        self.runtime_sha256 = _IMPORTED_SOURCE_SHA256
        self.bindings, self.bindings_sha256 = _json(self.root / "RUNTIME_BINDINGS.json",
                                                   m["runtime_bindings_sha256"])
        _require(self.bindings.get("runtime_source_sha256") == self.runtime_sha256
                 and self.bindings.get("runtime_imports") == ["numpy", "stdlib"]
                 and self.bindings.get("FE_generator_on_runtime_path") is False,
                 "runtime source/resource binding differs")
        for value in (memory_cache_bytes, disk_cache_bytes):
            _require(type(value) is int and value >= 0, "nonnegative cache budget required")
        self.memory_cache_bytes, self.disk_cache_bytes = memory_cache_bytes, disk_cache_bytes
        self.cache_dir = None if cache_dir is None else Path(cache_dir).absolute()
        if self.cache_dir is not None:
            _require(self.cache_dir.resolve() == self.cache_dir and not set(self.cache_dir.parts).intersection(
                {"#snapshot", ".snapshot"}) and not self.cache_dir.is_relative_to(self.root),
                "explicit writable cache must be outside immutable distribution")
        self._coefficients = self._static = self._geometry_coefficients = None
        self._memory, self._used = OrderedDict(), 0
        self._lock = threading.RLock()
        self._statistics = dict(memory_hits=0, disk_hits=0, evaluations=0, disk_writes=0,
                                disk_budget_skips=0, disk_busy_skips=0)

    @classmethod
    def from_directory(cls, distribution, **kwargs):
        return cls(distribution, **kwargs)

    @property
    def manifest(self):
        return json.loads(_canonical(self._manifest))

    def metadata(self):
        active = (self.root / "ACTIVATE.json").is_file()
        return dict(family_id=self.manifest["id"], bracket_um=list(self.bracket_um),
            activation="AVAILABLE_FOR_VERIFICATION" if active else "PENDING_INDEPENDENT_BOW",
            single_bi_offset_um=11.5, prepared_dual_bi_offset_um=13.,
            capabilities=self._capabilities(), cache=self.cache_info())

    @staticmethod
    def _capabilities():
        return dict(experimental_thickness_interpolation=True,
            heldout_validation_scope="single_BI_native_FE_graded73_Si32",
            validated_at_every_thickness=False, experimental_prepared_d2w=True,
            second_BI_assembly_accuracy=False, prepared_assembly_accuracy=False,
            W2W_thinning=False, aged_state_transfer=False, FE_generation=False,
            FE_solves=False, target_POD=False, maximum_supported_micro_rank=48,
            rank_gt32_runtime_strategy="original_SparseModel_JVP_reference_preconditioner",
            legacy_local_cached_tangent_compatible=False)

    def _gate(self):
        path = self.root / "ACTIVATE.json"
        if not path.is_file():
            raise PendingValidationError("Thickness family pending independent native73 bow PASS: "
                                         + self.manifest["id"])
        activation, activation_sha = _json(path)
        _require({"assessment", "native_history", "reduced_history", "build_verification"}
                 <= activation.keys(), "activation proof chain incomplete")
        _require(activation.get("schema") == ACTIVATION_SCHEMA
                 and activation.get("family_sha256") == self.manifest_sha256
                 and activation.get("candidate_sha256") == self.manifest["candidate_sha256"]
                 and activation.get("state") == "INDEPENDENT_HELDOUT32_GRADED73_BOW_PASS",
                 "activation does not bind this frozen family")
        record = activation["assessment"]
        assessment, _ = _json(_path(self.root, record["path"]), record["sha256"])
        _require(assessment.get("passed") is True and assessment.get("failed_observations") == 0
                 and assessment.get("matched_states") == 73 and assessment.get("observations") == 292
                 and assessment.get("strict_pointwise_OR") is True
                 and assessment.get("spatial_only") is True,
                 "independent strict pointwise bow assessment has not passed")
        verification = activation["build_verification"]
        build, _ = _json(_path(self.root,verification["path"]),verification["sha256"])
        _require(build.get("candidate32_pack_equivalence_passed") is True
                 and build.get("family",{}).get("sha256") == self.manifest_sha256,
                 "runtime pack does not reproduce the frozen validated candidate")
        histories = {}
        for mode in ("native", "reduced"):
            row = activation[mode + "_history"]
            history, _ = _json(_path(self.root, row["path"]), row["sha256"])
            _require(history.get("states") == 73 and history.get("mode") == mode
                     and history.get("candidate", {}).get("sha256") == self.manifest["constructor_sha256"]
                     and history.get("context", {}).get("constructor_payload", {}).get("sha256")
                         == self.manifest["candidate_sha256"], "history/candidate activation lineage differs")
            histories[mode] = history["context"]
        for key in ("Cu_source", "acceleration_source", "thermal_policy", "selected_original_ordinals"):
            _require(histories["native"][key] == histories["reduced"][key], "history material/clock differs")
        _require(histories["native"].get("no_PORT64_or_POD") is True
                 and histories["native"].get("full_native_interior_coordinates") == 273372
                 and histories["native"].get("Cu_GP") == 185696
                 and histories["native"].get("Cu_state_adoption") is False
                 and histories["reduced"].get("Cu_state_adoption") is False,
                 "independent complete virgin native reference required")
        expected_clock = [0]+[360*c+j for c in range(3) for j in
            (*range(20,161,20), *(160+i for i in (1,2,4,8,10,20,30,40)), *range(220,361,20))]
        truth = histories["native"]
        _require(truth["selected_original_ordinals"] == expected_clock
                 and truth["Cu_source"] == self.manifest["material"]["Cu_source"]
                 and truth["acceleration_source"] == self.manifest["material"]["acceleration_source"]
                 and truth["thermal_policy"].get("primary_tolerance") == 1e-8
                 and truth["thermal_policy"].get("linear_tolerance") == 1e-9,
                 "activation clock, material or strict numerical gates differ")
        return activation_sha

    def resolve(self, *, si_um=None, single_bi_total_um=None, prepared_total_um=None):
        self._gate()
        supplied = [("si_um", si_um), ("single_bi_total_um", single_bi_total_um),
                    ("prepared_total_um", prepared_total_um)]
        selected = [(name, value) for name,value in supplied if value is not None]
        _require(len(selected) == 1, "supply exactly one thickness", ThicknessOperatorError)
        kind, incoming = selected[0]
        value = _number(incoming, kind)
        si = value - {"si_um":0., "single_bi_total_um":11.5, "prepared_total_um":13.}[kind]
        _require(self.bracket_um[0] <= si <= self.bracket_um[1],
                 "Si thickness outside endpoint bracket; no extrapolation",
                 ThicknessOperatorError)
        return ThicknessResolution(self.manifest["id"], si, si+11.5, si+13., kind)

    def _key(self, resolution, activation_sha):
        m = self.manifest
        return _digest(dict(schema="yap.thickness_cache.v1", family_sha256=self.manifest_sha256,
            activation_sha256=activation_sha, runtime_source_sha256=self.runtime_sha256,
            numpy_version=np.__version__, geometry_sha256=m["geometry_sha256"],
            material_sha256=m["material_sha256"], endpoints=m["endpoints"],
            coefficients_sha256=m["coefficients"]["sha256"], static_sha256=m["static"]["sha256"],
            si_um_hex=resolution.si_um.hex()))

    def cache_key(self, **request):
        activation = self._gate()
        return self._key(self.resolve(**request), activation)

    def resolve_entry(self, identifier):
        """Metadata adapter for a future exact-catalog-first component fallback."""
        _require(isinstance(identifier,str), "component identifier must be text", ThicknessOperatorError)
        name = identifier.removeprefix("component:")
        match = re.fullmatch(r"BI2_SI([0-9]+(?:[.p][0-9]+)?)",name)
        _require(match is not None, "identifier is not a BI2 numeric-Si family request", ThicknessOperatorError)
        resolution = self.resolve(si_um=float(match.group(1).replace("p",".")))
        activation = self._gate()
        physical = self.manifest["geometry"]
        return dict(schema="yap.parameterized_component_entry.v1",kind="endpoint_thickness_family",
            id="BI2_SI"+format(resolution.si_um,"g"),family_id=resolution.family_id,
            family_directory=str(self.root),family_sha256=self.manifest_sha256,
            cache_key=self._key(resolution,activation),
            geometry=dict(bi_pitch_um=physical["BI_pitch_um"],diameter_pitch_ratio=physical["diameter_pitch_ratio"],
                bi_thickness_um=physical["BI_um"],beol_thickness_um=physical["BEOL_um"],
                si_tsv_thickness_um=resolution.si_um,total_thickness_um=resolution.single_bi_total_um,
                bi_layer_count=1,tsv_pitch_um=physical["TSV_pitch_um"],cell_side_um=physical["cell_xz_um"][0]),
            experimental_prepared_total_um=resolution.prepared_total_um,
            validation=dict(activation_sha256=activation,
                scope="single_BI_native_FE_graded73_Si32",prepared_assembly_accuracy=False,
                every_interpolated_thickness_independently_validated=False),
            capabilities=self._capabilities(),
            runtime_strategy="original_SparseModel_JVP_reference_preconditioner")

    def _artifact(self, name):
        row = self.manifest[name]
        return _arrays(_path(self.root, row["path"]), expected=row["sha256"], size=row["bytes"])

    @staticmethod
    def _check(values):
        rank, gp = len(values["ff"]), len(values["weights"])
        shapes = dict(kaa=(134,134), c=(rank,134), kff=(rank,rank), fm=(134,), ff=(rank,),
            B_macro=(6*gp,134), B_free=(6*gp,rank), weights=(gp,), macro_map=(134,262))
        _require(1 <= rank <= 48 and gp > 0, "invalid reduced/GP census")
        _require(all(name in values and values[name].shape == shape for name,shape in shapes.items()),
                 "evaluated operator shapes differ")
        _require(all(np.isfinite(v).all() for v in values.values()) and np.all(values["weights"] > 0),
                 "nonfinite operator or nonpositive GP integration")
        k = values["kff"]
        _require(np.linalg.norm(k-k.T) <= 1e-10*max(np.linalg.norm(k),1e-300)
                 and np.linalg.eigvalsh((k+k.T)*.5).min() > 0, "reduced energy not symmetric positive definite")

    def _compute(self, resolution):
        if self._coefficients is None:
            self._coefficients = self._artifact("coefficients")
            self._static = self._artifact("static")
        y = (resolution.si_um-self.center_um)/self.half_span_um
        x = resolution.si_um/self.center_um
        values = {}
        for name, degree in self.manifest["coefficients"]["fields"].items():
            coefficients = self._coefficients[name]
            _require(len(coefficients) == degree+1, "polynomial degree mismatch")
            values[name] = np.polynomial.chebyshev.chebval(y, coefficients)
            if name not in ("weights", "macro_map"):
                values[name] /= x
        for name in ("reference_cu_stiffness", "reference_cu_alpha", "reference_temperature_c",
                     "reference_materials", "macro_offset"):
            values[name] = self._static[name]
        self._check(values)
        self._statistics["evaluations"] += 1
        return _immutable(values)

    def _disk_read(self, key):
        if self.cache_dir is None:
            return None
        path = self.cache_dir / (key + ".npz")
        if not path.is_file():
            return None
        values = _arrays(path, readonly=True)
        _require("_cache_metadata" in values, "cache metadata missing")
        metadata = json.loads(values.pop("_cache_metadata").tobytes().decode("ascii"))
        _require(metadata.get("key") == key and set(metadata["arrays"]) == set(values), "cache binding mismatch")
        for name,value in values.items():
            _require(hashlib.sha256(value.tobytes(order="C")).hexdigest() == metadata["arrays"][name],
                     "cached numeric content differs")
        self._check(values)
        self._statistics["disk_hits"] += 1
        return _immutable(values)

    def _disk_write(self, key, values):
        if self.cache_dir is None or not self.disk_cache_bytes:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lock = self.cache_dir / ".thickness-quota.lock"
        try:
            lock.mkdir()
        except FileExistsError:
            self._statistics["disk_busy_skips"] += 1
            return
        identity = _identity(lock.stat())
        try:
            self._disk_write_locked(key, values)
        finally:
            _require(_identity(lock.stat()) == identity, "cache quota lock identity changed")
            lock.rmdir()

    def _disk_write_locked(self, key, values):
        # The non-overwriting NFS directory lock serializes quota accounting
        # across processes, not only threads using this family object.
        used = 0
        with os.scandir(self.cache_dir) as entries:
            for count,entry in enumerate(entries):
                _require(count < 1024, "bounded direct cache inventory exceeded")
                if entry.is_file(follow_symlinks=False):
                    used += entry.stat(follow_symlinks=False).st_size
        required = sum(v.nbytes for v in values.values()) + 65536
        if used + required > self.disk_cache_bytes:
            self._statistics["disk_budget_skips"] += 1
            return
        metadata = dict(key=key, arrays={name:hashlib.sha256(v.tobytes(order="C")).hexdigest()
                                        for name,v in values.items()})
        target = self.cache_dir / (key + ".npz")
        temporary = self.cache_dir / (key + "." + uuid.uuid4().hex + ".pending")
        with temporary.open("xb") as stream:
            np.savez(stream, **values, _cache_metadata=np.frombuffer(_canonical(metadata).encode("ascii"),dtype=np.uint8))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        published = False
        try:
            os.link(temporary, target)
            published = True
        except FileExistsError:
            pass
        # Readback precedes releasing the sole disposable staging name.
        self._disk_read(key)
        temporary.unlink()
        self._statistics["disk_writes"] += int(published)

    def evaluate(self, **request):
        activation = self._gate()
        resolution = self.resolve(**request)
        key = self._key(resolution, activation)
        with self._lock:
            if key in self._memory:
                values = self._memory.pop(key)
                self._memory[key] = values
                self._statistics["memory_hits"] += 1
            else:
                values = self._disk_read(key)
                if values is None:
                    values = self._compute(resolution)
                    self._disk_write(key, values)
                required = sum(v.nbytes for v in values.values())
                if required <= self.memory_cache_bytes:
                    while self._memory and self._used+required > self.memory_cache_bytes:
                        _, expired = self._memory.popitem(last=False)
                        self._used -= sum(v.nbytes for v in expired.values())
                    self._memory[key] = values
                    self._used += required
        return ThicknessEvaluation(resolution, key, values, MappingProxyType(self._capabilities()),
            MappingProxyType(dict(family_sha256=self.manifest_sha256, activation_sha256=activation,
                endpoints=self.manifest["endpoints"], construction="endpoint_union_geometry_reintegration_coefficients")))

    def geometry(self, *, role="si", **request):
        """Affine coordinates and polynomial lifts, never a target FE generator."""
        resolution = self.resolve(**request)
        _require(role in ("bi","beol","si"), "unknown physical layer role", ThicknessOperatorError)
        if role != "si":
            row = self.manifest["fixed_geometry"][role]
            return _immutable(_arrays(_path(self.root,row["path"]),expected=row["sha256"],size=row["bytes"]))
        with self._lock:
            if self._static is None:
                self._static = self._artifact("static")
            if self._geometry_coefficients is None:
                self._geometry_coefficients = self._artifact("geometry_coefficients")
            static = self._static
            xyz = static["coordinates_unit"].copy()
            xyz[:,1] *= resolution.si_um/1000.
            positions = static["Cu_positions_unit"].copy()
            positions[:,1] = .0115 + positions[:,1]*resolution.si_um/1000.
            fields = np.polynomial.chebyshev.chebval((resolution.si_um-self.center_um)/self.half_span_um,
                self._geometry_coefficients["lifts134"])
            values = dict(coordinates_mm=xyz, Cu_positions_mm=positions, lifts134=fields)
            values.update({key:value for key,value in static.items() if key in
                ("V", "native_body_columns", "native_node_indices", "Cu_native_element_ids", "Cu_native_gauss_indices")
                or key.startswith("P_")})
            return _immutable(values)

    def fixed_component(self, role):
        self._gate()
        _require(role in ("bi", "beol"), "fixed role must be bi or beol", ThicknessOperatorError)
        row = self.manifest["fixed_components"][role]
        return _immutable(_arrays(_path(self.root,row["path"]),expected=row["sha256"],size=row["bytes"]))

    def topology(self):
        self._gate()
        return _immutable(self._artifact("topology"))

    def cache_info(self):
        return dict(self._statistics, memory_entries=len(self._memory), memory_bytes=self._used,
                    memory_budget_bytes=self.memory_cache_bytes, disk_budget_bytes=self.disk_cache_bytes,
                    immutable_family_bytes=sum(array.nbytes for group in
                        (self._coefficients,self._static,self._geometry_coefficients) if group is not None
                        for array in group.values()))

    def clear_memory_cache(self):
        with self._lock:
            self._memory.clear()
            self._used = 0


def open_family(distribution, **cache_options):
    return ThicknessOperatorFamily(distribution, **cache_options)


def resolve_entry(identifier, *, distribution, **cache_options):
    return open_family(distribution, **cache_options).resolve_entry(identifier)
