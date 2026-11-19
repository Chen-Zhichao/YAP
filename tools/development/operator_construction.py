"""Geometry-independent native operator algebra, with no I/O or launch authority.

The resource owner authenticates geometry, native FE arrays, displacement-only
training snapshots and GP row order before calling this module. Frozen provider
bytes are checked here, but numerical input provenance is NOT established here.
Training closure is not held-out bow validation or component/YAP admission.
"""
from __future__ import annotations

from collections import namedtuple
import hashlib
from types import MappingProxyType, ModuleType


SOURCE_PINS = MappingProxyType({
    "elastic": "56a6e4d077f5cecd3c4c8e0b97e221deebe29ad413ed8cd3c5086a9c133cb982",
    "span": "92e4c1840674d040ad488c2ba3b428814b1c951bdabbdafbd08b4e92f474ca26",
    "projection": "7be433f72d3a1ce3bddbeb2a36fa13b75b0ee78daf4f0093bdb846e704d0226a",
})
_BINDING_TOKEN = object()
Algebra = namedtuple("Algebra", "elastic span projection source_pins token")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def bind_sources(raw):
    """Compile exact pure providers supplied as bytes; never open their paths."""
    require(type(raw) is dict and set(raw) == set(SOURCE_PINS), "exact three-provider source closure required")
    modules = {}
    for name, pin in SOURCE_PINS.items():
        value = raw[name]
        require(type(value) is bytes and 0 < len(value) <= 256 * 1024
                and hashlib.sha256(value).hexdigest() == pin, "frozen construction source differs: " + name)
        module = ModuleType("_operator_construction_" + name)
        module.__file__ = "<authenticated-" + name + "-" + pin + ">"
        exec(compile(value, module.__file__, "exec", dont_inherit=True), vars(module))
        require(callable(getattr(module, "build", None)), "pure construction provider ABI differs")
        modules[name] = module
    require((modules["elastic"].RTOL, modules["elastic"].SMALL_RTOL, modules["elastic"].RANK_RTOL,
             modules["projection"].CERT_RTOL) == (1e-8, 1e-10, 1e-12, 1e-10),
            "original construction tolerances required")
    return Algebra(modules["elastic"], modules["span"], modules["projection"], SOURCE_PINS, _BINDING_TOKEN)


def training_frame(*, case, native_sha256):
    """Immutable declaration checked by the owner, not a file-authentication proof."""
    require(type(case) is str and bool(case) and type(native_sha256) is str
            and len(native_sha256) == 64 and set(native_sha256) <= set("0123456789abcdef"),
            "explicit native case and full raw-reference SHA required")
    return MappingProxyType(dict(case=case, native_sha256=native_sha256,
        coordinate_order="native_physical_q6_then_body_z", checkpoint_ordinals=(8, 16),
        original_ordinals=(160, 200), Cu_state_adopted=False, cooldown_used_for_training=False))


class Construction:
    """One native elastic -> displacement span -> full GP projection attempt.

    Dimensions come from authenticated operands, not a pitch-specific census.
    This uses the existing rank7..9 construction; a rank6 elastic result is
    returned honestly, but enrichment then stops rather than inventing a mode.
    Only small metadata is copied. Native arrays/maps are never concatenated,
    symmetrized, culled, rescaled or interpreted as Cu history states.
    """

    def __init__(self, np, sparse, algebra, *, case, native_sha256, check):
        require(type(algebra) is Algebra and algebra.token is _BINDING_TOKEN
                and algebra.source_pins == SOURCE_PINS,
                "authenticated construction providers required")
        require(callable(check), "resource/expiry guard supplied by owner required")
        self._frame = training_frame(case=case, native_sha256=native_sha256)
        self._np, self._sparse, self._algebra, self._check = np, sparse, algebra, check
        self._stage = "NEW"
        self._native_body = self._elastic = self._basis = None
        self._reports = {}
        check()

    @property
    def state(self):
        return self._stage

    @property
    def frame(self):
        return self._frame

    def _begin(self, expected, running):
        require(self._stage == expected, "wrong or terminal construction stage; no automatic retry")
        self._stage = running
        try:
            self._check()
        except BaseException:
            self._stage = "FAILED"
            raise

    def elastic(self, arrays):
        self._begin("NEW", "ELASTIC_RUNNING")
        try:
            require(type(arrays) is dict and set(arrays) == self._algebra.elastic.REQUIRED,
                    "elastic stage takes exact static native members, not Cu state")
            require(all(type(value) is self._np.ndarray and not value.flags.writeable for value in arrays.values()),
                    "owner supplies immutable decoded native arrays")
            result = self._algebra.elastic.build(self._np, self._sparse, arrays, self._check)
            self._native_body = {key: arrays[key] for key in self._algebra.span.NATIVE_NAMES}
            self._elastic = result
            self._reports["elastic"] = result["report"]
            self._check()
            self._stage = "ELASTIC_CLOSED"
            return result
        except BaseException:
            self._stage = "FAILED"
            raise

    def enrich(self, coordinates, *, frame):
        self._begin("ELASTIC_CLOSED", "SPAN_RUNNING")
        try:
            require(frame == self._frame, "training native case/reference/frame differs")
            require(type(coordinates) is dict and set(coordinates) == {8, 16}
                    and all(type(key) is int for key in coordinates),
                    "only CP8/16 displacement training is permitted; cooldown is held out")
            require(self._elastic["Qbody7"].shape[1] == 7,
                    "current nonlinear enrichment requires a resolved raw thermal-null mode; rank6 is not supported")
            np = self._np
            qm = np.column_stack((np.eye(6), np.zeros(6)))
            qm.setflags(write=False)
            basis = dict(Q_macro=qm, Q_body=self._elastic["Qbody7"],
                         Kr=self._elastic["Kr"], fr=self._elastic["fr"])
            result = self._algebra.span.build(np, self._sparse, self._algebra.elastic,
                self._native_body, basis, coordinates, self._check)
            self._basis = {key: result[key] for key in self._algebra.projection.BASIS_NAMES}
            self._reports["span"] = result["report"]
            self._native_body = self._elastic = None
            self._check()
            self._stage = "SPAN_CLOSED_TRAINING_ONLY"
            return result
        except BaseException:
            self._stage = "FAILED"
            raise

    def project(self, native_maps, *, expected_gp, block_gp=682):
        self._begin("SPAN_CLOSED_TRAINING_ONLY", "MAPS_RUNNING")
        try:
            require(type(native_maps) is dict and set(native_maps) == {"B_macro", "B_free"},
                    "exact static native GP strain maps required")
            require(type(expected_gp) is int and expected_gp > 0
                    and native_maps["B_macro"].shape == (6 * expected_gp, 6),
                    "owner-authenticated full GP census differs; no inferred truncation")
            arrays = dict(self._basis, **native_maps)
            result = self._algebra.projection.build(self._np, self._sparse, arrays,
                                                   self._check, block_gp=block_gp)
            require(result["report"]["GP_count"] == expected_gp
                    and result["report"]["complete_native_row_sequence_processed"] is True,
                    "all expected native GP rows must be projected")
            self._reports["projection"] = result["report"]
            self._check()
            self._stage = "MAPS_CLOSED_UNVALIDATED"
            return result
        except BaseException:
            self._stage = "FAILED"
            raise

    def evidence(self):
        """Metadata only; no file, input, history or operator admission follows."""
        return dict(state=self._stage, frame=dict(self._frame), source_pins=dict(SOURCE_PINS),
            reports=dict(self._reports), source_bytes_authenticated=True,
            numeric_input_custody_verified_here=False, resource_admission=False,
            Cu_updates=0, Cu_state_adoption=False, cooldown_training_used=False,
            geometry_family_validated=False, bow_accuracy_validated=False,
            component_interface_qualified=False, operator_admission=False)
