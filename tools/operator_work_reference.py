"""Offline extraction of native reactions into work-conjugate responses.

For public ROM q6=(eps_x, eps_y, gamma_xy, kappa_x, kappa_y, kappa_xy),
native physical coordinates are (ex, ez, gamma_xz, kx, kz, kappaxz).
The thickness-strain pilot ey is NOT a q6 coordinate. With the native macro
origin at stack-global y=0 and layer-local faces at y=0,h, c=0.0115+h/2 mm:

    Fq = Jpilot.T Rpilot + Lbottom.T Rbottom + Ltop.T Rtop.

Jpilot maps ex0=ex-c*kx, ez0=ez-c*kz and gamma0=gamma-c*kappaxz.
Thus only pilot curvature forces are shifted by -c times their membrane
force. Archived original face lifts are already centered and must not be
shifted again. Engineering shear/twist introduces no additional factor two.
Reactions retain their native sign; no eliminated/slave reactions are added.
The caller must independently establish that the native CE skips prescribed
faces and that pilot/face exports form disjoint prescribed reaction channels.

Outputs retain the ROM's generalized virtual-work scale (first three q6
components N mm, last three N mm^2); no area normalization is applied here.
Divide q6 by cell_area_mm2 externally for membrane/bending resultants. Port
forces are N for the original dimensionless modes and mm port coordinates.
This extractor validates data, not operator accuracy or solver admission.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np


Q6 = ("eps_x", "eps_y", "gamma_xy", "kappa_x", "kappa_y", "kappa_xy")
NATIVE_Q6 = ("ex", "ez", "gamma_xz", "kx", "kz", "kappaxz")
PILOT_COLUMNS = ("rf_p_ex", "rf_p_ez", "rf_p_gamma_xz",
                 "rf_p_kx", "rf_p_kz", "rf_p_kappa_xz")
REACTION_COLUMNS = ("rf_x_N", "rf_y_N", "rf_z_N")
SCHEMA = "yap.operator_work_reference.v1"
TIME_ATOL_S = 1e-8
GEOMETRY_ATOL_MM = 1e-12
MAX_CSV_LINE_BYTES = 65536


@dataclass(frozen=True)
class WorkReference:
    macro_response: np.ndarray
    pilot_contribution: np.ndarray
    bottom_contribution: np.ndarray
    top_contribution: np.ndarray
    bottom_ports: np.ndarray | None
    top_ports: np.ndarray | None
    coverage: dict[str, Any]

    @property
    def face_contribution(self) -> np.ndarray:
        return self.bottom_contribution + self.top_contribution

    @property
    def reference(self) -> np.ndarray:
        if self.bottom_ports is None:
            return self.macro_response
        return np.concatenate((self.macro_response, self.bottom_ports, self.top_ports), axis=1)

    @property
    def metadata(self) -> dict[str, Any]:
        return self.coverage


def _identity(stat: Any) -> dict[str, int]:
    return dict(device=stat.st_dev, inode=stat.st_ino, bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


class _CSVStream:
    """Hash the actual consumed bytes, and reject path replacement or drift."""

    def __init__(self, path: str | Path, required: tuple[str, ...], maximum_rows: int):
        self.path = Path(path)
        self.required = required
        self.maximum_rows = maximum_rows
        self.sha = hashlib.sha256()

    def __enter__(self):
        self.before = _identity(self.path.stat())
        self.stream = self.path.open("rb")
        try:
            self.reader = csv.DictReader(self._lines())
            names = self.reader.fieldnames
            if (not names or len(names) != len(set(names))
                    or not set(self.required).issubset(names)):
                raise ValueError(f"{self.path.name}: invalid or missing CSV columns")
        except BaseException:
            self.stream.close()
            raise
        return self

    def _lines(self):
        lines = 0
        while True:
            raw = self.stream.readline(MAX_CSV_LINE_BYTES + 1)
            if not raw:
                return
            lines += 1
            if len(raw) > MAX_CSV_LINE_BYTES or lines > 2 * self.maximum_rows + 1:
                raise ValueError(f"{self.path.name}: CSV line resource bound exceeded")
            self.sha.update(raw)
            yield raw.decode("utf-8")

    def __iter__(self):
        for count, row in enumerate(self.reader, start=1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{self.path.name}: malformed CSV row {self.reader.line_num}")
            if count > self.maximum_rows:
                raise ValueError(f"{self.path.name}: CSV row resource bound exceeded")
            yield row

    def __exit__(self, kind, value, traceback):
        self.stream.close()
        if kind is None:
            if self.before != _identity(self.path.stat()):
                raise ValueError(f"{self.path.name}: input changed during extraction")
            self.record = dict(path=str(self.path.resolve()), sha256=self.sha.hexdigest(),
                               **self.before)


def _archive(path: str | Path, keys: tuple[str, ...]):
    path = Path(path)
    before = _identity(path.stat())
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    with np.load(path, allow_pickle=False) as saved:
        if not set(keys).issubset(saved.files):
            raise ValueError(f"{path.name}: missing archive fields")
        arrays = {key: saved[key] for key in keys}
    if before != _identity(path.stat()):
        raise ValueError(f"{path.name}: input changed during extraction")
    return arrays, dict(path=str(path.resolve()), sha256=sha.hexdigest(), **before)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"invalid integer {name}")
    try:
        try:
            result = int(value)
        except ValueError:
            number = float(value)
            if not math.isfinite(number) or abs(number) > 2**53:
                raise ValueError
            result = int(number)
        if isinstance(value, str) and value.strip() == str(result):
            return result
        number = float(value)
        if not math.isfinite(number) or number != result or abs(number) > 2**53:
            raise ValueError
        return result
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"invalid integer {name}") from error


def _finite_real(array: np.ndarray) -> bool:
    return ((np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating))
            and bool(np.isfinite(array).all()))


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"invalid numeric {name}") from error
    if not math.isfinite(result):
        raise ValueError(f"nonfinite {name}")
    return result


def _check_time(row: dict[str, str], state: int, times: np.ndarray):
    time = _number(row["time_s"], "time_s")
    if abs(time - times[state - 1]) > TIME_ATOL_S:
        raise ValueError(f"state {state}: time differs from expected clock")


def _load_traces(lift_npz, modes_npz, h, area):
    data, lift_identity = _archive(lift_npz, ("node_ids", "coordinates_mm", "lift_fields",
        "input_labels", "bottom_node_indices", "top_node_indices"))
    ids, xyz, fields = (data[key] for key in ("node_ids", "coordinates_mm", "lift_fields"))
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids <= 0) or len(np.unique(ids)) != len(ids)):
        raise ValueError("invalid or duplicate lift node_ids")
    count = len(ids)
    labels = data["input_labels"]
    if (xyz.shape != (count, 3) or fields.ndim != 3
            or fields.shape[:2] != (count, 3) or fields.shape[2] not in (6, 134)
            or labels.shape != (fields.shape[2],) or tuple(map(str, labels[:6])) != Q6
            or not _finite_real(xyz) or not _finite_real(fields)):
        raise ValueError("invalid lift dimensions, finite values or q6 labels")
    if fields.shape[2] == 134:
        port_labels = tuple(f"{face}_mode_{i:02d}" for face in ("bottom", "top") for i in range(64))
        if tuple(map(str, labels[6:])) != port_labels:
            raise ValueError("invalid lift port labels")
    if count == 0 or not np.isclose(np.ptp(xyz[:, 1]), h, atol=GEOMETRY_ATOL_MM, rtol=0):
        raise ValueError("lift geometry thickness differs")
    if not np.isclose(np.ptp(xyz[:, 0]) * np.ptp(xyz[:, 2]), area,
                      atol=1e-18, rtol=1e-10):
        raise ValueError("lift geometry cell area differs")
    indices, traces = [], []
    for face, y in (("bottom", xyz[:, 1].min()), ("top", xyz[:, 1].max())):
        selected = data[face + "_node_indices"]
        if (selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer)
                or len(selected) == 0 or np.any(selected < 0) or np.any(selected >= count)
                or len(np.unique(selected)) != len(selected)):
            raise ValueError(f"invalid or duplicate {face} face indices")
        actual = np.flatnonzero(np.isclose(xyz[:, 1], y, atol=GEOMETRY_ATOL_MM, rtol=0))
        if not np.array_equal(np.sort(selected), actual):
            raise ValueError(f"{face} indices do not cover the complete geometric face")
        indices.append(selected)
        traces.append(fields[selected, :, :6].copy())
    if np.intersect1d(indices[0], indices[1]).size:
        raise ValueError("bottom/top overlap would double count reactions")
    port_maps = None
    mode_identity = None
    if modes_npz is not None:
        modes, mode_identity = _archive(modes_npz, ("coords_mm", "modes"))
        coordinates, values = modes["coords_mm"], modes["modes"]
        size = len(indices[0])
        if (len(indices[1]) != size or coordinates.shape != (size, 2)
                or values.shape != (size * 3, 64) or not _finite_real(coordinates)
                or not _finite_real(values) or len(np.unique(coordinates, axis=0)) != size):
            raise ValueError("invalid port mode dimensions, coordinates or finite values")
        order = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
        coordinates = coordinates[order]
        values = values.reshape(size, 3, 64)[order]
        port_maps = []
        for face, selected in enumerate(indices):
            xyz_order = np.lexsort((xyz[selected, 2], xyz[selected, 0]))
            if not np.allclose(xyz[selected[xyz_order]][:, (0, 2)], coordinates,
                               atol=GEOMETRY_ATOL_MM, rtol=0):
                raise ValueError("port coordinates differ from lift faces")
            mapped = np.empty_like(values)
            mapped[xyz_order] = values
            if fields.shape[2] == 134:
                own = fields[selected, :, 6 + 64 * face:70 + 64 * face]
                other = fields[selected, :, 70:134] if face == 0 else fields[selected, :, 6:70]
                if not np.allclose(own, mapped, atol=1e-10, rtol=0) or np.any(other != 0):
                    raise ValueError("original lift port traces differ or cross faces")
            port_maps.append(mapped)
    return ids, indices, traces, port_maps, lift_identity, mode_identity


def extract_work_reference(
    *, lift_npz: str | Path, pilot_csv: str | Path, face_csv: str | Path,
    expected_times_s, layer_thickness_mm: float, cell_area_mm2: float,
    maximum_result_set: int | None = None, modes_npz: str | Path | None = None,
) -> WorkReference:
    """Return q6, optionally full134, without materializing the face history.

    Pass the complete native clock even when requesting a shorter prefix.
    Both CSVs are consumed through EOF and hashed. Pilot states must be
    strictly consecutive from 1; face states must be nondecreasing and
    consecutive from 1. Higher states within the supplied clock are a suffix:
    their syntax, identities and clock (when present) are validated, but face
    node completeness/uniqueness is required only for the requested prefix.
    No early break can hide a duplicate or lower state after the prefix.
    The native face CSV has no time column; its clock is joined through the
    independently validated pilot state. If time_s exists it is checked too.
    """

    times = np.asarray(expected_times_s, dtype=float)
    if (times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all()
            or np.any(times < 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("invalid expected time series")
    n = len(times) if maximum_result_set is None else _integer(maximum_result_set, "maximum_result_set")
    if not 1 <= n <= len(times):
        raise ValueError("maximum_result_set outside expected time series")
    h = _number(layer_thickness_mm, "layer_thickness_mm")
    area = _number(cell_area_mm2, "cell_area_mm2")
    if h <= 0 or area <= 0:
        raise ValueError("thickness and area must be positive")
    ids, indices, traces, modes, lift_identity, mode_identity = _load_traces(lift_npz, modes_npz, h, area)
    pilot = np.zeros((n, 6))
    center = .0115 + .5 * h
    last_pilot = 0
    with _CSVStream(pilot_csv, ("result_set", "time_s", *PILOT_COLUMNS), len(times)) as source:
        for row in source:
            state = _integer(row["result_set"], "result_set")
            if state != last_pilot + 1 or state > len(times):
                raise ValueError("duplicate, unexpected or unsorted pilot state")
            _check_time(row, state, times)
            values = np.asarray([_number(row[key], key) for key in PILOT_COLUMNS])
            if "rf_p_ey" in row:
                _number(row["rf_p_ey"], "rf_p_ey")
            if state <= n:
                values[3:] -= center * values[:3]
                pilot[state - 1] = values
            last_pilot = state
    pilot_identity = source.record
    if last_pilot < n:
        raise ValueError("missing requested pilot states")
    lookup = {face: {int(ids[index]): j for j, index in enumerate(selected)}
              for face, selected in zip(("bottom", "top"), indices)}
    reactions = [np.zeros((len(selected), 3)) for selected in indices]
    seen = [np.zeros(len(selected), dtype=bool) for selected in indices]
    contributions = [np.zeros((n, 6)), np.zeros((n, 6))]
    ports = [np.zeros((n, 64)), np.zeros((n, 64))] if modes is not None else [None, None]

    def finish(state):
        if not 1 <= state <= n:
            return
        for face in range(2):
            if not seen[face].all():
                raise ValueError(f"state {state}: missing requested face nodes")
            contributions[face][state - 1] = np.einsum("ndq,nd->q", traces[face], reactions[face])
            if modes is not None:
                ports[face][state - 1] = np.einsum("ndq,nd->q", modes[face], reactions[face])
            seen[face].fill(False)

    current, rows_read, requested_rows = 0, 0, 0
    maximum_face_rows = len(times) * sum(len(selected) for selected in indices)
    with _CSVStream(face_csv, ("result_set", "face", "node_id", *REACTION_COLUMNS), maximum_face_rows) as source:
        has_face_times = "time_s" in source.reader.fieldnames
        for row in source:
            state = _integer(row["result_set"], "result_set")
            if state < current or state > current + 1 or not 1 <= state <= last_pilot:
                raise ValueError("unexpected or unsorted face state")
            if state != current:
                finish(current)
                current = state
            name = row["face"]
            if name not in lookup:
                raise ValueError("unexpected face")
            node = _integer(row["node_id"], "node_id")
            if node not in lookup[name]:
                raise ValueError("unexpected node on reference face")
            if has_face_times:
                _check_time(row, state, times)
            values = tuple(_number(row[key], key) for key in REACTION_COLUMNS)
            if state <= n:
                face = 0 if name == "bottom" else 1
                index = lookup[name][node]
                if seen[face][index]:
                    raise ValueError("duplicate reference face/state/node")
                seen[face][index] = True
                reactions[face][index] = values
                requested_rows += 1
            rows_read += 1
        finish(current)
    face_identity = source.record
    if current < n:
        raise ValueError("missing requested face states")
    macro = pilot + contributions[0] + contributions[1]
    if not all(np.isfinite(value).all() for value in (macro, pilot, *contributions, *[p for p in ports if p is not None])):
        raise ValueError("nonfinite integrated work response")
    inputs = dict(lift=lift_identity, pilot_csv=pilot_identity, face_csv=face_identity)
    if mode_identity is not None:
        inputs["modes"] = mode_identity
    coverage = dict(schema=SCHEMA, reference_method="centered_pilot_plus_original_fixed_face_trace_work",
        macro_labels=list(Q6), native_macro_order=list(NATIVE_Q6), states=n,
        pilot_states_scanned=last_pilot, face_states_scanned=current,
        face_rows_scanned=rows_read, requested_face_rows=requested_rows,
        expected_face_rows=n * sum(len(selected) for selected in indices),
        bottom_node_ids=ids[indices[0]].tolist(), top_node_ids=ids[indices[1]].tolist(),
        expected_times_s=times[:n].tolist(), pilot_center_mm=center,
        expected_clock_states=len(times),
        expected_clock_sha256=hashlib.sha256(times.astype("<f8", copy=False).tobytes()).hexdigest(),
        csv_maximum_face_rows=maximum_face_rows, csv_maximum_line_bytes=MAX_CSV_LINE_BYTES,
        layer_thickness_mm=h, cell_area_mm2=area, area_normalization_applied=False,
        native_reaction_sign_preserved=True, thickness_ey_excluded=True,
        face_time_source="validated_face_time_s_and_pilot_state" if has_face_times else "validated_pilot_state_clock",
        suffix_rule="whole_csv_scanned_sorted_consecutive_states_within_expected_clock; face_coverage_required_only_for_prefix",
        engineering_shear_factor=1., generalized_units=["N mm"] * 3 + ["N mm^2"] * 3,
        port_units="N" if modes is not None else None, inputs=inputs,
        operator_pass_granted=False, thresholds_changed=False,
        requires_independent_CE_sign_and_work_calibration=True)
    return WorkReference(macro, pilot, contributions[0], contributions[1], ports[0], ports[1], coverage)
