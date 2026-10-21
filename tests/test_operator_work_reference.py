"""Independent virtual-work examples and malformed-history rejection."""

import csv
import hashlib
import json
import os

import numpy as np
import pytest

from operator_work_reference import Q6, extract_work_reference
import operator_work_reference as work_reference


PILOT_HEADER = ("result_set", "time_s", "rf_p_ex", "rf_p_kx", "rf_p_ez",
                "rf_p_kz", "rf_p_ey", "rf_p_gamma_xz", "rf_p_kappa_xz")
FACE_HEADER = ("result_set", "face", "node_id", "rf_x_N", "rf_y_N", "rf_z_N")


def write_csv(path, header, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def case(tmp_path):
    # Intentionally shuffled node IDs, face indices, mode coordinates and rows.
    xyz = np.array([[0., 0., 0.], [2., 0., 0.], [0., 0., 3.], [2., 0., 3.],
                    [0., .02, 0.], [2., .02, 0.], [0., .02, 3.], [2., .02, 3.]])
    ids = np.array([31, 13, 42, 7, 82, 69, 91, 54])
    fields = np.zeros((8, 3, 134))
    x, z, y = xyz[:, 0] - 1., xyz[:, 2] - 1.5, xyz[:, 1] - .01
    fields[:, 0, 0] = x
    fields[:, 2, 1] = xyz[:, 2]
    fields[:, 0, 2], fields[:, 2, 2] = .5 * z, .5 * x
    fields[:, 0, 3], fields[:, 1, 3] = y * x, -.5 * x**2
    fields[:, 2, 4], fields[:, 1, 4] = y * z, -.5 * z**2
    fields[:, 0, 5], fields[:, 2, 5] = .5 * y * z, .5 * y * x
    fields[:, 1, 5] = -.5 * x * z
    modes = np.zeros((4, 3, 64))
    modes[:, 0, 0], modes[:, 2, 1], modes[:, 1, 63] = 1., [1., 2., 3., 4.], -2.
    fields[:4, :, 6:70], fields[4:, :, 70:134] = modes, modes
    labels = list(Q6) + [f"{face}_mode_{i:02d}" for face in ("bottom", "top") for i in range(64)]
    lift_data = dict(node_ids=ids, coordinates_mm=xyz, lift_fields=fields,
        input_labels=np.array(labels), bottom_node_indices=np.array([3, 1, 0, 2]),
        top_node_indices=np.array([6, 4, 7, 5]))
    paths = {key: tmp_path / name for key, name in
             (("lift_npz", "lift.npz"), ("pilot_csv", "pilot.csv"),
              ("face_csv", "face.csv"), ("modes_npz", "modes.npz"))}
    np.savez(paths["lift_npz"], **lift_data)
    order = [2, 0, 3, 1]
    np.savez(paths["modes_npz"], coords_mm=xyz[:4][order][:, (0, 2)],
             modes=modes[order].reshape(12, 64))
    pilots = [dict(zip(PILOT_HEADER, [s, 100. * s, 2. * s, 17. * s,
              3. * s, 19. * s, 1e6, 5. * s, 23. * s])) for s in (1, 2, 3)]
    faces = []
    for state in (1, 2, 3):
        for index in (7, 3, 0, 5, 2, 6, 1, 4):
            faces.append(dict(zip(FACE_HEADER, [state, "bottom" if index < 4 else "top",
                ids[index], (index + 1) * state, (-index - 2) * state, (2 * index + 1) * state])))
    write_csv(paths["pilot_csv"], PILOT_HEADER, pilots)
    write_csv(paths["face_csv"], FACE_HEADER, faces)
    return dict(paths=paths, lift_data=lift_data, pilots=pilots, faces=faces,
                kwargs=dict(**paths, expected_times_s=[100., 200., 300.],
                            layer_thickness_mm=.02, cell_area_mm2=6.))


def test_generalized_work_is_native_displacement_derivative(case):
    result = extract_work_reference(**case["kwargs"])
    dq = np.array([.2, -.3, .7, -.4, .9, -.6])
    # Independent finite difference of the scalar native reaction work,
    # including the exact prescribed displacement field, not the extractor's maps.
    def work(q, state):
        ex, ez, gamma, kx, kz, twist = q
        c = .0215
        p = case["pilots"][state - 1]
        value = (p["rf_p_ex"] * (ex - c * kx) + p["rf_p_ez"] * (ez - c * kz)
            + p["rf_p_gamma_xz"] * (gamma - c * twist) + p["rf_p_kx"] * kx
            + p["rf_p_kz"] * kz + p["rf_p_kappa_xz"] * twist)
        xyz = case["lift_data"]["coordinates_mm"]
        ids = case["lift_data"]["node_ids"]
        for row in case["faces"]:
            if row["result_set"] != state:
                continue
            index = list(ids).index(row["node_id"])
            x, y, z = xyz[index] - [1., .01, 1.5]
            ux = ex * x + .5 * gamma * z + kx * y * x + .5 * twist * y * z
            uy = -.5 * kx * x**2 - .5 * kz * z**2 - .5 * twist * x * z
            uz = ez * xyz[index, 2] + .5 * gamma * x + kz * y * z + .5 * twist * y * x
            value += row["rf_x_N"] * ux + row["rf_y_N"] * uy + row["rf_z_N"] * uz
        return value
    for state in (1, 2, 3):
        assert result.macro_response[state - 1] @ dq == pytest.approx(work(dq, state) - work(np.zeros(6), state), abs=1e-12)
    np.testing.assert_allclose(result.pilot_contribution[0], [2., 3., 5., 16.957, 18.9355, 22.8925])
    np.testing.assert_allclose(result.macro_response, result.pilot_contribution + result.face_contribution)
    assert result.reference.shape == (3, 134)
    # Port63: unit downward displacement multiplied by the original -2 mode.
    assert result.bottom_ports[0, 0] == 10.
    assert result.bottom_ports[0, 63] == 28.
    assert result.top_ports[0, 0] == 26.
    assert result.top_ports[0, 63] == 60.
    assert result.bottom_ports[0, 1] == 50.
    assert result.top_ports[0, 1] == 130.
    assert result.coverage["area_normalization_applied"] is False
    assert result.coverage["operator_pass_granted"] is False
    assert result.coverage["requested_face_rows"] == 24
    assert result.coverage["inputs"]["face_csv"]["sha256"] == hashlib.sha256(case["paths"]["face_csv"].read_bytes()).hexdigest()
    json.dumps(result.coverage)


def test_prefix_reads_and_hashes_suffix_without_storing_it(case):
    result = extract_work_reference(**case["kwargs"], maximum_result_set=1)
    assert result.reference.shape == (1, 134)
    assert result.coverage["face_rows_scanned"] == 24
    assert result.coverage["requested_face_rows"] == 8
    assert result.coverage["pilot_states_scanned"] == 3
    assert result.coverage["face_states_scanned"] == 3


def test_q6_only_and_six_column_lift(case):
    args = {key: value for key, value in case["kwargs"].items() if key != "modes_npz"}
    data = case["lift_data"].copy()
    data["lift_fields"] = data["lift_fields"][:, :, :6]
    data["input_labels"] = data["input_labels"][:6]
    np.savez(case["paths"]["lift_npz"], **data)
    result = extract_work_reference(**args)
    assert result.reference.shape == (3, 6)
    assert result.bottom_ports is None and result.top_ports is None


@pytest.mark.parametrize("change,match", [
    ("duplicate", "duplicate"), ("missing", "missing"),
    ("wrong_face", "unexpected face"), ("wrong_node", "unexpected node"),
    ("other_face_node", "unexpected node"), ("nan", "nonfinite"),
    ("fractional_node", "integer"), ("state_gap", "unsorted"),
    ("past_clock", "unexpected"), ("late_duplicate", "unsorted"),
])
def test_bad_face_rows_fail_even_when_prefix_has_already_completed(case, change, match):
    rows = [row.copy() for row in case["faces"]]
    if change == "duplicate":
        rows.insert(1, rows[0].copy())
    elif change == "missing":
        rows.pop(0)
    elif change == "wrong_face":
        rows[0]["face"] = "side"
    elif change == "wrong_node":
        rows[0]["node_id"] = 999
    elif change == "other_face_node":
        rows[0]["face"] = "bottom"
    elif change == "nan":
        rows[0]["rf_y_N"] = "nan"
    elif change == "fractional_node":
        rows[0]["node_id"] = "54.5"
    elif change == "state_gap":
        rows = rows[:8] + rows[16:]
    elif change == "past_clock":
        rows[-1]["result_set"] = 4
    elif change == "late_duplicate":
        rows[-1] = rows[0].copy()
    write_csv(case["paths"]["face_csv"], FACE_HEADER, rows)
    with pytest.raises(ValueError, match=match):
        extract_work_reference(**case["kwargs"], maximum_result_set=1)


@pytest.mark.parametrize("change,match", [
    ("duplicate", "pilot state"), ("missing", "pilot state"),
    ("time", "time differs"), ("suffix_time", "time differs"),
    ("nonfinite", "nonfinite"), ("missing_shear", "CSV columns"),
    ("fractional_state", "integer"), ("past_clock", "pilot state"),
])
def test_pilot_clock_and_order_are_strict(case, change, match):
    rows = [row.copy() for row in case["pilots"]]
    header = PILOT_HEADER
    if change == "duplicate":
        rows.insert(1, rows[0].copy())
    elif change == "missing":
        rows.pop(1)
    elif change == "time":
        rows[0]["time_s"] += 1e-4
    elif change == "suffix_time":
        rows[-1]["time_s"] += 1e-4
    elif change == "nonfinite":
        rows[0]["rf_p_ex"] = "inf"
    elif change == "missing_shear":
        header = PILOT_HEADER[:-1]
        for row in rows:
            row.pop("rf_p_kappa_xz")
    elif change == "fractional_state":
        rows[0]["result_set"] = "1.5"
    elif change == "past_clock":
        rows[-1]["result_set"] = 4
    write_csv(case["paths"]["pilot_csv"], header, rows)
    with pytest.raises(ValueError, match=match):
        extract_work_reference(**case["kwargs"], maximum_result_set=1)


def test_optional_face_clock_is_checked(case):
    rows = [dict(row, time_s=100. * row["result_set"]) for row in case["faces"]]
    write_csv(case["paths"]["face_csv"], (*FACE_HEADER, "time_s"), rows)
    assert extract_work_reference(**case["kwargs"]).coverage["face_time_source"].startswith("validated_face_time_s")
    rows[-1]["time_s"] = 100.
    write_csv(case["paths"]["face_csv"], (*FACE_HEADER, "time_s"), rows)
    with pytest.raises(ValueError, match="time differs"):
        extract_work_reference(**case["kwargs"], maximum_result_set=1)


@pytest.mark.parametrize("change", ["ids", "dimensions", "nonfinite", "labels", "face_duplicate", "face_missing", "face_swapped", "cross_port", "mode_trace"])
def test_invalid_archived_lift_is_not_accepted(case, change):
    data = {key: value.copy() for key, value in case["lift_data"].items()}
    if change == "ids":
        data["node_ids"][0] = data["node_ids"][1]
    elif change == "dimensions":
        data["lift_fields"] = data["lift_fields"][:, :2]
    elif change == "nonfinite":
        data["lift_fields"][0, 0, 0] = np.nan
    elif change == "labels":
        data["input_labels"][[1, 2]] = data["input_labels"][[2, 1]]
    elif change == "face_duplicate":
        data["bottom_node_indices"][0] = data["bottom_node_indices"][1]
    elif change == "face_missing":
        data["bottom_node_indices"] = data["bottom_node_indices"][:-1]
    elif change == "face_swapped":
        data["bottom_node_indices"] = data["top_node_indices"]
    elif change == "cross_port":
        data["lift_fields"][0, 0, 70] = .1
    elif change == "mode_trace":
        data["lift_fields"][0, 0, 6] += .1
    np.savez(case["paths"]["lift_npz"], **data)
    with pytest.raises(ValueError):
        extract_work_reference(**case["kwargs"])


@pytest.mark.parametrize("times", [[100., 100., 300.], [np.nan], [], [[100.]], [-1., 100., 200.]])
def test_invalid_expected_clock(case, times):
    args = dict(case["kwargs"], expected_times_s=times)
    with pytest.raises(ValueError, match="time series"):
        extract_work_reference(**args)


@pytest.mark.parametrize("n", [0, -1, 4, 1.5, True])
def test_invalid_prefix_bound(case, n):
    with pytest.raises(ValueError):
        extract_work_reference(**case["kwargs"], maximum_result_set=n)


@pytest.mark.parametrize("field,value", [("layer_thickness_mm", .03), ("layer_thickness_mm", 0.),
    ("cell_area_mm2", 7.), ("cell_area_mm2", np.inf)])
def test_geometry_and_units_must_match(case, field, value):
    with pytest.raises(ValueError):
        extract_work_reference(**dict(case["kwargs"], **{field: value}))


def test_requested_terminal_state_must_be_present(case):
    write_csv(case["paths"]["pilot_csv"], PILOT_HEADER, case["pilots"][:2])
    with pytest.raises(ValueError, match="missing requested pilot states"):
        extract_work_reference(**case["kwargs"])


def test_face_terminal_state_must_be_present(case):
    write_csv(case["paths"]["face_csv"], FACE_HEADER, case["faces"][:16])
    with pytest.raises(ValueError, match="missing requested face states"):
        extract_work_reference(**case["kwargs"])


def test_suffix_face_completeness_is_not_required(case):
    write_csv(case["paths"]["face_csv"], FACE_HEADER, case["faces"][:17])
    assert extract_work_reference(**case["kwargs"], maximum_result_set=1).reference.shape == (1, 134)


def test_malformed_csv_row_and_duplicate_header_fail(case):
    with case["paths"]["face_csv"].open("a") as stream:
        stream.write("3,bottom,31,1,2,3,unexpected\n")
    with pytest.raises(ValueError, match="malformed CSV"):
        extract_work_reference(**case["kwargs"])
    write_csv(case["paths"]["pilot_csv"], (*PILOT_HEADER, "rf_p_ex"), case["pilots"])
    with pytest.raises(ValueError, match="CSV columns"):
        extract_work_reference(**case["kwargs"])


def test_integral_native_csv_numbers_and_roundoff_clock_are_accepted(case):
    pilots = [dict(row, result_set=f"{row['result_set']:.1f}", time_s=row["time_s"] + 1e-9)
              for row in case["pilots"]]
    faces = [dict(row, result_set=f"{row['result_set']}.0", node_id=f"{row['node_id']}.0")
             for row in case["faces"]]
    write_csv(case["paths"]["pilot_csv"], PILOT_HEADER, pilots)
    write_csv(case["paths"]["face_csv"], FACE_HEADER, faces)
    assert extract_work_reference(**case["kwargs"]).reference.shape == (3, 134)


@pytest.mark.parametrize("change", ["missing_node", "duplicate_node", "nonfinite_mode", "coordinates", "nonreal_mode"])
def test_invalid_mode_archive(case, change):
    path = case["paths"]["modes_npz"]
    with np.load(path, allow_pickle=False) as saved:
        coordinates, modes = saved["coords_mm"], saved["modes"]
    if change == "missing_node":
        coordinates, modes = coordinates[:-1], modes[:-3]
    elif change == "duplicate_node":
        coordinates[0] = coordinates[1]
    elif change == "nonfinite_mode":
        modes[0, 0] = np.inf
    elif change == "coordinates":
        coordinates[0, 0] += .01
    elif change == "nonreal_mode":
        modes = modes.astype(complex)
    np.savez(path, coords_mm=coordinates, modes=modes)
    with pytest.raises(ValueError):
        extract_work_reference(**case["kwargs"])


def test_reaction_file_drift_is_rejected(case, monkeypatch):
    original = work_reference._number
    changed = False

    def change_mtime(value, name):
        nonlocal changed
        if name == "rf_x_N" and not changed:
            path = case["paths"]["face_csv"]
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
            changed = True
        return original(value, name)

    monkeypatch.setattr(work_reference, "_number", change_mtime)
    with pytest.raises(ValueError, match="input changed"):
        extract_work_reference(**case["kwargs"])


def test_oversized_csv_line_is_rejected_before_large_allocation(case):
    with case["paths"]["face_csv"].open("a") as stream:
        stream.write(" " * (work_reference.MAX_CSV_LINE_BYTES + 1) + "\n")
    with pytest.raises(ValueError, match="line resource bound"):
        extract_work_reference(**case["kwargs"], maximum_result_set=1)


def test_suffix_cannot_exceed_total_expected_row_bound(case):
    rows = case["faces"] + [case["faces"][-1]]
    write_csv(case["paths"]["face_csv"], FACE_HEADER, rows)
    with pytest.raises(ValueError, match="row resource bound"):
        extract_work_reference(**case["kwargs"], maximum_result_set=1)
