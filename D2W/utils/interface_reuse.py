#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import hashlib
import json
import os
import shutil
import tempfile
from collections import OrderedDict

import numpy as np
from omegaconf import OmegaConf


_CFG_SIGNATURE_EXCLUDE = {
    "INTERFACE",
    "INTERFACE_TOP",
    "INTERFACE_BOT",
    "DESIGN",
    "DEBUG",
    "plot_flag",
    "verbose",
}


def _normalize_cfg_value(value):
    if isinstance(value, dict):
        return {
            str(key): _normalize_cfg_value(sub_value)
            for key, sub_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_cfg_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return round(value, 12)
    return value


def _cfg_signature_dict(cfg):
    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    return {
        str(key): _normalize_cfg_value(value)
        for key, value in sorted(cfg_container.items(), key=lambda item: str(item[0]))
        if key not in _CFG_SIGNATURE_EXCLUDE
    }


def _hash_array(array):
    np_array = np.asarray(array)
    if np_array.dtype == bool:
        np_array = np_array.astype(np.uint8)
    elif np_array.dtype.kind == "f":
        np_array = np.nan_to_num(np_array.astype(np.float32), nan=1.0e30)
    else:
        np_array = np_array.astype(np.int64, copy=False)
    return hashlib.sha1(np_array.tobytes()).hexdigest()


def _append_file_suffix(filename, file_suffix):
    if not file_suffix:
        return filename
    stem, ext = os.path.splitext(filename)
    return f"{stem}{file_suffix}{ext}"


def _link_or_copy(src_path, dst_path):
    if os.path.exists(dst_path):
        os.remove(dst_path)
    try:
        os.link(src_path, dst_path)
    except OSError:
        shutil.copy2(src_path, dst_path)


def _redundant_group_signature(pad_bitmap_collection):
    criticality_info = pad_bitmap_collection["criticality_info"]
    group_records = []
    for net, physical_mask in pad_bitmap_collection["redundant_net_to_1d_physical_mask"].items():
        physical_mask = tuple(
            int(index) for index in np.sort(np.asarray(physical_mask, dtype=np.int64))
        )
        net_info = criticality_info[net]
        group_records.append(
            (
                len(physical_mask),
                int(net_info["tolerated_esd_failures"]),
                int(net_info["tolerated_mechanical_failures"]),
                physical_mask,
            )
        )
    group_records.sort()
    return group_records


def build_interface_signature(cfg, pad_bitmap_collection):
    signature_payload = {
        "cfg": _cfg_signature_dict(cfg),
        "critical_bitmap": _hash_array(pad_bitmap_collection["CRITICAL_PAD_BITMAP"]),
        "redundant_bitmap": _hash_array(pad_bitmap_collection["REDUNDANT_PAD_BITMAP"]),
        "dummy_bitmap": _hash_array(pad_bitmap_collection["DUMMY_PAD_BITMAP"]),
        "esd_bitmap": _hash_array(pad_bitmap_collection["ESD_CRITICAL_PAD_BITMAP"]),
        "pad_coords": _hash_array(pad_bitmap_collection["pad_coords"]),
        "redundant_groups": _redundant_group_signature(pad_bitmap_collection),
    }
    payload_json = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload_json.encode("utf-8")).hexdigest()


def group_identical_interfaces(cfg_dict, pad_bitmap_collection_dict):
    signature_to_members = OrderedDict()
    for interface_name in cfg_dict:
        signature = build_interface_signature(
            cfg=cfg_dict[interface_name],
            pad_bitmap_collection=pad_bitmap_collection_dict[interface_name],
        )
        signature_to_members.setdefault(signature, []).append(interface_name)

    grouped_interfaces = OrderedDict()
    for members in signature_to_members.values():
        representative = members[0]
        grouped_interfaces[representative] = members
    return grouped_interfaces


def build_risk_map_signature(cfg, pad_bitmap_collection):
    """
    Signature for pad-risk-map reuse.

    Pad risk maps depend on geometry and modeling parameters, but not on the
    semantic net/port assignment of bumps. In particular, repeated chiplets in a
    Random_* folder may have different signal naming / bump-kind assignment while
    still sharing identical pad coordinates and therefore identical risk maps.
    """
    signature_payload = {
        "cfg": _cfg_signature_dict(cfg),
        "pad_coords": _hash_array(pad_bitmap_collection["pad_coords"]),
    }
    payload_json = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload_json.encode("utf-8")).hexdigest()


def group_risk_equivalent_interfaces(cfg_dict, pad_bitmap_collection_dict):
    signature_to_members = OrderedDict()
    for interface_name in cfg_dict:
        signature = build_risk_map_signature(
            cfg=cfg_dict[interface_name],
            pad_bitmap_collection=pad_bitmap_collection_dict[interface_name],
        )
        signature_to_members.setdefault(signature, []).append(interface_name)

    grouped_interfaces = OrderedDict()
    for members in signature_to_members.values():
        representative = members[0]
        grouped_interfaces[representative] = members
    return grouped_interfaces


def _load_criticality_signature(criticality_path):
    entries = []
    with open(criticality_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            _, num_copy, tolerated_esd_failures, tolerated_mechanical_failures = parts
            entries.append(
                (
                    int(num_copy),
                    int(tolerated_esd_failures),
                    int(tolerated_mechanical_failures),
                )
            )
    entries.sort()
    return entries


def _raw_interface_signature_cache_path(cfg, bmap_path, criticality_path):
    cache_key = json.dumps(
        {
            "cfg": _cfg_signature_dict(cfg),
            "bmap_path": os.path.abspath(bmap_path),
            "criticality_path": os.path.abspath(criticality_path),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()
    return os.path.join(
        tempfile.gettempdir(),
        f"d2w_raw_interface_signature__{digest}.json",
    )


def _file_stat_signature(path):
    stat = os.stat(path)
    return {
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
    }


def build_raw_interface_signature(cfg, bmap_path, criticality_path):
    cache_path = _raw_interface_signature_cache_path(cfg, bmap_path, criticality_path)
    current_sources = {
        "bmap": _file_stat_signature(bmap_path),
        "criticality": _file_stat_signature(criticality_path),
    }
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cache_payload = json.load(f)
            if cache_payload.get("sources") == current_sources:
                cached_signature = cache_payload.get("signature")
                if cached_signature:
                    return cached_signature
        except (OSError, json.JSONDecodeError):
            pass

    criticality_signature = _load_criticality_signature(criticality_path)

    bump_rows = []
    with open(bmap_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            _, _, x, y, _, net = parts
            bump_rows.append((float(x), float(y), net))

    if not bump_rows:
        payload_json = json.dumps(
            {"cfg": _cfg_signature_dict(cfg), "empty": True},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha1(payload_json.encode("utf-8")).hexdigest()

    pad_array_left = min(x for x, _, _ in bump_rows)
    pad_array_right = max(x for x, _, _ in bump_rows)
    pad_array_top = max(y for _, y, _ in bump_rows)
    pad_array_bottom = min(y for _, y, _ in bump_rows)

    net_counts = OrderedDict()
    for _, _, net in bump_rows:
        net_counts[net] = net_counts.get(net, 0) + 1

    criticality_lookup = {}
    with open(criticality_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            net, _, tolerated_esd_failures, tolerated_mechanical_failures = parts
            criticality_lookup[net] = (
                int(tolerated_esd_failures),
                int(tolerated_mechanical_failures),
            )

    critical_positions = []
    redundant_positions = []
    dummy_positions = []
    esd_positions = []
    redundant_groups = {}

    for x, y, net in bump_rows:
        row = int(round((pad_array_top - y) / cfg.PITCH_r_um))
        col = int(round((x - pad_array_left) / cfg.PITCH_c_um))
        linear_idx = row * cfg.PAD_ARR_COL + col
        num_copies = net_counts[net]
        if "dummy" in net.lower():
            dummy_positions.append(linear_idx)
            continue
        if num_copies == 1:
            critical_positions.append(linear_idx)
            esd_positions.append(linear_idx)
            continue
        redundant_positions.append(linear_idx)
        redundant_groups.setdefault(net, []).append(linear_idx)
        tolerated_esd_failures, _ = criticality_lookup[net]
        if tolerated_esd_failures == 0:
            esd_positions.append(linear_idx)

    redundant_group_signature = []
    for net, physical_positions in redundant_groups.items():
        tolerated_esd_failures, tolerated_mechanical_failures = criticality_lookup[net]
        redundant_group_signature.append(
            (
                len(physical_positions),
                tolerated_esd_failures,
                tolerated_mechanical_failures,
                tuple(sorted(int(index) for index in physical_positions)),
            )
        )
    redundant_group_signature.sort()

    signature_payload = {
        "cfg": _cfg_signature_dict(cfg),
        "critical_positions": tuple(sorted(critical_positions)),
        "redundant_positions": tuple(sorted(redundant_positions)),
        "dummy_positions": tuple(sorted(dummy_positions)),
        "esd_positions": tuple(sorted(esd_positions)),
        "redundant_groups": redundant_group_signature,
        "criticality_signature": criticality_signature,
        "bbox": (
            round(pad_array_right - pad_array_left, 6),
            round(pad_array_top - pad_array_bottom, 6),
        ),
    }
    payload_json = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
    signature = hashlib.sha1(payload_json.encode("utf-8")).hexdigest()

    cache_payload = {
        "sources": current_sources,
        "signature": signature,
    }
    try:
        with open(cache_path, "w") as f:
            json.dump(cache_payload, f, sort_keys=True, separators=(",", ":"))
    except OSError:
        pass

    return signature


def group_raw_identical_interfaces(cfg_dict, bmap_path_dict, criticality_path_dict):
    signature_to_members = OrderedDict()
    for interface_name in cfg_dict:
        signature = build_raw_interface_signature(
            cfg=cfg_dict[interface_name],
            bmap_path=bmap_path_dict[interface_name],
            criticality_path=criticality_path_dict[interface_name],
        )
        signature_to_members.setdefault(signature, []).append(interface_name)

    grouped_interfaces = OrderedDict()
    for members in signature_to_members.values():
        grouped_interfaces[members[0]] = members
    return grouped_interfaces


def has_reused_interfaces(grouped_interfaces):
    return any(len(members) > 1 for members in grouped_interfaces.values())


def format_group_summary(grouped_interfaces):
    lines = []
    for representative, members in grouped_interfaces.items():
        if len(members) == 1:
            lines.append(f"{representative} (unique)")
        else:
            lines.append(
                f"{representative} (x{len(members)}): {', '.join(members)}"
            )
    return "\n".join(lines)


def write_group_metadata(output_root, grouped_interfaces, filename="collapsed_interface_groups.txt"):
    metadata_path = os.path.join(output_root, filename)
    with open(metadata_path, "w") as f:
        for representative, members in grouped_interfaces.items():
            f.write(f"{representative} {len(members)} {' '.join(members)}\n")
    return metadata_path


def copy_representative_bitmap_outputs(output_root, representative, duplicate):
    rep_dir = os.path.join(output_root, representative)
    dup_dir = os.path.join(output_root, duplicate)
    os.makedirs(dup_dir, exist_ok=True)

    copy_plan = [
        (f"{representative}_pad_bitmap.png", f"{duplicate}_pad_bitmap.png"),
    ]
    for src_name, dst_name in copy_plan:
        src_path = os.path.join(rep_dir, src_name)
        if os.path.exists(src_path):
            _link_or_copy(src_path, os.path.join(dup_dir, dst_name))

    note_path = os.path.join(dup_dir, f"{duplicate}_bitmap_reused_from.txt")
    with open(note_path, "w") as f:
        f.write(
            f"{duplicate} reuses the in-memory bitmap collection of representative interface "
            f"{representative}.\n"
        )


def copy_representative_risk_outputs(output_root, representative, duplicate, file_suffix=""):
    rep_dir = os.path.join(output_root, representative)
    dup_dir = os.path.join(output_root, duplicate)
    os.makedirs(dup_dir, exist_ok=True)

    copy_plan = [
        (
            _append_file_suffix(f"{representative}_risk.map", file_suffix),
            _append_file_suffix(f"{duplicate}_risk.map", file_suffix),
        ),
        (
            _append_file_suffix(f"{representative}_esd_risk_map.png", file_suffix),
            _append_file_suffix(f"{duplicate}_esd_risk_map.png", file_suffix),
        ),
        (
            _append_file_suffix(f"{representative}_overlay_risk_map.png", file_suffix),
            _append_file_suffix(f"{duplicate}_overlay_risk_map.png", file_suffix),
        ),
        (
            _append_file_suffix(f"{representative}_particle_risk_map.png", file_suffix),
            _append_file_suffix(f"{duplicate}_particle_risk_map.png", file_suffix),
        ),
        (
            _append_file_suffix(f"{representative}_mechanical_risk_map.png", file_suffix),
            _append_file_suffix(f"{duplicate}_mechanical_risk_map.png", file_suffix),
        ),
        (
            _append_file_suffix(f"{representative}_overall_risk_map.png", file_suffix),
            _append_file_suffix(f"{duplicate}_overall_risk_map.png", file_suffix),
        ),
    ]
    for src_name, dst_name in copy_plan:
        src_path = os.path.join(rep_dir, src_name)
        if os.path.exists(src_path):
            _link_or_copy(src_path, os.path.join(dup_dir, dst_name))


def copy_representative_simulation_outputs(output_root, representative, duplicate, file_suffix=""):
    rep_dir = os.path.join(output_root, representative)
    dup_dir = os.path.join(output_root, duplicate)
    os.makedirs(dup_dir, exist_ok=True)

    pattern = _append_file_suffix("simulation_failure_map_*.png", file_suffix)
    for src_path in glob.glob(os.path.join(rep_dir, pattern)):
        _link_or_copy(src_path, os.path.join(dup_dir, os.path.basename(src_path)))


def write_per_interface_yield_file(output_root, per_interface_yield_dict, file_suffix=""):
    yield_path = os.path.join(
        output_root,
        _append_file_suffix("assembly_yield_per_interface.txt", file_suffix),
    )
    with open(yield_path, "w") as f:
        for interface_name, interface_yield in per_interface_yield_dict.items():
            f.write(f"{interface_name} {interface_yield:.8f}\n")
    return yield_path
