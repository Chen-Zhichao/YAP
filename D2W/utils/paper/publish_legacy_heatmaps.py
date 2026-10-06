#!/usr/bin/env python3
"""Promote checked Legacy-Mixed heatmaps, preserving previous artifacts."""
from __future__ import annotations

import argparse
import csv
import fcntl
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

import repair_original_tier_heatmaps as repair

ROOT = repair.ROOT
DEST = repair.YAP / "D2W/output/case_studies/d2d_size_thickness_heatmaps"


def checked_files():
    from initial_bow_distribution import D2D_INITIAL_BOW

    if (ROOT / "PUBLICATION_HOLD.json").exists():
        raise ValueError("publication held pending physical operator alignment")
    if repair.read(ROOT / "STATUS.json")["state"] != "COMPLETE":
        raise ValueError("primary campaign is not complete")
    source = ROOT / "figures"
    manifest = repair.read(source / "MANIFEST.json")
    if manifest["contract"] != repair.CONTRACT or manifest["distribution"] != D2D_INITIAL_BOW.audit():
        raise ValueError("orientation/source mismatch")
    if len(manifest["source_packages"]) != 14:
        raise ValueError("not all thicknesses qualified")
    for ready in manifest["source_packages"].values():
        if repair.common.sha(Path(ready["package"])) != ready["package_sha256"]:
            raise ValueError("qualified IBL changed")
    regression = manifest["baseline_regression"]
    if len(regression) != 5 or not all(r["passed"] and r["absolute_error"] < 1e-10 for r in regression):
        raise ValueError("baseline gamma-sensitivity regression failed")
    with (source / "ALL_DATA.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    expected = {(p, h, s, g) for p in (6, 2) for h in range(20, 81, 10)
                for s in range(5, 16) for g in ((.75, .8) if p == 6 else (.75, .8, .85))}
    actual = {(int(r["bond_pitch_um"]), int(r["total_die_thickness_um"]),
               float(r["die_size_mm"]), float(r["gamma"])) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("missing or duplicate heatmap cells")
    files = {p.name: p for p in source.iterdir() if p.suffix in (".csv", ".png")}
    if len([p for p in files.values() if p.suffix == ".png"]) != 7:
        raise ValueError("expected seven generated PNGs")
    # Maintain the established figure paths as aliases to the checked new plots.
    for name, path in list(files.items()):
        if name.endswith(".png"):
            alias = name.replace("_CRITICAL_30PCT", "")
            if "BI6_BI2" in alias:
                alias = alias.replace("BI6_BI2", "BI2_BI6")
            files[alias] = path
    return files, manifest


def publish():
    files, manifest = checked_files()
    archive = DEST / "archive_before_legacy_mixed_20260918"
    archive.mkdir(parents=True, exist_ok=True)
    published = {}
    for name, source in files.items():
        target = DEST / name
        if target.exists() and not (archive / name).exists():
            shutil.copy2(target, archive / name)
        temporary = target.with_name(target.name + ".publishing")
        shutil.copy2(source, temporary)
        temporary.replace(target)
        published[name] = repair.common.sha(target)
    for suffix in ("BI6_BI2_CRITICAL_30PCT_GAMMA_0p75", "BI6_BI2_CRITICAL_30PCT_GAMMA_0p80",
                   "BI2_CRITICAL_30PCT_GAMMA_0p85"):
        target = DEST / f"MANIFEST_{suffix}.json"
        if target.exists() and not (archive / target.name).exists():
            shutil.copy2(target, archive / target.name)
        repair.write(target, dict(**manifest, promotion_source=str(ROOT / "figures/MANIFEST.json"),
                                  published_files_sha256=published))
    record = dict(state="COMPLETE", contract=repair.CONTRACT, files_sha256=published,
                  destination=str(DEST), previous_artifacts=str(archive), time=time.time())
    repair.write(ROOT / "PUBLICATION.json", record)
    print(record, flush=True)


def watch():
    with (ROOT / ".publication.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + 14*3600
        while time.monotonic() < deadline:
            if (ROOT / "PUBLICATION_HOLD.json").exists():
                raise RuntimeError("publication held pending physical operator alignment")
            status = repair.read(ROOT / "STATUS.json")
            if status["state"] == "COMPLETE":
                publish()
                return
            if status.get("failed") or status["state"] in ("CHECKS_FAILED", "PLOT_FAILED"):
                raise RuntimeError("primary campaign requires review; old plots preserved")
            if (ROOT / "TWO_HOST_ERROR.json").exists():
                raise RuntimeError("primary coordinator failed; old plots preserved")
            pid = repair.read(ROOT / "TWO_HOST_LAUNCH.json")["pid"]
            command = Path(f"/proc/{pid}/cmdline")
            if not command.exists() or b"dispatch_heatmap_two_hosts.py" not in command.read_bytes():
                raise RuntimeError("primary coordinator is no longer alive; inspect status")
            time.sleep(30)
        raise RuntimeError("publication wait exceeded 14-hour limit")


def launch():
    if socket.gethostname().split(".")[0] != "sysml":
        raise ValueError("watch on the primary coordinator host")
    receipt = ROOT / "PUBLICATION_LAUNCH.json"
    if receipt.exists():
        raise ValueError("publication watcher already launched")
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "watch"]
    with (ROOT / "PUBLICATION.log").open("x") as log:
        proc = subprocess.Popen(command, cwd=repair.YAP, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    repair.write(receipt, dict(host="sysml", pid=proc.pid, command=command, time=time.time()))
    print(proc.pid, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "watch", "publish", "check"))
    args = parser.parse_args()
    if args.action == "launch":
        launch()
    elif args.action == "check":
        files, _ = checked_files()
        print(sorted(files))
    else:
        try:
            watch() if args.action == "watch" else publish()
        except Exception:
            repair.write(ROOT / "PUBLICATION_ERROR.json", dict(traceback=traceback.format_exc()))
            raise


if __name__ == "__main__":
    main()
