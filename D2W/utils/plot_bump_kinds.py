import argparse
import os
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm


PG_PREFIXES = (
    "VDD",
    "VSS",
    "VPP",
    "VCC",
    "GND",
    "VAA",
    "VDDQ",
    "VDDQL",
    "AVDD",
    "AVSS",
    "DVDD",
    "DVSS",
    "VDDA",
    "VSSA",
    "PVDD",
    "PVSS",
)


def infer_bump_kind(net_name):
    lower = net_name.lower()
    upper = net_name.upper()

    if lower == "dummy":
        return "dummy"
    if "_rd_" in lower or lower.startswith("rd_") or lower.endswith("_rd"):
        return "redundant"
    if upper.startswith(PG_PREFIXES):
        return "pg"
    return "critical"


def load_bumps(bmap_path):
    bumps = []
    with open(bmap_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            x = float(parts[2])
            y = float(parts[3])
            net = parts[4]
            bumps.append(
                {
                    "x": x,
                    "y": y,
                    "net": net,
                    "kind": infer_bump_kind(net),
                }
            )

    if not bumps:
        raise ValueError(f"No valid bump records found in {bmap_path}")

    return bumps


def build_kind_bitmap(bumps):
    x_values = sorted({b["x"] for b in bumps})
    y_values = sorted({b["y"] for b in bumps}, reverse=True)

    x_to_col = {x: idx for idx, x in enumerate(x_values)}
    y_to_row = {y: idx for idx, y in enumerate(y_values)}

    bitmap = np.zeros((len(y_values), len(x_values)), dtype=int)
    kind_to_code = {
        "critical": 1,
        "redundant": 2,
        "pg": 3,
        "dummy": 4,
    }

    for bump in bumps:
        row = y_to_row[bump["y"]]
        col = x_to_col[bump["x"]]
        bitmap[row, col] = kind_to_code[bump["kind"]]

    return bitmap


def draw_bitmap(bitmap, title, output_path):
    cmap = ListedColormap(
        [
            (0.92, 0.92, 0.92),  # non-pad
            (1.0, 0.5, 0.5),     # critical
            (0.4, 0.4, 0.9),     # redundant
            (1.0, 0.82, 0.25),   # pg
            (0.0, 0.6, 0.0),     # dummy
        ]
    )
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=cmap.N)

    plt.figure(figsize=(10, 10))
    plt.imshow(bitmap, cmap=cmap, norm=norm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")

    legend_handles = [
        patches.Patch(color=(1.0, 0.5, 0.5), label="Critical"),
        patches.Patch(color=(0.4, 0.4, 0.9), label="Redundant"),
        patches.Patch(color=(1.0, 0.82, 0.25), label="PG"),
        patches.Patch(color=(0.0, 0.6, 0.0), label="Dummy"),
        patches.Patch(color=(0.92, 0.92, 0.92), label="Non-pad"),
    ]
    plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot bump-kind bitmap from a .bmap file.")
    parser.add_argument("--bmap", required=True, help="Path to input .bmap")
    parser.add_argument("--output", help="Path to output PNG")
    args = parser.parse_args()

    bmap_path = args.bmap
    output_path = args.output
    if not output_path:
        stem = os.path.splitext(bmap_path)[0]
        output_path = stem + "_kind_bitmap.png"

    bumps = load_bumps(bmap_path)
    bitmap = build_kind_bitmap(bumps)
    counts = Counter(b["kind"] for b in bumps)

    title = os.path.basename(os.path.splitext(bmap_path)[0]) + " bump kinds"
    draw_bitmap(bitmap, title, output_path)

    print(f"Saved: {output_path}")
    print(f"Counts: critical={counts['critical']} redundant={counts['redundant']} pg={counts['pg']} dummy={counts['dummy']}")


if __name__ == "__main__":
    main()
