import argparse
import os
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
import yaml


CHIPLET_ID_RE = re.compile(r"chiplet_(\d+)")
SHARED_NET_RE = re.compile(r"chiplet_(\d+)_chiplet_(\d+)_link_")


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _first_region_bmap(chiplet_def):
    regions = chiplet_def.get("regions", {})
    if not regions:
        return None
    return next(iter(regions.values())).get("bmap")


def _detect_chiplet_id_from_bmap(bmap_path):
    counter = Counter()
    if not bmap_path or not os.path.exists(bmap_path):
        return None

    with open(bmap_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            _, _, _, _, port, net = parts
            for token in (port, net):
                match = CHIPLET_ID_RE.search(token)
                if match:
                    counter[match.group(1)] += 1

    if not counter:
        return None
    return counter.most_common(1)[0][0]


def _parse_shared_net_counts(shared_root):
    pair_counts = Counter()
    if not shared_root or not os.path.isdir(shared_root):
        return pair_counts

    for name in sorted(os.listdir(shared_root)):
        if not name.endswith("_shared_nets.txt"):
            continue
        path = os.path.join(shared_root, name)
        with open(path, "r") as f:
            for line in f:
                match = SHARED_NET_RE.search(line.strip())
                if not match:
                    continue
                a, b = sorted((match.group(1), match.group(2)), key=int)
                pair_counts[(a, b)] += 1
    return pair_counts


def _short_instance_label(instance_name):
    if instance_name.startswith("Substrate_"):
        return instance_name.replace("_0", "")
    label = instance_name
    if "_" in instance_name:
        label = instance_name.rsplit("_", 1)[0]
    return label.replace("_", "\n")


def plot_design(design_dir, shared_root, output_path):
    chiplet_defs = _load_yaml(os.path.join(design_dir, "generated_chiplet_definitions.3dbv"))["ChipletDef"]
    stack_cfg = _load_yaml(os.path.join(design_dir, "generated_stack_config.3dbx"))
    chiplet_insts = stack_cfg["ChipletInst"]
    stack = stack_cfg["Stack"]

    root_instance = next(name for name, item in stack.items() if item.get("root"))
    root_ref = chiplet_insts[root_instance]["reference"]
    root_w, root_h = chiplet_defs[root_ref]["design_area"]

    top_instances = []
    chiplet_id_to_instance = {}
    for instance_name, inst_cfg in chiplet_insts.items():
        ref_name = inst_cfg["reference"]
        x0, y0 = stack[instance_name]["loc"]
        w, h = chiplet_defs[ref_name]["design_area"]
        entry = {
            "instance": instance_name,
            "reference": ref_name,
            "x": x0,
            "y": y0,
            "w": w,
            "h": h,
        }
        if instance_name == root_instance:
            root_entry = entry
            continue

        region_bmap = _first_region_bmap(chiplet_defs[ref_name])
        chiplet_id = _detect_chiplet_id_from_bmap(
            os.path.join(shared_root, region_bmap) if region_bmap else None
        )
        entry["chiplet_id"] = chiplet_id
        top_instances.append(entry)
        if chiplet_id is not None:
            chiplet_id_to_instance[chiplet_id] = entry

    pair_counts = _parse_shared_net_counts(shared_root)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.set_xlim(-0.05 * root_w, 1.05 * root_w)
    ax.set_ylim(-0.05 * root_h, 1.05 * root_h)
    ax.axis("off")

    substrate = Rectangle(
        (root_entry["x"], root_entry["y"]),
        root_entry["w"],
        root_entry["h"],
        facecolor="#d9d9d9",
        edgecolor="#444444",
        linewidth=2.5,
        zorder=0,
    )
    ax.add_patch(substrate)
    ax.text(
        root_entry["x"] + root_entry["w"] / 2,
        root_entry["y"] + root_entry["h"] / 2,
        _short_instance_label(root_instance),
        ha="center",
        va="center",
        fontsize=16,
        color="#4a4a4a",
        zorder=1,
    )

    palette = [
        "#457b9d",
        "#e76f51",
        "#2a9d8f",
        "#f4a261",
        "#8d99ae",
        "#a8dadc",
    ]

    centers = {}
    for idx, entry in enumerate(sorted(top_instances, key=lambda item: item["instance"])):
        color = palette[idx % len(palette)]
        rect = Rectangle(
            (entry["x"], entry["y"]),
            entry["w"],
            entry["h"],
            facecolor=color,
            edgecolor="#1f1f1f",
            linewidth=2.0,
            alpha=0.9,
            zorder=2,
        )
        ax.add_patch(rect)
        cx = entry["x"] + entry["w"] / 2
        cy = entry["y"] + entry["h"] / 2
        centers[entry["instance"]] = (cx, cy)

        chiplet_tag = (
            f"\nchiplet_{entry['chiplet_id']}"
            if entry.get("chiplet_id") is not None
            else ""
        )
        ax.text(
            cx,
            cy,
            f"{_short_instance_label(entry['instance'])}{chiplet_tag}",
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            weight="bold",
            zorder=3,
        )

    for (chiplet_a, chiplet_b), count in sorted(pair_counts.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))):
        if chiplet_a not in chiplet_id_to_instance or chiplet_b not in chiplet_id_to_instance:
            continue
        inst_a = chiplet_id_to_instance[chiplet_a]["instance"]
        inst_b = chiplet_id_to_instance[chiplet_b]["instance"]
        (x1, y1), (x2, y2) = centers[inst_a], centers[inst_b]

        edge = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-",
            linewidth=3.0,
            color="#c1121f",
            alpha=0.9,
            zorder=4,
        )
        ax.add_patch(edge)
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            f"{count} shared\nbump nets",
            ha="center",
            va="center",
            fontsize=10,
            color="#9d0208",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#c1121f", "alpha": 0.9},
            zorder=5,
        )

    legend_items = [
        Rectangle((0, 0), 1, 1, facecolor="#d9d9d9", edgecolor="#444444", label="Substrate / base die"),
        Rectangle((0, 0), 1, 1, facecolor=palette[0], edgecolor="#1f1f1f", label="Top chiplet"),
        Line2D([0], [0], color="#c1121f", linewidth=3, label="Shared bump connection"),
    ]
    ax.legend(handles=legend_items, loc="upper right", frameon=True)

    design_name = os.path.basename(os.path.normpath(design_dir))
    ratio_name = os.path.basename(os.path.normpath(shared_root))
    ax.set_title(f"{design_name} Topology\n(shared nets from {ratio_name})", fontsize=18, pad=18)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot a top-view chiplet topology with shared bump-net links.")
    parser.add_argument("--design-dir", required=True, help="Design root containing generated_chiplet_definitions.3dbv and generated_stack_config.3dbx")
    parser.add_argument("--shared-root", required=True, help="Directory containing *_shared_nets.txt and .bmap files used for chiplet-id inference")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()

    plot_design(
        design_dir=args.design_dir,
        shared_root=args.shared_root,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
