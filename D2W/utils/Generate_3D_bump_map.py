import random
import numpy as np

def generate_bump_map(
    x_size, y_size,
    num_vdd, num_vss,
    num_signal,
    redundant_range=(1, 20),
    num_dummy=0,
    seed=None,
    pitch=10.0,
    start_x=-5407.5, start_y=-4455.0
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    num_dummy=(x_size * y_size) - num_vdd - num_vss - num_signal

    total_bumps = x_size * y_size
    total_assigned = num_vdd + num_vss + num_signal + num_dummy
    if total_assigned > total_bumps:
        raise ValueError("Too many bumps assigned for given map size.")

    # Initialize map
    bump_map = np.full((x_size, y_size), "DUMMY", dtype=object)
    all_positions = [(i, j) for i in range(x_size) for j in range(y_size)]
    random.shuffle(all_positions)

    # Helper function
    def assign(label, count):
        nonlocal all_positions
        assigned = all_positions[:count]
        all_positions = all_positions[count:]
        for (i, j) in assigned:
            bump_map[i, j] = label

    # Assign bumps
    assign("VDD", num_vdd)
    assign("VSS", num_vss)
    assign("DUMMY", num_dummy)

    # Assign signals with redundant groups
    remaining_signal = num_signal
    signal_id = 0
    while remaining_signal > 0:
        group_size = random.randint(*redundant_range)
        if group_size > remaining_signal:
            group_size = remaining_signal
        label = f"s{signal_id}"
        assign(label, group_size)
        signal_id += 1
        remaining_signal -= group_size

    # Generate output lines
    output_lines = []
    dummy_count = 1
    for i in range(x_size):
        for j in range(y_size):
            bump_type = bump_map[i, j]
            x_coord = start_x + j * pitch
            y_coord = start_y - i * pitch

            if bump_type == "DUMMY":
                line = f"Dummy_{dummy_count} uBUMP {x_coord:.1f} {y_coord:.1f} dummy dummy"
                dummy_count += 1
            elif bump_type == "VDD":
                line = f"P{i*y_size + j +1} uBUMP {x_coord:.1f} {y_coord:.1f} VDD VDD"
            elif bump_type == "VSS":
                line = f"G{i*y_size + j +1} uBUMP {x_coord:.1f} {y_coord:.1f} VSS VSS"
            else:  # signal
                line = f"{bump_type} uBUMP {x_coord:.1f} {y_coord:.1f} {bump_type} {bump_type}"

            output_lines.append(line)

    return bump_map, output_lines

# ===== Example usage =====
if __name__ == "__main__":
    bump_map, bump_list = generate_bump_map(
        x_size=150, y_size=100,
        num_vdd=750,
        num_vss=750,
        num_signal=10000,
        redundant_range=(1, 5),
        num_dummy=5,
        seed=42
    )

    with open("/w/ee.00/puneet/jooyeon/2026_DAC/bump_map_output.txt", "w") as f:
        for line in bump_list:
            f.write(line + "\n")

    print("Bump map generated and saved to bump_map_output.txt")
