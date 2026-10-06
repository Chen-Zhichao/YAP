"""
Generate Criticality File from Bump Map

This script generates a starter criticality file based on a bump map file.
For each net in the bump map, criticality is specified using the new format:
<net> <group_size> <tolerated_esd_failures> <tolerated_mechanical_failures>

Where:
- group_size = number of bumps/pads for the net
- tolerated_esd_failures = number of ESD failures that can be tolerated (default: group_size - 1)
- tolerated_mechanical_failures = number of mechanical failures that can be tolerated (default: group_size - 1)

The actual criticality values are calculated when the file is read:
- esd_criticality = (group_size - tolerated_esd_failures) / group_size
- mechanical_criticality = (group_size - tolerated_mechanical_failures) / group_size

Usage:
    python generate_criticality.py <input_bmap_file> [--force]

Arguments:
    input_bmap_file  - Path to the input bump map file (.bmap)
    --force          - Optional flag to overwrite existing output file without prompting

Output:
    Creates a criticality file named <input_name>_criticality.txt

Example:
    python generate_criticality.py UCIe_standard.bmap
    python generate_criticality.py UCIe_standard.bmap --force

Example Output Format (see UCIe_advanced_criticality.txt):
    rxckRD rxckn rxckp rxtrk 4 1 1
    rxcksb rxcksbRD 2 1 1
    vccfwdio 5 5 4
    vccio 30 30 29
"""

import sys
import os
from collections import defaultdict


def read_bmap_nets(filename):
    """
    Read bump map file and count the number of bumps per net.
    
    Args:
        filename: Path to the input .bmap file
    
    Returns:
        Dictionary mapping net names to bump counts
    """
    net_counts = defaultdict(int)
    
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse bump map line: <instance> <bump_type> <x> <y> <port> <net>
                parts = line.split()
                if len(parts) != 6:
                    print(f"Warning: Line {line_num} has {len(parts)} fields (expected 6), skipping: {line}")
                    continue
                
                instance, bump_type, x, y, port, net = parts
                net_counts[net] += 1
        
        if not net_counts:
            print(f"Error: No valid bump entries found in {filename}")
            sys.exit(1)
        
        return net_counts
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading bump map file: {e}")
        sys.exit(1)


def generate_criticality_file(net_counts, output_filename, force=False):
    """
    Generate criticality file based on net counts.
    
    New format: <net> <group_size> <tolerated_esd_failures> <tolerated_mechanical_failures>
    
    By default, sets tolerated failures to (group_size - 1), meaning only 1 failure
    causes the net to fail. This results in criticality = 1/group_size.
    
    Args:
        net_counts: Dictionary mapping net names to bump counts
        output_filename: Path to the output criticality file
        force: If True, overwrite existing file without prompting
    """
    # Check if output file exists
    if os.path.exists(output_filename) and not force:
        response = input(f"Output file '{output_filename}' already exists. Overwrite? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("Operation cancelled.")
            sys.exit(0)
    
    try:
        with open(output_filename, 'w') as f:
            # Sort nets alphabetically for consistent output
            sorted_nets = sorted(net_counts.keys())
            
            for net in sorted_nets:
                group_size = net_counts[net]
                # Default: tolerate (group_size - 1) failures
                # This means only 1 working pad is needed, so criticality = 1/group_size
                tolerated_esd = group_size - 1
                tolerated_mech = group_size - 1
                
                # Format: <net> <group_size> <tolerated_esd_failures> <tolerated_mechanical_failures>
                f.write(f"{net} {group_size} {tolerated_esd} {tolerated_mech}\n")
        
        print(f"Successfully generated criticality file: {output_filename}")
        print(f"Total nets: {len(net_counts)}")
        print("\nCriticality summary:")
        
        # Show some statistics
        criticality_values = defaultdict(list)
        for net, count in net_counts.items():
            criticality = 1.0 / count
            criticality_values[criticality].append(net)
        
        for criticality in sorted(criticality_values.keys(), reverse=True):
            nets = criticality_values[criticality]
            bump_count = int(1.0 / criticality)
            print(f"  Criticality {criticality:.4f} ({bump_count} bump{'s' if bump_count > 1 else ''}): {len(nets)} net{'s' if len(nets) > 1 else ''}")
        
        print("\nFormat: <net> <group_size> <tolerated_esd_failures> <tolerated_mechanical_failures>")
        print("Default: tolerated_failures = group_size - 1 (criticality = 1/group_size)")
        print("\nNote: Please review and modify the generated file to set appropriate failure tolerances.")
        print("      See UCIe_advanced_criticality.txt for an example of the new format.")
    
    except Exception as e:
        print(f"Error writing criticality file: {e}")
        sys.exit(1)


def get_output_filename(input_filename):
    """
    Generate output filename based on input filename.
    
    Args:
        input_filename: Path to input .bmap file
    
    Returns:
        Output filename with _criticality.txt suffix
    """
    # Remove file extension if present
    base_name = os.path.splitext(input_filename)[0]
    return f"{base_name}_criticality.txt"


def main():
    """Main function to parse arguments and generate criticality file."""
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Error: Missing required argument")
        print("\nUsage: python generate_criticality.py <input_bmap_file> [--force]")
        print("\nArguments:")
        print("  input_bmap_file  - Path to the input bump map file (.bmap)")
        print("  --force          - Optional flag to overwrite existing output file without prompting")
        print("\nExample:")
        print("  python generate_criticality.py UCIe_standard.bmap")
        print("  python generate_criticality.py UCIe_standard.bmap --force")
        sys.exit(1)
    
    input_filename = sys.argv[1]
    force = '--force' in sys.argv
    
    # Validate input file exists
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found")
        sys.exit(1)
    
    # Generate output filename
    output_filename = get_output_filename(input_filename)
    
    print(f"Reading bump map: {input_filename}")
    print(f"Output file: {output_filename}")
    print()
    
    # Read bump map and count nets
    net_counts = read_bmap_nets(input_filename)
    
    # Generate criticality file
    generate_criticality_file(net_counts, output_filename, force)


if __name__ == "__main__":
    main()
