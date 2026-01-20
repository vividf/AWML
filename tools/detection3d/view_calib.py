#!/usr/bin/env python
"""
View and compare calibration cache files.

Usage:
    # View a single calibration cache
    python tools/detection3d/view_calib.py view work_dirs/centerpoint_ptq_128_rand.calib

    # Compare two calibration caches
    python tools/detection3d/view_calib.py compare \
        work_dirs/centerpoint_ptq_128_seq.calib \
        work_dirs/centerpoint_ptq_128_rand.calib

    # Show statistics
    python tools/detection3d/view_calib.py stats work_dirs/centerpoint_ptq_128_rand.calib
"""

import argparse
import sys
from pathlib import Path

import torch


def view_calib(calib_path: str):
    """View calibration cache file."""
    calib_path = Path(calib_path)
    if not calib_path.exists():
        print(f"Error: File not found: {calib_path}")
        sys.exit(1)

    print("=" * 80)
    print(f"Calibration Cache: {calib_path}")
    print("=" * 80)

    cache = torch.load(calib_path, map_location="cpu")

    print(f"\nTotal quantizers: {len(cache)}")
    print("\nQuantizer amax values:")
    print("-" * 80)
    print(f"{'Quantizer Name':<60} {'Amax Value':>15}")
    print("-" * 80)

    # Sort by name for easier reading
    sorted_keys = sorted(cache.keys())

    for key in sorted_keys:
        amax = cache[key]
        if isinstance(amax, torch.Tensor):
            amax_val = amax.item() if amax.numel() == 1 else amax.tolist()
        else:
            amax_val = amax

        if isinstance(amax_val, (list, tuple)):
            print(f"{key:<60} {str(amax_val):>15}")
        else:
            print(f"{key:<60} {amax_val:>15.6f}")

    print("=" * 80)


def compare_calib(calib_path1: str, calib_path2: str, threshold: float = 0.01):
    """Compare two calibration cache files."""
    calib_path1 = Path(calib_path1)
    calib_path2 = Path(calib_path2)

    if not calib_path1.exists():
        print(f"Error: File not found: {calib_path1}")
        sys.exit(1)
    if not calib_path2.exists():
        print(f"Error: File not found: {calib_path2}")
        sys.exit(1)

    cache1 = torch.load(calib_path1, map_location="cpu")
    cache2 = torch.load(calib_path2, map_location="cpu")

    print("=" * 80)
    print(f"Comparing Calibration Caches")
    print("=" * 80)
    print(f"File 1: {calib_path1}")
    print(f"File 2: {calib_path2}")
    print(f"Threshold for significant difference: {threshold * 100:.1f}%")
    print("=" * 80)

    # Get all keys
    all_keys = set(cache1.keys()) | set(cache2.keys())
    only_in_1 = set(cache1.keys()) - set(cache2.keys())
    only_in_2 = set(cache2.keys()) - set(cache1.keys())

    if only_in_1:
        print(f"\nWarning: {len(only_in_1)} quantizers only in file 1:")
        for key in sorted(only_in_1):
            print(f"  - {key}")

    if only_in_2:
        print(f"\nWarning: {len(only_in_2)} quantizers only in file 2:")
        for key in sorted(only_in_2):
            print(f"  - {key}")

    # Compare common keys
    common_keys = set(cache1.keys()) & set(cache2.keys())
    print(f"\nCommon quantizers: {len(common_keys)}")

    differences = []
    for key in sorted(common_keys):
        amax1 = cache1[key]
        amax2 = cache2[key]

        # Convert to scalar if tensor
        if isinstance(amax1, torch.Tensor):
            amax1_val = amax1.item() if amax1.numel() == 1 else amax1.tolist()
        else:
            amax1_val = amax1

        if isinstance(amax2, torch.Tensor):
            amax2_val = amax2.item() if amax2.numel() == 1 else amax2.tolist()
        else:
            amax2_val = amax2

        # Handle list/tuple case
        if isinstance(amax1_val, (list, tuple)) or isinstance(amax2_val, (list, tuple)):
            if amax1_val != amax2_val:
                differences.append((key, amax1_val, amax2_val, None))
            continue

        # Calculate relative difference
        if amax1_val == 0 and amax2_val == 0:
            rel_diff = 0.0
        elif amax1_val == 0:
            rel_diff = float("inf")
        else:
            rel_diff = abs(amax1_val - amax2_val) / abs(amax1_val)

        if rel_diff > threshold:
            differences.append((key, amax1_val, amax2_val, rel_diff))

    if differences:
        print(f"\nFound {len(differences)} quantizers with significant differences:")
        print("-" * 100)
        print(f"{'Quantizer Name':<60} {'File 1':>15} {'File 2':>15} {'Rel Diff %':>12}")
        print("-" * 100)

        # Sort by relative difference
        differences.sort(key=lambda x: x[3] if x[3] is not None else 0, reverse=True)

        for key, val1, val2, rel_diff in differences:
            if rel_diff is not None:
                print(f"{key:<60} {val1:>15.6f} {val2:>15.6f} {rel_diff*100:>11.2f}%")
            else:
                print(f"{key:<60} {str(val1):>15} {str(val2):>15} {'N/A':>12}")

        print("-" * 100)
    else:
        print(f"\nNo significant differences found (all < {threshold * 100:.1f}%)")

    # Summary statistics
    if common_keys:
        rel_diffs = []
        for key in common_keys:
            amax1 = cache1[key]
            amax2 = cache2[key]

            if isinstance(amax1, torch.Tensor):
                amax1_val = amax1.item() if amax1.numel() == 1 else amax1.tolist()
            else:
                amax1_val = amax1

            if isinstance(amax2, torch.Tensor):
                amax2_val = amax2.item() if amax2.numel() == 1 else amax2.tolist()
            else:
                amax2_val = amax2

            if not isinstance(amax1_val, (list, tuple)) and not isinstance(amax2_val, (list, tuple)):
                if amax1_val != 0:
                    rel_diff = abs(amax1_val - amax2_val) / abs(amax1_val)
                    rel_diffs.append(rel_diff)

        if rel_diffs:
            import numpy as np

            print(f"\nSummary Statistics:")
            print(f"  Mean relative difference: {np.mean(rel_diffs) * 100:.2f}%")
            print(f"  Median relative difference: {np.median(rel_diffs) * 100:.2f}%")
            print(f"  Max relative difference: {np.max(rel_diffs) * 100:.2f}%")
            print(f"  Min relative difference: {np.min(rel_diffs) * 100:.2f}%")
            print(f"  Std relative difference: {np.std(rel_diffs) * 100:.2f}%")

    print("=" * 80)


def stats_calib(calib_path: str):
    """Show statistics about calibration cache."""
    calib_path = Path(calib_path)
    if not calib_path.exists():
        print(f"Error: File not found: {calib_path}")
        sys.exit(1)

    cache = torch.load(calib_path, map_location="cpu")

    print("=" * 80)
    print(f"Calibration Cache Statistics: {calib_path}")
    print("=" * 80)

    amax_values = []
    for key, amax in cache.items():
        if isinstance(amax, torch.Tensor):
            if amax.numel() == 1:
                amax_values.append(amax.item())
            else:
                # Flatten multi-element tensors
                amax_values.extend(amax.flatten().tolist())
        else:
            if isinstance(amax, (list, tuple)):
                amax_values.extend(amax)
            else:
                amax_values.append(amax)

    if amax_values:
        import numpy as np

        amax_arr = np.array(amax_values)
        print(f"\nTotal amax values: {len(amax_values)}")
        print(f"  Mean: {np.mean(amax_arr):.6f}")
        print(f"  Median: {np.median(amax_arr):.6f}")
        print(f"  Std: {np.std(amax_arr):.6f}")
        print(f"  Min: {np.min(amax_arr):.6f}")
        print(f"  Max: {np.max(amax_arr):.6f}")

        # Count by component
        print(f"\nQuantizers by component:")
        components = {}
        for key in cache.keys():
            if "." in key:
                component = key.split(".")[0]
            else:
                component = "other"
            components[component] = components.get(component, 0) + 1

        for comp, count in sorted(components.items()):
            print(f"  {comp}: {count}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="View and compare calibration cache files")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command")

    # View command
    view_parser = subparsers.add_parser("view", help="View calibration cache file")
    view_parser.add_argument("calib_path", help="Path to calibration cache file")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two calibration cache files")
    compare_parser.add_argument("calib_path1", help="Path to first calibration cache file")
    compare_parser.add_argument("calib_path2", help="Path to second calibration cache file")
    compare_parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Threshold for significant difference (default: 0.01 = 1%%)",
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics about calibration cache")
    stats_parser.add_argument("calib_path", help="Path to calibration cache file")

    args = parser.parse_args()

    if args.command == "view":
        view_calib(args.calib_path)
    elif args.command == "compare":
        compare_calib(args.calib_path1, args.calib_path2, args.threshold)
    elif args.command == "stats":
        stats_calib(args.calib_path)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
