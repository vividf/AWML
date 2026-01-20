#!/usr/bin/env python
"""
Analyze why calibration results differ even with all samples.

This script investigates the root cause of calibration differences
when using sequential vs random order with all calibration samples.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def analyze_differences(calib_path1: str, calib_path2: str, threshold: float = 0.01):
    """Analyze differences between two calibration caches."""
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

    print("=" * 100)
    print("Calibration Difference Analysis")
    print("=" * 100)
    print(f"File 1: {calib_path1.name}")
    print(f"File 2: {calib_path2.name}")
    print("=" * 100)

    # Get common keys
    common_keys = set(cache1.keys()) & set(cache2.keys())
    print(f"\nTotal quantizers: {len(common_keys)}")

    # Analyze differences
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
            continue

        # Calculate differences
        abs_diff = abs(amax1_val - amax2_val)
        if amax1_val == 0:
            rel_diff = float("inf") if amax2_val != 0 else 0.0
        else:
            rel_diff = abs_diff / abs(amax1_val)

        differences.append(
            {
                "key": key,
                "amax1": amax1_val,
                "amax2": amax2_val,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
            }
        )

    # Sort by relative difference
    differences.sort(key=lambda x: x["rel_diff"], reverse=True)

    # Print summary
    print(f"\nDifferences Summary:")
    print(
        f"  Mean relative difference: {np.mean([d['rel_diff'] for d in differences]):.4f} ({np.mean([d['rel_diff'] for d in differences]) * 100:.2f}%)"
    )
    print(
        f"  Median relative difference: {np.median([d['rel_diff'] for d in differences]):.4f} ({np.median([d['rel_diff'] for d in differences]) * 100:.2f}%)"
    )
    print(
        f"  Max relative difference: {np.max([d['rel_diff'] for d in differences]):.4f} ({np.max([d['rel_diff'] for d in differences]) * 100:.2f}%)"
    )
    print(
        f"  Std relative difference: {np.std([d['rel_diff'] for d in differences]):.4f} ({np.std([d['rel_diff'] for d in differences]) * 100:.2f}%)"
    )

    # Count significant differences
    significant = [d for d in differences if d["rel_diff"] > threshold]
    print(f"\nSignificant differences (> {threshold * 100:.1f}%): {len(significant)}/{len(differences)}")

    # Group by component
    print(f"\nDifferences by Component:")
    components = {}
    for d in significant:
        if "." in d["key"]:
            component = d["key"].split(".")[0]
        else:
            component = "other"
        if component not in components:
            components[component] = []
        components[component].append(d)

    for comp in sorted(components.keys()):
        comp_diffs = components[comp]
        max_rel_diff = max(d["rel_diff"] for d in comp_diffs)
        mean_rel_diff = np.mean([d["rel_diff"] for d in comp_diffs])
        print(
            f"  {comp:30s}: {len(comp_diffs):3d} quantizers, max={max_rel_diff*100:5.2f}%, mean={mean_rel_diff*100:5.2f}%"
        )

    # Print top differences
    print(f"\nTop 10 Largest Differences:")
    print("-" * 100)
    print(f"{'Quantizer Name':<70} {'File 1':>12} {'File 2':>12} {'Rel Diff %':>10}")
    print("-" * 100)
    for d in differences[:10]:
        print(f"{d['key']:<70} {d['amax1']:>12.6f} {d['amax2']:>12.6f} {d['rel_diff']*100:>9.2f}%")
    print("-" * 100)

    # Analysis of why this happens
    print(f"\n" + "=" * 100)
    print("Root Cause Analysis")
    print("=" * 100)
    print(
        """
Even when using ALL calibration samples, differences can occur due to:

1. **Histogram Accumulation Order Sensitivity**:
   - pytorch-quantization's HistogramCalibrator accumulates statistics incrementally
   - Different sample orders lead to different accumulation paths
   - Floating-point arithmetic is not perfectly associative: (a+b)+c ≠ a+(b+c)
   - This causes tiny differences in histogram bins

2. **MSE Optimization Sensitivity**:
   - MSE method finds optimal amax by minimizing quantization error
   - Small histogram differences can shift the optimal amax value
   - The optimization is sensitive to histogram shape, not just final counts

3. **Multi-Process DataLoader (num_workers > 0)**:
   - With num_workers=32, different workers may load samples in different orders
   - Even with seed set, worker initialization order can affect data loading
   - This introduces non-determinism in sample processing order

4. **Histogram Binning Edge Cases**:
   - Values near bin boundaries may be assigned differently
   - Accumulation order affects which bin gets the value
   - This is especially true for values exactly at bin edges

Recommendations:
- Use num_workers=0 for deterministic calibration (slower but reproducible)
- Consider using 'max' method instead of 'mse' for more stable results
- Or accept that small differences (< 1%) are normal and use sequential order
- Sequential order generally gives better results anyway (as you observed)
    """
    )

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Analyze calibration cache differences and root causes")
    parser.add_argument("calib_path1", help="Path to first calibration cache file")
    parser.add_argument("calib_path2", help="Path to second calibration cache file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Threshold for significant difference (default: 0.01 = 1%%)",
    )

    args = parser.parse_args()
    analyze_differences(args.calib_path1, args.calib_path2, args.threshold)


if __name__ == "__main__":
    main()
