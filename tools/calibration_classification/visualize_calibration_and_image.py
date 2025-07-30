#!/usr/bin/env python3
import argparse
import os
import pickle

# Import the transform and dataset classes
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append("/home/yihsiangfang/ml_workspace/AWML")

# Import mmpretrain registry to ensure transforms are registered
from mmpretrain.registry import TRANSFORMS

# Import the transform module to ensure it's registered
import autoware_ml.calibration_classification.datasets.transforms.calibration_classification_transform
from autoware_ml.calibration_classification.datasets.t4_calibration_classification_dataset import (
    T4CalibrationClassificationDataset,
)
from autoware_ml.calibration_classification.datasets.transforms.calibration_classification_transform import (
    CalibrationClassificationTransform,
)


def load_info_pkl(info_pkl_path):
    """Load info.pkl file and return the data."""
    with open(info_pkl_path, "rb") as f:
        return pickle.load(f)


def visualize_using_transform_directly(info_pkl_path, output_dir=None, indices=None, data_root=None):
    """Visualize using CalibrationClassificationTransform directly without dataset."""

    # Load info.pkl data
    info_data = load_info_pkl(info_pkl_path)

    if not info_data:
        print("[WARNING] No data found in info.pkl")
        return

    # Handle different data formats
    if isinstance(info_data, dict):
        # Check if it's the new format with data_list
        if "data_list" in info_data:
            samples_list = info_data["data_list"]
            print(f"[INFO] Loaded {len(samples_list)} samples from info.pkl (data_list format)")
        elif "samples" in info_data:
            samples_list = info_data["samples"]
            print(f"[INFO] Loaded {len(samples_list)} samples from info.pkl (samples format)")
        elif "data" in info_data:
            samples_list = info_data["data"]
            print(f"[INFO] Loaded {len(samples_list)} samples from info.pkl (data format)")
        else:
            # Assume the dict contains sample data directly
            samples_list = [info_data]
            print(f"[INFO] Loaded 1 sample from info.pkl (direct dict format)")
    elif isinstance(info_data, list):
        samples_list = info_data
        print(f"[INFO] Loaded {len(samples_list)} samples from info.pkl (list format)")
    else:
        print(f"[ERROR] Unexpected data format: {type(info_data)}")
        return

    # Debug: Check data structure
    if len(samples_list) > 0:
        print(f"[DEBUG] First sample type: {type(samples_list[0])}")
        if isinstance(samples_list[0], dict):
            print(f"[DEBUG] First sample keys: {list(samples_list[0].keys())}")
            if "image" in samples_list[0]:
                print(f"[DEBUG] Image data type: {type(samples_list[0]['image'])}")
                if isinstance(samples_list[0]["image"], dict):
                    print(f"[DEBUG] Image keys: {list(samples_list[0]['image'].keys())}")
        else:
            print(f"[DEBUG] First sample content: {samples_list[0]}")

    # Initialize transform directly
    transform = CalibrationClassificationTransform(
        mode="test",
        undistort=True,
        data_root=data_root,  # Add data_root parameter
        projection_vis_dir=output_dir,
        results_vis_dir=output_dir,
        enable_augmentation=False,
    )

    # Process samples
    if indices is not None:
        samples_to_process = [samples_list[i] for i in indices if i < len(samples_list)]
    else:
        samples_to_process = samples_list

    print(f"[INFO] Processing {len(samples_to_process)} samples")

    for i, sample in enumerate(samples_to_process):
        try:
            # Debug: Print sample structure
            print(f"[DEBUG] Processing sample {i+1}, type: {type(sample)}")
            if isinstance(sample, dict):
                print(f"[DEBUG] Sample keys: {list(sample.keys())}")
                if "image" in sample:
                    print(f"[DEBUG] Sample has 'image' key")
                    print(f"[DEBUG] Image keys: {list(sample['image'].keys())}")
                else:
                    print(f"[DEBUG] Sample does NOT have 'image' key")
                    print(f"[DEBUG] Available keys: {list(sample.keys())}")
            else:
                print(f"[DEBUG] Sample is not a dict: {sample}")

            result = transform(sample)
            print(f"[INFO] Successfully processed sample {i+1}/{len(samples_to_process)}")
        except Exception as e:
            print(f"[ERROR] Failed to process sample {i+1}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"[INFO] Finished processing all samples")


def visualize_single_sample_directly(info_pkl_path, sample_idx=0, output_path=None, data_root=None):
    """Visualize a single sample from info.pkl using transform directly."""

    # Load info.pkl data
    info_data = load_info_pkl(info_pkl_path)

    if not info_data:
        print("[WARNING] No data found in info.pkl")
        return None

    # Handle different data formats
    if isinstance(info_data, dict):
        if "data_list" in info_data:
            samples_list = info_data["data_list"]
        elif "samples" in info_data:
            samples_list = info_data["samples"]
        elif "data" in info_data:
            samples_list = info_data["data"]
        else:
            samples_list = [info_data]
    elif isinstance(info_data, list):
        samples_list = info_data
    else:
        print(f"[ERROR] Unexpected data format: {type(info_data)}")
        return None

    if sample_idx >= len(samples_list):
        print(f"[ERROR] Sample index {sample_idx} out of range (0-{len(samples_list)-1})")
        return None

    print(f"[INFO] Processing sample {sample_idx} from {len(samples_list)} total samples")

    # Initialize transform directly
    transform = CalibrationClassificationTransform(
        mode="test",
        undistort=True,
        data_root=data_root,  # Add data_root parameter
        projection_vis_dir=os.path.dirname(output_path) if output_path else None,
        results_vis_dir=os.path.dirname(output_path) if output_path else None,
        enable_augmentation=False,
    )

    # Process the single sample
    try:
        result = transform(samples_list[sample_idx])
        print(f"[INFO] Successfully processed sample {sample_idx}")

        if output_path:
            print(f"[INFO] Visualizations saved to directory: {os.path.dirname(output_path)}")

        return result

    except Exception as e:
        print(f"[ERROR] Failed to process sample {sample_idx}: {e}")
        return None


def main():
    """Main entry point for the script. Parses arguments and runs the appropriate visualization."""
    parser = argparse.ArgumentParser(description="Visualize LiDAR points projected on camera images using info.pkl")
    parser.add_argument("--info_pkl", required=True, help="Path to info.pkl file")
    parser.add_argument("--output_dir", help="Output directory for saving visualizations")
    parser.add_argument("--data_root", help="Root directory for data files")
    parser.add_argument("--sample_idx", type=int, help="Specific sample index to process (0-based)")
    parser.add_argument("--indices", nargs="+", type=int, help="Specific sample indices to process (0-based)")
    parser.add_argument(
        "--show_point_details", action="store_true", help="Show detailed point cloud field information"
    )

    args = parser.parse_args()

    if not os.path.exists(args.info_pkl):
        raise FileNotFoundError(f"Info.pkl file not found: {args.info_pkl}")

    if args.sample_idx is not None:
        # Process single sample
        output_path = None
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, f"sample_{args.sample_idx}.jpg")

        visualize_single_sample_directly(args.info_pkl, args.sample_idx, output_path, args.data_root)

    else:
        # Process all samples or specific indices
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)

        visualize_using_transform_directly(args.info_pkl, args.output_dir, args.indices, args.data_root)


if __name__ == "__main__":
    main()
