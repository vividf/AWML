import json
import os
import re
import warnings
from argparse import ArgumentParser

import numpy as np
import yaml
from mmdet3d.datasets.utils import convert_quaternion_to_matrix
from mmengine.config import Config
from t4_devkit import Tier4
from t4_devkit.schema import Sample

from autoware_ml.registry import DATA_SELECTOR
from tools.detection3d.create_data_t4dataset import get_lidar_token
from tools.detection3d.t4dataset_converters.t4converter import extract_tier4_data, obtain_sensor2top

DEFAULT_T4_CAMERA = ["CAM_FRONT", "CAM_BACK"]


class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


from typing import Any, Dict, List, Optional

import numpy as np


def get_input_info(
    t4: Tier4,
    sample: Sample,
    camera_types: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Retrieves image paths and calibration data for specified camera types,
    along with the point cloud data path from a T4 sample.

    Args:
        t4 (Sample): The T4 dataset object.
        sample (Dict[str, Any]): The sample data containing sensor tokens.
        camera_types (List[str]): List of camera sensor types to include.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing image and point cloud paths
        with calibration data, or None if lidar data is unavailable.
    """
    # Obtain the lidar token from the sample
    lidar_token = get_lidar_token(sample)
    if lidar_token is None:
        print(f"Sample {sample.token} doesn't have lidar data.")
        return None

    # Extract necessary transformation matrices and translations
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        ego_to_global_rot,
        lidar_to_ego_rot,
        ego_to_global_trans,
        lidar_to_ego_trans,
    ) = extract_tier4_data(t4, sample, lidar_token)

    # Initialize the information dictionary
    info: Dict[str, Any] = {}

    # Process each specified camera type
    for cam in camera_types:
        if cam in sample.data:
            cam_token = sample.data[cam]
            # Retrieve camera data
            image_path, _, cam_intrinsic = t4.get_sample_data(cam_token)

            # Obtain camera-to-top (global) transformation info
            cam_info = obtain_sensor2top(
                t4,
                cam_token,
                lidar_to_ego_trans,
                lidar_to_ego_rot,
                ego_to_global_trans,
                ego_to_global_rot,
                cam,
            )
            cam_info["cam_intrinsic"] = cam_intrinsic

            # Initialize camera info dictionary
            info[cam] = {}
            info[cam]["img_path"] = image_path
            info[cam]["cam2img"] = cam_intrinsic.tolist()
            info[cam]["sample_data_token"] = cam_info["sample_data_token"]
            # Adjust timestamp to seconds
            info[cam]["timestamp"] = cam_info["timestamp"] / 1e6
            # Compute camera-to-ego transformation matrix
            info[cam]["cam2ego"] = convert_quaternion_to_matrix(
                cam_info["sensor2ego_rotation"],
                cam_info["sensor2ego_translation"],
            )

            # Compute the lidar-to-camera transformation matrix
            lidar_to_camera = np.eye(4)
            rotation = cam_info["sensor2lidar_rotation"]
            translation = cam_info["sensor2lidar_translation"]
            lidar_to_camera[:3, :3] = rotation.T
            lidar_to_camera[:3, 3] = -np.dot(rotation.T, translation)
            info[cam]["lidar2cam"] = lidar_to_camera.astype(np.float32).tolist()

    # Retrieve the point cloud data path
    points_path, _, _ = t4.get_sample_data(lidar_token)

    return {"images": info, "points": points_path}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Config file")
    parser.add_argument(
        "--dataset-configs",
        type=str,
        required=True,
        nargs="+",
        help="One or more YAML dataset configuration files.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/t4dataset",
        help="Root directory for datasets.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory of images or prediction results.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=np.inf,
        help="Maximum number of samples to select from the provided yaml file",
    )
    parser.add_argument(
        "--true-ratio",
        type=float,
        default=0.1,
        help="Rate of timestamps for which is_target_scene is True",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Name of the experiment, used for output files and symbolic links.",
    )
    parser.add_argument(
        "--create-symbolic-links",
        action="store_true",
        help="If passed, creates symbolic links under {root-dir}/{experiment_name} for datasets in train, val, and test. So, you can start training directly with generated yml",
    )
    parser.add_argument(
        "--show-visualization",
        action="store_true",
        help="If passed, creates a folder and stores visualizations under {out-dir}/visualize/{dataset_id}.",
    )

    args = parser.parse_args()
    return args


def load_yaml_files(root_path, file_paths):
    """Load multiple YAML files and return combined train, val, test lists."""
    train_list, val_list, test_list = [], [], []
    for file_path in file_paths:
        dir_name = os.path.split(file_path)[1].split(".")[0]
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            for did in data.get("train", []):
                train_list.append(get_scene_root_dir_path(root_path, dir_name, did))
            for did in data.get("val", []):
                val_list.append(get_scene_root_dir_path(root_path, dir_name, did))
            for did in data.get("test", []):
                test_list.append(get_scene_root_dir_path(root_path, dir_name, did))
    return train_list, val_list, test_list


def get_scene_root_dir_path(
    root_path: str,
    dataset_version: str,
    scene_id: str,
) -> str:
    version_pattern = re.compile(r"^\d+$")
    scene_root_dir_path = os.path.join(root_path, dataset_version, scene_id)
    version_dirs = [d for d in os.listdir(scene_root_dir_path) if version_pattern.match(d)]

    if version_dirs:
        version_id = sorted(version_dirs, key=int)[-1]
        return os.path.join(scene_root_dir_path, version_id)
    else:
        warnings.warn(
            f"The directory structure of T4 Dataset is deprecated. Please update to the latest version.",
            DeprecationWarning,
        )
        return scene_root_dir_path


def main():
    args = parse_args()

    # Load the config file
    cfg = Config.fromfile(args.config)
    scene_selector = DATA_SELECTOR.build(cfg.scene_selector)

    # Load datasets
    if args.dataset_configs:
        print(f"Only train datasets will be used for sampling.")
        train_list, val_list, test_list = load_yaml_files(args.data_root, args.dataset_configs)
        print(f"Found {len(train_list)} sequences to sample from.")
    else:
        print("No dataset config files provided.")
        exit(0)

    # Select scenarios from train data
    selected_scenarios = []
    scene_metadata = {}  # this can be used for analysis later

    np.random.shuffle(train_list)
    for dataset_path in train_list:
        dataset_id = dataset_path.split("/")[-2]
        t4_info = Tier4(data_root=dataset_path, verbose=False)
        sensor_data = []
        for sample in t4_info.sample:
            sensor_data.append(get_input_info(t4_info, sample, cfg.camera_types))

        # Handle visualization output path
        if args.show_visualization:
            visualize_dir = os.path.join(
                args.out_dir,
                "visualize",
                dataset_id,
            )
            os.makedirs(visualize_dir, exist_ok=True)
            results_path = visualize_dir
        else:
            results_path = ""

        is_target, metadata = scene_selector.is_target_scene(
            sensor_data,
            return_counts=True,
            results_path=results_path,
        )
        target_scene_ratio = np.mean(is_target)
        scene_metadata[dataset_id] = {"target_scene_ratio": target_scene_ratio, "metadata": metadata}

        if target_scene_ratio > args.true_ratio:
            selected_scenarios.append(dataset_path)
            print(f"{dataset_id} sampled @ {target_scene_ratio}")
        else:
            print(f"{dataset_id} WAS NOT sampled @ {target_scene_ratio}")

        if len(selected_scenarios) >= args.max_samples:
            break

    # Save selected scenarios along with test and val datasets
    output_data = {
        "train": [path.split("/")[-2] for path in selected_scenarios],
        "val": [path.split("/")[-2] for path in val_list],
        "test": [path.split("/")[-2] for path in test_list],
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{args.experiment_name}.yaml"), "w") as outfile:
        yaml.dump(output_data, outfile)
    with open(os.path.join(args.out_dir, f"{args.experiment_name}_object_counts.json"), "w") as outfile:
        json.dump(scene_metadata, outfile, indent=4, cls=NpEncoder)

    # Optionally create symbolic links
    if args.create_symbolic_links:
        experiment_dir = os.path.join(args.data_root, args.experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)

        for dataset_path in selected_scenarios + val_list + test_list:
            src = os.path.abspath(os.path.split(dataset_path)[0])
            os.symlink(
                src,
                os.path.join(experiment_dir, os.path.basename(src)),
                target_is_directory=True,
            )


if __name__ == "__main__":
    main()
