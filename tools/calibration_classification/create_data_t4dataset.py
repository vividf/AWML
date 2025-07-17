import argparse
import json
import os
import os.path as osp
import re
import sys
from collections import defaultdict
from typing import Any, Dict

import mmengine
import numpy as np
import yaml
from mmengine.config import Config
from pyquaternion import Quaternion


def load_json(path):
    """Load a JSON file from the given path."""
    with open(path, "r") as f:
        return json.load(f)


def convert_quaternion_to_matrix(rotation, translation):
    """Convert quaternion rotation and translation to a 4x4 transformation matrix.
    Args:
        rotation (list): Quaternion [w, x, y, z].
        translation (list): Translation [x, y, z].
    Returns:
        list: 4x4 transformation matrix as nested lists.
    """

    rot = Quaternion(rotation).rotation_matrix
    trans = np.array(translation).reshape(3, 1)
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3:4] = trans
    return mat.tolist()


def get_pose_dict(ego_pose_json):
    """Create a dict mapping pose token to pose entry from ego_pose_json."""
    return {e["token"]: e for e in ego_pose_json}


def get_calib_dict(calib_json):
    """Create a dict mapping calibration token to calibration entry from calib_json."""
    return {c["token"]: c for c in calib_json}


def get_all_channels(sample_data_json):
    """Return all unique camera channels by parsing filename in sample_data_json."""
    return sorted(set(s["filename"].split("/")[1] for s in sample_data_json if s["filename"].startswith("data/CAM_")))


def extract_frame_index(filename):
    """
    Extract the frame index (all digits before the first dot in the basename).
    E.g. 00001.jpg -> 00001, 123.pcd.bin -> 123, 5.jpg -> 5
    """
    import re

    base = osp.basename(filename)
    # Get the part before the first dot
    before_dot = base.split(".", 1)[0]
    # Find all digits in that part
    digits = re.findall(r"(\d+)", before_dot)
    if digits:
        return digits[0]
    return before_dot  # fallback: just return the part before the first dot


def generate_calib_info(annotation_dir, cam_channels=None, lidar_channel="LIDAR_CONCAT", scene_root=None):
    """
    Generate calibration info for each frame (grouped by filename index).
    Each info contains all sensor sample_data (lidar, cameras, etc) belonging to the same frame.
    """
    sample_data_json = load_json(osp.join(annotation_dir, "sample_data.json"))
    ego_pose_json = load_json(osp.join(annotation_dir, "ego_pose.json"))
    calib_json = load_json(osp.join(annotation_dir, "calibrated_sensor.json"))
    calib_dict = get_calib_dict(calib_json)
    ego_pose_dict = get_pose_dict(ego_pose_json)
    if cam_channels is None:
        cam_channels = get_all_channels(sample_data_json)

    # Group all sample_data by frame index
    frame_groups = defaultdict(list)
    for sd in sample_data_json:
        if "filename" not in sd or not sd["filename"]:
            continue
        frame_idx = extract_frame_index(sd["filename"])
        frame_groups[frame_idx].append(sd)

    infos = []
    for idx, (frame_idx, frame_sample_data) in enumerate(sorted(frame_groups.items())):
        info = build_frame_info(
            frame_idx, frame_sample_data, calib_dict, ego_pose_dict, scene_root, cam_channels, lidar_channel
        )
        if info is not None:
            infos.append(info)
    return infos


def build_frame_info(frame_idx, frame_sample_data, calib_dict, ego_pose_dict, scene_root, cam_channels, lidar_channel):
    info = {
        "frame_idx": frame_idx,
        "images": {},
        "lidar_points": None,
    }
    for sd in frame_sample_data:
        filename = sd["filename"]
        if filename.startswith(f"data/{lidar_channel}/"):
            # lidar
            lidar_calib = calib_dict[sd["calibrated_sensor_token"]]
            lidar_pose = ego_pose_dict[sd["ego_pose_token"]]
            info["lidar_points"] = {
                "lidar_path": osp.join(scene_root, filename),
                "lidar_pose": convert_quaternion_to_matrix(lidar_pose["rotation"], lidar_pose["translation"]),
                "lidar2ego": convert_quaternion_to_matrix(lidar_calib["rotation"], lidar_calib["translation"]),
                "timestamp": sd["timestamp"],
                "sample_data_token": sd["token"],
            }
        else:
            for cam in cam_channels:
                if filename.startswith(f"data/{cam}/"):
                    cam_calib = calib_dict[sd["calibrated_sensor_token"]]
                    cam_pose = ego_pose_dict[sd["ego_pose_token"]]
                    info["images"][cam] = {
                        "img_path": osp.join(scene_root, filename),
                        "cam2img": cam_calib.get("camera_intrinsic"),
                        "cam2ego": convert_quaternion_to_matrix(cam_calib["rotation"], cam_calib["translation"]),
                        "cam_pose": convert_quaternion_to_matrix(cam_pose["rotation"], cam_pose["translation"]),
                        "sample_data_token": sd["token"],
                        "timestamp": sd["timestamp"],
                        "height": sd.get("height"),
                        "width": sd.get("width"),
                    }
    # Fill missing cameras with None values
    for cam in cam_channels:
        if cam not in info["images"]:
            info["images"][cam] = {
                "img_path": None,
                "cam2img": None,
                "cam2ego": None,
                "cam_pose": None,
                "sample_data_token": None,
                "timestamp": None,
                "height": None,
                "width": None,
            }
    return info


def parse_args():
    parser = argparse.ArgumentParser(description="Create calibration info for T4dataset (classification version)")
    parser.add_argument("--config", type=str, required=True, help="config for T4dataset")
    parser.add_argument("--root_path", type=str, required=True, help="specify the root path of dataset")
    parser.add_argument("--version", type=str, required=True, help="product version")
    parser.add_argument("-o", "--out_dir", type=str, required=True, help="output directory of info file")
    parser.add_argument("--lidar_channel", default="LIDAR_CONCAT", help="Lidar channel name (default: LIDAR_CONCAT)")
    parser.add_argument("--cam_channels", nargs="*", default=None, help="Camera channel names (default: all CAM_*)")
    return parser.parse_args()


def get_scene_root_dir_path(root_path, dataset_version, scene_id):
    scene_root_dir_path = osp.join(root_path, dataset_version, scene_id)
    version_dirs = [
        d for d in os.listdir(scene_root_dir_path) if d.isdigit() and osp.isdir(osp.join(scene_root_dir_path, d))
    ]
    if version_dirs:
        version_id = sorted(version_dirs, key=int)[-1]
        return osp.join(scene_root_dir_path, version_id)
    return scene_root_dir_path


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    abs_root_path = osp.abspath(args.root_path)

    split_infos = {"train": [], "val": [], "test": []}
    for dataset_version in cfg.dataset_version_list:
        dataset_list = osp.join(cfg.dataset_version_config_root, dataset_version + ".yaml")
        with open(dataset_list, "r") as f:
            dataset_list_dict = yaml.safe_load(f)
        for split in ["train", "val", "test"]:
            for scene_id in dataset_list_dict.get(split, []):
                scene_root = get_scene_root_dir_path(args.root_path, dataset_version, scene_id)
                annotation_dir = osp.join(scene_root, "annotation")
                print(
                    f"[DEBUG] split={split}, scene_id={scene_id}, annotation_dir={annotation_dir}, exists={osp.isdir(annotation_dir)}"
                )
                if not osp.isdir(annotation_dir):
                    print(f"[WARN] Annotation dir not found: {annotation_dir}, skip.")
                    continue
                print(f"[INFO] Generating calibration info for {scene_id} ({split}) ...")
                rel_scene_root = osp.relpath(scene_root, abs_root_path)
                scene_infos = generate_calib_info(
                    annotation_dir, args.cam_channels, args.lidar_channel, rel_scene_root
                )
                split_infos[split].extend(scene_infos)
    # Save per split
    metainfo = dict(version=args.version)
    for split in ["train", "val", "test"]:
        out_path = osp.join(args.out_dir, f"t4dataset_{args.version}_calib_infos_{split}.pkl")
        mmengine.dump(dict(data_list=split_infos[split], metainfo=metainfo), out_path)
        print(f"[INFO] Saved {len(split_infos[split])} samples to {out_path}")


if __name__ == "__main__":
    main()
