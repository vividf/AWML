import argparse
import json
import os
import os.path as osp
import sys
from typing import Any, Dict

import mmengine
import numpy as np
import yaml
from mmengine.config import Config


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
    from pyquaternion import Quaternion

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


def build_sample_info(idx, sample_data_token, sample_data, ego_pose_dict, calib_dict, cam_channels, lidar_channel):
    """Build a dictionary containing sample information for a given sample_data_token.
    Args:
        idx (int): Index of the sample.
        sample_data_token (str): Token for the sample data.
        sample_data (dict): Mapping from token to sample data entry.
        ego_pose_dict (dict): Mapping from token to ego pose entry.
        calib_dict (dict): Mapping from token to calibration entry.
        cam_channels (list): List of camera channel names.
        lidar_channel (str): Lidar channel name.
    Returns:
        dict: Sample information dictionary.
    """
    s = sample_data[sample_data_token]
    pose = ego_pose_dict[s["ego_pose_token"]]
    calib = calib_dict[s["calibrated_sensor_token"]]
    info = {
        "sample_idx": idx,
        "token": s["sample_token"],
        "timestamp": s["timestamp"],
        "ego2global": convert_quaternion_to_matrix(pose["rotation"], pose["translation"]),
        "images": {},
        "lidar_points": None,
    }
    # Images
    for cam in cam_channels:
        cam_sample = next((x for x in sample_data.values() if f"data/{cam}/" in x.get("filename", "")), None)
        if cam_sample:
            cam_calib = calib_dict[cam_sample["calibrated_sensor_token"]]
            cam_pose = ego_pose_dict[cam_sample["ego_pose_token"]]
            info["images"][cam] = {
                "img_path": cam_sample["filename"],
                "cam2img": cam_calib.get("camera_intrinsic"),
                "cam2ego": convert_quaternion_to_matrix(cam_pose["rotation"], cam_pose["translation"]),
                "sample_data_token": cam_sample["token"],
                "timestamp": cam_sample["timestamp"],
                "lidar2cam": None,  # Optional: can be filled if needed
                "height": None,
                "width": None,
                "depth_map": None,
            }
        else:
            info["images"][cam] = {
                "img_path": None,
                "cam2img": None,
                "cam2ego": None,
                "sample_data_token": None,
                "timestamp": None,
                "lidar2cam": None,
                "height": None,
                "width": None,
                "depth_map": None,
            }
    # Lidar
    lidar_sample = next(
        (x for x in sample_data.values() if x.get("filename", "").startswith(f"data/{lidar_channel}/")), None
    )
    if lidar_sample:
        lidar_calib = calib_dict[lidar_sample["calibrated_sensor_token"]]
        info["lidar_points"] = {
            "num_pts_feats": 5,
            "lidar_path": lidar_sample["filename"],
            "lidar2ego": convert_quaternion_to_matrix(lidar_calib["rotation"], lidar_calib["translation"]),
        }
    return info


def generate_calib_info(annotation_dir, cam_channels=None, lidar_channel="LIDAR_CONCAT"):
    """Generate calibration info for a single scene annotation directory.
    Args:
        annotation_dir (str): Directory containing annotation JSON files.
        cam_channels (list, optional): List of camera channels to include. If None, all channels are used.
        lidar_channel (str, optional): Lidar channel name. Default is 'LIDAR_CONCAT'.
    Returns:
        list: List of sample info dicts for this scene.
    """
    sample_data_json = load_json(osp.join(annotation_dir, "sample_data.json"))
    ego_pose_json = load_json(osp.join(annotation_dir, "ego_pose.json"))
    calib_json = load_json(osp.join(annotation_dir, "calibrated_sensor.json"))
    sample_data = {s["token"]: s for s in sample_data_json}
    ego_pose_dict = get_pose_dict(ego_pose_json)
    calib_dict = get_calib_dict(calib_json)
    if cam_channels is None:
        cam_channels = get_all_channels(sample_data_json)
    lidar_tokens = [s["token"] for s in sample_data_json if s.get("filename", "").startswith(f"data/{lidar_channel}/")]
    infos = []
    for idx, token in enumerate(lidar_tokens):
        info = build_sample_info(idx, token, sample_data, ego_pose_dict, calib_dict, cam_channels, lidar_channel)
        infos.append(info)
    return infos


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
                scene_infos = generate_calib_info(annotation_dir, args.cam_channels, args.lidar_channel)
                split_infos[split].extend(scene_infos)
    # Save per split
    metainfo = dict(version=args.version)
    for split in ["train", "val", "test"]:
        out_path = osp.join(args.out_dir, f"t4dataset_{args.version}_calib_infos_{split}.pkl")
        mmengine.dump(dict(data_list=split_infos[split], metainfo=metainfo), out_path)
        print(f"[INFO] Saved {len(split_infos[split])} samples to {out_path}")


if __name__ == "__main__":
    main()
