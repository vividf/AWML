import argparse
import json
import os
import os.path as osp

import numpy as np
import yaml
from mmengine import dump
from mmengine.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Create data info for Calibration Classification T4dataset")
    parser.add_argument("--config", type=str, required=True, help="config for T4dataset")
    parser.add_argument("--root_path", type=str, required=True, help="specify the root path of dataset")
    parser.add_argument("--version", type=str, required=True, help="product version")
    parser.add_argument("-o", "--out_dir", type=str, required=True, help="output directory of info file")
    return parser.parse_args()


def collect_samples(scene_root, camera_name, lidar_folder):
    annotation_path = osp.join(scene_root, "annotation")
    data_path = osp.join(scene_root, "data")
    calib_json = osp.join(annotation_path, "calibrated_sensor.json")
    sensor_json = osp.join(annotation_path, "sensor.json")
    if not (osp.exists(calib_json) and osp.exists(sensor_json)):
        return []
    with open(calib_json, "r") as f:
        calib_data = json.load(f)
    with open(sensor_json, "r") as f:
        sensor_data = json.load(f)
    # Find camera sensor_token
    cam_token = None
    for s in sensor_data:
        if s.get("modality", "").lower() == "camera" and s.get("channel", "") == camera_name:
            cam_token = s["token"]
            break
    if cam_token is None:
        return []
    # Find lidar sensor_token
    lidar_token = None
    for s in sensor_data:
        if s.get("modality", "").lower() == "lidar" and s.get("channel", "") == lidar_folder:
            lidar_token = s["token"]
            break
    if lidar_token is None:
        # fallback: just use the first lidar
        for s in sensor_data:
            if s.get("modality", "").lower() == "lidar":
                lidar_token = s["token"]
                break
    if lidar_token is None:
        return []
    # Find camera calibration
    cam_calib = None
    for c in calib_data:
        if c["sensor_token"] == cam_token:
            cam_calib = c
            break
    if cam_calib is None:
        return []
    # Find lidar calibration
    lidar_calib = None
    for c in calib_data:
        if c["sensor_token"] == lidar_token:
            lidar_calib = c
            break
    if lidar_calib is None:
        return []
    calibration_dict = {
        "camera_matrix": np.array(cam_calib["camera_intrinsic"], dtype=np.float32),
        "distortion_coefficients": np.array(cam_calib["camera_distortion"], dtype=np.float32),
        "camera": {
            "rotation": np.array(cam_calib["rotation"], dtype=np.float32),
            "translation": np.array(cam_calib["translation"], dtype=np.float32),
        },
        "lidar": {
            "rotation": np.array(lidar_calib["rotation"], dtype=np.float32),
            "translation": np.array(lidar_calib["translation"], dtype=np.float32),
        },
    }
    cam_dir = osp.join(data_path, camera_name)
    pc_dir = osp.join(data_path, lidar_folder)
    if not (osp.exists(cam_dir) and osp.exists(pc_dir)):
        return []
    samples = []
    for fname in sorted(os.listdir(cam_dir)):
        if not fname.endswith(".jpg"):
            continue
        frame_id = os.path.splitext(fname)[0]
        img_path = osp.join(cam_dir, fname)
        pc_path = osp.join(pc_dir, f"{frame_id}.pcd.bin")
        if not osp.exists(pc_path):
            continue
        samples.append({"img_path": img_path, "pointcloud_path": pc_path, "calibration": calibration_dict})
    return samples


def find_scene_root(scene_root):
    # 如果 scene_root 下有純數字的子目錄（如 0, 1, 2...），取最大那個
    version_dirs = [d for d in os.listdir(scene_root) if d.isdigit() and os.path.isdir(os.path.join(scene_root, d))]
    if version_dirs:
        version_id = sorted(version_dirs, key=int)[-1]
        return os.path.join(scene_root, version_id)
    return scene_root


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
                scene_root = osp.join(args.root_path, dataset_version, scene_id)
                if not osp.isdir(scene_root):
                    continue
                real_scene_root = find_scene_root(scene_root)
                samples = collect_samples(real_scene_root, cfg.camera_types[0], cfg.lidar_folder)
                split_infos[split].extend(samples)
    for split in ["train", "val", "test"]:
        out_path = osp.join(args.out_dir, f"t4dataset_{args.version}_calib_infos_{split}.pkl")
        dump(split_infos[split], out_path)
        print(f"Saved {len(split_infos[split])} samples to {out_path}")


if __name__ == "__main__":
    main()
