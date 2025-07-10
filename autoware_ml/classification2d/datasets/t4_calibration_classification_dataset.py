import json
import os

import numpy as np
from mmcv.transforms import Compose
from mmpretrain.registry import DATASETS
from torch.utils.data import Dataset


@DATASETS.register_module()
class T4CalibrationClassificationDataset(Dataset):
    def __init__(
        self, data_root, split="train", camera_name="CAM_FRONT", pointcloud_folder="LIDAR_CONCAT", pipeline=None
    ):
        self.data_root = data_root
        self.split = split
        self.camera_name = camera_name
        self.pointcloud_folder = pointcloud_folder
        if pipeline is not None and isinstance(pipeline, list):
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = pipeline
        self.samples = []
        self._collect_samples()

    def _collect_samples(self):
        # 遍歷所有 scene
        for uuid in os.listdir(self.data_root):
            uuid_path = os.path.join(self.data_root, uuid)
            if not os.path.isdir(uuid_path):
                continue
            for scene_id in os.listdir(uuid_path):
                scene_path = os.path.join(uuid_path, scene_id)
                if not os.path.isdir(scene_path):
                    continue
                annotation_path = os.path.join(scene_path, "annotation")
                data_path = os.path.join(scene_path, "data")
                if not (os.path.exists(annotation_path) and os.path.exists(data_path)):
                    continue
                # 讀取 calibration 相關 json
                calib_json = os.path.join(annotation_path, "calibrated_sensor.json")
                sensor_json = os.path.join(annotation_path, "sensor.json")
                if not (os.path.exists(calib_json) and os.path.exists(sensor_json)):
                    continue
                with open(calib_json, "r") as f:
                    calib_data = json.load(f)
                with open(sensor_json, "r") as f:
                    sensor_data = json.load(f)
                # 找到 camera_name 的 sensor_token
                cam_token = None
                for s in sensor_data:
                    if s.get("modality", "").lower() == "camera" and s.get("channel", "") == self.camera_name:
                        cam_token = s["token"]
                        break
                if cam_token is None:
                    continue
                # 找到 calibration 參數
                cam_calib = None
                for c in calib_data:
                    if c["sensor_token"] == cam_token:
                        cam_calib = c
                        break
                if cam_calib is None:
                    continue
                # 組 calibration dict
                calibration_dict = {
                    "camera_matrix": np.array(cam_calib["camera_intrinsic"], dtype=np.float32),
                    "distortion_coefficients": np.array(cam_calib["camera_distortion"], dtype=np.float32),
                    "translation": np.array(cam_calib["translation"], dtype=np.float32),
                    "rotation": np.array(cam_calib["rotation"], dtype=np.float32),
                }
                # 遍歷所有影像
                cam_dir = os.path.join(data_path, self.camera_name)
                pc_dir = os.path.join(data_path, self.pointcloud_folder)
                if not (os.path.exists(cam_dir) and os.path.exists(pc_dir)):
                    continue
                for fname in os.listdir(cam_dir):
                    if not fname.endswith(".jpg"):
                        continue
                    frame_id = os.path.splitext(fname)[0]
                    img_path = os.path.join(cam_dir, fname)
                    pc_path = os.path.join(pc_dir, f"{frame_id}.pcd.bin")
                    if not os.path.exists(pc_path):
                        continue
                    self.samples.append(
                        {"img_path": img_path, "pointcloud_path": pc_path, "calibration": calibration_dict}
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.pipeline is not None:
            sample = self.pipeline(sample)
        return sample
