import json
import os

import mmengine
import numpy as np
from mmcv.transforms import Compose
from mmpretrain.registry import DATASETS
from torch.utils.data import Dataset


@DATASETS.register_module()
class T4CalibrationClassificationDataset(Dataset):
    """
    Dataset class for T4 Calibration Classification. Loads image and point cloud samples with calibration info.
    Args:
        data_root (str): Root directory of the dataset.
        ann_file (str): Path to the annotation file (optional).
        split (str): Dataset split (train/val/test).
        camera_name (str): Name of the camera channel.
        pointcloud_folder (str): Name of the pointcloud folder.
        pipeline (list or callable): Data processing pipeline.
    """

    def __init__(
        self,
        data_root=None,
        ann_file=None,
        split="train",
        camera_name="CAM_FRONT",
        pointcloud_folder="LIDAR_CONCAT",
        pipeline=None,
    ):
        self.data_root = data_root
        self.ann_file = ann_file
        self.split = split
        self.camera_name = camera_name
        self.pointcloud_folder = pointcloud_folder
        if pipeline is not None and isinstance(pipeline, list):
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = pipeline
        self.samples = []
        if self.ann_file is not None:
            self.samples = mmengine.load(self.ann_file)
        else:
            self._collect_samples()

    def _collect_samples(self):
        """
        Iterate through all scenes and collect samples with calibration information.
        """
        # Iterate through all scenes
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
                # Read calibration related json files
                calib_json = os.path.join(annotation_path, "calibrated_sensor.json")
                sensor_json = os.path.join(annotation_path, "sensor.json")
                if not (os.path.exists(calib_json) and os.path.exists(sensor_json)):
                    continue
                with open(calib_json, "r") as f:
                    calib_data = json.load(f)
                with open(sensor_json, "r") as f:
                    sensor_data = json.load(f)
                # Find sensor_token for camera_name
                cam_token = None
                for s in sensor_data:
                    if s.get("modality", "").lower() == "camera" and s.get("channel", "") == self.camera_name:
                        cam_token = s["token"]
                        break
                if cam_token is None:
                    continue
                # Find calibration parameters
                cam_calib = None
                for c in calib_data:
                    if c["sensor_token"] == cam_token:
                        cam_calib = c
                        break
                if cam_calib is None:
                    continue
                # Build calibration dict
                calibration_dict = {
                    "camera_matrix": np.array(cam_calib["camera_intrinsic"], dtype=np.float32),
                    "distortion_coefficients": np.array(cam_calib["camera_distortion"], dtype=np.float32),
                    "translation": np.array(cam_calib["translation"], dtype=np.float32),
                    "rotation": np.array(cam_calib["rotation"], dtype=np.float32),
                }
                # Iterate through all images
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
        """
        Returns the number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample by index, applying the pipeline if provided.
        Args:
            idx (int): Index of the sample.
        Returns:
            dict: Sample dictionary with image path, pointcloud path, and calibration info.
        """
        sample = self.samples[idx]
        if self.pipeline is not None:
            sample = self.pipeline(sample)
        return sample
