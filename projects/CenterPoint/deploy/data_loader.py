"""
CenterPoint DataLoader for deployment.

This module implements the BaseDataLoader interface for CenterPoint 3D detection
using MMDet3D's preprocessing pipeline.
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from mmengine.config import Config

from deployment.core import BaseDataLoader, build_preprocessing_pipeline


class CenterPointDataLoader(BaseDataLoader):
    """
    DataLoader for CenterPoint 3D object detection.

    This loader uses MMDet3D's preprocessing pipeline to ensure consistency
    between training and deployment.

    Attributes:
        info_file: Path to info.pkl file containing dataset information
        pipeline: MMDet3D preprocessing pipeline
        data_infos: List of data information dictionaries
    """

    def __init__(
        self,
        info_file: str,
        model_cfg: Config,
        device: str = "cpu",
        task_type: Optional[str] = None,
    ):
        """
        Initialize CenterPoint DataLoader.

        Args:
            info_file: Path to info.pkl file (e.g., centerpoint_infos_val.pkl)
            model_cfg: Model configuration containing test pipeline
            device: Device to load tensors on ('cpu', 'cuda', etc.)
            task_type: Task type for pipeline building. If None, will try to get from
                      model_cfg.task_type or model_cfg.deploy.task_type.

        Raises:
            FileNotFoundError: If info_file doesn't exist
            ValueError: If info file format is invalid
        """
        super().__init__(
            config={
                "info_file": info_file,
                "device": device,
            }
        )

        # Validate info file
        if not os.path.exists(info_file):
            raise FileNotFoundError(f"Info file not found: {info_file}")

        self.info_file = info_file
        self.model_cfg = model_cfg
        self.device = device

        # Load info.pkl
        self.data_infos = self._load_info_file()

        # Build preprocessing pipeline
        # task_type should be provided from deploy_config
        self.pipeline = build_preprocessing_pipeline(model_cfg, task_type=task_type)

    def _to_tensor(
        self,
        data: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
        name: str = "data",
    ) -> torch.Tensor:
        """
        Convert various data types to a torch.Tensor on the target device.

        Args:
            data: Input data (torch.Tensor, np.ndarray, or list of either)
            name: Name of the data for error messages

        Returns:
            torch.Tensor on self.device

        Raises:
            ValueError: If data type is unsupported or list is empty
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device)

        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self.device)

        if isinstance(data, list):
            if len(data) == 0:
                raise ValueError(f"Empty list for '{name}' in pipeline output.")

            first_item = data[0]
            if isinstance(first_item, torch.Tensor):
                return first_item.to(self.device)
            if isinstance(first_item, np.ndarray):
                return torch.from_numpy(first_item).to(self.device)

            raise ValueError(
                f"Unexpected type for {name}[0]: {type(first_item)}. " f"Expected torch.Tensor or np.ndarray."
            )

        raise ValueError(
            f"Unexpected type for '{name}': {type(data)}. "
            f"Expected torch.Tensor, np.ndarray, or list of tensors/arrays."
        )

    def _load_info_file(self) -> list:
        """
        Load and parse info.pkl file.

        Returns:
            List of data information dictionaries

        Raises:
            ValueError: If file format is invalid
        """
        try:
            with open(self.info_file, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load info file: {e}") from e

        # Extract data_list
        if isinstance(data, dict):
            if "data_list" in data:
                data_list = data["data_list"]
            elif "infos" in data:
                # Alternative key name
                data_list = data["infos"]
            else:
                raise ValueError(
                    f"Expected 'data_list' or 'infos' key in info file, " f"found keys: {list(data.keys())}"
                )
        elif isinstance(data, list):
            data_list = data
        else:
            raise ValueError(f"Unexpected info file format: {type(data)}")

        if not data_list:
            raise ValueError("No samples found in info file")

        return data_list

    def load_sample(self, index: int) -> Dict[str, Any]:
        """
        Load a single sample with point cloud and annotations.

        Args:
            index: Sample index to load

        Returns:
            Dictionary containing:
            - lidar_points: Dict with lidar_path
            - gt_bboxes_3d: 3D bounding boxes (if available)
            - gt_labels_3d: 3D labels (if available)
            - Additional metadata

        Raises:
            IndexError: If index is out of range
        """
        if index >= len(self.data_infos):
            raise IndexError(f"Sample index {index} out of range (0-{len(self.data_infos)-1})")

        info = self.data_infos[index]

        # Extract lidar points info
        lidar_points = info.get("lidar_points", {})
        if not lidar_points:
            # Try alternative key
            lidar_path = info.get("lidar_path", info.get("velodyne_path", ""))
            lidar_points = {"lidar_path": lidar_path}

        # Add data_root to lidar_path if it's relative
        if "lidar_path" in lidar_points and not lidar_points["lidar_path"].startswith("/"):
            # Get data_root from model config
            data_root = getattr(self.model_cfg, "data_root", "data/t4dataset/")
            # Ensure data_root ends with '/'
            if not data_root.endswith("/"):
                data_root += "/"
            # Check if the path already starts with data_root to avoid duplication
            if not lidar_points["lidar_path"].startswith(data_root):
                lidar_points["lidar_path"] = data_root + lidar_points["lidar_path"]

        # Extract annotations (if available)
        instances = info.get("instances", [])

        sample = {
            "lidar_points": lidar_points,
            "sample_idx": info.get("sample_idx", index),
            "timestamp": info.get("timestamp", 0),  # Add timestamp for pipeline
        }

        # Add ground truth if available
        if instances:
            # Extract 3D bounding boxes and labels from instances
            gt_bboxes_3d = []
            gt_labels_3d = []

            for instance in instances:
                if "bbox_3d" in instance and "bbox_label_3d" in instance:
                    # Check if bbox is valid
                    if instance.get("bbox_3d_isvalid", True):
                        gt_bboxes_3d.append(instance["bbox_3d"])
                        gt_labels_3d.append(instance["bbox_label_3d"])

            if gt_bboxes_3d:
                sample["gt_bboxes_3d"] = np.array(gt_bboxes_3d, dtype=np.float32)
                sample["gt_labels_3d"] = np.array(gt_labels_3d, dtype=np.int64)

        # Add camera info if available (for multi-modal models)
        if "images" in info or "img_path" in info:
            sample["images"] = info.get("images", {})
            if "img_path" in info:
                sample["img_path"] = info["img_path"]

        return sample

    def preprocess(self, sample: Dict[str, Any]) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Preprocess using MMDet3D pipeline.

        For CenterPoint, the test pipeline typically outputs only 'points' (not voxelized).
        Voxelization is performed by the model's data_preprocessor during inference.
        This method returns the point cloud tensor for use by the deployment pipeline.

        Args:
            sample: Sample dictionary from load_sample()

        Returns:
            Dictionary containing:
            - points: Point cloud tensor [N, point_features] (typically [N, 5] for x, y, z, intensity, timestamp)

        Raises:
            ValueError: If pipeline output format is unexpected
        """
        # Apply pipeline
        results = self.pipeline(sample)

        # Validate expected format (MMDet3D 3.x format)
        if "inputs" not in results:
            raise ValueError(
                f"Expected 'inputs' key in pipeline results (MMDet3D 3.x format). "
                f"Found keys: {list(results.keys())}. "
                f"Please ensure your test pipeline includes Pack3DDetInputs transform."
            )

        pipeline_inputs = results["inputs"]

        # For CenterPoint, pipeline should output 'points' (voxelization happens in data_preprocessor)
        if "points" not in pipeline_inputs:
            available_keys = list(pipeline_inputs.keys())
            raise ValueError(
                f"Expected 'points' key in pipeline inputs for CenterPoint. "
                f"Available keys: {available_keys}. "
                f"Note: For CenterPoint, voxelization is performed by the model's data_preprocessor, "
                f"not in the test pipeline. The pipeline should output raw points using Pack3DDetInputs."
            )

        # Convert points to tensor using helper
        points_tensor = self._to_tensor(pipeline_inputs["points"], name="points")

        # Validate points shape
        if points_tensor.ndim != 2:
            raise ValueError(f"Expected points tensor with shape [N, point_features], got shape {points_tensor.shape}")

        return {"points": points_tensor}

    def get_num_samples(self) -> int:
        """
        Get total number of samples.

        Returns:
            Number of samples in the dataset
        """
        return len(self.data_infos)

    def get_ground_truth(self, index: int) -> Dict[str, Any]:
        """
        Get ground truth annotations for evaluation.

        Args:
            index: Sample index

        Returns:
            Dictionary containing:
            - gt_bboxes_3d: 3D bounding boxes (N, 7) where 7 = (x, y, z, w, l, h, yaw)
            - gt_labels_3d: 3D class labels (N,)
            - sample_idx: Sample identifier
        """
        sample = self.load_sample(index)

        gt_bboxes_3d = sample.get("gt_bboxes_3d", np.zeros((0, 7), dtype=np.float32))
        gt_labels_3d = sample.get("gt_labels_3d", np.zeros((0,), dtype=np.int64))

        # Convert to numpy if needed
        if isinstance(gt_bboxes_3d, (list, tuple)):
            gt_bboxes_3d = np.array(gt_bboxes_3d, dtype=np.float32)
        if isinstance(gt_labels_3d, (list, tuple)):
            gt_labels_3d = np.array(gt_labels_3d, dtype=np.int64)

        return {
            "gt_bboxes_3d": gt_bboxes_3d,
            "gt_labels_3d": gt_labels_3d,
            "sample_idx": sample.get("sample_idx", index),
        }

    def get_class_names(self) -> List[str]:
        """
        Get class names from config.

        Returns:
            List of class names

        Raises:
            ValueError: If class_names not found in model_cfg
        """
        if hasattr(self.model_cfg, "class_names"):
            return self.model_cfg.class_names

        raise ValueError(
            "class_names must be defined in model_cfg. "
            "Check your model config file includes class_names definition."
        )
