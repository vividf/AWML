"""
YOLOX_opt_elan DataLoader for deployment.

This module implements the BaseDataLoader interface for YOLOX_opt_elan object detection
using MMDet's preprocessing pipeline.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from mmengine.config import Config

from deployment.core import BaseDataLoader, build_preprocessing_pipeline


class YOLOXOptElanDataLoader(BaseDataLoader):
    """
    DataLoader for YOLOX_opt_elan object detection.

    This loader uses MMDet's preprocessing pipeline to ensure consistency
    between training and deployment. It handles T4Dataset format info files.

    Attributes:
        info_file: Path to T4Dataset format info file
        pipeline: MMDet preprocessing pipeline
        data_list: List of data samples
        classes: List of class names
    """

    def __init__(
        self,
        info_file: str,
        model_cfg: Config,
        device: str = "cpu",
        task_type: Optional[str] = None,
    ):
        """
        Initialize YOLOX_opt_elan DataLoader.

        Args:
            info_file: Path to T4Dataset info file (e.g., yolox_infos_val.json or yolox_infos_val.pkl)
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

        # Validate path
        if not os.path.exists(info_file):
            raise FileNotFoundError(f"Info file not found: {info_file}")

        self.info_file = info_file
        self.model_cfg = model_cfg
        self.device = device

        # Load T4Dataset format info file (supports both JSON and PKL)
        if info_file.endswith(".pkl"):
            import pickle

            with open(info_file, "rb") as f:
                ann_data = pickle.load(f)
        else:
            # Assume JSON format
            with open(info_file, "r") as f:
                ann_data = json.load(f)

        self.metainfo = ann_data.get("metainfo", {})
        self.classes = self.metainfo.get("classes", [])
        self.data_list = ann_data.get("data_list", [])

        # Build preprocessing pipeline from model config
        # task_type should be provided from deploy_config
        self.pipeline = build_preprocessing_pipeline(model_cfg, task_type=task_type)

    def load_sample(self, index: int) -> Dict[str, Any]:
        """
        Load a single sample with image and annotations.

        Args:
            index: Sample index to load

        Returns:
            Dictionary containing:
            - img_id: Image ID
            - img_path: Full path to image file
            - height: Image height
            - width: Image width
            - instances: List of instance annotations with bbox and label

        Raises:
            IndexError: If index is out of range
        """
        if index >= len(self.data_list):
            raise IndexError(f"Sample index {index} out of range (0-{len(self.data_list)-1})")

        # Get sample from T4Dataset data_list
        data_info = self.data_list[index]

        # T4Dataset format already contains all needed info
        img_id = data_info.get("img_id", index)
        img_path = data_info["img_path"]

        # Get instances - T4Dataset format already has bbox and bbox_label
        instances = []
        for inst in data_info.get("instances", []):
            if inst.get("ignore_flag", 0):
                continue

            # T4Dataset bbox format: [x1, y1, x2, y2], keep as [x1, y1, x2, y2] to match PackDetInputs
            bbox = inst["bbox"]
            # Keep original format [x1, y1, x2, y2] to match PackDetInputs behavior

            instances.append(
                {
                    "bbox": bbox,
                    "bbox_label": inst["bbox_label"],
                    "ignore_flag": inst.get("ignore_flag", 0),
                }
            )

        return {
            "img_id": img_id,
            "img_path": img_path,
            "height": data_info["height"],
            "width": data_info["width"],
            "instances": instances,
            "scale_factor": self._calculate_scale_factor(data_info["height"], data_info["width"]),
        }

    def preprocess(self, sample: Dict[str, Any]) -> torch.Tensor:
        """
        Preprocess using MMDet pipeline.

        This ensures consistency with training preprocessing.
        Expects MMDet 3.x format with 'inputs' key containing image tensor.

        Args:
            sample: Sample dictionary from load_sample()

        Returns:
            Preprocessed image tensor with shape (1, C, H, W) and dtype float32

        Raises:
            ValueError: If pipeline output format is unexpected
        """
        # Apply pipeline
        results = self.pipeline(sample)

        # Validate expected format (MMDet 3.x format)
        if "inputs" not in results:
            raise ValueError(
                f"Expected 'inputs' key in pipeline results (MMDet 3.x format). " f"Found keys: {list(results.keys())}"
            )

        inputs = results["inputs"]

        # Convert to tensor if needed
        if isinstance(inputs, torch.Tensor):
            tensor = inputs
        elif isinstance(inputs, np.ndarray):
            tensor = torch.from_numpy(inputs)
        else:
            raise ValueError(f"Unexpected type for 'inputs': {type(inputs)}. " f"Expected torch.Tensor or np.ndarray.")

        # Ensure batch dimension
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 4:
            raise ValueError(
                f"Expected tensor with 3 or 4 dimensions (C, H, W) or (B, C, H, W), " f"got shape {tensor.shape}"
            )

        # Convert to float32 if still in uint8 (ByteTensor)
        # This is crucial for ONNX export and model inference
        if tensor.dtype == torch.uint8:
            tensor = tensor.float()

        return tensor.to(self.device)

    def _calculate_scale_factor(self, orig_height: int, orig_width: int) -> List[float]:
        """
        Calculate scale factor for coordinate transformation.

        This matches MMDetection's Resize transform behavior.
        """
        # Model input size is 960x960 for YOLOX_opt_elan
        target_size = (960, 960)

        # Calculate scale factors (same as MMDetection Resize with keep_ratio=True)
        scale_w = target_size[1] / orig_width
        scale_h = target_size[0] / orig_height

        # Use the smaller scale to maintain aspect ratio
        scale = min(scale_w, scale_h)

        # Return scale factor in format [scale_w, scale_h, scale_w, scale_h]
        # This matches MMDetection's PackDetInputs format
        return [scale, scale, scale, scale]

    def get_num_samples(self) -> int:
        """
        Get total number of samples.

        Returns:
            Number of images in the dataset
        """
        return len(self.data_list)

    def get_ground_truth(self, index: int) -> Dict[str, Any]:
        """
        Get ground truth annotations for evaluation.

        Args:
            index: Sample index

        Returns:
            Dictionary containing:
            - gt_bboxes: List of bounding boxes [[x, y, w, h], ...]
            - gt_labels: List of class labels
            - img_info: Image metadata
        """
        sample = self.load_sample(index)

        gt_bboxes = [inst["bbox"] for inst in sample["instances"]]
        gt_labels = [inst["bbox_label"] for inst in sample["instances"]]

        return {
            "gt_bboxes": np.array(gt_bboxes, dtype=np.float32) if gt_bboxes else np.zeros((0, 4), dtype=np.float32),
            "gt_labels": np.array(gt_labels, dtype=np.int64) if gt_labels else np.zeros((0,), dtype=np.int64),
            "img_info": {
                "img_id": sample["img_id"],
                "height": sample["height"],
                "width": sample["width"],
                "scale_factor": sample["scale_factor"],
            },
        }

    def get_category_names(self) -> list:
        """
        Get category names for detected objects.

        Returns:
            List of category names in order
        """
        return self.classes
