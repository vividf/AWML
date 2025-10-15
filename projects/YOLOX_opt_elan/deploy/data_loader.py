"""
YOLOX_opt_elan DataLoader for deployment.

This module implements the BaseDataLoader interface for YOLOX_opt_elan object detection
using MMDet's preprocessing pipeline.
"""

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from mmengine.config import Config

from autoware_ml.deployment.core import BaseDataLoader
from autoware_ml.deployment.utils import build_test_pipeline


class YOLOXOptElanDataLoader(BaseDataLoader):
    """
    DataLoader for YOLOX_opt_elan object detection.

    This loader uses MMDet's preprocessing pipeline to ensure consistency
    between training and deployment. It handles T4Dataset format annotations.

    Attributes:
        ann_file: Path to T4Dataset format annotation file
        img_prefix: Path prefix for images
        pipeline: MMDet preprocessing pipeline
        data_list: List of data samples
        classes: List of class names
    """

    def __init__(
        self,
        ann_file: str,
        img_prefix: str,
        model_cfg: Config,
        device: str = "cpu",
        use_pipeline: bool = True,
    ):
        """
        Initialize YOLOX_opt_elan DataLoader.

        Args:
            ann_file: Path to T4Dataset annotation file (e.g., 2d_info_infos_val.json)
            img_prefix: Directory path containing images (can be empty if full paths in annotations)
            model_cfg: Model configuration containing test pipeline
            device: Device to load tensors on ('cpu', 'cuda', etc.)
            use_pipeline: Whether to use MMDet pipeline (True) or simple preprocessing (False)

        Raises:
            FileNotFoundError: If ann_file doesn't exist
            ValueError: If annotation file is invalid
        """
        super().__init__(
            config={
                "ann_file": ann_file,
                "img_prefix": img_prefix,
                "device": device,
                "use_pipeline": use_pipeline,
            }
        )

        # Validate paths
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
        # img_prefix can be empty for T4Dataset as full paths might be in annotations
        if img_prefix and not os.path.exists(img_prefix):
            raise FileNotFoundError(f"Image directory not found: {img_prefix}")

        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.model_cfg = model_cfg
        self.device = device
        self.use_pipeline = use_pipeline

        # Load T4Dataset format annotations
        with open(ann_file, "r") as f:
            ann_data = json.load(f)

        self.metainfo = ann_data.get("metainfo", {})
        self.classes = self.metainfo.get("classes", [])
        self.data_list = ann_data.get("data_list", [])

        # Build preprocessing pipeline from model config
        if self.use_pipeline:
            self.pipeline = build_test_pipeline(model_cfg)
        else:
            self.pipeline = None

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

        # Handle img_path - ensure it's absolute or relative to img_prefix
        if not os.path.isabs(img_path) and self.img_prefix:
            img_path = os.path.join(self.img_prefix, img_path)

        # Get instances - T4Dataset format already has bbox and bbox_label
        instances = []
        for inst in data_info.get("instances", []):
            if inst.get("ignore_flag", 0):
                continue

            # T4Dataset bbox format: [x1, y1, x2, y2], convert to [x, y, w, h]
            bbox = inst["bbox"]
            x1, y1, x2, y2 = bbox
            bbox_xywh = [x1, y1, x2 - x1, y2 - y1]

            instances.append(
                {
                    "bbox": bbox_xywh,
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
        }

    def preprocess(self, sample: Dict[str, Any]) -> torch.Tensor:
        """
        Preprocess sample using MMDet pipeline or simple preprocessing.

        Args:
            sample: Raw sample data from load_sample()

        Returns:
            Preprocessed tensor with shape (1, C, H, W) for batch size 1

        Raises:
            ValueError: If sample format is invalid
        """
        if self.use_pipeline:
            return self._preprocess_with_pipeline(sample)
        else:
            return self._preprocess_simple(sample)

    def _preprocess_with_pipeline(self, sample: Dict[str, Any]) -> torch.Tensor:
        """
        Preprocess using MMDet pipeline (recommended).

        This ensures consistency with training preprocessing.
        """
        # Apply pipeline
        results = self.pipeline(sample)

        # Extract model input
        # MMDet 3.x uses 'inputs' key
        if "inputs" in results:
            inputs = results["inputs"]
        elif "img" in results:
            # Fallback for older versions
            inputs = results["img"]
        else:
            raise ValueError(f"Unexpected pipeline output keys: {results.keys()}")

        # Convert to tensor if needed
        if isinstance(inputs, torch.Tensor):
            tensor = inputs
        else:
            tensor = torch.from_numpy(inputs)

        # Ensure batch dimension
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        # Convert to float32 if still in uint8 (ByteTensor)
        # This is crucial for ONNX export and model inference
        if tensor.dtype == torch.uint8:
            tensor = tensor.float()

        return tensor.to(self.device)

    def _preprocess_simple(self, sample: Dict[str, Any]) -> torch.Tensor:
        """
        Simple preprocessing without MMDet pipeline.

        This is a fallback option but may not match training preprocessing exactly.
        Uses 960x960 input size for YOLOX_opt_elan.
        """
        import cv2

        # Load image
        img_path = sample["img_path"]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to model input size (960x960 for YOLOX_opt_elan)
        input_size = (960, 960)
        img_resized = cv2.resize(img, input_size)

        # Pad to square (already square but keeping consistency)
        # Normalize using standard ImageNet mean/std or keep as is
        img_float = img_resized.astype(np.float32)

        # Convert to tensor (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(img_float).permute(2, 0, 1)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def get_num_samples(self) -> int:
        """
        Get total number of samples.

        Returns:
            Number of images in the dataset
        """
        return len(self.data_list)

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validate sample structure.

        Args:
            sample: Sample to validate

        Returns:
            True if valid, False otherwise
        """
        required_keys = ["img_id", "img_path", "height", "width", "instances"]
        if not all(key in sample for key in required_keys):
            return False

        # Validate image file exists
        if not os.path.exists(sample["img_path"]):
            return False

        return True

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
            },
        }

    def get_category_names(self) -> list:
        """
        Get category names for detected objects.

        Returns:
            List of category names in order
        """
        return self.classes
