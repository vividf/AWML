"""
YOLOX DataLoader for deployment.

This module implements the BaseDataLoader interface for YOLOX 2D detection
using MMDet's preprocessing pipeline.
"""

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from mmengine.config import Config
from pycocotools.coco import COCO

from autoware_ml.deployment.core import BaseDataLoader
from autoware_ml.deployment.utils import build_test_pipeline


class YOLOXDataLoader(BaseDataLoader):
    """
    DataLoader for YOLOX 2D object detection.

    This loader uses MMDet's preprocessing pipeline to ensure consistency
    between training and deployment.

    Attributes:
        ann_file: Path to COCO annotation file
        img_prefix: Path prefix for images
        pipeline: MMDet preprocessing pipeline
        coco: COCO API instance for annotations
        img_ids: List of image IDs
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
        Initialize YOLOX DataLoader.

        Args:
            ann_file: Path to COCO format annotation file
            img_prefix: Directory path containing images
            model_cfg: Model configuration containing test pipeline
            device: Device to load tensors on ('cpu', 'cuda', etc.)
            use_pipeline: Whether to use MMDet pipeline (True) or simple preprocessing (False)

        Raises:
            FileNotFoundError: If ann_file or img_prefix doesn't exist
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
        if not os.path.exists(img_prefix):
            raise FileNotFoundError(f"Image directory not found: {img_prefix}")

        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.model_cfg = model_cfg
        self.device = device
        self.use_pipeline = use_pipeline

        # Load COCO annotations
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.cat_ids = self.coco.getCatIds()

        # Build preprocessing pipeline
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
        if index >= len(self.img_ids):
            raise IndexError(f"Sample index {index} out of range (0-{len(self.img_ids)-1})")

        # Get image info
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs([img_id])[0]

        # Get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)

        # Convert to MMDet format
        instances = []
        for ann in ann_info:
            # Filter out invalid annotations
            if ann.get("ignore", False):
                continue
            if ann.get("iscrowd", False):
                continue

            # COCO bbox format: [x, y, width, height]
            bbox = ann["bbox"]
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue

            instances.append(
                {
                    "bbox": bbox,  # [x, y, w, h]
                    "bbox_label": self.cat_ids.index(ann["category_id"]),  # Map to 0-based index
                    "ignore_flag": 0,
                }
            )

        return {
            "img_id": img_id,
            "img_path": os.path.join(self.img_prefix, img_info["file_name"]),
            "height": img_info["height"],
            "width": img_info["width"],
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

        return tensor.to(self.device)

    def _preprocess_simple(self, sample: Dict[str, Any]) -> torch.Tensor:
        """
        Simple preprocessing without MMDet pipeline.

        This is a fallback option but may not match training preprocessing exactly.
        """
        import cv2

        # Load image
        img_path = sample["img_path"]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to model input size (e.g., 640x640 for YOLOX)
        input_size = self.model_cfg.get("img_scale", (640, 640))
        img_resized = cv2.resize(img, input_size)

        # Convert to float and normalize
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
        return len(self.img_ids)

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
        Get category names.

        Returns:
            List of category names in order
        """
        cats = self.coco.loadCats(self.cat_ids)
        return [cat["name"] for cat in cats]
