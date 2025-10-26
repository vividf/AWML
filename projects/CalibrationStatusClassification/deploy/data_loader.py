"""
CalibrationStatusClassification DataLoader for deployment.

This module implements the BaseDataLoader interface for loading and preprocessing
calibration status classification data from info.pkl files.
"""

import os
import pickle
from typing import Any, Dict, Optional

import torch
from mmengine.config import Config

from autoware_ml.calibration_classification.datasets.transforms.calibration_classification_transform import (
    CalibrationClassificationTransform,
)
from autoware_ml.deployment.core import BaseDataLoader


class CalibrationDataLoader(BaseDataLoader):
    """
    DataLoader for CalibrationStatusClassification task.

    Loads samples from info.pkl files and preprocesses them using
    CalibrationClassificationTransform.
    """

    def __init__(
        self,
        info_pkl_path: str,
        model_cfg: Config,
        miscalibration_probability: float = 0.0,
        device: str = "cpu",
    ):
        """
        Initialize CalibrationDataLoader.

        Args:
            info_pkl_path: Path to info.pkl file containing samples
            model_cfg: Model configuration containing transform settings
            miscalibration_probability: Probability of loading miscalibrated sample (0.0 or 1.0)
            device: Device to load tensors on
        """
        super().__init__(
            config={
                "info_pkl_path": info_pkl_path,
                "miscalibration_probability": miscalibration_probability,
                "device": device,
            }
        )

        self.info_pkl_path = info_pkl_path
        self.model_cfg = model_cfg
        self.miscalibration_probability = miscalibration_probability
        self.device = device

        # Load samples list
        self._samples_list = self._load_info_pkl_file()

        # Create transform
        self._transform = self._create_transform()

    def _load_info_pkl_file(self) -> list:
        """
        Load and parse info.pkl file.

        Returns:
            List of samples from data_list
        """
        if not os.path.exists(self.info_pkl_path):
            raise FileNotFoundError(f"Info.pkl file not found: {self.info_pkl_path}")

        try:
            with open(self.info_pkl_path, "rb") as f:
                info_data = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load info.pkl file: {e}")

        # Extract samples from info.pkl
        if isinstance(info_data, dict):
            if "data_list" in info_data:
                samples_list = info_data["data_list"]
            else:
                raise ValueError(f"Expected 'data_list' key in info_data, " f"found keys: {list(info_data.keys())}")
        else:
            raise ValueError(f"Expected dict format, got {type(info_data)}")

        if not samples_list:
            raise ValueError("No samples found in info.pkl")

        return samples_list

    def _create_transform(self) -> CalibrationClassificationTransform:
        """
        Create CalibrationClassificationTransform with model configuration.

        Returns:
            Configured transform instance
        """
        data_root = self.model_cfg.get("data_root")
        if data_root is None:
            raise ValueError("data_root not found in model configuration")

        transform_config = self.model_cfg.get("transform_config")
        if transform_config is None:
            raise ValueError("transform_config not found in model configuration")

        return CalibrationClassificationTransform(
            transform_config=transform_config,
            mode="test",
            max_depth=self.model_cfg.get("max_depth", 128.0),
            dilation_size=self.model_cfg.get("dilation_size", 1),
            undistort=True,
            miscalibration_probability=self.miscalibration_probability,
            enable_augmentation=False,
            data_root=data_root,
            projection_vis_dir=self.model_cfg.get("test_projection_vis_dir", None),
            results_vis_dir=self.model_cfg.get("test_results_vis_dir", None),
            binary_save_dir=self.model_cfg.get("binary_save_dir", None),
        )

    def load_sample(self, index: int) -> Dict[str, Any]:
        """
        Load a single sample from info.pkl.

        Args:
            index: Sample index to load

        Returns:
            Sample dictionary with 'image', 'lidar_points', etc.
        """
        if index >= len(self._samples_list):
            raise IndexError(f"Sample index {index} out of range (0-{len(self._samples_list)-1})")

        sample = self._samples_list[index]

        # Validate sample structure
        required_keys = ["image", "lidar_points"]
        if not all(key in sample for key in required_keys):
            raise ValueError(f"Sample {index} has invalid structure. " f"Required keys: {required_keys}")

        return sample

    def preprocess(self, sample: Dict[str, Any]) -> torch.Tensor:
        """
        Preprocess sample using CalibrationClassificationTransform.

        Args:
            sample: Raw sample data from load_sample()

        Returns:
            Preprocessed tensor with shape (1, C, H, W)
        """
        # Apply transform
        results = self._transform.transform(sample)
        input_data_processed = results["fused_img"]  # (H, W, 5)

        # Convert numpy array (H, W, C) to tensor (1, C, H, W)
        tensor = torch.from_numpy(input_data_processed).permute(2, 0, 1).float()
        return tensor.unsqueeze(0).to(self.device)

    def get_num_samples(self) -> int:
        """
        Get total number of samples.

        Returns:
            Number of samples in info.pkl
        """
        return len(self._samples_list)

