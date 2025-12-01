"""
CenterPoint TensorRT export workflow using composition.

This workflow orchestrates multi-file TensorRT export for CenterPoint models.
It converts multiple ONNX files to TensorRT engines.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional, Tuple

import torch

from deployment.core import Artifact, BaseDataLoader, BaseDeploymentConfig
from deployment.exporters.centerpoint.constants import ONNX_TO_TRT_MAPPINGS
from deployment.exporters.common.factory import ExporterFactory
from deployment.exporters.workflows.base import TensorRTExportWorkflow


class CenterPointTensorRTExportWorkflow(TensorRTExportWorkflow):
    """
    CenterPoint TensorRT export workflow.

    Converts CenterPoint ONNX files to multiple TensorRT engines:
    - pts_voxel_encoder.onnx → pts_voxel_encoder.engine
    - pts_backbone_neck_head.onnx → pts_backbone_neck_head.engine

    Uses TENSORRT_FILE_MAPPINGS from constants module for file name configuration.
    """

    # Pattern for validating CUDA device strings
    _CUDA_DEVICE_PATTERN = re.compile(r"^cuda:\d+$")

    def __init__(
        self,
        exporter_factory: type[ExporterFactory],
        config: BaseDeploymentConfig,
        logger: Optional[logging.Logger] = None,
        file_mappings: Optional[Tuple[Tuple[str, str], ...]] = None,
    ):
        """
        Initialize CenterPoint TensorRT export workflow.

        Args:
            exporter_factory: Factory class for creating exporters
            config: Deployment configuration
            logger: Optional logger instance
            file_mappings: Optional tuple of (onnx_file, engine_file) pairs.
                          Defaults to ONNX_TO_TRT_MAPPINGS from exporter constants.
        """
        self.exporter_factory = exporter_factory
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.file_mappings = file_mappings or ONNX_TO_TRT_MAPPINGS

    def _validate_cuda_device(self, device: str) -> int:
        """
        Validate CUDA device string and extract device ID.

        Args:
            device: Device string (expected format: "cuda:N")

        Returns:
            Device ID as integer

        Raises:
            ValueError: If device format is invalid
        """
        if not self._CUDA_DEVICE_PATTERN.match(device):
            raise ValueError(
                f"Invalid CUDA device format: '{device}'. " f"Expected format: 'cuda:N' (e.g., 'cuda:0', 'cuda:1')"
            )
        return int(device.split(":")[1])

    def export(
        self,
        *,
        onnx_path: str,
        output_dir: str,
        config: BaseDeploymentConfig,
        device: str,
        data_loader: BaseDataLoader,
    ) -> Artifact:
        """
        Export CenterPoint ONNX files to TensorRT engines.

        Args:
            onnx_path: Path to directory containing ONNX files
            output_dir: Output directory for TensorRT engines
            config: Deployment configuration (not used, kept for interface)
            device: CUDA device string (e.g., "cuda:0")
            data_loader: Data loader (not used for TensorRT)

        Returns:
            Artifact pointing to output directory with multi_file=True

        Raises:
            ValueError: If device format is invalid or onnx_path is not a directory
            FileNotFoundError: If ONNX files are missing
            RuntimeError: If TensorRT conversion fails
        """
        onnx_dir = onnx_path

        # Validate inputs
        if device is None:
            raise ValueError("CUDA device must be provided for TensorRT export")

        if onnx_dir is None:
            raise ValueError("onnx_dir must be provided for CenterPoint TensorRT export")

        if not os.path.isdir(onnx_dir):
            raise ValueError(f"onnx_path must be a directory for multi-file export, got: {onnx_dir}")

        # Validate and set CUDA device
        device_id = self._validate_cuda_device(device)
        torch.cuda.set_device(device_id)
        self.logger.info(f"Using CUDA device: {device}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Convert each ONNX file to TensorRT using configured file mappings
        num_files = len(self.file_mappings)
        for i, (onnx_file, trt_file) in enumerate(self.file_mappings, 1):
            onnx_file_path = os.path.join(onnx_dir, onnx_file)
            trt_path = os.path.join(output_dir, trt_file)

            # Validate ONNX file exists
            if not os.path.exists(onnx_file_path):
                raise FileNotFoundError(f"ONNX file not found: {onnx_file_path}")

            self.logger.info(f"\n[{i}/{num_files}] Converting {onnx_file} to TensorRT...")

            # Create fresh exporter (no caching)
            exporter = self.exporter_factory.create_tensorrt_exporter(config=self.config, logger=self.logger)

            # Export to TensorRT
            try:
                artifact = exporter.export(
                    model=None,  # Not needed for TensorRT conversion
                    sample_input=None,  # Shape info from config
                    output_path=trt_path,
                    onnx_path=onnx_file_path,
                )
                self.logger.info(f"TensorRT engine saved: {artifact.path}")
            except Exception as exc:
                self.logger.error(f"Failed to convert {onnx_file} to TensorRT")
                raise RuntimeError(f"TensorRT export failed for {onnx_file}") from exc

        self.logger.info(f"\nAll TensorRT engines exported successfully to {output_dir}")
        return Artifact(path=output_dir, multi_file=True)
