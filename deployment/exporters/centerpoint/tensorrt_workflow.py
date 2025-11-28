"""
CenterPoint TensorRT export workflow using composition.

This workflow orchestrates multi-file TensorRT export for CenterPoint models.
It converts multiple ONNX files to TensorRT engines.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from deployment.core import Artifact, BaseDataLoader, BaseDeploymentConfig
from deployment.core.contexts import ExportContext
from deployment.exporters.common.factory import ExporterFactory
from deployment.exporters.workflows.base import TensorRTExportWorkflow


class CenterPointTensorRTExportWorkflow(TensorRTExportWorkflow):
    """
    CenterPoint TensorRT export workflow.

    Converts CenterPoint ONNX files to multiple TensorRT engines:
    - pts_voxel_encoder.onnx → pts_voxel_encoder.engine
    - pts_backbone_neck_head.onnx → pts_backbone_neck_head.engine
    """

    def __init__(
        self,
        exporter_factory: type[ExporterFactory],
        config: BaseDeploymentConfig,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize CenterPoint TensorRT export workflow.

        Args:
            exporter_factory: Factory class for creating exporters
            config: Deployment configuration
            logger: Optional logger instance
        """
        self.exporter_factory = exporter_factory
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def export(
        self,
        *,
        onnx_path: str,
        output_dir: str,
        config: BaseDeploymentConfig,
        device: str,
        data_loader: BaseDataLoader,
        context: Optional[ExportContext] = None,
    ) -> Artifact:
        """
        Export CenterPoint ONNX files to TensorRT engines.

        Args:
            onnx_path: Path to directory containing ONNX files
            output_dir: Output directory for TensorRT engines
            config: Deployment configuration (not used, kept for interface)
            device: CUDA device string (e.g., "cuda:0")
            data_loader: Data loader (not used for TensorRT)
            context: Export context with project-specific parameters (currently unused,
                     but available for future extensions)

        Returns:
            Artifact pointing to output directory with multi_file=True
        """
        # context available for future extensions
        _ = context
        onnx_dir = onnx_path

        # Validate inputs
        if device is None:
            raise ValueError("CUDA device must be provided for TensorRT export")

        if onnx_dir is None:
            raise ValueError("onnx_dir must be provided for CenterPoint TensorRT export")

        if not os.path.isdir(onnx_dir):
            raise ValueError(f"onnx_path must be a directory for multi-file export, got: {onnx_dir}")

        # Set CUDA device
        device_id = int(device.split(":", 1)[1])
        torch.cuda.set_device(device_id)
        self.logger.info(f"Using CUDA device: {device}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Define ONNX → TensorRT file pairs
        onnx_files = [
            ("pts_voxel_encoder.onnx", "pts_voxel_encoder.engine"),
            ("pts_backbone_neck_head.onnx", "pts_backbone_neck_head.engine"),
        ]

        # Convert each ONNX file to TensorRT
        for i, (onnx_file, trt_file) in enumerate(onnx_files, 1):
            onnx_file_path = os.path.join(onnx_dir, onnx_file)
            trt_path = os.path.join(output_dir, trt_file)

            # Validate ONNX file exists
            if not os.path.exists(onnx_file_path):
                raise FileNotFoundError(f"ONNX file not found: {onnx_file_path}")

            self.logger.info(f"\n[{i}/{len(onnx_files)}] Converting {onnx_file} to TensorRT...")

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
