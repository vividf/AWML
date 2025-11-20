"""
CenterPoint-specific TensorRT exporter.

This module provides a specialized exporter for CenterPoint models that need
to export multiple TensorRT engines from multiple ONNX files.
"""

import logging
import os
from typing import Any, Dict, Optional

import torch

from deployment.exporters.base.configs import TensorRTExportConfig
from deployment.exporters.base.tensorrt_exporter import TensorRTExporter


class CenterPointTensorRTExporter(TensorRTExporter):
    """
    Specialized exporter for CenterPoint multi-file TensorRT export.

    Inherits from TensorRTExporter and overrides export() to handle CenterPoint's
    multi-file export requirements.

    CenterPoint generates two ONNX files that need to be converted to TensorRT:
    1. pts_voxel_encoder.onnx → pts_voxel_encoder.engine
    2. pts_backbone_neck_head.onnx → pts_backbone_neck_head.engine
    """

    def __init__(
        self,
        config: TensorRTExportConfig,
        model_wrapper: Optional[Any] = None,
        logger: logging.Logger = None,
    ):
        """
        Initialize CenterPoint TensorRT exporter.

        Args:
            config: TensorRT export configuration dataclass instance.
            model_wrapper: Optional model wrapper class
            logger: Optional logger instance
        """
        super().__init__(config, model_wrapper=model_wrapper, logger=logger)

    def export(
        self,
        model: torch.nn.Module = None,  # Not used, kept for interface compatibility
        sample_input: Any = None,  # Not used, kept for interface compatibility
        output_path: str = None,  # Not used, kept for interface compatibility
        onnx_path: str = None,  # Not used, kept for interface compatibility
        onnx_dir: str = None,
        output_dir: str = None,
        device: str = "cuda:0",
    ) -> None:
        """
        Export CenterPoint ONNX models to TensorRT engines.

        Overrides parent class export() to handle CenterPoint's multi-file export.

        Args:
            model: Not used (kept for interface compatibility)
            sample_input: Not used (kept for interface compatibility)
            output_path: Not used (kept for interface compatibility)
            onnx_path: Not used (kept for interface compatibility)
            onnx_dir: Directory containing ONNX model files
            output_dir: Directory to save TensorRT engines (default: onnx_dir/tensorrt)
            device: Device for TensorRT export

        Raises:
            FileNotFoundError: If required ONNX files are missing
            RuntimeError: If TensorRT conversion fails
        """
        if device is None:
            raise ValueError("CUDA device must be provided for TensorRT export")

        device_id = int(device.split(":", 1)[1])
        torch.cuda.set_device(device_id)
        self.logger.info(f"Using CUDA device: {device}")

        # Use onnx_dir if provided, otherwise fall back to onnx_path
        if onnx_dir is None:
            if onnx_path is None:
                raise ValueError("Either onnx_dir or onnx_path must be provided")
            onnx_dir = os.path.dirname(onnx_path) if os.path.dirname(onnx_path) else "."

        self.logger.info("=" * 80)
        self.logger.info("Exporting CenterPoint to TensorRT (Multi-file)")
        self.logger.info("=" * 80)

        # Determine output directory
        if output_dir is None:
            output_dir = os.path.join(onnx_dir, "tensorrt")
        os.makedirs(output_dir, exist_ok=True)

        self.logger.info(f"ONNX directory: {onnx_dir}")
        self.logger.info(f"TensorRT output directory: {output_dir}")

        # Define the ONNX files to convert
        onnx_files = [
            ("pts_voxel_encoder.onnx", "pts_voxel_encoder.engine"),
            ("pts_backbone_neck_head.onnx", "pts_backbone_neck_head.engine"),
        ]

        success_count = 0
        total_count = len(onnx_files)

        for onnx_file, trt_file in onnx_files:
            onnx_file_path = os.path.join(onnx_dir, onnx_file)
            trt_path = os.path.join(output_dir, trt_file)

            if not os.path.exists(onnx_file_path):
                raise FileNotFoundError(f"ONNX file not found: {onnx_file_path}")

            self.logger.info(f"\nConverting {onnx_file} to TensorRT...")

            # Export to TensorRT using parent class's single-file export method
            try:
                artifact = self._export_single_file(
                    onnx_path=onnx_file_path,
                    output_path=trt_path,
                    sample_input=None,
                )
                self.logger.info(f"✅ TensorRT engine saved: {artifact.path}")
                success_count += 1
            except Exception as exc:
                self.logger.error(f"❌ Failed to convert {onnx_file} to TensorRT")
                raise RuntimeError("CenterPoint TensorRT export failed") from exc

        # Summary
        self.logger.info("\n" + "=" * 80)
        if success_count == total_count:
            self.logger.info(f"✅ All TensorRT engines exported successfully ({success_count}/{total_count})")
            self.logger.info(f"Output directory: {output_dir}")
