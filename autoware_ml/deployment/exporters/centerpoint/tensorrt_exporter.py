"""
CenterPoint-specific TensorRT exporter.

This module provides a specialized exporter for CenterPoint models that need
to export multiple TensorRT engines from multiple ONNX files.
"""

import logging
import os
from typing import Any, Dict, Optional

import torch

from autoware_ml.deployment.exporters.base.tensorrt_exporter import TensorRTExporter


class CenterPointTensorRTExporter(TensorRTExporter):
    """
    Specialized exporter for CenterPoint multi-file TensorRT export.

    Inherits from TensorRTExporter and overrides export() to handle CenterPoint's
    multi-file export requirements.

    CenterPoint generates two ONNX files that need to be converted to TensorRT:
    1. pts_voxel_encoder.onnx → pts_voxel_encoder.engine
    2. pts_backbone_neck_head.onnx → pts_backbone_neck_head.engine
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger = None, model_wrapper: Optional[Any] = None):
        """
        Initialize CenterPoint TensorRT exporter.

        Args:
            config: TensorRT export configuration
            logger: Optional logger instance
            model_wrapper: Optional model wrapper class (usually not needed for TensorRT)
        """
        super().__init__(config, logger, model_wrapper=model_wrapper)

    def export(
        self,
        model: torch.nn.Module = None,  # Not used, kept for interface compatibility
        sample_input: torch.Tensor = None,  # Not used, kept for interface compatibility
        output_path: str = None,  # Not used, kept for interface compatibility
        onnx_path: str = None,  # Not used, kept for interface compatibility
        onnx_dir: str = None,
        output_dir: str = None,
        device: str = "cuda:0",
    ) -> bool:
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

        Returns:
            True if all exports succeeded
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
                self.logger.warning(f"ONNX file not found: {onnx_file_path}")
                continue

            self.logger.info(f"\nConverting {onnx_file} to TensorRT...")

            # TODO(vividf): check this
            # Create dummy sample input for shape configuration
            # For CenterPoint, we need different sample inputs for each component
            if "voxel_encoder" in onnx_file:
                # Voxel encoder input: (num_voxels, num_max_points, point_dim)
                # Use realistic voxel counts for T4Dataset - actual shape is (num_voxels, 32, 11)
                sample_input = torch.randn(10000, 32, 11, device=device)
            else:
                # Backbone/neck/head input: (batch_size, channels, height, width)
                # Use realistic spatial feature dimensions - actual shape is (batch_size, 32, H, W)
                # NOTE: Actual evaluation data can produce up to 760x760, so use 800x800 for max_shape
                sample_input = torch.randn(1, 32, 200, 200, device=device)

            # Export to TensorRT using parent class method
            success = super().export(
                model=None,  # Not used for TensorRT
                sample_input=sample_input,
                output_path=trt_path,
                onnx_path=onnx_file_path,
            )

            if success:
                self.logger.info(f"✅ TensorRT engine saved: {trt_path}")
                success_count += 1
            else:
                self.logger.error(f"❌ Failed to convert {onnx_file} to TensorRT")

        # Summary
        self.logger.info("\n" + "=" * 80)
        if success_count == total_count:
            self.logger.info(f"✅ All TensorRT engines exported successfully ({success_count}/{total_count})")
            self.logger.info(f"Output directory: {output_dir}")
            return True
        elif success_count > 0:
            self.logger.warning(f"⚠️  Partial success: {success_count}/{total_count} engines exported")
            self.logger.info(f"Output directory: {output_dir}")
            return False
        else:
            self.logger.error("❌ All TensorRT exports failed")
            return False
