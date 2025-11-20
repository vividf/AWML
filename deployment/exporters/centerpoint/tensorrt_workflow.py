"""
CenterPoint TensorRT export workflow using composition.

This workflow converts the CenterPoint multi-file ONNX export into multiple
TensorRT engines without subclassing the TensorRTExporter directly.
"""

import logging
import os
from typing import Optional

import torch

from deployment.exporters.base.tensorrt_exporter import TensorRTExporter


class CenterPointTensorRTExportWorkflow:
    """
    CenterPoint TensorRT export workflow.

    Converts CenterPoint ONNX files to multiple TensorRT engines:
    - pts_voxel_encoder.onnx → pts_voxel_encoder.engine
    - pts_backbone_neck_head.onnx → pts_backbone_neck_head.engine
    """

    def __init__(self, exporter: TensorRTExporter, logger: Optional[logging.Logger] = None):
        self._exporter = exporter
        self.logger = logger or logging.getLogger(__name__)

    def export(
        self,
        onnx_dir: str,
        output_dir: Optional[str] = None,
        device: str = "cuda:0",
    ) -> None:
        if device is None:
            raise ValueError("CUDA device must be provided for TensorRT export")

        device_id = int(device.split(":", 1)[1])
        torch.cuda.set_device(device_id)
        self.logger.info(f"Using CUDA device: {device}")

        if onnx_dir is None:
            raise ValueError("onnx_dir must be provided for CenterPoint TensorRT export")

        if output_dir is None:
            output_dir = os.path.join(onnx_dir, "tensorrt")
        os.makedirs(output_dir, exist_ok=True)

        onnx_files = [
            ("pts_voxel_encoder.onnx", "pts_voxel_encoder.engine"),
            ("pts_backbone_neck_head.onnx", "pts_backbone_neck_head.engine"),
        ]

        for onnx_file, trt_file in onnx_files:
            onnx_file_path = os.path.join(onnx_dir, onnx_file)
            trt_path = os.path.join(output_dir, trt_file)

            if not os.path.exists(onnx_file_path):
                raise FileNotFoundError(f"ONNX file not found: {onnx_file_path}")

            self.logger.info(f"\nConverting {onnx_file} to TensorRT...")

            artifact = self._exporter.export(
                model=None,
                sample_input=None,
                output_path=trt_path,
                onnx_path=onnx_file_path,
            )
            self.logger.info(f"TensorRT engine saved: {artifact.path}")

        self.logger.info(f"All TensorRT engines exported successfully to {output_dir}")
