"""
CenterPoint TensorRT export workflow using composition.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional, Union

import torch

from deployment.core import Artifact, BaseDataLoader, BaseDeploymentConfig
from deployment.exporters.base.tensorrt_exporter import TensorRTExporter
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
        exporter: Union[TensorRTExporter, Callable[[], TensorRTExporter]],
        logger: Optional[logging.Logger] = None,
    ):
        self._exporter_provider = exporter
        self._exporter_cache: Optional[TensorRTExporter] = None
        self.logger = logger or logging.getLogger(__name__)

    def export(
        self,
        *,
        onnx_path: str,
        output_dir: str,
        config: BaseDeploymentConfig,
        device: str,
        data_loader: BaseDataLoader,
        **_: object,
    ) -> Artifact:
        onnx_dir = onnx_path
        if device is None:
            raise ValueError("CUDA device must be provided for TensorRT export")

        device_id = int(device.split(":", 1)[1])
        torch.cuda.set_device(device_id)
        self.logger.info(f"Using CUDA device: {device}")

        if onnx_dir is None:
            raise ValueError("onnx_dir must be provided for CenterPoint TensorRT export")

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

            artifact = self._get_exporter().export(
                model=None,
                sample_input=None,
                output_path=trt_path,
                onnx_path=onnx_file_path,
            )
            self.logger.info(f"TensorRT engine saved: {artifact.path}")

        self.logger.info(f"All TensorRT engines exported successfully to {output_dir}")
        return Artifact(path=output_dir, multi_file=True)

    def _get_exporter(self) -> TensorRTExporter:
        if self._exporter_cache is None:
            exporter = self._exporter_provider() if callable(self._exporter_provider) else self._exporter_provider
            if exporter is None:
                raise RuntimeError("CenterPoint TensorRT workflow requires a TensorRTExporter instance")
            self._exporter_cache = exporter
        return self._exporter_cache
