"""BEVFusion TensorRT export pipeline.

Converts a BEVFusion ONNX model to a TensorRT engine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch

from deployment.configs import BaseDeploymentConfig, ComponentsConfig
from deployment.core.artifacts import Artifact
from deployment.core.device import DeviceSpec
from deployment.exporters.common.factory import ExporterFactory
from deployment.exporters.export_pipelines.base import TensorRTExportPipeline


class BEVFusionTensorRTExportPipeline(TensorRTExportPipeline):
    """TensorRT export pipeline for BEVFusion.

    Converts the single BEVFusion ONNX model into a TensorRT engine.
    """

    def __init__(
        self,
        exporter_factory: type[ExporterFactory],
        components_cfg: ComponentsConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.exporter_factory = exporter_factory
        self._components_cfg = components_cfg
        self.logger = logger or logging.getLogger(__name__)

    def export(
        self,
        *,
        onnx_path: str,
        output_dir: str,
        config: BaseDeploymentConfig,
        device: DeviceSpec,
    ) -> Artifact:
        if not device.is_cuda:
            raise ValueError(f"TensorRT export requires CUDA device, got: {device}")

        torch.cuda.set_device(device.index)

        onnx_dir = Path(onnx_path)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        component_cfg = self._components_cfg.get_component("bevfusion_main_body")
        onnx_file = onnx_dir / component_cfg.onnx_file
        engine_file = output_dir_path / component_cfg.engine_file

        if not onnx_file.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_file}")

        self.logger.info("=" * 80)
        self.logger.info("Converting BEVFusion ONNX to TensorRT")
        self.logger.info("=" * 80)
        self.logger.info(f"ONNX: {onnx_file}")
        self.logger.info(f"Engine: {engine_file}")

        exporter = self.exporter_factory.create_tensorrt_exporter(
            config=config,
            logger=self.logger,
            component_name="bevfusion_main_body",
        )

        artifact = exporter.export(
            model=None,
            sample_input=None,
            output_path=str(engine_file),
            onnx_path=str(onnx_file),
        )

        self.logger.info(f"TensorRT engine saved: {artifact.path}")
        self.logger.info("=" * 80)

        return Artifact(path=str(output_dir_path))
