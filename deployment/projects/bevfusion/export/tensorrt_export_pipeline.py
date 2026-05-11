"""BEVFusion TensorRT export pipeline.

Converts BEVFusion ONNX (single or split sparse+dense) to TensorRT engine(s).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch

from deployment.configs.base import BaseDeploymentConfig
from deployment.configs.schema import ComponentsConfig
from deployment.core.artifacts import Artifact
from deployment.core.device import DeviceSpec
from deployment.exporters.common.factory import ExporterFactory
from deployment.exporters.export_pipelines.base import TensorRTExportPipeline
from deployment.projects.bevfusion.io.component_utils import is_split_bevfusion_components


class BEVFusionTensorRTExportPipeline(TensorRTExportPipeline):
    """TensorRT export for BEVFusion (one engine or sparse+dense pair)."""

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

        if is_split_bevfusion_components(self._components_cfg):
            return self._export_split_engines(onnx_dir, output_dir_path, config, device)

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

    def _export_split_engines(
        self,
        onnx_dir: Path,
        output_dir_path: Path,
        config: BaseDeploymentConfig,
        device: DeviceSpec,
    ) -> Artifact:
        if not onnx_dir.is_dir():
            raise ValueError(f"Split TensorRT export expects ONNX directory, got: {onnx_dir}")

        onnx_files = sorted(
            (path for path in onnx_dir.iterdir() if path.is_file() and path.suffix.lower() == ".onnx"),
            key=lambda p: p.name,
        )
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX files in {onnx_dir}")

        engine_file_map = self._build_engine_file_map()
        onnx_stem_to_component = self._build_onnx_stem_to_component_map()

        self.logger.info("=" * 80)
        self.logger.info("Converting split BEVFusion ONNX → TensorRT (sparse + dense)")
        self.logger.info("=" * 80)

        for i, onnx_file in enumerate(onnx_files, 1):
            onnx_stem = onnx_file.stem
            if onnx_stem not in engine_file_map:
                raise KeyError(f"ONNX file '{onnx_file.name}' is not declared in deploy config components.*.onnx_file")
            engine_file = engine_file_map[onnx_stem]
            trt_path = output_dir_path / engine_file
            trt_path.parent.mkdir(parents=True, exist_ok=True)

            component_name = onnx_stem_to_component[onnx_stem]
            self.logger.info("[%d/%d] %s → %s", i, len(onnx_files), onnx_file.name, trt_path.name)

            exporter = self.exporter_factory.create_tensorrt_exporter(
                config=config,
                logger=self.logger,
                component_name=component_name,
            )
            exporter.export(
                model=None,
                sample_input=None,
                output_path=str(trt_path),
                onnx_path=str(onnx_file),
            )

        self.logger.info("Split TensorRT engines written to %s", output_dir_path)
        self.logger.info("=" * 80)
        return Artifact(path=str(output_dir_path))

    def _build_engine_file_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for _name, comp in self._components_cfg.items():
            mapping[Path(comp.onnx_file).stem] = comp.engine_file
        return mapping

    def _build_onnx_stem_to_component_map(self) -> Dict[str, str]:
        return {
            Path(component_cfg.onnx_file).stem: component_name
            for component_name, component_cfg in self._components_cfg.items()
        }
