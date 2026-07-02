"""BEVFusion TensorRT export pipeline.

Converts BEVFusion ONNX (single or split sparse+dense) to TensorRT engine(s).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from deployment.config.base import BaseDeploymentConfig
from deployment.config.schema import ComponentsConfig
from deployment.export.exporters.tensorrt_exporter import TensorRTExporter
from deployment.primitives.artifacts import Artifact
from deployment.primitives.device import DeviceSpec
from deployment.primitives.tensorrt_plugins import load_tensorrt_plugin_libraries
from deployment.projects.bevfusion.io.component_utils import is_split_bevfusion_components


class BEVFusionTensorRTExportPipeline:
    """TensorRT export for BEVFusion (one engine or sparse+dense pair)."""

    def __init__(
        self,
        components_cfg: ComponentsConfig,
        plugin_libraries: Tuple[str, ...] = (),
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._components_cfg = components_cfg
        self._plugin_libraries = plugin_libraries
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

        # Load any custom plugin .so libraries (e.g. BEVFusion spconv INT8) before building.
        # No-op when plugin_libraries is empty.
        load_tensorrt_plugin_libraries(self.logger, getattr(self, "_plugin_libraries", ()))

        onnx_dir = Path(onnx_path)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        if is_split_bevfusion_components(self._components_cfg):
            return self._export_split_engines(onnx_dir, output_dir_path, config)

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

        artifact = TensorRTExporter(config=config.get_tensorrt_settings("bevfusion_main_body")).export(
            onnx_path=str(onnx_file),
            output_path=str(engine_file),
        )

        self.logger.info(f"TensorRT engine saved: {artifact.path}")
        self.logger.info("=" * 80)

        return Artifact(path=str(output_dir_path))

    def _export_split_engines(
        self,
        onnx_dir: Path,
        output_dir_path: Path,
        config: BaseDeploymentConfig,
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

            TensorRTExporter(config=config.get_tensorrt_settings(component_name)).export(
                onnx_path=str(onnx_file),
                output_path=str(trt_path),
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
