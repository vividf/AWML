"""
CenterPoint TensorRT export pipeline using composition.

Uses typed ComponentsCfg and TensorRTConfig; no dict/key/type checks in pipeline code.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Optional

import torch

from deployment.core import Artifact, BaseDeploymentConfig
from deployment.core.config.base_config import ComponentsCfg, TensorRTConfig
from deployment.exporters.common.configs import TensorRTExportConfig, TensorRTModelInputConfig
from deployment.exporters.common.factory import ExporterFactory
from deployment.exporters.export_pipelines.base import TensorRTExportPipeline


class CenterPointTensorRTExportPipeline(TensorRTExportPipeline):
    """TensorRT export pipeline for CenterPoint.

    Consumes a directory of ONNX files (multi-file export) and builds a TensorRT
    engine per component into `output_dir`. Uses ComponentsCfg and TensorRTConfig
    dataclasses; no config dict access.
    """

    _CUDA_DEVICE_PATTERN = re.compile(r"^cuda:\d+$")

    def __init__(
        self,
        exporter_factory: type[ExporterFactory],
        components_cfg: ComponentsCfg,
        tensorrt_config: TensorRTConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.exporter_factory = exporter_factory
        self._components_cfg = components_cfg
        self._tensorrt_config = tensorrt_config
        self.logger = logger or logging.getLogger(__name__)

    def _validate_cuda_device(self, device: str) -> int:
        if not self._CUDA_DEVICE_PATTERN.match(device):
            raise ValueError(
                f"Invalid CUDA device format: '{device}'. Expected format: 'cuda:N' (e.g., 'cuda:0', 'cuda:1')"
            )
        return int(device.split(":")[1])

    def export(
        self,
        *,
        onnx_path: str,
        output_dir: str,
        config: BaseDeploymentConfig,
        device: str,
    ) -> Artifact:
        if device is None:
            raise ValueError("CUDA device must be provided for TensorRT export")
        if onnx_path is None:
            raise ValueError("onnx_path must be provided for CenterPoint TensorRT export")

        onnx_dir_path = Path(onnx_path)
        if not onnx_dir_path.is_dir():
            raise ValueError(f"onnx_path must be a directory for multi-file export, got: {onnx_path}")

        device_id = self._validate_cuda_device(device)
        torch.cuda.set_device(device_id)
        self.logger.info(f"Using CUDA device: {device}")

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        onnx_files = self._discover_onnx_files(onnx_dir_path)
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX files found in {onnx_dir_path}")

        engine_file_map = self._build_engine_file_map()
        onnx_stem_to_component = self._build_onnx_stem_to_component_map()

        num_files = len(onnx_files)
        for i, onnx_file in enumerate(onnx_files, 1):
            onnx_stem = onnx_file.stem
            if onnx_stem not in engine_file_map:
                raise KeyError(f"ONNX file '{onnx_file.name}' is not declared in deploy config components.*.onnx_file")
            engine_file = engine_file_map[onnx_stem]
            trt_path = output_dir_path / engine_file
            trt_path.parent.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"\n[{i}/{num_files}] Converting {onnx_file.name} → {trt_path.name}...")

            component_name = onnx_stem_to_component[onnx_stem]
            exporter = self._build_tensorrt_exporter_for_component(config, component_name)

            artifact = exporter.export(
                model=None,
                sample_input=None,
                output_path=str(trt_path),
                onnx_path=str(onnx_file),
            )
            self.logger.info(f"TensorRT engine saved: {artifact.path}")

        self.logger.info(f"\nAll TensorRT engines exported successfully to {output_dir_path}")
        return Artifact(path=str(output_dir_path))

    def _discover_onnx_files(self, onnx_dir: Path) -> List[Path]:
        return sorted(
            (path for path in onnx_dir.iterdir() if path.is_file() and path.suffix.lower() == ".onnx"),
            key=lambda p: p.name,
        )

    def _build_engine_file_map(self) -> Dict[str, str]:
        """Build mapping from ONNX stem -> engine_file from ComponentsCfg."""
        mapping: Dict[str, str] = {}
        for name, comp in self._components_cfg.items():
            mapping[Path(comp.onnx_file).stem] = comp.engine_file
        return mapping

    def _build_onnx_stem_to_component_map(self) -> Dict[str, str]:
        """Build mapping from ONNX stem -> component name."""
        return {Path(comp.onnx_file).stem: name for name, comp in self._components_cfg.items()}

    def _build_tensorrt_exporter_for_component(
        self,
        config: BaseDeploymentConfig,
        component_name: str,
    ):
        """Build TensorRT exporter with component-specific profile from typed config."""
        comp = self._components_cfg.get_component(component_name)
        input_shapes = dict(comp.tensorrt_profile)

        model_inputs = (TensorRTModelInputConfig(input_shapes=MappingProxyType(input_shapes)),)
        self.logger.info(f"Using TensorRT profile for component '{component_name}': {list(input_shapes.keys())}")

        trt_export_config = TensorRTExportConfig(
            precision_policy=self._tensorrt_config.precision_policy,
            max_workspace_size=self._tensorrt_config.max_workspace_size,
            model_inputs=model_inputs,
        )

        return self.exporter_factory.create_tensorrt_exporter(
            config=config,
            logger=self.logger,
            config_override=trt_export_config,
        )
