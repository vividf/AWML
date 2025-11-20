"""
CenterPoint ONNX export workflow using composition.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from typing import Any, Callable, Optional, Tuple, Union

import torch

from deployment.core import Artifact, BaseDataLoader, BaseDeploymentConfig
from deployment.exporters.base.configs import ONNXExportConfig
from deployment.exporters.base.onnx_exporter import ONNXExporter
from deployment.exporters.workflows.base import OnnxExportWorkflow


class CenterPointONNXExportWorkflow(OnnxExportWorkflow):
    """
    CenterPoint ONNX export workflow.

    Orchestrates multi-file ONNX export using a generic ONNXExporter.
    - voxel encoder → pts_voxel_encoder.onnx
    - backbone+neck+head → pts_backbone_neck_head.onnx
    """

    def __init__(
        self,
        exporter: Union[ONNXExporter, Callable[[], ONNXExporter]],
        logger: Optional[logging.Logger] = None,
    ):
        self._exporter_provider = exporter
        self._exporter_cache: Optional[ONNXExporter] = None
        self.logger = logger or logging.getLogger(__name__)

    def export(
        self,
        *,
        model: torch.nn.Module,
        data_loader: BaseDataLoader,
        output_dir: str,
        config: BaseDeploymentConfig,
        sample_idx: int = 0,
        **_: Any,
    ) -> Artifact:
        output_dir = self._resolve_output_dir(output_dir)
        input_features, voxel_dict = self._extract_features(model, data_loader, sample_idx, output_dir)
        voxel_encoder_path = self._export_voxel_encoder(model, input_features, output_dir)
        backbone_path = self._export_backbone_neck_head(model, input_features, voxel_dict, output_dir)
        self._log_summary(voxel_encoder_path, backbone_path)
        return Artifact(path=output_dir, multi_file=True)

    def _get_exporter(self) -> ONNXExporter:
        if self._exporter_cache is None:
            exporter = self._exporter_provider() if callable(self._exporter_provider) else self._exporter_provider
            if exporter is None:
                raise RuntimeError("CenterPoint ONNX workflow requires an ONNXExporter instance")
            self._exporter_cache = exporter
        return self._exporter_cache

    def _get_base_config(self) -> ONNXExportConfig:
        return self._get_exporter().config

    def _resolve_output_dir(self, output_dir: str) -> str:
        if not output_dir:
            raise ValueError("output_dir must be provided for CenterPoint ONNX export")

        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _extract_features(
        self, model: torch.nn.Module, data_loader: Any, sample_idx: int, output_dir: str
    ) -> Tuple[Any, dict]:
        self.logger.info("=" * 80)
        self.logger.info("Exporting CenterPoint to ONNX (Multi-file)")
        self.logger.info("=" * 80)
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Using real data (sample_idx={sample_idx})")

        try:
            self.logger.info("Extracting features from real data...")
            input_features, voxel_dict = model._extract_features(data_loader, sample_idx)
            return input_features, voxel_dict
        except Exception as exc:
            self.logger.error(f"Failed to extract features: {exc}")
            raise RuntimeError("Feature extraction failed") from exc

    def _export_voxel_encoder(self, model: torch.nn.Module, input_features: Any, output_dir: str) -> str:
        self.logger.info("\n[1/2] Exporting voxel encoder...")
        voxel_encoder_path = os.path.join(output_dir, "pts_voxel_encoder.onnx")

        voxel_encoder_cfg = self._build_voxel_encoder_config(self._get_base_config())

        try:
            self._get_exporter().export(
                model=model.pts_voxel_encoder,
                sample_input=input_features,
                output_path=voxel_encoder_path,
                config_override=voxel_encoder_cfg,
            )
            return voxel_encoder_path
        except Exception as exc:
            self.logger.error("Failed to export voxel encoder")
            raise RuntimeError("Voxel encoder export failed") from exc

    def _build_voxel_encoder_config(self, base_config: ONNXExportConfig) -> ONNXExportConfig:
        return replace(
            base_config,
            input_names=("input_features",),
            output_names=("pillar_features",),
            dynamic_axes={
                "input_features": {0: "num_voxels", 1: "num_max_points"},
                "pillar_features": {0: "num_voxels"},
            },
        )

    def _export_backbone_neck_head(
        self, model: torch.nn.Module, input_features: Any, voxel_dict: dict, output_dir: str
    ) -> str:
        self.logger.info("\n[2/2] Exporting backbone + neck + head...")

        with torch.no_grad():
            voxel_features = model.pts_voxel_encoder(input_features).squeeze(1)
            coors = voxel_dict["coors"]
            batch_size = coors[-1, 0] + 1
            spatial_features = model.pts_middle_encoder(voxel_features, coors, batch_size)

        from projects.CenterPoint.models.detectors.centerpoint_onnx import CenterPointHeadONNX

        backbone_neck_head = CenterPointHeadONNX(model.pts_backbone, model.pts_neck, model.pts_bbox_head)

        output_names = self._get_output_names(model)

        backbone_path = os.path.join(output_dir, "pts_backbone_neck_head.onnx")
        backbone_cfg = self._build_backbone_config(self._get_base_config(), output_names)

        try:
            self._get_exporter().export(
                model=backbone_neck_head,
                sample_input=spatial_features,
                output_path=backbone_path,
                config_override=backbone_cfg,
            )
            return backbone_path
        except Exception as exc:
            self.logger.error("Failed to export backbone+neck+head")
            raise RuntimeError("Backbone+neck+head export failed") from exc

    def _get_output_names(self, model: torch.nn.Module) -> Tuple[str, ...]:
        output_names = model.pts_bbox_head.output_names if hasattr(model.pts_bbox_head, "output_names") else None
        if not output_names:
            output_names = ["reg", "height", "dim", "rot", "vel", "heatmap"]
        return tuple(output_names) if isinstance(output_names, (list, tuple)) else (output_names,)

    def _build_backbone_config(
        self,
        base_config: ONNXExportConfig,
        output_names: Tuple[str, ...],
    ) -> ONNXExportConfig:
        return replace(
            base_config,
            input_names=("spatial_features",),
            output_names=output_names,
            dynamic_axes={
                "spatial_features": {0: "batch_size", 2: "height", 3: "width"},
            },
        )

    def _log_summary(self, voxel_encoder_path: str, backbone_path: str) -> None:
        self.logger.info("\n" + "=" * 80)
        self.logger.info("✅ CenterPoint ONNX export successful")
        self.logger.info("=" * 80)
        self.logger.info(f"Voxel Encoder: {voxel_encoder_path}")
        self.logger.info(f"Backbone+Neck+Head: {backbone_path}")
