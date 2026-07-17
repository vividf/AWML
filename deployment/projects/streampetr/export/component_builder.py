"""StreamPETR component builder: the three chained export components.

The component split and tensor names are a frozen contract (see the deploy config header);
this builder only assembles the tracing modules and their real sample inputs.
"""

from __future__ import annotations

import logging
from typing import List

import torch

from deployment.export.pipelines.component_builder import ExportableComponent, ModelComponentBuilder
from deployment.projects.streampetr.config.streampetr_deployment_config import StreamPETRDeploymentConfig
from deployment.projects.streampetr.export.onnx_models.encoder_onnx import StreamPETREncoderONNX
from deployment.projects.streampetr.export.onnx_models.position_embedding_onnx import (
    StreamPETRPositionEmbeddingONNX,
)
from deployment.projects.streampetr.export.onnx_models.pts_head_onnx import StreamPETRPtsHeadONNX
from deployment.projects.streampetr.io.sample_types import StreamPETRExportSample

logger = logging.getLogger(__name__)


class StreamPETRComponentBuilder(ModelComponentBuilder):
    """Build the three exportable StreamPETR components from a typed sample."""

    def __init__(self, config: StreamPETRDeploymentConfig) -> None:
        """Initialize StreamPETR component builder.

        Args:
            config: StreamPETR deploy config supplying the component layout.
        """
        self._config = config

    def build_components(
        self,
        model: torch.nn.Module,
        sample: StreamPETRExportSample,
    ) -> List[ExportableComponent]:
        """Build exportable StreamPETR components from a typed sample.

        Args:
            model: StreamPETR (Petr3D) model.
            sample: Typed export sample with chained component inputs.

        Returns:
            The three exportable components, in dependency order.
        """
        logger.info("Extracting StreamPETR components for export...")

        # Denoising is train-only; the deployed graph is traced without it
        # (mirrors `tm.mod.pts_bbox_head.with_dn = False` in the original exporter).
        model.pts_bbox_head.with_dn = False

        components = [
            self._build_encoder(model, sample),
            self._build_position_embedding(model, sample),
            self._build_pts_head(model, sample),
        ]
        logger.info("Extracted 3 components: extract_img_feat, position_embedding, pts_head_memory")
        return components

    def _build_encoder(self, model: torch.nn.Module, sample: StreamPETRExportSample) -> ExportableComponent:
        component_cfg = self._config.components_cfg.get_component("extract_img_feat")
        return ExportableComponent(
            name=component_cfg.name,
            module=StreamPETREncoderONNX(model),
            # extract_img_feat squeezes the input in place during tracing: keep our copy intact.
            sample_input=sample.img.clone(),
        )

    def _build_position_embedding(self, model: torch.nn.Module, sample: StreamPETRExportSample) -> ExportableComponent:
        component_cfg = self._config.components_cfg.get_component("position_embedding")
        return ExportableComponent(
            name=component_cfg.name,
            module=StreamPETRPositionEmbeddingONNX(model),
            sample_input=(
                sample.img_metas_pad,
                sample.img_feats,
                sample.intrinsics,
                sample.img2lidar,
            ),
        )

    def _build_pts_head(self, model: torch.nn.Module, sample: StreamPETRExportSample) -> ExportableComponent:
        component_cfg = self._config.components_cfg.get_component("pts_head_memory")
        return ExportableComponent(
            name=component_cfg.name,
            module=StreamPETRPtsHeadONNX(model),
            sample_input=(
                sample.img_feats,
                sample.pos_embed,
                sample.cone,
                sample.data_timestamp,
                sample.data_ego_pose,
                sample.data_ego_pose_inv,
                sample.memory_embedding,
                sample.memory_reference_point,
                sample.memory_timestamp,
                sample.memory_egopose,
                sample.memory_velo,
            ),
        )
