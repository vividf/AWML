"""
CenterPoint-specific deployment runner.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from mmengine.config import Config

from deployment.configs.base import BaseDeploymentConfig
from deployment.core.contexts import CenterPointExportContext, ExportContext
from deployment.core.device import DeviceSpec
from deployment.core.evaluation.backend_executor import BackendExecutor
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.exporters.common.factory import ExporterFactory
from deployment.exporters.common.model_wrappers import IdentityWrapper
from deployment.exporters.export_pipelines.base import OnnxExportPipeline, TensorRTExportPipeline
from deployment.projects.centerpoint.eval.evaluator import CenterPointEvaluator
from deployment.projects.centerpoint.export.component_builder import CenterPointComponentBuilder
from deployment.projects.centerpoint.export.onnx_export_pipeline import CenterPointONNXExportPipeline
from deployment.projects.centerpoint.export.tensorrt_export_pipeline import CenterPointTensorRTExportPipeline
from deployment.projects.centerpoint.io.model_loader import build_centerpoint_onnx_model
from deployment.projects.centerpoint.io.sample_adapter import CenterPointSampleAdapter
from deployment.runtime.runner import BaseDeploymentRunner


class CenterPointDeploymentRunner(BaseDeploymentRunner):
    """CenterPoint deployment runner.

    Implements project-specific model loading and wiring to export pipelines,
    while reusing the project-agnostic orchestration in `BaseDeploymentRunner`.

    Attributes:
        model_cfg: Training MMEngine config (from checkpoint experiment file); not replaced after load.
        evaluator: CenterPoint evaluator instance.
    """

    def __init__(
        self,
        data_loader: BaseDataLoader,
        evaluator: CenterPointEvaluator,
        executor: BackendExecutor,
        config: BaseDeploymentConfig,
        model_cfg: Config,
        logger: logging.Logger,
        onnx_pipeline: Optional[OnnxExportPipeline] = None,
        tensorrt_pipeline: Optional[TensorRTExportPipeline] = None,
    ) -> None:
        """Initialize CenterPoint deployment runner.

        Args:
            data_loader: Data loader for loading samples.
            evaluator: Evaluator for computing metrics.
            executor: Backend execution primitives shared with the evaluator/verification runner.
            config: Deployment configuration.
            model_cfg: MMEngine model configuration.
            logger: Logger instance.
            onnx_pipeline: Optional custom ONNX export pipeline.
            tensorrt_pipeline: Optional custom TensorRT export pipeline.
        """

        if onnx_pipeline is None:
            sample_adapter = CenterPointSampleAdapter(logger=logger)
            component_builder = CenterPointComponentBuilder(components_cfg=config.components_cfg, logger=logger)
            onnx_pipeline = CenterPointONNXExportPipeline(
                exporter_factory=ExporterFactory,
                sample_adapter=sample_adapter,
                component_builder=component_builder,
                logger=logger,
            )

        if tensorrt_pipeline is None:
            tensorrt_pipeline = CenterPointTensorRTExportPipeline(
                exporter_factory=ExporterFactory,
                components_cfg=config.components_cfg,
                logger=logger,
            )

        super().__init__(
            data_loader=data_loader,
            evaluator=evaluator,
            executor=executor,
            config=config,
            model_cfg=model_cfg,
            logger=logger,
            onnx_wrapper_cls=IdentityWrapper,
            onnx_pipeline=onnx_pipeline,
            tensorrt_pipeline=tensorrt_pipeline,
        )

    def load_pytorch_model(self, checkpoint_path: str, context: ExportContext) -> torch.nn.Module:
        """Load and return the PyTorch model for export.

        Args:
            checkpoint_path: Path to the checkpoint file.
            context: Export context with additional parameters.

        Returns:
            Loaded PyTorch model.
        """
        rot_y_axis_reference = self._extract_rot_y_axis_reference(context)
        self.logger.info("Export option rot_y_axis_reference = %s", rot_y_axis_reference)

        model, _ = build_centerpoint_onnx_model(
            base_model_cfg=self.model_cfg,
            checkpoint_path=checkpoint_path,
            device=DeviceSpec.from_value("cpu"),
            rot_y_axis_reference=rot_y_axis_reference,
        )
        return model

    def _extract_rot_y_axis_reference(self, context: ExportContext) -> bool:
        """Extract rot_y_axis_reference from the export context.

        Args:
            context: Export context; must be a ``CenterPointExportContext``.

        Returns:
            Boolean value for rot_y_axis_reference.
        """
        if not isinstance(context, CenterPointExportContext):
            raise TypeError(f"CenterPoint export requires a CenterPointExportContext, got {type(context).__name__}.")
        return context.rot_y_axis_reference
