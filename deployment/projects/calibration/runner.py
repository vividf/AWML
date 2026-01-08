"""
CalibrationStatusClassification-specific deployment runner.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from mmengine.config import Config
from mmpretrain.apis import get_model

from deployment.core import BaseDeploymentConfig
from deployment.core.contexts import CalibrationExportContext, ExportContext
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.exporters.common.factory import ExporterFactory
from deployment.exporters.common.model_wrappers import IdentityWrapper
from deployment.exporters.export_pipelines.base import OnnxExportPipeline, TensorRTExportPipeline
from deployment.projects.calibration.evaluator import CalibrationEvaluator
from deployment.runtime.runner import BaseDeploymentRunner


class CalibrationDeploymentRunner(BaseDeploymentRunner):
    """CalibrationStatusClassification deployment runner.

    Implements project-specific model loading and wiring to export pipelines,
    while reusing the project-agnostic orchestration in `BaseDeploymentRunner`.

    Attributes:
        model_cfg: MMEngine model configuration.
        evaluator: Calibration evaluator instance.
    """

    def __init__(
        self,
        data_loader: BaseDataLoader,
        evaluator: CalibrationEvaluator,
        config: BaseDeploymentConfig,
        model_cfg: Config,
        logger: logging.Logger,
        onnx_pipeline: Optional[OnnxExportPipeline] = None,
        tensorrt_pipeline: Optional[TensorRTExportPipeline] = None,
    ) -> None:
        """Initialize Calibration deployment runner.

        Args:
            data_loader: Data loader for loading samples.
            evaluator: Evaluator for computing metrics.
            config: Deployment configuration.
            model_cfg: MMEngine model configuration.
            logger: Logger instance.
            onnx_pipeline: Optional custom ONNX export pipeline.
            tensorrt_pipeline: Optional custom TensorRT export pipeline.
        """
        super().__init__(
            data_loader=data_loader,
            evaluator=evaluator,
            config=config,
            model_cfg=model_cfg,
            logger=logger,
            onnx_wrapper_cls=IdentityWrapper,
            onnx_pipeline=onnx_pipeline,
            tensorrt_pipeline=tensorrt_pipeline,
        )

    def load_pytorch_model(self, checkpoint_path: str, context: ExportContext) -> torch.nn.Module:
        """Load PyTorch model for export.

        Args:
            checkpoint_path: Path to the checkpoint file.
            context: Export context with additional parameters.

        Returns:
            Loaded PyTorch model.
        """
        # context is available for future extensions
        _ = context

        torch_device = torch.device("cpu")
        model = get_model(self.model_cfg, checkpoint_path, device=torch_device)
        model.eval()

        # Inject model to evaluator via setter (single-direction injection)
        if hasattr(self.evaluator, "set_pytorch_model"):
            self.evaluator.set_pytorch_model(model)
            self.logger.info("Updated evaluator with pre-built PyTorch model via set_pytorch_model()")

        return model
