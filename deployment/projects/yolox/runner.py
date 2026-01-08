"""
YOLOX-specific deployment runner.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from mmdet.apis import init_detector
from mmengine.config import Config

from deployment.core import BaseDeploymentConfig
from deployment.core.contexts import ExportContext, YOLOXExportContext
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.exporters.export_pipelines.base import OnnxExportPipeline, TensorRTExportPipeline
from deployment.projects.yolox.evaluator import YOLOXEvaluator
from deployment.projects.yolox.model_wrappers import YOLOXOptElanONNXWrapper
from deployment.runtime.runner import BaseDeploymentRunner


class YOLOXDeploymentRunner(BaseDeploymentRunner):
    """YOLOX deployment runner.

    Implements project-specific model loading and wiring to export pipelines,
    while reusing the project-agnostic orchestration in `BaseDeploymentRunner`.

    Handles YOLOX-specific requirements:
    - ReLU6 to ReLU replacement for ONNX compatibility

    Attributes:
        model_cfg: MMEngine model configuration.
        evaluator: YOLOX evaluator instance.
    """

    def __init__(
        self,
        data_loader: BaseDataLoader,
        evaluator: YOLOXEvaluator,
        config: BaseDeploymentConfig,
        model_cfg: Config,
        logger: logging.Logger,
        onnx_pipeline: Optional[OnnxExportPipeline] = None,
        tensorrt_pipeline: Optional[TensorRTExportPipeline] = None,
    ) -> None:
        """Initialize YOLOX deployment runner.

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
            onnx_wrapper_cls=YOLOXOptElanONNXWrapper,
            onnx_pipeline=onnx_pipeline,
            tensorrt_pipeline=tensorrt_pipeline,
        )

    def load_pytorch_model(self, checkpoint_path: str, context: ExportContext) -> torch.nn.Module:
        """Load PyTorch model for export.

        Performs YOLOX-specific preprocessing:
        - Replaces ReLU6 with ReLU for better ONNX compatibility

        Args:
            checkpoint_path: Path to the checkpoint file.
            context: Export context. Use YOLOXExportContext for type-safe access
                     to model_cfg_path. Falls back to context.extra for compatibility.

        Returns:
            Loaded PyTorch model.
        """
        # Extract model_cfg_path from typed context or extra dict
        model_cfg_path: Optional[str] = None
        if isinstance(context, YOLOXExportContext):
            model_cfg_path = context.model_cfg
        else:
            raise ValueError("context must be a YOLOXExportContext")

        if model_cfg_path is None:
            # Try to get from model_cfg if it's a file path
            if hasattr(self.model_cfg, "filename"):
                model_cfg_path = self.model_cfg.filename
            else:
                raise ValueError(
                    "model_cfg is required for YOLOX model loading. "
                    "Use YOLOXExportContext(model_cfg='path/to/config.py') "
                    "or ensure model_cfg has a 'filename' attribute."
                )

        model = init_detector(model_cfg_path, checkpoint_path, device="cpu")
        model.eval()

        # Replace ReLU6 with ReLU for better ONNX compatibility
        def replace_relu6_with_relu(module):
            for name, child in module.named_children():
                if isinstance(child, torch.nn.ReLU6):
                    setattr(module, name, torch.nn.ReLU(inplace=child.inplace))
                else:
                    replace_relu6_with_relu(child)

        replace_relu6_with_relu(model)

        # Inject model to evaluator via setter (single-direction injection)
        if hasattr(self.evaluator, "set_pytorch_model"):
            self.evaluator.set_pytorch_model(model)
            self.logger.info("Updated evaluator with pre-built PyTorch model via set_pytorch_model()")

        return model
