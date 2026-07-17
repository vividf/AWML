"""
StreamPETR-specific deployment runner.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from mmengine.config import Config
from typing_extensions import override

from deployment.evaluation.detection_3d_evaluator import Detection3DEvaluator
from deployment.execution.backend_executor import BackendExecutor
from deployment.export.pipelines.onnx_export_pipeline import OnnxExportPipeline
from deployment.export.pipelines.tensorrt_export_pipeline import TensorRTExportPipeline
from deployment.io.base_data_loader import BaseDataLoader
from deployment.primitives.device import DeviceSpec
from deployment.projects.streampetr.config.streampetr_deployment_config import StreamPETRDeploymentConfig
from deployment.projects.streampetr.export.component_builder import StreamPETRComponentBuilder
from deployment.projects.streampetr.export.sample_extractor import StreamPETRSampleExtractor
from deployment.projects.streampetr.io.model_loader import build_streampetr_model
from deployment.runtime.runner import BaseDeploymentRunner

logger = logging.getLogger(__name__)


class StreamPETRDeploymentRunner(BaseDeploymentRunner):
    """StreamPETR deployment runner.

    Implements project-specific model loading and wiring to export pipelines, while reusing
    the project-agnostic orchestration in `BaseDeploymentRunner`. StreamPETR drives the
    shared `OnnxExportPipeline` with its own sample extractor + component builder because the
    model is exported as three chained components (a frozen runtime contract).
    """

    def __init__(
        self,
        data_loader: BaseDataLoader,
        evaluator: Detection3DEvaluator,
        executor: BackendExecutor,
        config: StreamPETRDeploymentConfig,
        model_cfg: Config,
        onnx_pipeline: Optional[OnnxExportPipeline] = None,
        tensorrt_pipeline: Optional[TensorRTExportPipeline] = None,
    ) -> None:
        """Initialize StreamPETR deployment runner.

        Args:
            data_loader: Data loader for loading samples (clip-ordered).
            evaluator: Evaluator for computing metrics.
            executor: Backend execution primitives shared with the evaluator/verification runner.
            config: Deployment configuration.
            model_cfg: MMEngine model configuration.
            onnx_pipeline: Optional custom ONNX export pipeline.
            tensorrt_pipeline: Optional custom TensorRT export pipeline.
        """
        if onnx_pipeline is None:
            onnx_pipeline = OnnxExportPipeline(
                sample_extractor=StreamPETRSampleExtractor(),
                component_builder=StreamPETRComponentBuilder(config=config),
            )
        super().__init__(
            data_loader=data_loader,
            evaluator=evaluator,
            executor=executor,
            config=config,
            model_cfg=model_cfg,
            onnx_pipeline=onnx_pipeline,
            tensorrt_pipeline=tensorrt_pipeline,
        )

    @override
    def load_pytorch_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load and return the PyTorch model for export.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Returns:
            Loaded PyTorch model (flash attention already swapped for the exportable variant).
        """
        return build_streampetr_model(
            model_cfg=self.model_cfg,
            checkpoint_path=checkpoint_path,
            device=DeviceSpec.from_value("cpu"),
        )
