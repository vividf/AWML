"""YOLOX-specific deployment runner."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from mmengine.config import Config
from typing_extensions import override

from deployment.evaluation.detection_2d_evaluator import Detection2DEvaluator
from deployment.execution.backend_executor import BackendExecutor
from deployment.export.pipelines.onnx_export_pipeline import OnnxExportPipeline
from deployment.export.pipelines.tensorrt_export_pipeline import TensorRTExportPipeline
from deployment.io.base_data_loader import BaseDataLoader
from deployment.primitives.device import DeviceSpec
from deployment.projects.yolox.config.yolox_deployment_config import YOLOXDeploymentConfig
from deployment.projects.yolox.export.model_wrappers import YOLOXONNXWrapper
from deployment.projects.yolox.io.model_loader import build_yolox_model
from deployment.runtime.runner import BaseDeploymentRunner

logger = logging.getLogger(__name__)


class YOLOXDeploymentRunner(BaseDeploymentRunner):
    """YOLOX deployment runner.

    A thin ``BaseDeploymentRunner`` subclass: it loads the mmdet YOLOX model and wires the shared
    whole-model ONNX export path with :class:`YOLOXONNXWrapper` (the Tier4 output layout). YOLOX
    exports as a single component, so it uses the framework's ``DefaultSampleExtractor`` +
    ``DefaultComponentBuilder`` (selected by passing ``onnx_wrapper_cls``) rather than a custom
    export pipeline.
    """

    def __init__(
        self,
        data_loader: BaseDataLoader,
        evaluator: Detection2DEvaluator,
        executor: BackendExecutor,
        config: YOLOXDeploymentConfig,
        model_cfg: Config,
        onnx_pipeline: Optional[OnnxExportPipeline] = None,
        tensorrt_pipeline: Optional[TensorRTExportPipeline] = None,
    ) -> None:
        super().__init__(
            data_loader=data_loader,
            evaluator=evaluator,
            executor=executor,
            config=config,
            model_cfg=model_cfg,
            onnx_wrapper_cls=YOLOXONNXWrapper,
            onnx_pipeline=onnx_pipeline,
            tensorrt_pipeline=tensorrt_pipeline,
        )

    @override
    def load_pytorch_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load the mmdet YOLOX model on CPU (the executor moves it to the eval/verify device)."""
        return build_yolox_model(
            model_cfg=self.model_cfg,
            checkpoint_path=checkpoint_path,
            device=DeviceSpec.from_value("cpu"),
        )
