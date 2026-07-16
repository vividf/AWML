"""Calibration classifier deployment runner."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from mmengine.config import Config
from typing_extensions import override

from deployment.evaluation.classification_evaluator import ClassificationEvaluator
from deployment.execution.backend_executor import BackendExecutor
from deployment.export.exporters.model_wrappers import IdentityWrapper
from deployment.export.pipelines.onnx_export_pipeline import OnnxExportPipeline
from deployment.export.pipelines.tensorrt_export_pipeline import TensorRTExportPipeline
from deployment.io.base_data_loader import BaseDataLoader
from deployment.primitives.device import DeviceSpec
from deployment.projects.calibration.config.calibration_deployment_config import CalibrationDeploymentConfig
from deployment.projects.calibration.io.model_loader import build_calibration_model
from deployment.runtime.runner import BaseDeploymentRunner

logger = logging.getLogger(__name__)


class CalibrationDeploymentRunner(BaseDeploymentRunner):
    """Calibration classifier deployment runner.

    A thin ``BaseDeploymentRunner`` subclass: it loads the mmpretrain classifier and wires the
    shared whole-model ONNX export with ``IdentityWrapper`` (the graph emits raw logits; softmax is
    applied in postprocess). Single-component export, so it uses the framework default export
    pipeline (``DefaultSampleExtractor`` + ``DefaultComponentBuilder``).
    """

    def __init__(
        self,
        data_loader: BaseDataLoader,
        evaluator: ClassificationEvaluator,
        executor: BackendExecutor,
        config: CalibrationDeploymentConfig,
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
            onnx_wrapper_cls=IdentityWrapper,
            onnx_pipeline=onnx_pipeline,
            tensorrt_pipeline=tensorrt_pipeline,
        )

    @override
    def load_pytorch_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load the mmpretrain classifier on CPU (the executor moves it to the eval/verify device)."""
        return build_calibration_model(
            model_cfg=self.model_cfg,
            checkpoint_path=checkpoint_path,
            device=DeviceSpec.from_value("cpu"),
        )
