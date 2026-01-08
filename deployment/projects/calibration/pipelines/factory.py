"""
Calibration Pipeline Factory.

Registers Calibration pipelines into the global pipeline_registry so evaluators can create pipelines
via `deployment.pipelines.factory.PipelineFactory`.
"""

import logging
from typing import Any, Mapping, Optional

from deployment.core.backend import Backend
from deployment.core.evaluation.evaluator_types import ModelSpec
from deployment.pipelines.base_factory import BasePipelineFactory
from deployment.pipelines.base_pipeline import BaseDeploymentPipeline
from deployment.pipelines.registry import pipeline_registry
from deployment.projects.calibration.pipelines.onnx import CalibrationONNXPipeline
from deployment.projects.calibration.pipelines.pytorch import CalibrationPyTorchPipeline
from deployment.projects.calibration.pipelines.tensorrt import CalibrationTensorRTPipeline

logger = logging.getLogger(__name__)


@pipeline_registry.register
class CalibrationPipelineFactory(BasePipelineFactory):
    """Pipeline factory for Calibration across supported backends.

    Supports passing `components_cfg` to configure component file paths.
    """

    @classmethod
    def get_project_name(cls) -> str:
        return "calibration"

    @classmethod
    def create_pipeline(
        cls,
        model_spec: ModelSpec,
        pytorch_model: Any,
        device: Optional[str] = None,
        components_cfg: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> BaseDeploymentPipeline:
        """Create a Calibration pipeline for the specified backend.

        Args:
            model_spec: Model specification (backend/device/path)
            pytorch_model: PyTorch model instance for preprocessing
            device: Override device (uses model_spec.device if None)
            components_cfg: Unified component configuration dict from deploy_config.
                           Used to resolve artifact file paths.
            **kwargs: Additional arguments (unused)

        Returns:
            Pipeline instance for the specified backend
        """
        device = device or model_spec.device
        backend = model_spec.backend

        cls._validate_backend(backend)

        if backend is Backend.PYTORCH:
            logger.info(f"Creating Calibration PyTorch pipeline on {device}")
            return CalibrationPyTorchPipeline(pytorch_model, device=device)

        if backend is Backend.ONNX:
            logger.info(f"Creating Calibration ONNX pipeline from {model_spec.path} on {device}")
            return CalibrationONNXPipeline(
                onnx_path=model_spec.path,
                device=device,
                components_cfg=components_cfg,
            )

        if backend is Backend.TENSORRT:
            logger.info(f"Creating Calibration TensorRT pipeline from {model_spec.path} on {device}")
            return CalibrationTensorRTPipeline(
                engine_path=model_spec.path,
                device=device,
                components_cfg=components_cfg,
            )

        raise ValueError(f"Unsupported backend: {backend.value}")
