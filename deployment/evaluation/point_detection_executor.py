"""Shared backend-execution primitives for point-cloud 3D detectors.

``PointDetectionExecutor`` factors out the pipeline-construction and ``(points, metainfo)``
input-prep that the CenterPoint and BEVFusion executors previously duplicated (~85% overlap).
Subclasses declare the three backend pipeline classes and, optionally, override
``get_output_names`` / ``_tensorrt_pipeline_kwargs``; everything else is shared.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Type

from typing_extensions import override

from deployment.config.enums import Backend
from deployment.config.schema import ComponentsConfig
from deployment.evaluation.backend_executor import BackendExecutor
from deployment.evaluation.evaluator_types import InferenceInput, ModelSpec
from deployment.inference.base_inference_pipeline import BaseInferencePipeline
from deployment.io.base_data_loader import BaseDataLoader
from deployment.primitives.device import DeviceSpec

logger = logging.getLogger(__name__)


class PointDetectionExecutor(BackendExecutor):
    """Backend execution primitives shared by point-cloud 3D detectors (CenterPoint, BEVFusion).

    Subclasses set the three pipeline classes and (optionally) override ``get_output_names``
    and ``_tensorrt_pipeline_kwargs``. Pipeline construction and input prep are shared.

    Args:
        components_cfg: Unified components configuration, forwarded to the ONNX/TensorRT
            pipelines so they can resolve split vs merged artifacts.
    """

    #: Human-readable task name, used in log lines and error messages.
    task_name: str = "point detector"
    #: Backend pipeline classes; set as class attributes by subclasses.
    pytorch_pipeline_cls: Optional[Type[BaseInferencePipeline]] = None
    onnx_pipeline_cls: Optional[Type[BaseInferencePipeline]] = None
    tensorrt_pipeline_cls: Optional[Type[BaseInferencePipeline]] = None

    def __init__(self, components_cfg: ComponentsConfig) -> None:
        super().__init__()
        self._components_cfg = components_cfg

    def _tensorrt_pipeline_kwargs(self) -> Mapping[str, Any]:
        """Extra keyword args forwarded to the TensorRT pipeline (default: none).

        BEVFusion overrides this to pass its custom spconv INT8 ``plugin_libraries``.
        """
        return {}

    @override
    def create_pipeline(self, model_spec: ModelSpec, device: DeviceSpec) -> BaseInferencePipeline:
        """Create a backend inference pipeline for ``model_spec.backend`` on ``device``."""
        backend = model_spec.backend
        self._validate_backend(backend)

        if backend is Backend.PYTORCH:
            logger.info("Creating %s PyTorch pipeline on %s", self.task_name, device)
            return self.pytorch_pipeline_cls(self.pytorch_model, device=device)

        if backend is Backend.ONNX:
            logger.info("Creating %s ONNX pipeline from %s on %s", self.task_name, model_spec.artifact.path, device)
            return self.onnx_pipeline_cls(
                self.pytorch_model,
                onnx_dir=model_spec.artifact.path,
                device=device,
                components_cfg=self._components_cfg,
            )

        if backend is Backend.TENSORRT:
            logger.info(
                "Creating %s TensorRT pipeline from %s on %s", self.task_name, model_spec.artifact.path, device
            )
            return self.tensorrt_pipeline_cls(
                self.pytorch_model,
                tensorrt_dir=model_spec.artifact.path,
                device=device,
                components_cfg=self._components_cfg,
                **self._tensorrt_pipeline_kwargs(),
            )

        raise ValueError(f"Unsupported backend: {backend.value}")

    @override
    def prepare_input(
        self,
        sample: Mapping[str, Any],
        data_loader: BaseDataLoader,
        device: DeviceSpec,
    ) -> InferenceInput:
        """Build InferenceInput from a sample containing ``points`` and ``metainfo``."""
        if "points" not in sample:
            raise ValueError(f"Expected 'points' in sample. Got keys: {list(sample.keys())}")
        if "metainfo" not in sample:
            raise KeyError(f"Sample must contain 'metainfo' for {self.task_name} postprocess.")
        return InferenceInput(data=sample["points"], metadata=sample["metainfo"])
