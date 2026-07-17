"""
StreamPETR backend executor.

Implements the task-specific backend execution primitives (pipeline creation and input
preparation) for StreamPETR, shared by the evaluator and the verification runner via
`~deployment.execution.backend_executor.BackendExecutor`.

Camera model: subclasses `BackendExecutor` directly (not `PointCloudBackendExecutor`) and
prepares the multi-view camera input dict itself.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from typing_extensions import override

from deployment.config.enums import Backend
from deployment.config.schema import ComponentsConfig
from deployment.execution.backend_executor import BackendExecutor
from deployment.inference.base_inference_pipeline import BaseInferencePipeline
from deployment.io.base_data_loader import BaseDataLoader
from deployment.primitives.device import DeviceSpec
from deployment.primitives.evaluator_types import InferenceInput, ModelSpec
from deployment.projects.streampetr.inference.onnx_inference_pipeline import (
    StreamPETRONNXInferencePipeline,
)
from deployment.projects.streampetr.inference.pytorch_inference_pipeline import (
    StreamPETRPyTorchInferencePipeline,
)
from deployment.projects.streampetr.inference.tensorrt_inference_pipeline import (
    StreamPETRTensorRTInferencePipeline,
)

logger = logging.getLogger(__name__)


class StreamPETRExecutor(BackendExecutor):
    """Backend execution primitives for StreamPETR (pipeline creation, input prep).

    Args:
        components_cfg: Unified components configuration, forwarded to the pipelines.
    """

    def __init__(self, components_cfg: ComponentsConfig) -> None:
        super().__init__()
        self._components_cfg = components_cfg

    @override
    def get_output_names(self) -> Optional[List[str]]:
        """Return the head output names from the components config for verification logging."""
        return [out.name for out in self._components_cfg.get_component("pts_head_memory").io.outputs]

    @override
    def create_pipeline(self, model_spec: ModelSpec, device: DeviceSpec) -> BaseInferencePipeline:
        """Create a StreamPETR inference pipeline for the given backend and device.

        Pipelines are stateful (temporal memory queue lives on the pipeline instance), so
        one pipeline instance must serve a whole clip-ordered run — which is exactly how the
        evaluator and verifier use `create_pipeline`.

        Raises:
            ValueError: If ``model_spec.backend`` is not a supported backend.
        """
        backend = model_spec.backend
        self._validate_backend(backend)

        if backend is Backend.PYTORCH:
            logger.info("Creating StreamPETR PyTorch pipeline on %s", device)
            return StreamPETRPyTorchInferencePipeline(self.pytorch_model, device=device)

        if backend is Backend.ONNX:
            logger.info("Creating StreamPETR ONNX pipeline from %s on %s", model_spec.artifact.path, device)
            return StreamPETRONNXInferencePipeline(
                self.pytorch_model,
                onnx_dir=model_spec.artifact.path,
                device=device,
                components_cfg=self._components_cfg,
            )

        if backend is Backend.TENSORRT:
            logger.info("Creating StreamPETR TensorRT pipeline from %s on %s", model_spec.artifact.path, device)
            return StreamPETRTensorRTInferencePipeline(
                self.pytorch_model,
                tensorrt_dir=model_spec.artifact.path,
                device=device,
                components_cfg=self._components_cfg,
            )

        raise ValueError(f"Unsupported backend: {backend.value}")

    @override
    def prepare_input(self, sample: Any, data_loader: BaseDataLoader, device: DeviceSpec) -> InferenceInput:
        """Prepare the per-frame multi-view camera input for inference.

        Args:
            sample: Sample loaded by :class:`StreamPETRDataLoader`.
            data_loader: The data loader that produced ``sample``.
            device: Target device (placement is handled inside the pipelines).

        Returns:
            InferenceInput with the per-frame tensor dict. ``is_sequence_start`` (computed by
            the loader, index-0 corrected) is injected into the data dict because the
            pipeline's memory-queue reset happens in ``run_model`` — before ``postprocess``,
            the only stage that receives metadata.
        """
        metadata = dict(sample.get("metadata", {}))
        data = dict(data_loader.preprocess(sample))
        data["is_sequence_start"] = metadata.get("is_sequence_start", False)
        return InferenceInput(data=data, metadata=metadata)
