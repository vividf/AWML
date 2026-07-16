"""YOLOX backend executor.

Implements the task-specific backend execution primitives (pipeline creation and input
preparation) for YOLOX, shared by the evaluator and the verification runner via
``deployment.execution.backend_executor.BackendExecutor``. YOLOX is image-based, so it subclasses
``BackendExecutor`` directly (there is no shared point-cloud input to reuse) and builds the
``InferenceInput`` from the loader's preprocessed image tensor plus the decode metadata.
"""

from __future__ import annotations

import logging
from typing import Any, List, Mapping, Optional

from typing_extensions import override

from deployment.config.enums import Backend
from deployment.config.schema import ComponentsConfig
from deployment.execution.backend_executor import BackendExecutor
from deployment.inference.base_inference_pipeline import BaseInferencePipeline
from deployment.io.base_data_loader import BaseDataLoader
from deployment.primitives.device import DeviceSpec
from deployment.primitives.evaluator_types import InferenceInput, ModelSpec
from deployment.projects.yolox.inference.base_inference_pipeline import YOLOXDecodeParams
from deployment.projects.yolox.inference.onnx_inference_pipeline import YOLOXONNXInferencePipeline
from deployment.projects.yolox.inference.pytorch_inference_pipeline import YOLOXPyTorchInferencePipeline
from deployment.projects.yolox.inference.tensorrt_inference_pipeline import YOLOXTensorRTInferencePipeline

logger = logging.getLogger(__name__)


class YOLOXExecutor(BackendExecutor):
    """Backend execution primitives for YOLOX (pipeline creation, input prep).

    Args:
        components_cfg: Unified components configuration (single ``model`` component), used to
            resolve ONNX/engine artifact filenames.
        decode_params: Postprocess parameters (classes/strides/thresholds) forwarded to pipelines.
    """

    def __init__(self, components_cfg: ComponentsConfig, decode_params: YOLOXDecodeParams) -> None:
        super().__init__()
        self._components_cfg = components_cfg
        self._decode_params = decode_params

    @override
    def get_output_names(self) -> Optional[List[str]]:
        """Return the ``model`` component's output names for verification logging."""
        return [out.name for out in self._components_cfg.get_component("model").io.outputs]

    @override
    def create_pipeline(self, model_spec: ModelSpec, device: DeviceSpec) -> BaseInferencePipeline:
        """Create a YOLOX inference pipeline for the given backend and device."""
        backend = model_spec.backend
        self._validate_backend(backend)

        if backend is Backend.PYTORCH:
            logger.info("Creating YOLOX PyTorch pipeline on %s", device)
            return YOLOXPyTorchInferencePipeline(self.pytorch_model, device=device, decode_params=self._decode_params)

        if backend is Backend.ONNX:
            logger.info("Creating YOLOX ONNX pipeline from %s on %s", model_spec.artifact.path, device)
            return YOLOXONNXInferencePipeline(
                onnx_dir=model_spec.artifact.path,
                device=device,
                decode_params=self._decode_params,
                components_cfg=self._components_cfg,
            )

        if backend is Backend.TENSORRT:
            logger.info("Creating YOLOX TensorRT pipeline from %s on %s", model_spec.artifact.path, device)
            return YOLOXTensorRTInferencePipeline(
                tensorrt_dir=model_spec.artifact.path,
                device=device,
                decode_params=self._decode_params,
                components_cfg=self._components_cfg,
            )

        raise ValueError(f"Unsupported backend: {backend.value}")

    @override
    def prepare_input(
        self,
        sample: Mapping[str, Any],
        data_loader: BaseDataLoader,
        device: DeviceSpec,
    ) -> InferenceInput:
        """Build an ``InferenceInput`` from the preprocessed image tensor + decode metadata.

        The metadata (``scale_factor`` / ``input_shape`` / ``original_shape``) is produced per-sample
        by the data loader and carried through to postprocess for box decode + rescale.
        """
        tensor = data_loader.preprocess(sample).to(device.to_torch_device())
        return InferenceInput(data=tensor, metadata=dict(sample.get("metadata", {})))
