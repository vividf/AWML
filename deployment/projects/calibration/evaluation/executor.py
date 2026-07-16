"""Calibration classifier backend executor.

Implements pipeline creation and input preparation for the calibration classifier, shared by the
evaluator and the verification runner via ``deployment.execution.backend_executor.BackendExecutor``.
The classifier is image-based with no auxiliary postprocess metadata, so ``prepare_input`` simply
wraps the loader's preprocessed 5-channel tensor.
"""

from __future__ import annotations

import logging
from typing import Any, List, Mapping, Optional, Sequence

from typing_extensions import override

from deployment.config.enums import Backend
from deployment.config.schema import ComponentsConfig
from deployment.execution.backend_executor import BackendExecutor
from deployment.inference.base_inference_pipeline import BaseInferencePipeline
from deployment.io.base_data_loader import BaseDataLoader
from deployment.primitives.device import DeviceSpec
from deployment.primitives.evaluator_types import InferenceInput, ModelSpec
from deployment.projects.calibration.inference.onnx_inference_pipeline import CalibrationONNXInferencePipeline
from deployment.projects.calibration.inference.pytorch_inference_pipeline import CalibrationPyTorchInferencePipeline
from deployment.projects.calibration.inference.tensorrt_inference_pipeline import CalibrationTensorRTInferencePipeline

logger = logging.getLogger(__name__)


class CalibrationExecutor(BackendExecutor):
    """Backend execution primitives for the calibration classifier (pipeline creation, input prep).

    Args:
        components_cfg: Unified components configuration (single ``model`` component), used to
            resolve ONNX/engine artifact filenames.
        class_names: Class names in label-index order, forwarded to the pipelines for labelling.
    """

    def __init__(self, components_cfg: ComponentsConfig, class_names: Sequence[str]) -> None:
        super().__init__()
        self._components_cfg = components_cfg
        self._class_names = list(class_names)

    @override
    def get_output_names(self) -> Optional[List[str]]:
        """Return the ``model`` component's output names for verification logging."""
        return [out.name for out in self._components_cfg.get_component("model").io.outputs]

    @override
    def create_pipeline(self, model_spec: ModelSpec, device: DeviceSpec) -> BaseInferencePipeline:
        """Create a calibration inference pipeline for the given backend and device."""
        backend = model_spec.backend
        self._validate_backend(backend)

        if backend is Backend.PYTORCH:
            logger.info("Creating calibration PyTorch pipeline on %s", device)
            return CalibrationPyTorchInferencePipeline(
                self.pytorch_model, device=device, class_names=self._class_names
            )

        if backend is Backend.ONNX:
            logger.info("Creating calibration ONNX pipeline from %s on %s", model_spec.artifact.path, device)
            return CalibrationONNXInferencePipeline(
                onnx_dir=model_spec.artifact.path,
                device=device,
                class_names=self._class_names,
                components_cfg=self._components_cfg,
            )

        if backend is Backend.TENSORRT:
            logger.info("Creating calibration TensorRT pipeline from %s on %s", model_spec.artifact.path, device)
            return CalibrationTensorRTInferencePipeline(
                tensorrt_dir=model_spec.artifact.path,
                device=device,
                class_names=self._class_names,
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
        """Build an ``InferenceInput`` from the loader's preprocessed 5-channel image tensor."""
        tensor = data_loader.preprocess(sample).to(device.to_torch_device())
        return InferenceInput(data=tensor, metadata={})
