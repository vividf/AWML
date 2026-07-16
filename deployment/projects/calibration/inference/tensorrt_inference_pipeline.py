"""Calibration classifier TensorRT inference pipeline (single-engine, shared TRT runner)."""

from __future__ import annotations

import logging
from typing import Dict, Sequence, Tuple

import numpy as np
import pycuda.autoinit  # noqa: F401 - initializes the CUDA context as a side effect
import tensorrt as trt
import torch
from typing_extensions import override

from deployment.config.enums import Backend
from deployment.config.schema import ComponentsConfig
from deployment.inference.gpu_resource_mixin import GPUResourceMixin, release_tensorrt_resources
from deployment.inference.tensorrt_runner import list_trt_io_names, load_trt_engine, run_trt_engine
from deployment.primitives.artifacts import resolve_artifact_path
from deployment.primitives.device import DeviceSpec
from deployment.projects.calibration.inference.base_inference_pipeline import ClassificationInferencePipeline

logger = logging.getLogger(__name__)


class CalibrationTensorRTInferencePipeline(GPUResourceMixin, ClassificationInferencePipeline):
    """TensorRT backend for the calibration classifier (single ``model`` engine)."""

    def __init__(
        self,
        tensorrt_dir: str,
        device: DeviceSpec,
        class_names: Sequence[str],
        components_cfg: ComponentsConfig,
    ) -> None:
        super().__init__(
            model=None,
            backend_type=Backend.TENSORRT,
            device=device,
            class_names=class_names,
        )
        self._trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self._trt_logger, "")
        runtime = trt.Runtime(self._trt_logger)

        engine_path = resolve_artifact_path(
            base_dir=tensorrt_dir,
            components_cfg=components_cfg,
            component_name="model",
            file_key="engine_file",
        )
        self._engine, self._context = load_trt_engine(runtime, engine_path, component_name="model")
        self._input_names, self._output_names = list_trt_io_names(self._engine)
        logger.info("Loaded calibration TensorRT engine: %s", engine_path)

    @override
    def run_model(self, preprocessed_input: torch.Tensor) -> Tuple[np.ndarray, Dict[str, float]]:
        """Run the engine and return raw class logits ``[1, num_classes]``."""
        input_array = self.to_numpy(preprocessed_input, dtype=np.float32)
        outputs, gpu_ms = run_trt_engine(
            self._engine,
            self._context,
            {self._input_names[0]: input_array},
            self._output_names,
        )
        return outputs[self._output_names[0]], {"model_gpu_ms": gpu_ms}

    @override
    def _release_gpu_resources(self) -> None:
        """Release the TensorRT engine and context."""
        release_tensorrt_resources(
            engines={"model": getattr(self, "_engine", None)},
            contexts={"model": getattr(self, "_context", None)},
        )
