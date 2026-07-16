"""Calibration classifier ONNX Runtime inference pipeline."""

from __future__ import annotations

import logging
from typing import Dict, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import torch
from typing_extensions import override

from deployment.config.enums import Backend
from deployment.config.schema import ComponentsConfig
from deployment.primitives.artifacts import resolve_artifact_path
from deployment.primitives.device import DeviceSpec
from deployment.projects.calibration.inference.base_inference_pipeline import ClassificationInferencePipeline

logger = logging.getLogger(__name__)


class CalibrationONNXInferencePipeline(ClassificationInferencePipeline):
    """ONNX Runtime backend for the calibration classifier (single ``model`` component)."""

    def __init__(
        self,
        onnx_dir: str,
        device: DeviceSpec,
        class_names: Sequence[str],
        components_cfg: ComponentsConfig,
    ) -> None:
        super().__init__(
            model=None,
            backend_type=Backend.ONNX,
            device=device,
            class_names=class_names,
        )
        onnx_path = resolve_artifact_path(
            base_dir=onnx_dir,
            components_cfg=components_cfg,
            component_name="model",
            file_key="onnx_file",
        )

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        self._session = ort.InferenceSession(
            onnx_path, sess_options=session_options, providers=device.to_ort_provider()
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        logger.info("Loaded calibration ONNX model: %s (providers=%s)", onnx_path, self._session.get_providers())

    @override
    def run_model(self, preprocessed_input: torch.Tensor) -> Tuple[np.ndarray, Dict[str, float]]:
        """Run the ONNX session and return raw class logits ``[1, num_classes]``."""
        input_array = self.to_numpy(preprocessed_input, dtype=np.float32)
        outputs = self._session.run([self._output_name], {self._input_name: input_array})
        return outputs[0], {}
