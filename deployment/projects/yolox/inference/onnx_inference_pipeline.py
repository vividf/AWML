"""YOLOX ONNX Runtime inference pipeline."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import onnxruntime as ort
import torch
from typing_extensions import override

from deployment.config.enums import Backend
from deployment.config.schema import ComponentsConfig
from deployment.primitives.artifacts import resolve_artifact_path
from deployment.primitives.device import DeviceSpec
from deployment.projects.yolox.inference.base_inference_pipeline import YOLOXDecodeParams, YOLOXInferencePipeline

logger = logging.getLogger(__name__)


class YOLOXONNXInferencePipeline(YOLOXInferencePipeline):
    """ONNX Runtime backend for YOLOX (single ``model`` component)."""

    def __init__(
        self,
        onnx_dir: str,
        device: DeviceSpec,
        decode_params: YOLOXDecodeParams,
        components_cfg: ComponentsConfig,
    ) -> None:
        super().__init__(
            model=None,
            backend_type=Backend.ONNX,
            device=device,
            decode_params=decode_params,
        )
        onnx_path = resolve_artifact_path(
            base_dir=onnx_dir,
            components_cfg=components_cfg,
            component_name="model",
            file_key="onnx_file",
        )

        session_options = ort.SessionOptions()
        # Disable graph optimizations so ONNX numerics stay close to PyTorch for verification
        # (matches the CenterPoint ONNX pipeline).
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        self._session = ort.InferenceSession(
            onnx_path, sess_options=session_options, providers=device.to_ort_provider()
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        logger.info("Loaded YOLOX ONNX model: %s (providers=%s)", onnx_path, self._session.get_providers())

    @override
    def run_model(self, preprocessed_input: torch.Tensor) -> Tuple[np.ndarray, Dict[str, float]]:
        """Run the ONNX session and return the raw output ``[1, num_anchors, 4+1+num_classes]``."""
        input_array = self.to_numpy(preprocessed_input, dtype=np.float32)
        outputs = self._session.run([self._output_name], {self._input_name: input_array})
        return outputs[0], {}
