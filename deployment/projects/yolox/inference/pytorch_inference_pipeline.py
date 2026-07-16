"""YOLOX PyTorch inference pipeline (reference backend)."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import torch
from typing_extensions import override

from deployment.config.enums import Backend
from deployment.primitives.device import DeviceSpec
from deployment.projects.yolox.export.model_wrappers import YOLOXONNXWrapper
from deployment.projects.yolox.inference.base_inference_pipeline import YOLOXDecodeParams, YOLOXInferencePipeline

logger = logging.getLogger(__name__)


class YOLOXPyTorchInferencePipeline(YOLOXInferencePipeline):
    """PyTorch backend for YOLOX.

    Runs the reference model through :class:`YOLOXONNXWrapper` — the *same* module used for ONNX
    export — so the raw output is byte-identical in structure to the ONNX/TensorRT graphs, keeping
    verification meaningful.
    """

    def __init__(
        self,
        pytorch_model: torch.nn.Module,
        device: DeviceSpec,
        decode_params: YOLOXDecodeParams,
    ) -> None:
        super().__init__(
            model=pytorch_model,
            backend_type=Backend.PYTORCH,
            device=device,
            decode_params=decode_params,
        )
        # No parameters of its own; delegates to the (already device-placed) reference model.
        self._export_model = YOLOXONNXWrapper(pytorch_model)
        self._export_model.eval()

    @override
    def run_model(self, preprocessed_input: torch.Tensor) -> Tuple[np.ndarray, Dict[str, float]]:
        """Run backbone+neck+head and return the raw Tier4-layout output as numpy."""
        with torch.no_grad():
            output = self._export_model(preprocessed_input)
        return output.detach().cpu().numpy(), {}
