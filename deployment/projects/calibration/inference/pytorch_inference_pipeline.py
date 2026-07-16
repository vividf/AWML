"""Calibration classifier PyTorch inference pipeline (reference backend)."""

from __future__ import annotations

import logging
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from typing_extensions import override

from deployment.config.enums import Backend
from deployment.primitives.device import DeviceSpec
from deployment.projects.calibration.inference.base_inference_pipeline import ClassificationInferencePipeline

logger = logging.getLogger(__name__)


class CalibrationPyTorchInferencePipeline(ClassificationInferencePipeline):
    """PyTorch backend: runs the mmpretrain classifier in ``mode='tensor'`` to get raw logits."""

    def __init__(
        self,
        pytorch_model: torch.nn.Module,
        device: DeviceSpec,
        class_names: Sequence[str],
    ) -> None:
        super().__init__(
            model=pytorch_model,
            backend_type=Backend.PYTORCH,
            device=device,
            class_names=class_names,
        )

    @override
    def run_model(self, preprocessed_input: torch.Tensor) -> Tuple[np.ndarray, Dict[str, float]]:
        """Return raw class logits ``[1, num_classes]`` as numpy."""
        with torch.no_grad():
            logits = self.model(preprocessed_input)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        logits_np = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else np.asarray(logits)
        return logits_np, {}
