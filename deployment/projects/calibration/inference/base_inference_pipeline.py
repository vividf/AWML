"""Calibration-classifier inference pipeline base (shared preprocess + softmax postprocess).

The model emits raw class logits ``[1, num_classes]`` (softmax is not in the exported graph), so
postprocess applies a numerically-stable softmax + argmax and returns a classification result dict.
Backends override only ``run_model``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from deployment.config.enums import Backend
from deployment.inference.base_inference_pipeline import BaseInferencePipeline
from deployment.primitives.device import DeviceSpec

logger = logging.getLogger(__name__)


class ClassificationInferencePipeline(BaseInferencePipeline):
    """Base single-label classification pipeline (preprocess + softmax postprocess shared)."""

    def __init__(
        self,
        model: Any,
        backend_type: Backend,
        device: DeviceSpec,
        class_names: Sequence[str],
    ) -> None:
        super().__init__(model=model, backend_type=backend_type, device=device)
        self.class_names: List[str] = list(class_names)

    def preprocess(self, input_data: Any) -> torch.Tensor:
        """Return the (already loader-preprocessed) input tensor as float32 on the pipeline device."""
        tensor = input_data
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        return tensor.to(self.torch_device)

    def postprocess(
        self,
        model_output: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Softmax the logits and return ``{class_id, class_name, confidence, probabilities, top_k}``."""
        if isinstance(model_output, torch.Tensor):
            logits = model_output.detach().cpu().numpy()
        else:
            logits = np.asarray(model_output)
        if logits.ndim == 2:
            logits = logits[0]

        # Numerically stable softmax.
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        class_id = int(np.argmax(probabilities))
        order = np.argsort(probabilities)[::-1]
        top_k = [
            {
                "class_id": int(i),
                "class_name": self._name(int(i)),
                "confidence": float(probabilities[i]),
            }
            for i in order
        ]
        return {
            "class_id": class_id,
            "class_name": self._name(class_id),
            "confidence": float(probabilities[class_id]),
            "probabilities": probabilities,
            "top_k": top_k,
        }

    def _name(self, class_id: int) -> str:
        """Class name for an index, falling back to ``class_<id>`` when out of range."""
        return self.class_names[class_id] if 0 <= class_id < len(self.class_names) else f"class_{class_id}"
