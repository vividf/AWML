"""PyTorch inference backend."""

import time
from typing import Tuple

import numpy as np
import torch

from .base_backend import BaseBackend


class PyTorchBackend(BaseBackend):
    """
    PyTorch inference backend.

    Runs inference using native PyTorch models.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        """
        Initialize PyTorch backend.

        Args:
            model: PyTorch model instance (already loaded)
            device: Device to run inference on
        """
        super().__init__(model_path="", device=device)
        self._model = model
        self._torch_device = torch.device(device)
        self._model.to(self._torch_device)
        self._model.eval()

    def load_model(self) -> None:
        """Model is already loaded in __init__."""
        if self._model is None:
            raise RuntimeError("Model was not provided during initialization")

    def infer(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        Run inference on input tensor.

        Args:
            input_tensor: Input tensor for inference

        Returns:
            Tuple of (output_array, latency_ms)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Move input to correct device
        input_tensor = input_tensor.to(self._torch_device)

        # Run inference with timing
        with torch.no_grad():
            start_time = time.perf_counter()
            output = self._model(input_tensor)
            end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        # Handle different output formats
        if hasattr(output, "output"):
            output = output.output
        elif isinstance(output, dict) and "output" in output:
            output = output["output"]

        if not isinstance(output, torch.Tensor):
            raise ValueError(f"Unexpected PyTorch output type: {type(output)}")

        # Convert to numpy
        output_array = output.cpu().numpy()

        return output_array, latency_ms

    def cleanup(self) -> None:
        """Clean up PyTorch resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
