"""ONNX Runtime inference backend."""

import os
import time
from typing import Tuple

import numpy as np
import onnxruntime as ort
import torch

from .base_backend import BaseBackend


class ONNXBackend(BaseBackend):
    """
    ONNX Runtime inference backend.

    Runs inference using ONNX Runtime on CPU or CUDA.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize ONNX backend.

        Args:
            model_path: Path to ONNX model file
            device: Device to run inference on ('cpu' or 'cuda')
        """
        super().__init__(model_path, device)
        self._session = None

    def load_model(self) -> None:
        """Load ONNX model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        # Select execution provider based on device
        if self.device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        try:
            self._session = ort.InferenceSession(self.model_path, providers=providers)
            self._model = self._session  # For is_loaded check
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")

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

        input_array = input_tensor.cpu().numpy()

        # Prepare input dictionary
        input_name = self._session.get_inputs()[0].name
        onnx_input = {input_name: input_array}

        # Run inference
        start_time = time.perf_counter()
        output = self._session.run(None, onnx_input)[0]
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        return output, latency_ms

    def cleanup(self) -> None:
        """Clean up ONNX Runtime resources."""
        self._session = None
        self._model = None
