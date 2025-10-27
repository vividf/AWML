"""ONNX Runtime inference backend."""

import logging
import os
import time
from typing import Optional, Tuple, Dict, Any

import numpy as np
import onnxruntime as ort
import torch

from .base_backend import BaseBackend

class ONNXBackend(BaseBackend):
    """
    ONNX Runtime inference backend.

    Runs inference using ONNX Runtime on CPU or CUDA.
    """

    def __init__(self, model_path: str, device: str = "cpu", num_classes: int = None, pytorch_model=None):
        """
        Initialize ONNX backend.

        Args:
            model_path: Path to ONNX model file or directory (for CenterPoint)
            device: Device to run inference on ('cpu' or 'cuda')
            num_classes: Number of classes (used to filter multi-output ONNX models)
            pytorch_model: PyTorch model (required for CenterPoint)
        """
        super().__init__(model_path, device)
        self._session = None
        self._fallback_attempted = False
        self._logger = logging.getLogger(__name__)
        self.num_classes = num_classes
        self.pytorch_model = pytorch_model
        
        self.centerpoint_helper = None

    def load_model(self) -> None:
        """Load ONNX model(s)."""
        self._load_standard_model()

    def _load_standard_model(self) -> None:
        """Load standard single-file ONNX model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        # Select execution provider based on device
        if self.device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._logger.info("Attempting to use CUDA acceleration (will fallback to CPU if needed)...")
        else:
            providers = ["CPUExecutionProvider"]
            self._logger.info("Using CPU for ONNX inference")

        try:
            self._session = ort.InferenceSession(self.model_path, providers=providers)
            self._model = self._session  # For is_loaded check
            self._logger.info(f"ONNX session using providers: {self._session.get_providers()}")
        except Exception as e:
            # If CUDA provider fails, try CPU fallback
            if self.device.startswith("cuda") and not self._fallback_attempted:
                self._logger.warning(f"CUDA provider failed: {e}")
                self._logger.warning("Attempting CPU fallback...")
                self._fallback_attempted = True
                try:
                    self._session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
                    self._model = self._session
                    self._logger.info(f"ONNX session using providers: {self._session.get_providers()}")
                except Exception as cpu_e:
                    raise RuntimeError(
                        f"Failed to load ONNX model with both CUDA and CPU providers: CUDA={e}, CPU={cpu_e}"
                    )
            else:
                raise RuntimeError(f"Failed to load ONNX model: {e}")

    def infer(self, input_tensor) -> Tuple[np.ndarray, float]:
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

        # Check if ONNX model expects different batch size than input
        input_name = self._session.get_inputs()[0].name
        expected_shape = self._session.get_inputs()[0].shape
        
        # Extract expected batch size (handle dynamic dimensions)
        expected_batch_size = self._extract_batch_size(expected_shape)
        
        # Handle batch size mismatch for fixed batch size models
        if expected_batch_size is not None and input_array.shape[0] != expected_batch_size:
            self._logger.info(f"ONNX model expects batch size {expected_batch_size}, input has batch size {input_array.shape[0]}")
            if input_array.shape[0] == 1 and expected_batch_size > 1:
                # Repeat single sample to match expected batch size
                self._logger.info(f"Repeating single sample to batch size {expected_batch_size}")
                input_array = np.repeat(input_array, expected_batch_size, axis=0)
            else:
                raise ValueError(f"Batch size mismatch: ONNX expects {expected_batch_size}, got {input_array.shape[0]}")

        # Prepare input dictionary
        onnx_input = {input_name: input_array}

        # Run inference
        start_time = time.perf_counter()
        outputs = self._session.run(None, onnx_input)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        output = outputs[0]
        self._logger.info(f"ONNX output shape: {output.shape}")
        
        return output, latency_ms


    def _extract_batch_size(self, shape) -> Optional[int]:
        """
        Extract batch size from ONNX model input shape.
        
        Args:
            shape: ONNX model input shape (can contain dynamic dimensions as strings)
            
        Returns:
            Batch size as integer if fixed, None if dynamic
        """
        if not shape or len(shape) == 0:
            return None
        
        batch_dim = shape[0]
        return batch_dim if isinstance(batch_dim, int) and batch_dim > 0 else None

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid activation to numpy array."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

    def cleanup(self) -> None:
        """Clean up ONNX Runtime resources."""
        self._session = None
        self._model = None
        self._fallback_attempted = False
