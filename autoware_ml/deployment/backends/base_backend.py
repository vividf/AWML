"""
Abstract base class for inference backends.

Provides a unified interface for running inference across different
backend formats (PyTorch, ONNX, TensorRT, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
import torch


class BaseBackend(ABC):
    """
    Abstract base class for inference backends.

    This class defines a unified interface for running inference
    across different model formats and runtime engines.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize backend.

        Args:
            model_path: Path to model file
            device: Device to run inference on ('cpu', 'cuda', 'cuda:0', etc.)
        """
        self.model_path = model_path
        self.device = device
        self._model = None

    @abstractmethod
    def load_model(self) -> None:
        """
        Load model from file.

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        pass

    @abstractmethod
    def infer(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        Run inference on input tensor.

        Args:
            input_tensor: Input tensor for inference

        Returns:
            Tuple of (output_array, latency_ms):
            - output_array: Model output as numpy array
            - latency_ms: Inference time in milliseconds

        Raises:
            RuntimeError: If inference fails
            ValueError: If input format is invalid
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def cleanup(self) -> None:
        """
        Clean up resources.

        Override this method in subclasses to release backend-specific resources.
        """
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
