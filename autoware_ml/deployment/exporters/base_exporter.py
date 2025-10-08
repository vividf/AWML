"""
Abstract base class for model exporters.

Provides a unified interface for exporting models to different formats.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class BaseExporter(ABC):
    """
    Abstract base class for model exporters.

    This class defines a unified interface for exporting models
    to different backend formats (ONNX, TensorRT, TorchScript, etc.).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize exporter.

        Args:
            config: Configuration dictionary for export settings
        """
        self.config = config

    @abstractmethod
    def export(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        output_path: str,
    ) -> bool:
        """
        Export model to target format.

        Args:
            model: PyTorch model to export
            sample_input: Sample input tensor for tracing/shape inference
            output_path: Path to save exported model

        Returns:
            True if export succeeded, False otherwise

        Raises:
            RuntimeError: If export fails
        """
        pass

    def validate_export(self, output_path: str) -> bool:
        """
        Validate that the exported model file is valid.

        Override this in subclasses to add format-specific validation.

        Args:
            output_path: Path to exported model file

        Returns:
            True if valid, False otherwise
        """
        import os

        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
