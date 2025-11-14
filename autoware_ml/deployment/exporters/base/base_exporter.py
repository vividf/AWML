"""
Abstract base class for model exporters.

Provides a unified interface for exporting models to different formats.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable
import logging

import torch


class BaseExporter(ABC):
    """
    Abstract base class for model exporters.

    This class defines a unified interface for exporting models
    to different backend formats (ONNX, TensorRT, TorchScript, etc.).
    
    Enhanced features:
    - Support for model wrappers (preprocessing before export)
    - Flexible configuration with overrides
    - Better logging and error handling
    """

    def __init__(
        self, 
        config: Dict[str, Any], 
        logger: logging.Logger = None,
        model_wrapper: Optional[Any] = None
    ):
        """
        Initialize exporter.

        Args:
            config: Configuration dictionary for export settings
            logger: Optional logger instance
            model_wrapper: Optional model wrapper class or instance.
                         If a class is provided, it will be instantiated with the model.
                         If an instance is provided, it should be a callable that takes a model.
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._model_wrapper = model_wrapper
    
    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Prepare model for export (apply wrapper if configured).
        
        Args:
            model: Original PyTorch model
            
        Returns:
            Prepared model (wrapped if wrapper configured)
        """
        if self._model_wrapper is None:
            return model
        
        self.logger.info("Applying model wrapper for export")
        
        # If model_wrapper is a class, instantiate it with the model
        if isinstance(self._model_wrapper, type):
            return self._model_wrapper(model)
        # If model_wrapper is a callable (function or instance with __call__), use it
        elif callable(self._model_wrapper):
            return self._model_wrapper(model)
        else:
            raise TypeError(
                f"model_wrapper must be a class or callable, got {type(self._model_wrapper)}"
            )

    @abstractmethod
    def export(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        output_path: str,
        **kwargs
    ) -> bool:
        """
        Export model to target format.

        Args:
            model: PyTorch model to export
            sample_input: Sample input tensor for tracing/shape inference
            output_path: Path to save exported model
            **kwargs: Additional format-specific arguments

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
