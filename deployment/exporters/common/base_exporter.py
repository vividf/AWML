"""
Abstract base class for model exporters.

Provides a interface for exporting models to different formats.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from deployment.core.artifacts import Artifact
from deployment.exporters.common.configs import BaseExporterConfig
from deployment.exporters.common.model_wrappers import BaseModelWrapper


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
        config: BaseExporterConfig,
        logger: logging.Logger,
        model_wrapper: Optional[BaseModelWrapper] = None,
    ) -> None:
        """
        Initialize exporter.

        Args:
            config: Typed export configuration dataclass (e.g., ``ONNXExportConfig``,
                ``TensorRTExportConfig``). This ensures type safety and clear schema.
            logger: Logger instance for export progress and diagnostics.
            model_wrapper: Optional model wrapper class or callable.
                         If a class is provided, it will be instantiated with the model.
                         If an instance is provided, it should be a callable that takes a model.
        """
        self.config: BaseExporterConfig = config
        self.logger = logger
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

        return self._model_wrapper(model)

    @abstractmethod
    def export(
        self,
        model: torch.nn.Module,
        sample_input: Any,
        output_path: str,
        onnx_path: Optional[str] = None,
    ) -> Optional["Artifact"]:
        """
        Export model to target format.

        Args:
            model: PyTorch model to export
            sample_input: Example model input(s) for tracing/shape inference
            output_path: Path to save exported model
            onnx_path: Path to a source ONNX model, for exporters that convert from
                ONNX (e.g. TensorRT). Ignored by exporters that export directly from
                a PyTorch model.

        Returns:
            Artifact representing the exported model, or None if the exporter
            does not produce one (the model is written to `output_path` either way).

        Raises:
            RuntimeError: If export fails
        """
        raise NotImplementedError
