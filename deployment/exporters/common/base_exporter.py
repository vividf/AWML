"""
Base class for model exporters.

Provides shared construction (typed config + optional model wrapper) and the
``prepare_model`` hook. Concrete exporters define their own ``export`` signature:
``ONNXExporter`` exports from a PyTorch model while ``TensorRTExporter`` converts
from an existing ONNX file, so they intentionally do not share one ``export``
contract (forcing one would leave each implementation with dead parameters).
"""

import logging
from typing import Optional

import torch

from deployment.exporters.common.configs import BaseExporterConfig
from deployment.exporters.common.model_wrappers import BaseModelWrapper

logger = logging.getLogger(__name__)


class BaseExporter:
    """
    Shared base for model exporters.

    Owns the typed export configuration and the optional model wrapper applied
    before export. Concrete exporters (ONNX, TensorRT, ...) add their own
    ``export`` method with a signature that fits the format they produce.
    """

    def __init__(
        self,
        config: BaseExporterConfig,
        model_wrapper: Optional[BaseModelWrapper] = None,
    ) -> None:
        """
        Initialize exporter.

        Args:
            config: Typed export configuration dataclass (e.g., ``ONNXExportConfig``,
                ``TensorRTExportConfig``). This ensures type safety and clear schema.
            model_wrapper: Optional model wrapper class or callable.
                         If a class is provided, it will be instantiated with the model.
                         If an instance is provided, it should be a callable that takes a model.
        """
        self.config: BaseExporterConfig = config
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

        logger.info("Applying model wrapper for export")

        return self._model_wrapper(model)
