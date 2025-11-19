"""
YOLOX-specific TensorRT exporter.

This is a thin wrapper around the base TensorRTExporter.
YOLOX uses the standard TensorRT export flow.
"""

import logging
from typing import Any, Dict, Optional

from deployment.exporters.base.configs import TensorRTExportConfig
from deployment.exporters.base.tensorrt_exporter import TensorRTExporter


class YOLOXTensorRTExporter(TensorRTExporter):
    """
    YOLOX-specific TensorRT exporter.

    Inherits from TensorRTExporter and uses all base functionality.
    No YOLOX-specific modifications needed for TensorRT export.

    This class exists for architectural consistency - all model-specific
    exporters follow the same pattern, even if they just use base functionality.
    """

    def __init__(
        self,
        config: TensorRTExportConfig,
        model_wrapper: Optional[Any] = None,
        logger: logging.Logger = None,
    ):
        """
        Initialize YOLOX TensorRT exporter.

        Args:
            config: TensorRT export configuration dataclass instance.
            model_wrapper: Optional model wrapper class (usually not needed for TensorRT)
            logger: Optional logger instance
        """
        super().__init__(config, model_wrapper=model_wrapper, logger=logger)
