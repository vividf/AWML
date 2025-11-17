"""
YOLOX-specific ONNX exporter.
YOLOX uses the standard ONNX export flow with a model wrapper.
"""

import logging
from typing import Any, Dict, Optional

from autoware_ml.deployment.exporters.base.onnx_exporter import ONNXExporter


class YOLOXONNXExporter(ONNXExporter):
    """
    YOLOX-specific ONNX exporter.

    Inherits from ONNXExporter and uses all base functionality.
    The model wrapper (YOLOXONNXWrapper) is automatically applied by default.

    This class exists for architectural consistency - all model-specific
    exporters follow the same pattern, even if they just use base functionality.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_wrapper: Optional[Any] = None,
        logger: logging.Logger = None,
    ):
        """
        Initialize YOLOX ONNX exporter.

        Args:
            config: ONNX export configuration
            model_wrapper: Optional model wrapper class (e.g., YOLOXONNXWrapper)
            logger: Optional logger instance
        """
        super().__init__(config, model_wrapper=model_wrapper, logger=logger)
