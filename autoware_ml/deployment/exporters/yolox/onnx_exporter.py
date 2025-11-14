"""
YOLOX-specific ONNX exporter.

This is a thin wrapper around the base ONNXExporter.
YOLOX uses the standard ONNX export flow with a model wrapper
configured via config (model_wrapper: 'yolox').
"""

import logging
from typing import Any, Dict

from autoware_ml.deployment.exporters.base.onnx_exporter import ONNXExporter


class YOLOXONNXExporter(ONNXExporter):
    """
    YOLOX-specific ONNX exporter.
    
    Inherits from ONNXExporter and uses all base functionality.
    The model wrapper (YOLOXONNXWrapper) is automatically applied
    via config (model_wrapper: 'yolox').
    
    This class exists for architectural consistency - all model-specific
    exporters follow the same pattern, even if they just use base functionality.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger = None):
        """
        Initialize YOLOX ONNX exporter.
        
        Args:
            config: ONNX export configuration (should include model_wrapper: 'yolox')
            logger: Optional logger instance
        """
        super().__init__(config, logger)

