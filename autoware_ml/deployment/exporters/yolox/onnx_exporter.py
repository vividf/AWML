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
        logger: logging.Logger = None,
        model_wrapper: Optional[Any] = None
    ):
        """
        Initialize YOLOX ONNX exporter.
        
        Args:
            config: ONNX export configuration
            logger: Optional logger instance
            model_wrapper: Optional model wrapper class (default: YOLOXONNXWrapper)
        """
        from autoware_ml.deployment.exporters.yolox.model_wrappers import YOLOXONNXWrapper
        # Use provided wrapper or default to YOLOXONNXWrapper
        if model_wrapper is None:
            model_wrapper = YOLOXONNXWrapper
        super().__init__(config, logger, model_wrapper=model_wrapper)

