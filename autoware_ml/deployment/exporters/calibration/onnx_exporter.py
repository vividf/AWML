"""
CalibrationStatusClassification-specific ONNX exporter.

This is a thin wrapper around the base ONNXExporter.
Calibration uses the standard ONNX export flow without special modifications.
"""

import logging
from typing import Any, Dict, Optional

from autoware_ml.deployment.exporters.base.onnx_exporter import ONNXExporter


class CalibrationONNXExporter(ONNXExporter):
    """
    CalibrationStatusClassification-specific ONNX exporter.
    
    Inherits from ONNXExporter and uses all base functionality.
    No Calibration-specific modifications needed for ONNX export.
    
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
        Initialize Calibration ONNX exporter.
        
        Args:
            config: ONNX export configuration
            logger: Optional logger instance
            model_wrapper: Optional model wrapper class (default: IdentityWrapper)
        """
        from autoware_ml.deployment.exporters.base.model_wrappers import IdentityWrapper
        # Use provided wrapper or default to IdentityWrapper
        if model_wrapper is None:
            model_wrapper = IdentityWrapper
        super().__init__(config, logger, model_wrapper=model_wrapper)

