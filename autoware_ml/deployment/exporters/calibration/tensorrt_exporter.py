"""
CalibrationStatusClassification-specific TensorRT exporter.

This is a thin wrapper around the base TensorRTExporter.
Calibration uses the standard TensorRT export flow without special modifications.
"""

import logging
from typing import Any, Dict, Optional

from autoware_ml.deployment.exporters.base.tensorrt_exporter import TensorRTExporter


class CalibrationTensorRTExporter(TensorRTExporter):
    """
    CalibrationStatusClassification-specific TensorRT exporter.

    Inherits from TensorRTExporter and uses all base functionality.
    No Calibration-specific modifications needed for TensorRT export.

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
        Initialize Calibration TensorRT exporter.

        Args:
            config: TensorRT export configuration
            model_wrapper: Optional model wrapper class (usually not needed for TensorRT)
            logger: Optional logger instance
        """
        super().__init__(config, model_wrapper=model_wrapper, logger=logger)
