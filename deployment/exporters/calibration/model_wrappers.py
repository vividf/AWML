"""
CalibrationStatusClassification-specific model wrappers for ONNX export.

Calibration models don't require special output format conversion,
so we use IdentityWrapper (no modification to model output).
"""

from deployment.exporters.common.model_wrappers import BaseModelWrapper, IdentityWrapper

# Calibration doesn't need special wrapper, use IdentityWrapper
CalibrationONNXWrapper = IdentityWrapper

__all__ = [
    "CalibrationONNXWrapper",
]
