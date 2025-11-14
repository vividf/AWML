"""
CenterPoint-specific model wrappers for ONNX export.

CenterPoint models don't require special output format conversion,
so we use IdentityWrapper (no modification to model output).
"""

from autoware_ml.deployment.exporters.base.model_wrappers import IdentityWrapper, BaseModelWrapper

# CenterPoint doesn't need special wrapper, use IdentityWrapper
CenterPointONNXWrapper = IdentityWrapper

__all__ = [
    "CenterPointONNXWrapper",
]

