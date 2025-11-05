"""
Deployment Pipelines for Complex Models.

This module provides pipeline abstractions for models that require
multi-stage processing with mixed PyTorch and optimized backend inference.
"""

# CenterPoint pipelines (3D detection)
from .centerpoint_pipeline import CenterPointDeploymentPipeline
from .centerpoint_pytorch import CenterPointPyTorchPipeline
from .centerpoint_onnx import CenterPointONNXPipeline
from .centerpoint_tensorrt import CenterPointTensorRTPipeline

# YOLOX pipelines (2D detection)
from .yolox import (
    YOLOXDeploymentPipeline,
    YOLOXPyTorchPipeline,
    YOLOXONNXPipeline,
    YOLOXTensorRTPipeline,
)

__all__ = [
    # CenterPoint
    'CenterPointDeploymentPipeline',
    'CenterPointPyTorchPipeline',
    'CenterPointONNXPipeline',
    'CenterPointTensorRTPipeline',
    # YOLOX
    'YOLOXDeploymentPipeline',
    'YOLOXPyTorchPipeline',
    'YOLOXONNXPipeline',
    'YOLOXTensorRTPipeline',
]

