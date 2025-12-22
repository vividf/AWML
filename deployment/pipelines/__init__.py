"""
Deployment Pipelines for Complex Models.

This module provides pipeline abstractions for models that require
multi-stage processing with mixed PyTorch and optimized backend inference.
"""

# CenterPoint pipelines
from deployment.pipelines.centerpoint import (
    CenterPointDeploymentPipeline,
    CenterPointONNXPipeline,
    CenterPointPyTorchPipeline,
    CenterPointTensorRTPipeline,
)

# Pipeline factory
from deployment.pipelines.factory import PipelineFactory

# Add pipelines here


__all__ = [
    # Factory
    "PipelineFactory",
    # CenterPoint
    "CenterPointDeploymentPipeline",
    "CenterPointPyTorchPipeline",
    "CenterPointONNXPipeline",
    "CenterPointTensorRTPipeline",
    # Add pipelines here
]
