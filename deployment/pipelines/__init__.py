"""
Deployment Pipelines for Complex Models.

This module provides pipeline abstractions for models that require
multi-stage processing with mixed PyTorch and optimized backend inference.
"""

# Calibration pipelines (classification)
# from deployment.pipelines.calibration import (
#     CalibrationDeploymentPipeline,
#     CalibrationONNXPipeline,
#     CalibrationPyTorchPipeline,
#     CalibrationTensorRTPipeline,
# )

# CenterPoint pipelines (3D detection)
from deployment.pipelines.centerpoint import (
    CenterPointDeploymentPipeline,
    CenterPointONNXPipeline,
    CenterPointPyTorchPipeline,
    CenterPointTensorRTPipeline,
)

# Pipeline factory
from deployment.pipelines.factory import PipelineFactory

# YOLOX pipelines (2D detection)
# from deployment.pipelines.yolox import (
#     YOLOXDeploymentPipeline,
#     YOLOXONNXPipeline,
#     YOLOXPyTorchPipeline,
#     YOLOXTensorRTPipeline,
# )

__all__ = [
    # Factory
    "PipelineFactory",
    # CenterPoint
    "CenterPointDeploymentPipeline",
    "CenterPointPyTorchPipeline",
    "CenterPointONNXPipeline",
    "CenterPointTensorRTPipeline",
    # YOLOX
    # "YOLOXDeploymentPipeline",
    # "YOLOXPyTorchPipeline",
    # "YOLOXONNXPipeline",
    # "YOLOXTensorRTPipeline",
    # Calibration
    # "CalibrationDeploymentPipeline",
    # "CalibrationPyTorchPipeline",
    # "CalibrationONNXPipeline",
    # "CalibrationTensorRTPipeline",
]
