"""
Deployment Pipelines for Complex Models.

This module provides pipeline abstractions for models that require
multi-stage processing with mixed PyTorch and optimized backend inference.
"""

# Calibration pipelines (classification)
from autoware_ml.deployment.pipelines.calibration import (
    CalibrationDeploymentPipeline,
    CalibrationONNXPipeline,
    CalibrationPyTorchPipeline,
    CalibrationTensorRTPipeline,
)

# CenterPoint pipelines (3D detection)
from autoware_ml.deployment.pipelines.centerpoint import (
    CenterPointDeploymentPipeline,
    CenterPointONNXPipeline,
    CenterPointPyTorchPipeline,
    CenterPointTensorRTPipeline,
)

# YOLOX pipelines (2D detection)
from autoware_ml.deployment.pipelines.yolox import (
    YOLOXDeploymentPipeline,
    YOLOXONNXPipeline,
    YOLOXPyTorchPipeline,
    YOLOXTensorRTPipeline,
)

__all__ = [
    # CenterPoint
    "CenterPointDeploymentPipeline",
    "CenterPointPyTorchPipeline",
    "CenterPointONNXPipeline",
    "CenterPointTensorRTPipeline",
    # YOLOX
    "YOLOXDeploymentPipeline",
    "YOLOXPyTorchPipeline",
    "YOLOXONNXPipeline",
    "YOLOXTensorRTPipeline",
    # Calibration
    "CalibrationDeploymentPipeline",
    "CalibrationPyTorchPipeline",
    "CalibrationONNXPipeline",
    "CalibrationTensorRTPipeline",
]
