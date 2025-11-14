"""
Deployment Pipelines for Complex Models.

This module provides pipeline abstractions for models that require
multi-stage processing with mixed PyTorch and optimized backend inference.
"""

# CenterPoint pipelines (3D detection)
from autoware_ml.deployment.pipelines.centerpoint import (
    CenterPointDeploymentPipeline,
    CenterPointPyTorchPipeline,
    CenterPointONNXPipeline,
    CenterPointTensorRTPipeline,
)

# YOLOX pipelines (2D detection)
from autoware_ml.deployment.pipelines.yolox import (
    YOLOXDeploymentPipeline,
    YOLOXPyTorchPipeline,
    YOLOXONNXPipeline,
    YOLOXTensorRTPipeline,
)

# Calibration pipelines (classification)
from autoware_ml.deployment.pipelines.calibration import (
    CalibrationDeploymentPipeline,
    CalibrationPyTorchPipeline,
    CalibrationONNXPipeline,
    CalibrationTensorRTPipeline,
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
    # Calibration
    'CalibrationDeploymentPipeline',
    'CalibrationPyTorchPipeline',
    'CalibrationONNXPipeline',
    'CalibrationTensorRTPipeline',
]

