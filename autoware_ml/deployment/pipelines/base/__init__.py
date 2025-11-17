"""
Base Pipeline Classes for Deployment Framework.

This module provides the base abstract classes for all deployment pipelines,
including base pipeline, classification, 2D detection, and 3D detection pipelines.
"""

from autoware_ml.deployment.pipelines.base.base_pipeline import BaseDeploymentPipeline
from autoware_ml.deployment.pipelines.base.classification_pipeline import ClassificationPipeline
from autoware_ml.deployment.pipelines.base.detection_2d_pipeline import Detection2DPipeline
from autoware_ml.deployment.pipelines.base.detection_3d_pipeline import Detection3DPipeline

__all__ = [
    "BaseDeploymentPipeline",
    "ClassificationPipeline",
    "Detection2DPipeline",
    "Detection3DPipeline",
]
