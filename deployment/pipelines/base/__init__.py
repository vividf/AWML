"""
Base Pipeline Classes for Deployment Framework.

This module provides the base abstract classes for all deployment pipelines,
including base pipeline, classification, 2D detection, and 3D detection pipelines.
"""

from deployment.pipelines.base.base_pipeline import BaseDeploymentPipeline

__all__ = [
    "BaseDeploymentPipeline",
]
