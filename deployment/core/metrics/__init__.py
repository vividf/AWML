"""
Metrics adapters for AWML deployment framework.

This module provides adapters to integrate autoware_perception_evaluation metrics
into the deployment framework for consistent evaluation across training and deployment.
"""

from .detection_3d_metrics import Detection3DMetricsAdapter

__all__ = ["Detection3DMetricsAdapter"]
