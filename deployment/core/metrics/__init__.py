"""
Unified Metrics Adapters for AWML Deployment Framework.

This module provides task-specific metric adapters that use autoware_perception_evaluation
as the single source of truth for metric computation. This ensures consistency between
training evaluation (T4MetricV2) and deployment evaluation.

Design Principles:
    1. 3D Detection → Detection3DMetricsAdapter (mAP, mAPH using autoware_perception_eval)
    2. 2D Detection → Detection2DMetricsAdapter (mAP using autoware_perception_eval, 2D mode)
    3. Classification → ClassificationMetricsAdapter (accuracy, precision, recall, F1)

Usage:
    # For 3D detection (CenterPoint, etc.)
    from deployment.core.metrics import Detection3DMetricsAdapter, Detection3DMetricsConfig

    config = Detection3DMetricsConfig(
        class_names=["car", "truck", "bus", "bicycle", "pedestrian"],
    )
    adapter = Detection3DMetricsAdapter(config)
    adapter.add_frame(predictions, ground_truths)
    metrics = adapter.compute_metrics()

    # For 2D detection (YOLOX, etc.)
    from deployment.core.metrics import Detection2DMetricsAdapter, Detection2DMetricsConfig

    config = Detection2DMetricsConfig(
        class_names=["car", "truck", "bus", ...],
    )
    adapter = Detection2DMetricsAdapter(config)
    adapter.add_frame(predictions, ground_truths)
    metrics = adapter.compute_metrics()

    # For classification (Calibration, etc.)
    from deployment.core.metrics import ClassificationMetricsAdapter, ClassificationMetricsConfig

    config = ClassificationMetricsConfig(
        class_names=["miscalibrated", "calibrated"],
    )
    adapter = ClassificationMetricsAdapter(config)
    adapter.add_frame(prediction_label, ground_truth_label, probabilities)
    metrics = adapter.compute_metrics()
"""

from deployment.core.metrics.base_metrics_adapter import (
    BaseMetricsAdapter,
    BaseMetricsConfig,
)
from deployment.core.metrics.classification_metrics import (
    ClassificationMetricsAdapter,
    ClassificationMetricsConfig,
)
from deployment.core.metrics.detection_2d_metrics import (
    Detection2DMetricsAdapter,
    Detection2DMetricsConfig,
)
from deployment.core.metrics.detection_3d_metrics import (
    Detection3DMetricsAdapter,
    Detection3DMetricsConfig,
)

__all__ = [
    # Base classes
    "BaseMetricsAdapter",
    "BaseMetricsConfig",
    # 3D Detection
    "Detection3DMetricsAdapter",
    "Detection3DMetricsConfig",
    # 2D Detection
    "Detection2DMetricsAdapter",
    "Detection2DMetricsConfig",
    # Classification
    "ClassificationMetricsAdapter",
    "ClassificationMetricsConfig",
]
