"""
YOLOX_opt_elan deployment utilities.

This module provides utility functions for YOLOX_opt_elan model deployment,
including metrics configuration extraction for autoware_perception_evaluation.
"""

import logging
from typing import List, Optional

from mmengine.config import Config

from deployment.core import Detection2DMetricsConfig


def extract_detection2d_metrics_config(
    model_cfg: Config,
    class_names: Optional[List[str]] = None,
    iou_thresholds: Optional[List[float]] = None,
    frame_id: str = "cam_front",
    logger: Optional[logging.Logger] = None,
) -> Detection2DMetricsConfig:
    """
    Extract or create Detection2DMetricsConfig for autoware_perception_evaluation.

    This function extracts class names and evaluation settings from model config
    to create a Detection2DMetricsConfig that is compatible with
    autoware_perception_evaluation for 2D object detection.

    Args:
        model_cfg: Model configuration (mmdet config)
        class_names: Optional list of class names. If not provided, will be extracted from model_cfg.
        iou_thresholds: Optional list of IoU thresholds for evaluation. Defaults to [0.5, 0.75].
        frame_id: Camera frame ID for evaluation. Valid values:
            "cam_front", "cam_front_right", "cam_front_left", "cam_front_lower",
            "cam_back", "cam_back_left", "cam_back_right",
            "cam_traffic_light_near", "cam_traffic_light_far", "cam_traffic_light"
            Defaults to "cam_front".
        logger: Optional logger instance for logging

    Returns:
        Detection2DMetricsConfig configured for autoware_perception_evaluation

    Raises:
        ValueError: If class_names cannot be extracted from model_cfg
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Get class names from config if not provided
    if class_names is None:
        if hasattr(model_cfg, "classes"):
            classes = model_cfg.classes
            if not isinstance(classes, (tuple, list)):
                raise ValueError(
                    f"Config 'classes' must be a tuple or list, got {type(classes)}. "
                    f"Please check your dataset config file."
                )
            class_names = list(classes)
            logger.info(f"Extracted {len(class_names)} classes from model config: {class_names}")
        else:
            raise ValueError(
                "class_names not provided and config file does not contain 'classes' attribute. "
                "Please provide class_names or ensure your dataset config file defines 'classes'."
            )

    # Set default IoU thresholds if not provided
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]  # Pascal VOC style thresholds

    logger.info("=" * 60)
    logger.info("Creating Detection2DMetricsConfig for autoware_perception_evaluation")
    logger.info("=" * 60)
    logger.info(f"  - class_names: {class_names}")
    logger.info(f"  - frame_id: {frame_id}")
    logger.info(f"  - iou_thresholds: {iou_thresholds}")

    # Create evaluation config dict for 2D detection
    evaluation_config_dict = {
        "evaluation_task": "detection2d",
        "target_labels": class_names,
        "iou_2d_thresholds": iou_thresholds,
        "center_distance_bev_thresholds": None,
        "plane_distance_thresholds": None,
        "iou_3d_thresholds": None,
        "label_prefix": "autoware",
    }

    # Create critical object filter config
    critical_object_filter_config = {
        "target_labels": class_names,
        "ignore_attributes": None,
    }

    # Create frame pass fail config
    num_classes = len(class_names)
    frame_pass_fail_config = {
        "target_labels": class_names,
        "matching_threshold_list": [0.5] * num_classes,
        "confidence_threshold_list": None,
    }

    logger.info("=" * 60)

    return Detection2DMetricsConfig(
        class_names=class_names,
        frame_id=frame_id,
        iou_thresholds=iou_thresholds,
        evaluation_config_dict=evaluation_config_dict,
        critical_object_filter_config=critical_object_filter_config,
        frame_pass_fail_config=frame_pass_fail_config,
    )
