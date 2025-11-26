"""
CalibrationStatusClassification deployment utilities.

This module provides utility functions for CalibrationStatusClassification model deployment,
including metrics configuration extraction for autoware_perception_evaluation.
"""

import logging
from typing import List, Optional

from mmengine.config import Config

from deployment.core import ClassificationMetricsConfig

# Default class names for calibration status classification
DEFAULT_CLASS_NAMES = ["miscalibrated", "calibrated"]


def extract_classification_metrics_config(
    model_cfg: Config,
    class_names: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> ClassificationMetricsConfig:
    """
    Extract or create ClassificationMetricsConfig for autoware_perception_evaluation.

    This function extracts class names from model config to create a
    ClassificationMetricsConfig that is compatible with autoware_perception_evaluation
    for classification tasks.

    Args:
        model_cfg: Model configuration (mmpretrain config)
        class_names: Optional list of class names. If not provided, will try to extract
                    from model_cfg or use defaults ["miscalibrated", "calibrated"].
        logger: Optional logger instance for logging

    Returns:
        ClassificationMetricsConfig configured for autoware_perception_evaluation
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Get class names
    if class_names is None:
        # Try to extract from model config
        if hasattr(model_cfg, "classes"):
            classes = model_cfg.classes
            if isinstance(classes, (tuple, list)):
                class_names = list(classes)
                logger.info(f"Extracted {len(class_names)} classes from model config: {class_names}")
            else:
                logger.warning(f"Config 'classes' is not a list/tuple, using defaults. " f"Got {type(classes)}")
                class_names = DEFAULT_CLASS_NAMES
        elif hasattr(model_cfg, "model") and hasattr(model_cfg.model, "head"):
            # Try to get num_classes from model head
            num_classes = getattr(model_cfg.model.head, "num_classes", None)
            if num_classes == 2:
                class_names = DEFAULT_CLASS_NAMES
                logger.info(f"Using default class names for binary classification: {class_names}")
            else:
                logger.warning(f"Cannot determine class names. num_classes={num_classes}. " f"Using defaults.")
                class_names = DEFAULT_CLASS_NAMES
        else:
            # Use default class names for calibration status classification
            class_names = DEFAULT_CLASS_NAMES
            logger.info(f"Using default class names: {class_names}")

    logger.info("=" * 60)
    logger.info("Creating ClassificationMetricsConfig for autoware_perception_evaluation")
    logger.info("=" * 60)
    logger.info(f"  - class_names: {class_names}")
    logger.info(f"  - num_classes: {len(class_names)}")
    logger.info("=" * 60)

    return ClassificationMetricsConfig(
        class_names=class_names,
        frame_id="classification",  # Default frame_id for classification
    )
