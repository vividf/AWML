"""
Centralized constants for the deployment framework.

This module consolidates magic numbers and constants that were scattered
across multiple files into a single, configurable location.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationDefaults:
    """Default values for evaluation settings."""

    LOG_INTERVAL: int = 50
    GPU_CLEANUP_INTERVAL: int = 10
    VERIFICATION_TOLERANCE: float = 0.1
    DEFAULT_NUM_SAMPLES: int = 10
    DEFAULT_NUM_VERIFY_SAMPLES: int = 3


@dataclass(frozen=True)
class ExportDefaults:
    """Default values for export settings."""

    ONNX_DIR_NAME: str = "onnx"
    TENSORRT_DIR_NAME: str = "tensorrt"
    DEFAULT_ENGINE_FILENAME: str = "model.engine"
    DEFAULT_ONNX_FILENAME: str = "model.onnx"
    DEFAULT_OPSET_VERSION: int = 16
    DEFAULT_WORKSPACE_SIZE: int = 1 << 30  # 1 GB


@dataclass(frozen=True)
class TaskDefaults:
    """Default values for task-specific settings."""

    # Default class names for T4Dataset
    DETECTION_3D_CLASSES: tuple = ("car", "truck", "bus", "bicycle", "pedestrian")
    DETECTION_2D_CLASSES: tuple = ("unknown", "car", "truck", "bus", "trailer", "motorcycle", "pedestrian", "bicycle")
    CLASSIFICATION_CLASSES: tuple = ("miscalibrated", "calibrated")

    # Default input sizes
    DEFAULT_2D_INPUT_SIZE: tuple = (960, 960)
    DEFAULT_CLASSIFICATION_INPUT_SIZE: tuple = (224, 224)


# Singleton instances for easy import
EVALUATION_DEFAULTS = EvaluationDefaults()
EXPORT_DEFAULTS = ExportDefaults()
TASK_DEFAULTS = TaskDefaults()
