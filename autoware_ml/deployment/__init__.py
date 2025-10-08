"""
Autoware ML Unified Deployment Framework

This package provides a unified, task-agnostic deployment framework for
exporting, verifying, and evaluating machine learning models across different
tasks (classification, detection, segmentation, etc.) and backends (ONNX,
TensorRT, TorchScript, etc.).
"""

from .core.base_config import BaseDeploymentConfig
from .core.base_data_loader import BaseDataLoader
from .core.base_evaluator import BaseEvaluator
from .utils import build_test_pipeline

__all__ = [
    "BaseDeploymentConfig",
    "BaseDataLoader",
    "BaseEvaluator",
    "build_test_pipeline",
]

__version__ = "1.0.0"
