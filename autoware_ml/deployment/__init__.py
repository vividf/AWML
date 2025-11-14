"""
Autoware ML Unified Deployment Framework

This package provides a unified, task-agnostic deployment framework for
exporting, verifying, and evaluating machine learning models across different
tasks (classification, detection, segmentation, etc.) and backends (ONNX,
TensorRT, TorchScript, etc.).
"""

from autoware_ml.deployment.core.base_config import BaseDeploymentConfig
from autoware_ml.deployment.core.base_data_loader import BaseDataLoader
from autoware_ml.deployment.core.base_evaluator import BaseEvaluator
from autoware_ml.deployment.core.preprocessing_builder import build_preprocessing_pipeline
from autoware_ml.deployment.runners import DeploymentRunner

__all__ = [
    "BaseDeploymentConfig",
    "BaseDataLoader",
    "BaseEvaluator",
    "DeploymentRunner",
    "build_preprocessing_pipeline",
]

__version__ = "1.0.0"
