"""Core components for deployment framework."""

from deployment.core.artifacts import Artifact
from deployment.core.backend import Backend
from deployment.core.base_config import (
    BackendConfig,
    BaseDeploymentConfig,
    EvaluationConfig,
    ExportConfig,
    ExportMode,
    RuntimeConfig,
    VerificationConfig,
    VerificationScenario,
    parse_base_args,
    setup_logging,
)
from deployment.core.base_data_loader import BaseDataLoader
from deployment.core.base_evaluator import (
    BaseEvaluator,
    EvalResultDict,
    ModelSpec,
    VerifyResultDict,
)
from deployment.core.metrics import (
    BaseMetricsAdapter,
    BaseMetricsConfig,
    ClassificationMetricsAdapter,
    ClassificationMetricsConfig,
    Detection2DMetricsAdapter,
    Detection2DMetricsConfig,
    Detection3DMetricsAdapter,
    Detection3DMetricsConfig,
)
from deployment.core.preprocessing_builder import build_preprocessing_pipeline

__all__ = [
    # Backend and configuration
    "Backend",
    "BaseDeploymentConfig",
    "ExportConfig",
    "ExportMode",
    "RuntimeConfig",
    "BackendConfig",
    "EvaluationConfig",
    "VerificationConfig",
    "VerificationScenario",
    "setup_logging",
    "parse_base_args",
    # Data loading
    "BaseDataLoader",
    # Evaluation
    "BaseEvaluator",
    "EvalResultDict",
    "VerifyResultDict",
    # Artifacts
    "Artifact",
    "ModelSpec",
    # Preprocessing
    "build_preprocessing_pipeline",
    # Metrics adapters (using autoware_perception_evaluation)
    "BaseMetricsAdapter",
    "BaseMetricsConfig",
    "Detection3DMetricsAdapter",
    "Detection3DMetricsConfig",
    "Detection2DMetricsAdapter",
    "Detection2DMetricsConfig",
    "ClassificationMetricsAdapter",
    "ClassificationMetricsConfig",
]
