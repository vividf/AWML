"""Core components for deployment framework."""

from deployment.core.artifacts import Artifact
from deployment.core.backend import Backend
from deployment.core.config.base_config import (
    BackendConfig,
    BaseDeploymentConfig,
    DeviceConfig,
    EvaluationConfig,
    ExportConfig,
    ExportMode,
    RuntimeConfig,
    VerificationConfig,
    VerificationScenario,
    parse_base_args,
    setup_logging,
)
from deployment.core.config.task_config import TaskConfig, TaskType
from deployment.core.contexts import (
    CalibrationExportContext,
    CenterPointExportContext,
    ExportContext,
    YOLOXExportContext,
)
from deployment.core.evaluation.base_evaluator import (
    EVALUATION_DEFAULTS,
    BaseEvaluator,
    EvalResultDict,
    EvaluationDefaults,
    ModelSpec,
    TaskProfile,
    VerifyResultDict,
)
from deployment.core.evaluation.results import (
    ClassificationEvaluationMetrics,
    ClassificationResult,
    Detection2DEvaluationMetrics,
    Detection2DResult,
    Detection3DEvaluationMetrics,
    Detection3DResult,
    EvaluationMetrics,
    LatencyStats,
    StageLatencyBreakdown,
)
from deployment.core.evaluation.verification_mixin import VerificationMixin
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.core.io.preprocessing_builder import build_preprocessing_pipeline
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

__all__ = [
    # Backend and configuration
    "Backend",
    # Typed contexts
    "ExportContext",
    "YOLOXExportContext",
    "CenterPointExportContext",
    "CalibrationExportContext",
    "BaseDeploymentConfig",
    "ExportConfig",
    "ExportMode",
    "RuntimeConfig",
    "BackendConfig",
    "DeviceConfig",
    "EvaluationConfig",
    "VerificationConfig",
    "VerificationScenario",
    "setup_logging",
    "parse_base_args",
    # Task configuration
    "TaskConfig",
    "TaskType",
    # Runtime configuration
    "RuntimeConfig",
    # Constants
    "EVALUATION_DEFAULTS",
    "EvaluationDefaults",
    # Data loading
    "BaseDataLoader",
    # Evaluation
    "BaseEvaluator",
    "TaskProfile",
    "EvalResultDict",
    "VerifyResultDict",
    "VerificationMixin",
    # Artifacts
    "Artifact",
    "ModelSpec",
    # Results (typed)
    "Detection3DResult",
    "Detection2DResult",
    "ClassificationResult",
    "LatencyStats",
    "StageLatencyBreakdown",
    "EvaluationMetrics",
    "Detection3DEvaluationMetrics",
    "Detection2DEvaluationMetrics",
    "ClassificationEvaluationMetrics",
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
