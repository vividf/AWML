"""Core components for deployment framework."""

from deployment.core.artifacts import Artifact
from deployment.core.backend import Backend
from deployment.core.config.base_config import (
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
from deployment.core.config.constants import (
    EVALUATION_DEFAULTS,
    EXPORT_DEFAULTS,
    TASK_DEFAULTS,
    EvaluationDefaults,
    ExportDefaults,
    TaskDefaults,
)
from deployment.core.config.runtime_config import (
    BaseRuntimeConfig,
    ClassificationRuntimeConfig,
    Detection2DRuntimeConfig,
    Detection3DRuntimeConfig,
)
from deployment.core.config.task_config import TaskConfig, TaskType
from deployment.core.contexts import (
    CalibrationExportContext,
    CenterPointExportContext,
    ExportContext,
    ExportContextType,
    PreprocessContext,
    YOLOXExportContext,
    create_export_context,
)
from deployment.core.evaluation.base_evaluator import (
    BaseEvaluator,
    EvalResultDict,
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
    "ExportContextType",
    "YOLOXExportContext",
    "CenterPointExportContext",
    "CalibrationExportContext",
    "PreprocessContext",
    "create_export_context",
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
    # Task configuration
    "TaskConfig",
    "TaskType",
    # Runtime configurations (typed)
    "BaseRuntimeConfig",
    "Detection3DRuntimeConfig",
    "Detection2DRuntimeConfig",
    "ClassificationRuntimeConfig",
    # Constants
    "EVALUATION_DEFAULTS",
    "EXPORT_DEFAULTS",
    "TASK_DEFAULTS",
    "EvaluationDefaults",
    "ExportDefaults",
    "TaskDefaults",
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
