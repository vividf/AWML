"""Evaluation subpackage for deployment core."""

from deployment.core.evaluation.base_evaluator import BaseEvaluator, TaskProfile
from deployment.core.evaluation.evaluator_types import (
    EvalResultDict,
    ModelSpec,
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

__all__ = [
    "BaseEvaluator",
    "TaskProfile",
    "EvalResultDict",
    "ModelSpec",
    "VerifyResultDict",
    "ClassificationResult",
    "Detection2DResult",
    "Detection3DResult",
    "ClassificationEvaluationMetrics",
    "Detection2DEvaluationMetrics",
    "Detection3DEvaluationMetrics",
    "EvaluationMetrics",
    "LatencyStats",
    "StageLatencyBreakdown",
    "VerificationMixin",
]
