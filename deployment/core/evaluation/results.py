"""
Typed result classes for deployment framework.

This module provides strongly-typed result classes instead of Dict[str, Any],
enabling better IDE support and catching errors at development time.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Detection3DResult:
    """Result for a single 3D detection (immutable)."""

    bbox_3d: Tuple[float, ...]  # [x, y, z, l, w, h, yaw] or with velocity [x, y, z, l, w, h, yaw, vx, vy]
    score: float
    label: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "bbox_3d": list(self.bbox_3d),
            "score": self.score,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detection3DResult":
        """Create from dictionary."""
        return cls(
            bbox_3d=tuple(data["bbox_3d"]),
            score=data["score"],
            label=data["label"],
        )


@dataclass(frozen=True)
class Detection2DResult:
    """Result for a single 2D detection (immutable)."""

    bbox: Tuple[float, ...]  # [x1, y1, x2, y2]
    score: float
    class_id: int
    class_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "bbox": list(self.bbox),
            "score": self.score,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detection2DResult":
        """Create from dictionary."""
        return cls(
            bbox=tuple(data["bbox"]),
            score=data["score"],
            class_id=data.get("class_id", data.get("label", 0)),
            class_name=data.get("class_name", ""),
        )


@dataclass
class ClassificationResult:
    """Result for a classification prediction."""

    class_id: int
    class_name: str
    confidence: float
    probabilities: np.ndarray
    top_k: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "top_k": self.top_k,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassificationResult":
        """Create from dictionary."""
        return cls(
            class_id=data["class_id"],
            class_name=data["class_name"],
            confidence=data["confidence"],
            probabilities=data["probabilities"],
            top_k=data.get("top_k", []),
        )


@dataclass(frozen=True)
class LatencyStats:
    """Latency statistics (immutable)."""

    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "median_ms": self.median_ms,
        }

    @classmethod
    def from_latencies(cls, latencies: List[float]) -> "LatencyStats":
        """Compute statistics from a list of latency values."""
        if not latencies:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0)

        arr = np.array(latencies)
        return cls(
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            median_ms=float(np.median(arr)),
        )


@dataclass(frozen=True)
class StageLatencyBreakdown:
    """Latency breakdown by inference stage (immutable)."""

    preprocessing_ms: Optional[LatencyStats] = None
    voxel_encoder_ms: Optional[LatencyStats] = None
    middle_encoder_ms: Optional[LatencyStats] = None
    backbone_head_ms: Optional[LatencyStats] = None
    postprocessing_ms: Optional[LatencyStats] = None
    model_ms: Optional[LatencyStats] = None

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Convert to dictionary format."""
        result = {}
        if self.preprocessing_ms:
            result["preprocessing_ms"] = self.preprocessing_ms.to_dict()
        if self.voxel_encoder_ms:
            result["voxel_encoder_ms"] = self.voxel_encoder_ms.to_dict()
        if self.middle_encoder_ms:
            result["middle_encoder_ms"] = self.middle_encoder_ms.to_dict()
        if self.backbone_head_ms:
            result["backbone_head_ms"] = self.backbone_head_ms.to_dict()
        if self.postprocessing_ms:
            result["postprocessing_ms"] = self.postprocessing_ms.to_dict()
        if self.model_ms:
            result["model_ms"] = self.model_ms.to_dict()
        return result


@dataclass
class EvaluationMetrics:
    """Base class for evaluation metrics."""

    num_samples: int
    latency: LatencyStats
    latency_breakdown: Optional[StageLatencyBreakdown] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "num_samples": self.num_samples,
            "latency": self.latency.to_dict(),
        }
        if self.latency_breakdown:
            result["latency_breakdown"] = self.latency_breakdown.to_dict()
        return result


@dataclass
class Detection3DEvaluationMetrics(EvaluationMetrics):
    """Evaluation metrics for 3D detection."""

    mAP: float = 0.0
    mAPH: float = 0.0
    per_class_ap: Dict[str, float] = field(default_factory=dict)
    total_predictions: int = 0
    total_ground_truths: int = 0
    per_class_predictions: Dict[int, int] = field(default_factory=dict)
    per_class_ground_truths: Dict[int, int] = field(default_factory=dict)
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = super().to_dict()
        result.update(
            {
                "mAP": self.mAP,
                "mAPH": self.mAPH,
                "per_class_ap": self.per_class_ap,
                "total_predictions": self.total_predictions,
                "total_ground_truths": self.total_ground_truths,
                "per_class_predictions": self.per_class_predictions,
                "per_class_ground_truths": self.per_class_ground_truths,
                "detailed_metrics": self.detailed_metrics,
            }
        )
        return result


@dataclass
class Detection2DEvaluationMetrics(EvaluationMetrics):
    """Evaluation metrics for 2D detection."""

    mAP: float = 0.0
    mAP_50: float = 0.0
    mAP_75: float = 0.0
    per_class_ap: Dict[str, float] = field(default_factory=dict)
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = super().to_dict()
        result.update(
            {
                "mAP": self.mAP,
                "mAP_50": self.mAP_50,
                "mAP_75": self.mAP_75,
                "per_class_ap": self.per_class_ap,
                "detailed_metrics": self.detailed_metrics,
            }
        )
        return result


@dataclass
class ClassificationEvaluationMetrics(EvaluationMetrics):
    """Evaluation metrics for classification."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1score: float = 0.0
    correct_predictions: int = 0
    total_samples: int = 0
    per_class_accuracy: Dict[str, float] = field(default_factory=dict)
    per_class_count: Dict[int, int] = field(default_factory=dict)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = super().to_dict()
        result.update(
            {
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1score": self.f1score,
                "correct_predictions": self.correct_predictions,
                "total_samples": self.total_samples,
                "per_class_accuracy": self.per_class_accuracy,
                "per_class_count": self.per_class_count,
                "confusion_matrix": self.confusion_matrix,
                "detailed_metrics": self.detailed_metrics,
            }
        )
        return result
