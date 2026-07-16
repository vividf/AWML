"""
Classification Metrics Interface using autoware_perception_evaluation.

This module provides an interface to compute classification metrics (accuracy, precision,
recall, F1) using autoware_perception_evaluation, ensuring consistent metrics between
training evaluation and deployment evaluation.

Usage:
    config = ClassificationMetricsConfig(
        class_names=["miscalibrated", "calibrated"],
    )
    interface = ClassificationMetricsInterface(config)

    for pred_label, gt_label in zip(predictions, ground_truths):
        interface.add_frame(prediction=pred_label, ground_truth=gt_label)

    metrics = interface.compute_metrics()
    # Returns: {"accuracy": 0.95, "precision": 0.94, "recall": 0.96, "f1score": 0.95, ...}
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mmengine.config import Config
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.object2d import DynamicObject2D
from perception_eval.common.schema import FrameID
from perception_eval.evaluation.metrics import MetricsScore
from perception_eval.manager import PerceptionEvaluationManager

from deployment.metrics.base_metrics_interface import (
    BaseMetricsConfig,
    BaseMetricsInterface,
    validate_2d_frame_id,
)

logger = logging.getLogger(__name__)

# Single-evaluator task: register the evaluator under this name (empty => no key prefix).
_DEFAULT_EVALUATOR = ""


@dataclass(frozen=True)
class ClassificationSummary:
    """Structured summary for classification metrics."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1score: float = 0.0
    per_class_accuracy: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    num_samples: int = 0
    detailed_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1score": self.f1score,
            "per_class_accuracy": dict(self.per_class_accuracy),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "num_samples": self.num_samples,
            "detailed_metrics": dict(self.detailed_metrics),
        }


@dataclass(frozen=True)
class ClassificationMetricsConfig(BaseMetricsConfig):
    """Configuration for classification metrics.

    Attributes:
        class_names: List of class names for evaluation.
        frame_id: Camera frame ID for evaluation (default: "cam_front").
        evaluation_config_dict: Configuration dict for perception evaluation.
        critical_object_filter_config: Config for filtering critical objects.
        frame_pass_fail_config: Config for pass/fail criteria.
    """

    frame_id: str = "cam_front"
    evaluation_config_dict: Optional[Dict[str, Any]] = None
    critical_object_filter_config: Optional[Dict[str, Any]] = None
    frame_pass_fail_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        validate_2d_frame_id(self.frame_id, "classification")

        if self.evaluation_config_dict is None:
            object.__setattr__(
                self,
                "evaluation_config_dict",
                {
                    "evaluation_task": "classification2d",
                    "target_labels": self.class_names,
                    "center_distance_thresholds": None,
                    "center_distance_bev_thresholds": None,
                    "plane_distance_thresholds": None,
                    "iou_2d_thresholds": None,
                    "iou_3d_thresholds": None,
                    "label_prefix": "autoware",
                },
            )

        if self.critical_object_filter_config is None:
            object.__setattr__(
                self,
                "critical_object_filter_config",
                {
                    "target_labels": self.class_names,
                    "ignore_attributes": None,
                },
            )

        if self.frame_pass_fail_config is None:
            object.__setattr__(
                self,
                "frame_pass_fail_config",
                {
                    "target_labels": self.class_names,
                    # classification2d is thresholdless: perception_eval keys its object results
                    # under the sentinel -1.0 and requires matching_threshold_list to be None
                    # (see PassFailNusceneResult.__init__ / _build_label_thresholds). A concrete
                    # list here (e.g. [1.0, ...]) makes the label→threshold lookup miss the bucket.
                    "matching_threshold_list": None,
                    "confidence_threshold_list": None,
                },
            )


def extract_classification_metrics_config(
    model_cfg: Config,
    class_names: Optional[List[str]] = None,
    frame_id: str = "cam_front",
) -> ClassificationMetricsConfig:
    """Build a `ClassificationMetricsConfig` for a classification deployment project.

    The classification sibling of
    :func:`~deployment.metrics.detection_2d_metrics.extract_detection2d_metrics_config`. Class
    names are the *only* task input, and classifier model configs do not always carry them
    (mmpretrain heads record ``num_classes``, not label strings), so ``class_names`` may be passed
    explicitly by the project entrypoint (e.g. from a deploy-config key). When omitted, they are
    read from ``model_cfg.classes``.

    Args:
        model_cfg: MMEngine model configuration; consulted for ``classes`` only when
            ``class_names`` is not given.
        class_names: Explicit class names in label-index order; overrides ``model_cfg.classes``.
        frame_id: perception_eval camera frame id (must be a valid 2D frame).

    Returns:
        ClassificationMetricsConfig with the resolved class names.

    Raises:
        ValueError: If ``class_names`` is not given and ``model_cfg`` has no list/tuple ``classes``.
    """
    if class_names is None:
        classes = getattr(model_cfg, "classes", None)
        if classes is None and hasattr(model_cfg, "get"):
            classes = model_cfg.get("classes")
        if not isinstance(classes, (tuple, list)):
            raise ValueError(
                "class_names not provided and model_cfg has no list/tuple 'classes'. "
                "Pass class_names explicitly (e.g. from the deploy config's class_names key)."
            )
        class_names = list(classes)

    return ClassificationMetricsConfig(class_names=list(class_names), frame_id=frame_id)


class ClassificationMetricsInterface(BaseMetricsInterface):
    """Interface for computing classification metrics using autoware_perception_evaluation.

    Metrics computed:
    - Accuracy: TP / (num_predictions + num_gt - TP)
    - Precision: TP / (TP + FP)
    - Recall: TP / num_gt
    - F1 Score: 2 * precision * recall / (precision + recall)
    - Per-class accuracy, precision, recall, F1
    """

    def __init__(
        self,
        config: ClassificationMetricsConfig,
        data_root: str = "data/t4dataset/",
        result_root_directory: str = "/tmp/perception_eval_classification/",
    ) -> None:
        """Initialize the classification metrics interface.

        Args:
            config: Configuration for classification metrics.
            data_root: Root directory of the dataset.
            result_root_directory: Directory for saving evaluation results.
        """
        super().__init__(config)
        self.config: ClassificationMetricsConfig = config

        self._evaluator_specs[_DEFAULT_EVALUATOR] = self._build_evaluator_spec(
            config.evaluation_config_dict,
            data_root=data_root,
            result_root_directory=result_root_directory,
            critical_object_filter_config=config.critical_object_filter_config,
            frame_pass_fail_config=config.frame_pass_fail_config,
        )

        # Matched per-frame results from the last compute, captured for the confusion matrix.
        self._frame_results: Optional[List[Any]] = None

    def reset(self) -> None:
        """Reset the interface for a new evaluation session."""
        super().reset()
        self._frame_results = None

    def _capture_evaluator(self, name: str, evaluator: PerceptionEvaluationManager) -> None:
        """Capture matched per-frame results for the confusion matrix (single evaluator)."""
        self._frame_results = list(evaluator.frame_results)

    def _create_dynamic_object_2d(
        self,
        label_index: int,
        unix_time: int,
        score: float = 1.0,
        uuid: Optional[str] = None,
    ) -> DynamicObject2D:
        """Create a DynamicObject2D for classification (roi=None for image-level)."""
        return DynamicObject2D(
            unix_time=unix_time,
            frame_id=FrameID.from_value(self.frame_id),
            semantic_score=score,
            semantic_label=self._convert_index_to_label(label_index),
            roi=None,
            uuid=uuid,
        )

    def add_frame(
        self,
        prediction: int,
        ground_truth: int,
        probabilities: Optional[List[float]] = None,
        frame_name: Optional[str] = None,
    ) -> None:
        """Buffer a single prediction and ground truth for evaluation.

        Args:
            prediction: Predicted class index.
            ground_truth: Ground truth class index.
            probabilities: Optional probability scores for each class.
            frame_name: Optional name for the frame.
        """
        unix_time = int(time.time() * 1e6)
        if frame_name is None:
            frame_name = str(self._frame_count)

        # Get confidence score from probabilities if available
        score = 1.0
        if probabilities is not None and 0 <= prediction < len(probabilities):
            score = float(probabilities[prediction])

        estimated_object = self._create_dynamic_object_2d(
            label_index=prediction, unix_time=unix_time, score=score, uuid=frame_name
        )
        gt_object = self._create_dynamic_object_2d(
            label_index=ground_truth, unix_time=unix_time, score=1.0, uuid=frame_name
        )

        frame_ground_truth = FrameGroundTruth(
            unix_time=unix_time,
            frame_name=frame_name,
            objects=[gt_object],
            transforms=None,
            raw_data=None,
        )

        self._buffer_frame(unix_time, [estimated_object], frame_ground_truth)

    @staticmethod
    def _summarize_classification_score(classification_score: Any) -> Tuple[float, float, float, float]:
        """Read overall (accuracy, precision, recall, f1) from a perception_eval score.

        ``ClassificationMetricsScore._summarize`` returns a nested
        ``{MatchingMode: {threshold: ClassificationScores}}`` mapping. Classification is
        single-mode (CLASSIFICATION_2D) and thresholdless (one bucket, key -1.0), so there is
        exactly one ``ClassificationScores`` to read; an empty mapping (no matched frames) → zeros.
        """
        summarize = getattr(classification_score, "_summarize", None)
        if not callable(summarize):
            raise AttributeError(
                "perception_eval classification score no longer exposes '_summarize'; "
                "update ClassificationMetricsInterface to the current perception_eval API."
            )
        for threshold_scores in summarize().values():
            for scores in threshold_scores.values():
                return scores.accuracy, scores.precision, scores.recall, scores.f1score
        return 0.0, 0.0, 0.0, 0.0

    @staticmethod
    def _finite_or_zero(value: float) -> float:
        """Coerce inf/nan (e.g. from empty divisions or 0/0) to 0.0."""
        return float(value) if np.isfinite(value) else 0.0

    def _process_metrics_score(self, metrics_score: MetricsScore, prefix: Optional[str] = None) -> Dict[str, float]:
        """Process MetricsScore into a flat dictionary."""
        metric_dict: Dict[str, float] = {}

        for classification_score in metrics_score.classification_scores:
            # Overall metrics
            accuracy, precision, recall, f1score = self._summarize_classification_score(classification_score)
            metric_dict["accuracy"] = self._finite_or_zero(accuracy)
            metric_dict["precision"] = self._finite_or_zero(precision)
            metric_dict["recall"] = self._finite_or_zero(recall)
            metric_dict["f1score"] = self._finite_or_zero(f1score)

            # Per-class metrics. `accuracies` is now nested
            # {MatchingMode: {LabelType: {threshold: ClassificationAccuracy}}}; classification is
            # single-mode and single-threshold, so flatten to the leaf ClassificationAccuracy.
            for label_accuracies in classification_score.accuracies.values():
                for threshold_accuracies in label_accuracies.values():
                    for acc in threshold_accuracies.values():
                        if not acc.target_labels:
                            continue

                        target_label = acc.target_labels[0]
                        class_name = getattr(target_label, "name", str(target_label))

                        metric_dict[f"{class_name}_accuracy"] = self._finite_or_zero(acc.accuracy)
                        metric_dict[f"{class_name}_precision"] = self._finite_or_zero(acc.precision)
                        metric_dict[f"{class_name}_recall"] = self._finite_or_zero(acc.recall)
                        metric_dict[f"{class_name}_f1score"] = self._finite_or_zero(acc.f1score)
                        metric_dict[f"{class_name}_tp"] = acc.num_tp
                        metric_dict[f"{class_name}_fp"] = acc.num_fp
                        metric_dict[f"{class_name}_num_gt"] = acc.num_ground_truth
                        metric_dict[f"{class_name}_num_pred"] = acc.objects_results_num

        metric_dict["total_samples"] = self._frame_count
        return metric_dict

    # TODO(vividf): Remove after autoware_perception_evaluation supports confusion matrix.
    @property
    def confusion_matrix(self) -> np.ndarray:
        """Get the confusion matrix.

        Returns:
            2D numpy array where cm[i][j] = count of ground truth i predicted as j.
        """
        num_classes = len(self.class_names)
        if self._frame_count == 0:
            return np.zeros((num_classes, num_classes), dtype=int)

        # Matched results are produced during compute_metrics; ensure it has run.
        if self._frame_results is None:
            self.compute_metrics()

        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        # perception_eval now stores matched results as a nested
        # {MatchingMode: {LabelType: {threshold: [DynamicObjectWithPerceptionResult]}}} mapping;
        # each object result appears exactly once across the label buckets, so walking every leaf
        # list counts each prediction/GT pair once.
        for frame_result in self._frame_results or []:
            mode_results = getattr(frame_result, "nuscene_object_results", None) or {}
            for label_results in mode_results.values():
                for threshold_results in label_results.values():
                    for object_results in threshold_results.values():
                        for obj_result in object_results:
                            if obj_result.ground_truth_object is None:
                                continue

                            pred_name = obj_result.estimated_object.semantic_label.name
                            gt_name = obj_result.ground_truth_object.semantic_label.name

                            pred_idx = next(
                                (i for i, n in enumerate(self.class_names) if n.lower() == pred_name.lower()),
                                -1,
                            )
                            gt_idx = next(
                                (i for i, n in enumerate(self.class_names) if n.lower() == gt_name.lower()),
                                -1,
                            )

                            if 0 <= pred_idx < num_classes and 0 <= gt_idx < num_classes:
                                confusion_matrix[gt_idx, pred_idx] += 1

        return confusion_matrix

    @property
    def summary(self) -> ClassificationSummary:
        """Get a summary of the evaluation.

        Returns:
            ClassificationSummary with aggregate metrics.
        """
        metrics = self.compute_metrics()

        if not metrics:
            return ClassificationSummary()

        per_class_accuracy = {
            name: metrics[f"{name}_accuracy"] for name in self.class_names if f"{name}_accuracy" in metrics
        }

        return ClassificationSummary(
            accuracy=metrics.get("accuracy", 0.0),
            precision=metrics.get("precision", 0.0),
            recall=metrics.get("recall", 0.0),
            f1score=metrics.get("f1score", 0.0),
            per_class_accuracy=per_class_accuracy,
            confusion_matrix=self.confusion_matrix.tolist(),
            num_samples=self._frame_count,
            detailed_metrics=metrics,
        )
