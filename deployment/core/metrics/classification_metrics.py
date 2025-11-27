"""
Classification Metrics Adapter using autoware_perception_evaluation.

This module provides an adapter to compute classification metrics (accuracy, precision,
recall, F1) using autoware_perception_evaluation, ensuring consistent metrics between
training evaluation and deployment evaluation.

Usage:
    config = ClassificationMetricsConfig(
        class_names=["miscalibrated", "calibrated"],
    )
    adapter = ClassificationMetricsAdapter(config)

    # Add frames
    for pred_label, gt_label, probs in zip(predictions, ground_truths, probabilities):
        adapter.add_frame(
            prediction=pred_label,  # int (class index)
            ground_truth=gt_label,  # int (class index)
            probabilities=probs,    # List[float] (optional)
        )

    # Compute metrics
    metrics = adapter.compute_metrics()
    # Returns: {"accuracy": 0.95, "precision": 0.94, "recall": 0.96, "f1score": 0.95, ...}
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deployment.core.metrics.base_metrics_adapter import BaseMetricsAdapter, BaseMetricsConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassificationMetricsConfig(BaseMetricsConfig):
    """Configuration for classification metrics.

    Attributes:
        class_names: List of class names for evaluation (e.g., ["miscalibrated", "calibrated"]).
        frame_id: Frame ID for evaluation (not used for classification but kept for consistency).
    """

    # Override default frame_id for classification (not actually used but kept for interface consistency)
    frame_id: str = "classification"


class ClassificationMetricsAdapter(BaseMetricsAdapter):
    """
    Adapter for computing classification metrics.

    This adapter provides a simplified interface for the deployment framework to
    compute accuracy, precision, recall, F1, and per-class metrics for classification
    tasks (e.g., calibration status classification).

    The adapter accumulates predictions and ground truths, then computes metrics
    using formulas consistent with autoware_perception_evaluation's ClassificationMetricsScore.

    Metrics computed:
    - Accuracy: TP / (num_predictions + num_gt - TP)
    - Precision: TP / (TP + FP)
    - Recall: TP / num_gt
    - F1 Score: 2 * precision * recall / (precision + recall)
    - Per-class accuracy, precision, recall, F1

    Example usage:
        config = ClassificationMetricsConfig(
            class_names=["miscalibrated", "calibrated"],
        )
        adapter = ClassificationMetricsAdapter(config)

        # Add frames
        for pred_label, gt_label, probs in zip(predictions, ground_truths, probabilities):
            adapter.add_frame(
                prediction=pred_label,
                ground_truth=gt_label,
                probabilities=probs,
            )

        # Compute metrics
        metrics = adapter.compute_metrics()
    """

    def __init__(self, config: ClassificationMetricsConfig):
        """
        Initialize the classification metrics adapter.

        Args:
            config: Configuration for classification metrics.
        """
        super().__init__(config)
        self.config: ClassificationMetricsConfig = config
        self.num_classes = len(config.class_names)

        # Storage for accumulated results
        self._predictions: List[int] = []
        self._ground_truths: List[int] = []
        self._probabilities: List[List[float]] = []

    def reset(self) -> None:
        """Reset the adapter for a new evaluation session."""
        self._predictions = []
        self._ground_truths = []
        self._probabilities = []
        self._frame_count = 0

    def add_frame(
        self,
        prediction: int,
        ground_truth: int,
        probabilities: Optional[List[float]] = None,
        frame_name: Optional[str] = None,
    ) -> None:
        """Add a single prediction and ground truth for evaluation.

        Args:
            prediction: Predicted class index.
            ground_truth: Ground truth class index.
            probabilities: Optional probability scores for each class.
            frame_name: Optional name for the frame (not used but kept for consistency).
        """
        self._predictions.append(prediction)
        self._ground_truths.append(ground_truth)
        if probabilities is not None:
            self._probabilities.append(probabilities)
        self._frame_count += 1

    def compute_metrics(self) -> Dict[str, float]:
        """Compute metrics from all added predictions.

        Returns:
            Dictionary of metrics including:
                - accuracy: Overall accuracy
                - precision: Overall precision
                - recall: Overall recall
                - f1score: Overall F1 score
                - {class_name}_accuracy: Per-class accuracy
                - {class_name}_precision: Per-class precision
                - {class_name}_recall: Per-class recall
                - {class_name}_f1score: Per-class F1 score
        """
        if self._frame_count == 0:
            logger.warning("No samples to evaluate")
            return {}

        predictions = np.array(self._predictions)
        ground_truths = np.array(self._ground_truths)

        metrics = {}

        # Compute overall metrics
        overall_accuracy, overall_precision, overall_recall, overall_f1 = self._compute_overall_metrics(
            predictions, ground_truths
        )
        metrics["accuracy"] = overall_accuracy
        metrics["precision"] = overall_precision
        metrics["recall"] = overall_recall
        metrics["f1score"] = overall_f1

        # Compute per-class metrics
        for class_idx, class_name in enumerate(self.class_names):
            class_metrics = self._compute_class_metrics(predictions, ground_truths, class_idx)
            metrics[f"{class_name}_accuracy"] = class_metrics["accuracy"]
            metrics[f"{class_name}_precision"] = class_metrics["precision"]
            metrics[f"{class_name}_recall"] = class_metrics["recall"]
            metrics[f"{class_name}_f1score"] = class_metrics["f1score"]
            metrics[f"{class_name}_tp"] = class_metrics["tp"]
            metrics[f"{class_name}_fp"] = class_metrics["fp"]
            metrics[f"{class_name}_fn"] = class_metrics["fn"]
            metrics[f"{class_name}_num_gt"] = class_metrics["num_gt"]

        # Add total counts
        metrics["total_samples"] = len(predictions)
        metrics["correct_predictions"] = int((predictions == ground_truths).sum())

        return metrics

    def _compute_overall_metrics(
        self,
        predictions: np.ndarray,
        ground_truths: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Compute overall metrics following autoware_perception_evaluation formulas.

        The formulas follow ClassificationMetricsScore._summarize() from
        autoware_perception_evaluation.

        Args:
            predictions: Array of predicted class indices.
            ground_truths: Array of ground truth class indices.

        Returns:
            Tuple of (accuracy, precision, recall, f1score).
        """
        num_est = len(predictions)
        num_gt = len(ground_truths)

        # Count TP (correct predictions) and FP (incorrect predictions)
        num_tp = int((predictions == ground_truths).sum())
        num_fp = num_est - num_tp

        # Accuracy formula from autoware_perception_evaluation:
        # accuracy = num_tp / (num_est + num_gt - num_tp)
        # This is equivalent to Jaccard index / IoU
        denominator = num_est + num_gt - num_tp
        accuracy = num_tp / denominator if denominator != 0 else 0.0

        # Precision = TP / (TP + FP)
        precision = num_tp / (num_tp + num_fp) if (num_tp + num_fp) != 0 else 0.0

        # Recall = TP / num_gt
        recall = num_tp / num_gt if num_gt != 0 else 0.0

        # F1 = 2 * precision * recall / (precision + recall)
        f1score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0

        return accuracy, precision, recall, f1score

    def _compute_class_metrics(
        self,
        predictions: np.ndarray,
        ground_truths: np.ndarray,
        class_idx: int,
    ) -> Dict[str, float]:
        """Compute metrics for a single class.

        Args:
            predictions: Array of predicted class indices.
            ground_truths: Array of ground truth class indices.
            class_idx: Class index to compute metrics for.

        Returns:
            Dictionary with accuracy, precision, recall, f1score, tp, fp, fn, num_gt.
        """
        # For binary per-class evaluation:
        # - TP: predicted class_idx and ground truth is class_idx
        # - FP: predicted class_idx but ground truth is not class_idx
        # - FN: not predicted class_idx but ground truth is class_idx

        pred_is_class = predictions == class_idx
        gt_is_class = ground_truths == class_idx

        tp = int((pred_is_class & gt_is_class).sum())
        fp = int((pred_is_class & ~gt_is_class).sum())
        fn = int((~pred_is_class & gt_is_class).sum())
        num_gt = int(gt_is_class.sum())
        num_pred = int(pred_is_class.sum())

        # Precision for this class
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0

        # Recall for this class
        recall = tp / num_gt if num_gt != 0 else 0.0

        # F1 for this class
        f1score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0

        # Accuracy for this class (matching autoware_perception_evaluation formula)
        denominator = num_pred + num_gt - tp
        accuracy = tp / denominator if denominator != 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1score": f1score,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "num_gt": num_gt,
        }

    def get_confusion_matrix(self) -> np.ndarray:
        """Get the confusion matrix.

        Returns:
            2D numpy array where cm[i][j] = count of samples with ground truth i
            predicted as class j.
        """
        if self._frame_count == 0:
            return np.zeros((self.num_classes, self.num_classes), dtype=int)

        predictions = np.array(self._predictions)
        ground_truths = np.array(self._ground_truths)

        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for gt, pred in zip(ground_truths, predictions):
            if 0 <= gt < self.num_classes and 0 <= pred < self.num_classes:
                confusion_matrix[int(gt), int(pred)] += 1

        return confusion_matrix

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the evaluation.

        Returns:
            Dictionary with summary metrics including:
                - accuracy: Overall accuracy
                - per_class_accuracy: Dict mapping class names to accuracies
                - confusion_matrix: 2D list
                - num_samples: Total number of samples
        """
        metrics = self.compute_metrics()

        if not metrics:
            return {
                "accuracy": 0.0,
                "per_class_accuracy": {},
                "confusion_matrix": [],
                "num_samples": 0,
            }

        per_class_accuracy = {}
        for class_name in self.class_names:
            key = f"{class_name}_accuracy"
            if key in metrics:
                per_class_accuracy[class_name] = metrics[key]

        return {
            "accuracy": metrics.get("accuracy", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1score": metrics.get("f1score", 0.0),
            "per_class_accuracy": per_class_accuracy,
            "confusion_matrix": self.get_confusion_matrix().tolist(),
            "num_samples": self._frame_count,
            "detailed_metrics": metrics,
        }
