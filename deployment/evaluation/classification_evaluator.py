"""Shared evaluator for single-label image classifiers.

Scoring depends only on the classifier's output (a predicted class index plus per-class
probabilities) and the sample's integer ground-truth label, not on the backbone or the input
modality, so any classification project reuses this. It feeds the shared
:class:`~deployment.metrics.classification_metrics.ClassificationMetricsInterface`
(accuracy / precision / recall / F1 + confusion matrix via ``autoware_perception_evaluation``).

The evaluator is a pure metrics adapter: the standard ``BaseEvaluator`` loop drives it, one sample
at a time. Tasks whose ground truth is *synthetic* (e.g. calibration status, where each base
sample is evaluated once "calibrated" and once "miscalibrated") express that in their **data
loader's** sample indexing, not by overriding ``evaluate`` here.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping

import numpy as np
from mmengine.config import Config
from typing_extensions import override

from deployment.evaluation.base_evaluator import BaseEvaluator, EvalResultDict
from deployment.execution.backend_executor import BackendExecutor
from deployment.metrics.classification_metrics import ClassificationMetricsConfig, ClassificationMetricsInterface

logger = logging.getLogger(__name__)


class ClassificationEvaluator(BaseEvaluator):
    """Evaluator for single-label classification backed by ``ClassificationMetricsInterface``.

    Args:
        model_cfg: Model configuration (kept for parity with other evaluators).
        metrics_config: Configuration for classification metrics (carries the class names).
        executor: Backend execution primitives, shared with the verification runner.
    """

    def __init__(
        self,
        model_cfg: Config,
        metrics_config: ClassificationMetricsConfig,
        executor: BackendExecutor,
    ) -> None:
        # Class names live on the metrics config; keep a copy for per-class result labelling.
        self.class_names: List[str] = list(metrics_config.class_names)
        super().__init__(
            metrics_interface=ClassificationMetricsInterface(metrics_config),
            model_cfg=model_cfg,
            executor=executor,
        )

    @override
    def _parse_predictions(self, pipeline_output: Any) -> Dict[str, Any]:
        """Return the pipeline's classification result dict (``class_id`` + ``probabilities``) as-is."""
        return pipeline_output

    @override
    def _parse_ground_truths(self, sample: Mapping[str, Any]) -> int:
        """Extract the integer class label from the sample's synthetic/stored ground truth.

        Raises:
            KeyError: If the sample has no ``ground_truth`` entry.
        """
        gt_data = sample["ground_truth"]
        return int(gt_data.get("gt_label", 0))

    @override
    def _add_to_interface(self, predictions: Dict[str, Any], ground_truths: int) -> None:
        """Feed one prediction/ground-truth pair (with probabilities) to the metrics interface."""
        prob = predictions.get("probabilities", [])
        prob_list = prob.tolist() if isinstance(prob, np.ndarray) else list(prob)
        self.metrics_interface.add_frame(
            prediction=int(predictions["class_id"]),
            ground_truth=ground_truths,
            probabilities=prob_list,
        )

    @override
    def _build_results(
        self,
        latencies: List[float],
        latency_breakdowns: List[Dict[str, float]],
        num_samples: int,
    ) -> EvalResultDict:
        """Aggregate accuracy/precision/recall/F1, confusion matrix, and latency into an ``EvalResultDict``."""
        interface_metrics = self.metrics_interface.compute_metrics()
        summary_dict = self.metrics_interface.summary.to_dict()
        confusion_matrix = self.metrics_interface.confusion_matrix
        latency_stats = self.compute_latency_stats(latencies)

        # correct_predictions = diagonal of the confusion matrix (perception_eval reports no such key).
        correct = int(np.trace(confusion_matrix)) if hasattr(confusion_matrix, "shape") else 0

        result: EvalResultDict = {
            "accuracy": interface_metrics.get("accuracy", 0.0),
            "precision": interface_metrics.get("precision", 0.0),
            "recall": interface_metrics.get("recall", 0.0),
            "f1score": interface_metrics.get("f1score", 0.0),
            "correct_predictions": correct,
            "per_class_accuracy": summary_dict.get("per_class_accuracy", {}),
            "confusion_matrix": confusion_matrix.tolist() if hasattr(confusion_matrix, "tolist") else confusion_matrix,
            "detailed_metrics": interface_metrics,
            "latency": latency_stats,
            "num_samples": num_samples,
        }

        if latency_breakdowns:
            result["latency_breakdown"] = self._compute_latency_breakdown(latency_breakdowns)

        return result

    @override
    def summarize_for_comparison(self, results: EvalResultDict) -> List[str]:
        """Summarize accuracy for the cross-backend comparison."""
        lines: List[str] = [f"  Accuracy: {results.get('accuracy', 0.0):.4f}"]
        lines.extend(super().summarize_for_comparison(results))
        return lines

    @override
    def print_results(self, results: EvalResultDict) -> None:
        """Log classification metrics, per-class accuracy, confusion matrix, and latency."""
        logger.info("")
        logger.info("Classification Metrics:")
        logger.info("  Correct predictions: %s", results.get("correct_predictions", 0))
        logger.info("  Accuracy:  %.4f", results.get("accuracy", 0.0))
        logger.info("  Precision: %.4f", results.get("precision", 0.0))
        logger.info("  Recall:    %.4f", results.get("recall", 0.0))
        logger.info("  F1 Score:  %.4f", results.get("f1score", 0.0))

        per_class = results.get("per_class_accuracy") or {}
        if per_class:
            logger.info("")
            logger.info("Per-class Accuracy:")
            for class_name, acc in per_class.items():
                logger.info("  %-15s: %.4f", class_name, acc)

        confusion_matrix = results.get("confusion_matrix")
        if confusion_matrix:
            logger.info("")
            logger.info("Confusion Matrix (rows=GT, cols=Pred):")
            header = "           " + " ".join(f"{name[:8]:>8}" for name in self.class_names)
            logger.info(header)
            for i, row in enumerate(confusion_matrix):
                gt_name = self.class_names[i] if i < len(self.class_names) else str(i)
                logger.info("  %-8s %s", gt_name[:8], " ".join(f"{int(v):>8}" for v in row))

        if "latency" in results:
            latency_dict = results["latency"].to_dict()
            logger.info("")
            logger.info("Latency Statistics:")
            logger.info("  Mean:   %.2f ms", latency_dict["mean_ms"])
            logger.info("  Std:    %.2f ms", latency_dict["std_ms"])
            logger.info("  Min:    %.2f ms", latency_dict["min_ms"])
            logger.info("  Max:    %.2f ms", latency_dict["max_ms"])
            logger.info("  Median: %.2f ms", latency_dict["median_ms"])

        logger.info("")
        logger.info("Total Samples: %s", results["num_samples"])
