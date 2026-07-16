"""Shared evaluator for 2D detectors.

Scoring depends only on the 2D-detection outputs (image-space ``bbox`` + ``label`` + ``score``),
not on the detector architecture, so any 2D detector (YOLOX, …) reuses this. It is the 2D sibling
of :class:`~deployment.evaluation.detection_3d_evaluator.Detection3DEvaluator`: both feed a shared
:class:`~deployment.metrics.detection_base.DetectionMetricsInterface` (mAP via
``autoware_perception_evaluation``) and build the same ``DetectionSummary``. 2D differs only in:

- ground-truth parsing — image-space ``[x1, y1, x2, y2]`` boxes, no LiDAR-point filtering,
- no heading — the shared metrics interface reports ``mAPH_by_mode`` empty (``_supports_aph=False``).

Projects subclass this only to customise ``print_results``; the default prints a generic
mAP + per-class-AP + latency report.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping

from mmengine.config import Config
from typing_extensions import override

from deployment.evaluation.base_evaluator import BaseEvaluator, EvalResultDict
from deployment.execution.backend_executor import BackendExecutor
from deployment.metrics.detection_2d_metrics import Detection2DMetricsConfig, Detection2DMetricsInterface

logger = logging.getLogger(__name__)

# Top-level stage keys logged without extra indentation in the latency breakdown.
_TOP_LEVEL_STAGES = {"preprocessing_ms", "model_ms", "postprocessing_ms"}


class Detection2DEvaluator(BaseEvaluator):
    """Evaluator for 2D detection backed by ``Detection2DMetricsInterface`` (architecture-agnostic).

    Args:
        model_cfg: Model configuration (kept for parity with other evaluators; class names come
            from ``metrics_config``).
        metrics_config: Configuration for 2D detection metrics (IoU-2D mAP).
        executor: Backend execution primitives, shared with the verification runner.
    """

    def __init__(
        self,
        model_cfg: Config,
        metrics_config: Detection2DMetricsConfig,
        executor: BackendExecutor,
    ) -> None:
        super().__init__(
            metrics_interface=Detection2DMetricsInterface(metrics_config),
            model_cfg=model_cfg,
            executor=executor,
        )

    @override
    def _parse_predictions(self, pipeline_output: Any) -> List[Dict]:
        """Normalize pipeline detections into ``{bbox, label, score}`` dicts.

        The pipeline emits ``{bbox: [x1, y1, x2, y2], class_id/label, score}`` per detection;
        anything that is not such a dict is dropped.
        """
        predictions: List[Dict] = []
        if not isinstance(pipeline_output, list):
            return predictions
        for det in pipeline_output:
            if isinstance(det, dict) and "bbox" in det:
                predictions.append(
                    {
                        "bbox": list(det["bbox"]),
                        "label": int(det.get("class_id", det.get("label", 0))),
                        "score": float(det.get("score", 0.0)),
                    }
                )
        return predictions

    @override
    def _parse_ground_truths(self, sample: Mapping[str, Any]) -> List[Dict]:
        """Convert ``gt_bboxes`` / ``gt_labels`` into a list of ``{bbox, label}`` dicts.

        Raises:
            KeyError: If the sample has no ``ground_truth`` entry.
        """
        gt_data = sample["ground_truth"]
        gt_bboxes = gt_data.get("gt_bboxes", [])
        gt_labels = gt_data.get("gt_labels", [])

        ground_truths: List[Dict] = []
        for bbox, label in zip(gt_bboxes, gt_labels):
            ground_truths.append({"bbox": list(bbox), "label": int(label)})
        return ground_truths

    @override
    def _add_to_interface(self, predictions: List[Dict], ground_truths: List[Dict]) -> None:
        """Add one frame of predictions and ground truths to the metrics interface."""
        self.metrics_interface.add_frame(predictions, ground_truths)

    @override
    def _build_results(
        self,
        latencies: List[float],
        latency_breakdowns: List[Dict[str, float]],
        num_samples: int,
    ) -> EvalResultDict:
        """Aggregate mAP, per-class AP, and latency into an ``EvalResultDict``."""
        latency_stats = self.compute_latency_stats(latencies)

        map_results = self.metrics_interface.compute_metrics()
        summary_dict = self.metrics_interface.summary.to_dict()

        result: EvalResultDict = {
            "mAP_by_mode": summary_dict.get("mAP_by_mode", {}),
            "per_class_ap_by_mode": summary_dict.get("per_class_ap_by_mode", {}),
            "detailed_metrics": map_results,
            "latency": latency_stats,
            "num_samples": num_samples,
        }

        if latency_breakdowns:
            result["latency_breakdown"] = self._compute_latency_breakdown(latency_breakdowns)

        return result

    @override
    def summarize_for_comparison(self, results: EvalResultDict) -> List[str]:
        """Summarize mAP per matching mode for the cross-backend comparison."""
        lines: List[str] = []
        for mode, map_value in (results.get("mAP_by_mode") or {}).items():
            lines.append(f"  mAP ({mode}): {map_value:.4f}")
        lines.extend(super().summarize_for_comparison(results))
        return lines

    @override
    def print_results(self, results: EvalResultDict) -> None:
        """Log mAP per mode, per-class AP, latency stats, and an optional stage breakdown."""
        logger.info("")
        logger.info("2D Detection Metrics:")
        for mode, map_value in (results.get("mAP_by_mode") or {}).items():
            logger.info("  mAP (%s): %.4f", mode, map_value)

        for mode, per_class in (results.get("per_class_ap_by_mode") or {}).items():
            logger.info("  Per-class AP (%s):", mode)
            for class_name, ap_value in per_class.items():
                logger.info("    %-25s: %.4f", class_name, ap_value)

        if "latency" in results:
            latency_dict = results["latency"].to_dict()
            logger.info("")
            logger.info("Latency Statistics:")
            logger.info("  Mean:   %.2f ms", latency_dict["mean_ms"])
            logger.info("  Std:    %.2f ms", latency_dict["std_ms"])
            logger.info("  Min:    %.2f ms", latency_dict["min_ms"])
            logger.info("  Max:    %.2f ms", latency_dict["max_ms"])
            logger.info("  Median: %.2f ms", latency_dict["median_ms"])

        if "latency_breakdown" in results:
            breakdown_dict = results["latency_breakdown"].to_dict()
            if breakdown_dict:
                logger.info("")
                logger.info("Stage-wise Latency Breakdown:")
                for stage, stats_dict in breakdown_dict.items():
                    stage_name = stage.replace("_ms", "").replace("_", " ").title()
                    output_format = (
                        "  %-18s: %.2f ± %.2f ms" if stage in _TOP_LEVEL_STAGES else "    %-16s: %.2f ± %.2f ms"
                    )
                    logger.info(output_format, stage_name, stats_dict["mean_ms"], stats_dict["std_ms"])

        logger.info("")
        logger.info("Total Samples: %s", results["num_samples"])
