"""Shared evaluator for point-cloud 3D detectors (CenterPoint, BEVFusion).

Factors out the metrics hooks (prediction/GT parsing, metric accumulation, result building,
comparison summary) that the two projects' evaluators previously duplicated (~6 methods, near
verbatim). Subclasses override only ``print_results`` (backend-specific latency-breakdown
layout) and may reuse ``_log_latency_stats`` for the shared latency block.

Note: this unifies the two former ``_build_results`` copies to the **stricter** CenterPoint
behavior — it validates the required summary keys and populates ``detailed_metrics`` with the
computed metrics. The old BEVFusion copy was lax (``.get(..., {})`` and empty detailed metrics);
that drift is intentionally removed here.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Mapping

import numpy as np
from mmengine.config import Config
from typing_extensions import override

from deployment.evaluation.backend_executor import BackendExecutor
from deployment.evaluation.base_evaluator import BaseEvaluator, EvalResultDict
from deployment.metrics.detection_3d_metrics import Detection3DMetricsConfig, Detection3DMetricsInterface

logger = logging.getLogger(__name__)

_REQUIRED_SUMMARY_KEYS = ("mAP_by_mode", "mAPH_by_mode", "per_class_ap_by_mode")


class Detection3DEvaluator(BaseEvaluator):
    """Evaluator base for 3D-detection deployment: metrics hooks shared across detectors.

    Backend execution (pipeline creation, input prep, device handling) is delegated to the
    ``executor``; this class implements the task-generic metrics hooks. Subclasses provide only
    ``print_results`` (the latency-breakdown layout differs per model).

    Args:
        model_cfg: Model configuration; must have ``class_names``.
        metrics_config: Configuration for 3D detection metrics (e.g. T4MetricV2).
        executor: Backend execution primitives, shared with the verification runner.
    """

    def __init__(
        self,
        model_cfg: Config,
        metrics_config: Detection3DMetricsConfig,
        executor: BackendExecutor,
    ) -> None:
        if not hasattr(model_cfg, "class_names"):
            raise ValueError("class_names must be provided via model_cfg.class_names.")
        super().__init__(
            metrics_interface=Detection3DMetricsInterface(metrics_config),
            model_cfg=model_cfg,
            executor=executor,
        )

    @override
    def _parse_predictions(self, pipeline_output: object) -> List[Dict]:
        """Return pipeline output as a list of prediction dicts (empty list if not a list)."""
        return pipeline_output if isinstance(pipeline_output, list) else []

    @override
    def _parse_ground_truths(self, gt_data: Mapping[str, object]) -> List[Dict]:
        """Convert ``gt_bboxes_3d`` / ``gt_labels_3d`` into ``[{"bbox_3d": [...], "label": int}]``."""
        if "gt_bboxes_3d" not in gt_data:
            raise KeyError("gt_bboxes_3d not found in ground truth data.")
        if "gt_labels_3d" not in gt_data:
            raise KeyError("gt_labels_3d not found in ground truth data.")

        gt_bboxes_3d = np.asarray(gt_data["gt_bboxes_3d"], dtype=np.float32)
        box_dim = gt_bboxes_3d.shape[-1] if gt_bboxes_3d.ndim > 1 else 7
        gt_bboxes_3d = gt_bboxes_3d.reshape(-1, box_dim)
        gt_labels_3d = np.asarray(gt_data["gt_labels_3d"], dtype=np.int64).reshape(-1)
        return [{"bbox_3d": gt_bboxes_3d[i].tolist(), "label": int(gt_labels_3d[i])} for i in range(len(gt_bboxes_3d))]

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
        """Build the result dict (mAP/mAPH, per-class AP, detailed metrics, latency, breakdown)."""
        latency_stats = self.compute_latency_stats(latencies)
        map_results = self.metrics_interface.compute_metrics()
        summary_dict = self.metrics_interface.summary.to_dict()
        missing = [k for k in _REQUIRED_SUMMARY_KEYS if k not in summary_dict]
        if missing:
            raise KeyError(f"Missing required metrics summary keys: {missing}")

        result: EvalResultDict = {
            "mAP_by_mode": summary_dict["mAP_by_mode"],
            "mAPH_by_mode": summary_dict["mAPH_by_mode"],
            "per_class_ap_by_mode": summary_dict["per_class_ap_by_mode"],
            "detailed_metrics": map_results,
            "latency": latency_stats,
            "num_samples": num_samples,
        }
        if latency_breakdowns:
            result["latency_breakdown"] = self._compute_latency_breakdown(latency_breakdowns)
        return result

    @override
    def summarize_for_comparison(self, results: EvalResultDict) -> List[str]:
        """Summarize mAP/mAPH per mode for the cross-backend comparison."""
        lines: List[str] = []
        for mode, map_value in (results.get("mAP_by_mode") or {}).items():
            lines.append(f"  mAP ({mode}): {map_value:.4f}")
        for mode, maph_value in (results.get("mAPH_by_mode") or {}).items():
            lines.append(f"  mAPH ({mode}): {maph_value:.4f}")
        lines.extend(super().summarize_for_comparison(results))
        return lines

    def _log_latency_stats(self, results: EvalResultDict) -> None:
        """Log the shared latency-statistics block (mean/std/min/max/median)."""
        latency_dict = results["latency"].to_dict()
        logger.info("")
        logger.info("Latency Statistics:")
        logger.info("  Mean:   %.2f ms", latency_dict["mean_ms"])
        logger.info("  Std:    %.2f ms", latency_dict["std_ms"])
        logger.info("  Min:    %.2f ms", latency_dict["min_ms"])
        logger.info("  Max:    %.2f ms", latency_dict["max_ms"])
        logger.info("  Median: %.2f ms", latency_dict["median_ms"])
