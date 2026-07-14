"""Shared evaluator for 3D detectors.

Scoring depends only on the 3D-detection outputs (``bbox_3d`` + ``label``), not on the input
modality, so any 3D detector — point-cloud (CenterPoint, BEVFusion) or camera-based — reuses this.
CenterPoint and BEVFusion score predictions the same way — via
:class:`~deployment.metrics.detection_3d_metrics.Detection3DMetricsInterface` — so the metrics
plumbing (prediction/GT parsing, result building, comparison summary) lives here once. Projects
subclass this and only override :meth:`print_results` when they want a custom latency layout;
the default here prints a generic metrics + latency + stage breakdown report.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping

import numpy as np
from mmdet3d.structures.ops import box_np_ops
from mmengine.config import Config
from typing_extensions import override

from deployment.evaluation.base_evaluator import BaseEvaluator, EvalResultDict
from deployment.execution.backend_executor import BackendExecutor
from deployment.metrics.detection_3d_metrics import Detection3DMetricsConfig, Detection3DMetricsInterface

logger = logging.getLogger(__name__)


class Detection3DEvaluator(BaseEvaluator):
    """Evaluator for 3D detection backed by ``Detection3DMetricsInterface`` (modality-agnostic).

    Args:
        model_cfg: Model configuration; must have ``class_names``.
        metrics_config: Configuration for 3D detection metrics (e.g. T4MetricV2).
        executor: Backend execution primitives, shared with the verification runner.

    Raises:
        ValueError: If ``model_cfg`` does not have ``class_names``.
    """

    def __init__(
        self,
        model_cfg: Config,
        metrics_config: Detection3DMetricsConfig,
        executor: BackendExecutor,
    ) -> None:
        if not hasattr(model_cfg, "class_names"):
            raise ValueError("class_names must be provided via model_cfg.class_names.")

        # Mirror T4MetricV2: drop GT boxes containing fewer than this many LiDAR points.
        self._min_num_points = metrics_config.min_num_points

        super().__init__(
            metrics_interface=Detection3DMetricsInterface(metrics_config),
            model_cfg=model_cfg,
            executor=executor,
        )

    @override
    def _parse_predictions(self, pipeline_output: Any) -> List[Dict]:
        """Return pipeline output as a list of prediction dicts (empty list if not a list)."""
        return pipeline_output if isinstance(pipeline_output, list) else []

    @override
    def _parse_ground_truths(self, sample: Mapping[str, Any]) -> List[Dict]:
        """Convert ``gt_bboxes_3d`` / ``gt_labels_3d`` into a list of ``{bbox_3d, label, num_lidar_pts}`` dicts.

        When ``min_num_points > 0`` (from T4MetricV2's config), GT boxes containing fewer than
        that many LiDAR points are dropped, exactly as T4MetricV2 does during training/testing:
        the point count is recomputed from the same input point cloud via
        :func:`box_np_ops.points_in_rbbox` (not read from the stored ``num_lidar_pts``), so the
        evaluated GT set matches ``test.py`` and mAP is comparable across the two pipelines.

        Raises:
            KeyError: If ``gt_bboxes_3d`` or ``gt_labels_3d`` is missing.
        """
        gt_data = sample["ground_truth"]
        if "gt_bboxes_3d" not in gt_data:
            raise KeyError("gt_bboxes_3d not found in ground truth data.")
        if "gt_labels_3d" not in gt_data:
            raise KeyError("gt_labels_3d not found in ground truth data.")

        gt_bboxes_3d = gt_data["gt_bboxes_3d"]
        gt_labels_3d = gt_data["gt_labels_3d"]

        gt_bboxes_3d = np.asarray(gt_bboxes_3d, dtype=np.float32).reshape(
            -1, np.asarray(gt_bboxes_3d).shape[-1] if np.asarray(gt_bboxes_3d).ndim > 1 else 7
        )
        gt_labels_3d = np.asarray(gt_labels_3d, dtype=np.int64).reshape(-1)

        num_lidar_pts = gt_data.get("num_lidar_pts", None)
        num_lidar_pts = None if num_lidar_pts is None else np.asarray(num_lidar_pts).reshape(-1)

        gt_bboxes_3d, gt_labels_3d, num_lidar_pts = self._filter_by_min_num_points(
            gt_bboxes_3d, gt_labels_3d, num_lidar_pts, sample.get("points")
        )

        ground_truths: List[Dict] = []
        for i in range(len(gt_bboxes_3d)):
            entry: Dict[str, Any] = {"bbox_3d": gt_bboxes_3d[i].tolist(), "label": int(gt_labels_3d[i])}
            if num_lidar_pts is not None:
                entry["num_lidar_pts"] = int(num_lidar_pts[i])
            ground_truths.append(entry)
        return ground_truths

    def _filter_by_min_num_points(
        self,
        gt_bboxes_3d: np.ndarray,
        gt_labels_3d: np.ndarray,
        num_lidar_pts: Any,
        points: Any,
    ) -> tuple:
        """Drop GT boxes with fewer than ``self._min_num_points`` points inside them.

        Point counts are recomputed from ``points`` (the same cloud fed to the model) via
        ``box_np_ops.points_in_rbbox``, matching T4MetricV2 exactly. No-op when filtering is
        disabled, there are no boxes, or no point cloud is available (e.g. camera-based 3D).
        """
        if self._min_num_points <= 0 or not len(gt_bboxes_3d) or points is None:
            return gt_bboxes_3d, gt_labels_3d, num_lidar_pts

        points_np = points.cpu().numpy() if hasattr(points, "cpu") else np.asarray(points)
        indices = box_np_ops.points_in_rbbox(points_np[:, :3], gt_bboxes_3d[:, :7])
        num_points_in_gt = indices.sum(0)
        mask = num_points_in_gt >= self._min_num_points

        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_labels_3d = gt_labels_3d[mask]
        if num_lidar_pts is not None:
            num_lidar_pts = num_lidar_pts[mask]
        return gt_bboxes_3d, gt_labels_3d, num_lidar_pts

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
        """Aggregate mAP/mAPH, per-class AP, latency, and optional breakdown into an EvalResultDict.

        Raises:
            KeyError: If the metrics summary is missing required keys.
        """
        latency_stats = self.compute_latency_stats(latencies)

        map_results = self.metrics_interface.compute_metrics()
        summary_dict = self.metrics_interface.summary.to_dict()
        required_summary_keys = ("mAP_by_mode", "mAPH_by_mode", "per_class_ap_by_mode")
        missing = [k for k in required_summary_keys if k not in summary_dict]
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

    def _log_metrics_report(self) -> None:
        """Log the metrics interface's formatted report line by line."""
        metrics_report = self.metrics_interface.format_metrics_report()
        if metrics_report:
            for line in metrics_report.rstrip().split("\n"):
                logger.info(line)

    def _log_latency_stats(self, results: EvalResultDict) -> None:
        """Log the latency-statistics block (mean/std/min/max/median).

        Raises:
            ValueError: If ``latency`` is missing from ``results``.
        """
        if "latency" not in results:
            raise ValueError(
                "Latency statistics not found in results. Ensure that evaluation has been run with latency tracking."
            )
        latency_dict = results["latency"].to_dict()
        logger.info("")
        logger.info("Latency Statistics:")
        logger.info("  Mean:   %.2f ms", latency_dict["mean_ms"])
        logger.info("  Std:    %.2f ms", latency_dict["std_ms"])
        logger.info("  Min:    %.2f ms", latency_dict["min_ms"])
        logger.info("  Max:    %.2f ms", latency_dict["max_ms"])
        logger.info("  Median: %.2f ms", latency_dict["median_ms"])

    @override
    def print_results(self, results: EvalResultDict) -> None:
        """Log the metrics report, latency statistics, and a generic stage-wise breakdown.

        Subclasses override this only when they want a custom breakdown layout.
        """
        self._log_metrics_report()
        self._log_latency_stats(results)

        if "latency_breakdown" in results:
            breakdown_dict = results["latency_breakdown"].to_dict()
            if breakdown_dict:
                logger.info("")
                logger.info("Stage-wise Latency Breakdown:")
                top_level_stages = {"preprocessing_ms", "model_ms", "postprocessing_ms"}
                for stage, stats_dict in breakdown_dict.items():
                    stage_name = stage.replace("_ms", "").replace("_", " ").title()
                    output_format = (
                        "  %-18s: %.2f ± %.2f ms" if stage in top_level_stages else "    %-16s: %.2f ± %.2f ms"
                    )
                    logger.info(output_format, stage_name, stats_dict["mean_ms"], stats_dict["std_ms"])

        logger.info("")
        logger.info("Total Samples: %s", results["num_samples"])
