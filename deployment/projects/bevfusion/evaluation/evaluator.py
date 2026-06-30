"""BEVFusion Evaluator for deployment."""

from __future__ import annotations

import logging
from typing import Dict, List, Mapping, Tuple

import numpy as np
from mmengine.config import Config
from typing_extensions import override

from deployment.evaluation.backend_executor import BackendExecutor
from deployment.evaluation.base_evaluator import BaseEvaluator, EvalResultDict
from deployment.metrics.detection_3d_metrics import Detection3DMetricsConfig, Detection3DMetricsInterface

logger = logging.getLogger(__name__)

# (stage_key, indent_level): indent 0 = top-level; each +1 adds one leading space before the label.
_BEVFUSION_LATENCY_STAGE_LAYOUT: Tuple[Tuple[str, int], ...] = (
    ("preprocessing_ms", 0),
    ("model_ms", 0),
    ("bevfusion_ms", 0),
    ("sparse_encoder_ms", 1),
    ("dense_engine_ms", 1),
    ("voxel_encoder_ms", 1),
    ("backbone_ms", 2),
    ("neck_ms", 2),
    ("head_ms", 2),
    ("post_scoring_ms", 2),
    ("dense_unattributed_ms", 2),
    ("postprocessing_ms", 0),
)

_BEVFUSION_STAGE_DISPLAY_NAME: Dict[str, str] = {
    "preprocessing_ms": "Preprocessing",
    "model_ms": "Model",
    "postprocessing_ms": "Postprocessing",
    "bevfusion_ms": "Bevfusion",
    "sparse_encoder_ms": "Sparse Encoder",
    "dense_engine_ms": "Dense Engine",
    "voxel_encoder_ms": "Voxel Encoder",
    "backbone_ms": "Backbone",
    "neck_ms": "Neck",
    "head_ms": "Head",
    "post_scoring_ms": "Post Scoring",
    "dense_unattributed_ms": "Dense Unattributed",
}


def _bevfusion_stage_display_name(stage_key: str) -> str:
    return _BEVFUSION_STAGE_DISPLAY_NAME.get(
        stage_key,
        stage_key.replace("_ms", "").replace("_", " ").title(),
    )


class BEVFusionEvaluator(BaseEvaluator):
    """Evaluator for BEVFusion 3D detection deployment.

    Backend execution (pipeline creation, input preparation, device handling) is delegated
    to a `BEVFusionExecutor`; this class only implements the task-specific metrics hooks.

    Args:
        model_cfg: Model configuration; must have class_names.
        metrics_config: Configuration for 3D detection metrics (e.g. T4MetricV2).
        executor: Backend execution primitives (a `BEVFusionExecutor`), shared with the
            verification runner.
    """

    def __init__(
        self,
        model_cfg: Config,
        metrics_config: Detection3DMetricsConfig,
        executor: BackendExecutor,
    ) -> None:
        if not hasattr(model_cfg, "class_names"):
            raise ValueError("class_names must be provided via model_cfg.class_names.")

        metrics_interface = Detection3DMetricsInterface(metrics_config)

        super().__init__(
            metrics_interface=metrics_interface,
            model_cfg=model_cfg,
            executor=executor,
        )

    @override
    def _parse_predictions(self, pipeline_output: object) -> List[Dict]:
        return pipeline_output if isinstance(pipeline_output, list) else []

    @override
    def _parse_ground_truths(self, gt_data: Mapping[str, object]) -> List[Dict]:
        if "gt_bboxes_3d" not in gt_data:
            raise KeyError("gt_bboxes_3d not found in ground truth data.")
        if "gt_labels_3d" not in gt_data:
            raise KeyError("gt_labels_3d not found in ground truth data.")

        gt_bboxes_3d = np.asarray(gt_data["gt_bboxes_3d"], dtype=np.float32)
        if gt_bboxes_3d.ndim == 1:
            gt_bboxes_3d = gt_bboxes_3d.reshape(-1, 7)
        gt_labels_3d = np.asarray(gt_data["gt_labels_3d"], dtype=np.int64).reshape(-1)

        ground_truths = []
        for i in range(len(gt_bboxes_3d)):
            ground_truths.append({"bbox_3d": gt_bboxes_3d[i].tolist(), "label": int(gt_labels_3d[i])})
        return ground_truths

    @override
    def _add_to_interface(self, predictions: List[Dict], ground_truths: List[Dict]) -> None:
        self.metrics_interface.add_frame(predictions, ground_truths)

    @override
    def _build_results(
        self,
        latencies: List[float],
        latency_breakdowns: List[Dict[str, float]],
        num_samples: int,
    ) -> EvalResultDict:
        latency_stats = self.compute_latency_stats(latencies)
        self.metrics_interface.compute_metrics()
        summary = self.metrics_interface.summary
        summary_dict = summary.to_dict()

        result: EvalResultDict = {
            "mAP_by_mode": summary_dict.get("mAP_by_mode", {}),
            "mAPH_by_mode": summary_dict.get("mAPH_by_mode", {}),
            "per_class_ap_by_mode": summary_dict.get("per_class_ap_by_mode", {}),
            "detailed_metrics": {},
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

    @override
    def print_results(self, results: EvalResultDict) -> None:
        metrics_report = self.metrics_interface.format_metrics_report()
        if metrics_report:
            for line in metrics_report.rstrip().split("\n"):
                logger.info(line)

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
            breakdown = results["latency_breakdown"]
            breakdown_dict = breakdown.to_dict() if hasattr(breakdown, "to_dict") else breakdown
            if breakdown_dict:
                logger.info("")
                logger.info("Stage-wise Latency Breakdown:")
                printed: set[str] = set()
                for stage_key, indent_level in _BEVFUSION_LATENCY_STAGE_LAYOUT:
                    if stage_key not in breakdown_dict:
                        continue
                    stats = breakdown_dict[stage_key]
                    stats_dict = stats.to_dict() if hasattr(stats, "to_dict") else stats
                    mean_ms = stats_dict.get("mean_ms", 0.0)
                    std_ms = stats_dict.get("std_ms", 0.0)
                    if mean_ms == 0.0 and std_ms == 0.0:
                        continue
                    printed.add(stage_key)
                    prefix = " " * (2 + indent_level)
                    label = _bevfusion_stage_display_name(stage_key)
                    logger.info("%s%-18s: %.2f ± %.2f ms", prefix, label, mean_ms, std_ms)

                extra_keys = sorted(k for k in breakdown_dict if k not in printed)
                for stage_key in extra_keys:
                    stats = breakdown_dict[stage_key]
                    stats_dict = stats.to_dict() if hasattr(stats, "to_dict") else stats
                    mean_ms = stats_dict.get("mean_ms", 0.0)
                    std_ms = stats_dict.get("std_ms", 0.0)
                    if mean_ms == 0.0 and std_ms == 0.0:
                        continue
                    label = _bevfusion_stage_display_name(stage_key)
                    logger.info("  %-18s: %.2f ± %.2f ms", label, mean_ms, std_ms)

        logger.info("")
        logger.info("Total Samples: %s", results["num_samples"])
