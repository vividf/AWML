"""BEVFusion evaluator for deployment.

Thin subclass of ``Detection3DEvaluator``: only ``print_results`` (BEVFusion's indented
sparse/dense stage-wise latency layout) is BEVFusion-specific; the metrics hooks
(parse/accumulate/build/summarize) are shared with the base
(see ``deployment.evaluation.detection3d_evaluator``).
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

from typing_extensions import override

from deployment.evaluation.base_evaluator import EvalResultDict
from deployment.evaluation.detection3d_evaluator import Detection3DEvaluator

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


class BEVFusionEvaluator(Detection3DEvaluator):
    """Evaluator for BEVFusion 3D detection deployment."""

    @override
    def print_results(self, results: EvalResultDict) -> None:
        """Log the metrics report, latency statistics, and BEVFusion's indented breakdown."""
        metrics_report = self.metrics_interface.format_metrics_report()
        if metrics_report:
            for line in metrics_report.rstrip().split("\n"):
                logger.info(line)

        if "latency" in results:
            self._log_latency_stats(results)

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
