"""CenterPoint evaluator for deployment.

Thin subclass of ``Detection3DEvaluator``: only ``print_results`` (the flat latency-breakdown
layout) is CenterPoint-specific; the metrics hooks (parse/accumulate/build/summarize) are shared
with the base (see ``deployment.evaluation.detection3d_evaluator``).
"""

import logging

from typing_extensions import override

from deployment.evaluation.base_evaluator import EvalResultDict
from deployment.evaluation.detection3d_evaluator import Detection3DEvaluator

logger = logging.getLogger(__name__)


class CenterPointEvaluator(Detection3DEvaluator):
    """Evaluator for CenterPoint 3D detection deployment."""

    @override
    def print_results(self, results: EvalResultDict) -> None:
        """Log the metrics report, latency statistics, and stage-wise breakdown."""
        metrics_report = self.metrics_interface.format_metrics_report()
        for line in metrics_report.rstrip().split("\n"):
            logger.info(line)

        if "latency" not in results:
            raise ValueError(
                "Latency statistics not found in results. Ensure that evaluation has been run with latency tracking."
            )
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
                    logger.info(
                        output_format,
                        stage_name,
                        stats_dict["mean_ms"],
                        stats_dict["std_ms"],
                    )

        logger.info("")
        logger.info("Total Samples: %s", results["num_samples"])
