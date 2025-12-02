"""
CenterPoint Evaluator for deployment.

This module implements evaluation for CenterPoint 3D object detection models.
Uses autoware_perception_evaluation via Detection3DMetricsAdapter for consistent
metric computation between training (T4MetricV2) and deployment.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from mmengine.config import Config

from deployment.core import (
    BaseEvaluator,
    Detection3DMetricsAdapter,
    Detection3DMetricsConfig,
    EvalResultDict,
    ModelSpec,
    TaskProfile,
)
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.pipelines import PipelineFactory
from projects.CenterPoint.deploy.configs.deploy_config import model_io

logger = logging.getLogger(__name__)


class CenterPointEvaluator(BaseEvaluator):
    """
    Evaluator for CenterPoint 3D object detection.

    Extends BaseEvaluator with CenterPoint-specific:
    - Pipeline creation (multi-stage 3D detection)
    - Point cloud input preparation
    - 3D bounding box ground truth parsing
    - Detection3DMetricsAdapter integration
    """

    def __init__(
        self,
        model_cfg: Config,
        metrics_config: Detection3DMetricsConfig,
    ):
        """
        Initialize CenterPoint evaluator.

        Args:
            model_cfg: Model configuration.
            metrics_config: Configuration for the metrics adapter.
        """
        # Determine class names - must come from config
        if hasattr(model_cfg, "class_names"):
            class_names = model_cfg.class_names
        else:
            raise ValueError(
                "class_names must be provided via model_cfg.class_names. "
                "Check your model config file includes class_names definition."
            )

        # Create task profile
        task_profile = TaskProfile(
            task_name="centerpoint_3d_detection",
            display_name="CenterPoint 3D Object Detection",
            class_names=tuple(class_names),
            num_classes=len(class_names),
        )

        metrics_adapter = Detection3DMetricsAdapter(metrics_config)

        super().__init__(
            metrics_adapter=metrics_adapter,
            task_profile=task_profile,
            model_cfg=model_cfg,
        )

    def set_onnx_config(self, model_cfg: Config) -> None:
        """Set ONNX-compatible model config."""
        self.model_cfg = model_cfg

    # ================== VerificationMixin Override ==================

    def _get_output_names(self) -> List[str]:
        """Provide meaningful names for CenterPoint head outputs."""
        return list(model_io["head_output_names"])

    # ================== BaseEvaluator Implementation ==================

    def _create_pipeline(self, model_spec: ModelSpec, device: str) -> Any:
        """Create CenterPoint pipeline."""
        return PipelineFactory.create_centerpoint_pipeline(
            model_spec=model_spec,
            pytorch_model=self.pytorch_model,
            device=device,
        )

    def _prepare_input(
        self,
        sample: Dict[str, Any],
        data_loader: BaseDataLoader,
        device: str,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Prepare point cloud input for CenterPoint."""
        if "points" in sample:
            points = sample["points"]
        else:
            input_data = data_loader.preprocess(sample)
            points = input_data.get("points", input_data)

        metadata = sample.get("metainfo", {})
        return points, metadata

    def _parse_predictions(self, pipeline_output: Any) -> List[Dict]:
        """Parse CenterPoint predictions (already in standard format)."""
        # Pipeline already returns list of dicts with bbox_3d, score, label
        return pipeline_output if isinstance(pipeline_output, list) else []

    def _parse_ground_truths(self, gt_data: Dict[str, Any]) -> List[Dict]:
        """Parse 3D ground truth bounding boxes."""
        ground_truths = []

        if "gt_bboxes_3d" in gt_data and "gt_labels_3d" in gt_data:
            gt_bboxes_3d = gt_data["gt_bboxes_3d"]
            gt_labels_3d = gt_data["gt_labels_3d"]

            for i in range(len(gt_bboxes_3d)):
                ground_truths.append(
                    {
                        "bbox_3d": gt_bboxes_3d[i].tolist(),
                        "label": int(gt_labels_3d[i]),
                    }
                )

        return ground_truths

    def _add_to_adapter(self, predictions: List[Dict], ground_truths: List[Dict]) -> None:
        """Add frame to Detection3DMetricsAdapter."""
        self.metrics_adapter.add_frame(predictions, ground_truths)

    def _build_results(
        self,
        latencies: List[float],
        latency_breakdowns: List[Dict[str, float]],
        num_samples: int,
    ) -> EvalResultDict:
        """Build CenterPoint evaluation results."""
        # Compute latency statistics
        latency_stats = self.compute_latency_stats(latencies)

        # Add stage-wise breakdown if available
        if latency_breakdowns:
            latency_stats["latency_breakdown"] = self._compute_latency_breakdown(latency_breakdowns)

        # Get metrics from adapter
        map_results = self.metrics_adapter.compute_metrics()
        summary = self.metrics_adapter.get_summary()

        return {
            "mAP": summary.get("mAP", 0.0),
            "mAPH": summary.get("mAPH", 0.0),
            "per_class_ap": summary.get("per_class_ap", {}),
            "detailed_metrics": map_results,
            "latency": latency_stats,
            "num_samples": num_samples,
        }

    def print_results(self, results: EvalResultDict) -> None:
        """Pretty print evaluation results."""
        print("\n" + "=" * 80)
        print(f"{self.task_profile.display_name} - Evaluation Results")
        print("(Using autoware_perception_evaluation for consistent metrics)")
        print("=" * 80)

        print(f"\nDetection Metrics:")
        print(f"  mAP: {results.get('mAP', 0.0):.4f}")
        print(f"  mAPH: {results.get('mAPH', 0.0):.4f}")

        if "per_class_ap" in results:
            print(f"\nPer-Class AP:")
            for class_id, ap in results["per_class_ap"].items():
                class_name = (
                    class_id
                    if isinstance(class_id, str)
                    else (self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}")
                )
                print(f"  {class_name:<12}: {ap:.4f}")

        if "latency" in results:
            latency = results["latency"]
            print(f"\nLatency Statistics:")
            print(f"  Mean:   {latency['mean_ms']:.2f} ms")
            print(f"  Std:    {latency['std_ms']:.2f} ms")
            print(f"  Min:    {latency['min_ms']:.2f} ms")
            print(f"  Max:    {latency['max_ms']:.2f} ms")
            print(f"  Median: {latency['median_ms']:.2f} ms")

            if "latency_breakdown" in latency:
                breakdown = latency["latency_breakdown"]
                print(f"\nStage-wise Latency Breakdown:")
                # Sub-stages that belong under "Model"
                model_substages = {"voxel_encoder_ms", "middle_encoder_ms", "backbone_head_ms"}
                for stage, stats in breakdown.items():
                    stage_name = stage.replace("_ms", "").replace("_", " ").title()
                    # Use extra indentation for model sub-stages
                    if stage in model_substages:
                        print(f"    {stage_name:16s}: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
                    else:
                        print(f"  {stage_name:18s}: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")

        print(f"\nTotal Samples: {results.get('num_samples', 0)}")
        print("=" * 80)
