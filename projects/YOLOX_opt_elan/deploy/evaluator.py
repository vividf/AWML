"""
YOLOX_opt_elan Evaluator for deployment.

This module implements evaluation for YOLOX_opt_elan object detection models.
Uses autoware_perception_evaluation via Detection2DMetricsAdapter for consistent
metric computation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from mmengine.config import Config

from deployment.core import (
    BaseEvaluator,
    Detection2DMetricsAdapter,
    Detection2DMetricsConfig,
    EvalResultDict,
    ModelSpec,
    TaskProfile,
)
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.pipelines import PipelineFactory

logger = logging.getLogger(__name__)


class YOLOXOptElanEvaluator(BaseEvaluator):
    """
    Evaluator for YOLOX_opt_elan object detection.

    Extends BaseEvaluator with YOLOX-specific:
    - Pipeline creation (2D detection)
    - Image tensor input preparation
    - 2D bounding box ground truth parsing
    - Detection2DMetricsAdapter integration
    """

    def __init__(
        self,
        model_cfg: Config,
        class_names: Optional[List[str]] = None,
        metrics_config: Optional[Detection2DMetricsConfig] = None,
    ):
        """
        Initialize YOLOX_opt_elan evaluator.

        Args:
            model_cfg: Model configuration.
            class_names: List of class names (optional).
            metrics_config: Optional configuration for the 2D detection metrics adapter.
        """

        # Determine class names
        if class_names is not None:
            names = class_names
        elif hasattr(model_cfg, "classes"):
            classes = model_cfg.classes
            if not isinstance(classes, (tuple, list)):
                raise ValueError(f"Config 'classes' must be a tuple or list, got {type(classes)}.")
            names = list(classes)
        else:
            raise ValueError("Config file must contain 'classes' attribute.")

        # Create task profile
        task_profile = TaskProfile(
            task_name="yolox_2d_detection",
            display_name="YOLOX_opt_elan Object Detection",
            class_names=tuple(names),
            num_classes=len(names),
        )

        # Create metrics adapter
        if metrics_config is None:
            metrics_config = Detection2DMetricsConfig(
                class_names=list(names),
                iou_thresholds=[0.5, 0.75],
            )
        metrics_adapter = Detection2DMetricsAdapter(metrics_config)

        super().__init__(
            metrics_adapter=metrics_adapter,
            task_profile=task_profile,
            model_cfg=model_cfg,
        )

    # ================== BaseEvaluator Implementation ==================

    def _create_pipeline(self, model_spec: ModelSpec, device: str) -> Any:
        """Create YOLOX pipeline."""
        return PipelineFactory.create_yolox_pipeline(
            model_spec=model_spec,
            pytorch_model=self.pytorch_model,
            num_classes=self.task_profile.num_classes,
            class_names=list(self.class_names),
            device=device,
        )

    def _prepare_input(
        self,
        sample: Dict[str, Any],
        data_loader: BaseDataLoader,
        device: str,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Prepare image tensor input for YOLOX."""
        input_tensor = data_loader.preprocess(sample)
        input_tensor = input_tensor.to(torch.device(device))

        # Get image info for inference kwargs
        sample_idx = sample.get("sample_idx", 0)
        gt_data = data_loader.get_ground_truth(sample_idx)
        infer_kwargs = {"img_info": gt_data.get("img_info", {})}

        return input_tensor, infer_kwargs

    def _parse_predictions(self, pipeline_output: Any) -> List[Dict]:
        """Normalize YOLOX predictions to standard format."""
        predictions = []
        for det in pipeline_output:
            if isinstance(det, dict) and "bbox" in det:
                predictions.append(
                    {
                        "bbox": det["bbox"],
                        "label": int(det.get("class_id", det.get("label", 0))),
                        "score": float(det.get("score", 0.0)),
                    }
                )
        return predictions

    def _parse_ground_truths(self, gt_data: Dict[str, Any]) -> List[Dict]:
        """Parse 2D ground truth bounding boxes."""
        ground_truths = []
        gt_bboxes = gt_data.get("gt_bboxes", [])
        gt_labels = gt_data.get("gt_labels", [])

        for bbox, label in zip(gt_bboxes, gt_labels):
            ground_truths.append(
                {
                    "bbox": list(bbox),
                    "label": int(label),
                }
            )

        return ground_truths

    def _add_to_adapter(self, predictions: List[Dict], ground_truths: List[Dict]) -> None:
        """Add frame to Detection2DMetricsAdapter."""
        self.metrics_adapter.add_frame(predictions, ground_truths)

    def _build_results(
        self,
        latencies: List[float],
        latency_breakdowns: List[Dict[str, float]],
        num_samples: int,
    ) -> EvalResultDict:
        """Build YOLOX evaluation results."""
        latency_stats = self.compute_latency_stats(latencies)

        map_results = self.metrics_adapter.compute_metrics()
        summary = self.metrics_adapter.get_summary()

        return {
            "mAP": summary.get("mAP", 0.0),
            "mAP_50": map_results.get("mAP_iou_2d_0.5", summary.get("mAP", 0.0)),
            "mAP_75": map_results.get("mAP_iou_2d_0.75", 0.0),
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
        print(f"  mAP @ IoU=0.50: {results.get('mAP_50', 0.0):.4f}")
        print(f"  mAP @ IoU=0.75: {results.get('mAP_75', 0.0):.4f}")

        if "per_class_ap" in results:
            print(f"\nPer-Class AP:")
            for class_id, ap in results["per_class_ap"].items():
                class_name = (
                    class_id
                    if isinstance(class_id, str)
                    else (
                        self.class_names[class_id]
                        if isinstance(class_id, int) and class_id < len(self.class_names)
                        else f"class_{class_id}"
                    )
                )
                ap_value = ap.get("ap", 0.0) if isinstance(ap, dict) else float(ap) if ap is not None else 0.0
                print(f"  {class_name:25s}: {ap_value:.4f}")

        if "latency" in results:
            latency = results["latency"]
            print(f"\nLatency Statistics:")
            print(f"  Mean:   {latency['mean_ms']:.2f} ms")
            print(f"  Std:    {latency['std_ms']:.2f} ms")
            print(f"  Min:    {latency['min_ms']:.2f} ms")
            print(f"  Max:    {latency['max_ms']:.2f} ms")
            print(f"  Median: {latency['median_ms']:.2f} ms")

        print(f"\nTotal Samples: {results['num_samples']}")
        print("=" * 80)
