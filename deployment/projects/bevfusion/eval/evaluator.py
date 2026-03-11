"""BEVFusion Evaluator for deployment."""

from __future__ import annotations

import logging
from typing import Dict, List, Mapping, Optional

import numpy as np
from mmengine.config import Config
from typing_extensions import override

from deployment.configs import ComponentsConfig
from deployment.core import (
    BaseEvaluator,
    Detection3DMetricsConfig,
    Detection3DMetricsInterface,
    EvalResultDict,
    InferenceInput,
    ModelSpec,
    TaskProfile,
)
from deployment.core.device import DeviceSpec
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.pipelines.base_pipeline import BaseDeploymentPipeline
from deployment.pipelines.factory import PipelineFactory

logger = logging.getLogger(__name__)


class BEVFusionEvaluator(BaseEvaluator):
    """Evaluator for BEVFusion 3D detection deployment."""

    def __init__(
        self,
        model_cfg: Config,
        metrics_config: Detection3DMetricsConfig,
        components_cfg: ComponentsConfig,
    ) -> None:
        if hasattr(model_cfg, "class_names"):
            class_names = model_cfg.class_names
        else:
            raise ValueError("class_names must be provided via model_cfg.class_names.")

        self._components_cfg = components_cfg

        task_profile = TaskProfile(
            task_name="bevfusion_3d_detection",
            display_name="BEVFusion 3D Object Detection",
            class_names=tuple(class_names),
            num_classes=len(class_names),
        )

        metrics_interface = Detection3DMetricsInterface(metrics_config)

        super().__init__(
            metrics_interface=metrics_interface,
            task_profile=task_profile,
            model_cfg=model_cfg,
        )

    @override
    def _get_output_names(self) -> Optional[List[str]]:
        return [out.name for out in self._components_cfg.get_component("bevfusion_main_body").io.outputs]

    @override
    def _create_pipeline(self, model_spec: ModelSpec, device: DeviceSpec) -> BaseDeploymentPipeline:
        return PipelineFactory.create(
            project_name="bevfusion",
            model_spec=model_spec,
            pytorch_model=self.pytorch_model,
            device=device,
            components_cfg=self._components_cfg,
        )

    @override
    def _prepare_input(
        self,
        sample: Mapping[str, object],
        data_loader: BaseDataLoader,
        device: DeviceSpec,
    ) -> InferenceInput:
        if "points" not in sample:
            raise ValueError(f"Expected 'points' in sample. Got keys: {list(sample.keys())}")
        if "metainfo" not in sample:
            raise KeyError("Sample must contain 'metainfo' for BEVFusion postprocess.")
        return InferenceInput(data=sample["points"], metadata=sample["metainfo"])

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
    def print_results(self, results: EvalResultDict) -> None:
        metrics_report = self.metrics_interface.format_metrics_report()
        if metrics_report:
            print(metrics_report)

        if "latency" in results:
            latency_dict = results["latency"].to_dict()
            print("\nLatency Statistics:")
            print(f"  Mean:   {latency_dict['mean_ms']:.2f} ms")
            print(f"  Std:    {latency_dict['std_ms']:.2f} ms")
            print(f"  Min:    {latency_dict['min_ms']:.2f} ms")
            print(f"  Max:    {latency_dict['max_ms']:.2f} ms")
            print(f"  Median: {latency_dict['median_ms']:.2f} ms")

        if "latency_breakdown" in results:
            breakdown = results["latency_breakdown"]
            breakdown_dict = breakdown.to_dict() if hasattr(breakdown, "to_dict") else breakdown
            if breakdown_dict:
                print("\nStage-wise Latency Breakdown:")
                for stage, stats in breakdown_dict.items():
                    stats_dict = stats.to_dict() if hasattr(stats, "to_dict") else stats
                    stage_name = stage.replace("_ms", "").replace("_", " ").title()
                    print(f"  {stage_name:18s}: {stats_dict['mean_ms']:.2f} ± {stats_dict['std_ms']:.2f} ms")

        print(f"\nTotal Samples: {results['num_samples']}")
