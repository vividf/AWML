"""
CalibrationStatusClassification Evaluator for deployment.

This module implements the BaseEvaluator interface for evaluating
calibration status classification models.
Uses ClassificationMetricsInterface for consistent metric computation.
"""

import gc
import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from mmengine.config import Config

from deployment.core import (
    EVALUATION_DEFAULTS,
    Backend,
    BaseEvaluator,
    ClassificationMetricsConfig,
    ClassificationMetricsInterface,
    EvalResultDict,
    ModelSpec,
    TaskProfile,
)
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.pipelines.factory import PipelineFactory
from deployment.projects.calibration.data_loader import CalibrationDataLoader

logger = logging.getLogger(__name__)

LABELS = {"0": "miscalibrated", "1": "calibrated"}
CLASS_NAMES = ["miscalibrated", "calibrated"]


class CalibrationEvaluator(BaseEvaluator):
    """
    Evaluator for CalibrationStatusClassification task.

    Extends BaseEvaluator with classification-specific:
    - Pipeline creation (classification)
    - Image tensor input preparation
    - Classification ground truth parsing
    - ClassificationMetricsInterface integration

    Note: This evaluator has special handling for calibration data,
    evaluating both calibrated and miscalibrated versions of each sample.

    Args:
        model_cfg: Model configuration.
        class_names: List of class names. Defaults to ["miscalibrated", "calibrated"].
        metrics_config: Optional configuration for the classification metrics interface.
        components_cfg: Optional unified components configuration dict.
                       Used to resolve artifact file paths.
    """

    def __init__(
        self,
        model_cfg: Config,
        class_names: Optional[List[str]] = None,
        metrics_config: Optional[ClassificationMetricsConfig] = None,
        components_cfg: Optional[Mapping[str, Any]] = None,
    ):
        self._components_cfg = components_cfg or {}
        names = class_names if class_names is not None else CLASS_NAMES

        # Create task profile
        task_profile = TaskProfile(
            task_name="calibration_classification",
            display_name="Calibration Status Classification",
            class_names=tuple(names),
            num_classes=len(names),
        )

        # Create metrics interface
        if metrics_config is None:
            metrics_config = ClassificationMetricsConfig(class_names=list(names))
        metrics_interface = ClassificationMetricsInterface(metrics_config)

        super().__init__(
            metrics_interface=metrics_interface,
            task_profile=task_profile,
            model_cfg=model_cfg,
        )

    # ================== BaseEvaluator Implementation ==================

    def _create_pipeline(self, model_spec: ModelSpec, device: str) -> Any:
        """Create Calibration classification pipeline."""
        return PipelineFactory.create(
            project_name="calibration",
            model_spec=model_spec,
            pytorch_model=self.pytorch_model,
            device=device,
            components_cfg=self._components_cfg,
        )

    def _prepare_input(
        self,
        sample: Dict[str, Any],
        data_loader: BaseDataLoader,
        device: str,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Prepare image tensor input for classification."""
        input_tensor = data_loader.preprocess(sample)
        input_tensor = input_tensor.to(torch.device(device))
        return input_tensor, {}

    def _parse_predictions(self, pipeline_output: Any) -> Dict[str, Any]:
        """Parse classification output."""
        # Pipeline returns dict with class_id, class_name, confidence, probabilities
        return pipeline_output

    def _parse_ground_truths(self, gt_data: Dict[str, Any]) -> int:
        """Parse classification ground truth label."""
        return int(gt_data.get("gt_label", 0))

    def _add_to_interface(self, predictions: Dict[str, Any], ground_truths: int) -> None:
        """Add frame to ClassificationMetricsInterface."""
        prob = predictions.get("probabilities", [])
        prob_list = prob.tolist() if isinstance(prob, np.ndarray) else list(prob)

        self.metrics_interface.add_frame(
            prediction=int(predictions["class_id"]),
            ground_truth=ground_truths,
            probabilities=prob_list,
        )

    def _build_results(
        self,
        latencies: List[float],
        latency_breakdowns: List[Dict[str, float]],
        num_samples: int,
    ) -> EvalResultDict:
        """Build classification evaluation results."""
        interface_metrics = self.metrics_interface.compute_metrics()
        summary = self.metrics_interface.get_summary()
        summary_dict = summary.to_dict() if hasattr(summary, "to_dict") else summary
        confusion_matrix = self.metrics_interface.get_confusion_matrix()
        latency_stats = self.compute_latency_stats(latencies)

        return {
            "accuracy": interface_metrics.get("accuracy", 0.0),
            "precision": interface_metrics.get("precision", 0.0),
            "recall": interface_metrics.get("recall", 0.0),
            "f1score": interface_metrics.get("f1score", 0.0),
            "correct_predictions": interface_metrics.get("correct_predictions", 0),
            "total_samples": interface_metrics.get("total_samples", num_samples),
            "per_class_accuracy": summary_dict.get("per_class_accuracy", {}),
            "per_class_count": {
                i: int(interface_metrics.get(f"{self.class_names[i]}_num_gt", 0)) for i in range(len(self.class_names))
            },
            "confusion_matrix": confusion_matrix.tolist() if hasattr(confusion_matrix, "tolist") else confusion_matrix,
            "latency": latency_stats,  # Unified key name across all evaluators
            "detailed_metrics": interface_metrics,
            "num_samples": num_samples,
        }

    # ================== Override evaluate for dual-loader pattern ==================

    def evaluate(
        self,
        model: ModelSpec,
        data_loader: CalibrationDataLoader,
        num_samples: int,
        verbose: bool = False,
    ) -> EvalResultDict:
        """
        Run evaluation on classification model.

        This evaluator has special handling: it evaluates both calibrated
        and miscalibrated versions of each sample.

        Args:
            model: Model specification (backend/device/path)
            data_loader: Data loader for calibrated samples
            num_samples: Number of samples to evaluate
            verbose: Whether to print progress

        Returns:
            Evaluation results dictionary
        """
        logger.info(f"\nEvaluating {model.backend.value} model: {model.path}")
        logger.info(f"Number of samples: {num_samples}")

        # Create pipeline
        self._ensure_model_on_device(model.device)
        pipeline = self._create_pipeline(model, model.device)

        # Create both calibrated and miscalibrated loaders
        data_loader_miscalibrated = CalibrationDataLoader(
            info_pkl_path=data_loader.info_pkl_path,
            model_cfg=data_loader.model_cfg,
            miscalibration_probability=1.0,
            device=model.device,
        )
        data_loader_calibrated = data_loader

        # Reset metrics interface
        self.metrics_interface.reset()

        latencies = []
        latency_breakdowns = []

        actual_samples = min(num_samples, data_loader.get_num_samples())
        log_interval = EVALUATION_DEFAULTS.LOG_INTERVAL
        cleanup_interval = EVALUATION_DEFAULTS.GPU_CLEANUP_INTERVAL

        for idx in range(actual_samples):
            if verbose and idx % log_interval == 0:
                logger.info(f"Processing sample {idx + 1}/{actual_samples}")

            # Process both versions of each sample
            for loader in [data_loader_miscalibrated, data_loader_calibrated]:
                sample = loader.load_sample(idx)
                gt_data = loader.get_ground_truth(idx)

                input_tensor, infer_kwargs = self._prepare_input(sample, loader, model.device)
                ground_truth = self._parse_ground_truths(gt_data)

                raw_output = pipeline.infer(input_tensor, **infer_kwargs)
                latencies.append(raw_output.latency_ms)
                if raw_output.breakdown:
                    latency_breakdowns.append(raw_output.breakdown)

                predictions = self._parse_predictions(raw_output.output)
                self._add_to_interface(predictions, ground_truth)

            # GPU memory cleanup
            if model.backend is Backend.TENSORRT and idx % cleanup_interval == 0:
                self._clear_gpu_memory()

        return self._build_results(latencies, latency_breakdowns, len(latencies))

    def _clear_gpu_memory(self) -> None:
        """Clear GPU cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def print_results(self, results: EvalResultDict) -> None:
        """Pretty print evaluation results."""
        if "error" in results:
            logger.error(f"Evaluation error: {results['error']}")
            return

        print(f"\n{'='*70}")
        print(f"{self.task_profile.display_name} - Evaluation Results")
        print("(Using ClassificationMetricsInterface for consistent metrics)")
        print(f"{'='*70}")

        print(f"\nClassification Metrics:")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Correct predictions: {results['correct_predictions']}")
        print(f"  Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"  Precision: {results.get('precision', 0.0):.4f}")
        print(f"  Recall: {results.get('recall', 0.0):.4f}")
        print(f"  F1 Score: {results.get('f1score', 0.0):.4f}")

        if "latency" in results:
            print(f"\n{self.format_latency_stats(results['latency'])}")

        if results.get("per_class_accuracy"):
            print(f"\nPer-class accuracy:")
            for cls_name, acc in results["per_class_accuracy"].items():
                label = cls_name if isinstance(cls_name, str) else LABELS.get(str(cls_name), f"class_{cls_name}")
                count = results.get("per_class_count", {}).get(
                    list(self.class_names).index(cls_name) if cls_name in self.class_names else 0, 0
                )
                print(f"  {label}: {acc:.4f} ({acc*100:.2f}%) - {count} samples")

        if results.get("confusion_matrix"):
            cm = np.array(results["confusion_matrix"])
            if cm.size > 0:
                print(f"\nConfusion Matrix:")
                print(f"  Predicted →")
                print(f"  GT ↓        {' '.join([f'{i:>8}' for i in range(len(cm))])}")
                for i, row in enumerate(cm):
                    print(f"  {i:>8}    {' '.join([f'{val:>8}' for val in row])}")

        print(f"{'='*70}\n")
