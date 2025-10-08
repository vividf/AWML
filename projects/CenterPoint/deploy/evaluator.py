"""
CenterPoint Evaluator for deployment.

This module implements evaluation for CenterPoint 3D object detection models.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import torch
from mmengine.config import Config

from autoware_ml.deployment.backends import ONNXBackend, PyTorchBackend, TensorRTBackend
from autoware_ml.deployment.core import BaseEvaluator

from .data_loader import CenterPointDataLoader

# Constants
LOG_INTERVAL = 50
GPU_CLEANUP_INTERVAL = 10


class CenterPointEvaluator(BaseEvaluator):
    """
    Evaluator for CenterPoint 3D object detection.

    Computes 3D detection metrics including mAP, NDS, and latency statistics.

    Note: For production, should integrate with mmdet3d's evaluation metrics.
    """

    def __init__(self, model_cfg: Config, class_names: List[str] = None):
        """
        Initialize CenterPoint evaluator.

        Args:
            model_cfg: Model configuration
            class_names: List of class names (optional)
        """
        super().__init__(config={})
        self.model_cfg = model_cfg

        # Get class names
        if class_names is not None:
            self.class_names = class_names
        elif hasattr(model_cfg, "class_names"):
            self.class_names = model_cfg.class_names
        else:
            # Default for T4Dataset
            self.class_names = ["VEHICLE", "PEDESTRIAN", "CYCLIST"]

    def evaluate(
        self,
        model_path: str,
        data_loader: CenterPointDataLoader,
        num_samples: int,
        backend: str = "pytorch",
        device: str = "cpu",
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run full evaluation on CenterPoint model.

        Args:
            model_path: Path to model checkpoint/weights
            data_loader: CenterPoint DataLoader
            num_samples: Number of samples to evaluate
            backend: Backend to use ('pytorch', 'onnx', 'tensorrt')
            device: Device to run inference on
            verbose: Whether to print detailed progress

        Returns:
            Dictionary containing evaluation metrics
        """
        logger = logging.getLogger(__name__)
        logger.info(f"\nEvaluating {backend.upper()} model: {model_path}")
        logger.info(f"Number of samples: {num_samples}")

        # Limit num_samples
        total_samples = data_loader.get_num_samples()
        num_samples = min(num_samples, total_samples)

        # Create backend
        inference_backend = self._create_backend(backend, model_path, device, logger)

        # Run inference
        all_predictions = []
        all_ground_truths = []
        latencies = []

        with inference_backend:
            for idx in range(num_samples):
                if idx % LOG_INTERVAL == 0:
                    logger.info(f"Processing sample {idx + 1}/{num_samples}")

                # Load and preprocess
                sample = data_loader.load_sample(idx)
                input_data = data_loader.preprocess(sample)

                # Get ground truth
                gt_data = data_loader.get_ground_truth(idx)

                # Run inference
                output, latency = inference_backend.infer(input_data)
                latencies.append(latency)

                # Parse predictions
                predictions = self._parse_predictions(output)
                all_predictions.append(predictions)

                # Parse ground truths
                ground_truths = self._parse_ground_truths(gt_data)
                all_ground_truths.append(ground_truths)

                # GPU cleanup for TensorRT
                if backend == "tensorrt" and idx % GPU_CLEANUP_INTERVAL == 0:
                    torch.cuda.empty_cache()

        # Compute metrics
        results = self._compute_metrics(all_predictions, all_ground_truths, latencies, logger)

        return results

    def _create_backend(self, backend: str, model_path: str, device: str, logger):
        """Create inference backend."""
        if backend == "pytorch":
            return PyTorchBackend(model_path, self.model_cfg, device)
        elif backend == "onnx":
            return ONNXBackend(model_path, device)
        elif backend == "tensorrt":
            return TensorRTBackend(model_path, device)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _parse_predictions(self, output) -> List[Dict]:
        """
        Parse model output to prediction format.

        Args:
            output: Raw model output (can be dict or array)

        Returns:
            List of prediction dicts with 'bbox_3d', 'label', 'score'
        """
        predictions = []

        # CenterPoint output can be dict or tensor
        if isinstance(output, dict):
            # Dict format: {'boxes_3d', 'scores_3d', 'labels_3d'}
            boxes_3d = output.get("boxes_3d", output.get("bboxes_3d", np.array([])))
            scores = output.get("scores_3d", output.get("scores", np.array([])))
            labels = output.get("labels_3d", output.get("labels", np.array([])))
        else:
            # Tensor/array format: need to parse based on model structure
            # This is model-specific, adjust as needed
            logger = logging.getLogger(__name__)
            logger.warning("Output is not dict format, attempting to parse...")

            # Placeholder: assume no predictions
            return predictions

        # Convert to prediction format
        for bbox, score, label in zip(boxes_3d, scores, labels):
            if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) >= 7:
                predictions.append(
                    {
                        "bbox_3d": bbox[:7].tolist() if isinstance(bbox, np.ndarray) else list(bbox[:7]),
                        "label": int(label),
                        "score": float(score),
                    }
                )

        return predictions

    def _parse_ground_truths(self, gt_data: Dict) -> List[Dict]:
        """
        Parse ground truth data.

        Args:
            gt_data: Ground truth data from data_loader

        Returns:
            List of ground truth dicts with 'bbox_3d', 'label'
        """
        ground_truths = []

        gt_bboxes_3d = gt_data["gt_bboxes_3d"]
        gt_labels_3d = gt_data["gt_labels_3d"]

        for bbox, label in zip(gt_bboxes_3d, gt_labels_3d):
            ground_truths.append(
                {"bbox_3d": bbox.tolist() if isinstance(bbox, np.ndarray) else bbox, "label": int(label)}
            )

        return ground_truths

    def _compute_metrics(
        self,
        predictions_list: List[List[Dict]],
        ground_truths_list: List[List[Dict]],
        latencies: List[float],
        logger,
    ) -> Dict[str, Any]:
        """
        Compute evaluation metrics.

        Note: This is a simplified version. For production, should use
        mmdet3d's official evaluation metrics.
        """
        # Compute basic statistics
        total_predictions = sum(len(preds) for preds in predictions_list)
        total_ground_truths = sum(len(gts) for gts in ground_truths_list)

        # Per-class statistics
        per_class_preds = {i: 0 for i in range(len(self.class_names))}
        per_class_gts = {i: 0 for i in range(len(self.class_names))}

        for preds in predictions_list:
            for pred in preds:
                label = pred["label"]
                if label < len(self.class_names):
                    per_class_preds[label] += 1

        for gts in ground_truths_list:
            for gt in gts:
                label = gt["label"]
                if label < len(self.class_names):
                    per_class_gts[label] += 1

        # Compute latency statistics
        latency_stats = self.compute_latency_stats(latencies)

        # Combine results
        results = {
            "total_predictions": total_predictions,
            "total_ground_truths": total_ground_truths,
            "per_class_predictions": per_class_preds,
            "per_class_ground_truths": per_class_gts,
            "latency": latency_stats,
            "num_samples": len(predictions_list),
        }

        # Note: For production, add proper 3D mAP computation
        logger.warning(
            "Using simplified metrics. For production, integrate with "
            "mmdet3d.core.evaluation for proper 3D detection metrics (mAP, NDS, etc.)"
        )

        return results

    def print_results(self, results: Dict[str, Any]) -> None:
        """
        Pretty print evaluation results.

        Args:
            results: Results dictionary from evaluate()
        """
        print("\n" + "=" * 80)
        print("CenterPoint Evaluation Results")
        print("=" * 80)

        # Basic statistics
        print(f"\nDetection Statistics:")
        print(f"  Total Predictions: {results['total_predictions']}")
        print(f"  Total Ground Truths: {results['total_ground_truths']}")

        # Per-class statistics
        print(f"\nPer-Class Statistics:")
        for class_id in range(len(self.class_names)):
            class_name = self.class_names[class_id]
            num_preds = results["per_class_predictions"].get(class_id, 0)
            num_gts = results["per_class_ground_truths"].get(class_id, 0)
            print(f"  {class_name}:")
            print(f"    Predictions: {num_preds}")
            print(f"    Ground Truths: {num_gts}")

        # Latency
        print(f"\nLatency Statistics:")
        latency = results["latency"]
        print(f"  Mean: {latency['mean_ms']:.2f} ms")
        print(f"  Std:  {latency['std_ms']:.2f} ms")
        print(f"  Min:  {latency['min_ms']:.2f} ms")
        print(f"  Max:  {latency['max_ms']:.2f} ms")
        print(f"  Median: {latency['median_ms']:.2f} ms")

        print(f"\nTotal Samples: {results['num_samples']}")

        print("\n" + "⚠" * 40)
        print("NOTE: Using simplified metrics.")
        print("For production, integrate with mmdet3d.core.evaluation")
        print("for proper 3D detection metrics (mAP, NDS, mATE, etc.)")
        print("⚠" * 40)

        print("=" * 80)
