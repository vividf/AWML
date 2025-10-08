"""
YOLOX Evaluator for deployment.

This module implements evaluation for YOLOX 2D object detection models.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import torch
from mmengine.config import Config

from autoware_ml.deployment.backends import ONNXBackend, PyTorchBackend, TensorRTBackend
from autoware_ml.deployment.core import BaseEvaluator

from .data_loader import YOLOXDataLoader

# Constants
LOG_INTERVAL = 100
GPU_CLEANUP_INTERVAL = 10


class YOLOXEvaluator(BaseEvaluator):
    """
    Evaluator for YOLOX 2D object detection.

    Computes detection metrics including mAP, per-class AP, and latency statistics.
    """

    def __init__(self, model_cfg: Config, class_names: List[str] = None):
        """
        Initialize YOLOX evaluator.

        Args:
            model_cfg: Model configuration
            class_names: List of class names (optional, will try to get from config)
        """
        super().__init__(config={})
        self.model_cfg = model_cfg

        # Get class names
        if class_names is not None:
            self.class_names = class_names
        elif hasattr(model_cfg, "class_names"):
            self.class_names = model_cfg.class_names
        elif "model" in model_cfg and "bbox_head" in model_cfg.model:
            num_classes = model_cfg.model.bbox_head.get("num_classes", 80)
            self.class_names = [f"class_{i}" for i in range(num_classes)]
        else:
            self.class_names = [f"class_{i}" for i in range(80)]  # COCO default

    def evaluate(
        self,
        model_path: str,
        data_loader: YOLOXDataLoader,
        num_samples: int,
        backend: str = "pytorch",
        device: str = "cpu",
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run full evaluation on YOLOX model.

        Args:
            model_path: Path to model checkpoint/weights
            data_loader: YOLOX DataLoader
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
                input_tensor = data_loader.preprocess(sample)

                # Get ground truth
                gt_data = data_loader.get_ground_truth(idx)

                # Run inference
                output, latency = inference_backend.infer(input_tensor)
                latencies.append(latency)

                # Parse predictions
                predictions = self._parse_predictions(output, gt_data["img_info"])
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

    def _parse_predictions(self, output: np.ndarray, img_info: Dict) -> List[Dict]:
        """
        Parse model output to prediction format.

        Args:
            output: Raw model output
            img_info: Image metadata

        Returns:
            List of prediction dicts with 'bbox', 'label', 'score'
        """
        predictions = []

        # YOLOX output format: [N, 7] where 7 = [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
        # or can be dict with 'bboxes', 'scores', 'labels'

        if isinstance(output, dict):
            # Dict format
            bboxes = output.get("bboxes", np.array([]))
            scores = output.get("scores", np.array([]))
            labels = output.get("labels", np.array([]))
        else:
            # Array format: assume shape [N, 7]
            if len(output.shape) == 2 and output.shape[1] >= 7:
                bboxes = output[:, :4]  # [x1, y1, x2, y2]
                scores = output[:, 4] * output[:, 5]  # obj_conf * cls_conf
                labels = output[:, 6].astype(int)
            else:
                # No detections
                return predictions

        # Convert to [x, y, w, h] format
        for bbox, score, label in zip(bboxes, scores, labels):
            if isinstance(bbox, np.ndarray) and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                predictions.append(
                    {
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
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
            List of ground truth dicts with 'bbox', 'label'
        """
        ground_truths = []

        gt_bboxes = gt_data["gt_bboxes"]
        gt_labels = gt_data["gt_labels"]

        for bbox, label in zip(gt_bboxes, gt_labels):
            ground_truths.append(
                {"bbox": bbox.tolist() if isinstance(bbox, np.ndarray) else bbox, "label": int(label)}
            )

        return ground_truths

    def _compute_metrics(
        self,
        predictions_list: List[List[Dict]],
        ground_truths_list: List[List[Dict]],
        latencies: List[float],
        logger,
    ) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        from autoware_ml.deployment.metrics import compute_map_coco

        # Compute mAP
        map_results = compute_map_coco(
            predictions_list,
            ground_truths_list,
            num_classes=len(self.class_names),
        )

        # Compute latency statistics
        latency_stats = self.compute_latency_stats(latencies)

        # Combine results
        results = {
            **map_results,
            "latency": latency_stats,
            "num_samples": len(predictions_list),
        }

        return results

    def print_results(self, results: Dict[str, Any]) -> None:
        """
        Pretty print evaluation results.

        Args:
            results: Results dictionary from evaluate()
        """
        print("\n" + "=" * 80)
        print("YOLOX Evaluation Results")
        print("=" * 80)

        # Detection metrics
        print(f"\nDetection Metrics:")
        print(f"  mAP (0.5:0.95): {results['mAP']:.4f}")
        print(f"  mAP @ IoU=0.50: {results['mAP_50']:.4f}")
        print(f"  mAP @ IoU=0.75: {results['mAP_75']:.4f}")

        # Per-class AP
        if "per_class_ap" in results and len(self.class_names) <= 20:
            print(f"\nPer-Class AP:")
            for class_id, ap in results["per_class_ap"].items():
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                print(f"  {class_name}: {ap:.4f}")

        # Latency
        print(f"\nLatency Statistics:")
        latency = results["latency"]
        print(f"  Mean: {latency['mean_ms']:.2f} ms")
        print(f"  Std:  {latency['std_ms']:.2f} ms")
        print(f"  Min:  {latency['min_ms']:.2f} ms")
        print(f"  Max:  {latency['max_ms']:.2f} ms")
        print(f"  Median: {latency['median_ms']:.2f} ms")

        print(f"\nTotal Samples: {results['num_samples']}")
        print("=" * 80)
