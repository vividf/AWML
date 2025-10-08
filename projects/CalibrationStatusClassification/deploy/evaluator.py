"""
Classification Evaluator for CalibrationStatusClassification.

This module implements the BaseEvaluator interface for evaluating
calibration status classification models.
"""

import gc
import logging
from typing import Any, Dict

import numpy as np
import torch
from mmengine.config import Config
from mmpretrain.apis import get_model

from autoware_ml.deployment.backends import ONNXBackend, PyTorchBackend, TensorRTBackend
from autoware_ml.deployment.core import BaseEvaluator

from .data_loader import CalibrationDataLoader

# Label mapping
LABELS = {"0": "miscalibrated", "1": "calibrated"}

# Constants for evaluation
LOG_INTERVAL = 100  # Log progress every N samples
GPU_CLEANUP_INTERVAL = 10  # Clear GPU memory every N samples for TensorRT


class ClassificationEvaluator(BaseEvaluator):
    """
    Evaluator for classification tasks.

    Computes accuracy, per-class metrics, confusion matrix, and latency statistics.
    """

    def __init__(self, model_cfg: Config):
        """
        Initialize classification evaluator.

        Args:
            model_cfg: Model configuration
        """
        super().__init__(config={})
        self.model_cfg = model_cfg

    def evaluate(
        self,
        model_path: str,
        data_loader: CalibrationDataLoader,
        num_samples: int,
        backend: str = "pytorch",
        device: str = "cpu",
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run full evaluation on a model.

        Args:
            model_path: Path to model checkpoint/weights
            data_loader: DataLoader for loading samples (calibrated version)
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

        # Limit num_samples to available data
        total_samples = data_loader.get_num_samples()
        num_samples = min(num_samples, total_samples)

        # Create backend
        inference_backend = self._create_backend(backend, model_path, device, logger)

        # Create data loaders for both calibrated and miscalibrated versions
        # This avoids reinitializing the transform for each sample
        data_loader_miscalibrated = CalibrationDataLoader(
            info_pkl_path=data_loader.info_pkl_path,
            model_cfg=data_loader.model_cfg,
            miscalibration_probability=1.0,
            device=device,
        )
        data_loader_calibrated = data_loader  # Already calibrated (prob=0.0)

        # Run inference on all samples
        predictions = []
        ground_truths = []
        probabilities = []
        latencies = []

        with inference_backend:
            for idx in range(num_samples):
                if idx % LOG_INTERVAL == 0:
                    logger.info(f"Processing sample {idx + 1}/{num_samples}")

                try:
                    # Process both calibrated and miscalibrated versions
                    pred, gt, prob, lat = self._process_sample(
                        idx, data_loader_calibrated, data_loader_miscalibrated, inference_backend, verbose, logger
                    )

                    predictions.extend(pred)
                    ground_truths.extend(gt)
                    probabilities.extend(prob)
                    latencies.extend(lat)

                    # Clear GPU memory periodically for TensorRT
                    if backend == "tensorrt" and idx % GPU_CLEANUP_INTERVAL == 0:
                        self._clear_gpu_memory()

                except Exception as e:
                    logger.error(f"Error processing sample {idx}: {e}")
                    continue

        # Convert to numpy arrays
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        probabilities = np.array(probabilities)
        latencies = np.array(latencies)

        # Compute metrics
        results = self._compute_metrics(predictions, ground_truths, probabilities, latencies)
        results["backend"] = backend
        results["num_samples"] = len(predictions)

        return results

    def _create_backend(
        self,
        backend: str,
        model_path: str,
        device: str,
        logger: logging.Logger,
    ):
        """Create appropriate backend instance."""
        if backend == "pytorch":
            # Load PyTorch model
            logger.info(f"Loading PyTorch model from {model_path}")
            model = get_model(self.model_cfg, model_path, device=device)
            return PyTorchBackend(model, device=device)
        elif backend == "onnx":
            logger.info(f"Loading ONNX model from {model_path}")
            return ONNXBackend(model_path, device=device)
        elif backend == "tensorrt":
            logger.info(f"Loading TensorRT engine from {model_path}")
            return TensorRTBackend(model_path, device="cuda")
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _process_sample(
        self,
        sample_idx: int,
        data_loader_calibrated: CalibrationDataLoader,
        data_loader_miscalibrated: CalibrationDataLoader,
        backend,
        verbose: bool,
        logger: logging.Logger,
    ):
        """
        Process a single sample with both calibrated and miscalibrated versions.

        Args:
            sample_idx: Index of sample to process
            data_loader_calibrated: DataLoader for calibrated samples
            data_loader_miscalibrated: DataLoader for miscalibrated samples
            backend: Inference backend
            verbose: Verbose logging
            logger: Logger instance

        Returns:
            Tuple of (predictions, ground_truths, probabilities, latencies)
        """
        predictions = []
        ground_truths = []
        probabilities = []
        latencies = []

        # Process both miscalibrated (0) and calibrated (1) versions
        for gt_label, loader in [(0, data_loader_miscalibrated), (1, data_loader_calibrated)]:
            # Load and preprocess using pre-created data loader
            input_tensor = loader.load_and_preprocess(sample_idx)

            # Run inference
            output, latency = backend.infer(input_tensor)

            # Get prediction
            if output.shape[-1] == 2:  # Binary classification
                predicted_label = int(np.argmax(output[0]))
                prob_scores = output[0]
            else:
                raise ValueError(f"Unexpected output shape: {output.shape}")

            predictions.append(predicted_label)
            ground_truths.append(gt_label)
            probabilities.append(prob_scores)
            latencies.append(latency)

            if verbose:
                logger.info(
                    f"  Sample {sample_idx}, GT: {LABELS[str(gt_label)]}, "
                    f"Pred: {LABELS[str(predicted_label)]}, "
                    f"Scores: {prob_scores}, Latency: {latency:.2f}ms"
                )

        return predictions, ground_truths, probabilities, latencies

    def _compute_metrics(
        self,
        predictions: np.ndarray,
        ground_truths: np.ndarray,
        probabilities: np.ndarray,
        latencies: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute all evaluation metrics."""
        if len(predictions) == 0:
            return {"accuracy": 0.0, "error": "No samples were processed successfully"}

        # Overall accuracy
        correct = (predictions == ground_truths).sum()
        accuracy = correct / len(predictions)

        # Per-class metrics
        per_class_acc = {}
        per_class_count = {}
        for cls in np.unique(ground_truths):
            mask = ground_truths == cls
            cls_correct = (predictions[mask] == ground_truths[mask]).sum()
            cls_total = mask.sum()
            per_class_acc[int(cls)] = cls_correct / cls_total if cls_total > 0 else 0.0
            per_class_count[int(cls)] = int(cls_total)

        # Confusion matrix
        num_classes = len(np.unique(ground_truths))
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for gt, pred in zip(ground_truths, predictions):
            confusion_matrix[int(gt), int(pred)] += 1

        # Latency statistics
        latency_stats = self.compute_latency_stats(latencies.tolist())

        return {
            "accuracy": float(accuracy),
            "correct_predictions": int(correct),
            "total_samples": len(predictions),
            "per_class_accuracy": per_class_acc,
            "per_class_count": per_class_count,
            "confusion_matrix": confusion_matrix.tolist(),
            "latency_stats": latency_stats,
        }

    def print_results(self, results: Dict[str, Any]) -> None:
        """
        Pretty print evaluation results.

        Args:
            results: Results dictionary from evaluate()
        """
        logger = logging.getLogger(__name__)

        if "error" in results:
            logger.error(f"Evaluation error: {results['error']}")
            return

        backend = results.get("backend", "unknown")

        logger.info(f"\n{'='*70}")
        logger.info(f"{backend.upper()} Model Evaluation Results")
        logger.info(f"{'='*70}")

        # Overall metrics
        logger.info(f"Total samples: {results['total_samples']}")
        logger.info(f"Correct predictions: {results['correct_predictions']}")
        logger.info(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")

        # Latency
        logger.info(f"\n{self.format_latency_stats(results['latency_stats'])}")

        # Per-class accuracy
        logger.info(f"\nPer-class accuracy:")
        for cls, acc in results["per_class_accuracy"].items():
            count = results["per_class_count"][cls]
            label = LABELS[str(cls)]
            logger.info(f"  Class {cls} ({label}): {acc:.4f} ({acc*100:.2f}%) - {count} samples")

        # Confusion matrix
        logger.info(f"\nConfusion Matrix:")
        cm = np.array(results["confusion_matrix"])
        logger.info(f"  Predicted →")
        logger.info(f"  GT ↓        {' '.join([f'{i:>8}' for i in range(len(cm))])}")
        for i, row in enumerate(cm):
            logger.info(f"  {i:>8}    {' '.join([f'{val:>8}' for val in row])}")

        logger.info(f"{'='*70}\n")

    def _clear_gpu_memory(self) -> None:
        """Clear GPU cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def get_models_to_evaluate(eval_cfg: Dict[str, Any], logger: logging.Logger) -> list:
    """
    Get list of models to evaluate from config.

    Args:
        eval_cfg: Evaluation configuration
        logger: Logger instance

    Returns:
        List of tuples (backend_name, model_path)
    """
    models_config = eval_cfg.get("models", {})
    models_to_evaluate = []

    backend_mapping = {
        "pytorch": "pytorch",
        "onnx": "onnx",
        "tensorrt": "tensorrt",
    }

    for backend_key, model_path in models_config.items():
        backend_name = backend_mapping.get(backend_key.lower())
        if backend_name and model_path:
            import os

            if os.path.exists(model_path):
                models_to_evaluate.append((backend_name, model_path))
                logger.info(f"  - {backend_name}: {model_path}")
            else:
                logger.warning(f"  - {backend_name}: {model_path} (not found, skipping)")

    return models_to_evaluate


def run_full_evaluation(
    models_to_evaluate: list,
    model_cfg: Config,
    info_pkl: str,
    device: str,
    num_samples: int,
    verbose: bool,
    logger: logging.Logger,
) -> None:
    """
    Run full evaluation on all specified models.

    Args:
        models_to_evaluate: List of (backend, model_path) tuples
        model_cfg: Model configuration
        info_pkl: Path to info.pkl file
        device: Device for inference
        num_samples: Number of samples to evaluate
        verbose: Verbose mode
        logger: Logger instance
    """
    if not models_to_evaluate:
        logger.warning("No models specified for evaluation")
        return

    logger.info(f"\nModels to evaluate:")
    for backend, path in models_to_evaluate:
        logger.info(f"  - {backend}: {path}")

    # Create evaluator
    evaluator = ClassificationEvaluator(model_cfg)

    # Create data loader (with miscalibration_probability=0.0 as default)
    data_loader = CalibrationDataLoader(
        info_pkl_path=info_pkl,
        model_cfg=model_cfg,
        miscalibration_probability=0.0,
        device=device,
    )

    # Evaluate each model
    all_results = {}
    for backend, model_path in models_to_evaluate:
        try:
            results = evaluator.evaluate(
                model_path=model_path,
                data_loader=data_loader,
                num_samples=num_samples,
                backend=backend,
                device=device,
                verbose=verbose,
            )
            all_results[backend] = results
            evaluator.print_results(results)
        except Exception as e:
            logger.error(f"Failed to evaluate {backend} model: {e}")
            import traceback

            logger.error(traceback.format_exc())

    # Print comparison summary if multiple models
    if len(all_results) > 1:
        logger.info(f"\n{'='*70}")
        logger.info("Comparison Summary")
        logger.info(f"{'='*70}")
        logger.info(f"{'Backend':<15} {'Accuracy':<12} {'Avg Latency (ms)':<20}")
        logger.info(f"{'-'*70}")
        for backend, results in all_results.items():
            if "error" not in results:
                acc = results["accuracy"]
                lat = results["latency_stats"]["mean_ms"]
                logger.info(f"{backend:<15} {acc:<12.4f} {lat:<20.2f}")
        logger.info(f"{'='*70}\n")
