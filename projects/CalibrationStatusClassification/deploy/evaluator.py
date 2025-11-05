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
        Run full evaluation on a model using unified pipeline architecture.

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
        from autoware_ml.deployment.pipelines.calibration import (
            CalibrationPyTorchPipeline,
            CalibrationONNXPipeline,
            CalibrationTensorRTPipeline,
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"\nEvaluating {backend.upper()} model: {model_path}")
        logger.info(f"Number of samples: {num_samples}")

        # Limit num_samples to available data
        total_samples = data_loader.get_num_samples()
        num_samples = min(num_samples, total_samples)

        # Create pipeline instead of generic backend
        pipeline = self._create_pipeline(backend, model_path, device, logger)

        # Create data loaders for both calibrated and miscalibrated versions
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

        for idx in range(num_samples):
            if idx % LOG_INTERVAL == 0:
                logger.info(f"Processing sample {idx + 1}/{num_samples}")

            try:
                # Process both calibrated and miscalibrated versions
                pred, gt, prob, lat = self._process_sample_with_pipeline(
                    idx, data_loader_calibrated, data_loader_miscalibrated, pipeline, verbose, logger
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

    def _create_pipeline(
        self,
        backend: str,
        model_path: str,
        device: str,
        logger: logging.Logger,
    ):
        """Create appropriate pipeline instance."""
        from autoware_ml.deployment.pipelines.calibration import (
            CalibrationPyTorchPipeline,
            CalibrationONNXPipeline,
            CalibrationTensorRTPipeline,
        )
        
        if backend == "pytorch":
            logger.info(f"Loading PyTorch model from {model_path}")
            model = get_model(self.model_cfg, model_path, device=device)
            return CalibrationPyTorchPipeline(
                pytorch_model=model,
                device=device,
                num_classes=2,
                class_names=["miscalibrated", "calibrated"]
            )
        elif backend == "onnx":
            logger.info(f"Loading ONNX model from {model_path}")
            return CalibrationONNXPipeline(
                onnx_path=model_path,
                device=device,
                num_classes=2,
                class_names=["miscalibrated", "calibrated"]
            )
        elif backend == "tensorrt":
            logger.info(f"Loading TensorRT engine from {model_path}")
            return CalibrationTensorRTPipeline(
                engine_path=model_path,
                device=device,
                num_classes=2,
                class_names=["miscalibrated", "calibrated"]
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _process_sample_with_pipeline(
        self,
        sample_idx: int,
        data_loader_calibrated: CalibrationDataLoader,
        data_loader_miscalibrated: CalibrationDataLoader,
        pipeline,
        verbose: bool,
        logger: logging.Logger,
    ):
        """
        Process a single sample with both calibrated and miscalibrated versions using pipeline.

        Args:
            sample_idx: Index of sample to process
            data_loader_calibrated: DataLoader for calibrated samples
            data_loader_miscalibrated: DataLoader for miscalibrated samples
            pipeline: Pipeline instance for inference
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
        for loader in [data_loader_miscalibrated, data_loader_calibrated]:
            # Get ground truth using get_ground_truth method
            gt_data = loader.get_ground_truth(sample_idx)
            gt_label = gt_data['gt_label']
            
            # Load and preprocess using pre-created data loader
            input_tensor = loader.load_and_preprocess(sample_idx)

            # Run inference via pipeline
            result, latency, _ = pipeline.infer(input_tensor)
            
            # Extract prediction from result dict
            predicted_label = result['class_id']
            prob_scores = result['probabilities']

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

    def verify(
        self,
        pytorch_model_path: str,
        onnx_model_path: str = None,
        tensorrt_model_path: str = None,
        data_loader: CalibrationDataLoader = None,
        num_samples: int = 3,
        device: str = "cpu",
        tolerance: float = 0.1,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Verify exported models against PyTorch reference by comparing raw outputs.
        
        This method uses the unified pipeline architecture for verification.
        
        Args:
            pytorch_model_path: Path to PyTorch checkpoint (reference)
            onnx_model_path: Optional path to ONNX model file
            tensorrt_model_path: Optional path to TensorRT engine file
            data_loader: Data loader for test samples
            num_samples: Number of samples to verify
            device: Device to run verification on
            tolerance: Maximum allowed difference for verification to pass
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary containing verification results:
            {
                'sample_0_onnx': bool (passed/failed),
                'sample_0_tensorrt': bool (passed/failed),
                ...
                'summary': {'passed': int, 'failed': int, 'skipped': int, 'total': int}
            }
        """
        from autoware_ml.deployment.pipelines.calibration import (
            CalibrationPyTorchPipeline,
            CalibrationONNXPipeline,
            CalibrationTensorRTPipeline,
        )
        
        logger = logging.getLogger(__name__)
        
        logger.info("\n" + "=" * 60)
        logger.info("CalibrationStatusClassification Model Verification")
        logger.info("=" * 60)
        logger.info(f"PyTorch reference: {pytorch_model_path}")
        if onnx_model_path:
            logger.info(f"ONNX model: {onnx_model_path}")
        if tensorrt_model_path:
            logger.info(f"TensorRT model: {tensorrt_model_path}")
        logger.info(f"Number of samples: {num_samples}")
        logger.info(f"Tolerance: {tolerance}")
        logger.info("=" * 60)
        
        results = {}
        skipped_backends = []
        
        # Load PyTorch model and create pipeline (reference)
        logger.info("\nInitializing PyTorch reference pipeline...")
        pytorch_model = get_model(self.model_cfg, pytorch_model_path, device=device)
        pytorch_model.eval()
        pytorch_pipeline = CalibrationPyTorchPipeline(
            pytorch_model=pytorch_model,
            device=device,
            num_classes=2,
            class_names=["miscalibrated", "calibrated"]
        )
        
        # Create ONNX pipeline if requested
        onnx_pipeline = None
        if onnx_model_path:
            logger.info("\nInitializing ONNX pipeline...")
            try:
                onnx_pipeline = CalibrationONNXPipeline(
                    onnx_path=onnx_model_path,
                    device=device,
                    num_classes=2,
                    class_names=["miscalibrated", "calibrated"]
                )
            except Exception as e:
                logger.warning(f"Failed to create ONNX pipeline, skipping ONNX verification: {e}")
                skipped_backends.append("onnx")
        
        # Create TensorRT pipeline if requested
        tensorrt_pipeline = None
        if tensorrt_model_path:
            logger.info("\nInitializing TensorRT pipeline...")
            if not device.startswith("cuda"):
                logger.warning("TensorRT requires CUDA device, skipping TensorRT verification")
                skipped_backends.append("tensorrt")
            else:
                try:
                    tensorrt_pipeline = CalibrationTensorRTPipeline(
                        engine_path=tensorrt_model_path,
                        device=device,
                        num_classes=2,
                        class_names=["miscalibrated", "calibrated"]
                    )
                except Exception as e:
                    logger.warning(f"Failed to create TensorRT pipeline, skipping TensorRT verification: {e}")
                    skipped_backends.append("tensorrt")
        
        # Create data loaders for both calibrated and miscalibrated versions
        data_loader_miscalibrated = CalibrationDataLoader(
            info_pkl_path=data_loader.info_pkl_path,
            model_cfg=data_loader.model_cfg,
            miscalibration_probability=1.0,
            device=device,
        )
        data_loader_calibrated = CalibrationDataLoader(
            info_pkl_path=data_loader.info_pkl_path,
            model_cfg=data_loader.model_cfg,
            miscalibration_probability=0.0,
            device=device,
        )
        
        # Verify each sample
        try:
            num_samples_to_verify = min(num_samples, data_loader.get_num_samples())
            for i in range(num_samples_to_verify):
                logger.info(f"\n{'='*60}")
                logger.info(f"Verifying sample {i}")
                logger.info(f"{'='*60}")
                
                # Process both calibrated and miscalibrated versions
                for loader_name, loader in [("miscalibrated", data_loader_miscalibrated), ("calibrated", data_loader_calibrated)]:
                    # Load sample and preprocess
                    input_tensor = loader.load_and_preprocess(i)
                    
                    # Get PyTorch reference outputs (raw logits)
                    logger.info(f"\nRunning PyTorch reference ({loader_name})...")
                    try:
                        pytorch_output, pytorch_latency, _ = pytorch_pipeline.infer(input_tensor, return_raw_outputs=True)
                        logger.info(f"  PyTorch latency: {pytorch_latency:.2f} ms")
                        logger.info(f"  PyTorch output shape: {pytorch_output.shape}")
                        logger.info(f"  PyTorch output range: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")
                    except Exception as e:
                        logger.error(f"  PyTorch inference failed: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                    
                    # Verify ONNX
                    if onnx_pipeline:
                        logger.info(f"\nVerifying ONNX pipeline ({loader_name})...")
                        onnx_passed = self._verify_single_backend(
                            onnx_pipeline,
                            input_tensor,
                            pytorch_output,
                            pytorch_latency,
                            tolerance,
                            f"ONNX ({loader_name})",
                            logger
                        )
                        results[f"sample_{i}_{loader_name}_onnx"] = onnx_passed
                    
                    # Verify TensorRT
                    if tensorrt_pipeline:
                        logger.info(f"\nVerifying TensorRT pipeline ({loader_name})...")
                        tensorrt_passed = self._verify_single_backend(
                            tensorrt_pipeline,
                            input_tensor,
                            pytorch_output,
                            pytorch_latency,
                            tolerance,
                            f"TensorRT ({loader_name})",
                            logger
                        )
                        results[f"sample_{i}_{loader_name}_tensorrt"] = tensorrt_passed
                    
                    # Cleanup GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
        
        # Compute summary
        passed = sum(1 for v in results.values() if v)
        failed = sum(1 for v in results.values() if not v)
        total = len(results)
        skipped = len(skipped_backends) * num_samples * 2  # 2 versions per sample
        
        results['summary'] = {
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'total': total
        }
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Verification Summary")
        logger.info("=" * 60)
        for key, value in results.items():
            if key != 'summary':
                status = "✓ PASSED" if value else "✗ FAILED"
                logger.info(f"  {key}: {status}")
        
        if skipped_backends:
            logger.info("")
            for backend in skipped_backends:
                logger.info(f"  {backend}: ⊝ SKIPPED")
        
        logger.info("=" * 60)
        summary_parts = [f"{passed}/{total} passed"]
        if failed > 0:
            summary_parts.append(f"{failed}/{total} failed")
        if skipped > 0:
            summary_parts.append(f"{skipped} skipped")
        logger.info(f"Total: {', '.join(summary_parts)}")
        logger.info("=" * 60)
        
        return results
    
    def _verify_single_backend(
        self,
        pipeline,
        input_tensor: torch.Tensor,
        reference_output: np.ndarray,
        reference_latency: float,
        tolerance: float,
        backend_name: str,
        logger,
    ) -> bool:
        """
        Verify a single pipeline against PyTorch reference outputs.
        
        Args:
            pipeline: Pipeline instance to verify
            input_tensor: Preprocessed input tensor
            reference_output: Reference output from PyTorch (raw logits)
            reference_latency: Reference inference latency
            tolerance: Maximum allowed difference
            backend_name: Name of backend for logging
            logger: Logger instance
            
        Returns:
            bool: True if verification passed, False otherwise
        """
        try:
            # Run inference with raw outputs
            backend_output, backend_latency, _ = pipeline.infer(input_tensor, return_raw_outputs=True)
            
            logger.info(f"  {backend_name} latency: {backend_latency:.2f} ms")
            logger.info(f"  {backend_name} output shape: {backend_output.shape}")
            logger.info(f"  {backend_name} output range: [{backend_output.min():.6f}, {backend_output.max():.6f}]")
            
            # Convert outputs to numpy if needed
            if isinstance(backend_output, torch.Tensor):
                backend_output = backend_output.cpu().numpy()
            if isinstance(reference_output, torch.Tensor):
                reference_output = reference_output.cpu().numpy()
            
            # Ensure same shape
            if backend_output.shape != reference_output.shape:
                logger.error(f"  Shape mismatch: {backend_output.shape} vs {reference_output.shape}")
                return False
            
            # Compute difference
            diff = np.abs(backend_output - reference_output)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            logger.info(f"  Max difference: {max_diff:.6f}")
            logger.info(f"  Mean difference: {mean_diff:.6f}")
            
            # Check if within tolerance
            passed = max_diff <= tolerance
            if passed:
                logger.info(f"  ✓ Verification PASSED (max_diff={max_diff:.6f} <= tolerance={tolerance})")
            else:
                logger.warning(f"  ✗ Verification FAILED (max_diff={max_diff:.6f} > tolerance={tolerance})")
            
            return passed
            
        except Exception as e:
            logger.error(f"  Verification error: {e}")
            import traceback
            traceback.print_exc()
            return False

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
