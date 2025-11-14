"""
Classification Evaluator for CalibrationStatusClassification.

This module implements the BaseEvaluator interface for evaluating
calibration status classification models.
"""

import gc
import logging
from typing import Any, Dict, Optional

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
            sample = loader.load_sample(sample_idx)
            input_tensor = loader.preprocess(sample)

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
        ref_backend: str,
        ref_device: str,
        ref_path: str,
        test_backend: str,
        test_device: str,
        test_path: str,
        data_loader: CalibrationDataLoader,
        num_samples: int = 1,
        tolerance: float = 0.1,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Verify exported models using policy-based verification.
        
        This method compares outputs from a reference backend against a test backend
        as specified by the verification policy.
        
        Args:
            ref_backend: Reference backend name ('pytorch' or 'onnx')
            ref_device: Device for reference backend (e.g., 'cpu', 'cuda:0')
            ref_path: Path to reference model (checkpoint for pytorch, model path for onnx)
            test_backend: Test backend name ('onnx' or 'tensorrt')
            test_device: Device for test backend (e.g., 'cpu', 'cuda:0')
            test_path: Path to test model (model path for onnx, engine path for tensorrt)
            data_loader: Data loader for test samples
            num_samples: Number of samples to verify
            tolerance: Maximum allowed difference for verification to pass
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary containing verification results:
            {
                'sample_0': bool (passed/failed),
                'sample_1': bool (passed/failed),
                ...
                'summary': {'passed': int, 'failed': int, 'total': int}
            }
        """
        from autoware_ml.deployment.pipelines.calibration import (
            CalibrationPyTorchPipeline,
            CalibrationONNXPipeline,
            CalibrationTensorRTPipeline,
        )
        
        logger = logging.getLogger(__name__)
        
        # Enforce device scenarios
        if ref_backend == "pytorch" and ref_device.startswith("cuda"):
            logger.warning("PyTorch verification is forced to CPU; overriding device to 'cpu'")
            ref_device = "cpu"
        
        if test_backend == "tensorrt":
            if not test_device.startswith("cuda"):
                logger.warning("TensorRT verification requires CUDA device. Skipping verification.")
                return {"error": "TensorRT requires CUDA"}
            if test_device != "cuda:0":
                logger.warning("TensorRT verification only supports 'cuda:0'. Overriding device to 'cuda:0'.")
                test_device = "cuda:0"
        
        logger.info("\n" + "=" * 60)
        logger.info("CalibrationStatusClassification Model Verification (Policy-Based)")
        logger.info("=" * 60)
        logger.info(f"Reference: {ref_backend} on {ref_device} - {ref_path}")
        logger.info(f"Test: {test_backend} on {test_device} - {test_path}")
        logger.info(f"Number of samples: {num_samples}")
        logger.info(f"Tolerance: {tolerance}")
        logger.info("=" * 60)
        
        # Create reference pipeline
        logger.info(f"\nInitializing {ref_backend} reference pipeline...")
        if ref_backend == "pytorch":
            pytorch_model = get_model(self.model_cfg, ref_path, device=ref_device)
            pytorch_model.eval()
            ref_pipeline = CalibrationPyTorchPipeline(
                pytorch_model=pytorch_model,
                device=ref_device,
                num_classes=2,
                class_names=["miscalibrated", "calibrated"]
            )
        elif ref_backend == "onnx":
            ref_pipeline = CalibrationONNXPipeline(
                onnx_path=ref_path,
                device=ref_device,
                num_classes=2,
                class_names=["miscalibrated", "calibrated"]
            )
        else:
            logger.error(f"Unsupported reference backend: {ref_backend}")
            return {"error": f"Unsupported reference backend: {ref_backend}"}
        
        if ref_pipeline is None:
            logger.error(f"Failed to create {ref_backend} reference pipeline")
            return {"error": f"Failed to create {ref_backend} reference pipeline"}
        
        # Create test pipeline
        logger.info(f"\nInitializing {test_backend} test pipeline...")
        if test_backend == "onnx":
            test_pipeline = CalibrationONNXPipeline(
                onnx_path=test_path,
                device=test_device,
                num_classes=2,
                class_names=["miscalibrated", "calibrated"]
            )
        elif test_backend == "tensorrt":
            test_pipeline = CalibrationTensorRTPipeline(
                engine_path=test_path,
                device=test_device,
                num_classes=2,
                class_names=["miscalibrated", "calibrated"]
            )
        else:
            logger.error(f"Unsupported test backend: {test_backend}")
            return {"error": f"Unsupported test backend: {test_backend}"}
        
        if test_pipeline is None:
            logger.error(f"Failed to create {test_backend} test pipeline")
            return {"error": f"Failed to create {test_backend} test pipeline"}
        
        # Verify each sample
        results = {}
        
        try:
            for i in range(min(num_samples, data_loader.get_num_samples())):
                logger.info(f"\n{'='*60}")
                logger.info(f"Verifying sample {i}")
                logger.info(f"{'='*60}")
                
                # Load sample and preprocess
                sample = data_loader.load_sample(i)
                input_tensor = data_loader.preprocess(sample)
                
                # Ensure input tensor is on the correct device for reference backend
                # data_loader may have a different device, so we need to move the tensor
                ref_device_obj = torch.device(ref_device)
                if input_tensor.device != ref_device_obj:
                    input_tensor = input_tensor.to(ref_device_obj)
                
                # Get reference outputs
                logger.info(f"\nRunning {ref_backend} reference ({ref_device})...")
                try:
                    ref_output, ref_latency, _ = ref_pipeline.infer(input_tensor, return_raw_outputs=True)
                    logger.info(f"  {ref_backend} latency: {ref_latency:.2f} ms")
                    logger.info(f"  {ref_backend} output shape: {ref_output.shape}")
                except Exception as e:
                    logger.error(f"  {ref_backend} inference failed: {e}")
                    import traceback
                    traceback.print_exc()
                    results[f"sample_{i}"] = False
                    continue
                
                # Ensure input tensor is on the correct device for test backend
                test_device_obj = torch.device(test_device)
                test_input_tensor = input_tensor.to(test_device_obj) if input_tensor.device != test_device_obj else input_tensor
                
                # Verify test backend against reference
                ref_name = f"{ref_backend} ({ref_device})"
                test_name = f"{test_backend} ({test_device})"
                passed = self._verify_single_backend(
                    test_pipeline,
                    test_input_tensor,
                    ref_output,
                    ref_latency,
                    tolerance,
                    test_name,
                    logger
                )
                results[f"sample_{i}"] = passed
                
                # Cleanup GPU memory after each sample
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
        
        # Compute summary
        # Use == instead of 'is' to handle numpy bool values
        passed = sum(1 for v in results.values() if v == True)
        failed = sum(1 for v in results.values() if v == False)
        total = len(results)
        
        results['summary'] = {
            'passed': passed,
            'failed': failed,
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
        
        logger.info("=" * 60)
        logger.info(f"Total: {passed}/{total} passed, {failed}/{total} failed")
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
            passed = bool(max_diff <= tolerance)  # Convert numpy bool to Python bool
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
