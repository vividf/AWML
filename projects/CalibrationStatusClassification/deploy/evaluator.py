"""
Classification Evaluator for CalibrationStatusClassification.

This module implements the BaseEvaluator interface for evaluating
calibration status classification models.
Uses ClassificationMetricsAdapter for consistent metric computation
with autoware_perception_evaluation formulas.
"""

import gc
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from mmengine.config import Config

from deployment.core import (
    Backend,
    BaseEvaluator,
    ClassificationMetricsAdapter,
    ClassificationMetricsConfig,
    EvalResultDict,
    ModelSpec,
    VerifyResultDict,
)

from .data_loader import CalibrationDataLoader

# Label mapping
LABELS = {"0": "miscalibrated", "1": "calibrated"}
CLASS_NAMES = ["miscalibrated", "calibrated"]

# Constants for evaluation
LOG_INTERVAL = 100  # Log progress every N samples
GPU_CLEANUP_INTERVAL = 10  # Clear GPU memory every N samples for TensorRT


class ClassificationEvaluator(BaseEvaluator):
    """
    Evaluator for classification tasks.

    Computes accuracy, per-class metrics, confusion matrix, and latency statistics
    using ClassificationMetricsAdapter for consistent metrics with
    autoware_perception_evaluation formulas.
    """

    def __init__(
        self,
        model_cfg: Config,
        class_names: Optional[List[str]] = None,
        metrics_config: Optional[ClassificationMetricsConfig] = None,
    ):
        """
        Initialize classification evaluator.

        Args:
            model_cfg: Model configuration
            class_names: List of class names. Defaults to ["miscalibrated", "calibrated"].
            metrics_config: Optional configuration for the classification metrics adapter.
                           If not provided, will use default configuration.

        IMPORTANT:
        - `pytorch_model` will be injected by DeploymentRunner after model loading.
        - This design ensures clear ownership: Runner loads model, Evaluator only evaluates.
        """
        super().__init__(config={})
        self.model_cfg = model_cfg
        self.pytorch_model: Any = None  # Will be injected by runner after model loading

        # Set class names
        self.class_names = class_names if class_names is not None else CLASS_NAMES

        # Initialize classification metrics adapter for consistent evaluation
        if metrics_config is None:
            metrics_config = ClassificationMetricsConfig(class_names=self.class_names)
        self.metrics_adapter = ClassificationMetricsAdapter(metrics_config)

    def set_pytorch_model(self, pytorch_model: Any) -> None:
        """
        Set PyTorch model (called by deployment runner).

        This is the official API for injecting the loaded PyTorch model
        into the evaluator after the runner loads it.

        Args:
            pytorch_model: Loaded PyTorch model
        """
        self.pytorch_model = pytorch_model

    def evaluate(
        self,
        model: ModelSpec,
        data_loader: CalibrationDataLoader,
        num_samples: int,
        verbose: bool = False,
    ) -> EvalResultDict:
        """
        Run full evaluation on a model using unified pipeline architecture.

        Args:
            model: Backend/device/path specification for the artifact to evaluate
            data_loader: DataLoader for loading samples (calibrated version)
            num_samples: Number of samples to evaluate
            verbose: Whether to print detailed progress

        Returns:
            Dictionary containing evaluation metrics
        """
        from deployment.pipelines.calibration import (
            CalibrationONNXPipeline,
            CalibrationPyTorchPipeline,
            CalibrationTensorRTPipeline,
        )

        logger = logging.getLogger(__name__)
        backend = model.backend
        logger.info(f"\nEvaluating {backend.upper()} model: {model.path}")
        logger.info(f"Number of samples: {num_samples}")

        # Limit num_samples to available data
        total_samples = data_loader.get_num_samples()
        num_samples = min(num_samples, total_samples)

        # Create pipeline instead of generic backend
        pipeline = self._create_pipeline(model, logger)

        # Create data loaders for both calibrated and miscalibrated versions
        data_loader_miscalibrated = CalibrationDataLoader(
            info_pkl_path=data_loader.info_pkl_path,
            model_cfg=data_loader.model_cfg,
            miscalibration_probability=1.0,
            device=model.device,
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
                if backend is Backend.TENSORRT and idx % GPU_CLEANUP_INTERVAL == 0:
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
        results["backend"] = backend.value
        results["num_samples"] = len(predictions)

        return results

    def _create_pipeline(
        self,
        model_spec: ModelSpec,
        logger: logging.Logger,
    ):
        """Create appropriate pipeline instance."""
        from deployment.pipelines.calibration import (
            CalibrationONNXPipeline,
            CalibrationPyTorchPipeline,
            CalibrationTensorRTPipeline,
        )

        backend = model_spec.backend
        device = model_spec.device
        model_path = model_spec.path

        if backend is Backend.PYTORCH:
            # Use PyTorch model injected by runner
            model = self.pytorch_model
            if model is None:
                raise RuntimeError(
                    "ClassificationEvaluator.pytorch_model is None. "
                    "DeploymentRunner must set evaluator.pytorch_model before calling evaluate/verify."
                )

            # Move model to correct device if needed
            current_device = next(model.parameters()).device
            target_device = torch.device(device)
            if current_device != target_device:
                logger.info(f"Moving PyTorch model from {current_device} to {target_device}")
                model = model.to(target_device)
                self.pytorch_model = model

            return CalibrationPyTorchPipeline(
                pytorch_model=model, device=device, num_classes=2, class_names=["miscalibrated", "calibrated"]
            )
        elif backend is Backend.ONNX:
            logger.info(f"Loading ONNX model from {model_path}")
            return CalibrationONNXPipeline(
                onnx_path=model_path, device=device, num_classes=2, class_names=["miscalibrated", "calibrated"]
            )
        elif backend is Backend.TENSORRT:
            logger.info(f"Loading TensorRT engine from {model_path}")
            return CalibrationTensorRTPipeline(
                engine_path=model_path, device=device, num_classes=2, class_names=["miscalibrated", "calibrated"]
            )
        else:
            raise ValueError(f"Unsupported backend: {backend.value}")

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
            gt_label = gt_data["gt_label"]

            # Load and preprocess using pre-created data loader
            sample = loader.load_sample(sample_idx)
            input_tensor = loader.preprocess(sample)

            # Run inference via pipeline
            result, latency, _ = pipeline.infer(input_tensor)

            # Extract prediction from result dict
            predicted_label = result["class_id"]
            prob_scores = result["probabilities"]

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
        """
        Compute all evaluation metrics using ClassificationMetricsAdapter.

        This ensures consistent metrics with autoware_perception_evaluation formulas.
        """
        logger = logging.getLogger(__name__)

        if len(predictions) == 0:
            return {"accuracy": 0.0, "error": "No samples were processed successfully"}

        # Use ClassificationMetricsAdapter for consistent metrics
        try:
            # Reset adapter for new evaluation
            self.metrics_adapter.reset()

            # Add all samples to the adapter
            for pred, gt, prob in zip(predictions, ground_truths, probabilities):
                prob_list = prob.tolist() if isinstance(prob, np.ndarray) else list(prob)
                self.metrics_adapter.add_frame(
                    prediction=int(pred),
                    ground_truth=int(gt),
                    probabilities=prob_list,
                )

            # Compute metrics using the adapter
            adapter_metrics = self.metrics_adapter.compute_metrics()

            # Get summary and confusion matrix
            summary = self.metrics_adapter.get_summary()
            confusion_matrix = self.metrics_adapter.get_confusion_matrix()

            # Compute latency statistics
            latency_stats = self.compute_latency_stats(latencies.tolist())

            # Build results dict with adapter metrics
            results = {
                "accuracy": adapter_metrics.get("accuracy", 0.0),
                "precision": adapter_metrics.get("precision", 0.0),
                "recall": adapter_metrics.get("recall", 0.0),
                "f1score": adapter_metrics.get("f1score", 0.0),
                "correct_predictions": adapter_metrics.get("correct_predictions", 0),
                "total_samples": adapter_metrics.get("total_samples", len(predictions)),
                "per_class_accuracy": summary.get("per_class_accuracy", {}),
                "per_class_count": {
                    i: int(adapter_metrics.get(f"{self.class_names[i]}_num_gt", 0))
                    for i in range(len(self.class_names))
                },
                "confusion_matrix": confusion_matrix.tolist(),
                "latency_stats": latency_stats,
                "detailed_metrics": adapter_metrics,
            }

            logger.info("✅ Successfully computed metrics using ClassificationMetricsAdapter")
            logger.info(f"   Accuracy: {results['accuracy']:.4f}, F1: {results['f1score']:.4f}")

            return results

        except Exception as e:
            logger.warning(f"Failed to compute metrics using ClassificationMetricsAdapter: {e}")
            import traceback

            traceback.print_exc()

            # Fallback to basic computation
            correct = (predictions == ground_truths).sum()
            accuracy = correct / len(predictions)
            latency_stats = self.compute_latency_stats(latencies.tolist())

            return {
                "accuracy": float(accuracy),
                "correct_predictions": int(correct),
                "total_samples": len(predictions),
                "per_class_accuracy": {},
                "per_class_count": {},
                "confusion_matrix": [],
                "latency_stats": latency_stats,
            }

    def print_results(self, results: EvalResultDict) -> None:
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
        logger.info("(Using ClassificationMetricsAdapter for consistent metrics)")
        logger.info(f"{'='*70}")

        # Overall metrics (from autoware_perception_evaluation formulas)
        logger.info(f"\nClassification Metrics:")
        logger.info(f"  Total samples: {results['total_samples']}")
        logger.info(f"  Correct predictions: {results['correct_predictions']}")
        logger.info(f"  Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        logger.info(f"  Precision: {results.get('precision', 0.0):.4f}")
        logger.info(f"  Recall: {results.get('recall', 0.0):.4f}")
        logger.info(f"  F1 Score: {results.get('f1score', 0.0):.4f}")

        # Latency
        logger.info(f"\n{self.format_latency_stats(results['latency_stats'])}")

        # Per-class accuracy
        if results.get("per_class_accuracy"):
            logger.info(f"\nPer-class accuracy:")
            for cls_name, acc in results["per_class_accuracy"].items():
                # Handle both class name (str) and class index (int) keys
                if isinstance(cls_name, int):
                    label = LABELS.get(str(cls_name), f"class_{cls_name}")
                    count = results.get("per_class_count", {}).get(cls_name, 0)
                else:
                    label = cls_name
                    count = results.get("per_class_count", {}).get(
                        self.class_names.index(cls_name) if cls_name in self.class_names else 0, 0
                    )
                logger.info(f"  {label}: {acc:.4f} ({acc*100:.2f}%) - {count} samples")

        # Confusion matrix
        if results.get("confusion_matrix"):
            logger.info(f"\nConfusion Matrix:")
            cm = np.array(results["confusion_matrix"])
            if cm.size > 0:
                logger.info(f"  Predicted →")
                logger.info(f"  GT ↓        {' '.join([f'{i:>8}' for i in range(len(cm))])}")
                for i, row in enumerate(cm):
                    logger.info(f"  {i:>8}    {' '.join([f'{val:>8}' for val in row])}")

        logger.info(f"{'='*70}\n")

    def verify(
        self,
        reference: ModelSpec,
        test: ModelSpec,
        data_loader: CalibrationDataLoader,
        num_samples: int = 1,
        tolerance: float = 0.1,
        verbose: bool = False,
    ) -> VerifyResultDict:
        """
        Verify exported models using policy-based verification.

        This method compares outputs from a reference backend against a test backend
        as specified by the verification policy.

        Args:
            reference: Specification for reference backend/device/path
            test: Specification for test backend/device/path
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
        from deployment.pipelines.calibration import (
            CalibrationONNXPipeline,
            CalibrationPyTorchPipeline,
            CalibrationTensorRTPipeline,
        )

        logger = logging.getLogger(__name__)
        ref_backend = reference.backend
        ref_device = reference.device
        ref_path = reference.path
        test_backend = test.backend
        test_device = test.device
        test_path = test.path

        results: VerifyResultDict = {
            "summary": {"passed": 0, "failed": 0, "total": 0},
            "samples": {},
        }

        # Enforce device scenarios
        if ref_backend is Backend.PYTORCH and ref_device.startswith("cuda"):
            logger.warning("PyTorch verification is forced to CPU; overriding device to 'cpu'")
            ref_device = "cpu"

        if test_backend is Backend.TENSORRT:
            if not test_device.startswith("cuda"):
                logger.warning("TensorRT verification requires CUDA device. Skipping verification.")
                results["error"] = "TensorRT requires CUDA"
                return results
            if test_device != "cuda:0":
                logger.warning("TensorRT verification only supports 'cuda:0'. Overriding device to 'cuda:0'.")
                test_device = "cuda:0"

        logger.info("\n" + "=" * 60)
        logger.info("CalibrationStatusClassification Model Verification (Policy-Based)")
        logger.info("=" * 60)
        logger.info(f"Reference: {ref_backend.value} on {ref_device} - {ref_path}")
        logger.info(f"Test: {test_backend.value} on {test_device} - {test_path}")
        logger.info(f"Number of samples: {num_samples}")
        logger.info(f"Tolerance: {tolerance}")
        logger.info("=" * 60)

        # Create reference pipeline
        logger.info(f"\nInitializing {ref_backend.value} reference pipeline...")
        if ref_backend is Backend.PYTORCH:
            # Use PyTorch model injected by runner
            pytorch_model = self.pytorch_model
            if pytorch_model is None:
                raise RuntimeError(
                    "ClassificationEvaluator.pytorch_model is None. "
                    "DeploymentRunner must set evaluator.pytorch_model before calling verify."
                )

            # Move model to correct device if needed
            current_device = next(pytorch_model.parameters()).device
            target_device = torch.device(ref_device)
            if current_device != target_device:
                logger.info(f"Moving PyTorch model from {current_device} to {target_device}")
                pytorch_model = pytorch_model.to(target_device)
                self.pytorch_model = pytorch_model

            ref_pipeline = CalibrationPyTorchPipeline(
                pytorch_model=pytorch_model,
                device=ref_device,
                num_classes=2,
                class_names=["miscalibrated", "calibrated"],
            )
        elif ref_backend is Backend.ONNX:
            ref_pipeline = CalibrationONNXPipeline(
                onnx_path=ref_path, device=ref_device, num_classes=2, class_names=["miscalibrated", "calibrated"]
            )
        else:
            logger.error(f"Unsupported reference backend: {ref_backend.value}")
            results["error"] = f"Unsupported reference backend: {ref_backend.value}"
            return results

        if ref_pipeline is None:
            logger.error(f"Failed to create {ref_backend.value} reference pipeline")
            results["error"] = f"Failed to create {ref_backend.value} reference pipeline"
            return results

        # Create test pipeline
        logger.info(f"\nInitializing {test_backend.value} test pipeline...")
        if test_backend is Backend.ONNX:
            test_pipeline = CalibrationONNXPipeline(
                onnx_path=test_path, device=test_device, num_classes=2, class_names=["miscalibrated", "calibrated"]
            )
        elif test_backend is Backend.TENSORRT:
            test_pipeline = CalibrationTensorRTPipeline(
                engine_path=test_path, device=test_device, num_classes=2, class_names=["miscalibrated", "calibrated"]
            )
        else:
            logger.error(f"Unsupported test backend: {test_backend.value}")
            results["error"] = f"Unsupported test backend: {test_backend.value}"
            return results

        if test_pipeline is None:
            logger.error(f"Failed to create {test_backend.value} test pipeline")
            results["error"] = f"Failed to create {test_backend.value} test pipeline"
            return results

        # Verify each sample
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
                logger.info(f"\nRunning {ref_backend.value} reference ({ref_device})...")
                try:
                    ref_output, ref_latency, _ = ref_pipeline.infer(input_tensor, return_raw_outputs=True)
                    logger.info(f"  {ref_backend.value} latency: {ref_latency:.2f} ms")
                    logger.info(f"  {ref_backend.value} output shape: {ref_output.shape}")
                except Exception as e:
                    logger.error(f"  {ref_backend.value} inference failed: {e}")
                    import traceback

                    traceback.print_exc()
                    results["samples"][f"sample_{i}"] = False
                    continue

                # Ensure input tensor is on the correct device for test backend
                test_device_obj = torch.device(test_device)
                test_input_tensor = (
                    input_tensor.to(test_device_obj) if input_tensor.device != test_device_obj else input_tensor
                )

                # Verify test backend against reference
                ref_name = f"{ref_backend.value} ({ref_device})"
                test_name = f"{test_backend.value} ({test_device})"
                passed = self._verify_single_backend(
                    test_pipeline, test_input_tensor, ref_output, ref_latency, tolerance, test_name, logger
                )
                results["samples"][f"sample_{i}"] = passed

                # Cleanup GPU memory after each sample
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error during verification: {e}")
            import traceback

            traceback.print_exc()
            results["error"] = str(e)
            return results

        # Compute summary
        # Use == instead of 'is' to handle numpy bool values
        sample_values = results["samples"].values()
        passed = sum(1 for v in sample_values if v == True)
        failed = sum(1 for v in sample_values if v == False)
        total = len(results["samples"])

        results["summary"] = {"passed": passed, "failed": failed, "total": total}

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Verification Summary")
        logger.info("=" * 60)
        for key, value in results["samples"].items():
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
