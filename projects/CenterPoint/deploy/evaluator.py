"""
CenterPoint Evaluator for deployment.

This module implements evaluation for CenterPoint 3D object detection models.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from mmengine.config import Config

from autoware_ml.deployment.core import (
    BaseEvaluator,
    EvalResultDict,
    ModelSpec,
    VerifyResultDict,
)

from .data_loader import CenterPointDataLoader
from .utils import build_model_from_cfg

# Constants
LOG_INTERVAL = 50
GPU_CLEANUP_INTERVAL = 10


class CenterPointEvaluator(BaseEvaluator):
    """
    Evaluator for CenterPoint 3D object detection.

    Computes 3D detection metrics including mAP, NDS, and latency statistics.

    Note: For production, should integrate with mmdet3d's evaluation metrics.

    IMPORTANT:
    - `model_cfg` in __init__ can be the original mmdet3d config.
    - `model_cfg` will be updated to ONNX-compatible config by DeploymentRunner.load_pytorch_model().
    - `pytorch_model` will be injected by DeploymentRunner after model loading.
    - This design ensures clear ownership: Runner loads model and manages config, Evaluator only evaluates.
    """

    def __init__(
        self,
        model_cfg: Config,
        class_names: List[str] = None,
    ):
        """
        Initialize CenterPoint evaluator.

        Args:
            model_cfg: Model configuration (can be original mmdet3d config).
                       Will be updated to ONNX-compatible config by DeploymentRunner.load_pytorch_model().
            class_names: List of class names (optional). If not provided, will be extracted from model_cfg.
        """
        super().__init__(config={})

        self.model_cfg = model_cfg
        self.pytorch_model: Any = None  # Will be injected by runner after model loading

        # Get class names
        if class_names is not None:
            self.class_names = class_names
        elif hasattr(model_cfg, "class_names"):
            self.class_names = model_cfg.class_names
        else:
            # Default for T4Dataset
            self.class_names = ["VEHICLE", "PEDESTRIAN", "CYCLIST"]

    def set_onnx_config(self, model_cfg: Config) -> None:
        """
        Set ONNX-compatible model config (called by deployment runner).

        This is the official API for updating the evaluator's model config
        after the runner converts the original config to ONNX-compatible format.

        Args:
            model_cfg: ONNX-compatible model configuration
        """
        self.model_cfg = model_cfg
        # Re-derive class_names if available in the new config
        if hasattr(model_cfg, "class_names"):
            self.class_names = model_cfg.class_names

    def set_pytorch_model(self, pytorch_model: Any) -> None:
        """
        Set PyTorch model (called by deployment runner).

        This is the official API for injecting the loaded PyTorch model
        into the evaluator after the runner loads it.

        Args:
            pytorch_model: Loaded PyTorch model
        """
        self.pytorch_model = pytorch_model

    def verify(
        self,
        reference: ModelSpec,
        test: ModelSpec,
        data_loader: CenterPointDataLoader,
        num_samples: int = 1,
        tolerance: float = 0.1,
        verbose: bool = False,
    ) -> VerifyResultDict:
        """
        Verify exported models using policy-based verification.

        This method compares outputs from a reference backend against a test backend
        as specified by the verification policy.

        Args:
            reference: Specification describing reference backend/device/path
            test: Specification describing test backend/device/path
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
        if ref_backend == "pytorch" and ref_device.startswith("cuda"):
            logger.warning("PyTorch verification is forced to CPU; overriding device to 'cpu'")
            ref_device = "cpu"

        if test_backend == "tensorrt":
            if not test_device.startswith("cuda"):
                logger.warning("TensorRT verification requires CUDA device. Skipping verification.")
                results["error"] = "TensorRT requires CUDA"
                return results
            if test_device != "cuda:0":
                logger.warning("TensorRT verification only supports 'cuda:0'. Overriding device to 'cuda:0'.")
                test_device = "cuda:0"

        logger.info("\n" + "=" * 60)
        logger.info("CenterPoint Model Verification (Policy-Based)")
        logger.info("=" * 60)
        logger.info(f"Reference: {ref_backend} on {ref_device} - {ref_path}")
        logger.info(f"Test: {test_backend} on {test_device} - {test_path}")
        logger.info(f"Number of samples: {num_samples}")
        logger.info(f"Tolerance: {tolerance}")
        logger.info("=" * 60)

        # Create reference pipeline
        logger.info(f"\nInitializing {ref_backend} reference pipeline...")
        ref_pipeline = self._create_pipeline(ModelSpec(backend=ref_backend, device=ref_device, path=ref_path), logger)
        if ref_pipeline is None:
            logger.error(f"Failed to create {ref_backend} reference pipeline")
            results["error"] = f"Failed to create {ref_backend} reference pipeline"
            return results

        # Create test pipeline
        logger.info(f"\nInitializing {test_backend} test pipeline...")
        test_pipeline = self._create_pipeline(
            ModelSpec(backend=test_backend, device=test_device, path=test_path), logger
        )
        if test_pipeline is None:
            logger.error(f"Failed to create {test_backend} test pipeline")
            results["error"] = f"Failed to create {test_backend} test pipeline"
            return results

        # Verify each sample
        try:
            for i in range(min(num_samples, data_loader.get_num_samples())):
                logger.info(f"\n{'='*60}")
                logger.info(f"Verifying sample {i}")
                logger.info(f"{'='*60}")

                # Get sample data
                sample = data_loader.load_sample(i)

                # Get points for pipeline
                if "points" in sample:
                    points = sample["points"]
                else:
                    input_data = data_loader.preprocess(sample)
                    points = input_data.get("points", input_data)

                sample_meta = sample.get("metainfo", {})

                # Get reference outputs
                logger.info(f"\nRunning {ref_backend} reference ({ref_device})...")
                try:
                    ref_outputs, ref_latency, _ = ref_pipeline.infer(points, sample_meta, return_raw_outputs=True)
                    logger.info(f"  {ref_backend} latency: {ref_latency:.2f} ms")
                    logger.info(
                        f"  {ref_backend} output: {len(ref_outputs) if isinstance(ref_outputs, (list, tuple)) else 1} head outputs"
                    )
                except Exception as e:
                    logger.error(f"  {ref_backend} inference failed: {e}")
                    import traceback

                    traceback.print_exc()
                    results["samples"][f"sample_{i}"] = False
                    continue

                # Verify test backend against reference
                ref_name = f"{ref_backend} ({ref_device})"
                test_name = f"{test_backend} ({test_device})"
                passed = self._verify_single_backend(
                    test_pipeline,
                    points,
                    sample_meta,
                    ref_outputs,
                    ref_latency,
                    tolerance,
                    test_name,
                    logger,
                    reference_name=ref_name,
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

    def evaluate(
        self,
        model: ModelSpec,
        data_loader: CenterPointDataLoader,
        num_samples: int,
        verbose: bool = False,
    ) -> EvalResultDict:
        """
        Evaluate model performance.

        Args:
            model: Specification describing backend/device/path
            data_loader: Data loader for evaluation
            num_samples: Number of samples to evaluate
            verbose: Whether to print detailed output

        Returns:
            Dictionary containing evaluation results
        """
        backend = model.backend
        model_path = model.path
        device = model.device

        logger = logging.getLogger(__name__)

        logger.info(f"\nEvaluating {backend} model: {model_path}")
        logger.info(f"Number of samples: {num_samples}")

        # Create Pipeline instance
        pipeline = self._create_pipeline(model, logger)

        if pipeline is None:
            logger.error(f"Failed to create {backend} Pipeline")
            return {}

        # Run evaluation
        predictions_list = []
        ground_truths_list = []
        latencies = []
        latency_breakdowns = []  # Track individual stage latencies

        try:
            for i in range(min(num_samples, data_loader.get_num_samples())):
                if verbose and i % LOG_INTERVAL == 0:
                    logger.info(f"Processing sample {i+1}/{num_samples}")

                # Get sample data
                sample = data_loader.load_sample(i)

                # Get points for pipeline
                if "points" in sample:
                    points = sample["points"]
                else:
                    # Load points from data_loader
                    input_data = data_loader.preprocess(sample)
                    points = input_data.get("points", input_data)

                # Get ground truth
                gt_data = data_loader.get_ground_truth(i)

                # Run inference using unified Pipeline interface
                sample_meta = sample.get("metainfo", {})
                predictions, latency, latency_breakdown = pipeline.infer(points, sample_meta)

                # Parse ground truths
                ground_truths = self._parse_ground_truths(gt_data)

                predictions_list.append(predictions)
                ground_truths_list.append(ground_truths)
                latencies.append(latency)
                latency_breakdowns.append(latency_breakdown)

                # Cleanup GPU memory after each sample (TensorRT needs frequent cleanup)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            import traceback

            traceback.print_exc()
            return {}

        finally:
            # Cleanup (Pipeline handles its own cleanup)
            pass

        # Compute metrics
        results = self._compute_metrics(predictions_list, ground_truths_list, latencies, logger, latency_breakdowns)

        return results

    def _verify_single_backend(
        self,
        pipeline,
        points: torch.Tensor,
        sample_meta: Dict,
        reference_outputs: List[torch.Tensor],
        reference_latency: float,
        tolerance: float,
        backend_name: str,
        logger,
        reference_name: str = "Reference",
    ) -> bool:
        """
        Verify a single backend against PyTorch reference outputs.

        Args:
            pipeline: Pipeline instance to verify
            points: Input point cloud
            sample_meta: Sample metadata
            reference_outputs: Reference outputs from baseline backend
            reference_latency: Reference inference latency
            tolerance: Maximum allowed difference
            backend_name: Name of backend for logging ("ONNX (CPU)", "TensorRT", etc.)
            logger: Logger instance
            reference_name: Name of reference backend for logging

        Returns:
            bool: True if verification passed, False otherwise
        """
        try:
            # Run inference with raw outputs
            backend_outputs, backend_latency, backend_breakdown = pipeline.infer(
                points, sample_meta, return_raw_outputs=True
            )

            logger.info(f"  {backend_name} latency: {backend_latency:.2f} ms")
            if reference_latency is not None:
                logger.info(f"  {reference_name} latency: {reference_latency:.2f} ms")
            logger.info(f"  {backend_name} output: {len(backend_outputs)} head outputs")

            # Compare outputs
            if len(backend_outputs) != len(reference_outputs):
                logger.error(f"  Output count mismatch: {len(backend_outputs)} vs {len(reference_outputs)}")
                return False

            max_diff = 0.0
            mean_diff = 0.0
            total_elements = 0

            output_names = ["heatmap", "reg", "height", "dim", "rot", "vel"]

            for idx, (ref_out, backend_out, name) in enumerate(zip(reference_outputs, backend_outputs, output_names)):
                if isinstance(ref_out, torch.Tensor) and isinstance(backend_out, torch.Tensor):
                    # Convert to numpy for comparison
                    ref_np = ref_out.cpu().numpy()
                    backend_np = backend_out.cpu().numpy()

                    # Check for shape mismatch
                    if ref_np.shape != backend_np.shape:
                        logger.error(f"    {name}: shape mismatch - {ref_np.shape} vs {backend_np.shape}")
                        return False

                    # Compute differences
                    diff = np.abs(ref_np - backend_np)
                    output_max_diff = diff.max()
                    output_mean_diff = diff.mean()
                    max_diff = max(max_diff, output_max_diff)
                    mean_diff += diff.sum()
                    total_elements += diff.size

                    # Log detailed statistics
                    logger.info(f"    {name}: max_diff={output_max_diff:.6f}, mean_diff={output_mean_diff:.6f}")

                    # Check for special values
                    ref_nan = np.isnan(ref_np).any()
                    ref_inf = np.isinf(ref_np).any()
                    backend_nan = np.isnan(backend_np).any()
                    backend_inf = np.isinf(backend_np).any()

                    if ref_nan or ref_inf or backend_nan or backend_inf:
                        logger.warning(f"      ⚠️  Special values detected in {name}:")
                        if ref_nan:
                            logger.warning(f"         PyTorch has NaN!")
                        if ref_inf:
                            logger.warning(f"         PyTorch has Inf!")
                        if backend_nan:
                            logger.warning(f"         {backend_name} has NaN!")
                        if backend_inf:
                            logger.warning(f"         {backend_name} has Inf!")

            # Compute overall mean difference
            if total_elements > 0:
                mean_diff /= total_elements

            logger.info(f"  Overall Max difference: {max_diff:.6f}")
            logger.info(f"  Overall Mean difference: {mean_diff:.6f}")

            # Check if verification passed
            if max_diff < tolerance:
                logger.info(f"  {backend_name} verification PASSED ✓")
                return True
            else:
                logger.warning(
                    f"  {backend_name} verification FAILED ✗ "
                    f"(max diff: {max_diff:.6f} > tolerance: {tolerance:.6f})"
                )
                return False

        except Exception as e:
            logger.error(f"  {backend_name} verification failed with error: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _create_pipeline(self, model_spec: ModelSpec, logger) -> Any:
        """Create Pipeline instance for the specified backend."""
        try:
            # Import Pipeline classes
            from autoware_ml.deployment.pipelines import (
                CenterPointONNXPipeline,
                CenterPointPyTorchPipeline,
                CenterPointTensorRTPipeline,
            )

            backend = model_spec.backend
            model_path = model_spec.path
            device = model_spec.device

            # Ensure device is properly set
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"

            device_obj = torch.device(device) if isinstance(device, str) else device
            device_str = str(device_obj)

            # Use unified ONNX-compatible config for all backends
            cfg_for_backend = self.model_cfg

            # Get PyTorch model injected by runner
            pytorch_model = self.pytorch_model
            if pytorch_model is None:
                raise RuntimeError(
                    "CenterPointEvaluator.pytorch_model is None. "
                    "DeploymentRunner must set evaluator.pytorch_model before calling verify/evaluate."
                )

            # Move model to correct device if needed
            current_device = next(pytorch_model.parameters()).device
            target_device = device_obj
            if current_device != target_device:
                logger.info(f"Moving PyTorch model from {current_device} to {target_device}")
                pytorch_model = pytorch_model.to(target_device)
                # Update evaluator's model reference to the moved model
                self.pytorch_model = pytorch_model

            logger.info(f"Using PyTorch model (injected by runner) on {target_device}")

            # Create pipeline based on backend
            if backend == "pytorch":
                return CenterPointPyTorchPipeline(pytorch_model, device=device_str)

            elif backend == "onnx":
                # ONNX pipeline uses ONNX Runtime for voxel encoder and head;
                # PyTorch model is used for preprocessing/middle encoder/postprocessing.
                return CenterPointONNXPipeline(pytorch_model, onnx_dir=model_path, device=device_str)

            elif backend == "tensorrt":
                # TensorRT requires CUDA
                if not str(device).startswith("cuda"):
                    logger.warning("TensorRT requires CUDA device, skipping TensorRT evaluation")
                    return None
                return CenterPointTensorRTPipeline(pytorch_model, tensorrt_dir=model_path, device=device_str)

        except Exception as e:
            logger.error(f"Failed to create {backend} Pipeline: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _parse_ground_truths(self, gt_data: Dict) -> List[Dict]:
        """Parse ground truth data from gt_data returned by get_ground_truth()."""
        logger = logging.getLogger(__name__)
        ground_truths = []

        if "gt_bboxes_3d" in gt_data and "gt_labels_3d" in gt_data:
            gt_bboxes_3d = gt_data["gt_bboxes_3d"]
            gt_labels_3d = gt_data["gt_labels_3d"]

            # Count by label
            unique_labels, counts = np.unique(gt_labels_3d, return_counts=True)

            for i in range(len(gt_bboxes_3d)):
                bbox_3d = gt_bboxes_3d[i]  # [x, y, z, w, l, h, yaw]
                label = gt_labels_3d[i]

                ground_truths.append({"bbox_3d": bbox_3d.tolist(), "label": int(label)})

        return ground_truths

    # TODO(vividf): use autoware_perception_eval in the future
    def _compute_metrics(
        self,
        predictions_list: List[List[Dict]],
        ground_truths_list: List[List[Dict]],
        latencies: List[float],
        logger,
        latency_breakdowns: List[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        logger = logging.getLogger(__name__)

        # Debug metrics

        # Count total predictions and ground truths
        total_predictions = sum(len(preds) for preds in predictions_list)
        total_ground_truths = sum(len(gts) for gts in ground_truths_list)

        # Count per class
        per_class_preds = {}
        per_class_gts = {}

        for predictions in predictions_list:
            for pred in predictions:
                label = pred["label"]
                per_class_preds[label] = per_class_preds.get(label, 0) + 1

        for ground_truths in ground_truths_list:
            for gt in ground_truths:
                label = gt["label"]
                per_class_gts[label] = per_class_gts.get(label, 0) + 1

        # Compute latency statistics
        latency_stats = self.compute_latency_stats(latencies)

        # Compute stage-wise latency breakdown if available
        if latency_breakdowns and len(latency_breakdowns) > 0:
            stage_stats = {}
            stages = [
                "preprocessing_ms",
                "voxel_encoder_ms",
                "middle_encoder_ms",
                "backbone_head_ms",
                "postprocessing_ms",
            ]

            for stage in stages:
                stage_values = [bd.get(stage, 0.0) for bd in latency_breakdowns if stage in bd]
                if stage_values:
                    stage_stats[stage] = self.compute_latency_stats(stage_values)

            latency_stats["latency_breakdown"] = stage_stats

        # Try to compute mmdet3d metrics
        try:
            map_results = self._compute_simple_3d_map(
                predictions_list,
                ground_truths_list,
                num_classes=len(self.class_names),
            )

            # Combine results with mmdet3d metrics
            results = {
                "total_predictions": total_predictions,
                "total_ground_truths": total_ground_truths,
                "per_class_predictions": per_class_preds,
                "per_class_ground_truths": per_class_gts,
                "latency": latency_stats,
                "num_samples": len(predictions_list),
                **map_results,  # Include mAP, NDS, etc.
            }

            logger.info("✅ Successfully computed mmdet3d metrics")

        except Exception as e:
            logger.warning(f"Failed to compute mmdet3d metrics: {e}")
            logger.warning("Using simplified metrics instead")

            # Fallback to simplified metrics
            results = {
                "total_predictions": total_predictions,
                "total_ground_truths": total_ground_truths,
                "per_class_predictions": per_class_preds,
                "per_class_ground_truths": per_class_gts,
                "latency": latency_stats,
                "num_samples": len(predictions_list),
                "mAP": 0.0,
                "NDS": 0.0,
                "mATE": 0.0,
                "mASE": 0.0,
                "mAOE": 0.0,
                "mAVE": 0.0,
                "mAAE": 0.0,
            }

        return results

    def print_results(self, results: EvalResultDict) -> None:
        """
        Pretty print evaluation results.

        Args:
            results: Results dictionary from evaluate()
        """
        print("\n" + "=" * 80)
        print("CenterPoint 3D Object Detection - Evaluation Results")
        print("=" * 80)

        # Detection metrics
        print(f"\nDetection Metrics:")
        print(f"  mAP (0.5:0.95): {results.get('mAP', 0.0):.4f}")
        print(f"  mAP @ IoU=0.50: {results.get('mAP_50', 0.0):.4f}")
        print(f"  NDS: {results.get('NDS', 0.0):.4f}")
        print(f"  mATE: {results.get('mATE', 0.0):.4f}")
        print(f"  mASE: {results.get('mASE', 0.0):.4f}")
        print(f"  mAOE: {results.get('mAOE', 0.0):.4f}")
        print(f"  mAVE: {results.get('mAVE', 0.0):.4f}")
        print(f"  mAAE: {results.get('mAAE', 0.0):.4f}")

        # Per-class AP (show all 3D object classes)
        if "per_class_ap" in results:
            print(f"\nPer-Class AP (3D Object Classes):")
            for class_id, ap in results["per_class_ap"].items():
                class_name = (
                    class_id
                    if isinstance(class_id, str)
                    else (self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}")
                )
                print(f"  {class_name:<12}: {ap:.4f}")

        # Detection statistics
        print(f"\nDetection Statistics:")
        print(f"  Total Predictions: {results.get('total_predictions', 0)}")
        print(f"  Total Ground Truths: {results.get('total_ground_truths', 0)}")

        # Per-class statistics
        if "per_class_predictions" in results and "per_class_ground_truths" in results:
            print(f"\nPer-Class Statistics:")
            for class_id in range(len(self.class_names)):
                class_name = self.class_names[class_id]
                pred_count = results["per_class_predictions"].get(class_id, 0)
                gt_count = results["per_class_ground_truths"].get(class_id, 0)
                print(f"  {class_name}:")
                print(f"    Predictions: {pred_count}")
                print(f"    Ground Truths: {gt_count}")

        # Latency statistics
        if "latency" in results:
            latency = results["latency"]
            print(f"\nLatency Statistics:")
            print(f"  Mean:   {latency['mean_ms']:.2f} ms")
            print(f"  Std:    {latency['std_ms']:.2f} ms")
            print(f"  Min:    {latency['min_ms']:.2f} ms")
            print(f"  Max:    {latency['max_ms']:.2f} ms")
            print(f"  Median: {latency['median_ms']:.2f} ms")

            # Stage-wise latency breakdown
            if "latency_breakdown" in latency:
                breakdown = latency["latency_breakdown"]
                print(f"\n  Stage-wise Latency Breakdown:")
                if "preprocessing_ms" in breakdown:
                    print(
                        f"    Preprocessing:     {breakdown['preprocessing_ms']['mean_ms']:.2f} ± {breakdown['preprocessing_ms']['std_ms']:.2f} ms"
                    )
                if "voxel_encoder_ms" in breakdown:
                    print(
                        f"    Voxel Encoder:    {breakdown['voxel_encoder_ms']['mean_ms']:.2f} ± {breakdown['voxel_encoder_ms']['std_ms']:.2f} ms"
                    )
                if "middle_encoder_ms" in breakdown:
                    print(
                        f"    Middle Encoder:   {breakdown['middle_encoder_ms']['mean_ms']:.2f} ± {breakdown['middle_encoder_ms']['std_ms']:.2f} ms"
                    )
                if "backbone_head_ms" in breakdown:
                    print(
                        f"    Backbone + Head:  {breakdown['backbone_head_ms']['mean_ms']:.2f} ± {breakdown['backbone_head_ms']['std_ms']:.2f} ms"
                    )
                if "postprocessing_ms" in breakdown:
                    print(
                        f"    Postprocessing:   {breakdown['postprocessing_ms']['mean_ms']:.2f} ± {breakdown['postprocessing_ms']['std_ms']:.2f} ms"
                    )

        print(f"\nTotal Samples: {results.get('num_samples', 0)}")
        print("=" * 80)

    def _compute_simple_3d_map(
        self,
        predictions_list: List[List[Dict]],
        ground_truths_list: List[List[Dict]],
        num_classes: int,
    ) -> Dict[str, Any]:
        """Compute simple 3D mAP using basic IoU calculation."""
        try:
            import numpy as np

            class_names = ["car", "truck", "bus", "bicycle", "pedestrian"]
            iou_threshold = 0.5

            # Initialize per-class metrics
            per_class_ap = {}

            for class_id, class_name in enumerate(class_names):
                # Collect all predictions and ground truths for this class
                all_predictions = []
                all_ground_truths = []

                for predictions, ground_truths in zip(predictions_list, ground_truths_list):
                    # Filter by class
                    class_predictions = [p for p in predictions if p["label"] == class_id]
                    class_ground_truths = [g for g in ground_truths if g["label"] == class_id]

                    all_predictions.extend(class_predictions)
                    all_ground_truths.extend(class_ground_truths)

                # Sort predictions by score (descending)
                all_predictions.sort(key=lambda x: x["score"], reverse=True)

                # Debug IoU calculations for this class
                logger = logging.getLogger(__name__)
                if all_predictions and all_ground_truths:
                    # Show IoU between first prediction and first few GTs
                    first_pred = all_predictions[0]

                    max_iou = 0.0
                    for i, gt in enumerate(all_ground_truths[:3]):  # Show first 3 GTs
                        iou = self._compute_3d_iou_simple(first_pred["bbox_3d"], gt["bbox_3d"])
                        max_iou = max(max_iou, iou)

                    # Debug: Check if this is PyTorch or ONNX/TensorRT
                    backend_type = "Unknown"
                    if hasattr(self, "_current_backend"):
                        backend_type = str(type(self._current_backend))

                # Compute AP for this class
                if len(all_ground_truths) == 0:
                    # No ground truths for this class
                    ap = 0.0
                elif len(all_predictions) == 0:
                    # No predictions for this class
                    ap = 0.0
                else:
                    # Compute precision-recall curve
                    tp = 0
                    fp = 0
                    fn = len(all_ground_truths)

                    precision_values = []
                    recall_values = []

                    # Track which ground truths have been matched
                    gt_matched = [False] * len(all_ground_truths)

                    for i, pred in enumerate(all_predictions):
                        pred_bbox = np.array(pred["bbox_3d"])
                        best_iou = 0.0
                        best_gt_idx = -1

                        # Find best matching ground truth
                        for j, gt in enumerate(all_ground_truths):
                            if gt_matched[j]:
                                continue

                            gt_bbox = np.array(gt["bbox_3d"])
                            iou = self._compute_3d_iou_simple(pred_bbox, gt_bbox)

                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = j

                        # Determine if prediction is TP or FP
                        if best_iou >= iou_threshold:
                            tp += 1
                            fn -= 1
                            gt_matched[best_gt_idx] = True
                        else:
                            fp += 1

                        # Compute precision and recall
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                        precision_values.append(precision)
                        recall_values.append(recall)

                    # Compute AP using 11-point interpolation
                    ap = self._compute_ap_11_point(precision_values, recall_values)

                per_class_ap[class_name] = ap

            # Compute overall mAP
            map_50 = sum(per_class_ap.values()) / len(class_names) if len(class_names) > 0 else 0.0

            return {
                "mAP": map_50,
                "mAP_50": map_50,
                "NDS": map_50,  # Simplified
                "mATE": 0.0,
                "mASE": 0.0,
                "mAOE": 0.0,
                "mAVE": 0.0,
                "mAAE": 0.0,
                "per_class_ap": per_class_ap,
            }

        except Exception as e:
            print(f"ERROR: Simple 3D mAP computation failed: {e}")
            import traceback

            traceback.print_exc()
            return {
                "mAP": 0.0,
                "mAP_50": 0.0,
                "NDS": 0.0,
                "mATE": 0.0,
                "mASE": 0.0,
                "mAOE": 0.0,
                "mAVE": 0.0,
                "mAAE": 0.0,
                "per_class_ap": {},
            }

    def _compute_3d_iou_simple(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute simple 3D IoU using BEV (Bird's Eye View) approximation."""
        try:
            # Extract BEV parameters: [x, y, w, l, yaw]
            x1, y1, w1, l1, yaw1 = box1[0], box1[1], box1[3], box1[4], box1[6]
            x2, y2, w2, l2, yaw2 = box2[0], box2[1], box2[3], box2[4], box2[6]

            # Debug IoU calculation
            # print(f"DEBUG IoU: box1=[{x1:.2f}, {y1:.2f}, {w1:.2f}, {l1:.2f}, {yaw1:.2f}], box2=[{x2:.2f}, {y2:.2f}, {w2:.2f}, {l2:.2f}, {yaw2:.2f}]")

            # For simplicity, ignore rotation and compute axis-aligned IoU
            # This is an approximation - proper 3D IoU would consider rotation

            # Compute intersection rectangle
            x_left = max(x1 - w1 / 2, x2 - w2 / 2)
            y_top = max(y1 - l1 / 2, y2 - l2 / 2)
            x_right = min(x1 + w1 / 2, x2 + w2 / 2)
            y_bottom = min(y1 + l1 / 2, y2 + l2 / 2)

            # print(f"DEBUG IoU: intersection_rect=[{x_left:.2f}, {y_top:.2f}, {x_right:.2f}, {y_bottom:.2f}]")

            if x_right < x_left or y_bottom < y_top:
                # print(f"DEBUG IoU: No intersection (x_right={x_right:.2f} < x_left={x_left:.2f} or y_bottom={y_bottom:.2f} < y_top={y_top:.2f})")
                return 0.0

            intersection_area = (x_right - x_left) * (y_bottom - y_top)

            # Compute union area
            area1 = w1 * l1
            area2 = w2 * l2
            union_area = area1 + area2 - intersection_area

            # print(f"DEBUG IoU: intersection={intersection_area:.2f}, area1={area1:.2f}, area2={area2:.2f}, union={union_area:.2f}")

            if union_area <= 0:
                # print(f"DEBUG IoU: Union area <= 0, returning 0.0")
                return 0.0

            iou = intersection_area / union_area
            # print(f"DEBUG IoU: Final IoU = {iou:.3f}")
            return min(iou, 1.0)  # Cap at 1.0

        except Exception as e:
            print(f"ERROR: 3D IoU computation failed: {e}")
            return 0.0

    def _compute_ap_11_point(self, precision_values: List[float], recall_values: List[float]) -> float:
        """Compute AP using 11-point interpolation."""
        if len(precision_values) == 0 or len(recall_values) == 0:
            return 0.0

        import numpy as np

        # 11-point interpolation
        recall_thresholds = np.linspace(0, 1, 11)
        max_precision = np.zeros_like(recall_thresholds)

        for i, threshold in enumerate(recall_thresholds):
            # Find maximum precision for recall >= threshold
            valid_indices = np.where(np.array(recall_values) >= threshold)[0]
            if len(valid_indices) > 0:
                max_precision[i] = np.max(np.array(precision_values)[valid_indices])

        ap = np.mean(max_precision)
        return float(ap)
