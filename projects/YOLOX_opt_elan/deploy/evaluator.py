"""
YOLOX_opt_elan Evaluator for deployment.

This module implements evaluation for YOLOX_opt_elan object detection models.
Uses autoware_perception_evaluation via Detection2DMetricsAdapter for consistent
metric computation.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from mmengine.config import Config

from deployment.core import (
    Backend,
    BaseEvaluator,
    Detection2DMetricsAdapter,
    Detection2DMetricsConfig,
    EvalResultDict,
    ModelSpec,
    VerifyResultDict,
)

from .data_loader import YOLOXOptElanDataLoader

# Constants
LOG_INTERVAL = 50  # Log more frequently for smaller datasets
GPU_CLEANUP_INTERVAL = 10


def generate_yolox_priors(img_size=(960, 960)):
    """
    Generate YOLOX priors for bbox decoding.

    Args:
        img_size: (height, width) of input image

    Returns:
        priors: [num_anchors, 4] with [center_x, center_y, stride_w, stride_h]
    """
    priors = []

    # YOLOX uses 3 detection levels with strides [8, 16, 32]
    # YOLOX uses offset=0 (not 0.5 like other detectors)
    strides = [8, 16, 32]

    for stride in strides:
        # Calculate feature map size
        feat_h = img_size[0] // stride
        feat_w = img_size[1] // stride

        # Generate grid centers with offset=0 (YOLOX specific)
        for y in range(feat_h):
            for x in range(feat_w):
                center_x = x * stride  # offset=0, not 0.5
                center_y = y * stride  # offset=0, not 0.5
                priors.append([center_x, center_y, stride, stride])

    return np.array(priors, dtype=np.float32)


class YOLOXOptElanEvaluator(BaseEvaluator):
    """
    Evaluator for YOLOX_opt_elan object detection.

    Computes detection metrics including mAP, per-class AP, and latency statistics
    specifically for object detection task (8 classes).
    """

    def __init__(
        self,
        model_cfg: Config,
        model_cfg_path: str,
        class_names: List[str] = None,
        metrics_config: Optional[Detection2DMetricsConfig] = None,
    ):
        """
        Initialize YOLOX_opt_elan evaluator.

        Args:
            model_cfg: Model configuration
            model_cfg_path: Path to model config file
            class_names: List of class names (optional, will try to get from config)
            metrics_config: Optional configuration for the 2D detection metrics adapter.
                           If not provided, will use default configuration based on class_names.

        IMPORTANT:
        - `pytorch_model` will be injected by DeploymentRunner after model loading.
        - This design ensures clear ownership: Runner loads model, Evaluator only evaluates.
        """
        super().__init__(config={})
        self.model_cfg = model_cfg
        self.model_cfg_path = model_cfg_path
        self.pytorch_model: Any = None  # Will be injected by runner after model loading

        # Get class names from config
        if class_names is not None:
            self.class_names = class_names
        elif hasattr(model_cfg, "classes"):
            # Get classes from dataset config (from _base_)
            classes = model_cfg.classes
            if not isinstance(classes, (tuple, list)):
                raise ValueError(
                    f"Config 'classes' must be a tuple or list, got {type(classes)}. "
                    f"Please check your dataset config file."
                )
            self.class_names = list(classes)
        else:
            raise ValueError(
                "Config file must contain 'classes' attribute. "
                "Please ensure your dataset config file (referenced via _base_) defines 'classes'."
            )

        # Initialize 2D detection metrics adapter for consistent evaluation
        if metrics_config is None:
            metrics_config = Detection2DMetricsConfig(
                class_names=self.class_names,
                iou_thresholds=[0.5, 0.75],  # Pascal VOC style thresholds
            )
        self.metrics_adapter = Detection2DMetricsAdapter(metrics_config)

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
        data_loader: YOLOXOptElanDataLoader,
        num_samples: int = 1,
        tolerance: float = 0.1,
        verbose: bool = False,
    ) -> VerifyResultDict:
        """
        Verify exported models using policy-based verification.

        This method compares outputs from a reference backend against a test backend
        as specified by the verification policy.

        Args:
            reference: Specification describing the reference backend/device/path
            test: Specification describing the test backend/device/path
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
        from deployment.pipelines.yolox import (
            YOLOXONNXPipeline,
            YOLOXPyTorchPipeline,
            YOLOXTensorRTPipeline,
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
        logger.info("YOLOX_opt_elan Model Verification (Policy-Based)")
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
                    "YOLOXOptElanEvaluator.pytorch_model is None. "
                    "DeploymentRunner must set evaluator.pytorch_model before calling verify."
                )

            # Move model to correct device if needed
            current_device = next(pytorch_model.parameters()).device
            target_device = torch.device(ref_device)
            if current_device != target_device:
                logger.info(f"Moving PyTorch model from {current_device} to {target_device}")
                pytorch_model = pytorch_model.to(target_device)
                self.pytorch_model = pytorch_model

            ref_pipeline = YOLOXPyTorchPipeline(
                pytorch_model=pytorch_model,
                device=ref_device,
                num_classes=len(self.class_names),
                class_names=self.class_names,
            )
        elif ref_backend is Backend.ONNX:
            ref_pipeline = YOLOXONNXPipeline(
                onnx_path=ref_path, device=ref_device, num_classes=len(self.class_names), class_names=self.class_names
            )
        else:
            logger.error(f"Unsupported reference backend: {ref_backend}")
            results["error"] = f"Unsupported reference backend: {ref_backend}"
            return results

        if ref_pipeline is None:
            logger.error(f"Failed to create {ref_backend} reference pipeline")
            results["error"] = f"Failed to create {ref_backend} reference pipeline"
            return results

        # Create test pipeline
        logger.info(f"\nInitializing {test_backend.value} test pipeline...")
        if test_backend is Backend.ONNX:
            test_pipeline = YOLOXONNXPipeline(
                onnx_path=test_path,
                device=test_device,
                num_classes=len(self.class_names),
                class_names=self.class_names,
            )
        elif test_backend is Backend.TENSORRT:
            test_pipeline = YOLOXTensorRTPipeline(
                engine_path=test_path,
                device=test_device,
                num_classes=len(self.class_names),
                class_names=self.class_names,
            )
        else:
            logger.error(f"Unsupported test backend: {test_backend}")
            results["error"] = f"Unsupported test backend: {test_backend}"
            return results

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

                # Load sample and preprocess
                sample = data_loader.load_sample(i)
                input_tensor = data_loader.preprocess(sample)

                # Ensure input tensor is on the correct device for reference backend
                # data_loader may have a different device, so we need to move the tensor
                ref_device_obj = torch.device(ref_device)
                if input_tensor.device != ref_device_obj:
                    input_tensor = input_tensor.to(ref_device_obj)

                # Get ground truth and img_info (required for YOLOX preprocessing)
                gt_data = data_loader.get_ground_truth(i)
                img_info = gt_data["img_info"]

                # Get reference outputs
                logger.info(f"\nRunning {ref_backend} reference ({ref_device})...")
                try:
                    ref_output, ref_latency, _ = ref_pipeline.infer(
                        input_tensor, return_raw_outputs=True, img_info=img_info
                    )
                    logger.info(f"  {ref_backend} latency: {ref_latency:.2f} ms")
                    logger.info(f"  {ref_backend} output shape: {ref_output.shape}")
                except Exception as e:
                    logger.error(f"  {ref_backend} inference failed: {e}")
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
                    test_pipeline,
                    test_input_tensor,
                    ref_output,
                    ref_latency,
                    tolerance,
                    test_name,
                    logger,
                    img_info=img_info,
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
        img_info: Dict = None,
    ) -> bool:
        """
        Verify a single backend against PyTorch reference outputs.

        Args:
            pipeline: Pipeline instance to verify
            input_tensor: Preprocessed input tensor from MMDet pipeline
            reference_output: Reference output from PyTorch (raw, before postprocessing)
            reference_latency: Reference inference latency
            tolerance: Maximum allowed difference
            backend_name: Name of backend for logging ("ONNX", "TensorRT")
            logger: Logger instance
            img_info: Image metadata (required for YOLOX preprocessing)

        Returns:
            bool: True if verification passed, False otherwise
        """
        try:
            # Run inference with raw outputs (using MMDet preprocessed tensor)
            # img_info is required for YOLOX preprocessing
            if img_info is not None:
                backend_output, backend_latency, _ = pipeline.infer(
                    input_tensor, return_raw_outputs=True, img_info=img_info
                )
            else:
                return False

            logger.info(f"  {backend_name} latency: {backend_latency:.2f} ms")
            logger.info(f"  {backend_name} output shape: {backend_output.shape}")
            logger.info(f"  {backend_name} output range: [{backend_output.min():.6f}, {backend_output.max():.6f}]")

            # Compare outputs
            if reference_output.shape != backend_output.shape:
                logger.error(f"  Output shape mismatch: {reference_output.shape} vs {backend_output.shape}")
                return False

            # Compute differences
            diff = np.abs(reference_output - backend_output)
            max_diff = diff.max()
            mean_diff = diff.mean()

            logger.info(f"  Max difference: {max_diff:.6f}")
            logger.info(f"  Mean difference: {mean_diff:.6f}")

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

    def evaluate(
        self,
        model: ModelSpec,
        data_loader: YOLOXOptElanDataLoader,
        num_samples: int,
        verbose: bool = False,
    ) -> EvalResultDict:
        """
        Run full evaluation on YOLOX_opt_elan model.

        Args:
            model: Specification describing backend/device/path
            data_loader: YOLOX_opt_elan DataLoader
            num_samples: Number of samples to evaluate
            verbose: Whether to print detailed progress

        Returns:
            Dictionary containing evaluation metrics
        """
        backend = model.backend
        logger = logging.getLogger(__name__)
        logger.info(f"\nEvaluating {backend.upper()} model: {model.path}")
        logger.info(f"Number of samples: {num_samples}")

        # Limit num_samples
        total_samples = data_loader.get_num_samples()
        num_samples = min(num_samples, total_samples)

        # Create YOLOX Pipeline instead of generic backend
        pipeline = self._create_backend(model, logger)

        # Run inference via Pipeline on MMDet-preprocessed tensors
        all_predictions = []
        all_ground_truths = []
        latencies = []

        for idx in range(num_samples):
            if idx % LOG_INTERVAL == 0:
                logger.info(f"Processing sample {idx + 1}/{num_samples}")

            # Load and preprocess via MMDet test pipeline
            sample = data_loader.load_sample(idx)
            input_tensor = data_loader.preprocess(sample)

            # Get ground truth and img_info
            gt_data = data_loader.get_ground_truth(idx)

            # Run pipeline inference with preprocessed tensor (like CenterPoint)
            # pipeline.infer() automatically detects tensor input and uses img_info from kwargs
            detections, latency, _ = pipeline.infer(input_tensor, img_info=gt_data["img_info"])  # type: ignore
            latencies.append(latency)

            # Normalize detections to expected format for eval_map
            predictions = []
            for det in detections:
                if isinstance(det, dict) and "bbox" in det:
                    predictions.append(
                        {
                            "bbox": det["bbox"],
                            "label": int(det.get("class_id", det.get("label", 0))),
                            "score": float(det.get("score", 0.0)),
                        }
                    )
            all_predictions.append(predictions)

            # Parse ground truths
            ground_truths = self._parse_ground_truths(gt_data)
            all_ground_truths.append(ground_truths)

            if verbose:
                logger.info(f"  Sample {idx}: {len(predictions)} predictions, {len(ground_truths)} ground truths")

            # GPU cleanup for TensorRT
            if backend is Backend.TENSORRT and idx % GPU_CLEANUP_INTERVAL == 0:
                torch.cuda.empty_cache()

        # Compute metrics
        results = self._compute_metrics(all_predictions, all_ground_truths, latencies, logger)

        return results

    def _create_backend(self, model: ModelSpec, logger):
        """Create YOLOX Pipeline for the specified backend."""
        import os

        from deployment.pipelines.yolox import (
            YOLOXONNXPipeline,
            YOLOXPyTorchPipeline,
            YOLOXTensorRTPipeline,
        )

        # Determine classes from config
        # Get class names from config (from dataset config via _base_)
        if not hasattr(self.model_cfg, "classes"):
            raise ValueError(
                "Config file must contain 'classes' attribute. "
                "Please ensure your dataset config file (referenced via _base_) defines 'classes'."
            )

        classes = self.model_cfg.classes
        if not isinstance(classes, (tuple, list)):
            raise ValueError(
                f"Config 'classes' must be a tuple or list, got {type(classes)}. "
                f"Please check your dataset config file."
            )

        class_names = list(classes)

        # Get num_classes from model config or infer from class_names
        if hasattr(self.model_cfg, "model") and hasattr(self.model_cfg.model, "bbox_head"):
            num_classes = self.model_cfg.model.bbox_head.get("num_classes", len(class_names))
            # Validate consistency
            if num_classes != len(class_names):
                raise ValueError(
                    f"Number of classes mismatch: model.bbox_head.num_classes={num_classes} "
                    f"but config has {len(class_names)} classes. "
                    f"Please check your config file."
                )
        else:
            num_classes = len(class_names)

        backend = model.backend
        model_path = model.path
        device = model.device

        if backend is Backend.PYTORCH:
            # Use PyTorch model injected by runner
            model = self.pytorch_model
            if model is None:
                raise RuntimeError(
                    "YOLOXOptElanEvaluator.pytorch_model is None. "
                    "DeploymentRunner must set evaluator.pytorch_model before calling evaluate/verify."
                )

            # Move model to correct device if needed
            current_device = next(model.parameters()).device
            target_device = torch.device(device)
            if current_device != target_device:
                logger.info(f"Moving PyTorch model from {current_device} to {target_device}")
                model = model.to(target_device)
                self.pytorch_model = model

            return YOLOXPyTorchPipeline(
                pytorch_model=model,
                device=device,
                num_classes=num_classes,
                class_names=class_names,
            )
        elif backend is Backend.ONNX:
            if not (os.path.isfile(model_path) and model_path.endswith(".onnx")):
                raise ValueError("ONNX evaluation expects a .onnx file path")
            return YOLOXONNXPipeline(
                onnx_path=model_path,
                device=device,
                num_classes=num_classes,
                class_names=class_names,
            )
        elif backend is Backend.TENSORRT:
            if not (os.path.isfile(model_path) and (model_path.endswith(".engine") or model_path.endswith(".trt"))):
                raise ValueError("TensorRT evaluation expects an engine (.engine/.trt) file path")
            return YOLOXTensorRTPipeline(
                engine_path=model_path,
                device=device,
                num_classes=num_classes,
                class_names=class_names,
            )
        else:
            raise ValueError(f"Unknown backend: {backend.value}")

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

        # Handle different output formats
        if isinstance(output, dict):
            # Dict format
            bboxes = output.get("bboxes", np.array([]))
            scores = output.get("scores", np.array([]))
            labels = output.get("labels", np.array([]))
        else:
            # Array format - handle both 2D and 3D outputs
            if len(output.shape) == 2 and output.shape[1] == 6:
                # 2D format: [num_detections, 6] - Standard MMDetection PyTorch output
                # Format: [x1, y1, x2, y2, score, label]

                if len(output) == 0:
                    return predictions

                bboxes = output[:, :4]  # [x1, y1, x2, y2]
                scores = output[:, 4]  # scores
                labels = output[:, 5]  # labels

                # For 2D format, we already have final predictions, just need to filter by score
                score_thr = 0.3  # Default score threshold
                if len(scores) > 0:
                    valid_mask = scores >= score_thr
                    bboxes = bboxes[valid_mask]
                    scores = scores[valid_mask]
                    labels = labels[valid_mask]

                # Keep in [x1, y1, x2, y2] format for evaluation (MMDetection format)
                if len(bboxes) > 0:
                    for i in range(len(bboxes)):
                        predictions.append(
                            {
                                "bbox": [
                                    float(bboxes[i, 0]),
                                    float(bboxes[i, 1]),
                                    float(bboxes[i, 2]),
                                    float(bboxes[i, 3]),
                                ],
                                "label": int(labels[i]),
                                "score": float(scores[i]),
                            }
                        )

                return predictions

            elif len(output.shape) == 3 and output.shape[2] >= 13:
                # 3D format: [batch, num_anchors, features] - TensorRT/ONNX wrapper output
                # Format: [bbox_reg(4), objectness(1), class_scores(8)]
                batch_output = output[0]  # Take first batch

                # Extract components
                bbox_reg = batch_output[:, :4]  # [dx, dy, dw, dh] - raw regression
                objectness = batch_output[:, 4]  # objectness confidence
                class_scores = batch_output[:, 5:13]  # class scores for 8 classes

                # Check if values are already sigmoid-activated (0-1 range)
                # PyTorch models output raw values, ONNX/TensorRT models output sigmoid values
                if (
                    objectness.min() >= 0
                    and objectness.max() <= 1
                    and class_scores.min() >= 0
                    and class_scores.max() <= 1
                ):
                    objectness_sigmoid = objectness
                    class_scores_sigmoid = class_scores
                else:
                    # Apply sigmoid to objectness and class scores
                    objectness_sigmoid = 1 / (1 + np.exp(-objectness))
                    class_scores_sigmoid = 1 / (1 + np.exp(-class_scores))

                # Get max class score and corresponding label (exactly like MMDetection)
                max_class_scores = np.max(class_scores_sigmoid, axis=1)
                labels = np.argmax(class_scores_sigmoid, axis=1)

                # Calculate final scores (exactly like MMDetection)
                # MMDetection: scores = cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid()
                # We need to use the max class score for each detection
                scores = max_class_scores * objectness_sigmoid

                # Decode bbox regression to actual coordinates using proper priors
                img_h, img_w = img_info.get("img_shape", (960, 960))[:2]
                priors = generate_yolox_priors((img_h, img_w))

                # Ensure we have the right number of priors
                if len(priors) != len(bbox_reg):
                    print(f"WARNING: Prior count mismatch: {len(priors)} vs {len(bbox_reg)}")
                    # Use a subset if there's a mismatch
                    min_len = min(len(priors), len(bbox_reg))
                    priors = priors[:min_len]
                    bbox_reg = bbox_reg[:min_len]
                    objectness = objectness[:min_len]
                    class_scores = class_scores[:min_len]

                # Proper YOLOX bbox decoding
                # bbox_reg: [dx, dy, dw, dh] - regression offsets
                # priors: [center_x, center_y, stride_w, stride_h]

                # Decode bbox using MMDetection's exact logic
                # MMDetection: xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
                # MMDetection: whs = bbox_preds[..., 2:].exp() * priors[:, 2:]

                # Decode center coordinates (exactly like MMDetection)
                center_x = bbox_reg[:, 0] * priors[:, 2] + priors[:, 0]
                center_y = bbox_reg[:, 1] * priors[:, 3] + priors[:, 1]

                # Decode width and height (exactly like MMDetection)
                width = np.exp(bbox_reg[:, 2]) * priors[:, 2]
                height = np.exp(bbox_reg[:, 3]) * priors[:, 3]

                # Convert to corner format [x1, y1, x2, y2] (exactly like MMDetection)
                x1 = center_x - width / 2
                y1 = center_y - height / 2
                x2 = center_x + width / 2
                y2 = center_y + height / 2

                bboxes = np.stack([x1, y1, x2, y2], axis=1)

                # Apply rescale to match test.py coordinate system (like MMDetection rescale=True)
                # Get scale factor from img_info
                scale_factor = img_info.get("scale_factor", [1.0, 1.0, 1.0, 1.0])
                if len(scale_factor) >= 2:
                    scale_x = scale_factor[0]
                    scale_y = scale_factor[1]

                    # Rescale bboxes from model coordinates to original image coordinates
                    bboxes[:, 0] /= scale_x  # x1
                    bboxes[:, 1] /= scale_y  # y1
                    bboxes[:, 2] /= scale_x  # x2
                    bboxes[:, 3] /= scale_y  # y2

                # Check bbox validity (allow bboxes outside 960x960 like test.py)
                valid_bboxes = (x2 > x1) & (y2 > y1) & (x1 >= 0) & (y1 >= 0)
                if not valid_bboxes.all():
                    invalid_indices = np.where(~valid_bboxes)[0]

                # Apply bbox validity filter (only check x2>x1, y2>y1, x1>=0, y1>=0)
                if np.any(valid_bboxes):
                    bboxes = bboxes[valid_bboxes]
                    scores = scores[valid_bboxes]
                    labels = labels[valid_bboxes]
                    objectness_sigmoid = objectness_sigmoid[valid_bboxes]
                    max_class_scores = max_class_scores[valid_bboxes]
                else:
                    bboxes = np.array([])
                    scores = np.array([])
                    labels = np.array([])
                    objectness_sigmoid = np.array([])
                    max_class_scores = np.array([])

            elif len(output.shape) == 2 and output.shape[1] >= 7:
                # 2D format: [N, 7] where 7 = [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
                bboxes = output[:, :4]  # [x1, y1, x2, y2]
                scores = output[:, 4] * output[:, 5]  # obj_conf * cls_conf
                labels = output[:, 6].astype(int)
                # For 2D format, we don't have objectness_sigmoid and max_class_scores
                # Use scores directly for filtering
                objectness_sigmoid = output[:, 4]  # obj_conf
                max_class_scores = output[:, 5]  # cls_conf
            else:
                # No detections
                return predictions

        # Apply score threshold filtering (exactly like MMDetection)
        # MMDetection: valid_mask = flatten_objectness[img_id] * max_scores >= cfg.score_thr
        score_thr = 0.01  # Default threshold from test_cfg
        if len(scores) > 0:
            # Use objectness * max_class_scores for filtering (not final scores)
            # For 2D format, this is already obj_conf * cls_conf = scores
            # For 3D format, this is objectness_sigmoid * max_class_scores
            if "objectness_sigmoid" in locals() and "max_class_scores" in locals():
                valid_mask = objectness_sigmoid * max_class_scores >= score_thr
            else:
                # Fallback: use scores directly
                valid_mask = scores >= score_thr
            bboxes = bboxes[valid_mask]
            scores = scores[valid_mask]
            labels = labels[valid_mask]

            # Apply NMS (matching test.py behavior)
            if len(scores) > 0:
                # Convert to torch tensors for NMS
                bboxes_tensor = torch.from_numpy(bboxes).float()
                scores_tensor = torch.from_numpy(scores).float()
                labels_tensor = torch.from_numpy(labels).long()

                # Apply batched NMS (exactly like MMDetection)
                try:
                    from mmcv.ops import batched_nms

                    # Use batched NMS (same as MMDetection)
                    nms_cfg = dict(type="nms", iou_threshold=0.65)

                    if len(bboxes_tensor) > 0:
                        det_bboxes, keep_idxs = batched_nms(bboxes_tensor, scores_tensor, labels_tensor, nms_cfg)

                        # Keep only NMS results
                        bboxes = bboxes[keep_idxs.cpu().numpy()]
                        scores = det_bboxes[:, -1].cpu().numpy()  # NMS may reweight scores
                        labels = labels[keep_idxs.cpu().numpy()]

                    else:
                        bboxes = np.array([])
                        scores = np.array([])
                        labels = np.array([])

                except ImportError:
                    bboxes = np.array([])
                    scores = np.array([])
                    labels = np.array([])

        # YOLOX does NOT apply max_per_img limit (unlike other detectors)
        # This is confirmed by checking YOLOXHead._bbox_post_process method
        # which only applies NMS but no max_per_img filtering

        # Convert to [x, y, w, h] format
        for bbox, score, label in zip(bboxes, scores, labels):
            if isinstance(bbox, np.ndarray) and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                # Keep in [x1, y1, x2, y2] format to match ground truth
                predictions.append(
                    {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
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
        img_info = gt_data["img_info"]

        # Get original image size and model input size
        orig_h = img_info.get("height", 1080)
        orig_w = img_info.get("width", 1440)
        # Keep GT in original image coordinates to match rescaled predictions

        for bbox, label in zip(gt_bboxes, gt_labels):
            # Keep GT in original image coordinates (no scaling needed)
            x1, y1, x2, y2 = bbox
            # Keep in [x1, y1, x2, y2] format to match prediction format
            gt_bbox = [x1, y1, x2, y2]

            ground_truths.append({"bbox": gt_bbox, "label": int(label)})

        return ground_truths

    def _compute_metrics(
        self,
        predictions_list: List[List[Dict]],
        ground_truths_list: List[List[Dict]],
        latencies: List[float],
        logger,
    ) -> Dict[str, Any]:
        """
        Compute evaluation metrics using autoware_perception_evaluation.

        This method uses Detection2DMetricsAdapter to ensure consistent metrics
        with the autoware_perception_evaluation library.
        """
        # Compute latency statistics
        latency_stats = self.compute_latency_stats(latencies)

        # Use Detection2DMetricsAdapter for consistent metrics
        try:
            # Reset adapter for new evaluation
            self.metrics_adapter.reset()

            # Add all frames to the adapter
            for predictions, ground_truths in zip(predictions_list, ground_truths_list):
                self.metrics_adapter.add_frame(predictions, ground_truths)

            # Compute metrics using autoware_perception_evaluation
            map_results = self.metrics_adapter.compute_metrics()

            # Get summary for primary metrics
            summary = self.metrics_adapter.get_summary()

            # Combine results
            results = {
                "mAP": summary.get("mAP", 0.0),
                "mAP_50": map_results.get("mAP_iou_2d_0.5", summary.get("mAP", 0.0)),
                "mAP_75": map_results.get("mAP_iou_2d_0.75", 0.0),
                "per_class_ap": summary.get("per_class_ap", {}),
                "detailed_metrics": map_results,
                "latency": latency_stats,
                "num_samples": len(predictions_list),
            }

            logger.info("Successfully computed metrics using autoware_perception_evaluation")
            logger.info(f"   mAP: {results['mAP']:.4f}")

        except Exception as e:
            logger.warning(f"Failed to compute metrics using autoware_perception_evaluation: {e}")
            import traceback

            traceback.print_exc()

            # Fallback to empty metrics
            results = {
                "mAP": 0.0,
                "mAP_50": 0.0,
                "mAP_75": 0.0,
                "per_class_ap": {},
                "latency": latency_stats,
                "num_samples": len(predictions_list),
            }

        return results

    def print_results(self, results: EvalResultDict) -> None:
        """
        Pretty print evaluation results.

        Args:
            results: Results dictionary from evaluate()
        """
        print("\n" + "=" * 80)
        print("YOLOX_opt_elan Object Detection - Evaluation Results")
        print("(Using autoware_perception_evaluation for consistent metrics)")
        print("=" * 80)

        # Detection metrics from autoware_perception_evaluation
        print(f"\nDetection Metrics (autoware_perception_evaluation):")
        print(f"  mAP: {results.get('mAP', 0.0):.4f}")
        print(f"  mAP @ IoU=0.50: {results.get('mAP_50', 0.0):.4f}")
        print(f"  mAP @ IoU=0.75: {results.get('mAP_75', 0.0):.4f}")

        # Per-class AP (show all 8 object classes)
        if "per_class_ap" in results:
            print(f"\nPer-Class AP (Object Classes):")
            for class_id, ap in results["per_class_ap"].items():
                # class_id can be string (class name) or int (index)
                if isinstance(class_id, str):
                    class_name = class_id
                elif isinstance(class_id, int) and class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                else:
                    class_name = f"class_{class_id}"
                # Handle both numeric and dict AP values
                if isinstance(ap, dict):
                    ap_value = ap.get("ap", 0.0)
                else:
                    ap_value = float(ap) if ap is not None else 0.0
                print(f"  {class_name:25s}: {ap_value:.4f}")

        # Latency
        print(f"\nLatency Statistics:")
        latency = results["latency"]
        print(f"  Mean:   {latency['mean_ms']:.2f} ms")
        print(f"  Std:    {latency['std_ms']:.2f} ms")
        print(f"  Min:    {latency['min_ms']:.2f} ms")
        print(f"  Max:    {latency['max_ms']:.2f} ms")
        print(f"  Median: {latency['median_ms']:.2f} ms")
        # Optional percentiles (if computed)
        if "p95_ms" in latency:
            print(f"  P95:    {latency['p95_ms']:.2f} ms")
        if "p99_ms" in latency:
            print(f"  P99:    {latency['p99_ms']:.2f} ms")

        print(f"\nTotal Samples: {results['num_samples']}")
        print("=" * 80)
