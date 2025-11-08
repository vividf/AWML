"""
YOLOX_opt_elan Evaluator for deployment.

This module implements evaluation for YOLOX_opt_elan object detection models.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import torch
from mmengine.config import Config

from autoware_ml.deployment.core import BaseEvaluator

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

    def __init__(self, model_cfg: Config, class_names: List[str] = None):
        """
        Initialize YOLOX_opt_elan evaluator.

        Args:
            model_cfg: Model configuration
            class_names: List of class names (optional, will try to get from config)
        """
        super().__init__(config={})
        self.model_cfg = model_cfg

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

    def verify(
        self,
        pytorch_model_path: str,
        onnx_model_path: str = None,
        tensorrt_model_path: str = None,
        data_loader: YOLOXOptElanDataLoader = None,
        num_samples: int = 1,
        device: str = "cpu",
        tolerance: float = 0.1,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Verify exported models against PyTorch reference by comparing raw outputs.
        
        This is similar to evaluate() but focuses on numerical consistency
        rather than detection metrics. It compares raw model outputs before postprocessing.
        
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
                'summary': {'passed': int, 'failed': int, 'total': int}
            }
        """
        from .main import load_pytorch_model
        from autoware_ml.deployment.pipelines.yolox import (
            YOLOXPyTorchPipeline,
            YOLOXONNXPipeline,
            YOLOXTensorRTPipeline,
        )
        
        logger = logging.getLogger(__name__)
        
        logger.info("\n" + "=" * 60)
        logger.info("YOLOX_opt_elan Model Verification")
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
        pytorch_model = load_pytorch_model(self.model_cfg, pytorch_model_path, device)

        # TODO(vividf): check this
        # Create PyTorch pipeline (replace ReLU6 with ReLU to match ONNX export)
        # This ensures verification uses the same model state as export
        def replace_relu6_with_relu(module):
            for name, child in module.named_children():
                if isinstance(child, torch.nn.ReLU6):
                    setattr(module, name, torch.nn.ReLU(inplace=child.inplace))
                else:
                    replace_relu6_with_relu(child)
        
        replace_relu6_with_relu(pytorch_model)
        
        pytorch_pipeline = YOLOXPyTorchPipeline(
            pytorch_model=pytorch_model,
            device=device,
            num_classes=len(self.class_names),
            class_names=self.class_names
        )
        
        # Create ONNX pipeline if requested
        onnx_pipeline = None
        if onnx_model_path:
            logger.info("\nInitializing ONNX pipeline...")
            try:
                onnx_pipeline = YOLOXONNXPipeline(
                    onnx_path=onnx_model_path,
                    device=device,
                    num_classes=len(self.class_names),
                    class_names=self.class_names
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
                    tensorrt_pipeline = YOLOXTensorRTPipeline(
                        engine_path=tensorrt_model_path,
                        device=device,
                        num_classes=len(self.class_names),
                        class_names=self.class_names
                    )
                except Exception as e:
                    logger.warning(f"Failed to create TensorRT pipeline, skipping TensorRT verification: {e}")
                    skipped_backends.append("tensorrt")
        
        # Verify each sample
        try:
            for i in range(min(num_samples, data_loader.get_num_samples())):
                logger.info(f"\n{'='*60}")
                logger.info(f"Verifying sample {i}")
                logger.info(f"{'='*60}")
                
                # Load sample and preprocess via MMDet test pipeline (same as evaluation)
                sample = data_loader.load_sample(i)
                input_tensor = data_loader.preprocess(sample)
                
                # Get ground truth and img_info (required for YOLOX preprocessing)
                gt_data = data_loader.get_ground_truth(i)
                img_info = gt_data["img_info"]
                
                # Get PyTorch reference outputs (raw, before postprocessing)
                logger.info("\nRunning PyTorch reference (raw outputs)...")
                try:
                    pytorch_output, pytorch_latency, _ = pytorch_pipeline.infer(
                        input_tensor, 
                        return_raw_outputs=True,
                        img_info=img_info
                    )
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
                    logger.info("\nVerifying ONNX pipeline...")
                    onnx_passed = self._verify_single_backend(
                        onnx_pipeline,
                        input_tensor,
                        pytorch_output,
                        pytorch_latency,
                        tolerance,
                        "ONNX",
                        logger,
                        img_info=img_info
                    )
                    results[f"sample_{i}_onnx"] = onnx_passed
                
                # Verify TensorRT
                if tensorrt_pipeline:
                    logger.info("\nVerifying TensorRT pipeline...")
                    tensorrt_passed = self._verify_single_backend(
                        tensorrt_pipeline,
                        input_tensor,
                        pytorch_output,
                        pytorch_latency,
                        tolerance,
                        "TensorRT",
                        logger,
                        img_info=img_info
                    )
                    results[f"sample_{i}_tensorrt"] = tensorrt_passed
                
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
        skipped = len(skipped_backends) * num_samples
        
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
                    input_tensor, 
                    return_raw_outputs=True,
                    img_info=img_info
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
        model_path: str,
        data_loader: YOLOXOptElanDataLoader,
        num_samples: int,
        backend: str = "pytorch",
        device: str = "cpu",
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run full evaluation on YOLOX_opt_elan model.

        Args:
            model_path: Path to model checkpoint/weights
            data_loader: YOLOX_opt_elan DataLoader
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

        # Create YOLOX Pipeline instead of generic backend
        pipeline = self._create_backend(backend, model_path, device, logger)

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
                if isinstance(det, dict) and 'bbox' in det:
                    predictions.append({
                        'bbox': det['bbox'],
                        'label': int(det.get('class_id', det.get('label', 0))),
                        'score': float(det.get('score', 0.0)),
                    })
            all_predictions.append(predictions)

            # Parse ground truths
            ground_truths = self._parse_ground_truths(gt_data)
            all_ground_truths.append(ground_truths)

            if verbose:
                logger.info(
                    f"  Sample {idx}: {len(predictions)} predictions, {len(ground_truths)} ground truths"
                )

            # GPU cleanup for TensorRT
            if backend == "tensorrt" and idx % GPU_CLEANUP_INTERVAL == 0:
                torch.cuda.empty_cache()

        # Compute metrics
        results = self._compute_metrics(all_predictions, all_ground_truths, latencies, logger)

        return results

    def _create_backend(self, backend: str, model_path: str, device: str, logger):
        """Create YOLOX Pipeline for the specified backend."""
        from autoware_ml.deployment.pipelines.yolox import (
            YOLOXPyTorchPipeline,
            YOLOXONNXPipeline,
            YOLOXTensorRTPipeline,
        )
        from mmdet.apis import init_detector
        import os

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

        if backend == "pytorch":
            model = init_detector(self.model_cfg, model_path, device=device)
            # Replace ReLU6 with ReLU to match ONNX export (for consistency)
            def replace_relu6_with_relu(module):
                for name, child in module.named_children():
                    if isinstance(child, torch.nn.ReLU6):
                        setattr(module, name, torch.nn.ReLU(inplace=child.inplace))
                    else:
                        replace_relu6_with_relu(child)
            replace_relu6_with_relu(model)
            return YOLOXPyTorchPipeline(
                pytorch_model=model,
                device=device,
                num_classes=num_classes,
                class_names=class_names,
            )
        elif backend == "onnx":
            if not (os.path.isfile(model_path) and model_path.endswith(".onnx")):
                raise ValueError("ONNX evaluation expects a .onnx file path")
            return YOLOXONNXPipeline(
                onnx_path=model_path,
                device=device,
                num_classes=num_classes,
                class_names=class_names,
            )
        elif backend == "tensorrt":
            if not (os.path.isfile(model_path) and (model_path.endswith(".engine") or model_path.endswith(".trt"))):
                raise ValueError("TensorRT evaluation expects an engine (.engine/.trt) file path")
            return YOLOXTensorRTPipeline(
                engine_path=model_path,
                device=device,
                num_classes=num_classes,
                class_names=class_names,
            )
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
                scores = output[:, 4]   # scores
                labels = output[:, 5]   # labels
                
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
                        predictions.append({
                            'bbox': [float(bboxes[i, 0]), float(bboxes[i, 1]), float(bboxes[i, 2]), float(bboxes[i, 3])],
                            'label': int(labels[i]),
                            'score': float(scores[i])
                        })
                
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
                if objectness.min() >= 0 and objectness.max() <= 1 and class_scores.min() >= 0 and class_scores.max() <= 1:
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
                img_h, img_w = img_info.get('img_shape', (960, 960))[:2]
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
                scale_factor = img_info.get('scale_factor', [1.0, 1.0, 1.0, 1.0])
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
                max_class_scores = output[:, 5]   # cls_conf
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
            if 'objectness_sigmoid' in locals() and 'max_class_scores' in locals():
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
                        det_bboxes, keep_idxs = batched_nms(
                            bboxes_tensor, scores_tensor, labels_tensor, nms_cfg
                        )
                        
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

            ground_truths.append(
                {"bbox": gt_bbox, "label": int(label)}
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
        # Use Pascal VOC evaluation to match test.py
        map_results = self._compute_voc_map(
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

    def _compute_voc_map(
        self,
        predictions_list: List[List[Dict]],
        ground_truths_list: List[List[Dict]],
        num_classes: int,
    ) -> Dict[str, Any]:
        """Compute Pascal VOC mAP to match test.py evaluation."""
        from mmdet.evaluation.functional import eval_map
        import numpy as np
        
        # Convert to MMDetection format for eval_map
        det_results = []
        annotations = []
        
        for predictions, ground_truths in zip(predictions_list, ground_truths_list):
            # Group predictions by class - each class gets a numpy array
            class_detections = []
            
            for class_id in range(num_classes):
                class_preds = []
                for pred in predictions:
                    if pred['label'] == class_id:
                        # Format: [x1, y1, x2, y2, score]
                        bbox_with_score = pred['bbox'] + [pred['score']]
                        class_preds.append(bbox_with_score)
                
                # Convert to numpy array
                if class_preds:
                    class_detections.append(np.array(class_preds))
                else:
                    class_detections.append(np.zeros((0, 5)))
            
            # Convert ground truths to numpy arrays
            gt_bboxes = []
            gt_labels = []
            
            for gt in ground_truths:
                gt_bboxes.append(gt['bbox'])
                gt_labels.append(gt['label'])
            
            det_results.append(class_detections)
            annotations.append({
                'bboxes': np.array(gt_bboxes) if gt_bboxes else np.zeros((0, 4)),
                'labels': np.array(gt_labels) if gt_labels else np.zeros(0, dtype=int),
            })
        
        # Compute Pascal VOC mAP (IoU@0.5)
        try:
            map_results = eval_map(
                det_results,
                annotations,
                iou_thr=0.5,  # Pascal VOC uses IoU@0.5
                scale_ranges=None,
                eval_mode='area',  # Use area mode like VOC2012
            )
            
            # Extract results - eval_map returns (mean_ap, per_class_ap)
            map_50 = map_results[0]  # mAP@0.5
            per_class_ap_list = map_results[1]  # Per-class AP@0.5 (list)
            
            # Convert per_class_ap list to dict for consistency
            per_class_ap = {}
            for i, ap in enumerate(per_class_ap_list):
                per_class_ap[i] = ap
            
            return {
                "mAP": map_50,
                "mAP_50": map_50,
                "mAP_75": map_50,  # Same as mAP_50 for Pascal VOC
                "per_class_ap": per_class_ap,
            }
            
        except Exception as e:
            print(f"Error computing VOC mAP: {e}")
            import traceback
            traceback.print_exc()
            return {
                "mAP": 0.0,
                "mAP_50": 0.0,
                "mAP_75": 0.0,
                "per_class_ap": {i: 0.0 for i in range(num_classes)},
            }

    def print_results(self, results: Dict[str, Any]) -> None:
        """
        Pretty print evaluation results.

        Args:
            results: Results dictionary from evaluate()
        """
        print("\n" + "=" * 80)
        print("YOLOX_opt_elan Object Detection - Evaluation Results")
        print("=" * 80)

        # Detection metrics
        print(f"\nDetection Metrics:")
        print(f"  mAP (0.5:0.95): {results['mAP']:.4f}")
        print(f"  mAP @ IoU=0.50: {results['mAP_50']:.4f}")
        print(f"  mAP @ IoU=0.75: {results['mAP_75']:.4f}")

        # Per-class AP (show all 8 object classes)
        if "per_class_ap" in results:
            print(f"\nPer-Class AP (Object Classes):")
            for class_id, ap in results["per_class_ap"].items():
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                # Handle both numeric and dict AP values
                if isinstance(ap, dict):
                    ap_value = ap.get('ap', 0.0)
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
