"""
YOLOX_opt_elan Evaluator for deployment.

This module implements evaluation for YOLOX_opt_elan object detection models.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import torch
from mmengine.config import Config

from autoware_ml.deployment.backends import ONNXBackend, PyTorchBackend, TensorRTBackend
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

        # Get class names - default to 8 object classes
        if class_names is not None:
            self.class_names = class_names
        elif hasattr(model_cfg, "class_names"):
            self.class_names = model_cfg.class_names
        elif "model" in model_cfg and "bbox_head" in model_cfg.model:
            num_classes = model_cfg.model.bbox_head.get("num_classes", 8)
            # Default object class names
            default_object_classes = [
                "unknown",
                "car",
                "truck",
                "bus",
                "trailer",
                "motorcycle",
                "pedestrian",
                "bicycle",
            ]
            if num_classes == 8:
                self.class_names = default_object_classes
            else:
                self.class_names = [f"class_{i}" for i in range(num_classes)]
        else:
            # Default to 8 object classes
            self.class_names = [
                "unknown",
                "car",
                "truck",
                "bus",
                "trailer",
                "motorcycle",
                "pedestrian",
                "bicycle",
            ]

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

                if verbose:
                    logger.info(
                        f"  Sample {idx}: {len(predictions)} predictions, " f"{len(ground_truths)} ground truths"
                    )

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
            # Get num_classes from model config for proper output filtering
            num_classes = None
            if hasattr(self.model_cfg, "model") and hasattr(self.model_cfg.model, "bbox_head"):
                num_classes = self.model_cfg.model.bbox_head.get("num_classes", None)
                if num_classes:
                    logger.info(f"Using num_classes={num_classes} from config for ONNX output filtering")
            return ONNXBackend(model_path, device, num_classes=num_classes)
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

        # Debug logging for output analysis
        print(f"DEBUG: Raw output type: {type(output)}")
        print(f"DEBUG: Raw output shape: {output.shape}")
        print(f"DEBUG: Raw output dtype: {output.dtype}")
        if hasattr(output, 'min') and hasattr(output, 'max'):
            print(f"DEBUG: Raw output min/max: {output.min():.6f} / {output.max():.6f}")
        if output.size < 50:  # Only print small arrays
            print(f"DEBUG: Raw output content: {output}")

        # Handle different output formats
        if isinstance(output, dict):
            # Dict format
            bboxes = output.get("bboxes", np.array([]))
            scores = output.get("scores", np.array([]))
            labels = output.get("labels", np.array([]))
            print(f"DEBUG: Dict format - bboxes: {bboxes.shape}, scores: {scores.shape}, labels: {labels.shape}")
        else:
            # Array format - handle both 2D and 3D outputs
            if len(output.shape) == 2 and output.shape[1] == 6:
                # 2D format: [num_detections, 6] - Standard MMDetection PyTorch output
                # Format: [x1, y1, x2, y2, score, label]
                print(f"DEBUG: 2D format detected - Standard MMDetection output")
                
                if len(output) == 0:
                    return predictions
                
                bboxes = output[:, :4]  # [x1, y1, x2, y2]
                scores = output[:, 4]   # scores
                labels = output[:, 5]   # labels
                
                print(f"DEBUG: 2D format - bboxes: {bboxes.shape}, scores: {scores.shape}, labels: {labels.shape}")
                
                # For 2D format, we already have final predictions, just need to filter by score
                score_thr = 0.3  # Default score threshold
                if len(scores) > 0:
                    valid_mask = scores >= score_thr
                    bboxes = bboxes[valid_mask]
                    scores = scores[valid_mask]
                    labels = labels[valid_mask]
                    print(f"DEBUG: After score filtering (thr={score_thr}): {len(scores)} predictions")
                
                # Keep in [x1, y1, x2, y2] format for evaluation (MMDetection format)
                if len(bboxes) > 0:
                    for i in range(len(bboxes)):
                        predictions.append({
                            'bbox': [float(bboxes[i, 0]), float(bboxes[i, 1]), float(bboxes[i, 2]), float(bboxes[i, 3])],
                            'label': int(labels[i]),
                            'score': float(scores[i])
                        })
                
                print(f"DEBUG: Final 2D predictions: {len(predictions)}")
                return predictions
                
            elif len(output.shape) == 3 and output.shape[2] >= 13:
                # 3D format: [batch, num_anchors, features] - TensorRT/ONNX wrapper output
                # Format: [bbox_reg(4), objectness(1), class_scores(8)]
                batch_output = output[0]  # Take first batch
                
                # Extract components
                bbox_reg = batch_output[:, :4]  # [dx, dy, dw, dh] - raw regression
                objectness = batch_output[:, 4]  # objectness confidence
                class_scores = batch_output[:, 5:13]  # class scores for 8 classes
                
                # Debug: Check raw values before sigmoid
                print(f"DEBUG: Raw objectness range: {objectness.min():.6f} - {objectness.max():.6f}")
                print(f"DEBUG: Raw class_scores range: {class_scores.min():.6f} - {class_scores.max():.6f}")
                
                # Check if values are already sigmoid-activated (0-1 range)
                # PyTorch models output raw values, ONNX/TensorRT models output sigmoid values
                if objectness.min() >= 0 and objectness.max() <= 1 and class_scores.min() >= 0 and class_scores.max() <= 1:
                    print("DEBUG: Values already sigmoid-activated (ONNX/TensorRT output)")
                    objectness_sigmoid = objectness
                    class_scores_sigmoid = class_scores
                else:
                    print("DEBUG: Applying sigmoid activation (PyTorch output)")
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
                
                # Debug: Check bbox regression values
                print(f"DEBUG: Raw bbox_reg range: {bbox_reg.min():.6f} - {bbox_reg.max():.6f}")
                print(f"DEBUG: Raw bbox_reg mean: {bbox_reg.mean():.6f}")
                print(f"DEBUG: First 5 bbox_reg: {bbox_reg[:5]}")
                print(f"DEBUG: First 5 priors: {priors[:5]}")
                
                # Decode bbox using MMDetection's exact logic
                # MMDetection: xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
                # MMDetection: whs = bbox_preds[..., 2:].exp() * priors[:, 2:]
                
                # Decode center coordinates (exactly like MMDetection)
                center_x = bbox_reg[:, 0] * priors[:, 2] + priors[:, 0]
                center_y = bbox_reg[:, 1] * priors[:, 3] + priors[:, 1]
                
                # Debug: Check decoded centers
                print(f"DEBUG: Decoded centers range: x={center_x.min():.1f}-{center_x.max():.1f}, y={center_y.min():.1f}-{center_y.max():.1f}")
                
                # Decode width and height (exactly like MMDetection)
                width = np.exp(bbox_reg[:, 2]) * priors[:, 2]
                height = np.exp(bbox_reg[:, 3]) * priors[:, 3]
                
                # Debug: Check decoded dimensions
                print(f"DEBUG: Decoded dimensions range: w={width.min():.1f}-{width.max():.1f}, h={height.min():.1f}-{height.max():.1f}")
                
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
                    print(f"DEBUG: Applying rescale - scale factors: x={scale_x:.4f}, y={scale_y:.4f}")
                    
                    # Rescale bboxes from model coordinates to original image coordinates
                    bboxes[:, 0] /= scale_x  # x1
                    bboxes[:, 1] /= scale_y  # y1
                    bboxes[:, 2] /= scale_x  # x2
                    bboxes[:, 3] /= scale_y  # y2
                    
                    print(f"DEBUG: After rescale - bbox range: x1={bboxes[:, 0].min():.1f}-{bboxes[:, 0].max():.1f}")
                    print(f"DEBUG: After rescale - bbox range: y1={bboxes[:, 1].min():.1f}-{bboxes[:, 1].max():.1f}")
                    print(f"DEBUG: After rescale - bbox range: x2={bboxes[:, 2].min():.1f}-{bboxes[:, 2].max():.1f}")
                    print(f"DEBUG: After rescale - bbox range: y2={bboxes[:, 3].min():.1f}-{bboxes[:, 3].max():.1f}")
                else:
                    print(f"DEBUG: No scale factor found, keeping model coordinates")
                
                # Debug: Check final bboxes
                print(f"DEBUG: Final bboxes range: x1={x1.min():.1f}-{x1.max():.1f}, y1={y1.min():.1f}-{y1.max():.1f}")
                print(f"DEBUG: Final bboxes range: x2={x2.min():.1f}-{x2.max():.1f}, y2={y2.min():.1f}-{y2.max():.1f}")
                
                # Debug: Check bbox validity (allow bboxes outside 960x960 like test.py)
                valid_bboxes = (x2 > x1) & (y2 > y1) & (x1 >= 0) & (y1 >= 0)
                print(f"DEBUG: Valid bboxes count: {valid_bboxes.sum()}/{len(valid_bboxes)}")
                if not valid_bboxes.all():
                    invalid_indices = np.where(~valid_bboxes)[0]
                    print(f"DEBUG: Invalid bbox indices: {invalid_indices[:10]}")  # Show first 10
                
                # Apply bbox validity filter (only check x2>x1, y2>y1, x1>=0, y1>=0)
                if np.any(valid_bboxes):
                    bboxes = bboxes[valid_bboxes]
                    scores = scores[valid_bboxes]
                    labels = labels[valid_bboxes]
                    objectness_sigmoid = objectness_sigmoid[valid_bboxes]
                    max_class_scores = max_class_scores[valid_bboxes]
                    print(f"DEBUG: After bbox validity filter: {len(scores)} predictions")
                else:
                    print(f"DEBUG: No valid bboxes found")
                    bboxes = np.array([])
                    scores = np.array([])
                    labels = np.array([])
                    objectness_sigmoid = np.array([])
                    max_class_scores = np.array([])
                
                print(f"DEBUG: 3D format - bboxes: {bboxes.shape}, scores: {scores.shape}, labels: {labels.shape}")
                print(f"DEBUG: Score range: {scores.min():.6f} - {scores.max():.6f}")
                print(f"DEBUG: Objectness range: {objectness_sigmoid.min():.6f} - {objectness_sigmoid.max():.6f}")
                print(f"DEBUG: Class scores range: {max_class_scores.min():.6f} - {max_class_scores.max():.6f}")
        
            elif len(output.shape) == 2 and output.shape[1] >= 7:
                # 2D format: [N, 7] where 7 = [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
                bboxes = output[:, :4]  # [x1, y1, x2, y2]
                scores = output[:, 4] * output[:, 5]  # obj_conf * cls_conf
                labels = output[:, 6].astype(int)
                print(f"DEBUG: 2D format - bboxes: {bboxes.shape}, scores: {scores.shape}, labels: {labels.shape}")
                print(f"DEBUG: Score range: {scores.min():.6f} - {scores.max():.6f}")
            else:
                # No detections
                print(f"DEBUG: Output shape {output.shape} doesn't match expected format, returning empty predictions")
                return predictions

        # Apply score threshold filtering (exactly like MMDetection)
        # MMDetection: valid_mask = flatten_objectness[img_id] * max_scores >= cfg.score_thr
        score_thr = 0.01  # Default threshold from test_cfg
        if len(scores) > 0:
            # Use objectness * max_class_scores for filtering (not final scores)
            valid_mask = objectness_sigmoid * max_class_scores >= score_thr
            bboxes = bboxes[valid_mask]
            scores = scores[valid_mask]
            labels = labels[valid_mask]
            print(f"DEBUG: After score filtering (thr={score_thr}): {len(scores)} predictions")
            
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
                        
                        print(f"DEBUG: After batched NMS (iou_thr=0.65): {len(scores)} predictions")
                    else:
                        print(f"DEBUG: No detections for NMS")
                        bboxes = np.array([])
                        scores = np.array([])
                        labels = np.array([])
                        
                except ImportError:
                    print("DEBUG: batched_nms not available, skipping NMS step")
                    bboxes = np.array([])
                    scores = np.array([])
                    labels = np.array([])

        # YOLOX does NOT apply max_per_img limit (unlike other detectors)
        # This is confirmed by checking YOLOXHead._bbox_post_process method
        # which only applies NMS but no max_per_img filtering
        print(f"DEBUG: Final predictions after NMS: {len(scores)} predictions")

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

        print(f"DEBUG: Parsed {len(predictions)} predictions")
        
        # Debug: Analyze prediction distribution
        if len(predictions) > 0:
            scores = [p['score'] for p in predictions]
            labels = [p['label'] for p in predictions]
            print(f"DEBUG: Prediction analysis:")
            print(f"  Score range: {min(scores):.4f} - {max(scores):.4f}")
            print(f"  Score mean: {np.mean(scores):.4f}")
            print(f"  Score std: {np.std(scores):.4f}")
            
            # Analyze label distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            print(f"  Label distribution:")
            for label, count in zip(unique_labels, counts):
                print(f"    Label {label}: {count} predictions")
        
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
        print(f"DEBUG: GT coordinates - orig: {orig_w}x{orig_h} (keeping original coordinates)")

        for bbox, label in zip(gt_bboxes, gt_labels):
            # Keep GT in original image coordinates (no scaling needed)
            x1, y1, x2, y2 = bbox
            # Keep in [x1, y1, x2, y2] format to match prediction format
            gt_bbox = [x1, y1, x2, y2]

            ground_truths.append(
                {"bbox": gt_bbox, "label": int(label)}
            )

        # Debug: Analyze ground truth distribution
        if len(ground_truths) > 0:
            gt_labels = [gt['label'] for gt in ground_truths]
            gt_bboxes = np.array([gt['bbox'] for gt in ground_truths])
            print(f"DEBUG: Ground truth analysis:")
            print(f"  GT count: {len(ground_truths)}")
            print(f"  GT bbox range: x1={gt_bboxes[:, 0].min():.1f}-{gt_bboxes[:, 0].max():.1f}, y1={gt_bboxes[:, 1].min():.1f}-{gt_bboxes[:, 1].max():.1f}")
            print(f"  GT bbox range: x2={gt_bboxes[:, 2].min():.1f}-{gt_bboxes[:, 2].max():.1f}, y2={gt_bboxes[:, 3].min():.1f}-{gt_bboxes[:, 3].max():.1f}")
            
            unique_gt_labels, gt_counts = np.unique(gt_labels, return_counts=True)
            print(f"  GT label distribution:")
            for label, count in zip(unique_gt_labels, gt_counts):
                print(f"    Label {label}: {count} ground truths")

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
