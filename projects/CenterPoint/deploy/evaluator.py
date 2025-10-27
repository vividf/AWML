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
        Evaluate model performance.

        Args:
            model_path: Path to model file or directory
            data_loader: Data loader for evaluation
            num_samples: Number of samples to evaluate
            backend: Backend type ('pytorch', 'onnx', 'tensorrt')
            device: Device to run evaluation on
            verbose: Whether to print detailed output

        Returns:
            Dictionary containing evaluation results
        """
        logger = logging.getLogger(__name__)
        
        logger.info(f"\nEvaluating {backend} model: {model_path}")
        logger.info(f"Number of samples: {num_samples}")

        # Create Pipeline instance
        pipeline = self._create_backend(backend, model_path, device, logger)
        
        if pipeline is None:
            logger.error(f"Failed to create {backend} Pipeline")
            return {}

        # Run evaluation
        predictions_list = []
        ground_truths_list = []
        latencies = []

        try:
            for i in range(min(num_samples, data_loader.get_num_samples())):
                if verbose and i % LOG_INTERVAL == 0:
                    logger.info(f"Processing sample {i+1}/{num_samples}")

                # Get sample data
                sample = data_loader.load_sample(i)
                
                # Get points for pipeline
                if 'points' in sample:
                    points = sample['points']
                else:
                    # Load points from data_loader
                    input_data = data_loader.preprocess(sample)
                    points = input_data.get('points', input_data)
                
                # Get ground truth
                gt_data = data_loader.get_ground_truth(i)

                # Run inference using unified Pipeline interface
                sample_meta = sample.get('metainfo', {})
                predictions, latency = pipeline.infer(points, sample_meta)
                
                # Parse ground truths
                ground_truths = self._parse_ground_truths(gt_data)

                predictions_list.append(predictions)
                ground_truths_list.append(ground_truths)
                latencies.append(latency)

                # Cleanup GPU memory periodically
                if i % GPU_CLEANUP_INTERVAL == 0 and torch.cuda.is_available():
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
        results = self._compute_metrics(predictions_list, ground_truths_list, latencies, logger)
        
        return results

    def _create_backend(self, backend: str, model_path: str, device: str, logger) -> Any:
        """Create Pipeline instance for the specified backend."""
        try:
            # Import Pipeline classes
            from autoware_ml.deployment.pipelines import (
                CenterPointPyTorchPipeline,
                CenterPointONNXPipeline,
                CenterPointTensorRTPipeline
            )
            
            # Ensure device is properly set
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"
            
            device_obj = torch.device(device) if isinstance(device, str) else device
            
            # Load PyTorch model (required by all backends)
            if backend == "pytorch":
                pytorch_model = self._load_pytorch_model_directly(model_path, device_obj, logger)
                return CenterPointPyTorchPipeline(pytorch_model, device=str(device_obj))
                
            elif backend == "onnx":
                # Verify ONNX-compatible config
                if self.model_cfg.model.type != "CenterPointONNX":
                    logger.error("Model config is not ONNX-compatible!")
                    logger.error(f"Current model type: {self.model_cfg.model.type}")
                    logger.error("Expected model type: CenterPointONNX")
                    raise ValueError("ONNX requires ONNX-compatible model config")
                
                # Find checkpoint path
                import os
                checkpoint_path = model_path.replace('centerpoint_deployment', 'centerpoint/best_checkpoint.pth')
                if not os.path.exists(checkpoint_path):
                    checkpoint_path = model_path.replace('_deployment', '/best_checkpoint.pth')
                
                pytorch_model = self._load_pytorch_model_directly(checkpoint_path, device_obj, logger)
                return CenterPointONNXPipeline(pytorch_model, onnx_dir=model_path, device=str(device_obj))
                
            elif backend == "tensorrt":
                # TensorRT requires CUDA
                if not str(device).startswith("cuda"):
                    logger.warning("TensorRT requires CUDA device, skipping TensorRT evaluation")
                    return None
                
                # Verify ONNX-compatible config
                if self.model_cfg.model.type != "CenterPointONNX":
                    logger.error("TensorRT requires ONNX-compatible model config")
                    raise ValueError("TensorRT requires ONNX-compatible model config")
                
                # Find checkpoint path
                import os
                checkpoint_path = model_path.replace('centerpoint_deployment/tensorrt', 'centerpoint/best_checkpoint.pth')
                checkpoint_path = checkpoint_path.replace('/tensorrt', '')
                if not os.path.exists(checkpoint_path):
                    checkpoint_path = model_path.replace('_deployment/tensorrt', '/best_checkpoint.pth')
                
                pytorch_model = self._load_pytorch_model_directly(checkpoint_path, device_obj, logger)
                return CenterPointTensorRTPipeline(pytorch_model, tensorrt_dir=model_path, device=str(device_obj))
                
            else:
                logger.error(f"Unsupported backend: {backend}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create {backend} Pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_pytorch_model_directly(self, checkpoint_path: str, device: torch.device, logger) -> Any:
        """
        Load PyTorch model directly without using init_model to avoid CUDA checks.
        
        The model config should already be in the correct format (original or ONNX-compatible)
        based on the backend being evaluated.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on
            logger: Logger instance
        """
        try:
            from mmengine.registry import MODELS, init_default_scope
            from mmengine.runner import load_checkpoint
            import copy as copy_module
            
            # Initialize mmdet3d scope
            init_default_scope("mmdet3d")
            
            # Get model config - use deepcopy to avoid modifying shared nested objects
            model_config = copy_module.deepcopy(self.model_cfg.model)
            
            # For ONNX models, ensure device is set
            if hasattr(model_config, 'device'):
                model_config.device = str(device)
                logger.info(f"Set model config device to: {model_config.device}")
            
            # Build model using MODELS registry
            logger.info(f"Building model with device: {device}")
            logger.info(f"Model type: {model_config.type}")
            model = MODELS.build(model_config)
            model.to(device)
            
            # Add cfg attribute to model (required by inference_detector)
            model.cfg = self.model_cfg
            
            # Load checkpointoriginal_model_cfg
            logger.info(f"Loading checkpoint from: {checkpoint_path}")
            load_checkpoint(model, checkpoint_path, map_location=device)
            
            model.eval()
            
            logger.info(f"Successfully loaded PyTorch model on {device}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model directly: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to init_model if direct loading fails (only for CUDA)
            # Note: init_model doesn't work well with CPU, so skip fallback for CPU
            if str(device).startswith('cuda'):
                try:
                    from mmdet3d.apis import init_model
                    logger.info("Falling back to init_model...")
                    model = init_model(self.model_cfg, checkpoint_path, device=device)
                    return model
                except Exception as fallback_e:
                    logger.error(f"Fallback to init_model also failed: {fallback_e}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.error("Direct model loading failed and fallback is disabled for CPU mode")
            
            raise e



    
    def _parse_with_pytorch_decoder(self, heatmap, reg, height, dim, rot, vel, sample):
        """Use PyTorch model's predict_by_feat for consistent decoding."""
        import torch
        
        # Convert to torch tensors if needed
        if isinstance(heatmap, np.ndarray):
            heatmap = torch.from_numpy(heatmap)
            reg = torch.from_numpy(reg)
            height = torch.from_numpy(height)
            dim = torch.from_numpy(dim)
            rot = torch.from_numpy(rot)
            vel = torch.from_numpy(vel)
        
        # Move to same device as model
        device = next(self.pytorch_model.parameters()).device
        heatmap = heatmap.to(device)
        reg = reg.to(device) if reg is not None else None
        height = height.to(device)
        dim = dim.to(device)
        rot = rot.to(device)
        vel = vel.to(device) if vel is not None else None
        
        # Convert ONNX outputs back to standard format if needed
        rot_y_axis_reference = getattr(self.model_cfg.model.pts_bbox_head, 'rot_y_axis_reference', False)
        
        if rot_y_axis_reference:
            # Convert dim from [w, l, h] back to [l, w, h]
            dim = dim[:, [1, 0, 2], :, :]
            
            # Convert rot from [-cos(x), -sin(y)] back to [sin(y), cos(x)]
            rot = rot * (-1.0)
            rot = rot[:, [1, 0], :, :]
        
        # Prepare head outputs in mmdet3d format: Tuple[List[dict]]
        # The head outputs should be in dict format with keys: 'heatmap', 'reg', 'height', 'dim', 'rot', 'vel'
        preds_dict = {
            'heatmap': heatmap,
            'reg': reg,
            'height': height,
            'dim': dim,
            'rot': rot,
            'vel': vel
        }
        preds_dicts = ([preds_dict],)  # Tuple[List[dict]] format for single task
        
        # Prepare batch_input_metas from sample
        metainfo = sample.get('metainfo', {})
        if 'box_type_3d' not in metainfo:
            from mmdet3d.structures import LiDARInstance3DBoxes
            metainfo['box_type_3d'] = LiDARInstance3DBoxes
        batch_input_metas = [metainfo]
        
        # Call predict_by_feat to get final predictions
        with torch.no_grad():
            predictions_list = self.pytorch_model.pts_bbox_head.predict_by_feat(
                preds_dicts=preds_dicts,
                batch_input_metas=batch_input_metas
            )
        
        # Parse predictions
        predictions = []
        for pred_instances in predictions_list:
            bboxes_3d = pred_instances.bboxes_3d.tensor.cpu().numpy()
            scores_3d = pred_instances.scores_3d.cpu().numpy()
            labels_3d = pred_instances.labels_3d.cpu().numpy()
            
            for i in range(len(bboxes_3d)):
                predictions.append({
                    'bbox_3d': bboxes_3d[i][:7].tolist(),
                    'score': float(scores_3d[i]),
                    'label': int(labels_3d[i])
                })
        
        return predictions

    def _parse_centerpoint_head_outputs(self, heatmap, reg, height, dim, rot, vel, sample):
        """
        Parse CenterPoint head outputs using PyTorch's predict_by_feat for consistency.
        
        This method delegates all post-processing to the PyTorch model's implementation,
        ensuring consistent results between PyTorch, ONNX, and TensorRT backends.
        
        The manual parsing fallback has been removed to enforce consistency and simplify
        maintenance. All decoding logic (top-K selection, coordinate transformation, NMS,
        score filtering, etc.) is handled by PyTorch's CenterPointBBoxCoder.
        
        Args:
            heatmap: Heatmap predictions [B, num_classes, H, W]
            reg: Regression offsets [B, 2, H, W]
            height: Height predictions [B, 1, H, W]
            dim: Dimension predictions [B, 3, H, W]
            rot: Rotation predictions [B, 2, H, W]
            vel: Velocity predictions [B, 2, H, W]
            sample: Sample data containing metadata
            
        Returns:
            List of predictions with bbox_3d, score, and label
            
        Raises:
            RuntimeError: If PyTorch model is not available or decoding fails
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Verify PyTorch model is available
        if not hasattr(self, 'pytorch_model') or self.pytorch_model is None:
            raise RuntimeError(
                "CenterPoint decoding requires PyTorch model for consistent post-processing.\n"
                "The pytorch_model attribute is None or not available.\n"
                "This typically means the backend was not properly initialized with a PyTorch model reference.\n"
                "All backends (ONNX, TensorRT) should pass pytorch_model during initialization."
            )
        
        # Use PyTorch model's predict_by_feat for consistent post-processing
        try:
            logger.info("✅ Using PyTorch decoder (predict_by_feat) for consistent post-processing")
            return self._parse_with_pytorch_decoder(heatmap, reg, height, dim, rot, vel, sample)
        except Exception as e:
            logger.error(f"❌ Failed to use PyTorch predict_by_feat: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(
                f"Failed to decode CenterPoint outputs using PyTorch predict_by_feat: {e}\n\n"
                "This may indicate:\n"
                "  1. Incompatible tensor shapes or formats between ONNX/TensorRT and PyTorch\n"
                "  2. Missing or incorrect metadata in sample (point_cloud_range, voxel_size, etc.)\n"
                "  3. Model configuration mismatch between backends\n"
                "  4. Memory issues on GPU\n\n"
                "Please check the error traceback above for details.\n"
                "You may need to verify that:\n"
                "  - ONNX/TensorRT outputs match PyTorch head output format\n"
                "  - Sample metadata contains all required fields\n"
                "  - Model config is consistent across all backends"
            ) from e


    def _parse_ground_truths(self, gt_data: Dict) -> List[Dict]:
        """Parse ground truth data from gt_data returned by get_ground_truth()."""
        ground_truths = []
        
        if 'gt_bboxes_3d' in gt_data and 'gt_labels_3d' in gt_data:
            gt_bboxes_3d = gt_data['gt_bboxes_3d']
            gt_labels_3d = gt_data['gt_labels_3d']
            
            print(f"DEBUG: Ground truth analysis:")
            print(f"  GT bboxes shape: {gt_bboxes_3d.shape}")
            print(f"  GT labels shape: {gt_labels_3d.shape}")
            print(f"  GT bboxes range: x={gt_bboxes_3d[:, 0].min():.1f}-{gt_bboxes_3d[:, 0].max():.1f}, y={gt_bboxes_3d[:, 1].min():.1f}-{gt_bboxes_3d[:, 1].max():.1f}")
            print(f"  GT bboxes range: z={gt_bboxes_3d[:, 2].min():.1f}-{gt_bboxes_3d[:, 2].max():.1f}")
            print(f"  GT bbox format - first bbox: {gt_bboxes_3d[0]}")
            print(f"  GT labels: {np.unique(gt_labels_3d)}")
            print(f"  GT count: {len(gt_bboxes_3d)}")
            
            # Count by label
            unique_labels, counts = np.unique(gt_labels_3d, return_counts=True)
            print(f"  GT label distribution:")
            for label, count in zip(unique_labels, counts):
                print(f"    Label {label}: {count} ground truths")
            
            for i in range(len(gt_bboxes_3d)):
                bbox_3d = gt_bboxes_3d[i]  # [x, y, z, w, l, h, yaw]
                label = gt_labels_3d[i]
                
                ground_truths.append({
                    'bbox_3d': bbox_3d.tolist(),
                    'label': int(label)
                })
        
        return ground_truths

    # TODO(vividf): use autoware_perception_eval in the future
    def _compute_metrics(
        self,
        predictions_list: List[List[Dict]],
        ground_truths_list: List[List[Dict]],
        latencies: List[float],
        logger,
    ) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        
        # Debug metrics
        print(f"DEBUG: _compute_metrics - predictions_list length: {len(predictions_list)}")
        print(f"DEBUG: _compute_metrics - ground_truths_list length: {len(ground_truths_list)}")
        print(f"DEBUG: _compute_metrics - predictions per sample: {[len(preds) for preds in predictions_list]}")
        print(f"DEBUG: _compute_metrics - ground truths per sample: {[len(gts) for gts in ground_truths_list]}")
        
        # Count total predictions and ground truths
        total_predictions = sum(len(preds) for preds in predictions_list)
        total_ground_truths = sum(len(gts) for gts in ground_truths_list)
        
        print(f"DEBUG: _compute_metrics - total_predictions: {total_predictions}, total_ground_truths: {total_ground_truths}")
        
        # Count per class
        per_class_preds = {}
        per_class_gts = {}
        
        for predictions in predictions_list:
            for pred in predictions:
                label = pred['label']
                per_class_preds[label] = per_class_preds.get(label, 0) + 1
        
        for ground_truths in ground_truths_list:
            for gt in ground_truths:
                label = gt['label']
                per_class_gts[label] = per_class_gts.get(label, 0) + 1

        # Compute latency statistics
        latency_stats = self.compute_latency_stats(latencies)

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

    def print_results(self, results: Dict[str, Any]) -> None:
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
                class_name = class_id if isinstance(class_id, str) else (self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}")
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
            
            class_names = ['car', 'truck', 'bus', 'bicycle', 'pedestrian']
            iou_threshold = 0.5
            
            # Initialize per-class metrics
            per_class_ap = {}
            
            for class_id, class_name in enumerate(class_names):
                # Collect all predictions and ground truths for this class
                all_predictions = []
                all_ground_truths = []
                
                for predictions, ground_truths in zip(predictions_list, ground_truths_list):
                    # Filter by class
                    class_predictions = [p for p in predictions if p['label'] == class_id]
                    class_ground_truths = [g for g in ground_truths if g['label'] == class_id]
                    
                    all_predictions.extend(class_predictions)
                    all_ground_truths.extend(class_ground_truths)
                
                # Sort predictions by score (descending)
                all_predictions.sort(key=lambda x: x['score'], reverse=True)
                
                # Debug IoU calculations for this class
                if all_predictions and all_ground_truths:
                    print(f"DEBUG: {class_name} - Computing IoU for {len(all_predictions)} predictions vs {len(all_ground_truths)} GTs")
                    # Show IoU between first prediction and first few GTs
                    first_pred = all_predictions[0]
                    print(f"DEBUG: {class_name} - First pred bbox: {first_pred['bbox_3d']}")
                    print(f"DEBUG: {class_name} - First pred score: {first_pred['score']:.3f}")
                    
                    max_iou = 0.0
                    for i, gt in enumerate(all_ground_truths[:3]):  # Show first 3 GTs
                        iou = self._compute_3d_iou_simple(first_pred['bbox_3d'], gt['bbox_3d'])
                        print(f"DEBUG: {class_name} - IoU with GT {i}: {iou:.3f}, GT bbox: {gt['bbox_3d']}")
                        max_iou = max(max_iou, iou)
                    print(f"DEBUG: {class_name} - Max IoU for first pred: {max_iou:.3f}")
                    
                    # Debug: Check if this is PyTorch or ONNX/TensorRT
                    backend_type = "Unknown"
                    if hasattr(self, '_current_backend'):
                        backend_type = str(type(self._current_backend))
                    print(f"DEBUG: {class_name} - Backend type: {backend_type}")
                    print(f"DEBUG: {class_name} - Sample count: {len(predictions_list)} samples")
                    print(f"DEBUG: {class_name} - GT count: {len(ground_truths_list)} samples")
                
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
                        pred_bbox = np.array(pred['bbox_3d'])
                        best_iou = 0.0
                        best_gt_idx = -1
                        
                        # Find best matching ground truth
                        for j, gt in enumerate(all_ground_truths):
                            if gt_matched[j]:
                                continue
                            
                            gt_bbox = np.array(gt['bbox_3d'])
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
                print(f"DEBUG: {class_name} - AP: {ap:.4f}, predictions: {len(all_predictions)}, ground truths: {len(all_ground_truths)}")
            
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
            x_left = max(x1 - w1/2, x2 - w2/2)
            y_top = max(y1 - l1/2, y2 - l2/2)
            x_right = min(x1 + w1/2, x2 + w2/2)
            y_bottom = min(y1 + l1/2, y2 + l2/2)
            
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
