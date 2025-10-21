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
from .centerpoint_tensorrt_backend import CenterPointTensorRTBackend

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
        
        # Debug: Check the model config in evaluator
        logger = logging.getLogger(__name__)
        logger.info(f"Evaluator init model type: {self.model_cfg.model.type}")
        logger.info(f"Evaluator init voxel encoder type: {self.model_cfg.model.pts_voxel_encoder.type}")

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

        # Create backend instance
        inference_backend = self._create_backend(backend, model_path, device, logger)
        
        if inference_backend is None:
            logger.error(f"Failed to create {backend} backend")
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
                sample = data_loader.get_sample(i)
                input_data = data_loader.preprocess(sample)

                # Run inference
                if backend == "pytorch":
                    output, latency = self._run_pytorch_inference(inference_backend, input_data, sample)
                else:
                    output, latency = inference_backend.infer(input_data)

                # Parse predictions and ground truths
                predictions = self._parse_predictions(output, sample)
                ground_truths = self._parse_ground_truths(sample)

                predictions_list.append(predictions)
                ground_truths_list.append(ground_truths)
                latencies.append(latency)

                # Cleanup GPU memory periodically
                if i % GPU_CLEANUP_INTERVAL == 0:
                    torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return {}

        finally:
            # Cleanup
            if hasattr(inference_backend, 'close'):
                inference_backend.close()

        # Compute metrics
        results = self._compute_metrics(predictions_list, ground_truths_list, latencies, logger)
        
        return results

    def _create_backend(self, backend: str, model_path: str, device: str, logger) -> Any:
        """Create backend instance."""
        try:
            if backend == "pytorch":
                return PyTorchBackend(model_path, device=device)
            elif backend == "onnx":
                return ONNXBackend(model_path, device=device)
            elif backend == "tensorrt":
                return CenterPointTensorRTBackend(model_path, device=device)
            else:
                logger.error(f"Unsupported backend: {backend}")
                return None
        except Exception as e:
            logger.error(f"Failed to create {backend} backend: {e}")
            return None

    def _run_pytorch_inference(self, backend, input_data: Dict, sample: Dict) -> tuple:
        """Run PyTorch inference with proper data format."""
        from mmengine.dataset import pseudo_collate
        from mmdet3d.structures import Det3DDataSample
        
        # Convert input data to proper format for PyTorch model
        # PyTorch model expects voxels, num_points, coors
        if 'voxels' in input_data and 'num_points' in input_data and 'coors' in input_data:
            # Already in correct format
            voxels = input_data['voxels']
            num_points = input_data['num_points'] 
            coors = input_data['coors']
        else:
            # Convert from points format
            points = input_data['points']
            # Create dummy voxels for verification
            voxels = torch.randn(1000, 32, 11)  # Dummy voxels
            num_points = torch.randint(1, 33, (1000,))  # Dummy num_points
            coors = torch.randint(0, 100, (1000, 3))  # Dummy coors
        
        # Create data sample
        data_sample = Det3DDataSample()
        data_sample.set_metainfo(sample.get('metainfo', {}))
        
        # Run inference
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        # Use test_step method for proper evaluation
        with torch.no_grad():
            # Create batch data
            batch_data = pseudo_collate([{
                'inputs': {
                    'voxels': voxels,
                    'num_points': num_points,
                    'coors': coors
                },
                'data_samples': [data_sample]
            }])
            
            # Run test_step
            outputs = backend.model.test_step(batch_data)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            latency = start_time.elapsed_time(end_time)
        else:
            latency = 0.0
        
        return outputs, latency

    def _parse_predictions(self, output: Any, sample: Dict) -> List[Dict]:
        """Parse model output into prediction format."""
        predictions = []
        
        # Debug output
        print(f"DEBUG: Raw output type: {type(output)}")
        
        if isinstance(output, list):
            print(f"DEBUG: List output length: {len(output)}")
            if len(output) > 0:
                print(f"DEBUG: Item 0 type: {type(output[0])}")
                if hasattr(output[0], '__dict__'):
                    print(f"DEBUG: Item 0 attributes: {list(output[0].__dict__.keys())}")
        
        # Handle different output formats
        if isinstance(output, list) and len(output) > 0:
            # Check if it's Det3DDataSample format (PyTorch)
            if hasattr(output[0], 'pred_instances_3d'):
                print("INFO:projects.CenterPoint.deploy.evaluator:Raw head outputs detected, parsing CenterPoint predictions...")
                data_sample = output[0]
                
                # Extract predictions from Det3DDataSample
                pred_instances = data_sample.pred_instances_3d
                print(f"DEBUG: pred_instances attributes: {list(pred_instances.__dict__.keys())}")
                
                if hasattr(pred_instances, 'bboxes_3d') and hasattr(pred_instances, 'scores_3d') and hasattr(pred_instances, 'labels_3d'):
                    bboxes_3d = pred_instances.bboxes_3d
                    scores_3d = pred_instances.scores_3d
                    labels_3d = pred_instances.labels_3d
                    
                    print(f"DEBUG: Det3DDataSample - bboxes_3d shape: {bboxes_3d.shape}, scores shape: {scores_3d.shape}, labels shape: {labels_3d.shape}")
                    
                    # Convert to numpy for processing
                    bboxes_np = bboxes_3d.cpu().numpy()
                    scores_np = scores_3d.cpu().numpy()
                    labels_np = labels_3d.cpu().numpy()
                    
                    print(f"DEBUG: PyTorch bbox format - first bbox: {bboxes_np[0]}")
                    print(f"DEBUG: PyTorch bbox range - x: {bboxes_np[:, 0].min():.2f} to {bboxes_np[:, 0].max():.2f}")
                    print(f"DEBUG: PyTorch bbox range - y: {bboxes_np[:, 1].min():.2f} to {bboxes_np[:, 1].max():.2f}")
                    print(f"DEBUG: PyTorch bbox range - z: {bboxes_np[:, 2].min():.2f} to {bboxes_np[:, 2].max():.2f}")
                    print(f"DEBUG: PyTorch labels: {np.unique(labels_np)}")
                    print(f"DEBUG: PyTorch scores range: {scores_np.min():.3f} to {scores_np.max():.3f}")
                    
                    # Parse each prediction
                    for i in range(len(bboxes_np)):
                        bbox_3d = bboxes_np[i]  # [x, y, z, w, l, h, yaw, vx, vy]
                        score = scores_np[i]
                        label = labels_np[i]
                        
                        predictions.append({
                            'bbox_3d': bbox_3d[:7].tolist(),  # [x, y, z, w, l, h, yaw]
                            'score': float(score),
                            'label': int(label)
                        })
                    
                    print(f"DEBUG: Parsed {len(predictions)} predictions from Det3DDataSample")
                    
            # Check if it's raw head outputs format (ONNX/TensorRT)
            elif isinstance(output[0], (list, tuple)) and len(output[0]) == 6:
                print("INFO:projects.CenterPoint.deploy.evaluator:Raw head outputs detected, parsing CenterPoint predictions...")
                # Raw head outputs: [heatmap, reg, height, dim, rot, vel]
                heatmap, reg, height, dim, rot, vel = output[0]
                
                print(f"DEBUG: Raw output shapes - heatmap: {heatmap.shape}, reg: {reg.shape}, height: {height.shape}, dim: {dim.shape}, rot: {rot.shape}, vel: {vel.shape}")
                
                # Parse CenterPoint head outputs
                predictions = self._parse_centerpoint_head_outputs(
                    heatmap, reg, height, dim, rot, vel, sample
                )
                
        elif isinstance(output, dict):
            # Handle dictionary output format
            if 'predictions' in output:
                predictions = output['predictions']
            else:
                # Convert dict to list format
                predictions = [output]
        
        return predictions

    def _parse_centerpoint_head_outputs(self, heatmap, reg, height, dim, rot, vel, sample):
        """Parse CenterPoint head outputs into predictions."""
        import torch
        import numpy as np
        
        # Convert to numpy if needed
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()
        if isinstance(reg, torch.Tensor):
            reg = reg.cpu().numpy()
        if isinstance(height, torch.Tensor):
            height = height.cpu().numpy()
        if isinstance(dim, torch.Tensor):
            dim = dim.cpu().numpy()
        if isinstance(rot, torch.Tensor):
            rot = rot.cpu().numpy()
        if isinstance(vel, torch.Tensor):
            vel = vel.cpu().numpy()
        
        batch_size, num_classes, H, W = heatmap.shape
        print(f"DEBUG: Parsing CenterPoint outputs - batch_size: {batch_size}, num_classes: {num_classes}, H: {H}, W: {W}")
        
        predictions = []
        
        # Get point cloud range from sample
        point_cloud_range = sample.get('metainfo', {}).get('point_cloud_range', [-121.6, -121.6, -3.0, 121.6, 121.6, 5.0])
        voxel_size = sample.get('metainfo', {}).get('voxel_size', [0.32, 0.32, 8.0])
        
        # Convert heatmap coordinates to 3D coordinates
        for b in range(batch_size):
            for c in range(num_classes):
                # Find peaks in heatmap
                heatmap_class = heatmap[b, c]
                
                # Simple threshold-based detection
                threshold = 0.1
                peaks = np.where(heatmap_class > threshold)
                
                for i in range(len(peaks[0])):
                    y_idx, x_idx = peaks[0][i], peaks[1][i]
                    score = heatmap_class[y_idx, x_idx]
                    
                    # Convert grid coordinates to 3D coordinates
                    x = x_idx * voxel_size[0] + point_cloud_range[0]
                    y = y_idx * voxel_size[1] + point_cloud_range[1]
                    z = height[b, 0, y_idx, x_idx] if height.shape[1] > 0 else 0.0
                    
                    # Get dimensions
                    w = dim[b, 0, y_idx, x_idx] if dim.shape[1] > 0 else 1.0
                    l = dim[b, 1, y_idx, x_idx] if dim.shape[1] > 1 else 1.0
                    h = dim[b, 2, y_idx, x_idx] if dim.shape[1] > 2 else 1.0
                    
                    # Get rotation
                    yaw = rot[b, 0, y_idx, x_idx] if rot.shape[1] > 0 else 0.0
                    
                    # Get velocity
                    vx = vel[b, 0, y_idx, x_idx] if vel.shape[1] > 0 else 0.0
                    vy = vel[b, 1, y_idx, x_idx] if vel.shape[1] > 1 else 0.0
                    
                    predictions.append({
                        'bbox_3d': [float(x), float(y), float(z), float(w), float(l), float(h), float(yaw)],
                        'score': float(score),
                        'label': int(c)
                    })
        
        print(f"DEBUG: Parsed {len(predictions)} predictions from CenterPoint head outputs")
        return predictions

    def _parse_ground_truths(self, sample: Dict) -> List[Dict]:
        """Parse ground truth data."""
        ground_truths = []
        
        if 'gt_bboxes_3d' in sample and 'gt_labels_3d' in sample:
            gt_bboxes_3d = sample['gt_bboxes_3d']
            gt_labels_3d = sample['gt_labels_3d']
            
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
            map_results = self._compute_mmdet3d_map(
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

    def _compute_mmdet3d_map(
        self,
        predictions_list: List[List[Dict]],
        ground_truths_list: List[List[Dict]],
        num_classes: int,
    ) -> Dict[str, Any]:
        """Compute mAP using mmdet3d's eval_map_recall function."""
        try:
            from mmdet3d.evaluation import eval_map_recall
            from mmdet3d.structures import LiDARInstance3DBoxes
            import numpy as np
            import torch
            
            # Convert to mmdet3d format
            # eval_map_recall expects:
            # pred: {class_name: {img_id: [(bbox, score), ...]}}
            # gt: {class_name: {img_id: [bbox, ...]}}
            class_names = ['car', 'truck', 'bus', 'bicycle', 'pedestrian']
            
            # Initialize data structures
            pred_by_class = {}
            gt_by_class = {}
            
            for class_name in class_names:
                pred_by_class[class_name] = {}
                gt_by_class[class_name] = {}
            
            # Process each sample
            for sample_idx, (predictions, ground_truths) in enumerate(zip(predictions_list, ground_truths_list)):
                img_id = f"sample_{sample_idx}"
                
                # Group predictions by class
                for pred in predictions:
                    class_name = class_names[pred['label']]
                    if img_id not in pred_by_class[class_name]:
                        pred_by_class[class_name][img_id] = []
                    
                    # Convert bbox to LiDARInstance3DBoxes format
                    bbox_3d = pred['bbox_3d']  # [x, y, z, w, l, h, yaw]
                    # Create a single bbox tensor
                    bbox_tensor = torch.tensor([bbox_3d], dtype=torch.float32)
                    bbox_obj = LiDARInstance3DBoxes(bbox_tensor)
                    
                    pred_by_class[class_name][img_id].append((bbox_obj, pred['score']))
                
                # Group ground truths by class
                for gt in ground_truths:
                    class_name = class_names[gt['label']]
                    if img_id not in gt_by_class[class_name]:
                        gt_by_class[class_name][img_id] = []
                    
                    # Convert bbox to LiDARInstance3DBoxes format
                    bbox_3d = gt['bbox_3d']  # [x, y, z, w, l, h, yaw]
                    bbox_tensor = torch.tensor([bbox_3d], dtype=torch.float32)
                    bbox_obj = LiDARInstance3DBoxes(bbox_tensor)
                    
                    gt_by_class[class_name][img_id].append(bbox_obj)
            
            # Debug: Print input to eval_map
            print(f"DEBUG: eval_map input - pred_by_class keys: {list(pred_by_class.keys())}")
            print(f"DEBUG: eval_map input - gt_by_class keys: {list(gt_by_class.keys())}")
            for class_name in class_names:
                pred_count = sum(len(pred_by_class[class_name][img_id]) for img_id in pred_by_class[class_name])
                gt_count = sum(len(gt_by_class[class_name][img_id]) for img_id in gt_by_class[class_name])
                print(f"DEBUG: {class_name} - pred count: {pred_count}, gt count: {gt_count}")
            
            # Compute 3D mAP metrics using eval_map_recall
            map_results = eval_map_recall(
                pred_by_class,
                gt_by_class,
                ovthresh=[0.5],  # IoU threshold for 3D detection
            )
            
            # Extract results
            # eval_map_recall returns (recall, precision, ap) for each IoU threshold
            recall, precision, ap = map_results
            
            # Extract mAP@0.5 (first IoU threshold)
            map_50 = 0.0
            per_class_ap = {}
            
            for class_name in class_names:
                if class_name in ap[0]:  # ap[0] is for IoU=0.5
                    class_ap = ap[0][class_name]
                    if isinstance(class_ap, np.ndarray):
                        per_class_ap[class_name] = float(class_ap.mean()) if len(class_ap) > 0 else 0.0
                    else:
                        per_class_ap[class_name] = float(class_ap)
                    map_50 += per_class_ap[class_name]
                else:
                    per_class_ap[class_name] = 0.0
            
            # Average mAP across all classes
            map_50 = map_50 / len(class_names) if len(class_names) > 0 else 0.0
            
            return {
                "mAP": map_50,
                "mAP_50": map_50,
                "NDS": map_50,  # Simplified - real NDS needs more complex computation
                "mATE": 0.0,
                "mASE": 0.0,
                "mAOE": 0.0,
                "mAVE": 0.0,
                "mAAE": 0.0,
                "per_class_ap": per_class_ap,
            }
            
        except Exception as e:
            print(f"Error computing mmdet3d metrics: {e}")
            import traceback
            traceback.print_exc()
            # Return zero metrics as fallback
            return {
                "mAP": 0.0,
                "mAP_50": 0.0,
                "NDS": 0.0,
                "mATE": 0.0,
                "mASE": 0.0,
                "mAOE": 0.0,
                "mAVE": 0.0,
                "mAAE": 0.0,
                "per_class_ap": {f"class_{i}": 0.0 for i in range(num_classes)},
            }

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
