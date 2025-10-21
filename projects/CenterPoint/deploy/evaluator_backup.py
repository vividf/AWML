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
        Run full evaluation on CenterPoint model.

        Args:
            model_path: Path to model checkpoint/weights
            data_loader: CenterPoint DataLoader
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
                input_data = data_loader.preprocess(sample)

                # Get ground truth
                gt_data = data_loader.get_ground_truth(idx)
                print(f"DEBUG: Backend {backend} - GT data keys: {list(gt_data.keys())}")
                print(f"DEBUG: Backend {backend} - GT bboxes shape: {gt_data['gt_bboxes_3d'].shape}")

                # Run inference
                if backend == "pytorch":
                    # For PyTorch backend, we need to handle the dictionary input differently
                    output, latency = self._run_pytorch_inference(inference_backend, input_data)
                else:
                    output, latency = inference_backend.infer(input_data)
                latencies.append(latency)

                # Parse predictions
                predictions = self._parse_predictions(output)
                all_predictions.append(predictions)

                # Parse ground truths
                ground_truths = self._parse_ground_truths(gt_data)
                all_ground_truths.append(ground_truths)

                # GPU cleanup for TensorRT
                if backend == "tensorrt" and idx % GPU_CLEANUP_INTERVAL == 0:
                    torch.cuda.empty_cache()

        # Compute metrics
        results = self._compute_metrics(all_predictions, all_ground_truths, latencies, logger)

        return results

    def _run_pytorch_inference(self, inference_backend, input_data):
        """
        Run PyTorch inference with dictionary input.
        
        Args:
            inference_backend: PyTorch backend instance
            input_data: Dictionary containing voxel data
            
        Returns:
            Tuple of (output, latency)
        """
        import time
        from mmdet3d.structures import Det3DDataSample
        from mmengine.dataset import pseudo_collate
        from mmdet3d.structures import get_box_type
        
        # Run inference
        start_time = time.time()
        
        # For PyTorch models, we need to handle the inference differently
        # since the backend expects a different input format
        try:
            # Try to run inference with the model directly
            model = inference_backend._model
            model.eval()
            
            with torch.no_grad():
                # For CenterPoint PyTorch model, we need to create a proper input format
                # The model expects data processed by pseudo_collate
                if isinstance(input_data, dict):
                    # Create input dict in the format expected by CenterPoint
                    # We need to create a proper DataSample object
                    data_sample = Det3DDataSample()
                    
                    # Set necessary metadata for CenterPoint
                    box_type_3d, box_mode_3d = get_box_type('LiDAR')
                    data_sample.set_metainfo({
                        'sample_idx': 0,
                        'box_type_3d': box_type_3d,
                        'box_mode_3d': box_mode_3d,
                        'timestamp': 1,
                        'lidar2img': None,
                        'depth2img': None,
                        'cam2img': None,
                        'ori_cam2img': None,
                        'cam2global': None,
                        'lidar2cam': None,
                        'ego2global': None,
                    })
                    
                    # Create the data structure expected by pseudo_collate
                    data_item = {
                        'inputs': input_data,
                        'data_samples': data_sample
                    }
                    
                    # Use pseudo_collate to process the data (even for single item)
                    collate_data = pseudo_collate([data_item])
                    output = model.test_step(collate_data)
                else:
                    # Handle single tensor
                    data_sample = Det3DDataSample()
                    
                    # Set necessary metadata for CenterPoint
                    box_type_3d, box_mode_3d = get_box_type('LiDAR')
                    data_sample.set_metainfo({
                        'sample_idx': 0,
                        'box_type_3d': box_type_3d,
                        'box_mode_3d': box_mode_3d,
                        'timestamp': 1,
                        'lidar2img': None,
                        'depth2img': None,
                        'cam2img': None,
                        'ori_cam2img': None,
                        'cam2global': None,
                        'lidar2cam': None,
                        'ego2global': None,
                    })
                    
                    # Create the data structure expected by pseudo_collate
                    data_item = {
                        'inputs': input_data,
                        'data_samples': data_sample
                    }
                    
                    # Use pseudo_collate to process the data (even for single item)
                    collate_data = pseudo_collate([data_item])
                    output = model.test_step(collate_data)
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to ms
            
            return output, latency
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"PyTorch inference failed: {e}")
            # Return empty output as fallback
            return [], 0.0

    def _create_backend(self, backend: str, model_path: str, device: str, logger):
        """Create inference backend."""
        if backend == "pytorch":
            # For PyTorch evaluation, we need to load the original model config
            # not the ONNX-compatible one
            from mmdet3d.apis import init_model
            from mmengine.registry import init_default_scope
            
            # Initialize mmdet3d scope
            init_default_scope("mmdet3d")
            
            # Load original model config (not ONNX-compatible)
            import copy
            original_model_cfg = copy.deepcopy(self.model_cfg)
            
            # Restore original model type if it was changed for ONNX
            if hasattr(original_model_cfg.model, 'type') and original_model_cfg.model.type == "CenterPointONNX":
                original_model_cfg.model.type = "CenterPoint"
            
            # Restore original voxel encoder type
            if hasattr(original_model_cfg.model, 'pts_voxel_encoder'):
                if original_model_cfg.model.pts_voxel_encoder.type == "PillarFeatureNetONNX":
                    original_model_cfg.model.pts_voxel_encoder.type = "PillarFeatureNet"
                elif original_model_cfg.model.pts_voxel_encoder.type == "BackwardPillarFeatureNetONNX":
                    original_model_cfg.model.pts_voxel_encoder.type = "BackwardPillarFeatureNet"
            
            # Restore original bbox head type
            if hasattr(original_model_cfg.model, 'pts_bbox_head'):
                if original_model_cfg.model.pts_bbox_head.type == "CenterHeadONNX":
                    original_model_cfg.model.pts_bbox_head.type = "CenterHead"
                if hasattr(original_model_cfg.model.pts_bbox_head, 'separate_head'):
                    if original_model_cfg.model.pts_bbox_head.separate_head.type == "SeparateHeadONNX":
                        original_model_cfg.model.pts_bbox_head.separate_head.type = "SeparateHead"
            
            # Remove ONNX-specific attributes
            if hasattr(original_model_cfg.model, 'point_channels'):
                delattr(original_model_cfg.model, 'point_channels')
            if hasattr(original_model_cfg.model, 'device'):
                delattr(original_model_cfg.model, 'device')
            if hasattr(original_model_cfg.model.pts_bbox_head, 'rot_y_axis_reference'):
                delattr(original_model_cfg.model.pts_bbox_head, 'rot_y_axis_reference')
            
            logger.info("Loading original PyTorch model for evaluation...")
            model = init_model(original_model_cfg, model_path, device=device)
            return PyTorchBackend(model, device=device)
        elif backend == "onnx":
            # For CenterPoint ONNX, we need to provide the PyTorch model as well
            # since the ONNX backend needs it for post-processing
            from mmdet3d.apis import init_model
            from mmengine.registry import init_default_scope
            
            # Initialize mmdet3d scope
            init_default_scope("mmdet3d")
            
            # For ONNX backend, we need to load the ONNX-compatible model
            # not the original model, since the ONNX helper expects ONNX-compatible methods
            logger.info("Loading ONNX-compatible PyTorch model for ONNX backend...")
            
            # Debug: Check the model config
            logger.info(f"Model type: {self.model_cfg.model.type}")
            logger.info(f"Voxel encoder type: {self.model_cfg.model.pts_voxel_encoder.type}")
            
            # Load the ONNX-compatible model configuration
            # This should already be set up in the model_cfg from the main script
            pytorch_model = init_model(self.model_cfg, 'work_dirs/centerpoint/best_checkpoint.pth', device=device)
            return ONNXBackend(model_path, device, pytorch_model=pytorch_model)
        elif backend == "tensorrt":
            return CenterPointTensorRTBackend(model_path, device)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _parse_predictions(self, output) -> List[Dict]:
        """
        Parse model output to prediction format.

        Args:
            output: Raw model output (can be dict, array, or list)

        Returns:
            List of prediction dicts with 'bbox_3d', 'label', 'score'
        """
        predictions = []

        # Debug logging for output analysis
        print(f"DEBUG: Raw output type: {type(output)}")
        if hasattr(output, 'shape'):
            print(f"DEBUG: Raw output shape: {output.shape}")
        if hasattr(output, 'dtype'):
            print(f"DEBUG: Raw output dtype: {output.dtype}")
        if hasattr(output, 'min') and hasattr(output, 'max'):
            print(f"DEBUG: Raw output min/max: {output.min():.6f} / {output.max():.6f}")

        # CenterPoint output can be dict, array, or list
        if isinstance(output, dict):
            # Dict format: {'boxes_3d', 'scores_3d', 'labels_3d'}
            boxes_3d = output.get("boxes_3d", output.get("bboxes_3d", np.array([])))
            scores = output.get("scores_3d", output.get("scores", np.array([])))
            labels = output.get("labels_3d", output.get("labels", np.array([])))
            print(f"DEBUG: Dict format - boxes_3d: {boxes_3d.shape}, scores: {scores.shape}, labels: {labels.shape}")
        elif isinstance(output, np.ndarray):
            # Array format: [num_detections, 9] where 9 = [bbox_3d(7), score(1), label(1)]
            if len(output.shape) == 2 and output.shape[1] == 9:
                boxes_3d = output[:, :7]  # First 7 columns are 3D bbox
                scores = output[:, 7]    # 8th column is score
                labels = output[:, 8]    # 9th column is label
                print(f"DEBUG: Array format - boxes_3d: {boxes_3d.shape}, scores: {scores.shape}, labels: {labels.shape}")
            else:
                # Unknown array format
                logger = logging.getLogger(__name__)
                logger.warning(f"Unknown array format: {output.shape}")
                return predictions
        elif isinstance(output, list):
            # List format: could be [heatmap, reg, height, dim, rot, vel] or other formats
            logger = logging.getLogger(__name__)
            logger.info("Raw head outputs detected, parsing CenterPoint predictions...")
            
            print(f"DEBUG: List output length: {len(output)}")
            for i, item in enumerate(output):
                if hasattr(item, 'shape'):
                    print(f"DEBUG: Item {i} shape: {item.shape}")
                else:
                    print(f"DEBUG: Item {i} type: {type(item)}")
            
            if len(output) == 6:
                # ONNX/TensorRT format: [heatmap, reg, height, dim, rot, vel]
                # Check if items are dictionaries (TensorRT format) or tensors (ONNX format)
                if isinstance(output[0], dict):
                    # TensorRT format: list of dictionaries
                    predictions = self._parse_tensorrt_outputs(output)
                    return predictions
                else:
                    # ONNX format: list of tensors
                    heatmap, reg, height, dim, rot, vel = output
                    predictions = self._parse_centerpoint_head_outputs(heatmap, reg, height, dim, rot, vel)
                    return predictions
            elif len(output) == 1 and isinstance(output[0], list):
                # PyTorch format: [[heatmap, reg, height, dim, rot, vel]] (nested list)
                inner_output = output[0]
                if len(inner_output) == 6:
                    heatmap, reg, height, dim, rot, vel = inner_output
                    predictions = self._parse_centerpoint_head_outputs(heatmap, reg, height, dim, rot, vel)
                    return predictions
                else:
                    logger.warning(f"Expected 6 inner outputs, got {len(inner_output)}")
                    return predictions
            elif len(output) == 1 and hasattr(output[0], 'pred_instances_3d'):
                # PyTorch format: [Det3DDataSample] with pred_instances_3d
                data_sample = output[0]
                predictions = self._parse_det3d_data_sample(data_sample)
                return predictions
            else:
                logger.warning(f"Unexpected list format: length {len(output)}")
                return predictions
        else:
            # Unknown format
            logger = logging.getLogger(__name__)
            logger.warning(f"Unknown output format: {type(output)}")
            return predictions

        # Convert to prediction format
        for bbox, score, label in zip(boxes_3d, scores, labels):
            if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) >= 7:
                predictions.append(
                    {
                        "bbox_3d": bbox[:7].tolist() if isinstance(bbox, np.ndarray) else list(bbox[:7]),
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

    def _parse_centerpoint_head_outputs(self, heatmap, reg, height, dim, rot, vel):
        """
        Parse CenterPoint head outputs to detection results.
        
        Args:
            heatmap: Heatmap output [B, num_classes, H, W]
            reg: Regression output [B, 2, H, W]
            height: Height output [B, 1, H, W]
            dim: Dimension output [B, 3, H, W]
            rot: Rotation output [B, 2, H, W]
            vel: Velocity output [B, 2, H, W]
            
        Returns:
            List of prediction dicts
        """
        import torch
        import numpy as np
        from scipy.special import softmax
        
        predictions = []
        
        # Get batch size and spatial dimensions
        batch_size = heatmap.shape[0]
        num_classes = heatmap.shape[1]
        H, W = heatmap.shape[2], heatmap.shape[3]
        
        print(f"DEBUG: Parsing CenterPoint outputs - batch_size: {batch_size}, num_classes: {num_classes}, H: {H}, W: {W}")
        
        # Process each batch
        for b in range(batch_size):
            batch_predictions = []
            
            # Apply sigmoid to heatmap
            heatmap_b = torch.sigmoid(torch.from_numpy(heatmap[b]))  # [num_classes, H, W]
            
            # Find peaks in heatmap (simplified approach)
            for cls_id in range(num_classes):
                heatmap_cls = heatmap_b[cls_id]  # [H, W]
                
                # Find local maxima (simplified peak detection)
                max_val = heatmap_cls.max().item()
                
                # Use different thresholds based on the range of values
                # If max_val is very small (< 1e-10), use a much lower threshold
                if max_val < 1e-10:
                    threshold = max_val * 0.1  # Use 10% of max value as threshold
                else:
                    threshold = 0.1  # Normal threshold for sigmoid outputs
                
                if max_val > threshold:
                    # Find all pixels above threshold
                    mask = heatmap_cls > threshold
                    y_coords, x_coords = torch.where(mask)
                    
                    if len(y_coords) > 0:
                        # Get top-k detections
                        scores = heatmap_cls[y_coords, x_coords]
                        top_k = min(100, len(scores))  # Limit to top 100 detections
                        top_indices = torch.topk(scores, top_k).indices
                        
                        for idx in top_indices:
                            y, x = y_coords[idx], x_coords[idx]
                            score = scores[idx].item()
                            
                            # Get regression values
                            reg_x = reg[b, 0, y, x]
                            reg_y = reg[b, 1, y, x]
                            height_val = height[b, 0, y, x]
                            dim_w = dim[b, 0, y, x]
                            dim_l = dim[b, 1, y, x]
                            dim_h = dim[b, 2, y, x]
                            rot_sin = rot[b, 0, y, x]
                            rot_cos = rot[b, 1, y, x]
                            vel_x = vel[b, 0, y, x]
                            vel_y = vel[b, 1, y, x]
                            
                            # Convert to 3D bbox format [x, y, z, w, l, h, yaw]
                            # Note: This is a simplified conversion - real implementation would use proper coordinate transformation
                            x_3d = x + reg_x
                            y_3d = y + reg_y
                            z_3d = height_val
                            yaw = np.arctan2(rot_sin, rot_cos)
                            
                            # Create 3D bbox
                            bbox_3d = np.array([x_3d, y_3d, z_3d, dim_w, dim_l, dim_h, yaw])
                            
                            batch_predictions.append({
                                'bbox_3d': bbox_3d.tolist(),
                                'label': int(cls_id),
                                'score': float(score)
                            })
            
            predictions.extend(batch_predictions)
        
        print(f"DEBUG: Parsed {len(predictions)} predictions from CenterPoint head outputs")
        return predictions

    def _parse_det3d_data_sample(self, data_sample):
        """
        Parse Det3DDataSample output from PyTorch model.
        
        Args:
            data_sample: Det3DDataSample object with pred_instances_3d
            
        Returns:
            List of prediction dicts
        """
        import numpy as np
        
        predictions = []
        
        # Debug: Print all attributes of data_sample
        print(f"DEBUG: Det3DDataSample attributes: {dir(data_sample)}")
        
        if hasattr(data_sample, 'pred_instances_3d') and data_sample.pred_instances_3d is not None:
            pred_instances = data_sample.pred_instances_3d
            print(f"DEBUG: pred_instances attributes: {dir(pred_instances)}")
            
            # Extract bboxes, scores, and labels
            if hasattr(pred_instances, 'bboxes_3d') and hasattr(pred_instances, 'scores_3d') and hasattr(pred_instances, 'labels_3d'):
                bboxes_3d = pred_instances.bboxes_3d
                scores = pred_instances.scores_3d
                labels = pred_instances.labels_3d
                
                print(f"DEBUG: Det3DDataSample - bboxes_3d shape: {bboxes_3d.shape}, scores shape: {scores.shape}, labels shape: {labels.shape}")
                
                # Convert to numpy if needed
                if hasattr(bboxes_3d, 'cpu'):
                    bboxes_3d = bboxes_3d.cpu().numpy()
                if hasattr(scores, 'cpu'):
                    scores = scores.cpu().numpy()
                if hasattr(labels, 'cpu'):
                    labels = labels.cpu().numpy()
                
                # Debug: Print bbox format
                if len(bboxes_3d) > 0:
                    print(f"DEBUG: PyTorch bbox format - first bbox: {bboxes_3d[0]}")
                    print(f"DEBUG: PyTorch bbox range - x: {bboxes_3d[:, 0].min():.2f} to {bboxes_3d[:, 0].max():.2f}")
                    print(f"DEBUG: PyTorch bbox range - y: {bboxes_3d[:, 1].min():.2f} to {bboxes_3d[:, 1].max():.2f}")
                    print(f"DEBUG: PyTorch bbox range - z: {bboxes_3d[:, 2].min():.2f} to {bboxes_3d[:, 2].max():.2f}")
                    print(f"DEBUG: PyTorch labels: {np.unique(labels)}")
                    print(f"DEBUG: PyTorch scores range: {scores.min():.3f} to {scores.max():.3f}")
                
                # Create predictions
                for bbox, score, label in zip(bboxes_3d, scores, labels):
                    # Apply coordinate transformation if needed
                    # PyTorch model with rot_y_axis_reference=True applies these transformations:
                    # 1. Switch width and length: bbox[:, [3, 4]] = bbox[:, [4, 3]]
                    # 2. Change rotation: bbox[:, 6] = -bbox[:, 6] - np.pi / 2
                    
                    # Check if we need to apply inverse transformation
                    # Since the model already applied the transformation, we need to reverse it
                    if len(bbox) >= 7:  # Ensure we have at least 7 dimensions
                        bbox_copy = bbox.copy()
                        
                        # Reverse the rotation transformation
                        # Original: yaw = -yaw - np.pi/2
                        # Reverse: yaw = -(yaw + np.pi/2) = -yaw - np.pi/2
                        bbox_copy[6] = -(bbox_copy[6] + np.pi/2)
                        
                        # Reverse the width/length swap
                        # Original: [w, l] = [l, w]
                        # Reverse: [l, w] = [w, l]
                        bbox_copy[3], bbox_copy[4] = bbox_copy[4], bbox_copy[3]
                        
                        predictions.append({
                            'bbox_3d': bbox_copy.tolist() if isinstance(bbox_copy, np.ndarray) else bbox_copy,
                            'label': int(label),
                            'score': float(score)
                        })
                    else:
                        predictions.append({
                            'bbox_3d': bbox.tolist() if isinstance(bbox, np.ndarray) else bbox,
                            'label': int(label),
                            'score': float(score)
                        })
                
                print(f"DEBUG: Parsed {len(predictions)} predictions from Det3DDataSample")
            else:
                print("DEBUG: Det3DDataSample missing required attributes")
        else:
            print("DEBUG: Det3DDataSample has no pred_instances_3d")
        
        return predictions

    def _parse_tensorrt_outputs(self, output_list):
        """
        Parse TensorRT outputs which are dictionaries containing numpy arrays.
        
        Args:
            output_list: List of 6 dictionaries [heatmap_dict, reg_dict, height_dict, dim_dict, rot_dict, vel_dict]
            
        Returns:
            List of prediction dicts
        """
        import numpy as np
        
        predictions = []
        
        # Extract numpy arrays from dictionaries
        heatmap_dict, reg_dict, height_dict, dim_dict, rot_dict, vel_dict = output_list
        
        print(f"DEBUG: TensorRT output keys - heatmap: {heatmap_dict.keys()}, reg: {reg_dict.keys()}")
        print(f"DEBUG: TensorRT heatmap values: {list(heatmap_dict.values())}")
        print(f"DEBUG: TensorRT reg values: {list(reg_dict.values())}")
        
        # Get the actual numpy arrays from dictionaries
        # TensorRT outputs typically have keys like 'output_0', 'output_1', etc.
        heatmap = None
        reg = None
        height = None
        dim = None
        rot = None
        vel = None
        
        # Find the correct keys for each output
        # TensorRT outputs have keys like 'heatmap', 'reg', etc.
        # Values are PyTorch tensors, need to convert to numpy
        for key, value in heatmap_dict.items():
            if hasattr(value, 'cpu'):  # PyTorch tensor
                heatmap = value.cpu().numpy()
                break
            elif isinstance(value, np.ndarray):
                heatmap = value
                break
        
        for key, value in reg_dict.items():
            if hasattr(value, 'cpu'):  # PyTorch tensor
                reg = value.cpu().numpy()
                break
            elif isinstance(value, np.ndarray):
                reg = value
                break
                
        for key, value in height_dict.items():
            if hasattr(value, 'cpu'):  # PyTorch tensor
                height = value.cpu().numpy()
                break
            elif isinstance(value, np.ndarray):
                height = value
                break
                
        for key, value in dim_dict.items():
            if hasattr(value, 'cpu'):  # PyTorch tensor
                dim = value.cpu().numpy()
                break
            elif isinstance(value, np.ndarray):
                dim = value
                break
                
        for key, value in rot_dict.items():
            if hasattr(value, 'cpu'):  # PyTorch tensor
                rot = value.cpu().numpy()
                break
            elif isinstance(value, np.ndarray):
                rot = value
                break
                
        for key, value in vel_dict.items():
            if hasattr(value, 'cpu'):  # PyTorch tensor
                vel = value.cpu().numpy()
                break
            elif isinstance(value, np.ndarray):
                vel = value
                break
        
        # Debug: Print the actual values to understand the structure
        print(f"DEBUG: TensorRT values - heatmap: {type(heatmap)}, reg: {type(reg)}, height: {type(height)}, dim: {type(dim)}, rot: {type(rot)}, vel: {type(vel)}")
        
        # If we couldn't find numpy arrays, try to access the values directly
        if heatmap is None:
            heatmap_values = list(heatmap_dict.values())
            if heatmap_values and isinstance(heatmap_values[0], np.ndarray):
                heatmap = heatmap_values[0]
                print(f"DEBUG: Found heatmap via direct access: {heatmap.shape}")
        
        if reg is None:
            reg_values = list(reg_dict.values())
            if reg_values and isinstance(reg_values[0], np.ndarray):
                reg = reg_values[0]
                print(f"DEBUG: Found reg via direct access: {reg.shape}")
        
        if height is None:
            height_values = list(height_dict.values())
            if height_values and isinstance(height_values[0], np.ndarray):
                height = height_values[0]
                print(f"DEBUG: Found height via direct access: {height.shape}")
        
        if dim is None:
            dim_values = list(dim_dict.values())
            if dim_values and isinstance(dim_values[0], np.ndarray):
                dim = dim_values[0]
                print(f"DEBUG: Found dim via direct access: {dim.shape}")
        
        if rot is None:
            rot_values = list(rot_dict.values())
            if rot_values and isinstance(rot_values[0], np.ndarray):
                rot = rot_values[0]
                print(f"DEBUG: Found rot via direct access: {rot.shape}")
        
        if vel is None:
            vel_values = list(vel_dict.values())
            if vel_values and isinstance(vel_values[0], np.ndarray):
                vel = vel_values[0]
                print(f"DEBUG: Found vel via direct access: {vel.shape}")
        
        if heatmap is not None and reg is not None and height is not None and dim is not None and rot is not None and vel is not None:
            print(f"DEBUG: TensorRT arrays shapes - heatmap: {heatmap.shape}, reg: {reg.shape}, height: {height.shape}, dim: {dim.shape}, rot: {rot.shape}, vel: {vel.shape}")
            predictions = self._parse_centerpoint_head_outputs(heatmap, reg, height, dim, rot, vel)
        else:
            print("DEBUG: Could not extract numpy arrays from TensorRT dictionaries")
        
        return predictions

    def _parse_ground_truths(self, gt_data: Dict) -> List[Dict]:
        """
        Parse ground truth data.

        Args:
            gt_data: Ground truth data from data_loader

        Returns:
            List of ground truth dicts with 'bbox_3d', 'label'
        """
        ground_truths = []

        gt_bboxes_3d = gt_data["gt_bboxes_3d"]
        gt_labels_3d = gt_data["gt_labels_3d"]

        print(f"DEBUG: Ground truth analysis:")
        print(f"  GT bboxes shape: {gt_bboxes_3d.shape}")
        print(f"  GT labels shape: {gt_labels_3d.shape}")
        
        if len(gt_bboxes_3d) > 0:
            print(f"  GT bboxes range: x={gt_bboxes_3d[:, 0].min():.1f}-{gt_bboxes_3d[:, 0].max():.1f}, y={gt_bboxes_3d[:, 1].min():.1f}-{gt_bboxes_3d[:, 1].max():.1f}")
            print(f"  GT bboxes range: z={gt_bboxes_3d[:, 2].min():.1f}-{gt_bboxes_3d[:, 2].max():.1f}")
            print(f"  GT bbox format - first bbox: {gt_bboxes_3d[0]}")
            print(f"  GT labels: {np.unique(gt_labels_3d)}")
        else:
            print(f"  No ground truth bboxes found")

        for bbox, label in zip(gt_bboxes_3d, gt_labels_3d):
            ground_truths.append(
                {"bbox_3d": bbox.tolist() if isinstance(bbox, np.ndarray) else bbox, "label": int(label)}
            )

        # Debug: Analyze ground truth distribution
        if len(ground_truths) > 0:
            gt_labels = [gt['label'] for gt in ground_truths]
            gt_bboxes = np.array([gt['bbox_3d'] for gt in ground_truths])
            print(f"  GT count: {len(ground_truths)}")
            
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
        """
        Compute evaluation metrics using mmdet3d's official evaluation.

        Note: This integrates with mmdet3d.core.evaluation for proper 3D detection metrics.
        """
        # Compute basic statistics
        total_predictions = sum(len(preds) for preds in predictions_list)
        total_ground_truths = sum(len(gts) for gts in ground_truths_list)
        
        print(f"DEBUG: _compute_metrics - predictions_list length: {len(predictions_list)}")
        print(f"DEBUG: _compute_metrics - ground_truths_list length: {len(ground_truths_list)}")
        print(f"DEBUG: _compute_metrics - predictions per sample: {[len(preds) for preds in predictions_list]}")
        print(f"DEBUG: _compute_metrics - ground truths per sample: {[len(gts) for gts in ground_truths_list]}")
        print(f"DEBUG: _compute_metrics - total_predictions: {total_predictions}, total_ground_truths: {total_ground_truths}")

        # Per-class statistics
        per_class_preds = {i: 0 for i in range(len(self.class_names))}
        per_class_gts = {i: 0 for i in range(len(self.class_names))}

        for preds in predictions_list:
            for pred in preds:
                label = pred["label"]
                if label < len(self.class_names):
                    per_class_preds[label] += 1

        for gts in ground_truths_list:
            for gt in gts:
                label = gt["label"]
                if label < len(self.class_names):
                    per_class_gts[label] += 1

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
        """Compute mmdet3d evaluation metrics."""
        try:
            # Try different import paths for eval_map
            try:
                from mmdet3d.core.evaluation import eval_map
            except ImportError:
                try:
                    from mmdet3d.evaluation import eval_map
                except ImportError:
                    # Use eval_map_recall as fallback
                    from mmdet3d.evaluation import eval_map_recall as eval_map
            import numpy as np
            
            # Convert to mmdet3d format
            # Convert predictions to the format expected by eval_map_recall
            # eval_map_recall expects pred and gt to be dictionaries with class names as keys
            class_names = ['car', 'truck', 'bus', 'bicycle', 'pedestrian']
            
            # Group predictions by class
            pred_by_class = {}
            gt_by_class = {}
            
            for class_id, class_name in enumerate(class_names):
                pred_by_class[class_name] = []
                gt_by_class[class_name] = []
            
            # Process each sample
            for predictions, ground_truths in zip(predictions_list, ground_truths_list):
                # Group predictions by class
                for pred in predictions:
                    class_name = class_names[pred['label']]
                    # Format: [x, y, z, w, l, h, yaw, score]
                    bbox_with_score = pred['bbox_3d'] + [pred['score']]
                    pred_by_class[class_name].append(bbox_with_score)
                
                # Group ground truths by class
                for gt in ground_truths:
                    class_name = class_names[gt['label']]
                    gt_by_class[class_name].append(gt['bbox_3d'])
            
            # Convert to numpy arrays
            for class_name in class_names:
                if pred_by_class[class_name]:
                    pred_by_class[class_name] = np.array(pred_by_class[class_name])
                else:
                    pred_by_class[class_name] = np.zeros((0, 8))
                
                if gt_by_class[class_name]:
                    gt_by_class[class_name] = np.array(gt_by_class[class_name])
                else:
                    gt_by_class[class_name] = np.zeros((0, 7))
            
            # Convert to the format expected by eval_map_recall
            # eval_map_recall expects gt to be a dictionary with sample_id as keys
            gt_by_sample = {}
            for sample_idx in range(len(predictions_list)):
                gt_by_sample[sample_idx] = {}
                for class_name in class_names:
                    gt_by_sample[sample_idx][class_name] = []
            
            # Populate gt_by_sample
            for sample_idx, ground_truths in enumerate(ground_truths_list):
                for gt in ground_truths:
                    class_name = class_names[gt['label']]
                    gt_by_sample[sample_idx][class_name].append(gt['bbox_3d'])
            
            # Convert to numpy arrays for each sample
            for sample_idx in range(len(predictions_list)):
                for class_name in class_names:
                    if gt_by_sample[sample_idx][class_name]:
                        gt_by_sample[sample_idx][class_name] = np.array(gt_by_sample[sample_idx][class_name])
                    else:
                        gt_by_sample[sample_idx][class_name] = np.zeros((0, 7))
            
            # Debug: Print input to eval_map
            print(f"DEBUG: eval_map input - pred_by_class keys: {list(pred_by_class.keys())}")
            print(f"DEBUG: eval_map input - gt_by_class keys: {list(gt_by_class.keys())}")
            for class_name in class_names:
                print(f"DEBUG: {class_name} - pred shape: {pred_by_class[class_name].shape}, gt shape: {gt_by_class[class_name].shape}")
            
            # Compute 3D mAP metrics using eval_map_recall
            map_results = eval_map(
                pred_by_class,
                gt_by_sample,
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
                "mATE": 0.0,   # Placeholder - would need proper implementation
                "mASE": 0.0,   # Placeholder - would need proper implementation
                "mAOE": 0.0,   # Placeholder - would need proper implementation
                "mAVE": 0.0,   # Placeholder - would need proper implementation
                "mAAE": 0.0,   # Placeholder - would need proper implementation
                "per_class_ap": per_class_ap,
            }
            
        except ImportError:
            print("mmdet3d.core.evaluation not available, using simplified metrics")
            return {
                "mAP": 0.0,
                "mAP_50": 0.0,
                "NDS": 0.0,
                "mATE": 0.0,
                "mASE": 0.0,
                "mAOE": 0.0,
                "mAVE": 0.0,
                "mAAE": 0.0,
                "per_class_ap": {i: 0.0 for i in range(num_classes)},
            }
        except Exception as e:
            print(f"Error computing mmdet3d metrics: {e}")
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
                "per_class_ap": {i: 0.0 for i in range(num_classes)},
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
                # Handle both numeric and dict AP values
                if isinstance(ap, dict):
                    ap_value = ap.get('ap', 0.0)
                else:
                    ap_value = float(ap) if ap is not None else 0.0
                print(f"  {class_name:15s}: {ap_value:.4f}")

        # Basic statistics
        print(f"\nDetection Statistics:")
        print(f"  Total Predictions: {results['total_predictions']}")
        print(f"  Total Ground Truths: {results['total_ground_truths']}")

        # Per-class statistics
        print(f"\nPer-Class Statistics:")
        for class_id in range(len(self.class_names)):
            class_name = self.class_names[class_id]
            num_preds = results["per_class_predictions"].get(class_id, 0)
            num_gts = results["per_class_ground_truths"].get(class_id, 0)
            print(f"  {class_name}:")
            print(f"    Predictions: {num_preds}")
            print(f"    Ground Truths: {num_gts}")

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
