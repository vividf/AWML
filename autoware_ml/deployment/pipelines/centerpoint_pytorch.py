"""
CenterPoint PyTorch Pipeline Implementation.

This module implements the CenterPoint pipeline using pure PyTorch,
providing a baseline for comparison with optimized backends.
"""

import logging
from typing import List, Dict, Tuple

import torch

from .centerpoint_pipeline import CenterPointDeploymentPipeline


logger = logging.getLogger(__name__)


class CenterPointPyTorchPipeline(CenterPointDeploymentPipeline):
    """
    PyTorch implementation of CenterPoint pipeline.
    
    Uses pure PyTorch for all components, providing maximum flexibility
    and ease of debugging at the cost of inference speed.
    
    For standard CenterPoint models (non-ONNX), uses end-to-end inference.
    """
    
    def __init__(self, pytorch_model, device: str = "cuda"):
        """
        Initialize PyTorch pipeline.
        
        Args:
            pytorch_model: PyTorch CenterPoint model
            device: Device for inference
        """
        super().__init__(pytorch_model, device)
        
        # Check if this is an ONNX-compatible model
        self.is_onnx_model = hasattr(pytorch_model.pts_voxel_encoder, 'get_input_features')
        
        if self.is_onnx_model:
            logger.info("PyTorch pipeline initialized (ONNX-compatible model)")
        else:
            logger.info("PyTorch pipeline initialized (standard model, using end-to-end inference)")
    
    def infer(self, points: torch.Tensor, sample_meta: Dict = None) -> Tuple[List[Dict], float]:
        """
        Complete inference pipeline.
        
        For standard models, uses mmdet3d's inference_detector for end-to-end inference.
        For ONNX-compatible models, uses the staged pipeline.
        """
        import time
        
        if sample_meta is None:
            sample_meta = {}
        
        # For standard models, use end-to-end inference
        if not self.is_onnx_model:
            return self._infer_end_to_end(points, sample_meta)
        
        # For ONNX models, use staged pipeline
        return super().infer(points, sample_meta)
    
    def _infer_end_to_end(self, points: torch.Tensor, sample_meta: Dict) -> Tuple[List[Dict], float]:
        """End-to-end inference for standard PyTorch models."""
        import time
        from mmdet3d.apis import inference_detector
        
        start_time = time.time()
        
        try:
            # Convert points to numpy for inference_detector
            if isinstance(points, torch.Tensor):
                points_np = points.cpu().numpy()
            else:
                points_np = points
            
            # Use mmdet3d's inference API
            with torch.no_grad():
                results = inference_detector(self.pytorch_model, points_np)
            
            # Parse results
            predictions = []
            if len(results) > 0 and hasattr(results[0], 'pred_instances_3d'):
                pred_instances = results[0].pred_instances_3d
                
                if hasattr(pred_instances, 'bboxes_3d'):
                    bboxes_3d = pred_instances.bboxes_3d.tensor.cpu().numpy()
                    scores_3d = pred_instances.scores_3d.cpu().numpy()
                    labels_3d = pred_instances.labels_3d.cpu().numpy()
                    
                    for i in range(len(bboxes_3d)):
                        predictions.append({
                            'bbox_3d': bboxes_3d[i][:7].tolist(),  # [x, y, z, w, l, h, yaw]
                            'score': float(scores_3d[i]),
                            'label': int(labels_3d[i])
                        })
            
            latency_ms = (time.time() - start_time) * 1000
            
            logger.debug(f"End-to-end inference completed in {latency_ms:.2f}ms with {len(predictions)} detections")
            
            return predictions, latency_ms
            
        except Exception as e:
            logger.error(f"End-to-end inference failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def run_voxel_encoder(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Run voxel encoder using PyTorch.
        
        Note: Only used for ONNX-compatible models.
        
        Args:
            input_features: Input features [N_voxels, max_points, feature_dim]
            
        Returns:
            voxel_features: Voxel features [N_voxels, feature_dim]
        """
        if input_features is None:
            raise ValueError("input_features is None. This should not happen for ONNX models.")
        
        input_features = input_features.to(self.device)
        
        with torch.no_grad():
            voxel_features = self.pytorch_model.pts_voxel_encoder(input_features)
        
        logger.debug(f"PyTorch voxel encoder output shape: {voxel_features.shape}")
        
        return voxel_features
    
    def run_backbone_head(self, spatial_features: torch.Tensor) -> List[torch.Tensor]:
        """
        Run backbone + neck + head using PyTorch.
        
        Note: Only used for ONNX-compatible models.
        
        Args:
            spatial_features: Spatial features [B, C, H, W]
            
        Returns:
            List of head outputs: [heatmap, reg, height, dim, rot, vel]
        """
        spatial_features = spatial_features.to(self.device)
        
        with torch.no_grad():
            # Backbone
            x = self.pytorch_model.pts_backbone(spatial_features)
            
            # Neck
            if hasattr(self.pytorch_model, 'pts_neck') and self.pytorch_model.pts_neck is not None:
                x = self.pytorch_model.pts_neck(x)
            
            # Head - returns tuple of task head outputs
            head_outputs_tuple = self.pytorch_model.pts_bbox_head(x)
            
            # Extract outputs for single-task head
            # head_outputs_tuple format: (List[Dict],)
            if isinstance(head_outputs_tuple, tuple) and len(head_outputs_tuple) > 0:
                task_outputs = head_outputs_tuple[0]  # Get first task
                if isinstance(task_outputs, list) and len(task_outputs) > 0:
                    preds_dict = task_outputs[0]  # Get first (and only) dict
                    
                    # Extract individual outputs
                    head_outputs = [
                        preds_dict['heatmap'],
                        preds_dict['reg'],
                        preds_dict['height'],
                        preds_dict['dim'],
                        preds_dict['rot'],
                        preds_dict['vel']
                    ]
                else:
                    raise ValueError(f"Unexpected task_outputs format: {type(task_outputs)}")
            else:
                raise ValueError(f"Unexpected head_outputs format: {type(head_outputs_tuple)}")
        
        logger.debug(f"PyTorch backbone+head output: {[out.shape for out in head_outputs]}")
        
        return head_outputs

