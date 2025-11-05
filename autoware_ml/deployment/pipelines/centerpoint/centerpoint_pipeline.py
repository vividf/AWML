"""
CenterPoint Deployment Pipeline Base Class.

This module provides the abstract base class for CenterPoint deployment,
defining the unified pipeline that shares PyTorch processing while allowing
backend-specific optimizations for voxel encoder and backbone/head.
"""

from abc import abstractmethod
from typing import Dict, List, Tuple, Any
import logging

import torch
import numpy as np

from autoware_ml.deployment.core.detection_3d_pipeline import Detection3DPipeline


logger = logging.getLogger(__name__)


class CenterPointDeploymentPipeline(Detection3DPipeline):
    """
    Abstract base class for CenterPoint deployment pipeline.
    
    This class defines the complete inference flow for CenterPoint, with:
    - Shared preprocessing (voxelization + input features)
    - Shared middle encoder processing
    - Shared postprocessing (predict_by_feat)
    - Abstract methods for backend-specific voxel encoder and backbone/head
    
    The design eliminates code duplication by centralizing PyTorch processing
    while allowing ONNX/TensorRT backends to optimize the convertible parts.
    """
    
    def __init__(
        self, 
        pytorch_model, 
        device: str = "cuda",
        backend_type: str = "unknown"
    ):
        """
        Initialize CenterPoint pipeline.
        
        Args:
            pytorch_model: PyTorch model (used for preprocessing, middle encoder, postprocessing)
            device: Device for inference ('cuda' or 'cpu')
            backend_type: Backend type ('pytorch', 'onnx', 'tensorrt')
        """
        # Get class names from model config if available
        class_names = ["VEHICLE", "PEDESTRIAN", "CYCLIST"]  # Default T4Dataset classes
        if hasattr(pytorch_model, 'CLASSES'):
            class_names = pytorch_model.CLASSES
        elif hasattr(pytorch_model, 'cfg') and hasattr(pytorch_model.cfg, 'class_names'):
            class_names = pytorch_model.cfg.class_names
        
        # Get point cloud range and voxel size from model config
        point_cloud_range = None
        voxel_size = None
        if hasattr(pytorch_model, 'cfg'):
            if hasattr(pytorch_model.cfg, 'point_cloud_range'):
                point_cloud_range = pytorch_model.cfg.point_cloud_range
            if hasattr(pytorch_model.cfg, 'voxel_size'):
                voxel_size = pytorch_model.cfg.voxel_size
        
        # Initialize parent class
        super().__init__(
            model=pytorch_model,
            device=device,
            num_classes=len(class_names),
            class_names=class_names,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            backend_type=backend_type
        )
        
        self.pytorch_model = pytorch_model
    
    # ========== Shared Methods (All backends use same logic) ==========
    
    def preprocess(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Preprocess: voxelization + input features preparation.
        
        ONNX/TensorRT backends use this for voxelization and input feature preparation.
        PyTorch backend may override this method for end-to-end inference.
        
        Args:
            points: Input point cloud [N, point_features]
            
        Returns:
            Dictionary containing:
            - input_features: 11-dim features for voxel encoder [N_voxels, max_points, 11]
            - voxels: Raw voxel data
            - num_points: Number of points per voxel
            - coors: Voxel coordinates [N_voxels, 4] (batch_idx, z, y, x)
        """
        from mmdet3d.structures import Det3DDataSample
        
        # Ensure points are on correct device
        points_tensor = points.to(self.device)
        
        # Step 1: Voxelization using PyTorch data_preprocessor
        data_samples = [Det3DDataSample()]
        
        with torch.no_grad():
            batch_inputs = self.pytorch_model.data_preprocessor(
                {'inputs': {'points': [points_tensor]}, 'data_samples': data_samples}
            )
        
        voxel_dict = batch_inputs['inputs']['voxels']
        voxels = voxel_dict['voxels']
        num_points = voxel_dict['num_points']
        coors = voxel_dict['coors']
        
        # Step 2: Get input features (only for ONNX/TensorRT models)
        input_features = None
        with torch.no_grad():
            if hasattr(self.pytorch_model.pts_voxel_encoder, 'get_input_features'):
                input_features = self.pytorch_model.pts_voxel_encoder.get_input_features(
                    voxels, num_points, coors
                )
                logger.debug(f"Preprocessed with input_features: shape={input_features.shape}")
            else:
                logger.debug(f"Preprocessed without input_features (standard model)")
        
        return {
            'input_features': input_features,
            'voxels': voxels,
            'num_points': num_points,
            'coors': coors
        }
    
    def process_middle_encoder(
        self, 
        voxel_features: torch.Tensor, 
        coors: torch.Tensor
    ) -> torch.Tensor:
        """
        Process through middle encoder using PyTorch.
        
        All backends use the same middle encoder processing because sparse convolution
        cannot be converted to ONNX/TensorRT efficiently.
        
        Args:
            voxel_features: Features from voxel encoder [N_voxels, feature_dim]
            coors: Voxel coordinates [N_voxels, 4]
            
        Returns:
            spatial_features: Spatial features [B, C, H, W]
        """
        # Ensure tensors are on correct device
        voxel_features = voxel_features.to(self.device)
        coors = coors.to(self.device)
        
        # Calculate batch size
        batch_size = int(coors[-1, 0].item()) + 1 if len(coors) > 0 else 1
        
        # Process through PyTorch middle encoder
        with torch.no_grad():
            spatial_features = self.pytorch_model.pts_middle_encoder(
                voxel_features, coors, batch_size
            )
        
        logger.debug(f"Middle encoder output: spatial_features shape={spatial_features.shape}")
        
        return spatial_features
    
    def postprocess(
        self, 
        head_outputs: List[torch.Tensor],
        sample_meta: Dict
    ) -> List[Dict]:
        """
        Postprocess: decode head outputs using PyTorch's predict_by_feat.
        
        All backends use the same postprocessing to ensure consistent results.
        This includes NMS, coordinate transformation, and score filtering.
        
        Args:
            head_outputs: List of [heatmap, reg, height, dim, rot, vel]
            sample_meta: Sample metadata (point_cloud_range, voxel_size, etc.)
            
        Returns:
            List of predictions with bbox_3d, score, and label
        """
        # Ensure all outputs are on correct device
        head_outputs = [out.to(self.device) for out in head_outputs]
        
        # Organize head outputs: [heatmap, reg, height, dim, rot, vel]
        if len(head_outputs) != 6:
            raise ValueError(f"Expected 6 head outputs, got {len(head_outputs)}")
        
        heatmap, reg, height, dim, rot, vel = head_outputs
        
        # Check if rot_y_axis_reference conversion is needed
        # When ONNX/TensorRT outputs use rot_y_axis_reference format, we need to convert back
        # to standard format before passing to PyTorch's predict_by_feat
        if hasattr(self.pytorch_model, 'pts_bbox_head'):
            rot_y_axis_reference = getattr(
                self.pytorch_model.pts_bbox_head, 
                '_rot_y_axis_reference', 
                False
            )
            
            if rot_y_axis_reference:
                # Convert dim from [w, l, h] back to [l, w, h]
                dim = dim[:, [1, 0, 2], :, :]
                
                # Convert rot from [-cos(x), -sin(y)] back to [sin(y), cos(x)]
                rot = rot * (-1.0)
                rot = rot[:, [1, 0], :, :]
                
                logger.debug("Converted outputs from rot_y_axis_reference format to standard format")
        
        # Convert to mmdet3d format
        preds_dict = {
            'heatmap': heatmap,
            'reg': reg,
            'height': height,
            'dim': dim,
            'rot': rot,
            'vel': vel
        }
        preds_dicts = ([preds_dict],)  # Tuple[List[dict]] format
        
        # Prepare metadata
        from mmdet3d.structures import LiDARInstance3DBoxes
        if 'box_type_3d' not in sample_meta:
            sample_meta['box_type_3d'] = LiDARInstance3DBoxes
        batch_input_metas = [sample_meta]
        
        # Use PyTorch's predict_by_feat for consistent decoding
        with torch.no_grad():
            predictions_list = self.pytorch_model.pts_bbox_head.predict_by_feat(
                preds_dicts=preds_dicts,
                batch_input_metas=batch_input_metas
            )
        
        # Parse predictions
        results = []
        for pred_instances in predictions_list:
            bboxes_3d = pred_instances.bboxes_3d.tensor.cpu().numpy()
            scores_3d = pred_instances.scores_3d.cpu().numpy()
            labels_3d = pred_instances.labels_3d.cpu().numpy()
            
            for i in range(len(bboxes_3d)):
                results.append({
                    'bbox_3d': bboxes_3d[i][:7].tolist(),  # [x, y, z, w, l, h, yaw]
                    'score': float(scores_3d[i]),
                    'label': int(labels_3d[i])
                })
        
        logger.debug(f"Postprocessing: {len(results)} detections")
        
        return results
    
    # ========== Abstract Methods (Backend-specific implementations) ==========
    
    @abstractmethod
    def run_voxel_encoder(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Run voxel encoder inference.
        
        This method must be implemented by each backend (PyTorch/ONNX/TensorRT)
        to provide optimized voxel encoder inference.
        
        Args:
            input_features: Input features [N_voxels, max_points, feature_dim]
            
        Returns:
            voxel_features: Voxel features [N_voxels, feature_dim]
        """
        pass
    
    @abstractmethod
    def run_backbone_head(self, spatial_features: torch.Tensor) -> List[torch.Tensor]:
        """
        Run backbone + neck + head inference.
        
        This method must be implemented by each backend (PyTorch/ONNX/TensorRT)
        to provide optimized backbone/neck/head inference.
        
        Args:
            spatial_features: Spatial features [B, C, H, W]
            
        Returns:
            List of head outputs: [heatmap, reg, height, dim, rot, vel]
        """
        pass
    
    # ========== Main Inference Pipeline ==========
    
    def infer(
        self, 
        points: torch.Tensor,
        sample_meta: Dict = None,
        return_raw_outputs: bool = False
    ) -> Tuple[Any, float, Dict[str, float]]:
        """
        Complete inference pipeline.
        
        This method orchestrates the entire inference flow:
        1. Preprocessing (PyTorch)
        2. Voxel Encoder (backend-specific)
        3. Middle Encoder (PyTorch)
        4. Backbone + Head (backend-specific)
        5. Postprocessing (PyTorch) - optional
        
        Args:
            points: Input point cloud [N, point_features]
            sample_meta: Sample metadata (optional)
            return_raw_outputs: If True, return raw head outputs instead of postprocessed predictions
            
        Returns:
            If return_raw_outputs=False:
                predictions: List of detection results
                total_latency_ms: Total inference latency in milliseconds
                latency_breakdown: Dict with latencies for each stage
            If return_raw_outputs=True:
                head_outputs: List of raw head outputs [heatmap, reg, height, dim, rot, vel]
                total_latency_ms: Total inference latency in milliseconds
                latency_breakdown: Dict with latencies for each stage
        """
        import time
        start_time = time.time()
        t_prev = start_time
        
        if sample_meta is None:
            sample_meta = {}
        
        try:
            # 1. Preprocess (PyTorch)
            preprocessed = self.preprocess(points)
            t_after_pre = time.time()
            pre_ms = (t_after_pre - t_prev) * 1000
            t_prev = t_after_pre
            logger.info(f"1. Preprocessing:       {pre_ms:8.2f} ms")
            
            # 2. Voxel Encoder (backend-specific)
            voxel_features = self.run_voxel_encoder(preprocessed['input_features'])
            t_after_voxel = time.time()
            voxel_ms = (t_after_voxel - t_prev) * 1000
            t_prev = t_after_voxel
            logger.info(f"2. Voxel Encoder:       {voxel_ms:8.2f} ms")
            
            # 3. Middle Encoder (PyTorch)
            spatial_features = self.process_middle_encoder(
                voxel_features, 
                preprocessed['coors']
            )
            t_after_middle = time.time()
            middle_ms = (t_after_middle - t_prev) * 1000
            t_prev = t_after_middle
            logger.info(f"3. Middle Encoder:       {middle_ms:8.2f} ms")
            
            # 4. Backbone + Head (backend-specific)
            head_outputs = self.run_backbone_head(spatial_features)
            t_after_backbone = time.time()
            backbone_ms = (t_after_backbone - t_prev) * 1000
            t_prev = t_after_backbone
            logger.info(f"4. Backbone + Head:    {backbone_ms:8.2f} ms")
            
            # 5. Postprocess (PyTorch) - optional
            latency_breakdown = {
                'preprocessing_ms': pre_ms,
                'voxel_encoder_ms': voxel_ms,
                'middle_encoder_ms': middle_ms,
                'backbone_head_ms': backbone_ms,
            }
            
            if return_raw_outputs:
                total_ms = (time.time() - start_time) * 1000
                logger.info(f"Total (no post):       {total_ms:8.2f} ms")
                logger.debug(f"Inference completed in {total_ms:.2f}ms (returning raw outputs)")
                return head_outputs, total_ms, latency_breakdown
            else:
                predictions = self.postprocess(head_outputs, sample_meta)
                t_after_post = time.time()
                post_ms = (t_after_post - t_prev) * 1000
                total_ms = (t_after_post - start_time) * 1000
                logger.info(f"5. Postprocessing:      {post_ms:8.2f} ms")
                logger.info(f"Total:                 {total_ms:8.2f} ms")
                latency_breakdown['postprocessing_ms'] = post_ms
                logger.debug(f"Inference completed in {total_ms:.2f}ms with {len(predictions)} detections")
                return predictions, total_ms, latency_breakdown
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def __repr__(self):
        return f"{self.__class__.__name__}(device={self.device})"

