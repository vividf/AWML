"""
CenterPoint ONNX 輔助模組

為 CenterPoint 多文件 ONNX 導出提供輔助功能，
與現有的 ONNXBackend 配合使用。
"""

import logging
import os
import time
from typing import Dict, Any, Optional, Tuple

import numpy as np
import onnxruntime as ort
import torch

logger = logging.getLogger(__name__)


class CenterPointONNXHelper:
    """CenterPoint ONNX 輔助類，處理多文件 ONNX 的特殊需求。"""

    def __init__(self, onnx_dir: str, device: str = "cpu", pytorch_model=None):
        """
        Initialize CenterPoint ONNX helper.

        Args:
            onnx_dir: Directory containing CenterPoint ONNX files
            device: Device to run inference on ('cpu' or 'cuda')
            pytorch_model: PyTorch model for voxelization (needed for proper preprocessing)
        """
        self.onnx_dir = onnx_dir
        self.device = device
        self.pytorch_model = pytorch_model
        
        # Paths to ONNX files
        self.voxel_encoder_path = os.path.join(onnx_dir, "pts_voxel_encoder.onnx")
        self.backbone_head_path = os.path.join(onnx_dir, "pts_backbone_neck_head.onnx")
        
        # Verify files exist
        if not os.path.exists(self.voxel_encoder_path):
            raise FileNotFoundError(f"Voxel encoder ONNX not found: {self.voxel_encoder_path}")
        if not os.path.exists(self.backbone_head_path):
            raise FileNotFoundError(f"Backbone head ONNX not found: {self.backbone_head_path}")
        
        # Initialize ONNX Runtime sessions
        self._init_sessions()
        
    def _init_sessions(self):
        """Initialize ONNX Runtime sessions."""
        try:
            # Set execution providers
            if self.device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
            
            # Initialize voxel encoder session
            self.voxel_encoder_session = ort.InferenceSession(
                self.voxel_encoder_path, 
                providers=providers
            )
            
            # Initialize backbone head session
            self.backbone_head_session = ort.InferenceSession(
                self.backbone_head_path, 
                providers=providers
            )
            
            logger.info(f"CenterPoint ONNX sessions initialized on {self.device}")
            logger.info(f"Voxel encoder: {self.voxel_encoder_path}")
            logger.info(f"Backbone head: {self.backbone_head_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ONNX sessions: {e}")
            raise
    
    def _voxelize_points(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Voxelize points using PyTorch model's data_preprocessor."""
        if self.pytorch_model is None:
            raise ValueError("PyTorch model is required for voxelization")
        
        # Convert to torch tensor
        points_tensor = torch.from_numpy(points).float()
        
        # Use PyTorch model's data_preprocessor for voxelization
        from mmdet3d.structures import Det3DDataSample
        data_samples = [Det3DDataSample()]
        
        batch_inputs = self.pytorch_model.data_preprocessor(
            {'inputs': {'points': [points_tensor]}, 'data_samples': data_samples}
        )
        
        voxel_dict = batch_inputs['inputs']['voxels']
        
        # Convert to numpy
        voxels = voxel_dict['voxels'].cpu().numpy()
        num_points = voxel_dict['num_points'].cpu().numpy()
        coors = voxel_dict['coors'].cpu().numpy()
        
        return voxels, num_points, coors
    
    def _get_input_features(self, voxels: np.ndarray, num_points: np.ndarray, coors: np.ndarray) -> np.ndarray:
        """Get input features for voxel encoder using PyTorch model."""
        if self.pytorch_model is None:
            raise ValueError("PyTorch model is required for input feature generation")
        
        # Convert to torch tensors
        voxels_tensor = torch.from_numpy(voxels).float()
        num_points_tensor = torch.from_numpy(num_points).long()
        coors_tensor = torch.from_numpy(coors).long()
        
        # Use PyTorch model's voxel encoder to get input features
        # Check if the model has get_input_features method (ONNX version)
        if hasattr(self.pytorch_model.pts_voxel_encoder, 'get_input_features'):
            input_features = self.pytorch_model.pts_voxel_encoder.get_input_features(
                voxels_tensor, num_points_tensor, coors_tensor
            )
        else:
            # For standard PillarFeatureNet, we need to manually create input features
            # This implements the same logic as PillarFeatureNetONNX.get_input_features
            features_ls = [voxels_tensor]
            
            # Get voxel encoder configuration
            voxel_encoder = self.pytorch_model.pts_voxel_encoder
            
            # Find distance of x, y, and z from cluster center
            if hasattr(voxel_encoder, '_with_cluster_center') and voxel_encoder._with_cluster_center:
                points_mean = voxels_tensor[:, :, :3].sum(dim=1, keepdim=True) / num_points_tensor.type_as(voxels_tensor).view(-1, 1, 1)
                f_cluster = voxels_tensor[:, :, :3] - points_mean
                features_ls.append(f_cluster)
            
            # Find distance of x, y, and z from pillar center
            if hasattr(voxel_encoder, '_with_voxel_center') and voxel_encoder._with_voxel_center:
                dtype = voxels_tensor.dtype
                if hasattr(voxel_encoder, 'legacy') and not voxel_encoder.legacy:
                    f_center = torch.zeros_like(voxels_tensor[:, :, :3])
                    f_center[:, :, 0] = voxels_tensor[:, :, 0] - (coors_tensor[:, 3].to(dtype).unsqueeze(1) * voxel_encoder.vx + voxel_encoder.x_offset)
                    f_center[:, :, 1] = voxels_tensor[:, :, 1] - (coors_tensor[:, 2].to(dtype).unsqueeze(1) * voxel_encoder.vy + voxel_encoder.y_offset)
                    f_center[:, :, 2] = voxels_tensor[:, :, 2] - (coors_tensor[:, 1].to(dtype).unsqueeze(1) * voxel_encoder.vz + voxel_encoder.z_offset)
                else:
                    f_center = voxels_tensor[:, :, :3]
                    f_center[:, :, 0] = f_center[:, :, 0] - (
                        coors_tensor[:, 3].type_as(voxels_tensor).unsqueeze(1) * voxel_encoder.vx + voxel_encoder.x_offset
                    )
                    f_center[:, :, 1] = f_center[:, :, 1] - (
                        coors_tensor[:, 2].type_as(voxels_tensor).unsqueeze(1) * voxel_encoder.vy + voxel_encoder.y_offset
                    )
                    f_center[:, :, 2] = f_center[:, :, 2] - (
                        coors_tensor[:, 1].type_as(voxels_tensor).unsqueeze(1) * voxel_encoder.vz + voxel_encoder.z_offset
                    )
                features_ls.append(f_center)
            
            if hasattr(voxel_encoder, '_with_distance') and voxel_encoder._with_distance:
                points_dist = torch.norm(voxels_tensor[:, :, :3], 2, 2, keepdim=True)
                features_ls.append(points_dist)
            
            # Combine together feature decorations
            input_features = torch.cat(features_ls, dim=-1)
            
            # Apply mask for empty pillars
            from mmdet3d.models.voxel_encoders.utils import get_paddings_indicator
            voxel_count = input_features.shape[1]
            mask = get_paddings_indicator(num_points_tensor, voxel_count, axis=0)
            mask = torch.unsqueeze(mask, -1).type_as(input_features)
            input_features *= mask
        
        return input_features.cpu().numpy()
    
    def _process_middle_encoder(self, voxel_features: np.ndarray, coors: np.ndarray) -> np.ndarray:
        """Process through middle encoder using PyTorch model."""
        if self.pytorch_model is None:
            raise ValueError("PyTorch model is required for middle encoder processing")
        
        # Convert to torch tensors
        voxel_features_tensor = torch.from_numpy(voxel_features).float()
        coors_tensor = torch.from_numpy(coors).long()
        
        # Process through middle encoder
        batch_size = coors_tensor[-1, 0] + 1
        spatial_features = self.pytorch_model.pts_middle_encoder(
            voxel_features_tensor, coors_tensor, batch_size
        )
        
        return spatial_features.cpu().numpy()
    
    def preprocess_for_onnx(self, input_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Preprocess input data for ONNX inference.
        
        Args:
            input_data: Dictionary containing 'points' tensor
            
        Returns:
            Dictionary with preprocessed data for ONNX inference
        """
        if 'points' not in input_data:
            raise ValueError("Input data must contain 'points' key")
        
        points = input_data['points']
        
        # Convert points to numpy if needed
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        
        # Step 1: Voxelize points using PyTorch model
        voxels, num_points, coors = self._voxelize_points(points)
        
        # Step 2: Get input features for voxel encoder
        input_features = self._get_input_features(voxels, num_points, coors)
        
        # Step 3: Run voxel encoder ONNX
        voxel_encoder_inputs = {"input_features": input_features}
        voxel_features = self.voxel_encoder_session.run(
            ["pillar_features"], 
            voxel_encoder_inputs
        )[0]
        
        # Squeeze middle dimension (ONNX models output 3D, need 2D)
        if voxel_features.ndim == 3:
            voxel_features = voxel_features.squeeze(1)
        
        # Step 4: Process through middle encoder using PyTorch model
        spatial_features = self._process_middle_encoder(voxel_features, coors)
        
        # Return preprocessed data for backbone/neck/head ONNX
        return {"spatial_features": spatial_features}
    
    def infer(self, input_data: Dict[str, Any]) -> Tuple[list, float]:
        """
        Run inference using CenterPoint ONNX models.
        
        Args:
            input_data: Dictionary containing 'points' tensor
            
        Returns:
            Tuple of (outputs, latency)
        """
        start_time = time.time()
        
        # Preprocess input data
        preprocessed_data = self.preprocess_for_onnx(input_data)
        
        # Run backbone/neck/head ONNX inference
        onnx_outputs = self.backbone_head_session.run(
            None,  # Use all outputs
            {"spatial_features": preprocessed_data["spatial_features"]}
        )
        
        # Postprocess outputs
        processed_outputs = self.postprocess_from_onnx(onnx_outputs)
        
        latency = time.time() - start_time
        
        return processed_outputs, latency
    
    def postprocess_from_onnx(self, onnx_outputs: list) -> list:
        """
        Postprocess ONNX outputs to match PyTorch format.
        
        Args:
            onnx_outputs: Raw outputs from backbone/neck/head ONNX
            
        Returns:
            Processed outputs matching PyTorch format
        """
        # ONNX outputs are in order: ['heatmap', 'reg', 'height', 'dim', 'rot', 'vel']
        # PyTorch outputs are in order: ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
        # We need to reorder ONNX outputs to match PyTorch order
        
        if len(onnx_outputs) != 6:
            logger.warning(f"Expected 6 ONNX outputs, got {len(onnx_outputs)}")
            return onnx_outputs
        
        # Reorder ONNX outputs to match PyTorch order
        # ONNX: [heatmap, reg, height, dim, rot, vel]
        # PyTorch: [reg, height, dim, rot, vel, heatmap]
        reordered_outputs = [
            onnx_outputs[1],  # reg
            onnx_outputs[2],  # height  
            onnx_outputs[3],  # dim
            onnx_outputs[4],  # rot
            onnx_outputs[5],  # vel
            onnx_outputs[0],  # heatmap
        ]
        
        return reordered_outputs


def is_centerpoint_onnx(onnx_path: str) -> bool:
    """
    Check if the given path is a CenterPoint multi-file ONNX export.
    
    Args:
        onnx_path: Path to ONNX model or directory
        
    Returns:
        True if it's a CenterPoint multi-file ONNX export
    """
    if not os.path.isdir(onnx_path):
        return False
    
    voxel_encoder_path = os.path.join(onnx_path, "pts_voxel_encoder.onnx")
    backbone_head_path = os.path.join(onnx_path, "pts_backbone_neck_head.onnx")
    
    return os.path.exists(voxel_encoder_path) and os.path.exists(backbone_head_path)
