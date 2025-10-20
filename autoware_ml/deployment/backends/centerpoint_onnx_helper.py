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
        input_features = self.pytorch_model.pts_voxel_encoder.get_input_features(
            voxels_tensor, num_points_tensor, coors_tensor
        )
        
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
    
    def postprocess_from_onnx(self, onnx_outputs: list) -> list:
        """
        Postprocess ONNX outputs to match PyTorch format.
        
        Args:
            onnx_outputs: Raw outputs from backbone/neck/head ONNX
            
        Returns:
            Processed outputs matching PyTorch format
        """
        # For CenterPoint, ONNX outputs should already match PyTorch format
        # This method can be extended for additional postprocessing if needed
        return onnx_outputs


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
