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
            # Create session options with disabled graph optimization for numerical consistency
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            so.log_severity_level = 1  # Enable detailed logging to debug memcpy issues
            
            # Set execution providers
            if self.device.startswith("cuda"):
                # # CUDA provider settings for numerical consistency and performance
                # cuda_provider_options = {
                #     "device_id": 0,
                #     "arena_extend_strategy": "kNextPowerOfTwo",
                #     "cudnn_conv_algo_search": "HEURISTIC",  # Fixed algorithm search
                #     "do_copy_in_default_stream": True,
                #     "enable_cuda_graph": False,  # Disable CUDA Graph due to partitioning issues
                #     "cudnn_conv1d_pad_to_nc1d": True,  # Optimize conv1d padding
                #     "cudnn_conv_use_max_workspace": True,  # Use max workspace for better performance
                #     # Note: enable_cuda_graph_capture is not supported in this ONNX Runtime version
                #     # Note: enable_tf32 is not supported in this ONNX Runtime version
                #     # Note: user_compute_stream is not supported in this ONNX Runtime version
                # }
                # providers = [
                #     ("CUDAExecutionProvider", cuda_provider_options),
                #     "CPUExecutionProvider"
                # ]
                # logger.info("Attempting to use CUDA acceleration (will fallback to CPU if needed)...")
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
                logger.info("Using CPU for ONNX inference")
            
            # Initialize voxel encoder session
            self.voxel_encoder_session = ort.InferenceSession(
                self.voxel_encoder_path, 
                sess_options=so,
                providers=providers
            )
            
            # Initialize backbone head session
            self.backbone_head_session = ort.InferenceSession(
                self.backbone_head_path, 
                sess_options=so,
                providers=providers
            )
            
            # Check which providers are actually being used
            voxel_providers = self.voxel_encoder_session.get_providers()
            backbone_providers = self.backbone_head_session.get_providers()
            
            # logger.info(f"CenterPoint ONNX sessions initialized")
            # logger.info(f"Voxel encoder providers: {voxel_providers}")
            # logger.info(f"Backbone head providers: {backbone_providers}")
            logger.info(f"Voxel encoder: {self.voxel_encoder_path}")
            logger.info(f"Backbone head: {self.backbone_head_path}")
            
            # Update device based on actual providers used
            if "CUDAExecutionProvider" in voxel_providers and "CUDAExecutionProvider" in backbone_providers:
                self.actual_device = "cuda"
                logger.info("Successfully using CUDA execution providers")
            else:
                self.actual_device = "cpu"
                logger.warning("Fell back to CPU execution providers")
            
        except Exception as e:
            logger.error(f"Failed to initialize ONNX sessions: {e}")
            raise
    
    def _voxelize_points(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Voxelize points using PyTorch model's data_preprocessor."""
        if self.pytorch_model is None:
            raise ValueError("PyTorch model is required for voxelization")
        
        # Convert to torch tensor and move to the same device as the model
        points_tensor = torch.from_numpy(points).float()
        if hasattr(self.pytorch_model, '_torch_device'):
            points_tensor = points_tensor.to(self.pytorch_model._torch_device)
        
        # Use PyTorch model's data_preprocessor for voxelization
        from mmdet3d.structures import Det3DDataSample
        data_samples = [Det3DDataSample()]
        
        batch_inputs = self.pytorch_model.data_preprocessor(
            {'inputs': {'points': [points_tensor]}, 'data_samples': data_samples}
        )
        
        voxel_dict = batch_inputs['inputs']['voxels']
        
        # Convert to numpy - keep on same device as PyTorch model for consistency
        voxels = voxel_dict['voxels'].cpu().numpy()
        num_points = voxel_dict['num_points'].cpu().numpy()
        coors = voxel_dict['coors'].cpu().numpy()
        
        return voxels, num_points, coors
    
    def _get_input_features(self, voxels: np.ndarray, num_points: np.ndarray, coors: np.ndarray) -> np.ndarray:
        """Get input features for voxel encoder using PyTorch model."""
        if self.pytorch_model is None:
            raise ValueError("PyTorch model is required for input feature generation")
        
        # Convert to torch tensors and move to the same device as the model
        device = next(self.pytorch_model.parameters()).device
        voxels_tensor = torch.from_numpy(voxels).float().to(device)
        num_points_tensor = torch.from_numpy(num_points).long().to(device)
        coors_tensor = torch.from_numpy(coors).long().to(device)
        
        # Use PyTorch model's voxel encoder to get input features
        # Check if it's ONNX version (only takes features) or standard version
        if hasattr(self.pytorch_model.pts_voxel_encoder, 'get_input_features'):
            # Use get_input_features for ONNX models
            input_features = self.pytorch_model.pts_voxel_encoder.get_input_features(
                voxels_tensor, 
                num_points_tensor, 
                coors_tensor
            )
            # Then call forward with just the features
            input_features = self.pytorch_model.pts_voxel_encoder(input_features)
        else:
            # Standard model
            input_features = self.pytorch_model.pts_voxel_encoder(
                voxels_tensor, num_points_tensor, coors_tensor
            )
        
        # Debug: Check output shape
        print(f"DEBUG: PyTorch voxel encoder output shape: {input_features.shape}")
        
        # The issue is that PyTorch outputs (20752, 1, 32) but ONNX expects (20752, 32, 11)
        # We need to understand what these dimensions represent:
        # - PyTorch: (num_voxels, 1, 32) where 32 is the feature dimension
        # - ONNX: (num_voxels, 32, 11) where 32 is max_points and 11 is feature_channels
        
        # The correct approach is to NOT reshape, but use the raw features from get_input_features
        # Let's check what get_input_features actually returns
        if hasattr(self.pytorch_model.pts_voxel_encoder, 'get_input_features'):
            # Get the raw input features (before PFN processing)
            raw_features = self.pytorch_model.pts_voxel_encoder.get_input_features(
                voxels_tensor, 
                num_points_tensor, 
                coors_tensor
            )
            print(f"DEBUG: Raw input features shape: {raw_features.shape}")
            
            # Use raw features directly - this should be (num_voxels, 32, 11)
            input_features = raw_features
        else:
            # Fallback: try to reshape the processed features
            if len(input_features.shape) == 3 and input_features.shape[1] == 1:
                # (num_voxels, 1, 32) -> we need to expand this to (num_voxels, 32, 11)
                # This is wrong - we should not be doing this reshaping
                print(f"WARNING: Attempting incorrect reshaping from {input_features.shape}")
                # Don't reshape - this causes numerical differences
                raise ValueError(f"Cannot reshape {input_features.shape} to ONNX format - use get_input_features instead")
        
        print(f"DEBUG: Final input features shape for ONNX: {input_features.shape}")
        
        return input_features.detach().cpu().numpy()
    
    def _process_middle_encoder(self, voxel_features: np.ndarray, coors: np.ndarray) -> np.ndarray:
        """Process through middle encoder using PyTorch model."""
        if self.pytorch_model is None:
            raise ValueError("PyTorch model is required for middle encoder processing")
        
        # Convert to torch tensors and move to the same device as the model
        device = next(self.pytorch_model.parameters()).device
        voxel_features_tensor = torch.from_numpy(voxel_features).float().to(device)
        coors_tensor = torch.from_numpy(coors).long().to(device)
        
        # Process through middle encoder
        batch_size = coors_tensor[-1, 0] + 1
        spatial_features = self.pytorch_model.pts_middle_encoder(
            voxel_features_tensor, coors_tensor, batch_size
        )
        
        return spatial_features.detach().cpu().numpy()
    
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
