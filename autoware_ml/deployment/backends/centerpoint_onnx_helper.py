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
        """
        Get input features for ONNX voxel encoder using PyTorch model.
        
        This method prepares the raw input features (NOT processed features) that the
        ONNX voxel encoder expects. The ONNX voxel encoder will then process these features.
        
        Args:
            voxels: Voxel data (num_voxels, max_points, point_features)
            num_points: Number of points per voxel (num_voxels,)
            coors: Voxel coordinates (num_voxels, 4)
            
        Returns:
            Raw input features for ONNX voxel encoder (num_voxels, max_points, feature_channels)
        """
        if self.pytorch_model is None:
            raise ValueError("PyTorch model is required for input feature generation")
        
        # Convert to torch tensors and move to the same device as the model
        device = next(self.pytorch_model.parameters()).device
        voxels_tensor = torch.from_numpy(voxels).float().to(device)
        num_points_tensor = torch.from_numpy(num_points).long().to(device)
        coors_tensor = torch.from_numpy(coors).long().to(device)
        
        # Get RAW input features (not processed features)
        # This is what the ONNX voxel encoder expects as input
        if hasattr(self.pytorch_model.pts_voxel_encoder, 'get_input_features'):
            # Use get_input_features for ONNX models to get raw features
            # DO NOT call forward() - that would give processed features
            input_features = self.pytorch_model.pts_voxel_encoder.get_input_features(
                voxels_tensor, 
                num_points_tensor, 
                coors_tensor
            )
            logger.info(f"Got raw input features from PyTorch model: shape {input_features.shape}")
        else:
            # Standard model doesn't have get_input_features
            # We need to manually construct the input features
            raise NotImplementedError(
                "Standard voxel encoder not supported for ONNX inference. "
                "Please use CenterPointONNX with PillarFeatureNetONNX."
            )
        
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
            Processed outputs as numpy arrays (for verification compatibility)
        """
        # ONNX outputs are in the format: [heatmap, reg, height, dim, rot, vel]
        # Each output is a numpy array with shape [batch_size, channels, H, W]
        
        # Keep outputs as numpy arrays for verification compatibility
        # The verification code expects numpy arrays for comparison
        processed_outputs = []
        
        for output in onnx_outputs:
            if isinstance(output, np.ndarray):
                # Keep as numpy array for verification
                processed_outputs.append(output)
            else:
                # Convert to numpy if needed
                if hasattr(output, 'numpy'):
                    processed_outputs.append(output.numpy())
                else:
                    processed_outputs.append(np.array(output))
        
        logger.info(f"ONNX postprocess - output length: {len(processed_outputs)}")
        for i, output in enumerate(processed_outputs):
            if isinstance(output, np.ndarray):
                logger.info(f"  Output[{i}] shape: {output.shape}, dtype: {output.dtype}")
        
        return processed_outputs


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
