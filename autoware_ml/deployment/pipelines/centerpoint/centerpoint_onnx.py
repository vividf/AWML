"""
CenterPoint ONNX Pipeline Implementation.

This module implements the CenterPoint pipeline using ONNX Runtime,
optimizing voxel encoder and backbone/head while keeping middle encoder in PyTorch.
"""

import logging
import os
from typing import List

import numpy as np
import onnxruntime as ort
import torch

from .centerpoint_pipeline import CenterPointDeploymentPipeline


logger = logging.getLogger(__name__)


class CenterPointONNXPipeline(CenterPointDeploymentPipeline):
    """
    ONNX Runtime implementation of CenterPoint pipeline.
    
    Uses ONNX Runtime for voxel encoder and backbone/head,
    while keeping middle encoder in PyTorch (sparse convolution cannot be converted).
    
    Provides good cross-platform compatibility and moderate speedup.
    """
    
    def __init__(self, pytorch_model, onnx_dir: str, device: str = "cpu"):
        """
        Initialize ONNX pipeline.
        
        Args:
            pytorch_model: PyTorch model (for preprocessing, middle encoder, postprocessing)
            onnx_dir: Directory containing ONNX model files
            device: Device for inference ('cpu' or 'cuda')
        """
        super().__init__(pytorch_model, device, backend_type="onnx")
        
        self.onnx_dir = onnx_dir
        self._load_onnx_models(device)
        
        logger.info(f"ONNX pipeline initialized with models from: {onnx_dir}")
    
    def _load_onnx_models(self, device: str):
        """Load ONNX models for voxel encoder and backbone/head."""
        # Define model paths
        voxel_encoder_path = os.path.join(self.onnx_dir, "pts_voxel_encoder.onnx")
        backbone_head_path = os.path.join(self.onnx_dir, "pts_backbone_neck_head.onnx")
        
        # Verify files exist
        if not os.path.exists(voxel_encoder_path):
            raise FileNotFoundError(f"Voxel encoder ONNX not found: {voxel_encoder_path}")
        if not os.path.exists(backbone_head_path):
            raise FileNotFoundError(f"Backbone head ONNX not found: {backbone_head_path}")
        
        # Create session options
        so = ort.SessionOptions()
        # Disable graph optimization for numerical consistency with PyTorch
        # Graph optimizations can reorder operations and fuse layers, causing numerical differences
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.log_severity_level = 3  # ERROR level
        
        # Set execution providers
        # IMPORTANT: For verification, always use CPU provider for numerical consistency
        # with ONNX export (which is done on CPU). For evaluation, can use CUDA for speed.
        # This ensures verification passes when ONNX was exported on CPU.
        if device.startswith("cuda"):
            # For evaluation, can use CUDA for speed
            # For verification, should use CPU (device will be "cpu" when passed from verification)
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.info("Using CUDA execution provider for ONNX (evaluation mode)")
            logger.info("  Note: For verification, device should be 'cpu' to ensure numerical consistency")
        else:
            providers = ["CPUExecutionProvider"]
            logger.info("Using CPU execution provider for ONNX")
        
        # Load voxel encoder
        try:
            self.voxel_encoder_session = ort.InferenceSession(
                voxel_encoder_path,
                sess_options=so,
                providers=providers
            )
            logger.info(f"Loaded voxel encoder: {voxel_encoder_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load voxel encoder ONNX: {e}")
        
        # Load backbone + head
        try:
            self.backbone_head_session = ort.InferenceSession(
                backbone_head_path,
                sess_options=so,
                providers=providers
            )
            logger.info(f"Loaded backbone+head: {backbone_head_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load backbone+head ONNX: {e}")
    
    def run_voxel_encoder(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Run voxel encoder using ONNX Runtime.
        
        Args:
            input_features: Input features [N_voxels, max_points, feature_dim]
            
        Returns:
            voxel_features: Voxel features [N_voxels, feature_dim]
        """
        # Convert to numpy
        input_array = input_features.cpu().numpy().astype(np.float32)
        
        # Get input and output names from ONNX model
        input_name = self.voxel_encoder_session.get_inputs()[0].name
        output_name = self.voxel_encoder_session.get_outputs()[0].name
        
        # Run ONNX inference with explicit output name for consistency
        outputs = self.voxel_encoder_session.run(
            [output_name],  # Specify output name explicitly
            {input_name: input_array}
        )
        
        # Convert back to torch tensor
        voxel_features = torch.from_numpy(outputs[0]).to(self.device)
        
        # Squeeze middle dimension if present (ONNX may output 3D)
        if voxel_features.ndim == 3 and voxel_features.shape[1] == 1:
            voxel_features = voxel_features.squeeze(1)
        
        
        return voxel_features
    
    def run_backbone_head(self, spatial_features: torch.Tensor) -> List[torch.Tensor]:
        """
        Run backbone + neck + head using ONNX Runtime.
        
        Args:
            spatial_features: Spatial features [B, C, H, W]
            
        Returns:
            List of head outputs: [heatmap, reg, height, dim, rot, vel]
        """
        # Convert to numpy
        input_array = spatial_features.cpu().numpy().astype(np.float32)
        
        # Get input and output names from ONNX model
        input_name = self.backbone_head_session.get_inputs()[0].name
        output_names = [output.name for output in self.backbone_head_session.get_outputs()]
        
        # Run ONNX inference with explicit output names for consistency
        outputs = self.backbone_head_session.run(
            output_names,  # Specify output names explicitly
            {input_name: input_array}
        )
        
        # IMPORTANT: Reorder outputs to match expected order [heatmap, reg, height, dim, rot, vel]
        # ONNX export may use different order (e.g., ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap'])
        # But pipeline expects: [heatmap, reg, height, dim, rot, vel]
        expected_order = ['heatmap', 'reg', 'height', 'dim', 'rot', 'vel']
        
        # Create a mapping from output name to output value
        output_dict = {name: out for name, out in zip(output_names, outputs)}
        
        # Reorder outputs according to expected order
        head_outputs = []
        for expected_name in expected_order:
            if expected_name in output_dict:
                head_outputs.append(torch.from_numpy(output_dict[expected_name]).to(self.device))
            else:
                raise ValueError(f"Expected output '{expected_name}' not found in ONNX model outputs: {output_names}")
        
        if len(head_outputs) != 6:
            raise ValueError(f"Expected 6 head outputs, got {len(head_outputs)}")
        
        
        return head_outputs

