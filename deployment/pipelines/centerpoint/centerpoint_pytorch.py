"""
CenterPoint PyTorch Pipeline Implementation.

This module implements the CenterPoint pipeline using pure PyTorch,
providing a baseline for comparison with optimized backends.
"""

import logging
from typing import Dict, List, Tuple

import torch

from deployment.pipelines.centerpoint.centerpoint_pipeline import CenterPointDeploymentPipeline

logger = logging.getLogger(__name__)


class CenterPointPyTorchPipeline(CenterPointDeploymentPipeline):
    """
    PyTorch implementation of the staged CenterPoint deployment pipeline.

    Uses PyTorch for preprocessing, middle encoder, backbone, and head while
    sharing the same staged execution flow as ONNX/TensorRT backends.
    """

    def __init__(self, pytorch_model, device: str = "cuda"):
        """
        Initialize PyTorch pipeline.

        Args:
            pytorch_model: PyTorch CenterPoint model
            device: Device for inference
        """
        super().__init__(pytorch_model, device, backend_type="pytorch")
        logger.info("PyTorch pipeline initialized (ONNX-compatible staged inference)")

    def infer(self, points: torch.Tensor, sample_meta: Dict = None, return_raw_outputs: bool = False) -> Tuple:
        """
        Complete inference pipeline.

        Uses the shared staged pipeline defined in CenterPointDeploymentPipeline.

        Args:
            points: Input point cloud
            sample_meta: Sample metadata
            return_raw_outputs: If True, return raw head outputs (only for ONNX models)
        """
        if sample_meta is None:
            sample_meta = {}
        return super().infer(points, sample_meta, return_raw_outputs=return_raw_outputs)

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

        # Ensure output is 2D: [N_voxels, feature_dim]
        # ONNX-compatible models may output 3D tensor that needs squeezing
        if voxel_features.ndim == 3:
            # Try to squeeze to 2D
            if voxel_features.shape[1] == 1:
                # Shape: [N_voxels, 1, feature_dim] -> [N_voxels, feature_dim]
                voxel_features = voxel_features.squeeze(1)
            elif voxel_features.shape[2] == 1:
                # Shape: [N_voxels, feature_dim, 1] -> [N_voxels, feature_dim]
                voxel_features = voxel_features.squeeze(2)
            else:
                # Cannot determine which dimension to squeeze
                # This might be the input features [N_voxels, max_points, feature_dim]
                # which should have been processed by the encoder
                raise RuntimeError(
                    f"Voxel encoder output has unexpected 3D shape: {voxel_features.shape}. "
                    f"Expected 2D output [N_voxels, feature_dim]. "
                    f"This may indicate the voxel encoder didn't process the input correctly. "
                    f"Input features shape was: {input_features.shape}"
                )
        elif voxel_features.ndim > 3:
            raise RuntimeError(
                f"Voxel encoder output has {voxel_features.ndim}D shape: {voxel_features.shape}. "
                f"Expected 2D output [N_voxels, feature_dim]."
            )

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
            if hasattr(self.pytorch_model, "pts_neck") and self.pytorch_model.pts_neck is not None:
                x = self.pytorch_model.pts_neck(x)

            # Head - returns tuple of task head outputs
            head_outputs_tuple = self.pytorch_model.pts_bbox_head(x)

            # Handle two possible output formats:
            # 1. ONNX head: Tuple[torch.Tensor] directly (heatmap, reg, height, dim, rot, vel)
            # 2. Standard head: Tuple[List[Dict]] format

            if isinstance(head_outputs_tuple, tuple) and len(head_outputs_tuple) > 0:
                first_element = head_outputs_tuple[0]

                # Check if this is ONNX format (tuple of tensors)
                if isinstance(first_element, torch.Tensor):
                    # ONNX format: (heatmap, reg, height, dim, rot, vel)
                    head_outputs = list(head_outputs_tuple)

                elif isinstance(first_element, list) and len(first_element) > 0:
                    # Standard format: (List[Dict],)
                    preds_dict = first_element[0]  # Get first (and only) dict

                    # Extract individual outputs
                    head_outputs = [
                        preds_dict["heatmap"],
                        preds_dict["reg"],
                        preds_dict["height"],
                        preds_dict["dim"],
                        preds_dict["rot"],
                        preds_dict["vel"],
                    ]
                else:
                    raise ValueError(f"Unexpected task_outputs format: {type(first_element)}")
            else:
                raise ValueError(f"Unexpected head_outputs format: {type(head_outputs_tuple)}")

        return head_outputs
