"""
Constants for CenterPoint export workflows.

These constants define the export file structure and model architecture
for CenterPoint models. They are kept in the deployment package since
they are part of the export interface.
"""

from typing import Tuple

# =============================================================================
# Model Architecture Constants
# =============================================================================

# CenterPoint head output names (tied to CenterHead architecture)
# Order matters for ONNX export
OUTPUT_NAMES: Tuple[str, ...] = ("heatmap", "reg", "height", "dim", "rot", "vel")

# =============================================================================
# Export File Structure Constants
# =============================================================================

# CenterPoint component names for multi-file ONNX export
# These match the model architecture (voxel encoder + backbone/neck/head)
VOXEL_ENCODER_NAME: str = "pts_voxel_encoder"
BACKBONE_HEAD_NAME: str = "pts_backbone_neck_head"

# ONNX file names
VOXEL_ENCODER_ONNX: str = f"{VOXEL_ENCODER_NAME}.onnx"
BACKBONE_HEAD_ONNX: str = f"{BACKBONE_HEAD_NAME}.onnx"

# TensorRT engine file names
VOXEL_ENCODER_ENGINE: str = f"{VOXEL_ENCODER_NAME}.engine"
BACKBONE_HEAD_ENGINE: str = f"{BACKBONE_HEAD_NAME}.engine"

__all__ = [
    # Model architecture
    "OUTPUT_NAMES",
    # Export file structure
    "VOXEL_ENCODER_NAME",
    "BACKBONE_HEAD_NAME",
    "VOXEL_ENCODER_ONNX",
    "BACKBONE_HEAD_ONNX",
    "VOXEL_ENCODER_ENGINE",
    "BACKBONE_HEAD_ENGINE",
]
