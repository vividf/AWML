"""
Constants for CenterPoint export workflows.

These constants define the export file structure for CenterPoint models.
They are kept in the deployment package since they are part of the
export interface, not project-specific configuration.
"""

from typing import Tuple

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

# Ordered list of ONNX to TensorRT file mappings
ONNX_TO_TRT_MAPPINGS: Tuple[Tuple[str, str], ...] = (
    (VOXEL_ENCODER_ONNX, VOXEL_ENCODER_ENGINE),
    (BACKBONE_HEAD_ONNX, BACKBONE_HEAD_ENGINE),
)

__all__ = [
    "VOXEL_ENCODER_NAME",
    "BACKBONE_HEAD_NAME",
    "VOXEL_ENCODER_ONNX",
    "BACKBONE_HEAD_ONNX",
    "VOXEL_ENCODER_ENGINE",
    "BACKBONE_HEAD_ENGINE",
    "ONNX_TO_TRT_MAPPINGS",
]
