"""
CenterPoint deployment utilities.

This module provides utility functions for CenterPoint model deployment,
including ONNX-compatible config creation and model building.
"""

import copy
from typing import Optional, Tuple

import torch
from mmengine.config import Config
from mmengine.registry import MODELS, init_default_scope
from mmengine.runner import load_checkpoint


def create_onnx_model_cfg(
    model_cfg: Config,
    device: str,
    rot_y_axis_reference: bool = False,
) -> Config:
    """
    Create an ONNX-friendly CenterPoint config.
    
    Args:
        model_cfg: Original model configuration
        device: Device string (e.g., "cpu", "cuda:0")
        rot_y_axis_reference: Whether to use y-axis rotation reference
        
    Returns:
        ONNX-compatible model configuration
    """
    onnx_cfg = model_cfg.copy()
    model_config = copy.deepcopy(onnx_cfg.model)

    model_config.type = "CenterPointONNX"
    model_config.point_channels = model_config.pts_voxel_encoder.in_channels
    model_config.device = device

    if model_config.pts_voxel_encoder.type == "PillarFeatureNet":
        model_config.pts_voxel_encoder.type = "PillarFeatureNetONNX"
    elif model_config.pts_voxel_encoder.type == "BackwardPillarFeatureNet":
        model_config.pts_voxel_encoder.type = "BackwardPillarFeatureNetONNX"

    model_config.pts_bbox_head.type = "CenterHeadONNX"
    model_config.pts_bbox_head.separate_head.type = "SeparateHeadONNX"
    model_config.pts_bbox_head.rot_y_axis_reference = rot_y_axis_reference

    if getattr(model_config, "pts_backbone", None) and \
       getattr(model_config.pts_backbone, "type", None) == "ConvNeXt_PC":
        model_config.pts_backbone.with_cp = False

    onnx_cfg.model = model_config
    return onnx_cfg


def build_model_from_cfg(model_cfg: Config, checkpoint_path: str, device: str) -> torch.nn.Module:
    """
    Build and load a model from config + checkpoint on the given device.
    
    Args:
        model_cfg: Model configuration
        checkpoint_path: Path to checkpoint file
        device: Device string (e.g., "cpu", "cuda:0")
        
    Returns:
        Loaded PyTorch model
    """
    init_default_scope("mmdet3d")
    model_config = copy.deepcopy(model_cfg.model)
    model = MODELS.build(model_config)
    model.to(device)
    load_checkpoint(model, checkpoint_path, map_location=device)
    model.eval()
    model.cfg = model_cfg
    return model


def build_centerpoint_onnx_model(
    base_model_cfg: Config,
    checkpoint_path: str,
    device: str,
    rot_y_axis_reference: bool = False,
) -> Tuple[torch.nn.Module, Config]:
    """
    Build an ONNX-friendly CenterPoint model from the *original* model_cfg.

    This is the single source of truth for building CenterPoint models from
    original config + checkpoint to ONNX-compatible model.

    Args:
        base_model_cfg: Original model configuration (mmdet3d config)
        checkpoint_path: Path to checkpoint file
        device: Device string (e.g., "cpu", "cuda:0")
        rot_y_axis_reference: Whether to use y-axis rotation reference
        
    Returns:
        Tuple of:
        - model: loaded torch.nn.Module (ONNX-compatible)
        - onnx_cfg: ONNX-compatible Config actually used to build the model
    """
    # 1) Convert original cfg to ONNX-friendly cfg
    onnx_cfg = create_onnx_model_cfg(
        base_model_cfg,
        device=device,
        rot_y_axis_reference=rot_y_axis_reference,
    )

    # 2) Use shared build_model_from_cfg to load checkpoint
    model = build_model_from_cfg(onnx_cfg, checkpoint_path, device=device)

    return model, onnx_cfg

