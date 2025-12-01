"""
CenterPoint deployment utilities.

This module provides utility functions for CenterPoint model deployment,
including ONNX-compatible config creation and model building.
"""

import copy
import logging
from typing import List, Optional, Tuple

import torch
from mmengine.config import Config
from mmengine.registry import MODELS, init_default_scope
from mmengine.runner import load_checkpoint

from deployment.core import Detection3DMetricsConfig


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

    if (
        getattr(model_config, "pts_backbone", None)
        and getattr(model_config.pts_backbone, "type", None) == "ConvNeXt_PC"
    ):
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


def extract_t4metric_v2_config(
    model_cfg: Config,
    class_names: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[Detection3DMetricsConfig]:
    """
    Extract T4MetricV2 configuration from model config.

    This function extracts evaluation settings from T4MetricV2 evaluator config
    in the model config to ensure deployment evaluation uses the same settings
    as training evaluation.

    Args:
        model_cfg: Model configuration (may contain val_evaluator or test_evaluator with T4MetricV2 settings)
        class_names: Optional list of class names. If not provided, will be extracted from model_cfg.
        logger: Optional logger instance for logging

    Returns:
        Detection3DMetricsConfig with settings from model_cfg, or None if T4MetricV2 config not found

    Note:
        Only supports T4MetricV2. T4Metric (v1) is not supported.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Get class names - must come from config or explicit parameter
    if class_names is None:
        if hasattr(model_cfg, "class_names"):
            class_names = model_cfg.class_names
        else:
            raise ValueError(
                "class_names must be provided either explicitly or via model_cfg.class_names. "
                "Check your model config file includes class_names definition."
            )

    # Try to extract T4MetricV2 configs from val_evaluator or test_evaluator
    evaluator_cfg = None
    if hasattr(model_cfg, "val_evaluator"):
        evaluator_cfg = model_cfg.val_evaluator
    elif hasattr(model_cfg, "test_evaluator"):
        evaluator_cfg = model_cfg.test_evaluator
    else:
        logger.warning("No val_evaluator or test_evaluator found in model_cfg")
        return None

    # Helper to get value from dict or ConfigDict
    def get_cfg_value(cfg, key, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    # Check if evaluator is T4MetricV2
    evaluator_type = get_cfg_value(evaluator_cfg, "type")
    if evaluator_type != "T4MetricV2":
        logger.warning(
            f"Evaluator type is '{evaluator_type}', not 'T4MetricV2'. " "Only T4MetricV2 is supported. Returning None."
        )
        return None

    logger.info("=" * 60)
    logger.info("Detected T4MetricV2 config!")
    logger.info("Extracting evaluation settings for deployment...")
    logger.info("=" * 60)

    # Extract perception_evaluator_configs
    perception_configs = get_cfg_value(evaluator_cfg, "perception_evaluator_configs", {})
    evaluation_config_dict = get_cfg_value(perception_configs, "evaluation_config_dict")
    frame_id = get_cfg_value(perception_configs, "frame_id", "base_link")

    # Extract critical_object_filter_config
    critical_object_filter_config = get_cfg_value(evaluator_cfg, "critical_object_filter_config")

    # Extract frame_pass_fail_config
    frame_pass_fail_config = get_cfg_value(evaluator_cfg, "frame_pass_fail_config")

    # Convert ConfigDict to regular dict if needed
    if evaluation_config_dict and hasattr(evaluation_config_dict, "to_dict"):
        evaluation_config_dict = dict(evaluation_config_dict)
    if critical_object_filter_config and hasattr(critical_object_filter_config, "to_dict"):
        critical_object_filter_config = dict(critical_object_filter_config)
    if frame_pass_fail_config and hasattr(frame_pass_fail_config, "to_dict"):
        frame_pass_fail_config = dict(frame_pass_fail_config)

    logger.info(f"Extracted settings:")
    logger.info(f"  - frame_id: {frame_id}")
    if evaluation_config_dict:
        logger.info(f"  - evaluation_config_dict: {list(evaluation_config_dict.keys())}")
        if "center_distance_bev_thresholds" in evaluation_config_dict:
            logger.info(
                f"    - center_distance_bev_thresholds: " f"{evaluation_config_dict['center_distance_bev_thresholds']}"
            )
    if critical_object_filter_config:
        logger.info(f"  - critical_object_filter_config: enabled")
        if "max_distance_list" in critical_object_filter_config:
            logger.info(f"    - max_distance_list: " f"{critical_object_filter_config['max_distance_list']}")
    if frame_pass_fail_config:
        logger.info(f"  - frame_pass_fail_config: enabled")
        if "matching_threshold_list" in frame_pass_fail_config:
            logger.info(f"    - matching_threshold_list: " f"{frame_pass_fail_config['matching_threshold_list']}")
    logger.info("=" * 60)

    return Detection3DMetricsConfig(
        class_names=class_names,
        frame_id=frame_id,
        evaluation_config_dict=evaluation_config_dict,
        critical_object_filter_config=critical_object_filter_config,
        frame_pass_fail_config=frame_pass_fail_config,
    )
