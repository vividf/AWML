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


def build_model_from_cfg(
    model_cfg: Config,
    checkpoint_path: str,
    device: str,
    quantization: Optional[dict] = None,
) -> torch.nn.Module:
    """
    Build and load a model from config + checkpoint on the given device.

    Args:
        model_cfg: Model configuration
        checkpoint_path: Path to checkpoint file
        device: Device string (e.g., "cpu", "cuda:0")
        quantization: Optional quantization config dict with keys:
            - enabled: bool, whether to enable quantization
            - mode: str, 'ptq' or 'qat'
            - fuse_bn: bool, whether to fuse BatchNorm (default: True)

    Returns:
        Loaded PyTorch model
    """
    init_default_scope("mmdet3d")
    model_config = copy.deepcopy(model_cfg.model)
    model = MODELS.build(model_config)
    model.to(device)

    # Handle quantized checkpoint loading
    if quantization and quantization.get("enabled", False):
        model = _load_quantized_checkpoint(model, checkpoint_path, device, quantization)
    else:
        load_checkpoint(model, checkpoint_path, map_location=device)

    model.eval()
    model.cfg = model_cfg
    return model


def _load_quantized_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str,
    quantization: dict,
) -> torch.nn.Module:
    """
    Load a quantized (PTQ/QAT) checkpoint into a model.

    This function applies the same transformations that were applied during
    quantization (BN fusion, Q/DQ node insertion) before loading the checkpoint.

    Args:
        model: Model to load checkpoint into
        checkpoint_path: Path to quantized checkpoint
        device: Device string
        quantization: Quantization config dict

    Returns:
        Model with quantized checkpoint loaded
    """
    try:
        from projects.CenterPoint.quantization import fuse_model_bn, quant_model
    except ImportError as e:
        raise ImportError(
            "Quantization modules not found. Make sure projects/CenterPoint/quantization "
            f"is properly installed. Error: {e}"
        )

    logger = logging.getLogger(__name__)
    logger.info("Loading quantized checkpoint with transformations...")

    # 1. Fuse BatchNorm if enabled (must be done before quantization)
    fuse_bn = quantization.get("fuse_bn", True)
    if fuse_bn:
        logger.info("Fusing BatchNorm layers...")
        model.eval()
        fuse_model_bn(model)

    # 2. Insert Q/DQ nodes
    logger.info("Inserting Q/DQ nodes...")
    skip_layers = set(quantization.get("sensitive_layers", []))
    quant_model(model, skip_names=skip_layers)

    # 3. Load the quantized checkpoint
    logger.info(f"Loading quantized checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Load with strict=False to handle any minor mismatches
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning(f"Missing keys in checkpoint: {len(missing)} keys")
        logger.debug(f"Missing keys: {missing[:10]}...")  # Show first 10
    if unexpected:
        logger.warning(f"Unexpected keys in checkpoint: {len(unexpected)} keys")
        logger.debug(f"Unexpected keys: {unexpected[:10]}...")  # Show first 10

    # 4. Move all quantizer amax values to the target device
    # This is necessary because TensorQuantizer stores _amax as a buffer
    # and we need to ensure it's on the same device as the inputs
    _move_quantizer_amax_to_device(model, device, logger)

    # 5. Disable quantization for sensitive layers (e.g., ConvTranspose2d)
    # These layers may not have TensorRT INT8 support
    _disable_quantization_for_sensitive_layers(model, skip_layers, logger)

    # 6. Configure pytorch-quantization for proper ONNX export
    # This enables Q/DQ nodes to be exported as QuantizeLinear/DequantizeLinear
    setup_quantization_for_onnx_export()

    logger.info("Quantized checkpoint loaded successfully")
    return model


def _move_quantizer_amax_to_device(
    model: torch.nn.Module,
    device: str,
    logger: logging.Logger,
) -> None:
    """
    Move all TensorQuantizer amax values to the specified device.

    This is necessary because TensorQuantizer stores _amax as a buffer,
    and when loading a checkpoint calibrated on GPU to CPU (or vice versa),
    the _amax tensors may be on the wrong device.

    Args:
        model: Model containing TensorQuantizers
        device: Target device
        logger: Logger instance
    """
    try:
        from pytorch_quantization.nn import TensorQuantizer
    except ImportError:
        return

    moved_count = 0
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            if hasattr(module, "_amax") and module._amax is not None:
                if module._amax.device != torch.device(device):
                    module._amax = module._amax.to(device)
                    moved_count += 1

    if moved_count > 0:
        logger.info(f"Moved {moved_count} quantizer amax tensors to {device}")


def _disable_quantization_for_sensitive_layers(
    model: torch.nn.Module,
    sensitive_layers: set,
    logger: logging.Logger,
) -> None:
    """
    Disable quantization for sensitive layers.

    Some layers (e.g., ConvTranspose2d) don't have good TensorRT INT8 support.
    This function disables the quantizers for these layers so they won't have
    Q/DQ nodes in the exported ONNX.

    Args:
        model: Model containing quantized layers
        sensitive_layers: Set of layer names to disable quantization for
        logger: Logger instance
    """
    if not sensitive_layers:
        return

    try:
        from pytorch_quantization.nn import TensorQuantizer
    except ImportError:
        return

    disabled_count = 0
    for name, module in model.named_modules():
        # Check if this module or its parent is in the sensitive layers list
        should_disable = False
        for sensitive_name in sensitive_layers:
            if name.startswith(sensitive_name) and isinstance(module, TensorQuantizer):
                should_disable = True
                break

        if should_disable:
            module.disable()
            disabled_count += 1
            logger.debug(f"Disabled quantizer: {name}")

    if disabled_count > 0:
        logger.info(f"Disabled {disabled_count} quantizers for sensitive layers: {sensitive_layers}")


def setup_quantization_for_onnx_export() -> None:
    """
    Configure pytorch-quantization for proper ONNX export.

    This function enables 'use_fb_fake_quant' mode which makes TensorQuantizer
    export as proper ONNX QuantizeLinear/DequantizeLinear nodes instead of
    custom ops that TensorRT can't recognize.

    Must be called before ONNX export when using quantized models.
    """
    try:
        from pytorch_quantization.nn import TensorQuantizer

        # Enable FakeQuantize export mode for proper ONNX Q/DQ nodes
        # This makes TensorQuantizer export as QuantizeLinear/DequantizeLinear
        TensorQuantizer.use_fb_fake_quant = True
        logging.getLogger(__name__).info("Enabled use_fb_fake_quant for ONNX export of quantized model")
    except ImportError:
        pass


def build_centerpoint_onnx_model(
    base_model_cfg: Config,
    checkpoint_path: str,
    device: str,
    rot_y_axis_reference: bool = False,
    quantization: Optional[dict] = None,
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
        quantization: Optional quantization config dict with keys:
            - enabled: bool, whether to enable quantization
            - mode: str, 'ptq' or 'qat'
            - fuse_bn: bool, whether to fuse BatchNorm (default: True)
            - sensitive_layers: list of layer names to skip quantization

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
    model = build_model_from_cfg(
        onnx_cfg,
        checkpoint_path,
        device=device,
        quantization=quantization,
    )

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
