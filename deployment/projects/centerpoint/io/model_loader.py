"""
CenterPoint model loading utilities.

This module provides ONNX-compatible model building from MMEngine configs.
"""

from __future__ import annotations

import copy
from typing import Tuple

import torch
from mmengine.config import Config
from mmengine.registry import MODELS, init_default_scope
from mmengine.runner import load_checkpoint

from deployment.core.device import DeviceSpec
from deployment.projects.centerpoint.onnx_models import (  # noqa: F401 - register MODELS
    centerpoint_head_onnx,
    centerpoint_onnx,
    pillar_encoder_onnx,
)


def create_onnx_model_cfg(
    model_cfg: Config,
    device: DeviceSpec,
    rot_y_axis_reference: bool = False,
) -> Config:
    """Create a model config that swaps modules to ONNX-friendly variants.

    This mutates the `model_cfg.model` subtree to reference classes registered by
    `deployment.projects.centerpoint.onnx_models` (e.g., `CenterPointONNX`).

    Args:
        model_cfg: Original MMEngine model configuration.
        device: Target device specification.
        rot_y_axis_reference: Whether to use y-axis rotation reference.

    Returns:
        New config whose ``model`` subtree builds the deployment export graph (e.g. ONNX-friendly types).
    """
    export_model_cfg = model_cfg.copy()
    model_config = copy.deepcopy(export_model_cfg.model)

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

    export_model_cfg.model = model_config
    return export_model_cfg


def build_model_from_cfg(
    model_cfg: Config,
    checkpoint_path: str,
    device: DeviceSpec,
) -> torch.nn.Module:
    """Build a model from MMEngine config and load checkpoint weights.

    Args:
        model_cfg: MMEngine model configuration.
        checkpoint_path: Path to the checkpoint file.
        device: Target device specification.
        quantization: Optional quantization config dict with keys:
            - enabled: bool, whether to enable quantization
            - mode: str, 'ptq' or 'qat'
            - fuse_bn: bool, whether to fuse BatchNorm (default: True)
            - quant_backbone, quant_neck, quant_head, quant_voxel_encoder: bool
            - quant_add, quant_linear_backbone: bool
            - quant_ese_mul_identity, quant_ese_pool_input: bool
            - quant_maxpool_input: bool
            - sensitive_layers: list of layer names to skip
            - skip_vovnet_stages: list of int (VoVNet only): 0=stem, 1=stage2, 2=stage3, 3=stage4;
              must match the config used during PTQ so model structure matches checkpoint
            - calib_cache_path: optional path to calibration cache

    Returns:
        Loaded and initialized PyTorch model in eval mode.
    """
    # Importing onnx_models above triggers MODELS registration for ONNX variants.
    init_default_scope("mmdet3d")

    model_config = copy.deepcopy(model_cfg.model)
    model = MODELS.build(model_config)
    torch_device = device.to_torch_device()
    model.to(torch_device)
    load_checkpoint(model, checkpoint_path, map_location=torch_device)
    model.eval()
    model.cfg = model_cfg
    return model


def _load_quantized_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str,
    quantization: dict,
) -> torch.nn.Module:
    """Load a quantized (PTQ/QAT) checkpoint into a model.

    Applies the same transformations used during quantization (BN fusion,
    Q/DQ node insertion) before loading the checkpoint so that the state_dict
    keys match exactly.

    Args:
        model: Model to load checkpoint into.
        checkpoint_path: Path to quantized checkpoint.
        device: Device string.
        quantization: Quantization config dict.

    Returns:
        Model with quantized checkpoint loaded.
    """
    try:
        from deployment.quantization import (
            CalibrationManager,
            fuse_model_bn,
            quant_model,
        )
    except ImportError as e:
        raise ImportError(
            "Quantization modules not found. Make sure deployment/quantization " f"is properly installed. Error: {e}"
        )

    logger.info("Loading quantized checkpoint with transformations...")

    # 1. Fuse BatchNorm if enabled (must be done before quantization)
    fuse_bn = quantization.get("fuse_bn", True)
    if fuse_bn:
        logger.info("Fusing BatchNorm layers...")
        model.eval()
        fuse_model_bn(model)

    # 2. Insert Q/DQ nodes
    logger.info("Inserting Q/DQ nodes...")
    skip_layers = _build_skip_layers(quantization)

    logger.info(
        "Quantization flags: backbone=%s, neck=%s, head=%s, voxel_encoder=%s, "
        "add=%s, linear_backbone=%s, quant_ese_mul_identity=%s, quant_ese_pool_input=%s, quant_maxpool_input=%s",
        bool(quantization.get("quant_backbone", True)),
        bool(quantization.get("quant_neck", True)),
        bool(quantization.get("quant_head", True)),
        bool(quantization.get("quant_voxel_encoder", True)),
        bool(quantization.get("quant_add", False)),
        bool(quantization.get("quant_linear_backbone", False)),
        bool(quantization.get("quant_ese_mul_identity", False)),
        bool(quantization.get("quant_ese_pool_input", False)),
        bool(quantization.get("quant_maxpool_input", False)),
    )

    quant_model(
        model,
        quant_backbone=bool(quantization.get("quant_backbone", True)),
        quant_neck=bool(quantization.get("quant_neck", True)),
        quant_head=bool(quantization.get("quant_head", True)),
        quant_voxel_encoder=bool(quantization.get("quant_voxel_encoder", True)),
        quant_add=bool(quantization.get("quant_add", False)),
        quant_linear_backbone=bool(quantization.get("quant_linear_backbone", False)),
        quant_ese_mul_identity=bool(quantization.get("quant_ese_mul_identity", False)),
        quant_ese_pool_input=bool(quantization.get("quant_ese_pool_input", False)),
        quant_maxpool_input=bool(quantization.get("quant_maxpool_input", False)),
        skip_names=skip_layers,
    )

    # 2.5 Load calibration cache if provided
    calib_cache_path = quantization.get("calib_cache_path")
    if calib_cache_path:
        if os.path.exists(calib_cache_path):
            logger.info(f"Loading calibration cache from {calib_cache_path}...")
            calibrator = CalibrationManager(model)
            calibrator.load_calib_cache(calib_cache_path)
        else:
            logger.warning(f"Calibration cache not found: {calib_cache_path}")

    # 3. Load the quantized checkpoint
    logger.info(f"Loading quantized checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning(f"Missing keys in checkpoint: {len(missing)} keys")
        logger.debug(f"Missing keys: {missing[:10]}...")
    if unexpected:
        logger.warning(f"Unexpected keys in checkpoint: {len(unexpected)} keys")
        logger.debug(f"Unexpected keys: {unexpected[:10]}...")

    # 4. Move all quantizer amax values to the target device
    _move_quantizer_amax_to_device(model, device)

    # 5. Validate quantizer amax values (TensorRT requires positive scales)
    _validate_quantizer_amax(model)

    # 6. Disable quantization for sensitive layers
    _disable_quantization_for_sensitive_layers(model, skip_layers)

    # 7. Configure pytorch-quantization for proper ONNX export
    setup_quantization_for_onnx_export()

    logger.info("Quantized checkpoint loaded successfully")
    return model


def _build_skip_layers(quantization: dict) -> Set[str]:
    """Build the set of layer name prefixes to skip during quantization."""
    skip_layers: Set[str] = set(quantization.get("sensitive_layers", []) or [])

    # SECOND/ResNet: pts_backbone.blocks.*
    skip_first = int(quantization.get("skip_backbone_first_stages", 0) or 0)
    if skip_first > 0:
        for i in range(skip_first):
            skip_layers.add(f"pts_backbone.blocks.{i}")
    for i in quantization.get("skip_backbone_stages", []) or []:
        skip_layers.add(f"pts_backbone.blocks.{int(i)}")

    # VoVNet: pts_backbone.stem, .stage2, .stage3, .stage4 (must match PTQ so model structure = checkpoint)
    vovnet_stages = quantization.get("skip_vovnet_stages", None)
    if vovnet_stages is not None:
        _vovnet_names = ["stem", "stage2", "stage3", "stage4"]
        for idx in vovnet_stages:
            i = int(idx)
            if 0 <= i < len(_vovnet_names):
                skip_layers.add(f"pts_backbone.{_vovnet_names[i]}")

    return skip_layers


def _move_quantizer_amax_to_device(model: torch.nn.Module, device: str) -> None:
    """Move all TensorQuantizer amax values to the specified device."""
    tensor_quantizer_cls = _import_tensor_quantizer()
    if tensor_quantizer_cls is None:
        return

    moved_count = 0
    for _name, module in model.named_modules():
        if isinstance(module, tensor_quantizer_cls):
            if hasattr(module, "_amax") and module._amax is not None:
                if module._amax.device != torch.device(device):
                    module._amax = module._amax.to(device)
                    moved_count += 1

    if moved_count > 0:
        logger.info(f"Moved {moved_count} quantizer amax tensors to {device}")


def _disable_quantization_for_sensitive_layers(
    model: torch.nn.Module,
    sensitive_layers: Set[str],
) -> None:
    """Disable quantization for sensitive layers (e.g. ConvTranspose2d with limited INT8 support)."""
    if not sensitive_layers:
        return

    tensor_quantizer_cls = _import_tensor_quantizer()
    if tensor_quantizer_cls is None:
        return

    disabled_count = 0
    for name, module in model.named_modules():
        should_disable = False
        for sensitive_name in sensitive_layers:
            if name.startswith(sensitive_name) and isinstance(module, tensor_quantizer_cls):
                should_disable = True
                break

        if should_disable:
            module.disable()
            disabled_count += 1
            logger.debug(f"Disabled quantizer: {name}")

    if disabled_count > 0:
        logger.info(f"Disabled {disabled_count} quantizers for sensitive layers: {sensitive_layers}")


def _validate_quantizer_amax(model: torch.nn.Module) -> None:
    """Validate TensorQuantizers have valid amax values (TensorRT requires positive scales).

    Skips quantizers that are disabled. Disabled quantizers are not used in forward
    and may have amax=nan from never being calibrated.
    """
    tensor_quantizer_cls = _import_tensor_quantizer()
    if tensor_quantizer_cls is None:
        return

    invalid_names = []
    for name, module in model.named_modules():
        if not isinstance(module, tensor_quantizer_cls):
            continue

        if getattr(module, "_disabled", False):
            continue

        amax = getattr(module, "_amax", None)
        if amax is None:
            invalid_names.append((name, "amax=None"))
            continue

        if torch.is_tensor(amax):
            invalid = (not torch.isfinite(amax).all()) or (amax <= 0).any()
        else:
            try:
                invalid = (not torch.isfinite(torch.tensor(amax))) or (amax <= 0)
            except Exception:
                invalid = True

        if invalid:
            invalid_names.append((name, f"amax={amax}"))

    if invalid_names:
        for name, reason in invalid_names:
            logger.error(f"Invalid quantizer amax: {name} ({reason})")
        raise RuntimeError(f"Found {len(invalid_names)} TensorQuantizer modules with invalid amax values")


def setup_quantization_for_onnx_export() -> None:
    """Configure pytorch-quantization for proper ONNX export.

    Enables 'use_fb_fake_quant' mode so that TensorQuantizer exports as
    QuantizeLinear/DequantizeLinear ONNX ops (recognized by TensorRT).
    Must be called before ONNX export when using quantized models.
    """
    tensor_quantizer_cls = _import_tensor_quantizer()
    if tensor_quantizer_cls is None:
        return

    tensor_quantizer_cls.use_fb_fake_quant = True
    logger.info("Enabled use_fb_fake_quant for ONNX export of quantized model")


def build_centerpoint_onnx_model(
    base_model_cfg: Config,
    checkpoint_path: str,
    device: DeviceSpec,
    rot_y_axis_reference: bool = False,
) -> Tuple[torch.nn.Module, Config]:
    """Build an ONNX-compatible CenterPoint model.

    Convenience wrapper that creates ONNX config and builds the model.

    Args:
        base_model_cfg: Base MMEngine model configuration.
        checkpoint_path: Path to the checkpoint file.
        device: Target device specification.
        rot_y_axis_reference: Whether to use y-axis rotation reference.

    Returns:
        Tuple of ``(model, export_model_cfg)``; the latter matches ``model.cfg``.
    """
    export_model_cfg = create_onnx_model_cfg(
        base_model_cfg,
        device=device,
        rot_y_axis_reference=rot_y_axis_reference,
    )
    model = build_model_from_cfg(export_model_cfg, checkpoint_path, device=device)
    return model, export_model_cfg
