"""
CenterPoint model loading utilities.

This module provides ONNX-compatible model building from MMEngine configs, with an optional
PTQ/QAT quantized-checkpoint load path (BN fuse + Q/DQ insert + weight load) so the built graph
matches a quantized state_dict.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Optional

import torch
from mmengine.config import Config
from mmengine.registry import MODELS, init_default_scope

from deployment.config.schema import QuantizationConfig
from deployment.io.mmdet3d_model import build_mmdet3d_model
from deployment.primitives.device import DeviceSpec

# Imported for their side effect: registering CenterPoint's ONNX module variants into the
# MMDet3D registries so ``MODELS.build`` can resolve them during export.
from deployment.projects.centerpoint.export.onnx_models import (  # noqa: F401
    centerpoint_head_onnx,
    centerpoint_onnx,
    pillar_encoder_onnx,
)

logger = logging.getLogger(__name__)


def create_onnx_model_cfg(
    model_cfg: Config,
    device: DeviceSpec,
    rot_y_axis_reference: bool = False,
) -> Config:
    """Create a model config that swaps modules to ONNX-friendly variants.

    This mutates the `model_cfg.model` subtree to reference classes registered by
    `deployment.projects.centerpoint.export.onnx_models` (e.g., `CenterPointONNX`).

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


def build_centerpoint_model(
    model_cfg: Config,
    checkpoint_path: str,
    device: DeviceSpec,
    *,
    rot_y_axis_reference: bool = False,
    quantization: Optional[QuantizationConfig] = None,
) -> torch.nn.Module:
    """Build a CenterPoint model from config and load checkpoint weights (for export + reference eval).

    Swaps the model config to CenterPoint's ONNX-friendly module variants, then builds and
    loads it. The plain path uses the shared mmdet3d model core (mirrors
    :func:`build_bevfusion_model`); when ``quantization.enabled`` is True the model is built and
    then loaded via :func:`_load_quantized_checkpoint` (BN fuse + Q/DQ insert + weight load) so a
    PTQ/QAT ``state_dict`` lines up with the built (quantized) graph.

    Args:
        model_cfg: MMEngine model configuration.
        checkpoint_path: Path to the checkpoint file.
        device: Target device specification.
        rot_y_axis_reference: Whether to use y-axis rotation reference.
        quantization: Typed ``quantization`` section (parsed once by ``BaseDeploymentConfig``).
            When ``enabled``, loads the checkpoint via ``_load_quantized_checkpoint``.

    Returns:
        The loaded model in eval mode; the export config it was built from is available as ``model.cfg``.
    """
    export_model_cfg = create_onnx_model_cfg(
        model_cfg,
        device=device,
        rot_y_axis_reference=rot_y_axis_reference,
    )

    qcfg = quantization or QuantizationConfig()
    if qcfg.enabled:
        init_default_scope("mmdet3d")
        model = MODELS.build(copy.deepcopy(export_model_cfg.model))
        torch_device = device.to_torch_device()
        model.to(torch_device)
        model = _load_quantized_checkpoint(model, checkpoint_path, str(torch_device), qcfg)
        model.eval()
        model.cfg = export_model_cfg
        return model

    return build_mmdet3d_model(export_model_cfg, checkpoint_path, device)


def _load_quantized_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str,
    config: QuantizationConfig,
) -> torch.nn.Module:
    """Load a quantized (PTQ/QAT) checkpoint into a model.

    Applies the same transformations used during quantization (BN fusion,
    Q/DQ node insertion) before loading the checkpoint so that the state_dict
    keys match exactly.

    Args:
        model: Model to load checkpoint into.
        checkpoint_path: Path to quantized checkpoint.
        device: Device string.
        config: Typed quantization config (parsed once by ``BaseDeploymentConfig``).

    Returns:
        Model with quantized checkpoint loaded.
    """
    try:
        from deployment.projects.centerpoint.quantization.plan import build_centerpoint_plan
        from deployment.quantization import (
            CalibrationManager,
            disable_quantizers_in,
            expand_keep_fp16,
            move_quantizer_amax_to_device,
            setup_quantization_for_onnx_export,
        )
    except ImportError as e:
        raise ImportError(
            "Quantization modules not found. Make sure deployment/quantization " f"is properly installed. Error: {e}"
        )

    logger.info("Loading quantized checkpoint with transformations...")

    # 1-2. Build the SAME quantized module tree the PTQ producer built (BN fuse + Q/DQ insert),
    # via the shared plan, so the calibrated state_dict lines up on load.
    logger.info(
        "Building quantized tree via CenterPoint plan (fuse_bn=%s, keep_fp16=%s, disable_recipes=%s)",
        config.fuse_bn,
        list(config.keep_fp16),
        list(config.disable_recipes),
    )
    build_centerpoint_plan(config).prepare(model)
    # Resolve keep_fp16 → concrete module names for the disable loop below (same expansion the scheme
    # used; log=False to avoid duplicate per-pattern match logging).
    skip_layers = expand_keep_fp16(model, config.keep_fp16, log=False)

    # 2.5 Load calibration cache if provided
    calib_cache_path = config.calib_cache_path
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
    move_quantizer_amax_to_device(model, device)

    # 5. Validate quantizer amax values (TensorRT requires positive scales)
    _validate_quantizer_amax(model)

    # 6. Disable quantizers in the keep_fp16 subtrees — the SAME shared loop the PTQ producer and
    # QAT hook run, so producer and deploy sides can never diverge on match semantics.
    disable_quantizers_in(model, skip_layers)

    # 7. Configure the quantization backend for proper ONNX export
    setup_quantization_for_onnx_export()

    logger.info("Quantized checkpoint loaded successfully")
    return model


def _validate_quantizer_amax(model: torch.nn.Module) -> None:
    """Validate TensorQuantizers have valid amax values (TensorRT requires positive scales).

    Skips quantizers that are disabled. Disabled quantizers are not used in forward
    and may have amax=nan from never being calibrated.

    CenterPoint-only for now: the BEVFusion loader does not run this check (it instead forces
    quantizers into inference mode post-load — ``_set_tensor_quantizers_inference_mode``).
    TODO(Docker): decide whether both loaders should run both steps and, if so, move this next to
    ``move_quantizer_amax_to_device`` in ``deployment.quantization.core.utils`` (spec.md §5.2 4B.3).
    """
    from deployment.quantization import get_tensor_quantizer_cls

    tensor_quantizer_cls = get_tensor_quantizer_cls()
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
