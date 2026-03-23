"""Spconv INT8 quantization for BEVFusion sparse encoder.

Uses spconv's torch.ao.quantization FX graph mode to quantize
the sparse encoder with real INT8 (cumm kernels).

Flow: prepare_fx → calibrate → convert_fx → transform_qdq → remove_conv_add_dq
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _fuse_spconv_bn_in_encoder(sparse_encoder: nn.Module) -> int:
    """Fuse BatchNorm into sparse convolutions inside the given sparse encoder.

    Used by PTQ (before prepare_fx) and by deployment model_loader so that
    state_dict keys match. Returns the number of fused Conv-BN pairs.
    """
    try:
        from spconv.pytorch.quantization.utils import fuse_spconv_bn_eval
    except ImportError:
        logger.warning("spconv quantization utils not available")
        return 0

    from spconv.pytorch.conv import SparseConvolution

    sparse_encoder.eval()
    fused_count = 0

    for module in sparse_encoder.modules():
        children = list(module._modules.items())
        for i in range(len(children) - 1):
            left_name, left_mod = children[i]
            right_name, right_mod = children[i + 1]
            if isinstance(left_mod, SparseConvolution) and isinstance(right_mod, torch.nn.BatchNorm1d):
                fused_conv = fuse_spconv_bn_eval(left_mod, right_mod)
                setattr(module, left_name, fused_conv)
                setattr(module, right_name, torch.nn.Identity())
                fused_count += 1

    return fused_count


def _get_spconv_quantization_imports():
    """Lazily import spconv quantization utilities."""
    from spconv.pytorch.quantization import (
        get_default_spconv_qconfig_mapping,
        get_spconv_backend_config,
        get_spconv_convert_custom_config,
        get_spconv_prepare_custom_config,
        prepare_spconv_torch_inference,
        remove_conv_add_dq,
        transform_qdq,
    )
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    return {
        "prepare_fx": prepare_fx,
        "convert_fx": convert_fx,
        "get_default_spconv_qconfig_mapping": get_default_spconv_qconfig_mapping,
        "get_spconv_backend_config": get_spconv_backend_config,
        "get_spconv_convert_custom_config": get_spconv_convert_custom_config,
        "get_spconv_prepare_custom_config": get_spconv_prepare_custom_config,
        "prepare_spconv_torch_inference": prepare_spconv_torch_inference,
        "remove_conv_add_dq": remove_conv_add_dq,
        "transform_qdq": transform_qdq,
    }


def _ensure_torch_device(device: Union[torch.device, str]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    raise TypeError(f"Expected torch.device or str for device, got {type(device)!r}")


def _create_example_inputs(
    model: nn.Module,
    device: torch.device,
    in_channels: int = 5,
    num_voxels: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Create example inputs for FX tracing of the sparse encoder.

    Args:
        model: The sparse encoder module.
        device: Target device.
        in_channels: Number of input channels for voxel features.
        num_voxels: Number of example voxels.

    Returns:
        Tuple of (voxel_features, coors, batch_size).
    """
    dev = _ensure_torch_device(device)
    sparse_shape = getattr(model, "sparse_shape", [41, 1440, 1440])

    voxel_features = torch.randn((num_voxels, in_channels), device=dev)
    coors = torch.zeros((num_voxels, 4), dtype=torch.int32, device=dev)
    for i in range(num_voxels):
        coors[i, 0] = 0
        coors[i, 1] = i % sparse_shape[0]
        coors[i, 2] = i % sparse_shape[1]
        coors[i, 3] = i % sparse_shape[2]

    return voxel_features, coors, 1


def _enable_spconv_fx_trace_mode() -> None:
    """Spconv requires FX trace mode during prepare_fx (see spconv example/mnist, SPCONV_FX_TRACE_MODE).

    Disables strict SparseConvTensor __init__ checks and avoids trace failures from symbolic tensors.
    Must update both ``spconv.constants`` (source of truth) and ``spconv.pytorch.core`` (imported name).
    """
    try:
        import spconv.constants as spconv_constants
        import spconv.pytorch.core as spconv_core

        spconv_constants.SPCONV_FX_TRACE_MODE = True
        spconv_core.SPCONV_FX_TRACE_MODE = True
    except Exception:
        pass


def apply_spconv_int8_quantization(
    sparse_encoder: nn.Module,
    device: torch.device,
    in_channels: int = 5,
) -> nn.Module:
    """Apply spconv INT8 quantization to the sparse encoder using FX graph mode.

    This performs: prepare_fx → returns prepared model ready for calibration.
    After calibration, call convert_spconv_int8() to finalize.

    Args:
        sparse_encoder: The BEVFusionSparseEncoder module.
        device: Target device.
        in_channels: Number of voxel feature channels.

    Returns:
        Prepared sparse encoder (with observers inserted, ready for calibration).
    """
    _enable_spconv_fx_trace_mode()
    imports = _get_spconv_quantization_imports()

    imports["prepare_spconv_torch_inference"](with_linear=False)

    qconfig_mapping = imports["get_default_spconv_qconfig_mapping"](is_qat=False)
    backend_config = imports["get_spconv_backend_config"]()
    prepare_custom_config = imports["get_spconv_prepare_custom_config"]()

    example_inputs = _create_example_inputs(sparse_encoder, device, in_channels=in_channels)

    sparse_encoder.eval()
    logger.info("Running prepare_fx on sparse encoder for INT8 quantization...")
    prepared = imports["prepare_fx"](
        sparse_encoder,
        qconfig_mapping,
        example_inputs,
        backend_config=backend_config,
        prepare_custom_config=prepare_custom_config,
    )

    logger.info("Sparse encoder prepared for INT8 calibration")
    return prepared


def calibrate_spconv_model(
    prepared_encoder: nn.Module,
    calibration_data: List[Tuple[torch.Tensor, torch.Tensor, int]],
) -> None:
    """Run calibration data through the prepared sparse encoder.

    Args:
        prepared_encoder: Prepared (with observers) sparse encoder.
        calibration_data: List of (voxel_features, coors, batch_size) tuples.
    """
    prepared_encoder.eval()
    logger.info(f"Calibrating sparse encoder with {len(calibration_data)} samples...")

    with torch.no_grad():
        for i, (voxel_features, coors, batch_size) in enumerate(calibration_data):
            try:
                prepared_encoder(voxel_features, coors, batch_size)
                logger.debug(f"  Calibration sample {i + 1}/{len(calibration_data)}")
            except Exception as e:
                logger.warning(f"  Calibration sample {i + 1} failed: {e}")

    logger.info("Sparse encoder calibration complete")


def convert_spconv_int8(prepared_encoder: nn.Module) -> nn.Module:
    """Convert a calibrated prepared model to quantized INT8.

    Args:
        prepared_encoder: Calibrated prepared sparse encoder.

    Returns:
        Quantized sparse encoder using cumm INT8 kernels.
    """
    from deployment.projects.bevfusion.quantization.spconv_quantized_add_patch import (
        ensure_spconv_quantized_add_sparse_support,
    )

    ensure_spconv_quantized_add_sparse_support()

    imports = _get_spconv_quantization_imports()

    backend_config = imports["get_spconv_backend_config"]()
    convert_custom_config = imports["get_spconv_convert_custom_config"]()

    logger.info("Converting sparse encoder to INT8...")
    converted = imports["convert_fx"](
        prepared_encoder,
        convert_custom_config=convert_custom_config,
        backend_config=backend_config,
    )

    logger.info("Applying transform_qdq...")
    converted = imports["transform_qdq"](converted)

    logger.info("Applying remove_conv_add_dq...")
    converted = imports["remove_conv_add_dq"](converted)

    logger.info("Sparse encoder INT8 conversion complete")
    return converted
