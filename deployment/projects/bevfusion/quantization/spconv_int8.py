"""Spconv INT8 quantization for BEVFusion sparse encoder.

Uses spconv's torch.ao.quantization FX graph mode to quantize
the sparse encoder with real INT8 (cumm kernels).

Flow: prepare_fx → calibrate → convert_fx → transform_qdq → remove_conv_add_dq
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Spconv implicit_gemm / get_indice_pairs allocate ~ O(kernel_volume * N^2) int32 for N voxels.
# **PyTorch FX prepare_fx + calibrate on GPU** cannot use full-scene N (often 50k–120k): that is
# hundreds of GiB. Lidar AI Solution avoids this by **libspconv / C++ runtime**, not this path.
# Default cap keeps PTQ runnable; raise via deploy ``spconv_calib_max_voxels`` or env
# ``SPCONV_CALIB_MAX_VOXELS`` if you have headroom; lower if you still OOM.
_DEFAULT_SPCONV_CALIB_MAX_VOXELS = 4096


def _default_spconv_calib_max_voxels() -> int:
    """Return max voxels per calibration sample when config/CLI do not set one."""
    raw = os.environ.get("SPCONV_CALIB_MAX_VOXELS", "").strip()
    if raw:
        try:
            v = int(raw)
            if v <= 0:
                logger.warning(
                    "SPCONV_CALIB_MAX_VOXELS=%s <= 0: full-frame FX calibration usually OOMs; using default %d",
                    raw,
                    _DEFAULT_SPCONV_CALIB_MAX_VOXELS,
                )
                return _DEFAULT_SPCONV_CALIB_MAX_VOXELS
            return max(1, v)
        except ValueError:
            logger.warning(
                "Invalid SPCONV_CALIB_MAX_VOXELS=%r; using default %d", raw, _DEFAULT_SPCONV_CALIB_MAX_VOXELS
            )
    return _DEFAULT_SPCONV_CALIB_MAX_VOXELS


def cap_voxels_for_spconv_calibration(
    voxel_features: torch.Tensor,
    coors: torch.Tensor,
    max_voxels: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Subsample voxels so sparse calibration stays within GPU memory (see module doc)."""
    if max_voxels <= 0:
        return voxel_features, coors
    n = int(voxel_features.shape[0])
    if n <= max_voxels:
        return voxel_features, coors
    idx = torch.randperm(n, device=voxel_features.device)[:max_voxels]
    return voxel_features.index_select(0, idx), coors.index_select(0, idx)


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


def _sparse_basic_block_to_fx(block: nn.Module) -> nn.Module:
    """Build SparseBasicBlockFX with same spconv indice_key/stride/downsample; copy conv/norm weights."""
    from mmdet3d.models.layers.sparse_block import SparseBasicBlock

    from projects.BEVFusion.bevfusion.sparse_block_fx import SparseBasicBlockFX

    if not isinstance(block, SparseBasicBlock):
        raise TypeError(f"expected SparseBasicBlock, got {type(block)!r}")

    inplanes = block.conv1.in_channels
    planes = block.conv1.out_channels
    stride = block.conv1.stride
    if isinstance(stride, (tuple, list)):
        stride = tuple(int(s) for s in stride) if len(stride) > 1 else int(stride[0])
    else:
        stride = int(stride)
    downsample = block.downsample
    indice_key = getattr(block.conv1, "indice_key", None)
    device = next(block.parameters()).device

    fx = SparseBasicBlockFX(
        inplanes,
        planes,
        stride=stride,
        downsample=downsample,
        indice_key=indice_key,
        conv_cfg=None,
        norm_cfg=None,
    ).to(device)
    fx.load_state_dict(block.state_dict(), strict=False)
    return fx


def upgrade_pts_middle_encoder_basicblocks_to_fx(sparse_encoder: nn.Module) -> int:
    """Replace ``SparseBasicBlock`` with ``SparseBasicBlockFX`` under ``pts_middle_encoder``.

    PTQ spconv checkpoints are usually produced with ``block_type=basicblock_fx``; FX graphs name
    activations (e.g. ``relu_final_scale``). Rebuilding from a ``basicblock`` config yields
    different Q/DQ parameter names and many missing/unexpected keys. Call this **before**
    ``prepare_fx`` / ``convert_fx`` when loading such checkpoints.

    Returns:
        Number of blocks replaced.
    """
    from mmdet3d.models.layers.sparse_block import SparseBasicBlock

    replaced = 0

    def walk(m: nn.Module) -> None:
        nonlocal replaced
        for name, child in list(m._modules.items()):
            if child is None:
                continue
            if isinstance(child, SparseBasicBlock):
                m._modules[name] = _sparse_basic_block_to_fx(child)
                replaced += 1
            else:
                walk(child)

    walk(sparse_encoder)
    if replaced:
        logger.info(
            "Upgraded %d SparseBasicBlock -> SparseBasicBlockFX before spconv prepare_fx (PTQ key alignment)",
            replaced,
        )
    return replaced


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
    # Match BEVFusion 120m lidar grid (see default_lidar_intensity_120m.grid_size).
    sparse_shape = getattr(model, "sparse_shape", [1440, 1440, 41])

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


@contextlib.contextmanager
def _sparse_basic_block_skip_dim_assert_for_fx() -> Iterator[None]:
    """Patch mmdet3d ``SparseBasicBlock.forward`` only for the ``prepare_fx`` call.

    Upstream forward does ``assert x.features.dim() == 2``. Under symbolic trace that
    touches ``Proxy.__bool__`` and raises ``TraceError``. ``is_tracing()`` is not
    reliable when execution goes through spconv ``SparseSequential`` wrappers, so while
    this context is active we omit the assert entirely (original forward is restored
    right after ``prepare_fx``).

    Rest of forward matches OpenMMLab mmdet3d ``sparse_block.SparseBasicBlock`` (norm1/norm2).
    """
    try:
        import mmdet3d.models.layers.sparse_block as sb_mod
    except ImportError:
        yield
        return

    cls = sb_mod.SparseBasicBlock
    replace_feature = sb_mod.replace_feature
    orig_forward = cls.forward

    def forward_patched(self, x):
        identity = x.features
        out = self.conv1(x)
        out = replace_feature(out, self.norm1(out.features))
        out = replace_feature(out, self.relu(out.features))
        out = self.conv2(out)
        out = replace_feature(out, self.norm2(out.features))
        if self.downsample is not None:
            identity = self.downsample(x).features
        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))
        return out

    cls.forward = forward_patched
    try:
        yield
    finally:
        cls.forward = orig_forward


def _disable_spconv_fx_trace_mode() -> None:
    """Turn off spconv FX trace mode (both modules that cache the flag).

    Deployment sets ``SPCONV_FX_TRACE_MODE=1`` early for INT8 ONNX/spconv. That global relaxed mode can
    interact badly with ``pytorch_quantization`` while inserting TensorQuantizers (torch.fx Proxy +
    control-flow errors). Disable **only** for dense Q/DQ insertion; ``apply_spconv_int8_quantization``
    calls ``_enable_spconv_fx_trace_mode()`` again before ``prepare_fx``.
    """
    try:
        import spconv.constants as spconv_constants
        import spconv.pytorch.core as spconv_core

        spconv_constants.SPCONV_FX_TRACE_MODE = False
        spconv_core.SPCONV_FX_TRACE_MODE = False
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
    from deployment.projects.bevfusion.quantization.spconv_quantized_add_patch import (
        ensure_spconv_quantize_per_tensor_float_activations,
    )

    ensure_spconv_quantize_per_tensor_float_activations()

    imports = _get_spconv_quantization_imports()

    imports["prepare_spconv_torch_inference"](with_linear=False)

    qconfig_mapping = imports["get_default_spconv_qconfig_mapping"](is_qat=False)
    backend_config = imports["get_spconv_backend_config"]()
    prepare_custom_config = imports["get_spconv_prepare_custom_config"]()

    example_inputs = _create_example_inputs(sparse_encoder, device, in_channels=in_channels)

    sparse_encoder.eval()
    logger.info("Running prepare_fx on sparse encoder for INT8 quantization...")
    with _sparse_basic_block_skip_dim_assert_for_fx():
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
    *,
    max_voxels_per_sample: Optional[int] = None,
) -> None:
    """Run calibration data through the prepared sparse encoder.

    Args:
        prepared_encoder: Prepared (with observers) sparse encoder.
        calibration_data: List of (voxel_features, coors, batch_size) tuples.
        max_voxels_per_sample: Max voxels per sample; ``None`` uses env/default (4096).
            ``<= 0`` is treated as "use default" (full-frame FX calibrate is not supported on GPU).
    """
    prepared_encoder.eval()
    if max_voxels_per_sample is not None:
        cap = int(max_voxels_per_sample)
        if cap <= 0:
            cap = _default_spconv_calib_max_voxels()
    else:
        cap = _default_spconv_calib_max_voxels()
    cap_desc = "unlimited (full voxels)" if cap <= 0 else str(cap)
    logger.info(
        "Calibrating sparse encoder with %d samples (max_voxels_per_sample=%s; set positive "
        "SPCONV_CALIB_MAX_VOXELS or quantization.spconv_calib_max_voxels if OOM)",
        len(calibration_data),
        cap_desc,
    )

    with torch.no_grad():
        for i, (voxel_features, coors, batch_size) in enumerate(calibration_data):
            n0 = int(voxel_features.shape[0])
            voxel_features, coors = cap_voxels_for_spconv_calibration(voxel_features, coors, cap)
            if cap > 0 and n0 > cap:
                logger.info(
                    "  Sample %d: capped voxels %d -> %d for spconv calibration memory",
                    i + 1,
                    n0,
                    cap,
                )
            try:
                prepared_encoder(voxel_features, coors, batch_size)
                logger.debug(f"  Calibration sample {i + 1}/{len(calibration_data)}")
            except torch.cuda.OutOfMemoryError:
                logger.error(
                    "CUDA OOM during spconv sparse calibration (cap=%s). "
                    "Set a positive voxel cap (e.g. quantization.spconv_calib_max_voxels=4096 or "
                    "export SPCONV_CALIB_MAX_VOXELS=4096), reduce num_calibration_samples, or free GPU memory.",
                    cap_desc,
                )
                raise
            except Exception as e:
                logger.warning(f"  Calibration sample {i + 1} failed: {e}")

    logger.info("Sparse encoder calibration complete")


def convert_spconv_int8(
    prepared_encoder: nn.Module,
    *,
    attr_source: Optional[nn.Module] = None,
) -> nn.Module:
    """Convert a calibrated prepared model to quantized INT8.

    Args:
        prepared_encoder: Calibrated prepared sparse encoder.
        attr_source: Module to copy ``sparse_shape`` / ``encoder_channels`` / … onto the FX root
            after conversion (the pre-``prepare_fx`` ``BEVFusionSparseEncoder``). ``convert_fx`` often
            drops these attributes; without them, ONNX export cannot swap in an FP32 shadow encoder.

    Returns:
        Quantized sparse encoder using cumm INT8 kernels.
    """
    from deployment.projects.bevfusion.quantization.spconv_quantized_add_patch import (
        ensure_spconv_quantize_per_tensor_float_activations,
        ensure_spconv_quantized_add_sparse_support,
    )

    ensure_spconv_quantize_per_tensor_float_activations()
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

    try:
        from deployment.projects.bevfusion.export.sparse_encoder_float_shadow import (
            copy_sparse_encoder_public_attrs,
        )

        src = attr_source if attr_source is not None else prepared_encoder
        copy_sparse_encoder_public_attrs(src, converted)
    except Exception as e:
        logger.warning("Could not copy sparse encoder public attrs onto FX root: %s", e)

    logger.info("Sparse encoder INT8 conversion complete")
    return converted
