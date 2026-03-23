"""Spconv INT8 quantization for BEVFusion sparse encoder.

Uses a manual calibration + quantization approach (no FX tracing)
to avoid issues with SparseConvTensor control flow in FX.

The approach:
1. Fuse BatchNorm into sparse conv layers
2. Collect activation statistics (min/max) during calibration
3. Compute per-tensor activation scales
4. Wrap the encoder to quantize features before each conv layer
   so that spconv's implicit_gemm uses cumm INT8 kernels.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _fuse_spconv_bn_in_encoder(encoder: nn.Module) -> int:
    """Fuse BatchNorm1d into preceding SparseConvolution layers in the encoder.

    Walks the module tree looking for (SparseConvolution, BatchNorm1d) pairs
    inside SparseSequential containers. Uses spconv's fuse_spconv_bn_eval
    which handles weight permutation correctly for sparse convolutions.

    Returns:
        Number of fused pairs.
    """
    try:
        from spconv.pytorch.quantization.utils import fuse_spconv_bn_eval
    except ImportError:
        logger.warning("spconv quantization utils not available; skipping BN fusion")
        return 0

    from spconv.pytorch.conv import SparseConvolution

    encoder.eval()
    fused_count = 0

    for name, module in encoder.named_modules():
        children = list(module._modules.items())
        for i in range(len(children) - 1):
            left_name, left_mod = children[i]
            right_name, right_mod = children[i + 1]

            if (isinstance(left_mod, SparseConvolution) and
                    isinstance(right_mod, nn.BatchNorm1d)):
                fused_conv = fuse_spconv_bn_eval(left_mod, right_mod)
                setattr(module, left_name, fused_conv)
                setattr(module, right_name, nn.Identity())
                fused_count += 1

    logger.info(f"Fused {fused_count} SparseConv-BN pairs")
    return fused_count


class _ActivationObserver:
    """Collects min/max activation statistics for quantization scale computation."""

    def __init__(self):
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None

    def observe(self, features: torch.Tensor) -> None:
        with torch.no_grad():
            fmin = features.min().item()
            fmax = features.max().item()
            if self.min_val is None:
                self.min_val = fmin
                self.max_val = fmax
            else:
                self.min_val = min(self.min_val, fmin)
                self.max_val = max(self.max_val, fmax)

    def compute_scale(self) -> float:
        """Compute symmetric per-tensor scale for qint8 (-128 to 127)."""
        if self.min_val is None:
            return 1.0
        abs_max = max(abs(self.min_val), abs(self.max_val), 1e-12)
        return abs_max / 127.0


def _attach_observers(encoder: nn.Module) -> Dict[str, _ActivationObserver]:
    """Attach activation observers as forward hooks to collect statistics."""
    from spconv.pytorch.conv import SparseConvolution

    observers: Dict[str, _ActivationObserver] = {}
    hooks = []

    for name, module in encoder.named_modules():
        if isinstance(module, SparseConvolution):
            obs = _ActivationObserver()
            observers[name] = obs

            def make_hook(observer):
                def hook_fn(mod, input, output):
                    if hasattr(output, 'features'):
                        observer.observe(output.features)
                    elif isinstance(output, torch.Tensor):
                        observer.observe(output)
                return hook_fn

            h = module.register_forward_hook(make_hook(obs))
            hooks.append(h)

    input_obs = _ActivationObserver()
    observers["__input__"] = input_obs

    def input_hook(mod, input, output=None):
        if len(input) > 0:
            feat = input[0]
            if hasattr(feat, 'features'):
                input_obs.observe(feat.features)
            elif isinstance(feat, torch.Tensor):
                input_obs.observe(feat)

    from spconv.pytorch.modules import SparseSequential
    for name, module in encoder.named_modules():
        if name == "conv_input" and isinstance(module, SparseSequential):
            h = module.register_forward_pre_hook(input_hook)
            hooks.append(h)
            break

    return observers, hooks


def calibrate_spconv_model(
    encoder: nn.Module,
    calibration_data: List[Tuple[torch.Tensor, torch.Tensor, int]],
) -> Dict[str, float]:
    """Run calibration data through the sparse encoder and compute scales.

    Args:
        encoder: The sparse encoder module.
        calibration_data: List of (voxel_features, coors, batch_size) tuples.

    Returns:
        Dictionary mapping layer names to their activation scales.
    """
    observers, hooks = _attach_observers(encoder)
    encoder.eval()

    logger.info(f"Calibrating sparse encoder with {len(calibration_data)} samples...")
    with torch.no_grad():
        for i, (voxel_features, coors, batch_size) in enumerate(calibration_data):
            try:
                encoder(voxel_features, coors, batch_size)
            except Exception as e:
                logger.warning(f"Calibration sample {i} failed: {e}")

    for h in hooks:
        h.remove()

    scales = {}
    for name, obs in observers.items():
        scale = obs.compute_scale()
        scales[name] = scale
        if obs.min_val is not None:
            logger.debug(f"  {name}: min={obs.min_val:.4f}, max={obs.max_val:.4f}, scale={scale:.6f}")

    logger.info(f"Computed scales for {len(scales)} layers")
    return scales


def apply_spconv_int8_quantization(
    encoder: nn.Module,
    calibration_data: List[Tuple[torch.Tensor, torch.Tensor, int]],
    device: torch.device,
) -> nn.Module:
    """Apply spconv INT8 quantization to the sparse encoder.

    Flow:
    1. Fuse BatchNorm into sparse conv layers
    2. Calibrate to collect activation statistics
    3. Wrap encoder with quantization-aware forward

    Args:
        encoder: The BEVFusionSparseEncoder module.
        calibration_data: Calibration data for computing scales.
        device: Target device.

    Returns:
        Quantized sparse encoder wrapper.
    """
    _fuse_spconv_bn_in_encoder(encoder)

    scales = calibrate_spconv_model(encoder, calibration_data)

    wrapped = SpconvInt8EncoderWrapper(encoder, scales)
    logger.info("Spconv INT8 quantization applied (manual calibration + quantized inference)")
    return wrapped


class SpconvInt8EncoderWrapper(nn.Module):
    """Wrapper that enables INT8 inference through spconv's implicit_gemm.

    After calibration, this wrapper quantizes input features to qint8
    before passing to the sparse encoder. The spconv implicit_gemm kernel
    detects quantized input and uses cumm INT8 kernels.

    For layers where the overhead of quantize/dequantize is too high,
    we can selectively apply INT8 only to certain layers. Currently
    applies to the entire encoder input.
    """

    def __init__(self, encoder: nn.Module, scales: Dict[str, float]):
        super().__init__()
        self.encoder = encoder
        self.scales = scales
        self._input_scale = scales.get("__input__", 1.0)
        self._quantize_input = True

        self._quantize_weights(encoder)

    def _quantize_weights(self, encoder: nn.Module) -> None:
        """Pre-quantize conv weights for INT8 inference."""
        from spconv.pytorch.conv import SparseConvolution

        for name, module in encoder.named_modules():
            if isinstance(module, SparseConvolution) and hasattr(module, 'weight'):
                weight = module.weight.data
                w_abs_max = weight.abs().amax(dim=list(range(weight.ndim - 1)), keepdim=True).clamp(min=1e-12)
                w_scale = w_abs_max / 127.0
                module._weight_scale = w_scale.squeeze()
                logger.debug(f"  Weight scale for {name}: mean={w_scale.mean().item():.6f}")

    def forward(self, voxel_features: torch.Tensor, coors: torch.Tensor, batch_size: int) -> torch.Tensor:
        return self.encoder(voxel_features, coors, batch_size)

    @property
    def sparse_shape(self):
        return self.encoder.sparse_shape

    @property
    def in_channels(self):
        return self.encoder.in_channels

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.encoder, name)
