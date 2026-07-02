"""Spconv INT8 quantization for BEVFusion sparse encoder (NVIDIA path).

Uses ``pytorch_quantization.TensorQuantizer`` with histogram collection and
MSE-based amax selection. Each ``SparseConvolution`` gets ``_input_quantizer``
and ``_weight_quantizer`` submodules.
"""

from __future__ import annotations

import logging
import math
import types
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _fuse_spconv_bn_in_encoder(sparse_encoder: nn.Module) -> int:
    """Fuse BatchNorm into sparse convolutions inside the given sparse encoder.

    Used by PTQ and deployment ``model_loader`` so state_dict keys match.
    Returns the number of fused Conv-BN pairs.
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


# ---------------------------------------------------------------------------
# NVIDIA pytorch_quantization approach (adapted from CUDA-BEVFusion)
# ---------------------------------------------------------------------------


def _get_sparse_conv_types() -> tuple:
    """Return a tuple of SparseConvolution classes to quantize."""
    from spconv.pytorch.conv import SparseConvolution as SpconvSparseConvolution

    conv_types: list = [SpconvSparseConvolution]
    try:
        from projects.SparseConvolution.sparse_conv import SparseConvolution as CustomSparseConvolution

        conv_types.append(CustomSparseConvolution)
    except ImportError:
        pass
    return tuple(conv_types)


def _nvidia_quantized_forward(self, input):
    """Quantized forward that wraps input.features and weight with TensorQuantizer."""
    if input is not None and hasattr(input, "features"):
        input = input.replace_feature(self._input_quantizer(input.features))
    if self.weight is None:
        return self._original_forward(input)
    quant_weight = self._weight_quantizer(self.weight)
    # PTQ / inference runs under ``torch.no_grad()``: keep **full-precision** values in
    # ``self.weight.data`` across forwards.  Old behavior replaced ``self.weight`` with
    # ``Parameter(quant_weight)`` every call → checkpoint / ONNX carried **fake-quant-baked**
    # weights while ``_weight_quantizer._amax`` was calibrated for **unbaked** distributions;
    # TRT ImplicitGemmInt8 then re-quantized baked weights with those scales → systematic
    # drift and huge ``lidar_bev`` / NaNs in the dense head.
    if not torch.is_grad_enabled():
        orig = self.weight.data.clone()
        try:
            self.weight.data.copy_(quant_weight)
            return self._original_forward(input)
        finally:
            self.weight.data.copy_(orig)
    self.weight = Parameter(quant_weight)
    return self._original_forward(input)


def apply_nvidia_spconv_int8(
    sparse_encoder: nn.Module,
    exclude_patterns: Optional[List[str]] = None,
) -> nn.Module:
    """Add NVIDIA ``TensorQuantizer`` to every ``SparseConvolution`` in *sparse_encoder*.

    Follows the CUDA-BEVFusion pattern: histogram collection for activations,
    per-**output**-channel weight scales for 5-D sparse weights.

    Sparse conv weight layout is ``[C_out, k1, k2, k3, C_in]``.  Use ``axis=(0)``
    so ``_weight_quantizer._amax`` is one value per **output** channel (length
    ``C_out``).  ``axis=(4)`` would calibrate along **input** channels only
    (length ``C_in``), which breaks sparse INT8 / ``ImplicitGemmInt8`` which expects
    ``channel_scale`` shaped ``[C_out]``.

    ``conv_out`` sparse convs are quantized like the rest of the tower (no special-case skip).

    ``exclude_patterns`` (case-insensitive substrings matched on each module's
    ``named_modules()`` name, e.g. ``"conv_input.0"`` or
    ``"encoder_layer1.0.conv1"``) skip installing TensorQuantizer on those
    sparse convs entirely — no ``_input_quantizer`` / ``_weight_quantizer``
    submodule is attached and the module's ``forward`` is left untouched.
    This is the **correct** way to keep selected layers FP16 end-to-end:
    PTQ calibration then observes the genuine FP activation distribution
    downstream (no fake-quant contamination from excluded layers), so the
    retained ``_amax`` values match the runtime behavior exactly. Any later
    ONNX / TRT ``ImplicitGemm`` skip logic is defense in depth, not the root
    control. Matching is done on PyTorch module names only — **never** on
    ONNX tensor names — to avoid the scope-path contamination bug where
    downstream tensors carry an upstream producer's name as a substring.

    Returns the same *sparse_encoder* (modified in-place).
    """
    from pytorch_quantization import calib
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization.tensor_quant import QuantDescriptor

    input_desc = QuantDescriptor(num_bits=8, calib_method="histogram")
    weight_desc = QuantDescriptor(num_bits=8, axis=(0))
    conv_types = _get_sparse_conv_types()

    norm_patterns: List[str] = [p.lower() for p in (exclude_patterns or []) if p]
    pattern_hits: Dict[str, int] = {p: 0 for p in norm_patterns}

    count = 0
    skipped_fp16 = 0
    for name, module in list(sparse_encoder.named_modules()):
        if not isinstance(module, conv_types):
            continue

        low_name = name.lower()
        matched_pat: Optional[str] = None
        for pat in norm_patterns:
            if pat in low_name:
                matched_pat = pat
                break
        if matched_pat is not None:
            pattern_hits[matched_pat] += 1
            skipped_fp16 += 1
            logger.info(
                "  [nvidia-quant] SKIP (kept FP16 per exclude_patterns='%s'): %s",
                matched_pat,
                name,
            )
            continue

        iq = quant_nn.TensorQuantizer(input_desc)
        wq = quant_nn.TensorQuantizer(weight_desc)
        if isinstance(getattr(iq, "_calibrator", None), calib.HistogramCalibrator):
            iq._calibrator._torch_hist = True
        if isinstance(getattr(wq, "_calibrator", None), calib.HistogramCalibrator):
            wq._calibrator._torch_hist = True

        module.register_module("_input_quantizer", iq)
        module.register_module("_weight_quantizer", wq)

        module._original_forward = module.forward
        module.forward = types.MethodType(_nvidia_quantized_forward, module)

        count += 1
        if count <= 3:
            logger.info(
                "  [nvidia-quant] %s: added TensorQuantizer (in=%d, out=%d)",
                name,
                module.in_channels,
                module.out_channels,
            )

    logger.info("Applied NVIDIA TensorQuantizer to %d sparse conv modules", count)
    if norm_patterns:
        logger.info(
            "  [nvidia-quant] FP16 exclusion summary: %d sparse convs kept FP16 (no TensorQuantizer)",
            skipped_fp16,
        )
        for pat, hits in pattern_hits.items():
            logger.info("  [nvidia-quant]   pattern='%s' -> %d module(s)", pat, hits)
        unmatched = [p for p, h in pattern_hits.items() if h == 0]
        if unmatched:
            logger.warning(
                "  [nvidia-quant] exclude_patterns with ZERO matches (typo?): %s",
                unmatched,
            )
    return sparse_encoder


def _collect_sparse_tail_absmax_before_conv_out(
    encoder: nn.Module,
    calibration_data: List[Tuple[torch.Tensor, torch.Tensor, int]],
) -> None:
    """Record max |features| at the FP32 ``conv_out`` boundary (CUDA-BEVFusion / libspconv semantics).

    CUDA-BEVFusion ONNX stores per-layer ``input_dynamic_range`` and sets ``conv_out`` to
    ``output_precision=fp16`` while the backbone stays int8: the **last int8 conv's output scale**
    must match the **real tensor** entering ``conv_out``. That tensor is not covered by a
    Sparse INT8 ONNX transform can read ``pts_middle_encoder._sparse_tail_absmax`` saved here.

    Must run **after** ``enable_quant`` so the max matches inference fake-quant behavior.
    """
    conv_out = getattr(encoder, "conv_out", None)
    if conv_out is None or not calibration_data:
        return

    device = next(encoder.parameters()).device
    mx = torch.zeros((), device=device, dtype=torch.float32)

    def _pre_hook(_mod: nn.Module, inp: Tuple[object, ...]) -> None:
        st = inp[0]
        feats = getattr(st, "features", None)
        if feats is None or feats.numel() == 0:
            return
        v = feats.detach().abs().max().to(device=mx.device, dtype=torch.float32)
        mx.copy_(torch.maximum(mx, v))

    hook = conv_out.register_forward_pre_hook(_pre_hook)
    try:
        with torch.no_grad():
            for voxel_features, coors, batch_size in calibration_data:
                encoder(voxel_features, coors, batch_size)
    finally:
        hook.remove()

    encoder.register_buffer("_sparse_tail_absmax", mx.detach().clone())
    val = float(mx.detach().cpu().item())
    if val <= 0.0 or not math.isfinite(val):
        raise RuntimeError(
            "[nvidia-calib] sparse INT8: _sparse_tail_absmax is invalid (0 or non-finite). "
            "Encoder forwards failed, or conv_out pre-hook never saw non-empty features."
        )
    print(
        f"[nvidia-calib] sparse INT8: max |sparse features| before conv_out = {val:.6f} "
        f"(saved as pts_middle_encoder._sparse_tail_absmax for sparse_int8_onnx_transform)"
    )


def _module_from_pts_stem(encoder: nn.Module, stem: str) -> nn.Module:
    """Resolve ``stem`` (checkpoint-relative under ``pts_middle_encoder``) to a submodule."""
    cur: nn.Module = encoder
    for tok in stem.split("."):
        if tok.isdigit():
            cur = cur[int(tok)]
        else:
            cur = getattr(cur, tok)
    return cur


def _nvidia_quantized_sparse_conv_stems(encoder: nn.Module) -> List[str]:
    """Module names (relative to *encoder*) of SparseConvs that have NVIDIA input quantizers."""
    conv_types = _get_sparse_conv_types()
    stems: List[str] = []
    for name, m in encoder.named_modules():
        if not name:
            continue
        if not isinstance(m, conv_types):
            continue
        if getattr(m, "_input_quantizer", None) is None:
            continue
        stems.append(name)
    return stems


def _collect_last_int8_conv_output_absmax(
    encoder: nn.Module,
    calibration_data: List[Tuple[torch.Tensor, torch.Tensor, int]],
) -> None:
    """Record max |features| at the **output of the last INT8 SparseConv** (pre tail BN/ReLU).

    ``_sparse_tail_absmax`` measures the tensor **entering** ``conv_out``, i.e. after the
    final encoder block's norm/residual/ReLU.  ``ImplicitGemmInt8``'s ``output_scale`` must match
    the **linear conv output** of the last quantized sparse conv (what the ONNX node emits before
    downstream FP ops).  Using the conv_out-boundary value there **over-``output_scale``s** and
    blows up ``lidar_bev`` in TensorRT while PyTorch fake-quant stays sane.

    Saved as ``pts_middle_encoder._last_int8_conv_output_absmax`` for
    ``sparse_int8_onnx_transform`` (preferred terminal scale).
    """
    stems = _nvidia_quantized_sparse_conv_stems(encoder)
    if not stems or not calibration_data:
        return
    from deployment.quantization.sparse.naming import (
        topologically_sorted_sparse_stems as _topologically_sorted_sparse_stems,
    )

    topo = _topologically_sorted_sparse_stems(stems)
    last_stem = topo[-1]
    try:
        mod = _module_from_pts_stem(encoder, last_stem)
    except Exception as e:
        raise RuntimeError(f"sparse INT8: could not resolve stem {last_stem!r} for last-int8 absmax") from e

    device = next(encoder.parameters()).device
    mx = torch.zeros((), device=device, dtype=torch.float32)

    def _hook(_m: nn.Module, _inp: Tuple[object, ...], out: object) -> None:
        feats = getattr(out, "features", None)
        if feats is None or feats.numel() == 0:
            return
        v = feats.detach().abs().max().to(device=mx.device, dtype=torch.float32)
        mx.copy_(torch.maximum(mx, v))

    hook_h = mod.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            for voxel_features, coors, batch_size in calibration_data:
                encoder(voxel_features, coors, batch_size)
    finally:
        hook_h.remove()

    encoder.register_buffer("_last_int8_conv_output_absmax", mx.detach().clone())
    val = float(mx.detach().cpu().item())
    if val <= 0.0 or not math.isfinite(val):
        raise RuntimeError(
            "[nvidia-calib] sparse INT8: _last_int8_conv_output_absmax is invalid (0 or non-finite). "
            "Encoder forwards failed, or last INT8 sparse conv never produced non-empty features."
        )
    print(
        f"[nvidia-calib] sparse INT8: max |features| after last INT8 sparse conv ({last_stem}) = "
        f"{val:.6f} (saved as pts_middle_encoder._last_int8_conv_output_absmax; "
        "preferred for ONNX terminal output_scale)"
    )


def calibrate_spconv_nvidia(
    encoder: nn.Module,
    calibration_data: List[Tuple[torch.Tensor, torch.Tensor, int]],
) -> None:
    """Calibrate sparse encoder with histogram + MSE (CUDA-BEVFusion approach).

    1. Enable calibration mode on all TensorQuantizers (collect histograms, no fake-quant).
    2. Run calibration data through the encoder.
    3. compute_amax(method=mse) for all quantizers.
    4. Re-enable fake-quantization.
    5. Collect ``_sparse_tail_absmax`` at the conv_out boundary for sparse INT8 TRT export.
    """
    from pytorch_quantization import calib
    from pytorch_quantization import nn as quant_nn

    encoder.eval()
    device = next(encoder.parameters()).device
    n_samples = len(calibration_data)

    n_quantizers = 0
    for name, mod in encoder.named_modules():
        if isinstance(mod, quant_nn.TensorQuantizer):
            n_quantizers += 1
            if mod._calibrator is not None:
                mod.disable_quant()
                mod.enable_calib()
            else:
                mod.disable()
    print(f"[nvidia-calib] {n_quantizers} TensorQuantizers in calibration mode")
    print(f"[nvidia-calib] Collecting histograms over {n_samples} samples...")

    with torch.no_grad():
        pbar = tqdm(calibration_data, total=n_samples, desc="NVIDIA calib", leave=True)
        for i, (voxel_features, coors, batch_size) in enumerate(pbar):
            try:
                encoder(voxel_features, coors, batch_size)
            except Exception as e:
                raise RuntimeError(
                    f"[nvidia-calib] sample {i + 1}/{n_samples} failed during histogram collection"
                ) from e

    print("[nvidia-calib] Computing amax (method=mse) from histograms...")
    for name, mod in encoder.named_modules():
        if isinstance(mod, quant_nn.TensorQuantizer):
            if mod._calibrator is not None:
                if isinstance(mod._calibrator, calib.MaxCalibrator):
                    mod.load_calib_amax(strict=False)
                else:
                    mod.load_calib_amax(strict=False, method="mse")
                if mod._amax is not None:
                    mod._amax = mod._amax.to(device)

    for name, mod in encoder.named_modules():
        if isinstance(mod, quant_nn.TensorQuantizer):
            if mod._calibrator is not None:
                mod.enable_quant()
                mod.disable_calib()
            else:
                mod.enable()

    _collect_sparse_tail_absmax_before_conv_out(encoder, calibration_data)
    _collect_last_int8_conv_output_absmax(encoder, calibration_data)
    _report_nvidia_quantizer_stats(encoder)


def _report_nvidia_quantizer_stats(encoder: nn.Module) -> None:
    """Print amax summary for all NVIDIA TensorQuantizers after calibration."""
    from pytorch_quantization import nn as quant_nn

    count = 0
    for name, mod in encoder.named_modules():
        if isinstance(mod, quant_nn.TensorQuantizer):
            amax = getattr(mod, "_amax", None)
            if amax is not None:
                if amax.numel() <= 3:
                    vals = amax.flatten().tolist()
                    print(f"  [nvidia-amax] {name}: amax={vals}")
                else:
                    print(
                        f"  [nvidia-amax] {name}: amax shape={tuple(amax.shape)}, "
                        f"min={amax.min():.4f}, max={amax.max():.4f}, mean={amax.mean():.4f}"
                    )
                count += 1
    print(f"[nvidia-calib] {count} quantizers with calibrated amax")
