"""Patch spconv ``quantize_per_tensor`` for float voxel features (export / dense QAT paths).

Coerces activations to float32 before ``torch.quantize_per_tensor`` where needed.
Pip-installed spconv may not include this fix; the patch applies at runtime.

Optional ``quantized_add`` sparse-aware wrapper remains for any remaining quantized graphs.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_PATCHED = False
_QPT_PATCHED = False


def _activation_tensor_for_quant(ten: torch.Tensor) -> torch.Tensor:
    """Return a float tensor suitable for ``torch.quantize_per_tensor``.

    Do **not** branch on ``ndim`` / ``size(1)`` / ``dtype`` with Python ``in`` checks:
    during ``torch.jit.trace`` / ONNX export, dimensions can be traced tensors, which
    triggers ``TracerWarning`` and can make ``(tensor(N), tensor(C))`` comparisons
    behave incorrectly. A correct FX graph should only pass float activations here;
    any non-float input is coerced to float32 so export can proceed.
    """
    if ten.is_floating_point():
        return ten
    return ten.to(dtype=torch.float32)


def _quantize_per_tensor_impl(ten, scale, zero_point, dtype):
    """Body used by patched ``spconv.pytorch.quantization.core.quantize_per_tensor``."""
    from spconv.pytorch.core import SparseConvTensor

    if isinstance(ten, (list, tuple)):
        res = []
        for i, v in enumerate(ten):
            if isinstance(v, SparseConvTensor):
                f = _activation_tensor_for_quant(v.features)
                res.append(v.replace_feature(torch.quantize_per_tensor(f, scale[i], zero_point[i], dtype)))
            else:
                v = _activation_tensor_for_quant(v)
                res.append(torch.quantize_per_tensor(v, scale[i], zero_point[i], dtype))
        return res
    if isinstance(ten, SparseConvTensor):
        f = _activation_tensor_for_quant(ten.features)
        return ten.replace_feature(torch.quantize_per_tensor(f, scale, zero_point, dtype))
    ten = _activation_tensor_for_quant(ten)
    return torch.quantize_per_tensor(ten, scale, zero_point, dtype)


def ensure_spconv_quantize_per_tensor_float_activations() -> None:
    """Patch ``spconv.pytorch.quantization.core.quantize_per_tensor`` (idempotent)."""
    global _QPT_PATCHED
    if _QPT_PATCHED:
        return
    try:
        import spconv.pytorch.quantization.core as core
    except ImportError:
        logger.warning("spconv quantization core not importable; skip quantize_per_tensor patch")
        return

    def quantize_per_tensor_patched(ten, scale, zero_point, dtype):
        return _quantize_per_tensor_impl(ten, scale, zero_point, dtype)

    core.quantize_per_tensor = quantize_per_tensor_patched
    _QPT_PATCHED = True
    logger.debug("Patched spconv core.quantize_per_tensor for float voxel features")


def ensure_spconv_quantized_add_sparse_support() -> None:
    """Idempotently wrap ``spconv.pytorch.quantization.core.quantized_add``."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        import spconv.pytorch.quantization.core as core
        import spconv.pytorch.quantization.graph as qgraph
    except ImportError:
        logger.warning("spconv quantization not importable; skip quantized_add sparse patch")
        return

    dense_impl = core.quantized_add

    def quantized_add_sparse_aware(x, y, scale, zero_point):
        from spconv.pytorch.core import SparseConvTensor

        if isinstance(x, SparseConvTensor) and isinstance(y, SparseConvTensor):
            return x.replace_feature(dense_impl(x.features, y.features, scale, zero_point))
        if isinstance(x, SparseConvTensor):
            return x.replace_feature(dense_impl(x.features, y, scale, zero_point))
        if isinstance(y, SparseConvTensor):
            return y.replace_feature(dense_impl(x, y.features, scale, zero_point))
        return dense_impl(x, y, scale, zero_point)

    core.quantized_add = quantized_add_sparse_aware
    # Older spconv: graph.py did ``from ...core import quantized_add`` (stale binding).
    if hasattr(qgraph, "quantized_add"):
        qgraph.quantized_add = quantized_add_sparse_aware

    _PATCHED = True
    logger.debug("Patched spconv quantization quantized_add for SparseConvTensor operands")
