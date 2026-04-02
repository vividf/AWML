"""Optional forward hooks on pts_middle_encoder sparse conv layers (compare to TRT ImplicitGemmInt8).

Enable::

    export BEVFUSION_SPARSE_ENCODER_HOOK_DEBUG=1
    # optional: max full sparse-encoder forwards (default 2)
    export BEVFUSION_SPARSE_ENCODER_HOOK_MAX_PASSES=2

Each hook fires after a ``SubMConv3d`` / ``SparseConv3d`` forward; stats are on
``SparseConvTensor.features`` (dense ``[N, C]``), same tensor layout as the TRT plugin's
FP16 output buffer.

**Semantic mismatch (important for numeric diff):** the hook sees **conv output only**
(BN/ReLU run on the next modules in the same ``SparseSequential``). The TRT
``ImplicitGemmInt8`` node implements **fused conv + INT8 epilogue** (``channel_scale`` /
``bias_scaled`` absorb BN etc.), and ``BEVFUSION_INT8_GEMM_DEBUG`` dumps that **final FP16**
activations. Large ratios vs PyTorch (e.g. ~40× on layer 0) can reflect **BN scaling**,
not only a wrong engine or scales. For apples-to-apples, compare against the **same**
point in the PT graph (e.g. after the block's BN) or run fake-quant forward aligned
with export.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
from torch import nn

_LOG_PREFIX = "[BEVFUSION][PyTorch][sparse-conv-hook]"


def _env_truthy(key: str) -> bool:
    v = os.environ.get(key, "").strip().lower()
    return v in ("1", "true", "yes", "y", "t")


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)).strip())
    except ValueError:
        return default


def _is_sparse_spconv_conv_module(module: nn.Module) -> bool:
    name = module.__class__.__name__
    if name == "SubMConv3d":
        return True
    if name == "SparseConv3d":
        return True
    return False


def _features_stats_line(feat: torch.Tensor, layer_display: str, pass_idx: int, seq: int) -> str:
    t = feat.detach().float()
    ne = t.numel()
    if ne == 0:
        return (
            f"{_LOG_PREFIX} pass={pass_idx} seq={seq} layer={layer_display} "
            f"out_features shape={tuple(feat.shape)} (empty)"
        )
    nz = int(t.count_nonzero().item())
    return (
        f"{_LOG_PREFIX} pass={pass_idx} seq={seq} layer={layer_display} "
        f"out_features shape={tuple(feat.shape)} dtype={feat.dtype} "
        f"min={t.min().item():.6f} max={t.max().item():.6f} "
        f"mean={t.mean().item():.6f} abs_mean={t.abs().mean().item():.6f} "
        f"nonzero={nz}/{ne}"
    )


def try_register_sparse_encoder_sparse_conv_hooks(
    model: nn.Module,
    *,
    encoder_attr: str = "pts_middle_encoder",
) -> None:
    """Register hooks if ``BEVFUSION_SPARSE_ENCODER_HOOK_DEBUG`` is set. No-op otherwise."""
    if not _env_truthy("BEVFUSION_SPARSE_ENCODER_HOOK_DEBUG"):
        return

    encoder = getattr(model, encoder_attr, None)
    if encoder is None or not isinstance(encoder, nn.Module):
        return

    if getattr(encoder, "_bevfusion_sparse_conv_hooks_registered", False):
        return

    max_passes = max(1, _env_int("BEVFUSION_SPARSE_ENCODER_HOOK_MAX_PASSES", 2))

    state: Dict[str, Any] = {"pass_idx": 0, "layer_seq": 0}

    def _pre_hook(_mod: nn.Module, _args: Any) -> None:
        state["pass_idx"] += 1
        state["layer_seq"] = 0

    def _make_conv_hook(full_name: str):
        layer_display = f"pts_middle_encoder.{full_name}"

        def _hook(_mod: nn.Module, _inp: Any, out: Any) -> None:
            if state["pass_idx"] > max_passes:
                return
            if not hasattr(out, "features"):
                return
            seq = state["layer_seq"]
            state["layer_seq"] = seq + 1
            feat = out.features
            if not isinstance(feat, torch.Tensor):
                return
            print(_features_stats_line(feat, layer_display, state["pass_idx"], seq))

        return _hook

    handles: List[Any] = []

    # Reset layer index at the start of each encoder forward.
    handles.append(encoder.register_forward_pre_hook(_pre_hook))

    for full_name, mod in encoder.named_modules():
        if not _is_sparse_spconv_conv_module(mod):
            continue
        handles.append(mod.register_forward_hook(_make_conv_hook(full_name)))

    encoder._bevfusion_sparse_conv_hooks_registered = True  # type: ignore[attr-defined]
    encoder._bevfusion_sparse_conv_hook_handles = handles  # type: ignore[attr-defined]

    print(
        f"{_LOG_PREFIX} registered {len(handles) - 1} sparse conv hooks on `{encoder_attr}` "
        f"(max_passes={max_passes}); compare seq order to TRT BEVFUSION_INT8_GEMM_DEBUG"
    )
