# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
# flash-attention
import math
import os

import torch
import torch.nn as nn
from einops import rearrange
from flash_attn.bert_padding import unpad_input
from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func
from torch.nn.functional import linear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_


def _fp32_attention_enabled() -> bool:
    """Whether to run attention in true fp32 instead of casting down to fp16.

    ``FlashMHA`` is constructed with ``dtype=torch.float16`` and this module
    calls ``q.half()`` unconditionally, so attention is computed in fp16 even
    when the surrounding model runs in fp32 - the configured precision does not
    reach it. Opt in with ``STREAMPETR_FP32_ATTENTION=1`` to take an exact fp32
    path instead, which is what the overfit probe uses to measure how much of
    the autoware-ml gap this accounts for. Off by default: turning it on changes
    training numerics and costs speed.
    """
    return os.environ.get("STREAMPETR_FP32_ATTENTION", "") == "1"


_ATTENTION_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _attention_dtype_override():
    """Explicit cross-attention compute dtype from ``STREAMPETR_ATTENTION_DTYPE``.

    Ablation hook for the overfit probe. Cross-attention precision is otherwise
    welded to the surrounding autocast dtype, so it cannot be varied on its own
    - which is exactly what is needed to ask whether the fp16-vs-bf16
    cross-attention difference against autoware-ml matters. Accepts
    ``fp16`` / ``bf16`` / ``fp32``; unset means "follow the caller".
    """
    return _ATTENTION_DTYPES.get(os.environ.get("STREAMPETR_ATTENTION_DTYPE", "").lower())


def _fp32_math_attention(q, kv, softmax_scale, dropout_p, causal):
    """Exact fp32 scaled-dot-product attention matching the flash path's shapes.

    Takes q as ``(B, T, H, D)`` and kv as ``(B, S, 2, H, D)`` and returns
    ``(B, T, H, D)``, the same layout ``flash_attn_varlen_kvpacked_func``
    produces, so callers need no changes.
    """
    key, value = kv[:, :, 0], kv[:, :, 1]
    output = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(),
        key.transpose(1, 2).float(),
        value.transpose(1, 2).float(),
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    )
    return output.transpose(1, 2)


def _deterministic_backward() -> bool:
    """Whether FlashAttention should use its deterministic backward pass.

    FlashAttention accumulates dq with atomics, so its backward is
    nondeterministic by default. It is a custom CUDA op, so
    ``torch.use_deterministic_algorithms`` neither overrides it nor warns about
    it - a run looks reproducible and silently is not. Gating on torch's own
    flag lets the overfit probe get bitwise reproducible traces while normal
    training keeps the faster nondeterministic path.
    """
    return torch.are_deterministic_algorithms_enabled()


def _in_projection_packed(q, k, v, w, b=None):
    w_q, w_k, w_v = w.chunk(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
    """

    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.fp16_enabled = True

    def forward(self, q, kv, causal=False, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, T, H, D)
            kv: The tensor containing the key, and value. (B, S, 2, H, D)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        # assert q.dtype in [torch.float16, torch.bfloat16] and kv.dtype in [torch.float16, torch.bfloat16]
        # assert q.is_cuda and kv.is_cuda
        assert q.shape[0] == kv.shape[0] and q.shape[-2] == kv.shape[-2] and q.shape[-1] == kv.shape[-1]

        input_dtype = q.dtype
        fp16 = input_dtype in [torch.float16, torch.bfloat16]
        # Run flash_attn in the caller's dtype instead of unconditionally
        # casting to fp16. Before this, `.half()` silently downcast bf16
        # inputs to fp16, so switching the training recipe to bf16 never
        # actually changed the attention precision (it invalidated the
        # 2026-07-29 bf16 ablation). fp32 inputs still take the legacy fp16
        # cast, since flash_attn only ships 16-bit kernels; use
        # STREAMPETR_ATTENTION_DTYPE / STREAMPETR_FP32_ATTENTION to override.
        override = _attention_dtype_override()
        attn_dtype = override if override is not None else (input_dtype if fp16 else torch.float16)
        batch_size = q.shape[0]
        seqlen_q, seqlen_k = q.shape[1], kv.shape[1]
        # flash_attn has no fp32 kernel, so fp32 needs the math path. The padded
        # branch would need the mask translated into an additive bias, so it
        # keeps the flash path regardless.
        want_fp32 = attn_dtype == torch.float32 or (not fp16 and _fp32_attention_enabled())
        if want_fp32 and key_padding_mask is None:
            output = _fp32_math_attention(
                q,
                kv,
                self.softmax_scale,
                self.dropout_p if self.training else 0.0,
                causal,
            )
            return output.to(input_dtype), None
        if key_padding_mask is None:
            q, kv = rearrange(q, "b s ... -> (b s) ..."), rearrange(kv, "b s ... -> (b s) ...")
            max_sq, max_sk = seqlen_q, seqlen_k
            cu_seqlens_q = torch.arange(
                0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device
            )
            cu_seqlens_k = torch.arange(
                0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=kv.device
            )
            output = flash_attn_varlen_kvpacked_func(
                q.to(attn_dtype),
                kv.to(attn_dtype),
                cu_seqlens_q,
                cu_seqlens_k,
                max_sq,
                max_sk,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
                deterministic=_deterministic_backward(),
            )
            output = rearrange(output, "(b s) ... -> b s ...", b=batch_size)
        else:
            nheads = kv.shape[-2]
            q = rearrange(q, "b s ... -> (b s) ...")
            max_sq = seqlen_q
            cu_seqlens_q = torch.arange(
                0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device
            )
            x = rearrange(kv, "b s two h d -> b s (two h d)")
            x_unpad, indices, cu_seqlens_k, max_sk = unpad_input(x, key_padding_mask)
            x_unpad = rearrange(x_unpad, "nnz (two h d) -> nnz two h d", two=2, h=nheads)
            output_unpad = flash_attn_varlen_kvpacked_func(
                q.to(attn_dtype),
                x_unpad.to(attn_dtype),
                cu_seqlens_q,
                cu_seqlens_k,
                max_sq,
                max_sk,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
                deterministic=_deterministic_backward(),
            )
            output = rearrange(output_unpad, "(b s) ... -> b s ...", b=batch_size)

        # Restore the caller's dtype. This replaces two `if not fp16:
        # output = output.float()` blocks, the second of which referenced
        # `output` before assignment and would have raised UnboundLocalError on
        # any fp32 call that supplied a key_padding_mask.
        return output.to(input_dtype), None


class FlashMHA(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        batch_first=True,
        attention_dropout=0.0,
        causal=False,
        device=None,
        dtype=None,
        **kwargs
    ) -> None:
        assert batch_first
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.bias = bias

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.inner_attn = FlashAttention(attention_dropout=attention_dropout, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, key_padding_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        # q, k, v = self.Wq(q), self.Wk(k), self.Wv(v)
        q, k, v = _in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)
        q = rearrange(q, "b s (h d) -> b s h d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.num_heads)
        kv = torch.stack([k, v], dim=2)

        context, attn_weights = self.inner_attn(q, kv, key_padding_mask=key_padding_mask, causal=self.causal)
        return self.out_proj(rearrange(context, "b s h d -> b s (h d)")), attn_weights
