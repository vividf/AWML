# Copyright (c) OpenMMLab. All rights reserved.
"""Build the BEVFusion :class:`QuantizationPlan` from a deploy ``quantization`` config.

This is the single place that maps BEVFusion's quantization config onto schemes. The deploy loader
and the PTQ producer both call it, so they compose an *identical* set of schemes and therefore an
identical quantized module tree — the PTQ ``state_dict`` and the deploy ``load_state_dict`` line up
by construction, not by a "keep these two code paths in sync" comment.

Composition (mirrors the historical inline behavior exactly):

* Dense tower (``pts_backbone`` / ``pts_neck`` / ``bbox_head``): a :class:`DenseQDQScheme`.
  BN is fused across **all three** dense submodules when ``fuse_bn`` (matching the old
  ``_fuse_dense_bn``); only the enabled ones are quantized. If no dense conv is enabled but
  ``fuse_bn`` is set, a fuse-only dense scheme is still added.
* Sparse tower (``pts_middle_encoder``): a :class:`SpconvInt8Scheme`, added only when
  ``include_sparse`` (i.e. the PTQ-checkpoint load path, where the old code fused/quantized the
  sparse encoder before ``load_state_dict``). ``int8`` follows ``spconv_int8``; when INT8 is off
  but ``fuse_bn`` is on, the sparse scheme fuses BN only.
"""

from __future__ import annotations

from typing import Any, Mapping

from deployment.quantization.schemes.base import QuantizationPlan

from .schemes import SpconvInt8Scheme

DENSE_SUBMODULES = ("pts_backbone", "pts_neck", "bbox_head")


def build_bevfusion_plan(
    quant_cfg: Mapping[str, Any],
    *,
    include_sparse: bool,
) -> QuantizationPlan:
    """Compose the dense + (optional) sparse schemes for BEVFusion.

    Args:
        quant_cfg: The deploy ``quantization`` dict (``spconv_int8_fp16_layers`` already hoisted in).
        include_sparse: Add the sparse-encoder scheme (True on the PTQ-checkpoint load path).

    Returns:
        A :class:`QuantizationPlan` (may be empty if nothing is enabled).
    """
    # Import here so ``pytorch_quantization`` is only required when a plan is actually built.
    from deployment.quantization.schemes import DenseQDQScheme

    fuse_bn = bool(quant_cfg.get("fuse_bn", True))
    quant_backbone = bool(quant_cfg.get("quant_backbone", True))
    quant_neck = bool(quant_cfg.get("quant_neck", True))
    quant_head = bool(quant_cfg.get("quant_head", True))
    quant_add = bool(quant_cfg.get("quant_add", False))
    sensitive_layers = list(quant_cfg.get("sensitive_layers", []) or [])
    spconv_int8 = bool(quant_cfg.get("spconv_int8", False))
    fp16_layers = list(quant_cfg.get("spconv_int8_fp16_layers", []) or [])

    plan = QuantizationPlan()

    enabled = [name for name, flag in zip(DENSE_SUBMODULES, (quant_backbone, quant_neck, quant_head)) if flag]
    dense_conv_enabled = bool(enabled)

    # Dense scheme: quantize enabled submodules; fuse BN across all dense submodules (matching the
    # old ``_fuse_dense_bn``, which fused all three regardless of which were quantized).
    if dense_conv_enabled:
        plan.add(
            DenseQDQScheme(
                quant_targets=enabled,
                fuse_targets=DENSE_SUBMODULES,
                quant_add=quant_add,
                sensitive_layers=sensitive_layers,
                fuse_bn=fuse_bn,
            )
        )
    elif fuse_bn:
        # No dense conv quantized, but the old path still fused dense BN under ``fuse_bn``.
        plan.add(
            DenseQDQScheme(
                quant_targets=(),
                fuse_targets=DENSE_SUBMODULES,
                quant_add=False,
                sensitive_layers=(),
                fuse_bn=True,
            )
        )

    # Sparse scheme (PTQ-checkpoint load path only). INT8 attaches quantizers; otherwise BN-fuse
    # only when ``fuse_bn`` (matching the old ``if fuse_bn: _fuse_spconv_bn`` in the PTQ branch).
    if include_sparse and (spconv_int8 or fuse_bn):
        plan.add(
            SpconvInt8Scheme(
                int8=spconv_int8,
                fuse_bn=fuse_bn,
                fp16_layers=fp16_layers,
                register_load_buffers=True,
            )
        )

    return plan
