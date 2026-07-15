# Copyright (c) OpenMMLab. All rights reserved.
"""Build the BEVFusion :class:`QuantizationPlan` from a :class:`QuantizationConfig`.

This is the single place that maps BEVFusion's quantization config onto schemes. The deploy loader
and the PTQ producer both call it, so they compose an *identical* set of schemes and therefore an
identical quantized module tree — the PTQ ``state_dict`` and the deploy ``load_state_dict`` line up
by construction, not by a "keep these two code paths in sync" comment.

Composition:

* Dense tower (``pts_backbone`` / ``pts_neck`` / ``bbox_head``): a :class:`DenseQDQScheme`.
  BN is fused across **all three** dense submodules when ``fuse_bn``; only the enabled ones are
  quantized. If no dense conv is enabled but ``fuse_bn`` is set, a fuse-only dense scheme is still
  added.
* Sparse tower (``pts_middle_encoder``): a :class:`SpconvInt8Scheme`, added only when
  ``include_sparse`` (the PTQ-checkpoint load path). ``int8`` follows ``spconv_int8``; when INT8 is
  off but ``fuse_bn`` is on, the sparse scheme fuses BN only.
"""

from __future__ import annotations

from deployment.config.schema import QuantizationConfig
from deployment.quantization.schemes.base import QuantizationPlan

from .schemes import SpconvInt8Scheme

DENSE_SUBMODULES = ("pts_backbone", "pts_neck", "bbox_head")


def build_bevfusion_plan(config: QuantizationConfig, *, include_sparse: bool) -> QuantizationPlan:
    """Compose the dense + (optional) sparse schemes for BEVFusion.

    Args:
        config: Parsed deploy ``quantization`` block.
        include_sparse: Add the sparse-encoder scheme (True on the PTQ-checkpoint load path).

    Returns:
        A :class:`QuantizationPlan` (may be empty if nothing is enabled).
    """
    # Import here so ``pytorch_quantization`` is only required when a plan is actually built.
    from deployment.quantization.schemes import DenseQDQScheme

    plan = QuantizationPlan()

    enabled = [
        name
        for name, flag in zip(DENSE_SUBMODULES, (config.quant_backbone, config.quant_neck, config.quant_head))
        if flag
    ]

    # Dense scheme: quantize enabled submodules; fuse BN across all dense submodules regardless of
    # which are quantized (so the module tree — and the state_dict keys — match on both sides).
    if enabled:
        plan.add(
            DenseQDQScheme(
                quant_targets=enabled,
                fuse_targets=DENSE_SUBMODULES,
                quant_add=config.quant_add,
                sensitive_layers=config.sensitive_layers,
                fuse_bn=config.fuse_bn,
            )
        )
    elif config.fuse_bn:
        # No dense conv quantized, but still fuse dense BN under ``fuse_bn``.
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
    # only when ``fuse_bn``.
    if include_sparse and (config.spconv_int8 or config.fuse_bn):
        plan.add(
            SpconvInt8Scheme(
                int8=config.spconv_int8,
                fuse_bn=config.fuse_bn,
                fp16_layers=config.spconv_int8_fp16_layers,
                register_load_buffers=True,
            )
        )

    return plan
