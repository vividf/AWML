# Copyright (c) OpenMMLab. All rights reserved.
"""Build the BEVFusion :class:`QuantizationPlan` from a :class:`QuantizationConfig`.

This is the single place that maps BEVFusion's quantization config onto schemes. The deploy loader
and the PTQ producer both call it, so they compose an *identical* set of schemes and therefore an
identical quantized module tree — the PTQ ``state_dict`` and the deploy ``load_state_dict`` line up
by construction, not by a "keep these two code paths in sync" comment.

Composition:

* Dense tower (``pts_backbone`` / ``pts_neck`` / ``bbox_head``): a :class:`DenseQDQScheme`. BN is fused
  across all three dense submodules when ``fuse_bn``; every Conv2d is quantized except the ``keep_fp16``
  subtrees. Residual-add is attached unless ``"add"`` is in ``disable_recipes`` (BEVFusion ships that).
* Sparse tower (``pts_middle_encoder``): a :class:`SpconvBnFuseScheme` (FP16 deploy — SparseConv+BN fold
  only), added unconditionally when ``fuse_bn``. The fold is an inference identity, so building it into
  the plan (rather than gating it on an ``include_sparse`` argument that the PTQ producer and deploy
  loader set differently) makes both sides pass identical arguments and the identical-tree invariant
  literally true (spec.md §3.8(3), R2).
"""

from __future__ import annotations

from deployment.config.schema import QuantizationConfig
from deployment.quantization.schemes.base import QuantizationPlan

from .schemes import SpconvBnFuseScheme

DENSE_SUBMODULES = ("pts_backbone", "pts_neck", "bbox_head")


def build_bevfusion_plan(config: QuantizationConfig) -> QuantizationPlan:
    """Compose the dense + sparse schemes for BEVFusion.

    Args:
        config: Parsed deploy ``quantization`` block.

    Returns:
        A :class:`QuantizationPlan`.
    """
    # Import here so ``pytorch_quantization`` is only required when a plan is actually built.
    from deployment.quantization.schemes import DenseQDQScheme

    plan = QuantizationPlan()

    # Dense scheme: fuse BN across all dense submodules (so state_dict keys match on both sides) and
    # quantize every Conv2d except the keep_fp16 subtrees.
    plan.add(
        DenseQDQScheme(
            DENSE_SUBMODULES,
            keep_fp16=config.keep_fp16,
            disable_recipes=config.disable_recipes,
            fuse_bn=config.fuse_bn,
        )
    )

    # Sparse scheme: FP16 deploy, SparseConv+BN fold only. Always part of the plan when fuse_bn, so PTQ
    # and deploy build the identical tree by construction (no divergent include_sparse argument).
    if config.fuse_bn:
        plan.add(SpconvBnFuseScheme(fuse_bn=config.fuse_bn))

    return plan
