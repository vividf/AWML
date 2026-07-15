# Copyright (c) OpenMMLab. All rights reserved.
"""Sparse-convolution (spconv) primitives — model-agnostic.

The sparse encoder deploys in FP16; the only shared primitive is SparseConv+BN folding, which the
PTQ producer, the deploy loader, and the FP16 sparse export path all reuse so their module trees
cannot drift.
"""

from .fusion import fuse_spconv_bn_in_encoder

__all__ = [
    "fuse_spconv_bn_in_encoder",
]
