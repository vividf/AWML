# Copyright (c) OpenMMLab. All rights reserved.
"""Layer fusion utilities for quantization."""

from .bn_fusion import (
    find_bn_conv_pairs,
    find_conv_bn_pairs,
    fuse_bn_conv,
    fuse_bn_conv_weights,
    fuse_bn_weights,
    fuse_conv_bn,
    fuse_model_bn,
    fuse_model_bn_old,
)

__all__ = [
    "fuse_bn_weights",
    "fuse_bn_conv_weights",
    "fuse_conv_bn",
    "fuse_bn_conv",
    "find_conv_bn_pairs",
    "find_bn_conv_pairs",
    "fuse_model_bn",
    "fuse_model_bn_old",
]
