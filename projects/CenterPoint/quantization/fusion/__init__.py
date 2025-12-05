# Copyright (c) OpenMMLab. All rights reserved.
"""Layer fusion utilities for quantization."""

from .bn_fusion import find_conv_bn_pairs, fuse_bn_weights, fuse_conv_bn, fuse_model_bn

__all__ = [
    "fuse_bn_weights",
    "fuse_conv_bn",
    "find_conv_bn_pairs",
    "fuse_model_bn",
]
