# Copyright (c) OpenMMLab. All rights reserved.
"""SparseConv + BatchNorm folding for SECOND-style spconv encoders.

The fold itself is **not** quantization — it rewrites the module tree so the exported sparse ONNX is
BN-free and matches the runtime graph. It is the single source of truth used by the PTQ producer,
the deploy loader, and the BEVFusion FP16 sparse export path, so the module tree — and therefore the
``state_dict`` keys — line up on every side.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def fuse_spconv_bn_in_encoder(sparse_encoder: nn.Module) -> int:
    """Fuse each ``SparseConvolution + BatchNorm1d`` pair in ``sparse_encoder`` (eval mode).

    Single source of truth for SECOND-style spconv BN folding: used by the PTQ producer, the
    deploy loader, and the FP16 sparse export path, so the module tree — and therefore the
    ``state_dict`` keys — line up on every side. Returns the number of fused Conv-BN pairs.
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
