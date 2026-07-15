# Copyright (c) OpenMMLab. All rights reserved.
"""BEVFusion sparse-encoder quantization scheme.

The sparse tower (``pts_middle_encoder``, spconv ``SparseConvolution``) deploys in **FP16** — it does
not use TensorRT INT8. The only structural step it needs is SparseConv+BN folding, so PTQ and deploy
build an identical BN-folded module tree and the ``state_dict`` keys line up on both sides.
"""

from __future__ import annotations

import logging
from typing import Any

from deployment.quantization.schemes.base import QuantizationScheme

logger = logging.getLogger(__name__)

SPARSE_SUBMODULE = "pts_middle_encoder"


class SpconvBnFuseScheme(QuantizationScheme):
    """Fuse SparseConv+BN in the BEVFusion sparse encoder (FP16 deploy path).

    Args:
        fuse_bn: Fuse SparseConv+BN in the encoder. When False, this scheme is a no-op.
    """

    name = "spconv_bn_fuse"

    def __init__(self, *, fuse_bn: bool = True) -> None:
        self.fuse_bn = bool(fuse_bn)

    def _encoder(self, model: Any):
        return getattr(model, SPARSE_SUBMODULE, None)

    def prepare(self, model: Any) -> Any:
        encoder = self._encoder(model)
        if encoder is None:
            logger.warning("[spconv_bn_fuse] no %s found; skipping sparse scheme", SPARSE_SUBMODULE)
            return model

        if self.fuse_bn:
            from deployment.quantization.sparse import fuse_spconv_bn_in_encoder

            count = fuse_spconv_bn_in_encoder(encoder)
            logger.info("  [spconv_bn_fuse] fused %d SparseConv-BN pairs in %s", count, SPARSE_SUBMODULE)
        return model
