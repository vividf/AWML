"""Fuse ``SparseConvolution + BatchNorm1d`` in a BEVFusion sparse encoder (FP16 export path).

Thin re-export of the framework's single BN-fold implementation
(:func:`deployment.quantization.sparse.fuse_spconv_bn_in_encoder`). The fold itself is *not*
quantization — it only rewrites the module tree so the exported sparse ONNX is BN-free and matches
the runtime graph — but it lives in ``quantization.sparse`` so the PTQ producer, the deploy loader,
and this export path all share one implementation and cannot drift.
"""

from __future__ import annotations

from deployment.quantization.sparse import fuse_spconv_bn_in_encoder

__all__ = ["fuse_spconv_bn_in_encoder"]
