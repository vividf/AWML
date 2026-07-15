# Copyright (c) OpenMMLab. All rights reserved.
"""BEVFusion sparse-encoder quantization scheme.

The sparse tower (``pts_middle_encoder``, spconv ``SparseConvolution``) cannot use TensorRT-native
INT8; it needs a custom ``ImplicitGemmInt8`` plugin plus a post-export ONNX graph rewrite. This
scheme owns the *structural* step (:meth:`prepare`): fuse SparseConv+BN and, for INT8, attach NVIDIA
``TensorQuantizer`` to every ``SparseConvolution`` so the PTQ ``_amax`` keys load and the exported
ONNX can later be rewritten to ``ImplicitGemmInt8``.

Two modes (selected by ``int8``):

* ``int8=False`` — keep the sparse tower FP16 but still fuse SparseConv+BN (matches the PTQ
  checkpoint, which is BN-folded, and keeps the FP16 sparse subgraph comparable to the INT8 one).
* ``int8=True``  — fuse BN, then attach NVIDIA ``TensorQuantizer`` to every ``SparseConvolution``
  (via :func:`apply_nvidia_spconv_int8`).

The post-export ONNX rewrite (``ImplicitGemm`` -> ``ImplicitGemmInt8``) lives in the export pipeline
(:mod:`deployment.projects.bevfusion_l.export.onnx_export_pipeline`), driven by
:mod:`deployment.projects.bevfusion_l.export.sparse_int8_onnx_transform`; this scheme only prepares the
module tree so PTQ and deploy load an identical structure.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List

import torch

from deployment.quantization.schemes.base import QuantizationScheme

logger = logging.getLogger(__name__)

SPARSE_SUBMODULE = "pts_middle_encoder"


class SpconvInt8Scheme(QuantizationScheme):
    """Quantization scheme for the BEVFusion spconv sparse encoder.

    Args:
        int8: Attach NVIDIA ``TensorQuantizer`` to sparse convs (INT8). If False, only BN fusion.
        fuse_bn: Fuse SparseConv+BN in the encoder before quantizing.
        fp16_layers: Case-insensitive substrings matched on sparse-conv module names; matched
            convs are kept FP16 (no quantizer). MUST match the PTQ run so the module tree lines
            up with the checkpoint. Only meaningful when ``int8=True``.
        register_load_buffers: Register the two sparse-scale buffers so a PTQ ``load_state_dict``
            fills them instead of reporting unexpected keys (deploy side). Only for ``int8=True``.
    """

    name = "spconv_int8"

    def __init__(
        self,
        *,
        int8: bool = True,
        fuse_bn: bool = True,
        fp16_layers: Iterable[str] = (),
        register_load_buffers: bool = True,
    ) -> None:
        self.int8 = bool(int8)
        self.fuse_bn = bool(fuse_bn)
        self.fp16_layers: List[str] = [str(p) for p in (fp16_layers or []) if str(p).strip()]
        self.register_load_buffers = bool(register_load_buffers)

    def _encoder(self, model: Any):
        return getattr(model, SPARSE_SUBMODULE, None)

    def prepare(self, model: Any) -> Any:
        encoder = self._encoder(model)
        if encoder is None:
            logger.warning("[spconv_int8] no %s found; skipping sparse scheme", SPARSE_SUBMODULE)
            return model

        if self.fuse_bn:
            from deployment.quantization.sparse import fuse_spconv_bn_in_encoder

            count = fuse_spconv_bn_in_encoder(encoder)
            logger.info("  [spconv_int8] fused %d SparseConv-BN pairs in %s", count, SPARSE_SUBMODULE)

        if self.int8:
            from deployment.quantization.sparse.spconv_int8 import apply_nvidia_spconv_int8

            encoder.eval()
            apply_nvidia_spconv_int8(encoder, exclude_patterns=list(self.fp16_layers))
            if self.register_load_buffers:
                self._register_scale_buffers(encoder)
            logger.info(
                "  [spconv_int8] attached NVIDIA TensorQuantizer to %s (fp16_layers=%d)",
                SPARSE_SUBMODULE,
                len(self.fp16_layers),
            )
        return model

    @staticmethod
    def _register_scale_buffers(encoder: torch.nn.Module) -> None:
        """Register the two sparse-scale buffers PTQ saves, so ``load_state_dict`` fills them."""
        for buf_name in ("_sparse_tail_absmax", "_last_int8_conv_output_absmax"):
            if not hasattr(encoder, buf_name):
                encoder.register_buffer(buf_name, torch.tensor(0.0, dtype=torch.float32))
