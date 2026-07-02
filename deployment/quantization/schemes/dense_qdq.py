# Copyright (c) OpenMMLab. All rights reserved.
"""Dense Q/DQ quantization scheme (pytorch_quantization ``QuantConv2d`` path).

Model-agnostic: it operates on named submodules of the model (e.g. ``pts_backbone``,
``pts_neck``, ``bbox_head``). It wraps the existing shared primitives
(:func:`deployment.quantization.fuse_model_bn`, :func:`deployment.quantization.quant_conv_module`,
:func:`deployment.quantization.recipes.attach.attach_quant_add`) behind the uniform
:class:`~deployment.quantization.schemes.base.QuantizationScheme` lifecycle, so the deploy loader
and the PTQ producer build an identical dense module tree by calling the *same* scheme.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Sequence

from .base import QuantizationScheme

logger = logging.getLogger(__name__)


class DenseQDQScheme(QuantizationScheme):
    """Insert Conv2d Q/DQ into dense submodules and (optionally) residual-add quantizers.

    Args:
        quant_targets: Submodule names to quantize (Conv2d -> QuantConv2d), e.g.
            ``("pts_backbone", "pts_neck", "bbox_head")``.
        fuse_targets: Submodule names whose Conv+BN are fused before quantization. Fusing an
            un-quantized submodule's BN is numerically an identity for inference, but it changes
            state_dict keys — so PTQ and deploy MUST fuse the exact same set. Defaults to
            ``quant_targets`` when not given; pass the full dense set explicitly to match a
            producer that fuses more than it quantizes.
        quant_add: Attach residual-add quantizers (``attach_quant_add`` on the whole model).
        sensitive_layers: Layer-name prefixes left in FP (skipped by ``quant_conv_module``).
        fuse_bn: Master switch for BN fusion.
    """

    name = "dense_qdq"

    def __init__(
        self,
        quant_targets: Sequence[str],
        *,
        fuse_targets: Optional[Sequence[str]] = None,
        quant_add: bool = False,
        sensitive_layers: Iterable[str] = (),
        fuse_bn: bool = True,
    ) -> None:
        self.quant_targets: List[str] = list(quant_targets)
        self.fuse_targets: List[str] = list(fuse_targets) if fuse_targets is not None else list(quant_targets)
        self.quant_add = bool(quant_add)
        self.sensitive_layers: List[str] = list(sensitive_layers or ())
        self.fuse_bn = bool(fuse_bn)

    def prepare(self, model: Any) -> Any:
        from deployment.quantization import fuse_model_bn, quant_conv_module
        from deployment.quantization.recipes.attach import attach_quant_add

        if self.fuse_bn:
            for target_name in self.fuse_targets:
                submodule = getattr(model, target_name, None)
                if submodule is None:
                    continue
                submodule.eval()
                fuse_model_bn(submodule)
                logger.info("  [dense_qdq] fused BN in %s", target_name)

        skip = set(self.sensitive_layers)
        for target_name in self.quant_targets:
            submodule = getattr(model, target_name, None)
            if submodule is None:
                continue
            quant_conv_module(submodule, skip, target_name)
            logger.info("  [dense_qdq] quantized %s (Conv2d -> QuantConv2d)", target_name)

        if self.quant_add:
            attach_quant_add(model)
            logger.info("  [dense_qdq] attached residual-add quantizers")
        return model
