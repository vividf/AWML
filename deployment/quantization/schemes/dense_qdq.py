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
from typing import Any, Iterable, List, Sequence

from .base import QuantizationScheme

logger = logging.getLogger(__name__)


class DenseQDQScheme(QuantizationScheme):
    """Insert Conv2d Q/DQ into dense submodules and (unless disabled) residual-add quantizers.

    Model-agnostic and declarative: **every** dense submodule is BN-fused and its Conv2d quantized,
    except the subtrees named by ``keep_fp16`` (see :func:`deployment.quantization.expand_keep_fp16`).
    BN is fused across **all** submodules regardless of ``keep_fp16`` — fusing an un-quantized module's
    BN is an inference identity but *changes state_dict keys*, so PTQ and deploy must fuse the exact
    same set; ``keep_fp16`` only subtracts from the *quantized* set, never the *fused* set (spec §3.3b).

    Args:
        dense_submodules: Submodule names forming the dense tower (fused, and quantized minus
            ``keep_fp16``), e.g. ``("pts_backbone", "pts_neck", "bbox_head")``.
        keep_fp16: Glob patterns (subtree match) left in FP16; expanded against the model at
            ``prepare`` time.
        disable_recipes: Recipe names to skip. Only ``"add"`` (residual-add) applies to this scheme.
        fuse_bn: Master switch for BN fusion.
    """

    name = "dense_qdq"

    def __init__(
        self,
        dense_submodules: Sequence[str],
        *,
        keep_fp16: Iterable[str] = (),
        disable_recipes: Iterable[str] = (),
        fuse_bn: bool = True,
    ) -> None:
        self.dense_submodules: List[str] = list(dense_submodules)
        self.keep_fp16: List[str] = list(keep_fp16 or ())
        self.disable_recipes = set(disable_recipes)
        self.fuse_bn = bool(fuse_bn)

    def prepare(self, model: Any) -> Any:
        from deployment.quantization import expand_keep_fp16, fuse_model_bn, quant_conv_module
        from deployment.quantization.recipes.attach import attach_quant_add

        # Fuse BN across ALL dense submodules (keeps state_dict keys aligned on both sides).
        if self.fuse_bn:
            for target_name in self.dense_submodules:
                submodule = getattr(model, target_name, None)
                if submodule is None:
                    continue
                submodule.eval()
                fuse_model_bn(submodule)
                logger.info("  [dense_qdq] fused BN in %s", target_name)

        # Quantize every dense submodule's Conv2d, minus the keep_fp16 subtrees.
        skip = expand_keep_fp16(model, self.keep_fp16)
        for target_name in self.dense_submodules:
            submodule = getattr(model, target_name, None)
            if submodule is None:
                continue
            quant_conv_module(submodule, skip, target_name)
            logger.info("  [dense_qdq] quantized %s (Conv2d -> QuantConv2d)", target_name)

        if "add" not in self.disable_recipes:
            attach_quant_add(model)
            logger.info("  [dense_qdq] attached residual-add quantizers")
        return model
