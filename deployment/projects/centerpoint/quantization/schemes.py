# Copyright (c) OpenMMLab. All rights reserved.
"""CenterPoint dense quantization scheme.

Wraps CenterPoint's :func:`quant_model` composition (Conv/Linear Q/DQ on the named submodules plus
the eSE / MaxPool / residual-add recipes) behind the uniform
:class:`~deployment.quantization.schemes.base.QuantizationScheme` lifecycle. Because the deploy
loader, the PTQ producer, and the QAT hook all build the tree by calling the *same* scheme's
:meth:`prepare`, they compose an identical quantized module tree and the PTQ ``state_dict`` lines up
with the deploy ``load_state_dict`` by construction — the same guarantee BEVFusion gets from its
:class:`~deployment.projects.bevfusion_l.quantization.schemes.SpconvBnFuseScheme`.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List

from deployment.quantization.schemes.base import QuantizationScheme

logger = logging.getLogger(__name__)


class CenterPointDenseScheme(QuantizationScheme):
    """Fuse BN and insert CenterPoint's Q/DQ (Conv/Linear + eSE/MaxPool/residual recipes).

    Declarative surface (see spec.md §3): every standard tower is quantized INT8 except the subtrees
    named by ``keep_fp16``, and the architecture recipes are always attached (class-gated) unless named
    in ``disable_recipes``. All placement detail lives in :func:`quant_model`.

    Args:
        keep_fp16: Glob patterns (subtree match) left in FP16; expanded against the model at
            ``prepare`` time by :func:`deployment.quantization.expand_keep_fp16`.
        disable_recipes: Recipe names to skip ("add" / "ese" / "maxpool").
        fuse_bn: Fuse Conv+BN before quantizing (default True).
    """

    name = "centerpoint_dense"

    def __init__(
        self,
        *,
        keep_fp16: Iterable[str] = (),
        disable_recipes: Iterable[str] = (),
        fuse_bn: bool = True,
    ) -> None:
        self.keep_fp16: List[str] = list(keep_fp16 or ())
        self.disable_recipes = tuple(disable_recipes)
        self.fuse_bn = bool(fuse_bn)

    def prepare(self, model: Any) -> Any:
        from deployment.quantization import expand_keep_fp16, fuse_model_bn

        from .quant_model import quant_model

        if self.fuse_bn:
            model.eval()
            fuse_model_bn(model)
            logger.info("  [centerpoint_dense] fused Conv-BN")

        skip_names = expand_keep_fp16(model, self.keep_fp16)
        quant_model(model, skip_names=skip_names, disable_recipes=self.disable_recipes)
        logger.info("  [centerpoint_dense] inserted Q/DQ nodes")
        return model
