# Copyright (c) OpenMMLab. All rights reserved.
"""CenterPoint dense quantization scheme.

Wraps CenterPoint's :func:`quant_model` composition (Conv/Linear Q/DQ on the named submodules plus
the eSE / MaxPool / residual-add recipes) behind the uniform
:class:`~deployment.quantization.schemes.base.QuantizationScheme` lifecycle. Because the deploy
loader, the PTQ producer, and the QAT hook all build the tree by calling the *same* scheme's
:meth:`prepare`, they compose an identical quantized module tree and the PTQ ``state_dict`` lines up
with the deploy ``load_state_dict`` by construction — the same guarantee BEVFusion gets from its
:class:`~deployment.projects.bevfusion_l.quantization.schemes.SpconvInt8Scheme`.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List

from deployment.quantization.schemes.base import QuantizationScheme

logger = logging.getLogger(__name__)


class CenterPointDenseScheme(QuantizationScheme):
    """Fuse BN and insert CenterPoint's Q/DQ (Conv/Linear + eSE/MaxPool/residual recipes).

    Args:
        quant_backbone / quant_neck / quant_head / quant_voxel_encoder: Quantize each named tower.
        quant_add: Attach residual-add quantizers (ResNet / Sparse residual blocks).
        quant_linear_backbone: Quantize Linear layers inside ``pts_backbone`` (ConvNeXt).
        quant_ese_mul_identity / quant_ese_pool_input / quant_maxpool_input: VoVNet eSE / MaxPool
            Q/DQ placement.
        sensitive_layers: Fully-resolved dotted module names left in FP (skipped by ``quant_model``).
        fuse_bn: Fuse Conv+BN before quantizing (default True).
    """

    name = "centerpoint_dense"

    def __init__(
        self,
        *,
        quant_backbone: bool = True,
        quant_neck: bool = True,
        quant_head: bool = True,
        quant_voxel_encoder: bool = True,
        quant_add: bool = False,
        quant_linear_backbone: bool = False,
        quant_ese_mul_identity: bool = False,
        quant_ese_pool_input: bool = False,
        quant_maxpool_input: bool = False,
        sensitive_layers: Iterable[str] = (),
        fuse_bn: bool = True,
    ) -> None:
        self.quant_backbone = bool(quant_backbone)
        self.quant_neck = bool(quant_neck)
        self.quant_head = bool(quant_head)
        self.quant_voxel_encoder = bool(quant_voxel_encoder)
        self.quant_add = bool(quant_add)
        self.quant_linear_backbone = bool(quant_linear_backbone)
        self.quant_ese_mul_identity = bool(quant_ese_mul_identity)
        self.quant_ese_pool_input = bool(quant_ese_pool_input)
        self.quant_maxpool_input = bool(quant_maxpool_input)
        self.sensitive_layers: List[str] = list(sensitive_layers or ())
        self.fuse_bn = bool(fuse_bn)

    def prepare(self, model: Any) -> Any:
        from deployment.quantization import fuse_model_bn

        from .quant_model import quant_model

        if self.fuse_bn:
            model.eval()
            fuse_model_bn(model)
            logger.info("  [centerpoint_dense] fused Conv-BN")

        quant_model(
            model,
            quant_backbone=self.quant_backbone,
            quant_neck=self.quant_neck,
            quant_head=self.quant_head,
            quant_voxel_encoder=self.quant_voxel_encoder,
            quant_add=self.quant_add,
            quant_linear_backbone=self.quant_linear_backbone,
            quant_ese_mul_identity=self.quant_ese_mul_identity,
            quant_ese_pool_input=self.quant_ese_pool_input,
            quant_maxpool_input=self.quant_maxpool_input,
            skip_names=set(self.sensitive_layers),
        )
        logger.info("  [centerpoint_dense] inserted Q/DQ nodes")
        return model
