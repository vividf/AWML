# Copyright (c) OpenMMLab. All rights reserved.
"""CenterPoint-specific quantization composition.

``quant_model`` applies the framework's generic Q/DQ engine plus the architecture recipes to a
CenterPoint model's named components (``pts_backbone`` / ``pts_neck`` / ``pts_bbox_head`` /
``pts_voxel_encoder``). It is the model-specific glue that the framework deliberately does not
own (:mod:`deployment.quantization` stays model-agnostic); the deploy loader and the PTQ producer
CLI both call it.
"""

from typing import Optional, Set

import torch.nn as nn

from deployment.quantization.core.replace import quant_conv_module, quant_linear_module
from deployment.quantization.recipes.attach import (
    attach_ese_mul_identity_quantizer,
    attach_ese_pool_input_quantizer,
    attach_maxpool_input_quantizer,
    attach_quant_add,
)


def quant_model(
    model: nn.Module,
    quant_backbone: bool = True,
    quant_neck: bool = True,
    quant_head: bool = True,
    quant_voxel_encoder: bool = True,
    quant_add: bool = False,
    quant_linear_backbone: bool = False,
    quant_ese_mul_identity: bool = False,
    quant_ese_pool_input: bool = False,
    quant_maxpool_input: bool = False,
    skip_names: Optional[Set[str]] = None,
):
    """
    Apply quantization to CenterPoint model components.

    This is a convenience function that applies quantization to specified
    components of a CenterPoint model.

    Args:
        model: CenterPoint model
        quant_backbone: Whether to quantize pts_backbone
        quant_neck: Whether to quantize pts_neck
        quant_head: Whether to quantize pts_bbox_head
        quant_voxel_encoder: Whether to quantize pts_voxel_encoder
        quant_linear_backbone: Whether to quantize Linear layers in pts_backbone
        quant_ese_mul_identity: Whether to quantize both inputs to eSE Mul (identity + gate) for INT8; both get Q-DQ before Mul.
        quant_ese_pool_input: Whether to add Q/DQ before pooling layer in eSE (VoVNet)
        quant_maxpool_input: Whether to add Q/DQ before MaxPool2d (e.g. VoVNet _OSA_stage)
        skip_names: Set of module names to skip

    Example:
        >>> model = CenterPoint(...)
        >>> quant_model(model, skip_names={'pts_backbone.blocks.0'})
        >>> quant_model(model, quant_ese_mul_identity=True, quant_ese_pool_input=True)  # eSE INT8
        >>> quant_model(model, quant_maxpool_input=True)   # QDQ before MaxPool2d
    """
    skip_names = skip_names or set()

    if quant_backbone and hasattr(model, "pts_backbone"):
        quant_conv_module(model.pts_backbone, skip_names, "pts_backbone")
        if quant_linear_backbone:
            quant_linear_module(model.pts_backbone, skip_names, "pts_backbone")

    if quant_neck and hasattr(model, "pts_neck"):
        quant_conv_module(model.pts_neck, skip_names, "pts_neck")

    if quant_head and hasattr(model, "pts_bbox_head"):
        quant_conv_module(model.pts_bbox_head, skip_names, "pts_bbox_head")

    if quant_voxel_encoder and hasattr(model, "pts_voxel_encoder"):
        quant_linear_module(model.pts_voxel_encoder, skip_names, "pts_voxel_encoder")

    if quant_add:
        attach_quant_add(model)

    if quant_ese_pool_input:
        attach_ese_pool_input_quantizer(model)
    if quant_ese_mul_identity:
        attach_ese_mul_identity_quantizer(model)

    if quant_maxpool_input:
        attach_maxpool_input_quantizer(model, skip_names)
