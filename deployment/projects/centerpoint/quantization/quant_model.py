# Copyright (c) OpenMMLab. All rights reserved.
"""CenterPoint-specific quantization composition.

``quant_model`` applies the framework's generic Q/DQ engine plus the architecture recipes to a
CenterPoint model's named components (``pts_backbone`` / ``pts_neck`` / ``pts_bbox_head`` /
``pts_voxel_encoder``). It is the model-specific glue that the framework deliberately does not
own (:mod:`deployment.quantization` stays model-agnostic); the deploy loader and the PTQ producer
CLI both call it.
"""

from typing import Iterable, Optional, Set

import torch.nn as nn

from deployment.quantization.core.replace import quant_conv_module, quant_linear_module
from deployment.quantization.recipes.attach import (
    attach_ese_quantizers,
    attach_maxpool_input_quantizer,
    attach_quant_add,
)


def quant_model(
    model: nn.Module,
    skip_names: Optional[Set[str]] = None,
    disable_recipes: Iterable[str] = (),
):
    """Apply CenterPoint Q/DQ: quantize every quantizable module in the standard towers (minus
    ``skip_names``), then attach the always-on architecture recipes.

    Precision placement is declarative: ``skip_names`` — the expanded ``keep_fp16`` set (see
    :func:`deployment.quantization.expand_keep_fp16`) — is the single opt-out, and it skips a matched
    module and its whole subtree. Every tower is reached unconditionally; a fully-kept tower (e.g.
    ``pts_voxel_encoder``) is skipped because its descendants are all in ``skip_names``.

    The per-tower module kinds are architecture facts and live here in code, not in config:
    ``pts_backbone`` → Conv **and** Linear (ConvNeXt pointwise), ``pts_neck`` / ``pts_bbox_head`` →
    Conv, ``pts_voxel_encoder`` → Linear.

    Recipes (residual-add, eSE, maxpool) are attached always and are class-gated, so each fires only
    where the architecture has that module. ``disable_recipes`` turns one off by name ("add" / "ese" /
    "maxpool") — needed only where the architecture *has* the module but the config wants it FP16
    (e.g. SECOND / BEVFusion keep ``add`` off). The eSE recipe is the single-Q-at-eSE-input,
    reformat-minimizing INT8 path (see :mod:`deployment.quantization.recipes.forward_hooks`).

    Args:
        model: CenterPoint model.
        skip_names: Module names (and their subtrees) to leave in FP16.
        disable_recipes: Recipe names to skip ("add", "ese", "maxpool").

    Example:
        >>> quant_model(model, skip_names={"pts_voxel_encoder", "pts_backbone.stem"})
        >>> quant_model(model, disable_recipes=["add"])   # keep residual-add in FP16
    """
    skip_names = skip_names or set()
    disabled = set(disable_recipes)

    if hasattr(model, "pts_backbone"):
        quant_conv_module(model.pts_backbone, skip_names, "pts_backbone")
        quant_linear_module(model.pts_backbone, skip_names, "pts_backbone")
    if hasattr(model, "pts_neck"):
        quant_conv_module(model.pts_neck, skip_names, "pts_neck")
    if hasattr(model, "pts_bbox_head"):
        quant_conv_module(model.pts_bbox_head, skip_names, "pts_bbox_head")
    if hasattr(model, "pts_voxel_encoder"):
        quant_linear_module(model.pts_voxel_encoder, skip_names, "pts_voxel_encoder")

    if "add" not in disabled:
        attach_quant_add(model)
    if "ese" not in disabled:
        attach_ese_quantizers(model)
    if "maxpool" not in disabled:
        attach_maxpool_input_quantizer(model, skip_names)
