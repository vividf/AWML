# Copyright (c) OpenMMLab. All rights reserved.
"""Module replacement functions for quantization."""

from typing import Optional, Set, Type

import torch.nn as nn

from .modules import QuantConv2d, QuantConvTranspose2d, QuantLinear


def transfer_to_quantization(nn_instance: nn.Module, quant_module: Type) -> nn.Module:
    """
    Transfer weights and attributes from original module to quantized version.

    This function creates a new quantized module instance and copies all
    attributes from the original module, then initializes the quantizers.

    Args:
        nn_instance: Original PyTorch module (Conv2d, Linear, etc.)
        quant_module: Quantized module class (QuantConv2d, QuantLinear, etc.)

    Returns:
        Quantized module with copied weights and initialized quantizers
    """
    # Create new instance without calling __init__
    quant_instance = quant_module.__new__(quant_module)

    # Copy all attributes from original module
    for k, val in vars(nn_instance).items():
        setattr(quant_instance, k, val)

    # Initialize quantizers
    quant_instance.init_quantizer(
        quant_module.default_quant_desc_input,
        quant_module.default_quant_desc_weight,
    )

    return quant_instance


def quant_conv_module(model: nn.Module, skip_names: Optional[Set[str]] = None, prefix: str = ""):
    """
    Replace all Conv2d and ConvTranspose2d modules with quantized versions.

    This function recursively traverses the model and replaces all Conv2d
    and ConvTranspose2d modules with QuantConv2d and QuantConvTranspose2d
    respectively, except for modules whose names are in skip_names.

    Args:
        model: PyTorch model to modify
        skip_names: Set of module names to skip (full path from model root)
        prefix: Current prefix for module naming (used in recursion)

    Example:
        >>> model = CenterPoint(...)
        >>> quant_conv_module(model.pts_backbone)
        >>> quant_conv_module(model.pts_neck)
        >>> quant_conv_module(model.pts_bbox_head)
    """
    skip_names = skip_names or set()

    for name in list(model._modules.keys()):
        submodule = model._modules[name]
        full_name = f"{prefix}.{name}" if prefix else name

        # Recursively process submodules
        quant_conv_module(submodule, skip_names, full_name)

        # Skip if in skip list
        if full_name in skip_names:
            continue

        # Replace Conv2d with QuantConv2d
        if isinstance(submodule, nn.Conv2d) and not isinstance(submodule, QuantConv2d):
            model._modules[name] = transfer_to_quantization(submodule, QuantConv2d)

        # Replace ConvTranspose2d with QuantConvTranspose2d
        elif isinstance(submodule, nn.ConvTranspose2d) and not isinstance(submodule, QuantConvTranspose2d):
            model._modules[name] = transfer_to_quantization(submodule, QuantConvTranspose2d)


def quant_linear_module(model: nn.Module, skip_names: Optional[Set[str]] = None, prefix: str = ""):
    """
    Replace all Linear modules with quantized versions.

    This function recursively traverses the model and replaces all Linear
    modules with QuantLinear, except for modules whose names are in skip_names.

    Args:
        model: PyTorch model to modify
        skip_names: Set of module names to skip (full path from model root)
        prefix: Current prefix for module naming (used in recursion)

    Example:
        >>> model = CenterPoint(...)
        >>> quant_linear_module(model.pts_voxel_encoder)
    """
    skip_names = skip_names or set()

    for name in list(model._modules.keys()):
        submodule = model._modules[name]
        full_name = f"{prefix}.{name}" if prefix else name

        # Recursively process submodules
        quant_linear_module(submodule, skip_names, full_name)

        # Skip if in skip list
        if full_name in skip_names:
            continue

        # Replace Linear with QuantLinear
        if isinstance(submodule, nn.Linear) and not isinstance(submodule, QuantLinear):
            model._modules[name] = transfer_to_quantization(submodule, QuantLinear)


def quant_model(
    model: nn.Module,
    quant_backbone: bool = True,
    quant_neck: bool = True,
    quant_head: bool = True,
    quant_voxel_encoder: bool = True,
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
        skip_names: Set of module names to skip

    Example:
        >>> model = CenterPoint(...)
        >>> quant_model(model, skip_names={'pts_backbone.blocks.0'})
    """
    skip_names = skip_names or set()

    if quant_backbone and hasattr(model, "pts_backbone"):
        quant_conv_module(model.pts_backbone, skip_names, "pts_backbone")

    if quant_neck and hasattr(model, "pts_neck"):
        quant_conv_module(model.pts_neck, skip_names, "pts_neck")

    if quant_head and hasattr(model, "pts_bbox_head"):
        quant_conv_module(model.pts_bbox_head, skip_names, "pts_bbox_head")

    if quant_voxel_encoder and hasattr(model, "pts_voxel_encoder"):
        quant_linear_module(model.pts_voxel_encoder, skip_names, "pts_voxel_encoder")
