# Copyright (c) OpenMMLab. All rights reserved.
"""Generic module-replacement engine for quantization.

Model-agnostic Q/DQ insertion: swap ``nn.Conv2d`` / ``nn.ConvTranspose2d`` / ``nn.Linear`` for their
quantized subclasses. Architecture-specific placement (residual-add / eSE / OSA forward hooks) lives
in :mod:`deployment.quantization.recipes`; project composition (``quant_model``) lives in the project
bundle (e.g. ``deployment.projects.centerpoint.quantization``).
"""

from typing import Optional, Set, Type

import torch
import torch.nn as nn

from .descriptors import (
    conv2d_weight_desc,
    conv_transpose2d_weight_desc,
    default_input_desc,
    linear_weight_desc,
)
from .modules import QuantConv2d, QuantConvTranspose2d, QuantLinear

# Flag to track if quantization descriptors have been initialized
_quant_descriptors_initialized = False


def ensure_quant_descriptors_initialized():
    """Populate the ``default_quant_desc_*`` class attributes from the shared descriptor factory.

    Must be called before :func:`transfer_to_quantization` / the module rebuilds, since the
    ``default_quant_desc_*`` attributes are otherwise only set the first time a quantized module is
    constructed. Idempotent. The descriptor *choices* live in :mod:`.descriptors` so this path and
    the module ``__init__`` path cannot disagree.
    """
    global _quant_descriptors_initialized
    if _quant_descriptors_initialized:
        return

    if QuantConv2d.default_quant_desc_input is None:
        QuantConv2d.default_quant_desc_input = default_input_desc()
    if QuantConv2d.default_quant_desc_weight is None:
        QuantConv2d.default_quant_desc_weight = conv2d_weight_desc()

    if QuantConvTranspose2d.default_quant_desc_input is None:
        QuantConvTranspose2d.default_quant_desc_input = default_input_desc()
    if QuantConvTranspose2d.default_quant_desc_weight is None:
        QuantConvTranspose2d.default_quant_desc_weight = conv_transpose2d_weight_desc()

    # Guard rail: if someone set a per-channel ConvTranspose2d weight descriptor, force it back to
    # per-tensor (see :func:`~deployment.quantization.core.descriptors.conv_transpose2d_weight_desc`).
    try:
        qdw = QuantConvTranspose2d.default_quant_desc_weight
        if getattr(qdw, "axis", None) not in (None, (), []):
            QuantConvTranspose2d.default_quant_desc_weight = conv_transpose2d_weight_desc()
    except Exception:
        # Be conservative: never fail descriptor initialization due to this guard.
        pass

    if QuantLinear.default_quant_desc_input is None:
        QuantLinear.default_quant_desc_input = default_input_desc()
    if QuantLinear.default_quant_desc_weight is None:
        QuantLinear.default_quant_desc_weight = linear_weight_desc()

    _quant_descriptors_initialized = True


def _rebuild_conv2d_as_quant(conv: nn.Conv2d) -> QuantConv2d:
    """Build QuantConv2d via ``__init__`` + weight copy (no ``__dict__`` transplant).

    Copying ``vars(conv)`` onto a ``QuantConv2d`` shell can carry MMEngine/spconv hooks or
    half-initialized state that interacts badly with fake tensors during
    ``TensorQuantizer`` setup. PTQ deploy load uses this path for robustness.
    """
    ensure_quant_descriptors_initialized()
    q = QuantConv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    q = q.to(device=conv.weight.device, dtype=conv.weight.dtype)
    with torch.no_grad():
        q.weight.copy_(conv.weight)
        if conv.bias is not None:
            q.bias.copy_(conv.bias)
    q.init_quantizer(
        QuantConv2d.default_quant_desc_input,
        QuantConv2d.default_quant_desc_weight,
    )
    return q


def _rebuild_conv_transpose2d_as_quant(conv: nn.ConvTranspose2d) -> QuantConvTranspose2d:
    """Same as :func:`_rebuild_conv2d_as_quant` for transposed conv (FPN upsample)."""
    ensure_quant_descriptors_initialized()
    q = QuantConvTranspose2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        output_padding=conv.output_padding,
        groups=conv.groups,
        bias=conv.bias is not None,
        dilation=conv.dilation,
        padding_mode=conv.padding_mode,
    )
    q = q.to(device=conv.weight.device, dtype=conv.weight.dtype)
    with torch.no_grad():
        q.weight.copy_(conv.weight)
        if conv.bias is not None:
            q.bias.copy_(conv.bias)
    q.init_quantizer(
        QuantConvTranspose2d.default_quant_desc_input,
        QuantConvTranspose2d.default_quant_desc_weight,
    )
    return q


def _clear_module_hooks(module: nn.Module) -> None:
    """Remove forward/backward/state_dict hooks copied from the original nn.Conv2d.

    Rare third-party / registry hooks can interact badly with quantization init; hooks are not needed
    on the quantized clone for deployment load.
    """
    for name in (
        "_forward_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_state_dict_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    ):
        d = getattr(module, name, None)
        if d is not None and hasattr(d, "clear"):
            d.clear()


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
    # Ensure quantization descriptors are initialized
    ensure_quant_descriptors_initialized()

    # Create new instance without calling __init__
    quant_instance = quant_module.__new__(quant_module)

    # Copy all attributes from original module
    for k, val in vars(nn_instance).items():
        setattr(quant_instance, k, val)

    _clear_module_hooks(quant_instance)

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

    # Check if model is None or not a valid nn.Module
    if model is None or not isinstance(model, nn.Module):
        return

    for name in list(model._modules.keys()):
        submodule = model._modules[name]
        full_name = f"{prefix}.{name}" if prefix else name

        # Skip entire subtree if this module name is in skip list
        # (This enables skipping containers like 'pts_backbone.blocks.0')
        if full_name in skip_names:
            continue

        # Recursively process submodules (only if submodule is not None)
        if submodule is not None:
            quant_conv_module(submodule, skip_names, full_name)

        # Replace Conv2d with QuantConv2d
        if isinstance(submodule, nn.Conv2d) and not isinstance(submodule, QuantConv2d):
            model._modules[name] = _rebuild_conv2d_as_quant(submodule)

        # Replace ConvTranspose2d with QuantConvTranspose2d
        elif isinstance(submodule, nn.ConvTranspose2d) and not isinstance(submodule, QuantConvTranspose2d):
            model._modules[name] = _rebuild_conv_transpose2d_as_quant(submodule)


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

    # Check if model is None or not a valid nn.Module
    if model is None or not isinstance(model, nn.Module):
        return

    for name in list(model._modules.keys()):
        submodule = model._modules[name]
        full_name = f"{prefix}.{name}" if prefix else name

        # Skip entire subtree if this module name is in skip list
        if full_name in skip_names:
            continue

        # Recursively process submodules (only if submodule is not None)
        if submodule is not None:
            quant_linear_module(submodule, skip_names, full_name)

        # Replace Linear with QuantLinear
        if isinstance(submodule, nn.Linear) and not isinstance(submodule, QuantLinear):
            model._modules[name] = transfer_to_quantization(submodule, QuantLinear)
