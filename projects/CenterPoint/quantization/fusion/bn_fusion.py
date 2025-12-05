# Copyright (c) OpenMMLab. All rights reserved.
"""BatchNorm fusion utilities for quantization.

Fusing BatchNorm into preceding convolutions is important for quantization because:
1. It reduces the number of operations, improving inference speed
2. It eliminates a source of quantization error (BN scaling after quantized conv)
3. It's required for accurate fake quantization during QAT training
"""

from typing import List, Tuple, Union

import torch
import torch.nn as nn


def fuse_bn_weights(
    conv_weight: torch.Tensor,
    conv_bias: Union[torch.Tensor, None],
    bn_mean: torch.Tensor,
    bn_var: torch.Tensor,
    bn_eps: float,
    bn_weight: Union[torch.Tensor, None],
    bn_bias: Union[torch.Tensor, None],
) -> Tuple[nn.Parameter, nn.Parameter]:
    """
    Fuse BatchNorm parameters into convolution weights.

    The fused convolution computes:
        y = (W * x + b - mean) * (gamma / sqrt(var + eps)) + beta

    Which can be rewritten as:
        y = (W * gamma / sqrt(var + eps)) * x + (b - mean) * gamma / sqrt(var + eps) + beta

    So the fused weights are:
        W_fused = W * gamma / sqrt(var + eps)
        b_fused = (b - mean) * gamma / sqrt(var + eps) + beta

    Args:
        conv_weight: Convolution weight tensor [out_channels, in_channels, ...]
        conv_bias: Convolution bias tensor [out_channels] or None
        bn_mean: BatchNorm running mean [out_channels]
        bn_var: BatchNorm running variance [out_channels]
        bn_eps: BatchNorm epsilon
        bn_weight: BatchNorm weight (gamma) [out_channels] or None
        bn_bias: BatchNorm bias (beta) [out_channels] or None

    Returns:
        Tuple of (fused_weight, fused_bias) as nn.Parameters
    """
    # Handle None values
    if conv_bias is None:
        conv_bias = torch.zeros_like(bn_mean)
    if bn_weight is None:
        bn_weight = torch.ones_like(bn_mean)
    if bn_bias is None:
        bn_bias = torch.zeros_like(bn_mean)

    # Compute 1 / sqrt(var + eps)
    bn_var_rsqrt = torch.rsqrt(bn_var + bn_eps)

    # Compute scale factor: gamma / sqrt(var + eps)
    scale = bn_weight * bn_var_rsqrt

    # Reshape for broadcasting with conv weights
    # conv_weight shape: [out_channels, in_channels, ...] for Conv2d
    # or [out_channels, kernel_size, kernel_size, kernel_size, in_channels] for spconv
    shape = [-1] + [1] * (conv_weight.ndim - 1)

    # Fuse weights: W_fused = W * scale
    fused_weight = conv_weight * scale.reshape(shape)

    # Fuse bias: b_fused = (b - mean) * scale + beta
    fused_bias = (conv_bias - bn_mean) * scale + bn_bias

    return nn.Parameter(fused_weight.contiguous()), nn.Parameter(fused_bias.contiguous())


def fuse_conv_bn(conv: nn.Module, bn: nn.Module):
    """
    Fuse Conv and BatchNorm modules in-place.

    This modifies the conv module's weight and bias parameters to include
    the BatchNorm transformation, so the BN can be replaced with Identity.

    Args:
        conv: Convolution module (Conv1d, Conv2d, ConvTranspose2d, or Linear)
        bn: BatchNorm module (BatchNorm1d or BatchNorm2d)

    Raises:
        AssertionError: If modules are in training mode
    """
    assert not conv.training and not bn.training, "Fusion only works in eval mode"

    conv.weight, conv.bias = fuse_bn_weights(
        conv.weight,
        conv.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
    )


def find_conv_bn_pairs(model: nn.Module) -> List[Tuple[str, str]]:
    """
    Find all Conv-BN pairs in the model.

    This function identifies consecutive Conv and BatchNorm layers that
    can be fused together. It matches:
    - Conv1d + BatchNorm1d
    - Conv2d + BatchNorm2d
    - ConvTranspose2d + BatchNorm2d
    - Linear + BatchNorm1d

    Args:
        model: PyTorch model

    Returns:
        List of (conv_name, bn_name) tuples
    """
    pairs = []
    prev_name = None
    prev_module = None

    # Mapping of conv types to their expected BN types
    conv_to_bn = {
        nn.Conv1d: nn.BatchNorm1d,
        nn.Conv2d: nn.BatchNorm2d,
        nn.ConvTranspose2d: nn.BatchNorm2d,
        nn.Linear: nn.BatchNorm1d,
    }

    for name, module in model.named_modules():
        # Check if current module is a BN that follows a conv
        if prev_module is not None:
            for conv_type, bn_type in conv_to_bn.items():
                if isinstance(prev_module, conv_type) and isinstance(module, bn_type):
                    pairs.append((prev_name, name))
                    break

        prev_name = name
        prev_module = module

    return pairs


def _get_parent_module(model: nn.Module, name: str) -> Tuple[nn.Module, str]:
    """
    Get parent module and attribute name for a nested module.

    Args:
        model: Root model
        name: Dot-separated path to module (e.g., "backbone.layer1.conv1")

    Returns:
        Tuple of (parent_module, attr_name)
    """
    parts = name.split(".")
    parent = model

    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    return parent, parts[-1]


def fuse_model_bn(model: nn.Module, inplace: bool = True) -> nn.Module:
    """
    Fuse all Conv-BN pairs in the model.

    This function:
    1. Finds all Conv-BN pairs
    2. Fuses the BN parameters into the Conv weights
    3. Replaces the BN layers with Identity

    Args:
        model: PyTorch model
        inplace: If True, modify model in-place. If False, return a copy.

    Returns:
        Model with fused Conv-BN layers

    Example:
        >>> model.eval()
        >>> fuse_model_bn(model)
        >>> # Now all BN layers are replaced with Identity
    """
    if not inplace:
        import copy

        model = copy.deepcopy(model)

    # Must be in eval mode for fusion
    model.eval()

    # Find all Conv-BN pairs
    pairs = find_conv_bn_pairs(model)

    if len(pairs) == 0:
        print("No Conv-BN pairs found to fuse")
        return model

    # Build modules dict for fast lookup
    modules_dict = dict(model.named_modules())

    # Fuse each pair
    for conv_name, bn_name in pairs:
        conv = modules_dict[conv_name]
        bn = modules_dict[bn_name]

        # Fuse BN into conv
        fuse_conv_bn(conv, bn)

        # Replace BN with Identity
        parent, attr = _get_parent_module(model, bn_name)
        if attr.isdigit():
            parent[int(attr)] = nn.Identity()
        else:
            setattr(parent, attr, nn.Identity())

    print(f"Fused {len(pairs)} Conv-BN pairs")
    return model
