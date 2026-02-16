# Copyright (c) OpenMMLab. All rights reserved.
"""BatchNorm fusion utilities for quantization.

Fusing BatchNorm into preceding convolutions is important for quantization because:
1. It reduces the number of operations, improving inference speed
2. It eliminates a source of quantization error (BN scaling after quantized conv)
3. It's required for accurate fake quantization during QAT training
"""

from typing import Iterator, List, Tuple, Union

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
    is_transposed: bool = False,
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
        conv_weight: Convolution weight tensor
            - For Conv2d: [out_channels, in_channels, H, W]
            - For ConvTranspose2d: [in_channels, out_channels, H, W]
        conv_bias: Convolution bias tensor [out_channels] or None
        bn_mean: BatchNorm running mean [out_channels]
        bn_var: BatchNorm running variance [out_channels]
        bn_eps: BatchNorm epsilon
        bn_weight: BatchNorm weight (gamma) [out_channels] or None
        bn_bias: BatchNorm bias (beta) [out_channels] or None
        is_transposed: If True, conv_weight is from ConvTranspose2d with shape
            [in_channels, out_channels, H, W] where scale applies to dim 1

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
    # Conv2d weight shape: [out_channels, in_channels, H, W] -> scale on dim 0
    # ConvTranspose2d weight shape: [in_channels, out_channels, H, W] -> scale on dim 1
    if is_transposed:
        # For ConvTranspose2d: scale applies to dimension 1 (out_channels)
        shape = [1, -1] + [1] * (conv_weight.ndim - 2)
    else:
        # For Conv2d/Linear: scale applies to dimension 0 (out_channels)
        shape = [-1] + [1] * (conv_weight.ndim - 1)

    # Fuse weights: W_fused = W * scale
    fused_weight = conv_weight * scale.reshape(shape)

    # Fuse bias: b_fused = (b - mean) * scale + beta
    fused_bias = (conv_bias - bn_mean) * scale + bn_bias

    return nn.Parameter(fused_weight.contiguous()), nn.Parameter(fused_bias.contiguous())


def fuse_bn_conv_weights(
    conv_weight: torch.Tensor,
    conv_bias: Union[torch.Tensor, None],
    bn_mean: torch.Tensor,
    bn_var: torch.Tensor,
    bn_eps: float,
    bn_weight: Union[torch.Tensor, None],
    bn_bias: Union[torch.Tensor, None],
    is_transposed: bool = False,
    groups: int = 1,
) -> Tuple[nn.Parameter, nn.Parameter]:
    """
    Fuse a **preceding** BatchNorm into the **following** convolution weights.

    For the BN → Conv pattern::

        s_i = alpha_i * x_i + beta_i          (BN)
        y_o = sum_i W_{o,i} * s_i + b_o       (Conv)

    Expanding::

        y_o = sum_i (W_{o,i} * alpha_i) * x_i + (sum_i W_{o,i} * beta_i + b_o)

    So the fused weights are::

        W_fused_{o,i} = W_{o,i} * alpha_i           (scale along **input** channel)
        b_fused_o     = b_o + sum_i W_{o,i} * beta_i

    where::

        alpha_i = gamma_i / sqrt(var_i + eps)
        beta_i  = bn_bias_i - alpha_i * mean_i

    For grouped convolutions the scaling and bias summation are performed
    independently per group.

    Args:
        conv_weight: Convolution weight tensor
            - For Conv2d: [out_channels, in_channels/groups, H, W]
            - For ConvTranspose2d: [in_channels, out_channels/groups, H, W]
            - For Linear: [out_features, in_features]
        conv_bias: Convolution bias tensor or None
        bn_mean: BatchNorm running mean [num_features]
        bn_var: BatchNorm running variance [num_features]
        bn_eps: BatchNorm epsilon
        bn_weight: BatchNorm weight (gamma) [num_features] or None
        bn_bias: BatchNorm bias (beta) [num_features] or None
        is_transposed: If True, conv_weight is from ConvTranspose2d where
            dim 0 is in_channels and dim 1 is out_channels/groups
        groups: Number of groups in the convolution (default 1)

    Returns:
        Tuple of (fused_weight, fused_bias) as nn.Parameters
    """
    num_features = bn_mean.shape[0]

    # Handle None values
    if bn_weight is None:
        bn_weight = torch.ones_like(bn_mean)
    if bn_bias is None:
        bn_bias = torch.zeros_like(bn_mean)

    # BN affine: alpha = gamma / sqrt(var + eps),  beta = bn_bias - alpha * mean
    bn_var_rsqrt = torch.rsqrt(bn_var + bn_eps)
    alpha = bn_weight * bn_var_rsqrt  # [I]
    beta = bn_bias - alpha * bn_mean  # [I]

    I_per_group = num_features // groups

    if is_transposed:
        # ConvTranspose2d weight: [in_channels, out_channels/groups, ...]
        O_per_group = conv_weight.shape[1]
        spatial = conv_weight.shape[2:]

        if conv_bias is None:
            conv_bias = torch.zeros(O_per_group * groups, device=conv_weight.device, dtype=conv_weight.dtype)

        # Scale weights along dim 0 (input channels)
        fused_weight = conv_weight * alpha.reshape(-1, *([1] * (conv_weight.ndim - 1)))

        # Bias: b'_o = b_o + sum_{i in group} beta_i * sum_{spatial} W[i, o_local, ...]
        W_grouped = conv_weight.reshape(groups, I_per_group, O_per_group, *spatial)
        beta_r = beta.reshape(groups, I_per_group, *([1] * (1 + len(spatial))))
        # Sum over dim 1 (I/G) and spatial dims (3, 4, ...)
        sum_dims = (1,) + tuple(range(3, 3 + len(spatial)))
        bias_offset = (W_grouped * beta_r).sum(dim=sum_dims)  # [G, O/G]
        fused_bias = conv_bias + bias_offset.reshape(-1)
    else:
        # Conv2d weight: [out_channels, in_channels/groups, ...]
        # Linear weight: [out_features, in_features]
        O = conv_weight.shape[0]
        O_per_group = O // groups
        spatial = conv_weight.shape[2:]

        if conv_bias is None:
            conv_bias = torch.zeros(O, device=conv_weight.device, dtype=conv_weight.dtype)

        # Reshape into groups to scale per-group input channels
        W_grouped = conv_weight.reshape(groups, O_per_group, I_per_group, *spatial)
        alpha_r = alpha.reshape(groups, 1, I_per_group, *([1] * len(spatial)))
        fused_weight = (W_grouped * alpha_r).reshape(conv_weight.shape)

        # Bias: b'_o = b_o + sum_{i in group} W_{o,i} * beta_i
        beta_r = beta.reshape(groups, 1, I_per_group, *([1] * len(spatial)))
        sum_dims = tuple(range(2, 2 + 1 + len(spatial)))  # I/G and spatial dims
        bias_offset = (W_grouped * beta_r).sum(dim=sum_dims)  # [G, O/G]
        fused_bias = conv_bias + bias_offset.reshape(-1)

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

    # Check if this is a transposed convolution
    is_transposed = isinstance(conv, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d))

    conv.weight, conv.bias = fuse_bn_weights(
        conv.weight,
        conv.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
        is_transposed=is_transposed,
    )


def fuse_bn_conv(bn: nn.Module, conv: nn.Module):
    """
    Fuse a preceding BatchNorm into the following Conv module in-place (BN → Conv).

    This modifies the conv module's weight and bias parameters to absorb
    the BatchNorm transformation, so the BN can be replaced with Identity.

    Args:
        bn: BatchNorm module (BatchNorm1d or BatchNorm2d)
        conv: Convolution module (Conv1d, Conv2d, ConvTranspose2d, or Linear)

    Raises:
        AssertionError: If modules are in training mode
    """
    assert not bn.training and not conv.training, "Fusion only works in eval mode"

    is_transposed = isinstance(conv, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d))
    groups = getattr(conv, "groups", 1)

    conv.weight, conv.bias = fuse_bn_conv_weights(
        conv.weight,
        conv.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
        is_transposed=is_transposed,
        groups=groups,
    )


def _get_conv_out_channels(conv: nn.Module) -> int:
    """Get output channels from a Conv or Linear module."""
    if isinstance(conv, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return conv.out_channels
    elif isinstance(conv, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        return conv.out_channels
    elif isinstance(conv, nn.Linear):
        return conv.out_features
    else:
        raise ValueError(f"Unsupported module type: {type(conv)}")


def _get_conv_in_channels(conv: nn.Module) -> int:
    """Get input channels from a Conv or Linear module."""
    if isinstance(
        conv,
        (
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        ),
    ):
        return conv.in_channels
    elif isinstance(conv, nn.Linear):
        return conv.in_features
    else:
        raise ValueError(f"Unsupported module type: {type(conv)}")


def _get_bn_num_features(bn: nn.Module) -> int:
    """Get num_features from a BatchNorm module."""
    return bn.num_features


def _iter_adjacent_named_children(
    model: nn.Module, prefix: str = ""
) -> Iterator[Tuple[str, nn.Module, str, nn.Module]]:
    """
    Iterate adjacent sibling module pairs in the module tree.

    Unlike scanning ``named_modules()`` linearly, this only emits adjacent
    modules that share the same parent container, preventing accidental
    cross-boundary pairing (e.g., last BN of one block with first Conv of
    another block).
    """
    children = list(model._modules.items())

    # Adjacent siblings under the same parent.
    for i in range(len(children) - 1):
        left_name, left_module = children[i]
        right_name, right_module = children[i + 1]
        if left_module is None or right_module is None:
            continue

        left_full = f"{prefix}.{left_name}" if prefix else left_name
        right_full = f"{prefix}.{right_name}" if prefix else right_name
        yield left_full, left_module, right_full, right_module

    # Recurse into each child.
    for child_name, child_module in children:
        if child_module is None:
            continue
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        yield from _iter_adjacent_named_children(child_module, child_prefix)


def find_conv_bn_pairs(model: nn.Module) -> List[Tuple[str, str]]:
    """
    Find all Conv-BN pairs in the model.

    This function identifies consecutive Conv and BatchNorm layers that
    can be fused together. It matches:
    - Conv1d + BatchNorm1d
    - Conv2d + BatchNorm2d
    - ConvTranspose2d + BatchNorm2d
    - Linear + BatchNorm1d

    The function also validates that the Conv output channels match the
    BatchNorm num_features to ensure correct pairing.

    Args:
        model: PyTorch model

    Returns:
        List of (conv_name, bn_name) tuples
    """
    pairs = []

    # Mapping of conv types to their expected BN types
    conv_to_bn = {
        nn.Conv1d: nn.BatchNorm1d,
        nn.Conv2d: nn.BatchNorm2d,
        nn.ConvTranspose2d: nn.BatchNorm2d,
        nn.Linear: nn.BatchNorm1d,
    }

    for left_name, left_module, right_name, right_module in _iter_adjacent_named_children(model):
        for conv_type, bn_type in conv_to_bn.items():
            if isinstance(left_module, conv_type) and isinstance(right_module, bn_type):
                # Validate that channel dimensions match
                conv_out_channels = _get_conv_out_channels(left_module)
                bn_num_features = _get_bn_num_features(right_module)
                if conv_out_channels == bn_num_features:
                    pairs.append((left_name, right_name))
                break

    return pairs


def find_bn_conv_pairs(model: nn.Module) -> List[Tuple[str, str]]:
    """
    Find all BN-Conv pairs in the model (BN **followed by** Conv).

    This function identifies consecutive BatchNorm and Conv layers where the
    BatchNorm *precedes* the convolution. It matches:
    - BatchNorm1d + Conv1d
    - BatchNorm1d + Linear
    - BatchNorm2d + Conv2d
    - BatchNorm2d + ConvTranspose2d

    The function validates that the BN num_features matches the Conv
    in_channels to ensure correct pairing.

    Args:
        model: PyTorch model

    Returns:
        List of (bn_name, conv_name) tuples
    """
    pairs = []

    # Mapping of BN types to their acceptable following conv types
    bn_to_conv = {
        nn.BatchNorm1d: (nn.Conv1d, nn.Linear),
        nn.BatchNorm2d: (nn.Conv2d, nn.ConvTranspose2d),
    }

    for left_name, left_module, right_name, right_module in _iter_adjacent_named_children(model):
        for bn_type, conv_types in bn_to_conv.items():
            if isinstance(left_module, bn_type) and isinstance(right_module, conv_types):
                # Validate that channel dimensions match
                bn_num_features = _get_bn_num_features(left_module)
                conv_in_channels = _get_conv_in_channels(right_module)
                if bn_num_features == conv_in_channels:
                    pairs.append((left_name, right_name))
                break

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


def _replace_bn_with_identity(model: nn.Module, bn_name: str):
    """Replace a BatchNorm module with ``nn.Identity`` by name."""
    parent, attr = _get_parent_module(model, bn_name)
    if attr.isdigit():
        parent[int(attr)] = nn.Identity()
    else:
        setattr(parent, attr, nn.Identity())


def fuse_model_bn(model: nn.Module, inplace: bool = True, mode: str = "new") -> nn.Module:
    """
    Fuse BatchNorm into surrounding Conv/Linear layers.

    Modes:
        - ``mode='new'`` (default): Fuse Conv→BN and BN→Conv pairs.
        - ``mode='old'``: Fuse only Conv→BN pairs (legacy behavior).

    Args:
        model: PyTorch model
        inplace: If True, modify model in-place. If False, return a copy.
        mode: Fusion mode, one of ``'new'`` or ``'old'``.

    Returns:
        Model with fused BatchNorm layers.

    Example:
        >>> model.eval()
        >>> fuse_model_bn(model)
        >>> # Now all fused BN layers are replaced with Identity
    """
    mode = mode.lower()
    if mode not in {"new", "old"}:
        raise ValueError(f"Unsupported fuse mode: {mode}. Expected 'new' or 'old'.")

    if mode == "old":
        return fuse_model_bn_old(model, inplace=inplace)

    if not inplace:
        import copy

        model = copy.deepcopy(model)

    # Must be in eval mode for fusion
    model.eval()

    # ------------------------------------------------------------------
    # 1. Conv → BN pairs (standard pattern)
    # ------------------------------------------------------------------
    conv_bn_pairs = find_conv_bn_pairs(model)

    # Collect BN names already claimed by Conv → BN so that we don't
    # accidentally fuse the same BN a second time as BN → Conv.
    claimed_bn_names = {bn_name for _, bn_name in conv_bn_pairs}

    # ------------------------------------------------------------------
    # 2. BN → Conv pairs (reverse pattern, e.g. downsample: BN → Conv)
    # ------------------------------------------------------------------
    bn_conv_pairs_all = find_bn_conv_pairs(model)
    bn_conv_pairs = [
        (bn_name, conv_name) for bn_name, conv_name in bn_conv_pairs_all if bn_name not in claimed_bn_names
    ]

    total = len(conv_bn_pairs) + len(bn_conv_pairs)
    if total == 0:
        print("No Conv-BN or BN-Conv pairs found to fuse")
        return model

    # Build modules dict for fast lookup
    modules_dict = dict(model.named_modules())

    # --- Fuse Conv → BN ---
    for conv_name, bn_name in conv_bn_pairs:
        conv = modules_dict[conv_name]
        bn = modules_dict[bn_name]
        fuse_conv_bn(conv, bn)
        _replace_bn_with_identity(model, bn_name)

    # --- Fuse BN → Conv ---
    for bn_name, conv_name in bn_conv_pairs:
        bn = modules_dict[bn_name]
        conv = modules_dict[conv_name]
        fuse_bn_conv(bn, conv)
        _replace_bn_with_identity(model, bn_name)

    print(f"Fused {len(conv_bn_pairs)} Conv-BN pairs and {len(bn_conv_pairs)} BN-Conv pairs")
    return model


def fuse_model_bn_old(model: nn.Module, inplace: bool = True) -> nn.Module:
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
