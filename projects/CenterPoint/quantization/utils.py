# Copyright (c) OpenMMLab. All rights reserved.
"""Utility functions for quantization."""

from typing import List

import torch.nn as nn

try:
    from pytorch_quantization.nn import TensorQuantizer

    PYTORCH_QUANTIZATION_AVAILABLE = True
except ImportError:
    PYTORCH_QUANTIZATION_AVAILABLE = False
    TensorQuantizer = None


def _check_pytorch_quantization():
    """Check if pytorch-quantization is available."""
    if not PYTORCH_QUANTIZATION_AVAILABLE:
        raise ImportError(
            "pytorch-quantization is required for quantization support. "
            "Install it with: pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com"
        )


# Backward-compatible alias expected by some modules (e.g., deploy/export paths)
def check_pytorch_quantization():
    return _check_pytorch_quantization()


class disable_quantization:
    """
    Context manager / callable to disable quantization for specific modules.

    This class can be used as a context manager or by calling apply() directly
    to disable quantization for specific layers that are sensitive to quantization.

    Example:
        >>> # As context manager
        >>> with disable_quantization(model):
        ...     output = model(input)

        >>> # As callable
        >>> disable_quantization(model.backbone.conv1).apply()

        >>> # Re-enable
        >>> disable_quantization(model.backbone.conv1).apply(disabled=False)
    """

    def __init__(self, model: nn.Module):
        """
        Initialize with model to disable quantization for.

        Args:
            model: PyTorch model or submodule
        """
        _check_pytorch_quantization()
        self.model = model

    def apply(self, disabled: bool = True):
        """
        Apply disable/enable to all TensorQuantizers in the model.

        Args:
            disabled: If True, disable quantization. If False, enable.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        """Enter context: disable quantization."""
        self.apply(True)
        return self

    def __exit__(self, *args, **kwargs):
        """Exit context: re-enable quantization."""
        self.apply(False)


class enable_quantization:
    """
    Context manager / callable to enable quantization for specific modules.

    This is the inverse of disable_quantization.

    Example:
        >>> enable_quantization(model.backbone).apply()
    """

    def __init__(self, model: nn.Module):
        """
        Initialize with model to enable quantization for.

        Args:
            model: PyTorch model or submodule
        """
        _check_pytorch_quantization()
        self.model = model

    def apply(self, enabled: bool = True):
        """
        Apply enable/disable to all TensorQuantizers in the model.

        Args:
            enabled: If True, enable quantization. If False, disable.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, TensorQuantizer):
                module._disabled = not enabled

    def __enter__(self):
        """Enter context: enable quantization."""
        self.apply(True)
        return self

    def __exit__(self, *args, **kwargs):
        """Exit context: disable quantization."""
        self.apply(False)


def print_quantizer_status(model: nn.Module):
    """
    Print the status of all TensorQuantizers in the model.

    This is useful for debugging to see which layers have quantization
    enabled or disabled.

    Args:
        model: PyTorch model

    Example:
        >>> print_quantizer_status(model)
        TensorQuantizer name: backbone.conv1._input_quantizer, disabled: False
        TensorQuantizer name: backbone.conv1._weight_quantizer, disabled: False
        ...
    """
    _check_pytorch_quantization()

    print("=" * 80)
    print("Quantizer Status")
    print("=" * 80)

    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            status = "DISABLED" if module._disabled else "ENABLED"
            print(f"  {name}: {status}")
            if hasattr(module, "_amax") and module._amax is not None:
                amax = module._amax
                if amax.numel() == 1:
                    # Scalar amax (per-tensor quantization)
                    print(f"    amax: {amax.item():.6f}")
                else:
                    # Multi-element amax (per-channel quantization)
                    print(
                        f"    amax: [{amax.numel()} elements] min={amax.min().item():.6f}, max={amax.max().item():.6f}"
                    )

    print("=" * 80)


def get_quantizer_names(model: nn.Module) -> List[str]:
    """
    Get names of all TensorQuantizers in the model.

    Args:
        model: PyTorch model

    Returns:
        List of quantizer names
    """
    _check_pytorch_quantization()

    names = []
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            names.append(name)
    return names


def count_quantizers(model: nn.Module) -> dict:
    """
    Count enabled and disabled quantizers in the model.

    Args:
        model: PyTorch model

    Returns:
        Dict with 'enabled', 'disabled', and 'total' counts
    """
    _check_pytorch_quantization()

    enabled = 0
    disabled = 0

    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            if module._disabled:
                disabled += 1
            else:
                enabled += 1

    return {"enabled": enabled, "disabled": disabled, "total": enabled + disabled}


def set_quantizer_by_name(model: nn.Module, name_pattern: str, disabled: bool = True):
    """
    Enable/disable quantizers matching a name pattern.

    Args:
        model: PyTorch model
        name_pattern: Pattern to match in quantizer names (substring match)
        disabled: If True, disable matching quantizers. If False, enable.

    Example:
        >>> # Disable all input quantizers
        >>> set_quantizer_by_name(model, '_input_quantizer', disabled=True)

        >>> # Enable backbone quantizers
        >>> set_quantizer_by_name(model, 'backbone', disabled=False)
    """
    _check_pytorch_quantization()

    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and name_pattern in name:
            module._disabled = disabled
