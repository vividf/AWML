# Copyright (c) OpenMMLab. All rights reserved.
"""Utility functions for quantization."""

import logging
from typing import Iterable, Optional, Type, Union

import torch
import torch.nn as nn

try:
    from pytorch_quantization.nn import TensorQuantizer

    PYTORCH_QUANTIZATION_AVAILABLE = True
except ImportError:
    PYTORCH_QUANTIZATION_AVAILABLE = False
    TensorQuantizer = None

from deployment.quantization.core.availability import require_pytorch_quantization

logger = logging.getLogger(__name__)


def _check_pytorch_quantization():
    """Check if pytorch-quantization is available (raises ImportError if not)."""
    require_pytorch_quantization(PYTORCH_QUANTIZATION_AVAILABLE)


def get_tensor_quantizer_cls() -> Optional[Type]:
    """Return the ``TensorQuantizer`` class, or ``None`` when pytorch_quantization is not installed.

    Also re-asserts the deployment logging configuration: importing pytorch_quantization pulls in
    ``absl.logging``, which hijacks the root logger (installs its own handler, only WARNING+ reaches
    stderr) — silently swallowing every later log record of ONNX/TensorRT export and evaluation.
    ``restore_deployment_logging`` re-asserts the canonical config captured by the CLI at
    ``setup_logging`` time; it is a no-op when the CLI did not configure logging (unit tests, QAT).
    The import is guarded so the engine keeps working without the CLI package.
    """
    try:
        from deployment.cli.args import restore_deployment_logging

        restore_deployment_logging()
    except Exception:
        pass
    return TensorQuantizer


def move_quantizer_amax_to_device(model: nn.Module, device: Union[str, torch.device]) -> int:
    """Move every ``TensorQuantizer._amax`` tensor to ``device`` (post checkpoint-load fixup).

    Shared by the CenterPoint and BEVFusion deploy loaders after ``load_state_dict``.

    Returns:
        Number of amax tensors moved.
    """
    tensor_quantizer_cls = get_tensor_quantizer_cls()
    if tensor_quantizer_cls is None:
        return 0
    device = torch.device(device)
    moved = 0
    for _name, module in model.named_modules():
        if isinstance(module, tensor_quantizer_cls):
            if getattr(module, "_amax", None) is not None and module._amax.device != device:
                module._amax = module._amax.to(device)
                moved += 1
    if moved:
        logger.info("Moved %d quantizer amax tensors to %s", moved, device)
    return moved


def setup_quantization_for_onnx_export() -> None:
    """Configure pytorch-quantization for proper ONNX export.

    Enables ``use_fb_fake_quant`` so ``TensorQuantizer`` exports as QuantizeLinear/DequantizeLinear
    ONNX ops (recognized by TensorRT). Global flag; must be set before ONNX export of a quantized
    model. No-op when pytorch_quantization is not installed.
    """
    tensor_quantizer_cls = get_tensor_quantizer_cls()
    if tensor_quantizer_cls is None:
        return
    tensor_quantizer_cls.use_fb_fake_quant = True
    logger.info("Enabled use_fb_fake_quant for ONNX export of quantized models")


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


def disable_quantizers_in(model: nn.Module, module_names: Iterable[str]) -> int:
    """Disable every ``TensorQuantizer`` inside the named modules — the ``keep_fp16`` disable loop.

    The single spelling of "turn the ``keep_fp16`` subtrees off after calibration / checkpoint load,"
    shared by the PTQ producers, the QAT hook, and the deploy loaders. ``module_names`` is the concrete
    set produced by :func:`~deployment.quantization.core.replace.expand_keep_fp16` (matched modules
    plus all descendants), so an exact ``named_modules()`` lookup per name is sufficient;
    :class:`disable_quantization` then recursively disables the quantizers under each hit.

    Args:
        model: Model whose quantizers to disable.
        module_names: Concrete dotted module names (typically from ``expand_keep_fp16``).

    Returns:
        Number of named modules found and disabled. Names not present in the model are logged as
        warnings and skipped (an expanded set can never miss, so a miss means stale input).
    """
    _check_pytorch_quantization()
    modules = dict(model.named_modules())
    count = 0
    for name in sorted(module_names):
        module = modules.get(name)
        if module is None:
            logger.warning("disable_quantizers_in: module not found, skipping: %s", name)
            continue
        disable_quantization(module).apply()
        count += 1
    if count:
        logger.info("Disabled quantizers in %d keep_fp16 module(s)", count)
    return count


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
