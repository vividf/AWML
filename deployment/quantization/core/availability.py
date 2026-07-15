# Copyright (c) OpenMMLab. All rights reserved.
"""Shared pytorch-quantization availability guard.

Leaf module: it imports nothing from ``deployment.quantization``, so every quantization
submodule can import it without creating an import cycle. Centralizes the install hint
(notably the ``--extra-index-url`` URL) that was previously copy-pasted into ~6 modules.
"""

from __future__ import annotations


def pytorch_quantization_install_hint(purpose: str = "quantization support") -> str:
    """Return the standard 'pytorch-quantization is required ...' install message.

    Single source of the ``--extra-index-url`` install URL (was copy-pasted across ~11 sites).
    Use directly in ``except ImportError: raise ImportError(...)`` reraise sites.
    """
    return (
        f"pytorch-quantization is required for {purpose}. "
        "Install it with: pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com"
    )


def require_pytorch_quantization(available: bool, purpose: str = "quantization support") -> None:
    """Raise ImportError with an install hint when pytorch-quantization is unavailable.

    Args:
        available: The module-local ``PYTORCH_QUANTIZATION_AVAILABLE`` flag.
        purpose: Short phrase describing what needs it (e.g. ``"sensitivity analysis"``).
    """
    if not available:
        raise ImportError(pytorch_quantization_install_hint(purpose))


def require_pytorch_quantization_installed(purpose: str = "quantization support") -> None:
    """Raise a descriptive ImportError if ``pytorch_quantization`` cannot be imported.

    Availability probe (no import side effect) for functions that will build ``TensorQuantizer``s,
    so a missing dependency surfaces the install hint for *that* feature instead of a generic error
    deep in a helper.
    """
    import importlib.util

    if importlib.util.find_spec("pytorch_quantization") is None:
        raise ImportError(pytorch_quantization_install_hint(purpose))
