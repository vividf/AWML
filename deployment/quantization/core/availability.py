# Copyright (c) OpenMMLab. All rights reserved.
"""Shared quantization-backend availability guard.

Leaf module: it imports nothing from ``deployment.quantization`` except the (equally leaf)
:mod:`.backend` resolver, so every quantization submodule can import it without creating an import
cycle. Centralizes the install hint that was previously copy-pasted into ~6 modules.
"""

from __future__ import annotations

from . import backend as _backend


def quant_backend_install_hint(purpose: str = "quantization support") -> str:
    """Return the standard 'nvidia-modelopt is required ...' install message.

    Single source of the install hint. Use directly in
    ``except ImportError: raise ImportError(...)`` reraise sites.
    """
    return _backend.install_hint(purpose)


def require_quant_backend(available: bool, purpose: str = "quantization support") -> None:
    """Raise ImportError with an install hint when the quantization backend is unavailable.

    Args:
        available: The module-local availability flag (typically ``backend.available()``).
        purpose: Short phrase describing what needs it (e.g. ``"sensitivity analysis"``).
    """
    if not available:
        raise ImportError(quant_backend_install_hint(purpose))


def require_quant_backend_installed(purpose: str = "quantization support") -> None:
    """Raise a descriptive ImportError if the quantization backend cannot be imported.

    Availability probe for functions that will build ``TensorQuantizer``s, so a missing dependency
    surfaces the install hint for *that* feature instead of a generic error deep in a helper.
    """
    _backend.require(purpose)
