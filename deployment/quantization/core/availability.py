# Copyright (c) OpenMMLab. All rights reserved.
"""Shared quantization-backend availability guard.

Leaf module: it imports nothing from ``deployment.quantization`` except the (equally leaf)
:mod:`.backend` resolver, so every quantization submodule can import it without creating an import
cycle. Centralizes the install hint that was previously copy-pasted into ~6 modules.

The historical function names say ``pytorch_quantization`` because that used to be the only
backend; they are kept (many call sites, including other projects) but now accept either backend
— pytorch-quantization or nvidia-modelopt — via :mod:`.backend`.
"""

from __future__ import annotations

from . import backend as _backend


def pytorch_quantization_install_hint(purpose: str = "quantization support") -> str:
    """Return the standard 'a quantization backend is required ...' install message.

    Single source of the install URLs (was copy-pasted across ~11 sites). Use directly in
    ``except ImportError: raise ImportError(...)`` reraise sites.
    """
    return _backend.install_hint(purpose)


def require_pytorch_quantization(available: bool, purpose: str = "quantization support") -> None:
    """Raise ImportError with an install hint when no quantization backend is available.

    Args:
        available: The module-local availability flag (now typically ``backend.available()``).
        purpose: Short phrase describing what needs it (e.g. ``"sensitivity analysis"``).
    """
    if not available:
        raise ImportError(pytorch_quantization_install_hint(purpose))


def require_pytorch_quantization_installed(purpose: str = "quantization support") -> None:
    """Raise a descriptive ImportError if no quantization backend can be imported.

    Availability probe for functions that will build ``TensorQuantizer``s, so a missing dependency
    surfaces the install hint for *that* feature instead of a generic error deep in a helper.
    """
    _backend.require(purpose)
