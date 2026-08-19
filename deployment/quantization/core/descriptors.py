# Copyright (c) OpenMMLab. All rights reserved.
"""Default INT8 quantization descriptors — single source of the descriptor choices.

The quantized modules (:mod:`.modules`) and the module-replacement engine (:mod:`.replace`) both
need to know *which* descriptor each layer type uses. Historically those choices (per-tensor vs
per-channel weights, histogram activations) were written out as literals in both places. This leaf
module defines them once so the ``__init__`` path and the ``ensure_quant_descriptors_initialized``
path can never disagree.

Backend-agnostic: descriptors are built through :mod:`.backend`, which translates to the active
library's dialect (pytorch-quantization ``QuantDescriptor`` or modelopt
``QuantizerAttributeConfig``). The ``QUANT_DESC_*`` presets exist under the same names in both.

Leaf module: it imports only :mod:`.backend` (which imports the library lazily), so every core
submodule can import it without a cycle.
"""

from __future__ import annotations

from typing import Any

from . import backend as _backend


def default_input_desc() -> Any:
    """Per-tensor histogram INT8 activation descriptor (shared by Conv2d / ConvTranspose2d / Linear)."""
    return _backend.make_quant_desc(num_bits=8, calib_method="histogram")


def conv2d_weight_desc() -> Any:
    """Per-output-channel INT8 weight descriptor for Conv2d."""
    return _backend.get_preset_desc("QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL")


def conv_transpose2d_weight_desc() -> Any:
    """Per-tensor INT8 weight descriptor for ConvTranspose2d.

    TensorRT INT8 transposed conv is fragile with per-channel weight scales (it can fail the engine
    build with ``vol == 1`` / ``Could not find any implementation``), so ConvTranspose2d weights are
    quantized per-tensor.
    """
    return _backend.get_preset_desc("QUANT_DESC_8BIT_PER_TENSOR")


def linear_weight_desc() -> Any:
    """Per-output-channel (per-row) INT8 weight descriptor for Linear."""
    return _backend.make_quant_desc(num_bits=8, axis=(0,))
