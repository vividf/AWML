# Copyright (c) OpenMMLab. All rights reserved.
"""Quantization-aware nn.Module subclasses (Conv2d / ConvTranspose2d / Linear)."""

from .quant_conv import QuantConv2d, QuantConvTranspose2d
from .quant_linear import QuantLinear

__all__ = [
    "QuantConv2d",
    "QuantConvTranspose2d",
    "QuantLinear",
]
