# Copyright (c) OpenMMLab. All rights reserved.
"""Quantized Conv2d and ConvTranspose2d modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from deployment.quantization.core import backend as quant_backend
from deployment.quantization.core.availability import require_quant_backend

QUANT_BACKEND_AVAILABLE = quant_backend.available()
from deployment.quantization.core.descriptors import (
    conv2d_weight_desc,
    conv_transpose2d_weight_desc,
    default_input_desc,
)


def _check_quant_backend():
    """Check that a quantization backend is available (raises ImportError if not)."""
    require_quant_backend(QUANT_BACKEND_AVAILABLE)


def _skip_fake_quant_for_export_trace() -> bool:
    """Whether to bypass TensorQuantizer during JIT trace / ONNX export.

    modelopt's TensorQuantizer traces to Q/DQ ops natively, so the guard must never skip —
    bypassing quantizers during tracing would produce FP32-only ONNX (no
    QuantizeLinear/DequantizeLinear). Skipping remains only for the backend-less case, where a
    trace with legacy fake-quant modules would break.
    """
    if quant_backend.exports_qdq_natively():
        return False

    if torch.jit.is_tracing():
        return True
    try:
        is_onnx = getattr(torch.onnx, "is_in_onnx_export", None)
        if callable(is_onnx) and is_onnx():
            return True
    except Exception:
        pass
    return False


class QuantConv2d(nn.Conv2d):
    """
    Quantized Conv2d with per-channel weight quantization.

    This module extends nn.Conv2d with input and weight quantizers from
    NVIDIA's modelopt library. During forward pass, both input
    activations and weights are quantized using fake quantization (Q/DQ nodes).

    Args:
        Same as nn.Conv2d

    Attributes:
        _input_quantizer: TensorQuantizer for input activations
        _weight_quantizer: TensorQuantizer for weights
    """

    # Default quantization descriptors
    default_quant_desc_input = None
    default_quant_desc_weight = None

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        _check_quant_backend()
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        # Set default quantization descriptors (single source: core.descriptors)
        if QuantConv2d.default_quant_desc_input is None:
            QuantConv2d.default_quant_desc_input = default_input_desc()
        if QuantConv2d.default_quant_desc_weight is None:
            QuantConv2d.default_quant_desc_weight = conv2d_weight_desc()

        self._input_quantizer = None
        self._weight_quantizer = None

    def init_quantizer(self, quant_desc_input=None, quant_desc_weight=None):
        """Initialize input and weight quantizers."""
        _check_quant_backend()
        TensorQuantizer = quant_backend.get_tensor_quantizer_cls()

        quant_desc_input = quant_desc_input or self.default_quant_desc_input
        quant_desc_weight = quant_desc_weight or self.default_quant_desc_weight

        self._input_quantizer = TensorQuantizer(quant_desc_input)
        self._weight_quantizer = TensorQuantizer(quant_desc_weight)

    def forward(self, x):
        """Forward with quantized input and weights."""
        if self._input_quantizer is not None and self._weight_quantizer is not None:
            if _skip_fake_quant_for_export_trace():
                quant_input = x
                quant_weight = self.weight
            else:
                quant_input = self._input_quantizer(x)
                quant_weight = self._weight_quantizer(self.weight)
        else:
            quant_input = x
            quant_weight = self.weight

        return self._conv_forward(quant_input, quant_weight, self.bias)


class QuantConvTranspose2d(nn.ConvTranspose2d):
    """
    Quantized ConvTranspose2d with per-tensor weight quantization.

    This module extends nn.ConvTranspose2d for FPN upsample layers with
    input and weight quantizers..

    Args:
        Same as nn.ConvTranspose2d

    Attributes:
        _input_quantizer: TensorQuantizer for input activations
        _weight_quantizer: TensorQuantizer for weights
    """

    # Default quantization descriptors
    default_quant_desc_input = None
    default_quant_desc_weight = None

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        _check_quant_backend()
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        # Set default quantization descriptors (single source: core.descriptors)
        if QuantConvTranspose2d.default_quant_desc_input is None:
            QuantConvTranspose2d.default_quant_desc_input = default_input_desc()
        if QuantConvTranspose2d.default_quant_desc_weight is None:
            QuantConvTranspose2d.default_quant_desc_weight = conv_transpose2d_weight_desc()

        self._input_quantizer = None
        self._weight_quantizer = None

    def init_quantizer(self, quant_desc_input=None, quant_desc_weight=None):
        """Initialize input and weight quantizers."""
        _check_quant_backend()
        TensorQuantizer = quant_backend.get_tensor_quantizer_cls()

        quant_desc_input = quant_desc_input or self.default_quant_desc_input
        quant_desc_weight = quant_desc_weight or self.default_quant_desc_weight

        self._input_quantizer = TensorQuantizer(quant_desc_input)
        self._weight_quantizer = TensorQuantizer(quant_desc_weight)

    def forward(self, x, output_size=None):
        """Forward with quantized input and weights."""
        if self._input_quantizer is not None and self._weight_quantizer is not None:
            if _skip_fake_quant_for_export_trace():
                quant_input = x
                quant_weight = self.weight
            else:
                quant_input = self._input_quantizer(x)
                quant_weight = self._weight_quantizer(self.weight)
        else:
            quant_input = x
            quant_weight = self.weight

        # Compute output padding
        if output_size is None:
            output_padding = self.output_padding
        else:
            output_padding = self._output_padding(
                quant_input,
                output_size,
                self.stride,
                self.padding,
                self.kernel_size,
                num_spatial_dims=2,
                dilation=self.dilation,
            )

        return F.conv_transpose2d(
            quant_input,
            quant_weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
