# Copyright (c) OpenMMLab. All rights reserved.
"""Quantized Conv2d and ConvTranspose2d modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_quantization import tensor_quant
    from pytorch_quantization.nn.modules import _utils

    PYTORCH_QUANTIZATION_AVAILABLE = True
except ImportError:
    PYTORCH_QUANTIZATION_AVAILABLE = False
    _utils = None
    tensor_quant = None


def _check_pytorch_quantization():
    """Check if pytorch-quantization is available."""
    if not PYTORCH_QUANTIZATION_AVAILABLE:
        raise ImportError(
            "pytorch-quantization is required for quantization support. "
            "Install it with: pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com"
        )


class QuantConv2d(nn.Conv2d):
    """
    Quantized Conv2d with per-channel weight quantization.

    This module extends nn.Conv2d with input and weight quantizers from
    NVIDIA's pytorch-quantization library. During forward pass, both input
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
        _check_pytorch_quantization()
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        # Set default quantization descriptors
        if QuantConv2d.default_quant_desc_input is None:
            QuantConv2d.default_quant_desc_input = tensor_quant.QuantDescriptor(num_bits=8, calib_method="histogram")
        if QuantConv2d.default_quant_desc_weight is None:
            QuantConv2d.default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL

        self._input_quantizer = None
        self._weight_quantizer = None

    def init_quantizer(self, quant_desc_input=None, quant_desc_weight=None):
        """Initialize input and weight quantizers."""
        _check_pytorch_quantization()
        from pytorch_quantization.nn import TensorQuantizer

        quant_desc_input = quant_desc_input or self.default_quant_desc_input
        quant_desc_weight = quant_desc_weight or self.default_quant_desc_weight

        self._input_quantizer = TensorQuantizer(quant_desc_input)
        self._weight_quantizer = TensorQuantizer(quant_desc_weight)

    def forward(self, x):
        """Forward with quantized input and weights."""
        if self._input_quantizer is not None and self._weight_quantizer is not None:
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
        _check_pytorch_quantization()
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        # Set default quantization descriptors
        if QuantConvTranspose2d.default_quant_desc_input is None:
            QuantConvTranspose2d.default_quant_desc_input = tensor_quant.QuantDescriptor(
                num_bits=8, calib_method="histogram"
            )
        if QuantConvTranspose2d.default_quant_desc_weight is None:
            # Use per-tensor weight quantization for TensorRT compatibility.
            QuantConvTranspose2d.default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

        self._input_quantizer = None
        self._weight_quantizer = None

    def init_quantizer(self, quant_desc_input=None, quant_desc_weight=None):
        """Initialize input and weight quantizers."""
        _check_pytorch_quantization()
        from pytorch_quantization.nn import TensorQuantizer

        quant_desc_input = quant_desc_input or self.default_quant_desc_input
        quant_desc_weight = quant_desc_weight or self.default_quant_desc_weight

        self._input_quantizer = TensorQuantizer(quant_desc_input)
        self._weight_quantizer = TensorQuantizer(quant_desc_weight)

    def forward(self, x, output_size=None):
        """Forward with quantized input and weights."""
        if self._input_quantizer is not None and self._weight_quantizer is not None:
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
