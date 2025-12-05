# Copyright (c) OpenMMLab. All rights reserved.
"""Quantized Add module for skip connections."""

import torch
import torch.nn as nn

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


class QuantAdd(nn.Module):
    """
    Quantized addition for skip connections.

    This module quantizes both inputs to element-wise addition using the same
    quantizer, ensuring they share the same scale. This is important for
    residual connections where the main path and skip path are added together.

    Attributes:
        _input_quantizer: TensorQuantizer for both inputs
    """

    # Default quantization descriptor
    default_quant_desc_input = None

    def __init__(self):
        _check_pytorch_quantization()
        super().__init__()

        # Set default quantization descriptor
        if QuantAdd.default_quant_desc_input is None:
            QuantAdd.default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

        self._input_quantizer = None

    def init_quantizer(self, quant_desc_input=None):
        """Initialize input quantizer."""
        _check_pytorch_quantization()
        from pytorch_quantization.nn import TensorQuantizer

        quant_desc_input = quant_desc_input or self.default_quant_desc_input
        self._input_quantizer = TensorQuantizer(quant_desc_input)

    def forward(self, x1, x2):
        """
        Forward with quantized inputs.

        Args:
            x1: First input tensor
            x2: Second input tensor (typically skip connection)

        Returns:
            Sum of quantized inputs
        """
        if self._input_quantizer is not None:
            quant_x1 = self._input_quantizer(x1)
            quant_x2 = self._input_quantizer(x2)
        else:
            quant_x1 = x1
            quant_x2 = x2

        return torch.add(quant_x1, quant_x2)
