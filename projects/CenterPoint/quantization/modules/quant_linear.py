# Copyright (c) OpenMMLab. All rights reserved.
"""Quantized Linear module for PillarFeatureNet."""

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


class QuantLinear(nn.Linear):
    """
    Quantized Linear module for PillarFeatureNet.

    This module extends nn.Linear with input and weight quantizers from
    NVIDIA's pytorch-quantization library. Used in PFNLayer of the
    pillar feature encoder.

    Args:
        Same as nn.Linear

    Attributes:
        _input_quantizer: TensorQuantizer for input activations
        _weight_quantizer: TensorQuantizer for weights
    """

    # Default quantization descriptors
    default_quant_desc_input = None
    default_quant_desc_weight = None

    def __init__(self, in_features, out_features, bias=True, **kwargs):
        _check_pytorch_quantization()
        super().__init__(in_features, out_features, bias, **kwargs)

        # Set default quantization descriptors
        if QuantLinear.default_quant_desc_input is None:
            QuantLinear.default_quant_desc_input = tensor_quant.QuantDescriptor(num_bits=8, calib_method="histogram")
        if QuantLinear.default_quant_desc_weight is None:
            # Per-row quantization for Linear layers
            QuantLinear.default_quant_desc_weight = tensor_quant.QuantDescriptor(
                num_bits=8, axis=(0,)  # Per output channel
            )

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

        return F.linear(quant_input, quant_weight, self.bias)
