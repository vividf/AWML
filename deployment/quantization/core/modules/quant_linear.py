# Copyright (c) OpenMMLab. All rights reserved.
"""Quantized Linear module for PillarFeatureNet."""

import importlib.util

import torch.nn as nn
import torch.nn.functional as F

PYTORCH_QUANTIZATION_AVAILABLE = importlib.util.find_spec("pytorch_quantization") is not None

from deployment.quantization.core.availability import require_pytorch_quantization
from deployment.quantization.core.descriptors import default_input_desc, linear_weight_desc


def _check_pytorch_quantization():
    """Check if pytorch-quantization is available (raises ImportError if not)."""
    require_pytorch_quantization(PYTORCH_QUANTIZATION_AVAILABLE)


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

        # Set default quantization descriptors (single source: core.descriptors)
        if QuantLinear.default_quant_desc_input is None:
            QuantLinear.default_quant_desc_input = default_input_desc()
        if QuantLinear.default_quant_desc_weight is None:
            QuantLinear.default_quant_desc_weight = linear_weight_desc()

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
        """Forward with quantized input and weights.

        Unlike ``QuantConv2d`` / ``QuantConvTranspose2d``, there is no
        ``_skip_fake_quant_for_export_trace`` guard here: every shipped export path sets
        ``use_fb_fake_quant=True`` first (``setup_quantization_for_onnx_export``), which makes the
        conv guard a no-op anyway — and quantized Linears have exported fine without it.
        TODO(Docker): confirm tracing a quantized Linear *without* fb fake quant either works or
        should adopt the shared guard (spec.md §5.4 4D.3).
        """
        if self._input_quantizer is not None and self._weight_quantizer is not None:
            quant_input = self._input_quantizer(x)
            quant_weight = self._weight_quantizer(self.weight)
        else:
            quant_input = x
            quant_weight = self.weight

        return F.linear(quant_input, quant_weight, self.bias)
