"""
Quantized add module.

Provides a small wrapper that quantizes both inputs with the same TensorQuantizer
before performing elementwise addition. This mirrors CUDA-CenterPoint's QuantAdd.
"""

import torch
import torch.nn as nn

from ..utils import check_pytorch_quantization


class QuantAdd(nn.Module):
    """Quantized add with shared input quantizer."""

    def __init__(self, quant_desc_input=None):
        super().__init__()
        check_pytorch_quantization()
        try:
            from pytorch_quantization import tensor_quant
            from pytorch_quantization.nn import TensorQuantizer
        except ImportError as e:
            raise ImportError(
                "pytorch-quantization is required for QuantAdd. "
                "Install it with: pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com"
            ) from e

        # Default activation quant descriptor: 8-bit histogram (same as other activations)
        quant_desc_input = quant_desc_input or tensor_quant.QuantDescriptor(num_bits=8, calib_method="histogram")
        self._input_quantizer = TensorQuantizer(quant_desc_input)

    def forward(self, x, y):
        qx = self._input_quantizer(x)
        qy = self._input_quantizer(y)
        return torch.add(qx, qy)
