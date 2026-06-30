"""
Quantized add module.

Provides a small wrapper that quantizes both inputs with the same TensorQuantizer
before performing elementwise addition. This mirrors CUDA-CenterPoint's QuantAdd.

The QuantAdd module inherits from QuantInputMixin to ensure proper ONNX export
and TensorRT fusion support. Both inputs share the same quantizer to align scales.
"""

import torch
import torch.nn as nn

from ..utils import check_pytorch_quantization

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
            "pytorch-quantization is required for QuantAdd. "
            "Install it with: pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com"
        )


# Create QuantAdd class with proper inheritance
if PYTORCH_QUANTIZATION_AVAILABLE:

    class QuantAdd(nn.Module, _utils.QuantInputMixin):
        """
        Quantized add with shared input quantizer.

        This module quantizes both inputs with the same TensorQuantizer before
        performing elementwise addition. This ensures both inputs have the same
        quantization scale, which is critical for TensorRT fusion.

        Inherits from QuantInputMixin to ensure proper ONNX export support.
        """

        # Default quantization descriptor: 8-bit per-tensor (same scale for both inputs)
        default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

        def __init__(self, quant_desc_input=None):
            super().__init__()
            quant_desc_input = quant_desc_input or QuantAdd.default_quant_desc_input
            # Use QuantInputMixin's init_quantizer method for proper initialization
            self.init_quantizer(quant_desc_input)

        def forward(self, x, y):
            """
            Forward pass with quantized inputs.

            Both inputs are quantized using the same quantizer to ensure
            they have the same scale, enabling TensorRT to fuse the add operation.
            """
            quant_input1 = self._input_quantizer(x)
            quant_input2 = self._input_quantizer(y)
            return torch.add(quant_input1, quant_input2)

else:
    # Fallback class when pytorch-quantization is not available
    class QuantAdd(nn.Module):
        """
        Quantized add with shared input quantizer (fallback implementation).

        This is a fallback implementation that should not be used in practice.
        pytorch-quantization must be installed for proper functionality.
        """

        def __init__(self, quant_desc_input=None):
            _check_pytorch_quantization()  # Will raise ImportError
            super().__init__()

        def forward(self, x, y):
            """Forward pass (should not be called without pytorch-quantization)."""
            raise RuntimeError("QuantAdd requires pytorch-quantization to be installed")
