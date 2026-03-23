"""BEVFusion quantization utilities.

Provides spconv INT8 quantization for sparse encoder and
pytorch_quantization integration for dense parts.
"""

from .spconv_int8 import (
    apply_spconv_int8_quantization,
    calibrate_spconv_model,
)

__all__ = [
    "apply_spconv_int8_quantization",
    "calibrate_spconv_model",
]
