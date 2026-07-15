# Copyright (c) OpenMMLab. All rights reserved.
"""Sparse-convolution (spconv) INT8 quantization primitives — model-agnostic.

Reusable by any SECOND-style spconv encoder (BEVFusion, CenterPoint, …). Projects compose these
via a ``QuantizationScheme`` (e.g. BEVFusion's ``SpconvInt8Scheme``); the ONNX ``ImplicitGemmInt8``
graph transform and TensorRT plugin remain in the project that owns that deployment path.
"""

from .naming import tail_without_encoder_layers, topologically_sorted_sparse_stems
from .spconv_add_patch import ensure_spconv_quantize_per_tensor_float_activations
from .spconv_int8 import apply_nvidia_spconv_int8, calibrate_spconv_nvidia, fuse_spconv_bn_in_encoder

__all__ = [
    "apply_nvidia_spconv_int8",
    "calibrate_spconv_nvidia",
    "fuse_spconv_bn_in_encoder",
    "ensure_spconv_quantize_per_tensor_float_activations",
    "tail_without_encoder_layers",
    "topologically_sorted_sparse_stems",
]
