# Copyright (c) OpenMMLab. All rights reserved.
"""
Quantization framework (model-agnostic).

PTQ / QAT building blocks based on NVIDIA's modelopt toolkit, organized in layers:

- :mod:`~deployment.quantization.core`    — model-agnostic engine (quant modules, BN fusion,
  calibration, the Q/DQ module-replacement engine, sensitivity, utils, ONNX symbolic).
- :mod:`~deployment.quantization.recipes` — architecture-specific Q/DQ placement (forward hooks +
  ``attach_*`` for ResNet / VoVNet / ConvNeXt residual blocks).
- :mod:`~deployment.quantization.schemes` — the seam between deployment stages and quantization
  (``QuantizationScheme`` / ``QuantizationPlan`` + generic ``DenseQDQScheme``).
- :mod:`~deployment.quantization.sparse`  — spconv sparse-encoder helpers (FP16 deploy —
  SparseConv+BN fold only; the spconv INT8 subsystem was removed).

Model-specific composition (e.g. CenterPoint's ``quant_model`` / PTQ pipeline) lives in the
project bundle, e.g. :mod:`deployment.projects.centerpoint.quantization`.

The names exported here are the package's real external API — everything a project bundle or
deploy loader imports. Deeper internals (quant module classes, per-layer rebuild helpers, single
Conv-BN fold) stay importable from their defining ``core.*`` modules but are deliberately not
re-exported.
"""

from .core.calibration import CalibrationManager
from .core.fusion import fuse_model_bn
from .core.replace import expand_keep_fp16, quant_conv_module, quant_linear_module
from .core.utils import (
    disable_quantization,
    disable_quantizers_in,
    get_tensor_quantizer_cls,
    move_quantizer_amax_to_device,
    print_quantizer_status,
    setup_quantization_for_onnx_export,
)
from .schemes import DenseQDQScheme, QuantizationPlan, QuantizationScheme

__all__ = [
    # Replace / placement
    "quant_conv_module",
    "quant_linear_module",
    "expand_keep_fp16",
    # Schemes (uniform seam between deployment stages and quantization)
    "QuantizationScheme",
    "QuantizationPlan",
    "DenseQDQScheme",
    # Calibration
    "CalibrationManager",
    # Fusion
    "fuse_model_bn",
    # Utils
    "disable_quantization",
    "disable_quantizers_in",
    "get_tensor_quantizer_cls",
    "move_quantizer_amax_to_device",
    "print_quantizer_status",
    "setup_quantization_for_onnx_export",
]
