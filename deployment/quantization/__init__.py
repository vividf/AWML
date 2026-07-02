# Copyright (c) OpenMMLab. All rights reserved.
"""
Quantization framework (model-agnostic).

PTQ / QAT building blocks based on NVIDIA's pytorch-quantization toolkit, organized in layers:

- :mod:`~deployment.quantization.core`    — model-agnostic engine (quant modules, BN fusion,
  calibration, the Q/DQ module-replacement engine, sensitivity, utils, ONNX symbolic).
- :mod:`~deployment.quantization.recipes` — architecture-specific Q/DQ placement (forward hooks +
  ``attach_*`` for ResNet / VoVNet / ConvNeXt residual blocks).
- :mod:`~deployment.quantization.schemes` — the seam between deployment stages and quantization
  (``QuantizationScheme`` / ``QuantizationPlan`` + generic ``DenseQDQScheme``).
- :mod:`~deployment.quantization.sparse`  — spconv INT8 subsystem.

Model-specific composition (e.g. CenterPoint's ``quant_model`` / PTQ pipeline) lives in the
project bundle, e.g. :mod:`deployment.projects.centerpoint.quantization`.

Usage:
    from deployment.quantization import (
        quant_conv_module,
        quant_linear_module,
        CalibrationManager,
        fuse_model_bn,
    )
"""

from .core.calibration import CalibrationManager
from .core.fusion import (
    fuse_conv_bn,
    fuse_model_bn,
)
from .core.modules import QuantConv2d, QuantConvTranspose2d, QuantLinear
from .core.replace import quant_conv_module, quant_linear_module, transfer_to_quantization
from .core.utils import disable_quantization, print_quantizer_status
from .schemes import DenseQDQScheme, QuantizationPlan, QuantizationScheme

__all__ = [
    # Modules
    "QuantConv2d",
    "QuantConvTranspose2d",
    "QuantLinear",
    # Replace functions
    "quant_conv_module",
    "quant_linear_module",
    "transfer_to_quantization",
    # Schemes (uniform seam between deployment stages and quantization)
    "QuantizationScheme",
    "QuantizationPlan",
    "DenseQDQScheme",
    # Calibration
    "CalibrationManager",
    # Fusion
    "fuse_conv_bn",
    "fuse_model_bn",
    # Utils
    "disable_quantization",
    "print_quantizer_status",
]
