# Copyright (c) OpenMMLab. All rights reserved.
"""
CenterPoint Quantization Module.

This module provides PTQ (Post-Training Quantization) and QAT (Quantization-Aware Training)
support for CenterPoint models, based on NVIDIA's pytorch-quantization toolkit.

Usage:
    from projects.CenterPoint.quantization import (
        quant_conv_module,
        quant_linear_module,
        quant_model,
        CalibrationManager,
        fuse_model_bn,
        quantize_ptq,
    )

Example PTQ workflow:
    >>> from projects.CenterPoint.quantization import quantize_ptq
    >>> model = init_model(cfg, checkpoint)
    >>> model.eval()
    >>> quantized_model = quantize_ptq(model, val_dataloader, num_calibration_batches=100)
    >>> torch.save({'state_dict': quantized_model.state_dict()}, 'ptq.pth')
"""

from .calibration import CalibrationManager
from .fusion import fuse_conv_bn, fuse_model_bn
from .hooks import QATHook
from .modules import QuantAdd, QuantConv2d, QuantConvTranspose2d, QuantLinear
from .ptq import load_ptq_model, quantize_ptq, save_ptq_model
from .replace import quant_conv_module, quant_linear_module, quant_model, transfer_to_quantization
from .sensitivity import build_sensitivity_profile, get_sensitive_layers
from .utils import disable_quantization, enable_quantization, print_quantizer_status

__all__ = [
    # Modules
    "QuantConv2d",
    "QuantConvTranspose2d",
    "QuantLinear",
    "QuantAdd",
    # Replace functions
    "quant_conv_module",
    "quant_linear_module",
    "quant_model",
    "transfer_to_quantization",
    # Calibration
    "CalibrationManager",
    # Fusion
    "fuse_conv_bn",
    "fuse_model_bn",
    # PTQ
    "quantize_ptq",
    "save_ptq_model",
    "load_ptq_model",
    # Sensitivity
    "build_sensitivity_profile",
    "get_sensitive_layers",
    # Hooks
    "QATHook",
    # Utils
    "disable_quantization",
    "enable_quantization",
    "print_quantizer_status",
]
