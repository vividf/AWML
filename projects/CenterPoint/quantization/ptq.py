# Copyright (c) OpenMMLab. All rights reserved.
"""PTQ (Post-Training Quantization) pipeline for CenterPoint."""

from pathlib import Path
from typing import Any, Optional, Set

import torch
import torch.nn as nn

from .calibration import CalibrationManager
from .fusion import fuse_model_bn
from .replace import quant_model
from .utils import disable_quantization, print_quantizer_status


def quantize_ptq(
    model: nn.Module,
    dataloader: Any,
    num_calibration_batches: int = 100,
    amax_method: str = "mse",
    fuse_bn: bool = True,
    sensitive_layers: Optional[Set[str]] = None,
    forward_fn: Optional[callable] = None,
    verbose: bool = True,
) -> nn.Module:
    """
    Apply PTQ (Post-Training Quantization) to a CenterPoint model.

    This function performs the complete PTQ workflow:
    1. Fuse BatchNorm layers into convolutions (optional)
    2. Insert Q/DQ nodes by replacing modules with quantized versions
    3. Calibrate to determine optimal quantization scales
    4. Disable quantization for sensitive layers

    Args:
        model: CenterPoint model (should be in eval mode)
        dataloader: DataLoader for calibration data
        num_calibration_batches: Number of batches for calibration
        amax_method: Method for computing amax ('mse', 'entropy', 'percentile', 'max')
        fuse_bn: Whether to fuse BatchNorm before quantization
        sensitive_layers: Set of layer names to skip quantization
        forward_fn: Optional custom forward function for calibration
        verbose: Whether to print progress information

    Returns:
        Quantized model

    Example:
        >>> from projects.CenterPoint.quantization import quantize_ptq
        >>> model = init_model(cfg, checkpoint)
        >>> model.eval()
        >>> quantized_model = quantize_ptq(model, val_dataloader)
        >>> torch.save({'state_dict': quantized_model.state_dict()}, 'ptq.pth')
    """
    model.eval()
    sensitive_layers = sensitive_layers or set()

    # Step 1: Fuse BatchNorm
    if fuse_bn:
        if verbose:
            print("Step 1: Fusing BatchNorm layers...")
        fuse_model_bn(model)
    else:
        if verbose:
            print("Step 1: Skipping BatchNorm fusion")

    # Step 2: Insert Q/DQ nodes
    if verbose:
        print("Step 2: Inserting Q/DQ nodes...")
    quant_model(model, skip_names=sensitive_layers)

    # Step 3: Calibrate
    if verbose:
        print(f"Step 3: Calibrating with {num_calibration_batches} batches...")
    calibrator = CalibrationManager(model)
    calibrator.calibrate(
        dataloader,
        num_batches=num_calibration_batches,
        method=amax_method,
        forward_fn=forward_fn,
    )

    # Step 4: Disable sensitive layers
    if sensitive_layers and verbose:
        print(f"Step 4: Disabling {len(sensitive_layers)} sensitive layers...")
    for layer_name in sensitive_layers:
        try:
            layer = dict(model.named_modules())[layer_name]
            disable_quantization(layer).apply()
        except KeyError:
            if verbose:
                print(f"  Warning: Layer not found: {layer_name}")

    if verbose:
        print("\nPTQ complete!")
        print_quantizer_status(model)

    return model


def save_ptq_model(model: nn.Module, path: str, save_calibration: bool = True):
    """
    Save PTQ quantized model.

    Args:
        model: Quantized model
        path: Path to save checkpoint
        save_calibration: Whether to save calibration cache separately
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save model state dict
    torch.save({"state_dict": model.state_dict()}, path)
    print(f"Saved PTQ model to {path}")

    # Optionally save calibration cache
    if save_calibration:
        calib_path = path.with_suffix(".calib")
        calibrator = CalibrationManager(model)
        calibrator.save_calib_cache(str(calib_path))


def load_ptq_model(
    model: nn.Module,
    checkpoint_path: str,
    calib_cache_path: Optional[str] = None,
    fuse_bn: bool = True,
    sensitive_layers: Optional[Set[str]] = None,
) -> nn.Module:
    """
    Load a PTQ quantized model.

    This function:
    1. Inserts Q/DQ nodes (to match the saved model structure)
    2. Loads the checkpoint
    3. Optionally loads calibration cache

    Args:
        model: Unquantized CenterPoint model
        checkpoint_path: Path to PTQ checkpoint
        calib_cache_path: Optional path to calibration cache
        fuse_bn: Whether BatchNorm was fused during PTQ
        sensitive_layers: Layers that were skipped during PTQ

    Returns:
        Loaded PTQ model
    """
    model.eval()
    sensitive_layers = sensitive_layers or set()

    # Fuse BN if it was done during PTQ
    if fuse_bn:
        fuse_model_bn(model)

    # Insert Q/DQ nodes
    quant_model(model, skip_names=sensitive_layers)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    # Load calibration cache if provided
    if calib_cache_path:
        calibrator = CalibrationManager(model)
        calibrator.load_calib_cache(calib_cache_path)

    print(f"Loaded PTQ model from {checkpoint_path}")
    return model
