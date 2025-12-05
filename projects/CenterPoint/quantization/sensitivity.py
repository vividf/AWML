# Copyright (c) OpenMMLab. All rights reserved.
"""Sensitivity analysis for quantization."""

import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from pytorch_quantization.nn import TensorQuantizer

    PYTORCH_QUANTIZATION_AVAILABLE = True
except ImportError:
    PYTORCH_QUANTIZATION_AVAILABLE = False
    TensorQuantizer = None


def _check_pytorch_quantization():
    """Check if pytorch-quantization is available."""
    if not PYTORCH_QUANTIZATION_AVAILABLE:
        raise ImportError(
            "pytorch-quantization is required for sensitivity analysis. "
            "Install it with: pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com"
        )


def get_quantized_layer_names(model: nn.Module) -> List[str]:
    """
    Get unique layer names that have quantizers.

    Args:
        model: Quantized PyTorch model

    Returns:
        List of layer names (without _input/_weight_quantizer suffix)
    """
    _check_pytorch_quantization()

    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            # Remove quantizer suffix to get layer name
            layer_name = name.replace("._input_quantizer", "").replace("._weight_quantizer", "")
            if layer_name not in layer_names:
                layer_names.append(layer_name)

    return layer_names


def disable_all_quantizers(model: nn.Module):
    """Disable all TensorQuantizers in the model."""
    _check_pytorch_quantization()

    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            module.disable()


def enable_quantizers_for_layer(model: nn.Module, layer_name: str):
    """Enable quantizers for a specific layer."""
    _check_pytorch_quantization()

    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and layer_name in name:
            module.enable()


def disable_quantizers_for_layer(model: nn.Module, layer_name: str):
    """Disable quantizers for a specific layer."""
    _check_pytorch_quantization()

    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and layer_name in name:
            module.disable()


def build_sensitivity_profile(
    model: nn.Module,
    eval_fn: Callable[[nn.Module], float],
    output_file: Optional[str] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Analyze the impact of quantizing each layer on model performance.

    This function tests each quantized layer individually to measure its
    impact on accuracy. Layers with high delta (accuracy drop) are candidates
    for keeping in FP16.

    Args:
        model: Quantized PyTorch model (must have been calibrated)
        eval_fn: Evaluation function that takes model and returns metric (e.g., mAP)
        output_file: Optional path to save results as CSV
        verbose: Whether to print progress

    Returns:
        List of dicts with 'layer', 'metric', and 'delta' for each layer,
        sorted by delta (highest impact first)

    Example:
        >>> def eval_fn(model):
        ...     # Run evaluation and return mAP
        ...     return evaluate(model, val_dataloader)
        ...
        >>> results = build_sensitivity_profile(model, eval_fn)
        >>> sensitive_layers = [r['layer'] for r in results if r['delta'] > 0.5]
    """
    _check_pytorch_quantization()

    layer_names = get_quantized_layer_names(model)

    if verbose:
        print(f"Found {len(layer_names)} quantized layers")
        print("Disabling all quantizers for baseline evaluation...")

    # Disable all quantizers for baseline
    disable_all_quantizers(model)

    # Get baseline metric (FP32)
    model.eval()
    with torch.no_grad():
        baseline_metric = eval_fn(model)

    if verbose:
        print(f"Baseline metric (FP32): {baseline_metric:.4f}")
        print("\nTesting layer sensitivity...")

    results = []

    # Test each layer
    iterator = tqdm(enumerate(layer_names), total=len(layer_names)) if verbose else enumerate(layer_names)

    for i, layer_name in iterator:
        # Enable only this layer's quantizers
        enable_quantizers_for_layer(model, layer_name)

        # Evaluate
        with torch.no_grad():
            layer_metric = eval_fn(model)

        delta = baseline_metric - layer_metric

        results.append(
            {
                "layer": layer_name,
                "metric": layer_metric,
                "delta": delta,
            }
        )

        if verbose:
            tqdm.write(f"  {layer_name}: metric={layer_metric:.4f}, delta={delta:.4f}")

        # Disable this layer's quantizers
        disable_quantizers_for_layer(model, layer_name)

    # Sort by impact (highest delta first)
    results.sort(key=lambda x: x["delta"], reverse=True)

    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["layer", "metric", "delta"])
            writer.writeheader()
            writer.writerows(results)

        if verbose:
            print(f"\nResults saved to {output_path}")

    return results


def get_sensitive_layers(
    results: List[Dict[str, Any]],
    threshold: float = 0.5,
    top_k: Optional[int] = None,
) -> List[str]:
    """
    Extract sensitive layer names from sensitivity analysis results.

    Args:
        results: Results from build_sensitivity_profile
        threshold: Minimum delta to consider a layer sensitive
        top_k: If provided, return only top K most sensitive layers

    Returns:
        List of sensitive layer names
    """
    # Filter by threshold
    sensitive = [r["layer"] for r in results if r["delta"] >= threshold]

    # Limit to top_k if specified
    if top_k is not None:
        sensitive = sensitive[:top_k]

    return sensitive
