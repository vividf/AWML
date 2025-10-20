"""
Unified model verification module.

Provides utilities for verifying exported models against reference PyTorch outputs.
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from ..backends import BaseBackend, ONNXBackend, PyTorchBackend, TensorRTBackend

DEFAULT_TOLERANCE = 1e-3


def verify_model_outputs(
    pytorch_model: torch.nn.Module,
    test_inputs: Dict[str, torch.Tensor],
    onnx_path: Optional[str] = None,
    tensorrt_path: Optional[str] = None,
    device: str = "cpu",
    tolerance: float = DEFAULT_TOLERANCE,
    logger: logging.Logger = None,
) -> Dict[str, bool]:
    """
    Verify exported models against PyTorch reference.

    Args:
        pytorch_model: Reference PyTorch model
        test_inputs: Dictionary of test inputs (e.g., {'sample1': tensor1, ...})
        onnx_path: Optional path to ONNX model
        tensorrt_path: Optional path to TensorRT engine
        device: Device for PyTorch inference
        tolerance: Maximum allowed difference
        logger: Optional logger instance

    Returns:
        Dictionary with verification results for each backend
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    results = {}

    # Run PyTorch inference to get reference outputs
    logger.info("=" * 60)
    logger.info("Running verification...")
    logger.info("=" * 60)

    # Determine if we need raw output based on model type
    # Both detection models (like YOLOX) and classification models need raw_output=True for ONNX verification
    # Detection models need it for bbox_head outputs, classification models need it for raw logits
    # 3D detection models (CenterPoint) need special handling for voxel-based inputs
    model_type = type(pytorch_model).__name__.lower()
    use_raw_output = True  # Always use raw output for verification to match ONNX format
    
    pytorch_backend = PyTorchBackend(pytorch_model, device=device, raw_output=use_raw_output)
    pytorch_backend.load_model()

    # Verify each backend
    for sample_name, input_tensor in test_inputs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Verifying sample: {sample_name}")
        logger.info(f"{'='*60}")

        # Get PyTorch reference
        logger.info("Running PyTorch inference...")
        pytorch_output, pytorch_latency = pytorch_backend.infer(input_tensor)
        logger.info(f"  PyTorch latency: {pytorch_latency:.2f} ms")
        logger.info(f"  PyTorch output type: {type(pytorch_output)}")
        if isinstance(pytorch_output, list):
            logger.info(f"  PyTorch output: {len(pytorch_output)} outputs")
            for i, out in enumerate(pytorch_output):
                logger.info(f"  PyTorch output[{i}] shape: {out.shape}")
        else:
            logger.info(f"  PyTorch output: {pytorch_output}")

        # Verify ONNX - 使用統一的 ONNXBackend
        if onnx_path:
            logger.info("\nVerifying ONNX model...")
            
            # ONNXBackend 自動檢測是否為 CenterPoint
            onnx_backend = ONNXBackend(
                onnx_path, 
                device=device, 
                pytorch_model=pytorch_model  # 傳遞 PyTorch 模型用於 CenterPoint
            )
            
            onnx_success, onnx_output, onnx_latency = _verify_backend(
                onnx_backend,
                input_tensor,
                pytorch_output,
                tolerance,
                "ONNX",
                logger,
            )

            # If ONNX backend fell back to CPU, update PyTorch backend, recompute reference, and re-compare
            if hasattr(onnx_backend, '_session') and onnx_backend.device == "cpu" and device.startswith("cuda"):
                logger.warning("ONNX backend fell back to CPU, updating PyTorch backend for consistency...")
                pytorch_backend.device = "cpu"
                pytorch_backend._torch_device = torch.device("cpu")
                pytorch_backend._model = pytorch_backend._model.cpu()
                logger.info("PyTorch backend updated to use CPU")

                logger.info("Re-running PyTorch inference on CPU...")
                pytorch_output, pytorch_latency = pytorch_backend.infer(input_tensor)
                logger.info(f"  PyTorch latency (CPU): {pytorch_latency:.2f} ms")
                logger.info(f"  PyTorch output (CPU): {pytorch_output}")

                # Recompute differences now that both are on CPU
                max_diff = _compute_max_difference(pytorch_output, onnx_output)
                mean_diff = _compute_mean_difference(pytorch_output, onnx_output)
                logger.info(f"  Recomputed Max difference: {max_diff:.6f}")
                logger.info(f"  Recomputed Mean difference: {mean_diff:.6f}")
                onnx_success = max_diff < tolerance
                if onnx_success:
                    logger.info("  ONNX verification PASSED ✓ (after CPU fallback)")
                else:
                    logger.warning(
                        f"  ONNX verification FAILED ✗ (after CPU fallback) (max diff: {max_diff:.6f} > tolerance: {tolerance:.6f})"
                    )

            results[f"{sample_name}_onnx"] = onnx_success

        # Verify TensorRT
        if tensorrt_path:
            logger.info("\nVerifying TensorRT model...")
            trt_success = _verify_backend(
                TensorRTBackend(tensorrt_path, device="cuda"),
                input_tensor,
                pytorch_output,
                tolerance,
                "TensorRT",
                logger,
            )
            results[f"{sample_name}_tensorrt"] = trt_success

    logger.info(f"\n{'='*60}")
    logger.info("Verification Summary")
    logger.info(f"{'='*60}")
    for key, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"  {key}: {status}")
    logger.info(f"{'='*60}")

    return results


def _verify_backend(
    backend: BaseBackend,
    input_tensor: torch.Tensor,
    reference_output,  # Can be np.ndarray or list of np.ndarray
    tolerance: float,
    backend_name: str,
    logger: logging.Logger,
) -> tuple[bool, np.ndarray, float]:
    """
    Verify a single backend against reference output.

    Args:
        backend: Backend instance to verify
        input_tensor: Input tensor
        reference_output: Reference output from PyTorch
        tolerance: Maximum allowed difference
        backend_name: Name of backend for logging
        logger: Logger instance

    Returns:
        Tuple of (passed, output, latency_ms)
    """
    try:
        with backend:
            output, latency = backend.infer(input_tensor)

        logger.info(f"  {backend_name} latency: {latency:.2f} ms")
        logger.info(f"  {backend_name} output: {output}")

        # Compare outputs
        # Handle different output formats
        if isinstance(output, list) and isinstance(reference_output, list):
            # Both are lists (e.g., CenterPoint head outputs)
            max_diff = 0.0
            mean_diff = 0.0
            total_elements = 0
            
            for ref_out, out in zip(reference_output, output):
                if isinstance(ref_out, np.ndarray) and isinstance(out, np.ndarray):
                    if ref_out.shape != out.shape:
                        logger.error(f"Shape mismatch: ref {ref_out.shape} vs out {out.shape}")
                        # Set max_diff to inf to fail verification
                        max_diff = np.inf
                        continue
                    diff = np.abs(ref_out - out)
                    max_diff = max(max_diff, float(diff.max()))
                    mean_diff += float(diff.sum())
                    total_elements += int(diff.size)
            
            if total_elements > 0:
                mean_diff /= total_elements
        else:
            # Standard array comparison
            max_diff = np.abs(reference_output - output).max()
            mean_diff = np.abs(reference_output - output).mean()

        logger.info(f"  Max difference: {max_diff:.6f}")
        logger.info(f"  Mean difference: {mean_diff:.6f}")

        if max_diff < tolerance:
            logger.info(f"  {backend_name} verification PASSED ✓")
            return True, output, latency
        else:
            logger.warning(
                f"  {backend_name} verification FAILED ✗ " f"(max diff: {max_diff:.6f} > tolerance: {tolerance:.6f})"
            )
            return False, output, latency

    except Exception as e:
        logger.error(f"  {backend_name} verification failed with error: {e}")
        return False, None, 0.0


def _compute_max_difference(output1, output2) -> float:
    """
    Compute maximum difference between two outputs.
    
    Args:
        output1: First output (reference)
        output2: Second output (to compare)
        
    Returns:
        Maximum absolute difference
    """
    # Handle different output formats
    if isinstance(output1, list) and isinstance(output2, list):
        # Both are lists (e.g., CenterPoint head outputs)
        max_diff = 0.0
        
        for ref_out, out in zip(output1, output2):
            if isinstance(ref_out, np.ndarray) and isinstance(out, np.ndarray):
                diff = np.abs(ref_out - out)
                max_diff = max(max_diff, diff.max())
        
        return max_diff
    else:
        # Standard array comparison
        return np.abs(output1 - output2).max()


def _compute_mean_difference(output1, output2) -> float:
    """
    Compute mean difference between two outputs.
    
    Args:
        output1: First output (reference)
        output2: Second output (to compare)
        
    Returns:
        Mean absolute difference
    """
    # Handle different output formats
    if isinstance(output1, list) and isinstance(output2, list):
        # Both are lists (e.g., CenterPoint head outputs)
        mean_diff = 0.0
        total_elements = 0
        
        for ref_out, out in zip(output1, output2):
            if isinstance(ref_out, np.ndarray) and isinstance(out, np.ndarray):
                diff = np.abs(ref_out - out)
                mean_diff += diff.sum()
                total_elements += diff.size
        
        return mean_diff / total_elements if total_elements > 0 else 0.0
    else:
        # Standard array comparison
        return np.abs(output1 - output2).mean()


def compare_outputs(
    output1: np.ndarray,
    output2: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Dict[str, float]:
    """
    Compare two model outputs and return difference statistics.

    Args:
        output1: First output array
        output2: Second output array
        tolerance: Maximum allowed difference

    Returns:
        Dictionary with difference statistics
    """
    max_diff = _compute_max_difference(output1, output2)
    mean_diff = _compute_mean_difference(output1, output2)

    return {
        "max_difference": max_diff,
        "mean_difference": mean_diff,
        "passed": max_diff < tolerance,
    }
