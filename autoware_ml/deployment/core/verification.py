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
    model_type: str = "auto",  # "auto", "CenterPoint", "YOLOX", etc.
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
        model_type: Model type ("auto", "CenterPoint", "YOLOX", etc.)

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
        # logger.info(f"  PyTorch output: {pytorch_output}")
        logger.info(f"  PyTorch output type: {type(pytorch_output)}")
        logger.info(f"  PyTorch output length: {len(pytorch_output)}")

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

            # Compute differences for device consistency analysis
            max_diff = _compute_max_difference(pytorch_output, onnx_output)
            mean_diff = _compute_mean_difference(pytorch_output, onnx_output)

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
            else:
                # Check if there's a device mismatch that wasn't caught
                pytorch_device = pytorch_backend.device
                onnx_device = onnx_backend.device
                logger.info(f"Device consistency check: PyTorch={pytorch_device}, ONNX={onnx_device}")
                
                # If PyTorch is on CUDA but ONNX reports CUDA but has large differences, 
                # there might be a hidden device mismatch
                if pytorch_device.startswith("cuda") and onnx_device == "cuda" and max_diff > tolerance * 10:
                    logger.warning(f"Large numerical difference ({max_diff:.6f}) detected despite both backends reporting CUDA")
                    logger.warning("This may indicate a hidden device mismatch or ONNX Runtime CUDA issues")
                    logger.warning("Consider forcing CPU mode for more consistent results")

            results[f"{sample_name}_onnx"] = onnx_success

        # Verify TensorRT
        if tensorrt_path:
            logger.info("\nVerifying TensorRT model...")
            
            trt_backend = TensorRTBackend(tensorrt_path, device="cuda")
            
            # Use standard backend verification
            trt_success, trt_output, trt_latency = _verify_backend(
                trt_backend,
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
    reference_output: np.ndarray,
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
        logger.info(f"  {backend_name} output type: {type(output)}")
        if output is not None:
            logger.info(f"  {backend_name} output length: {len(output) if hasattr(output, '__len__') else 'N/A'}")
        else:
            logger.info(f"  {backend_name} output: None")

        # Compare outputs
        # Handle None outputs
        if output is None or reference_output is None:
            logger.error(f"  {backend_name} verification FAILED: None output detected")
            return False, None, 0.0
        
        # Handle empty list outputs
        if isinstance(output, list) and len(output) == 0:
            logger.error(f"  {backend_name} verification FAILED: Empty output list")
            return False, None, 0.0
        if isinstance(reference_output, list) and len(reference_output) == 0:
            logger.error(f"  {backend_name} verification FAILED: Empty reference output list")
            return False, None, 0.0
        
        # Log detailed output information for debugging
        logger.info(f"  Reference output details:")
        if isinstance(reference_output, list):
            for i, out in enumerate(reference_output):
                if isinstance(out, np.ndarray):
                    logger.info(f"    Output[{i}] shape: {out.shape}, dtype: {out.dtype}")
                    logger.info(f"    Output[{i}] range: [{out.min():.6f}, {out.max():.6f}]")
                    logger.info(f"    Output[{i}] mean: {out.mean():.6f}, std: {out.std():.6f}")
        elif isinstance(reference_output, np.ndarray):
            logger.info(f"    Shape: {reference_output.shape}, dtype: {reference_output.dtype}")
            logger.info(f"    Range: [{reference_output.min():.6f}, {reference_output.max():.6f}]")
            logger.info(f"    Mean: {reference_output.mean():.6f}, std: {reference_output.std():.6f}")
        
        logger.info(f"  {backend_name} output details:")
        if isinstance(output, list):
            for i, out in enumerate(output):
                if isinstance(out, np.ndarray):
                    logger.info(f"    Output[{i}] shape: {out.shape}, dtype: {out.dtype}")
                    logger.info(f"    Output[{i}] range: [{out.min():.6f}, {out.max():.6f}]")
                    logger.info(f"    Output[{i}] mean: {out.mean():.6f}, std: {out.std():.6f}")
        elif isinstance(output, np.ndarray):
            logger.info(f"    Shape: {output.shape}, dtype: {output.dtype}")
            logger.info(f"    Range: [{output.min():.6f}, {output.max():.6f}]")
            logger.info(f"    Mean: {output.mean():.6f}, std: {output.std():.6f}")
        
        # Handle different output formats
        if isinstance(output, list) and isinstance(reference_output, list):
            # Both are lists (e.g., CenterPoint head outputs)
            max_diff = 0.0
            mean_diff = 0.0
            total_elements = 0
            
            logger.info(f"  Computing differences for {len(output)} outputs...")
            for i, (ref_out, out) in enumerate(zip(reference_output, output)):
                if isinstance(ref_out, np.ndarray) and isinstance(out, np.ndarray):
                    diff = np.abs(ref_out - out)
                    output_max_diff = diff.max()
                    output_mean_diff = diff.mean()
                    max_diff = max(max_diff, output_max_diff)
                    mean_diff += diff.sum()
                    total_elements += diff.size
                    logger.info(f"    Output[{i}] - max_diff: {output_max_diff:.6f}, mean_diff: {output_mean_diff:.6f}")
            
            if total_elements > 0:
                mean_diff /= total_elements
            else:
                logger.warning(f"  No elements found for comparison!")
        else:
            # Standard array comparison
            max_diff = np.abs(reference_output - output).max()
            mean_diff = np.abs(reference_output - output).mean()

        logger.info(f"  Overall Max difference: {max_diff:.6f}")
        logger.info(f"  Overall Mean difference: {mean_diff:.6f}")

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
    # Handle None outputs
    if output1 is None or output2 is None:
        return float('inf')
    
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
    # Handle None outputs
    if output1 is None or output2 is None:
        return float('inf')
    
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


def verify_centerpoint_pipeline(
    pytorch_model: torch.nn.Module,
    test_inputs: Dict[str, torch.Tensor],
    onnx_dir: str = None,
    tensorrt_dir: str = None,
    device: str = "cuda",
    tolerance: float = DEFAULT_TOLERANCE,
    logger: logging.Logger = None,
) -> Dict[str, bool]:
    """
    Verify CenterPoint models using pipeline-based verification.
    
    This approach integrates verification into the evaluation pipeline architecture,
    allowing verification and evaluation to share the same inference path while
    differing only in whether postprocessing is applied.
    
    Args:
        pytorch_model: Reference PyTorch model (ONNX-compatible)
        test_inputs: Dictionary of test inputs (e.g., {'sample_0': tensor, ...})
        onnx_dir: Optional directory containing ONNX models
        tensorrt_dir: Optional directory containing TensorRT engines
        device: Device for inference
        tolerance: Maximum allowed difference
        logger: Optional logger instance
        
    Returns:
        Dictionary with verification results for each backend
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Import Pipeline classes
    from ..pipelines import (
        CenterPointPyTorchPipeline,
        CenterPointONNXPipeline,
        CenterPointTensorRTPipeline
    )
    
    results = {}
    
    logger.info("=" * 60)
    logger.info("Running CenterPoint Pipeline-based Verification")
    logger.info("=" * 60)
    
    # Create PyTorch pipeline for reference
    pytorch_pipeline = CenterPointPyTorchPipeline(pytorch_model, device=device)
    
    # Verify each sample
    for sample_name, input_tensor in test_inputs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Verifying sample: {sample_name}")
        logger.info(f"{'='*60}")
        
        # Get PyTorch reference (raw outputs)
        logger.info("Running PyTorch inference (raw outputs)...")
        try:
            pytorch_output, pytorch_latency = pytorch_pipeline.infer(
                input_tensor, 
                sample_meta={},
                return_raw_outputs=True
            )
            logger.info(f"  PyTorch latency: {pytorch_latency:.2f} ms")
            logger.info(f"  PyTorch output: {len(pytorch_output)} head outputs")
            for i, out in enumerate(pytorch_output):
                if isinstance(out, torch.Tensor):
                    logger.info(f"    Output[{i}] shape: {out.shape}")
        except Exception as e:
            logger.error(f"  PyTorch inference failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Verify ONNX
        if onnx_dir:
            logger.info("\nVerifying ONNX pipeline...")
            try:
                onnx_pipeline = CenterPointONNXPipeline(
                    pytorch_model, 
                    onnx_dir=onnx_dir,
                    device=device
                )
                
                onnx_output, onnx_latency = onnx_pipeline.infer(
                    input_tensor,
                    sample_meta={},
                    return_raw_outputs=True
                )
                
                logger.info(f"  ONNX latency: {onnx_latency:.2f} ms")
                logger.info(f"  ONNX output: {len(onnx_output)} head outputs")
                
                # Compare outputs
                max_diff = 0.0
                mean_diff = 0.0
                total_elements = 0
                
                output_names = ['heatmap', 'reg', 'height', 'dim', 'rot', 'vel']
                
                for i, (ref_out, onnx_out) in enumerate(zip(pytorch_output, onnx_output)):
                    if isinstance(ref_out, torch.Tensor) and isinstance(onnx_out, torch.Tensor):
                        ref_np = ref_out.cpu().numpy()
                        onnx_np = onnx_out.cpu().numpy()
                        
                        # Log value ranges for debugging
                        output_name = output_names[i] if i < len(output_names) else f"output_{i}"
                        logger.info(f"    {output_name}: shape={ref_np.shape}")
                        
                        # Check for NaN or Inf
                        pytorch_nan = np.isnan(ref_np).any()
                        pytorch_inf = np.isinf(ref_np).any()
                        onnx_nan = np.isnan(onnx_np).any()
                        onnx_inf = np.isinf(onnx_np).any()
                        
                        if pytorch_nan or pytorch_inf or onnx_nan or onnx_inf:
                            logger.warning(f"      ⚠️  Special values detected:")
                            if pytorch_nan: logger.warning(f"         PyTorch has NaN!")
                            if pytorch_inf: logger.warning(f"         PyTorch has Inf!")
                            if onnx_nan: logger.warning(f"         ONNX has NaN!")
                            if onnx_inf: logger.warning(f"         ONNX has Inf!")
                        
                        logger.info(f"      PyTorch range: [{ref_np.min():.3f}, {ref_np.max():.3f}], mean: {ref_np.mean():.3f}, std: {ref_np.std():.3f}")
                        logger.info(f"      ONNX range: [{onnx_np.min():.3f}, {onnx_np.max():.3f}], mean: {onnx_np.mean():.3f}, std: {onnx_np.std():.3f}")
                        
                        diff = np.abs(ref_np - onnx_np)
                        output_max_diff = diff.max()
                        output_mean_diff = diff.mean()
                        max_diff = max(max_diff, output_max_diff)
                        mean_diff += diff.sum()
                        total_elements += diff.size
                        logger.info(f"      max_diff: {output_max_diff:.6f}, mean_diff: {output_mean_diff:.6f}")
                
                if total_elements > 0:
                    mean_diff /= total_elements
                
                logger.info(f"  Overall Max difference: {max_diff:.6f}")
                logger.info(f"  Overall Mean difference: {mean_diff:.6f}")
                
                onnx_success = max_diff < tolerance
                if onnx_success:
                    logger.info("  ONNX verification PASSED ✓")
                else:
                    logger.warning(f"  ONNX verification FAILED ✗ (max diff: {max_diff:.6f} > tolerance: {tolerance:.6f})")
                
                results[f"{sample_name}_onnx"] = onnx_success
                
            except Exception as e:
                logger.error(f"  ONNX verification failed with error: {e}")
                import traceback
                traceback.print_exc()
                results[f"{sample_name}_onnx"] = False
        
        # Verify TensorRT
        if tensorrt_dir:
            logger.info("\nVerifying TensorRT pipeline...")
            
            # Check if CUDA is available
            if not device.startswith("cuda"):
                logger.warning("  TensorRT requires CUDA device, skipping")
                results[f"{sample_name}_tensorrt"] = False
                continue
            
            try:
                trt_pipeline = CenterPointTensorRTPipeline(
                    pytorch_model,
                    tensorrt_dir=tensorrt_dir,
                    device=device
                )
                
                trt_output, trt_latency = trt_pipeline.infer(
                    input_tensor,
                    sample_meta={},
                    return_raw_outputs=True
                )
                
                logger.info(f"  TensorRT latency: {trt_latency:.2f} ms")
                logger.info(f"  TensorRT output: {len(trt_output)} head outputs")
                
                # Compare outputs
                max_diff = 0.0
                mean_diff = 0.0
                total_elements = 0
                
                for i, (ref_out, trt_out) in enumerate(zip(pytorch_output, trt_output)):
                    if isinstance(ref_out, torch.Tensor) and isinstance(trt_out, torch.Tensor):
                        ref_np = ref_out.cpu().numpy()
                        trt_np = trt_out.cpu().numpy()
                        
                        diff = np.abs(ref_np - trt_np)
                        output_max_diff = diff.max()
                        output_mean_diff = diff.mean()
                        max_diff = max(max_diff, output_max_diff)
                        mean_diff += diff.sum()
                        total_elements += diff.size
                        logger.info(f"    Output[{i}] - max_diff: {output_max_diff:.6f}, mean_diff: {output_mean_diff:.6f}")
                
                if total_elements > 0:
                    mean_diff /= total_elements
                
                logger.info(f"  Overall Max difference: {max_diff:.6f}")
                logger.info(f"  Overall Mean difference: {mean_diff:.6f}")
                
                trt_success = max_diff < tolerance
                if trt_success:
                    logger.info("  TensorRT verification PASSED ✓")
                else:
                    logger.warning(f"  TensorRT verification FAILED ✗ (max diff: {max_diff:.6f} > tolerance: {tolerance:.6f})")
                
                results[f"{sample_name}_tensorrt"] = trt_success
                
            except Exception as e:
                logger.error(f"  TensorRT verification failed with error: {e}")
                import traceback
                traceback.print_exc()
                results[f"{sample_name}_tensorrt"] = False
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("Verification Summary")
    logger.info(f"{'='*60}")
    for key, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"  {key}: {status}")
    logger.info(f"{'='*60}")
    
    return results
