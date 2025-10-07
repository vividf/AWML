"""
Model export and inference verification utilities.
"""

import gc
import logging
import os
import os.path as osp
import pickle
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
from config import DEFAULT_VERIFICATION_TOLERANCE, LABELS, DeploymentConfig
from mmengine.config import Config

from autoware_ml.calibration_classification.datasets.transforms.calibration_classification_transform import (
    CalibrationClassificationTransform,
)

# ============================================================================
# Data Loading Utilities
# ============================================================================


def _load_info_pkl_file(info_pkl_path: str) -> list:
    """
    Load and parse info.pkl file.

    Args:
        info_pkl_path: Path to the info.pkl file

    Returns:
        List of samples from data_list

    Raises:
        FileNotFoundError: If info.pkl file doesn't exist
        ValueError: If data format is unexpected
    """
    if not os.path.exists(info_pkl_path):
        raise FileNotFoundError(f"Info.pkl file not found: {info_pkl_path}")

    try:
        with open(info_pkl_path, "rb") as f:
            info_data = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load info.pkl file: {e}")

    # Extract samples from info.pkl
    if isinstance(info_data, dict):
        if "data_list" in info_data:
            samples_list = info_data["data_list"]
        else:
            raise ValueError(f"Expected 'data_list' key in info_data, found keys: {list(info_data.keys())}")
    else:
        raise ValueError(f"Expected dict format, got {type(info_data)}")

    if not samples_list:
        raise ValueError("No samples found in info.pkl")

    return samples_list


def load_info_pkl_data(info_pkl_path: str, sample_idx: int = 0) -> Dict[str, Any]:
    """
    Load a single sample from info.pkl file.

    Args:
        info_pkl_path: Path to the info.pkl file
        sample_idx: Index of the sample to load (default: 0)

    Returns:
        Sample dictionary with the required structure for CalibrationClassificationTransform

    Raises:
        FileNotFoundError: If info.pkl file doesn't exist
        ValueError: If data format is unexpected or sample index is invalid
    """
    samples_list = _load_info_pkl_file(info_pkl_path)

    if sample_idx >= len(samples_list):
        raise ValueError(f"Sample index {sample_idx} out of range (0-{len(samples_list)-1})")

    sample = samples_list[sample_idx]

    # Validate sample structure
    required_keys = ["image", "lidar_points"]
    if not all(key in sample for key in required_keys):
        raise ValueError(f"Sample {sample_idx} has invalid structure. Required keys: {required_keys}")

    return sample


def _create_transform_with_config(
    model_cfg: Config,
    miscalibration_probability: float = 0.0,
    projection_vis_dir: Optional[str] = None,
    results_vis_dir: Optional[str] = None,
    binary_save_dir: Optional[str] = None,
) -> CalibrationClassificationTransform:
    """
    Create CalibrationClassificationTransform with model configuration.

    Args:
        model_cfg: Model configuration
        miscalibration_probability: Probability of miscalibration
        projection_vis_dir: Optional projection visualization directory
        results_vis_dir: Optional results visualization directory
        binary_save_dir: Optional binary save directory

    Returns:
        Configured CalibrationClassificationTransform
    """
    data_root = model_cfg.get("data_root")
    if data_root is None:
        raise ValueError("data_root not found in model configuration")

    transform_config = model_cfg.get("transform_config")
    if transform_config is None:
        raise ValueError("transform_config not found in model configuration")

    return CalibrationClassificationTransform(
        transform_config=transform_config,
        mode="test",
        max_depth=model_cfg.get("max_depth", 128.0),
        dilation_size=model_cfg.get("dilation_size", 1),
        undistort=True,
        miscalibration_probability=miscalibration_probability,
        enable_augmentation=False,
        data_root=data_root,
        projection_vis_dir=projection_vis_dir,
        results_vis_dir=results_vis_dir,
        binary_save_dir=binary_save_dir,
    )


def _numpy_to_tensor(numpy_array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """
    Convert numpy array (H, W, C) to tensor (1, C, H, W).

    Args:
        numpy_array: Input numpy array with shape (H, W, C)
        device: Device to load tensor on

    Returns:
        Tensor with shape (1, C, H, W)
    """
    tensor = torch.from_numpy(numpy_array).permute(2, 0, 1).float()  # (C, H, W)
    return tensor.unsqueeze(0).to(device)  # (1, C, H, W)


def _clear_gpu_memory() -> None:
    """Clear GPU cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def load_sample_data_from_info_pkl(
    info_pkl_path: str,
    model_cfg: Config,
    miscalibration_probability: float,
    sample_idx: int = 0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Load and preprocess sample data from info.pkl using CalibrationClassificationTransform.

    Args:
        info_pkl_path: Path to the info.pkl file
        model_cfg: Model configuration containing data_root setting
        miscalibration_probability: Probability of loading a miscalibrated sample
        sample_idx: Index of the sample to load (default: 0)
        device: Device to load tensor on

    Returns:
        Preprocessed tensor ready for model inference
    """
    # Load sample data from info.pkl
    sample_data = load_info_pkl_data(info_pkl_path, sample_idx)

    # Create transform
    transform = _create_transform_with_config(
        model_cfg=model_cfg,
        miscalibration_probability=miscalibration_probability,
        projection_vis_dir=model_cfg.get("test_projection_vis_dir", None),
        results_vis_dir=model_cfg.get("test_results_vis_dir", None),
        binary_save_dir=model_cfg.get("binary_save_dir", None),
    )

    # Apply transform
    results = transform.transform(sample_data)
    input_data_processed = results["fused_img"]  # (H, W, 5)

    # Convert to tensor
    return _numpy_to_tensor(input_data_processed, device)


# ============================================================================
# Model Export Functions
# ============================================================================


def export_to_onnx(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    output_path: str,
    config: DeploymentConfig,
    logger: logging.Logger,
) -> None:
    """Export model to ONNX format."""
    settings = config.onnx_settings
    model.eval()

    logger.info("Exporting model to ONNX format...")
    logger.info(f"Input shape: {input_tensor.shape}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"ONNX opset version: {settings['opset_version']}")

    with torch.no_grad():
        torch.onnx.export(
            model,
            input_tensor,
            output_path,
            export_params=settings["export_params"],
            keep_initializers_as_inputs=settings["keep_initializers_as_inputs"],
            opset_version=settings["opset_version"],
            do_constant_folding=settings["do_constant_folding"],
            input_names=settings["input_names"],
            output_names=settings["output_names"],
            dynamic_axes=settings["dynamic_axes"],
            verbose=False,
        )

    logger.info(f"ONNX export completed: {output_path}")

    # Optional model simplification
    _optimize_onnx_model(output_path, logger)


def _optimize_onnx_model(onnx_path: str, logger: logging.Logger) -> None:
    """Optimize ONNX model using onnxsim."""
    logger.info("Simplifying ONNX model...")
    model_simplified, success = onnxsim.simplify(onnx_path)
    if success:
        onnx.save(model_simplified, onnx_path)
        logger.info(f"ONNX model simplified successfully. Saved to {onnx_path}")
    else:
        logger.warning("ONNX model simplification failed")


def export_to_tensorrt(
    onnx_path: str, output_path: str, input_tensor: torch.Tensor, config: DeploymentConfig, logger: logging.Logger
) -> bool:
    """Export ONNX model to a TensorRT engine with precision policy support."""
    settings = config.tensorrt_settings
    precision_policy = settings["precision_policy"]
    policy_flags = settings["policy_flags"]

    logger.info(f"Building TensorRT engine with precision policy: {precision_policy}")

    # Initialize TensorRT
    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(trt_logger, "")

    builder = trt.Builder(trt_logger)
    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(pool=trt.MemoryPoolType.WORKSPACE, pool_size=settings["max_workspace_size"])

    # Create network with appropriate flags
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    # Handle strongly typed flag (network creation flag, not builder flag)
    if policy_flags.get("STRONGLY_TYPED"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        logger.info("Using strongly typed TensorRT network creation")

    network = builder.create_network(flags)

    # Apply precision flags to builder config
    for flag_name, enabled in policy_flags.items():
        if flag_name == "STRONGLY_TYPED":
            continue  # Already handled as network creation flag above
        if enabled and hasattr(trt.BuilderFlag, flag_name):
            builder_config.set_flag(getattr(trt.BuilderFlag, flag_name))
            logger.info(f"BuilderFlag.{flag_name} enabled")

    # Setup optimization profile
    profile = builder.create_optimization_profile()
    _configure_input_shapes(profile, input_tensor, settings["model_inputs"], logger)
    builder_config.add_optimization_profile(profile)

    # Parse ONNX model
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            _log_parser_errors(parser, logger)
            return False
        logger.info("Successfully parsed the ONNX file")

    # Build engine
    logger.info("Building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, builder_config)

    if serialized_engine is None:
        logger.error("Failed to build TensorRT engine")
        return False

    # Save engine
    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    logger.info(f"TensorRT engine saved to {output_path}")
    logger.info(f"Engine max workspace size: {settings['max_workspace_size'] / (1024**3):.2f} GB")

    return True


def _configure_input_shapes(profile, input_tensor: torch.Tensor, model_inputs: list, logger: logging.Logger) -> None:
    """Configure input shapes for TensorRT optimization profile."""
    if model_inputs:
        input_shapes = model_inputs[0].get("input_shapes", {})
        for input_name, shapes in input_shapes.items():
            min_shape = shapes.get("min_shape", list(input_tensor.shape))
            opt_shape = shapes.get("opt_shape", list(input_tensor.shape))
            max_shape = shapes.get("max_shape", list(input_tensor.shape))

            logger.info(f"Setting input shapes - min: {min_shape}, opt: {opt_shape}, max: {max_shape}")
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    else:
        # Default shapes based on input tensor
        input_shape = list(input_tensor.shape)
        logger.info(f"Using default input shape: {input_shape}")
        profile.set_shape("input", input_shape, input_shape, input_shape)


def _log_parser_errors(parser, logger: logging.Logger) -> None:
    """Log TensorRT parser errors."""
    logger.error("Failed to parse ONNX model")
    for error in range(parser.num_errors):
        logger.error(f"Parser error: {parser.get_error(error)}")


# ============================================================================
# Inference and Verification Functions
# ============================================================================


def run_pytorch_inference(
    model: torch.nn.Module, input_tensor: torch.Tensor, logger: logging.Logger
) -> Tuple[torch.Tensor, float]:
    """Run PyTorch inference on CPU for verification and return output with latency."""
    # Move to CPU to avoid GPU memory issues
    model_cpu = model.cpu()
    input_cpu = input_tensor.cpu()

    model_cpu.eval()
    with torch.no_grad():
        # Measure inference time
        start_time = time.perf_counter()
        output = model_cpu(input_cpu)
        end_time = time.perf_counter()

        latency = (end_time - start_time) * 1000  # Convert to milliseconds

        # Handle different output formats
        if hasattr(output, "output"):
            output = output.output
        elif isinstance(output, dict) and "output" in output:
            output = output["output"]

        if not isinstance(output, torch.Tensor):
            raise ValueError(f"Unexpected PyTorch output type: {type(output)}")

    logger.info(f"PyTorch inference latency: {latency:.2f} ms")
    logger.info(f"Output verification:")
    logger.info(f"  Output: {output.cpu().numpy()}")
    return output, latency


def run_onnx_inference(
    onnx_path: str,
    input_tensor: torch.Tensor,
    ref_output: torch.Tensor,
    logger: logging.Logger,
) -> bool:
    """Verify ONNX model output against PyTorch model."""
    # Clear GPU cache
    _clear_gpu_memory()

    # ONNX inference with timing
    providers = ["CPUExecutionProvider"]
    ort_session = ort.InferenceSession(onnx_path, providers=providers)
    onnx_input = {ort_session.get_inputs()[0].name: input_tensor.cpu().numpy()}

    start_time = time.perf_counter()
    onnx_output = ort_session.run(None, onnx_input)[0]
    end_time = time.perf_counter()
    onnx_latency = (end_time - start_time) * 1000

    logger.info(f"ONNX inference latency: {onnx_latency:.2f} ms")

    # Ensure onnx_output is numpy array before comparison
    if not isinstance(onnx_output, np.ndarray):
        logger.error(f"Unexpected ONNX output type: {type(onnx_output)}")
        return False

    # Compare outputs
    return _compare_outputs(ref_output.cpu().numpy(), onnx_output, "ONNX", logger)


def run_tensorrt_inference(
    tensorrt_path: str,
    input_tensor: torch.Tensor,
    ref_output: torch.Tensor,
    logger: logging.Logger,
) -> bool:
    """Verify TensorRT model output against PyTorch model."""
    # Clear GPU cache
    _clear_gpu_memory()

    # Load TensorRT engine
    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(trt_logger, "")
    runtime = trt.Runtime(trt_logger)

    with open(tensorrt_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        logger.error("Failed to deserialize TensorRT engine")
        return False

    # Run TensorRT inference with timing
    trt_output, latency = _run_tensorrt_inference(engine, input_tensor.cpu(), logger)
    logger.info(f"TensorRT inference latency: {latency:.2f} ms")

    # Compare outputs
    return _compare_outputs(ref_output.cpu().numpy(), trt_output, "TensorRT", logger)


def _run_tensorrt_inference(engine, input_tensor: torch.Tensor, logger: logging.Logger) -> Tuple[np.ndarray, float]:
    """Run TensorRT inference and return output with timing."""
    context = engine.create_execution_context()
    stream = cuda.Stream()
    start = cuda.Event()
    end = cuda.Event()

    # Get tensor names and shapes
    input_name, output_name = None, None
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            input_name = tensor_name
        elif engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
            output_name = tensor_name

    if input_name is None or output_name is None:
        raise RuntimeError("Could not find input/output tensor names")

    # Prepare arrays
    input_np = input_tensor.numpy().astype(np.float32)
    if not input_np.flags["C_CONTIGUOUS"]:
        input_np = np.ascontiguousarray(input_np)

    context.set_input_shape(input_name, input_np.shape)
    output_shape = context.get_tensor_shape(output_name)
    output_np = np.empty(output_shape, dtype=np.float32)
    if not output_np.flags["C_CONTIGUOUS"]:
        output_np = np.ascontiguousarray(output_np)

    # Allocate GPU memory
    d_input = cuda.mem_alloc(input_np.nbytes)
    d_output = cuda.mem_alloc(output_np.nbytes)

    try:
        # Set tensor addresses
        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output_name, int(d_output))

        # Run inference with timing
        cuda.memcpy_htod_async(d_input, input_np, stream)
        start.record(stream)
        context.execute_async_v3(stream_handle=stream.handle)
        end.record(stream)
        cuda.memcpy_dtoh_async(output_np, d_output, stream)
        stream.synchronize()

        latency = end.time_since(start)
        return output_np, latency

    finally:
        # Cleanup
        try:
            d_input.free()
            d_output.free()
        except:
            pass


def _compare_outputs(
    pytorch_output: np.ndarray, backend_output: np.ndarray, backend_name: str, logger: logging.Logger
) -> bool:
    """Compare outputs between PyTorch and backend."""
    if not isinstance(backend_output, np.ndarray):
        logger.error(f"Unexpected {backend_name} output type: {type(backend_output)}")
        return False

    max_diff = np.abs(pytorch_output - backend_output).max()
    mean_diff = np.abs(pytorch_output - backend_output).mean()

    logger.info(f"Output verification:")
    logger.info(f"  {backend_name} output: {backend_output}")
    logger.info(f"  Max difference with PyTorch: {max_diff:.6f}")
    logger.info(f"  Mean difference with PyTorch: {mean_diff:.6f}")

    success = max_diff < DEFAULT_VERIFICATION_TOLERANCE
    if not success:
        logger.warning(f"Large difference detected: {max_diff:.6f}")

    return success


def run_verification(
    model: torch.nn.Module,
    onnx_path: str,
    trt_path: Optional[str],
    input_tensors: Dict[str, torch.Tensor],
    logger: logging.Logger,
) -> None:
    """Run model verification for available backends."""
    _clear_gpu_memory()

    for key, input_tensor in input_tensors.items():
        logger.info("=" * 50)
        logger.info(f"Verifying {LABELS[key]} sample...")
        logger.info("-" * 50)
        logger.info("Verifying PyTorch model...")
        pytorch_output, pytorch_latency = run_pytorch_inference(model, input_tensor, logger)
        logger.info(
            f"PyTorch output for {LABELS[key]}: [SCORE_MISCALIBRATED, SCORE_CALIBRATED] = {pytorch_output.cpu().numpy()}"
        )
        score_calibrated = pytorch_output.cpu().numpy()[0, 1] - pytorch_output.cpu().numpy()[0, 0]
        if key == "0" and score_calibrated < 0:
            logger.info(f"Negative calibration score detected for {LABELS[key]} sample: {score_calibrated:.6f}")
        elif key == "0" and score_calibrated > 0:
            logger.warning(f"Positive calibration score detected for {LABELS[key]} sample: {score_calibrated:.6f}")
        elif key == "1" and score_calibrated > 0:
            logger.info(f"Positive calibration score detected for {LABELS[key]} sample: {score_calibrated:.6f}")
        elif key == "1" and score_calibrated < 0:
            logger.warning(f"Negative calibration score detected for {LABELS[key]} sample: {score_calibrated:.6f}")

        if onnx_path and osp.exists(onnx_path):
            logger.info("-" * 50)
            logger.info("Verifying ONNX model...")
            if run_onnx_inference(onnx_path, input_tensor, pytorch_output, logger):
                logger.info("ONNX model verification passed!")
            else:
                logger.error("ONNX model verification failed!")

        if trt_path and osp.exists(trt_path):
            logger.info("-" * 50)
            logger.info("Verifying TensorRT model...")
            if run_tensorrt_inference(trt_path, input_tensor, pytorch_output, logger):
                logger.info("TensorRT model verification passed!")
            else:
                logger.error("TensorRT model verification failed!")
    logger.info("=" * 50)


# ============================================================================
# Export Pipeline Helper Functions
# ============================================================================


def validate_and_prepare_paths(
    config: DeploymentConfig, work_dir: str, existing_onnx: Optional[str], logger: logging.Logger
) -> Tuple[Optional[str], Optional[str]]:
    """Determine and validate export paths for ONNX and TensorRT models.

    Returns:
        Tuple of (onnx_path, trt_path)
    """
    onnx_path = None
    trt_path = None

    if config.should_export_onnx:
        onnx_settings = config.onnx_settings
        onnx_path = osp.join(work_dir, onnx_settings["save_file"])

    if config.should_export_tensorrt:
        # Use existing ONNX if provided, otherwise use the one we'll export
        if existing_onnx and not config.should_export_onnx:
            onnx_path = existing_onnx
            if not osp.exists(onnx_path):
                logger.error(f"Provided ONNX file does not exist: {onnx_path}")
                return None, None
            logger.info(f"Using existing ONNX file: {onnx_path}")
        elif not onnx_path:
            # Need ONNX for TensorRT but neither export nor existing file specified
            logger.error("TensorRT export requires ONNX file. Set mode='both' or provide onnx_file in config.")
            return None, None

        # Set TensorRT output path
        onnx_settings = config.onnx_settings
        trt_file = onnx_settings["save_file"].replace(".onnx", ".engine")
        trt_path = osp.join(work_dir, trt_file)

    return onnx_path, trt_path


def export_models(
    model: torch.nn.Module,
    config: DeploymentConfig,
    onnx_path: Optional[str],
    trt_path: Optional[str],
    input_tensor_calibrated: torch.Tensor,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[bool, torch.device]:
    """Export models to ONNX and/or TensorRT formats.

    Returns:
        Tuple of (success, updated_device)
    """
    # Export ONNX
    if config.should_export_onnx and onnx_path:
        export_to_onnx(model, input_tensor_calibrated, onnx_path, config, logger)

    # Export TensorRT
    if config.should_export_tensorrt and trt_path and onnx_path:
        logger.info("Converting ONNX to TensorRT...")

        # Ensure CUDA device for TensorRT
        if device.type != "cuda":
            logger.warning("TensorRT requires CUDA device, switching to cuda")
            device = torch.device("cuda")

        success = export_to_tensorrt(onnx_path, trt_path, input_tensor_calibrated, config, logger)
        if success:
            logger.info(f"TensorRT conversion successful: {trt_path}")
        else:
            logger.error("TensorRT conversion failed")
            return False, device

    return True, device


def run_model_verification(
    model: torch.nn.Module,
    config: DeploymentConfig,
    onnx_path: Optional[str],
    trt_path: Optional[str],
    input_tensor_calibrated: torch.Tensor,
    input_tensor_miscalibrated: torch.Tensor,
    existing_onnx: Optional[str],
    logger: logging.Logger,
) -> None:
    """Run model verification for exported formats."""
    if not config.should_verify:
        return

    logger.info(
        "Running verification for miscalibrated and calibrated samples with "
        "an output array [SCORE_MISCALIBRATED, SCORE_CALIBRATED]..."
    )
    input_tensors = {"0": input_tensor_miscalibrated, "1": input_tensor_calibrated}

    # Only verify formats that were exported or provided
    onnx_path_for_verify = onnx_path if (config.should_export_onnx or existing_onnx) else None
    trt_path_for_verify = trt_path if config.should_export_tensorrt else None

    run_verification(model, onnx_path_for_verify, trt_path_for_verify, input_tensors, logger)
