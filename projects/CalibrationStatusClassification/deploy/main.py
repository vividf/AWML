"""
CalibrationStatusClassification Model Deployment Script

This script exports CalibrationStatusClassification models to ONNX and TensorRT formats,
with comprehensive verification and performance benchmarking.

Features:
- ONNX export with optimization
- TensorRT conversion with precision policy support
- Dual verification (ONNX + TensorRT) on single samples
- Full model evaluation on multiple samples with metrics
- Performance benchmarking with latency statistics
- Confusion matrix and per-class accuracy analysis
"""

import argparse
import gc
import logging
import os
import os.path as osp
import pickle
import time
from typing import Any, Dict, Optional, Tuple

import mmengine
import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmpretrain.apis import get_model

from autoware_ml.calibration_classification.datasets.transforms.calibration_classification_transform import (
    CalibrationClassificationTransform,
)

# Constants
DEFAULT_VERIFICATION_TOLERANCE = 1e-3
DEFAULT_WORKSPACE_SIZE = 1 << 30  # 1 GB
EXPECTED_CHANNELS = 5  # RGB + Depth + Intensity
LABELS = {"0": "miscalibrated", "1": "calibrated"}

# Precision policy mapping
PRECISION_POLICIES = {
    "auto": {},  # No special flags, TensorRT decides
    "fp16": {"FP16": True},
    "fp32_tf32": {"TF32": True},  # TF32 for FP32 operations
    "explicit_int8": {"INT8": True},
    "strongly_typed": {"STRONGLY_TYPED": True},  # Network creation flag
}


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

    if sample_idx >= len(samples_list):
        raise ValueError(f"Sample index {sample_idx} out of range (0-{len(samples_list)-1})")

    sample = samples_list[sample_idx]

    # Validate sample structure
    required_keys = ["image", "lidar_points"]
    if not all(key in sample for key in required_keys):
        raise ValueError(f"Sample {sample_idx} has invalid structure. Required keys: {required_keys}")

    return sample


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

    # Get data_root from model config
    data_root = model_cfg.get("data_root", None)
    if data_root is None:
        raise ValueError("data_root not found in model configuration")

    # Create transform for deployment
    transform_config = model_cfg.get("transform_config", None)
    if transform_config is None:
        raise ValueError("transform_config not found in model configuration")

    transform = CalibrationClassificationTransform(
        transform_config=transform_config,
        mode="test",
        max_depth=model_cfg.get("max_depth", 128.0),
        dilation_size=model_cfg.get("dilation_size", 1),
        undistort=True,
        miscalibration_probability=miscalibration_probability,
        enable_augmentation=False,
        data_root=data_root,
        projection_vis_dir=model_cfg.get("test_projection_vis_dir", None),
        results_vis_dir=model_cfg.get("test_results_vis_dir", None),
        binary_save_dir=model_cfg.get("binary_save_dir", None),
    )

    # Apply transform
    results = transform.transform(sample_data)
    input_data_processed = results["fused_img"]  # (H, W, 5)

    # Convert to tensor
    input_tensor = torch.from_numpy(input_data_processed).permute(2, 0, 1).float()  # (5, H, W)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # (1, 5, H, W)

    return input_tensor


class DeploymentConfig:
    """Enhanced configuration container for deployment settings with validation."""

    def __init__(self, deploy_cfg: Config):
        self.deploy_cfg = deploy_cfg
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration structure and required fields."""
        if "export" not in self.deploy_cfg:
            raise ValueError(
                "Missing 'export' section in deploy config. "
                "Please update your config to the new format with 'export', 'runtime_io' sections."
            )

        if "runtime_io" not in self.deploy_cfg:
            raise ValueError("Missing 'runtime_io' section in deploy config.")

        # Validate export mode
        valid_modes = ["onnx", "trt", "both", "none"]
        mode = self.export_config.get("mode", "both")
        if mode not in valid_modes:
            raise ValueError(f"Invalid export mode '{mode}'. Must be one of {valid_modes}")

        # Validate precision policy if present
        backend_cfg = self.deploy_cfg.get("backend_config", {})
        common_cfg = backend_cfg.get("common_config", {})
        precision_policy = common_cfg.get("precision_policy", "auto")
        if precision_policy not in PRECISION_POLICIES:
            raise ValueError(
                f"Invalid precision_policy '{precision_policy}'. " f"Must be one of {list(PRECISION_POLICIES.keys())}"
            )

    @property
    def export_config(self) -> Dict:
        """Get export configuration (mode, verify, device, work_dir)."""
        return self.deploy_cfg.get("export", {})

    @property
    def runtime_io_config(self) -> Dict:
        """Get runtime I/O configuration (info_pkl, sample_idx, onnx_file)."""
        return self.deploy_cfg.get("runtime_io", {})

    @property
    def evaluation_config(self) -> Dict:
        """Get evaluation configuration (enabled, num_samples, verbose, model paths)."""
        return self.deploy_cfg.get("evaluation", {})

    @property
    def should_export_onnx(self) -> bool:
        """Check if ONNX export is requested."""
        mode = self.export_config.get("mode", "both")
        return mode in ["onnx", "both"]

    @property
    def should_export_tensorrt(self) -> bool:
        """Check if TensorRT export is requested."""
        mode = self.export_config.get("mode", "both")
        return mode in ["trt", "both"]

    @property
    def should_verify(self) -> bool:
        """Check if verification is requested."""
        return self.export_config.get("verify", False)

    @property
    def device(self) -> str:
        """Get device for export."""
        return self.export_config.get("device", "cuda:0")

    @property
    def work_dir(self) -> str:
        """Get working directory."""
        return self.export_config.get("work_dir", os.getcwd())

    @property
    def onnx_settings(self) -> Dict:
        """Get ONNX export settings."""
        onnx_config = self.deploy_cfg.get("onnx_config", {})
        return {
            "opset_version": onnx_config.get("opset_version", 16),
            "do_constant_folding": onnx_config.get("do_constant_folding", True),
            "input_names": onnx_config.get("input_names", ["input"]),
            "output_names": onnx_config.get("output_names", ["output"]),
            "dynamic_axes": onnx_config.get("dynamic_axes"),
            "export_params": onnx_config.get("export_params", True),
            "keep_initializers_as_inputs": onnx_config.get("keep_initializers_as_inputs", False),
            "save_file": onnx_config.get("save_file", "calibration_classifier.onnx"),
        }

    @property
    def tensorrt_settings(self) -> Dict:
        """Get TensorRT export settings with precision policy support."""
        backend_config = self.deploy_cfg.get("backend_config", {})
        common_config = backend_config.get("common_config", {})

        # Get precision policy
        precision_policy = common_config.get("precision_policy", "auto")
        policy_flags = PRECISION_POLICIES.get(precision_policy, {})

        return {
            "max_workspace_size": common_config.get("max_workspace_size", DEFAULT_WORKSPACE_SIZE),
            "precision_policy": precision_policy,
            "policy_flags": policy_flags,
            "model_inputs": backend_config.get("model_inputs", []),
        }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export CalibrationStatusClassification model to ONNX/TensorRT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("deploy_cfg", help="Deploy config path")
    parser.add_argument("model_cfg", help="Model config path")
    parser.add_argument(
        "checkpoint", nargs="?", default=None, help="Model checkpoint path (optional when mode='none')"
    )

    # Optional overrides
    parser.add_argument("--work-dir", help="Override output directory from config")
    parser.add_argument("--device", help="Override device from config")
    parser.add_argument("--log-level", default="INFO", choices=list(logging._nameToLevel.keys()), help="Logging level")
    parser.add_argument("--info-pkl", help="Override info.pkl path from config")
    parser.add_argument("--sample-idx", type=int, help="Override sample index from config")

    # Evaluation mode
    parser.add_argument("--evaluate", action="store_true", help="Run full evaluation on multiple samples")
    parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of samples to evaluate (only with --evaluate)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging during evaluation")

    return parser.parse_args()


def setup_logging(level: str) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(level=getattr(logging, level), format="%(levelname)s:%(name)s:%(message)s")
    return logging.getLogger("mmdeploy")


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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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


def evaluate_exported_model(
    model_path: str,
    model_type: str,
    model_cfg: Config,
    info_pkl_path: str,
    logger: logging.Logger,
    device: str = "cpu",
    num_samples: int = 10,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate exported model (ONNX or TensorRT) using info.pkl data.

    Args:
        model_path: Path to model file (.onnx or .engine)
        model_type: Type of model ("onnx" or "tensorrt")
        model_cfg: Model configuration
        info_pkl_path: Path to info.pkl file
        logger: Logger instance
        device: Device for preprocessing
        num_samples: Number of samples to evaluate
        verbose: Enable verbose logging

    Returns:
        Tuple of (predictions, ground_truth, probabilities, latencies)
    """

    # Load info.pkl data
    try:
        with open(info_pkl_path, "rb") as f:
            info_data = pickle.load(f)

        if "data_list" not in info_data:
            raise ValueError("Expected 'data_list' key in info.pkl")

        samples_list = info_data["data_list"]
        logger.info(f"Loaded {len(samples_list)} samples from info.pkl")

        # Limit number of samples for evaluation
        num_samples = min(num_samples, len(samples_list))
        logger.info(f"Evaluating {num_samples} samples")

    except Exception as e:
        logger.error(f"Failed to load info.pkl: {e}")
        raise

    # Initialize transform once and reuse it for all samples
    data_root = model_cfg.get("data_root", None)
    if data_root is None:
        raise ValueError("data_root not found in model configuration")

    transform_config = model_cfg.get("transform_config", None)
    if transform_config is None:
        raise ValueError("transform_config not found in model configuration")

    transform = CalibrationClassificationTransform(
        transform_config=transform_config,
        mode="test",
        max_depth=model_cfg.get("max_depth", 128.0),
        dilation_size=model_cfg.get("dilation_size", 1),
        undistort=True,
        miscalibration_probability=0.0,  # Will be updated per sample
        enable_augmentation=False,
        data_root=data_root,
        projection_vis_dir=None,
        results_vis_dir=None,
        binary_save_dir=None,
    )

    # Initialize inference engine
    if model_type == "onnx":
        logger.info(f"Using ONNX model: {model_path}")
        logger.info("Creating ONNX InferenceSession...")

        # Configure execution providers
        if device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.info("Attempting to use CUDA acceleration for ONNX inference...")
            try:
                available_providers = ort.get_available_providers()
                if "CUDAExecutionProvider" not in available_providers:
                    logger.warning(
                        f"CUDAExecutionProvider not available. Available: {available_providers}. "
                        "Install onnxruntime-gpu for CUDA acceleration"
                    )
                    providers = ["CPUExecutionProvider"]
                else:
                    logger.info("CUDAExecutionProvider is available")
            except Exception as e:
                logger.warning(f"Error checking CUDA provider: {e}. Falling back to CPU")
                providers = ["CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
            logger.info("Using CPU for ONNX inference")

        # Create session once and reuse it
        ort_session = ort.InferenceSession(model_path, providers=providers)
        logger.info(f"ONNX session using providers: {ort_session.get_providers()}")

        # Define inference function
        def inference_func(input_tensor):
            input_name = ort_session.get_inputs()[0].name
            onnx_input = {input_name: input_tensor.cpu().numpy()}
            start_time = time.perf_counter()
            onnx_output = ort_session.run(None, onnx_input)[0]
            latency = (time.perf_counter() - start_time) * 1000
            return onnx_output, latency

    elif model_type == "tensorrt":
        logger.info(f"Using TensorRT model: {model_path}")

        # Load TensorRT engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(trt_logger, "")
        runtime = trt.Runtime(trt_logger)

        with open(model_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        # Define inference function
        def inference_func(input_tensor):
            return _run_tensorrt_inference(engine, input_tensor.cpu(), logger)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Lists to store results
    all_predictions = []
    all_ground_truth = []
    all_probabilities = []
    all_latencies = []

    # Evaluate samples
    for sample_idx in range(num_samples):
        if sample_idx % 100 == 0:
            logger.info(f"Processing sample {sample_idx + 1}/{num_samples}")

        try:
            # Test both calibrated and miscalibrated versions
            for miscalibration_prob, expected_label in [(0.0, 1), (1.0, 0)]:
                # Update transform's miscalibration probability
                transform.miscalibration_probability = miscalibration_prob

                # Load and preprocess sample
                sample_data = load_info_pkl_data(info_pkl_path, sample_idx)
                sample_data["sample_idx"] = sample_idx

                results = transform.transform(sample_data)
                input_data_processed = results["fused_img"]
                gt_label = results["gt_label"]

                # Convert to tensor
                input_tensor = torch.from_numpy(input_data_processed).permute(2, 0, 1).float()
                input_tensor = input_tensor.unsqueeze(0).to(device)

                # Run inference
                output_np, latency = inference_func(input_tensor)

                if output_np is None:
                    logger.error(f"Failed to get output for sample {sample_idx}")
                    continue

                # Convert logits to probabilities
                logits = torch.from_numpy(output_np)
                probabilities = F.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities.max().item()

                # Store results
                all_predictions.append(predicted_class)
                all_ground_truth.append(gt_label)
                all_probabilities.append(probabilities.cpu().numpy())
                all_latencies.append(latency)

                # Print sample results only in verbose mode
                if verbose:
                    logger.info(
                        f"Sample {sample_idx + 1} (GT={gt_label}): "
                        f"Pred={predicted_class}, Confidence={confidence:.4f}, Latency={latency:.2f}ms"
                    )

            # Clear GPU memory periodically
            if model_type == "tensorrt" and sample_idx % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            logger.error(f"Error processing sample {sample_idx}: {e}")
            continue

    return (
        np.array(all_predictions),
        np.array(all_ground_truth),
        np.array(all_probabilities),
        np.array(all_latencies),
    )


def print_evaluation_results(
    all_predictions: np.ndarray,
    all_ground_truth: np.ndarray,
    all_probabilities: np.ndarray,
    all_latencies: np.ndarray,
    model_type: str,
    logger: logging.Logger,
) -> None:
    """Print comprehensive evaluation results with metrics and statistics."""

    if len(all_predictions) == 0:
        logger.error("No samples were processed successfully.")
        return

    correct_predictions = (all_predictions == all_ground_truth).sum()
    total_samples = len(all_predictions)
    accuracy = correct_predictions / total_samples
    avg_latency = np.mean(all_latencies)

    # Print header
    logger.info(f"\n{'='*60}")
    logger.info(f"{model_type.upper()} Model Evaluation Results")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Correct predictions: {correct_predictions}")
    logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"Average latency: {avg_latency:.2f} ms")

    # Calculate per-class accuracy
    unique_classes = np.unique(all_ground_truth)
    logger.info(f"\nPer-class accuracy:")
    for cls in unique_classes:
        cls_mask = all_ground_truth == cls
        cls_correct = (all_predictions[cls_mask] == all_ground_truth[cls_mask]).sum()
        cls_total = cls_mask.sum()
        cls_accuracy = cls_correct / cls_total if cls_total > 0 else 0
        logger.info(
            f"  Class {cls} ({LABELS[str(cls)]}): "
            f"{cls_correct}/{cls_total} = {cls_accuracy:.4f} ({cls_accuracy*100:.2f}%)"
        )

    # Calculate average confidence
    avg_confidence = np.mean([prob.max() for prob in all_probabilities])
    logger.info(f"\nAverage confidence: {avg_confidence:.4f}")

    # Calculate latency statistics
    min_latency = np.min(all_latencies)
    max_latency = np.max(all_latencies)
    std_latency = np.std(all_latencies)
    p50_latency = np.percentile(all_latencies, 50)
    p95_latency = np.percentile(all_latencies, 95)
    p99_latency = np.percentile(all_latencies, 99)

    logger.info(f"\nLatency Statistics:")
    logger.info(f"  Average: {avg_latency:.2f} ms")
    logger.info(f"  Median (P50): {p50_latency:.2f} ms")
    logger.info(f"  P95: {p95_latency:.2f} ms")
    logger.info(f"  P99: {p99_latency:.2f} ms")
    logger.info(f"  Min: {min_latency:.2f} ms")
    logger.info(f"  Max: {max_latency:.2f} ms")
    logger.info(f"  Std: {std_latency:.2f} ms")

    # Show confusion matrix
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"{'':>10} Predicted")
    logger.info(f"{'Actual':>10} {'0 (misc)':>10} {'1 (calib)':>10}")
    logger.info(f"{'-'*32}")
    for true_cls in unique_classes:
        row_label = f"{true_cls} ({LABELS[str(true_cls)][:4]})"
        row = [f"{row_label:>10}"]
        for pred_cls in unique_classes:
            count = ((all_ground_truth == true_cls) & (all_predictions == pred_cls)).sum()
            row.append(f"{count:>10}")
        logger.info(" ".join(row))

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation Summary:")
    logger.info(f"  Model Type: {model_type.upper()}")
    logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"  Avg Latency: {avg_latency:.2f} ms")
    logger.info(f"  Throughput: {1000/avg_latency:.2f} samples/sec")
    logger.info(f"{'='*60}\n")


def main():
    """Main deployment function."""
    args = parse_args()
    logger = setup_logging(args.log_level)

    # Load configurations
    logger.info(f"Loading deploy config from: {args.deploy_cfg}")
    deploy_cfg = Config.fromfile(args.deploy_cfg)
    config = DeploymentConfig(deploy_cfg)

    logger.info(f"Loading model config from: {args.model_cfg}")
    model_cfg = Config.fromfile(args.model_cfg)

    # Get settings from config with CLI overrides
    work_dir = args.work_dir if args.work_dir else config.work_dir
    device = args.device if args.device else config.device
    info_pkl = args.info_pkl if args.info_pkl else config.runtime_io_config.get("info_pkl")
    sample_idx = args.sample_idx if args.sample_idx is not None else config.runtime_io_config.get("sample_idx", 0)
    existing_onnx = config.runtime_io_config.get("onnx_file")

    # Validate required parameters
    if not info_pkl:
        logger.error("info_pkl path must be provided either in config or via --info-pkl")
        return

    # Setup working directory
    mmengine.mkdir_or_exist(osp.abspath(work_dir))
    logger.info(f"Working directory: {work_dir}")
    logger.info(f"Device: {device}")
    export_mode = config.export_config.get("mode", "both")
    logger.info(f"Export mode: {export_mode}")

    # Check if eval-only mode
    is_eval_only = export_mode == "none"

    # Validate checkpoint requirement
    if not is_eval_only and not args.checkpoint:
        logger.error("Checkpoint is required when export mode is not 'none'")
        return

    # Determine export paths
    onnx_path = None
    trt_path = None

    if not is_eval_only:
        if config.should_export_onnx:
            onnx_settings = config.onnx_settings
            onnx_path = osp.join(work_dir, onnx_settings["save_file"])

        if config.should_export_tensorrt:
            # Use existing ONNX if provided, otherwise use the one we'll export
            if existing_onnx and not config.should_export_onnx:
                onnx_path = existing_onnx
                if not osp.exists(onnx_path):
                    logger.error(f"Provided ONNX file does not exist: {onnx_path}")
                    return
                logger.info(f"Using existing ONNX file: {onnx_path}")
            elif not onnx_path:
                # Need ONNX for TensorRT but neither export nor existing file specified
                logger.error("TensorRT export requires ONNX file. Set mode='both' or provide onnx_file in config.")
                return

            # Set TensorRT output path
            onnx_settings = config.onnx_settings
            trt_file = onnx_settings["save_file"].replace(".onnx", ".engine")
            trt_path = osp.join(work_dir, trt_file)

        # Load model
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        device = torch.device(device)
        model = get_model(model_cfg, args.checkpoint, device=device)

        # Load sample data
        logger.info(f"Loading sample data from info.pkl: {info_pkl}")
        input_tensor_calibrated = load_sample_data_from_info_pkl(info_pkl, model_cfg, 0.0, sample_idx, device=device)
        input_tensor_miscalibrated = load_sample_data_from_info_pkl(
            info_pkl, model_cfg, 1.0, sample_idx, device=device
        )

        # Export models
        if config.should_export_onnx:
            export_to_onnx(model, input_tensor_calibrated, onnx_path, config, logger)

        if config.should_export_tensorrt:
            logger.info("Converting ONNX to TensorRT...")

            # Ensure CUDA device for TensorRT
            if device.type != "cuda":
                logger.warning("TensorRT requires CUDA device, switching to cuda")
                device = torch.device("cuda")
                input_tensor_calibrated = input_tensor_calibrated.to(device)
                input_tensor_miscalibrated = input_tensor_miscalibrated.to(device)

            success = export_to_tensorrt(onnx_path, trt_path, input_tensor_calibrated, config, logger)
            if success:
                logger.info(f"TensorRT conversion successful: {trt_path}")
            else:
                logger.error("TensorRT conversion failed")

        # Run verification if requested
        if config.should_verify:
            logger.info(
                "Running verification for miscalibrated and calibrated samples with an output array [SCORE_MISCALIBRATED, SCORE_CALIBRATED]..."
            )
            input_tensors = {"0": input_tensor_miscalibrated, "1": input_tensor_calibrated}

            # Only verify formats that were exported or provided
            onnx_path_for_verify = onnx_path if (config.should_export_onnx or existing_onnx) else None
            trt_path_for_verify = trt_path if config.should_export_tensorrt else None

            run_verification(model, onnx_path_for_verify, trt_path_for_verify, input_tensors, logger)
    else:
        logger.info("Evaluation-only mode: Skipping model loading and export")

    # Get evaluation settings from config and CLI
    eval_cfg = config.evaluation_config
    should_evaluate = args.evaluate or eval_cfg.get("enabled", False)
    num_samples = args.num_samples if args.num_samples != 10 else eval_cfg.get("num_samples", 10)
    verbose_mode = args.verbose or eval_cfg.get("verbose", False)

    # Run full evaluation if requested
    if should_evaluate:
        logger.info(f"\n{'='*60}")
        logger.info("Starting full model evaluation...")
        logger.info(f"{'='*60}")

        # Determine which models to evaluate
        models_to_evaluate = []

        # Get model paths from config
        eval_onnx_path = eval_cfg.get("onnx_model")
        eval_trt_path = eval_cfg.get("tensorrt_model")

        # If both paths are None, skip evaluation (no auto-detection)
        if eval_onnx_path is None and eval_trt_path is None:
            logger.warning("Both onnx_model and tensorrt_model are None. Skipping evaluation.")
            logger.warning("To enable evaluation, specify at least one model path in the config.")
        else:
            # Only evaluate models that are explicitly specified (not None)
            if eval_onnx_path is not None:
                # Evaluate ONNX model
                if osp.exists(eval_onnx_path):
                    models_to_evaluate.append(("onnx", eval_onnx_path))
                    logger.info(f"Using config-specified ONNX model: {eval_onnx_path}")
                else:
                    logger.warning(f"Config-specified ONNX model not found: {eval_onnx_path}")

            if eval_trt_path is not None:
                # Evaluate TensorRT model
                if osp.exists(eval_trt_path):
                    models_to_evaluate.append(("tensorrt", eval_trt_path))
                    logger.info(f"Using config-specified TensorRT model: {eval_trt_path}")
                else:
                    logger.warning(f"Config-specified TensorRT model not found: {eval_trt_path}")

        if not models_to_evaluate:
            logger.error(
                "No models available for evaluation. Please export models first or specify model paths in config."
            )
        else:
            # Evaluate each model
            for model_type, model_path in models_to_evaluate:
                logger.info(f"\nEvaluating {model_type.upper()} model...")
                logger.info(f"Model path: {model_path}")
                logger.info(f"Number of samples: {num_samples}")
                logger.info(f"Verbose mode: {verbose_mode}")

                try:
                    # Run evaluation
                    predictions, ground_truth, probabilities, latencies = evaluate_exported_model(
                        model_path=model_path,
                        model_type=model_type,
                        model_cfg=model_cfg,
                        info_pkl_path=info_pkl,
                        logger=logger,
                        device=device.type if isinstance(device, torch.device) else device,
                        num_samples=num_samples,
                        verbose=verbose_mode,
                    )

                    # Print results
                    print_evaluation_results(predictions, ground_truth, probabilities, latencies, model_type, logger)

                    # Cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                except Exception as e:
                    logger.error(f"Evaluation failed for {model_type.upper()} model: {e}")
                    import traceback

                    logger.error(traceback.format_exc())
                    continue

    logger.info("Deployment completed successfully!")

    # Log what was exported
    if not is_eval_only:
        exported_formats = []
        if config.should_export_onnx:
            exported_formats.append("ONNX")
        if config.should_export_tensorrt:
            exported_formats.append("TensorRT")

        if exported_formats:
            logger.info(f"Exported formats: {', '.join(exported_formats)}")
    else:
        logger.info("Evaluation-only mode: No models were exported")


if __name__ == "__main__":
    main()
