"""
Model evaluation utilities for exported models.
"""

import gc
import logging
import os.path as osp
import time
from typing import Any, Tuple

import numpy as np
import onnxruntime as ort
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torch.nn.functional as F
from config import LABELS
from exporters import (
    _clear_gpu_memory,
    _create_transform_with_config,
    _load_info_pkl_file,
    _numpy_to_tensor,
    _run_tensorrt_inference,
    load_info_pkl_data,
)
from mmengine.config import Config


def _load_samples_from_info_pkl(info_pkl_path: str, num_samples: int, logger: logging.Logger) -> list:
    """Load and validate samples from info.pkl file."""
    try:
        samples_list = _load_info_pkl_file(info_pkl_path)
        logger.info(f"Loaded {len(samples_list)} samples from info.pkl")

        # Limit number of samples
        num_samples = min(num_samples, len(samples_list))
        logger.info(f"Evaluating {num_samples} samples")

        return samples_list[:num_samples]

    except Exception as e:
        logger.error(f"Failed to load info.pkl: {e}")
        raise


def _create_transform(model_cfg: Config):
    """Create and configure CalibrationClassificationTransform for evaluation."""
    return _create_transform_with_config(
        model_cfg=model_cfg,
        miscalibration_probability=0.0,
        projection_vis_dir=None,
        results_vis_dir=None,
        binary_save_dir=None,
    )


def _create_onnx_inference_func(model_path: str, device: str, logger: logging.Logger):
    """Create ONNX inference function with configured providers."""
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

    # Create session
    ort_session = ort.InferenceSession(model_path, providers=providers)
    logger.info(f"ONNX session using providers: {ort_session.get_providers()}")

    # Return inference function
    def inference_func(input_tensor):
        input_name = ort_session.get_inputs()[0].name
        onnx_input = {input_name: input_tensor.cpu().numpy()}
        start_time = time.perf_counter()
        onnx_output = ort_session.run(None, onnx_input)[0]
        latency = (time.perf_counter() - start_time) * 1000
        return onnx_output, latency

    return inference_func


def _create_tensorrt_inference_func(model_path: str, logger: logging.Logger):
    """Create TensorRT inference function with loaded engine."""
    logger.info(f"Using TensorRT model: {model_path}")

    # Load TensorRT engine
    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(trt_logger, "")
    runtime = trt.Runtime(trt_logger)

    with open(model_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine")

    # Return inference function
    def inference_func(input_tensor):
        return _run_tensorrt_inference(engine, input_tensor.cpu(), logger)

    return inference_func


def _process_single_sample(
    sample_idx: int,
    info_pkl_path: str,
    transform,
    inference_func,
    device: str,
    logger: logging.Logger,
    verbose: bool = False,
) -> Tuple[list, list, list, list]:
    """Process a single sample with both calibrated and miscalibrated versions.

    Returns:
        Tuple of (predictions, ground_truth, probabilities, latencies) for this sample
    """
    predictions = []
    ground_truth = []
    probabilities = []
    latencies = []

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
        input_tensor = _numpy_to_tensor(input_data_processed, device)

        # Run inference
        output_np, latency = inference_func(input_tensor)

        if output_np is None:
            logger.error(f"Failed to get output for sample {sample_idx}")
            continue

        # Convert logits to probabilities
        logits = torch.from_numpy(output_np)
        probs = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs.max().item()

        # Store results
        predictions.append(predicted_class)
        ground_truth.append(gt_label)
        probabilities.append(probs.cpu().numpy())
        latencies.append(latency)

        # Print sample results only in verbose mode
        if verbose:
            logger.info(
                f"Sample {sample_idx + 1} (GT={gt_label}): "
                f"Pred={predicted_class}, Confidence={confidence:.4f}, Latency={latency:.2f}ms"
            )

    return predictions, ground_truth, probabilities, latencies


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
    # Load samples
    samples_list = _load_samples_from_info_pkl(info_pkl_path, num_samples, logger)

    # Create transform
    transform = _create_transform(model_cfg)

    # Create inference function based on model type
    if model_type == "onnx":
        inference_func = _create_onnx_inference_func(model_path, device, logger)
    elif model_type == "tensorrt":
        inference_func = _create_tensorrt_inference_func(model_path, logger)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Process all samples
    all_predictions = []
    all_ground_truth = []
    all_probabilities = []
    all_latencies = []

    for sample_idx in range(len(samples_list)):
        if sample_idx % 100 == 0:
            logger.info(f"Processing sample {sample_idx + 1}/{len(samples_list)}")

        try:
            predictions, ground_truth, probabilities, latencies = _process_single_sample(
                sample_idx=sample_idx,
                info_pkl_path=info_pkl_path,
                transform=transform,
                inference_func=inference_func,
                device=device,
                logger=logger,
                verbose=verbose,
            )

            all_predictions.extend(predictions)
            all_ground_truth.extend(ground_truth)
            all_probabilities.extend(probabilities)
            all_latencies.extend(latencies)

            # Clear GPU memory periodically
            if model_type == "tensorrt" and sample_idx % 10 == 0:
                _clear_gpu_memory()

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


def get_models_to_evaluate(eval_cfg: dict, logger: logging.Logger) -> list:
    """Determine which models to evaluate based on config.

    Returns:
        List of (model_type, model_path) tuples
    """
    models_to_evaluate = []

    # Get model paths from config
    eval_onnx_path = eval_cfg.get("onnx_model")
    eval_trt_path = eval_cfg.get("tensorrt_model")

    # If both paths are None, skip evaluation
    if eval_onnx_path is None and eval_trt_path is None:
        logger.warning("Both onnx_model and tensorrt_model are None. Skipping evaluation.")
        logger.warning("To enable evaluation, specify at least one model path in the config.")
        return models_to_evaluate

    # Only evaluate models that are explicitly specified (not None)
    if eval_onnx_path is not None:
        if osp.exists(eval_onnx_path):
            models_to_evaluate.append(("onnx", eval_onnx_path))
            logger.info(f"Using config-specified ONNX model: {eval_onnx_path}")
        else:
            logger.warning(f"Config-specified ONNX model not found: {eval_onnx_path}")

    if eval_trt_path is not None:
        if osp.exists(eval_trt_path):
            models_to_evaluate.append(("tensorrt", eval_trt_path))
            logger.info(f"Using config-specified TensorRT model: {eval_trt_path}")
        else:
            logger.warning(f"Config-specified TensorRT model not found: {eval_trt_path}")

    return models_to_evaluate


def run_full_evaluation(
    models_to_evaluate: list,
    model_cfg: Config,
    info_pkl: str,
    device: Any,
    num_samples: int,
    verbose_mode: bool,
    logger: logging.Logger,
) -> None:
    """Run evaluation for all specified models."""
    if not models_to_evaluate:
        logger.error(
            "No models available for evaluation. Please export models first or specify model paths in config."
        )
        return

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
            _clear_gpu_memory()

        except Exception as e:
            logger.error(f"Evaluation failed for {model_type.upper()} model: {e}")
            import traceback

            logger.error(traceback.format_exc())
            continue
