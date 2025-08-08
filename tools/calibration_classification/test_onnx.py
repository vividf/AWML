import logging
import os
import time
from typing import Tuple

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.registry import DATASETS, TRANSFORMS
from mmpretrain.datasets.transforms.formatting import PackInputs

# Ensure the transforms are registered with the correct registry
# Also import the transform from mmpretrain registry to ensure it's registered
from mmpretrain.registry import DATASETS as MMPRETRAIN_DATASETS
from mmpretrain.registry import TRANSFORMS as MMPRETRAIN_TRANSFORMS

# Import the custom dataset and transform classes to register them
# These imports will register the classes with the mmengine registry
import autoware_ml.calibration_classification.datasets.t4_calibration_classification_dataset
import autoware_ml.calibration_classification.datasets.transforms.calibration_classification_transform

# Now we can import the classes directly
from autoware_ml.calibration_classification.datasets.t4_calibration_classification_dataset import (
    T4CalibrationClassificationDataset,
)
from autoware_ml.calibration_classification.datasets.transforms.calibration_classification_transform import (
    CalibrationClassificationTransform,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Manually register the transform and dataset with mmengine registry as well
try:
    from mmengine.registry import DATASETS as MMENGINE_DATASETS
    from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS

    if "CalibrationClassificationTransform" not in MMENGINE_TRANSFORMS:
        MMENGINE_TRANSFORMS.register_module(
            name="CalibrationClassificationTransform", module=CalibrationClassificationTransform
        )
        logger.info("Successfully registered CalibrationClassificationTransform with mmengine registry")
    if "T4CalibrationClassificationDataset" not in MMENGINE_DATASETS:
        MMENGINE_DATASETS.register_module(
            name="T4CalibrationClassificationDataset", module=T4CalibrationClassificationDataset
        )
        logger.info("Successfully registered T4CalibrationClassificationDataset with mmengine registry")
    if "PackInputs" not in MMENGINE_TRANSFORMS:
        MMENGINE_TRANSFORMS.register_module(name="PackInputs", module=PackInputs)
        logger.info("Successfully registered PackInputs with mmengine registry")
except Exception as e:
    logger.warning(f"Failed to register with mmengine registry: {e}")

# Configuration
deploy_cfg_path = "projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch_onnxruntime.py"
model_cfg_path = "projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb8-25e_j6gen2.py"
device = "cpu"
onnx_model_path = "/workspace/work_dirs/end2end_train_200_ptq.quant.onnx"

# Constants
DEFAULT_VERIFICATION_TOLERANCE = 1e-3
LABELS = {"0": "miscalibrated", "1": "calibrated"}


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def run_onnx_inference_direct(
    onnx_path: str,
    input_tensor: torch.Tensor,
    logger: logging.Logger,
) -> Tuple[np.ndarray, float]:
    """Run ONNX inference directly and return output and latency."""
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Convert input tensor to float32
    input_tensor = input_tensor.float()

    # Debug: Print input tensor info before preprocessing
    logger.debug(
        f"Input tensor before preprocessing - Shape: {input_tensor.shape}, Dtype: {input_tensor.dtype}, Min: {input_tensor.min():.4f}, Max: {input_tensor.max():.4f}"
    )

    # Add batch dimension if needed (ONNX expects 4D input: batch, channels, height, width)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        logger.debug(f"Added batch dimension - Shape: {input_tensor.shape}")

    # ONNX inference with timing
    providers = ["CPUExecutionProvider"]
    ort_session = ort.InferenceSession(onnx_path, providers=providers)

    # Debug: Print ONNX model input info
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    input_type = ort_session.get_inputs()[0].type
    logger.debug(f"ONNX model expects - Input name: {input_name}, Shape: {input_shape}, Type: {input_type}")

    onnx_input = {input_name: input_tensor.cpu().numpy().astype(np.float32)}

    start_time = time.perf_counter()
    onnx_output = ort_session.run(None, onnx_input)[0]
    end_time = time.perf_counter()
    onnx_latency = (end_time - start_time) * 1000

    logger.info(f"ONNX inference latency: {onnx_latency:.2f} ms")

    # Ensure onnx_output is numpy array
    if not isinstance(onnx_output, np.ndarray):
        logger.error(f"Unexpected ONNX output type: {type(onnx_output)}")
        return None, 0.0

    return onnx_output, onnx_latency


def main():
    """Main evaluation function."""
    logger = setup_logging()

    # Check if config files exist
    if not os.path.exists(deploy_cfg_path):
        raise FileNotFoundError(f"Deploy config file not found: {deploy_cfg_path}")
    if not os.path.exists(model_cfg_path):
        raise FileNotFoundError(f"Model config file not found: {model_cfg_path}")
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

    # Load model config
    model_cfg = Config.fromfile(model_cfg_path)

    # Build dataset
    dataset_cfg = model_cfg.test_dataloader.dataset

    # Try to build dataset with error handling
    try:
        dataset = MMPRETRAIN_DATASETS.build(dataset_cfg)
        logger.info(f"Test dataset created with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to build dataset with MMPRETRAIN_DATASETS: {e}")
        logger.info("Trying with mmengine DATASETS...")
        try:
            dataset = DATASETS.build(dataset_cfg)
            logger.info(f"Test dataset created with {len(dataset)} samples")
        except Exception as e2:
            logger.error(f"Failed to build dataset with mmengine DATASETS: {e2}")
            raise RuntimeError(f"Could not build dataset with either registry: {e}, {e2}")
    logger.info(f"Using ONNX model: {onnx_model_path}")

    # Lists to store results
    all_predictions = []
    all_ground_truth = []
    all_probabilities = []
    all_latencies = []

    # Evaluate entire dataset
    for sample_idx in range(len(dataset)):
        if sample_idx % 10 == 0:
            logger.info(f"Processing sample {sample_idx + 1}/{len(dataset)}")

        # Get a single sample from dataset
        data_sample = dataset[sample_idx]
        input_tensor = data_sample["inputs"]
        gt_label = data_sample["data_samples"].gt_label.item()

        # Debug: Print input tensor info
        if sample_idx < 3:
            logger.info(f"Sample {sample_idx + 1} input tensor:")
            logger.info(f"  Shape: {input_tensor.shape}")
            logger.info(f"  Dtype: {input_tensor.dtype}")
            logger.info(f"  Min: {input_tensor.min():.4f}, Max: {input_tensor.max():.4f}")
            # Convert to float for mean and std calculation
            input_tensor_float = input_tensor.float()
            logger.info(f"  Mean: {input_tensor_float.mean():.4f}, Std: {input_tensor_float.std():.4f}")

        # Run ONNX inference
        onnx_output, latency = run_onnx_inference_direct(onnx_model_path, input_tensor, logger)

        if onnx_output is None:
            logger.error(f"Failed to get ONNX output for sample {sample_idx}")
            continue

        # Convert logits to probabilities
        logits = torch.from_numpy(onnx_output)
        probabilities = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities.max().item()

        # Store results
        all_predictions.append(predicted_class)
        all_ground_truth.append(gt_label)
        all_probabilities.append(probabilities.cpu().numpy())
        all_latencies.append(latency)

        # Print first few samples for debugging
        if sample_idx < 3:
            logger.info(f"Sample {sample_idx + 1}: GT={gt_label}, Pred={predicted_class}, Confidence={confidence:.4f}")
            logger.info(f"  Logits: {logits}")
            logger.info(f"  Probabilities: {probabilities}")

    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_ground_truth = np.array(all_ground_truth)
    all_probabilities = np.array(all_probabilities)
    all_latencies = np.array(all_latencies)

    correct_predictions = (all_predictions == all_ground_truth).sum()
    total_samples = len(all_predictions)
    accuracy = correct_predictions / total_samples
    avg_latency = np.mean(all_latencies)

    # Print results
    logger.info(f"\n{'='*50}")
    logger.info(f"ONNX Model Evaluation Results")
    logger.info(f"{'='*50}")
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
            f"  Class {cls} ({LABELS[str(cls)]}): {cls_correct}/{cls_total} = {cls_accuracy:.4f} ({cls_accuracy*100:.2f}%)"
        )

    # Calculate average confidence
    avg_confidence = np.mean([prob.max() for prob in all_probabilities])
    logger.info(f"\nAverage confidence: {avg_confidence:.4f}")

    # Show confusion matrix
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"Predicted ->")
    logger.info(f"Actual    0    1")
    for true_cls in unique_classes:
        row = []
        for pred_cls in unique_classes:
            count = ((all_ground_truth == true_cls) & (all_predictions == pred_cls)).sum()
            row.append(f"{count:4d}")
        logger.info(f"  {true_cls}    {' '.join(row)}")

    logger.info(f"\nONNX model evaluation completed successfully!")
    logger.info(f"Model accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"Average latency: {avg_latency:.2f} ms")


if __name__ == "__main__":
    main()
