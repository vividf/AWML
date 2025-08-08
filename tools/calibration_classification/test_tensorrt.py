import gc
import logging
import os
import signal
import sys
import time
from typing import Tuple

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torch.nn.functional as F
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
from mmengine.registry import DATASETS, TRANSFORMS
from mmpretrain.datasets.transforms.formatting import PackInputs

# Also import the transform from mmpretrain registry to ensure it's registered
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


def signal_handler(signum, frame):
    """Handle segmentation faults and other signals gracefully."""
    print(f"\nReceived signal {signum}. Cleaning up...")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Cleanup completed.")
    except:
        pass
    sys.exit(1)


# Register signal handlers
signal.signal(signal.SIGSEGV, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def _run_tensorrt_inference(engine, input_tensor: torch.Tensor, logger: logging.Logger) -> Tuple[np.ndarray, float]:
    """Run TensorRT inference and return output with timing."""
    context = None
    stream = None
    start = None
    end = None
    d_input = None
    d_output = None

    try:
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

        # Validate input shape
        expected_shape = engine.get_tensor_shape(input_name)
        if input_np.shape != tuple(expected_shape):
            logger.warning(f"Input shape mismatch: expected {expected_shape}, got {input_np.shape}")
            # Try to reshape if possible
            if len(input_np.shape) == len(expected_shape):
                try:
                    input_np = input_np.reshape(expected_shape)
                    logger.info(f"Reshaped input to {input_np.shape}")
                except Exception as e:
                    raise RuntimeError(f"Cannot reshape input from {input_np.shape} to {expected_shape}: {e}")
            elif len(input_np.shape) == len(expected_shape) - 1 and expected_shape[0] == 1:
                # Add batch dimension if missing
                try:
                    input_np = input_np.reshape(expected_shape)
                    logger.info(f"Added batch dimension to input: {input_np.shape}")
                except Exception as e:
                    raise RuntimeError(
                        f"Cannot add batch dimension to input from {input_np.shape} to {expected_shape}: {e}"
                    )
            else:
                raise RuntimeError(
                    f"Input shape mismatch: expected {expected_shape}, got {input_np.shape}. Please ensure input has correct batch dimension."
                )

        context.set_input_shape(input_name, input_np.shape)
        output_shape = context.get_tensor_shape(output_name)
        output_np = np.empty(output_shape, dtype=np.float32)
        if not output_np.flags["C_CONTIGUOUS"]:
            output_np = np.ascontiguousarray(output_np)

        # Allocate GPU memory
        d_input = cuda.mem_alloc(input_np.nbytes)
        d_output = cuda.mem_alloc(output_np.nbytes)

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

    except Exception as e:
        logger.error(f"TensorRT inference failed: {e}")
        raise
    finally:
        # Cleanup with better error handling
        try:
            if d_input is not None:
                d_input.free()
        except Exception as e:
            logger.warning(f"Failed to free input memory: {e}")

        try:
            if d_output is not None:
                d_output.free()
        except Exception as e:
            logger.warning(f"Failed to free output memory: {e}")

        # Note: Don't try to free stream, start, end, or context as they are managed by TensorRT
        # and may cause issues if freed manually


def main():
    """Main function to run TensorRT evaluation."""
    try:
        # Use the correct deploy and model config files for TensorRT
        deploy_cfg = "projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch_tensorrt.py"
        model_cfg = "projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb8-25e_j6gen2.py"
        device = "cuda"  # TensorRT requires CUDA
        backend_model = ["/workspace/work_dirs/end2end_train_200_ptq.engine"]
        # backend_model = ['/workspace/work_dirs/end2end.engine']

        # Check if config files exist
        if not os.path.exists(deploy_cfg):
            raise FileNotFoundError(f"Deploy config file not found: {deploy_cfg}")
        if not os.path.exists(model_cfg):
            raise FileNotFoundError(f"Model config file not found: {model_cfg}")

        # read deploy_cfg and model_cfg
        deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

        # Check if TensorRT model exists
        if not os.path.exists(backend_model[0]):
            print(f"Warning: TensorRT model not found at {backend_model[0]}")
            print("You need to export the model to TensorRT first using the deploy script.")
            print(
                "Example: python projects/CalibrationStatusClassification/deploy/main.py --deploy-cfg projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch_tensorrt.py --model-cfg projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb8-25e_j6gen2.py --checkpoint your_checkpoint.pth"
            )
            exit(1)

        # Load TensorRT engine directly with error handling for version compatibility
        logger = logging.getLogger(__name__)
        trt_logger = trt.Logger(trt.Logger.WARNING)

        try:
            with open(backend_model[0], "rb") as f:
                engine_bytes = f.read()
            engine = trt.Runtime(trt_logger).deserialize_cuda_engine(engine_bytes)
            if engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
        except Exception as e:
            print(f"Error loading TensorRT engine: {e}")
            print("This might be due to TensorRT version incompatibility.")
            print("Please rebuild the TensorRT engine with the current TensorRT version.")
            print("You can do this by running the deployment script again:")
            print(
                "python projects/CalibrationStatusClassification/deploy/main.py --deploy-cfg projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch_tensorrt.py --model-cfg projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb8-25e_j6gen2.py --checkpoint your_checkpoint.pth"
            )
            exit(1)

        print("Testing TensorRT model with entire test dataset...")

        # Build dataset directly from config
        dataset_cfg = model_cfg.test_dataloader.dataset

        # Check if the dataset type is registered
        if "T4CalibrationClassificationDataset" not in DATASETS:
            print("Warning: T4CalibrationClassificationDataset not found in registry.")
            print("Trying to register it manually...")
            try:
                # Manually register the dataset class
                DATASETS.register_module(
                    name="T4CalibrationClassificationDataset", module=T4CalibrationClassificationDataset
                )
                print("Successfully registered T4CalibrationClassificationDataset")
            except Exception as e:
                print(f"Failed to register dataset: {e}")
                print("Please check if the dataset class is properly imported.")
                exit(1)

        # Check if the transform is registered
        if "CalibrationClassificationTransform" not in TRANSFORMS:
            print("Warning: CalibrationClassificationTransform not found in registry.")
            print("Trying to register it manually...")
            try:
                # Manually register the transform class
                TRANSFORMS.register_module(
                    name="CalibrationClassificationTransform", module=CalibrationClassificationTransform
                )
                print("Successfully registered CalibrationClassificationTransform")
            except Exception as e:
                print(f"Failed to register transform: {e}")
                print("Please check if the transform class is properly imported.")
                exit(1)

        # Also check if it's registered in mmpretrain registry
        if "CalibrationClassificationTransform" not in MMPRETRAIN_TRANSFORMS:
            print("Warning: CalibrationClassificationTransform not found in mmpretrain registry.")
            print("Trying to register it manually...")
            try:
                # Manually register the transform class in mmpretrain registry
                MMPRETRAIN_TRANSFORMS.register_module(
                    name="CalibrationClassificationTransform", module=CalibrationClassificationTransform
                )
                print("Successfully registered CalibrationClassificationTransform in mmpretrain registry")
            except Exception as e:
                print(f"Failed to register transform in mmpretrain registry: {e}")
                print("Please check if the transform class is properly imported.")
                exit(1)

        # Check if PackInputs is registered
        if "PackInputs" not in TRANSFORMS:
            print("Warning: PackInputs not found in registry.")
            print("Trying to register it manually...")
            try:
                # Manually register the PackInputs class
                TRANSFORMS.register_module(name="PackInputs", module=PackInputs)
                print("Successfully registered PackInputs")
            except Exception as e:
                print(f"Failed to register PackInputs: {e}")
                print("Please check if PackInputs is properly imported.")
                exit(1)

        # Also check if PackInputs is registered in mmpretrain registry
        if "PackInputs" not in MMPRETRAIN_TRANSFORMS:
            print("Warning: PackInputs not found in mmpretrain registry.")
            print("Trying to register it manually...")
            try:
                # Manually register the PackInputs class in mmpretrain registry
                MMPRETRAIN_TRANSFORMS.register_module(name="PackInputs", module=PackInputs)
                print("Successfully registered PackInputs in mmpretrain registry")
            except Exception as e:
                print(f"Failed to register PackInputs in mmpretrain registry: {e}")
                print("Please check if PackInputs is properly imported.")
                exit(1)

        # Try to build dataset, but provide fallback if it fails
        try:
            dataset = DATASETS.build(dataset_cfg)
            print(f"Test dataset created with {len(dataset)} samples")
            use_dataset = True
        except Exception as e:
            print(f"Failed to build dataset: {e}")
            print("Dataset configuration:")
            print(dataset_cfg)
            print("\nTrying fallback approach with simple test...")
            use_dataset = False

        # Lists to store predictions and ground truth
        all_predictions = []
        all_ground_truth = []
        all_probabilities = []
        all_latencies = []

        if use_dataset:
            # Evaluate entire dataset
            print(f"Starting evaluation with {len(dataset)} real samples...")
            for sample_idx in range(len(dataset)):
                if sample_idx % 10 == 0:
                    print(f"Processing sample {sample_idx + 1}/{len(dataset)}")

                try:
                    # Get a single sample from dataset (keeping existing data acquisition)
                    data_sample = dataset[sample_idx]

                    # Get input tensor (keeping existing data format)
                    input_tensor = data_sample["inputs"]  # This is already a torch.Tensor

                    # Add batch dimension if needed (TensorRT expects batch dimension)
                    if input_tensor.dim() == 3:  # (C, H, W) -> (1, C, H, W)
                        original_shape = input_tensor.shape
                        input_tensor = input_tensor.unsqueeze(0)
                        print(f"Added batch dimension: {original_shape} -> {input_tensor.shape}")

                    # Run TensorRT inference using the provided method
                    output_np, latency = _run_tensorrt_inference(engine, input_tensor, logger)

                    # Convert output to torch tensor for processing
                    output_tensor = torch.from_numpy(output_np)

                    # Apply softmax to convert logits to probabilities
                    probabilities = F.softmax(output_tensor, dim=-1)
                    predicted_class = torch.argmax(probabilities, dim=-1).item()
                    confidence = probabilities.max().item()

                    # Extract ground truth label
                    gt_label = data_sample["data_samples"].gt_label.item()

                    # Store results
                    all_predictions.append(predicted_class)
                    all_ground_truth.append(gt_label)
                    all_probabilities.append(probabilities.numpy())
                    all_latencies.append(latency)

                    # Print first few samples for debugging
                    if sample_idx < 3:
                        print(
                            f"Sample {sample_idx + 1}: GT={gt_label}, Pred={predicted_class}, Confidence={confidence:.4f}, Latency={latency:.2f}ms"
                        )
                        print(f"  Input shape: {input_tensor.shape}")
                        print(f"  Output shape: {output_np.shape}")
                        print(f"  Probabilities: {probabilities}")

                    # Clear GPU memory periodically
                    if sample_idx % 50 == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

                except Exception as e:
                    print(f"Error processing sample {sample_idx}: {e}")
                    continue
        else:
            # Fallback: Test with a simple dummy input
            print("Testing TensorRT model with dummy input...")
            print("Note: This is using dummy data. To use real data, please check the dataset configuration.")

            # Get the expected input shape from the TensorRT engine
            input_shape = None
            for i in range(engine.num_io_tensors):
                tensor_name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    # Get the optimal shape for this input
                    input_shape = engine.get_tensor_shape(tensor_name)
                    print(f"Found input tensor '{tensor_name}' with shape: {input_shape}")
                    break

            if input_shape is None:
                # Fallback to the shape we saw in the error message
                input_shape = [1, 5, 1860, 2880]
                print(f"Could not determine input shape from engine, using fallback: {input_shape}")
            else:
                print(f"Using input shape from engine: {input_shape}")

            # Create a dummy input tensor with the expected shape
            dummy_input = torch.randn(*input_shape, dtype=torch.float32)
            print(f"Created dummy input with shape: {dummy_input.shape}")

            try:
                # Run TensorRT inference
                output_np, latency = _run_tensorrt_inference(engine, dummy_input, logger)

                # Convert output to torch tensor for processing
                output_tensor = torch.from_numpy(output_np)

                # Apply softmax to convert logits to probabilities
                probabilities = F.softmax(output_tensor, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities.max().item()

                print(f"Dummy test: Pred={predicted_class}, Confidence={confidence:.4f}, Latency={latency:.2f}ms")
                print(f"  Output shape: {output_np.shape}")
                print(f"  Probabilities: {probabilities}")

                # Store results for consistency
                all_predictions.append(predicted_class)
                all_ground_truth.append(0)  # Dummy ground truth
                all_probabilities.append(probabilities.numpy())
                all_latencies.append(latency)

            except Exception as e:
                print(f"Error in dummy test: {e}")
                print("This might be due to input shape mismatch or TensorRT engine issues.")
                print("Please check the TensorRT engine was built with the correct input dimensions.")
                exit(1)

        # Calculate accuracy
        all_predictions = np.array(all_predictions)
        all_ground_truth = np.array(all_ground_truth)
        all_probabilities = np.array(all_probabilities)
        all_latencies = np.array(all_latencies)

        if len(all_predictions) > 0:
            correct_predictions = (all_predictions == all_ground_truth).sum()
            total_samples = len(all_predictions)
            accuracy = correct_predictions / total_samples

            print(f"\n{'='*50}")
            print(f"TensorRT Model Evaluation Results")
            print(f"{'='*50}")
            print(f"Total samples: {total_samples}")
            print(f"Correct predictions: {correct_predictions}")
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

            # Calculate per-class accuracy (only if we have multiple samples and classes)
            if total_samples > 1:
                unique_classes = np.unique(all_ground_truth)
                print(f"\nPer-class accuracy:")
                for cls in unique_classes:
                    cls_mask = all_ground_truth == cls
                    cls_correct = (all_predictions[cls_mask] == all_ground_truth[cls_mask]).sum()
                    cls_total = cls_mask.sum()
                    cls_accuracy = cls_correct / cls_total if cls_total > 0 else 0
                    print(f"  Class {cls}: {cls_correct}/{cls_total} = {cls_accuracy:.4f} ({cls_accuracy*100:.2f}%)")

            # Calculate average confidence
            avg_confidence = np.mean([prob.max() for prob in all_probabilities])
            print(f"\nAverage confidence: {avg_confidence:.4f}")

            # Calculate latency statistics
            avg_latency = np.mean(all_latencies)
            min_latency = np.min(all_latencies)
            max_latency = np.max(all_latencies)
            std_latency = np.std(all_latencies)

            print(f"\nLatency Statistics:")
            print(f"  Average latency: {avg_latency:.2f} ms")
            print(f"  Min latency: {min_latency:.2f} ms")
            print(f"  Max latency: {max_latency:.2f} ms")
            print(f"  Std latency: {std_latency:.2f} ms")

            # Show confusion matrix (only if we have multiple samples)
            if total_samples > 1:
                print(f"\nConfusion Matrix:")
                print(f"Predicted ->")
                print(f"Actual    0    1")
                unique_classes = np.unique(all_ground_truth)
                for true_cls in unique_classes:
                    row = []
                    for pred_cls in unique_classes:
                        count = ((all_ground_truth == true_cls) & (all_predictions == pred_cls)).sum()
                        row.append(f"{count:4d}")
                    print(f"  {true_cls}    {' '.join(row)}")

            print(f"\nTensorRT model evaluation completed successfully!")
            if total_samples > 1:
                print(f"Model accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"Average inference latency: {avg_latency:.2f} ms")

            # Cleanup TensorRT resources
            try:
                if "engine" in locals():
                    del engine
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                print("TensorRT resources cleaned up successfully.")
            except Exception as e:
                print(f"Warning: Error during cleanup: {e}")

            print("Script completed successfully!")
        else:
            print("No samples were processed successfully.")

    except Exception as e:
        print(f"Error in main execution: {e}")
        # Cleanup on error
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
