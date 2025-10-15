"""ONNX Runtime inference backend."""

import logging
import os
import time
from typing import Tuple

import numpy as np
import onnxruntime as ort
import torch

from .base_backend import BaseBackend


class ONNXBackend(BaseBackend):
    """
    ONNX Runtime inference backend.

    Runs inference using ONNX Runtime on CPU or CUDA.
    """

    def __init__(self, model_path: str, device: str = "cpu", num_classes: int = None):
        """
        Initialize ONNX backend.

        Args:
            model_path: Path to ONNX model file
            device: Device to run inference on ('cpu' or 'cuda')
            num_classes: Number of classes (used to filter multi-output ONNX models)
        """
        super().__init__(model_path, device)
        self._session = None
        self._fallback_attempted = False
        self._logger = logging.getLogger(__name__)
        self.num_classes = num_classes

    def load_model(self) -> None:
        """Load ONNX model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        # Select execution provider based on device
        if self.device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._logger.info("Attempting to use CUDA acceleration (will fallback to CPU if needed)...")
        else:
            providers = ["CPUExecutionProvider"]
            self._logger.info("Using CPU for ONNX inference")

        try:
            self._session = ort.InferenceSession(self.model_path, providers=providers)
            self._model = self._session  # For is_loaded check
            self._logger.info(f"ONNX session using providers: {self._session.get_providers()}")
        except Exception as e:
            # If CUDA provider fails, try CPU fallback
            if self.device.startswith("cuda") and not self._fallback_attempted:
                self._logger.warning(f"CUDA provider failed: {e}")
                self._logger.warning("Attempting CPU fallback...")
                self._fallback_attempted = True
                try:
                    self._session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
                    self._model = self._session
                    self._logger.info(f"ONNX session using providers: {self._session.get_providers()}")
                except Exception as cpu_e:
                    raise RuntimeError(
                        f"Failed to load ONNX model with both CUDA and CPU providers: CUDA={e}, CPU={cpu_e}"
                    )
            else:
                raise RuntimeError(f"Failed to load ONNX model: {e}")

    def infer(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        Run inference on input tensor.

        Args:
            input_tensor: Input tensor for inference

        Returns:
            Tuple of (output_array, latency_ms)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        input_array = input_tensor.cpu().numpy()

        # Prepare input dictionary
        input_name = self._session.get_inputs()[0].name
        onnx_input = {input_name: input_array}

        try:
            # Run inference
            start_time = time.perf_counter()
            outputs = self._session.run(None, onnx_input)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000

            # Handle multi-output models (e.g., YOLOX with multi-scale detection heads)
            if len(outputs) > 1:
                # Debug: log output shapes
                self._logger.info(
                    f"ONNX returned {len(outputs)} outputs with shapes: {[out.shape for out in outputs]}"
                )

                # YOLOX exports separate outputs: cls (num_classes ch), reg (4 ch), obj (1 ch) for each scale
                # PyTorch test mode only returns cls predictions (num_classes channels)
                # Filter for outputs with num_classes channels (class predictions)
                if all(len(out.shape) == 4 for out in outputs):
                    if self.num_classes is not None:
                        # Use num_classes from config to identify class prediction outputs
                        detection_outputs = [out for out in outputs if out.shape[1] == self.num_classes]
                        self._logger.info(
                            f"Filtered to {len(detection_outputs)} outputs with {self.num_classes} channels (class predictions from config)"
                        )
                    else:
                        # Fallback: Find max channels to identify class prediction outputs
                        max_channels = max(out.shape[1] for out in outputs)
                        detection_outputs = [out for out in outputs if out.shape[1] == max_channels]
                        self._logger.info(
                            f"Filtered to {len(detection_outputs)} outputs with {max_channels} channels (class predictions, fallback)"
                        )
                else:
                    # Mixed shapes, use 4D tensors
                    detection_outputs = [out for out in outputs if len(out.shape) == 4]
                    self._logger.info(f"Filtered to {len(detection_outputs)} 4D outputs")

                if detection_outputs:
                    # Use filtered detection outputs
                    flattened = [np.reshape(out, (out.shape[0], -1)) for out in detection_outputs]
                    output = np.concatenate(flattened, axis=1)
                    self._logger.info(f"Final output shape after concat: {output.shape}")
                else:
                    # Fallback: use all outputs
                    self._logger.warning(f"No suitable outputs found, using all {len(outputs)} outputs")
                    flattened = [np.reshape(out, (out.shape[0], -1)) for out in outputs]
                    output = np.concatenate(flattened, axis=1)
            else:
                output = outputs[0]

                # Handle 3D output format (e.g., YOLOX wrapper output [batch, anchors, features])
                # Flatten to match PyTorch backend format for verification
                if len(output.shape) == 3:
                    self._logger.info(f"Flattening 3D output {output.shape} to match PyTorch format")
                    output = output.reshape(output.shape[0], -1)  # Flatten to [batch, anchors*features]
                    self._logger.info(f"Flattened output shape: {output.shape}")

            return output, latency_ms
        except Exception as e:
            # Check if this is a CUDA/PTX error and we haven't tried CPU fallback yet
            if ("PTX" in str(e) or "CUDA" in str(e)) and not self._fallback_attempted:
                self._logger.warning(f"CUDA runtime error detected: {e}")
                self._logger.warning("Recreating session with CPU provider...")
                self._logger.warning("Device will fallback to CPU for consistent verification")
                self._fallback_attempted = True

                # Update device to CPU for consistency
                self.device = "cpu"

                # Recreate session with CPU provider
                self._session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
                self._logger.info(f"Session recreated with providers: {self._session.get_providers()}")

                # Retry inference with CPU
                input_name = self._session.get_inputs()[0].name
                onnx_input = {input_name: input_array}
                start_time = time.perf_counter()
                outputs = self._session.run(None, onnx_input)
                end_time = time.perf_counter()

                latency_ms = (end_time - start_time) * 1000

                # Handle multi-output models
                if len(outputs) > 1:
                    # Filter for class prediction outputs
                    if all(len(out.shape) == 4 for out in outputs):
                        if self.num_classes is not None:
                            # Use num_classes from config
                            detection_outputs = [out for out in outputs if out.shape[1] == self.num_classes]
                        else:
                            # Fallback: use maximum channels
                            max_channels = max(out.shape[1] for out in outputs)
                            detection_outputs = [out for out in outputs if out.shape[1] == max_channels]
                    else:
                        detection_outputs = [out for out in outputs if len(out.shape) == 4]

                    if detection_outputs:
                        flattened = [np.reshape(out, (out.shape[0], -1)) for out in detection_outputs]
                        output = np.concatenate(flattened, axis=1)
                    else:
                        # Fallback: use all outputs
                        flattened = [np.reshape(out, (out.shape[0], -1)) for out in outputs]
                        output = np.concatenate(flattened, axis=1)
                else:
                    output = outputs[0]

                    # Handle 3D output format (e.g., YOLOX wrapper output [batch, anchors, features])
                    # Keep 3D format to match PyTorch backend format for verification
                    if len(output.shape) == 3:
                        self._logger.info(f"Keeping 3D output format {output.shape} to match PyTorch format")
                        # Keep the 3D format (1, 18900, 13) - no flattening needed
                        self._logger.info(f"Output shape maintained: {output.shape}")

                return output, latency_ms
            else:
                raise

    def cleanup(self) -> None:
        """Clean up ONNX Runtime resources."""
        self._session = None
        self._model = None
        self._fallback_attempted = False
