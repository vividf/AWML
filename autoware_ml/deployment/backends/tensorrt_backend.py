"""TensorRT inference backend."""

import os
import time
from typing import Tuple

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
import torch

from .base_backend import BaseBackend


class TensorRTBackend(BaseBackend):
    """
    TensorRT inference backend.

    Runs inference using NVIDIA TensorRT engine.
    """

    def __init__(self, model_path: str, device: str = "cuda:0"):
        """
        Initialize TensorRT backend.

        Args:
            model_path: Path to TensorRT engine file
            device: CUDA device to use (ignored, TensorRT uses current CUDA context)
        """
        super().__init__(model_path, device)
        self._engine = None
        self._context = None
        self._logger = trt.Logger(trt.Logger.WARNING)

    def load_model(self) -> None:
        """Load TensorRT engine."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"TensorRT engine not found: {self.model_path}")

        # Initialize TensorRT
        trt.init_libnvinfer_plugins(self._logger, "")
        runtime = trt.Runtime(self._logger)

        # Load engine
        try:
            with open(self.model_path, "rb") as f:
                self._engine = runtime.deserialize_cuda_engine(f.read())

            if self._engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")

            self._context = self._engine.create_execution_context()
            self._model = self._engine  # For is_loaded check
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorRT engine: {e}")

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

        # Convert to numpy and ensure correct format
        input_array = input_tensor.cpu().numpy().astype(np.float32)
        if not input_array.flags["C_CONTIGUOUS"]:
            input_array = np.ascontiguousarray(input_array)

        # Get tensor names
        input_name = None
        output_name = None
        for i in range(self._engine.num_io_tensors):
            tensor_name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                input_name = tensor_name
            elif self._engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                output_name = tensor_name

        if input_name is None or output_name is None:
            raise RuntimeError("Could not find input/output tensor names")

        # Set input shape and get output shape
        self._context.set_input_shape(input_name, input_array.shape)
        output_shape = self._context.get_tensor_shape(output_name)
        output_array = np.empty(output_shape, dtype=np.float32)
        if not output_array.flags["C_CONTIGUOUS"]:
            output_array = np.ascontiguousarray(output_array)

        # Allocate GPU memory
        d_input = cuda.mem_alloc(input_array.nbytes)
        d_output = cuda.mem_alloc(output_array.nbytes)

        # Create CUDA stream and events for timing
        stream = cuda.Stream()
        start = cuda.Event()
        end = cuda.Event()

        try:
            # Set tensor addresses
            self._context.set_tensor_address(input_name, int(d_input))
            self._context.set_tensor_address(output_name, int(d_output))

            # Run inference with timing
            cuda.memcpy_htod_async(d_input, input_array, stream)
            start.record(stream)
            self._context.execute_async_v3(stream_handle=stream.handle)
            end.record(stream)
            cuda.memcpy_dtoh_async(output_array, d_output, stream)
            stream.synchronize()

            latency_ms = end.time_since(start)

            return output_array, latency_ms

        finally:
            # Cleanup GPU memory
            try:
                d_input.free()
                d_output.free()
            except Exception:
                # Silently ignore cleanup errors
                pass

    def cleanup(self) -> None:
        """Clean up TensorRT resources."""
        self._context = None
        self._engine = None
        self._model = None
