"""
YOLOX TensorRT Pipeline Implementation.

This module provides the TensorRT backend implementation for YOLOX deployment.
Requires TensorRT >= 8.5 (uses I/O tensors API).
"""

import logging
from typing import List, Tuple

import numpy as np
import torch

from deployment.pipelines.common.gpu_resource_mixin import (
    GPUResourceMixin,
    release_tensorrt_resources,
)
from deployment.pipelines.yolox.yolox_pipeline import YOLOXDeploymentPipeline

logger = logging.getLogger(__name__)


class YOLOXTensorRTPipeline(GPUResourceMixin, YOLOXDeploymentPipeline):
    """
    YOLOX TensorRT backend implementation.

    This pipeline uses TensorRT for maximum inference performance on NVIDIA GPUs.
    Provides the fastest inference speed for production deployment.

    Resource Management:
        This pipeline implements GPUResourceMixin for proper resource cleanup.
        Use as a context manager for automatic cleanup:

            with YOLOXTensorRTPipeline(...) as pipeline:
                results = pipeline.infer(data)
            # Resources automatically released
    """

    def __init__(
        self,
        engine_path: str,
        device: str = "cuda",
        num_classes: int = 8,
        class_names: List[str] = None,
        input_size: Tuple[int, int] = (960, 960),
        score_threshold: float = 0.01,
        nms_threshold: float = 0.65,
        max_detections: int = 300,
    ):
        """
        Initialize YOLOX TensorRT pipeline.

        Args:
            engine_path: Path to TensorRT engine file
            device: Device for inference (must be 'cuda' or 'cuda:X')
            num_classes: Number of object classes
            class_names: List of class names
            input_size: Model input size (height, width)
            score_threshold: Confidence threshold for filtering
            nms_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections per image
        """
        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
            import tensorrt as trt
        except ImportError:
            raise ImportError(
                "TensorRT and pycuda are required for TensorRT pipeline. " "Please install TensorRT and pycuda."
            )

        if not device.startswith("cuda"):
            raise ValueError(f"TensorRT requires CUDA device, got: {device}")

        self.trt = trt
        self.cuda = cuda

        # Load TensorRT engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            engine_data = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # Discover I/O using I/O tensors API (TensorRT >= 8.5)
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None

        num_io = self.engine.num_io_tensors
        for i in range(num_io):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            if mode == trt.TensorIOMode.INPUT and self.input_name is None:
                self.input_name = name
                self.input_shape = tuple(shape)
            elif mode == trt.TensorIOMode.OUTPUT and self.output_name is None:
                self.output_name = name
                self.output_shape = tuple(shape)

        logger.info(f"Loaded TensorRT engine from: {engine_path}")
        logger.info(f"Input: {self.input_name}, shape: {self.input_shape}")
        logger.info(f"Output: {self.output_name}, shape: {self.output_shape}")

        # I/O tensors API - allocate lazily
        self.stream = self.cuda.Stream()
        self.d_input = None
        self.d_output = None
        self.h_output = None
        self._cleanup_called = False  # For GPUResourceMixin

        # Initialize parent class (pass engine as model)
        super().__init__(
            model=self.engine,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            input_size=input_size,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            max_detections=max_detections,
            backend_type="tensorrt",
        )

    def run_model(self, preprocessed_input: torch.Tensor) -> np.ndarray:
        """
        Run TensorRT model inference.

        Args:
            preprocessed_input: Preprocessed image tensor [1, C, H, W]

        Returns:
            Model output [1, num_predictions, 4+1+num_classes]
            Format: [bbox(4), objectness(1), class_scores(num_classes)]
        """
        # Convert torch tensor to numpy
        input_np = preprocessed_input.cpu().numpy()
        input_np = np.ascontiguousarray(input_np, dtype=np.float32)
        input_shape = tuple(input_np.shape)

        # Handle dynamic shapes using I/O tensors API (TensorRT >= 8.5)
        if -1 in self.engine.get_tensor_shape(self.input_name):
            self.context.set_input_shape(self.input_name, input_shape)

        # Allocate device buffers lazily based on actual shapes
        in_nbytes = input_np.nbytes
        if self.d_input is None:
            self.d_input = self.cuda.mem_alloc(in_nbytes)

        # Query output shape from context
        try:
            out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        except Exception:
            # Fallback to engine declared shape
            engine_out = self.engine.get_tensor_shape(self.output_name)
            out_shape = (input_np.shape[0],) + tuple(engine_out[1:])

        out_nbytes = int(np.prod(out_shape)) * np.dtype(np.float32).itemsize
        if self.d_output is None or self.h_output is None or self.h_output.nbytes != out_nbytes:
            if self.d_output is not None:
                self.d_output.free()
            self.d_output = self.cuda.mem_alloc(out_nbytes)
            self.h_output = np.empty(out_shape, dtype=np.float32)

        # Copy input, set tensor addresses, and execute
        self.cuda.memcpy_htod_async(self.d_input, input_np, self.stream)
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy output back
        self.cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        return self.h_output

    def _release_gpu_resources(self) -> None:
        """Release TensorRT resources (GPUResourceMixin implementation)."""
        # Free CUDA device buffers
        cuda_buffers = []
        for attr in ("d_input", "d_output"):
            buf = getattr(self, attr, None)
            if buf is not None:
                cuda_buffers.append(buf)
            setattr(self, attr, None)

        # Release engine and context
        engines = {"engine": getattr(self, "engine", None)} if hasattr(self, "engine") else None
        contexts = {"context": getattr(self, "context", None)} if hasattr(self, "context") else None

        release_tensorrt_resources(engines=engines, contexts=contexts, cuda_buffers=cuda_buffers)

        # Clear host buffer
        self.h_output = None
        self.engine = None
        self.context = None
