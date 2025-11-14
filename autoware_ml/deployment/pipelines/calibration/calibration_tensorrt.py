"""
CalibrationStatusClassification TensorRT Pipeline Implementation.

This module provides the TensorRT backend implementation for CalibrationStatusClassification deployment.
Requires TensorRT >= 8.5 (uses I/O tensors API).
"""

from typing import Dict, List, Tuple, Any
import logging

import torch
import numpy as np

from autoware_ml.deployment.pipelines.calibration.calibration_pipeline import CalibrationDeploymentPipeline


logger = logging.getLogger(__name__)


class CalibrationTensorRTPipeline(CalibrationDeploymentPipeline):
    """
    CalibrationStatusClassification TensorRT backend implementation.
    
    This pipeline uses TensorRT for maximum inference performance on NVIDIA GPUs.
    Provides the fastest inference speed for production deployment.
    """
    
    def __init__(
        self, 
        engine_path: str,
        device: str = "cuda",
        num_classes: int = 2,
        class_names: List[str] = None,
    ):
        """
        Initialize CalibrationStatusClassification TensorRT pipeline.
        
        Args:
            engine_path: Path to TensorRT engine file
            device: Device for inference (must be 'cuda' or 'cuda:X')
            num_classes: Number of classes (2 for binary classification)
            class_names: List of class names (default: ["miscalibrated", "calibrated"])
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError:
            raise ImportError(
                "TensorRT and pycuda are required for TensorRT pipeline. "
                "Please install TensorRT and pycuda."
            )
        
        if not device.startswith("cuda"):
            raise ValueError(f"TensorRT requires CUDA device, got: {device}")
        
        self.trt = trt
        self.cuda = cuda
        
        # Load TensorRT engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Discover I/O using I/O tensors API (TensorRT >= 8.5)
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        self.input_dtype = None
        self.output_dtype = None
        
        num_io = self.engine.num_io_tensors
        for i in range(num_io):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            if mode == trt.TensorIOMode.INPUT and self.input_name is None:
                self.input_name = name
                self.input_shape = tuple(shape)
                self.input_dtype = dtype
                logger.info(f"  Input: {name}, shape={shape}, dtype={dtype}")
            elif mode == trt.TensorIOMode.OUTPUT and self.output_name is None:
                self.output_name = name
                self.output_shape = tuple(shape)
                self.output_dtype = dtype
                logger.info(f"  Output: {name}, shape={shape}, dtype={dtype}")
        
        logger.info(f"Loaded TensorRT engine from: {engine_path}")
        
        # I/O tensors API - allocate lazily
        self.d_input = None
        self.d_output = None
        self.h_output = None
        self.stream = cuda.Stream()
        
        # Initialize parent with dummy model (we use engine instead)
        super().__init__(
            model=self.engine,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            backend_type="tensorrt"
        )
        
        logger.info(f"Initialized CalibrationTensorRTPipeline on {device}")
        logger.info(f"  Engine: {engine_path}")
    
    def run_model(self, preprocessed_input: torch.Tensor) -> torch.Tensor:
        """
        Run TensorRT model inference using I/O tensors API (TensorRT >= 8.5).
        
        Args:
            preprocessed_input: Preprocessed tensor [1, C, H, W]
            
        Returns:
            Model output (logits) [1, num_classes]
        """
        # Convert to numpy
        input_np = preprocessed_input.cpu().numpy()
        input_np = np.ascontiguousarray(input_np, dtype=self.input_dtype)
        input_shape = tuple(input_np.shape)
        
        # Handle dynamic shapes
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
        
        out_nbytes = int(np.prod(out_shape)) * np.dtype(self.output_dtype).itemsize
        if self.d_output is None or self.h_output is None or self.h_output.nbytes != out_nbytes:
            if self.d_output is not None:
                self.d_output.free()
            self.d_output = self.cuda.mem_alloc(out_nbytes)
            self.h_output = np.empty(out_shape, dtype=self.output_dtype)
        
        # Copy input, set tensor addresses, and execute
        self.cuda.memcpy_htod_async(self.d_input, input_np, self.stream)
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Copy output back
        self.cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        
        return torch.from_numpy(self.h_output).to(self.device)

