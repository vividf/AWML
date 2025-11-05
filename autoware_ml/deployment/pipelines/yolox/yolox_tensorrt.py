"""
YOLOX TensorRT Pipeline Implementation.

This module provides the TensorRT backend implementation for YOLOX deployment.
"""

from typing import Dict, List, Tuple, Any
import logging

import torch
import numpy as np

from .yolox_pipeline import YOLOXDeploymentPipeline


logger = logging.getLogger(__name__)


class YOLOXTensorRTPipeline(YOLOXDeploymentPipeline):
    """
    YOLOX TensorRT backend implementation.
    
    This pipeline uses TensorRT for maximum inference performance on NVIDIA GPUs.
    Provides the fastest inference speed for production deployment.
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
        max_detections: int = 300
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
        
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Discover I/O using either classic bindings API or newer I/O tensors API
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        self.use_io_tensors = False
        

        # TODO(vividf): check this, old implementation don't have this
        if hasattr(self.engine, "num_bindings"):
            # Classic bindings API
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                shape = self.engine.get_binding_shape(i)
                if self.engine.binding_is_input(i):
                    self.input_name = name
                    self.input_shape = tuple(shape)
                else:
                    self.output_name = name
                    self.output_shape = tuple(shape)
        else:
            # New I/O tensors API (TensorRT >= 8.5)
            self.use_io_tensors = True
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
        
        # Allocate device memory (for classic bindings). For I/O tensors we will allocate lazily per shape.
        if not self.use_io_tensors:
            self._allocate_buffers()
        else:
            # Create CUDA stream for enqueue_v3
            self.stream = self.cuda.Stream()
            self.d_input = None
            self.d_output = None
            self.h_output = None
        
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
            backend_type="tensorrt"
        )
    
    def _allocate_buffers(self):
        """Allocate device memory for input and output."""
        import numpy as np
        
        # Calculate buffer sizes
        input_nbytes = np.prod(self.input_shape) * np.dtype(np.float32).itemsize
        output_nbytes = np.prod(self.output_shape) * np.dtype(np.float32).itemsize
        
        # Allocate device memory
        self.d_input = self.cuda.mem_alloc(input_nbytes)
        self.d_output = self.cuda.mem_alloc(output_nbytes)
        
        # Create host output buffer
        self.h_output = np.empty(self.output_shape, dtype=np.float32)
        
        logger.debug(f"Allocated TensorRT buffers: input={input_nbytes} bytes, output={output_nbytes} bytes")
    
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
        
        # Ensure contiguous memory layout
        input_np = np.ascontiguousarray(input_np)
        
        if not self.use_io_tensors:
            # Classic bindings path
            if self.input_shape[0] == -1:
                # Dynamic batch size - set binding shape
                self.context.set_binding_shape(0, tuple(input_np.shape))
                # Optionally reallocate host/output buffers based on new shape
                if self.h_output is None or self.h_output.shape[0] != input_np.shape[0]:
                    self._allocate_buffers_dynamic(tuple(input_np.shape))
            
            # Copy input to device and run
            self.cuda.memcpy_htod(self.d_input, input_np)
            self.context.execute_v2([int(self.d_input), int(self.d_output)])
            # Copy output back
            self.cuda.memcpy_dtoh(self.h_output, self.d_output)
            logger.debug(f"TensorRT inference output shape: {self.h_output.shape}")
            return self.h_output
        else:
            # I/O tensors path
            # Determine current output shape; if dynamic, set input shape first
            if -1 in self.engine.get_tensor_shape(self.input_name):
                # Set the input shape on the context (explicit batch)
                if hasattr(self.context, "set_input_shape"):
                    self.context.set_input_shape(self.input_name, tuple(input_np.shape))
                else:
                    # Fallback for some versions
                    self.context.set_binding_shape(0, tuple(input_np.shape))
            
            # Allocate device buffers lazily based on actual shapes
            in_nbytes = input_np.nbytes
            if self.d_input is None or self.h_output is None:
                self.d_input = self.cuda.mem_alloc(in_nbytes)
            # Query output shape from context if supported
            try:
                out_shape = tuple(self.context.get_tensor_shape(self.output_name))
            except Exception:
                # Fallback to engine declared shape (may include -1); assume batch from input
                engine_out = self.engine.get_tensor_shape(self.output_name)
                out_shape = (input_np.shape[0],) + tuple(engine_out[1:])
            out_nbytes = int(np.prod(out_shape)) * np.dtype(np.float32).itemsize
            if (self.d_output is None) or (self.h_output is None) or (self.h_output.nbytes != out_nbytes):
                self.d_output = self.cuda.mem_alloc(out_nbytes)
                self.h_output = np.empty(out_shape, dtype=np.float32)
            
            # Copy input, set tensor addresses, and enqueue
            if hasattr(self.context, "set_tensor_address"):
                self.cuda.memcpy_htod_async(self.d_input, input_np, self.stream)
                # Set tensor addresses
                self.context.set_tensor_address(self.input_name, int(self.d_input))
                self.context.set_tensor_address(self.output_name, int(self.d_output))
                
                # Check if enqueue_v3 is available (TensorRT 8.5+)
                if hasattr(self.context, "enqueue_v3"):
                    self.context.enqueue_v3(self.stream.handle)
                    # Copy output back
                    self.cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
                    self.stream.synchronize()
                else:
                    # Fallback for older TensorRT versions - use synchronous execution
                    logger.debug("Using synchronous execution (TRT < 8.5)")
                    self.cuda.memcpy_htod(self.d_input, input_np)
                    self.context.execute_v2([int(self.d_input), int(self.d_output)])
                    self.cuda.memcpy_dtoh(self.h_output, self.d_output)
            else:
                # Fallback to execute_v2 with bindings list
                self.cuda.memcpy_htod(self.d_input, input_np)
                self.context.execute_v2([int(self.d_input), int(self.d_output)])
                self.cuda.memcpy_dtoh(self.h_output, self.d_output)
            
            logger.debug(f"TensorRT (I/O tensors) output shape: {self.h_output.shape}")
            return self.h_output
    
    def _allocate_buffers_dynamic(self, input_shape):
        """Reallocate buffers for dynamic batch size."""
        import numpy as np
        
        # Calculate new buffer sizes
        input_nbytes = np.prod(input_shape) * np.dtype(np.float32).itemsize
        
        # Update output shape
        batch_size = input_shape[0]
        output_shape = (batch_size,) + self.output_shape[1:]
        output_nbytes = np.prod(output_shape) * np.dtype(np.float32).itemsize
        
        # Reallocate device memory
        self.d_input = self.cuda.mem_alloc(input_nbytes)
        self.d_output = self.cuda.mem_alloc(output_nbytes)
        
        # Create new host output buffer
        self.h_output = np.empty(output_shape, dtype=np.float32)
        
        logger.debug(f"Reallocated TensorRT buffers for batch_size={batch_size}")

