"""
CenterPoint TensorRT Pipeline Implementation.

This module implements the CenterPoint pipeline using TensorRT,
providing maximum inference speed on NVIDIA GPUs.
"""

import logging
import os
from typing import List

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
import torch

from .centerpoint_pipeline import CenterPointDeploymentPipeline


logger = logging.getLogger(__name__)


class CenterPointTensorRTPipeline(CenterPointDeploymentPipeline):
    """
    TensorRT implementation of CenterPoint pipeline.
    
    Uses TensorRT for voxel encoder and backbone/head,
    while keeping middle encoder in PyTorch (sparse convolution cannot be converted).
    
    Provides maximum inference speed on NVIDIA GPUs with INT8/FP16 optimization.
    """
    
    def __init__(self, pytorch_model, tensorrt_dir: str, device: str = "cuda"):
        """
        Initialize TensorRT pipeline.
        
        Args:
            pytorch_model: PyTorch model (for preprocessing, middle encoder, postprocessing)
            tensorrt_dir: Directory containing TensorRT engine files
            device: Device for inference (must be 'cuda')
        """
        if not device.startswith("cuda"):
            raise ValueError("TensorRT requires CUDA device")
        
        super().__init__(pytorch_model, device, backend_type="tensorrt")
        
        self.tensorrt_dir = tensorrt_dir
        self._engines = {}
        self._contexts = {}
        self._logger = trt.Logger(trt.Logger.WARNING)
        
        self._load_tensorrt_engines()
        
        logger.info(f"TensorRT pipeline initialized with engines from: {tensorrt_dir}")
    
    def _load_tensorrt_engines(self):
        """Load TensorRT engines for voxel encoder and backbone/head."""
        # Initialize TensorRT
        trt.init_libnvinfer_plugins(self._logger, "")
        runtime = trt.Runtime(self._logger)
        
        # Define engine files
        engine_files = {
            "voxel_encoder": "pts_voxel_encoder.engine",
            "backbone_neck_head": "pts_backbone_neck_head.engine"
        }
        
        for component, engine_file in engine_files.items():
            engine_path = os.path.join(self.tensorrt_dir, engine_file)
            
            if not os.path.exists(engine_path):
                raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
            
            try:
                # Load engine
                with open(engine_path, "rb") as f:
                    engine = runtime.deserialize_cuda_engine(f.read())
                
                if engine is None:
                    raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
                
                # Create execution context
                context = engine.create_execution_context()
                
                self._engines[component] = engine
                self._contexts[component] = context
                
                logger.info(f"Loaded TensorRT engine: {component}")
                
            except Exception as e:
                raise RuntimeError(f"Failed to load TensorRT engine {component}: {e}")
    
    def run_voxel_encoder(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Run voxel encoder using TensorRT.
        
        Args:
            input_features: Input features [N_voxels, max_points, feature_dim]
            
        Returns:
            voxel_features: Voxel features [N_voxels, feature_dim]
        """
        engine = self._engines["voxel_encoder"]
        context = self._contexts["voxel_encoder"]
        
        # Convert to numpy
        input_array = input_features.cpu().numpy().astype(np.float32)
        
        # Ensure contiguous
        if not input_array.flags["C_CONTIGUOUS"]:
            input_array = np.ascontiguousarray(input_array)
        
        # Get tensor names
        input_name = None
        output_name = None
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                input_name = tensor_name
            elif engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                output_name = tensor_name
        
        if input_name is None or output_name is None:
            raise RuntimeError("Could not find input/output tensor names for voxel encoder")
        
        # Set input shape and get output shape
        context.set_input_shape(input_name, input_array.shape)
        output_shape = context.get_tensor_shape(output_name)
        output_array = np.empty(output_shape, dtype=np.float32)
        
        if not output_array.flags["C_CONTIGUOUS"]:
            output_array = np.ascontiguousarray(output_array)
        
        # Allocate GPU memory
        d_input = cuda.mem_alloc(input_array.nbytes)
        d_output = cuda.mem_alloc(output_array.nbytes)
        
        # Create CUDA stream
        stream = cuda.Stream()
        
        try:
            # Set tensor addresses
            context.set_tensor_address(input_name, int(d_input))
            context.set_tensor_address(output_name, int(d_output))
            
            # Run inference
            cuda.memcpy_htod_async(d_input, input_array, stream)
            context.execute_async_v3(stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(output_array, d_output, stream)
            stream.synchronize()
            
            # Convert to torch tensor
            voxel_features = torch.from_numpy(output_array).to(self.device)
            
            # Squeeze middle dimension if present
            if voxel_features.ndim == 3 and voxel_features.shape[1] == 1:
                voxel_features = voxel_features.squeeze(1)
            
            logger.debug(f"TensorRT voxel encoder output shape: {voxel_features.shape}")
            
            return voxel_features
            
        finally:
            # Cleanup GPU memory
            try:
                d_input.free()
                d_output.free()
            except Exception:
                pass
    
    def run_backbone_head(self, spatial_features: torch.Tensor) -> List[torch.Tensor]:
        """
        Run backbone + neck + head using TensorRT.
        
        Args:
            spatial_features: Spatial features [B, C, H, W]
            
        Returns:
            List of head outputs: [heatmap, reg, height, dim, rot, vel]
        """
        engine = self._engines["backbone_neck_head"]
        context = self._contexts["backbone_neck_head"]
        
        # Convert to numpy
        input_array = spatial_features.cpu().numpy().astype(np.float32)
        
        # Ensure contiguous
        if not input_array.flags["C_CONTIGUOUS"]:
            input_array = np.ascontiguousarray(input_array)
        
        # Get tensor names
        input_name = None
        output_names = []
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                input_name = tensor_name
            elif engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                output_names.append(tensor_name)
        
        if input_name is None or not output_names:
            raise RuntimeError("Could not find input/output tensor names for backbone_neck_head")
        
        # Set input shape
        context.set_input_shape(input_name, input_array.shape)
        
        # Get output shapes and allocate memory
        output_arrays = {}
        d_outputs = {}
        
        for output_name in output_names:
            output_shape = context.get_tensor_shape(output_name)
            output_array = np.empty(output_shape, dtype=np.float32)
            if not output_array.flags["C_CONTIGUOUS"]:
                output_array = np.ascontiguousarray(output_array)
            output_arrays[output_name] = output_array
            d_outputs[output_name] = cuda.mem_alloc(output_array.nbytes)
        
        # Allocate input memory
        d_input = cuda.mem_alloc(input_array.nbytes)
        
        # Create CUDA stream
        stream = cuda.Stream()
        
        try:
            # Set tensor addresses
            context.set_tensor_address(input_name, int(d_input))
            for output_name in output_names:
                context.set_tensor_address(output_name, int(d_outputs[output_name]))
            
            # Run inference
            cuda.memcpy_htod_async(d_input, input_array, stream)
            context.execute_async_v3(stream_handle=stream.handle)
            
            # Copy outputs
            for output_name in output_names:
                cuda.memcpy_dtoh_async(output_arrays[output_name], d_outputs[output_name], stream)
            
            stream.synchronize()
            
            # Convert outputs to torch tensors
            # outputs should be: [heatmap, reg, height, dim, rot, vel]
            head_outputs = [torch.from_numpy(output_arrays[name]).to(self.device) 
                          for name in output_names]
            
            if len(head_outputs) != 6:
                raise ValueError(f"Expected 6 head outputs, got {len(head_outputs)}")
            
            logger.debug(f"TensorRT backbone+head output: {[out.shape for out in head_outputs]}")
            
            return head_outputs
            
        finally:
            # Cleanup GPU memory
            try:
                d_input.free()
                for d_output in d_outputs.values():
                    d_output.free()
            except Exception:
                pass
    
    def __del__(self):
        """Cleanup TensorRT resources."""
        self._contexts = {}
        self._engines = {}

