"""CenterPoint TensorRT inference backend for multi-engine setup."""

import os
import time
from typing import Dict, List, Tuple, Any

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
import torch

from autoware_ml.deployment.backends.base_backend import BaseBackend


class CenterPointTensorRTBackend(BaseBackend):
    """
    CenterPoint TensorRT inference backend.

    Handles the multi-engine setup for CenterPoint:
    1. pts_voxel_encoder.trt - voxel feature extraction
    2. pts_backbone_neck_head.trt - backbone, neck, and head processing
    
    Note: Middle encoder is not converted to TensorRT and runs in PyTorch.
    """

    def __init__(self, model_path: str, device: str = "cuda:0", pytorch_model=None):
        """
        Initialize CenterPoint TensorRT backend.

        Args:
            model_path: Path to directory containing TensorRT engines
            device: CUDA device to use (ignored, TensorRT uses current CUDA context)
            pytorch_model: PyTorch model for middle encoder (required)
        """
        super().__init__(model_path, device)
        self._engines = {}
        self._contexts = {}
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._is_loaded = False
        self.pytorch_model = pytorch_model

    def load_model(self) -> None:
        """Load TensorRT engines for both components."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"TensorRT engine directory not found: {self.model_path}")

        # Initialize TensorRT
        trt.init_libnvinfer_plugins(self._logger, "")
        runtime = trt.Runtime(self._logger)

        # Define engine files
        engine_files = {
            "voxel_encoder": "pts_voxel_encoder.engine",
            "backbone_neck_head": "pts_backbone_neck_head.engine"
        }

        for component, engine_file in engine_files.items():
            engine_path = os.path.join(self.model_path, engine_file)
            
            if not os.path.exists(engine_path):
                raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

            try:
                # Load engine
                with open(engine_path, "rb") as f:
                    engine = runtime.deserialize_cuda_engine(f.read())

                if engine is None:
                    raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

                context = engine.create_execution_context()
                
                self._engines[component] = engine
                self._contexts[component] = context
                
                print(f"✅ Loaded TensorRT engine: {component}")

            except Exception as e:
                raise RuntimeError(f"Failed to load TensorRT engine {component}: {e}")

        self._is_loaded = True
        self._model = self._engines  # For is_loaded check

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def infer(self, input_data) -> Tuple[List[Dict[str, torch.Tensor]], float]:
        """
        Run inference on CenterPoint input data.

        Args:
            input_data: Either:
                - Dictionary containing 'voxels', 'num_points', 'coors' (for full pipeline)
                - Single tensor (for verification compatibility)

        Returns:
            Tuple of (detection_results, total_latency_ms)
        """
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        total_start_time = time.time()

        # Handle different input formats
        if isinstance(input_data, dict):
            # Debug: Check input_data keys
            
            # Full pipeline input
            # Check if input_data contains 'voxels' or 'points'
            if 'voxels' in input_data:
                # Full pipeline input with voxels
                # Debug: Check if coors is present and valid
                if 'coors' in input_data:
                    coors = input_data['coors']
                else:
                    logger.warning(f"Input contains voxels but no coors!")
                
                voxel_features = self._run_voxel_encoder(input_data)
                voxel_features_processed = self._process_middle_encoder(voxel_features, input_data)
                detection_results = self._run_backbone_neck_head(voxel_features_processed)
            elif 'points' in input_data:
                # Raw points input - need to voxelize first
                points = input_data['points']
                # Create dummy voxel data for verification
                # Assume points shape is (N, 5) where 5 = (x, y, z, intensity, timestamp)
                num_points = points.shape[0]
                # Create dummy voxels with shape (num_voxels, max_points_per_voxel, point_dim)
                # For verification, use a reasonable number of voxels
                num_voxels = min(10000, num_points // 10)  # Assume ~10 points per voxel
                max_points_per_voxel = 32
                point_dim = 11  # TensorRT engine expects 11 dimensions
                
                # Create dummy voxels
                dummy_voxels = torch.zeros(num_voxels, max_points_per_voxel, point_dim, 
                                         dtype=points.dtype, device=points.device)
                
                # Fill with some real point data (pad to 11 dimensions)
                for i in range(num_voxels):
                    start_idx = i * 10
                    end_idx = min(start_idx + max_points_per_voxel, num_points)
                    if start_idx < num_points:
                        voxel_points = points[start_idx:end_idx, :4]  # Take first 4 dimensions
                        # Pad to 11 dimensions
                        padded_points = torch.zeros(end_idx-start_idx, point_dim, 
                                                  dtype=points.dtype, device=points.device)
                        padded_points[:, :4] = voxel_points
                        dummy_voxels[i, :end_idx-start_idx, :] = padded_points
                
                # Create dummy input data
                dummy_input = {
                    'voxels': dummy_voxels,
                    'num_points': torch.ones(num_voxels, dtype=torch.int32, device=points.device) * max_points_per_voxel,
                    'coors': torch.zeros(num_voxels, 4, dtype=torch.int32, device=points.device)
                }
                
                voxel_features = self._run_voxel_encoder(dummy_input)
                voxel_features_processed = self._process_middle_encoder(voxel_features, dummy_input)
                detection_results = self._run_backbone_neck_head(voxel_features_processed)
            else:
                raise ValueError(f"Unsupported input data keys: {list(input_data.keys())}")
        else:
            # Single tensor input (for verification)
            # Create dummy input data for verification
            dummy_input = {
                'voxels': input_data,  # Assume input_data is voxels
                'num_points': torch.ones(input_data.shape[0], dtype=torch.int32, device=input_data.device),
                'coors': torch.zeros(input_data.shape[0], 4, dtype=torch.int32, device=input_data.device)
            }
            voxel_features = self._run_voxel_encoder(dummy_input)
            voxel_features_processed = self._process_middle_encoder(voxel_features, dummy_input)
            detection_results = self._run_backbone_neck_head(voxel_features_processed)

        total_latency_ms = (time.time() - total_start_time) * 1000

        return detection_results, total_latency_ms

    def _run_voxel_encoder(self, input_data) -> torch.Tensor:
        """Run voxel encoder inference."""
        engine = self._engines["voxel_encoder"]
        context = self._contexts["voxel_encoder"]

        # Handle both dict and tensor inputs
        if isinstance(input_data, dict):
            voxels = input_data["voxels"].cpu().numpy().astype(np.float32)
        else:
            # Single tensor input
            voxels = input_data.cpu().numpy().astype(np.float32)
            
        if not voxels.flags["C_CONTIGUOUS"]:
            voxels = np.ascontiguousarray(voxels)

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
        context.set_input_shape(input_name, voxels.shape)
        output_shape = context.get_tensor_shape(output_name)
        output_array = np.empty(output_shape, dtype=np.float32)
        if not output_array.flags["C_CONTIGUOUS"]:
            output_array = np.ascontiguousarray(output_array)

        # Allocate GPU memory
        d_input = cuda.mem_alloc(voxels.nbytes)
        d_output = cuda.mem_alloc(output_array.nbytes)

        # Create CUDA stream
        stream = cuda.Stream()

        try:
            # Set tensor addresses
            context.set_tensor_address(input_name, int(d_input))
            context.set_tensor_address(output_name, int(d_output))

            # Run inference
            cuda.memcpy_htod_async(d_input, voxels, stream)
            context.execute_async_v3(stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(output_array, d_output, stream)
            stream.synchronize()

            voxel_features = torch.from_numpy(output_array)
            
            # Debug: Check voxel encoder output
            
            return voxel_features

        finally:
            # Cleanup GPU memory
            try:
                d_input.free()
                d_output.free()
            except Exception:
                pass

    def _process_middle_encoder(self, voxel_features: torch.Tensor, input_data) -> torch.Tensor:
        """
        Process voxel features through middle encoder using PyTorch model.
        
        Note: Middle encoder is not converted to TensorRT and runs in PyTorch.
        This is necessary because it uses sparse convolutions which are not well-supported in TensorRT.
        """
        if self.pytorch_model is None:
            raise ValueError("PyTorch model is required for middle encoder processing")
        
        # Ensure voxel_features is on the correct device
        device = next(self.pytorch_model.parameters()).device
        voxel_features = voxel_features.to(device)
        
        # Handle both dict and tensor inputs
        if isinstance(input_data, dict):
            coors = input_data["coors"]
            batch_size = int(coors[-1, 0].item()) + 1 if len(coors) > 0 else 1
        else:
            # Single tensor input - use batch size 1 for verification
            batch_size = 1
            # Create dummy coors for middle encoder
            coors = torch.zeros(voxel_features.shape[0], 4, dtype=torch.int32, device=device)
        
        # Ensure coors is on the correct device
        coors = coors.to(device)
        
        # Debug: Check coors
        
        # Process voxel features (shape: [num_voxels, 1, feature_dim] or [num_voxels, feature_dim])
        if voxel_features.dim() == 3:
            voxel_features = voxel_features.squeeze(1)  # Remove middle dimension if present
        
        # Convert to torch tensor if numpy
        if isinstance(voxel_features, np.ndarray):
            voxel_features = torch.from_numpy(voxel_features).to(device)
        
        # Run PyTorch middle encoder
        with torch.no_grad():
            spatial_features = self.pytorch_model.pts_middle_encoder(
                voxel_features, coors, batch_size
            )
        
        # Debug: Check spatial features
        
        return spatial_features

    def _run_backbone_neck_head(self, features: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """Run backbone, neck, and head inference."""
        engine = self._engines["backbone_neck_head"]
        context = self._contexts["backbone_neck_head"]

        # Convert to numpy
        features_np = features.cpu().numpy().astype(np.float32)
        if not features_np.flags["C_CONTIGUOUS"]:
            features_np = np.ascontiguousarray(features_np)

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
        context.set_input_shape(input_name, features_np.shape)

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
        d_input = cuda.mem_alloc(features_np.nbytes)

        # Create CUDA stream
        stream = cuda.Stream()

        try:
            # Set tensor addresses
            context.set_tensor_address(input_name, int(d_input))
            for output_name in output_names:
                context.set_tensor_address(output_name, int(d_outputs[output_name]))

            # Run inference
            cuda.memcpy_htod_async(d_input, features_np, stream)
            context.execute_async_v3(stream_handle=stream.handle)
            
            # Copy outputs
            for output_name in output_names:
                cuda.memcpy_dtoh_async(output_arrays[output_name], d_outputs[output_name], stream)
            
            stream.synchronize()

            # Convert outputs to detection results format
            detection_results = []
            for output_name, output_array in output_arrays.items():
                detection_results.append({
                    output_name: torch.from_numpy(output_array)
                })

            return detection_results

        finally:
            # Cleanup GPU memory
            try:
                d_input.free()
                for d_output in d_outputs.values():
                    d_output.free()
            except Exception:
                pass

    def cleanup(self) -> None:
        """Clean up TensorRT resources."""
        self._contexts = {}
        self._engines = {}
        self._model = None
        self._is_loaded = False
