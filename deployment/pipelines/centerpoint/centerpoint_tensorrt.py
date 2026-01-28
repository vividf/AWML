"""
CenterPoint TensorRT Pipeline Implementation.

This module implements the CenterPoint pipeline using TensorRT,
providing maximum inference speed on NVIDIA GPUs.
"""

import logging
import os.path as osp
import time
from typing import Dict, List, Tuple

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
import torch

from deployment.pipelines.centerpoint.centerpoint_pipeline import CenterPointDeploymentPipeline
from deployment.pipelines.common.gpu_resource_mixin import (
    GPUResourceMixin,
    TensorRTResourceManager,
    release_tensorrt_resources,
)
from projects.CenterPoint.deploy.configs.deploy_config import onnx_config

logger = logging.getLogger(__name__)


class CenterPointTensorRTPipeline(GPUResourceMixin, CenterPointDeploymentPipeline):
    """
    TensorRT implementation of CenterPoint pipeline.

    Uses TensorRT for voxel encoder and backbone/head,
    while keeping middle encoder in PyTorch (sparse convolution cannot be converted).

    Provides maximum inference speed on NVIDIA GPUs with INT8/FP16 optimization.

    Resource Management:
        This pipeline implements GPUResourceMixin for proper resource cleanup.
        Use as a context manager for automatic cleanup:

            with CenterPointTensorRTPipeline(...) as pipeline:
                results = pipeline.infer(data)
            # Resources automatically released
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
        self._cleanup_called = False  # For GPUResourceMixin

        # Create CUDA events for GPU timing measurements
        self._backbone_start_event = cuda.Event()
        self._backbone_end_event = cuda.Event()
        self._voxel_encoder_start_event = cuda.Event()
        self._voxel_encoder_end_event = cuda.Event()

        self._load_tensorrt_engines()

        logger.info(f"TensorRT pipeline initialized with engines from: {tensorrt_dir}")

    def _load_tensorrt_engines(self):
        """Load TensorRT engines for voxel encoder and backbone/head."""
        # Initialize TensorRT
        trt.init_libnvinfer_plugins(self._logger, "")
        runtime = trt.Runtime(self._logger)

        # Resolve engine filenames from deploy config with sane fallbacks
        component_cfg = onnx_config.get("components", {})
        voxel_cfg = component_cfg.get("voxel_encoder", {})
        backbone_cfg = component_cfg.get("backbone_head", {})
        engine_files = {
            "voxel_encoder": voxel_cfg.get("engine_file", "pts_voxel_encoder.engine"),
            "backbone_neck_head": backbone_cfg.get("engine_file", "pts_backbone_neck_head.engine"),
        }

        for component, engine_file in engine_files.items():
            engine_path = osp.join(self.tensorrt_dir, engine_file)

            if not osp.exists(engine_path):
                raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

            try:
                # Load engine
                with open(engine_path, "rb") as f:
                    engine = runtime.deserialize_cuda_engine(f.read())

                if engine is None:
                    raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

                # Create execution context
                context = engine.create_execution_context()

                # Check if context creation succeeded
                if context is None:
                    raise RuntimeError(
                        f"Failed to create execution context for {component}. "
                        "This is likely due to GPU out-of-memory. "
                        "Try reducing batch size or closing other GPU processes."
                    )

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

        if context is None:
            raise RuntimeError("voxel_encoder context is None - likely failed to initialize due to GPU OOM")

        # Convert to numpy
        input_array = input_features.cpu().numpy().astype(np.float32)
        if not input_array.flags["C_CONTIGUOUS"]:
            input_array = np.ascontiguousarray(input_array)

        # Get tensor names
        input_name, output_name = self._get_io_names(engine, single_output=True)

        # Set input shape and get output shape
        context.set_input_shape(input_name, input_array.shape)
        output_shape = context.get_tensor_shape(output_name)
        output_array = np.empty(output_shape, dtype=np.float32)
        if not output_array.flags["C_CONTIGUOUS"]:
            output_array = np.ascontiguousarray(output_array)

        # Use resource manager for automatic cleanup
        with TensorRTResourceManager() as manager:
            d_input = manager.allocate(input_array.nbytes)
            d_output = manager.allocate(output_array.nbytes)
            stream = manager.get_stream()

            context.set_tensor_address(input_name, int(d_input))
            context.set_tensor_address(output_name, int(d_output))

            # Memory transfer: CPU -> GPU (not timed)
            cuda.memcpy_htod_async(d_input, input_array, stream)

            # Record start event and execute inference (pure GPU time)
            self._voxel_encoder_start_event.record(stream)
            context.execute_async_v3(stream_handle=stream.handle)
            self._voxel_encoder_end_event.record(stream)

            # Memory transfer: GPU -> CPU (not timed)
            cuda.memcpy_dtoh_async(output_array, d_output, stream)
            manager.synchronize()

        voxel_features = torch.from_numpy(output_array).to(self.device)
        voxel_features = voxel_features.squeeze(1)
        return voxel_features

    def _get_io_names(self, engine, single_output: bool = False):
        """Extract input/output tensor names from TensorRT engine."""
        input_name = None
        output_names = []

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                input_name = tensor_name
            elif engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                output_names.append(tensor_name)

        if input_name is None:
            raise RuntimeError("Could not find input tensor name")
        if not output_names:
            raise RuntimeError("Could not find output tensor names")

        if single_output:
            return input_name, output_names[0]
        return input_name, output_names

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

        if context is None:
            raise RuntimeError("backbone_neck_head context is None - likely failed to initialize due to GPU OOM")

        # Convert to numpy
        input_array = spatial_features.cpu().numpy().astype(np.float32)
        if not input_array.flags["C_CONTIGUOUS"]:
            input_array = np.ascontiguousarray(input_array)

        # Get tensor names
        input_name, output_names = self._get_io_names(engine, single_output=False)

        # Set input shape
        context.set_input_shape(input_name, input_array.shape)

        # Prepare output arrays
        output_arrays = {}
        for output_name in output_names:
            output_shape = context.get_tensor_shape(output_name)
            output_array = np.empty(output_shape, dtype=np.float32)
            if not output_array.flags["C_CONTIGUOUS"]:
                output_array = np.ascontiguousarray(output_array)
            output_arrays[output_name] = output_array

        # Use resource manager for automatic cleanup
        with TensorRTResourceManager() as manager:
            d_input = manager.allocate(input_array.nbytes)
            d_outputs = {name: manager.allocate(arr.nbytes) for name, arr in output_arrays.items()}
            stream = manager.get_stream()

            context.set_tensor_address(input_name, int(d_input))
            for output_name in output_names:
                context.set_tensor_address(output_name, int(d_outputs[output_name]))

            # Memory transfer: CPU -> GPU (not timed)
            cuda.memcpy_htod_async(d_input, input_array, stream)

            # Record start event and execute inference (pure GPU time)
            self._backbone_start_event.record(stream)
            context.execute_async_v3(stream_handle=stream.handle)
            self._backbone_end_event.record(stream)

            # Memory transfer: GPU -> CPU (not timed)
            for output_name in output_names:
                cuda.memcpy_dtoh_async(output_arrays[output_name], d_outputs[output_name], stream)

            manager.synchronize()

        head_outputs = [torch.from_numpy(output_arrays[name]).to(self.device) for name in output_names]

        if len(head_outputs) != 6:
            raise ValueError(f"Expected 6 head outputs, got {len(head_outputs)}")

        return head_outputs

    def run_model(self, preprocessed_input: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        """
        Run complete multi-stage model inference with GPU timing using CUDA events.

        This override uses CUDA events to measure pure GPU inference time for
        TensorRT operations, matching the C++ implementation's timing methodology.

        Args:
            preprocessed_input: Dict from preprocess() containing:
                - 'input_features': Input features for voxel encoder [N_voxels, max_points, 11]
                - 'coors': Voxel coordinates [N_voxels, 4]
                - 'voxels': Raw voxel data
                - 'num_points': Number of points per voxel

        Returns:
            Tuple of (head_outputs, stage_latencies):
            - head_outputs: List of head outputs [heatmap, reg, height, dim, rot, vel]
            - stage_latencies: Dict mapping stage names to latency in ms
                - 'voxel_encoder_ms': Pure GPU inference time (CUDA events)
                - 'middle_encoder_ms': Wall-clock time (PyTorch)
                - 'backbone_head_ms': Pure GPU inference time (CUDA events)
        """
        # Use local variable for thread safety
        stage_latencies = {}

        # Stage 1: Voxel Encoder (TensorRT - GPU timing)
        voxel_features = self.run_voxel_encoder(preprocessed_input["input_features"])
        # Calculate GPU time from CUDA events (pure inference time, no memory transfers)
        # Note: manager.synchronize() in run_voxel_encoder ensures events are recorded
        # Synchronize end event to ensure it's complete before querying time
        self._voxel_encoder_end_event.synchronize()
        voxel_encoder_gpu_time_ms = self._voxel_encoder_end_event.time_since(self._voxel_encoder_start_event)
        stage_latencies["voxel_encoder_ms"] = voxel_encoder_gpu_time_ms

        # Stage 2: Middle Encoder (PyTorch - wall-clock timing)
        start = time.perf_counter()
        spatial_features = self.process_middle_encoder(voxel_features, preprocessed_input["coors"])
        stage_latencies["middle_encoder_ms"] = (time.perf_counter() - start) * 1000

        # Stage 3: Backbone + Head (TensorRT - GPU timing)
        head_outputs = self.run_backbone_head(spatial_features)
        # Calculate GPU time from CUDA events (pure inference time, no memory transfers)
        # Note: manager.synchronize() in run_backbone_head ensures events are recorded
        # Synchronize end event to ensure it's complete before querying time
        self._backbone_end_event.synchronize()
        backbone_head_gpu_time_ms = self._backbone_end_event.time_since(self._backbone_start_event)
        stage_latencies["backbone_head_ms"] = backbone_head_gpu_time_ms

        return head_outputs, stage_latencies

    def _release_gpu_resources(self) -> None:
        """Release TensorRT engines and contexts (GPUResourceMixin implementation)."""
        # Destroy CUDA events
        if hasattr(self, "_backbone_start_event"):
            try:
                del self._backbone_start_event
            except Exception:
                pass
        if hasattr(self, "_backbone_end_event"):
            try:
                del self._backbone_end_event
            except Exception:
                pass
        if hasattr(self, "_voxel_encoder_start_event"):
            try:
                del self._voxel_encoder_start_event
            except Exception:
                pass
        if hasattr(self, "_voxel_encoder_end_event"):
            try:
                del self._voxel_encoder_end_event
            except Exception:
                pass

        release_tensorrt_resources(
            engines=getattr(self, "_engines", None),
            contexts=getattr(self, "_contexts", None),
        )
