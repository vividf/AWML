"""BEVFusion TensorRT Pipeline Implementation."""

from __future__ import annotations

import logging
import os.path as osp
import time
from typing import Dict, List, Tuple

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
import torch
from typing_extensions import override

from deployment.configs import ComponentsConfig
from deployment.core.artifacts import resolve_artifact_path
from deployment.core.backend import Backend
from deployment.core.device import DeviceSpec
from deployment.core.tensorrt_plugins import load_tensorrt_plugin_libraries
from deployment.pipelines.gpu_resource_mixin import (
    GPUResourceMixin,
    TensorRTResourceManager,
    release_tensorrt_resources,
)
from deployment.projects.bevfusion.pipelines.bevfusion_pipeline import BEVFusionDeploymentPipeline

logger = logging.getLogger(__name__)


class BEVFusionTensorRTPipeline(GPUResourceMixin, BEVFusionDeploymentPipeline):
    """TensorRT-based BEVFusion pipeline.

    Loads a single TensorRT engine for BEVFusion with dynamic voxel count.
    """

    def __init__(
        self,
        pytorch_model: torch.nn.Module,
        tensorrt_dir: str,
        device: DeviceSpec,
        components_cfg: ComponentsConfig,
        plugin_libraries: Tuple[str, ...] = (),
    ) -> None:
        super().__init__(pytorch_model=pytorch_model, backend_type=Backend.TENSORRT, device=device)

        self.tensorrt_dir = tensorrt_dir
        self._components_cfg = components_cfg
        self._plugin_libraries = plugin_libraries
        self._trt_logger = trt.Logger(trt.Logger.WARNING)
        self._engine = None
        self._context = None

        self._start_event = cuda.Event()
        self._end_event = cuda.Event()

        self._load_tensorrt_engine()
        logger.info(f"BEVFusion TensorRT pipeline initialized from: {tensorrt_dir}")

    def _load_tensorrt_engine(self) -> None:
        load_tensorrt_plugin_libraries(logger, self._plugin_libraries)
        trt.init_libnvinfer_plugins(self._trt_logger, "")
        runtime = trt.Runtime(self._trt_logger)

        engine_path = resolve_artifact_path(
            base_dir=self.tensorrt_dir,
            components_cfg=self._components_cfg,
            component_name="bevfusion_main_body",
            file_key="engine_file",
        )
        if not osp.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Failed to create TensorRT execution context (OOM?)")

        logger.info(f"Loaded TensorRT engine: {engine_path}")

    @staticmethod
    def _trt_dtype_to_numpy(trt_dtype: trt.DataType) -> np.dtype:
        """Map TensorRT dtype to numpy dtype for correctly sized host buffers."""
        try:
            return np.dtype(trt.nptype(trt_dtype))
        except Exception:
            # Safe fallback for older/newer TRT dtype variations.
            mapping = {}
            for key, npdt in (
                ("FLOAT", np.float32),
                ("HALF", np.float16),
                ("INT8", np.int8),
                ("INT32", np.int32),
                ("BOOL", np.bool_),
                ("UINT8", np.uint8),
                ("FP8", np.float16),
                ("BF16", np.float16),
                ("INT64", np.int64),
            ):
                dt = getattr(trt.DataType, key, None)
                if dt is not None:
                    mapping[dt] = npdt
            return np.dtype(mapping.get(trt_dtype, np.float32))

    @override
    def run_bevfusion(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
    ) -> List[torch.Tensor]:
        engine = self._engine
        context = self._context

        voxels_np = self.to_numpy(voxels, dtype=np.float32)
        coors_np = self.to_numpy(coors, dtype=np.int32)
        num_points_np = self.to_numpy(num_points_per_voxel, dtype=np.int32)

        input_map = {}
        output_names = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if "voxel" in name.lower() and "num" not in name.lower():
                    input_map[name] = voxels_np
                elif "coor" in name.lower():
                    input_map[name] = coors_np
                elif "num" in name.lower():
                    input_map[name] = num_points_np
            else:
                output_names.append(name)

        for name, arr in input_map.items():
            context.set_input_shape(name, arr.shape)

        output_arrays = {}
        for name in output_names:
            shape = context.get_tensor_shape(name)
            trt_dtype = engine.get_tensor_dtype(name)
            np_dtype = self._trt_dtype_to_numpy(trt_dtype)
            arr = np.empty(shape, dtype=np_dtype)
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            output_arrays[name] = arr

        with TensorRTResourceManager() as mgr:
            d_inputs = {name: mgr.allocate(arr.nbytes) for name, arr in input_map.items()}
            d_outputs = {name: mgr.allocate(arr.nbytes) for name, arr in output_arrays.items()}
            stream = mgr.stream

            for name, arr in input_map.items():
                context.set_tensor_address(name, int(d_inputs[name]))
                cuda.memcpy_htod_async(d_inputs[name], arr, stream)

            for name in output_names:
                context.set_tensor_address(name, int(d_outputs[name]))

            self._start_event.record(stream)
            ok = context.execute_async_v3(stream_handle=stream.handle)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 returned failure status.")
            self._end_event.record(stream)

            for name in output_names:
                cuda.memcpy_dtoh_async(output_arrays[name], d_outputs[name], stream)

            mgr.synchronize()

        # Keep output order consistent with config / ONNX export wrapper:
        # [bbox_pred, score, label_pred].
        component_cfg = self._components_cfg.get_component("bevfusion_main_body")
        expected_output_names = [out.name for out in component_cfg.io.outputs]
        ordered_names = [n for n in expected_output_names if n in output_arrays]
        ordered_names += [n for n in output_names if n not in ordered_names]
        return [torch.from_numpy(output_arrays[name]).to(self.torch_device) for name in ordered_names]

    @override
    def run_model(self, preprocessed_input: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        stage_latencies: Dict[str, float] = {}

        outputs = self.run_bevfusion(
            preprocessed_input["voxels"],
            preprocessed_input["coors"],
            preprocessed_input["num_points_per_voxel"],
        )

        self._end_event.synchronize()
        gpu_time_ms = self._end_event.time_since(self._start_event)
        stage_latencies["bevfusion_ms"] = gpu_time_ms

        return outputs, stage_latencies

    def _release_gpu_resources(self) -> None:
        for attr in ("_start_event", "_end_event"):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass
        release_tensorrt_resources(
            engines={"main": self._engine} if self._engine else None,
            contexts={"main": self._context} if self._context else None,
        )
