"""BEVFusion TensorRT Pipeline Implementation."""

from __future__ import annotations

import logging
import os.path as osp
from typing import Dict, List, Optional, Tuple

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
from deployment.projects.bevfusion.io.component_utils import is_split_bevfusion_components
from deployment.projects.bevfusion.pipelines.bevfusion_pipeline import BEVFusionDeploymentPipeline

logger = logging.getLogger(__name__)


class _TRTLayerProfiler(trt.IProfiler):
    """Collects per-layer execution times for TensorRT engine."""

    def __init__(self) -> None:
        try:
            trt.IProfiler.__init__(self)
        except Exception:
            pass
        self.layer_times: List[Tuple[str, float]] = []

    def report_layer_time(self, layer_name: str, ms: float) -> None:
        self.layer_times.append((str(layer_name), float(ms)))


def _aggregate_trt_layers_to_stages(layer_times: List[Tuple[str, float]]) -> Dict[str, float]:
    """Map TensorRT layer names to BEVFusion stages and sum times (ms).

    Uses substring matching on layer names (e.g. from ONNX export).
    Order of patterns matters: first match wins. Unmatched layers go to bevfusion_ms.
    """
    stage_sums: Dict[str, float] = {
        "voxel_encoder_ms": 0.0,
        "sparse_encoder_ms": 0.0,
        "backbone_ms": 0.0,
        "neck_ms": 0.0,
        "head_ms": 0.0,
        "post_scoring_ms": 0.0,
        "bevfusion_ms": 0.0,
    }
    # Patterns (substring in layer name) -> stage key. Check in order.
    stage_patterns: List[Tuple[str, str]] = [
        ("pts_middle_encoder", "sparse_encoder_ms"),
        ("middle_encoder", "sparse_encoder_ms"),
        ("encoder_layer", "sparse_encoder_ms"),
        ("conv_input", "sparse_encoder_ms"),
        ("pts_backbone", "backbone_ms"),
        ("backbone", "backbone_ms"),
        ("blocks.", "backbone_ms"),
        ("pts_neck", "neck_ms"),
        ("neck", "neck_ms"),
        ("deblocks", "neck_ms"),
        ("bbox_head", "head_ms"),
        ("heatmap", "head_ms"),
        ("shared_conv", "head_ms"),
        ("sigmoid", "post_scoring_ms"),
        ("query_labels", "post_scoring_ms"),
        ("post_scoring", "post_scoring_ms"),
        ("voxel", "voxel_encoder_ms"),
    ]
    for layer_name, ms in layer_times:
        name_lower = layer_name.lower()
        assigned = False
        for pattern, stage_key in stage_patterns:
            if pattern.lower() in name_lower or (pattern in layer_name):
                stage_sums[stage_key] += ms
                assigned = True
                break
        if not assigned:
            stage_sums["bevfusion_ms"] += ms
    return stage_sums


class BEVFusionTensorRTPipeline(GPUResourceMixin, BEVFusionDeploymentPipeline):
    """TensorRT-based BEVFusion pipeline.

    Single engine (full graph) or split sparse + dense engines.
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
        self._split = is_split_bevfusion_components(components_cfg)
        self._engine = None
        self._context = None
        self._engine_sparse = None
        self._context_sparse = None
        self._engine_dense = None
        self._context_dense = None

        self._start_event = cuda.Event()
        self._end_event = cuda.Event()

        self._load_tensorrt_engine()
        logger.info(f"BEVFusion TensorRT pipeline initialized from: {tensorrt_dir} (split={self._split})")

    def _load_tensorrt_engine(self) -> None:
        load_tensorrt_plugin_libraries(logger, self._plugin_libraries)
        trt.init_libnvinfer_plugins(self._trt_logger, "")
        runtime = trt.Runtime(self._trt_logger)

        if self._split:
            sparse_path = resolve_artifact_path(
                base_dir=self.tensorrt_dir,
                components_cfg=self._components_cfg,
                component_name="bevfusion_sparse",
                file_key="engine_file",
            )
            dense_path = resolve_artifact_path(
                base_dir=self.tensorrt_dir,
                components_cfg=self._components_cfg,
                component_name="bevfusion_dense",
                file_key="engine_file",
            )
            if not osp.exists(sparse_path):
                raise FileNotFoundError(f"Sparse TensorRT engine not found: {sparse_path}")
            if not osp.exists(dense_path):
                raise FileNotFoundError(f"Dense TensorRT engine not found: {dense_path}")

            with open(sparse_path, "rb") as f:
                self._engine_sparse = runtime.deserialize_cuda_engine(f.read())
            with open(dense_path, "rb") as f:
                self._engine_dense = runtime.deserialize_cuda_engine(f.read())
            if self._engine_sparse is None or self._engine_dense is None:
                raise RuntimeError("Failed to deserialize split TensorRT engines")

            self._context_sparse = self._engine_sparse.create_execution_context()
            self._context_dense = self._engine_dense.create_execution_context()
            if self._context_sparse is None or self._context_dense is None:
                raise RuntimeError("Failed to create TensorRT contexts for split engines")
            logger.info("Loaded split TensorRT engines: %s , %s", sparse_path, dense_path)
            return

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

    def _trt_infer_voxel_inputs(
        self,
        engine: trt.ICudaEngine,
        context: trt.IExecutionContext,
        voxels_np: np.ndarray,
        coors_np: np.ndarray,
        num_points_np: np.ndarray,
        profiler: Optional[_TRTLayerProfiler],
        record_cuda_events: bool,
    ) -> Dict[str, np.ndarray]:
        input_map: Dict[str, np.ndarray] = {}
        output_names: List[str] = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                ln = name.lower()
                if "voxel" in ln and "num" not in ln:
                    input_map[name] = voxels_np
                elif "coor" in ln:
                    input_map[name] = coors_np
                elif "num" in ln:
                    input_map[name] = num_points_np
            else:
                output_names.append(name)
        return self._trt_infer_bound(engine, context, input_map, output_names, profiler, record_cuda_events)

    def _trt_infer_named_input(
        self,
        engine: trt.ICudaEngine,
        context: trt.IExecutionContext,
        input_map: Dict[str, np.ndarray],
        profiler: Optional[_TRTLayerProfiler],
        record_cuda_events: bool,
    ) -> Dict[str, np.ndarray]:
        output_names: List[str] = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                output_names.append(name)
        return self._trt_infer_bound(engine, context, input_map, output_names, profiler, record_cuda_events)

    def _trt_infer_bound(
        self,
        engine: trt.ICudaEngine,
        context: trt.IExecutionContext,
        input_map: Dict[str, np.ndarray],
        output_names: List[str],
        profiler: Optional[_TRTLayerProfiler],
        record_cuda_events: bool,
    ) -> Dict[str, np.ndarray]:
        for name, arr in input_map.items():
            context.set_input_shape(name, arr.shape)

        output_arrays: Dict[str, np.ndarray] = {}
        for name in output_names:
            shape = context.get_tensor_shape(name)
            trt_dtype = engine.get_tensor_dtype(name)
            np_dtype = self._trt_dtype_to_numpy(trt_dtype)
            arr = np.empty(shape, dtype=np_dtype)
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            output_arrays[name] = arr

        prev_profiler = None
        if profiler is not None and hasattr(context, "profiler"):
            prev_profiler = getattr(context, "profiler", None)
            context.profiler = profiler
            profiler.layer_times.clear()

        try:
            with TensorRTResourceManager() as mgr:
                d_inputs = {name: mgr.allocate(arr.nbytes) for name, arr in input_map.items()}
                d_outputs = {name: mgr.allocate(arr.nbytes) for name, arr in output_arrays.items()}
                stream = mgr.stream

                for name, arr in input_map.items():
                    context.set_tensor_address(name, int(d_inputs[name]))
                    cuda.memcpy_htod_async(d_inputs[name], arr, stream)

                for name in output_names:
                    context.set_tensor_address(name, int(d_outputs[name]))

                if record_cuda_events:
                    self._start_event.record(stream)
                ok = context.execute_async_v3(stream_handle=stream.handle)
                if not ok:
                    raise RuntimeError("TensorRT execute_async_v3 returned failure status.")
                if record_cuda_events:
                    self._end_event.record(stream)

                for name in output_names:
                    cuda.memcpy_dtoh_async(output_arrays[name], d_outputs[name], stream)

                mgr.synchronize()
        finally:
            if profiler is not None and hasattr(context, "profiler"):
                context.profiler = prev_profiler

        return output_arrays

    @override
    def run_bevfusion(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
        profiler: _TRTLayerProfiler | None = None,
    ) -> List[torch.Tensor]:
        voxels_np = self.to_numpy(voxels, dtype=np.float32)
        coors_np = self.to_numpy(coors, dtype=np.int32)
        num_points_np = self.to_numpy(num_points_per_voxel, dtype=np.int32)

        if self._split:
            assert self._engine_sparse is not None and self._context_sparse is not None
            assert self._engine_dense is not None and self._context_dense is not None

            self._start_event.record()
            sparse_out = self._trt_infer_voxel_inputs(
                self._engine_sparse,
                self._context_sparse,
                voxels_np,
                coors_np,
                num_points_np,
                profiler,
                record_cuda_events=False,
            )
            if len(sparse_out) != 1:
                raise RuntimeError(f"Sparse engine: expected 1 output, got {list(sparse_out.keys())}")
            bev_name = next(iter(sparse_out))
            bev_arr = np.ascontiguousarray(sparse_out[bev_name].astype(np.float32))

            dense_in_name = None
            for i in range(self._engine_dense.num_io_tensors):
                n = self._engine_dense.get_tensor_name(i)
                if self._engine_dense.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                    dense_in_name = n
                    break
            if dense_in_name is None:
                raise RuntimeError("Dense engine has no input tensor")

            dense_out = self._trt_infer_named_input(
                self._engine_dense,
                self._context_dense,
                {dense_in_name: bev_arr},
                profiler,
                record_cuda_events=False,
            )
            self._end_event.record()

            component_cfg = self._components_cfg.get_component("bevfusion_dense")
            expected_output_names = [out.name for out in component_cfg.io.outputs]
            out_keys = list(dense_out.keys())
            ordered_names = [n for n in expected_output_names if n in dense_out]
            ordered_names += [n for n in out_keys if n not in ordered_names]
            return [torch.from_numpy(dense_out[name]).to(self.torch_device) for name in ordered_names]

        engine = self._engine
        context = self._context
        assert engine is not None and context is not None

        output_arrays = self._trt_infer_voxel_inputs(
            engine, context, voxels_np, coors_np, num_points_np, profiler, record_cuda_events=True
        )
        output_names = list(output_arrays.keys())

        component_cfg = self._components_cfg.get_component("bevfusion_main_body")
        expected_output_names = [out.name for out in component_cfg.io.outputs]
        ordered_names = [n for n in expected_output_names if n in output_arrays]
        ordered_names += [n for n in output_names if n not in ordered_names]
        return [torch.from_numpy(output_arrays[name]).to(self.torch_device) for name in ordered_names]

    # Stage keys aligned with BEVFusionPyTorchPipeline for consistent Stage-wise Latency Breakdown.
    BEVFUSION_STAGE_KEYS = (
        "voxel_encoder_ms",
        "sparse_encoder_ms",
        "backbone_ms",
        "neck_ms",
        "head_ms",
        "post_scoring_ms",
        "bevfusion_ms",
    )

    @override
    def run_model(self, preprocessed_input: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        stage_latencies: Dict[str, float] = {k: 0.0 for k in self.BEVFUSION_STAGE_KEYS}

        profiler = _TRTLayerProfiler()
        outputs = self.run_bevfusion(
            preprocessed_input["voxels"],
            preprocessed_input["coors"],
            preprocessed_input["num_points_per_voxel"],
            profiler=profiler,
        )

        self._end_event.synchronize()
        gpu_time_ms = self._end_event.time_since(self._start_event)
        stage_latencies["bevfusion_ms"] = gpu_time_ms

        if profiler.layer_times:
            aggregated = _aggregate_trt_layers_to_stages(profiler.layer_times)
            for k in self.BEVFUSION_STAGE_KEYS:
                if k == "bevfusion_ms":
                    continue
                if k in aggregated and aggregated[k] > 0:
                    stage_latencies[k] = aggregated[k]
        # bevfusion_ms stays total engine time; sub-stages from profiler when layer names are available

        return outputs, stage_latencies

    def _release_gpu_resources(self) -> None:
        for attr in ("_start_event", "_end_event"):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass
        if self._split:
            release_tensorrt_resources(
                engines={
                    "sparse": self._engine_sparse,
                    "dense": self._engine_dense,
                },
                contexts={
                    "sparse": self._context_sparse,
                    "dense": self._context_dense,
                },
            )
        else:
            release_tensorrt_resources(
                engines={"main": self._engine} if self._engine else None,
                contexts={"main": self._context} if self._context else None,
            )
