"""BEVFusion TensorRT Pipeline Implementation."""

from __future__ import annotations

import logging
import os
import os.path as osp
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
import torch
from typing_extensions import override

from deployment.config.enums import Backend
from deployment.config.schema import ComponentsConfig
from deployment.inference.gpu_resource_mixin import (
    GPUResourceMixin,
    TensorRTResourceManager,
    release_tensorrt_resources,
)
from deployment.primitives.artifacts import resolve_artifact_path
from deployment.primitives.device import DeviceSpec
from deployment.primitives.tensorrt_plugins import load_tensorrt_plugin_libraries
from deployment.projects.bevfusion.inference.bevfusion_inference_pipeline import BEVFusionDeploymentPipeline
from deployment.projects.bevfusion.inference.trt_profiling import (
    _SPARSE_BUCKET_ORDER,
    _scale_dense_substages,
    _sum_layers_by_stage,
    _summarize_sparse_layers,
    _TRTLayerProfiler,
)
from deployment.projects.bevfusion.io.component_utils import has_component, is_split_bevfusion_components
from deployment.projects.bevfusion.io.coors_contract import voxel_indices_xyz_to_graph_input_zyx

logger = logging.getLogger(__name__)


def _env_truthy(key: str) -> bool:
    return os.environ.get(key, "").strip().lower() in ("1", "true", "yes")


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)).strip())
    except ValueError:
        return default


_TRT_DEBUG_SPLIT = _env_truthy("BEVFUSION_TRT_DEBUG_SPLIT")
_TRT_LOG_IO = _env_truthy("BEVFUSION_TRT_LOG_IO")
# Priority A — in-situ per-layer breakdown for the split sparse engine.
#   BEVFUSION_TRT_SPARSE_PROFILE=1          attaches trt.IProfiler to the sparse context
#   BEVFUSION_TRT_SPARSE_PROFILE_EVERY=1    log breakdown on every frame (default: every 10 frames)
# The standalone tool ``benchmark/profile_sparse_encoder.py`` produces a cleaner report;
# this env var is a sanity overlay when running the real eval path (step 5).
_TRT_SPARSE_PROFILE = _env_truthy("BEVFUSION_TRT_SPARSE_PROFILE")
_TRT_SPARSE_PROFILE_EVERY = max(1, _env_int("BEVFUSION_TRT_SPARSE_PROFILE_EVERY", 10))
# ImplicitGemmInt8 TRT plugin (C++): per-layer FP16 output stats to stderr after each enqueue.
#   BEVFUSION_INT8_GEMM_DEBUG=1
#   BEVFUSION_INT8_GEMM_DEBUG_MAX=60   # max layer-dumps (default 60 ≈ 3 sparse passes × 20 layers)
# Rebuild: deployment/projects/bevfusion/cpp/int8_plugin/
#
# PyTorch sparse conv hooks (align seq with TRT): BEVFUSION_SPARSE_ENCODER_HOOK_DEBUG=1
#   BEVFUSION_SPARSE_ENCODER_HOOK_MAX_PASSES=2   # full pts_middle_encoder forwards (default 2)
# See deployment/projects/bevfusion/debug/sparse_encoder_hooks.py
# First N eval frames: print pooled-voxel + lidar_bev stats to stdout (align with PyTorch pipeline).
_TRT_TENSOR_LOG_FRAMES = max(0, _env_int("BEVFUSION_TRT_TENSOR_LOG_FRAMES", 2))
_TRT_TENSOR_LOG_PREFIX = "[BEVFUSION][TensorRT][tensors]"


def _np_tensor_stats(arr: np.ndarray, name: str) -> str:
    """Compact numpy stats for debug lines (matches PyTorch _tensor_stats fields)."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    nz = int(np.count_nonzero(a))
    return (
        f"{_TRT_TENSOR_LOG_PREFIX} {name}: shape={arr.shape} dtype={arr.dtype} "
        f"min={float(a.min()):.4f} max={float(a.max()):.4f} "
        f"mean={float(a.mean()):.4f} std={float(a.std()):.4f} "
        f"abs_mean={float(np.mean(np.abs(a))):.4f} "
        f"nonzero={nz}/{a.size}"
    )


def _list_trt_io_names(engine: trt.ICudaEngine) -> Tuple[List[str], List[str]]:
    """Return (input_names, output_names) in TensorRT tensor index order."""
    inputs: List[str] = []
    outputs: List[str] = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(name)
        else:
            outputs.append(name)
    return inputs, outputs


def _pick_bound_input_name(engine: trt.ICudaEngine, expected_in_order: Sequence[str]) -> str:
    """Match deploy_cfg input names to the engine; avoid relying on arbitrary TRT ordering."""
    found, _out = _list_trt_io_names(engine)
    for want in expected_in_order:
        if want in found:
            return want
    if len(found) == 1:
        if expected_in_order and found[0] != expected_in_order[0]:
            logger.warning(
                "TensorRT dense engine input is %r but deploy_cfg expects %r — using engine binding. "
                "If mAP=0, verify ONNX export names match deploy components.bevfusion_dense.io.inputs.",
                found[0],
                expected_in_order[0],
            )
        return found[0]
    raise RuntimeError(f"Could not map deploy_cfg inputs {list(expected_in_order)} to engine inputs {found}")


def _log_engine_schema(tag: str, engine: trt.ICudaEngine) -> None:
    ins, outs = _list_trt_io_names(engine)
    lines = [f"[trt-io] {tag} inputs={ins} outputs={outs}"]
    for name in ins + outs:
        shp = engine.get_tensor_shape(name)
        dt = engine.get_tensor_dtype(name)
        lines.append(f"[trt-io]   {name}: shape={shp} dtype={dt}")
    logger.warning("\n".join(lines))


def _log_engine_input_dtypes_line(tag: str, engine: trt.ICudaEngine) -> None:
    """Log binding dtypes (P1: FP32 vs FP16 voxels for split sparse engines).

    HALF voxel bindings are supported via ``_host_buffer_for_engine_tensor``; this line makes
    the contract visible without enabling ``BEVFUSION_TRT_LOG_IO=1``.
    """
    parts: List[str] = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            continue
        dt = engine.get_tensor_dtype(name)
        parts.append(f"{name}={dt}")
        ln = name.lower()
        if "voxel" in ln and "num" not in ln and dt == trt.DataType.HALF:
            logger.warning(
                "[trt-io] %s: voxel-like input %r is HALF — host numpy is cast before "
                "``set_tensor_address`` (see INFO ``casting host buffer`` on first infer). "
                "If that cast is missing, ImplicitGemmInt8 inputs corrupt (lidar_bev explodes).",
                tag,
                name,
            )
    if parts:
        logger.info("[trt-io] %s engine INPUT dtypes: %s", tag, ", ".join(parts))


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
        split_layout = is_split_bevfusion_components(components_cfg)
        merged_engine_available = False
        if split_layout and has_component(components_cfg, "bevfusion_main_body"):
            merged_engine_path = resolve_artifact_path(
                base_dir=tensorrt_dir,
                components_cfg=components_cfg,
                component_name="bevfusion_main_body",
                file_key="engine_file",
            )
            merged_engine_available = osp.exists(merged_engine_path)
        self._split = split_layout and not merged_engine_available
        self._engine = None
        self._context = None
        self._engine_sparse = None
        self._context_sparse = None
        self._engine_dense = None
        self._context_dense = None

        self._start_event = cuda.Event()
        self._end_event = cuda.Event()
        # Split-engine GPU intervals (same stream as each TRT execute, excludes D2H).
        self._sparse_ev_s = cuda.Event()
        self._sparse_ev_e = cuda.Event()
        self._dense_ev_s = cuda.Event()
        self._dense_ev_e = cuda.Event()
        self._last_split_sparse_gpu_ms: float = 0.0
        self._last_split_dense_gpu_ms: float = 0.0
        self._split_debug_frames_done: int = 0
        self._split_debug_max: int = max(0, _env_int("BEVFUSION_TRT_DEBUG_SPLIT_FRAMES", 2))
        self._split_tensor_log_frames_done: int = 0
        # Priority A: accumulators for sparse encoder bucket breakdown across eval frames.
        self._sparse_profile_frame_count: int = 0
        self._sparse_profile_bucket_sum: Dict[str, float] = {b: 0.0 for b in _SPARSE_BUCKET_ORDER}
        self._sparse_profile_top_layers: Dict[str, float] = {}  # name -> accumulated ms
        self._last_sparse_profile_buckets: Dict[str, float] = {}

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
            assert self._engine_sparse is not None and self._engine_dense is not None
            _log_engine_input_dtypes_line("bevfusion_sparse", self._engine_sparse)
            _log_engine_input_dtypes_line("bevfusion_dense", self._engine_dense)
            if _TRT_LOG_IO:
                _log_engine_schema("bevfusion_sparse", self._engine_sparse)
                _log_engine_schema("bevfusion_dense", self._engine_dense)
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

    def _host_buffer_for_engine_tensor(self, engine: trt.ICudaEngine, tensor_name: str, arr: np.ndarray) -> np.ndarray:
        """Cast / layout host memory to match *engine* binding dtype (critical for FP16 engines).

        Split sparse ONNX is often traced with FP32 voxels, but TensorRT ``fp16`` builds may bind
        ``voxels`` as ``HALF``. Feeding float32 nbytes into a HALF binding misaligns the GPU
        buffer and corrupts the first ImplicitGemmInt8 inputs (lidar_bev explosion while numpy
        voxel stats still look sane).
        """
        trt_dtype = engine.get_tensor_dtype(tensor_name)
        want = self._trt_dtype_to_numpy(trt_dtype)
        if arr.dtype != want:
            logger.info(
                "[trt-io] casting host buffer for tensor %r: numpy %s → %s (engine binding %s)",
                tensor_name,
                arr.dtype,
                want,
                trt_dtype,
            )
            arr = np.asarray(arr, dtype=want)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return arr

    def _trt_infer_voxel_inputs(
        self,
        engine: trt.ICudaEngine,
        context: trt.IExecutionContext,
        voxels_np: np.ndarray,
        coors_np: np.ndarray,
        num_points_np: np.ndarray,
        profiler: Optional[_TRTLayerProfiler],
        gpu_interval_events: Optional[Tuple[cuda.Event, cuda.Event]],
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
        return self._trt_infer_bound(engine, context, input_map, output_names, profiler, gpu_interval_events)

    def _trt_infer_named_input(
        self,
        engine: trt.ICudaEngine,
        context: trt.IExecutionContext,
        input_map: Dict[str, np.ndarray],
        profiler: Optional[_TRTLayerProfiler],
        gpu_interval_events: Optional[Tuple[cuda.Event, cuda.Event]],
    ) -> Dict[str, np.ndarray]:
        output_names: List[str] = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                output_names.append(name)
        return self._trt_infer_bound(engine, context, input_map, output_names, profiler, gpu_interval_events)

    def _trt_infer_bound(
        self,
        engine: trt.ICudaEngine,
        context: trt.IExecutionContext,
        input_map: Dict[str, np.ndarray],
        output_names: List[str],
        profiler: Optional[_TRTLayerProfiler],
        gpu_interval_events: Optional[Tuple[cuda.Event, cuda.Event]],
    ) -> Dict[str, np.ndarray]:
        input_map = {name: self._host_buffer_for_engine_tensor(engine, name, arr) for name, arr in input_map.items()}
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

                if gpu_interval_events is not None:
                    gpu_interval_events[0].record(stream)
                ok = context.execute_async_v3(stream_handle=stream.handle)
                if not ok:
                    raise RuntimeError("TensorRT execute_async_v3 returned failure status.")
                if gpu_interval_events is not None:
                    gpu_interval_events[1].record(stream)

                for name in output_names:
                    cuda.memcpy_dtoh_async(output_arrays[name], d_outputs[name], stream)

                mgr.synchronize()
        finally:
            # TensorRT rejects setProfiler(nullptr); only restore when a previous profiler existed.
            if profiler is not None and hasattr(context, "profiler") and prev_profiler is not None:
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
        coors_np = self.to_numpy(voxel_indices_xyz_to_graph_input_zyx(coors), dtype=np.int32)
        num_points_np = self.to_numpy(num_points_per_voxel, dtype=np.int32)
        # Match ``extract_pts_feat`` / PTQ: mean-pool must not divide by zero (NaN BEV → dense NaN).
        num_points_np = np.maximum(num_points_np, 1)

        if self._split:
            assert self._engine_sparse is not None and self._context_sparse is not None
            assert self._engine_dense is not None and self._context_dense is not None

            sparse_cfg = self._components_cfg.get_component("bevfusion_sparse")
            dense_cfg = self._components_cfg.get_component("bevfusion_dense")
            exp_sparse_out = [o.name for o in sparse_cfg.io.outputs]
            exp_dense_in = [i.name for i in dense_cfg.io.inputs]

            do_tensor_log = _TRT_TENSOR_LOG_FRAMES > 0 and self._split_tensor_log_frames_done < _TRT_TENSOR_LOG_FRAMES
            if do_tensor_log:
                self._split_tensor_log_frames_done += 1
                fi = self._split_tensor_log_frames_done
                print(
                    f"{_TRT_TENSOR_LOG_PREFIX} frame={fi}/{_TRT_TENSOR_LOG_FRAMES} "
                    f"(sparse TRT engine → lidar_bev → dense TRT engine → bbox/score/label)"
                )
                voxelize_reduce = getattr(self.pytorch_model, "voxelize_reduce", True)
                if voxelize_reduce and voxels_np.ndim == 3:
                    # Match pytorch.py: [N,P,C].sum(1) / npt with npt [N,1] → [N,C] (not [N] / [N,C]).
                    npt = np.maximum(num_points_np.astype(np.float32).reshape(-1, 1), 1.0)
                    voxel_feat_np = voxels_np.sum(axis=1, keepdims=False) / npt
                    print(_np_tensor_stats(voxel_feat_np, "voxel_features_input (numpy mean-pool, same as PyTorch)"))
                elif voxels_np.ndim == 2:
                    print(
                        _np_tensor_stats(
                            voxels_np,
                            "voxel_features_input (already [N,C], no per-point dim — same as fed to TRT)",
                        )
                    )
                else:
                    print(
                        f"{_TRT_TENSOR_LOG_PREFIX} voxel_features_input: skipped "
                        f"(voxelize_reduce={voxelize_reduce}, voxels_ndim={voxels_np.ndim})"
                    )

            # Sparse (spconv) engine: CUDA-timed separately. If BEVFUSION_TRT_SPARSE_PROFILE=1,
            # also attach a dedicated IProfiler here so we can answer Priority A's question
            # ("where does the sparse time actually go?") without a separate run.
            sparse_profiler: Optional[_TRTLayerProfiler] = _TRTLayerProfiler() if _TRT_SPARSE_PROFILE else None
            sparse_out = self._trt_infer_voxel_inputs(
                self._engine_sparse,
                self._context_sparse,
                voxels_np,
                coors_np,
                num_points_np,
                profiler=sparse_profiler,
                gpu_interval_events=(self._sparse_ev_s, self._sparse_ev_e),
            )
            if sparse_profiler is not None:
                self._record_sparse_profile(sparse_profiler.layer_times)
            if len(sparse_out) != 1:
                raise RuntimeError(f"Sparse engine: expected 1 output, got {list(sparse_out.keys())}")
            bev_name = next(iter(sparse_out))
            if exp_sparse_out and bev_name not in exp_sparse_out:
                logger.warning(
                    "[trt-split] sparse engine output tensor is %r but deploy_cfg bevfusion_sparse.io.outputs "
                    "names=%s — check ONNX export / TRT binding names.",
                    bev_name,
                    exp_sparse_out,
                )
            bev_arr = np.ascontiguousarray(sparse_out[bev_name].astype(np.float32))

            if do_tensor_log:
                bn = bev_arr.reshape(-1)
                print(_np_tensor_stats(bev_arr, f"sparse_encoder_output ({bev_name}, TRT sparse engine)"))
                if bool(np.isnan(bn).any()) or bool(np.isinf(bn).any()):
                    print(
                        f"{_TRT_TENSOR_LOG_PREFIX} WARNING: lidar_bev has nan={bool(np.isnan(bn).any())} "
                        f"inf={bool(np.isinf(bn).any())}"
                    )

            do_split_dbg = _TRT_DEBUG_SPLIT and self._split_debug_frames_done < self._split_debug_max
            if do_split_dbg:
                self._split_debug_frames_done += 1
                if not do_tensor_log:
                    bn = bev_arr.reshape(-1)
                    logger.warning(
                        "[BEVFUSION][TensorRT][debug-split] frame=%d/%d sparse->dense %s: shape=%s dtype=%s "
                        "min=%.6f max=%.6f mean=%.6f std=%.6f abs_mean=%.6f nan=%s inf=%s",
                        self._split_debug_frames_done,
                        self._split_debug_max,
                        bev_name,
                        bev_arr.shape,
                        bev_arr.dtype,
                        float(bn.min()),
                        float(bn.max()),
                        float(bn.mean()),
                        float(bn.std()),
                        float(np.mean(np.abs(bn))),
                        bool(np.isnan(bn).any()),
                        bool(np.isinf(bn).any()),
                    )

            dense_in_name = _pick_bound_input_name(self._engine_dense, exp_dense_in)

            if do_split_dbg:
                ctx = self._context_dense
                exp_shape = tuple(ctx.get_tensor_shape(dense_in_name))
                logger.warning(
                    "[BEVFUSION][TensorRT][debug-split] dense input %r engine_expected_shape=%s feed_shape=%s "
                    "deploy_cfg_inputs=%s",
                    dense_in_name,
                    exp_shape,
                    bev_arr.shape,
                    exp_dense_in,
                )
                if tuple(bev_arr.shape) != exp_shape and not any(d < 0 for d in exp_shape):
                    logger.warning(
                        "[BEVFUSION][TensorRT][debug-split] SHAPE MISMATCH: lidar_bev numpy shape %s vs TRT context %s — "
                        "dense engine will error or broadcast wrong; common cause: H×W vs export grid.",
                        bev_arr.shape,
                        exp_shape,
                    )

            dense_out = self._trt_infer_named_input(
                self._engine_dense,
                self._context_dense,
                {dense_in_name: bev_arr},
                profiler,
                gpu_interval_events=(self._dense_ev_s, self._dense_ev_e),
            )

            self._sparse_ev_e.synchronize()
            self._dense_ev_e.synchronize()
            self._last_split_sparse_gpu_ms = float(self._sparse_ev_e.time_since(self._sparse_ev_s))
            self._last_split_dense_gpu_ms = float(self._dense_ev_e.time_since(self._dense_ev_s))

            expected_output_names = [out.name for out in dense_cfg.io.outputs]
            out_keys = list(dense_out.keys())
            ordered_names = [n for n in expected_output_names if n in dense_out]
            ordered_names += [n for n in out_keys if n not in ordered_names]
            tensors = [torch.from_numpy(dense_out[name]).to(self.torch_device) for name in ordered_names]
            if do_tensor_log and tensors:
                for i, name in enumerate(ordered_names):
                    t = tensors[i].detach()
                    t_f = t.float().reshape(-1)
                    extra = ""
                    if name == "bbox_pred" and t.ndim >= 2 and t.shape[0] >= 2:
                        cx = t[0].float().reshape(-1)
                        cy = t[1].float().reshape(-1)
                        extra = (
                            f" center_x[min,max]=({float(cx.min()):.4f},{float(cx.max()):.4f}) "
                            f"center_y[min,max]=({float(cy.min()):.4f},{float(cy.max()):.4f})"
                        )
                    if name == "label_pred":
                        lp = t.reshape(-1).long()
                        uniq = torch.unique(lp)
                        extra = (
                            f" label_unique_count={int(uniq.numel())} label_min={int(lp.min())} "
                            f"label_max={int(lp.max())}"
                        )
                    if name == "score":
                        extra = (
                            f" score>0.1_count={int((t_f > 0.1).sum())} " f"score>0.5_count={int((t_f > 0.5).sum())}"
                        )
                    print(
                        f"{_TRT_TENSOR_LOG_PREFIX} dense_out[{i}] {name} (TRT dense engine): "
                        f"shape={tuple(t.shape)} dtype={t.dtype} "
                        f"min={float(t_f.min().item()):.4f} max={float(t_f.max().item()):.4f} "
                        f"mean={float(t_f.mean().item()):.4f}{extra}"
                    )
            elif do_split_dbg and tensors:
                for i, name in enumerate(ordered_names):
                    t = tensors[i].detach()
                    t_f = t.float().reshape(-1)
                    extra = ""
                    if name == "bbox_pred" and t.ndim >= 2 and t.shape[0] >= 2:
                        cx = t[0].float().reshape(-1)
                        cy = t[1].float().reshape(-1)
                        extra = f" center_x[min,max]=({float(cx.min())},{float(cx.max())}) center_y[min,max]=({float(cy.min())},{float(cy.max())})"
                    if name == "label_pred":
                        lp = t.reshape(-1).long()
                        uniq = torch.unique(lp)
                        extra = f" label_unique_count={int(uniq.numel())} label_min={int(lp.min())} label_max={int(lp.max())}"
                    if name == "score":
                        extra = f" score>0.1_count={int((t_f > 0.1).sum())} score>0.5_count={int((t_f > 0.5).sum())}"
                    logger.warning(
                        "[BEVFUSION][TensorRT][debug-split] dense_out[%s] %s: shape=%s dtype=%s min=%.6f max=%.6f mean=%.6f%s",
                        i,
                        name,
                        tuple(t.shape),
                        t.dtype,
                        float(t_f.min().item()),
                        float(t_f.max().item()),
                        float(t_f.mean().item()),
                        extra,
                    )
            return tensors

        engine = self._engine
        context = self._context
        assert engine is not None and context is not None

        output_arrays = self._trt_infer_voxel_inputs(
            engine,
            context,
            voxels_np,
            coors_np,
            num_points_np,
            profiler,
            gpu_interval_events=(self._start_event, self._end_event),
        )
        output_names = list(output_arrays.keys())

        component_cfg = self._components_cfg.get_component("bevfusion_main_body")
        expected_output_names = [out.name for out in component_cfg.io.outputs]
        ordered_names = [n for n in expected_output_names if n in output_arrays]
        ordered_names += [n for n in output_names if n not in ordered_names]
        return [torch.from_numpy(output_arrays[name]).to(self.torch_device) for name in ordered_names]

    # Stage keys aligned with BEVFusionPyTorchPipeline for consistent Stage-wise Latency Breakdown.
    # ``dense_engine_ms`` is the dense branch GPU time (split: CUDA events, merged: derived residual).
    BEVFUSION_STAGE_KEYS = (
        "voxel_encoder_ms",
        "sparse_encoder_ms",
        "dense_engine_ms",
        "backbone_ms",
        "neck_ms",
        "head_ms",
        "post_scoring_ms",
        "dense_unattributed_ms",
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

        # ------------------------------------------------------------------
        # Step 1: authoritative top-line GPU intervals (CUDA events).
        #   - bevfusion_ms : total TRT GPU time for the BEVFusion model.
        #   - sparse_encoder_ms / dense_engine_ms : the two top-level branches.
        # Split has two physical engines (separate CUDA-event intervals). Merged
        # is one engine, so we split its single interval by the per-layer profiler
        # proportions (same classifier as the sub-stages below) — keeping every
        # number on one clock and one naming contract.
        # ------------------------------------------------------------------
        stage_sums = _sum_layers_by_stage(profiler.layer_times) if profiler.layer_times else None

        if self._split:
            sparse_ms = self._last_split_sparse_gpu_ms
            dense_ms = self._last_split_dense_gpu_ms
            stage_latencies["bevfusion_ms"] = sparse_ms + dense_ms
        else:
            self._end_event.synchronize()
            bevfusion_ms = float(self._end_event.time_since(self._start_event))
            stage_latencies["bevfusion_ms"] = bevfusion_ms
            if stage_sums is not None:
                total_raw = sum(stage_sums.values())
                sparse_frac = (stage_sums["sparse_encoder_ms"] / total_raw) if total_raw > 0.0 else 0.0
                sparse_ms = bevfusion_ms * sparse_frac
            else:
                sparse_ms = 0.0
            dense_ms = max(bevfusion_ms - sparse_ms, 0.0)

        stage_latencies["sparse_encoder_ms"] = sparse_ms
        stage_latencies["dense_engine_ms"] = dense_ms

        # ------------------------------------------------------------------
        # Step 2: dense sub-stage breakdown — IDENTICAL path for merged & split.
        # Per-layer (order-independent) classification gives the relative weight
        # of backbone/neck/head/post_scoring, rescaled to the dense GPU interval.
        # ------------------------------------------------------------------
        if stage_sums is not None:
            dense_dist = _scale_dense_substages(stage_sums, dense_ms)
            stage_latencies["backbone_ms"] = dense_dist["backbone_ms"]
            stage_latencies["neck_ms"] = dense_dist["neck_ms"]
            stage_latencies["head_ms"] = dense_dist["head_ms"]
            stage_latencies["post_scoring_ms"] = dense_dist["post_scoring_ms"]
            stage_latencies["dense_unattributed_ms"] = dense_dist["dense_unattributed_ms"]
        else:
            stage_latencies["dense_unattributed_ms"] = dense_ms

        # Align "Model" with the same interval semantics across merged/split TensorRT:
        # report model_ms as the BEVFusion TRT GPU segment (not wall-clock Python overhead).
        stage_latencies["model_ms"] = stage_latencies.get("bevfusion_ms", 0.0)

        return outputs, stage_latencies

    def _record_sparse_profile(self, layer_times: List[Tuple[str, float]]) -> None:
        """Priority A in-situ overlay: accumulate sparse-engine bucket breakdown.

        We keep running sums across all eval frames so that after the run the user can
        read off a 'mean sparse encoder bucket' right next to the normal latency table.
        """
        if not layer_times:
            return
        buckets = _summarize_sparse_layers(layer_times)
        self._last_sparse_profile_buckets = buckets
        self._sparse_profile_frame_count += 1
        for b, ms in buckets.items():
            self._sparse_profile_bucket_sum[b] = self._sparse_profile_bucket_sum.get(b, 0.0) + ms
        for name, ms in layer_times:
            self._sparse_profile_top_layers[name] = self._sparse_profile_top_layers.get(name, 0.0) + ms

        if self._sparse_profile_frame_count % _TRT_SPARSE_PROFILE_EVERY == 0:
            total = sum(buckets.values()) or 1e-9
            parts = [
                f"{b}={buckets[b]:.3f}ms ({buckets[b] / total * 100.0:.1f}%)"
                for b in _SPARSE_BUCKET_ORDER
                if buckets.get(b, 0.0) > 0.0
            ]
            logger.info(
                "[priority-a][sparse-profile] frame=%d sparse_layer_sum=%.3fms | %s",
                self._sparse_profile_frame_count,
                total,
                " ".join(parts),
            )

    def print_sparse_profile_summary(self) -> None:
        """Print Priority A mean-per-frame sparse-engine bucket breakdown.

        Called by the evaluator at the end of the run; no-op if the env var was off.
        """
        n = self._sparse_profile_frame_count
        if n <= 0:
            return
        logger.info("=" * 72)
        logger.info("[priority-a] Sparse encoder in-situ bucket breakdown (mean/frame, n=%d)", n)
        logger.info("=" * 72)
        total_mean = sum(self._sparse_profile_bucket_sum.values()) / n
        for b in _SPARSE_BUCKET_ORDER:
            s = self._sparse_profile_bucket_sum.get(b, 0.0)
            if s <= 0.0:
                continue
            mean = s / n
            pct = (s / (total_mean * n)) * 100.0 if total_mean > 0.0 else 0.0
            logger.info("  %-20s %8.3f ms  (%5.2f%%)", b, mean, pct)
        logger.info("  %-20s %8.3f ms", "SUM", total_mean)
        top_items = sorted(self._sparse_profile_top_layers.items(), key=lambda kv: -kv[1])[:10]
        logger.info("Top 10 sparse layers (mean/frame):")
        for name, acc in top_items:
            logger.info("  %8.3f ms  %s", acc / n, name)
        logger.info("=" * 72)

    def _release_gpu_resources(self) -> None:
        # Priority A — emit the sparse-profile summary before we tear engines down.
        try:
            self.print_sparse_profile_summary()
        except Exception as exc:
            logger.warning("[priority-a] sparse-profile summary failed: %s", exc)
        for attr in (
            "_start_event",
            "_end_event",
            "_sparse_ev_s",
            "_sparse_ev_e",
            "_dense_ev_s",
            "_dense_ev_e",
        ):
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
