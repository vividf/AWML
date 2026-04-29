"""Priority A — Sparse encoder TensorRT 專項 profile 工具。

對應 ``docs/15_README_AWML_SPCONV_INT8_ACCEL_PLAN.md``:

  * **A1** — 分離 ``implicit_gemm`` / pair-gen / sort / elementwise / scatter
    的實際時間佔比，而不是只看總 latency。
  * **A2** — 量測 ``GetIndicePairsImplicitGemm`` 這個 bucket（pair 建構等；未必含 argsort）。
    會讀 ``--deploy-cfg`` 的 ``spconv_do_sort``，避免在 ``do_sort=false`` 時仍誤導成「要去關 sort」。

使用流程（配合部署 step 0~5，詳見 ``docs/16_PRIORITY_A_PROFILING_USAGE.md``）::

    # 已經跑過 step 0 ~ step 5，有了 bevfusion_sparse.engine 之後：
    python -m deployment.projects.bevfusion.benchmark.profile_sparse_encoder \\
        --engine work_dirs/bevfusion_split_int8_deployment/tensorrt/bevfusion_sparse.engine \\
        --deploy-cfg deployment/projects/bevfusion/config/deploy_config_split_int8.py \\
        --model-cfg projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m_fx.py \\
        --warmup 20 --iterations 200 \\
        --output work_dirs/bevfusion_split_int8_deployment/sparse_profile.json

工具負責：

  * 載入 ``bevfusion_sparse.engine``（含 ``ImplicitGemmInt8`` / pair-gen /
    scatter 等 plugin）。
  * 產生真實 voxel input（透過 ``BEVFusionDataLoader`` + ``pts_voxel_layer``）
    或在 CI 裡用 ``--synthetic`` 走簡易合成 input 做冒煙測試。
  * 分離 warmup 與 measured 階段，回報 CUDA-event 穩態總時間。
  * 掛上 ``trt.IProfiler``，蒐集 per-layer 時間，再依 layer 名稱分類成
    ``pair_gen`` / ``implicit_gemm_int8`` / ``implicit_gemm_fp`` / ``relu`` /
    ``add`` / ``scatter_nd`` / ``cast`` / ``other`` 等 bucket。
  * 依 encoder block（``conv_input`` / ``encoder_layer.0..3`` / ``conv_out``）
    做第二層 roll-up，對照現有第 8 章的 layer 結構。
  * 輸出 JSON 報告，以及 top-N 層級的文字表格。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
import re
import statistics
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# CRITICAL — CUDA context selection.
#
# If we use ``pycuda.autoinit``, pycuda creates a *non-primary* CUDA context and
# makes it current. PyTorch, on the other hand, uses the device's *primary*
# context (via ``cuDevicePrimaryCtxRetain``). When both coexist, switching
# between them during ``execute_async_v3`` produces::
#
#   CUDA error 400 launching __myl_Mov kernel   (cudaErrorInvalidResourceHandle)
#
# because the engine (deserialized on pycuda's context) then tries to launch
# kernels while PyTorch's primary context is active.
#
# ``pycuda.autoprimaryctx`` retains and pushes the *primary* context, so pycuda
# and PyTorch share the same context and TRT sees consistent resources. Fall
# back to ``autoinit`` only if the helper module isn't available (very old
# pycuda); in that case we also document the order-of-operations workaround.
try:
    import pycuda.autoprimaryctx  # noqa: F401
except ImportError:  # pragma: no cover — only old pycuda hits this
    import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

logger = logging.getLogger("priority_a.profile_sparse_encoder")


# =============================================================================
# Layer classification
# =============================================================================
#
# Sparse encoder 內常見的 plugin / node 來源（見 ``projects/SparseConvolution/
# sparse_functional.py``）：
#
# * ``autoware::GetIndicePairsImplicitGemm``   → pair_gen（含 argsort 輸出）
# * ``autoware::ImplicitGemmInt8``             → implicit_gemm（INT8 分支）
# * ``autoware::ImplicitGemm``                 → implicit_gemm（FP16 / conv_out）
# * ``ScatterND``                              → sparse → dense 投影
# * ``Add`` / ``Relu``                          → sparse residual 鏈
# * ``Cast`` / ``QuantizeLinear`` / ``DequantizeLinear`` → dtype / Q-DQ 邊界
#
# 分類直接讀 TRT layer 名；TRT 會把 ONNX 節點名（加上 ``PWN`` 前綴等）餵進
# ``IProfiler::reportLayerTime``，所以字串 match 是穩定可觀察的。
_BUCKET_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    # pair-gen / sort — 這是 A2 要放大檢視的 bucket
    (re.compile(r"GetIndicePairsImplicitGemm", re.IGNORECASE), "pair_gen"),
    (re.compile(r"GetIndicePairs(?!ImplicitGemm)", re.IGNORECASE), "pair_gen"),
    # implicit gemm（INT8 / FP16）
    # NOTE: TRT layer names use mixed casing/underscores e.g. ``ImplicitGemm_int8``
    # so the INT8 rule needs to tolerate a separator between ``ImplicitGemm`` and
    # ``int8`` and the FP rule must explicitly anchor on a non-int8 ending.
    (re.compile(r"ImplicitGemm[_\- ]?Int8", re.IGNORECASE), "implicit_gemm_int8"),
    (re.compile(r"ImplicitGemm(?![_\- ]?Int8)", re.IGNORECASE), "implicit_gemm_fp"),
    (re.compile(r"IndiceConv", re.IGNORECASE), "implicit_gemm_fp"),
    # dense / scatter
    (re.compile(r"ScatterND", re.IGNORECASE), "scatter_nd"),
    # elementwise / dtype
    (re.compile(r"(?:^|/|_)Relu", re.IGNORECASE), "relu"),
    (re.compile(r"(?:^|/|_)Add(?:_|$|/|\[)", re.IGNORECASE), "add"),
    (re.compile(r"QuantizeLinear|DequantizeLinear", re.IGNORECASE), "quant_dquant"),
    (re.compile(r"(?:^|/|_)Cast", re.IGNORECASE), "cast"),
    (re.compile(r"Reshape|Transpose|Concat|Slice|Gather|Squeeze|Unsqueeze", re.IGNORECASE), "layout"),
)

_BUCKET_ORDER: Tuple[str, ...] = (
    "pair_gen",
    "implicit_gemm_int8",
    "implicit_gemm_fp",
    "scatter_nd",
    "add",
    "relu",
    "quant_dquant",
    "cast",
    "layout",
    "other",
)


def classify_bucket(layer_name: str) -> str:
    for pattern, bucket in _BUCKET_PATTERNS:
        if pattern.search(layer_name):
            return bucket
    return "other"


# ``pts_middle_encoder`` 的典型 block（see sparse encoder module naming）。
# Match 在 layer name 裡出現的 sub-module path，做第二層 roll-up。
_ENCODER_BLOCKS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"pts_middle_encoder/conv_input|/conv_input/|conv_input\."), "conv_input"),
    (re.compile(r"encoder_layers\.0|encoder_layer\.0|encoder_layers/0"), "encoder_layer.0"),
    (re.compile(r"encoder_layers\.1|encoder_layer\.1|encoder_layers/1"), "encoder_layer.1"),
    (re.compile(r"encoder_layers\.2|encoder_layer\.2|encoder_layers/2"), "encoder_layer.2"),
    (re.compile(r"encoder_layers\.3|encoder_layer\.3|encoder_layers/3"), "encoder_layer.3"),
    (re.compile(r"conv_out"), "conv_out"),
)


def classify_block(layer_name: str) -> str:
    for pattern, block in _ENCODER_BLOCKS:
        if pattern.search(layer_name):
            return block
    return "other"


# =============================================================================
# TRT profiler
# =============================================================================
class _LayerProfiler(trt.IProfiler):
    """收每個 TensorRT layer 在 enqueue 期間的 wall-clock 時間（ms）。"""

    def __init__(self) -> None:
        try:
            trt.IProfiler.__init__(self)
        except Exception:
            pass
        self._records: Dict[str, List[float]] = {}

    def report_layer_time(self, layer_name: str, ms: float) -> None:
        self._records.setdefault(str(layer_name), []).append(float(ms))

    def reset(self) -> None:
        self._records.clear()

    @property
    def layer_records(self) -> Dict[str, List[float]]:
        return self._records


# =============================================================================
# Input generation
# =============================================================================
@dataclass
class SparseInputs:
    voxels: np.ndarray  # [N, P, C] or [N, C]
    coors: np.ndarray  # [N, 4]
    num_points: np.ndarray  # [N]
    source: str = "synthetic"

    @property
    def num_voxels(self) -> int:
        return int(self.voxels.shape[0])


def _load_real_sparse_inputs(
    *,
    deploy_cfg_path: str,
    model_cfg_path: str,
    checkpoint_path: Optional[str],
    info_file_override: Optional[str],
    sample_idx: int,
) -> SparseInputs:
    """以 BEVFusion voxelizer 產生一個真實 voxel input（和 eval 路徑一致）。

    為避免 profile 工具順便跑了 PTQ / spconv_int8 的重建流程，這裡不走
    ``BEVFusionDeploymentRunner``，而是直接 build 資料集 + voxel layer。
    """
    from mmengine.config import Config  # noqa: WPS433

    from deployment.core.device import DeviceSpec  # noqa: WPS433
    from deployment.projects.bevfusion.io.data_loader import BEVFusionDataLoader  # noqa: WPS433
    from deployment.projects.bevfusion.io.model_loader import build_bevfusion_model  # noqa: WPS433

    model_cfg = Config.fromfile(model_cfg_path)
    deploy_cfg = Config.fromfile(deploy_cfg_path)

    info_file = info_file_override or deploy_cfg.runtime_io["info_file"]
    data_loader = BEVFusionDataLoader(info_file=info_file, model_cfg=model_cfg)
    logger.info("[inputs] dataset=%s samples=%d, using sample_idx=%d", info_file, data_loader.num_samples, sample_idx)

    # FP32 模型足夠，這條路只是 voxelize；checkpoint_path 可從 deploy_cfg 解析。
    if checkpoint_path is None:
        checkpoint_path = getattr(deploy_cfg, "checkpoint_path", None)
    if checkpoint_path is None:
        raise ValueError(
            "checkpoint_path is required (either via --checkpoint or deploy_cfg.checkpoint_path) "
            "to instantiate pts_voxel_layer for real sparse inputs."
        )

    import torch  # noqa: WPS433

    # ``build_bevfusion_model`` expects a ``DeviceSpec`` (it calls ``.to_torch_device()``
    # internally). Pass DeviceSpec, not a raw ``torch.device``.
    cuda_available = torch.cuda.is_available()
    device_spec = DeviceSpec(kind="cuda", index=0) if cuda_available else DeviceSpec(kind="cpu", index=0)
    torch_device = device_spec.to_torch_device()

    # ``quantization=None`` keeps the heavy PTQ / FX rebuild off this path; we only need
    # ``pts_voxel_layer`` to produce realistic voxel/coors/npt inputs for the TRT engine.
    #
    # Note on state_dict warnings: PTQ checkpoints carry fused-BN weights plus QDQ ``_amax``
    # keys, while the skeleton we build here is FP32 with BN. ``load_state_dict`` will warn
    # about missing BN / unexpected ``_amax`` keys — **safe to ignore** in this tool, since
    # only ``pts_voxel_layer`` (parameterless) is exercised.
    logger.info(
        "[inputs] Building FP32 skeleton just for voxelization — state_dict BN/_amax mismatches from "
        "a PTQ checkpoint are expected and will be ignored (we only invoke pts_voxel_layer)."
    )
    model = build_bevfusion_model(
        model_cfg=model_cfg,
        checkpoint_path=checkpoint_path,
        device=device_spec,
        quantization=None,
    )
    model.eval()

    sample = data_loader.load_sample(sample_idx)
    points = sample["points"]
    if not hasattr(points, "to"):  # numpy fallback
        points = torch.from_numpy(np.asarray(points))
    points = points.to(torch_device).float()

    with torch.no_grad():
        ret = model.pts_voxel_layer(points)
        if len(ret) == 3:
            feats, coords, sizes = ret
        else:
            feats, coords = ret
            sizes = None

        # TRT sparse engine (see deploy_config_split_int8.components.bevfusion_sparse.io.inputs)
        # binds ``coors`` as a 2-D ``[N, 3]`` tensor — batch column is *not* part of the engine
        # contract (batch=1 is handled inside the plugin). The PyTorch sparse encoder wants
        # ``[N, 4]`` because spconv prepends a batch column, but that's an internal detail of the
        # PyTorch path and must not be replicated when feeding the exported engine.
        # So: keep coords exactly as ``pts_voxel_layer`` returns (``[N, 3]``).

        voxels_np = feats.detach().cpu().numpy().astype(np.float32)
        coors_np = coords.detach().cpu().numpy().astype(np.int32)
        if sizes is None:
            npt_np = np.ones(voxels_np.shape[0], dtype=np.int32)
        else:
            npt_np = np.maximum(sizes.detach().cpu().numpy().astype(np.int32), 1)

    logger.info(
        "[inputs] real voxel input: voxels=%s coors=%s npt=%s (dataset=%s idx=%d)",
        voxels_np.shape,
        coors_np.shape,
        npt_np.shape,
        info_file,
        sample_idx,
    )
    return SparseInputs(voxels=voxels_np, coors=coors_np, num_points=npt_np, source=f"real:{info_file}#{sample_idx}")


def _synthetic_sparse_inputs(num_voxels: int, voxel_points: int, voxel_channels: int) -> SparseInputs:
    """冒煙測試用：產生隨機 voxel input，shape 與 real 路徑一致。

    * 注意合成 input 不會反映真實資料 sparsity，所以 pair_gen / scatter
      相對成本可能被低估；真正判斷請用 real input。
    * ``coors`` shape 與 TRT engine binding 一致：``[N, 3]``（z, y, x；不含 batch）。
    """
    rng = np.random.default_rng(0)
    voxels = rng.standard_normal((num_voxels, voxel_points, voxel_channels), dtype=np.float32) * 0.1
    coors = np.zeros((num_voxels, 3), dtype=np.int32)
    coors[:, 0] = rng.integers(0, 16, size=num_voxels, dtype=np.int32)
    coors[:, 1] = rng.integers(0, 1440, size=num_voxels, dtype=np.int32)
    coors[:, 2] = rng.integers(0, 1440, size=num_voxels, dtype=np.int32)
    npt = rng.integers(1, voxel_points + 1, size=num_voxels, dtype=np.int32)
    logger.warning(
        "[inputs] using SYNTHETIC inputs (N=%d, P=%d, C=%d). "
        "Real-data profiling is strongly recommended for Priority A conclusions.",
        num_voxels,
        voxel_points,
        voxel_channels,
    )
    return SparseInputs(voxels=voxels, coors=coors, num_points=npt, source="synthetic")


# =============================================================================
# TensorRT execution helpers
# =============================================================================
def _trt_dtype_to_numpy(trt_dtype: trt.DataType) -> np.dtype:
    try:
        return np.dtype(trt.nptype(trt_dtype))
    except Exception:
        mapping = {
            trt.DataType.FLOAT: np.float32,
            trt.DataType.HALF: np.float16,
            trt.DataType.INT32: np.int32,
            trt.DataType.INT8: np.int8,
        }
        return np.dtype(mapping.get(trt_dtype, np.float32))


def _load_plugins(plugin_libraries: Sequence[str]) -> None:
    for path in plugin_libraries:
        if not path:
            continue
        if not osp.exists(path):
            logger.warning("[plugin] library not found, skipping: %s", path)
            continue
        try:
            import ctypes  # noqa: WPS433

            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            logger.info("[plugin] loaded: %s", path)
        except OSError as exc:
            logger.warning("[plugin] failed to dlopen %s: %s", path, exc)


def _bind_sparse_inputs(
    engine: trt.ICudaEngine,
    inputs: SparseInputs,
) -> Dict[str, np.ndarray]:
    """以 tensorrt.py 相同規則，把 SparseInputs mapping 到 engine binding。"""
    mapping: Dict[str, np.ndarray] = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            continue
        ln = name.lower()
        if "voxel" in ln and "num" not in ln:
            arr = inputs.voxels
        elif "coor" in ln:
            arr = inputs.coors
        elif "num" in ln:
            arr = inputs.num_points
        else:
            logger.warning("[trt-io] unknown input %r; skipping", name)
            continue
        want = _trt_dtype_to_numpy(engine.get_tensor_dtype(name))
        if arr.dtype != want:
            arr = np.asarray(arr, dtype=want)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        mapping[name] = arr
    return mapping


@dataclass
class TimingRecord:
    total_ms: List[float] = field(default_factory=list)  # CUDA-event total per iter
    per_layer_ms: Dict[str, List[float]] = field(default_factory=dict)


def _run_benchmark(
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
    inputs: SparseInputs,
    warmup: int,
    iterations: int,
    profile_every_iter: bool,
) -> TimingRecord:
    input_map = _bind_sparse_inputs(engine, inputs)
    output_names: List[str] = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            output_names.append(name)

    for name, arr in input_map.items():
        context.set_input_shape(name, arr.shape)

    output_arrays: Dict[str, np.ndarray] = {}
    for name in output_names:
        shape = context.get_tensor_shape(name)
        np_dtype = _trt_dtype_to_numpy(engine.get_tensor_dtype(name))
        output_arrays[name] = np.empty(shape, dtype=np_dtype)

    d_inputs: Dict[str, int] = {}
    d_outputs: Dict[str, int] = {}
    for name, arr in input_map.items():
        d_inputs[name] = cuda.mem_alloc(arr.nbytes)
    for name, arr in output_arrays.items():
        d_outputs[name] = cuda.mem_alloc(arr.nbytes)

    stream = cuda.Stream()
    start_ev = cuda.Event()
    end_ev = cuda.Event()

    for name, arr in input_map.items():
        cuda.memcpy_htod_async(d_inputs[name], arr, stream)
        context.set_tensor_address(name, int(d_inputs[name]))
    for name in output_names:
        context.set_tensor_address(name, int(d_outputs[name]))
    stream.synchronize()

    profiler = _LayerProfiler()

    # ---- warmup (no profiler attached — 避免第一次 shape-fit 干擾 bucket 統計) ----
    logger.info("[bench] warmup: %d iterations", warmup)
    for _ in range(warmup):
        ok = context.execute_async_v3(stream_handle=stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed during warmup.")
    stream.synchronize()

    # ---- measured ----
    logger.info("[bench] measured: %d iterations (per-layer profiler=%s)", iterations, profile_every_iter)
    record = TimingRecord()

    if profile_every_iter:
        context.profiler = profiler

    for _ in range(iterations):
        start_ev.record(stream)
        ok = context.execute_async_v3(stream_handle=stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed during measured iter.")
        end_ev.record(stream)
        end_ev.synchronize()
        record.total_ms.append(float(end_ev.time_since(start_ev)))

    # Merge per-layer records（sum across iterations, 等下做除法）。
    for name, values in profiler.layer_records.items():
        record.per_layer_ms.setdefault(name, []).extend(values)

    # Free GPU memory — 非必要但避免留底
    for ptr in list(d_inputs.values()) + list(d_outputs.values()):
        try:
            ptr.free()
        except Exception:
            pass

    return record


# =============================================================================
# Reporting
# =============================================================================
def _resolve_spconv_do_sort(deploy_cfg_path: Optional[str]) -> Optional[bool]:
    """Read ``spconv_do_sort`` from deploy config (same default as ``entrypoint._apply_spconv_do_sort``).

    ``None`` means no config was provided or parsing failed — the report cannot
    infer whether the **exported** engine skips argsort.
    """
    if not deploy_cfg_path:
        return None
    try:
        from mmengine.config import Config  # noqa: WPS433

        deploy_cfg = Config.fromfile(deploy_cfg_path)
        return bool(deploy_cfg.get("spconv_do_sort", True))
    except Exception as exc:
        logger.warning("[report] could not read spconv_do_sort from %s: %s", deploy_cfg_path, exc)
        return None


def _stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0}
    v = list(values)
    return {
        "count": len(v),
        "mean_ms": float(statistics.fmean(v)),
        "std_ms": float(statistics.pstdev(v)) if len(v) > 1 else 0.0,
        "min_ms": float(min(v)),
        "max_ms": float(max(v)),
        "median_ms": float(statistics.median(v)),
    }


def _summarize_layers(
    record: TimingRecord,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], List[Tuple[str, float]]]:
    """把 per-layer 時間 aggregate 成 bucket + block 兩層 roll-up。

    回傳:
      * bucket_stats: {bucket: {sum_ms, pct, count, top_mean_ms, ...}}
      * block_stats:  {block: {sum_ms, pct, count}}
      * layer_means:  [(layer_name, mean_ms)]，方便印 top-N
    """
    layer_means: List[Tuple[str, float]] = []
    bucket_totals: Dict[str, float] = {b: 0.0 for b in _BUCKET_ORDER}
    bucket_counts: Dict[str, int] = {b: 0 for b in _BUCKET_ORDER}
    block_totals: Dict[str, float] = {}
    block_counts: Dict[str, int] = {}

    for name, values in record.per_layer_ms.items():
        if not values:
            continue
        mean_ms = float(statistics.fmean(values))
        layer_means.append((name, mean_ms))
        bucket = classify_bucket(name)
        bucket_totals[bucket] = bucket_totals.get(bucket, 0.0) + mean_ms
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        block = classify_block(name)
        block_totals[block] = block_totals.get(block, 0.0) + mean_ms
        block_counts[block] = block_counts.get(block, 0) + 1

    total_per_iter_mean = sum(mean for _, mean in layer_means)

    bucket_stats: Dict[str, Dict[str, float]] = {}
    for bucket in _BUCKET_ORDER:
        s = bucket_totals.get(bucket, 0.0)
        bucket_stats[bucket] = {
            "sum_ms": s,
            "count": int(bucket_counts.get(bucket, 0)),
            "pct_of_layer_sum": (s / total_per_iter_mean * 100.0) if total_per_iter_mean > 0 else 0.0,
        }

    block_stats: Dict[str, Dict[str, float]] = {}
    for block, s in block_totals.items():
        block_stats[block] = {
            "sum_ms": s,
            "count": int(block_counts.get(block, 0)),
            "pct_of_layer_sum": (s / total_per_iter_mean * 100.0) if total_per_iter_mean > 0 else 0.0,
        }

    layer_means.sort(key=lambda kv: kv[1], reverse=True)
    return bucket_stats, block_stats, layer_means


def _format_report(
    *,
    engine_path: str,
    inputs: SparseInputs,
    record: TimingRecord,
    bucket_stats: Dict[str, Dict[str, float]],
    block_stats: Dict[str, Dict[str, float]],
    layer_means: List[Tuple[str, float]],
    top_n: int,
    spconv_do_sort: Optional[bool],
) -> str:
    lines: List[str] = []
    total_stats = _stats(record.total_ms)
    lines.append("=" * 78)
    lines.append("Priority A — Sparse Encoder Profile Report")
    lines.append("=" * 78)
    lines.append(f"Engine  : {engine_path}")
    lines.append(f"Inputs  : {inputs.source} (num_voxels={inputs.num_voxels})")
    lines.append(f"Total GPU latency (CUDA-event, steady-state):")
    if total_stats.get("count", 0):
        lines.append(
            "  mean={mean_ms:.3f} ± {std_ms:.3f} ms  median={median_ms:.3f}  "
            "min={min_ms:.3f}  max={max_ms:.3f}  n={count}".format(**total_stats)
        )
    else:
        lines.append("  (no measurements)")

    lines.append("")
    lines.append("-" * 78)
    lines.append("Layer-sum breakdown by op-bucket (mean per-iteration sum):")
    lines.append("-" * 78)
    lines.append(f"  {'bucket':<20s}  {'count':>6s}  {'sum_ms':>10s}  {'% of layers':>12s}")
    for bucket in _BUCKET_ORDER:
        s = bucket_stats.get(bucket, {})
        if s.get("count", 0) == 0:
            continue
        lines.append(f"  {bucket:<20s}  {int(s['count']):>6d}  {s['sum_ms']:>10.3f}  {s['pct_of_layer_sum']:>11.2f}%")

    # A2 highlight
    pair_pct = bucket_stats.get("pair_gen", {}).get("pct_of_layer_sum", 0.0)
    pair_sum = bucket_stats.get("pair_gen", {}).get("sum_ms", 0.0)
    lines.append("")
    lines.append(
        "[A2] pair_gen (GetIndicePairsImplicitGemm) = " f"{pair_sum:.3f} ms/iter ({pair_pct:.2f}% of layer-sum)"
    )
    # Clarify: this bucket is the whole pair plugin, not "sort time" alone.
    if spconv_do_sort is None:
        lines.append(
            "     Deploy spconv_do_sort: **unknown** (add `--deploy-cfg` to show export intent). "
            "pair_gen ≠ argsort only — it includes hash/scan/gather even when sort is off."
        )
    elif spconv_do_sort:
        lines.append(
            "     Deploy spconv_do_sort: **True** — ONNX export bakes pair-mask argsort ON (FP16-style). "
            "High pair_gen may include DeviceMergeSort-style work."
        )
    else:
        lines.append(
            "     Deploy spconv_do_sort: **False** — export targets `do_sort_i=0` (no pair-mask argsort). "
            "Remaining pair_gen time is index/pair construction, not a missing sort opt."
        )

    if pair_pct >= 10.0:
        if spconv_do_sort is False:
            lines.append(
                "     → Pair-gen share is large but **sort is already off by design**; use Nsight on "
                "hash/scan/prefix kernels, not as a cue to «turn off sort again»."
            )
        elif spconv_do_sort is True:
            lines.append(
                "     → pair-gen / sort 佔比明顯；`do_sort=false` 於 INT8 export 可能仍有收益，"
                "值得用 Nsight Compute 對照 `DeviceMergeSort*` / scan kernel。"
            )
        else:
            lines.append(
                "     → pair-gen 佔比高；請對照 nsys 是否有 `DeviceMergeSort*` 以判斷 argsort 是否仍在執行，"
                "並確認 engine 是否由含 `spconv_do_sort=False` 的 deploy 匯出。"
            )
    elif pair_pct >= 3.0:
        if spconv_do_sort is False:
            lines.append("     → pair-gen 中等（多為非-sort 的配對開銷）；再壓榨優先級通常低於 implicit_gemm / 邊界。")
        else:
            lines.append("     → pair-gen 中等；若要再壓榨，優先順序低於 implicit_gemm / 邊界優化。")
    else:
        if spconv_do_sort is False:
            lines.append("     → pair-gen 佔比低（於 do_sort=false 仍屬正常）。")
        else:
            lines.append("     → pair-gen 佔比很低；pair-mask sort 改造空間通常有限（與 8.1 類似結論）。")

    lines.append("")
    lines.append("-" * 78)
    lines.append("Block roll-up (encoder_layer granularity):")
    lines.append("-" * 78)
    lines.append(f"  {'block':<20s}  {'count':>6s}  {'sum_ms':>10s}  {'% of layers':>12s}")
    for block, s in sorted(block_stats.items(), key=lambda kv: -kv[1]["sum_ms"]):
        if s.get("count", 0) == 0:
            continue
        lines.append(f"  {block:<20s}  {int(s['count']):>6d}  {s['sum_ms']:>10.3f}  {s['pct_of_layer_sum']:>11.2f}%")

    lines.append("")
    lines.append("-" * 78)
    lines.append(f"Top {top_n} layers by mean time:")
    lines.append("-" * 78)
    lines.append(f"  {'#':>3s}  {'mean_ms':>10s}  {'bucket':<20s}  layer_name")
    for idx, (name, mean_ms) in enumerate(layer_means[:top_n], start=1):
        lines.append(f"  {idx:>3d}  {mean_ms:>10.3f}  {classify_bucket(name):<20s}  {name}")
    lines.append("=" * 78)

    total_layer_sum = sum(mean for _, mean in layer_means)
    total_event_mean = total_stats.get("mean_ms", 0.0) if total_stats.get("count", 0) else 0.0
    if total_event_mean > 0.0 and total_layer_sum > 0.0:
        overhead = total_event_mean - total_layer_sum
        lines.append(
            f"sanity: per-layer sum={total_layer_sum:.3f} ms, event total={total_event_mean:.3f} ms, "
            f"delta={overhead:+.3f} ms (plugin launch / memcpy / TRT overhead)"
        )
    return "\n".join(lines)


def _dump_json(
    path: str,
    *,
    engine_path: str,
    inputs: SparseInputs,
    record: TimingRecord,
    bucket_stats: Dict[str, Dict[str, float]],
    block_stats: Dict[str, Dict[str, float]],
    layer_means: List[Tuple[str, float]],
    spconv_do_sort: Optional[bool],
) -> None:
    payload = {
        "engine": engine_path,
        "spconv_do_sort_from_deploy_cfg": spconv_do_sort,
        "inputs": {
            "source": inputs.source,
            "num_voxels": inputs.num_voxels,
            "voxels_shape": list(inputs.voxels.shape),
            "coors_shape": list(inputs.coors.shape),
        },
        "total_gpu_ms": _stats(record.total_ms),
        "buckets": bucket_stats,
        "blocks": block_stats,
        "top_layers": [
            {"name": n, "mean_ms": m, "bucket": classify_bucket(n), "block": classify_block(n)} for n, m in layer_means
        ],
    }
    os.makedirs(osp.dirname(osp.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fp:
        json.dump(payload, fp, indent=2)
    logger.info("[report] JSON saved to %s", path)


# =============================================================================
# Main
# =============================================================================
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Priority A — BEVFusion sparse encoder TensorRT profile (A1 + A2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--engine", required=True, help="Path to bevfusion_sparse.engine")
    parser.add_argument(
        "--plugin-lib",
        action="append",
        default=[],
        help="TensorRT plugin .so to preload (may repeat). " "Defaults to plugins from --deploy-cfg if set.",
    )
    parser.add_argument("--deploy-cfg", help="deploy_config_*.py (to infer plugins + info_file)")
    parser.add_argument("--model-cfg", help="model config .py (required unless --synthetic)")
    parser.add_argument(
        "--checkpoint", help="PTQ checkpoint for voxelization only (default: deploy_cfg.checkpoint_path)"
    )
    parser.add_argument("--info-file", help="Override deploy_cfg.runtime_io.info_file")
    parser.add_argument("--sample-idx", type=int, default=0, help="Dataset sample index to profile")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Skip dataset load; use random voxels. Coarse estimate only; do NOT use for conclusions.",
    )
    parser.add_argument("--num-voxels", type=int, default=40000, help="Synthetic voxel count")
    parser.add_argument("--voxel-points", type=int, default=10, help="Synthetic per-voxel point count")
    parser.add_argument("--voxel-channels", type=int, default=5, help="Synthetic voxel feature channels")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--no-per-layer",
        action="store_true",
        help="Disable IProfiler (total GPU time only; useful for noise-free steady-state timing)",
    )
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument(
        "--output", default=None, help="JSON output path (default: beside engine as sparse_profile.json)"
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def _resolve_plugins(args: argparse.Namespace) -> List[str]:
    plugins: List[str] = list(args.plugin_lib)
    if args.deploy_cfg and not plugins:
        try:
            from mmengine.config import Config  # noqa: WPS433

            deploy_cfg = Config.fromfile(args.deploy_cfg)
            plugins = list(getattr(deploy_cfg, "tensorrt_config", {}).get("plugin_libraries", []))
        except Exception as exc:
            logger.warning("[plugin] failed to read plugins from %s: %s", args.deploy_cfg, exc)
    return plugins


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    engine_path = osp.abspath(args.engine)
    if not osp.exists(engine_path):
        logger.error("engine not found: %s", engine_path)
        return 2

    # ---- Prepare inputs FIRST ----
    # PyTorch lazily initializes its CUDA primary context the first time a
    # tensor is moved onto the device. If we deserialize the TensorRT engine
    # *before* that happens, the engine gets bound to whatever context is
    # current at deserialization (pycuda's), which can differ from the context
    # used later for ``execute_async_v3`` (PyTorch's primary). To avoid this
    # cross-context hazard we run the PyTorch voxelization first so the primary
    # context is already retained & current by the time TRT resources are built.
    if args.synthetic:
        inputs = _synthetic_sparse_inputs(args.num_voxels, args.voxel_points, args.voxel_channels)
    else:
        if not args.model_cfg or not args.deploy_cfg:
            logger.error("--model-cfg and --deploy-cfg are required unless --synthetic is set.")
            return 5
        inputs = _load_real_sparse_inputs(
            deploy_cfg_path=args.deploy_cfg,
            model_cfg_path=args.model_cfg,
            checkpoint_path=args.checkpoint,
            info_file_override=args.info_file,
            sample_idx=args.sample_idx,
        )

    # ---- Now build TensorRT resources on the (single) current context ----
    plugins = _resolve_plugins(args)
    _load_plugins(plugins)
    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(trt_logger, "")
    runtime = trt.Runtime(trt_logger)
    with open(engine_path, "rb") as fp:
        engine = runtime.deserialize_cuda_engine(fp.read())
    if engine is None:
        logger.error("failed to deserialize engine: %s", engine_path)
        return 3
    context = engine.create_execution_context()
    if context is None:
        logger.error("failed to create execution context")
        return 4

    record = _run_benchmark(
        engine=engine,
        context=context,
        inputs=inputs,
        warmup=args.warmup,
        iterations=args.iterations,
        profile_every_iter=not args.no_per_layer,
    )

    bucket_stats, block_stats, layer_means = _summarize_layers(record)
    spconv_do_sort = _resolve_spconv_do_sort(args.deploy_cfg)
    print(
        _format_report(
            engine_path=engine_path,
            inputs=inputs,
            record=record,
            bucket_stats=bucket_stats,
            block_stats=block_stats,
            layer_means=layer_means,
            top_n=args.top_n,
            spconv_do_sort=spconv_do_sort,
        )
    )

    out_path = args.output
    if out_path is None:
        out_path = osp.join(osp.dirname(engine_path) or ".", "sparse_profile.json")
    _dump_json(
        out_path,
        engine_path=engine_path,
        inputs=inputs,
        record=record,
        bucket_stats=bucket_stats,
        block_stats=block_stats,
        layer_means=layer_means,
        spconv_do_sort=spconv_do_sort,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
