# Copyright (c) OpenMMLab. All rights reserved.
"""Per-layer TensorRT profiling + BEVFusion stage attribution.

Pure, order-independent helpers used by :class:`~...tensorrt_inference_pipeline.BEVFusionTensorRTPipeline`
to turn a TRT layer-time list into (a) Priority-A sparse-encoder buckets and (b) BEVFusion stage sums
(sparse / backbone / neck / head / post-scoring). Kept out of the inference pipeline so the runtime
path is not interleaved with ~200 lines of profiling/attribution logic.
"""

import re
from typing import Dict, List, Tuple

import tensorrt as trt


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


# Priority A — bucket classification for sparse encoder in-situ profile.
_SPARSE_BUCKET_ORDER: Tuple[str, ...] = (
    "pair_gen",
    "implicit_gemm_fp",
    "scatter_nd",
    "add",
    "relu",
    "cast",
    "layout",
    "other",
)


def _classify_sparse_bucket(layer_name: str) -> str:
    """Match sparse encoder TRT layer names to a Priority A bucket.

    Keep patterns simple & case-insensitive — TRT forwards ONNX node names with
    occasional prefixes, so pure substring matching is enough and cheap.
    """
    n = layer_name.lower()
    # Normalize common separators so ``ImplicitGemm`` / ``implicit-gemm`` / ``implicit gemm``
    # all collapse to ``implicitgemm`` before substring matching.
    n_norm = n.replace("_", "").replace("-", "").replace(" ", "")
    if "getindicepairsimplicitgemm" in n_norm or ("getindicepairs" in n_norm and "implicitgemm" not in n_norm):
        return "pair_gen"
    if "implicitgemm" in n_norm or "indiceconv" in n_norm:
        return "implicit_gemm_fp"
    if "scatternd" in n:
        return "scatter_nd"
    # Guard: "add"/"relu"/"cast" must be word-like to avoid matching paths.
    if "relu" in n:
        return "relu"
    if "/add" in n or n.endswith("_add") or n.startswith("add"):
        return "add"
    if "/cast" in n or "_cast_" in n or n.startswith("cast"):
        return "cast"
    if any(k in n for k in ("reshape", "transpose", "concat", "slice", "gather", "squeeze", "unsqueeze")):
        return "layout"
    return "other"


def _summarize_sparse_layers(layer_times: List[Tuple[str, float]]) -> Dict[str, float]:
    """Sum sparse TRT layer times per Priority A bucket (ms per frame)."""
    sums: Dict[str, float] = {b: 0.0 for b in _SPARSE_BUCKET_ORDER}
    for layer_name, ms in layer_times:
        sums[_classify_sparse_bucket(layer_name)] += ms
    return sums


# ============================================================================
# Unified BEVFusion stage attribution (merged & split use the SAME logic).
# ----------------------------------------------------------------------------
# Grounded in the BEVFusion ONNX module hierarchy, which is IDENTICAL for the
# merged full graph and the split dense graph (the merged graph only adds
# ``sparse/`` and ``dense/`` prefixes):
#   pts_middle_encoder / spconv / ImplicitGemm ... -> sparse encoder
#   pts_backbone (``blocks``)                      -> backbone
#   pts_neck     (``deblocks``)                    -> neck
#   bbox_head    (decoder / prediction_heads / heatmap_head) -> head
#   score ops    (sigmoid / one_hot / query_*)     -> post scoring
#
# Classification is per-layer and ORDER-INDEPENDENT. This is the critical
# property: TensorRT freely fuses/reorders layers, so the previous order-based
# state machine mis-attributed cost (e.g. one early ``bbox_head`` layer flipped
# the whole stream to "head" and starved backbone/neck). A pure substring
# bucket per layer is stable regardless of fusion/order.
# ============================================================================

_BEVFUSION_DENSE_SUBSTAGE_KEYS: Tuple[str, ...] = (
    "backbone_ms",
    "neck_ms",
    "head_ms",
    "post_scoring_ms",
)
_STAGE_OTHER = "other_ms"


def _classify_bevfusion_layer(layer_name: str) -> str:
    """Classify one TensorRT layer into a BEVFusion stage key (order-independent).

    Returns one of: ``sparse_encoder_ms``, ``backbone_ms``, ``neck_ms``,
    ``head_ms``, ``post_scoring_ms``, or ``other_ms`` (shape/glue, ~0 GPU time).
    """
    n = layer_name.lower()
    # Normalize separators so ``ImplicitGemm``/``implicit_gemm``/``getindicepairs`` all match.
    nn = n.replace("_", "").replace("-", "").replace(" ", "")

    if any(
        k in n
        for k in (
            "pts_middle_encoder",
            "middle_encoder",
            "spconv",
            "sparse_conv",
            "subm",
            "encoder_layer",
            "conv_input",
            "conv_out",
        )
    ) or any(k in nn for k in ("implicitgemm", "getindicepairs", "scatternd")):
        return "sparse_encoder_ms"

    # Neck before backbone: ``deblocks`` contains the substring ``blocks``.
    if "pts_neck" in n or "deblocks" in n:
        return "neck_ms"

    if "pts_backbone" in n or re.search(r"(^|[/.])blocks([/.]|$)", n):
        return "backbone_ms"

    # ``bbox_head`` covers the transformer decoder, prediction_heads and heatmap_head.
    if "bbox_head" in n:
        return "head_ms"

    # Post-scoring ops typically live OUTSIDE bbox_head (top-level sigmoid/one_hot/query_*).
    if any(k in nn for k in ("queryheatmapscore", "querylabels", "onehot")) or any(
        k in n for k in ("sigmoid", "/topk", "argmax")
    ):
        return "post_scoring_ms"

    return _STAGE_OTHER


def _sum_layers_by_stage(layer_times: List[Tuple[str, float]]) -> Dict[str, float]:
    """Sum profiler layer times into BEVFusion stage buckets (order-independent)."""
    sums: Dict[str, float] = {
        "sparse_encoder_ms": 0.0,
        "backbone_ms": 0.0,
        "neck_ms": 0.0,
        "head_ms": 0.0,
        "post_scoring_ms": 0.0,
        _STAGE_OTHER: 0.0,
    }
    for layer_name, ms in layer_times:
        sums[_classify_bevfusion_layer(layer_name)] += ms
    return sums


def _scale_dense_substages(stage_sums: Dict[str, float], dense_total_ms: float) -> Dict[str, float]:
    """Distribute the (CUDA-timed) dense total across backbone/neck/head/post_scoring.

    The per-layer profiler sums give the RELATIVE weight of each dense stage; we
    rescale them so they add up exactly to ``dense_total_ms`` (the authoritative
    GPU interval). ``other`` (shape/glue, ~0 GPU time) is absorbed proportionally,
    so ``dense_unattributed_ms`` stays 0 whenever named stages are present.
    """
    out: Dict[str, float] = {k: 0.0 for k in _BEVFUSION_DENSE_SUBSTAGE_KEYS}
    out["dense_unattributed_ms"] = 0.0
    if dense_total_ms <= 0.0:
        return out

    named_sum = sum(stage_sums.get(k, 0.0) for k in _BEVFUSION_DENSE_SUBSTAGE_KEYS)
    if named_sum > 0.0:
        scale = dense_total_ms / named_sum
        for k in _BEVFUSION_DENSE_SUBSTAGE_KEYS:
            out[k] = stage_sums.get(k, 0.0) * scale
        out["dense_unattributed_ms"] = 0.0
    else:
        out["dense_unattributed_ms"] = dense_total_ms
    return out
