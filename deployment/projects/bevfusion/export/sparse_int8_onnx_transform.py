"""Post-process an ONNX model to replace ImplicitGemm → ImplicitGemmInt8.

Path B approach: The standard Autoware ONNX export (via torch.onnx.export +
sparse_functional.py symbolic methods) produces autoware::ImplicitGemm nodes
with 5 inputs. This script enriches them to autoware::ImplicitGemmInt8 nodes
with 7 inputs (+ channel_scale + bias_scaled) and INT8 scale attributes.

Usage::

    python -m deployment.projects.bevfusion.export.sparse_int8_onnx_transform \\
        --onnx work_dirs/bevfusion/sparse_encoder.onnx \\
        --checkpoint work_dirs/bevfusion/bevfusion_epoch_30_ptq_sparse_only.pth \\
        --config projects/BEVFusion/configs/.../bevfusion_..._120m.py \\
        --output work_dirs/bevfusion/sparse_encoder_int8_pathb.onnx

The output ONNX can be loaded by TensorRT with the ImplicitGemmInt8Plugin.

Debugging / scale audit::

    # Per-layer JSON (matched ONNX node ↔ PTQ stem, input/output scales, channel_scale stats)
    python -m deployment.projects.bevfusion.export.sparse_int8_onnx_transform ... --audit-report int8_layers.json

    # Read-only dump of an already-transformed INT8 ONNX (no checkpoint)
    python -m deployment.projects.bevfusion.export.sparse_int8_onnx_audit --onnx sparse_int8.onnx
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from deployment.projects.bevfusion.export.onnx_fuse_implicit_gemm_activation import (
    _normalize_attr,
    _try_get_constant_numpy,
)


@dataclass
class DeployTransformOptions:
    """Optional transform knobs loaded from a BEVFusion deploy_config."""

    fp16_layer_patterns: List[str]
    plugin_timing_enabled: bool
    plugin_timing_max_logs: int
    fuse_implicit_gemm_relu: bool


def _load_deploy_transform_options(deploy_cfg_path: Optional[str]) -> DeployTransformOptions:
    """Read sparse-int8 transform options from deploy config.

    Supported keys:
    - spconv_int8_fp16_layers
    - implicit_gemm_int8_plugin_timing
    - implicit_gemm_int8_plugin_timing_max_logs
    - spconv_int8_fuse_implicit_gemm_relu
    """
    default = DeployTransformOptions(
        fp16_layer_patterns=[],
        plugin_timing_enabled=False,
        plugin_timing_max_logs=1000,
        fuse_implicit_gemm_relu=True,
    )
    if not deploy_cfg_path:
        return default

    try:
        from mmengine import Config  # type: ignore
    except ImportError as e:
        raise SystemExit("--deploy-cfg requires mmengine; install it or avoid --deploy-cfg.") from e

    deploy_cfg = Config.fromfile(deploy_cfg_path)

    cfg_list = deploy_cfg.get("spconv_int8_fp16_layers", []) or []
    if not isinstance(cfg_list, (list, tuple)):
        raise SystemExit(
            f"spconv_int8_fp16_layers in {deploy_cfg_path!r} must be a list of strings, "
            f"got {type(cfg_list).__name__}."
        )

    return DeployTransformOptions(
        fp16_layer_patterns=[str(p) for p in cfg_list if str(p).strip()],
        plugin_timing_enabled=bool(deploy_cfg.get("implicit_gemm_int8_plugin_timing", False)),
        plugin_timing_max_logs=int(deploy_cfg.get("implicit_gemm_int8_plugin_timing_max_logs", 1000)),
        fuse_implicit_gemm_relu=bool(deploy_cfg.get("spconv_int8_fuse_implicit_gemm_relu", True)),
    )


def _load_amax_from_checkpoint(
    checkpoint_path: str,
) -> Dict[str, torch.Tensor]:
    """Load all _amax tensors from the PTQ checkpoint.

    Returns dict mapping dotted key → amax tensor.
    Keys look like:
        pts_middle_encoder.conv_input.0._input_quantizer._amax
        pts_middle_encoder.conv_input.0._weight_quantizer._amax
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    amax_dict = {}
    for k, v in sd.items():
        if "_amax" in k:
            amax_dict[k] = v
    return amax_dict


def _build_layer_scale_map(
    amax_dict: Dict[str, torch.Tensor],
    encoder_prefix: str = "pts_middle_encoder.",
) -> Dict[str, dict]:
    """Build a map: layer_stem → {input_amax, weight_amax, output_amax}.

    For each SparseConvolution layer, we extract:
    - input_amax: _input_quantizer._amax
    - weight_amax: _weight_quantizer._amax (per-**C_out**; ``TensorQuantizer(axis=(0))`` on
      5D weights ``[C_out,k,k,k,C_in]`` — not legacy axis=4 / C_in-only blobs)

    The output_amax for layer N is the input_amax for layer N+1
    (or the final encoder output scale).
    """
    layers = OrderedDict()

    pattern = re.compile(
        r"^(?:module\.)?(?P<prefix>pts_middle_encoder\.)" r"(?P<stem>.+?)\._(?P<kind>input|weight)_quantizer\._amax$"
    )

    for key, val in sorted(amax_dict.items()):
        m = pattern.match(key)
        if m is None:
            continue
        stem = m.group("stem")
        kind = m.group("kind")
        if stem not in layers:
            layers[stem] = {}
        layers[stem][f"{kind}_amax"] = val.cpu().numpy().astype(np.float32)

    return dict(layers)


def _conv_out_input_amax_from_checkpoint(
    amax_dict: Dict[str, torch.Tensor],
) -> Optional[np.ndarray]:
    """Optional ``conv_out.*._input_quantizer._amax`` if PTQ attached a quantizer there."""
    for k, v in amax_dict.items():
        if "pts_middle_encoder." not in k and "module.pts_middle_encoder." not in k:
            continue
        if "conv_out" not in k:
            continue
        if "_input_quantizer._amax" in k:
            return v.cpu().numpy().astype(np.float32)
    return None


def _pathb_sparse_tail_absmax_from_state_dict(
    encoder_sd: Optional[Dict[str, torch.Tensor]],
) -> Optional[np.ndarray]:
    """``calibrate_spconv_nvidia`` saves max |features| **entering** ``conv_out`` (post tail BN/ReLU)."""
    if encoder_sd is None:
        return None
    for k in (
        "pts_middle_encoder._pathb_sparse_tail_absmax",
        "module.pts_middle_encoder._pathb_sparse_tail_absmax",
        "_pathb_sparse_tail_absmax",
    ):
        if k not in encoder_sd:
            continue
        t = encoder_sd[k]
        v = float(torch.as_tensor(t).float().reshape(-1)[0].item())
        if v > 0.0 and np.isfinite(v):
            return np.array([v], dtype=np.float32)
    return None


def _pathb_last_int8_conv_output_from_state_dict(
    encoder_sd: Optional[Dict[str, torch.Tensor]],
) -> Optional[np.ndarray]:
    """Max |features| at **output of last quantized SparseConv** (matches ImplicitGemmInt8 output)."""
    if encoder_sd is None:
        return None
    for k in (
        "pts_middle_encoder._pathb_last_int8_conv_output_absmax",
        "module.pts_middle_encoder._pathb_last_int8_conv_output_absmax",
        "_pathb_last_int8_conv_output_absmax",
    ):
        if k not in encoder_sd:
            continue
        t = encoder_sd[k]
        v = float(torch.as_tensor(t).float().reshape(-1)[0].item())
        if v > 0.0 and np.isfinite(v):
            return np.array([v], dtype=np.float32)
    return None


def _terminal_boundary_amax_for_pathb(
    encoder_sd: Optional[Dict[str, torch.Tensor]],
    amax_dict: Optional[Dict[str, torch.Tensor]],
    *,
    override_terminal_absmax: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """Scalar amax for the **last** INT8 layer's ``output_scale`` (that conv's linear FP output).

    Prefer ``_pathb_last_int8_conv_output_absmax`` (post-conv, pre tail BN/ReLU).  The older
    ``_pathb_sparse_tail_absmax`` tracks the tensor entering ``conv_out`` and can be **too large**
    for ``ImplicitGemmInt8.output_scale``, inflating TRT ``lidar_bev``.
    """
    if override_terminal_absmax is not None:
        v = float(override_terminal_absmax)
        if v > 0.0 and np.isfinite(v):
            return np.array([v], dtype=np.float32), "cli_override"
    li = _pathb_last_int8_conv_output_from_state_dict(encoder_sd)
    if li is not None:
        return li, "pathb_last_int8_conv_output_absmax"
    pb = _pathb_sparse_tail_absmax_from_state_dict(encoder_sd)
    if pb is not None:
        return pb, "pathb_sparse_tail_absmax_legacy"
    ad = amax_dict or {}
    cq = _conv_out_input_amax_from_checkpoint(ad)
    if cq is not None:
        return cq, "conv_out_input_quantizer"
    return None, "missing"


def _resolve_int8_output_amax(
    stem: str,
    layer_scales: Dict[str, dict],
    topo_stems: List[str],
    terminal_boundary_np: Optional[np.ndarray],
    terminal_src: str,
    verbose: bool,
) -> Tuple[np.ndarray, str]:
    """Return (output_amax_array, reason_tag) for ImplicitGemmInt8 ``output_scale``."""
    nxt = _successor_stem_for_int8_output_scale(stem, topo_stems)
    if nxt is not None and nxt in layer_scales:
        oa = layer_scales[nxt].get("input_amax")
        if oa is not None:
            return oa, f"next_layer_input:{nxt}"
        raise ValueError(
            f"Path B INT8: stem {stem!r} needs output_scale from successor {nxt!r} "
            f"input_amax, but layer_scales[{nxt!r}]['input_amax'] is missing. "
            "Re-run sparse PTQ or fix checkpoint _amax keys."
        )

    if terminal_boundary_np is not None:
        if verbose:
            hint = {
                "pathb_last_int8_conv_output_absmax": "last INT8 SparseConv output (pre tail BN/ReLU)",
                "pathb_sparse_tail_absmax_legacy": "conv_out input (legacy; can over-scale last Gemm)",
                "cli_override": "--pathb-terminal-absmax",
                "conv_out_input_quantizer": "conv_out._input_quantizer._amax",
            }.get(terminal_src, terminal_src)
            print(f"  [int8-output-scale] {stem}: terminal layer → {terminal_src} ({hint})")
        return terminal_boundary_np, terminal_src

    raise ValueError(
        f"Path B INT8: no output_amax for stem {stem!r} (successor {nxt!r} is outside "
        "quantized stems, e.g. conv_out). Checkpoint must contain "
        "pts_middle_encoder._pathb_last_int8_conv_output_absmax (preferred), "
        "_pathb_sparse_tail_absmax, or conv_out _input_quantizer._amax; or pass "
        "--pathb-terminal-absmax to sparse_int8_onnx_transform."
    )


def _tail_without_encoder_layers(stem: str) -> str:
    if stem.startswith("encoder_layers."):
        return stem[len("encoder_layers.") :]
    return stem


def _parse_encoder_stage_block(tail: str) -> Optional[Tuple[int, int]]:
    m = re.match(r"^encoder_layer(\d+)\.(\d+)\.", tail)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _topologically_sorted_sparse_stems(stems: List[str]) -> List[str]:
    """Order sparse conv stems in forward-ish order (not lexicographic on full string).

    Lexicographic order on checkpoint keys breaks residual blocks: ``conv2`` is followed by
    ``downsample`` in the same ``encoder_layerS.B`` group, but those ops are **parallel**
    branches; the quantized output of ``conv2`` should align with the **next block's**
    input scale (or conv_out), not with ``downsample``'s input amax. Wrong pairing zeros
    BEV features in TRT and yields mAP 0.

    ONNX often names the stage tail stride conv ``encoder_layerS.B.0`` (no ``downsample`` in
    the path); those stems must sort **after** the previous block's convs and **before** the
    next stage's ``conv1``.
    """

    def sort_key(s: str) -> Tuple[int, int, int, int, str]:
        tail = _tail_without_encoder_layers(s)
        m_ci = re.match(r"^conv_input(?:\.(\d+))?$", tail)
        if m_ci:
            return (-1, 0, 0, int(m_ci.group(1) or 0), s)
        m_c = re.match(r"^encoder_layer(\d+)\.(\d+)\.(conv[12])(?:\.\d+)?$", tail)
        if m_c:
            stage, blk = int(m_c.group(1)), int(m_c.group(2))
            branch = 0 if m_c.group(3) == "conv1" else 1
            return (stage, blk, branch, 0, s)
        m_d = re.match(r"^encoder_layer(\d+)\.(\d+)\.downsample(?:\.(\d+))?$", tail)
        if m_d:
            stage, blk = int(m_d.group(1)), int(m_d.group(2))
            sub = int(m_d.group(3) or 0)
            return (stage, blk, 2, sub, s)
        # e.g. encoder_layer1.2.0 — SparseSequential stride block (ONNX path has no "downsample")
        m_tail = re.match(r"^encoder_layer(\d+)\.(\d+)\.(\d+)$", tail)
        if m_tail and "conv" not in tail and "downsample" not in tail:
            return (int(m_tail.group(1)), int(m_tail.group(2)), 3, int(m_tail.group(3)), s)
        return (10_000, 10_000, 10_000, 0, s)

    return sorted(stems, key=sort_key)


def _successor_stem_for_int8_output_scale(stem: str, topo: List[str]) -> Optional[str]:
    """Stem whose ``input_amax`` should supply ``output_scale`` for ``stem`` (activation chain)."""
    try:
        idx = topo.index(stem)
    except ValueError:
        return None
    tail = _tail_without_encoder_layers(stem)
    if re.match(r"^conv_input", tail):
        return topo[idx + 1] if idx + 1 < len(topo) else None

    pb = _parse_encoder_stage_block(tail)
    is_conv2 = ".conv2" in tail
    is_downsample = ".downsample" in tail
    # Stride block named encoder_layerS.B.0 (no conv/downsample in tail) — same skip rule as conv2.
    is_stride_tail = bool(
        re.match(r"^encoder_layer\d+\.\d+\.\d+$", tail) and "conv" not in tail and "downsample" not in tail
    )
    if pb is not None and (is_conv2 or is_downsample or is_stride_tail):
        stage, block = pb
        j = idx + 1
        while j < len(topo):
            tj = _tail_without_encoder_layers(topo[j])
            pb2 = _parse_encoder_stage_block(tj)
            if pb2 == (stage, block):
                j += 1
                continue
            break
        return topo[j] if j < len(topo) else None

    return topo[idx + 1] if idx + 1 < len(topo) else None


def _compute_int8_scales(
    input_amax: np.ndarray,
    weight_amax: np.ndarray,
    output_amax: np.ndarray,
    bias: Optional[np.ndarray],
) -> dict:
    """Compute INT8 quantization scales for a single layer.

    Returns dict with:
        input_scale: float (input_amax / 127)
        output_scale: float (output_amax / 127)
        channel_scale: np.ndarray [C_out]
        bias_scaled: np.ndarray [C_out]
    """
    input_scale = float(input_amax.flatten()[0]) / 127.0
    w_scales = weight_amax.flatten() / 127.0

    if output_amax is None:
        raise ValueError(
            "Path B INT8: output_amax is None in _compute_int8_scales — "
            "caller must resolve a valid successor input_amax or terminal boundary amax."
        )
    output_scale = float(output_amax.flatten()[0]) / 127.0

    channel_scale = (input_scale * w_scales) / output_scale

    if bias is not None:
        bias_scaled = bias / output_scale
    else:
        bias_scaled = np.zeros_like(channel_scale)

    return {
        "input_scale": input_scale,
        "output_scale": output_scale,
        "channel_scale": channel_scale.astype(np.float32),
        "bias_scaled": bias_scaled.astype(np.float32),
    }


def _collect_occupied_tensor_names(graph: onnx.GraphProto) -> set[str]:
    """All tensor names already used in the graph (avoid collisions with new initializers)."""
    out: set[str] = set()
    for init in graph.initializer:
        out.add(init.name)
    for vi in graph.input:
        out.add(vi.name)
    for vi in graph.output:
        out.add(vi.name)
    for vi in graph.value_info:
        out.add(vi.name)
    for n in graph.node:
        for t in list(n.input) + list(n.output):
            if t:
                out.add(t)
    return out


def _allocate_unique_tensor_name(suggested: str, occupied: set[str]) -> str:
    """Return ``suggested`` if unused; otherwise ``suggested_1``, ``suggested_2``, …"""
    name = suggested
    i = 0
    while name in occupied:
        i += 1
        name = f"{suggested}_{i}"
    occupied.add(name)
    return name


def _safe_trt_scale_names(stem: str, occupied: set[str]) -> tuple[str, str]:
    """Build unique initializer / ValueInfo names that TensorRT's ONNX parser accepts.

    Stems like ``encoder_layers.encoder_layer1.0.conv1`` produce dotted names; TRT often fails
    with ``INVALID_GRAPH: Failed to import initializer`` on those. Use alphanumerics + underscore.
    """
    base = re.sub(r"[^0-9A-Za-z_]", "_", stem)
    base = re.sub(r"_+", "_", base).strip("_")
    if not base:
        base = "layer"
    if base[0].isdigit():
        base = "L_" + base

    cs = f"{base}_channel_scale"
    bs = f"{base}_bias_scaled"
    suffix = 0
    while cs in occupied or bs in occupied:
        suffix += 1
        cs = f"{base}_channel_scale_{suffix}"
        bs = f"{base}_bias_scaled_{suffix}"
    occupied.add(cs)
    occupied.add(bs)
    return cs, bs


def _stem_variants_base(stem: str) -> List[str]:
    """Variants for substring matching.

    Only strip the known ``encoder_layers.`` prefix. Do **not** strip at the first dot in
    general: e.g. ``conv_input.0`` would become ``0`` and falsely match every ``*.0`` ONNX
    path (encoder_layer1.0.conv1, ...).
    """
    variants = [stem]
    if stem.startswith("encoder_layers."):
        variants.append(stem[len("encoder_layers.") :])
    return variants


def _stem_variants_for_onnx_match(stem: str) -> List[str]:
    """Variants that may appear in ONNX tensor paths (includes downsample alias)."""
    out: List[str] = []
    seen: set[str] = set()
    tail = _tail_without_encoder_layers(stem)
    # Checkpoint: encoder_layerS.B.downsample.0 — ONNX: ...encoder_layerS.B.0...
    m_ds = re.match(r"^encoder_layer(\d+)\.(\d+)\.downsample\.(\d+)$", tail)
    alias: Optional[str] = None
    if m_ds:
        alias = f"encoder_layer{m_ds.group(1)}.{m_ds.group(2)}.{m_ds.group(3)}"
    for v in _stem_variants_base(stem):
        for x in (v, alias) if alias else (v,):
            if x and x not in seen:
                seen.add(x)
                out.append(x)
    return out


def _implicit_gemm_filter_c_out_c_in(
    model: onnx.ModelProto, node: onnx.NodeProto
) -> Tuple[Optional[int], Optional[int]]:
    """Return ``(C_out, C_in)`` from 5D filter initializer (indices 0 and 4)."""
    if len(node.input) < 2:
        return None, None
    w = _get_initializer_data(model, node.input[1])
    if w is None or w.ndim != 5:
        return None, None
    return int(w.shape[0]), int(w.shape[4])


def _conv_weight_shape_from_state_dict(encoder_sd: Dict[str, torch.Tensor], stem: str) -> Optional[Tuple[int, int]]:
    """``(C_out, C_in)`` for ``pts_middle_encoder.{stem}.weight`` if present."""
    for prefix in ("pts_middle_encoder.", "module.pts_middle_encoder."):
        k = f"{prefix}{stem}.weight"
        if k not in encoder_sd:
            continue
        t = encoder_sd[k]
        sh = getattr(t, "shape", None)
        if sh is not None and len(sh) == 5:
            return int(sh[0]), int(sh[-1])
    return None


_LEGACY_W_AMAX_PTQ = (
    "Re-run sparse PTQ after updating apply_nvidia_spconv_int8 to use "
    "QuantDescriptor(axis=(0)) for 5D sparse weights (not axis=(4)). "
    "See deployment/projects/bevfusion/quantization/spconv_int8.py."
)


def _weight_amax_per_cout_vector(
    weight_amax: np.ndarray,
    c_out: int,
    c_in: int,
    stem: str,
) -> np.ndarray:
    """Build length-``c_out`` vector for ``_compute_int8_scales`` (one scale per output channel).

    Expects calibration from ``TensorQuantizer(..., axis=(0))`` on weights shaped
    ``[C_out, k1, k2, k3, C_in]``.  Rejects legacy ``axis=(4)`` checkpoints where
    ``_amax`` is tied to **C_in** (shape ending in ``C_in`` with leading singletons).
    """
    w = np.asarray(weight_amax, dtype=np.float32)
    flat = w.reshape(-1)

    if flat.size == 1:
        return np.full(c_out, float(flat[0]), dtype=np.float32)

    if flat.size == c_out:
        return flat.astype(np.float32)

    if w.ndim == 5 and int(w.shape[0]) == c_out:
        return w.reshape(c_out, -1).max(axis=1).astype(np.float32)

    if (w.ndim == 5 and int(w.shape[0]) == 1 and int(w.shape[-1]) == c_in and c_in != c_out) or (
        flat.size == c_in and c_in != c_out
    ):
        raise ValueError(
            f"{stem}: weight _amax shape {tuple(w.shape)} matches **C_in={c_in}** calibration "
            f"(legacy axis=4). INT8 Path B needs **C_out={c_out}** per-output scales. " + _LEGACY_W_AMAX_PTQ
        )

    raise ValueError(
        f"{stem}: weight _amax shape {tuple(w.shape)} is not usable as {c_out} per-output "
        f"scales (C_in={c_in}). " + _LEGACY_W_AMAX_PTQ
    )


def _weight_amax_matches_cout_layout(
    weight_amax: np.ndarray,
    c_out: int,
    c_in: Optional[int] = None,
) -> bool:
    """True if ``_weight_amax_per_cout_vector`` would accept this tensor (no legacy C_in-only)."""
    if weight_amax is None or c_out <= 0:
        return False
    w = np.asarray(weight_amax)
    flat = w.reshape(-1)
    if flat.size in (1, c_out):
        return True
    if w.ndim == 5 and int(w.shape[0]) == c_out:
        return True
    if c_in is not None:
        if flat.size == c_in and c_in != c_out:
            return False
        if w.ndim == 5 and int(w.shape[0]) == 1 and int(w.shape[-1]) == c_in and c_in != c_out:
            return False
    return False


def _all_substring_matching_stems(
    node: onnx.NodeProto,
    layer_stems: List[str],
) -> List[str]:
    if not node.input:
        return []
    normalized_inputs: List[str] = []
    for inp in node.input:
        normalized_inputs.append(inp.lstrip("/").replace("/", "."))
    found: List[str] = []
    seen: set[str] = set()
    for stem in layer_stems:
        for variant in _stem_variants_for_onnx_match(stem):
            variant_slash = variant.replace(".", "/")
            hit = False
            for norm in normalized_inputs:
                if variant in norm:
                    hit = True
                    break
            if not hit:
                for inp in node.input:
                    if variant_slash in inp:
                        hit = True
                        break
            if hit and stem not in seen:
                seen.add(stem)
                found.append(stem)
                break
    return found


def _stem_node_name_match_score(node_name: str, stem: str) -> int:
    """How strongly ``stem`` is supported by the ImplicitGemm node's **name** (scope path).

    Input tensor names can still contain ``conv1`` when they carry activations **into** ``conv2``
    (Residual naming), which makes substring-only matching ambiguous for sibling convs that share
    ``(C_out, C_in)``.

    ONNX node names use ``/`` (e.g. ``...encoder_layer3.0/conv2/...``) while PTQ stems use ``.``
    (``encoder_layer3.0.conv2``). Flattening ``/`` → ``.`` makes dotted stem variants match.
    """
    if not node_name or not stem:
        return 0
    n_lower = node_name.lower().replace("\\", "/")
    # Align ``…/encoder_layer3.0/conv2/…`` with stem substring ``encoder_layer3.0.conv2``.
    n_dots = n_lower.replace("/", ".")
    while ".." in n_dots:
        n_dots = n_dots.replace("..", ".")

    best = 0
    for v in _stem_variants_for_onnx_match(stem):
        if not v:
            continue
        vl = v.lower()
        if vl and vl in n_dots:
            best = max(best, len(vl))
        vs = vl.replace(".", "/")
        if vs and vs in n_lower:
            best = max(best, len(vs))

    # Strong tie-break: stem tail (``conv1`` vs ``conv2``) as a path segment in node.name.
    tail = stem.rstrip(".").split(".")[-1].lower()
    if tail and (
        f"/{tail}/" in n_lower or f"/{tail}." in n_lower or n_lower.endswith("/" + tail) or f".{tail}." in n_dots
    ):
        best += 4096

    return best


def _pick_stem_disambiguated(node: onnx.NodeProto, stems: List[str]) -> Optional[str]:
    """Choose one stem when multiple candidates share weight-shape / _amax compatibility."""
    if not stems:
        return None
    if len(stems) == 1:
        return stems[0]
    name = node.name or ""
    return max(stems, key=lambda s: (_stem_node_name_match_score(name, s), len(s)))


def _match_onnx_node_to_layer(
    node: onnx.NodeProto,
    model: onnx.ModelProto,
    layer_stems: List[str],
    layer_scales: Dict[str, dict],
    encoder_sd: Optional[Dict[str, torch.Tensor]] = None,
    verbose: bool = False,
) -> Optional[str]:
    """Match ImplicitGemm node to calibration stem.

    Prefer ``state_dict`` 5D weight ``(C_out, C_in)`` vs ONNX filter (strongest). Then fall back
    to ``_amax`` shape heuristics. Substring matching is kept; ``encoder_layers.`` is the only
    auto-stripped prefix (never strip ``conv_input.0`` → ``0``).

    When several stems share the same weight layout (e.g. ``conv1`` / ``conv2`` same channels),
    ties are broken using **node.name** scope, not longest stem string alone.
    """
    c_out, c_in = _implicit_gemm_filter_c_out_c_in(model, node)
    candidates = _all_substring_matching_stems(node, layer_stems)
    if verbose:
        print(
            f"  [debug-match] node={node.name!r} filter_c_out_c_in=({c_out},{c_in}) "
            f"candidates({len(candidates)})={candidates[:8]}{'...' if len(candidates) > 8 else ''}"
        )
        for s in candidates[:12]:
            w = layer_scales.get(s, {}).get("weight_amax")
            sh = getattr(w, "shape", None)
            sd_sh = _conv_weight_shape_from_state_dict(encoder_sd or {}, s) if encoder_sd else None
            print(f"    stem={s!r} weight_amax.shape={sh} state_dict_weight={sd_sh}")

    if not candidates:
        return None

    if encoder_sd and c_out is not None and c_in is not None:
        ok_sd = [s for s in candidates if _conv_weight_shape_from_state_dict(encoder_sd, s) == (c_out, c_in)]
        if len(ok_sd) == 1:
            return ok_sd[0]
        if len(ok_sd) > 1:
            return _pick_stem_disambiguated(node, ok_sd)
        if verbose and candidates:
            print(
                "  [debug-match] no stem with state_dict weight (C_out,C_in) matching ONNX filter; "
                "falling back to _amax heuristic"
            )

    if c_out is not None:
        ok = [
            s
            for s in candidates
            if _weight_amax_matches_cout_layout(layer_scales.get(s, {}).get("weight_amax"), c_out, c_in)
        ]
        if len(ok) == 1:
            return ok[0]
        if len(ok) > 1:
            return _pick_stem_disambiguated(node, ok)
        if verbose:
            print(f"  [debug-match] no stem with _amax compatible with C_out={c_out} C_in={c_in}.")
        return None

    return _pick_stem_disambiguated(node, candidates)


def _get_initializer_data(model: onnx.ModelProto, name: str) -> Optional[np.ndarray]:
    """Get numpy data from an ONNX initializer by name."""
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def _implicit_gemm_to_int8_path(node: onnx.NodeProto, fp16_patterns_norm: List[str]) -> bool:
    """Whether this ``autoware::ImplicitGemm`` should be replaced by ``ImplicitGemmInt8``."""
    if node.op_type != "ImplicitGemm" or node.domain != "autoware":
        return False
    if _implicit_gemm_matches_fp16_pattern(node, fp16_patterns_norm) is not None:
        return False
    return True


def _implicit_gemm_matches_fp16_pattern(node: onnx.NodeProto, patterns: Optional[List[str]]) -> Optional[str]:
    """Return the matching pattern (or None) if an ``ImplicitGemm`` node should be kept FP16.

    Each ``patterns`` entry is a case-insensitive substring matched **only against
    ``node.name``** — NOT against ``inputs`` / ``outputs``. PyTorch's ONNX exporter
    names output tensors with their producer's scope path (e.g. the Relu after
    ``conv_input.0`` still contains the literal substring ``conv_input.0`` in its
    tensor name), which then appears as an *input* on the *next* ImplicitGemm.
    Matching on the full text blob would therefore silently FP16-ify the downstream
    layer too — a subtle cause of large mAP drops. Matching only ``node.name``
    avoids that contamination because each node has a unique scope-qualified name.

    Exposed to users via ``spconv_int8_fp16_layers`` in the BEVFusion
    deploy_config.
    """
    if not patterns:
        return None
    name = (node.name or "").lower().replace("\\", "/")
    for pat in patterns:
        if not pat:
            continue
        if pat.lower() in name:
            return pat
    return None


def _implicit_gemm_filter_c_out(model: onnx.ModelProto, node: onnx.NodeProto) -> Optional[int]:
    c_out, _ = _implicit_gemm_filter_c_out_c_in(model, node)
    return c_out


def _implicit_gemm_attrs_from_node(node: onnx.NodeProto) -> Dict[str, object]:
    """Read ``ImplicitGemm`` attributes into a dict keyed by normalized names (no ``_f``/``_i``)."""
    out: Dict[str, object] = {}
    for attr in node.attribute:
        base = _normalize_attr(attr.name)
        if attr.type == onnx.AttributeProto.FLOAT:
            out[base] = float(attr.f)
        elif attr.type == onnx.AttributeProto.INT:
            out[base] = int(attr.i)
        elif attr.type == onnx.AttributeProto.STRING:
            out[base] = attr.s.decode("utf-8") if isinstance(attr.s, bytes) else str(attr.s)
    return out


def _append_implicit_gemm_int8_plugin_attributes(
    int8_node: onnx.NodeProto,
    attrs: Dict[str, object],
) -> None:
    """Set plugin fields exactly as TensorRT ``ImplicitGemmInt8PluginCreator`` expects (no ``_f`` names)."""

    def _f(key: str, default: float) -> float:
        v = attrs.get(key, default)
        return float(v) if v is not None else default

    def _i(key: str, default: int) -> int:
        v = attrs.get(key, default)
        return int(v) if v is not None else default

    int8_node.attribute.extend(
        [
            helper.make_attribute("act_alpha", _f("act_alpha", 0.0)),
            helper.make_attribute("act_beta", _f("act_beta", 0.0)),
            helper.make_attribute("is_subm", _i("is_subm", 0)),
            helper.make_attribute("output_scale", _f("output_scale", 1.0)),
            helper.make_attribute("input_scale", _f("input_scale", 1.0)),
            helper.make_attribute("timing_enabled", _i("timing_enabled", 0)),
            helper.make_attribute("timing_max_logs", _i("timing_max_logs", 1000)),
            helper.make_attribute("act_type", _i("act_type", 0)),
        ]
    )
    # Keep Autoware ImplicitGemm extras if present (shape / legacy parsers).
    for key in ("is_train", "fp32_accum", "output_add_scale"):
        if key not in attrs:
            continue
        v = attrs[key]
        if isinstance(v, bool):
            v = int(v)
        if isinstance(v, int):
            int8_node.attribute.append(helper.make_attribute(key, v))
        elif isinstance(v, float):
            int8_node.attribute.append(helper.make_attribute(key, v))


def transform_onnx_int8(
    model: onnx.ModelProto,
    layer_scales: Dict[str, dict],
    encoder_sd: Optional[Dict[str, torch.Tensor]] = None,
    verbose: bool = False,
    amax_dict: Optional[Dict[str, torch.Tensor]] = None,
    override_terminal_absmax: Optional[float] = None,
    audit_records: Optional[List[Dict[str, Any]]] = None,
    fp16_layer_patterns: Optional[List[str]] = None,
    plugin_timing_enabled: bool = False,
    plugin_timing_max_logs: int = 1000,
    fuse_implicit_gemm_trailing_relu: bool = True,
) -> onnx.ModelProto:
    """Replace ImplicitGemm nodes with ImplicitGemmInt8 nodes.

    Args:
        model: ONNX model with autoware::ImplicitGemm nodes.
        layer_scales: dict from _build_layer_scale_map.
        encoder_sd: **Required** for strict weight ``_amax`` layout (5D ``(C_out,C_in)``)
            and per-``C_out`` vectors; also used for bias.
        amax_dict: Raw ``_amax`` keys from the PTQ checkpoint (same as ``_load_amax_from_checkpoint``).
            Used for terminal ``output_scale`` via ``conv_out.*._input_quantizer._amax`` when Path-B
            buffers are absent. Prefer ``encoder_sd['_pathb_last_int8_conv_output_absmax']`` from sparse
            PTQ, or pass ``--pathb-terminal-absmax``.
        audit_records: If provided, append one JSON-serializable dict per converted
            ``ImplicitGemmInt8`` (for ``--audit-report`` or custom tooling).
        fp16_layer_patterns: Optional list of case-insensitive substring patterns.
            Any ``ImplicitGemm`` node whose **name** contains one of the patterns is
            **kept as FP16** ``ImplicitGemm`` (skipped INT8 replacement). Driven by
            ``spconv_int8_fp16_layers`` in the BEVFusion deploy_config. ``conv_out`` follows
            the same rule as other layers (no ONNX special-case skip).
        plugin_timing_enabled: When True, ONNX attributes ``timing_enabled`` / ``timing_max_logs``
            are set on each ``ImplicitGemmInt8`` so the TensorRT plugin logs CUDA-event splits
            (deploy_config: ``implicit_gemm_int8_plugin_timing``).
        plugin_timing_max_logs: Upper bound on timing log lines (stderr); same key in deploy_config.
        fuse_implicit_gemm_trailing_relu: When True, run ``fuse_autoware_implicit_gemm_trailing_relu``
            so standalone ``Relu`` chains on sparse conv outputs become ``ImplicitGemm.act_type=kReLU``.

    Returns:
        Modified ONNX model with autoware::ImplicitGemmInt8 nodes.

    Raises:
        ValueError: if terminal boundary amax is missing, stem/scale resolution fails, ONNX nodes
            cannot be matched, or checkpoint used legacy weight quantizer ``axis=(4)``.
        RuntimeError: if the count of converted ImplicitGemm nodes does not match the graph.
    """
    if encoder_sd is None:
        raise ValueError(
            "transform_onnx_int8 requires encoder state_dict (--checkpoint) to validate "
            "weight _amax vs each layer's (C_out, C_in). " + _LEGACY_W_AMAX_PTQ
        )

    model = copy.deepcopy(model)

    if fuse_implicit_gemm_trailing_relu:
        from deployment.projects.bevfusion.export.onnx_fuse_implicit_gemm_activation import (
            fuse_autoware_implicit_gemm_trailing_relu,
        )

        n_fused = fuse_autoware_implicit_gemm_trailing_relu(model)
        if n_fused:
            print(
                f"  [onnx-fuse] Merged {n_fused} ImplicitGemm→Relu chain(s): "
                "act_type=kReLU (1), Relu nodes removed."
            )

    graph = model.graph
    layer_stems = list(layer_scales.keys())
    occupied_names = _collect_occupied_tensor_names(graph)

    # Derive output_amax from the **activation successor** in topo order (not raw dict / lexical
    # order). See _topologically_sorted_sparse_stems docstring.
    topo_stems = _topologically_sorted_sparse_stems(list(layer_scales.keys()))
    if verbose:
        print(f"  [debug-topo] first_stems={topo_stems[:6]}... last={topo_stems[-3:]}")

    term_np, term_src = _terminal_boundary_amax_for_pathb(
        encoder_sd, amax_dict, override_terminal_absmax=override_terminal_absmax
    )
    if term_np is None:
        raise ValueError(
            "Path B ONNX transform: checkpoint has no terminal boundary amax. Sparse PTQ must save "
            "pts_middle_encoder._pathb_last_int8_conv_output_absmax (preferred) or "
            "_pathb_sparse_tail_absmax, or calibrate conv_out._input_quantizer._amax, or pass "
            "--pathb-terminal-absmax when running sparse_int8_onnx_transform."
        )
    print(
        f"  [int8-output-scale] Terminal boundary: source={term_src} "
        f"amax={float(term_np.reshape(-1)[0]):.6f} → output_scale={float(term_np.reshape(-1)[0]) / 127.0:.6f}"
    )
    if verbose and term_src == "pathb_sparse_tail_absmax_legacy":
        print(
            "  [int8-output-scale] Using pts_middle_encoder._pathb_sparse_tail_absmax for terminal "
            "scale (legacy). Prefer re-running sparse PTQ to get "
            "_pathb_last_int8_conv_output_absmax — tail-at-conv_out can over-scale the last "
            "ImplicitGemmInt8 and inflate TRT lidar_bev."
        )

    scale_info = {}
    for stem in topo_stems:
        info = layer_scales[stem]
        input_amax = info.get("input_amax")
        weight_amax = info.get("weight_amax")

        if input_amax is None or weight_amax is None:
            raise ValueError(
                f"Path B INT8: stem {stem!r} is missing input_amax or weight_amax in layer_scales; "
                "fix sparse PTQ / _build_layer_scale_map."
            )

        output_amax, oa_tag = _resolve_int8_output_amax(stem, layer_scales, topo_stems, term_np, term_src, verbose)

        # Try to get bias from state_dict.
        bias = None
        if encoder_sd is not None:
            for bk in [f"{stem}.bias", f"pts_middle_encoder.{stem}.bias"]:
                if bk in encoder_sd:
                    b = encoder_sd[bk]
                    if hasattr(b, "numpy"):
                        bias = b.cpu().numpy().astype(np.float32)
                    break

        sh_w = _conv_weight_shape_from_state_dict(encoder_sd, stem)
        if sh_w is None:
            raise ValueError(
                f"{stem}: no pts_middle_encoder.{stem}.weight (5D) in checkpoint; "
                "cannot validate weight _amax. " + _LEGACY_W_AMAX_PTQ
            )
        w_for_scale = _weight_amax_per_cout_vector(weight_amax, sh_w[0], sh_w[1], stem)

        scale_info[stem] = _compute_int8_scales(input_amax, w_for_scale, output_amax, bias)
        if verbose:
            print(
                f"  [debug-scale] {stem}: w_amax.shape={np.shape(weight_amax)} "
                f"output_amax_tag={oa_tag!r} output_scale={scale_info[stem]['output_scale']:.6f}"
            )

    fp16_patterns_norm: List[str] = [p.lower() for p in (fp16_layer_patterns or []) if p]
    if fp16_patterns_norm:
        print(f"  [int8] spconv_int8_fp16_layers patterns (kept FP16): {fp16_patterns_norm}")

    n_expected_int8 = sum(1 for n in graph.node if _implicit_gemm_to_int8_path(n, fp16_patterns_norm))
    # Track which patterns matched nodes, to warn about typos / dead patterns.
    fp16_pattern_hits: Dict[str, int] = {p: 0 for p in fp16_patterns_norm}

    # Replace nodes.
    new_nodes = []
    transform_count = 0
    stem_assigned_to_node: Dict[str, str] = {}
    init_map: Dict[str, np.ndarray] = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    for node in graph.node:
        if node.op_type != "ImplicitGemm" or node.domain != "autoware":
            new_nodes.append(node)
            continue

        matched_fp16 = _implicit_gemm_matches_fp16_pattern(node, fp16_patterns_norm)
        if matched_fp16 is not None:
            fp16_pattern_hits[matched_fp16] += 1
            print(
                f"  [int8] Keep FP16 ImplicitGemm per spconv_int8_fp16_layers "
                f"(pattern={matched_fp16!r}): name={node.name!r}"
            )
            new_nodes.append(node)
            continue

        stem = _match_onnx_node_to_layer(
            node,
            model,
            layer_stems,
            layer_scales,
            encoder_sd=encoder_sd,
            verbose=verbose,
        )
        if stem is None or stem not in scale_info:
            raise ValueError(
                "Path B ONNX transform: ImplicitGemm node could not be matched to a calibrated stem "
                f"or scales were not built for it: name={node.name!r} inputs={list(node.input)}. "
                "Use --verbose on sparse_int8_onnx_transform to debug stem matching."
            )

        prev = stem_assigned_to_node.get(stem)
        if prev is not None:
            raise ValueError(
                "Path B ONNX transform: duplicate PTQ stem assignment — two ImplicitGemm nodes "
                f"matched the same stem {stem!r} (first={prev!r}, second={node.name!r}). "
                "This usually means substring-based matching is ambiguous; run with --verbose "
                "and fix ONNX tensor naming or matching heuristics."
            )
        stem_assigned_to_node[stem] = node.name or f"<unnamed_{transform_count}>"

        si = scale_info[stem]
        c_scale = len(si["channel_scale"])
        c_filter = _implicit_gemm_filter_c_out(model, node)
        if c_filter is not None and c_scale != c_filter:
            raise ValueError(
                "Path B ONNX transform: channel_scale length does not match filter C_out "
                f"(wrong stem match?): stem={stem!r} channel_scale_len={c_scale} "
                f"filter_c_out={c_filter} node={node.name!r}"
            )

        c_out = c_scale

        # FP16 fusion may add a 6th tensor input (Add folded into ImplicitGemm). INT8 uses 5 sparse
        # tensors + scales; merge that extra FP32/Half bias into bias_scaled (= bias / output_scale).
        bs_arr = np.asarray(si["bias_scaled"], dtype=np.float32).reshape(-1).copy()
        if len(node.input) == 6:
            extra_name = node.input[5]
            extra = _try_get_constant_numpy(graph, extra_name, init_map)
            if extra is None:
                raise ValueError(
                    "Path B ONNX transform: ImplicitGemm "
                    f"{node.name!r} has 6 inputs (ONNX-fused bias) but constant "
                    f"{extra_name!r} is not an initializer or Constant node value."
                )
            ex = np.asarray(extra, dtype=np.float32).reshape(-1)
            if ex.size != bs_arr.size:
                raise ValueError(
                    "Path B ONNX transform: fused 6th-input bias length "
                    f"{ex.size} != C_out {bs_arr.size} (stem={stem!r}, node={node.name!r})."
                )
            out_sc = float(si["output_scale"])
            bs_arr = bs_arr + (ex / out_sc)
            if verbose:
                print(
                    f"  [int8] Merged ONNX 6th-input fused bias into bias_scaled "
                    f"(stem={stem!r}, node={node.name!r})"
                )

        # Create ONNX initializers for channel_scale and bias_scaled.
        cs_name, bs_name = _safe_trt_scale_names(stem, occupied_names)

        cs_init = numpy_helper.from_array(si["channel_scale"], name=cs_name)
        bs_init = numpy_helper.from_array(bs_arr, name=bs_name)
        graph.initializer.append(cs_init)
        graph.initializer.append(bs_init)
        init_map[cs_name] = np.asarray(si["channel_scale"], dtype=np.float32)
        init_map[bs_name] = bs_arr

        # TRT's ONNX parser requires graph.input entries with type info
        # for all initializers referenced by custom plugin nodes.
        cs_vi = helper.make_tensor_value_info(cs_name, TensorProto.FLOAT, [c_out])
        bs_vi = helper.make_tensor_value_info(bs_name, TensorProto.FLOAT, [c_out])
        graph.input.append(cs_vi)
        graph.input.append(bs_vi)

        # Preserve existing attributes (normalize names), override INT8 scales from PTQ.
        attrs = _implicit_gemm_attrs_from_node(node)
        attrs["output_scale"] = float(si["output_scale"])
        attrs["input_scale"] = float(si["input_scale"])
        attrs["timing_enabled"] = int(bool(plugin_timing_enabled))
        attrs["timing_max_logs"] = int(plugin_timing_max_logs)

        # Fused FP16 export may have 6 inputs (optional per-channel bias); Int8 uses 5 sparse + scales.
        if len(node.input) not in (5, 6):
            raise ValueError(
                f"Path B: autoware::ImplicitGemm {node.name!r} has {len(node.input)} inputs; "
                "expected 5 or 6 (6 = ONNX-fused bias). Take first 5 as sparse tensors."
            )
        sparse_in = list(node.input[:5])

        int8_node = helper.make_node(
            "ImplicitGemmInt8",
            inputs=sparse_in + [cs_name, bs_name],
            outputs=list(node.output),
            domain="autoware",
            name=f"{node.name}_int8" if node.name else f"ImplicitGemmInt8_{transform_count}",
        )
        _append_implicit_gemm_int8_plugin_attributes(int8_node, attrs)

        new_nodes.append(int8_node)
        transform_count += 1
        print(
            f"  [int8] {stem}: input_scale={si['input_scale']:.6f} "
            f"output_scale={si['output_scale']:.6f} "
            f"channel_scale_shape={si['channel_scale'].shape}"
        )

        if audit_records is not None:
            cs = si["channel_scale"].reshape(-1).astype(np.float64)
            c_out_i, c_in_i = _implicit_gemm_filter_c_out_c_in(model, node)
            audit_records.append(
                {
                    "implicit_gemm_node_name": node.name or "",
                    "implicit_gemm_int8_node_name": int8_node.name or "",
                    "matched_stem": stem,
                    "filter_input": node.input[1] if len(node.input) > 1 else "",
                    "c_out": int(c_out_i) if c_out_i is not None else None,
                    "c_in": int(c_in_i) if c_in_i is not None else None,
                    "input_scale": float(si["input_scale"]),
                    "output_scale": float(si["output_scale"]),
                    "channel_scale_len": int(cs.size),
                    "channel_scale_min": float(cs.min()) if cs.size else None,
                    "channel_scale_max": float(cs.max()) if cs.size else None,
                    "channel_scale_mean": float(cs.mean()) if cs.size else None,
                    "channel_scale_initializer": cs_name,
                    "bias_scaled_initializer": bs_name,
                }
            )

    # Replace nodes in graph.
    del graph.node[:]
    graph.node.extend(new_nodes)

    if transform_count != n_expected_int8:
        raise RuntimeError(
            f"Path B ONNX transform: expected {n_expected_int8} ImplicitGemm → ImplicitGemmInt8 "
            f"replacements (excluding conv_out), got {transform_count}. "
            "Graph/calibration mismatch."
        )

    unused_stems = set(scale_info.keys()) - set(stem_assigned_to_node.keys())
    if unused_stems:
        print(
            "\n  [int8-audit] WARNING: calibrated stems with no matched ImplicitGemm node "
            f"(count={len(unused_stems)}): {sorted(unused_stems)[:12]}"
            f"{'...' if len(unused_stems) > 12 else ''}"
        )

    # Final census: enumerate every autoware::ImplicitGemm{,Int8} node that will be shipped
    # to TensorRT, so the user can eyeball exactly which sparse-conv layers run FP16 vs INT8.
    # Indispensable when debugging spconv_int8_fp16_layers / mAP regressions: if a node you
    # expected to be FP16 shows up under "INT8 nodes", the fp16 keep-list did not match.
    final_fp16_nodes: List[str] = []
    final_int8_nodes: List[str] = []
    for n in graph.node:
        if n.domain != "autoware":
            continue
        if n.op_type == "ImplicitGemm":
            final_fp16_nodes.append(n.name or "<unnamed>")
        elif n.op_type == "ImplicitGemmInt8":
            final_int8_nodes.append(n.name or "<unnamed>")
    print("\n  [int8-census] Final autoware::ImplicitGemm node types in output ONNX:")
    print(f"  [int8-census]   ImplicitGemm     (FP16, kept): {len(final_fp16_nodes)}")
    for nm in final_fp16_nodes:
        print(f"  [int8-census]     - {nm}")
    print(f"  [int8-census]   ImplicitGemmInt8 (INT8 conv): {len(final_int8_nodes)}")
    for nm in final_int8_nodes:
        print(f"  [int8-census]     - {nm}")

    unmatched_fp16 = [p for p, hits in fp16_pattern_hits.items() if hits == 0]
    if unmatched_fp16:
        print(
            "\n  [int8-audit] WARNING: spconv_int8_fp16_layers patterns did NOT match any node "
            f"(likely a typo or stale entry): {unmatched_fp16}"
        )
    if fp16_pattern_hits and any(v > 0 for v in fp16_pattern_hits.values()):
        matched_summary = ", ".join(f"{p!r}:{v}" for p, v in fp16_pattern_hits.items() if v > 0)
        print(f"\n  [int8-audit] spconv_int8_fp16_layers match counts: {matched_summary}")

    print(f"\nTransformed {transform_count} ImplicitGemm → ImplicitGemmInt8 nodes")
    return model


def main():
    parser = argparse.ArgumentParser(description="Transform ONNX ImplicitGemm nodes to ImplicitGemmInt8")
    parser.add_argument("--onnx", required=True, help="Input ONNX model path")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="PTQ checkpoint with NVIDIA _amax calibration values",
    )
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument(
        "--config",
        default=None,
        help="MMEngine config (optional, for bias extraction from fresh model)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print topology, scale chain, and per-node stem matching diagnostics",
    )
    parser.add_argument(
        "--pathb-terminal-absmax",
        type=float,
        default=None,
        help="Override scalar amax for the last INT8 layer output_scale (= absmax/127). "
        "Use if the checkpoint lacks Path-B buffers or you need a one-off fix.",
    )
    parser.add_argument(
        "--audit-report",
        default=None,
        help="Write JSON array of per-layer INT8 scale summaries (matched stem, scales, channel_scale stats).",
    )
    parser.add_argument(
        "--fp16-layers",
        default=None,
        help=(
            "Comma-separated list of case-insensitive substring patterns. Any ImplicitGemm "
            "node whose name/inputs/outputs contains one of these substrings is kept FP16 "
            "instead of being replaced by ImplicitGemmInt8 (for accuracy tuning). "
            "Example: --fp16-layers 'encoder_layer3.encoder_layer3.2,conv_input.0'"
        ),
    )
    parser.add_argument(
        "--deploy-cfg",
        default=None,
        help=(
            "Loads deploy_config .py for "
            "spconv_int8_fp16_layers, implicit_gemm_int8_plugin_timing, and "
            "implicit_gemm_int8_plugin_timing_max_logs."
        ),
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Load ONNX.
    print(f"Loading ONNX: {args.onnx}")
    model = onnx.load(args.onnx)

    # Load _amax values.
    print(f"Loading _amax from: {args.checkpoint}")
    amax_dict = _load_amax_from_checkpoint(args.checkpoint)
    print(f"  Found {len(amax_dict)} _amax keys")

    layer_scales = _build_layer_scale_map(amax_dict)
    print(f"  Matched {len(layer_scales)} sparse conv layers")
    ci0 = layer_scales.get("conv_input.0", {})
    wa0 = ci0.get("weight_amax")
    if wa0 is not None:
        s = np.asarray(wa0).shape
        print(
            f"  [sanity] conv_input.0 weight_amax shape={s} "
            f"(Path B expects per-C_out, e.g. (16,1,1,1,1); "
            f"(1,1,1,1,5) means an old checkpoint or wrong --checkpoint path)"
        )

    # Optional: load encoder state_dict for bias.
    encoder_sd = None
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    encoder_sd = ckpt.get("state_dict", ckpt)

    deploy_cfg_path = getattr(args, "deploy_cfg", None)
    deploy_opts = _load_deploy_transform_options(deploy_cfg_path)
    fp16_layer_patterns = list(deploy_opts.fp16_layer_patterns)

    if args.fp16_layers:
        fp16_layer_patterns.extend(p.strip() for p in args.fp16_layers.split(",") if p.strip())

    fp16_layer_patterns = list(dict.fromkeys(fp16_layer_patterns))

    print(f"\nTransforming ImplicitGemm → ImplicitGemmInt8...")
    if fp16_layer_patterns:
        print(f"  FP16 keep-list ({len(fp16_layer_patterns)}): {fp16_layer_patterns}")
    print(
        f"  ImplicitGemmInt8 plugin CUDA timing attrs: timing_enabled={deploy_opts.plugin_timing_enabled} "
        f"timing_max_logs={deploy_opts.plugin_timing_max_logs}"
        + (f" (from deploy_cfg {deploy_cfg_path!r})" if deploy_cfg_path else "")
    )
    print(
        "  ImplicitGemm ReLU/Add ONNX fuse: "
        f"{'enabled' if deploy_opts.fuse_implicit_gemm_relu else 'disabled'}"
        + (f" (from deploy_cfg {deploy_cfg_path!r})" if deploy_cfg_path else " (default)")
    )
    audit_records: List[Dict[str, Any]] = []
    model = transform_onnx_int8(
        model,
        layer_scales,
        encoder_sd,
        verbose=args.verbose,
        amax_dict=amax_dict,
        override_terminal_absmax=args.pathb_terminal_absmax,
        audit_records=audit_records if args.audit_report else None,
        fp16_layer_patterns=fp16_layer_patterns,
        plugin_timing_enabled=deploy_opts.plugin_timing_enabled,
        plugin_timing_max_logs=deploy_opts.plugin_timing_max_logs,
        fuse_implicit_gemm_trailing_relu=deploy_opts.fuse_implicit_gemm_relu,
    )

    if args.audit_report:
        path = args.audit_report
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "terminal": {
                        "note": "See print log [int8-output-scale] for terminal amax source",
                    },
                    "layers": audit_records,
                },
                f,
                indent=2,
            )
        print(f"\n  [int8-audit] Wrote {len(audit_records)} layer entries to {path!r}")

    # Save (save_model avoids some TRT edge cases with large graphs).
    onnx.save_model(model, args.output)
    print(f"\nSaved INT8 ONNX: {args.output}")


if __name__ == "__main__":
    main()
