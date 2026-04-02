"""Post-process an ONNX model to replace ImplicitGemm → ImplicitGemmInt8.

Path B approach: The standard Autoware ONNX export (via torch.onnx.export +
sparse_functional.py symbolic methods) produces autoware::ImplicitGemm nodes
with 5 inputs. This script enriches them to autoware::ImplicitGemmInt8 nodes
with 7 inputs (+ channel_scale + bias_scaled) and INT8 scale attributes.

Usage::

    python -m deployment.projects.bevfusion.export.sparse_int8_onnx_transform \\
        --onnx work_dirs/bevfusion/sparse_encoder.onnx \\
        --checkpoint work_dirs/bevfusion/bevfusion_epoch_30_ptq_sparse_only.pth \\
        --config projects/BEVFusion/configs/.../bevfusion_..._fx.py \\
        --output work_dirs/bevfusion/sparse_encoder_int8_pathb.onnx

The output ONNX can be loaded by TensorRT with the ImplicitGemmInt8Plugin.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch


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
        r"^(?:module\.)?(?P<prefix>pts_middle_encoder\.)"
        r"(?P<stem>.+?)\._(?P<kind>input|weight)_quantizer\._amax$"
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
            print(
                f"  [int8-output-scale] {stem}: terminal layer → {terminal_src} ({hint})"
            )
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
        re.match(r"^encoder_layer\d+\.\d+\.\d+$", tail)
        and "conv" not in tail
        and "downsample" not in tail
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


def _conv_weight_shape_from_state_dict(
    encoder_sd: Dict[str, torch.Tensor], stem: str
) -> Optional[Tuple[int, int]]:
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
            f"(legacy axis=4). INT8 Path B needs **C_out={c_out}** per-output scales. "
            + _LEGACY_W_AMAX_PTQ
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

    if (
        encoder_sd
        and c_out is not None
        and c_in is not None
    ):
        ok_sd = [
            s
            for s in candidates
            if _conv_weight_shape_from_state_dict(encoder_sd, s) == (c_out, c_in)
        ]
        if len(ok_sd) == 1:
            return ok_sd[0]
        if len(ok_sd) > 1:
            return max(ok_sd, key=len)
        if verbose and candidates:
            print(
                "  [debug-match] no stem with state_dict weight (C_out,C_in) matching ONNX filter; "
                "falling back to _amax heuristic"
            )

    if c_out is not None:
        ok = [
            s
            for s in candidates
            if _weight_amax_matches_cout_layout(
                layer_scales.get(s, {}).get("weight_amax"), c_out, c_in
            )
        ]
        if len(ok) == 1:
            return ok[0]
        if len(ok) > 1:
            return max(ok, key=len)
        if verbose:
            print(
                f"  [debug-match] no stem with _amax compatible with C_out={c_out} C_in={c_in}."
            )
        return None

    return max(candidates, key=len)


def _get_initializer_data(model: onnx.ModelProto, name: str) -> Optional[np.ndarray]:
    """Get numpy data from an ONNX initializer by name."""
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def _onnx_node_text_blob(node: onnx.NodeProto) -> str:
    parts = [node.name or ""]
    parts.extend(node.input)
    parts.extend(node.output)
    return " ".join(parts).lower().replace("\\", "/")


def _implicit_gemm_is_conv_out(node: onnx.NodeProto) -> bool:
    """PTQ keeps ``conv_out`` FP32; ONNX node must stay ``ImplicitGemm``, not Int8."""
    return "conv_out" in _onnx_node_text_blob(node)


def _implicit_gemm_filter_c_out(model: onnx.ModelProto, node: onnx.NodeProto) -> Optional[int]:
    c_out, _ = _implicit_gemm_filter_c_out_c_in(model, node)
    return c_out


def _normalize_onnx_attr_field_name(name: str) -> str:
    """Map ONNX attribute storage name to logical name.

    Some exporters store ``output_scale_f`` as the literal ``AttributeProto.name``; TensorRT's
    ONNX→plugin mapper expects ``output_scale`` (matching ``PluginField`` names). Netron often
    shows ``input_scale_f`` meaning *float* attribute ``input_scale`` — but if the protobuf name
    actually ends with ``_f`` / ``_i``, strip it so we merge with standard keys.
    """
    for suf in ("_f", "_i", "_s", "_l"):
        if name.endswith(suf) and len(name) > len(suf):
            return name[: -len(suf)]
    return name


def _implicit_gemm_attrs_from_node(node: onnx.NodeProto) -> Dict[str, object]:
    """Read ``ImplicitGemm`` attributes into a dict keyed by normalized names (no ``_f``/``_i``)."""
    out: Dict[str, object] = {}
    for attr in node.attribute:
        base = _normalize_onnx_attr_field_name(attr.name)
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

        output_amax, oa_tag = _resolve_int8_output_amax(
            stem, layer_scales, topo_stems, term_np, term_src, verbose
        )

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
        w_for_scale = _weight_amax_per_cout_vector(
            weight_amax, sh_w[0], sh_w[1], stem
        )

        scale_info[stem] = _compute_int8_scales(
            input_amax, w_for_scale, output_amax, bias
        )
        if verbose:
            print(
                f"  [debug-scale] {stem}: w_amax.shape={np.shape(weight_amax)} "
                f"output_amax_tag={oa_tag!r} output_scale={scale_info[stem]['output_scale']:.6f}"
            )

    n_expected_int8 = sum(
        1
        for n in graph.node
        if n.op_type == "ImplicitGemm"
        and n.domain == "autoware"
        and not _implicit_gemm_is_conv_out(n)
    )

    # Replace nodes.
    new_nodes = []
    transform_count = 0

    for node in graph.node:
        if node.op_type != "ImplicitGemm" or node.domain != "autoware":
            new_nodes.append(node)
            continue

        if _implicit_gemm_is_conv_out(node):
            print(
                "  [int8] Keep FP32 ImplicitGemm for conv_out (PTQ): "
                f"name={node.name!r}"
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

        # Create ONNX initializers for channel_scale and bias_scaled.
        cs_name, bs_name = _safe_trt_scale_names(stem, occupied_names)

        cs_init = numpy_helper.from_array(si["channel_scale"], name=cs_name)
        bs_init = numpy_helper.from_array(si["bias_scaled"], name=bs_name)
        graph.initializer.append(cs_init)
        graph.initializer.append(bs_init)

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

        int8_node = helper.make_node(
            "ImplicitGemmInt8",
            inputs=list(node.input) + [cs_name, bs_name],
            outputs=list(node.output),
            domain="autoware",
            name=f"{node.name}_int8" if node.name else f"ImplicitGemmInt8_{transform_count}",
        )
        _append_implicit_gemm_int8_plugin_attributes(int8_node, attrs)

        new_nodes.append(int8_node)
        transform_count += 1
        print(f"  [int8] {stem}: input_scale={si['input_scale']:.6f} "
              f"output_scale={si['output_scale']:.6f} "
              f"channel_scale_shape={si['channel_scale'].shape}")

    # Replace nodes in graph.
    del graph.node[:]
    graph.node.extend(new_nodes)

    if transform_count != n_expected_int8:
        raise RuntimeError(
            f"Path B ONNX transform: expected {n_expected_int8} ImplicitGemm → ImplicitGemmInt8 "
            f"replacements (excluding conv_out), got {transform_count}. "
            "Graph/calibration mismatch."
        )

    print(f"\nTransformed {transform_count} ImplicitGemm → ImplicitGemmInt8 nodes")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Transform ONNX ImplicitGemm nodes to ImplicitGemmInt8"
    )
    parser.add_argument("--onnx", required=True, help="Input ONNX model path")
    parser.add_argument(
        "--checkpoint", required=True,
        help="PTQ checkpoint with NVIDIA _amax calibration values",
    )
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument(
        "--config", default=None,
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

    # Transform.
    print(f"\nTransforming ImplicitGemm → ImplicitGemmInt8...")
    model = transform_onnx_int8(
        model,
        layer_scales,
        encoder_sd,
        verbose=args.verbose,
        amax_dict=amax_dict,
        override_terminal_absmax=args.pathb_terminal_absmax,
    )

    # Save (save_model avoids some TRT edge cases with large graphs).
    onnx.save_model(model, args.output)
    print(f"\nSaved INT8 ONNX: {args.output}")


if __name__ == "__main__":
    main()
