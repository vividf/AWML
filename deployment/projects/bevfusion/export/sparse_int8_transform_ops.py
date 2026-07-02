# Copyright (c) OpenMMLab. All rights reserved.
"""Helper operations for the sparse INT8 ONNX transform.

Pure, one-directional helpers used by :func:`.sparse_int8_onnx_transform.transform_onnx_int8`:
PTQ checkpoint / amax loading, INT8 scale computation, ONNX-node ↔ PTQ-stem matching, and
ImplicitGemm(Int8) node precision/attribute editing. Extracted from the (formerly 1348-line)
transform module so the orchestrator (``transform_onnx_int8`` + CLI ``main``) stays readable; none
of these call back into the orchestrator.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

from deployment.projects.bevfusion.export.onnx_fuse_implicit_gemm_activation import _normalize_attr


def deploy_cfg_fuse_implicit_gemm_relu(deploy_cfg: Any, *, default: bool = True) -> bool:
    """Read unified ``spconv_fuse_implicit_gemm_relu`` (FP + INT8 sparse ONNX paths).

    Falls back to deprecated ``spconv_int8_fuse_implicit_gemm_relu`` when only the legacy
    key is set.
    """
    val = deploy_cfg.get("spconv_fuse_implicit_gemm_relu", None)
    if val is not None:
        return bool(val)
    legacy = deploy_cfg.get("spconv_int8_fuse_implicit_gemm_relu", None)
    if legacy is not None:
        return bool(legacy)
    return default


@dataclass
class DeployTransformOptions:
    """Optional transform knobs loaded from a BEVFusion deploy_config."""

    fp16_layer_patterns: List[str]
    fuse_implicit_gemm_relu: bool


def _load_deploy_transform_options(deploy_cfg_path: Optional[str]) -> DeployTransformOptions:
    """Read sparse-int8 transform options from deploy config.

    Supported keys:
    - spconv_int8_fp16_layers
    - spconv_fuse_implicit_gemm_relu (FP + INT8; legacy: spconv_int8_fuse_implicit_gemm_relu)
    """
    default = DeployTransformOptions(
        fp16_layer_patterns=[],
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
        fuse_implicit_gemm_relu=deploy_cfg_fuse_implicit_gemm_relu(deploy_cfg, default=True),
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


def _sparse_tail_absmax_from_state_dict(
    encoder_sd: Optional[Dict[str, torch.Tensor]],
) -> Optional[np.ndarray]:
    """``calibrate_spconv_nvidia`` saves max |features| **entering** ``conv_out`` (post tail BN/ReLU)."""
    if encoder_sd is None:
        return None
    for k in (
        "pts_middle_encoder._sparse_tail_absmax",
        "module.pts_middle_encoder._sparse_tail_absmax",
        "_sparse_tail_absmax",
    ):
        if k not in encoder_sd:
            continue
        t = encoder_sd[k]
        v = float(torch.as_tensor(t).float().reshape(-1)[0].item())
        if v > 0.0 and np.isfinite(v):
            return np.array([v], dtype=np.float32)
    return None


def _last_int8_conv_output_from_state_dict(
    encoder_sd: Optional[Dict[str, torch.Tensor]],
) -> Optional[np.ndarray]:
    """Max |features| at **output of last quantized SparseConv** (matches ImplicitGemmInt8 output)."""
    if encoder_sd is None:
        return None
    for k in (
        "pts_middle_encoder._last_int8_conv_output_absmax",
        "module.pts_middle_encoder._last_int8_conv_output_absmax",
        "_last_int8_conv_output_absmax",
    ):
        if k not in encoder_sd:
            continue
        t = encoder_sd[k]
        v = float(torch.as_tensor(t).float().reshape(-1)[0].item())
        if v > 0.0 and np.isfinite(v):
            return np.array([v], dtype=np.float32)
    return None


def _terminal_boundary_amax(
    encoder_sd: Optional[Dict[str, torch.Tensor]],
    amax_dict: Optional[Dict[str, torch.Tensor]],
    *,
    override_terminal_absmax: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """Scalar amax for the **last** INT8 layer's ``output_scale`` (that conv's linear FP output).

    Prefer ``_last_int8_conv_output_absmax`` (post-conv, pre tail BN/ReLU).  The older
    ``_sparse_tail_absmax`` tracks the tensor entering ``conv_out`` and can be **too large**
    for ``ImplicitGemmInt8.output_scale``, inflating TRT ``lidar_bev``.
    """
    if override_terminal_absmax is not None:
        v = float(override_terminal_absmax)
        if v > 0.0 and np.isfinite(v):
            return np.array([v], dtype=np.float32), "cli_override"
    li = _last_int8_conv_output_from_state_dict(encoder_sd)
    if li is not None:
        return li, "last_int8_conv_output_absmax"
    pb = _sparse_tail_absmax_from_state_dict(encoder_sd)
    if pb is not None:
        return pb, "sparse_tail_absmax_legacy"
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
            f"Sparse INT8: stem {stem!r} needs output_scale from successor {nxt!r} "
            f"input_amax, but layer_scales[{nxt!r}]['input_amax'] is missing. "
            "Re-run sparse PTQ or fix checkpoint _amax keys."
        )

    if terminal_boundary_np is not None:
        if verbose:
            hint = {
                "last_int8_conv_output_absmax": "last INT8 SparseConv output (pre tail BN/ReLU)",
                "sparse_tail_absmax_legacy": "conv_out input (legacy; can over-scale last Gemm)",
                "cli_override": "--terminal-absmax",
                "conv_out_input_quantizer": "conv_out._input_quantizer._amax",
            }.get(terminal_src, terminal_src)
            print(f"  [int8-output-scale] {stem}: terminal layer → {terminal_src} ({hint})")
        return terminal_boundary_np, terminal_src

    raise ValueError(
        f"Sparse INT8: no output_amax for stem {stem!r} (successor {nxt!r} is outside "
        "quantized stems, e.g. conv_out). Checkpoint must contain "
        "pts_middle_encoder._last_int8_conv_output_absmax (preferred), "
        "_sparse_tail_absmax, or conv_out _input_quantizer._amax; or pass "
        "--terminal-absmax to sparse_int8_onnx_transform."
    )


# Sparse-encoder stem-ordering helpers now live in the framework (shared with spconv_int8).
# Aliased to the historical private names so the call sites below stay unchanged.
from deployment.quantization.sparse.naming import (  # noqa: E402
    tail_without_encoder_layers as _tail_without_encoder_layers,
)


def _parse_encoder_stage_block(tail: str) -> Optional[Tuple[int, int]]:
    m = re.match(r"^encoder_layer(\d+)\.(\d+)\.", tail)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


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
            "Sparse INT8: output_amax is None in _compute_int8_scales — "
            "caller must resolve a valid successor input_amax or terminal boundary amax."
        )
    output_scale = float(output_amax.flatten()[0]) / 127.0

    # Fail loud on degenerate calibration. A zero / NaN / inf input or output amax silently
    # produces a zero or non-finite scale here (channel_scale divides by output_scale), which
    # ships an INT8 node that quantizes everything to ~0 → mAP collapses with no error surfaced.
    # Valid PTQ amax is always finite and > 0, so these never trip on a good checkpoint.
    if not np.isfinite(input_scale) or input_scale <= 0.0:
        raise ValueError(
            f"Sparse INT8: degenerate input_scale={input_scale} (input_amax="
            f"{float(input_amax.flatten()[0])}). Input activation calibration is invalid "
            "(zero/NaN/inf amax) — re-run sparse PTQ; a bad input_scale zeros the layer output."
        )
    if not np.isfinite(output_scale) or output_scale <= 0.0:
        raise ValueError(
            f"Sparse INT8: degenerate output_scale={output_scale} (output_amax="
            f"{float(output_amax.flatten()[0])}). Output/successor calibration is invalid "
            "(zero/NaN/inf amax) — re-run sparse PTQ; a bad output_scale yields inf/NaN scales."
        )

    channel_scale = (input_scale * w_scales) / output_scale

    if bias is not None:
        bias_scaled = bias / output_scale
    else:
        bias_scaled = np.zeros_like(channel_scale)

    # channel_scale may legitimately be 0 for a dead output channel (weight_amax==0), so reject
    # only non-finite values (NaN/inf), not zeros.
    if not np.all(np.isfinite(channel_scale)):
        raise ValueError("Sparse INT8: channel_scale has NaN/inf (degenerate weight _amax); re-run sparse PTQ.")
    if not np.all(np.isfinite(bias_scaled)):
        raise ValueError("Sparse INT8: bias_scaled has NaN/inf (degenerate bias / output_scale); re-run sparse PTQ.")

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
            f"(legacy axis=4). Sparse INT8 needs **C_out={c_out}** per-output scales. " + _LEGACY_W_AMAX_PTQ
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


def _implicit_gemm_node_precision(node: onnx.NodeProto) -> int:
    """Read the ``precision`` attribute of an ``ImplicitGemm`` node (0 = FP, 1 = INT8, default 0)."""
    for attr in node.attribute:
        if _normalize_attr(attr.name) == "precision" and attr.type == onnx.AttributeProto.INT:
            return int(attr.i)
    return 0


def _set_implicit_gemm_node_precision(node: onnx.NodeProto, value: int) -> None:
    """Set/overwrite the ``precision`` attribute on an ``ImplicitGemm`` node.

    The Autoware plugin treats a missing ``precision`` as FP (0), but we stamp it
    explicitly on FP-kept nodes too so the output ONNX is self-describing (every
    ``ImplicitGemm`` carries precision=0 for FP16 / precision=1 for INT8).
    """
    kept = [a for a in node.attribute if _normalize_attr(a.name) != "precision"]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute("precision", int(value)))


def _implicit_gemm_to_int8_path(node: onnx.NodeProto, fp16_patterns_norm: List[str]) -> bool:
    """Whether this ``autoware::ImplicitGemm`` (FP) should be converted to the INT8 path.

    INT8 nodes keep op_type ``ImplicitGemm`` but carry ``precision=1``; such already-converted
    nodes are skipped so the transform stays idempotent.
    """
    if node.op_type != "ImplicitGemm" or node.domain != "autoware":
        return False
    if _implicit_gemm_node_precision(node) == 1:
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
    """Set plugin fields exactly as the Autoware ``ImplicitGemm`` plugin INT8 path expects.

    The node keeps op_type ``ImplicitGemm`` (same op/creator as FP16); ``precision=1`` switches the
    plugin into its INT8 branch. The two extra FP32 inputs (channel_scale, bias_scaled) plus the
    scalar ``input_scale`` / ``output_scale`` attributes drive the in-plugin quantization.
    """

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
            helper.make_attribute("act_type", _i("act_type", 0)),
            # precision=1 → Autoware ImplicitGemm plugin runs its INT8 branch.
            helper.make_attribute("precision", 1),
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


def _build_scale_info(
    topo_stems: List[str],
    layer_scales: Dict[str, dict],
    encoder_sd: Dict[str, torch.Tensor],
    term_np: np.ndarray,
    term_src: str,
    verbose: bool,
) -> Dict[str, dict]:
    """Compute per-stem INT8 scales (input/output/channel_scale/bias_scaled) in topo order.

    Raises ValueError if a stem is missing input/weight ``_amax`` or its 5D weight is absent.
    """
    scale_info: Dict[str, dict] = {}
    for stem in topo_stems:
        info = layer_scales[stem]
        input_amax = info.get("input_amax")
        weight_amax = info.get("weight_amax")

        if input_amax is None or weight_amax is None:
            raise ValueError(
                f"Sparse INT8: stem {stem!r} is missing input_amax or weight_amax in layer_scales; "
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
    return scale_info


def _print_int8_census(graph: onnx.GraphProto, fp16_pattern_hits: Dict[str, int]) -> None:
    """Print the final FP16-vs-INT8 ImplicitGemm census + the fp16 keep-list match audit.

    Purely diagnostic (reads the transformed graph, writes stdout); does not mutate the model.
    """
    final_fp16_nodes: List[str] = []
    final_int8_nodes: List[str] = []
    for n in graph.node:
        if n.domain != "autoware" or n.op_type != "ImplicitGemm":
            continue
        if _implicit_gemm_node_precision(n) == 1:
            final_int8_nodes.append(n.name or "<unnamed>")
        else:
            final_fp16_nodes.append(n.name or "<unnamed>")
    print("\n  [int8-census] Final autoware::ImplicitGemm node types in output ONNX:")
    print(f"  [int8-census]   ImplicitGemm precision=0 (FP16, kept): {len(final_fp16_nodes)}")
    for nm in final_fp16_nodes:
        print(f"  [int8-census]     - {nm}")
    print(f"  [int8-census]   ImplicitGemm precision=1 (INT8 conv): {len(final_int8_nodes)}")
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
