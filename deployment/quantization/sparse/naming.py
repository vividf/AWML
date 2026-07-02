# Copyright (c) OpenMMLab. All rights reserved.
"""Sparse-encoder layer-name ordering helpers (SECOND-style spconv encoders).

These operate purely on module/checkpoint *stem* strings (e.g. ``conv_input.0``,
``encoder_layer1.0.conv1``) and encode the forward-ish ordering of a SECOND-style sparse
encoder — shared by both the sparse INT8 quant primitives (:mod:`.spconv_int8`) and the
BEVFusion ``ImplicitGemmInt8`` ONNX transform. Kept in the framework so neither side has to
import the other (the transform lives in the BEVFusion project; framework code must not import
from a project).
"""

from __future__ import annotations

import re
from typing import List, Tuple


def tail_without_encoder_layers(stem: str) -> str:
    """Strip a leading ``encoder_layers.`` prefix (checkpoint vs. module-path naming)."""
    if stem.startswith("encoder_layers."):
        return stem[len("encoder_layers.") :]
    return stem


def topologically_sorted_sparse_stems(stems: List[str]) -> List[str]:
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
        tail = tail_without_encoder_layers(s)
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
