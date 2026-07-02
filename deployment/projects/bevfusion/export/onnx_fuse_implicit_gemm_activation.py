"""Fuse post-spconv activation into ``autoware`` ImplicitGemm plugin ``act_type``.

TensorRT does not fuse standard ONNX ``Relu`` with custom ops, so we fold
``ImplicitGemm → Relu`` by setting ``act_type=kReLU`` on the producer node and
removing the standalone ``Relu`` node.
"""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Set

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _normalize_attr(name: str) -> str:
    """Strip ONNX type suffix (``_f``, ``_i``, ``_s``, ``_l``) from an attribute name."""
    for suf in ("_f", "_i", "_s", "_l"):
        if name.endswith(suf) and len(name) > len(suf):
            return name[: -len(suf)]
    return name


def _try_get_constant_numpy(
    graph: onnx.GraphProto,
    name: str,
    init_map: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    """Return the constant numpy array for tensor ``name``, or ``None`` if not a constant.

    Checks ``init_map`` (pre-built from ``graph.initializer``) first, then searches
    for a ``Constant`` node that produces ``name``.
    """
    if name in init_map:
        return init_map[name]
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        if name in node.output:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR:
                    return numpy_helper.to_array(attr.t)
    return None


def _read_implicit_gemm_attrs(node: onnx.NodeProto) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for attr in node.attribute:
        base = _normalize_attr(attr.name)
        if attr.type == onnx.AttributeProto.INT:
            out[base] = int(attr.i)
        elif attr.type == onnx.AttributeProto.FLOAT:
            out[base] = float(attr.f)
    return out


def _replace_tensor_name(graph: onnx.GraphProto, old: str, new: str) -> None:
    if old == new:
        return
    for n in graph.node:
        for i, inp in enumerate(n.input):
            if inp == old:
                n.input[i] = new
    for out in graph.output:
        if out.name == old:
            out.name = new
    for vi in graph.value_info:
        if vi.name == old:
            vi.name = new


def _set_implicit_gemm_act_type(node: onnx.NodeProto, act_type: int) -> None:
    kept = [a for a in node.attribute if _normalize_attr(a.name) != "act_type"]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute("act_type", int(act_type)))


def fuse_autoware_implicit_gemm_trailing_relu(model: onnx.ModelProto) -> int:
    """Set ``act_type`` = kReLU (1) on ImplicitGemm and remove redundant ``Relu`` nodes."""

    graph = model.graph
    n_removed = 0

    remove_idx: Set[int] = set()

    for ri, relu in enumerate(graph.node):
        if ri in remove_idx:
            continue
        if relu.op_type != "Relu":
            continue
        if relu.domain not in ("", "ai.onnx"):
            continue
        if len(relu.input) < 1 or not relu.input[0]:
            continue
        if len(relu.output) < 1 or not relu.output[0]:
            continue

        users: DefaultDict[str, List[int]] = defaultdict(list)
        for ni, n in enumerate(graph.node):
            if ni in remove_idx:
                continue
            for inp in n.input:
                if inp:
                    users[inp].append(ni)

        g_out = relu.input[0]
        r_out = relu.output[0]

        if len(users.get(g_out, [])) != 1:
            continue

        producer_i: int | None = None
        producer: onnx.NodeProto | None = None
        for ni, n in enumerate(graph.node):
            if g_out in n.output:
                producer_i = ni
                producer = n
                break
        if producer is None or producer_i is None:
            continue
        if producer.op_type != "ImplicitGemm" or producer.domain != "autoware":
            continue
        if producer_i in remove_idx:
            continue

        attrs = _read_implicit_gemm_attrs(producer)
        cur = int(attrs.get("act_type", 0) or 0)
        if cur not in (0, 1):
            continue

        _set_implicit_gemm_act_type(producer, 1)
        _replace_tensor_name(graph, r_out, g_out)
        remove_idx.add(ri)
        n_removed += 1

    if not remove_idx:
        return 0

    new_nodes: List[onnx.NodeProto] = []
    for ni, n in enumerate(graph.node):
        if ni not in remove_idx:
            new_nodes.append(n)
    del graph.node[:]
    graph.node.extend(new_nodes)
    return n_removed
