"""Utilities to make Q/DQ-heavy ONNX graphs easier to inspect in Netron.

This module intentionally does not change model semantics:
- Convert Constant->initializer only for Q/DQ scale/zero-point inputs.
- Annotate QuantizeLinear / DequantizeLinear node names with scale/zp summaries.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Optional, Set, Tuple

import onnx
from onnx import numpy_helper

_QDQ_OPS = {"QuantizeLinear", "DequantizeLinear"}


def _build_consumers(graph: onnx.GraphProto) -> Dict[str, List[onnx.NodeProto]]:
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    return consumers


def _collect_constant_tensor_by_output(graph: onnx.GraphProto) -> Dict[str, onnx.TensorProto]:
    out: Dict[str, onnx.TensorProto] = {}
    for node in graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            value = onnx.helper.get_attribute_value(attr)
            if isinstance(value, onnx.TensorProto):
                tensor = copy.deepcopy(value)
                tensor.name = node.output[0]
                out[node.output[0]] = tensor
            break
    return out


def _format_tensor_preview(tensor: onnx.TensorProto, max_items: int = 3) -> str:
    arr = numpy_helper.to_array(tensor).reshape(-1)
    if arr.size == 0:
        return "empty"
    vals = arr[:max_items]
    if str(arr.dtype).startswith(("int", "uint", "bool")):
        head = ",".join(str(int(v)) for v in vals.tolist())
    else:
        head = ",".join(f"{float(v):.6g}" for v in vals.tolist())
    if arr.size > max_items:
        return f"{head},...({arr.size})"
    return head


def _promote_qdq_constants_to_initializers(
    model: onnx.ModelProto,
) -> Tuple[int, Set[str], Dict[str, onnx.TensorProto]]:
    graph = model.graph
    init_by_name: Dict[str, onnx.TensorProto] = {init.name: init for init in graph.initializer}
    const_by_output = _collect_constant_tensor_by_output(graph)
    consumers = _build_consumers(graph)

    promoted = 0
    removable_constant_outputs: Set[str] = set()

    for node in graph.node:
        if node.op_type not in _QDQ_OPS:
            continue
        for input_idx in (1, 2):  # scale, zero_point
            if len(node.input) <= input_idx:
                continue
            name = node.input[input_idx]
            if not name:
                continue
            if name in init_by_name:
                continue
            tensor = const_by_output.get(name)
            if tensor is None:
                continue

            graph.initializer.append(copy.deepcopy(tensor))
            init_by_name[name] = graph.initializer[-1]
            promoted += 1

            linked_nodes = consumers.get(name, [])
            if linked_nodes and all(n.op_type in _QDQ_OPS for n in linked_nodes):
                removable_constant_outputs.add(name)

    return promoted, removable_constant_outputs, init_by_name


def _remove_constant_nodes_by_output(graph: onnx.GraphProto, outputs: Iterable[str]) -> int:
    targets = set(outputs)
    if not targets:
        return 0
    kept: List[onnx.NodeProto] = []
    removed = 0
    for node in graph.node:
        if node.op_type == "Constant" and len(node.output) == 1 and node.output[0] in targets:
            removed += 1
            continue
        kept.append(node)
    if removed:
        del graph.node[:]
        graph.node.extend(kept)
    return removed


def _annotate_qdq_node_names(model: onnx.ModelProto, init_by_name: Dict[str, onnx.TensorProto]) -> int:
    used_names = {node.name for node in model.graph.node if node.name}
    updated = 0
    for idx, node in enumerate(model.graph.node):
        if node.op_type not in _QDQ_OPS:
            continue

        scale_preview = "?"
        zp_preview = "?"
        if len(node.input) > 1 and node.input[1] in init_by_name:
            scale_preview = _format_tensor_preview(init_by_name[node.input[1]])
        if len(node.input) > 2 and node.input[2] in init_by_name:
            zp_preview = _format_tensor_preview(init_by_name[node.input[2]])

        prefix = "Q" if node.op_type == "QuantizeLinear" else "DQ"
        base = f"{prefix}[s={scale_preview}|z={zp_preview}]"
        candidate = base
        suffix = 0
        while candidate in used_names:
            suffix += 1
            candidate = f"{base}#{suffix}"
        node.name = candidate
        used_names.add(candidate)
        updated += 1
    return updated


def make_qdq_readable(model: onnx.ModelProto) -> Tuple[int, int, int]:
    """Return (qdq_nodes_annotated, constants_promoted, constant_nodes_removed)."""
    promoted, removable_outputs, init_by_name = _promote_qdq_constants_to_initializers(model)
    removed = _remove_constant_nodes_by_output(model.graph, removable_outputs)
    updated = _annotate_qdq_node_names(model, init_by_name)
    return updated, promoted, removed
