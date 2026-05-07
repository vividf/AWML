"""Fuse post-spconv activations into ``autoware`` ImplicitGemm plugins (``act_type``).

TensorRT does not fuse standard ONNX ``Relu`` with custom ops. We support:

1. **FP16 (pre-INT8):** ``ImplicitGemm → Add(const) → Relu`` → append 6th input (bias) to the plugin,
   ``act_type=kReLU``, remove ``Add``/``Relu`` (needs Autoware ``ImplicitGemm`` plugin with optional bias).

2. **FP16:** ``ImplicitGemm → Relu`` (no Add) → merge Relu into ``act_type``.

3. **Path-B INT8 ONNX:** ``ImplicitGemmInt8 → Add → Relu`` when ``Add`` adds a **constant** bias
   (initializer or ``Constant``) and the **other** branch is not a residual shortcut. The exported
   ``+ bias`` from PyTorch is folded into the plugin ``bias_scaled`` tensor
   (``bias_scaled += bias_fp / output_scale``) and ``Relu`` is merged into ``act_type``.

Residual ``Add`` (two dynamic feature tensors) is **not** fused.

Returns the number of Relu nodes removed (each fusion removes one Relu).
"""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Set

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _normalize_attr(name: str) -> str:
    for suf in ("_f", "_i", "_s", "_l"):
        if name.endswith(suf) and len(name) > len(suf):
            return name[: -len(suf)]
    return name


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


def _try_get_constant_numpy(
    graph: onnx.GraphProto, tensor_name: str, init_map: Dict[str, np.ndarray]
) -> Optional[np.ndarray]:
    if tensor_name in init_map:
        return init_map[tensor_name]
    for n in graph.node:
        if n.op_type != "Constant" or not n.output or n.output[0] != tensor_name:
            continue
        for a in n.attribute:
            if a.name == "value" and a.t.data_type:
                return numpy_helper.to_array(a.t)
    return None


def _replace_initializer_array(
    graph: onnx.GraphProto,
    name: str,
    arr: np.ndarray,
    init_map: Dict[str, np.ndarray],
) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    init_map[name] = arr
    new_inits: List[onnx.TensorProto] = []
    replaced = False
    for init in graph.initializer:
        if init.name == name:
            new_inits.append(numpy_helper.from_array(arr, name=name))
            replaced = True
        else:
            new_inits.append(init)
    if not replaced:
        raise ValueError(f"initializer {name!r} not found for bias merge")
    del graph.initializer[:]
    graph.initializer.extend(new_inits)


def _onnx_elem_type_from_np(arr: np.ndarray) -> int:
    if arr.dtype == np.float16:
        return int(TensorProto.FLOAT16)
    if arr.dtype == np.float32:
        return int(TensorProto.FLOAT)
    return int(TensorProto.FLOAT)


def _ensure_graph_input_for_initializer(graph: onnx.GraphProto, name: str, shape: List[int], elem_type: int) -> None:
    """TensorRT ONNX parser often requires ``graph.input`` entries for tensors fed to plugins."""
    if any(i.name == name for i in graph.input):
        return
    graph.input.append(helper.make_tensor_value_info(name, elem_type, shape))


def fuse_autoware_implicit_gemm_fp16_add_relu(model: onnx.ModelProto) -> int:
    """Fold ``ImplicitGemm → Add(const) → Relu`` into 6-input ``ImplicitGemm`` + ``act_type`` (FP16)."""

    graph = model.graph
    init_map: Dict[str, np.ndarray] = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    total = 0

    while True:
        fused = False
        for ai, add in enumerate(graph.node):
            if add.op_type != "Add" or add.domain not in ("", "ai.onnx"):
                continue
            if len(add.input) < 2 or not add.output:
                continue
            x0, x1 = add.input[0], add.input[1]
            add_out = add.output[0]

            relu_idx: Optional[int] = None
            for ri, n in enumerate(graph.node):
                if n.op_type == "Relu" and n.domain in ("", "ai.onnx") and n.input and n.input[0] == add_out:
                    relu_idx = ri
                    break
            if relu_idx is None:
                continue

            users_add = [ni for ni, n in enumerate(graph.node) for inp in n.input if inp == add_out]
            if len(users_add) != 1:
                continue

            relu = graph.node[relu_idx]
            if not relu.output:
                continue
            relu_out = relu.output[0]

            for gemm_in_name, other_name in ((x0, x1), (x1, x0)):
                gemm_node: Optional[onnx.NodeProto] = None
                for n in graph.node:
                    if gemm_in_name in n.output:
                        gemm_node = n
                        break
                if gemm_node is None:
                    continue
                if gemm_node.op_type != "ImplicitGemm" or gemm_node.domain != "autoware":
                    continue
                if not gemm_node.output or gemm_node.output[0] != gemm_in_name:
                    continue
                if len(gemm_node.input) != 5:
                    continue

                users_gemm = [ni for ni, n in enumerate(graph.node) for inp in n.input if inp == gemm_in_name]
                if len(users_gemm) != 1:
                    continue

                extra_bias = _try_get_constant_numpy(graph, other_name, init_map)
                if extra_bias is None:
                    continue

                attrs = _read_implicit_gemm_attrs(gemm_node)
                cur_act = int(attrs.get("act_type", 0) or 0)
                if cur_act not in (0, 1):
                    continue

                b_np = np.asarray(extra_bias)
                c_out = int(b_np.reshape(-1).shape[0])
                elem_type = _onnx_elem_type_from_np(b_np)
                _ensure_graph_input_for_initializer(graph, other_name, [c_out], elem_type)

                new_inputs = list(gemm_node.input) + [other_name]
                del gemm_node.input[:]
                gemm_node.input.extend(new_inputs)

                _set_implicit_gemm_act_type(gemm_node, 1)
                _replace_tensor_name(graph, relu_out, gemm_in_name)

                remove_idx = {ai, relu_idx}
                new_nodes = [n for i, n in enumerate(graph.node) if i not in remove_idx]
                del graph.node[:]
                graph.node.extend(new_nodes)

                total += 1
                fused = True
                break

            if fused:
                break

        if not fused:
            break

    return total


def fuse_autoware_implicit_gemm_int8_add_relu(model: onnx.ModelProto) -> int:
    """Fold ``ImplicitGemmInt8 → Add(const) → Relu`` into plugin ``bias_scaled`` + ``act_type``."""

    graph = model.graph
    init_map: Dict[str, np.ndarray] = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    total = 0

    while True:
        fused = False
        for ai, add in enumerate(graph.node):
            if add.op_type != "Add" or add.domain not in ("", "ai.onnx"):
                continue
            if len(add.input) < 2 or not add.output:
                continue
            x0, x1 = add.input[0], add.input[1]
            add_out = add.output[0]

            relu_idx: Optional[int] = None
            for ri, n in enumerate(graph.node):
                if n.op_type == "Relu" and n.domain in ("", "ai.onnx") and n.input and n.input[0] == add_out:
                    relu_idx = ri
                    break
            if relu_idx is None:
                continue

            users_add = [ni for ni, n in enumerate(graph.node) for inp in n.input if inp == add_out]
            if len(users_add) != 1:
                continue

            relu = graph.node[relu_idx]
            if not relu.output:
                continue
            relu_out = relu.output[0]

            for gemm_in_name, other_name in ((x0, x1), (x1, x0)):
                gemm_node: Optional[onnx.NodeProto] = None
                for n in graph.node:
                    if gemm_in_name in n.output:
                        gemm_node = n
                        break
                if gemm_node is None:
                    continue
                if gemm_node.op_type != "ImplicitGemmInt8" or gemm_node.domain != "autoware":
                    continue
                if not gemm_node.output or gemm_node.output[0] != gemm_in_name:
                    continue

                users_gemm = [ni for ni, n in enumerate(graph.node) for inp in n.input if inp == gemm_in_name]
                if len(users_gemm) != 1:
                    continue

                extra_bias = _try_get_constant_numpy(graph, other_name, init_map)
                if extra_bias is None:
                    continue

                if len(gemm_node.input) < 7:
                    continue
                bs_name = gemm_node.input[6]
                if bs_name not in init_map:
                    continue

                attrs = _read_implicit_gemm_attrs(gemm_node)
                cur_act = int(attrs.get("act_type", 0) or 0)
                if cur_act not in (0, 1):
                    continue

                out_scale = float(attrs.get("output_scale", 1.0) or 1.0)
                if out_scale == 0.0:
                    continue

                bs = np.asarray(init_map[bs_name], dtype=np.float32).reshape(-1)
                eb = np.asarray(extra_bias, dtype=np.float32)
                try:
                    eb_b = np.broadcast_to(eb, bs.shape).reshape(-1)
                except ValueError:
                    continue

                # Path-B stores bias_scaled = bias_pt / output_scale. ONNX Add often repeats the same
                # PyTorch bias as FP; if it matches bs * output_scale, folding would double-count — keep bs.
                recon_pt = bs * out_scale
                if np.allclose(eb_b, recon_pt, rtol=1e-3, atol=1e-4):
                    merged = bs
                else:
                    merged = bs + eb_b / out_scale
                _replace_initializer_array(graph, bs_name, merged, init_map)

                _set_implicit_gemm_act_type(gemm_node, 1)
                _replace_tensor_name(graph, relu_out, gemm_in_name)

                remove_idx = {ai, relu_idx}
                new_nodes = [n for i, n in enumerate(graph.node) if i not in remove_idx]
                del graph.node[:]
                graph.node.extend(new_nodes)

                total += 1
                fused = True
                break

            if fused:
                break

        if not fused:
            break

    return total


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
