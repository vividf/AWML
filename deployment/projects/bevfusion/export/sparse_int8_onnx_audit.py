"""Read-only audit of Path-B ``ImplicitGemmInt8`` ONNX (plugin scales + channel tensors).

Use after ``sparse_int8_onnx_transform`` to verify how TensorRT will see each layer without
re-running PTQ.  Does not need a checkpoint.

Example::

    python -m deployment.projects.bevfusion.export.sparse_int8_onnx_audit \\
        --onnx work_dirs/bevfusion_split_int8_deployment/onnx/bevfusion_sparse_int8.onnx
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import onnx

from deployment.projects.bevfusion.export.sparse_int8_onnx_transform import (
    _get_initializer_data,
    _implicit_gemm_attrs_from_node,
)


def audit_implicit_gemm_int8_onnx(model: onnx.ModelProto, *, stream=sys.stdout) -> int:
    """Print one line per ``autoware::ImplicitGemmInt8`` node. Returns node count."""
    graph = model.graph
    count = 0
    for node in graph.node:
        if node.op_type != "ImplicitGemmInt8" or node.domain != "autoware":
            continue
        count += 1
        attrs = _implicit_gemm_attrs_from_node(node)
        in_s = float(attrs.get("input_scale", 0.0) or 0.0)
        out_s = float(attrs.get("output_scale", 0.0) or 0.0)
        cs_name = node.input[5] if len(node.input) > 5 else ""
        bs_name = node.input[6] if len(node.input) > 6 else ""
        cs = _get_initializer_data(model, cs_name) if cs_name else None
        if cs is not None and cs.size:
            csf = cs.reshape(-1).astype(np.float64)
            cs_line = f"cs_len={csf.size} cs_min={csf.min():.6g} cs_max={csf.max():.6g} " f"cs_mean={csf.mean():.6g}"
        else:
            cs_line = "channel_scale=(missing or not initializer)"
        print(
            f"[ImplicitGemmInt8] name={node.name!r} in_scale={in_s:.6g} out_scale={out_s:.6g} "
            f"{cs_line} cs_tensor={cs_name!r} bias_scaled_tensor={bs_name!r}",
            file=stream,
        )
    if count == 0:
        print(
            "No autoware::ImplicitGemmInt8 nodes found (expect Path-B INT8 ONNX).",
            file=stream,
        )
    else:
        print(f"\nTotal ImplicitGemmInt8 nodes: {count}", file=stream)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ImplicitGemmInt8 scales in ONNX")
    parser.add_argument("--onnx", required=True, help="INT8 sparse ONNX path")
    args = parser.parse_args()
    model = onnx.load(args.onnx)
    audit_implicit_gemm_int8_onnx(model)


if __name__ == "__main__":
    main()
