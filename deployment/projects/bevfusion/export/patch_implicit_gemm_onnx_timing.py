#!/usr/bin/env python3
"""Bake ``timing_enabled`` / ``timing_max_logs`` into FP16 ``autoware::ImplicitGemm`` ONNX nodes.

TensorRT reads these as TensorRT plugin fields for ``ImplicitGemmPlugin`` (Autoware
``autoware_tensorrt_plugins``). Use after exporting ``bevfusion_sparse.onnx``, before building
the FP16 sparse engine.

Example::

    python -m deployment.projects.bevfusion.export.patch_implicit_gemm_onnx_timing \\
        --onnx work_dirs/.../bevfusion_sparse.onnx \\
        --output work_dirs/.../bevfusion_sparse_timing.onnx \\
        --deploy-cfg deployment/projects/bevfusion/config/deploy_config_split_fp16_no_sort.py

Deploy keys (optional; defaults below)::

    implicit_gemm_plugin_timing = False
    implicit_gemm_plugin_timing_max_logs = 1000
"""

from __future__ import annotations

import argparse
import os
import sys

import onnx
from onnx import helper


def _strip_timing_attrs(node: onnx.NodeProto) -> None:
    keep = [a for a in node.attribute if a.name not in ("timing_enabled", "timing_max_logs")]
    del node.attribute[:]
    node.attribute.extend(keep)


def patch_implicit_gemm_timing(
    model: onnx.ModelProto,
    *,
    timing_enabled: bool,
    timing_max_logs: int,
) -> onnx.ModelProto:
    """Mutate ``model`` in place; return same instance with timing attrs on each ImplicitGemm."""
    graph = model.graph
    for node in graph.node:
        if node.op_type != "ImplicitGemm" or node.domain != "autoware":
            continue
        _strip_timing_attrs(node)
        node.attribute.append(
            helper.make_attribute("timing_enabled", int(bool(timing_enabled))),
        )
        node.attribute.append(helper.make_attribute("timing_max_logs", int(timing_max_logs)))
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Add ImplicitGemm plugin timing ONNX attributes")
    parser.add_argument("--onnx", required=True, help="Input sparse ONNX path")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument(
        "--deploy-cfg",
        default=None,
        help="Deploy config .py with implicit_gemm_plugin_timing keys",
    )
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Force timing on (overrides deploy_cfg implicit_gemm_plugin_timing)",
    )
    parser.add_argument(
        "--disable",
        action="store_true",
        help="Force timing off",
    )
    parser.add_argument(
        "--timing-max-logs",
        type=int,
        default=None,
        help="Override max stderr timing lines",
    )
    args = parser.parse_args()

    enabled = False
    max_logs = 1000
    if args.deploy_cfg:
        try:
            from mmengine import Config  # type: ignore
        except ImportError as e:
            print("patch_implicit_gemm_onnx_timing: mmengine required for --deploy-cfg", file=sys.stderr)
            raise SystemExit(1) from e
        cfg = Config.fromfile(args.deploy_cfg)
        enabled = bool(cfg.get("implicit_gemm_plugin_timing", False))
        max_logs = int(cfg.get("implicit_gemm_plugin_timing_max_logs", 1000))

    if args.enable:
        enabled = True
    if args.disable:
        enabled = False
    if args.timing_max_logs is not None:
        max_logs = int(args.timing_max_logs)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    print(f"Loading {args.onnx}")
    model = onnx.load(args.onnx)
    patched = patch_implicit_gemm_timing(model, timing_enabled=enabled, timing_max_logs=max_logs)
    onnx.save_model(patched, args.output)
    print(f"Saved {args.output} (implicit_gemm_plugin_timing={enabled} max_logs={max_logs})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
