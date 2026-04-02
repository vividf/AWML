"""P2: Compare sparse first-layer (or chosen stem) conv weights — ONNX vs PyTorch checkpoint.

Resolves wrong-export / stale-ONNX suspicions before deep-diving plugin epilogues.

Example::

    python -m deployment.projects.bevfusion.debug.compare_sparse_onnx_pt_weights \\
        --onnx /path/to/sparse_int8.onnx \\
        --checkpoint /path/to/ptq.pth \\
        --stem conv_input.0

Requires: ``onnx``, ``numpy``, ``torch``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Optional, Tuple

import numpy as np


def _get_initializer_array(model: "onnx.ModelProto", name: str) -> Optional[np.ndarray]:
    from onnx import numpy_helper

    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def _find_int8_node_for_stem(model: "onnx.ModelProto", stem: str):
    """Pick ``ImplicitGemmInt8`` whose name or inputs reference ``stem`` (e.g. ``conv_input.0``)."""
    stem_l = stem.lower()
    alt = stem.replace(".", "/").lower()
    for node in model.graph.node:
        if node.op_type != "ImplicitGemmInt8":
            continue
        blob = ((node.name or "") + " " + " ".join(node.input)).lower()
        if stem_l in blob or alt in blob:
            return node
    return None


def _load_encoder_weights(ckpt_path: str) -> Dict[str, np.ndarray]:
    import torch

    obj = torch.load(ckpt_path, map_location="cpu")
    sd = obj
    if isinstance(obj, dict):
        if "state_dict" in obj:
            sd = obj["state_dict"]
        elif "model" in obj and hasattr(obj["model"], "state_dict"):
            sd = obj["model"].state_dict()
    if not isinstance(sd, dict):
        raise TypeError(f"Could not extract state_dict from {ckpt_path!r}")
    out: Dict[str, np.ndarray] = {}
    for k, v in sd.items():
        if hasattr(v, "detach"):
            out[k] = v.detach().cpu().float().numpy()
        else:
            out[k] = np.asarray(v, dtype=np.float32)
    return out


def _pt_weight_key(stem: str) -> Tuple[str, ...]:
    base = stem if not stem.startswith("pts_middle_encoder.") else stem.split(".", 2)[-1]
    return (
        f"pts_middle_encoder.{base}.weight",
        f"module.pts_middle_encoder.{base}.weight",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, help="Sparse ONNX after ImplicitGemmInt8 transform")
    parser.add_argument("--checkpoint", required=True, help="PTQ .pth / checkpoint with conv weights")
    parser.add_argument(
        "--stem",
        default="conv_input.0",
        help="PTS middle encoder stem (default: conv_input.0)",
    )
    parser.add_argument(
        "--filter-input",
        default="",
        help="Optional: ONNX initializer name for ImplicitGemm filter (input[1]); "
        "default: resolve from ImplicitGemmInt8 node matching --stem",
    )
    args = parser.parse_args()

    try:
        import onnx
    except ImportError:
        print("install onnx: pip install onnx", file=sys.stderr)
        return 2

    model = onnx.load(args.onnx)
    filter_name = args.filter_input.strip()
    if not filter_name:
        node = _find_int8_node_for_stem(model, args.stem)
        if node is None:
            print(
                f"No ImplicitGemmInt8 node matched stem {args.stem!r}. " "Pass --filter-input explicitly.",
                file=sys.stderr,
            )
            return 1
        if len(node.input) < 2:
            print(f"Node {node.name!r} has no filter input[1].", file=sys.stderr)
            return 1
        filter_name = node.input[1]
        print(f"[compare] ONNX node={node.name!r} filter initializer={filter_name!r}")

    w_onnx = _get_initializer_array(model, filter_name)
    if w_onnx is None:
        print(f"Initializer {filter_name!r} not found in ONNX graph.", file=sys.stderr)
        return 1

    sd = _load_encoder_weights(args.checkpoint)
    w_pt = None
    pt_key = None
    for k in _pt_weight_key(args.stem):
        if k in sd:
            w_pt = sd[k]
            pt_key = k
            break
    if w_pt is None:
        print(
            f"No PT weight for stem {args.stem!r}; tried {_pt_weight_key(args.stem)}",
            file=sys.stderr,
        )
        return 1

    a = np.asarray(w_onnx, dtype=np.float32).reshape(-1)
    b = np.asarray(w_pt, dtype=np.float32).reshape(-1)
    if a.size != b.size:
        print(
            f"[compare] shape mismatch ONNX {tuple(w_onnx.shape)} vs PT {tuple(w_pt.shape)}",
            file=sys.stderr,
        )
        return 1
    a = a.reshape(w_pt.shape)
    b = b.reshape(w_pt.shape)
    diff = np.abs(a - b)
    print(f"[compare] PT key={pt_key!r} shape={tuple(b.shape)}")
    print(
        f"[compare] max_abs_diff={float(diff.max()):.6g} mean_abs_diff={float(diff.mean()):.6g} "
        f"rel_mean={float((diff / (np.abs(b) + 1e-8)).mean()):.6g}"
    )
    if np.allclose(a, b, rtol=1e-3, atol=1e-5):
        print("[compare] OK: weights match within rtol=1e-3 atol=1e-5")
        return 0
    print("[compare] MISMATCH: ONNX vs checkpoint weights differ — re-export or wrong stem match.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
