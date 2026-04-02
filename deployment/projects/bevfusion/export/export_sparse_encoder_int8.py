"""Export BEVFusion sparse encoder to libspconv INT8 ONNX format.

Produces a custom ONNX consumable by ``libspconv``'s C++ ONNX parser
(``lidar-scn-onnx-parser.cpp``).  The sparse encoder runs on libspconv
with INT8 ``cumm`` kernels, while the dense backbone/neck/head runs on
TensorRT (FP16).

Usage::

    python -m deployment.projects.bevfusion.export.export_sparse_encoder_int8 \\
        --config projects/BEVFusion/configs/.../bevfusion_..._fx.py \\
        --checkpoint work_dirs/bevfusion/bevfusion_epoch_30_ptq_sparse_only.pth \\
        --output work_dirs/bevfusion/sparse_encoder_int8.onnx
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _build_fresh_encoder(
    config_path: str,
    device: torch.device,
) -> nn.Module:
    """Build a fresh BEVFusionSparseEncoder from config (no quantization)."""
    from mmengine.config import Config
    from mmengine.registry import MODELS, init_default_scope

    import projects.BEVFusion.bevfusion  # noqa: F401 — register modules

    init_default_scope("mmdet3d")

    cfg = Config.fromfile(config_path)
    model_cfg = cfg.model
    enc_cfg = model_cfg.get("pts_middle_encoder", None)
    if enc_cfg is None:
        raise ValueError(f"Config {config_path} has no pts_middle_encoder")

    enc_cfg = copy.deepcopy(enc_cfg)

    if "block_type" not in enc_cfg:
        enc_cfg["block_type"] = "conv_module"

    encoder = MODELS.build(enc_cfg)
    encoder.to(device)
    encoder.eval()
    return encoder


def _load_weights_from_ptq_checkpoint(
    encoder: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> dict:
    """Load FP32 weights from PTQ checkpoint into the fresh encoder.

    Returns the full checkpoint state_dict (for _amax extraction).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    enc_sd = encoder.state_dict()

    prefix_candidates = [
        "pts_middle_encoder.",
        "module.pts_middle_encoder.",
        "module.",
        "",
    ]

    def _find_value(key: str):
        for prefix in prefix_candidates:
            full_key = f"{prefix}{key}"
            if full_key in state_dict:
                v = state_dict[full_key]
                if hasattr(v, "is_quantized") and v.is_quantized:
                    v = v.dequantize()
                return v
        return None

    def _align_5d(src: torch.Tensor, target_shape: torch.Size):
        if src.shape == target_shape:
            return src
        for perm in [(1, 2, 3, 4, 0), (4, 0, 1, 2, 3)]:
            try:
                aligned = src.permute(*perm).contiguous()
                if aligned.shape == target_shape:
                    return aligned
            except RuntimeError:
                continue
        return None

    loaded = 0
    with torch.no_grad():
        for k, target in enc_sd.items():
            v = _find_value(k)
            if v is None:
                continue
            if v.dim() == 5 and target.dim() == 5 and v.shape != target.shape:
                v = _align_5d(v, target.shape)
                if v is None:
                    continue
            if v.shape != target.shape:
                continue
            parent_path, _, leaf = k.rpartition(".")
            if not parent_path:
                continue
            try:
                sub = encoder.get_submodule(parent_path)
            except AttributeError:
                continue
            dst = getattr(sub, leaf, None)
            if dst is not None and torch.is_tensor(dst):
                dst.copy_(v.to(device=dst.device, dtype=dst.dtype))
                loaded += 1

    print(f"[load] Copied {loaded}/{len(enc_sd)} tensors from checkpoint")
    return state_dict


def _fuse_bn(encoder: nn.Module) -> int:
    """Fuse BatchNorm into sparse convolutions."""
    from deployment.projects.bevfusion.quantization.spconv_int8 import (
        _fuse_spconv_bn_in_encoder,
    )
    count = _fuse_spconv_bn_in_encoder(encoder)
    print(f"[fuse-bn] Fused {count} Conv-BN pairs")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Export BEVFusion sparse encoder to libspconv INT8 ONNX"
    )
    parser.add_argument(
        "--config", required=True,
        help="MMEngine config for BEVFusion (must define pts_middle_encoder)",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="PTQ checkpoint with NVIDIA _amax calibration values",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output .onnx file path",
    )
    parser.add_argument(
        "--in-channel", type=int, default=5,
        help="Number of input voxel feature channels (default: 5)",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="Device (default: cuda:0)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # 1. Build fresh encoder
    print("=" * 60)
    print("Step 1: Building fresh BEVFusionSparseEncoder from config")
    print("=" * 60)
    encoder = _build_fresh_encoder(args.config, device)

    # 2. Fuse BN
    print("\nStep 2: Fusing BatchNorm into SparseConvolution")
    _fuse_bn(encoder)

    # 3. Load weights from PTQ checkpoint
    print(f"\nStep 3: Loading weights from {args.checkpoint}")
    full_state_dict = _load_weights_from_ptq_checkpoint(encoder, args.checkpoint, device)

    # 4. Extract dynamic ranges from _amax values
    print("\nStep 4: Extracting dynamic ranges from _amax calibration")
    from deployment.projects.bevfusion.export.libspconv_onnx_exporter import (
        LibspconvExporter,
        extract_dynamic_ranges_from_checkpoint,
        set_precision_attributes,
    )
    n_ranges = extract_dynamic_ranges_from_checkpoint(encoder, full_state_dict)
    print(f"  Set dynamic ranges on {n_ranges} modules")

    # 5. Set precision attributes
    print("\nStep 5: Setting precision attributes (INT8 with FP16 boundaries)")
    set_precision_attributes(encoder)

    # 6. Convert to FP16
    print("\nStep 6: Converting encoder to FP16")
    encoder.half()
    encoder.eval()

    # 7. Create dummy input and export
    print(f"\nStep 7: Exporting to {args.output}")
    voxels = torch.zeros(1, args.in_channel, device=device, dtype=torch.float16)
    coors = torch.zeros(1, 4, device=device, dtype=torch.int32)

    exporter = LibspconvExporter()
    exporter.export(encoder, voxels, coors, batch_size=1, save_path=args.output)

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"  ONNX: {args.output}")
    print(f"  Use with: spconv::load_engine_from_onnx(\"{args.output}\", Precision::Int8)")
    print("=" * 60)


if __name__ == "__main__":
    main()
