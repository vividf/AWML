#!/usr/bin/env python
"""Verify that BEVFusion PTQ model uses INT8 for pts_middle_encoder (spconv).

Loads the model with deploy config and checks:
  1. pts_middle_encoder is FX GraphModule with qint8 parameters / quantized modules
  2. Optionally: time sparse encoder forward (median over N runs)

Usage:
  python -m deployment.projects.bevfusion.scripts.verify_spconv_int8 \\
    deployment/projects/bevfusion/config/deploy_config_int8.py \\
    projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m_fx.py \\
    [--timing-runs 50]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

# Import torch after path is set (required for timing and model load)
import torch


def main():
    parser = argparse.ArgumentParser(description="Verify spconv INT8 encoder and optionally benchmark it.")
    parser.add_argument("deploy_cfg", help="Path to deploy config (e.g. deploy_config_int8.py)")
    parser.add_argument("model_cfg", help="Path to model config (e.g. bevfusion_*_120m_fx.py)")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument(
        "--timing-runs", type=int, default=0, help="Number of forward runs for sparse encoder timing (0 = skip)"
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs before timing")
    args = parser.parse_args()

    from mmengine.config import Config

    from deployment.core.device import DeviceSpec
    from deployment.projects.bevfusion.io.model_loader import (
        build_bevfusion_model,
        verify_spconv_int8_encoder,
    )

    deploy_cfg = Config.fromfile(args.deploy_cfg)
    model_cfg = Config.fromfile(args.model_cfg)
    checkpoint_path = getattr(deploy_cfg, "checkpoint_path", None)
    if not checkpoint_path or not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Set checkpoint_path in deploy config to your PTQ .pth file.")
        sys.exit(1)

    quantization = getattr(deploy_cfg, "quantization", None) or {}
    if not quantization.get("enabled") or not quantization.get("spconv_int8"):
        print("WARNING: deploy config quantization.enabled or quantization.spconv_int8 is not True.")
        print("Verification may report non-INT8 encoder.")

    device = DeviceSpec(args.device)
    print("Loading model...")
    model = build_bevfusion_model(model_cfg, checkpoint_path, device, quantization=quantization)
    model.eval()

    info = verify_spconv_int8_encoder(model)
    print()
    print("=" * 60)
    print("Spconv INT8 encoder verification")
    print("=" * 60)
    print(f"  pts_middle_encoder type: {info['encoder_type']}")
    print(f"  Is GraphModule (FX converted): {info['is_graph_module']}")
    print(f"  Total params in encoder: {info['total_param_count']}")
    print(f"  Quantized (qint8) params: {info['quantized_param_count']}")
    print(f"  Quantized module type count: {len(info['quantized_module_types'])}")
    if info["quantized_module_types"]:
        for t in sorted(info["quantized_module_types"])[:10]:
            print(f"    - {t}")
        if len(info["quantized_module_types"]) > 10:
            print(f"    ... and {len(info['quantized_module_types']) - 10} more")
    print()
    if info["is_int8"]:
        print("  Result: INT8 encoder VERIFIED (GraphModule + quantized params/modules)")
    else:
        print("  Result: INT8 encoder NOT verified (encoder may be FP16/FP32)")
        print("  Ensure PTQ was run with spconv_int8 and config block_type='basicblock_fx'.")
    print("=" * 60)

    if args.timing_runs > 0 and hasattr(model, "pts_middle_encoder"):
        from deployment.projects.bevfusion.quantization.spconv_int8 import _create_example_inputs

        enc = model.pts_middle_encoder
        sparse_shape = getattr(enc, "sparse_shape", [41, 1440, 1440])
        in_ch = getattr(enc, "in_channels", 5)
        device_torch = device.to_torch_device()
        voxel_features, coors, batch_size = _create_example_inputs(
            enc,
            device_torch,
            in_channels=in_ch,
            num_voxels=min(50000, sparse_shape[0] * sparse_shape[1] * sparse_shape[2] // 10),
        )
        enc.eval()
        with torch.no_grad():
            for _ in range(args.warmup):
                enc(voxel_features, coors, batch_size)
            if device_torch.type == "cuda":
                torch.cuda.synchronize()
            import time

            times = []
            for _ in range(args.timing_runs):
                if device_torch.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                enc(voxel_features, coors, batch_size)
                if device_torch.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
            import numpy as np

            median_ms = float(np.median(times))
            print()
            print("Sparse encoder only timing (median over {} runs): {:.2f} ms".format(args.timing_runs, median_ms))
            print("(Compare with non-PTQ model to see INT8 speedup; 16-channel layers may still run FP16.)")
    elif args.timing_runs > 0:
        print("No pts_middle_encoder; skipping timing.")

    return 0 if info["is_int8"] else 1


if __name__ == "__main__":
    sys.exit(main())
