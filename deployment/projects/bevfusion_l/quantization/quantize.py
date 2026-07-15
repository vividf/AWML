#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
BEVFusion Quantization Tools

This script provides CLI commands for PTQ (Post-Training Quantization) for BEVFusion models. It
mirrors centerpoint/quantization/quantize.py. The dense tower (pts_backbone, pts_neck, bbox_head) is
quantized with pytorch_quantization Q/DQ; the sparse encoder (pts_middle_encoder) deploys in FP16 and
is only BN-folded so the PTQ and deploy module trees line up.

Usage:
    # PTQ Mode - Quantize a pre-trained BEVFusion model (dense INT8, sparse FP16)
    python -m deployment.projects.bevfusion_l.quantization.quantize ptq \
        --config projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
        --checkpoint work_dirs/bevfusion/epoch_30.pth \
        --deploy-cfg deployment/projects/bevfusion_l/config/deploy_config_split_sparse_fp16_dense_int8_2_8.py \
        --calibrate-samples 256 \
        --batch-size 1 \
        --calib-seed 0 \
        --output work_dirs/bevfusion/epoch_30_ptq.pth
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(
        description="BEVFusion Quantization Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    ptq_parser = subparsers.add_parser("ptq", help="Post-Training Quantization")
    ptq_parser.add_argument("--config", required=True, help="Model config file path")
    ptq_parser.add_argument("--checkpoint", required=True, help="Model checkpoint file path")
    ptq_parser.add_argument(
        "--deploy-cfg",
        required=True,
        help="Deployment config path with quantization settings.",
    )
    ptq_parser.add_argument(
        "--calibrate-samples",
        type=int,
        default=256,
        help="Total number of samples for calibration (default: 256).",
    )
    ptq_parser.add_argument("--output", required=True, help="Output PTQ checkpoint path")
    ptq_parser.add_argument("--device", default="cuda:0", help="Device for calibration")
    ptq_parser.add_argument("--calib-shuffle", action="store_true", help="Shuffle calibration data")
    ptq_parser.add_argument("--calib-seed", type=int, default=None, help="Random seed for calibration")
    ptq_parser.add_argument("--batch-size", type=int, default=1, help="Batch size for calibration")
    return parser.parse_args()


def _build_bevfusion_model(config_path: str, checkpoint_path: str, device: str):
    """Build BEVFusion model using mmdet3d init_model."""
    from mmdet3d.apis import init_model
    from mmengine.registry import init_default_scope

    import projects.BEVFusion.bevfusion  # noqa: F401
    import projects.SparseConvolution  # noqa: F401

    init_default_scope("mmdet3d")
    model = init_model(config_path, checkpoint_path, device=device)
    model.eval()
    return model


def _calibrate_dense(model, dataloader, num_batches, method="mse"):
    """Run calibration for dense Q/DQ nodes using CalibrationManager."""
    import torch

    from deployment.quantization import CalibrationManager

    def _force_float_voxel_inputs(batch):
        """Best-effort dtype normalization before test_step during calibration.

        Some dataloader/preprocessor paths may provide integer voxel features.
        Integer voxel features or points are coerced to float32 where needed for
        dense Q/DQ calibration.
        """
        if not isinstance(batch, dict):
            return batch
        inputs = batch.get("inputs", None)
        if not isinstance(inputs, dict):
            return batch

        vox = inputs.get("voxels", None)
        if isinstance(vox, dict):
            v = vox.get("voxels", None)
            if isinstance(v, torch.Tensor) and not v.is_floating_point():
                vox["voxels"] = v.to(dtype=torch.float32).contiguous()

        points = inputs.get("points", None)
        if isinstance(points, (list, tuple)):
            normalized = []
            changed = False
            for p in points:
                if isinstance(p, torch.Tensor) and not p.is_floating_point():
                    normalized.append(p.to(dtype=torch.float32))
                    changed = True
                else:
                    normalized.append(p)
            if changed:
                inputs["points"] = type(points)(normalized) if isinstance(points, tuple) else normalized
        return batch

    def _forward_for_calibration(m, batch):
        batch = _force_float_voxel_inputs(batch)
        if hasattr(m, "test_step"):
            return m.test_step(batch)
        if isinstance(batch, dict):
            return m(**batch)
        if isinstance(batch, (list, tuple)):
            return m(*batch)
        return m(batch)

    calibrator = CalibrationManager(model)
    calibrator.calibrate(dataloader, num_batches=num_batches, method=method, forward_fn=_forward_for_calibration)
    return calibrator


def run_ptq(args):
    """Run BEVFusion PTQ pipeline (dense INT8 Q/DQ; sparse encoder stays FP16)."""
    import math

    from mmengine.config import Config

    from deployment.quantization.producer import build_calib_dataloader, save_ptq_checkpoint

    num_batches = math.ceil(args.calibrate_samples / args.batch_size)
    actual_samples = num_batches * args.batch_size

    print("=" * 80)
    print("BEVFusion PTQ Quantization (dense INT8, sparse FP16)")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Deploy cfg: {args.deploy_cfg}")
    print(f"Calibration: {args.calibrate_samples} samples -> {num_batches} batches x {args.batch_size}")
    print(f"Actual calibration samples: {actual_samples}")
    if args.calib_seed is not None:
        print(f"Calibration seed: {args.calib_seed}")
    print(f"Output: {args.output}")
    print("=" * 80)

    from deployment.config.schema import load_quantization_config
    from deployment.projects.bevfusion_l.quantization.plan import build_bevfusion_plan

    # Single source of truth: parse the deploy ``quantization`` block once.
    config, _ = load_quantization_config(args.deploy_cfg)

    from deployment.quantization import expand_keep_fp16

    fuse_bn = config.fuse_bn
    dense_on = config.enabled

    # [1/5] Load model
    print("\n[1/5] Loading BEVFusion model...")
    model = _build_bevfusion_model(args.config, args.checkpoint, args.device)

    # Resolve keep_fp16 → concrete module names for the disable loop below (same expansion the plan
    # uses internally; log=False to avoid duplicate per-pattern match logging).
    skip_layers = expand_keep_fp16(model, config.keep_fp16, log=False)

    # [2-3/5] Dense BN fuse + dense Q/DQ AND sparse (FP16) SparseConv-BN fold, all via the SHARED
    # QuantizationPlan. The deploy loader builds the SAME plan, so the PTQ state_dict and the deployed
    # module tree line up by construction. The sparse fold is part of the plan (gated on ``fuse_bn``),
    # so PTQ and deploy pass identical arguments — there is no divergent ``include_sparse`` (spec §3.8(3)).
    if fuse_bn:
        print("\n[2-3/5] Dense + sparse BN fuse + dense Q/DQ via shared QuantizationPlan...")
    else:
        print("\n[2-3/5] Dense Q/DQ via shared QuantizationPlan (BN fusion disabled)...")
    build_bevfusion_plan(config).prepare(model)

    # [4/5] Build dataloader + calibrate dense Q/DQ (shared PTQ-producer helper; BEVFusion caps
    # dataloader workers and disables persistent workers to keep calibration memory bounded).
    print("\n[4/5] Building calibration dataloader...")
    cfg = Config.fromfile(args.config)
    dataloader = build_calib_dataloader(
        cfg,
        batch_size=args.batch_size,
        seed=args.calib_seed,
        shuffle=args.calib_shuffle,
        max_num_workers=4,
        persistent_workers=False,
    )
    total_ds = len(dataloader.dataset)
    print(f"  Dataset size: {total_ds}")
    print(f"  Calibration: {num_batches} batches x {args.batch_size} = {actual_samples} samples")

    if dense_on:
        print(f"  Calibrating dense Q/DQ nodes ({num_batches} batches, method=mse)...")
        calibrator = _calibrate_dense(model, dataloader, num_batches, method="mse")
    else:
        print("  Skipping dense calibration (no dense TensorQuantizer modules).")
        calibrator = None

    if skip_layers and dense_on:
        from deployment.quantization import disable_quantizers_in

        print(f"\n  Disabling quantizers in {len(skip_layers)} keep_fp16 module(s)...")
        disable_quantizers_in(model, skip_layers)

    # [5/5] Print status and save
    print("\n[5/5] Saving PTQ checkpoint...")

    try:
        from deployment.quantization import print_quantizer_status

        print("\nQuantizer Status:")
        print_quantizer_status(model)
    except Exception:
        pass

    output_path, calib_path = save_ptq_checkpoint(model, args.output, calibrator)

    print("\n" + "=" * 80)
    print("BEVFusion PTQ Complete!")
    print(f"Model saved to: {output_path}")
    if calib_path is not None:
        print(f"Calibration cache saved to: {calib_path}")
    print("=" * 80)
    print("\nTo use this PTQ checkpoint for deployment:")
    print(f'  1. Set checkpoint_path = "{args.output}" in your deploy config')
    print(f"  2. Set quantization.ptq_checkpoint = True")
    print(f"  3. Run:")
    print(f"     python -m deployment.cli.main bevfusion_l \\")
    print(f"       {args.deploy_cfg} \\")
    print(f"       {args.config}")


def main():
    from deployment.quantization.producer import init_quant_logging

    args = parse_args()
    init_quant_logging()

    if args.command == "ptq":
        run_ptq(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
