#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
BEVFusion Quantization Tools

This script provides CLI commands for PTQ (Post-Training Quantization) and QAT
(Quantization-Aware Training) for BEVFusion models. It mirrors
centerpoint/quantization/quantize.py. The dense tower (pts_backbone, pts_neck, bbox_head) is
quantized with pytorch_quantization Q/DQ; the sparse encoder (pts_middle_encoder) deploys in FP16 and
is only BN-folded so the PTQ and deploy module trees line up. QAT is a frozen-amax STE fine-tune of
the same tree (spec_qat.md); the packaged QAT checkpoint deploys exactly like a PTQ one.

Usage:
    # PTQ Mode - config-driven (settings from the deploy config's quantization.ptq block;
    # --output defaults to the config's checkpoint_path)
    python -m deployment.projects.bevfusion_l.quantization.quantize ptq \
        --deploy-cfg deployment/projects/bevfusion_l/config/deploy_config_split_sparse_fp16_dense_int8_2_8.py

    # PTQ Mode - CLI-driven (flags override / substitute the ptq block)
    python -m deployment.projects.bevfusion_l.quantization.quantize ptq \
        --deploy-cfg deployment/projects/bevfusion_l/config/deploy_config_split_sparse_fp16_dense_int8_2_8.py \
        --config projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
        --checkpoint work_dirs/bevfusion/epoch_30.pth \
        --calibrate-samples 256 \
        --batch-size 1 \
        --calib-seed 0 \
        --output work_dirs/bevfusion/epoch_30_ptq.pth

    # QAT Mode - Fine-tune the quantized model (single GPU; AMP forced off)
    python -m deployment.projects.bevfusion_l.quantization.quantize qat \
        --deploy-cfg deployment/projects/bevfusion_l/config/deploy_config_split_sparse_fp16_dense_int8_2_8.py \
        --config projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
        --checkpoint work_dirs/bevfusion/epoch_30.pth \
        --epochs 3 --lr 1e-4 --calibrate-samples 400 \
        --output work_dirs/bevfusion/epoch_30_qat.pth
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
    ptq_parser.add_argument(
        "--deploy-cfg",
        required=True,
        help=(
            "Required deployment config path. Its `quantization` block is the single source of "
            "truth for placement (keep_fp16 / disable_recipes / fuse_bn); with mode='ptq' its "
            "`ptq` sub-block supplies the producer settings (CLI flags below override them) — "
            "same shape as the QAT command."
        ),
    )
    ptq_parser.add_argument(
        "--config", default=None, help="Model config path (overrides the deploy config's top-level model_cfg)."
    )
    ptq_parser.add_argument(
        "--checkpoint",
        default=None,
        help="FP checkpoint path to quantize (overrides quantization.ptq.checkpoint).",
    )
    ptq_parser.add_argument(
        "--calibrate-samples",
        type=int,
        default=None,
        help=(
            "Total number of samples for calibration "
            "(overrides quantization.ptq.calibrate_samples; 256 without either)."
        ),
    )
    ptq_parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output PTQ checkpoint path (+ sibling .calib). Defaults to the deploy config's "
            "`checkpoint_path`, so the run produces exactly the artifact the deploy config expects."
        ),
    )
    ptq_parser.add_argument("--device", default="cuda:0", help="Device for calibration")
    ptq_parser.add_argument(
        "--calib-shuffle",
        action="store_true",
        default=None,
        help="Shuffle calibration data (overrides quantization.ptq.calib_shuffle)",
    )
    ptq_parser.add_argument(
        "--calib-seed",
        type=int,
        default=None,
        help="Random seed for calibration (overrides quantization.ptq.calib_seed)",
    )
    ptq_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for calibration (overrides quantization.ptq.batch_size; 1 without either)",
    )

    qat_parser = subparsers.add_parser(
        "qat",
        help="Quantization-Aware Training (dense INT8 fine-tune; sparse encoder stays FP16)",
    )
    qat_parser.add_argument(
        "--deploy-cfg",
        required=True,
        help=(
            "Required deployment config path. Its `quantization` block is the single source of "
            "truth for placement (keep_fp16 / disable_recipes / fuse_bn); with mode='qat' its "
            "`qat` sub-block supplies the training settings (CLI flags below override them)."
        ),
    )
    qat_parser.add_argument(
        "--config", default=None, help="Model training config path (overrides quantization.qat.train_cfg)."
    )
    qat_parser.add_argument(
        "--checkpoint", default=None, help="Initial FP checkpoint path (overrides quantization.qat.checkpoint)."
    )
    qat_parser.add_argument(
        "--calibrate-samples",
        type=int,
        default=None,
        help=(
            "Total number of samples for the epoch-0 calibration "
            "(overrides quantization.qat.calibrate_samples; reference: 400). "
            "Without either, all training batches are used."
        ),
    )
    qat_parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for calibration/training dataloaders (default: 1)"
    )
    qat_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Number of fine-tuning epochs (overrides quantization.qat.epochs). "
            "Reference recipe: ~10%% of the original training epochs (spec_qat.md §2)."
        ),
    )
    qat_parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "Learning rate for fine-tuning (overrides quantization.qat.lr). "
            "Reference recipe: 1e-4 (CUDA-CenterPoint / modelopt CNN QAT)."
        ),
    )
    qat_parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output path for the packaged QAT checkpoint (`{'state_dict'}` + sibling .calib). "
            "Defaults to the deploy config's `checkpoint_path`."
        ),
    )
    qat_parser.add_argument(
        "--work-dir", default=None, help="Working directory for training (overrides quantization.qat.work_dir)."
    )
    qat_parser.add_argument(
        "--ptq-calib-cache",
        default=None,
        help="Path to PTQ calibration cache (.calib file) — overrides quantization.qat.calib_cache. "
        "If provided, QAT will load existing amax values instead of running new calibration.",
    )
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
    from deployment.quantization import CalibrationManager

    from .calibration import calibration_forward

    calibrator = CalibrationManager(model)
    calibrator.calibrate(dataloader, num_batches=num_batches, method=method, forward_fn=calibration_forward)
    return calibrator


def run_ptq(args):
    """Run BEVFusion PTQ pipeline (dense INT8 Q/DQ; sparse encoder stays FP16)."""
    import math

    from mmengine.config import Config

    from deployment.config.schema import load_quantization_config
    from deployment.projects.bevfusion_l.quantization.plan import build_bevfusion_plan
    from deployment.quantization.producer import build_calib_dataloader, resolve_ptq_settings, save_ptq_checkpoint

    # The deploy config is the single source of truth: placement (keep_fp16 / disable_recipes /
    # fuse_bn) always; producer settings via its `ptq` block; model_cfg / checkpoint_path from the
    # top-level manifest keys — CLI flags override any of them.
    config, deploy_checkpoint_path, deploy_model_cfg = load_quantization_config(args.deploy_cfg)
    if config.mode != "ptq":
        print(
            f'Note: deploy config has quantization.mode="{config.mode}" — a PTQ run usually pairs '
            'with mode="ptq" (+ a ptq block). Proceeding with CLI settings.'
        )
    settings = resolve_ptq_settings(
        args, config, deploy_checkpoint_path, deploy_model_cfg, default_calibrate_samples=256
    )
    batch_size = settings["batch_size"]

    num_batches = math.ceil(settings["calibrate_samples"] / batch_size)
    actual_samples = num_batches * batch_size

    print("=" * 80)
    print("BEVFusion PTQ Quantization (dense INT8, sparse FP16)")
    print("=" * 80)
    print(f"Deploy cfg: {args.deploy_cfg}")
    print(f"Config: {settings['model_cfg']}")
    print(f"Checkpoint: {settings['checkpoint']}")
    print(f"Calibration: {settings['calibrate_samples']} samples -> {num_batches} batches x {batch_size}")
    print(f"Actual calibration samples: {actual_samples}")
    if settings["calib_seed"] is not None:
        print(f"Calibration seed: {settings['calib_seed']}")
    print(f"Output: {settings['output']}")
    print("=" * 80)

    from deployment.quantization import expand_keep_fp16

    fuse_bn = config.fuse_bn
    dense_on = config.enabled

    # [1/5] Load model
    print("\n[1/5] Loading BEVFusion model...")
    model = _build_bevfusion_model(settings["model_cfg"], settings["checkpoint"], args.device)

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
    cfg = Config.fromfile(settings["model_cfg"])
    dataloader = build_calib_dataloader(
        cfg,
        batch_size=batch_size,
        seed=settings["calib_seed"],
        shuffle=settings["calib_shuffle"],
        max_num_workers=4,
        persistent_workers=False,
    )
    total_ds = len(dataloader.dataset)
    print(f"  Dataset size: {total_ds}")
    print(f"  Calibration: {num_batches} batches x {batch_size} = {actual_samples} samples")

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

    output_path, calib_path = save_ptq_checkpoint(model, settings["output"], calibrator)

    print("\n" + "=" * 80)
    print("BEVFusion PTQ Complete!")
    print(f"Model saved to: {output_path}")
    if calib_path is not None:
        print(f"Calibration cache saved to: {calib_path}")
    print("=" * 80)
    print("\nTo use this PTQ checkpoint for deployment:")
    print(f'  1. Ensure checkpoint_path = "{settings["output"]}" in your deploy config')
    print(f"  2. Set quantization.ptq_checkpoint = True")
    print(f"  3. Run:")
    print(f"     python -m deployment.cli.main bevfusion_l \\")
    print(f"       {args.deploy_cfg} \\")
    print(f"       {settings['model_cfg']}")


def run_qat(args):
    """Run QAT training pipeline (shared driver; BEVFusion supplies only project constants)."""
    import math

    from deployment.config.schema import load_quantization_config
    from deployment.quantization.producer import resolve_qat_settings, run_qat_training

    # The deploy config is the single source of truth: placement (keep_fp16 / disable_recipes /
    # fuse_bn) always; training settings via its `qat` block, with CLI flags as overrides.
    # (The top-level model_cfg is the DEPLOY pairing — QAT trains on qat.train_cfg instead.)
    config, deploy_checkpoint_path, _ = load_quantization_config(args.deploy_cfg)
    if config.mode != "qat":
        print(
            f'Note: deploy config has quantization.mode="{config.mode}" — a QAT run usually pairs '
            'with mode="qat" (+ a qat block). Proceeding with CLI settings.'
        )
    settings = resolve_qat_settings(args, config, deploy_checkpoint_path)

    if settings["calibrate_samples"] is not None:
        calibration_batches = math.ceil(settings["calibrate_samples"] / args.batch_size)
    else:
        calibration_batches = 0

    print("=" * 80)
    print("BEVFusion QAT Training (dense INT8 frozen-amax STE fine-tune; sparse FP16)")
    print("=" * 80)
    print(f"Deploy cfg: {args.deploy_cfg}")
    print(f"Train config: {settings['train_cfg']}")
    print(f"Init checkpoint: {settings['checkpoint']}")
    if calibration_batches > 0:
        print(
            f"Calibration: {calibration_batches} batches "
            f"(from calibrate_samples={settings['calibrate_samples']}, batch_size={args.batch_size})"
        )
    else:
        print("Calibration: using all training batches (or loading calib cache if provided)")
    print(f"Epochs: {settings['epochs']}")
    print(f"Learning rate: {settings['lr']}")
    print(f"Output: {settings['output']}")
    print("=" * 80)

    output_path, calib_path = run_qat_training(
        train_cfg_path=settings["train_cfg"],
        checkpoint=settings["checkpoint"],
        hook_import="deployment.projects.bevfusion_l.quantization.qat_hook",
        hook_type="BEVFusionQATHook",
        quant_config=config,
        epochs=settings["epochs"],
        lr=settings["lr"],
        output=settings["output"],
        batch_size=args.batch_size,
        calibration_batches=calibration_batches,
        calib_cache=settings["calib_cache"],
        work_dir=settings["work_dir"],
        # Model-registry imports the BEVFusion train config needs before the Runner builds.
        # Deliberately WITHOUT projects.SparseConvolution: that package force-registers the
        # deploy-only SparseConv3d/SubMConv3d fork whose forward raises NotImplementedError in
        # training mode (projects/SparseConvolution/sparse_conv.py). QAT must train on the stock
        # spconv classes, exactly like FP training does; state_dict keys and the SparseConv+BN
        # fold are identical either way, so the deploy side (which does import it) still lines up.
        extra_imports=("projects.BEVFusion.bevfusion",),
    )

    print("\n" + "=" * 80)
    print("QAT training completed!")
    print(f"Packaged checkpoint: {output_path}")
    print(f"Calibration cache: {calib_path}")
    print("Deploy it exactly like a PTQ checkpoint (same loader path; set ptq_checkpoint=True).")
    print("=" * 80)


def main():
    from deployment.quantization.producer import init_quant_logging

    args = parse_args()
    init_quant_logging()

    if args.command == "ptq":
        run_ptq(args)
    elif args.command == "qat":
        run_qat(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
