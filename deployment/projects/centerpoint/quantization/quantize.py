#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
CenterPoint Quantization Tools

This script provides CLI commands for PTQ (Post-Training Quantization) and
QAT (Quantization-Aware Training) for CenterPoint models.

Usage:
    # PTQ Mode - config-driven (settings from the deploy config's quantization.ptq block;
    # --output defaults to the config's checkpoint_path)
    python -m deployment.projects.centerpoint.quantization.quantize ptq \
        --deploy-cfg deployment/projects/centerpoint/config/deploy_config_int8_second_2_6_quant_release.py

    # PTQ Mode - CLI-driven (flags override / substitute the ptq block)
    python -m deployment.projects.centerpoint.quantization.quantize ptq \
        --deploy-cfg deployment/projects/centerpoint/config/deploy_config_int8.py \
        --config projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py \
        --checkpoint work_dirs/centerpoint/best.pth \
        --calibrate-samples 938 \
        --batch-size 4 \
        --calib-seed 0 \
        --output work_dirs/centerpoint_ptq.pth

    # QAT Mode - config-driven (settings from the deploy config's quantization.qat block)
    python -m deployment.projects.centerpoint.quantization.quantize qat \
        --deploy-cfg deployment/projects/centerpoint/config/deploy_config_int8_second_qat.py

    # QAT Mode - CLI-driven (flags override / substitute the qat block)
    python -m deployment.projects.centerpoint.quantization.quantize qat \
        --deploy-cfg deployment/projects/centerpoint/config/deploy_config_int8.py \
        --config projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py \
        --checkpoint work_dirs/centerpoint/best.pth \
        --calibrate-samples 400 \
        --epochs 3 \
        --lr 0.0001 \
        --output work_dirs/centerpoint_qat.pth
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CenterPoint Quantization Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # =========================================================================
    # PTQ command
    # =========================================================================
    ptq_parser = subparsers.add_parser(
        "ptq",
        help="Post-Training Quantization",
        description="Apply PTQ to a pre-trained CenterPoint model",
    )
    ptq_parser.add_argument(
        "--deploy-cfg",
        required=True,
        help=(
            "Required deployment config path. Its `quantization` block is the single source of "
            "truth for placement (default_precision / keep_fp16 / disable_recipes / fuse_bn); with "
            "mode='ptq' its `ptq` sub-block supplies the producer settings (CLI flags below "
            "override them) — same shape as the QAT command."
        ),
    )
    ptq_parser.add_argument(
        "--config",
        default=None,
        help="Model config path (overrides the deploy config's top-level model_cfg).",
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
            "Total number of samples for calibration (overrides quantization.ptq.calibrate_samples; "
            "100 without either). Batches are auto-calculated from the batch size."
        ),
    )
    ptq_parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output checkpoint path (+ sibling .calib). Defaults to the deploy config's "
            "`checkpoint_path`, so the run produces exactly the artifact the deploy config expects."
        ),
    )
    ptq_parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for calibration (default: cuda:0)",
    )
    ptq_parser.add_argument(
        "--calib-shuffle",
        action="store_true",
        default=None,
        help="Shuffle the calibration data (overrides quantization.ptq.calib_shuffle).",
    )
    ptq_parser.add_argument(
        "--calib-seed",
        type=int,
        default=None,
        help="Random seed for calibration data shuffling (overrides quantization.ptq.calib_seed)",
    )
    ptq_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Batch size for calibration (overrides quantization.ptq.batch_size; 1 without either). "
            "Larger batch size can reduce seed sensitivity."
        ),
    )

    # =========================================================================
    # QAT command
    # =========================================================================
    qat_parser = subparsers.add_parser(
        "qat",
        help="Quantization-Aware Training",
        description="Fine-tune model with quantization-aware training",
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
        "--config",
        default=None,
        help="Model training config path (overrides quantization.qat.train_cfg).",
    )
    qat_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Initial FP checkpoint path (overrides quantization.qat.checkpoint).",
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
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for calibration/training dataloaders (default: 1)",
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
            "Defaults to the deploy config's `checkpoint_path`, so the run produces exactly the "
            "artifact the deploy config expects."
        ),
    )
    qat_parser.add_argument(
        "--work-dir",
        default=None,
        help="Working directory for training (overrides quantization.qat.work_dir).",
    )
    qat_parser.add_argument(
        "--ptq-calib-cache",
        default=None,
        help="Path to PTQ calibration cache (.calib file) — overrides quantization.qat.calib_cache. "
        "If provided, QAT will load existing amax values instead of running new calibration.",
    )
    return parser.parse_args()


def run_ptq(args):
    """Run PTQ quantization pipeline."""
    import math

    from mmdet3d.apis import init_model
    from mmengine.config import Config

    from deployment.config.schema import load_quantization_config
    from deployment.quantization import (
        CalibrationManager,
        disable_quantizers_in,
        expand_keep_fp16,
        print_quantizer_status,
    )
    from deployment.quantization.producer import build_calib_dataloader, resolve_ptq_settings, save_ptq_checkpoint

    from .plan import build_centerpoint_plan

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
        args, config, deploy_checkpoint_path, deploy_model_cfg, default_calibrate_samples=100
    )
    batch_size = settings["batch_size"]

    # Auto-calculate calibrate_batches from calibrate_samples and batch_size
    calibrate_batches = math.ceil(settings["calibrate_samples"] / batch_size)
    actual_samples = calibrate_batches * batch_size
    print(
        f"Auto-calculated: {settings['calibrate_samples']} samples → "
        f"{calibrate_batches} batches × {batch_size} = {actual_samples} samples"
    )

    print("=" * 80)
    print("CenterPoint PTQ Quantization")
    print("=" * 80)
    print(f"Deploy cfg: {args.deploy_cfg}")
    print(f"Config: {settings['model_cfg']}")
    print(f"Checkpoint: {settings['checkpoint']}")
    print(f"Calibration batches: {calibrate_batches}")
    print(f"Batch size: {batch_size}")
    print(f"Calibration shuffle: {settings['calib_shuffle']}")
    if settings["calib_seed"] is not None:
        print(f"Calibration seed: {settings['calib_seed']}")
    print("Amax method: mse")
    print(f"Output: {settings['output']}")
    print("=" * 80)

    if "add" not in config.disable_recipes:
        print("Note: Residual-add quantization enabled (only the identity branch is quantized, to")
        print("      enable TensorRT Conv+Add fusion; class-gated). Disable via disable_recipes=['add'].")

    # Load model
    print("\n[1/5] Loading model...")
    cfg = Config.fromfile(settings["model_cfg"])
    model = init_model(cfg, settings["checkpoint"], device=args.device)
    model.eval()

    # Resolve keep_fp16 → concrete module names for the disable loop below (needs the model; same
    # expansion the plan uses internally, log=False to avoid duplicate per-pattern match logging).
    skip_layers = expand_keep_fp16(model, config.keep_fp16, log=False)

    # Fuse BN + insert Q/DQ via the SHARED CenterPoint plan. The deploy loader builds the same plan,
    # so the PTQ state_dict and the deployed module tree line up by construction.
    print("\n[2-3/5] Fusing BatchNorm + inserting Q/DQ via shared CenterPoint plan...")
    build_centerpoint_plan(config).prepare(model)

    print(
        "  - Architecture recipes (residual-add / eSE / maxpool): always-on & class-gated; "
        f"disabled: {sorted(config.disable_recipes) or 'none'}"
    )

    # Build dataloader (shared PTQ-producer helper: batch_size override + seed + shuffle)
    print("\n[4/5] Building calibration dataloader...")
    dataloader = build_calib_dataloader(
        cfg,
        batch_size=batch_size,
        seed=settings["calib_seed"],
        shuffle=settings["calib_shuffle"],
    )

    # Print dataset size (best-effort)
    try:
        total_samples = len(dataloader.dataset)
        print(f"  Total samples in dataset: {total_samples}")
        print(f"  Total calibration samples: {actual_samples}")
    except Exception:
        pass

    # Calibrate
    print(f"\n[5/5] Calibrating with {calibrate_batches} batches ({actual_samples} samples)...")
    calibrator = CalibrationManager(model)
    calibrator.calibrate(
        dataloader,
        num_batches=calibrate_batches,
        method="mse",  # fixed to mse to match CUDA-CenterPoint behavior
    )

    # Disable quantizers in the keep_fp16 subtrees (shared loop — same as QAT hook and deploy loader).
    disabled = disable_quantizers_in(model, skip_layers)
    if disabled:
        print(f"  Disabled quantization in {disabled} keep_fp16 module(s)")

    # Print status (covers every quantizer, including the recipe-attached residual/eSE ones)
    print("\nQuantizer Status:")
    print_quantizer_status(model)

    # Save checkpoint + calibration cache (shared PTQ-producer helper)
    output_path, calib_path = save_ptq_checkpoint(model, settings["output"], calibrator)

    print("\n" + "=" * 80)
    print("PTQ Complete!")
    print(f"Model saved to: {output_path}")
    print(f"Calibration cache saved to: {calib_path}")
    print("=" * 80)


def run_qat(args):
    """Run QAT training pipeline (shared driver; CenterPoint supplies only project constants)."""
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
    print("CenterPoint QAT Training (frozen-amax STE fine-tune)")
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
        hook_import="deployment.projects.centerpoint.quantization.qat_hook",
        hook_type="QATHook",
        quant_config=config,
        epochs=settings["epochs"],
        lr=settings["lr"],
        output=settings["output"],
        batch_size=args.batch_size,
        calibration_batches=calibration_batches,
        calib_cache=settings["calib_cache"],
        work_dir=settings["work_dir"],
    )

    print("\n" + "=" * 80)
    print("QAT training completed!")
    print(f"Packaged checkpoint: {output_path}")
    print(f"Calibration cache: {calib_path}")
    print("Deploy it exactly like a PTQ checkpoint (same loader path; spec_qat.md D6).")
    print("=" * 80)


def main():
    """Main entry point."""
    from deployment.quantization.producer import init_quant_logging

    args = parse_args()

    # Initialize quantization library
    init_quant_logging()

    # Run the appropriate command
    if args.command == "ptq":
        run_ptq(args)
    elif args.command == "qat":
        run_qat(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
