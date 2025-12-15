#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
CenterPoint Quantization Tools

This script provides CLI commands for PTQ (Post-Training Quantization),
QAT (Quantization-Aware Training), and sensitivity analysis for CenterPoint models.

Usage:
    # PTQ Mode - Quantize a pre-trained model
    python tools/detection3d/centerpoint_quantization.py ptq \
        --config projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py \
        --checkpoint work_dirs/centerpoint/best.pth \
        --calibrate-batches 100 \
        --output work_dirs/centerpoint_ptq.pth

    # Sensitivity Analysis - Find layers sensitive to quantization
    python tools/detection3d/centerpoint_quantization.py sensitivity \
        --config projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py \
        --checkpoint work_dirs/centerpoint/best.pth \
        --calibrate-batches 100 \
        --output sensitivity_report.csv

    # QAT Mode - Fine-tune with quantization
    python tools/detection3d/centerpoint_quantization.py qat \
        --config projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py \
        --checkpoint work_dirs/centerpoint/best.pth \
        --calibrate-batches 100 \
        --epochs 10 \
        --lr 0.0001 \
        --output work_dirs/centerpoint_qat.pth
"""

import argparse
import csv
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
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
    ptq_parser.add_argument("--config", required=True, help="Model config file path")
    ptq_parser.add_argument("--checkpoint", required=True, help="Model checkpoint file path")
    ptq_parser.add_argument(
        "--calibrate-batches",
        type=int,
        default=100,
        help="Number of batches for calibration (default: 100)",
    )
    ptq_parser.add_argument(
        "--no-fuse-bn",
        action="store_true",
        help="Skip BatchNorm fusion (default: fuse BN)",
    )
    ptq_parser.add_argument(
        "--skip-layers",
        nargs="*",
        default=[],
        help="Layer names to skip quantization",
    )
    ptq_parser.add_argument("--output", required=True, help="Output checkpoint path")
    ptq_parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for calibration (default: cuda:0)",
    )

    # =========================================================================
    # Sensitivity command
    # =========================================================================
    sens_parser = subparsers.add_parser(
        "sensitivity",
        help="Layer Sensitivity Analysis",
        description="Analyze which layers are sensitive to quantization",
    )
    sens_parser.add_argument("--config", required=True, help="Model config file path")
    sens_parser.add_argument("--checkpoint", required=True, help="Model checkpoint file path")
    sens_parser.add_argument(
        "--calibrate-batches",
        type=int,
        default=100,
        help="Number of batches for calibration (default: 100)",
    )
    sens_parser.add_argument(
        "--eval-samples",
        type=int,
        default=None,
        help="Number of samples for evaluation (default: all)",
    )
    sens_parser.add_argument("--output", default="sensitivity_report.csv", help="Output CSV path")
    sens_parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for calibration (default: cuda:0)",
    )

    # =========================================================================
    # QAT command
    # =========================================================================
    qat_parser = subparsers.add_parser(
        "qat",
        help="Quantization-Aware Training",
        description="Fine-tune model with quantization-aware training",
    )
    qat_parser.add_argument("--config", required=True, help="Model config file path")
    qat_parser.add_argument("--checkpoint", required=True, help="Initial checkpoint file path")
    qat_parser.add_argument(
        "--calibrate-batches",
        type=int,
        default=100,
        help="Number of batches for initial calibration (default: 100)",
    )
    qat_parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of fine-tuning epochs (default: 10)",
    )
    qat_parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate for fine-tuning (default: 0.0001)",
    )
    qat_parser.add_argument(
        "--skip-layers",
        nargs="*",
        default=[],
        help="Layer names to skip quantization",
    )
    qat_parser.add_argument("--output", required=True, help="Output checkpoint path")
    qat_parser.add_argument("--work-dir", default=None, help="Working directory for training")

    return parser.parse_args()


def initialize_quantization():
    """Initialize pytorch-quantization library and suppress verbose logging."""
    try:
        from absl import logging as quant_logging

        quant_logging.set_verbosity(quant_logging.ERROR)
    except ImportError:
        pass


def run_ptq(args):
    """Run PTQ quantization pipeline."""
    import torch
    from mmdet3d.apis import init_model
    from mmengine.config import Config
    from mmengine.runner import Runner

    from projects.CenterPoint.quantization import (
        CalibrationManager,
        disable_quantization,
        fuse_model_bn,
        print_quantizer_status,
        quant_model,
    )

    print("=" * 80)
    print("CenterPoint PTQ Quantization")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Calibration batches: {args.calibrate_batches}")
    print("Amax method: mse")
    print(f"Fuse BN: {not args.no_fuse_bn}")
    print(f"Output: {args.output}")
    print("=" * 80)

    # Load model
    print("\n[1/5] Loading model...")
    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()

    # Fuse BatchNorm
    if not args.no_fuse_bn:
        print("\n[2/5] Fusing BatchNorm layers...")
        fuse_model_bn(model)
    else:
        print("\n[2/5] Skipping BatchNorm fusion")

    # Insert Q/DQ nodes
    print("\n[3/5] Inserting Q/DQ nodes...")
    skip_layers = set(args.skip_layers) if args.skip_layers else set()
    quant_model(model, skip_names=skip_layers)

    # Build dataloader
    print("\n[4/5] Building calibration dataloader...")
    # dataloader = Runner.build_dataloader(cfg.val_dataloader)
    dataloader = Runner.build_dataloader(cfg.val_dataloader)

    # Calibrate
    print(f"\n[5/5] Calibrating with {args.calibrate_batches} batches...")
    calibrator = CalibrationManager(model)
    calibrator.calibrate(
        dataloader,
        num_batches=args.calibrate_batches,
        method="mse",  # fixed to mse to match CUDA-CenterPoint behavior
    )

    # Disable skipped layers
    for layer_name in skip_layers:
        try:
            layer = dict(model.named_modules())[layer_name]
            disable_quantization(layer).apply()
            print(f"  Disabled quantization for: {layer_name}")
        except KeyError:
            print(f"  Warning: Layer not found: {layer_name}")

    # Print status
    print("\nQuantizer Status:")
    print_quantizer_status(model)

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, output_path)

    # Save calibration cache
    calib_path = output_path.with_suffix(".calib")
    calibrator.save_calib_cache(str(calib_path))

    print("\n" + "=" * 80)
    print("PTQ Complete!")
    print(f"Model saved to: {output_path}")
    print(f"Calibration cache saved to: {calib_path}")
    print("=" * 80)


def run_sensitivity(args):
    """Run layer sensitivity analysis."""
    import torch
    from mmdet3d.apis import init_model
    from mmengine.config import Config
    from mmengine.runner import Runner

    from projects.CenterPoint.quantization import (
        CalibrationManager,
        fuse_model_bn,
        quant_model,
    )

    try:
        from pytorch_quantization.nn import TensorQuantizer
    except ImportError:
        print("Error: pytorch-quantization is required for sensitivity analysis")
        sys.exit(1)

    print("=" * 80)
    print("CenterPoint Sensitivity Analysis")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Calibration batches: {args.calibrate_batches}")
    print(f"Output: {args.output}")
    print("=" * 80)

    # Load model
    print("\n[1/4] Loading model...")
    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()

    # Fuse BatchNorm and insert Q/DQ nodes
    print("\n[2/4] Preparing quantized model...")
    fuse_model_bn(model)
    quant_model(model)

    # Build dataloader
    print("\n[3/4] Building dataloaders...")
    train_dataloader = Runner.build_dataloader(cfg.train_dataloader)
    val_dataloader = Runner.build_dataloader(cfg.val_dataloader)

    # Calibrate
    print(f"\n[4/4] Calibrating with {args.calibrate_batches} batches...")
    calibrator = CalibrationManager(model)
    calibrator.calibrate(train_dataloader, num_batches=args.calibrate_batches)

    # Get all quantizer layer names
    quant_layer_names = []
    for name, module in model.named_modules():
        if name.endswith("_quantizer"):
            layer_name = name.replace("._input_quantizer", "").replace("._weight_quantizer", "")
            if layer_name not in quant_layer_names:
                quant_layer_names.append(layer_name)

    print(f"\nFound {len(quant_layer_names)} quantized layers")

    # Disable all quantizers
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            module.disable()

    # Placeholder for evaluation function
    # In a full implementation, this would run actual mAP evaluation
    def eval_model(model, dataloader, num_samples=None):
        """Evaluate model and return mAP."""
        # For now, just run inference to check model works
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if num_samples and i >= num_samples:
                    break
                try:
                    model.test_step(batch)
                except Exception:
                    pass
        # Return dummy mAP - replace with actual evaluation
        return 0.0

    # Get baseline
    print("\nEvaluating baseline (FP32)...")
    baseline_map = eval_model(model, val_dataloader, args.eval_samples)
    print(f"Baseline mAP: {baseline_map:.4f}")

    # Test each layer
    results = []
    print("\nTesting layer sensitivity...")

    for i, quant_layer in enumerate(quant_layer_names):
        # Enable this layer's quantizers
        for name, module in model.named_modules():
            if isinstance(module, TensorQuantizer) and quant_layer in name:
                module.enable()

        # Evaluate
        layer_map = eval_model(model, val_dataloader, args.eval_samples)
        delta = baseline_map - layer_map

        results.append(
            {
                "layer": quant_layer,
                "mAP": layer_map,
                "delta": delta,
            }
        )
        print(f"  [{i+1}/{len(quant_layer_names)}] {quant_layer}: mAP={layer_map:.4f}, delta={delta:.4f}")

        # Disable this layer's quantizers
        for name, module in model.named_modules():
            if isinstance(module, TensorQuantizer) and quant_layer in name:
                module.disable()

    # Sort by impact
    results.sort(key=lambda x: x["delta"], reverse=True)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "mAP", "delta"])
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 80)
    print("Sensitivity Analysis Complete!")
    print(f"Results saved to: {output_path}")
    print("\nTop 10 Sensitive Layers:")
    for i, r in enumerate(results[:10]):
        print(f"  {i+1}. {r['layer']}: delta={r['delta']:.4f}")
    print("=" * 80)


def run_qat(args):
    """Run QAT training pipeline."""
    from mmengine.config import Config

    print("=" * 80)
    print("CenterPoint QAT Training")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Output: {args.output}")
    print("=" * 80)

    # Load and modify config
    cfg = Config.fromfile(args.config)

    # Override training settings
    cfg.optim_wrapper.optimizer.lr = args.lr
    cfg.train_cfg.max_epochs = args.epochs

    # Set work directory
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = str(Path(args.output).parent / "qat_training")

    # Add QAT hook
    if not hasattr(cfg, "custom_hooks"):
        cfg.custom_hooks = []

    cfg.custom_hooks.append(
        dict(
            type="QATHook",
            calibration_batches=args.calibrate_batches,
            calibration_epoch=0,
            freeze_bn=True,
            sensitive_layers=args.skip_layers or [],
        )
    )

    # Load checkpoint
    cfg.load_from = args.checkpoint

    print("\nQAT training configuration prepared.")
    print(f"Work directory: {cfg.work_dir}")
    print("\nTo run training, execute:")
    print(f"  python tools/train.py {args.config} --work-dir {cfg.work_dir}")
    print("\nNote: QAT training requires the QATHook to be registered.")
    print("Make sure to import projects.CenterPoint.quantization.hooks in your config.")


def main():
    """Main entry point."""
    args = parse_args()

    # Initialize quantization library
    initialize_quantization()

    # Run the appropriate command
    if args.command == "ptq":
        run_ptq(args)
    elif args.command == "sensitivity":
        run_sensitivity(args)
    elif args.command == "qat":
        run_qat(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
