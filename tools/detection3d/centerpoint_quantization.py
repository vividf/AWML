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
from typing import Any, Dict, Optional, Set, Tuple

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
        "--deploy-cfg",
        default=None,
        help=(
            "Optional deployment config path (e.g. projects/CenterPoint/deploy/configs/deploy_config_int8.py). "
            "If provided, PTQ will use its `quantization` settings as the single source of truth "
            "(sensitive_layers, quant_* flags, skip_backbone_* and fuse_bn)."
        ),
    )
    ptq_parser.add_argument(
        "--calibrate-batches",
        type=int,
        default=100,
        help="Number of batches for calibration (default: 100)",
    )
    ptq_parser.add_argument("--output", required=True, help="Output checkpoint path")
    ptq_parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for calibration (default: cuda:0)",
    )
    ptq_parser.add_argument(
        "--fuse-bn-mode",
        choices=["new", "old"],
        default=None,
        help=("BatchNorm fusion mode override. " "'new' fuses Conv-BN and BN-Conv, 'old' fuses Conv-BN only."),
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
    sens_parser.add_argument(
        "--fuse-bn-mode",
        choices=["new", "old"],
        default="new",
        help="BatchNorm fusion mode for sensitivity run.",
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
        "--deploy-cfg",
        default=None,
        help=(
            "Optional deployment config path (e.g. projects/CenterPoint/deploy/configs/deploy_config_int8.py). "
            "If provided, QAT will use its `quantization` settings as the single source of truth "
            "for sensitive layers and component quantization toggles."
        ),
    )
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
    qat_parser.add_argument("--output", required=True, help="Output checkpoint path")
    qat_parser.add_argument("--work-dir", default=None, help="Working directory for training")
    qat_parser.add_argument(
        "--fuse-bn-mode",
        choices=["new", "old"],
        default=None,
        help=(
            "BatchNorm fusion mode override for QAT hook. "
            "'new' fuses Conv-BN and BN-Conv, 'old' fuses Conv-BN only."
        ),
    )

    return parser.parse_args()


def initialize_quantization():
    """Initialize pytorch-quantization library and suppress verbose logging."""
    try:
        from absl import logging as quant_logging

        quant_logging.set_verbosity(quant_logging.ERROR)
    except ImportError:
        pass


def _load_deploy_quantization_cfg(
    deploy_cfg_path: str,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Load `quantization` dict and (optional) `checkpoint_path` from a deploy config file.
    """
    from mmengine.config import Config

    deploy_cfg = Config.fromfile(deploy_cfg_path)
    quant = dict(getattr(deploy_cfg, "quantization", {}) or {})
    ckpt = getattr(deploy_cfg, "checkpoint_path", None)
    return quant, ckpt


def _build_ptq_quant_settings(args) -> Tuple[bool, str, Set[str], Dict[str, bool]]:
    """
    Build PTQ quantization settings from (optional) deploy config.

    Returns:
        fuse_bn: bool
        fuse_bn_mode: str
        skip_layers: Set[str]
        quant_flags: Dict[str, bool] with keys:
            - quant_voxel_encoder
            - quant_backbone
            - quant_neck
            - quant_head
            - quant_add
            - quant_linear_backbone
    """
    # Baseline: from deploy config if provided, otherwise defaults.
    fuse_bn = True
    fuse_bn_mode = "new"
    skip_layers: Set[str] = set()
    quant_flags: Dict[str, bool] = {
        "quant_voxel_encoder": True,
        "quant_backbone": True,
        "quant_neck": True,
        "quant_head": True,
        "quant_add": False,  # Default to False for backward compatibility
        "quant_linear_backbone": False,  # ConvNeXt pointwise linear support
    }

    # Deploy config baseline
    if args.deploy_cfg:
        quant_cfg, _ = _load_deploy_quantization_cfg(args.deploy_cfg)

        # BN fusion baseline
        if "fuse_bn" in quant_cfg:
            fuse_bn = bool(quant_cfg.get("fuse_bn", True))
        if "fuse_bn_mode" in quant_cfg:
            fuse_bn_mode = str(quant_cfg.get("fuse_bn_mode", "new")).lower()

        # Quant flags baseline
        for k in list(quant_flags.keys()):
            if k in quant_cfg:
                quant_flags[k] = bool(quant_cfg[k])

        # Handle quant_add specifically (for ResNet-style backbones)
        if "quant_add" in quant_cfg:
            quant_flags["quant_add"] = bool(quant_cfg["quant_add"])
        if "quant_linear_backbone" in quant_cfg:
            quant_flags["quant_linear_backbone"] = bool(quant_cfg["quant_linear_backbone"])

        # Sensitive layers baseline (deployment terminology)
        skip_layers |= set(quant_cfg.get("sensitive_layers", []) or [])

        # Optional backbone stage skips baseline (deployment terminology)
        skip_first = int(quant_cfg.get("skip_backbone_first_stages", 0) or 0)
        if skip_first > 0:
            for i in range(skip_first):
                skip_layers.add(f"pts_backbone.blocks.{i}")
        for i in quant_cfg.get("skip_backbone_stages", []) or []:
            skip_layers.add(f"pts_backbone.blocks.{int(i)}")

        # Optional ConvNeXt downsample skips (deployment terminology)
        # Useful when stride-2 downsample convs are PTQ-sensitive.
        if bool(quant_cfg.get("skip_backbone_downsample_all", False)):
            skip_layers.add("pts_backbone.downsample_layers")
        for i in quant_cfg.get("skip_backbone_downsample_layers", []) or []:
            skip_layers.add(f"pts_backbone.downsample_layers.{int(i)}")

    # CLI override has the highest priority
    if getattr(args, "fuse_bn_mode", None) is not None:
        fuse_bn_mode = args.fuse_bn_mode

    if fuse_bn_mode not in {"new", "old"}:
        raise ValueError(f"Unsupported fuse_bn_mode={fuse_bn_mode}. Expected 'new' or 'old'.")

    return fuse_bn, fuse_bn_mode, skip_layers, quant_flags


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
    if args.deploy_cfg:
        print(f"Deploy cfg: {args.deploy_cfg}")
    print(f"Output: {args.output}")
    print("=" * 80)

    # Build quantization settings to show quant_add status
    _, _, _, quant_flags = _build_ptq_quant_settings(args)
    if quant_flags["quant_add"]:
        print("Note: Residual quantization enabled for residual connections (ResNet-style backbones)")
        print("      Using residual_quantizer (only quantizes identity branch, not conv path)")

    # Load model
    print("\n[1/5] Loading model...")
    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()

    # Build quantization settings
    fuse_bn, fuse_bn_mode, skip_layers, quant_flags = _build_ptq_quant_settings(args)

    if fuse_bn:
        print(f"\n[2/5] Fusing BatchNorm layers... (mode={fuse_bn_mode})")
        fuse_model_bn(model, mode=fuse_bn_mode)
    else:
        print("\n[2/5] Skipping BatchNorm fusion")

    # Insert Q/DQ nodes
    print("\n[3/5] Inserting Q/DQ nodes...")
    quant_model(
        model,
        quant_backbone=quant_flags["quant_backbone"],
        quant_neck=quant_flags["quant_neck"],
        quant_head=quant_flags["quant_head"],
        quant_voxel_encoder=quant_flags["quant_voxel_encoder"],
        quant_add=quant_flags["quant_add"],
        quant_linear_backbone=quant_flags["quant_linear_backbone"],
        skip_names=skip_layers,
    )

    if quant_flags["quant_add"]:
        print("  - Residual quantizer attached to residual blocks (BasicBlock/SparseBasicBlock)")
        print("    Only identity branch is quantized to enable TensorRT Conv+Add fusion")

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

    # Debug: Check residual_quantizer status
    if quant_flags["quant_add"]:
        print("\nResidual Quantizer Status:")
        residual_count = 0
        for name, module in model.named_modules():
            if hasattr(module, "residual_quantizer"):
                residual_count += 1
                rq = module.residual_quantizer
                has_calibrator = hasattr(rq, "_calibrator") and rq._calibrator is not None
                has_amax = hasattr(rq, "_amax") and rq._amax is not None
                is_disabled = getattr(rq, "_disabled", False)
                print(f"  {name}.residual_quantizer:")
                print(f"    - Has calibrator: {has_calibrator}")
                print(f"    - Has amax: {has_amax}")
                print(f"    - Disabled: {is_disabled}")
                if has_amax:
                    print(
                        f"    - Amax value: {rq._amax.item() if rq._amax.numel() == 1 else f'[{rq._amax.numel()} elements]'}"
                    )
        print(f"  Total residual quantizers: {residual_count}")

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
    fuse_model_bn(model, mode=args.fuse_bn_mode)
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

    # Sensitive layers: use deploy config as the single source of truth (if provided).
    sensitive_layers = []
    quant_add = False
    fuse_bn_mode = "new"
    if args.deploy_cfg:
        quant_cfg, _ = _load_deploy_quantization_cfg(args.deploy_cfg)
        sensitive_layers = list(quant_cfg.get("sensitive_layers", []) or [])

        # Expand backbone stage skips to match PTQ behavior
        skip_first = int(quant_cfg.get("skip_backbone_first_stages", 0) or 0)
        if skip_first > 0:
            for i in range(skip_first):
                sensitive_layers.append(f"pts_backbone.blocks.{i}")
        for i in quant_cfg.get("skip_backbone_stages", []) or []:
            sensitive_layers.append(f"pts_backbone.blocks.{int(i)}")

        # If deploy config disables whole components, treat as sensitive roots
        if not bool(quant_cfg.get("quant_voxel_encoder", True)):
            sensitive_layers.append("pts_voxel_encoder")
        if not bool(quant_cfg.get("quant_backbone", True)):
            sensitive_layers.append("pts_backbone")
        if not bool(quant_cfg.get("quant_neck", True)):
            sensitive_layers.append("pts_neck")
        if not bool(quant_cfg.get("quant_head", True)):
            sensitive_layers.append("pts_bbox_head")

        # Read quant_add setting
        quant_add = bool(quant_cfg.get("quant_add", False))
        fuse_bn_mode = str(quant_cfg.get("fuse_bn_mode", "new")).lower()

    if args.fuse_bn_mode is not None:
        fuse_bn_mode = args.fuse_bn_mode

    if fuse_bn_mode not in {"new", "old"}:
        raise ValueError(f"Unsupported fuse_bn_mode={fuse_bn_mode}. Expected 'new' or 'old'.")

    # De-duplicate while preserving order
    deduped = []
    seen = set()
    for x in sensitive_layers:
        if x not in seen:
            deduped.append(x)
            seen.add(x)

    cfg.custom_hooks.append(
        dict(
            type="QATHook",
            calibration_batches=args.calibrate_batches,
            calibration_epoch=0,
            freeze_bn=True,
            fuse_bn_mode=fuse_bn_mode,
            sensitive_layers=deduped,
            quant_add=quant_add,
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
