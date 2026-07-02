#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
CenterPoint Quantization Tools

This script provides CLI commands for PTQ (Post-Training Quantization) and
QAT (Quantization-Aware Training) for CenterPoint models.

Usage:
    # PTQ Mode - Quantize a pre-trained model (required: --deploy-cfg)
    python -m deployment.projects.centerpoint.quantization.quantize ptq \
        --config projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py \
        --checkpoint work_dirs/centerpoint/best.pth \
        --deploy-cfg deployment/projects/centerpoint/config/deploy_config_int8.py \
        --calibrate-samples 938 \
        --batch-size 4 \
        --calib-seed 0 \
        --output work_dirs/centerpoint_ptq.pth

    # PTQ Mode with larger batch_size (reduces seed sensitivity)
    python -m deployment.projects.centerpoint.quantization.quantize ptq \
        --config projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py \
        --checkpoint work_dirs/centerpoint/best.pth \
        --deploy-cfg deployment/projects/centerpoint/config/deploy_config_int8.py \
        --calibrate-samples 938 \
        --batch-size 16 \
        --calib-seed 0 \
        --output work_dirs/centerpoint_ptq.pth

    # QAT Mode - Fine-tune with quantization (required: --deploy-cfg)
    python -m deployment.projects.centerpoint.quantization.quantize qat \
        --config projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py \
        --checkpoint work_dirs/centerpoint/best.pth \
        --deploy-cfg deployment/projects/centerpoint/config/deploy_config_int8.py \
        --calibrate-samples 100 \
        --batch-size 1 \
        --epochs 10 \
        --lr 0.0001 \
        --output work_dirs/centerpoint_qat.pth
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

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
    ptq_parser.add_argument("--config", required=True, help="Model config file path")
    ptq_parser.add_argument("--checkpoint", required=True, help="Model checkpoint file path")
    ptq_parser.add_argument(
        "--deploy-cfg",
        required=True,
        help=(
            "Required deployment config path (e.g. deployment/projects/centerpoint/config/deploy_config_int8.py). "
            "PTQ uses its `quantization` settings as the single source of truth "
            "(sensitive_layers, quant_* flags, skip_backbone_* and fuse_bn)."
        ),
    )
    ptq_parser.add_argument(
        "--calibrate-samples",
        type=int,
        default=100,
        help=(
            "Total number of samples for calibration (default: 100). "
            "Batches will be auto-calculated based on --batch-size."
        ),
    )
    ptq_parser.add_argument("--output", required=True, help="Output checkpoint path")
    ptq_parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for calibration (default: cuda:0)",
    )
    ptq_parser.add_argument(
        "--calib-shuffle",
        action="store_true",
        help="Shuffle the calibration data",
    )
    ptq_parser.add_argument(
        "--calib-seed",
        type=int,
        default=None,
        help="Random seed for calibration data shuffling",
    )
    ptq_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for calibration (default: 1). Larger batch size can reduce seed sensitivity.",
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
        required=True,
        help=(
            "Required deployment config path (e.g. deployment/projects/centerpoint/config/deploy_config_int8.py). "
            "QAT uses its `quantization` settings as the single source of truth "
            "for sensitive layers and component quantization toggles."
        ),
    )
    qat_parser.add_argument(
        "--calibrate-samples",
        type=int,
        default=None,
        help=(
            "Total number of samples for initial calibration. "
            "If not specified, use all training batches (recommended for QAT). "
            "Batches will be auto-calculated based on --batch-size if specified."
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
        "--ptq-calib-cache",
        default=None,
        help="Path to PTQ calibration cache (.calib file). If provided, QAT will load "
        "existing amax values instead of running new calibration, significantly "
        "speeding up the process.",
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


def _build_ptq_quant_settings(args) -> Tuple[bool, Set[str], Dict[str, bool]]:
    """
    Build PTQ quantization settings from (optional) deploy config.

    Returns:
        fuse_bn: bool
        skip_layers: Set[str]
        quant_flags: Dict[str, bool] with keys:
            - quant_voxel_encoder
            - quant_backbone
            - quant_neck
            - quant_head
            - quant_add
            - quant_linear_backbone
            - quant_linear_backbone
            - quant_ese_mul_identity
            - quant_ese_pool_input
            - quant_maxpool_input
    """
    # Baseline: from deploy config if provided, otherwise defaults.
    fuse_bn = True
    skip_layers: Set[str] = set()
    quant_flags: Dict[str, bool] = {
        "quant_voxel_encoder": True,
        "quant_backbone": True,
        "quant_neck": True,
        "quant_head": True,
        "quant_add": False,  # Default to False for backward compatibility
        "quant_linear_backbone": False,  # ConvNeXt pointwise linear support
        "quant_ese_mul_identity": False,  # Quantize both inputs to eSE Mul (identity + gate) for INT8
        "quant_ese_pool_input": False,  # Q/DQ before pooling layer in eSE
        "quant_maxpool_input": False,  # Q/DQ before MaxPool2d (e.g. VoVNet _OSA_stage)
    }

    # Deploy config baseline
    if args.deploy_cfg:
        quant_cfg, _ = _load_deploy_quantization_cfg(args.deploy_cfg)

        # BN fusion baseline
        if "fuse_bn" in quant_cfg:
            fuse_bn = bool(quant_cfg.get("fuse_bn", True))

        # Quant flags baseline
        for k in list(quant_flags.keys()):
            if k in quant_cfg:
                quant_flags[k] = bool(quant_cfg[k])

        # Handle quant_add specifically (for ResNet-style backbones)
        if "quant_add" in quant_cfg:
            quant_flags["quant_add"] = bool(quant_cfg["quant_add"])
        if "quant_linear_backbone" in quant_cfg:
            quant_flags["quant_linear_backbone"] = bool(quant_cfg["quant_linear_backbone"])
        if "quant_ese_mul_identity" in quant_cfg:
            quant_flags["quant_ese_mul_identity"] = bool(quant_cfg["quant_ese_mul_identity"])
        if "quant_ese_pool_input" in quant_cfg:
            quant_flags["quant_ese_pool_input"] = bool(quant_cfg["quant_ese_pool_input"])
        if "quant_maxpool_input" in quant_cfg:
            quant_flags["quant_maxpool_input"] = bool(quant_cfg["quant_maxpool_input"])

        # Sensitive layers baseline (deployment terminology)
        skip_layers |= set(quant_cfg.get("sensitive_layers", []) or [])

        # Optional backbone stage skips (SECOND/ResNet use .blocks; VoVNet uses .stem, .stage2, .stage3, .stage4)
        skip_first = int(quant_cfg.get("skip_backbone_first_stages", 0) or 0)
        if skip_first > 0:
            for i in range(skip_first):
                skip_layers.add(f"pts_backbone.blocks.{i}")
        for i in quant_cfg.get("skip_backbone_stages", []) or []:
            skip_layers.add(f"pts_backbone.blocks.{int(i)}")

        # VoVNet-specific: skip backbone stages by index (0=stem, 1=stage2, 2=stage3, 3=stage4)
        vovnet_stages = quant_cfg.get("skip_vovnet_stages", None)
        if vovnet_stages is not None:
            _vovnet_names = ["stem", "stage2", "stage3", "stage4"]
            for idx in vovnet_stages:
                i = int(idx)
                if 0 <= i < len(_vovnet_names):
                    skip_layers.add(f"pts_backbone.{_vovnet_names[i]}")
    return fuse_bn, skip_layers, quant_flags


def run_ptq(args):
    """Run PTQ quantization pipeline."""
    import math

    import torch
    from mmdet3d.apis import init_model
    from mmengine.config import Config
    from mmengine.runner import Runner

    from deployment.quantization import (
        CalibrationManager,
        disable_quantization,
        fuse_model_bn,
        print_quantizer_status,
    )

    from .quant_model import quant_model

    # Auto-calculate calibrate_batches from calibrate_samples and batch_size
    args.calibrate_batches = math.ceil(args.calibrate_samples / args.batch_size)
    actual_samples = args.calibrate_batches * args.batch_size
    print(
        f"Auto-calculated: {args.calibrate_samples} samples → "
        f"{args.calibrate_batches} batches × {args.batch_size} = {actual_samples} samples"
    )

    print("=" * 80)
    print("CenterPoint PTQ Quantization")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Calibration batches: {args.calibrate_batches}")
    print(f"Batch size: {args.batch_size}")
    print(f"Calibration shuffle: {args.calib_shuffle}")
    if args.calib_seed is not None:
        print(f"Calibration seed: {args.calib_seed}")
    print("Amax method: mse")
    if args.deploy_cfg:
        print(f"Deploy cfg: {args.deploy_cfg}")
    print(f"Output: {args.output}")
    print("=" * 80)

    # Build quantization settings to show quant_add status
    _, _, quant_flags = _build_ptq_quant_settings(args)
    if quant_flags["quant_add"]:
        print("Note: Residual quantization enabled for residual connections (ResNet-style backbones)")
        print("      Using residual_quantizer (only quantizes identity branch, not conv path)")

    # Load model
    print("\n[1/5] Loading model...")
    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()

    # Build quantization settings
    fuse_bn, skip_layers, quant_flags = _build_ptq_quant_settings(args)

    if fuse_bn:
        print("\n[2/5] Fusing BatchNorm layers...")
        fuse_model_bn(model)
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
        quant_ese_mul_identity=quant_flags.get("quant_ese_mul_identity", False),
        quant_ese_pool_input=quant_flags.get("quant_ese_pool_input", False),
        quant_maxpool_input=quant_flags.get("quant_maxpool_input", False),
        skip_names=skip_layers,
    )

    if quant_flags["quant_add"]:
        print("  - Residual quantizer attached to residual blocks (BasicBlock/SparseBasicBlock)")
        print("    Only identity branch is quantized to enable TensorRT Conv+Add fusion")

    if quant_flags.get("quant_ese_mul_identity"):
        print("  - eSE Mul: both inputs quantized (identity + gate) for INT8 → Q-DQ on both sides before Mul")

    if quant_flags.get("quant_ese_pool_input"):
        print("  - eSE Pool: Q/DQ before avg_pool for INT8 (input -> QDQ -> avg_pool -> fc -> Hsigmoid)")

    if quant_flags.get("quant_maxpool_input"):
        print("  - MaxPool: Q/DQ before MaxPool2d for INT8 (e.g. VoVNet _OSA_stage)")

    # Build dataloader
    print("\n[4/5] Building calibration dataloader...")
    # Override batch_size for PTQ calibration (best-effort)
    if isinstance(cfg.val_dataloader, dict):
        cfg.val_dataloader["batch_size"] = args.batch_size

        # Handle shuffle + seed for calibration
        if args.calib_seed is not None:
            import random

            import numpy as np

            random.seed(args.calib_seed)
            np.random.seed(args.calib_seed)
            torch.manual_seed(args.calib_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.calib_seed)

        if args.calib_shuffle:
            # Remove existing sampler to allow shuffle (they are mutually exclusive)
            if "sampler" in cfg.val_dataloader:
                del cfg.val_dataloader["sampler"]
            cfg.val_dataloader["shuffle"] = True

    dataloader = Runner.build_dataloader(cfg.val_dataloader)

    # Print dataset size (best-effort)
    try:
        total_samples = len(dataloader.dataset)
        total_calib_samples = args.calibrate_batches * args.batch_size
        print(f"  Total samples in dataset: {total_samples}")
        print(f"  Total calibration samples: {total_calib_samples}")
    except Exception:
        pass

    # Calibrate
    total_calib_samples = args.calibrate_batches * args.batch_size
    print(f"\n[5/5] Calibrating with {args.calibrate_batches} batches ({total_calib_samples} samples)...")
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


def run_qat(args):
    """Run QAT training pipeline."""
    import math

    import torch
    from mmengine.config import Config

    # Auto-calculate calibrate_batches from calibrate_samples and batch_size
    # If not specified, use all training batches (handled inside QATHook)
    if args.calibrate_samples is not None:
        calibration_batches = math.ceil(args.calibrate_samples / args.batch_size)
    else:
        calibration_batches = 0

    print("=" * 80)
    print("CenterPoint QAT Training")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    if calibration_batches > 0:
        print(
            f"Calibration: {calibration_batches} batches "
            f"(from calibrate_samples={args.calibrate_samples}, batch_size={args.batch_size})"
        )
    else:
        print("Calibration: using all training batches (or loading calib cache if provided)")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Output: {args.output}")
    print("=" * 80)

    # Load and modify config
    cfg = Config.fromfile(args.config)

    # Ensure QATHook is registered
    if not hasattr(cfg, "custom_imports"):
        cfg.custom_imports = dict(imports=[], allow_failed_imports=False)
    if "imports" not in cfg.custom_imports:
        cfg.custom_imports["imports"] = []

    qat_hook_import = "deployment.projects.centerpoint.quantization.qat_hook"
    if qat_hook_import not in cfg.custom_imports["imports"]:
        cfg.custom_imports["imports"].append(qat_hook_import)

    try:
        __import__(qat_hook_import)
        print(f"  Imported QATHook module: {qat_hook_import}")
    except ImportError as e:
        raise ImportError(
            f"Failed to import QATHook module '{qat_hook_import}'. "
            f"Please ensure the module is available. Error: {e}"
        ) from e

    # Override training settings
    cfg.optim_wrapper.optimizer.lr = args.lr
    cfg.train_cfg.max_epochs = args.epochs

    # Check if AmpOptimWrapper is used but GPU is not available
    if cfg.optim_wrapper.get("type") == "AmpOptimWrapper":
        if not torch.cuda.is_available():
            print("Warning: AmpOptimWrapper detected but CUDA is not available.")
            print("Switching to OptimWrapper for CPU/GPU compatibility...")
            cfg.optim_wrapper.type = "OptimWrapper"
            cfg.optim_wrapper.pop("dtype", None)
            cfg.optim_wrapper.pop("loss_scale", None)
            print("Optimizer wrapper changed to: OptimWrapper")

    # Override dataloader batch_size (best-effort)
    if isinstance(getattr(cfg, "train_dataloader", None), dict):
        cfg.train_dataloader["batch_size"] = args.batch_size
    if isinstance(getattr(cfg, "val_dataloader", None), dict):
        cfg.val_dataloader["batch_size"] = args.batch_size

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
    quant_linear_backbone = False
    quant_backbone = True
    quant_neck = True
    quant_head = True
    quant_voxel_encoder = True

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

        # Read quantization toggle settings
        quant_add = bool(quant_cfg.get("quant_add", False))
        quant_linear_backbone = bool(quant_cfg.get("quant_linear_backbone", False))
        quant_backbone = bool(quant_cfg.get("quant_backbone", True))
        quant_neck = bool(quant_cfg.get("quant_neck", True))
        quant_head = bool(quant_cfg.get("quant_head", True))
        quant_voxel_encoder = bool(quant_cfg.get("quant_voxel_encoder", True))

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
            calibration_batches=calibration_batches,
            calibration_epoch=0,
            freeze_bn=True,
            sensitive_layers=deduped,
            quant_add=quant_add,
            quant_linear_backbone=quant_linear_backbone,
            quant_backbone=quant_backbone,
            quant_neck=quant_neck,
            quant_head=quant_head,
            quant_voxel_encoder=quant_voxel_encoder,
            calib_cache_path=args.ptq_calib_cache,
        )
    )

    # Load checkpoint
    cfg.load_from = args.checkpoint

    print("\nQAT training configuration prepared.")
    print(f"Work directory: {cfg.work_dir}")
    if args.ptq_calib_cache:
        print(f"Using PTQ calibration cache: {args.ptq_calib_cache}")
        print("Note: This will skip the initial calibration phase and use existing amax values.")
    print("\nStarting QAT training...")
    print("=" * 80)

    # Import custom modules before building runner (ensures registries are populated)
    if hasattr(cfg, "custom_imports") and "imports" in cfg.custom_imports:
        for module_path in cfg.custom_imports["imports"]:
            try:
                __import__(module_path)
                print(f"  Imported: {module_path}")
            except ImportError as e:
                if not cfg.custom_imports.get("allow_failed_imports", False):
                    raise ImportError(
                        f"Failed to import module '{module_path}'. "
                        f"Please ensure the module is available. Error: {e}"
                    ) from e
                else:
                    print(f"  Warning: Failed to import {module_path}: {e}")

    from mmengine.registry import RUNNERS
    from mmengine.runner import Runner

    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.train()

    print("\n" + "=" * 80)
    print("QAT training completed!")
    print(f"Model saved in: {cfg.work_dir}")
    print("=" * 80)


def main():
    """Main entry point."""
    args = parse_args()

    # Initialize quantization library
    initialize_quantization()

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
