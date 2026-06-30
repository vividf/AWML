#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
Simple submodule quantization (PTQ on SimpleOSA / Simple_eSE with random calibration).

Use this script to run PTQ on minimal submodules (osa / ese) with random tensors—
no dataset or full CenterPoint config required. Export to ONNX is done via
export_simple_submodule_onnx.py.

Usage:
    # PTQ on SimpleOSA
    python deployment/quantization/simple_quantization.py ptq-simple \
        --submodule osa \
        --deploy-cfg deployment/projects/centerpoint/config/deploy_config_int8_vov99.py \
        --calibrate-samples 64 --batch-size 2 --seed 0 \
        --output work_dirs/simple_osa_ptq.pth

    # PTQ on Simple_eSE
    python deployment/quantization/simple_quantization.py ptq-simple \
        --submodule ese \
        --deploy-cfg deployment/projects/centerpoint/config/deploy_config_int8_vov99.py \
        --calibrate-samples 64 --batch-size 2 --seed 0 \
        --output work_dirs/simple_ese_ptq.pth

    # Export to ONNX (separate script)
    python deployment/quantization/export_simple_submodule_onnx.py \
        --submodule osa \
        --checkpoint work_dirs/simple_osa_ptq.pth \
        --deploy-cfg deployment/projects/centerpoint/config/deploy_config_int8_vov99.py \
        --output work_dirs/simple_osa.onnx

    # PTQ and export for SimpleOSA3 (three OSA blocks, single Q at identity fork)
    python deployment/quantization/simple_quantization.py ptq-simple \
        --submodule osa3 \
        --deploy-cfg deployment/projects/centerpoint/config/deploy_config_int8_vov99.py \
        --calibrate-samples 64 --batch-size 2 --seed 0 \
        --output work_dirs/simple_osa3_ptq.pth
    python deployment/quantization/export_simple_submodule_onnx.py \
        --submodule osa3 \
        --checkpoint work_dirs/simple_osa3_ptq.pth \
        --deploy-cfg deployment/projects/centerpoint/config/deploy_config_int8_vov99.py \
        --output work_dirs/simple_osa3.onnx
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simple submodule quantization (PTQ on SimpleOSA / Simple_eSE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    ptq_simple_parser = subparsers.add_parser(
        "ptq-simple",
        help="PTQ on simple submodules (SimpleOSA or Simple_eSE) with random calibration data",
        description="Build SimpleOSA or Simple_eSE with random weights, insert Q/DQ, calibrate with random tensors, save. No dataset/info required.",
    )
    ptq_simple_parser.add_argument(
        "--submodule",
        required=True,
        choices=["osa", "osa3", "ese"],
        help="Submodule to test: 'osa' (one OSA block), 'osa3' (three OSA blocks, identity 3-way Q), or 'ese' (one eSE block)",
    )
    ptq_simple_parser.add_argument(
        "--deploy-cfg",
        required=True,
        help="Deployment config path for quantization flags (e.g. deploy_config_int8_vov99.py)",
    )
    ptq_simple_parser.add_argument(
        "--output", required=True, help="Output checkpoint path (e.g. work_dirs/simple_ese_ptq.pth)"
    )
    ptq_simple_parser.add_argument(
        "--calibrate-samples",
        type=int,
        default=64,
        help="Number of random samples for calibration (default: 64)",
    )
    ptq_simple_parser.add_argument("--batch-size", type=int, default=2, help="Batch size for calibration (default: 2)")
    ptq_simple_parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for weights and calibration (default: 0)"
    )
    ptq_simple_parser.add_argument("--device", default="cuda:0", help="Device (default: cuda:0)")

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
    """Load `quantization` dict and (optional) `checkpoint_path` from a deploy config file."""
    from mmengine.config import Config

    deploy_cfg = Config.fromfile(deploy_cfg_path)
    quant = dict(getattr(deploy_cfg, "quantization", {}) or {})
    ckpt = getattr(deploy_cfg, "checkpoint_path", None)
    return quant, ckpt


def run_ptq_simple(args):
    """PTQ on SimpleOSA or Simple_eSE with random calibration (no dataset)."""
    import torch

    from deployment.quantization import (
        CalibrationManager,
        fuse_model_bn,
        print_quantizer_status,
        quant_model,
    )
    from deployment.quantization.simple_submodules import (
        build_simple_model,
        get_simple_input_shape,
    )

    args.calibrate_batches = math.ceil(args.calibrate_samples / args.batch_size)
    total_samples = args.calibrate_batches * args.batch_size

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    try:
        import numpy as np

        np.random.seed(args.seed)
    except ImportError:
        pass
    try:
        import random

        random.seed(args.seed)
    except ImportError:
        pass

    print("=" * 80)
    print("PTQ-Simple: QDQ test on submodule with random calibration")
    print("=" * 80)
    print(f"Submodule: {args.submodule}")
    print(f"Deploy cfg: {args.deploy_cfg}")
    print(f"Calibration: {args.calibrate_batches} batches x {args.batch_size} = {total_samples} random samples")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output}")
    print("=" * 80)

    # Load quantization flags from deploy config
    quant_cfg, _ = _load_deploy_quantization_cfg(args.deploy_cfg)
    fuse_bn = bool(quant_cfg.get("fuse_bn", True))
    skip_layers = set(quant_cfg.get("sensitive_layers", []) or [])
    quant_flags = {
        "quant_backbone": True,
        "quant_neck": False,
        "quant_head": False,
        "quant_voxel_encoder": False,
        "quant_add": bool(quant_cfg.get("quant_add", False)),
        "quant_linear_backbone": bool(quant_cfg.get("quant_linear_backbone", False)),
        "quant_ese_mul_identity": bool(quant_cfg.get("quant_ese_mul_identity", False)),
        "quant_ese_pool_input": bool(quant_cfg.get("quant_ese_pool_input", False)),
        "quant_maxpool_input": bool(quant_cfg.get("quant_maxpool_input", False)),
    }

    # Build simple model (random weights)
    print("\n[1/5] Building simple model (random weights)...")
    model = build_simple_model(args.submodule, device=args.device)
    model.eval()

    if fuse_bn:
        print("\n[2/5] Fusing BatchNorm...")
        fuse_model_bn(model)
    else:
        print("\n[2/5] Skipping BatchNorm fusion")

    print("\n[3/5] Inserting Q/DQ nodes...")
    quant_model(
        model,
        quant_backbone=quant_flags["quant_backbone"],
        quant_neck=quant_flags["quant_neck"],
        quant_head=quant_flags["quant_head"],
        quant_voxel_encoder=quant_flags["quant_voxel_encoder"],
        quant_add=quant_flags["quant_add"],
        quant_linear_backbone=quant_flags["quant_linear_backbone"],
        quant_ese_mul_identity=quant_flags["quant_ese_mul_identity"],
        quant_ese_pool_input=quant_flags["quant_ese_pool_input"],
        quant_maxpool_input=quant_flags["quant_maxpool_input"],
        skip_names=skip_layers,
    )
    if quant_flags["quant_ese_mul_identity"]:
        print("  - eSE Mul: both inputs quantized (identity + gate)")
    if quant_flags["quant_ese_pool_input"]:
        print("  - eSE Pool: Q/DQ before avg_pool")
    if quant_flags["quant_maxpool_input"]:
        print("  - MaxPool: Q/DQ before MaxPool2d")

    # Random calibration dataloader (yields tensors)
    C, H, W = get_simple_input_shape(args.submodule)

    def random_batches():
        for _ in range(args.calibrate_batches):
            x = torch.randn(args.batch_size, C, H, W, device=args.device) * 0.5
            yield x

    print(
        f"\n[4/5] Calibrating with {args.calibrate_batches} batches (random tensors {args.batch_size}x{C}x{H}x{W})..."
    )
    calibrator = CalibrationManager(model)
    calibrator.calibrate(
        random_batches(),
        num_batches=args.calibrate_batches,
        method="mse",
        forward_fn=lambda m, batch: m(batch),
    )

    for layer_name in skip_layers:
        try:
            from deployment.quantization import disable_quantization

            layer = dict(model.named_modules())[layer_name]
            disable_quantization(layer).apply()
            print(f"  Disabled quantization for: {layer_name}")
        except KeyError:
            pass

    print("\nQuantizer Status:")
    print_quantizer_status(model)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, output_path)
    calib_path = output_path.with_suffix(".calib")
    calibrator.save_calib_cache(str(calib_path))

    print("\n" + "=" * 80)
    print("PTQ-Simple Complete!")
    print(f"Model saved to: {output_path}")
    print(f"Calibration cache saved to: {calib_path}")
    print("=" * 80)


def main():
    """Main entry point."""
    args = parse_args()

    initialize_quantization()

    if args.command == "ptq-simple":
        run_ptq_simple(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
