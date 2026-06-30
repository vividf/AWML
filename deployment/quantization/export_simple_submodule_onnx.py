#!/usr/bin/env python
"""
Simple ONNX export for SimpleOSA / Simple_eSE submodules.

Uses the same export logic as AWML (torch.onnx.export with deploy_config onnx_config).
Load a PTQ-simple checkpoint and export the submodule to ONNX for testing QDQ.

Usage:
    # Export Simple_eSE (after ptq-simple for ese)
    python deployment/quantization/export_simple_submodule_onnx.py \
        --submodule ese \
        --checkpoint work_dirs/simple_ese_ptq.pth \
        --deploy-cfg deployment/projects/centerpoint/config/deploy_config_int8_vov99.py \
        --output work_dirs/simple_ese.onnx

    # Export SimpleOSA (after ptq-simple for osa)
    python deployment/quantization/export_simple_submodule_onnx.py \
        --submodule osa \
        --checkpoint work_dirs/simple_osa_ptq.pth \
        --deploy-cfg deployment/projects/centerpoint/config/deploy_config_int8_vov99.py \
        --output work_dirs/simple_osa.onnx
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Project root
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch


def _load_deploy_quantization_cfg(deploy_cfg_path: str):
    from mmengine.config import Config

    cfg = Config.fromfile(deploy_cfg_path)
    return dict(getattr(cfg, "quantization", {}) or {}), getattr(cfg, "onnx_config", None) or {}


def main():
    parser = argparse.ArgumentParser(
        description="Export SimpleOSA or Simple_eSE to ONNX (same logic as AWML deployment)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--submodule", required=True, choices=["osa", "osa3", "ese"], help="Submodule: osa, osa3, or ese"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to PTQ-simple checkpoint (.pth)")
    parser.add_argument(
        "--deploy-cfg",
        required=True,
        help="Deploy config path (for quantization flags and onnx_config)",
    )
    parser.add_argument("--output", required=True, help="Output ONNX path (e.g. work_dirs/simple_ese.onnx)")
    parser.add_argument("--device", default="cpu", help="Device for export (default: cpu)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for export (default: 1)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    quant_cfg, onnx_cfg = _load_deploy_quantization_cfg(args.deploy_cfg)
    if not onnx_cfg:
        onnx_cfg = {
            "opset_version": 22,
            "do_constant_folding": True,
            "export_params": True,
            "keep_initializers_as_inputs": False,
            "simplify": False,
        }
    opset_version = int(onnx_cfg.get("opset_version", 22))
    # Legacy torch.onnx.export supports up to opset 20; cap to avoid warning/failure
    if opset_version > 20:
        logger.info("Capping opset_version to 20 (legacy ONNX exporter max)")
        opset_version = 20
    do_constant_folding = bool(onnx_cfg.get("do_constant_folding", True))
    export_params = bool(onnx_cfg.get("export_params", True))
    keep_initializers_as_inputs = bool(onnx_cfg.get("keep_initializers_as_inputs", False))
    simplify = bool(onnx_cfg.get("simplify", False))

    from deployment.projects.centerpoint.io.model_loader import (
        _move_quantizer_amax_to_device,
        setup_quantization_for_onnx_export,
    )
    from deployment.quantization import fuse_model_bn, quant_model
    from deployment.quantization.simple_submodules import (
        build_simple_model,
        get_simple_input_shape,
    )

    logger.info("=" * 60)
    logger.info("Export Simple submodule to ONNX (AWML-compatible)")
    logger.info("=" * 60)
    logger.info("Submodule: %s", args.submodule)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Output: %s", args.output)
    logger.info("Opset: %s, constant_folding: %s", opset_version, do_constant_folding)

    # 1. Build model (same structure as ptq-simple)
    model = build_simple_model(args.submodule, device=args.device)
    if quant_cfg.get("fuse_bn", True):
        model.eval()
        fuse_model_bn(model)
    quant_model(
        model,
        quant_backbone=True,
        quant_neck=False,
        quant_head=False,
        quant_voxel_encoder=False,
        quant_add=bool(quant_cfg.get("quant_add", False)),
        quant_linear_backbone=bool(quant_cfg.get("quant_linear_backbone", False)),
        quant_ese_mul_identity=bool(quant_cfg.get("quant_ese_mul_identity", False)),
        quant_ese_pool_input=bool(quant_cfg.get("quant_ese_pool_input", False)),
        quant_maxpool_input=bool(quant_cfg.get("quant_maxpool_input", False)),
        skip_names=set(quant_cfg.get("sensitive_layers", []) or []),
    )

    # 2. Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Ensure model and all quantizer amax are on export device (avoid scale on cuda / input on cpu)
    device = torch.device(args.device)
    model = model.to(device)
    _move_quantizer_amax_to_device(model, args.device)

    # 3. QDQ export mode (same as deployment)
    setup_quantization_for_onnx_export()

    # 4. Submodule to export (pts_backbone = SimpleOSA or Simple_eSE)
    submodule = model.pts_backbone
    C, H, W = get_simple_input_shape(args.submodule)
    dummy_input = torch.randn(args.batch_size, C, H, W, device=device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {"input": {0: "batch_size", 2: "H", 3: "W"}, "output": {0: "batch_size", 2: "H", 3: "W"}}

    logger.info("Exporting to ONNX...")
    logger.info("  Input shape: %s", list(dummy_input.shape))
    logger.info("  Input names: %s", input_names)
    logger.info("  Output names: %s", output_names)

    with torch.no_grad():
        torch.onnx.export(
            submodule,
            dummy_input,
            str(output_path),
            export_params=export_params,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False,
        )

    if simplify:
        try:
            import onnx
            import onnxsim

            logger.info("Simplifying ONNX...")
            model_onnx, success = onnxsim.simplify(str(output_path))
            if success:
                onnx.save(model_onnx, str(output_path))
                logger.info("Simplified successfully")
            else:
                logger.warning("Simplification failed")
        except Exception as e:
            logger.warning("Simplification error: %s", e)

    logger.info("Done: %s", output_path)


if __name__ == "__main__":
    main()
