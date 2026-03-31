#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
BEVFusion Quantization Tools

This script provides CLI commands for PTQ (Post-Training Quantization)
for BEVFusion models. It mirrors centerpoint_quantization.py but handles:
  - Dense parts (pts_backbone, pts_neck, bbox_head): pytorch_quantization Q/DQ
  - Sparse encoder (pts_middle_encoder): spconv BN fusion + manual calibration
  - Voxel encoder (pts_voxel_encoder): optional Linear quantization

Usage:
    # PTQ Mode - Quantize a pre-trained BEVFusion model
    python deployment/quantization/bevfusion_quantization.py ptq \
        --config projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
        --checkpoint work_dirs/bevfusion/epoch_30.pth \
        --deploy-cfg deployment/projects/bevfusion/config/deploy_config_int8.py \
        --calibrate-samples 256 \
        --batch-size 1 \
        --calib-seed 0 \
        --output work_dirs/bevfusion/epoch_30_ptq.pth
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Before any import that loads spconv.constants (prepare_fx / sparse layers need relaxed asserts).
import os

os.environ.setdefault("SPCONV_FX_TRACE_MODE", "1")


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
    ptq_parser.add_argument(
        "--skip-spconv-int8",
        action="store_true",
        help="Skip spconv INT8 calibration for sparse encoder (only do dense Q/DQ).",
    )
    ptq_parser.add_argument(
        "--spconv-calib-max-voxels",
        type=int,
        default=None,
        help=(
            "Max voxels per frame during spconv FX calibration (implicit_gemm uses ~O(N^2) GPU memory). "
            "Overrides deploy quantization.spconv_calib_max_voxels. If unset, uses deploy → "
            "SPCONV_CALIB_MAX_VOXELS → default 4096."
        ),
    )
    return parser.parse_args()


def _load_deploy_quantization_cfg(deploy_cfg_path: str) -> Tuple[Dict[str, Any], Optional[str]]:
    from mmengine.config import Config

    deploy_cfg = Config.fromfile(deploy_cfg_path)
    # MMEngine Config: prefer .get("quantization"); getattr() can miss keys on some Config wrappers.
    quant_raw = deploy_cfg.get("quantization", None)
    if quant_raw is None:
        quant_raw = getattr(deploy_cfg, "quantization", None)
    if quant_raw is None:
        quant = {}
    else:
        try:
            quant = {k: quant_raw[k] for k in quant_raw}
        except Exception:
            quant = dict(quant_raw)
    ckpt = deploy_cfg.get("checkpoint_path", None)
    if ckpt is None:
        ckpt = getattr(deploy_cfg, "checkpoint_path", None)
    return quant, ckpt


def _resolve_spconv_calib_voxel_cap(quant_cfg: Dict[str, Any], cli_cap: Optional[int]) -> int:
    """Effective voxels/sample for spconv FX calibration (prepare_fx observers).

    implicit_gemm / get_indice_pairs allocate buffers ~ O(N^2); full-scene N is not viable on GPU
    for this PyTorch FX path (unlike libspconv in Lidar AI Solution).

    Priority: CLI ``--spconv-calib-max-voxels`` > deploy ``spconv_calib_max_voxels`` >
    env ``SPCONV_CALIB_MAX_VOXELS`` > library default.
    Values ``<= 0`` fall back to the library default (never "unlimited" here — that OOMs).
    """
    from deployment.projects.bevfusion.quantization.spconv_int8 import _default_spconv_calib_max_voxels

    def _positive_or_default(v: int) -> int:
        return int(v) if int(v) > 0 else _default_spconv_calib_max_voxels()

    if cli_cap is not None:
        try:
            return _positive_or_default(int(cli_cap))
        except (TypeError, ValueError):
            return _default_spconv_calib_max_voxels()
    raw = quant_cfg.get("spconv_calib_max_voxels")
    if raw is not None:
        try:
            return _positive_or_default(int(raw))
        except (TypeError, ValueError):
            return _default_spconv_calib_max_voxels()
    return _default_spconv_calib_max_voxels()


def _build_ptq_quant_settings(args) -> Tuple[bool, Set[str], Dict[str, bool]]:
    """Build PTQ quantization settings from deploy config.

    BEVFusion uses bbox_head (not pts_bbox_head), so we handle
    component names explicitly.
    """
    fuse_bn = True
    skip_layers: Set[str] = set()
    quant_flags: Dict[str, bool] = {
        "quant_backbone": True,
        "quant_neck": True,
        "quant_head": True,
        "quant_add": False,
    }

    if args.deploy_cfg:
        quant_cfg, _ = _load_deploy_quantization_cfg(args.deploy_cfg)

        if "fuse_bn" in quant_cfg:
            fuse_bn = bool(quant_cfg.get("fuse_bn", True))

        for k in list(quant_flags.keys()):
            if k in quant_cfg:
                quant_flags[k] = bool(quant_cfg[k])

        skip_layers |= set(quant_cfg.get("sensitive_layers", []) or [])

    return fuse_bn, skip_layers, quant_flags


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


def _fuse_dense_bn(model):
    """Fuse BatchNorm in dense parts only (skip sparse encoder).

    BEVFusion dense parts: pts_backbone, pts_neck, bbox_head.
    """
    from deployment.quantization import fuse_model_bn

    for name in ["pts_backbone", "pts_neck", "bbox_head"]:
        submodule = getattr(model, name, None)
        if submodule is not None:
            submodule.eval()
            fuse_model_bn(submodule)
            print(f"  Fused BN in {name}")


def _fuse_spconv_bn(model):
    """Fuse BatchNorm in sparse encoder (pts_middle_encoder).

    Uses spconv's fuse_spconv_bn_eval for correct sparse weight permutation.
    """
    sparse_encoder = getattr(model, "pts_middle_encoder", None)
    if sparse_encoder is None:
        print("  No pts_middle_encoder found; skipping spconv BN fusion")
        return 0

    try:
        from spconv.pytorch.quantization.utils import fuse_spconv_bn_eval
    except ImportError:
        print("  spconv quantization utils not available; skipping spconv BN fusion")
        return 0

    import torch.nn as nn
    from spconv.pytorch.conv import SparseConvolution

    sparse_encoder.eval()
    fused_count = 0

    for name, module in sparse_encoder.named_modules():
        children = list(module._modules.items())
        for i in range(len(children) - 1):
            left_name, left_mod = children[i]
            right_name, right_mod = children[i + 1]

            if isinstance(left_mod, SparseConvolution) and isinstance(right_mod, nn.BatchNorm1d):
                fused_conv = fuse_spconv_bn_eval(left_mod, right_mod)
                setattr(module, left_name, fused_conv)
                setattr(module, right_name, nn.Identity())
                fused_count += 1

    print(f"  Fused {fused_count} SparseConv-BN pairs in pts_middle_encoder")
    return fused_count


def _insert_dense_qdq(model, quant_flags, skip_layers):
    """Insert Q/DQ nodes for dense parts of BEVFusion.

    BEVFusion uses bbox_head (not pts_bbox_head like CenterPoint),
    so we call quant_conv_module on individual submodules.
    """
    from deployment.quantization.replace import attach_quant_add, quant_conv_module

    if quant_flags["quant_backbone"] and hasattr(model, "pts_backbone"):
        quant_conv_module(model.pts_backbone, skip_layers, "pts_backbone")
        print(f"  Quantized pts_backbone (Conv2d -> QuantConv2d)")

    if quant_flags["quant_neck"] and hasattr(model, "pts_neck"):
        quant_conv_module(model.pts_neck, skip_layers, "pts_neck")
        print(f"  Quantized pts_neck (Conv2d -> QuantConv2d)")

    if quant_flags["quant_head"] and hasattr(model, "bbox_head"):
        quant_conv_module(model.bbox_head, skip_layers, "bbox_head")
        print(f"  Quantized bbox_head (Conv2d -> QuantConv2d)")

    if quant_flags["quant_add"]:
        attach_quant_add(model)
        print("  Attached residual quantizers")


def _calibrate_dense(model, dataloader, num_batches, method="mse"):
    """Run calibration for dense Q/DQ nodes using CalibrationManager."""
    from deployment.quantization import CalibrationManager

    def _force_float_voxel_inputs(batch):
        """Best-effort dtype normalization before test_step during calibration.

        Some dataloader/preprocessor paths may provide integer voxel features.
        When sparse encoder is already FX INT8-converted, quantize ops inside that
        graph require float activations and can raise:
        "Quantize only works on Float Tensor, got Int".
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


def _calibrate_spconv(
    model,
    dataloader,
    num_samples,
    device,
    output_path,
    quant_cfg,
    *,
    spconv_calib_max_voxels_cli: Optional[int] = None,
):
    """Run spconv INT8 for sparse encoder via FX path (prepare_fx → calibrate → convert_fx).

    Requires pts_middle_encoder to be FX-traceable (use config with
    block_type='basicblock_fx', e.g. bevfusion_*_120m_fx.py). Replaces
    model.pts_middle_encoder with the quantized module so the saved PTQ
    checkpoint contains the INT8 sparse encoder.
    """
    import torch

    from deployment.projects.bevfusion.quantization.spconv_int8 import (
        apply_spconv_int8_quantization,
        calibrate_spconv_model,
        convert_spconv_int8,
    )

    sparse_encoder = getattr(model, "pts_middle_encoder", None)
    if sparse_encoder is None:
        print("  No pts_middle_encoder; skipping spconv INT8 calibration")
        return

    calibration_data = []
    actual = min(num_samples, len(dataloader.dataset))
    print(f"  Collecting {actual} voxelized samples for spconv calibration...")

    for i, batch in enumerate(dataloader):
        if len(calibration_data) >= actual:
            break
        try:
            points_list = batch.get("inputs", {}).get("points", None)
            if points_list is None:
                data_samples = batch.get("data_samples", [])
                if data_samples and hasattr(data_samples[0], "point_cloud"):
                    continue
                continue

            for points in points_list:
                if len(calibration_data) >= actual:
                    break
                if not isinstance(points, torch.Tensor):
                    points = torch.from_numpy(points)
                points = points.to(device).float()

                with torch.no_grad():
                    ret = model.pts_voxel_layer(points)
                    if len(ret) == 3:
                        feats, coords, sizes = ret
                    else:
                        feats, coords = ret
                        sizes = None

                    batch_coors = torch.zeros(coords.shape[0], 1, device=device, dtype=coords.dtype)
                    coords = torch.cat([batch_coors, coords], dim=1).contiguous()

                    if sizes is not None and getattr(model, "voxelize_reduce", True):
                        feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                        feats = feats.contiguous()

                    calibration_data.append((feats, coords.int(), 1))

        except Exception as e:
            print(f"  Warning: batch {i} failed: {e}")
            continue

    if not calibration_data:
        print("  No calibration data for spconv; skipping")
        return

    print(f"  Collected {len(calibration_data)} samples")

    calib_cap = _resolve_spconv_calib_voxel_cap(quant_cfg, spconv_calib_max_voxels_cli)
    print(
        f"  Spconv FX calibration: max {calib_cap} voxels/sample "
        "(implicit_gemm ~O(N^2); order: --spconv-calib-max-voxels > deploy "
        "spconv_calib_max_voxels > SPCONV_CALIB_MAX_VOXELS > default)."
    )

    in_channels = getattr(sparse_encoder, "in_channels", 5)
    # Must match deployment ``_replace_encoder_with_fx_converted_structure`` (model_loader): deploy
    # calls ``upgrade_pts_middle_encoder_basicblocks_to_fx`` before prepare_fx when
    # ``spconv_ptq_basicblock_fx`` is True. Skipping here yields PTQ checkpoints whose state_dict
    # keys (e.g. ``..._bn1_scale_0``) do not load into the deploy-rebuilt GraphModule (missing
    # ``bn1.weight``, unexpected scale tensors) → mAP collapse and false INT8 verification.
    use_fx_bb = bool(quant_cfg.get("spconv_ptq_basicblock_fx", True))
    if use_fx_bb:
        from deployment.projects.bevfusion.quantization.spconv_int8 import (
            upgrade_pts_middle_encoder_basicblocks_to_fx,
        )

        n_bb = upgrade_pts_middle_encoder_basicblocks_to_fx(sparse_encoder)
        if n_bb:
            print(
                f"  Upgraded {n_bb} SparseBasicBlock → SparseBasicBlockFX before spconv FX "
                "(matches deploy loader; required unless checkpoint uses only basicblock_fx config)."
            )
    else:
        print("  spconv_ptq_basicblock_fx=False: skipping SparseBasicBlock→FX upgrade (legacy checkpoint path)")

    try:
        print("  Applying spconv FX path: prepare_fx → calibrate → convert_fx → transform_qdq")
        prepared = apply_spconv_int8_quantization(sparse_encoder, torch.device(device), in_channels=in_channels)
        calibrate_spconv_model(prepared, calibration_data, max_voxels_per_sample=calib_cap)
        converted = convert_spconv_int8(prepared, attr_source=sparse_encoder)
        model.pts_middle_encoder = converted
        print("  Sparse encoder replaced with INT8 quantized module (FX path)")
    except Exception as e:
        print(f"  Spconv FX INT8 failed: {e}")
        print("  Ensure config uses block_type='basicblock_fx' (e.g. bevfusion_*_120m_fx.py)")
        import traceback

        traceback.print_exc()


def _disable_sensitive_layers(model, skip_layers):
    """Disable quantization for sensitive layers."""
    from deployment.quantization import disable_quantization

    for layer_name in skip_layers:
        try:
            layer = dict(model.named_modules())[layer_name]
            disable_quantization(layer).apply()
            print(f"  Disabled quantization for: {layer_name}")
        except KeyError:
            print(f"  Warning: Layer not found: {layer_name}")


def run_ptq(args):
    """Run BEVFusion PTQ pipeline."""
    import math

    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner

    num_batches = math.ceil(args.calibrate_samples / args.batch_size)
    actual_samples = num_batches * args.batch_size

    print("=" * 80)
    print("BEVFusion PTQ Quantization")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Deploy cfg: {args.deploy_cfg}")
    print(f"Calibration: {args.calibrate_samples} samples -> {num_batches} batches x {args.batch_size}")
    print(f"Actual calibration samples: {actual_samples}")
    print(f"Skip spconv INT8: {args.skip_spconv_int8}")
    if args.calib_seed is not None:
        print(f"Calibration seed: {args.calib_seed}")
    print(f"Output: {args.output}")
    print("=" * 80)

    fuse_bn, skip_layers, quant_flags = _build_ptq_quant_settings(args)

    # [1/6] Load model
    print("\n[1/6] Loading BEVFusion model...")
    model = _build_bevfusion_model(args.config, args.checkpoint, args.device)

    # [2/6] Fuse BN
    if fuse_bn:
        print("\n[2/6] Fusing BatchNorm layers...")
        _fuse_dense_bn(model)
        _fuse_spconv_bn(model)
    else:
        print("\n[2/6] Skipping BatchNorm fusion")

    # [3/6] Insert Q/DQ nodes for dense parts
    print("\n[3/6] Inserting Q/DQ nodes (dense parts)...")
    _insert_dense_qdq(model, quant_flags, skip_layers)

    # [4/6] Build dataloader + calibrate dense Q/DQ
    print("\n[4/6] Building calibration dataloader...")
    cfg = Config.fromfile(args.config)
    if isinstance(cfg.val_dataloader, dict):
        cfg.val_dataloader["batch_size"] = args.batch_size
        cfg.val_dataloader["num_workers"] = min(cfg.val_dataloader.get("num_workers", 4), 4)
        cfg.val_dataloader["persistent_workers"] = False

        if args.calib_seed is not None:
            import random

            import numpy as np

            random.seed(args.calib_seed)
            np.random.seed(args.calib_seed)
            torch.manual_seed(args.calib_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.calib_seed)

        if args.calib_shuffle:
            if "sampler" in cfg.val_dataloader:
                del cfg.val_dataloader["sampler"]
            cfg.val_dataloader["shuffle"] = True

    dataloader = Runner.build_dataloader(cfg.val_dataloader)
    total_ds = len(dataloader.dataset)
    print(f"  Dataset size: {total_ds}")
    print(f"  Calibration: {num_batches} batches x {args.batch_size} = {actual_samples} samples")

    # Spconv INT8 *before* dense CalibrationManager stats. If dense is calibrated while sparse is still
    # FP32, TensorQuantizer amax matches the wrong BEV distribution → inference (INT8 sparse → dense)
    # gets OOD activations and mAP can collapse (~0) no matter how many spconv voxel caps you use.
    quant_cfg_for_sparse: Dict[str, Any] = {}
    if not args.skip_spconv_int8:
        quant_cfg_for_sparse, _ = _load_deploy_quantization_cfg(args.deploy_cfg)
        if quant_cfg_for_sparse.get("spconv_int8", False):
            print("\n[4b/6] Spconv INT8 for sparse encoder (FX path, runs before dense PTQ calib)...")
            spconv_samples = min(total_ds, int(args.calibrate_samples))
            print(
                f"  Spconv calibration frames: {spconv_samples} "
                f"(dataset {total_ds}; same as --calibrate-samples={args.calibrate_samples})"
            )
            try:
                _calibrate_spconv(
                    model,
                    dataloader,
                    spconv_samples,
                    args.device,
                    args.output,
                    quant_cfg_for_sparse,
                    spconv_calib_max_voxels_cli=args.spconv_calib_max_voxels,
                )
            except Exception as e:
                print(f"  Spconv INT8 failed: {e}")
                print("  Sparse encoder will remain FP32 in the PTQ checkpoint")
                import traceback

                traceback.print_exc()
        else:
            print("\n[4b/6] Spconv INT8 off in deploy config (spconv_int8=False); dense calib uses FP32 sparse.")
    else:
        print("\n[4b/6] Skipping spconv INT8 (--skip-spconv-int8); dense calib uses FP32 sparse.")

    print(
        f"\n[5/6] Calibrating dense Q/DQ nodes ({num_batches} batches, method=mse) "
        f"with current sparse encoder (INT8 if step 4b succeeded)..."
    )
    calibrator = _calibrate_dense(model, dataloader, num_batches, method="mse")

    if skip_layers:
        print(f"\n  Disabling {len(skip_layers)} sensitive layers...")
        _disable_sensitive_layers(model, skip_layers)

    # [6/6] Print status and save
    print("\n[6/6] Saving PTQ checkpoint...")

    try:
        from deployment.quantization import print_quantizer_status

        print("\nQuantizer Status:")
        print_quantizer_status(model)
    except Exception:
        pass

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, output_path)

    calib_path = output_path.with_suffix(".calib")
    try:
        calibrator.save_calib_cache(str(calib_path))
    except Exception as e:
        print(f"  Warning: could not save calib cache: {e}")

    print("\n" + "=" * 80)
    print("BEVFusion PTQ Complete!")
    print(f"Model saved to: {output_path}")
    if calib_path.exists():
        print(f"Calibration cache saved to: {calib_path}")
    print("=" * 80)
    print("\nTo use this PTQ checkpoint for deployment:")
    print(f'  1. Set checkpoint_path = "{args.output}" in your deploy config')
    print(f"  2. Set quantization.ptq_checkpoint = True")
    print(f"  3. Run:")
    print(f"     python -m deployment.cli.main bevfusion \\")
    print(f"       deployment/projects/bevfusion/config/deploy_config_int8.py \\")
    print(f"       {args.config} --module main_body")


def main():
    args = parse_args()

    try:
        from absl import logging as quant_logging

        quant_logging.set_verbosity(quant_logging.ERROR)
    except ImportError:
        pass

    if args.command == "ptq":
        run_ptq(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
