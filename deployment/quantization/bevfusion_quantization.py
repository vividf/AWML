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

    # PTQ sparse encoder only (spconv INT8; dense stays FP32). Deploy eval must set
    # quant_backbone/neck/head=False. Use deploy cfg with spconv_int8=True.
    python deployment/quantization/bevfusion_quantization.py ptq ... \
        --deploy-cfg deployment/projects/bevfusion/config/deploy_config_split_int8.py \
        --sparse-int8-only --output work_dirs/bevfusion/epoch_30_ptq_sparse_only.pth
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
        "--sparse-int8-only",
        action="store_true",
        help=(
            "PTQ only pts_middle_encoder (spconv INT8). Skips dense pytorch_quantization Q/DQ "
            "(no QuantConv2d). Deploy eval must set quant_backbone/neck/head=False (and ptq_checkpoint=True, "
            "spconv_int8=True) to load this checkpoint. Requires quantization.spconv_int8=True in deploy cfg."
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

    # Hoist top-level ``spconv_int8_fp16_layers`` (kept top-level for consistency with
    # ``spconv_do_sort``) into the returned quant dict so downstream PTQ/loader code
    # only has to look in ONE place. Entries are substring-matched against
    # ``named_modules()`` names in ``apply_nvidia_spconv_int8(exclude_patterns=...)``.
    if "spconv_int8_fp16_layers" not in quant:
        fp16_layers = deploy_cfg.get("spconv_int8_fp16_layers", None)
        if fp16_layers is None:
            fp16_layers = getattr(deploy_cfg, "spconv_int8_fp16_layers", None)
        if fp16_layers is not None:
            try:
                quant["spconv_int8_fp16_layers"] = list(fp16_layers)
            except TypeError:
                quant["spconv_int8_fp16_layers"] = []

    ckpt = deploy_cfg.get("checkpoint_path", None)
    if ckpt is None:
        ckpt = getattr(deploy_cfg, "checkpoint_path", None)
    return quant, ckpt


def _report_converted_scale_buffers(converted_encoder) -> None:
    """After convert_fx, report scale/zero_point buffers saved in the encoder."""
    import torch

    scale_bufs = {}
    zp_bufs = {}
    for name, buf in converted_encoder.named_buffers():
        if "scale" in name:
            scale_bufs[name] = buf
        if "zero_point" in name:
            zp_bufs[name] = buf

    print(
        f"  [ptq-scale-check] Converted encoder has {len(scale_bufs)} scale buffers, "
        f"{len(zp_bufs)} zero_point buffers"
    )

    n_valid = 0
    n_default = 0
    for name, buf in scale_bufs.items():
        val = float(buf.flatten()[0]) if buf.numel() > 0 else -1.0
        if abs(val - 1.0) < 1e-6 or val <= 0:
            n_default += 1
            print(f"    WARNING: {name} = {val:.6f} (looks uncalibrated!)")
        else:
            n_valid += 1
            if n_valid <= 5:
                print(f"    {name} = {val:.6f}")

    print(f"  [ptq-scale-check] {n_valid} valid scales, {n_default} uncalibrated (=1.0)")

    total_params = sum(p.numel() for p in converted_encoder.parameters())
    total_bufs = sum(b.numel() for b in converted_encoder.buffers())
    print(f"  [ptq-model-check] params={total_params}, buffers={total_bufs}")

    sd = converted_encoder.state_dict()
    sparse_scale_keys = [k for k in sd if "scale" in k or "zero_point" in k]
    print(f"  [ptq-model-check] state_dict has {len(sd)} keys, {len(sparse_scale_keys)} are scale/zp")
    if sparse_scale_keys:
        print(f"  [ptq-model-check] sample scale/zp keys: {sparse_scale_keys[:5]}")


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

    if getattr(args, "sparse_int8_only", False):
        quant_flags["quant_backbone"] = False
        quant_flags["quant_neck"] = False
        quant_flags["quant_head"] = False
        quant_flags["quant_add"] = False

    return fuse_bn, skip_layers, quant_flags


def _dense_quant_enabled(quant_flags: Dict[str, bool]) -> bool:
    return bool(
        quant_flags.get("quant_backbone")
        or quant_flags.get("quant_neck")
        or quant_flags.get("quant_head")
        or quant_flags.get("quant_add")
    )


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
    import torch

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
):
    """Run spconv INT8 for sparse encoder using NVIDIA TensorQuantizer (histogram + MSE).

    Adapted from CUDA-BEVFusion: each SparseConvolution gets ``_input_quantizer``
    and ``_weight_quantizer`` (NVIDIA TensorQuantizer). Calibration collects
    histograms, then ``compute_amax(method=mse)`` picks the optimal clipping.
    """
    import torch

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
                        sz = sizes.type_as(feats).view(-1, 1).clamp(min=1.0)
                        feats = feats.sum(dim=1, keepdim=False) / sz
                        feats = feats.contiguous()

                    n_vox = int(feats.shape[0])
                    print(
                        f"  [voxel-stats] Sample {len(calibration_data)+1}: {n_vox} voxels, "
                        f"feats shape={tuple(feats.shape)}, coords shape={tuple(coords.shape)}"
                    )
                    calibration_data.append((feats, coords.int(), 1))

        except Exception as e:
            raise RuntimeError(f"spconv INT8 calibration: batch {i} failed while collecting voxel samples") from e

    if not calibration_data:
        raise RuntimeError("spconv INT8 calibration: no voxel samples collected (check dataloader / points inputs).")

    voxel_counts = [int(f.shape[0]) for f, _, _ in calibration_data]
    print(f"  Collected {len(calibration_data)} samples")
    print(
        f"  [voxel-stats] Voxel counts: min={min(voxel_counts)}, max={max(voxel_counts)}, "
        f"mean={sum(voxel_counts)/len(voxel_counts):.0f}, median={sorted(voxel_counts)[len(voxel_counts)//2]}"
    )

    from deployment.projects.bevfusion.quantization.spconv_int8 import (
        apply_nvidia_spconv_int8,
        calibrate_spconv_nvidia,
    )

    fp16_layers: List[str] = list(quant_cfg.get("spconv_int8_fp16_layers", []) or [])
    if fp16_layers:
        print(f"  [nvidia-quant] spconv_int8_fp16_layers active ({len(fp16_layers)} pattern(s)): {fp16_layers}")
        print(
            "  [nvidia-quant] These modules will NOT get _input_quantizer/_weight_quantizer → "
            "downstream PTQ _amax is calibrated against TRUE FP activations (no fake-quant contamination)."
        )

    print("  Applying NVIDIA TensorQuantizer path (histogram + MSE, adapted from CUDA-BEVFusion)")
    apply_nvidia_spconv_int8(
        sparse_encoder,
        exclude_conv_out=True,
        exclude_patterns=fp16_layers,
    )
    calibrate_spconv_nvidia(sparse_encoder, calibration_data)

    amax_keys = [k for k in sparse_encoder.state_dict() if "_amax" in k]
    print(f"  [save-check] {len(amax_keys)} _amax keys in sparse encoder state_dict")
    if amax_keys:
        sd = sparse_encoder.state_dict()
        for k in amax_keys[:5]:
            v = sd[k]
            print(f"    {k} = {v.flatten().tolist()[:3]}")

    print("  Sparse encoder calibrated with NVIDIA approach (histogram + MSE)")


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
    print(f"Sparse INT8 only (no dense Q/DQ): {getattr(args, 'sparse_int8_only', False)}")
    if args.calib_seed is not None:
        print(f"Calibration seed: {args.calib_seed}")
    print(f"Output: {args.output}")
    print("=" * 80)

    fuse_bn, skip_layers, quant_flags = _build_ptq_quant_settings(args)
    dense_on = _dense_quant_enabled(quant_flags)

    if getattr(args, "sparse_int8_only", False):
        print("\n  --sparse-int8-only: dense backbone/neck/head will stay FP32 (no QuantConv2d in .pth).")
    if getattr(args, "sparse_int8_only", False) and args.skip_spconv_int8:
        print("  Warning: --sparse-int8-only with --skip-spconv-int8 → no INT8 path is calibrated.")
    if getattr(args, "sparse_int8_only", False):
        qwarn, _ = _load_deploy_quantization_cfg(args.deploy_cfg)
        if not qwarn.get("spconv_int8", False):
            print(
                "  Warning: deploy quantization.spconv_int8=False; spconv INT8 step will be skipped. "
                "Use a deploy cfg with spconv_int8=True for sparse-only PTQ."
            )

    # [1/6] Load model
    print("\n[1/6] Loading BEVFusion model...")
    model = _build_bevfusion_model(args.config, args.checkpoint, args.device)

    from deployment.projects.bevfusion.quantization.spconv_int8 import (
        install_spconv_quantize_per_tensor_float_input_guard,
    )

    install_spconv_quantize_per_tensor_float_input_guard()

    # [2/6] Fuse BN
    if fuse_bn:
        print("\n[2/6] Fusing BatchNorm layers...")
        _fuse_dense_bn(model)
        _fuse_spconv_bn(model)
    else:
        print("\n[2/6] Skipping BatchNorm fusion")

    # [3/6] Insert Q/DQ nodes for dense parts
    if dense_on:
        print("\n[3/6] Inserting Q/DQ nodes (dense parts)...")
        _insert_dense_qdq(model, quant_flags, skip_layers)
    else:
        print("\n[3/6] Skipping dense Q/DQ insertion (sparse INT8 only or all quant_* False in deploy).")

    # [4/6] Build dataloader (spconv calib + optional dense calib)
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
    # gets OOD activations and mAP can collapse (~0).
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

    if dense_on:
        print(
            f"\n[5/6] Calibrating dense Q/DQ nodes ({num_batches} batches, method=mse) "
            f"with current sparse encoder (INT8 if step 4b succeeded)..."
        )
        calibrator = _calibrate_dense(model, dataloader, num_batches, method="mse")
    else:
        print("\n[5/6] Skipping dense calibration (no dense TensorQuantizer modules).")
        calibrator = None

    if skip_layers and dense_on:
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
    save_sd = model.state_dict()
    torch.save({"state_dict": save_sd}, output_path)

    sparse_keys = [k for k in save_sd if k.startswith("pts_middle_encoder.")]
    amax_keys = [k for k in sparse_keys if "_amax" in k]
    scale_keys = [k for k in sparse_keys if "scale" in k or "zero_point" in k]
    quant_keys = amax_keys or scale_keys
    tag = "_amax" if amax_keys else "scale/zp"
    print(
        f"\n  [save-check] Saved {len(save_sd)} total keys, "
        f"{len(sparse_keys)} pts_middle_encoder keys, {len(quant_keys)} {tag} keys"
    )
    _pathb = "pts_middle_encoder._pathb_sparse_tail_absmax"
    _pathb_m = "module.pts_middle_encoder._pathb_sparse_tail_absmax"
    if _pathb in save_sd or _pathb_m in save_sd:
        k = _pathb if _pathb in save_sd else _pathb_m
        v = float(save_sd[k].float().reshape(-1)[0].cpu().item())
        print(f"  [save-check] Path-B conv_out-input tail amax: {k} = {v:.6f}")
    else:
        print(
            "  [save-check] Path-B: no pts_middle_encoder._pathb_sparse_tail_absmax in state_dict "
            "(optional; ONNX transform prefers _pathb_last_int8_conv_output_absmax for terminal scale)."
        )
    _li = "pts_middle_encoder._pathb_last_int8_conv_output_absmax"
    _li_m = "module.pts_middle_encoder._pathb_last_int8_conv_output_absmax"
    if _li in save_sd or _li_m in save_sd:
        k2 = _li if _li in save_sd else _li_m
        v2 = float(save_sd[k2].float().reshape(-1)[0].cpu().item())
        print(f"  [save-check] Path-B last INT8 conv output amax (preferred for TRT): {k2} = {v2:.6f}")
    else:
        print(
            "  [save-check] Path-B: no pts_middle_encoder._pathb_last_int8_conv_output_absmax — "
            "re-run sparse PTQ with current AWML for best split-TRT terminal output_scale."
        )
    if quant_keys:
        print(f"  [save-check] sample {tag} keys: {quant_keys[:5]}")
        for k in quant_keys[:5]:
            v = save_sd[k]
            print(f"    {k} = {v.flatten().tolist()[:3]}")

    calib_path = output_path.with_suffix(".calib")
    if calibrator is not None:
        try:
            calibrator.save_calib_cache(str(calib_path))
        except Exception as e:
            print(f"  Warning: could not save calib cache: {e}")
    else:
        print("  No dense calib cache (sparse-int8-only checkpoint).")

    print("\n" + "=" * 80)
    print("BEVFusion PTQ Complete!")
    print(f"Model saved to: {output_path}")
    if calibrator is not None and calib_path.exists():
        print(f"Calibration cache saved to: {calib_path}")
    print("=" * 80)
    print("\nTo use this PTQ checkpoint for deployment:")
    print(f'  1. Set checkpoint_path = "{args.output}" in your deploy config')
    print(f"  2. Set quantization.ptq_checkpoint = True")
    if not dense_on:
        print(
            "  2b. Sparse-only: also set quant_backbone=False, quant_neck=False, quant_head=False "
            "(must match this checkpoint; else load_state_dict / mAP will be wrong)."
        )
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
