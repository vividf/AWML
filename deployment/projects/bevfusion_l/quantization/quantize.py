#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
BEVFusion Quantization Tools

This script provides CLI commands for PTQ (Post-Training Quantization)
for BEVFusion models. It mirrors centerpoint/quantization/quantize.py but handles:
  - Dense parts (pts_backbone, pts_neck, bbox_head): pytorch_quantization Q/DQ
  - Sparse encoder (pts_middle_encoder): spconv BN fusion + manual calibration
  - Voxel encoder (pts_voxel_encoder): optional Linear quantization

Usage:
    # PTQ Mode - Quantize a pre-trained BEVFusion model
    python -m deployment.projects.bevfusion_l.quantization.quantize ptq \
        --config projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
        --checkpoint work_dirs/bevfusion/epoch_30.pth \
        --deploy-cfg deployment/projects/bevfusion_l/config/deploy_config_int8.py \
        --calibrate-samples 256 \
        --batch-size 1 \
        --calib-seed 0 \
        --output work_dirs/bevfusion/epoch_30_ptq.pth

    # PTQ sparse encoder only (spconv INT8; dense stays FP32). Deploy eval must set
    # quant_backbone/neck/head=False. Use deploy cfg with spconv_int8=True.
    python -m deployment.projects.bevfusion_l.quantization.quantize ptq ... \
        --deploy-cfg deployment/projects/bevfusion_l/config/deploy_config_split_int8.py \
        --sparse-int8-only --output work_dirs/bevfusion/epoch_30_ptq_sparse_only.pth
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


def _fuse_spconv_bn(model) -> int:
    """Fuse SparseConv+BN in ``pts_middle_encoder`` via the shared framework implementation."""
    from deployment.quantization.sparse import fuse_spconv_bn_in_encoder

    sparse_encoder = getattr(model, "pts_middle_encoder", None)
    if sparse_encoder is None:
        print("  No pts_middle_encoder found; skipping spconv BN fusion")
        return 0

    fused_count = fuse_spconv_bn_in_encoder(sparse_encoder)
    print(f"  Fused {fused_count} SparseConv-BN pairs in pts_middle_encoder")
    return fused_count


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


def _calibrate_spconv(
    model,
    dataloader,
    num_samples,
    device,
    output_path,
    fp16_layers,
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

                    # Reproduce the real forward path (BEVFusion.extract_pts_feat):
                    #   feats = pts_voxel_encoder(feats, sizes, coords)
                    #   x = pts_middle_encoder(feats, coords, batch_size)
                    # The voxel encoder produces exactly the feature layout the sparse
                    # encoder's ``conv_input`` expects. ``HardSimpleVoxelSinCosEncoder``
                    # does the per-voxel mean reduction internally AND expands the raw
                    # dims into sin/cos (fourier) channels (e.g. 5 -> 50), so a hard-coded
                    # mean reduction here would feed ``conv_input`` the wrong channel count
                    # ("channel size mismatch"). Running the model's own encoder keeps this
                    # correct for any voxel encoder (HardSimpleVFE, sin/cos, ...).
                    voxel_encoder = getattr(model, "pts_voxel_encoder", None)
                    if voxel_encoder is not None:
                        feats = voxel_encoder(feats, sizes, coords).contiguous()
                    elif sizes is not None and getattr(model, "voxelize_reduce", True):
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

    from deployment.quantization.sparse.spconv_int8 import (
        apply_nvidia_spconv_int8,
        calibrate_spconv_nvidia,
    )

    fp16_layers = list(fp16_layers or [])
    if fp16_layers:
        print(f"  [nvidia-quant] spconv_int8_fp16_layers active ({len(fp16_layers)} pattern(s)): {fp16_layers}")
        print(
            "  [nvidia-quant] These modules will NOT get _input_quantizer/_weight_quantizer → "
            "downstream PTQ _amax is calibrated against TRUE FP activations (no fake-quant contamination)."
        )

    print("  Applying NVIDIA TensorQuantizer path (histogram + MSE, adapted from CUDA-BEVFusion)")
    apply_nvidia_spconv_int8(
        sparse_encoder,
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


def _print_ptq_save_check(save_sd: dict) -> None:
    """Diagnostic: report saved sparse-encoder quant keys and sparse INT8 terminal amax buffers."""
    sparse_keys = [k for k in save_sd if k.startswith("pts_middle_encoder.")]
    amax_keys = [k for k in sparse_keys if "_amax" in k]
    scale_keys = [k for k in sparse_keys if "scale" in k or "zero_point" in k]
    quant_keys = amax_keys or scale_keys
    tag = "_amax" if amax_keys else "scale/zp"
    print(
        f"\n  [save-check] Saved {len(save_sd)} total keys, "
        f"{len(sparse_keys)} pts_middle_encoder keys, {len(quant_keys)} {tag} keys"
    )
    _tail = "pts_middle_encoder._sparse_tail_absmax"
    _tail_m = "module.pts_middle_encoder._sparse_tail_absmax"
    if _tail in save_sd or _tail_m in save_sd:
        k = _tail if _tail in save_sd else _tail_m
        v = float(save_sd[k].float().reshape(-1)[0].cpu().item())
        print(f"  [save-check] sparse INT8 conv_out-input tail amax: {k} = {v:.6f}")
    else:
        print(
            "  [save-check] sparse INT8: no pts_middle_encoder._sparse_tail_absmax in state_dict "
            "(optional; ONNX transform prefers _last_int8_conv_output_absmax for terminal scale)."
        )
    _li = "pts_middle_encoder._last_int8_conv_output_absmax"
    _li_m = "module.pts_middle_encoder._last_int8_conv_output_absmax"
    if _li in save_sd or _li_m in save_sd:
        k2 = _li if _li in save_sd else _li_m
        v2 = float(save_sd[k2].float().reshape(-1)[0].cpu().item())
        print(f"  [save-check] sparse INT8 last INT8 conv output amax (preferred for TRT): {k2} = {v2:.6f}")
    else:
        print(
            "  [save-check] sparse INT8: no pts_middle_encoder._last_int8_conv_output_absmax — "
            "re-run sparse PTQ with current AWML for best split-TRT terminal output_scale."
        )
    if quant_keys:
        print(f"  [save-check] sample {tag} keys: {quant_keys[:5]}")
        for k in quant_keys[:5]:
            v = save_sd[k]
            print(f"    {k} = {v.flatten().tolist()[:3]}")


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

    from deployment.config.schema import load_quantization_config
    from deployment.projects.bevfusion_l.quantization.plan import build_bevfusion_plan

    # Single source of truth: parse the deploy ``quantization`` block once.
    config, _ = load_quantization_config(args.deploy_cfg)
    if getattr(args, "sparse_int8_only", False):
        # Sparse-only PTQ: keep the dense towers FP32 (no QuantConv2d in the .pth).
        config = config.with_overrides(quant_backbone=False, quant_neck=False, quant_head=False, quant_add=False)

    fuse_bn = config.fuse_bn
    skip_layers = config.resolved_sensitive_layers()
    dense_on = config.dense_quant_enabled()

    if getattr(args, "sparse_int8_only", False):
        print("\n  --sparse-int8-only: dense backbone/neck/head will stay FP32 (no QuantConv2d in .pth).")
        if args.skip_spconv_int8:
            print("  Warning: --sparse-int8-only with --skip-spconv-int8 → no INT8 path is calibrated.")
        if not config.spconv_int8:
            print(
                "  Warning: deploy quantization.spconv_int8=False; spconv INT8 step will be skipped. "
                "Use a deploy cfg with spconv_int8=True for sparse-only PTQ."
            )

    # [1/6] Load model
    print("\n[1/6] Loading BEVFusion model...")
    model = _build_bevfusion_model(args.config, args.checkpoint, args.device)

    # [2/6] Sparse BN fuse (sparse INT8 quantizers are attached + calibrated later in [4b]).
    if fuse_bn:
        print("\n[2/6] Fusing sparse SparseConv-BN...")
        _fuse_spconv_bn(model)
    else:
        print("\n[2/6] Skipping BatchNorm fusion")

    # [3/6] Dense BN fuse + Q/DQ via the SHARED QuantizationPlan.
    # The deploy loader (model_loader._load_with_quantization) builds the SAME dense scheme, so the
    # PTQ state_dict and the deployed module tree line up by construction (no drift by convention).
    # ``include_sparse=False``: the sparse tower is handled by [2]/[4b], not this dense plan.
    if dense_on:
        print("\n[3/6] Dense BN fuse + Q/DQ via shared QuantizationPlan...")
    else:
        print("\n[3/6] Dense Q/DQ skipped (sparse INT8 only); dense BN still fused if fuse_bn.")
    build_bevfusion_plan(config, include_sparse=False).prepare(model)

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
    if not args.skip_spconv_int8:
        if config.spconv_int8:
            print("\n[4b/6] Spconv INT8 for sparse encoder (NVIDIA TensorQuantizer, runs before dense PTQ calib)...")
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
                    config.spconv_int8_fp16_layers,
                )
            except Exception as e:
                # Fail loud: spconv INT8 was explicitly requested (spconv_int8=True). Silently
                # continuing would save a checkpoint that *looks* quantized but keeps the sparse
                # encoder in FP32; dense Q/DQ then calibrates against the wrong BEV distribution
                # and mAP can collapse (~0) with no error surfaced. Abort instead.
                raise RuntimeError(
                    f"Spconv INT8 calibration failed for the sparse encoder: {e}. "
                    "The deploy config requested spconv_int8=True, so refusing to save a checkpoint "
                    "with an un-quantized (FP32) sparse encoder. Fix the error above, or set "
                    "spconv_int8=False if an FP32 sparse encoder is intended."
                ) from e
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

    _print_ptq_save_check(save_sd)

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
    print(f"       deployment/projects/bevfusion_l/config/deploy_config_int8.py \\")
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
