#!/usr/bin/env python
"""Step 1: PTQ calibration with per-sample histogram tracing.

This is a faithful re-implementation of
``deployment/projects/centerpoint/quantization/quantize.py run_ptq`` with ONE
addition: after every calibration sample we snapshot the internal state of
every activation (input) quantizer's ``HistogramCalibrator``:

  * the raw histogram counts + bin edges,
  * the running max |x| seen so far,
  * the amax that the MSE criterion *would* pick from the histogram at that
    point (i.e. the amax trajectory over calibration samples).

The snapshots are the raw material for the tutorial figures: they show how the
histogram fills up / stretches as data streams in, and how the chosen clipping
range (amax) converges.

Run inside the deployment container from the AWML repo root:

    python work_dirs/centerpoint_tutorial/scripts/01_ptq_with_histogram_trace.py \
        --deploy-cfg work_dirs/centerpoint_tutorial/configs/deploy_config_int8_tutorial.py \
        --checkpoint work_dirs/centerpoint_tutorial/checkpoints/epoch_29_fp_reconstructed.pth \
        --output work_dirs/centerpoint_tutorial/checkpoints/epoch_29_ptq_tutorial.pth \
        --trace-dir work_dirs/centerpoint_tutorial/calib_trace
"""

import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

# AWML repo root (this file lives in work_dirs/centerpoint_tutorial/scripts/).
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy-cfg", required=True)
    ap.add_argument("--checkpoint", required=True, help="FP checkpoint to calibrate")
    ap.add_argument("--output", required=True, help="Output PTQ checkpoint path (+ sibling .calib)")
    ap.add_argument("--trace-dir", required=True, help="Where to write the per-sample snapshots")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--calibrate-samples", type=int, default=None, help="Override quantization.ptq.calibrate_samples")
    ap.add_argument(
        "--amax-every",
        type=int,
        default=1,
        help="Compute the (expensive) per-method amax trajectory every N samples (histograms are always saved every sample)",
    )
    return ap.parse_args()


def snapshot_histograms(model, TensorQuantizer, HistogramCalibrator):
    """Copy the current histogram state of every input quantizer to CPU numpy."""
    snap = {}
    for name, module in model.named_modules():
        if not isinstance(module, TensorQuantizer):
            continue
        cal = getattr(module, "_calibrator", None)
        if cal is None or not isinstance(cal, HistogramCalibrator):
            continue
        hist = getattr(cal, "_calib_hist", None)
        edges = getattr(cal, "_calib_bin_edges", None)
        if hist is None or edges is None:
            continue
        if isinstance(hist, torch.Tensor):
            hist = hist.detach().cpu().numpy()
        else:
            hist = np.asarray(hist)
        if isinstance(edges, torch.Tensor):
            edges = edges.detach().cpu().numpy()
        else:
            edges = np.asarray(edges)
        snap[name] = dict(hist=hist.astype(np.float64), edges=edges.astype(np.float64))
    return snap


def mse_amax_from_calibrator(cal):
    """Compute the amax the MSE criterion would currently choose (non-destructive)."""
    try:
        c = copy.deepcopy(cal)
        amax = c.compute_amax(method="mse")
        return float(amax.item()) if amax is not None else None
    except Exception:
        return None


def main():
    args = parse_args()

    from mmdet3d.apis import init_model
    from mmengine.config import Config

    from deployment.quantization.core import backend as quant_backend

    TensorQuantizer = quant_backend.get_tensor_quantizer_cls()
    calib = quant_backend.get_calib()
    print(f"Quantization backend: {quant_backend.resolve()}")

    from deployment.config.schema import load_quantization_config
    from deployment.projects.centerpoint.quantization.plan import build_centerpoint_plan
    from deployment.quantization import (
        CalibrationManager,
        disable_quantizers_in,
        expand_keep_fp16,
        print_quantizer_status,
    )
    from deployment.quantization.core.calibration import _allow_nondeterministic_algorithms
    from deployment.quantization.producer import build_calib_dataloader, init_quant_logging, save_ptq_checkpoint

    init_quant_logging()

    config, deploy_checkpoint_path, deploy_model_cfg = load_quantization_config(args.deploy_cfg)
    ptq = config.ptq
    calibrate_samples = args.calibrate_samples or (ptq.calibrate_samples if ptq else 100)
    batch_size = ptq.batch_size if ptq else 1
    calib_seed = ptq.calib_seed if ptq else None

    print("=" * 80)
    print("CenterPoint PTQ with per-sample histogram tracing (tutorial)")
    print("=" * 80)
    print(f"Model cfg   : {deploy_model_cfg}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Samples     : {calibrate_samples} (batch_size={batch_size})")
    print(f"Trace dir   : {args.trace_dir}")

    # [1] Build the model WITHOUT loading the checkpoint yet.
    #
    # The standard producer (run_ptq) does init_model(cfg, fp_checkpoint) because its input
    # is an UNFUSED training checkpoint. Ours is the BN-FUSED reconstruction (step 00): the
    # fused conv biases have no home in the unfused tree (those convs are bias=False), so
    # loading first would silently drop every bias. Instead we follow the deploy loader's
    # order: build -> fuse BN + insert Q/DQ -> THEN load the fused state_dict.
    cfg = Config.fromfile(deploy_model_cfg)
    model = init_model(cfg, None, device=args.device)
    model.eval()

    skip_layers = expand_keep_fp16(model, config.keep_fp16, log=False)

    # [2] Fuse BN + insert Q/DQ via the shared CenterPoint plan (same as producer + deploy).
    build_centerpoint_plan(config).prepare(model)

    # [2b] Now the module tree matches the fused checkpoint: load it. The only missing keys
    # should be the quantizer _amax buffers (they are about to be calibrated).
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    non_amax_missing = [k for k in missing if "_amax" not in k]
    if non_amax_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/tree mismatch: missing(non-amax)={non_amax_missing[:5]} unexpected={list(unexpected)[:5]}"
        )
    print(f"Loaded fused checkpoint: {len(missing)} amax buffers left for calibration to fill")

    # [3] Calibration dataloader from the model config's val split.
    dataloader = build_calib_dataloader(cfg, batch_size=batch_size, seed=calib_seed, shuffle=False)
    total = len(dataloader.dataset)
    num_batches = min(int(np.ceil(calibrate_samples / batch_size)), len(dataloader))
    print(f"Dataset size: {total}; using {num_batches} batches")

    # [4] Manual calibration loop == CalibrationManager.collect_stats + per-sample snapshots.
    mgr = CalibrationManager(model)
    mgr.set_quantizer_fast()  # torch-native histograms (GPU)
    mgr._enable_calibration_mode()

    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    hist_trace = []  # per-sample: {layer: {hist, edges}}
    amax_trace = []  # per-sample: {layer: mse_amax}

    t0 = time.time()
    with torch.no_grad(), _allow_nondeterministic_algorithms():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            model.test_step(batch)

            snap = snapshot_histograms(model, TensorQuantizer, calib.HistogramCalibrator)
            hist_trace.append(snap)

            if i % args.amax_every == 0 or i == num_batches - 1:
                step_amax = {}
                for name, module in model.named_modules():
                    if isinstance(module, TensorQuantizer) and isinstance(
                        getattr(module, "_calibrator", None), calib.HistogramCalibrator
                    ):
                        step_amax[name] = mse_amax_from_calibrator(module._calibrator)
                amax_trace.append(dict(step=i, amax=step_amax))

            if (i + 1) % 10 == 0 or i == num_batches - 1:
                print(f"  calibrated {i + 1}/{num_batches} samples ({time.time() - t0:.1f}s)")

    mgr._disable_calibration_mode()

    # [5] Final amax under all supported criteria (for the method-comparison figure),
    # BEFORE load_calib_amax so the deployed values come from the standard path below.
    method_comparison = {}
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and isinstance(
            getattr(module, "_calibrator", None), calib.HistogramCalibrator
        ):
            entry = {}
            for method in ("mse", "entropy", "max"):
                try:
                    c = copy.deepcopy(module._calibrator)
                    a = c.compute_amax(method=method)
                    entry[method] = float(a.item()) if a is not None else None
                except Exception as e:
                    entry[method] = f"error: {e}"
            for pct in (99.9, 99.99):
                try:
                    c = copy.deepcopy(module._calibrator)
                    a = c.compute_amax(method="percentile", percentile=pct)
                    entry[f"percentile_{pct}"] = float(a.item()) if a is not None else None
                except Exception as e:
                    entry[f"percentile_{pct}"] = f"error: {e}"
            method_comparison[name] = entry

    # [6] Standard finish: load MSE amax into quantizers, honor keep_fp16, save artifact.
    mgr.compute_amax("mse")
    disabled = disable_quantizers_in(model, skip_layers)
    if disabled:
        print(f"Disabled quantization in {disabled} keep_fp16 module(s)")

    print("\nQuantizer Status:")
    print_quantizer_status(model)

    output_path, calib_path = save_ptq_checkpoint(model, args.output, mgr)
    print(f"\nPTQ checkpoint : {output_path}")
    print(f"Calib cache    : {calib_path}")

    # [7] Persist traces.
    with open(trace_dir / "hist_trace.pkl", "wb") as f:
        pickle.dump(hist_trace, f)
    with open(trace_dir / "amax_trace.json", "w") as f:
        json.dump(amax_trace, f, indent=2)
    with open(trace_dir / "method_comparison.json", "w") as f:
        json.dump(method_comparison, f, indent=2)

    # Final weight-quantizer amax too (MaxCalibrator, per-channel) — used in the docs
    # to contrast per-channel weight quantization vs per-tensor activation quantization.
    weight_amax = {}
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and module._amax is not None:
            weight_amax[name] = module._amax.detach().cpu().numpy()
    with open(trace_dir / "final_amax_all_quantizers.pkl", "wb") as f:
        pickle.dump(weight_amax, f)

    print(f"\nTraces written to {trace_dir}/")
    print("  hist_trace.pkl              per-sample activation histograms")
    print("  amax_trace.json             per-sample MSE-amax trajectory")
    print("  method_comparison.json      final amax under mse/entropy/percentile/max")
    print("  final_amax_all_quantizers.pkl  final amax of every quantizer (incl. per-channel weights)")


if __name__ == "__main__":
    main()
