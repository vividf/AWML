# CenterPoint INT8 Quantization Notes

This document explains where Q/DQ nodes are inserted, how the AWML pipeline differs from the `Lidar_AI_Solution` README example, and practical tips to reduce mAP drop when using INT8.

## What gets quantized (Q/DQ insertion)
- **Entry point (PTQ CLI):** `tools/detection3d/centerpoint_quantization.py ptq` calls `quant_model(...)` from `projects/CenterPoint/quantization/replace.py`.
- **What `quant_model` replaces:**
  - `Conv2d` → `QuantConv2d` (adds input/weight TensorQuantizer)
  - `ConvTranspose2d` → `QuantConvTranspose2d`
  - `Linear` in `pts_voxel_encoder` → `QuantLinear`
- **Skipped layers:** Anything listed in `--skip-layers` (PTQ CLI) or `quantization.sensitive_layers` (deployment config) will **not** receive Q/DQ nodes; their quantizers are disabled after loading.
- **Calibration:** `CalibrationManager` enables calibration mode on all inserted quantizers, collects histograms for `num_calibration_batches`, then loads amax with the chosen `--amax-method` (`mse|entropy|percentile|max`).
- **Export/Deploy:** When `deploy_config_int8.py` has `quantization.enabled=True`, the deploy runner loads the checkpoint, fuses BN (if requested), inserts the same Q/DQ nodes (respecting `sensitive_layers`), loads the quantized weights, and disables quantizers for the skipped layers before ONNX/TRT export.

## Model / setup differences vs `Lidar_AI_Solution`
- **Backbone/head:** README in `libraries/3DSparseConvolution` reports an SCN-based CenterPoint on nuScenes; here we use **SECOND/Pillar + SECONDFPN + CenterHead** for T4Dataset. Different architectures respond differently to INT8.
- **Dataset/metric:** README numbers are nuScenes (mAP≈0.59 INT8). Current runs are on T4Dataset (j6gen2) with T4MetricV2; scores are not directly comparable.
- **TensorRT support:** We skip `pts_neck.deblocks.[0|1|2].0` (ConvTranspose2d) because TRT INT8 support for deconvs is limited; leaving them FP16/FP32 avoids build/accuracy issues.
- **Eval sample size:** Deploy config defaults to `evaluation.num_samples=100`, so reported mAP is from a small slice and can swing more than full-validation results.

## Why deblock layers are skipped
- ConvTranspose2d layers in the neck have limited/unstable INT8 support in TensorRT.
- Skipping them (keeping FP16/FP32) avoids engine build failures and large accuracy drops while still quantizing most of the backbone/head.

## Tips to improve INT8 accuracy
1. **Align skips across PTQ and deploy:** Use the same skip list when quantizing and exporting, e.g.\
   `--skip-layers pts_neck.deblocks.0.0 pts_neck.deblocks.1.0 pts_neck.deblocks.2.0`
2. **Increase calibration coverage:** Try `--calibrate-batches 300` (or more if feasible) with diverse samples.
3. **Experiment with amax:** `--amax-method entropy` or `percentile` can help on Pillar/SECOND backbones.
4. **Evaluate on more samples:** Set `evaluation.num_samples=-1` or a larger number to get a stable mAP gap.
5. **Consider QAT for tight budgets:** Add `QATHook` and fine-tune a few epochs to recover mAP if PTQ is still low.
6. **Sensitivity-driven skips:** Run the `sensitivity` command to find layers that hurt most when quantized, then add them to `--skip-layers` / `sensitive_layers`.

## Quick sanity checklist
- BN fused before quantization (`fuse_bn=True` or omit `--no-fuse-bn`).
- Same skip list used in PTQ and deployment.
- Calibration on validation-like data, not training augmentation-only streams.
- Verify FP16/FP32 baseline with `deploy_config.py` to measure true INT8 delta.
