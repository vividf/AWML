# 03 — Full Pipeline Walkthrough: PTQ → ONNX → TensorRT → Evaluation

*English version — [中文版 / Chinese](03_pipeline_walkthrough.md)*

> This is the "follow along and you can reproduce it" run log. All outputs live in
> `work_dirs/centerpoint_tutorial/`; one-shot re-run:
> `bash work_dirs/centerpoint_tutorial/scripts/run_all.sh` (from the AWML repo root).

## 0. Environment and materials

| Item | Value |
|---|---|
| Container | `bevfusion-deployment:latest` (torch 2.8.0+cu129 / TensorRT 10.8.0 / pytorch-quantization 2.1.3) |
| GPU | RTX PRO 6000 Blackwell |
| Model | CenterPoint (SECOND backbone) 2.6, `second_secfpn_8xb16_121m_j6gen2_base_amp_t4metric_v2` |
| Data | local `db_j6gen2_v3` (60-frame val split; the release used the full 5179-frame val set) |
| Starting checkpoint | the release's `epoch_29_ptq.pth` (PTQ does not touch weights → stripping the amax recovers the FP weights) |

How the container is started (shared by every step):

```bash
docker run --rm --gpus all --shm-size=32g \
    -v $PWD:/workspace -w /workspace \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    bevfusion-deployment:latest <command>
```

> Note: `--shm-size=32g` is mandatory (the 64MB default kills the DataLoader workers).

## 1. Recovering the FP checkpoint (step 00)

The official flow takes a training-produced FP checkpoint (`epoch_29.pth`) as input. We do not
have that file locally, but **PTQ calibration only adds amax buffers and never touches the
weights**, so:

```bash
python3 work_dirs/centerpoint_tutorial/scripts/00_reconstruct_fp_checkpoint.py \
    --ptq-checkpoint ~/Desktop/centerpoint_2_6_1_quant/epoch_29_ptq.pth \
    --output work_dirs/centerpoint_tutorial/checkpoints/epoch_29_fp_reconstructed.pth
```

132 keys → 76 weight keys (kept) + 56 amax keys (stripped, saved separately as
`original_release_amax.pth` as the comparison baseline).

The recovered weights are the **BN-already-fused** version (the producer does fuse_bn before
calibration). When they are loaded back into an unfused model, the backbone's BN keys are
missing → BN stays at its default initialization (γ=1, β=0, μ=0, σ²=1) → numerically a no-op
(error ~5e-6, coming from eps). So this checkpoint is equivalent to the genuine FP checkpoint
along the deployment path.

## 2. PTQ calibration + per-sample histogram recording (step 01)

The official flow is a one-liner:

```bash
python -m deployment.projects.centerpoint.quantization.quantize ptq \
    --deploy-cfg <deploy_config_int8_*.py>
```

The tutorial uses an equivalent script with recording added
(`01_ptq_with_histogram_trace.py`); internally the flow matches `run_ptq`
(fuse BN → insert Q/DQ → calibrate → save), it just stores an extra histogram / amax snapshot
after each sample's forward pass.

**The only difference is the load order, and it was forced on us by a real bug**: `run_ptq`'s
input is an unfused training checkpoint, so it does `init_model(cfg, ckpt)` first and fuses
afterwards; our input is a BN-already-fused recovered checkpoint — if we loaded first and fused
afterwards, the fused conv biases would have nowhere to land in the unfused model (those convs
are `bias=False`), so **26 biases would be silently dropped by strict=False**, and then
fuse_bn would fill the biases in with zeros. The model looks runnable, calibration completes,
the checkpoint saves — but mAP = 0, and the calibrated activation amax comes out systematically
3–10× too large. The correct approach is the same as the deploy loader's:
**fuse BN + insert Q/DQ to build the tree first, then load the state_dict**, and check that the
missing keys are empty apart from `_amax` (the script raises directly).

The `quantization` block of the deploy config is the single source of truth
(the release recipe):

```python
quantization = dict(
    enabled=True, mode="ptq", fuse_bn=True, default_precision="int8",
    keep_fp16=["pts_voxel_encoder", "pts_backbone.blocks.0"],  # these two are not quantized
    disable_recipes=["add"],       # SECOND has no residual add
    ptq=dict(calibrate_samples=60, batch_size=1, calib_seed=0),  # the release used 400
)
```

Artifacts:

```
checkpoints/epoch_29_ptq_tutorial.pth    # BN-fused weights + 56 amax values
checkpoints/epoch_29_ptq_tutorial.calib  # amax cache (reusable by QAT, handy for debugging)
calib_trace/hist_trace.pkl               # histogram snapshots, 60 samples × 28 input quantizers
calib_trace/amax_trace.json              # per-sample MSE-amax trajectory
calib_trace/method_comparison.json       # final amax comparison across mse/entropy/percentile/max
```

For how the histogram → amax mechanism works, see
[02 — PTQ calibration](02_ptq_calibration_histogram.en.md).

## 3. amax reproducibility verification (step 03)

We re-ran calibration with **different calibration data** (60 local samples vs. the release's
400 from the full val set) and compared every quantizer's amax
(`calib_trace/amax_comparison.md`):

- **weight amax bit-identical** (same weights, deterministic MaxCalibrator) → verifies that the
  whole fuse-BN → insert-Q/DQ → calibrate pipeline did not drift.
- **activation amax differs but stays in the same order of magnitude** (different calibration
  data; this is expected behavior).

## 4. FP16 (before PTQ) deploy: export → engine → eval (step 4)

```bash
python -m deployment.cli.main centerpoint \
    work_dirs/centerpoint_tutorial/configs/deploy_config_fp16_tutorial.py
```

There is one detail in the config worth discussing: it still carries a `quantization` block,
but with `keep_fp16=["*"]` — matching every module, so **not a single quantizer gets inserted**.
Why not just drop the block? Because the checkpoint we recovered has **BN already fused**
(the conv biases carry the entire BN shift), while the ordinary FP load path builds an
**unfused** model whose convs are `bias=False` — so those 26 biases would be treated as
unexpected keys and dropped as a group, breaking the model outright (we hit this landmine on
our first run, mAP=0). `enabled=True, fuse_bn=True` routes loading down the same
"fuse BN first, then load" path, and the keys line up.

The CLI then does, in order: PyTorch load → ONNX export (two files:
`pts_voxel_encoder.onnx` + `pts_backbone_neck_head.onnx`) → TensorRT engine build
(`precision_policy="fp16"`) → backend evaluation (pytorch / tensorrt).

The FP16 ONNX graph contains **no QuantizeLinear/DequantizeLinear at all** — that is what
"before PTQ" means.

> Lesson (worth putting in the onboarding material): **fuse_bn is part of the deployment
> contract**. The producer, the loader and the export must agree on "which BNs are fused",
> otherwise the state_dict keys will not match. This is exactly why the framework routes all
> three through the same `build_centerpoint_plan().prepare()`.

## 5. INT8 (after PTQ) deploy (step 5)

```bash
python -m deployment.cli.main centerpoint \
    work_dirs/centerpoint_tutorial/configs/deploy_config_int8_tutorial.py
```

The differences: the `quantization` block is enabled, and the checkpoint points at
`epoch_29_ptq_tutorial.pth`. The loader applies the same fuse BN + insert Q/DQ transformation
to the freshly built model **before loading the state_dict** (so that the module tree matches
the checkpoint), and export enables `use_fb_fake_quant`, writing each TensorQuantizer out as an
ONNX Q/DQ pair:

```
op statistics of the INT8 pts_backbone_neck_head.onnx:
QuantizeLinear × 56, DequantizeLinear × 56, Conv × 31, Relu × 26, ConvTranspose × 1 ...
```

Seeing Q/DQ, TensorRT goes down the explicit-quantization path and fuses `DQ→Conv→ReLU→Q` into
an INT8 kernel.

## 6. Results (local 60-frame val split)

| backend | mAP (center dist BEV) | mAP (plane dist) | TRT backbone+head latency |
|---|---|---|---|
| PyTorch FP (fused) | 0.4973 | 0.5164 | — |
| **TensorRT FP16 (before PTQ)** | **0.4996** | **0.5189** | 5.92 ± 0.28 ms |
| PyTorch fake-quant (after PTQ) | 0.4857 | 0.5035 | — |
| **TensorRT INT8 (after PTQ)** | **0.4938** | **0.5120** | **3.47 ± 0.22 ms (1.71×)** |

INT8's accuracy loss: −0.006 mAP on TRT; −0.012 in the fake-quant preview.
amax reproducibility: weight amax **bit-identical** to the release; activation amax median
difference **0.4%** (under different calibration data) — full table in
`calib_trace/amax_comparison.md`.

For reference, the release's results on the full 5179-frame val set
(`work_dirs/centerpoint_2_6_skip_stage_0_by_distance/deployment.log`):

| backend | mAP (center dist BEV) | mAP (plane dist) | mAPH (center dist BEV) |
|---|---|---|---|
| PyTorch fake-quant | 0.7401 | 0.7574 | 0.6856 |
| TensorRT INT8 | 0.7391 | 0.7555 | 0.6852 |

Two important ways to read this:

1. **PyTorch fake-quant ≈ TensorRT INT8** (Δ ≈ 0.001): the numerical behavior simulated by
   fake quant in the float domain is nearly identical to a real INT8 kernel — so after PTQ you
   can estimate INT8 accuracy without building an engine.
2. The absolute values from our local 60 frames are not comparable to the release (different
   data, and a class distribution dominated by car/truck only). What to look at is the
   **FP16 vs. INT8 gap** and the **consistency between backends**.

→ Next: [04 — Per-backbone quantization special handling](04_backbone_recipes.en.md)
