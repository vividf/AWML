# BEVVoVNet V-99-eSE + SECONDFPN Architecture

## 1. Overview

This document describes the architecture of the CenterPoint backbone variant using **BEVVoVNet V-99-eSE** as `pts_backbone` and **SECONDFPN** as `pts_neck`.

- **Config**: `vov99_secfpn_4xb16_121m_j6gen2_base_amp.py`
- **Base**: `second_secfpn_4xb16_121m_j6gen2_base_amp.py`
- **Input BEV feature**: `(B, 32, 1020, 1020)`

### What does "99" mean?

The number **99** in VoVNet (V-99-eSE) comes from the **original VoVNet/OSANet naming**: it indicates the **network depth** (number of layers). VoVNet variants are named by approximate layer count:

- **V-19**: ~19 layers (lightweight)
- **V-39**: ~39 layers
- **V-57**: ~57 layers
- **V-99**: ~99 layers (deepest standard variant)

So "VoV99" = VoVNet with depth ~99. The same spec uses **OSA (One-Shot Aggregation)** blocks: each block has 5 conv layers, and the count 1+3+9+3 blocks × 5 layers plus stem gives a deep network suited for higher accuracy. The "eSE" suffix means **eSE (effective Squeeze-Excitation)** attention is used in the OSA modules.

### Key Config Values

```python
pts_backbone = dict(
    type="BEVVoVNet",
    spec_name="V-99-eSE",
    input_ch=32,
    stem_strides=(1, 1, 1),       # No downsampling in stem
    out_features=("stage2", "stage3", "stage4"),
    frozen_stages=-1,
    norm_eval=False,
)
pts_neck = dict(
    type="SECONDFPN",
    in_channels=[256, 512, 768],
    out_channels=[128, 128, 128],
    upsample_strides=[0.5, 1, 2],
)
```

### V-99-eSE Specification

| Parameter | Value |
|-----------|-------|
| Stem channels | [64, 64, 128] |
| Stage conv channels | [128, 160, 192, 224] |
| Stage output channels | [256, 512, 768, 1024] |
| Layers per OSA block | 5 |
| Blocks per stage | [1, 3, 9, 3] |
| eSE (Squeeze-Excitation) | Yes |
| Depthwise separable | No |

### Why are the OSA block counts 1, 3, 9, 3?

The pattern **1, 3, 9, 3** (for stages 2, 3, 4, 5) comes from the **original VoVNet paper** (CVPRW 2019, Lee et al.). The design follows a common backbone principle: **spend more depth where spatial size is smaller** (so compute per layer is lower) and **keep the first stage light** (highest resolution is the most expensive).

| Stage | Blocks | Spatial (BEV) | Rationale |
|-------|--------|----------------|-----------|
| **Stage 2** | **1** | 1020×1020 | Highest resolution; one OSA block is enough to build low-level features without blowing up compute. |
| **Stage 3** | **3** | 510×510 | After 2× downsampling; a few more blocks add capacity at moderate cost. |
| **Stage 4** | **9** | 255×255 | Heaviest stage. Resolution is ¼ of stage 2, so each conv is much cheaper; more blocks here give strong semantic features for detection. |
| **Stage 5** | **3** | (pruned in BEV) | Smallest resolution in original image VoVNet; fewer blocks than stage 4 to balance depth and avoid overfitting, and to hit the target ~99-layer depth. |

So: **light at high resolution (1) → ramp up in the middle (3, 9) → taper at the end (3)**. The exact numbers 1-3-9-3 were chosen empirically in the paper to get a good accuracy/speed trade-off and the desired depth; lighter variants (e.g. V-39, V-57) use fewer blocks per stage (e.g. [1,1,2,2] or [1,1,4,3]).

> **Note**: Stage5 (1024ch, 3 blocks) is pruned since `out_features` only requests up to `stage4`.

---

## 2. Backbone: BEVVoVNet

### 2.1 Stem (No Downsampling)

Unlike the original VoVNet which uses `stem_strides=(2, 1, 2)` for 4x downsampling, BEVVoVNet uses `stem_strides=(1, 1, 1)` to preserve full BEV spatial resolution.

| Layer | Operation | Output |
|-------|-----------|--------|
| stem_1 | Conv2d(32 → 64, k=3, s=1, p=1) + BN + ReLU | `(B, 64, 1020, 1020)` |
| stem_2 | Conv2d(64 → 64, k=3, s=1, p=1) + BN + ReLU | `(B, 64, 1020, 1020)` |
| stem_3 | Conv2d(64 → 128, k=3, s=1, p=1) + BN + ReLU | `(B, 128, 1020, 1020)` |

### 2.2 Stage2 (256ch, no MaxPool)

- **No MaxPool** because `stage_num == 2` (first stage after stem).
- **1 OSA block**: 5 conv layers + concat + 1x1 conv + eSE attention.
- Input: `(B, 128, 1020, 1020)` → Output: `(B, 256, 1020, 1020)`

**OSA Block Structure**:
```text
input (128ch)
  ├─ Conv3x3 → 128ch  (layer 0)
  ├─ Conv3x3 → 128ch  (layer 1)
  ├─ Conv3x3 → 128ch  (layer 2)
  ├─ Conv3x3 → 128ch  (layer 3)
  └─ Conv3x3 → 128ch  (layer 4)
Concat([input, layer0..4]) → (128 + 5×128 = 768ch)
Conv1x1(768 → 256) + eSE → 256ch
```

### 2.3 Stage3 (512ch, MaxPool ↓2)

- **MaxPool2d(k=3, s=2, ceil_mode=True)**: `1020 → 510`
- **3 OSA blocks** (block_per_stage[1] = 3):
  - First block: stage_conv_ch=160, no identity shortcut, no eSE
  - Middle block: identity shortcut, no eSE
  - Last block: identity shortcut + eSE attention
- Input: `(B, 256, 1020, 1020)` → After MaxPool: `(B, 256, 510, 510)` → Output: `(B, 512, 510, 510)`

### 2.4 Stage4 (768ch, MaxPool ↓2)

- **MaxPool2d(k=3, s=2, ceil_mode=True)**: `510 → 255`
- **9 OSA blocks** (block_per_stage[2] = 9) — the heaviest stage.
  - stage_conv_ch=192, output=768ch
  - Last block has eSE; all blocks after the first have identity shortcut.
- Input: `(B, 512, 510, 510)` → After MaxPool: `(B, 512, 255, 255)` → Output: `(B, 768, 255, 255)`

### 2.5 Backbone Summary

```text
Input: (B, 32, 1020, 1020)

BEVVoVNet V-99-eSE:
    Stem (s=1,1,1) → (B, 128, 1020, 1020)
    Stage2 (1 OSA)  → (B, 256, 1020, 1020)   ← out_features
    Stage3 (3 OSA)  → (B, 512, 510, 510)      ← out_features
    Stage4 (9 OSA)  → (B, 768, 255, 255)      ← out_features
```

---

## 3. Neck: SECONDFPN

### Configuration

```python
pts_neck = dict(
    type="SECONDFPN",
    in_channels=[256, 512, 768],
    out_channels=[128, 128, 128],
    upsample_strides=[0.5, 1, 2],
    norm_cfg=dict(type="BN", eps=1e-5, momentum=0.01),
    upsample_cfg=dict(type="deconv", bias=False),
    use_conv_for_no_stride=True,
)
```

### Branch Operations

| Branch | Input | upsample_stride | Operation | Output |
|--------|-------|-----------------|-----------|--------|
| 0 (stage2) | `256 x 1020 x 1020` | 0.5 | Conv2d(256→128, k=2, s=2) | `128 x 510 x 510` |
| 1 (stage3) | `512 x 510 x 510` | 1 | Conv2d(512→128, k=1, s=1) | `128 x 510 x 510` |
| 2 (stage4) | `768 x 255 x 255` | 2 | ConvTranspose2d(768→128, k=2, s=2) | `128 x 510 x 510` |

### Concatenation

```text
up0 = (B, 128, 510, 510)
up1 = (B, 128, 510, 510)
up2 = (B, 128, 510, 510)

out = torch.cat([up0, up1, up2], dim=1) → (B, 384, 510, 510)
```

---

## 4. Full Architecture Diagram

```text
Point Cloud
  → Voxelization (max_num_points=32)
  → PillarFeatureNet (in=5, out=32)
  → PointPillarsScatter
  → BEV Feature: 32 x 1020 x 1020

  → BEVVoVNet V-99-eSE Stem (stride=1,1,1, no downsampling)
     → 128 x 1020 x 1020
  → Stage2 (1 OSA block, 5 layers, eSE)
     → 256 x 1020 x 1020
  → Stage3 (MaxPool ↓2, 3 OSA blocks, 5 layers each, eSE on last)
     → 512 x 510 x 510
  → Stage4 (MaxPool ↓2, 9 OSA blocks, 5 layers each, eSE on last)
     → 768 x 255 x 255

  → SECONDFPN
     branch 0: 256 x 1020 → Conv(s=2) → 128 x 510
     branch 1: 512 x 510  → Conv(s=1) → 128 x 510
     branch 2: 768 x 255  → Deconv(s=2) → 128 x 510
     concat → 384 x 510 x 510

  → CenterHead (in_channels=384, out_size_factor=2)
```

---

## 5. Comparison with SECOND and BEVResNet34

| | SECOND | BEVResNet34 | BEVVoVNet V-99-eSE |
|---|---|---|---|
| **Stage outputs (ch)** | 64 / 128 / 256 | 64 / 128 / 256 | 256 / 512 / 768 |
| **Stage outputs (spatial)** | 510 / 255 / 128 | 1020 / 510 / 255 | 1020 / 510 / 255 |
| **Blocks** | 3+5+5 Conv | 3+4+6 BasicBlock | 1+3+9 OSA (5 layers each) |
| **Neck in_channels** | [64, 128, 256] | [64, 128, 256] | [256, 512, 768] |
| **Key feature** | Simple stacked Conv | Residual connections | Dense aggregation + eSE |
| **Batch size** | 16 | 8 | 8 |
| **Activation Checkpoint** | No | Yes (`pts_backbone`) | Yes (`pts_backbone`) |
| **Capacity** | Low | Medium | High |

BEVVoVNet has significantly more parameters and FLOPs due to:
1. Higher channel counts (256/512/768 vs 64/128/256)
2. Dense OSA aggregation (each OSA block concatenates all intermediate features)
3. 9 OSA blocks in stage4, each with 5 conv layers = 45 conv layers in stage4 alone

---

## 7. BEVVoVNet (AWML) vs StreamPETR VoVNet

Both use the **same V-99-eSE spec** (stage_out_ch [256, 512, 768, 1024], block_per_stage [1, 3, 9, 3], layer_per_block=5, eSE). The differences are:

| Aspect | StreamPETR VoVNet | AWML BEVVoVNet |
|--------|-------------------|----------------|
| **Class** | `VoVNet` | `BEVVoVNet` (subclass of VoVNet) |
| **Input** | Camera/image features (e.g. 3ch or feature dim) | BEV pillar features (32ch, 1020×1020) |
| **Stem strides** | Default `(2, 1, 2)` → **4× spatial downsampling** | **`(1, 1, 1)`** → no downsampling |
| **Spatial size after stem** | 4× smaller (e.g. 256×256 for 1024 input) | Same as input (1020×1020) |
| **Purpose** | Image backbone for multi-view 3D (camera) | BEV backbone for LiDAR/point-cloud BEV |
| **Neck** | Projection/FPN for camera stream | SECONDFPN with in_channels=[256, 512, 768] |

**Summary**: StreamPETR uses VoVNet as designed for **image** backbones (stem downsamples 4×). AWML adds **BEVVoVNet**, which keeps the same OSA/stage structure but **rebuilds the stem with configurable strides** so that with `stem_strides=(1,1,1)` the BEV grid keeps full resolution. The V-99-eSE **spec (channels, blocks, eSE) is identical**; only the stem and usage context differ.

---

## 8. Why Stage Output Channels Differ from SECOND / BEVResNet34

| Backbone | Stage output channels | Neck `in_channels` |
|----------|------------------------|--------------------|
| SECOND | 64 / 128 / 256 | [64, 128, 256] |
| BEVResNet34 | 64 / 128 / 256 | [64, 128, 256] |
| BEVVoVNet V-99-eSE | **256 / 512 / 768** | **[256, 512, 768]** |

**Reasons:**

1. **Different backbone families**  
   SECOND and BEVResNet34 use a **lightweight** channel schedule: base 64, then 64→128→256 per stage. VoVNet comes from the **original paper** (CVPRW 2019, ETRI/Megvii) and uses a **heavier** schedule: stage_out_ch = [256, 512, 768, 1024]. So 256/512/768 are the **native** VoVNet design for higher capacity, not chosen to match SECOND.

2. **SECONDFPN is backbone-agnostic**  
   The neck only requires that `in_channels` matches the backbone’s multi-scale output channels. It then projects each branch to 128 and concatenates → 384 for the head. So we set `pts_neck.in_channels=[256, 512, 768]` when using VoV99; no need to change SECOND or ResNet.

3. **Accuracy vs cost**  
   VoV99’s larger channels (and OSA depth) aim for **higher accuracy** at the cost of more parameters and compute. SECOND/ResNet34’s 64/128/256 are a **smaller, faster** design. The stage output channel difference is therefore an intentional capacity choice per backbone family, not a bug or inconsistency.

---

## 9. Latency Reduction Strategies

If you need to reduce the overall network latency of the VoVNet-based CenterPoint pipeline, consider the following directions:

### 9.1 Backbone Architecture Changes

1. **Use a lighter VoVNet variant**: Replace V-99-eSE with a smaller spec (e.g., V-39-eSE or V-57-eSE) by defining a new spec dict with fewer `block_per_stage` and smaller channels. The heaviest component is Stage4 with 9 OSA blocks — reducing this to 3-5 blocks yields the largest speedup.

2. **Reduce stage4 depth**: Stage4 dominates compute with 9 OSA blocks at 255x255 resolution. Reducing `block_per_stage[2]` from 9 to 3-5 blocks can cut backbone latency by ~30-40% with moderate accuracy impact.

3. **Skip stage2 features (high-resolution branch)**: Stage2 outputs at 1020x1020 are the most expensive to process in the neck (branch 0 downsamples from 1020 to 510). Removing stage2 from `out_features` and only using stage3/stage4 (with adjusted `upsample_strides=[1, 2]`) eliminates this bottleneck. The trade-off is losing fine-grained spatial detail.

4. **Enable depthwise separable convolutions**: Set `dw=True` in the VoVNet spec to use depthwise separable conv3x3 in OSA blocks. This reduces parameter count and FLOPs at each layer.

5. **Reduce OSA layers per block**: The `layer_per_block=5` setting means each OSA block has 5 intermediate conv layers. Reducing to 3 decreases both compute and the concat dimension.

### 9.2 Input Resolution / Voxelization

6. **Increase voxel size**: Changing `voxel_size` from `[0.24, 0.24, 8.0]` to `[0.32, 0.32, 8.0]` reduces the BEV grid from 1020x1020 to 765x765. This quadratically reduces compute in the backbone (especially stage2 at full resolution). This is the single most impactful change for latency.

7. **Reduce point cloud range**: Narrowing the detection range (e.g., from 122.4m to 100m) shrinks the grid proportionally.

### 9.3 Deployment Optimizations

8. **INT8 quantization (PTQ/QAT)**: Apply post-training quantization or quantization-aware training for TensorRT deployment. INT8 can provide ~2x speedup over FP16 for conv-heavy backbones. See `deployment/projects/centerpoint/config/` for reference INT8 configs.

9. **TensorRT FP16**: Ensure the model is exported with FP16 precision in TensorRT. The config already uses `AmpOptimWrapper` with `dtype="float16"`, which helps maintain numerical compatibility.

10. **Operator fusion**: TensorRT automatically fuses Conv+BN+ReLU sequences. Ensure BN is fused before export (`fuse_bn=True` in deploy config) for maximum throughput.

### 9.4 Recommended Priority

For the best latency-accuracy trade-off, the recommended approach (from most impactful to least):

| Priority | Strategy | Expected Speedup | Accuracy Impact |
|----------|----------|------------------|-----------------|
| 1 | Increase voxel_size (0.24→0.32) | ~40-50% | Moderate (loss of fine detail) |
| 2 | Reduce stage4 blocks (9→3-5) | ~20-30% | Small to moderate |
| 3 | INT8 quantization | ~2x over FP16 | Small (with proper calibration) |
| 4 | Skip stage2 (drop high-res branch) | ~15-20% | Moderate |
| 5 | Use lighter VoVNet variant | ~30-50% | Depends on variant |

### 9.5 Notes

- When changing voxel_size or point_cloud_range, remember to update `grid_size`, `out_size_factor`, and all dependent configs (neck upsample_strides, head, train_cfg, test_cfg).
- Reducing backbone capacity may require tuning the learning rate schedule, batch size, and training epochs to maintain convergence.
- For deployment, always profile on the target hardware (e.g., NVIDIA Orin) since inference characteristics differ from training GPUs.
